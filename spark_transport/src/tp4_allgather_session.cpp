#include "spark_transport/tp4_allgather_session.hpp"

#include "cuda_event_gate.hpp"
#include "spark_transport/control_channel.hpp"
#include "spark_transport/eager_staging_timeout.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_tp4_allgather.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace spark_transport {
namespace {

constexpr std::uint64_t kDefaultMaxInflight = 64;
constexpr std::uint64_t kMaximumMaxInflight = 4096;

std::uint64_t max_inflight_collectives() {
  const char* value = std::getenv("SPARK_TP4_MAX_INFLIGHT");
  if (value == nullptr || value[0] == '\0') {
    return kDefaultMaxInflight;
  }
  std::size_t consumed{};
  const std::string text(value);
  const auto parsed = std::stoull(text, &consumed);
  if (consumed != text.size() || parsed == 0 ||
      parsed > kMaximumMaxInflight) {
    throw std::invalid_argument(
        "SPARK_TP4_MAX_INFLIGHT must be an integer in [1, 4096]");
  }
  return parsed;
}

std::chrono::seconds eager_protocol_timeout() {
  return parse_eager_staging_timeout(
      std::getenv("SPARK_TP4_EAGER_PROTOCOL_TIMEOUT_SECONDS"),
      "SPARK_TP4_EAGER_PROTOCOL_TIMEOUT_SECONDS");
}

bool protocol_trace_enabled() {
  const char* value = std::getenv("SPARK_TP4_PROTOCOL_TRACE");
  return value != nullptr && std::string_view(value) == "1";
}

std::uint64_t load_sequence(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_sequence(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

void trace_protocol(
    bool enabled, std::uint32_t rank, std::size_t input_bytes,
    std::uint64_t sequence, const char* state,
    const DoorbellControl& round0,
    const DoorbellControl& round1) noexcept {
  if (!enabled || sequence > 2) {
    return;
  }
  std::fprintf(
      stderr,
      "TP4_AG_PROTOCOL rank=%u input_bytes=%zu sequence=%llu state=%s "
      "r0_producer=%llu r0_remote=%llu r0_consumer=%llu r0_ack=%llu "
      "r1_producer=%llu r1_remote=%llu r1_consumer=%llu r1_ack=%llu\n",
      rank, input_bytes, static_cast<unsigned long long>(sequence), state,
      static_cast<unsigned long long>(
          load_sequence(&round0.producer_sequence)),
      static_cast<unsigned long long>(
          load_sequence(&round0.remote_sequence)),
      static_cast<unsigned long long>(
          load_sequence(&round0.consumer_sequence)),
      static_cast<unsigned long long>(
          load_sequence(&round0.acknowledgement_sequence)),
      static_cast<unsigned long long>(
          load_sequence(&round1.producer_sequence)),
      static_cast<unsigned long long>(
          load_sequence(&round1.remote_sequence)),
      static_cast<unsigned long long>(
          load_sequence(&round1.consumer_sequence)),
      static_cast<unsigned long long>(
          load_sequence(&round1.acknowledgement_sequence)));
  std::fflush(stderr);
}

void wait_for_sequence(const std::uint64_t* address, std::uint64_t expected,
                       const char* name,
                       std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (load_sequence(address) < expected) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(std::string("timed out waiting for ") + name);
    }
  }
}

ControlChannel open_channel(const Tp4RoundPlan& plan,
                            const std::string& peer,
                            std::uint16_t port) {
  return plan.server ? ControlChannel::listen_and_accept(port)
                     : ControlChannel::connect(peer, port);
}

void exchange_round(VerbsEndpoint& endpoint, DoorbellControl& control,
                    const Tp2BufferLayout& layout, std::size_t bytes,
                    std::uint64_t sequence,
                    std::chrono::seconds timeout) {
  store_sequence(&control.producer_sequence, sequence);
  endpoint.write(layout.send_offset, layout.receive_offset, bytes, sequence,
                 false);
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, producer_sequence),
      layout.control_offset + offsetof(DoorbellControl, remote_sequence),
      sizeof(sequence), sequence);
  endpoint.wait_for_send(sequence);
  wait_for_sequence(&control.consumer_sequence, sequence,
                    "GPU all-gather consumption", timeout);
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, consumer_sequence),
      layout.control_offset +
          offsetof(DoorbellControl, acknowledgement_sequence),
      sizeof(sequence), sequence, false);
  wait_for_sequence(&control.acknowledgement_sequence, sequence,
                    "peer all-gather consumption", timeout);
}

[[noreturn]] void fatal_async_failure(const char* message) noexcept {
  std::fprintf(stderr, "FATAL asynchronous TP4 all-gather failed: %s\n",
               message);
  std::fflush(stderr);
  std::abort();
}

}  // namespace

class Tp4AllgatherSession::Impl {
 public:
  explicit Impl(Tp4AllgatherOptions options)
      : options_(std::move(options)),
        layout_(make_tp4_allgather_buffer_layout(options_.input_bytes)),
        max_inflight_(max_inflight_collectives()),
        eager_protocol_timeout_(eager_protocol_timeout()),
        protocol_trace_enabled_(protocol_trace_enabled()) {
    if (options_.rank >= 4 || options_.peer0.empty() ||
        options_.peer1.empty() || options_.device0.empty() ||
        options_.device1.empty()) {
      throw std::invalid_argument("invalid TP4 all-gather options");
    }
    eager_input_gates_.emplace(
        max_inflight_, eager_protocol_timeout_,
        "GPU TP4 all-gather eager input readiness");

    const auto plan0 = make_tp4_round_plan(options_.rank, 0);
    const auto plan1 = make_tp4_round_plan(options_.rank, 1);
    channel0_.emplace(
        open_channel(plan0, options_.peer0, options_.control_port0));
    buffer0_ = MemoryBuffer::allocate(MemoryKind::kCudaMapped,
                                     layout_.round0.total_bytes);
    endpoint0_ = std::make_unique<VerbsEndpoint>(
        options_.device0, 1, options_.gid0, *buffer0_);
    endpoint0_->connect(channel0_->exchange(endpoint0_->local_info()));

    channel1_.emplace(
        open_channel(plan1, options_.peer1, options_.control_port1));
    buffer1_ = MemoryBuffer::allocate(MemoryKind::kCudaMapped,
                                     layout_.round1.total_bytes);
    endpoint1_ = std::make_unique<VerbsEndpoint>(
        options_.device1, 1, options_.gid1, *buffer1_);
    endpoint1_->connect(channel1_->exchange(endpoint1_->local_info()));

    control0_ = reinterpret_cast<DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer0_->host_data()) +
        layout_.round0.control_offset);
    control1_ = reinterpret_cast<DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer1_->host_data()) +
        layout_.round1.control_offset);
    worker_ = std::make_unique<GpuTp4AllgatherWorker>(
        options_.rank, layout_, buffer0_->device_data(),
        buffer1_->device_data());
    channel0_->barrier();
    channel1_->barrier();
    start_progress_thread();
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(submission_mutex_);
      stopping_ = true;
    }
    progress_cv_.notify_one();
    completion_cv_.notify_all();
    if (progress_thread_.joinable()) {
      progress_thread_.join();
    }
    if (caller_stream_set_) {
      const auto result =
          cudaStreamSynchronize(static_cast<cudaStream_t>(caller_stream_));
      if (result != cudaSuccess) {
        fatal_async_failure(cudaGetErrorString(result));
      }
    }
    worker_.reset();
  }

  void all_gather(const void* input, void* output, void* cuda_stream) {
    if (input == nullptr || output == nullptr) {
      throw std::invalid_argument("TP4 all-gather tensor pointer is null");
    }
    {
      std::unique_lock<std::mutex> lock(submission_mutex_);
      completion_cv_.wait(lock, [this] {
        return stopping_ || poisoned_ ||
               sequence_ - completed_sequence_ < max_inflight_;
      });
      if (stopping_) {
        throw std::runtime_error("TP4 all-gather session is stopping");
      }
      if (poisoned_) {
        throw std::runtime_error("TP4 all-gather session is poisoned");
      }
      if (sequence_ == std::numeric_limits<std::uint64_t>::max() - 1) {
        throw std::overflow_error("TP4 all-gather sequence exhausted");
      }
      if (caller_stream_set_ && caller_stream_ != cuda_stream) {
        throw std::invalid_argument(
            "TP4 all-gather requires one stable caller CUDA stream");
      }
      const std::uint64_t next_sequence = sequence_ + 1;
      submissions_.push_back(next_sequence);
      try {
        trace_protocol(
            protocol_trace_enabled_, options_.rank, layout_.input_bytes,
            next_sequence, "submit-before-event", *control0_, *control1_);
        if (!eager_input_gates_.has_value()) {
          throw std::logic_error(
              "TP4 all-gather eager input gate is unavailable");
        }
        eager_input_gates_->record(
            next_sequence, static_cast<cudaStream_t>(cuda_stream));
        worker_->enqueue(input, output, cuda_stream, next_sequence);
        trace_protocol(
            protocol_trace_enabled_, options_.rank, layout_.input_bytes,
            next_sequence, "submit-after-enqueue", *control0_, *control1_);
      } catch (...) {
        submissions_.pop_back();
        poisoned_ = true;
        completion_cv_.notify_all();
        throw;
      }
      sequence_ = next_sequence;
      caller_stream_ = cuda_stream;
      caller_stream_set_ = true;
    }
    progress_cv_.notify_one();
  }

 private:
  void start_progress_thread() {
    std::promise<std::string> startup_promise;
    auto startup = startup_promise.get_future();
    progress_thread_ = std::thread(
        [this, promise = std::move(startup_promise)]() mutable {
          try {
            if (!eager_input_gates_.has_value()) {
              throw std::logic_error(
                  "TP4 all-gather eager input gate is unavailable");
            }
            eager_input_gates_->bind_current_thread();
            promise.set_value({});
          } catch (const std::exception& error) {
            promise.set_value(error.what());
            return;
          } catch (...) {
            promise.set_value(
                "unknown TP4 all-gather progress startup failure");
            return;
          }
          progress_loop();
        });
    const std::string startup_error = startup.get();
    if (!startup_error.empty()) {
      progress_thread_.join();
      throw std::runtime_error(startup_error);
    }
  }

  void progress_loop() noexcept {
    while (true) {
      std::uint64_t sequence{};
      {
        std::unique_lock<std::mutex> lock(submission_mutex_);
        progress_cv_.wait(
            lock, [this] { return stopping_ || !submissions_.empty(); });
        if (submissions_.empty()) {
          if (stopping_) {
            return;
          }
          continue;
        }
        sequence = submissions_.front();
        submissions_.pop_front();
      }
      try {
        trace_protocol(
            protocol_trace_enabled_, options_.rank, layout_.input_bytes,
            sequence, "progress-dequeue", *control0_, *control1_);
        if (!eager_input_gates_.has_value()) {
          throw std::logic_error(
              "TP4 all-gather eager input gate is unavailable");
        }
        eager_input_gates_->wait(sequence);
        trace_protocol(
            protocol_trace_enabled_, options_.rank, layout_.input_bytes,
            sequence, "event-ready", *control0_, *control1_);
        wait_for_sequence(&control0_->producer_sequence, sequence,
                          "GPU all-gather input staging",
                          eager_protocol_timeout_);
        trace_protocol(
            protocol_trace_enabled_, options_.rank, layout_.input_bytes,
            sequence, "producer-ready", *control0_, *control1_);
        exchange_round(*endpoint0_, *control0_, layout_.round0,
                       layout_.input_bytes, sequence,
                       eager_protocol_timeout_);
        trace_protocol(
            protocol_trace_enabled_, options_.rank, layout_.input_bytes,
            sequence, "round0-complete", *control0_, *control1_);
        exchange_round(*endpoint1_, *control1_, layout_.round1,
                       layout_.input_bytes * 2, sequence,
                       eager_protocol_timeout_);
        trace_protocol(
            protocol_trace_enabled_, options_.rank, layout_.input_bytes,
            sequence, "round1-complete", *control0_, *control1_);
      } catch (const std::exception& error) {
        fatal_async_failure(error.what());
      } catch (...) {
        fatal_async_failure("unknown error");
      }
      {
        std::lock_guard<std::mutex> lock(submission_mutex_);
        completed_sequence_ = sequence;
      }
      completion_cv_.notify_all();
    }
  }

  Tp4AllgatherOptions options_;
  Tp4AllgatherBufferLayout layout_{};
  std::optional<ControlChannel> channel0_;
  std::optional<ControlChannel> channel1_;
  std::unique_ptr<MemoryBuffer> buffer0_;
  std::unique_ptr<MemoryBuffer> buffer1_;
  std::unique_ptr<VerbsEndpoint> endpoint0_;
  std::unique_ptr<VerbsEndpoint> endpoint1_;
  DoorbellControl* control0_{};
  DoorbellControl* control1_{};
  std::unique_ptr<GpuTp4AllgatherWorker> worker_;
  std::optional<CudaEventGatePool> eager_input_gates_;
  std::mutex submission_mutex_;
  std::condition_variable progress_cv_;
  std::condition_variable completion_cv_;
  std::deque<std::uint64_t> submissions_;
  std::thread progress_thread_;
  const std::uint64_t max_inflight_;
  const std::chrono::seconds eager_protocol_timeout_;
  const bool protocol_trace_enabled_;
  std::uint64_t sequence_{};
  std::uint64_t completed_sequence_{};
  void* caller_stream_{};
  bool caller_stream_set_{};
  bool stopping_{};
  bool poisoned_{};
};

Tp4AllgatherSession::Tp4AllgatherSession(Tp4AllgatherOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Tp4AllgatherSession::~Tp4AllgatherSession() = default;

void Tp4AllgatherSession::all_gather(const void* input, void* output,
                                     void* cuda_stream) {
  impl_->all_gather(input, output, cuda_stream);
}

}  // namespace spark_transport
