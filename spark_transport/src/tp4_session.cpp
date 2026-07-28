#include "spark_transport/tp4_session.hpp"

#include "cuda_event_gate.hpp"
#include "cuda_stream_handoff.hpp"
#include "spark_transport/control_channel.hpp"
#include "spark_transport/eager_staging_timeout.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/gpu_tp4_tensor.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/tp4_graph_command.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace spark_transport {
namespace {

constexpr std::uint64_t kDefaultMaxInflight = 64;
constexpr std::uint64_t kMaximumMaxInflight = 4096;
constexpr auto kGraphProtocolTimeout = std::chrono::seconds(5);
constexpr std::size_t kQ1PayloadBytes =
    tp4_graph_payload_bytes(1);

bool graph_capacity_supported(std::size_t payload_bytes) {
  for (std::uint32_t q = 1;
       q <= kTp4GraphAllreduceMaximumQ; ++q) {
    if (payload_bytes == tp4_graph_payload_bytes(q)) {
      return true;
    }
  }
  return false;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

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

std::uint64_t load_sequence(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_sequence(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

void wait_for_sequence(const std::uint64_t* address, std::uint64_t expected,
                       const char* name, bool require_exact = false,
                       std::chrono::seconds timeout =
                           kGraphProtocolTimeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const std::uint64_t observed = load_sequence(address);
    if (observed == expected ||
        (!require_exact && observed > expected)) {
      return;
    }
    if (require_exact && observed > expected) {
      throw std::runtime_error(
          std::string("mismatched ") + name + " token");
    }
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
                    std::uint64_t doorbell_token, std::uint32_t rank,
                    std::uint32_t round, bool trace,
                    bool require_exact_doorbell,
                    std::chrono::seconds timeout) {
  store_sequence(&control.producer_sequence, doorbell_token);
  endpoint.write(layout.send_offset, layout.receive_offset, bytes,
                 doorbell_token, false);
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, producer_sequence),
      layout.control_offset + offsetof(DoorbellControl, remote_sequence),
      sizeof(doorbell_token), doorbell_token);
  if (trace) {
    std::fprintf(stderr,
                 "EXCHANGE rank=%u round=%u state=doorbell_posted\n",
                 rank, round);
  }
  endpoint.wait_for_send(doorbell_token);
  if (trace) {
    std::fprintf(stderr,
                 "EXCHANGE rank=%u round=%u state=send_complete\n",
                 rank, round);
  }

  wait_for_sequence(&control.consumer_sequence, doorbell_token,
                    "GPU tensor reduction", require_exact_doorbell,
                    timeout);
  if (trace) {
    std::fprintf(stderr,
                 "EXCHANGE rank=%u round=%u state=gpu_complete\n",
                 rank, round);
  }
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, consumer_sequence),
      layout.control_offset +
          offsetof(DoorbellControl, acknowledgement_sequence),
      sizeof(doorbell_token), doorbell_token, false);
  wait_for_sequence(&control.acknowledgement_sequence, doorbell_token,
                    "peer tensor consumption", require_exact_doorbell,
                    timeout);
  if (trace) {
    std::fprintf(stderr,
                 "EXCHANGE rank=%u round=%u state=peer_consumed\n",
                 rank, round);
  }
}

[[noreturn]] void fatal_async_failure(const char* message) noexcept {
  std::fprintf(stderr, "FATAL asynchronous TP4 progress failed: %s\n",
               message);
  std::fflush(stderr);
  std::abort();
}

void adaptive_graph_poll_pause(std::uint32_t& misses) noexcept {
  ++misses;
#if defined(__aarch64__)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
  if (misses >= 4096) {
    misses = 1024;
    std::this_thread::yield();
  }
}

void require_exclusive_current_cpu(std::uint32_t expected_cpu) {
#if defined(__linux__)
  if (expected_cpu >= CPU_SETSIZE) {
    throw std::invalid_argument("graph submit CPU exceeds CPU_SETSIZE");
  }
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  const int result =
      pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity);
  if (result != 0) {
    throw std::runtime_error(
        std::string("pthread_getaffinity_np graph submit thread: ") +
        std::strerror(result));
  }
  std::size_t enabled{};
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    enabled += CPU_ISSET(cpu, &affinity) ? 1U : 0U;
  }
  if (enabled != 1 || !CPU_ISSET(expected_cpu, &affinity)) {
    throw std::runtime_error(
        "graph submit thread must be pinned exclusively to configured CPU " +
        std::to_string(expected_cpu));
  }
#else
  (void)expected_cpu;
  throw std::runtime_error(
      "graph CPU affinity requires Linux pthread affinity support");
#endif
}

void pin_current_thread_to_cpu(std::uint32_t cpu, const char* role) {
#if defined(__linux__)
  if (cpu >= CPU_SETSIZE) {
    throw std::invalid_argument(std::string(role) +
                                " CPU exceeds CPU_SETSIZE");
  }
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(cpu, &affinity);
  const int result =
      pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
  if (result != 0) {
    throw std::runtime_error(
        std::string("pthread_setaffinity_np ") + role + ": " +
        std::strerror(result));
  }
  require_exclusive_current_cpu(cpu);
#else
  (void)cpu;
  (void)role;
  throw std::runtime_error(
      "graph CPU affinity requires Linux pthread affinity support");
#endif
}

}  // namespace

class Tp4AllreduceSession::Impl {
 public:
  explicit Impl(Tp4AllreduceOptions options)
      : options_(std::move(options)),
        max_inflight_(max_inflight_collectives()) {
    if (options_.rank >= 4 || options_.peer0.empty() ||
        options_.peer1.empty() || options_.device0.empty() ||
        options_.device1.empty() || options_.payload_bytes == 0 ||
        options_.payload_bytes % 2 != 0) {
      throw std::invalid_argument("invalid TP4 all-reduce options");
    }
    const bool submit_cpu_set = options_.graph_submit_cpu.has_value();
    const bool progress_cpu_set = options_.graph_progress_cpu.has_value();
    if (submit_cpu_set != progress_cpu_set) {
      throw std::invalid_argument(
          "graph submit/progress CPUs must be configured together");
    }
    if (submit_cpu_set) {
      if (options_.graph_submit_cpu == options_.graph_progress_cpu) {
        throw std::invalid_argument(
            "graph submit/progress CPUs must be distinct");
      }
      pin_current_thread_to_cpu(*options_.graph_submit_cpu,
                                "graph submission thread");
      submit_affinity_verified_ = true;
    } else {
      eager_protocol_timeout_ = eager_protocol_timeout();
      eager_input_gates_.emplace(
          max_inflight_, eager_protocol_timeout_,
          "GPU TP4 eager input readiness");
    }

    const auto plan0 = make_tp4_round_plan(options_.rank, 0);
    const auto plan1 = make_tp4_round_plan(options_.rank, 1);
    layout_ = make_tp2_buffer_layout(options_.payload_bytes);

    channel0_.emplace(
        open_channel(plan0, options_.peer0, options_.control_port0));
    buffer0_ =
        MemoryBuffer::allocate(MemoryKind::kCudaMapped, layout_.total_bytes);
    endpoint0_ = std::make_unique<VerbsEndpoint>(
        options_.device0, 1, options_.gid0, *buffer0_);
    endpoint0_->connect(channel0_->exchange(endpoint0_->local_info()));

    channel1_.emplace(
        open_channel(plan1, options_.peer1, options_.control_port1));
    buffer1_ =
        MemoryBuffer::allocate(MemoryKind::kCudaMapped, layout_.total_bytes);
    endpoint1_ = std::make_unique<VerbsEndpoint>(
        options_.device1, 1, options_.gid1, *buffer1_);
    endpoint1_->connect(channel1_->exchange(endpoint1_->local_info()));

    control0_ = reinterpret_cast<DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer0_->host_data()) +
        layout_.control_offset);
    control1_ = reinterpret_cast<DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer1_->host_data()) +
        layout_.control_offset);

    if (graph_capacity_supported(options_.payload_bytes)) {
      int device{};
      int host_native_atomics{};
      check_cuda(cudaGetDevice(&device), "cudaGetDevice graph TP4");
      check_cuda(
          cudaDeviceGetAttribute(
              &host_native_atomics,
              cudaDevAttrHostNativeAtomicSupported, device),
          "cudaDeviceGetAttribute host native atomics");
      graph_host_native_atomics_supported_ = host_native_atomics != 0;
      graph_commands_buffer_ = MemoryBuffer::allocate(
          MemoryKind::kCudaMapped, sizeof(Tp4GraphCommandRing));
      graph_commands_host_ = static_cast<Tp4GraphCommandRing*>(
          graph_commands_buffer_->host_data());
      graph_commands_device_ = static_cast<Tp4GraphCommandRing*>(
          graph_commands_buffer_->device_data());
    }

    worker_ = std::make_unique<GpuTp4TensorWorker>(
        options_.payload_bytes, buffer0_->device_data(), layout_,
        buffer1_->device_data(), layout_);
    channel0_->barrier();
    channel1_->barrier();
    start_progress_thread();
  }

  ~Impl() {
    // Graph kernels publish work only when replay reaches them. Keep the
    // verbs progress thread alive while draining the caller stream so a
    // queued replay cannot strand its GPU kernel waiting for RDMA.
    if (caller_stream_set_) {
      const auto result =
          cudaStreamSynchronize(static_cast<cudaStream_t>(caller_stream_));
      if (result != cudaSuccess) {
        fatal_async_failure(cudaGetErrorString(result));
      }
    }
    {
      std::lock_guard<std::mutex> lock(submission_mutex_);
      stopping_ = true;
      stop_requested_.store(true, std::memory_order_release);
    }
    progress_cv_.notify_one();
    completion_cv_.notify_all();
    if (progress_thread_.joinable()) {
      progress_thread_.join();
    }
    worker_.reset();
  }

  void all_reduce(const void* input, void* output, void* cuda_stream) {
    if (input == nullptr || output == nullptr) {
      throw std::invalid_argument("TP4 all-reduce tensor pointer is null");
    }
    if (options_.graph_submit_cpu.has_value()) {
      throw std::logic_error(
          "graph-configured TP4 session rejects eager submission");
    }
    const bool trace = std::getenv("SPARK_TRANSPORT_TRACE") != nullptr;
    {
      std::unique_lock<std::mutex> lock(submission_mutex_);
      if (graph_capture_configured_) {
        throw std::logic_error(
            "eager TP4 submission cannot follow graph capture setup");
      }
      if (poisoned_) {
        throw std::runtime_error("TP4 session is poisoned");
      }
      completion_cv_.wait(lock, [this] {
        return stopping_ || poisoned_ ||
               sequence_ - completed_sequence_ < max_inflight_;
      });
      if (stopping_) {
        throw std::runtime_error("TP4 session is stopping");
      }
      if (poisoned_) {
        throw std::runtime_error("TP4 session is poisoned");
      }
      if (sequence_ == std::numeric_limits<std::uint64_t>::max() - 1) {
        throw std::overflow_error("TP4 all-reduce sequence exhausted");
      }
      if (caller_stream_set_ && caller_stream_ != cuda_stream) {
        try {
          stream_handoff_.order(
              static_cast<cudaStream_t>(caller_stream_),
              static_cast<cudaStream_t>(cuda_stream));
          // The new stream now owns a queued wait. Retain it even if the
          // following kernel launch is rejected so destruction drains it.
          caller_stream_ = cuda_stream;
          caller_stream_set_ = true;
        } catch (...) {
          poisoned_ = true;
          completion_cv_.notify_all();
          throw;
        }
      }

      const std::uint64_t next_sequence = sequence_ + 1;
      submissions_.push_back(Submission{next_sequence, trace});
      try {
        if (!eager_input_gates_.has_value()) {
          throw std::logic_error("TP4 eager input gate is unavailable");
        }
        eager_input_gates_->record(
            next_sequence, static_cast<cudaStream_t>(cuda_stream));
        worker_->enqueue(input, output, cuda_stream, next_sequence);
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

  void capture_all_reduce(const void* input, void* output, std::uint32_t q,
                          void* cuda_stream) {
    if (input == nullptr || output == nullptr) {
      throw std::invalid_argument(
          "graph TP4 all-reduce tensor pointer is null");
    }
    const std::uint32_t active_payload_bytes =
        tp4_graph_payload_bytes(q);
    if (!tp4_graph_allreduce_command_descriptor_valid(
            q, active_payload_bytes) ||
        active_payload_bytes > options_.payload_bytes ||
        graph_commands_host_ == nullptr) {
      throw std::invalid_argument(
          "graph TP4 all-reduce requires BF16 [q, 6144], q in [1, 512], "
          "within session capacity");
    }
    if (!graph_host_native_atomics_supported_) {
      throw std::runtime_error(
          "graph TP4 requires host-native GPU atomics");
    }

    const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
    cudaStreamCaptureStatus capture_status{};
    check_cuda(cudaStreamIsCapturing(caller_stream, &capture_status),
               "cudaStreamIsCapturing graph TP4");
    if (capture_status != cudaStreamCaptureStatusActive) {
      throw std::logic_error(
          "graph TP4 setup requires an active CUDA stream capture");
    }

    {
      std::lock_guard<std::mutex> lock(submission_mutex_);
      if (stopping_) {
        throw std::runtime_error("TP4 session is stopping");
      }
      if (sequence_ != 0 || completed_sequence_ != 0 ||
          !submissions_.empty()) {
        throw std::logic_error(
            "graph TP4 capture must precede eager submissions");
      }
      if (caller_stream_set_ && caller_stream_ != cuda_stream) {
        throw std::invalid_argument(
            "TP4 session requires one stable caller CUDA stream");
      }
      if (tp4_graph_command_published(graph_commands_host_) != 0 ||
          tp4_graph_command_consumed(graph_commands_host_) != 0 ||
          tp4_graph_command_completed(graph_commands_host_) != 0 ||
          tp4_graph_command_overflow(graph_commands_host_) != 0) {
        throw std::logic_error(
            "graph TP4 capture cannot add nodes after the first replay");
      }
      if (!graph_capture_configured_) {
        graph_trace_ = std::getenv("SPARK_TRANSPORT_TRACE") != nullptr;
        graph_capture_configured_ = true;
        caller_stream_ = cuda_stream;
        caller_stream_set_ = true;
      }
    }

    try {
      worker_->enqueue_graph(input, output, q, cuda_stream,
                             graph_commands_device_, graph_trace_);
    } catch (...) {
      std::lock_guard<std::mutex> lock(submission_mutex_);
      if (graph_capture_nodes_ == 0) {
        graph_capture_configured_ = false;
        caller_stream_ = nullptr;
        caller_stream_set_ = false;
      }
      throw;
    }
    graph_capture_nodes_.fetch_add(1, std::memory_order_release);
    graph_polling_enabled_.store(true, std::memory_order_release);
    progress_cv_.notify_one();
  }

  Tp4GraphReplayStatus graph_replay_status() const noexcept {
    return Tp4GraphReplayStatus{
        graph_capture_nodes_.load(std::memory_order_acquire),
        tp4_graph_command_published(graph_commands_host_),
        tp4_graph_command_consumed(graph_commands_host_),
        tp4_graph_command_completed(graph_commands_host_),
        tp4_graph_command_overflow(graph_commands_host_),
        graph_capture_configured_.load(std::memory_order_acquire),
        graph_polling_enabled_.load(std::memory_order_acquire),
        graph_host_native_atomics_supported_,
        submit_affinity_verified_,
        progress_affinity_verified_.load(std::memory_order_acquire),
        options_.graph_submit_cpu.has_value()
            ? static_cast<int>(*options_.graph_submit_cpu)
            : -1,
        options_.graph_progress_cpu.has_value()
            ? static_cast<int>(*options_.graph_progress_cpu)
            : -1};
  }

 private:
  struct Submission {
    std::uint64_t sequence;
    bool trace;
  };

  void start_progress_thread() {
    std::promise<std::string> startup_promise;
    auto startup = startup_promise.get_future();
    progress_thread_ = std::thread(
        [this, promise = std::move(startup_promise)]() mutable {
          try {
            if (eager_input_gates_.has_value()) {
              eager_input_gates_->bind_current_thread();
            }
            if (options_.graph_progress_cpu.has_value()) {
              pin_current_thread_to_cpu(*options_.graph_progress_cpu,
                                        "graph progress thread");
              progress_affinity_verified_.store(true,
                                                std::memory_order_release);
            }
            promise.set_value({});
          } catch (const std::exception& error) {
            promise.set_value(error.what());
            return;
          } catch (...) {
            promise.set_value("unknown graph progress affinity failure");
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
      if (graph_polling_enabled_.load(std::memory_order_acquire)) {
        progress_graph_commands();
        return;
      }

      Submission submission{};
      {
        std::unique_lock<std::mutex> lock(submission_mutex_);
        progress_cv_.wait(lock, [this] {
          return stopping_ || !submissions_.empty() ||
                 graph_polling_enabled_.load(std::memory_order_acquire);
        });
        if (graph_polling_enabled_.load(std::memory_order_acquire)) {
          continue;
        }
        if (stopping_) {
          return;
        }
        submission = submissions_.front();
        submissions_.pop_front();
      }
      try {
        if (!eager_input_gates_.has_value()) {
          throw std::logic_error("TP4 eager input gate is unavailable");
        }
        eager_input_gates_->wait(submission.sequence);
        progress(submission.sequence, submission.trace);
      } catch (const std::exception& error) {
        fatal_async_failure(error.what());
      } catch (...) {
        fatal_async_failure("unknown error");
      }
      {
        std::lock_guard<std::mutex> lock(submission_mutex_);
        completed_sequence_ = submission.sequence;
      }
      completion_cv_.notify_all();
    }
  }

  void progress_graph_commands() noexcept {
    std::uint32_t poll_misses{};
    while (!stop_requested_.load(std::memory_order_acquire)) {
      if (tp4_graph_command_overflow(graph_commands_host_) != 0) {
        fatal_async_failure("CUDA Graph command ring overflow");
      }

      Tp4GraphCommand command{};
      const std::uint64_t expected = graph_consumed_sequence_ + 1;
      if (!tp4_graph_allreduce_command_try_consume_descriptor(
              graph_commands_host_, expected, &command)) {
        adaptive_graph_poll_pause(poll_misses);
        continue;
      }
      poll_misses = 0;
      graph_consumed_sequence_ = command.sequence;
      try {
        if (!tp4_graph_doorbell_token_valid(
                command.sequence, command.q)) {
          fatal_async_failure("invalid graph TP4 doorbell token");
        }
        progress(command.sequence, command.trace != 0,
                 command.payload_bytes,
                 tp4_graph_doorbell_token(
                     command.sequence, command.q),
                 true);
      } catch (const std::exception& error) {
        fatal_async_failure(error.what());
      } catch (...) {
        fatal_async_failure("unknown error");
      }
      tp4_graph_command_complete(graph_commands_host_, command.sequence);
    }
  }

  void progress(std::uint64_t sequence, bool trace) {
    progress(sequence, trace, options_.payload_bytes, sequence, false,
             eager_protocol_timeout_);
  }

  void progress(std::uint64_t sequence, bool trace,
                std::size_t payload_bytes,
                std::uint64_t doorbell_token,
                bool require_exact_doorbell,
                std::chrono::seconds timeout =
                    kGraphProtocolTimeout) {
    wait_for_sequence(&control0_->producer_sequence, doorbell_token,
                      "GPU tensor input staging",
                      require_exact_doorbell, timeout);
    if (trace) {
      std::fprintf(
          stderr,
          "SESSION rank=%u state=round0 sequence=%llu token=%llu bytes=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), payload_bytes);
    }
    exchange_round(*endpoint0_, *control0_, layout_, payload_bytes,
                   doorbell_token, options_.rank, 0, trace,
                   require_exact_doorbell, timeout);
    if (trace) {
      std::fprintf(
          stderr,
          "SESSION rank=%u state=round1 sequence=%llu token=%llu bytes=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), payload_bytes);
    }
    exchange_round(*endpoint1_, *control1_, layout_, payload_bytes,
                   doorbell_token, options_.rank, 1, trace,
                   require_exact_doorbell, timeout);
    if (trace) {
      std::fprintf(stderr, "SESSION rank=%u state=output_ready\n",
                   options_.rank);
    }
  }

  Tp4AllreduceOptions options_;
  Tp2BufferLayout layout_{};
  std::optional<ControlChannel> channel0_;
  std::optional<ControlChannel> channel1_;
  std::unique_ptr<MemoryBuffer> buffer0_;
  std::unique_ptr<MemoryBuffer> buffer1_;
  std::unique_ptr<VerbsEndpoint> endpoint0_;
  std::unique_ptr<VerbsEndpoint> endpoint1_;
  DoorbellControl* control0_{};
  DoorbellControl* control1_{};
  std::unique_ptr<MemoryBuffer> graph_commands_buffer_;
  Tp4GraphCommandRing* graph_commands_host_{};
  Tp4GraphCommandRing* graph_commands_device_{};
  std::unique_ptr<GpuTp4TensorWorker> worker_;
  std::optional<CudaEventGatePool> eager_input_gates_;
  CudaStreamHandoff stream_handoff_;
  std::mutex submission_mutex_;
  std::condition_variable progress_cv_;
  std::condition_variable completion_cv_;
  std::deque<Submission> submissions_;
  std::thread progress_thread_;
  const std::uint64_t max_inflight_;
  std::chrono::seconds eager_protocol_timeout_{
      kGraphProtocolTimeout};
  std::atomic<bool> graph_polling_enabled_{false};
  std::atomic<bool> stop_requested_{false};
  std::uint64_t sequence_{};
  std::uint64_t completed_sequence_{};
  std::uint64_t graph_consumed_sequence_{};
  void* caller_stream_{};
  bool caller_stream_set_{};
  std::atomic<bool> graph_capture_configured_{false};
  std::atomic<std::uint64_t> graph_capture_nodes_{0};
  bool graph_host_native_atomics_supported_{};
  bool submit_affinity_verified_{};
  std::atomic<bool> progress_affinity_verified_{false};
  bool graph_trace_{};
  bool stopping_{};
  bool poisoned_{};
};

Tp4AllreduceSession::Tp4AllreduceSession(Tp4AllreduceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Tp4AllreduceSession::~Tp4AllreduceSession() = default;

void Tp4AllreduceSession::all_reduce(const void* input, void* output,
                                     void* cuda_stream) {
  impl_->all_reduce(input, output, cuda_stream);
}

void Tp4AllreduceSession::capture_q1_all_reduce(
    const void* input, void* output, void* cuda_stream) {
  impl_->capture_all_reduce(input, output, 1, cuda_stream);
}

void Tp4AllreduceSession::capture_all_reduce(
    const void* input, void* output, std::uint32_t q, void* cuda_stream) {
  impl_->capture_all_reduce(input, output, q, cuda_stream);
}

Tp4GraphReplayStatus Tp4AllreduceSession::graph_replay_status()
    const noexcept {
  return impl_->graph_replay_status();
}

}  // namespace spark_transport
