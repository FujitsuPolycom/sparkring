#include "spark_transport/tp4_indexer_graph_session.hpp"

#include "spark_transport/control_channel.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_tp4_allgather.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/tp4_graph_command.hpp"
#include "spark_transport/tp4_indexer_graph.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace spark_transport {
namespace {

constexpr auto kGraphProtocolTimeout = std::chrono::seconds(5);

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

std::uint64_t load_sequence(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_sequence(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

void wait_for_exact_sequence(
    const std::uint64_t* address, std::uint64_t expected,
    const char* name) {
  const auto deadline =
      std::chrono::steady_clock::now() + kGraphProtocolTimeout;
  while (true) {
    const std::uint64_t observed = load_sequence(address);
    if (observed == expected) {
      return;
    }
    if (observed > expected) {
      throw std::runtime_error(
          std::string("mismatched ") + name + " token");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          std::string("timed out waiting for ") + name);
    }
  }
}

ControlChannel open_channel(
    const Tp4RoundPlan& plan, const std::string& peer,
    std::uint16_t port) {
  return plan.server ? ControlChannel::listen_and_accept(port)
                     : ControlChannel::connect(peer, port);
}

void exchange_round(
    VerbsEndpoint& endpoint, DoorbellControl& control,
    const Tp2BufferLayout& layout, std::size_t bytes,
    std::uint64_t doorbell_token, const char* family) {
  store_sequence(&control.producer_sequence, doorbell_token);
  endpoint.write(
      layout.send_offset, layout.receive_offset, bytes,
      doorbell_token, false);
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, producer_sequence),
      layout.control_offset + offsetof(DoorbellControl, remote_sequence),
      sizeof(doorbell_token), doorbell_token);
  endpoint.wait_for_send(doorbell_token);
  wait_for_exact_sequence(
      &control.consumer_sequence, doorbell_token, family);
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, consumer_sequence),
      layout.control_offset +
          offsetof(DoorbellControl, acknowledgement_sequence),
      sizeof(doorbell_token), doorbell_token, false);
  wait_for_exact_sequence(
      &control.acknowledgement_sequence, doorbell_token, family);
}

[[noreturn]] void fatal_async_failure(const char* message) noexcept {
  std::fprintf(
      stderr, "FATAL asynchronous TP4 indexer graph failed: %s\n",
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
    throw std::invalid_argument(
        "indexer graph submit CPU exceeds CPU_SETSIZE");
  }
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  const int result =
      pthread_getaffinity_np(
          pthread_self(), sizeof(affinity), &affinity);
  if (result != 0) {
    throw std::runtime_error(
        std::string("pthread_getaffinity_np indexer graph thread: ") +
        std::strerror(result));
  }
  std::size_t enabled{};
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    enabled += CPU_ISSET(cpu, &affinity) ? 1U : 0U;
  }
  if (enabled != 1 || !CPU_ISSET(expected_cpu, &affinity)) {
    throw std::runtime_error(
        "indexer graph thread must be pinned exclusively to CPU " +
        std::to_string(expected_cpu));
  }
#else
  (void)expected_cpu;
  throw std::runtime_error("indexer graph CPU affinity requires Linux");
#endif
}

void pin_current_thread_to_cpu(std::uint32_t cpu, const char* role) {
#if defined(__linux__)
  if (cpu >= CPU_SETSIZE) {
    throw std::invalid_argument(
        std::string(role) + " CPU exceeds CPU_SETSIZE");
  }
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(cpu, &affinity);
  const int result =
      pthread_setaffinity_np(
          pthread_self(), sizeof(affinity), &affinity);
  if (result != 0) {
    throw std::runtime_error(
        std::string("pthread_setaffinity_np ") + role + ": " +
        std::strerror(result));
  }
  require_exclusive_current_cpu(cpu);
#else
  (void)cpu;
  (void)role;
  throw std::runtime_error("indexer graph CPU affinity requires Linux");
#endif
}

}  // namespace

class Tp4IndexerGraphSession::Impl {
 public:
  explicit Impl(Tp4IndexerGraphOptions options)
      : options_(std::move(options)),
        layout_(make_tp4_allgather_buffer_layout(
            kTp4IndexerGraphMaximumInputBytes)) {
    if (options_.rank >= kTp4IndexerGraphWorldSize ||
        options_.peer0.empty() || options_.peer1.empty() ||
        options_.device0.empty() || options_.device1.empty() ||
        options_.control_port0 == 0 || options_.control_port1 == 0 ||
        options_.control_port0 == options_.control_port1 ||
        !options_.graph_submit_cpu.has_value() ||
        !options_.graph_progress_cpu.has_value() ||
        options_.graph_submit_cpu == options_.graph_progress_cpu) {
      throw std::invalid_argument(
          "invalid graph-only TP4 indexer options");
    }
    pin_current_thread_to_cpu(
        *options_.graph_submit_cpu,
        "indexer graph submission thread");
    submit_affinity_verified_ = true;

    const auto plan0 = make_tp4_round_plan(options_.rank, 0);
    const auto plan1 = make_tp4_round_plan(options_.rank, 1);
    channel0_.emplace(
        open_channel(plan0, options_.peer0, options_.control_port0));
    buffer0_ = MemoryBuffer::allocate(
        MemoryKind::kCudaMapped, layout_.round0.total_bytes);
    endpoint0_ = std::make_unique<VerbsEndpoint>(
        options_.device0, 1, options_.gid0, *buffer0_);
    endpoint0_->connect(
        channel0_->exchange(endpoint0_->local_info()));

    channel1_.emplace(
        open_channel(plan1, options_.peer1, options_.control_port1));
    buffer1_ = MemoryBuffer::allocate(
        MemoryKind::kCudaMapped, layout_.round1.total_bytes);
    endpoint1_ = std::make_unique<VerbsEndpoint>(
        options_.device1, 1, options_.gid1, *buffer1_);
    endpoint1_->connect(
        channel1_->exchange(endpoint1_->local_info()));

    control0_ = reinterpret_cast<DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer0_->host_data()) +
        layout_.round0.control_offset);
    control1_ = reinterpret_cast<DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer1_->host_data()) +
        layout_.round1.control_offset);

    int device{};
    int host_native_atomics{};
    check_cuda(
        cudaGetDevice(&device), "cudaGetDevice indexer graph");
    check_cuda(
        cudaDeviceGetAttribute(
            &host_native_atomics,
            cudaDevAttrHostNativeAtomicSupported, device),
        "cudaDeviceGetAttribute indexer host native atomics");
    graph_host_native_atomics_supported_ = host_native_atomics != 0;
    graph_commands_buffer_ = MemoryBuffer::allocate(
        MemoryKind::kCudaMapped, sizeof(Tp4GraphCommandRing));
    graph_commands_host_ = static_cast<Tp4GraphCommandRing*>(
        graph_commands_buffer_->host_data());
    graph_commands_device_ = static_cast<Tp4GraphCommandRing*>(
        graph_commands_buffer_->device_data());

    worker_ = std::make_unique<GpuTp4AllgatherWorker>(
        options_.rank, layout_, buffer0_->device_data(),
        buffer1_->device_data());
    channel0_->barrier();
    channel1_->barrier();
    start_progress_thread();
  }

  ~Impl() {
    // Keep progress alive while every replay that can block in a graph kernel
    // drains on every stream used during capture.
    for (void* graph_stream : graph_capture_streams_) {
      const auto result = cudaStreamSynchronize(
          static_cast<cudaStream_t>(graph_stream));
      if (result != cudaSuccess) {
        fatal_async_failure(cudaGetErrorString(result));
      }
    }
    stop_requested_.store(true, std::memory_order_release);
    progress_cv_.notify_one();
    if (progress_thread_.joinable()) {
      progress_thread_.join();
    }
    worker_.reset();
  }

  void capture_all_gather(
      const void* input, void* output, std::uint32_t q,
      void* cuda_stream) {
    Tp4IndexerGraphDescriptor descriptor{};
    if (input == nullptr || output == nullptr || cuda_stream == nullptr ||
        !tp4_indexer_graph_descriptor_from_q(q, &descriptor)) {
      throw std::invalid_argument(
          "graph TP4 indexer requires contiguous INT32 "
          "[Q,2,2048], Q in [1,40]");
    }
    if (!graph_host_native_atomics_supported_) {
      throw std::runtime_error(
          "graph TP4 indexer requires host-native GPU atomics");
    }

    const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
    cudaStreamCaptureStatus capture_status{};
    check_cuda(
        cudaStreamIsCapturing(caller_stream, &capture_status),
        "cudaStreamIsCapturing indexer graph");
    if (capture_status != cudaStreamCaptureStatusActive) {
      throw std::logic_error(
          "indexer graph setup requires active CUDA stream capture");
    }

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (tp4_graph_command_published(graph_commands_host_) != 0 ||
          tp4_graph_command_consumed(graph_commands_host_) != 0 ||
          tp4_graph_command_completed(graph_commands_host_) != 0 ||
          tp4_graph_command_overflow(graph_commands_host_) != 0) {
        throw std::logic_error(
            "indexer graph cannot add nodes after replay");
      }
      if (!graph_capture_configured_) {
        graph_trace_ =
            std::getenv("SPARK_TRANSPORT_TRACE") != nullptr;
        graph_capture_configured_ = true;
      }
      if (std::find(
              graph_capture_streams_.begin(),
              graph_capture_streams_.end(), cuda_stream) ==
          graph_capture_streams_.end()) {
        graph_capture_streams_.push_back(cuda_stream);
      }
    }

    try {
      worker_->enqueue_graph(
          input, output, q, descriptor.input_bytes, cuda_stream,
          graph_commands_device_, graph_trace_);
    } catch (...) {
      rollback_empty_capture();
      throw;
    }
    captured_q_mask_.fetch_or(
        std::uint64_t{1} << (q - 1), std::memory_order_release);
    graph_capture_nodes_.fetch_add(1, std::memory_order_release);
    graph_polling_enabled_.store(true, std::memory_order_release);
    progress_cv_.notify_one();
  }

  Tp4IndexerGraphReplayStatus graph_replay_status() const noexcept {
    return Tp4IndexerGraphReplayStatus{
        graph_capture_nodes_.load(std::memory_order_acquire),
        captured_q_mask_.load(std::memory_order_acquire),
        tp4_graph_command_published(graph_commands_host_),
        tp4_graph_command_consumed(graph_commands_host_),
        tp4_graph_command_completed(graph_commands_host_),
        tp4_graph_command_overflow(graph_commands_host_),
        graph_capture_configured_.load(std::memory_order_acquire),
        graph_polling_enabled_.load(std::memory_order_acquire),
        graph_host_native_atomics_supported_,
        submit_affinity_verified_,
        progress_affinity_verified_.load(std::memory_order_acquire),
        static_cast<int>(*options_.graph_submit_cpu),
        static_cast<int>(*options_.graph_progress_cpu)};
  }

 private:
  void rollback_empty_capture() noexcept {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (graph_capture_nodes_.load(std::memory_order_acquire) == 0) {
      graph_capture_configured_ = false;
      graph_capture_streams_.clear();
    }
  }

  void start_progress_thread() {
    std::promise<std::string> startup_promise;
    auto startup = startup_promise.get_future();
    progress_thread_ = std::thread(
        [this, promise = std::move(startup_promise)]() mutable {
          try {
            pin_current_thread_to_cpu(
                *options_.graph_progress_cpu,
                "indexer graph progress thread");
            progress_affinity_verified_.store(
                true, std::memory_order_release);
            promise.set_value({});
          } catch (const std::exception& error) {
            promise.set_value(error.what());
            return;
          } catch (...) {
            promise.set_value(
                "unknown indexer graph progress affinity failure");
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
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      progress_cv_.wait(lock, [this] {
        return stop_requested_.load(std::memory_order_acquire) ||
               graph_polling_enabled_.load(std::memory_order_acquire);
      });
    }
    if (!stop_requested_.load(std::memory_order_acquire)) {
      progress_graph_commands();
    }
  }

  void progress_graph_commands() noexcept {
    std::uint32_t poll_misses{};
    while (!stop_requested_.load(std::memory_order_acquire)) {
      if (tp4_graph_command_overflow(graph_commands_host_) != 0) {
        fatal_async_failure(
            "indexer CUDA Graph command ring overflow");
      }

      Tp4GraphCommand command{};
      const std::uint64_t expected = graph_consumed_sequence_ + 1;
      if (!tp4_graph_command_try_consume_tagged_layout(
              graph_commands_host_, expected,
              Tp4GraphCommandKind::kIndexerAllgather,
              kTp4IndexerGraphDescriptorVersion,
              kTp4IndexerGraphBytesPerRow,
              kTp4IndexerGraphMaximumQ, &command)) {
        if (tp4_graph_command_overflow(graph_commands_host_) != 0) {
          fatal_async_failure(
              "invalid indexer CUDA Graph family descriptor");
        }
        adaptive_graph_poll_pause(poll_misses);
        continue;
      }
      poll_misses = 0;
      graph_consumed_sequence_ = command.sequence;
      if (!tp4_graph_doorbell_token_valid(
              command.sequence, command.q)) {
        fatal_async_failure(
            "invalid indexer graph doorbell descriptor");
      }
      try {
        progress(
            command.sequence, command.q, command.payload_bytes,
            tp4_graph_doorbell_token(command.sequence, command.q),
            command.trace != 0);
      } catch (const std::exception& error) {
        fatal_async_failure(error.what());
      } catch (...) {
        fatal_async_failure("unknown indexer graph error");
      }
      tp4_graph_command_complete(
          graph_commands_host_, command.sequence);
    }
  }

  void progress(
      std::uint64_t sequence, std::uint32_t q,
      std::size_t input_bytes, std::uint64_t doorbell_token,
      bool trace) {
    wait_for_exact_sequence(
        &control0_->producer_sequence, doorbell_token,
        "GPU indexer input staging");
    if (trace) {
      std::fprintf(
          stderr,
          "INDEXER rank=%u state=round0 sequence=%llu token=%llu "
          "q=%u bytes=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), q,
          input_bytes);
    }
    exchange_round(
        *endpoint0_, *control0_, layout_.round0, input_bytes,
        doorbell_token, "GPU indexer round0 consumption");
    if (trace) {
      std::fprintf(
          stderr,
          "INDEXER rank=%u state=round1 sequence=%llu token=%llu "
          "q=%u bytes=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), q,
          input_bytes * 2);
    }
    exchange_round(
        *endpoint1_, *control1_, layout_.round1, input_bytes * 2,
        doorbell_token, "GPU indexer round1 consumption");
  }

  Tp4IndexerGraphOptions options_;
  Tp4AllgatherBufferLayout layout_{};
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
  std::unique_ptr<GpuTp4AllgatherWorker> worker_;
  std::mutex state_mutex_;
  std::condition_variable progress_cv_;
  std::thread progress_thread_;
  std::vector<void*> graph_capture_streams_;
  std::uint64_t graph_consumed_sequence_{};
  std::atomic<bool> graph_capture_configured_{false};
  std::atomic<bool> graph_polling_enabled_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<std::uint64_t> graph_capture_nodes_{0};
  std::atomic<std::uint64_t> captured_q_mask_{0};
  bool graph_host_native_atomics_supported_{};
  bool submit_affinity_verified_{};
  std::atomic<bool> progress_affinity_verified_{false};
  bool graph_trace_{};
};

Tp4IndexerGraphSession::Tp4IndexerGraphSession(
    Tp4IndexerGraphOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Tp4IndexerGraphSession::~Tp4IndexerGraphSession() = default;

void Tp4IndexerGraphSession::capture_all_gather(
    const void* input, void* output, std::uint32_t q,
    void* cuda_stream) {
  impl_->capture_all_gather(input, output, q, cuda_stream);
}

Tp4IndexerGraphReplayStatus
Tp4IndexerGraphSession::graph_replay_status() const noexcept {
  return impl_->graph_replay_status();
}

}  // namespace spark_transport
