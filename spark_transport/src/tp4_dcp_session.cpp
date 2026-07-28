#include "spark_transport/tp4_dcp_session.hpp"

#include "cuda_event_gate.hpp"
#include "cuda_stream_handoff.hpp"
#include "spark_transport/control_channel.hpp"
#include "spark_transport/eager_staging_timeout.hpp"
#include "spark_transport/graph_poll_policy.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_tp4_dcp_combine.hpp"
#include "spark_transport/gpu_tp4_dcp_query.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/tp4_graph_command.hpp"
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
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace spark_transport {
namespace {

constexpr std::uint64_t kDefaultMaxInflight = 64;
constexpr std::uint64_t kMaximumMaxInflight = 4096;
constexpr auto kDcpStagingTimeout = std::chrono::seconds(5);

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

std::chrono::seconds eager_input_ready_timeout() {
  return parse_eager_staging_timeout(
      std::getenv("SPARK_TP4_DCP_EAGER_READY_TIMEOUT_SECONDS"),
      "SPARK_TP4_DCP_EAGER_READY_TIMEOUT_SECONDS");
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

void wait_for_sequence(const std::uint64_t* address,
                       std::uint64_t expected, const char* name,
                       bool require_exact = false,
                       std::chrono::seconds timeout =
                           kDcpStagingTimeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const std::uint64_t observed = load_sequence(address);
    if (observed == expected ||
        (!require_exact && observed > expected)) {
      return;
    }
    if (require_exact && observed > expected) {
      throw std::runtime_error(
          std::string("mismatched ") + name + " token: expected=" +
          std::to_string(expected) +
          " observed=" + std::to_string(observed));
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          std::string("timed out waiting for ") + name +
          ": expected=" + std::to_string(expected) +
          " observed=" + std::to_string(observed));
    }
  }
}

ControlChannel open_channel(const Tp4RoundPlan& plan,
                            const std::string& peer,
                            std::uint16_t port) {
  return plan.server ? ControlChannel::listen_and_accept(port)
                     : ControlChannel::connect(peer, port);
}

struct RoundPhaseTiming {
  double publish_and_post_us{};
  double send_completion_us{};
  double local_consume_us{};
  double acknowledgement_post_us{};
  double peer_consume_us{};
  double total_us{};
};

double elapsed_microseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) noexcept {
  return std::chrono::duration<double, std::micro>(end - begin).count();
}

RoundPhaseTiming exchange_round(
    VerbsEndpoint& endpoint, DoorbellControl& control,
    const Tp2BufferLayout& layout, std::size_t bytes,
    std::uint64_t doorbell_token, bool require_exact_doorbell,
    std::chrono::seconds timeout, bool measure_phases) {
  using Clock = std::chrono::steady_clock;
  const auto begin = measure_phases ? Clock::now() : Clock::time_point{};
  store_sequence(&control.producer_sequence, doorbell_token);
  endpoint.write(layout.send_offset, layout.receive_offset, bytes,
                 doorbell_token, false);
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, producer_sequence),
      layout.control_offset + offsetof(DoorbellControl, remote_sequence),
      sizeof(doorbell_token), doorbell_token);
  const auto posted = measure_phases ? Clock::now() : Clock::time_point{};
  endpoint.wait_for_send(doorbell_token);
  const auto sent = measure_phases ? Clock::now() : Clock::time_point{};
  wait_for_sequence(&control.consumer_sequence, doorbell_token,
                    "GPU DCP operation consumption",
                    require_exact_doorbell, timeout);
  const auto consumed = measure_phases ? Clock::now() : Clock::time_point{};
  endpoint.write(
      layout.control_offset + offsetof(DoorbellControl, consumer_sequence),
      layout.control_offset +
          offsetof(DoorbellControl, acknowledgement_sequence),
      sizeof(doorbell_token), doorbell_token, false);
  const auto acknowledgement_posted =
      measure_phases ? Clock::now() : Clock::time_point{};
  wait_for_sequence(&control.acknowledgement_sequence, doorbell_token,
                    "peer DCP operation consumption",
                    require_exact_doorbell, timeout);
  const auto acknowledged =
      measure_phases ? Clock::now() : Clock::time_point{};
  if (!measure_phases) {
    return {};
  }
  return RoundPhaseTiming{
      elapsed_microseconds(begin, posted),
      elapsed_microseconds(posted, sent),
      elapsed_microseconds(sent, consumed),
      elapsed_microseconds(consumed, acknowledgement_posted),
      elapsed_microseconds(acknowledgement_posted, acknowledged),
      elapsed_microseconds(begin, acknowledged)};
}

[[noreturn]] void fatal_async_failure(const char* message) noexcept {
  std::fprintf(stderr, "FATAL asynchronous TP4 DCP operation failed: %s\n",
               message);
  std::fflush(stderr);
  std::abort();
}

enum class SubmissionKind {
  kQuery,
  kCombine,
};

const char* submission_name(SubmissionKind kind) {
  return kind == SubmissionKind::kQuery ? "query" : "combine";
}

[[noreturn]] void fatal_submission_failure(
    SubmissionKind kind, std::uint64_t sequence, std::uint32_t q,
    std::uint32_t head_dimension, const char* message) noexcept {
  std::fprintf(
      stderr,
      "FATAL asynchronous TP4 DCP %s sequence %llu q=%u "
      "parameter=%u failed: %s\n",
      submission_name(kind), static_cast<unsigned long long>(sequence), q,
      head_dimension, message);
  std::fflush(stderr);
  std::abort();
}

struct Submission {
  std::uint64_t sequence{};
  SubmissionKind kind{};
  std::uint32_t q{};
  std::uint32_t head_dimension{};
  bool trace{};
};

void adaptive_graph_poll_pause(
    std::uint32_t& misses, GraphPollPolicy policy) noexcept {
  ++misses;
#if defined(__aarch64__)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
  if (
      policy == GraphPollPolicy::kAdaptiveYield &&
      misses >= 4096) {
    misses = 1024;
    std::this_thread::yield();
  }
}

void require_exclusive_current_cpu(std::uint32_t expected_cpu) {
#if defined(__linux__)
  if (expected_cpu >= CPU_SETSIZE) {
    throw std::invalid_argument("DCP graph CPU exceeds CPU_SETSIZE");
  }
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  const int result =
      pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity);
  if (result != 0) {
    throw std::runtime_error(
        std::string("pthread_getaffinity_np DCP graph thread: ") +
        std::strerror(result));
  }
  std::size_t enabled{};
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    enabled += CPU_ISSET(cpu, &affinity) ? 1U : 0U;
  }
  if (enabled != 1 || !CPU_ISSET(expected_cpu, &affinity)) {
    throw std::runtime_error(
        "DCP graph thread must be pinned exclusively to CPU " +
        std::to_string(expected_cpu));
  }
#else
  (void)expected_cpu;
  throw std::runtime_error("DCP graph CPU affinity requires Linux");
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
  throw std::runtime_error("DCP graph CPU affinity requires Linux");
#endif
}

}  // namespace

class Tp4DcpSession::Impl {
 public:
  explicit Impl(Tp4DcpOptions options)
      : options_(std::move(options)),
        layout_(make_tp4_dcp_query_buffer_layout()),
        max_inflight_(max_inflight_collectives()),
        eager_protocol_timeout_(eager_protocol_timeout()),
        graph_poll_policy_(parse_graph_poll_policy(
            std::getenv("SPARK_TP4_DCP_GRAPH_POLL_POLICY"),
            "SPARK_TP4_DCP_GRAPH_POLL_POLICY")) {
    if (options_.rank >= kTp4DcpQueryWorldSize ||
        options_.peer0.empty() || options_.peer1.empty() ||
        options_.device0.empty() || options_.device1.empty()) {
      throw std::invalid_argument("invalid TP4 DCP options");
    }
    const bool submit_cpu_set = options_.graph_submit_cpu.has_value();
    const bool progress_cpu_set = options_.graph_progress_cpu.has_value();
    if (submit_cpu_set != progress_cpu_set) {
      throw std::invalid_argument(
          "DCP graph submit/progress CPUs must be configured together");
    }
    validate_graph_poll_policy_configuration(
        graph_poll_policy_, progress_cpu_set,
        "SPARK_TP4_DCP_GRAPH_POLL_POLICY");
    if (submit_cpu_set) {
      if (options_.graph_submit_cpu == options_.graph_progress_cpu) {
        throw std::invalid_argument(
            "DCP graph submit/progress CPUs must be distinct");
      }
      pin_current_thread_to_cpu(*options_.graph_submit_cpu,
                                "DCP graph submission thread");
      submit_affinity_verified_ = true;
    } else {
      eager_input_gates_.emplace(
          max_inflight_, eager_input_ready_timeout(),
          "GPU DCP eager input readiness");
    }

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
    if (submit_cpu_set) {
      int device{};
      int host_native_atomics{};
      check_cuda(cudaGetDevice(&device), "cudaGetDevice DCP graph");
      check_cuda(
          cudaDeviceGetAttribute(
              &host_native_atomics,
              cudaDevAttrHostNativeAtomicSupported, device),
          "cudaDeviceGetAttribute DCP host native atomics");
      graph_host_native_atomics_supported_ = host_native_atomics != 0;
      graph_commands_buffer_ = MemoryBuffer::allocate(
          MemoryKind::kCudaMapped, sizeof(Tp4GraphCommandRing));
      graph_commands_host_ = static_cast<Tp4GraphCommandRing*>(
          graph_commands_buffer_->host_data());
      graph_commands_device_ = static_cast<Tp4GraphCommandRing*>(
          graph_commands_buffer_->device_data());
    }
    query_worker_ = std::make_unique<GpuTp4DcpQueryWorker>(
        options_.rank, layout_, buffer0_->device_data(),
        buffer1_->device_data());
    combine_worker_ = std::make_unique<GpuTp4DcpCombineWorker>(
        options_.rank, layout_, buffer0_->device_data(),
        buffer1_->device_data());
    channel0_->barrier();
    channel1_->barrier();
    start_progress_thread();
  }

  ~Impl() {
    // Graph kernels can be blocked on the progress thread. Keep progress alive
    // until every queued replay on every captured stream has drained.
    for (void* graph_stream : graph_capture_streams_) {
      const auto result =
          cudaStreamSynchronize(static_cast<cudaStream_t>(graph_stream));
      if (result != cudaSuccess) {
        fatal_async_failure(cudaGetErrorString(result));
      }
    }
    if (caller_stream_set_ && graph_capture_streams_.empty()) {
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
    query_worker_.reset();
    combine_worker_.reset();
  }

  void query_all_gather(const void* input, void* output, std::uint32_t q,
                        void* cuda_stream) {
    static_cast<void>(tp4_dcp_query_input_bytes(q));
    if (input == nullptr || output == nullptr) {
      throw std::invalid_argument("TP4 DCP query tensor pointer is null");
    }
    submit(SubmissionKind::kQuery, q, 0, cuda_stream,
           [this, input, output, q, cuda_stream](
               std::uint64_t sequence) {
             query_worker_->enqueue(input, output, q, cuda_stream,
                                    sequence);
           });
  }

  void combine(const void* output_bf16, const void* lse_fp32,
               void* reduced_output_bf16, void* reduced_lse_fp32,
               std::uint32_t q, std::uint32_t head_dimension,
               std::uint32_t query_stride, std::uint32_t head_stride,
               void* cuda_stream) {
    static_cast<void>(
        tp4_dcp_combine_round0_frame(q, head_dimension));
    if (output_bf16 == nullptr || lse_fp32 == nullptr ||
        reduced_output_bf16 == nullptr || reduced_lse_fp32 == nullptr) {
      throw std::invalid_argument(
          "TP4 DCP combine tensor pointer is null");
    }
    submit(SubmissionKind::kCombine, q, head_dimension, cuda_stream,
           [this, output_bf16, lse_fp32, reduced_output_bf16,
            reduced_lse_fp32, q, head_dimension, query_stride,
            head_stride, cuda_stream](std::uint64_t sequence) {
             combine_worker_->enqueue(
                 output_bf16, lse_fp32, reduced_output_bf16,
                 reduced_lse_fp32, q, head_dimension, query_stride,
                 head_stride, cuda_stream, sequence);
           });
  }

  void capture_query_all_gather(
      const void* input, void* output, std::uint32_t q,
      void* cuda_stream) {
    static_cast<void>(tp4_dcp_query_input_bytes(q));
    if (input == nullptr || output == nullptr) {
      throw std::invalid_argument(
          "graph TP4 DCP query tensor pointer is null");
    }
    prepare_graph_capture(cuda_stream);
    try {
      query_worker_->enqueue_graph(
          input, output, q, cuda_stream, graph_commands_device_,
          graph_trace_);
    } catch (...) {
      rollback_empty_graph_capture();
      throw;
    }
    graph_capture_nodes_.fetch_add(1, std::memory_order_release);
    graph_query_nodes_.fetch_add(1, std::memory_order_release);
    enable_graph_polling();
  }

  void capture_combine(
      const void* output_bf16, const void* lse_fp32,
      void* reduced_output_bf16, void* reduced_lse_fp32,
      std::uint32_t q, std::uint32_t head_dimension,
      std::uint32_t query_stride, std::uint32_t head_stride,
      void* cuda_stream) {
    static_cast<void>(
        tp4_dcp_combine_round0_frame(q, head_dimension));
    if (output_bf16 == nullptr || lse_fp32 == nullptr ||
        reduced_output_bf16 == nullptr || reduced_lse_fp32 == nullptr) {
      throw std::invalid_argument(
          "graph TP4 DCP combine tensor pointer is null");
    }
    prepare_graph_capture(cuda_stream);
    try {
      combine_worker_->enqueue_graph(
          output_bf16, lse_fp32, reduced_output_bf16,
          reduced_lse_fp32, q, head_dimension, query_stride,
          head_stride, cuda_stream, graph_commands_device_,
          graph_trace_);
    } catch (...) {
      rollback_empty_graph_capture();
      throw;
    }
    graph_capture_nodes_.fetch_add(1, std::memory_order_release);
    graph_combine_nodes_.fetch_add(1, std::memory_order_release);
    enable_graph_polling();
  }

  Tp4DcpGraphReplayStatus graph_replay_status() const noexcept {
    return Tp4DcpGraphReplayStatus{
        graph_capture_nodes_.load(std::memory_order_acquire),
        graph_query_nodes_.load(std::memory_order_acquire),
        graph_combine_nodes_.load(std::memory_order_acquire),
        tp4_graph_command_published(graph_commands_host_),
        tp4_graph_command_consumed(graph_commands_host_),
        tp4_graph_command_completed(graph_commands_host_),
        tp4_graph_command_overflow(graph_commands_host_),
        graph_capture_configured_.load(std::memory_order_acquire),
        graph_polling_enabled_.load(std::memory_order_acquire),
        graph_poll_policy_ == GraphPollPolicy::kDedicatedSpin,
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
  template <typename Enqueue>
  void submit(SubmissionKind kind, std::uint32_t q,
              std::uint32_t head_dimension, void* cuda_stream,
              Enqueue&& enqueue) {
    if (options_.graph_submit_cpu.has_value()) {
      throw std::logic_error(
          "graph-configured DCP session rejects eager submission");
    }
    const bool trace = std::getenv("SPARK_TRANSPORT_TRACE") != nullptr;
    {
      std::unique_lock<std::mutex> lock(submission_mutex_);
      if (graph_capture_configured_) {
        throw std::logic_error(
            "eager DCP submission cannot follow graph capture");
      }
      if (poisoned_) {
        throw std::runtime_error("TP4 DCP session is poisoned");
      }
      completion_cv_.wait(lock, [this] {
        return stopping_ || poisoned_ ||
               sequence_ - completed_sequence_ < max_inflight_;
      });
      if (stopping_) {
        throw std::runtime_error("TP4 DCP session is stopping");
      }
      if (poisoned_) {
        throw std::runtime_error("TP4 DCP session is poisoned");
      }
      if (sequence_ == std::numeric_limits<std::uint64_t>::max() - 1) {
        throw std::overflow_error("TP4 DCP sequence exhausted");
      }
      if (caller_stream_set_ && caller_stream_ != cuda_stream) {
        try {
          stream_handoff_.order(
              static_cast<cudaStream_t>(caller_stream_),
              static_cast<cudaStream_t>(cuda_stream));
          // The new stream owns the queued dependency even if the following
          // family kernel is rejected, so destruction must drain this stream.
          caller_stream_ = cuda_stream;
          caller_stream_set_ = true;
        } catch (...) {
          poisoned_ = true;
          completion_cv_.notify_all();
          throw;
        }
      }
      const std::uint64_t next_sequence = sequence_ + 1;
      submissions_.push_back(
          {next_sequence, kind, q, head_dimension, trace});
      try {
        if (!eager_input_gates_.has_value()) {
          throw std::logic_error("DCP eager input gate is unavailable");
        }
        eager_input_gates_->record(
            next_sequence, static_cast<cudaStream_t>(cuda_stream));
        enqueue(next_sequence);
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

  void prepare_graph_capture(void* cuda_stream) {
    if (graph_commands_host_ == nullptr) {
      throw std::invalid_argument(
          "graph TP4 DCP requires a graph-configured session");
    }
    if (!graph_host_native_atomics_supported_) {
      throw std::runtime_error(
          "graph TP4 DCP requires host-native GPU atomics");
    }
    const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
    cudaStreamCaptureStatus capture_status{};
    check_cuda(cudaStreamIsCapturing(caller_stream, &capture_status),
               "cudaStreamIsCapturing DCP graph");
    if (capture_status != cudaStreamCaptureStatusActive) {
      throw std::logic_error(
          "DCP graph setup requires active stream capture");
    }

    std::lock_guard<std::mutex> lock(submission_mutex_);
    if (stopping_) {
      throw std::runtime_error("TP4 DCP session is stopping");
    }
    if (poisoned_) {
      throw std::runtime_error("TP4 DCP session is poisoned");
    }
    if (sequence_ != 0 || completed_sequence_ != 0 ||
        !submissions_.empty()) {
      throw std::logic_error(
          "DCP graph capture must precede eager submissions");
    }
    if (tp4_graph_command_published(graph_commands_host_) != 0 ||
        tp4_graph_command_consumed(graph_commands_host_) != 0 ||
        tp4_graph_command_completed(graph_commands_host_) != 0 ||
        tp4_graph_command_overflow(graph_commands_host_) != 0) {
      throw std::logic_error(
          "DCP graph cannot add nodes after replay");
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

  void rollback_empty_graph_capture() noexcept {
    std::lock_guard<std::mutex> lock(submission_mutex_);
    if (graph_capture_nodes_.load(std::memory_order_acquire) == 0) {
      graph_capture_configured_ = false;
      graph_capture_streams_.clear();
    }
  }

  void enable_graph_polling() {
    graph_polling_enabled_.store(true, std::memory_order_release);
    progress_cv_.notify_one();
  }

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
              pin_current_thread_to_cpu(
                  *options_.graph_progress_cpu,
                  "DCP graph progress thread");
              progress_affinity_verified_.store(
                  true, std::memory_order_release);
            }
            promise.set_value({});
          } catch (const std::exception& error) {
            promise.set_value(error.what());
            return;
          } catch (...) {
            promise.set_value(
                "unknown DCP graph progress affinity failure");
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
        if (submissions_.empty()) {
          if (stopping_) {
            return;
          }
          continue;
        }
        submission = submissions_.front();
        submissions_.pop_front();
      }
      try {
        if (!eager_input_gates_.has_value()) {
          throw std::logic_error("DCP eager input gate is unavailable");
        }
        eager_input_gates_->wait(submission.sequence);
        progress(submission.sequence, submission.kind, submission.q,
                 submission.head_dimension, submission.sequence, false,
                 submission.trace);
      } catch (const std::exception& error) {
        fatal_submission_failure(
            submission.kind, submission.sequence, submission.q,
            submission.head_dimension, error.what());
      } catch (...) {
        fatal_submission_failure(
            submission.kind, submission.sequence, submission.q,
            submission.head_dimension, "unknown eager DCP error");
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
        fatal_async_failure("DCP CUDA Graph command ring overflow");
      }

      const std::uint64_t expected = graph_consumed_sequence_ + 1;
      Tp4GraphCommand peek{};
      if (!tp4_graph_command_try_peek(
              graph_commands_host_, expected, &peek)) {
        adaptive_graph_poll_pause(poll_misses, graph_poll_policy_);
        continue;
      }

      bool consumed{};
      SubmissionKind kind{};
      std::uint32_t head_dimension{};
      if (peek.kind == Tp4GraphCommandKind::kDcpQuery) {
        kind = SubmissionKind::kQuery;
        consumed = tp4_graph_command_try_consume_tagged_layout(
            graph_commands_host_, expected,
            Tp4GraphCommandKind::kDcpQuery, 0,
            static_cast<std::uint32_t>(kTp4DcpQueryBytesPerQ),
            kTp4DcpQueryMaxQ, &peek);
      } else if (
          peek.kind == Tp4GraphCommandKind::kDcpCombine &&
          tp4_dcp_combine_head_dimension_supported(peek.parameter)) {
        kind = SubmissionKind::kCombine;
        head_dimension = peek.parameter;
        const auto bytes_per_row =
            tp4_dcp_combine_round0_frame(1, head_dimension).total_bytes;
        consumed = tp4_graph_command_try_consume_tagged_layout(
            graph_commands_host_, expected,
            Tp4GraphCommandKind::kDcpCombine, head_dimension,
            static_cast<std::uint32_t>(bytes_per_row),
            kTp4DcpCombineMaxQ, &peek);
      } else {
        // Force the shared command layer to record the released descriptor as
        // a fatal family mismatch without acknowledging it.
        (void)tp4_graph_command_try_consume_tagged_layout(
            graph_commands_host_, expected,
            Tp4GraphCommandKind::kDcpQuery, 0,
            static_cast<std::uint32_t>(kTp4DcpQueryBytesPerQ),
            kTp4DcpQueryMaxQ, &peek);
      }

      if (!consumed) {
        if (tp4_graph_command_overflow(graph_commands_host_) != 0) {
          fatal_async_failure(
              "invalid DCP CUDA Graph family descriptor");
        }
        adaptive_graph_poll_pause(poll_misses, graph_poll_policy_);
        continue;
      }
      poll_misses = 0;
      graph_consumed_sequence_ = peek.sequence;
      if (!tp4_graph_doorbell_token_valid(peek.sequence, peek.q)) {
        fatal_async_failure(
            "invalid DCP CUDA Graph doorbell descriptor");
      }
      try {
        progress(
            peek.sequence, kind, peek.q, head_dimension,
            tp4_graph_doorbell_token(peek.sequence, peek.q), true,
            peek.trace != 0);
      } catch (const std::exception& error) {
        fatal_submission_failure(
            kind, peek.sequence, peek.q, head_dimension, error.what());
      } catch (...) {
        fatal_submission_failure(
            kind, peek.sequence, peek.q, head_dimension,
            "unknown DCP CUDA Graph error");
      }
      tp4_graph_command_complete(graph_commands_host_, peek.sequence);
    }
  }

  void progress(std::uint64_t sequence, SubmissionKind kind,
                std::uint32_t q, std::uint32_t head_dimension,
                std::uint64_t doorbell_token,
                bool require_exact_doorbell, bool trace) {
    const auto protocol_timeout =
        require_exact_doorbell ? kDcpStagingTimeout
                               : eager_protocol_timeout_;
    wait_for_sequence(
        &control0_->producer_sequence, doorbell_token,
        "GPU DCP operation input staging", require_exact_doorbell,
        protocol_timeout);
    const bool query = kind == SubmissionKind::kQuery;
    const std::size_t round0_bytes =
        query ? tp4_dcp_query_input_bytes(q)
              : tp4_dcp_combine_round0_frame(q, head_dimension)
                    .total_bytes;
    const std::size_t round1_bytes =
        query ? tp4_dcp_query_input_bytes(q) * 2
              : tp4_dcp_combine_round1_frame(q, head_dimension)
                    .total_bytes;
    if (trace) {
      std::fprintf(
          stderr,
          "DCP rank=%u family=%s state=round0 sequence=%llu token=%llu "
          "q=%u parameter=%u bytes=%zu\n",
          options_.rank, submission_name(kind),
          static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), q,
          head_dimension, round0_bytes);
    }
    const auto round0_timing = exchange_round(
        *endpoint0_, *control0_, layout_.round0, round0_bytes,
        doorbell_token, require_exact_doorbell, protocol_timeout,
        trace);
    if (trace) {
      std::fprintf(
          stderr,
          "DCP rank=%u family=%s state=round1 sequence=%llu token=%llu "
          "q=%u parameter=%u bytes=%zu\n",
          options_.rank, submission_name(kind),
          static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), q,
          head_dimension, round1_bytes);
    }
    const auto round1_timing = exchange_round(
        *endpoint1_, *control1_, layout_.round1, round1_bytes,
        doorbell_token, require_exact_doorbell, protocol_timeout,
        trace);
    if (trace) {
      std::fprintf(
          stderr,
          "DCP_PHASE rank=%u family=%s sequence=%llu q=%u parameter=%u "
          "round0_post_us=%.3f round0_send_cq_us=%.3f "
          "round0_local_gpu_us=%.3f round0_ack_post_us=%.3f "
          "round0_peer_gpu_us=%.3f round0_total_us=%.3f "
          "round1_post_us=%.3f round1_send_cq_us=%.3f "
          "round1_local_gpu_us=%.3f round1_ack_post_us=%.3f "
          "round1_peer_gpu_us=%.3f round1_total_us=%.3f "
          "total_us=%.3f\n",
          options_.rank, submission_name(kind),
          static_cast<unsigned long long>(sequence), q, head_dimension,
          round0_timing.publish_and_post_us,
          round0_timing.send_completion_us,
          round0_timing.local_consume_us,
          round0_timing.acknowledgement_post_us,
          round0_timing.peer_consume_us,
          round0_timing.total_us,
          round1_timing.publish_and_post_us,
          round1_timing.send_completion_us,
          round1_timing.local_consume_us,
          round1_timing.acknowledgement_post_us,
          round1_timing.peer_consume_us,
          round1_timing.total_us,
          round0_timing.total_us + round1_timing.total_us);
    }
  }

  Tp4DcpOptions options_;
  Tp4DcpQueryBufferLayout layout_{};
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
  std::unique_ptr<GpuTp4DcpQueryWorker> query_worker_;
  std::unique_ptr<GpuTp4DcpCombineWorker> combine_worker_;
  std::optional<CudaEventGatePool> eager_input_gates_;
  CudaStreamHandoff stream_handoff_;
  std::mutex submission_mutex_;
  std::condition_variable progress_cv_;
  std::condition_variable completion_cv_;
  std::deque<Submission> submissions_;
  std::thread progress_thread_;
  const std::uint64_t max_inflight_;
  const std::chrono::seconds eager_protocol_timeout_;
  const GraphPollPolicy graph_poll_policy_;
  std::atomic<bool> graph_polling_enabled_{false};
  std::atomic<bool> stop_requested_{false};
  std::uint64_t sequence_{};
  std::uint64_t completed_sequence_{};
  std::uint64_t graph_consumed_sequence_{};
  void* caller_stream_{};
  bool caller_stream_set_{};
  std::vector<void*> graph_capture_streams_;
  std::atomic<bool> graph_capture_configured_{false};
  std::atomic<std::uint64_t> graph_capture_nodes_{0};
  std::atomic<std::uint64_t> graph_query_nodes_{0};
  std::atomic<std::uint64_t> graph_combine_nodes_{0};
  bool graph_host_native_atomics_supported_{};
  bool submit_affinity_verified_{};
  std::atomic<bool> progress_affinity_verified_{false};
  bool graph_trace_{};
  bool stopping_{};
  bool poisoned_{};
};

Tp4DcpSession::Tp4DcpSession(Tp4DcpOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Tp4DcpSession::~Tp4DcpSession() = default;

void Tp4DcpSession::query_all_gather(
    const void* input, void* output, std::uint32_t q,
    void* cuda_stream) {
  impl_->query_all_gather(input, output, q, cuda_stream);
}

void Tp4DcpSession::combine(
    const void* output_bf16, const void* lse_fp32,
    void* reduced_output_bf16, void* reduced_lse_fp32,
    std::uint32_t q, std::uint32_t head_dimension,
    std::uint32_t query_stride, std::uint32_t head_stride,
    void* cuda_stream) {
  impl_->combine(output_bf16, lse_fp32, reduced_output_bf16,
                 reduced_lse_fp32, q, head_dimension, query_stride,
                 head_stride, cuda_stream);
}

void Tp4DcpSession::capture_query_all_gather(
    const void* input, void* output, std::uint32_t q,
    void* cuda_stream) {
  impl_->capture_query_all_gather(input, output, q, cuda_stream);
}

void Tp4DcpSession::capture_combine(
    const void* output_bf16, const void* lse_fp32,
    void* reduced_output_bf16, void* reduced_lse_fp32,
    std::uint32_t q, std::uint32_t head_dimension,
    std::uint32_t query_stride, std::uint32_t head_stride,
    void* cuda_stream) {
  impl_->capture_combine(
      output_bf16, lse_fp32, reduced_output_bf16, reduced_lse_fp32,
      q, head_dimension, query_stride, head_stride, cuda_stream);
}

Tp4DcpGraphReplayStatus
Tp4DcpSession::graph_replay_status() const noexcept {
  return impl_->graph_replay_status();
}

}  // namespace spark_transport
