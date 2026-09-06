#include "spark_transport/tp4_session.hpp"

#include "cuda_event_gate.hpp"
#include "cuda_stream_handoff.hpp"
#include "tp4_dual_port_striped_host.hpp"
#include "spark_transport/control_channel.hpp"
#include "spark_transport/eager_staging_timeout.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_tp4_tensor.hpp"
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
constexpr auto kGraphProtocolTimeout = std::chrono::seconds(5);
// The record layout remains EndpointInfo v1. A distinct wire version makes a
// mixed old/new deferred-credit deployment fail on both peers before either
// peer enters the data path; old binaries otherwise ignore reserved tags.
constexpr std::uint16_t kTwoSlotDeferredEndpointVersion = 2;
constexpr std::uint16_t kGraphLayoutEndpointVersion = 4;
// Constant tag: version-4 graph peers exchange the exact
// GraphGeometryInfo record below instead of folding geometry into
// this 16-bit field.
constexpr std::uint16_t kGraphLayoutEndpointTag = 0xc211;

// Exact graph-row geometry record, exchanged over the control channel
// after EndpointInfo for every graph session. Both peers must present
// identical values in every field; geometry identity is never hashed
// or truncated.
constexpr std::uint32_t kGraphGeometryMagic = 0x53474d47;  // "SGMG"

struct GraphGeometryInfo {
  std::uint32_t magic{kGraphGeometryMagic};
  std::uint16_t version{1};
  std::uint16_t reserved{};
  std::uint32_t elements_per_row{};
  std::uint32_t bytes_per_row{};
};

static_assert(sizeof(GraphGeometryInfo) == 16);
static_assert(std::is_trivially_copyable_v<GraphGeometryInfo>);

bool graph_capacity_supported(std::size_t payload_bytes,
                              std::uint32_t bytes_per_row) {
  for (std::uint32_t q = 1;
       q <= kTp4GraphAllreduceMaximumQ; ++q) {
    if (payload_bytes == tp4_graph_payload_bytes(q, bytes_per_row)) {
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

bool graph_direct_doorbell_enabled() {
  const char* value = std::getenv("SPARK_TP4_GRAPH_DIRECT_DOORBELL");
  if (value == nullptr || value[0] == '\0' ||
      std::strcmp(value, "0") == 0) {
    return false;
  }
  if (std::strcmp(value, "1") == 0) {
    return true;
  }
  throw std::invalid_argument(
      "SPARK_TP4_GRAPH_DIRECT_DOORBELL must be 0 or 1");
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

void exchange_and_connect_endpoint(
    ControlChannel& channel, VerbsEndpoint& endpoint,
    Tp4AllreduceProtocol protocol, Tp4AllreduceSchedule schedule,
    bool graph_session, std::uint32_t elements_per_row,
    std::uint32_t bytes_per_row, bool graph_direct_doorbell) {
  EndpointInfo local = endpoint.local_info();
  if (schedule == Tp4AllreduceSchedule::kDualPortStriped) {
    local.version = detail::kTp4DualPortStripedEndpointVersion;
    local.reserved = detail::kTp4DualPortStripedEndpointTag;
  } else if (graph_session) {
    local.version = kGraphLayoutEndpointVersion;
    local.reserved = kGraphLayoutEndpointTag;
  } else if (tp4_protocol_uses_deferred_ack(protocol)) {
    local.version = kTwoSlotDeferredEndpointVersion;
    local.reserved = tp4_endpoint_protocol_tag(protocol);
  } else {
    local.reserved = tp4_endpoint_protocol_tag(protocol);
  }
  const EndpointInfo remote = channel.exchange(local);
  if (remote.version != local.version) {
    throw std::runtime_error(
        "TP4 all-reduce endpoint version mismatch between peers");
  }
  if (remote.reserved != local.reserved) {
    throw std::runtime_error(
        "TP4 all-reduce protocol mismatch between peers");
  }
  if (remote.buffer_bytes != local.buffer_bytes) {
    throw std::runtime_error(
        "TP4 all-reduce payload arena mismatch between peers");
  }
  if (graph_session) {
    GraphGeometryInfo local_geometry{};
    local_geometry.reserved = graph_direct_doorbell ? 1U : 0U;
    local_geometry.elements_per_row = elements_per_row;
    local_geometry.bytes_per_row = bytes_per_row;
    const GraphGeometryInfo remote_geometry =
        channel.exchange(local_geometry);
    if (remote_geometry.magic != local_geometry.magic ||
        remote_geometry.version != local_geometry.version ||
        remote_geometry.reserved != local_geometry.reserved) {
      throw std::runtime_error(
          "TP4 graph geometry record mismatch between peers");
    }
    if (remote_geometry.elements_per_row !=
            local_geometry.elements_per_row ||
        remote_geometry.bytes_per_row != local_geometry.bytes_per_row) {
      throw std::runtime_error(
          "TP4 graph row-geometry mismatch between peers");
    }
  }
  endpoint.connect(remote, local.version);
}

DoorbellControl* slot_control(MemoryBuffer& buffer,
                              const ExchangeBufferLayout& layout,
                              std::size_t slot) {
  return reinterpret_cast<DoorbellControl*>(
      static_cast<std::uint8_t*>(buffer.host_data()) +
       slot * layout.total_bytes + layout.control_offset);
}

DoorbellControl* striped_lane_control(
    MemoryBuffer& buffer, const Tp4StripedEndpointLayout& layout,
    std::size_t generation, Tp4TensorStripe stripe) {
  const Tp4StripedLaneRegion region =
      tp4_striped_lane_region(layout, generation, stripe);
  return reinterpret_cast<DoorbellControl*>(
      static_cast<std::uint8_t*>(buffer.host_data()) +
      region.control_offset);
}

void post_striped_transfer(
    VerbsEndpoint& endpoint, const Tp4StripedLaneRegion& region,
    std::size_t bytes, std::uint64_t doorbell_token,
    std::uint64_t work_id) {
  endpoint.write(region.send_offset, region.receive_offset, bytes,
                 work_id, false);
  endpoint.write(
      region.control_offset +
          offsetof(DoorbellControl, producer_sequence),
      region.control_offset +
          offsetof(DoorbellControl, remote_sequence),
      sizeof(doorbell_token), work_id);
}

void post_striped_credit(
    VerbsEndpoint& endpoint, DoorbellControl& control,
    const Tp4StripedLaneRegion& region, std::uint64_t sequence,
    std::uint64_t work_id) {
  store_sequence(&control.reserved, sequence);
  endpoint.write(
      region.control_offset + offsetof(DoorbellControl, reserved),
      region.control_offset +
          offsetof(DoorbellControl, acknowledgement_sequence),
      sizeof(sequence), work_id, true);
}

void exchange_round(VerbsEndpoint& endpoint, DoorbellControl& control,
                    const ExchangeBufferLayout& layout, std::size_t bytes,
                    std::size_t slot_offset, std::uint64_t sequence,
                    std::uint64_t doorbell_token, std::uint32_t rank,
                    std::uint32_t round, bool trace,
                    Tp4AllreduceProtocol protocol,
                    bool require_exact_doorbell,
                    std::chrono::seconds timeout) {
  store_sequence(&control.producer_sequence, doorbell_token);
  endpoint.write(slot_offset + layout.send_offset,
                 slot_offset + layout.receive_offset, bytes,
                 doorbell_token, false);
  endpoint.write(
      slot_offset + layout.control_offset +
          offsetof(DoorbellControl, producer_sequence),
      slot_offset + layout.control_offset +
          offsetof(DoorbellControl, remote_sequence),
      sizeof(doorbell_token), doorbell_token);
  if (trace) {
    std::fprintf(stderr,
                 "EXCHANGE rank=%u round=%u state=doorbell_posted\n",
                 rank, round);
  }
  if (tp4_protocol_uses_deferred_ack(protocol)) {
    endpoint.wait_for_send_through(doorbell_token);
  } else {
    endpoint.wait_for_send(doorbell_token);
  }
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
  if (tp4_protocol_uses_deferred_ack(protocol)) {
    store_sequence(&control.reserved, sequence);
    endpoint.write(
        slot_offset + layout.control_offset +
            offsetof(DoorbellControl, reserved),
        slot_offset + layout.control_offset +
            offsetof(DoorbellControl, acknowledgement_sequence),
        sizeof(sequence), doorbell_token, true);
    if (trace) {
      std::fprintf(stderr,
                   "EXCHANGE rank=%u round=%u state=credit_posted\n",
                   rank, round);
    }
  } else {
    endpoint.write(
        slot_offset + layout.control_offset +
            offsetof(DoorbellControl, consumer_sequence),
        slot_offset + layout.control_offset +
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
  if (misses == std::numeric_limits<std::uint32_t>::max()) {
    // The graph progress thread owns an exclusively pinned CPU. Entering the
    // scheduler between model-layer collectives adds exposed wake latency to
    // every replay; retain only the architecture pause hint and wrap the
    // diagnostic counter without yielding that dedicated core.
    misses = 0;
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
        graph_direct_doorbell_(graph_direct_doorbell_enabled()),
        max_inflight_(max_inflight_collectives()) {
    if (options_.rank >= 4 || options_.peer0.empty() ||
        options_.peer1.empty() || options_.device0.empty() ||
        options_.device1.empty() || options_.payload_bytes == 0 ||
        options_.payload_bytes % 2 != 0 || options_.control_port0 == 0 ||
        options_.control_port1 == 0) {
      throw std::invalid_argument("invalid TP4 all-reduce options");
    }
    if (options_.elements_per_row == 0 || options_.bytes_per_row == 0 ||
        static_cast<std::uint64_t>(options_.bytes_per_row) !=
            static_cast<std::uint64_t>(options_.elements_per_row) *
                kTp4GraphBytesPerElement ||
        static_cast<std::uint64_t>(options_.bytes_per_row) *
                kTp4GraphAllreduceMaximumQ >
            std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument(
          "TP4 graph row geometry must describe contiguous BF16 rows");
    }
    (void)tp4_allreduce_protocol_from_wire(
        static_cast<std::uint32_t>(options_.protocol));
    if (!tp4_graph_kernel_strategy_valid(
            options_.graph_kernel_strategy)) {
      throw std::invalid_argument("invalid graph TP4 kernel strategy");
    }
    if (!tp4_allreduce_schedule_valid(options_.schedule)) {
      throw std::invalid_argument("invalid TP4 all-reduce schedule");
    }
    if (graph_direct_doorbell_ &&
        (options_.schedule != Tp4AllreduceSchedule::kSequential ||
         !tp4_protocol_uses_deferred_ack(options_.protocol))) {
      throw std::invalid_argument(
          "direct-doorbell graph TP4 requires sequential two-slot "
          "deferred ACK");
    }
    const bool submit_cpu_set = options_.graph_submit_cpu.has_value();
    const bool progress_cpu_set = options_.graph_progress_cpu.has_value();
    if (submit_cpu_set != progress_cpu_set) {
      throw std::invalid_argument(
          "graph submit/progress CPUs must be configured together");
    }
    if (options_.schedule == Tp4AllreduceSchedule::kDualPortStriped &&
        !detail::tp4_dual_port_striped_options_valid(options_)) {
      throw std::invalid_argument(
          "dual-port striped TP4 requires distinct graph submit/progress "
          "CPUs, two-slot deferred ACK, the fused graph selector, and an "
          "exact Q40 or Q512 session capacity");
    }
    if (tp4_protocol_uses_deferred_ack(options_.protocol) &&
        (!submit_cpu_set ||
         !graph_capacity_supported(options_.payload_bytes,
                                   options_.bytes_per_row))) {
      throw std::invalid_argument(
          "two-slot deferred ACK requires a graph-only TP4 all-reduce "
          "session with a supported graph payload capacity");
    }
    if (tp4_graph_kernel_strategy_is_graph_only(
            options_.graph_kernel_strategy)) {
      if (!submit_cpu_set ||
          !graph_capacity_supported(options_.payload_bytes,
                                    options_.bytes_per_row)) {
        throw std::invalid_argument(
            std::string(tp4_graph_kernel_strategy_name(
                            options_.graph_kernel_strategy)) +
            " graph TP4 requires a graph-only session with exact Q1 "
            "through Q512 capacity");
      }
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
    layout_ = make_exchange_buffer_layout(options_.payload_bytes);
    if (options_.schedule == Tp4AllreduceSchedule::kDualPortStriped) {
      striped_layout_ =
          make_tp4_striped_endpoint_layout(options_.payload_bytes);
      arena_bytes_ = striped_layout_.total_bytes;
    } else {
      arena_bytes_ =
          tp4_payload_arena_bytes(layout_.total_bytes, options_.protocol);
    }

    channel0_.emplace(
        open_channel(plan0, options_.peer0, options_.control_port0));
    buffer0_ =
        MemoryBuffer::allocate(MemoryKind::kCudaMapped, arena_bytes_);
    endpoint0_ = std::make_unique<VerbsEndpoint>(
        options_.device0, 1, options_.gid0, *buffer0_);
    exchange_and_connect_endpoint(*channel0_, *endpoint0_,
                                  options_.protocol, options_.schedule,
                                  submit_cpu_set, options_.elements_per_row,
                                  options_.bytes_per_row,
                                  graph_direct_doorbell_);

    channel1_.emplace(
        open_channel(plan1, options_.peer1, options_.control_port1));
    buffer1_ =
        MemoryBuffer::allocate(MemoryKind::kCudaMapped, arena_bytes_);
    endpoint1_ = std::make_unique<VerbsEndpoint>(
        options_.device1, 1, options_.gid1, *buffer1_);
    exchange_and_connect_endpoint(*channel1_, *endpoint1_,
                                  options_.protocol, options_.schedule,
                                  submit_cpu_set, options_.elements_per_row,
                                  options_.bytes_per_row,
                                  graph_direct_doorbell_);

    if (graph_capacity_supported(options_.payload_bytes,
                                 options_.bytes_per_row)) {
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
        options_.payload_bytes, options_.bytes_per_row,
        buffer0_->device_data(), layout_,
        buffer1_->device_data(), layout_, options_.protocol,
        options_.graph_kernel_strategy, options_.schedule,
        graph_direct_doorbell_);
    channel0_->barrier();
    channel1_->barrier();
    start_progress_thread();
  }

  ~Impl() {
    // Graph kernels publish work only when replay reaches them. Keep the
    // verbs progress thread alive while draining the caller stream so a
    // queued replay cannot strand its GPU kernel waiting for RDMA.
    for (cudaStream_t stream : graph_capture_streams_) {
      const auto result = cudaStreamSynchronize(stream);
      if (result != cudaSuccess) {
        fatal_async_failure(cudaGetErrorString(result));
      }
    }
    if (graph_capture_streams_.empty() && caller_stream_set_) {
      const auto result = cudaStreamSynchronize(
          static_cast<cudaStream_t>(caller_stream_));
      if (result != cudaSuccess) {
        fatal_async_failure(cudaGetErrorString(result));
      }
    }
    if (tp4_protocol_uses_deferred_ack(options_.protocol)) {
      try {
        const std::uint64_t published =
            tp4_graph_command_published(graph_commands_host_);
        wait_for_sequence(
            &graph_commands_host_->consumer.completed_sequence, published,
            "graph TP4 teardown completion", true);
      } catch (const std::exception& error) {
        fatal_async_failure(error.what());
      }
    }
    {
      std::lock_guard<std::mutex> lock(submission_mutex_);
      stopping_ = true;
      health_stopping_.store(true, std::memory_order_release);
      stop_requested_.store(true, std::memory_order_release);
    }
    progress_cv_.notify_one();
    completion_cv_.notify_all();
    if (progress_thread_.joinable()) {
      progress_thread_.join();
    }
    if (tp4_protocol_uses_deferred_ack(options_.protocol)) {
      try {
        drain_deferred_credits();
      } catch (const std::exception& error) {
        fatal_async_failure(error.what());
      } catch (...) {
        fatal_async_failure("unknown deferred-credit drain failure");
      }
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
          health_poisoned_.store(true, std::memory_order_release);
          health_failing_sequence_.store(sequence_ + 1U,
                                         std::memory_order_release);
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
        health_poisoned_.store(true, std::memory_order_release);
        health_failing_sequence_.store(next_sequence,
                                       std::memory_order_release);
        completion_cv_.notify_all();
        throw;
      }
      sequence_ = next_sequence;
      health_submitted_sequence_.store(next_sequence,
                                       std::memory_order_release);
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
        tp4_graph_payload_bytes(q, options_.bytes_per_row);
    if (!tp4_graph_command_layout_valid(
            q, active_payload_bytes, options_.bytes_per_row,
            kTp4GraphAllreduceMaximumQ) ||
        active_payload_bytes > options_.payload_bytes ||
        graph_commands_host_ == nullptr) {
      throw std::invalid_argument(
          "graph TP4 all-reduce requires the configured BF16 [q, width], "
          "q in [1, 512], "
          "within session capacity");
    }
    if (!graph_host_native_atomics_supported_) {
      throw std::runtime_error(
          "graph TP4 requires host-native GPU atomics");
    }

    const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
    cudaStreamCaptureStatus capture_status{};
    unsigned long long capture_id{};
    check_cuda(
        cudaStreamGetCaptureInfo(caller_stream, &capture_status,
                                 &capture_id),
        "cudaStreamGetCaptureInfo graph TP4");
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
      const auto capture_record = std::find_if(
          graph_capture_records_.begin(), graph_capture_records_.end(),
          [capture_id](const auto& record) {
            return record.first == capture_id;
          });
      if (capture_record != graph_capture_records_.end() &&
          capture_record->second != caller_stream) {
        throw std::invalid_argument(
            "graph TP4 capture record reused a different CUDA stream");
      }
      if (capture_record == graph_capture_records_.end()) {
        graph_capture_records_.emplace_back(capture_id, caller_stream);
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
      }
      if (std::find(graph_capture_streams_.begin(),
                    graph_capture_streams_.end(), caller_stream) ==
          graph_capture_streams_.end()) {
        graph_capture_streams_.push_back(caller_stream);
      }
    }

    try {
      worker_->enqueue_graph(input, output, q, cuda_stream,
                             graph_commands_device_, graph_trace_);
    } catch (...) {
      std::lock_guard<std::mutex> lock(submission_mutex_);
      if (graph_capture_nodes_ == 0) {
        graph_capture_configured_ = false;
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
        tp4_protocol_uses_deferred_ack(options_.protocol),
        graph_direct_doorbell_,
        options_.graph_submit_cpu.has_value()
            ? static_cast<int>(*options_.graph_submit_cpu)
            : -1,
        options_.graph_progress_cpu.has_value()
            ? static_cast<int>(*options_.graph_progress_cpu)
            : -1};
  }

  Tp4HealthStatus health_status() const noexcept {
    const bool progress_running =
        progress_thread_running_.load(std::memory_order_acquire);
    const bool poisoned = health_poisoned_.load(std::memory_order_acquire);
    const bool stopping = health_stopping_.load(std::memory_order_acquire);
    return Tp4HealthStatus{
        !poisoned && !stopping && progress_running,
        poisoned,
        progress_running,
        stopping,
        health_submitted_sequence_.load(std::memory_order_acquire),
        health_completed_sequence_.load(std::memory_order_acquire),
        health_failing_sequence_.load(std::memory_order_acquire),
        poisoned ? 1 : 0};
  }

 private:
  struct Submission {
    std::uint64_t sequence;
    bool trace;
  };

  void drain_striped_deferred_credits() {
    if (last_credit_work_id0_ != 0) {
      endpoint0_->wait_for_send(last_credit_work_id0_);
    }
    if (last_credit_work_id1_ != 0) {
      endpoint1_->wait_for_send(last_credit_work_id1_);
    }

    for (std::size_t generation = 0;
         generation < kTp4StripedGenerationCount; ++generation) {
      const std::uint64_t expected0 = tp4_latest_slot_sequence(
          last_credit_sequence0_, generation, options_.protocol);
      const std::uint64_t expected1 = tp4_latest_slot_sequence(
          last_credit_sequence1_, generation, options_.protocol);
      for (const Tp4TensorStripe stripe :
           {Tp4TensorStripe::kLowerHalf,
            Tp4TensorStripe::kUpperHalf}) {
        if (expected0 != 0) {
          wait_for_sequence(
              &striped_lane_control(*buffer0_, striped_layout_,
                                    generation, stripe)
                   ->acknowledgement_sequence,
              expected0, "endpoint-0 striped deferred-credit retirement",
              true);
        }
        if (expected1 != 0) {
          wait_for_sequence(
              &striped_lane_control(*buffer1_, striped_layout_,
                                    generation, stripe)
                   ->acknowledgement_sequence,
              expected1, "endpoint-1 striped deferred-credit retirement",
              true);
        }
      }
    }
  }

  void drain_deferred_credits() {
    if (options_.schedule == Tp4AllreduceSchedule::kDualPortStriped) {
      drain_striped_deferred_credits();
      return;
    }
    if (last_credit_work_id0_ != 0) {
      endpoint0_->wait_for_send(last_credit_work_id0_);
    }
    if (last_credit_work_id1_ != 0) {
      endpoint1_->wait_for_send(last_credit_work_id1_);
    }

    const std::size_t slots = tp4_payload_slot_count(options_.protocol);
    for (std::size_t slot = 0; slot < slots; ++slot) {
      const std::uint64_t expected0 = tp4_latest_slot_sequence(
          last_credit_sequence0_, slot, options_.protocol);
      const std::uint64_t expected1 = tp4_latest_slot_sequence(
          last_credit_sequence1_, slot, options_.protocol);
      if (expected0 != 0) {
        wait_for_sequence(
            &slot_control(*buffer0_, layout_, slot)
                 ->acknowledgement_sequence,
            expected0, "round-0 deferred-credit retirement", true);
      }
      if (expected1 != 0) {
        wait_for_sequence(
            &slot_control(*buffer1_, layout_, slot)
                 ->acknowledgement_sequence,
            expected1, "round-1 deferred-credit retirement", true);
      }
    }
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
          progress_thread_running_.store(true, std::memory_order_release);
          progress_loop();
          progress_thread_running_.store(false, std::memory_order_release);
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
        health_completed_sequence_.store(submission.sequence,
                                         std::memory_order_release);
      }
      completion_cv_.notify_all();
    }
  }

  void progress_graph_commands() noexcept {
    if (graph_direct_doorbell_) {
      progress_direct_graph_doorbells();
      return;
    }
    std::uint32_t poll_misses{};
    while (!stop_requested_.load(std::memory_order_acquire)) {
      if (tp4_graph_command_overflow(graph_commands_host_) != 0) {
        fatal_async_failure("CUDA Graph command ring overflow");
      }

      Tp4GraphCommand command{};
      const std::uint64_t expected = graph_consumed_sequence_ + 1;
      if (!tp4_graph_command_try_consume_layout(
              graph_commands_host_, expected, options_.bytes_per_row,
              kTp4GraphAllreduceMaximumQ, &command)) {
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

  void progress_direct_graph_doorbells() noexcept {
    std::uint32_t poll_misses{};
    while (!stop_requested_.load(std::memory_order_acquire)) {
      if (tp4_graph_command_overflow(graph_commands_host_) != 0) {
        fatal_async_failure("direct-doorbell graph TP4 overflow");
      }

      const std::uint64_t expected = graph_consumed_sequence_ + 1;
      const std::size_t slot =
          tp4_payload_slot_index(expected, options_.protocol);
      DoorbellControl* const control0 =
          slot_control(*buffer0_, layout_, slot);
      const std::uint64_t doorbell_token =
          load_sequence(&control0->producer_sequence);
      const std::uint64_t observed_sequence =
          doorbell_token >> kTp4GraphDoorbellQBits;
      if (observed_sequence < expected) {
        adaptive_graph_poll_pause(poll_misses);
        continue;
      }
      poll_misses = 0;
      if (observed_sequence != expected) {
        fatal_async_failure(
            "direct-doorbell graph TP4 skipped a slot generation");
      }
      const std::uint32_t q = static_cast<std::uint32_t>(
          doorbell_token & kTp4GraphDoorbellQMask);
      if (!tp4_graph_doorbell_token_valid(expected, q)) {
        fatal_async_failure("invalid direct-doorbell graph TP4 token");
      }
      const std::uint32_t payload_bytes =
          tp4_graph_payload_bytes(q, options_.bytes_per_row);
      if (payload_bytes == 0 || payload_bytes > options_.payload_bytes) {
        fatal_async_failure(
            "direct-doorbell graph TP4 payload exceeds session capacity");
      }

      graph_consumed_sequence_ = expected;
      store_sequence(
          &graph_commands_host_->producer.claimed_sequence, expected);
      store_sequence(
          &graph_commands_host_->producer.published_sequence, expected);
      store_sequence(
          &graph_commands_host_->consumer.consumed_sequence, expected);
      try {
        progress(expected, graph_trace_, payload_bytes, doorbell_token,
                 true);
      } catch (const std::exception& error) {
        fatal_async_failure(error.what());
      } catch (...) {
        fatal_async_failure("unknown direct-doorbell graph TP4 error");
      }
      tp4_graph_command_complete(graph_commands_host_, expected);
    }
  }

  void progress(std::uint64_t sequence, bool trace) {
    progress(sequence, trace, options_.payload_bytes, sequence, false,
             eager_protocol_timeout_);
  }

  void progress_dual_port_striped(
      std::uint64_t sequence, bool trace, std::size_t payload_bytes,
      std::uint64_t doorbell_token, bool require_exact_doorbell,
      std::chrono::seconds timeout) {
    if (!require_exact_doorbell || payload_bytes == 0 ||
        payload_bytes % 4U != 0 ||
        payload_bytes / 2U > striped_layout_.stripe_bytes) {
      throw std::logic_error(
          "invalid dual-port striped graph progress request");
    }

    const std::size_t generation =
        tp4_striped_generation_slot(sequence);
    const std::size_t stripe_bytes = payload_bytes / 2U;
    const Tp4StripedLaneRegion phase1_endpoint0_region =
        tp4_striped_lane_region(
            striped_layout_, generation,
            Tp4TensorStripe::kLowerHalf);
    const Tp4StripedLaneRegion phase1_endpoint1_region =
        tp4_striped_lane_region(
            striped_layout_, generation,
            Tp4TensorStripe::kUpperHalf);
    const Tp4StripedLaneRegion phase2_endpoint0_region =
        tp4_striped_lane_region(
            striped_layout_, generation,
            Tp4TensorStripe::kUpperHalf);
    const Tp4StripedLaneRegion phase2_endpoint1_region =
        tp4_striped_lane_region(
            striped_layout_, generation,
            Tp4TensorStripe::kLowerHalf);
    DoorbellControl* const phase1_endpoint0_control =
        striped_lane_control(*buffer0_, striped_layout_, generation,
                             Tp4TensorStripe::kLowerHalf);
    DoorbellControl* const phase1_endpoint1_control =
        striped_lane_control(*buffer1_, striped_layout_, generation,
                             Tp4TensorStripe::kUpperHalf);
    DoorbellControl* const phase2_endpoint0_control =
        striped_lane_control(*buffer0_, striped_layout_, generation,
                             Tp4TensorStripe::kUpperHalf);
    DoorbellControl* const phase2_endpoint1_control =
        striped_lane_control(*buffer1_, striped_layout_, generation,
                             Tp4TensorStripe::kLowerHalf);

    wait_for_sequence(
        &phase1_endpoint0_control->producer_sequence, doorbell_token,
        "GPU striped phase-1 endpoint-0 staging", true, timeout);
    wait_for_sequence(
        &phase1_endpoint1_control->producer_sequence, doorbell_token,
        "GPU striped phase-1 endpoint-1 staging", true, timeout);

    const std::uint64_t phase1_work_id =
        detail::tp4_striped_work_id(
            sequence, detail::Tp4StripedWorkEvent::kPhase1Doorbell);
    post_striped_transfer(*endpoint0_, phase1_endpoint0_region,
                          stripe_bytes, doorbell_token, phase1_work_id);
    post_striped_transfer(*endpoint1_, phase1_endpoint1_region,
                          stripe_bytes, doorbell_token, phase1_work_id);
    endpoint0_->wait_for_send_through(phase1_work_id);
    endpoint1_->wait_for_send_through(phase1_work_id);
    if (trace) {
      std::fprintf(
          stderr,
          "SESSION rank=%u state=striped_phase1_sent sequence=%llu "
          "token=%llu bytes=%zu generation=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), stripe_bytes,
          generation);
    }

    wait_for_sequence(
        &phase1_endpoint0_control->consumer_sequence, doorbell_token,
        "GPU striped phase-1 endpoint-0 reduction", true, timeout);
    wait_for_sequence(
        &phase1_endpoint1_control->consumer_sequence, doorbell_token,
        "GPU striped phase-1 endpoint-1 reduction", true, timeout);
    const std::uint64_t phase1_credit_work_id =
        detail::tp4_striped_work_id(
            sequence, detail::Tp4StripedWorkEvent::kPhase1Credit);
    post_striped_credit(*endpoint0_, *phase1_endpoint0_control,
                         phase1_endpoint0_region, sequence,
                         phase1_credit_work_id);
    post_striped_credit(*endpoint1_, *phase1_endpoint1_control,
                         phase1_endpoint1_region, sequence,
                         phase1_credit_work_id);

    // Phase-1 consumer publication is the system-fenced GPU-to-host handoff
    // for both phase-2 send lanes. The host owns phase-2 producer publication.
    store_sequence(&phase2_endpoint0_control->producer_sequence,
                   doorbell_token);
    store_sequence(&phase2_endpoint1_control->producer_sequence,
                   doorbell_token);
    const std::uint64_t phase2_work_id =
        detail::tp4_striped_work_id(
            sequence, detail::Tp4StripedWorkEvent::kPhase2Doorbell);
    post_striped_transfer(*endpoint0_, phase2_endpoint0_region,
                          stripe_bytes, doorbell_token, phase2_work_id);
    post_striped_transfer(*endpoint1_, phase2_endpoint1_region,
                          stripe_bytes, doorbell_token, phase2_work_id);
    endpoint0_->wait_for_send_through(phase2_work_id);
    endpoint1_->wait_for_send_through(phase2_work_id);
    if (trace) {
      std::fprintf(
          stderr,
          "SESSION rank=%u state=striped_phase2_sent sequence=%llu "
          "token=%llu bytes=%zu generation=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), stripe_bytes,
          generation);
    }

    wait_for_sequence(
        &phase2_endpoint0_control->consumer_sequence, doorbell_token,
        "GPU striped phase-2 endpoint-0 reduction", true, timeout);
    wait_for_sequence(
        &phase2_endpoint1_control->consumer_sequence, doorbell_token,
        "GPU striped phase-2 endpoint-1 reduction", true, timeout);
    const std::uint64_t phase2_credit_work_id =
        detail::tp4_striped_work_id(
            sequence, detail::Tp4StripedWorkEvent::kPhase2Credit);
    post_striped_credit(*endpoint0_, *phase2_endpoint0_control,
                         phase2_endpoint0_region, sequence,
                         phase2_credit_work_id);
    post_striped_credit(*endpoint1_, *phase2_endpoint1_control,
                         phase2_endpoint1_region, sequence,
                         phase2_credit_work_id);
    last_credit_work_id0_ = phase2_credit_work_id;
    last_credit_work_id1_ = phase2_credit_work_id;
    last_credit_sequence0_ = sequence;
    last_credit_sequence1_ = sequence;
    if (trace) {
      std::fprintf(stderr,
                   "SESSION rank=%u state=striped_output_ready\n",
                   options_.rank);
    }
  }

  void progress(std::uint64_t sequence, bool trace,
                std::size_t payload_bytes,
                std::uint64_t doorbell_token,
                bool require_exact_doorbell,
                std::chrono::seconds timeout =
                    kGraphProtocolTimeout) {
    if (options_.schedule == Tp4AllreduceSchedule::kDualPortStriped) {
      progress_dual_port_striped(
          sequence, trace, payload_bytes, doorbell_token,
          require_exact_doorbell, timeout);
      return;
    }
    const std::size_t slot =
        tp4_payload_slot_index(sequence, options_.protocol);
    const std::size_t slot_offset = slot * layout_.total_bytes;
    DoorbellControl* const control0 =
        slot_control(*buffer0_, layout_, slot);
    DoorbellControl* const control1 =
        slot_control(*buffer1_, layout_, slot);
    wait_for_sequence(&control0->producer_sequence, doorbell_token,
                      "GPU tensor input staging",
                      require_exact_doorbell, timeout);
    if (trace) {
      std::fprintf(
          stderr,
          "SESSION rank=%u state=round0 sequence=%llu token=%llu "
          "bytes=%zu slot=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), payload_bytes,
          slot);
    }
    exchange_round(*endpoint0_, *control0, layout_, payload_bytes,
                   slot_offset, sequence, doorbell_token, options_.rank, 0,
                   trace, options_.protocol,
                   require_exact_doorbell, timeout);
    if (tp4_protocol_uses_deferred_ack(options_.protocol)) {
      last_credit_work_id0_ = doorbell_token;
      last_credit_sequence0_ = sequence;
    }
    if (trace) {
      std::fprintf(
          stderr,
          "SESSION rank=%u state=round1 sequence=%llu token=%llu "
          "bytes=%zu slot=%zu\n",
          options_.rank, static_cast<unsigned long long>(sequence),
          static_cast<unsigned long long>(doorbell_token), payload_bytes,
          slot);
    }
    exchange_round(*endpoint1_, *control1, layout_, payload_bytes,
                   slot_offset, sequence, doorbell_token, options_.rank, 1,
                   trace, options_.protocol,
                   require_exact_doorbell, timeout);
    if (tp4_protocol_uses_deferred_ack(options_.protocol)) {
      last_credit_work_id1_ = doorbell_token;
      last_credit_sequence1_ = sequence;
    }
    if (trace) {
      std::fprintf(stderr, "SESSION rank=%u state=output_ready\n",
                   options_.rank);
    }
  }

  Tp4AllreduceOptions options_;
  const bool graph_direct_doorbell_{};
  ExchangeBufferLayout layout_{};
  Tp4StripedEndpointLayout striped_layout_{};
  std::size_t arena_bytes_{};
  std::optional<ControlChannel> channel0_;
  std::optional<ControlChannel> channel1_;
  std::unique_ptr<MemoryBuffer> buffer0_;
  std::unique_ptr<MemoryBuffer> buffer1_;
  std::unique_ptr<VerbsEndpoint> endpoint0_;
  std::unique_ptr<VerbsEndpoint> endpoint1_;
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
  std::atomic<bool> progress_thread_running_{false};
  std::atomic<bool> health_stopping_{false};
  std::atomic<bool> health_poisoned_{false};
  std::atomic<std::uint64_t> health_submitted_sequence_{};
  std::atomic<std::uint64_t> health_completed_sequence_{};
  std::atomic<std::uint64_t> health_failing_sequence_{};
  const std::uint64_t max_inflight_;
  std::chrono::seconds eager_protocol_timeout_{
      kGraphProtocolTimeout};
  std::atomic<bool> graph_polling_enabled_{false};
  std::atomic<bool> stop_requested_{false};
  std::uint64_t sequence_{};
  std::uint64_t completed_sequence_{};
  std::uint64_t graph_consumed_sequence_{};
  std::uint64_t last_credit_work_id0_{};
  std::uint64_t last_credit_work_id1_{};
  std::uint64_t last_credit_sequence0_{};
  std::uint64_t last_credit_sequence1_{};
  void* caller_stream_{};
  bool caller_stream_set_{};
  std::vector<std::pair<unsigned long long, cudaStream_t>>
      graph_capture_records_;
  std::vector<cudaStream_t> graph_capture_streams_;
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

Tp4HealthStatus Tp4AllreduceSession::health_status() const noexcept {
  return impl_->health_status();
}

}  // namespace spark_transport
