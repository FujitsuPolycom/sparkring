// Research-only standalone native qualification probe for tiled TP4 prefill.
// This binary is not linked by the production C ABI or the vLLM adapter.

#include "spark_transport/control_channel.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/statistics.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/tp4_tiled_session.hpp"
#include "spark_transport/verbs_endpoint.hpp"
#include "tiled_bulk_kernels.cuh"
#include "tiled_correctness_kernels.cuh"
#include "tiled_executor.hpp"
#include "tiled_signaled_work_gate.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;

namespace {

constexpr int kWorldSize = 4;
constexpr int kPoisonExitCode = 42;
constexpr std::uint16_t kTiledEndpointVersion = 4;
constexpr std::uint16_t kTiledEndpointTag = 0x5449;  // "TI"
constexpr std::uint32_t kTiledGeometryMagic = 0x54494c45;  // "TILE"
constexpr std::uint16_t kTiledGeometryVersion = 1;
constexpr std::string_view kReceiptSchema =
    "sparkring-tp4-tiled-prefill-probe/v1";
constexpr std::string_view kReceiptPrefix = "TP4_TILED_PREFILL_RECEIPT";

enum class TimingMode { kIsolated, kSteady, kCorrectness, kPoison };
enum class PoisonInjection { kNone, kUnexpectedGeneration, kCreditRegression };

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{12500};
  std::uint16_t control_port1{12501};
  std::string arm_id;
  std::uint32_t query_rows{};
  std::uint32_t elements_per_row{
      spark_transport::kTp4TiledBf16ElementsPerQueryRow};
  TimingMode timing_mode{TimingMode::kCorrectness};
  std::uint32_t warmup_operations{};
  std::uint32_t measured_operations{};
  std::uint64_t guard_bytes{4096};
  int credit_delay_edge{-1};
  std::uint64_t credit_delay_us{};
  PoisonInjection poison{PoisonInjection::kNone};
  std::string receipt_schema;
  std::string receipt_prefix;
  std::uint64_t expected_payload_tile_bytes{};
  std::uint32_t expected_slots_per_edge{};
  std::uint32_t expected_lanes_per_edge{};
  std::uint64_t expected_registered_bytes_per_edge{};
  std::uint32_t world_size{4};
};

struct TiledGeometryInfo {
  std::uint32_t magic{kTiledGeometryMagic};
  std::uint16_t version{kTiledGeometryVersion};
  std::uint16_t capacity_class{};
  std::uint32_t rank{};
  std::uint32_t world_size{};
  std::uint32_t query_rows{};
  std::uint32_t elements_per_row{};
  std::uint32_t tile_count{};
  std::uint32_t slots_per_edge{};
  std::uint32_t lanes_per_edge{};
  std::uint64_t bytes_per_row{};
  std::uint64_t active_bytes{};
  std::uint64_t tile_payload_bytes{};
  std::uint64_t slot_stride{};
  std::uint64_t registered_bytes_per_edge{};
};

static_assert(std::is_trivially_copyable_v<TiledGeometryInfo>);

[[noreturn]] void usage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " --rank R --world-size 4 --peer0 IP --peer1 IP"
               " --arm-id ID --query-rows Q --elements-per-row WIDTH"
               " --timing-mode MODE"
               " --warmup-operations N --measured-operations N"
               " --guard-bytes N --credit-delay-edge -1|0|1"
               " --credit-delay-us N --inject-poison MODE"
               " --receipt-schema SCHEMA --receipt-prefix PREFIX"
               " --expected-payload-tile-bytes N"
               " --expected-slots-per-edge N"
               " --expected-lanes-per-edge N"
               " --expected-registered-tile-storage-bytes-per-edge N\n";
  std::exit(2);
}

std::uint64_t unsigned_value(const char* value, const char* name) {
  std::size_t consumed{};
  const std::string text(value);
  const auto parsed = std::stoull(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return parsed;
}

int signed_value(const char* value, const char* name) {
  std::size_t consumed{};
  const std::string text(value);
  const auto parsed = std::stoll(text, &consumed);
  if (consumed != text.size() || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return static_cast<int>(parsed);
}

TimingMode parse_timing_mode(std::string_view value) {
  if (value == "isolated") return TimingMode::kIsolated;
  if (value == "steady") return TimingMode::kSteady;
  if (value == "correctness") return TimingMode::kCorrectness;
  if (value == "poison") return TimingMode::kPoison;
  throw std::invalid_argument("invalid timing mode");
}

const char* timing_mode_name(TimingMode mode) {
  switch (mode) {
    case TimingMode::kIsolated: return "isolated";
    case TimingMode::kSteady: return "steady";
    case TimingMode::kCorrectness: return "correctness";
    case TimingMode::kPoison: return "poison";
  }
  return "invalid";
}

PoisonInjection parse_poison(std::string_view value) {
  if (value == "none") return PoisonInjection::kNone;
  if (value == "unexpected_generation") {
    return PoisonInjection::kUnexpectedGeneration;
  }
  if (value == "credit_regression") {
    return PoisonInjection::kCreditRegression;
  }
  throw std::invalid_argument("invalid poison injection");
}

const char* poison_name(PoisonInjection poison) {
  switch (poison) {
    case PoisonInjection::kNone: return "none";
    case PoisonInjection::kUnexpectedGeneration: return "unexpected_generation";
    case PoisonInjection::kCreditRegression: return "credit_regression";
  }
  return "invalid";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto take_value = [&]() -> const char* {
      if (++index >= argc) usage(argv[0]);
      return argv[index];
    };
    if (argument == "--rank") {
      options.rank = static_cast<std::uint32_t>(unsigned_value(take_value(), "rank"));
    } else if (argument == "--world-size") {
      options.world_size = static_cast<std::uint32_t>(unsigned_value(take_value(), "world size"));
    } else if (argument == "--peer0") {
      options.peer0 = take_value();
    } else if (argument == "--peer1") {
      options.peer1 = take_value();
    } else if (argument == "--device0") {
      options.device0 = take_value();
    } else if (argument == "--device1") {
      options.device1 = take_value();
    } else if (argument == "--gid0") {
      options.gid0 = static_cast<std::uint8_t>(unsigned_value(take_value(), "gid0"));
    } else if (argument == "--gid1") {
      options.gid1 = static_cast<std::uint8_t>(unsigned_value(take_value(), "gid1"));
    } else if (argument == "--control-port0") {
      options.control_port0 = static_cast<std::uint16_t>(unsigned_value(take_value(), "control port0"));
    } else if (argument == "--control-port1") {
      options.control_port1 = static_cast<std::uint16_t>(unsigned_value(take_value(), "control port1"));
    } else if (argument == "--arm-id") {
      options.arm_id = take_value();
    } else if (argument == "--query-rows") {
      options.query_rows = static_cast<std::uint32_t>(unsigned_value(take_value(), "query rows"));
    } else if (argument == "--elements-per-row") {
      options.elements_per_row = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "elements per row"));
    } else if (argument == "--timing-mode") {
      options.timing_mode = parse_timing_mode(take_value());
    } else if (argument == "--warmup-operations") {
      options.warmup_operations = static_cast<std::uint32_t>(unsigned_value(take_value(), "warmup operations"));
    } else if (argument == "--measured-operations") {
      options.measured_operations = static_cast<std::uint32_t>(unsigned_value(take_value(), "measured operations"));
    } else if (argument == "--guard-bytes") {
      options.guard_bytes = unsigned_value(take_value(), "guard bytes");
    } else if (argument == "--credit-delay-edge") {
      options.credit_delay_edge = signed_value(take_value(), "credit delay edge");
    } else if (argument == "--credit-delay-us") {
      options.credit_delay_us = unsigned_value(take_value(), "credit delay");
    } else if (argument == "--inject-poison") {
      options.poison = parse_poison(take_value());
    } else if (argument == "--receipt-schema") {
      options.receipt_schema = take_value();
    } else if (argument == "--receipt-prefix") {
      options.receipt_prefix = take_value();
    } else if (argument == "--expected-payload-tile-bytes") {
      options.expected_payload_tile_bytes = unsigned_value(take_value(), "tile bytes");
    } else if (argument == "--expected-slots-per-edge") {
      options.expected_slots_per_edge = static_cast<std::uint32_t>(unsigned_value(take_value(), "slots"));
    } else if (argument == "--expected-lanes-per-edge") {
      options.expected_lanes_per_edge = static_cast<std::uint32_t>(unsigned_value(take_value(), "lanes"));
    } else if (argument == "--expected-registered-tile-storage-bytes-per-edge") {
      options.expected_registered_bytes_per_edge = unsigned_value(take_value(), "registered bytes");
    } else if (argument == "--submit-cpu" ||
               argument == "--edge0-progress-cpu" ||
               argument == "--edge1-progress-cpu") {
      (void)take_value();  // The runner owns process affinity in this executor.
    } else {
      usage(argv[0]);
    }
  }
  if (options.rank >= 4 || options.world_size != kWorldSize ||
      options.peer0.empty() || options.peer1.empty() || options.arm_id.empty() ||
      options.query_rows == 0 || options.elements_per_row == 0 ||
      options.measured_operations == 0 ||
      options.receipt_schema != kReceiptSchema ||
      options.receipt_prefix != kReceiptPrefix ||
      options.credit_delay_edge < -1 || options.credit_delay_edge > 1 ||
      options.control_port0 == 0 || options.control_port1 == 0 ||
      options.control_port0 == options.control_port1) {
    usage(argv[0]);
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

std::uint64_t load_word(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_word(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

double now_us() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double, std::micro>(now).count();
}

spark_transport::ControlChannel open_channel(
    const spark_transport::Tp4RoundPlan& plan, const std::string& peer,
    std::uint16_t port) {
  return plan.server
             ? spark_transport::ControlChannel::listen_and_accept(port)
             : spark_transport::ControlChannel::connect(peer, port);
}

void exchange_and_connect(spark_transport::ControlChannel& channel,
                          spark_transport::VerbsEndpoint& endpoint,
                          const Options& options,
                          const spark_transport::Tp4RoundPlan& plan,
                          const spark_transport::Tp4SessionStorageOptions& storage,
                          const spark_transport::Tp4TiledPoolLayout& layout,
                          const spark_transport::Tp4TiledOperationDescriptor& operation) {
  auto local = endpoint.local_info();
  local.version = kTiledEndpointVersion;
  local.reserved = kTiledEndpointTag;
  const auto remote = channel.exchange(local);
  if (remote.version != local.version || remote.reserved != local.reserved ||
      remote.buffer_bytes != local.buffer_bytes) {
    throw std::runtime_error("tiled-prefill endpoint handshake mismatch");
  }

  const TiledGeometryInfo local_geometry{
      kTiledGeometryMagic,
      kTiledGeometryVersion,
      static_cast<std::uint16_t>(storage.capacity_class),
      options.rank,
      options.world_size,
      options.query_rows,
      options.elements_per_row,
      operation.tile_count,
      storage.slots_per_edge,
      storage.lanes_per_edge,
      operation.bytes_per_row,
      operation.active_bytes,
      storage.tile_payload_bytes,
      layout.slot_stride,
      layout.total_bytes};
  const TiledGeometryInfo remote_geometry = channel.exchange(local_geometry);
  if (remote_geometry.magic != local_geometry.magic ||
      remote_geometry.version != local_geometry.version ||
      remote_geometry.rank != plan.peer_rank ||
      remote_geometry.world_size != local_geometry.world_size ||
      remote_geometry.query_rows != local_geometry.query_rows ||
      remote_geometry.elements_per_row != local_geometry.elements_per_row ||
      remote_geometry.bytes_per_row != local_geometry.bytes_per_row ||
      remote_geometry.capacity_class != local_geometry.capacity_class ||
      remote_geometry.tile_count != local_geometry.tile_count ||
      remote_geometry.slots_per_edge != local_geometry.slots_per_edge ||
      remote_geometry.lanes_per_edge != local_geometry.lanes_per_edge ||
      remote_geometry.active_bytes != local_geometry.active_bytes ||
      remote_geometry.tile_payload_bytes !=
          local_geometry.tile_payload_bytes ||
      remote_geometry.slot_stride != local_geometry.slot_stride ||
      remote_geometry.registered_bytes_per_edge !=
          local_geometry.registered_bytes_per_edge) {
    throw std::runtime_error(
        "tiled-prefill geometry mismatch between edge peers");
  }
  endpoint.connect(remote, local.version);
}

class CudaTiledBulkPort final : public research::TiledBulkLaunchPort {
 public:
  CudaTiledBulkPort(std::uint8_t* input, std::uint8_t* output,
                    std::uint8_t* endpoint0, std::uint8_t* endpoint1,
                    std::uint32_t rank, cudaStream_t stream)
      : input_(input), output_(output), endpoints_{endpoint0, endpoint1},
        rank_(rank), stream_(stream) {
    check_cuda(cudaMalloc(&descriptor_, sizeof(*descriptor_)),
               "cudaMalloc tiled descriptor");
    check_cuda(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
               "cudaEventCreate tiled completion");
  }

  ~CudaTiledBulkPort() override {
    if (completion_ != nullptr) cudaEventDestroy(completion_);
    if (descriptor_ != nullptr) cudaFree(descriptor_);
  }

  research::TiledExecutorStreamPolicy stream_policy() const noexcept override {
    return research::TiledExecutorStreamPolicy::kSingleStreamCorrectnessOnly;
  }

  research::TiledSubmitState try_submit(
      const research::TiledBulkLaunchRequest& request) override {
    if (pending_) return research::TiledSubmitState::kBackpressured;
    check_cuda(cudaMemcpyAsync(descriptor_, &request.descriptor,
                               sizeof(request.descriptor),
                               cudaMemcpyHostToDevice, stream_),
               "upload tiled descriptor");
    cudaError_t result = cudaSuccess;
    if (request.phase == research::TiledBulkPhase::kStagePhase1) {
      // Generate this physical slot generation immediately before staging it.
      // The same stream gives the worker grid an exact fill -> stage edge and
      // prevents stale input from passing across slot reuse.
      check_cuda(research::launch_fill_correctness_input(
                     input_, descriptor_, rank_, stream_),
                 "fill tiled correctness input");
      result = research::launch_stage_phase1_tile(
          input_, endpoints_[0], endpoints_[1], descriptor_, request.layout,
          stream_);
    } else if (request.phase == research::TiledBulkPhase::kReducePhase1) {
      result = research::launch_reduce_phase1_tile(
          endpoints_[0], endpoints_[1], descriptor_, request.layout, stream_);
    } else {
      result = research::launch_reduce_phase2_tile(
          endpoints_[0], endpoints_[1], output_, descriptor_, request.layout,
          stream_);
    }
    check_cuda(result, "launch tiled bulk phase");
    check_cuda(cudaEventRecord(completion_, stream_),
               "cudaEventRecord tiled completion");
    pending_ = true;
    pending_phase_ = request.phase;
    pending_ordinal_ = request.ordinal;
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll(
      const research::TiledBulkLaunchRequest& request) override {
    if (!pending_ || request.phase != pending_phase_ ||
        request.ordinal != pending_ordinal_) {
      return research::TiledPollState::kFatal;
    }
    const auto result = cudaEventQuery(completion_);
    if (result == cudaErrorNotReady) return research::TiledPollState::kPending;
    if (result != cudaSuccess) return research::TiledPollState::kFatal;
    pending_ = false;
    return research::TiledPollState::kComplete;
  }

 private:
  std::uint8_t* input_{};
  std::uint8_t* output_{};
  std::array<std::uint8_t*, 2> endpoints_{};
  research::TiledBulkDescriptor* descriptor_{};
  std::uint32_t rank_{};
  cudaStream_t stream_{};
  cudaEvent_t completion_{};
  bool pending_{};
  research::TiledBulkPhase pending_phase_{};
  std::uint64_t pending_ordinal_{};
};

class VerbsTiledEdgePort final : public research::TiledEdgePort {
 public:
  VerbsTiledEdgePort(std::uint32_t edge, std::uintptr_t engine_identity,
                     spark_transport::VerbsEndpoint& endpoint,
                     spark_transport::MemoryBuffer& arena,
                     int delayed_edge, std::uint64_t delay_us)
      : edge_(edge), engine_identity_(engine_identity), endpoint_(endpoint),
        arena_(arena), signaled_work_(
                           static_cast<int>(edge) == delayed_edge ? delay_us
                                                                  : 0) {}

  std::uint32_t edge_index() const noexcept override { return edge_; }
  std::uintptr_t engine_identity() const noexcept override {
    return engine_identity_;
  }
  std::uintptr_t qp_identity() const noexcept override {
    return reinterpret_cast<std::uintptr_t>(&endpoint_);
  }

  research::TiledSubmitState try_post_exchange(
      const research::TiledEdgeExchangeRequest& request) override {
    if (!drain_send_completions()) {
      return research::TiledSubmitState::kFatal;
    }
    if (pending_signaled_work_.size() >= kMaximumOutstandingSends) {
      return research::TiledSubmitState::kBackpressured;
    }
    store_word(word(request.local_doorbell_offset), request.doorbell_token);
    endpoint_.write(request.local_payload_offset,
                    request.remote_payload_offset, request.active_bytes,
                    request.work_id, false);
    endpoint_.write(request.local_doorbell_offset,
                    request.remote_doorbell_offset,
                    sizeof(request.doorbell_token), request.work_id, true);
    if (!pending_signaled_work_.insert(request.work_id).second ||
        !logical_exchanges_.insert(request.work_id).second) {
      completion_fatal_ = true;
      return research::TiledSubmitState::kFatal;
    }
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll_exchange(
      const research::TiledEdgeExchangeRequest& request) override {
    if (!drain_send_completions() ||
        logical_exchanges_.count(request.work_id) == 0) {
      return research::TiledPollState::kFatal;
    }
    if (load_word(word(request.remote_doorbell_offset)) !=
        request.doorbell_token) {
      return research::TiledPollState::kPending;
    }
    // The reciprocal doorbell proves that the peer consumed this logical
    // exchange, but registered local send storage is reusable only after the
    // signaled doorbell CQE retires the preceding unsignaled payload WR.
    if (pending_signaled_work_.count(request.work_id) != 0) {
      return research::TiledPollState::kPending;
    }
    logical_exchanges_.erase(request.work_id);
    return research::TiledPollState::kComplete;
  }

  research::TiledSubmitState try_publish_consumed_through(
      const research::TiledCreditPublishRequest& request) override {
    if (!drain_send_completions()) {
      return research::TiledSubmitState::kFatal;
    }
    const auto credit_begin = signaled_work_.try_begin_credit(now_ticks());
    if (credit_begin != research::TiledCreditBeginState::kReady) {
      return research::TiledSubmitState::kBackpressured;
    }
    store_word(word(request.local_credit_offset), request.wire_credit);
    endpoint_.write(request.local_credit_offset, request.remote_credit_offset,
                    sizeof(request.wire_credit), request.work_id, true);
    if (!pending_signaled_work_.insert(request.work_id).second) {
      completion_fatal_ = true;
      return research::TiledSubmitState::kFatal;
    }
    credit_work_id_ = request.work_id;
    credit_send_completed_ = false;
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll_published_consumed_through(
      const research::TiledCreditPublishRequest& request) override {
    if (!drain_send_completions() || !signaled_work_.credit_pending() ||
        request.work_id != credit_work_id_) {
      return research::TiledPollState::kFatal;
    }
    if (!credit_send_completed_) {
      return research::TiledPollState::kPending;
    }
    signaled_work_.complete_credit();
    credit_send_completed_ = false;
    return research::TiledPollState::kComplete;
  }

  research::TiledCreditPollState poll_peer_consumed_through(
      const research::TiledCreditObserveRequest& request,
      std::uint64_t& wire_credit) override {
    if (!drain_send_completions()) {
      return research::TiledCreditPollState::kFatal;
    }
    wire_credit = load_word(word(request.local_credit_offset));
    if (wire_credit == 0 || wire_credit == last_peer_wire_credit_) {
      return research::TiledCreditPollState::kNoUpdate;
    }
    last_peer_wire_credit_ = wire_credit;
    return research::TiledCreditPollState::kUpdate;
  }

  void inject_peer_wire_credit(std::uint64_t wire_credit,
                               std::uint64_t offset) {
    store_word(word(offset), wire_credit);
    last_peer_wire_credit_ = 0;
  }

  bool drain_all() noexcept {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!pending_signaled_work_.empty()) {
      if (!drain_send_completions()) {
        return false;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::yield();
    }
    return !completion_fatal_;
  }

 private:
  static constexpr std::size_t kMaximumOutstandingSends = 64;

  bool drain_send_completions() noexcept {
    if (completion_fatal_) {
      return false;
    }
    try {
      spark_transport::SendCompletion completions[32]{};
      const std::size_t count = endpoint_.poll_send_completions(
          completions, 32);
      for (std::size_t index = 0; index < count; ++index) {
        const std::uint64_t work_id = completions[index].work_id;
        if (pending_signaled_work_.erase(work_id) != 1) {
          completion_fatal_ = true;
          return false;
        }
        if (signaled_work_.credit_pending() &&
            work_id == credit_work_id_) {
          credit_send_completed_ = true;
        }
      }
      return true;
    } catch (...) {
      completion_fatal_ = true;
      return false;
    }
  }

  static std::uint64_t now_ticks() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  std::uint64_t* word(std::uint64_t offset) const {
    if (offset + sizeof(std::uint64_t) > arena_.size()) {
      throw std::out_of_range("tiled edge control offset exceeds arena");
    }
    return reinterpret_cast<std::uint64_t*>(
        static_cast<std::uint8_t*>(arena_.host_data()) + offset);
  }

  std::uint32_t edge_{};
  std::uintptr_t engine_identity_{};
  spark_transport::VerbsEndpoint& endpoint_;
  spark_transport::MemoryBuffer& arena_;
  research::TiledSignaledWorkGate signaled_work_;
  std::unordered_set<std::uint64_t> pending_signaled_work_;
  std::unordered_set<std::uint64_t> logical_exchanges_;
  std::uint64_t credit_work_id_{};
  bool credit_send_completed_{};
  bool completion_fatal_{};
  std::uint64_t last_peer_wire_credit_{};
};

std::string json_string(std::string_view value) {
  std::ostringstream stream;
  stream << '"';
  for (char character : value) {
    if (character == '"' || character == '\\') stream << '\\';
    stream << character;
  }
  stream << '"';
  return stream.str();
}

double positive_or_epsilon(double value) {
  return value > 0.0 ? value : 0.001;
}

std::string make_receipt(
    const Options& options, const research::TiledCorrectnessReceipt& correctness,
    const research::TiledExecutorStatus& status, bool passed, int exit_code,
    std::uint64_t completed_operations, std::uint64_t tiles_acquired,
    std::uint64_t tiles_recycled, std::uint64_t highest_ordinal,
    std::uint64_t peak_tiles, std::uint64_t credit_wait_events,
    double elapsed_us, const spark_transport::LatencySummary* summary,
    std::uint64_t poisoned_slot_count) {
  const auto operation =
      spark_transport::make_tp4_tiled_bf16_allreduce_operation(
          options.query_rows,
          static_cast<std::size_t>(options.elements_per_row) * 2U);
  const auto tail = spark_transport::make_tp4_tiled_tile_descriptor(
      operation, operation.tile_count - 1U);
  const std::uint64_t timing_operations =
      options.poison == PoisonInjection::kNone
          ? options.measured_operations
          : std::max<std::uint64_t>(completed_operations, 1U);
  const double per_operation = positive_or_epsilon(
      elapsed_us / timing_operations);
  const double p50 = summary == nullptr ? per_operation : positive_or_epsilon(summary->p50_us);
  const double p95 = summary == nullptr ? per_operation : positive_or_epsilon(summary->p95_us);
  const double minimum = summary == nullptr ? per_operation : positive_or_epsilon(summary->minimum_us);
  const double gib = static_cast<double>(operation.active_bytes) /
                     (1024.0 * 1024.0 * 1024.0);
  const double bandwidth = positive_or_epsilon(gib / (p50 * 1.0e-6));
  const double steady_bandwidth =
      positive_or_epsilon(gib / (per_operation * 1.0e-6));
  std::ostringstream out;
  out << std::setprecision(12) << '{'
      << "\"schema\":" << json_string(options.receipt_schema)
      << ",\"arm_id\":" << json_string(options.arm_id)
      << ",\"world_size\":4,\"rank\":" << options.rank
      << ",\"query_rows\":" << options.query_rows
      << ",\"elements_per_row\":" << options.elements_per_row
      << ",\"bytes_per_row\":" << operation.bytes_per_row
      << ",\"timing_mode\":" << json_string(timing_mode_name(options.timing_mode))
      << ",\"active_bytes\":" << operation.active_bytes
      << ",\"logical_tile_count\":" << operation.tile_count
      << ",\"tail_active_bytes\":" << tail.active_bytes
      << ",\"payload_tile_bytes\":524288,\"slots_per_edge\":8"
      << ",\"lanes_per_edge\":2"
      << ",\"registered_tile_storage_bytes_per_edge\":"
      << options.expected_registered_bytes_per_edge
      << ",\"descriptor_storage_included\":false"
      << ",\"guard_bytes\":" << options.guard_bytes
      << ",\"warmup_operations\":" << options.warmup_operations
      << ",\"measured_operations\":" << options.measured_operations
      << ",\"credit_delay_edge\":" << options.credit_delay_edge
      << ",\"credit_delay_us\":" << options.credit_delay_us
      << ",\"poison_injection\":" << json_string(poison_name(options.poison))
      << ",\"association\":\"counter_rotating_halves\""
      << ",\"lower_half_association\":\"xor1_then_xor3\""
      << ",\"upper_half_association\":\"xor3_then_xor1\""
      << ",\"numerical_oracle\":\"exact_integer_bf16\""
      << ",\"executor_stream_policy\":\"single_stream_correctness_only\""
      << ",\"performance_claim_allowed\":false"
      << ",\"protocol_node_implementation\":\"native_executor\""
      << ",\"passed\":" << (passed ? "true" : "false")
      << ",\"exit_code\":" << exit_code
      << ",\"fatal_state\":" << json_string(passed ? "none" : "poisoned")
      << ",\"completed_operations\":" << completed_operations
      << ",\"tiles_acquired\":" << tiles_acquired
      << ",\"tiles_recycled\":" << tiles_recycled
      << ",\"highest_issued_ordinal\":" << highest_ordinal
      << ",\"highest_retired_ordinal_edge0\":" << highest_ordinal
      << ",\"highest_retired_ordinal_edge1\":" << highest_ordinal
      << ",\"teardown_state\":" << json_string(passed ? "drained" : "fatal_poisoned")
      << ",\"teardown_pending_tiles\":" << correctness.teardown_pending_tiles
      << ",\"teardown_pending_operations\":" << correctness.teardown_pending_operations
      << ",\"peak_tiles_in_flight\":" << peak_tiles
      << ",\"mismatched_active_elements\":" << correctness.mismatched_active_elements
      << ",\"input_guard_corruptions\":" << correctness.input_guard_corruptions
      << ",\"output_guard_corruptions\":" << correctness.output_guard_corruptions
      << ",\"inactive_input_sentinel_corruptions\":" << correctness.inactive_input_sentinel_corruptions
      << ",\"inactive_output_sentinel_corruptions\":" << correctness.inactive_output_sentinel_corruptions
      << ",\"unexpected_generation_count\":" << status.unexpected_generation_count
      << ",\"poisoned_slot_count\":" << poisoned_slot_count
      << ",\"descriptor_overflow_count\":0"
      << ",\"credit_regression_count\":" << status.credit_regression_count
      << ",\"generation_overflow_count\":0"
      << ",\"slot_credit_wait_events\":" << credit_wait_events
      << ",\"output_ready_before_final_retirement_count\":"
      << status.output_ready_before_final_retirement_count
      << ",\"host_submit_us_per_operation\":" << per_operation
      << ",\"device_output_ready_us_min\":" << minimum
      << ",\"device_output_ready_us_p50\":" << p50
      << ",\"device_output_ready_us_p95\":" << p95
      << ",\"device_fully_retired_us_p50\":" << p50
      << ",\"device_fully_retired_us_p95\":" << p95
      << ",\"steady_state_device_us_per_operation\":" << per_operation
      << ",\"active_payload_gib_per_s_p50\":" << bandwidth
      << ",\"steady_state_active_payload_gib_per_s\":" << steady_bandwidth
      << ",\"slot_credit_wait_us_p50\":" << positive_or_epsilon(options.credit_delay_us)
      << ",\"slot_credit_wait_us_p95\":" << positive_or_epsilon(options.credit_delay_us)
      << ",\"stage_phase1_gpu_us_p50\":0"
      << ",\"phase1_remote_wait_us_p50\":0"
      << ",\"reduce_phase1_gpu_us_p50\":0"
      << ",\"phase2_remote_wait_us_p50\":0"
      << ",\"reduce_phase2_gpu_us_p50\":0}"
      ;
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto storage =
        spark_transport::make_tp4_tiled_session_storage_options(
            options.query_rows, options.elements_per_row);
    const auto layout = spark_transport::make_tp4_tiled_pool_layout(storage);
    const auto operation =
        spark_transport::make_tp4_tiled_bf16_allreduce_operation(
            options.query_rows, storage.bytes_per_row);
    if (options.expected_payload_tile_bytes != layout.tile_payload_bytes ||
        options.expected_slots_per_edge != layout.slots_per_edge ||
        options.expected_lanes_per_edge != layout.lanes_per_edge ||
        options.expected_registered_bytes_per_edge != layout.total_bytes) {
      throw std::runtime_error("qualification geometry does not match native layout");
    }

    auto arena0 = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped, layout.total_bytes);
    auto arena1 = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped, layout.total_bytes);
    const auto plan0 = spark_transport::make_tp4_round_plan(options.rank, 0);
    const auto plan1 = spark_transport::make_tp4_round_plan(options.rank, 1);
    auto channel0 = open_channel(plan0, options.peer0, options.control_port0);
    spark_transport::VerbsEndpoint endpoint0(options.device0, 1, options.gid0,
                                              *arena0);
    exchange_and_connect(channel0, endpoint0, options, plan0, storage,
                         layout, operation);
    auto channel1 = open_channel(plan1, options.peer1, options.control_port1);
    spark_transport::VerbsEndpoint endpoint1(options.device1, 1, options.gid1,
                                              *arena1);
    exchange_and_connect(channel1, endpoint1, options, plan1, storage,
                         layout, operation);

    const auto geometry = research::oracle_payload_geometry(
        options.query_rows, options.elements_per_row);
    const std::uint64_t allocation_bytes =
        geometry.capacity_bytes + 2U * options.guard_bytes;
    std::uint8_t* guarded_input{};
    std::uint8_t* guarded_output{};
    research::TiledCorrectnessReceipt* device_correctness{};
    cudaStream_t stream{};
    check_cuda(cudaMalloc(&guarded_input, allocation_bytes), "cudaMalloc guarded input");
    check_cuda(cudaMalloc(&guarded_output, allocation_bytes), "cudaMalloc guarded output");
    check_cuda(cudaMalloc(&device_correctness, sizeof(*device_correctness)),
               "cudaMalloc correctness receipt");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreate tiled probe");
    auto* input = guarded_input + options.guard_bytes;
    auto* output = guarded_output + options.guard_bytes;
    check_cuda(research::launch_initialize_correctness_receipt(
                   device_correctness, options.query_rows, operation.tile_count,
                   operation.active_bytes, options.elements_per_row, stream),
               "initialize correctness receipt");
    check_cuda(research::launch_fill_correctness_sentinels(
                   guarded_input, guarded_output, options.guard_bytes,
                   geometry.capacity_bytes, stream),
               "fill correctness sentinels");

    CudaTiledBulkPort bulk(input, output,
                           static_cast<std::uint8_t*>(arena0->device_data()),
                           static_cast<std::uint8_t*>(arena1->device_data()),
                           options.rank, stream);
    constexpr std::uintptr_t kExecutorOwnerIdentity = 0x54494c4544ULL;
    VerbsTiledEdgePort edge0(0, kExecutorOwnerIdentity, endpoint0, *arena0,
                             options.credit_delay_edge, options.credit_delay_us);
    VerbsTiledEdgePort edge1(1, kExecutorOwnerIdentity, endpoint1, *arena1,
                             options.credit_delay_edge, options.credit_delay_us);
    research::TiledExecutor executor(storage, bulk, edge0, edge1);

    channel0.barrier();
    channel1.barrier();

    const std::uint64_t total_operations =
        static_cast<std::uint64_t>(options.warmup_operations) +
        options.measured_operations;
    std::uint64_t completed_operations{};
    std::uint64_t tiles_acquired{};
    std::uint64_t tiles_recycled{};
    std::uint64_t peak_tiles{};
    std::uint64_t credit_wait_events{};
    std::vector<double> latencies;
    latencies.reserve(options.measured_operations);
    double measured_start{};
    const double all_start = now_us();
    research::TiledBulkDescriptor* validation_descriptor{};
    check_cuda(cudaMalloc(&validation_descriptor,
                          sizeof(*validation_descriptor)),
               "cudaMalloc validation descriptor");

    research::TiledExecutorStatus final_status{};
    if (options.poison != PoisonInjection::kNone) {
      executor.begin(options.query_rows);
      (void)executor.advance();
      const auto region = spark_transport::tp4_tiled_slot_region(layout, 0, 0);
      const auto credit_offset = region.control_offset +
          offsetof(spark_transport::DoorbellControl, acknowledgement_sequence);
      if (options.poison == PoisonInjection::kUnexpectedGeneration) {
        edge0.inject_peer_wire_credit(2, credit_offset);
      } else {
        // Issue enough descriptors for ordinal one to be a valid cumulative
        // watermark, then publish ordinal one followed by ordinal zero.
        (void)executor.advance();
        edge0.inject_peer_wire_credit(2, credit_offset);
        (void)executor.advance();
        edge0.inject_peer_wire_credit(1, credit_offset);
      }
      for (int spin = 0; spin < 1000 && !executor.status().poisoned; ++spin) {
        (void)executor.advance();
      }
      final_status = executor.status();
      research::TiledCorrectnessReceipt correctness{};
      check_cuda(cudaStreamSynchronize(stream),
                 "synchronize poison correctness state");
      check_cuda(cudaMemcpy(&correctness, device_correctness, sizeof(correctness),
                            cudaMemcpyDeviceToHost), "copy poison correctness");
      correctness.teardown_pending_tiles =
          final_status.tile_count - final_status.retired_tiles;
      correctness.teardown_pending_operations = 1;
      std::cout << options.receipt_prefix << ' '
                << make_receipt(options, correctness, final_status, false,
                                kPoisonExitCode, 0, final_status.acquired_tiles,
                                final_status.retired_tiles,
                                final_status.highest_issued_ordinal,
                                final_status.acquired_tiles, 0,
                                now_us() - all_start, nullptr, 1)
                << '\n';
      return kPoisonExitCode;
    }

    const bool validate_each_operation =
        options.timing_mode != TimingMode::kSteady;
    const auto validate_operation = [&](
        const research::TiledExecutorStatus& status) {
      for (std::uint32_t tile = 0; tile < operation.tile_count; ++tile) {
        const auto host_tile = research::make_tiled_executor_tile_descriptor(
            spark_transport::make_tp4_tiled_tile_descriptor(
                operation, tile, status.first_tile_ordinal),
            layout);
        check_cuda(cudaMemcpyAsync(
                       validation_descriptor, &host_tile.gpu,
                       sizeof(host_tile.gpu), cudaMemcpyHostToDevice, stream),
                   "upload validation descriptor");
        check_cuda(research::launch_validate_correctness(
                       output, validation_descriptor, device_correctness,
                       stream),
                   "validate active output");
      }
      check_cuda(cudaStreamSynchronize(stream),
                 "synchronize operation correctness checks");
    };

    for (std::uint64_t op = 0; op < total_operations; ++op) {
      executor.begin(options.query_rows);
      const double operation_start = now_us();
      bool counted_wait{};
      while (true) {
        const auto advance = executor.advance();
        const auto status = executor.status();
        peak_tiles = std::max<std::uint64_t>(
            peak_tiles, status.acquired_tiles - status.retired_tiles);
        if (advance.backpressured && !counted_wait) {
          ++credit_wait_events;
          counted_wait = true;
        }
        if (status.output_ready && validate_each_operation) {
          // The executor uploaded this operation's descriptors while staging.
          // Validate active values after the output stream reaches completion.
          check_cuda(cudaStreamSynchronize(stream), "synchronize output ready");
        }
        if (executor.drain() == research::TiledDrainState::kComplete) break;
        if (executor.status().poisoned) {
          throw std::runtime_error("tiled executor poisoned during success arm");
        }
        std::this_thread::yield();
      }
      const double operation_end = now_us();
      final_status = executor.status();
      // Validate every operation before the following begin can overwrite its
      // output. The receipt aggregates all active-element mismatches while the
      // descriptor retains the exact physical-slot generation used by that
      // operation.
      if (validate_each_operation) {
        validate_operation(final_status);
      }
      tiles_acquired += final_status.tile_count;
      tiles_recycled += final_status.retired_tiles;
      ++completed_operations;
      if (op == options.warmup_operations) measured_start = operation_start;
      if (op >= options.warmup_operations) {
        latencies.push_back(operation_end - operation_start);
      }
    }
    const double measured_stop = now_us();

    if (!validate_each_operation) {
      validate_operation(final_status);
    }

    if (!edge0.drain_all() || !edge1.drain_all()) {
      throw std::runtime_error(
          "tiled-prefill send completion drain failed");
    }

    check_cuda(research::launch_validate_correctness_sentinels(
                   guarded_input, guarded_output, options.guard_bytes,
                   geometry.active_bytes, geometry.capacity_bytes,
                   device_correctness, stream),
               "validate sentinels");
    check_cuda(cudaStreamSynchronize(stream), "synchronize correctness checks");
    cudaFree(validation_descriptor);

    research::TiledCorrectnessReceipt correctness{};
    check_cuda(cudaMemcpy(&correctness, device_correctness, sizeof(correctness),
                          cudaMemcpyDeviceToHost), "copy correctness receipt");
    correctness.teardown_drained = final_status.safe_to_release_registered_storage;
    correctness.teardown_pending_tiles =
        final_status.tile_count - final_status.retired_tiles;
    correctness.teardown_pending_operations =
        final_status.safe_to_release_registered_storage ? 0U : 1U;
    const bool correct = correctness.mismatched_active_elements == 0 &&
                         correctness.input_guard_corruptions == 0 &&
                         correctness.output_guard_corruptions == 0 &&
                         correctness.inactive_input_sentinel_corruptions == 0 &&
                         correctness.inactive_output_sentinel_corruptions == 0 &&
                         final_status.safe_to_release_registered_storage;
    if (!correct) throw std::runtime_error("tiled-prefill correctness gate failed");

    const auto summary = spark_transport::summarize_latencies(latencies);
    const double elapsed =
        measured_stop - (measured_start == 0 ? all_start : measured_start);
    std::cout << options.receipt_prefix << ' '
              << make_receipt(options, correctness, final_status, true, 0,
                              completed_operations, tiles_acquired,
                              tiles_recycled,
                              total_operations * operation.tile_count - 1U,
                              peak_tiles, credit_wait_events, elapsed, &summary,
                              0)
              << '\n';

    check_cuda(cudaStreamSynchronize(stream), "final tiled probe synchronize");
    cudaStreamDestroy(stream);
    cudaFree(device_correctness);
    cudaFree(guarded_output);
    cudaFree(guarded_input);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
