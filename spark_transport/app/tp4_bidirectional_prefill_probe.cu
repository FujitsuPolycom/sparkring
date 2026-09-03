// Research-only adaptive-Q width-4096 BF16 bidirectional-ring qualification.
// This executable is deliberately absent from the production C ABI and vLLM.

#include "bidirectional_bulk_kernels.cuh"
#include "bidirectional_ring_executor.hpp"
#include "spark_transport/control_channel.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/statistics.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/tp4_tiled_session.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;

namespace {

constexpr std::uint16_t kEndpointVersion = 6;
constexpr std::uint16_t kEndpointTag = 0x4244;  // "BD"
constexpr std::uint32_t kGeometryMagic = 0x42445247;  // "BDRG"
constexpr std::uint16_t kGeometryVersion = 1;
constexpr std::uint64_t kGuardBytes = 4096;
constexpr std::uint8_t kInputGuard = 0xa5;
constexpr std::uint8_t kOutputGuard = 0x5a;
constexpr std::uint16_t kExpectedBf16Ten = 0x4120;

enum class Mode { kCorrectness, kSteady };
enum class RailMode { kSingle, kDual };

struct Options {
  std::uint32_t rank{4};
  std::uint32_t query_rows{2048};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t port0{19000};
  std::uint16_t port1{19001};
  std::string secondary_peer0;
  std::string secondary_peer1;
  std::string secondary_device0;
  std::string secondary_device1;
  std::uint8_t secondary_gid0{3};
  std::uint8_t secondary_gid1{3};
  std::uint16_t secondary_port0{19100};
  std::uint16_t secondary_port1{19101};
  RailMode rail_mode{RailMode::kSingle};
  bool allow_shared_peer_ip{};
  std::uint32_t warmup{};
  std::uint32_t iterations{1};
  std::uint32_t timeout_seconds{120};
  Mode mode{Mode::kCorrectness};
};

struct Geometry {
  std::uint32_t magic{kGeometryMagic};
  std::uint16_t version{kGeometryVersion};
  std::uint16_t reserved{};
  std::uint16_t rail_count{1};
  std::uint16_t rail_index{};
  std::uint32_t rank{};
  std::uint32_t peer_rank{};
  std::uint32_t world_size{spark_transport::kTp4PrefillRankCount};
  std::uint32_t query_rows{spark_transport::kTp4PrefillQ2048Rows};
  std::uint32_t elements_per_row{spark_transport::kTp4PrefillWidth4096};
  std::uint32_t tiles_per_shard{research::kBidirectionalRingTilesPerShard};
  std::uint32_t slots_per_direction{
      research::kBidirectionalRingSlotsPerDirection};
  std::uint32_t lanes_per_slot{1};
  std::uint64_t payload_bytes{spark_transport::kTp4PrefillPayloadBytes};
  std::uint64_t tile_bytes{research::kBidirectionalRingTileBytes};
  std::uint64_t slot_stride{};
  std::uint64_t arena_bytes{};
};

static_assert(std::is_trivially_copyable_v<Geometry>);

[[noreturn]] void usage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " --rank R --query-rows 1024|2048|4096|8192"
               " --peer0 IP --peer1 IP [--device0 HCA]"
               " [--device1 HCA] [--gid0 N] [--gid1 N]"
               " [--port0 N] [--port1 N] [--mode correctness|steady]"
               " [--rail-mode single|dual] [--secondary-peer0 IP]"
               " [--secondary-peer1 IP] [--secondary-device0 HCA]"
               " [--secondary-device1 HCA] [--secondary-port0 N]"
               " [--secondary-port1 N]"
               " [--allow-shared-peer-ip]"
               " [--warmup N] [--iterations N] [--timeout-seconds N]\n";
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

Options parse_options(int argc, char** argv) {
  Options options;
  if (const char* value = std::getenv("SPARK_TP4_BIDIRECTIONAL_RAIL_MODE")) {
    const std::string_view mode(value);
    if (mode == "dual") options.rail_mode = RailMode::kDual;
    else if (mode != "single") throw std::invalid_argument(
        "SPARK_TP4_BIDIRECTIONAL_RAIL_MODE must be single or dual");
  }
  if (const char* value = std::getenv("SPARK_TP4_BIDIRECTIONAL_SECONDARY_PEER0"))
    options.secondary_peer0 = value;
  if (const char* value = std::getenv("SPARK_TP4_BIDIRECTIONAL_SECONDARY_PEER1"))
    options.secondary_peer1 = value;
  if (const char* value = std::getenv("SPARK_TP4_BIDIRECTIONAL_SECONDARY_DEVICE0"))
    options.secondary_device0 = value;
  if (const char* value = std::getenv("SPARK_TP4_BIDIRECTIONAL_SECONDARY_DEVICE1"))
    options.secondary_device1 = value;
  if (const char* value = std::getenv("SPARK_TP4_BIDIRECTIONAL_SECONDARY_PORT0"))
    options.secondary_port0 = static_cast<std::uint16_t>(unsigned_value(value, "secondary port0"));
  if (const char* value = std::getenv("SPARK_TP4_BIDIRECTIONAL_SECONDARY_PORT1"))
    options.secondary_port1 = static_cast<std::uint16_t>(unsigned_value(value, "secondary port1"));
  if (const char* value = std::getenv(
          "SPARK_TP4_BIDIRECTIONAL_ALLOW_SHARED_PEER_IP")) {
    options.allow_shared_peer_ip = std::string_view(value) == "1";
  }
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto take = [&]() -> const char* {
      if (++index >= argc) usage(argv[0]);
      return argv[index];
    };
    if (argument == "--rank") {
      options.rank = static_cast<std::uint32_t>(unsigned_value(take(), "rank"));
    } else if (argument == "--query-rows") {
      options.query_rows = static_cast<std::uint32_t>(
          unsigned_value(take(), "query rows"));
    } else if (argument == "--peer0") {
      options.peer0 = take();
    } else if (argument == "--peer1") {
      options.peer1 = take();
    } else if (argument == "--device0") {
      options.device0 = take();
    } else if (argument == "--device1") {
      options.device1 = take();
    } else if (argument == "--gid0") {
      options.gid0 = static_cast<std::uint8_t>(unsigned_value(take(), "gid0"));
    } else if (argument == "--gid1") {
      options.gid1 = static_cast<std::uint8_t>(unsigned_value(take(), "gid1"));
    } else if (argument == "--port0") {
      options.port0 = static_cast<std::uint16_t>(unsigned_value(take(), "port0"));
    } else if (argument == "--port1") {
      options.port1 = static_cast<std::uint16_t>(unsigned_value(take(), "port1"));
    } else if (argument == "--rail-mode") {
      const std::string_view mode(take());
      if (mode == "single") options.rail_mode = RailMode::kSingle;
      else if (mode == "dual") options.rail_mode = RailMode::kDual;
      else usage(argv[0]);
    } else if (argument == "--secondary-peer0") {
      options.secondary_peer0 = take();
    } else if (argument == "--secondary-peer1") {
      options.secondary_peer1 = take();
    } else if (argument == "--secondary-device0") {
      options.secondary_device0 = take();
    } else if (argument == "--secondary-device1") {
      options.secondary_device1 = take();
    } else if (argument == "--secondary-gid0") {
      options.secondary_gid0 = static_cast<std::uint8_t>(unsigned_value(take(), "secondary gid0"));
    } else if (argument == "--secondary-gid1") {
      options.secondary_gid1 = static_cast<std::uint8_t>(unsigned_value(take(), "secondary gid1"));
    } else if (argument == "--secondary-port0") {
      options.secondary_port0 = static_cast<std::uint16_t>(unsigned_value(take(), "secondary port0"));
    } else if (argument == "--secondary-port1") {
      options.secondary_port1 = static_cast<std::uint16_t>(unsigned_value(take(), "secondary port1"));
    } else if (argument == "--allow-shared-peer-ip") {
      options.allow_shared_peer_ip = true;
    } else if (argument == "--warmup") {
      options.warmup = static_cast<std::uint32_t>(unsigned_value(take(), "warmup"));
    } else if (argument == "--iterations") {
      options.iterations = static_cast<std::uint32_t>(unsigned_value(take(), "iterations"));
    } else if (argument == "--timeout-seconds") {
      options.timeout_seconds = static_cast<std::uint32_t>(unsigned_value(take(), "timeout"));
    } else if (argument == "--mode") {
      const std::string_view mode(take());
      if (mode == "correctness") options.mode = Mode::kCorrectness;
      else if (mode == "steady") options.mode = Mode::kSteady;
      else usage(argv[0]);
    } else {
      usage(argv[0]);
    }
  }
  if (options.rank >= spark_transport::kTp4PrefillRankCount ||
      !research::bidirectional_ring_query_rows_supported(options.query_rows) ||
      options.peer0.empty() || options.peer1.empty() || options.port0 == 0 ||
      options.port1 == 0 || options.port0 == options.port1 ||
      options.iterations == 0 || options.timeout_seconds == 0) {
    usage(argv[0]);
  }
  if (options.rail_mode == RailMode::kDual &&
      (options.secondary_peer0.empty() || options.secondary_peer1.empty() ||
       options.secondary_device0.empty() || options.secondary_device1.empty() ||
       options.secondary_port0 == 0 || options.secondary_port1 == 0 ||
       options.secondary_device0 == options.device0 ||
       options.secondary_device1 == options.device1 ||
       (!options.allow_shared_peer_ip &&
        (options.secondary_peer0 == options.peer0 ||
         options.secondary_peer1 == options.peer1)) ||
       std::unordered_set<std::uint16_t>{
           options.port0, options.port1, options.secondary_port0,
           options.secondary_port1}.size() != 4)) {
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

std::size_t endpoint_index(spark_transport::Tp4PrefillEndpoint endpoint) {
  return endpoint == spark_transport::Tp4PrefillEndpoint::kXor1 ? 0U : 1U;
}

std::size_t direction_index(spark_transport::Tp4PrefillDirection direction) {
  return direction == spark_transport::Tp4PrefillDirection::kClockwise ? 0U
                                                                       : 1U;
}

std::uint64_t load_acquire(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_release(std::uint64_t* address, std::uint64_t value) {
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

void connect_endpoint(spark_transport::ControlChannel& channel,
                      spark_transport::VerbsEndpoint& endpoint,
                      const spark_transport::Tp4RoundPlan& plan,
                      const spark_transport::Tp4TiledPoolLayout& layout,
                      const research::BidirectionalRingGeometry& ring,
                      std::uint32_t rank, std::uint16_t rail_count,
                      std::uint16_t rail_index) {
  auto local = endpoint.local_info();
  local.version = kEndpointVersion;
  local.reserved = kEndpointTag;
  const auto remote = channel.exchange(local);
  if (remote.version != local.version || remote.reserved != local.reserved ||
      remote.buffer_bytes != local.buffer_bytes) {
    throw std::runtime_error("bidirectional endpoint handshake mismatch");
  }
  const Geometry local_geometry{kGeometryMagic,
                                kGeometryVersion,
                                0,
                                rail_count,
                                rail_index,
                                rank,
                                plan.peer_rank,
                                spark_transport::kTp4PrefillRankCount,
                                ring.query_rows,
                                ring.elements_per_row,
                                research::kBidirectionalRingTilesPerShard,
                                research::kBidirectionalRingSlotsPerDirection,
                                1,
                                ring.payload_bytes,
                                ring.tile_bytes,
                                layout.slot_stride,
                                layout.total_bytes};
  const Geometry remote_geometry = channel.exchange(local_geometry);
  if (remote_geometry.magic != local_geometry.magic ||
      remote_geometry.version != local_geometry.version ||
      remote_geometry.reserved != 0 ||
      remote_geometry.rail_count != local_geometry.rail_count ||
      remote_geometry.rail_index != local_geometry.rail_index ||
      remote_geometry.rank != plan.peer_rank ||
      remote_geometry.peer_rank != rank ||
      remote_geometry.world_size != local_geometry.world_size ||
      remote_geometry.query_rows != local_geometry.query_rows ||
      remote_geometry.elements_per_row != local_geometry.elements_per_row ||
      remote_geometry.tiles_per_shard != local_geometry.tiles_per_shard ||
      remote_geometry.slots_per_direction !=
          local_geometry.slots_per_direction ||
      remote_geometry.lanes_per_slot != local_geometry.lanes_per_slot ||
      remote_geometry.payload_bytes != local_geometry.payload_bytes ||
      remote_geometry.tile_bytes != local_geometry.tile_bytes ||
      remote_geometry.slot_stride != local_geometry.slot_stride ||
      remote_geometry.arena_bytes != local_geometry.arena_bytes) {
    throw std::runtime_error("bidirectional geometry handshake mismatch");
  }
  endpoint.connect(remote, local.version);
}

void require_dual_mtu_4096(const spark_transport::VerbsEndpoint& primary,
                           const spark_transport::VerbsEndpoint& secondary) {
  const auto primary_mtu = primary.active_mtu_bytes();
  const auto secondary_mtu = secondary.active_mtu_bytes();
  if (primary_mtu != 4096 || secondary_mtu != 4096 ||
      primary_mtu != secondary_mtu) {
    throw std::runtime_error(
        "dual rail preflight requires both active MTUs equal to 4096");
  }
}

class CudaBulkPort final : public research::BidirectionalRingBulkPort {
 public:
  CudaBulkPort(std::uint8_t* input, std::uint8_t* output,
               std::array<std::uint8_t*, 2> endpoints, std::uint32_t rank,
               spark_transport::Tp4TiledPoolLayout layout,
               research::BidirectionalRingGeometry geometry,
               cudaStream_t stream, std::uint32_t rail_count)
      : input_(input), output_(output), endpoints_(endpoints), rank_(rank),
        layout_(layout), geometry_(geometry), stream_(stream),
        rail_count_(rail_count) {
    check_cuda(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
               "create bidirectional completion event");
  }

  ~CudaBulkPort() override {
    if (completion_ != nullptr) cudaEventDestroy(completion_);
  }

  research::RingSubmitState try_submit(
      const research::BidirectionalRingBulkRequest& request) override {
    if (pending_) return research::RingSubmitState::kBackpressured;
    if (request.active_bytes != geometry_.tile_bytes) {
      return research::RingSubmitState::kFatal;
    }

    const bool initial = request.action == research::RingBulkAction::kStageInitial;
    const bool finish = request.action == research::RingBulkAction::kGatherFinish;
    const std::uint32_t descriptor_stage =
        initial ? 0U
                : research::bidirectional_ring_consumed_stage(
                      request.next_exchange_stage);
    std::uint64_t send_offset{};
    std::uint64_t receive_offset{};
    if (!finish) {
      const auto region = spark_transport::tp4_tiled_slot_region(
          layout_, request.destination_ticket.slot, 0);
      send_offset = region.send_offset;
    }
    if (!initial) {
      const auto region = spark_transport::tp4_tiled_slot_region(
          layout_, request.source_ticket.slot, 0);
      receive_offset = region.receive_offset;
    }

    const auto outgoing = endpoint_index(request.outgoing_endpoint);
    const auto incoming = endpoint_index(request.incoming_endpoint);
    const auto aligned = [](const void* pointer, std::uint64_t offset) {
      return (reinterpret_cast<std::uintptr_t>(pointer) + offset) % 16U == 0;
    };
    if ((!finish && !aligned(endpoints_[outgoing], send_offset)) ||
        (!initial && !aligned(endpoints_[incoming], receive_offset)) ||
        !aligned(input_, request.operation_offset_bytes) ||
        !aligned(output_, request.operation_offset_bytes) ||
        (!initial &&
         (request.incoming_doorbell_offset % alignof(std::uint64_t) != 0 ||
          request.incoming_doorbell_offset + sizeof(std::uint64_t) >
              layout_.total_bytes ||
          request.consumed_doorbell_token == 0)) ||
        (rail_count_ == 2 && !initial &&
         (request.secondary_incoming_doorbell_offset %
                  alignof(std::uint64_t) !=
              0 ||
          request.secondary_consumed_doorbell_token == 0))) {
      return research::RingSubmitState::kFatal;
    }

    const research::BidirectionalBulkDescriptor host_descriptor =
        initial
            ? research::make_bidirectional_stage_initial_descriptor(
                  rank_, request.direction, request.tile_in_shard, send_offset,
                  geometry_.query_rows)
            : research::make_bidirectional_post_exchange_descriptor(
                  rank_, request.direction, descriptor_stage,
                  request.tile_in_shard, send_offset, receive_offset,
                  request.incoming_doorbell_offset,
                  request.consumed_doorbell_token, geometry_.query_rows,
                  rail_count_ == 2 ? request.secondary_incoming_doorbell_offset
                                   : 0U,
                  rail_count_ == 2 ? request.secondary_consumed_doorbell_token
                                   : 0U);
    if (host_descriptor.tensor_offset_bytes !=
            request.operation_offset_bytes ||
        host_descriptor.shard != request.shard ||
        host_descriptor.stage != descriptor_stage ||
        host_descriptor.half != static_cast<std::uint32_t>(request.half) ||
        host_descriptor.active_bytes != request.active_bytes) {
      return research::RingSubmitState::kFatal;
    }
    cudaError_t result = cudaSuccess;
    switch (request.action) {
      case research::RingBulkAction::kStageInitial:
        result = research::launch_bidirectional_stage_initial(
            input_, endpoints_[outgoing], host_descriptor, stream_,
            geometry_.query_rows);
        break;
      case research::RingBulkAction::kReduceForward:
        result = research::launch_bidirectional_reduce_forward(
            input_, endpoints_[incoming], endpoints_[outgoing], host_descriptor,
            stream_, geometry_.query_rows);
        break;
      case research::RingBulkAction::kReduceFinalizeAndSeedGather:
        result = research::launch_bidirectional_reduce_finalize_seed_gather(
            input_, endpoints_[incoming], endpoints_[outgoing], output_,
            host_descriptor, stream_, geometry_.query_rows);
        break;
      case research::RingBulkAction::kGatherForward:
        result = research::launch_bidirectional_gather_forward(
            endpoints_[incoming], endpoints_[outgoing], output_, host_descriptor,
            stream_, geometry_.query_rows);
        break;
      case research::RingBulkAction::kGatherFinish:
        result = research::launch_bidirectional_gather_finish(
            endpoints_[incoming], output_, host_descriptor, stream_,
            geometry_.query_rows);
        break;
    }
    check_cuda(result, "launch bidirectional bulk kernel");
    check_cuda(cudaEventRecord(completion_, stream_),
               "record bidirectional bulk completion");
    pending_ = true;
    pending_action_ = request.action;
    pending_direction_ = request.direction;
    pending_stage_ = request.next_exchange_stage;
    pending_tile_ = request.tile_in_shard;
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll(
      const research::BidirectionalRingBulkRequest& request) override {
    if (!pending_ || pending_action_ != request.action ||
        pending_direction_ != request.direction ||
        pending_stage_ != request.next_exchange_stage ||
        pending_tile_ != request.tile_in_shard) {
      return research::RingPollState::kFatal;
    }
    const auto result = cudaEventQuery(completion_);
    if (result == cudaErrorNotReady) return research::RingPollState::kPending;
    if (result != cudaSuccess) return research::RingPollState::kFatal;
    pending_ = false;
    return research::RingPollState::kComplete;
  }

 private:
  std::uint8_t* input_{};
  std::uint8_t* output_{};
  std::array<std::uint8_t*, 2> endpoints_{};
  std::uint32_t rank_{};
  spark_transport::Tp4TiledPoolLayout layout_{};
  research::BidirectionalRingGeometry geometry_{};
  cudaStream_t stream_{};
  cudaEvent_t completion_{};
  std::uint32_t rail_count_{1};
  bool pending_{};
  research::RingBulkAction pending_action_{};
  spark_transport::Tp4PrefillDirection pending_direction_{};
  std::uint32_t pending_stage_{};
  std::uint32_t pending_tile_{};
};

class VerbsEdgePort final : public research::BidirectionalRingEdgePort {
 public:
  using EndpointMatrix =
      std::array<std::array<spark_transport::VerbsEndpoint*, 2>, 2>;

  VerbsEdgePort(EndpointMatrix endpoints,
                std::array<spark_transport::MemoryBuffer*, 2> arenas,
                spark_transport::Tp4TiledPoolLayout layout,
                std::uint32_t rail_count)
      : endpoints_(endpoints), arenas_(arenas), layout_(layout),
        rail_count_(rail_count) {
    if (rail_count_ != 1 && rail_count_ != 2) {
      throw std::invalid_argument("rail count must be one or two");
    }
    if (rail_count_ == 2) {
      for (const auto& neighbor : endpoints_) {
        if (neighbor[0] == nullptr || neighbor[1] == nullptr ||
            neighbor[0]->active_mtu_bytes() != 4096 ||
            neighbor[1]->active_mtu_bytes() != 4096 ||
            neighbor[0]->active_mtu_bytes() !=
                neighbor[1]->active_mtu_bytes()) {
          throw std::runtime_error(
              "dual rail requires equal active MTU 4096");
        }
      }
    }
  }
  research::RingSubmitState try_post_exchange(
      const research::BidirectionalRingExchangeRequest& request) override {
    if (!drain_completions()) return research::RingSubmitState::kFatal;
    const auto direction = direction_index(request.direction);
    const auto endpoint = endpoint_index(request.outgoing_endpoint);
    if (exchanges_[direction].count(request.work_id) != 0 ||
        request.span.active_bytes != layout_.tile_payload_bytes ||
        request.span.local_send_offset % 16U != 0 ||
        request.span.remote_receive_offset % 16U != 0 ||
        request.span.local_send_offset + request.span.active_bytes >
            arenas_[endpoint]->size() ||
        request.span.remote_receive_offset + request.span.active_bytes >
            arenas_[endpoint]->size()) {
      return research::RingSubmitState::kFatal;
    }
    const auto region = spark_transport::tp4_tiled_slot_region(
        layout_, request.ticket.slot, 0);
    const std::uint32_t split = rail_count_ == 2 ? 2U : 1U;
    const std::uint32_t rail_bytes = request.span.active_bytes / split;
    if (request.span.active_bytes % split != 0 || rail_bytes % 16U != 0) {
      return research::RingSubmitState::kFatal;
    }
    // Preflight every rail before posting either half; a half-post exchange is
    // unrecoverable because the peer cannot consume or credit a partial tile.
    for (std::uint32_t rail = 0; rail < rail_count_; ++rail) {
      if (outstanding_wqes_[endpoint][rail] + 2U >
          endpoints_[endpoint][rail]->maximum_send_work_requests()) {
        return research::RingSubmitState::kBackpressured;
      }
    }
    ExchangeState state{};
    state.rail_count = rail_count_;
    for (std::uint32_t rail = 0; rail < rail_count_; ++rail) {
      const std::uint64_t local_doorbell =
          region.control_offset +
          (rail == 0
               ? offsetof(spark_transport::DoorbellControl, command_sequence)
               : offsetof(spark_transport::DoorbellControl, mismatch_count));
      const std::uint64_t remote_doorbell =
          region.control_offset +
          (rail == 0
               ? offsetof(spark_transport::DoorbellControl, remote_sequence)
               : offsetof(spark_transport::DoorbellControl, reserved));
      const std::uint64_t completion_id = allocate_completion_id(endpoint, rail);
      state.completion_ids[rail] = completion_id;
      pending_cq_[endpoint][rail].insert(completion_id);
      completion_wqes_[endpoint][rail].emplace(completion_id, 2U);
      outstanding_wqes_[endpoint][rail] += 2U;
      store_release(word(endpoint, local_doorbell), request.doorbell_token);
      endpoints_[endpoint][rail]->write(
          request.span.local_send_offset + rail * rail_bytes,
          request.span.remote_receive_offset + rail * rail_bytes,
          rail_bytes, completion_id, false);
      endpoints_[endpoint][rail]->write(
          local_doorbell, remote_doorbell, sizeof(request.doorbell_token),
          completion_id, true);
      payload_bytes_[endpoint][rail] += rail_bytes;
      doorbell_bytes_[endpoint][rail] += sizeof(request.doorbell_token);
    }
    exchanges_[direction].emplace(request.work_id, state);
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_exchange(
      const research::BidirectionalRingExchangeRequest& request) override {
    if (!drain_completions()) return research::RingPollState::kFatal;
    const auto direction = direction_index(request.direction);
    const auto found = exchanges_[direction].find(request.work_id);
    if (found == exchanges_[direction].end()) {
      return research::RingPollState::kFatal;
    }
    const auto incoming = endpoint_index(request.incoming_endpoint);
    const auto outgoing = endpoint_index(request.outgoing_endpoint);
    const auto region = spark_transport::tp4_tiled_slot_region(
        layout_, request.ticket.slot, 0);
    const auto inbound_offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, remote_sequence);
    for (std::uint32_t rail = 0; rail < found->second.rail_count; ++rail) {
      const auto offset = rail == 0
                              ? inbound_offset
                              : region.control_offset +
                                    offsetof(spark_transport::DoorbellControl,
                                             reserved);
      const auto observed = load_acquire(word(incoming, offset));
      if (observed > request.doorbell_token) {
        fatal_ = true;
        return research::RingPollState::kFatal;
      }
      if (observed < request.doorbell_token ||
          pending_cq_[outgoing][rail].count(
              found->second.completion_ids[rail]) != 0) {
        return research::RingPollState::kPending;
      }
    }
    exchanges_[direction].erase(found);
    return research::RingPollState::kComplete;
  }

  research::RingSubmitState try_publish_consumed_through(
      const research::BidirectionalRingCreditRequest& request) override {
    if (!drain_completions()) return research::RingSubmitState::kFatal;
    const auto direction = direction_index(request.direction);
    if (credits_[direction].active) {
      return research::RingSubmitState::kBackpressured;
    }
    const auto endpoint = endpoint_index(request.endpoint);
    const auto region = spark_transport::tp4_tiled_slot_region(layout_, 0, 0);
    const auto local_offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, consumer_sequence);
    const auto remote_offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, acknowledgement_sequence);
    constexpr std::uint32_t rail = 0;
    if (outstanding_wqes_[endpoint][rail] + 1U >
        endpoints_[endpoint][rail]->maximum_send_work_requests()) {
      return research::RingSubmitState::kBackpressured;
    }
    const std::uint64_t completion_id = allocate_completion_id(endpoint, rail);
    credits_[direction] = {true, request.work_id, completion_id};
    pending_cq_[endpoint][rail].insert(completion_id);
    completion_wqes_[endpoint][rail].emplace(completion_id, 1U);
    ++outstanding_wqes_[endpoint][rail];
    store_release(word(endpoint, local_offset), request.wire_credit);
    endpoints_[endpoint][rail]->write(local_offset, remote_offset,
                                      sizeof(request.wire_credit), completion_id,
                                      true);
    credit_bytes_[endpoint][rail] += sizeof(request.wire_credit);
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_published_consumed_through(
      const research::BidirectionalRingCreditRequest& request) override {
    if (!drain_completions()) return research::RingPollState::kFatal;
    const auto direction = direction_index(request.direction);
    auto& credit = credits_[direction];
    if (!credit.active || credit.logical_work_id != request.work_id) {
      return research::RingPollState::kFatal;
    }
    const auto endpoint = endpoint_index(request.endpoint);
    if (pending_cq_[endpoint][0].count(credit.completion_id) != 0) {
      return research::RingPollState::kPending;
    }
    credit = {};
    return research::RingPollState::kComplete;
  }

  research::RingCreditPollState poll_peer_consumed_through(
      spark_transport::Tp4PrefillDirection direction,
      std::uint64_t& wire_credit) override {
    if (!drain_completions()) return research::RingCreditPollState::kFatal;
    const auto direction_value = direction_index(direction);
    // Endpoint identity depends on rank, so use the opposite of the credit's
    // incoming endpoint as encoded by the direction-local exchange traffic.
    const auto endpoint = direction ==
                                  spark_transport::Tp4PrefillDirection::kClockwise
                              ? clockwise_outgoing_endpoint_
                              : counterclockwise_outgoing_endpoint_;
    const auto region = spark_transport::tp4_tiled_slot_region(layout_, 0, 0);
    const auto offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, acknowledgement_sequence);
    wire_credit = load_acquire(word(endpoint, offset));
    if (wire_credit == 0 || wire_credit == peer_credit_[direction_value]) {
      return research::RingCreditPollState::kNoUpdate;
    }
    peer_credit_[direction_value] = wire_credit;
    return research::RingCreditPollState::kUpdate;
  }

  void set_direction_endpoints(std::uint32_t rank) {
    clockwise_outgoing_endpoint_ = endpoint_index(
        spark_transport::tp4_prefill_outgoing_endpoint(
            rank, spark_transport::Tp4PrefillDirection::kClockwise));
    counterclockwise_outgoing_endpoint_ = endpoint_index(
        spark_transport::tp4_prefill_outgoing_endpoint(
            rank, spark_transport::Tp4PrefillDirection::kCounterClockwise));
  }

  bool drain_all() noexcept {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (has_pending_cq() && !fatal_) {
      if (!drain_completions()) return false;
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::yield();
    }
    return !fatal_ && exchanges_[0].empty() && exchanges_[1].empty() &&
           !credits_[0].active && !credits_[1].active;
  }

  const std::array<std::array<std::uint64_t, 2>, 2>& payload_bytes() const noexcept {
    return payload_bytes_;
  }
  const std::array<std::array<std::uint64_t, 2>, 2>& doorbell_bytes() const noexcept {
    return doorbell_bytes_;
  }
  const std::array<std::array<std::uint64_t, 2>, 2>& credit_bytes() const noexcept {
    return credit_bytes_;
  }
  const std::array<std::array<std::uint64_t, 2>, 2>& cq_completions() const noexcept {
    return cq_completions_;
  }

 private:
  struct CreditState {
    bool active{};
    std::uint64_t logical_work_id{};
    std::uint64_t completion_id{};
  };

  struct ExchangeState {
    std::uint32_t rail_count{};
    std::array<std::uint64_t, 2> completion_ids{};
  };

  std::uint64_t allocate_completion_id(std::size_t endpoint,
                                       std::size_t rail) {
    if (next_completion_id_[endpoint][rail] ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("verbs completion ID exhausted");
    }
    return next_completion_id_[endpoint][rail]++;
  }

  bool has_pending_cq() const noexcept {
    for (const auto& neighbor : pending_cq_) {
      for (const auto& rail : neighbor) {
        if (!rail.empty()) return true;
      }
    }
    return false;
  }

  bool drain_completions() noexcept {
    if (fatal_) return false;
    try {
      for (std::size_t endpoint = 0; endpoint < endpoints_.size(); ++endpoint) {
        for (std::size_t rail = 0; rail < rail_count_; ++rail) {
          spark_transport::SendCompletion completions[32]{};
          const auto count = endpoints_[endpoint][rail]->poll_send_completions(
              completions, 32);
          for (std::size_t index = 0; index < count; ++index) {
            const auto id = completions[index].work_id;
            if (pending_cq_[endpoint][rail].erase(id) != 1) {
              fatal_ = true;
              return false;
            }
            const auto found = completion_wqes_[endpoint][rail].find(id);
            if (found == completion_wqes_[endpoint][rail].end() ||
                outstanding_wqes_[endpoint][rail] < found->second) {
              fatal_ = true;
              return false;
            }
            outstanding_wqes_[endpoint][rail] -= found->second;
            completion_wqes_[endpoint][rail].erase(found);
            ++cq_completions_[endpoint][rail];
          }
        }
      }
      return true;
    } catch (...) {
      fatal_ = true;
      return false;
    }
  }

  std::uint64_t* word(std::size_t endpoint, std::uint64_t offset) const {
    if (offset + sizeof(std::uint64_t) > arenas_[endpoint]->size() ||
        offset % alignof(std::uint64_t) != 0) {
      throw std::out_of_range("bidirectional control offset outside arena");
    }
    return reinterpret_cast<std::uint64_t*>(
        static_cast<std::uint8_t*>(arenas_[endpoint]->host_data()) + offset);
  }

  EndpointMatrix endpoints_{};
  std::array<spark_transport::MemoryBuffer*, 2> arenas_{};
  spark_transport::Tp4TiledPoolLayout layout_{};
  std::uint32_t rail_count_{1};
  std::array<std::array<std::uint64_t, 2>, 2> next_completion_id_{{{1, 1}, {1, 1}}};
  std::array<std::array<std::unordered_set<std::uint64_t>, 2>, 2> pending_cq_{};
  std::array<std::array<std::unordered_map<std::uint64_t, std::uint32_t>, 2>, 2>
      completion_wqes_{};
  std::array<std::array<std::uint32_t, 2>, 2> outstanding_wqes_{};
  std::array<std::unordered_map<std::uint64_t, ExchangeState>, 2> exchanges_{};
  std::array<CreditState, 2> credits_{};
  std::array<std::uint64_t, 2> peer_credit_{};
  std::array<std::array<std::uint64_t, 2>, 2> payload_bytes_{};
  std::array<std::array<std::uint64_t, 2>, 2> doorbell_bytes_{};
  std::array<std::array<std::uint64_t, 2>, 2> credit_bytes_{};
  std::array<std::array<std::uint64_t, 2>, 2> cq_completions_{};
  std::size_t clockwise_outgoing_endpoint_{};
  std::size_t counterclockwise_outgoing_endpoint_{1};
  bool fatal_{};
};

std::uint16_t input_value(std::uint32_t rank) {
  constexpr std::array<std::uint16_t, 4> values{0x3f80, 0x4000, 0x4040,
                                                0x4080};
  return values.at(rank);
}

struct Validation {
  std::uint64_t output_mismatches{};
  std::uint64_t input_guard_corruptions{};
  std::uint64_t output_guard_corruptions{};
};

Validation validate(std::uint8_t* guarded_input, std::uint8_t* guarded_output,
                    std::uint32_t rank, std::uint64_t payload_bytes) {
  const auto total = payload_bytes + 2U * kGuardBytes;
  std::vector<std::uint8_t> input(total);
  std::vector<std::uint8_t> output(total);
  check_cuda(cudaMemcpy(input.data(), guarded_input, total,
                        cudaMemcpyDeviceToHost),
             "copy guarded input");
  check_cuda(cudaMemcpy(output.data(), guarded_output, total,
                        cudaMemcpyDeviceToHost),
             "copy guarded output");
  Validation result{};
  for (std::uint64_t index = 0; index < kGuardBytes; ++index) {
    result.input_guard_corruptions += input[index] != kInputGuard;
    result.output_guard_corruptions += output[index] != kOutputGuard;
    result.input_guard_corruptions +=
        input[kGuardBytes + payload_bytes + index] !=
        kInputGuard;
    result.output_guard_corruptions +=
        output[kGuardBytes + payload_bytes + index] !=
        kOutputGuard;
  }
  const auto* input_words = reinterpret_cast<const std::uint16_t*>(
      input.data() + kGuardBytes);
  const auto* output_words = reinterpret_cast<const std::uint16_t*>(
      output.data() + kGuardBytes);
  const std::size_t words = payload_bytes / 2U;
  for (std::size_t index = 0; index < words; ++index) {
    result.output_mismatches += output_words[index] != kExpectedBf16Ten;
    if (input_words[index] != input_value(rank)) {
      ++result.input_guard_corruptions;
    }
  }
  return result;
}

const char* mode_name(Mode mode) {
  return mode == Mode::kCorrectness ? "correctness" : "steady";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::uint16_t rail_count =
        options.rail_mode == RailMode::kDual ? 2U : 1U;
    check_cuda(cudaSetDevice(0), "select CUDA device");
    const auto geometry = research::make_bidirectional_ring_geometry(
        options.query_rows, research::kBidirectionalRingElementsPerRow);
    const auto layout = spark_transport::make_tp4_tiled_pool_layout(
        geometry.tile_bytes,
        research::kBidirectionalRingSlotsPerDirection, 1);

    auto arena0 = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped, layout.total_bytes);
    auto arena1 = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped, layout.total_bytes);
    arena0->fill_from_cpu(0);
    arena1->fill_from_cpu(0);

    const auto plan0 = spark_transport::make_tp4_round_plan(options.rank, 0);
    const auto plan1 = spark_transport::make_tp4_round_plan(options.rank, 1);
    auto channel0 = open_channel(plan0, options.peer0, options.port0);
    spark_transport::VerbsEndpoint endpoint0(options.device0, 1, options.gid0,
                                              *arena0);
    connect_endpoint(channel0, endpoint0, plan0, layout, geometry, options.rank,
                     rail_count, 0);
    auto channel1 = open_channel(plan1, options.peer1, options.port1);
    spark_transport::VerbsEndpoint endpoint1(options.device1, 1, options.gid1,
                                              *arena1);
    connect_endpoint(channel1, endpoint1, plan1, layout, geometry, options.rank,
                     rail_count, 0);
    std::optional<spark_transport::ControlChannel> secondary_channel0;
    std::optional<spark_transport::ControlChannel> secondary_channel1;
    std::unique_ptr<spark_transport::VerbsEndpoint> secondary_endpoint0;
    std::unique_ptr<spark_transport::VerbsEndpoint> secondary_endpoint1;
    if (rail_count == 2) {
      secondary_endpoint0 = std::make_unique<spark_transport::VerbsEndpoint>(
          options.secondary_device0, 1, options.secondary_gid0, *arena0);
      require_dual_mtu_4096(endpoint0, *secondary_endpoint0);
      secondary_channel0.emplace(
          open_channel(plan0, options.secondary_peer0,
                       options.secondary_port0));
      connect_endpoint(*secondary_channel0, *secondary_endpoint0, plan0,
                       layout, geometry, options.rank, rail_count, 1);
      secondary_endpoint1 = std::make_unique<spark_transport::VerbsEndpoint>(
          options.secondary_device1, 1, options.secondary_gid1, *arena1);
      require_dual_mtu_4096(endpoint1, *secondary_endpoint1);
      secondary_channel1.emplace(
          open_channel(plan1, options.secondary_peer1,
                       options.secondary_port1));
      connect_endpoint(*secondary_channel1, *secondary_endpoint1, plan1,
                       layout, geometry, options.rank, rail_count, 1);
    }

    const std::uint64_t allocation_bytes =
        geometry.payload_bytes + 2U * kGuardBytes;
    std::uint8_t* guarded_input{};
    std::uint8_t* guarded_output{};
    cudaStream_t stream{};
    check_cuda(cudaMalloc(&guarded_input, allocation_bytes),
               "allocate guarded input");
    check_cuda(cudaMalloc(&guarded_output, allocation_bytes),
               "allocate guarded output");
    check_cuda(cudaMemset(guarded_input, kInputGuard, allocation_bytes),
               "initialize input guards");
    check_cuda(cudaMemset(guarded_output, kOutputGuard, allocation_bytes),
               "initialize output guards");
    const std::size_t words = geometry.payload_bytes / 2U;
    std::vector<std::uint16_t> host_input(words, input_value(options.rank));
    auto* input = guarded_input + kGuardBytes;
    auto* output = guarded_output + kGuardBytes;
    check_cuda(cudaMemcpy(input, host_input.data(),
                          geometry.payload_bytes,
                          cudaMemcpyHostToDevice),
               "upload exact BF16 input");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create bidirectional stream");

    CudaBulkPort bulk(
        input, output,
        {static_cast<std::uint8_t*>(arena0->device_data()),
         static_cast<std::uint8_t*>(arena1->device_data())},
        options.rank, layout, geometry, stream, rail_count);
    VerbsEdgePort edge(
        {{{&endpoint0, secondary_endpoint0.get()},
          {&endpoint1, secondary_endpoint1.get()}}},
        {arena0.get(), arena1.get()}, layout, rail_count);
    edge.set_direction_endpoints(options.rank);
    research::BidirectionalRingExecutor executor(options.rank, geometry, bulk,
                                                  edge);

    channel0.barrier();
    channel1.barrier();
    if (rail_count == 2) {
      secondary_channel0->barrier();
      secondary_channel1->barrier();
    }
    const std::uint64_t total =
        static_cast<std::uint64_t>(options.warmup) + options.iterations;
    std::vector<double> measured;
    measured.reserve(options.iterations);
    research::BidirectionalRingStatus final_status{};
    for (std::uint64_t operation = 0; operation < total; ++operation) {
      executor.begin();
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(options.timeout_seconds);
      const double start = now_us();
      while (true) {
        const auto progress = executor.advance();
        if (progress.poisoned || executor.status().poisoned) {
          throw std::runtime_error("bidirectional executor poisoned");
        }
        const auto drain = executor.drain();
        if (drain == research::BidirectionalRingDrainState::kComplete) break;
        if (drain == research::BidirectionalRingDrainState::kPoisoned) {
          throw std::runtime_error("bidirectional drain poisoned");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("bidirectional operation timed out");
        }
        std::this_thread::yield();
      }
      const double stop = now_us();
      final_status = executor.status();
      if (!final_status.safe_to_release_registered_storage) {
        throw std::runtime_error("bidirectional operation did not fully retire");
      }
      if (operation >= options.warmup) measured.push_back(stop - start);
      if (options.mode == Mode::kCorrectness) {
        check_cuda(cudaStreamSynchronize(stream),
                   "synchronize correctness operation");
        const auto receipt = validate(guarded_input, guarded_output,
                                      options.rank, geometry.payload_bytes);
        if (receipt.output_mismatches != 0 ||
            receipt.input_guard_corruptions != 0 ||
            receipt.output_guard_corruptions != 0) {
          throw std::runtime_error("bidirectional exact-output gate failed");
        }
      }
    }

    if (!edge.drain_all()) {
      throw std::runtime_error("bidirectional verbs drain failed");
    }
    check_cuda(cudaStreamSynchronize(stream), "final bidirectional synchronize");
    const auto validation = validate(guarded_input, guarded_output,
                                     options.rank, geometry.payload_bytes);
    if (validation.output_mismatches != 0 ||
        validation.input_guard_corruptions != 0 ||
        validation.output_guard_corruptions != 0) {
      throw std::runtime_error("bidirectional final correctness gate failed");
    }

    const auto timing = spark_transport::summarize_latencies(measured);
    const auto& payload = edge.payload_bytes();
    const auto& doorbells = edge.doorbell_bytes();
    const auto& credits = edge.credit_bytes();
    const auto& cq = edge.cq_completions();
    const std::uint64_t expected_payload_per_endpoint =
        total * geometry.bytes_per_direction;
    if (payload[0][0] + payload[0][1] != expected_payload_per_endpoint ||
        payload[1][0] + payload[1][1] != expected_payload_per_endpoint) {
      throw std::runtime_error("bidirectional endpoint byte accounting mismatch");
    }

    std::cout << std::setprecision(12)
              << "TP4_BIDIRECTIONAL_PREFILL_RECEIPT {"
              << "\"schema\":\"sparkring-tp4-bidirectional-prefill/v1\""
              << ",\"rank\":" << options.rank
              << ",\"mode\":\"" << mode_name(options.mode) << "\""
              << ",\"rail_mode\":\""
              << (rail_count == 2 ? "dual" : "single") << "\""
              << ",\"rail_count\":" << rail_count
              << ",\"query_rows\":" << geometry.query_rows
              << ",\"elements_per_row\":" << geometry.elements_per_row
              << ",\"payload_bytes\":" << geometry.payload_bytes
              << ",\"tile_bytes\":" << geometry.tile_bytes
              << ",\"warmup_operations\":" << options.warmup
              << ",\"measured_operations\":" << options.iterations
              << ",\"passed\":true"
              << ",\"output_mismatches\":" << validation.output_mismatches
              << ",\"input_guard_corruptions\":"
              << validation.input_guard_corruptions
              << ",\"output_guard_corruptions\":"
              << validation.output_guard_corruptions
              << ",\"fully_retired\":"
              << (final_status.fully_retired ? "true" : "false")
              << ",\"host_wall_us_min\":" << timing.minimum_us
              << ",\"host_wall_us_p50\":" << timing.p50_us
              << ",\"host_wall_us_p95\":" << timing.p95_us
              << ",\"host_wall_us_mean\":" << timing.mean_us
              << ",\"endpoint0_rail0_payload_bytes\":" << payload[0][0]
              << ",\"endpoint0_rail1_payload_bytes\":" << payload[0][1]
              << ",\"endpoint1_rail0_payload_bytes\":" << payload[1][0]
              << ",\"endpoint1_rail1_payload_bytes\":" << payload[1][1]
              << ",\"endpoint0_rail0_doorbell_bytes\":" << doorbells[0][0]
              << ",\"endpoint0_rail1_doorbell_bytes\":" << doorbells[0][1]
              << ",\"endpoint1_rail0_doorbell_bytes\":" << doorbells[1][0]
              << ",\"endpoint1_rail1_doorbell_bytes\":" << doorbells[1][1]
              << ",\"endpoint0_rail0_credit_bytes\":" << credits[0][0]
              << ",\"endpoint0_rail1_credit_bytes\":" << credits[0][1]
              << ",\"endpoint1_rail0_credit_bytes\":" << credits[1][0]
              << ",\"endpoint1_rail1_credit_bytes\":" << credits[1][1]
              << ",\"endpoint0_payload_bytes\":"
              << payload[0][0] + payload[0][1]
              << ",\"endpoint1_payload_bytes\":"
              << payload[1][0] + payload[1][1]
              << ",\"endpoint0_doorbell_bytes\":"
              << doorbells[0][0] + doorbells[0][1]
              << ",\"endpoint1_doorbell_bytes\":"
              << doorbells[1][0] + doorbells[1][1]
              << ",\"endpoint0_credit_bytes\":"
              << credits[0][0] + credits[0][1]
              << ",\"endpoint1_credit_bytes\":"
              << credits[1][0] + credits[1][1]
              << ",\"endpoint0_rail0_cqes\":" << cq[0][0]
              << ",\"endpoint0_rail1_cqes\":" << cq[0][1]
              << ",\"endpoint1_rail0_cqes\":" << cq[1][0]
              << ",\"endpoint1_rail1_cqes\":" << cq[1][1]
              << ",\"executor_clockwise_payload_bytes\":"
              << final_status.transmitted_bytes[0]
              << ",\"executor_counterclockwise_payload_bytes\":"
              << final_status.transmitted_bytes[1]
              << "}\n";

    cudaStreamDestroy(stream);
    cudaFree(guarded_output);
    cudaFree(guarded_input);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
