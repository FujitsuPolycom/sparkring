#include "spark_transport/tp4_bidirectional_prefill_session.hpp"

#include "../experiments/tiled_prefill/bidirectional_bulk_kernels.cuh"
#include "../experiments/tiled_prefill/bidirectional_ring_executor.hpp"
#include "spark_transport/control_channel.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace spark_transport {
namespace research = tiled_prefill_research;
namespace {

constexpr std::uint16_t kEndpointVersion = 7;
constexpr std::uint16_t kEndpointTag = 0x4250;  // "BP"
constexpr std::uint32_t kGeometryMagic = 0x42505247;  // "BPRG"
constexpr std::uint16_t kGeometryVersion = 1;

struct WireGeometry {
  std::uint32_t magic{kGeometryMagic};
  std::uint16_t version{kGeometryVersion};
  std::uint16_t reserved{};
  std::uint32_t rank{};
  std::uint32_t peer_rank{};
  std::uint32_t world_size{kTp4PrefillRankCount};
  std::uint32_t query_rows{};
  std::uint32_t elements_per_row{};
  std::uint32_t tiles_per_shard{research::kBidirectionalRingTilesPerShard};
  std::uint32_t slots_per_direction{
      research::kBidirectionalRingSlotsPerDirection};
  std::uint32_t lanes_per_slot{1};
  std::uint32_t rail_count{1};
  std::uint32_t rail_index{};
  std::uint64_t payload_bytes{};
  std::uint64_t tile_bytes{};
  std::uint64_t slot_stride{};
  std::uint64_t arena_bytes{};
};
static_assert(std::is_trivially_copyable_v<WireGeometry>);

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

std::size_t endpoint_index(Tp4PrefillEndpoint endpoint) noexcept {
  return endpoint == Tp4PrefillEndpoint::kXor1 ? 0U : 1U;
}

std::size_t direction_index(Tp4PrefillDirection direction) noexcept {
  return direction == Tp4PrefillDirection::kClockwise ? 0U : 1U;
}

std::uint64_t load_acquire(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_release(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

ControlChannel open_channel(const Tp4RoundPlan& plan,
                            const std::string& peer,
                            std::uint16_t port) {
  return plan.server ? ControlChannel::listen_and_accept(port)
                     : ControlChannel::connect(peer, port);
}

void connect_endpoint(ControlChannel& channel, VerbsEndpoint& endpoint,
                      const Tp4RoundPlan& plan,
                      const Tp4TiledPoolLayout& layout,
                      const research::BidirectionalRingGeometry& geometry,
                      std::uint32_t rank, std::uint32_t rail_count = 1,
                      std::uint32_t rail_index = 0) {
  auto local = endpoint.local_info();
  local.version = kEndpointVersion;
  local.reserved = kEndpointTag;
  const auto remote = channel.exchange(local);
  if (remote.version != local.version || remote.reserved != local.reserved ||
      remote.buffer_bytes != local.buffer_bytes) {
    throw std::runtime_error("bidirectional prefill endpoint mismatch");
  }
  const WireGeometry wire{kGeometryMagic,
                          kGeometryVersion,
                          0,
                          rank,
                          plan.peer_rank,
                          kTp4PrefillRankCount,
                          geometry.query_rows,
                          geometry.elements_per_row,
                          research::kBidirectionalRingTilesPerShard,
                          research::kBidirectionalRingSlotsPerDirection,
                          1,
                          rail_count,
                          rail_index,
                          geometry.payload_bytes,
                          geometry.tile_bytes,
                          layout.slot_stride,
                          layout.total_bytes};
  const auto peer = channel.exchange(wire);
  if (peer.magic != wire.magic || peer.version != wire.version ||
      peer.reserved != 0 || peer.rank != plan.peer_rank ||
      peer.peer_rank != rank || peer.world_size != wire.world_size ||
      peer.query_rows != wire.query_rows ||
      peer.elements_per_row != wire.elements_per_row ||
      peer.tiles_per_shard != wire.tiles_per_shard ||
      peer.slots_per_direction != wire.slots_per_direction ||
      peer.lanes_per_slot != 1 || peer.payload_bytes != wire.payload_bytes ||
      peer.rail_count != wire.rail_count ||
      peer.rail_index != wire.rail_index ||
      peer.tile_bytes != wire.tile_bytes ||
      peer.slot_stride != wire.slot_stride ||
      peer.arena_bytes != wire.arena_bytes) {
    throw std::runtime_error("bidirectional prefill geometry mismatch");
  }
  endpoint.connect(remote, local.version);
}

class CudaBulkPort final : public research::BidirectionalRingBulkPort {
 public:
  CudaBulkPort(std::array<std::uint8_t*, 2> endpoints,
               std::uint32_t rank, Tp4TiledPoolLayout layout,
               research::BidirectionalRingGeometry geometry,
               std::uint32_t rail_count)
      : endpoints_(endpoints), rank_(rank), layout_(layout),
        geometry_(geometry), rail_count_(rail_count) {
    check_cuda(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
               "create bidirectional prefill event");
  }

  ~CudaBulkPort() override {
    if (completion_ != nullptr) (void)cudaEventDestroy(completion_);
  }

  void bind(const void* input, void* output, cudaStream_t stream) {
    if (pending_ || input == nullptr || output == nullptr) {
      throw std::logic_error("invalid bidirectional prefill bulk binding");
    }
    input_ = static_cast<const std::uint8_t*>(input);
    output_ = static_cast<std::uint8_t*>(output);
    stream_ = stream;
  }

  research::RingSubmitState try_submit(
      const research::BidirectionalRingBulkRequest& request) override {
    if (pending_) return research::RingSubmitState::kBackpressured;
    if (input_ == nullptr || output_ == nullptr ||
        request.active_bytes != geometry_.tile_bytes) {
      return research::RingSubmitState::kFatal;
    }
    const bool initial =
        request.action == research::RingBulkAction::kStageInitial;
    const bool finish =
        request.action == research::RingBulkAction::kGatherFinish;
    const std::uint32_t consumed_stage =
        initial ? 0U : research::bidirectional_ring_consumed_stage(
                           request.next_exchange_stage);
    std::uint64_t send_offset{};
    std::uint64_t receive_offset{};
    if (!finish) {
      send_offset = tp4_tiled_slot_region(
                        layout_, request.destination_ticket.slot, 0)
                        .send_offset;
    }
    if (!initial) {
      receive_offset = tp4_tiled_slot_region(
                           layout_, request.source_ticket.slot, 0)
                           .receive_offset;
    }
    const auto descriptor =
        initial
            ? research::make_bidirectional_stage_initial_descriptor(
                  rank_, request.direction, request.tile_in_shard,
                  send_offset, geometry_.query_rows)
            : research::make_bidirectional_post_exchange_descriptor(
                  rank_, request.direction, consumed_stage,
                  request.tile_in_shard, send_offset, receive_offset,
                  request.incoming_doorbell_offset,
                  request.consumed_doorbell_token, geometry_.query_rows,
                  rail_count_ == 2
                      ? request.secondary_incoming_doorbell_offset
                      : 0U,
                  rail_count_ == 2
                      ? request.secondary_consumed_doorbell_token
                      : 0U);
    if (descriptor.tensor_offset_bytes != request.operation_offset_bytes ||
        descriptor.shard != request.shard ||
        descriptor.active_bytes != request.active_bytes) {
      return research::RingSubmitState::kFatal;
    }
    const auto outgoing = endpoint_index(request.outgoing_endpoint);
    const auto incoming = endpoint_index(request.incoming_endpoint);
    cudaError_t launched = cudaSuccess;
    switch (request.action) {
      case research::RingBulkAction::kStageInitial:
        launched = research::launch_bidirectional_stage_initial(
            input_, endpoints_[outgoing], descriptor, stream_,
            geometry_.query_rows);
        break;
      case research::RingBulkAction::kReduceForward:
        launched = research::launch_bidirectional_reduce_forward(
            input_, endpoints_[incoming], endpoints_[outgoing], descriptor,
            stream_, geometry_.query_rows);
        break;
      case research::RingBulkAction::kReduceFinalizeAndSeedGather:
        launched = research::launch_bidirectional_reduce_finalize_seed_gather(
            input_, endpoints_[incoming], endpoints_[outgoing], output_,
            descriptor, stream_, geometry_.query_rows);
        break;
      case research::RingBulkAction::kGatherForward:
        launched = research::launch_bidirectional_gather_forward(
            endpoints_[incoming], endpoints_[outgoing], output_, descriptor,
            stream_, geometry_.query_rows);
        break;
      case research::RingBulkAction::kGatherFinish:
        launched = research::launch_bidirectional_gather_finish(
            endpoints_[incoming], output_, descriptor, stream_,
            geometry_.query_rows);
        break;
    }
    check_cuda(launched, "launch bidirectional prefill bulk");
    check_cuda(cudaEventRecord(completion_, stream_),
               "record bidirectional prefill bulk");
    pending_ = true;
    action_ = request.action;
    direction_ = request.direction;
    stage_ = request.next_exchange_stage;
    tile_ = request.tile_in_shard;
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll(
      const research::BidirectionalRingBulkRequest& request) override {
    if (!pending_ || action_ != request.action ||
        direction_ != request.direction ||
        stage_ != request.next_exchange_stage ||
        tile_ != request.tile_in_shard) {
      return research::RingPollState::kFatal;
    }
    const auto result = cudaEventQuery(completion_);
    if (result == cudaErrorNotReady) return research::RingPollState::kPending;
    if (result != cudaSuccess) return research::RingPollState::kFatal;
    pending_ = false;
    return research::RingPollState::kComplete;
  }

 private:
  std::array<std::uint8_t*, 2> endpoints_{};
  std::uint32_t rank_{};
  Tp4TiledPoolLayout layout_{};
  research::BidirectionalRingGeometry geometry_{};
  std::uint32_t rail_count_{1};
  const std::uint8_t* input_{};
  std::uint8_t* output_{};
  cudaStream_t stream_{};
  cudaEvent_t completion_{};
  bool pending_{};
  research::RingBulkAction action_{};
  Tp4PrefillDirection direction_{};
  std::uint32_t stage_{};
  std::uint32_t tile_{};
};

class EdgePort final : public research::BidirectionalRingEdgePort {
 public:
  EdgePort(std::array<VerbsEndpoint*, 2> endpoints,
           std::array<MemoryBuffer*, 2> arenas, Tp4TiledPoolLayout layout,
           std::uint32_t rank)
      : endpoints_(endpoints), arenas_(arenas), layout_(layout) {
    outgoing_[0] = endpoint_index(tp4_prefill_outgoing_endpoint(
        rank, Tp4PrefillDirection::kClockwise));
    outgoing_[1] = endpoint_index(tp4_prefill_outgoing_endpoint(
        rank, Tp4PrefillDirection::kCounterClockwise));
  }

  research::RingSubmitState try_post_exchange(
      const research::BidirectionalRingExchangeRequest& request) override {
    if (!poll_cqs()) return research::RingSubmitState::kFatal;
    const auto direction = direction_index(request.direction);
    const auto endpoint = endpoint_index(request.outgoing_endpoint);
    const auto region = tp4_tiled_slot_region(layout_, request.ticket.slot, 0);
    if (exchanges_[direction].count(request.work_id) != 0 ||
        request.span.local_send_offset != region.send_offset ||
        request.span.remote_receive_offset != region.receive_offset ||
        request.span.active_bytes != layout_.tile_payload_bytes) {
      return research::RingSubmitState::kFatal;
    }
    const auto local_doorbell =
        region.control_offset + offsetof(DoorbellControl, command_sequence);
    const auto remote_doorbell =
        region.control_offset + offsetof(DoorbellControl, remote_sequence);
    const auto completion = next_completion(endpoint);
    exchanges_[direction].emplace(request.work_id, completion);
    pending_[endpoint].insert(completion);
    store_release(word(endpoint, local_doorbell), request.doorbell_token);
    endpoints_[endpoint]->write(region.send_offset, region.receive_offset,
                                request.span.active_bytes, completion, false);
    endpoints_[endpoint]->write(local_doorbell, remote_doorbell,
                                sizeof(request.doorbell_token), completion,
                                true);
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_exchange(
      const research::BidirectionalRingExchangeRequest& request) override {
    if (!poll_cqs()) return research::RingPollState::kFatal;
    const auto direction = direction_index(request.direction);
    const auto found = exchanges_[direction].find(request.work_id);
    if (found == exchanges_[direction].end()) {
      return research::RingPollState::kFatal;
    }
    const auto incoming = endpoint_index(request.incoming_endpoint);
    const auto outgoing = endpoint_index(request.outgoing_endpoint);
    const auto region = tp4_tiled_slot_region(layout_, request.ticket.slot, 0);
    const auto observed = load_acquire(word(
        incoming, region.control_offset +
                      offsetof(DoorbellControl, remote_sequence)));
    if (observed > request.doorbell_token) {
      fatal_ = true;
      return research::RingPollState::kFatal;
    }
    if (observed < request.doorbell_token ||
        pending_[outgoing].count(found->second) != 0) {
      return research::RingPollState::kPending;
    }
    exchanges_[direction].erase(found);
    return research::RingPollState::kComplete;
  }

  research::RingSubmitState try_publish_consumed_through(
      const research::BidirectionalRingCreditRequest& request) override {
    if (!poll_cqs()) return research::RingSubmitState::kFatal;
    const auto direction = direction_index(request.direction);
    if (credits_[direction].active) {
      return research::RingSubmitState::kBackpressured;
    }
    const auto endpoint = endpoint_index(request.endpoint);
    const auto region = tp4_tiled_slot_region(layout_, 0, 0);
    const auto local =
        region.control_offset + offsetof(DoorbellControl, consumer_sequence);
    const auto remote = region.control_offset +
                        offsetof(DoorbellControl, acknowledgement_sequence);
    const auto completion = next_completion(endpoint);
    credits_[direction] = {true, request.work_id, completion};
    pending_[endpoint].insert(completion);
    store_release(word(endpoint, local), request.wire_credit);
    endpoints_[endpoint]->write(local, remote, sizeof(request.wire_credit),
                                completion, true);
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_published_consumed_through(
      const research::BidirectionalRingCreditRequest& request) override {
    if (!poll_cqs()) return research::RingPollState::kFatal;
    auto& credit = credits_[direction_index(request.direction)];
    if (!credit.active || credit.logical != request.work_id) {
      return research::RingPollState::kFatal;
    }
    if (pending_[endpoint_index(request.endpoint)].count(credit.completion)) {
      return research::RingPollState::kPending;
    }
    credit = {};
    return research::RingPollState::kComplete;
  }

  research::RingCreditPollState poll_peer_consumed_through(
      Tp4PrefillDirection direction, std::uint64_t& wire_credit) override {
    if (!poll_cqs()) return research::RingCreditPollState::kFatal;
    const auto index = direction_index(direction);
    const auto region = tp4_tiled_slot_region(layout_, 0, 0);
    wire_credit = load_acquire(word(
        outgoing_[index], region.control_offset +
                              offsetof(DoorbellControl,
                                       acknowledgement_sequence)));
    if (wire_credit == 0 || wire_credit == peer_credit_[index]) {
      return research::RingCreditPollState::kNoUpdate;
    }
    peer_credit_[index] = wire_credit;
    return research::RingCreditPollState::kUpdate;
  }

  bool drain(std::chrono::steady_clock::time_point deadline) noexcept {
    while ((!pending_[0].empty() || !pending_[1].empty()) && !fatal_) {
      if (!poll_cqs() || std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::yield();
    }
    return !fatal_ && exchanges_[0].empty() && exchanges_[1].empty() &&
           !credits_[0].active && !credits_[1].active;
  }

 private:
  struct Credit {
    bool active{};
    std::uint64_t logical{};
    std::uint64_t completion{};
  };

  std::uint64_t next_completion(std::size_t endpoint) {
    if (next_completion_[endpoint] ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("bidirectional prefill CQ ID exhausted");
    }
    return next_completion_[endpoint]++;
  }

  std::uint64_t* word(std::size_t endpoint, std::uint64_t offset) {
    if (offset + sizeof(std::uint64_t) > arenas_[endpoint]->size() ||
        offset % alignof(std::uint64_t) != 0) {
      throw std::out_of_range("bidirectional prefill control offset");
    }
    return reinterpret_cast<std::uint64_t*>(
        static_cast<std::uint8_t*>(arenas_[endpoint]->host_data()) + offset);
  }

  bool poll_cqs() noexcept {
    if (fatal_) return false;
    try {
      for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
        SendCompletion completions[32]{};
        const auto count =
            endpoints_[endpoint]->poll_send_completions(completions, 32);
        for (std::size_t index = 0; index < count; ++index) {
          if (pending_[endpoint].erase(completions[index].work_id) != 1) {
            fatal_ = true;
            return false;
          }
        }
      }
      return true;
    } catch (...) {
      fatal_ = true;
      return false;
    }
  }

  std::array<VerbsEndpoint*, 2> endpoints_{};
  std::array<MemoryBuffer*, 2> arenas_{};
  Tp4TiledPoolLayout layout_{};
  std::array<std::size_t, 2> outgoing_{};
  std::array<std::uint64_t, 2> next_completion_{1, 1};
  std::array<std::unordered_set<std::uint64_t>, 2> pending_{};
  std::array<std::unordered_map<std::uint64_t, std::uint64_t>, 2> exchanges_{};
  std::array<Credit, 2> credits_{};
  std::array<std::uint64_t, 2> peer_credit_{};
  bool fatal_{};
};

#include "tp4_bidirectional_prefill_dual_edge.inc"

}  // namespace

class Tp4BidirectionalPrefillSession::Impl {
 public:
  explicit Impl(Tp4BidirectionalPrefillOptions options)
      : options_(std::move(options)),
        geometry_(research::make_bidirectional_ring_geometry(
            options_.query_rows, options_.elements_per_row)),
        layout_(make_tp4_tiled_pool_layout(
            geometry_.tile_bytes,
            research::kBidirectionalRingSlotsPerDirection, 1)) {
    if (options_.rank >= kTp4PrefillRankCount || options_.peer0.empty() ||
        options_.peer1.empty() || options_.device0.empty() ||
        options_.device1.empty() || options_.control_port0 == 0 ||
        options_.control_port1 == 0 ||
        options_.control_port0 == options_.control_port1 ||
        (options_.rail_count != 1 && options_.rail_count != 2) ||
        options_.timeout_seconds == 0) {
      throw std::invalid_argument("invalid bidirectional prefill options");
    }
    const auto plan0 = make_tp4_round_plan(options_.rank, 0);
    const auto plan1 = make_tp4_round_plan(options_.rank, 1);
    channel0_.emplace(open_channel(plan0, options_.peer0,
                                   options_.control_port0));
    arena0_ = MemoryBuffer::allocate(MemoryKind::kCudaMapped,
                                     layout_.total_bytes);
    arena0_->fill_from_cpu(0);
    endpoint0_ = std::make_unique<VerbsEndpoint>(
        options_.device0, 1, options_.gid0, *arena0_);
    connect_endpoint(*channel0_, *endpoint0_, plan0, layout_, geometry_,
                     options_.rank, options_.rail_count, 0);
    if (endpoint0_->active_mtu_bytes() != 4096 ||
        endpoint0_->maximum_send_work_requests() < 3) {
      throw std::runtime_error("primary rail-0 MTU/SQ preflight failed");
    }
    channel1_.emplace(open_channel(plan1, options_.peer1,
                                   options_.control_port1));
    arena1_ = MemoryBuffer::allocate(MemoryKind::kCudaMapped,
                                     layout_.total_bytes);
    arena1_->fill_from_cpu(0);
    endpoint1_ = std::make_unique<VerbsEndpoint>(
        options_.device1, 1, options_.gid1, *arena1_);
    connect_endpoint(*channel1_, *endpoint1_, plan1, layout_, geometry_,
                     options_.rank, options_.rail_count, 0);
    if (endpoint1_->active_mtu_bytes() != 4096 ||
        endpoint1_->maximum_send_work_requests() < 3) {
      throw std::runtime_error("primary rail-1 MTU/SQ preflight failed");
    }
    if (options_.rail_count == 2) {
      if (options_.secondary_peer0.empty() ||
          options_.secondary_peer1.empty() ||
          options_.secondary_device0.empty() ||
          options_.secondary_device1.empty() ||
          options_.secondary_control_port0 == 0 ||
          options_.secondary_control_port1 == 0 ||
          options_.secondary_control_port0 == options_.control_port0 ||
          options_.secondary_control_port1 == options_.control_port1 ||
          options_.secondary_control_port0 == options_.control_port1 ||
          options_.secondary_control_port1 == options_.control_port0 ||
          options_.secondary_control_port0 ==
              options_.secondary_control_port1 ||
          options_.secondary_peer0 == options_.peer0 ||
          options_.secondary_peer1 == options_.peer1 ||
          options_.secondary_device0 == options_.device0 ||
          options_.secondary_device1 == options_.device1) {
        throw std::invalid_argument("invalid secondary rail topology");
      }
      secondary_channel0_.emplace(open_channel(
          plan0, options_.secondary_peer0,
          options_.secondary_control_port0));
      secondary_endpoint0_ = std::make_unique<VerbsEndpoint>(
          options_.secondary_device0, 1, options_.secondary_gid0, *arena0_);
      connect_endpoint(*secondary_channel0_, *secondary_endpoint0_, plan0,
                       layout_, geometry_, options_.rank, 2, 1);
      if (secondary_endpoint0_->active_mtu_bytes() != 4096 ||
          secondary_endpoint0_->maximum_send_work_requests() < 2) {
        throw std::runtime_error("secondary rail-0 MTU/SQ preflight failed");
      }
      secondary_channel1_.emplace(open_channel(
          plan1, options_.secondary_peer1,
          options_.secondary_control_port1));
      secondary_endpoint1_ = std::make_unique<VerbsEndpoint>(
          options_.secondary_device1, 1, options_.secondary_gid1, *arena1_);
      connect_endpoint(*secondary_channel1_, *secondary_endpoint1_, plan1,
                       layout_, geometry_, options_.rank, 2, 1);
      if (secondary_endpoint1_->active_mtu_bytes() != 4096 ||
          secondary_endpoint1_->maximum_send_work_requests() < 2) {
        throw std::runtime_error("secondary rail-1 MTU/SQ preflight failed");
      }
    }
    bulk_ = std::make_unique<CudaBulkPort>(
        std::array<std::uint8_t*, 2>{
            static_cast<std::uint8_t*>(arena0_->device_data()),
            static_cast<std::uint8_t*>(arena1_->device_data())},
        options_.rank, layout_, geometry_, options_.rail_count);
    if (options_.rail_count == 2) {
      auto dual = std::make_unique<DualRailEdgePort>(
          DualRailEdgePort::EndpointMatrix{{
              {endpoint0_.get(), secondary_endpoint0_.get()},
              {endpoint1_.get(), secondary_endpoint1_.get()}}},
          std::array<MemoryBuffer*, 2>{arena0_.get(), arena1_.get()}, layout_,
          2);
      dual->set_direction_endpoints(options_.rank);
      edge_ = std::move(dual);
    } else {
      edge_ = std::make_unique<EdgePort>(
          std::array<VerbsEndpoint*, 2>{endpoint0_.get(), endpoint1_.get()},
          std::array<MemoryBuffer*, 2>{arena0_.get(), arena1_.get()}, layout_,
          options_.rank);
    }
    executor_ = std::make_unique<research::BidirectionalRingExecutor>(
        options_.rank, geometry_, *bulk_, *edge_);
    channel0_->barrier();
    channel1_->barrier();
    if (options_.rail_count == 2) {
      secondary_channel0_->barrier();
      secondary_channel1_->barrier();
    }
  }

  ~Impl() {
    if (edge_) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      (void)drain_edge(deadline);
    }
    executor_.reset();
    bulk_.reset();
    edge_.reset();
  }

  void all_reduce(const void* input, void* output, void* stream_pointer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (input == nullptr || output == nullptr) {
      throw std::invalid_argument("bidirectional prefill tensor is null");
    }
    if (poisoned_) {
      throw std::runtime_error("bidirectional prefill session is poisoned");
    }
    // A null cudaStream_t is CUDA's valid legacy/default stream. The C ABI
    // transports the handle as void*, so it must not be treated as missing.
    auto stream = static_cast<cudaStream_t>(stream_pointer);
    try {
      bulk_->bind(input, output, stream);
      executor_->begin();
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(options_.timeout_seconds);
      while (true) {
        const auto progress = executor_->advance();
        const auto drain = executor_->drain();
        if (progress.poisoned ||
            drain == research::BidirectionalRingDrainState::kPoisoned) {
          throw std::runtime_error("bidirectional prefill transport poisoned");
        }
        if (drain == research::BidirectionalRingDrainState::kComplete) break;
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("bidirectional prefill operation timed out");
        }
        std::this_thread::yield();
      }
    } catch (...) {
      poisoned_ = true;
      // Every launched bulk kernel is finite once its host-observed doorbell
      // is handed to the device acquire gate. Quiesce this call's stream and
      // reap any posted CQ work before stack unwinding can release arenas.
      (void)cudaStreamSynchronize(stream);
      const auto cleanup_deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      (void)drain_edge(cleanup_deadline);
      throw;
    }
  }

 private:
  bool drain_edge(std::chrono::steady_clock::time_point deadline) noexcept {
    if (options_.rail_count == 2) {
      return static_cast<DualRailEdgePort*>(edge_.get())->drain_all();
    }
    return static_cast<EdgePort*>(edge_.get())->drain(deadline);
  }

  Tp4BidirectionalPrefillOptions options_;
  research::BidirectionalRingGeometry geometry_{};
  Tp4TiledPoolLayout layout_{};
  std::optional<ControlChannel> channel0_;
  std::optional<ControlChannel> channel1_;
  std::optional<ControlChannel> secondary_channel0_;
  std::optional<ControlChannel> secondary_channel1_;
  std::unique_ptr<MemoryBuffer> arena0_;
  std::unique_ptr<MemoryBuffer> arena1_;
  std::unique_ptr<VerbsEndpoint> endpoint0_;
  std::unique_ptr<VerbsEndpoint> endpoint1_;
  std::unique_ptr<VerbsEndpoint> secondary_endpoint0_;
  std::unique_ptr<VerbsEndpoint> secondary_endpoint1_;
  std::unique_ptr<CudaBulkPort> bulk_;
  std::unique_ptr<research::BidirectionalRingEdgePort> edge_;
  std::unique_ptr<research::BidirectionalRingExecutor> executor_;
  std::mutex mutex_;
  bool poisoned_{};
};

Tp4BidirectionalPrefillSession::Tp4BidirectionalPrefillSession(
    Tp4BidirectionalPrefillOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Tp4BidirectionalPrefillSession::~Tp4BidirectionalPrefillSession() = default;

void Tp4BidirectionalPrefillSession::all_reduce(
    const void* input, void* output, void* cuda_stream) {
  impl_->all_reduce(input, output, cuda_stream);
}

}  // namespace spark_transport
