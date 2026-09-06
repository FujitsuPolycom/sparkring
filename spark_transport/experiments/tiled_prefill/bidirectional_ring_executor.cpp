#include "bidirectional_ring_executor.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace spark_transport::tiled_prefill_research {
namespace {

constexpr std::array<Tp4PrefillDirection, 2> kDirections{
    Tp4PrefillDirection::kClockwise,
    Tp4PrefillDirection::kCounterClockwise};

constexpr std::size_t direction_index(
    Tp4PrefillDirection direction) noexcept {
  return direction == Tp4PrefillDirection::kClockwise ? 0U : 1U;
}

constexpr std::uint64_t checked_add(std::uint64_t left,
                                    std::uint64_t right) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error("bidirectional ring ordinal exhausted");
  }
  return left + right;
}

enum class TilePhase : std::uint8_t {
  kAwaitingCredit,
  kBulkSubmit,
  kBulkPending,
  kExchangeSubmit,
  kExchangePending,
  kFinalBulkSubmit,
  kFinalBulkPending,
  kOutputReady,
};

struct TileExecution {
  std::uint32_t stage{};
  std::uint32_t tile_in_shard{};
  TilePhase phase{TilePhase::kAwaitingCredit};
};

struct CreditPublication {
  bool submitted{};
  bool completed{};
  std::uint64_t target{};
};

constexpr RingBulkAction preparation_action(std::uint32_t stage) {
  if (stage == 0) {
    return RingBulkAction::kStageInitial;
  }
  if (stage <= 2) {
    return RingBulkAction::kReduceForward;
  }
  if (stage == 3) {
    return RingBulkAction::kReduceFinalizeAndSeedGather;
  }
  if (stage <= 5) {
    return RingBulkAction::kGatherForward;
  }
  throw std::out_of_range("bidirectional ring bulk stage is invalid");
}

}  // namespace

class BidirectionalRingExecutor::Impl {
 public:
  Impl(std::uint32_t rank, BidirectionalRingGeometry geometry,
       BidirectionalRingBulkPort& bulk_port,
       BidirectionalRingEdgePort& edge_port)
      : rank_(rank),
        geometry_(geometry),
        bulk_port_(bulk_port),
        edge_port_(edge_port),
        layout_(make_tp4_tiled_pool_layout(
            geometry_.tile_bytes,
            kBidirectionalRingSlotsPerDirection, 1)),
        credit_windows_{
            Tp4TiledCreditWindow(kBidirectionalRingSlotsPerDirection),
            Tp4TiledCreditWindow(kBidirectionalRingSlotsPerDirection)} {
    if (rank_ >= kTp4PrefillRankCount) {
      throw std::out_of_range("bidirectional ring rank must be in [0, 3]");
    }
    if (!bidirectional_ring_geometry_valid(geometry_)) {
      throw std::invalid_argument("invalid bidirectional ring geometry");
    }
    static_assert(kBidirectionalRingTransfersPerDirection == 24);
  }

  void begin() {
    if (poisoned_) {
      throw std::logic_error(
          "poisoned bidirectional ring executor rejects new work");
    }
    if (active_operation_ && !fully_retired()) {
      throw std::logic_error(
          "bidirectional ring operation is not fully retired");
    }
    for (std::size_t direction = 0; direction < kDirections.size();
         ++direction) {
      if (credit_windows_[direction].exhausted()) {
        throw std::overflow_error(
            "bidirectional ring ticket sequence exhausted");
      }
      first_ordinals_[direction] =
          credit_windows_[direction].next_ordinal();
      for (std::uint32_t tile = 0;
           tile < kBidirectionalRingTilesPerShard; ++tile) {
        tiles_[direction][tile] = TileExecution{0, tile};
      }
      local_consumed_[direction].fill(false);
      local_credit_publications_[direction] = {};
      posted_transfers_[direction] = 0;
      completed_transfers_[direction] = 0;
      transmitted_bytes_[direction] = 0;
    }
    output_ready_tiles_ = 0;
    active_operation_ = true;
  }

  BidirectionalRingAdvanceResult advance() noexcept {
    BidirectionalRingAdvanceResult result{};
    if (poisoned_) {
      result.poisoned = true;
      return result;
    }
    if (!active_operation_) {
      return result;
    }
    try {
      poll_peer_credits(result);
      for (std::size_t direction = 0;
           direction < kDirections.size() && !poisoned_; ++direction) {
        for (auto& tile : tiles_[direction]) {
          progress_tile(direction, tile, result);
          if (poisoned_) {
            break;
          }
        }
      }
      for (std::size_t direction = 0;
           direction < kDirections.size() && !poisoned_; ++direction) {
        progress_credit_publication(direction, result);
      }
    } catch (...) {
      poison(BidirectionalRingFailure::kPortException);
    }
    result.poisoned = poisoned_;
    return result;
  }

  BidirectionalRingStatus status() const noexcept {
    BidirectionalRingStatus result{};
    result.active_operation = active_operation_;
    result.poisoned = poisoned_;
    result.geometry = geometry_;
    result.output_ready_tiles = output_ready_tiles_;
    result.posted_transfers = posted_transfers_;
    result.completed_transfers = completed_transfers_;
    result.transmitted_bytes = transmitted_bytes_;
    result.peer_consumed_through = peer_consumed_through_;
    result.peer_credit_observed = peer_credit_observed_;
    result.local_consumed_through = local_consumed_through_;
    result.local_credit_observed = local_credit_observed_;
    result.failure = failure_;
    result.output_ready = active_operation_ &&
                          output_ready_tiles_ ==
                              2U * kBidirectionalRingTilesPerShard;
    result.fully_retired = result.output_ready && credits_fully_retired();
    result.safe_to_release_registered_storage = result.fully_retired;
    return result;
  }

  BidirectionalRingDrainState drain() noexcept {
    if (poisoned_) {
      return BidirectionalRingDrainState::kPoisoned;
    }
    if (!active_operation_) {
      return BidirectionalRingDrainState::kIdle;
    }
    (void)advance();
    if (poisoned_) {
      return BidirectionalRingDrainState::kPoisoned;
    }
    return fully_retired() ? BidirectionalRingDrainState::kComplete
                           : BidirectionalRingDrainState::kPending;
  }

 private:
  std::uint64_t ordinal(std::size_t direction, std::uint32_t stage,
                        std::uint32_t tile) const {
    return checked_add(first_ordinals_[direction],
                       bidirectional_ring_ordinal(stage, tile));
  }

  Tp4TileTicket ticket(std::size_t direction, std::uint32_t stage,
                       std::uint32_t tile) const {
    return tp4_tiled_ticket_from_ordinal(
        ordinal(direction, stage, tile),
        kBidirectionalRingSlotsPerDirection);
  }

  std::uint64_t operation_offset(Tp4PrefillDirection direction,
                                 std::uint32_t shard,
                                 std::uint32_t tile) const noexcept {
    const std::uint64_t half_offset =
        direction == Tp4PrefillDirection::kClockwise
            ? 0U
            : geometry_.half_bytes;
    return half_offset +
           static_cast<std::uint64_t>(shard) * geometry_.shard_bytes +
           static_cast<std::uint64_t>(tile) *
               geometry_.tile_bytes;
  }

  BidirectionalRingBulkRequest bulk_request(
      std::size_t direction_index_value, const TileExecution& tile,
      bool final) const {
    const auto direction = kDirections[direction_index_value];
    if (final) {
      const auto last = tp4_prefill_stage(
          rank_, direction, kTp4PrefillStageCount - 1U);
      return {RingBulkAction::kGatherFinish,
              direction,
              last.half,
              kTp4PrefillStageCount,
              tile.tile_in_shard,
              last.receive_shard,
              ticket(direction_index_value, kTp4PrefillStageCount - 1U,
                     tile.tile_in_shard),
              {},
              last.incoming_endpoint,
              last.outgoing_endpoint,
              operation_offset(direction, last.receive_shard,
                               tile.tile_in_shard),
              tp4_tiled_slot_region(
                  layout_,
                  ticket(direction_index_value,
                         kTp4PrefillStageCount - 1U,
                         tile.tile_in_shard)
                      .slot,
                  0)
                      .control_offset +
                  offsetof(DoorbellControl, remote_sequence),
              checked_add(
                  ordinal(direction_index_value,
                          kTp4PrefillStageCount - 1U,
                          tile.tile_in_shard),
                  1U),
              tp4_tiled_slot_region(
                  layout_,
                  ticket(direction_index_value,
                         kTp4PrefillStageCount - 1U,
                         tile.tile_in_shard)
                      .slot,
                  0)
                      .control_offset +
                  offsetof(DoorbellControl, reserved),
              checked_add(
                  ordinal(direction_index_value,
                          kTp4PrefillStageCount - 1U,
                          tile.tile_in_shard),
                  1U),
              geometry_.tile_bytes};
    }

    const auto transfer = tp4_prefill_stage(
        rank_, direction, tile.stage);
    const Tp4TileTicket source =
        tile.stage == 0
            ? Tp4TileTicket{}
            : ticket(direction_index_value, tile.stage - 1U,
                     tile.tile_in_shard);
    const std::uint64_t incoming_doorbell_offset =
        tile.stage == 0
            ? 0U
            : tp4_tiled_slot_region(layout_, source.slot, 0)
                      .control_offset +
                  offsetof(DoorbellControl, remote_sequence);
    const std::uint64_t consumed_doorbell_token =
        tile.stage == 0
            ? 0U
            : checked_add(
                  ordinal(direction_index_value,
                          bidirectional_ring_consumed_stage(tile.stage),
                          tile.tile_in_shard),
                  1U);
    return {preparation_action(tile.stage),
            direction,
            transfer.half,
            tile.stage,
            tile.tile_in_shard,
            transfer.send_shard,
            source,
            ticket(direction_index_value, tile.stage,
                   tile.tile_in_shard),
            transfer.incoming_endpoint,
            transfer.outgoing_endpoint,
            operation_offset(direction, transfer.send_shard,
                             tile.tile_in_shard),
            incoming_doorbell_offset,
            consumed_doorbell_token,
            tile.stage == 0
                ? 0U
                : tp4_tiled_slot_region(layout_, source.slot, 0)
                          .control_offset +
                      offsetof(DoorbellControl, reserved),
            consumed_doorbell_token,
            geometry_.tile_bytes};
  }

  BidirectionalRingExchangeRequest exchange_request(
      std::size_t direction_index_value,
      const TileExecution& tile) const {
    const auto direction = kDirections[direction_index_value];
    const auto transfer = tp4_prefill_stage(rank_, direction, tile.stage);
    const auto transfer_ticket = ticket(
        direction_index_value, tile.stage, tile.tile_in_shard);
    const auto region = tp4_tiled_slot_region(
        layout_, transfer_ticket.slot, 0);
    const BidirectionalRingPayloadSpan span{
        region.send_offset, region.receive_offset,
        geometry_.tile_bytes};
    const std::uint64_t value =
        ordinal(direction_index_value, tile.stage, tile.tile_in_shard);
    return {direction,
            tile.stage,
            tile.tile_in_shard,
            transfer.send_shard,
            transfer_ticket,
            transfer.outgoing_endpoint,
            transfer.incoming_endpoint,
            transfer.outgoing_peer,
            transfer.incoming_peer,
            span,
            checked_add(value, 1U),
            checked_add(value, 1U)};
  }

  BidirectionalRingCreditRequest credit_request(
      std::size_t direction) const {
    const std::uint64_t target =
        local_credit_publications_[direction].target;
    return {kDirections[direction],
            tp4_prefill_incoming_endpoint(rank_, kDirections[direction]),
            target,
            checked_add(target, 1U),
            checked_add(target, 1U)};
  }

  void poison(BidirectionalRingFailure failure) noexcept {
    poisoned_ = true;
    if (failure_ == BidirectionalRingFailure::kNone) {
      failure_ = failure;
    }
  }

  void mark_local_consumed(std::size_t direction,
                           std::uint64_t consumed_ordinal) {
    if (consumed_ordinal < first_ordinals_[direction] ||
        consumed_ordinal >=
            first_ordinals_[direction] +
                kBidirectionalRingTransfersPerDirection) {
      poison(BidirectionalRingFailure::kCreditProtocol);
      return;
    }
    const std::size_t relative = static_cast<std::size_t>(
        consumed_ordinal - first_ordinals_[direction]);
    local_consumed_[direction][relative] = true;

    std::size_t contiguous = 0;
    while (contiguous < local_consumed_[direction].size() &&
           local_consumed_[direction][contiguous]) {
      ++contiguous;
    }
    if (contiguous == 0) {
      return;
    }
    const std::uint64_t watermark =
        first_ordinals_[direction] + contiguous - 1U;
    if (!local_credit_observed_[direction] ||
        watermark > local_consumed_through_[direction]) {
      local_consumed_through_[direction] = watermark;
      local_credit_observed_[direction] = true;
    }
  }

  void poll_peer_credits(BidirectionalRingAdvanceResult& result) {
    for (std::size_t direction = 0; direction < kDirections.size();
         ++direction) {
      std::uint64_t wire_credit{};
      const auto state = edge_port_.poll_peer_consumed_through(
          kDirections[direction], wire_credit);
      if (state == RingCreditPollState::kFatal) {
        poison(BidirectionalRingFailure::kCreditProtocol);
        return;
      }
      if (state != RingCreditPollState::kUpdate || wire_credit == 0) {
        continue;
      }
      const std::uint64_t watermark = wire_credit - 1U;
      if ((peer_credit_observed_[direction] &&
           watermark < peer_consumed_through_[direction]) ||
          !credit_windows_[direction].has_issued() ||
          watermark > credit_windows_[direction].highest_issued()) {
        poison(BidirectionalRingFailure::kCreditProtocol);
        return;
      }
      const auto update = credit_windows_[direction].observe_consumed_through(
          static_cast<std::uint32_t>(direction), watermark);
      // Each direction has its own window but the reusable helper models two
      // edges. Mirror the same direction-local watermark into both indices.
      const auto mirror = credit_windows_[direction].observe_consumed_through(
          static_cast<std::uint32_t>(1U - direction), watermark);
      if (update == Tp4TiledCreditUpdateState::kPoisoned ||
          mirror == Tp4TiledCreditUpdateState::kPoisoned) {
        poison(BidirectionalRingFailure::kCreditProtocol);
        return;
      }
      peer_consumed_through_[direction] = watermark;
      peer_credit_observed_[direction] = true;
      result.made_progress = true;
    }
  }

  void progress_tile(std::size_t direction, TileExecution& tile,
                     BidirectionalRingAdvanceResult& result) {
    switch (tile.phase) {
      case TilePhase::kAwaitingCredit: {
        const auto next = ordinal(direction, tile.stage,
                                  tile.tile_in_shard);
        if (next != credit_windows_[direction].next_ordinal()) {
          result.backpressured = true;
          return;
        }
        const auto acquired = credit_windows_[direction].try_acquire(
            ticket(direction, tile.stage, tile.tile_in_shard));
        if (acquired == Tp4TiledAcquireState::kWaitingForCredit) {
          result.backpressured = true;
          return;
        }
        if (acquired != Tp4TiledAcquireState::kReady) {
          poison(BidirectionalRingFailure::kTicketProtocol);
          return;
        }
        tile.phase = TilePhase::kBulkSubmit;
        result.made_progress = true;
        return;
      }
      case TilePhase::kBulkSubmit: {
        const auto request = bulk_request(direction, tile, false);
        const auto submitted = bulk_port_.try_submit(request);
        if (submitted == RingSubmitState::kBackpressured) {
          result.backpressured = true;
          return;
        }
        if (submitted == RingSubmitState::kFatal) {
          poison(BidirectionalRingFailure::kBulkSubmit);
          return;
        }
        tile.phase = TilePhase::kBulkPending;
        result.made_progress = true;
        return;
      }
      case TilePhase::kBulkPending: {
        const auto request = bulk_request(direction, tile, false);
        const auto polled = bulk_port_.poll(request);
        if (polled == RingPollState::kPending) {
          return;
        }
        if (polled == RingPollState::kFatal) {
          poison(BidirectionalRingFailure::kBulkPoll);
          return;
        }
        if (tile.stage != 0) {
          mark_local_consumed(
              direction,
              ordinal(direction, tile.stage - 1U,
                      tile.tile_in_shard));
        }
        tile.phase = TilePhase::kExchangeSubmit;
        result.made_progress = true;
        return;
      }
      case TilePhase::kExchangeSubmit: {
        const auto request = exchange_request(direction, tile);
        const auto submitted = edge_port_.try_post_exchange(request);
        if (submitted == RingSubmitState::kBackpressured) {
          result.backpressured = true;
          return;
        }
        if (submitted == RingSubmitState::kFatal) {
          poison(BidirectionalRingFailure::kExchangeSubmit);
          return;
        }
        ++posted_transfers_[direction];
        transmitted_bytes_[direction] += geometry_.tile_bytes;
        tile.phase = TilePhase::kExchangePending;
        result.made_progress = true;
        return;
      }
      case TilePhase::kExchangePending: {
        const auto request = exchange_request(direction, tile);
        const auto polled = edge_port_.poll_exchange(request);
        if (polled == RingPollState::kPending) {
          return;
        }
        if (polled == RingPollState::kFatal) {
          poison(BidirectionalRingFailure::kExchangePoll);
          return;
        }
        ++completed_transfers_[direction];
        if (tile.stage + 1U < kTp4PrefillStageCount) {
          ++tile.stage;
          tile.phase = TilePhase::kAwaitingCredit;
        } else {
          tile.phase = TilePhase::kFinalBulkSubmit;
        }
        result.made_progress = true;
        return;
      }
      case TilePhase::kFinalBulkSubmit: {
        const auto request = bulk_request(direction, tile, true);
        const auto submitted = bulk_port_.try_submit(request);
        if (submitted == RingSubmitState::kBackpressured) {
          result.backpressured = true;
          return;
        }
        if (submitted == RingSubmitState::kFatal) {
          poison(BidirectionalRingFailure::kBulkSubmit);
          return;
        }
        tile.phase = TilePhase::kFinalBulkPending;
        result.made_progress = true;
        return;
      }
      case TilePhase::kFinalBulkPending: {
        const auto request = bulk_request(direction, tile, true);
        const auto polled = bulk_port_.poll(request);
        if (polled == RingPollState::kPending) {
          return;
        }
        if (polled == RingPollState::kFatal) {
          poison(BidirectionalRingFailure::kBulkPoll);
          return;
        }
        mark_local_consumed(
            direction,
            ordinal(direction, kTp4PrefillStageCount - 1U,
                    tile.tile_in_shard));
        tile.phase = TilePhase::kOutputReady;
        ++output_ready_tiles_;
        result.made_progress = true;
        return;
      }
      case TilePhase::kOutputReady:
        return;
    }
  }

  void progress_credit_publication(
      std::size_t direction, BidirectionalRingAdvanceResult& result) {
    auto& publication = local_credit_publications_[direction];
    if (publication.submitted && !publication.completed) {
      const auto polled = edge_port_.poll_published_consumed_through(
          credit_request(direction));
      if (polled == RingPollState::kFatal) {
        poison(BidirectionalRingFailure::kCreditPublish);
        return;
      }
      if (polled == RingPollState::kPending) {
        return;
      }
      publication.completed = true;
      result.made_progress = true;
    }
    if (!local_credit_observed_[direction] ||
        (publication.submitted && !publication.completed) ||
        (publication.completed &&
         publication.target >= local_consumed_through_[direction])) {
      return;
    }
    publication = {false, false, local_consumed_through_[direction]};
    const auto submitted = edge_port_.try_publish_consumed_through(
        credit_request(direction));
    if (submitted == RingSubmitState::kBackpressured) {
      result.backpressured = true;
      return;
    }
    if (submitted == RingSubmitState::kFatal) {
      poison(BidirectionalRingFailure::kCreditPublish);
      return;
    }
    publication.submitted = true;
    result.made_progress = true;
  }

  bool credits_fully_retired() const noexcept {
    for (std::size_t direction = 0; direction < kDirections.size();
         ++direction) {
      const std::uint64_t final_ordinal =
          first_ordinals_[direction] +
          kBidirectionalRingTransfersPerDirection - 1U;
      const auto& publication = local_credit_publications_[direction];
      if (!peer_credit_observed_[direction] ||
          peer_consumed_through_[direction] < final_ordinal ||
          !local_credit_observed_[direction] ||
          local_consumed_through_[direction] < final_ordinal ||
          !publication.submitted || !publication.completed ||
          publication.target < final_ordinal) {
        return false;
      }
    }
    return true;
  }

  bool fully_retired() const noexcept { return status().fully_retired; }

  std::uint32_t rank_{};
  BidirectionalRingGeometry geometry_{};
  BidirectionalRingBulkPort& bulk_port_;
  BidirectionalRingEdgePort& edge_port_;
  Tp4TiledPoolLayout layout_{};
  std::array<Tp4TiledCreditWindow, 2> credit_windows_;
  std::array<std::array<TileExecution, kBidirectionalRingTilesPerShard>, 2>
      tiles_{};
  std::array<std::uint64_t, 2> first_ordinals_{};
  std::array<std::array<bool, kBidirectionalRingTransfersPerDirection>, 2>
      local_consumed_{};
  std::array<CreditPublication, 2> local_credit_publications_{};
  std::array<std::uint32_t, 2> posted_transfers_{};
  std::array<std::uint32_t, 2> completed_transfers_{};
  std::array<std::uint64_t, 2> transmitted_bytes_{};
  std::array<std::uint64_t, 2> peer_consumed_through_{};
  std::array<bool, 2> peer_credit_observed_{};
  std::array<std::uint64_t, 2> local_consumed_through_{};
  std::array<bool, 2> local_credit_observed_{};
  std::uint32_t output_ready_tiles_{};
  bool active_operation_{};
  bool poisoned_{};
  BidirectionalRingFailure failure_{BidirectionalRingFailure::kNone};
};

BidirectionalRingExecutor::BidirectionalRingExecutor(
    std::uint32_t rank, BidirectionalRingBulkPort& bulk_port,
    BidirectionalRingEdgePort& edge_port)
    : BidirectionalRingExecutor(
          rank, make_bidirectional_ring_geometry(kTp4PrefillQ2048Rows),
          bulk_port, edge_port) {}

BidirectionalRingExecutor::BidirectionalRingExecutor(
    std::uint32_t rank, BidirectionalRingGeometry geometry,
    BidirectionalRingBulkPort& bulk_port,
    BidirectionalRingEdgePort& edge_port)
    : impl_(std::make_unique<Impl>(rank, geometry, bulk_port, edge_port)) {}

BidirectionalRingExecutor::~BidirectionalRingExecutor() = default;

void BidirectionalRingExecutor::begin() { impl_->begin(); }

BidirectionalRingAdvanceResult BidirectionalRingExecutor::advance() noexcept {
  return impl_->advance();
}

BidirectionalRingDrainState BidirectionalRingExecutor::drain() noexcept {
  return impl_->drain();
}

BidirectionalRingStatus BidirectionalRingExecutor::status() const noexcept {
  return impl_->status();
}

}  // namespace spark_transport::tiled_prefill_research
