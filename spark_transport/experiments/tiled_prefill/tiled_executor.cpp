#include "tiled_executor.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spark_transport::tiled_prefill_research {
namespace {

constexpr std::uint64_t kTiledWorkEventsPerOrdinal = 3;

bool tiled_work_event_valid(TiledWorkEvent event) noexcept {
  switch (event) {
    case TiledWorkEvent::kPhase1Exchange:
    case TiledWorkEvent::kPhase2Exchange:
    case TiledWorkEvent::kCredit:
      return true;
  }
  return false;
}

enum class TileExecutionPhase : std::uint8_t {
  kAwaitingCredit,
  kStageReady,
  kStagePending,
  kPhase1Exchange,
  kReduce1Ready,
  kReduce1Pending,
  kPhase2Exchange,
  kReduce2Ready,
  kReduce2Pending,
  kOutputReady,
  kRetired,
};

bool tile_output_ready(TileExecutionPhase phase) noexcept {
  return phase == TileExecutionPhase::kOutputReady ||
         phase == TileExecutionPhase::kRetired;
}

struct TileExecution {
  TiledExecutorTileDescriptor descriptor{};
  TileExecutionPhase phase{TileExecutionPhase::kAwaitingCredit};
  std::array<bool, kTp4TiledEdgeCount> edge_submitted{};
  std::array<bool, kTp4TiledEdgeCount> edge_completed{};
};

}  // namespace

std::uint64_t tiled_doorbell_token(std::uint64_t ordinal) {
  if (ordinal == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("tiled TP4 doorbell token exhausted");
  }
  return ordinal + 1;
}

std::uint64_t tiled_work_id(std::uint64_t ordinal,
                            TiledWorkEvent event) {
  if (!tiled_work_event_valid(event)) {
    throw std::invalid_argument("invalid tiled TP4 work event");
  }
  if (ordinal >
      (std::numeric_limits<std::uint64_t>::max() -
       static_cast<std::uint64_t>(event)) /
          kTiledWorkEventsPerOrdinal) {
    throw std::overflow_error("tiled TP4 work ID exhausted");
  }
  return ordinal * kTiledWorkEventsPerOrdinal +
         static_cast<std::uint64_t>(event);
}

std::uint64_t tiled_wire_credit(std::uint64_t consumed_through) {
  if (consumed_through ==
      std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("tiled TP4 credit encoding exhausted");
  }
  return consumed_through + 1;
}

std::uint64_t tiled_consumed_through_from_wire(
    std::uint64_t wire_credit) {
  if (wire_credit == 0) {
    throw std::invalid_argument(
        "zero tiled TP4 wire credit means no publication");
  }
  return wire_credit - 1;
}

TiledExecutorTileDescriptor make_tiled_executor_tile_descriptor(
    const Tp4TiledTileDescriptor& descriptor,
    const Tp4TiledPoolLayout& layout) {
  const std::uint64_t ordinal =
      tp4_tiled_ticket_ordinal(descriptor.ticket,
                               layout.slots_per_edge);
  if (layout.lanes_per_edge != kTp4TiledDefaultLanesPerEdge ||
      descriptor.active_bytes == 0 ||
      descriptor.active_bytes > layout.tile_payload_bytes ||
      descriptor.active_bytes % 32U != 0 ||
      descriptor.operation_offset_bytes % 16U != 0) {
    throw std::invalid_argument(
        "tile descriptor does not satisfy the tiled GPU launch ABI");
  }
  const std::uint64_t slot_storage_offset =
      static_cast<std::uint64_t>(descriptor.ticket.slot) *
      layout.slot_stride;
  if (slot_storage_offset + layout.slot_stride > layout.total_bytes) {
    throw std::out_of_range(
        "tile descriptor exceeds registered endpoint storage");
  }
  return {descriptor.ticket,
          ordinal,
          TiledBulkDescriptor{
              descriptor.operation_offset_bytes,
              descriptor.operation_offset_bytes,
              slot_storage_offset,
              descriptor.active_bytes,
              descriptor.ticket.generation}};
}

class TiledExecutor::Impl {
 public:
  Impl(Tp4SessionStorageOptions storage_options,
       TiledBulkLaunchPort& bulk_port, TiledEdgePort& edge0,
       TiledEdgePort& edge1)
      : storage_options_(std::move(storage_options)),
        layout_(make_tp4_tiled_pool_layout(storage_options_)),
        credit_window_(storage_options_.slots_per_edge),
        bulk_port_(bulk_port),
        edges_{&edge0, &edge1} {
    if (storage_options_.lanes_per_edge !=
        kTp4TiledDefaultLanesPerEdge) {
      throw std::invalid_argument(
          "tiled executor requires exactly two lanes per edge");
    }
    if (bulk_port_.stream_policy() !=
        TiledExecutorStreamPolicy::kSingleStreamCorrectnessOnly) {
      throw std::invalid_argument(
          "tiled executor supports single-stream correctness only");
    }
    if (edge0.edge_index() != 0 || edge1.edge_index() != 1 ||
        edge0.engine_identity() == 0 ||
        edge0.engine_identity() != edge1.engine_identity() ||
        edge0.qp_identity() == 0 || edge1.qp_identity() == 0 ||
        edge0.qp_identity() == edge1.qp_identity()) {
      throw std::invalid_argument(
          "tiled executor requires two stable edge QPs with one owner");
    }
  }

  void begin(std::uint32_t query_rows) {
    if (poisoned_) {
      throw std::logic_error("poisoned tiled executor rejects new work");
    }
    if (active_operation_ && !fully_retired()) {
      throw std::logic_error(
          "tiled executor already has an unretired operation");
    }
    if (query_rows == 0 ||
        query_rows > storage_options_.maximum_query_rows) {
      throw std::out_of_range(
          "query rows exceed tiled executor session capacity");
    }
    if (credit_window_.exhausted()) {
      throw std::overflow_error("tiled executor ticket sequence exhausted");
    }

    const auto operation =
        make_tp4_tiled_bf16_allreduce_operation(
            query_rows, storage_options_.bytes_per_row);
    const std::uint64_t first_ordinal = credit_window_.next_ordinal();
    std::vector<TileExecution> tiles;
    tiles.reserve(operation.tile_count);
    for (std::uint32_t tile = 0; tile < operation.tile_count; ++tile) {
      const auto descriptor = make_tp4_tiled_tile_descriptor(
          operation, tile, first_ordinal);
      tiles.push_back(TileExecution{
          make_tiled_executor_tile_descriptor(descriptor, layout_)});
    }

    tiles_ = std::move(tiles);
    active_operation_ = true;
    query_rows_ = query_rows;
    active_payload_bytes_ = operation.active_bytes;
    first_tile_ordinal_ = first_ordinal;
    counted_output_before_retirement_ = false;
  }

  TiledExecutorAdvanceResult advance() noexcept {
    TiledExecutorAdvanceResult result{};
    if (poisoned_) {
      result.poisoned = true;
      return result;
    }
    if (!active_operation_) {
      return result;
    }

    try {
      poll_peer_credits(result);
      if (poisoned_) {
        result.poisoned = true;
        return result;
      }
      for (auto& tile : tiles_) {
        progress_tile(tile, result);
        if (poisoned_) {
          result.poisoned = true;
          return result;
        }
      }
      publish_local_credit(result);
      retire_credited_tiles(result);
      const auto snapshot = status();
      if (snapshot.output_ready && !snapshot.fully_retired &&
          !counted_output_before_retirement_) {
        ++output_ready_before_final_retirement_count_;
        counted_output_before_retirement_ = true;
      }
    } catch (...) {
      poison(TiledExecutorFailure::kPortException);
    }
    result.poisoned = poisoned_;
    return result;
  }

  TiledExecutorStatus status() const noexcept {
    TiledExecutorStatus result{};
    result.active_operation = active_operation_;
    result.poisoned = poisoned_;
    result.query_rows = query_rows_;
    result.tile_count = static_cast<std::uint32_t>(tiles_.size());
    result.active_payload_bytes = active_payload_bytes_;
    result.first_tile_ordinal = first_tile_ordinal_;
    if (credit_window_.has_issued()) {
      result.highest_issued_ordinal = credit_window_.highest_issued();
    }
    result.unexpected_generation_count = unexpected_generation_count_;
    result.credit_regression_count = credit_regression_count_;
    result.output_ready_before_final_retirement_count =
        output_ready_before_final_retirement_count_;
    result.failure = failure_;
    for (const auto& tile : tiles_) {
      if (tile.phase != TileExecutionPhase::kAwaitingCredit) {
        ++result.acquired_tiles;
      }
      if (tile_output_ready(tile.phase)) {
        ++result.output_ready_tiles;
      }
      if (tile.phase == TileExecutionPhase::kRetired) {
        ++result.retired_tiles;
      }
    }
    result.output_ready = active_operation_ && !tiles_.empty() &&
                          result.output_ready_tiles == tiles_.size();
    result.outbound_credits_complete =
        local_credit_published_[0] && local_credit_completed_[0] &&
        local_consumed_through_[0] >= local_required_through_[0] &&
        local_credit_published_[1] && local_credit_completed_[1] &&
        local_consumed_through_[1] >= local_required_through_[1];
    result.fully_retired = result.output_ready &&
                           result.retired_tiles == tiles_.size() &&
                           result.outbound_credits_complete;
    result.safe_to_release_registered_storage = result.fully_retired;
    return result;
  }

  TiledDrainState drain() noexcept {
    if (poisoned_) {
      return TiledDrainState::kPoisoned;
    }
    if (!active_operation_) {
      return TiledDrainState::kIdle;
    }
    (void)advance();
    if (poisoned_) {
      return TiledDrainState::kPoisoned;
    }
    return status().safe_to_release_registered_storage
               ? TiledDrainState::kComplete
               : TiledDrainState::kPending;
  }

 private:
  bool fully_retired() const noexcept { return status().fully_retired; }

  void poison(TiledExecutorFailure failure) noexcept {
    poisoned_ = true;
    if (failure_ == TiledExecutorFailure::kNone) {
      failure_ = failure;
    }
  }

  TiledBulkLaunchRequest bulk_request(
      const TileExecution& tile, TiledBulkPhase phase) const noexcept {
    return {phase,
            tile.descriptor.ticket,
            tile.descriptor.ordinal,
            tile.descriptor.gpu,
            EndpointTileStorageLayout{layout_.lane_receive_offset,
                                      layout_.lane_stride}};
  }

  TiledEdgeExchangeRequest edge_request(
      const TileExecution& tile, TiledExchangePhase phase,
      std::uint32_t edge) const {
    const std::uint32_t lane =
        phase == TiledExchangePhase::kPhase1 ? edge : 1U - edge;
    const auto region = tp4_tiled_slot_region(
        layout_, tile.descriptor.ticket.slot, lane);
    const auto event =
        phase == TiledExchangePhase::kPhase1
            ? TiledWorkEvent::kPhase1Exchange
            : TiledWorkEvent::kPhase2Exchange;
    return {phase,
            tile.descriptor.ticket,
            tile.descriptor.ordinal,
            tiled_doorbell_token(tile.descriptor.ordinal),
            tiled_work_id(tile.descriptor.ordinal, event),
            region.send_offset,
            region.receive_offset,
            region.control_offset +
                offsetof(DoorbellControl, producer_sequence),
            region.control_offset +
                offsetof(DoorbellControl, remote_sequence),
            tile.descriptor.gpu.active_bytes / 2U,
            edge,
            lane};
  }

  TiledCreditPublishRequest credit_request(
      std::uint64_t target, std::uint32_t edge) const {
    // One fixed cacheline per edge carries the cumulative watermark. The
    // executor waits for its prior outbound CQE before overwriting reserved.
    const auto region = tp4_tiled_slot_region(layout_, 0, 0);
    return {target,
            tiled_wire_credit(target),
            tiled_work_id(target, TiledWorkEvent::kCredit),
            region.control_offset + offsetof(DoorbellControl, reserved),
            region.control_offset +
                offsetof(DoorbellControl, acknowledgement_sequence),
            edge};
  }

  TiledCreditObserveRequest credit_observe_request(
      std::uint32_t edge) const {
    const auto region = tp4_tiled_slot_region(layout_, 0, 0);
    return {region.control_offset +
                offsetof(DoorbellControl, acknowledgement_sequence),
            edge};
  }

  void poll_peer_credits(TiledExecutorAdvanceResult& result) {
    for (std::uint32_t edge = 0; edge < kTp4TiledEdgeCount; ++edge) {
      std::uint64_t wire_credit{};
      const auto poll = edges_[edge]->poll_peer_consumed_through(
          credit_observe_request(edge), wire_credit);
      if (poll == TiledCreditPollState::kFatal) {
        poison(TiledExecutorFailure::kEdgePoll);
        return;
      }
      if (poll != TiledCreditPollState::kUpdate) {
        continue;
      }
      // A mapped acknowledgement word is zero before the first remote write.
      // It is not the inclusive ordinal-zero watermark.
      if (wire_credit == 0) {
        continue;
      }
      const std::uint64_t watermark =
          tiled_consumed_through_from_wire(wire_credit);
      if (peer_credit_observed_[edge] &&
          watermark < peer_consumed_through_[edge]) {
        ++credit_regression_count_;
        poison(TiledExecutorFailure::kCreditProtocol);
        return;
      }
      if (!credit_window_.has_issued() ||
          watermark > credit_window_.highest_issued()) {
        ++unexpected_generation_count_;
        poison(TiledExecutorFailure::kCreditProtocol);
        return;
      }
      const auto update =
          credit_window_.observe_consumed_through(edge, watermark);
      if (update == Tp4TiledCreditUpdateState::kPoisoned) {
        poison(TiledExecutorFailure::kCreditProtocol);
        return;
      }
      peer_credit_observed_[edge] = true;
      peer_consumed_through_[edge] = watermark;
      result.made_progress = true;
    }
  }

  void submit_bulk(TileExecution& tile, TiledBulkPhase phase,
                   TileExecutionPhase pending,
                   TiledExecutorAdvanceResult& result) {
    const auto request = bulk_request(tile, phase);
    const auto submit = bulk_port_.try_submit(request);
    if (submit == TiledSubmitState::kFatal) {
      poison(TiledExecutorFailure::kBulkSubmit);
      return;
    }
    if (submit == TiledSubmitState::kBackpressured) {
      result.backpressured = true;
      return;
    }
    tile.phase = pending;
    result.made_progress = true;
  }

  void poll_bulk(TileExecution& tile, TiledBulkPhase phase,
                 TileExecutionPhase complete,
                 TiledExecutorAdvanceResult& result) {
    const auto poll = bulk_port_.poll(bulk_request(tile, phase));
    if (poll == TiledPollState::kFatal) {
      poison(TiledExecutorFailure::kBulkPoll);
      return;
    }
    if (poll == TiledPollState::kComplete) {
      tile.phase = complete;
      result.made_progress = true;
    }
  }

  void progress_exchange(TileExecution& tile,
                         TiledExchangePhase phase,
                         TileExecutionPhase complete,
                         TiledExecutorAdvanceResult& result) {
    for (std::uint32_t edge = 0; edge < kTp4TiledEdgeCount; ++edge) {
      const auto request = edge_request(tile, phase, edge);
      if (!tile.edge_submitted[edge]) {
        const auto submit = edges_[edge]->try_post_exchange(request);
        if (submit == TiledSubmitState::kFatal) {
          poison(TiledExecutorFailure::kEdgeSubmit);
          return;
        }
        if (submit == TiledSubmitState::kBackpressured) {
          result.backpressured = true;
          continue;
        }
        tile.edge_submitted[edge] = true;
        result.made_progress = true;
      }
      if (tile.edge_submitted[edge] &&
          !tile.edge_completed[edge]) {
        const auto poll = edges_[edge]->poll_exchange(request);
        if (poll == TiledPollState::kFatal) {
          poison(TiledExecutorFailure::kEdgePoll);
          return;
        }
        if (poll == TiledPollState::kComplete) {
          tile.edge_completed[edge] = true;
          result.made_progress = true;
        }
      }
    }
    if (tile.edge_completed[0] && tile.edge_completed[1]) {
      tile.edge_submitted = {};
      tile.edge_completed = {};
      tile.phase = complete;
    }
  }

  void progress_tile(TileExecution& tile,
                      TiledExecutorAdvanceResult& result) {
    switch (tile.phase) {
      case TileExecutionPhase::kAwaitingCredit: {
        if (tile.descriptor.ordinal != credit_window_.next_ordinal()) {
          return;
        }
        // Peer credit proves the remote endpoint consumed the prior
        // generation; it does not prove this rank finished using the same
        // physical slot. Keep local slot ownership until that earlier tile is
        // fully retired, even if another slot retires out of order.
        for (const auto& owner : tiles_) {
          if (owner.descriptor.ordinal >= tile.descriptor.ordinal) {
            break;
          }
          if (owner.descriptor.ticket.slot == tile.descriptor.ticket.slot &&
              owner.phase != TileExecutionPhase::kRetired) {
            result.backpressured = true;
            return;
          }
        }
        const auto acquire =
            credit_window_.try_acquire(tile.descriptor.ticket);
        if (acquire == Tp4TiledAcquireState::kReady) {
          tile.phase = TileExecutionPhase::kStageReady;
          result.made_progress = true;
        } else if (acquire ==
                   Tp4TiledAcquireState::kWaitingForCredit) {
          result.backpressured = true;
        } else if (acquire ==
                   Tp4TiledAcquireState::kUnexpectedTicket) {
          ++unexpected_generation_count_;
          poison(TiledExecutorFailure::kUnexpectedTicket);
        } else if (acquire == Tp4TiledAcquireState::kExhausted) {
          poison(TiledExecutorFailure::kTicketExhausted);
        } else {
          poison(TiledExecutorFailure::kCreditProtocol);
        }
        return;
      }
      case TileExecutionPhase::kStageReady:
        submit_bulk(tile, TiledBulkPhase::kStagePhase1,
                    TileExecutionPhase::kStagePending, result);
        return;
      case TileExecutionPhase::kStagePending:
        poll_bulk(tile, TiledBulkPhase::kStagePhase1,
                  TileExecutionPhase::kPhase1Exchange, result);
        return;
      case TileExecutionPhase::kPhase1Exchange:
        progress_exchange(tile, TiledExchangePhase::kPhase1,
                          TileExecutionPhase::kReduce1Ready, result);
        return;
      case TileExecutionPhase::kReduce1Ready:
        submit_bulk(tile, TiledBulkPhase::kReducePhase1,
                    TileExecutionPhase::kReduce1Pending, result);
        return;
      case TileExecutionPhase::kReduce1Pending:
        poll_bulk(tile, TiledBulkPhase::kReducePhase1,
                  TileExecutionPhase::kPhase2Exchange, result);
        return;
      case TileExecutionPhase::kPhase2Exchange:
        progress_exchange(tile, TiledExchangePhase::kPhase2,
                          TileExecutionPhase::kReduce2Ready, result);
        return;
      case TileExecutionPhase::kReduce2Ready:
        submit_bulk(tile, TiledBulkPhase::kReducePhase2,
                    TileExecutionPhase::kReduce2Pending, result);
        return;
      case TileExecutionPhase::kReduce2Pending:
        poll_bulk(tile, TiledBulkPhase::kReducePhase2,
                  TileExecutionPhase::kOutputReady, result);
        return;
      case TileExecutionPhase::kOutputReady:
      case TileExecutionPhase::kRetired:
        return;
    }
  }

  void publish_local_credit(TiledExecutorAdvanceResult& result) {
    bool has_target{};
    std::uint64_t target{};
    for (const auto& tile : tiles_) {
      if (!tile_output_ready(tile.phase)) {
        break;
      }
      has_target = true;
      target = tile.descriptor.ordinal;
    }
    if (!has_target) {
      return;
    }
    for (std::uint32_t edge = 0; edge < kTp4TiledEdgeCount; ++edge) {
      if (local_required_through_[edge] < target) {
        local_required_through_[edge] = target;
      }
      // The fixed local reserved word is the source of every cumulative
      // credit write. Do not overwrite it with a newer watermark until the
      // preceding signaled write has a completion.
      if (local_credit_published_[edge] &&
          !local_credit_completed_[edge]) {
        const auto outstanding =
            credit_request(local_consumed_through_[edge], edge);
        const auto poll =
            edges_[edge]->poll_published_consumed_through(outstanding);
        if (poll == TiledPollState::kFatal) {
          poison(TiledExecutorFailure::kCreditPublish);
          return;
        }
        if (poll == TiledPollState::kPending) {
          continue;
        }
        local_credit_completed_[edge] = true;
        result.made_progress = true;
      }
      if (!local_credit_published_[edge] ||
          local_consumed_through_[edge] < target) {
        const auto request = credit_request(target, edge);
        const auto submit =
            edges_[edge]->try_publish_consumed_through(request);
        if (submit == TiledSubmitState::kFatal) {
          poison(TiledExecutorFailure::kCreditPublish);
          return;
        }
        if (submit == TiledSubmitState::kBackpressured) {
          result.backpressured = true;
          continue;
        }
        local_credit_published_[edge] = true;
        local_credit_completed_[edge] = false;
        local_consumed_through_[edge] = target;
        result.made_progress = true;
      }
      if (!local_credit_completed_[edge] &&
          local_consumed_through_[edge] == target) {
        const auto request = credit_request(target, edge);
        const auto poll =
            edges_[edge]->poll_published_consumed_through(request);
        if (poll == TiledPollState::kFatal) {
          poison(TiledExecutorFailure::kCreditPublish);
          return;
        }
        if (poll == TiledPollState::kComplete) {
          local_credit_completed_[edge] = true;
          result.made_progress = true;
        }
      }
    }
  }

  void retire_credited_tiles(TiledExecutorAdvanceResult& result) {
    for (auto& tile : tiles_) {
      if (tile.phase != TileExecutionPhase::kOutputReady) {
        continue;
      }
      bool retired = true;
      for (std::size_t edge = 0; edge < kTp4TiledEdgeCount; ++edge) {
        retired = retired && peer_credit_observed_[edge] &&
                  peer_consumed_through_[edge] >=
                      tile.descriptor.ordinal;
      }
      if (retired) {
        tile.phase = TileExecutionPhase::kRetired;
        result.made_progress = true;
      }
    }
  }

  Tp4SessionStorageOptions storage_options_;
  Tp4TiledPoolLayout layout_;
  Tp4TiledCreditWindow credit_window_;
  TiledBulkLaunchPort& bulk_port_;
  std::array<TiledEdgePort*, kTp4TiledEdgeCount> edges_;
  std::vector<TileExecution> tiles_;
  std::array<std::uint64_t, kTp4TiledEdgeCount>
      peer_consumed_through_{};
  std::array<std::uint64_t, kTp4TiledEdgeCount>
      local_consumed_through_{};
  std::array<std::uint64_t, kTp4TiledEdgeCount>
      local_required_through_{};
  std::array<bool, kTp4TiledEdgeCount> peer_credit_observed_{};
  std::array<bool, kTp4TiledEdgeCount> local_credit_published_{};
  std::array<bool, kTp4TiledEdgeCount> local_credit_completed_{};
  bool active_operation_{};
  bool poisoned_{};
  bool counted_output_before_retirement_{};
  std::uint32_t query_rows_{};
  std::uint64_t active_payload_bytes_{};
  std::uint64_t first_tile_ordinal_{};
  std::uint64_t unexpected_generation_count_{};
  std::uint64_t credit_regression_count_{};
  std::uint64_t output_ready_before_final_retirement_count_{};
  TiledExecutorFailure failure_{TiledExecutorFailure::kNone};
};

TiledExecutor::TiledExecutor(
    Tp4SessionStorageOptions storage_options,
    TiledBulkLaunchPort& bulk_port, TiledEdgePort& edge0,
    TiledEdgePort& edge1)
    : impl_(std::make_unique<Impl>(std::move(storage_options),
                                   bulk_port, edge0, edge1)) {}

TiledExecutor::~TiledExecutor() = default;

void TiledExecutor::begin(std::uint32_t query_rows) {
  impl_->begin(query_rows);
}

TiledExecutorAdvanceResult TiledExecutor::advance() noexcept {
  return impl_->advance();
}

TiledDrainState TiledExecutor::drain() noexcept {
  return impl_->drain();
}

TiledExecutorStatus TiledExecutor::status() const noexcept {
  return impl_->status();
}

}  // namespace spark_transport::tiled_prefill_research
