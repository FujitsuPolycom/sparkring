#include "bidirectional_ring_executor.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;
using spark_transport::Tp4PrefillDirection;

namespace {

std::size_t direction_index(Tp4PrefillDirection direction) {
  return direction == Tp4PrefillDirection::kClockwise ? 0U : 1U;
}

class FakeBulkPort final : public research::BidirectionalRingBulkPort {
 public:
  research::RingSubmitState try_submit(
      const research::BidirectionalRingBulkRequest& request) override {
    requests.push_back(request);
    return fail_submit ? research::RingSubmitState::kFatal
                       : research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll(
      const research::BidirectionalRingBulkRequest& request) override {
    if (fail_poll) {
      return research::RingPollState::kFatal;
    }
    if (!complete) return research::RingPollState::kPending;
    if (on_complete) on_complete(request);
    return research::RingPollState::kComplete;
  }

  bool complete{true};
  bool fail_submit{};
  bool fail_poll{};
  std::function<void(const research::BidirectionalRingBulkRequest&)>
      on_complete;
  std::vector<research::BidirectionalRingBulkRequest> requests;
};

class FakeEdgePort final : public research::BidirectionalRingEdgePort {
 public:
  research::RingSubmitState try_post_exchange(
      const research::BidirectionalRingExchangeRequest& request) override {
    assert(request.span.active_bytes != 0);
    assert(request.span.remote_receive_offset ==
           request.span.local_send_offset +
               request.span.active_bytes);
    exchanges.push_back(request);
    return fail_exchange_submit ? research::RingSubmitState::kFatal
                                : research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_exchange(
      const research::BidirectionalRingExchangeRequest&) override {
    if (fail_exchange_poll) {
      return research::RingPollState::kFatal;
    }
    if (!complete_exchanges) {
      return research::RingPollState::kPending;
    }
    return research::RingPollState::kComplete;
  }

  research::RingSubmitState try_publish_consumed_through(
      const research::BidirectionalRingCreditRequest& request) override {
    published_credits.push_back(request);
    return fail_credit_submit ? research::RingSubmitState::kFatal
                              : research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_published_consumed_through(
      const research::BidirectionalRingCreditRequest&) override {
    if (fail_credit_poll) {
      return research::RingPollState::kFatal;
    }
    return complete_credit_publications
               ? research::RingPollState::kComplete
               : research::RingPollState::kPending;
  }

  research::RingCreditPollState poll_peer_consumed_through(
      Tp4PrefillDirection direction, std::uint64_t& wire_credit) override {
    const auto index = direction_index(direction);
    if (fail_peer_credit_poll) {
      return research::RingCreditPollState::kFatal;
    }
    if (pending_peer_credit[index] == 0) {
      return research::RingCreditPollState::kNoUpdate;
    }
    wire_credit = pending_peer_credit[index];
    pending_peer_credit[index] = 0;
    return research::RingCreditPollState::kUpdate;
  }

  bool complete_exchanges{true};
  bool complete_credit_publications{true};
  bool fail_exchange_submit{};
  bool fail_exchange_poll{};
  bool fail_credit_submit{};
  bool fail_credit_poll{};
  bool fail_peer_credit_poll{};
  std::array<bool, 2> auto_peer_credit{true, true};
  std::array<std::uint64_t, 2> pending_peer_credit{};
  std::vector<research::BidirectionalRingExchangeRequest> exchanges;
  std::vector<research::BidirectionalRingCreditRequest> published_credits;
};

void connect_symmetric_consumption(FakeBulkPort& bulk, FakeEdgePort& edge) {
  bulk.on_complete = [&edge](
                         const research::BidirectionalRingBulkRequest& request) {
    if (request.action == research::RingBulkAction::kStageInitial) return;
    const auto index = direction_index(request.direction);
    if (!edge.auto_peer_credit[index]) return;
    const auto ordinal = spark_transport::tp4_tiled_ticket_ordinal(
        request.source_ticket,
        research::kBidirectionalRingSlotsPerDirection);
    edge.pending_peer_credit[index] =
        std::max(edge.pending_peer_credit[index], ordinal + 1U);
  };
}

void drive_until_retired(research::BidirectionalRingExecutor& executor,
                         std::uint32_t limit = 2000) {
  for (std::uint32_t iteration = 0; iteration < limit; ++iteration) {
    if (executor.status().fully_retired) {
      return;
    }
    const auto result = executor.advance();
    assert(!result.poisoned);
  }
  assert(false && "bidirectional ring executor did not retire");
}

void test_fixed_shape_completes_with_exact_traffic() {
  FakeBulkPort bulk;
  FakeEdgePort edge;
  connect_symmetric_consumption(bulk, edge);
  research::BidirectionalRingExecutor executor(0, bulk, edge);

  assert(executor.drain() == research::BidirectionalRingDrainState::kIdle);
  executor.begin();
  drive_until_retired(executor);

  const auto status = executor.status();
  assert(status.output_ready);
  assert(status.fully_retired);
  assert(status.safe_to_release_registered_storage);
  assert(status.output_ready_tiles == 8);
  assert((status.posted_transfers == std::array<std::uint32_t, 2>{24, 24}));
  assert((status.completed_transfers ==
          std::array<std::uint32_t, 2>{24, 24}));
  assert((status.transmitted_bytes ==
          std::array<std::uint64_t, 2>{12U * 1024U * 1024U,
                                       12U * 1024U * 1024U}));
  assert(status.transmitted_bytes[0] + status.transmitted_bytes[1] ==
         24U * 1024U * 1024U);
  assert(edge.exchanges.size() == 48);
  assert(bulk.requests.size() == 56);
  assert(executor.drain() ==
         research::BidirectionalRingDrainState::kComplete);

  for (const auto& exchange : edge.exchanges) {
    const auto layout = spark_transport::make_tp4_tiled_pool_layout(
        research::kBidirectionalRingTileBytes,
        research::kBidirectionalRingSlotsPerDirection, 1);
    const auto region = spark_transport::tp4_tiled_slot_region(
        layout, exchange.ticket.slot, 0);
    assert(exchange.stage < 6);
    assert(exchange.tile_in_shard < 4);
    const auto ordinal = research::bidirectional_ring_ordinal(
        exchange.stage, exchange.tile_in_shard);
    assert(exchange.ticket.slot == ordinal %
                                       research::kBidirectionalRingSlotsPerDirection);
    assert(exchange.ticket.generation ==
           ordinal / research::kBidirectionalRingSlotsPerDirection + 1U);
    assert(exchange.span.active_bytes ==
           research::kBidirectionalRingTileBytes);
    assert(exchange.span.remote_receive_offset ==
           exchange.span.local_send_offset +
               research::kBidirectionalRingTileBytes);
    assert(exchange.span.local_send_offset == region.send_offset);
    assert(exchange.span.remote_receive_offset == region.receive_offset);
  }
  const auto action_count = [&bulk](research::RingBulkAction action) {
    return std::count_if(
        bulk.requests.begin(), bulk.requests.end(),
        [action](const auto& request) { return request.action == action; });
  };
  assert(action_count(research::RingBulkAction::kStageInitial) == 8);
  assert(action_count(research::RingBulkAction::kReduceForward) == 16);
  assert(action_count(
             research::RingBulkAction::kReduceFinalizeAndSeedGather) == 8);
  assert(action_count(research::RingBulkAction::kGatherForward) == 16);
  assert(action_count(research::RingBulkAction::kGatherFinish) == 8);
  for (const auto& request : bulk.requests) {
    if (request.action == research::RingBulkAction::kStageInitial) {
      assert(request.next_exchange_stage == 0);
      assert(request.incoming_doorbell_offset == 0);
      assert(request.consumed_doorbell_token == 0);
      continue;
    }
    const auto consumed = research::bidirectional_ring_consumed_stage(
        request.next_exchange_stage);
    const auto layout = spark_transport::make_tp4_tiled_pool_layout(
        research::kBidirectionalRingTileBytes,
        research::kBidirectionalRingSlotsPerDirection, 1);
    const auto source_region = spark_transport::tp4_tiled_slot_region(
        layout, request.source_ticket.slot, 0);
    assert(spark_transport::tp4_tiled_ticket_ordinal(
               request.source_ticket,
               research::kBidirectionalRingSlotsPerDirection) %
               research::kBidirectionalRingTransfersPerDirection ==
           research::bidirectional_ring_ordinal(consumed,
                                                request.tile_in_shard));
    assert(request.incoming_doorbell_offset ==
           source_region.control_offset +
               offsetof(spark_transport::DoorbellControl,
                        remote_sequence));
    assert(request.consumed_doorbell_token ==
           spark_transport::tp4_tiled_ticket_ordinal(
               request.source_ticket,
               research::kBidirectionalRingSlotsPerDirection) +
               1U);
    switch (request.action) {
      case research::RingBulkAction::kReduceForward:
        assert(consumed <= 1);
        break;
      case research::RingBulkAction::kReduceFinalizeAndSeedGather:
        assert(consumed == 2);
        break;
      case research::RingBulkAction::kGatherForward:
        assert(consumed == 3 || consumed == 4);
        break;
      case research::RingBulkAction::kGatherFinish:
        assert(consumed == 5);
        break;
      case research::RingBulkAction::kStageInitial:
        assert(false);
        break;
    }
  }

  executor.begin();
  drive_until_retired(executor);
  assert(edge.exchanges.size() == 96);
  assert(edge.exchanges.back().ticket.generation == 6);
  assert((executor.status().transmitted_bytes ==
          std::array<std::uint64_t, 2>{12U * 1024U * 1024U,
                                       12U * 1024U * 1024U}));
}

void test_adaptive_geometries_and_multiple_operations() {
  constexpr std::array<std::uint32_t, 4> query_rows{1024, 2048, 4096, 8192};
  for (const auto q : query_rows) {
    const auto geometry = research::make_bidirectional_ring_geometry(q);
    assert(research::bidirectional_ring_geometry_valid(geometry));
    assert(geometry.payload_bytes ==
           static_cast<std::uint64_t>(q) * 4096U * 2U);
    assert(geometry.half_bytes == geometry.payload_bytes / 2U);
    assert(geometry.shard_bytes == geometry.payload_bytes / 8U);
    assert(geometry.tile_bytes == geometry.payload_bytes / 32U);
    assert(geometry.bytes_per_direction == 6U * geometry.shard_bytes);
    assert(geometry.bytes_per_rank == 2U * geometry.bytes_per_direction);

    FakeBulkPort bulk;
    FakeEdgePort edge;
    connect_symmetric_consumption(bulk, edge);
    research::BidirectionalRingExecutor executor(0, geometry, bulk, edge);
    for (int operation = 0; operation < 2; ++operation) {
      executor.begin();
      drive_until_retired(executor);
      const auto status = executor.status();
      assert(status.geometry.query_rows == q);
      assert(status.geometry.tile_bytes == geometry.tile_bytes);
      assert(status.output_ready);
      assert(status.fully_retired);
      assert((status.posted_transfers ==
              std::array<std::uint32_t, 2>{24, 24}));
      assert(status.transmitted_bytes[0] == geometry.bytes_per_direction);
      assert(status.transmitted_bytes[1] == geometry.bytes_per_direction);
      assert(status.transmitted_bytes[0] + status.transmitted_bytes[1] ==
             geometry.bytes_per_rank);
    }
    assert(edge.exchanges.size() == 96);
    assert(bulk.requests.size() == 112);
    for (const auto& exchange : edge.exchanges) {
      assert(exchange.span.active_bytes == geometry.tile_bytes);
      assert(exchange.span.remote_receive_offset ==
             exchange.span.local_send_offset + geometry.tile_bytes);
    }
    for (const auto& request : bulk.requests) {
      assert(request.active_bytes == geometry.tile_bytes);
      assert(request.operation_offset_bytes + request.active_bytes <=
             geometry.payload_bytes);
    }
  }

  bool rejected = false;
  try {
    static_cast<void>(research::make_bidirectional_ring_geometry(512));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  auto invalid = research::make_bidirectional_ring_geometry(1024);
  ++invalid.tile_bytes;
  assert(!research::bidirectional_ring_geometry_valid(invalid));
}

void test_both_directions_are_in_flight_before_poll_completion() {
  FakeBulkPort bulk;
  FakeEdgePort edge;
  connect_symmetric_consumption(bulk, edge);
  edge.complete_exchanges = false;
  research::BidirectionalRingExecutor executor(1, bulk, edge);
  executor.begin();

  for (int iteration = 0; iteration < 8; ++iteration) {
    (void)executor.advance();
  }
  const auto status = executor.status();
  assert((status.posted_transfers == std::array<std::uint32_t, 2>{4, 4}));
  assert(!status.output_ready);
  assert(!status.poisoned);
}

void test_direction_credit_windows_advance_independently() {
  FakeBulkPort bulk;
  FakeEdgePort edge;
  connect_symmetric_consumption(bulk, edge);
  edge.auto_peer_credit = {false, true};
  research::BidirectionalRingExecutor executor(2, bulk, edge);
  executor.begin();

  for (int iteration = 0; iteration < 80; ++iteration) {
    const auto result = executor.advance();
    assert(!result.poisoned);
  }
  const auto status = executor.status();
  assert(status.posted_transfers[0] == 8);
  assert(status.posted_transfers[1] > 8);
  assert(!status.peer_credit_observed[0]);
  assert(status.peer_credit_observed[1]);
  assert(!status.output_ready);

  // Stages 0 and 1 occupy distinct four-slot generations. Stage 2 may reuse
  // stage-0 slots only after the peer reports consuming all four stage-0
  // tiles; it then advances exactly one additional stage without stage-1
  // credit.
  edge.pending_peer_credit[0] = 4;
  for (int iteration = 0; iteration < 80; ++iteration) {
    const auto result = executor.advance();
    assert(!result.poisoned);
  }
  assert(executor.status().posted_transfers[0] == 12);
}

void test_output_ready_is_not_safe_until_credit_writes_retire() {
  FakeBulkPort bulk;
  FakeEdgePort edge;
  connect_symmetric_consumption(bulk, edge);
  edge.complete_credit_publications = false;
  research::BidirectionalRingExecutor executor(3, bulk, edge);
  executor.begin();

  for (int iteration = 0; iteration < 400; ++iteration) {
    (void)executor.advance();
    if (executor.status().output_ready) {
      break;
    }
  }
  auto status = executor.status();
  assert(status.output_ready);
  assert(!status.fully_retired);
  assert(!status.safe_to_release_registered_storage);
  assert(executor.drain() ==
         research::BidirectionalRingDrainState::kPending);

  edge.complete_credit_publications = true;
  drive_until_retired(executor);
  status = executor.status();
  assert(status.safe_to_release_registered_storage);
}

void test_failure_poison_is_sticky_and_fail_closed() {
  FakeBulkPort bulk;
  FakeEdgePort edge;
  connect_symmetric_consumption(bulk, edge);
  edge.fail_exchange_submit = true;
  research::BidirectionalRingExecutor executor(0, bulk, edge);
  executor.begin();

  for (int iteration = 0; iteration < 10 && !executor.status().poisoned;
       ++iteration) {
    (void)executor.advance();
  }
  const auto status = executor.status();
  assert(status.poisoned);
  assert(status.failure ==
         research::BidirectionalRingFailure::kExchangeSubmit);
  assert(!status.safe_to_release_registered_storage);
  assert(executor.drain() ==
         research::BidirectionalRingDrainState::kPoisoned);

  bool rejected = false;
  try {
    executor.begin();
  } catch (const std::logic_error&) {
    rejected = true;
  }
  assert(rejected);
}

void test_unissued_peer_credit_poison_is_fail_closed() {
  FakeBulkPort bulk;
  FakeEdgePort edge;
  connect_symmetric_consumption(bulk, edge);
  edge.pending_peer_credit[0] = 1000;
  research::BidirectionalRingExecutor executor(0, bulk, edge);
  executor.begin();

  const auto result = executor.advance();
  assert(result.poisoned);
  assert(executor.status().failure ==
         research::BidirectionalRingFailure::kCreditProtocol);
  assert(executor.drain() ==
         research::BidirectionalRingDrainState::kPoisoned);
}

}  // namespace

int main() {
  static_assert(research::kBidirectionalRingTilesPerShard == 4);
  static_assert(research::kBidirectionalRingTransfersPerDirection == 24);
  static_assert(research::kBidirectionalRingTileBytes == 512U * 1024U);
  static_assert(research::bidirectional_ring_ordinal(5, 3) == 23);
  static_assert(research::bidirectional_ring_ticket(5, 3).slot == 7);
  static_assert(research::bidirectional_ring_ticket(5, 3).generation == 3);
  static_assert(research::bidirectional_ring_consumed_stage(1) == 0);
  static_assert(research::bidirectional_ring_consumed_stage(6) == 5);

  bool rejected_consumed_stage = false;
  try {
    static_cast<void>(research::bidirectional_ring_consumed_stage(0));
  } catch (const std::out_of_range&) {
    rejected_consumed_stage = true;
  }
  assert(rejected_consumed_stage);
  rejected_consumed_stage = false;
  try {
    static_cast<void>(research::bidirectional_ring_consumed_stage(7));
  } catch (const std::out_of_range&) {
    rejected_consumed_stage = true;
  }
  assert(rejected_consumed_stage);

  test_fixed_shape_completes_with_exact_traffic();
  test_adaptive_geometries_and_multiple_operations();
  test_both_directions_are_in_flight_before_poll_completion();
  test_direction_credit_windows_advance_independently();
  test_output_ready_is_not_safe_until_credit_writes_retire();
  test_failure_poison_is_sticky_and_fail_closed();
  test_unissued_peer_credit_poison_is_fail_closed();
}
