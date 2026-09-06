#include "tiled_executor.hpp"

#include <cassert>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;

namespace {

class ScriptedBulkPort final : public research::TiledBulkLaunchPort {
 public:
  research::TiledExecutorStreamPolicy stream_policy()
      const noexcept override {
    return research::TiledExecutorStreamPolicy::
        kSingleStreamCorrectnessOnly;
  }

  research::TiledSubmitState try_submit(
      const research::TiledBulkLaunchRequest& request) override {
    if (backpressure_submissions != 0) {
      --backpressure_submissions;
      return research::TiledSubmitState::kBackpressured;
    }
    submissions.push_back(request);
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll(
      const research::TiledBulkLaunchRequest&) override {
    if (pending_polls != 0) {
      --pending_polls;
      return research::TiledPollState::kPending;
    }
    return research::TiledPollState::kComplete;
  }

  std::uint32_t backpressure_submissions{};
  std::uint32_t pending_polls{};
  std::vector<research::TiledBulkLaunchRequest> submissions;
};

class LoopbackEdgePort final : public research::TiledEdgePort {
 public:
  LoopbackEdgePort(std::uint32_t edge, std::uintptr_t engine_identity,
                   std::uintptr_t qp_identity)
      : edge_(edge),
        engine_identity_(engine_identity),
        qp_identity_(qp_identity) {}

  std::uint32_t edge_index() const noexcept override { return edge_; }
  std::uintptr_t engine_identity() const noexcept override {
    return engine_identity_;
  }
  std::uintptr_t qp_identity() const noexcept override {
    return qp_identity_;
  }

  research::TiledSubmitState try_post_exchange(
      const research::TiledEdgeExchangeRequest& request) override {
    if (backpressure_exchanges != 0) {
      --backpressure_exchanges;
      return research::TiledSubmitState::kBackpressured;
    }
    exchanges.push_back(request);
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll_exchange(
      const research::TiledEdgeExchangeRequest&) override {
    if (pending_exchange_polls != 0) {
      --pending_exchange_polls;
      return research::TiledPollState::kPending;
    }
    return research::TiledPollState::kComplete;
  }

  research::TiledSubmitState try_publish_consumed_through(
      const research::TiledCreditPublishRequest& request) override {
    if (backpressure_publications != 0) {
      --backpressure_publications;
      return research::TiledSubmitState::kBackpressured;
    }
    publications.push_back(request);
    if (loopback_credits) {
      peer_wire_credits.push_back(request.wire_credit);
    }
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll_published_consumed_through(
      const research::TiledCreditPublishRequest&) override {
    if (pending_publication_polls != 0) {
      --pending_publication_polls;
      return research::TiledPollState::kPending;
    }
    return research::TiledPollState::kComplete;
  }

  research::TiledCreditPollState poll_peer_consumed_through(
      const research::TiledCreditObserveRequest& request,
      std::uint64_t& consumed_through) override {
    observations.push_back(request);
    if (peer_wire_credits.empty()) {
      return research::TiledCreditPollState::kNoUpdate;
    }
    consumed_through = peer_wire_credits.front();
    peer_wire_credits.pop_front();
    return research::TiledCreditPollState::kUpdate;
  }

  std::vector<research::TiledEdgeExchangeRequest> exchanges;
  std::vector<research::TiledCreditPublishRequest> publications;
  std::vector<research::TiledCreditObserveRequest> observations;
  std::uint32_t backpressure_exchanges{};
  std::uint32_t pending_exchange_polls{};
  std::uint32_t backpressure_publications{};
  std::uint32_t pending_publication_polls{};
  bool loopback_credits{true};

  void inject_peer_credit(std::uint64_t watermark) {
    peer_wire_credits.push_back(research::tiled_wire_credit(watermark));
  }

  void inject_peer_wire_credit(std::uint64_t wire_credit) {
    peer_wire_credits.push_back(wire_credit);
  }

 private:
  std::uint32_t edge_{};
  std::uintptr_t engine_identity_{};
  std::uintptr_t qp_identity_{};
  std::deque<std::uint64_t> peer_wire_credits;
};

void drive_until_retired(research::TiledExecutor& executor) {
  for (int iteration = 0; iteration < 256; ++iteration) {
    (void)executor.advance();
    if (executor.status().fully_retired) {
      return;
    }
  }
  assert(false && "tiled executor did not retire bounded fake work");
}

void test_one_tile_runs_through_gpu_and_both_stable_edge_qps() {
  constexpr std::uintptr_t engine_identity = 0x1234;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0x1000);
  LoopbackEdgePort edge1(1, engine_identity, 0x2000);
  const auto storage =
      spark_transport::make_tp4_tiled_session_storage_options(40);
  research::TiledExecutor executor(storage, bulk, edge0, edge1);

  executor.begin(40);
  drive_until_retired(executor);

  const auto status = executor.status();
  assert(!status.poisoned);
  assert(status.query_rows == 40);
  assert(status.tile_count == 1);
  assert(status.output_ready_tiles == 1);
  assert(status.retired_tiles == 1);
  assert(status.output_ready);
  assert(status.fully_retired);

  assert(bulk.submissions.size() == 3);
  assert(bulk.submissions[0].phase ==
         research::TiledBulkPhase::kStagePhase1);
  assert(bulk.submissions[1].phase ==
         research::TiledBulkPhase::kReducePhase1);
  assert(bulk.submissions[2].phase ==
         research::TiledBulkPhase::kReducePhase2);
  assert(bulk.submissions[0].descriptor.active_bytes == 40U * 6144U * 2U);
  assert(bulk.submissions[0].descriptor.generation == 1);
  assert(bulk.submissions[0].ticket.generation == 1);
  assert(bulk.submissions[0].ticket.slot == 0);
  assert(bulk.submissions[0].layout.lane_receive_offset == 256U * 1024U);
  assert(bulk.submissions[0].layout.lane_stride == 512U * 1024U + 64U);

  assert(edge0.exchanges.size() == 2);
  assert(edge1.exchanges.size() == 2);
  assert(edge0.exchanges[0].lane == 0);
  assert(edge1.exchanges[0].lane == 1);
  assert(edge0.exchanges[1].lane == 1);
  assert(edge1.exchanges[1].lane == 0);
  assert(edge0.exchanges[0].local_payload_offset == 0);
  assert(edge0.exchanges[0].remote_payload_offset == 256U * 1024U);
  assert(edge0.exchanges[0].local_doorbell_offset ==
         512U * 1024U + 8U);
  assert(edge0.exchanges[0].remote_doorbell_offset ==
         512U * 1024U + 16U);
  assert(edge1.exchanges[0].local_payload_offset == 512U * 1024U + 64U);
  assert(edge1.exchanges[0].remote_payload_offset ==
         768U * 1024U + 64U);
  assert(edge0.exchanges[0].active_bytes == 40U * 6144U);
  assert(edge1.exchanges[0].active_bytes == 40U * 6144U);
  assert(edge0.publications.size() == 1);
  assert(edge1.publications.size() == 1);
  assert(edge0.publications[0].consumed_through == 0);
  assert(edge1.publications[0].consumed_through == 0);
  assert(edge0.publications[0].wire_credit == 1);
  assert(edge1.publications[0].wire_credit == 1);
  assert(edge0.publications[0].local_credit_offset ==
         512U * 1024U + 56U);
  assert(edge0.publications[0].remote_credit_offset ==
         512U * 1024U + 32U);
  assert(!edge0.observations.empty());
  assert(edge0.observations[0].local_credit_offset ==
         512U * 1024U + 32U);
}

void test_backpressure_retries_without_duplicate_posts_or_poison() {
  constexpr std::uintptr_t engine_identity = 0x4321;
  ScriptedBulkPort bulk;
  bulk.backpressure_submissions = 1;
  bulk.pending_polls = 1;
  LoopbackEdgePort edge0(0, engine_identity, 0x3000);
  LoopbackEdgePort edge1(1, engine_identity, 0x4000);
  edge0.backpressure_exchanges = 1;
  edge1.pending_exchange_polls = 1;
  edge0.backpressure_publications = 1;
  edge1.pending_publication_polls = 2;
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(40),
      bulk, edge0, edge1);

  executor.begin(40);
  drive_until_retired(executor);

  assert(!executor.status().poisoned);
  assert(bulk.submissions.size() == 3);
  assert(edge0.exchanges.size() == 2);
  assert(edge1.exchanges.size() == 2);
  assert(edge0.publications.size() == 1);
  assert(edge1.publications.size() == 1);
}

void test_output_ready_precedes_drain_until_outbound_and_inbound_credits() {
  constexpr std::uintptr_t engine_identity = 0x5678;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0x5000);
  LoopbackEdgePort edge1(1, engine_identity, 0x6000);
  edge0.loopback_credits = false;
  edge1.loopback_credits = false;
  edge0.pending_publication_polls = 1;
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(40),
      bulk, edge0, edge1);

  executor.begin(40);
  for (int iteration = 0; iteration < 64 &&
                          !executor.status().output_ready; ++iteration) {
    (void)executor.advance();
  }
  auto status = executor.status();
  assert(status.output_ready);
  assert(!status.fully_retired);
  assert(!status.safe_to_release_registered_storage);
  assert(status.output_ready_before_final_retirement_count == 1);

  edge0.inject_peer_credit(0);
  (void)executor.advance();
  assert(!executor.status().fully_retired);
  edge1.inject_peer_credit(0);
  drive_until_retired(executor);
  status = executor.status();
  assert(status.fully_retired);
  assert(status.outbound_credits_complete);
  assert(status.safe_to_release_registered_storage);
}

void test_ninth_tile_waits_for_both_edge_watermarks() {
  constexpr std::uintptr_t engine_identity = 0x6789;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0x7000);
  LoopbackEdgePort edge1(1, engine_identity, 0x8000);
  edge0.loopback_credits = false;
  edge1.loopback_credits = false;
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(512),
      bulk, edge0, edge1);

  executor.begin(512);
  for (int iteration = 0; iteration < 128; ++iteration) {
    (void)executor.advance();
    if (executor.status().output_ready_tiles == 8) {
      break;
    }
  }
  auto status = executor.status();
  assert(status.tile_count == 12);
  assert(status.highest_issued_ordinal == 7);
  assert(status.output_ready_tiles == 8);
  assert(!status.output_ready);
  assert(bulk.submissions.size() == 8U * 3U);

  edge0.inject_peer_credit(7);
  for (int iteration = 0; iteration < 8; ++iteration) {
    (void)executor.advance();
  }
  assert(executor.status().highest_issued_ordinal == 7);
  assert(bulk.submissions.size() == 8U * 3U);

  edge1.inject_peer_credit(7);
  for (int iteration = 0; iteration < 128 &&
                          !executor.status().output_ready; ++iteration) {
    (void)executor.advance();
  }
  status = executor.status();
  assert(status.highest_issued_ordinal == 11);
  assert(status.output_ready);
  assert(!status.fully_retired);
  assert(bulk.submissions.size() == 12U * 3U);

  edge0.inject_peer_credit(11);
  edge1.inject_peer_credit(11);
  drive_until_retired(executor);
  assert(executor.status().safe_to_release_registered_storage);
}

void test_fast_peer_credits_never_exceed_the_physical_slot_window() {
  constexpr std::uintptr_t engine_identity = 0x6abc;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0x8100);
  LoopbackEdgePort edge1(1, engine_identity, 0x8200);
  edge0.loopback_credits = false;
  edge1.loopback_credits = false;
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(512),
      bulk, edge0, edge1);

  executor.begin(512);
  (void)executor.advance();
  assert(executor.status().acquired_tiles == 8);
  assert(executor.status().retired_tiles == 0);

  // A peer can finish its corresponding tiles before this rank. Credits may
  // therefore cover all first-generation slots while their local owners are
  // still executing. They must not authorize local physical-slot aliasing.
  edge0.inject_peer_credit(7);
  edge1.inject_peer_credit(7);
  (void)executor.advance();
  const auto status = executor.status();
  assert(status.acquired_tiles - status.retired_tiles <=
         spark_transport::kTp4TiledDefaultSlotsPerEdge);
}

void test_cumulative_credit_never_overwrites_an_uncompleted_publish() {
  constexpr std::uintptr_t engine_identity = 0x789a;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0x9000);
  LoopbackEdgePort edge1(1, engine_identity, 0xa000);
  edge0.loopback_credits = false;
  edge1.loopback_credits = false;
  edge0.pending_publication_polls = 1000;
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(512),
      bulk, edge0, edge1);

  executor.begin(512);
  for (int iteration = 0; iteration < 128; ++iteration) {
    (void)executor.advance();
    if (executor.status().output_ready_tiles == 8) {
      break;
    }
  }
  assert(edge0.publications.size() == 1);
  assert(edge0.publications[0].consumed_through == 7);

  edge0.inject_peer_credit(7);
  edge1.inject_peer_credit(7);
  for (int iteration = 0; iteration < 128 &&
                          !executor.status().output_ready; ++iteration) {
    (void)executor.advance();
  }
  assert(executor.status().output_ready);
  assert(edge0.publications.size() == 1);

  edge0.pending_publication_polls = 0;
  for (int iteration = 0; iteration < 8 &&
                          edge0.publications.size() != 2; ++iteration) {
    (void)executor.advance();
  }
  assert(edge0.publications.size() == 2);
  assert(edge0.publications[1].consumed_through == 11);

  edge0.inject_peer_credit(11);
  edge1.inject_peer_credit(11);
  drive_until_retired(executor);
}

void test_backpressured_new_credit_blocks_next_operation_retirement() {
  constexpr std::uintptr_t engine_identity = 0x7abc;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0x9100);
  LoopbackEdgePort edge1(1, engine_identity, 0xa100);
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(40),
      bulk, edge0, edge1);

  executor.begin(40);
  drive_until_retired(executor);

  edge0.loopback_credits = false;
  edge0.backpressure_publications = 1000;
  executor.begin(40);
  for (int iteration = 0; iteration < 64 &&
                          !executor.status().output_ready; ++iteration) {
    (void)executor.advance();
  }
  edge0.inject_peer_credit(1);
  for (int iteration = 0; iteration < 16; ++iteration) {
    (void)executor.advance();
  }

  auto status = executor.status();
  assert(status.output_ready);
  assert(!status.outbound_credits_complete);
  assert(!status.fully_retired);
  assert(!status.safe_to_release_registered_storage);

  edge0.backpressure_publications = 0;
  drive_until_retired(executor);
  status = executor.status();
  assert(status.fully_retired);
}

void test_explicit_drain_reports_the_registered_storage_release_gate() {
  constexpr std::uintptr_t engine_identity = 0x89ab;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0xb000);
  LoopbackEdgePort edge1(1, engine_identity, 0xc000);
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(40),
      bulk, edge0, edge1);

  assert(executor.drain() == research::TiledDrainState::kIdle);
  executor.begin(40);
  research::TiledDrainState drain = research::TiledDrainState::kPending;
  for (int iteration = 0; iteration < 64; ++iteration) {
    drain = executor.drain();
    if (drain == research::TiledDrainState::kComplete) {
      break;
    }
  }
  assert(drain == research::TiledDrainState::kComplete);
  assert(executor.status().safe_to_release_registered_storage);
}

void test_impossible_and_regressing_peer_credits_poison_fail_closed() {
  constexpr std::uintptr_t engine_identity = 0x9abc;
  ScriptedBulkPort future_bulk;
  LoopbackEdgePort future0(0, engine_identity, 0xd000);
  LoopbackEdgePort future1(1, engine_identity, 0xe000);
  future0.loopback_credits = false;
  future1.loopback_credits = false;
  research::TiledExecutor future(
      spark_transport::make_tp4_tiled_session_storage_options(40),
      future_bulk, future0, future1);
  future.begin(40);
  future0.inject_peer_credit(1);
  (void)future.advance();
  auto status = future.status();
  assert(status.poisoned);
  assert(status.unexpected_generation_count == 1);
  assert(status.failure == research::TiledExecutorFailure::kCreditProtocol);
  assert(future.drain() == research::TiledDrainState::kPoisoned);
  assert(!status.safe_to_release_registered_storage);

  ScriptedBulkPort regression_bulk;
  LoopbackEdgePort regression0(0, engine_identity, 0xf000);
  LoopbackEdgePort regression1(1, engine_identity, 0x10000);
  regression0.loopback_credits = false;
  regression1.loopback_credits = false;
  research::TiledExecutor regression(
      spark_transport::make_tp4_tiled_session_storage_options(512),
      regression_bulk, regression0, regression1);
  regression.begin(512);
  for (int iteration = 0; iteration < 128; ++iteration) {
    (void)regression.advance();
    if (regression.status().output_ready_tiles == 8) {
      break;
    }
  }
  regression0.inject_peer_credit(7);
  (void)regression.advance();
  regression0.inject_peer_credit(6);
  (void)regression.advance();
  status = regression.status();
  assert(status.poisoned);
  assert(status.credit_regression_count == 1);
  assert(status.failure == research::TiledExecutorFailure::kCreditProtocol);
}

void test_constructor_requires_one_owner_and_two_distinct_stable_qps() {
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, 0xaaaa, 0x11000);
  LoopbackEdgePort wrong_owner(1, 0xbbbb, 0x12000);
  bool rejected_owner{};
  try {
    research::TiledExecutor executor(
        spark_transport::make_tp4_tiled_session_storage_options(40),
        bulk, edge0, wrong_owner);
  } catch (const std::invalid_argument&) {
    rejected_owner = true;
  }
  assert(rejected_owner);

  LoopbackEdgePort duplicate_qp(1, 0xaaaa, 0x11000);
  bool rejected_qp{};
  try {
    research::TiledExecutor executor(
        spark_transport::make_tp4_tiled_session_storage_options(40),
        bulk, edge0, duplicate_qp);
  } catch (const std::invalid_argument&) {
    rejected_qp = true;
  }
  assert(rejected_qp);
}

void test_credit_wire_encoding_reserves_zero_and_rejects_exhaustion() {
  assert(research::tiled_wire_credit(0) == 1);
  assert(research::tiled_consumed_through_from_wire(1) == 0);

  bool rejected_zero{};
  try {
    (void)research::tiled_consumed_through_from_wire(0);
  } catch (const std::invalid_argument&) {
    rejected_zero = true;
  }
  assert(rejected_zero);

  bool rejected_exhaustion{};
  try {
    (void)research::tiled_wire_credit(
        std::numeric_limits<std::uint64_t>::max());
  } catch (const std::overflow_error&) {
    rejected_exhaustion = true;
  }
  assert(rejected_exhaustion);
}

void test_zero_wire_credit_is_unpublished_and_not_ordinal_zero() {
  constexpr std::uintptr_t engine_identity = 0xabcd;
  ScriptedBulkPort bulk;
  LoopbackEdgePort edge0(0, engine_identity, 0x13000);
  LoopbackEdgePort edge1(1, engine_identity, 0x14000);
  edge0.loopback_credits = false;
  edge1.loopback_credits = false;
  research::TiledExecutor executor(
      spark_transport::make_tp4_tiled_session_storage_options(40),
      bulk, edge0, edge1);
  executor.begin(40);
  edge0.inject_peer_wire_credit(0);
  (void)executor.advance();
  assert(!executor.status().poisoned);
  assert(!executor.status().fully_retired);
}

}  // namespace

int main() {
  test_one_tile_runs_through_gpu_and_both_stable_edge_qps();
  test_backpressure_retries_without_duplicate_posts_or_poison();
  test_output_ready_precedes_drain_until_outbound_and_inbound_credits();
  test_ninth_tile_waits_for_both_edge_watermarks();
  test_fast_peer_credits_never_exceed_the_physical_slot_window();
  test_cumulative_credit_never_overwrites_an_uncompleted_publish();
  test_backpressured_new_credit_blocks_next_operation_retirement();
  test_explicit_drain_reports_the_registered_storage_release_gate();
  test_impossible_and_regressing_peer_credits_poison_fail_closed();
  test_constructor_requires_one_owner_and_two_distinct_stable_qps();
  test_credit_wire_encoding_reserves_zero_and_rejects_exhaustion();
  test_zero_wire_credit_is_unpublished_and_not_ordinal_zero();
}
