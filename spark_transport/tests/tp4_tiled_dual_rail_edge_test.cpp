#include "tiled_dual_rail_edge.hpp"

#include <cassert>
#include <cstdint>
#include <deque>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;

namespace {

class ScriptedRail final : public research::TiledEdgePort {
 public:
  ScriptedRail(std::uint32_t edge, std::uintptr_t engine,
               std::uintptr_t qp)
      : edge_(edge), engine_(engine), qp_(qp) {}

  std::uint32_t edge_index() const noexcept override { return edge_; }
  std::uintptr_t engine_identity() const noexcept override { return engine_; }
  std::uintptr_t qp_identity() const noexcept override { return qp_; }

  research::TiledSubmitState try_post_exchange(
      const research::TiledEdgeExchangeRequest& request) override {
    if (fatal) return research::TiledSubmitState::kFatal;
    if (exchange_backpressure != 0) {
      --exchange_backpressure;
      return research::TiledSubmitState::kBackpressured;
    }
    exchanges.push_back(request);
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll_exchange(
      const research::TiledEdgeExchangeRequest&) override {
    if (fatal) return research::TiledPollState::kFatal;
    if (exchange_pending != 0) {
      --exchange_pending;
      return research::TiledPollState::kPending;
    }
    return research::TiledPollState::kComplete;
  }

  research::TiledSubmitState try_publish_consumed_through(
      const research::TiledCreditPublishRequest& request) override {
    if (fatal) return research::TiledSubmitState::kFatal;
    if (credit_backpressure != 0) {
      --credit_backpressure;
      return research::TiledSubmitState::kBackpressured;
    }
    credits.push_back(request);
    return research::TiledSubmitState::kAccepted;
  }

  research::TiledPollState poll_published_consumed_through(
      const research::TiledCreditPublishRequest&) override {
    if (fatal) return research::TiledPollState::kFatal;
    if (credit_pending != 0) {
      --credit_pending;
      return research::TiledPollState::kPending;
    }
    return research::TiledPollState::kComplete;
  }

  research::TiledCreditPollState poll_peer_consumed_through(
      const research::TiledCreditObserveRequest&,
      std::uint64_t& wire_credit) override {
    if (fatal) return research::TiledCreditPollState::kFatal;
    if (peer_credits.empty()) {
      return research::TiledCreditPollState::kNoUpdate;
    }
    wire_credit = peer_credits.front();
    peer_credits.pop_front();
    return research::TiledCreditPollState::kUpdate;
  }

  std::vector<research::TiledEdgeExchangeRequest> exchanges;
  std::vector<research::TiledCreditPublishRequest> credits;
  std::deque<std::uint64_t> peer_credits;
  std::uint32_t exchange_backpressure{};
  std::uint32_t exchange_pending{};
  std::uint32_t credit_backpressure{};
  std::uint32_t credit_pending{};
  bool fatal{};

 private:
  std::uint32_t edge_{};
  std::uintptr_t engine_{};
  std::uintptr_t qp_{};
};

research::TiledEdgeExchangeRequest exchange_request() {
  return {research::TiledExchangePhase::kPhase1,
          {3, 2},
          18,
          19,
          55,
          4096,
          8192,
          128,
          192,
          512U * 1024U,
          0,
          1};
}

research::TiledCreditPublishRequest credit_request() {
  return {18, 19, 57, 256, 320, 0};
}

void test_exchange_splits_contiguous_aligned_ranges_and_retries_once() {
  ScriptedRail rail0(0, 0x1234, 0x10);
  ScriptedRail rail1(0, 0x1234, 0x11);
  rail1.exchange_backpressure = 1;
  research::DualRailStripedEdgePort edge(0, rail0, rail1);
  const auto request = exchange_request();

  assert(edge.try_post_exchange(request) ==
         research::TiledSubmitState::kBackpressured);
  assert(rail0.exchanges.size() == 1);
  assert(rail1.exchanges.empty());
  assert(edge.try_post_exchange(request) ==
         research::TiledSubmitState::kAccepted);
  assert(rail0.exchanges.size() == 1);
  assert(rail1.exchanges.size() == 1);

  const auto& lower = rail0.exchanges.front();
  const auto& upper = rail1.exchanges.front();
  assert(lower.active_bytes == 256U * 1024U);
  assert(upper.active_bytes == 256U * 1024U);
  assert(lower.local_payload_offset == request.local_payload_offset);
  assert(upper.local_payload_offset ==
         request.local_payload_offset + lower.active_bytes);
  assert(lower.remote_payload_offset == request.remote_payload_offset);
  assert(upper.remote_payload_offset ==
         request.remote_payload_offset + lower.active_bytes);
  assert(lower.local_doorbell_offset == upper.local_doorbell_offset);
  assert(lower.remote_doorbell_offset == upper.remote_doorbell_offset);

  rail1.exchange_pending = 1;
  assert(edge.poll_exchange(request) == research::TiledPollState::kPending);
  assert(edge.status().pending_exchanges == 1);
  assert(edge.poll_exchange(request) == research::TiledPollState::kComplete);
  assert(edge.status().pending_exchanges == 0);
}

void test_credit_publish_and_observe_require_both_rails() {
  ScriptedRail rail0(0, 0x1234, 0x20);
  ScriptedRail rail1(0, 0x1234, 0x21);
  research::DualRailStripedEdgePort edge(0, rail0, rail1);
  const auto credit = credit_request();

  rail1.credit_backpressure = 1;
  assert(edge.try_publish_consumed_through(credit) ==
         research::TiledSubmitState::kBackpressured);
  assert(rail0.credits.size() == 1 && rail1.credits.empty());
  assert(edge.try_publish_consumed_through(credit) ==
         research::TiledSubmitState::kAccepted);
  rail0.credit_pending = 1;
  assert(edge.poll_published_consumed_through(credit) ==
         research::TiledPollState::kPending);
  assert(edge.poll_published_consumed_through(credit) ==
         research::TiledPollState::kComplete);

  research::TiledCreditObserveRequest observe{320, 0};
  std::uint64_t wire{};
  rail0.peer_credits.push_back(11);
  assert(edge.poll_peer_consumed_through(observe, wire) ==
         research::TiledCreditPollState::kNoUpdate);
  rail1.peer_credits.push_back(8);
  assert(edge.poll_peer_consumed_through(observe, wire) ==
         research::TiledCreditPollState::kUpdate);
  assert(wire == 8);
  rail1.peer_credits.push_back(11);
  assert(edge.poll_peer_consumed_through(observe, wire) ==
         research::TiledCreditPollState::kUpdate);
  assert(wire == 11);
}

void test_drain_is_observable_and_child_failure_poison_is_permanent() {
  ScriptedRail rail0(0, 0x1234, 0x30);
  ScriptedRail rail1(0, 0x1234, 0x31);
  research::DualRailStripedEdgePort edge(0, rail0, rail1);
  assert(edge.drain() == research::DualRailDrainState::kIdle);

  const auto request = exchange_request();
  assert(edge.try_post_exchange(request) ==
         research::TiledSubmitState::kAccepted);
  rail1.exchange_pending = 1;
  assert(edge.drain() == research::DualRailDrainState::kPending);
  assert(!edge.status().safe_to_release_registered_storage);
  assert(edge.drain() == research::DualRailDrainState::kComplete);
  assert(edge.status().safe_to_release_registered_storage);

  const auto credit = credit_request();
  rail1.fatal = true;
  assert(edge.try_publish_consumed_through(credit) ==
         research::TiledSubmitState::kFatal);
  assert(edge.status().poisoned);
  assert(!edge.status().safe_to_release_registered_storage);
  assert(edge.drain() == research::DualRailDrainState::kPoisoned);
}

void test_invalid_or_regressing_rail_credit_fails_closed() {
  ScriptedRail rail0(0, 0x1234, 0x40);
  ScriptedRail rail1(0, 0x1234, 0x41);
  research::DualRailStripedEdgePort edge(0, rail0, rail1);
  research::TiledCreditObserveRequest observe{320, 0};
  std::uint64_t wire{};
  rail0.peer_credits.push_back(5);
  rail1.peer_credits.push_back(5);
  assert(edge.poll_peer_consumed_through(observe, wire) ==
         research::TiledCreditPollState::kUpdate);
  rail0.peer_credits.push_back(4);
  assert(edge.poll_peer_consumed_through(observe, wire) ==
         research::TiledCreditPollState::kFatal);
  assert(edge.status().poisoned);
}

void test_wrong_logical_edge_fails_closed_before_reaching_children() {
  ScriptedRail rail0(0, 0x1234, 0x50);
  ScriptedRail rail1(0, 0x1234, 0x51);
  research::DualRailStripedEdgePort edge(0, rail0, rail1);
  auto credit = credit_request();
  credit.edge = 1;
  assert(edge.try_publish_consumed_through(credit) ==
         research::TiledSubmitState::kFatal);
  assert(rail0.credits.empty() && rail1.credits.empty());
  assert(edge.status().poisoned);
}

}  // namespace

int main() {
  test_exchange_splits_contiguous_aligned_ranges_and_retries_once();
  test_credit_publish_and_observe_require_both_rails();
  test_drain_is_observable_and_child_failure_poison_is_permanent();
  test_invalid_or_regressing_rail_credit_fails_closed();
  test_wrong_logical_edge_fails_closed_before_reaching_children();
}
