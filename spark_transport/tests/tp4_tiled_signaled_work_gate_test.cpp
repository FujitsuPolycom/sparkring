#include "tiled_signaled_work_gate.hpp"

#include <cassert>
#include <cstdint>

namespace research = spark_transport::tiled_prefill_research;

namespace {

void test_scheduled_delayed_credit_reserves_the_signaled_cq_lane() {
  research::TiledSignaledWorkGate gate(1000);

  assert(gate.try_begin_credit(0) ==
         research::TiledCreditBeginState::kDelaying);
  for (std::uint64_t tick = 1; tick < 1000; ++tick) {
    assert(!gate.try_begin_exchange());
    assert(gate.try_begin_credit(tick) ==
           research::TiledCreditBeginState::kDelaying);
  }
  assert(gate.try_begin_credit(1000) ==
         research::TiledCreditBeginState::kReady);
  assert(!gate.try_begin_exchange());
  gate.complete_credit();
  assert(gate.try_begin_exchange());
  gate.complete_exchange();
}

void test_zero_delay_credit_posts_without_a_delay_state() {
  research::TiledSignaledWorkGate gate(0);
  assert(gate.try_begin_credit(7) ==
         research::TiledCreditBeginState::kReady);
  gate.complete_credit();
  assert(gate.try_begin_exchange());
  gate.complete_exchange();
}

}  // namespace

int main() {
  test_scheduled_delayed_credit_reserves_the_signaled_cq_lane();
  test_zero_delay_credit_posts_without_a_delay_state();
}
