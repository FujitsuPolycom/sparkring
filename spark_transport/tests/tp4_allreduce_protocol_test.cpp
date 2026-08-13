#include "spark_transport/tp4_allreduce_protocol.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

int main() {
  using spark_transport::Tp4AllreduceProtocol;
  using spark_transport::Tp4CreditReuseState;
  using spark_transport::parse_tp4_allreduce_protocol;
  using spark_transport::tp4_allreduce_protocol_from_wire;
  using spark_transport::tp4_credit_reuse_state;
  using spark_transport::tp4_expected_reuse_credit;
  using spark_transport::tp4_latest_slot_sequence;
  using spark_transport::tp4_payload_arena_bytes;
  using spark_transport::tp4_payload_slot_count;
  using spark_transport::tp4_payload_slot_index;

  const auto serial = Tp4AllreduceProtocol::kSerialAck;
  const auto deferred = Tp4AllreduceProtocol::kTwoSlotDeferredAck;
  assert(tp4_payload_slot_count(serial) == 1);
  assert(tp4_payload_slot_count(deferred) == 2);
  assert(tp4_payload_slot_index(1, deferred) == 0);
  assert(tp4_payload_slot_index(2, deferred) == 1);
  assert(tp4_payload_slot_index(3, deferred) == 0);
  assert(tp4_expected_reuse_credit(1, deferred) == 0);
  assert(tp4_expected_reuse_credit(2, deferred) == 0);
  assert(tp4_expected_reuse_credit(3, deferred) == 1);

  // Sequence 2 can occupy slot 1 while slot 0 remains uncredited.
  assert(tp4_credit_reuse_state(2, 0, deferred) ==
         Tp4CreditReuseState::kReady);
  // Sequence 3 cannot reuse slot 0 until its exact sequence-1 credit arrives.
  assert(tp4_credit_reuse_state(3, 0, deferred) ==
         Tp4CreditReuseState::kWaiting);
  assert(tp4_credit_reuse_state(3, 1, deferred) ==
         Tp4CreditReuseState::kReady);
  assert(tp4_credit_reuse_state(3, 2, deferred) ==
         Tp4CreditReuseState::kFutureGeneration);

  // The two edges retire independently; one edge cannot release the other.
  assert(tp4_credit_reuse_state(5, 3, deferred) ==
         Tp4CreditReuseState::kReady);
  assert(tp4_credit_reuse_state(5, 1, deferred) ==
         Tp4CreditReuseState::kWaiting);

  for (std::uint64_t sequence = 1; sequence <= 133; ++sequence) {
    const std::uint64_t expected =
        tp4_expected_reuse_credit(sequence, deferred);
    assert(tp4_credit_reuse_state(sequence, expected, deferred) ==
           Tp4CreditReuseState::kReady);
    if (expected != 0) {
      assert(tp4_credit_reuse_state(sequence, expected - 1, deferred) ==
             Tp4CreditReuseState::kWaiting);
    }
  }
  assert(tp4_latest_slot_sequence(133, 0, deferred) == 133);
  assert(tp4_latest_slot_sequence(133, 1, deferred) == 132);
  assert(tp4_latest_slot_sequence(2, 0, deferred) == 1);
  assert(tp4_latest_slot_sequence(2, 1, deferred) == 2);

  assert(tp4_payload_arena_bytes(64, serial) == 64);
  assert(tp4_payload_arena_bytes(64, deferred) == 128);
  bool rejected_overflow{};
  try {
    (void)tp4_payload_arena_bytes(
        std::numeric_limits<std::size_t>::max(), deferred);
  } catch (const std::overflow_error&) {
    rejected_overflow = true;
  }
  assert(rejected_overflow);

  assert(parse_tp4_allreduce_protocol("serial_ack") == serial);
  assert(parse_tp4_allreduce_protocol("two_slot_deferred_ack") == deferred);
  assert(tp4_allreduce_protocol_from_wire(0) == serial);
  assert(tp4_allreduce_protocol_from_wire(1) == deferred);
  bool rejected_protocol{};
  try {
    (void)tp4_allreduce_protocol_from_wire(2);
  } catch (const std::invalid_argument& error) {
    rejected_protocol =
        std::string(error.what()).find("protocol") != std::string::npos;
  }
  assert(rejected_protocol);
}
