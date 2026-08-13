#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace spark_transport {

enum class Tp4AllreduceProtocol : std::uint32_t {
  kSerialAck = 0,
  kTwoSlotDeferredAck = 1,
};

enum class Tp4CreditReuseState : std::uint8_t {
  kReady,
  kWaiting,
  kFutureGeneration,
};

constexpr std::uint16_t kTp4TwoSlotEndpointTag = 0xc202;

constexpr bool tp4_protocol_uses_deferred_ack(
    Tp4AllreduceProtocol protocol) noexcept {
  return protocol == Tp4AllreduceProtocol::kTwoSlotDeferredAck;
}

constexpr std::size_t tp4_payload_slot_count(
    Tp4AllreduceProtocol protocol) noexcept {
  return tp4_protocol_uses_deferred_ack(protocol) ? 2U : 1U;
}

constexpr std::uint16_t tp4_endpoint_protocol_tag(
    Tp4AllreduceProtocol protocol) noexcept {
  return tp4_protocol_uses_deferred_ack(protocol)
             ? kTp4TwoSlotEndpointTag
             : 0U;
}

constexpr std::size_t tp4_payload_slot_index(
    std::uint64_t sequence, Tp4AllreduceProtocol protocol) noexcept {
  if (sequence == 0) {
    return 0;
  }
  return static_cast<std::size_t>(
      (sequence - 1) % tp4_payload_slot_count(protocol));
}

constexpr std::uint64_t tp4_expected_reuse_credit(
    std::uint64_t sequence, Tp4AllreduceProtocol protocol) noexcept {
  const std::uint64_t slots = tp4_payload_slot_count(protocol);
  return tp4_protocol_uses_deferred_ack(protocol) && sequence > slots
             ? sequence - slots
             : 0;
}

constexpr Tp4CreditReuseState tp4_credit_reuse_state(
    std::uint64_t sequence, std::uint64_t observed_credit,
    Tp4AllreduceProtocol protocol) noexcept {
  const std::uint64_t expected =
      tp4_expected_reuse_credit(sequence, protocol);
  if (observed_credit == expected) {
    return Tp4CreditReuseState::kReady;
  }
  return observed_credit < expected
             ? Tp4CreditReuseState::kWaiting
             : Tp4CreditReuseState::kFutureGeneration;
}

constexpr std::uint64_t tp4_latest_slot_sequence(
    std::uint64_t final_sequence, std::size_t slot,
    Tp4AllreduceProtocol protocol) noexcept {
  const std::size_t slots = tp4_payload_slot_count(protocol);
  if (final_sequence == 0 || slot >= slots || final_sequence <= slot) {
    return 0;
  }
  const std::uint64_t distance =
      (final_sequence - 1 - slot) % slots;
  return final_sequence - distance;
}

inline std::size_t tp4_payload_arena_bytes(
    std::size_t slot_bytes, Tp4AllreduceProtocol protocol) {
  const std::size_t slots = tp4_payload_slot_count(protocol);
  if (slot_bytes == 0 ||
      slot_bytes > std::numeric_limits<std::size_t>::max() / slots) {
    throw std::overflow_error("TP4 payload-slot arena size overflow");
  }
  return slot_bytes * slots;
}

inline Tp4AllreduceProtocol tp4_allreduce_protocol_from_wire(
    std::uint32_t value) {
  switch (value) {
    case static_cast<std::uint32_t>(Tp4AllreduceProtocol::kSerialAck):
      return Tp4AllreduceProtocol::kSerialAck;
    case static_cast<std::uint32_t>(
        Tp4AllreduceProtocol::kTwoSlotDeferredAck):
      return Tp4AllreduceProtocol::kTwoSlotDeferredAck;
    default:
      throw std::invalid_argument("invalid TP4 all-reduce protocol");
  }
}

inline Tp4AllreduceProtocol parse_tp4_allreduce_protocol(
    std::string_view value) {
  if (value == "serial_ack") {
    return Tp4AllreduceProtocol::kSerialAck;
  }
  if (value == "two_slot_deferred_ack") {
    return Tp4AllreduceProtocol::kTwoSlotDeferredAck;
  }
  throw std::invalid_argument(
      "TP4 all-reduce protocol must be serial_ack or "
      "two_slot_deferred_ack");
}

inline const char* tp4_allreduce_protocol_name(
    Tp4AllreduceProtocol protocol) noexcept {
  return tp4_protocol_uses_deferred_ack(protocol)
             ? "two_slot_deferred_ack"
             : "serial_ack";
}

}  // namespace spark_transport
