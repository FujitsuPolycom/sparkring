#include "spark_transport/tp4_graph_command.hpp"

#include <limits>

namespace spark_transport {
namespace {

std::uint64_t load_acquire(const std::uint64_t* address) noexcept {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_release(std::uint64_t* address, std::uint64_t value) noexcept {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

bool compare_exchange(std::uint64_t* address, std::uint64_t* expected,
                      std::uint64_t desired) noexcept {
  return __atomic_compare_exchange_n(
      address, expected, desired, false, __ATOMIC_ACQ_REL,
      __ATOMIC_ACQUIRE);
}

void record_first_overflow(Tp4GraphCommandRing* ring,
                           std::uint64_t sequence) noexcept {
  std::uint64_t expected{};
  (void)compare_exchange(
      &ring->producer.overflow_sequence, &expected, sequence);
}

bool try_consume(Tp4GraphCommandRing* ring,
                 std::uint64_t expected_sequence,
                 Tp4GraphCommand* command,
                 std::uint32_t bytes_per_row,
                 std::uint32_t maximum_q,
                 bool validate_tag = false,
                 Tp4GraphCommandKind expected_kind =
                     Tp4GraphCommandKind::kLegacy,
                 std::uint32_t expected_parameter = 0) noexcept {
  if (ring == nullptr || command == nullptr || expected_sequence == 0 ||
      load_acquire(&ring->producer.overflow_sequence) != 0 ||
      load_acquire(&ring->producer.published_sequence) <
          expected_sequence) {
    return false;
  }

  const auto& slot =
      ring->commands[(expected_sequence - 1) %
                     kTp4GraphCommandCapacity];
  if (load_acquire(&slot.sequence) != expected_sequence) {
    return false;
  }
  if (bytes_per_row != 0 &&
      !tp4_graph_command_layout_valid(
          slot.q, slot.payload_bytes, bytes_per_row, maximum_q)) {
    record_first_overflow(ring, expected_sequence);
    return false;
  }
  if (validate_tag &&
      (slot.kind != expected_kind ||
       slot.parameter != expected_parameter)) {
    record_first_overflow(ring, expected_sequence);
    return false;
  }

  command->sequence = expected_sequence;
  command->trace = slot.trace;
  command->q = slot.q;
  command->payload_bytes = slot.payload_bytes;
  command->kind = slot.kind;
  command->parameter = slot.parameter;
  store_release(&ring->consumer.consumed_sequence, expected_sequence);
  return true;
}

bool publish_descriptor(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes, std::uint32_t bytes_per_row,
    std::uint32_t maximum_q, Tp4GraphCommandKind kind,
    std::uint32_t parameter,
    std::uint64_t* sequence) noexcept {
  if (ring == nullptr || !tp4_graph_command_kind_valid(kind) ||
      !tp4_graph_command_layout_valid(
          q, payload_bytes, bytes_per_row, maximum_q)) {
    return false;
  }

  std::uint64_t claimed =
      load_acquire(&ring->producer.claimed_sequence);
  while (true) {
    if (claimed >= kTp4GraphMaximumDoorbellSequence) {
      store_release(&ring->producer.overflow_sequence, claimed);
      return false;
    }
    const std::uint64_t next = claimed + 1;
    const std::uint64_t completed =
        load_acquire(&ring->consumer.completed_sequence);
    if (completed > claimed) {
      store_release(&ring->producer.overflow_sequence, next);
      return false;
    }
    if (next - completed > kTp4GraphCommandCapacity) {
      return false;
    }
    if (compare_exchange(
            &ring->producer.claimed_sequence, &claimed, next)) {
      claimed = next;
      break;
    }
  }

  auto& command =
      ring->commands[(claimed - 1) % kTp4GraphCommandCapacity];
  command.trace = trace ? 1U : 0U;
  command.q = q;
  command.payload_bytes = payload_bytes;
  command.kind = kind;
  command.parameter = parameter;
  store_release(&command.sequence, claimed);

  std::uint64_t expected = claimed - 1;
  if (!compare_exchange(
          &ring->producer.published_sequence, &expected, claimed)) {
    store_release(&ring->producer.overflow_sequence, claimed);
    return false;
  }
  if (sequence != nullptr) {
    *sequence = claimed;
  }
  return true;
}

}  // namespace

bool tp4_graph_command_publish(Tp4GraphCommandRing* ring, bool trace,
                               std::uint64_t* sequence) noexcept {
  return tp4_graph_command_publish_descriptor(
      ring, trace, 1, tp4_graph_payload_bytes(1), sequence);
}

bool tp4_graph_command_publish_descriptor(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes, std::uint64_t* sequence) noexcept {
  return publish_descriptor(
      ring, trace, q, payload_bytes,
      kTp4GraphElementsPerRow * kTp4GraphBytesPerElement,
      kTp4GraphMaximumQ, Tp4GraphCommandKind::kLegacy, 0,
      sequence);
}

bool tp4_graph_allreduce_command_publish_descriptor(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes, std::uint64_t* sequence) noexcept {
  return publish_descriptor(
      ring, trace, q, payload_bytes,
      kTp4GraphElementsPerRow * kTp4GraphBytesPerElement,
      kTp4GraphAllreduceMaximumQ, Tp4GraphCommandKind::kLegacy, 0,
      sequence);
}

bool tp4_graph_command_publish_tagged_layout(
    Tp4GraphCommandRing* ring, bool trace, Tp4GraphCommandKind kind,
    std::uint32_t parameter, std::uint32_t q,
    std::uint32_t payload_bytes, std::uint32_t bytes_per_row,
    std::uint32_t maximum_q, std::uint64_t* sequence) noexcept {
  if (kind == Tp4GraphCommandKind::kLegacy) {
    return false;
  }
  return publish_descriptor(
      ring, trace, q, payload_bytes, bytes_per_row, maximum_q,
      kind, parameter, sequence);
}

bool tp4_graph_command_try_consume(Tp4GraphCommandRing* ring,
                                   std::uint64_t expected_sequence,
                                   Tp4GraphCommand* command) noexcept {
  return try_consume(
      ring, expected_sequence, command, 0, 0);
}

bool tp4_graph_command_try_peek(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommand* command) noexcept {
  if (ring == nullptr || command == nullptr || expected_sequence == 0 ||
      load_acquire(&ring->producer.overflow_sequence) != 0 ||
      load_acquire(&ring->producer.published_sequence) <
          expected_sequence) {
    return false;
  }
  const auto& slot =
      ring->commands[(expected_sequence - 1) %
                     kTp4GraphCommandCapacity];
  if (load_acquire(&slot.sequence) != expected_sequence) {
    return false;
  }
  command->sequence = expected_sequence;
  command->trace = slot.trace;
  command->q = slot.q;
  command->payload_bytes = slot.payload_bytes;
  command->kind = slot.kind;
  command->parameter = slot.parameter;
  return true;
}

bool tp4_graph_command_try_consume_descriptor(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommand* command) noexcept {
  return try_consume(
      ring, expected_sequence, command,
      kTp4GraphElementsPerRow * kTp4GraphBytesPerElement,
      kTp4GraphMaximumQ);
}

bool tp4_graph_allreduce_command_try_consume_descriptor(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommand* command) noexcept {
  return try_consume(
      ring, expected_sequence, command,
      kTp4GraphElementsPerRow * kTp4GraphBytesPerElement,
      kTp4GraphAllreduceMaximumQ);
}

bool tp4_graph_command_try_consume_layout(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    std::uint32_t bytes_per_row, std::uint32_t maximum_q,
    Tp4GraphCommand* command) noexcept {
  if (bytes_per_row == 0 || maximum_q == 0) {
    return false;
  }
  return try_consume(
      ring, expected_sequence, command, bytes_per_row, maximum_q);
}

bool tp4_graph_command_try_consume_tagged_layout(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommandKind expected_kind,
    std::uint32_t expected_parameter, std::uint32_t bytes_per_row,
    std::uint32_t maximum_q, Tp4GraphCommand* command) noexcept {
  if (bytes_per_row == 0 || maximum_q == 0 ||
      expected_kind == Tp4GraphCommandKind::kLegacy ||
      !tp4_graph_command_kind_valid(expected_kind)) {
    return false;
  }
  return try_consume(
      ring, expected_sequence, command, bytes_per_row, maximum_q,
      true, expected_kind, expected_parameter);
}

void tp4_graph_command_complete(Tp4GraphCommandRing* ring,
                                std::uint64_t sequence) noexcept {
  if (ring == nullptr || sequence == 0) {
    return;
  }
  const std::uint64_t completed =
      load_acquire(&ring->consumer.completed_sequence);
  if (completed == std::numeric_limits<std::uint64_t>::max() ||
      sequence != completed + 1) {
    store_release(&ring->producer.overflow_sequence, sequence);
    return;
  }
  store_release(&ring->consumer.completed_sequence, sequence);
}

std::uint64_t tp4_graph_command_published(
    const Tp4GraphCommandRing* ring) noexcept {
  return ring == nullptr
             ? 0
             : load_acquire(&ring->producer.published_sequence);
}

std::uint64_t tp4_graph_command_claimed(
    const Tp4GraphCommandRing* ring) noexcept {
  return ring == nullptr
             ? 0
             : load_acquire(&ring->producer.claimed_sequence);
}

std::uint64_t tp4_graph_command_consumed(
    const Tp4GraphCommandRing* ring) noexcept {
  return ring == nullptr
             ? 0
             : load_acquire(&ring->consumer.consumed_sequence);
}

std::uint64_t tp4_graph_command_completed(
    const Tp4GraphCommandRing* ring) noexcept {
  return ring == nullptr
             ? 0
             : load_acquire(&ring->consumer.completed_sequence);
}

std::uint64_t tp4_graph_command_overflow(
    const Tp4GraphCommandRing* ring) noexcept {
  return ring == nullptr
             ? 0
             : load_acquire(&ring->producer.overflow_sequence);
}

}  // namespace spark_transport
