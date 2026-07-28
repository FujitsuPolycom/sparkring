#include "spark_transport/tp4_indexer_graph.hpp"

namespace spark_transport {

bool tp4_indexer_graph_descriptor_from_q(
    std::uint32_t q,
    Tp4IndexerGraphDescriptor* descriptor) noexcept {
  if (descriptor == nullptr || q < 1 ||
      q > kTp4IndexerGraphMaximumQ) {
    return false;
  }
  Tp4IndexerGraphDescriptor candidate{};
  candidate.q = q;
  candidate.input_bytes = q * kTp4IndexerGraphBytesPerRow;
  candidate.output_bytes =
      candidate.input_bytes * kTp4IndexerGraphWorldSize;
  if (!tp4_indexer_graph_descriptor_valid(candidate)) {
    return false;
  }
  *descriptor = candidate;
  return true;
}

bool tp4_indexer_graph_descriptor_from_input_bytes(
    std::uint32_t input_bytes,
    Tp4IndexerGraphDescriptor* descriptor) noexcept {
  if (input_bytes == 0 ||
      input_bytes % kTp4IndexerGraphBytesPerRow != 0) {
    return false;
  }
  return tp4_indexer_graph_descriptor_from_q(
      input_bytes / kTp4IndexerGraphBytesPerRow, descriptor);
}

Tp4IndexerGraphFoundation::Tp4IndexerGraphFoundation(
    Tp4GraphCommandRing* command_ring) noexcept
    : command_ring_(command_ring) {}

bool Tp4IndexerGraphFoundation::capture_q(std::uint32_t q) noexcept {
  Tp4IndexerGraphDescriptor descriptor{};
  if (!tp4_indexer_graph_descriptor_from_q(q, &descriptor)) {
    rejected_captures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return capture_descriptor(descriptor);
}

bool Tp4IndexerGraphFoundation::capture_descriptor(
    const Tp4IndexerGraphDescriptor& descriptor) noexcept {
  if (command_ring_ == nullptr ||
      !tp4_indexer_graph_descriptor_valid(descriptor) ||
      tp4_graph_command_published(command_ring_) != 0 ||
      tp4_graph_command_consumed(command_ring_) != 0 ||
      tp4_graph_command_completed(command_ring_) != 0 ||
      tp4_graph_command_overflow(command_ring_) != 0) {
    rejected_captures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  captured_q_mask_.fetch_or(q_bit(descriptor.q),
                            std::memory_order_release);
  captured_nodes_.fetch_add(1, std::memory_order_release);
  return true;
}

bool Tp4IndexerGraphFoundation::publish_reference_replay(
    const Tp4IndexerGraphDescriptor& descriptor, bool trace,
    std::uint64_t* sequence) noexcept {
  const std::uint64_t captured_q =
      captured_q_mask_.load(std::memory_order_acquire);
  if (command_ring_ == nullptr ||
      !tp4_indexer_graph_descriptor_valid(descriptor) ||
      (captured_q & q_bit(descriptor.q)) == 0) {
    rejected_reference_replays_.fetch_add(
        1, std::memory_order_relaxed);
    return false;
  }
  const bool published = tp4_graph_command_publish_tagged_layout(
      command_ring_, trace,
      Tp4GraphCommandKind::kIndexerAllgather,
      kTp4IndexerGraphDescriptorVersion, descriptor.q,
      descriptor.input_bytes, kTp4IndexerGraphBytesPerRow,
      kTp4IndexerGraphMaximumQ, sequence);
  if (!published) {
    rejected_reference_replays_.fetch_add(
        1, std::memory_order_relaxed);
  }
  return published;
}

bool Tp4IndexerGraphFoundation::try_consume_next(
    Tp4GraphCommand* command) noexcept {
  if (command_ring_ == nullptr || command == nullptr) {
    return false;
  }
  const std::uint64_t expected =
      tp4_graph_command_consumed(command_ring_) + 1;
  const std::uint64_t published =
      tp4_graph_command_published(command_ring_);
  if (published < expected) {
    return false;
  }
  if (tp4_graph_command_try_consume_tagged_layout(
          command_ring_, expected,
          Tp4GraphCommandKind::kIndexerAllgather,
          kTp4IndexerGraphDescriptorVersion,
          kTp4IndexerGraphBytesPerRow,
          kTp4IndexerGraphMaximumQ, command)) {
    return true;
  }
  invalid_commands_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool Tp4IndexerGraphFoundation::complete(
    std::uint64_t sequence) noexcept {
  if (command_ring_ == nullptr || sequence == 0) {
    invalid_commands_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (tp4_graph_command_overflow(command_ring_) != 0) {
    invalid_commands_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (tp4_graph_command_consumed(command_ring_) != sequence ||
      tp4_graph_command_completed(command_ring_) + 1 != sequence) {
    invalid_commands_.fetch_add(1, std::memory_order_relaxed);
    tp4_graph_command_complete(command_ring_, sequence);
    return false;
  }
  tp4_graph_command_complete(command_ring_, sequence);
  if (tp4_graph_command_completed(command_ring_) != sequence) {
    invalid_commands_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

Tp4IndexerGraphStatus Tp4IndexerGraphFoundation::status()
    const noexcept {
  return Tp4IndexerGraphStatus{
      captured_nodes_.load(std::memory_order_acquire),
      captured_q_mask_.load(std::memory_order_acquire),
      rejected_captures_.load(std::memory_order_acquire),
      rejected_reference_replays_.load(std::memory_order_acquire),
      invalid_commands_.load(std::memory_order_acquire),
      tp4_graph_command_published(command_ring_),
      tp4_graph_command_consumed(command_ring_),
      tp4_graph_command_completed(command_ring_),
      tp4_graph_command_overflow(command_ring_),
      captured_nodes_.load(std::memory_order_acquire) != 0};
}

}  // namespace spark_transport
