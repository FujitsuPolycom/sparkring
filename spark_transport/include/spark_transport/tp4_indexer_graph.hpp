#pragma once

#include "spark_transport/tp4_graph_command.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace spark_transport {

// One semantic family covers every exact contiguous INT32 [Q, 2, 2048]
// indexer input for Q1..Q40. The output is four rank-ordered input segments.
// Q is dynamic at the session level but fixed in each captured graph node.
constexpr std::uint32_t kTp4IndexerGraphWorldSize = 4;
constexpr std::uint32_t kTp4IndexerGraphMaximumQ = 40;
constexpr std::uint32_t kTp4IndexerGraphElementsPerRow = 2U * 2048U;
constexpr std::uint32_t kTp4IndexerGraphBytesPerElement = 4;
constexpr std::uint32_t kTp4IndexerGraphBytesPerRow =
    kTp4IndexerGraphElementsPerRow *
    kTp4IndexerGraphBytesPerElement;
constexpr std::uint32_t kTp4IndexerGraphMaximumInputBytes =
    kTp4IndexerGraphMaximumQ * kTp4IndexerGraphBytesPerRow;
constexpr std::uint32_t kTp4IndexerGraphMaximumOutputBytes =
    kTp4IndexerGraphWorldSize *
    kTp4IndexerGraphMaximumInputBytes;
constexpr std::uint32_t kTp4IndexerGraphControlBytes = 64;
constexpr std::uint32_t kTp4IndexerGraphRound0ArenaBytes =
    2U * kTp4IndexerGraphMaximumInputBytes +
    kTp4IndexerGraphControlBytes;
constexpr std::uint32_t kTp4IndexerGraphRound1ArenaBytes =
    4U * kTp4IndexerGraphMaximumInputBytes +
    kTp4IndexerGraphControlBytes;
constexpr std::uint32_t kTp4IndexerGraphMappedArenaBytes =
    kTp4IndexerGraphRound0ArenaBytes +
    kTp4IndexerGraphRound1ArenaBytes +
    sizeof(Tp4GraphCommandRing);

// The parameter is a semantic-layout version, not a shape or port slot.
// Together with kIndexerAllgather it prevents a same-byte payload from
// another collective family from entering this progress engine.
constexpr std::uint32_t kTp4IndexerGraphDescriptorVersion = 1;

// The graph family uses one audited pair outside the formula all-reduce and
// exact all-gather namespaces. Q must never allocate a new port slot.
constexpr std::uint16_t kTp4IndexerGraphDefaultControlPort0 = 9462;
constexpr std::uint16_t kTp4IndexerGraphDefaultControlPort1 = 9463;

struct Tp4IndexerGraphDescriptor {
  Tp4GraphCommandKind family{
      Tp4GraphCommandKind::kIndexerAllgather};
  std::uint32_t version{kTp4IndexerGraphDescriptorVersion};
  std::uint32_t q{};
  std::uint32_t input_bytes{};
  std::uint32_t output_bytes{};
};

struct Tp4IndexerGraphArenaContract {
  std::uint32_t maximum_q{kTp4IndexerGraphMaximumQ};
  std::uint32_t input_capacity_bytes{
      kTp4IndexerGraphMaximumInputBytes};
  std::uint32_t round0_transfer_capacity_bytes{
      kTp4IndexerGraphMaximumInputBytes};
  std::uint32_t round1_transfer_capacity_bytes{
      2U * kTp4IndexerGraphMaximumInputBytes};
  std::uint32_t output_capacity_bytes{
      kTp4IndexerGraphMaximumOutputBytes};
  std::uint32_t round0_arena_bytes{
      kTp4IndexerGraphRound0ArenaBytes};
  std::uint32_t round1_arena_bytes{
      kTp4IndexerGraphRound1ArenaBytes};
  std::uint32_t command_ring_bytes{
      sizeof(Tp4GraphCommandRing)};
  std::uint32_t total_mapped_bytes{
      kTp4IndexerGraphMappedArenaBytes};
};

struct Tp4IndexerGraphPortPair {
  std::uint16_t control_port0{};
  std::uint16_t control_port1{};
};

struct Tp4IndexerGraphStatus {
  std::uint64_t captured_nodes{};
  std::uint64_t captured_q_mask{};
  std::uint64_t rejected_captures{};
  std::uint64_t rejected_reference_replays{};
  std::uint64_t invalid_commands{};
  std::uint64_t published_sequence{};
  std::uint64_t consumed_sequence{};
  std::uint64_t completed_sequence{};
  std::uint64_t overflow_sequence{};
  bool capture_configured{};
};

constexpr Tp4IndexerGraphArenaContract
tp4_indexer_graph_arena_contract() noexcept {
  return {};
}

constexpr bool tp4_indexer_graph_descriptor_valid(
    const Tp4IndexerGraphDescriptor& descriptor) noexcept {
  return descriptor.family ==
             Tp4GraphCommandKind::kIndexerAllgather &&
         descriptor.version ==
             kTp4IndexerGraphDescriptorVersion &&
         tp4_graph_command_layout_valid(
             descriptor.q, descriptor.input_bytes,
             kTp4IndexerGraphBytesPerRow,
             kTp4IndexerGraphMaximumQ) &&
         static_cast<std::uint64_t>(descriptor.input_bytes) *
                 kTp4IndexerGraphWorldSize ==
             descriptor.output_bytes &&
         descriptor.input_bytes <=
             kTp4IndexerGraphMaximumInputBytes &&
         descriptor.output_bytes <=
             kTp4IndexerGraphMaximumOutputBytes;
}

bool tp4_indexer_graph_descriptor_from_q(
    std::uint32_t q,
    Tp4IndexerGraphDescriptor* descriptor) noexcept;

bool tp4_indexer_graph_descriptor_from_input_bytes(
    std::uint32_t input_bytes,
    Tp4IndexerGraphDescriptor* descriptor) noexcept;

constexpr Tp4IndexerGraphPortPair
tp4_indexer_graph_default_port_pair() noexcept {
  return {
      kTp4IndexerGraphDefaultControlPort0,
      kTp4IndexerGraphDefaultControlPort1};
}

// GPU-free reference control plane for the future native session. Production
// capture will enqueue a graph-native publisher kernel into the active CUDA
// capture; this class deliberately never captures or launches the eager
// all-gather spin-wait kernel. It validates capture descriptors, models the
// shared command-ring sequence domain, and exposes counters used by tests and
// the eventual C status API.
class Tp4IndexerGraphFoundation {
 public:
  explicit Tp4IndexerGraphFoundation(
      Tp4GraphCommandRing* command_ring) noexcept;

  Tp4IndexerGraphFoundation(
      const Tp4IndexerGraphFoundation&) = delete;
  Tp4IndexerGraphFoundation& operator=(
      const Tp4IndexerGraphFoundation&) = delete;

  bool capture_q(std::uint32_t q) noexcept;
  bool capture_descriptor(
      const Tp4IndexerGraphDescriptor& descriptor) noexcept;

  // CPU publisher used only as the reference model for the graph publisher.
  // A live implementation must call gpu_graph_command::publish_command from
  // a capture-safe kernel with this same tag/version/Q/byte descriptor.
  bool publish_reference_replay(
      const Tp4IndexerGraphDescriptor& descriptor, bool trace,
      std::uint64_t* sequence) noexcept;

  bool try_consume_next(Tp4GraphCommand* command) noexcept;
  bool complete(std::uint64_t sequence) noexcept;

  Tp4IndexerGraphStatus status() const noexcept;

 private:
  static constexpr std::uint64_t q_bit(std::uint32_t q) noexcept {
    return q >= 1 && q <= kTp4IndexerGraphMaximumQ
               ? std::uint64_t{1} << (q - 1)
               : 0;
  }

  Tp4GraphCommandRing* command_ring_{};
  std::atomic<std::uint64_t> captured_nodes_{0};
  std::atomic<std::uint64_t> captured_q_mask_{0};
  std::atomic<std::uint64_t> rejected_captures_{0};
  std::atomic<std::uint64_t> rejected_reference_replays_{0};
  std::atomic<std::uint64_t> invalid_commands_{0};
};

static_assert(kTp4IndexerGraphMaximumQ < 64);
static_assert(kTp4IndexerGraphMaximumQ <=
              kTp4GraphDoorbellQMask);
static_assert(kTp4IndexerGraphMaximumInputBytes == 655360);
static_assert(kTp4IndexerGraphMaximumOutputBytes == 2621440);

}  // namespace spark_transport
