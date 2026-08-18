#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace spark_transport {

constexpr std::size_t kTp4GraphCommandCapacity = 64;
constexpr std::uint32_t kTp4GraphElementsPerRow = 6144;
constexpr std::uint32_t kTp4GraphBytesPerElement = 2;
constexpr std::uint32_t kTp4GraphBytesPerRow =
    kTp4GraphElementsPerRow * kTp4GraphBytesPerElement;
// Eight active sequences at maximum MTP4 produce at most Q40.
constexpr std::uint32_t kTp4GraphMaximumQ = 40;
// All-reduce alone admits the first bounded prefill tier. Vocabulary and
// DCP descriptors remain capped by kTp4GraphMaximumQ.
constexpr std::uint32_t kTp4GraphAllreduceMaximumQ = 512;
// Q is encoded directly (not Q-1), so Q512 needs ten low-order bits.
constexpr std::uint32_t kTp4GraphDoorbellQBits = 10;
constexpr std::uint64_t kTp4GraphDoorbellQMask =
    (std::uint64_t{1} << kTp4GraphDoorbellQBits) - 1;
constexpr std::uint64_t kTp4GraphMaximumDoorbellSequence =
    std::numeric_limits<std::uint64_t>::max() >>
    kTp4GraphDoorbellQBits;
static_assert(kTp4GraphMaximumQ <= kTp4GraphDoorbellQMask);
static_assert(kTp4GraphAllreduceMaximumQ <= kTp4GraphDoorbellQMask);
static_assert(
    static_cast<std::uint64_t>(kTp4GraphAllreduceMaximumQ) *
        kTp4GraphElementsPerRow * kTp4GraphBytesPerElement <=
    std::numeric_limits<std::uint32_t>::max());

constexpr std::uint32_t tp4_graph_payload_bytes(
    std::uint32_t q,
    std::uint32_t bytes_per_row = kTp4GraphBytesPerRow) noexcept {
  return q * bytes_per_row;
}

constexpr bool tp4_graph_command_layout_valid(
    std::uint32_t q, std::uint32_t payload_bytes,
    std::uint32_t bytes_per_row, std::uint32_t maximum_q) noexcept {
  return bytes_per_row > 0 && maximum_q > 0 && q >= 1 &&
         q <= maximum_q &&
         static_cast<std::uint64_t>(q) * bytes_per_row ==
             payload_bytes;
}

constexpr bool tp4_graph_command_descriptor_valid(
    std::uint32_t q, std::uint32_t payload_bytes) noexcept {
  return tp4_graph_command_layout_valid(
      q, payload_bytes,
      kTp4GraphBytesPerRow,
      kTp4GraphMaximumQ);
}

constexpr bool tp4_graph_allreduce_command_descriptor_valid(
    std::uint32_t q, std::uint32_t payload_bytes) noexcept {
  return tp4_graph_command_layout_valid(
      q, payload_bytes,
      kTp4GraphBytesPerRow,
      kTp4GraphAllreduceMaximumQ);
}

constexpr bool tp4_graph_doorbell_token_valid(
    std::uint64_t sequence, std::uint32_t q) noexcept {
  return sequence >= 1 &&
         sequence <= kTp4GraphMaximumDoorbellSequence &&
         q >= 1 && q <= kTp4GraphAllreduceMaximumQ;
}

constexpr std::uint64_t tp4_graph_doorbell_token(
    std::uint64_t sequence, std::uint32_t q) noexcept {
  return (sequence << kTp4GraphDoorbellQBits) | q;
}

enum class Tp4GraphCommandKind : std::uint32_t {
  kLegacy = 0,
  kDcpQuery = 1,
  kDcpCombine = 2,
  kIndexerAllgather = 3,
};

constexpr bool tp4_graph_command_kind_valid(
    Tp4GraphCommandKind kind) noexcept {
  return kind == Tp4GraphCommandKind::kLegacy ||
         kind == Tp4GraphCommandKind::kDcpQuery ||
         kind == Tp4GraphCommandKind::kDcpCombine ||
         kind == Tp4GraphCommandKind::kIndexerAllgather;
}

// One cache line per command prevents the GPU publisher and verbs progress
// thread from sharing a line with an adjacent replay.
struct alignas(64) Tp4GraphCommand {
  std::uint64_t sequence{};
  std::uint64_t trace{};
  std::uint32_t q{};
  std::uint32_t payload_bytes{};
  Tp4GraphCommandKind kind{Tp4GraphCommandKind::kLegacy};
  std::uint32_t parameter{};
  std::uint64_t reserved[4]{};
};

struct alignas(64) Tp4GraphProducerState {
  std::uint64_t claimed_sequence{};
  std::uint64_t published_sequence{};
  std::uint64_t overflow_sequence{};
  std::uint64_t reserved[5]{};
};

struct alignas(64) Tp4GraphConsumerState {
  std::uint64_t consumed_sequence{};
  std::uint64_t completed_sequence{};
  std::uint64_t reserved[6]{};
};

// This object lives in CUDA-mapped pinned memory. Graph kernels atomically
// claim and publish replay sequences, while the verbs progress thread is the
// sole CPU consumer. Producer and consumer counters use separate cache lines
// to avoid coherence ping-pong during graph-ring polling.
struct alignas(64) Tp4GraphCommandRing {
  Tp4GraphProducerState producer{};
  Tp4GraphConsumerState consumer{};
  Tp4GraphCommand commands[kTp4GraphCommandCapacity]{};
};

static_assert(sizeof(Tp4GraphCommand) == 64);
static_assert(alignof(Tp4GraphCommand) == 64);
static_assert(offsetof(Tp4GraphCommand, q) == 16);
static_assert(offsetof(Tp4GraphCommand, payload_bytes) == 20);
static_assert(offsetof(Tp4GraphCommand, kind) == 24);
static_assert(offsetof(Tp4GraphCommand, parameter) == 28);
static_assert(offsetof(Tp4GraphCommand, reserved) == 32);
static_assert(std::is_standard_layout_v<Tp4GraphCommand>);
static_assert(sizeof(Tp4GraphProducerState) == 64);
static_assert(sizeof(Tp4GraphConsumerState) == 64);
static_assert(std::is_standard_layout_v<Tp4GraphProducerState>);
static_assert(std::is_standard_layout_v<Tp4GraphConsumerState>);
static_assert(offsetof(Tp4GraphCommandRing, consumer) == 64);
static_assert(offsetof(Tp4GraphCommandRing, commands) == 128);
static_assert(sizeof(Tp4GraphCommandRing) ==
              128 + kTp4GraphCommandCapacity * sizeof(Tp4GraphCommand));
static_assert(std::is_standard_layout_v<Tp4GraphCommandRing>);

// These functions contain no CUDA or verbs calls. The CPU publisher is the
// reference model used by layout tests; graph replay publishes from device
// code and the production progress thread only consumes/completes commands.
// A false publish with overflow_sequence still zero means ring backpressure,
// while a nonzero overflow_sequence denotes a broken ordering invariant.
//
// The legacy entrypoints retain the current Q1 device-publisher contract.
// New mixed-Q publishers/consumers must use the descriptor entrypoints,
// which validate q and payload_bytes before publication and consumption.
bool tp4_graph_command_publish(Tp4GraphCommandRing* ring, bool trace,
                               std::uint64_t* sequence) noexcept;

// Returns false without claiming a sequence when q/payload_bytes is invalid.
// Valid descriptors are fully written before sequence release-publication.
bool tp4_graph_command_publish_descriptor(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes, std::uint64_t* sequence) noexcept;

// All-reduce-only publisher. Unlike the shared Q40 descriptor entrypoint,
// this accepts exact contiguous BF16 rows through Q512.
bool tp4_graph_allreduce_command_publish_descriptor(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes, std::uint64_t* sequence) noexcept;

// Tagged descriptors let multiple collective families share one ordered graph
// ring. The generic layer validates the caller-supplied exact row layout; the
// family consumer must independently supply the same expected contract.
bool tp4_graph_command_publish_tagged_layout(
    Tp4GraphCommandRing* ring, bool trace, Tp4GraphCommandKind kind,
    std::uint32_t parameter, std::uint32_t q,
    std::uint32_t payload_bytes, std::uint32_t bytes_per_row,
    std::uint32_t maximum_q, std::uint64_t* sequence) noexcept;

bool tp4_graph_command_try_consume(Tp4GraphCommandRing* ring,
                                   std::uint64_t expected_sequence,
                                   Tp4GraphCommand* command) noexcept;

// Acquire-copy a published descriptor without acknowledging it. This permits
// an interleaved-family progress thread to select the strict family contract
// before consumption.
bool tp4_graph_command_try_peek(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommand* command) noexcept;

// Strict mixed-Q consumer. A released descriptor with invalid q/payload_bytes
// is not acknowledged and records expected_sequence as a fatal overflow.
bool tp4_graph_command_try_consume_descriptor(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommand* command) noexcept;

bool tp4_graph_allreduce_command_try_consume_descriptor(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommand* command) noexcept;

// Family-aware strict consumer. It validates q and q*bytes_per_row before
// acknowledging the command. A released invalid descriptor records overflow
// and remains unconsumed.
bool tp4_graph_command_try_consume_layout(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    std::uint32_t bytes_per_row, std::uint32_t maximum_q,
    Tp4GraphCommand* command) noexcept;

bool tp4_graph_command_try_consume_tagged_layout(
    Tp4GraphCommandRing* ring, std::uint64_t expected_sequence,
    Tp4GraphCommandKind expected_kind,
    std::uint32_t expected_parameter, std::uint32_t bytes_per_row,
    std::uint32_t maximum_q, Tp4GraphCommand* command) noexcept;

void tp4_graph_command_complete(Tp4GraphCommandRing* ring,
                                std::uint64_t sequence) noexcept;

std::uint64_t tp4_graph_command_published(
    const Tp4GraphCommandRing* ring) noexcept;

std::uint64_t tp4_graph_command_claimed(
    const Tp4GraphCommandRing* ring) noexcept;

std::uint64_t tp4_graph_command_consumed(
    const Tp4GraphCommandRing* ring) noexcept;

std::uint64_t tp4_graph_command_completed(
    const Tp4GraphCommandRing* ring) noexcept;

std::uint64_t tp4_graph_command_overflow(
    const Tp4GraphCommandRing* ring) noexcept;

}  // namespace spark_transport
