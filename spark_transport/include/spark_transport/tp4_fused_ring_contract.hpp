#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace spark_transport {

constexpr std::uint32_t kFusedRingRanks = 4;
constexpr std::uint32_t kFusedRingDirections = 2;
constexpr std::uint32_t kFusedRingTilesPerShard = 4;
constexpr std::uint32_t kFusedRingFlows = 8;
constexpr std::uint32_t kFusedRingStages = 6;
constexpr std::uint32_t kFusedRingSlots = 8;
constexpr std::uint64_t kFusedRingPayloadBytes = 8192ULL * 4096ULL * 2ULL;
constexpr std::uint64_t kFusedRingHalfBytes = kFusedRingPayloadBytes / 2U;
constexpr std::uint64_t kFusedRingShardBytes = kFusedRingPayloadBytes / 8U;
constexpr std::uint64_t kFusedRingTileBytes =
    kFusedRingShardBytes / kFusedRingTilesPerShard;
constexpr std::uint64_t kFusedRingRailBytes = kFusedRingTileBytes / 2U;
constexpr std::uint64_t kFusedRingBytesPerEndpoint =
    kFusedRingStages * kFusedRingShardBytes;
constexpr std::uint64_t kFusedRingBytesPerRail =
    kFusedRingBytesPerEndpoint / 2U;
constexpr std::uint64_t kFusedRingBytesPerRank =
    2U * kFusedRingBytesPerEndpoint;

struct FusedRingFlow {
  std::uint32_t direction{};
  std::uint32_t tile{};
};

constexpr FusedRingFlow fused_ring_flow(std::uint32_t flow) {
  if (flow >= kFusedRingFlows) throw std::out_of_range("fused flow");
  return {flow / kFusedRingTilesPerShard,
          flow % kFusedRingTilesPerShard};
}

constexpr std::uint64_t fused_ring_ordinal(std::uint64_t operation_base,
                                           std::uint32_t stage,
                                           std::uint32_t tile) {
  if (stage >= kFusedRingStages || tile >= kFusedRingTilesPerShard)
    throw std::out_of_range("fused stage/tile");
  return operation_base + stage * kFusedRingTilesPerShard + tile;
}

constexpr std::uint32_t fused_ring_slot(std::uint64_t ordinal) noexcept {
  return static_cast<std::uint32_t>(ordinal % kFusedRingSlots);
}

constexpr std::uint64_t fused_ring_token(std::uint64_t ordinal) {
  if (ordinal == UINT64_MAX) throw std::overflow_error("fused token");
  return ordinal + 1U;
}

constexpr bool fused_ring_stage_reuses_slot(std::uint32_t stage) noexcept {
  return stage >= 2U;
}

constexpr std::uint64_t fused_ring_reused_ordinal(
    std::uint64_t operation_base, std::uint32_t stage, std::uint32_t tile) {
  if (!fused_ring_stage_reuses_slot(stage))
    throw std::out_of_range("fused stage has no reuse gate");
  return fused_ring_ordinal(operation_base, stage - 2U, tile);
}

constexpr bool fused_ring_reuse_ready(std::uint64_t acknowledgement,
                                      std::uint64_t observed,
                                      std::uint64_t prior_ordinal) noexcept {
  const std::uint64_t expected = prior_ordinal + 1U;
  return acknowledgement >= expected && observed == expected;
}

enum class FusedRingControlWord : std::uint8_t {
  kProducer,
  kPrimaryDoorbellSource,
  kPrimaryDoorbellArrival,
  kSecondaryDoorbellSource,
  kSecondaryDoorbellArrival,
  kConsumer,
  kAcknowledgement,
  kBothRailCqObserved,
};

static_assert(kFusedRingTileBytes == 2U * 1024U * 1024U);
static_assert(kFusedRingRailBytes == 1024U * 1024U);
static_assert(kFusedRingBytesPerEndpoint == 48U * 1024U * 1024U);
static_assert(kFusedRingBytesPerRail == 24U * 1024U * 1024U);
static_assert(kFusedRingBytesPerRank == 96U * 1024U * 1024U);
static_assert(fused_ring_slot(fused_ring_ordinal(0, 0, 0)) == 0);
static_assert(fused_ring_slot(fused_ring_ordinal(0, 1, 0)) == 4);
static_assert(fused_ring_slot(fused_ring_ordinal(0, 2, 0)) == 0);

}  // namespace spark_transport
