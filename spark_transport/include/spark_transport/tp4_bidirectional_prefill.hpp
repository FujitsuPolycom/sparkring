#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace spark_transport {

// CPU-only contract for the TP4 bidirectional prefill ring.  This header
// deliberately describes topology and ownership only; it does not select a
// production transport implementation.
constexpr std::uint32_t kTp4PrefillRankCount = 4;
constexpr std::uint32_t kTp4PrefillReduceScatterStages = 3;
constexpr std::uint32_t kTp4PrefillAllGatherStages = 3;
constexpr std::uint32_t kTp4PrefillStageCount =
    kTp4PrefillReduceScatterStages + kTp4PrefillAllGatherStages;
constexpr std::uint32_t kTp4PrefillQ2048Rows = 2048;
constexpr std::uint32_t kTp4PrefillWidth4096 = 4096;
constexpr std::size_t kTp4PrefillBf16Bytes = 2;
constexpr std::size_t kTp4PrefillPayloadBytes =
    static_cast<std::size_t>(kTp4PrefillQ2048Rows) *
    kTp4PrefillWidth4096 * kTp4PrefillBf16Bytes;
constexpr std::size_t kTp4PrefillHalfBytes =
    kTp4PrefillPayloadBytes / 2U;
constexpr std::size_t kTp4PrefillShardBytes =
    kTp4PrefillHalfBytes / kTp4PrefillRankCount;
constexpr std::size_t kTp4PrefillBytesPerDirection =
    kTp4PrefillStageCount * kTp4PrefillShardBytes;
constexpr std::size_t kTp4PrefillBytesPerRank =
    2U * kTp4PrefillBytesPerDirection;

enum class Tp4PrefillDirection : std::int8_t {
  kClockwise = 1,
  kCounterClockwise = -1,
};

enum class Tp4PrefillEndpoint : std::uint8_t {
  kXor1 = 0,
  kXor3 = 1,
};

enum class Tp4PrefillHalf : std::uint8_t {
  kLower = 0,
  kUpper = 1,
};

enum class Tp4PrefillPhase : std::uint8_t {
  kReduceScatter = 0,
  kAllGather = 1,
};

constexpr bool tp4_prefill_direction_valid(
    Tp4PrefillDirection direction) noexcept {
  return direction == Tp4PrefillDirection::kClockwise ||
         direction == Tp4PrefillDirection::kCounterClockwise;
}

constexpr std::int32_t tp4_prefill_direction_step(
    Tp4PrefillDirection direction) {
  if (!tp4_prefill_direction_valid(direction)) {
    throw std::invalid_argument("invalid TP4 prefill direction");
  }
  return static_cast<std::int32_t>(direction);
}

constexpr std::uint32_t tp4_prefill_wrap_rank(std::int32_t rank) noexcept {
  constexpr std::int32_t count =
      static_cast<std::int32_t>(kTp4PrefillRankCount);
  const std::int32_t remainder = rank % count;
  return static_cast<std::uint32_t>(
      remainder < 0 ? remainder + count : remainder);
}

constexpr std::uint32_t tp4_prefill_successor(
    std::uint32_t rank, Tp4PrefillDirection direction) {
  if (rank >= kTp4PrefillRankCount) {
    throw std::out_of_range("TP4 prefill rank must be in [0, 3]");
  }
  return tp4_prefill_wrap_rank(
      static_cast<std::int32_t>(rank) +
      tp4_prefill_direction_step(direction));
}

constexpr Tp4PrefillEndpoint tp4_prefill_outgoing_endpoint(
    std::uint32_t rank, Tp4PrefillDirection direction) {
  const std::uint32_t peer = tp4_prefill_successor(rank, direction);
  if (peer == (rank ^ 1U)) {
    return Tp4PrefillEndpoint::kXor1;
  }
  if (peer == (rank ^ 3U)) {
    return Tp4PrefillEndpoint::kXor3;
  }
  throw std::logic_error("TP4 prefill successor is not a direct neighbor");
}

constexpr Tp4PrefillEndpoint tp4_prefill_incoming_endpoint(
    std::uint32_t rank, Tp4PrefillDirection direction) {
  const auto opposite =
      direction == Tp4PrefillDirection::kClockwise
          ? Tp4PrefillDirection::kCounterClockwise
          : Tp4PrefillDirection::kClockwise;
  return tp4_prefill_outgoing_endpoint(rank, opposite);
}

constexpr Tp4PrefillHalf tp4_prefill_half(
    Tp4PrefillDirection direction) {
  if (!tp4_prefill_direction_valid(direction)) {
    throw std::invalid_argument("invalid TP4 prefill direction");
  }
  return direction == Tp4PrefillDirection::kClockwise
             ? Tp4PrefillHalf::kLower
             : Tp4PrefillHalf::kUpper;
}

struct Tp4PrefillStage {
  std::uint32_t rank{};
  Tp4PrefillDirection direction{};
  Tp4PrefillHalf half{};
  Tp4PrefillPhase phase{};
  std::uint32_t phase_stage{};
  Tp4PrefillEndpoint outgoing_endpoint{};
  Tp4PrefillEndpoint incoming_endpoint{};
  std::uint32_t outgoing_peer{};
  std::uint32_t incoming_peer{};
  std::uint32_t send_shard{};
  std::uint32_t receive_shard{};
  std::size_t bytes{kTp4PrefillShardBytes};
};

constexpr Tp4PrefillStage tp4_prefill_stage(
    std::uint32_t rank, Tp4PrefillDirection direction,
    std::uint32_t stage) {
  if (rank >= kTp4PrefillRankCount) {
    throw std::out_of_range("TP4 prefill rank must be in [0, 3]");
  }
  if (!tp4_prefill_direction_valid(direction)) {
    throw std::invalid_argument("invalid TP4 prefill direction");
  }
  if (stage >= kTp4PrefillStageCount) {
    throw std::out_of_range("TP4 prefill stage must be in [0, 5]");
  }

  const std::int32_t rank_value = static_cast<std::int32_t>(rank);
  const std::int32_t step = tp4_prefill_direction_step(direction);
  const bool reduce_scatter = stage < kTp4PrefillReduceScatterStages;
  const std::uint32_t phase_stage =
      reduce_scatter ? stage : stage - kTp4PrefillReduceScatterStages;
  const std::int32_t phase_step =
      static_cast<std::int32_t>(phase_stage);
  const std::uint32_t send_shard = tp4_prefill_wrap_rank(
      reduce_scatter ? rank_value - step * phase_step
                     : rank_value + step - step * phase_step);
  const std::uint32_t receive_shard = tp4_prefill_wrap_rank(
      reduce_scatter ? rank_value - step * (phase_step + 1)
                     : rank_value - step * phase_step);
  const auto outgoing_endpoint =
      tp4_prefill_outgoing_endpoint(rank, direction);
  const auto incoming_endpoint =
      tp4_prefill_incoming_endpoint(rank, direction);

  return {rank,
          direction,
          tp4_prefill_half(direction),
          reduce_scatter ? Tp4PrefillPhase::kReduceScatter
                         : Tp4PrefillPhase::kAllGather,
          phase_stage,
          outgoing_endpoint,
          incoming_endpoint,
          tp4_prefill_successor(rank, direction),
          tp4_prefill_successor(
              rank, direction == Tp4PrefillDirection::kClockwise
                        ? Tp4PrefillDirection::kCounterClockwise
                        : Tp4PrefillDirection::kClockwise),
          send_shard,
          receive_shard,
          kTp4PrefillShardBytes};
}

using Tp4PrefillContributorMask = std::uint8_t;
using Tp4PrefillShardContributors =
    std::array<Tp4PrefillContributorMask, kTp4PrefillRankCount>;
using Tp4PrefillRankContributors =
    std::array<Tp4PrefillShardContributors, kTp4PrefillRankCount>;
using Tp4PrefillDirectionContributors =
    std::array<Tp4PrefillRankContributors, 2>;

struct Tp4BidirectionalPrefillVerification {
  Tp4PrefillDirectionContributors contributors{};
  std::array<std::array<std::size_t, kTp4PrefillRankCount>, 2>
      transmitted_bytes{};

  constexpr bool complete() const noexcept {
    constexpr Tp4PrefillContributorMask all_ranks =
        static_cast<Tp4PrefillContributorMask>(
            (1U << kTp4PrefillRankCount) - 1U);
    for (const auto& direction : contributors) {
      for (const auto& rank : direction) {
        for (const auto shard : rank) {
          if (shard != all_ranks) {
            return false;
          }
        }
      }
    }
    for (const auto& direction : transmitted_bytes) {
      for (const auto bytes : direction) {
        if (bytes != kTp4PrefillBytesPerDirection) {
          return false;
        }
      }
    }
    return true;
  }
};

constexpr std::size_t tp4_prefill_direction_index(
    Tp4PrefillDirection direction) {
  if (!tp4_prefill_direction_valid(direction)) {
    throw std::invalid_argument("invalid TP4 prefill direction");
  }
  return direction == Tp4PrefillDirection::kClockwise ? 0U : 1U;
}

constexpr Tp4BidirectionalPrefillVerification
verify_tp4_bidirectional_prefill() noexcept {
  Tp4BidirectionalPrefillVerification result{};
  constexpr std::array<Tp4PrefillDirection, 2> directions{
      Tp4PrefillDirection::kClockwise,
      Tp4PrefillDirection::kCounterClockwise};

  for (std::size_t direction_index = 0;
       direction_index < directions.size(); ++direction_index) {
    for (std::uint32_t rank = 0; rank < kTp4PrefillRankCount; ++rank) {
      const auto local = static_cast<Tp4PrefillContributorMask>(1U << rank);
      for (std::uint32_t shard = 0; shard < kTp4PrefillRankCount; ++shard) {
        result.contributors[direction_index][rank][shard] = local;
      }
    }

    for (std::uint32_t stage = 0; stage < kTp4PrefillStageCount; ++stage) {
      const auto before = result.contributors[direction_index];
      for (std::uint32_t rank = 0; rank < kTp4PrefillRankCount; ++rank) {
        const auto transfer = tp4_prefill_stage(rank, directions[direction_index],
                                                stage);
        const std::uint32_t receiver = transfer.outgoing_peer;
        if (transfer.phase == Tp4PrefillPhase::kReduceScatter) {
          result.contributors[direction_index][receiver]
                             [transfer.send_shard] =
              static_cast<Tp4PrefillContributorMask>(
                  before[receiver][transfer.send_shard] |
                  before[rank][transfer.send_shard]);
        } else {
          result.contributors[direction_index][receiver]
                             [transfer.send_shard] =
              before[rank][transfer.send_shard];
        }
        result.transmitted_bytes[direction_index][rank] +=
            transfer.bytes;
      }
    }
  }
  return result;
}

static_assert(kTp4PrefillPayloadBytes == 16U * 1024U * 1024U);
static_assert(kTp4PrefillHalfBytes == 8U * 1024U * 1024U);
static_assert(kTp4PrefillShardBytes == 2U * 1024U * 1024U);
static_assert(kTp4PrefillBytesPerDirection == 12U * 1024U * 1024U);
static_assert(kTp4PrefillBytesPerRank == 24U * 1024U * 1024U);
static_assert(verify_tp4_bidirectional_prefill().complete());

}  // namespace spark_transport
