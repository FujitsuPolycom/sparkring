#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "spark_transport/tp4_bidirectional_prefill.hpp"

namespace spark_transport::tiled_prefill_research {

constexpr std::uint32_t kBidirectionalBulkTilesPerShard = 4;
constexpr std::uint32_t kBidirectionalBulkElementsPerRow = 4096;
constexpr std::uint32_t kBidirectionalBulkDefaultQueryRows = 2048;
constexpr std::size_t kBidirectionalBulkCtaBytes = 64U * 1024U;

constexpr bool bidirectional_bulk_query_rows_supported(
    std::uint32_t query_rows) noexcept {
  return query_rows == 1024 || query_rows == 2048 || query_rows == 4096 ||
         query_rows == 8192;
}

constexpr std::uint64_t bidirectional_bulk_payload_bytes(
    std::uint32_t query_rows) {
  if (!bidirectional_bulk_query_rows_supported(query_rows)) {
    throw std::out_of_range(
        "bidirectional bulk query rows must be 1024, 2048, 4096, or 8192");
  }
  return static_cast<std::uint64_t>(query_rows) *
         kBidirectionalBulkElementsPerRow * kTp4PrefillBf16Bytes;
}

constexpr std::uint64_t bidirectional_bulk_half_bytes(
    std::uint32_t query_rows) {
  return bidirectional_bulk_payload_bytes(query_rows) / 2U;
}

constexpr std::uint64_t bidirectional_bulk_shard_bytes(
    std::uint32_t query_rows) {
  return bidirectional_bulk_half_bytes(query_rows) / kTp4PrefillRankCount;
}

constexpr std::uint32_t bidirectional_bulk_tile_bytes(
    std::uint32_t query_rows) {
  return static_cast<std::uint32_t>(
      bidirectional_bulk_shard_bytes(query_rows) /
      kBidirectionalBulkTilesPerShard);
}

// Stable host/device descriptor for one of four tiles in a width-4096
// bidirectional reduce-scatter/all-gather half-shard.
struct alignas(16) BidirectionalBulkDescriptor {
  std::uint64_t tensor_offset_bytes{};
  std::uint64_t send_offset_bytes{};
  std::uint64_t receive_offset_bytes{};
  // Required for every post-exchange action. The incoming endpoint's ordered
  // payload write precedes this token write on the same reliable QP.
  std::uint64_t inbound_doorbell_offset_bytes{};
  std::uint64_t expected_doorbell_token{};
  std::uint64_t secondary_inbound_doorbell_offset_bytes{};
  std::uint64_t secondary_expected_doorbell_token{};
  std::uint32_t active_bytes{};
  std::uint32_t rank{};
  std::uint32_t stage{};
  std::uint32_t shard{};
  std::uint32_t tile_in_shard{};
  std::uint32_t half{};
  std::int32_t direction{};
  std::uint32_t query_rows{};
  std::uint32_t elements_per_row{};
};

static_assert(kTp4PrefillShardBytes == 2U * 1024U * 1024U);
static_assert(kBidirectionalBulkTilesPerShard == 4);
static_assert(bidirectional_bulk_tile_bytes(1024) == 256U * 1024U);
static_assert(bidirectional_bulk_tile_bytes(2048) == 512U * 1024U);
static_assert(bidirectional_bulk_tile_bytes(4096) == 1024U * 1024U);
static_assert(bidirectional_bulk_tile_bytes(8192) == 2U * 1024U * 1024U);
static_assert(sizeof(BidirectionalBulkDescriptor) == 96);
static_assert(std::is_trivially_copyable_v<BidirectionalBulkDescriptor>);

}  // namespace spark_transport::tiled_prefill_research
