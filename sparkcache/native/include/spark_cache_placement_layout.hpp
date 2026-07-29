#ifndef SPARK_CACHE_PLACEMENT_LAYOUT_HPP_
#define SPARK_CACHE_PLACEMENT_LAYOUT_HPP_

#include "spark_cache_placement.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_set>

namespace spark_cache::placement {

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;
constexpr std::uint64_t kArena64MiB = 64ULL * kMiB;
constexpr std::uint64_t kArena128MiB = 128ULL * kMiB;
constexpr std::uint64_t kArena256MiB = 256ULL * kMiB;
constexpr std::uint32_t kMaximumDestinations = 1024;
constexpr std::uint32_t kMaximumSlots = 1U << 24;
constexpr std::uint32_t kMaximumChunksPerSlab = 4096;

static_assert(sizeof(SparkCacheDestinationDescriptor) == 32);
static_assert(alignof(SparkCacheDestinationDescriptor) == 8);
static_assert(sizeof(SparkCacheChunkDescriptor) == 64);
static_assert(alignof(SparkCacheChunkDescriptor) == 8);
static_assert(sizeof(SparkCacheTransposedSource) == 16);
static_assert(alignof(SparkCacheTransposedSource) == 8);
static_assert(sizeof(SparkCachePlacementConfig) == 48);
static_assert(sizeof(SparkCachePlacementStats) == 56);
static_assert(sizeof(SparkCachePlacementAbiInfo) == 64);
static_assert(sizeof(SparkCacheArenaView) == 40);

inline bool checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* output) noexcept {
  if (output == nullptr ||
      left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  *output = left + right;
  return true;
}

inline bool checked_mul(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* output) noexcept {
  if (output == nullptr ||
      (left != 0 &&
       right > std::numeric_limits<std::uint64_t>::max() / left)) {
    return false;
  }
  *output = left * right;
  return true;
}

inline bool valid_arena_bytes(std::uint64_t bytes) noexcept {
  // 64 MiB is useful for the direct encoded-chunk A/B. The product lane
  // explicitly requires and tests both 128 and 256 MiB.
  return bytes == kArena64MiB || bytes == kArena128MiB ||
         bytes == kArena256MiB;
}

inline bool validate_config(
    const SparkCachePlacementConfig& config,
    std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (config.abi_version != SPARK_CACHE_PLACEMENT_ABI_VERSION) {
    return fail("unsupported placement ABI");
  }
  if (config.arena_mode < SPARK_CACHE_ARENA_MAPPED_HOST ||
      config.arena_mode > SPARK_CACHE_ARENA_STAGED_DEVICE) {
    return fail("unsupported arena mode");
  }
  if (!valid_arena_bytes(config.arena_bytes)) {
    return fail("arena must be exactly 64, 128, or 256 MiB");
  }
  if (config.max_destinations == 0 ||
      config.max_destinations > kMaximumDestinations) {
    return fail("max_destinations is outside the validated range");
  }
  if (config.max_slots == 0 || config.max_slots > kMaximumSlots) {
    return fail("max_slots is outside the validated range");
  }
  if (config.max_chunks_per_slab == 0 ||
      config.max_chunks_per_slab > kMaximumChunksPerSlab) {
    return fail("max_chunks_per_slab is outside the validated range");
  }
  if (config.device_ordinal < 0) {
    return fail("device_ordinal must be nonnegative");
  }
  if ((config.flags & ~SPARK_CACHE_CONFIG_PREFETCH_MANAGED) != 0) {
    return fail("unknown placement config flags");
  }
  if ((config.flags & SPARK_CACHE_CONFIG_PREFETCH_MANAGED) != 0 &&
      config.arena_mode != SPARK_CACHE_ARENA_MANAGED) {
    return fail("managed prefetch flag requires a managed arena");
  }
  if (std::any_of(
          std::begin(config.reserved),
          std::end(config.reserved),
          [](std::uint32_t value) { return value != 0; })) {
    return fail("reserved placement config fields must be zero");
  }
  return true;
}

inline bool validate_destinations(
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count,
    std::uint32_t maximum_destinations,
    std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (destinations == nullptr || destination_count == 0 ||
      destination_count > maximum_destinations) {
    return fail("invalid destination table length");
  }
  for (std::uint32_t index = 0; index < destination_count; ++index) {
    const auto& descriptor = destinations[index];
    if (descriptor.destination_base == 0 ||
        descriptor.destination_rows == 0) {
      return fail("destination base and row capacity must be nonzero");
    }
    if (descriptor.bytes_per_token == 0 ||
        descriptor.destination_row_stride_bytes <
            descriptor.bytes_per_token) {
      return fail("destination row width/stride is invalid");
    }
    if (descriptor.record_kind >=
        SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS) {
      return fail("destination record kind is invalid");
    }
  }
  return true;
}

inline bool validate_slots(
    const std::uint32_t* slots,
    std::uint32_t slot_count,
    std::uint32_t maximum_slots,
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count,
    std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (slots == nullptr || slot_count == 0 || slot_count > maximum_slots) {
    return fail("invalid slot vector length");
  }
  std::uint64_t capacity = std::numeric_limits<std::uint64_t>::max();
  for (std::uint32_t index = 0; index < destination_count; ++index) {
    capacity = std::min(capacity, destinations[index].destination_rows);
  }
  std::unordered_set<std::uint32_t> unique;
  unique.reserve(slot_count);
  for (std::uint32_t index = 0; index < slot_count; ++index) {
    if (slots[index] >= capacity) {
      return fail("physical slot exceeds a destination row capacity");
    }
    if (!unique.insert(slots[index]).second) {
      return fail("physical slot vector contains a duplicate");
    }
  }
  return true;
}

inline bool validate_direct_slab(
    std::uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    std::uint32_t chunk_count,
    std::uint32_t maximum_chunks,
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count,
    std::uint32_t slot_count,
    std::uint32_t expected_first_slot,
    std::uint32_t* next_first_slot,
    std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (chunks == nullptr || chunk_count == 0 ||
      chunk_count > maximum_chunks) {
    return fail("invalid direct slab chunk count");
  }
  std::uint64_t expected_slot = expected_first_slot;
  for (std::uint32_t chunk_index = 0; chunk_index < chunk_count;
       ++chunk_index) {
    const auto& chunk = chunks[chunk_index];
    if (chunk.row_count == 0 || chunk.first_slot_index != expected_slot) {
      return fail("direct chunks must cover contiguous slot-vector rows");
    }
    expected_slot += chunk.row_count;
    if (expected_slot > slot_count) {
      return fail("direct chunk rows exceed the slot vector");
    }
    std::uint64_t encoded_end = 0;
    if (chunk.encoded_bytes == 0 ||
        !checked_add(
            chunk.arena_offset_bytes, chunk.encoded_bytes, &encoded_end) ||
        encoded_end > arena_used_bytes ||
        chunk.payload_offset_bytes > chunk.encoded_bytes) {
      return fail("direct chunk encoded span exceeds the arena");
    }
    if (chunk.flags != 0) {
      return fail("unknown direct chunk flags");
    }
    const std::uint64_t payload_bytes =
        chunk.encoded_bytes - chunk.payload_offset_bytes;
    for (std::uint32_t kind = 0;
         kind < SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS;
         ++kind) {
      if ((chunk.record_mask & (1U << kind)) == 0) {
        if (chunk.record_offset_bytes[kind] != 0 ||
            chunk.record_length_bytes[kind] != 0) {
          return fail("absent record kind carries a nonzero span");
        }
        continue;
      }
      std::uint64_t record_end = 0;
      if (!checked_add(
              chunk.record_offset_bytes[kind],
              chunk.record_length_bytes[kind],
              &record_end) ||
          record_end > payload_bytes) {
        return fail("record span exceeds encoded chunk payload");
      }
    }
    for (std::uint32_t destination_index = 0;
         destination_index < destination_count;
         ++destination_index) {
      const auto& destination = destinations[destination_index];
      const std::uint32_t kind = destination.record_kind;
      if ((chunk.record_mask & (1U << kind)) == 0) {
        return fail("chunk is missing a destination record kind");
      }
      std::uint64_t layer_rows_bytes = 0;
      std::uint64_t layer_end = 0;
      if (!checked_mul(
              chunk.row_count,
              destination.bytes_per_token,
              &layer_rows_bytes) ||
          !checked_mul(
              static_cast<std::uint64_t>(
                  destination.source_layer_ordinal) +
                  1,
              layer_rows_bytes,
              &layer_end) ||
          layer_end > chunk.record_length_bytes[kind]) {
        return fail("destination layer exceeds its chunk record span");
      }
    }
  }
  if (next_first_slot != nullptr) {
    *next_first_slot = static_cast<std::uint32_t>(expected_slot);
  }
  return true;
}

inline bool validate_transposed_slab(
    std::uint64_t arena_used_bytes,
    const SparkCacheTransposedSource* sources,
    std::uint32_t source_count,
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count,
    std::uint32_t slot_count,
    std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (sources == nullptr || source_count == 0 ||
      source_count > destination_count) {
    return fail("invalid transposed source count");
  }
  std::unordered_set<std::uint32_t> unique;
  for (std::uint32_t index = 0; index < source_count; ++index) {
    const auto& source = sources[index];
    if (source.flags != 0 ||
        source.destination_index >= destination_count ||
        !unique.insert(source.destination_index).second) {
      return fail("invalid or duplicate transposed destination index");
    }
    const auto& destination = destinations[source.destination_index];
    std::uint64_t source_bytes = 0;
    std::uint64_t source_end = 0;
    if (!checked_mul(
            slot_count, destination.bytes_per_token, &source_bytes) ||
        !checked_add(
            source.source_offset_bytes, source_bytes, &source_end) ||
        source_end > arena_used_bytes) {
      return fail("transposed source span exceeds the arena");
    }
  }
  return true;
}

}  // namespace spark_cache::placement

#endif  // SPARK_CACHE_PLACEMENT_LAYOUT_HPP_
