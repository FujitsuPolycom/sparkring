#include "spark_cache_snapshot_ring.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace spark_cache::snapshot {
namespace {

constexpr std::uint64_t kPayloadAlignment = 64;

bool fail(std::string* detail, const char* message) {
  if (detail != nullptr) {
    *detail = message;
  }
  return false;
}

bool checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* output) {
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *output = left * right;
  return true;
}

bool checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t* output) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

bool align_up(
    std::uint64_t value,
    std::uint64_t alignment,
    std::uint64_t* output) {
  const auto remainder = value % alignment;
  return remainder == 0 ||
         checked_add(value, alignment - remainder, output);
}

}  // namespace

bool validate_config(
    const SparkCacheSnapshotConfig& config,
    std::string* detail) {
  if (config.abi_version != SPARK_CACHE_SNAPSHOT_ABI_VERSION) {
    return fail(detail, "snapshot ABI version mismatch");
  }
  if (config.arena_mode != SPARK_CACHE_SNAPSHOT_MAPPED_HOST &&
      config.arena_mode != SPARK_CACHE_SNAPSHOT_MANAGED) {
    return fail(detail, "unsupported snapshot arena mode");
  }
  if (config.slot_count < SPARK_CACHE_SNAPSHOT_MIN_SLOTS ||
      config.slot_count > SPARK_CACHE_SNAPSHOT_MAX_SLOTS) {
    return fail(detail, "snapshot slot_count must be two or three");
  }
  if (config.slot_bytes == 0 || config.max_sources == 0 ||
      config.max_rows == 0) {
    return fail(detail, "snapshot capacity values must be nonzero");
  }
  if (config.flags != 0) {
    return fail(detail, "snapshot config flags are not implemented");
  }
  if (config.slot_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return fail(detail, "snapshot slot_bytes exceeds address space");
  }
  for (const auto reserved : config.reserved) {
    if (reserved != 0) {
      return fail(detail, "snapshot config reserved fields must be zero");
    }
  }
  if (detail != nullptr) {
    detail->clear();
  }
  return true;
}

bool validate_sources(
    const SparkCacheSnapshotSource* sources,
    std::uint32_t source_count,
    std::uint32_t max_sources,
    std::string* detail) {
  if (sources == nullptr || source_count == 0 ||
      source_count > max_sources) {
    return fail(detail, "invalid snapshot source table");
  }
  std::array<std::uint32_t, SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS>
      next_ordinal{};
  std::array<std::uint32_t, SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS>
      width{};
  std::array<bool, SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS> seen{};

  for (std::uint32_t index = 0; index < source_count; ++index) {
    const auto& source = sources[index];
    if (source.source_base == 0 || source.source_rows == 0 ||
        source.bytes_per_token == 0 ||
        source.source_row_stride_bytes < source.bytes_per_token ||
        source.record_kind >= SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS) {
      return fail(detail, "invalid snapshot source descriptor");
    }
    const auto kind = source.record_kind;
    if (source.source_layer_ordinal != next_ordinal[kind]) {
      return fail(
          detail,
          "snapshot source layer ordinals must be dense and ordered");
    }
    next_ordinal[kind] += 1;
    if (!seen[kind]) {
      seen[kind] = true;
      width[kind] = source.bytes_per_token;
    } else if (width[kind] != source.bytes_per_token) {
      return fail(
          detail,
          "snapshot sources of one record kind must share a row width");
    }
  }
  if (detail != nullptr) {
    detail->clear();
  }
  return true;
}

bool calculate_payload_layout(
    const SparkCacheSnapshotSource* sources,
    std::uint32_t source_count,
    std::uint32_t row_count,
    std::uint64_t slot_bytes,
    SparkCacheSnapshotReadyView* output,
    std::string* detail) {
  if (output == nullptr || row_count == 0 || slot_bytes == 0) {
    return fail(detail, "invalid snapshot payload request");
  }
  if (!validate_sources(sources, source_count, source_count, detail)) {
    return false;
  }

  SparkCacheSnapshotReadyView view{};
  view.capacity_bytes = slot_bytes;
  view.row_count = row_count;
  std::array<std::uint32_t, SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS>
      layer_count{};
  std::array<std::uint32_t, SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS>
      width{};
  for (std::uint32_t index = 0; index < source_count; ++index) {
    const auto& source = sources[index];
    if (source.source_rows < row_count) {
      return fail(detail, "snapshot source has fewer rows than submission");
    }
    layer_count[source.record_kind] += 1;
    width[source.record_kind] = source.bytes_per_token;
  }

  std::uint64_t cursor = 0;
  for (std::uint32_t kind = 0;
       kind < SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS;
       ++kind) {
    if (layer_count[kind] == 0) {
      continue;
    }
    std::uint64_t aligned = cursor;
    if (!align_up(cursor, kPayloadAlignment, &aligned)) {
      return fail(detail, "snapshot payload alignment overflow");
    }
    std::uint64_t row_bytes = 0;
    std::uint64_t record_bytes = 0;
    if (!checked_multiply(row_count, width[kind], &row_bytes) ||
        !checked_multiply(
            row_bytes, layer_count[kind], &record_bytes) ||
        record_bytes > std::numeric_limits<std::uint32_t>::max() ||
        aligned > std::numeric_limits<std::uint32_t>::max()) {
      return fail(detail, "snapshot record exceeds ABI range");
    }
    std::uint64_t end = 0;
    if (!checked_add(aligned, record_bytes, &end) ||
        end > slot_bytes) {
      return fail(detail, "snapshot payload exceeds arena slot");
    }
    view.record_mask |= 1U << kind;
    view.record_offset_bytes[kind] =
        static_cast<std::uint32_t>(aligned);
    view.record_length_bytes[kind] =
        static_cast<std::uint32_t>(record_bytes);
    cursor = end;
  }
  view.used_bytes = cursor;
  *output = view;
  if (detail != nullptr) {
    detail->clear();
  }
  return true;
}

}  // namespace spark_cache::snapshot
