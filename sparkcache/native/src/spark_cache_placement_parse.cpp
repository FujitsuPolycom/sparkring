#include "spark_cache_placement.h"
#include "spark_cache_placement_layout.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

namespace {

using spark_cache::placement::validate_destinations;
using spark_cache::placement::validate_direct_slab;
using spark_cache::placement::validate_slots;

constexpr std::array<std::uint8_t, 8> kChunkMagic{
    'S', 'P', 'C', 'K', 'V', '0', '0', '1'};
constexpr std::uint32_t kFormatAbi = 1;
constexpr std::size_t kChunkPrefixBytes = 16;

void write_error(char* output, std::size_t capacity, std::string_view text) {
  if (output == nullptr || capacity == 0) {
    return;
  }
  const std::size_t copied = std::min(capacity - 1, text.size());
  std::memcpy(output, text.data(), copied);
  output[copied] = '\0';
}

std::uint32_t load_u32_le(const std::uint8_t* value) {
  return static_cast<std::uint32_t>(value[0]) |
         (static_cast<std::uint32_t>(value[1]) << 8U) |
         (static_cast<std::uint32_t>(value[2]) << 16U) |
         (static_cast<std::uint32_t>(value[3]) << 24U);
}

class CanonicalHeaderCursor {
 public:
  CanonicalHeaderCursor(const char* begin, const char* end)
      : current_(begin), end_(end) {}

  bool expect(std::string_view literal) {
    if (static_cast<std::size_t>(end_ - current_) < literal.size() ||
        std::memcmp(current_, literal.data(), literal.size()) != 0) {
      return false;
    }
    current_ += literal.size();
    return true;
  }

  bool parse_u32(std::uint32_t* output) {
    if (output == nullptr || current_ == end_ ||
        !std::isdigit(static_cast<unsigned char>(*current_))) {
      return false;
    }
    if (*current_ == '0' && current_ + 1 != end_ &&
        std::isdigit(static_cast<unsigned char>(current_[1]))) {
      return false;
    }
    std::uint64_t value = 0;
    while (current_ != end_ &&
           std::isdigit(static_cast<unsigned char>(*current_))) {
      value = value * 10 + static_cast<unsigned>(*current_ - '0');
      if (value > std::numeric_limits<std::uint32_t>::max()) {
        return false;
      }
      ++current_;
    }
    *output = static_cast<std::uint32_t>(value);
    return true;
  }

  bool parse_plain_string(std::string_view* output) {
    if (output == nullptr || !expect("\"")) {
      return false;
    }
    const char* begin = current_;
    while (current_ != end_ && *current_ != '"') {
      const unsigned char value = static_cast<unsigned char>(*current_);
      if (value < 0x20 || *current_ == '\\') {
        return false;
      }
      ++current_;
    }
    if (current_ == end_) {
      return false;
    }
    *output = std::string_view(
        begin, static_cast<std::size_t>(current_ - begin));
    ++current_;
    return true;
  }

  bool at_end() const noexcept { return current_ == end_; }

 private:
  const char* current_;
  const char* end_;
};

bool lowercase_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

struct ParsedRecord {
  bool is_positions = false;
  std::uint32_t data_kind = 0;
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
};

bool record_kind(
    std::string_view text,
    bool* is_positions,
    std::uint32_t* data_kind) {
  if (is_positions == nullptr || data_kind == nullptr) {
    return false;
  }
  *is_positions = false;
  if (text == "target_ckv") {
    *data_kind = SPARK_CACHE_RECORD_TARGET_CKV;
  } else if (text == "sparse_indexer") {
    *data_kind = SPARK_CACHE_RECORD_SPARSE_INDEXER;
  } else if (text == "mtp_draft_kv") {
    *data_kind = SPARK_CACHE_RECORD_MTP_DRAFT_KV;
  } else if (text == "boundary_hidden") {
    *data_kind = SPARK_CACHE_RECORD_BOUNDARY_HIDDEN;
  } else if (text == "logical_positions") {
    *is_positions = true;
    *data_kind = 0;
  } else {
    return false;
  }
  return true;
}

bool parse_canonical_header(
    const std::uint8_t* header,
    std::uint32_t header_bytes,
    std::uint32_t payload_bytes,
    std::uint32_t expected_logical_start,
    std::uint32_t dcp_degree,
    std::uint32_t dcp_rank,
    const std::uint8_t* payload,
    std::uint32_t required_data_record_mask,
    SparkCacheChunkDescriptor* output,
    std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  CanonicalHeaderCursor cursor(
      reinterpret_cast<const char*>(header),
      reinterpret_cast<const char*>(header) + header_bytes);
  std::uint32_t abi = 0;
  std::uint32_t logical_end = 0;
  std::uint32_t logical_start = 0;
  if (!cursor.expect("{\"format_abi\":") || !cursor.parse_u32(&abi) ||
      !cursor.expect(",\"logical_end\":") ||
      !cursor.parse_u32(&logical_end) ||
      !cursor.expect(",\"logical_start\":") ||
      !cursor.parse_u32(&logical_start) ||
      !cursor.expect(",\"records\":[")) {
    return fail("chunk header is not canonical SparkCache v1 JSON");
  }
  if (abi != kFormatAbi || logical_start != expected_logical_start ||
      logical_end <= logical_start || dcp_degree == 0 ||
      dcp_rank >= dcp_degree ||
      (logical_end - logical_start) % dcp_degree != 0) {
    return fail("chunk logical range or DCP identity is invalid");
  }

  std::uint32_t next_record_offset = 0;
  std::uint32_t data_mask = 0;
  bool positions_seen = false;
  std::uint32_t positions_offset = 0;
  std::uint32_t positions_length = 0;
  bool first = true;
  while (true) {
    if (!first && cursor.expect("]}")) {
      break;
    }
    if (first && cursor.expect("]}")) {
      return fail("chunk carries no records");
    }
    if (!first && !cursor.expect(",")) {
      return fail("malformed canonical record list");
    }
    first = false;
    if (!cursor.expect("{\"kind\":")) {
      return fail("record descriptor is not canonical");
    }
    std::string_view kind_text;
    std::string_view sha_text;
    std::uint32_t length = 0;
    std::uint32_t offset = 0;
    if (!cursor.parse_plain_string(&kind_text) ||
        !cursor.expect(",\"length\":") || !cursor.parse_u32(&length) ||
        !cursor.expect(",\"offset\":") || !cursor.parse_u32(&offset) ||
        !cursor.expect(",\"sha256\":") ||
        !cursor.parse_plain_string(&sha_text) || !cursor.expect("}")) {
      return fail("record descriptor fields are malformed");
    }
    bool is_positions = false;
    std::uint32_t kind = 0;
    if (!record_kind(kind_text, &is_positions, &kind) ||
        !lowercase_sha256(sha_text) || length == 0 ||
        offset != next_record_offset ||
        length > payload_bytes - std::min(payload_bytes, offset)) {
      return fail("record kind, checksum text, or byte span is invalid");
    }
    if (is_positions) {
      if (positions_seen) {
        return fail("logical_positions appears more than once");
      }
      positions_seen = true;
      positions_offset = offset;
      positions_length = length;
    } else {
      const std::uint32_t bit = 1U << kind;
      if ((data_mask & bit) != 0) {
        return fail("data record kind appears more than once");
      }
      data_mask |= bit;
      output->record_offset_bytes[kind] = offset;
      output->record_length_bytes[kind] = length;
    }
    next_record_offset += length;
  }
  if (!cursor.at_end() || next_record_offset != payload_bytes ||
      !positions_seen ||
      (data_mask & required_data_record_mask) != required_data_record_mask) {
    return fail("chunk record set is incomplete or has trailing bytes");
  }

  const std::uint32_t rows =
      (logical_end - logical_start) / dcp_degree;
  std::uint64_t expected_positions_bytes = 0;
  if (!spark_cache::placement::checked_mul(rows, 4, &expected_positions_bytes) ||
      positions_length != expected_positions_bytes) {
    return fail("logical_positions has the wrong byte length");
  }
  const std::uint8_t* positions = payload + positions_offset;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::uint64_t expected =
        static_cast<std::uint64_t>(logical_start) + dcp_rank +
        static_cast<std::uint64_t>(row) * dcp_degree;
    if (expected >= logical_end ||
        load_u32_le(positions + static_cast<std::size_t>(row) * 4) !=
            expected) {
      return fail("logical_positions disagrees with DCP ownership");
    }
  }
  output->row_count = rows;
  output->record_mask = data_mask;
  return true;
}

}  // namespace

extern "C" SparkCachePlacementStatus
spark_cache_parse_verified_v1_chunk(
    const void* arena_base,
    std::uint64_t arena_used_bytes,
    std::uint64_t arena_offset_bytes,
    std::uint32_t encoded_bytes,
    std::uint32_t expected_logical_start,
    std::uint32_t dcp_degree,
    std::uint32_t dcp_rank,
    std::uint32_t first_slot_index,
    std::uint32_t required_data_record_mask,
    SparkCacheChunkDescriptor* output,
    char* error,
    std::size_t error_capacity) {
  auto fail = [&](std::string_view message) {
    write_error(error, error_capacity, message);
    return SPARK_CACHE_PLACEMENT_FORMAT_ERROR;
  };
  if (arena_base == nullptr || output == nullptr ||
      encoded_bytes < kChunkPrefixBytes ||
      arena_offset_bytes > arena_used_bytes ||
      encoded_bytes > arena_used_bytes - arena_offset_bytes ||
      (required_data_record_mask &
       ~((1U << SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS) - 1U)) != 0) {
    return fail("invalid encoded chunk parser arguments");
  }
  const auto* encoded =
      static_cast<const std::uint8_t*>(arena_base) + arena_offset_bytes;
  if (!std::equal(kChunkMagic.begin(), kChunkMagic.end(), encoded) ||
      load_u32_le(encoded + 8) != kFormatAbi) {
    return fail("unsupported chunk magic or ABI");
  }
  const std::uint32_t header_bytes = load_u32_le(encoded + 12);
  if (header_bytes > encoded_bytes - kChunkPrefixBytes) {
    return fail("truncated chunk header");
  }
  const std::uint32_t payload_offset =
      static_cast<std::uint32_t>(kChunkPrefixBytes) + header_bytes;
  const std::uint32_t payload_bytes = encoded_bytes - payload_offset;

  SparkCacheChunkDescriptor parsed{};
  parsed.arena_offset_bytes = arena_offset_bytes;
  parsed.encoded_bytes = encoded_bytes;
  parsed.payload_offset_bytes = payload_offset;
  parsed.first_slot_index = first_slot_index;
  std::string detail;
  if (!parse_canonical_header(
          encoded + kChunkPrefixBytes,
          header_bytes,
          payload_bytes,
          expected_logical_start,
          dcp_degree,
          dcp_rank,
          encoded + payload_offset,
          required_data_record_mask,
          &parsed,
          &detail)) {
    return fail(detail);
  }
  *output = parsed;
  write_error(error, error_capacity, "");
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_reference_scatter_direct(
    const void* arena_base,
    std::uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    std::uint32_t chunk_count,
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count,
    const std::uint32_t* slots,
    std::uint32_t slot_count,
    char* error,
    std::size_t error_capacity) {
  auto fail = [&](SparkCachePlacementStatus status, std::string_view message) {
    write_error(error, error_capacity, message);
    return status;
  };
  std::string detail;
  if (arena_base == nullptr ||
      !validate_destinations(
          destinations, destination_count, destination_count, &detail) ||
      !validate_slots(
          slots,
          slot_count,
          slot_count,
          destinations,
          destination_count,
          &detail)) {
    return fail(SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT, detail);
  }
  std::uint32_t next = 0;
  if (!validate_direct_slab(
          arena_used_bytes,
          chunks,
          chunk_count,
          chunk_count,
          destinations,
          destination_count,
          slot_count,
          0,
          &next,
          &detail) ||
      next != slot_count) {
    if (detail.empty()) {
      detail = "direct reference chunks do not cover every slot row";
    }
    return fail(SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT, detail);
  }

  const auto* arena = static_cast<const std::uint8_t*>(arena_base);
  for (std::uint32_t chunk_index = 0; chunk_index < chunk_count;
       ++chunk_index) {
    const auto& chunk = chunks[chunk_index];
    const auto* payload =
        arena + chunk.arena_offset_bytes + chunk.payload_offset_bytes;
    for (std::uint32_t destination_index = 0;
         destination_index < destination_count;
         ++destination_index) {
      const auto& destination = destinations[destination_index];
      const std::size_t layer_bytes =
          static_cast<std::size_t>(chunk.row_count) *
          destination.bytes_per_token;
      const auto* source =
          payload + chunk.record_offset_bytes[destination.record_kind] +
          static_cast<std::size_t>(destination.source_layer_ordinal) *
              layer_bytes;
      auto* destination_base = reinterpret_cast<std::uint8_t*>(
          static_cast<std::uintptr_t>(destination.destination_base));
      for (std::uint32_t row = 0; row < chunk.row_count; ++row) {
        const std::uint32_t slot =
            slots[chunk.first_slot_index + row];
        std::memcpy(
            destination_base +
                static_cast<std::size_t>(slot) *
                    destination.destination_row_stride_bytes,
            source +
                static_cast<std::size_t>(row) *
                    destination.bytes_per_token,
            destination.bytes_per_token);
      }
    }
  }
  write_error(error, error_capacity, "");
  return SPARK_CACHE_PLACEMENT_OK;
}
