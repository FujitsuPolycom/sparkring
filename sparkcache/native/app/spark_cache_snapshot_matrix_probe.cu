#include "spark_cache_snapshot.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kSourceRows = 2048;
constexpr std::uint64_t kMebibyte = 1024ULL * 1024ULL;
constexpr auto kClaimTimeout = std::chrono::seconds(30);
constexpr std::size_t kMaximumMismatchSamples = 16;

enum class FixtureProfile {
  kCompact,
  kGlm52,
};

struct Options {
  std::uint32_t arena_mode = SPARK_CACHE_SNAPSHOT_MAPPED_HOST;
  const char* arena_name = "mapped";
  std::uint32_t slot_count = 2;
  std::uint32_t rank = 0;
  std::uint32_t row_count = 64;
  std::uint32_t iterations = 1;
  std::uint32_t compare_every = 1;
  std::uint32_t pipeline_depth = 1;
  std::uint32_t writer_hold_us = 0;
  std::uint32_t saturation_cycles = 100;
  std::uint32_t overlap_samples = 100;
  FixtureProfile profile = FixtureProfile::kCompact;
  const char* profile_name = "compact";
  std::uint64_t slot_bytes = 2ULL * kMebibyte;
};

struct Fixture {
  std::string name;
  std::uint32_t kind;
  std::uint32_t ordinal;
  std::uint32_t stride;
  std::uint32_t width;
  std::vector<std::uint8_t> host;
  std::uint8_t* device = nullptr;
};

struct Mismatch {
  std::string source;
  std::uint32_t kind;
  std::uint32_t layer;
  std::uint32_t output_row;
  std::uint32_t physical_row;
  std::uint32_t byte;
  std::uint64_t payload_offset;
  std::uint8_t expected;
  std::uint8_t actual;
};

struct Result {
  bool passed = false;
  std::string error;
  std::vector<double> submit_us;
  std::vector<double> gather_us;
  std::vector<double> total_us;
  std::vector<std::uint32_t> byte_checked_iterations;
  std::uint64_t mismatch_count = 0;
  std::vector<Mismatch> mismatches;
  SparkCacheSnapshotReadyView last_view{};
  SparkCacheSnapshotStats stats{};
  bool saturation_passed = false;
  std::uint64_t intentional_would_block = 0;
  std::uint64_t unexpected_would_block = 0;
  SparkCacheSnapshotStats saturation_stats{};
  std::uint32_t saturation_cycles_completed = 0;
  std::uint32_t max_outstanding = 0;
  std::uint32_t distinct_slots_observed = 0;
  std::uint64_t cpu_readback_bytes = 0;
  std::uint64_t cpu_readback_checksum = 0;
  std::uint64_t cpu_consume_passes = 0;
  std::uint64_t cpu_warm_read_bytes = 0;
  std::uint64_t cpu_warm_read_passes = 0;
  std::uint64_t cpu_exact_check_bytes = 0;
  std::uint64_t cpu_read_during_gpu_fill_samples = 0;
  std::vector<double> cpu_first_touch_ms;
  std::vector<double> cpu_warm_read_ms;
  std::vector<double> end_to_end_ms;
  std::uint64_t device_free_before_create = 0;
  std::uint64_t device_total_before_create = 0;
  std::uint64_t device_free_after_configure = 0;
  std::uint64_t device_total_after_configure = 0;
  std::uint64_t device_free_after_shutdown = 0;
  std::uint64_t device_total_after_shutdown = 0;
};

struct PendingTicket {
  SparkCacheSnapshotTicket ticket{};
  std::uint64_t context_sequence = 0;
  std::uint64_t logical_start = 0;
  std::uint32_t iteration = 0;
  std::chrono::steady_clock::time_point started{};
  std::chrono::steady_clock::time_point submitted{};
};

bool parse_u32(
    const char* text,
    std::uint32_t minimum,
    std::uint32_t maximum,
    std::uint32_t* output) {
  if (text == nullptr || *text == '\0' || output == nullptr) {
    return false;
  }
  std::uint32_t value = 0;
  const char* end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end ||
      value < minimum || value > maximum) {
    return false;
  }
  *output = value;
  return true;
}

bool parse_options(
    int argc,
    char** argv,
    Options* options,
    std::string* error) {
  if (options == nullptr || error == nullptr) {
    return false;
  }
  std::array<bool, 12> seen{};
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      *error = "every option requires a value";
      return false;
    }
    const std::string key(argv[index]);
    const char* value = argv[index + 1];
    std::size_t seen_index = 0;
    if (key == "--arena") {
      seen_index = 0;
      if (std::strcmp(value, "mapped") == 0) {
        options->arena_mode = SPARK_CACHE_SNAPSHOT_MAPPED_HOST;
        options->arena_name = "mapped";
      } else if (std::strcmp(value, "managed") == 0) {
        options->arena_mode = SPARK_CACHE_SNAPSHOT_MANAGED;
        options->arena_name = "managed";
      } else {
        *error = "--arena must be mapped or managed";
        return false;
      }
    } else if (key == "--slots") {
      seen_index = 1;
      if (!parse_u32(value, 2, 3, &options->slot_count)) {
        *error = "--slots must be 2 or 3";
        return false;
      }
    } else if (key == "--rank") {
      seen_index = 2;
      if (!parse_u32(value, 0, 3, &options->rank)) {
        *error = "--rank must be in [0, 3]";
        return false;
      }
    } else if (key == "--rows") {
      seen_index = 3;
      if (!parse_u32(value, 64, 1024, &options->row_count) ||
          (options->row_count != 64 && options->row_count != 1024)) {
        *error = "--rows must be 64 or 1024";
        return false;
      }
    } else if (key == "--iterations") {
      seen_index = 4;
      if (!parse_u32(value, 1, 1000000, &options->iterations)) {
        *error = "--iterations must be in [1, 1000000]";
        return false;
      }
    } else if (key == "--compare-every") {
      seen_index = 5;
      if (!parse_u32(value, 1, 1000000, &options->compare_every)) {
        *error = "--compare-every must be in [1, 1000000]";
        return false;
      }
    } else if (key == "--pipeline-depth") {
      seen_index = 6;
      if (!parse_u32(value, 1, 3, &options->pipeline_depth)) {
        *error = "--pipeline-depth must be in [1, 3]";
        return false;
      }
    } else if (key == "--writer-hold-us") {
      seen_index = 7;
      if (!parse_u32(value, 0, 10000000, &options->writer_hold_us)) {
        *error = "--writer-hold-us must be in [0, 10000000]";
        return false;
      }
    } else if (key == "--profile") {
      seen_index = 8;
      if (std::strcmp(value, "compact") == 0) {
        options->profile = FixtureProfile::kCompact;
        options->profile_name = "compact";
      } else if (std::strcmp(value, "glm52") == 0) {
        options->profile = FixtureProfile::kGlm52;
        options->profile_name = "glm52";
      } else {
        *error = "--profile must be compact or glm52";
        return false;
      }
    } else if (key == "--slot-mib") {
      seen_index = 9;
      std::uint32_t slot_mib = 0;
      if (!parse_u32(value, 2, 64, &slot_mib) ||
          (slot_mib != 2 && slot_mib != 32 && slot_mib != 64)) {
        *error = "--slot-mib must be 2, 32, or 64";
        return false;
      }
      options->slot_bytes =
          static_cast<std::uint64_t>(slot_mib) * kMebibyte;
    } else if (key == "--saturation-cycles") {
      seen_index = 10;
      if (!parse_u32(value, 1, 1000000, &options->saturation_cycles)) {
        *error = "--saturation-cycles must be in [1, 1000000]";
        return false;
      }
    } else if (key == "--overlap-samples") {
      seen_index = 11;
      if (!parse_u32(value, 0, 1000000, &options->overlap_samples)) {
        *error = "--overlap-samples must be in [0, 1000000]";
        return false;
      }
    } else {
      *error = "unknown option: " + key;
      return false;
    }
    if (seen[seen_index]) {
      *error = "duplicate option: " + key;
      return false;
    }
    seen[seen_index] = true;
  }
  if (!std::all_of(
          seen.begin(), seen.begin() + 5, [](bool value) {
            return value;
          })) {
    *error =
        "required: --arena mapped|managed --slots 2|3 --rank 0..3 "
        "--rows 64|1024 --iterations N";
    return false;
  }
  if (options->pipeline_depth > options->slot_count) {
    *error = "--pipeline-depth cannot exceed --slots";
    return false;
  }
  if (options->profile == FixtureProfile::kGlm52 && !seen[9]) {
    options->slot_bytes = 64ULL * kMebibyte;
  }
  if (options->profile == FixtureProfile::kGlm52 &&
      options->slot_bytes < 32ULL * kMebibyte) {
    *error = "glm52 profile requires --slot-mib 32 or 64";
    return false;
  }
  return true;
}

std::string json_string(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20) {
          output << "\\u"
                 << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character)
                 << std::dec << std::setfill(' ');
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

std::uint8_t fixture_byte(
    std::uint32_t kind,
    std::uint32_t layer,
    std::uint32_t row,
    std::uint32_t byte) {
  return static_cast<std::uint8_t>(
      (kind * 71U + layer * 43U + row * 17U + byte * 29U) &
      0xFFU);
}

std::string layer_name(
    std::uint32_t layer,
    const char* suffix) {
  std::ostringstream output;
  output << "model.layers." << std::setw(2) << std::setfill('0')
         << layer << "." << suffix;
  return output.str();
}

std::vector<Fixture> make_fixtures(const Options& options) {
  std::vector<Fixture> fixtures;
  if (options.profile == FixtureProfile::kCompact) {
    fixtures = {
        {
            "model.layers.00.mla",
            SPARK_CACHE_SNAPSHOT_TARGET_CKV,
            0,
            512,
            368,
            {},
            nullptr,
        },
        {
            "model.layers.01.mla",
            SPARK_CACHE_SNAPSHOT_TARGET_CKV,
            1,
            512,
            368,
            {},
            nullptr,
        },
        {
            "model.layers.00.indexer",
            SPARK_CACHE_SNAPSHOT_SPARSE_INDEXER,
            0,
            256,
            132,
            {},
            nullptr,
        },
        {
            "model.layers.00.mtp",
            SPARK_CACHE_SNAPSHOT_MTP_DRAFT_KV,
            0,
            512,
            368,
            {},
            nullptr,
        },
    };
  } else {
    fixtures.reserve(101);
    for (std::uint32_t layer = 0; layer < 79; ++layer) {
      fixtures.push_back(
          {
              layer_name(layer, "mla"),
              SPARK_CACHE_SNAPSHOT_TARGET_CKV,
              layer,
              368,
              368,
              {},
              nullptr,
          });
    }
    for (std::uint32_t layer = 0; layer < 22; ++layer) {
      fixtures.push_back(
          {
              layer_name(layer, "indexer"),
              SPARK_CACHE_SNAPSHOT_SPARSE_INDEXER,
              layer,
              132,
              132,
              {},
              nullptr,
          });
    }
  }
  for (auto& fixture : fixtures) {
    fixture.host.assign(
        static_cast<std::size_t>(kSourceRows) * fixture.stride,
        0xEE);
    for (std::uint32_t row = 0; row < kSourceRows; ++row) {
      const auto row_start =
          static_cast<std::size_t>(row) * fixture.stride;
      for (std::uint32_t byte = 0; byte < fixture.width; ++byte) {
        fixture.host[row_start + byte] =
            fixture_byte(fixture.kind, fixture.ordinal, row, byte);
      }
    }
  }
  return fixtures;
}

std::vector<std::uint32_t> make_slots(
    std::uint32_t rank,
    std::uint32_t rows) {
  std::vector<std::uint32_t> slots(rows);
  for (std::uint32_t index = 0; index < rows; ++index) {
    slots[index] = (11U + rank * 53U + index * 37U) % 2039U;
  }
  return slots;
}

std::uint64_t align64(std::uint64_t value) {
  return (value + 63U) & ~std::uint64_t{63U};
}

bool validate_geometry(
    const Options& options,
    const SparkCacheSnapshotReadyView& view,
    std::uint64_t context_sequence,
  std::uint64_t logical_start,
  std::string* error) {
  const std::uint64_t target_layers =
      options.profile == FixtureProfile::kGlm52 ? 79U : 2U;
  const std::uint64_t indexer_layers =
      options.profile == FixtureProfile::kGlm52 ? 22U : 1U;
  const std::uint64_t mtp_layers =
      options.profile == FixtureProfile::kGlm52 ? 0U : 1U;
  const std::uint64_t target_bytes =
      static_cast<std::uint64_t>(options.row_count) *
      368U * target_layers;
  const std::uint64_t indexer_offset = align64(target_bytes);
  const std::uint64_t indexer_bytes =
      static_cast<std::uint64_t>(options.row_count) *
      132U * indexer_layers;
  const std::uint64_t mtp_offset =
      align64(indexer_offset + indexer_bytes);
  const std::uint64_t mtp_bytes =
      static_cast<std::uint64_t>(options.row_count) *
      368U * mtp_layers;
  const std::uint64_t used_bytes =
      mtp_layers == 0
          ? indexer_offset + indexer_bytes
          : mtp_offset + mtp_bytes;
  const std::uint32_t expected_mask =
      mtp_layers == 0 ? 0b011U : 0b111U;

  const bool valid =
      view.capacity_bytes == options.slot_bytes &&
      view.used_bytes == used_bytes &&
      view.context_sequence == context_sequence &&
      view.logical_start == logical_start &&
      view.generation != 0 &&
      view.row_count == options.row_count &&
      view.slot_index < options.slot_count &&
      view.record_mask == expected_mask &&
      view.state == SPARK_CACHE_SNAPSHOT_SLOT_WRITING &&
      view.record_offset_bytes[0] == 0 &&
      view.record_offset_bytes[1] == indexer_offset &&
      view.record_offset_bytes[2] ==
          (mtp_layers == 0 ? 0U : mtp_offset) &&
      view.record_length_bytes[0] == target_bytes &&
      view.record_length_bytes[1] == indexer_bytes &&
      view.record_length_bytes[2] == mtp_bytes &&
      view.record_offset_bytes[3] == 0 &&
      view.record_length_bytes[3] == 0;
  if (!valid) {
    *error = "ready-view geometry or transaction metadata mismatch";
  }
  return valid;
}

void compare_fixture(
    const Fixture& fixture,
    const std::vector<std::uint32_t>& slots,
    const SparkCacheSnapshotReadyView& view,
    Result* result) {
  const volatile auto* payload =
      reinterpret_cast<const volatile std::uint8_t*>(
      static_cast<std::uintptr_t>(view.host_address));
  const std::uint64_t layer_bytes =
      static_cast<std::uint64_t>(slots.size()) * fixture.width;
  const std::uint64_t layer_start =
      view.record_offset_bytes[fixture.kind] +
      static_cast<std::uint64_t>(fixture.ordinal) * layer_bytes;
  for (std::uint32_t output_row = 0;
       output_row < slots.size();
       ++output_row) {
    const auto physical_row = slots[output_row];
    for (std::uint32_t byte = 0; byte < fixture.width; ++byte) {
      const std::uint64_t payload_offset =
          layer_start +
          static_cast<std::uint64_t>(output_row) * fixture.width +
          byte;
      const auto expected = fixture_byte(
          fixture.kind, fixture.ordinal, physical_row, byte);
      const auto actual = payload[payload_offset];
      if (actual == expected) {
        continue;
      }
      result->mismatch_count += 1;
      if (result->mismatches.size() < kMaximumMismatchSamples) {
        result->mismatches.push_back(
            {
                fixture.name,
                fixture.kind,
                fixture.ordinal,
                output_row,
                physical_row,
                byte,
                payload_offset,
                expected,
                actual,
            });
      }
    }
  }
}

// This is a full-byte consumer, not a cryptographic hash. The 64-bit additive
// reduction is deliberately compiler-vectorizable; exact correctness remains
// the sparse source-by-source byte comparison below.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
std::uint64_t consume_payload_vectorized(
    const SparkCacheSnapshotReadyView& view) {
  const auto* payload = reinterpret_cast<const std::uint8_t*>(
      static_cast<std::uintptr_t>(view.host_address));
  const auto address = reinterpret_cast<std::uintptr_t>(payload);
  const std::uint64_t alignment_bytes =
      (sizeof(std::uint64_t) -
       (address & (sizeof(std::uint64_t) - 1U))) &
      (sizeof(std::uint64_t) - 1U);
  const std::uint64_t prefix_bytes =
      std::min(alignment_bytes, view.used_bytes);

  std::uint64_t checksum = 0;
  std::uint64_t offset = 0;
  for (; offset < prefix_bytes; ++offset) {
    checksum += payload[offset];
  }
  const auto* lanes = reinterpret_cast<const std::uint64_t*>(
      payload + offset);
  const std::uint64_t lane_count =
      (view.used_bytes - offset) / sizeof(std::uint64_t);
  std::uint64_t lane_checksum = 0;
  for (std::uint64_t lane = 0; lane < lane_count; ++lane) {
    lane_checksum += lanes[lane];
  }
  checksum += lane_checksum;
  offset += lane_count * sizeof(std::uint64_t);
  for (; offset < view.used_bytes; ++offset) {
    checksum += payload[offset];
  }
  return checksum;
}

std::string snapshot_error(
    SparkCacheSnapshot* snapshot,
    SparkCacheSnapshotStatus status,
    const char* operation) {
  std::array<char, 512> detail{};
  (void)spark_cache_snapshot_copy_last_error(
      snapshot, detail.data(), detail.size());
  std::ostringstream output;
  output << operation << ": "
         << spark_cache_snapshot_status_string(status);
  if (detail[0] != '\0') {
    output << ": " << detail.data();
  }
  return output.str();
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(
      std::ceil(quantile * static_cast<double>(values.size())));
  return values[std::max<std::size_t>(1, rank) - 1];
}

bool should_compare(
    std::uint32_t iteration,
    const Options& options) {
  return iteration == 0 ||
         iteration + 1 == options.iterations ||
         iteration % options.compare_every == 0;
}

void write_latency_json(
    std::ostringstream* output,
    const char* name,
    const std::vector<double>& values) {
  *output << "\"" << name << "\":{"
          << "\"samples\":" << values.size() << ","
          << "\"p50\":" << percentile(values, 0.50) << ","
          << "\"p95\":" << percentile(values, 0.95) << ","
          << "\"p99\":" << percentile(values, 0.99) << "}";
}

void write_stats_json(
    std::ostringstream* output,
    const SparkCacheSnapshotStats& stats) {
  *output << "{"
          << "\"submitted_bytes\":" << stats.submitted_bytes << ","
          << "\"completed_bytes\":" << stats.completed_bytes << ","
          << "\"released_bytes\":" << stats.released_bytes << ","
          << "\"submissions\":" << stats.submissions << ","
          << "\"claims\":" << stats.claims << ","
          << "\"releases\":" << stats.releases << ","
          << "\"would_block\":" << stats.would_block << ","
          << "\"abandoned\":" << stats.abandoned << ","
          << "\"stale_tickets\":" << stats.stale_tickets << "}";
}

std::string result_json(
    const Options& options,
    const Result& result) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3);
  output << "{"
         << "\"schema\":\"sparkcache.snapshot_matrix.v1\","
         << "\"passed\":" << (result.passed ? "true" : "false") << ","
         << "\"error\":"
         << (result.error.empty() ? "null" : json_string(result.error))
         << ","
         << "\"config\":{"
         << "\"arena\":" << json_string(options.arena_name) << ","
         << "\"slots\":" << options.slot_count << ","
         << "\"rank\":" << options.rank << ","
         << "\"rows\":" << options.row_count << ","
         << "\"iterations\":" << options.iterations << ","
         << "\"compare_every\":" << options.compare_every << ","
         << "\"pipeline_depth\":" << options.pipeline_depth << ","
         << "\"writer_hold_us\":" << options.writer_hold_us << ","
         << "\"saturation_cycles\":" << options.saturation_cycles << ","
         << "\"overlap_samples\":" << options.overlap_samples << ","
         << "\"profile\":" << json_string(options.profile_name) << ","
         << "\"slot_bytes\":" << options.slot_bytes << "},"
         << "\"memory\":{"
         << "\"nominal_arena_bytes\":"
         << options.slot_bytes * options.slot_count << ","
         << "\"before_create\":{"
         << "\"free\":" << result.device_free_before_create << ","
         << "\"total\":" << result.device_total_before_create << "},"
         << "\"after_configure\":{"
         << "\"free\":" << result.device_free_after_configure << ","
         << "\"total\":" << result.device_total_after_configure << "},"
         << "\"after_shutdown\":{"
         << "\"free\":" << result.device_free_after_shutdown << ","
         << "\"total\":" << result.device_total_after_shutdown << "}},"
         << "\"would_block\":{"
         << "\"intentional\":" << result.intentional_would_block << ","
         << "\"unexpected\":" << result.unexpected_would_block << "},"
         << "\"saturation\":{"
         << "\"passed\":"
         << (result.saturation_passed ? "true" : "false") << ","
         << "\"cycles_completed\":"
         << result.saturation_cycles_completed << ","
         << "\"max_outstanding\":" << result.max_outstanding << ","
         << "\"distinct_slots_observed\":"
         << result.distinct_slots_observed << ","
         << "\"stats\":";
  write_stats_json(&output, result.saturation_stats);
  output << "},"
         << "\"latency_us\":{";
  write_latency_json(&output, "submit", result.submit_us);
  output << ",";
  write_latency_json(&output, "gather", result.gather_us);
  output << ",";
  write_latency_json(&output, "total", result.total_us);
  output << "},"
         << "\"cpu_consume_ms\":{";
  write_latency_json(
      &output, "first_touch", result.cpu_first_touch_ms);
  output << ",";
  write_latency_json(
      &output, "warm_read", result.cpu_warm_read_ms);
  output << ",";
  write_latency_json(
      &output, "end_to_end", result.end_to_end_ms);
  output << "},"
         << "\"cpu_readback\":{"
         << "\"bytes\":" << result.cpu_readback_bytes << ","
         << "\"consume_passes\":" << result.cpu_consume_passes << ","
         << "\"warm_read_bytes\":"
         << result.cpu_warm_read_bytes << ","
         << "\"warm_read_passes\":"
         << result.cpu_warm_read_passes << ","
         << "\"exact_check_bytes\":"
         << result.cpu_exact_check_bytes << ","
         << "\"checksum\":" << result.cpu_readback_checksum << ","
         << "\"mismatches\":" << result.mismatch_count << ","
         << "\"checks\":"
         << result.byte_checked_iterations.size() << ","
         << "\"read_during_gpu_fill_samples\":"
         << result.cpu_read_during_gpu_fill_samples << "},"
         << "\"standalone_rank\":{"
         << "\"production_payload_bytes\":"
         << result.last_view.used_bytes << ","
         << "\"production_byte_checks\":"
         << result.byte_checked_iterations.size() << ","
         << "\"cpu_readback_bytes\":"
         << result.cpu_exact_check_bytes << ","
         << "\"cpu_readback_mismatches\":"
         << result.mismatch_count << ","
         << "\"saturation_cycles\":"
         << result.saturation_cycles_completed << ","
         << "\"max_outstanding\":"
         << result.max_outstanding << ","
         << "\"distinct_slots_observed\":"
         << result.distinct_slots_observed << ","
         << "\"depth_plus_one_would_block\":"
         << (result.intentional_would_block ==
                     options.saturation_cycles
                 ? "true"
                 : "false")
         << ","
         << "\"cpu_read_during_gpu_fill_samples\":"
         << result.cpu_read_during_gpu_fill_samples << ","
         << "\"cpu_first_touch_p95_ms\":"
         << percentile(result.cpu_first_touch_ms, 0.95) << ","
         << "\"cpu_warm_read_p95_ms\":"
         << percentile(result.cpu_warm_read_ms, 0.95) << ","
         << "\"end_to_end_p95_ms\":"
         << percentile(result.end_to_end_ms, 0.95) << ","
         << "\"end_to_end_p99_ms\":"
         << percentile(result.end_to_end_ms, 0.99) << "},"
         << "\"byte_checked_iterations\":[";
  for (std::size_t index = 0;
       index < result.byte_checked_iterations.size();
       ++index) {
    if (index != 0) {
      output << ",";
    }
    output << result.byte_checked_iterations[index];
  }
  output << "],"
         << "\"geometry\":{"
         << "\"record_mask\":" << result.last_view.record_mask << ","
         << "\"record_offsets\":["
         << result.last_view.record_offset_bytes[0] << ","
         << result.last_view.record_offset_bytes[1] << ","
         << result.last_view.record_offset_bytes[2] << ","
         << result.last_view.record_offset_bytes[3] << "],"
         << "\"record_lengths\":["
         << result.last_view.record_length_bytes[0] << ","
         << result.last_view.record_length_bytes[1] << ","
         << result.last_view.record_length_bytes[2] << ","
         << result.last_view.record_length_bytes[3] << "],"
         << "\"used_bytes\":" << result.last_view.used_bytes << "},"
         << "\"stats\":";
  write_stats_json(&output, result.stats);
  output << ","
         << "\"mismatch_count\":" << result.mismatch_count << ","
         << "\"mismatches\":[";
  for (std::size_t index = 0; index < result.mismatches.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    const auto& mismatch = result.mismatches[index];
    output << "{"
           << "\"source\":" << json_string(mismatch.source) << ","
           << "\"kind\":" << mismatch.kind << ","
           << "\"layer\":" << mismatch.layer << ","
           << "\"output_row\":" << mismatch.output_row << ","
           << "\"physical_row\":" << mismatch.physical_row << ","
           << "\"byte\":" << mismatch.byte << ","
           << "\"payload_offset\":" << mismatch.payload_offset << ","
           << "\"expected\":" << static_cast<unsigned int>(mismatch.expected)
           << ","
           << "\"actual\":" << static_cast<unsigned int>(mismatch.actual)
           << "}";
  }
  output << "]}";
  return output.str();
}

bool cuda_call(
    cudaError_t status,
    const char* operation,
    Result* result) {
  if (status == cudaSuccess) {
    return true;
  }
  result->error =
      std::string(operation) + ": " + cudaGetErrorString(status);
  return false;
}

bool capture_device_memory(
    std::uint64_t* free_bytes,
    std::uint64_t* total_bytes,
    const char* operation,
    Result* result) {
  std::size_t free_value = 0;
  std::size_t total_value = 0;
  if (!cuda_call(
          cudaMemGetInfo(&free_value, &total_value),
          operation,
          result)) {
    return false;
  }
  *free_bytes = static_cast<std::uint64_t>(free_value);
  *total_bytes = static_cast<std::uint64_t>(total_value);
  return true;
}

Result run_matrix(const Options& options) {
  Result result;
  auto fixtures = make_fixtures(options);
  const auto physical_slots =
      make_slots(options.rank, options.row_count);
  cudaStream_t stream = nullptr;
  SparkCacheSnapshot* snapshot = nullptr;
  bool initialized = true;

  for (auto& fixture : fixtures) {
    if (!cuda_call(
            cudaMalloc(
                reinterpret_cast<void**>(&fixture.device),
                fixture.host.size()),
            "cudaMalloc(fixture)",
            &result) ||
        !cuda_call(
            cudaMemcpy(
                fixture.device,
                fixture.host.data(),
                fixture.host.size(),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(fixture)",
            &result)) {
      initialized = false;
      break;
    }
  }
  if (initialized) {
    initialized = cuda_call(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate",
        &result);
  }

  SparkCacheSnapshotConfig config{};
  config.abi_version = SPARK_CACHE_SNAPSHOT_ABI_VERSION;
  config.arena_mode = options.arena_mode;
  config.slot_bytes = options.slot_bytes;
  config.slot_count = options.slot_count;
  config.max_sources =
      static_cast<std::uint32_t>(fixtures.size());
  config.max_rows = 1024;
  config.device_ordinal = 0;

  std::vector<SparkCacheSnapshotSource> sources(fixtures.size());
  for (std::size_t index = 0; index < fixtures.size(); ++index) {
    const auto& fixture = fixtures[index];
    sources[index] = {
        reinterpret_cast<std::uintptr_t>(fixture.device),
        kSourceRows,
        fixture.stride,
        fixture.width,
        fixture.kind,
        fixture.ordinal,
    };
  }
  const auto create_and_configure =
      [&](SparkCacheSnapshot** output) -> bool {
    auto status = spark_cache_snapshot_create(&config, output);
    if (status != SPARK_CACHE_SNAPSHOT_OK) {
      result.error = snapshot_error(*output, status, "create");
      return false;
    }
    status = spark_cache_snapshot_configure_sources(
        *output, sources.data(), sources.size());
    if (status != SPARK_CACHE_SNAPSHOT_OK) {
      result.error =
          snapshot_error(*output, status, "configure_sources");
      return false;
    }
    return true;
  };

  // Saturation uses a separate runtime so its deliberate WOULD_BLOCK and
  // abandonment accounting never contaminate latency or matrix statistics.
  if (initialized) {
    initialized = create_and_configure(&snapshot);
  }
  if (initialized) {
    std::array<bool, SPARK_CACHE_SNAPSHOT_MAX_SLOTS>
        observed_saturation_slots{};
    for (std::uint32_t cycle = 0;
         initialized && cycle < options.saturation_cycles;
         ++cycle) {
      const std::uint64_t saturation_context =
          0x5341545500000000ULL +
          static_cast<std::uint64_t>(options.rank) * 1000000ULL +
          cycle + 1;
      std::vector<SparkCacheSnapshotTicket> saturation_tickets(
          options.slot_count);
      std::array<bool, SPARK_CACHE_SNAPSHOT_MAX_SLOTS> cycle_slots{};
      for (std::uint32_t index = 0;
           initialized && index < options.slot_count;
           ++index) {
        SparkCacheSnapshotSubmission submission{};
        submission.context_sequence = saturation_context;
        submission.logical_start =
            0x100000ULL + static_cast<std::uint64_t>(cycle) *
                               options.slot_count * options.row_count +
            static_cast<std::uint64_t>(index) * options.row_count;
        submission.row_count = options.row_count;
        const auto status = spark_cache_snapshot_try_submit(
            snapshot,
            &submission,
            physical_slots.data(),
            reinterpret_cast<std::uintptr_t>(stream),
            &saturation_tickets[index]);
        if (status == SPARK_CACHE_SNAPSHOT_WOULD_BLOCK) {
          result.unexpected_would_block += 1;
        }
        if (status != SPARK_CACHE_SNAPSHOT_OK) {
          result.error =
              snapshot_error(snapshot, status, "saturation fill submit");
          initialized = false;
          break;
        }
        const auto slot_index = saturation_tickets[index].slot_index;
        if (slot_index >= options.slot_count ||
            cycle_slots[slot_index]) {
          result.error =
              "saturation drill did not occupy distinct slots";
          initialized = false;
          break;
        }
        cycle_slots[slot_index] = true;
        observed_saturation_slots[slot_index] = true;
      }
      result.max_outstanding = std::max(
          result.max_outstanding,
          initialized ? options.slot_count : 0U);
      if (initialized) {
        SparkCacheSnapshotSubmission extra{};
        extra.context_sequence = saturation_context;
        extra.logical_start = 0x200000ULL + cycle;
        extra.row_count = options.row_count;
        SparkCacheSnapshotTicket unused{};
        const auto status = spark_cache_snapshot_try_submit(
            snapshot,
            &extra,
            physical_slots.data(),
            reinterpret_cast<std::uintptr_t>(stream),
            &unused);
        if (status == SPARK_CACHE_SNAPSHOT_WOULD_BLOCK) {
          result.intentional_would_block += 1;
        } else {
          if (status == SPARK_CACHE_SNAPSHOT_OK) {
            result.error =
                "saturation drill accepted a submit beyond slot capacity";
          } else {
            result.error = snapshot_error(
                snapshot, status, "saturation extra submit");
          }
          initialized = false;
        }
      }
      const auto abandon_status = spark_cache_snapshot_abandon_context(
          snapshot, saturation_context);
      if (abandon_status != SPARK_CACHE_SNAPSHOT_OK &&
          result.error.empty()) {
        result.error = snapshot_error(
            snapshot, abandon_status, "saturation abandon");
        initialized = false;
      }
      if (!cuda_call(
              cudaStreamSynchronize(stream),
              "cudaStreamSynchronize(saturation)",
              &result)) {
        initialized = false;
      }
      if (initialized) {
        result.saturation_cycles_completed += 1;
      }
    }
    result.distinct_slots_observed = static_cast<std::uint32_t>(
        std::count(
            observed_saturation_slots.begin(),
            observed_saturation_slots.end(),
            true));
    if (snapshot != nullptr) {
      const auto stats_status = spark_cache_snapshot_get_stats(
          snapshot, &result.saturation_stats);
      if (stats_status != SPARK_CACHE_SNAPSHOT_OK &&
          result.error.empty()) {
        result.error =
            snapshot_error(snapshot, stats_status, "saturation get_stats");
        initialized = false;
      }
      const auto shutdown_status = spark_cache_snapshot_shutdown(snapshot);
      if (shutdown_status != SPARK_CACHE_SNAPSHOT_OK &&
          result.error.empty()) {
        result.error =
            snapshot_error(snapshot, shutdown_status, "saturation shutdown");
        initialized = false;
      }
      if (shutdown_status == SPARK_CACHE_SNAPSHOT_OK) {
        spark_cache_snapshot_destroy(snapshot);
        snapshot = nullptr;
      }
    }
    const bool saturation_stats_valid =
        result.saturation_stats.submissions ==
            static_cast<std::uint64_t>(options.slot_count) *
                options.saturation_cycles &&
        result.saturation_stats.claims == 0 &&
        result.saturation_stats.releases == 0 &&
        result.saturation_stats.would_block ==
            options.saturation_cycles &&
        result.saturation_stats.abandoned ==
            static_cast<std::uint64_t>(options.slot_count) *
                options.saturation_cycles &&
        result.saturation_stats.stale_tickets == 0;
    result.saturation_passed =
        initialized &&
        result.saturation_cycles_completed ==
            options.saturation_cycles &&
        result.max_outstanding == options.slot_count &&
        result.distinct_slots_observed == options.slot_count &&
        result.intentional_would_block ==
            options.saturation_cycles &&
        result.unexpected_would_block == 0 &&
        saturation_stats_valid;
    if (initialized && !result.saturation_passed) {
      result.error = "saturation drill stats disagree with slot capacity";
      initialized = false;
    }
  }

  if (initialized) {
    initialized = capture_device_memory(
        &result.device_free_before_create,
        &result.device_total_before_create,
        "cudaMemGetInfo(before create)",
        &result);
  }
  if (initialized) {
    initialized = create_and_configure(&snapshot);
  }
  if (initialized) {
    initialized = capture_device_memory(
        &result.device_free_after_configure,
        &result.device_total_after_configure,
        "cudaMemGetInfo(after configure)",
        &result);
  }
  std::deque<PendingTicket> pending;
  std::uint32_t next_iteration = 0;
  const std::uint64_t logical_base =
      4096ULL + static_cast<std::uint64_t>(options.rank) * 4096ULL;
  while (initialized &&
         (next_iteration < options.iterations || !pending.empty())) {
    while (initialized &&
           next_iteration < options.iterations &&
           pending.size() < options.pipeline_depth) {
      SparkCacheSnapshotSubmission submission{};
      submission.context_sequence =
          0x534E415000000000ULL +
          static_cast<std::uint64_t>(options.rank) * 1000000ULL +
          next_iteration + 1;
      submission.logical_start =
          logical_base +
          static_cast<std::uint64_t>(next_iteration) *
              options.row_count * 4ULL;
      submission.row_count = options.row_count;
      PendingTicket pending_ticket{};
      pending_ticket.context_sequence = submission.context_sequence;
      pending_ticket.logical_start = submission.logical_start;
      pending_ticket.iteration = next_iteration;
      pending_ticket.started = std::chrono::steady_clock::now();
      const auto status = spark_cache_snapshot_try_submit(
          snapshot,
          &submission,
          physical_slots.data(),
          reinterpret_cast<std::uintptr_t>(stream),
          &pending_ticket.ticket);
      pending_ticket.submitted = std::chrono::steady_clock::now();
      if (status == SPARK_CACHE_SNAPSHOT_WOULD_BLOCK) {
        result.unexpected_would_block += 1;
      }
      if (status != SPARK_CACHE_SNAPSHOT_OK) {
        result.error = snapshot_error(snapshot, status, "try_submit");
        initialized = false;
        break;
      }
      pending.push_back(pending_ticket);
      next_iteration += 1;
    }
    if (!initialized || pending.empty()) {
      break;
    }

    PendingTicket current = pending.front();
    pending.pop_front();
    SparkCacheSnapshotReadyView ready{};
    const auto deadline = current.started + kClaimTimeout;
    auto status = SPARK_CACHE_SNAPSHOT_NOT_READY;
    while (true) {
      status = spark_cache_snapshot_claim(
          snapshot, &current.ticket, &ready);
      if (status == SPARK_CACHE_SNAPSHOT_OK) {
        break;
      }
      if (status != SPARK_CACHE_SNAPSHOT_NOT_READY) {
        result.error = snapshot_error(snapshot, status, "claim");
        initialized = false;
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        result.error = "claim timed out after 30 seconds";
        initialized = false;
        break;
      }
      std::this_thread::yield();
    }
    if (!initialized) {
      (void)spark_cache_snapshot_abandon_context(
          snapshot, current.context_sequence);
      break;
    }
    const auto finished = std::chrono::steady_clock::now();
    result.submit_us.push_back(
        std::chrono::duration<double, std::micro>(
            current.submitted - current.started)
            .count());
    result.gather_us.push_back(
        std::chrono::duration<double, std::micro>(
            finished - current.submitted)
            .count());
    result.total_us.push_back(
        std::chrono::duration<double, std::micro>(
            finished - current.started)
            .count());

    std::string geometry_error;
    if (!validate_geometry(
            options,
            ready,
            current.context_sequence,
            current.logical_start,
            &geometry_error)) {
      result.error = geometry_error;
      initialized = false;
    } else {
      const bool exact_check =
          should_compare(current.iteration, options);
      const bool seek_overlap =
          options.pipeline_depth > 1 &&
          result.cpu_read_during_gpu_fill_samples <
              options.overlap_samples;
      const bool consume_on_cpu = exact_check || seek_overlap;
      bool later_fill_in_flight = false;
      if (consume_on_cpu && !pending.empty()) {
        for (const auto& later : pending) {
          SparkCacheSnapshotReadyView later_view{};
          const auto later_status = spark_cache_snapshot_poll(
              snapshot, &later.ticket, &later_view);
          if (later_status == SPARK_CACHE_SNAPSHOT_NOT_READY) {
            later_fill_in_flight = true;
            break;
          }
          if (later_status != SPARK_CACHE_SNAPSHOT_OK) {
            result.error = snapshot_error(
                snapshot, later_status, "poll later ticket");
            initialized = false;
            break;
          }
        }
      }
      if (consume_on_cpu && initialized) {
        const auto first_touch_started =
            std::chrono::steady_clock::now();
        const auto first_checksum =
            consume_payload_vectorized(ready);
        const auto first_touch_finished =
            std::chrono::steady_clock::now();
        result.cpu_readback_checksum ^= first_checksum;
        result.cpu_readback_bytes += ready.used_bytes;
        result.cpu_consume_passes += 1;
        result.cpu_first_touch_ms.push_back(
            std::chrono::duration<double, std::milli>(
                first_touch_finished - first_touch_started)
                .count());
        result.end_to_end_ms.push_back(
            std::chrono::duration<double, std::milli>(
                first_touch_finished - current.started)
                .count());
        if (exact_check) {
          const auto warm_read_started =
              std::chrono::steady_clock::now();
          const auto warm_checksum =
              consume_payload_vectorized(ready);
          const auto warm_read_finished =
              std::chrono::steady_clock::now();
          result.cpu_readback_checksum += warm_checksum;
          result.cpu_warm_read_bytes += ready.used_bytes;
          result.cpu_warm_read_passes += 1;
          result.cpu_warm_read_ms.push_back(
              std::chrono::duration<double, std::milli>(
                  warm_read_finished - warm_read_started)
                  .count());
          for (const auto& fixture : fixtures) {
            compare_fixture(
                fixture, physical_slots, ready, &result);
          }
          result.cpu_exact_check_bytes += ready.used_bytes;
          result.byte_checked_iterations.push_back(
              current.iteration);
        }
        if (later_fill_in_flight) {
          result.cpu_read_during_gpu_fill_samples += 1;
        }
      }
    }
    result.last_view = ready;
    if (options.writer_hold_us != 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(options.writer_hold_us));
    }
    status = spark_cache_snapshot_release(snapshot, &current.ticket);
    if (status != SPARK_CACHE_SNAPSHOT_OK) {
      result.error = snapshot_error(snapshot, status, "release");
      initialized = false;
    }
    if (result.mismatch_count != 0) {
      result.error = "byte comparison failed";
      initialized = false;
    }
  }

  if (snapshot != nullptr) {
    const auto stats_status =
        spark_cache_snapshot_get_stats(snapshot, &result.stats);
    if (stats_status != SPARK_CACHE_SNAPSHOT_OK && result.error.empty()) {
      result.error =
          snapshot_error(snapshot, stats_status, "get_stats");
      initialized = false;
    }
    const auto shutdown_status =
        spark_cache_snapshot_shutdown(snapshot);
    if (shutdown_status != SPARK_CACHE_SNAPSHOT_OK) {
      if (result.error.empty()) {
        result.error =
            snapshot_error(snapshot, shutdown_status, "shutdown");
      }
      initialized = false;
    } else {
      spark_cache_snapshot_destroy(snapshot);
      snapshot = nullptr;
    }
  }
  if (result.device_total_before_create != 0 &&
      !capture_device_memory(
          &result.device_free_after_shutdown,
          &result.device_total_after_shutdown,
          "cudaMemGetInfo(after shutdown)",
          &result)) {
    initialized = false;
  }
  if (stream != nullptr &&
      !cuda_call(cudaStreamDestroy(stream), "cudaStreamDestroy", &result)) {
    initialized = false;
  }
  for (auto& fixture : fixtures) {
    if (fixture.device != nullptr &&
        !cuda_call(cudaFree(fixture.device), "cudaFree(fixture)", &result)) {
      initialized = false;
    }
  }

  const std::uint64_t expected_bytes =
      result.last_view.used_bytes * options.iterations;
  const bool stats_valid =
      result.stats.submissions == options.iterations &&
      result.stats.claims == options.iterations &&
      result.stats.releases == options.iterations &&
      result.stats.submitted_bytes == expected_bytes &&
      result.stats.completed_bytes == expected_bytes &&
      result.stats.released_bytes == expected_bytes &&
      result.stats.would_block == 0 &&
      result.stats.abandoned == 0 &&
      result.stats.stale_tickets == 0;
  if (initialized && !stats_valid) {
    result.error = "native stats disagree with completed matrix";
    initialized = false;
  }
  result.passed =
      initialized && result.error.empty() &&
      result.saturation_passed &&
      result.intentional_would_block ==
          options.saturation_cycles &&
      result.unexpected_would_block == 0 &&
      result.mismatch_count == 0 &&
      result.submit_us.size() == options.iterations &&
      result.gather_us.size() == options.iterations &&
      result.total_us.size() == options.iterations &&
      !result.byte_checked_iterations.empty() &&
      result.byte_checked_iterations.front() == 0 &&
      result.byte_checked_iterations.back() + 1 == options.iterations;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string parse_error;
  if (!parse_options(argc, argv, &options, &parse_error)) {
    Result result;
    result.error = parse_error;
    std::cout << result_json(options, result) << "\n";
    return 2;
  }
  Result result = run_matrix(options);
  std::cout << result_json(options, result) << "\n";
  return result.passed ? 0 : 1;
}
