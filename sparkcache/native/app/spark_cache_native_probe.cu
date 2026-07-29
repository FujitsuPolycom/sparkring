#include "spark_cache_placement.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void append_u32_le(std::vector<std::uint8_t>* output, std::uint32_t value) {
  output->push_back(static_cast<std::uint8_t>(value));
  output->push_back(static_cast<std::uint8_t>(value >> 8U));
  output->push_back(static_cast<std::uint8_t>(value >> 16U));
  output->push_back(static_cast<std::uint8_t>(value >> 24U));
}

std::vector<std::uint8_t> make_chunk() {
  const std::string sha(64, '0');
  const std::string header =
      "{\"format_abi\":1,\"logical_end\":8,\"logical_start\":0,"
      "\"records\":["
      "{\"kind\":\"logical_positions\",\"length\":16,\"offset\":0,"
      "\"sha256\":\"" +
      sha +
      "\"},"
      "{\"kind\":\"sparse_indexer\",\"length\":4,\"offset\":16,"
      "\"sha256\":\"" +
      sha +
      "\"},"
      "{\"kind\":\"target_ckv\",\"length\":16,\"offset\":20,"
      "\"sha256\":\"" +
      sha + "\"}]}";
  std::vector<std::uint8_t> encoded{
      'S', 'P', 'C', 'K', 'V', '0', '0', '1'};
  append_u32_le(&encoded, 1);
  append_u32_le(&encoded, static_cast<std::uint32_t>(header.size()));
  encoded.insert(encoded.end(), header.begin(), header.end());
  for (const std::uint32_t position : {0U, 2U, 4U, 6U}) {
    append_u32_le(&encoded, position);
  }
  encoded.insert(encoded.end(), {21, 22, 23, 24});
  encoded.insert(encoded.end(), {1, 2, 3, 4, 5, 6, 7, 8});
  encoded.insert(
      encoded.end(), {11, 12, 13, 14, 15, 16, 17, 18});
  return encoded;
}

bool cuda_ok(cudaError_t result, const char* operation) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(
      stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
  return false;
}

bool placement_ok(
    SparkCachePlacementStatus status,
    SparkCachePlacement* placement,
    const char* operation) {
  if (status == SPARK_CACHE_PLACEMENT_OK) {
    return true;
  }
  std::fprintf(
      stderr,
      "%s failed: status=%d detail=%s\n",
      operation,
      static_cast<int>(status),
      spark_cache_placement_last_error(placement));
  return false;
}

}  // namespace

int main() {
  SparkCachePlacementConfig config{};
  config.abi_version = SPARK_CACHE_PLACEMENT_ABI_VERSION;
  config.arena_mode = SPARK_CACHE_ARENA_MAPPED_HOST;
  config.arena_bytes = 64ULL * 1024ULL * 1024ULL;
  config.max_destinations = 8;
  config.max_slots = 16;
  config.max_chunks_per_slab = 8;
  config.device_ordinal = 0;

  SparkCachePlacement* placement = nullptr;
  if (!placement_ok(
          spark_cache_placement_create(&config, &placement),
          placement,
          "create")) {
    return 1;
  }
  std::uint8_t* target0 = nullptr;
  std::uint8_t* target1 = nullptr;
  std::uint8_t* indexer = nullptr;
  if (!cuda_ok(
          cudaMalloc(reinterpret_cast<void**>(&target0), 16),
          "cudaMalloc(target0)") ||
      !cuda_ok(
          cudaMalloc(reinterpret_cast<void**>(&target1), 16),
          "cudaMalloc(target1)") ||
      !cuda_ok(
          cudaMalloc(reinterpret_cast<void**>(&indexer), 8),
          "cudaMalloc(indexer)") ||
      !cuda_ok(cudaMemset(target0, 0, 16), "cudaMemset(target0)") ||
      !cuda_ok(cudaMemset(target1, 0, 16), "cudaMemset(target1)") ||
      !cuda_ok(cudaMemset(indexer, 0, 8), "cudaMemset(indexer)")) {
    spark_cache_placement_destroy(placement);
    return 1;
  }

  const std::array<SparkCacheDestinationDescriptor, 3> destinations{{
      {
          reinterpret_cast<std::uintptr_t>(target0),
          8,
          2,
          2,
          SPARK_CACHE_RECORD_TARGET_CKV,
          0,
      },
      {
          reinterpret_cast<std::uintptr_t>(target1),
          8,
          2,
          2,
          SPARK_CACHE_RECORD_TARGET_CKV,
          1,
      },
      {
          reinterpret_cast<std::uintptr_t>(indexer),
          8,
          1,
          1,
          SPARK_CACHE_RECORD_SPARSE_INDEXER,
          0,
      },
  }};
  const std::array<std::uint32_t, 4> slots{5, 1, 7, 3};
  bool ok = placement_ok(
      spark_cache_placement_configure_destinations(
          placement, destinations.data(), destinations.size()),
      placement,
      "configure_destinations");
  ok = ok && placement_ok(
      spark_cache_placement_begin_restore(
          placement, slots.data(), slots.size()),
      placement,
      "begin_restore");

  void* arena = nullptr;
  std::uint64_t capacity = 0;
  ok = ok && placement_ok(
      spark_cache_placement_acquire_arena(
          placement, 0, &arena, &capacity),
      placement,
      "acquire_arena");
  const auto encoded = make_chunk();
  if (ok && encoded.size() <= capacity) {
    std::memcpy(arena, encoded.data(), encoded.size());
  } else {
    ok = false;
  }
  SparkCacheChunkDescriptor chunk{};
  std::array<char, 256> parse_error{};
  const auto parsed = spark_cache_parse_verified_v1_chunk(
      arena,
      encoded.size(),
      0,
      static_cast<std::uint32_t>(encoded.size()),
      0,
      2,
      0,
      0,
      (1U << SPARK_CACHE_RECORD_TARGET_CKV) |
          (1U << SPARK_CACHE_RECORD_SPARSE_INDEXER),
      &chunk,
      parse_error.data(),
      parse_error.size());
  if (parsed != SPARK_CACHE_PLACEMENT_OK) {
    std::fprintf(stderr, "parse failed: %s\n", parse_error.data());
    ok = false;
  }
  ok = ok && placement_ok(
      spark_cache_placement_submit_direct_slab(
          placement, 0, encoded.size(), &chunk, 1),
      placement,
      "submit_direct_slab");
  SparkCachePlacementStats stats{};
  ok = ok && placement_ok(
      spark_cache_placement_finish_restore(placement, &stats),
      placement,
      "finish_restore");

  std::array<std::uint8_t, 16> got_target0{};
  std::array<std::uint8_t, 16> got_target1{};
  std::array<std::uint8_t, 8> got_indexer{};
  ok = ok && cuda_ok(
      cudaMemcpy(
          got_target0.data(),
          target0,
          got_target0.size(),
          cudaMemcpyDeviceToHost),
      "copy target0");
  ok = ok && cuda_ok(
      cudaMemcpy(
          got_target1.data(),
          target1,
          got_target1.size(),
          cudaMemcpyDeviceToHost),
      "copy target1");
  ok = ok && cuda_ok(
      cudaMemcpy(
          got_indexer.data(),
          indexer,
          got_indexer.size(),
          cudaMemcpyDeviceToHost),
      "copy indexer");
  ok = ok && got_target0[slots[0] * 2] == 1 &&
       got_target0[slots[0] * 2 + 1] == 2 &&
       got_target0[slots[3] * 2] == 7 &&
       got_target0[slots[3] * 2 + 1] == 8 &&
       got_target1[slots[1] * 2] == 13 &&
       got_target1[slots[1] * 2 + 1] == 14 &&
       got_indexer[slots[2]] == 23;

  std::printf(
      "mapped-direct probe %s: slot_uploads=%u table_uploads=%u"
      " slabs=%u kernels=%u rows=%llu staged_h2d=%llu\n",
      ok ? "PASS" : "FAIL",
      stats.slot_uploads,
      stats.destination_table_uploads,
      stats.slabs_submitted,
      stats.scatter_kernel_launches,
      static_cast<unsigned long long>(stats.restored_rows),
      static_cast<unsigned long long>(stats.staged_h2d_bytes));
  cudaFree(target0);
  cudaFree(target1);
  cudaFree(indexer);
  spark_cache_placement_destroy(placement);
  return ok ? 0 : 1;
}
