#include "spark_cache_snapshot.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool cuda_ok(cudaError_t result, const char* operation) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(
      stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
  return false;
}

bool api_ok(SparkCacheSnapshotStatus status, const char* operation) {
  if (status == SPARK_CACHE_SNAPSHOT_OK) {
    return true;
  }
  std::fprintf(
      stderr,
      "%s failed: %s\n",
      operation,
      spark_cache_snapshot_status_string(status));
  return false;
}

}  // namespace

int main() {
  constexpr std::uint32_t kRows = 8;
  constexpr std::uint32_t kTargetWidth = 16;
  constexpr std::uint32_t kIndexerWidth = 8;
  std::vector<std::uint8_t> target(kRows * kTargetWidth);
  std::vector<std::uint8_t> indexer(kRows * kIndexerWidth);
  for (std::size_t index = 0; index < target.size(); ++index) {
    target[index] = static_cast<std::uint8_t>(index);
  }
  for (std::size_t index = 0; index < indexer.size(); ++index) {
    indexer[index] = static_cast<std::uint8_t>(0x80U + index);
  }

  std::uint8_t* device_target = nullptr;
  std::uint8_t* device_indexer = nullptr;
  cudaStream_t stream = nullptr;
  bool ok =
      cuda_ok(
          cudaMalloc(
              reinterpret_cast<void**>(&device_target), target.size()),
          "cudaMalloc(target)") &&
      cuda_ok(
          cudaMalloc(
              reinterpret_cast<void**>(&device_indexer), indexer.size()),
          "cudaMalloc(indexer)") &&
      cuda_ok(
          cudaMemcpy(
              device_target,
              target.data(),
              target.size(),
              cudaMemcpyHostToDevice),
          "cudaMemcpy(target)") &&
      cuda_ok(
          cudaMemcpy(
              device_indexer,
              indexer.data(),
              indexer.size(),
              cudaMemcpyHostToDevice),
          "cudaMemcpy(indexer)") &&
      cuda_ok(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate");
  if (!ok) {
    cudaFree(device_target);
    cudaFree(device_indexer);
    return 1;
  }

  SparkCacheSnapshotConfig config{};
  config.abi_version = SPARK_CACHE_SNAPSHOT_ABI_VERSION;
  config.arena_mode = SPARK_CACHE_SNAPSHOT_MAPPED_HOST;
  config.slot_bytes = 1024 * 1024;
  config.slot_count = 2;
  config.max_sources = 4;
  config.max_rows = 64;
  config.device_ordinal = 0;

  SparkCacheSnapshot* snapshot = nullptr;
  ok = api_ok(
      spark_cache_snapshot_create(&config, &snapshot), "create");
  std::array<SparkCacheSnapshotSource, 2> sources{{
      {
          reinterpret_cast<std::uintptr_t>(device_target),
          kRows,
          kTargetWidth,
          kTargetWidth,
          SPARK_CACHE_SNAPSHOT_TARGET_CKV,
          0,
      },
      {
          reinterpret_cast<std::uintptr_t>(device_indexer),
          kRows,
          kIndexerWidth,
          kIndexerWidth,
          SPARK_CACHE_SNAPSHOT_SPARSE_INDEXER,
          0,
      },
  }};
  ok = ok && api_ok(
      spark_cache_snapshot_configure_sources(
          snapshot, sources.data(), sources.size()),
      "configure_sources");

  const std::array<std::uint32_t, 4> slots{{3, 1, 7, 0}};
  SparkCacheSnapshotSubmission submission{};
  submission.context_sequence = 77;
  submission.logical_start = 1024;
  submission.row_count = slots.size();
  SparkCacheSnapshotTicket ticket{};
  ok = ok && api_ok(
      spark_cache_snapshot_try_submit(
          snapshot,
          &submission,
          slots.data(),
          reinterpret_cast<std::uintptr_t>(stream),
          &ticket),
      "try_submit");
  ok = ok && cuda_ok(cudaStreamSynchronize(stream), "stream sync");

  SparkCacheSnapshotReadyView ready{};
  ok = ok && api_ok(
      spark_cache_snapshot_claim(snapshot, &ticket, &ready), "claim");
  ok = ok &&
       spark_cache_snapshot_shutdown(snapshot) ==
           SPARK_CACHE_SNAPSHOT_INVALID_STATE;
  if (ok) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(
        static_cast<std::uintptr_t>(ready.host_address));
    for (std::size_t output_row = 0;
         output_row < slots.size();
         ++output_row) {
      const auto source_row = slots[output_row];
      ok = ok &&
           std::memcmp(
               bytes + ready.record_offset_bytes[0] +
                   output_row * kTargetWidth,
               target.data() + source_row * kTargetWidth,
               kTargetWidth) == 0;
      ok = ok &&
           std::memcmp(
               bytes + ready.record_offset_bytes[1] +
                   output_row * kIndexerWidth,
               indexer.data() + source_row * kIndexerWidth,
               kIndexerWidth) == 0;
    }
  }
  ok = ok &&
       api_ok(spark_cache_snapshot_release(snapshot, &ticket), "release");

  SparkCacheSnapshotTicket first{};
  SparkCacheSnapshotTicket second{};
  SparkCacheSnapshotTicket blocked{};
  submission.context_sequence = 78;
  ok = ok && api_ok(
      spark_cache_snapshot_try_submit(
          snapshot,
          &submission,
          slots.data(),
          reinterpret_cast<std::uintptr_t>(stream),
          &first),
      "fill slot zero");
  ok = ok && api_ok(
      spark_cache_snapshot_try_submit(
          snapshot,
          &submission,
          slots.data(),
          reinterpret_cast<std::uintptr_t>(stream),
          &second),
      "fill slot one");
  const auto saturation = spark_cache_snapshot_try_submit(
      snapshot,
      &submission,
      slots.data(),
      reinterpret_cast<std::uintptr_t>(stream),
      &blocked);
  ok = ok && saturation == SPARK_CACHE_SNAPSHOT_WOULD_BLOCK;
  ok = ok && api_ok(
      spark_cache_snapshot_abandon_context(snapshot, 78), "abandon");
  ok = ok && cuda_ok(cudaStreamSynchronize(stream), "drain abandoned");
  SparkCacheSnapshotReadyView ignored{};
  ok = ok &&
       spark_cache_snapshot_poll(snapshot, &first, &ignored) ==
           SPARK_CACHE_SNAPSHOT_DROPPED;
  ok = ok &&
       spark_cache_snapshot_poll(snapshot, &second, &ignored) ==
           SPARK_CACHE_SNAPSHOT_DROPPED;

  SparkCacheSnapshotStats stats{};
  ok = ok && api_ok(
      spark_cache_snapshot_get_stats(snapshot, &stats), "get_stats");
  ok = ok && stats.submissions == 3 && stats.would_block == 1;

  ok = ok && api_ok(
      spark_cache_snapshot_shutdown(snapshot), "shutdown");
  spark_cache_snapshot_destroy(snapshot);
  cudaStreamDestroy(stream);
  cudaFree(device_target);
  cudaFree(device_indexer);
  if (!ok) {
    std::fprintf(stderr, "snapshot probe mismatch\n");
    return 1;
  }
  std::printf(
      "snapshot probe PASS: submissions=%llu would_block=%llu\n",
      static_cast<unsigned long long>(stats.submissions),
      static_cast<unsigned long long>(stats.would_block));
  return 0;
}
