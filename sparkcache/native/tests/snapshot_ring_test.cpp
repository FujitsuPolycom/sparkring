#include "spark_cache_snapshot_ring.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <string>

namespace {

using spark_cache::snapshot::RingState;
using spark_cache::snapshot::calculate_payload_layout;
using spark_cache::snapshot::validate_config;
using spark_cache::snapshot::validate_sources;

static_assert(sizeof(SparkCacheSnapshotConfig) == 48);
static_assert(sizeof(SparkCacheSnapshotSource) == 32);
static_assert(sizeof(SparkCacheSnapshotSubmission) == 32);
static_assert(sizeof(SparkCacheSnapshotTicket) == 16);
static_assert(sizeof(SparkCacheSnapshotReadyView) == 104);
static_assert(sizeof(SparkCacheSnapshotStats) == 96);
static_assert(sizeof(SparkCacheSnapshotAbiInfo) == 64);

SparkCacheSnapshotConfig config(std::uint32_t slots) {
  SparkCacheSnapshotConfig result{};
  result.abi_version = SPARK_CACHE_SNAPSHOT_ABI_VERSION;
  result.arena_mode = SPARK_CACHE_SNAPSHOT_MAPPED_HOST;
  result.slot_bytes = 32ULL * 1024ULL * 1024ULL;
  result.slot_count = slots;
  result.max_sources = 128;
  result.max_rows = 256;
  result.device_ordinal = 0;
  return result;
}

std::array<SparkCacheSnapshotSource, 4> sources() {
  return {{
      {0x1000, 4096, 512, 368, SPARK_CACHE_SNAPSHOT_TARGET_CKV, 0},
      {0x2000, 4096, 512, 368, SPARK_CACHE_SNAPSHOT_TARGET_CKV, 1},
      {0x3000, 4096, 256, 132, SPARK_CACHE_SNAPSHOT_SPARSE_INDEXER, 0},
      {0x4000, 4096, 512, 368, SPARK_CACHE_SNAPSHOT_MTP_DRAFT_KV, 0},
  }};
}

void test_validation_and_layout() {
  std::string detail;
  auto value = config(3);
  assert(validate_config(value, &detail));
  value.slot_count = 4;
  assert(!validate_config(value, &detail));

  const auto table = sources();
  assert(validate_sources(
      table.data(), table.size(), table.size(), &detail));
  SparkCacheSnapshotReadyView layout{};
  assert(calculate_payload_layout(
      table.data(),
      table.size(),
      64,
      32ULL * 1024ULL * 1024ULL,
      &layout,
      &detail));
  assert(layout.record_mask == 0b111);
  assert(layout.record_offset_bytes[0] == 0);
  assert(layout.record_length_bytes[0] == 2 * 64 * 368);
  assert(layout.record_offset_bytes[1] % 64 == 0);
  assert(layout.record_length_bytes[1] == 64 * 132);
  assert(layout.record_offset_bytes[2] % 64 == 0);
  assert(layout.record_length_bytes[2] == 64 * 368);
  assert(layout.used_bytes <= layout.capacity_bytes);

  auto bad = table;
  bad[1].source_layer_ordinal = 2;
  assert(!validate_sources(
      bad.data(), bad.size(), bad.size(), &detail));
}

void test_ring_never_blocks_and_rejects_stale_tickets() {
  RingState ring(2);
  assert(ring.valid());
  SparkCacheSnapshotTicket first{};
  SparkCacheSnapshotTicket second{};
  SparkCacheSnapshotTicket overflow{};
  assert(ring.reserve(11, 4096, &first) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(11, 4096, &second) == SPARK_CACHE_SNAPSHOT_OK);
  assert(
      ring.reserve(11, 4096, &overflow) ==
      SPARK_CACHE_SNAPSHOT_WOULD_BLOCK);
  assert(ring.complete(first) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.claim(first) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.release(first) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.complete(first) == SPARK_CACHE_SNAPSHOT_DROPPED);

  SparkCacheSnapshotTicket replacement{};
  assert(
      ring.reserve(12, 2048, &replacement) ==
      SPARK_CACHE_SNAPSHOT_OK);
  assert(replacement.slot_index == first.slot_index);
  assert(replacement.generation != first.generation);
}

void test_abandon_drains_gpu_and_writer_ownership() {
  RingState ring(3);
  SparkCacheSnapshotTicket filling{};
  SparkCacheSnapshotTicket ready{};
  SparkCacheSnapshotTicket writing{};
  assert(ring.reserve(50, 1, &filling) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(50, 1, &ready) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(50, 1, &writing) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.complete(ready) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.complete(writing) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.claim(writing) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.abandon(50) == 3);

  assert(ring.complete(filling) == SPARK_CACHE_SNAPSHOT_DROPPED);
  assert(ring.claim(ready) == SPARK_CACHE_SNAPSHOT_DROPPED);
  assert(ring.release(writing) == SPARK_CACHE_SNAPSHOT_OK);

  SparkCacheSnapshotTicket one{};
  SparkCacheSnapshotTicket two{};
  SparkCacheSnapshotTicket three{};
  assert(ring.reserve(51, 1, &one) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(51, 1, &two) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(51, 1, &three) == SPARK_CACHE_SNAPSHOT_OK);
}

void test_internal_reaper_preserves_generation_and_writer_ownership() {
  RingState ring(2);
  SparkCacheSnapshotTicket abandoned{};
  SparkCacheSnapshotTicket writer{};
  SparkCacheSnapshotTicket blocked{};
  assert(ring.reserve(60, 10, &abandoned) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(61, 20, &writer) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.complete(writer) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.claim(writer) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.abandon(60) == 1);
  assert(
      ring.reserve(62, 30, &blocked) ==
      SPARK_CACHE_SNAPSHOT_WOULD_BLOCK);
  assert(
      ring.reap_discarded(abandoned.slot_index) ==
      SPARK_CACHE_SNAPSHOT_OK);

  SparkCacheSnapshotTicket replacement{};
  assert(
      ring.reserve(62, 30, &replacement) ==
      SPARK_CACHE_SNAPSHOT_OK);
  assert(replacement.slot_index == abandoned.slot_index);
  assert(replacement.generation != abandoned.generation);
  assert(ring.complete(abandoned) == SPARK_CACHE_SNAPSHOT_DROPPED);
  assert(
      ring.reap_discarded(writer.slot_index) ==
      SPARK_CACHE_SNAPSHOT_INVALID_STATE);
}

void test_shutdown_refuses_writer_then_drains_abandoned_gpu_work() {
  RingState ring(3);
  SparkCacheSnapshotTicket filling{};
  SparkCacheSnapshotTicket writer{};
  assert(ring.reserve(70, 10, &filling) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(70, 10, &writer) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.complete(writer) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.claim(writer) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.has_writing());
  assert(ring.release(writer) == SPARK_CACHE_SNAPSHOT_OK);
  assert(!ring.has_writing());
  assert(ring.abandon_all() == 1);
  assert(
      ring.reap_discarded(filling.slot_index) ==
      SPARK_CACHE_SNAPSHOT_OK);
  assert(
      ring.inspect(filling.slot_index)->state ==
      SPARK_CACHE_SNAPSHOT_SLOT_FREE);
}

void test_post_launch_failure_stays_quarantined_until_explicit_drain() {
  RingState ring(2);
  SparkCacheSnapshotTicket launched{};
  SparkCacheSnapshotTicket other{};
  SparkCacheSnapshotTicket blocked{};
  assert(ring.reserve(80, 10, &launched) == SPARK_CACHE_SNAPSHOT_OK);
  assert(ring.reserve(81, 10, &other) == SPARK_CACHE_SNAPSHOT_OK);

  // Models a post-launch failure: abandon marks the in-flight slot for
  // discard, but it remains occupied until CUDA stream drain is proven.
  assert(ring.abandon(80) == 1);
  const auto* quarantined = ring.inspect(launched.slot_index);
  assert(quarantined != nullptr);
  assert(
      quarantined->state == SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING);
  assert(quarantined->discard);
  assert(
      ring.reserve(82, 10, &blocked) ==
      SPARK_CACHE_SNAPSHOT_WOULD_BLOCK);

  // The CUDA owner invokes this only after its stream synchronization seam
  // succeeds. Reuse before this transition is impossible.
  assert(
      ring.reap_discarded(launched.slot_index) ==
      SPARK_CACHE_SNAPSHOT_OK);
  SparkCacheSnapshotTicket replacement{};
  assert(
      ring.reserve(82, 10, &replacement) ==
      SPARK_CACHE_SNAPSHOT_OK);
  assert(replacement.slot_index == launched.slot_index);
  assert(replacement.generation != launched.generation);
}

}  // namespace

int main() {
  test_validation_and_layout();
  test_ring_never_blocks_and_rejects_stale_tickets();
  test_abandon_drains_gpu_and_writer_ownership();
  test_internal_reaper_preserves_generation_and_writer_ownership();
  test_shutdown_refuses_writer_then_drains_abandoned_gpu_work();
  test_post_launch_failure_stays_quarantined_until_explicit_drain();
  return 0;
}
