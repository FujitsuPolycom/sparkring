#ifndef SPARK_CACHE_SNAPSHOT_H_
#define SPARK_CACHE_SNAPSHOT_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(SPARK_CACHE_SNAPSHOT_BUILD)
#define SPARK_CACHE_SNAPSHOT_API __declspec(dllexport)
#else
#define SPARK_CACHE_SNAPSHOT_API __declspec(dllimport)
#endif
#else
#define SPARK_CACHE_SNAPSHOT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This is a separate ABI from restore placement. Snapshot publication is
 * opportunistic and fail-open; restore placement remains fail-closed.
 */
#define SPARK_CACHE_SNAPSHOT_ABI_VERSION 1u
#define SPARK_CACHE_SNAPSHOT_MIN_SLOTS 2u
#define SPARK_CACHE_SNAPSHOT_MAX_SLOTS 3u
#define SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS 4u

typedef enum SparkCacheSnapshotStatus {
  SPARK_CACHE_SNAPSHOT_OK = 0,
  SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT = 1,
  SPARK_CACHE_SNAPSHOT_INVALID_STATE = 2,
  SPARK_CACHE_SNAPSHOT_CUDA_ERROR = 3,
  /*
   * Expected fail-open outcomes. The caller must continue inference and may
   * abandon cache publication without retrying on the critical path.
   */
  SPARK_CACHE_SNAPSHOT_WOULD_BLOCK = 4,
  SPARK_CACHE_SNAPSHOT_NOT_READY = 5,
  SPARK_CACHE_SNAPSHOT_DROPPED = 6
} SparkCacheSnapshotStatus;

typedef enum SparkCacheSnapshotArenaMode {
  SPARK_CACHE_SNAPSHOT_MAPPED_HOST = 1,
  SPARK_CACHE_SNAPSHOT_MANAGED = 2
} SparkCacheSnapshotArenaMode;

typedef enum SparkCacheSnapshotSlotState {
  SPARK_CACHE_SNAPSHOT_SLOT_FREE = 0,
  SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING = 1,
  SPARK_CACHE_SNAPSHOT_SLOT_READY = 2,
  SPARK_CACHE_SNAPSHOT_SLOT_WRITING = 3
} SparkCacheSnapshotSlotState;

typedef enum SparkCacheSnapshotRecordKind {
  SPARK_CACHE_SNAPSHOT_TARGET_CKV = 0,
  SPARK_CACHE_SNAPSHOT_SPARSE_INDEXER = 1,
  SPARK_CACHE_SNAPSHOT_MTP_DRAFT_KV = 2,
  SPARK_CACHE_SNAPSHOT_BOUNDARY_HIDDEN = 3
} SparkCacheSnapshotRecordKind;

enum {
  SPARK_CACHE_SNAPSHOT_CAP_MAPPED_HOST = 1u << 0,
  SPARK_CACHE_SNAPSHOT_CAP_MANAGED = 1u << 1,
  SPARK_CACHE_SNAPSHOT_CAP_EXTERNAL_STREAM = 1u << 2,
  SPARK_CACHE_SNAPSHOT_CAP_NONBLOCKING_ACQUIRE = 1u << 3,
  SPARK_CACHE_SNAPSHOT_CAP_CONTEXT_ABANDON = 1u << 4,
  SPARK_CACHE_SNAPSHOT_CAP_ORDERLY_SHUTDOWN = 1u << 5
};

/*
 * The caller owns block leases for all physical slots supplied to submit().
 * It may release those leases after poll() returns READY. The native ring
 * never persists current physical-slot coordinates.
 */
typedef struct SparkCacheSnapshotSource {
  uint64_t source_base;
  uint64_t source_rows;
  uint32_t source_row_stride_bytes;
  uint32_t bytes_per_token;
  uint32_t record_kind;
  uint32_t source_layer_ordinal;
} SparkCacheSnapshotSource;

typedef struct SparkCacheSnapshotConfig {
  uint32_t abi_version;
  uint32_t arena_mode;
  uint64_t slot_bytes;
  uint32_t slot_count;
  uint32_t max_sources;
  uint32_t max_rows;
  int32_t device_ordinal;
  uint32_t flags;
  uint32_t reserved[3];
} SparkCacheSnapshotConfig;

typedef struct SparkCacheSnapshotSubmission {
  uint64_t context_sequence;
  uint64_t logical_start;
  uint32_t row_count;
  uint32_t flags;
  uint32_t reserved[2];
} SparkCacheSnapshotSubmission;

typedef struct SparkCacheSnapshotTicket {
  uint64_t generation;
  uint32_t slot_index;
  uint32_t reserved;
} SparkCacheSnapshotTicket;

/*
 * A READY/WRITING view is immutable until release(). record_* fields describe
 * a layer-major raw payload; the background writer adds the canonical SPCC
 * header and positions without another GPU read.
 */
typedef struct SparkCacheSnapshotReadyView {
  uint64_t host_address;
  uint64_t device_address;
  uint64_t capacity_bytes;
  uint64_t used_bytes;
  uint64_t context_sequence;
  uint64_t logical_start;
  uint64_t generation;
  uint32_t row_count;
  uint32_t slot_index;
  uint32_t record_mask;
  uint32_t state;
  uint32_t record_offset_bytes[SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS];
  uint32_t record_length_bytes[SPARK_CACHE_SNAPSHOT_MAX_RECORD_KINDS];
} SparkCacheSnapshotReadyView;

typedef struct SparkCacheSnapshotStats {
  uint64_t submitted_bytes;
  uint64_t completed_bytes;
  uint64_t released_bytes;
  uint64_t submissions;
  uint64_t claims;
  uint64_t releases;
  uint64_t would_block;
  uint64_t abandoned;
  uint64_t stale_tickets;
  uint64_t reserved[3];
} SparkCacheSnapshotStats;

typedef struct SparkCacheSnapshotAbiInfo {
  uint32_t abi_version;
  uint32_t cudart_version;
  uint32_t min_slots;
  uint32_t max_slots;
  uint32_t max_record_kinds;
  uint32_t sizeof_config;
  uint32_t sizeof_source;
  uint32_t sizeof_submission;
  uint32_t sizeof_ticket;
  uint32_t sizeof_ready_view;
  uint32_t sizeof_stats;
  uint32_t capability_flags;
  uint32_t reserved[4];
} SparkCacheSnapshotAbiInfo;

typedef struct SparkCacheSnapshot SparkCacheSnapshot;

SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_query_abi(SparkCacheSnapshotAbiInfo* output);

SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_create(
    const SparkCacheSnapshotConfig* config,
    SparkCacheSnapshot** output);

SPARK_CACHE_SNAPSHOT_API void spark_cache_snapshot_destroy(
    SparkCacheSnapshot* snapshot);

/*
 * Checked shutdown edge. It refuses while any writer owns a WRITING view.
 * Otherwise it abandons unclaimed bytes, synchronizes GPU_FILLING slots, and
 * leaves every slot FREE. destroy() requires this condition and deliberately
 * leaves the handle allocated rather than invalidating a writer's view.
 */
SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_shutdown(SparkCacheSnapshot* snapshot);

/*
 * Sources are static for a loaded model. All source layers of one record kind
 * must have equal bytes_per_token and dense layer ordinals [0, n).
 */
SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_configure_sources(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotSource* sources,
    uint32_t source_count);

/*
 * Nonblocking producer edge. `producer_stream` is the integer CUDA stream
 * handle on which the source tensors became valid. The gather and completion
 * event are enqueued on that stream, preserving stream ordering.
 *
 * WOULD_BLOCK and DROPPED are ordinary cache misses, never inference errors.
 */
SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_try_submit(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotSubmission* submission,
    const uint32_t* physical_slots,
    uint64_t producer_stream,
    SparkCacheSnapshotTicket* output);

/*
 * poll() never blocks. READY means the GPU no longer reads leased KV blocks.
 * The returned bytes remain immutable until release().
 */
SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_poll(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket* ticket,
    SparkCacheSnapshotReadyView* output);

/*
 * Exactly one writer may claim a READY ticket. The writer may hash, encode,
 * persist, or send these immutable bytes on the diagonal 10GbE sideband.
 */
SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_claim(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket* ticket,
    SparkCacheSnapshotReadyView* output);

SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_release(
    SparkCacheSnapshot* snapshot,
    const SparkCacheSnapshotTicket* ticket);

/*
 * Marks every slot for `context_sequence` as discarded. READY slots are freed
 * immediately, GPU_FILLING slots drain their event before reuse, and WRITING
 * slots free only when their owner calls release().
 */
SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_abandon_context(
    SparkCacheSnapshot* snapshot,
    uint64_t context_sequence);

SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_get_stats(
    SparkCacheSnapshot* snapshot,
    SparkCacheSnapshotStats* output);

SPARK_CACHE_SNAPSHOT_API SparkCacheSnapshotStatus
spark_cache_snapshot_copy_last_error(
    SparkCacheSnapshot* snapshot,
    char* output,
    size_t output_capacity);

SPARK_CACHE_SNAPSHOT_API const char*
spark_cache_snapshot_runtime_last_error(void);

SPARK_CACHE_SNAPSHOT_API const char*
spark_cache_snapshot_status_string(SparkCacheSnapshotStatus status);

#ifdef __cplusplus
}
#endif

#endif  // SPARK_CACHE_SNAPSHOT_H_
