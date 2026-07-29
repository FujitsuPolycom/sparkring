#ifndef SPARK_CACHE_PLACEMENT_H_
#define SPARK_CACHE_PLACEMENT_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(SPARK_CACHE_PLACEMENT_BUILD)
#define SPARK_CACHE_API __declspec(dllexport)
#else
#define SPARK_CACHE_API __declspec(dllimport)
#endif
#else
#define SPARK_CACHE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_CACHE_PLACEMENT_ABI_VERSION 1u
#define SPARK_CACHE_PLACEMENT_ARENA_COUNT 2u
#define SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS 4u

typedef enum SparkCachePlacementStatus {
  SPARK_CACHE_PLACEMENT_OK = 0,
  SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT = 1,
  SPARK_CACHE_PLACEMENT_INVALID_STATE = 2,
  SPARK_CACHE_PLACEMENT_FORMAT_ERROR = 3,
  SPARK_CACHE_PLACEMENT_CUDA_ERROR = 4,
  SPARK_CACHE_PLACEMENT_DEVICE_ERROR = 5
} SparkCachePlacementStatus;

typedef enum SparkCacheArenaMode {
  /*
   * cudaHostAllocMapped(): CPU pread/hash and the GPU scatter kernel touch the
   * same physical bytes. This is the preferred DGX Spark UMA experiment.
   */
  SPARK_CACHE_ARENA_MAPPED_HOST = 1,
  /*
   * cudaMallocManaged(): useful as an A/B, but page-fault/prefetch behavior
   * must be measured rather than assumed to beat mapped host memory.
   */
  SPARK_CACHE_ARENA_MANAGED = 2,
  /*
   * Pinned host producer arena plus a device arena and one H2D per slab.
   * This is the conservative discrete-GPU fallback.
   */
  SPARK_CACHE_ARENA_STAGED_DEVICE = 3
} SparkCacheArenaMode;

typedef enum SparkCacheRecordKind {
  SPARK_CACHE_RECORD_TARGET_CKV = 0,
  SPARK_CACHE_RECORD_SPARSE_INDEXER = 1,
  SPARK_CACHE_RECORD_MTP_DRAFT_KV = 2,
  SPARK_CACHE_RECORD_BOUNDARY_HIDDEN = 3
} SparkCacheRecordKind;

enum {
  SPARK_CACHE_CONFIG_PREFETCH_MANAGED = 1u << 0
};

enum {
  SPARK_CACHE_CAP_MAPPED_HOST = 1u << 0,
  SPARK_CACHE_CAP_MANAGED = 1u << 1,
  SPARK_CACHE_CAP_STAGED_DEVICE = 1u << 2,
  SPARK_CACHE_CAP_DIRECT_ENCODED = 1u << 3,
  SPARK_CACHE_CAP_TRANSPOSED = 1u << 4,
  SPARK_CACHE_CAP_LOW_PRIORITY_STREAMS = 1u << 5
};

enum {
  SPARK_CACHE_ARENA_VIEW_ACQUIRED = 1u << 0,
  SPARK_CACHE_ARENA_VIEW_MAPPED_HOST = 1u << 1,
  SPARK_CACHE_ARENA_VIEW_MANAGED = 1u << 2,
  SPARK_CACHE_ARENA_VIEW_STAGED_DEVICE = 1u << 3
};

/*
 * One stable destination entry per registered cache tensor. The table is
 * uploaded once after register_kv_caches(), not once per restored context.
 *
 * The source record is layer-major:
 *   [layer_ordinal][row_in_chunk][bytes_per_token].
 */
typedef struct SparkCacheDestinationDescriptor {
  uint64_t destination_base;
  uint64_t destination_rows;
  uint32_t destination_row_stride_bytes;
  uint32_t bytes_per_token;
  uint32_t record_kind;
  uint32_t source_layer_ordinal;
} SparkCacheDestinationDescriptor;

/*
 * One descriptor for an integrity-verified encoded .spcc chunk residing in
 * an arena. Offsets are relative to arena_base or the encoded payload.
 *
 * The descriptor deliberately contains no persisted physical KV slot. Slots
 * are supplied separately from the current request's allocator.
 */
typedef struct SparkCacheChunkDescriptor {
  uint64_t arena_offset_bytes;
  uint32_t encoded_bytes;
  uint32_t payload_offset_bytes;
  uint32_t first_slot_index;
  uint32_t row_count;
  uint32_t record_mask;
  uint32_t flags;
  uint32_t record_offset_bytes[SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS];
  uint32_t record_length_bytes[SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS];
} SparkCacheChunkDescriptor;

/*
 * Fallback input for the existing vectorized Python/NumPy slab assembler.
 * Each source is a layer-major [slot_count, bytes_per_token] byte matrix.
 */
typedef struct SparkCacheTransposedSource {
  uint64_t source_offset_bytes;
  uint32_t destination_index;
  uint32_t flags;
} SparkCacheTransposedSource;

typedef struct SparkCachePlacementConfig {
  uint32_t abi_version;
  uint32_t arena_mode;
  uint64_t arena_bytes;
  uint32_t max_destinations;
  uint32_t max_slots;
  uint32_t max_chunks_per_slab;
  int32_t device_ordinal;
  uint32_t flags;
  uint32_t reserved[3];
} SparkCachePlacementConfig;

typedef struct SparkCachePlacementStats {
  uint64_t source_bytes;
  uint64_t staged_h2d_bytes;
  uint64_t restored_rows;
  uint32_t slot_uploads;
  uint32_t destination_table_uploads;
  uint32_t slabs_submitted;
  uint32_t scatter_kernel_launches;
  uint32_t device_error;
  uint32_t reserved[3];
} SparkCachePlacementStats;

/*
 * Runtime-owned ABI fingerprint. A ctypes caller must compare every reported
 * sizeof value with its local Structure before passing any descriptor.
 */
typedef struct SparkCachePlacementAbiInfo {
  uint32_t abi_version;
  uint32_t cudart_version;
  uint32_t arena_count;
  uint32_t max_record_kinds;
  uint32_t sizeof_config;
  uint32_t sizeof_destination;
  uint32_t sizeof_chunk;
  uint32_t sizeof_transposed_source;
  uint32_t sizeof_stats;
  uint32_t sizeof_arena_view;
  uint32_t capability_flags;
  uint32_t reserved[5];
} SparkCachePlacementAbiInfo;

/*
 * Integer addresses make the arena ABI unambiguous for ctypes. `host_address`
 * is suitable for `(ctypes.c_ubyte * capacity).from_address(...)` followed by
 * a writable `memoryview(...).cast("B")` passed to os.preadv().
 */
typedef struct SparkCacheArenaView {
  uint64_t host_address;
  uint64_t device_address;
  uint64_t capacity_bytes;
  uint32_t arena_index;
  uint32_t arena_mode;
  uint32_t flags;
  uint32_t reserved;
} SparkCacheArenaView;

typedef struct SparkCachePlacement SparkCachePlacement;

SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_query_abi(
    SparkCachePlacementAbiInfo* output);

/*
 * Parse only after the caller has verified the manifest's outer SHA-256 over
 * the complete encoded chunk. This parser validates the v1 prefix, canonical
 * header, record bounds/order, required records, and every logical position.
 * It never allocates or copies record payloads.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_parse_verified_v1_chunk(
    const void* arena_base,
    uint64_t arena_used_bytes,
    uint64_t arena_offset_bytes,
    uint32_t encoded_bytes,
    uint32_t expected_logical_start,
    uint32_t dcp_degree,
    uint32_t dcp_rank,
    uint32_t first_slot_index,
    uint32_t required_data_record_mask,
    SparkCacheChunkDescriptor* output,
    char* error,
    size_t error_capacity);

/*
 * CPU byte-for-byte oracle for tests. Destination bases must be host pointers
 * when calling this function. Production GPU destinations must use the CUDA
 * placement API below.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_reference_scatter_direct(
    const void* arena_base,
    uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    uint32_t chunk_count,
    const SparkCacheDestinationDescriptor* destinations,
    uint32_t destination_count,
    const uint32_t* slots,
    uint32_t slot_count,
    char* error,
    size_t error_capacity);

SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_create(
    const SparkCachePlacementConfig* config,
    SparkCachePlacement** output);

SPARK_CACHE_API void spark_cache_placement_destroy(
    SparkCachePlacement* placement);

SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_configure_destinations(
    SparkCachePlacement* placement,
    const SparkCacheDestinationDescriptor* destinations,
    uint32_t destination_count);

/*
 * Begins one restore transaction and uploads the remapped physical slot vector
 * exactly once. All slots must be unique and in range for every destination.
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_begin_restore(
    SparkCachePlacement* placement,
    const uint32_t* slots,
    uint32_t slot_count);

/*
 * Waits until an arena is no longer read by a prior kernel, then returns the
 * CPU producer address. The caller may pread directly into this allocation.
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_acquire_arena(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    void** host_pointer,
    uint64_t* capacity_bytes);

/*
 * ctypes-preferred equivalent of acquire_arena(). Both calls acquire the
 * arena; call exactly one of them for a given fill/submit cycle.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_acquire_arena_view(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    SparkCacheArenaView* output);

/*
 * Direct path: scatter all configured destinations from verified encoded
 * chunks. Chunk row ranges must be contiguous, non-overlapping, and cover the
 * slot vector exactly across all submitted slabs.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_submit_direct_slab(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    uint32_t chunk_count);

/*
 * Fallback path: scatter a subset of already-transposed layer slabs. Every
 * destination must appear exactly once across all submitted slabs.
 */
SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_submit_transposed_slab(
    SparkCachePlacement* placement,
    uint32_t arena_index,
    uint64_t arena_used_bytes,
    const SparkCacheTransposedSource* sources,
    uint32_t source_count);

/*
 * Synchronizes both arena streams, verifies complete coverage and the device
 * error word, and only then makes the restored request eligible to resume.
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_finish_restore(
    SparkCachePlacement* placement,
    SparkCachePlacementStats* stats);

/*
 * A CUDA kernel cannot be safely cancelled. Abort waits for submitted work,
 * discards the transaction, and leaves the request's parked KV blocks for the
 * caller to free/recompute. No consumer may observe them before finish().
 */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_abort_restore(
    SparkCachePlacement* placement);

/* Snapshot is available before, during, or after a restore. */
SPARK_CACHE_API SparkCachePlacementStatus spark_cache_placement_get_stats(
    const SparkCachePlacement* placement,
    SparkCachePlacementStats* output);

SPARK_CACHE_API const char* spark_cache_placement_last_error(
    const SparkCachePlacement* placement);

/*
 * The runtime error is thread-local and remains useful when create() failed
 * before returning a handle. Prefer copy_last_error for exception-safe ctypes
 * wrappers that do not retain a borrowed C string.
 */
SPARK_CACHE_API const char* spark_cache_placement_runtime_last_error(void);

SPARK_CACHE_API SparkCachePlacementStatus
spark_cache_placement_copy_last_error(
    const SparkCachePlacement* placement,
    char* output,
    size_t output_capacity);

SPARK_CACHE_API const char* spark_cache_placement_status_string(
    SparkCachePlacementStatus status);

#ifdef __cplusplus
}
#endif

#endif  // SPARK_CACHE_PLACEMENT_H_
