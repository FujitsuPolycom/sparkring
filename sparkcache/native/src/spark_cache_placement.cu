#include "spark_cache_placement.h"
#include "spark_cache_placement_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

using spark_cache::placement::validate_config;
using spark_cache::placement::validate_destinations;
using spark_cache::placement::validate_direct_slab;
using spark_cache::placement::validate_slots;
using spark_cache::placement::validate_transposed_slab;

constexpr std::uint32_t kThreadsPerBlock = 256;
constexpr std::uint32_t kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr std::uint32_t kTransposedRowsPerBlock = 64;
thread_local std::array<char, 512> g_runtime_error{};

enum DeviceError : std::uint32_t {
  kDeviceOk = 0,
  kDeviceChunkBounds = 1,
  kDeviceRecordBounds = 2,
  kDeviceSlotBounds = 3,
  kDeviceDestinationBounds = 4
};

struct ArenaState {
  void* host = nullptr;
  std::uint8_t* device = nullptr;
  SparkCacheChunkDescriptor* host_chunks = nullptr;
  SparkCacheChunkDescriptor* device_chunks = nullptr;
  SparkCacheTransposedSource* host_sources = nullptr;
  SparkCacheTransposedSource* device_sources = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t complete = nullptr;
  bool acquired = false;
  bool in_flight = false;
};

enum class RestoreMode {
  kUnset,
  kDirect,
  kTransposed
};

}  // namespace

struct SparkCachePlacement {
  SparkCachePlacementConfig config{};
  std::array<ArenaState, SPARK_CACHE_PLACEMENT_ARENA_COUNT> arenas{};
  SparkCacheDestinationDescriptor* device_destinations = nullptr;
  std::uint32_t* device_slots = nullptr;
  std::uint32_t* device_error = nullptr;
  std::vector<SparkCacheDestinationDescriptor> destinations;
  std::vector<bool> submitted_destinations;
  std::uint32_t destination_count = 0;
  std::uint32_t slot_count = 0;
  std::uint32_t submitted_rows = 0;
  RestoreMode restore_mode = RestoreMode::kUnset;
  bool restore_active = false;
  SparkCachePlacementStats stats{};
  std::array<char, 512> last_error{};
};

namespace {

void set_error(SparkCachePlacement* placement, const std::string& message) {
  std::snprintf(
      g_runtime_error.data(),
      g_runtime_error.size(),
      "%s",
      message.c_str());
  if (placement != nullptr) {
    std::snprintf(
        placement->last_error.data(),
        placement->last_error.size(),
        "%s",
        message.c_str());
  }
}

SparkCachePlacementStatus cuda_failure(
    SparkCachePlacement* placement,
    const char* operation,
    cudaError_t result) {
  std::string message = operation;
  message += ": ";
  message += cudaGetErrorString(result);
  set_error(placement, message);
  return SPARK_CACHE_PLACEMENT_CUDA_ERROR;
}

__device__ __forceinline__ void set_device_error(
    std::uint32_t* error,
    std::uint32_t value) {
  atomicCAS(
      reinterpret_cast<unsigned int*>(error),
      static_cast<unsigned int>(kDeviceOk),
      static_cast<unsigned int>(value));
}

__device__ void copy_row_bytes(
    const std::uint8_t* source,
    std::uint8_t* destination,
    std::uint32_t bytes,
    std::uint32_t lane) {
  const auto alignment =
      reinterpret_cast<std::uintptr_t>(source) |
      reinterpret_cast<std::uintptr_t>(destination) | bytes;
  if ((alignment & 3U) == 0) {
    const auto* source_words =
        reinterpret_cast<const std::uint32_t*>(source);
    auto* destination_words = reinterpret_cast<std::uint32_t*>(destination);
    const std::uint32_t words = bytes / 4;
    for (std::uint32_t index = lane; index < words; index += 32) {
      destination_words[index] = source_words[index];
    }
    return;
  }
  for (std::uint32_t index = lane; index < bytes; index += 32) {
    destination[index] = source[index];
  }
}

__global__ void scatter_direct_kernel(
    const std::uint8_t* arena,
    std::uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    std::uint32_t chunk_count,
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count,
    const std::uint32_t* slots,
    std::uint32_t slot_count,
    std::uint32_t* device_error) {
  const std::uint32_t chunk_index = blockIdx.x;
  const std::uint32_t destination_index = blockIdx.y;
  if (chunk_index >= chunk_count ||
      destination_index >= destination_count) {
    return;
  }
  const SparkCacheChunkDescriptor chunk = chunks[chunk_index];
  const SparkCacheDestinationDescriptor destination =
      destinations[destination_index];
  const std::uint32_t kind = destination.record_kind;
  const std::uint64_t encoded_end =
      chunk.arena_offset_bytes + chunk.encoded_bytes;
  if (encoded_end < chunk.arena_offset_bytes ||
      encoded_end > arena_used_bytes ||
      chunk.payload_offset_bytes > chunk.encoded_bytes) {
    set_device_error(device_error, kDeviceChunkBounds);
    return;
  }
  if (kind >= SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS ||
      (chunk.record_mask & (1U << kind)) == 0) {
    set_device_error(device_error, kDeviceRecordBounds);
    return;
  }
  const std::uint64_t layer_bytes =
      static_cast<std::uint64_t>(chunk.row_count) *
      destination.bytes_per_token;
  const std::uint64_t source_layer_offset =
      static_cast<std::uint64_t>(destination.source_layer_ordinal) *
      layer_bytes;
  if (source_layer_offset + layer_bytes < source_layer_offset ||
      source_layer_offset + layer_bytes >
          chunk.record_length_bytes[kind]) {
    set_device_error(device_error, kDeviceRecordBounds);
    return;
  }
  const auto* source_layer =
      arena + chunk.arena_offset_bytes + chunk.payload_offset_bytes +
      chunk.record_offset_bytes[kind] + source_layer_offset;
  auto* destination_base = reinterpret_cast<std::uint8_t*>(
      static_cast<std::uintptr_t>(destination.destination_base));
  const std::uint32_t warp = threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x % 32;
  for (std::uint32_t row = warp; row < chunk.row_count;
       row += kWarpsPerBlock) {
    const std::uint64_t slot_index =
        static_cast<std::uint64_t>(chunk.first_slot_index) + row;
    if (slot_index >= slot_count) {
      set_device_error(device_error, kDeviceSlotBounds);
      return;
    }
    const std::uint32_t slot = slots[slot_index];
    if (slot >= destination.destination_rows) {
      set_device_error(device_error, kDeviceDestinationBounds);
      return;
    }
    copy_row_bytes(
        source_layer +
            static_cast<std::uint64_t>(row) *
                destination.bytes_per_token,
        destination_base +
            static_cast<std::uint64_t>(slot) *
                destination.destination_row_stride_bytes,
        destination.bytes_per_token,
        lane);
  }
}

__global__ void scatter_transposed_kernel(
    const std::uint8_t* arena,
    std::uint64_t arena_used_bytes,
    const SparkCacheTransposedSource* sources,
    std::uint32_t source_count,
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count,
    const std::uint32_t* slots,
    std::uint32_t slot_count,
    std::uint32_t* device_error) {
  const std::uint32_t source_index = blockIdx.y;
  if (source_index >= source_count) {
    return;
  }
  const SparkCacheTransposedSource source = sources[source_index];
  if (source.destination_index >= destination_count) {
    set_device_error(device_error, kDeviceDestinationBounds);
    return;
  }
  const SparkCacheDestinationDescriptor destination =
      destinations[source.destination_index];
  const std::uint64_t source_bytes =
      static_cast<std::uint64_t>(slot_count) *
      destination.bytes_per_token;
  if (source.source_offset_bytes + source_bytes <
          source.source_offset_bytes ||
      source.source_offset_bytes + source_bytes > arena_used_bytes) {
    set_device_error(device_error, kDeviceChunkBounds);
    return;
  }
  const std::uint32_t tile_start =
      blockIdx.x * kTransposedRowsPerBlock;
  const std::uint32_t warp = threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x % 32;
  const auto* source_base = arena + source.source_offset_bytes;
  auto* destination_base = reinterpret_cast<std::uint8_t*>(
      static_cast<std::uintptr_t>(destination.destination_base));
  for (std::uint32_t local_row = warp;
       local_row < kTransposedRowsPerBlock;
       local_row += kWarpsPerBlock) {
    const std::uint32_t row = tile_start + local_row;
    if (row >= slot_count) {
      continue;
    }
    const std::uint32_t slot = slots[row];
    if (slot >= destination.destination_rows) {
      set_device_error(device_error, kDeviceDestinationBounds);
      return;
    }
    copy_row_bytes(
        source_base +
            static_cast<std::uint64_t>(row) *
                destination.bytes_per_token,
        destination_base +
            static_cast<std::uint64_t>(slot) *
                destination.destination_row_stride_bytes,
        destination.bytes_per_token,
        lane);
  }
}

void release_arena(ArenaState* arena, std::uint32_t arena_mode) {
  if (arena == nullptr) {
    return;
  }
  if (arena->in_flight && arena->complete != nullptr) {
    (void)cudaEventSynchronize(arena->complete);
  }
  if (arena->complete != nullptr) {
    cudaEventDestroy(arena->complete);
  }
  if (arena->stream != nullptr) {
    cudaStreamDestroy(arena->stream);
  }
  if (arena->host_chunks != nullptr) {
    cudaFreeHost(arena->host_chunks);
  }
  if (arena->host_sources != nullptr) {
    cudaFreeHost(arena->host_sources);
  }
  if (arena_mode == SPARK_CACHE_ARENA_MANAGED) {
    if (arena->device != nullptr) {
      cudaFree(arena->device);
    }
  } else {
    if (arena_mode == SPARK_CACHE_ARENA_STAGED_DEVICE &&
        arena->device != nullptr) {
      cudaFree(arena->device);
    }
    if (arena->host != nullptr) {
      cudaFreeHost(arena->host);
    }
  }
  *arena = ArenaState{};
}

void release_placement(SparkCachePlacement* placement) {
  if (placement == nullptr) {
    return;
  }
  for (auto& arena : placement->arenas) {
    release_arena(&arena, placement->config.arena_mode);
  }
  if (placement->device_destinations != nullptr) {
    cudaFree(placement->device_destinations);
  }
  if (placement->device_slots != nullptr) {
    cudaFree(placement->device_slots);
  }
  if (placement->device_error != nullptr) {
    cudaFree(placement->device_error);
  }
}

SparkCachePlacementStatus wait_arena(
    SparkCachePlacement* placement,
    ArenaState* arena) {
  if (!arena->in_flight) {
    return SPARK_CACHE_PLACEMENT_OK;
  }
  const cudaError_t result = cudaEventSynchronize(arena->complete);
  if (result != cudaSuccess) {
    return cuda_failure(placement, "cudaEventSynchronize", result);
  }
  arena->in_flight = false;
  return SPARK_CACHE_PLACEMENT_OK;
}

SparkCachePlacementStatus prepare_source_arena(
    SparkCachePlacement* placement,
    ArenaState* arena,
    std::uint64_t used_bytes) {
  if (placement->config.arena_mode ==
      SPARK_CACHE_ARENA_STAGED_DEVICE) {
    const cudaError_t result = cudaMemcpyAsync(
        arena->device,
        arena->host,
        static_cast<std::size_t>(used_bytes),
        cudaMemcpyHostToDevice,
        arena->stream);
    if (result != cudaSuccess) {
      return cuda_failure(placement, "cudaMemcpyAsync(arena)", result);
    }
    placement->stats.staged_h2d_bytes += used_bytes;
  } else if (
      placement->config.arena_mode == SPARK_CACHE_ARENA_MANAGED &&
      (placement->config.flags & SPARK_CACHE_CONFIG_PREFETCH_MANAGED) !=
          0) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = placement->config.device_ordinal;
    const cudaError_t result = cudaMemPrefetchAsync(
        arena->device,
        static_cast<std::size_t>(used_bytes),
        location,
        0,
        arena->stream);
#else
    const cudaError_t result = cudaMemPrefetchAsync(
        arena->device,
        static_cast<std::size_t>(used_bytes),
        placement->config.device_ordinal,
        arena->stream);
#endif
    if (result != cudaSuccess) {
      return cuda_failure(placement, "cudaMemPrefetchAsync", result);
    }
  }
  return SPARK_CACHE_PLACEMENT_OK;
}

SparkCachePlacementStatus require_active_arena(
    SparkCachePlacement* placement,
    std::uint32_t arena_index,
    ArenaState** arena) {
  if (placement == nullptr ||
      arena_index >= SPARK_CACHE_PLACEMENT_ARENA_COUNT ||
      arena == nullptr) {
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  if (!placement->restore_active) {
    set_error(placement, "no restore transaction is active");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  *arena = &placement->arenas[arena_index];
  if (!(*arena)->acquired) {
    set_error(placement, "arena must be acquired before submission");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  return SPARK_CACHE_PLACEMENT_OK;
}

}  // namespace

extern "C" SparkCachePlacementStatus spark_cache_placement_query_abi(
    SparkCachePlacementAbiInfo* output) {
  if (output == nullptr) {
    set_error(nullptr, "ABI output pointer is null");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  SparkCachePlacementAbiInfo info{};
  info.abi_version = SPARK_CACHE_PLACEMENT_ABI_VERSION;
  info.cudart_version = CUDART_VERSION;
  info.arena_count = SPARK_CACHE_PLACEMENT_ARENA_COUNT;
  info.max_record_kinds = SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS;
  info.sizeof_config = sizeof(SparkCachePlacementConfig);
  info.sizeof_destination = sizeof(SparkCacheDestinationDescriptor);
  info.sizeof_chunk = sizeof(SparkCacheChunkDescriptor);
  info.sizeof_transposed_source = sizeof(SparkCacheTransposedSource);
  info.sizeof_stats = sizeof(SparkCachePlacementStats);
  info.sizeof_arena_view = sizeof(SparkCacheArenaView);
  info.capability_flags =
      SPARK_CACHE_CAP_MAPPED_HOST |
      SPARK_CACHE_CAP_MANAGED |
      SPARK_CACHE_CAP_STAGED_DEVICE |
      SPARK_CACHE_CAP_DIRECT_ENCODED |
      SPARK_CACHE_CAP_TRANSPOSED |
      SPARK_CACHE_CAP_LOW_PRIORITY_STREAMS;
  *output = info;
  set_error(nullptr, "");
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus spark_cache_placement_create(
    const SparkCachePlacementConfig* config,
    SparkCachePlacement** output) {
  if (config == nullptr || output == nullptr) {
    set_error(nullptr, "config or output pointer is null");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  *output = nullptr;
  std::string detail;
  if (!validate_config(*config, &detail)) {
    set_error(nullptr, detail);
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  auto* placement = new (std::nothrow) SparkCachePlacement();
  if (placement == nullptr) {
    set_error(nullptr, "cannot allocate placement handle");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  placement->config = *config;
  set_error(placement, "");
  cudaError_t result = cudaSetDevice(config->device_ordinal);
  if (result != cudaSuccess) {
    const auto status = cuda_failure(
        placement, "cudaSetDevice", result);
    delete placement;
    return status;
  }
  int lowest_stream_priority = 0;
  int highest_stream_priority = 0;
  result = cudaDeviceGetStreamPriorityRange(
      &lowest_stream_priority, &highest_stream_priority);
  if (result != cudaSuccess) {
    const auto status = cuda_failure(
        placement, "cudaDeviceGetStreamPriorityRange", result);
    delete placement;
    return status;
  }
  (void)highest_stream_priority;

  result = cudaMalloc(
      reinterpret_cast<void**>(&placement->device_destinations),
      static_cast<std::size_t>(config->max_destinations) *
          sizeof(SparkCacheDestinationDescriptor));
  if (result == cudaSuccess) {
    result = cudaMalloc(
        reinterpret_cast<void**>(&placement->device_slots),
        static_cast<std::size_t>(config->max_slots) *
            sizeof(std::uint32_t));
  }
  if (result == cudaSuccess) {
    result = cudaMalloc(
        reinterpret_cast<void**>(&placement->device_error),
        sizeof(std::uint32_t));
  }
  if (result != cudaSuccess) {
    const auto status = cuda_failure(
        placement, "cudaMalloc(control arrays)", result);
    release_placement(placement);
    delete placement;
    return status;
  }

  for (auto& arena : placement->arenas) {
    if (config->arena_mode == SPARK_CACHE_ARENA_MANAGED) {
      result = cudaMallocManaged(
          reinterpret_cast<void**>(&arena.device),
          static_cast<std::size_t>(config->arena_bytes),
          cudaMemAttachGlobal);
      arena.host = arena.device;
    } else {
      const unsigned int host_flags =
          config->arena_mode == SPARK_CACHE_ARENA_MAPPED_HOST
              ? cudaHostAllocMapped | cudaHostAllocPortable
              : cudaHostAllocPortable;
      result = cudaHostAlloc(
          &arena.host,
          static_cast<std::size_t>(config->arena_bytes),
          host_flags);
      if (result == cudaSuccess &&
          config->arena_mode == SPARK_CACHE_ARENA_MAPPED_HOST) {
        result = cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&arena.device), arena.host, 0);
      } else if (
          result == cudaSuccess &&
          config->arena_mode == SPARK_CACHE_ARENA_STAGED_DEVICE) {
        result = cudaMalloc(
            reinterpret_cast<void**>(&arena.device),
            static_cast<std::size_t>(config->arena_bytes));
      }
    }
    if (result == cudaSuccess) {
      result = cudaHostAlloc(
          reinterpret_cast<void**>(&arena.host_chunks),
          static_cast<std::size_t>(config->max_chunks_per_slab) *
              sizeof(SparkCacheChunkDescriptor),
          cudaHostAllocMapped | cudaHostAllocPortable);
    }
    if (result == cudaSuccess) {
      result = cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&arena.device_chunks),
          arena.host_chunks,
          0);
    }
    if (result == cudaSuccess) {
      result = cudaHostAlloc(
          reinterpret_cast<void**>(&arena.host_sources),
          static_cast<std::size_t>(config->max_destinations) *
              sizeof(SparkCacheTransposedSource),
          cudaHostAllocMapped | cudaHostAllocPortable);
    }
    if (result == cudaSuccess) {
      result = cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&arena.device_sources),
          arena.host_sources,
          0);
    }
    if (result == cudaSuccess) {
      result = cudaStreamCreateWithPriority(
          &arena.stream,
          cudaStreamNonBlocking,
          lowest_stream_priority);
    }
    if (result == cudaSuccess) {
      result = cudaEventCreateWithFlags(
          &arena.complete, cudaEventDisableTiming);
    }
    if (result != cudaSuccess) {
      const auto status = cuda_failure(
          placement, "allocate placement arena", result);
      release_placement(placement);
      delete placement;
      return status;
    }
  }
  placement->submitted_destinations.resize(config->max_destinations);
  *output = placement;
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" void spark_cache_placement_destroy(
    SparkCachePlacement* placement) {
  if (placement == nullptr) {
    return;
  }
  release_placement(placement);
  delete placement;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_configure_destinations(
    SparkCachePlacement* placement,
    const SparkCacheDestinationDescriptor* destinations,
    std::uint32_t destination_count) {
  if (placement == nullptr) {
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  if (placement->restore_active) {
    set_error(
        placement, "cannot change destinations during a restore");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  std::string detail;
  if (!validate_destinations(
          destinations,
          destination_count,
          placement->config.max_destinations,
          &detail)) {
    set_error(placement, detail);
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  const cudaError_t result = cudaMemcpy(
      placement->device_destinations,
      destinations,
      static_cast<std::size_t>(destination_count) *
          sizeof(SparkCacheDestinationDescriptor),
      cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    return cuda_failure(
        placement, "cudaMemcpy(destination table)", result);
  }
  placement->destinations.assign(
      destinations, destinations + destination_count);
  placement->destination_count = destination_count;
  placement->stats.destination_table_uploads += 1;
  set_error(placement, "");
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_begin_restore(
    SparkCachePlacement* placement,
    const std::uint32_t* slots,
    std::uint32_t slot_count) {
  if (placement == nullptr || placement->destination_count == 0) {
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  if (placement->restore_active) {
    set_error(placement, "a restore transaction is already active");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  for (auto& arena : placement->arenas) {
    const auto status = wait_arena(placement, &arena);
    if (status != SPARK_CACHE_PLACEMENT_OK) {
      return status;
    }
    arena.acquired = false;
  }
  std::string detail;
  if (!validate_slots(
          slots,
          slot_count,
          placement->config.max_slots,
          placement->destinations.data(),
          placement->destination_count,
          &detail)) {
    set_error(placement, detail);
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  cudaError_t result = cudaMemcpy(
      placement->device_slots,
      slots,
      static_cast<std::size_t>(slot_count) * sizeof(std::uint32_t),
      cudaMemcpyHostToDevice);
  if (result == cudaSuccess) {
    result = cudaMemset(
        placement->device_error, 0, sizeof(std::uint32_t));
  }
  if (result != cudaSuccess) {
    return cuda_failure(
        placement, "initialize restore control arrays", result);
  }
  placement->slot_count = slot_count;
  placement->submitted_rows = 0;
  placement->restore_mode = RestoreMode::kUnset;
  placement->restore_active = true;
  placement->stats = SparkCachePlacementStats{};
  placement->stats.slot_uploads = 1;
  placement->stats.destination_table_uploads = 1;
  std::fill(
      placement->submitted_destinations.begin(),
      placement->submitted_destinations.end(),
      false);
  set_error(placement, "");
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_acquire_arena(
    SparkCachePlacement* placement,
    std::uint32_t arena_index,
    void** host_pointer,
    std::uint64_t* capacity_bytes) {
  if (host_pointer == nullptr || capacity_bytes == nullptr) {
    set_error(placement, "arena output pointer is null");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  SparkCacheArenaView view{};
  const auto status = spark_cache_placement_acquire_arena_view(
      placement, arena_index, &view);
  if (status != SPARK_CACHE_PLACEMENT_OK) {
    return status;
  }
  *host_pointer = reinterpret_cast<void*>(
      static_cast<std::uintptr_t>(view.host_address));
  *capacity_bytes = view.capacity_bytes;
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_acquire_arena_view(
    SparkCachePlacement* placement,
    std::uint32_t arena_index,
    SparkCacheArenaView* output) {
  if (placement == nullptr || output == nullptr ||
      arena_index >= SPARK_CACHE_PLACEMENT_ARENA_COUNT) {
    set_error(placement, "invalid arena-view arguments");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  if (!placement->restore_active) {
    set_error(placement, "no restore transaction is active");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  ArenaState& arena = placement->arenas[arena_index];
  if (arena.acquired) {
    set_error(placement, "arena is already acquired");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  const auto status = wait_arena(placement, &arena);
  if (status != SPARK_CACHE_PLACEMENT_OK) {
    return status;
  }
  arena.acquired = true;
  SparkCacheArenaView view{};
  view.host_address = reinterpret_cast<std::uintptr_t>(arena.host);
  view.device_address =
      reinterpret_cast<std::uintptr_t>(arena.device);
  view.capacity_bytes = placement->config.arena_bytes;
  view.arena_index = arena_index;
  view.arena_mode = placement->config.arena_mode;
  view.flags = SPARK_CACHE_ARENA_VIEW_ACQUIRED;
  if (placement->config.arena_mode ==
      SPARK_CACHE_ARENA_MAPPED_HOST) {
    view.flags |= SPARK_CACHE_ARENA_VIEW_MAPPED_HOST;
  } else if (
      placement->config.arena_mode == SPARK_CACHE_ARENA_MANAGED) {
    view.flags |= SPARK_CACHE_ARENA_VIEW_MANAGED;
  } else {
    view.flags |= SPARK_CACHE_ARENA_VIEW_STAGED_DEVICE;
  }
  *output = view;
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_submit_direct_slab(
    SparkCachePlacement* placement,
    std::uint32_t arena_index,
    std::uint64_t arena_used_bytes,
    const SparkCacheChunkDescriptor* chunks,
    std::uint32_t chunk_count) {
  ArenaState* arena = nullptr;
  const auto required =
      require_active_arena(placement, arena_index, &arena);
  if (required != SPARK_CACHE_PLACEMENT_OK) {
    return required;
  }
  if (placement->restore_mode == RestoreMode::kTransposed) {
    set_error(placement, "cannot mix direct and transposed restore slabs");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  if (arena_used_bytes == 0 ||
      arena_used_bytes > placement->config.arena_bytes) {
    set_error(placement, "direct slab byte count exceeds arena");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  std::uint32_t next = 0;
  std::string detail;
  if (!validate_direct_slab(
          arena_used_bytes,
          chunks,
          chunk_count,
          placement->config.max_chunks_per_slab,
          placement->destinations.data(),
          placement->destination_count,
          placement->slot_count,
          placement->submitted_rows,
          &next,
          &detail)) {
    set_error(placement, detail);
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  std::memcpy(
      arena->host_chunks,
      chunks,
      static_cast<std::size_t>(chunk_count) *
          sizeof(SparkCacheChunkDescriptor));
  std::atomic_thread_fence(std::memory_order_release);
  auto status = prepare_source_arena(
      placement, arena, arena_used_bytes);
  if (status != SPARK_CACHE_PLACEMENT_OK) {
    return status;
  }
  const dim3 grid(chunk_count, placement->destination_count, 1);
  scatter_direct_kernel<<<grid, kThreadsPerBlock, 0, arena->stream>>>(
      arena->device,
      arena_used_bytes,
      arena->device_chunks,
      chunk_count,
      placement->device_destinations,
      placement->destination_count,
      placement->device_slots,
      placement->slot_count,
      placement->device_error);
  cudaError_t result = cudaGetLastError();
  if (result == cudaSuccess) {
    result = cudaEventRecord(arena->complete, arena->stream);
  }
  if (result != cudaSuccess) {
    return cuda_failure(
        placement, "launch direct scatter kernel", result);
  }
  arena->acquired = false;
  arena->in_flight = true;
  placement->restore_mode = RestoreMode::kDirect;
  placement->submitted_rows = next;
  placement->stats.source_bytes += arena_used_bytes;
  placement->stats.restored_rows =
      placement->submitted_rows;
  placement->stats.slabs_submitted += 1;
  placement->stats.scatter_kernel_launches += 1;
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_submit_transposed_slab(
    SparkCachePlacement* placement,
    std::uint32_t arena_index,
    std::uint64_t arena_used_bytes,
    const SparkCacheTransposedSource* sources,
    std::uint32_t source_count) {
  ArenaState* arena = nullptr;
  const auto required =
      require_active_arena(placement, arena_index, &arena);
  if (required != SPARK_CACHE_PLACEMENT_OK) {
    return required;
  }
  if (placement->restore_mode == RestoreMode::kDirect) {
    set_error(placement, "cannot mix transposed and direct restore slabs");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  if (arena_used_bytes == 0 ||
      arena_used_bytes > placement->config.arena_bytes) {
    set_error(placement, "transposed slab byte count exceeds arena");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  std::string detail;
  if (!validate_transposed_slab(
          arena_used_bytes,
          sources,
          source_count,
          placement->destinations.data(),
          placement->destination_count,
          placement->slot_count,
          &detail)) {
    set_error(placement, detail);
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  for (std::uint32_t index = 0; index < source_count; ++index) {
    if (placement->submitted_destinations[
            sources[index].destination_index]) {
      set_error(
          placement,
          "transposed destination was submitted more than once");
      return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
    }
  }
  std::memcpy(
      arena->host_sources,
      sources,
      static_cast<std::size_t>(source_count) *
          sizeof(SparkCacheTransposedSource));
  std::atomic_thread_fence(std::memory_order_release);
  auto status = prepare_source_arena(
      placement, arena, arena_used_bytes);
  if (status != SPARK_CACHE_PLACEMENT_OK) {
    return status;
  }
  const std::uint32_t row_tiles =
      (placement->slot_count + kTransposedRowsPerBlock - 1) /
      kTransposedRowsPerBlock;
  const dim3 grid(row_tiles, source_count, 1);
  scatter_transposed_kernel<<<grid, kThreadsPerBlock, 0, arena->stream>>>(
      arena->device,
      arena_used_bytes,
      arena->device_sources,
      source_count,
      placement->device_destinations,
      placement->destination_count,
      placement->device_slots,
      placement->slot_count,
      placement->device_error);
  cudaError_t result = cudaGetLastError();
  if (result == cudaSuccess) {
    result = cudaEventRecord(arena->complete, arena->stream);
  }
  if (result != cudaSuccess) {
    return cuda_failure(
        placement, "launch transposed scatter kernel", result);
  }
  for (std::uint32_t index = 0; index < source_count; ++index) {
    placement->submitted_destinations[
        sources[index].destination_index] = true;
  }
  arena->acquired = false;
  arena->in_flight = true;
  placement->restore_mode = RestoreMode::kTransposed;
  placement->stats.source_bytes += arena_used_bytes;
  placement->stats.restored_rows = placement->slot_count;
  placement->stats.slabs_submitted += 1;
  placement->stats.scatter_kernel_launches += 1;
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_finish_restore(
    SparkCachePlacement* placement,
    SparkCachePlacementStats* stats) {
  if (placement == nullptr || !placement->restore_active) {
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  if (placement->restore_mode == RestoreMode::kUnset) {
    set_error(placement, "restore contains no placement slabs");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  if (placement->restore_mode == RestoreMode::kDirect &&
      placement->submitted_rows != placement->slot_count) {
    set_error(placement, "direct slabs do not cover every slot row");
    return SPARK_CACHE_PLACEMENT_INVALID_STATE;
  }
  if (placement->restore_mode == RestoreMode::kTransposed) {
    for (std::uint32_t index = 0;
         index < placement->destination_count;
         ++index) {
      if (!placement->submitted_destinations[index]) {
        set_error(
            placement,
            "transposed slabs do not cover every destination");
        return SPARK_CACHE_PLACEMENT_INVALID_STATE;
      }
    }
  }
  for (auto& arena : placement->arenas) {
    const auto status = wait_arena(placement, &arena);
    if (status != SPARK_CACHE_PLACEMENT_OK) {
      return status;
    }
  }
  std::uint32_t device_error = 0;
  const cudaError_t result = cudaMemcpy(
      &device_error,
      placement->device_error,
      sizeof(device_error),
      cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    return cuda_failure(
        placement, "cudaMemcpy(device error)", result);
  }
  placement->stats.device_error = device_error;
  placement->restore_active = false;
  placement->restore_mode = RestoreMode::kUnset;
  if (stats != nullptr) {
    *stats = placement->stats;
  }
  if (device_error != kDeviceOk) {
    set_error(
        placement,
        "device scatter rejected a descriptor or slot bound");
    return SPARK_CACHE_PLACEMENT_DEVICE_ERROR;
  }
  set_error(placement, "");
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_abort_restore(SparkCachePlacement* placement) {
  if (placement == nullptr) {
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  for (auto& arena : placement->arenas) {
    const auto status = wait_arena(placement, &arena);
    if (status != SPARK_CACHE_PLACEMENT_OK) {
      return status;
    }
    arena.acquired = false;
  }
  placement->restore_active = false;
  placement->restore_mode = RestoreMode::kUnset;
  placement->slot_count = 0;
  placement->submitted_rows = 0;
  set_error(placement, "");
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" const char* spark_cache_placement_last_error(
    const SparkCachePlacement* placement) {
  return placement == nullptr ? g_runtime_error.data()
                              : placement->last_error.data();
}

extern "C" SparkCachePlacementStatus spark_cache_placement_get_stats(
    const SparkCachePlacement* placement,
    SparkCachePlacementStats* output) {
  if (placement == nullptr || output == nullptr) {
    set_error(
        const_cast<SparkCachePlacement*>(placement),
        "invalid stats snapshot arguments");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  *output = placement->stats;
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" const char* spark_cache_placement_runtime_last_error(void) {
  return g_runtime_error.data();
}

extern "C" SparkCachePlacementStatus
spark_cache_placement_copy_last_error(
    const SparkCachePlacement* placement,
    char* output,
    std::size_t output_capacity) {
  if (output == nullptr || output_capacity == 0) {
    set_error(
        const_cast<SparkCachePlacement*>(placement),
        "error output buffer is empty");
    return SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT;
  }
  const char* source = placement == nullptr
                           ? g_runtime_error.data()
                           : placement->last_error.data();
  std::snprintf(output, output_capacity, "%s", source);
  return SPARK_CACHE_PLACEMENT_OK;
}

extern "C" const char* spark_cache_placement_status_string(
    SparkCachePlacementStatus status) {
  switch (status) {
    case SPARK_CACHE_PLACEMENT_OK:
      return "ok";
    case SPARK_CACHE_PLACEMENT_INVALID_ARGUMENT:
      return "invalid argument";
    case SPARK_CACHE_PLACEMENT_INVALID_STATE:
      return "invalid state";
    case SPARK_CACHE_PLACEMENT_FORMAT_ERROR:
      return "format error";
    case SPARK_CACHE_PLACEMENT_CUDA_ERROR:
      return "CUDA error";
    case SPARK_CACHE_PLACEMENT_DEVICE_ERROR:
      return "device error";
  }
  return "unknown status";
}
