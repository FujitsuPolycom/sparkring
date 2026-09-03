#pragma once

#include <cstdint>
#include <type_traits>

namespace spark_transport::tiled_prefill_research {

enum class TiledExecutorStreamPolicy : std::uint8_t {
  kSingleStreamCorrectnessOnly = 1,
};

// Device-resident descriptor consumed by exactly one tiled bulk-kernel
// launch. The host executor validates its generation, slot-derived storage
// offset, active range, and alignment before an adapter uploads it.
struct TiledBulkDescriptor {
  std::uint64_t input_offset_bytes;
  std::uint64_t output_offset_bytes;
  std::uint64_t slot_storage_offset_bytes;
  std::uint32_t active_bytes;
  std::uint64_t generation;
};

struct EndpointTileStorageLayout {
  std::uint64_t lane_receive_offset;
  std::uint64_t lane_stride;
};

static_assert(std::is_standard_layout_v<TiledBulkDescriptor>);
static_assert(std::is_trivially_copyable_v<TiledBulkDescriptor>);
static_assert(sizeof(TiledBulkDescriptor) == 40);
static_assert(std::is_standard_layout_v<EndpointTileStorageLayout>);
static_assert(std::is_trivially_copyable_v<EndpointTileStorageLayout>);
static_assert(sizeof(EndpointTileStorageLayout) == 16);

}  // namespace spark_transport::tiled_prefill_research
