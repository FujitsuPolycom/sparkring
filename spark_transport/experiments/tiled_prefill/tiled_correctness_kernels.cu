// Research-only CUDA input generation and correctness checks for the
// standalone tiled-prefill probe. This file has no protocol or production
// transport dependency.

#include "tiled_correctness_kernels.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace spark_transport::tiled_prefill_research {
namespace {

constexpr unsigned int kCorrectnessThreads = 256U;
constexpr unsigned int kMaximumCorrectnessBlocks = 1024U;

static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long));
static_assert(
    static_cast<std::uint8_t>(TiledHalfAssociation::kXor1ThenXor3) ==
    static_cast<std::uint8_t>(TiledOracleHalf::kLowerXor1ThenXor3));
static_assert(
    static_cast<std::uint8_t>(TiledHalfAssociation::kXor3ThenXor1) ==
    static_cast<std::uint8_t>(TiledOracleHalf::kUpperXor3ThenXor1));

unsigned int correctness_blocks(std::uint64_t items) {
  const std::uint64_t requested =
      (items + kCorrectnessThreads - 1U) / kCorrectnessThreads;
  return static_cast<unsigned int>(
      requested < kMaximumCorrectnessBlocks
          ? requested
          : kMaximumCorrectnessBlocks);
}

bool valid_guarded_geometry(std::uint64_t guard_bytes,
                            std::uint64_t active_payload_bytes,
                            std::uint64_t payload_capacity_bytes) {
  return guard_bytes != 0U && active_payload_bytes != 0U &&
         active_payload_bytes <= payload_capacity_bytes &&
         payload_capacity_bytes % kOraclePayloadTileBytes == 0U &&
         guard_bytes <=
             (std::numeric_limits<std::uint64_t>::max() -
              payload_capacity_bytes) /
                 2U;
}

__device__ void increment(std::uint64_t* counter) {
  atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ULL);
}

__device__ bool valid_descriptor(const TiledBulkDescriptor& descriptor) {
  return descriptor.active_bytes != 0U &&
         descriptor.active_bytes <= kOraclePayloadTileBytes &&
         descriptor.active_bytes % (2U * kOracleBf16Bytes) == 0U &&
         descriptor.input_offset_bytes % kOracleBf16Bytes == 0U &&
         descriptor.output_offset_bytes % kOracleBf16Bytes == 0U &&
         descriptor.generation != 0U;
}

__global__ void initialize_correctness_receipt(
    TiledCorrectnessReceipt* receipt, std::uint32_t query_rows,
    std::uint32_t tile_count, std::uint64_t active_payload_bytes) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  TiledCorrectnessReceipt initialized{};
  initialized.query_rows = query_rows;
  initialized.tile_count = tile_count;
  initialized.active_payload_bytes = active_payload_bytes;
  initialized.lower_half_association =
      TiledHalfAssociation::kXor1ThenXor3;
  initialized.upper_half_association =
      TiledHalfAssociation::kXor3ThenXor1;
  initialized.executor_stream_policy =
      TiledExecutorStreamPolicy::kSingleStreamCorrectnessOnly;
  initialized.teardown_drained = false;
  initialized.performance_claim_allowed = false;
  receipt[0] = initialized;
}

__global__ void fill_correctness_sentinels(
    std::uint8_t* guarded_input, std::uint8_t* guarded_output,
    std::uint64_t guard_bytes, std::uint64_t payload_capacity_bytes) {
  const std::uint64_t allocation_bytes =
      payload_capacity_bytes + 2U * guard_bytes;
  for (std::uint64_t index =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       index < allocation_bytes;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const bool guard =
        index < guard_bytes || index >= guard_bytes + payload_capacity_bytes;
    guarded_input[index] =
        guard ? kInputGuardSentinel : kInactiveInputSentinel;
    guarded_output[index] =
        guard ? kOutputGuardSentinel : kInactiveOutputSentinel;
  }
}

__global__ void fill_correctness_input(
    std::uint8_t* input_payload,
    const TiledBulkDescriptor* descriptor_pointer, std::uint32_t rank) {
  const TiledBulkDescriptor descriptor = descriptor_pointer[0];
  if (!valid_descriptor(descriptor)) {
    return;
  }
  const std::uint64_t active_elements =
      descriptor.active_bytes / kOracleBf16Bytes;
  const std::uint64_t first_element =
      descriptor.input_offset_bytes / kOracleBf16Bytes;
  auto* destination = reinterpret_cast<std::uint16_t*>(
      input_payload + descriptor.input_offset_bytes);
  for (std::uint64_t local_element =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       local_element < active_elements;
       local_element +=
       static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    destination[local_element] = input_bf16_bits(
        rank, first_element + local_element, descriptor.generation);
  }
}

__global__ void validate_correctness(
    const std::uint8_t* output_payload,
    const TiledBulkDescriptor* descriptor_pointer,
    TiledCorrectnessReceipt* receipt) {
  const TiledBulkDescriptor descriptor = descriptor_pointer[0];
  if (!valid_descriptor(descriptor)) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      increment(&receipt->mismatched_active_elements);
    }
    return;
  }
  const std::uint64_t active_elements =
      descriptor.active_bytes / kOracleBf16Bytes;
  const std::uint64_t lower_elements = active_elements / 2U;
  const std::uint64_t first_element =
      descriptor.input_offset_bytes / kOracleBf16Bytes;
  const auto* actual = reinterpret_cast<const std::uint16_t*>(
      output_payload + descriptor.output_offset_bytes);
  for (std::uint64_t local_element =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       local_element < active_elements;
       local_element +=
       static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const TiledOracleHalf half =
        local_element < lower_elements
            ? TiledOracleHalf::kLowerXor1ThenXor3
            : TiledOracleHalf::kUpperXor3ThenXor1;
    const std::uint16_t expected = expected_output_bf16_bits(
        first_element + local_element, descriptor.generation, half);
    if (actual[local_element] != expected) {
      increment(&receipt->mismatched_active_elements);
    }
  }
}

__global__ void validate_correctness_sentinels(
    const std::uint8_t* guarded_input,
    const std::uint8_t* guarded_output, std::uint64_t guard_bytes,
    std::uint64_t active_payload_bytes,
    std::uint64_t payload_capacity_bytes,
    TiledCorrectnessReceipt* receipt) {
  const std::uint64_t payload_begin = guard_bytes;
  const std::uint64_t inactive_begin =
      payload_begin + active_payload_bytes;
  const std::uint64_t payload_end =
      payload_begin + payload_capacity_bytes;
  const std::uint64_t allocation_bytes = payload_end + guard_bytes;
  for (std::uint64_t index =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       index < allocation_bytes;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (index < payload_begin || index >= payload_end) {
      if (guarded_input[index] != kInputGuardSentinel) {
        increment(&receipt->input_guard_corruptions);
      }
      if (guarded_output[index] != kOutputGuardSentinel) {
        increment(&receipt->output_guard_corruptions);
      }
    } else if (index >= inactive_begin) {
      if (guarded_input[index] != kInactiveInputSentinel) {
        increment(&receipt->inactive_input_sentinel_corruptions);
      }
      if (guarded_output[index] != kInactiveOutputSentinel) {
        increment(&receipt->inactive_output_sentinel_corruptions);
      }
    }
  }
}

}  // namespace

cudaError_t launch_initialize_correctness_receipt(
    TiledCorrectnessReceipt* receipt, std::uint32_t query_rows,
    std::uint32_t tile_count, std::uint64_t active_payload_bytes,
    std::uint32_t elements_per_row, cudaStream_t stream) {
  const TiledPayloadGeometry geometry =
      oracle_payload_geometry(query_rows, elements_per_row);
  const std::uint64_t expected_tile_count =
      geometry.capacity_bytes / kOraclePayloadTileBytes;
  if (receipt == nullptr || query_rows == 0U || elements_per_row == 0U ||
      tile_count != expected_tile_count ||
      active_payload_bytes != geometry.active_bytes) {
    return cudaErrorInvalidValue;
  }
  initialize_correctness_receipt<<<1U, 1U, 0U, stream>>>(
      receipt, query_rows, tile_count, active_payload_bytes);
  return cudaGetLastError();
}

cudaError_t launch_fill_correctness_sentinels(
    std::uint8_t* guarded_input, std::uint8_t* guarded_output,
    std::uint64_t guard_bytes, std::uint64_t payload_capacity_bytes,
    cudaStream_t stream) {
  if (guarded_input == nullptr || guarded_output == nullptr ||
      !valid_guarded_geometry(
          guard_bytes, payload_capacity_bytes, payload_capacity_bytes)) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t allocation_bytes =
      payload_capacity_bytes + 2U * guard_bytes;
  fill_correctness_sentinels<<<
      correctness_blocks(allocation_bytes), kCorrectnessThreads, 0U, stream>>>(
      guarded_input, guarded_output, guard_bytes, payload_capacity_bytes);
  return cudaGetLastError();
}

cudaError_t launch_fill_correctness_input(
    std::uint8_t* input_payload,
    const TiledBulkDescriptor* descriptor_pointer, std::uint32_t rank,
    cudaStream_t stream) {
  if (input_payload == nullptr || descriptor_pointer == nullptr || rank >= 4U) {
    return cudaErrorInvalidValue;
  }
  constexpr std::uint64_t kMaximumElements =
      kOraclePayloadTileBytes / kOracleBf16Bytes;
  fill_correctness_input<<<
      correctness_blocks(kMaximumElements), kCorrectnessThreads, 0U, stream>>>(
      input_payload, descriptor_pointer, rank);
  return cudaGetLastError();
}

cudaError_t launch_validate_correctness(
    const std::uint8_t* output_payload,
    const TiledBulkDescriptor* descriptor_pointer,
    TiledCorrectnessReceipt* receipt, cudaStream_t stream) {
  if (output_payload == nullptr || descriptor_pointer == nullptr ||
      receipt == nullptr) {
    return cudaErrorInvalidValue;
  }
  constexpr std::uint64_t kMaximumElements =
      kOraclePayloadTileBytes / kOracleBf16Bytes;
  validate_correctness<<<
      correctness_blocks(kMaximumElements), kCorrectnessThreads, 0U, stream>>>(
      output_payload, descriptor_pointer, receipt);
  return cudaGetLastError();
}

cudaError_t launch_validate_correctness_sentinels(
    const std::uint8_t* guarded_input,
    const std::uint8_t* guarded_output, std::uint64_t guard_bytes,
    std::uint64_t active_payload_bytes,
    std::uint64_t payload_capacity_bytes,
    TiledCorrectnessReceipt* receipt, cudaStream_t stream) {
  if (guarded_input == nullptr || guarded_output == nullptr ||
      receipt == nullptr ||
      !valid_guarded_geometry(
          guard_bytes, active_payload_bytes, payload_capacity_bytes)) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t allocation_bytes =
      payload_capacity_bytes + 2U * guard_bytes;
  validate_correctness_sentinels<<<
      correctness_blocks(allocation_bytes), kCorrectnessThreads, 0U, stream>>>(
      guarded_input, guarded_output, guard_bytes, active_payload_bytes,
      payload_capacity_bytes, receipt);
  return cudaGetLastError();
}

}  // namespace spark_transport::tiled_prefill_research
