#pragma once

// Research-only CUDA input generation and correctness checks for the
// standalone tiled-prefill probe. Production transport does not consume this
// interface.

#include <cuda_runtime.h>

#include <cstdint>

#include "tiled_bulk_kernels.cuh"
#include "tiled_correctness_oracle.hpp"

namespace spark_transport::tiled_prefill_research {

// Initializes every field of the qualification ABI receipt. Teardown state is
// deliberately false until the standalone probe has drained its executor.
cudaError_t launch_initialize_correctness_receipt(
    TiledCorrectnessReceipt* receipt, std::uint32_t query_rows,
    std::uint32_t tile_count, std::uint64_t active_payload_bytes,
    std::uint32_t elements_per_row, cudaStream_t stream);

// guarded_input and guarded_output point to the prefix guard. Each allocation
// contains guard_bytes + payload_capacity_bytes + guard_bytes bytes.
cudaError_t launch_fill_correctness_sentinels(
    std::uint8_t* guarded_input, std::uint8_t* guarded_output,
    std::uint64_t guard_bytes, std::uint64_t payload_capacity_bytes,
    cudaStream_t stream);

// input_payload points immediately after the prefix guard. Exactly one
// descriptor's active range is generated, using its physical-slot generation.
cudaError_t launch_fill_correctness_input(
    std::uint8_t* input_payload,
    const TiledBulkDescriptor* descriptor_pointer, std::uint32_t rank,
    cudaStream_t stream);

// output_payload points immediately after the prefix guard. The lower active
// half is checked as xor1-then-xor3 and the upper active half as
// xor3-then-xor1, with BF16 rounding after each association edge.
cudaError_t launch_validate_correctness(
    const std::uint8_t* output_payload,
    const TiledBulkDescriptor* descriptor_pointer,
    TiledCorrectnessReceipt* receipt, cudaStream_t stream);

// Counts prefix/suffix guard corruption and inactive-capacity corruption in
// the exact fields already defined by TiledCorrectnessReceipt.
cudaError_t launch_validate_correctness_sentinels(
    const std::uint8_t* guarded_input,
    const std::uint8_t* guarded_output, std::uint64_t guard_bytes,
    std::uint64_t active_payload_bytes,
    std::uint64_t payload_capacity_bytes,
    TiledCorrectnessReceipt* receipt, cudaStream_t stream);

}  // namespace spark_transport::tiled_prefill_research
