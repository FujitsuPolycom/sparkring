#pragma once

// Research-only interface for bounded tiled TP4 prefill bulk kernels.
// Production transport code does not include this header.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "tiled_bulk_abi.hpp"

namespace spark_transport::tiled_prefill_research {

// Diagnostic envelopes may overlap when independent tiles pipeline. They are
// not additive components of device_output_ready_us.
struct TiledTimingReceipt {
  std::uint64_t isolated_warmup_operations;
  std::uint64_t isolated_measured_operations;
  std::uint64_t steady_state_warmup_operations;
  std::uint64_t steady_state_measured_operations;
  double host_submit_us_per_operation;
  double device_output_ready_us_min;
  double device_output_ready_us_p50;
  double device_output_ready_us_p95;
  double steady_state_device_us_per_operation;
  double device_fully_retired_us_p50;
  double device_fully_retired_us_p95;
  double slot_credit_wait_us_p50;
  double slot_credit_wait_us_p95;
  double stage_phase1_gpu_us_p50;
  double phase1_remote_wait_us_p50;
  double reduce_phase1_gpu_us_p50;
  double phase2_remote_wait_us_p50;
  double reduce_phase2_gpu_us_p50;
  double active_payload_gib_per_s_p50;
  double steady_state_active_payload_gib_per_s;
};

enum class TiledHalfAssociation : std::uint8_t {
  kXor1ThenXor3 = 1,
  kXor3ThenXor1 = 2,
};

struct TiledCorrectnessReceipt {
  std::uint32_t query_rows;
  std::uint32_t tile_count;
  std::uint64_t active_payload_bytes;
  std::uint64_t mismatched_active_elements;
  std::uint64_t input_guard_corruptions;
  std::uint64_t output_guard_corruptions;
  std::uint64_t inactive_input_sentinel_corruptions;
  std::uint64_t inactive_output_sentinel_corruptions;
  std::uint64_t unexpected_generation_count;
  std::uint64_t credit_regression_count;
  std::uint64_t output_ready_before_final_retirement_count;
  std::uint64_t teardown_pending_tiles;
  std::uint64_t teardown_pending_operations;
  TiledHalfAssociation lower_half_association;
  TiledHalfAssociation upper_half_association;
  TiledExecutorStreamPolicy executor_stream_policy;
  bool teardown_drained;
  bool performance_claim_allowed;
};

// descriptor_pointer names exactly one validated device-resident descriptor.
// A caller must not batch later generations into one launch because their
// physical slots may still await prior-generation peer credit.
cudaError_t launch_stage_phase1_tile(
    const std::uint8_t* input, std::uint8_t* endpoint0,
    std::uint8_t* endpoint1,
    const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout, cudaStream_t stream);

cudaError_t launch_reduce_phase1_tile(
    std::uint8_t* endpoint0, std::uint8_t* endpoint1,
    const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout, cudaStream_t stream);

cudaError_t launch_reduce_phase2_tile(
    const std::uint8_t* endpoint0, const std::uint8_t* endpoint1,
    std::uint8_t* output,
    const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout, cudaStream_t stream);

}  // namespace spark_transport::tiled_prefill_research
