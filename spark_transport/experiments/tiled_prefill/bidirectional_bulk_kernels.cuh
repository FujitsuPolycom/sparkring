#pragma once

#include "bidirectional_bulk_abi.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace spark_transport::tiled_prefill_research {

inline BidirectionalBulkDescriptor make_bidirectional_stage_initial_descriptor(
    std::uint32_t rank, Tp4PrefillDirection direction,
    std::uint32_t tile_in_shard, std::uint64_t send_offset_bytes = 0,
    std::uint32_t query_rows = kBidirectionalBulkDefaultQueryRows) {
  if (tile_in_shard >= kBidirectionalBulkTilesPerShard) {
    throw std::out_of_range("bidirectional tile index must be in [0, 3]");
  }
  const auto transfer = tp4_prefill_stage(rank, direction, 0);
  const auto half = static_cast<std::uint32_t>(transfer.half);
  const auto tile_bytes = bidirectional_bulk_tile_bytes(query_rows);
  const std::uint64_t tensor_offset =
      half * bidirectional_bulk_half_bytes(query_rows) +
      transfer.send_shard * bidirectional_bulk_shard_bytes(query_rows) +
      tile_in_shard * tile_bytes;
  return {tensor_offset,
          send_offset_bytes,
          0,
          0,
          0,
          0,
          0,
          tile_bytes,
          rank,
          0,
          transfer.send_shard,
          tile_in_shard,
          half,
          static_cast<std::int32_t>(direction),
          query_rows,
          kBidirectionalBulkElementsPerRow};
}

inline BidirectionalBulkDescriptor make_bidirectional_post_exchange_descriptor(
    std::uint32_t rank, Tp4PrefillDirection direction, std::uint32_t stage,
    std::uint32_t tile_in_shard, std::uint64_t send_offset_bytes = 0,
    std::uint64_t receive_offset_bytes = 0,
    std::uint64_t inbound_doorbell_offset_bytes = 0,
    std::uint64_t expected_doorbell_token = 0,
    std::uint32_t query_rows = kBidirectionalBulkDefaultQueryRows,
    std::uint64_t secondary_inbound_doorbell_offset_bytes = 0,
    std::uint64_t secondary_expected_doorbell_token = 0) {
  // The executor bridge passes the source exchange (exchange_stage - 1)
  // together with that exchange's incoming doorbell offset and exact token.
  // Host observation alone is not the GPU payload-visibility contract.
  if (stage >= kTp4PrefillStageCount ||
      tile_in_shard >= kBidirectionalBulkTilesPerShard) {
    throw std::out_of_range("invalid bidirectional stage or tile index");
  }
  if (expected_doorbell_token == 0 ||
      inbound_doorbell_offset_bytes % alignof(std::uint64_t) != 0) {
    throw std::invalid_argument(
        "post-exchange descriptor requires an aligned inbound doorbell token");
  }
  const auto transfer = tp4_prefill_stage(rank, direction, stage);
  const auto half = static_cast<std::uint32_t>(transfer.half);
  const auto tile_bytes = bidirectional_bulk_tile_bytes(query_rows);
  const std::uint64_t tensor_offset =
      half * bidirectional_bulk_half_bytes(query_rows) +
      transfer.receive_shard * bidirectional_bulk_shard_bytes(query_rows) +
      tile_in_shard * tile_bytes;
  return {tensor_offset,
          send_offset_bytes,
          receive_offset_bytes,
          inbound_doorbell_offset_bytes,
          expected_doorbell_token,
          secondary_inbound_doorbell_offset_bytes,
          secondary_expected_doorbell_token,
          tile_bytes,
          rank,
          stage,
          transfer.receive_shard,
          tile_in_shard,
          half,
          static_cast<std::int32_t>(direction),
          query_rows,
          kBidirectionalBulkElementsPerRow};
}

cudaError_t launch_bidirectional_stage_initial(
    const std::uint8_t* input, std::uint8_t* outgoing_endpoint,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows = kBidirectionalBulkDefaultQueryRows);

cudaError_t launch_bidirectional_reduce_forward(
    const std::uint8_t* input, const std::uint8_t* incoming_endpoint,
    std::uint8_t* outgoing_endpoint,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows = kBidirectionalBulkDefaultQueryRows);

cudaError_t launch_bidirectional_reduce_finalize_seed_gather(
    const std::uint8_t* input, const std::uint8_t* incoming_endpoint,
    std::uint8_t* outgoing_endpoint, std::uint8_t* output,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows = kBidirectionalBulkDefaultQueryRows);

cudaError_t launch_bidirectional_gather_forward(
    const std::uint8_t* incoming_endpoint, std::uint8_t* outgoing_endpoint,
    std::uint8_t* output, const BidirectionalBulkDescriptor& descriptor,
    cudaStream_t stream,
    std::uint32_t query_rows = kBidirectionalBulkDefaultQueryRows);

cudaError_t launch_bidirectional_gather_finish(
    const std::uint8_t* incoming_endpoint, std::uint8_t* output,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows = kBidirectionalBulkDefaultQueryRows);

}  // namespace spark_transport::tiled_prefill_research
