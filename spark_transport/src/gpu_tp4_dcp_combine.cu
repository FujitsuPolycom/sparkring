#include "spark_transport/gpu_tp4_dcp_combine.hpp"

#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_graph_command.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace spark_transport {
namespace {

constexpr unsigned int kFullWarpMask = 0xffffffffU;

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

__device__ void publish_sequence_block(std::uint64_t* address,
                                       std::uint64_t sequence) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    reinterpret_cast<volatile std::uint64_t*>(address)[0] = sequence;
  }
  __syncthreads();
}

__device__ std::uint32_t round0_keep_chunk(std::uint32_t rank,
                                           std::uint32_t slot) {
  if (rank == 0U || rank == 3U) {
    return slot == 0U ? 0U : 3U;
  }
  return slot == 0U ? 1U : 2U;
}

__device__ std::uint32_t round0_send_chunk(std::uint32_t rank,
                                           std::uint32_t slot) {
  if (rank == 0U || rank == 3U) {
    return slot == 0U ? 1U : 2U;
  }
  return slot == 0U ? 0U : 3U;
}

__device__ float sanitize_lse_device(float value) {
  return isnan(value) || value == CUDART_INF_F ? -CUDART_INF_F : value;
}

struct MergeParameters {
  float weight_a{};
  float weight_b{};
  float inverse_denominator{};
  float lse{};
  int empty{};
};

__device__ __forceinline__ __nv_bfloat162 merge_output_pair(
    MergeParameters parameters, __nv_bfloat162 value_a_pair,
    __nv_bfloat162 value_b_pair) {
  float merged_x = 0.0F;
  float merged_y = 0.0F;
  if (parameters.empty == 0) {
    const float2 value_a = __bfloat1622float2(value_a_pair);
    const float2 value_b = __bfloat1622float2(value_b_pair);
    const float weighted_a_x =
        parameters.weight_a == 0.0F
            ? 0.0F
            : parameters.weight_a * value_a.x;
    const float weighted_a_y =
        parameters.weight_a == 0.0F
            ? 0.0F
            : parameters.weight_a * value_a.y;
    const float weighted_b_x =
        parameters.weight_b == 0.0F
            ? 0.0F
            : parameters.weight_b * value_b.x;
    const float weighted_b_y =
        parameters.weight_b == 0.0F
            ? 0.0F
            : parameters.weight_b * value_b.y;
    merged_x = (weighted_a_x + weighted_b_x) *
               parameters.inverse_denominator;
    merged_y = (weighted_a_y + weighted_b_y) *
               parameters.inverse_denominator;
  }
  return __floats2bfloat162_rn(merged_x, merged_y);
}

__device__ MergeParameters merge_parameters_device(float lse_a,
                                                    float lse_b) {
  MergeParameters parameters{};
  lse_a = sanitize_lse_device(lse_a);
  lse_b = sanitize_lse_device(lse_b);
  if (lse_a == -CUDART_INF_F && lse_b == -CUDART_INF_F) {
    parameters.lse = -CUDART_INF_F;
    parameters.empty = 1;
    return parameters;
  }
  const float maximum = fmaxf(lse_a, lse_b);
  parameters.weight_a = expf(lse_a - maximum);
  parameters.weight_b = expf(lse_b - maximum);
  const float denominator =
      parameters.weight_a + parameters.weight_b;
  parameters.inverse_denominator = 1.0F / denominator;
  parameters.lse = maximum + logf(denominator);
  return parameters;
}

__device__ MergeParameters broadcast_merge_parameters(
    MergeParameters parameters) {
  parameters.weight_a =
      __shfl_sync(kFullWarpMask, parameters.weight_a, 0);
  parameters.weight_b =
      __shfl_sync(kFullWarpMask, parameters.weight_b, 0);
  parameters.inverse_denominator = __shfl_sync(
      kFullWarpMask, parameters.inverse_denominator, 0);
  parameters.lse = __shfl_sync(kFullWarpMask, parameters.lse, 0);
  parameters.empty =
      __shfl_sync(kFullWarpMask, parameters.empty, 0);
  return parameters;
}

__device__ void pack_round0(
    std::uint32_t rank, const __nv_bfloat16* output, const float* lse,
    __nv_bfloat16* packed_output, float* packed_lse, std::uint32_t q,
    std::uint32_t head_dimension, std::uint32_t query_stride,
    std::uint32_t head_stride) {
  constexpr std::size_t packed_heads =
      kTp4DcpCombineHeadsPerRank * 2;
  const std::size_t output_pairs =
      static_cast<std::size_t>(q) * packed_heads *
      (head_dimension / 2);
  auto* packed_output_pairs =
      reinterpret_cast<__nv_bfloat162*>(packed_output);
  for (std::size_t pair_index = threadIdx.x;
       pair_index < output_pairs; pair_index += blockDim.x) {
    const std::size_t dimension_pair =
        pair_index % (head_dimension / 2);
    const std::size_t dimension =
        dimension_pair * 2;
    const std::size_t state =
        pair_index / (head_dimension / 2);
    const std::size_t packed_head = state % packed_heads;
    const std::size_t query_index = state / packed_heads;
    const std::uint32_t chunk = round0_send_chunk(
        rank, static_cast<std::uint32_t>(
                  packed_head / kTp4DcpCombineHeadsPerRank));
    const std::size_t local_head =
        packed_head % kTp4DcpCombineHeadsPerRank;
    const std::size_t global_head =
        static_cast<std::size_t>(chunk) *
            kTp4DcpCombineHeadsPerRank +
        local_head;
    const std::size_t input_index =
        tp4_dcp_combine_strided_output_index(
            query_index, global_head, dimension, query_stride,
            head_stride);
    packed_output_pairs[pair_index] =
        *reinterpret_cast<const __nv_bfloat162*>(
            output + input_index);
  }
  const std::size_t lse_elements =
      static_cast<std::size_t>(q) * packed_heads;
  for (std::size_t index = threadIdx.x; index < lse_elements;
       index += blockDim.x) {
    const std::size_t packed_head = index % packed_heads;
    const std::size_t query_index = index / packed_heads;
    const std::uint32_t chunk = round0_send_chunk(
        rank, static_cast<std::uint32_t>(
                  packed_head / kTp4DcpCombineHeadsPerRank));
    const std::size_t local_head =
        packed_head % kTp4DcpCombineHeadsPerRank;
    const std::size_t global_head =
        static_cast<std::size_t>(chunk) *
            kTp4DcpCombineHeadsPerRank +
        local_head;
    packed_lse[index] =
        lse[query_index * kTp4DcpCombineGlobalHeads + global_head];
  }
  __syncthreads();
}

__device__ void merge_round0(
    std::uint32_t rank, const __nv_bfloat16* local_output,
    const float* local_lse, const __nv_bfloat16* remote_output,
    const float* remote_lse, __nv_bfloat16* round1_output,
    float* round1_lse, __nv_bfloat16* reduced_output,
    float* reduced_lse, std::uint32_t q,
    std::uint32_t head_dimension, std::uint32_t query_stride,
    std::uint32_t head_stride) {
  constexpr std::size_t retained_heads =
      kTp4DcpCombineHeadsPerRank * 2;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t warps = blockDim.x >> 5U;
  const std::size_t states =
      static_cast<std::size_t>(q) * retained_heads;
  for (std::size_t state = warp; state < states; state += warps) {
    const std::size_t retained_head = state % retained_heads;
    const std::size_t query_index = state / retained_heads;
    const std::uint32_t chunk = round0_keep_chunk(
        rank, static_cast<std::uint32_t>(
                  retained_head / kTp4DcpCombineHeadsPerRank));
    const std::size_t local_head =
        retained_head % kTp4DcpCombineHeadsPerRank;
    const std::size_t global_head =
        static_cast<std::size_t>(chunk) *
            kTp4DcpCombineHeadsPerRank +
        local_head;
    const std::size_t local_state =
        query_index * kTp4DcpCombineGlobalHeads + global_head;
    MergeParameters parameters{};
    if (lane == 0U) {
      parameters = merge_parameters_device(
          local_lse[local_state], remote_lse[state]);
    }
    parameters = broadcast_merge_parameters(parameters);

    const bool owned = chunk == rank;
    const std::size_t destination_state =
        query_index * kTp4DcpCombineHeadsPerRank + local_head;
    if (lane == 0U) {
      (owned ? reduced_lse : round1_lse)[destination_state] =
          parameters.lse;
    }
    for (std::size_t dimension = lane * 2;
         dimension < head_dimension; dimension += 64) {
      const std::size_t local_index =
          tp4_dcp_combine_strided_output_index(
              query_index, global_head, dimension, query_stride,
              head_stride);
      const std::size_t remote_index =
          state * head_dimension + dimension;
      const std::size_t destination_index =
          tp4_dcp_combine_token_major_reduced_output_index(
              query_index, local_head, dimension, head_dimension);
      *reinterpret_cast<__nv_bfloat162*>(
          (owned ? reduced_output : round1_output) +
          destination_index) = merge_output_pair(
              parameters,
              *reinterpret_cast<const __nv_bfloat162*>(
                  local_output + local_index),
              *reinterpret_cast<const __nv_bfloat162*>(
                  remote_output + remote_index));
    }
  }
  __syncthreads();
}

__device__ void merge_round1(
    const __nv_bfloat16* remote_output, const float* remote_lse,
    __nv_bfloat16* reduced_output, float* reduced_lse,
    std::uint32_t q, std::uint32_t head_dimension) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t warps = blockDim.x >> 5U;
  const std::size_t states =
      static_cast<std::size_t>(q) * kTp4DcpCombineHeadsPerRank;
  for (std::size_t state = warp; state < states; state += warps) {
    MergeParameters parameters{};
    if (lane == 0U) {
      parameters = merge_parameters_device(
          reduced_lse[state], remote_lse[state]);
      reduced_lse[state] = parameters.lse;
    }
    parameters = broadcast_merge_parameters(parameters);
    for (std::size_t dimension = lane * 2;
         dimension < head_dimension; dimension += 64) {
      const std::size_t index =
          state * head_dimension + dimension;
      *reinterpret_cast<__nv_bfloat162*>(reduced_output + index) =
          merge_output_pair(
              parameters,
              *reinterpret_cast<const __nv_bfloat162*>(
                  reduced_output + index),
              *reinterpret_cast<const __nv_bfloat162*>(
                  remote_output + index));
    }
  }
  __syncthreads();
}

__global__ void tp4_dcp_softmax_state_rs(
    std::uint32_t rank, std::uint8_t* round0_buffer,
    Tp2BufferLayout round0_layout, std::uint8_t* round1_buffer,
    Tp2BufferLayout round1_layout, const __nv_bfloat16* output,
    const float* lse, __nv_bfloat16* reduced_output,
    float* reduced_lse, std::uint32_t q,
    std::uint32_t head_dimension, std::uint32_t query_stride,
    std::uint32_t head_stride, std::uint64_t fixed_sequence,
    Tp4GraphCommandRing* graph_commands, bool graph_trace) {
  const std::size_t round0_output_bytes =
      static_cast<std::size_t>(q) *
      (kTp4DcpCombineHeadsPerRank * 2) *
      head_dimension *
      kTp4DcpCombineOutputElementBytes;
  const std::size_t round1_output_bytes =
      static_cast<std::size_t>(q) * kTp4DcpCombineHeadsPerRank *
      head_dimension *
      kTp4DcpCombineOutputElementBytes;
  const std::size_t round0_total_bytes =
      round0_output_bytes +
      static_cast<std::size_t>(q) *
          (kTp4DcpCombineHeadsPerRank * 2) *
          kTp4DcpCombineLseElementBytes;
  __shared__ std::uint64_t graph_sequence;
  if (graph_commands != nullptr) {
    if (threadIdx.x == 0) {
      graph_sequence = gpu_graph_command::publish_command(
          graph_commands, graph_trace, q,
          static_cast<std::uint32_t>(round0_total_bytes),
          Tp4GraphCommandKind::kDcpCombine, head_dimension);
    }
    __syncthreads();
  }
  const std::uint64_t sequence =
      graph_commands == nullptr ? fixed_sequence : graph_sequence;
  const std::uint64_t doorbell_sequence =
      graph_commands == nullptr
          ? sequence
          : tp4_graph_doorbell_token(sequence, q);

  auto* send0 = round0_buffer + round0_layout.send_offset;
  const auto* receive0 =
      round0_buffer + round0_layout.receive_offset;
  auto* control0 = reinterpret_cast<DoorbellControl*>(
      round0_buffer + round0_layout.control_offset);
  auto* send1 = round1_buffer + round1_layout.send_offset;
  const auto* receive1 =
      round1_buffer + round1_layout.receive_offset;
  auto* control1 = reinterpret_cast<DoorbellControl*>(
      round1_buffer + round1_layout.control_offset);

  auto* send0_output = reinterpret_cast<__nv_bfloat16*>(send0);
  auto* send0_lse =
      reinterpret_cast<float*>(send0 + round0_output_bytes);
  const auto* receive0_output =
      reinterpret_cast<const __nv_bfloat16*>(receive0);
  const auto* receive0_lse =
      reinterpret_cast<const float*>(receive0 + round0_output_bytes);
  auto* send1_output = reinterpret_cast<__nv_bfloat16*>(send1);
  auto* send1_lse =
      reinterpret_cast<float*>(send1 + round1_output_bytes);
  const auto* receive1_output =
      reinterpret_cast<const __nv_bfloat16*>(receive1);
  const auto* receive1_lse =
      reinterpret_cast<const float*>(receive1 + round1_output_bytes);

  pack_round0(rank, output, lse, send0_output, send0_lse, q,
              head_dimension, query_stride, head_stride);
  publish_sequence_block(
      &control0->producer_sequence, doorbell_sequence);
  gpu_graph_command::wait_for_sequence_block(
      &control0->remote_sequence, doorbell_sequence, graph_commands,
      sequence);

  merge_round0(rank, output, lse, receive0_output, receive0_lse,
               send1_output, send1_lse, reduced_output, reduced_lse, q,
               head_dimension, query_stride, head_stride);
  publish_sequence_block(
      &control0->consumer_sequence, doorbell_sequence);
  gpu_graph_command::wait_for_sequence_block(
      &control0->acknowledgement_sequence, doorbell_sequence,
      graph_commands, sequence);

  gpu_graph_command::wait_for_sequence_block(
      &control1->remote_sequence, doorbell_sequence, graph_commands,
      sequence);
  merge_round1(receive1_output, receive1_lse, reduced_output,
               reduced_lse, q, head_dimension);
  publish_sequence_block(
      &control1->consumer_sequence, doorbell_sequence);
  gpu_graph_command::wait_for_sequence_block(
      &control1->acknowledgement_sequence, doorbell_sequence,
      graph_commands, sequence);
  publish_sequence_block(
      &control1->observed_sequence, doorbell_sequence);
}

bool aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
}

}  // namespace

GpuTp4DcpCombineWorker::GpuTp4DcpCombineWorker(
    std::uint32_t rank, const Tp4DcpQueryBufferLayout& shared_layout,
    void* round0_mapped_device_buffer,
    void* round1_mapped_device_buffer)
    : rank_(rank),
      layout_(shared_layout),
      round0_buffer_(round0_mapped_device_buffer),
      round1_buffer_(round1_mapped_device_buffer) {
  const auto expected = make_tp4_dcp_query_buffer_layout();
  const auto maximum_round0 =
      tp4_dcp_combine_round0_frame(
          kTp4DcpCombineMaxQ, kTp4DcpCombineMaxHeadDimension);
  const auto maximum_round1 =
      tp4_dcp_combine_round1_frame(
          kTp4DcpCombineMaxQ, kTp4DcpCombineMaxHeadDimension);
  if (rank_ >= kTp4DcpCombineWorldSize ||
      layout_.max_input_bytes != expected.max_input_bytes ||
      layout_.max_output_bytes != expected.max_output_bytes ||
      layout_.round0.send_offset != expected.round0.send_offset ||
      layout_.round0.receive_offset != expected.round0.receive_offset ||
      layout_.round0.control_offset != expected.round0.control_offset ||
      layout_.round0.total_bytes != expected.round0.total_bytes ||
      layout_.round1.send_offset != expected.round1.send_offset ||
      layout_.round1.receive_offset != expected.round1.receive_offset ||
      layout_.round1.control_offset != expected.round1.control_offset ||
      layout_.round1.total_bytes != expected.round1.total_bytes ||
      maximum_round0.total_bytes > layout_.round0.receive_offset ||
      maximum_round1.total_bytes > layout_.round1.receive_offset ||
      round0_buffer_ == nullptr || round1_buffer_ == nullptr) {
    throw std::invalid_argument("invalid TP4 DCP combine worker");
  }
}

void GpuTp4DcpCombineWorker::enqueue(
    const void* output_bf16, const void* lse_fp32,
    void* reduced_output_bf16, void* reduced_lse_fp32,
    std::uint32_t q, std::uint32_t head_dimension,
    std::uint32_t query_stride, std::uint32_t head_stride,
    void* cuda_stream, std::uint64_t sequence) {
  static_cast<void>(
      tp4_dcp_combine_round0_frame(q, head_dimension));
  const bool token_major =
      query_stride ==
          kTp4DcpCombineGlobalHeads * head_dimension &&
      head_stride == head_dimension;
  const bool head_major =
      query_stride == head_dimension &&
      head_stride == static_cast<std::uint32_t>(
                         static_cast<std::size_t>(q) *
                         head_dimension);
  if (output_bf16 == nullptr || lse_fp32 == nullptr ||
      reduced_output_bf16 == nullptr || reduced_lse_fp32 == nullptr ||
      sequence == 0 || !aligned_16(output_bf16) ||
      !aligned_16(lse_fp32) || !aligned_16(reduced_output_bf16) ||
      !aligned_16(reduced_lse_fp32) ||
      (!token_major && !head_major)) {
    throw std::invalid_argument("invalid TP4 DCP combine operation");
  }
  constexpr int threads = 1024;
  tp4_dcp_softmax_state_rs<<<
      1, threads, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
      rank_, static_cast<std::uint8_t*>(round0_buffer_), layout_.round0,
      static_cast<std::uint8_t*>(round1_buffer_), layout_.round1,
      static_cast<const __nv_bfloat16*>(output_bf16),
      static_cast<const float*>(lse_fp32),
      static_cast<__nv_bfloat16*>(reduced_output_bf16),
      static_cast<float*>(reduced_lse_fp32), q, head_dimension,
      query_stride, head_stride, sequence, nullptr, false);
  check_cuda(cudaGetLastError(), "tp4_dcp_softmax_state_rs launch");
}

void GpuTp4DcpCombineWorker::enqueue_graph(
    const void* output_bf16, const void* lse_fp32,
    void* reduced_output_bf16, void* reduced_lse_fp32,
    std::uint32_t q, std::uint32_t head_dimension,
    std::uint32_t query_stride, std::uint32_t head_stride,
    void* cuda_stream, Tp4GraphCommandRing* command_ring, bool trace) {
  static_cast<void>(
      tp4_dcp_combine_round0_frame(q, head_dimension));
  const bool token_major =
      query_stride ==
          kTp4DcpCombineGlobalHeads * head_dimension &&
      head_stride == head_dimension;
  const bool head_major =
      query_stride == head_dimension &&
      head_stride == static_cast<std::uint32_t>(
                         static_cast<std::size_t>(q) *
                         head_dimension);
  if (output_bf16 == nullptr || lse_fp32 == nullptr ||
      reduced_output_bf16 == nullptr || reduced_lse_fp32 == nullptr ||
      command_ring == nullptr || !aligned_16(output_bf16) ||
      !aligned_16(lse_fp32) || !aligned_16(reduced_output_bf16) ||
      !aligned_16(reduced_lse_fp32) ||
      (!token_major && !head_major)) {
    throw std::invalid_argument(
        "invalid graph TP4 DCP combine operation");
  }
  constexpr int threads = 1024;
  tp4_dcp_softmax_state_rs<<<
      1, threads, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
      rank_, static_cast<std::uint8_t*>(round0_buffer_), layout_.round0,
      static_cast<std::uint8_t*>(round1_buffer_), layout_.round1,
      static_cast<const __nv_bfloat16*>(output_bf16),
      static_cast<const float*>(lse_fp32),
      static_cast<__nv_bfloat16*>(reduced_output_bf16),
      static_cast<float*>(reduced_lse_fp32), q, head_dimension,
      query_stride, head_stride, 0, command_ring, trace);
  check_cuda(
      cudaGetLastError(), "graph tp4_dcp_softmax_state_rs launch");
}

}  // namespace spark_transport
