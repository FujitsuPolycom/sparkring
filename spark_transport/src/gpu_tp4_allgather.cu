#include "spark_transport/gpu_tp4_allgather.hpp"

#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_graph_command.cuh"
#include "spark_transport/tp4_indexer_graph.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace spark_transport {
namespace {

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

__device__ void copy_aligned_16(std::uint8_t* destination,
                                const std::uint8_t* source,
                                std::size_t bytes) {
  auto* destination_words = reinterpret_cast<uint4*>(destination);
  const auto* source_words = reinterpret_cast<const uint4*>(source);
  const std::size_t words = bytes / sizeof(uint4);
  for (std::size_t index = threadIdx.x; index < words;
       index += blockDim.x) {
    destination_words[index] = source_words[index];
  }
  __syncthreads();
}

__global__ void tp4_tensor_all_gather(
    std::uint32_t rank, std::uint8_t* round0_buffer,
    Tp2BufferLayout round0_layout, std::uint8_t* round1_buffer,
    Tp2BufferLayout round1_layout, const std::uint8_t* input,
    std::uint8_t* output, std::size_t input_bytes,
    std::uint32_t q, std::uint64_t fixed_sequence,
    Tp4GraphCommandRing* graph_commands, bool graph_trace) {
  __shared__ std::uint64_t graph_sequence;
  if (graph_commands != nullptr) {
    if (threadIdx.x == 0) {
      graph_sequence = gpu_graph_command::publish_command(
          graph_commands, graph_trace, q,
          static_cast<std::uint32_t>(input_bytes),
          Tp4GraphCommandKind::kIndexerAllgather,
          kTp4IndexerGraphDescriptorVersion);
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
  const auto* receive0 = round0_buffer + round0_layout.receive_offset;
  auto* control0 = reinterpret_cast<DoorbellControl*>(
      round0_buffer + round0_layout.control_offset);

  auto* send1 = round1_buffer + round1_layout.send_offset;
  const auto* receive1 = round1_buffer + round1_layout.receive_offset;
  auto* control1 = reinterpret_cast<DoorbellControl*>(
      round1_buffer + round1_layout.control_offset);

  copy_aligned_16(send0, input, input_bytes);
  publish_sequence_block(
      &control0->producer_sequence, doorbell_sequence);
  gpu_graph_command::wait_for_sequence_block(
      &control0->remote_sequence, doorbell_sequence, graph_commands,
      sequence);

  if ((rank & 1U) == 0U) {
    copy_aligned_16(send1, send0, input_bytes);
    copy_aligned_16(send1 + input_bytes, receive0, input_bytes);
  } else {
    copy_aligned_16(send1, receive0, input_bytes);
    copy_aligned_16(send1 + input_bytes, send0, input_bytes);
  }
  publish_sequence_block(
      &control0->consumer_sequence, doorbell_sequence);
  gpu_graph_command::wait_for_sequence_block(
      &control0->acknowledgement_sequence, doorbell_sequence,
      graph_commands, sequence);

  gpu_graph_command::wait_for_sequence_block(
      &control1->remote_sequence, doorbell_sequence, graph_commands,
      sequence);
  if (rank < 2U) {
    copy_aligned_16(output, send1, input_bytes * 2);
    copy_aligned_16(output + input_bytes * 2, receive1, input_bytes * 2);
  } else {
    copy_aligned_16(output, receive1, input_bytes * 2);
    copy_aligned_16(output + input_bytes * 2, send1, input_bytes * 2);
  }
  publish_sequence_block(
      &control1->consumer_sequence, doorbell_sequence);
  gpu_graph_command::wait_for_sequence_block(
      &control1->acknowledgement_sequence, doorbell_sequence,
      graph_commands, sequence);
  publish_sequence_block(
      &control1->observed_sequence, doorbell_sequence);
}

}  // namespace

Tp4AllgatherBufferLayout make_tp4_allgather_buffer_layout(
    std::size_t input_bytes) {
  if (input_bytes == 0 || input_bytes % sizeof(uint4) != 0 ||
      input_bytes > std::numeric_limits<std::size_t>::max() / 4) {
    throw std::invalid_argument(
        "TP4 all-gather input must be a nonzero multiple of 16 bytes");
  }
  Tp4AllgatherBufferLayout layout{};
  layout.round0 = make_tp2_buffer_layout(input_bytes);
  layout.round1 = make_tp2_buffer_layout(input_bytes * 2);
  layout.input_bytes = input_bytes;
  layout.output_bytes = input_bytes * 4;
  return layout;
}

GpuTp4AllgatherWorker::GpuTp4AllgatherWorker(
    std::uint32_t rank, const Tp4AllgatherBufferLayout& layout,
    void* round0_mapped_device_buffer,
    void* round1_mapped_device_buffer)
    : rank_(rank),
      layout_(layout),
      round0_buffer_(round0_mapped_device_buffer),
      round1_buffer_(round1_mapped_device_buffer) {
  if (rank_ >= 4 || layout_.input_bytes == 0 ||
      layout_.output_bytes != layout_.input_bytes * 4 ||
      round0_buffer_ == nullptr || round1_buffer_ == nullptr) {
    throw std::invalid_argument("invalid TP4 all-gather worker");
  }
}

void GpuTp4AllgatherWorker::enqueue(
    const void* external_input, void* external_output, void* cuda_stream,
    std::uint64_t sequence) {
  if (external_input == nullptr || external_output == nullptr ||
      sequence == 0) {
    throw std::invalid_argument("invalid TP4 all-gather operation");
  }
  constexpr int threads = 256;
  tp4_tensor_all_gather<<<1, threads, 0,
                         static_cast<cudaStream_t>(cuda_stream)>>>(
      rank_, static_cast<std::uint8_t*>(round0_buffer_), layout_.round0,
      static_cast<std::uint8_t*>(round1_buffer_), layout_.round1,
      static_cast<const std::uint8_t*>(external_input),
      static_cast<std::uint8_t*>(external_output), layout_.input_bytes,
      0, sequence, nullptr, false);
  check_cuda(cudaGetLastError(), "tp4_tensor_all_gather launch");
}

void GpuTp4AllgatherWorker::enqueue_graph(
    const void* external_input, void* external_output,
    std::uint32_t q, std::size_t active_input_bytes,
    void* cuda_stream, Tp4GraphCommandRing* command_ring, bool trace) {
  Tp4IndexerGraphDescriptor descriptor{};
  if (external_input == nullptr || external_output == nullptr ||
      command_ring == nullptr ||
      active_input_bytes > std::numeric_limits<std::uint32_t>::max() ||
      !tp4_indexer_graph_descriptor_from_q(q, &descriptor) ||
      descriptor.input_bytes != active_input_bytes ||
      active_input_bytes > layout_.input_bytes) {
    throw std::invalid_argument(
        "invalid graph TP4 indexer all-gather operation");
  }
  constexpr int threads = 256;
  tp4_tensor_all_gather<<<1, threads, 0,
                         static_cast<cudaStream_t>(cuda_stream)>>>(
      rank_, static_cast<std::uint8_t*>(round0_buffer_), layout_.round0,
      static_cast<std::uint8_t*>(round1_buffer_), layout_.round1,
      static_cast<const std::uint8_t*>(external_input),
      static_cast<std::uint8_t*>(external_output), active_input_bytes,
      q, 0, command_ring, trace);
  check_cuda(
      cudaGetLastError(), "graph tp4_tensor_all_gather launch");
}

}  // namespace spark_transport
