#include "spark_transport/gpu_tp4_vocab_allgather.hpp"

#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_graph_command.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace spark_transport {
namespace {

static_assert(kTp4VocabBytesPerRankRow % sizeof(uint4) == 0);

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

__device__ void copy_token_major(
    std::uint8_t* output, const std::uint8_t* bundle01,
    const std::uint8_t* bundle23, std::uint32_t query_rows) {
  constexpr std::size_t words_per_rank_row =
      kTp4VocabBytesPerRankRow / sizeof(uint4);
  constexpr std::size_t words_per_output_row =
      words_per_rank_row * kTp4VocabWorldSize;
  const std::size_t rank_segment_words =
      static_cast<std::size_t>(query_rows) * words_per_rank_row;
  const std::size_t output_words =
      static_cast<std::size_t>(query_rows) * words_per_output_row;
  auto* destination = reinterpret_cast<uint4*>(output);

  for (std::size_t output_index = threadIdx.x;
       output_index < output_words; output_index += blockDim.x) {
    const std::size_t query_index =
        output_index / words_per_output_row;
    const std::size_t word_in_output_row =
        output_index % words_per_output_row;
    const std::uint32_t source_rank = static_cast<std::uint32_t>(
        word_in_output_row / words_per_rank_row);
    const std::size_t word_in_rank_row =
        word_in_output_row % words_per_rank_row;
    const auto* source_bundle = reinterpret_cast<const uint4*>(
        source_rank < 2U ? bundle01 : bundle23);
    const std::size_t source_index =
        static_cast<std::size_t>(source_rank & 1U) *
            rank_segment_words +
        query_index * words_per_rank_row + word_in_rank_row;
    destination[output_index] = source_bundle[source_index];
  }
  __syncthreads();
}

__global__ void tp4_vocab_all_gather(
    std::uint32_t rank, std::uint8_t* round0_buffer,
    Tp2BufferLayout round0_layout, std::uint8_t* round1_buffer,
    Tp2BufferLayout round1_layout, const std::uint8_t* input,
    std::uint8_t* output, std::uint32_t query_rows,
    std::uint64_t fixed_sequence,
    Tp4GraphCommandRing* graph_commands, bool graph_trace) {
  const std::size_t input_bytes =
      static_cast<std::size_t>(query_rows) *
      kTp4VocabBytesPerRankRow;
  __shared__ std::uint64_t graph_sequence;
  if (graph_commands != nullptr) {
    if (threadIdx.x == 0) {
      graph_sequence = gpu_graph_command::publish_command(
          graph_commands, graph_trace, query_rows,
          static_cast<std::uint32_t>(input_bytes));
    }
    __syncthreads();
  }
  const std::uint64_t sequence =
      graph_commands == nullptr ? fixed_sequence : graph_sequence;
  const std::uint64_t doorbell_sequence =
      graph_commands == nullptr
          ? sequence
          : tp4_graph_doorbell_token(sequence, query_rows);
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
  const std::uint8_t* bundle01 = rank < 2U ? send1 : receive1;
  const std::uint8_t* bundle23 = rank < 2U ? receive1 : send1;
  copy_token_major(output, bundle01, bundle23, query_rows);

  publish_sequence_block(
      &control1->consumer_sequence, doorbell_sequence);
  gpu_graph_command::wait_for_sequence_block(
      &control1->acknowledgement_sequence, doorbell_sequence,
      graph_commands, sequence);
  publish_sequence_block(
      &control1->observed_sequence, doorbell_sequence);
}

bool aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % sizeof(uint4) == 0;
}

void validate_query_rows(std::uint32_t query_rows) {
  if (query_rows == 0 || query_rows > kTp4VocabMaxQueryRows) {
    throw std::invalid_argument(
        "TP4 vocabulary all-gather Q must be in [1, 40]");
  }
}

}  // namespace

std::size_t tp4_vocab_input_bytes(std::uint32_t query_rows) {
  validate_query_rows(query_rows);
  return static_cast<std::size_t>(query_rows) *
         kTp4VocabBytesPerRankRow;
}

std::size_t tp4_vocab_output_bytes(std::uint32_t query_rows) {
  return tp4_vocab_input_bytes(query_rows) * kTp4VocabWorldSize;
}

std::size_t tp4_vocab_rank_major_offset(
    std::uint32_t query_rows, std::uint32_t query_index,
    std::uint32_t source_rank, std::size_t byte_in_rank_row) {
  validate_query_rows(query_rows);
  if (query_index >= query_rows ||
      source_rank >= kTp4VocabWorldSize ||
      byte_in_rank_row >= kTp4VocabBytesPerRankRow) {
    throw std::invalid_argument(
        "invalid TP4 vocabulary rank-major coordinate");
  }
  return (static_cast<std::size_t>(source_rank) * query_rows +
          query_index) *
             kTp4VocabBytesPerRankRow +
         byte_in_rank_row;
}

std::size_t tp4_vocab_pair_bundle_offset(
    std::uint32_t query_rows, std::uint32_t query_index,
    std::uint32_t source_rank, std::size_t byte_in_rank_row) {
  validate_query_rows(query_rows);
  if (query_index >= query_rows ||
      source_rank >= kTp4VocabWorldSize ||
      byte_in_rank_row >= kTp4VocabBytesPerRankRow) {
    throw std::invalid_argument(
        "invalid TP4 vocabulary pair-bundle coordinate");
  }
  return (static_cast<std::size_t>(source_rank & 1U) * query_rows +
          query_index) *
             kTp4VocabBytesPerRankRow +
         byte_in_rank_row;
}

std::size_t tp4_vocab_token_major_offset(
    std::uint32_t query_index, std::uint32_t source_rank,
    std::size_t byte_in_rank_row) {
  if (query_index >= kTp4VocabMaxQueryRows ||
      source_rank >= kTp4VocabWorldSize ||
      byte_in_rank_row >= kTp4VocabBytesPerRankRow) {
    throw std::invalid_argument(
        "invalid TP4 vocabulary token-major coordinate");
  }
  return (static_cast<std::size_t>(query_index) *
              kTp4VocabWorldSize +
          source_rank) *
             kTp4VocabBytesPerRankRow +
         byte_in_rank_row;
}

Tp4VocabAllgatherBufferLayout
make_tp4_vocab_allgather_buffer_layout() {
  Tp4VocabAllgatherBufferLayout layout{};
  layout.max_input_bytes =
      tp4_vocab_input_bytes(kTp4VocabMaxQueryRows);
  layout.max_output_bytes =
      tp4_vocab_output_bytes(kTp4VocabMaxQueryRows);
  layout.round0 = make_tp2_buffer_layout(layout.max_input_bytes);
  layout.round1 =
      make_tp2_buffer_layout(layout.max_input_bytes * 2);
  return layout;
}

GpuTp4VocabAllgatherWorker::GpuTp4VocabAllgatherWorker(
    std::uint32_t rank,
    const Tp4VocabAllgatherBufferLayout& layout,
    void* round0_mapped_device_buffer,
    void* round1_mapped_device_buffer)
    : rank_(rank),
      layout_(layout),
      round0_buffer_(round0_mapped_device_buffer),
      round1_buffer_(round1_mapped_device_buffer) {
  const auto expected = make_tp4_vocab_allgather_buffer_layout();
  if (rank_ >= kTp4VocabWorldSize ||
      layout_.max_input_bytes != expected.max_input_bytes ||
      layout_.max_output_bytes != expected.max_output_bytes ||
      layout_.round0.send_offset != expected.round0.send_offset ||
      layout_.round0.receive_offset !=
          expected.round0.receive_offset ||
      layout_.round0.control_offset != expected.round0.control_offset ||
      layout_.round0.total_bytes != expected.round0.total_bytes ||
      layout_.round1.send_offset != expected.round1.send_offset ||
      layout_.round1.receive_offset !=
          expected.round1.receive_offset ||
      layout_.round1.control_offset != expected.round1.control_offset ||
      layout_.round1.total_bytes != expected.round1.total_bytes ||
      round0_buffer_ == nullptr || round1_buffer_ == nullptr) {
    throw std::invalid_argument(
        "invalid TP4 vocabulary all-gather worker");
  }
}

void GpuTp4VocabAllgatherWorker::enqueue(
    const void* input, void* output, std::uint32_t query_rows,
    void* cuda_stream, std::uint64_t sequence) {
  static_cast<void>(tp4_vocab_input_bytes(query_rows));
  if (input == nullptr || output == nullptr || sequence == 0 ||
      !aligned_16(input) || !aligned_16(output)) {
    throw std::invalid_argument(
        "invalid TP4 vocabulary all-gather operation");
  }
  constexpr int threads = 256;
  check_cuda(
      cudaGetLastError(),
      "prior CUDA error before tp4_vocab_all_gather launch");
  tp4_vocab_all_gather<<<
      1, threads, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
      rank_, static_cast<std::uint8_t*>(round0_buffer_),
      layout_.round0, static_cast<std::uint8_t*>(round1_buffer_),
      layout_.round1, static_cast<const std::uint8_t*>(input),
      static_cast<std::uint8_t*>(output), query_rows, sequence, nullptr,
      false);
  check_cuda(cudaGetLastError(), "tp4_vocab_all_gather launch");
}

void GpuTp4VocabAllgatherWorker::enqueue_graph(
    const void* input, void* output, std::uint32_t query_rows,
    void* cuda_stream, Tp4GraphCommandRing* command_ring, bool trace) {
  static_cast<void>(tp4_vocab_input_bytes(query_rows));
  if (input == nullptr || output == nullptr || command_ring == nullptr ||
      !aligned_16(input) || !aligned_16(output)) {
    throw std::invalid_argument(
        "invalid graph TP4 vocabulary all-gather operation");
  }
  constexpr int threads = 256;
  tp4_vocab_all_gather<<<
      1, threads, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
      rank_, static_cast<std::uint8_t*>(round0_buffer_),
      layout_.round0, static_cast<std::uint8_t*>(round1_buffer_),
      layout_.round1, static_cast<const std::uint8_t*>(input),
      static_cast<std::uint8_t*>(output), query_rows, 0, command_ring,
      trace);
  check_cuda(
      cudaGetLastError(), "graph tp4_vocab_all_gather launch");
}

}  // namespace spark_transport
