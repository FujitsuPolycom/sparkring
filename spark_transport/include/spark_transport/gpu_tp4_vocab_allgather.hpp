#pragma once

#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/tp4_graph_command.hpp"

namespace spark_transport {

inline constexpr std::uint32_t kTp4VocabWorldSize = 4;
inline constexpr std::uint32_t kTp4VocabMaxQueryRows =
    kTp4GraphMaximumQ;
inline constexpr std::size_t kTp4VocabShardElements = 38720;
inline constexpr std::size_t kTp4VocabElementBytes = 2;
inline constexpr std::size_t kTp4VocabBytesPerRankRow =
    kTp4VocabShardElements * kTp4VocabElementBytes;

struct Tp4VocabAllgatherBufferLayout {
  Tp2BufferLayout round0;
  Tp2BufferLayout round1;
  std::size_t max_input_bytes{};
  std::size_t max_output_bytes{};
};

std::size_t tp4_vocab_input_bytes(std::uint32_t query_rows);
std::size_t tp4_vocab_output_bytes(std::uint32_t query_rows);
std::size_t tp4_vocab_rank_major_offset(
    std::uint32_t query_rows, std::uint32_t query_index,
    std::uint32_t source_rank, std::size_t byte_in_rank_row);
std::size_t tp4_vocab_pair_bundle_offset(
    std::uint32_t query_rows, std::uint32_t query_index,
    std::uint32_t source_rank, std::size_t byte_in_rank_row);
std::size_t tp4_vocab_token_major_offset(
    std::uint32_t query_index, std::uint32_t source_rank,
    std::size_t byte_in_rank_row);
Tp4VocabAllgatherBufferLayout make_tp4_vocab_allgather_buffer_layout();

class GpuTp4VocabAllgatherWorker {
 public:
  GpuTp4VocabAllgatherWorker(
      const GpuTp4VocabAllgatherWorker&) = delete;
  GpuTp4VocabAllgatherWorker& operator=(
      const GpuTp4VocabAllgatherWorker&) = delete;

  GpuTp4VocabAllgatherWorker(
      std::uint32_t rank,
      const Tp4VocabAllgatherBufferLayout& layout,
      void* round0_mapped_device_buffer,
      void* round1_mapped_device_buffer);

  // Input is contiguous BF16 [Q, 38720]. Output is contiguous BF16
  // [Q, 154880], with rank shards concatenated within each query row.
  void enqueue(const void* input, void* output, std::uint32_t query_rows,
               void* cuda_stream, std::uint64_t sequence);

  // Adds one fixed-Q vocabulary gather to an active CUDA stream capture.
  // The command ring publishes a fresh sequence and Q on every replay.
  void enqueue_graph(
      const void* input, void* output, std::uint32_t query_rows,
      void* cuda_stream, Tp4GraphCommandRing* command_ring, bool trace);

 private:
  std::uint32_t rank_{};
  Tp4VocabAllgatherBufferLayout layout_{};
  void* round0_buffer_{};
  void* round1_buffer_{};
};

}  // namespace spark_transport
