#pragma once

#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/tp4_graph_command.hpp"

namespace spark_transport {

struct Tp4AllgatherBufferLayout {
  Tp2BufferLayout round0;
  Tp2BufferLayout round1;
  std::size_t input_bytes{};
  std::size_t output_bytes{};
};

Tp4AllgatherBufferLayout make_tp4_allgather_buffer_layout(
    std::size_t input_bytes);

class GpuTp4AllgatherWorker {
 public:
  GpuTp4AllgatherWorker(const GpuTp4AllgatherWorker&) = delete;
  GpuTp4AllgatherWorker& operator=(const GpuTp4AllgatherWorker&) = delete;

  GpuTp4AllgatherWorker(
      std::uint32_t rank, const Tp4AllgatherBufferLayout& layout,
      void* round0_mapped_device_buffer,
      void* round1_mapped_device_buffer);

  // Enqueues one rank-ordered fixed-size all-gather on the caller stream.
  void enqueue(const void* external_input, void* external_output,
               void* cuda_stream, std::uint64_t sequence);

  // Adds one fixed-Q INT32 [Q, 2, 2048] gather to active CUDA capture.
  // The worker owns a fixed Q40 layout; active_input_bytes selects the exact
  // prefix used by this captured node.
  void enqueue_graph(
      const void* external_input, void* external_output,
      std::uint32_t q, std::size_t active_input_bytes,
      void* cuda_stream, Tp4GraphCommandRing* command_ring, bool trace);

 private:
  std::uint32_t rank_{};
  Tp4AllgatherBufferLayout layout_{};
  void* round0_buffer_{};
  void* round1_buffer_{};
};

}  // namespace spark_transport
