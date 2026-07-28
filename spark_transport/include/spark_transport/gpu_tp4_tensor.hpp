#pragma once

#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/tp4_graph_command.hpp"

namespace spark_transport {

class GpuTp4TensorWorker {
 public:
  GpuTp4TensorWorker(const GpuTp4TensorWorker&) = delete;
  GpuTp4TensorWorker& operator=(const GpuTp4TensorWorker&) = delete;

  GpuTp4TensorWorker(std::size_t payload_bytes,
                     void* round0_mapped_device_buffer,
                     const Tp2BufferLayout& round0_layout,
                     void* round1_mapped_device_buffer,
                     const Tp2BufferLayout& round1_layout);
  ~GpuTp4TensorWorker() = default;

  // Enqueues one fused all-reduce kernel on the caller stream. The kernel
  // publishes GPU/CPU doorbells while a dedicated host thread drives RDMA.
  void enqueue(const void* external_input, void* external_output,
               void* cuda_stream, std::uint64_t sequence);

  // Enqueues the graph-replay kernel. The kernel atomically claims and
  // publishes a fresh replay sequence through command_ring before using that
  // same sequence for its transport doorbells.
  void enqueue_graph(const void* external_input, void* external_output,
                     std::uint32_t q, void* cuda_stream,
                     Tp4GraphCommandRing* command_ring, bool trace);

 private:
  void* round0_buffer_{};
  void* round1_buffer_{};
  Tp2BufferLayout round0_layout_{};
  Tp2BufferLayout round1_layout_{};
  std::size_t payload_bytes_{};
};

}  // namespace spark_transport
