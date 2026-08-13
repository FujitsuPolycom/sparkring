#pragma once

#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/tp4_allreduce_protocol.hpp"
#include "spark_transport/tp4_dual_port_striped_allreduce.hpp"
#include "spark_transport/tp4_graph_command.hpp"
#include "spark_transport/tp4_graph_kernel_strategy.hpp"

namespace spark_transport {

class GpuTp4TensorWorker {
 public:
  GpuTp4TensorWorker(const GpuTp4TensorWorker&) = delete;
  GpuTp4TensorWorker& operator=(const GpuTp4TensorWorker&) = delete;

  GpuTp4TensorWorker(std::size_t payload_bytes,
                     void* round0_mapped_device_buffer,
                     const Tp2BufferLayout& round0_layout,
                     void* round1_mapped_device_buffer,
                     const Tp2BufferLayout& round1_layout,
                     Tp4AllreduceProtocol protocol,
                     Tp4GraphKernelStrategy graph_kernel_strategy,
                     Tp4AllreduceSchedule schedule);
  ~GpuTp4TensorWorker();

  // Enqueues one fused all-reduce kernel on the caller stream. The kernel
  // publishes GPU/CPU doorbells while a dedicated host thread drives RDMA.
  void enqueue(const void* external_input, void* external_output,
               void* cuda_stream, std::uint64_t sequence);

  // Enqueues the graph-replay kernel. The kernel atomically claims and
  // publishes a fresh replay sequence through command_ring before using that
  // same sequence for its transport doorbells. Research graph strategies may
  // expand the logical operation into an ordered graph-node chain without
  // changing the host progress protocol. The tiered strategy makes that
  // choice independently for every captured q.
  void enqueue_graph(const void* external_input, void* external_output,
                     std::uint32_t q, void* cuda_stream,
                     Tp4GraphCommandRing* command_ring, bool trace);

 private:
  void* round0_buffer_{};
  void* round1_buffer_{};
  Tp2BufferLayout round0_layout_{};
  Tp2BufferLayout round1_layout_{};
  std::size_t payload_bytes_{};
  Tp4AllreduceProtocol protocol_{Tp4AllreduceProtocol::kSerialAck};
  Tp4GraphKernelStrategy graph_kernel_strategy_{
      Tp4GraphKernelStrategy::kFused};
  Tp4AllreduceSchedule schedule_{Tp4AllreduceSchedule::kSequential};
  void* split_graph_state_{};
  void* striped_graph_state_{};
};

}  // namespace spark_transport
