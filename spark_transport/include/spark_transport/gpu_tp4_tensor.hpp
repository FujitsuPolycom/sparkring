#pragma once

#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_doorbell.hpp"
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
                     std::uint32_t bytes_per_row,
                     void* round0_mapped_device_buffer,
                     const ExchangeBufferLayout& round0_layout,
                     void* round1_mapped_device_buffer,
                     const ExchangeBufferLayout& round1_layout,
                     Tp4AllreduceProtocol protocol,
                     Tp4GraphKernelStrategy graph_kernel_strategy,
                     Tp4AllreduceSchedule schedule,
                     bool graph_direct_doorbell);
  ~GpuTp4TensorWorker();

  // Enqueues one fused all-reduce kernel on the caller stream. The kernel
  // publishes GPU/CPU doorbells while a dedicated host thread drives RDMA.
  void enqueue(const void* external_input, void* external_output,
               void* cuda_stream, std::uint64_t sequence);

  // Enqueues the graph-replay kernel. The default path atomically claims and
  // publishes a replay descriptor through command_ring. The opt-in direct
  // path claims a device ticket and uses the round-0 payload doorbell as the
  // host notification. Research graph strategies may expand the operation
  // into an ordered graph-node chain; the tiered choice is per captured q.
  void enqueue_graph(const void* external_input, void* external_output,
                     std::uint32_t q, void* cuda_stream,
                     Tp4GraphCommandRing* command_ring, bool trace);

 private:
  void* round0_buffer_{};
  void* round1_buffer_{};
  ExchangeBufferLayout round0_layout_{};
  ExchangeBufferLayout round1_layout_{};
  std::size_t payload_bytes_{};
  std::uint32_t bytes_per_row_{};
  Tp4AllreduceProtocol protocol_{Tp4AllreduceProtocol::kSerialAck};
  Tp4GraphKernelStrategy graph_kernel_strategy_{
      Tp4GraphKernelStrategy::kFused};
  Tp4AllreduceSchedule schedule_{Tp4AllreduceSchedule::kSequential};
  bool graph_direct_doorbell_{};
  void* direct_graph_sequence_{};
  void* split_graph_state_{};
  void* striped_graph_state_{};
};

}  // namespace spark_transport
