#include "spark_transport/gpu_tp4_tensor.hpp"

#include "spark_transport/gpu_doorbell.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

using GraphAtomicWord = unsigned long long;
static_assert(sizeof(GraphAtomicWord) == sizeof(std::uint64_t));

__device__ GraphAtomicWord* graph_atomic_word(
    std::uint64_t* address) {
  return reinterpret_cast<GraphAtomicWord*>(address);
}

__device__ std::uint64_t graph_load_system(
    const std::uint64_t* address) {
  auto* mutable_address = const_cast<std::uint64_t*>(address);
  return atomicCAS_system(graph_atomic_word(mutable_address), 0ULL, 0ULL);
}

__device__ void graph_store_system(std::uint64_t* address,
                                   std::uint64_t value) {
  auto* atomic_address = graph_atomic_word(address);
  GraphAtomicWord current = atomicCAS_system(atomic_address, 0ULL, 0ULL);
  while (atomicCAS_system(atomic_address, current, value) != current) {
    current = atomicCAS_system(atomic_address, 0ULL, 0ULL);
  }
}

__device__ void graph_publish_overflow(Tp4GraphCommandRing* ring,
                                       std::uint64_t sequence) {
  atomicCAS_system(
      graph_atomic_word(&ring->producer.overflow_sequence), 0ULL,
      sequence);
}

__device__ __forceinline__ void graph_fatal_wait() {
  while (true) {
    __nanosleep(1024);
  }
}

__device__ std::uint64_t graph_claim_sequence(
    Tp4GraphCommandRing* ring) {
  while (true) {
    if (graph_load_system(&ring->producer.overflow_sequence) != 0) {
      graph_fatal_wait();
    }
    const std::uint64_t claimed =
        graph_load_system(&ring->producer.claimed_sequence);
    const std::uint64_t completed =
        graph_load_system(&ring->consumer.completed_sequence);
    __threadfence_system();
    if (claimed >= kTp4GraphMaximumDoorbellSequence ||
        completed > claimed) {
      graph_publish_overflow(
          ring,
          claimed >= kTp4GraphMaximumDoorbellSequence
              ? claimed
              : claimed + 1);
      graph_fatal_wait();
    }
    if (claimed - completed >= kTp4GraphCommandCapacity) {
      __nanosleep(64);
      continue;
    }
    const std::uint64_t next = claimed + 1;
    if (atomicCAS_system(
            graph_atomic_word(&ring->producer.claimed_sequence), claimed,
            next) == claimed) {
      return next;
    }
  }
}

__device__ std::uint64_t graph_publish_command(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes) {
  const std::uint64_t sequence = graph_claim_sequence(ring);
  auto& command =
      ring->commands[(sequence - 1) % kTp4GraphCommandCapacity];
  command.trace = trace ? 1U : 0U;
  command.q = q;
  command.payload_bytes = payload_bytes;
  command.kind = Tp4GraphCommandKind::kLegacy;
  command.parameter = 0;
  __threadfence_system();
  graph_store_system(&command.sequence, sequence);
  __threadfence_system();

  while (true) {
    const std::uint64_t expected = sequence - 1;
    const std::uint64_t published =
        atomicCAS_system(
            graph_atomic_word(&ring->producer.published_sequence),
            expected, sequence);
    if (published == expected) {
      __threadfence_system();
      return sequence;
    }
    if (published >= sequence) {
      graph_publish_overflow(ring, sequence);
      graph_fatal_wait();
    }
    __nanosleep(64);
  }
}

__device__ void wait_for_sequence_block(const std::uint64_t* address,
                                        std::uint64_t sequence,
                                        Tp4GraphCommandRing* graph_commands,
                                        std::uint64_t graph_sequence) {
  if (threadIdx.x == 0) {
    while (true) {
      const std::uint64_t observed =
          reinterpret_cast<const volatile std::uint64_t*>(address)[0];
      if (observed == sequence ||
          (graph_commands == nullptr && observed > sequence)) {
        break;
      }
      if (graph_commands != nullptr && observed > sequence) {
        graph_publish_overflow(graph_commands, graph_sequence);
        graph_fatal_wait();
      }
      __nanosleep(64);
    }
  }
  __syncthreads();
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

__global__ void tp4_tensor_all_reduce(
    std::uint8_t* round0_buffer, Tp2BufferLayout round0_layout,
    std::uint8_t* round1_buffer, Tp2BufferLayout round1_layout,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    std::size_t payload_bytes, std::uint64_t fixed_sequence,
    Tp4GraphCommandRing* graph_commands, std::uint32_t graph_q,
    bool graph_trace) {
  __shared__ std::uint64_t graph_sequence;
  if (graph_commands != nullptr) {
    if (threadIdx.x == 0) {
      graph_sequence =
          graph_publish_command(
              graph_commands, graph_trace, graph_q,
              static_cast<std::uint32_t>(payload_bytes));
    }
    __syncthreads();
  }
  const std::uint64_t sequence =
      graph_commands == nullptr ? fixed_sequence : graph_sequence;
  const std::uint64_t doorbell_sequence =
      graph_commands == nullptr
          ? sequence
          : (sequence << kTp4GraphDoorbellQBits) | graph_q;

  auto* send0 = reinterpret_cast<__nv_bfloat16*>(
      round0_buffer + round0_layout.send_offset);
  const auto* receive0 = reinterpret_cast<const __nv_bfloat16*>(
      round0_buffer + round0_layout.receive_offset);
  auto* control0 = reinterpret_cast<DoorbellControl*>(
      round0_buffer + round0_layout.control_offset);

  auto* send1 = reinterpret_cast<__nv_bfloat16*>(
      round1_buffer + round1_layout.send_offset);
  const auto* receive1 = reinterpret_cast<const __nv_bfloat16*>(
      round1_buffer + round1_layout.receive_offset);
  auto* control1 = reinterpret_cast<DoorbellControl*>(
      round1_buffer + round1_layout.control_offset);
  const std::size_t elements = payload_bytes / sizeof(__nv_bfloat16);
  const std::size_t pairs = elements / 2;

  for (std::size_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    send0[index] = input[index];
  }
  publish_sequence_block(&control0->producer_sequence, doorbell_sequence);

  wait_for_sequence_block(&control0->remote_sequence, doorbell_sequence,
                          graph_commands, sequence);
  const auto* send0_pairs =
      reinterpret_cast<const __nv_bfloat162*>(send0);
  const auto* receive0_pairs =
      reinterpret_cast<const __nv_bfloat162*>(receive0);
  auto* send1_pairs = reinterpret_cast<__nv_bfloat162*>(send1);
  for (std::size_t index = threadIdx.x; index < pairs;
       index += blockDim.x) {
    send1_pairs[index] =
        __hadd2(send0_pairs[index], receive0_pairs[index]);
  }
  if (elements % 2 != 0 && threadIdx.x == 0) {
    send1[elements - 1] =
        __hadd(send0[elements - 1], receive0[elements - 1]);
  }
  publish_sequence_block(&control0->consumer_sequence,
                         doorbell_sequence);
  wait_for_sequence_block(&control0->acknowledgement_sequence,
                          doorbell_sequence, graph_commands, sequence);

  wait_for_sequence_block(&control1->remote_sequence, doorbell_sequence,
                          graph_commands, sequence);
  const auto* round1_send_pairs =
      reinterpret_cast<const __nv_bfloat162*>(send1);
  const auto* round1_receive_pairs =
      reinterpret_cast<const __nv_bfloat162*>(receive1);
  auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output);
  for (std::size_t index = threadIdx.x; index < pairs;
       index += blockDim.x) {
    output_pairs[index] =
        __hadd2(round1_send_pairs[index], round1_receive_pairs[index]);
  }
  if (elements % 2 != 0 && threadIdx.x == 0) {
    output[elements - 1] =
        __hadd(send1[elements - 1], receive1[elements - 1]);
  }
  publish_sequence_block(&control1->consumer_sequence,
                         doorbell_sequence);
  wait_for_sequence_block(&control1->acknowledgement_sequence,
                          doorbell_sequence, graph_commands, sequence);
  publish_sequence_block(&control1->observed_sequence,
                         doorbell_sequence);
}

}  // namespace

GpuTp4TensorWorker::GpuTp4TensorWorker(
    std::size_t payload_bytes, void* round0_mapped_device_buffer,
    const Tp2BufferLayout& round0_layout,
    void* round1_mapped_device_buffer,
    const Tp2BufferLayout& round1_layout)
    : round0_buffer_(round0_mapped_device_buffer),
      round1_buffer_(round1_mapped_device_buffer),
      round0_layout_(round0_layout),
      round1_layout_(round1_layout),
      payload_bytes_(payload_bytes) {
  if (payload_bytes_ == 0 ||
      payload_bytes_ % sizeof(__nv_bfloat16) != 0) {
    throw std::invalid_argument(
        "TP4 tensor size must be nonzero BF16 data");
  }
  if (round0_buffer_ == nullptr || round1_buffer_ == nullptr) {
    throw std::invalid_argument("TP4 tensor worker received a null buffer");
  }

}

void GpuTp4TensorWorker::enqueue(const void* external_input,
                                 void* external_output, void* cuda_stream,
                                 std::uint64_t sequence) {
  if (external_input == nullptr || external_output == nullptr ||
      sequence == 0) {
    throw std::invalid_argument("invalid TP4 tensor operation");
  }
  const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
  constexpr int threads = 256;
  check_cuda(
      cudaGetLastError(),
      "prior CUDA error before tp4_tensor_all_reduce launch");
  tp4_tensor_all_reduce<<<1, threads, 0, caller_stream>>>(
      static_cast<std::uint8_t*>(round0_buffer_), round0_layout_,
      static_cast<std::uint8_t*>(round1_buffer_), round1_layout_,
      static_cast<const __nv_bfloat16*>(external_input),
      static_cast<__nv_bfloat16*>(external_output), payload_bytes_, sequence,
      nullptr, 0, false);
  check_cuda(cudaGetLastError(), "tp4_tensor_all_reduce launch");
}

void GpuTp4TensorWorker::enqueue_graph(
    const void* external_input, void* external_output, std::uint32_t q,
    void* cuda_stream, Tp4GraphCommandRing* command_ring, bool trace) {
  const std::uint32_t active_payload_bytes =
      tp4_graph_payload_bytes(q);
  if (!tp4_graph_allreduce_command_descriptor_valid(
          q, active_payload_bytes) ||
      active_payload_bytes > payload_bytes_) {
    throw std::invalid_argument(
        "graph TP4 all-reduce requires BF16 [q, 6144], q in [1, 512], "
        "within session capacity");
  }
  if (external_input == nullptr || external_output == nullptr ||
      command_ring == nullptr) {
    throw std::invalid_argument("invalid graph TP4 tensor operation");
  }
  const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
  constexpr int threads = 256;
  tp4_tensor_all_reduce<<<1, threads, 0, caller_stream>>>(
      static_cast<std::uint8_t*>(round0_buffer_), round0_layout_,
      static_cast<std::uint8_t*>(round1_buffer_), round1_layout_,
      static_cast<const __nv_bfloat16*>(external_input),
      static_cast<__nv_bfloat16*>(external_output), active_payload_bytes, 0,
      command_ring, q, trace);
  check_cuda(cudaGetLastError(), "graph tp4_tensor_all_reduce launch");
}

}  // namespace spark_transport
