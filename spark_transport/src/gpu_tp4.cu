#include "spark_transport/gpu_tp4.hpp"

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

__device__ float tp4_value(std::uint32_t parity, std::uint32_t rank,
                           std::size_t element) {
  return static_cast<float>(
      ((parity + element * 3U + rank * 5U) & 7U) + 1U);
}

__global__ void initialize_tp4_inputs(__nv_bfloat16* input,
                                      std::size_t elements,
                                      std::uint32_t rank) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= elements) {
    return;
  }
  input[index] = __float2bfloat16(tp4_value(0, rank, index));
  input[elements + index] =
      __float2bfloat16(tp4_value(1, rank, index));
}

__device__ void wait_for_sequence(const std::uint64_t* address,
                                  std::uint64_t sequence) {
  while (reinterpret_cast<const volatile std::uint64_t*>(address)[0] <
         sequence) {
    __nanosleep(64);
  }
}

__device__ void publish_sequence(std::uint64_t* address,
                                 std::uint64_t sequence) {
  __threadfence_system();
  reinterpret_cast<volatile std::uint64_t*>(address)[0] = sequence;
}

__global__ void tp4_worker_kernel(
    std::uint8_t* round0_buffer, Tp2BufferLayout round0_layout,
    std::uint8_t* round1_buffer, Tp2BufferLayout round1_layout,
    const __nv_bfloat16* input, __nv_bfloat16* intermediate,
    __nv_bfloat16* output, std::size_t payload_bytes, std::uint32_t rank,
    std::uint64_t final_sequence) {
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
  __shared__ unsigned int mismatch;

  for (std::uint64_t sequence = 1; sequence <= final_sequence; ++sequence) {
    wait_for_sequence(&control0->command_sequence, sequence);

    const std::uint32_t parity = static_cast<std::uint32_t>(sequence & 1U);
    const auto* current_input = input + parity * elements;
    const auto* input_pairs =
        reinterpret_cast<const __nv_bfloat162*>(current_input);
    auto* send0_pairs = reinterpret_cast<__nv_bfloat162*>(send0);
    for (std::size_t index = threadIdx.x; index < pairs;
         index += blockDim.x) {
      send0_pairs[index] = input_pairs[index];
    }
    if (elements % 2 != 0 && threadIdx.x == 0) {
      send0[elements - 1] = current_input[elements - 1];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      publish_sequence(&control0->producer_sequence, sequence);
    }
    __syncthreads();

    wait_for_sequence(&control0->remote_sequence, sequence);
    const auto* receive0_pairs =
        reinterpret_cast<const __nv_bfloat162*>(receive0);
    auto* intermediate_pairs =
        reinterpret_cast<__nv_bfloat162*>(intermediate);
    auto* send1_pairs = reinterpret_cast<__nv_bfloat162*>(send1);
    for (std::size_t index = threadIdx.x; index < pairs;
         index += blockDim.x) {
      const __nv_bfloat162 pair_sum =
          __hadd2(send0_pairs[index], receive0_pairs[index]);
      intermediate_pairs[index] = pair_sum;
      send1_pairs[index] = pair_sum;
    }
    if (elements % 2 != 0 && threadIdx.x == 0) {
      const __nv_bfloat16 pair_sum =
          __hadd(send0[elements - 1], receive0[elements - 1]);
      intermediate[elements - 1] = pair_sum;
      send1[elements - 1] = pair_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      publish_sequence(&control0->consumer_sequence, sequence);
    }
    __syncthreads();

    wait_for_sequence(&control0->acknowledgement_sequence, sequence);
    if (threadIdx.x == 0) {
      publish_sequence(&control0->observed_sequence, sequence);
    }
    __syncthreads();

    wait_for_sequence(&control1->command_sequence, sequence);
    if (threadIdx.x == 0) {
      publish_sequence(&control1->producer_sequence, sequence);
    }
    __syncthreads();

    wait_for_sequence(&control1->remote_sequence, sequence);
    if (threadIdx.x == 0) {
      mismatch = 0;
    }
    __syncthreads();
    const auto* receive1_pairs =
        reinterpret_cast<const __nv_bfloat162*>(receive1);
    auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output);
    for (std::size_t index = threadIdx.x; index < pairs;
         index += blockDim.x) {
      const __nv_bfloat162 total =
          __hadd2(send1_pairs[index], receive1_pairs[index]);
      output_pairs[index] = total;
      const float2 actual = __bfloat1622float2(total);
      const std::size_t element = index * 2;
      float expected_x = 0.0F;
      float expected_y = 0.0F;
      for (std::uint32_t source_rank = 0; source_rank < 4; ++source_rank) {
        expected_x += tp4_value(parity, source_rank, element);
        expected_y += tp4_value(parity, source_rank, element + 1);
      }
      if (actual.x != expected_x || actual.y != expected_y) {
        atomicOr(&mismatch, 1U);
      }
    }
    if (elements % 2 != 0 && threadIdx.x == 0) {
      const __nv_bfloat16 total =
          __hadd(send1[elements - 1], receive1[elements - 1]);
      output[elements - 1] = total;
      float expected = 0.0F;
      for (std::uint32_t source_rank = 0; source_rank < 4; ++source_rank) {
        expected += tp4_value(parity, source_rank, elements - 1);
      }
      if (__bfloat162float(total) != expected) {
        atomicOr(&mismatch, 1U);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (mismatch != 0) {
        ++control1->mismatch_count;
      }
      publish_sequence(&control1->consumer_sequence, sequence);
    }
    __syncthreads();

    wait_for_sequence(&control1->acknowledgement_sequence, sequence);
    if (threadIdx.x == 0) {
      publish_sequence(&control1->observed_sequence, sequence);
    }
    __syncthreads();
  }
}

}  // namespace

GpuTp4Worker::GpuTp4Worker(std::size_t payload_bytes)
    : payload_bytes_(payload_bytes) {
  if (payload_bytes_ == 0 ||
      payload_bytes_ % sizeof(__nv_bfloat16) != 0) {
    throw std::invalid_argument("TP4 output size must be nonzero BF16 data");
  }

  try {
    check_cuda(cudaMalloc(&input_, payload_bytes_ * 2),
               "cudaMalloc TP4 input");
    check_cuda(cudaMalloc(&intermediate_, payload_bytes_),
               "cudaMalloc TP4 intermediate");
    check_cuda(cudaMalloc(&output_, payload_bytes_),
               "cudaMalloc TP4 output");
  } catch (...) {
    if (input_ != nullptr) {
      cudaFree(input_);
      input_ = nullptr;
    }
    if (intermediate_ != nullptr) {
      cudaFree(intermediate_);
      intermediate_ = nullptr;
    }
    if (output_ != nullptr) {
      cudaFree(output_);
      output_ = nullptr;
    }
    throw;
  }
}

GpuTp4Worker::~GpuTp4Worker() {
  if (input_ != nullptr) {
    cudaFree(input_);
  }
  if (intermediate_ != nullptr) {
    cudaFree(intermediate_);
  }
  if (output_ != nullptr) {
    cudaFree(output_);
  }
}

void GpuTp4Worker::launch(void* round0_mapped_device_buffer,
                          const Tp2BufferLayout& round0_layout,
                          void* round1_mapped_device_buffer,
                          const Tp2BufferLayout& round1_layout,
                          std::uint32_t rank,
                          std::uint64_t final_sequence) {
  if (rank >= 4) {
    throw std::invalid_argument("TP4 rank must be in [0, 3]");
  }
  const std::size_t elements = payload_bytes_ / sizeof(__nv_bfloat16);
  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((elements + threads - 1) / threads);
  initialize_tp4_inputs<<<blocks, threads>>>(
      static_cast<__nv_bfloat16*>(input_), elements, rank);
  check_cuda(cudaGetLastError(), "initialize_tp4_inputs launch");
  check_cuda(cudaDeviceSynchronize(), "initialize_tp4_inputs synchronize");

  tp4_worker_kernel<<<1, threads>>>(
      static_cast<std::uint8_t*>(round0_mapped_device_buffer), round0_layout,
      static_cast<std::uint8_t*>(round1_mapped_device_buffer), round1_layout,
      static_cast<const __nv_bfloat16*>(input_),
      static_cast<__nv_bfloat16*>(intermediate_),
      static_cast<__nv_bfloat16*>(output_), payload_bytes_, rank,
      final_sequence);
  check_cuda(cudaGetLastError(), "tp4_worker_kernel launch");
}

void GpuTp4Worker::synchronize() {
  check_cuda(cudaDeviceSynchronize(), "TP4 worker synchronize");
}

}  // namespace spark_transport
