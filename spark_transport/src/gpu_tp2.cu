#include "spark_transport/gpu_tp2.hpp"

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

__device__ float tp2_value(std::uint32_t parity, std::uint32_t rank,
                           std::size_t element) {
  return static_cast<float>(
      ((parity + element * 3U + rank * 5U) & 7U) + 1U);
}

__global__ void initialize_tp2_inputs(__nv_bfloat16* input,
                                      std::size_t elements,
                                      std::uint32_t rank) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= elements) {
    return;
  }
  input[index] = __float2bfloat16(tp2_value(0, rank, index));
  input[elements + index] =
      __float2bfloat16(tp2_value(1, rank, index));
}

__global__ void tp2_worker_kernel(
    std::uint8_t* mapped_buffer, const __nv_bfloat16* input,
    __nv_bfloat16* output,
    std::size_t payload_bytes, Tp2BufferLayout layout, std::uint32_t rank,
    std::uint64_t final_sequence) {
  auto* send = reinterpret_cast<__nv_bfloat16*>(
      mapped_buffer + layout.send_offset);
  const auto* receive = reinterpret_cast<const __nv_bfloat16*>(
      mapped_buffer + layout.receive_offset);
  auto* control = reinterpret_cast<DoorbellControl*>(
      mapped_buffer + layout.control_offset);
  const std::size_t elements = payload_bytes / sizeof(__nv_bfloat16);
  const std::size_t pairs = elements / 2;

  __shared__ unsigned int mismatch;
  for (std::uint64_t sequence = 1; sequence <= final_sequence; ++sequence) {
    while (reinterpret_cast<volatile std::uint64_t*>(
               &control->command_sequence)[0] < sequence) {
      __nanosleep(64);
    }

    const std::uint32_t parity = static_cast<std::uint32_t>(sequence & 1U);
    const auto* current_input = input + parity * elements;
    const auto* input_pairs =
        reinterpret_cast<const __nv_bfloat162*>(current_input);
    auto* send_pairs = reinterpret_cast<__nv_bfloat162*>(send);
    for (std::size_t index = threadIdx.x; index < pairs;
         index += blockDim.x) {
      send_pairs[index] = input_pairs[index];
    }
    if (elements % 2 != 0 && threadIdx.x == 0) {
      send[elements - 1] = current_input[elements - 1];
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) {
      reinterpret_cast<volatile std::uint64_t*>(
          &control->producer_sequence)[0] = sequence;
    }
    __syncthreads();

    while (reinterpret_cast<volatile std::uint64_t*>(
               &control->remote_sequence)[0] < sequence) {
      __nanosleep(64);
    }

    if (threadIdx.x == 0) {
      mismatch = 0;
    }
    __syncthreads();
    const auto* send_pair =
        reinterpret_cast<const __nv_bfloat162*>(send);
    const auto* receive_pair =
        reinterpret_cast<const __nv_bfloat162*>(receive);
    auto* output_pair = reinterpret_cast<__nv_bfloat162*>(output);
    for (std::size_t index = threadIdx.x; index < pairs;
         index += blockDim.x) {
      const __nv_bfloat162 sum = __hadd2(send_pair[index],
                                        receive_pair[index]);
      output_pair[index] = sum;
      const float2 actual = __bfloat1622float2(sum);
      const std::size_t element = index * 2;
      const float expected_x = tp2_value(parity, 0, element) +
                               tp2_value(parity, 1, element);
      const float expected_y = tp2_value(parity, 0, element + 1) +
                               tp2_value(parity, 1, element + 1);
      if (actual.x != expected_x || actual.y != expected_y) {
        atomicOr(&mismatch, 1U);
      }
    }
    if (elements % 2 != 0 && threadIdx.x == 0) {
      const __nv_bfloat16 sum =
          __hadd(send[elements - 1], receive[elements - 1]);
      output[elements - 1] = sum;
      const float expected = tp2_value(parity, 0, elements - 1) +
                             tp2_value(parity, 1, elements - 1);
      if (__bfloat162float(sum) != expected) {
        atomicOr(&mismatch, 1U);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (mismatch != 0) {
        ++control->mismatch_count;
      }
      __threadfence_system();
      reinterpret_cast<volatile std::uint64_t*>(
          &control->consumer_sequence)[0] = sequence;
    }
    __syncthreads();

    while (reinterpret_cast<volatile std::uint64_t*>(
               &control->acknowledgement_sequence)[0] < sequence) {
      __nanosleep(64);
    }
    if (threadIdx.x == 0) {
      reinterpret_cast<volatile std::uint64_t*>(
          &control->observed_sequence)[0] = sequence;
    }
    __syncthreads();
  }
}

}  // namespace

Tp2BufferLayout make_tp2_buffer_layout(std::size_t payload_bytes) {
  if (payload_bytes == 0 ||
      payload_bytes % sizeof(__nv_bfloat16) != 0) {
    throw std::invalid_argument(
        "TP2 payload must contain a nonzero whole number of BF16 elements");
  }

  Tp2BufferLayout layout{};
  layout.send_offset = 0;
  layout.receive_offset = aligned_control_offset(payload_bytes);
  layout.control_offset =
      aligned_control_offset(layout.receive_offset + payload_bytes);
  layout.total_bytes = layout.control_offset + sizeof(DoorbellControl);
  return layout;
}

GpuTp2Worker::GpuTp2Worker(std::size_t payload_bytes)
    : payload_bytes_(payload_bytes) {
  if (payload_bytes_ == 0 ||
      payload_bytes_ % sizeof(__nv_bfloat16) != 0) {
    throw std::invalid_argument("TP2 output size must be nonzero BF16 data");
  }
  check_cuda(cudaMalloc(&input_, payload_bytes_ * 2),
             "cudaMalloc TP2 input");
  try {
    check_cuda(cudaMalloc(&output_, payload_bytes_), "cudaMalloc TP2 output");
  } catch (...) {
    cudaFree(input_);
    input_ = nullptr;
    throw;
  }
}

GpuTp2Worker::~GpuTp2Worker() {
  if (input_ != nullptr) {
    cudaFree(input_);
  }
  if (output_ != nullptr) {
    cudaFree(output_);
  }
}

void GpuTp2Worker::launch(void* mapped_device_buffer,
                          const Tp2BufferLayout& layout, std::uint32_t rank,
                          std::uint64_t final_sequence) {
  if (rank > 1) {
    throw std::invalid_argument("TP2 rank must be zero or one");
  }
  auto* bytes = static_cast<std::uint8_t*>(mapped_device_buffer);
  const std::size_t elements = payload_bytes_ / sizeof(__nv_bfloat16);
  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((elements + threads - 1) / threads);
  initialize_tp2_inputs<<<blocks, threads>>>(
      static_cast<__nv_bfloat16*>(input_), elements, rank);
  check_cuda(cudaGetLastError(), "initialize_tp2_inputs launch");
  check_cuda(cudaDeviceSynchronize(), "initialize_tp2_inputs synchronize");

  tp2_worker_kernel<<<1, 256>>>(
      bytes, static_cast<const __nv_bfloat16*>(input_),
      static_cast<__nv_bfloat16*>(output_), payload_bytes_, layout, rank,
      final_sequence);
  check_cuda(cudaGetLastError(), "tp2_worker_kernel launch");
}

void GpuTp2Worker::synchronize() {
  check_cuda(cudaDeviceSynchronize(), "TP2 worker synchronize");
}

}  // namespace spark_transport
