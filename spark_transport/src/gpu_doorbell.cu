#include "spark_transport/gpu_doorbell.hpp"

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

__device__ std::uint8_t pattern_for(std::uint64_t sequence) {
  return static_cast<std::uint8_t>((sequence * 131U + 0xa5U) & 0xffU);
}

__global__ void sender_doorbell_kernel(std::uint8_t* payload,
                                       std::size_t payload_bytes,
                                       DoorbellControl* control,
                                       std::uint64_t final_sequence) {
  for (std::uint64_t sequence = 1; sequence <= final_sequence; ++sequence) {
    while (reinterpret_cast<volatile std::uint64_t*>(
               &control->command_sequence)[0] < sequence) {
      __nanosleep(64);
    }

    const std::uint8_t pattern = pattern_for(sequence);
    const std::uint32_t word =
        static_cast<std::uint32_t>(pattern) * 0x01010101U;
    const uint4 vector = make_uint4(word, word, word, word);
    auto* vectors = reinterpret_cast<uint4*>(payload);
    const std::size_t vector_count = payload_bytes / sizeof(uint4);
    for (std::size_t index = threadIdx.x; index < vector_count;
         index += blockDim.x) {
      vectors[index] = vector;
    }
    for (std::size_t offset = vector_count * sizeof(uint4) + threadIdx.x;
         offset < payload_bytes; offset += blockDim.x) {
      payload[offset] = pattern;
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) {
      reinterpret_cast<volatile std::uint64_t*>(
          &control->producer_sequence)[0] = sequence;
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

__global__ void receiver_doorbell_kernel(std::uint8_t* payload,
                                         std::size_t payload_bytes,
                                         DoorbellControl* control,
                                         std::uint64_t final_sequence,
                                         bool verify_payload) {
  __shared__ unsigned int mismatch;
  for (std::uint64_t sequence = 1; sequence <= final_sequence; ++sequence) {
    while (reinterpret_cast<volatile std::uint64_t*>(
               &control->remote_sequence)[0] < sequence) {
      __nanosleep(64);
    }

    if (threadIdx.x == 0) {
      mismatch = 0;
    }
    __syncthreads();
    if (verify_payload) {
      const std::uint8_t expected = pattern_for(sequence);
      const std::uint32_t word =
          static_cast<std::uint32_t>(expected) * 0x01010101U;
      const uint4 expected_vector = make_uint4(word, word, word, word);
      const auto* vectors = reinterpret_cast<const uint4*>(payload);
      const std::size_t vector_count = payload_bytes / sizeof(uint4);
      for (std::size_t index = threadIdx.x; index < vector_count;
           index += blockDim.x) {
        const uint4 actual = vectors[index];
        if (actual.x != expected_vector.x ||
            actual.y != expected_vector.y ||
            actual.z != expected_vector.z ||
            actual.w != expected_vector.w) {
          atomicOr(&mismatch, 1U);
        }
      }
      for (std::size_t offset = vector_count * sizeof(uint4) + threadIdx.x;
           offset < payload_bytes; offset += blockDim.x) {
        if (payload[offset] != expected) {
          atomicOr(&mismatch, 1U);
        }
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
  }
}

}  // namespace

std::size_t aligned_control_offset(std::size_t payload_bytes) {
  constexpr std::size_t alignment = alignof(DoorbellControl);
  return (payload_bytes + alignment - 1) & ~(alignment - 1);
}

ExchangeBufferLayout make_exchange_buffer_layout(
    std::size_t payload_bytes) {
  if (payload_bytes == 0 ||
      payload_bytes % sizeof(std::uint16_t) != 0) {
    throw std::invalid_argument(
        "exchange payload must contain a nonzero whole number of "
        "16-bit elements");
  }

  ExchangeBufferLayout layout{};
  layout.send_offset = 0;
  layout.receive_offset = aligned_control_offset(payload_bytes);
  layout.control_offset =
      aligned_control_offset(layout.receive_offset + payload_bytes);
  layout.total_bytes = layout.control_offset + sizeof(DoorbellControl);
  return layout;
}

void launch_sender_doorbell(void* device_buffer, std::size_t payload_bytes,
                            std::size_t control_offset,
                            std::uint64_t final_sequence) {
  auto* bytes = static_cast<std::uint8_t*>(device_buffer);
  auto* control =
      reinterpret_cast<DoorbellControl*>(bytes + control_offset);
  sender_doorbell_kernel<<<1, 256>>>(bytes, payload_bytes, control,
                                    final_sequence);
  check_cuda(cudaGetLastError(), "sender_doorbell_kernel launch");
}

void launch_receiver_doorbell(void* device_buffer, std::size_t payload_bytes,
                              std::size_t control_offset,
                              std::uint64_t final_sequence,
                              bool verify_payload) {
  auto* bytes = static_cast<std::uint8_t*>(device_buffer);
  auto* control =
      reinterpret_cast<DoorbellControl*>(bytes + control_offset);
  receiver_doorbell_kernel<<<1, 1024>>>(bytes, payload_bytes, control,
                                       final_sequence, verify_payload);
  check_cuda(cudaGetLastError(), "receiver_doorbell_kernel launch");
}

void synchronize_doorbell() {
  check_cuda(cudaDeviceSynchronize(), "doorbell kernel synchronize");
}

}  // namespace spark_transport
