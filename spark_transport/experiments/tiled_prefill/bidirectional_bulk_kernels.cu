#include "bidirectional_bulk_kernels.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace spark_transport::tiled_prefill_research {
namespace {

constexpr unsigned int kWorkerThreads = 256;
constexpr std::size_t kPacketBytes = 16;

struct alignas(16) Bf16Packet {
  __nv_bfloat162 pairs[4];
};

struct U32Packet {
  std::uint32_t words[4];
};

static_assert(sizeof(Bf16Packet) == kPacketBytes);

__device__ U32Packet load_relaxed_system(const void* address) {
  U32Packet result{};
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(result.words[0]), "=r"(result.words[1]),
                 "=r"(result.words[2]), "=r"(result.words[3])
               : "l"(address)
               : "memory");
  return result;
}

__device__ std::uint64_t load_acquire_system_u64(const void* address) {
  std::uint64_t result{};
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(result)
               : "l"(address)
               : "memory");
  return result;
}

__device__ void require_inbound_doorbell(
    const std::uint8_t* incoming,
    const BidirectionalBulkDescriptor& descriptor) {
  if (threadIdx.x == 0) {
    if (descriptor.expected_doorbell_token == 0 ||
        descriptor.inbound_doorbell_offset_bytes % alignof(std::uint64_t) !=
            0) {
      asm volatile("trap;");
    }
    const auto observed = load_acquire_system_u64(
        incoming + descriptor.inbound_doorbell_offset_bytes);
    if (observed != descriptor.expected_doorbell_token) {
      asm volatile("trap;");
    }
    if (descriptor.secondary_expected_doorbell_token != 0) {
      if (descriptor.secondary_inbound_doorbell_offset_bytes %
              alignof(std::uint64_t) != 0) {
        asm volatile("trap;");
      }
      const auto secondary = load_acquire_system_u64(
          incoming + descriptor.secondary_inbound_doorbell_offset_bytes);
      if (secondary != descriptor.secondary_expected_doorbell_token) {
        asm volatile("trap;");
      }
    }
  }
  __syncthreads();
}

__device__ Bf16Packet as_bf16(U32Packet value) {
  union PacketBits {
    U32Packet words;
    Bf16Packet bf16;
  } bits{};
  bits.words = value;
  return bits.bf16;
}

__device__ void copy_packet(const std::uint8_t* source,
                            std::uint8_t* destination,
                            std::size_t byte) {
  reinterpret_cast<Bf16Packet*>(destination + byte)[0] =
      reinterpret_cast<const Bf16Packet*>(source + byte)[0];
}

__device__ void copy_system_packet(const std::uint8_t* source,
                                   std::uint8_t* destination,
                                   std::size_t byte) {
  const U32Packet packet = load_relaxed_system(source + byte);
  reinterpret_cast<U32Packet*>(destination + byte)[0] = packet;
}

__device__ Bf16Packet reduce_packet(const Bf16Packet& left,
                                    const Bf16Packet& right) {
  Bf16Packet result{};
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    result.pairs[pair] = __hadd2(left.pairs[pair], right.pairs[pair]);
  }
  return result;
}

__device__ bool descriptor_valid(const BidirectionalBulkDescriptor& descriptor,
                                 std::uint32_t first_stage,
                                 std::uint32_t last_stage) {
  const bool supported_query =
      descriptor.query_rows == 1024 || descriptor.query_rows == 2048 ||
      descriptor.query_rows == 4096 || descriptor.query_rows == 8192;
  const std::uint64_t payload_bytes =
      static_cast<std::uint64_t>(descriptor.query_rows) *
      kBidirectionalBulkElementsPerRow * kTp4PrefillBf16Bytes;
  const std::uint64_t half_bytes = payload_bytes / 2U;
  const std::uint64_t shard_bytes = half_bytes / kTp4PrefillRankCount;
  const std::uint64_t tile_bytes =
      shard_bytes / kBidirectionalBulkTilesPerShard;
  const std::uint64_t expected_offset =
      static_cast<std::uint64_t>(descriptor.half) * half_bytes +
      static_cast<std::uint64_t>(descriptor.shard) * shard_bytes +
      static_cast<std::uint64_t>(descriptor.tile_in_shard) * tile_bytes;
  const bool expected_half =
      (descriptor.direction ==
           static_cast<std::int32_t>(Tp4PrefillDirection::kClockwise) &&
       descriptor.half == static_cast<std::uint32_t>(Tp4PrefillHalf::kLower)) ||
      (descriptor.direction == static_cast<std::int32_t>(
                                   Tp4PrefillDirection::kCounterClockwise) &&
       descriptor.half == static_cast<std::uint32_t>(Tp4PrefillHalf::kUpper));
  return supported_query &&
         descriptor.elements_per_row == kBidirectionalBulkElementsPerRow &&
         descriptor.active_bytes == tile_bytes &&
         descriptor.active_bytes % kBidirectionalBulkCtaBytes == 0 &&
         descriptor.active_bytes % kPacketBytes == 0 &&
         descriptor.rank < kTp4PrefillRankCount &&
         descriptor.stage >= first_stage && descriptor.stage <= last_stage &&
         descriptor.shard < kTp4PrefillRankCount &&
         descriptor.tile_in_shard < kBidirectionalBulkTilesPerShard &&
         expected_half && descriptor.tensor_offset_bytes == expected_offset;
}

__device__ std::size_t block_begin() {
  return static_cast<std::size_t>(blockIdx.x) *
         kBidirectionalBulkCtaBytes;
}

__device__ std::size_t block_end(
    const BidirectionalBulkDescriptor& descriptor) {
  const std::size_t candidate = block_begin() + kBidirectionalBulkCtaBytes;
  return candidate < descriptor.active_bytes ? candidate
                                              : descriptor.active_bytes;
}

unsigned int launch_blocks(std::uint32_t query_rows) {
  if (!bidirectional_bulk_query_rows_supported(query_rows)) return 0;
  return bidirectional_bulk_tile_bytes(query_rows) /
         kBidirectionalBulkCtaBytes;
}

__global__ void stage_initial_kernel(
    const std::uint8_t* input, std::uint8_t* outgoing,
    BidirectionalBulkDescriptor descriptor) {
  if (!descriptor_valid(descriptor, 0, 0)) return;
  for (std::size_t byte = block_begin() + threadIdx.x * kPacketBytes;
       byte < block_end(descriptor); byte += blockDim.x * kPacketBytes) {
    copy_packet(input + descriptor.tensor_offset_bytes,
                outgoing + descriptor.send_offset_bytes, byte);
  }
  __threadfence_system();
}

__global__ void reduce_forward_kernel(
    const std::uint8_t* input, const std::uint8_t* incoming,
    std::uint8_t* outgoing,
    BidirectionalBulkDescriptor descriptor) {
  if (!descriptor_valid(descriptor, 0, 1)) return;
  require_inbound_doorbell(incoming, descriptor);
  for (std::size_t byte = block_begin() + threadIdx.x * kPacketBytes;
       byte < block_end(descriptor); byte += blockDim.x * kPacketBytes) {
    const auto local = reinterpret_cast<const Bf16Packet*>(
        input + descriptor.tensor_offset_bytes + byte)[0];
    const auto remote = as_bf16(load_relaxed_system(
        incoming + descriptor.receive_offset_bytes + byte));
    reinterpret_cast<Bf16Packet*>(
        outgoing + descriptor.send_offset_bytes + byte)[0] =
        reduce_packet(remote, local);
  }
  __threadfence_system();
}

__global__ void reduce_finalize_seed_gather_kernel(
    const std::uint8_t* input, const std::uint8_t* incoming,
    std::uint8_t* outgoing, std::uint8_t* output,
    BidirectionalBulkDescriptor descriptor) {
  if (!descriptor_valid(descriptor, 2, 2)) return;
  require_inbound_doorbell(incoming, descriptor);
  for (std::size_t byte = block_begin() + threadIdx.x * kPacketBytes;
       byte < block_end(descriptor); byte += blockDim.x * kPacketBytes) {
    const auto local = reinterpret_cast<const Bf16Packet*>(
        input + descriptor.tensor_offset_bytes + byte)[0];
    const auto remote = as_bf16(load_relaxed_system(
        incoming + descriptor.receive_offset_bytes + byte));
    const auto reduced = reduce_packet(remote, local);
    reinterpret_cast<Bf16Packet*>(
        output + descriptor.tensor_offset_bytes + byte)[0] = reduced;
    reinterpret_cast<Bf16Packet*>(
        outgoing + descriptor.send_offset_bytes + byte)[0] = reduced;
  }
  __threadfence_system();
}

__global__ void gather_forward_kernel(
    const std::uint8_t* incoming, std::uint8_t* outgoing,
    std::uint8_t* output,
    BidirectionalBulkDescriptor descriptor) {
  if (!descriptor_valid(descriptor, 3, 4)) return;
  require_inbound_doorbell(incoming, descriptor);
  for (std::size_t byte = block_begin() + threadIdx.x * kPacketBytes;
       byte < block_end(descriptor); byte += blockDim.x * kPacketBytes) {
    const auto packet = load_relaxed_system(
        incoming + descriptor.receive_offset_bytes + byte);
    reinterpret_cast<U32Packet*>(
        output + descriptor.tensor_offset_bytes + byte)[0] = packet;
    reinterpret_cast<U32Packet*>(
        outgoing + descriptor.send_offset_bytes + byte)[0] = packet;
  }
  __threadfence_system();
}

__global__ void gather_finish_kernel(
    const std::uint8_t* incoming, std::uint8_t* output,
    BidirectionalBulkDescriptor descriptor) {
  if (!descriptor_valid(descriptor, 5, 5)) return;
  require_inbound_doorbell(incoming, descriptor);
  for (std::size_t byte = block_begin() + threadIdx.x * kPacketBytes;
       byte < block_end(descriptor); byte += blockDim.x * kPacketBytes) {
    copy_system_packet(incoming + descriptor.receive_offset_bytes,
                       output + descriptor.tensor_offset_bytes, byte);
  }
}

bool launch_arguments_valid(const void* first, const void* second) {
  return first != nullptr && second != nullptr;
}

}  // namespace

cudaError_t launch_bidirectional_stage_initial(
    const std::uint8_t* input, std::uint8_t* outgoing_endpoint,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows) {
  const auto blocks = launch_blocks(query_rows);
  if (!launch_arguments_valid(input, outgoing_endpoint) || blocks == 0) {
    return cudaErrorInvalidValue;
  }
  stage_initial_kernel<<<blocks, kWorkerThreads, 0, stream>>>(
      input, outgoing_endpoint, descriptor);
  return cudaGetLastError();
}

cudaError_t launch_bidirectional_reduce_forward(
    const std::uint8_t* input, const std::uint8_t* incoming_endpoint,
    std::uint8_t* outgoing_endpoint,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows) {
  const auto blocks = launch_blocks(query_rows);
  if (!launch_arguments_valid(input, incoming_endpoint) ||
      outgoing_endpoint == nullptr || blocks == 0) {
    return cudaErrorInvalidValue;
  }
  reduce_forward_kernel<<<blocks, kWorkerThreads, 0, stream>>>(
      input, incoming_endpoint, outgoing_endpoint, descriptor);
  return cudaGetLastError();
}

cudaError_t launch_bidirectional_reduce_finalize_seed_gather(
    const std::uint8_t* input, const std::uint8_t* incoming_endpoint,
    std::uint8_t* outgoing_endpoint, std::uint8_t* output,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows) {
  const auto blocks = launch_blocks(query_rows);
  if (!launch_arguments_valid(input, incoming_endpoint) ||
      outgoing_endpoint == nullptr || output == nullptr || blocks == 0) {
    return cudaErrorInvalidValue;
  }
  reduce_finalize_seed_gather_kernel<<<blocks, kWorkerThreads, 0, stream>>>(
      input, incoming_endpoint, outgoing_endpoint, output, descriptor);
  return cudaGetLastError();
}

cudaError_t launch_bidirectional_gather_forward(
    const std::uint8_t* incoming_endpoint, std::uint8_t* outgoing_endpoint,
    std::uint8_t* output, const BidirectionalBulkDescriptor& descriptor,
    cudaStream_t stream, std::uint32_t query_rows) {
  const auto blocks = launch_blocks(query_rows);
  if (!launch_arguments_valid(incoming_endpoint, outgoing_endpoint) ||
      output == nullptr || blocks == 0) {
    return cudaErrorInvalidValue;
  }
  gather_forward_kernel<<<blocks, kWorkerThreads, 0, stream>>>(
      incoming_endpoint, outgoing_endpoint, output, descriptor);
  return cudaGetLastError();
}

cudaError_t launch_bidirectional_gather_finish(
    const std::uint8_t* incoming_endpoint, std::uint8_t* output,
    const BidirectionalBulkDescriptor& descriptor, cudaStream_t stream,
    std::uint32_t query_rows) {
  const auto blocks = launch_blocks(query_rows);
  if (!launch_arguments_valid(incoming_endpoint, output) ||
      blocks == 0) {
    return cudaErrorInvalidValue;
  }
  gather_finish_kernel<<<blocks, kWorkerThreads, 0, stream>>>(
      incoming_endpoint, output, descriptor);
  return cudaGetLastError();
}

}  // namespace spark_transport::tiled_prefill_research
