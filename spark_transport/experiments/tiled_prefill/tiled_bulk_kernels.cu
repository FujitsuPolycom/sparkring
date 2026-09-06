// Research-only bulk CUDA kernels for the tiled TP4 prefill contract.
//
// These kernels deliberately exclude doorbells, credits, RDMA progress, and
// production session wiring.  An executor must place each launch behind
// the dependencies emitted by gpu_harness.py.  Worker CTAs never poll a
// protocol flag: readiness must be established by a CUDA graph edge or event
// before launch, avoiding an occupancy-dependent polling deadlock.  Every
// wrapper launches exactly one descriptor as eight CTAs; batching descriptors
// here would bypass the per-slot credit edges required after eight tiles.

#include "tiled_bulk_kernels.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace spark_transport::tiled_prefill_research {
namespace {

constexpr std::size_t kPayloadTileBytes = 512U * 1024U;
constexpr std::size_t kWorkerCtaBytes = 64U * 1024U;
constexpr unsigned int kWorkerCtasPerTile =
    static_cast<unsigned int>(kPayloadTileBytes / kWorkerCtaBytes);
constexpr unsigned int kWorkerCtasPerStripe = kWorkerCtasPerTile / 2U;
constexpr unsigned int kWorkerThreads = 256U;
constexpr std::size_t kPacketBytes = 16U;

struct alignas(16) Bf16Packet {
  __nv_bfloat162 pairs[4];
};

static_assert(sizeof(Bf16Packet) == kPacketBytes);
static_assert(kWorkerCtasPerTile == 8U);

struct ActiveSpan {
  std::size_t stripe;
  std::size_t begin;
  std::size_t end;
  bool active;
};

__device__ ActiveSpan active_span(std::uint32_t active_bytes) {
  const unsigned int local_cta = blockIdx.x;
  const std::size_t stripe = local_cta / kWorkerCtasPerStripe;
  const std::size_t stripe_worker = local_cta % kWorkerCtasPerStripe;
  const std::size_t stripe_active_bytes = active_bytes / 2U;
  const std::size_t begin = stripe_worker * kWorkerCtaBytes;
  const std::size_t candidate_end = begin + kWorkerCtaBytes;
  const std::size_t end =
      candidate_end < stripe_active_bytes ? candidate_end
                                          : stripe_active_bytes;
  return {stripe, begin, end, begin < end};
}

__device__ void copy_active_bf16(const std::uint8_t* source,
                                 std::uint8_t* destination,
                                 std::size_t begin,
                                 std::size_t end) {
  const std::size_t packet_end = end - end % kPacketBytes;
  for (std::size_t byte = begin + threadIdx.x * kPacketBytes;
       byte < packet_end;
       byte += blockDim.x * kPacketBytes) {
    reinterpret_cast<Bf16Packet*>(destination + byte)[0] =
        reinterpret_cast<const Bf16Packet*>(source + byte)[0];
  }
  const std::size_t tail_begin =
      packet_end < begin ? begin : packet_end;
  for (std::size_t byte =
           tail_begin + threadIdx.x * sizeof(__nv_bfloat16);
       byte < end;
       byte += blockDim.x * sizeof(__nv_bfloat16)) {
    reinterpret_cast<__nv_bfloat16*>(destination + byte)[0] =
        reinterpret_cast<const __nv_bfloat16*>(source + byte)[0];
  }
}

__device__ void reduce_active_bf16(const std::uint8_t* left,
                                   const std::uint8_t* right,
                                   std::uint8_t* output,
                                   std::size_t begin,
                                   std::size_t end) {
  const std::size_t packet_end = end - end % kPacketBytes;
  for (std::size_t byte = begin + threadIdx.x * kPacketBytes;
       byte < packet_end;
       byte += blockDim.x * kPacketBytes) {
    const Bf16Packet left_packet =
        reinterpret_cast<const Bf16Packet*>(left + byte)[0];
    const Bf16Packet right_packet =
        reinterpret_cast<const Bf16Packet*>(right + byte)[0];
    Bf16Packet result{};
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      result.pairs[pair] =
          __hadd2(left_packet.pairs[pair], right_packet.pairs[pair]);
    }
    reinterpret_cast<Bf16Packet*>(output + byte)[0] = result;
  }
  const std::size_t tail_begin =
      packet_end < begin ? begin : packet_end;
  const std::size_t pair_end =
      end - (end - tail_begin) % sizeof(__nv_bfloat162);
  for (std::size_t byte =
           tail_begin + threadIdx.x * sizeof(__nv_bfloat162);
       byte < pair_end;
       byte += blockDim.x * sizeof(__nv_bfloat162)) {
    reinterpret_cast<__nv_bfloat162*>(output + byte)[0] = __hadd2(
        reinterpret_cast<const __nv_bfloat162*>(left + byte)[0],
        reinterpret_cast<const __nv_bfloat162*>(right + byte)[0]);
  }
  for (std::size_t byte =
           pair_end + threadIdx.x * sizeof(__nv_bfloat16);
       byte < end;
       byte += blockDim.x * sizeof(__nv_bfloat16)) {
    reinterpret_cast<__nv_bfloat16*>(output + byte)[0] = __hadd(
        reinterpret_cast<const __nv_bfloat16*>(left + byte)[0],
        reinterpret_cast<const __nv_bfloat16*>(right + byte)[0]);
  }
}

}  // namespace

__global__ void stage_phase1_tile(
    const std::uint8_t* input, std::uint8_t* endpoint0,
    std::uint8_t* endpoint1, const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout) {
  const TiledBulkDescriptor descriptor = descriptor_pointer[0];
  if (descriptor.active_bytes == 0U ||
      descriptor.active_bytes > kPayloadTileBytes ||
      descriptor.active_bytes % sizeof(__nv_bfloat162) != 0U) {
    return;
  }
  const ActiveSpan span = active_span(descriptor.active_bytes);
  if (!span.active) {
    return;
  }
  const std::size_t stripe_active_bytes = descriptor.active_bytes / 2U;
  const auto* source =
      input + descriptor.input_offset_bytes + span.stripe * stripe_active_bytes;
  auto* endpoint = span.stripe == 0U ? endpoint0 : endpoint1;
  auto* send = endpoint + descriptor.slot_storage_offset_bytes +
               span.stripe * layout.lane_stride;
  copy_active_bf16(source, send, span.begin, span.end);
}

__global__ void reduce_phase1_tile(
    std::uint8_t* endpoint0, std::uint8_t* endpoint1,
    const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout) {
  const TiledBulkDescriptor descriptor = descriptor_pointer[0];
  if (descriptor.active_bytes == 0U ||
      descriptor.active_bytes > kPayloadTileBytes ||
      descriptor.active_bytes % sizeof(__nv_bfloat162) != 0U) {
    return;
  }
  const ActiveSpan span = active_span(descriptor.active_bytes);
  if (!span.active) {
    return;
  }
  auto* source_endpoint = span.stripe == 0U ? endpoint0 : endpoint1;
  auto* destination_endpoint = span.stripe == 0U ? endpoint1 : endpoint0;
  const std::size_t lane = descriptor.slot_storage_offset_bytes +
                           span.stripe * layout.lane_stride;
  const auto* send = source_endpoint + lane;
  const auto* receive = send + layout.lane_receive_offset;
  auto* phase2_send = destination_endpoint + lane;
  reduce_active_bf16(send, receive, phase2_send, span.begin, span.end);
}

__global__ void reduce_phase2_tile(
    const std::uint8_t* endpoint0, const std::uint8_t* endpoint1,
    std::uint8_t* output, const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout) {
  const TiledBulkDescriptor descriptor = descriptor_pointer[0];
  if (descriptor.active_bytes == 0U ||
      descriptor.active_bytes > kPayloadTileBytes ||
      descriptor.active_bytes % sizeof(__nv_bfloat162) != 0U) {
    return;
  }
  const ActiveSpan span = active_span(descriptor.active_bytes);
  if (!span.active) {
    return;
  }
  const std::size_t stripe_active_bytes = descriptor.active_bytes / 2U;
  const auto* endpoint = span.stripe == 0U ? endpoint1 : endpoint0;
  const std::size_t lane = descriptor.slot_storage_offset_bytes +
                           span.stripe * layout.lane_stride;
  const auto* send = endpoint + lane;
  const auto* receive = send + layout.lane_receive_offset;
  auto* destination = output + descriptor.output_offset_bytes +
                      span.stripe * stripe_active_bytes;
  reduce_active_bf16(send, receive, destination, span.begin, span.end);
}

cudaError_t launch_stage_phase1_tile(
    const std::uint8_t* input, std::uint8_t* endpoint0,
    std::uint8_t* endpoint1,
    const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout,
    cudaStream_t stream) {
  if (input == nullptr || endpoint0 == nullptr || endpoint1 == nullptr ||
      descriptor_pointer == nullptr) {
    return cudaErrorInvalidValue;
  }
  stage_phase1_tile<<<kWorkerCtasPerTile, kWorkerThreads, 0, stream>>>(
      input, endpoint0, endpoint1, descriptor_pointer, layout);
  return cudaGetLastError();
}

cudaError_t launch_reduce_phase1_tile(
    std::uint8_t* endpoint0, std::uint8_t* endpoint1,
    const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout, cudaStream_t stream) {
  if (endpoint0 == nullptr || endpoint1 == nullptr ||
      descriptor_pointer == nullptr) {
    return cudaErrorInvalidValue;
  }
  reduce_phase1_tile<<<kWorkerCtasPerTile, kWorkerThreads, 0, stream>>>(
      endpoint0, endpoint1, descriptor_pointer, layout);
  return cudaGetLastError();
}

cudaError_t launch_reduce_phase2_tile(
    const std::uint8_t* endpoint0, const std::uint8_t* endpoint1,
    std::uint8_t* output,
    const TiledBulkDescriptor* descriptor_pointer,
    EndpointTileStorageLayout layout,
    cudaStream_t stream) {
  if (endpoint0 == nullptr || endpoint1 == nullptr || output == nullptr ||
      descriptor_pointer == nullptr) {
    return cudaErrorInvalidValue;
  }
  reduce_phase2_tile<<<kWorkerCtasPerTile, kWorkerThreads, 0, stream>>>(
      endpoint0, endpoint1, output, descriptor_pointer, layout);
  return cudaGetLastError();
}

}  // namespace spark_transport::tiled_prefill_research
