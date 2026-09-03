#include "tiled_bulk_kernels.cuh"
#include "tiled_correctness_kernels.cuh"
#include "tiled_correctness_oracle.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace research = spark_transport::tiled_prefill_research;

namespace {

constexpr std::size_t kRanks = 4U;
constexpr std::uint64_t kGuardBytes = 64U;
constexpr std::uint64_t kTileBytes = 512U * 1024U;
constexpr std::uint64_t kStripeCapacityBytes = kTileBytes / 2U;
constexpr std::uint64_t kReceiveOffsetBytes = kStripeCapacityBytes;
constexpr std::uint64_t kLaneStrideBytes =
    2U * kStripeCapacityBytes + 64U;
constexpr std::uint64_t kSlotStrideBytes = 2U * kLaneStrideBytes;
constexpr std::uint64_t kSlotsPerEdge = 8U;
constexpr std::uint64_t kEndpointBytes =
    kSlotsPerEdge * kSlotStrideBytes;

void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(result));
    std::exit(1);
  }
}

template <typename T>
T* allocate(std::size_t count) {
  T* pointer{};
  check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)),
        "cudaMalloc");
  return pointer;
}

void free_pointer(void* pointer) {
  if (pointer != nullptr) {
    check(cudaFree(pointer), "cudaFree");
  }
}

void run_width(std::uint32_t query_rows) {
  const research::TiledPayloadGeometry geometry =
      research::oracle_payload_geometry(query_rows);
  const std::uint32_t tile_count = static_cast<std::uint32_t>(
      geometry.capacity_bytes / kTileBytes);
  const std::uint64_t guarded_bytes =
      geometry.capacity_bytes + 2U * kGuardBytes;
  const research::EndpointTileStorageLayout layout{
      kReceiveOffsetBytes, kLaneStrideBytes};

  std::array<std::uint8_t*, kRanks> guarded_inputs{};
  std::array<std::uint8_t*, kRanks> guarded_outputs{};
  std::array<std::uint8_t*, kRanks> endpoint0{};
  std::array<std::uint8_t*, kRanks> endpoint1{};
  auto* descriptor = allocate<research::TiledBulkDescriptor>(1U);
  auto* receipt = allocate<research::TiledCorrectnessReceipt>(1U);
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  for (std::size_t rank = 0; rank < kRanks; ++rank) {
    guarded_inputs[rank] = allocate<std::uint8_t>(guarded_bytes);
    guarded_outputs[rank] = allocate<std::uint8_t>(guarded_bytes);
    endpoint0[rank] = allocate<std::uint8_t>(kEndpointBytes);
    endpoint1[rank] = allocate<std::uint8_t>(kEndpointBytes);
    check(cudaMemsetAsync(endpoint0[rank], 0xCD, kEndpointBytes, stream),
          "cudaMemsetAsync endpoint0");
    check(cudaMemsetAsync(endpoint1[rank], 0xCD, kEndpointBytes, stream),
          "cudaMemsetAsync endpoint1");
    check(research::launch_fill_correctness_sentinels(
              guarded_inputs[rank], guarded_outputs[rank], kGuardBytes,
              geometry.capacity_bytes, stream),
          "launch_fill_correctness_sentinels");
  }
  check(research::launch_initialize_correctness_receipt(
            receipt, query_rows, tile_count, geometry.active_bytes,
            spark_transport::kTp4TiledBf16ElementsPerQueryRow, stream),
        "launch_initialize_correctness_receipt");

  for (std::uint32_t ordinal = 0; ordinal < tile_count; ++ordinal) {
    const std::uint64_t payload_offset = ordinal * kTileBytes;
    const auto active_bytes = static_cast<std::uint32_t>(std::min(
        kTileBytes, geometry.active_bytes - payload_offset));
    const std::uint64_t slot_offset =
        (ordinal % kSlotsPerEdge) * kSlotStrideBytes;
    const research::TiledBulkDescriptor host_descriptor{
        payload_offset, payload_offset, slot_offset, active_bytes,
        ordinal / kSlotsPerEdge + 1U};
    check(cudaMemcpyAsync(descriptor, &host_descriptor,
                          sizeof(host_descriptor), cudaMemcpyHostToDevice,
                          stream),
          "cudaMemcpyAsync descriptor");

    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
      auto* input = guarded_inputs[rank] + kGuardBytes;
      check(research::launch_fill_correctness_input(
                input, descriptor, rank, stream),
            "launch_fill_correctness_input");
      check(research::launch_stage_phase1_tile(
                input, endpoint0[rank], endpoint1[rank], descriptor, layout,
                stream),
            "launch_stage_phase1_tile");
    }

    const std::uint64_t stripe_bytes = active_bytes / 2U;
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
      const std::uint32_t xor1 = rank ^ 1U;
      const std::uint32_t xor3 = rank ^ 3U;
      check(cudaMemcpyAsync(
                endpoint0[rank] + slot_offset + kReceiveOffsetBytes,
                endpoint0[xor1] + slot_offset, stripe_bytes,
                cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync phase1 lower");
      check(cudaMemcpyAsync(
                endpoint1[rank] + slot_offset + kLaneStrideBytes +
                    kReceiveOffsetBytes,
                endpoint1[xor3] + slot_offset + kLaneStrideBytes,
                stripe_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync phase1 upper");
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
      check(research::launch_reduce_phase1_tile(
                endpoint0[rank], endpoint1[rank], descriptor, layout, stream),
            "launch_reduce_phase1_tile");
    }

    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
      const std::uint32_t xor1 = rank ^ 1U;
      const std::uint32_t xor3 = rank ^ 3U;
      check(cudaMemcpyAsync(
                endpoint1[rank] + slot_offset + kReceiveOffsetBytes,
                endpoint1[xor3] + slot_offset, stripe_bytes,
                cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync phase2 lower");
      check(cudaMemcpyAsync(
                endpoint0[rank] + slot_offset + kLaneStrideBytes +
                    kReceiveOffsetBytes,
                endpoint0[xor1] + slot_offset + kLaneStrideBytes,
                stripe_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync phase2 upper");
    }
    for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
      auto* output = guarded_outputs[rank] + kGuardBytes;
      check(research::launch_reduce_phase2_tile(
                endpoint0[rank], endpoint1[rank], output, descriptor, layout,
                stream),
            "launch_reduce_phase2_tile");
      check(research::launch_validate_correctness(
                output, descriptor, receipt, stream),
            "launch_validate_correctness");
    }
  }

  for (std::uint32_t rank = 0; rank < kRanks; ++rank) {
    check(research::launch_validate_correctness_sentinels(
              guarded_inputs[rank], guarded_outputs[rank], kGuardBytes,
              geometry.active_bytes, geometry.capacity_bytes, receipt,
              stream),
          "launch_validate_correctness_sentinels");
  }

  research::TiledCorrectnessReceipt host_receipt{};
  check(cudaMemcpyAsync(&host_receipt, receipt, sizeof(host_receipt),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync receipt");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  const bool correct =
      host_receipt.mismatched_active_elements == 0U &&
      host_receipt.input_guard_corruptions == 0U &&
      host_receipt.output_guard_corruptions == 0U &&
      host_receipt.inactive_input_sentinel_corruptions == 0U &&
      host_receipt.inactive_output_sentinel_corruptions == 0U;
  if (!correct) {
    std::fprintf(
        stderr,
        "Q%u mismatch=%llu input_guard=%llu output_guard=%llu "
        "inactive_input=%llu inactive_output=%llu\n",
        query_rows,
        static_cast<unsigned long long>(
            host_receipt.mismatched_active_elements),
        static_cast<unsigned long long>(host_receipt.input_guard_corruptions),
        static_cast<unsigned long long>(host_receipt.output_guard_corruptions),
        static_cast<unsigned long long>(
            host_receipt.inactive_input_sentinel_corruptions),
        static_cast<unsigned long long>(
            host_receipt.inactive_output_sentinel_corruptions));
    std::exit(1);
  }
  std::printf(
      "TP4_TILED_CUDA_SMOKE {\"query_rows\":%u,\"tiles\":%u,"
      "\"active_bytes\":%llu,\"mismatches\":0,\"sentinel_corruptions\":0}\n",
      query_rows, tile_count,
      static_cast<unsigned long long>(geometry.active_bytes));

  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  free_pointer(receipt);
  free_pointer(descriptor);
  for (std::size_t rank = 0; rank < kRanks; ++rank) {
    free_pointer(endpoint1[rank]);
    free_pointer(endpoint0[rank]);
    free_pointer(guarded_outputs[rank]);
    free_pointer(guarded_inputs[rank]);
  }
}

}  // namespace

int main() {
  run_width(40U);
  run_width(512U);
  run_width(513U);
  run_width(1025U);
  run_width(4096U);
  return 0;
}
