#include "bidirectional_bulk_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;

namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

__global__ void fill_rank_input(__nv_bfloat16* input, std::size_t elements,
                                float value) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t element = index; element < elements; element += stride) {
    input[element] = __float2bfloat16(value);
  }
}

struct MappedEndpoint {
  std::uint8_t* host{};
  std::uint8_t* device{};

  MappedEndpoint() {
    constexpr auto maximum_tile =
        research::bidirectional_bulk_tile_bytes(8192);
    check_cuda(cudaHostAlloc(&host,
                             2U * maximum_tile +
                                 2U * sizeof(std::uint64_t),
                             cudaHostAllocMapped),
               "cudaHostAlloc mapped endpoint");
    check_cuda(cudaHostGetDevicePointer(&device, host, 0),
               "cudaHostGetDevicePointer mapped endpoint");
    std::memset(host, 0,
                2U * maximum_tile +
                    2U * sizeof(std::uint64_t));
  }

  ~MappedEndpoint() {
    if (host != nullptr) cudaFreeHost(host);
  }
};

using RankPointers =
    std::array<std::uint8_t*, spark_transport::kTp4PrefillRankCount>;

constexpr std::uint64_t doorbell_token(std::uint32_t stage,
                                       std::uint32_t tile) {
  return static_cast<std::uint64_t>(stage) *
             research::kBidirectionalBulkTilesPerShard +
         tile + 1U;
}

void transfer_stage(
    const std::array<MappedEndpoint, spark_transport::kTp4PrefillRankCount>&
        endpoints,
    spark_transport::Tp4PrefillDirection direction,
    std::uint64_t doorbell_token, std::uint32_t tile_bytes) {
  for (std::uint32_t rank = 0;
       rank < spark_transport::kTp4PrefillRankCount; ++rank) {
    const std::uint32_t receiver =
        spark_transport::tp4_prefill_successor(rank, direction);
    std::memcpy(endpoints[receiver].host +
                    tile_bytes,
                endpoints[rank].host,
                tile_bytes);
    __atomic_store_n(
        reinterpret_cast<std::uint64_t*>(
            endpoints[receiver].host +
            2U * tile_bytes),
        doorbell_token, __ATOMIC_RELEASE);
    __atomic_store_n(
        reinterpret_cast<std::uint64_t*>(
            endpoints[receiver].host + 2U * tile_bytes +
            sizeof(std::uint64_t)),
        doorbell_token, __ATOMIC_RELEASE);
  }
}

void run_direction(
    spark_transport::Tp4PrefillDirection direction,
    std::uint32_t query_rows,
    const RankPointers& inputs, const RankPointers& outputs,
    std::array<MappedEndpoint, spark_transport::kTp4PrefillRankCount>&
        endpoints,
    cudaStream_t stream) {
  constexpr std::uint64_t send_offset = 0;
  const std::uint32_t tile_bytes =
      research::bidirectional_bulk_tile_bytes(query_rows);
  const std::uint64_t receive_offset = tile_bytes;
  const std::uint64_t inbound_doorbell_offset = 2U * tile_bytes;
  for (std::uint32_t tile = 0;
       tile < research::kBidirectionalBulkTilesPerShard; ++tile) {
    for (std::uint32_t rank = 0;
         rank < spark_transport::kTp4PrefillRankCount; ++rank) {
      const auto descriptor =
          research::make_bidirectional_stage_initial_descriptor(
              rank, direction, tile, send_offset, query_rows);
      check_cuda(research::launch_bidirectional_stage_initial(
                     inputs[rank], endpoints[rank].device, descriptor,
                     stream, query_rows),
                 "launch stage initial");
    }
    check_cuda(cudaStreamSynchronize(stream), "complete initial stage");
    transfer_stage(endpoints, direction, doorbell_token(0, tile), tile_bytes);

    for (std::uint32_t hop = 0; hop < 2; ++hop) {
      for (std::uint32_t rank = 0;
           rank < spark_transport::kTp4PrefillRankCount; ++rank) {
        const auto descriptor =
            research::make_bidirectional_post_exchange_descriptor(
                rank, direction, hop, tile, send_offset, receive_offset,
                inbound_doorbell_offset,
                doorbell_token(hop, tile), query_rows,
                inbound_doorbell_offset + sizeof(std::uint64_t),
                doorbell_token(hop, tile));
        check_cuda(research::launch_bidirectional_reduce_forward(
                       inputs[rank], endpoints[rank].device,
                       endpoints[rank].device, descriptor, stream,
                       query_rows),
                   "launch reduce forward");
      }
      check_cuda(cudaStreamSynchronize(stream), "complete reduce forward");
      transfer_stage(
          endpoints, direction,
          doorbell_token(hop + 1U, tile), tile_bytes);
    }

    for (std::uint32_t rank = 0;
         rank < spark_transport::kTp4PrefillRankCount; ++rank) {
      const auto descriptor =
          research::make_bidirectional_post_exchange_descriptor(
              rank, direction, 2, tile, send_offset, receive_offset,
              inbound_doorbell_offset,
              doorbell_token(2, tile), query_rows,
              inbound_doorbell_offset + sizeof(std::uint64_t),
              doorbell_token(2, tile));
      check_cuda(research::launch_bidirectional_reduce_finalize_seed_gather(
                     inputs[rank], endpoints[rank].device,
                     endpoints[rank].device, outputs[rank], descriptor,
                     stream, query_rows),
                 "launch reduce finalize");
    }
    check_cuda(cudaStreamSynchronize(stream), "complete reduce finalize");
    transfer_stage(endpoints, direction, doorbell_token(3, tile), tile_bytes);

    for (std::uint32_t hop = 0; hop < 2; ++hop) {
      const std::uint32_t stage =
          spark_transport::kTp4PrefillReduceScatterStages + hop;
      for (std::uint32_t rank = 0;
           rank < spark_transport::kTp4PrefillRankCount; ++rank) {
        const auto descriptor =
            research::make_bidirectional_post_exchange_descriptor(
                rank, direction, stage, tile, send_offset, receive_offset,
                inbound_doorbell_offset,
                doorbell_token(stage, tile), query_rows,
                inbound_doorbell_offset + sizeof(std::uint64_t),
                doorbell_token(stage, tile));
        check_cuda(research::launch_bidirectional_gather_forward(
                       endpoints[rank].device, endpoints[rank].device,
                       outputs[rank], descriptor, stream, query_rows),
                   "launch gather forward");
      }
      check_cuda(cudaStreamSynchronize(stream), "complete gather forward");
      transfer_stage(
          endpoints, direction,
          doorbell_token(stage + 1U, tile), tile_bytes);
    }

    for (std::uint32_t rank = 0;
         rank < spark_transport::kTp4PrefillRankCount; ++rank) {
      const auto descriptor =
          research::make_bidirectional_post_exchange_descriptor(
              rank, direction, 5, tile, send_offset, receive_offset,
              inbound_doorbell_offset,
              doorbell_token(5, tile), query_rows,
              inbound_doorbell_offset + sizeof(std::uint64_t),
              doorbell_token(5, tile));
      check_cuda(research::launch_bidirectional_gather_finish(
                     endpoints[rank].device, outputs[rank], descriptor,
                     stream, query_rows),
                 "launch gather finish");
    }
    check_cuda(cudaStreamSynchronize(stream), "complete gather finish");
  }
}

float deterministic_noninteger_input(std::uint32_t rank,
                                     std::size_t element) {
  const std::int32_t centered = static_cast<std::int32_t>(
      (element * 37U + static_cast<std::size_t>(rank) * 53U) % 257U) -
      128;
  const float scale = 0.0078125F * static_cast<float>(rank + 1U);
  const float offset =
      static_cast<float>((element + 3U * rank) % 11U) * 0.001953125F;
  return static_cast<float>(centered) * scale + offset;
}

struct OracleComparison {
  double max_abs{};
  double max_rel{};
  std::uint64_t mismatches{};
};

OracleComparison compare_to_rounded_fp32(
    const RankPointers& outputs,
    const std::vector<__nv_bfloat16>& expected,
    std::vector<__nv_bfloat16>& observed, double atol, double rtol) {
  OracleComparison comparison{};
  for (std::uint32_t rank = 0;
       rank < spark_transport::kTp4PrefillRankCount; ++rank) {
    check_cuda(cudaMemcpy(observed.data(), outputs[rank],
                          expected.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "copy noninteger all-reduce output");
    for (std::size_t element = 0; element < expected.size(); ++element) {
      const double actual = __bfloat162float(observed[element]);
      const double reference = __bfloat162float(expected[element]);
      const double absolute = std::abs(actual - reference);
      const double relative =
          absolute / std::max(std::abs(reference), 1.0e-12);
      comparison.max_abs = std::max(comparison.max_abs, absolute);
      comparison.max_rel = std::max(comparison.max_rel, relative);
      if (absolute > atol + rtol * std::abs(reference)) {
        ++comparison.mismatches;
      }
    }
  }
  return comparison;
}

}  // namespace

int main() {
  bool rejected_missing_gate{};
  try {
    (void)research::make_bidirectional_post_exchange_descriptor(
        0, spark_transport::Tp4PrefillDirection::kClockwise, 0, 0);
  } catch (const std::invalid_argument&) {
    rejected_missing_gate = true;
  }
  assert(rejected_missing_gate);
  bool rejected_misaligned_gate{};
  try {
    (void)research::make_bidirectional_post_exchange_descriptor(
        0, spark_transport::Tp4PrefillDirection::kClockwise, 0, 0, 0, 0,
        3, 1);
  } catch (const std::invalid_argument&) {
    rejected_misaligned_gate = true;
  }
  assert(rejected_misaligned_gate);

  check_cuda(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
  cudaStream_t stream{};
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreate");

  RankPointers inputs{};
  RankPointers outputs{};
  constexpr std::array<std::uint32_t, 4> query_shapes{1024, 2048, 4096,
                                                      8192};
  constexpr std::uint64_t maximum_payload =
      research::bidirectional_bulk_payload_bytes(8192);
  constexpr std::size_t maximum_elements =
      maximum_payload / sizeof(__nv_bfloat16);
  for (std::uint32_t rank = 0;
       rank < spark_transport::kTp4PrefillRankCount; ++rank) {
    check_cuda(cudaMalloc(&inputs[rank], maximum_payload),
               "cudaMalloc input");
    check_cuda(cudaMalloc(&outputs[rank], maximum_payload),
               "cudaMalloc output");
    fill_rank_input<<<128, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(inputs[rank]), maximum_elements,
        static_cast<float>(rank + 1));
    check_cuda(cudaMemsetAsync(outputs[rank], 0,
                               maximum_payload,
                               stream),
               "clear output");
  }
  check_cuda(cudaStreamSynchronize(stream), "initialize inputs");

  std::array<MappedEndpoint, spark_transport::kTp4PrefillRankCount>
      clockwise_endpoints;
  std::array<MappedEndpoint, spark_transport::kTp4PrefillRankCount>
      counter_clockwise_endpoints;
  std::array<std::uint64_t, query_shapes.size()> exact_mismatches{};
  std::vector<__nv_bfloat16> host(maximum_elements);
  for (std::size_t shape = 0; shape < query_shapes.size(); ++shape) {
    const std::uint32_t query_rows = query_shapes[shape];
    const std::uint64_t payload =
        research::bidirectional_bulk_payload_bytes(query_rows);
    const std::size_t elements = payload / sizeof(__nv_bfloat16);
    for (std::uint32_t rank = 0;
         rank < spark_transport::kTp4PrefillRankCount; ++rank) {
      check_cuda(cudaMemsetAsync(outputs[rank], 0, payload, stream),
                 "clear adaptive exact output");
    }
    run_direction(spark_transport::Tp4PrefillDirection::kClockwise,
                  query_rows, inputs, outputs, clockwise_endpoints,
                  stream);
    run_direction(spark_transport::Tp4PrefillDirection::kCounterClockwise,
                  query_rows, inputs, outputs, counter_clockwise_endpoints,
                  stream);
    for (std::uint32_t rank = 0;
         rank < spark_transport::kTp4PrefillRankCount; ++rank) {
      check_cuda(cudaMemcpy(host.data(), outputs[rank], payload,
                            cudaMemcpyDeviceToHost),
                 "copy adaptive exact output");
      for (std::size_t element = 0; element < elements; ++element) {
        if (__bfloat162float(host[element]) != 10.0F) {
          ++exact_mismatches[shape];
        }
      }
    }
    assert(exact_mismatches[shape] == 0);
  }

  constexpr std::uint32_t representative_query_rows = 2048;
  constexpr std::uint64_t representative_payload =
      research::bidirectional_bulk_payload_bytes(representative_query_rows);
  constexpr std::size_t elements =
      representative_payload / sizeof(__nv_bfloat16);
  std::vector<float> fp32_reference(elements, 0.0F);
  std::vector<__nv_bfloat16> input_host(elements);
  for (std::uint32_t rank = 0;
       rank < spark_transport::kTp4PrefillRankCount; ++rank) {
    for (std::size_t element = 0; element < elements; ++element) {
      const auto rounded = __float2bfloat16(
          deterministic_noninteger_input(rank, element));
      input_host[element] = rounded;
      fp32_reference[element] += __bfloat162float(rounded);
    }
    check_cuda(cudaMemcpy(inputs[rank], input_host.data(),
                          representative_payload,
                          cudaMemcpyHostToDevice),
               "upload noninteger BF16 input");
    check_cuda(cudaMemsetAsync(outputs[rank], 0,
                               representative_payload,
                               stream),
               "clear noninteger output");
  }
  std::vector<__nv_bfloat16> rounded_reference(elements);
  for (std::size_t element = 0; element < elements; ++element) {
    rounded_reference[element] = __float2bfloat16(fp32_reference[element]);
  }

  run_direction(spark_transport::Tp4PrefillDirection::kClockwise,
                representative_query_rows, inputs, outputs,
                clockwise_endpoints, stream);
  run_direction(spark_transport::Tp4PrefillDirection::kCounterClockwise,
                representative_query_rows, inputs, outputs,
                counter_clockwise_endpoints, stream);

  constexpr double kAbsoluteTolerance = 0.0625;
  constexpr double kRelativeTolerance = 0.02;
  const OracleComparison comparison = compare_to_rounded_fp32(
      outputs, rounded_reference, host, kAbsoluteTolerance,
      kRelativeTolerance);
  std::cout << std::setprecision(12)
            << "BIDIRECTIONAL_BF16_ORACLE exact_integer_mismatches="
            << (exact_mismatches[0] + exact_mismatches[1] +
                exact_mismatches[2] + exact_mismatches[3])
            << " exact_q1024_mismatches=" << exact_mismatches[0]
            << " exact_q2048_mismatches=" << exact_mismatches[1]
            << " exact_q4096_mismatches=" << exact_mismatches[2]
            << " exact_q8192_mismatches=" << exact_mismatches[3]
            << " noninteger_max_abs=" << comparison.max_abs
            << " noninteger_max_rel=" << comparison.max_rel
            << " noninteger_mismatches=" << comparison.mismatches
            << " atol=" << kAbsoluteTolerance
            << " rtol=" << kRelativeTolerance
            << " elements_per_rank=" << elements
            << " ranks=" << spark_transport::kTp4PrefillRankCount << '\n';
  assert(comparison.mismatches == 0);

  for (std::uint32_t rank = 0;
       rank < spark_transport::kTp4PrefillRankCount; ++rank) {
    cudaFree(outputs[rank]);
    cudaFree(inputs[rank]);
  }
  cudaStreamDestroy(stream);
}
