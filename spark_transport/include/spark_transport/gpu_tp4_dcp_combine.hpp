#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_tp4_dcp_query.hpp"

namespace spark_transport {

inline constexpr std::uint32_t kTp4DcpCombineWorldSize = 4;
inline constexpr std::uint32_t kTp4DcpCombineMaxQ =
    kTp4DcpQueryMaxQ;
inline constexpr std::size_t kTp4DcpCombineGlobalHeads = 64;
inline constexpr std::size_t kTp4DcpCombineHeadsPerRank = 16;
inline constexpr std::size_t kTp4DcpCombineProjectedHeadDimension = 256;
inline constexpr std::size_t kTp4DcpCombineLatentHeadDimension = 512;
inline constexpr std::size_t kTp4DcpCombineMaxHeadDimension =
    kTp4DcpCombineLatentHeadDimension;
inline constexpr std::size_t kTp4DcpCombineOutputElementBytes = 2;
inline constexpr std::size_t kTp4DcpCombineLseElementBytes = 4;

static_assert(kTp4DcpCombineWorldSize == kTp4DcpQueryWorldSize);
static_assert(kTp4DcpCombineMaxQ == kTp4DcpQueryMaxQ);

#if defined(__CUDACC__)
#define SPARK_TRANSPORT_DCP_HOST_DEVICE __host__ __device__
#else
#define SPARK_TRANSPORT_DCP_HOST_DEVICE
#endif

// The live vLLM tensor has logical shape [Q,64,D]. The exact B12X paths use
// either token-major strides (64*D,D,1) or head-major strides (D,Q*D,1).
// Pass the two observed leading strides to the worker; the last stride is
// required to be one.
SPARK_TRANSPORT_DCP_HOST_DEVICE inline constexpr std::size_t
tp4_dcp_combine_strided_output_index(
    std::size_t query_index, std::size_t global_head,
    std::size_t dimension, std::size_t query_stride,
    std::size_t head_stride) {
  return query_index * query_stride + global_head * head_stride +
         dimension;
}

SPARK_TRANSPORT_DCP_HOST_DEVICE inline constexpr std::size_t
tp4_dcp_combine_token_major_reduced_output_index(
    std::size_t query_index, std::size_t local_head,
    std::size_t dimension, std::size_t head_dimension) {
  return (query_index * kTp4DcpCombineHeadsPerRank + local_head) *
             head_dimension +
         dimension;
}

#undef SPARK_TRANSPORT_DCP_HOST_DEVICE

struct Tp4DcpCombineFrameLayout {
  std::size_t heads{};
  std::size_t output_bytes{};
  std::size_t lse_offset{};
  std::size_t total_bytes{};
};

struct Tp4DcpSoftmaxScalarState {
  float output{};
  float lse{};
};

bool tp4_dcp_combine_head_dimension_supported(std::uint32_t head_dimension);
std::size_t tp4_dcp_combine_input_output_bytes(
    std::uint32_t q, std::uint32_t head_dimension);
std::size_t tp4_dcp_combine_input_lse_bytes(std::uint32_t q);
std::size_t tp4_dcp_combine_reduced_output_bytes(
    std::uint32_t q, std::uint32_t head_dimension);
std::size_t tp4_dcp_combine_reduced_lse_bytes(std::uint32_t q);
Tp4DcpCombineFrameLayout tp4_dcp_combine_round0_frame(
    std::uint32_t q, std::uint32_t head_dimension);
Tp4DcpCombineFrameLayout tp4_dcp_combine_round1_frame(
    std::uint32_t q, std::uint32_t head_dimension);

std::array<std::uint32_t, 2> tp4_dcp_combine_round0_keep_chunks(
    std::uint32_t rank);
std::array<std::uint32_t, 2> tp4_dcp_combine_round0_send_chunks(
    std::uint32_t rank);
std::uint32_t tp4_dcp_combine_round1_send_chunk(std::uint32_t rank);

float tp4_dcp_sanitize_lse(float value);
Tp4DcpSoftmaxScalarState tp4_dcp_merge_scalar_state(
    Tp4DcpSoftmaxScalarState a, Tp4DcpSoftmaxScalarState b);

class GpuTp4DcpCombineWorker {
 public:
  GpuTp4DcpCombineWorker(const GpuTp4DcpCombineWorker&) = delete;
  GpuTp4DcpCombineWorker& operator=(const GpuTp4DcpCombineWorker&) = delete;

  GpuTp4DcpCombineWorker(
      std::uint32_t rank, const Tp4DcpQueryBufferLayout& shared_layout,
      void* round0_mapped_device_buffer,
      void* round1_mapped_device_buffer);

  // Output input has logical BF16 shape [Q,64,D], D in {256,512}, and one of
  // the two exact live layouts described above. LSE input is contiguous FP32
  // [Q,64]. Results are contiguous token-major BF16 [Q,16,D] output and FP32
  // [Q,16] LSE for the 16-head chunk owned by this DCP rank.
  void enqueue(const void* output_bf16, const void* lse_fp32,
               void* reduced_output_bf16, void* reduced_lse_fp32,
               std::uint32_t q, std::uint32_t head_dimension,
               std::uint32_t query_stride, std::uint32_t head_stride,
               void* cuda_stream,
               std::uint64_t sequence);

  // Adds the same operation to an active CUDA stream capture. Replay claims
  // its monotonic sequence on device and publishes a kDcpCombine command to
  // the shared DCP progress ring. head_dimension is carried as the strict
  // family parameter so D256 and D512 cannot be confused by the consumer.
  void enqueue_graph(const void* output_bf16, const void* lse_fp32,
                     void* reduced_output_bf16,
                     void* reduced_lse_fp32, std::uint32_t q,
                     std::uint32_t head_dimension,
                     std::uint32_t query_stride,
                     std::uint32_t head_stride, void* cuda_stream,
                     Tp4GraphCommandRing* command_ring, bool trace);

 private:
  std::uint32_t rank_{};
  Tp4DcpQueryBufferLayout layout_{};
  void* round0_buffer_{};
  void* round1_buffer_{};
};

}  // namespace spark_transport
