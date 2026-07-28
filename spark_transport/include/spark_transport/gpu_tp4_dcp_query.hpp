#pragma once

#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/tp4_graph_command.hpp"

namespace spark_transport {

inline constexpr std::uint32_t kTp4DcpQueryWorldSize = 4;
inline constexpr std::uint32_t kTp4DcpQueryMaxQ =
    kTp4GraphMaximumQ;
inline constexpr std::size_t kTp4DcpQueryHeadsPerRank = 16;
inline constexpr std::size_t kTp4DcpQueryHeadDimension = 576;
inline constexpr std::size_t kTp4DcpQueryElementBytes = 2;
inline constexpr std::size_t kTp4DcpQueryBytesPerQ =
    kTp4DcpQueryHeadsPerRank * kTp4DcpQueryHeadDimension *
    kTp4DcpQueryElementBytes;
// The shared DCP session also carries the widest live softmax-state combine:
// Q=max, 32 transported heads, latent D=512 BF16, plus FP32 LSE.
inline constexpr std::size_t kTp4DcpRound0PayloadCapacity =
    static_cast<std::size_t>(kTp4DcpQueryMaxQ) * 32 * 512 * 2 +
    static_cast<std::size_t>(kTp4DcpQueryMaxQ) * 32 * 4;
// Query round 1 transports two Qx16x576 rank segments and remains wider than
// the corresponding Qx16x512 combine frame.
inline constexpr std::size_t kTp4DcpRound1PayloadCapacity =
    static_cast<std::size_t>(kTp4DcpQueryMaxQ) *
    kTp4DcpQueryHeadsPerRank * kTp4DcpQueryHeadDimension *
    kTp4DcpQueryElementBytes * 2;

struct Tp4DcpQueryBufferLayout {
  Tp2BufferLayout round0;
  Tp2BufferLayout round1;
  std::size_t max_input_bytes{};
  std::size_t max_output_bytes{};
};

std::size_t tp4_dcp_query_input_bytes(std::uint32_t q);
std::size_t tp4_dcp_query_output_bytes(std::uint32_t q);
std::size_t tp4_dcp_query_output_offset(
    std::uint32_t query_index, std::uint32_t source_rank,
    std::size_t byte_in_rank_query);
Tp4DcpQueryBufferLayout make_tp4_dcp_query_buffer_layout();

class GpuTp4DcpQueryWorker {
 public:
  GpuTp4DcpQueryWorker(const GpuTp4DcpQueryWorker&) = delete;
  GpuTp4DcpQueryWorker& operator=(const GpuTp4DcpQueryWorker&) = delete;

  GpuTp4DcpQueryWorker(
      std::uint32_t rank, const Tp4DcpQueryBufferLayout& layout,
      void* round0_mapped_device_buffer,
      void* round1_mapped_device_buffer);

  // Input is contiguous BF16 [Q,16,576]. Output is contiguous BF16
  // [Q,64,576], with the four rank segments concatenated on the head axis.
  void enqueue(const void* input, void* output, std::uint32_t q,
               void* cuda_stream, std::uint64_t sequence);

  // Adds the same operation to an active CUDA stream capture. Replay claims
  // its monotonic sequence on device and publishes a kDcpQuery command to the
  // shared DCP progress ring.
  void enqueue_graph(const void* input, void* output, std::uint32_t q,
                     void* cuda_stream,
                     Tp4GraphCommandRing* command_ring, bool trace);

 private:
  std::uint32_t rank_{};
  Tp4DcpQueryBufferLayout layout_{};
  void* round0_buffer_{};
  void* round1_buffer_{};
};

}  // namespace spark_transport
