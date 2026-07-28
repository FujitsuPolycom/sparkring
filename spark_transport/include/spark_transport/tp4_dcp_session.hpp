#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace spark_transport {

struct Tp4DcpOptions {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9890};
  std::uint16_t control_port1{9891};
  std::optional<std::uint32_t> graph_submit_cpu;
  std::optional<std::uint32_t> graph_progress_cpu;
};

struct Tp4DcpGraphReplayStatus {
  std::uint64_t captured_nodes{};
  std::uint64_t captured_query_nodes{};
  std::uint64_t captured_combine_nodes{};
  std::uint64_t published_sequence{};
  std::uint64_t consumed_sequence{};
  std::uint64_t completed_sequence{};
  std::uint64_t overflow_sequence{};
  bool capture_configured{};
  bool polling_enabled{};
  bool dedicated_spin{};
  bool host_native_atomics_supported{};
  bool submit_affinity_verified{};
  bool progress_affinity_verified{};
  int graph_submit_cpu{-1};
  int graph_progress_cpu{-1};
};

class Tp4DcpSession {
 public:
  Tp4DcpSession(const Tp4DcpSession&) = delete;
  Tp4DcpSession& operator=(const Tp4DcpSession&) = delete;

  explicit Tp4DcpSession(Tp4DcpOptions options);
  ~Tp4DcpSession();

  // Gathers contiguous BF16 [Q,16,576] inputs into contiguous BF16
  // [Q,64,576], with rank order on the head axis. Q must be in [1,40].
  // Work is ordered across caller-stream changes and returns asynchronously.
  // SPARK_TP4_MAX_INFLIGHT (default 64) bounds queued collectives.
  void query_all_gather(const void* input, void* output, std::uint32_t q,
                        void* cuda_stream);

  // Combines normalized local BF16 output with logical shape [Q,64,D],
  // D in {256,512}, and exact supplied leading element strides, plus
  // contiguous FP32 [Q,64] local LSE states. Writes this rank's contiguous
  // token-major BF16 [Q,16,D] output and FP32 [Q,16] global LSE.
  void combine(const void* output_bf16, const void* lse_fp32,
               void* reduced_output_bf16, void* reduced_lse_fp32,
               std::uint32_t q, std::uint32_t head_dimension,
               std::uint32_t query_stride, std::uint32_t head_stride,
               void* cuda_stream);

  // Graph-configured sessions reject eager submission. These methods add
  // family-tagged nodes from active captures to one shared ordered DCP
  // command ring. A device-claimed sequence totally orders replays even when
  // vLLM captures target and proposer graphs on different CUDA streams.
  void capture_query_all_gather(
      const void* input, void* output, std::uint32_t q,
      void* cuda_stream);
  void capture_combine(
      const void* output_bf16, const void* lse_fp32,
      void* reduced_output_bf16, void* reduced_lse_fp32,
      std::uint32_t q, std::uint32_t head_dimension,
      std::uint32_t query_stride, std::uint32_t head_stride,
      void* cuda_stream);

  Tp4DcpGraphReplayStatus graph_replay_status() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport
