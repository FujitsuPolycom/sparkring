#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "spark_transport/tp4_allreduce_protocol.hpp"
#include "spark_transport/tp4_dual_port_striped_allreduce.hpp"
#include "spark_transport/tp4_graph_command.hpp"
#include "spark_transport/tp4_graph_kernel_strategy.hpp"

namespace spark_transport {

struct Tp4AllreduceOptions {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9470};
  std::uint16_t control_port1{9471};
  std::size_t payload_bytes{12288};
  // Graph all-reduce row geometry. Legacy callers retain the GLM BF16
  // [Q, 6144] contract; versioned callers may select another BF16 width.
  std::uint32_t elements_per_row{kTp4GraphElementsPerRow};
  std::uint32_t bytes_per_row{kTp4GraphBytesPerRow};
  Tp4AllreduceProtocol protocol{Tp4AllreduceProtocol::kSerialAck};
  Tp4GraphKernelStrategy graph_kernel_strategy{
      Tp4GraphKernelStrategy::kFused};
  Tp4AllreduceSchedule schedule{Tp4AllreduceSchedule::kSequential};
  // When both are set, graph-only deployment pins the calling/submission
  // thread exclusively to graph_submit_cpu and pins the persistent verbs
  // progress thread exclusively to graph_progress_cpu. They must be distinct.
  std::optional<std::uint32_t> graph_submit_cpu;
  std::optional<std::uint32_t> graph_progress_cpu;
};

struct Tp4GraphReplayStatus {
  std::uint64_t captured_nodes{};
  std::uint64_t published_sequence{};
  std::uint64_t consumed_sequence{};
  std::uint64_t completed_sequence{};
  std::uint64_t overflow_sequence{};
  bool capture_configured{};
  bool polling_enabled{};
  bool host_native_atomics_supported{};
  bool submit_affinity_verified{};
  bool progress_affinity_verified{};
  bool two_slot_deferred_ack{};
  bool direct_doorbell{};
  int graph_submit_cpu{-1};
  int graph_progress_cpu{-1};
};

struct Tp4HealthStatus {
  bool healthy{};
  bool poisoned{};
  bool progress_thread_running{};
  bool stopping{};
  std::uint64_t submitted_sequence{};
  std::uint64_t completed_sequence{};
  std::uint64_t failing_sequence{};
  std::int32_t error_code{};
};

class Tp4AllreduceSession {
 public:
  Tp4AllreduceSession(const Tp4AllreduceSession&) = delete;
  Tp4AllreduceSession& operator=(const Tp4AllreduceSession&) = delete;

  explicit Tp4AllreduceSession(Tp4AllreduceOptions options);
  ~Tp4AllreduceSession();

  // Enqueues a stream-ordered reduction and returns before it completes.
  // Later work on the same stream observes the completed output. One session
  // accepts submissions from exactly one stable CUDA stream. Destruction
  // drains submitted collectives and synchronizes that stream. Submission is
  // bounded by SPARK_TP4_MAX_INFLIGHT (default 64) to prevent the caller from
  // saturating CUDA's launch queue.
  void all_reduce(const void* input, void* output, void* cuda_stream);

  // Adds a Q1 BF16 [1, elements_per_row] all-reduce to capture.
  // The captured graph is tied to these stable input/output addresses and
  // must be replayed on the same stream. This session must outlive every
  // graph replay. Replay sequences use either the mapped command ring or the
  // opt-in device-ticket/direct-doorbell lane. This may be called repeatedly
  // on the same active capture stream to define the many Q1 nodes in one or
  // more graphs, but always before any eager submission or graph replay.
  void capture_q1_all_reduce(const void* input, void* output,
                             void* cuda_stream);

  // Adds one exact contiguous BF16 [q, elements_per_row] all-reduce to CUDA
  // stream capture. q must be in [1, 512] and the session capacity must be at
  // least q rows for the session's configured row geometry.
  // arena and can serve decode plus bounded-prefill graph nodes while
  // preserving one ordered sequence.
  void capture_all_reduce(const void* input, void* output, std::uint32_t q,
                          void* cuda_stream);

  Tp4GraphReplayStatus graph_replay_status() const noexcept;
  Tp4HealthStatus health_status();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport
