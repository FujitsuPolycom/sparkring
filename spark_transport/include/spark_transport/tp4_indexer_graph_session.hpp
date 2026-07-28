#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace spark_transport {

struct Tp4IndexerGraphOptions {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9462};
  std::uint16_t control_port1{9463};
  std::optional<std::uint32_t> graph_submit_cpu;
  std::optional<std::uint32_t> graph_progress_cpu;
};

struct Tp4IndexerGraphReplayStatus {
  std::uint64_t captured_nodes{};
  std::uint64_t captured_q_mask{};
  std::uint64_t published_sequence{};
  std::uint64_t consumed_sequence{};
  std::uint64_t completed_sequence{};
  std::uint64_t overflow_sequence{};
  bool capture_configured{};
  bool polling_enabled{};
  bool host_native_atomics_supported{};
  bool submit_affinity_verified{};
  bool progress_affinity_verified{};
  int graph_submit_cpu{-1};
  int graph_progress_cpu{-1};
};

// Graph-only dynamic-Q indexer all-gather. One fixed Q40 arena and one
// command-ring sequence domain serve captured INT32 [Q,2,2048] nodes for
// every Q1..Q40. This class intentionally has no eager submission method.
class Tp4IndexerGraphSession {
 public:
  Tp4IndexerGraphSession(const Tp4IndexerGraphSession&) = delete;
  Tp4IndexerGraphSession& operator=(
      const Tp4IndexerGraphSession&) = delete;

  explicit Tp4IndexerGraphSession(Tp4IndexerGraphOptions options);
  ~Tp4IndexerGraphSession();

  void capture_all_gather(
      const void* input, void* output, std::uint32_t q,
      void* cuda_stream);

  Tp4IndexerGraphReplayStatus graph_replay_status() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport
