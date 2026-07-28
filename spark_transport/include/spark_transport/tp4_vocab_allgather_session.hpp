#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace spark_transport {

struct Tp4VocabAllgatherOptions {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9990};
  std::uint16_t control_port1{9991};
  std::optional<std::uint32_t> graph_submit_cpu;
  std::optional<std::uint32_t> graph_progress_cpu;
};

struct Tp4VocabGraphReplayStatus {
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
  int graph_submit_cpu{-1};
  int graph_progress_cpu{-1};
};

class Tp4VocabAllgatherSession {
 public:
  Tp4VocabAllgatherSession(
      const Tp4VocabAllgatherSession&) = delete;
  Tp4VocabAllgatherSession& operator=(
      const Tp4VocabAllgatherSession&) = delete;

  explicit Tp4VocabAllgatherSession(
      Tp4VocabAllgatherOptions options);
  ~Tp4VocabAllgatherSession();

  // Gathers BF16 [Q, 38720] into token-major BF16 [Q, 154880].
  // Q must be in [1, 40]. Work is enqueued on one stable caller stream.
  void all_gather(const void* input, void* output,
                  std::uint32_t query_rows, void* cuda_stream);

  // Adds BF16 [Q, 38720] -> [Q, 154880], Q in [1, 40], to an active
  // CUDA stream capture. The session, stable allocations, and stream must
  // outlive every replay.
  void capture_all_gather(
      const void* input, void* output, std::uint32_t query_rows,
      void* cuda_stream);

  Tp4VocabGraphReplayStatus graph_replay_status() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport
