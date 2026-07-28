#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace spark_transport {

struct Tp4AllgatherOptions {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9490};
  std::uint16_t control_port1{9491};
  std::size_t input_bytes{};
};

class Tp4AllgatherSession {
 public:
  Tp4AllgatherSession(const Tp4AllgatherSession&) = delete;
  Tp4AllgatherSession& operator=(const Tp4AllgatherSession&) = delete;

  explicit Tp4AllgatherSession(Tp4AllgatherOptions options);
  ~Tp4AllgatherSession();

  // Output contains four input-sized rank segments in rank order. The call
  // enqueues stream-ordered work and returns before the collective completes.
  // Submission is bounded by SPARK_TP4_MAX_INFLIGHT (default 64) to prevent
  // the caller from saturating CUDA's launch queue.
  void all_gather(const void* input, void* output, void* cuda_stream);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport
