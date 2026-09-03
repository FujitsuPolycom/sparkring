#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace spark_transport {

struct Tp4BidirectionalPrefillOptions {
  std::uint32_t rank{};
  std::string peer0;
  std::string peer1;
  std::string device0;
  std::string device1;
  std::uint8_t gid0{};
  std::uint8_t gid1{};
  std::uint16_t control_port0{};
  std::uint16_t control_port1{};
  std::uint32_t rail_count{1};
  std::string secondary_peer0;
  std::string secondary_peer1;
  std::string secondary_device0;
  std::string secondary_device1;
  std::uint8_t secondary_gid0{};
  std::uint8_t secondary_gid1{};
  std::uint16_t secondary_control_port0{};
  std::uint16_t secondary_control_port1{};
  std::uint32_t query_rows{};
  std::uint32_t elements_per_row{4096};
  std::uint32_t timeout_seconds{120};
  std::int32_t fused_proxy_cpu{12};
};

class Tp4BidirectionalPrefillSession {
 public:
  Tp4BidirectionalPrefillSession(
      const Tp4BidirectionalPrefillSession&) = delete;
  Tp4BidirectionalPrefillSession& operator=(
      const Tp4BidirectionalPrefillSession&) = delete;

  explicit Tp4BidirectionalPrefillSession(
      Tp4BidirectionalPrefillOptions options);
  ~Tp4BidirectionalPrefillSession();

  // Executes one exact contiguous CUDA BF16 [query_rows, 4096] all-reduce.
  // The call is bounded and returns only after output and registered-storage
  // retirement. Any partial native failure poisons this session permanently;
  // callers must not fall back for that collective.
  void all_reduce(const void* input, void* output, void* cuda_stream);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport
