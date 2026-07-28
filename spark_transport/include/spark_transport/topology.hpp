#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spark_transport {

enum class TransportBackend {
  kVerbsRc,
  kUdp,
  kAfXdp,
};

struct EdgeConfig {
  std::uint32_t local_rank{};
  std::uint32_t peer_rank{};
  TransportBackend backend{TransportBackend::kVerbsRc};
  std::string device;
  std::uint8_t port{1};
  std::uint8_t gid_index{3};
  std::string peer_control_address;
  std::uint16_t control_port{};
};

class Topology {
 public:
  explicit Topology(std::vector<EdgeConfig> edges);

  std::vector<EdgeConfig> edges_for(std::uint32_t local_rank) const;
  const EdgeConfig& edge(std::uint32_t local_rank,
                         std::uint32_t peer_rank) const;
  std::vector<std::uint32_t> ranks() const;

  static Topology four_spark_tp2_pp2();

 private:
  std::vector<EdgeConfig> edges_;
};

}  // namespace spark_transport
