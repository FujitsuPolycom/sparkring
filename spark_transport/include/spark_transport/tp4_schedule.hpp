#pragma once

#include <cstdint>

namespace spark_transport {

struct Tp4RoundPlan {
  std::uint32_t round{};
  std::uint32_t rank{};
  std::uint32_t peer_rank{};
  std::uint32_t device_index{};
  bool server{};
};

Tp4RoundPlan make_tp4_round_plan(std::uint32_t rank, std::uint32_t round);

}  // namespace spark_transport
