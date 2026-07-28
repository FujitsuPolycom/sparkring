#include "spark_transport/tp4_schedule.hpp"

#include <stdexcept>

namespace spark_transport {

Tp4RoundPlan make_tp4_round_plan(std::uint32_t rank, std::uint32_t round) {
  if (rank >= 4) {
    throw std::invalid_argument("TP4 rank must be in [0, 3]");
  }
  if (round >= 2) {
    throw std::invalid_argument("TP4 round must be zero or one");
  }

  const std::uint32_t mask = round == 0 ? 1U : 3U;
  const std::uint32_t peer = rank ^ mask;
  Tp4RoundPlan plan{};
  plan.round = round;
  plan.rank = rank;
  plan.peer_rank = peer;
  plan.device_index = round;
  plan.server = rank > peer;
  return plan;
}

}  // namespace spark_transport
