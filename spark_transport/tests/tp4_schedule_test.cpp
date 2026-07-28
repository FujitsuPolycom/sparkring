#include "spark_transport/tp4_schedule.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <set>
#include <stdexcept>
#include <utility>

int main() {
  std::set<std::pair<std::uint32_t, std::uint32_t>> round0_edges;
  std::set<std::pair<std::uint32_t, std::uint32_t>> round1_edges;

  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    const auto round0 = spark_transport::make_tp4_round_plan(rank, 0);
    const auto round1 = spark_transport::make_tp4_round_plan(rank, 1);

    assert(round0.peer_rank == (rank ^ 1U));
    assert(round1.peer_rank == (rank ^ 3U));
    assert(round0.device_index == 0);
    assert(round1.device_index == 1);
    assert(round0.server == (rank > round0.peer_rank));
    assert(round1.server == (rank > round1.peer_rank));

    round0_edges.emplace(
        std::min(rank, round0.peer_rank),
        std::max(rank, round0.peer_rank));
    round1_edges.emplace(
        std::min(rank, round1.peer_rank),
        std::max(rank, round1.peer_rank));
  }

  assert((round0_edges ==
          std::set<std::pair<std::uint32_t, std::uint32_t>>{
              {0, 1}, {2, 3}}));
  assert((round1_edges ==
          std::set<std::pair<std::uint32_t, std::uint32_t>>{
              {0, 3}, {1, 2}}));

  bool rejected_rank = false;
  try {
    static_cast<void>(spark_transport::make_tp4_round_plan(4, 0));
  } catch (const std::invalid_argument&) {
    rejected_rank = true;
  }
  assert(rejected_rank);

  bool rejected_round = false;
  try {
    static_cast<void>(spark_transport::make_tp4_round_plan(0, 2));
  } catch (const std::invalid_argument&) {
    rejected_round = true;
  }
  assert(rejected_round);
}
