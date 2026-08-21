#include "spark_transport/topology.hpp"

#include <cassert>
#include <set>

int main() {
  const auto topology = spark_transport::Topology::four_spark_direct_cycle();
  assert((topology.ranks() == std::vector<std::uint32_t>{0, 1, 2, 3}));

  const std::set<std::pair<std::uint32_t, std::uint32_t>> expected{
      {0, 1}, {1, 0}, {2, 3}, {3, 2},
      {0, 2}, {2, 0}, {1, 3}, {3, 1},
  };
  for (const auto& [local, peer] : expected) {
    const auto& edge = topology.edge(local, peer);
    assert(edge.local_rank == local);
    assert(edge.peer_rank == peer);
  }

  assert(topology.edges_for(0).size() == 2);
  assert(topology.edges_for(1).size() == 2);
  assert(topology.edges_for(2).size() == 2);
  assert(topology.edges_for(3).size() == 2);
}
