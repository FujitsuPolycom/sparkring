#include "spark_transport/topology.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace spark_transport {

Topology::Topology(std::vector<EdgeConfig> edges) : edges_(std::move(edges)) {
  std::set<std::pair<std::uint32_t, std::uint32_t>> unique_edges;
  for (const auto& edge : edges_) {
    if (edge.local_rank == edge.peer_rank) {
      throw std::invalid_argument("a transport edge cannot target its own rank");
    }
    if (!unique_edges.emplace(edge.local_rank, edge.peer_rank).second) {
      throw std::invalid_argument("duplicate directed transport edge");
    }
    if (edge.backend == TransportBackend::kVerbsRc && edge.device.empty()) {
      throw std::invalid_argument("an RC edge requires an explicit device");
    }
  }
}

std::vector<EdgeConfig> Topology::edges_for(std::uint32_t local_rank) const {
  std::vector<EdgeConfig> selected;
  std::copy_if(edges_.begin(), edges_.end(), std::back_inserter(selected),
               [local_rank](const EdgeConfig& edge) {
                 return edge.local_rank == local_rank;
               });
  if (selected.empty()) {
    throw std::out_of_range("rank has no configured transport edges");
  }
  return selected;
}

const EdgeConfig& Topology::edge(std::uint32_t local_rank,
                                 std::uint32_t peer_rank) const {
  const auto found =
      std::find_if(edges_.begin(), edges_.end(),
                   [local_rank, peer_rank](const EdgeConfig& candidate) {
                     return candidate.local_rank == local_rank &&
                            candidate.peer_rank == peer_rank;
                   });
  if (found == edges_.end()) {
    throw std::out_of_range("transport edge is not configured");
  }
  return *found;
}

std::vector<std::uint32_t> Topology::ranks() const {
  std::set<std::uint32_t> unique;
  for (const auto& edge : edges_) {
    unique.insert(edge.local_rank);
    unique.insert(edge.peer_rank);
  }
  return {unique.begin(), unique.end()};
}

Topology Topology::four_spark_tp2_pp2() {
  // vLLM global ranks are deliberately mapped to logical nodes:
  //   0=spark0, 1=spark1, 2=spark3, 3=spark2.
  //
  // TP edges: 0<->1 and 2<->3.
  // PP edges: 0<->2 and 1<->3.
  //
  // Each pair is a directly attached 200G edge. Device selection is local:
  // the spark0/spark1 TP pair uses cage 0; the spark3/spark2 TP pair and the
  // two PP lanes use the corresponding direct cage on each physical node.
  // Exact device assignment is injected at deployment; these logical edges
  // intentionally do not embed host-specific interface names.
  std::vector<EdgeConfig> edges;
  const auto add_pair = [&edges](std::uint32_t left, std::uint32_t right,
                                 std::uint16_t port_base) {
    edges.push_back(
        {left, right, TransportBackend::kVerbsRc, "DEPLOY", 1, 3, "",
         port_base});
    edges.push_back(
        {right, left, TransportBackend::kVerbsRc, "DEPLOY", 1, 3, "",
         port_base});
  };
  add_pair(0, 1, 9410);
  add_pair(2, 3, 9420);
  add_pair(0, 2, 9430);
  add_pair(1, 3, 9440);
  return Topology(std::move(edges));
}

}  // namespace spark_transport
