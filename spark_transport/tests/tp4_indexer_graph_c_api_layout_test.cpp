#include "spark_transport/tp4_indexer_graph_c_api.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

int main() {
  static_assert(sizeof(spark_tp4_indexer_graph_status) == 64);
  static_assert(
      offsetof(spark_tp4_indexer_graph_status, captured_nodes) == 8);
  static_assert(
      offsetof(spark_tp4_indexer_graph_status, captured_q_mask) == 16);
  static_assert(
      offsetof(
          spark_tp4_indexer_graph_status, published_sequence) == 24);
  static_assert(
      offsetof(
          spark_tp4_indexer_graph_status,
          graph_submit_cpu_plus_one) == 56);

  spark_tp4_indexer_graph_status status{};
  status.struct_size = sizeof(status);
  status.flags =
      SPARK_TP4_INDEXER_GRAPH_CAPTURE_CONFIGURED |
      SPARK_TP4_INDEXER_GRAPH_POLLING_ENABLED |
      SPARK_TP4_INDEXER_GRAPH_HOST_NATIVE_ATOMICS |
      SPARK_TP4_INDEXER_GRAPH_SUBMIT_AFFINITY_VERIFIED |
      SPARK_TP4_INDEXER_GRAPH_PROGRESS_AFFINITY_VERIFIED;
  status.captured_q_mask =
      (std::uint64_t{1} << 0) | (std::uint64_t{1} << 39);
  assert(status.struct_size == 64);
  assert((status.captured_q_mask & 1U) != 0);
  assert((status.captured_q_mask & (std::uint64_t{1} << 39)) != 0);
  assert(
      (status.flags & SPARK_TP4_INDEXER_GRAPH_OVERFLOW_FATAL) == 0);
  return 0;
}
