#include "spark_transport/tp4_graph_kernel_strategy.hpp"
#include "spark_transport/tp4_session.hpp"

#include <cassert>
#include <cstdint>
#include <string_view>

int main() {
  using spark_transport::Tp4AllreduceOptions;
  using spark_transport::Tp4GraphKernelStrategy;
  using spark_transport::kTp4TieredSplitMinimumQ;
  using spark_transport::tp4_graph_kernel_strategy_is_graph_only;
  using spark_transport::tp4_graph_kernel_strategy_name;
  using spark_transport::tp4_graph_kernel_strategy_valid;
  using spark_transport::tp4_graph_kernel_uses_split;

  constexpr auto fused = Tp4GraphKernelStrategy::kFused;
  constexpr auto split = Tp4GraphKernelStrategy::kSplit64KiB;
  constexpr auto tiered = Tp4GraphKernelStrategy::kTiered64KiB;
  constexpr auto invalid =
      static_cast<Tp4GraphKernelStrategy>(UINT8_C(255));

  static_assert(kTp4TieredSplitMinimumQ == 7);
  static_assert(tp4_graph_kernel_strategy_valid(fused));
  static_assert(tp4_graph_kernel_strategy_valid(split));
  static_assert(tp4_graph_kernel_strategy_valid(tiered));
  static_assert(!tp4_graph_kernel_strategy_valid(invalid));
  static_assert(!tp4_graph_kernel_strategy_is_graph_only(fused));
  static_assert(tp4_graph_kernel_strategy_is_graph_only(split));
  static_assert(tp4_graph_kernel_strategy_is_graph_only(tiered));

  static_assert(!tp4_graph_kernel_uses_split(fused, 512));
  static_assert(tp4_graph_kernel_uses_split(split, 1));
  static_assert(!tp4_graph_kernel_uses_split(tiered, 1));
  static_assert(!tp4_graph_kernel_uses_split(tiered, 6));
  static_assert(tp4_graph_kernel_uses_split(tiered, 7));
  static_assert(tp4_graph_kernel_uses_split(tiered, 512));
  static_assert(!tp4_graph_kernel_uses_split(invalid, 512));

  assert(std::string_view(tp4_graph_kernel_strategy_name(fused)) ==
         "fused");
  assert(std::string_view(tp4_graph_kernel_strategy_name(split)) ==
         "split_64k");
  assert(std::string_view(tp4_graph_kernel_strategy_name(tiered)) ==
         "tiered_64k");
  assert(std::string_view(tp4_graph_kernel_strategy_name(invalid)) ==
         "invalid");

  // Default construction is part of the public C++ compatibility contract.
  const Tp4AllreduceOptions options;
  assert(options.graph_kernel_strategy == fused);
}
