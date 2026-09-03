#pragma once

#include <cstdint>

namespace spark_transport {

// Research-only CUDA kernel selection for graph TP4 all-reduce. The public C
// ABI and vLLM adapter do not expose this selector; their default-constructed
// C++ options remain on the fused kernel.
enum class Tp4GraphKernelStrategy : std::uint8_t {
  kFused = 0,
  kSplit64KiB = 1,
  // Research-only per-node selector. One graph session retains the fused
  // latency path for decode-sized nodes and selects the split grid for larger
  // nodes without changing transport sequence or payload-slot ownership.
  kTiered64KiB = 2,
};

// The fused kernel is the measured latency path through one 64-KiB tile.
// Selecting by bytes keeps the crossover stable when row width changes.
constexpr std::uint32_t kTp4TieredFusedMaximumBytes = 64U * 1024U;

constexpr bool tp4_graph_kernel_strategy_valid(
    Tp4GraphKernelStrategy strategy) noexcept {
  switch (strategy) {
    case Tp4GraphKernelStrategy::kFused:
    case Tp4GraphKernelStrategy::kSplit64KiB:
    case Tp4GraphKernelStrategy::kTiered64KiB:
      return true;
  }
  return false;
}

constexpr bool tp4_graph_kernel_strategy_is_graph_only(
    Tp4GraphKernelStrategy strategy) noexcept {
  return strategy == Tp4GraphKernelStrategy::kSplit64KiB ||
         strategy == Tp4GraphKernelStrategy::kTiered64KiB;
}

constexpr bool tp4_graph_kernel_uses_split(
    Tp4GraphKernelStrategy strategy,
    std::uint32_t active_payload_bytes) noexcept {
  switch (strategy) {
    case Tp4GraphKernelStrategy::kFused:
      return false;
    case Tp4GraphKernelStrategy::kSplit64KiB:
      return true;
    case Tp4GraphKernelStrategy::kTiered64KiB:
      return active_payload_bytes > kTp4TieredFusedMaximumBytes;
  }
  return false;
}

constexpr const char* tp4_graph_kernel_strategy_name(
    Tp4GraphKernelStrategy strategy) noexcept {
  switch (strategy) {
    case Tp4GraphKernelStrategy::kFused:
      return "fused";
    case Tp4GraphKernelStrategy::kSplit64KiB:
      return "split_64k";
    case Tp4GraphKernelStrategy::kTiered64KiB:
      return "tiered_64k";
  }
  return "invalid";
}

}  // namespace spark_transport
