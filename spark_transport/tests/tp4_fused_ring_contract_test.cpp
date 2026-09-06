#include "spark_transport/tp4_fused_ring_contract.hpp"

#include <cassert>

int main() {
  using namespace spark_transport;
  for (std::uint32_t flow = 0; flow < kFusedRingFlows; ++flow) {
    const auto value = fused_ring_flow(flow);
    assert(value.direction < 2 && value.tile < 4);
  }
  for (std::uint32_t stage = 0; stage < kFusedRingStages; ++stage) {
    for (std::uint32_t tile = 0; tile < 4; ++tile) {
      const auto ordinal = fused_ring_ordinal(24, stage, tile);
      assert(fused_ring_token(ordinal) == ordinal + 1);
      if (stage >= 2) {
        const auto prior = fused_ring_reused_ordinal(24, stage, tile);
        assert(fused_ring_slot(prior) == fused_ring_slot(ordinal));
        assert(!fused_ring_reuse_ready(prior, prior + 1, prior));
        assert(fused_ring_reuse_ready(prior + 1, prior + 1, prior));
      }
    }
  }
}
