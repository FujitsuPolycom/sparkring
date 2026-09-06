#include "fused_prefill_verbs_proxy.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace research = spark_transport::tiled_prefill_research;

int main() {
  static_assert(research::kFusedPrefillEndpointCount == 4);
  static_assert(research::kFusedPrefillArenaBytes ==
                64U * 1024U * 1024U + 1024U);
  for (std::uint32_t flow = 0; flow < research::kFusedPrefillFlows; ++flow) {
    const auto base = research::fused_prefill_flow_base(flow);
    const auto control = research::fused_prefill_control_offset(flow);
    assert(base % research::kFusedPrefillPlaneAlignment == 0);
    assert(control % research::kFusedPrefillPlaneAlignment == 0);
    assert(control + sizeof(research::FusedPrefillHostControl) <=
           research::kFusedPrefillArenaBytes);
    for (std::uint32_t plane = 0; plane < 4; ++plane) {
      const auto offset = research::fused_prefill_plane_offset(
          flow, static_cast<research::FusedPrefillPlane>(plane));
      assert(offset % research::kFusedPrefillPlaneAlignment == 0);
      assert(offset >= base);
      assert(offset + research::kFusedPrefillRailPlaneBytes <= control);
      for (std::uint32_t parity = 0;
           parity < research::kFusedPrefillParitySlots; ++parity) {
        assert(research::fused_prefill_slot_offset(
                   flow, static_cast<research::FusedPrefillPlane>(plane),
                   parity) ==
               offset + parity * research::kFusedPrefillRailBytes);
      }
    }
  }
  assert(research::fused_prefill_flow(1, 0) == 0);
  assert(research::fused_prefill_flow(1, 3) == 3);
  assert(research::fused_prefill_flow(-1, 0) == 4);
  assert(research::fused_prefill_flow(-1, 3) == 7);
  assert(research::fused_prefill_endpoint_index(1, 0) == 0);
  assert(research::fused_prefill_endpoint_index(1, 1) == 1);
  assert(research::fused_prefill_endpoint_index(-1, 0) == 2);
  assert(research::fused_prefill_endpoint_index(-1, 1) == 3);
  return 0;
}
