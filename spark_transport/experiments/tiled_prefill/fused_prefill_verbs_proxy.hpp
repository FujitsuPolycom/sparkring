#pragma once

#include "fused_prefill_abi.hpp"

#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/tp4_bidirectional_prefill.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace spark_transport::tiled_prefill_research {

constexpr std::uint32_t kFusedPrefillRailCount = 2;
constexpr std::uint32_t kFusedPrefillEndpointCount = 4;
constexpr std::uint64_t kFusedPrefillFlowPlaneBytes =
    4U * kFusedPrefillRailPlaneBytes;
constexpr std::uint64_t kFusedPrefillFlowStride =
    kFusedPrefillFlowPlaneBytes + sizeof(FusedPrefillHostControl);
constexpr std::uint64_t kFusedPrefillArenaBytes =
    kFusedPrefillFlows * kFusedPrefillFlowStride;

enum class FusedPrefillPlane : std::uint32_t {
  kOutgoingPrimary = 0,
  kOutgoingSecondary = 1,
  kIncomingPrimary = 2,
  kIncomingSecondary = 3,
};

constexpr std::uint32_t fused_prefill_flow(std::int32_t direction,
                                            std::uint32_t tile) {
  if ((direction != 1 && direction != -1) ||
      tile >= kFusedPrefillTilesPerShard) {
    throw "invalid fused prefill flow";
  }
  return (direction == 1 ? 0U : kFusedPrefillTilesPerShard) + tile;
}

constexpr std::uint64_t fused_prefill_flow_base(std::uint32_t flow) {
  if (flow >= kFusedPrefillFlows) throw "invalid fused prefill flow";
  return static_cast<std::uint64_t>(flow) * kFusedPrefillFlowStride;
}

constexpr std::uint64_t fused_prefill_plane_offset(
    std::uint32_t flow, FusedPrefillPlane plane) {
  return fused_prefill_flow_base(flow) +
         static_cast<std::uint32_t>(plane) * kFusedPrefillRailPlaneBytes;
}

constexpr std::uint64_t fused_prefill_slot_offset(
    std::uint32_t flow, FusedPrefillPlane plane, std::uint32_t parity) {
  if (parity >= kFusedPrefillParitySlots) {
    throw "invalid fused prefill parity";
  }
  return fused_prefill_plane_offset(flow, plane) +
         static_cast<std::uint64_t>(parity) * kFusedPrefillRailBytes;
}

constexpr std::uint64_t fused_prefill_control_offset(std::uint32_t flow) {
  return fused_prefill_flow_base(flow) + kFusedPrefillFlowPlaneBytes;
}

constexpr std::uint32_t fused_prefill_endpoint_index(
    std::int32_t direction, std::uint32_t rail) {
  if ((direction != 1 && direction != -1) ||
      rail >= kFusedPrefillRailCount) {
    throw "invalid fused prefill endpoint";
  }
  return (direction == 1 ? 0U : kFusedPrefillRailCount) + rail;
}

constexpr FusedPrefillPlane fused_prefill_outgoing_plane(
    std::uint32_t rail) {
  if (rail >= kFusedPrefillRailCount) throw "invalid fused prefill rail";
  return rail == 0 ? FusedPrefillPlane::kOutgoingPrimary
                   : FusedPrefillPlane::kOutgoingSecondary;
}

constexpr FusedPrefillPlane fused_prefill_incoming_plane(
    std::uint32_t rail) {
  if (rail >= kFusedPrefillRailCount) throw "invalid fused prefill rail";
  return rail == 0 ? FusedPrefillPlane::kIncomingPrimary
                   : FusedPrefillPlane::kIncomingSecondary;
}

struct FusedPrefillArenaView {
  std::uint8_t* registered_host{};
  std::uint8_t* registered_device{};
  std::uint8_t* host{};
  std::uint8_t* device{};
  std::size_t bytes{};
  std::size_t registered_bytes{};
  std::uint64_t registered_offset{};

  FusedPrefillHostControl* host_control(std::uint32_t flow) const;
  FusedPrefillHostControl* device_control(std::uint32_t flow) const;
  std::uint8_t* device_plane(std::uint32_t flow,
                             FusedPrefillPlane plane) const;
};

// All four endpoints must have been constructed over the same CUDA-mapped
// MemoryBuffer. VerbsEndpoint consequently registers this one arena in every
// QP's protection domain while retaining distinct QP/CQ ownership per rail.
struct FusedPrefillVerbsProxyConfig {
  std::uint32_t rank{};
  std::int32_t cpu{-1};
  std::uint32_t timeout_milliseconds{5000};
  std::array<VerbsEndpoint*, kFusedPrefillEndpointCount> endpoints{};
};

struct FusedPrefillVerbsProxyReceipt {
  std::uint64_t operation_sequence{};
  std::uint64_t payload_writes{};
  std::uint64_t doorbell_writes{};
  std::uint64_t credit_writes{};
  std::uint64_t completions{};
  std::uint64_t cq_batches{};
  std::uint64_t spin_passes{};
  std::array<std::uint64_t, kFusedPrefillEndpointCount> payload_bytes{};
  std::array<std::uint64_t, kFusedPrefillEndpointCount> doorbell_bytes{};
  std::array<std::uint64_t, kFusedPrefillEndpointCount> credit_bytes{};
  std::array<std::uint64_t, kFusedPrefillEndpointCount> cq_completions{};
};

FusedPrefillArenaView make_fused_prefill_arena_view(
    MemoryBuffer& arena, std::uint32_t operation_slot = 0);

FusedPrefillDescriptor make_fused_prefill_descriptor(
    const FusedPrefillArenaView& arena, FusedPrefillDeviceSync* device_sync,
    const void* input, void* output, std::uint32_t rank,
    std::int32_t direction, std::uint32_t tile,
    std::uint64_t operation_sequence, std::uint32_t spin_limit,
    std::uint64_t payload_bytes = kFusedPrefillPayloadBytes,
    std::uint32_t operation_slots = 1);

class FusedPrefillVerbsProxy {
 public:
  FusedPrefillVerbsProxy(const FusedPrefillVerbsProxy&) = delete;
  FusedPrefillVerbsProxy& operator=(const FusedPrefillVerbsProxy&) = delete;

  FusedPrefillVerbsProxy(FusedPrefillArenaView arena,
                          FusedPrefillVerbsProxyConfig config);
  ~FusedPrefillVerbsProxy();

  // Run this on the sole proxy thread concurrently with the fused kernel.
  // The method pins that thread when cpu >= 0 and returns only after every
  // signaled write on all four QPs has been drained.
  FusedPrefillVerbsProxyReceipt run_operation(
      std::uint64_t operation_sequence,
      std::uint64_t rail_bytes = kFusedPrefillRailBytes,
      std::uint32_t operation_slot = 0);
  bool poisoned() const noexcept;

 private:
  class Impl;
  Impl* impl_{};
};

static_assert(kFusedPrefillFlowStride % kFusedPrefillPlaneAlignment == 0);
static_assert(kFusedPrefillArenaBytes ==
              64U * 1024U * 1024U + 1024U);
static_assert(fused_prefill_slot_offset(
                  fused_prefill_flow(1, 3),
                  FusedPrefillPlane::kOutgoingSecondary, 1) %
                  kFusedPrefillPlaneAlignment ==
              0);

}  // namespace spark_transport::tiled_prefill_research
