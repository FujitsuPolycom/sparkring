#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace spark_transport::tiled_prefill_research {

#if defined(__CUDACC__)
#define SPARK_FUSED_PREFILL_HD __host__ __device__
#else
#define SPARK_FUSED_PREFILL_HD
#endif

constexpr std::uint32_t kFusedPrefillRanks = 4;
constexpr std::uint32_t kFusedPrefillDirections = 2;
constexpr std::uint32_t kFusedPrefillTilesPerShard = 4;
constexpr std::uint32_t kFusedPrefillFlows = 8;
constexpr std::uint32_t kFusedPrefillStages = 6;
constexpr std::uint32_t kFusedPrefillBarrierPhases = 7;
constexpr std::uint32_t kFusedPrefillParitySlots = 2;
constexpr std::uint32_t kFusedPrefillCtasPerFlow = 4;
constexpr std::uint32_t kFusedPrefillThreads = 256;
constexpr std::uint32_t kFusedPrefillQueryRows = 8192;
constexpr std::uint32_t kFusedPrefillElementsPerRow = 4096;
constexpr std::uint64_t kFusedPrefillPayloadBytes =
    static_cast<std::uint64_t>(kFusedPrefillQueryRows) *
    kFusedPrefillElementsPerRow * 2U;
constexpr std::uint64_t kFusedPrefillHalfBytes =
    kFusedPrefillPayloadBytes / kFusedPrefillDirections;
constexpr std::uint64_t kFusedPrefillShardBytes =
    kFusedPrefillHalfBytes / kFusedPrefillRanks;
constexpr std::uint64_t kFusedPrefillTileBytes =
    kFusedPrefillShardBytes / kFusedPrefillTilesPerShard;
constexpr std::uint64_t kFusedPrefillRailBytes =
    kFusedPrefillTileBytes / 2U;
constexpr std::uint64_t kFusedPrefillCtaBytes =
    kFusedPrefillTileBytes / kFusedPrefillCtasPerFlow;
constexpr std::uint64_t kFusedPrefillRailPlaneBytes =
    kFusedPrefillParitySlots * kFusedPrefillRailBytes;
constexpr std::uint64_t kFusedPrefillPlaneAlignment = 128;

// One cache-line pair per flow, mapped into both host and device address spaces.
// Producer is the GPU -> CPU-proxy outgoing-ready handoff. Doorbells and reuse
// are CPU-proxy -> GPU tokens; consumer returns inbound credit to the proxy.
// A slot is not reusable until the CPU proxy has
// observed both rail CQEs and the peer credit, then publishes reuse[parity].
struct alignas(128) FusedPrefillHostControl {
  std::uint64_t producer[kFusedPrefillParitySlots]{};
  std::uint64_t primary_doorbell[kFusedPrefillParitySlots]{};
  std::uint64_t secondary_doorbell[kFusedPrefillParitySlots]{};
  std::uint64_t consumer[kFusedPrefillParitySlots]{};
  std::uint64_t reuse[kFusedPrefillParitySlots]{};
  // Written by the peer's primary QP after its GPU consumes our payload.
  // The local proxy requires this exact token plus both local payload/DB
  // CQEs before it releases the corresponding parity slot for reuse.
  std::uint64_t peer_credit[kFusedPrefillParitySlots]{};
  std::uint64_t poison_sequence{};
  std::uint64_t reserved[3]{};
};

// Kept in device memory. Putting the 32-CTA barriers in mapped host memory
// adds PCIe atomics to every stage and obscures the transport experiment.
struct alignas(64) FusedPrefillDeviceSync {
  // Phase zero covers the GPU-produced initial exchange. Phases one through
  // six cover consumption of network stages zero through five.
  std::uint32_t arrivals[kFusedPrefillBarrierPhases]{};
  std::uint32_t sense[kFusedPrefillBarrierPhases]{};
  std::uint32_t poison{};
  std::uint32_t reserved{};
};

// One descriptor per flow. The six tensor offsets name the shard consumed by
// that flow at each RS/AG stage. Endpoint arrays contain two parity slots per
// flow and rail; each rail slot is exactly 1 MiB. The four rail-plane bases
// must name disjoint, 128-byte-aligned 2 MiB ranges even when all four live in
// one registered MR. Primary and secondary must never share a plane base.
struct alignas(16) FusedPrefillDescriptor {
  const std::uint8_t* input{};
  std::uint8_t* output{};
  const std::uint8_t* primary_incoming{};
  const std::uint8_t* secondary_incoming{};
  std::uint8_t* primary_outgoing{};
  std::uint8_t* secondary_outgoing{};
  FusedPrefillHostControl* host_control{};
  FusedPrefillDeviceSync* device_sync{};
  std::uint64_t initial_tensor_offset_bytes{};
  std::uint64_t tensor_offset_bytes[kFusedPrefillStages]{};
  std::uint64_t operation_sequence{};
  // Active bytes for this call. The registered planes retain Q8192 capacity,
  // while geometry for any Q <= 8192 is derived from this value.
  std::uint64_t payload_bytes{kFusedPrefillPayloadBytes};
  std::uint32_t operation_slots{1};
  std::uint32_t rank{};
  std::int32_t direction{};
  std::uint32_t tile{};
  std::uint32_t spin_limit{};
};

SPARK_FUSED_PREFILL_HD constexpr std::uint64_t fused_prefill_stage_token(
    std::uint64_t operation_sequence, std::uint32_t stage) noexcept {
  return operation_sequence * kFusedPrefillStages + stage + 1U;
}

SPARK_FUSED_PREFILL_HD constexpr std::uint32_t fused_prefill_parity(
    std::uint32_t stage) noexcept {
  return stage & 1U;
}

static_assert(kFusedPrefillPayloadBytes == 64U * 1024U * 1024U);
static_assert(kFusedPrefillTileBytes == 2U * 1024U * 1024U);
static_assert(kFusedPrefillRailBytes == 1024U * 1024U);
static_assert(kFusedPrefillRailPlaneBytes == 2U * 1024U * 1024U);
static_assert(kFusedPrefillCtaBytes == 512U * 1024U);
static_assert(sizeof(FusedPrefillHostControl) == 128);
static_assert(alignof(FusedPrefillHostControl) == 128);
static_assert(sizeof(FusedPrefillDeviceSync) == 64);
static_assert(sizeof(FusedPrefillDescriptor) == 160);
static_assert(std::is_trivially_copyable_v<FusedPrefillDescriptor>);

#undef SPARK_FUSED_PREFILL_HD

}  // namespace spark_transport::tiled_prefill_research
