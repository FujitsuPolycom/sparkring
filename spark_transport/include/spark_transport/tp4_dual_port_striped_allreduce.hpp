#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace spark_transport {

constexpr std::size_t kTp4StripedRankCount = 4;
constexpr std::size_t kTp4StripedPhaseCount = 2;
constexpr std::size_t kTp4StripedEndpointCount = 2;
constexpr std::size_t kTp4StripedLaneCount = 2;
constexpr std::size_t kTp4StripedGenerationCount = 2;
constexpr std::size_t kTp4StripedControlAlignment = 64;
constexpr std::size_t kTp4StripedControlBytes = 64;

// Internal C++ selector. The public C ABI and vLLM adapter remain on the
// sequential schedule unless a research-only native caller opts in.
enum class Tp4AllreduceSchedule : std::uint8_t {
  kSequential = 0,
  kDualPortStriped = 1,
};

constexpr bool tp4_allreduce_schedule_valid(
    Tp4AllreduceSchedule schedule) noexcept {
  return schedule == Tp4AllreduceSchedule::kSequential ||
         schedule == Tp4AllreduceSchedule::kDualPortStriped;
}

constexpr const char* tp4_allreduce_schedule_name(
    Tp4AllreduceSchedule schedule) noexcept {
  switch (schedule) {
    case Tp4AllreduceSchedule::kSequential:
      return "sequential";
    case Tp4AllreduceSchedule::kDualPortStriped:
      return "dual_port_striped";
  }
  return "invalid";
}

enum class Tp4TensorStripe : std::uint8_t {
  kLowerHalf = 0,
  kUpperHalf = 1,
};

enum class Tp4DirectMatching : std::uint8_t {
  kXor1 = 1,
  kXor3 = 3,
};

struct Tp4StripedPhaseStrategy {
  Tp4TensorStripe endpoint0_stripe{};
  Tp4TensorStripe endpoint1_stripe{};
};

// Research-only schedule description. It is intentionally independent of the
// implemented transport protocol and does not identify a compatible wire
// layout. A transport implementation must use a distinct endpoint identity.
struct Tp4DualPortStripedAllreduceStrategy {
  std::array<Tp4StripedPhaseStrategy, kTp4StripedPhaseCount> phases{};
};

inline constexpr Tp4DualPortStripedAllreduceStrategy
    kTp4DualPortOppositeOrderStrategy{
        std::array<Tp4StripedPhaseStrategy, kTp4StripedPhaseCount>{
            Tp4StripedPhaseStrategy{Tp4TensorStripe::kLowerHalf,
                                    Tp4TensorStripe::kUpperHalf},
            Tp4StripedPhaseStrategy{Tp4TensorStripe::kUpperHalf,
                                    Tp4TensorStripe::kLowerHalf},
        }};

struct Tp4StripedPhaseTransfer {
  std::uint32_t rank{};
  std::uint32_t peer_rank{};
  std::uint32_t phase{};
  std::uint32_t endpoint{};
  Tp4DirectMatching matching{};
  Tp4TensorStripe stripe{};
};

constexpr Tp4DirectMatching tp4_striped_endpoint_matching(
    std::uint32_t endpoint) {
  if (endpoint >= kTp4StripedEndpointCount) {
    throw std::out_of_range("TP4 striped endpoint must be zero or one");
  }
  return endpoint == 0 ? Tp4DirectMatching::kXor1
                       : Tp4DirectMatching::kXor3;
}

constexpr std::uint32_t tp4_striped_matching_mask(
    Tp4DirectMatching matching) noexcept {
  return static_cast<std::uint32_t>(matching);
}

constexpr Tp4StripedPhaseTransfer tp4_striped_phase_transfer(
    const Tp4DualPortStripedAllreduceStrategy& strategy,
    std::uint32_t rank, std::uint32_t phase, std::uint32_t endpoint) {
  if (rank >= kTp4StripedRankCount) {
    throw std::out_of_range("TP4 striped rank must be in [0, 3]");
  }
  if (phase >= kTp4StripedPhaseCount) {
    throw std::out_of_range("TP4 striped phase must be zero or one");
  }
  const Tp4DirectMatching matching =
      tp4_striped_endpoint_matching(endpoint);
  const auto& phase_strategy = strategy.phases[phase];
  const Tp4TensorStripe stripe =
      endpoint == 0 ? phase_strategy.endpoint0_stripe
                    : phase_strategy.endpoint1_stripe;
  return {rank, rank ^ tp4_striped_matching_mask(matching), phase,
          endpoint, matching, stripe};
}

struct Tp4StripedScheduleVerification {
  bool valid{};
  std::array<std::uint8_t, kTp4StripedRankCount>
      lower_half_contributors{};
  std::array<std::uint8_t, kTp4StripedRankCount>
      upper_half_contributors{};

  constexpr bool complete() const noexcept {
    if (!valid) {
      return false;
    }
    for (std::size_t rank = 0; rank < kTp4StripedRankCount; ++rank) {
      if (lower_half_contributors[rank] != 0x0fU ||
          upper_half_contributors[rank] != 0x0fU) {
        return false;
      }
    }
    return true;
  }
};

constexpr bool tp4_tensor_stripe_valid(Tp4TensorStripe stripe) noexcept {
  return stripe == Tp4TensorStripe::kLowerHalf ||
         stripe == Tp4TensorStripe::kUpperHalf;
}

constexpr std::array<std::uint8_t, kTp4StripedRankCount>
tp4_exchange_symbolic_contributors(
    const std::array<std::uint8_t, kTp4StripedRankCount>& contributors,
    std::uint32_t matching_mask) noexcept {
  std::array<std::uint8_t, kTp4StripedRankCount> exchanged{};
  for (std::uint32_t rank = 0; rank < kTp4StripedRankCount; ++rank) {
    exchanged[rank] = static_cast<std::uint8_t>(
        contributors[rank] | contributors[rank ^ matching_mask]);
  }
  return exchanged;
}

constexpr Tp4StripedScheduleVerification verify_tp4_striped_allreduce(
    const Tp4DualPortStripedAllreduceStrategy& strategy) noexcept {
  Tp4StripedScheduleVerification result{};
  for (std::uint32_t rank = 0; rank < kTp4StripedRankCount; ++rank) {
    const auto contributor = static_cast<std::uint8_t>(1U << rank);
    result.lower_half_contributors[rank] = contributor;
    result.upper_half_contributors[rank] = contributor;
  }

  for (const auto& phase : strategy.phases) {
    if (!tp4_tensor_stripe_valid(phase.endpoint0_stripe) ||
        !tp4_tensor_stripe_valid(phase.endpoint1_stripe) ||
        phase.endpoint0_stripe == phase.endpoint1_stripe) {
      return result;
    }
    if (phase.endpoint0_stripe == Tp4TensorStripe::kLowerHalf) {
      result.lower_half_contributors =
          tp4_exchange_symbolic_contributors(
              result.lower_half_contributors, 1U);
      result.upper_half_contributors =
          tp4_exchange_symbolic_contributors(
              result.upper_half_contributors, 3U);
    } else {
      result.upper_half_contributors =
          tp4_exchange_symbolic_contributors(
              result.upper_half_contributors, 1U);
      result.lower_half_contributors =
          tp4_exchange_symbolic_contributors(
              result.lower_half_contributors, 3U);
    }
  }
  result.valid = true;
  return result;
}

struct Tp4StripedEndpointLayout {
  std::size_t payload_bytes{};
  std::size_t stripe_bytes{};
  std::size_t lane_receive_offset{};
  std::size_t lane_control_offset{};
  std::size_t lane_stride{};
  std::size_t generation_stride{};
  std::size_t total_bytes{};
};

struct Tp4StripedLaneRegion {
  std::size_t send_offset{};
  std::size_t receive_offset{};
  std::size_t control_offset{};
  std::size_t end_offset{};
};

constexpr std::size_t tp4_striped_generation_slot(
    std::uint64_t sequence) {
  if (sequence == 0) {
    throw std::out_of_range("TP4 striped sequence must be positive");
  }
  return static_cast<std::size_t>(
      (sequence - 1U) % kTp4StripedGenerationCount);
}

namespace detail {

constexpr std::size_t tp4_striped_checked_add(std::size_t left,
                                              std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error("TP4 striped layout size overflow");
  }
  return left + right;
}

constexpr std::size_t tp4_striped_checked_multiply(std::size_t left,
                                                   std::size_t right) {
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("TP4 striped layout size overflow");
  }
  return left * right;
}

}  // namespace detail

constexpr std::size_t tp4_striped_align_control(
    std::size_t bytes) {
  const std::size_t remainder = bytes % kTp4StripedControlAlignment;
  const std::size_t padding =
      remainder == 0 ? 0 : kTp4StripedControlAlignment - remainder;
  return detail::tp4_striped_checked_add(bytes, padding);
}

constexpr Tp4StripedEndpointLayout make_tp4_striped_endpoint_layout(
    std::size_t payload_bytes) {
  constexpr std::size_t bytes_per_bf16 = 2;
  constexpr std::size_t stripe_count = 2;
  if (payload_bytes == 0 ||
      payload_bytes % (bytes_per_bf16 * stripe_count) != 0) {
    throw std::invalid_argument(
        "TP4 striped payload must split into two nonempty BF16 halves");
  }
  const std::size_t stripe_bytes = payload_bytes / 2U;
  const std::size_t receive_offset =
      tp4_striped_align_control(stripe_bytes);
  const std::size_t control_offset =
      tp4_striped_align_control(
          detail::tp4_striped_checked_add(receive_offset, stripe_bytes));
  const std::size_t lane_stride =
      detail::tp4_striped_checked_add(control_offset,
                                      kTp4StripedControlBytes);
  const std::size_t generation_stride =
      detail::tp4_striped_checked_multiply(lane_stride,
                                          kTp4StripedLaneCount);
  return {payload_bytes,
          stripe_bytes,
          receive_offset,
          control_offset,
          lane_stride,
          generation_stride,
          detail::tp4_striped_checked_multiply(
              generation_stride, kTp4StripedGenerationCount)};
}

constexpr Tp4StripedLaneRegion tp4_striped_lane_region(
    const Tp4StripedEndpointLayout& layout, std::size_t generation,
    Tp4TensorStripe stripe) {
  if (generation >= kTp4StripedGenerationCount) {
    throw std::out_of_range(
        "TP4 striped generation must be zero or one");
  }
  if (!tp4_tensor_stripe_valid(stripe)) {
    throw std::out_of_range("invalid TP4 tensor stripe");
  }
  const std::size_t stripe_index =
      static_cast<std::size_t>(stripe);
  const std::size_t send_offset =
      generation * layout.generation_stride +
      stripe_index * layout.lane_stride;
  return {send_offset,
          send_offset + layout.lane_receive_offset,
          send_offset + layout.lane_control_offset,
          send_offset + layout.lane_stride};
}

}  // namespace spark_transport
