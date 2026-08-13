#include "spark_transport/tp4_dual_port_striped_allreduce.hpp"
#include "spark_transport/tp4_session.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

int main() {
  using spark_transport::Tp4AllreduceOptions;
  using spark_transport::Tp4AllreduceSchedule;
  using spark_transport::Tp4TensorStripe;
  using spark_transport::kTp4DualPortOppositeOrderStrategy;
  using spark_transport::tp4_striped_phase_transfer;
  using spark_transport::verify_tp4_striped_allreduce;

  const Tp4AllreduceOptions default_options;
  assert(default_options.schedule == Tp4AllreduceSchedule::kSequential);

  constexpr auto verification =
      verify_tp4_striped_allreduce(kTp4DualPortOppositeOrderStrategy);
  static_assert(verification.valid);
  static_assert(verification.complete());

  constexpr spark_transport::Tp4DualPortStripedAllreduceStrategy
      incomplete_strategy{
          std::array<spark_transport::Tp4StripedPhaseStrategy, 2>{
              spark_transport::Tp4StripedPhaseStrategy{
                  Tp4TensorStripe::kLowerHalf,
                  Tp4TensorStripe::kUpperHalf},
              spark_transport::Tp4StripedPhaseStrategy{
                  Tp4TensorStripe::kLowerHalf,
                  Tp4TensorStripe::kUpperHalf},
          }};
  constexpr auto incomplete_verification =
      verify_tp4_striped_allreduce(incomplete_strategy);
  static_assert(incomplete_verification.valid);
  static_assert(!incomplete_verification.complete());

  constexpr spark_transport::Tp4DualPortStripedAllreduceStrategy
      malformed_strategy{
          std::array<spark_transport::Tp4StripedPhaseStrategy, 2>{
              spark_transport::Tp4StripedPhaseStrategy{
                  Tp4TensorStripe::kLowerHalf,
                  Tp4TensorStripe::kLowerHalf},
              spark_transport::Tp4StripedPhaseStrategy{
                  Tp4TensorStripe::kUpperHalf,
                  Tp4TensorStripe::kLowerHalf},
          }};
  constexpr auto malformed_verification =
      verify_tp4_striped_allreduce(malformed_strategy);
  static_assert(!malformed_verification.valid);
  static_assert(!malformed_verification.complete());

  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    const auto phase0_endpoint0 = tp4_striped_phase_transfer(
        kTp4DualPortOppositeOrderStrategy, rank, 0, 0);
    const auto phase0_endpoint1 = tp4_striped_phase_transfer(
        kTp4DualPortOppositeOrderStrategy, rank, 0, 1);
    const auto phase1_endpoint0 = tp4_striped_phase_transfer(
        kTp4DualPortOppositeOrderStrategy, rank, 1, 0);
    const auto phase1_endpoint1 = tp4_striped_phase_transfer(
        kTp4DualPortOppositeOrderStrategy, rank, 1, 1);

    // Tensor half A uses M0 then M1. Tensor half B uses M1 then M0.
    assert(phase0_endpoint0.stripe == Tp4TensorStripe::kLowerHalf);
    assert(phase0_endpoint0.peer_rank == (rank ^ 1U));
    assert(phase0_endpoint1.stripe == Tp4TensorStripe::kUpperHalf);
    assert(phase0_endpoint1.peer_rank == (rank ^ 3U));
    assert(phase1_endpoint0.stripe == Tp4TensorStripe::kUpperHalf);
    assert(phase1_endpoint0.peer_rank == (rank ^ 1U));
    assert(phase1_endpoint1.stripe == Tp4TensorStripe::kLowerHalf);
    assert(phase1_endpoint1.peer_rank == (rank ^ 3U));

    assert(verification.lower_half_contributors[rank] == 0x0fU);
    assert(verification.upper_half_contributors[rank] == 0x0fU);
  }

  constexpr std::size_t q40_payload_bytes = 40U * 6144U * 2U;
  constexpr auto layout =
      spark_transport::make_tp4_striped_endpoint_layout(q40_payload_bytes);
  static_assert(layout.stripe_bytes == q40_payload_bytes / 2U);
  static_assert(layout.lane_stride == q40_payload_bytes + 64U);
  static_assert(layout.generation_stride == 2U * layout.lane_stride);
  static_assert(layout.total_bytes == 2U * layout.generation_stride);

  constexpr std::size_t q512_payload_bytes = 512U * 6144U * 2U;
  constexpr auto q512_layout =
      spark_transport::make_tp4_striped_endpoint_layout(
          q512_payload_bytes);
  static_assert(q512_layout.stripe_bytes == 3'145'728U);
  static_assert(q512_layout.lane_stride == 6'291'520U);
  static_assert(q512_layout.total_bytes == 25'166'080U);

  const std::array<spark_transport::Tp4StripedLaneRegion, 4> regions{
      spark_transport::tp4_striped_lane_region(
          layout, 0, Tp4TensorStripe::kLowerHalf),
      spark_transport::tp4_striped_lane_region(
          layout, 0, Tp4TensorStripe::kUpperHalf),
      spark_transport::tp4_striped_lane_region(
          layout, 1, Tp4TensorStripe::kLowerHalf),
      spark_transport::tp4_striped_lane_region(
          layout, 1, Tp4TensorStripe::kUpperHalf),
  };
  for (std::size_t index = 0; index < regions.size(); ++index) {
    const auto& region = regions[index];
    assert(region.send_offset == index * layout.lane_stride);
    assert(region.receive_offset ==
           region.send_offset + layout.stripe_bytes);
    assert(region.control_offset ==
           region.receive_offset + layout.stripe_bytes);
    assert(region.end_offset == region.control_offset + 64U);
    assert(region.send_offset % 64U == 0);
    assert(region.receive_offset % 64U == 0);
    assert(region.control_offset % 64U == 0);
    if (index != 0) {
      assert(regions[index - 1].end_offset <= region.send_offset);
    }
  }

  static_assert(spark_transport::tp4_striped_generation_slot(1) == 0);
  static_assert(spark_transport::tp4_striped_generation_slot(2) == 1);
  static_assert(spark_transport::tp4_striped_generation_slot(3) == 0);
  bool rejected_zero_sequence{};
  try {
    static_cast<void>(spark_transport::tp4_striped_generation_slot(0));
  } catch (const std::out_of_range&) {
    rejected_zero_sequence = true;
  }
  assert(rejected_zero_sequence);

  bool rejected_layout_overflow{};
  try {
    static_cast<void>(spark_transport::make_tp4_striped_endpoint_layout(
        std::numeric_limits<std::size_t>::max() - 3U));
  } catch (const std::overflow_error&) {
    rejected_layout_overflow = true;
  }
  assert(rejected_layout_overflow);

  bool rejected_partial_bf16_stripes{};
  try {
    static_cast<void>(
        spark_transport::make_tp4_striped_endpoint_layout(2));
  } catch (const std::invalid_argument&) {
    rejected_partial_bf16_stripes = true;
  }
  assert(rejected_partial_bf16_stripes);
}
