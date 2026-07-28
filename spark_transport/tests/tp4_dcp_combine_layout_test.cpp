#include "spark_transport/gpu_tp4_dcp_combine.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

template <typename Function>
void expect_invalid(Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  return value ^ (value >> 16U);
}

}  // namespace

int main() {
  using spark_transport::Tp4DcpSoftmaxScalarState;
  using spark_transport::kTp4DcpCombineMaxQ;

  for (std::uint32_t q = 1; q <= kTp4DcpCombineMaxQ; ++q) {
    assert(spark_transport::tp4_dcp_combine_input_lse_bytes(q) ==
           static_cast<std::size_t>(q) * 256);
    assert(spark_transport::tp4_dcp_combine_reduced_lse_bytes(q) ==
           static_cast<std::size_t>(q) * 64);
    for (const std::uint32_t dimension : {256U, 512U}) {
      assert(
          spark_transport::tp4_dcp_combine_head_dimension_supported(
              dimension));
      assert(spark_transport::tp4_dcp_combine_input_output_bytes(
                 q, dimension) ==
             static_cast<std::size_t>(q) * 64 * dimension * 2);
      assert(spark_transport::tp4_dcp_combine_reduced_output_bytes(
                 q, dimension) ==
             static_cast<std::size_t>(q) * 16 * dimension * 2);

      const std::size_t head_major_query_stride = dimension;
      const std::size_t head_major_head_stride =
          static_cast<std::size_t>(q) * dimension;
      assert(spark_transport::tp4_dcp_combine_strided_output_index(
                 0, 0, 0, head_major_query_stride,
                 head_major_head_stride) == 0);
      assert(spark_transport::tp4_dcp_combine_strided_output_index(
                 q - 1, 0, dimension - 1,
                 head_major_query_stride, head_major_head_stride) ==
             static_cast<std::size_t>(q) * dimension - 1);
      assert(spark_transport::tp4_dcp_combine_strided_output_index(
                 0, 1, 0, head_major_query_stride,
                 head_major_head_stride) ==
             static_cast<std::size_t>(q) * dimension);
      assert(spark_transport::tp4_dcp_combine_strided_output_index(
                 q - 1, 63, dimension - 1,
                 head_major_query_stride, head_major_head_stride) ==
             static_cast<std::size_t>(q) * 64 * dimension - 1);
      assert(
          spark_transport::tp4_dcp_combine_token_major_reduced_output_index(
              0, 1, 0, dimension) == dimension);
      assert(
          spark_transport::tp4_dcp_combine_token_major_reduced_output_index(
              1, 0, 0, dimension) == 16 * dimension);

      const auto round0 =
          spark_transport::tp4_dcp_combine_round0_frame(q, dimension);
      assert(round0.heads == 32);
      assert(round0.output_bytes ==
             static_cast<std::size_t>(q) * 32 * dimension * 2);
      assert(round0.lse_offset == round0.output_bytes);
      assert(round0.total_bytes ==
             static_cast<std::size_t>(q) *
                 (32 * dimension * 2 + 32 * 4));

      const auto round1 =
          spark_transport::tp4_dcp_combine_round1_frame(q, dimension);
      assert(round1.heads == 16);
      assert(round1.output_bytes ==
             static_cast<std::size_t>(q) * 16 * dimension * 2);
      assert(round1.lse_offset == round1.output_bytes);
      assert(round1.total_bytes ==
             static_cast<std::size_t>(q) *
                 (16 * dimension * 2 + 16 * 4));
      assert(round0.total_bytes % 16 == 0);
      assert(round1.total_bytes % 16 == 0);
    }
  }
  assert(!spark_transport::tp4_dcp_combine_head_dimension_supported(384));
  expect_invalid([] {
    static_cast<void>(
        spark_transport::tp4_dcp_combine_round0_frame(0, 256));
  });
  expect_invalid([] {
    static_cast<void>(
        spark_transport::tp4_dcp_combine_round1_frame(
            spark_transport::kTp4DcpCombineMaxQ + 1, 512));
  });
  expect_invalid([] {
    static_cast<void>(
        spark_transport::tp4_dcp_combine_round0_frame(1, 384));
  });

  assert(spark_transport::tp4_dcp_combine_round0_frame(
             kTp4DcpCombineMaxQ, 512)
             .total_bytes ==
         spark_transport::kTp4DcpRound0PayloadCapacity);
  assert(spark_transport::tp4_dcp_combine_round1_frame(
             kTp4DcpCombineMaxQ, 512)
             .total_bytes <=
         spark_transport::kTp4DcpRound1PayloadCapacity);

  assert((spark_transport::tp4_dcp_combine_round0_keep_chunks(0) ==
          std::array<std::uint32_t, 2>{0, 3}));
  assert((spark_transport::tp4_dcp_combine_round0_keep_chunks(1) ==
          std::array<std::uint32_t, 2>{1, 2}));
  assert((spark_transport::tp4_dcp_combine_round0_keep_chunks(2) ==
          std::array<std::uint32_t, 2>{1, 2}));
  assert((spark_transport::tp4_dcp_combine_round0_keep_chunks(3) ==
          std::array<std::uint32_t, 2>{0, 3}));
  assert((spark_transport::tp4_dcp_combine_round0_send_chunks(0) ==
          std::array<std::uint32_t, 2>{1, 2}));
  assert((spark_transport::tp4_dcp_combine_round0_send_chunks(1) ==
          std::array<std::uint32_t, 2>{0, 3}));
  assert((spark_transport::tp4_dcp_combine_round0_send_chunks(2) ==
          std::array<std::uint32_t, 2>{0, 3}));
  assert((spark_transport::tp4_dcp_combine_round0_send_chunks(3) ==
          std::array<std::uint32_t, 2>{1, 2}));
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    assert(spark_transport::tp4_dcp_combine_round1_send_chunk(rank) ==
           (rank ^ 3U));
  }
  expect_invalid([] {
    static_cast<void>(
        spark_transport::tp4_dcp_combine_round0_keep_chunks(4));
  });

  const float infinity = std::numeric_limits<float>::infinity();
  assert(spark_transport::tp4_dcp_sanitize_lse(infinity) == -infinity);
  assert(spark_transport::tp4_dcp_sanitize_lse(-infinity) == -infinity);
  assert(spark_transport::tp4_dcp_sanitize_lse(
             std::numeric_limits<float>::quiet_NaN()) == -infinity);

  const auto empty = spark_transport::tp4_dcp_merge_scalar_state(
      {123.0F, -infinity}, {-456.0F, -infinity});
  assert(empty.output == 0.0F);
  assert(empty.lse == -infinity);

  const auto one_valid = spark_transport::tp4_dcp_merge_scalar_state(
      {std::numeric_limits<float>::quiet_NaN(), infinity},
      {-2.5F, 0.75F});
  assert(one_valid.output == -2.5F);
  assert(one_valid.lse == 0.75F);

  const std::array<Tp4DcpSoftmaxScalarState, 4> states{{
      {1.25F, -0.30F},
      {-2.50F, 0.10F},
      {4.00F, 0.70F},
      {0.50F, -1.20F},
  }};
  const auto pair01 = spark_transport::tp4_dcp_merge_scalar_state(
      states[0], states[1]);
  const auto pair23 = spark_transport::tp4_dcp_merge_scalar_state(
      states[2], states[3]);
  const auto two_round = spark_transport::tp4_dcp_merge_scalar_state(
      pair01, pair23);

  float maximum = states[0].lse;
  for (const auto& state : states) {
    maximum = std::max(maximum, state.lse);
  }
  float denominator = 0.0F;
  float numerator = 0.0F;
  for (const auto& state : states) {
    const float weight = std::exp(state.lse - maximum);
    denominator += weight;
    numerator += weight * state.output;
  }
  assert(close(two_round.output, numerator / denominator));
  assert(close(two_round.lse, maximum + std::log(denominator)));

  for (std::uint32_t q = 1; q <= kTp4DcpCombineMaxQ; ++q) {
    for (std::uint32_t head = 0; head < 64; ++head) {
      std::array<Tp4DcpSoftmaxScalarState, 4> generated{};
      for (std::uint32_t rank = 0; rank < 4; ++rank) {
        const std::uint32_t output_bits =
            mix(0x9e3779b9U ^ q ^ (head << 8U) ^ (rank << 20U));
        const std::uint32_t lse_bits =
            mix(0x243f6a88U ^ q ^ (head << 10U) ^ (rank << 22U));
        generated[rank].output =
            (static_cast<std::int32_t>(output_bits % 4097U) - 2048) /
            512.0F;
        generated[rank].lse =
            (static_cast<std::int32_t>(lse_bits % 12001U) - 6000) /
            1000.0F;
      }
      const auto generated_pair01 =
          spark_transport::tp4_dcp_merge_scalar_state(
              generated[0], generated[1]);
      const auto generated_pair23 =
          spark_transport::tp4_dcp_merge_scalar_state(
              generated[2], generated[3]);
      const auto generated_two_round =
          spark_transport::tp4_dcp_merge_scalar_state(
              generated_pair01, generated_pair23);
      float generated_maximum = generated[0].lse;
      for (const auto& state : generated) {
        generated_maximum =
            std::max(generated_maximum, state.lse);
      }
      float generated_denominator = 0.0F;
      float generated_numerator = 0.0F;
      for (const auto& state : generated) {
        const float weight =
            std::exp(state.lse - generated_maximum);
        generated_denominator += weight;
        generated_numerator += weight * state.output;
      }
      assert(close(generated_two_round.output,
                   generated_numerator / generated_denominator,
                   2.0e-5F));
      assert(close(generated_two_round.lse,
                   generated_maximum +
                       std::log(generated_denominator),
                   2.0e-5F));
    }
  }
  return 0;
}
