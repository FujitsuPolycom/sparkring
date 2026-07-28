#include "spark_transport/gpu_tp4_dcp_combine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace spark_transport {
namespace {

void validate_q(std::uint32_t q) {
  if (q == 0 || q > kTp4DcpCombineMaxQ) {
    throw std::invalid_argument("TP4 DCP combine Q must be in [1, 40]");
  }
}

void validate_rank(std::uint32_t rank) {
  if (rank >= kTp4DcpCombineWorldSize) {
    throw std::invalid_argument("TP4 DCP combine rank must be in [0, 3]");
  }
}

void validate_head_dimension(std::uint32_t head_dimension) {
  if (!tp4_dcp_combine_head_dimension_supported(head_dimension)) {
    throw std::invalid_argument(
        "TP4 DCP combine head dimension must be 256 or 512");
  }
}

Tp4DcpCombineFrameLayout make_frame(std::uint32_t q,
                                    std::size_t heads,
                                    std::uint32_t head_dimension) {
  validate_q(q);
  validate_head_dimension(head_dimension);
  Tp4DcpCombineFrameLayout frame{};
  frame.heads = heads;
  frame.output_bytes =
      static_cast<std::size_t>(q) * heads *
      head_dimension * kTp4DcpCombineOutputElementBytes;
  frame.lse_offset = frame.output_bytes;
  frame.total_bytes =
      frame.output_bytes +
      static_cast<std::size_t>(q) * heads *
          kTp4DcpCombineLseElementBytes;
  return frame;
}

}  // namespace

bool tp4_dcp_combine_head_dimension_supported(
    std::uint32_t head_dimension) {
  return head_dimension == kTp4DcpCombineProjectedHeadDimension ||
         head_dimension == kTp4DcpCombineLatentHeadDimension;
}

std::size_t tp4_dcp_combine_input_output_bytes(
    std::uint32_t q, std::uint32_t head_dimension) {
  validate_q(q);
  validate_head_dimension(head_dimension);
  return static_cast<std::size_t>(q) * kTp4DcpCombineGlobalHeads *
         head_dimension *
         kTp4DcpCombineOutputElementBytes;
}

std::size_t tp4_dcp_combine_input_lse_bytes(std::uint32_t q) {
  validate_q(q);
  return static_cast<std::size_t>(q) * kTp4DcpCombineGlobalHeads *
         kTp4DcpCombineLseElementBytes;
}

std::size_t tp4_dcp_combine_reduced_output_bytes(
    std::uint32_t q, std::uint32_t head_dimension) {
  validate_q(q);
  validate_head_dimension(head_dimension);
  return static_cast<std::size_t>(q) * kTp4DcpCombineHeadsPerRank *
         head_dimension *
         kTp4DcpCombineOutputElementBytes;
}

std::size_t tp4_dcp_combine_reduced_lse_bytes(std::uint32_t q) {
  validate_q(q);
  return static_cast<std::size_t>(q) * kTp4DcpCombineHeadsPerRank *
         kTp4DcpCombineLseElementBytes;
}

Tp4DcpCombineFrameLayout tp4_dcp_combine_round0_frame(
    std::uint32_t q, std::uint32_t head_dimension) {
  return make_frame(
      q, kTp4DcpCombineHeadsPerRank * 2, head_dimension);
}

Tp4DcpCombineFrameLayout tp4_dcp_combine_round1_frame(
    std::uint32_t q, std::uint32_t head_dimension) {
  return make_frame(q, kTp4DcpCombineHeadsPerRank, head_dimension);
}

std::array<std::uint32_t, 2> tp4_dcp_combine_round0_keep_chunks(
    std::uint32_t rank) {
  validate_rank(rank);
  return rank == 0 || rank == 3
             ? std::array<std::uint32_t, 2>{0, 3}
             : std::array<std::uint32_t, 2>{1, 2};
}

std::array<std::uint32_t, 2> tp4_dcp_combine_round0_send_chunks(
    std::uint32_t rank) {
  validate_rank(rank);
  return rank == 0 || rank == 3
             ? std::array<std::uint32_t, 2>{1, 2}
             : std::array<std::uint32_t, 2>{0, 3};
}

std::uint32_t tp4_dcp_combine_round1_send_chunk(
    std::uint32_t rank) {
  validate_rank(rank);
  return rank ^ 3U;
}

float tp4_dcp_sanitize_lse(float value) {
  if (std::isnan(value) ||
      value == std::numeric_limits<float>::infinity()) {
    return -std::numeric_limits<float>::infinity();
  }
  return value;
}

Tp4DcpSoftmaxScalarState tp4_dcp_merge_scalar_state(
    Tp4DcpSoftmaxScalarState a, Tp4DcpSoftmaxScalarState b) {
  a.lse = tp4_dcp_sanitize_lse(a.lse);
  b.lse = tp4_dcp_sanitize_lse(b.lse);
  const float negative_infinity =
      -std::numeric_limits<float>::infinity();
  if (a.lse == negative_infinity && b.lse == negative_infinity) {
    return {0.0F, negative_infinity};
  }
  const float maximum = std::max(a.lse, b.lse);
  const float weight_a = std::exp(a.lse - maximum);
  const float weight_b = std::exp(b.lse - maximum);
  const float denominator = weight_a + weight_b;
  const float weighted_a =
      weight_a == 0.0F ? 0.0F : weight_a * a.output;
  const float weighted_b =
      weight_b == 0.0F ? 0.0F : weight_b * b.output;
  return {
      (weighted_a + weighted_b) / denominator,
      maximum + std::log(denominator),
  };
}

}  // namespace spark_transport
