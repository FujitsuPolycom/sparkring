#pragma once

// Portable, research-only numerical oracle for the standalone tiled-prefill
// probe. Production transport code does not include this header.

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define SPARK_TILED_ORACLE_HOST_DEVICE __host__ __device__
#else
#define SPARK_TILED_ORACLE_HOST_DEVICE
#endif

namespace spark_transport::tiled_prefill_research {

constexpr std::uint32_t kInputGenerationPeriod = 7U;
constexpr std::size_t kOraclePayloadTileBytes = 512U * 1024U;
constexpr std::size_t kOracleModelWidth = 6144U;
constexpr std::size_t kOracleBf16Bytes = 2U;
constexpr std::uint8_t kInputGuardSentinel = 0xA5U;
constexpr std::uint8_t kOutputGuardSentinel = 0x5AU;
constexpr std::uint8_t kInactiveInputSentinel = 0xD3U;
constexpr std::uint8_t kInactiveOutputSentinel = 0x6CU;

static_assert(kInputGenerationPeriod != 8U);
static_assert(kOraclePayloadTileBytes % (2U * kOracleBf16Bytes) == 0U);

enum class TiledOracleHalf : std::uint8_t {
  kLowerXor1ThenXor3 = 1,
  kUpperXor3ThenXor1 = 2,
};

struct TiledPayloadGeometry {
  std::uint64_t active_bytes;
  std::uint64_t capacity_bytes;
  std::uint64_t inactive_bytes;
};

SPARK_TILED_ORACLE_HOST_DEVICE inline constexpr TiledPayloadGeometry
oracle_payload_geometry(
    std::uint32_t query_rows,
    std::size_t model_width = kOracleModelWidth) {
  const std::uint64_t active_bytes =
      static_cast<std::uint64_t>(query_rows) * model_width *
      kOracleBf16Bytes;
  const std::uint64_t tile_count =
      (active_bytes + kOraclePayloadTileBytes - 1U) /
      kOraclePayloadTileBytes;
  const std::uint64_t capacity_bytes =
      tile_count * kOraclePayloadTileBytes;
  return {active_bytes, capacity_bytes, capacity_bytes - active_bytes};
}

SPARK_TILED_ORACLE_HOST_DEVICE inline std::uint32_t oracle_float_bits(
    float value) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(value);
#else
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

SPARK_TILED_ORACLE_HOST_DEVICE inline float oracle_bits_float(
    std::uint32_t bits) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(bits);
#else
  float value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

SPARK_TILED_ORACLE_HOST_DEVICE inline std::uint16_t float_to_bf16_bits(
    float value) {
  std::uint32_t bits = oracle_float_bits(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

SPARK_TILED_ORACLE_HOST_DEVICE inline float bf16_bits_to_float(
    std::uint16_t bits) {
  return oracle_bits_float(static_cast<std::uint32_t>(bits) << 16U);
}

SPARK_TILED_ORACLE_HOST_DEVICE inline std::uint16_t bf16_add_bits(
    std::uint16_t left, std::uint16_t right) {
  // The generated integer inputs and their partials sum exactly in FP32.
  // Round after every tree edge so the oracle models BF16 association.
  return float_to_bf16_bits(
      bf16_bits_to_float(left) + bf16_bits_to_float(right));
}

SPARK_TILED_ORACLE_HOST_DEVICE inline constexpr std::uint16_t
rank_base_bf16_bits(std::uint32_t rank) {
  return rank == 0U   ? 0x3F80U  // 1
         : rank == 1U ? 0x4000U  // 2
         : rank == 2U ? 0x4040U  // 3
                      : 0x4380U;  // 256
}

SPARK_TILED_ORACLE_HOST_DEVICE inline constexpr std::uint16_t
input_bf16_bits(std::uint32_t rank, std::uint64_t element,
                std::uint64_t generation) {
  const std::uint32_t scale_exponent = static_cast<std::uint32_t>(
      (element % kInputGenerationPeriod +
       generation % kInputGenerationPeriod) %
      kInputGenerationPeriod);
  return static_cast<std::uint16_t>(
      rank_base_bf16_bits(rank) + (scale_exponent << 7U));
}

SPARK_TILED_ORACLE_HOST_DEVICE inline std::uint16_t
expected_output_bf16_bits(std::uint64_t element, std::uint64_t generation,
                          TiledOracleHalf half) {
  const std::uint16_t inputs[4]{
      input_bf16_bits(0U, element, generation),
      input_bf16_bits(1U, element, generation),
      input_bf16_bits(2U, element, generation),
      input_bf16_bits(3U, element, generation),
  };
  if (half == TiledOracleHalf::kLowerXor1ThenXor3) {
    const std::uint16_t xor1_left = bf16_add_bits(inputs[0], inputs[1]);
    const std::uint16_t xor1_right = bf16_add_bits(inputs[2], inputs[3]);
    return bf16_add_bits(xor1_left, xor1_right);
  }
  const std::uint16_t xor3_left = bf16_add_bits(inputs[0], inputs[3]);
  const std::uint16_t xor3_right = bf16_add_bits(inputs[1], inputs[2]);
  return bf16_add_bits(xor3_left, xor3_right);
}

}  // namespace spark_transport::tiled_prefill_research

#undef SPARK_TILED_ORACLE_HOST_DEVICE
