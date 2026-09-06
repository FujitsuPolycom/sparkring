#include "tiled_correctness_oracle.hpp"

#include <cassert>
#include <cstdint>

namespace oracle = spark_transport::tiled_prefill_research;

constexpr auto kQ40 = oracle::oracle_payload_geometry(40U);
constexpr auto kQ513 = oracle::oracle_payload_geometry(513U);
constexpr auto kQ1025 = oracle::oracle_payload_geometry(1025U);
constexpr auto kGlmQ2048 = oracle::oracle_payload_geometry(2048U, 4096U);

static_assert(kQ40.active_bytes == 491520U);
static_assert(kQ40.capacity_bytes == 524288U);
static_assert(kQ40.inactive_bytes == 32768U);
static_assert(kQ513.active_bytes == 6303744U);
static_assert(kQ513.capacity_bytes == 6815744U);
static_assert(kQ513.inactive_bytes == 512000U);
static_assert(kQ1025.active_bytes == 12595200U);
static_assert(kQ1025.capacity_bytes == 13107200U);
static_assert(kQ1025.inactive_bytes == 512000U);
static_assert(kGlmQ2048.active_bytes == 16U * 1024U * 1024U);
static_assert(kGlmQ2048.capacity_bytes == kGlmQ2048.active_bytes);
static_assert(kGlmQ2048.inactive_bytes == 0U);

int main() {
  const std::uint16_t lower = oracle::expected_output_bf16_bits(
      0U, 7U, oracle::TiledOracleHalf::kLowerXor1ThenXor3);
  const std::uint16_t upper = oracle::expected_output_bf16_bits(
      0U, 7U, oracle::TiledOracleHalf::kUpperXor3ThenXor1);
  assert(lower == 0x4384U);  // BF16 264.
  assert(upper == 0x4382U);  // BF16 260.
  assert(lower != upper);

  for (std::uint32_t rank = 0U; rank < 4U; ++rank) {
    for (std::uint64_t generation = 1U; generation <= 7U; ++generation) {
      assert(oracle::input_bf16_bits(rank, 19U, generation) !=
             oracle::input_bf16_bits(rank, 19U, generation + 8U));
    }
  }
  for (std::uint64_t generation = 1U; generation <= 14U; ++generation) {
    for (std::uint64_t element = 0U; element < 14U; ++element) {
      assert(oracle::expected_output_bf16_bits(
                 element, generation,
                 oracle::TiledOracleHalf::kLowerXor1ThenXor3) !=
             oracle::expected_output_bf16_bits(
                 element, generation,
                 oracle::TiledOracleHalf::kUpperXor3ThenXor1));
    }
  }
  return 0;
}
