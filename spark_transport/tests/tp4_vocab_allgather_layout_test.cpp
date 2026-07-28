#include "spark_transport/gpu_tp4_vocab_allgather.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

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

std::uint16_t expected(std::uint32_t query,
                       std::uint32_t rank,
                       std::size_t column) {
  return static_cast<std::uint16_t>(
      query * 1000U + rank * 100U + column % 97U);
}

}  // namespace

int main() {
  using spark_transport::kTp4VocabBytesPerRankRow;
  using spark_transport::kTp4VocabElementBytes;
  using spark_transport::kTp4VocabMaxQueryRows;
  using spark_transport::kTp4VocabShardElements;
  using spark_transport::kTp4VocabWorldSize;

  static_assert(kTp4VocabBytesPerRankRow == 77440);
  static_assert(kTp4VocabShardElements == 38720);
  static_assert(kTp4VocabElementBytes == sizeof(std::uint16_t));

  for (std::uint32_t q = 1; q <= kTp4VocabMaxQueryRows; ++q) {
    assert(spark_transport::tp4_vocab_input_bytes(q) ==
           static_cast<std::size_t>(q) *
               kTp4VocabBytesPerRankRow);
    assert(spark_transport::tp4_vocab_output_bytes(q) ==
           static_cast<std::size_t>(q) *
               kTp4VocabBytesPerRankRow * kTp4VocabWorldSize);

    std::vector<std::uint16_t> rank_major(
        spark_transport::tp4_vocab_output_bytes(q) /
        sizeof(std::uint16_t));
    std::vector<std::uint16_t> token_major(rank_major.size());
    for (std::uint32_t rank = 0; rank < kTp4VocabWorldSize; ++rank) {
      for (std::uint32_t query = 0; query < q; ++query) {
        for (std::size_t column = 0;
             column < kTp4VocabShardElements; ++column) {
          const std::size_t byte_in_row =
              column * sizeof(std::uint16_t);
          const std::size_t source =
              spark_transport::tp4_vocab_rank_major_offset(
                  q, query, rank, byte_in_row) /
              sizeof(std::uint16_t);
          const std::size_t destination =
              spark_transport::tp4_vocab_token_major_offset(
                  query, rank, byte_in_row) /
              sizeof(std::uint16_t);
          const std::size_t pair_bundle =
              spark_transport::tp4_vocab_pair_bundle_offset(
                  q, query, rank, byte_in_row);
          const std::size_t expected_pair_bundle =
              (static_cast<std::size_t>(rank & 1U) * q + query) *
                  kTp4VocabBytesPerRankRow +
              byte_in_row;
          assert(pair_bundle == expected_pair_bundle);
          rank_major[source] = expected(query, rank, column);
          token_major[destination] = rank_major[source];
        }
      }
    }

    for (std::uint32_t query = 0; query < q; ++query) {
      for (std::uint32_t rank = 0; rank < kTp4VocabWorldSize; ++rank) {
        for (std::size_t column = 0;
             column < kTp4VocabShardElements; ++column) {
          const std::size_t output_index =
              (static_cast<std::size_t>(query) *
                   kTp4VocabWorldSize *
                   kTp4VocabShardElements +
               static_cast<std::size_t>(rank) *
                   kTp4VocabShardElements +
               column);
          assert(token_major[output_index] ==
                 expected(query, rank, column));
        }
      }
    }
  }

  expect_invalid(
      [] { static_cast<void>(spark_transport::tp4_vocab_input_bytes(0)); });
  expect_invalid(
      [] {
        static_cast<void>(spark_transport::tp4_vocab_input_bytes(
            spark_transport::kTp4VocabMaxQueryRows + 1));
      });
  expect_invalid([] {
    static_cast<void>(
        spark_transport::tp4_vocab_rank_major_offset(3, 3, 0, 0));
  });
  expect_invalid([] {
    static_cast<void>(
        spark_transport::tp4_vocab_rank_major_offset(3, 0, 4, 0));
  });
  expect_invalid([] {
    static_cast<void>(
        spark_transport::tp4_vocab_pair_bundle_offset(3, 3, 0, 0));
  });
  expect_invalid([] {
    static_cast<void>(spark_transport::tp4_vocab_token_major_offset(
        0, 0, spark_transport::kTp4VocabBytesPerRankRow));
  });

  const auto layout =
      spark_transport::make_tp4_vocab_allgather_buffer_layout();
  assert(layout.max_input_bytes ==
         kTp4VocabMaxQueryRows * kTp4VocabBytesPerRankRow);
  assert(layout.max_output_bytes ==
         kTp4VocabMaxQueryRows * kTp4VocabBytesPerRankRow *
             kTp4VocabWorldSize);
  assert(layout.round0.send_offset == 0);
  assert(layout.round0.receive_offset >= layout.max_input_bytes);
  assert(layout.round0.control_offset >=
         layout.round0.receive_offset + layout.max_input_bytes);
  assert(layout.round1.send_offset == 0);
  assert(layout.round1.receive_offset >= layout.max_input_bytes * 2);
  assert(layout.round1.control_offset >=
         layout.round1.receive_offset + layout.max_input_bytes * 2);
  return 0;
}
