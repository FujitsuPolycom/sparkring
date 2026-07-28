#include "spark_transport/gpu_tp4_dcp_query.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
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

}  // namespace

int main() {
  using spark_transport::kTp4DcpQueryBytesPerQ;
  using spark_transport::kTp4DcpQueryMaxQ;
  using spark_transport::kTp4DcpQueryWorldSize;

  for (std::uint32_t q = 1; q <= kTp4DcpQueryMaxQ; ++q) {
    assert(spark_transport::tp4_dcp_query_input_bytes(q) ==
           static_cast<std::size_t>(q) * kTp4DcpQueryBytesPerQ);
    assert(spark_transport::tp4_dcp_query_output_bytes(q) ==
           static_cast<std::size_t>(q) * kTp4DcpQueryBytesPerQ *
               kTp4DcpQueryWorldSize);
  }
  expect_invalid(
      [] { static_cast<void>(spark_transport::tp4_dcp_query_input_bytes(0)); });
  expect_invalid([] {
    static_cast<void>(spark_transport::tp4_dcp_query_input_bytes(
        spark_transport::kTp4DcpQueryMaxQ + 1));
  });

  const auto layout =
      spark_transport::make_tp4_dcp_query_buffer_layout();
  assert(layout.max_input_bytes ==
         kTp4DcpQueryMaxQ * kTp4DcpQueryBytesPerQ);
  assert(layout.max_output_bytes ==
         kTp4DcpQueryMaxQ * kTp4DcpQueryBytesPerQ *
             kTp4DcpQueryWorldSize);
  assert(layout.round0.send_offset == 0);
  assert(layout.round0.receive_offset >= layout.max_input_bytes);
  assert(layout.round0.control_offset >=
         layout.round0.receive_offset + layout.max_input_bytes);
  assert(layout.round1.send_offset == 0);
  assert(layout.round1.receive_offset >= layout.max_input_bytes * 2);
  assert(layout.round1.control_offset >=
         layout.round1.receive_offset + layout.max_input_bytes * 2);

  for (std::uint32_t query_index = 0;
       query_index < kTp4DcpQueryMaxQ; ++query_index) {
    for (std::uint32_t rank = 0; rank < kTp4DcpQueryWorldSize; ++rank) {
      const std::size_t expected =
          static_cast<std::size_t>(query_index) *
              kTp4DcpQueryWorldSize * kTp4DcpQueryBytesPerQ +
          static_cast<std::size_t>(rank) * kTp4DcpQueryBytesPerQ;
      assert(spark_transport::tp4_dcp_query_output_offset(
                 query_index, rank, 0) == expected);
      assert(spark_transport::tp4_dcp_query_output_offset(
                 query_index, rank, kTp4DcpQueryBytesPerQ - 1) ==
             expected + kTp4DcpQueryBytesPerQ - 1);
    }
  }

  expect_invalid([] {
    static_cast<void>(spark_transport::tp4_dcp_query_output_offset(
        spark_transport::kTp4DcpQueryMaxQ, 0, 0));
  });
  expect_invalid([] {
    static_cast<void>(spark_transport::tp4_dcp_query_output_offset(0, 4, 0));
  });
  expect_invalid([] {
    static_cast<void>(spark_transport::tp4_dcp_query_output_offset(
        0, 0, spark_transport::kTp4DcpQueryBytesPerQ));
  });
  return 0;
}
