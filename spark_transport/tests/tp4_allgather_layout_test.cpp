#include "spark_transport/gpu_tp4_allgather.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>

int main() {
  constexpr std::size_t input_bytes = 753664;
  const auto layout =
      spark_transport::make_tp4_allgather_buffer_layout(input_bytes);
  assert(layout.input_bytes == input_bytes);
  assert(layout.output_bytes == input_bytes * 4);
  assert(layout.round0.receive_offset >= input_bytes);
  assert(layout.round1.receive_offset >= input_bytes * 2);

  for (const std::size_t invalid : {std::size_t{0}, std::size_t{15},
                                    std::size_t{17}}) {
    bool rejected = false;
    try {
      static_cast<void>(
          spark_transport::make_tp4_allgather_buffer_layout(invalid));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }

  bool rejected_overflow = false;
  try {
    static_cast<void>(spark_transport::make_tp4_allgather_buffer_layout(
        std::numeric_limits<std::size_t>::max() - 15));
  } catch (const std::invalid_argument&) {
    rejected_overflow = true;
  }
  assert(rejected_overflow);
}
