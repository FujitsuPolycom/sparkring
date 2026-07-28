#include "spark_transport/gpu_tp2.hpp"

#include <cassert>

int main() {
  const auto layout = spark_transport::make_tp2_buffer_layout(16 * 1024);
  assert(layout.send_offset == 0);
  assert(layout.receive_offset == 16 * 1024);
  assert(layout.control_offset == 32 * 1024);
  assert(layout.total_bytes == 32 * 1024 + 64);
}
