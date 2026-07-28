#include "spark_transport/gpu_doorbell.hpp"

#include <cassert>
#include <cstddef>

int main() {
  using spark_transport::DoorbellControl;
  using spark_transport::aligned_control_offset;

  static_assert(alignof(DoorbellControl) == 64);
  static_assert(sizeof(DoorbellControl) == 64);
  assert(aligned_control_offset(1) == 64);
  assert(aligned_control_offset(64) == 64);
  assert(aligned_control_offset(65) == 128);
  assert(aligned_control_offset(16 * 1024) == 16 * 1024);
}
