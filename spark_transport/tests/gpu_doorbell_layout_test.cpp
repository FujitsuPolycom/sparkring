#include "spark_transport/gpu_doorbell.hpp"

#include <cassert>
#include <cstddef>
#include <type_traits>

int main() {
  using spark_transport::DoorbellControl;
  using spark_transport::aligned_control_offset;

  static_assert(alignof(DoorbellControl) == 64);
  static_assert(sizeof(DoorbellControl) == 64);
  static_assert(std::is_standard_layout_v<DoorbellControl>);
  static_assert(std::is_trivially_copyable_v<DoorbellControl>);
  static_assert(offsetof(DoorbellControl, command_sequence) == 0);
  static_assert(offsetof(DoorbellControl, producer_sequence) == 8);
  static_assert(offsetof(DoorbellControl, remote_sequence) == 16);
  static_assert(offsetof(DoorbellControl, consumer_sequence) == 24);
  static_assert(offsetof(DoorbellControl, acknowledgement_sequence) == 32);
  static_assert(offsetof(DoorbellControl, observed_sequence) == 40);
  static_assert(offsetof(DoorbellControl, mismatch_count) == 48);
  static_assert(offsetof(DoorbellControl, reserved) == 56);
  assert(aligned_control_offset(1) == 64);
  assert(aligned_control_offset(64) == 64);
  assert(aligned_control_offset(65) == 128);
  assert(aligned_control_offset(16 * 1024) == 16 * 1024);
  const auto exchange =
      spark_transport::make_exchange_buffer_layout(16 * 1024);
  assert(exchange.send_offset == 0);
  assert(exchange.receive_offset == 16 * 1024);
  assert(exchange.control_offset == 32 * 1024);
  assert(exchange.total_bytes == 32 * 1024 + 64);
}
