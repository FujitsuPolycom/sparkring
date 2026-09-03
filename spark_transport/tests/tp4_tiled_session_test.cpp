#include "spark_transport/tp4_tiled_session.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

using spark_transport::Tp4SessionStorageOptions;
using spark_transport::Tp4SessionStorageSelector;
using spark_transport::Tp4TiledCapacityClass;

constexpr Tp4SessionStorageOptions kDefaultStorageOptions{};
static_assert(kDefaultStorageOptions.selector ==
              Tp4SessionStorageSelector::kExactPayload);
static_assert(spark_transport::tp4_session_storage_options_valid(
    kDefaultStorageOptions));

void test_capacity_selector_and_inert_default() {
  using spark_transport::make_tp4_tiled_session_storage_options;
  using spark_transport::tp4_select_tiled_capacity_class;
  using spark_transport::tp4_tiled_capacity_maximum_query_rows;

  assert(tp4_select_tiled_capacity_class(1) ==
         Tp4TiledCapacityClass::kLatencyQ40);
  assert(tp4_select_tiled_capacity_class(40) ==
         Tp4TiledCapacityClass::kLatencyQ40);
  assert(tp4_select_tiled_capacity_class(41) ==
         Tp4TiledCapacityClass::kMediumQ512);
  assert(tp4_select_tiled_capacity_class(512) ==
         Tp4TiledCapacityClass::kMediumQ512);
  assert(tp4_select_tiled_capacity_class(513) ==
         Tp4TiledCapacityClass::kLargeQ1024);
  assert(tp4_select_tiled_capacity_class(1024) ==
         Tp4TiledCapacityClass::kLargeQ1024);
  assert(tp4_select_tiled_capacity_class(1025) ==
         Tp4TiledCapacityClass::kStreamingQ4096);
  assert(tp4_select_tiled_capacity_class(4096) ==
         Tp4TiledCapacityClass::kStreamingQ4096);
  assert(tp4_select_tiled_capacity_class(4097) ==
         Tp4TiledCapacityClass::kExtendedQ8192);
  assert(tp4_select_tiled_capacity_class(8192) ==
         Tp4TiledCapacityClass::kExtendedQ8192);
  assert(tp4_tiled_capacity_maximum_query_rows(
             Tp4TiledCapacityClass::kLargeQ1024) == 1024);
  assert(tp4_tiled_capacity_maximum_query_rows(
             Tp4TiledCapacityClass::kStreamingQ4096) == 4096);
  assert(tp4_tiled_capacity_maximum_query_rows(
             Tp4TiledCapacityClass::kExtendedQ8192) == 8192);

  const auto q41 = make_tp4_tiled_session_storage_options(41);
  const auto q512 = make_tp4_tiled_session_storage_options(512);
  assert(q41.selector ==
         Tp4SessionStorageSelector::kCapacityTieredTiles);
  assert(q41.capacity_class == Tp4TiledCapacityClass::kMediumQ512);
  assert(q41.maximum_query_rows == 512);
  assert(q41.tile_payload_bytes == 512U * 1024U);
  assert(q41.slots_per_edge == 8);
  assert(q41.lanes_per_edge == 2);
  assert(q41 == q512);
  assert(spark_transport::tp4_session_storage_options_valid(q41));
  const auto q41_layout =
      spark_transport::make_tp4_tiled_pool_layout(q41);
  assert(q41_layout.tile_payload_bytes == q41.tile_payload_bytes);
  assert(q41_layout.slots_per_edge == q41.slots_per_edge);
  assert(q41_layout.lanes_per_edge == q41.lanes_per_edge);

  const auto q513 = make_tp4_tiled_session_storage_options(513);
  const auto q1024 = make_tp4_tiled_session_storage_options(1024);
  const auto q1025 = make_tp4_tiled_session_storage_options(1025);
  assert(q513.capacity_class == Tp4TiledCapacityClass::kLargeQ1024);
  assert(q513.maximum_query_rows == 1024);
  assert(q513 == q1024);
  assert(q1025.capacity_class ==
         Tp4TiledCapacityClass::kStreamingQ4096);
  assert(q1025.maximum_query_rows == 4096);
  assert(q1024 != q1025);

  auto invalid_geometry = q41;
  invalid_geometry.lanes_per_edge = 1;
  assert(!spark_transport::tp4_session_storage_options_valid(
      invalid_geometry));

  bool rejected_zero{};
  try {
    (void)make_tp4_tiled_session_storage_options(0);
  } catch (const std::out_of_range&) {
    rejected_zero = true;
  }
  assert(rejected_zero);

  bool rejected_unbounded{};
  try {
    (void)make_tp4_tiled_session_storage_options(8193);
  } catch (const std::out_of_range&) {
    rejected_unbounded = true;
  }
  assert(rejected_unbounded);
}

void test_pool_layout_has_disjoint_send_receive_and_control_regions() {
  using spark_transport::make_tp4_tiled_pool_layout;
  using spark_transport::tp4_tiled_slot_region;

  const auto layout = make_tp4_tiled_pool_layout(512U * 1024U, 8, 2);
  assert(layout.tile_payload_bytes == 512U * 1024U);
  assert(layout.slots_per_edge == 8);
  assert(layout.lanes_per_edge == 2);
  assert(layout.lane_payload_bytes == 256U * 1024U);
  assert(layout.lane_receive_offset == 256U * 1024U);
  assert(layout.lane_control_offset == 512U * 1024U);
  assert(layout.lane_stride == 512U * 1024U + 64U);
  assert(layout.slot_stride == 1024U * 1024U + 128U);
  assert(layout.total_bytes == 8U * (1024U * 1024U + 128U));

  const auto first_lower = tp4_tiled_slot_region(layout, 0, 0);
  assert(first_lower.send_offset == 0);
  assert(first_lower.receive_offset == 256U * 1024U);
  assert(first_lower.control_offset == 512U * 1024U);
  assert(first_lower.end_offset == 512U * 1024U + 64U);

  const auto first_upper = tp4_tiled_slot_region(layout, 0, 1);
  assert(first_upper.send_offset == layout.lane_stride);
  assert(first_upper.receive_offset ==
         first_upper.send_offset + 256U * 1024U);
  assert(first_upper.control_offset ==
         first_upper.send_offset + 512U * 1024U);
  assert(first_upper.end_offset == layout.slot_stride);

  const auto last = tp4_tiled_slot_region(layout, 7, 1);
  assert(last.send_offset ==
         7U * layout.slot_stride + layout.lane_stride);
  assert(last.end_offset == layout.total_bytes);

  bool rejected_overflow{};
  try {
    const auto largest_aligned =
        std::numeric_limits<std::size_t>::max() - 63U;
    (void)make_tp4_tiled_pool_layout(largest_aligned, 8, 2);
  } catch (const std::overflow_error&) {
    rejected_overflow = true;
  }
  assert(rejected_overflow);
}

void test_operation_descriptors_cover_only_active_bytes() {
  using spark_transport::make_tp4_tiled_bf16_allreduce_operation;
  using spark_transport::make_tp4_tiled_tile_descriptor;

  const auto operation = make_tp4_tiled_bf16_allreduce_operation(513);
  assert(spark_transport::tp4_tiled_operation_descriptor_valid(operation));
  assert(operation.query_rows == 513);
  assert(operation.active_bytes == 513ULL * 6144ULL * 2ULL);
  assert(operation.tile_count == 13);
  assert(operation.capacity_class ==
         Tp4TiledCapacityClass::kLargeQ1024);

  const auto streaming_operation =
      make_tp4_tiled_bf16_allreduce_operation(1025);
  assert(streaming_operation.capacity_class ==
         Tp4TiledCapacityClass::kStreamingQ4096);

  std::uint64_t described_bytes{};
  for (std::uint32_t tile_index = 0;
       tile_index < operation.tile_count; ++tile_index) {
    const auto tile = make_tp4_tiled_tile_descriptor(
        operation, tile_index, 6);
    assert(tile.operation_offset_bytes ==
           static_cast<std::uint64_t>(tile_index) * 512ULL * 1024ULL);
    assert(tile.active_bytes > 0);
    assert(tile.active_bytes <= 512U * 1024U);
    described_bytes += tile.active_bytes;
  }
  assert(described_bytes == operation.active_bytes);

  const auto first = make_tp4_tiled_tile_descriptor(operation, 0, 6);
  const auto second = make_tp4_tiled_tile_descriptor(operation, 1, 6);
  const auto third = make_tp4_tiled_tile_descriptor(operation, 2, 6);
  const auto tail = make_tp4_tiled_tile_descriptor(operation, 12, 6);
  assert(first.ticket.generation == 1 && first.ticket.slot == 6);
  assert(second.ticket.generation == 1 && second.ticket.slot == 7);
  assert(third.ticket.generation == 2 && third.ticket.slot == 0);
  assert(tail.ticket.generation == 3 && tail.ticket.slot == 2);
  assert(tail.active_bytes == 6144U * 2U);

  auto corrupt_operation = operation;
  --corrupt_operation.active_bytes;
  assert(!spark_transport::tp4_tiled_operation_descriptor_valid(
      corrupt_operation));
  bool rejected_corrupt_operation{};
  try {
    (void)make_tp4_tiled_tile_descriptor(corrupt_operation, 0, 0);
  } catch (const std::invalid_argument&) {
    rejected_corrupt_operation = true;
  }
  assert(rejected_corrupt_operation);
}

void test_glm53_width4096_q2048_geometry() {
  const auto storage =
      spark_transport::make_tp4_tiled_session_storage_options(2048, 4096);
  assert(spark_transport::tp4_session_storage_options_valid(storage));
  assert(storage.elements_per_row == 4096);
  assert(storage.bytes_per_row == 8192);

  const auto operation =
      spark_transport::make_tp4_tiled_bf16_allreduce_operation(2048, 8192);
  assert(spark_transport::tp4_tiled_operation_descriptor_valid(operation));
  assert(operation.bytes_per_row == 8192);
  assert(operation.active_bytes == 16ULL * 1024ULL * 1024ULL);
  assert(operation.tile_count == 32);

  auto mismatched = operation;
  mismatched.bytes_per_row = 12288;
  assert(!spark_transport::tp4_tiled_operation_descriptor_valid(mismatched));
}

void test_operation_rejects_tile_count_narrowing() {
  bool rejected{};
  try {
    constexpr std::size_t bytes_per_row =
        (static_cast<std::size_t>(
             std::numeric_limits<std::uint32_t>::max()) +
         1U) *
        spark_transport::kTp4TiledDefaultTilePayloadBytes;
    (void)spark_transport::make_tp4_tiled_bf16_allreduce_operation(
        1, bytes_per_row);
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  assert(rejected);
}

void test_ticket_round_trip_and_generation_bounds() {
  using spark_transport::tp4_tiled_ticket_from_ordinal;
  using spark_transport::tp4_tiled_ticket_ordinal;

  for (std::uint64_t ordinal = 0; ordinal < 257; ++ordinal) {
    const auto ticket = tp4_tiled_ticket_from_ordinal(ordinal, 8);
    assert(ticket.generation == ordinal / 8 + 1);
    assert(ticket.slot == ordinal % 8);
    assert(tp4_tiled_ticket_ordinal(ticket, 8) == ordinal);
  }

  bool rejected_zero_generation{};
  try {
    (void)tp4_tiled_ticket_ordinal({0, 0}, 8);
  } catch (const std::invalid_argument&) {
    rejected_zero_generation = true;
  }
  assert(rejected_zero_generation);

  bool rejected_generation_exhaustion{};
  try {
    (void)tp4_tiled_ticket_from_ordinal(
        std::numeric_limits<std::uint64_t>::max(), 1);
  } catch (const std::overflow_error&) {
    rejected_generation_exhaustion = true;
  }
  assert(rejected_generation_exhaustion);
}

void test_every_supported_query_width_has_exact_reversible_tiles() {
  using spark_transport::make_tp4_tiled_bf16_allreduce_operation;
  using spark_transport::make_tp4_tiled_tile_descriptor;
  using spark_transport::tp4_tiled_ticket_ordinal;

  for (std::uint32_t query_rows = 1; query_rows <= 8192;
       ++query_rows) {
    const auto operation =
        make_tp4_tiled_bf16_allreduce_operation(query_rows);
    const std::uint64_t first_ordinal = query_rows % 17;
    std::uint64_t described_bytes{};
    for (std::uint32_t tile_index = 0;
         tile_index < operation.tile_count; ++tile_index) {
      const auto tile = make_tp4_tiled_tile_descriptor(
          operation, tile_index, first_ordinal);
      assert(tp4_tiled_ticket_ordinal(tile.ticket, 8) ==
             first_ordinal + tile_index);
      described_bytes += tile.active_bytes;
    }
    assert(described_bytes == operation.active_bytes);
  }

  const auto maximum = make_tp4_tiled_bf16_allreduce_operation(8192);
  assert(maximum.active_bytes == 96ULL * 1024ULL * 1024ULL);
  assert(maximum.tile_count == 192);
}

void test_cumulative_edge_credits_release_reuse_independently() {
  using spark_transport::Tp4TiledAcquireState;
  using spark_transport::Tp4TiledCreditUpdateState;
  using spark_transport::Tp4TiledCreditWindow;
  using spark_transport::tp4_tiled_ticket_from_ordinal;

  Tp4TiledCreditWindow credits(8);
  for (std::uint64_t ordinal = 0; ordinal < 8; ++ordinal) {
    assert(credits.try_acquire(
               tp4_tiled_ticket_from_ordinal(ordinal, 8)) ==
           Tp4TiledAcquireState::kReady);
  }
  assert(credits.next_ordinal() == 8);
  assert(credits.try_acquire(tp4_tiled_ticket_from_ordinal(8, 8)) ==
         Tp4TiledAcquireState::kWaitingForCredit);
  assert(!credits.poisoned());

  assert(credits.observe_consumed_through(0, 0) ==
         Tp4TiledCreditUpdateState::kAdvanced);
  assert(credits.try_acquire(tp4_tiled_ticket_from_ordinal(8, 8)) ==
         Tp4TiledAcquireState::kWaitingForCredit);
  assert(credits.observe_consumed_through(1, 0) ==
         Tp4TiledCreditUpdateState::kAdvanced);
  assert(credits.try_acquire(tp4_tiled_ticket_from_ordinal(8, 8)) ==
         Tp4TiledAcquireState::kReady);

  // A cumulative jump retires every earlier ordinal on that edge.
  assert(credits.observe_consumed_through(0, 7) ==
         Tp4TiledCreditUpdateState::kAdvanced);
  assert(credits.observe_consumed_through(1, 7) ==
         Tp4TiledCreditUpdateState::kAdvanced);
  for (std::uint64_t ordinal = 9; ordinal < 16; ++ordinal) {
    assert(credits.try_acquire(
               tp4_tiled_ticket_from_ordinal(ordinal, 8)) ==
           Tp4TiledAcquireState::kReady);
  }
  assert(credits.try_acquire(tp4_tiled_ticket_from_ordinal(16, 8)) ==
         Tp4TiledAcquireState::kWaitingForCredit);
}

void test_unexpected_tickets_and_impossible_credits_poison_the_window() {
  using spark_transport::Tp4TiledAcquireState;
  using spark_transport::Tp4TiledCreditUpdateState;
  using spark_transport::Tp4TiledCreditWindow;
  using spark_transport::tp4_tiled_ticket_from_ordinal;

  Tp4TiledCreditWindow unexpected_ticket(8);
  assert(unexpected_ticket.try_acquire(
             tp4_tiled_ticket_from_ordinal(8, 8)) ==
         Tp4TiledAcquireState::kUnexpectedTicket);
  assert(unexpected_ticket.poisoned());
  assert(unexpected_ticket.try_acquire(
             tp4_tiled_ticket_from_ordinal(0, 8)) ==
         Tp4TiledAcquireState::kPoisoned);

  Tp4TiledCreditWindow future_credit(8);
  assert(future_credit.try_acquire(
             tp4_tiled_ticket_from_ordinal(0, 8)) ==
         Tp4TiledAcquireState::kReady);
  assert(future_credit.observe_consumed_through(0, 1) ==
         Tp4TiledCreditUpdateState::kPoisoned);
  assert(future_credit.poisoned());

  Tp4TiledCreditWindow regressed_credit(8);
  for (std::uint64_t ordinal = 0; ordinal < 4; ++ordinal) {
    assert(regressed_credit.try_acquire(
               tp4_tiled_ticket_from_ordinal(ordinal, 8)) ==
           Tp4TiledAcquireState::kReady);
  }
  assert(regressed_credit.observe_consumed_through(0, 2) ==
         Tp4TiledCreditUpdateState::kAdvanced);
  assert(regressed_credit.observe_consumed_through(0, 2) ==
         Tp4TiledCreditUpdateState::kDuplicate);
  assert(regressed_credit.observe_consumed_through(0, 1) ==
         Tp4TiledCreditUpdateState::kPoisoned);
  assert(regressed_credit.poisoned());
}

}  // namespace

int main() {
  test_capacity_selector_and_inert_default();
  test_pool_layout_has_disjoint_send_receive_and_control_regions();
  test_operation_descriptors_cover_only_active_bytes();
  test_glm53_width4096_q2048_geometry();
  test_operation_rejects_tile_count_narrowing();
  test_ticket_round_trip_and_generation_bounds();
  test_every_supported_query_width_has_exact_reversible_tiles();
  test_cumulative_edge_credits_release_reuse_independently();
  test_unexpected_tickets_and_impossible_credits_poison_the_window();
}
