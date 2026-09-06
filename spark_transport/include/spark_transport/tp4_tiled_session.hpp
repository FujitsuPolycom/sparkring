#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "spark_transport/gpu_doorbell.hpp"

namespace spark_transport {

// Status: offline-validated, research-only. These types define a bounded
// native integration seam but no production session, CUDA kernel, verbs
// endpoint, public C function, or vLLM adapter consumes them.

constexpr std::uint32_t kTp4TiledLatencyMaximumQ = 40;
constexpr std::uint32_t kTp4TiledMediumMaximumQ = 512;
constexpr std::uint32_t kTp4TiledLargeMaximumQ = 1024;
constexpr std::uint32_t kTp4TiledStreamingMaximumQ = 4096;
constexpr std::uint32_t kTp4TiledExtendedMaximumQ = 8192;
constexpr std::uint32_t kTp4TiledBf16ElementsPerQueryRow = 6144;
constexpr std::size_t kTp4TiledBytesPerBf16 = 2;
constexpr std::size_t kTp4TiledBytesPerQueryRow =
    kTp4TiledBf16ElementsPerQueryRow * kTp4TiledBytesPerBf16;
constexpr std::size_t kTp4TiledDefaultTilePayloadBytes = 512U * 1024U;
constexpr std::uint32_t kTp4TiledDefaultSlotsPerEdge = 8;
constexpr std::uint32_t kTp4TiledDefaultLanesPerEdge = 2;
constexpr std::size_t kTp4TiledEdgeCount = 2;
constexpr std::size_t kTp4TiledControlAlignment = alignof(DoorbellControl);
constexpr std::size_t kTp4TiledControlBytes = sizeof(DoorbellControl);

static_assert(kTp4TiledControlAlignment == 64);
static_assert(kTp4TiledControlBytes == 64);

// This source-level selector is not consumed by Tp4AllreduceSession or the C
// ABI. Keeping the default exact-payload state in a separate options object
// lets a later explicit factory opt into tiled storage without changing the
// layout or behavior of the existing session options.
enum class Tp4SessionStorageSelector : std::uint8_t {
  kExactPayload = 0,
  kCapacityTieredTiles = 1,
};

enum class Tp4TiledCapacityClass : std::uint8_t {
  kUnspecified = 0,
  kLatencyQ40 = 1,
  kMediumQ512 = 2,
  kLargeQ1024 = 3,
  kStreamingQ4096 = 4,
  kExtendedQ8192 = 5,
};

struct Tp4SessionStorageOptions {
  Tp4SessionStorageSelector selector{
      Tp4SessionStorageSelector::kExactPayload};
  Tp4TiledCapacityClass capacity_class{
      Tp4TiledCapacityClass::kUnspecified};
  std::uint32_t maximum_query_rows{};
  std::size_t tile_payload_bytes{};
  std::uint32_t slots_per_edge{};
  std::uint32_t lanes_per_edge{};
  std::uint32_t elements_per_row{};
  std::size_t bytes_per_row{};
};

constexpr bool operator==(const Tp4SessionStorageOptions& left,
                          const Tp4SessionStorageOptions& right) noexcept {
  return left.selector == right.selector &&
         left.capacity_class == right.capacity_class &&
         left.maximum_query_rows == right.maximum_query_rows &&
         left.tile_payload_bytes == right.tile_payload_bytes &&
         left.slots_per_edge == right.slots_per_edge &&
         left.lanes_per_edge == right.lanes_per_edge &&
         left.elements_per_row == right.elements_per_row &&
         left.bytes_per_row == right.bytes_per_row;
}

constexpr bool operator!=(const Tp4SessionStorageOptions& left,
                          const Tp4SessionStorageOptions& right) noexcept {
  return !(left == right);
}

constexpr std::uint32_t tp4_tiled_capacity_maximum_query_rows(
    Tp4TiledCapacityClass capacity_class) noexcept {
  switch (capacity_class) {
    case Tp4TiledCapacityClass::kLatencyQ40:
      return kTp4TiledLatencyMaximumQ;
    case Tp4TiledCapacityClass::kMediumQ512:
      return kTp4TiledMediumMaximumQ;
    case Tp4TiledCapacityClass::kLargeQ1024:
      return kTp4TiledLargeMaximumQ;
    case Tp4TiledCapacityClass::kStreamingQ4096:
      return kTp4TiledStreamingMaximumQ;
    case Tp4TiledCapacityClass::kExtendedQ8192:
      return kTp4TiledExtendedMaximumQ;
    case Tp4TiledCapacityClass::kUnspecified:
      return 0;
  }
  return 0;
}

constexpr Tp4TiledCapacityClass tp4_select_tiled_capacity_class(
    std::uint32_t query_rows) {
  if (query_rows == 0 || query_rows > kTp4TiledExtendedMaximumQ) {
    throw std::out_of_range("tiled TP4 query rows must be in [1, 8192]");
  }
  if (query_rows <= kTp4TiledLatencyMaximumQ) {
    return Tp4TiledCapacityClass::kLatencyQ40;
  }
  if (query_rows <= kTp4TiledMediumMaximumQ) {
    return Tp4TiledCapacityClass::kMediumQ512;
  }
  if (query_rows <= kTp4TiledLargeMaximumQ) {
    return Tp4TiledCapacityClass::kLargeQ1024;
  }
  if (query_rows <= kTp4TiledStreamingMaximumQ) {
    return Tp4TiledCapacityClass::kStreamingQ4096;
  }
  if (query_rows <= kTp4TiledExtendedMaximumQ) {
    return Tp4TiledCapacityClass::kExtendedQ8192;
  }
  throw std::out_of_range("tiled TP4 capacity selector is inconsistent");
}

constexpr Tp4SessionStorageOptions
make_tp4_tiled_session_storage_options(
    std::uint32_t query_rows,
    std::uint32_t elements_per_row =
        kTp4TiledBf16ElementsPerQueryRow) {
  if (elements_per_row == 0 ||
      elements_per_row >
          std::numeric_limits<std::size_t>::max() /
              kTp4TiledBytesPerBf16) {
    throw std::invalid_argument(
        "tiled TP4 row geometry must contain BF16 elements");
  }
  const auto capacity_class =
      tp4_select_tiled_capacity_class(query_rows);
  return {Tp4SessionStorageSelector::kCapacityTieredTiles,
          capacity_class,
          tp4_tiled_capacity_maximum_query_rows(capacity_class),
          kTp4TiledDefaultTilePayloadBytes,
          kTp4TiledDefaultSlotsPerEdge,
          kTp4TiledDefaultLanesPerEdge,
          elements_per_row,
          static_cast<std::size_t>(elements_per_row) *
              kTp4TiledBytesPerBf16};
}

constexpr bool tp4_session_storage_options_valid(
    const Tp4SessionStorageOptions& options) noexcept {
  if (options.selector == Tp4SessionStorageSelector::kExactPayload) {
    return options.capacity_class ==
               Tp4TiledCapacityClass::kUnspecified &&
           options.maximum_query_rows == 0 &&
           options.tile_payload_bytes == 0 &&
           options.slots_per_edge == 0 && options.lanes_per_edge == 0 &&
           options.elements_per_row == 0 && options.bytes_per_row == 0;
  }
  if (options.selector !=
      Tp4SessionStorageSelector::kCapacityTieredTiles) {
    return false;
  }
  const auto maximum = tp4_tiled_capacity_maximum_query_rows(
      options.capacity_class);
  return maximum != 0 && options.maximum_query_rows == maximum &&
         options.tile_payload_bytes ==
             kTp4TiledDefaultTilePayloadBytes &&
         options.slots_per_edge == kTp4TiledDefaultSlotsPerEdge &&
         options.lanes_per_edge == kTp4TiledDefaultLanesPerEdge &&
         options.elements_per_row != 0 &&
         options.bytes_per_row ==
             static_cast<std::size_t>(options.elements_per_row) *
                 kTp4TiledBytesPerBf16;
}

struct Tp4TiledPoolLayout {
  std::size_t tile_payload_bytes{};
  std::uint32_t slots_per_edge{};
  std::uint32_t lanes_per_edge{};
  std::size_t lane_payload_bytes{};
  std::size_t lane_receive_offset{};
  std::size_t lane_control_offset{};
  std::size_t lane_stride{};
  std::size_t slot_stride{};
  std::size_t total_bytes{};
};

struct Tp4TiledSlotRegion {
  std::size_t send_offset{};
  std::size_t receive_offset{};
  std::size_t control_offset{};
  std::size_t end_offset{};
};

namespace detail {

constexpr std::size_t tp4_tiled_checked_add(std::size_t left,
                                            std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error("tiled TP4 layout size overflow");
  }
  return left + right;
}

constexpr std::size_t tp4_tiled_checked_multiply(std::size_t left,
                                                 std::size_t right) {
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("tiled TP4 layout size overflow");
  }
  return left * right;
}

constexpr std::uint64_t tp4_tiled_checked_add_u64(
    std::uint64_t left, std::uint64_t right) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error("tiled TP4 tile ordinal overflow");
  }
  return left + right;
}

constexpr std::uint64_t tp4_tiled_checked_multiply_u64(
    std::uint64_t left, std::uint64_t right) {
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error("tiled TP4 tile ordinal overflow");
  }
  return left * right;
}

}  // namespace detail

constexpr std::size_t tp4_tiled_align_control(std::size_t bytes) {
  const std::size_t remainder = bytes % kTp4TiledControlAlignment;
  const std::size_t padding =
      remainder == 0 ? 0 : kTp4TiledControlAlignment - remainder;
  return detail::tp4_tiled_checked_add(bytes, padding);
}

constexpr Tp4TiledPoolLayout make_tp4_tiled_pool_layout(
    std::size_t tile_payload_bytes, std::uint32_t slots_per_edge,
    std::uint32_t lanes_per_edge) {
  if (tile_payload_bytes == 0 ||
      tile_payload_bytes % kTp4TiledBytesPerBf16 != 0) {
    throw std::invalid_argument(
        "tiled TP4 payload must contain whole BF16 elements");
  }
  if (slots_per_edge == 0) {
    throw std::invalid_argument(
        "tiled TP4 pool requires at least one slot per edge");
  }
  if (lanes_per_edge == 0 ||
      tile_payload_bytes % lanes_per_edge != 0) {
    throw std::invalid_argument(
        "tiled TP4 payload must split exactly across nonempty lanes");
  }
  const std::size_t lane_payload_bytes =
      tile_payload_bytes / lanes_per_edge;
  if (lane_payload_bytes % kTp4TiledBytesPerBf16 != 0) {
    throw std::invalid_argument(
        "tiled TP4 lane must contain whole BF16 elements");
  }
  const std::size_t receive_offset =
      tp4_tiled_align_control(lane_payload_bytes);
  const std::size_t control_offset = tp4_tiled_align_control(
      detail::tp4_tiled_checked_add(receive_offset,
                                    lane_payload_bytes));
  const std::size_t lane_stride = tp4_tiled_align_control(
      detail::tp4_tiled_checked_add(control_offset,
                                    kTp4TiledControlBytes));
  const std::size_t slot_stride =
      detail::tp4_tiled_checked_multiply(lane_stride,
                                         lanes_per_edge);
  return {tile_payload_bytes,
          slots_per_edge,
          lanes_per_edge,
          lane_payload_bytes,
          receive_offset,
          control_offset,
          lane_stride,
          slot_stride,
          detail::tp4_tiled_checked_multiply(slot_stride,
                                             slots_per_edge)};
}

constexpr Tp4TiledPoolLayout make_tp4_tiled_pool_layout(
    const Tp4SessionStorageOptions& options) {
  if (options.selector !=
          Tp4SessionStorageSelector::kCapacityTieredTiles ||
      !tp4_session_storage_options_valid(options)) {
    throw std::invalid_argument(
        "tiled TP4 pool requires valid capacity-tiered storage options");
  }
  return make_tp4_tiled_pool_layout(options.tile_payload_bytes,
                                    options.slots_per_edge,
                                    options.lanes_per_edge);
}

constexpr Tp4TiledSlotRegion tp4_tiled_slot_region(
    const Tp4TiledPoolLayout& layout, std::uint32_t slot,
    std::uint32_t lane) {
  if (slot >= layout.slots_per_edge) {
    throw std::out_of_range("tiled TP4 slot is outside the pool");
  }
  if (lane >= layout.lanes_per_edge) {
    throw std::out_of_range("tiled TP4 lane is outside the slot");
  }
  const std::size_t slot_offset =
      detail::tp4_tiled_checked_multiply(layout.slot_stride, slot);
  const std::size_t lane_offset =
      detail::tp4_tiled_checked_multiply(layout.lane_stride, lane);
  const std::size_t send_offset =
      detail::tp4_tiled_checked_add(slot_offset, lane_offset);
  return {send_offset,
          detail::tp4_tiled_checked_add(send_offset,
                                        layout.lane_receive_offset),
          detail::tp4_tiled_checked_add(send_offset,
                                        layout.lane_control_offset),
          detail::tp4_tiled_checked_add(send_offset,
                                        layout.lane_stride)};
}

struct Tp4TileTicket {
  std::uint64_t generation{};
  std::uint32_t slot{};
};

constexpr bool operator==(const Tp4TileTicket& left,
                          const Tp4TileTicket& right) noexcept {
  return left.generation == right.generation && left.slot == right.slot;
}

constexpr bool operator!=(const Tp4TileTicket& left,
                          const Tp4TileTicket& right) noexcept {
  return !(left == right);
}

constexpr Tp4TileTicket tp4_tiled_ticket_from_ordinal(
    std::uint64_t ordinal, std::uint32_t slots_per_edge) {
  if (slots_per_edge == 0) {
    throw std::invalid_argument(
        "tiled TP4 ticket requires at least one slot per edge");
  }
  const std::uint64_t generation_index = ordinal / slots_per_edge;
  if (generation_index ==
      std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("tiled TP4 ticket generation exhausted");
  }
  return {generation_index + 1,
          static_cast<std::uint32_t>(ordinal % slots_per_edge)};
}

constexpr std::uint64_t tp4_tiled_ticket_ordinal(
    Tp4TileTicket ticket, std::uint32_t slots_per_edge) {
  if (slots_per_edge == 0 || ticket.generation == 0 ||
      ticket.slot >= slots_per_edge) {
    throw std::invalid_argument("invalid tiled TP4 ticket");
  }
  const auto generation_offset =
      detail::tp4_tiled_checked_multiply_u64(
          ticket.generation - 1, slots_per_edge);
  return detail::tp4_tiled_checked_add_u64(generation_offset,
                                           ticket.slot);
}

struct Tp4TiledOperationDescriptor {
  Tp4TiledCapacityClass capacity_class{
      Tp4TiledCapacityClass::kUnspecified};
  std::uint32_t query_rows{};
  std::uint32_t tile_count{};
  std::size_t bytes_per_row{};
  std::uint64_t active_bytes{};
};

struct Tp4TiledTileDescriptor {
  Tp4TileTicket ticket{};
  std::uint64_t operation_offset_bytes{};
  std::uint32_t active_bytes{};
};

constexpr Tp4TiledOperationDescriptor
make_tp4_tiled_bf16_allreduce_operation(
    std::uint32_t query_rows,
    std::size_t bytes_per_row = kTp4TiledBytesPerQueryRow) {
  if (bytes_per_row == 0 ||
      bytes_per_row % kTp4TiledBytesPerBf16 != 0) {
    throw std::invalid_argument(
        "tiled TP4 row geometry must contain whole BF16 elements");
  }
  const auto capacity_class =
      tp4_select_tiled_capacity_class(query_rows);
  if (query_rows >
      std::numeric_limits<std::uint64_t>::max() / bytes_per_row) {
    throw std::overflow_error("tiled TP4 active byte count overflow");
  }
  const std::uint64_t active_bytes =
      static_cast<std::uint64_t>(query_rows) * bytes_per_row;
  const std::uint64_t tile_count =
      (active_bytes + kTp4TiledDefaultTilePayloadBytes - 1U) /
      kTp4TiledDefaultTilePayloadBytes;
  if (tile_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("tiled TP4 tile count exceeds uint32");
  }
  return {capacity_class,
          query_rows,
          static_cast<std::uint32_t>(tile_count),
          bytes_per_row,
          active_bytes};
}

constexpr bool tp4_tiled_operation_descriptor_valid(
    const Tp4TiledOperationDescriptor& operation) noexcept {
  if (operation.query_rows == 0 ||
      operation.query_rows > kTp4TiledExtendedMaximumQ ||
      operation.bytes_per_row == 0 ||
      operation.bytes_per_row % kTp4TiledBytesPerBf16 != 0) {
    return false;
  }
  const Tp4TiledCapacityClass expected_capacity =
      operation.query_rows <= kTp4TiledLatencyMaximumQ
          ? Tp4TiledCapacityClass::kLatencyQ40
          : operation.query_rows <= kTp4TiledMediumMaximumQ
                ? Tp4TiledCapacityClass::kMediumQ512
                : operation.query_rows <= kTp4TiledLargeMaximumQ
                      ? Tp4TiledCapacityClass::kLargeQ1024
                      : operation.query_rows <=
                                kTp4TiledStreamingMaximumQ
                            ? Tp4TiledCapacityClass::kStreamingQ4096
                            : Tp4TiledCapacityClass::kExtendedQ8192;
  const std::uint64_t expected_active_bytes =
      static_cast<std::uint64_t>(operation.query_rows) *
      operation.bytes_per_row;
  const std::uint64_t expected_tile_count =
      (expected_active_bytes + kTp4TiledDefaultTilePayloadBytes - 1U) /
      kTp4TiledDefaultTilePayloadBytes;
  return operation.capacity_class == expected_capacity &&
         operation.active_bytes == expected_active_bytes &&
         operation.tile_count == expected_tile_count;
}

constexpr Tp4TiledTileDescriptor make_tp4_tiled_tile_descriptor(
    const Tp4TiledOperationDescriptor& operation,
    std::uint32_t tile_index, std::uint64_t first_tile_ordinal = 0) {
  if (!tp4_tiled_operation_descriptor_valid(operation)) {
    throw std::invalid_argument(
        "invalid tiled TP4 operation descriptor");
  }
  if (tile_index >= operation.tile_count) {
    throw std::out_of_range(
        "tiled TP4 tile index is outside the operation");
  }
  const std::uint64_t operation_offset =
      static_cast<std::uint64_t>(tile_index) *
      kTp4TiledDefaultTilePayloadBytes;
  const std::uint64_t remaining =
      operation.active_bytes - operation_offset;
  const std::uint64_t active_bytes =
      remaining < kTp4TiledDefaultTilePayloadBytes
          ? remaining
          : kTp4TiledDefaultTilePayloadBytes;
  const std::uint64_t ordinal = detail::tp4_tiled_checked_add_u64(
      first_tile_ordinal, tile_index);
  return {tp4_tiled_ticket_from_ordinal(
              ordinal, kTp4TiledDefaultSlotsPerEdge),
          operation_offset,
          static_cast<std::uint32_t>(active_bytes)};
}

enum class Tp4TiledAcquireState : std::uint8_t {
  kReady,
  kWaitingForCredit,
  kUnexpectedTicket,
  kPoisoned,
  kExhausted,
};

enum class Tp4TiledCreditUpdateState : std::uint8_t {
  kAdvanced,
  kDuplicate,
  kPoisoned,
};

// Plain host state for one logical tile sequence spanning both TP4 edges.
// Credits are inclusive cumulative tile ordinals. Slot reuse requires both
// edges to have consumed that slot's prior generation. A stale ticket,
// regressing watermark, credit for unissued work, or invalid edge poisons the
// state permanently; production integration must translate that condition
// into the transport's process-fatal protocol path. The object has one host
// owner; concurrent edge engines must synchronize updates before calling it.
class Tp4TiledCreditWindow {
 public:
  explicit Tp4TiledCreditWindow(std::uint32_t slots_per_edge)
      : slots_per_edge_(slots_per_edge) {
    if (slots_per_edge_ == 0) {
      throw std::invalid_argument(
          "tiled TP4 credit window requires at least one slot");
    }
  }

  Tp4TiledAcquireState try_acquire(Tp4TileTicket ticket) noexcept {
    if (poisoned_) {
      return Tp4TiledAcquireState::kPoisoned;
    }
    if (exhausted_) {
      return Tp4TiledAcquireState::kExhausted;
    }

    const std::uint64_t generation_index =
        next_ordinal_ / slots_per_edge_;
    if (generation_index ==
        std::numeric_limits<std::uint64_t>::max()) {
      exhausted_ = true;
      return Tp4TiledAcquireState::kExhausted;
    }
    const Tp4TileTicket expected{
        generation_index + 1,
        static_cast<std::uint32_t>(next_ordinal_ % slots_per_edge_)};
    if (ticket != expected) {
      poisoned_ = true;
      return Tp4TiledAcquireState::kUnexpectedTicket;
    }

    if (next_ordinal_ >= slots_per_edge_) {
      const std::uint64_t required_consumed_ordinal =
          next_ordinal_ - slots_per_edge_;
      for (std::size_t edge = 0; edge < kTp4TiledEdgeCount; ++edge) {
        if (!credit_observed_[edge] ||
            consumed_through_[edge] < required_consumed_ordinal) {
          return Tp4TiledAcquireState::kWaitingForCredit;
        }
      }
    }

    has_issued_ = true;
    highest_issued_ = next_ordinal_;
    if (next_ordinal_ ==
        std::numeric_limits<std::uint64_t>::max()) {
      exhausted_ = true;
    } else {
      ++next_ordinal_;
    }
    return Tp4TiledAcquireState::kReady;
  }

  Tp4TiledCreditUpdateState observe_consumed_through(
      std::uint32_t edge, std::uint64_t consumed_ordinal) noexcept {
    if (poisoned_) {
      return Tp4TiledCreditUpdateState::kPoisoned;
    }
    if (edge >= kTp4TiledEdgeCount || !has_issued_ ||
        consumed_ordinal > highest_issued_) {
      poisoned_ = true;
      return Tp4TiledCreditUpdateState::kPoisoned;
    }
    if (credit_observed_[edge]) {
      if (consumed_ordinal < consumed_through_[edge]) {
        poisoned_ = true;
        return Tp4TiledCreditUpdateState::kPoisoned;
      }
      if (consumed_ordinal == consumed_through_[edge]) {
        return Tp4TiledCreditUpdateState::kDuplicate;
      }
    }
    consumed_through_[edge] = consumed_ordinal;
    credit_observed_[edge] = true;
    return Tp4TiledCreditUpdateState::kAdvanced;
  }

  constexpr std::uint32_t slots_per_edge() const noexcept {
    return slots_per_edge_;
  }

  constexpr std::uint64_t next_ordinal() const noexcept {
    return next_ordinal_;
  }

  constexpr bool has_issued() const noexcept { return has_issued_; }

  constexpr std::uint64_t highest_issued() const noexcept {
    return highest_issued_;
  }

  constexpr bool poisoned() const noexcept { return poisoned_; }

  constexpr bool exhausted() const noexcept { return exhausted_; }

 private:
  std::uint32_t slots_per_edge_{};
  std::uint64_t next_ordinal_{};
  std::uint64_t highest_issued_{};
  std::array<std::uint64_t, kTp4TiledEdgeCount> consumed_through_{};
  std::array<bool, kTp4TiledEdgeCount> credit_observed_{};
  bool has_issued_{};
  bool poisoned_{};
  bool exhausted_{};
};

}  // namespace spark_transport
