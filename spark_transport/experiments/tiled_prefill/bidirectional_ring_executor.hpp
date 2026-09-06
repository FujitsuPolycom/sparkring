#pragma once

#include "spark_transport/tp4_bidirectional_prefill.hpp"
#include "spark_transport/tp4_tiled_session.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace spark_transport::tiled_prefill_research {

// Research-only adaptive-geometry CPU state machine. It owns no CUDA or verbs
// resources and is deliberately absent from the production library and C ABI.

enum class RingSubmitState : std::uint8_t {
  kAccepted,
  kBackpressured,
  kFatal,
};

enum class RingPollState : std::uint8_t {
  kPending,
  kComplete,
  kFatal,
};

enum class RingCreditPollState : std::uint8_t {
  kNoUpdate,
  kUpdate,
  kFatal,
};

enum class RingBulkAction : std::uint8_t {
  kStageInitial,
  kReduceForward,
  kReduceFinalizeAndSeedGather,
  kGatherForward,
  kGatherFinish,
};

enum class BidirectionalRingFailure : std::uint8_t {
  kNone,
  kTicketProtocol,
  kBulkSubmit,
  kBulkPoll,
  kExchangeSubmit,
  kExchangePoll,
  kCreditProtocol,
  kCreditPublish,
  kPortException,
};

enum class BidirectionalRingDrainState : std::uint8_t {
  kIdle,
  kPending,
  kComplete,
  kPoisoned,
};

constexpr std::uint32_t kBidirectionalRingTilesPerShard = 4;
constexpr std::uint32_t kBidirectionalRingTransfersPerDirection =
    kTp4PrefillStageCount * kBidirectionalRingTilesPerShard;
// Fixed-Q2048 probe tile size retained by the bidirectional-ring ABI.
constexpr std::size_t kBidirectionalRingTileBytes = 512U * 1024U;
// Eight slots keep one complete four-tile source generation alive while the
// following stage is produced. Four slots create a circular wait: stage N+1
// cannot consume stage N until it acquires the same physical slots, while the
// peer cannot return stage-N credit until that consumption occurs.
constexpr std::uint32_t kBidirectionalRingSlotsPerDirection = 8;
constexpr std::uint32_t kBidirectionalRingElementsPerRow = 4096;
constexpr std::size_t kBidirectionalRingBytesPerRow =
    kBidirectionalRingElementsPerRow * 2U;

struct BidirectionalRingGeometry {
  std::uint32_t query_rows{};
  std::uint32_t elements_per_row{};
  std::uint64_t payload_bytes{};
  std::uint64_t half_bytes{};
  std::uint64_t shard_bytes{};
  std::uint32_t tile_bytes{};
  std::uint64_t bytes_per_direction{};
  std::uint64_t bytes_per_rank{};
};

constexpr bool bidirectional_ring_query_rows_supported(
    std::uint32_t query_rows) noexcept {
  return query_rows == 1024 || query_rows == 2048 || query_rows == 4096 ||
         query_rows == 8192;
}

constexpr BidirectionalRingGeometry make_bidirectional_ring_geometry(
    std::uint32_t query_rows,
    std::uint32_t elements_per_row = kBidirectionalRingElementsPerRow) {
  if (!bidirectional_ring_query_rows_supported(query_rows) ||
      elements_per_row != kBidirectionalRingElementsPerRow) {
    throw std::invalid_argument(
        "bidirectional ring requires Q1024/Q2048/Q4096/Q8192 BF16 width4096");
  }
  const std::uint64_t payload_bytes =
      static_cast<std::uint64_t>(query_rows) * kBidirectionalRingBytesPerRow;
  const std::uint64_t half_bytes = payload_bytes / 2U;
  const std::uint64_t shard_bytes = payload_bytes / 8U;
  const std::uint64_t tile_bytes =
      shard_bytes / kBidirectionalRingTilesPerShard;
  return {query_rows,
          elements_per_row,
          payload_bytes,
          half_bytes,
          shard_bytes,
          static_cast<std::uint32_t>(tile_bytes),
          kTp4PrefillStageCount * shard_bytes,
          2U * kTp4PrefillStageCount * shard_bytes};
}

constexpr bool bidirectional_ring_geometry_valid(
    const BidirectionalRingGeometry& geometry) noexcept {
  if (!bidirectional_ring_query_rows_supported(geometry.query_rows) ||
      geometry.elements_per_row != kBidirectionalRingElementsPerRow) {
    return false;
  }
  const std::uint64_t expected_payload =
      static_cast<std::uint64_t>(geometry.query_rows) *
      kBidirectionalRingBytesPerRow;
  return geometry.payload_bytes == expected_payload &&
         geometry.half_bytes == expected_payload / 2U &&
         geometry.shard_bytes == expected_payload / 8U &&
         geometry.tile_bytes ==
             expected_payload /
                 (8U * kBidirectionalRingTilesPerShard) &&
         geometry.bytes_per_direction ==
             kTp4PrefillStageCount * geometry.shard_bytes &&
         geometry.bytes_per_rank ==
             2U * geometry.bytes_per_direction;
}

static_assert(make_bidirectional_ring_geometry(1024).tile_bytes ==
              256U * 1024U);
static_assert(make_bidirectional_ring_geometry(2048).tile_bytes ==
              512U * 1024U);
static_assert(make_bidirectional_ring_geometry(4096).tile_bytes ==
              1024U * 1024U);
static_assert(make_bidirectional_ring_geometry(8192).tile_bytes ==
              2U * 1024U * 1024U);

struct BidirectionalRingBulkRequest {
  RingBulkAction action{};
  Tp4PrefillDirection direction{};
  Tp4PrefillHalf half{};
  // Zero names initial staging. Values 1..6 name the exchange that this
  // bulk operation prepares; post-exchange CUDA descriptors instead name
  // the just-consumed stage and must use bidirectional_ring_consumed_stage.
  std::uint32_t next_exchange_stage{};
  std::uint32_t tile_in_shard{};
  std::uint32_t shard{};
  Tp4TileTicket source_ticket{};
  Tp4TileTicket destination_ticket{};
  Tp4PrefillEndpoint incoming_endpoint{};
  Tp4PrefillEndpoint outgoing_endpoint{};
  std::uint64_t operation_offset_bytes{};
  // Post-exchange actions acquire this exact mapped incoming doorbell before
  // reading source_ticket's receive region. Initial staging uses zero/zero.
  std::uint64_t incoming_doorbell_offset{};
  std::uint64_t consumed_doorbell_token{};
  // The dual-rail probe uses a distinct ordered doorbell word for rail 1.
  // A single-rail adapter passes a zero token to disable the second GPU gate.
  std::uint64_t secondary_incoming_doorbell_offset{};
  std::uint64_t secondary_consumed_doorbell_token{};
  std::uint32_t active_bytes{};
};

struct BidirectionalRingPayloadSpan {
  std::uint64_t local_send_offset{};
  std::uint64_t remote_receive_offset{};
  std::uint32_t active_bytes{};
};

struct BidirectionalRingExchangeRequest {
  Tp4PrefillDirection direction{};
  std::uint32_t stage{};
  std::uint32_t tile_in_shard{};
  std::uint32_t shard{};
  Tp4TileTicket ticket{};
  Tp4PrefillEndpoint outgoing_endpoint{};
  Tp4PrefillEndpoint incoming_endpoint{};
  std::uint32_t outgoing_peer{};
  std::uint32_t incoming_peer{};
  BidirectionalRingPayloadSpan span{};
  std::uint64_t doorbell_token{};
  std::uint64_t work_id{};
};

struct BidirectionalRingCreditRequest {
  Tp4PrefillDirection direction{};
  Tp4PrefillEndpoint endpoint{};
  std::uint64_t consumed_through{};
  std::uint64_t wire_credit{};
  std::uint64_t work_id{};
};

class BidirectionalRingBulkPort {
 public:
  virtual ~BidirectionalRingBulkPort() = default;

  // A CUDA adapter implements these two nonblocking calls on the caller's
  // stable stream. Completion is the payload-before-doorbell handoff.
  virtual RingSubmitState try_submit(
      const BidirectionalRingBulkRequest& request) = 0;
  virtual RingPollState poll(
      const BidirectionalRingBulkRequest& request) = 0;
};

class BidirectionalRingEdgePort {
 public:
  virtual ~BidirectionalRingEdgePort() = default;

  // Completion means the ordered geometry-sized payload write has a local CQ
  // proof
  // and the matching inbound doorbell has been observed.
  virtual RingSubmitState try_post_exchange(
      const BidirectionalRingExchangeRequest& request) = 0;
  virtual RingPollState poll_exchange(
      const BidirectionalRingExchangeRequest& request) = 0;
  virtual RingSubmitState try_publish_consumed_through(
      const BidirectionalRingCreditRequest& request) = 0;
  virtual RingPollState poll_published_consumed_through(
      const BidirectionalRingCreditRequest& request) = 0;
  virtual RingCreditPollState poll_peer_consumed_through(
      Tp4PrefillDirection direction, std::uint64_t& wire_credit) = 0;
};

struct BidirectionalRingAdvanceResult {
  bool made_progress{};
  bool backpressured{};
  bool poisoned{};
};

struct BidirectionalRingStatus {
  bool active_operation{};
  bool output_ready{};
  bool fully_retired{};
  bool safe_to_release_registered_storage{};
  bool poisoned{};
  BidirectionalRingGeometry geometry{};
  std::uint32_t output_ready_tiles{};
  std::array<std::uint32_t, 2> posted_transfers{};
  std::array<std::uint32_t, 2> completed_transfers{};
  std::array<std::uint64_t, 2> transmitted_bytes{};
  std::array<std::uint64_t, 2> peer_consumed_through{};
  std::array<bool, 2> peer_credit_observed{};
  std::array<std::uint64_t, 2> local_consumed_through{};
  std::array<bool, 2> local_credit_observed{};
  BidirectionalRingFailure failure{BidirectionalRingFailure::kNone};
};

constexpr std::uint64_t bidirectional_ring_ordinal(
    std::uint32_t stage, std::uint32_t tile_in_shard) {
  if (stage >= kTp4PrefillStageCount ||
      tile_in_shard >= kBidirectionalRingTilesPerShard) {
    throw std::out_of_range("bidirectional ring stage/tile is invalid");
  }
  return static_cast<std::uint64_t>(stage) *
             kBidirectionalRingTilesPerShard +
         tile_in_shard;
}

constexpr Tp4TileTicket bidirectional_ring_ticket(
    std::uint32_t stage, std::uint32_t tile_in_shard) {
  return tp4_tiled_ticket_from_ordinal(
      bidirectional_ring_ordinal(stage, tile_in_shard),
      kBidirectionalRingSlotsPerDirection);
}

constexpr std::uint32_t bidirectional_ring_consumed_stage(
    std::uint32_t next_exchange_stage) {
  if (next_exchange_stage == 0 ||
      next_exchange_stage > kTp4PrefillStageCount) {
    throw std::out_of_range(
        "bidirectional ring next exchange stage must be in [1, 6]");
  }
  return next_exchange_stage - 1U;
}

class BidirectionalRingExecutor {
 public:
  BidirectionalRingExecutor(const BidirectionalRingExecutor&) = delete;
  BidirectionalRingExecutor& operator=(const BidirectionalRingExecutor&) =
      delete;

  BidirectionalRingExecutor(std::uint32_t rank,
                            BidirectionalRingBulkPort& bulk_port,
                            BidirectionalRingEdgePort& edge_port);
  BidirectionalRingExecutor(std::uint32_t rank,
                            BidirectionalRingGeometry geometry,
                            BidirectionalRingBulkPort& bulk_port,
                            BidirectionalRingEdgePort& edge_port);
  ~BidirectionalRingExecutor();

  void begin();
  BidirectionalRingAdvanceResult advance() noexcept;
  BidirectionalRingDrainState drain() noexcept;
  BidirectionalRingStatus status() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport::tiled_prefill_research
