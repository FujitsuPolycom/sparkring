#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "spark_transport/tp4_tiled_session.hpp"
#include "tiled_bulk_abi.hpp"

namespace spark_transport::tiled_prefill_research {

// Status: research-only, offline-validated when built by its CPU test target.
// This interface deliberately does not inherit from VerbsEndpoint, expose a C
// symbol, or select itself from the production vLLM adapter.

enum class TiledSubmitState : std::uint8_t {
  kAccepted,
  kBackpressured,
  kFatal,
};

enum class TiledPollState : std::uint8_t {
  kPending,
  kComplete,
  kFatal,
};

enum class TiledCreditPollState : std::uint8_t {
  kNoUpdate,
  kUpdate,
  kFatal,
};

enum class TiledDrainState : std::uint8_t {
  kIdle,
  kPending,
  kComplete,
  kPoisoned,
};

enum class TiledBulkPhase : std::uint8_t {
  kStagePhase1,
  kReducePhase1,
  kReducePhase2,
};

enum class TiledExchangePhase : std::uint8_t {
  kPhase1,
  kPhase2,
};

enum class TiledWorkEvent : std::uint8_t {
  kPhase1Exchange = 1,
  kPhase2Exchange = 2,
  kCredit = 3,
};

enum class TiledExecutorFailure : std::uint8_t {
  kNone,
  kTicketExhausted,
  kUnexpectedTicket,
  kCreditProtocol,
  kBulkSubmit,
  kBulkPoll,
  kEdgeSubmit,
  kEdgePoll,
  kCreditPublish,
  kPortException,
};

struct TiledExecutorTileDescriptor {
  Tp4TileTicket ticket{};
  std::uint64_t ordinal{};
  TiledBulkDescriptor gpu{};
};

struct TiledBulkLaunchRequest {
  TiledBulkPhase phase{};
  Tp4TileTicket ticket{};
  std::uint64_t ordinal{};
  TiledBulkDescriptor descriptor{};
  EndpointTileStorageLayout layout{};
};

struct TiledEdgeExchangeRequest {
  TiledExchangePhase phase{};
  Tp4TileTicket ticket{};
  std::uint64_t ordinal{};
  std::uint64_t doorbell_token{};
  std::uint64_t work_id{};
  std::uint64_t local_payload_offset{};
  std::uint64_t remote_payload_offset{};
  std::uint64_t local_doorbell_offset{};
  std::uint64_t remote_doorbell_offset{};
  std::uint32_t active_bytes{};
  std::uint32_t edge{};
  std::uint32_t lane{};
};

struct TiledCreditPublishRequest {
  std::uint64_t consumed_through{};
  // Zero is reserved for an untouched mapped word. The adapter stores and
  // transmits this one-based value, never consumed_through directly.
  std::uint64_t wire_credit{};
  std::uint64_t work_id{};
  std::uint64_t local_credit_offset{};
  std::uint64_t remote_credit_offset{};
  std::uint32_t edge{};
};

struct TiledCreditObserveRequest {
  std::uint64_t local_credit_offset{};
  std::uint32_t edge{};
};

class TiledBulkLaunchPort {
 public:
  virtual ~TiledBulkLaunchPort() = default;

  // Implementations return kSingleStreamCorrectnessOnly. Performance evidence
  // requires a separately reviewed multi-stream state machine.
  virtual TiledExecutorStreamPolicy stream_policy() const noexcept = 0;
  virtual TiledSubmitState try_submit(
      const TiledBulkLaunchRequest& request) = 0;
  virtual TiledPollState poll(
      const TiledBulkLaunchRequest& request) = 0;
};

// One instance represents one stable edge QP owned outside the research
// executor. Both tensor lanes and both exchange phases use this same port.
class TiledEdgePort {
 public:
  virtual ~TiledEdgePort() = default;

  virtual std::uint32_t edge_index() const noexcept = 0;
  virtual std::uintptr_t engine_identity() const noexcept = 0;
  virtual std::uintptr_t qp_identity() const noexcept = 0;

  virtual TiledSubmitState try_post_exchange(
      const TiledEdgeExchangeRequest& request) = 0;
  virtual TiledPollState poll_exchange(
      const TiledEdgeExchangeRequest& request) = 0;
  virtual TiledSubmitState try_publish_consumed_through(
      const TiledCreditPublishRequest& request) = 0;
  // Completion is the CQ/fence proof that the remote credit write no longer
  // reads the registered local control record. Accepted submission alone is
  // insufficient for arena or QP teardown.
  virtual TiledPollState poll_published_consumed_through(
      const TiledCreditPublishRequest& request) = 0;
  // Returns the raw one-based wire word. kNoUpdate is preferred when that
  // word is zero; if kUpdate returns zero, the executor ignores it.
  virtual TiledCreditPollState poll_peer_consumed_through(
      const TiledCreditObserveRequest& request,
      std::uint64_t& wire_credit) = 0;
};

struct TiledExecutorAdvanceResult {
  bool made_progress{};
  bool backpressured{};
  bool poisoned{};
};

struct TiledExecutorStatus {
  bool active_operation{};
  bool output_ready{};
  bool outbound_credits_complete{};
  bool fully_retired{};
  bool safe_to_release_registered_storage{};
  bool poisoned{};
  std::uint32_t query_rows{};
  std::uint32_t tile_count{};
  std::uint32_t output_ready_tiles{};
  std::uint32_t retired_tiles{};
  std::uint32_t acquired_tiles{};
  std::uint64_t active_payload_bytes{};
  std::uint64_t first_tile_ordinal{};
  std::uint64_t highest_issued_ordinal{};
  std::uint64_t unexpected_generation_count{};
  std::uint64_t credit_regression_count{};
  std::uint64_t output_ready_before_final_retirement_count{};
  TiledExecutorFailure failure{TiledExecutorFailure::kNone};
};

std::uint64_t tiled_doorbell_token(std::uint64_t ordinal);
std::uint64_t tiled_work_id(std::uint64_t ordinal, TiledWorkEvent event);
std::uint64_t tiled_wire_credit(std::uint64_t consumed_through);
std::uint64_t tiled_consumed_through_from_wire(std::uint64_t wire_credit);

TiledExecutorTileDescriptor make_tiled_executor_tile_descriptor(
    const Tp4TiledTileDescriptor& descriptor,
    const Tp4TiledPoolLayout& layout);

class TiledExecutor {
 public:
  TiledExecutor(const TiledExecutor&) = delete;
  TiledExecutor& operator=(const TiledExecutor&) = delete;

  TiledExecutor(Tp4SessionStorageOptions storage_options,
                TiledBulkLaunchPort& bulk_port,
                TiledEdgePort& edge0, TiledEdgePort& edge1);
  ~TiledExecutor();

  // Begins one operation on the persistent tile/credit sequence. The caller
  // retains input/output storage until output_ready; endpoint arenas and QPs
  // remain live until drain permits release. A completed operation may be
  // replaced only after both peer watermarks retire it.
  void begin(std::uint32_t query_rows);
  TiledExecutorAdvanceResult advance() noexcept;
  // Drives the same nonblocking state machine and is the authoritative
  // teardown gate. Only kIdle or kComplete permits the external owner to
  // destroy QPs or deregister the endpoint arenas.
  TiledDrainState drain() noexcept;
  TiledExecutorStatus status() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport::tiled_prefill_research
