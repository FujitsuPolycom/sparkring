#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tiled_executor.hpp"

namespace spark_transport::tiled_prefill_research {

// CPU-testable logical-edge adapter. Each child owns a distinct rail/QP and
// registers the same logical arena layout. Payload offsets therefore remain
// relative to each child's arena while the adapter assigns disjoint byte
// ranges to the two rails.
struct DualRailStripedEdgeStatus {
  bool poisoned{};
  bool safe_to_release_registered_storage{};
  std::size_t pending_exchanges{};
  bool pending_credit_publication{};
  std::uint64_t peer_wire_credit{};
};

enum class DualRailDrainState : std::uint8_t {
  kIdle,
  kPending,
  kComplete,
  kPoisoned,
};

class DualRailStripedEdgePort final : public TiledEdgePort {
 public:
  DualRailStripedEdgePort(const DualRailStripedEdgePort&) = delete;
  DualRailStripedEdgePort& operator=(const DualRailStripedEdgePort&) = delete;

  DualRailStripedEdgePort(std::uint32_t logical_edge,
                          TiledEdgePort& rail0, TiledEdgePort& rail1,
                          std::uint32_t stripe_alignment = 16);
  ~DualRailStripedEdgePort() override;

  std::uint32_t edge_index() const noexcept override;
  std::uintptr_t engine_identity() const noexcept override;
  std::uintptr_t qp_identity() const noexcept override;

  TiledSubmitState try_post_exchange(
      const TiledEdgeExchangeRequest& request) override;
  TiledPollState poll_exchange(
      const TiledEdgeExchangeRequest& request) override;
  TiledSubmitState try_publish_consumed_through(
      const TiledCreditPublishRequest& request) override;
  TiledPollState poll_published_consumed_through(
      const TiledCreditPublishRequest& request) override;
  TiledCreditPollState poll_peer_consumed_through(
      const TiledCreditObserveRequest& request,
      std::uint64_t& wire_credit) override;

  DualRailDrainState drain() noexcept;
  DualRailStripedEdgeStatus status() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport::tiled_prefill_research
