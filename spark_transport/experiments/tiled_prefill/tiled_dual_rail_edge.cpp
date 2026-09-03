#include "tiled_dual_rail_edge.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace spark_transport::tiled_prefill_research {
namespace {

bool same_exchange(const TiledEdgeExchangeRequest& left,
                   const TiledEdgeExchangeRequest& right) noexcept {
  return left.phase == right.phase && left.ticket == right.ticket &&
         left.ordinal == right.ordinal &&
         left.doorbell_token == right.doorbell_token &&
         left.work_id == right.work_id &&
         left.local_payload_offset == right.local_payload_offset &&
         left.remote_payload_offset == right.remote_payload_offset &&
         left.local_doorbell_offset == right.local_doorbell_offset &&
         left.remote_doorbell_offset == right.remote_doorbell_offset &&
         left.active_bytes == right.active_bytes && left.edge == right.edge &&
         left.lane == right.lane;
}

bool same_credit(const TiledCreditPublishRequest& left,
                 const TiledCreditPublishRequest& right) noexcept {
  return left.consumed_through == right.consumed_through &&
         left.wire_credit == right.wire_credit &&
         left.work_id == right.work_id &&
         left.local_credit_offset == right.local_credit_offset &&
         left.remote_credit_offset == right.remote_credit_offset &&
         left.edge == right.edge;
}

std::uint64_t checked_offset(std::uint64_t base, std::uint64_t delta) {
  if (base > std::numeric_limits<std::uint64_t>::max() - delta) {
    throw std::overflow_error("dual-rail stripe offset overflow");
  }
  return base + delta;
}

struct ExchangeState {
  TiledEdgeExchangeRequest logical{};
  std::array<TiledEdgeExchangeRequest, 2> rails{};
  std::array<bool, 2> submitted{};
  std::array<bool, 2> completed{};
};

struct CreditState {
  TiledCreditPublishRequest logical{};
  std::array<bool, 2> submitted{};
  std::array<bool, 2> completed{};
};

}  // namespace

class DualRailStripedEdgePort::Impl {
 public:
  Impl(std::uint32_t logical_edge, TiledEdgePort& rail0,
       TiledEdgePort& rail1, std::uint32_t stripe_alignment)
      : logical_edge_(logical_edge), rails_{&rail0, &rail1},
        stripe_alignment_(stripe_alignment),
        engine_identity_(rail0.engine_identity()) {
    if (logical_edge_ >= kTp4TiledEdgeCount || stripe_alignment_ == 0 ||
        (stripe_alignment_ & (stripe_alignment_ - 1U)) != 0 ||
        rail0.edge_index() != logical_edge_ ||
        rail1.edge_index() != logical_edge_ || engine_identity_ == 0 ||
        rail1.engine_identity() != engine_identity_ ||
        rail0.qp_identity() == 0 || rail1.qp_identity() == 0 ||
        rail0.qp_identity() == rail1.qp_identity()) {
      throw std::invalid_argument(
          "dual-rail edge requires aligned stripes and two distinct child QPs");
    }
  }

  std::uint32_t edge_index() const noexcept { return logical_edge_; }
  std::uintptr_t engine_identity() const noexcept {
    return engine_identity_;
  }
  std::uintptr_t qp_identity() const noexcept {
    return reinterpret_cast<std::uintptr_t>(this);
  }

  TiledSubmitState try_post_exchange(
      const TiledEdgeExchangeRequest& request) {
    if (poisoned_) return TiledSubmitState::kFatal;
    try {
      auto [position, inserted] = exchanges_.try_emplace(request.work_id);
      ExchangeState& state = position->second;
      if (inserted) {
        state.logical = request;
        state.rails = split(request);
        ever_active_ = true;
      } else if (!same_exchange(state.logical, request)) {
        return poison_submit();
      }

      bool backpressured{};
      for (std::size_t rail = 0; rail < rails_.size(); ++rail) {
        if (state.submitted[rail]) continue;
        const auto result = rails_[rail]->try_post_exchange(
            state.rails[rail]);
        if (result == TiledSubmitState::kFatal) return poison_submit();
        if (result == TiledSubmitState::kBackpressured) {
          backpressured = true;
        } else {
          state.submitted[rail] = true;
        }
      }
      return backpressured || !state.submitted[0] || !state.submitted[1]
                 ? TiledSubmitState::kBackpressured
                 : TiledSubmitState::kAccepted;
    } catch (...) {
      return poison_submit();
    }
  }

  TiledPollState poll_exchange(const TiledEdgeExchangeRequest& request) {
    if (poisoned_) return TiledPollState::kFatal;
    const auto position = exchanges_.find(request.work_id);
    if (position == exchanges_.end() ||
        !same_exchange(position->second.logical, request) ||
        !position->second.submitted[0] ||
        !position->second.submitted[1]) {
      return poison_poll();
    }
    ExchangeState& state = position->second;
    for (std::size_t rail = 0; rail < rails_.size(); ++rail) {
      if (state.completed[rail]) continue;
      try {
        const auto result = rails_[rail]->poll_exchange(state.rails[rail]);
        if (result == TiledPollState::kFatal) return poison_poll();
        if (result == TiledPollState::kComplete) state.completed[rail] = true;
      } catch (...) {
        return poison_poll();
      }
    }
    if (!state.completed[0] || !state.completed[1]) {
      return TiledPollState::kPending;
    }
    exchanges_.erase(position);
    return TiledPollState::kComplete;
  }

  TiledSubmitState try_publish_consumed_through(
      const TiledCreditPublishRequest& request) {
    if (poisoned_) return TiledSubmitState::kFatal;
    if (request.edge != logical_edge_) return poison_submit();
    if (!credit_.has_value()) {
      credit_.emplace();
      credit_->logical = request;
      ever_active_ = true;
    } else if (!same_credit(credit_->logical, request)) {
      return poison_submit();
    }
    bool backpressured{};
    for (std::size_t rail = 0; rail < rails_.size(); ++rail) {
      if (credit_->submitted[rail]) continue;
      try {
        const auto result =
            rails_[rail]->try_publish_consumed_through(request);
        if (result == TiledSubmitState::kFatal) return poison_submit();
        if (result == TiledSubmitState::kBackpressured) {
          backpressured = true;
        } else {
          credit_->submitted[rail] = true;
        }
      } catch (...) {
        return poison_submit();
      }
    }
    return backpressured || !credit_->submitted[0] || !credit_->submitted[1]
               ? TiledSubmitState::kBackpressured
               : TiledSubmitState::kAccepted;
  }

  TiledPollState poll_published_consumed_through(
      const TiledCreditPublishRequest& request) {
    if (poisoned_ || !credit_.has_value() ||
        !same_credit(credit_->logical, request) || !credit_->submitted[0] ||
        !credit_->submitted[1]) {
      return poison_poll();
    }
    for (std::size_t rail = 0; rail < rails_.size(); ++rail) {
      if (credit_->completed[rail]) continue;
      try {
        const auto result =
            rails_[rail]->poll_published_consumed_through(request);
        if (result == TiledPollState::kFatal) return poison_poll();
        if (result == TiledPollState::kComplete) credit_->completed[rail] = true;
      } catch (...) {
        return poison_poll();
      }
    }
    if (!credit_->completed[0] || !credit_->completed[1]) {
      return TiledPollState::kPending;
    }
    credit_.reset();
    return TiledPollState::kComplete;
  }

  TiledCreditPollState poll_peer_consumed_through(
      const TiledCreditObserveRequest& request,
      std::uint64_t& wire_credit) {
    if (poisoned_) return TiledCreditPollState::kFatal;
    if (request.edge != logical_edge_) return poison_credit();
    for (std::size_t rail = 0; rail < rails_.size(); ++rail) {
      std::uint64_t observed{};
      try {
        const auto result = rails_[rail]->poll_peer_consumed_through(
            request, observed);
        if (result == TiledCreditPollState::kFatal) return poison_credit();
        if (result != TiledCreditPollState::kUpdate) continue;
      } catch (...) {
        return poison_credit();
      }
      if (observed == 0 ||
          (peer_credit_observed_[rail] &&
           observed < peer_wire_credits_[rail])) {
        return poison_credit();
      }
      peer_credit_observed_[rail] = true;
      peer_wire_credits_[rail] = observed;
    }
    if (!peer_credit_observed_[0] || !peer_credit_observed_[1]) {
      return TiledCreditPollState::kNoUpdate;
    }
    const std::uint64_t safe_credit =
        std::min(peer_wire_credits_[0], peer_wire_credits_[1]);
    if (safe_credit <= emitted_peer_wire_credit_) {
      return TiledCreditPollState::kNoUpdate;
    }
    emitted_peer_wire_credit_ = safe_credit;
    wire_credit = safe_credit;
    return TiledCreditPollState::kUpdate;
  }

  DualRailDrainState drain() noexcept {
    if (poisoned_) return DualRailDrainState::kPoisoned;
    for (auto position = exchanges_.begin(); position != exchanges_.end();) {
      const auto request = position->second.logical;
      if (!position->second.submitted[0] || !position->second.submitted[1]) {
        if (try_post_exchange(request) == TiledSubmitState::kFatal) {
          return DualRailDrainState::kPoisoned;
        }
      }
      position = exchanges_.find(request.work_id);
      if (position == exchanges_.end()) continue;
      if (position->second.submitted[0] && position->second.submitted[1] &&
          poll_exchange(request) == TiledPollState::kFatal) {
        return DualRailDrainState::kPoisoned;
      }
      position = exchanges_.begin();
      if (!exchanges_.empty()) break;
    }
    if (credit_.has_value()) {
      const auto request = credit_->logical;
      if (!credit_->submitted[0] || !credit_->submitted[1]) {
        if (try_publish_consumed_through(request) ==
            TiledSubmitState::kFatal) {
          return DualRailDrainState::kPoisoned;
        }
      }
      if (credit_.has_value() && credit_->submitted[0] &&
          credit_->submitted[1] &&
          poll_published_consumed_through(request) ==
              TiledPollState::kFatal) {
        return DualRailDrainState::kPoisoned;
      }
    }
    if (!exchanges_.empty() || credit_.has_value()) {
      return DualRailDrainState::kPending;
    }
    return ever_active_ ? DualRailDrainState::kComplete
                        : DualRailDrainState::kIdle;
  }

  DualRailStripedEdgeStatus status() const noexcept {
    const bool safe = !poisoned_ && exchanges_.empty() && !credit_.has_value();
    return {poisoned_, safe, exchanges_.size(), credit_.has_value(),
            emitted_peer_wire_credit_};
  }

 private:
  std::array<TiledEdgeExchangeRequest, 2> split(
      const TiledEdgeExchangeRequest& request) const {
    if (request.edge != logical_edge_ ||
        request.active_bytes < 2U * stripe_alignment_ ||
        request.active_bytes % stripe_alignment_ != 0) {
      throw std::invalid_argument("exchange cannot be split into aligned rails");
    }
    const std::uint64_t lower =
        (request.active_bytes / 2U / stripe_alignment_) * stripe_alignment_;
    const std::uint64_t upper = request.active_bytes - lower;
    if (lower == 0 || upper == 0 || upper % stripe_alignment_ != 0 ||
        lower > std::numeric_limits<std::uint32_t>::max() ||
        upper > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("exchange has an invalid dual-rail stripe");
    }
    auto rail0 = request;
    auto rail1 = request;
    rail0.active_bytes = static_cast<std::uint32_t>(lower);
    rail1.local_payload_offset =
        checked_offset(request.local_payload_offset, lower);
    rail1.remote_payload_offset =
        checked_offset(request.remote_payload_offset, lower);
    rail1.active_bytes = static_cast<std::uint32_t>(upper);
    return {rail0, rail1};
  }

  TiledSubmitState poison_submit() noexcept {
    poisoned_ = true;
    return TiledSubmitState::kFatal;
  }
  TiledPollState poison_poll() noexcept {
    poisoned_ = true;
    return TiledPollState::kFatal;
  }
  TiledCreditPollState poison_credit() noexcept {
    poisoned_ = true;
    return TiledCreditPollState::kFatal;
  }

  std::uint32_t logical_edge_{};
  std::array<TiledEdgePort*, 2> rails_{};
  std::uint32_t stripe_alignment_{};
  std::uintptr_t engine_identity_{};
  std::unordered_map<std::uint64_t, ExchangeState> exchanges_;
  std::optional<CreditState> credit_;
  std::array<std::uint64_t, 2> peer_wire_credits_{};
  std::array<bool, 2> peer_credit_observed_{};
  std::uint64_t emitted_peer_wire_credit_{};
  bool ever_active_{};
  bool poisoned_{};
};

DualRailStripedEdgePort::DualRailStripedEdgePort(
    std::uint32_t logical_edge, TiledEdgePort& rail0,
    TiledEdgePort& rail1, std::uint32_t stripe_alignment)
    : impl_(std::make_unique<Impl>(logical_edge, rail0, rail1,
                                   stripe_alignment)) {}

DualRailStripedEdgePort::~DualRailStripedEdgePort() = default;

std::uint32_t DualRailStripedEdgePort::edge_index() const noexcept {
  return impl_->edge_index();
}

std::uintptr_t DualRailStripedEdgePort::engine_identity() const noexcept {
  return impl_->engine_identity();
}

std::uintptr_t DualRailStripedEdgePort::qp_identity() const noexcept {
  return impl_->qp_identity();
}

TiledSubmitState DualRailStripedEdgePort::try_post_exchange(
    const TiledEdgeExchangeRequest& request) {
  return impl_->try_post_exchange(request);
}

TiledPollState DualRailStripedEdgePort::poll_exchange(
    const TiledEdgeExchangeRequest& request) {
  return impl_->poll_exchange(request);
}

TiledSubmitState DualRailStripedEdgePort::try_publish_consumed_through(
    const TiledCreditPublishRequest& request) {
  return impl_->try_publish_consumed_through(request);
}

TiledPollState DualRailStripedEdgePort::poll_published_consumed_through(
    const TiledCreditPublishRequest& request) {
  return impl_->poll_published_consumed_through(request);
}

TiledCreditPollState DualRailStripedEdgePort::poll_peer_consumed_through(
    const TiledCreditObserveRequest& request,
    std::uint64_t& wire_credit) {
  return impl_->poll_peer_consumed_through(request, wire_credit);
}

DualRailDrainState DualRailStripedEdgePort::drain() noexcept {
  return impl_->drain();
}

DualRailStripedEdgeStatus DualRailStripedEdgePort::status() const noexcept {
  return impl_->status();
}

}  // namespace spark_transport::tiled_prefill_research
