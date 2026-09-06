#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace spark_transport::tiled_prefill_research {

enum class TiledCreditBeginState : std::uint8_t {
  kBackpressured,
  kDelaying,
  kReady,
};

// One edge QP has one send CQ and the correctness adapter has independent
// exchange and credit callers. Once a delayed cumulative credit is scheduled,
// it reserves the signaled-work lane; otherwise an uninterrupted exchange
// stream can win every advance and starve the credit that releases peer slots.
class TiledSignaledWorkGate {
 public:
  explicit TiledSignaledWorkGate(std::uint64_t credit_delay_ticks)
      : credit_delay_ticks_(credit_delay_ticks) {}

  bool try_begin_exchange() noexcept {
    if (exchange_pending_ || credit_pending_ || credit_delay_scheduled_) {
      return false;
    }
    exchange_pending_ = true;
    return true;
  }

  void complete_exchange() {
    if (!exchange_pending_) {
      throw std::logic_error("no tiled exchange is pending");
    }
    exchange_pending_ = false;
  }

  TiledCreditBeginState try_begin_credit(std::uint64_t now_ticks) {
    if (exchange_pending_ || credit_pending_) {
      return TiledCreditBeginState::kBackpressured;
    }
    if (!credit_delay_scheduled_) {
      if (now_ticks > std::numeric_limits<std::uint64_t>::max() -
                          credit_delay_ticks_) {
        throw std::overflow_error("tiled credit delay deadline exhausted");
      }
      credit_delay_scheduled_ = true;
      credit_ready_tick_ = now_ticks + credit_delay_ticks_;
    }
    if (now_ticks < credit_ready_tick_) {
      return TiledCreditBeginState::kDelaying;
    }
    credit_delay_scheduled_ = false;
    credit_pending_ = true;
    return TiledCreditBeginState::kReady;
  }

  void complete_credit() {
    if (!credit_pending_) {
      throw std::logic_error("no tiled credit is pending");
    }
    credit_pending_ = false;
  }

  constexpr bool exchange_pending() const noexcept {
    return exchange_pending_;
  }
  constexpr bool credit_pending() const noexcept { return credit_pending_; }
  constexpr bool credit_delay_scheduled() const noexcept {
    return credit_delay_scheduled_;
  }

 private:
  std::uint64_t credit_delay_ticks_{};
  std::uint64_t credit_ready_tick_{};
  bool exchange_pending_{};
  bool credit_pending_{};
  bool credit_delay_scheduled_{};
};

}  // namespace spark_transport::tiled_prefill_research
