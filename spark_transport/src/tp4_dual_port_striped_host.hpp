#pragma once

#include "spark_transport/tp4_graph_command.hpp"
#include "spark_transport/tp4_session.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace spark_transport::detail {

// A peer that only understands the legacy or ordinary two-slot layout must
// fail the setup handshake before either side posts RDMA into a striped arena.
constexpr std::uint16_t kTp4DualPortStripedEndpointVersion = 4;
// The tag stays a constant: row geometry is not folded into this 16-bit
// field. Version 4 peers additionally exchange the exact GraphGeometryInfo
// record and compare every field, so geometry identity is never hashed or
// truncated.
constexpr std::uint16_t kTp4DualPortStripedEndpointTag = 0xc204;

enum class Tp4StripedWorkEvent : std::uint64_t {
  kPhase1Doorbell = 1,
  kPhase1Credit = 2,
  kPhase2Doorbell = 3,
  kPhase2Credit = 4,
};

constexpr bool tp4_striped_work_event_valid(
    Tp4StripedWorkEvent event) noexcept {
  switch (event) {
    case Tp4StripedWorkEvent::kPhase1Doorbell:
    case Tp4StripedWorkEvent::kPhase1Credit:
    case Tp4StripedWorkEvent::kPhase2Doorbell:
    case Tp4StripedWorkEvent::kPhase2Credit:
      return true;
  }
  return false;
}

constexpr std::uint64_t tp4_striped_work_id(
    std::uint64_t sequence, Tp4StripedWorkEvent event) {
  if (sequence == 0 || !tp4_striped_work_event_valid(event)) {
    throw std::invalid_argument(
        "TP4 striped WR ID requires a positive sequence and valid event");
  }
  if (sequence > std::numeric_limits<std::uint64_t>::max() / 4U) {
    throw std::overflow_error("TP4 striped WR ID overflow");
  }
  return 4U * (sequence - 1U) + static_cast<std::uint64_t>(event);
}

inline bool tp4_dual_port_striped_options_valid(
    const Tp4AllreduceOptions& options) noexcept {
  const bool graph_cpus_valid =
      options.graph_submit_cpu.has_value() &&
      options.graph_progress_cpu.has_value() &&
      options.graph_submit_cpu != options.graph_progress_cpu;
  const bool capacity_valid =
      options.payload_bytes ==
          tp4_graph_payload_bytes(40, options.bytes_per_row) ||
      options.payload_bytes ==
          tp4_graph_payload_bytes(512, options.bytes_per_row);
  return options.schedule == Tp4AllreduceSchedule::kDualPortStriped &&
         options.protocol == Tp4AllreduceProtocol::kTwoSlotDeferredAck &&
         options.graph_kernel_strategy == Tp4GraphKernelStrategy::kFused &&
         graph_cpus_valid && capacity_valid;
}

}  // namespace spark_transport::detail
