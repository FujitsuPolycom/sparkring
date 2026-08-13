#include "tp4_dual_port_striped_host.hpp"

#include <array>
#include <cassert>

int main() {
  using spark_transport::Tp4AllreduceOptions;
  using spark_transport::Tp4AllreduceProtocol;
  using spark_transport::Tp4AllreduceSchedule;
  using spark_transport::Tp4GraphKernelStrategy;
  using spark_transport::detail::kTp4DualPortStripedEndpointTag;
  using spark_transport::detail::kTp4DualPortStripedEndpointVersion;
  using spark_transport::detail::Tp4StripedWorkEvent;
  using spark_transport::detail::tp4_dual_port_striped_options_valid;
  using spark_transport::detail::tp4_striped_work_id;

  static_assert(kTp4DualPortStripedEndpointVersion != 1);
  static_assert(kTp4DualPortStripedEndpointVersion != 2);
  static_assert(kTp4DualPortStripedEndpointTag != 0);
  static_assert(kTp4DualPortStripedEndpointTag !=
                spark_transport::kTp4TwoSlotEndpointTag);

  constexpr std::array<std::uint64_t, 4> sequence1_work_ids{
      tp4_striped_work_id(1, Tp4StripedWorkEvent::kPhase1Doorbell),
      tp4_striped_work_id(1, Tp4StripedWorkEvent::kPhase1Credit),
      tp4_striped_work_id(1, Tp4StripedWorkEvent::kPhase2Doorbell),
      tp4_striped_work_id(1, Tp4StripedWorkEvent::kPhase2Credit),
  };
  static_assert(sequence1_work_ids[0] == 1);
  static_assert(sequence1_work_ids[1] == 2);
  static_assert(sequence1_work_ids[2] == 3);
  static_assert(sequence1_work_ids[3] == 4);
  static_assert(
      tp4_striped_work_id(2, Tp4StripedWorkEvent::kPhase1Doorbell) == 5);

  Tp4AllreduceOptions options;
  options.schedule = Tp4AllreduceSchedule::kDualPortStriped;
  options.protocol = Tp4AllreduceProtocol::kTwoSlotDeferredAck;
  options.graph_kernel_strategy = Tp4GraphKernelStrategy::kFused;
  options.payload_bytes = spark_transport::tp4_graph_payload_bytes(40);
  options.graph_submit_cpu = 10;
  options.graph_progress_cpu = 11;
  assert(tp4_dual_port_striped_options_valid(options));

  options.payload_bytes = spark_transport::tp4_graph_payload_bytes(512);
  assert(tp4_dual_port_striped_options_valid(options));

  options.protocol = Tp4AllreduceProtocol::kSerialAck;
  assert(!tp4_dual_port_striped_options_valid(options));
  options.protocol = Tp4AllreduceProtocol::kTwoSlotDeferredAck;

  options.graph_kernel_strategy = Tp4GraphKernelStrategy::kSplit64KiB;
  assert(!tp4_dual_port_striped_options_valid(options));
  options.graph_kernel_strategy = Tp4GraphKernelStrategy::kTiered64KiB;
  assert(!tp4_dual_port_striped_options_valid(options));
  options.graph_kernel_strategy = Tp4GraphKernelStrategy::kFused;

  options.graph_progress_cpu.reset();
  assert(!tp4_dual_port_striped_options_valid(options));
  options.graph_progress_cpu = 11;

  options.payload_bytes = spark_transport::tp4_graph_payload_bytes(39);
  assert(!tp4_dual_port_striped_options_valid(options));
}
