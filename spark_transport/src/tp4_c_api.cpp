#include "spark_transport/tp4_c_api.h"

#include "spark_transport/tp4_session.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

void copy_error(const char* message, char* error, std::size_t error_bytes) {
  if (error == nullptr || error_bytes == 0) {
    return;
  }
  const std::size_t length =
      std::min(std::strlen(message), error_bytes - 1);
  std::memcpy(error, message, length);
  error[length] = '\0';
}

spark_transport::Tp4AllreduceOptions translate(
    const spark_tp4_config& config) {
  if (config.peer0 == nullptr || config.peer1 == nullptr ||
      config.device0 == nullptr || config.device1 == nullptr) {
    throw std::invalid_argument("TP4 C API config contains a null string");
  }
  spark_transport::Tp4AllreduceOptions options;
  options.rank = config.rank;
  options.peer0 = config.peer0;
  options.peer1 = config.peer1;
  options.device0 = config.device0;
  options.device1 = config.device1;
  options.gid0 = config.gid0;
  options.gid1 = config.gid1;
  options.control_port0 = config.control_port0;
  options.control_port1 = config.control_port1;
  options.payload_bytes = config.payload_bytes;
  const bool submit_cpu_set = config.graph_submit_cpu_plus_one != 0;
  const bool progress_cpu_set = config.graph_progress_cpu_plus_one != 0;
  if (submit_cpu_set != progress_cpu_set) {
    throw std::invalid_argument(
        "graph submit/progress CPUs must be configured together");
  }
  if (submit_cpu_set) {
    options.graph_submit_cpu = config.graph_submit_cpu_plus_one - 1;
    options.graph_progress_cpu = config.graph_progress_cpu_plus_one - 1;
    if (options.graph_submit_cpu == options.graph_progress_cpu) {
      throw std::invalid_argument(
          "graph submit/progress CPUs must be distinct");
    }
  }
  return options;
}

spark_transport::Tp4AllreduceOptions translate(
    const spark_tp4_config_v2& config) {
  if (config.struct_size < sizeof(spark_tp4_config_v2)) {
    throw std::invalid_argument("TP4 C API v2 config is too small");
  }
  auto options = translate(config.base);
  options.elements_per_row = config.elements_per_row;
  options.bytes_per_row = config.bytes_per_row;
  return options;
}


spark_transport::Tp4GraphKernelStrategy graph_kernel_from_wire(
    std::uint32_t value) {
  using spark_transport::Tp4GraphKernelStrategy;
  switch (value) {
    case SPARK_TP4_GRAPH_KERNEL_FUSED:
      return Tp4GraphKernelStrategy::kFused;
    case SPARK_TP4_GRAPH_KERNEL_SPLIT_64K:
      return Tp4GraphKernelStrategy::kSplit64KiB;
    case SPARK_TP4_GRAPH_KERNEL_TIERED_64K:
      return Tp4GraphKernelStrategy::kTiered64KiB;
    default:
      throw std::invalid_argument("invalid TP4 graph kernel strategy");
  }
}

spark_transport::Tp4AllreduceSchedule schedule_from_wire(
    std::uint32_t value) {
  using spark_transport::Tp4AllreduceSchedule;
  switch (value) {
    case SPARK_TP4_WIRE_SCHEDULE_SEQUENTIAL:
      return Tp4AllreduceSchedule::kSequential;
    case SPARK_TP4_WIRE_SCHEDULE_DUAL_PORT_STRIPED:
      return Tp4AllreduceSchedule::kDualPortStriped;
    default:
      throw std::invalid_argument("invalid TP4 wire schedule");
  }
}

struct Tp4CAllreduceHandle {
  explicit Tp4CAllreduceHandle(
      spark_transport::Tp4AllreduceOptions options)
      : graph_kernel_strategy(options.graph_kernel_strategy),
        schedule(options.schedule),
        session(std::move(options)) {}

  spark_transport::Tp4GraphKernelStrategy graph_kernel_strategy;
  spark_transport::Tp4AllreduceSchedule schedule;
  spark_transport::Tp4AllreduceSession session;
};

Tp4CAllreduceHandle* tp4_allreduce_handle(spark_tp4_handle handle) {
  return static_cast<Tp4CAllreduceHandle*>(handle);
}

}  // namespace

extern "C" spark_tp4_handle spark_tp4_create(
    const spark_tp4_config* config, char* error, std::size_t error_bytes) {
  return spark_tp4_create_with_protocol_and_graph_kernel(
      config, SPARK_TP4_ALLREDUCE_PROTOCOL_SERIAL_ACK,
      SPARK_TP4_GRAPH_KERNEL_FUSED, error, error_bytes);
}

extern "C" spark_tp4_handle spark_tp4_create_with_protocol(
    const spark_tp4_config* config, std::uint32_t protocol, char* error,
    std::size_t error_bytes) {
  return spark_tp4_create_with_protocol_and_graph_kernel(
      config, protocol, SPARK_TP4_GRAPH_KERNEL_FUSED, error,
      error_bytes);
}

extern "C" spark_tp4_handle
spark_tp4_create_with_protocol_and_graph_kernel(
    const spark_tp4_config* config, std::uint32_t protocol,
    std::uint32_t graph_kernel, char* error,
    std::size_t error_bytes) {
  return spark_tp4_create_with_protocol_graph_kernel_and_schedule(
      config, protocol, graph_kernel,
      SPARK_TP4_WIRE_SCHEDULE_SEQUENTIAL, error, error_bytes);
}

extern "C" spark_tp4_handle
spark_tp4_create_with_protocol_graph_kernel_and_schedule(
    const spark_tp4_config* config, std::uint32_t protocol,
    std::uint32_t graph_kernel, std::uint32_t wire_schedule,
    char* error, std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument("TP4 C API config is null");
    }
    auto options = translate(*config);
    options.protocol =
        spark_transport::tp4_allreduce_protocol_from_wire(protocol);
    options.graph_kernel_strategy = graph_kernel_from_wire(graph_kernel);
    options.schedule = schedule_from_wire(wire_schedule);
    auto handle =
        std::make_unique<Tp4CAllreduceHandle>(std::move(options));
    return handle.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error("unknown TP4 create failure", error, error_bytes);
    return nullptr;
  }
}

extern "C" spark_tp4_handle spark_tp4_create_v2(
    const spark_tp4_config_v2* config, std::uint32_t protocol,
    std::uint32_t graph_kernel, std::uint32_t wire_schedule,
    char* error, std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument("TP4 C API v2 config is null");
    }
    auto options = translate(*config);
    options.protocol =
        spark_transport::tp4_allreduce_protocol_from_wire(protocol);
    options.graph_kernel_strategy = graph_kernel_from_wire(graph_kernel);
    options.schedule = schedule_from_wire(wire_schedule);
    auto handle =
        std::make_unique<Tp4CAllreduceHandle>(std::move(options));
    return handle.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error("unknown TP4 v2 create failure", error, error_bytes);
    return nullptr;
  }
}

extern "C" int spark_tp4_all_reduce(
    spark_tp4_handle handle, const void* input, void* output,
    void* cuda_stream, char* error, std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("TP4 C API handle is null");
    }
    tp4_allreduce_handle(handle)->session.all_reduce(input, output,
                                                      cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 all-reduce failure", error, error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_capture_all_reduce(
    spark_tp4_handle handle, const void* input, void* output,
    std::uint32_t q, void* cuda_stream, char* error,
    std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("TP4 C API handle is null");
    }
    tp4_allreduce_handle(handle)->session.capture_all_reduce(
        input, output, q, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 graph capture failure", error, error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_capture_q1_all_reduce(
    spark_tp4_handle handle, const void* input, void* output,
    void* cuda_stream, char* error, std::size_t error_bytes) {
  return spark_tp4_capture_all_reduce(
      handle, input, output, 1, cuda_stream, error, error_bytes);
}

extern "C" int spark_tp4_get_graph_status(
    spark_tp4_handle handle, spark_tp4_graph_status* status,
    std::size_t status_bytes, char* error, std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("TP4 C API handle is null");
    }
    if (status == nullptr) {
      throw std::invalid_argument("TP4 graph status is null");
    }
    if (status_bytes < sizeof(*status)) {
      throw std::invalid_argument("TP4 graph status buffer is too small");
    }

    const auto* c_handle = tp4_allreduce_handle(handle);
    const auto snapshot = c_handle->session.graph_replay_status();
    spark_tp4_graph_status result{};
    result.struct_size = sizeof(result);
    if (snapshot.capture_configured) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_CAPTURE_CONFIGURED;
    }
    if (snapshot.polling_enabled) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_POLLING_ENABLED;
    }
    if (snapshot.host_native_atomics_supported) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_HOST_NATIVE_ATOMICS;
    }
    if (snapshot.submit_affinity_verified) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED;
    }
    if (snapshot.progress_affinity_verified) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED;
    }
    if (snapshot.two_slot_deferred_ack) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_TWO_SLOT_DEFERRED_ACK;
    }
    if (snapshot.direct_doorbell) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_DIRECT_DOORBELL;
    }
    if (c_handle->graph_kernel_strategy ==
        spark_transport::Tp4GraphKernelStrategy::kSplit64KiB) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_SPLIT_64K;
    } else if (c_handle->graph_kernel_strategy ==
               spark_transport::Tp4GraphKernelStrategy::kTiered64KiB) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_TIERED_64K;
    }
    if (c_handle->schedule ==
        spark_transport::Tp4AllreduceSchedule::kDualPortStriped) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_DUAL_PORT_STRIPED;
    }
    if (snapshot.overflow_sequence != 0) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_OVERFLOW_FATAL;
    }
    result.captured_nodes = snapshot.captured_nodes;
    result.published_sequence = snapshot.published_sequence;
    result.consumed_sequence = snapshot.consumed_sequence;
    result.completed_sequence = snapshot.completed_sequence;
    result.overflow_sequence = snapshot.overflow_sequence;
    if (snapshot.graph_submit_cpu >= 0) {
      result.graph_submit_cpu_plus_one =
          static_cast<std::uint32_t>(snapshot.graph_submit_cpu) + 1;
    }
    if (snapshot.graph_progress_cpu >= 0) {
      result.graph_progress_cpu_plus_one =
          static_cast<std::uint32_t>(snapshot.graph_progress_cpu) + 1;
    }
    *status = result;
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 graph status failure", error, error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_get_health_status(
    spark_tp4_handle handle, spark_tp4_health_status* status,
    std::size_t status_bytes, char* error, std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("TP4 C API handle is null");
    }
    if (status == nullptr) {
      throw std::invalid_argument("TP4 health status is null");
    }
    if (status_bytes < sizeof(*status)) {
      throw std::invalid_argument("TP4 health status buffer is too small");
    }
    const auto snapshot = tp4_allreduce_handle(handle)->session.health_status();
    spark_tp4_health_status result{};
    result.struct_size = sizeof(result);
    if (snapshot.healthy) result.flags |= SPARK_TP4_HEALTHY;
    if (snapshot.poisoned) result.flags |= SPARK_TP4_HEALTH_POISONED;
    if (snapshot.progress_thread_running) {
      result.flags |= SPARK_TP4_HEALTH_PROGRESS_THREAD_RUNNING;
    }
    if (snapshot.stopping) result.flags |= SPARK_TP4_HEALTH_STOPPING;
    result.submitted_sequence = snapshot.submitted_sequence;
    result.completed_sequence = snapshot.completed_sequence;
    result.failing_sequence = snapshot.failing_sequence;
    result.error_code = snapshot.error_code;
    result.failing_stage = -1;
    result.failing_rail = -1;
    result.failing_peer = -1;
    *status = result;
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 health status failure", error, error_bytes);
    return 1;
  }
}

extern "C" void spark_tp4_destroy(spark_tp4_handle handle) {
  delete tp4_allreduce_handle(handle);
}
