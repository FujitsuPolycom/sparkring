#include "spark_transport/tp4_c_api.h"

#include "spark_transport/tp4_allgather_session.hpp"
#include "spark_transport/tp4_dcp_session.hpp"
#include "spark_transport/tp4_session.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>

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

spark_transport::Tp4AllgatherOptions translate(
    const spark_tp4_allgather_config& config) {
  if (config.peer0 == nullptr || config.peer1 == nullptr ||
      config.device0 == nullptr || config.device1 == nullptr) {
    throw std::invalid_argument(
        "TP4 all-gather C API config contains a null string");
  }
  spark_transport::Tp4AllgatherOptions options;
  options.rank = config.rank;
  options.peer0 = config.peer0;
  options.peer1 = config.peer1;
  options.device0 = config.device0;
  options.device1 = config.device1;
  options.gid0 = config.gid0;
  options.gid1 = config.gid1;
  options.control_port0 = config.control_port0;
  options.control_port1 = config.control_port1;
  options.input_bytes = config.input_bytes;
  return options;
}

spark_transport::Tp4DcpOptions translate(
    const spark_tp4_dcp_config& config) {
  if (config.peer0 == nullptr || config.peer1 == nullptr ||
      config.device0 == nullptr || config.device1 == nullptr) {
    throw std::invalid_argument(
        "TP4 DCP query C API config contains a null string");
  }
  spark_transport::Tp4DcpOptions options;
  options.rank = config.rank;
  options.peer0 = config.peer0;
  options.peer1 = config.peer1;
  options.device0 = config.device0;
  options.device1 = config.device1;
  options.gid0 = config.gid0;
  options.gid1 = config.gid1;
  options.control_port0 = config.control_port0;
  options.control_port1 = config.control_port1;
  return options;
}

spark_transport::Tp4DcpOptions translate(
    const spark_tp4_dcp_graph_config& config) {
  if (config.peer0 == nullptr || config.peer1 == nullptr ||
      config.device0 == nullptr || config.device1 == nullptr) {
    throw std::invalid_argument(
        "TP4 DCP graph C API config contains a null string");
  }
  const bool submit_cpu_set = config.graph_submit_cpu_plus_one != 0;
  const bool progress_cpu_set = config.graph_progress_cpu_plus_one != 0;
  if (!submit_cpu_set || !progress_cpu_set) {
    throw std::invalid_argument(
        "DCP graph submit/progress CPUs must both be configured");
  }
  spark_transport::Tp4DcpOptions options;
  options.rank = config.rank;
  options.peer0 = config.peer0;
  options.peer1 = config.peer1;
  options.device0 = config.device0;
  options.device1 = config.device1;
  options.gid0 = config.gid0;
  options.gid1 = config.gid1;
  options.control_port0 = config.control_port0;
  options.control_port1 = config.control_port1;
  options.graph_submit_cpu = config.graph_submit_cpu_plus_one - 1;
  options.graph_progress_cpu = config.graph_progress_cpu_plus_one - 1;
  if (options.graph_submit_cpu == options.graph_progress_cpu) {
    throw std::invalid_argument(
        "DCP graph submit/progress CPUs must be distinct");
  }
  return options;
}

}  // namespace

extern "C" spark_tp4_handle spark_tp4_create(
    const spark_tp4_config* config, char* error, std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument("TP4 C API config is null");
    }
    auto session = std::make_unique<spark_transport::Tp4AllreduceSession>(
        translate(*config));
    return session.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error("unknown TP4 create failure", error, error_bytes);
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
    static_cast<spark_transport::Tp4AllreduceSession*>(handle)->all_reduce(
        input, output, cuda_stream);
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
    static_cast<spark_transport::Tp4AllreduceSession*>(handle)
        ->capture_all_reduce(input, output, q, cuda_stream);
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

    const auto snapshot =
        static_cast<spark_transport::Tp4AllreduceSession*>(handle)
            ->graph_replay_status();
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

extern "C" void spark_tp4_destroy(spark_tp4_handle handle) {
  delete static_cast<spark_transport::Tp4AllreduceSession*>(handle);
}

extern "C" spark_tp4_allgather_handle spark_tp4_allgather_create(
    const spark_tp4_allgather_config* config, char* error,
    std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument("TP4 all-gather C API config is null");
    }
    auto session =
        std::make_unique<spark_transport::Tp4AllgatherSession>(
            translate(*config));
    return session.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error("unknown TP4 all-gather create failure", error, error_bytes);
    return nullptr;
  }
}

extern "C" int spark_tp4_allgather(
    spark_tp4_allgather_handle handle, const void* input, void* output,
    void* cuda_stream, char* error, std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("TP4 all-gather C API handle is null");
    }
    static_cast<spark_transport::Tp4AllgatherSession*>(handle)->all_gather(
        input, output, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 all-gather failure", error, error_bytes);
    return 1;
  }
}

extern "C" void spark_tp4_allgather_destroy(
    spark_tp4_allgather_handle handle) {
  delete static_cast<spark_transport::Tp4AllgatherSession*>(handle);
}

extern "C" spark_tp4_dcp_handle spark_tp4_dcp_create(
    const spark_tp4_dcp_config* config, char* error,
    std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument("TP4 DCP query C API config is null");
    }
    auto session =
        std::make_unique<spark_transport::Tp4DcpSession>(
            translate(*config));
    return session.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error("unknown TP4 DCP query create failure", error, error_bytes);
    return nullptr;
  }
}

extern "C" spark_tp4_dcp_handle spark_tp4_dcp_graph_create(
    const spark_tp4_dcp_graph_config* config, char* error,
    std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument(
          "TP4 DCP graph C API config is null");
    }
    auto session =
        std::make_unique<spark_transport::Tp4DcpSession>(
            translate(*config));
    return session.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error("unknown TP4 DCP graph create failure", error,
               error_bytes);
    return nullptr;
  }
}

extern "C" int spark_tp4_dcp_query_all_gather(
    spark_tp4_dcp_handle handle, const void* input, void* output,
    std::uint32_t q, void* cuda_stream, char* error,
    std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("TP4 DCP query C API handle is null");
    }
    static_cast<spark_transport::Tp4DcpSession*>(handle)
        ->query_all_gather(input, output, q, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 DCP query all-gather failure", error,
               error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_dcp_capture_query_all_gather(
    spark_tp4_dcp_handle handle, const void* input, void* output,
    std::uint32_t q, void* cuda_stream, char* error,
    std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument(
          "TP4 DCP graph query C API handle is null");
    }
    static_cast<spark_transport::Tp4DcpSession*>(handle)
        ->capture_query_all_gather(input, output, q, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 DCP graph query failure", error,
               error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_dcp_combine(
    spark_tp4_dcp_handle handle, const void* output_bf16,
    const void* lse_fp32, void* reduced_output_bf16,
    void* reduced_lse_fp32, std::uint32_t q,
    std::uint32_t head_dimension, std::uint32_t query_stride,
    std::uint32_t head_stride, void* cuda_stream,
    char* error, std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("TP4 DCP combine C API handle is null");
    }
    static_cast<spark_transport::Tp4DcpSession*>(handle)->combine(
        output_bf16, lse_fp32, reduced_output_bf16, reduced_lse_fp32,
        q, head_dimension, query_stride, head_stride, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 DCP combine failure", error, error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_dcp_capture_combine(
    spark_tp4_dcp_handle handle, const void* output_bf16,
    const void* lse_fp32, void* reduced_output_bf16,
    void* reduced_lse_fp32, std::uint32_t q,
    std::uint32_t head_dimension, std::uint32_t query_stride,
    std::uint32_t head_stride, void* cuda_stream,
    char* error, std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument(
          "TP4 DCP graph combine C API handle is null");
    }
    static_cast<spark_transport::Tp4DcpSession*>(handle)
        ->capture_combine(
            output_bf16, lse_fp32, reduced_output_bf16,
            reduced_lse_fp32, q, head_dimension, query_stride,
            head_stride, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 DCP graph combine failure", error,
               error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_dcp_get_graph_status(
    spark_tp4_dcp_handle handle, spark_tp4_dcp_graph_status* status,
    std::size_t status_bytes, char* error,
    std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument(
          "TP4 DCP graph status C API handle is null");
    }
    if (status == nullptr ||
        status_bytes < sizeof(spark_tp4_dcp_graph_status)) {
      throw std::invalid_argument(
          "TP4 DCP graph status buffer is too small");
    }
    const auto snapshot =
        static_cast<spark_transport::Tp4DcpSession*>(handle)
            ->graph_replay_status();
    spark_tp4_dcp_graph_status result{};
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
    if (snapshot.dedicated_spin) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_DEDICATED_SPIN;
    }
    if (snapshot.overflow_sequence != 0) {
      result.flags |= SPARK_TP4_GRAPH_STATUS_OVERFLOW_FATAL;
    }
    result.captured_nodes = snapshot.captured_nodes;
    result.captured_query_nodes = snapshot.captured_query_nodes;
    result.captured_combine_nodes = snapshot.captured_combine_nodes;
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
    std::memcpy(status, &result, sizeof(result));
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error("unknown TP4 DCP graph status failure", error,
               error_bytes);
    return 1;
  }
}

extern "C" void spark_tp4_dcp_destroy(spark_tp4_dcp_handle handle) {
  delete static_cast<spark_transport::Tp4DcpSession*>(handle);
}
