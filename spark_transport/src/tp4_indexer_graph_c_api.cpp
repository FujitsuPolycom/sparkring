#include "spark_transport/tp4_indexer_graph_c_api.h"

#include "spark_transport/tp4_indexer_graph_session.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>

namespace {

void copy_error(
    const char* message, char* error, std::size_t error_bytes) {
  if (error == nullptr || error_bytes == 0) {
    return;
  }
  const std::size_t length =
      std::min(std::strlen(message), error_bytes - 1);
  std::memcpy(error, message, length);
  error[length] = '\0';
}

spark_transport::Tp4IndexerGraphOptions translate(
    const spark_tp4_indexer_graph_config& config) {
  if (config.peer0 == nullptr || config.peer1 == nullptr ||
      config.device0 == nullptr || config.device1 == nullptr) {
    throw std::invalid_argument(
        "TP4 indexer graph config contains a null string");
  }
  if (config.graph_submit_cpu_plus_one == 0 ||
      config.graph_progress_cpu_plus_one == 0) {
    throw std::invalid_argument(
        "TP4 indexer graph CPUs must be encoded as CPU plus one");
  }
  spark_transport::Tp4IndexerGraphOptions options;
  options.rank = config.rank;
  options.peer0 = config.peer0;
  options.peer1 = config.peer1;
  options.device0 = config.device0;
  options.device1 = config.device1;
  options.gid0 = config.gid0;
  options.gid1 = config.gid1;
  options.control_port0 = config.control_port0;
  options.control_port1 = config.control_port1;
  options.graph_submit_cpu =
      config.graph_submit_cpu_plus_one - 1;
  options.graph_progress_cpu =
      config.graph_progress_cpu_plus_one - 1;
  return options;
}

}  // namespace

extern "C" spark_tp4_indexer_graph_handle
spark_tp4_indexer_graph_create(
    const spark_tp4_indexer_graph_config* config, char* error,
    std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument(
          "TP4 indexer graph config is null");
    }
    auto session =
        std::make_unique<spark_transport::Tp4IndexerGraphSession>(
            translate(*config));
    return session.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error(
        "unknown TP4 indexer graph create failure", error,
        error_bytes);
    return nullptr;
  }
}

extern "C" int spark_tp4_indexer_capture_allgather(
    spark_tp4_indexer_graph_handle handle, const void* input,
    void* output, std::uint32_t q, void* cuda_stream, char* error,
    std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument(
          "TP4 indexer graph handle is null");
    }
    static_cast<spark_transport::Tp4IndexerGraphSession*>(handle)
        ->capture_all_gather(input, output, q, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return 1;
  } catch (...) {
    copy_error(
        "unknown TP4 indexer graph capture failure", error,
        error_bytes);
    return 1;
  }
}

extern "C" int spark_tp4_indexer_get_graph_status(
    spark_tp4_indexer_graph_handle handle,
    spark_tp4_indexer_graph_status* status, std::size_t status_bytes,
    char* error, std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument(
          "TP4 indexer graph handle is null");
    }
    if (status == nullptr || status_bytes < sizeof(*status)) {
      throw std::invalid_argument(
          "TP4 indexer graph status buffer is invalid");
    }
    const auto snapshot =
        static_cast<spark_transport::Tp4IndexerGraphSession*>(handle)
            ->graph_replay_status();
    spark_tp4_indexer_graph_status result{};
    result.struct_size = sizeof(result);
    if (snapshot.capture_configured) {
      result.flags |= SPARK_TP4_INDEXER_GRAPH_CAPTURE_CONFIGURED;
    }
    if (snapshot.polling_enabled) {
      result.flags |= SPARK_TP4_INDEXER_GRAPH_POLLING_ENABLED;
    }
    if (snapshot.host_native_atomics_supported) {
      result.flags |= SPARK_TP4_INDEXER_GRAPH_HOST_NATIVE_ATOMICS;
    }
    if (snapshot.submit_affinity_verified) {
      result.flags |=
          SPARK_TP4_INDEXER_GRAPH_SUBMIT_AFFINITY_VERIFIED;
    }
    if (snapshot.progress_affinity_verified) {
      result.flags |=
          SPARK_TP4_INDEXER_GRAPH_PROGRESS_AFFINITY_VERIFIED;
    }
    if (snapshot.overflow_sequence != 0) {
      result.flags |= SPARK_TP4_INDEXER_GRAPH_OVERFLOW_FATAL;
    }
    result.captured_nodes = snapshot.captured_nodes;
    result.captured_q_mask = snapshot.captured_q_mask;
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
    copy_error(
        "unknown TP4 indexer graph status failure", error,
        error_bytes);
    return 1;
  }
}

extern "C" void spark_tp4_indexer_graph_destroy(
    spark_tp4_indexer_graph_handle handle) {
  delete static_cast<spark_transport::Tp4IndexerGraphSession*>(handle);
}
