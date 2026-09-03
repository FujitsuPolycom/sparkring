#include "spark_transport/tp4_bidirectional_prefill_c_api.h"

#include "spark_transport/tp4_bidirectional_prefill_session.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

void copy_error(const char* message, char* error, std::size_t error_bytes) {
  if (error == nullptr || error_bytes == 0) return;
  const std::size_t length =
      std::min(std::strlen(message), error_bytes - 1U);
  std::memcpy(error, message, length);
  error[length] = '\0';
}

spark_transport::Tp4BidirectionalPrefillOptions translate(
    const spark_tp4_bidirectional_prefill_config_v1& config) {
  if (config.struct_size < sizeof(config) ||
      config.primary.struct_size < sizeof(config.primary)) {
    throw std::invalid_argument("bidirectional prefill config is too small");
  }
  const auto& primary = config.primary;
  if (primary.base.peer0 == nullptr || primary.base.peer1 == nullptr ||
      primary.base.device0 == nullptr || primary.base.device1 == nullptr) {
    throw std::invalid_argument(
        "bidirectional prefill config contains a null string");
  }
  if (primary.base.graph_submit_cpu_plus_one != 0 ||
      primary.base.graph_progress_cpu_plus_one != 0) {
    throw std::invalid_argument(
        "bidirectional eager prefill does not accept graph CPU affinity");
  }
  spark_transport::Tp4BidirectionalPrefillOptions options;
  options.rank = primary.base.rank;
  options.peer0 = primary.base.peer0;
  options.peer1 = primary.base.peer1;
  options.device0 = primary.base.device0;
  options.device1 = primary.base.device1;
  options.gid0 = primary.base.gid0;
  options.gid1 = primary.base.gid1;
  options.control_port0 = primary.base.control_port0;
  options.control_port1 = primary.base.control_port1;
  options.rail_count = config.rail_count;
  options.query_rows = config.query_rows;
  options.elements_per_row = primary.elements_per_row;
  options.timeout_seconds = config.timeout_seconds;
  const bool secondary_absent = config.secondary_peer0 == nullptr &&
      config.secondary_peer1 == nullptr && config.secondary_device0 == nullptr &&
      config.secondary_device1 == nullptr && config.secondary_gid0 == 0 &&
      config.secondary_gid1 == 0 && config.secondary_control_port0 == 0 &&
      config.secondary_control_port1 == 0;
  if (config.rail_count == 1 && !secondary_absent) {
    throw std::invalid_argument("single-rail config must omit secondary topology");
  }
  if (config.rail_count == 2) {
    if (config.secondary_peer0 == nullptr || config.secondary_peer1 == nullptr ||
        config.secondary_device0 == nullptr || config.secondary_device1 == nullptr) {
      throw std::invalid_argument("dual-rail config requires secondary topology");
    }
    options.secondary_peer0 = config.secondary_peer0;
    options.secondary_peer1 = config.secondary_peer1;
    options.secondary_device0 = config.secondary_device0;
    options.secondary_device1 = config.secondary_device1;
    options.secondary_gid0 = config.secondary_gid0;
    options.secondary_gid1 = config.secondary_gid1;
    options.secondary_control_port0 = config.secondary_control_port0;
    options.secondary_control_port1 = config.secondary_control_port1;
  } else if (config.rail_count != 1) {
    throw std::invalid_argument("rail_count must be one or two");
  }
  const std::uint64_t expected_payload =
      static_cast<std::uint64_t>(config.query_rows) * primary.bytes_per_row;
  if (primary.bytes_per_row != primary.elements_per_row * 2U ||
      primary.base.payload_bytes != expected_payload) {
    throw std::invalid_argument(
        "bidirectional prefill config geometry/payload mismatch");
  }
  return options;
}

struct Handle {
  explicit Handle(spark_transport::Tp4BidirectionalPrefillOptions options)
      : session(std::move(options)) {}
  spark_transport::Tp4BidirectionalPrefillSession session;
};

}  // namespace

extern "C" spark_tp4_bidirectional_prefill_handle
spark_tp4_bidirectional_prefill_create(
    const spark_tp4_bidirectional_prefill_config_v1* config,
    char* error, std::size_t error_bytes) {
  try {
    if (config == nullptr) {
      throw std::invalid_argument("bidirectional prefill config is null");
    }
    auto handle = std::make_unique<Handle>(translate(*config));
    return handle.release();
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return nullptr;
  } catch (...) {
    copy_error("unknown bidirectional prefill create failure", error,
               error_bytes);
    return nullptr;
  }
}

extern "C" int spark_tp4_bidirectional_prefill_all_reduce(
    spark_tp4_bidirectional_prefill_handle handle, const void* input,
    void* output, void* cuda_stream, char* error,
    std::size_t error_bytes) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("bidirectional prefill handle is null");
    }
    static_cast<Handle*>(handle)->session.all_reduce(
        input, output, cuda_stream);
    return 0;
  } catch (const std::exception& exception) {
    copy_error(exception.what(), error, error_bytes);
    return -1;
  } catch (...) {
    copy_error("unknown bidirectional prefill all-reduce failure", error,
               error_bytes);
    return -1;
  }
}

extern "C" void spark_tp4_bidirectional_prefill_destroy(
    spark_tp4_bidirectional_prefill_handle handle) {
  delete static_cast<Handle*>(handle);
}
