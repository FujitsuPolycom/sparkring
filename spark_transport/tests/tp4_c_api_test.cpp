#include "spark_transport/tp4_c_api.h"

#include <cassert>
#include <cstring>

namespace {

void expect_message_contains(const char* message, const char* needle) {
  assert(std::strstr(message, needle) != nullptr);
}

}  // namespace

int main() {
  char error[256]{};

  assert(spark_tp4_create_with_protocol(
             nullptr, SPARK_TP4_ALLREDUCE_PROTOCOL_TWO_SLOT_DEFERRED_ACK,
             error, sizeof(error)) == nullptr);
  expect_message_contains(error, "config is null");

  spark_tp4_config protocol_config{};
  protocol_config.rank = 0;
  protocol_config.peer0 = "127.0.0.1";
  protocol_config.peer1 = "127.0.0.1";
  protocol_config.device0 = "unused0";
  protocol_config.device1 = "unused1";
  protocol_config.control_port0 = 9470;
  protocol_config.control_port1 = 9471;
  protocol_config.payload_bytes = 12288;
  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_create_with_protocol(
             &protocol_config, 99, error, sizeof(error)) == nullptr);
  expect_message_contains(error, "invalid TP4 all-reduce protocol");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_create_with_protocol_and_graph_kernel(
             nullptr, SPARK_TP4_ALLREDUCE_PROTOCOL_TWO_SLOT_DEFERRED_ACK,
             SPARK_TP4_GRAPH_KERNEL_TIERED_64K, error,
             sizeof(error)) == nullptr);
  expect_message_contains(error, "config is null");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_create_with_protocol_and_graph_kernel(
             &protocol_config, 99, SPARK_TP4_GRAPH_KERNEL_TIERED_64K,
             error, sizeof(error)) == nullptr);
  expect_message_contains(error, "invalid TP4 all-reduce protocol");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_create_with_protocol_and_graph_kernel(
             &protocol_config, SPARK_TP4_ALLREDUCE_PROTOCOL_SERIAL_ACK,
             99, error, sizeof(error)) == nullptr);
  expect_message_contains(error, "invalid TP4 graph kernel strategy");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_create_with_protocol_graph_kernel_and_schedule(
             &protocol_config,
             SPARK_TP4_ALLREDUCE_PROTOCOL_TWO_SLOT_DEFERRED_ACK,
             SPARK_TP4_GRAPH_KERNEL_FUSED, 99, error,
             sizeof(error)) == nullptr);
  expect_message_contains(error, "invalid TP4 wire schedule");

  assert(spark_tp4_capture_q1_all_reduce(
             nullptr, nullptr, nullptr, nullptr, error, sizeof(error)) == 1);
  expect_message_contains(error, "handle is null");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_capture_all_reduce(
             nullptr, nullptr, nullptr, 5, nullptr, error,
             sizeof(error)) == 1);
  expect_message_contains(error, "handle is null");

  spark_tp4_graph_status graph_status{};
  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_get_graph_status(
             nullptr, &graph_status, sizeof(graph_status), error,
             sizeof(error)) == 1);
  expect_message_contains(error, "handle is null");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_get_graph_status(
             reinterpret_cast<spark_tp4_handle>(1), nullptr, 0, error,
             sizeof(error)) == 1);
  expect_message_contains(error, "status is null");

  std::memset(error, 0, sizeof(error));
  spark_tp4_config invalid_affinity{};
  invalid_affinity.rank = 0;
  invalid_affinity.peer0 = "127.0.0.1";
  invalid_affinity.peer1 = "127.0.0.1";
  invalid_affinity.device0 = "unused0";
  invalid_affinity.device1 = "unused1";
  invalid_affinity.payload_bytes = 12288;
  invalid_affinity.graph_submit_cpu_plus_one = 11;
  invalid_affinity.graph_progress_cpu_plus_one = 11;
  assert(spark_tp4_create(
             &invalid_affinity, error, sizeof(error)) == nullptr);
  expect_message_contains(error, "must be distinct");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_allgather_create(nullptr, error, sizeof(error)) ==
         nullptr);
  expect_message_contains(error, "config is null");

  std::memset(error, 0, sizeof(error));
  spark_tp4_allgather_config config{};
  config.rank = 0;
  config.peer0 = nullptr;
  config.peer1 = "127.0.0.1";
  config.device0 = "unused0";
  config.device1 = "unused1";
  config.input_bytes = 16384;
  assert(spark_tp4_allgather_create(&config, error, sizeof(error)) ==
         nullptr);
  expect_message_contains(error, "null string");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_allgather(nullptr, nullptr, nullptr, nullptr, error,
                            sizeof(error)) == 1);
  expect_message_contains(error, "handle is null");

  // Destroying a null handle follows delete-null semantics and is harmless.
  spark_tp4_destroy(nullptr);
  spark_tp4_allgather_destroy(nullptr);
  return 0;
}
