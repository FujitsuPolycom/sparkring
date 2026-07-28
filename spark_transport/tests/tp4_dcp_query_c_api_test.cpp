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

  assert(spark_tp4_dcp_create(nullptr, error, sizeof(error)) == nullptr);
  expect_message_contains(error, "config is null");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_dcp_graph_create(nullptr, error, sizeof(error)) ==
         nullptr);
  expect_message_contains(error, "graph C API config is null");

  std::memset(error, 0, sizeof(error));
  spark_tp4_dcp_config config{};
  config.rank = 0;
  config.peer0 = nullptr;
  config.peer1 = "127.0.0.1";
  config.device0 = "unused0";
  config.device1 = "unused1";
  assert(spark_tp4_dcp_create(&config, error, sizeof(error)) == nullptr);
  expect_message_contains(error, "null string");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_dcp_query_all_gather(
             nullptr, nullptr, nullptr, 3, nullptr, error,
             sizeof(error)) == 1);
  expect_message_contains(error, "handle is null");

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_dcp_capture_query_all_gather(
             nullptr, nullptr, nullptr, 3, nullptr, error,
             sizeof(error)) == 1);
  expect_message_contains(error, "graph query C API handle is null");

  std::memset(error, 0, sizeof(error));
  spark_tp4_dcp_graph_status status{};
  assert(spark_tp4_dcp_get_graph_status(
             nullptr, &status, sizeof(status), error,
             sizeof(error)) == 1);
  expect_message_contains(error, "graph status C API handle is null");

  spark_tp4_dcp_destroy(nullptr);
  return 0;
}
