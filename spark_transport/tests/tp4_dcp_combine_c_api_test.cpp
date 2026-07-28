#include "spark_transport/tp4_c_api.h"

#include <cassert>
#include <cstring>

int main() {
  char error[256]{};
  assert(spark_tp4_dcp_combine(
             nullptr, nullptr, nullptr, nullptr, nullptr, 3, 512, 512,
             3 * 512, nullptr, error, sizeof(error)) == 1);
  assert(std::strstr(error, "combine C API handle is null") != nullptr);

  std::memset(error, 0, sizeof(error));
  assert(spark_tp4_dcp_capture_combine(
             nullptr, nullptr, nullptr, nullptr, nullptr, 3, 512, 512,
             3 * 512, nullptr, error, sizeof(error)) == 1);
  assert(std::strstr(error, "graph combine C API handle is null") != nullptr);
  return 0;
}
