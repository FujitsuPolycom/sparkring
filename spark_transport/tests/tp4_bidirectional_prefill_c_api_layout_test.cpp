#include "spark_transport/tp4_bidirectional_prefill_c_api.h"

#include <type_traits>
#include <cstddef>

static_assert(std::is_same_v<spark_tp4_bidirectional_prefill_handle, void*>);
static_assert(std::is_standard_layout_v<
              spark_tp4_bidirectional_prefill_config_v1>);
static_assert(std::is_trivially_copyable_v<
              spark_tp4_bidirectional_prefill_config_v1>);
static_assert(sizeof(void*) != 8 ||
              sizeof(spark_tp4_bidirectional_prefill_config_v1) == 144);
static_assert(sizeof(void*) != 8 ||
              offsetof(spark_tp4_bidirectional_prefill_config_v1, primary) ==
                  8);
static_assert(sizeof(void*) != 8 ||
              offsetof(spark_tp4_bidirectional_prefill_config_v1,
                       rail_count) == 88);
static_assert(sizeof(void*) != 8 ||
              offsetof(spark_tp4_bidirectional_prefill_config_v1,
                       query_rows) == 92);
static_assert(sizeof(void*) != 8 ||
              offsetof(spark_tp4_bidirectional_prefill_config_v1,
                       secondary_peer0) == 96);
static_assert(sizeof(void*) != 8 ||
              offsetof(spark_tp4_bidirectional_prefill_config_v1,
                       timeout_seconds) == 136);
static_assert(std::is_same_v<
              decltype(&spark_tp4_bidirectional_prefill_create),
              spark_tp4_bidirectional_prefill_handle (*)(
                  const spark_tp4_bidirectional_prefill_config_v1*, char*,
                  size_t)>);
static_assert(std::is_same_v<
              decltype(&spark_tp4_bidirectional_prefill_all_reduce),
              int (*)(spark_tp4_bidirectional_prefill_handle, const void*,
                      void*, void*, char*, size_t)>);
static_assert(std::is_same_v<
              decltype(&spark_tp4_bidirectional_prefill_destroy),
              void (*)(spark_tp4_bidirectional_prefill_handle)>);

int main() {}
