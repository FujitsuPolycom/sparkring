#include "spark_transport/tp4_fused_prefill_c_api.h"
#include <type_traits>
static_assert(std::is_same_v<spark_tp4_fused_prefill_handle, void*>);
static_assert(std::is_same_v<decltype(&spark_tp4_fused_prefill_create),
    void* (*)(const spark_tp4_bidirectional_prefill_config_v1*, char*, size_t)>);
static_assert(std::is_same_v<decltype(&spark_tp4_fused_prefill_all_reduce),
    int (*)(void*, const void*, void*, void*, char*, size_t)>);
static_assert(std::is_same_v<
    decltype(&spark_tp4_fused_prefill_all_reduce_rows),
    int (*)(void*, const void*, void*, uint32_t, void*, char*, size_t)>);
int main() {}
