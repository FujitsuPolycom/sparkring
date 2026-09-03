#pragma once

#include "spark_transport/tp4_bidirectional_prefill_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* spark_tp4_fused_prefill_handle;

spark_tp4_fused_prefill_handle spark_tp4_fused_prefill_create(
    const spark_tp4_bidirectional_prefill_config_v1* config,
    char* error, size_t error_bytes);
int spark_tp4_fused_prefill_all_reduce(
    spark_tp4_fused_prefill_handle handle, const void* input, void* output,
    void* cuda_stream, char* error, size_t error_bytes);
int spark_tp4_fused_prefill_all_reduce_rows(
    spark_tp4_fused_prefill_handle handle, const void* input, void* output,
    uint32_t query_rows, void* cuda_stream, char* error, size_t error_bytes);
void spark_tp4_fused_prefill_destroy(spark_tp4_fused_prefill_handle handle);

#ifdef __cplusplus
}
#endif
