#pragma once

#include "spark_transport/tp4_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* spark_tp4_bidirectional_prefill_handle;

typedef struct spark_tp4_bidirectional_prefill_config_v1 {
  uint32_t struct_size;
  spark_tp4_config_v2 primary;
  uint32_t rail_count;
  uint32_t query_rows;
  const char* secondary_peer0;
  const char* secondary_peer1;
  const char* secondary_device0;
  const char* secondary_device1;
  uint8_t secondary_gid0;
  uint8_t secondary_gid1;
  uint16_t secondary_control_port0;
  uint16_t secondary_control_port1;
  uint32_t timeout_seconds;
} spark_tp4_bidirectional_prefill_config_v1;

spark_tp4_bidirectional_prefill_handle
spark_tp4_bidirectional_prefill_create(
    const spark_tp4_bidirectional_prefill_config_v1* config,
    char* error, size_t error_bytes);

int spark_tp4_bidirectional_prefill_all_reduce(
    spark_tp4_bidirectional_prefill_handle handle, const void* input,
    void* output, void* cuda_stream, char* error, size_t error_bytes);

/* Copies host state only. This function never synchronizes a CUDA stream. */
int spark_tp4_bidirectional_prefill_get_health_status(
    spark_tp4_bidirectional_prefill_handle handle,
    spark_tp4_health_status* status, size_t status_bytes,
    char* error, size_t error_bytes);

void spark_tp4_bidirectional_prefill_destroy(
    spark_tp4_bidirectional_prefill_handle handle);

#ifdef __cplusplus
}
#endif
