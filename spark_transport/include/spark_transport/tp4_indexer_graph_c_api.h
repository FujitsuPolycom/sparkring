#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spark_tp4_indexer_graph_config {
  uint32_t rank;
  const char* peer0;
  const char* peer1;
  const char* device0;
  const char* device1;
  uint8_t gid0;
  uint8_t gid1;
  uint16_t control_port0;
  uint16_t control_port1;
  /* Both are required and encode CPU index plus one. */
  uint32_t graph_submit_cpu_plus_one;
  uint32_t graph_progress_cpu_plus_one;
} spark_tp4_indexer_graph_config;

typedef void* spark_tp4_indexer_graph_handle;

typedef struct spark_tp4_indexer_graph_status {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t captured_nodes;
  uint64_t captured_q_mask;
  uint64_t published_sequence;
  uint64_t consumed_sequence;
  uint64_t completed_sequence;
  uint64_t overflow_sequence;
  uint32_t graph_submit_cpu_plus_one;
  uint32_t graph_progress_cpu_plus_one;
} spark_tp4_indexer_graph_status;

enum {
  SPARK_TP4_INDEXER_GRAPH_CAPTURE_CONFIGURED = 1U << 0,
  SPARK_TP4_INDEXER_GRAPH_POLLING_ENABLED = 1U << 1,
  SPARK_TP4_INDEXER_GRAPH_HOST_NATIVE_ATOMICS = 1U << 2,
  SPARK_TP4_INDEXER_GRAPH_SUBMIT_AFFINITY_VERIFIED = 1U << 3,
  SPARK_TP4_INDEXER_GRAPH_PROGRESS_AFFINITY_VERIFIED = 1U << 4,
  SPARK_TP4_INDEXER_GRAPH_OVERFLOW_FATAL = 1U << 5,
};

spark_tp4_indexer_graph_handle spark_tp4_indexer_graph_create(
    const spark_tp4_indexer_graph_config* config, char* error,
    size_t error_bytes);

/*
 * Adds contiguous INT32 [Q,2,2048] -> [4,Q,2,2048], Q in [1,40], to an
 * active CUDA stream capture. Stable tensor addresses and the handle must
 * outlive every replay.
 */
int spark_tp4_indexer_capture_allgather(
    spark_tp4_indexer_graph_handle handle, const void* input,
    void* output, uint32_t q, void* cuda_stream, char* error,
    size_t error_bytes);

int spark_tp4_indexer_get_graph_status(
    spark_tp4_indexer_graph_handle handle,
    spark_tp4_indexer_graph_status* status, size_t status_bytes,
    char* error, size_t error_bytes);

void spark_tp4_indexer_graph_destroy(
    spark_tp4_indexer_graph_handle handle);

#ifdef __cplusplus
}
#endif
