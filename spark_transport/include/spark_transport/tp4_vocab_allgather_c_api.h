#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spark_tp4_vocab_allgather_config {
  uint32_t rank;
  const char* peer0;
  const char* peer1;
  const char* device0;
  const char* device1;
  uint8_t gid0;
  uint8_t gid1;
  uint16_t control_port0;
  uint16_t control_port1;
} spark_tp4_vocab_allgather_config;

typedef struct spark_tp4_vocab_graph_config {
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
} spark_tp4_vocab_graph_config;

typedef void* spark_tp4_vocab_allgather_handle;

typedef struct spark_tp4_vocab_graph_status {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t captured_nodes;
  uint64_t published_sequence;
  uint64_t consumed_sequence;
  uint64_t completed_sequence;
  uint64_t overflow_sequence;
  uint32_t graph_submit_cpu_plus_one;
  uint32_t graph_progress_cpu_plus_one;
} spark_tp4_vocab_graph_status;

enum {
  SPARK_TP4_VOCAB_GRAPH_CAPTURE_CONFIGURED = 1U << 0,
  SPARK_TP4_VOCAB_GRAPH_POLLING_ENABLED = 1U << 1,
  SPARK_TP4_VOCAB_GRAPH_HOST_NATIVE_ATOMICS = 1U << 2,
  SPARK_TP4_VOCAB_GRAPH_SUBMIT_AFFINITY_VERIFIED = 1U << 3,
  SPARK_TP4_VOCAB_GRAPH_PROGRESS_AFFINITY_VERIFIED = 1U << 4,
  SPARK_TP4_VOCAB_GRAPH_OVERFLOW_FATAL = 1U << 5,
};

/*
 * Creates one dynamic-Q session for a fixed per-rank vocabulary shard.
 * Each call accepts BF16 [Q, 38720], Q in [1, 40]; 38,720 is the per-rank
 * shard width of the GLM-5.2 reference deployment's vocabulary.
 */
spark_tp4_vocab_allgather_handle spark_tp4_vocab_allgather_create(
    const spark_tp4_vocab_allgather_config* config, char* error,
    size_t error_bytes);

spark_tp4_vocab_allgather_handle spark_tp4_vocab_graph_create(
    const spark_tp4_vocab_graph_config* config, char* error,
    size_t error_bytes);

/*
 * Writes contiguous token-major BF16 [Q, 154880]. Rank shards appear in
 * ascending rank order within every query row.
 */
int spark_tp4_vocab_allgather(
    spark_tp4_vocab_allgather_handle handle, const void* input,
    void* output, uint32_t query_rows, void* cuda_stream, char* error,
    size_t error_bytes);

/*
 * Adds one fixed-Q vocabulary gather to an active CUDA stream capture.
 * The graph-configured handle, stable tensor addresses, and capture stream
 * must outlive all graph replays.
 */
int spark_tp4_vocab_capture_allgather(
    spark_tp4_vocab_allgather_handle handle, const void* input,
    void* output, uint32_t query_rows, void* cuda_stream, char* error,
    size_t error_bytes);

int spark_tp4_vocab_get_graph_status(
    spark_tp4_vocab_allgather_handle handle,
    spark_tp4_vocab_graph_status* status, size_t status_bytes,
    char* error, size_t error_bytes);

void spark_tp4_vocab_allgather_destroy(
    spark_tp4_vocab_allgather_handle handle);

#ifdef __cplusplus
}
#endif
