#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spark_tp4_config {
  uint32_t rank;
  const char* peer0;
  const char* peer1;
  const char* device0;
  const char* device1;
  uint8_t gid0;
  uint8_t gid1;
  uint16_t control_port0;
  uint16_t control_port1;
  size_t payload_bytes;
  /*
   * Zero disables the graph-affinity contract. A configured CPU N is encoded
   * as N + 1 so zero-initialized legacy callers remain ABI-source compatible.
   * The two fields must be configured together and name distinct CPUs.
   */
  uint32_t graph_submit_cpu_plus_one;
  uint32_t graph_progress_cpu_plus_one;
} spark_tp4_config;

typedef void* spark_tp4_handle;

enum {
  SPARK_TP4_GRAPH_STATUS_CAPTURE_CONFIGURED = 1U << 0,
  SPARK_TP4_GRAPH_STATUS_POLLING_ENABLED = 1U << 1,
  SPARK_TP4_GRAPH_STATUS_HOST_NATIVE_ATOMICS = 1U << 2,
  SPARK_TP4_GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED = 1U << 3,
  SPARK_TP4_GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED = 1U << 4,
  /*
   * An overflow is a fatal ordering invariant. Other asynchronous transport
   * failures abort immediately and therefore cannot be polled in-process.
   */
  SPARK_TP4_GRAPH_STATUS_OVERFLOW_FATAL = 1U << 5,
  /*
   * The graph progress thread never calls the OS scheduler yield primitive.
   * This is valid only with an exclusively affined progress CPU.
   */
  SPARK_TP4_GRAPH_STATUS_DEDICATED_SPIN = 1U << 6,
};

typedef struct spark_tp4_graph_status {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t captured_nodes;
  uint64_t published_sequence;
  uint64_t consumed_sequence;
  uint64_t completed_sequence;
  uint64_t overflow_sequence;
  uint32_t graph_submit_cpu_plus_one;
  uint32_t graph_progress_cpu_plus_one;
} spark_tp4_graph_status;

spark_tp4_handle spark_tp4_create(const spark_tp4_config* config,
                                  char* error, size_t error_bytes);

int spark_tp4_all_reduce(spark_tp4_handle handle, const void* input,
                         void* output, void* cuda_stream, char* error,
                         size_t error_bytes);

/*
 * Adds one exact contiguous BF16 [q, 6144] all-reduce to an active CUDA
 * stream capture. q must be in [1, 512], and the handle's payload capacity
 * must be at least q * 12,288 bytes. One maximum-capacity handle serves
 * every supported mixed/adaptive concurrency bucket.
 */
int spark_tp4_capture_all_reduce(
    spark_tp4_handle handle, const void* input, void* output, uint32_t q,
    void* cuda_stream, char* error, size_t error_bytes);

/*
 * Adds a Q1 BF16 [1, 6144] all-reduce to an active CUDA stream capture.
 * The graph is bound to the supplied tensor addresses, must replay on the
 * capture stream, and the handle must outlive all graph replays. The function
 * may be called repeatedly on that stream before the first replay so one
 * session can serve every Q1 node in a captured model graph. Replay uses
 * device-published commands and requires host-native GPU atomics.
 * This compatibility entrypoint is equivalent to
 * spark_tp4_capture_all_reduce(..., 1, ...).
 */
int spark_tp4_capture_q1_all_reduce(
    spark_tp4_handle handle, const void* input, void* output,
    void* cuda_stream, char* error, size_t error_bytes);

/*
 * Copies a nonblocking read-only snapshot of graph definition/replay progress.
 * Counters may advance while they are copied, so callers that require a
 * quiescent equality gate should sample after request synchronization. A
 * completed_sequence increase proves actual graph replay transport progress;
 * captured_nodes alone proves only graph definition.
 */
int spark_tp4_get_graph_status(
    spark_tp4_handle handle, spark_tp4_graph_status* status,
    size_t status_bytes, char* error, size_t error_bytes);

void spark_tp4_destroy(spark_tp4_handle handle);

/*
 * All-gather deliberately has a distinct configuration and handle type.
 * This keeps the ABI honest: an all-reduce handle cannot accidentally be
 * passed to an all-gather operation (or destroyed through the wrong path).
 */
typedef struct spark_tp4_allgather_config {
  uint32_t rank;
  const char* peer0;
  const char* peer1;
  const char* device0;
  const char* device1;
  uint8_t gid0;
  uint8_t gid1;
  uint16_t control_port0;
  uint16_t control_port1;
  size_t input_bytes;
} spark_tp4_allgather_config;

typedef void* spark_tp4_allgather_handle;

spark_tp4_allgather_handle spark_tp4_allgather_create(
    const spark_tp4_allgather_config* config, char* error,
    size_t error_bytes);

/*
 * Gathers one input_bytes segment from each rank. Output must hold
 * 4 * input_bytes bytes and is laid out in rank order.
 */
int spark_tp4_allgather(spark_tp4_allgather_handle handle,
                        const void* input, void* output, void* cuda_stream,
                        char* error, size_t error_bytes);

void spark_tp4_allgather_destroy(spark_tp4_allgather_handle handle);

/*
 * GLM-5.2 DCP query all-gather is a distinct dynamic-Q operation. One
 * persistent handle accepts Q in [1, 40], reads contiguous BF16
 * [Q, 16, 576], and writes contiguous BF16 [Q, 64, 576] with rank order
 * concatenated on the head axis.
 */
typedef struct spark_tp4_dcp_config {
  uint32_t rank;
  const char* peer0;
  const char* peer1;
  const char* device0;
  const char* device1;
  uint8_t gid0;
  uint8_t gid1;
  uint16_t control_port0;
  uint16_t control_port1;
} spark_tp4_dcp_config;

typedef void* spark_tp4_dcp_handle;

spark_tp4_dcp_handle spark_tp4_dcp_create(
    const spark_tp4_dcp_config* config, char* error, size_t error_bytes);

/*
 * Graph DCP uses a separate constructor so legacy eager callers cannot be
 * silently reinterpreted through a larger configuration ABI. One graph
 * handle carries both query and fused LSE/output families on a shared ordered
 * command ring.
 */
typedef struct spark_tp4_dcp_graph_config {
  uint32_t rank;
  const char* peer0;
  const char* peer1;
  const char* device0;
  const char* device1;
  uint8_t gid0;
  uint8_t gid1;
  uint16_t control_port0;
  uint16_t control_port1;
  uint32_t graph_submit_cpu_plus_one;
  uint32_t graph_progress_cpu_plus_one;
} spark_tp4_dcp_graph_config;

typedef struct spark_tp4_dcp_graph_status {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t captured_nodes;
  uint64_t captured_query_nodes;
  uint64_t captured_combine_nodes;
  uint64_t published_sequence;
  uint64_t consumed_sequence;
  uint64_t completed_sequence;
  uint64_t overflow_sequence;
  uint32_t graph_submit_cpu_plus_one;
  uint32_t graph_progress_cpu_plus_one;
} spark_tp4_dcp_graph_status;

spark_tp4_dcp_handle spark_tp4_dcp_graph_create(
    const spark_tp4_dcp_graph_config* config, char* error,
    size_t error_bytes);

int spark_tp4_dcp_query_all_gather(
    spark_tp4_dcp_handle handle, const void* input, void* output,
    uint32_t q, void* cuda_stream, char* error, size_t error_bytes);

int spark_tp4_dcp_capture_query_all_gather(
    spark_tp4_dcp_handle handle, const void* input, void* output,
    uint32_t q, void* cuda_stream, char* error, size_t error_bytes);

/*
 * Combines normalized local attention output and natural-log softmax state
 * across four DCP ranks. output_bf16 has logical shape [Q, 64, D], where
 * D is 256 or 512, with exact element strides query_stride/head_stride/1;
 * lse_fp32 is contiguous FP32 [Q, 64]. Outputs are this rank's contiguous
 * token-major BF16 [Q, 16, D] and FP32 [Q, 16].
 */
int spark_tp4_dcp_combine(
    spark_tp4_dcp_handle handle, const void* output_bf16,
    const void* lse_fp32, void* reduced_output_bf16,
    void* reduced_lse_fp32, uint32_t q, uint32_t head_dimension,
    uint32_t query_stride, uint32_t head_stride, void* cuda_stream,
    char* error, size_t error_bytes);

int spark_tp4_dcp_capture_combine(
    spark_tp4_dcp_handle handle, const void* output_bf16,
    const void* lse_fp32, void* reduced_output_bf16,
    void* reduced_lse_fp32, uint32_t q, uint32_t head_dimension,
    uint32_t query_stride, uint32_t head_stride, void* cuda_stream,
    char* error, size_t error_bytes);

int spark_tp4_dcp_get_graph_status(
    spark_tp4_dcp_handle handle, spark_tp4_dcp_graph_status* status,
    size_t status_bytes, char* error, size_t error_bytes);

void spark_tp4_dcp_destroy(spark_tp4_dcp_handle handle);

#ifdef __cplusplus
}
#endif
