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

/*
 * Versioned graph-row geometry. The embedded v1 configuration preserves the
 * established eager and GLM graph ABI. New callers set struct_size, then use
 * elements_per_row=4096 and bytes_per_row=8192 for DeepSeek BF16 tensors.
 * bytes_per_row must equal elements_per_row * 2.
 */
typedef struct spark_tp4_config_v2 {
  uint32_t struct_size;
  spark_tp4_config base;
  uint32_t elements_per_row;
  uint32_t bytes_per_row;
} spark_tp4_config_v2;

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
  /* The graph session uses two generation-tagged payload slots per edge. */
  SPARK_TP4_GRAPH_STATUS_TWO_SLOT_DEFERRED_ACK = 1U << 7,
  /* Every captured node uses the fixed 64 KiB split-kernel graph. */
  SPARK_TP4_GRAPH_STATUS_SPLIT_64K = 1U << 8,
  /* Captured nodes select fused or split kernels from their active Q. */
  SPARK_TP4_GRAPH_STATUS_TIERED_64K = 1U << 9,
  /* The session counter-rotates tensor halves across both physical links. */
  SPARK_TP4_GRAPH_STATUS_DUAL_PORT_STRIPED = 1U << 10,
  /* Replay uses a device ticket and round-0 payload doorbell notification. */
  SPARK_TP4_GRAPH_STATUS_DIRECT_DOORBELL = 1U << 11,
};

enum {
  SPARK_TP4_ALLREDUCE_PROTOCOL_SERIAL_ACK = 0,
  SPARK_TP4_ALLREDUCE_PROTOCOL_TWO_SLOT_DEFERRED_ACK = 1,
};

enum {
  SPARK_TP4_GRAPH_KERNEL_FUSED = 0,
  SPARK_TP4_GRAPH_KERNEL_SPLIT_64K = 1,
  SPARK_TP4_GRAPH_KERNEL_TIERED_64K = 2,
};

enum {
  SPARK_TP4_WIRE_SCHEDULE_SEQUENTIAL = 0,
  SPARK_TP4_WIRE_SCHEDULE_DUAL_PORT_STRIPED = 1,
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

enum {
  SPARK_TP4_HEALTHY = 1U << 0,
  SPARK_TP4_HEALTH_POISONED = 1U << 1,
  SPARK_TP4_HEALTH_PROGRESS_THREAD_RUNNING = 1U << 2,
  SPARK_TP4_HEALTH_STOPPING = 1U << 3,
};

typedef struct spark_tp4_health_status {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t submitted_sequence;
  uint64_t completed_sequence;
  uint64_t failing_sequence;
  int32_t error_code;
  int32_t failing_stage;
  int32_t failing_rail;
  int32_t failing_peer;
} spark_tp4_health_status;

spark_tp4_handle spark_tp4_create(const spark_tp4_config* config,
                                  char* error, size_t error_bytes);

/*
 * Selects a protocol without extending the unversioned spark_tp4_config ABI.
 * The graph kernel remains FUSED. spark_tp4_create remains equivalent to
 * SERIAL_ACK plus FUSED. The two-slot protocol is experimental and accepted
 * only for graph-only sessions.
 */
spark_tp4_handle spark_tp4_create_with_protocol(
    const spark_tp4_config* config, uint32_t protocol, char* error,
    size_t error_bytes);

/*
 * Selects protocol and graph-kernel strategy without changing
 * spark_tp4_config. Invalid scalar values fail before session construction.
 * SPLIT_64K and TIERED_64K are graph-only research strategies. Their selected
 * strategy is attested by the corresponding mutually exclusive graph-status
 * flag; FUSED sets neither strategy flag.
 */
spark_tp4_handle spark_tp4_create_with_protocol_and_graph_kernel(
    const spark_tp4_config* config, uint32_t protocol,
    uint32_t graph_kernel, char* error, size_t error_bytes);

/*
 * Additive constructor for a versioned wire schedule. Existing constructors
 * remain equivalent to SEQUENTIAL. DUAL_PORT_STRIPED is graph-only and the
 * native session rejects any incompatible protocol, graph kernel, affinity,
 * or payload capacity before endpoint setup.
 */
spark_tp4_handle
spark_tp4_create_with_protocol_graph_kernel_and_schedule(
    const spark_tp4_config* config, uint32_t protocol,
    uint32_t graph_kernel, uint32_t wire_schedule, char* error,
    size_t error_bytes);

/* Additive constructor for a non-GLM graph row layout. */
spark_tp4_handle spark_tp4_create_v2(
    const spark_tp4_config_v2* config, uint32_t protocol,
    uint32_t graph_kernel, uint32_t wire_schedule, char* error,
    size_t error_bytes);

int spark_tp4_all_reduce(spark_tp4_handle handle, const void* input,
                         void* output, void* cuda_stream, char* error,
                         size_t error_bytes);

/*
 * Adds one exact contiguous BF16 [q, elements_per_row] all-reduce to an active
 * CUDA stream capture. q must be in [1, 512], and the handle's payload
 * capacity must be at least q * bytes_per_row. One maximum-capacity handle serves
 * every supported mixed/adaptive concurrency bucket.
 */
int spark_tp4_capture_all_reduce(
    spark_tp4_handle handle, const void* input, void* output, uint32_t q,
    void* cuda_stream, char* error, size_t error_bytes);

/*
 * Adds a Q1 BF16 [1, elements_per_row] all-reduce to capture.
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
 * captured_nodes alone proves only graph definition. When
 * TWO_SLOT_DEFERRED_ACK is set, completed_sequence means the progress worker
 * observed output consumption and posted both credits. Device output can be
 * ready earlier; physical payload-slot retirement remains credit-gated.
 */
int spark_tp4_get_graph_status(
    spark_tp4_handle handle, spark_tp4_graph_status* status,
    size_t status_bytes, char* error, size_t error_bytes);

/* Copies host state only. This function never synchronizes a CUDA stream. */
int spark_tp4_get_health_status(
    spark_tp4_handle handle, spark_tp4_health_status* status,
    size_t status_bytes, char* error, size_t error_bytes);

void spark_tp4_destroy(spark_tp4_handle handle);


#ifdef __cplusplus
}
#endif
