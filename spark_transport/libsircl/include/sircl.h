#ifndef LIBSIRCL_H
#define LIBSIRCL_H
#include <stddef.h>
#include "nccl.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Stable capability description (JSON); no communicator or hardware access. */
const char *sirclGetInfo(void);
/* The flag-wait limit of a communicator's later launches: "startup"
 * (SIRCL_STARTUP_WAIT_S, default 600 s: peers may lag while they compile,
 * warm up or capture graphs) or "serving" (SIRCL_SERVING_WAIT_S, default 20 s:
 * a longer lag means a failed peer). Applies to eager launches and graph
 * replays from the next launch on. Every rank sets the same regime. */
ncclResult_t sirclSetWaitRegime(ncclComm_t comm, const char *regime);
/* The communicator's receipt (JSON, schema libsircl-receipt/v1): what carried
 * each call, refusals and native counters. Writes at most `length` bytes
 * including the terminator; *needed receives the size this call's receipt
 * needs. The receipt is made at each call from live counters, so it may grow
 * between a call that sizes it and the next: the output is whole only when
 * *needed <= length, and a caller retries with a larger buffer otherwise. */
ncclResult_t sirclGetReceipt(ncclComm_t comm, char *out, size_t length, size_t *needed);
/* SIRCL-EP: GPU-driven dynamic variable-length MoE token dispatch and combine over
 * SIRCL's point-to-point channels. Zero CPU synchronization, zero wasted RDMA bytes,
 * and CUDA-graph capturable. Requires a communicator of 3 to 8 ranks created with
 * LIBSIRCL_P2P_CHANNELS=on and channels to every peer. */
size_t sirclEpWorkspaceBytes(int world, int max_tokens, size_t hidden_bytes);
ncclResult_t sirclEpDispatch(const void *hidden_in, const int *topk_ids_in, const float *topk_weights_in,
                             void *hidden_out, int *topk_ids_out, float *topk_weights_out, int *recv_counts_out,
                             void *workspace, size_t workspace_bytes, int num_tokens, int max_tokens,
                             size_t hidden_bytes, int top_k, int num_experts, ncclComm_t comm, cudaStream_t stream);
ncclResult_t sirclEpCombine(const void *expert_out_in, void *hidden_out, void *workspace, size_t workspace_bytes,
                            int num_tokens, int max_tokens, size_t hidden_dim, ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
