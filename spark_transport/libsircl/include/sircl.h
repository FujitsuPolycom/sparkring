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
 * including the terminator; *needed receives the size it needs. */
ncclResult_t sirclGetReceipt(ncclComm_t comm, char *out, size_t length, size_t *needed);
#ifdef __cplusplus
}
#endif
#endif
