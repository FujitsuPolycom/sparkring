/* The SIRCL session engine of one communicator: arena, native progress
 * thread, kernel pack, stream ordering, capture rules and receipts.
 *
 * A communicator of W ranks owns one SIRCL ring session: a pinned host arena
 * that the GPU addresses and every RDMA device registers, device counters,
 * one reliable-connected queue pair per (peer, lane), and SIRCL's native
 * progress thread, which posts RDMA writes when a kernel rings the doorbell in
 * the arena's command ring. Collectives are kernel launches on the caller's
 * stream; the host never waits for the network. */
#ifndef SCCL_ENGINE_H
#define SCCL_ENGINE_H
#include <stdatomic.h>
#include <stddef.h>

#include "bootstrap.h"
#include "cuda_api.h"
#include "nccl.h"

typedef struct sccl_engine sccl_engine;

/* The calling thread's current CUDA context, or NULL. Communicators bind to the
 * context current when they are created. */
sccl_CUcontext sccl_engine_current_context(void);

/* Collective setup over the communicator's bootstrap (every rank calls it).
 * `position` (0-63) identifies this process in the route map, the same in every
 * communicator it joins; `ctx` is the creating thread's context. Returns an
 * ncclResult_t; on failure `err` names the failing ranks and reasons. */
int sccl_engine_create(sccl_bootstrap *bootstrap, int nranks, int rank, int position, sccl_CUcontext ctx,
                       const atomic_int *cancelled, sccl_engine **out, char *err, size_t err_len);
/* This process's position in the route map of the communicator. */
int sccl_engine_position(const sccl_engine *engine);
/* Reduce `count` elements of `datatype` from every rank with `op` into `recvbuff`
 * on every rank, enqueued on `stream`: float16, bfloat16 and float32 sums in the
 * transport kernels, every other datatype and built-in op by an all-gather and
 * the fold pack. */
ncclResult_t sccl_engine_allreduce(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream, char *err,
                                   size_t err_len);
/* Concatenate every rank's `count` elements in rank order into `recvbuff`. */
ncclResult_t sccl_engine_allgather(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, cudaStream_t stream, char *err, size_t err_len);
/* Chunk `rank` (of `count` elements) of the reduction of every rank's W chunks into `recvbuff`. */
ncclResult_t sccl_engine_reducescatter(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream, char *err,
                                       size_t err_len);
/* The reduction into `recvbuff` on rank `root` only (an all-reduce whose result other ranks drop). */
ncclResult_t sccl_engine_reduce(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op, int root, cudaStream_t stream, char *err,
                                size_t err_len);
/* Rank `root`'s `count` elements into every rank's `recvbuff` (an all-gather of the root's tiles). */
ncclResult_t sccl_engine_broadcast(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, int root, cudaStream_t stream, char *err, size_t err_len);
/* Chunk j (of `count` elements) of `sendbuff` to rank j; rank i's chunk for this rank at recvbuff + i * count. */
ncclResult_t sccl_engine_alltoall(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                  ncclDataType_t datatype, cudaStream_t stream, char *err, size_t err_len);
/* Every rank's `count` elements into `recvbuff` on rank `root` only, in rank order. */
ncclResult_t sccl_engine_gather(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, int root, cudaStream_t stream, char *err, size_t err_len);
/* Chunk j (of `count` elements) of rank `root`'s `sendbuff` into rank j's `recvbuff`. */
ncclResult_t sccl_engine_scatter(sccl_engine *engine, const void *sendbuff, void *recvbuff, size_t count,
                                 ncclDataType_t datatype, int root, cudaStream_t stream, char *err, size_t err_len);
/* One ncclSend (send = 1) or ncclRecv of `bytes` bytes with rank `peer`, as queued by the API layer. */
typedef struct {
  int send, peer;
  const void *buff;
  size_t bytes;
  cudaStream_t stream;
} sccl_p2p_op;
/* The point-to-point calls of one group (or one call outside a group) on this communicator, in issue
 * order, on one stream. Between ranks they are carried on two-rank communicators: the k-th send to the
 * peer and the k-th receive from it form exchange k. A rank's sends to itself match its receives from
 * itself in order and are local copies. */
ncclResult_t sccl_engine_p2p(sccl_engine *engine, const sccl_p2p_op *ops, unsigned count, char *err,
                             size_t err_len);
/* Bytes per element of an NCCL datatype; 0 for an invalid one. */
size_t sccl_engine_type_size(ncclDataType_t datatype);
/* ncclMemAlloc and ncclMemFree: device memory of the calling thread's current CUDA context (cuMemAlloc),
 * an ordinary device buffer for every collective, eager and in CUDA graphs. */
ncclResult_t sccl_engine_mem_alloc(void **ptr, size_t size, char *err, size_t len);
ncclResult_t sccl_engine_mem_free(void *ptr, char *err, size_t len);
/* ncclSuccess, or the asynchronous error with its description. Lock-free. */
ncclResult_t sccl_engine_async_error(sccl_engine *engine, char *message, size_t message_len);
/* Wait until every enqueued operation finished. */
ncclResult_t sccl_engine_finalize(sccl_engine *engine);
/* Stop the progress thread and free the session. With `abort` set, resources
 * a still-running kernel may touch are released only when its stream is idle;
 * otherwise they are left until process exit. */
/* Frees the engine; ncclSystemError (and `err`) when releasing the transport failed: its memory then stays
 * allocated. */
ncclResult_t sccl_engine_destroy(sccl_engine *engine, int abort, char *err, size_t len);
/* Teardown in steps, for ncclCommDestroy and ncclCommFinalize (api.c runs the rounds between them):
 * waits until this rank's enqueued work completed; */
ncclResult_t sccl_engine_teardown_sync(sccl_engine *e);
/* writes the receipt and link dump, then stops the native progress thread (it posts nothing more; the
 * queue pairs and registrations stay until sccl_engine_destroy); */
void sccl_engine_teardown_stop(sccl_engine *e);
/* the session's wait limit in the current regime, in milliseconds. */
int sccl_engine_wait_limit_ms(const sccl_engine *e);
/* Stop the native threads only (library unload; CUDA may be gone). */
void sccl_engine_shutdown(sccl_engine *engine);
/* Stop the fail-stop watcher (LIBSIRCL_FAIL_STOP), first at library unload; later communicators are not
 * watched. */
void sccl_engine_fail_stop_shutdown(void);
int sccl_engine_device(const sccl_engine *engine);
/* "startup" or "serving": the flag-wait limit of later launches. */
ncclResult_t sccl_engine_set_wait_regime(sccl_engine *engine, const char *regime);
/* The communicator's receipt (JSON) into `out`; returns the bytes it needs. */
size_t sccl_engine_receipt(sccl_engine *engine, char *out, size_t out_len);

#endif
