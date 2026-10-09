/* Control interface of the shared-memory verbs stand-in (shm_verbs.c).
 *
 * The stand-in implements the libibverbs subset that SIRCL's native proxy
 * uses (the declarations of fake_verbs/infiniband/verbs.h, renamed sccl_emu_*
 * by proxy_names.h) across processes of one host, so several processes, or
 * several threads of one process, act as the ranks of a group on one GPU.
 *
 * Fabric. One POSIX shared-memory segment (SIRCL_EMU_FABRIC, default
 * /sircl-emu-<uid>) holds the registry every process shares: devices (name,
 * GID, owning process), memory regions (keys, owner's address and length,
 * backing segment and offset) and queue pairs (number, device, state,
 * destination). Delivery is ideal: a write goes to the device whose GID its
 * queue pair names, and is applied only when the destination queue pair
 * exists and is connected back to the sender, as reliable connected queue
 * pairs require.
 *
 * Memory. A region that a peer may write must lie in a segment allocated by
 * sccl_emu_segment_alloc (POSIX shared memory mapped in its owner), named
 * /sircl-emu-seg-<pid>-<start>-<serial> (<start>: the process's start time, so
 * processes of different PID namespaces sharing /dev/shm never share a name; a
 * name that exists already is skipped, never unlinked). A writer
 * maps the destination segment on first use and copies into it; the GPU of the
 * owning process reads the same pages through its host registration.
 *
 * Execution. Each process runs one executor thread while it has queue pairs.
 * It executes the work requests posted on its own queue pairs, one at a time,
 * in a seeded random interleaving across queue pairs that keeps each queue
 * pair's order (the only order RC gives). Payload bytes are copied before the
 * write retires, with a full fence before and after; 4-byte aligned writes
 * (flag lines) are release stores. A signaled or failed write leaves a
 * completion on the sender's completion queue.
 *
 * Diagnostics. sccl_emu_report() describes the queue pairs of this process and
 * the writes that failed; the library writes it into its link dumps
 * (LIBSIRCL_LINK_DUMP).
 *
 * Settings read at attach: SIRCL_EMU_FABRIC (segment name), SIRCL_EMU_SEED
 * (interleaving seed, default 1), SIRCL_EMU_LATENCY_NS (each write executes no
 * earlier than this long after it was posted; default 0).
 *
 * Status: research-only test infrastructure; it carries no traffic between
 * hosts and models no cable, relay or failure. */
#ifndef SCCL_SHM_VERBS_H
#define SCCL_SHM_VERBS_H
#include <stddef.h>
#include <stdint.h>

/* Open or create the fabric segment. 0 on success; otherwise `err` says why. */
int sccl_emu_attach(char *err, size_t err_len);
/* A device owned by this process, for ibv_get_device_list. -1 on failure. */
int sccl_emu_add_device(const char *name, const uint8_t gid[16]);
/* Zeroed shared memory of at least `bytes`, page aligned; NULL on failure. */
void *sccl_emu_segment_alloc(size_t bytes, char *err, size_t err_len);
void sccl_emu_segment_free(void *base);
/* Writes executed by this process's executor. */
uint64_t sccl_emu_executed(void);
/* Diagnostics as JSON (at most `len` bytes into `out`, always terminated; returns the bytes needed): the
 * writes executed, the completions not delivered (completion queue full: cq_overflows; queue pair destroyed
 * while its write executed: replaced_drops), every local queue pair (number, destination, state, requests
 * queued, writes posted, executed and completions delivered, errored), and the newest 64 writes that did not
 * succeed: CLOCK_REALTIME nanoseconds, queue pairs, the process owning the destination device and whether
 * it is alive, work request id, rkey, remote address, length, status, the resolution check that failed,
 * and whether the completion was delivered. */
size_t sccl_emu_report(char *out, size_t len);
/* SIRCL_EMU_FAIL_DEREG=1, a test setting, makes every ibv_dereg_mr fail with EBUSY and keep its region
 * registered. */

#endif
