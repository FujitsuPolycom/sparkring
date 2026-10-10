#ifndef SCCL_BOOTSTRAP_H
#define SCCL_BOOTSTRAP_H
#include <stdatomic.h>

/* TCP rendezvous of a communicator's ranks. The root (the process that made
 * the ID) listens on 127.0.0.1, or on the IPv4 address that
 * SIRCL_BOOTSTRAP_ADDR names or that an interface named by
 * SIRCL_BOOTSTRAP_IFNAME or NCCL_SOCKET_IFNAME carries; a joining rank
 * contacts a non-loopback root only when one of these is set. No GPU is
 * accessed. Return values match ncclResult_t. A successful join keeps a
 * connection to the rendezvous broker until close; all-gather rounds run over
 * it. */
typedef struct sccl_bootstrap sccl_bootstrap;

/* Claims the calling process on first use. Fork after API use requires exec:
 * an inherited child returns zero and must avoid every process-state lock. */
int sccl_bootstrap_process_valid(void);
int sccl_bootstrap_id(unsigned char id[128]);
int sccl_bootstrap_join(const unsigned char id[128], int nranks, int rank,
                        sccl_bootstrap **out);
int sccl_bootstrap_join_cancel(const unsigned char id[128], int nranks, int rank,
                               sccl_bootstrap **out,
                               const atomic_int *cancelled);
void sccl_bootstrap_close(sccl_bootstrap *handle);
/* One collective round over the rendezvous broker: every rank contributes
 * `bytes` (at most 1 MiB); on success *out holds every rank's contribution
 * back to back in rank order (caller frees) and lengths[r] its size. Rounds
 * run in the same order on every rank. A rank that leaves after the group
 * completed disconnects every rank, so later rounds fail at once. */
int sccl_bootstrap_allgather(sccl_bootstrap *handle, const void *data, unsigned bytes,
                             void **out, unsigned *lengths, const atomic_int *cancelled);
/* The same round, bounded by `timeout_ms` (1 or more) instead of LIBSIRCL_SETUP_TIMEOUT_MS. */
int sccl_bootstrap_allgather_within(sccl_bootstrap *handle, const void *data, unsigned bytes, void **out,
                                    unsigned *lengths, int timeout_ms, const atomic_int *cancelled);
int sccl_bootstrap_fd(const sccl_bootstrap *handle);
const char *sccl_bootstrap_error(void);

#endif
