/* Independent NCCL C interface implementation: API layer, communicator
 * lifecycle and dispatch to the SIRCL session engine (engine.c). */
#define _POSIX_C_SOURCE 200809L
#include <strings.h>
#include "env_names.h"
#ifndef LIBSIRCL_VERSION
#error "LIBSIRCL_VERSION is set by the build from the VERSION file"
#endif
#include "internal.h"
#include "bootstrap.h"
#include "engine.h"
#include "sircl.h"
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum comm_state { INIT, READY, FINALIZED, REVOKED, FAILED };
struct ncclComm {
  struct ncclComm *next;
  int nranks, rank, device, blocking, refs, worker_started;
  int bootstrap_only;            /* LIBSIRCL_BOOTSTRAP_ONLY: a logical CPU communicator */
  int position;                  /* this process's route-map position (LIBSIRCL_POSITION, or a parent's) */
  unsigned char id[128];
  pthread_t worker;
  sccl_bootstrap *bootstrap;
  sccl_CUcontext ctx;            /* the creating thread's CUDA context */
  sccl_engine *engine;
  char error_message[1024];
  pthread_mutex_t error_lock;    /* an error published after the status (publish_error) with its text */
  int quiesced;                  /* the close began (ncclCommFinalize, or ncclCommDestroy) */
  int close_result;              /* its result, returned again by a repeated ncclCommFinalize or destroy */
  uint64_t serial;               /* creation number in this process: tells a reused address apart */
  atomic_int state, error, grouped, cancelled;
};
static pthread_mutex_t registry_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t registry_idle = PTHREAD_COND_INITIALIZER;
static struct ncclComm *registry;
static _Thread_local char last_error[512];
static _Thread_local unsigned group_depth;
static _Thread_local ncclResult_t group_error;
static _Thread_local struct ncclComm *pending[64];
static _Thread_local unsigned pending_count;
/* Point-to-point calls issued inside the calling thread's group, each holding a reference on its
 * communicator, carried out at the outermost ncclGroupEnd. */
typedef struct {
  struct ncclComm *comm;         /* no reference held; validated with `serial` at ncclGroupEnd */
  uint64_t serial;
  sccl_p2p_op op;
} queued_p2p;
static _Thread_local queued_p2p *p2p_queue;
static _Thread_local unsigned p2p_count, p2p_capacity;

/* Make `result` the communicator's terminal error with `message` as its text, unless it already has one;
 * 1 when this call set it. The text is written under the error lock, and readers copy it under the same
 * lock (read_error). */
static int publish_error(struct ncclComm *comm, ncclResult_t result, const char *message) {
  pthread_mutex_lock(&comm->error_lock);
  int expected = ncclSuccess;
  int set = atomic_compare_exchange_strong(&comm->error, &expected, result);
  if (set) snprintf(comm->error_message, sizeof(comm->error_message), "%s", message);
  pthread_mutex_unlock(&comm->error_lock);
  return set;
}
static void read_error(struct ncclComm *comm, char *out, size_t len) {
  pthread_mutex_lock(&comm->error_lock);
  snprintf(out, len, "%s", comm->error_message);
  pthread_mutex_unlock(&comm->error_lock);
}
static ncclResult_t fail(ncclResult_t result, const char *message) {
  snprintf(last_error, sizeof(last_error), "%s", message);
  if (group_depth && group_error == ncclSuccess) group_error = result;
  return result;
}
/* A refused collective or point-to-point call: the error and its text for this thread. An invalid argument
 * leaves the calling thread's group as it was (the call did nothing, and the work queued before it still
 * runs at ncclGroupEnd); any other error also becomes the group's result. */
static ncclResult_t refuse(ncclResult_t result, const char *message) {
  if (result != ncclInvalidArgument) return fail(result, message);
  snprintf(last_error, sizeof(last_error), "%s", message);
  return result;
}
#define CHECK_PROCESS() do { \
  if (!sccl_bootstrap_process_valid()) \
    return fail(ncclInvalidUsage, "fork after libsircl API use requires exec before further calls"); \
} while (0)
ncclResult_t sccl_unsupported(const char *name) {
  CHECK_PROCESS();
  static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;
  static const char *seen[128];
  static unsigned nseen;
  char message[384];
  pthread_mutex_lock(&log_lock);
  unsigned i;
  for (i = 0; i < nseen; ++i) if (!strcmp(seen[i], name)) break;
  if (i == nseen && nseen < 128) {
    seen[nseen++] = name;
    fprintf(stderr, "libsircl: %s unsupported by this build\n", name);
  }
  pthread_mutex_unlock(&log_lock);
  snprintf(message, sizeof(message), "%s: unsupported by this libsircl build", name);
  return fail(ncclInvalidUsage, message);
}
const char *sirclGetInfo(void) {
  return "{\"library\":\"libsircl\",\"version\":\"" LIBSIRCL_VERSION "\",\"collectives\":{\"all_reduce\":"
         "\"every datatype\",\"reduce_scatter\":\"every datatype\",\"reduce\":\"every datatype\","
         "\"ops\":[\"sum\",\"prod\",\"max\",\"min\",\"avg\"],\"transport_reductions\":[\"float16/sum\","
         "\"bfloat16/sum\",\"float32/sum\"],\"all_gather\":\"every datatype\",\"broadcast\":"
         "\"every datatype\",\"all_to_all\":\"every datatype\",\"gather\":\"every datatype\",\"scatter\":"
         "\"every datatype\",\"send_recv\":\"two-rank communicators, and a rank to itself\",\"split\":true},"
         "\"schedules\":{\"large_all_reduce\":[\"pieces\",\"chain\",\"auto\",\"ring\"],\"all_gather\":"
         "[\"pieces\",\"chain\",\"auto\",\"ring\"],\"reduce_scatter\":[\"pieces\",\"chain\",\"auto\",\"ring\"],"
         "\"default\":\"pieces\"},"
         "\"transports\":[\"verbs\",\"emulation\"],\"kernels\":\"ahead-of-time CUDA C++ "
         "(sm_120, sm_121): transport, fold and link packs\",\"bootstrap\":"
         "\"tcp (loopback or SIRCL_BOOTSTRAP_ADDR/IFNAME)\",\"world\":\"1-8\"}";
}
ncclResult_t ncclGetVersion(int *version) {
  CHECK_PROCESS();
  if (!version) return fail(ncclInvalidArgument, "version output is NULL");
  const char *value = sccl_env("LIBSIRCL_NCCL_API_VERSION");
  long code = 22705;
  if (value) {
    char *end;
    errno = 0;
    code = strtol(value, &end, 10);
    if (errno || end == value || *end || code <= 0 || code > INT_MAX)
      return fail(ncclInvalidArgument, "invalid LIBSIRCL_NCCL_API_VERSION");
  }
  *version = (int)code;
  return ncclSuccess;
}
const char *ncclGetErrorString(ncclResult_t result) {
  static const char *messages[] = {
    "no error", "unhandled CUDA error", "system error", "internal error",
    "invalid argument", "invalid usage", "remote error", "operation in progress", "timeout"
  };
  return result >= 0 && result < ncclNumResults ? messages[result] : "unknown error";
}
ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  CHECK_PROCESS();
  if (!id) return fail(ncclInvalidArgument, "unique ID output is NULL");
  ncclResult_t result = (ncclResult_t)sccl_bootstrap_id((unsigned char *)id->internal);
  return result == ncclSuccess ? result : fail(result, sccl_bootstrap_error());
}
/* Validate handles against the registry before dereferencing them. */
static struct ncclComm *acquire(ncclComm_t handle) {
  if (!sccl_bootstrap_process_valid()) return NULL;
  pthread_mutex_lock(&registry_lock);
  struct ncclComm *comm;
  for (comm = registry; comm && comm != handle; comm = comm->next) {}
  if (comm) ++comm->refs;
  pthread_mutex_unlock(&registry_lock);
  return comm;
}
static void release(struct ncclComm *comm) {
  pthread_mutex_lock(&registry_lock);
  if (--comm->refs == 0) pthread_cond_broadcast(&registry_idle);
  pthread_mutex_unlock(&registry_lock);
}
const char *ncclGetLastError(ncclComm_t handle) {
  if (!sccl_bootstrap_process_valid()) return "fork after libsircl API use requires exec before further calls";
  struct ncclComm *comm = acquire(handle);
  if (comm) {
    int error = atomic_load(&comm->error);
    if (error != ncclSuccess && error != ncclInProgress) read_error(comm, last_error, sizeof(last_error));
    release(comm);
  }
  return last_error[0] ? last_error : "no error";
}
static void *initialize(void *arg) {
  struct ncclComm *comm = arg;
  int result = sccl_bootstrap_join_cancel(comm->id, comm->nranks, comm->rank, &comm->bootstrap, &comm->cancelled);
  if (result)
    snprintf(comm->error_message, sizeof(comm->error_message), "rank %d: %s", comm->rank, sccl_bootstrap_error());
  if (!result && !comm->bootstrap_only) {
    result = sccl_engine_create(comm->bootstrap, comm->nranks, comm->rank, comm->position, comm->ctx,
                                &comm->cancelled, &comm->engine, comm->error_message, sizeof(comm->error_message));
    if (!result) comm->device = sccl_engine_device(comm->engine);
  }
  atomic_store(&comm->state, result == 0 ? READY : FAILED);
  atomic_store(&comm->error, result);
  return NULL;
}
static ncclResult_t start(struct ncclComm *comm) {
  if (pthread_create(&comm->worker, NULL, initialize, comm)) {
    atomic_store(&comm->state, FAILED);
    snprintf(comm->error_message, sizeof(comm->error_message), "cannot start bootstrap worker");
    atomic_store(&comm->error, ncclSystemError);
    return fail(ncclSystemError, "cannot start bootstrap worker");
  }
  comm->worker_started = 1;
  return ncclSuccess;
}
/* The blocking mode of a new communicator: the config's, or `inherited` when the config is NULL or leaves the
 * field undefined (1 for ncclCommInitRank*, the parent's for ncclCommSplit). */
static ncclResult_t config_blocking(const ncclConfig_t *config, int inherited, int *blocking) {
  *blocking = inherited;
  if (!config) return ncclSuccess;
  size_t size;
  memcpy(&size, config, sizeof(size));
  /* Read only the known prefix, including for older 72-byte configurations. */
  if (size < offsetof(ncclConfig_t, blocking) + sizeof(int) || size > 4096)
    return fail(ncclInvalidArgument, "config size does not contain the ABI prefix");
  unsigned magic, version;
  memcpy(&magic, (const char *)config + offsetof(ncclConfig_t, magic), sizeof(magic));
  memcpy(&version, (const char *)config + offsetof(ncclConfig_t, version), sizeof(version));
  if (magic != NCCL_API_MAGIC || version < 21400)
    return fail(ncclInvalidArgument, "config magic or version is invalid");
  memcpy(blocking, (const char *)config + offsetof(ncclConfig_t, blocking), sizeof(*blocking));
  if (*blocking == NCCL_CONFIG_UNDEF_INT) *blocking = inherited;
  if (*blocking != 0 && *blocking != 1)
    return fail(ncclInvalidArgument, "config blocking must be 0, 1, or UNDEF");
  return ncclSuccess;
}
/* LIBSIRCL_POSITION, or -1 when unset (the position is then the rank in the communicator). */
static int env_position(int *out) {
  const char *text = sccl_env("LIBSIRCL_POSITION");
  *out = -1;
  if (!text || !*text) return 0;
  char *end;
  errno = 0;
  long value = strtol(text, &end, 10);
  if (errno || *end || value < 0 || value > 63) return -1;
  *out = (int)value;
  return 0;
}
static ncclResult_t init_rank(ncclComm_t *out, int nranks, ncclUniqueId id, int rank, ncclConfig_t *config,
                              int position, int inherited_blocking);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *out, int nranks, ncclUniqueId id, int rank, ncclConfig_t *config) {
  CHECK_PROCESS();
  int position;
  if (env_position(&position)) return fail(ncclInvalidArgument, "LIBSIRCL_POSITION must be an integer 0-63");
  return init_rank(out, nranks, id, rank, config, position < 0 ? rank : position, 1);
}
/* Under NCCL_DEBUG (VERSION, WARN, INFO or TRACE), the first communicator creation of a process writes one
 * stderr line naming libsircl and the NCCL API level ncclGetVersion reports, so that framework logs
 * which print an NCCL version can be traced to libsircl. */
static void identify_once(void) {
  static atomic_int done;
  const char *debug = sccl_env("NCCL_DEBUG");
  if (!debug || (strcasecmp(debug, "VERSION") && strcasecmp(debug, "WARN") && strcasecmp(debug, "INFO") &&
                 strcasecmp(debug, "TRACE")))
    return;
  if (atomic_exchange(&done, 1)) return;
  int level = 0;
  if (ncclGetVersion(&level) != ncclSuccess) level = 0;
  fprintf(stderr, "libsircl %s (SIRCL's NCCL-compatible C API; not NVIDIA NCCL), NCCL API level %d\n",
          LIBSIRCL_VERSION, level);
}

static ncclResult_t init_rank(ncclComm_t *out, int nranks, ncclUniqueId id, int rank, ncclConfig_t *config,
                              int position, int inherited_blocking) {
  identify_once();
  if (!out) return fail(ncclInvalidArgument, "communicator output is NULL");
  *out = NULL;
  if (nranks < 1 || nranks > 8 || rank < 0 || rank >= nranks)
    return fail(ncclInvalidArgument, "bootstrap supports 1 to 8 ranks with valid rank numbers");
  int blocking;
  ncclResult_t result = config_blocking(config, inherited_blocking, &blocking);
  if (result != ncclSuccess) return result;
  const char *optin = sccl_env("LIBSIRCL_BOOTSTRAP_ONLY");
  int bootstrap_only = optin && !strcmp(optin, "1");
  sccl_CUcontext ctx = NULL;
  if (!bootstrap_only) {
    ctx = sccl_engine_current_context();
    if (!ctx)
      return fail(ncclInvalidUsage, "no current CUDA context: select the device (cudaSetDevice) before "
                                    "creating the communicator; LIBSIRCL_BOOTSTRAP_ONLY=1 creates CPU-only "
                                    "test communicators");
  }
  if (group_depth && pending_count == 64)
    return fail(ncclInvalidUsage, "group exceeds 64 pending initializations");
  struct ncclComm *comm = calloc(1, sizeof(*comm));
  if (!comm) return fail(ncclSystemError, "communicator allocation failed");
  pthread_mutex_init(&comm->error_lock, NULL);
  comm->nranks = nranks; comm->rank = rank; comm->blocking = blocking;
  comm->bootstrap_only = bootstrap_only; comm->ctx = ctx; comm->position = position;
  memcpy(comm->id, id.internal, 128);
  atomic_init(&comm->state, INIT);
  atomic_init(&comm->error, ncclInProgress);
  atomic_init(&comm->grouped, group_depth != 0);
  atomic_init(&comm->cancelled, 0);
  pthread_mutex_lock(&registry_lock);
  static uint64_t serials;
  comm->serial = ++serials;
  comm->next = registry; registry = comm;
  pthread_mutex_unlock(&registry_lock);
  if (group_depth) {
    *out = comm;
    pending[pending_count++] = comm;
    return ncclSuccess;
  }
  result = start(comm);
  if (result != ncclSuccess) { *out = comm; return result; }
  if (!blocking) { *out = comm; return ncclInProgress; }
  pthread_join(comm->worker, NULL);
  comm->worker_started = 0;
  *out = comm;
  result = (ncclResult_t)atomic_load(&comm->error);
  if (result != ncclSuccess) {
    char message[1200];
    snprintf(message, sizeof message, "communicator initialization failed: %s", comm->error_message);
    return fail(result, message);
  }
  return result;
}
ncclResult_t ncclCommInitRank(ncclComm_t *out, int nranks, ncclUniqueId id, int rank) {
  return ncclCommInitRankConfig(out, nranks, id, rank, NULL);
}

/* ncclCommSplit: two rounds over the parent's bootstrap. Round 1 gathers every rank's (color, key); the
 * ranks of one color form a communicator ordered by key, then parent rank. Round 2 gathers the unique id
 * each new rank 0 created; every member then joins its communicator with that id, keeping the parent's
 * route-map position. A rank passing NCCL_SPLIT_NOCOLOR takes part in both rounds and gets NULL. */
ncclResult_t ncclCommSplit(ncclComm_t handle, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  CHECK_PROCESS();
  if (!newcomm) return fail(ncclInvalidArgument, "ncclCommSplit: newcomm is NULL");
  *newcomm = NULL;
  if (color < NCCL_SPLIT_NOCOLOR) return fail(ncclInvalidArgument, "ncclCommSplit: color must be >= 0 or NOCOLOR");
  struct ncclComm *parent = acquire(handle);
  if (!parent) return fail(ncclInvalidArgument, "ncclCommSplit: unknown communicator handle");
  if (atomic_load(&parent->state) != READY || !parent->bootstrap) {
    release(parent);
    return fail(ncclInvalidUsage, "ncclCommSplit: the parent communicator is not ready");
  }
  int32_t mine[2] = {color, key};
  void *all = NULL;
  unsigned lengths[64];
  if (sccl_bootstrap_allgather(parent->bootstrap, mine, sizeof mine, &all, lengths, &parent->cancelled)) {
    char message[700];
    snprintf(message, sizeof message, "ncclCommSplit: exchanging colors: %s", sccl_bootstrap_error());
    release(parent);
    return fail(ncclSystemError, message);
  }
  const int32_t *records = all;
  int members[64], count = 0, rank = -1;
  for (int r = 0; r < parent->nranks; ++r)
    if (color != NCCL_SPLIT_NOCOLOR && records[2 * r] == color) members[count++] = r;
  /* Order by key, then parent rank (insertion sort of at most eight ranks). */
  for (int i = 1; i < count; ++i)
    for (int j = i; j > 0; --j) {
      int a = members[j - 1], b = members[j];
      if (records[2 * a + 1] < records[2 * b + 1] || (records[2 * a + 1] == records[2 * b + 1] && a < b)) break;
      members[j - 1] = b;
      members[j] = a;
    }
  for (int i = 0; i < count; ++i)
    if (members[i] == parent->rank) rank = i;
  free(all);
  ncclUniqueId id;
  memset(&id, 0, sizeof id);
  ncclResult_t result = ncclSuccess;
  if (rank == 0) result = ncclGetUniqueId(&id);
  if (result != ncclSuccess) {
    release(parent);
    return result;
  }
  if (sccl_bootstrap_allgather(parent->bootstrap, id.internal, sizeof id.internal, &all, lengths, &parent->cancelled)) {
    char message[700];
    snprintf(message, sizeof message, "ncclCommSplit: exchanging unique ids: %s", sccl_bootstrap_error());
    release(parent);
    return fail(ncclSystemError, message);
  }
  if (rank >= 0) memcpy(id.internal, (const unsigned char *)all + 128 * (size_t)members[0], 128);
  free(all);
  int position = parent->position, blocking = parent->blocking;
  release(parent);
  if (rank < 0) return ncclSuccess;
  return init_rank(newcomm, count, id, rank, config, position, blocking);
}
static void release(struct ncclComm *comm);
static ncclResult_t collective_ready(struct ncclComm *comm, const char *what, ncclDataType_t datatype,
                                     ncclRedOp_t op, char *message, size_t len);

/* Carry out (`run` set) or drop the queued point-to-point calls: per communicator, in the order of their
 * first call, all of that communicator's calls in issue order. A queued call holds no reference on its
 * communicator, so ncclCommAbort or ncclCommDestroy in another thread never waits for this thread's group;
 * the calls of a communicator destroyed before ncclGroupEnd fail with ncclInvalidUsage. */
static ncclResult_t flush_p2p(int run) {
  ncclResult_t result = ncclSuccess;
  char message[512] = {0};
  sccl_p2p_op *ops = run && p2p_count ? malloc(sizeof(*ops) * p2p_count) : NULL;
  if (run && p2p_count && !ops) {
    result = ncclSystemError;
    snprintf(message, sizeof message, "ncclGroupEnd: out of memory for %u point-to-point calls", p2p_count);
  }
  for (unsigned i = 0; i < p2p_count; ++i) {
    struct ncclComm *queued = p2p_queue[i].comm;
    if (!queued) continue;
    uint64_t serial = p2p_queue[i].serial;
    unsigned n = 0;
    for (unsigned j = i; j < p2p_count; ++j)
      if (p2p_queue[j].comm == queued && p2p_queue[j].serial == serial) {
        if (ops) ops[n++] = p2p_queue[j].op;
        p2p_queue[j].comm = NULL;
      }
    if (!ops || result != ncclSuccess) continue;
    struct ncclComm *comm = acquire(queued);
    if (!comm || comm->serial != serial) {
      if (comm) release(comm);
      result = ncclInvalidUsage;
      snprintf(message, sizeof message, "ncclGroupEnd: the communicator of %u queued point-to-point calls was "
               "destroyed or aborted before ncclGroupEnd", n);
      continue;
    }
    ncclResult_t r = collective_ready(comm, "ncclGroupEnd", ncclUint8, ncclSum, message, sizeof message);
    if (r == ncclSuccess) r = sccl_engine_p2p(comm->engine, ops, n, message, sizeof message);
    if (r != ncclSuccess) result = r;
    release(comm);
  }
  free(ops);
  p2p_count = 0;
  return result == ncclSuccess ? result : fail(result, message);
}

static ncclResult_t point_to_point(const char *what, int send, const void *buff, size_t count,
                                   ncclDataType_t datatype, int peer, ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  char message[512] = {0};
  if (!comm) {
    snprintf(message, sizeof message, "%s: unknown communicator handle", what);
    return refuse(ncclInvalidArgument, message);
  }
  ncclResult_t result = collective_ready(comm, what, datatype, ncclSum, message, sizeof message);
  if (result == ncclSuccess && (peer < 0 || peer >= comm->nranks)) {
    snprintf(message, sizeof message, "%s: peer %d is outside 0-%d", what, peer, comm->nranks - 1);
    result = ncclInvalidArgument;
  }
  size_t item = sccl_engine_type_size(datatype);
  if (result == ncclSuccess && item && count > SIZE_MAX / item) {
    snprintf(message, sizeof message, "%s: count overflows", what);
    result = ncclInvalidArgument;
  }
  if (result == ncclSuccess && count && !buff) {
    snprintf(message, sizeof message, "%s: NULL buffer", what);
    result = ncclInvalidArgument;
  }
  if (result != ncclSuccess) {
    release(comm);
    return refuse(result, message);
  }
  sccl_p2p_op op = {send, peer, buff, count * item, stream};
  if (group_depth) {
    if (p2p_count == p2p_capacity) {
      unsigned capacity = p2p_capacity ? 2 * p2p_capacity : 64;
      queued_p2p *grown = realloc(p2p_queue, sizeof(*grown) * capacity);
      if (!grown) {
        release(comm);
        return refuse(ncclSystemError, "out of memory queueing a point-to-point call");
      }
      p2p_queue = grown;
      p2p_capacity = capacity;
    }
    p2p_queue[p2p_count].comm = comm;
    p2p_queue[p2p_count].serial = comm->serial;
    p2p_queue[p2p_count].op = op;
    ++p2p_count;
    release(comm);
    return ncclSuccess;
  }
  result = sccl_engine_p2p(comm->engine, &op, 1, message, sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  return point_to_point("ncclSend", 1, sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  return point_to_point("ncclRecv", 0, recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t ncclGroupStart(void) {
  CHECK_PROCESS();
  if (group_depth == 256) return fail(ncclInvalidUsage, "group nesting exceeds 256");
  if (group_depth++ == 0) { group_error = ncclSuccess; pending_count = 0; }
  return ncclSuccess;
}
ncclResult_t ncclGroupEnd(void) {
  CHECK_PROCESS();
  if (!group_depth) return fail(ncclInvalidUsage, "GroupEnd without GroupStart");
  if (--group_depth) return ncclSuccess;
  ncclResult_t result = group_error;
  for (unsigned i = 0; i < pending_count; ++i) {
    if (result != ncclSuccess) {
      atomic_store(&pending[i]->state, FAILED);
      snprintf(pending[i]->error_message, sizeof(pending[i]->error_message), "group initialization rejected");
      atomic_store(&pending[i]->error, result);
    } else {
      ncclResult_t r = start(pending[i]);
      if (r != ncclSuccess) result = r;
    }
  }
  if (result != ncclSuccess) {
    for (unsigned i = 0; i < pending_count; ++i)
      if (pending[i]->worker_started) atomic_store(&pending[i]->cancelled, 1);
  }
  for (unsigned i = 0; i < pending_count; ++i) {
    struct ncclComm *comm = pending[i];
    if (comm->blocking && comm->worker_started) {
      pthread_join(comm->worker, NULL); comm->worker_started = 0;
      ncclResult_t r = (ncclResult_t)atomic_load(&comm->error);
      if ((result == ncclSuccess || result == ncclInProgress) && r != ncclSuccess) result = r;
    } else if (!comm->blocking && result == ncclSuccess) result = ncclInProgress;
    atomic_store(&comm->grouped, 0);
  }
  pending_count = 0;
  if (result != ncclSuccess && result != ncclInProgress) {
    flush_p2p(0);
    return fail(result, "group initialization failed");
  }
  ncclResult_t sent = flush_p2p(1);
  return sent != ncclSuccess ? sent : result;
}
ncclResult_t ncclCommInitAll(ncclComm_t *out, int ndev, const int *devlist) {
  CHECK_PROCESS();
  if (!out || ndev < 1 || ndev > 8) return fail(ncclInvalidArgument, "InitAll requires 1 to 8 output handles");
  for (int i = 0; i < ndev; ++i) out[i] = NULL;
  for (int i = 0; i < ndev; ++i) if (devlist && devlist[i] < 0)
    return fail(ncclInvalidArgument, "negative device number");
  ncclUniqueId id;
  ncclResult_t result = ncclGetUniqueId(&id);
  if (result != ncclSuccess) return result;
  result = ncclGroupStart();
  if (result != ncclSuccess) return result;
  for (int i = 0; i < ndev; ++i) {
    result = ncclCommInitRank(&out[i], ndev, id, i);
    if (result != ncclSuccess) break;
    out[i]->device = devlist ? devlist[i] : i;
  }
  ncclResult_t end = ncclGroupEnd();
  return result != ncclSuccess ? result : end;
}
static ncclResult_t query(ncclComm_t handle, int *out, int field) {
  CHECK_PROCESS();
  if (!out) return fail(ncclInvalidArgument, "query output is NULL");
  struct ncclComm *comm = acquire(handle);
  if (!comm) return fail(ncclInvalidArgument, "unknown communicator handle");
  ncclResult_t result = (ncclResult_t)atomic_load(&comm->error);
  if (result == ncclSuccess) *out = field == 0 ? comm->nranks : field == 1 ? comm->rank : comm->device;
  release(comm);
  return result;
}
ncclResult_t ncclCommCount(ncclComm_t comm, int *out) { return query(comm, out, 0); }

/* Device memory for buffers (nccl-tests allocates its buffers this way): cuMemAlloc in the current context. */
ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  CHECK_PROCESS();
  if (!ptr) return fail(ncclInvalidArgument, "ncclMemAlloc: pointer output is NULL");
  char message[256] = {0};
  ncclResult_t result = sccl_engine_mem_alloc(ptr, size, message, sizeof message);
  return result == ncclSuccess ? result : fail(result, message);
}
ncclResult_t ncclMemFree(void *ptr) {
  CHECK_PROCESS();
  char message[256] = {0};
  ncclResult_t result = sccl_engine_mem_free(ptr, message, sizeof message);
  return result == ncclSuccess ? result : fail(result, message);
}

/* Buffer and window registration are hints libsircl accepts without work: its collectives move data
 * through the session's own pinned arena, whatever the buffer. A buffer handle and a window are the
 * buffer's address; deregistration accepts them. ncclWinGetUserPtr returns the address. */
static ncclResult_t known(ncclComm_t handle, const char *what) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) {
    char message[128];
    snprintf(message, sizeof message, "%s: unknown communicator handle", what);
    return fail(ncclInvalidArgument, message);
  }
  release(comm);
  return ncclSuccess;
}
ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle) {
  (void)size;
  if (!handle) return fail(ncclInvalidArgument, "ncclCommRegister: handle output is NULL");
  ncclResult_t result = known(comm, "ncclCommRegister");
  if (result == ncclSuccess) *handle = buff;
  return result;
}
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  (void)handle;
  return known(comm, "ncclCommDeregister");
}
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size, ncclWindow_t *win, int winFlags) {
  (void)size;
  (void)winFlags;
  if (!win) return fail(ncclInvalidArgument, "ncclCommWindowRegister: window output is NULL");
  ncclResult_t result = known(comm, "ncclCommWindowRegister");
  if (result == ncclSuccess) *win = (ncclWindow_t)buff;
  return result;
}
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  (void)win;
  return known(comm, "ncclCommWindowDeregister");
}
ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void **outUserPtr) {
  if (!outUserPtr) return fail(ncclInvalidArgument, "ncclWinGetUserPtr: pointer output is NULL");
  ncclResult_t result = known(comm, "ncclWinGetUserPtr");
  if (result == ncclSuccess) *outUserPtr = (void *)win;
  return result;
}
ncclResult_t ncclCommUserRank(ncclComm_t comm, int *out) { return query(comm, out, 1); }
ncclResult_t ncclCommCuDevice(ncclComm_t comm, int *out) { return query(comm, out, 2); }
ncclResult_t ncclCommGetAsyncError(ncclComm_t handle, ncclResult_t *out) {
  CHECK_PROCESS();
  if (!out) return fail(ncclInvalidArgument, "async error output is NULL");
  struct ncclComm *comm = acquire(handle);
  if (!comm) return fail(ncclInvalidArgument, "unknown communicator handle");
  *out = (ncclResult_t)atomic_load(&comm->error);
  if (*out == ncclSuccess && comm->engine) {
    char message[512];
    ncclResult_t failure = sccl_engine_async_error(comm->engine, message, sizeof message);
    if (failure != ncclSuccess) {
      /* The first asynchronous failure is terminal and keeps its description. */
      if (publish_error(comm, failure, message)) atomic_store(&comm->state, FAILED);
      *out = (ncclResult_t)atomic_load(&comm->error);
    }
  }
  release(comm);
  return ncclSuccess;
}

/* The checks every collective makes before the engine: a ready SIRCL communicator, a valid datatype
 * and op. ncclSuccess when the call may proceed; otherwise `message` says why. */
static ncclResult_t collective_ready(struct ncclComm *comm, const char *what, ncclDataType_t datatype,
                                     ncclRedOp_t op, char *message, size_t len) {
  ncclResult_t result = (ncclResult_t)atomic_load(&comm->error);
  if (result != ncclSuccess) {
    char text[sizeof comm->error_message] = "initialization in progress";
    if (result != ncclInProgress) read_error(comm, text, sizeof text);
    snprintf(message, len, "%s: communicator is not ready: %s", what, text);
    return result == ncclInProgress ? ncclInvalidUsage : result;
  }
  if (atomic_load(&comm->state) != READY) {
    snprintf(message, len, "%s: communicator is finalized or revoked", what);
    return ncclInvalidUsage;
  }
  if (!comm->engine) {
    snprintf(message, len, "%s: CPU-only test communicators carry no collectives", what);
    return ncclInvalidUsage;
  }
  if ((int)datatype < 0 || (int)datatype >= (int)ncclNumTypes || (int)op < 0 || (int)op >= (int)ncclMaxRedOp) {
    snprintf(message, len, "%s: invalid datatype %d or op %d", what, (int)datatype, (int)op);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclAllReduce: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclAllReduce", datatype, op, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_allreduce(comm->engine, sendbuff, recvbuff, count, datatype, op, stream, message,
                                   sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclAllGather: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclAllGather", datatype, ncclSum, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_allgather(comm->engine, sendbuff, recvbuff, sendcount, datatype, stream, message,
                                   sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                        int root, ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclReduce: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclReduce", datatype, op, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_reduce(comm->engine, sendbuff, recvbuff, count, datatype, op, root, stream, message,
                                sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclBroadcast: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclBroadcast", datatype, ncclSum, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_broadcast(comm->engine, sendbuff, recvbuff, count, datatype, root, stream, message,
                                   sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclReduceScatter: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclReduceScatter", datatype, op, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_reducescatter(comm->engine, sendbuff, recvbuff, recvcount, datatype, op, stream, message,
                                       sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype,
                          ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclAlltoAll: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclAlltoAll", datatype, ncclSum, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_alltoall(comm->engine, sendbuff, recvbuff, count, datatype, stream, message, sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
                        ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclGather: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclGather", datatype, ncclSum, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_gather(comm->engine, sendbuff, recvbuff, count, datatype, root, stream, message,
                                sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
                         ncclComm_t handle, cudaStream_t stream) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return refuse(ncclInvalidArgument, "ncclScatter: unknown communicator handle");
  char message[512] = {0};
  ncclResult_t result = collective_ready(comm, "ncclScatter", datatype, ncclSum, message, sizeof message);
  if (result == ncclSuccess)
    result = sccl_engine_scatter(comm->engine, sendbuff, recvbuff, count, datatype, root, stream, message,
                                 sizeof message);
  release(comm);
  return result == ncclSuccess ? result : refuse(result, message);
}

ncclResult_t sirclSetWaitRegime(ncclComm_t handle, const char *regime) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return fail(ncclInvalidArgument, "unknown communicator handle");
  ncclResult_t result = comm->engine ? sccl_engine_set_wait_regime(comm->engine, regime) : ncclInvalidUsage;
  release(comm);
  return result == ncclSuccess ? result
                               : fail(result, "sirclSetWaitRegime: \"startup\" or \"serving\" on a SIRCL "
                                              "communicator");
}

ncclResult_t sirclGetReceipt(ncclComm_t handle, char *out, size_t length, size_t *needed) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return fail(ncclInvalidArgument, "unknown communicator handle");
  ncclResult_t result = ncclSuccess;
  size_t need = 0;
  if (comm->engine) {
    need = sccl_engine_receipt(comm->engine, out, out ? length : 0) + 1;
  } else {
    result = ncclInvalidUsage;
  }
  release(comm);
  if (needed) *needed = need;
  return result == ncclSuccess ? result : fail(result, "sirclGetReceipt: no SIRCL session on this communicator");
}
static ncclResult_t transition(ncclComm_t handle, int next) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return fail(ncclInvalidArgument, "unknown communicator handle");
  int state = atomic_load(&comm->state);
  ncclResult_t result = (ncclResult_t)atomic_load(&comm->error);
  if (result == ncclSuccess) {
    if (state == READY || (state == next)) atomic_store(&comm->state, next);
    else result = ncclInvalidUsage;
  }
  release(comm);
  return result == ncclSuccess ? result : fail(result, "communicator is not ready for this lifecycle operation");
}
/* Teardown of a communicator of two or more ranks, so that no rank frees what a peer still writes. This rank's
 * enqueued work completes; round 1 over the bootstrap carries every rank's status (its wait, then its health:
 * a flag wait that timed out, a failed progress thread, an earlier asynchronous error) and proves that every
 * rank's work completed, so every item and flag any rank's kernels wait for has landed; this rank's progress
 * thread stops and posts nothing more; round 2 proves every progress thread stopped, so nothing is posted
 * toward this rank's queue pairs or regions any more and destroy may free them. Each round is bounded by the
 * session's wait limit. The close is terminal: from its start the communicator refuses new work, and its
 * result (this rank's own error first, else ncclRemoteError naming the first rank that reported one, or a
 * round that not every rank reached) is kept, becomes the communicator's error when it failed, and is what a
 * repeated ncclCommFinalize or the destroy returns. */
static ncclResult_t quiesce(struct ncclComm *comm, const char *call) {
  if (comm->quiesced) return (ncclResult_t)comm->close_result;
  if (!comm->engine || comm->bootstrap_only || comm->nranks < 2 || !comm->bootstrap) return ncclSuccess;
  int state = atomic_load(&comm->state);
  if (state != READY && state != FINALIZED && state != FAILED) return ncclSuccess;
  comm->quiesced = 1;
  if (state == READY) atomic_store(&comm->state, FINALIZED);
  char message[700] = {0};
  ncclResult_t result = (ncclResult_t)atomic_load(&comm->error);
  if (result != ncclSuccess) read_error(comm, message, sizeof message);
  ncclResult_t synced = sccl_engine_teardown_sync(comm->engine);
  if (result == ncclSuccess && synced != ncclSuccess) {
    result = synced;
    snprintf(message, sizeof message, "%s: waiting for this rank's enqueued work failed", call);
  }
  if (result == ncclSuccess) result = sccl_engine_async_error(comm->engine, message, sizeof message);
  const atomic_int never = 0;
  int limit_ms = sccl_engine_wait_limit_ms(comm->engine), arrived = 0;
  for (int round = 1; round <= 2; ++round) {
    if (round == 2) sccl_engine_teardown_stop(comm->engine);
    if (round == 2 && !arrived) break;
    int32_t mine = round == 1 ? (int32_t)result : 0;
    void *all = NULL;
    unsigned lengths[64] = {0};
    arrived = sccl_bootstrap_allgather_within(comm->bootstrap, &mine, sizeof mine, &all, lengths, limit_ms,
                                              &never) == 0;
    if (arrived && round == 1 && result == ncclSuccess) {
      for (int r = 0; r < comm->nranks; ++r) {
        int32_t theirs = 0;
        if (lengths[r] == sizeof theirs) memcpy(&theirs, (const char *)all + sizeof theirs * (size_t)r, sizeof theirs);
        if (theirs != ncclSuccess) {
          result = ncclRemoteError;
          snprintf(message, sizeof message, "%s: rank %d reported %s at teardown", call, r,
                   ncclGetErrorString((ncclResult_t)theirs));
          break;
        }
      }
    }
    free(all);
    if (!arrived) {
      if (result == ncclSuccess) {
        result = ncclRemoteError;
        snprintf(message, sizeof message, "%s: teardown round %d: not every rank arrived within the wait limit "
                 "(%d ms): %s", call, round, limit_ms, sccl_bootstrap_error());
      }
      if (round == 1) sccl_engine_teardown_stop(comm->engine);
      break;
    }
  }
  comm->close_result = result;
  if (result != ncclSuccess) {
    char text[sizeof comm->error_message];
    if (strncmp(message, "rank ", 5)) snprintf(text, sizeof text, "rank %d: %s", comm->rank, message);
    else snprintf(text, sizeof text, "%s", message);
    publish_error(comm, result, text);
    read_error(comm, text, sizeof text);
    fail(result, text);
  }
  return result;
}

ncclResult_t ncclCommFinalize(ncclComm_t handle) {
  CHECK_PROCESS();
  struct ncclComm *comm = acquire(handle);
  if (!comm) return fail(ncclInvalidArgument, "unknown communicator handle");
  ncclResult_t flushed = comm->engine && atomic_load(&comm->state) == READY ? sccl_engine_finalize(comm->engine)
                                                                             : ncclSuccess;
  if (flushed == ncclSuccess) flushed = quiesce(comm, "ncclCommFinalize");
  release(comm);
  if (flushed != ncclSuccess) return fail(flushed, last_error[0] ? last_error : "ncclCommFinalize failed");
  return transition(handle, FINALIZED);
}
ncclResult_t ncclCommRevoke(ncclComm_t comm, int flags) {
  if (flags) return fail(ncclInvalidArgument, "revoke flags must be zero");
  return transition(comm, REVOKED);
}
static ncclResult_t dispose(ncclComm_t handle, int abort) {
  CHECK_PROCESS();
  /* Group-owned pending handles cannot be removed before GroupEnd consumes them. */
  for (unsigned i = 0; i < pending_count; ++i) if (pending[i] == handle)
    return fail(ncclInvalidUsage, "communicator initialization is pending in a group");
  pthread_mutex_lock(&registry_lock);
  struct ncclComm **link = &registry;
  while (*link && *link != handle) link = &(*link)->next;
  if (!*link) {
    pthread_mutex_unlock(&registry_lock);
    return fail(ncclInvalidArgument, "unknown communicator handle");
  }
  struct ncclComm *comm = *link;
  if (atomic_load(&comm->grouped)) {
    pthread_mutex_unlock(&registry_lock);
    return fail(ncclInvalidUsage, "communicator initialization is pending in a group");
  }
  *link = comm->next;
  while (comm->refs) pthread_cond_wait(&registry_idle, &registry_lock);
  pthread_mutex_unlock(&registry_lock);
  atomic_store(&comm->cancelled, 1);
  if (comm->worker_started) pthread_join(comm->worker, NULL);
  ncclResult_t result = abort ? ncclSuccess : quiesce(comm, "ncclCommDestroy");
  char message[sizeof comm->error_message];
  read_error(comm, message, sizeof message);
  char released[400] = {0};
  ncclResult_t freed = sccl_engine_destroy(comm->engine, abort, released, sizeof released);
  if (result == ncclSuccess && freed != ncclSuccess) {
    result = freed;
    snprintf(message, sizeof message, "%s", released);
  }
  sccl_bootstrap_close(comm->bootstrap);
  pthread_mutex_destroy(&comm->error_lock);
  free(comm);
  return result == ncclSuccess ? ncclSuccess : fail(result, message);
}
ncclResult_t ncclCommDestroy(ncclComm_t comm) { return dispose(comm, 0); }
ncclResult_t ncclCommAbort(ncclComm_t comm) { return dispose(comm, 1); }

/* A library unload must not leave a worker executing code in an unmapped DSO.
 * dlclose concurrent with API calls is outside the supported caller contract. */
__attribute__((destructor)) static void shutdown_communicators(void) {
  if (!sccl_bootstrap_process_valid()) return;
  pthread_mutex_lock(&registry_lock);
  struct ncclComm *comm = registry;
  registry = NULL;
  for (struct ncclComm *entry = comm; entry; entry = entry->next)
    atomic_store(&entry->cancelled, 1);
  pthread_mutex_unlock(&registry_lock);
  while (comm) {
    struct ncclComm *next = comm->next;
    if (comm->worker_started) pthread_join(comm->worker, NULL);
    /* At unload only the native threads stop; CUDA may already be torn down. */
    sccl_engine_shutdown(comm->engine);
    sccl_bootstrap_close(comm->bootstrap);
    free(comm);
    comm = next;
  }
}
