/* Shared-memory verbs stand-in: the libibverbs subset of SIRCL's native proxy
 * across the processes of one host. shm_verbs.h describes the model. */
#define _GNU_SOURCE
#ifndef SCCL_RENAME_VERBS
#define SCCL_RENAME_VERBS 1
#endif
#define SCCL_PROXY_TAG emu
#include "proxy_names.h"

#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "shm_verbs.h"

#define SV_MAGIC 0x31465653u /* "SVF1" in byte order */
#define SV_VERSION 1u
#define SV_MAX_DEVICES 256
#define SV_MAX_MRS 1024
#define SV_MAX_QPS 2048
#define SV_NAME 64
#define SV_INLINE 16
#define SV_LOCAL_QPS 1024
#define SV_LOCAL_CQS 256
#define SV_SEGMENTS 512

/* -- shared registry --------------------------------------------------------------------------- */

typedef struct {
  int32_t used, pid, failed, reserved;
  char name[SV_NAME];
  uint8_t gid[16];
} sv_dev;

typedef struct {
  int32_t used, pid, device, access;
  uint32_t lkey, rkey;
  uint64_t addr, length, seg_offset;
  char segment[SV_NAME];
} sv_mr;

typedef struct {
  int32_t used, pid, device, state;
  uint32_t qp_num, dest_qp_num;
  uint8_t dgid[16];
} sv_qp;

typedef struct {
  uint32_t magic, version;
  uint64_t bytes;
  pthread_mutex_t lock;
  uint32_t next_key, next_qpn, next_segment, reserved;
  sv_dev dev[SV_MAX_DEVICES];
  sv_mr mr[SV_MAX_MRS];
  sv_qp qp[SV_MAX_QPS];
} sv_fabric;

static sv_fabric *F;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t seed_state = 1;
static uint64_t latency_ns;

static void fab_lock(void) {
  if (pthread_mutex_lock(&F->lock) == EOWNERDEAD) pthread_mutex_consistent(&F->lock);
}
static void fab_unlock(void) { pthread_mutex_unlock(&F->lock); }

static int pid_alive(int pid) { return pid > 0 && (kill(pid, 0) == 0 || errno != ESRCH); }

/* Records of processes that exited without cleaning up (callers hold the lock). */
static void reap_dead(void) {
  for (int i = 0; i < SV_MAX_DEVICES; ++i)
    if (F->dev[i].used && !pid_alive(F->dev[i].pid)) F->dev[i].used = 0;
  for (int i = 0; i < SV_MAX_MRS; ++i)
    if (F->mr[i].used && !pid_alive(F->mr[i].pid)) F->mr[i].used = 0;
  for (int i = 0; i < SV_MAX_QPS; ++i)
    if (F->qp[i].used && !pid_alive(F->qp[i].pid)) F->qp[i].used = 0;
}

int sccl_emu_attach(char *err, size_t err_len) {
  pthread_mutex_lock(&attach_lock);
  if (F) {
    pthread_mutex_unlock(&attach_lock);
    return 0;
  }
  char name[SV_NAME];
  const char *configured = getenv("SIRCL_EMU_FABRIC");
  if (configured && *configured)
    snprintf(name, sizeof name, "%s%s", configured[0] == '/' ? "" : "/", configured);
  else
    snprintf(name, sizeof name, "/sircl-emu-%u", (unsigned)getuid());
  const char *seed = getenv("SIRCL_EMU_SEED");
  seed_state = seed && *seed ? strtoull(seed, NULL, 10) : 1;
  if (!seed_state) seed_state = 1;
  seed_state ^= (uint64_t)getpid() * 0x9E3779B97F4A7C15ull;
  const char *latency = getenv("SIRCL_EMU_LATENCY_NS");
  latency_ns = latency && *latency ? strtoull(latency, NULL, 10) : 0;
  int fd = shm_open(name, O_RDWR | O_CREAT, 0600);
  if (fd < 0) {
    snprintf(err, err_len, "emulation fabric %s: shm_open: %s", name, strerror(errno));
    pthread_mutex_unlock(&attach_lock);
    return -1;
  }
  flock(fd, LOCK_EX);
  struct stat st;
  int fresh = fstat(fd, &st) == 0 && (size_t)st.st_size < sizeof(sv_fabric);
  if (fresh && ftruncate(fd, (off_t)sizeof(sv_fabric)) != 0) {
    snprintf(err, err_len, "emulation fabric %s: ftruncate: %s", name, strerror(errno));
    flock(fd, LOCK_UN);
    close(fd);
    pthread_mutex_unlock(&attach_lock);
    return -1;
  }
  sv_fabric *f = mmap(NULL, sizeof(sv_fabric), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (f == MAP_FAILED) {
    snprintf(err, err_len, "emulation fabric %s: mmap: %s", name, strerror(errno));
    flock(fd, LOCK_UN);
    close(fd);
    pthread_mutex_unlock(&attach_lock);
    return -1;
  }
  if (fresh || f->magic != SV_MAGIC) {
    memset(f, 0, sizeof *f);
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&f->lock, &attr);
    pthread_mutexattr_destroy(&attr);
    f->bytes = sizeof *f;
    f->next_key = 0x100;
    f->next_qpn = 0x40;
    f->version = SV_VERSION;
    __atomic_store_n(&f->magic, SV_MAGIC, __ATOMIC_RELEASE);
  } else if (f->version != SV_VERSION || f->bytes != sizeof *f) {
    snprintf(err, err_len, "emulation fabric %s has another layout (version %u)", name, f->version);
    munmap(f, sizeof *f);
    flock(fd, LOCK_UN);
    close(fd);
    pthread_mutex_unlock(&attach_lock);
    return -1;
  }
  flock(fd, LOCK_UN);
  close(fd);
  F = f;
  fab_lock();
  reap_dead();
  fab_unlock();
  pthread_mutex_unlock(&attach_lock);
  return 0;
}

/* -- local state ------------------------------------------------------------------------------- */

typedef struct {
  char name[SV_NAME];
  uint8_t *base;
  size_t bytes;
  int owned;
  int used;
} sv_segment;

typedef struct {
  struct ibv_cq cq;
  int used;
  struct ibv_wc *entries;
  int capacity, head, count;
} sv_cq;

typedef struct {
  uint64_t wr_id;
  int signaled, is_inline;
  uint8_t inline_data[SV_INLINE];
  uint64_t local_addr;
  uint32_t length, lkey;
  uint64_t remote_addr;
  uint32_t rkey;
  uint64_t ready_ns;
} sv_wr;

typedef struct {
  struct ibv_qp qp;
  int shared;    /* index in F->qp */
  int device;    /* index in F->dev */
  int errored;
  uint32_t generation;
  uint32_t dest_qp_num;
  sv_wr *queue;
  int capacity, head, count;
  uint64_t posted, executed, completed; /* diagnostics: writes posted, executed, completions delivered */
} sv_lqp;

static pthread_mutex_t local = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t segments_lock = PTHREAD_MUTEX_INITIALIZER;
static sv_segment segments[SV_SEGMENTS];
static sv_cq cqs[SV_LOCAL_CQS];
static sv_lqp *qps[SV_LOCAL_QPS];
static int nqps;
static uint32_t generations;
static struct ibv_device devices[SV_MAX_DEVICES];
static pthread_t executor;
static int executor_running;
static volatile int executor_stop;
static uint64_t executed;

/* Diagnostics (sccl_emu_report): the newest SV_ERRORS writes that did not succeed, and the completions the
 * stand-in could not deliver (a full completion queue; a queue pair destroyed while its write executed). */
#define SV_ERRORS 64
typedef struct {
  uint64_t ns; /* CLOCK_REALTIME, the clock of the native proxy's event trace */
  uint64_t wr_id, remote_addr;
  uint32_t qp_num, dest_qp_num, rkey, length;
  int32_t status, signaled, is_inline, peer_pid, peer_alive, delivered;
  char reason[192];
} sv_error;
static sv_error errors[SV_ERRORS];
static uint64_t error_count, cq_overflows, replaced_drops;

static uint64_t realtime_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t next_random(void) {
  seed_state ^= seed_state << 13;
  seed_state ^= seed_state >> 7;
  seed_state ^= seed_state << 17;
  return seed_state;
}

int sccl_emu_add_device(const char *name, const uint8_t gid[16]) {
  if (!F || !name || strlen(name) >= SV_NAME) return -1;
  fab_lock();
  int index = -1;
  for (int i = 0; i < SV_MAX_DEVICES; ++i) {
    if (F->dev[i].used && F->dev[i].pid == (int)getpid() && !strcmp(F->dev[i].name, name)) {
      index = i;
      break;
    }
  }
  for (int i = 0; index < 0 && i < SV_MAX_DEVICES; ++i) {
    if (!F->dev[i].used) {
      memset(&F->dev[i], 0, sizeof F->dev[i]);
      F->dev[i].used = 1;
      F->dev[i].pid = (int)getpid();
      snprintf(F->dev[i].name, SV_NAME, "%s", name);
      memcpy(F->dev[i].gid, gid, 16);
      index = i;
    }
  }
  fab_unlock();
  return index;
}

/* The part of a segment name that tells this process apart: "<pid>-<start>", <start> the process's start
 * time in clock ticks since boot (field 22 of /proc/self/stat, hexadecimal), so a process of another PID
 * namespace that reuses the pid (containers sharing the host's /dev/shm) names its segments otherwise;
 * SIRCL_EMU_SEGMENT_TAG (a test setting) replaces it. */
static void segment_tag(char *out, size_t len) {
  const char *tag = getenv("SIRCL_EMU_SEGMENT_TAG");
  if (tag && *tag) {
    snprintf(out, len, "%s", tag);
    return;
  }
  unsigned long long start = 0;
  char stat[1024] = {0};
  FILE *f = fopen("/proc/self/stat", "r");
  if (f) {
    size_t n = fread(stat, 1, sizeof stat - 1, f);
    stat[n] = 0;
    fclose(f);
    /* Field 2 (the command, in parentheses) may hold spaces: count fields after its closing parenthesis.
     * Field 3 follows it; field 22 is the 20th after. */
    char *p = strrchr(stat, ')');
    for (int field = 3; p && *p && field <= 22; ++field) {
      p = strchr(p + 1, ' ');
      if (p && field == 22) start = strtoull(p + 1, NULL, 10);
    }
  }
  snprintf(out, len, "%d-%llx", (int)getpid(), start);
}

void *sccl_emu_segment_alloc(size_t bytes, char *err, size_t err_len) {
  if (!F) {
    snprintf(err, err_len, "emulation fabric is not attached");
    return NULL;
  }
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  bytes = (bytes + page - 1) / page * page;
  char tag[40], name[SV_NAME];
  segment_tag(tag, sizeof tag);
  /* A name that exists already belongs to a segment another process left (or holds): take the fabric's
   * next serial instead. It is never unlinked here; its owner may live in another PID namespace. */
  int fd = -1;
  for (int attempt = 0; attempt < 64; ++attempt) {
    fab_lock();
    uint32_t serial = ++F->next_segment;
    fab_unlock();
    snprintf(name, sizeof name, "/sircl-emu-seg-%s-%u", tag, serial);
    fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0 || errno != EEXIST) break;
  }
  if (fd < 0) {
    snprintf(err, err_len, "emulation segment %s: shm_open: %s", name, strerror(errno));
    return NULL;
  }
  if (ftruncate(fd, (off_t)bytes) != 0) {
    snprintf(err, err_len, "emulation segment %s of %zu bytes: ftruncate: %s", name, bytes, strerror(errno));
    close(fd);
    shm_unlink(name);
    return NULL;
  }
  void *base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) {
    snprintf(err, err_len, "emulation segment %s: mmap: %s", name, strerror(errno));
    shm_unlink(name);
    return NULL;
  }
  pthread_mutex_lock(&segments_lock);
  int slot = -1;
  for (int i = 0; i < SV_SEGMENTS; ++i)
    if (!segments[i].used) {
      slot = i;
      break;
    }
  if (slot < 0) {
    pthread_mutex_unlock(&segments_lock);
    munmap(base, bytes);
    shm_unlink(name);
    snprintf(err, err_len, "more than %d emulation segments in one process", SV_SEGMENTS);
    return NULL;
  }
  snprintf(segments[slot].name, SV_NAME, "%s", name);
  segments[slot].base = base;
  segments[slot].bytes = bytes;
  segments[slot].owned = 1;
  segments[slot].used = 1;
  pthread_mutex_unlock(&segments_lock);
  return base;
}

void sccl_emu_segment_free(void *base) {
  pthread_mutex_lock(&segments_lock);
  for (int i = 0; i < SV_SEGMENTS; ++i) {
    if (segments[i].used && segments[i].owned && segments[i].base == base) {
      munmap(segments[i].base, segments[i].bytes);
      shm_unlink(segments[i].name);
      segments[i].used = 0;
      break;
    }
  }
  pthread_mutex_unlock(&segments_lock);
}

uint64_t sccl_emu_executed(void) { return __atomic_load_n(&executed, __ATOMIC_RELAXED); }

/* The owned segment holding [addr, addr + length), or -1. */
static int owned_segment(uint64_t addr, uint64_t length) {
  for (int i = 0; i < SV_SEGMENTS; ++i) {
    sv_segment *s = &segments[i];
    uint64_t base = (uint64_t)(uintptr_t)s->base;
    if (s->used && s->owned && addr >= base && addr + length <= base + s->bytes) return i;
  }
  return -1;
}

/* This process's mapping of the named segment, mapped on first use. */
static uint8_t *mapped(const char *name, uint64_t *bytes) {
  pthread_mutex_lock(&segments_lock);
  for (int i = 0; i < SV_SEGMENTS; ++i) {
    if (segments[i].used && !strcmp(segments[i].name, name)) {
      uint8_t *base = segments[i].base;
      *bytes = segments[i].bytes;
      pthread_mutex_unlock(&segments_lock);
      return base;
    }
  }
  uint8_t *base = NULL;
  int fd = shm_open(name, O_RDWR, 0600);
  struct stat st;
  if (fd >= 0 && fstat(fd, &st) == 0) {
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p != MAP_FAILED) {
      for (int i = 0; i < SV_SEGMENTS; ++i) {
        if (!segments[i].used) {
          snprintf(segments[i].name, SV_NAME, "%s", name);
          segments[i].base = p;
          segments[i].bytes = (size_t)st.st_size;
          segments[i].owned = 0;
          segments[i].used = 1;
          base = p;
          *bytes = (uint64_t)st.st_size;
          break;
        }
      }
      if (!base) munmap(p, (size_t)st.st_size);
    }
  }
  if (fd >= 0) close(fd);
  pthread_mutex_unlock(&segments_lock);
  return base;
}

/* -- execution --------------------------------------------------------------------------------- */

/* Leave a completion on the queue pair's send queue (local lock held); 0 when the queue is full or gone. */
static int complete(sv_lqp *q, uint64_t wr_id, enum ibv_wc_status status) {
  sv_cq *cq = &cqs[q->qp.send_cq->fv_index];
  if (!cq->used || cq->count == cq->capacity) {
    cq_overflows++;
    return 0;
  }
  struct ibv_wc *wc = &cq->entries[(cq->head + cq->count) % cq->capacity];
  memset(wc, 0, sizeof *wc);
  wc->wr_id = wr_id;
  wc->status = status;
  wc->opcode = IBV_WC_RDMA_WRITE;
  wc->qp_num = q->qp.qp_num;
  cq->count++;
  return 1;
}

/* Resolve one write of queue pair `q` (fabric lock held): the destination pointer in this
 * process and the source pointer, or a completion status and, in `reason`, the check that failed.
 * `peer_pid` receives the process owning the destination device (0 when none is registered). */
static enum ibv_wc_status resolve_write(const sv_lqp *q, const sv_wr *w, char *segment, uint64_t *seg_offset,
                                        const void **source, char *reason, size_t reason_len, int *peer_pid) {
  const sv_qp *self = &F->qp[q->shared];
  int dst = -1;
  for (int i = 0; i < SV_MAX_DEVICES; ++i)
    if (F->dev[i].used && !memcmp(F->dev[i].gid, self->dgid, 16)) {
      *peer_pid = F->dev[i].pid;
      if (!F->dev[i].failed) {
        dst = i;
        break;
      }
    }
  if (dst < 0) {
    snprintf(reason, reason_len, *peer_pid ? "the destination device is marked failed"
                                           : "no registered device has the destination GID");
    return IBV_WC_RETRY_EXC_ERR;
  }
  if (F->dev[q->device].failed) {
    snprintf(reason, reason_len, "this queue pair's device is marked failed");
    return IBV_WC_RETRY_EXC_ERR;
  }
  const sv_qp *peer = NULL;
  for (int i = 0; i < SV_MAX_QPS; ++i)
    if (F->qp[i].used && F->qp[i].device == dst && F->qp[i].qp_num == self->dest_qp_num) {
      peer = &F->qp[i];
      break;
    }
  if (!peer) {
    snprintf(reason, reason_len, "destination queue pair %u is not registered (destroyed, or never created)",
             self->dest_qp_num);
    return IBV_WC_RETRY_EXC_ERR;
  }
  if (peer->state != IBV_QPS_RTR && peer->state != IBV_QPS_RTS) {
    snprintf(reason, reason_len, "destination queue pair %u is in state %d", peer->qp_num, peer->state);
    return IBV_WC_RETRY_EXC_ERR;
  }
  if (peer->dest_qp_num != self->qp_num || memcmp(peer->dgid, F->dev[q->device].gid, 16) != 0) {
    snprintf(reason, reason_len, "destination queue pair %u is connected to queue pair %u of another endpoint",
             peer->qp_num, peer->dest_qp_num);
    return IBV_WC_RETRY_EXC_ERR;
  }
  const sv_mr *target = NULL;
  for (int i = 0; i < SV_MAX_MRS; ++i) {
    const sv_mr *m = &F->mr[i];
    if (m->used && m->device == dst && m->rkey == w->rkey) {
      target = m;
      break;
    }
  }
  if (!target) {
    snprintf(reason, reason_len, "no registered region of the destination device has rkey 0x%x", w->rkey);
    return IBV_WC_REM_ACCESS_ERR;
  }
  if (!(target->access & IBV_ACCESS_REMOTE_WRITE)) {
    snprintf(reason, reason_len, "the region of rkey 0x%x is not remote-writable", w->rkey);
    return IBV_WC_REM_ACCESS_ERR;
  }
  if (w->remote_addr < target->addr || w->remote_addr + w->length > target->addr + target->length) {
    snprintf(reason, reason_len, "%u bytes at 0x%llx lie outside the region of rkey 0x%x (0x%llx, %llu bytes)",
             w->length, (unsigned long long)w->remote_addr, w->rkey, (unsigned long long)target->addr,
             (unsigned long long)target->length);
    return IBV_WC_REM_ACCESS_ERR;
  }
  snprintf(segment, SV_NAME, "%s", target->segment);
  *seg_offset = target->seg_offset + (w->remote_addr - target->addr);
  if (w->is_inline) {
    *source = w->inline_data;
    return IBV_WC_SUCCESS;
  }
  for (int i = 0; i < SV_MAX_MRS; ++i) {
    const sv_mr *m = &F->mr[i];
    if (m->used && m->pid == (int)getpid() && m->device == q->device && m->lkey == w->lkey &&
        w->local_addr >= m->addr && w->local_addr + w->length <= m->addr + m->length) {
      *source = (const void *)(uintptr_t)w->local_addr;
      return IBV_WC_SUCCESS;
    }
  }
  snprintf(reason, reason_len, "no region of this process with lkey 0x%x holds the %u source bytes", w->lkey,
           w->length);
  return IBV_WC_LOC_PROT_ERR;
}

static void apply(uint8_t *destination, const void *source, uint32_t length) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (length == 4 && ((uintptr_t)destination & 3u) == 0) {
    uint32_t word;
    memcpy(&word, source, 4);
    __atomic_store_n((uint32_t *)destination, word, __ATOMIC_RELEASE);
  } else if (length) {
    memmove(destination, source, length);
  }
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/* Execute the oldest work request of one ready queue pair; 0 when none was ready. */
static int execute_one(void) {
  static int ready[SV_LOCAL_QPS];
  pthread_mutex_lock(&local);
  uint64_t now = latency_ns ? now_ns() : 0;
  int count = 0;
  for (int i = 0; i < SV_LOCAL_QPS; ++i)
    if (qps[i] && qps[i]->count > 0 && qps[i]->queue[qps[i]->head].ready_ns <= now) ready[count++] = i;
  if (!count) {
    pthread_mutex_unlock(&local);
    return 0;
  }
  int index = ready[next_random() % (uint64_t)count];
  sv_lqp *q = qps[index];
  sv_wr w = q->queue[q->head];
  q->head = (q->head + 1) % q->capacity;
  q->count--;
  uint32_t generation = q->generation;
  int errored = q->errored;
  sv_lqp snapshot = *q;
  pthread_mutex_unlock(&local);

  enum ibv_wc_status status = IBV_WC_WR_FLUSH_ERR;
  char segment[SV_NAME];
  char reason[192] = "flushed: an earlier write of this queue pair failed";
  int peer_pid = 0;
  uint64_t seg_offset = 0;
  const void *source = NULL;
  if (!errored) {
    fab_lock();
    status = resolve_write(&snapshot, &w, segment, &seg_offset, &source, reason, sizeof reason, &peer_pid);
    fab_unlock();
  }
  if (status == IBV_WC_SUCCESS) {
    uint64_t bytes = 0;
    uint8_t *base = mapped(segment, &bytes);
    if (!base || seg_offset + w.length > bytes) {
      status = IBV_WC_REM_ACCESS_ERR;
      snprintf(reason, sizeof reason, "destination segment %s is not mapped or is shorter than offset %llu + %u",
               segment, (unsigned long long)seg_offset, w.length);
    } else {
      apply(base + seg_offset, source, w.length);
    }
  }
  int peer_alive = status != IBV_WC_SUCCESS && peer_pid ? pid_alive(peer_pid) : 0;
  uint64_t when = status != IBV_WC_SUCCESS ? realtime_ns() : 0;
  __atomic_fetch_add(&executed, 1, __ATOMIC_RELAXED);
  pthread_mutex_lock(&local);
  int delivered = 0, wants = w.signaled || status != IBV_WC_SUCCESS;
  if (qps[index] == q && q->generation == generation) {
    if (status != IBV_WC_SUCCESS) q->errored = 1;
    q->executed++;
    if (wants) {
      delivered = complete(q, w.wr_id, status);
      q->completed += (uint64_t)delivered;
    }
  } else if (wants) {
    replaced_drops++;
  }
  if (status != IBV_WC_SUCCESS) {
    sv_error *r = &errors[error_count++ % SV_ERRORS];
    memset(r, 0, sizeof *r);
    r->ns = when;
    r->wr_id = w.wr_id;
    r->remote_addr = w.remote_addr;
    r->qp_num = snapshot.qp.qp_num;
    r->dest_qp_num = snapshot.dest_qp_num;
    r->rkey = w.rkey;
    r->length = w.length;
    r->status = (int32_t)status;
    r->signaled = w.signaled;
    r->is_inline = w.is_inline;
    r->peer_pid = peer_pid;
    r->peer_alive = peer_alive;
    r->delivered = delivered;
    snprintf(r->reason, sizeof r->reason, "%s", reason);
  }
  pthread_mutex_unlock(&local);
  return 1;
}

/* JSON of the diagnostics into out (at most len bytes, always terminated); returns the bytes needed. */
size_t sccl_emu_report(char *out, size_t len) {
  static const char *const states[] = {"reset", "init", "rtr", "rts", "sqd", "sqe", "err"};
  size_t used = 0;
#define EMIT(...)                                                                                       \
  do {                                                                                                  \
    int n_ = snprintf(out && used < len ? out + used : NULL, out && used < len ? len - used : 0, __VA_ARGS__); \
    if (n_ > 0) used += (size_t)n_;                                                                     \
  } while (0)
  pthread_mutex_lock(&local);
  EMIT("{\"executed\":%llu,\"cq_overflows\":%llu,\"replaced_drops\":%llu,\"errors_total\":%llu,"
       "\"queue_pairs\":[",
       (unsigned long long)__atomic_load_n(&executed, __ATOMIC_RELAXED), (unsigned long long)cq_overflows,
       (unsigned long long)replaced_drops, (unsigned long long)error_count);
  int first = 1;
  for (int i = 0; i < SV_LOCAL_QPS; ++i) {
    const sv_lqp *q = qps[i];
    if (!q) continue;
    unsigned state = (unsigned)q->qp.state;
    EMIT("%s{\"qp\":%u,\"dest\":%u,\"state\":\"%s\",\"queued\":%d,\"posted\":%llu,\"executed\":%llu,"
         "\"completed\":%llu,\"errored\":%d}",
         first ? "" : ",", q->qp.qp_num, q->dest_qp_num, state < 7 ? states[state] : "other", q->count,
         (unsigned long long)q->posted, (unsigned long long)q->executed, (unsigned long long)q->completed,
         q->errored);
    first = 0;
  }
  EMIT("],\"errors\":[");
  uint64_t kept = error_count < SV_ERRORS ? error_count : SV_ERRORS;
  for (uint64_t k = error_count - kept; k < error_count; ++k) {
    const sv_error *r = &errors[k % SV_ERRORS];
    EMIT("%s{\"ns\":%llu,\"qp\":%u,\"dest\":%u,\"peer_pid\":%d,\"peer_alive\":%d,\"wr_id\":%llu,"
         "\"signaled\":%d,\"inline\":%d,\"rkey\":%u,\"remote_addr\":%llu,\"length\":%u,\"status\":\"%s\","
         "\"reason\":\"%s\",\"delivered\":%d}",
         k == error_count - kept ? "" : ",", (unsigned long long)r->ns, r->qp_num, r->dest_qp_num, r->peer_pid,
         r->peer_alive, (unsigned long long)r->wr_id, r->signaled, r->is_inline, r->rkey,
         (unsigned long long)r->remote_addr, r->length, ibv_wc_status_str((enum ibv_wc_status)r->status),
         r->reason, r->delivered);
  }
  EMIT("]}");
  pthread_mutex_unlock(&local);
#undef EMIT
  return used + 1;
}

static void *executor_main(void *arg) {
  (void)arg;
  const struct timespec nap = {0, 10000};
  unsigned idle = 0;
  while (!executor_stop) {
    if (execute_one()) {
      idle = 0;
      continue;
    }
    if (++idle < 2000) {
      sched_yield();
    } else {
      nanosleep(&nap, NULL);
    }
  }
  return NULL;
}

/* -- verbs ------------------------------------------------------------------------------------- */

struct ibv_device **ibv_get_device_list(int *num_devices) {
  char err[160];
  if (sccl_emu_attach(err, sizeof err) != 0) {
    errno = ENODEV;
    if (num_devices) *num_devices = 0;
    return NULL;
  }
  struct ibv_device **list = calloc(SV_MAX_DEVICES + 1, sizeof *list);
  if (!list) {
    errno = ENOMEM;
    return NULL;
  }
  int n = 0;
  fab_lock();
  pthread_mutex_lock(&local);
  for (int i = 0; i < SV_MAX_DEVICES; ++i) {
    if (F->dev[i].used && F->dev[i].pid == (int)getpid()) {
      snprintf(devices[i].name, sizeof devices[i].name, "%s", F->dev[i].name);
      devices[i].fv_index = i;
      list[n++] = &devices[i];
    }
  }
  pthread_mutex_unlock(&local);
  fab_unlock();
  if (num_devices) *num_devices = n;
  return list;
}

void ibv_free_device_list(struct ibv_device **list) { free(list); }

const char *ibv_get_device_name(struct ibv_device *device) { return device->name; }

struct ibv_context *ibv_open_device(struct ibv_device *device) {
  struct ibv_context *context = calloc(1, sizeof *context);
  if (!context) {
    errno = ENOMEM;
    return NULL;
  }
  context->device = device;
  context->fv_index = device->fv_index;
  return context;
}

int ibv_close_device(struct ibv_context *context) {
  free(context);
  return 0;
}

int ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) {
  if (port_num != 1 || !port_attr) return EINVAL;
  memset(port_attr, 0, sizeof *port_attr);
  fab_lock();
  int failed = F->dev[context->fv_index].failed;
  fab_unlock();
  port_attr->state = failed ? IBV_PORT_DOWN : IBV_PORT_ACTIVE;
  port_attr->max_mtu = IBV_MTU_4096;
  port_attr->active_mtu = IBV_MTU_4096;
  port_attr->gid_tbl_len = 256;
  return 0;
}

int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid) {
  if (port_num != 1 || index < 0 || index > 255) return EINVAL;
  fab_lock();
  memcpy(gid->raw, F->dev[context->fv_index].gid, 16);
  fab_unlock();
  return 0;
}

struct ibv_pd *ibv_alloc_pd(struct ibv_context *context) {
  struct ibv_pd *pd = calloc(1, sizeof *pd);
  if (!pd) {
    errno = ENOMEM;
    return NULL;
  }
  pd->context = context;
  return pd;
}

int ibv_dealloc_pd(struct ibv_pd *pd) {
  free(pd);
  return 0;
}

struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
  pthread_mutex_lock(&segments_lock);
  int s = owned_segment((uint64_t)(uintptr_t)addr, length);
  char name[SV_NAME] = {0};
  uint64_t offset = 0;
  if (s >= 0) {
    snprintf(name, sizeof name, "%s", segments[s].name);
    offset = (uint64_t)(uintptr_t)addr - (uint64_t)(uintptr_t)segments[s].base;
  }
  pthread_mutex_unlock(&segments_lock);
  if (s < 0) {
    errno = EINVAL; /* only memory from sccl_emu_segment_alloc can be registered */
    return NULL;
  }
  struct ibv_mr *mr = calloc(1, sizeof *mr);
  if (!mr) {
    errno = ENOMEM;
    return NULL;
  }
  fab_lock();
  int slot = -1;
  for (int i = 0; i < SV_MAX_MRS; ++i)
    if (!F->mr[i].used) {
      slot = i;
      break;
    }
  if (slot >= 0) {
    sv_mr *m = &F->mr[slot];
    memset(m, 0, sizeof *m);
    m->used = 1;
    m->pid = (int)getpid();
    m->device = pd->context->fv_index;
    m->access = access;
    m->lkey = F->next_key++;
    m->rkey = F->next_key++;
    m->addr = (uint64_t)(uintptr_t)addr;
    m->length = length;
    m->seg_offset = offset;
    snprintf(m->segment, SV_NAME, "%s", name);
    mr->lkey = m->lkey;
    mr->rkey = m->rkey;
  }
  fab_unlock();
  if (slot < 0) {
    free(mr);
    errno = ENOMEM;
    return NULL;
  }
  mr->context = pd->context;
  mr->pd = pd;
  mr->addr = addr;
  mr->length = length;
  mr->handle = (uint32_t)slot;
  return mr;
}

/* SIRCL_EMU_FAIL_DEREG=1 (a test setting): every deregistration fails with EBUSY and keeps its region
 * registered, so a caller's handling of a failed release can be exercised. */
int ibv_dereg_mr(struct ibv_mr *mr) {
  const char *fail = getenv("SIRCL_EMU_FAIL_DEREG");
  if (fail && !strcmp(fail, "1")) return EBUSY;
  fab_lock();
  F->mr[mr->handle].used = 0;
  fab_unlock();
  free(mr);
  return 0;
}

struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context,
                             struct ibv_comp_channel *channel, int comp_vector) {
  (void)cq_context;
  (void)channel;
  (void)comp_vector;
  pthread_mutex_lock(&local);
  for (int i = 0; i < SV_LOCAL_CQS; ++i) {
    if (!cqs[i].used) {
      sv_cq *c = &cqs[i];
      memset(c, 0, sizeof *c);
      c->entries = calloc((size_t)cqe, sizeof(struct ibv_wc));
      if (!c->entries) break;
      c->used = 1;
      c->capacity = cqe;
      c->cq.context = context;
      c->cq.cqe = cqe;
      c->cq.fv_index = i;
      pthread_mutex_unlock(&local);
      return &c->cq;
    }
  }
  pthread_mutex_unlock(&local);
  errno = ENOMEM;
  return NULL;
}

int ibv_destroy_cq(struct ibv_cq *cq) {
  pthread_mutex_lock(&local);
  sv_cq *c = &cqs[cq->fv_index];
  free(c->entries);
  memset(c, 0, sizeof *c);
  pthread_mutex_unlock(&local);
  return 0;
}

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr) {
  if (attr->qp_type != IBV_QPT_RC || !attr->send_cq || attr->cap.max_send_wr == 0) {
    errno = EINVAL;
    return NULL;
  }
  sv_lqp *q = calloc(1, sizeof *q);
  if (q) q->queue = calloc(attr->cap.max_send_wr, sizeof(sv_wr));
  if (!q || !q->queue) {
    if (q) free(q);
    errno = ENOMEM;
    return NULL;
  }
  q->capacity = (int)attr->cap.max_send_wr;
  q->device = pd->context->fv_index;
  fab_lock();
  int shared = -1;
  for (int i = 0; i < SV_MAX_QPS; ++i)
    if (!F->qp[i].used) {
      shared = i;
      break;
    }
  if (shared >= 0) {
    sv_qp *r = &F->qp[shared];
    memset(r, 0, sizeof *r);
    r->used = 1;
    r->pid = (int)getpid();
    r->device = q->device;
    r->state = IBV_QPS_RESET;
    r->qp_num = F->next_qpn++;
    q->qp.qp_num = r->qp_num;
  }
  fab_unlock();
  pthread_mutex_lock(&local);
  int slot = -1;
  for (int i = 0; shared >= 0 && i < SV_LOCAL_QPS; ++i)
    if (!qps[i]) {
      slot = i;
      break;
    }
  if (slot < 0) {
    pthread_mutex_unlock(&local);
    if (shared >= 0) {
      fab_lock();
      F->qp[shared].used = 0;
      fab_unlock();
    }
    free(q->queue);
    free(q);
    errno = ENOMEM;
    return NULL;
  }
  q->shared = shared;
  q->generation = ++generations;
  q->qp.context = pd->context;
  q->qp.pd = pd;
  q->qp.send_cq = attr->send_cq;
  q->qp.recv_cq = attr->recv_cq;
  q->qp.state = IBV_QPS_RESET;
  q->qp.fv_index = slot;
  qps[slot] = q;
  nqps++;
  if (!executor_running) {
    executor_stop = 0;
    if (pthread_create(&executor, NULL, executor_main, NULL) == 0) executor_running = 1;
  }
  pthread_mutex_unlock(&local);
  return &q->qp;
}

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
  if (!(attr_mask & IBV_QP_STATE)) return EINVAL;
  pthread_mutex_lock(&local);
  sv_lqp *q = qps[qp->fv_index];
  int rc = 0;
  fab_lock();
  sv_qp *r = &F->qp[q->shared];
  switch (attr->qp_state) {
    case IBV_QPS_INIT:
      rc = r->state == IBV_QPS_RESET ? 0 : EINVAL;
      break;
    case IBV_QPS_RTR:
      if (r->state != IBV_QPS_INIT || !(attr_mask & IBV_QP_AV) || !(attr_mask & IBV_QP_DEST_QPN) ||
          !attr->ah_attr.is_global) {
        rc = EINVAL;
        break;
      }
      r->dest_qp_num = attr->dest_qp_num;
      q->dest_qp_num = attr->dest_qp_num;
      memcpy(r->dgid, attr->ah_attr.grh.dgid.raw, 16);
      break;
    case IBV_QPS_RTS:
      rc = r->state == IBV_QPS_RTR ? 0 : EINVAL;
      break;
    case IBV_QPS_ERR:
    case IBV_QPS_RESET:
      break;
    default:
      rc = EINVAL;
  }
  if (rc == 0) {
    r->state = (int)attr->qp_state;
    q->qp.state = attr->qp_state;
  }
  fab_unlock();
  pthread_mutex_unlock(&local);
  return rc;
}

int ibv_destroy_qp(struct ibv_qp *qp) {
  int join = 0;
  pthread_mutex_lock(&local);
  sv_lqp *q = qps[qp->fv_index];
  qps[qp->fv_index] = NULL;
  nqps--;
  if (nqps == 0 && executor_running) {
    executor_stop = 1;
    executor_running = 0;
    join = 1;
  }
  pthread_mutex_unlock(&local);
  if (join) pthread_join(executor, NULL);
  fab_lock();
  F->qp[q->shared].used = 0;
  fab_unlock();
  free(q->queue);
  free(q);
  return 0;
}

int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr) {
  pthread_mutex_lock(&local);
  sv_lqp *q = qps[qp->fv_index];
  for (struct ibv_send_wr *w = wr; w; w = w->next) {
    int rc = 0;
    if (q->qp.state != IBV_QPS_RTS || w->opcode != IBV_WR_RDMA_WRITE || w->num_sge != 1)
      rc = EINVAL;
    else if ((w->send_flags & IBV_SEND_INLINE) && w->sg_list[0].length > SV_INLINE)
      rc = EINVAL;
    else if (q->count == q->capacity)
      rc = ENOMEM;
    if (rc) {
      if (bad_wr) *bad_wr = w;
      pthread_mutex_unlock(&local);
      return rc;
    }
    sv_wr *s = &q->queue[(q->head + q->count) % q->capacity];
    memset(s, 0, sizeof *s);
    s->wr_id = w->wr_id;
    s->signaled = (w->send_flags & IBV_SEND_SIGNALED) != 0;
    s->is_inline = (w->send_flags & IBV_SEND_INLINE) != 0;
    s->local_addr = w->sg_list[0].addr;
    s->length = w->sg_list[0].length;
    s->lkey = w->sg_list[0].lkey;
    s->remote_addr = w->wr.rdma.remote_addr;
    s->rkey = w->wr.rdma.rkey;
    s->ready_ns = latency_ns ? now_ns() + latency_ns : 0;
    if (s->is_inline) memcpy(s->inline_data, (const void *)(uintptr_t)s->local_addr, s->length);
    q->count++;
    q->posted++;
  }
  pthread_mutex_unlock(&local);
  return 0;
}

int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
  pthread_mutex_lock(&local);
  sv_cq *c = &cqs[cq->fv_index];
  int n = 0;
  while (n < num_entries && c->count > 0) {
    wc[n++] = c->entries[c->head];
    c->head = (c->head + 1) % c->capacity;
    c->count--;
  }
  pthread_mutex_unlock(&local);
  return n;
}

const char *ibv_wc_status_str(enum ibv_wc_status status) {
  switch (status) {
    case IBV_WC_SUCCESS: return "success";
    case IBV_WC_LOC_PROT_ERR: return "local protection error";
    case IBV_WC_WR_FLUSH_ERR: return "Work Request Flushed Error";
    case IBV_WC_REM_ACCESS_ERR: return "remote access error";
    case IBV_WC_RETRY_EXC_ERR: return "transport retry counter exceeded";
    default: return "other error";
  }
}
