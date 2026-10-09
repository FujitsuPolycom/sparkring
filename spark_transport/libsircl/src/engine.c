/* SIRCL session engine (engine.h). */
#define _GNU_SOURCE
#include "env_names.h"
#include "engine.h"

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "kernelpack.h"
#include "transport/shm_verbs.h"

/* -- SIRCL's native proxy (src/transport/sircl_roce_proxy.c), compiled once per transport ----- */

#define PROXY_DECLARE(tag)                                                                              \
  int sccl_##tag##_roce_abi_version(void);                                                              \
  int sccl_##tag##_roce_layout(int, uint64_t, uint64_t *);                                              \
  uint64_t sccl_##tag##_roce_blob_bytes(void);                                                          \
  void *sccl_##tag##_roce_create(int, int, const char *const *, int, const int *, int, const int *, int, \
                                 void *, uint64_t, uint64_t, char *, uint64_t);                          \
  int sccl_##tag##_roce_local_blob(void *, void *, uint64_t);                                           \
  int sccl_##tag##_roce_connect(void *, const void *, uint64_t);                                        \
  int sccl_##tag##_roce_lane_check(void *, int);                                                        \
  int sccl_##tag##_roce_start(void *);                                                                  \
  void sccl_##tag##_roce_stop(void *);                                                                  \
  int sccl_##tag##_roce_failed(void *);                                                                 \
  const char *sccl_##tag##_roce_error(void *);                                                          \
  uint64_t sccl_##tag##_roce_stat(void *, int);                                                         \
  int sccl_##tag##_roce_destroy(void *);                                                                \
  int sccl_##tag##_roce_chain_layout(int, int, uint64_t, uint64_t *);                                   \
  int sccl_##tag##_roce_set_chain(void *, int, int, int, uint64_t, uint64_t);                         \
  int sccl_##tag##_roce_set_forward(void *, const uint32_t *, uint32_t);                                \
  int sccl_##tag##_roce_link_layout(int, int, uint64_t, uint64_t *);                                    \
  int sccl_##tag##_roce_set_links(void *, int, int, int, int, int, uint32_t, int, uint64_t, uint64_t);     \
  int sccl_##tag##_roce_set_trace(void *, uint32_t);                                                    \
  int64_t sccl_##tag##_roce_trace_take(void *, uint64_t *, uint64_t, uint64_t *);
PROXY_DECLARE(hw)
PROXY_DECLARE(emu)

typedef struct {
  const char *name;
  int (*abi_version)(void);
  int (*layout)(int, uint64_t, uint64_t *);
  uint64_t (*blob_bytes)(void);
  void *(*create)(int, int, const char *const *, int, const int *, int, const int *, int, void *, uint64_t,
                  uint64_t, char *, uint64_t);
  int (*local_blob)(void *, void *, uint64_t);
  int (*connect)(void *, const void *, uint64_t);
  int (*lane_check)(void *, int);
  int (*start)(void *);
  void (*stop)(void *);
  int (*failed)(void *);
  const char *(*error)(void *);
  uint64_t (*stat)(void *, int);
  /* Stops the progress thread and releases every verbs object; the number of verbs calls that failed. */
  int (*destroy)(void *);
  int (*chain_layout)(int, int, uint64_t, uint64_t *);
  int (*set_chain)(void *, int, int, int, uint64_t, uint64_t);
  int (*set_forward)(void *, const uint32_t *, uint32_t);
  int (*link_layout)(int, int, uint64_t, uint64_t *);
  int (*set_links)(void *, int, int, int, int, int, uint32_t, int, uint64_t, uint64_t);
  /* The native event trace (diagnostics): its capacity in records, set before the progress thread starts,
   * and taking the records written since the last take (two words each). */
  int (*set_trace)(void *, uint32_t);
  int64_t (*trace_take)(void *, uint64_t *, uint64_t, uint64_t *);
} transport_ops;

#define PROXY_OPS(tag, label)                                                                          \
  {label, sccl_##tag##_roce_abi_version, sccl_##tag##_roce_layout, sccl_##tag##_roce_blob_bytes,         \
   sccl_##tag##_roce_create, sccl_##tag##_roce_local_blob, sccl_##tag##_roce_connect,                    \
   sccl_##tag##_roce_lane_check, sccl_##tag##_roce_start, sccl_##tag##_roce_stop, sccl_##tag##_roce_failed, \
   sccl_##tag##_roce_error, sccl_##tag##_roce_stat, sccl_##tag##_roce_destroy,                           \
   sccl_##tag##_roce_chain_layout, sccl_##tag##_roce_set_chain, sccl_##tag##_roce_set_forward,           \
   sccl_##tag##_roce_link_layout, sccl_##tag##_roce_set_links, sccl_##tag##_roce_set_trace,               \
   sccl_##tag##_roce_trace_take}
static const transport_ops verbs_ops = PROXY_OPS(hw, "verbs");
static const transport_ops emulation_ops = PROXY_OPS(emu, "emulation");

int sccl_verbs_available(void);

/* -- protocol constants (SIRCL protocol.py) ---------------------------------------------------- */

enum {
  PACK = 16,
  SLOT_ALIGNMENT = 4096,
  CTRL_ERROR_SEQ = 2,
  CTRL_MISSING_PEER = 3,
  CTRL_MISSING_LANE = 6,
  CTRL_WAIT_LIMIT_US = 7,
  MAX_PHASES = 8,
  MAX_DEVICES = 4,
  MAX_LANES = 2,
  MAX_WORLD = 8,
  MAX_POSITIONS = 64,
  RECORD_MAGIC = 0x4343534cu, /* "LSCC" */
  RECORD_VERSION = 1,
};
#define DEFAULT_CAPACITY (2u << 20)
#define DEFAULT_LARGE_PIECE (4u << 20)
#define DEFAULT_ONESHOT_MAX 131072u
#define DEFAULT_SPIN_LIMIT 20000000u
#define DEFAULT_STARTUP_WAIT_S 600.0
#define DEFAULT_SERVING_WAIT_S 20.0
/* The chain schedule's defaults, SIRCL's (oneshot/runtime.py). */
#define DEFAULT_CHAIN_SLOTS 4u
#define DEFAULT_CHAIN_SLOT_BYTES (1u << 20)
#define DEFAULT_CHAIN_CHUNK_BYTES (512u << 10)
#define DEFAULT_CHAIN_BLOCKS 4u
#define DEFAULT_CHAIN_UNROLL 4u
/* The link collectives' defaults, SIRCL's: link slots 2 W (8 to 32), slots of 512 KiB that grow to the
 * largest configured piece up to 1 MiB, pieces of 512 KiB, 4 blocks per role, 4 packs per thread per pass,
 * staggers of one round when the slots hold them. */
#define DEFAULT_LINK_SLOT_BYTES (512u << 10)
#define MAX_AUTO_LINK_SLOT_BYTES (1u << 20)
#define DEFAULT_LINK_CHUNK_BYTES (512u << 10)
#define DEFAULT_LINK_BLOCKS 4u
#define DEFAULT_LINK_UNROLL 4u
#define MIN_DEFAULT_LINK_SLOTS 8u
#define LINK_MAX_SLOTS 32u
#define MAX_RING_STAGGER 4u
/* Bytes of every rank's chunk one link reduce-scatter op carries (a call's chunks are split into column
 * tiles of at most this size; each element's arithmetic does not depend on the split). */
#define DEFAULT_LINK_TILE_BYTES (16u << 20)
enum { SCHEDULE_PIECES = 0, SCHEDULE_CHAIN = 1, SCHEDULE_AUTO = 2, SCHEDULE_RING = 3 };
/* The collectives with minimums and pieces of their own: the all-reduce (message bytes), the all-gather
 * (output bytes) and the reduce-scatter (input bytes). */
enum { COLL_REDUCE = 0, COLL_GATHER = 1, COLL_SCATTER = 2, COLLECTIVES = 3 };
static const uint64_t default_chain_mins[COLLECTIVES] = {8u << 20, 8u << 20, 4u << 20};
/* The pair plan (RUNBOOK.md section 3.6 gives the measurement): a pair without SIRCL_LARGE_SCHEDULE that is
 * joined by cables, or given a ring plan, runs all-reduces from PAIR_RING_MIN_BYTES, all-gathers from
 * PAIR_GATHER_RING_MIN_BYTES of output (1 MiB shards) and reduce-scatters from PAIR_SCATTER_RING_MIN_BYTES of
 * input (link ops in column tiles) as one ring op each, with the pieces and blocks per
 * role of the plan's step for the call's size (all-reduce: message bytes; all-gather: shard bytes;
 * reduce-scatter: input bytes), one
 * block per role throughout. Below the minimums the one-shot and two-shot pieces stay (up to 1.5 MiB the
 * two-shot op is as fast or faster; from 2 MiB the ring is faster on buffers that change between calls). A
 * pair
 * joined through relays without a ring plan runs all-reduces from the chain minimum as one chain op. */
#define PAIR_RING_MIN_BYTES (2u << 20)
#define PAIR_GATHER_RING_MIN_BYTES (2u << 20)
#define PAIR_SCATTER_RING_MIN_BYTES (4u << 20)
typedef struct {
  uint64_t from;
  uint32_t piece, blocks;
} plan_step;
static const plan_step pair_reduce_plan[] = {{0, 256u << 10, 1}};
static const plan_step pair_gather_plan[] = {{0, 128u << 10, 1}, {2u << 20, 256u << 10, 1}, {16u << 20, 512u << 10, 1}};
/* Reduce-scatters by input bytes (every rank's chunk). */
static const plan_step pair_scatter_plan[] = {{0, 256u << 10, 1}, {64u << 20, 512u << 10, 1}};
static const uint64_t default_ring_mins[COLLECTIVES] = {4u << 20, 8u << 20, 4u << 20};
static const char *const link_kind_names[SCCL_LINK_KINDS] = {"chain_gather", "chain_scatter", "ring_gather",
                                                             "ring_scatter", "ring_reduce"};
static const unsigned link_roles[SCCL_LINK_KINDS] = {4, 2, 2, 2, 3};

enum { ALG_AUTO = -1 };
/* LIBSIRCL_FAIL_STOP's modes (see fail_stop). */
enum { FAIL_STOP_EXIT = 1, FAIL_STOP_ABORT = 2 };

/* Settings every rank must share; compared field by field at setup. */
typedef struct {
  uint32_t magic, version, world, lanes;
  uint64_t slot_bytes, capacity, large_piece, oneshot_max;
  uint32_t threads, blocks, large_blocks, packs_per_thread;
  uint32_t spin_limit, one_block, algorithm_plus_one, transport;
  /* graph_staging: a call under CUDA graph capture that needs more staging than the communicator holds
   * takes a graph allocation (the driver and device support stream-ordered allocation). Agreed at setup,
   * so every rank stages captured calls the same way. */
  uint32_t startup_wait_us, serving_wait_us, proxy_abi, graph_staging;
  char kernel_pack[72], fold_pack[72], links_pack[72];
  /* The chain schedule: SIRCL_LARGE_SCHEDULE, geometry, and the chain order (ranks by chain index). */
  uint32_t large_schedule, chain_slots, chain_chunk, chain_blocks, chain_unroll, chain_reserved;
  uint64_t chain_slot_bytes;
  /* Smallest collective (COLL_*) that auto runs as a chain op, and that a ring schedule runs as a ring op. */
  uint64_t chain_mins[COLLECTIVES], ring_mins[COLLECTIVES];
  int32_t chain_order[8];
  /* The link collectives: schedules of the all-gather and the reduce-scatter, link geometry, pieces per
   * collective (COLL_*), staggers, whether the ring that closes the chain runs (LIBSIRCL_RING_WINDOW
   * set), the reduce-scatter's column tile. */
  uint32_t gather_schedule, scatter_schedule, link_slots, link_blocks, link_unroll, ring_on;
  uint32_t ring_stagger, ring_gather_stagger, link_chunk[COLLECTIVES], link_reserved;
  uint64_t link_slot_bytes, link_tile;
  /* Blocks per role and pieces per collective: chunk_set[c] when a link chunk variable sets the piece,
   * coll_blocks[c] from LIBSIRCL_<collective>_LINK_BLOCKS (0: unset), blocks_set when SIRCL_LINK_BLOCKS
   * sets link_blocks; pair_plan when the pair plan chooses the rest by size. */
  uint32_t chunk_set[COLLECTIVES], coll_blocks[COLLECTIVES], blocks_set, pair_plan;
} shared_settings;

typedef struct {
  char error[400];
  shared_settings settings;
} setup_record;

typedef struct {
  uint32_t ok;
  char message[300];
} verdict_record;

struct sccl_engine {
  int world, rank, lanes, device, position;
  int positions[MAX_WORLD];
  /* The chain schedule: on when chain_on; this rank's chain index and neighbors (-1 at an end), the
   * chain area's offset in the arena and the chain op counters (4 device words). */
  int chain_on, chain_index, chain_prev, chain_next;
  /* Forward windows (bytes in flight per lane toward each rank through relays; 0: direct) and chunk. */
  uint32_t forward[MAX_WORLD][MAX_LANES], forward_chunk, forward_lanes;
  uint64_t chain_off;
  sccl_CUdeviceptr chain_counters;
  sccl_CUfunction chain_fn[SCCL_DT_COUNT];
  /* The link collectives: on when link_on; the ring's neighbors and this rank's ring window; the link area's
   * offset, its counters (SCCL_LINK_COUNTER_WORDS device words), piece counters and decision words, the
   * reduce-scatter's partial scratch (one tile), and the staging buffers of unaligned link and chain ops
   * (grown outside graph capture; earlier ones are kept until destroy, since queued work and captured
   * graphs may still use them). */
  int link_on, ring_prev, ring_next;
  uint32_t ring_window;
  /* The CPUs the transport's progress thread may run on (a CPU list, or "unpinned"). */
  char progress_cpus[160];
  /* The ring all-reduce's form (LIBSIRCL_RING_REDUCE_PASSES): 2 (default), the relay stores its result to
   * link 3's slot, then copies the slot to the output; 1, it stores both in one pass. Rank-local: both forms
   * speak the same protocol and write the same bytes. */
  int ring_reduce_passes;
  uint64_t link_off;
  sccl_CUdeviceptr link_counters, piece_counters, link_scratch;
  sccl_CUdeviceptr stage, *retired_stage;
  uint64_t stage_bytes;
  int retired, retired_room;
  /* The graph allocation of the call in progress under capture (0: none), its bytes and the capturing
   * stream; freed on that stream at the call's end, so the graph owns it. */
  sccl_CUdeviceptr call_stage;
  uint64_t call_stage_bytes;
  sccl_CUstream call_stage_stream;
  sccl_CUfunction link_fn[SCCL_LINK_KINDS][SCCL_DT_COUNT];
  /* The link area's layout (the native layer's offsets: receive, own, flag lines, ready, consumed, sent,
   * credit, control, total) and whether its timed-out dump was written (LIBSIRCL_LINK_DUMP). */
  uint64_t link_layout[9];
  _Atomic int link_dumped;
  /* Link ops (pair exchanges included) by blocks per role, 1 to 64. */
  _Atomic uint64_t ops_by_blocks[65];
  /* The pair exchange entry (sircl_ring_exchange) and its op and byte counts. */
  sccl_CUfunction exchange_fn, exchange_reduce_fn[SCCL_DT_COUNT];
  _Atomic uint64_t exchange_ops, exchange_bytes;
  /* LIBSIRCL_LINK_BLOCKS_CYCLE (a test hook): blocks per role of successive link ops, cycling. */
  uint32_t blocks_cycle[16];
  int blocks_cycle_n;
  _Atomic uint64_t blocks_cycle_next;
  sccl_CUcontext ctx;
  const transport_ops *transport;
  void *proxy;
  /* arena */
  uint8_t *host;
  uint64_t region_bytes;
  sccl_CUdeviceptr dev_base;
  int host_registered, host_emulated;
  uint64_t slot_bytes, recv_off, flag_off, send_off, ctrl_off;
  volatile uint32_t *ctrl;
  /* device words: counters (epoch, stage, tail, poison, phase arrivals) and the arrival pair; scratch[0]
   * holds one slot (staged inputs, unaligned outputs), scratch[1] holds W slots (gathered rows) */
  sccl_CUdeviceptr counters, arrival, scratch[2];
  int classes;
  /* settings */
  shared_settings shared;
  int algorithm;
  sccl_CUfunction fn[SCCL_K_COUNT][SCCL_DT_COUNT];
  sccl_CUfunction fold[SCCL_FOLD_DTYPES][SCCL_FOLD_OPS];
  /* per-call state, under lock */
  pthread_mutex_t lock;
  sccl_CUstream last_stream;
  /* Teardown: the progress thread stopped (sccl_engine_teardown_stop), when its stop was requested and
   * when it had stopped (CLOCK_REALTIME nanoseconds). */
  int stopped;
  uint64_t stop_ns, stopped_ns;
  int have_last;
  sccl_CUevent order_event;
  unsigned long long capture_id;
  sccl_CUstream capture_stream;
  int serving;
  /* Fail-stop (LIBSIRCL_FAIL_STOP): FAIL_STOP_EXIT or FAIL_STOP_ABORT when this communicator is on the
   * watcher's list, and the next one on it. */
  int fail_stop;
  struct sccl_engine *watch_next;
  /* receipts */
  _Atomic uint64_t ops[SCCL_ALG_COUNT][SCCL_DT_COUNT];
  _Atomic uint64_t gather_calls, gather_ops, gather_bytes, gather_padded;
  _Atomic uint64_t scatter_calls, scatter_ops, scatter_bytes, scatter_padded;
  _Atomic uint64_t broadcast_calls, reduce_calls;
  _Atomic uint64_t alltoall_calls, alltoall_ops, alltoall_bytes, alltoall_padded, gather_root_calls,
      scatter_root_calls;
  _Atomic uint64_t folds[SCCL_FOLD_DTYPES][SCCL_FOLD_OPS];
  _Atomic uint64_t p2p_sends, p2p_recvs, p2p_exchanges, p2p_ops, p2p_bytes_sent, p2p_bytes_received;
  _Atomic uint64_t chain_ops[SCCL_DT_COUNT], chain_bytes;
  _Atomic uint64_t link_ops[SCCL_LINK_KINDS], link_bytes, link_staged, graph_staged;
  _Atomic uint64_t calls, captured_calls, bytes, padded_tails, staged_unaligned, local_copies;
  _Atomic uint64_t refused_op, refused_capture;
  /* Receipt file: this communicator's number among the process's communicators (in the file name), the
   * refresh interval (LIBSIRCL_RECEIPT_INTERVAL_S; 0: only at creation and destroy) and the next refresh. */
  int receipt_id;
  uint64_t receipt_interval_ns, receipt_next_ns;
};

static const sccl_cuda *cu;

static void put_error(char *err, size_t len, const char *format, ...) {
  if (!err || !len) return;
  va_list args;
  va_start(args, format);
  vsnprintf(err, len, format, args);
  va_end(args);
}

static uint64_t round_up(uint64_t value, uint64_t to) { return (value + to - 1) / to * to; }

sccl_CUcontext sccl_engine_current_context(void) {
  cu = sccl_cuda_get();
  sccl_CUcontext ctx = NULL;
  if (cu && cu->CtxGetCurrent(&ctx) != SCCL_CUDA_SUCCESS) ctx = NULL;
  return ctx;
}

/* -- settings -------------------------------------------------------------------------------- */

static int env_u64(const char *name, uint64_t fallback, uint64_t lo, uint64_t hi, uint64_t *out, char *err,
                   size_t len) {
  const char *text = sccl_env(name);
  if (!text || !*text) {
    *out = fallback;
    return 0;
  }
  char *end;
  errno = 0;
  unsigned long long value = strtoull(text, &end, 10);
  if (errno || *end || value < lo || value > hi) {
    put_error(err, len, "%s=%s must be an integer from %" PRIu64 " to %" PRIu64, name, text, lo, hi);
    return -1;
  }
  *out = value;
  return 0;
}

static int env_seconds_us(const char *name, double fallback, uint32_t *out, char *err, size_t len) {
  const char *text = sccl_env(name);
  double seconds = fallback;
  if (text && *text) {
    char *end;
    seconds = strtod(text, &end);
    if (*end || !(seconds >= 1e-6 && seconds <= 4294.0)) {
      put_error(err, len, "%s=%s must be between 1e-6 and 4294 seconds", name, text);
      return -1;
    }
  }
  double micros = seconds * 1e6 + 0.5;
  *out = micros >= 4294967295.0 ? 0xFFFFFFFFu : micros < 1 ? 1u : (uint32_t)micros;
  return 0;
}

static int is_power_of_two(uint64_t v) { return v && !(v & (v - 1)); }

/* A schedule variable: pieces (unset), chain, auto or ring. */
static int schedule_setting(const char *name, uint32_t *out, char *err, size_t len) {
  const char *text = sccl_env(name);
  if (!text || !*text || !strcmp(text, "pieces")) {
    *out = SCHEDULE_PIECES;
  } else if (!strcmp(text, "chain")) {
    *out = SCHEDULE_CHAIN;
  } else if (!strcmp(text, "auto")) {
    *out = SCHEDULE_AUTO;
  } else if (!strcmp(text, "ring")) {
    *out = SCHEDULE_RING;
  } else {
    put_error(err, len, "%s=%s: one of pieces, chain, auto, ring", name, text);
    return -1;
  }
  return 0;
}

/* Per-collective minimums: the defaults, or the variable's value for every collective (SIRCL's rule). */
static int minimum_setting(const char *name, const uint64_t *defaults, uint64_t *out, char *err, size_t len) {
  uint64_t v;
  const char *text = sccl_env(name);
  for (int c = 0; c < COLLECTIVES; ++c) out[c] = defaults[c];
  if (!text || !*text) return 0;
  if (env_u64(name, 0, 0, UINT64_MAX >> 1, &v, err, len)) return -1;
  for (int c = 0; c < COLLECTIVES; ++c) out[c] = v;
  return 0;
}

/* Link slots a staggered ring link needs (SIRCL's protocol.ring_stagger_slots). */
static uint32_t stagger_slots(int world, uint32_t stagger) {
  return stagger ? stagger * (uint32_t)(world - 1) + 2u : 2u;
}

/* A ring stagger variable: auto (unset) gives one round when the link slots hold it, else 0. */
static int stagger_setting(const char *name, const shared_settings *s, int world, uint32_t *out, char *err,
                           size_t len) {
  const char *text = sccl_env(name);
  if (!text || !*text || !strcmp(text, "auto")) {
    *out = s->link_slots >= stagger_slots(world, 1) ? 1u : 0u;
    return 0;
  }
  uint64_t v;
  if (env_u64(name, 0, 0, MAX_RING_STAGGER, &v, err, len)) return -1;
  if (s->link_slots < stagger_slots(world, (uint32_t)v)) {
    put_error(err, len, "%s=%s needs %u link slots (SIRCL_LINK_SLOTS), the session has %u", name, text,
              stagger_slots(world, (uint32_t)v), s->link_slots);
    return -1;
  }
  *out = (uint32_t)v;
  return 0;
}

/* The link collectives' settings (SIRCL's names and defaults) and the ring plan. `pair_plan`: the pair
 * plan applies (see PAIR_RING_MIN_BYTES), which runs the ring without LIBSIRCL_RING_WINDOW (window 0,
 * cables both ways). */
static int link_settings(sccl_engine *e, shared_settings *s, int pair_plan, char *err, size_t len) {
  static const char *const piece_names[COLLECTIVES] = {"SIRCL_REDUCE_LINK_CHUNK_BYTES",
                                                       "SIRCL_GATHER_LINK_CHUNK_BYTES",
                                                       "SIRCL_SCATTER_LINK_CHUNK_BYTES"};
  uint64_t v, configured[COLLECTIVES], session_piece, wanted = 0;
  uint32_t slots_default = 2u * (uint32_t)e->world;
  if (slots_default < MIN_DEFAULT_LINK_SLOTS) slots_default = MIN_DEFAULT_LINK_SLOTS;
  if (slots_default > LINK_MAX_SLOTS) slots_default = LINK_MAX_SLOTS;
  if (env_u64("SIRCL_LINK_SLOTS", slots_default, 2, LINK_MAX_SLOTS, &v, err, len)) return -1;
  s->link_slots = (uint32_t)v;
  if (env_u64("SIRCL_LINK_CHUNK_BYTES", 0, 0, 1u << 31, &session_piece, err, len)) return -1;
  wanted = session_piece;
  for (int c = 0; c < COLLECTIVES; ++c) {
    if (env_u64(piece_names[c], 0, 0, 1u << 31, &configured[c], err, len)) return -1;
    if (configured[c] > wanted) wanted = configured[c];
  }
  /* The slot holds the largest configured piece: rounded up to 4096 bytes, at least the default and,
   * unless SIRCL_LINK_SLOT_BYTES sets it, at most 1 MiB. */
  wanted = round_up(wanted, SLOT_ALIGNMENT);
  uint64_t auto_slot = wanted > DEFAULT_LINK_SLOT_BYTES && wanted <= MAX_AUTO_LINK_SLOT_BYTES ? wanted
                                                                                              : DEFAULT_LINK_SLOT_BYTES;
  if (env_u64("SIRCL_LINK_SLOT_BYTES", auto_slot, SLOT_ALIGNMENT, 1u << 31, &v, err, len)) return -1;
  if (v % SLOT_ALIGNMENT) {
    put_error(err, len, "SIRCL_LINK_SLOT_BYTES must be a multiple of %d bytes", SLOT_ALIGNMENT);
    return -1;
  }
  s->link_slot_bytes = v;
  int session_configured = session_piece != 0;
  if (!session_piece) session_piece = DEFAULT_LINK_CHUNK_BYTES < v ? DEFAULT_LINK_CHUNK_BYTES : v;
  for (int c = 0; c < COLLECTIVES; ++c) {
    uint64_t piece = configured[c] ? configured[c] : session_piece;
    s->chunk_set[c] = configured[c] || session_configured;
    if (piece < PACK || piece % PACK || piece > s->link_slot_bytes) {
      put_error(err, len, "%s of %" PRIu64 " bytes must be a multiple of 16 up to the link slot of %" PRIu64 " bytes",
                configured[c] ? piece_names[c] : "SIRCL_LINK_CHUNK_BYTES", piece, s->link_slot_bytes);
      return -1;
    }
    s->link_chunk[c] = (uint32_t)piece;
  }
  if (env_u64("SIRCL_LINK_BLOCKS", DEFAULT_LINK_BLOCKS, 1, 64, &v, err, len)) return -1;
  s->link_blocks = (uint32_t)v;
  const char *blocks = sccl_env("SIRCL_LINK_BLOCKS");
  s->blocks_set = blocks && *blocks;
  static const char *const block_names[COLLECTIVES] = {"LIBSIRCL_REDUCE_LINK_BLOCKS", "LIBSIRCL_GATHER_LINK_BLOCKS",
                                                       "LIBSIRCL_SCATTER_LINK_BLOCKS"};
  static const char *const sircl_block_names[COLLECTIVES] = {"SIRCL_REDUCE_LINK_BLOCKS", "SIRCL_GATHER_LINK_BLOCKS",
                                                             "SIRCL_SCATTER_LINK_BLOCKS"};
  for (int c = 0; c < COLLECTIVES; ++c) {
    /* LIBSIRCL_<collective>_LINK_BLOCKS, else SIRCL's SIRCL_<collective>_LINK_BLOCKS. */
    const char *own = sccl_env(block_names[c]);
    if (env_u64(own && *own ? block_names[c] : sircl_block_names[c], 0, 0, 64, &v, err, len)) return -1;
    s->coll_blocks[c] = (uint32_t)v;
  }
  /* A test hook: LIBSIRCL_LINK_BLOCKS_CYCLE=<n>[,<n>...] gives successive link ops these blocks per role in
   * turn (every rank issues the same ops, so every rank cycles alike), to exercise launches of one kernel
   * type with different grids. */
  e->blocks_cycle_n = 0;
  const char *cycle = sccl_env("LIBSIRCL_LINK_BLOCKS_CYCLE");
  if (cycle && *cycle) {
    char copy[160];
    snprintf(copy, sizeof copy, "%s", cycle);
    for (char *save = NULL, *item = strtok_r(copy, ",", &save); item; item = strtok_r(NULL, ",", &save)) {
      char *end;
      long n = strtol(item, &end, 10);
      if (*end || n < 1 || n > 64 || e->blocks_cycle_n == 16) {
        put_error(err, len, "LIBSIRCL_LINK_BLOCKS_CYCLE=%s: up to 16 block counts of 1 to 64", cycle);
        return -1;
      }
      e->blocks_cycle[e->blocks_cycle_n++] = (uint32_t)n;
    }
  }
  s->pair_plan = (uint32_t)pair_plan;
  if (env_u64("SIRCL_LINK_UNROLL", DEFAULT_LINK_UNROLL, 1, SCCL_CHAIN_MAX_UNROLL, &v, err, len)) return -1;
  s->link_unroll = (uint32_t)v;
  if (stagger_setting("SIRCL_RING_STAGGER", s, e->world, &s->ring_stagger, err, len) ||
      stagger_setting("SIRCL_RING_GATHER_STAGGER", s, e->world, &s->ring_gather_stagger, err, len))
    return -1;
  if (env_u64("LIBSIRCL_LINK_TILE_BYTES", DEFAULT_LINK_TILE_BYTES, PACK, 1u << 30, &v, err, len)) return -1;
  if (v % PACK || (v + s->link_chunk[COLL_SCATTER] - 1) / s->link_chunk[COLL_SCATTER] > SCCL_LINK_PIECE_COUNTERS) {
    put_error(err, len, "LIBSIRCL_LINK_TILE_BYTES must be a multiple of 16 of at most %d reduce-scatter pieces",
              SCCL_LINK_PIECE_COUNTERS);
    return -1;
  }
  s->link_tile = v;
  /* The ring that closes the chain runs when the layout's ring plan says so (SIRCL's routes.ring_window,
   * printed by tools/site_routes.py): LIBSIRCL_RING_WINDOW is this rank's ring window, the bytes each lane
   * toward the next rank of the ring keeps unacknowledged through relays (0: a cable). */
  const char *ring = sccl_env("LIBSIRCL_RING_WINDOW");
  s->ring_on = (ring && *ring) || pair_plan;
  e->ring_window = 0;
  if (ring && *ring) {
    if (env_u64("LIBSIRCL_RING_WINDOW", 0, 0, 0xFFFFFFFFu, &v, err, len)) return -1;
    e->ring_window = (uint32_t)v;
  }
  return 0;
}

/* Whether LIBSIRCL_FORWARD_WINDOWS gives a nonzero window on some lane toward rank `peer` (the peer is
 * reached through relays). A malformed value counts as none here; forward_windows() reports it. */
static int relayed_toward(const sccl_engine *e, int peer) {
  const char *text = sccl_env("LIBSIRCL_FORWARD_WINDOWS");
  if (!text || !*text) return 0;
  char copy[1024];
  snprintf(copy, sizeof copy, "%s", text);
  for (char *save = NULL, *entry = strtok_r(copy, ",", &save); entry; entry = strtok_r(NULL, ",", &save)) {
    char *equals = strchr(entry, '='), *end;
    if (!equals || strtol(entry, &end, 10) != e->positions[peer] || end != equals) continue;
    for (char *save2 = NULL, *value = strtok_r(equals + 1, "/", &save2); value; value = strtok_r(NULL, "/", &save2))
      if (strtoul(value, NULL, 10)) return 1;
  }
  return 0;
}

static int read_settings(sccl_engine *e, int emulation, char *err, size_t len) {
  shared_settings *s = &e->shared;
  uint64_t v;
  memset(s, 0, sizeof *s);
  s->magic = RECORD_MAGIC;
  s->version = RECORD_VERSION;
  s->world = (uint32_t)e->world;
  s->transport = (uint32_t)emulation;
  if (env_u64("LIBSIRCL_MAX_SIZE", DEFAULT_CAPACITY, PACK, 1u << 29, &v, err, len)) return -1;
  if (v % PACK) {
    put_error(err, len, "LIBSIRCL_MAX_SIZE must be a multiple of 16 bytes");
    return -1;
  }
  s->capacity = v;
  uint64_t piece_default = s->capacity > DEFAULT_LARGE_PIECE ? s->capacity : DEFAULT_LARGE_PIECE;
  if (env_u64("SIRCL_LARGE_PIECE_BYTES", piece_default, PACK, (1u << 30) - 1, &v, err, len)) return -1;
  if (v % PACK) {
    put_error(err, len, "SIRCL_LARGE_PIECE_BYTES must be a multiple of 16 bytes");
    return -1;
  }
  s->large_piece = v;
  s->slot_bytes = round_up(s->capacity > s->large_piece ? s->capacity : s->large_piece, SLOT_ALIGNMENT);
  if (env_u64("SIRCL_ONESHOT_MAX_BYTES", DEFAULT_ONESHOT_MAX, 0, 1u << 30, &v, err, len)) return -1;
  s->oneshot_max = v;
  if (env_u64("SIRCL_THREADS", 512, 32, 1024, &v, err, len)) return -1;
  if (v % 32) {
    put_error(err, len, "SIRCL_THREADS must be a multiple of 32 between 32 and 1024");
    return -1;
  }
  s->threads = (uint32_t)v;
  if (env_u64("SIRCL_BLOCKS", 8, 1, 1024, &v, err, len)) return -1;
  s->blocks = (uint32_t)v;
  if (env_u64("SIRCL_LARGE_BLOCKS", 32, 1, 1024, &v, err, len)) return -1;
  s->large_blocks = (uint32_t)v;
  if (!is_power_of_two(s->blocks) || !is_power_of_two(s->large_blocks)) {
    put_error(err, len, "SIRCL_BLOCKS and SIRCL_LARGE_BLOCKS must be powers of two");
    return -1;
  }
  if (env_u64("SIRCL_PACKS_PER_THREAD", 2, 1, 64, &v, err, len)) return -1;
  s->packs_per_thread = (uint32_t)v;
  if (env_u64("SIRCL_SPIN_LIMIT", DEFAULT_SPIN_LIMIT, 1, 0xFFFFFFFFu, &v, err, len)) return -1;
  s->spin_limit = (uint32_t)v;
  const char *pollers = sccl_env("SIRCL_FLAG_POLLERS");
  if (!pollers || !*pollers || !strcmp(pollers, "one-block")) {
    s->one_block = 1;
  } else if (!strcmp(pollers, "every-block")) {
    s->one_block = 0;
  } else {
    put_error(err, len, "SIRCL_FLAG_POLLERS=%s: one of one-block, every-block", pollers);
    return -1;
  }
  const char *algorithm = sccl_env("SIRCL_ALLREDUCE_ALGORITHM");
  if (!algorithm || !*algorithm || !strcmp(algorithm, "auto")) {
    e->algorithm = ALG_AUTO;
  } else if (!strcmp(algorithm, "oneshot")) {
    e->algorithm = SCCL_ALG_ONESHOT;
  } else if (!strcmp(algorithm, "twoshot")) {
    e->algorithm = SCCL_ALG_TWOSHOT;
  } else {
    put_error(err, len, "SIRCL_ALLREDUCE_ALGORITHM=%s: one of auto, oneshot, twoshot", algorithm);
    return -1;
  }
  s->algorithm_plus_one = (uint32_t)(e->algorithm + 1);
  if (env_seconds_us("SIRCL_STARTUP_WAIT_S", DEFAULT_STARTUP_WAIT_S, &s->startup_wait_us, err, len) ||
      env_seconds_us("SIRCL_SERVING_WAIT_S", DEFAULT_SERVING_WAIT_S, &s->serving_wait_us, err, len))
    return -1;
  const char *regime = sccl_env("LIBSIRCL_WAIT_REGIME");
  if (regime && *regime && strcmp(regime, "startup") && strcmp(regime, "serving")) {
    put_error(err, len, "LIBSIRCL_WAIT_REGIME=%s: startup or serving", regime);
    return -1;
  }
  e->serving = regime && !strcmp(regime, "serving");
  const char *stop = sccl_env("LIBSIRCL_FAIL_STOP");
  if (stop && *stop && strcmp(stop, "0") && strcmp(stop, "1") && strcmp(stop, "abort")) {
    put_error(err, len, "LIBSIRCL_FAIL_STOP=%s: 0, 1 or abort", stop);
    return -1;
  }
  e->fail_stop = !stop || !*stop || !strcmp(stop, "0") ? 0 : !strcmp(stop, "1") ? FAIL_STOP_EXIT : FAIL_STOP_ABORT;
  if (env_u64("LIBSIRCL_RING_REDUCE_PASSES", 2, 1, 2, &v, err, len)) return -1;
  e->ring_reduce_passes = (int)v;
  const char *policy = sccl_env("LIBSIRCL_CPU_POLICY");
  if (policy && *policy && strcmp(policy, "performance") && strcmp(policy, "none")) {
    put_error(err, len, "LIBSIRCL_CPU_POLICY=%s: performance or none", policy);
    return -1;
  }
  snprintf(s->kernel_pack, sizeof s->kernel_pack, "%s", sccl_kp_hash());
  snprintf(s->fold_pack, sizeof s->fold_pack, "%s", sccl_kp_fold_hash());
  snprintf(s->links_pack, sizeof s->links_pack, "%s", sccl_kp_links_hash());
  /* The chain schedule (SIRCL's settings and defaults); the pieces schedule unless chosen. */
  if (schedule_setting("SIRCL_LARGE_SCHEDULE", &s->large_schedule, err, len) ||
      schedule_setting("SIRCL_GATHER_SCHEDULE", &s->gather_schedule, err, len) ||
      schedule_setting("SIRCL_SCATTER_SCHEDULE", &s->scatter_schedule, err, len))
    return -1;
  if (env_u64("SIRCL_CHAIN_SLOTS", DEFAULT_CHAIN_SLOTS, 2, 32, &v, err, len)) return -1;
  s->chain_slots = (uint32_t)v;
  if (env_u64("SIRCL_CHAIN_SLOT_BYTES", DEFAULT_CHAIN_SLOT_BYTES, SLOT_ALIGNMENT, 1u << 31, &v, err, len)) return -1;
  if (v % SLOT_ALIGNMENT) {
    put_error(err, len, "SIRCL_CHAIN_SLOT_BYTES must be a multiple of %d bytes", SLOT_ALIGNMENT);
    return -1;
  }
  s->chain_slot_bytes = v;
  uint64_t chunk_default = DEFAULT_CHAIN_CHUNK_BYTES < s->chain_slot_bytes ? DEFAULT_CHAIN_CHUNK_BYTES
                                                                           : s->chain_slot_bytes;
  if (env_u64("SIRCL_CHAIN_CHUNK_BYTES", chunk_default, PACK, s->chain_slot_bytes, &v, err, len)) return -1;
  if (v % PACK) {
    put_error(err, len, "SIRCL_CHAIN_CHUNK_BYTES must be a multiple of 16 bytes");
    return -1;
  }
  s->chain_chunk = (uint32_t)v;
  if (env_u64("SIRCL_CHAIN_BLOCKS", DEFAULT_CHAIN_BLOCKS, 1, 64, &v, err, len)) return -1;
  s->chain_blocks = (uint32_t)v;
  if (env_u64("SIRCL_CHAIN_UNROLL", DEFAULT_CHAIN_UNROLL, 1, SCCL_CHAIN_MAX_UNROLL, &v, err, len)) return -1;
  s->chain_unroll = (uint32_t)v;
  if (minimum_setting("SIRCL_CHAIN_MIN_BYTES", default_chain_mins, s->chain_mins, err, len) ||
      minimum_setting("SIRCL_RING_MIN_BYTES", default_ring_mins, s->ring_mins, err, len))
    return -1;
  /* Pairs without SIRCL_LARGE_SCHEDULE (see PAIR_RING_MIN_BYTES): a cabled pair, or a relayed one given a
   * ring plan, runs the ring schedule from its pair minimums; a relayed pair without one runs the chain
   * from the chain minimum. On two ranks the ring and the chain add the same two values once, so their bits
   * equal the pieces'. Every variable set in the environment keeps its value; larger groups keep the
   * pieces schedule unless configured. */
  int pair_plan = 0;
  const char *large = sccl_env("SIRCL_LARGE_SCHEDULE");
  if ((!large || !*large) && e->world == 2 && s->threads >= SCCL_LINK_MIN_THREADS) {
    const char *ring = sccl_env("LIBSIRCL_RING_WINDOW"), *ring_min = sccl_env("SIRCL_RING_MIN_BYTES");
    const char *gather = sccl_env("SIRCL_GATHER_SCHEDULE"), *scatter = sccl_env("SIRCL_SCATTER_SCHEDULE");
    const char *chain_min = sccl_env("SIRCL_CHAIN_MIN_BYTES");
    if ((ring && *ring) || !relayed_toward(e, 1 - e->rank)) {
      pair_plan = 1;
      s->large_schedule = SCHEDULE_RING;
      if (!gather || !*gather) s->gather_schedule = SCHEDULE_RING;
      if (!scatter || !*scatter) s->scatter_schedule = SCHEDULE_RING;
      if (!ring_min || !*ring_min) {
        s->ring_mins[COLL_REDUCE] = PAIR_RING_MIN_BYTES;
        s->ring_mins[COLL_GATHER] = PAIR_GATHER_RING_MIN_BYTES;
        s->ring_mins[COLL_SCATTER] = PAIR_SCATTER_RING_MIN_BYTES;
      }
      /* Below its ring minimum a reduce-scatter keeps the pieces, not the chain (SIRCL's chain minimum of
       * 4 MiB of input is below the pair's ring minimum). */
      if (!chain_min || !*chain_min) s->chain_mins[COLL_SCATTER] = PAIR_SCATTER_RING_MIN_BYTES;
    } else {
      s->large_schedule = SCHEDULE_AUTO;
    }
  }
  int links = s->gather_schedule != SCHEDULE_PIECES || s->scatter_schedule != SCHEDULE_PIECES ||
              s->large_schedule == SCHEDULE_RING;
  if ((s->large_schedule != SCHEDULE_PIECES || links) && s->threads < SCCL_LINK_MIN_THREADS) {
    put_error(err, len, "the chain and link schedules need at least %d threads per block (SIRCL_THREADS)",
              SCCL_LINK_MIN_THREADS);
    return -1;
  }
  if (link_settings(e, s, pair_plan, err, len)) return -1;
  e->link_on = e->world > 1 && links;
  if (!s->ring_on && (s->large_schedule == SCHEDULE_RING || s->gather_schedule == SCHEDULE_RING ||
                      s->scatter_schedule == SCHEDULE_RING)) {
    put_error(err, len, "a ring schedule needs the ring plan of the layout: LIBSIRCL_RING_WINDOW on every rank "
                        "(tools/site_routes.py prints it when the ring can run)");
    return -1;
  }
  /* Chain order: LIBSIRCL_CHAIN_ORDER lists positions in chain order; unset, the ranks by position (the
   * cable order of a path or cycle whose positions are ring positions). */
  int order[MAX_WORLD] = {0}, count = 0;
  const char *text = sccl_env("LIBSIRCL_CHAIN_ORDER");
  if (text && *text) {
    char copy[256];
    snprintf(copy, sizeof copy, "%s", text);
    for (char *save = NULL, *item = strtok_r(copy, ",", &save); item; item = strtok_r(NULL, ",", &save)) {
      char *end;
      long position = strtol(item, &end, 10);
      int found = -1;
      for (int r = 0; r < e->world; ++r)
        if (e->positions[r] == position) found = r;
      for (int k = 0; k < count && found >= 0; ++k)
        if (order[k] == found) found = -1;
      if (*end || found < 0 || count == e->world) {
        put_error(err, len, "LIBSIRCL_CHAIN_ORDER=%s must list the positions of the communicator's ranks once",
                  text);
        return -1;
      }
      order[count++] = found;
    }
    if (count != e->world) {
      put_error(err, len, "LIBSIRCL_CHAIN_ORDER=%s names %d of %d ranks", text, count, e->world);
      return -1;
    }
  } else {
    for (int r = 0; r < e->world; ++r) order[r] = r;
    for (int i = 1; i < e->world; ++i)
      for (int k = i; k > 0 && e->positions[order[k - 1]] > e->positions[order[k]]; --k) {
        int t = order[k - 1];
        order[k - 1] = order[k];
        order[k] = t;
      }
  }
  for (int i = 0; i < 8; ++i) s->chain_order[i] = i < e->world ? order[i] : -1;
  for (int i = 0; i < e->world; ++i)
    if (order[i] == e->rank) e->chain_index = i;
  e->chain_prev = e->chain_index > 0 ? order[e->chain_index - 1] : -1;
  e->chain_next = e->chain_index < e->world - 1 ? order[e->chain_index + 1] : -1;
  e->ring_prev = s->ring_on ? order[(e->chain_index + e->world - 1) % e->world] : -1;
  e->ring_next = s->ring_on ? order[(e->chain_index + 1) % e->world] : -1;
  e->chain_on = e->world > 1 && s->large_schedule != SCHEDULE_PIECES;
  int classes = 0;
  uint32_t largest = s->blocks > s->large_blocks ? s->blocks : s->large_blocks;
  if (largest < 32) largest = 32;
  while ((1u << classes) <= largest) ++classes;
  e->classes = classes;
  return 0;
}

static const char *settings_difference(const shared_settings *a, const shared_settings *b) {
#define SAME(field, label) \
  if (a->field != b->field) return label
  SAME(magic, "record format");
  SAME(version, "record version");
  SAME(world, "world size");
  SAME(lanes, "lane count");
  SAME(transport, "transport");
  SAME(slot_bytes, "slot bytes");
  SAME(capacity, "LIBSIRCL_MAX_SIZE");
  SAME(large_piece, "SIRCL_LARGE_PIECE_BYTES");
  SAME(oneshot_max, "SIRCL_ONESHOT_MAX_BYTES");
  SAME(threads, "SIRCL_THREADS");
  SAME(packs_per_thread, "SIRCL_PACKS_PER_THREAD");
  SAME(spin_limit, "SIRCL_SPIN_LIMIT");
  SAME(algorithm_plus_one, "SIRCL_ALLREDUCE_ALGORITHM");
  SAME(startup_wait_us, "SIRCL_STARTUP_WAIT_S");
  SAME(serving_wait_us, "SIRCL_SERVING_WAIT_S");
  SAME(proxy_abi, "native proxy ABI");
  SAME(graph_staging, "stream-ordered allocation (cuMemAllocAsync and the device's memory pools), which "
                      "stages unaligned buffers under CUDA graph capture");
  SAME(large_schedule, "SIRCL_LARGE_SCHEDULE");
  SAME(chain_slots, "SIRCL_CHAIN_SLOTS");
  SAME(chain_slot_bytes, "SIRCL_CHAIN_SLOT_BYTES");
  SAME(chain_chunk, "SIRCL_CHAIN_CHUNK_BYTES");
  SAME(gather_schedule, "SIRCL_GATHER_SCHEDULE");
  SAME(scatter_schedule, "SIRCL_SCATTER_SCHEDULE");
  SAME(link_slots, "SIRCL_LINK_SLOTS");
  SAME(link_slot_bytes, "SIRCL_LINK_SLOT_BYTES");
  SAME(ring_on, "the ring plan (LIBSIRCL_RING_WINDOW)");
  SAME(ring_stagger, "SIRCL_RING_STAGGER");
  SAME(ring_gather_stagger, "SIRCL_RING_GATHER_STAGGER");
  SAME(link_tile, "LIBSIRCL_LINK_TILE_BYTES");
  SAME(link_blocks, "SIRCL_LINK_BLOCKS");
  SAME(blocks_set, "SIRCL_LINK_BLOCKS");
  SAME(pair_plan, "the pair plan (SIRCL_LARGE_SCHEDULE, LIBSIRCL_FORWARD_WINDOWS, LIBSIRCL_RING_WINDOW)");
#undef SAME
  if (memcmp(a->coll_blocks, b->coll_blocks, sizeof a->coll_blocks)) return "LIBSIRCL_*_LINK_BLOCKS";
  if (memcmp(a->chunk_set, b->chunk_set, sizeof a->chunk_set)) return "link pieces (SIRCL_*LINK_CHUNK_BYTES)";
  if (memcmp(a->chain_mins, b->chain_mins, sizeof a->chain_mins)) return "SIRCL_CHAIN_MIN_BYTES";
  if (memcmp(a->ring_mins, b->ring_mins, sizeof a->ring_mins)) return "SIRCL_RING_MIN_BYTES";
  if (memcmp(a->link_chunk, b->link_chunk, sizeof a->link_chunk)) return "link pieces (SIRCL_*LINK_CHUNK_BYTES)";
  if (strcmp(a->kernel_pack, b->kernel_pack)) return "transport kernel pack";
  if (strcmp(a->fold_pack, b->fold_pack)) return "fold kernel pack";
  if (strcmp(a->links_pack, b->links_pack)) return "link kernel pack";
  if (memcmp(a->chain_order, b->chain_order, sizeof a->chain_order)) return "chain order (LIBSIRCL_CHAIN_ORDER)";
  return NULL;
}

/* -- progress thread placement ------------------------------------------------------------------- */

/* The CPUs of the fastest class among `allowed` into `fast`: by the Arm part number of each CPU in
 * /proc/cpuinfo (Cortex-X925 0xd85 above Cortex-A725 0xd87 on a GB10), else by sysfs cpu_capacity within
 * 15% of the largest. Returns the count, or 0 when no source distinguishes classes. */
static int fastest_cpus(const cpu_set_t *allowed, cpu_set_t *fast) {
  static const struct { unsigned part; int tier; } tiers[] = {
      {0xd44, 3}, {0xd48, 3}, {0xd4e, 3}, {0xd82, 3}, {0xd85, 3},  /* Cortex-X1, X2, X3, X4, X925 */
      {0xd47, 2}, {0xd4d, 2}, {0xd81, 2}, {0xd87, 2},              /* Cortex-A710, A715, A720, A725 */
      {0xd46, 1}, {0xd80, 1}};                                     /* Cortex-A510, A520 */
  enum { CPUS = 1024 };
  int rank_of[CPUS];
  int cpu = -1, top = 0, low = 1 << 30, known = 0, count = 0;
  for (int c = 0; c < CPUS; ++c) rank_of[c] = 0;
  FILE *f = fopen("/proc/cpuinfo", "r");
  char line[256];
  while (f && fgets(line, sizeof line, f)) {
    unsigned part;
    if (sscanf(line, "processor : %d", &cpu) == 1 || sscanf(line, "processor\t: %d", &cpu) == 1) continue;
    char *colon = strchr(line, ':');
    if (!colon || strncmp(line, "CPU part", 8) || cpu < 0 || cpu >= CPUS || sscanf(colon + 1, "%x", &part) != 1)
      continue;
    rank_of[cpu] = 2;
    for (size_t t = 0; t < sizeof tiers / sizeof tiers[0]; ++t)
      if (tiers[t].part == part) rank_of[cpu] = tiers[t].tier;
  }
  if (f) fclose(f);
  for (int c = 0; c < CPUS; ++c)
    if (CPU_ISSET(c, allowed) && rank_of[c]) ++known;
  if (known != CPU_COUNT(allowed)) {
    /* No part numbers for every CPU: sysfs capacities. */
    known = 0;
    for (int c = 0; c < CPUS; ++c) {
      rank_of[c] = 0;
      if (!CPU_ISSET(c, allowed)) continue;
      char path[96];
      snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", c);
      FILE *g = fopen(path, "r");
      if (g && fscanf(g, "%d", &rank_of[c]) == 1 && rank_of[c] > 0) ++known;
      if (g) fclose(g);
    }
    if (known != CPU_COUNT(allowed)) return 0;
    int largest = 0;
    for (int c = 0; c < CPUS; ++c)
      if (rank_of[c] > largest) largest = rank_of[c];
    for (int c = 0; c < CPUS; ++c) rank_of[c] = rank_of[c] && rank_of[c] * 100 >= largest * 85 ? 2 : rank_of[c] ? 1 : 0;
  }
  for (int c = 0; c < CPUS; ++c)
    if (CPU_ISSET(c, allowed) && rank_of[c]) {
      if (rank_of[c] > top) top = rank_of[c];
      if (rank_of[c] < low) low = rank_of[c];
    }
  if (top == low) return 0;
  CPU_ZERO(fast);
  for (int c = 0; c < CPUS; ++c)
    if (CPU_ISSET(c, allowed) && rank_of[c] == top) {
      CPU_SET(c, fast);
      ++count;
    }
  return count;
}

static void format_cpus(const cpu_set_t *set, char *out, size_t len) {
  size_t used = 0;
  out[0] = 0;
  for (int c = 0; c < CPU_SETSIZE && used < len; ++c) {
    if (!CPU_ISSET(c, set) || (c > 0 && CPU_ISSET(c - 1, set))) continue;
    int last = c;
    while (last + 1 < CPU_SETSIZE && CPU_ISSET(last + 1, set)) ++last;
    used += (size_t)snprintf(out + used, len - used, last > c ? "%s%d-%d" : "%s%d", used ? "," : "", c, last);
  }
}

/* Start the transport's progress thread. SIRCL_PROGRESS_CPU, when set, pins it (the transport reads it).
 * Otherwise LIBSIRCL_CPU_POLICY=performance (the default) keeps it on the fastest CPU class this thread may
 * use, so it never runs on an efficiency core, where every op of the group waits for it; `none` leaves it
 * to the scheduler. The thread inherits its creator's affinity, so the start runs under that set and this
 * thread's own affinity is restored afterwards. */
static int start_progress(sccl_engine *e) {
  const char *pinned = getenv("SIRCL_PROGRESS_CPU");
  const char *policy = sccl_env("LIBSIRCL_CPU_POLICY");
  snprintf(e->progress_cpus, sizeof e->progress_cpus, "%s", pinned && *pinned ? pinned : "unpinned");
  cpu_set_t mine, fast;
  int placed = !(pinned && *pinned) && !(policy && !strcmp(policy, "none")) &&
               pthread_getaffinity_np(pthread_self(), sizeof mine, &mine) == 0 && fastest_cpus(&mine, &fast) > 0 &&
               pthread_setaffinity_np(pthread_self(), sizeof fast, &fast) == 0;
  if (placed) format_cpus(&fast, e->progress_cpus, sizeof e->progress_cpus);
  int rc = e->transport->start(e->proxy);
  if (placed) pthread_setaffinity_np(pthread_self(), sizeof mine, &mine);
  return rc;
}

/* -- forward windows of relayed lanes ------------------------------------------------------------ */

/* LIBSIRCL_FORWARD_WINDOWS: <position>=<bytes>[/<bytes>],... the forward window of each lane toward the
 * rank at that position (SIRCL's routes.forward_windows for the layout; tools/site_routes.py prints it),
 * 0 for a direct lane; positions outside the communicator are ignored. SIRCL_FORWARD_CHUNK_BYTES is the
 * chunk a windowed stripe is posted in (SIRCL's default 32,768). */
static int forward_windows(sccl_engine *e, char *err, size_t len) {
  memset(e->forward, 0, sizeof e->forward);
  e->forward_lanes = 0;
  uint64_t chunk;
  if (env_u64("SIRCL_FORWARD_CHUNK_BYTES", 32768, PACK, 1u << 30, &chunk, err, len)) return -1;
  if (chunk % PACK) {
    put_error(err, len, "SIRCL_FORWARD_CHUNK_BYTES must be a multiple of 16 bytes");
    return -1;
  }
  e->forward_chunk = (uint32_t)chunk;
  const char *text = sccl_env("LIBSIRCL_FORWARD_WINDOWS");
  if (!text || !*text) return 0;
  char copy[1024];
  snprintf(copy, sizeof copy, "%s", text);
  for (char *save = NULL, *entry = strtok_r(copy, ",", &save); entry; entry = strtok_r(NULL, ",", &save)) {
    char *equals = strchr(entry, '=');
    char *end;
    long position = equals ? strtol(entry, &end, 10) : -1;
    if (!equals || end != equals || position < 0) {
      put_error(err, len, "LIBSIRCL_FORWARD_WINDOWS entry '%s' is not <position>=<bytes>[/<bytes>]", entry);
      return -1;
    }
    int peer = -1;
    for (int r = 0; r < e->world; ++r)
      if (e->positions[r] == position) peer = r;
    int lane = 0;
    for (char *save2 = NULL, *value = strtok_r(equals + 1, "/", &save2); value;
         value = strtok_r(NULL, "/", &save2), ++lane) {
      char *stop;
      unsigned long bytes = strtoul(value, &stop, 10);
      if (*stop || lane >= MAX_LANES || bytes > 0xFFFFFFFFul) {
        put_error(err, len, "LIBSIRCL_FORWARD_WINDOWS entry '%s' needs 1 or 2 byte counts", entry);
        return -1;
      }
      if (peer < 0 || peer == e->rank || lane >= e->lanes) continue;
      e->forward[peer][lane] = (uint32_t)bytes;
      if (bytes) e->forward_lanes += 1;
    }
  }
  return 0;
}

/* -- routes and devices (hardware transport) -------------------------------------------------- */

typedef struct {
  int lanes;
  int n_devices;
  char devices[MAX_DEVICES][64];
  int lane_device[MAX_WORLD][MAX_LANES];
  int gid_index[MAX_DEVICES];
} route_table;

/* SIRCL_PEER_ROUTES: <position>=<device>[/<device>],... naming the position of every other rank of the
 * communicator (`positions[r]` is rank r's); entries for positions outside the communicator are ignored, so
 * one map serves every communicator a process joins. */
static int parse_routes(int world, int rank, const int *positions, route_table *t, char *err, size_t len) {
  const char *text = sccl_env("SIRCL_PEER_ROUTES");
  memset(t, 0, sizeof *t);
  for (int p = 0; p < MAX_WORLD; ++p)
    for (int l = 0; l < MAX_LANES; ++l) t->lane_device[p][l] = -1;
  if (!text || !*text) {
    put_error(err, len, "the verbs transport needs a route map: set SIRCL_PEER_ROUTES "
                        "(<peer>=<device>[/<device>],... for every other rank)");
    return -1;
  }
  char names[MAX_POSITIONS][MAX_LANES][64];
  int counts[MAX_POSITIONS] = {0}, seen[MAX_POSITIONS] = {0};
  char copy[1024];
  snprintf(copy, sizeof copy, "%s", text);
  for (char *save = NULL, *entry = strtok_r(copy, ",", &save); entry; entry = strtok_r(NULL, ",", &save)) {
    char *equals = strchr(entry, '=');
    char *end;
    long peer = equals ? strtol(entry, &end, 10) : -1;
    if (!equals || end != equals || peer < 0 || peer >= MAX_POSITIONS || peer == positions[rank] || seen[peer]) {
      put_error(err, len,
                "SIRCL_PEER_ROUTES entry '%s' must name another position (0-%d, this rank's is %d) once as "
                "<position>=<device>",
                entry, MAX_POSITIONS - 1, positions[rank]);
      return -1;
    }
    seen[peer] = 1;
    char *devices = equals + 1;
    for (char *save2 = NULL, *dev = strtok_r(devices, "/", &save2); dev; dev = strtok_r(NULL, "/", &save2)) {
      while (*dev == ' ') ++dev;
      if (counts[peer] == MAX_LANES || !*dev || strlen(dev) >= 64) {
        put_error(err, len, "SIRCL_PEER_ROUTES gives position %ld 1 or 2 device names", peer);
        return -1;
      }
      snprintf(names[peer][counts[peer]++], 64, "%s", dev);
    }
  }
  for (int p = 0; p < world; ++p) {
    if (p == rank) continue;
    int at = positions[p];
    if (!seen[at] || !counts[at]) {
      put_error(err, len, "SIRCL_PEER_ROUTES names no device toward rank %d (position %d)", p, at);
      return -1;
    }
    if (!t->lanes) t->lanes = counts[at];
    if (counts[at] != t->lanes) {
      put_error(err, len, "SIRCL_PEER_ROUTES gives rank %d (position %d) %d lanes and another rank %d", p, at,
                counts[at], t->lanes);
      return -1;
    }
    for (int l = 0; l < counts[at]; ++l) {
      int d = 0;
      while (d < t->n_devices && strcmp(t->devices[d], names[at][l])) ++d;
      if (d == t->n_devices) {
        if (t->n_devices == MAX_DEVICES) {
          put_error(err, len, "SIRCL_PEER_ROUTES names more than %d devices", MAX_DEVICES);
          return -1;
        }
        snprintf(t->devices[t->n_devices++], 64, "%s", names[at][l]);
      }
      t->lane_device[p][l] = d;
    }
  }
  return 0;
}

static int read_text(const char *path, char *out, size_t len) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  size_t n = fread(out, 1, len - 1, f);
  fclose(f);
  out[n] = 0;
  while (n && (out[n - 1] == '\n' || out[n - 1] == ' ')) out[--n] = 0;
  return 0;
}

/* The single RoCE v2 IPv4-mapped GID of `device` port 1, restricted to entries
 * of the device's own interface when exactly one is visible (SIRCL roce_gid). */
static int resolve_gid_index(const char *root, const char *device, int *out, char *err, size_t len) {
  char path[512], netdev[128] = {0};
  snprintf(path, sizeof path, "%s/%s/device/net", root, device);
  DIR *dir = opendir(path);
  if (dir) {
    int count = 0;
    for (struct dirent *d = readdir(dir); d; d = readdir(dir)) {
      if (d->d_name[0] == '.') continue;
      if (++count == 1) snprintf(netdev, sizeof netdev, "%s", d->d_name);
    }
    closedir(dir);
    if (count != 1) netdev[0] = 0;
  }
  snprintf(path, sizeof path, "%s/%s/ports/1/gids", root, device);
  dir = opendir(path);
  if (!dir) {
    put_error(err, len, "%s port 1 has no readable GID table under %s", device, path);
    return -1;
  }
  int found = -1, matches = 0;
  for (struct dirent *d = readdir(dir); d; d = readdir(dir)) {
    char *end;
    long index = strtol(d->d_name, &end, 10);
    if (*end || end == d->d_name) continue;
    char gid[64], type[64], owner[128];
    snprintf(path, sizeof path, "%s/%s/ports/1/gids/%ld", root, device, index);
    if (read_text(path, gid, sizeof gid)) continue;
    if (strncmp(gid, "0000:0000:0000:0000:0000:ffff:", 30) || !strcmp(gid + 30, "0000:0000")) continue;
    snprintf(path, sizeof path, "%s/%s/ports/1/gid_attrs/types/%ld", root, device, index);
    if (read_text(path, type, sizeof type) || strcasecmp(type, "RoCE v2")) continue;
    if (netdev[0]) {
      snprintf(path, sizeof path, "%s/%s/ports/1/gid_attrs/ndevs/%ld", root, device, index);
      if (read_text(path, owner, sizeof owner) || strcmp(owner, netdev)) continue;
    }
    found = (int)index;
    ++matches;
  }
  closedir(dir);
  if (matches != 1) {
    put_error(err, len, "%s port 1 has %d RoCE v2 IPv4 GIDs%s%s; set SIRCL_GID_INDEX", device, matches,
              netdev[0] ? " on " : "", netdev);
    return -1;
  }
  *out = found;
  return 0;
}

static int gid_indices(route_table *t, char *err, size_t len) {
  const char *names[] = {"SIRCL_GID_INDEX", "NCCL_IB_GID_INDEX"};
  for (int i = 0; i < 2; ++i) {
    const char *text = sccl_env(names[i]);
    if (text && *text) {
      char *end;
      long value = strtol(text, &end, 10);
      if (*end || value < 0 || value > 255) {
        put_error(err, len, "%s=%s is outside 0-255", names[i], text);
        return -1;
      }
      for (int d = 0; d < t->n_devices; ++d) t->gid_index[d] = (int)value;
      return 0;
    }
  }
  const char *root = sccl_env("LIBSIRCL_SYSFS_INFINIBAND");
  if (!root || !*root) root = "/sys/class/infiniband";
  for (int d = 0; d < t->n_devices; ++d)
    if (resolve_gid_index(root, t->devices[d], &t->gid_index[d], err, len)) return -1;
  return 0;
}

static int traffic_class(void) {
  const char *names[] = {"SIRCL_TRAFFIC_CLASS", "NCCL_IB_TC"};
  for (int i = 0; i < 2; ++i) {
    const char *text = sccl_env(names[i]);
    if (text && *text) {
      long value = strtol(text, NULL, 10);
      return value >= 0 && value <= 255 ? (int)value : 0;
    }
  }
  return 0;
}

/* Emulation: lane l of every peer leaves through this process's device emu<pid>.<l>. */
static int emulated_routes(int world, int rank, route_table *t, char *err, size_t len) {
  uint64_t lanes;
  if (env_u64("LIBSIRCL_EMU_LANES", 1, 1, 2, &lanes, err, len)) return -1;
  memset(t, 0, sizeof *t);
  t->lanes = (int)lanes;
  t->n_devices = (int)lanes;
  if (sccl_emu_attach(err, len)) return -1;
  for (int d = 0; d < t->n_devices; ++d) {
    snprintf(t->devices[d], 64, "emu%d.%d", (int)getpid(), d);
    uint8_t gid[16] = {0xfe, 0x80};
    uint32_t pid = (uint32_t)getpid();
    gid[8] = (uint8_t)(pid >> 24);
    gid[9] = (uint8_t)(pid >> 16);
    gid[10] = (uint8_t)(pid >> 8);
    gid[11] = (uint8_t)pid;
    gid[15] = (uint8_t)(d + 1);
    if (sccl_emu_add_device(t->devices[d], gid) < 0) {
      put_error(err, len, "the emulation fabric refused device %s", t->devices[d]);
      return -1;
    }
    t->gid_index[d] = 0;
  }
  for (int p = 0; p < MAX_WORLD; ++p)
    for (int l = 0; l < MAX_LANES; ++l) t->lane_device[p][l] = (p < world && p != rank && l < t->lanes) ? l : -1;
  return 0;
}

/* -- setup ----------------------------------------------------------------------------------- */

static int cuda_check(sccl_CUresult r, const char *what, char *err, size_t len) {
  if (r == SCCL_CUDA_SUCCESS) return 0;
  put_error(err, len, "%s: %s", what, sccl_cuda_result_text(r));
  return -1;
}

static int allocate(sccl_engine *e, char *err, size_t len) {
  uint64_t layout[7];
  if (e->transport->layout(e->world, e->shared.slot_bytes, layout) != 0) {
    put_error(err, len, "the native layer refused an arena of %d ranks and %" PRIu64 "-byte slots", e->world,
              e->shared.slot_bytes);
    return -1;
  }
  e->recv_off = layout[0];
  e->flag_off = layout[1];
  e->send_off = layout[2];
  e->ctrl_off = layout[3];
  e->region_bytes = layout[4];
  if (e->chain_on) {
    uint64_t chain[9];
    if (e->transport->chain_layout(e->lanes, (int)e->shared.chain_slots, e->shared.chain_slot_bytes, chain) != 0) {
      put_error(err, len, "the native layer refused a chain area of %u slots of %" PRIu64 " bytes",
                e->shared.chain_slots, e->shared.chain_slot_bytes);
      return -1;
    }
    e->chain_off = round_up(e->region_bytes, SLOT_ALIGNMENT);
    e->region_bytes = e->chain_off + chain[8];
  }
  if (e->link_on) {
    uint64_t link[9];
    if (e->transport->link_layout(e->lanes, (int)e->shared.link_slots, e->shared.link_slot_bytes, link) != 0) {
      put_error(err, len, "the native layer refused a link area of %u slots of %" PRIu64 " bytes",
                e->shared.link_slots, e->shared.link_slot_bytes);
      return -1;
    }
    e->link_off = round_up(e->region_bytes, SLOT_ALIGNMENT);
    e->region_bytes = e->link_off + link[8];
    memcpy(e->link_layout, link, sizeof e->link_layout);
  }
  if (e->transport == &emulation_ops) {
    e->host = sccl_emu_segment_alloc(e->region_bytes, err, len);
    if (!e->host) return -1;
    e->host_emulated = 1;
    if (cuda_check(cu->MemHostRegister(e->host, e->region_bytes,
                                       SCCL_CU_MEMHOSTREGISTER_PORTABLE | SCCL_CU_MEMHOSTREGISTER_DEVICEMAP),
                   "registering the emulated arena with CUDA", err, len))
      return -1;
    e->host_registered = 1;
  } else {
    void *host = NULL;
    if (cuda_check(cu->MemHostAlloc(&host, e->region_bytes,
                                    SCCL_CU_MEMHOSTALLOC_PORTABLE | SCCL_CU_MEMHOSTALLOC_DEVICEMAP),
                   "allocating the pinned arena", err, len))
      return -1;
    e->host = host;
    memset(e->host, 0, e->region_bytes);
  }
  if (cuda_check(cu->MemHostGetDevicePointer(&e->dev_base, e->host, 0), "mapping the arena", err, len)) return -1;
  e->ctrl = (volatile uint32_t *)(e->host + e->ctrl_off);
  size_t counter_bytes = 4u * (2u + (1u + MAX_PHASES) * (unsigned)e->classes);
  if (cuda_check(cu->MemAlloc(&e->counters, counter_bytes), "allocating the device counters", err, len) ||
      cuda_check(cu->MemAlloc(&e->arrival, 16), "allocating the arrival words", err, len) ||
      cuda_check(cu->MemAlloc(&e->scratch[0], e->shared.slot_bytes), "allocating scratch", err, len) ||
      cuda_check(cu->MemAlloc(&e->scratch[1], e->shared.slot_bytes * (uint64_t)e->world), "allocating scratch", err,
                 len))
    return -1;
  if (e->chain_on && cuda_check(cu->MemAlloc(&e->chain_counters, 16), "allocating the chain counters", err, len))
    return -1;
  size_t piece_bytes = 4u * (SCCL_LINK_PIECE_COUNTERS + SCCL_LINK_DECISION_WORDS);
  if (e->link_on &&
      (cuda_check(cu->MemAlloc(&e->link_counters, 4u * SCCL_LINK_COUNTER_WORDS), "allocating the link counters", err,
                  len) ||
       cuda_check(cu->MemAlloc(&e->piece_counters, piece_bytes), "allocating the piece counters", err, len) ||
       (e->shared.scatter_schedule != SCHEDULE_PIECES &&
        cuda_check(cu->MemAlloc(&e->link_scratch, e->shared.link_tile), "allocating the link scratch", err, len))))
    return -1;
  if (e->link_on) {
    void *zeros = calloc(1, piece_bytes);
    int failed = !zeros ||
                 cuda_check(cu->MemcpyHtoD(e->link_counters, zeros, 4u * SCCL_LINK_COUNTER_WORDS),
                            "zeroing the link counters", err, len) ||
                 cuda_check(cu->MemcpyHtoD(e->piece_counters, zeros, piece_bytes), "zeroing the piece counters", err,
                            len);
    if (!zeros) put_error(err, len, "out of memory");
    free(zeros);
    if (failed) return -1;
  }
  void *zero = calloc(1, counter_bytes > 16 ? counter_bytes : 16);
  if (!zero) {
    put_error(err, len, "out of memory");
    return -1;
  }
  int bad = cuda_check(cu->MemcpyHtoD(e->counters, zero, counter_bytes), "zeroing the counters", err, len) ||
            cuda_check(cu->MemcpyHtoD(e->arrival, zero, 16), "zeroing the arrival words", err, len) ||
            (e->chain_on &&
             cuda_check(cu->MemcpyHtoD(e->chain_counters, zero, 16), "zeroing the chain counters", err, len));
  free(zero);
  return bad ? -1 : 0;
}

static void release_memory(sccl_engine *e) {
  if (!cu) return;
  if (e->counters) cu->MemFree(e->counters);
  if (e->arrival) cu->MemFree(e->arrival);
  if (e->chain_counters) cu->MemFree(e->chain_counters);
  if (e->link_counters) cu->MemFree(e->link_counters);
  if (e->piece_counters) cu->MemFree(e->piece_counters);
  if (e->link_scratch) cu->MemFree(e->link_scratch);
  if (e->stage) cu->MemFree(e->stage);
  for (int i = 0; i < e->retired; ++i) cu->MemFree(e->retired_stage[i]);
  free(e->retired_stage);
  e->retired_stage = NULL;
  e->link_counters = e->piece_counters = e->link_scratch = e->stage = 0;
  e->retired = e->retired_room = 0;
  for (int i = 0; i < 2; ++i)
    if (e->scratch[i]) cu->MemFree(e->scratch[i]);
  if (e->host) {
    if (e->host_registered) cu->MemHostUnregister(e->host);
    if (e->host_emulated) {
      sccl_emu_segment_free(e->host);
    } else {
      cu->MemFreeHost(e->host);
    }
  }
  e->counters = e->arrival = e->scratch[0] = e->scratch[1] = e->chain_counters = 0;
  e->host = NULL;
}

/* One all-gather round of fixed-size records; returns the records or NULL. */
static void *exchange(sccl_bootstrap *b, const void *mine, unsigned bytes, int world, const atomic_int *cancelled,
                      char *err, size_t len) {
  void *all = NULL;
  unsigned lengths[MAX_WORLD];
  int result = sccl_bootstrap_allgather(b, mine, bytes, &all, lengths, cancelled);
  if (result != 0) {
    put_error(err, len, "setup exchange: %s", sccl_bootstrap_error());
    return NULL;
  }
  for (int r = 0; r < world; ++r) {
    if (lengths[r] != bytes) {
      free(all);
      put_error(err, len, "setup exchange: rank %d sent %u bytes, expected %u", r, lengths[r], bytes);
      return NULL;
    }
  }
  return all;
}

/* Every rank's verdict on a setup step; fails on every rank with every rank's reason. */
static int verdict(sccl_bootstrap *b, int world, int rank, int ok, const char *message,
                   const atomic_int *cancelled, char *err, size_t len) {
  verdict_record mine;
  memset(&mine, 0, sizeof mine);
  mine.ok = (uint32_t)ok;
  snprintf(mine.message, sizeof mine.message, "%s", ok ? "" : message);
  verdict_record *all = exchange(b, &mine, sizeof mine, world, cancelled, err, len);
  if (!all) return -1;
  char text[1024] = {0};
  size_t used = 0;
  int failed = 0;
  for (int r = 0; r < world; ++r) {
    if (all[r].ok) continue;
    failed = 1;
    used += (size_t)snprintf(text + used, used < sizeof text ? sizeof text - used : 0, "%srank %d: %.300s",
                             used ? "; " : "", r, all[r].message);
    if (used >= sizeof text) used = sizeof text - 1;
  }
  free(all);
  (void)rank;
  if (failed) put_error(err, len, "SIRCL session setup failed: %s", text);
  return failed ? -1 : 0;
}

static void start_receipts(sccl_engine *e);

/* -- fail-stop ----------------------------------------------------------------------------------- */

/* LIBSIRCL_FAIL_STOP=1 (or abort): one watcher thread per process checks every watched communicator of two
 * or more ranks for an asynchronous error (a flag wait that timed out, a failed progress thread) every
 * FAIL_STOP_POLL_MS, and at the first one writes it to stderr and ends the process: with 1 at once with exit
 * status FAIL_STOP_STATUS (_exit: no atexit handlers, no core dump, so the end is bounded), with abort by
 * abort() (SIGABRT; the system's core-dump handling may delay the end). A caller that only checks the codes
 * of enqueue calls then cannot keep using a failed collective's output beyond the wait limit plus one poll,
 * whether or not it calls the library again; one that reads the output within that poll after its stream
 * wait returns can still read it. A communicator joins the list when its creation succeeds and leaves it
 * before destroy or abort releases anything (under watch_lock, which every check holds); library unload
 * stops the watcher first. */
enum { FAIL_STOP_POLL_MS = 5, FAIL_STOP_STATUS = 70 };
static pthread_mutex_t watch_lock = PTHREAD_MUTEX_INITIALIZER;
static sccl_engine *watch_list;
static pthread_t watcher;
static int watcher_started;
static atomic_int watcher_stop;
static void write_receipt(sccl_engine *e);
static uint64_t realtime_ns(void);

static void fail_stop(sccl_engine *e, const char *message) {
  char line[1200];
  uint64_t now = realtime_ns();
  int n = snprintf(line, sizeof line,
                   "libsircl: LIBSIRCL_FAIL_STOP: ending the process at %" PRIu64 ".%03u (Unix time) on an "
                   "asynchronous error of communicator %d (rank %d of %d): %s\n",
                   now / 1000000000u, (unsigned)(now % 1000000000u / 1000000u), e->receipt_id, e->rank, e->world,
                   message);
  if (n < 0) n = 0;
  if (n > (int)sizeof line - 1) n = (int)sizeof line - 1;
  if (write(STDERR_FILENO, line, (size_t)n) < 0) {
    /* Nothing else can report it; the process ends either way. */
  }
  write_receipt(e);
  if (e->fail_stop == FAIL_STOP_ABORT) abort();
  _exit(FAIL_STOP_STATUS);
}

static void *fail_stop_watch(void *unused) {
  (void)unused;
  const struct timespec pause = {0, FAIL_STOP_POLL_MS * 1000000L};
  while (!atomic_load(&watcher_stop)) {
    nanosleep(&pause, NULL);
    pthread_mutex_lock(&watch_lock);
    for (sccl_engine *e = watch_list; e && !atomic_load(&watcher_stop); e = e->watch_next) {
      char message[900];
      if (sccl_engine_async_error(e, message, sizeof message) != ncclSuccess) fail_stop(e, message);
    }
    pthread_mutex_unlock(&watch_lock);
  }
  return NULL;
}

/* A forked child has no watcher thread and none of the parent's communicators (they are invalid there):
 * it starts empty, and its first communicator that asks for fail-stop starts its own watcher. */
static void watch_after_fork(void) {
  pthread_mutex_init(&watch_lock, NULL);
  watch_list = NULL;
  watcher_started = 0;
}

static void fork_handler_install(void) { pthread_atfork(NULL, NULL, watch_after_fork); }

/* The watcher, started by the first communicator that asks for it, before the setup verdict (so a rank
 * that cannot start it fails setup with every other rank). */
static int watch_start(char *err, size_t len) {
  static pthread_once_t fork_handler = PTHREAD_ONCE_INIT;
  pthread_once(&fork_handler, fork_handler_install);
  pthread_mutex_lock(&watch_lock);
  int rc = watcher_started || atomic_load(&watcher_stop) ? 0 : pthread_create(&watcher, NULL, fail_stop_watch, NULL);
  if (!rc) watcher_started = 1;
  pthread_mutex_unlock(&watch_lock);
  if (rc) put_error(err, len, "LIBSIRCL_FAIL_STOP: starting the watcher thread: %s", strerror(rc));
  return rc ? -1 : 0;
}

static void watch_add(sccl_engine *e) {
  pthread_mutex_lock(&watch_lock);
  e->watch_next = watch_list;
  watch_list = e;
  pthread_mutex_unlock(&watch_lock);
}

static void watch_remove(sccl_engine *e) {
  if (!e->fail_stop) return;
  pthread_mutex_lock(&watch_lock);
  for (sccl_engine **at = &watch_list; *at; at = &(*at)->watch_next)
    if (*at == e) {
      *at = e->watch_next;
      break;
    }
  pthread_mutex_unlock(&watch_lock);
}

void sccl_engine_fail_stop_shutdown(void) {
  pthread_mutex_lock(&watch_lock);
  int started = watcher_started;
  atomic_store(&watcher_stop, 1);
  watch_list = NULL;
  pthread_mutex_unlock(&watch_lock);
  if (started) pthread_join(watcher, NULL);
}

int sccl_engine_create(sccl_bootstrap *bootstrap, int nranks, int rank, int position, sccl_CUcontext ctx,
                       const atomic_int *cancelled, sccl_engine **out, char *err, size_t err_len) {
  *out = NULL;
  cu = sccl_cuda_get();
  sccl_engine *e = calloc(1, sizeof *e);
  if (!e) {
    put_error(err, err_len, "out of memory");
    return ncclSystemError;
  }
  e->world = nranks;
  e->rank = rank;
  e->ctx = ctx;
  e->position = position;
  e->positions[0] = position;
  pthread_mutex_init(&e->lock, NULL);
  char local[400] = {0};
  /* Exchange 0: every rank's position, which keys the route map (SIRCL_PEER_ROUTES). */
  if (nranks > 1 && nranks <= MAX_WORLD) {
    int32_t mine = position;
    int32_t *all = exchange(bootstrap, &mine, sizeof mine, nranks, cancelled, err, err_len);
    if (!all) {
      pthread_mutex_destroy(&e->lock);
      free(e);
      return ncclRemoteError;
    }
    for (int r = 0; r < nranks; ++r) e->positions[r] = all[r];
    free(all);
    for (int r = 0; r < nranks && !local[0]; ++r)
      for (int q = 0; q < r; ++q)
        if (e->positions[q] == e->positions[r]) {
          put_error(local, sizeof local, "ranks %d and %d have the same position %d (LIBSIRCL_POSITION)", q, r,
                    e->positions[r]);
          break;
        }
  }
  int pushed = 0;
  const char *transport = sccl_env("LIBSIRCL_TRANSPORT");
  int emulation = transport && !strcmp(transport, "emulation");
  route_table routes;
  memset(&routes, 0, sizeof routes);
  /* Local preparation; a failure is reported to every rank in the first exchange. */
  do {
    if (local[0]) break;
    if (transport && *transport && strcmp(transport, "verbs") && !emulation) {
      put_error(local, sizeof local, "LIBSIRCL_TRANSPORT=%s: verbs or emulation", transport);
      break;
    }
    e->transport = emulation ? &emulation_ops : &verbs_ops;
    if (!cu) {
      put_error(local, sizeof local, "%s", sccl_cuda_error());
      break;
    }
    if (!ctx) {
      put_error(local, sizeof local, "no current CUDA context: select the device before creating the communicator");
      break;
    }
    if (cuda_check(cu->CtxPushCurrent(ctx), "making the communicator's context current", local, sizeof local)) break;
    pushed = 1;
    sccl_CUdevice device;
    if (cuda_check(cu->CtxGetDevice(&device), "cuCtxGetDevice", local, sizeof local)) break;
    e->device = device;
    if (read_settings(e, emulation, local, sizeof local)) break;
    if (nranks == 1) e->fail_stop = 0;
    if (e->fail_stop && watch_start(local, sizeof local)) break;
    e->shared.proxy_abi = (uint32_t)e->transport->abi_version();
    int pools = 0;
    e->shared.graph_staging =
        cu->MemAllocAsync &&
        cu->DeviceGetAttribute(&pools, SCCL_CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device) == SCCL_CUDA_SUCCESS &&
        pools == 1;
    if (nranks == 1) {
      e->shared.lanes = 0;
      break;
    }
    if (nranks > SCCL_KP_MAX_WORLD) {
      put_error(local, sizeof local, "libsircl carries groups of 2 to %d ranks", SCCL_KP_MAX_WORLD);
      break;
    }
    if (emulation) {
      if (emulated_routes(nranks, rank, &routes, local, sizeof local)) break;
    } else {
      if (!sccl_verbs_available()) {
        put_error(local, sizeof local, "libibverbs.so.1 is not loadable; the verbs transport needs rdma-core");
        break;
      }
      if (parse_routes(nranks, rank, e->positions, &routes, local, sizeof local) ||
          gid_indices(&routes, local, sizeof local))
        break;
    }
    e->lanes = routes.lanes;
    e->shared.lanes = (uint32_t)routes.lanes;
    if (e->shared.threads < (uint32_t)(nranks * routes.lanes)) {
      put_error(local, sizeof local, "SIRCL_THREADS must be at least world * lanes");
      break;
    }
    if (sccl_kp_load(ctx)) {
      put_error(local, sizeof local, "%s", sccl_kp_error());
      break;
    }
    for (int a = 0; a < SCCL_ALG_COUNT; ++a)
      for (int d = 0; d < SCCL_DT_COUNT; ++d) sccl_kp_function(ctx, a, d, nranks, &e->fn[a][d]);
    for (int k = SCCL_K_SCATTER; k < SCCL_K_COUNT; ++k)
      for (int d = 0; d < SCCL_DT_COUNT; ++d) sccl_kp_function(ctx, k, d, nranks, &e->fn[k][d]);
    for (int d = 0; d < SCCL_FOLD_DTYPES; ++d)
      for (int o = 0; o < SCCL_FOLD_OPS; ++o) sccl_kp_fold_function(ctx, d, o, &e->fold[d][o]);
    for (int d = 0; d < SCCL_DT_COUNT; ++d) sccl_kp_chain_function(ctx, d, (int)e->shared.chain_unroll, &e->chain_fn[d]);
    for (int k = 0; k < SCCL_LINK_KINDS; ++k)
      for (int d = 0; d < SCCL_DT_COUNT; ++d)
        sccl_kp_link_function(ctx, k, d, (int)e->shared.link_unroll, &e->link_fn[k][d]);
    for (int d = 0; e->ring_reduce_passes == 2 && d < SCCL_DT_COUNT; ++d)
      sccl_kp_ring_reduce_two_pass_function(ctx, d, (int)e->shared.link_unroll, &e->link_fn[SCCL_RING_REDUCE][d]);
    sccl_kp_ring_exchange_function(ctx, (int)e->shared.link_unroll, &e->exchange_fn);
    for (int d = 0; d < SCCL_DT_COUNT; ++d)
      sccl_kp_ring_exchange_reduce_function(ctx, d, (int)e->shared.link_unroll, &e->exchange_reduce_fn[d]);
    if (allocate(e, local, sizeof local)) break;
    e->ctrl[CTRL_WAIT_LIMIT_US] = e->serving ? e->shared.serving_wait_us : e->shared.startup_wait_us;
    const char *names[MAX_DEVICES];
    for (int d = 0; d < routes.n_devices; ++d) names[d] = routes.devices[d];
    int lane_devices[MAX_WORLD * MAX_LANES];
    for (int p = 0; p < nranks; ++p)
      for (int l = 0; l < routes.lanes; ++l) lane_devices[p * routes.lanes + l] = routes.lane_device[p][l];
    char proxy_error[512] = {0};
    e->proxy = e->transport->create(nranks, rank, names, routes.n_devices, lane_devices, routes.lanes,
                                    routes.gid_index, traffic_class(), e->host, e->region_bytes,
                                    e->shared.slot_bytes, proxy_error, sizeof proxy_error);
    if (!e->proxy) {
      put_error(local, sizeof local, "native setup: %s", proxy_error);
    } else if (e->chain_on &&
               e->transport->set_chain(e->proxy, e->chain_prev, e->chain_next, (int)e->shared.chain_slots,
                                       e->shared.chain_slot_bytes, e->chain_off) != 0) {
      put_error(local, sizeof local, "chain schedule: %s", e->transport->error(e->proxy));
    } else if (e->link_on &&
               e->transport->set_links(e->proxy, e->chain_prev, e->chain_next, e->chain_index, e->ring_prev,
                                       e->ring_next, e->ring_window, (int)e->shared.link_slots,
                                       e->shared.link_slot_bytes, e->link_off) != 0) {
      put_error(local, sizeof local, "link collectives: %s", e->transport->error(e->proxy));
    } else if (forward_windows(e, local, sizeof local) == 0 && e->forward_lanes) {
      uint32_t table[MAX_WORLD * MAX_LANES];
      for (int p = 0; p < nranks; ++p)
        for (int l = 0; l < e->lanes; ++l) table[p * e->lanes + l] = e->forward[p][l];
      if (e->transport->set_forward(e->proxy, table, e->forward_chunk) != 0)
        put_error(local, sizeof local, "forward windows: %s", e->transport->error(e->proxy));
    }
  } while (0);

  int status = ncclSuccess;
  if (nranks == 1) {
    if (local[0]) {
      put_error(err, err_len, "rank 0: %s", local);
      status = ncclSystemError;
      goto fail;
    }
    if (pushed) cu->CtxPopCurrent(&ctx);
    start_receipts(e);
    *out = e;
    return ncclSuccess;
  }
  /* Exchange 1: local errors and shared settings. */
  setup_record record;
  memset(&record, 0, sizeof record);
  snprintf(record.error, sizeof record.error, "%s", local);
  record.settings = e->shared;
  setup_record *records = exchange(bootstrap, &record, sizeof record, nranks, cancelled, err, err_len);
  if (!records) {
    status = ncclRemoteError;
    goto fail;
  }
  {
    char text[1024] = {0};
    size_t used = 0;
    /* Settings are compared only when every rank prepared without a local error: a failed rank's record
     * holds the settings it read before failing. */
    int any_error = 0;
    for (int r = 0; r < nranks; ++r) any_error |= records[r].error[0] != 0;
    for (int r = 0; r < nranks; ++r) {
      const char *why = records[r].error[0] ? records[r].error : NULL;
      const char *field = why || any_error ? NULL : settings_difference(&records[r].settings, &records[0].settings);
      if (!why && !field) continue;
      used += (size_t)snprintf(text + used, used < sizeof text ? sizeof text - used : 0, "%srank %d: %s%s",
                               used ? "; " : "", r, why ? why : field, why ? "" : " differs from rank 0");
      if (used >= sizeof text) used = sizeof text - 1;
    }
    free(records);
    if (text[0]) {
      put_error(err, err_len, "SIRCL session setup failed: %s", text);
      status = local[0] ? ncclSystemError : ncclInvalidUsage;
      goto fail;
    }
  }
  /* Exchange 2: connection records; connect every lane. */
  {
    uint64_t blob_bytes = e->transport->blob_bytes();
    void *blob = calloc(1, blob_bytes);
    if (!blob || e->transport->local_blob(e->proxy, blob, blob_bytes) != 0) {
      free(blob);
      put_error(err, err_len, "connection record");
      status = ncclInternalError;
      goto fail;
    }
    void *blobs = exchange(bootstrap, blob, (unsigned)blob_bytes, nranks, cancelled, err, err_len);
    free(blob);
    if (!blobs) {
      status = ncclRemoteError;
      goto fail;
    }
    int rc = e->transport->connect(e->proxy, blobs, blob_bytes * (uint64_t)nranks);
    free(blobs);
    char why[400];
    snprintf(why, sizeof why, "queue-pair connection: %s", rc ? e->transport->error(e->proxy) : "");
    if (verdict(bootstrap, nranks, rank, rc == 0, why, cancelled, err, err_len)) {
      status = ncclSystemError;
      goto fail;
    }
  }
  /* Exchange 3: prove every lane with one write, then start the progress thread. */
  {
    uint64_t check_ms;
    if (env_u64("LIBSIRCL_LANE_CHECK_MS", 10000, 1, 600000, &check_ms, err, err_len)) {
      status = ncclInvalidArgument;
      goto fail;
    }
    int rc = e->transport->lane_check(e->proxy, (int)check_ms);
    const char *dump = sccl_env("LIBSIRCL_LINK_DUMP");
    if (rc == 0 && dump && *dump) e->transport->set_trace(e->proxy, 1u << 20);
    if (rc == 0) rc = start_progress(e);
    char why[400];
    snprintf(why, sizeof why, "lane check: %s", rc ? e->transport->error(e->proxy) : "");
    if (verdict(bootstrap, nranks, rank, rc == 0, why, cancelled, err, err_len)) {
      status = ncclSystemError;
      goto fail;
    }
  }
  if (cuda_check(cu->EventCreate(&e->order_event, SCCL_CU_EVENT_DISABLE_TIMING), "creating the order event", err,
                 err_len)) {
    status = ncclUnhandledCudaError;
    goto fail;
  }
  cu->CtxPopCurrent(&ctx);
  start_receipts(e);
  if (e->fail_stop) watch_add(e);
  *out = e;
  return ncclSuccess;
fail:;
  /* A queue pair or registration that could not be released may still be written: keep the memory. */
  int unreleased = e->proxy ? e->transport->destroy(e->proxy) : 0;
  e->proxy = NULL;
  if (pushed) {
    if (!unreleased) release_memory(e);
    cu->CtxPopCurrent(&ctx);
  }
  pthread_mutex_destroy(&e->lock);
  free(e);
  return status;
}

/* -- collectives ----------------------------------------------------------------------------- */

static int kernel_dtype(ncclDataType_t datatype, size_t *size) {
  switch (datatype) {
    case ncclFloat32: *size = 4; return SCCL_DT_F32;
    case ncclFloat16: *size = 2; return SCCL_DT_F16;
    case ncclBfloat16: *size = 2; return SCCL_DT_BF16;
    default: return -1;
  }
}

static size_t nccl_type_size(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclInt8: case ncclUint8: case ncclFloat8e4m3: case ncclFloat8e5m2: return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
    case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
    default: return 0;
  }
}

static unsigned grid_blocks(uint64_t packs, unsigned threads, unsigned cap, unsigned per_thread) {
  uint64_t per_block = (uint64_t)threads * per_thread;
  uint64_t required = (packs + per_block - 1) / per_block;
  if (required < 1) required = 1;
  uint64_t grid = 1;
  while (grid < required) grid <<= 1;
  return grid < cap ? (unsigned)grid : cap;
}

static int grid_class(unsigned grid) {
  int c = 0;
  while ((1u << (c + 1)) <= grid) ++c;
  return c;
}

/* The flag-wait failure recorded in the command ring, as text; 0 when none. */
static int timed_out(const sccl_engine *e, char *message, size_t len) {
  uint32_t seq = __atomic_load_n(&e->ctrl[CTRL_ERROR_SEQ], __ATOMIC_ACQUIRE);
  if (!seq) return 0;
  put_error(message, len,
            "rank %d: a flag wait for rank %u lane %u timed out at sequence %u (wait limit %.6g s, %s regime); "
            "the session is poisoned",
            e->rank, e->ctrl[CTRL_MISSING_PEER], e->ctrl[CTRL_MISSING_LANE], seq,
            (e->serving ? e->shared.serving_wait_us : e->shared.startup_wait_us) / 1e6,
            e->serving ? "serving" : "startup");
  return 1;
}

static void dump_links(sccl_engine *e, const char *why, int device);

/* A failed native progress thread is reported ahead of a timed-out wait, and both when both hold: a progress
 * thread that stopped leaves the flags its peers and its own kernels wait for unwritten, so the timeout is
 * usually its consequence and never hides it. */
ncclResult_t sccl_engine_async_error(sccl_engine *e, char *message, size_t len) {
  if (!e || e->world == 1) return ncclSuccess;
  int failed = e->proxy && e->transport->failed(e->proxy);
  char waited[400] = {0};
  int timeout = timed_out(e, waited, sizeof waited);
  if (failed) {
    put_error(message, len, "rank %d: native progress thread failed: %s%s%s", e->rank, e->transport->error(e->proxy),
              timeout ? "; then " : "", timeout ? waited : "");
    if (!atomic_exchange(&e->link_dumped, 1)) dump_links(e, "the native progress thread failed", 0);
    return ncclSystemError;
  }
  if (timeout) {
    put_error(message, len, "%s", waited);
    if (!atomic_exchange(&e->link_dumped, 1)) dump_links(e, "a wait of the session timed out", 0);
    return ncclRemoteError;
  }
  return ncclSuccess;
}

static sccl_CUresult copy_async(sccl_engine *e, sccl_CUdeviceptr dst, sccl_CUdeviceptr src, size_t bytes,
                                sccl_CUstream stream) {
  atomic_fetch_add_explicit(&e->local_copies, 1, memory_order_relaxed);
  return cu->MemcpyDtoDAsync(dst, src, bytes, stream);
}

static const char *const type_names[SCCL_FOLD_DTYPES] = {"int8",    "uint8",   "int32",    "uint32",
                                                         "int64",   "uint64",  "float16",  "float32",
                                                         "float64", "bfloat16", "float8e4m3", "float8e5m2"};
static const char *const op_names[SCCL_FOLD_OPS] = {"sum", "prod", "max", "min", "avg"};

/* Bytes of every rank's data in one all-gather op, and of every rank's chunk in one all-to-all or
 * reduce-scatter op. Both depend only on agreed settings, so every rank splits a call into the same ops
 * whatever its buffers' alignment. */
static uint64_t gather_tile(const sccl_engine *e) {
  uint64_t tile = e->shared.large_piece < e->shared.slot_bytes ? e->shared.large_piece : e->shared.slot_bytes;
  return tile / PACK * PACK;
}
static uint64_t scatter_tile(const sccl_engine *e) { return gather_tile(e) / (uint64_t)e->world / PACK * PACK; }

/* Fold the W rows of `count` elements at `rows` (`pitch` bytes apart) into `out`, through scratch[0]
 * when `out` is not aligned to the element. */
static ncclResult_t fold_into(sccl_engine *e, ncclDataType_t datatype, ncclRedOp_t op, size_t item,
                              sccl_CUdeviceptr rows, uint64_t pitch, sccl_CUdeviceptr out, uint64_t count,
                              sccl_CUstream stream, char *err, size_t len) {
  enum { FOLD_THREADS = 256, FOLD_MAX_GRID = 4096 };
  sccl_CUdeviceptr target = out % item ? e->scratch[0] : out;
  uint64_t blocks = (count + FOLD_THREADS - 1) / FOLD_THREADS;
  unsigned grid = blocks < 1 ? 1u : blocks < FOLD_MAX_GRID ? (unsigned)blocks : (unsigned)FOLD_MAX_GRID;
  sccl_CUresult r = sccl_kp_launch_fold(e->fold[datatype][op], rows, (int64_t)pitch, target, (int64_t)count,
                                        e->world, grid, FOLD_THREADS, stream);
  if (r != SCCL_CUDA_SUCCESS) {
    put_error(err, len, "launching the %s %s fold: %s", type_names[datatype], op_names[op], sccl_cuda_result_text(r));
    return ncclUnhandledCudaError;
  }
  atomic_fetch_add_explicit(&e->folds[datatype][op], 1, memory_order_relaxed);
  if (target != out && cuda_check(copy_async(e, out, target, count * item, stream), "copying to an unaligned output",
                                  err, len))
    return ncclUnhandledCudaError;
  return ncclSuccess;
}

/* One op of at most one slot: `nbytes` a positive multiple of 16, pointers 16-byte aligned. */
static ncclResult_t launch_op(sccl_engine *e, int dtype, sccl_CUdeviceptr src, sccl_CUdeviceptr dst,
                              uint64_t nbytes, sccl_CUstream stream, char *err, size_t len) {
  int algorithm = e->algorithm;
  if (algorithm == ALG_AUTO) algorithm = nbytes <= e->shared.oneshot_max ? SCCL_ALG_ONESHOT : SCCL_ALG_TWOSHOT;
  uint64_t packs = nbytes / PACK;
  unsigned cap = algorithm == SCCL_ALG_ONESHOT ? e->shared.blocks : e->shared.large_blocks;
  unsigned grid = grid_blocks(packs, e->shared.threads, cap, e->shared.packs_per_thread);
  int cls = grid_class(grid);
  sccl_allreduce_args a;
  memset(&a, 0, sizeof a);
  a.input = src;
  a.output = dst;
  a.size_packs = (int32_t)packs;
  a.nbytes = (int32_t)nbytes;
  a.recv_base = e->dev_base + e->recv_off;
  a.flag_base = e->dev_base + e->flag_off;
  a.send_base = e->dev_base + e->send_off;
  a.ctrl_base = e->dev_base + e->ctrl_off;
  a.slot_bytes = e->shared.slot_bytes;
  a.epoch = e->counters;
  a.stage_counter = e->counters + 4u * (unsigned)(1 + cls);
  a.tail_counter = e->counters + 4u * (unsigned)(1 + e->classes + cls);
  a.poison = e->counters + 4u * (unsigned)(1 + 2 * e->classes);
  a.phase_counter = e->counters + 4u * (unsigned)(2 + 2 * e->classes + cls);
  a.arrival = e->arrival;
  a.spin_limit = e->shared.spin_limit;
  a.rank = e->rank;
  a.lanes = e->lanes;
  a.one_block = (int32_t)e->shared.one_block;
  sccl_CUresult r = sccl_kp_launch_allreduce(e->fn[algorithm][dtype], algorithm, &a, grid, e->shared.threads, stream);
  if (r != SCCL_CUDA_SUCCESS) {
    put_error(err, len, "launching the %s all-reduce: %s", algorithm == SCCL_ALG_ONESHOT ? "one-shot" : "two-shot",
              sccl_cuda_result_text(r));
    return ncclUnhandledCudaError;
  }
  atomic_fetch_add_explicit(&e->ops[algorithm][dtype], 1, memory_order_relaxed);
  return ncclSuccess;
}

/* Threads per block of the link pack's kernels: the session's, at most SCCL_LINK_MAX_THREADS. */
static unsigned link_threads(const sccl_engine *e) {
  return e->shared.threads < SCCL_LINK_MAX_THREADS ? e->shared.threads : SCCL_LINK_MAX_THREADS;
}

/* The pair plan's step for a call of collective `coll` whose size key (all-reduce: message bytes; all-gather:
 * shard bytes; reduce-scatter: input bytes) is `key`. */
static const plan_step *pair_step(int coll, uint64_t key) {
  const plan_step *plan = coll == COLL_REDUCE ? pair_reduce_plan : coll == COLL_GATHER ? pair_gather_plan
                                                                                       : pair_scatter_plan;
  size_t steps = coll == COLL_REDUCE   ? sizeof pair_reduce_plan / sizeof pair_reduce_plan[0]
                 : coll == COLL_GATHER ? sizeof pair_gather_plan / sizeof pair_gather_plan[0]
                                       : sizeof pair_scatter_plan / sizeof pair_scatter_plan[0];
  const plan_step *chosen = NULL;
  for (size_t i = 0; i < steps; ++i)
    if (key >= plan[i].from) chosen = &plan[i];
  return chosen;
}

/* The piece of one link op of collective `coll` for size key `key`: the configured piece
 * (SIRCL_*LINK_CHUNK_BYTES), else the pair plan's (within the link slot), else the session's. Agreed
 * settings and sizes only, so every rank cuts the same pieces. */
static uint32_t link_piece(const sccl_engine *e, int coll, uint64_t key) {
  const shared_settings *s = &e->shared;
  const plan_step *step = s->pair_plan && !s->chunk_set[coll] ? pair_step(coll, key) : NULL;
  if (!step) return s->link_chunk[coll];
  return step->piece <= s->link_slot_bytes ? step->piece : (uint32_t)(s->link_slot_bytes / PACK * PACK);
}

/* Blocks per role of one link op of collective `coll` for size key `key`: LIBSIRCL_<collective>_LINK_BLOCKS
 * (or SIRCL_<collective>_LINK_BLOCKS), else SIRCL_LINK_BLOCKS, else the pair plan's, else DEFAULT_LINK_BLOCKS
 * (the test hook LIBSIRCL_LINK_BLOCKS_CYCLE before all of them). */
static uint32_t link_blocks_for(sccl_engine *e, int coll, uint64_t key) {
  const shared_settings *s = &e->shared;
  if (e->blocks_cycle_n)
    return e->blocks_cycle[atomic_fetch_add_explicit(&e->blocks_cycle_next, 1, memory_order_relaxed) %
                           (uint64_t)e->blocks_cycle_n];
  if (s->coll_blocks[coll]) return s->coll_blocks[coll];
  if (s->blocks_set) return s->link_blocks;
  const plan_step *step = s->pair_plan ? pair_step(coll, key) : NULL;
  return step ? step->blocks : s->link_blocks;
}

/* One link collective op (kernels/sircl_links.cu): `chunk` bytes per rank's block or chunk, chunks `stride`
 * bytes apart, in pieces of `piece` bytes, `blocks` blocks per role; src and dst 16-byte aligned. */
static ncclResult_t launch_link_entry(sccl_engine *e, sccl_CUfunction fn, unsigned roles, const char *what,
                                      sccl_CUdeviceptr src, sccl_CUdeviceptr dst, sccl_CUdeviceptr scratch,
                                      uint64_t chunk, uint64_t stride, uint32_t piece, uint32_t blocks, int32_t flags,
                                      sccl_CUstream stream, char *err, size_t len) {
  sccl_link_args a;
  memset(&a, 0, sizeof a);
  a.input = src;
  a.output = dst;
  a.scratch = scratch;
  a.reserved = flags;
  a.chunk_packs = (int32_t)(chunk / PACK);
  a.stride_packs = (int32_t)(stride / PACK);
  a.piece_packs = (int32_t)(piece / PACK);
  a.stagger = (int32_t)e->shared.ring_stagger;
  a.gather_stagger = (int32_t)e->shared.ring_gather_stagger;
  a.world = e->world;
  a.index = e->chain_index;
  a.prev = e->chain_prev;
  a.next = e->chain_next;
  a.rank = e->rank;
  a.lanes = e->lanes;
  a.slots = (int32_t)e->shared.link_slots;
  a.blocks_per_role = (int32_t)blocks;
  a.link_base = e->dev_base + e->link_off;
  a.counters = e->link_counters;
  a.piece_counters = e->piece_counters;
  a.ctrl_base = e->dev_base + e->ctrl_off;
  a.poison = e->counters + 4u * (unsigned)(1 + 2 * e->classes);
  a.slot_bytes = e->shared.link_slot_bytes;
  a.spin_limit = e->shared.spin_limit;
  for (int i = 0; i < SCCL_LINK_MAX_WORLD; ++i) a.order[i] = e->shared.chain_order[i] < 0 ? 0 : e->shared.chain_order[i];
  sccl_CUresult r = sccl_kp_launch_link(fn, &a, roles * blocks, link_threads(e), stream);
  if (r != SCCL_CUDA_SUCCESS) {
    put_error(err, len, "launching the %s: %s", what, sccl_cuda_result_text(r));
    return ncclUnhandledCudaError;
  }
  if (blocks <= 64) atomic_fetch_add_explicit(&e->ops_by_blocks[blocks], 1, memory_order_relaxed);
  return ncclSuccess;
}

static ncclResult_t launch_link(sccl_engine *e, int kind, int dtype, sccl_CUdeviceptr src, sccl_CUdeviceptr dst,
                                uint64_t chunk, uint64_t stride, uint32_t piece, uint32_t blocks, sccl_CUstream stream,
                                char *err, size_t len) {
  ncclResult_t result = launch_link_entry(e, e->link_fn[kind][dtype], link_roles[kind], link_kind_names[kind], src,
                                          dst, e->link_scratch, chunk, stride, piece, blocks, 0, stream, err, len);
  if (result != ncclSuccess) return result;
  atomic_fetch_add_explicit(&e->link_ops[kind], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&e->link_bytes, chunk * (uint64_t)e->world, memory_order_relaxed);
  return ncclSuccess;
}

/* Whether a block of `chunk` bytes per rank goes as one pair exchange: two ranks, the exchange entry
 * loaded, and the block an all-gather shard that the ring all-gather would carry (whole packs, the ring
 * schedule from its minimum). Agreed settings and sizes only, so both ranks decide alike. */
static int pair_exchange_ok(const sccl_engine *e, uint64_t chunk);
static ncclResult_t exchange_staged(sccl_engine *e, sccl_CUdeviceptr sent, sccl_CUdeviceptr received, uint64_t chunk,
                                    int capturing, sccl_CUstream stream, char *err, size_t len);
static ncclResult_t exchange_root(sccl_engine *e, sccl_CUdeviceptr sent, sccl_CUdeviceptr own, sccl_CUdeviceptr dst,
                                  uint64_t chunk, int capturing, sccl_CUstream stream, char *err, size_t len);

/* One pair exchange op (sircl_ring_exchange): `chunk` bytes each way, in the all-gather plan's pieces and
 * blocks for a shard of that size. `input` goes to the peer (0: the slot goes out unchanged), the peer's
 * block lands at output + peer * stride (output 0, or EXCHANGE_DISCARD in `flags`: discarded), and
 * `local` (0: none) is copied to output + rank * stride by a role of its own. `reduce_dtype` >= 0 selects
 * the reduce-to-root form instead: output = the dtype rounding of `local` + the peer's block. Buffers
 * 16-byte aligned. */
enum { EXCHANGE_DISCARD = 1 };
static ncclResult_t launch_exchange(sccl_engine *e, sccl_CUdeviceptr input, sccl_CUdeviceptr output,
                                    sccl_CUdeviceptr local, uint64_t chunk, uint64_t stride, int32_t flags,
                                    int reduce_dtype, sccl_CUstream stream, char *err, size_t len) {
  sccl_CUfunction fn = reduce_dtype >= 0 ? e->exchange_reduce_fn[reduce_dtype] : e->exchange_fn;
  ncclResult_t result = launch_link_entry(e, fn, 3, reduce_dtype >= 0 ? "pair reduce exchange" : "pair exchange",
                                          input, output, local, chunk, stride, link_piece(e, COLL_GATHER, chunk),
                                          link_blocks_for(e, COLL_GATHER, chunk), flags, stream, err, len);
  if (result != ncclSuccess) return result;
  atomic_fetch_add_explicit(&e->exchange_ops, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&e->exchange_bytes, chunk, memory_order_relaxed);
  return ncclSuccess;
}

/* A 16-byte-aligned staging buffer of at least `bytes` for a link or chain op whose buffers are not aligned
 * (or overlap where the op needs them apart, or whose output is discarded). Whether a rank stages depends
 * on its own buffers and role, so staging never decides whether a rank launches: outside capture the
 * communicator's buffer grows (the earlier one kept, see retired_stage); under capture a call it cannot
 * hold takes a graph allocation on the capturing stream, which end_call frees on that stream. Without
 * stream-ordered allocation (shared.graph_staging, agreed at setup) such a capture is refused on the rank
 * that needs the staging, until one eager call of that size has grown its buffer. */
static ncclResult_t stage_buffer(sccl_engine *e, uint64_t bytes, int capturing, sccl_CUstream stream,
                                 sccl_CUdeviceptr *out, char *err, size_t len) {
  if (bytes > e->stage_bytes && capturing) {
    if (!e->shared.graph_staging) {
      put_error(err, len, "a link or chain op on unaligned or overlapping buffers needs %" PRIu64 " bytes of "
                "staging, and this driver or device has no stream-ordered allocation for CUDA graph capture; "
                "issue one such call eagerly before capturing", bytes);
      return ncclInvalidUsage;
    }
    if (e->call_stage && e->call_stage_bytes < bytes) {
      if (cuda_check(cu->MemFreeAsync(e->call_stage, e->call_stage_stream), "freeing a graph staging allocation",
                     err, len))
        return ncclUnhandledCudaError;
      e->call_stage = 0;
    }
    if (!e->call_stage) {
      if (cuda_check(cu->MemAllocAsync(&e->call_stage, bytes, stream), "allocating graph staging", err, len)) {
        e->call_stage = 0;
        return ncclUnhandledCudaError;
      }
      e->call_stage_bytes = bytes;
      e->call_stage_stream = stream;
      atomic_fetch_add_explicit(&e->graph_staged, 1, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&e->link_staged, 1, memory_order_relaxed);
    *out = e->call_stage;
    return ncclSuccess;
  }
  if (bytes > e->stage_bytes) {
    if (e->stage && e->retired == e->retired_room) {
      int room = e->retired_room ? 2 * e->retired_room : 8;
      sccl_CUdeviceptr *grown = realloc(e->retired_stage, (size_t)room * sizeof *grown);
      if (!grown) {
        put_error(err, len, "out of host memory for the staging buffer list");
        return ncclSystemError;
      }
      e->retired_stage = grown;
      e->retired_room = room;
    }
    uint64_t size = e->stage_bytes * 2 > bytes ? e->stage_bytes * 2 : bytes;
    sccl_CUdeviceptr fresh = 0;
    if (cuda_check(cu->MemAlloc(&fresh, size), "allocating the link staging buffer", err, len))
      return ncclUnhandledCudaError;
    if (e->stage) e->retired_stage[e->retired++] = e->stage;
    e->stage = fresh;
    e->stage_bytes = size;
  }
  atomic_fetch_add_explicit(&e->link_staged, 1, memory_order_relaxed);
  *out = e->stage;
  return ncclSuccess;
}

/* The schedule `schedule` as it runs for a collective of `nbytes` (COLL_*): ring runs as auto without the
 * ring or below its ring minimum (SIRCL's _schedule); the same on every rank for the same size. */
static uint32_t effective_schedule(const sccl_engine *e, uint32_t schedule, uint64_t nbytes, int collective) {
  if (schedule == SCHEDULE_RING && (!e->shared.ring_on || nbytes < e->shared.ring_mins[collective]))
    return SCHEDULE_AUTO;
  return schedule;
}

static int gather_link_kind(const sccl_engine *e, uint64_t shard);
static int pair_exchange_ok(const sccl_engine *e, uint64_t chunk) {
  return e->world == 2 && e->exchange_fn && gather_link_kind(e, chunk) == SCCL_RING_GATHER;
}

/* The link op an all-gather of `shard` bytes per rank runs as (SCCL_RING_GATHER, SCCL_LINK_GATHER), or -1:
 * every rank's shard a positive multiple of 16 bytes with an output below 2^31 bytes (SIRCL's gather shape);
 * then the ring schedule from its minimum, the chain schedule, or auto from the chain minimum. Decided from
 * agreed settings and sizes only. */
static int gather_link_kind(const sccl_engine *e, uint64_t shard) {
  uint64_t output = shard * (uint64_t)e->world;
  if (!e->link_on || shard == 0 || shard % PACK || output >= (1ull << 31)) return -1;
  uint32_t schedule = effective_schedule(e, e->shared.gather_schedule, output, COLL_GATHER);
  if (schedule == SCHEDULE_RING) return SCCL_RING_GATHER;
  if (schedule == SCHEDULE_PIECES) return -1;
  return schedule == SCHEDULE_CHAIN || output >= e->shared.chain_mins[COLL_GATHER] ? SCCL_LINK_GATHER : -1;
}

/* The link op a reduce-scatter of `chunk` bytes per rank runs as (SCCL_RING_SCATTER, SCCL_LINK_SCATTER), or
 * -1; chunks a multiple of 16 bytes. */
static int scatter_link_kind(const sccl_engine *e, uint64_t chunk) {
  uint64_t input = chunk * (uint64_t)e->world;
  if (!e->link_on || chunk == 0 || chunk % PACK) return -1;
  uint32_t schedule = effective_schedule(e, e->shared.scatter_schedule, input, COLL_SCATTER);
  if (schedule == SCHEDULE_RING) return SCCL_RING_SCATTER;
  if (schedule == SCHEDULE_PIECES) return -1;
  return schedule == SCHEDULE_CHAIN || input >= e->shared.chain_mins[COLL_SCATTER] ? SCCL_LINK_SCATTER : -1;
}

/* The all-gather as one link op: unaligned buffers through the staging buffer (the shard, then the output). */
static ncclResult_t gather_link(sccl_engine *e, int kind, sccl_CUdeviceptr src, sccl_CUdeviceptr dst, uint64_t shard,
                                int capturing, sccl_CUstream stream, char *err, size_t len) {
  uint64_t output = shard * (uint64_t)e->world;
  uint32_t piece = link_piece(e, COLL_GATHER, shard), blocks = link_blocks_for(e, COLL_GATHER, shard);
  if (src % PACK == 0 && dst % PACK == 0)
    return launch_link(e, kind, 0, src, dst, shard, shard, piece, blocks, stream, err, len);
  sccl_CUdeviceptr stage;
  ncclResult_t result = stage_buffer(e, shard + output, capturing, stream, &stage, err, len);
  if (result != ncclSuccess) return result;
  if (cuda_check(copy_async(e, stage, src, shard, stream), "staging the shard", err, len))
    return ncclUnhandledCudaError;
  result = launch_link(e, kind, 0, stage, stage + shard, shard, shard, piece, blocks, stream, err, len);
  if (result == ncclSuccess &&
      cuda_check(copy_async(e, dst, stage + shard, output, stream), "returning the gathered shards", err, len))
    result = ncclUnhandledCudaError;
  return result;
}

/* The reduce-scatter as link ops over column tiles of every chunk (LIBSIRCL_LINK_TILE_BYTES); each
 * element's arithmetic does not depend on the tiles. Unaligned buffers, and an output inside the input (in
 * place), go through the staging buffer per tile. */
static ncclResult_t scatter_link(sccl_engine *e, int kind, int dtype, sccl_CUdeviceptr src, sccl_CUdeviceptr dst,
                                 uint64_t chunk, int capturing, sccl_CUstream stream, char *err, size_t len) {
  uint64_t input = chunk * (uint64_t)e->world, tile = e->shared.link_tile;
  uint32_t piece = link_piece(e, COLL_SCATTER, input), blocks = link_blocks_for(e, COLL_SCATTER, input);
  int direct = src % PACK == 0 && dst % PACK == 0 && (dst + chunk <= src || src + input <= dst);
  ncclResult_t result = ncclSuccess;
  sccl_CUdeviceptr stage = 0;
  if (!direct) {
    uint64_t width = chunk < tile ? chunk : tile;
    result = stage_buffer(e, width * (uint64_t)(e->world + 1), capturing, stream, &stage, err, len);
  }
  for (uint64_t column = 0; result == ncclSuccess && column < chunk; column += tile) {
    uint64_t width = chunk - column < tile ? chunk - column : tile;
    if (direct) {
      result = launch_link(e, kind, dtype, src + column, dst + column, width, chunk, piece, blocks, stream, err, len);
      continue;
    }
    /* Every rank's column of the tile, `width` bytes apart in the staging buffer, then the output. */
    for (int r = 0; result == ncclSuccess && r < e->world; ++r)
      if (cuda_check(copy_async(e, stage + (uint64_t)r * width, src + (uint64_t)r * chunk + column, width, stream),
                     "staging a chunk", err, len))
        result = ncclUnhandledCudaError;
    sccl_CUdeviceptr staged_out = stage + width * (uint64_t)e->world;
    if (result == ncclSuccess)
      result = launch_link(e, kind, dtype, stage, staged_out, width, width, piece, blocks, stream, err, len);
    if (result == ncclSuccess &&
        cuda_check(copy_async(e, dst + column, staged_out, width, stream), "returning the chunk", err, len))
      result = ncclUnhandledCudaError;
  }
  return result;
}

/* One chain or ring all-reduce op over `nbytes` (whole packs; for the ring W equal chunks): unaligned buffers
 * through the staging buffer. Both ops tolerate their input and output being one buffer. */
static ncclResult_t launch_chain(sccl_engine *e, int dtype, sccl_CUdeviceptr src, sccl_CUdeviceptr dst,
                                 uint64_t nbytes, sccl_CUstream stream, char *err, size_t len);
static ncclResult_t reduce_link(sccl_engine *e, int ring, int dtype, sccl_CUdeviceptr src, sccl_CUdeviceptr dst,
                                uint64_t nbytes, int capturing, sccl_CUstream stream, char *err, size_t len) {
  uint64_t chunk = nbytes / (uint64_t)e->world;
  sccl_CUdeviceptr in = src, out = dst;
  ncclResult_t result = ncclSuccess;
  /* A zero dst (ncclReduce off the root) or unaligned buffers: the op runs in place in the staging buffer
   * (a zero dst's result stays there, unread). */
  if (!dst || src % PACK || dst % PACK) {
    result = stage_buffer(e, nbytes, capturing, stream, &in, err, len);
    if (result != ncclSuccess) return result;
    out = in;
    if (cuda_check(copy_async(e, in, src, nbytes, stream), "staging the message", err, len))
      return ncclUnhandledCudaError;
  }
  result = ring ? launch_link(e, SCCL_RING_REDUCE, dtype, in, out, chunk, chunk, link_piece(e, COLL_REDUCE, nbytes),
                              link_blocks_for(e, COLL_REDUCE, nbytes), stream, err, len)
                : launch_chain(e, dtype, in, out, nbytes, stream, err, len);
  if (result == ncclSuccess && dst && out != dst &&
      cuda_check(copy_async(e, dst, out, nbytes, stream), "returning the message", err, len))
    result = ncclUnhandledCudaError;
  return result;
}

/* One chain op over `nbytes` (a positive multiple of 16) from src to dst, both 16-byte aligned. */
static ncclResult_t launch_chain(sccl_engine *e, int dtype, sccl_CUdeviceptr src, sccl_CUdeviceptr dst,
                                 uint64_t nbytes, sccl_CUstream stream, char *err, size_t len) {
  int32_t packs = (int32_t)(nbytes / PACK);
  sccl_chain_args a;
  memset(&a, 0, sizeof a);
  a.input = src;
  a.output = dst;
  a.a_packs = packs / 2;
  a.b_packs = packs - packs / 2;
  a.chunk_packs = (int32_t)(e->shared.chain_chunk / PACK);
  a.chain_base = e->dev_base + e->chain_off;
  a.counters = e->chain_counters;
  a.ctrl_base = e->dev_base + e->ctrl_off;
  a.poison = e->counters + 4u * (unsigned)(1 + 2 * e->classes);
  a.spin_limit = e->shared.spin_limit;
  a.world = e->world;
  a.index = e->chain_index;
  a.prev = e->chain_prev;
  a.next = e->chain_next;
  a.rank = e->rank;
  a.lanes = e->lanes;
  a.slots = (int32_t)e->shared.chain_slots;
  a.blocks_per_role = (int32_t)e->shared.chain_blocks;
  a.slot_bytes = e->shared.chain_slot_bytes;
  sccl_CUresult r = sccl_kp_launch_chain(e->chain_fn[dtype], &a, 4u * e->shared.chain_blocks, link_threads(e),
                                         stream);
  if (r != SCCL_CUDA_SUCCESS) {
    put_error(err, len, "launching the chain all-reduce: %s", sccl_cuda_result_text(r));
    return ncclUnhandledCudaError;
  }
  atomic_fetch_add_explicit(&e->chain_ops[dtype], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&e->chain_bytes, nbytes, memory_order_relaxed);
  return ncclSuccess;
}

/* One piece: stage unaligned pointers through scratch; a zero `dst` discards the result. */
static ncclResult_t launch_piece(sccl_engine *e, int dtype, sccl_CUdeviceptr src, sccl_CUdeviceptr dst,
                                 uint64_t nbytes, sccl_CUstream stream, char *err, size_t len) {
  sccl_CUdeviceptr kin = src, kout = dst;
  if (src % PACK) {
    atomic_fetch_add_explicit(&e->staged_unaligned, 1, memory_order_relaxed);
    if (cuda_check(copy_async(e, e->scratch[0], src, nbytes, stream), "staging an unaligned input", err, len))
      return ncclUnhandledCudaError;
    kin = e->scratch[0];
  }
  if (!dst || dst % PACK) kout = e->scratch[1];
  ncclResult_t result = launch_op(e, dtype, kin, kout, nbytes, stream, err, len);
  if (result == ncclSuccess && dst && kout != dst &&
      cuda_check(copy_async(e, dst, kout, nbytes, stream), "copying to an unaligned output", err, len))
    return ncclUnhandledCudaError;
  return result;
}

/* The per-call state every collective shares: the communicator's lock, its context made current,
 * and the stream's capture state. */
typedef struct {
  int pushed, capturing;
} call_state;

/* Lock the communicator, make its context current, read the stream's capture state and order the
 * call: outside a capture, a launch on another stream than the previous one waits for everything
 * queued on that stream; inside a capture, every collective of one capture uses one stream. */
static ncclResult_t begin_call(sccl_engine *e, const char *what, sccl_CUstream stream, call_state *call, char *err,
                               size_t len) {
  call->pushed = call->capturing = 0;
  pthread_mutex_lock(&e->lock);
  sccl_CUcontext current = NULL;
  if (cu->CtxGetCurrent(&current) != SCCL_CUDA_SUCCESS || current != e->ctx) {
    if (cuda_check(cu->CtxPushCurrent(e->ctx), "making the communicator's context current", err, len)) {
      pthread_mutex_unlock(&e->lock);
      return ncclUnhandledCudaError;
    }
    call->pushed = 1;
  }
  int status = SCCL_CU_STREAM_CAPTURE_STATUS_NONE;
  unsigned long long capture = 0;
  ncclResult_t result = ncclSuccess;
  if (cuda_check(cu->StreamGetCaptureInfo(stream, &status, &capture), "querying the stream's capture state", err,
                 len)) {
    result = ncclUnhandledCudaError;
  } else if (status == SCCL_CU_STREAM_CAPTURE_STATUS_INVALIDATED) {
    put_error(err, len, "%s: the stream's capture is invalidated", what);
    result = ncclInvalidUsage;
  } else if (status == SCCL_CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    call->capturing = 1;
    if (capture != e->capture_id) {
      e->capture_id = capture;
      e->capture_stream = stream;
    } else if (stream != e->capture_stream) {
      atomic_fetch_add_explicit(&e->refused_capture, 1, memory_order_relaxed);
      put_error(err, len, "%s: the collectives of one CUDA graph capture must use one stream", what);
      result = ncclInvalidUsage;
    }
  } else if (e->world > 1) {
    if (e->have_last && stream != e->last_stream &&
        (cuda_check(cu->EventRecord(e->order_event, e->last_stream), "recording the order event", err, len) ||
         cuda_check(cu->StreamWaitEvent(stream, e->order_event, 0), "ordering the stream", err, len)))
      result = ncclUnhandledCudaError;
    e->last_stream = stream;
    e->have_last = 1;
  }
  if (result != ncclSuccess) {
    sccl_CUcontext popped;
    if (call->pushed) cu->CtxPopCurrent(&popped);
    pthread_mutex_unlock(&e->lock);
  }
  return result;
}

/* End a call begun by begin_call: the call's graph staging allocation is freed on the capturing stream
 * (a free node after the call's ops), the receipt refreshed, the context restored and the communicator
 * unlocked. Returns `result`, or the free's failure. */
static void refresh_receipt(sccl_engine *e);
static ncclResult_t end_call(sccl_engine *e, call_state *call, ncclResult_t result, char *err, size_t len) {
  if (e->call_stage) {
    sccl_CUresult freed = cu->MemFreeAsync(e->call_stage, e->call_stage_stream);
    e->call_stage = 0;
    if (freed != SCCL_CUDA_SUCCESS && result == ncclSuccess) {
      put_error(err, len, "freeing a graph staging allocation: %s", sccl_cuda_result_text(freed));
      result = ncclUnhandledCudaError;
    }
  }
  refresh_receipt(e);
  sccl_CUcontext popped;
  if (call->pushed) cu->CtxPopCurrent(&popped);
  pthread_mutex_unlock(&e->lock);
  return result;
}

static ncclResult_t check_args(sccl_engine *e, const char *what, const void *sendbuff, void *recvbuff, size_t count,
                               size_t item, char *err, size_t len) {
  if (count > SIZE_MAX / item / (size_t)e->world) {
    put_error(err, len, "%s: count overflows", what);
    return ncclInvalidArgument;
  }
  if (!sendbuff || !recvbuff) {
    put_error(err, len, "%s: NULL buffer", what);
    return ncclInvalidArgument;
  }
  char message[400];
  ncclResult_t health = sccl_engine_async_error(e, message, sizeof message);
  if (health != ncclSuccess) put_error(err, len, "%s", message);
  return health;
}

/* One all-gather tile: `piece` bytes (a multiple of 16) of every rank's shard, starting at column `column`
 * of shards that sit `shard_stride` bytes apart in the output. */
static ncclResult_t launch_gather(sccl_engine *e, sccl_CUdeviceptr src, sccl_CUdeviceptr dst, uint64_t piece,
                                  uint64_t shard_stride, unsigned cap, sccl_CUstream stream, char *err, size_t len) {
  uint64_t packs = piece / PACK;
  unsigned grid = grid_blocks(packs, e->shared.threads, cap, e->shared.packs_per_thread);
  int cls = grid_class(grid);
  sccl_allgather_args a;
  memset(&a, 0, sizeof a);
  a.input = src;
  a.output = dst;
  a.shard_packs = (int32_t)packs;
  a.nbytes = (int32_t)piece;
  a.tile_cols = (int32_t)packs;
  a.in_row_stride = (int64_t)packs;
  a.out_row_stride = (int64_t)(shard_stride / PACK) * e->world;
  a.out_src_stride = (int64_t)(shard_stride / PACK);
  a.recv_base = e->dev_base + e->recv_off;
  a.flag_base = e->dev_base + e->flag_off;
  a.send_base = e->dev_base + e->send_off;
  a.ctrl_base = e->dev_base + e->ctrl_off;
  a.slot_bytes = e->shared.slot_bytes;
  a.epoch = e->counters;
  a.stage_counter = e->counters + 4u * (unsigned)(1 + cls);
  a.tail_counter = e->counters + 4u * (unsigned)(1 + e->classes + cls);
  a.poison = e->counters + 4u * (unsigned)(1 + 2 * e->classes);
  a.arrival = e->arrival;
  a.spin_limit = e->shared.spin_limit;
  a.rank = e->rank;
  a.lanes = e->lanes;
  a.one_block = (int32_t)e->shared.one_block;
  sccl_CUresult r = sccl_kp_launch_allgather(e->fn[SCCL_K_ALLGATHER][0], &a, grid, e->shared.threads, stream);
  if (r != SCCL_CUDA_SUCCESS) {
    put_error(err, len, "launching the all-gather: %s", sccl_cuda_result_text(r));
    return ncclUnhandledCudaError;
  }
  atomic_fetch_add_explicit(&e->gather_ops, 1, memory_order_relaxed);
  return ncclSuccess;
}

ncclResult_t sccl_engine_allgather(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, cudaStream_t stream_handle, char *err, size_t len) {
  size_t item = nccl_type_size(datatype);
  if (!item) {
    put_error(err, len, "ncclAllGather: invalid datatype %d", (int)datatype);
    return ncclInvalidArgument;
  }
  if (count == 0) return ncclSuccess;
  ncclResult_t result = check_args(e, "ncclAllGather", sendbuff, recvbuff, count, item, err, len);
  if (result != ncclSuccess) return result;
  sccl_CUstream stream = (sccl_CUstream)stream_handle;
  uint64_t shard = (uint64_t)count * item;
  sccl_CUdeviceptr src = (sccl_CUdeviceptr)(uintptr_t)sendbuff, dst = (sccl_CUdeviceptr)(uintptr_t)recvbuff;
  call_state call;
  result = begin_call(e, "ncclAllGather", stream, &call, err, len);
  if (result != ncclSuccess) return result;
  int link = e->world > 1 ? gather_link_kind(e, shard) : -1;
  if (e->world == 1) {
    if (src != dst && cuda_check(copy_async(e, dst, src, shard, stream), "one-rank all-gather copy", err, len))
      result = ncclUnhandledCudaError;
  } else if (link >= 0) {
    result = gather_link(e, link, src, dst, shard, call.capturing, stream, err, len);
  } else {
    /* Tiles of at most one slot of every rank's shard. A tile that is not whole 16-byte packs, or not
     * aligned, travels from scratch[0] padded to 16 bytes; an output that cannot take whole packs at
     * every shard's place receives the gathered rows in scratch[1], copied to their places. */
    uint64_t tile = gather_tile(e);
    unsigned cap = shard <= tile ? e->shared.blocks : e->shared.large_blocks;
    int direct_out = dst % PACK == 0 && shard % PACK == 0;
    if (!direct_out || src % PACK) atomic_fetch_add_explicit(&e->gather_padded, 1, memory_order_relaxed);
    for (uint64_t column = 0; result == ncclSuccess && column < shard; column += tile) {
      uint64_t piece = shard - column < tile ? shard - column : tile;
      uint64_t padded = (piece + PACK - 1) / PACK * PACK;
      sccl_CUdeviceptr in = src + column;
      if (in % PACK || piece % PACK) {
        if (cuda_check(copy_async(e, e->scratch[0], in, piece, stream), "staging the shard", err, len)) {
          result = ncclUnhandledCudaError;
          break;
        }
        in = e->scratch[0];
      }
      if (direct_out) {
        result = launch_gather(e, in, dst + column, piece, shard, cap, stream, err, len);
        continue;
      }
      result = launch_gather(e, in, e->scratch[1], padded, padded, cap, stream, err, len);
      for (int r = 0; result == ncclSuccess && r < e->world; ++r)
        if (cuda_check(copy_async(e, dst + (uint64_t)r * shard + column, e->scratch[1] + (uint64_t)r * padded, piece,
                                  stream),
                       "placing a gathered shard", err, len))
          result = ncclUnhandledCudaError;
    }
  }
  if (result == ncclSuccess) {
    atomic_fetch_add_explicit(&e->gather_calls, 1, memory_order_relaxed);
    if (call.capturing) atomic_fetch_add_explicit(&e->captured_calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&e->gather_bytes, shard * (uint64_t)e->world, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

/* One scatter op over `piece` bytes (a multiple of 16) of every rank's chunk, input chunks `src_stride` bytes
 * apart: the reduce-scatter kernel of a dtype reduces the own chunk into `dst`; the all-to-all kernel places
 * the chunk from rank s at `dst` + s * `dst_stride`. */
static ncclResult_t launch_scatter(sccl_engine *e, sccl_CUfunction fn, _Atomic uint64_t *counter, sccl_CUdeviceptr src,
                                   sccl_CUdeviceptr dst, uint64_t piece, uint64_t src_stride, uint64_t dst_stride,
                                   sccl_CUstream stream, char *err, size_t len) {
  uint64_t packs = piece / PACK * (uint64_t)e->world;
  unsigned grid = grid_blocks(packs, e->shared.threads, e->shared.large_blocks, e->shared.packs_per_thread);
  int cls = grid_class(grid);
  sccl_scatter_args a;
  memset(&a, 0, sizeof a);
  a.input = src;
  a.output = dst;
  a.size_packs = (int32_t)packs;
  a.nbytes = (int32_t)(packs * PACK);
  a.chunk_packs = (int32_t)(piece / PACK);
  a.src_stride = (int64_t)src_stride;
  a.dst_stride = (int64_t)dst_stride;
  a.recv_base = e->dev_base + e->recv_off;
  a.flag_base = e->dev_base + e->flag_off;
  a.send_base = e->dev_base + e->send_off;
  a.ctrl_base = e->dev_base + e->ctrl_off;
  a.slot_bytes = e->shared.slot_bytes;
  a.epoch = e->counters;
  a.stage_counter = e->counters + 4u * (unsigned)(1 + cls);
  a.tail_counter = e->counters + 4u * (unsigned)(1 + e->classes + cls);
  a.poison = e->counters + 4u * (unsigned)(1 + 2 * e->classes);
  a.spin_limit = e->shared.spin_limit;
  a.rank = e->rank;
  a.lanes = e->lanes;
  sccl_CUresult r = sccl_kp_launch_scatter(fn, &a, grid, e->shared.threads, stream);
  if (r != SCCL_CUDA_SUCCESS) {
    put_error(err, len, "launching a scatter op: %s", sccl_cuda_result_text(r));
    return ncclUnhandledCudaError;
  }
  atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
  return ncclSuccess;
}

/* Stage the `piece` bytes at column `column` of each of the W chunks (`chunk` bytes apart from `src`) into
 * scratch[0] at a pitch of `padded` bytes. */
static ncclResult_t stage_chunks(sccl_engine *e, sccl_CUdeviceptr src, uint64_t chunk, uint64_t column, uint64_t piece,
                                 uint64_t padded, sccl_CUstream stream, char *err, size_t len) {
  for (int r = 0; r < e->world; ++r)
    if (cuda_check(copy_async(e, e->scratch[0] + (uint64_t)r * padded, src + (uint64_t)r * chunk + column, piece,
                              stream),
                   "staging a chunk", err, len))
      return ncclUnhandledCudaError;
  return ncclSuccess;
}

/* The generic reduce-scatter: tiles of every rank's chunks exchanged by the all-to-all kernel into the
 * rows of scratch[1], then folded in rank order into `dst`. */
static ncclResult_t fold_scatter(sccl_engine *e, ncclDataType_t datatype, ncclRedOp_t op, size_t item,
                                 sccl_CUdeviceptr src, sccl_CUdeviceptr dst, uint64_t chunk, sccl_CUstream stream,
                                 char *err, size_t len) {
  uint64_t tile = scatter_tile(e);
  int direct_in = src % PACK == 0 && chunk % PACK == 0;
  ncclResult_t result = ncclSuccess;
  for (uint64_t column = 0; result == ncclSuccess && column < chunk; column += tile) {
    uint64_t piece = chunk - column < tile ? chunk - column : tile;
    uint64_t padded = (piece + PACK - 1) / PACK * PACK;
    sccl_CUdeviceptr in = src + column;
    uint64_t stride = chunk;
    if (!direct_in) {
      result = stage_chunks(e, src, chunk, column, piece, padded, stream, err, len);
      in = e->scratch[0];
      stride = padded;
    }
    if (result == ncclSuccess)
      result = launch_scatter(e, e->fn[SCCL_K_ALLTOALL][0], &e->scatter_ops, in, e->scratch[1], padded, stride,
                              padded, stream, err, len);
    if (result == ncclSuccess)
      result = fold_into(e, datatype, op, item, e->scratch[1], padded, dst + column, piece / item, stream, err, len);
  }
  return result;
}

/* The generic all-reduce: tiles of every rank's bytes all-gathered into the rows of scratch[1], then folded
 * in rank order into `dst` (a zero `dst` discards). */
static ncclResult_t fold_reduce(sccl_engine *e, ncclDataType_t datatype, ncclRedOp_t op, size_t item,
                                sccl_CUdeviceptr src, sccl_CUdeviceptr dst, uint64_t nbytes, sccl_CUstream stream,
                                char *err, size_t len) {
  uint64_t tile = gather_tile(e);
  unsigned cap = nbytes <= tile ? e->shared.blocks : e->shared.large_blocks;
  ncclResult_t result = ncclSuccess;
  for (uint64_t column = 0; result == ncclSuccess && column < nbytes; column += tile) {
    uint64_t piece = nbytes - column < tile ? nbytes - column : tile;
    uint64_t padded = (piece + PACK - 1) / PACK * PACK;
    sccl_CUdeviceptr in = src + column;
    if (in % PACK || piece % PACK) {
      if (cuda_check(copy_async(e, e->scratch[0], in, piece, stream), "staging the input", err, len))
        return ncclUnhandledCudaError;
      in = e->scratch[0];
    }
    result = launch_gather(e, in, e->scratch[1], padded, padded, cap, stream, err, len);
    if (result == ncclSuccess && dst)
      result = fold_into(e, datatype, op, item, e->scratch[1], padded, dst + column, piece / item, stream, err, len);
  }
  return result;
}

/* The reduction a call names: the transport kernels' own (float16, bfloat16, float32 sum) or the fold
 * pack's. Returns the transport dtype, SCCL_FOLD_PATH for the fold, or -1 with an error for an op that is
 * not built in. */
enum { SCCL_FOLD_PATH = 100 };
static int reduction_path(sccl_engine *e, const char *what, ncclDataType_t datatype, ncclRedOp_t op, size_t *item,
                          char *err, size_t len) {
  *item = nccl_type_size(datatype);
  if (!*item) {
    put_error(err, len, "%s: invalid datatype %d", what, (int)datatype);
    return -1;
  }
  if ((int)op < 0 || (int)op >= SCCL_FOLD_OPS) {
    atomic_fetch_add_explicit(&e->refused_op, 1, memory_order_relaxed);
    put_error(err, len, "%s: reduction op %d is not a built-in op (ncclRedOpCreatePreMulSum is unsupported)", what,
              (int)op);
    return -1;
  }
  size_t size = 0;
  int dtype = kernel_dtype(datatype, &size);
  return dtype >= 0 && op == ncclSum ? dtype : SCCL_FOLD_PATH;
}

ncclResult_t sccl_engine_reducescatter(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream_handle, char *err,
                                       size_t len) {
  size_t item = 0;
  int dtype = reduction_path(e, "ncclReduceScatter", datatype, op, &item, err, len);
  if (dtype < 0) return ncclInvalidArgument;
  if (count == 0) return ncclSuccess;
  ncclResult_t result = check_args(e, "ncclReduceScatter", sendbuff, recvbuff, count, item, err, len);
  if (result != ncclSuccess) return result;
  sccl_CUstream stream = (sccl_CUstream)stream_handle;
  uint64_t chunk = (uint64_t)count * item;
  sccl_CUdeviceptr src = (sccl_CUdeviceptr)(uintptr_t)sendbuff, dst = (sccl_CUdeviceptr)(uintptr_t)recvbuff;
  call_state call;
  result = begin_call(e, "ncclReduceScatter", stream, &call, err, len);
  if (result != ncclSuccess) return result;
  /* An op carries the same byte range of every chunk; its W pieces fit one slot and the large-message piece. */
  uint64_t op_bytes = e->shared.large_piece < e->shared.slot_bytes ? e->shared.large_piece : e->shared.slot_bytes;
  uint64_t tile = op_bytes / (uint64_t)e->world / PACK * PACK;
  if (e->world == 1) {
    if (src != dst && cuda_check(copy_async(e, dst, src, chunk, stream), "one-rank reduce-scatter copy", err, len))
      result = ncclUnhandledCudaError;
  } else if (dtype == SCCL_FOLD_PATH) {
    result = fold_scatter(e, datatype, op, item, src, dst, chunk, stream, err, len);
  } else if (scatter_link_kind(e, chunk) >= 0) {
    result = scatter_link(e, scatter_link_kind(e, chunk), dtype, src, dst, chunk, call.capturing, stream, err, len);
  } else if (chunk % PACK == 0 && src % PACK == 0 && dst % PACK == 0) {
    for (uint64_t column = 0; result == ncclSuccess && column < chunk; column += tile) {
      uint64_t piece = chunk - column < tile ? chunk - column : tile;
      result = launch_scatter(e, e->fn[SCCL_K_SCATTER][dtype], &e->scatter_ops, src + column, dst + column, piece,
                              chunk, piece, stream, err, len);
    }
  } else {
    /* Padded: every rank's slice of the tile is staged in scratch at a 16-byte pitch; the reduced own
     * slice is copied back. */
    atomic_fetch_add_explicit(&e->scatter_padded, 1, memory_order_relaxed);
    for (uint64_t column = 0; result == ncclSuccess && column < chunk; column += tile) {
      uint64_t piece = chunk - column < tile ? chunk - column : tile;
      uint64_t padded = (piece + PACK - 1) / PACK * PACK;
      result = stage_chunks(e, src, chunk, column, piece, padded, stream, err, len);
      if (result == ncclSuccess)
        result = launch_scatter(e, e->fn[SCCL_K_SCATTER][dtype], &e->scatter_ops, e->scratch[0], e->scratch[1],
                                padded, padded, padded, stream, err, len);
      if (result == ncclSuccess &&
          cuda_check(copy_async(e, dst + column, e->scratch[1], piece, stream), "returning the chunk", err, len))
        result = ncclUnhandledCudaError;
    }
  }
  if (result == ncclSuccess) {
    atomic_fetch_add_explicit(&e->scatter_calls, 1, memory_order_relaxed);
    if (call.capturing) atomic_fetch_add_explicit(&e->captured_calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&e->scatter_bytes, chunk * (uint64_t)e->world, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

/* The reduction of every rank's `count` elements into `recvbuff`, or discarded when `discard` is set
 * (ncclReduce on a rank other than the root, whose output stays in scratch). */
static ncclResult_t reduce_into(sccl_engine *e, const char *what, const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op, int discard, cudaStream_t stream_handle,
                                char *err, size_t len) {
  size_t item = 0;
  int dtype = reduction_path(e, what, datatype, op, &item, err, len);
  if (dtype < 0) return ncclInvalidArgument;
  if (count == 0) return ncclSuccess;
  if (count > SIZE_MAX / item) {
    put_error(err, len, "%s: count overflows", what);
    return ncclInvalidArgument;
  }
  if (!sendbuff || (!recvbuff && !discard)) {
    put_error(err, len, "%s: NULL buffer", what);
    return ncclInvalidArgument;
  }
  char message[400];
  ncclResult_t health = sccl_engine_async_error(e, message, sizeof message);
  if (health != ncclSuccess) {
    put_error(err, len, "%s", message);
    return health;
  }
  sccl_CUstream stream = (sccl_CUstream)stream_handle;
  uint64_t nbytes = (uint64_t)count * item;
  sccl_CUdeviceptr src = (sccl_CUdeviceptr)(uintptr_t)sendbuff;
  sccl_CUdeviceptr dst = discard ? 0 : (sccl_CUdeviceptr)(uintptr_t)recvbuff;
  call_state call;
  ncclResult_t result = begin_call(e, what, stream, &call, err, len);
  if (result != ncclSuccess) return result;
  int capturing = call.capturing;
  if (e->world == 1) {
    if (dst && src != dst && cuda_check(copy_async(e, dst, src, nbytes, stream), "one-rank copy", err, len))
      result = ncclUnhandledCudaError;
    goto counted;
  }
  if (dtype == SCCL_FOLD_PATH) {
    result = fold_reduce(e, datatype, op, item, src, dst, nbytes, stream, err, len);
    goto counted;
  }
  if (!strcmp(what, "ncclReduce") && op == ncclSum && nbytes % PACK == 0 && pair_exchange_ok(e, nbytes)) {
    /* A pair: the other rank streams its values one way; the root's kernel stores the dtype rounding of its
     * own values plus each arriving piece (two operands, the rank-order sum's bits). Unaligned root buffers
     * go through the staging buffer. */
    if (discard) {
      result = exchange_staged(e, src, 0, nbytes, capturing, stream, err, len);
    } else {
      sccl_CUdeviceptr own = src, out = dst, stage = 0;
      if (src % PACK || dst % PACK) {
        result = stage_buffer(e, nbytes, capturing, stream, &stage, err, len);
        if (result == ncclSuccess &&
            cuda_check(copy_async(e, stage, src, nbytes, stream), "staging the root's values", err, len))
          result = ncclUnhandledCudaError;
        own = out = stage;
      }
      if (result == ncclSuccess) result = launch_exchange(e, 0, out, own, nbytes, 0, 0, dtype, stream, err, len);
      if (result == ncclSuccess && out != dst &&
          cuda_check(copy_async(e, dst, out, nbytes, stream), "returning the sum", err, len))
        result = ncclUnhandledCudaError;
    }
    goto counted;
  }
  uint64_t body = nbytes / PACK * PACK, tail = nbytes - body, start = 0;
  /* SIRCL's plan (pieces.reduce_plan, _large_reduce_plan): under the ring schedule (from its minimum) one
   * ring op for the largest prefix of W equal chunks of whole packs; else under the chain schedule (or auto
   * from the chain minimum) one chain op for the 16-byte-aligned body; the rest in pieces, then the
   * zero-padded tail. Decided from agreed settings and sizes only, for calls that keep their result on
   * every rank (not ncclReduce); unaligned buffers go through the staging buffer. */
  if (body && e->shared.large_schedule != SCHEDULE_PIECES) {
    uint32_t schedule = effective_schedule(e, e->shared.large_schedule, nbytes, COLL_REDUCE);
    uint64_t ring_bytes = body / (PACK * (uint64_t)e->world) * PACK * (uint64_t)e->world;
    if (schedule == SCHEDULE_RING && e->link_on && ring_bytes && ring_bytes / (uint64_t)e->world < (1ull << 31)) {
      result = reduce_link(e, 1, dtype, src, dst, ring_bytes, capturing, stream, err, len);
      start = ring_bytes;
    } else if (schedule != SCHEDULE_RING && e->chain_on &&
               (schedule == SCHEDULE_CHAIN || body >= e->shared.chain_mins[COLL_REDUCE])) {
      result = reduce_link(e, 0, dtype, src, dst, body, capturing, stream, err, len);
      start = body;
    }
  }
  for (uint64_t offset = start; result == ncclSuccess && offset < body; offset += e->shared.large_piece) {
    uint64_t piece = body - offset < e->shared.large_piece ? body - offset : e->shared.large_piece;
    /* A zero dst (ncclReduce off the root) discards every piece; it never becomes an offset address. */
    result = launch_piece(e, dtype, src + offset, dst ? dst + offset : 0, piece, stream, err, len);
  }
  body = nbytes / PACK * PACK;
  if (result == ncclSuccess && tail) {
    /* A tail below 16 bytes travels zero-padded as a one-shot op. */
    atomic_fetch_add_explicit(&e->padded_tails, 1, memory_order_relaxed);
    if (cuda_check(cu->MemsetD8Async(e->scratch[0], 0, PACK, stream), "padding the tail", err, len) ||
        cuda_check(copy_async(e, e->scratch[0], src + body, tail, stream), "staging the tail", err, len)) {
      result = ncclUnhandledCudaError;
    } else {
      result = launch_op(e, dtype, e->scratch[0], e->scratch[1], PACK, stream, err, len);
      if (result == ncclSuccess && dst &&
          cuda_check(copy_async(e, dst + body, e->scratch[1], tail, stream), "returning the tail", err, len))
        result = ncclUnhandledCudaError;
    }
  }
counted:
  if (result == ncclSuccess) {
    atomic_fetch_add_explicit(strcmp(what, "ncclReduce") ? &e->calls : &e->reduce_calls, 1, memory_order_relaxed);
    if (capturing) atomic_fetch_add_explicit(&e->captured_calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&e->bytes, nbytes, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

ncclResult_t sccl_engine_allreduce(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream, char *err,
                                   size_t len) {
  return reduce_into(e, "ncclAllReduce", sendbuff, recvbuff, count, datatype, op, 0, stream, err, len);
}

ncclResult_t sccl_engine_reduce(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op, int root, cudaStream_t stream, char *err,
                                size_t len) {
  if (root < 0 || root >= e->world) {
    put_error(err, len, "ncclReduce: root %d is outside 0-%d", root, e->world - 1);
    return ncclInvalidArgument;
  }
  return reduce_into(e, "ncclReduce", sendbuff, recvbuff, count, datatype, op, e->rank != root, stream, err, len);
}

ncclResult_t sccl_engine_broadcast(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, int root, cudaStream_t stream_handle, char *err,
                                   size_t len) {
  size_t item = nccl_type_size(datatype);
  if (!item) {
    put_error(err, len, "ncclBroadcast: invalid datatype %d", (int)datatype);
    return ncclInvalidArgument;
  }
  if (root < 0 || root >= e->world) {
    put_error(err, len, "ncclBroadcast: root %d is outside 0-%d", root, e->world - 1);
    return ncclInvalidArgument;
  }
  if (count == 0) return ncclSuccess;
  /* Only the root's send buffer is read; other ranks may pass NULL. */
  ncclResult_t result = check_args(e, "ncclBroadcast", e->rank == root ? sendbuff : recvbuff, recvbuff, count, item,
                                   err, len);
  if (result != ncclSuccess) return result;
  sccl_CUstream stream = (sccl_CUstream)stream_handle;
  uint64_t bytes = (uint64_t)count * item;
  sccl_CUdeviceptr src = (sccl_CUdeviceptr)(uintptr_t)sendbuff, dst = (sccl_CUdeviceptr)(uintptr_t)recvbuff;
  call_state call;
  result = begin_call(e, "ncclBroadcast", stream, &call, err, len);
  if (result != ncclSuccess) return result;
  if (e->world == 1) {
    if (src != dst && cuda_check(copy_async(e, dst, src, bytes, stream), "one-rank broadcast copy", err, len))
      result = ncclUnhandledCudaError;
  } else if (pair_exchange_ok(e, bytes)) {
    /* A pair: one one-way exchange from the root, whose kernel also copies its bytes to its own output. */
    result = e->rank == root ? exchange_root(e, src, src, dst, bytes, call.capturing, stream, err, len)
                             : exchange_staged(e, 0, dst, bytes, call.capturing, stream, err, len);
  } else {
    /* An all-gather of tiles of the root's bytes through scratch; every rank keeps the root's tile. The
     * other ranks' tiles carry whatever scratch holds and are dropped. */
    uint64_t room = e->shared.slot_bytes / (uint64_t)e->world / PACK * PACK;
    uint64_t tile = room < e->shared.large_piece ? room : e->shared.large_piece / PACK * PACK;
    for (uint64_t column = 0; result == ncclSuccess && column < bytes; column += tile) {
      uint64_t piece = bytes - column < tile ? bytes - column : tile;
      uint64_t padded = (piece + PACK - 1) / PACK * PACK;
      if (e->rank == root &&
          cuda_check(copy_async(e, e->scratch[0], src + column, piece, stream), "staging the root's bytes", err, len)) {
        result = ncclUnhandledCudaError;
        break;
      }
      result = launch_gather(e, e->scratch[0], e->scratch[1], padded, padded, e->shared.large_blocks, stream, err,
                             len);
      if (result == ncclSuccess && cuda_check(copy_async(e, dst + column, e->scratch[1] + (uint64_t)root * padded,
                                                         piece, stream),
                                              "placing the root's bytes", err, len))
        result = ncclUnhandledCudaError;
    }
  }
  if (result == ncclSuccess) {
    atomic_fetch_add_explicit(&e->broadcast_calls, 1, memory_order_relaxed);
    if (call.capturing) atomic_fetch_add_explicit(&e->captured_calls, 1, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

static int overlaps(sccl_CUdeviceptr a, sccl_CUdeviceptr b, uint64_t bytes) { return a < b + bytes && b < a + bytes; }

ncclResult_t sccl_engine_alltoall(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                  ncclDataType_t datatype, cudaStream_t stream_handle, char *err, size_t len) {
  size_t item = nccl_type_size(datatype);
  if (!item) {
    put_error(err, len, "ncclAlltoAll: invalid datatype %d", (int)datatype);
    return ncclInvalidArgument;
  }
  if (count == 0) return ncclSuccess;
  ncclResult_t result = check_args(e, "ncclAlltoAll", sendbuff, recvbuff, count, item, err, len);
  if (result != ncclSuccess) return result;
  sccl_CUstream stream = (sccl_CUstream)stream_handle;
  uint64_t chunk = (uint64_t)count * item, total = chunk * (uint64_t)e->world;
  sccl_CUdeviceptr src = (sccl_CUdeviceptr)(uintptr_t)sendbuff, dst = (sccl_CUdeviceptr)(uintptr_t)recvbuff;
  call_state call;
  result = begin_call(e, "ncclAlltoAll", stream, &call, err, len);
  if (result != ncclSuccess) return result;
  if (e->world == 1) {
    if (src != dst && cuda_check(copy_async(e, dst, src, chunk, stream), "one-rank all-to-all copy", err, len))
      result = ncclUnhandledCudaError;
  } else if (pair_exchange_ok(e, chunk)) {
    /* A pair: one exchange sends the peer's block and copies the own block. Unaligned or overlapping
     * buffers go through the staging buffer (the input, and the output when unaligned). */
    sccl_CUdeviceptr in = src, out = dst;
    int staged = src % PACK || dst % PACK || overlaps(src, dst, total);
    if (staged) {
      sccl_CUdeviceptr stage = 0;
      result = stage_buffer(e, 2 * total, call.capturing, stream, &stage, err, len);
      if (result == ncclSuccess && cuda_check(copy_async(e, stage, src, total, stream), "staging the input", err, len))
        result = ncclUnhandledCudaError;
      in = stage;
      if (dst % PACK) out = stage + total;
    }
    uint64_t peer = (uint64_t)(1 - e->rank), self = (uint64_t)e->rank;
    if (result == ncclSuccess)
      result = launch_exchange(e, in + peer * chunk, out, in + self * chunk, chunk, chunk, 0, -1, stream, err, len);
    if (result == ncclSuccess && out != dst &&
        cuda_check(copy_async(e, dst, out, total, stream), "returning the output", err, len))
      result = ncclUnhandledCudaError;
  } else {
    /* Tiles of every chunk. Direct when every chunk is whole 16-byte packs, both buffers are aligned and they
     * do not overlap (a kernel writes received chunks while other blocks may still read the input); otherwise
     * the tiles travel through scratch. */
    uint64_t tile = scatter_tile(e);
    int direct = chunk % PACK == 0 && src % PACK == 0 && dst % PACK == 0 && !overlaps(src, dst, total);
    if (!direct) atomic_fetch_add_explicit(&e->alltoall_padded, 1, memory_order_relaxed);
    for (uint64_t column = 0; result == ncclSuccess && column < chunk; column += tile) {
      uint64_t piece = chunk - column < tile ? chunk - column : tile;
      uint64_t padded = (piece + PACK - 1) / PACK * PACK;
      if (direct) {
        result = launch_scatter(e, e->fn[SCCL_K_ALLTOALL][0], &e->alltoall_ops, src + column, dst + column, piece,
                                chunk, chunk, stream, err, len);
        continue;
      }
      result = stage_chunks(e, src, chunk, column, piece, padded, stream, err, len);
      if (result == ncclSuccess)
        result = launch_scatter(e, e->fn[SCCL_K_ALLTOALL][0], &e->alltoall_ops, e->scratch[0], e->scratch[1], padded,
                                padded, padded, stream, err, len);
      for (int r = 0; result == ncclSuccess && r < e->world; ++r)
        if (cuda_check(copy_async(e, dst + (uint64_t)r * chunk + column, e->scratch[1] + (uint64_t)r * padded, piece,
                                  stream),
                       "placing a received chunk", err, len))
          result = ncclUnhandledCudaError;
    }
  }
  if (result == ncclSuccess) {
    atomic_fetch_add_explicit(&e->alltoall_calls, 1, memory_order_relaxed);
    if (call.capturing) atomic_fetch_add_explicit(&e->captured_calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&e->alltoall_bytes, total, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

ncclResult_t sccl_engine_gather(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, int root, cudaStream_t stream_handle, char *err, size_t len) {
  size_t item = nccl_type_size(datatype);
  if (!item) {
    put_error(err, len, "ncclGather: invalid datatype %d", (int)datatype);
    return ncclInvalidArgument;
  }
  if (root < 0 || root >= e->world) {
    put_error(err, len, "ncclGather: root %d is outside 0-%d", root, e->world - 1);
    return ncclInvalidArgument;
  }
  if (count == 0) return ncclSuccess;
  /* Only the root's receive buffer is written; other ranks may pass NULL. */
  ncclResult_t result = check_args(e, "ncclGather", sendbuff, e->rank == root ? recvbuff : (void *)sendbuff, count,
                                   item, err, len);
  if (result != ncclSuccess) return result;
  sccl_CUstream stream = (sccl_CUstream)stream_handle;
  uint64_t shard = (uint64_t)count * item;
  sccl_CUdeviceptr src = (sccl_CUdeviceptr)(uintptr_t)sendbuff, dst = (sccl_CUdeviceptr)(uintptr_t)recvbuff;
  call_state call;
  result = begin_call(e, "ncclGather", stream, &call, err, len);
  if (result != ncclSuccess) return result;
  int link_kind = e->world > 1 ? gather_link_kind(e, shard) : -1;
  if (e->world == 1) {
    if (src != dst && cuda_check(copy_async(e, dst, src, shard, stream), "one-rank gather copy", err, len))
      result = ncclUnhandledCudaError;
  } else if (pair_exchange_ok(e, shard)) {
    /* A pair: the other rank sends its shard one way; the root's kernel receives it at its place and copies
     * the root's own shard to its place. Unaligned root buffers go through the staging buffer. */
    if (e->rank != root) {
      result = exchange_staged(e, src, 0, shard, call.capturing, stream, err, len);
    } else {
      sccl_CUdeviceptr own = src, out = dst, stage = 0;
      if (src % PACK || dst % PACK) {
        result = stage_buffer(e, 3 * shard, call.capturing, stream, &stage, err, len);
        if (result == ncclSuccess && src % PACK) {
          if (cuda_check(copy_async(e, stage + 2 * shard, src, shard, stream), "staging the root's shard", err, len))
            result = ncclUnhandledCudaError;
          own = stage + 2 * shard;
        }
        if (dst % PACK) out = stage;
      }
      if (result == ncclSuccess) result = launch_exchange(e, 0, out, own, shard, shard, 0, -1, stream, err, len);
      if (result == ncclSuccess && out != dst &&
          cuda_check(copy_async(e, dst, out, 2 * shard, stream), "returning the gathered shards", err, len))
        result = ncclUnhandledCudaError;
    }
  } else if (link_kind >= 0) {
    /* The link all-gather of the shards; the root's output is its receive buffer, another rank's lands in
     * the staging buffer, unread. */
    if (e->rank == root) {
      result = gather_link(e, link_kind, src, dst, shard, call.capturing, stream, err, len);
    } else {
      sccl_CUdeviceptr stage = 0, in = src;
      result = stage_buffer(e, shard + shard * (uint64_t)e->world, call.capturing, stream, &stage, err, len);
      if (result == ncclSuccess && src % PACK) {
        if (cuda_check(copy_async(e, stage, src, shard, stream), "staging the shard", err, len))
          result = ncclUnhandledCudaError;
        in = stage;
      }
      if (result == ncclSuccess)
        result = launch_link(e, link_kind, 0, in, stage + shard, shard, shard, link_piece(e, COLL_GATHER, shard),
                             link_blocks_for(e, COLL_GATHER, shard), stream, err, len);
    }
  } else {
    /* The all-gather's tiles; every rank receives every shard and only the root keeps them. */
    uint64_t tile = gather_tile(e);
    unsigned cap = shard <= tile ? e->shared.blocks : e->shared.large_blocks;
    int root_direct = e->rank == root && dst % PACK == 0 && shard % PACK == 0;
    for (uint64_t column = 0; result == ncclSuccess && column < shard; column += tile) {
      uint64_t piece = shard - column < tile ? shard - column : tile;
      uint64_t padded = (piece + PACK - 1) / PACK * PACK;
      sccl_CUdeviceptr in = src + column;
      if (in % PACK || piece % PACK) {
        if (cuda_check(copy_async(e, e->scratch[0], in, piece, stream), "staging the shard", err, len)) {
          result = ncclUnhandledCudaError;
          break;
        }
        in = e->scratch[0];
      }
      if (root_direct) {
        result = launch_gather(e, in, dst + column, piece, shard, cap, stream, err, len);
        continue;
      }
      result = launch_gather(e, in, e->scratch[1], padded, padded, cap, stream, err, len);
      for (int r = 0; result == ncclSuccess && e->rank == root && r < e->world; ++r)
        if (cuda_check(copy_async(e, dst + (uint64_t)r * shard + column, e->scratch[1] + (uint64_t)r * padded, piece,
                                  stream),
                       "placing a gathered shard", err, len))
          result = ncclUnhandledCudaError;
    }
  }
  if (result == ncclSuccess) {
    atomic_fetch_add_explicit(&e->gather_root_calls, 1, memory_order_relaxed);
    if (call.capturing) atomic_fetch_add_explicit(&e->captured_calls, 1, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

ncclResult_t sccl_engine_scatter(sccl_engine *e, const void *sendbuff, void *recvbuff, size_t count,
                                 ncclDataType_t datatype, int root, cudaStream_t stream_handle, char *err, size_t len) {
  size_t item = nccl_type_size(datatype);
  if (!item) {
    put_error(err, len, "ncclScatter: invalid datatype %d", (int)datatype);
    return ncclInvalidArgument;
  }
  if (root < 0 || root >= e->world) {
    put_error(err, len, "ncclScatter: root %d is outside 0-%d", root, e->world - 1);
    return ncclInvalidArgument;
  }
  if (count == 0) return ncclSuccess;
  /* Only the root's send buffer is read; other ranks may pass NULL. */
  ncclResult_t result = check_args(e, "ncclScatter", e->rank == root ? sendbuff : recvbuff, recvbuff, count, item,
                                   err, len);
  if (result != ncclSuccess) return result;
  sccl_CUstream stream = (sccl_CUstream)stream_handle;
  uint64_t chunk = (uint64_t)count * item;
  sccl_CUdeviceptr src = (sccl_CUdeviceptr)(uintptr_t)sendbuff, dst = (sccl_CUdeviceptr)(uintptr_t)recvbuff;
  call_state call;
  result = begin_call(e, "ncclScatter", stream, &call, err, len);
  if (result != ncclSuccess) return result;
  if (e->world == 1) {
    if (src != dst && cuda_check(copy_async(e, dst, src, chunk, stream), "one-rank scatter copy", err, len))
      result = ncclUnhandledCudaError;
  } else if (pair_exchange_ok(e, chunk)) {
    /* A pair: the root sends the peer's chunk in one one-way exchange, whose kernel also copies its own. */
    uint64_t peer = (uint64_t)(1 - root);
    result = e->rank == root ? exchange_root(e, src + peer * chunk, src + (uint64_t)root * chunk, dst, chunk,
                                             call.capturing, stream, err, len)
                             : exchange_staged(e, 0, dst, chunk, call.capturing, stream, err, len);
  } else {
    /* The all-to-all's tiles: the root sends chunk p to rank p; the other ranks send whatever scratch[0]
     * holds. Every rank keeps the root's row. */
    uint64_t tile = scatter_tile(e);
    int direct_in = e->rank == root && src % PACK == 0 && chunk % PACK == 0;
    for (uint64_t column = 0; result == ncclSuccess && column < chunk; column += tile) {
      uint64_t piece = chunk - column < tile ? chunk - column : tile;
      uint64_t padded = (piece + PACK - 1) / PACK * PACK;
      sccl_CUdeviceptr in = e->scratch[0];
      uint64_t stride = padded;
      if (direct_in) {
        in = src + column;
        stride = chunk;
      } else if (e->rank == root) {
        result = stage_chunks(e, src, chunk, column, piece, padded, stream, err, len);
      }
      if (result == ncclSuccess)
        result = launch_scatter(e, e->fn[SCCL_K_ALLTOALL][0], &e->alltoall_ops, in, e->scratch[1], padded, stride,
                                padded, stream, err, len);
      if (result == ncclSuccess && cuda_check(copy_async(e, dst + column, e->scratch[1] + (uint64_t)root * padded,
                                                         piece, stream),
                                              "placing the root's chunk", err, len))
        result = ncclUnhandledCudaError;
    }
  }
  if (result == ncclSuccess) {
    atomic_fetch_add_explicit(&e->scatter_root_calls, 1, memory_order_relaxed);
    if (call.capturing) atomic_fetch_add_explicit(&e->captured_calls, 1, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

size_t sccl_engine_type_size(ncclDataType_t datatype) { return nccl_type_size(datatype); }

ncclResult_t sccl_engine_mem_alloc(void **ptr, size_t size, char *err, size_t len) {
  cu = sccl_cuda_get();
  if (!cu) {
    put_error(err, len, "ncclMemAlloc: %s", sccl_cuda_error());
    return ncclSystemError;
  }
  sccl_CUcontext ctx = NULL;
  if (cu->CtxGetCurrent(&ctx) != SCCL_CUDA_SUCCESS || !ctx) {
    put_error(err, len, "ncclMemAlloc: no current CUDA context: select the device (cudaSetDevice) first");
    return ncclInvalidUsage;
  }
  sccl_CUdeviceptr memory = 0;
  if (cuda_check(cu->MemAlloc(&memory, size ? size : 1), "ncclMemAlloc", err, len)) return ncclUnhandledCudaError;
  *ptr = (void *)(uintptr_t)memory;
  return ncclSuccess;
}

ncclResult_t sccl_engine_mem_free(void *ptr, char *err, size_t len) {
  if (!ptr) return ncclSuccess;
  cu = sccl_cuda_get();
  if (!cu) {
    put_error(err, len, "ncclMemFree: %s", sccl_cuda_error());
    return ncclSystemError;
  }
  return cuda_check(cu->MemFree((sccl_CUdeviceptr)(uintptr_t)ptr), "ncclMemFree", err, len) ? ncclUnhandledCudaError
                                                                                             : ncclSuccess;
}

/* One pair exchange of `chunk` bytes each way: `sent` (0: nothing meaningful) to the peer, the peer's block
 * into `received` (0: discarded). Unaligned buffers go through the staging buffer. */
static ncclResult_t exchange_staged(sccl_engine *e, sccl_CUdeviceptr sent, sccl_CUdeviceptr received, uint64_t chunk,
                                    int capturing, sccl_CUstream stream, char *err, size_t len) {
  sccl_CUdeviceptr in = sent, out = received;
  ncclResult_t result = ncclSuccess;
  if (in % PACK || out % PACK) {
    sccl_CUdeviceptr stage = 0;
    result = stage_buffer(e, 2 * chunk, capturing, stream, &stage, err, len);
    if (result == ncclSuccess && in % PACK) {
      if (cuda_check(copy_async(e, stage, sent, chunk, stream), "staging a send", err, len))
        result = ncclUnhandledCudaError;
      in = stage;
    }
    if (out % PACK) out = stage + chunk;
  }
  if (result == ncclSuccess) result = launch_exchange(e, in, out, 0, chunk, 0, 0, -1, stream, err, len);
  if (result == ncclSuccess && out && out != received &&
      cuda_check(copy_async(e, received, out, chunk, stream), "placing a receive", err, len))
    result = ncclUnhandledCudaError;
  return result;
}

/* The sending root of a one-way pair exchange: `sent` to the peer, and its own block `own` copied to `dst`
 * (none when they are one buffer) by the kernel's local role; the peer's slots are discarded. Unaligned
 * buffers fall back to a separate copy and the staged exchange. */
static ncclResult_t exchange_root(sccl_engine *e, sccl_CUdeviceptr sent, sccl_CUdeviceptr own, sccl_CUdeviceptr dst,
                                  uint64_t chunk, int capturing, sccl_CUstream stream, char *err, size_t len) {
  if (sent % PACK == 0 && own % PACK == 0 && dst % PACK == 0)
    return launch_exchange(e, sent, own != dst ? dst : 0, own != dst ? own : 0, chunk, 0, EXCHANGE_DISCARD, -1,
                           stream, err, len);
  if (own != dst && cuda_check(copy_async(e, dst, own, chunk, stream), "the root's own block", err, len))
    return ncclUnhandledCudaError;
  return exchange_staged(e, sent, 0, chunk, capturing, stream, err, len);
}

/* One exchange between the two ranks: this rank's `sent` bytes (0 for none) travel to the peer while the
 * peer's bytes arrive into `received` (`received_bytes`, 0 for none). Both ranks derive the same op
 * geometry from the larger of the two sizes, which each of them knows (its own send and the matching
 * receive). Tiles of the all-to-all kernel carry the data: this rank's bytes travel from row `peer` of
 * scratch[0], the peer's arrive in row `peer` of scratch[1]. */
static ncclResult_t p2p_exchange(sccl_engine *e, sccl_CUdeviceptr sent, uint64_t sent_bytes,
                                 sccl_CUdeviceptr received, uint64_t received_bytes, int capturing,
                                 sccl_CUstream stream, char *err, size_t len) {
  uint64_t chunk = sent_bytes > received_bytes ? sent_bytes : received_bytes;
  /* One pair exchange when both directions carry the same bytes or one carries none (each rank knows both
   * sizes, so both decide alike): a one-way transfer sends the unused direction's slots unchanged and
   * discards them on arrival. Unaligned buffers go through the staging buffer. */
  if (pair_exchange_ok(e, chunk) && (sent_bytes == received_bytes || !sent_bytes || !received_bytes)) {
    ncclResult_t result = exchange_staged(e, sent_bytes ? sent : 0, received_bytes ? received : 0, chunk, capturing,
                                          stream, err, len);
    if (result == ncclSuccess) atomic_fetch_add_explicit(&e->p2p_exchanges, 1, memory_order_relaxed);
    return result;
  }
  uint64_t tile = scatter_tile(e);
  uint64_t peer = (uint64_t)(1 - e->rank);
  ncclResult_t result = ncclSuccess;
  for (uint64_t column = 0; result == ncclSuccess && column < chunk; column += tile) {
    uint64_t piece = chunk - column < tile ? chunk - column : tile;
    uint64_t padded = (piece + PACK - 1) / PACK * PACK;
    if (column < sent_bytes) {
      uint64_t n = sent_bytes - column < piece ? sent_bytes - column : piece;
      if (cuda_check(copy_async(e, e->scratch[0] + peer * padded, sent + column, n, stream), "staging a send", err,
                     len))
        return ncclUnhandledCudaError;
    }
    result = launch_scatter(e, e->fn[SCCL_K_ALLTOALL][0], &e->p2p_ops, e->scratch[0], e->scratch[1], padded, padded,
                            padded, stream, err, len);
    if (result == ncclSuccess && column < received_bytes) {
      uint64_t n = received_bytes - column < piece ? received_bytes - column : piece;
      if (cuda_check(copy_async(e, received + column, e->scratch[1] + peer * padded, n, stream), "placing a receive",
                     err, len))
        result = ncclUnhandledCudaError;
    }
  }
  if (result == ncclSuccess) atomic_fetch_add_explicit(&e->p2p_exchanges, 1, memory_order_relaxed);
  return result;
}

ncclResult_t sccl_engine_p2p(sccl_engine *e, const sccl_p2p_op *ops, unsigned count, char *err, size_t len) {
  if (!count) return ncclSuccess;
  sccl_CUstream stream = (sccl_CUstream)ops[0].stream;
  for (unsigned i = 0; i < count; ++i) {
    if ((sccl_CUstream)ops[i].stream != stream) {
      put_error(err, len, "%s: the point-to-point calls of one group on one communicator must use one stream",
                ops[i].send ? "ncclSend" : "ncclRecv");
      return ncclInvalidUsage;
    }
    if (ops[i].peer != e->rank && e->world != 2) {
      put_error(err, len, "%s: point-to-point between ranks is carried on communicators of two ranks; this one has %d",
                ops[i].send ? "ncclSend" : "ncclRecv", e->world);
      return ncclInvalidUsage;
    }
  }
  /* Calls to one peer match in issue order: the k-th send to the peer with the peer's k-th receive from
   * this rank. Exchange k pairs this rank's k-th send with its k-th receive. */
  enum { MAX_OPS = 1024 };
  if (count > MAX_OPS) {
    put_error(err, len, "ncclGroupEnd: more than %d point-to-point calls on one communicator in one group", MAX_OPS);
    return ncclInvalidUsage;
  }
  unsigned sends[2][MAX_OPS], recvs[2][MAX_OPS], nsend[2] = {0, 0}, nrecv[2] = {0, 0};
  for (unsigned i = 0; i < count; ++i) {
    int self = ops[i].peer == e->rank;
    if (ops[i].send)
      sends[self][nsend[self]++] = i;
    else
      recvs[self][nrecv[self]++] = i;
  }
  if (nsend[1] != nrecv[1]) {
    put_error(err, len, "ncclGroupEnd: %u sends to this rank itself and %u receives from it do not match", nsend[1],
              nrecv[1]);
    return ncclInvalidUsage;
  }
  for (unsigned k = 0; k < nsend[1]; ++k)
    if (ops[sends[1][k]].bytes != ops[recvs[1][k]].bytes) {
      put_error(err, len, "ncclGroupEnd: send %u to this rank itself carries %zu bytes, the receive %zu", k,
                ops[sends[1][k]].bytes, ops[recvs[1][k]].bytes);
      return ncclInvalidUsage;
    }
  char message[400];
  ncclResult_t health = sccl_engine_async_error(e, message, sizeof message);
  if (health != ncclSuccess) {
    put_error(err, len, "%s", message);
    return health;
  }
  call_state call;
  ncclResult_t result = begin_call(e, ops[0].send ? "ncclSend" : "ncclRecv", stream, &call, err, len);
  if (result != ncclSuccess) return result;
  /* The all-to-all of two ranks as point-to-point calls (torch's pattern, and nccl-tests' below NCCL API
   * level 22800): one send to and one receive from the peer and one send to and receive from this rank
   * itself, every block `chunk` bytes, the two receive blocks adjacent in rank order and neither send block
   * overlapping their span. One pair exchange carries it, its local role copying the own block in parallel
   * with the transfer (as ncclAlltoAll's exchange does) instead of a copy ahead of the exchange. */
  int fused = 0;
  if (nsend[1] == 1 && nsend[0] == 1 && nrecv[0] == 1) {
    const sccl_p2p_op *own_send = &ops[sends[1][0]], *own_recv = &ops[recvs[1][0]];
    const sccl_p2p_op *peer_send = &ops[sends[0][0]], *peer_recv = &ops[recvs[0][0]];
    uint64_t chunk = peer_send->bytes, rank = (uint64_t)e->rank;
    sccl_CUdeviceptr sent = (sccl_CUdeviceptr)(uintptr_t)peer_send->buff;
    sccl_CUdeviceptr own_in = (sccl_CUdeviceptr)(uintptr_t)own_send->buff;
    sccl_CUdeviceptr own_out = (sccl_CUdeviceptr)(uintptr_t)own_recv->buff;
    sccl_CUdeviceptr peer_out = (sccl_CUdeviceptr)(uintptr_t)peer_recv->buff;
    sccl_CUdeviceptr base = own_out - rank * chunk;
    if (chunk && peer_recv->bytes == chunk && own_send->bytes == chunk && own_out >= rank * chunk &&
        peer_out == base + (1 - rank) * chunk && pair_exchange_ok(e, chunk) && base % PACK == 0 &&
        sent % PACK == 0 && own_in % PACK == 0 && !(sent < base + 2 * chunk && base < sent + chunk) &&
        (own_in == own_out || !(own_in < base + 2 * chunk && base < own_in + chunk))) {
      result = launch_exchange(e, sent, base, own_in != own_out ? own_in : 0, chunk, chunk, 0, -1, stream, err, len);
      if (result == ncclSuccess) atomic_fetch_add_explicit(&e->p2p_exchanges, 1, memory_order_relaxed);
      fused = 1;
    }
  }
  for (unsigned k = 0; !fused && result == ncclSuccess && k < nsend[1]; ++k) {
    const sccl_p2p_op *from = &ops[sends[1][k]], *to = &ops[recvs[1][k]];
    if (from->bytes && from->buff != to->buff &&
        cuda_check(copy_async(e, (sccl_CUdeviceptr)(uintptr_t)to->buff, (sccl_CUdeviceptr)(uintptr_t)from->buff,
                              from->bytes, stream),
                   "a send to this rank itself", err, len))
      result = ncclUnhandledCudaError;
  }
  unsigned exchanges = fused ? 0 : nsend[0] > nrecv[0] ? nsend[0] : nrecv[0];
  for (unsigned k = 0; result == ncclSuccess && k < exchanges; ++k) {
    const sccl_p2p_op *send = k < nsend[0] ? &ops[sends[0][k]] : NULL;
    const sccl_p2p_op *recv = k < nrecv[0] ? &ops[recvs[0][k]] : NULL;
    result = p2p_exchange(e, send ? (sccl_CUdeviceptr)(uintptr_t)send->buff : 0, send ? send->bytes : 0,
                      recv ? (sccl_CUdeviceptr)(uintptr_t)recv->buff : 0, recv ? recv->bytes : 0, call.capturing,
                      stream, err, len);
  }
  if (result == ncclSuccess) {
    for (unsigned i = 0; i < count; ++i) {
      atomic_fetch_add_explicit(ops[i].send ? &e->p2p_sends : &e->p2p_recvs, 1, memory_order_relaxed);
      atomic_fetch_add_explicit(ops[i].send ? &e->p2p_bytes_sent : &e->p2p_bytes_received, ops[i].bytes,
                                memory_order_relaxed);
    }
    if (call.capturing) atomic_fetch_add_explicit(&e->captured_calls, count, memory_order_relaxed);
  }
  return end_call(e, &call, result, err, len);
}

ncclResult_t sccl_engine_teardown_sync(sccl_engine *e) {
  if (!e || !cu) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  int pushed = e->ctx && cu->CtxPushCurrent(e->ctx) == SCCL_CUDA_SUCCESS;
  /* Without the creating context the enqueued work cannot be waited for: a failure, not a skipped wait. */
  if (e->ctx && !pushed) result = ncclUnhandledCudaError;
  if (pushed && e->have_last && cu->StreamSynchronize(e->last_stream) != SCCL_CUDA_SUCCESS)
    result = ncclUnhandledCudaError;
  if (pushed) {
    sccl_CUcontext popped;
    cu->CtxPopCurrent(&popped);
  }
  return result;
}

int sccl_engine_wait_limit_ms(const sccl_engine *e) {
  uint64_t us = e->serving ? e->shared.serving_wait_us : e->shared.startup_wait_us;
  uint64_t ms = (us + 999) / 1000;
  return ms < 1 ? 1 : ms > INT32_MAX ? INT32_MAX : (int)ms;
}

ncclResult_t sccl_engine_finalize(sccl_engine *e) {
  if (!e || !cu) return ncclSuccess;
  pthread_mutex_lock(&e->lock);
  ncclResult_t result = ncclSuccess;
  if (e->have_last) {
    int pushed = cu->CtxPushCurrent(e->ctx) == SCCL_CUDA_SUCCESS;
    if (cu->StreamSynchronize(e->last_stream) != SCCL_CUDA_SUCCESS) result = ncclUnhandledCudaError;
    sccl_CUcontext popped;
    if (pushed) cu->CtxPopCurrent(&popped);
  }
  pthread_mutex_unlock(&e->lock);
  return result;
}

ncclResult_t sccl_engine_set_wait_regime(sccl_engine *e, const char *regime) {
  if (!regime || (strcmp(regime, "startup") && strcmp(regime, "serving"))) return ncclInvalidArgument;
  e->serving = !strcmp(regime, "serving");
  /* A plain host store: the next launch's kernels read it, eager or replayed. */
  if (e->ctrl) e->ctrl[CTRL_WAIT_LIMIT_US] = e->serving ? e->shared.serving_wait_us : e->shared.startup_wait_us;
  return ncclSuccess;
}

int sccl_engine_device(const sccl_engine *e) { return e ? e->device : -1; }
int sccl_engine_position(const sccl_engine *e) { return e ? e->position : -1; }

void sccl_engine_shutdown(sccl_engine *e) {
  if (!e || !e->proxy) return;
  e->transport->stop(e->proxy);
  e->transport->destroy(e->proxy);
  e->proxy = NULL;
}

/* `text` as a JSON string literal (quoted, escaped) into out, or null when text is NULL. */
static void json_string(char *out, size_t len, const char *text) {
  if (!len) return;
  if (!text) {
    snprintf(out, len, "null");
    return;
  }
  size_t n = 0;
  out[n++] = '"';
  for (const unsigned char *c = (const unsigned char *)text; *c && n + 8 < len; ++c) {
    if (*c == '"' || *c == '\\') {
      out[n++] = '\\';
      out[n++] = (char)*c;
    } else if (*c < 0x20) {
      n += (size_t)snprintf(out + n, len - n, "\\u%04x", *c);
    } else {
      out[n++] = (char)*c;
    }
  }
  out[n++] = '"';
  out[n] = 0;
}

static uint64_t realtime_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

size_t sccl_engine_receipt(sccl_engine *e, char *out, size_t len) {
  static const char *const dt[SCCL_DT_COUNT] = {"float32", "float16", "bfloat16"};
  char ops[512] = {0};
  size_t used = 0;
  for (int a = 0; a < SCCL_ALG_COUNT; ++a)
    for (int d = 0; d < SCCL_DT_COUNT; ++d) {
      uint64_t n = atomic_load_explicit(&e->ops[a][d], memory_order_relaxed);
      if (!n) continue;
      used += (size_t)snprintf(ops + used, sizeof ops - used, "%s\"%s/%s\":%" PRIu64, used ? "," : "",
                               a == SCCL_ALG_ONESHOT ? "oneshot" : "twoshot", dt[d], n);
      if (used >= sizeof ops) used = sizeof ops - 1;
    }
  char folds[2048] = {0};
  size_t fused = 0;
  for (int d = 0; d < SCCL_FOLD_DTYPES; ++d)
    for (int o = 0; o < SCCL_FOLD_OPS; ++o) {
      uint64_t n = atomic_load_explicit(&e->folds[d][o], memory_order_relaxed);
      if (!n) continue;
      fused += (size_t)snprintf(folds + fused, sizeof folds - fused, "%s\"%s/%s\":%" PRIu64, fused ? "," : "",
                                type_names[d], op_names[o], n);
      if (fused >= sizeof folds) fused = sizeof folds - 1;
    }
  char links[256] = {0};
  size_t lused = 0;
  for (int k = 0; k < SCCL_LINK_KINDS; ++k) {
    uint64_t n = atomic_load_explicit(&e->link_ops[k], memory_order_relaxed);
    if (!n) continue;
    lused += (size_t)snprintf(links + lused, sizeof links - lused, "%s\"%s\":%" PRIu64, lused ? "," : "",
                              link_kind_names[k], n);
    if (lused >= sizeof links) lused = sizeof links - 1;
  }
  char by_blocks[256] = {0};
  size_t bused = 0;
  for (int b = 1; b <= 64; ++b) {
    uint64_t n = atomic_load_explicit(&e->ops_by_blocks[b], memory_order_relaxed);
    if (!n) continue;
    bused += (size_t)snprintf(by_blocks + bused, sizeof by_blocks - bused, "%s\"%d\":%" PRIu64, bused ? "," : "", b, n);
    if (bused >= sizeof by_blocks) bused = sizeof by_blocks - 1;
  }
  char health[900] = {0}, health_json[1000];
  ncclResult_t status = sccl_engine_async_error(e, health, sizeof health);
  json_string(health_json, sizeof health_json, status == ncclSuccess ? NULL : health);
  return (size_t)snprintf(
      out, len,
      "{\"schema\":\"libsircl-receipt/v1\",\"library\":\"libsircl " LIBSIRCL_VERSION "\",\"pid\":%d,"
      "\"communicator\":%d,\"rank\":%d,\"world\":%d,\"position\":%d,\"lanes\":%d,"
      "\"device\":%d,"
      "\"transport\":\"%s\",\"kernel_pack\":\"%s\",\"fold_pack\":\"%s\",\"links_pack\":\"%s\","
      "\"slot_bytes\":%" PRIu64
      ",\"capacity\":%" PRIu64
      ",\"large_piece_bytes\":%" PRIu64 ",\"oneshot_max_bytes\":%" PRIu64 ",\"wait_regime\":\"%s\",\"fail_stop\":%s,"
      "\"progress_cpus\":\"%s\",\"pair_exchange\":{\"ops\":%" PRIu64 ",\"bytes\":%" PRIu64 "},\"pair_plan\":%s,\"link_blocks\":{\"session\":%u,\"reduce\":%u,"
      "\"gather\":%u,\"scatter\":%u,\"ops_by_blocks\":{%s}},"
      "\"all_reduce\":{\"calls\":%" PRIu64 ",\"captured_calls\":%" PRIu64 ",\"bytes\":%" PRIu64
      ",\"ops\":{%s},\"padded_tails\":%" PRIu64 ",\"unaligned_staged\":%" PRIu64 ",\"local_copies\":%" PRIu64
      "},\"all_gather\":{\"calls\":%" PRIu64 ",\"ops\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"padded\":%" PRIu64
      "},\"reduce_scatter\":{\"calls\":%" PRIu64 ",\"ops\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"padded\":%" PRIu64
      "},\"broadcast\":{\"calls\":%" PRIu64 "},\"reduce\":{\"calls\":%" PRIu64
      "},\"all_to_all\":{\"calls\":%" PRIu64 ",\"ops\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"padded\":%" PRIu64
      "},\"gather\":{\"calls\":%" PRIu64 "},\"scatter\":{\"calls\":%" PRIu64 "},\"fold\":{\"ops\":{%s}"
      "},\"chain\":{\"on\":%s,\"index\":%d,\"ops\":%" PRIu64 ",\"bytes\":%" PRIu64
      "},\"links\":{\"on\":%s,\"ring\":%s,\"ring_reduce_passes\":%d,\"ops\":{%s},\"bytes\":%" PRIu64
      ",\"staged\":%" PRIu64 ",\"stage_buffers\":%d,\"graph_staging\":%s,\"graph_staged\":%" PRIu64
      ",\"native_ops\":%" PRIu64 ",\"native_items\":%" PRIu64
      "},\"forward_windows\":{\"lanes\":%u,\"chunk_bytes\":%u,\"chunks_posted\":%" PRIu64
      ",\"ring_window_bytes\":%u,\"ring_window_chunks\":%" PRIu64
      "},\"point_to_point\":{\"sends\":%" PRIu64 ",\"receives\":%" PRIu64 ",\"exchanges\":%" PRIu64
      ",\"ops\":%" PRIu64 ",\"bytes_sent\":%" PRIu64 ",\"bytes_received\":%" PRIu64 "},\"refused\":{\"op\":%" PRIu64 ",\"capture_stream\":%" PRIu64
      "},\"forwarded\":0,\"native\":{\"ops_posted\":%" PRIu64 ",\"writes_completed\":%" PRIu64
      ",\"phases_posted\":%" PRIu64 "},\"healthy\":%s,\"error\":%s}",
      (int)getpid(), e->receipt_id, e->rank, e->world, e->position, e->lanes, e->device,
      e->world == 1 ? "none" : e->transport->name,
      sccl_kp_hash(),
      sccl_kp_fold_hash(), sccl_kp_links_hash(),
      e->shared.slot_bytes, e->shared.capacity, e->shared.large_piece, e->shared.oneshot_max,
      e->serving ? "serving" : "startup", e->fail_stop ? "true" : "false",
      e->progress_cpus[0] ? e->progress_cpus : "none",
      atomic_load(&e->exchange_ops), atomic_load(&e->exchange_bytes),
      e->shared.pair_plan ? "true" : "false", e->shared.link_blocks, e->shared.coll_blocks[COLL_REDUCE],
      e->shared.coll_blocks[COLL_GATHER], e->shared.coll_blocks[COLL_SCATTER], by_blocks,
      atomic_load(&e->calls), atomic_load(&e->captured_calls),
      atomic_load(&e->bytes), ops, atomic_load(&e->padded_tails), atomic_load(&e->staged_unaligned),
      atomic_load(&e->local_copies), atomic_load(&e->gather_calls), atomic_load(&e->gather_ops),
      atomic_load(&e->gather_bytes), atomic_load(&e->gather_padded), atomic_load(&e->scatter_calls),
      atomic_load(&e->scatter_ops), atomic_load(&e->scatter_bytes), atomic_load(&e->scatter_padded),
      atomic_load(&e->broadcast_calls), atomic_load(&e->reduce_calls), atomic_load(&e->alltoall_calls),
      atomic_load(&e->alltoall_ops), atomic_load(&e->alltoall_bytes), atomic_load(&e->alltoall_padded),
      atomic_load(&e->gather_root_calls), atomic_load(&e->scatter_root_calls), folds,
      e->chain_on ? "true" : "false", e->chain_index,
      atomic_load(&e->chain_ops[0]) + atomic_load(&e->chain_ops[1]) + atomic_load(&e->chain_ops[2]),
      atomic_load(&e->chain_bytes), e->link_on ? "true" : "false", e->shared.ring_on && e->link_on ? "true" : "false",
      e->ring_reduce_passes, links, atomic_load(&e->link_bytes), atomic_load(&e->link_staged),
      e->retired + (e->stage != 0), e->shared.graph_staging ? "true" : "false", atomic_load(&e->graph_staged),
      e->proxy ? e->transport->stat(e->proxy, 16) : 0, e->proxy ? e->transport->stat(e->proxy, 17) : 0,
      e->forward_lanes, e->forward_chunk, e->proxy ? e->transport->stat(e->proxy, 10) : 0, e->ring_window,
      e->proxy ? e->transport->stat(e->proxy, 24) : 0, atomic_load(&e->p2p_sends),
      atomic_load(&e->p2p_recvs), atomic_load(&e->p2p_exchanges), atomic_load(&e->p2p_ops),
      atomic_load(&e->p2p_bytes_sent), atomic_load(&e->p2p_bytes_received),
      atomic_load(&e->refused_op),
      atomic_load(&e->refused_capture), e->proxy ? e->transport->stat(e->proxy, 0) : 0,
      e->proxy ? e->transport->stat(e->proxy, 1) : 0, e->proxy ? e->transport->stat(e->proxy, 5) : 0,
      status == ncclSuccess ? "true" : "false", health_json);
}

/* <prefix>.rank<r>.<pid>.c<n>.json, n the communicator's number among the process's communicators, so
 * communicators of one process with the same rank keep files of their own. The file is replaced whole
 * (written beside it, then renamed), so a reader never sees a partial receipt. */
static void write_receipt(sccl_engine *e) {
  const char *prefix = sccl_env("LIBSIRCL_RECEIPT");
  if (!prefix || !*prefix) return;
  char path[600], partial[620];
  snprintf(path, sizeof path, "%s.rank%d.%d.c%d.json", prefix, e->rank, (int)getpid(), e->receipt_id);
  snprintf(partial, sizeof partial, "%s.partial", path);
  size_t need = sccl_engine_receipt(e, NULL, 0) + 1;
  char *text = malloc(need);
  if (!text) return;
  sccl_engine_receipt(e, text, need);
  FILE *f = fopen(partial, "w");
  if (f) {
    int ok = fprintf(f, "%s\n", text) > 0;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(partial, path) != 0) unlink(partial);
  }
  free(text);
}

static uint64_t monotonic_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* After a call: refresh the receipt file when its interval has passed, so a process that is killed rather
 * than finalized (a serving process, usually) still leaves a recent receipt. */
static void refresh_receipt(sccl_engine *e) {
  if (!e->receipt_interval_ns) return;
  uint64_t now = monotonic_ns();
  if (now < e->receipt_next_ns) return;
  e->receipt_next_ns = now + e->receipt_interval_ns;
  write_receipt(e);
}

/* At creation: the communicator's number, the refresh interval, and the first receipt file. */
static void start_receipts(sccl_engine *e) {
  static atomic_int communicators;
  e->receipt_id = atomic_fetch_add(&communicators, 1) + 1;
  const char *prefix = sccl_env("LIBSIRCL_RECEIPT");
  if (!prefix || !*prefix) return;
  const char *text = sccl_env("LIBSIRCL_RECEIPT_INTERVAL_S");
  double seconds = text && *text ? strtod(text, NULL) : 10.0;
  e->receipt_interval_ns = seconds > 0 ? (uint64_t)(seconds * 1e9) : 0;
  e->receipt_next_ns = monotonic_ns() + e->receipt_interval_ns;
  write_receipt(e);
}

/* LIBSIRCL_LINK_DUMP=<prefix>, a diagnostic: <prefix>.rank<r>.<pid>.c<n>.links.json holds, for every link of
 * the link area, the credit word, the sent word, the ready and consumed words of every slot and word 0 of
 * every inbound flag line (slot by lane), the native counters, and, when `device`, the 16 device link
 * counter words. Written when a wait of the session first times out (no device read: a kernel may still
 * run) and at destroy (`why`). */
/* On the emulation transport, the stand-in's report of this process (sccl_emu_report) as ,"emulation":{...}. */
static void dump_emulation(sccl_engine *e, FILE *f) {
  if (e->transport != &emulation_ops) return;
  size_t need = sccl_emu_report(NULL, 0);
  char *text = malloc(need);
  if (!text) return;
  sccl_emu_report(text, need);
  fprintf(f, ",\"emulation\":%s", text);
  free(text);
}

static void dump_links(sccl_engine *e, const char *why, int device) {
  const char *prefix = sccl_env("LIBSIRCL_LINK_DUMP");
  if (!prefix || !*prefix || !e->link_on || !e->host) return;
  char path[640];
  snprintf(path, sizeof path, "%s.rank%d.%d.c%d.links.json", prefix, e->rank, (int)getpid(), e->receipt_id);
  FILE *f = fopen(path, "a");
  if (!f) return;
  enum { LINKS = 4, STRIDE = 128 };
  const uint8_t *area = e->host + e->link_off;
  unsigned slots = e->shared.link_slots;
#define WORD(off) (__atomic_load_n((const uint32_t *)(area + (off)), __ATOMIC_ACQUIRE))
  int failed = e->proxy && e->transport->failed(e->proxy);
  char failure[1000];
  json_string(failure, sizeof failure, failed ? e->transport->error(e->proxy) : NULL);
  fprintf(f, "{\"why\":\"%s\",\"ns\":%" PRIu64 ",\"rank\":%d,\"position\":%d,\"chain_index\":%d,"
             "\"ring_prev\":%d,\"ring_next\":%d,\"slots\":%u,\"lanes\":%d,\"transport_failed\":%d,"
             "\"transport_error\":%s,\"links\":[",
          why, realtime_ns(), e->rank, e->position, e->chain_index, e->ring_prev, e->ring_next, slots, e->lanes,
          failed, failure);
  for (int l = 0; l < LINKS; ++l) {
    fprintf(f, "%s{\"link\":%d,\"credit\":%u,\"sent\":%u,\"ready\":[", l ? "," : "", l,
            WORD(e->link_layout[6] + (uint64_t)l * STRIDE), WORD(e->link_layout[5] + (uint64_t)l * STRIDE));
    for (unsigned m = 0; m < slots; ++m) fprintf(f, "%s%u", m ? "," : "", WORD(e->link_layout[3] + (uint64_t)l * STRIDE + 4u * m));
    fprintf(f, "],\"consumed\":[");
    for (unsigned m = 0; m < slots; ++m) fprintf(f, "%s%u", m ? "," : "", WORD(e->link_layout[4] + (uint64_t)l * STRIDE + 4u * m));
    fprintf(f, "],\"flags\":[");
    for (unsigned m = 0; m < slots; ++m) {
      fprintf(f, "%s[", m ? "," : "");
      for (int lane = 0; lane < e->lanes; ++lane)
        fprintf(f, "%s%u", lane ? "," : "",
                WORD(e->link_layout[2] + (((uint64_t)l * slots + m) * (uint64_t)e->lanes + (uint64_t)lane) * STRIDE));
      fprintf(f, "]");
    }
    fprintf(f, "]}");
  }
#undef WORD
  fprintf(f, "],\"native\":[");
  for (int k = 0; k < 32; ++k) fprintf(f, "%s%" PRIu64, k ? "," : "", e->proxy ? e->transport->stat(e->proxy, k) : 0);
  fprintf(f, "],\"device_counters\":[");
  uint32_t words[16] = {0};
  if (device && e->link_counters && cu->MemcpyDtoH(words, e->link_counters, sizeof words) == SCCL_CUDA_SUCCESS)
    for (int k = 0; k < 16; ++k) fprintf(f, "%s%u", k ? "," : "", words[k]);
  /* The native trace's records since the last dump: [nanoseconds, stream (4 + link), event, value]; events
   * 1 op taken, 2 ready, 3 posted, 4 done, 5 consumed, 6 credit out, 7 credit in. */
  fprintf(f, "],\"trace\":[");
  enum { TAKE = 4096 };
  static _Thread_local uint64_t records[2 * TAKE];
  uint64_t lost = 0, total_lost = 0;
  int first = 1;
  for (int64_t n; e->proxy && (n = e->transport->trace_take(e->proxy, records, TAKE, &lost)) > 0;) {
    total_lost += lost;
    for (int64_t i = 0; i < n; ++i) {
      uint64_t w = records[2 * i + 1];
      fprintf(f, "%s[%" PRIu64 ",%u,%u,%u]", first ? "" : ",", records[2 * i], (unsigned)(w >> 48),
              (unsigned)((w >> 32) & 0xFFFFu), (unsigned)(w & 0xFFFFFFFFu));
      first = 0;
    }
    if (n < TAKE) break;
  }
  fprintf(f, "],\"trace_lost\":%" PRIu64, total_lost);
  dump_emulation(e, f);
  fprintf(f, "}\n");
  fclose(f);
}

/* The teardown of ncclCommDestroy in the link dump: when the progress thread's stop was requested, when it
 * had stopped, and when the transport (queue pairs, registrations) was destroyed, on the trace's clock. */
static void dump_teardown(sccl_engine *e, uint64_t stop_ns, uint64_t stopped_ns, uint64_t destroyed_ns) {
  const char *prefix = sccl_env("LIBSIRCL_LINK_DUMP");
  if (!prefix || !*prefix || !e->link_on || !e->host) return;
  char path[640];
  snprintf(path, sizeof path, "%s.rank%d.%d.c%d.links.json", prefix, e->rank, (int)getpid(), e->receipt_id);
  FILE *f = fopen(path, "a");
  if (!f) return;
  fprintf(f, "{\"why\":\"proxy destroyed\",\"ns\":%" PRIu64 ",\"rank\":%d,\"position\":%d,\"stop_ns\":%" PRIu64
             ",\"stopped_ns\":%" PRIu64 ",\"destroyed_ns\":%" PRIu64,
          destroyed_ns, e->rank, e->position, stop_ns, stopped_ns, destroyed_ns);
  dump_emulation(e, f);
  fprintf(f, "}\n");
  fclose(f);
}

void sccl_engine_teardown_stop(sccl_engine *e) {
  if (!e || e->stopped) return;
  write_receipt(e);
  int pushed = cu && e->ctx && cu->CtxPushCurrent(e->ctx) == SCCL_CUDA_SUCCESS;
  dump_links(e, "destroy", pushed);
  if (pushed) {
    sccl_CUcontext popped;
    cu->CtxPopCurrent(&popped);
  }
  e->stop_ns = realtime_ns();
  if (e->proxy) e->transport->stop(e->proxy);
  e->stopped_ns = realtime_ns();
  e->stopped = 1;
}

ncclResult_t sccl_engine_destroy(sccl_engine *e, int abort, char *err, size_t len) {
  if (!e) return ncclSuccess;
  watch_remove(e);
  ncclResult_t result = ncclSuccess;
  int stopped = e->stopped;
  if (!stopped) write_receipt(e);
  int pushed = cu && e->ctx && cu->CtxPushCurrent(e->ctx) == SCCL_CUDA_SUCCESS;
  int idle = 1;
  if (pushed && e->have_last) {
    if (abort) {
      idle = cu->StreamQuery(e->last_stream) == SCCL_CUDA_SUCCESS;
    } else if (!stopped) {
      cu->StreamSynchronize(e->last_stream);
    }
  }
  if (!stopped) {
    dump_links(e, "destroy", pushed && idle);
    e->stop_ns = realtime_ns();
    if (e->proxy) e->transport->stop(e->proxy);
    e->stopped_ns = realtime_ns();
  }
  /* A queue pair, completion queue or registration whose release failed may still be live: the arena
   * and device memory stay allocated (quarantined) until the process exits. */
  int unreleased = e->proxy ? e->transport->destroy(e->proxy) : 0;
  e->proxy = NULL;
  int released = unreleased == 0;
  if (!released) {
    result = ncclSystemError;
    put_error(err, len, "rank %d: releasing the native transport's queue pairs or registrations failed (%d verbs "
              "calls); the arena stays allocated", e->rank, unreleased);
  }
  dump_teardown(e, e->stop_ns, e->stopped_ns, realtime_ns());
  /* After an abort with a launch still running, its kernel may read the arena and counters
   * until its wait limit; they stay allocated until the process exits. */
  if (pushed && idle && released) {
    if (e->order_event) cu->EventDestroy(e->order_event);
    release_memory(e);
  }
  if (pushed) {
    sccl_CUcontext popped;
    cu->CtxPopCurrent(&popped);
  }
  pthread_mutex_destroy(&e->lock);
  if (idle && released) free(e);
  return result;
}
