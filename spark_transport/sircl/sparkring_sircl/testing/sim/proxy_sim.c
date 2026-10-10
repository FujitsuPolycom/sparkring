/*
 * CPU simulator of SIRCL ring sessions.
 *
 * Runs the real native layer (sparkring_sircl/oneshot/_roce_proxy.c, built
 * with SIRCL_PROXY_TEST_HOOKS) against the in-memory verbs stand-in
 * (testing/fake_verbs). Every rank of every simulated session lives in this
 * process: its real progress thread, one kernel thread that plays the device
 * side of the command ring (stage, op word, doorbells, descriptors, flag
 * waits) and checks every received byte, and one scheduler thread that runs
 * posted writes in a seeded order keeping each queue pair's order.
 *
 * Fabrics are rings (cycles) or paths of W Sparks, cabled port 0 to the next
 * Spark's port 1, two links per cable. Route maps follow the lane rules of
 * the route derivation (one shortest path: primary and secondary function of its
 * first cable; two shortest paths: primary on the path leaving the smaller
 * position through port 0, secondary on the other), and relayed lanes are
 * tagged by destination as a site relay plan does.
 *
 * Usage: proxy_sim [case-prefix ...]; prints PASS/FAIL per case and exits 1
 * when any case fails. Build: see testing/native_build.py.
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "fake_verbs.h"

/* -- native layer ----------------------------------------------------------------- */

typedef struct roce_ctx roce_ctx_t;
int roce_abi_version(void);
int roce_layout(int world, uint64_t slot_bytes, uint64_t *out);
uint64_t roce_blob_bytes(void);
roce_ctx_t *roce_create(int world, int rank, const char *const *device_names, int n_devices,
                        const int *lane_devices, int lane_count, const int *gid_indices,
                        int traffic_class, void *region, uint64_t region_bytes, uint64_t slot_bytes,
                        char *err, uint64_t err_len);
int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len);
int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t len);
int roce_lane_check(roce_ctx_t *c, int timeout_ms);
int roce_start(roce_ctx_t *c);
void roce_stop(roce_ctx_t *c);
int roce_failed(roce_ctx_t *c);
const char *roce_error(roce_ctx_t *c);
uint64_t roce_stat(roce_ctx_t *c, int which);
int roce_destroy(roce_ctx_t *c);
void roce_test_set_hook(void (*fn)(void *, int, uint32_t, int), void *arg);
void roce_test_disable_multi_phase(roce_ctx_t *c);
int roce_set_forward(roce_ctx_t *c, const uint32_t *lane_window_bytes, uint32_t chunk_bytes);
uint32_t roce_test_qp_num(roce_ctx_t *c, int d, int p);
int roce_chain_layout(int lanes, int slots, uint64_t slot_bytes, uint64_t *out);
int roce_set_chain(roce_ctx_t *c, int prev, int next, int slots, uint64_t slot_bytes, uint64_t chain_off);
int roce_set_trace(roce_ctx_t *c, uint32_t capacity);
int64_t roce_trace_take(roce_ctx_t *c, uint64_t *out, uint64_t max_records, uint64_t *lost);

/* -- constants of the wire protocol (arena, command ring, flag lines, op words) ------- */

#define SLOTS 2
#define FLAG_STRIDE 128
#define OP_SHIFT 30
enum { OP_ONESHOT = 0, OP_TWOSHOT = 1, OP_DESCRIBED = 2, OP_SCATTER = 3 };
enum { CTRL_DOORBELL = 0, CTRL_NBYTES = 1, CTRL_OP_WORD = 4, CTRL_WAIT_LIMIT_US = 7, CTRL_PHASE = 9,
       CTRL_DESC = 17 };
enum { KIND_CYCLE = 0, KIND_PATH = 1 };
enum { ROLE_CWP = 0, ROLE_CWS = 1, ROLE_CCWP = 2, ROLE_CCWS = 3 };
static const char *ROLE_DEVICE[4] = {"rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"};
#define MAX_WORLD 16
/* Flag-wait limit of the emulated kernels when command ring word 7 holds 0. */
#define WAIT_LIMIT_NS 10000000000ull

/* -- shared helpers ----------------------------------------------------------------- */

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

/* Bytes [first, first + n) of the pattern `key`; n and first are multiples of 8. */
static void fill(uint8_t *dst, uint64_t first, uint64_t n, uint64_t key) {
    for (uint64_t i = 0; i < n; i += 8) {
        uint64_t v = mix(key ^ ((first + i) * 0x9E3779B97F4A7C15ull));
        memcpy(dst + i, &v, 8);
    }
}

static int matches(const uint8_t *src, uint64_t first, uint64_t n, uint64_t key) {
    for (uint64_t i = 0; i < n; i += 8) {
        uint64_t v = mix(key ^ ((first + i) * 0x9E3779B97F4A7C15ull));
        if (memcmp(src + i, &v, 8) != 0) return 0;
    }
    return 1;
}

static uint64_t pattern_key(int tag, int rank, uint32_t seq, int phase) {
    return ((uint64_t)tag << 56) ^ ((uint64_t)rank << 48) ^ ((uint64_t)phase << 40) ^ seq;
}

static void chunk_range(uint32_t packs, int world, int chunk, uint32_t *first, uint32_t *count) {
    uint32_t lo = (uint32_t)((uint64_t)chunk * packs / (uint32_t)world);
    uint32_t hi = (uint32_t)((uint64_t)(chunk + 1) * packs / (uint32_t)world);
    *first = lo;
    *count = hi - lo;
}

/* -- Swing schedule ---------------------------------------------------------------------- */

static int swing_peer(int world, int rank, int step) {
    int64_t rho = (1 - (int64_t)(step % 2 == 0 ? -1 : 1) * ((int64_t)1 << (step + 1))) / 3;
    /* rho(k) = (1 - (-2)^(k+1)) / 3: 1, -1, 3, -5, ... */
    int64_t peer = rank % 2 == 0 ? rank + rho : rank - rho;
    peer %= world;
    if (peer < 0) peer += world;
    return (int)peer;
}

/* Ranks reached from `rank` through steps `step` .. log2(world)-1, as a bit set. */
static uint32_t swing_reach(int world, int steps, int rank, int step) {
    if (step >= steps) return 1u << rank;
    return swing_reach(world, steps, rank, step + 1) |
           swing_reach(world, steps, swing_peer(world, rank, step), step + 1);
}

static int lowest(uint32_t set) { return __builtin_ctz(set); }

/* Chunk order: depth-first over the reach sets, children ordered by their lowest rank. */
static void swing_order(int world, int steps, int rank, int step, int *order, int *n) {
    if (step >= steps) {
        order[(*n)++] = rank;
        return;
    }
    int other = swing_peer(world, rank, step);
    uint32_t mine = swing_reach(world, steps, rank, step + 1);
    uint32_t theirs = swing_reach(world, steps, other, step + 1);
    if (lowest(mine) < lowest(theirs)) {
        swing_order(world, steps, lowest(mine), step + 1, order, n);
        swing_order(world, steps, lowest(theirs), step + 1, order, n);
    } else {
        swing_order(world, steps, lowest(theirs), step + 1, order, n);
        swing_order(world, steps, lowest(mine), step + 1, order, n);
    }
}

typedef struct {
    int peer, first, end, ns;
} phase_t;

/* Phases of `rank`: reduce-scatter steps, then the all-gather steps in reverse. */
static int swing_phases(int world, int rank, phase_t *out) {
    int steps = 0;
    while ((1 << steps) < world) steps++;
    int order[MAX_WORLD], n = 0, position[MAX_WORLD];
    swing_order(world, steps, 0, 0, order, &n);
    for (int i = 0; i < world; i++) position[order[i]] = i;
    int count = 0;
    for (int k = 0; k < steps; k++) {
        int q = swing_peer(world, rank, k);
        uint32_t set = swing_reach(world, steps, q, k + 1);
        int lo = world, hi = -1;
        for (int r = 0; r < world; r++) {
            if (set & (1u << r)) {
                if (position[r] < lo) lo = position[r];
                if (position[r] > hi) hi = position[r];
            }
        }
        out[count++] = (phase_t){q, lo, hi + 1, 0};
    }
    for (int k = steps - 1; k >= 0; k--) {
        int q = swing_peer(world, rank, k);
        uint32_t set = swing_reach(world, steps, rank, k + 1);
        int lo = world, hi = -1;
        for (int r = 0; r < world; r++) {
            if (set & (1u << r)) {
                if (position[r] < lo) lo = position[r];
                if (position[r] > hi) hi = position[r];
            }
        }
        out[count++] = (phase_t){q, lo, hi + 1, 1};
    }
    return count;
}

/* -- fabric and sessions ---------------------------------------------------------- */

typedef struct {
    int rank;
    int node;
    uint8_t *arena;
    uint8_t *arena_alloc;
    uint64_t arena_bytes;
    uint64_t layout[7];
    roce_ctx_t *ctx;
    int devices[4];          /* fake device index per role */
    char names[4][64];
    int n_open;
    int open_roles[4];
    int lane_devices[MAX_WORLD * 2];
    int lane_roles[MAX_WORLD][2];
    uint64_t chain_off;          /* chain area offset in the arena, 0: none */
    uint64_t chain_layout[9];
} rank_t;

typedef struct {
    int world, lanes, kind, node_base;
    uint64_t slot_bytes;
    rank_t ranks[MAX_WORLD];
    char error[512];
} session_t;

static void gid_of(int node, int role, uint8_t gid[16]) {
    memset(gid, 0, 16);
    gid[10] = 0xff;
    gid[11] = 0xff;
    gid[12] = 10;
    gid[13] = (uint8_t)node;
    gid[14] = (uint8_t)role;
    gid[15] = 1;
}

/* Lane roles from `r` to `p`: one shortest path uses the primary and secondary
 * function of its first cable; two use path A (leaving the smaller position
 * through port 0) for lane 0 and path B for lane 1. */
static void t4_roles(int kind, int world, int r, int p, int roles[2]) {
    if (kind == KIND_PATH) {
        roles[0] = p > r ? ROLE_CWP : ROLE_CCWP;
        roles[1] = p > r ? ROLE_CWS : ROLE_CCWS;
        return;
    }
    int d = (p - r + world) % world;
    if (2 * d < world) {
        roles[0] = ROLE_CWP;
        roles[1] = ROLE_CWS;
    } else if (2 * d > world) {
        roles[0] = ROLE_CCWP;
        roles[1] = ROLE_CCWS;
    } else if (r < p) {
        roles[0] = ROLE_CWP;
        roles[1] = ROLE_CCWS;
    } else {
        roles[0] = ROLE_CCWP;
        roles[1] = ROLE_CWS;
    }
}

static int hops(int kind, int world, int r, int p) {
    if (kind == KIND_PATH) return r > p ? r - p : p - r;
    int d = (p - r + world) % world;
    return d < world - d ? d : world - d;
}

/* Fake devices, cables and relay tags of one session's fabric. */
static void build_fabric(session_t *s) {
    int W = s->world;
    for (int r = 0; r < W; r++) {
        rank_t *k = &s->ranks[r];
        k->rank = r;
        k->node = s->node_base + r;
        for (int role = 0; role < 4; role++) {
            uint8_t gid[16];
            gid_of(k->node, role, gid);
            snprintf(k->names[role], sizeof(k->names[role]), "n%d.%s", k->node, ROLE_DEVICE[role]);
            k->devices[role] = fv_add_device(k->names[role], k->node, role / 2, role % 2, gid);
        }
        fv_set_relay(k->node, 1);
    }
    int cables = s->kind == KIND_PATH ? W - 1 : W;
    for (int i = 0; i < cables; i++) {
        fv_add_cable(s->node_base + i, 0, s->node_base + (i + 1) % W, 1, 3u);
    }
}

static void plan_routes(session_t *s, int swap_rank, int swap_peer) {
    int W = s->world;
    for (int r = 0; r < W; r++) {
        rank_t *k = &s->ranks[r];
        k->n_open = 0;
        for (int p = 0; p < W; p++) {
            if (p == r) continue;
            int roles[2];
            t4_roles(s->kind, W, r, p, roles);
            if (r == swap_rank && p == swap_peer) {
                int t = roles[0];
                roles[0] = roles[1];
                roles[1] = t;
            }
            for (int l = 0; l < s->lanes; l++) {
                k->lane_roles[p][l] = roles[l];
                int index = -1;
                for (int i = 0; i < k->n_open; i++) {
                    if (k->open_roles[i] == roles[l]) index = i;
                }
                if (index < 0) {
                    index = k->n_open;
                    k->open_roles[k->n_open++] = roles[l];
                }
                k->lane_devices[p * s->lanes + l] = index;
            }
        }
        for (int l = 0; l < s->lanes; l++) k->lane_devices[r * s->lanes + l] = -1;
    }
    /* Relay tags: every lane of h >= 2 hops is marked with h - 1 relays toward
     * the GID of the peer's device of the same lane. */
    for (int r = 0; r < W; r++) {
        for (int p = 0; p < W; p++) {
            if (p == r) continue;
            int h = hops(s->kind, W, r, p);
            if (h < 2) continue;
            for (int l = 0; l < s->lanes; l++) {
                uint8_t gid[16];
                gid_of(s->ranks[p].node, s->ranks[p].lane_roles[r][l], gid);
                fv_set_dest_tag(s->ranks[r].devices[s->ranks[r].lane_roles[p][l]], gid, (uint32_t)(h - 1));
            }
        }
    }
}

static const char *rank0_post_order;
/* Forward window and chunk for lanes through relays (0: no windows). */
static uint32_t sim_window, sim_chunk;
/* Chain schedule of new sessions: slots and slot bytes (0 slots: none); ranks in
 * rank order, the ends at ranks 0 and W - 1. */
static int sim_chain_slots;
static uint64_t sim_chain_slot_bytes;

static int create_session(session_t *s, int world, int lanes, int kind, int node_base,
                          uint64_t slot_bytes, int swap_rank, int swap_peer) {
    memset(s, 0, sizeof(*s));
    s->world = world;
    s->lanes = lanes;
    s->kind = kind;
    s->node_base = node_base;
    s->slot_bytes = slot_bytes;
    build_fabric(s);
    plan_routes(s, swap_rank, swap_peer);
    uint64_t record = roce_blob_bytes();
    uint8_t *blobs = (uint8_t *)calloc((size_t)(unsigned)world, (size_t)record);
    for (int r = 0; r < world; r++) {
        rank_t *k = &s->ranks[r];
        if (roce_layout(world, slot_bytes, k->layout) != 0) {
            snprintf(s->error, sizeof(s->error), "layout refused");
            free(blobs);
            return -1;
        }
        k->arena_bytes = k->layout[4];
        k->chain_off = 0;
        if (sim_chain_slots != 0) {
            if (roce_chain_layout(lanes, sim_chain_slots, sim_chain_slot_bytes, k->chain_layout) != 0) {
                snprintf(s->error, sizeof(s->error), "chain layout refused");
                free(blobs);
                return -1;
            }
            k->chain_off = (k->arena_bytes + 4095u) / 4096u * 4096u;
            k->arena_bytes = k->chain_off + k->chain_layout[8];
        }
        k->arena_alloc = (uint8_t *)calloc(1, (size_t)k->arena_bytes + 4096);
        k->arena = k->arena_alloc + ((4096 - ((uintptr_t)k->arena_alloc % 4096)) % 4096);
        const char *names[4];
        int gid_indices[4] = {3, 3, 3, 3};
        for (int i = 0; i < k->n_open; i++) names[i] = k->names[k->open_roles[i]];
        char err[400];
        if (rank0_post_order != NULL) {
            if (r == 0) setenv("SIRCL_POST_ORDER", rank0_post_order, 1);
            else unsetenv("SIRCL_POST_ORDER");
        }
        k->ctx = roce_create(world, r, names, k->n_open, k->lane_devices, lanes, gid_indices, 0,
                             k->arena, k->arena_bytes, slot_bytes, err, sizeof(err));
        if (k->ctx == NULL) {
            snprintf(s->error, sizeof(s->error), "rank %d create: %s", r, err);
            free(blobs);
            return -1;
        }
        if (sim_window != 0) {
            uint32_t windows[MAX_WORLD * 2];
            for (int p = 0; p < world; p++) {
                for (int l = 0; l < lanes; l++) {
                    windows[p * lanes + l] = (p != r && hops(kind, world, r, p) >= 2) ? sim_window : 0;
                }
            }
            if (roce_set_forward(k->ctx, windows, sim_chunk) != 0) {
                snprintf(s->error, sizeof(s->error), "rank %d forward windows: %s", r, roce_error(k->ctx));
                free(blobs);
                return -1;
            }
        }
        if (sim_chain_slots != 0 &&
            roce_set_chain(k->ctx, r > 0 ? r - 1 : -1, r < world - 1 ? r + 1 : -1, sim_chain_slots,
                           sim_chain_slot_bytes, k->chain_off) != 0) {
            snprintf(s->error, sizeof(s->error), "rank %d chain: %s", r, roce_error(k->ctx));
            free(blobs);
            return -1;
        }
        if (roce_local_blob(k->ctx, blobs + (size_t)r * record, record) != 0) {
            snprintf(s->error, sizeof(s->error), "rank %d blob", r);
            free(blobs);
            return -1;
        }
    }
    for (int r = 0; r < world; r++) {
        if (roce_connect(s->ranks[r].ctx, blobs, record * (uint64_t)world) != 0) {
            snprintf(s->error, sizeof(s->error), "rank %d connect: %s", r, roce_error(s->ranks[r].ctx));
            free(blobs);
            return -1;
        }
    }
    free(blobs);
    return 0;
}

static int lane_check_session(session_t *s, int timeout_ms) {
    for (int r = 0; r < s->world; r++) {
        if (roce_lane_check(s->ranks[r].ctx, timeout_ms) != 0) {
            snprintf(s->error, sizeof(s->error), "%s", roce_error(s->ranks[r].ctx));
            return -1;
        }
    }
    return 0;
}

static int start_session(session_t *s) {
    for (int r = 0; r < s->world; r++) {
        if (roce_start(s->ranks[r].ctx) != 0) {
            snprintf(s->error, sizeof(s->error), "rank %d start: %s", r, roce_error(s->ranks[r].ctx));
            return -1;
        }
    }
    return 0;
}

static void destroy_session(session_t *s) {
    for (int r = 0; r < s->world; r++) {
        if (s->ranks[r].ctx != NULL) roce_destroy(s->ranks[r].ctx);
        s->ranks[r].ctx = NULL;
        free(s->ranks[r].arena_alloc);
        s->ranks[r].arena_alloc = NULL;
    }
}

static volatile uint32_t *ctrl_of(rank_t *k) { return (volatile uint32_t *)(k->arena + k->layout[3]); }
static uint8_t *send_of(rank_t *k, session_t *s, uint32_t seq) {
    return k->arena + k->layout[2] + (uint64_t)(seq & 1u) * s->slot_bytes;
}
static uint8_t *recv_of(rank_t *k, session_t *s, int source, uint32_t seq) {
    return k->arena + ((uint64_t)source * SLOTS + (seq & 1u)) * s->slot_bytes;
}
static volatile uint32_t *flag_of(rank_t *k, session_t *s, int ns, int source, uint32_t seq, int lane) {
    uint64_t index = (uint64_t)ns * s->world * SLOTS * s->lanes +
                     ((uint64_t)source * SLOTS + (seq & 1u)) * s->lanes + lane;
    return (volatile uint32_t *)(k->arena + k->layout[1] + index * FLAG_STRIDE);
}

/* -- scheduler ----------------------------------------------------------------------- */

static atomic_int scheduler_running;
static uint64_t scheduler_seed = 1;
/* A timing case: the scheduler spins instead of sleeping, so writes run when their latency ends. */
static int sim_spin;

static void *scheduler_main(void *arg) {
    (void)arg;
    fv_progress(0, scheduler_seed);
    while (atomic_load(&scheduler_running)) {
        if (fv_progress(32, 0) == 0) {
            if (sim_spin) {
                sched_yield();
            } else {
                struct timespec ts = {0, 20000};
                nanosleep(&ts, NULL);
            }
        }
    }
    return NULL;
}

static pthread_t scheduler_thread;
static void scheduler_start(uint64_t seed) {
    scheduler_seed = seed;
    atomic_store(&scheduler_running, 1);
    pthread_create(&scheduler_thread, NULL, scheduler_main, NULL);
}
static void scheduler_stop(void) {
    atomic_store(&scheduler_running, 0);
    pthread_join(scheduler_thread, NULL);
}

/* -- test hooks: random progress-thread pauses ------------------------------------------- */

static atomic_int pause_enabled;
static atomic_int pauses_taken;
static atomic_uint pause_rng = 12345u;

static void pause_hook(void *arg, int point, uint32_t seq, int peer) {
    (void)arg;
    (void)point;
    (void)seq;
    (void)peer;
    if (!atomic_load(&pause_enabled)) return;
    unsigned x = atomic_fetch_add(&pause_rng, 2654435761u);
    x ^= x >> 15;
    if (x % 23u != 0) return;
    atomic_fetch_add(&pauses_taken, 1);
    struct timespec ts = {0, (long)(5000 + (x % 150) * 1000)};
    nanosleep(&ts, NULL);
}

/* -- kernel threads --------------------------------------------------------------------- */

enum { MIX_ONESHOT = 0, MIX_ALL = 1 };

typedef struct {
    session_t *s;
    int rank;
    int n_ops;
    int mix;
    uint32_t fixed_bytes;   /* nonzero: every op a two-shot of this many bytes, no data checked */
    uint32_t first_seq;   /* sequence of the first op */
    uint32_t rng;
    int stop_proxy_at;    /* op index around which rank 0's proxy is stopped, -1: never */
    int failed;
    uint32_t ops_done;
    char why[512];
} kernel_t;

static atomic_int abort_waits;

/* A rank whose kernel starts op `op` `ms` milliseconds late (rank -1: none). */
static struct { int rank; int op; int ms; } sim_lag = {-1, -1, 0};

/* Like the GPU kernels: the wait limit is command ring word 7 in microseconds,
 * read when the wait starts (0: WAIT_LIMIT_NS). */
static int wait_flag(kernel_t *t, volatile uint32_t *flag, uint32_t seq, const char *what) {
    uint64_t start = now_ns();
    uint32_t limit_us = __atomic_load_n(&ctrl_of(&t->s->ranks[t->rank])[CTRL_WAIT_LIMIT_US], __ATOMIC_RELAXED);
    uint64_t limit_ns = limit_us ? (uint64_t)limit_us * 1000ull : WAIT_LIMIT_NS;
    unsigned spins = 0;
    while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != seq) {
        if (++spins % 256 == 0) {
            sched_yield();
            uint64_t waited = now_ns() - start;
            if (atomic_load(&abort_waits) || waited > limit_ns) {
                snprintf(t->why, sizeof(t->why), "rank %d waited too long for %s at sequence %u "
                         "(%.3f s, limit %.3f s)", t->rank, what, seq, (double)waited * 1e-9,
                         (double)limit_ns * 1e-9);
                t->failed = 1;
                return -1;
            }
        }
    }
    return 0;
}

static int wait_peer_flags(kernel_t *t, int ns, int source, uint32_t seq) {
    rank_t *k = &t->s->ranks[t->rank];
    for (int l = 0; l < t->s->lanes; l++) {
        char what[96];
        snprintf(what, sizeof(what), "namespace %d flag of rank %d lane %d", ns, source, l);
        if (wait_flag(t, flag_of(k, t->s, ns, source, seq, l), seq, what) != 0) return -1;
    }
    return 0;
}

static int fail_data(kernel_t *t, const char *what, int source, uint32_t seq) {
    snprintf(t->why, sizeof(t->why), "rank %d received wrong %s from rank %d at sequence %u", t->rank,
             what, source, seq);
    t->failed = 1;
    return -1;
}

static void ring(rank_t *k, uint32_t seq, uint32_t word, uint32_t nbytes) {
    volatile uint32_t *ctrl = ctrl_of(k);
    ctrl[CTRL_OP_WORD + (seq & 1u)] = word;
    ctrl[CTRL_NBYTES] = nbytes;
    __atomic_store_n(&ctrl[CTRL_DOORBELL], seq, __ATOMIC_RELEASE);
}

static int run_oneshot(kernel_t *t, uint32_t seq, uint32_t nbytes) {
    session_t *s = t->s;
    rank_t *k = &s->ranks[t->rank];
    fill(send_of(k, s, seq), 0, nbytes, pattern_key(1, t->rank, seq, 0));
    ring(k, seq, ((uint32_t)OP_ONESHOT << OP_SHIFT) | nbytes, nbytes);
    for (int p = 0; p < s->world; p++) {
        if (p == t->rank) continue;
        if (wait_peer_flags(t, 0, p, seq) != 0) return -1;
        if (!matches(recv_of(k, s, p, seq), 0, nbytes, pattern_key(1, p, seq, 0))) {
            return fail_data(t, "one-shot payload", p, seq);
        }
    }
    return 0;
}

/* A two-shot op without data: doorbell, every peer's scatter flags, phase doorbell, gather flags. */
static int run_twoshot_timed(kernel_t *t, uint32_t seq, uint32_t nbytes) {
    session_t *s = t->s;
    rank_t *k = &s->ranks[t->rank];
    ring(k, seq, ((uint32_t)OP_TWOSHOT << OP_SHIFT) | nbytes, nbytes);
    for (int p = 0; p < s->world; p++) {
        if (p != t->rank && wait_peer_flags(t, 0, p, seq) != 0) return -1;
    }
    __atomic_store_n(&ctrl_of(k)[CTRL_PHASE + 1], seq, __ATOMIC_RELEASE);
    for (int p = 0; p < s->world; p++) {
        if (p != t->rank && wait_peer_flags(t, 1, p, seq) != 0) return -1;
    }
    return 0;
}

static int run_twoshot(kernel_t *t, uint32_t seq, uint32_t nbytes, int scatter) {
    session_t *s = t->s;
    rank_t *k = &s->ranks[t->rank];
    uint8_t *send = send_of(k, s, seq);
    uint32_t packs = nbytes / 16u, first, count;
    fill(send, 0, nbytes, pattern_key(2, t->rank, seq, 0));
    ring(k, seq, ((uint32_t)(scatter ? OP_SCATTER : OP_TWOSHOT) << OP_SHIFT) | nbytes, nbytes);
    chunk_range(packs, s->world, t->rank, &first, &count);
    for (int p = 0; p < s->world; p++) {
        if (p == t->rank) continue;
        if (wait_peer_flags(t, 0, p, seq) != 0) return -1;
        if (!matches(recv_of(k, s, p, seq) + (uint64_t)first * 16u, (uint64_t)first * 16u,
                     (uint64_t)count * 16u, pattern_key(2, p, seq, 0))) {
            return fail_data(t, scatter ? "scatter chunk" : "two-shot scatter chunk", p, seq);
        }
    }
    if (scatter) return 0;
    /* The reduced own chunk replaces the staged one at the same offsets. */
    fill(send + (uint64_t)first * 16u, (uint64_t)first * 16u, (uint64_t)count * 16u,
         pattern_key(3, t->rank, seq, 1));
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(&ctrl_of(k)[CTRL_PHASE + 1], seq, __ATOMIC_RELEASE);
    for (int p = 0; p < s->world; p++) {
        if (p == t->rank) continue;
        uint32_t pf, pc;
        chunk_range(packs, s->world, p, &pf, &pc);
        if (wait_peer_flags(t, 1, p, seq) != 0) return -1;
        if (!matches(recv_of(k, s, p, seq) + (uint64_t)pf * 16u, (uint64_t)pf * 16u,
                     (uint64_t)pc * 16u, pattern_key(3, p, seq, 1))) {
            return fail_data(t, "two-shot gather chunk", p, seq);
        }
    }
    return 0;
}

static uint32_t descriptor(phase_t ph) {
    return 0x80000000u | ((uint32_t)ph.ns << 14) | ((uint32_t)ph.peer << 10) |
           ((uint32_t)ph.end << 5) | (uint32_t)ph.first;
}

/* Byte range of chunk positions [first, end) of a `packs`-pack message. */
static void position_bytes(uint32_t packs, int world, int first, int end, uint64_t *lo, uint64_t *len) {
    uint32_t a, b, unused;
    chunk_range(packs, world, first, &a, &unused);
    chunk_range(packs, world, end, &b, &unused);
    *lo = (uint64_t)a * 16u;
    *len = (uint64_t)(b - a) * 16u;
}

/* Fill positions [first, end) with each position owner's final pattern. */
static void fill_final(uint8_t *send, uint32_t packs, int world, const int *order, int first, int end,
                       uint32_t seq) {
    for (int j = first; j < end; j++) {
        uint64_t lo, len;
        position_bytes(packs, world, j, j + 1, &lo, &len);
        fill(send + lo, lo, len, pattern_key(6, order[j], seq, 0));
    }
}

static int final_matches(const uint8_t *recv, uint32_t packs, int world, const int *order, int first,
                         int end, uint32_t seq) {
    for (int j = first; j < end; j++) {
        uint64_t lo, len;
        position_bytes(packs, world, j, j + 1, &lo, &len);
        if (!matches(recv + lo, lo, len, pattern_key(6, order[j], seq, 0))) return 0;
    }
    return 1;
}

/* A described op following the Swing schedule. The emulated kernel keeps the
 * real kernel's data flow: a reduce-scatter step writes partial sums only into
 * the rank's own remaining range (never into a range an earlier phase may
 * still be sending), the last step leaves the rank's own final chunk, and an
 * all-gather step copies the received final chunks into the send slot before
 * the next phase sends them on. */
static int run_described(kernel_t *t, uint32_t seq, uint32_t nbytes) {
    session_t *s = t->s;
    rank_t *k = &s->ranks[t->rank];
    uint8_t *send = send_of(k, s, seq);
    uint32_t packs = nbytes / 16u;
    phase_t mine[2 * 5], theirs[2 * 5];
    int phases = swing_phases(s->world, t->rank, mine);
    int steps = phases / 2;
    int order[MAX_WORLD], n_order = 0;
    swing_order(s->world, steps, 0, 0, order, &n_order);
    fill(send, 0, nbytes, pattern_key(5, t->rank, seq, 0));
    for (int ph = 0; ph < phases; ph++) {
        if (ph == steps) {
            /* After the reduce-scatter the own chunk holds the final sum. */
            int own = 0;
            while (order[own] != t->rank) own++;
            fill_final(send, packs, s->world, order, own, own + 1, seq);
        }
        ctrl_of(k)[CTRL_DESC + ph] = descriptor(mine[ph]);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (ph == 0) {
            ring(k, seq, ((uint32_t)OP_DESCRIBED << OP_SHIFT) | nbytes, nbytes);
        } else {
            __atomic_store_n(&ctrl_of(k)[CTRL_PHASE + ph], seq, __ATOMIC_RELEASE);
        }
        int q = mine[ph].peer;
        swing_phases(s->world, q, theirs);
        if (theirs[ph].peer != t->rank || theirs[ph].ns != mine[ph].ns) {
            snprintf(t->why, sizeof(t->why), "Swing schedule is not symmetric at rank %d phase %d", t->rank, ph);
            t->failed = 1;
            return -1;
        }
        if (wait_peer_flags(t, mine[ph].ns, q, seq) != 0) return -1;
        uint64_t lo, len;
        position_bytes(packs, s->world, theirs[ph].first, theirs[ph].end, &lo, &len);
        const uint8_t *got = recv_of(k, s, q, seq);
        if (ph < steps) {
            uint64_t key = ph == 0 ? pattern_key(5, q, seq, 0) : pattern_key(7, q, seq, ph);
            if (!matches(got + lo, lo, len, key)) return fail_data(t, "reduce-scatter range", q, seq);
            if (ph + 1 < steps) {
                /* Partial sums of the next step's range: the rank's own part. */
                uint64_t olo, olen;
                position_bytes(packs, s->world, theirs[ph].first, theirs[ph].end, &olo, &olen);
                (void)olo;
                (void)olen;
                phase_t next = mine[ph + 1];
                /* The range this rank sends at the next step is part of what it keeps now. */
                uint64_t nlo, nlen;
                position_bytes(packs, s->world, next.first, next.end, &nlo, &nlen);
                fill(send + nlo, nlo, nlen, pattern_key(7, t->rank, seq, ph + 1));
            }
        } else {
            if (!final_matches(got, packs, s->world, order, theirs[ph].first, theirs[ph].end, seq)) {
                return fail_data(t, "all-gather range", q, seq);
            }
            fill_final(send, packs, s->world, order, theirs[ph].first, theirs[ph].end, seq);
        }
    }
    return 0;
}

static void *kernel_main(void *arg) {
    kernel_t *t = (kernel_t *)arg;
    session_t *s = t->s;
    int power_of_two = (s->world & (s->world - 1)) == 0;
    for (int i = 0; i < t->n_ops && !t->failed; i++) {
        uint32_t seq = t->first_seq + (uint32_t)i;
        /* Every rank draws the same op kind and size for one sequence. */
        uint32_t draw = (uint32_t)mix(((uint64_t)seq << 8) ^ (uint64_t)t->rng);
        uint32_t nbytes = 16u * (1u + draw % (uint32_t)(s->slot_bytes / 16u));
        int kind = OP_ONESHOT;
        if (t->mix == MIX_ALL) {
            kind = (int)((draw >> 20) % 4u);
            if (kind == OP_DESCRIBED && !power_of_two) kind = OP_TWOSHOT;
        }
        rank_t *k = &s->ranks[t->rank];
        if (t->rank == sim_lag.rank && i == sim_lag.op) usleep((useconds_t)sim_lag.ms * 1000u);
        int stopped = 0;
        if (t->rank == 0 && i == t->stop_proxy_at) {
            roce_stop(k->ctx);
            stopped = 1;
        }
        int rc;
        if (t->fixed_bytes != 0) rc = run_twoshot_timed(t, seq, t->fixed_bytes);
        else if (kind == OP_ONESHOT) rc = run_oneshot(t, seq, nbytes);
        else if (kind == OP_DESCRIBED) rc = run_described(t, seq, nbytes);
        else rc = run_twoshot(t, seq, nbytes, kind == OP_SCATTER);
        if (stopped && rc == 0 && i + 1 < t->n_ops) {
            /* The proxy missed op i; ring op i+1 too, then restart it. */
            uint32_t next = seq + 1;
            uint32_t draw2 = (uint32_t)mix(((uint64_t)next << 8) ^ (uint64_t)t->rng);
            uint32_t nb2 = 16u * (1u + draw2 % (uint32_t)(s->slot_bytes / 16u));
            fill(send_of(k, s, next), 0, nb2, pattern_key(1, t->rank, next, 0));
            ring(k, next, nb2, nb2);
            if (roce_start(k->ctx) != 0) {
                snprintf(t->why, sizeof(t->why), "restart: %s", roce_error(k->ctx));
                t->failed = 1;
                break;
            }
            for (int p = 0; p < s->world && !t->failed; p++) {
                if (p == t->rank) continue;
                if (wait_peer_flags(t, 0, p, next) != 0) break;
                if (!matches(recv_of(k, s, p, next), 0, nb2, pattern_key(1, p, next, 0))) {
                    fail_data(t, "one-shot payload after a missed doorbell", p, next);
                }
            }
            i++;
            t->ops_done++;
        }
        if (rc != 0) break;
        t->ops_done++;
    }
    return NULL;
}

/* Every rank runs the same op script; ops of rank 0 that the scenario stops
 * the proxy around are one-shot so peers can finish them. */
static int run_ops(session_t *s, int n_ops, int mix, uint32_t first_seq, uint32_t seed,
                   int stop_proxy_at, char *why, size_t why_len) {
    pthread_t threads[MAX_WORLD];
    kernel_t kernels[MAX_WORLD];
    for (int r = 0; r < s->world; r++) {
        memset(&kernels[r], 0, sizeof(kernels[r]));
        kernels[r].s = s;
        kernels[r].rank = r;
        kernels[r].n_ops = n_ops;
        kernels[r].mix = mix;
        kernels[r].first_seq = first_seq;
        kernels[r].rng = seed;
        kernels[r].stop_proxy_at = stop_proxy_at;
        pthread_create(&threads[r], NULL, kernel_main, &kernels[r]);
    }
    for (int r = 0; r < s->world; r++) pthread_join(threads[r], NULL);
    for (int r = 0; r < s->world; r++) {
        if (kernels[r].failed) {
            const char *proxy = "";
            for (int q = 0; q < s->world; q++) {
                if (roce_failed(s->ranks[q].ctx)) proxy = roce_error(s->ranks[q].ctx);
            }
            snprintf(why, why_len, "%s%s%s", kernels[r].why, *proxy ? "; proxy: " : "", proxy);
            return -1;
        }
    }
    for (int r = 0; r < s->world; r++) {
        if (roce_failed(s->ranks[r].ctx)) {
            snprintf(why, why_len, "rank %d proxy failed: %s", r, roce_error(s->ranks[r].ctx));
            return -1;
        }
    }
    return 0;
}

/* Wait until every posted write ran and every proxy posted the last op. */
static void quiesce(session_t *s, uint32_t last_seq) {
    uint64_t start = now_ns();
    for (;;) {
        int done = fv_pending() == 0;
        for (int r = 0; r < s->world; r++) {
            if ((uint32_t)roce_stat(s->ranks[r].ctx, 2) != last_seq) done = 0;
        }
        if (done || now_ns() - start > 5000000000ull) return;
        usleep(200);
    }
}

/* -- cases ---------------------------------------------------------------------------- */

static int failures, cases;
static const char *const *selected;
static int n_selected;

static int wanted(const char *name) {
    if (n_selected == 0) return strstr(name, "-minutes") == NULL;   /* long cases run on request */
    for (int i = 0; i < n_selected; i++) {
        if (strncmp(name, selected[i], strlen(selected[i])) == 0) return 1;
    }
    return 0;
}

static void report(const char *name, int ok, const char *detail) {
    cases++;
    if (!ok) failures++;
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, detail && *detail ? ": " : "", detail ? detail : "");
    fflush(stdout);
}

static const char *kind_name(int kind) { return kind == KIND_CYCLE ? "cycle" : "path"; }

static void case_ops(const char *label, int world, int kind, int lanes, int n_ops, int mix,
                     uint32_t first_seq, int pauses, int stop_proxy_at, uint64_t slot_bytes) {
    char name[160];
    snprintf(name, sizeof(name), "%s/%s%d/lanes%d", label, kind_name(kind), world, lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    atomic_store(&pause_enabled, 0);
    atomic_store(&pauses_taken, 0);
    if (create_session(s, world, lanes, kind, 0, slot_bytes, -1, -1) != 0) {
        snprintf(why, sizeof(why), "%s", s->error);
    } else {
        scheduler_start(0x1234u + (uint64_t)world * 31u + (uint64_t)kind * 7u + (uint64_t)lanes);
        if (lane_check_session(s, 2000) != 0) {
            snprintf(why, sizeof(why), "%s", s->error);
        } else {
            /* Start every rank's doorbell just before the first op. */
            for (int r = 0; r < world; r++) {
                volatile uint32_t *ctrl = ctrl_of(&s->ranks[r]);
                ctrl[CTRL_DOORBELL] = first_seq - 1;
                for (int k = 1; k < 8; k++) ctrl[CTRL_PHASE + k] = first_seq - 1;
            }
            if (start_session(s) != 0) {
                snprintf(why, sizeof(why), "%s", s->error);
            } else {
                atomic_store(&pause_enabled, pauses);
                ok = run_ops(s, n_ops, mix, first_seq, 0xBEEFu + (uint32_t)world, stop_proxy_at, why,
                             sizeof(why)) == 0;
                atomic_store(&pause_enabled, 0);
                if (ok) {
                    quiesce(s, first_seq + (uint32_t)n_ops - 1);
                    uint64_t posted = 0;
                    for (int r = 0; r < world; r++) posted += roce_stat(s->ranks[r].ctx, 0);
                    if (posted != (uint64_t)world * (uint64_t)n_ops) {
                        ok = 0;
                        snprintf(why, sizeof(why), "%llu ops posted, %llu expected",
                                 (unsigned long long)posted, (unsigned long long)world * n_ops);
                    } else if (pauses && atomic_load(&pauses_taken) == 0) {
                        ok = 0;
                        snprintf(why, sizeof(why), "no progress-thread pause happened");
                    } else if (pauses) {
                        snprintf(why, sizeof(why), "%d pauses", atomic_load(&pauses_taken));
                    }
                }
            }
        }
        scheduler_stop();
    }
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* A deliberately malformed op on a running session must fail its proxy with
 * a message naming `expect`, and post nothing for that sequence. */
static void case_protocol(const char *name, int scenario, const char *expect) {
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, 3, 2, KIND_CYCLE, 0, 16384, -1, -1) == 0) {
        scheduler_start(77);
        if (scenario == 1) roce_test_disable_multi_phase(s->ranks[0].ctx);
        if (lane_check_session(s, 2000) == 0 && start_session(s) == 0) {
            rank_t *k = &s->ranks[0];
            volatile uint32_t *ctrl = ctrl_of(k);
            uint32_t seq = 1;
            fv_clear_events();
            if (scenario == 0) {
                /* Phase-1 doorbell of a one-shot op. */
                ring(k, seq, 256u, 256u);
                usleep(20000);
                __atomic_store_n(&ctrl[CTRL_PHASE + 1], seq, __ATOMIC_RELEASE);
            } else if (scenario == 1) {
                /* A two-shot op code on a session without multi-phase ops. */
                ring(k, seq, ((uint32_t)OP_TWOSHOT << OP_SHIFT) | 256u, 256u);
            } else {
                /* A described op whose descriptor names the own rank. */
                ctrl[CTRL_DESC + 0] = 0x80000000u | (0u << 10) | (3u << 5);
                ring(k, seq, ((uint32_t)OP_DESCRIBED << OP_SHIFT) | 256u, 256u);
            }
            uint64_t start = now_ns();
            while (!roce_failed(k->ctx) && now_ns() - start < 3000000000ull) usleep(1000);
            const char *error = roce_error(k->ctx);
            /* Scenario 0 posts phase 0 of its one-shot op legitimately; no
             * namespace-1 flag may follow. The others post nothing at all. */
            int posted_bad = 0;
            uint64_t n = fv_event_count();
            uint64_t lines_ns0 = (uint64_t)s->world * SLOTS * (uint64_t)s->lanes;
            for (uint64_t i = 0; i < n; i++) {
                fv_event_t e;
                fv_event(i, &e);
                int mine = 0;
                for (int role = 0; role < 4; role++) mine |= e.src_device == k->devices[role];
                if (!mine || e.phase != 0) continue;
                if (scenario != 0) {
                    posted_bad = 1;
                    continue;
                }
                for (int r = 1; r < s->world; r++) {
                    uint64_t base = (uint64_t)(uintptr_t)s->ranks[r].arena + s->ranks[r].layout[1];
                    if ((e.flags & 2u) && e.remote_addr >= base &&
                        (e.remote_addr - base) / FLAG_STRIDE >= lines_ns0 &&
                        e.remote_addr < base + (uint64_t)s->world * SLOTS * 4u * FLAG_STRIDE) {
                        posted_bad = 1;
                    }
                }
            }
            ok = roce_failed(k->ctx) && strstr(error, expect) != NULL && strstr(error, "sequence 1") != NULL &&
                 !posted_bad;
            snprintf(why, sizeof(why), "%s", error);
        } else {
            snprintf(why, sizeof(why), "%s", s->error);
        }
        scheduler_stop();
    } else {
        snprintf(why, sizeof(why), "%s", s->error);
    }
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

static void case_injected(void) {
    const char *name = "injected-failure/cycle4/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, 4, 2, KIND_CYCLE, 0, 16384, -1, -1) == 0) {
        scheduler_start(91);
        if (lane_check_session(s, 2000) == 0 && start_session(s) == 0) {
            /* Rank 1's lane 1 toward rank 3 fails at sequence 5. */
            uint64_t record = roce_blob_bytes();
            uint8_t *blob = (uint8_t *)malloc((size_t)record);
            roce_local_blob(s->ranks[1].ctx, blob, record);
            /* qp_num[device][peer] starts after the fixed header; read it through the layout */
            uint32_t qp_num = 0;
            {
                /* header: 6 u32, 2 u64, rkey[4], mtu[4], lid[4] (u16), gid[4][16] */
                size_t offset = 6 * 4 + 2 * 8 + 4 * 4 + 4 * 4 + 4 * 2 + 4 * 16;
                int device = s->ranks[1].lane_devices[3 * 2 + 1];
                memcpy(&qp_num, blob + offset + ((size_t)device * 16 + 3) * 4, 4);
            }
            free(blob);
            fv_inject_failure(qp_num, 5u);
            pthread_t threads[4];
            kernel_t kernels[4];
            for (int r = 0; r < 4; r++) {
                memset(&kernels[r], 0, sizeof(kernels[r]));
                kernels[r].s = s;
                kernels[r].rank = r;
                kernels[r].n_ops = 8;
                kernels[r].mix = MIX_ONESHOT;
                kernels[r].first_seq = 1;
                kernels[r].rng = 5;
                kernels[r].stop_proxy_at = -1;
            }
            /* Ranks wait at most a short time here: the failure stops sequence 5. */
            for (int r = 0; r < 4; r++) pthread_create(&threads[r], NULL, kernel_main, &kernels[r]);
            uint64_t start = now_ns();
            while (!roce_failed(s->ranks[1].ctx) && now_ns() - start < 5000000000ull) usleep(1000);
            const char *error = roce_error(s->ranks[1].ctx);
            ok = roce_failed(s->ranks[1].ctx) && strstr(error, "RDMA write of sequence 5 to rank 3 lane 1") != NULL;
            snprintf(why, sizeof(why), "%s", error);
            /* The failed sequence never completes: release the waiting kernel threads. */
            atomic_store(&abort_waits, 1);
            for (int r = 0; r < 4; r++) pthread_join(threads[r], NULL);
            atomic_store(&abort_waits, 0);
        } else {
            snprintf(why, sizeof(why), "%s", s->error);
        }
        scheduler_stop();
    } else {
        snprintf(why, sizeof(why), "%s", s->error);
    }
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

static int setenv_order(const char *value) {
    rank0_post_order = NULL;
    if (value != NULL && strcmp(value, "rank") != 0 && strcmp(value, "ring-farthest") != 0) {
        rank0_post_order = value;  /* an explicit list names the peers of one rank */
        return 0;
    }
    return value ? setenv("SIRCL_POST_ORDER", value, 1) : unsetenv("SIRCL_POST_ORDER");
}

static void case_post_order(const char *label, const char *value, const int *expected_rank0) {
    char name[96];
    snprintf(name, sizeof(name), "post-order/%s", label);
    if (!wanted(name)) return;
    fv_reset();
    setenv_order(value);
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, 8, 2, KIND_CYCLE, 0, 16384, -1, -1) == 0) {
        scheduler_start(13);
        if (lane_check_session(s, 2000) == 0 && start_session(s) == 0) {
            fv_clear_events();
            ok = run_ops(s, 3, MIX_ONESHOT, 1, 7, -1, why, sizeof(why)) == 0;
            quiesce(s, 3);
            if (ok) {
                /* Peers of rank 0's posted flag writes, per op, in post order. */
                int order[3][16], count[3] = {0, 0, 0};
                uint64_t n = fv_event_count();
                rank_t *k = &s->ranks[0];
                for (uint64_t i = 0; i < n; i++) {
                    fv_event_t e;
                    fv_event(i, &e);
                    if (e.phase != 0 || !(e.flags & 2u) || e.inline_word < 1 || e.inline_word > 3) continue;
                    int mine = 0;
                    for (int role = 0; role < 4; role++) mine |= e.src_device == k->devices[role];
                    if (!mine) continue;
                    int peer = (int)(e.wr_id & 0xFFu);
                    int op = (int)e.inline_word - 1;
                    if (count[op] == 0 || order[op][count[op] - 1] != peer) order[op][count[op]++] = peer;
                }
                for (int op = 0; op < 3 && ok; op++) {
                    if (count[op] != 7) ok = 0;
                    for (int i = 0; i < 7 && ok; i++) ok = order[op][i] == expected_rank0[i];
                }
                if (!ok) {
                    snprintf(why, sizeof(why), "rank 0 posted peers %d %d %d %d %d %d %d", order[0][0], order[0][1],
                             order[0][2], order[0][3], order[0][4], order[0][5], order[0][6]);
                }
            }
        } else {
            snprintf(why, sizeof(why), "%s", s->error);
        }
        scheduler_stop();
    } else {
        snprintf(why, sizeof(why), "%s", s->error);
    }
    destroy_session(s);
    free(s);
    setenv_order(NULL);
    unsetenv("SIRCL_POST_ORDER");
    report(name, ok, why);
}

static void case_two_sessions(void) {
    const char *name = "two-sessions/cycle4+path2";
    if (!wanted(name)) return;
    fv_reset();
    session_t *a = (session_t *)calloc(1, sizeof(session_t));
    session_t *b = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(a, 4, 2, KIND_CYCLE, 0, 16384, -1, -1) == 0 &&
        create_session(b, 2, 2, KIND_PATH, 8, 16384, -1, -1) == 0) {
        scheduler_start(21);
        if (lane_check_session(a, 2000) == 0 && lane_check_session(b, 2000) == 0 &&
            start_session(a) == 0 && start_session(b) == 0) {
            fv_clear_events();
            char wa[512] = "", wb[512] = "";
            /* Run both groups at the same time. */
            pthread_t ta[4], tb[2];
            kernel_t ka[4], kb[2];
            for (int r = 0; r < 4; r++) {
                memset(&ka[r], 0, sizeof(ka[r]));
                ka[r] = (kernel_t){.s = a, .rank = r, .n_ops = 300, .mix = MIX_ALL, .first_seq = 1, .rng = 3,
                                   .stop_proxy_at = -1};
                pthread_create(&ta[r], NULL, kernel_main, &ka[r]);
            }
            for (int r = 0; r < 2; r++) {
                memset(&kb[r], 0, sizeof(kb[r]));
                kb[r] = (kernel_t){.s = b, .rank = r, .n_ops = 300, .mix = MIX_ALL, .first_seq = 1, .rng = 4,
                                   .stop_proxy_at = -1};
                pthread_create(&tb[r], NULL, kernel_main, &kb[r]);
            }
            for (int r = 0; r < 4; r++) pthread_join(ta[r], NULL);
            for (int r = 0; r < 2; r++) pthread_join(tb[r], NULL);
            ok = 1;
            for (int r = 0; r < 4; r++) if (ka[r].failed) { ok = 0; snprintf(wa, sizeof(wa), "%s", ka[r].why); }
            for (int r = 0; r < 2; r++) if (kb[r].failed) { ok = 0; snprintf(wb, sizeof(wb), "%s", kb[r].why); }
            quiesce(a, 300);
            quiesce(b, 300);
            /* Every executed write lands in an arena of its own session. */
            uint64_t n = fv_event_count();
            for (uint64_t i = 0; i < n && ok; i++) {
                fv_event_t e;
                fv_event(i, &e);
                if (e.phase != 1) continue;
                session_t *own = NULL;
                for (int r = 0; r < 4; r++) for (int role = 0; role < 4; role++) if (a->ranks[r].devices[role] == e.src_device) own = a;
                for (int r = 0; r < 2; r++) for (int role = 0; role < 4; role++) if (b->ranks[r].devices[role] == e.src_device) own = b;
                int inside = 0;
                for (int r = 0; own != NULL && r < own->world; r++) {
                    uint64_t base = (uint64_t)(uintptr_t)own->ranks[r].arena;
                    if (e.remote_addr >= base && e.remote_addr + e.length <= base + own->ranks[r].arena_bytes) inside = 1;
                }
                if (!inside) {
                    ok = 0;
                    snprintf(wa, sizeof(wa), "a write left its session (event %llu)", (unsigned long long)i);
                }
            }
            snprintf(why, sizeof(why), "%s%s", wa, wb);
        } else {
            snprintf(why, sizeof(why), "%s%s", a->error, b->error);
        }
        scheduler_stop();
    } else {
        snprintf(why, sizeof(why), "%s%s", a->error, b->error);
    }
    destroy_session(a);
    destroy_session(b);
    free(a);
    free(b);
    report(name, ok, why);
}

static void case_unpaired(void) {
    const char *name = "unpaired-lanes/cycle4/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    /* Rank 0 swaps its two lanes toward rank 1: lane 0 leaves on a secondary
     * function while rank 1's lane 0 is a primary one. */
    if (create_session(s, 4, 2, KIND_CYCLE, 0, 16384, 0, 1) == 0) {
        scheduler_start(5);
        int rc = lane_check_session(s, 300);
        ok = rc != 0 && strstr(s->error, "lane check: lane ") != NULL &&
             strstr(s->error, "of rank 0 toward rank 1") != NULL;
        snprintf(why, sizeof(why), "%s", s->error);
        scheduler_stop();
    } else {
        snprintf(why, sizeof(why), "%s", s->error);
    }
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

static void case_phase_order(void) {
    /* Rank 0's proxy is stopped while a two-shot op runs: its peers deliver
     * phase 0, its kernel releases phase 1, and only then does the proxy
     * resume. It must post phase 0 before phase 1. */
    const char *name = "phase-order/cycle4/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, 4, 2, KIND_CYCLE, 0, 16384, -1, -1) == 0) {
        scheduler_start(17);
        if (lane_check_session(s, 2000) == 0 && start_session(s) == 0) {
            rank_t *k0 = &s->ranks[0];
            roce_stop(k0->ctx);
            fv_clear_events();
            /* All ranks stage a two-shot op of 4096 bytes at sequence 1. */
            uint32_t seq = 1, nbytes = 4096;
            for (int r = 0; r < 4; r++) {
                rank_t *k = &s->ranks[r];
                fill(send_of(k, s, seq), 0, nbytes, pattern_key(2, r, seq, 0));
                ring(k, seq, ((uint32_t)OP_TWOSHOT << OP_SHIFT) | nbytes, nbytes);
            }
            /* Rank 0 receives phase 0 from its peers, reduces, releases phase 1. */
            uint64_t start = now_ns();
            int arrived = 0;
            while (!arrived && now_ns() - start < 3000000000ull) {
                arrived = 1;
                for (int p = 1; p < 4; p++) for (int l = 0; l < 2; l++)
                    arrived &= *flag_of(k0, s, 0, p, seq, l) == seq;
                usleep(100);
            }
            __atomic_store_n(&ctrl_of(k0)[CTRL_PHASE + 1], seq, __ATOMIC_RELEASE);
            usleep(2000);
            roce_start(k0->ctx);
            /* Peers release phase 1 once they have rank 0's phase 0. */
            for (int r = 1; r < 4; r++) {
                rank_t *k = &s->ranks[r];
                start = now_ns();
                int got = 0;
                while (!got && now_ns() - start < 3000000000ull) {
                    got = 1;
                    for (int p = 0; p < 4; p++) if (p != r) for (int l = 0; l < 2; l++)
                        got &= *flag_of(k, s, 0, p, seq, l) == seq;
                    usleep(100);
                }
                __atomic_store_n(&ctrl_of(k)[CTRL_PHASE + 1], seq, __ATOMIC_RELEASE);
            }
            quiesce(s, seq);
            usleep(20000);
            /* In rank 0's posted writes, every namespace-0 flag precedes every namespace-1 flag. */
            uint64_t n = fv_event_count();
            uint64_t ns_boundary = (uint64_t)(uintptr_t)0;
            int seen_ns1 = 0, bad = 0, ns0 = 0, ns1 = 0;
            for (uint64_t i = 0; i < n; i++) {
                fv_event_t e;
                fv_event(i, &e);
                int mine = 0;
                for (int role = 0; role < 4; role++) mine |= e.src_device == k0->devices[role];
                if (!mine || e.phase != 0 || !(e.flags & 2u)) continue;
                /* namespace from the flag line: lines >= world*SLOTS*lanes are namespace 1 */
                int dest = -1;
                for (int r = 1; r < 4; r++) {
                    uint64_t base = (uint64_t)(uintptr_t)s->ranks[r].arena + s->ranks[r].layout[1];
                    if (e.remote_addr >= base && e.remote_addr < base + 4ull * 2 * 4 * FLAG_STRIDE) dest = r;
                    if (dest == r) {
                        uint64_t line = (e.remote_addr - base) / FLAG_STRIDE;
                        int ns = line >= 4u * 2u * 2u;
                        if (ns) { seen_ns1 = 1; ns1++; } else { ns0++; if (seen_ns1) bad = 1; }
                        break;
                    }
                }
            }
            (void)ns_boundary;
            ok = !bad && ns0 == 6 && ns1 == 6 && !roce_failed(k0->ctx);
            snprintf(why, sizeof(why), "rank 0 posted %d phase-0 and %d phase-1 flags%s", ns0, ns1, bad ? ", out of order" : "");
        } else {
            snprintf(why, sizeof(why), "%s", s->error);
        }
        scheduler_stop();
    } else {
        snprintf(why, sizeof(why), "%s", s->error);
    }
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* Lanes through relays post windowed chunks: data stays exact and no such
 * queue pair ever has more than the window (plus one 4-byte flag) in flight. */
static void case_windows(int world, int kind, int lanes) {
    char name[96];
    snprintf(name, sizeof(name), "forward-windows/%s%d/lanes%d", kind_name(kind), world, lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    sim_window = 32768;
    sim_chunk = 8192;
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, world, lanes, kind, 0, 262144, -1, -1) != 0) {
        snprintf(why, sizeof(why), "%s", s->error);
    } else {
        scheduler_start(4242u + (uint64_t)world);
        if (lane_check_session(s, 2000) == 0 && start_session(s) == 0) {
            ok = run_ops(s, 200, MIX_ALL, 1, 0xC0FFEEu + (uint32_t)world, -1, why, sizeof(why)) == 0;
            quiesce(s, 200);
            uint64_t chunks = 0, worst = 0, direct_worst = 0;
            for (int r = 0; ok && r < world; r++) {
                chunks += roce_stat(s->ranks[r].ctx, 10);
                for (int p = 0; p < world; p++) {
                    if (p == r) continue;
                    for (int l = 0; l < lanes; l++) {
                        uint32_t qpn = roce_test_qp_num(s->ranks[r].ctx, s->ranks[r].lane_devices[p * lanes + l], p);
                        uint64_t inflight = fv_qp_max_inflight(qpn);
                        if (hops(kind, world, r, p) >= 2) {
                            if (inflight > worst) worst = inflight;
                        } else if (inflight > direct_worst) {
                            direct_worst = inflight;
                        }
                    }
                }
            }
            if (ok && (worst > sim_window + 4u || chunks == 0)) {
                ok = 0;
                snprintf(why, sizeof(why), "windowed lanes had %llu bytes in flight (window %u), %llu chunks",
                         (unsigned long long)worst, sim_window, (unsigned long long)chunks);
            } else if (ok) {
                snprintf(why, sizeof(why), "%llu chunks; in flight at most %llu bytes on relayed lanes, %llu on direct",
                         (unsigned long long)chunks, (unsigned long long)worst, (unsigned long long)direct_worst);
            }
        } else {
            snprintf(why, sizeof(why), "%s", s->error);
        }
        scheduler_stop();
    }
    sim_window = sim_chunk = 0;
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

static uint64_t env_u64(const char *name, uint64_t fallback) {
    const char *text = getenv(name);
    return text != NULL && *text ? strtoull(text, NULL, 0) : fallback;
}

/* Two-shot ops of fixed sizes on a cycle of eight with two lanes, forward windows on every relayed lane
 * and path latency in the stand-in: the time per op and the forward-window waits of the progress
 * threads, per size. A measurement, run on request; it fails only when an op fails. */
static void case_twoshot_timing(void) {
    const char *name = "twoshot-timing-minutes/cycle8";
    if (!wanted(name)) return;
    const int world = 8, lanes = 2;
    uint64_t latency = env_u64("SIM_LATENCY_NS", 3000), relay = env_u64("SIM_RELAY_NS", 2000);
    int n_ops = (int)env_u64("SIM_OPS", 300);
    uint32_t sizes[16];
    int n_sizes = 0;
    const char *text = getenv("SIM_SIZES");
    if (text == NULL || !*text) text = "49152,98304,196608,262144,393216,589824";
    for (const char *cursor = text; *cursor && n_sizes < 16;) {
        char *end;
        unsigned long long value = strtoull(cursor, &end, 0);
        if (end == cursor) break;
        sizes[n_sizes++] = (uint32_t)value / 16u * 16u;
        cursor = *end == ',' ? end + 1 : end;
    }
    fv_reset();
    fv_set_recording(0);
    fv_set_payloads(0);
    fv_set_latency(latency, relay);
    fv_set_rate(env_u64("SIM_RATE", 12000));
    fv_set_ack_delay(env_u64("SIM_ACK_DELAY_NS", 0));
    sim_spin = 1;
    sim_window = (uint32_t)env_u64("SIM_WINDOW", 65536);
    sim_chunk = (uint32_t)env_u64("SIM_CHUNK", 32768);
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, world, lanes, KIND_CYCLE, 0, 1u << 20, -1, -1) != 0) {
        snprintf(why, sizeof(why), "%s", s->error);
    } else {
        scheduler_start(77u);
        if (lane_check_session(s, 5000) == 0 && start_session(s) == 0) {
            ok = 1;
            uint32_t seq = 1;
            for (int z = 0; ok && z < n_sizes; z++) {
                uint64_t waits0 = 0, wait_ns0 = 0, max_ns = 0;
                for (int r = 0; r < world; r++) {
                    waits0 += roce_stat(s->ranks[r].ctx, 25);
                    wait_ns0 += roce_stat(s->ranks[r].ctx, 26);
                }
                pthread_t threads[MAX_WORLD];
                kernel_t kernels[MAX_WORLD];
                uint64_t started = now_ns();
                for (int r = 0; r < world; r++) {
                    memset(&kernels[r], 0, sizeof(kernels[r]));
                    kernels[r] = (kernel_t){.s = s, .rank = r, .n_ops = n_ops, .mix = MIX_ONESHOT, .first_seq = seq,
                                            .rng = 1, .stop_proxy_at = -1, .fixed_bytes = sizes[z]};
                    pthread_create(&threads[r], NULL, kernel_main, &kernels[r]);
                }
                for (int r = 0; r < world; r++) pthread_join(threads[r], NULL);
                uint64_t elapsed = now_ns() - started;
                seq += (uint32_t)n_ops;
                uint64_t waits = 0, wait_ns = 0;
                for (int r = 0; r < world; r++) {
                    if (kernels[r].failed) {
                        ok = 0;
                        snprintf(why, sizeof(why), "%s", kernels[r].why);
                    }
                    waits += roce_stat(s->ranks[r].ctx, 25);
                    wait_ns += roce_stat(s->ranks[r].ctx, 26);
                    uint64_t m = roce_stat(s->ranks[r].ctx, 27);
                    if (m > max_ns) max_ns = m;
                }
                printf("  twoshot %7u B: %8.1f us per op; forward-window waits %.2f per op and rank, %.1f us per op "
                       "and rank, longest %.1f us\n", sizes[z], (double)elapsed / n_ops * 1e-3,
                       (double)(waits - waits0) / n_ops / world, (double)(wait_ns - wait_ns0) / n_ops / world * 1e-3,
                       (double)max_ns * 1e-3);
                fflush(stdout);
            }
            snprintf(why, sizeof(why), "latency %llu ns + %llu ns per relay, completions %llu ns later, %llu bytes "
                     "per us per queue pair, windows %u B, chunks %u B, %d ops per size, forward proof %s",
                     (unsigned long long)latency, (unsigned long long)relay,
                     (unsigned long long)env_u64("SIM_ACK_DELAY_NS", 0), (unsigned long long)env_u64("SIM_RATE", 12000),
                     sim_window, sim_chunk, n_ops, roce_stat(s->ranks[0].ctx, 29) ? "on" : "off");
        } else {
            snprintf(why, sizeof(why), "%s", s->error);
        }
        scheduler_stop();
    }
    sim_window = sim_chunk = 0;
    sim_spin = 0;
    fv_set_latency(0, 0);
    fv_set_rate(0);
    fv_set_ack_delay(0);
    fv_set_payloads(1);
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* Forward windows with completions that return long after their writes landed: the op order proves
 * delivery first (fwd_prove), so windows free without the completions. Payloads stay exact, no relayed
 * queue pair ever holds more than its window (plus one flag) undelivered, and proofs freed bytes.
 * `wrap` runs one-shot ops from sequence 0xFFFFFFFE through the wrap: no namespace-1 flag line is ever
 * written and the namespace-0 lines start zeroed, so a zero read near sequence 0 must prove nothing. */
static void case_windows_proof(int world, int kind, int lanes, int wrap) {
    char name[96];
    snprintf(name, sizeof(name), "forward-proof%s/%s%d/lanes%d", wrap ? "-wrap" : "", kind_name(kind), world,
             lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    fv_set_latency(20000, 10000);
    fv_set_ack_delay(150000);
    sim_spin = 1;
    sim_window = 32768;
    sim_chunk = 8192;
    const char *saved = getenv("SIRCL_FORWARD_PROOF");
    char saved_value[16] = "";
    if (saved != NULL) snprintf(saved_value, sizeof(saved_value), "%s", saved);
    setenv("SIRCL_FORWARD_PROOF", "1", 1);
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    int created = create_session(s, world, lanes, kind, 0, 262144, -1, -1);
    if (saved != NULL) setenv("SIRCL_FORWARD_PROOF", saved_value, 1);
    else unsetenv("SIRCL_FORWARD_PROOF");
    if (created != 0) {
        snprintf(why, sizeof(why), "%s", s->error);
    } else {
        scheduler_start(5150u + (uint64_t)world);
        uint32_t first = wrap ? 0xFFFFFFFEu : 1u;
        int n_ops = wrap ? 60 : 200;
        int checked = lane_check_session(s, 5000) == 0;
        /* Every rank's doorbell starts just before the first op. */
        for (int r = 0; checked && r < world; r++) {
            volatile uint32_t *ctrl = ctrl_of(&s->ranks[r]);
            ctrl[CTRL_DOORBELL] = first - 1;
            for (int k = 1; k < 8; k++) ctrl[CTRL_PHASE + k] = first - 1;
        }
        if (checked && start_session(s) == 0) {
            ok = run_ops(s, n_ops, wrap ? MIX_ONESHOT : MIX_ALL, first, 0xBEEFu + (uint32_t)world, -1, why,
                         sizeof(why)) == 0;
            quiesce(s, first + (uint32_t)n_ops - 1);
            uint64_t proven = 0, worst = 0;
            for (int r = 0; ok && r < world; r++) {
                proven += roce_stat(s->ranks[r].ctx, 28);
                for (int p = 0; p < world; p++) {
                    if (p == r || hops(kind, world, r, p) < 2) continue;
                    for (int l = 0; l < lanes; l++) {
                        uint32_t qpn = roce_test_qp_num(s->ranks[r].ctx, s->ranks[r].lane_devices[p * lanes + l], p);
                        uint64_t inflight = fv_qp_max_inflight(qpn);
                        if (inflight > worst) worst = inflight;
                    }
                }
            }
            if (ok && (worst > sim_window + 4u || proven == 0)) {
                ok = 0;
                snprintf(why, sizeof(why), "relayed lanes held %llu bytes undelivered (window %u); proofs freed %llu bytes",
                         (unsigned long long)worst, sim_window, (unsigned long long)proven);
            } else if (ok) {
                snprintf(why, sizeof(why), "proofs freed %llu bytes; at most %llu bytes undelivered on relayed lanes "
                         "(window %u)", (unsigned long long)proven, (unsigned long long)worst, sim_window);
            }
        } else {
            snprintf(why, sizeof(why), "%s", s->error);
        }
        scheduler_stop();
    }
    sim_window = sim_chunk = 0;
    sim_spin = 0;
    fv_set_latency(0, 0);
    fv_set_ack_delay(0);
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

static void set_wait_limit(session_t *s, uint32_t limit_us) {
    for (int r = 0; r < s->world; r++) {
        __atomic_store_n(&ctrl_of(&s->ranks[r])[CTRL_WAIT_LIMIT_US], limit_us, __ATOMIC_RELEASE);
    }
}

/* Two wait regimes on one session (path of four, two lanes): with the startup
 * limit, a rank that starts an op `lag_ms` late holds every peer until it
 * arrives, and the group finishes every op; with the serving limit, the same
 * lag fails the peers' waits after the serving limit. The progress threads
 * idle through both lags without failing. */
static void case_wait_regimes(const char *name, uint32_t startup_ms, uint32_t serving_ms, uint32_t lag_ms) {
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, 4, 2, KIND_PATH, 0, 16384, -1, -1) != 0) {
        snprintf(why, sizeof(why), "%s", s->error);
    } else {
        scheduler_start(0x5151u);
        if (lane_check_session(s, 2000) != 0 || start_session(s) != 0) {
            snprintf(why, sizeof(why), "%s", s->error);
        } else {
            set_wait_limit(s, startup_ms * 1000u);
            sim_lag.rank = 0;
            sim_lag.op = 4;
            sim_lag.ms = (int)lag_ms;
            uint64_t started = now_ns();
            char first[512] = "";
            int startup_ok = run_ops(s, 12, MIX_ALL, 1, 0x7777u, -1, first, sizeof(first)) == 0;
            double startup_s = (double)(now_ns() - started) * 1e-9;
            int proxies_ok = 1;
            for (int r = 0; r < 4; r++) proxies_ok &= !roce_failed(s->ranks[r].ctx);
            if (!startup_ok || !proxies_ok || startup_s < lag_ms * 1e-3) {
                snprintf(why, sizeof(why), "startup regime: %s (%.3f s, proxies %s)",
                         startup_ok ? "finished" : first, startup_s, proxies_ok ? "running" : "failed");
            } else {
                set_wait_limit(s, serving_ms * 1000u);
                sim_lag.op = 2;
                started = now_ns();
                char second[512] = "";
                int serving_failed = run_ops(s, 12, MIX_ALL, 13, 0x8888u, -1, second, sizeof(second)) != 0;
                double serving_s = (double)(now_ns() - started) * 1e-9;
                if (!serving_failed || strstr(second, "waited too long") == NULL) {
                    snprintf(why, sizeof(why), "serving regime: a %u ms lag did not fail the waits (%s)",
                             lag_ms, second);
                } else {
                    ok = 1;
                    snprintf(why, sizeof(why), "startup limit %.1f s: rank 0 lagged %.1f s, all 12 ops "
                             "finished in %.2f s; serving limit %.1f s: the same lag failed the group in "
                             "%.2f s (%s)", startup_ms * 1e-3, lag_ms * 1e-3, startup_s, serving_ms * 1e-3,
                             serving_s, second);
                }
            }
            sim_lag.rank = -1;
            sim_lag.op = -1;
            sim_lag.ms = 0;
        }
        scheduler_stop();
    }
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* -- chain schedule: the kernels' part ------------------------------------------------ */

enum { CH_RECV = 0, CH_SEND = 1, CH_RFLAG = 2, CH_READY = 3, CH_CONSUMED = 4, CH_SENT = 5, CH_CREDIT = 6,
       CH_CTRL = 7 };

static uint8_t *chain_at(rank_t *k, int area, uint64_t offset) {
    return k->arena + k->chain_off + k->chain_layout[area] + offset;
}
static uint32_t *chain_slot(rank_t *k, int area, int stream, uint32_t m) {
    return (uint32_t *)chain_at(k, area, ((uint64_t)stream * sim_chain_slots + m) * sim_chain_slot_bytes);
}
static volatile uint32_t *chain_flag(rank_t *k, session_t *s, int stream, uint32_t m, int lane) {
    return (volatile uint32_t *)chain_at(k, CH_RFLAG, (((uint64_t)stream * sim_chain_slots + m) * s->lanes + lane) *
                                                          FLAG_STRIDE);
}
static volatile uint32_t *chain_line(rank_t *k, int area, int stream, uint32_t m) {
    return (volatile uint32_t *)chain_at(k, area, (uint64_t)stream * FLAG_STRIDE + 4u * m);
}

/* One rank's kernel of chain ops: four roles of two blocks each, every block
 * taking every other chunk of its role, polled without blocking, like the GPU
 * kernel's blocks. Values are 32-bit words added modulo 2^32. */
typedef struct {
    session_t *s;
    int rank;
    int n_ops;
    uint32_t seed;
    uint32_t chunk_bytes;
    uint64_t max_bytes;
    int oneshot_every;       /* a one-shot op after every n-th chain op, 0: none */
    uint32_t base[2];        /* global chunk count of each half before this op */
    uint32_t chain_seq;
    uint32_t main_seq;
    int failed;
    char why[512];
    kernel_t oneshot;        /* the one-shot ops' kernel state */
} chain_kernel_t;

static uint32_t chain_input(int rank, uint32_t op, uint64_t word) {
    return (uint32_t)mix(((uint64_t)rank << 40) ^ ((uint64_t)op << 20) ^ word ^ 0x51CA);
}

static uint64_t chain_op_bytes(uint32_t seed, uint32_t op, uint64_t max_bytes) {
    uint64_t draw = mix(((uint64_t)seed << 32) ^ op);
    uint64_t packs = 1 + draw % (max_bytes / 16u);
    if (draw % 7u == 0) packs = 1 + draw % 3u;         /* tiny ops: an empty or one-pack half */
    return packs * 16u;
}

typedef struct { int role; int block; uint32_t next; } chain_block_t;

/* Try chunk j of a role; 1 when it ran, 0 when a wait is not satisfied. */
static int chain_try(chain_kernel_t *t, int role, uint32_t j, const uint32_t *x, uint32_t *out, uint64_t a_words,
                     uint64_t half_words[2], uint32_t n[2]) {
    session_t *s = t->s;
    rank_t *k = &s->ranks[t->rank];
    int W = s->world, r = t->rank;
    int half = (role == 0 || role == 2) ? 0 : 1;  /* roles: 0 A reduce, 1 B reduce, 2 A results, 3 B results */
    uint32_t K = (uint32_t)sim_chain_slots;
    uint32_t g = t->base[half] + j, m = g % K, tag = g + 1u;
    uint64_t chunk_words = t->chunk_bytes / 4u;
    uint64_t first = (uint64_t)j * chunk_words;
    uint64_t words = half_words[half] - first < chunk_words ? half_words[half] - first : chunk_words;
    uint64_t offset = half == 0 ? 0 : a_words;
    (void)n;
    if (role == 0 || role == 1) {
        int position = role == 0 ? r : W - 1 - r;           /* index along this half's reduction */
        int has_in = position > 0, last = position == W - 1;
        int in_stream = role == 0 ? 0 : 2;
        int out_stream = last ? (role == 0 ? 1 : 3) : in_stream;
        if (has_in) {
            for (int l = 0; l < s->lanes; l++) {
                if (__atomic_load_n(chain_flag(k, s, in_stream, m, l), __ATOMIC_ACQUIRE) != tag) return 0;
            }
        }
        uint32_t sent = __atomic_load_n(chain_line(k, CH_SENT, out_stream, 0), __ATOMIC_ACQUIRE);
        if ((int32_t)(sent - (tag - K)) < 0) return 0;
        uint32_t *in = chain_slot(k, CH_RECV, in_stream, m);
        uint32_t *dst = chain_slot(k, CH_SEND, out_stream, m);
        for (uint64_t w = 0; w < words; w++) {
            uint32_t v = x[offset + first + w] + (has_in ? in[w] : 0u);
            dst[w] = v;
            if (last) out[offset + first + w] = v;
        }
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (has_in) __atomic_store_n(chain_line(k, CH_CONSUMED, in_stream, m), tag, __ATOMIC_RELEASE);
        __atomic_store_n(chain_line(k, CH_READY, out_stream, m), tag, __ATOMIC_RELEASE);
        return 1;
    }
    int stream = role == 2 ? 1 : 3;
    for (int l = 0; l < s->lanes; l++) {
        if (__atomic_load_n(chain_flag(k, s, stream, m, l), __ATOMIC_ACQUIRE) != tag) return 0;
    }
    uint32_t *in = chain_slot(k, CH_RECV, stream, m);
    for (uint64_t w = 0; w < words; w++) out[offset + first + w] = in[w];
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(chain_line(k, CH_CONSUMED, stream, m), tag, __ATOMIC_RELEASE);
    return 1;
}

static int run_chain_op(chain_kernel_t *t, uint32_t op) {
    session_t *s = t->s;
    rank_t *k = &s->ranks[t->rank];
    int W = s->world, r = t->rank;
    uint64_t bytes = chain_op_bytes(t->seed, op, t->max_bytes);
    uint64_t packs = bytes / 16u, a_packs = packs / 2u, b_packs = packs - a_packs;
    uint64_t words = packs * 4u, a_words = a_packs * 4u;
    uint64_t half_words[2] = {a_words, b_packs * 4u};
    uint32_t chunk_packs = t->chunk_bytes / 16u;
    uint32_t n[2] = {(uint32_t)((a_packs + chunk_packs - 1) / chunk_packs),
                     (uint32_t)((b_packs + chunk_packs - 1) / chunk_packs)};
    uint32_t *x = (uint32_t *)malloc((size_t)words * 4u);
    uint32_t *out = (uint32_t *)calloc((size_t)words, 4u);
    for (uint64_t w = 0; w < words; w++) x[w] = chain_input(r, op, w);
    /* Doorbell: parameters into slot (seq & 1), then the sequence. */
    uint32_t seq = t->chain_seq + 1u;
    volatile uint32_t *ctrl = (volatile uint32_t *)chain_at(k, CH_CTRL, 0);
    ctrl[1 + 4 * (seq & 1u)] = (uint32_t)(a_packs * 16u);
    ctrl[2 + 4 * (seq & 1u)] = (uint32_t)(b_packs * 16u);
    ctrl[3 + 4 * (seq & 1u)] = t->chunk_bytes;
    __atomic_store_n(&ctrl[0], seq, __ATOMIC_RELEASE);
    /* Roles present: A reduce and B reduce everywhere, A results except at the last
     * rank, B results except at the first. */
    chain_block_t blocks[8];
    int n_blocks = 0;
    for (int role = 0; role < 4; role++) {
        if ((role == 2 && r == W - 1) || (role == 3 && r == 0)) continue;
        for (int b = 0; b < 2; b++) blocks[n_blocks++] = (chain_block_t){role, b, (uint32_t)b};
    }
    uint64_t start = now_ns(), last_progress = start;
    int done = 0;
    while (!done) {
        done = 1;
        int moved = 0;
        for (int i = 0; i < n_blocks; i++) {
            chain_block_t *bl = &blocks[i];
            int half = (bl->role == 0 || bl->role == 2) ? 0 : 1;
            while (bl->next < n[half]) {
                if (!chain_try(t, bl->role, bl->next, x, out, a_words, half_words, n)) break;
                bl->next += 2;
                moved = 1;
            }
            if (bl->next < n[half]) done = 0;
        }
        if (moved) {
            last_progress = now_ns();
        } else if (!done) {
            sched_yield();
            if (atomic_load(&abort_waits) || now_ns() - last_progress > WAIT_LIMIT_NS) {
                snprintf(t->why, sizeof(t->why), "rank %d chain op %u (%llu bytes) made no progress for %.1f s",
                         r, seq, (unsigned long long)bytes, (double)(now_ns() - last_progress) * 1e-9);
                t->failed = 1;
                free(x);
                free(out);
                return -1;
            }
        }
    }
    for (uint64_t w = 0; w < words && !t->failed; w++) {
        uint32_t want = 0;
        for (int q = 0; q < W; q++) want += chain_input(q, op, w);
        if (out[w] != want) {
            snprintf(t->why, sizeof(t->why), "rank %d chain op %u (%llu bytes): word %llu is 0x%08x, the sum is "
                     "0x%08x", r, seq, (unsigned long long)bytes, (unsigned long long)w, out[w], want);
            t->failed = 1;
        }
    }
    free(x);
    free(out);
    t->base[0] += n[0];
    t->base[1] += n[1];
    t->chain_seq = seq;
    return t->failed ? -1 : 0;
}

static void *chain_main(void *arg) {
    chain_kernel_t *t = (chain_kernel_t *)arg;
    for (int i = 0; i < t->n_ops && !t->failed; i++) {
        if (run_chain_op(t, (uint32_t)i) != 0) break;
        if (t->oneshot_every && (i + 1) % t->oneshot_every == 0) {
            uint32_t seq = ++t->main_seq;
            uint32_t draw = (uint32_t)mix(((uint64_t)seq << 8) ^ t->seed);
            uint32_t nbytes = 16u * (1u + draw % (uint32_t)(t->s->slot_bytes / 16u));
            if (run_oneshot(&t->oneshot, seq, nbytes) != 0) {
                snprintf(t->why, sizeof(t->why), "%s", t->oneshot.why);
                t->failed = 1;
            }
        }
    }
    return NULL;
}

enum { EV_OP = 1, EV_READY = 2, EV_POSTED = 3, EV_DONE = 4, EV_CONSUMED = 5, EV_CREDIT_OUT = 6 };

/* One rank's event trace after `n_ops` chain ops: every chain op taken once, and on every chain stream the chunks traced ready, posted and done in tag
 * order, each posted after it was ready and done after it was posted, every posted chunk
 * done, inbound chunks consumed in tag order and credits never decreasing. */
static int check_chain_trace(roce_ctx_t *ctx, int rank, int n_ops, uint32_t capacity, char *why, size_t len) {
    uint64_t *words = (uint64_t *)calloc(2u * (size_t)capacity, sizeof(uint64_t));
    uint64_t lost = 0;
    int64_t n = roce_trace_take(ctx, words, capacity, &lost);
    int ok = 1, ops = 0;
    uint32_t ready[4] = {0}, posted[4] = {0}, done[4] = {0}, consumed[4] = {0}, credit[4] = {0};
    if (n < 0 || lost != 0) {
        snprintf(why, len, "rank %d: trace returned %lld records and lost %llu", rank, (long long)n,
                 (unsigned long long)lost);
        ok = 0;
    }
    for (int64_t i = 0; ok && i < n; i++) {
        uint64_t ns = words[2 * i];
        uint32_t value = (uint32_t)(words[2 * i + 1] & 0xFFFFFFFFu);
        int event = (int)((words[2 * i + 1] >> 32) & 0xFFFFu), stream = (int)(words[2 * i + 1] >> 48);
        (void)ns;
        if (stream >= 8) {
            snprintf(why, len, "rank %d: record %lld of stream %d", rank, (long long)i, stream);
            ok = 0;
            break;
        }
        if (event == EV_OP) {
            ops += stream == 0;
            continue;
        }
        if (stream >= 4) continue;
        uint32_t *count = event == EV_READY ? ready : event == EV_POSTED ? posted : event == EV_DONE ? done
                          : event == EV_CONSUMED ? consumed : NULL;
        uint32_t *before = event == EV_POSTED ? ready : event == EV_DONE ? posted : NULL;
        if (count != NULL && (value != count[stream] + 1u || (before != NULL && value > before[stream]))) {
            snprintf(why, len, "rank %d: stream %d event %d of tag %u after %u (prerequisite %u)", rank, stream,
                     event, value, count[stream], before != NULL ? before[stream] : 0u);
            ok = 0;
        } else if (count != NULL) {
            count[stream] = value;
        } else if (event == EV_CREDIT_OUT) {
            if ((int32_t)(value - credit[stream]) < 0) {
                snprintf(why, len, "rank %d: stream %d credit %u after %u", rank, stream, value, credit[stream]);
                ok = 0;
            }
            credit[stream] = value;
        }
    }
    for (int s = 0; ok && s < 4; s++) {
        if (posted[s] != done[s]) {
            snprintf(why, len, "rank %d: stream %d posted %u chunks, %u done", rank, s, posted[s], done[s]);
            ok = 0;
        }
    }
    if (ok && ops != n_ops) {
        snprintf(why, len, "rank %d: %d chain ops traced, %d run", rank, ops, n_ops);
        ok = 0;
    }
    free(words);
    return ok;
}

/* Chain ops of random sizes on a chain of `world` ranks: every rank's result is the
 * word sum of every rank's input, with one-shot ops in between. With a trace capacity,
 * every rank keeps an event trace, checked after the ops (check_chain_trace). */
static void case_chain(const char *label, int world, int kind, int lanes, int slots, uint64_t slot_bytes,
                       uint32_t chunk_bytes, int n_ops, uint64_t max_bytes, int pauses, uint32_t trace) {
    char name[160];
    snprintf(name, sizeof(name), "%s/%s%d/lanes%d", label, kind_name(kind), world, lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    sim_chain_slots = slots;
    sim_chain_slot_bytes = slot_bytes;
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    atomic_store(&pauses_taken, 0);
    if (create_session(s, world, lanes, kind, 0, 16384, -1, -1) != 0) {
        snprintf(why, sizeof(why), "%s", s->error);
    } else {
        scheduler_start(0x3C3Cu + (uint64_t)world * 17u + (uint64_t)lanes);
        int traced = 1;
        for (int r = 0; r < world && trace != 0; r++) {
            if (roce_set_trace(s->ranks[r].ctx, trace) != 0) {
                snprintf(s->error, sizeof(s->error), "rank %d trace: %s", r, roce_error(s->ranks[r].ctx));
                traced = 0;
            }
        }
        if (!traced || lane_check_session(s, 2000) != 0 || start_session(s) != 0) {
            snprintf(why, sizeof(why), "%s", s->error);
        } else {
            atomic_store(&pause_enabled, pauses);
            pthread_t threads[MAX_WORLD];
            chain_kernel_t *kernels = (chain_kernel_t *)calloc((size_t)world, sizeof(chain_kernel_t));
            uint64_t started = now_ns();
            for (int r = 0; r < world; r++) {
                chain_kernel_t *t = &kernels[r];
                t->s = s;
                t->rank = r;
                t->n_ops = n_ops;
                t->seed = 0xC4A1u + (uint32_t)world;
                t->chunk_bytes = chunk_bytes;
                t->max_bytes = max_bytes;
                t->oneshot_every = 3;
                t->oneshot.s = s;
                t->oneshot.rank = r;
                pthread_create(&threads[r], NULL, chain_main, t);
            }
            for (int r = 0; r < world; r++) pthread_join(threads[r], NULL);
            atomic_store(&pause_enabled, 0);
            double seconds = (double)(now_ns() - started) * 1e-9;
            ok = 1;
            for (int r = 0; r < world && ok; r++) {
                if (kernels[r].failed) {
                    ok = 0;
                    const char *proxy = "";
                    for (int q = 0; q < world; q++) {
                        if (roce_failed(s->ranks[q].ctx)) proxy = roce_error(s->ranks[q].ctx);
                    }
                    snprintf(why, sizeof(why), "%s%s%s", kernels[r].why, *proxy ? "; proxy: " : "", proxy);
                } else if (roce_failed(s->ranks[r].ctx)) {
                    ok = 0;
                    snprintf(why, sizeof(why), "rank %d proxy: %s", r, roce_error(s->ranks[r].ctx));
                }
            }
            if (ok) {
                uint64_t chunks = 0, credits = 0, ops = 0;
                for (int r = 0; r < world; r++) {
                    chunks += roce_stat(s->ranks[r].ctx, 13);
                    credits += roce_stat(s->ranks[r].ctx, 14);
                    ops += roce_stat(s->ranks[r].ctx, 12);
                }
                if (ops != (uint64_t)world * (uint64_t)n_ops) {
                    ok = 0;
                    snprintf(why, sizeof(why), "%llu chain ops seen, %llu expected", (unsigned long long)ops,
                             (unsigned long long)world * (uint64_t)n_ops);
                } else {
                    snprintf(why, sizeof(why), "%d ops in %.2f s: %llu chunks, %llu credits%s%s", n_ops, seconds,
                             (unsigned long long)chunks, (unsigned long long)credits,
                             pauses ? (atomic_load(&pauses_taken) ? ", with pauses" : ", no pause happened") : "",
                             trace ? ", traces consistent" : "");
                    if (pauses && atomic_load(&pauses_taken) == 0) ok = 0;
                }
                for (int r = 0; r < world && ok && trace != 0; r++) {
                    ok = check_chain_trace(s->ranks[r].ctx, r, n_ops, trace, why, sizeof(why));
                }
            }
            free(kernels);
        }
        scheduler_stop();
    }
    destroy_session(s);
    free(s);
    sim_chain_slots = 0;
    sim_chain_slot_bytes = 0;
    report(name, ok, why);
}

/* Chain schedules that cannot work are refused with a message naming the problem. */
static void case_chain_refusals(void) {
    const char *name = "chain/refusals";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    sim_chain_slots = 3;
    sim_chain_slot_bytes = 8192;
    if (create_session(s, 3, 2, KIND_PATH, 0, 16384, -1, -1) != 0) {
        snprintf(why, sizeof(why), "%s", s->error);
    } else {
        rank_t *k = &s->ranks[1];
        roce_ctx_t *c = k->ctx;
        int geometry = roce_set_chain(c, 0, 2, 1, 8192, k->chain_off) != 0 &&
                       strstr(roce_error(c), "chain geometry") != NULL;
        int overlap = roce_set_chain(c, 0, 2, 3, 8192, 0) != 0 && strstr(roce_error(c), "does not fit") != NULL;
        int neighbors = roce_set_chain(c, 1, 2, 3, 8192, k->chain_off) != 0 &&
                        strstr(roce_error(c), "chain neighbors") != NULL;
        int ends = roce_set_chain(c, -1, -1, 3, 8192, k->chain_off) != 0;
        int fine = roce_set_chain(c, 0, 2, 3, 8192, k->chain_off) == 0;
        ok = geometry && overlap && neighbors && ends && fine;
        snprintf(why, sizeof(why), "geometry %d overlap %d neighbors %d ends %d fine %d", geometry, overlap,
                 neighbors, ends, fine);
    }
    sim_chain_slots = 0;
    sim_chain_slot_bytes = 0;
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* Windows that cannot work are refused with a message naming the lane. */
static void case_window_refusals(void) {
    const char *name = "forward-windows/refusals";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    char why[1024] = "";
    int ok = 0;
    if (create_session(s, 4, 2, KIND_CYCLE, 0, 16384, -1, -1) == 0) {
        roce_ctx_t *c = s->ranks[0].ctx;
        uint32_t windows[8] = {0};
        windows[2 * 2 + 1] = 4096;                      /* smaller than the chunk */
        int small = roce_set_forward(c, windows, 8192) != 0 && strstr(roce_error(c), "lane 1 toward rank 2");
        windows[2 * 2 + 1] = 0;
        windows[0] = 8192;                              /* the own rank's lane */
        int own = roce_set_forward(c, windows, 8192) != 0 && strstr(roce_error(c), "own lane 0");
        windows[0] = 0;
        windows[1 * 2 + 0] = 8192;
        int chunk = roce_set_forward(c, windows, 100) != 0 && strstr(roce_error(c), "multiple of 16");
        int fine = roce_set_forward(c, windows, 8192) == 0 && roce_set_forward(c, NULL, 0) == 0;
        ok = small && own && chunk && fine;
        snprintf(why, sizeof(why), "small %d own %d chunk %d fine %d", small, own, chunk, fine);
    } else {
        snprintf(why, sizeof(why), "%s", s->error);
    }
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* -- teardown ------------------------------------------------------------------------------ */

/* The gate of the teardown cases: while armed, a progress thread that takes a
 * doorbell blocks until released, and the first taker disarms the gate. This
 * holds a rank's whole posting of one op - data, flag and every later write it
 * would owe - at the moment its peer's context dies, which is the state a
 * close without a drain leaves a peer in whenever its kernel completed first. */
static atomic_int gate_armed;
static pthread_mutex_t gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cond = PTHREAD_COND_INITIALIZER;
static int gate_released;
/* The native test hook's doorbell point (the hook points of _roce_proxy.c). */
enum { ROCE_HOOK_DOORBELL = 0 };

static void teardown_gate_hook(void *arg, int point, uint32_t seq, int peer) {
    (void)arg;
    (void)seq;
    (void)peer;
    if (point != ROCE_HOOK_DOORBELL || !atomic_load(&gate_armed)) return;
    atomic_store(&gate_armed, 0);                 /* the first taker holds the debt */
    pthread_mutex_lock(&gate_mutex);
    while (!gate_released) pthread_cond_wait(&gate_cond, &gate_mutex);
    pthread_mutex_unlock(&gate_mutex);
}

static void gate_open(void) {
    pthread_mutex_lock(&gate_mutex);
    gate_released = 1;
    pthread_cond_broadcast(&gate_cond);
    pthread_mutex_unlock(&gate_mutex);
}

/* K one-shot ops on every rank of a 3-rank session, driven by the real kernel
 * threads; the session is left quiescent with every proxy alive. */
static int teardown_setup(session_t *s, int ops, kernel_t *kernels, pthread_t *threads, char *why, size_t len) {
    if (create_session(s, 3, 2, KIND_CYCLE, 0, 16384, -1, -1) != 0) {
        snprintf(why, len, "%s", s->error);
        return -1;
    }
    scheduler_start(0x7EAu);
    if (lane_check_session(s, 2000) != 0 || start_session(s) != 0) {
        snprintf(why, len, "%s", s->error);
        return -1;
    }
    for (int r = 0; r < 3; r++) {
        memset(&kernels[r], 0, sizeof(kernels[r]));
        kernels[r] = (kernel_t){.s = s, .rank = r, .n_ops = ops, .mix = MIX_ONESHOT, .first_seq = 1,
                                .rng = 9u, .stop_proxy_at = -1};
        pthread_create(&threads[r], NULL, kernel_main, &kernels[r]);
    }
    for (int r = 0; r < 3; r++) pthread_join(threads[r], NULL);
    for (int r = 0; r < 3; r++) {
        if (kernels[r].failed) {
            snprintf(why, len, "rank %d kernel: %s", r, kernels[r].why);
            return -1;
        }
    }
    quiesce(s, (uint32_t)ops);
    return 0;
}

/* Rank 0 destroys its context while rank 1's proxy still owes the whole posting
 * of the op whose doorbell it just took. Rank 1's proxy must fail on the writes
 * toward the destroyed context, and rank 2's must survive: the failure names
 * rank 0 and the retry counter, exactly what a serving rank's proxy reports
 * when a peer closed without waiting for the group. */
static void case_teardown_race(void) {
    const char *name = "teardown-race/cycle3/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    kernel_t kernels[3];
    pthread_t threads[3];
    char why[1024] = "";
    int ok = 0;
    atomic_store(&gate_armed, 0);
    gate_released = 0;
    roce_test_set_hook(teardown_gate_hook, NULL);
    if (teardown_setup(s, 40, kernels, threads, why, sizeof(why)) == 0) {
        uint32_t seq = 41, nbytes = 2048;
        atomic_store(&gate_armed, 1);
        /* Rank 1's kernel ends a round ahead of its proxy: the op's debt exists
         * while no kernel waits for it. */
        rank_t *k1 = &s->ranks[1];
        fill(send_of(k1, s, seq), 0, nbytes, pattern_key(1, 1, seq, 0));
        ring(k1, seq, ((uint32_t)OP_ONESHOT << OP_SHIFT) | nbytes, nbytes);
        uint64_t start = now_ns();
        while (atomic_load(&gate_armed) && now_ns() - start < 5000000000ull) usleep(100);
        if (atomic_load(&gate_armed)) {
            snprintf(why, sizeof(why), "rank 1's proxy never took doorbell %u", seq);
        } else {
            roce_destroy(s->ranks[0].ctx);
            s->ranks[0].ctx = NULL;
            gate_open();
            start = now_ns();
            while (!roce_failed(s->ranks[1].ctx) && now_ns() - start < 5000000000ull) usleep(100);
            const char *error = roce_error(s->ranks[1].ctx);
            int failed1 = roce_failed(s->ranks[1].ctx);
            int named = failed1 && strstr(error, "to rank 0") != NULL &&
                        strstr(error, "retry counter exceeded") != NULL;
            int failed2 = roce_failed(s->ranks[2].ctx);
            if (!failed1) {
                snprintf(why, sizeof(why), "rank 1's proxy survived writes toward rank 0's destroyed context");
            } else if (!named) {
                snprintf(why, sizeof(why), "rank 1's proxy failed otherwise: %s", error);
            } else if (failed2) {
                snprintf(why, sizeof(why), "rank 2's proxy failed too: %s", roce_error(s->ranks[2].ctx));
            } else {
                ok = 1;
                snprintf(why, sizeof(why), "rank 1: %.90s", error);
            }
        }
        gate_open();
    }
    scheduler_stop();
    roce_test_set_hook(pause_hook, NULL);
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* The two-round teardown: every rank's kernels are idle, round 1 waits for the
 * proxies to drain what they owe, every proxy stops, round 2 orders the
 * destruction after every stop, and only then do the contexts die. No proxy
 * may fail: nothing is posted toward a context that is gone; and every
 * roce_destroy releases every verbs object (returns 0). */
static void case_teardown_ordered(void) {
    const char *name = "teardown-ordered/cycle3/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    kernel_t kernels[3];
    pthread_t threads[3];
    char why[1024] = "";
    int ok = 0;
    if (teardown_setup(s, 40, kernels, threads, why, sizeof(why)) == 0) {
        /* Round 1 returns everywhere: nothing a kernel needs is owed, and the
         * still-running proxies finish their in-flight writes. */
        uint64_t start = now_ns();
        while (fv_pending() != 0 && now_ns() - start < 5000000000ull) usleep(100);
        for (int r = 0; r < 3; r++) roce_stop(s->ranks[r].ctx);     /* posts nothing more */
        /* Round 2: every proxy is stopped, so no write targets a context that
         * is about to die. */
        int failed = 0;
        for (int r = 0; r < 3; r++) {
            if (roce_failed(s->ranks[r].ctx)) {
                failed = 1;
                snprintf(why, sizeof(why), "rank %d's proxy failed: %s", r, roce_error(s->ranks[r].ctx));
            }
        }
        if (!failed) {
            int left = 0;
            for (int r = 0; r < 3; r++) {
                left += roce_destroy(s->ranks[r].ctx);
                s->ranks[r].ctx = NULL;
            }
            if (left != 0) {
                snprintf(why, sizeof(why), "%d verbs calls failed in the destroys", left);
            } else {
                ok = 1;
                snprintf(why, sizeof(why), "40 ops, two rounds, no proxy failed, every verbs object released");
            }
        }
    }
    scheduler_stop();
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

/* A destroy whose verbs calls fail reports how many: the stand-in fails the
 * next queue-pair destroy, so rank 0's roce_destroy returns 1 (its caller keeps
 * the arena) and the other ranks' return 0. */
static void case_teardown_failed_destroy(void) {
    const char *name = "teardown-failed-destroy/cycle3/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    session_t *s = (session_t *)calloc(1, sizeof(session_t));
    kernel_t kernels[3];
    pthread_t threads[3];
    char why[1024] = "";
    int ok = 0;
    if (teardown_setup(s, 10, kernels, threads, why, sizeof(why)) == 0) {
        for (int r = 0; r < 3; r++) roce_stop(s->ranks[r].ctx);
        fv_fail_teardown(1);
        int counts[3];
        for (int r = 0; r < 3; r++) {
            counts[r] = roce_destroy(s->ranks[r].ctx);
            s->ranks[r].ctx = NULL;
        }
        fv_fail_teardown(0);
        if (counts[0] == 1 && counts[1] == 0 && counts[2] == 0) {
            ok = 1;
            snprintf(why, sizeof(why), "rank 0's destroy reported 1 failed verbs call, ranks 1 and 2 none");
        } else {
            snprintf(why, sizeof(why), "failed verbs calls per rank %d, %d, %d (expected 1, 0, 0)", counts[0],
                     counts[1], counts[2]);
        }
    }
    scheduler_stop();
    destroy_session(s);
    free(s);
    report(name, ok, why);
}

static void case_swing_dump(void) {
    /* Print the Swing phases for cross-checking against tests/data/numeric.json. */
    if (!wanted("swing-schedule")) return;
    for (int world = 2; world <= 8; world *= 2) {
        for (int r = 0; r < world; r++) {
            phase_t ph[10];
            int n = swing_phases(world, r, ph);
            printf("SWING world=%d rank=%d", world, r);
            for (int i = 0; i < n; i++) printf(" %d,%d,%d,%d", ph[i].peer, ph[i].first, ph[i].end, ph[i].ns);
            printf("\n");
        }
    }
    report("swing-schedule", 1, "");
}

int main(int argc, char **argv) {
    selected = (const char *const *)(argv + 1);
    n_selected = argc - 1;
    roce_test_set_hook(pause_hook, NULL);
    setvbuf(stdout, NULL, _IOLBF, 0);
    static const int worlds[] = {2, 3, 4, 6, 8};
    for (int w = 0; w < 5; w++) {
        for (int kind = 0; kind < 2; kind++) {
            for (int lanes = 1; lanes <= 2; lanes++) {
                case_ops("oneshot-3000", worlds[w], kind, lanes, 3000, MIX_ONESHOT, 1, 0, -1, 16384);
            }
        }
    }
    for (int w = 0; w < 5; w++) {
        for (int kind = 0; kind < 2; kind++) {
            for (int lanes = 1; lanes <= 2; lanes++) {
                case_ops("mixed-ops", worlds[w], kind, lanes, 400, MIX_ALL, 1, 0, -1, 16384);
            }
        }
    }
    case_ops("sequence-wrap", 4, KIND_CYCLE, 2, 120, MIX_ALL, 0xFFFFFFC0u, 0, -1, 16384);
    case_ops("sequence-wrap", 3, KIND_PATH, 1, 120, MIX_ALL, 0xFFFFFFC0u, 0, -1, 16384);
    case_ops("pauses", 8, KIND_CYCLE, 2, 600, MIX_ALL, 1, 1, -1, 16384);
    case_ops("pauses", 6, KIND_PATH, 2, 600, MIX_ALL, 1, 1, -1, 16384);
    case_ops("missed-doorbell", 4, KIND_CYCLE, 2, 40, MIX_ONESHOT, 1, 0, 10, 16384);
    case_ops("missed-doorbell", 3, KIND_PATH, 1, 40, MIX_ONESHOT, 1, 0, 10, 16384);
    case_phase_order();
    case_protocol("protocol/phase-of-one-shot-op", 0, "phase 1 doorbell at sequence 1");
    case_protocol("protocol/op-code-without-multi-phase", 1, "op at sequence 1 has");
    case_protocol("protocol/invalid-descriptor", 2, "invalid phase 0 descriptor");
    case_injected();
    static const int rank_order[7] = {1, 2, 3, 4, 5, 6, 7};
    static const int farthest[7] = {4, 3, 5, 2, 6, 1, 7};
    static const int explicit_order[7] = {7, 1, 6, 2, 5, 3, 4};
    case_post_order("rank", NULL, rank_order);
    case_post_order("ring-farthest", "ring-farthest", farthest);
    case_post_order("explicit", "7,1,6,2,5,3,4", explicit_order);
    case_two_sessions();
    case_unpaired();
    case_windows(4, KIND_PATH, 2);
    case_windows(8, KIND_CYCLE, 2);
    case_windows(6, KIND_PATH, 1);
    case_windows_proof(8, KIND_CYCLE, 2, 0);
    case_windows_proof(4, KIND_PATH, 2, 0);
    case_windows_proof(8, KIND_CYCLE, 2, 1);
    case_windows_proof(4, KIND_PATH, 2, 1);
    case_window_refusals();
    case_wait_regimes("wait-regimes/path4/lanes2", 4000, 300, 2500);
    case_chain("chain", 2, KIND_PATH, 1, 3, 8192, 4096, 60, 98304, 0, 0);
    case_chain("chain", 3, KIND_PATH, 2, 3, 8192, 4096, 60, 98304, 0, 0);
    case_chain("chain", 4, KIND_PATH, 2, 4, 8192, 8192, 60, 131072, 0, 0);
    case_chain("chain", 4, KIND_PATH, 1, 2, 4096, 2048, 60, 65536, 0, 0);
    case_chain("chain", 6, KIND_PATH, 2, 3, 8192, 4096, 40, 98304, 0, 0);
    case_chain("chain", 8, KIND_CYCLE, 2, 4, 8192, 4096, 40, 131072, 0, 0);
    case_chain("chain-pauses", 4, KIND_PATH, 2, 3, 8192, 4096, 60, 98304, 1, 0);
    case_chain("chain-trace", 4, KIND_PATH, 2, 3, 8192, 4096, 30, 98304, 0, 1u << 16);
    case_chain_refusals();
    case_teardown_race();
    case_teardown_ordered();
    case_teardown_failed_destroy();
    case_wait_regimes("wait-regimes-minutes/path4/lanes2", 600000, 20000, 150000);
    case_swing_dump();
    case_twoshot_timing();
    printf("%d cases, %d failed\n", cases, failures);
    return failures ? 1 : 0;
}
