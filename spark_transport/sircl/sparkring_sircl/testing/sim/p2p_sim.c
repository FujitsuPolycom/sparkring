/*
 * CPU simulator of SIRCL's point-to-point channels.
 *
 * Runs the real native layer (sparkring_sircl/p2p/_p2p_proxy.c, built with
 * SIRCL_PROXY_TEST_HOOKS) against the in-memory verbs stand-in
 * (testing/fake_verbs). Every rank of a simulated group lives in this
 * process with its real progress thread. For every channel that carries
 * messages in a case, one sender thread plays the send kernel of its source
 * rank (wait for a free slot, stage the item's bytes, write the header and
 * the ready tag) and one receiver thread plays the receive kernel of its
 * destination rank (wait for every lane flag, check the header and every
 * byte, write the consumed tag); a scheduler thread runs posted writes in a
 * seeded order that keeps each queue pair's order.
 *
 * Fabrics are rings (cycles) or paths of W Sparks, cabled port 0 to the next
 * Spark's port 1, two links per cable. Route maps follow the lane rules of the
 * route derivation, and relayed lanes are tagged by destination as a site relay
 * plan does.
 *
 * Usage: p2p_sim [case-prefix ...]; prints PASS/FAIL per case and exits 1 when
 * any case fails. Build: testing/p2p_build.py.
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

/* -- native layer --------------------------------------------------------------------- */

typedef struct p2p_ctx p2p_ctx_t;
int p2p_abi_version(void);
int p2p_layout(int world, int lanes, int slots, uint64_t slot_bytes, uint64_t *out);
uint64_t p2p_blob_bytes(void);
p2p_ctx_t *p2p_create(int world, int rank, const char *const *device_names, int n_devices, const int *lane_devices,
                      int lane_count, const int *gid_indices, int traffic_class, const int *channels, void *region,
                      uint64_t region_bytes, int slots, uint64_t slot_bytes, char *err, uint64_t err_len);
int p2p_local_blob(p2p_ctx_t *c, void *out, uint64_t out_len);
int p2p_connect(p2p_ctx_t *c, const void *blobs, uint64_t blobs_len);
int p2p_lane_check(p2p_ctx_t *c, int timeout_ms);
int p2p_set_windows(p2p_ctx_t *c, const uint32_t *lane_window_bytes, uint32_t chunk_bytes);
int p2p_start(p2p_ctx_t *c);
void p2p_stop(p2p_ctx_t *c);
int p2p_failed(p2p_ctx_t *c);
const char *p2p_error(p2p_ctx_t *c);
uint64_t p2p_stat(p2p_ctx_t *c, int which);
uint64_t p2p_peer_stat(p2p_ctx_t *c, int peer, int which);
int p2p_destroy(p2p_ctx_t *c);
int p2p_test_set_base(p2p_ctx_t *c, uint32_t base);
uint32_t p2p_test_qp_num(p2p_ctx_t *c, int d, int p);

/* -- words of the protocol (p2p/protocol.py) ---------------------------------------------- */

enum { L_CONTROL = 0, L_BLOCK = 1, L_RECV = 2, L_SEND = 3, L_FLAG = 4, L_DESC = 5, L_READY = 6, L_CONSUMED = 7,
       L_SENT = 8, L_CREDIT = 9, L_TOTAL = 10 };
enum { C_WAIT_LIMIT_US = 0, C_ERROR_TAG = 1, C_ERROR_PEER = 2, C_ERROR_LANE = 3, C_ERROR_KIND = 4,
       C_ERROR_EXPECTED = 5, C_ERROR_GOT = 6, C_POISON = 7, C_ABORT = 8 };
enum { KIND_FLAG = 1, KIND_SLOT = 2, KIND_SIZE = 3 };
#define LINE 128u
#define LAST (1u << 31)
enum { KIND_CYCLE = 0, KIND_PATH = 1 };
enum { ROLE_CWP = 0, ROLE_CWS = 1, ROLE_CCWP = 2, ROLE_CCWS = 3 };
static const char *ROLE_DEVICE[4] = {"rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"};
#define MAX_WORLD 16
#define WAIT_LIMIT_NS 20000000000ull

/* -- helpers ------------------------------------------------------------------------------- */

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void sleep_us(uint64_t us) {
    struct timespec ts = {(time_t)(us / 1000000u), (long)((us % 1000000u) * 1000u)};
    nanosleep(&ts, NULL);
}

static uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

static uint8_t pattern_byte(uint64_t key, uint64_t offset) {
    uint64_t v = mix(key ^ ((offset / 8u) * 0x9E3779B97F4A7C15ull));
    return (uint8_t)(v >> (8u * (offset % 8u)));
}

static uint64_t message_key(int src, int dst, uint32_t message) {
    return ((uint64_t)src << 56) ^ ((uint64_t)dst << 48) ^ ((uint64_t)message * 0x100000001B3ull) ^ 0x5157u;
}

static void store_release(volatile uint32_t *address, uint32_t value) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

static uint32_t load_acquire(const volatile uint32_t *address) { return __atomic_load_n(address, __ATOMIC_ACQUIRE); }

static uint64_t rng_next(uint64_t *state) {
    *state = *state * 6364136223846793005ull + 1442695040888963407ull;
    return mix(*state);
}

/* -- fabric and groups -------------------------------------------------------------------- */

typedef struct {
    int rank, node;
    uint8_t *arena, *arena_alloc;
    uint64_t arena_bytes;
    p2p_ctx_t *ctx;
    int devices[4];
    char names[4][64];
    int n_open;
    int open_roles[4];
    int lane_devices[MAX_WORLD * 2];
    int lane_roles[MAX_WORLD][2];
} rank_t;

typedef struct {
    int world, lanes, kind, node_base, slots;
    uint64_t slot_bytes;
    uint64_t layout[11];
    int channels[MAX_WORLD][MAX_WORLD];
    uint32_t window;      /* forward window of relayed lanes (0: none) */
    uint32_t chunk;
    rank_t ranks[MAX_WORLD];
    char error[600];
} group_t;

static void gid_of(int node, int role, uint8_t gid[16]) {
    memset(gid, 0, 16);
    gid[10] = 0xff;
    gid[11] = 0xff;
    gid[12] = 10;
    gid[13] = (uint8_t)node;
    gid[14] = (uint8_t)role;
    gid[15] = 1;
}

/* Lane roles from `r` to `p` (the route derivation's rules). */
static void lane_roles(int kind, int world, int r, int p, int roles[2]) {
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

static void build_fabric(group_t *g) {
    int W = g->world;
    for (int r = 0; r < W; r++) {
        rank_t *k = &g->ranks[r];
        k->rank = r;
        k->node = g->node_base + r;
        for (int role = 0; role < 4; role++) {
            uint8_t gid[16];
            gid_of(k->node, role, gid);
            snprintf(k->names[role], sizeof(k->names[role]), "n%d.%s", k->node, ROLE_DEVICE[role]);
            k->devices[role] = fv_add_device(k->names[role], k->node, role / 2, role % 2, gid);
        }
        fv_set_relay(k->node, 1);
    }
    int cables = g->kind == KIND_PATH ? W - 1 : W;
    for (int i = 0; i < cables; i++) fv_add_cable(g->node_base + i, 0, g->node_base + (i + 1) % W, 1, 3u);
}

static void plan_routes(group_t *g) {
    int W = g->world;
    for (int r = 0; r < W; r++) {
        rank_t *k = &g->ranks[r];
        k->n_open = 0;
        for (int p = 0; p < W; p++) {
            for (int l = 0; l < g->lanes; l++) k->lane_devices[p * g->lanes + l] = -1;
            if (p == r || !g->channels[r][p]) continue;
            int roles[2];
            lane_roles(g->kind, W, r, p, roles);
            for (int l = 0; l < g->lanes; l++) {
                k->lane_roles[p][l] = roles[l];
                int index = -1;
                for (int i = 0; i < k->n_open; i++) {
                    if (k->open_roles[i] == roles[l]) index = i;
                }
                if (index < 0) {
                    index = k->n_open;
                    k->open_roles[k->n_open++] = roles[l];
                }
                k->lane_devices[p * g->lanes + l] = index;
            }
        }
    }
    for (int r = 0; r < W; r++) {
        for (int p = 0; p < W; p++) {
            if (p == r || !g->channels[r][p]) continue;
            int h = hops(g->kind, W, r, p);
            if (h < 2) continue;
            for (int l = 0; l < g->lanes; l++) {
                uint8_t gid[16];
                gid_of(g->ranks[p].node, g->ranks[p].lane_roles[r][l], gid);
                fv_set_dest_tag(g->ranks[r].devices[g->ranks[r].lane_roles[p][l]], gid, (uint32_t)(h - 1));
            }
        }
    }
}

static int relayed(const group_t *g, int r, int p) { return hops(g->kind, g->world, r, p) >= 2; }

/* All channels, or only those between ranks `adjacent` hops apart (0: every pair). */
static void set_channels(group_t *g, int adjacent) {
    for (int r = 0; r < g->world; r++) {
        for (int p = 0; p < g->world; p++) {
            g->channels[r][p] = r != p && (adjacent == 0 || hops(g->kind, g->world, r, p) == adjacent);
        }
    }
}

static volatile uint32_t *control_of(group_t *g, int r, int word) {
    return (volatile uint32_t *)(g->ranks[r].arena + 4u * (uint64_t)word);
}

static uint8_t *block_of(group_t *g, int r, int p) {
    return g->ranks[r].arena + g->layout[L_CONTROL] + (uint64_t)p * g->layout[L_BLOCK];
}

static volatile uint32_t *word_of(group_t *g, int r, int p, int area, uint32_t index) {
    return (volatile uint32_t *)(block_of(g, r, p) + g->layout[area] + 4u * (uint64_t)index);
}

static volatile uint32_t *flag_of(group_t *g, int r, int p, uint32_t m, int lane) {
    return (volatile uint32_t *)(block_of(g, r, p) + g->layout[L_FLAG] +
                                 ((uint64_t)m * (uint64_t)g->lanes + (uint64_t)lane) * LINE);
}

/* Every arena word a channel's items compare starts at `base`, the tag of item base - 1. */
static void set_words(group_t *g, uint32_t base) {
    for (int r = 0; r < g->world; r++) {
        for (int p = 0; p < g->world; p++) {
            if (!g->channels[r][p]) continue;
            for (int m = 0; m < g->slots; m++) {
                *word_of(g, r, p, L_READY, (uint32_t)m) = base;
                *word_of(g, r, p, L_CONSUMED, (uint32_t)m) = base;
                for (int l = 0; l < g->lanes; l++) *flag_of(g, r, p, (uint32_t)m, l) = base;
            }
            *word_of(g, r, p, L_SENT, 0) = base;
            *word_of(g, r, p, L_CREDIT, 0) = base;
        }
    }
}

static int create_group(group_t *g, int world, int lanes, int kind, int node_base, int slots, uint64_t slot_bytes,
                        uint32_t window, uint32_t chunk, int adjacent, uint32_t base) {
    memset(g, 0, sizeof(*g));
    g->world = world;
    g->lanes = lanes;
    g->kind = kind;
    g->node_base = node_base;
    g->slots = slots;
    g->slot_bytes = slot_bytes;
    g->window = window;
    g->chunk = chunk;
    set_channels(g, adjacent);
    if (p2p_layout(world, lanes, slots, slot_bytes, g->layout) != 0) {
        snprintf(g->error, sizeof(g->error), "layout refused");
        return -1;
    }
    build_fabric(g);
    plan_routes(g);
    uint64_t record = p2p_blob_bytes();
    uint8_t *blobs = (uint8_t *)calloc((size_t)(unsigned)world, (size_t)record);
    for (int r = 0; r < world; r++) {
        rank_t *k = &g->ranks[r];
        k->arena_bytes = g->layout[L_TOTAL];
        k->arena_alloc = (uint8_t *)calloc(1, (size_t)k->arena_bytes + 4096);
        k->arena = k->arena_alloc + ((4096 - ((uintptr_t)k->arena_alloc % 4096)) % 4096);
        const char *names[4];
        int gid_indices[4] = {3, 3, 3, 3};
        int channels[MAX_WORLD];
        for (int i = 0; i < k->n_open; i++) names[i] = k->names[k->open_roles[i]];
        for (int p = 0; p < world; p++) channels[p] = g->channels[r][p];
        char err[400];
        k->ctx = p2p_create(world, r, names, k->n_open, k->lane_devices, lanes, gid_indices, 0, channels, k->arena,
                            k->arena_bytes, slots, slot_bytes, err, sizeof(err));
        if (k->ctx == NULL) {
            snprintf(g->error, sizeof(g->error), "rank %d create: %s", r, err);
            free(blobs);
            return -1;
        }
        if (window != 0) {
            uint32_t windows[MAX_WORLD * 2];
            for (int p = 0; p < world; p++) {
                for (int l = 0; l < lanes; l++) {
                    windows[p * lanes + l] = (g->channels[r][p] && relayed(g, r, p)) ? window : 0;
                }
            }
            if (p2p_set_windows(k->ctx, windows, chunk) != 0) {
                snprintf(g->error, sizeof(g->error), "rank %d windows: %s", r, p2p_error(k->ctx));
                free(blobs);
                return -1;
            }
        }
        if (base != 0 && p2p_test_set_base(k->ctx, base) != 0) {
            snprintf(g->error, sizeof(g->error), "rank %d base refused", r);
            free(blobs);
            return -1;
        }
        if (p2p_local_blob(k->ctx, blobs + (size_t)r * record, record) != 0) {
            snprintf(g->error, sizeof(g->error), "rank %d blob", r);
            free(blobs);
            return -1;
        }
    }
    if (base != 0) set_words(g, base);
    for (int r = 0; r < world; r++) {
        if (p2p_connect(g->ranks[r].ctx, blobs, record * (uint64_t)world) != 0) {
            snprintf(g->error, sizeof(g->error), "rank %d connect: %s", r, p2p_error(g->ranks[r].ctx));
            free(blobs);
            return -1;
        }
    }
    free(blobs);
    for (int r = 0; r < world; r++) {
        if (p2p_lane_check(g->ranks[r].ctx, 5000) != 0) {
            snprintf(g->error, sizeof(g->error), "rank %d lane check: %s", r, p2p_error(g->ranks[r].ctx));
            return -1;
        }
    }
    for (int r = 0; r < world; r++) {
        if (p2p_start(g->ranks[r].ctx) != 0) {
            snprintf(g->error, sizeof(g->error), "rank %d start: %s", r, p2p_error(g->ranks[r].ctx));
            return -1;
        }
    }
    return 0;
}

static void destroy_group(group_t *g) {
    for (int r = 0; r < g->world; r++) {
        if (g->ranks[r].ctx != NULL) p2p_destroy(g->ranks[r].ctx);
        g->ranks[r].ctx = NULL;
        free(g->ranks[r].arena_alloc);
        g->ranks[r].arena_alloc = NULL;
    }
}

/* -- scheduler ------------------------------------------------------------------------------- */

static atomic_int scheduler_running;
static uint64_t scheduler_seed = 1;
static int scheduler_spin;

static void *scheduler_main(void *arg) {
    (void)arg;
    fv_progress(0, scheduler_seed);
    while (atomic_load(&scheduler_running)) {
        if (fv_progress(32, 0) == 0) {
            if (scheduler_spin) {
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
static void scheduler_start(uint64_t seed, int spin) {
    scheduler_seed = seed;
    scheduler_spin = spin;
    atomic_store(&scheduler_running, 1);
    pthread_create(&scheduler_thread, NULL, scheduler_main, NULL);
}
static void scheduler_stop(void) {
    atomic_store(&scheduler_running, 0);
    pthread_join(scheduler_thread, NULL);
}

/* -- kernel emulation: one sender and one receiver thread per channel ----------------------------- */

typedef struct {
    group_t *g;
    int src, dst;
    int messages;
    uint32_t base;
    uint64_t seed;
    uint64_t max_bytes;       /* largest message */
    uint64_t receiver_delay_us;   /* receiver: pause per item, then check the slot again (back-pressure) */
    uint64_t mismatch_message;    /* receiver expects other sizes from this message on (0: none) */
    /* results */
    int failed;
    char error[300];
    uint64_t items;
    uint64_t bytes;
    uint64_t slot_waits;      /* sender found no free slot */
} channel_job_t;

/* Message `index` of a channel: its size from the channel's seed (multiples of 16 and odd tails, empty
 * messages, single and multi-item). */
static uint64_t message_size(const channel_job_t *job, uint32_t index) {
    uint64_t state = job->seed ^ ((uint64_t)index * 0xA24BAED4963EE407ull);
    uint64_t r = rng_next(&state);
    uint64_t choice = r % 16u;
    if (choice == 0) return 0;
    if (choice < 4) return 1 + rng_next(&state) % 64u;
    if (choice < 8) return (1 + rng_next(&state) % (job->g->slot_bytes / 16u)) * 16u;
    return 1 + rng_next(&state) % job->max_bytes;
}

static uint32_t item_count(uint64_t nbytes, uint64_t slot_bytes) {
    uint64_t padded = (nbytes + 15u) / 16u * 16u;
    uint64_t count = (padded + slot_bytes - 1u) / slot_bytes;
    return count == 0 ? 1u : (uint32_t)count;
}

static uint32_t header_of(uint64_t nbytes, uint64_t slot_bytes, uint32_t index) {
    uint64_t padded = (nbytes + 15u) / 16u * 16u;
    uint64_t bytes = padded - (uint64_t)index * slot_bytes;
    if (bytes > slot_bytes) bytes = slot_bytes;
    uint32_t word = (uint32_t)bytes;
    if (index == item_count(nbytes, slot_bytes) - 1u) word |= LAST | (uint32_t)(nbytes & 15u);
    return word;
}

static int poisoned(group_t *g, int r) { return load_acquire(control_of(g, r, C_POISON)) != 0; }

/* A kernel's failure record: the error words, then the tag, then the poison word. */
static void kernel_fail(group_t *g, int r, int peer, int lane, int kind, uint32_t tag, uint32_t expected,
                        uint32_t got) {
    *control_of(g, r, C_ERROR_PEER) = (uint32_t)peer;
    *control_of(g, r, C_ERROR_LANE) = (uint32_t)lane;
    *control_of(g, r, C_ERROR_KIND) = (uint32_t)kind;
    *control_of(g, r, C_ERROR_EXPECTED) = expected;
    *control_of(g, r, C_ERROR_GOT) = got;
    store_release(control_of(g, r, C_ERROR_TAG), tag);
    store_release(control_of(g, r, C_POISON), 1u);
}

static void *sender_main(void *arg) {
    channel_job_t *job = (channel_job_t *)arg;
    group_t *g = job->g;
    uint32_t item = job->base;
    uint32_t slots = (uint32_t)g->slots;
    for (int message = 0; message < job->messages && !job->failed; message++) {
        uint64_t nbytes = message_size(job, (uint32_t)message);
        uint64_t key = message_key(job->src, job->dst, (uint32_t)message);
        uint32_t count = item_count(nbytes, g->slot_bytes);
        for (uint32_t i = 0; i < count; i++, item++) {
            uint32_t m = item % slots, tag = item + 1u;
            volatile uint32_t *sent = word_of(g, job->src, job->dst, L_SENT, 0);
            uint64_t started = now_ns();
            int waited = 0;
            while ((int32_t)(load_acquire(sent) - (tag - slots)) < 0) {
                waited = 1;
                if (poisoned(g, job->src)) {
                    job->failed = 1;
                    snprintf(job->error, sizeof(job->error), "rank %d poisoned while sending item %u", job->src, item);
                    return NULL;
                }
                if (now_ns() - started > WAIT_LIMIT_NS) {
                    kernel_fail(g, job->src, job->dst, 255, KIND_SLOT, tag, 0, 0);
                    job->failed = 1;
                    snprintf(job->error, sizeof(job->error), "send slot %u toward rank %d never freed (item %u)", m,
                             job->dst, item);
                    return NULL;
                }
                sched_yield();
            }
            job->slot_waits += (uint64_t)waited;
            uint32_t header = header_of(nbytes, g->slot_bytes, i);
            uint64_t bytes = header & ~(LAST | 15u);
            uint8_t *slot = block_of(g, job->src, job->dst) + g->layout[L_SEND] + (uint64_t)m * g->slot_bytes;
            for (uint64_t b = 0; b < bytes; b++) {
                uint64_t offset = (uint64_t)i * g->slot_bytes + b;
                slot[b] = offset < nbytes ? pattern_byte(key, offset) : 0;
            }
            *word_of(g, job->src, job->dst, L_DESC, m) = header;
            store_release(word_of(g, job->src, job->dst, L_READY, m), tag);
            job->items++;
            job->bytes += bytes;
        }
    }
    return NULL;
}

static int slot_matches(const uint8_t *slot, uint64_t bytes, uint64_t nbytes, uint64_t first, uint64_t key) {
    for (uint64_t b = 0; b < bytes; b++) {
        uint64_t offset = first + b;
        uint8_t want = offset < nbytes ? pattern_byte(key, offset) : 0;
        if (slot[b] != want) return 0;
    }
    return 1;
}

static void *receiver_main(void *arg) {
    channel_job_t *job = (channel_job_t *)arg;
    group_t *g = job->g;
    uint32_t item = job->base;
    uint32_t slots = (uint32_t)g->slots;
    for (int message = 0; message < job->messages; message++) {
        uint64_t nbytes = message_size(job, (uint32_t)message);
        if (job->mismatch_message != 0 && (uint64_t)message >= job->mismatch_message) nbytes += 32u;
        uint64_t key = message_key(job->src, job->dst, (uint32_t)message);
        uint32_t count = item_count(nbytes, g->slot_bytes);
        for (uint32_t i = 0; i < count; i++, item++) {
            uint32_t m = item % slots, tag = item + 1u;
            for (int lane = 0; lane < g->lanes; lane++) {
                volatile uint32_t *flag = flag_of(g, job->dst, job->src, m, lane);
                uint64_t started = now_ns();
                while (load_acquire(flag) != tag) {
                    if (poisoned(g, job->dst)) {
                        job->failed = 1;
                        snprintf(job->error, sizeof(job->error), "rank %d poisoned while receiving item %u", job->dst,
                                 item);
                        return NULL;
                    }
                    if (now_ns() - started > WAIT_LIMIT_NS) {
                        kernel_fail(g, job->dst, job->src, lane, KIND_FLAG, tag, 0, 0);
                        job->failed = 1;
                        snprintf(job->error, sizeof(job->error), "item %u from rank %d lane %d never arrived", item,
                                 job->src, lane);
                        return NULL;
                    }
                    sched_yield();
                }
            }
            uint32_t expected = header_of(nbytes, g->slot_bytes, i);
            uint32_t got = load_acquire(flag_of(g, job->dst, job->src, m, 0) + 1);
            if (got != expected) {
                kernel_fail(g, job->dst, job->src, 255, KIND_SIZE, tag, expected, got);
                job->failed = 1;
                snprintf(job->error, sizeof(job->error), "item %u from rank %d: header 0x%08x, expected 0x%08x", item,
                         job->src, got, expected);
                return NULL;
            }
            uint64_t bytes = expected & ~(LAST | 15u);
            const uint8_t *slot = block_of(g, job->dst, job->src) + g->layout[L_RECV] + (uint64_t)m * g->slot_bytes;
            if (!slot_matches(slot, bytes, nbytes, (uint64_t)i * g->slot_bytes, key)) {
                job->failed = 1;
                snprintf(job->error, sizeof(job->error), "item %u of message %d from rank %d to rank %d: bytes differ",
                         item, message, job->src, job->dst);
                return NULL;
            }
            if (job->receiver_delay_us != 0) {
                sleep_us(job->receiver_delay_us);
                if (!slot_matches(slot, bytes, nbytes, (uint64_t)i * g->slot_bytes, key)) {
                    job->failed = 1;
                    snprintf(job->error, sizeof(job->error), "item %u from rank %d was overwritten before the receiver "
                             "consumed it", item, job->src);
                    return NULL;
                }
            }
            store_release(word_of(g, job->dst, job->src, L_CONSUMED, m), tag);
            job->items++;
        }
    }
    return NULL;
}

/* -- cases --------------------------------------------------------------------------------- */

static const char *const *selected;
static int n_selected;
static int cases, failures;

static int wanted(const char *name) {
    if (n_selected == 0) return 1;
    for (int i = 0; i < n_selected; i++) {
        if (strncmp(name, selected[i], strlen(selected[i])) == 0) return 1;
    }
    return 0;
}

static void report(const char *name, int ok, const char *detail) {
    cases++;
    if (!ok) failures++;
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, detail[0] ? ": " : "", detail);
}

static const char *kind_text(int kind) { return kind == KIND_PATH ? "path" : "cycle"; }

typedef struct {
    int messages;
    uint64_t max_bytes;
    uint32_t base;
    uint64_t receiver_delay_us;
    int only_src, only_dst;       /* -1: every channel */
} traffic_t;

/* Run the case's traffic over every channel of the group at once; checks every byte and header. */
static int run_traffic(group_t *g, const traffic_t *t, uint64_t seed, char *detail, size_t detail_len,
                       uint64_t *items_out, uint64_t *slot_waits_out) {
    channel_job_t jobs[MAX_WORLD * MAX_WORLD];
    pthread_t senders[MAX_WORLD * MAX_WORLD], receivers[MAX_WORLD * MAX_WORLD];
    int n = 0;
    for (int a = 0; a < g->world; a++) {
        for (int b = 0; b < g->world; b++) {
            if (!g->channels[a][b]) continue;
            if (t->only_src >= 0 && (a != t->only_src || b != t->only_dst)) continue;
            channel_job_t *job = &jobs[n++];
            memset(job, 0, sizeof(*job));
            job->g = g;
            job->src = a;
            job->dst = b;
            job->messages = t->messages;
            job->base = t->base;
            job->seed = seed ^ ((uint64_t)a * 977u) ^ ((uint64_t)b * 131071u);
            job->max_bytes = t->max_bytes;
            job->receiver_delay_us = t->receiver_delay_us;
        }
    }
    for (int i = 0; i < n; i++) {
        pthread_create(&receivers[i], NULL, receiver_main, &jobs[i]);
        pthread_create(&senders[i], NULL, sender_main, &jobs[i]);
    }
    for (int i = 0; i < n; i++) {
        pthread_join(senders[i], NULL);
        pthread_join(receivers[i], NULL);
    }
    uint64_t items = 0, waits = 0;
    for (int i = 0; i < n; i++) {
        if (jobs[i].failed) {
            snprintf(detail, detail_len, "%d->%d: %s", jobs[i].src, jobs[i].dst, jobs[i].error);
            return -1;
        }
        items += jobs[i].items;
        waits += jobs[i].slot_waits;
    }
    for (int r = 0; r < g->world; r++) {
        if (p2p_failed(g->ranks[r].ctx)) {
            snprintf(detail, detail_len, "rank %d native failure: %s", r, p2p_error(g->ranks[r].ctx));
            return -1;
        }
    }
    if (items_out != NULL) *items_out = items;
    if (slot_waits_out != NULL) *slot_waits_out = waits;
    return 0;
}

/* Largest bytes in flight of every relayed queue pair, against its window (inline writes add 4 bytes each). */
static int windows_held(group_t *g, char *detail, size_t detail_len, uint64_t *largest) {
    *largest = 0;
    for (int r = 0; r < g->world; r++) {
        rank_t *k = &g->ranks[r];
        for (int p = 0; p < g->world; p++) {
            if (!g->channels[r][p] || !relayed(g, r, p)) continue;
            for (int l = 0; l < g->lanes; l++) {
                uint32_t qp = p2p_test_qp_num(k->ctx, k->lane_devices[p * g->lanes + l], p);
                uint64_t inflight = fv_qp_max_inflight(qp);
                if (inflight > *largest) *largest = inflight;
                if (inflight > (uint64_t)g->window + 64u) {
                    snprintf(detail, detail_len, "rank %d lane %d toward rank %d had %llu bytes in flight, window %u",
                             r, l, p, (unsigned long long)inflight, g->window);
                    return -1;
                }
            }
        }
    }
    return 0;
}

static void case_traffic(const char *label, int world, int kind, int lanes, int slots, uint64_t slot_bytes,
                         int messages, uint64_t max_bytes, int adjacent) {
    char name[128];
    snprintf(name, sizeof(name), "%s/%s%d/lanes%d", label, kind_text(kind), world, lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    group_t g;
    char detail[600] = "";
    scheduler_start(0x51u + (uint64_t)cases, 0);
    int ok = create_group(&g, world, lanes, kind, 0, slots, slot_bytes, 0, 0, adjacent, 0) == 0;
    if (!ok) snprintf(detail, sizeof(detail), "%s", g.error);
    if (ok) {
        traffic_t t = {messages, max_bytes, 0, 0, -1, -1};
        uint64_t items = 0;
        ok = run_traffic(&g, &t, 0x2026u + (uint64_t)world, detail, sizeof(detail), &items, NULL) == 0;
        if (ok) snprintf(detail, sizeof(detail), "%llu items, every byte exact", (unsigned long long)items);
    }
    destroy_group(&g);
    scheduler_stop();
    report(name, ok, detail);
}

static void case_windows(int world, int kind, int lanes, uint32_t window, uint64_t latency_ns, uint64_t relay_ns,
                         uint64_t ack_delay_ns, uint64_t rate) {
    char name[128];
    snprintf(name, sizeof(name), "relayed-windows/%s%d/lanes%d", kind_text(kind), world, lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    fv_set_latency(latency_ns, relay_ns);
    fv_set_ack_delay(ack_delay_ns);
    fv_set_rate(rate);
    group_t g;
    char detail[600] = "";
    scheduler_start(0x51u + (uint64_t)cases, 1);
    int ok = create_group(&g, world, lanes, kind, 0, 4, 65536, window, 16384, 0, 0) == 0;
    if (!ok) snprintf(detail, sizeof(detail), "%s", g.error);
    if (ok) {
        traffic_t t = {60, 3u * 65536u, 0, 0, -1, -1};
        uint64_t items = 0, largest = 0;
        ok = run_traffic(&g, &t, 0x77u, detail, sizeof(detail), &items, NULL) == 0;
        if (ok) ok = windows_held(&g, detail, sizeof(detail), &largest) == 0;
        uint64_t proven = 0, waits = 0;
        for (int r = 0; r < world; r++) {
            proven += p2p_stat(g.ranks[r].ctx, 9);
            waits += p2p_stat(g.ranks[r].ctx, 5);
        }
        if (ok && proven == 0) {
            ok = 0;
            snprintf(detail, sizeof(detail), "no credit proved a windowed write delivered");
        }
        if (ok) {
            snprintf(detail, sizeof(detail), "%llu items exact, largest in flight %llu bytes (window %u), %llu bytes "
                     "proven by credits, %llu window waits", (unsigned long long)items, (unsigned long long)largest,
                     window, (unsigned long long)proven, (unsigned long long)waits);
        }
    }
    destroy_group(&g);
    scheduler_stop();
    fv_set_latency(0, 0);
    fv_set_ack_delay(0);
    fv_set_rate(0);
    report(name, ok, detail);
}

static void case_wrap(int world, int kind, int lanes) {
    char name[128];
    snprintf(name, sizeof(name), "wrap/%s%d/lanes%d", kind_text(kind), world, lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    group_t g;
    char detail[600] = "";
    uint32_t base = 0xFFFFFFE0u;
    scheduler_start(0x51u + (uint64_t)cases, 0);
    int ok = create_group(&g, world, lanes, kind, 0, 8, 8192, 0, 0, 0, base) == 0;
    if (!ok) snprintf(detail, sizeof(detail), "%s", g.error);
    if (ok) {
        traffic_t t = {40, 3u * 8192u, base, 0, -1, -1};
        uint64_t items = 0;
        ok = run_traffic(&g, &t, 0x17u, detail, sizeof(detail), &items, NULL) == 0;
        uint32_t sent = *word_of(&g, 0, 1, L_SENT, 0);
        if (ok && (int32_t)(sent - base) <= 32) {
            ok = 0;
            snprintf(detail, sizeof(detail), "the item counter of 0->1 did not pass the 32-bit wrap (sent word %u)", sent);
        }
        if (ok) snprintf(detail, sizeof(detail), "%llu items from item 0x%08x across the wrap, every byte exact",
                         (unsigned long long)items, base);
    }
    destroy_group(&g);
    scheduler_stop();
    report(name, ok, detail);
}

static void case_back_pressure(int world, int kind, int lanes) {
    char name[128];
    snprintf(name, sizeof(name), "back-pressure/%s%d/lanes%d", kind_text(kind), world, lanes);
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    group_t g;
    char detail[600] = "";
    uint32_t window = 32768;
    scheduler_start(0x51u + (uint64_t)cases, 0);
    int ok = create_group(&g, world, lanes, kind, 0, 4, 16384, window, 8192, 0, 0) == 0;
    if (!ok) snprintf(detail, sizeof(detail), "%s", g.error);
    if (ok) {
        /* One relayed channel: rank 0 to the farthest rank; the receiver pauses 300 us per item and checks
         * that the slot still holds the item before it releases it. */
        traffic_t t = {30, 4u * 16384u, 0, 300, 0, kind == KIND_PATH ? world - 1 : world / 2};
        uint64_t items = 0, waits = 0, largest = 0;
        ok = run_traffic(&g, &t, 0x31u, detail, sizeof(detail), &items, &waits) == 0;
        if (ok) ok = windows_held(&g, detail, sizeof(detail), &largest) == 0;
        if (ok && waits == 0) {
            ok = 0;
            snprintf(detail, sizeof(detail), "the sender never waited for a slot; the receiver did not hold it back");
        }
        if (ok) snprintf(detail, sizeof(detail), "%llu items exact, the sender waited for a slot %llu times, no item "
                         "overwritten before release, largest in flight %llu bytes (window %u)",
                         (unsigned long long)items, (unsigned long long)waits, (unsigned long long)largest, window);
    }
    destroy_group(&g);
    scheduler_stop();
    report(name, ok, detail);
}

static int wait_failed(group_t *g, int r, uint64_t timeout_ms) {
    uint64_t deadline = now_ns() + timeout_ms * 1000000ull;
    while (now_ns() < deadline) {
        if (p2p_failed(g->ranks[r].ctx)) return 1;
        sleep_us(1000);
    }
    return 0;
}

static void case_size_mismatch(void) {
    const char *name = "size-mismatch/path4/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    group_t g;
    char detail[600] = "";
    scheduler_start(0x51u + (uint64_t)cases, 0);
    int ok = create_group(&g, 4, 2, KIND_PATH, 0, 4, 16384, 0, 0, 0, 0) == 0;
    if (!ok) snprintf(detail, sizeof(detail), "%s", g.error);
    if (ok) {
        channel_job_t job;
        memset(&job, 0, sizeof(job));
        job.g = &g;
        job.src = 0;
        job.dst = 3;
        job.messages = 8;
        job.seed = 0x99u;
        job.max_bytes = 40000;
        job.mismatch_message = 3;
        pthread_t sender, receiver;
        pthread_create(&receiver, NULL, receiver_main, &job);
        pthread_create(&sender, NULL, sender_main, &job);
        pthread_join(receiver, NULL);
        int receiver_failed = wait_failed(&g, 3, 2000);
        int sender_failed = wait_failed(&g, 0, 2000);
        pthread_join(sender, NULL);
        uint32_t kind = *control_of(&g, 3, C_ERROR_KIND);
        uint32_t poison0 = *control_of(&g, 0, C_POISON);
        ok = job.failed && kind == KIND_SIZE && receiver_failed && sender_failed && poison0 != 0 &&
             strstr(p2p_error(g.ranks[0].ctx), "on rank 3") != NULL;
        snprintf(detail, sizeof(detail), "receiver: %s; native rank 3: %s; native rank 0: %s",
                 job.error, p2p_error(g.ranks[3].ctx), p2p_error(g.ranks[0].ctx));
    }
    destroy_group(&g);
    scheduler_stop();
    report(name, ok, detail);
}

static void case_injected(void) {
    const char *name = "injected-failure/cycle4/lanes2";
    if (!wanted(name)) return;
    fv_reset();
    fv_set_recording(0);
    group_t g;
    char detail[600] = "";
    scheduler_start(0x51u + (uint64_t)cases, 0);
    int ok = create_group(&g, 4, 2, KIND_CYCLE, 0, 4, 16384, 0, 0, 0, 0) == 0;
    if (!ok) snprintf(detail, sizeof(detail), "%s", g.error);
    if (ok) {
        /* The flag write of item 5 on lane 1 from rank 1 to rank 2 fails. */
        rank_t *k = &g.ranks[1];
        uint32_t qp = p2p_test_qp_num(k->ctx, k->lane_devices[2 * g.lanes + 1], 2);
        fv_inject_failure(qp, 6u);
        channel_job_t job;
        memset(&job, 0, sizeof(job));
        job.g = &g;
        job.src = 1;
        job.dst = 2;
        job.messages = 40;
        job.seed = 0x15u;
        job.max_bytes = 20000;
        pthread_t sender, receiver;
        pthread_create(&receiver, NULL, receiver_main, &job);
        pthread_create(&sender, NULL, sender_main, &job);
        int one = wait_failed(&g, 1, 3000);
        int two = wait_failed(&g, 2, 3000);
        pthread_join(sender, NULL);
        pthread_join(receiver, NULL);
        ok = one && two && strstr(p2p_error(g.ranks[1].ctx), "flag write of item 5") != NULL &&
             strstr(p2p_error(g.ranks[2].ctx), "on rank 1") != NULL;
        snprintf(detail, sizeof(detail), "rank 1: %s; rank 2: %s", p2p_error(g.ranks[1].ctx), p2p_error(g.ranks[2].ctx));
    }
    destroy_group(&g);
    scheduler_stop();
    report(name, ok, detail);
}

static void case_refusals(void) {
    const char *name = "refusals";
    if (!wanted(name)) return;
    char detail[600] = "";
    int ok = 1;
    uint64_t layout[11];
    if (p2p_layout(4, 2, 6, 65536, layout) == 0) {
        ok = 0;
        snprintf(detail, sizeof(detail), "6 slots (not a power of two) accepted");
    }
    if (ok && p2p_layout(4, 2, 64, 65536, layout) == 0) {
        ok = 0;
        snprintf(detail, sizeof(detail), "64 slots accepted");
    }
    if (ok && p2p_layout(4, 3, 8, 65536, layout) == 0) {
        ok = 0;
        snprintf(detail, sizeof(detail), "3 lanes accepted");
    }
    if (ok && p2p_layout(4, 2, 8, 65537, layout) == 0) {
        ok = 0;
        snprintf(detail, sizeof(detail), "unaligned slot accepted");
    }
    if (ok) {
        /* Channels that one rank claims and its peer does not: connect refuses on every rank. */
        fv_reset();
        fv_set_recording(0);
        group_t g;
        memset(&g, 0, sizeof(g));
        g.world = 3;
        g.lanes = 1;
        g.kind = KIND_PATH;
        g.slots = 4;
        g.slot_bytes = 8192;
        set_channels(&g, 0);
        p2p_layout(3, 1, 4, 8192, g.layout);
        build_fabric(&g);
        plan_routes(&g);
        uint64_t record = p2p_blob_bytes();
        uint8_t *blobs = (uint8_t *)calloc(3, (size_t)record);
        for (int r = 0; r < 3 && ok; r++) {
            rank_t *k = &g.ranks[r];
            k->arena_bytes = g.layout[L_TOTAL];
            k->arena_alloc = (uint8_t *)calloc(1, (size_t)k->arena_bytes + 4096);
            k->arena = k->arena_alloc + ((4096 - ((uintptr_t)k->arena_alloc % 4096)) % 4096);
            const char *names[4];
            int gid_indices[4] = {3, 3, 3, 3};
            int channels[3] = {g.channels[r][0], g.channels[r][1], g.channels[r][2]};
            int devices[6];
            memcpy(devices, k->lane_devices, sizeof(devices));
            if (r == 2) {   /* rank 2 claims no channel with rank 0 */
                channels[0] = 0;
                devices[0] = -1;
            }
            for (int i = 0; i < k->n_open; i++) names[i] = k->names[k->open_roles[i]];
            char err[400];
            k->ctx = p2p_create(3, r, names, k->n_open, devices, 1, gid_indices, 0, channels, k->arena, k->arena_bytes,
                                4, 8192, err, sizeof(err));
            if (k->ctx == NULL) {
                ok = 0;
                snprintf(detail, sizeof(detail), "rank %d create: %s", r, err);
            } else {
                p2p_local_blob(k->ctx, blobs + (size_t)r * record, record);
            }
        }
        if (ok) {
            int refused = 0;
            for (int r = 0; r < 3; r++) refused += p2p_connect(g.ranks[r].ctx, blobs, record * 3u) != 0;
            const char *why = p2p_error(g.ranks[0].ctx);
            ok = refused == 3 && strstr(why, "channel") != NULL;
            snprintf(detail, sizeof(detail), "asymmetric channels refused on %d of 3 ranks: %s", refused, why);
        }
        if (ok) {
            uint32_t windows[3] = {0, 0, 1000};
            int refused = p2p_set_windows(g.ranks[0].ctx, windows, 4096) != 0;
            ok = refused && strstr(p2p_error(g.ranks[0].ctx), "must hold") != NULL;
            if (!ok) snprintf(detail, sizeof(detail), "a window smaller than its chunk was accepted: %s",
                              p2p_error(g.ranks[0].ctx));
        }
        free(blobs);
        destroy_group(&g);
    }
    report(name, ok, detail);
}

int main(int argc, char **argv) {
    selected = (const char *const *)(argv + 1);
    n_selected = argc - 1;
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (p2p_abi_version() != 1) {
        printf("FAIL abi: version %d\n", p2p_abi_version());
        return 1;
    }
    static const int worlds[] = {2, 3, 4, 8};
    for (int w = 0; w < 4; w++) {
        for (int kind = 0; kind < 2; kind++) {
            for (int lanes = 1; lanes <= 2; lanes++) {
                case_traffic("pairs", worlds[w], kind, lanes, 4, 8192, worlds[w] <= 4 ? 150 : 40, 5u * 8192u, 0);
            }
        }
    }
    case_traffic("slots2", 4, KIND_PATH, 2, 2, 4096, 30, 3u * 4096u, 0);
    case_traffic("slots32", 3, KIND_CYCLE, 2, 32, 4096, 30, 40u * 4096u, 0);
    case_traffic("adjacent-only", 8, KIND_CYCLE, 2, 8, 8192, 30, 4u * 8192u, 1);
    case_traffic("two-apart", 8, KIND_CYCLE, 2, 8, 8192, 20, 4u * 8192u, 2);
    case_windows(8, KIND_CYCLE, 2, 32768, 20000, 10000, 150000, 12000);
    case_windows(4, KIND_PATH, 2, 32768, 20000, 10000, 150000, 12000);
    case_windows(4, KIND_PATH, 1, 65536, 20000, 10000, 150000, 12000);
    case_wrap(4, KIND_CYCLE, 2);
    case_wrap(3, KIND_PATH, 1);
    case_back_pressure(4, KIND_PATH, 2);
    case_back_pressure(8, KIND_CYCLE, 2);
    case_size_mismatch();
    case_injected();
    case_refusals();
    printf("%d cases, %d failed\n", cases, failures);
    return failures ? 1 : 0;
}
