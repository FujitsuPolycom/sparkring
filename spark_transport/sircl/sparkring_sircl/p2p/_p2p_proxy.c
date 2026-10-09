/*
 * SIRCL point-to-point channels: verbs setup and the native progress thread.
 *
 * One context serves one rank of one group of 2 to 16 ranks. Every ordered
 * pair of ranks whose channel is enabled is a channel of `slots` slots of
 * `slot_bytes` bytes on both ends (p2p/protocol.py describes the words). The
 * rank owns one arena: pinned host memory that the GPU addresses at its host
 * pointer and that every opened RDMA device registers for local and remote
 * writes:
 *
 *   control line        CONTROL_BYTES  wait limit, error record, poison, abort, lane check
 *   block of rank p     block_bytes    (p = 0 .. world - 1; the own block is unused)
 *     recv slots        slots * slot_bytes   items from p, written by p's progress thread
 *     send slots        slots * slot_bytes   items for p, staged by this rank's kernel
 *     flag lines        slots * lanes * 128  tag of the item in a receive slot, per lane; lane 0's
 *                                            line holds the item's header at byte 4
 *     desc line         128                  header of each staged send slot (kernel)
 *     ready line        128                  tag of each staged send slot (kernel)
 *     consumed line     128                  tag of each receive slot the kernel finished
 *     sent line         128                  word 0: items toward p whose writes completed (this thread)
 *     credit line       128                  word 0: items from this rank that p released (p's thread)
 *
 * Item g of a channel (counted from 0 over the session, 32 bits, wrapping)
 * uses slot g % slots on both ends and carries tag g + 1. Outbound, this
 * thread posts item g toward p when the kernel staged it (ready word of its
 * slot holds the tag) and p released item g - slots (credit word >= g + 1 -
 * slots): every lane writes its stripe of the item (packs split over the
 * lanes as in the collective layer), lane 0 then the 4-byte header, and every
 * lane its 4-byte flag, all on the lane's queue pair, so a flag proves the
 * lane's bytes landed. When every lane's flag write of item g completed, the
 * sent word moves past g and the kernel may stage item g + slots in the slot.
 * Inbound, the thread advances over the items the kernel consumed and writes
 * their count into the sender's credit word.
 *
 * A lane through relays has a forward window (p2p_set_windows): its stripe
 * goes out as signaled writes of at most the chunk size, each while the bytes
 * posted and not yet delivered on the lane's queue pair stay within the
 * window. A write counts as delivered at its completion or as soon as the
 * receiver's credit passes its item (the receiver released the item, so every
 * byte of it landed).
 *
 * Failure: a kernel that times out or meets a header it did not expect
 * writes the error words and the poison word of the control line. This
 * thread then writes an abort notice (its rank + 1) into every peer's abort
 * word and stops; a peer that finds its abort word set poisons its own
 * context and stops. A failed completion does the same.
 *
 * Plain C over libibverbs and POSIX threads, built by sparkring_sircl.p2p's
 * build module with the host compiler. The CPU simulator builds it against the
 * test-only verbs stand-in in sparkring_sircl/testing/fake_verbs.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>

/* Native ABI of the point-to-point library; changes with the record, the layout, the control words
 * or any exported signature. */
#define P2P_ABI_VERSION 1
/* Local features: what this library offers its own process's binding, apart from the wire contract of
 * P2P_ABI_VERSION, which peers compare in their connection records. */
#define P2P_FEATURE_DESTROY_COUNT 1u  /* p2p_destroy returns the number of verbs calls that failed */
#define P2P_LOCAL_FEATURES P2P_FEATURE_DESTROY_COUNT
/* First word of every connection record ("SP2P" in byte order). */
#define P2P_RECORD_MAGIC 0x50325053u
#define P2P_MAX_PEERS 16
#define P2P_MAX_DEVICES 4
#define P2P_MAX_LANES 2
#define P2P_MIN_SLOTS 2
#define P2P_MAX_SLOTS 32
#define P2P_LINE 128u
#define P2P_CONTROL_BYTES 4096u
#define P2P_ALIGN 4096u
#define P2P_MAX_SLOT_BYTES (1ull << 29)
#define P2P_PORT 1
#define P2P_SEND_DEPTH 256
/* Signaled work requests in flight per queue pair (every work request is signaled). */
#define P2P_INFLIGHT (P2P_SEND_DEPTH / 2)
#define P2P_CQ_DEPTH (P2P_SEND_DEPTH * P2P_MAX_PEERS)
#define P2P_MIN_PIECE 4096u
#ifndef P2P_IDLE_SPINS
#define P2P_IDLE_SPINS 20000000ull
#endif
#define P2P_NAP_NS 20000L
#define P2P_NO_DEVICE 0xFFu
#define P2P_LAYOUT_WORDS 11
#define P2P_LAST (1u << 31)
#define P2P_BYTES_MASK (((1u << 30) - 1u) & ~15u)

enum { C_WAIT_LIMIT_US = 0, C_ERROR_TAG = 1, C_ERROR_PEER = 2, C_ERROR_LANE = 3, C_ERROR_KIND = 4,
       C_ERROR_EXPECTED = 5, C_ERROR_GOT = 6, C_POISON = 7, C_ABORT = 8, C_LANE_CHECK = 31 };
enum { L_CONTROL = 0, L_BLOCK = 1, L_RECV = 2, L_SEND = 3, L_FLAG = 4, L_DESC = 5, L_READY = 6, L_CONSUMED = 7,
       L_SENT = 8, L_CREDIT = 9, L_TOTAL = 10 };
/* Work-request ids: peer in bits 0-7, lane in 8-15, kind in 16-23, item in 32-63. */
enum { WR_DATA = 1, WR_HEADER = 2, WR_FLAG = 3, WR_CREDIT = 4, WR_CHECK = 5, WR_ABORT = 6 };

typedef struct {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t world;
    uint32_t rank;
    uint32_t lane_count;
    uint32_t n_devices;
    uint32_t slots;
    uint32_t channels;        /* bit p: this rank has a channel with rank p */
    uint64_t slot_bytes;
    uint64_t region_addr;
    uint32_t rkey[P2P_MAX_DEVICES];
    uint32_t mtu[P2P_MAX_DEVICES];
    uint16_t lid[P2P_MAX_DEVICES];
    uint8_t gid[P2P_MAX_DEVICES][16];
    uint32_t qp_num[P2P_MAX_DEVICES][P2P_MAX_PEERS];
    uint8_t lane_device[P2P_MAX_PEERS][P2P_MAX_LANES];
} p2p_record_t;

/* One signaled work request in flight on a queue pair: the bytes it holds in the lane's window
 * (data writes only), the item it belongs to and whether a credit already proved it delivered. */
typedef struct {
    uint32_t bytes;
    uint32_t item;
    uint8_t kind;
    uint8_t proven;
} p2p_ack_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[P2P_MAX_PEERS];
    p2p_ack_t ack[P2P_MAX_PEERS][P2P_SEND_DEPTH];
    uint16_t ack_head[P2P_MAX_PEERS];
    uint16_t ack_count[P2P_MAX_PEERS];
    uint64_t unacked[P2P_MAX_PEERS];   /* window bytes posted and not yet delivered */
    _Atomic uint64_t writes_completed;
    _Atomic uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
    int gid_index;
    char name[64];
} p2p_dev_t;

/* The channel toward one peer (outbound) and the channel from it (inbound). */
typedef struct {
    int enabled;
    /* outbound */
    uint32_t next_post;                    /* items posted on every lane */
    uint32_t lane_item[P2P_MAX_LANES];     /* item a lane is posting */
    uint32_t lane_bytes[P2P_MAX_LANES];    /* bytes of that item's stripe the lane posted */
    int lane_header[P2P_MAX_LANES];        /* lane 0: the item's header is posted */
    uint32_t lane_done[P2P_MAX_LANES];     /* items whose flag write completed, per lane */
    uint32_t done;                         /* items complete on every lane (the sent word) */
    uint32_t credit_seen;                  /* the newest credit read from the peer */
    /* inbound */
    uint32_t released;                     /* items the kernel consumed, contiguous */
    uint32_t credited;                     /* credit last written to the peer */
    uint64_t wait_started;                 /* a windowed lane waits for room since (0: not waiting) */
    _Atomic uint64_t items_posted, bytes_posted, items_released, credits_sent;
} p2p_channel_t;

typedef struct p2p_ctx {
    int world;
    int rank;
    int n_dev;
    int lane_count;
    int traffic_class;
    int slots;
    uint64_t slot_bytes;
    uint64_t layout[P2P_LAYOUT_WORDS];
    p2p_dev_t dev[P2P_MAX_DEVICES];
    int lane_device[P2P_MAX_PEERS][P2P_MAX_LANES];
    uint32_t peer_rkey[P2P_MAX_PEERS][P2P_MAX_LANES];
    uint8_t peer_gid[P2P_MAX_PEERS][P2P_MAX_LANES][16];
    uint64_t peer_addr[P2P_MAX_PEERS];
    p2p_channel_t ch[P2P_MAX_PEERS];
    uint32_t window[P2P_MAX_PEERS][P2P_MAX_LANES];   /* bytes in flight per relayed lane; 0: none */
    uint32_t chunk;
    uint8_t *region;
    uint64_t region_bytes;
    int connected;
    int started;
    int aborted;               /* the abort notices were posted */
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    atomic_int last_cpu;
    _Atomic uint64_t cpu_migrations;
    _Atomic uint64_t window_waits, window_wait_ns, window_wait_max_ns, window_max_unacked, proven_bytes;
    _Atomic uint64_t abort_from;   /* rank + 1 of a peer whose abort notice arrived */
    char err[512];
} p2p_ctx_t;

int p2p_destroy(p2p_ctx_t *c);

#define FAIL(c, ...) snprintf((c)->err, sizeof((c)->err), __VA_ARGS__)

/* -- arithmetic shared with the kernels (p2p/protocol.py) -------------------------- */

static void lane_split(uint32_t packs, int lanes, int lane, uint32_t *first, uint32_t *count) {
    uint32_t base = packs / (uint32_t)lanes;
    uint32_t rest = packs % (uint32_t)lanes;
    *first = (uint32_t)lane * base + ((uint32_t)lane < rest ? (uint32_t)lane : rest);
    *count = base + ((uint32_t)lane < rest ? 1u : 0u);
}

int p2p_abi_version(void) { return P2P_ABI_VERSION; }
unsigned int p2p_local_features(void) { return P2P_LOCAL_FEATURES; }

/* out = {control bytes, block bytes, recv, send, flag, desc, ready, consumed, sent, credit, total} */
int p2p_layout(int world, int lanes, int slots, uint64_t slot_bytes, uint64_t *out) {
    if (out == NULL || world < 2 || world > P2P_MAX_PEERS || lanes < 1 || lanes > P2P_MAX_LANES ||
        slots < P2P_MIN_SLOTS || slots > P2P_MAX_SLOTS || (slots & (slots - 1)) != 0 || slot_bytes == 0 ||
        slot_bytes % P2P_ALIGN != 0 || slot_bytes > P2P_MAX_SLOT_BYTES) {
        return -1;
    }
    uint64_t ring = (uint64_t)slots * slot_bytes;
    out[L_CONTROL] = P2P_CONTROL_BYTES;
    out[L_RECV] = 0;
    out[L_SEND] = ring;
    out[L_FLAG] = 2 * ring;
    out[L_DESC] = out[L_FLAG] + (uint64_t)slots * (uint64_t)lanes * P2P_LINE;
    out[L_READY] = out[L_DESC] + P2P_LINE;
    out[L_CONSUMED] = out[L_READY] + P2P_LINE;
    out[L_SENT] = out[L_CONSUMED] + P2P_LINE;
    out[L_CREDIT] = out[L_SENT] + P2P_LINE;
    out[L_BLOCK] = (out[L_CREDIT] + P2P_LINE + P2P_ALIGN - 1) / P2P_ALIGN * P2P_ALIGN;
    out[L_TOTAL] = P2P_CONTROL_BYTES + (uint64_t)world * out[L_BLOCK];
    return 0;
}

uint64_t p2p_blob_bytes(void) { return sizeof(p2p_record_t); }

/* Host-side stand-ins for the kernels' ordered accesses (the CPU harness and the tests that play the
 * kernel's part): a release store after a full fence, and an acquire load. */
void p2p_store_release_u32(volatile uint32_t *address, uint32_t value) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

uint32_t p2p_load_acquire_u32(const volatile uint32_t *address) {
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

/* -- arena words ------------------------------------------------------------------ */

static uint64_t block_off(const p2p_ctx_t *c, int p) {
    return c->layout[L_CONTROL] + (uint64_t)p * c->layout[L_BLOCK];
}

static volatile uint32_t *control(const p2p_ctx_t *c, int word) {
    return (volatile uint32_t *)(c->region + 4u * (uint64_t)word);
}

static volatile uint32_t *block_word(const p2p_ctx_t *c, int p, int area, uint32_t index) {
    return (volatile uint32_t *)(c->region + block_off(c, p) + c->layout[area] + 4u * (uint64_t)index);
}

/* Offset of slot m's flag line of `lane` in a block (lane 0's line holds the header at byte 4). */
static uint64_t flag_off(const p2p_ctx_t *c, uint32_t m, int lane) {
    return c->layout[L_FLAG] + ((uint64_t)m * (uint64_t)c->lane_count + (uint64_t)lane) * P2P_LINE;
}

/* -- settings read at start ------------------------------------------------------------- */

/* SIRCL_P2P_PROGRESS_CPU: CPU list such as 9 or 5-9,15-19. */
static int parse_cpu_list(const char *text, cpu_set_t *set) {
    CPU_ZERO(set);
    int any = 0;
    const char *p = text;
    while (*p != '\0') {
        char *end = NULL;
        long lo = strtol(p, &end, 10);
        if (end == p || lo < 0 || lo >= CPU_SETSIZE) return -1;
        long hi = lo;
        p = end;
        if (*p == '-') {
            hi = strtol(p + 1, &end, 10);
            if (end == p + 1 || hi < lo || hi >= CPU_SETSIZE) return -1;
            p = end;
        }
        for (long cpu = lo; cpu <= hi; cpu++) CPU_SET((int)cpu, set);
        any = 1;
        if (*p == ',') {
            p++;
        } else if (*p != '\0') {
            return -1;
        }
    }
    return any ? 0 : -1;
}

/* -- setup ---------------------------------------------------------------------------- */

static int open_device(p2p_ctx_t *c, int d, const char *name, int gid_index) {
    p2p_dev_t *dev = &c->dev[d];
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    struct ibv_device *found = NULL;
    snprintf(dev->name, sizeof(dev->name), "%s", name);
    dev->gid_index = gid_index;
    if (list == NULL) {
        FAIL(c, "ibv_get_device_list: %s", strerror(errno));
        return -1;
    }
    for (int i = 0; i < num; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
            found = list[i];
            break;
        }
    }
    if (found == NULL) {
        ibv_free_device_list(list);
        FAIL(c, "RDMA device %s not found", name);
        return -1;
    }
    dev->ctx = ibv_open_device(found);
    ibv_free_device_list(list);
    if (dev->ctx == NULL) {
        FAIL(c, "ibv_open_device(%s): %s", name, strerror(errno));
        return -1;
    }
    struct ibv_port_attr port;
    memset(&port, 0, sizeof(port));
    if (ibv_query_port(dev->ctx, P2P_PORT, &port) != 0) {
        FAIL(c, "ibv_query_port(%s): %s", name, strerror(errno));
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        FAIL(c, "RDMA device %s port %d is not active", name, P2P_PORT);
        return -1;
    }
    dev->lid = port.lid;
    dev->mtu = port.active_mtu;
    if (ibv_query_gid(dev->ctx, P2P_PORT, gid_index, &dev->gid) != 0) {
        FAIL(c, "RDMA device %s has no GID at index %d", name, gid_index);
        return -1;
    }
    static const uint8_t zero[16];
    if (memcmp(dev->gid.raw, zero, 16) == 0) {
        FAIL(c, "RDMA device %s has an empty GID at index %d", name, gid_index);
        return -1;
    }
    dev->pd = ibv_alloc_pd(dev->ctx);
    if (dev->pd == NULL) {
        FAIL(c, "ibv_alloc_pd(%s): %s", name, strerror(errno));
        return -1;
    }
    /* The arena is pinned host memory; GB10 has no GPU-memory registration. */
    dev->mr = ibv_reg_mr(dev->pd, c->region, (size_t)c->region_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (dev->mr == NULL) {
        FAIL(c, "ibv_reg_mr(%s, arena): %s", name, strerror(errno));
        return -1;
    }
    dev->cq = ibv_create_cq(dev->ctx, P2P_CQ_DEPTH, NULL, NULL, 0);
    if (dev->cq == NULL) {
        FAIL(c, "ibv_create_cq(%s): %s", name, strerror(errno));
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        int used = 0;
        for (int l = 0; l < c->lane_count; l++) used |= c->ch[p].enabled && c->lane_device[p][l] == d;
        if (!used) continue;
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = dev->cq;
        attr.recv_cq = dev->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = P2P_SEND_DEPTH;
        attr.cap.max_recv_wr = 1;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.cap.max_inline_data = 16;
        dev->qp[p] = ibv_create_qp(dev->pd, &attr);
        if (dev->qp[p] == NULL) {
            FAIL(c, "ibv_create_qp(%s toward rank %d): %s", name, p, strerror(errno));
            return -1;
        }
        struct ibv_qp_attr init;
        memset(&init, 0, sizeof(init));
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = P2P_PORT;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        int rc = ibv_modify_qp(dev->qp[p], &init,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc != 0) {
            FAIL(c, "ibv_modify_qp(INIT, %s toward rank %d): %s", name, p, strerror(rc));
            return -1;
        }
    }
    return 0;
}

/* Every enabled peer has lane_count lanes on distinct opened devices; the own rank and disabled peers
 * name none. */
static int check_route_table(int world, int rank, int n_dev, const int *lane_devices, int lane_count,
                             const int *channels, char *err, uint64_t err_len) {
    for (int p = 0; p < world; p++) {
        int enabled = p != rank && channels[p] != 0;
        for (int l = 0; l < lane_count; l++) {
            int d = lane_devices[p * lane_count + l];
            if (!enabled) {
                if (d != -1) {
                    snprintf(err, (size_t)err_len, "rank %d names a device for lane %d toward rank %d, which has "
                             "no channel", rank, l, p);
                    return -1;
                }
                continue;
            }
            if (d < 0 || d >= n_dev) {
                snprintf(err, (size_t)err_len, "rank %d lane %d toward rank %d names device index %d outside 0-%d",
                         rank, l, p, d, n_dev - 1);
                return -1;
            }
            for (int k = 0; k < l; k++) {
                if (lane_devices[p * lane_count + k] == d) {
                    snprintf(err, (size_t)err_len, "rank %d uses device index %d for two lanes toward rank %d",
                             rank, d, p);
                    return -1;
                }
            }
        }
    }
    return 0;
}

p2p_ctx_t *p2p_create(int world, int rank, const char *const *device_names, int n_devices,
                      const int *lane_devices, int lane_count, const int *gid_indices, int traffic_class,
                      const int *channels, void *region, uint64_t region_bytes, int slots, uint64_t slot_bytes,
                      char *err, uint64_t err_len) {
    uint64_t layout[P2P_LAYOUT_WORDS];
    if (err == NULL || err_len == 0) return NULL;
    err[0] = '\0';
    if (p2p_layout(world, lane_count, slots, slot_bytes, layout) != 0 || rank < 0 || rank >= world ||
        region == NULL || layout[L_TOTAL] > region_bytes || ((uintptr_t)region % P2P_ALIGN) != 0) {
        snprintf(err, (size_t)err_len,
                 "invalid point-to-point geometry: world %d, rank %d, %d lanes, %d slots of %llu bytes, arena of "
                 "%llu bytes (4096-aligned, at least the layout's total)", world, rank, lane_count, slots,
                 (unsigned long long)slot_bytes, (unsigned long long)region_bytes);
        return NULL;
    }
    if (n_devices < 1 || n_devices > P2P_MAX_DEVICES || device_names == NULL || channels == NULL ||
        lane_devices == NULL || gid_indices == NULL) {
        snprintf(err, (size_t)err_len, "a point-to-point context opens 1 to %d RDMA devices, got %d",
                 P2P_MAX_DEVICES, n_devices);
        return NULL;
    }
    if (traffic_class < 0 || traffic_class > 255) {
        snprintf(err, (size_t)err_len, "traffic class %d is outside 0-255", traffic_class);
        return NULL;
    }
    for (int d = 0; d < n_devices; d++) {
        if (device_names[d] == NULL || device_names[d][0] == '\0' || gid_indices[d] < 0 || gid_indices[d] > 255) {
            snprintf(err, (size_t)err_len, "device %d needs a name and a GID index in 0-255", d);
            return NULL;
        }
    }
    if (check_route_table(world, rank, n_devices, lane_devices, lane_count, channels, err, err_len) != 0) {
        return NULL;
    }
    p2p_ctx_t *c = (p2p_ctx_t *)calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, (size_t)err_len, "out of memory");
        return NULL;
    }
    c->world = world;
    c->rank = rank;
    c->n_dev = n_devices;
    c->lane_count = lane_count;
    c->traffic_class = traffic_class;
    c->slots = slots;
    c->slot_bytes = slot_bytes;
    memcpy(c->layout, layout, sizeof(layout));
    c->region = (uint8_t *)region;
    c->region_bytes = region_bytes;
    c->chunk = 32768u;
    atomic_store(&c->last_cpu, -1);
    for (int p = 0; p < P2P_MAX_PEERS; p++) {
        c->ch[p].enabled = p < world && p != rank && channels[p] != 0;
        for (int l = 0; l < P2P_MAX_LANES; l++) {
            c->lane_device[p][l] = (c->ch[p].enabled && l < lane_count) ? lane_devices[p * lane_count + l] : -1;
        }
    }
    for (int d = 0; d < n_devices; d++) {
        if (open_device(c, d, device_names[d], gid_indices[d]) != 0) {
            snprintf(err, (size_t)err_len, "%s", c->err);
            p2p_destroy(c);
            return NULL;
        }
    }
    return c;
}

int p2p_local_blob(p2p_ctx_t *c, void *out, uint64_t out_len) {
    p2p_record_t r;
    if (c == NULL || out == NULL || out_len < sizeof(r)) return -1;
    memset(&r, 0, sizeof(r));
    r.magic = P2P_RECORD_MAGIC;
    r.abi_version = P2P_ABI_VERSION;
    r.world = (uint32_t)c->world;
    r.rank = (uint32_t)c->rank;
    r.lane_count = (uint32_t)c->lane_count;
    r.n_devices = (uint32_t)c->n_dev;
    r.slots = (uint32_t)c->slots;
    r.slot_bytes = c->slot_bytes;
    r.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int p = 0; p < c->world; p++) {
        if (c->ch[p].enabled) r.channels |= 1u << p;
    }
    for (int d = 0; d < c->n_dev; d++) {
        r.rkey[d] = c->dev[d].mr->rkey;
        r.mtu[d] = (uint32_t)c->dev[d].mtu;
        r.lid[d] = c->dev[d].lid;
        memcpy(r.gid[d], c->dev[d].gid.raw, 16);
        for (int p = 0; p < c->world; p++) {
            r.qp_num[d][p] = c->dev[d].qp[p] != NULL ? c->dev[d].qp[p]->qp_num : 0;
        }
    }
    for (int p = 0; p < P2P_MAX_PEERS; p++) {
        for (int l = 0; l < P2P_MAX_LANES; l++) {
            int d = c->lane_device[p][l];
            r.lane_device[p][l] = d < 0 ? P2P_NO_DEVICE : (uint8_t)d;
        }
    }
    memcpy(out, &r, sizeof(r));
    return 0;
}

/* Every record before any queue pair moves: magic, ABI, geometry, symmetric channels and the lane
 * devices each rank claims toward this rank. */
static int validate_records(p2p_ctx_t *c, const p2p_record_t *all) {
    for (int p = 0; p < c->world; p++) {
        const p2p_record_t *r = &all[p];
        if (r->magic != P2P_RECORD_MAGIC) {
            FAIL(c, "rank %d published a record of another protocol (magic 0x%08x)", p, r->magic);
            return -1;
        }
        if (r->abi_version != P2P_ABI_VERSION || r->world != (uint32_t)c->world || r->rank != (uint32_t)p ||
            r->slot_bytes != c->slot_bytes || r->slots != (uint32_t)c->slots ||
            r->lane_count != (uint32_t)c->lane_count) {
            FAIL(c, "rank %d record differs: ABI %u/%d, world %u/%d, rank %u, slots %u/%d of %llu/%llu bytes, "
                    "lanes %u/%d", p, r->abi_version, P2P_ABI_VERSION, r->world, c->world, r->rank, r->slots,
                 c->slots, (unsigned long long)r->slot_bytes, (unsigned long long)c->slot_bytes, r->lane_count,
                 c->lane_count);
            return -1;
        }
        if (r->n_devices < 1 || r->n_devices > P2P_MAX_DEVICES || r->region_addr == 0) {
            FAIL(c, "rank %d record names %u devices (1 to %d needed) or no arena", p, r->n_devices, P2P_MAX_DEVICES);
            return -1;
        }
        for (int q = 0; q < c->world; q++) {
            int theirs = (r->channels >> q) & 1u;
            int mirror = (all[q].channels >> p) & 1u;
            if (theirs != mirror || (q == p && theirs)) {
                FAIL(c, "rank %d %s a channel with rank %d, and rank %d %s", p, theirs ? "has" : "has no", q, q,
                     mirror ? "has one" : "has none");
                return -1;
            }
            for (int l = 0; l < c->lane_count; l++) {
                uint8_t d = r->lane_device[q][l];
                if (!theirs) {
                    if (d != P2P_NO_DEVICE) {
                        FAIL(c, "rank %d record names a device for lane %d toward rank %d, which has no channel", p,
                             l, q);
                        return -1;
                    }
                    continue;
                }
                if (d >= r->n_devices || r->qp_num[d][q] == 0) {
                    FAIL(c, "rank %d record has no queue pair for lane %d toward rank %d", p, l, q);
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int connect_lane(p2p_ctx_t *c, int p, int l, const p2p_record_t *peer) {
    int a = c->lane_device[p][l];
    int b = peer->lane_device[c->rank][l];
    p2p_dev_t *dev = &c->dev[a];
    struct ibv_qp *qp = dev->qp[p];
    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[b] < (uint32_t)dev->mtu ? peer->mtu[b] : (uint32_t)dev->mtu);
    rtr.dest_qp_num = peer->qp_num[b][c->rank];
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer->lid[b];
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = P2P_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[b], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)dev->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = (uint8_t)c->traffic_class;
    rtr.ah_attr.grh.flow_label = 0;   /* relays match the destination address of flow label 0 */
    int rc = ibv_modify_qp(qp, &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                               IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
        FAIL(c, "ibv_modify_qp(RTR) of lane %d toward rank %d on %s: %s", l, p, dev->name, strerror(rc));
        return -1;
    }
    struct ibv_qp_attr rts;
    memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = 0;
    rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(qp, &rts,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                           IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        FAIL(c, "ibv_modify_qp(RTS) of lane %d toward rank %d on %s: %s", l, p, dev->name, strerror(rc));
        return -1;
    }
    c->peer_rkey[p][l] = peer->rkey[b];
    memcpy(c->peer_gid[p][l], peer->gid[b], 16);
    return 0;
}

int p2p_connect(p2p_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (c->connected) {
        FAIL(c, "the point-to-point context is already connected");
        return -1;
    }
    if (blobs == NULL || blobs_len < sizeof(p2p_record_t) * (uint64_t)c->world) {
        FAIL(c, "connection records: %llu bytes for %d ranks of %zu bytes", (unsigned long long)blobs_len, c->world,
             sizeof(p2p_record_t));
        return -1;
    }
    const p2p_record_t *all = (const p2p_record_t *)blobs;
    if (validate_records(c, all) != 0) return -1;
    for (int p = 0; p < c->world; p++) {
        if (!c->ch[p].enabled) continue;
        c->peer_addr[p] = all[p].region_addr;
        for (int l = 0; l < c->lane_count; l++) {
            if (connect_lane(c, p, l, &all[p]) != 0) return -1;
        }
    }
    c->connected = 1;
    return 0;
}

/* -- completions ------------------------------------------------------------------------- */

static void format_address(const uint8_t *gid, char *out, size_t out_len) {
    static const uint8_t mapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    if (memcmp(gid, mapped, 12) == 0) {
        snprintf(out, out_len, "%u.%u.%u.%u", gid[12], gid[13], gid[14], gid[15]);
    } else {
        char text[INET6_ADDRSTRLEN];
        if (inet_ntop(AF_INET6, gid, text, sizeof(text)) == NULL) {
            snprintf(out, out_len, "(unprintable GID)");
        } else {
            snprintf(out, out_len, "%s", text);
        }
    }
}

static uint64_t make_wr_id(int peer, int lane, int kind, uint32_t item) {
    return (uint64_t)(uint8_t)peer | ((uint64_t)(uint8_t)lane << 8) | ((uint64_t)(uint8_t)kind << 16) |
           ((uint64_t)item << 32);
}

static const char *kind_name(int kind) {
    switch (kind) {
    case WR_DATA: return "data";
    case WR_HEADER: return "header";
    case WR_FLAG: return "flag";
    case WR_CREDIT: return "credit";
    case WR_CHECK: return "lane check";
    case WR_ABORT: return "abort notice";
    default: return "write";
    }
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int room(const p2p_ctx_t *c, int p, int lane, int wrs) {
    const p2p_dev_t *dev = &c->dev[c->lane_device[p][lane]];
    return dev->ack_count[p] + wrs <= P2P_INFLIGHT;
}

static void ack_push(p2p_ctx_t *c, int p, int lane, uint32_t bytes, uint32_t item, int kind) {
    p2p_dev_t *dev = &c->dev[c->lane_device[p][lane]];
    uint32_t tail = (uint32_t)(dev->ack_head[p] + dev->ack_count[p]) % P2P_SEND_DEPTH;
    dev->ack[p][tail].bytes = bytes;
    dev->ack[p][tail].item = item;
    dev->ack[p][tail].kind = (uint8_t)kind;
    dev->ack[p][tail].proven = 0;
    dev->ack_count[p] += 1;
    dev->unacked[p] += bytes;
    if (dev->unacked[p] > atomic_load_explicit(&c->window_max_unacked, memory_order_relaxed)) {
        atomic_store_explicit(&c->window_max_unacked, dev->unacked[p], memory_order_relaxed);
    }
}

/* A credit of `credit` from p proves delivered every data write of items before it on every lane. */
static void prove(p2p_ctx_t *c, int p, uint32_t credit) {
    for (int lane = 0; lane < c->lane_count; lane++) {
        p2p_dev_t *dev = &c->dev[c->lane_device[p][lane]];
        for (uint32_t k = 0; k < dev->ack_count[p]; k++) {
            p2p_ack_t *a = &dev->ack[p][(dev->ack_head[p] + k) % P2P_SEND_DEPTH];
            if (a->kind != WR_DATA || a->proven) continue;
            if ((int32_t)(a->item - credit) >= 0) break;
            a->proven = 1;
            dev->unacked[p] -= a->bytes;
            atomic_fetch_add_explicit(&c->proven_bytes, a->bytes, memory_order_relaxed);
        }
    }
}

static int drain_device(p2p_ctx_t *c, int d) {
    p2p_dev_t *dev = &c->dev[d];
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(dev->cq, 32, wc);
    if (n < 0) {
        FAIL(c, "ibv_poll_cq(%s) failed", dev->name);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        int peer = (int)(wc[i].wr_id & 0xFFu);
        int lane = (int)((wc[i].wr_id >> 8) & 0xFFu);
        int kind = (int)((wc[i].wr_id >> 16) & 0xFFu);
        uint32_t item = (uint32_t)(wc[i].wr_id >> 32);
        if (wc[i].status != IBV_WC_SUCCESS) {
            char where[64] = "?";
            if (peer < c->world && lane < c->lane_count) format_address(c->peer_gid[peer][lane], where, sizeof(where));
            FAIL(c, "RDMA %s write of item %u to rank %d lane %d (destination %s) on %s failed: %s (vendor_err 0x%x)",
                 kind_name(kind), item, peer, lane, where, dev->name, ibv_wc_status_str(wc[i].status),
                 wc[i].vendor_err);
            return -1;
        }
        if (kind == WR_CHECK || kind == WR_ABORT || peer >= c->world || lane >= c->lane_count) continue;
        if (dev->ack_count[peer] != 0) {
            p2p_ack_t *a = &dev->ack[peer][dev->ack_head[peer]];
            if (!a->proven) dev->unacked[peer] -= a->bytes;
            dev->ack_head[peer] = (uint16_t)((dev->ack_head[peer] + 1) % P2P_SEND_DEPTH);
            dev->ack_count[peer] -= 1;
        }
        if (kind == WR_FLAG) c->ch[peer].lane_done[lane]++;
        atomic_fetch_add_explicit(&dev->writes_completed, 1, memory_order_relaxed);
    }
    return 0;
}

static int drain_all(p2p_ctx_t *c) {
    for (int d = 0; d < c->n_dev; d++) {
        if (drain_device(c, d) != 0) return -1;
    }
    return 0;
}

/* -- posting ----------------------------------------------------------------------------------- */

static int post(p2p_ctx_t *c, int p, int lane, int kind, uint32_t item, uint64_t local, uint32_t length,
                const uint32_t *inline_word, uint64_t remote_offset, uint32_t window_bytes) {
    p2p_dev_t *dev = &c->dev[c->lane_device[p][lane]];
    struct ibv_sge sge;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    if (inline_word != NULL) {
        sge.addr = (uint64_t)(uintptr_t)inline_word;
        sge.length = 4;
        sge.lkey = 0;
        wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    } else {
        sge.addr = (uint64_t)(uintptr_t)(c->region + local);
        sge.length = length;
        sge.lkey = dev->mr->lkey;
        wr.send_flags = IBV_SEND_SIGNALED;
    }
    wr.wr_id = make_wr_id(p, lane, kind, item);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr.rdma.remote_addr = c->peer_addr[p] + remote_offset;
    wr.wr.rdma.rkey = c->peer_rkey[p][lane];
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(dev->qp[p], &wr, &bad);
    if (rc != 0) {
        FAIL(c, "ibv_post_send of the %s write of item %u, lane %d to rank %d on %s: %s", kind_name(kind), item, lane,
             p, dev->name, strerror(rc));
        return -1;
    }
    ack_push(c, p, lane, window_bytes, item, kind);
    if (inline_word == NULL) atomic_fetch_add_explicit(&dev->bytes_posted, length, memory_order_relaxed);
    return 0;
}

static void note_wait(p2p_ctx_t *c, p2p_channel_t *ch, int waiting) {
    uint64_t now = monotonic_ns();
    if (waiting) {
        if (ch->wait_started == 0) {
            ch->wait_started = now;
            atomic_fetch_add_explicit(&c->window_waits, 1, memory_order_relaxed);
        }
        return;
    }
    if (ch->wait_started != 0) {
        uint64_t waited = now - ch->wait_started;
        atomic_fetch_add_explicit(&c->window_wait_ns, waited, memory_order_relaxed);
        if (waited > atomic_load_explicit(&c->window_wait_max_ns, memory_order_relaxed)) {
            atomic_store_explicit(&c->window_wait_max_ns, waited, memory_order_relaxed);
        }
        ch->wait_started = 0;
    }
}

/* Advance the channel toward p: every lane posts the stripes, header and flags of the ready items its
 * credit and window allow. Returns 1 when anything was posted, 0 when nothing, -1 on a failure. */
static int service_out(p2p_ctx_t *c, int p) {
    p2p_channel_t *ch = &c->ch[p];
    uint32_t slots = (uint32_t)c->slots;
    uint32_t credit = __atomic_load_n(block_word(c, p, L_CREDIT, 0), __ATOMIC_ACQUIRE);
    int work = 0;
    if (credit != ch->credit_seen) {
        prove(c, p, credit);
        ch->credit_seen = credit;
        work = 1;
    }
    uint64_t mine = block_off(c, c->rank);     /* this rank's block in the peer's arena */
    int window_blocked = 0;
    for (int lane = 0; lane < c->lane_count; lane++) {
        p2p_dev_t *dev = &c->dev[c->lane_device[p][lane]];
        uint32_t window = c->window[p][lane];
        for (;;) {
            uint32_t g = ch->lane_item[lane];
            uint32_t m = g % slots, tag = g + 1u;
            if ((int32_t)(credit - (tag - slots)) < 0) break;     /* the receiver has not freed the slot */
            if (__atomic_load_n(block_word(c, p, L_READY, m), __ATOMIC_ACQUIRE) != tag) break;
            uint32_t header = __atomic_load_n(block_word(c, p, L_DESC, m), __ATOMIC_ACQUIRE);
            uint32_t bytes = header & P2P_BYTES_MASK, first, count;
            if (bytes > c->slot_bytes) {
                FAIL(c, "item %u toward rank %d has a header of %u bytes, more than a slot of %llu bytes", g, p, bytes,
                     (unsigned long long)c->slot_bytes);
                return -1;
            }
            lane_split(bytes / 16u, c->lane_count, lane, &first, &count);
            uint32_t stripe = count * 16u;
            uint64_t src = block_off(c, p) + c->layout[L_SEND] + (uint64_t)m * c->slot_bytes + (uint64_t)first * 16u;
            uint64_t dst = mine + c->layout[L_RECV] + (uint64_t)m * c->slot_bytes + (uint64_t)first * 16u;
            while (ch->lane_bytes[lane] < stripe) {
                uint32_t n = stripe - ch->lane_bytes[lane];
                if (window != 0) {
                    if (n > c->chunk) n = c->chunk;
                    uint64_t free_bytes = dev->unacked[p] < window ? window - dev->unacked[p] : 0;
                    if (free_bytes < n) {
                        /* Post the part the window holds (at least 4 KiB, whole packs), else wait. */
                        uint32_t part = (uint32_t)(free_bytes / 16u * 16u);
                        if (part < P2P_MIN_PIECE || part == 0) {
                            window_blocked = 1;
                            break;
                        }
                        n = part;
                    }
                }
                if (!room(c, p, lane, 1)) break;
                if (post(c, p, lane, WR_DATA, g, src + ch->lane_bytes[lane], n, NULL, dst + ch->lane_bytes[lane],
                         window != 0 ? n : 0) != 0) return -1;
                ch->lane_bytes[lane] += n;
                work = 1;
            }
            if (ch->lane_bytes[lane] < stripe) break;
            if (lane == 0 && !ch->lane_header[0]) {
                if (!room(c, p, lane, 1)) break;
                if (post(c, p, lane, WR_HEADER, g, 0, 4, &header, mine + flag_off(c, m, 0) + 4u, 0) != 0) return -1;
                ch->lane_header[0] = 1;
                work = 1;
            }
            if (!room(c, p, lane, 1)) break;
            if (post(c, p, lane, WR_FLAG, g, 0, 4, &tag, mine + flag_off(c, m, lane), 0) != 0) return -1;
            ch->lane_item[lane] = g + 1u;
            ch->lane_bytes[lane] = 0;
            ch->lane_header[lane] = 0;
            work = 1;
        }
    }
    note_wait(c, ch, window_blocked);
    uint32_t posted = ch->lane_item[0];
    for (int lane = 1; lane < c->lane_count; lane++) {
        if ((int32_t)(ch->lane_item[lane] - posted) < 0) posted = ch->lane_item[lane];
    }
    while (ch->next_post != posted) {
        uint32_t m = ch->next_post % slots;
        uint32_t header = __atomic_load_n(block_word(c, p, L_DESC, m), __ATOMIC_ACQUIRE);
        atomic_fetch_add_explicit(&ch->items_posted, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&ch->bytes_posted, header & P2P_BYTES_MASK, memory_order_relaxed);
        ch->next_post++;
    }
    uint32_t done = ch->lane_done[0];
    for (int lane = 1; lane < c->lane_count; lane++) {
        if ((int32_t)(ch->lane_done[lane] - done) < 0) done = ch->lane_done[lane];
    }
    if (done != ch->done) {
        ch->done = done;
        __atomic_store_n(block_word(c, p, L_SENT, 0), done, __ATOMIC_RELEASE);
        work = 1;
    }
    return work;
}

/* Advance the channel from p: release the items the kernel consumed and return their count to p. */
static int service_in(p2p_ctx_t *c, int p) {
    p2p_channel_t *ch = &c->ch[p];
    uint32_t slots = (uint32_t)c->slots;
    int work = 0;
    while (__atomic_load_n(block_word(c, p, L_CONSUMED, ch->released % slots), __ATOMIC_ACQUIRE) ==
           ch->released + 1u) {
        ch->released++;
        atomic_fetch_add_explicit(&ch->items_released, 1, memory_order_relaxed);
        work = 1;
    }
    if (ch->released != ch->credited && room(c, p, 0, 1)) {
        uint32_t value = ch->released;
        uint64_t mine = block_off(c, c->rank);
        if (post(c, p, 0, WR_CREDIT, value, 0, 4, &value, mine + c->layout[L_CREDIT], 0) != 0) return -1;
        ch->credited = value;
        atomic_fetch_add_explicit(&ch->credits_sent, 1, memory_order_relaxed);
        work = 1;
    }
    return work;
}

static int outstanding(const p2p_ctx_t *c) {
    for (int d = 0; d < c->n_dev; d++) {
        for (int p = 0; p < c->world; p++) {
            if (c->dev[d].ack_count[p] != 0) return 1;
        }
    }
    return 0;
}

/* Write an abort notice naming the rank where the failure happened (`origin`) into every peer's abort
 * word (best effort, once), so every rank of the group stops and names the same origin. */
static void post_aborts(p2p_ctx_t *c, int origin) {
    if (c->aborted) return;
    c->aborted = 1;
    uint32_t value[P2P_MAX_PEERS];   /* inline data: copied when the work request is posted */
    for (int p = 0; p < c->world; p++) {
        if (!c->ch[p].enabled) continue;
        value[p] = (uint32_t)origin + 1u;
        p2p_dev_t *dev = &c->dev[c->lane_device[p][0]];
        if (dev->ack_count[p] >= P2P_SEND_DEPTH - 1) continue;
        struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)&value[p], .length = 4, .lkey = 0};
        struct ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = make_wr_id(p, 0, WR_ABORT, 0);
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        wr.wr.rdma.remote_addr = c->peer_addr[p] + 4u * (uint64_t)C_ABORT;
        wr.wr.rdma.rkey = c->peer_rkey[p][0];
        struct ibv_send_wr *bad = NULL;
        (void)ibv_post_send(dev->qp[p], &wr, &bad);
    }
    /* Give the notices a moment to leave before the thread stops polling. */
    uint64_t until = monotonic_ns() + 2000000ull;
    while (monotonic_ns() < until) {
        for (int d = 0; d < c->n_dev; d++) {
            struct ibv_wc wc[32];
            (void)ibv_poll_cq(c->dev[d].cq, 32, wc);
        }
    }
}

static void note_cpu(p2p_ctx_t *c) {
    int cpu = sched_getcpu();
    int last = atomic_load_explicit(&c->last_cpu, memory_order_relaxed);
    if (cpu != last) {
        if (last >= 0) atomic_fetch_add_explicit(&c->cpu_migrations, 1, memory_order_relaxed);
        atomic_store_explicit(&c->last_cpu, cpu, memory_order_relaxed);
    }
}

static void *progress_main(void *arg) {
    p2p_ctx_t *c = (p2p_ctx_t *)arg;
    uint64_t idle = 0;
    const struct timespec nap = {0, P2P_NAP_NS};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        /* Every stop records the error text, then the failed flag, then the poison word: a host that sees
         * the poison (`poisoned`) and asks for the error finds the text. */
        uint32_t abort_word = __atomic_load_n(control(c, C_ABORT), __ATOMIC_ACQUIRE);
        if (abort_word != 0) {
            atomic_store(&c->abort_from, abort_word);
            FAIL(c, "the point-to-point channels of this group stopped after a failure on rank %u (that rank's "
                    "error names it)", abort_word - 1u);
            atomic_store(&c->failed, 1);
            __atomic_store_n(control(c, C_POISON), 1u, __ATOMIC_RELEASE);
            post_aborts(c, (int)abort_word - 1);
            return NULL;
        }
        if (__atomic_load_n(control(c, C_ERROR_TAG), __ATOMIC_ACQUIRE) != 0 ||
            __atomic_load_n(control(c, C_POISON), __ATOMIC_ACQUIRE) != 0) {
            FAIL(c, "a point-to-point kernel of rank %d recorded a failure (control line: kind %u, peer %u, lane %u, "
                    "item tag %u)", c->rank, *control(c, C_ERROR_KIND), *control(c, C_ERROR_PEER),
                 *control(c, C_ERROR_LANE), *control(c, C_ERROR_TAG));
            atomic_store(&c->failed, 1);
            __atomic_store_n(control(c, C_POISON), 1u, __ATOMIC_RELEASE);
            post_aborts(c, c->rank);
            return NULL;
        }
        int work = 0;
        for (int p = 0; p < c->world; p++) {
            if (!c->ch[p].enabled) continue;
            int moved = service_out(c, p);
            if (moved < 0) goto native_failed;
            work |= moved;
            moved = service_in(c, p);
            if (moved < 0) goto native_failed;
            work |= moved;
        }
        if (outstanding(c) && drain_all(c) != 0) goto native_failed;
        if (work) {
            idle = 0;
            note_cpu(c);
            continue;
        }
        idle++;
        if (idle >= P2P_IDLE_SPINS) nanosleep(&nap, NULL);
    }
    return NULL;
native_failed:
    atomic_store(&c->failed, 1);
    __atomic_store_n(control(c, C_POISON), 1u, __ATOMIC_RELEASE);
    post_aborts(c, c->rank);
    return NULL;
}

/* -- lane check, windows and lifecycle ----------------------------------------------------- */

static int64_t monotonic_ms(void) { return (int64_t)(monotonic_ns() / 1000000ull); }

/* One signaled 4-byte write per (peer, lane) into the peer's lane-check word; every completion must
 * arrive successfully within `timeout_ms`. Runs after every rank connected and before the progress
 * thread starts. */
int p2p_lane_check(p2p_ctx_t *c, int timeout_ms) {
    int pending[P2P_MAX_PEERS][P2P_MAX_LANES];
    int remaining = 0;
    static const uint32_t value = P2P_RECORD_MAGIC;
    if (!c->connected || atomic_load(&c->running)) {
        FAIL(c, "the lane check runs on a connected context before its progress thread starts");
        return -1;
    }
    memset(pending, 0, sizeof(pending));
    for (int p = 0; p < c->world; p++) {
        if (!c->ch[p].enabled) continue;
        for (int l = 0; l < c->lane_count; l++) {
            p2p_dev_t *dev = &c->dev[c->lane_device[p][l]];
            struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
            struct ibv_send_wr wr;
            memset(&wr, 0, sizeof(wr));
            wr.wr_id = make_wr_id(p, l, WR_CHECK, 0);
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            wr.wr.rdma.remote_addr = c->peer_addr[p] + 4u * (uint64_t)C_LANE_CHECK;
            wr.wr.rdma.rkey = c->peer_rkey[p][l];
            struct ibv_send_wr *bad = NULL;
            int rc = ibv_post_send(dev->qp[p], &wr, &bad);
            if (rc != 0) {
                FAIL(c, "lane check: posting lane %d of rank %d toward rank %d on %s: %s", l, c->rank, p, dev->name,
                     strerror(rc));
                return -1;
            }
            pending[p][l] = 1;
            remaining++;
        }
    }
    int64_t deadline = monotonic_ms() + (timeout_ms > 0 ? timeout_ms : 2000);
    while (remaining > 0) {
        for (int d = 0; d < c->n_dev; d++) {
            struct ibv_wc wc[32];
            int n = ibv_poll_cq(c->dev[d].cq, 32, wc);
            if (n < 0) {
                FAIL(c, "lane check: ibv_poll_cq(%s) failed", c->dev[d].name);
                return -1;
            }
            for (int i = 0; i < n; i++) {
                int p = (int)(wc[i].wr_id & 0xFFu);
                int l = (int)((wc[i].wr_id >> 8) & 0xFFu);
                if (wc[i].status != IBV_WC_SUCCESS) {
                    char where[64] = "?";
                    if (p < c->world && l < c->lane_count) format_address(c->peer_gid[p][l], where, sizeof(where));
                    FAIL(c, "lane check: lane %d of rank %d toward rank %d (local device %s, destination %s) failed: "
                            "%s (vendor_err 0x%x)", l, c->rank, p, c->dev[d].name, where,
                         ibv_wc_status_str(wc[i].status), wc[i].vendor_err);
                    return -1;
                }
                if (p < c->world && l < c->lane_count && pending[p][l]) {
                    pending[p][l] = 0;
                    remaining--;
                }
            }
        }
        if (remaining > 0 && monotonic_ms() > deadline) {
            for (int p = 0; p < c->world; p++) {
                for (int l = 0; l < c->lane_count; l++) {
                    if (!pending[p][l]) continue;
                    char where[64];
                    format_address(c->peer_gid[p][l], where, sizeof(where));
                    FAIL(c, "lane check: lane %d of rank %d toward rank %d (local device %s, destination %s) did not "
                            "complete within %d ms", l, c->rank, p, c->dev[c->lane_device[p][l]].name, where,
                         timeout_ms > 0 ? timeout_ms : 2000);
                    return -1;
                }
            }
        }
    }
    return 0;
}

/* Forward windows of the lanes (world * lane_count entries in bytes, 0: a direct lane) and the chunk
 * of a windowed stripe; set before the progress thread starts. A NULL table removes every window. */
int p2p_set_windows(p2p_ctx_t *c, const uint32_t *lane_window_bytes, uint32_t chunk_bytes) {
    if (atomic_load(&c->running)) {
        FAIL(c, "forward windows are set before the progress thread starts");
        return -1;
    }
    memset(c->window, 0, sizeof(c->window));
    if (lane_window_bytes == NULL) return 0;
    if (chunk_bytes < 16 || chunk_bytes % 16 != 0) {
        FAIL(c, "forward chunk of %u bytes must be a positive multiple of 16", chunk_bytes);
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        for (int l = 0; l < c->lane_count; l++) {
            uint32_t w = lane_window_bytes[p * c->lane_count + l];
            if (w == 0) continue;
            if (!c->ch[p].enabled) {
                FAIL(c, "a forward window of %u bytes on lane %d toward rank %d, which has no channel", w, l, p);
                return -1;
            }
            if (w < chunk_bytes || w / chunk_bytes > P2P_INFLIGHT - 8) {
                FAIL(c, "forward window of %u bytes on lane %d toward rank %d must hold 1 to %d chunks of %u bytes",
                     w, l, p, P2P_INFLIGHT - 8, chunk_bytes);
                return -1;
            }
            c->window[p][l] = w;
        }
    }
    c->chunk = chunk_bytes;
    return 0;
}

int p2p_start(p2p_ctx_t *c) {
    if (!c->connected) {
        FAIL(c, "the progress thread needs a connected context");
        return -1;
    }
    if (atomic_load(&c->running)) return 0;
    if (atomic_load(&c->failed)) {
        FAIL(c, "a failed point-to-point context does not restart");
        return -1;
    }
    pthread_attr_t attr;
    pthread_attr_t *use = NULL;
    const char *cpus = getenv("SIRCL_P2P_PROGRESS_CPU");
    if (cpus != NULL && *cpus != '\0') {
        cpu_set_t set;
        if (parse_cpu_list(cpus, &set) != 0) {
            FAIL(c, "SIRCL_P2P_PROGRESS_CPU=%s is not a CPU list such as 9 or 5-9,15-19", cpus);
            return -1;
        }
        pthread_attr_init(&attr);
        int rc = pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
        if (rc != 0) {
            pthread_attr_destroy(&attr);
            FAIL(c, "pinning the progress thread to SIRCL_P2P_PROGRESS_CPU=%s: %s", cpus, strerror(rc));
            return -1;
        }
        use = &attr;
    }
    c->started = 1;
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, use, progress_main, c);
    if (use != NULL) pthread_attr_destroy(&attr);
    if (rc != 0) {
        atomic_store(&c->running, 0);
        FAIL(c, "pthread_create(progress thread): %s", strerror(rc));
        return -1;
    }
    return 0;
}

void p2p_stop(p2p_ctx_t *c) {
    if (c != NULL && atomic_exchange(&c->running, 0)) pthread_join(c->thread, NULL);
}

int p2p_failed(p2p_ctx_t *c) { return atomic_load(&c->failed); }

const char *p2p_error(p2p_ctx_t *c) { return c->err; }

uint64_t p2p_stat(p2p_ctx_t *c, int which) {
    uint64_t sum = 0;
    switch (which) {
    case 0:
    case 1:
    case 2:
    case 3:
        for (int p = 0; p < c->world; p++) {
            const p2p_channel_t *ch = &c->ch[p];
            if (which == 0) sum += atomic_load_explicit(&ch->items_posted, memory_order_relaxed);
            if (which == 1) sum += atomic_load_explicit(&ch->bytes_posted, memory_order_relaxed);
            if (which == 2) sum += atomic_load_explicit(&ch->items_released, memory_order_relaxed);
            if (which == 3) sum += atomic_load_explicit(&ch->credits_sent, memory_order_relaxed);
        }
        return sum;
    case 4:
        for (int d = 0; d < c->n_dev; d++) sum += atomic_load_explicit(&c->dev[d].writes_completed, memory_order_relaxed);
        return sum;
    case 5: return atomic_load_explicit(&c->window_waits, memory_order_relaxed);
    case 6: return atomic_load_explicit(&c->window_wait_ns, memory_order_relaxed);
    case 7: return atomic_load_explicit(&c->window_wait_max_ns, memory_order_relaxed);
    case 8: return atomic_load_explicit(&c->window_max_unacked, memory_order_relaxed);
    case 9: return atomic_load_explicit(&c->proven_bytes, memory_order_relaxed);
    case 10: return (uint64_t)(atomic_load_explicit(&c->last_cpu, memory_order_relaxed) + 1);
    case 11: return atomic_load_explicit(&c->cpu_migrations, memory_order_relaxed);
    case 12: return atomic_load_explicit(&c->abort_from, memory_order_relaxed);
    case 13: return (uint64_t)c->lane_count;
    default: return 0;
    }
}

/* Per peer: 0 items posted toward it, 1 bytes posted toward it, 2 items released from it, 3 credits
 * sent to it, 4 its newest credit, 5 the sent word (items whose writes completed). */
uint64_t p2p_peer_stat(p2p_ctx_t *c, int peer, int which) {
    if (peer < 0 || peer >= c->world) return UINT64_MAX;
    const p2p_channel_t *ch = &c->ch[peer];
    switch (which) {
    case 0: return atomic_load_explicit(&ch->items_posted, memory_order_relaxed);
    case 1: return atomic_load_explicit(&ch->bytes_posted, memory_order_relaxed);
    case 2: return atomic_load_explicit(&ch->items_released, memory_order_relaxed);
    case 3: return atomic_load_explicit(&ch->credits_sent, memory_order_relaxed);
    case 4: return ch->credit_seen;
    case 5: return ch->done;
    default: return 0;
    }
}

uint64_t p2p_hca_stat(p2p_ctx_t *c, int device, int which) {
    if (device < 0 || device >= c->n_dev) return UINT64_MAX;
    switch (which) {
    case 0: return atomic_load_explicit(&c->dev[device].writes_completed, memory_order_relaxed);
    case 1: return atomic_load_explicit(&c->dev[device].bytes_posted, memory_order_relaxed);
    default: return 0;
    }
}

#ifdef SIRCL_PROXY_TEST_HOOKS
/* Start every channel's item counters at `base` (before the progress thread starts); the arena's
 * ready, flag, consumed, sent and credit words must hold `base` too (the tag of item base - 1). */
int p2p_test_set_base(p2p_ctx_t *c, uint32_t base) {
    if (atomic_load(&c->running)) return -1;
    for (int p = 0; p < c->world; p++) {
        p2p_channel_t *ch = &c->ch[p];
        ch->next_post = ch->done = ch->released = ch->credited = ch->credit_seen = base;
        for (int l = 0; l < P2P_MAX_LANES; l++) {
            ch->lane_item[l] = base;
            ch->lane_done[l] = base;
            ch->lane_bytes[l] = 0;
            ch->lane_header[l] = 0;
        }
    }
    return 0;
}

uint32_t p2p_test_qp_num(p2p_ctx_t *c, int d, int p) {
    return c->dev[d].qp[p] != NULL ? c->dev[d].qp[p]->qp_num : 0;
}
#endif

/* Stops the progress thread and releases every verbs object; returns the number of verbs calls that
 * failed. A queue pair or memory registration that could not be released can still let a peer's write
 * reach the arena, so the caller keeps the arena allocated when the count is not zero. */
int p2p_destroy(p2p_ctx_t *c) {
    if (c == NULL) return 0;
    p2p_stop(c);
    int failed = 0;
    for (int d = 0; d < P2P_MAX_DEVICES; d++) {
        p2p_dev_t *dev = &c->dev[d];
        for (int p = 0; p < P2P_MAX_PEERS; p++) {
            if (dev->qp[p] != NULL && ibv_destroy_qp(dev->qp[p]) != 0) failed++;
        }
        if (dev->cq != NULL && ibv_destroy_cq(dev->cq) != 0) failed++;
        if (dev->mr != NULL && ibv_dereg_mr(dev->mr) != 0) failed++;
        if (dev->pd != NULL && ibv_dealloc_pd(dev->pd) != 0) failed++;
        if (dev->ctx != NULL && ibv_close_device(dev->ctx) != 0) failed++;
    }
    free(c);
    return failed;
}
