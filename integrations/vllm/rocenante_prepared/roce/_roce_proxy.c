// RDMA proxy for the b12x RoCE one-shot all-reduce.
//
// One rank owns one pinned host region laid out as:
//
//   recv[src][slot]  (world * SLOTS * slot_bytes)  filled by peers' RDMA writes
//   flag[row][slot][lane]  (world * SLOTS * 2 * FLAG_STRIDE) sequence number
//                         written after one stripe. Four-path TP4 mode keeps
//                         lanes 0/1 in the source row and stores opposite-path
//                         lanes 2/3 in the otherwise-unused receiver row.
//   send[slot]       (SLOTS * slot_bytes)          staged by the local GPU kernel
//   ctrl             (FLAG_STRIDE)                 u32 words:
//                                                    0 seq (doorbell), 1 nbytes,
//                                                    2 stopped seq, 3 missing peer,
//                                                    4-5 nbytes per slot,
//                                                    6 abort (host-written),
//                                                    7 completed seq,
//                                                    8 sticky wait-stopped flag,
//                                                    16+r notice from rank r
//
// The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq
// in ctrl, then spins on every active path flag for every peer. The proxy uses
// two half-payload paths per neighbor. Research-only TP4 mode uses four
// quarter-payload paths to the opposite rank. Each stripe is followed by its
// 4-byte sequence flag on the same reliable QP, so a path flag cannot become
// visible before its stripe. Nothing on the receive path involves the host.
//
// Peer waits are supervised from this thread. A waiting kernel also reads the
// abort word every 1024 polls and stops only when it is nonzero (or after its
// own device poll bound). The proxy writes the abort word when a peer's queue
// pair no longer acknowledges a check, when a peer reports that it stopped,
// when the peer's flags contradict this rank's sequence, when one wait
// exceeds the configured timeout, or when the kernel reached its device
// bound; a peer that is merely late is waited for. Stopping also writes a
// stop notice to every peer, so no rank keeps waiting for this one. Each
// stall and its end are logged to standard error with the flag values the
// host sees, and the waiting rank writes a notice into the late peer's
// control record so that the peer logs its own doorbell and posting state.
//
// This file is compiled by b12x.comm.roce._proxy at first use with the host
// gcc and libibverbs; it must stay plain C with no CUDA dependency.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_HCAS 4
#define ROCE_SLOTS 2
#define ROCE_LAYOUT_PATHS 2
#define ROCE_MAX_PATHS 4
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 256
#define ROCE_ABI_VERSION 5
// Control-record word indices; the kernels use the same byte offsets.
#define ROCE_CTRL_STOPPED_SEQ 2
#define ROCE_CTRL_MISSING_PEER 3
#define ROCE_CTRL_ABORT 6
#define ROCE_CTRL_DONE 7
#define ROCE_CTRL_FAILED 8
#define ROCE_CTRL_NOTICE 16
_Static_assert(ROCE_CTRL_NOTICE + ROCE_MAX_PEERS <= ROCE_FLAG_STRIDE / 4,
               "peer notices must fit in the control record");
// A notice word carries the low 31 bits of a sequence; the top bit marks a
// peer that stopped its runtime rather than one that waits for this rank.
#define ROCE_NOTICE_ABORT 0x80000000u
#define ROCE_NOTICE_SEQ 0x7fffffffu
// Work-request ID bit of notice writes, which are not payload traffic.
#define ROCE_NOTICE_WR (1ull << 63)
#define ROCE_DEFAULT_WAIT_TIMEOUT_NS 300000000000ull
#define ROCE_STALL_REPORT_NS 5000000000ull
#define ROCE_PROBE_INTERVAL_NS 1000000000ull
#define ROCE_LOOP_GAP_REPORT_NS 1000000000ull
// Keep the proxy hot across sub-millisecond compute gaps inside model graphs.
#define ROCE_IDLE_SPINS 20000000
#define ROCE_DEFAULT_TWO_WAVE_THRESHOLD_BYTES 131072u
#define ROCE_WAVE_MODE_TWO 0u
#define ROCE_WAVE_MODE_MIXED_TWO 1u
#define ROCE_WAVE_MODE_OPPOSITE_FIRST 2u
#define ROCE_WAVE_MODE_STRICT_THREE 3u
#define ROCE_WAVE_MODE_BALANCED32 4u
// Hardware-forwarded (two-hop) paths cross the intermediate ConnectX through
// a hairpin queue. The mlx5 driver sizes that queue at hairpin_queue_size
// 64-byte strides (64 KiB at the 1024 default, 512 KiB at the 8192 maximum),
// and a full hairpin queue drops packets without pausing the upstream link.
// Each drop costs the reliable connection a go-back-N resend or, for a lost
// tail packet, a retransmission timeout that stalls every rank. Stripes on
// forwarded paths are therefore posted as signaled chunks with a bounded
// number of unacknowledged bytes per queue pair, so the intermediate never
// buffers more than the window for one flow. The default window assumes
// hairpin_queue_size 8192; with the 1024 driver default, set a window no
// larger than 32 KiB.
#define ROCE_DEFAULT_FORWARD_WINDOW_BYTES 131072u
#define ROCE_DEFAULT_FORWARD_CHUNK_BYTES 32768u
#define ROCE_MAX_STREAMS (ROCE_MAX_PEERS * ROCE_MAX_PATHS)

typedef struct {
    uint32_t abi_version;
    uint32_t world;
    uint32_t rank;
    uint32_t n_hca;
    uint32_t layout_paths;
    uint32_t opposite_paths;
    uint64_t region_addr;
    uint32_t rkey[ROCE_MAX_HCAS];
    uint16_t lid[ROCE_MAX_HCAS];
    uint8_t gid[ROCE_MAX_HCAS][16];
    uint32_t mtu[ROCE_MAX_HCAS];
    uint32_t qp_num[ROCE_MAX_HCAS][ROCE_MAX_PEERS];
    uint8_t peer_hca[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
} roce_blob_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[ROCE_MAX_PEERS];
    uint32_t outstanding[ROCE_MAX_PEERS];
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
} roce_hca_t;

typedef struct {
    int world;
    int rank;
    int n_hca;
    int gid_index;
    roce_hca_t hca[ROCE_MAX_HCAS];
    uint8_t *region;
    size_t region_bytes;
    size_t slot_bytes;
    size_t recv_off;
    size_t flag_off;
    size_t send_off;
    size_t ctrl_off;
    int started;
    uint64_t peer_addr[ROCE_MAX_PEERS];
    uint32_t peer_rkey[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    uint32_t remote_qp[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    int peer_hca[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    int remote_hca[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    uint32_t peer_path_count[ROCE_MAX_PEERS];
    uint32_t opposite_paths;
    uint32_t physical_hops[ROCE_MAX_PEERS];
    atomic_uint_fast64_t payload_writes[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    atomic_uint_fast64_t payload_bytes[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    atomic_uint_fast64_t flag_writes[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    atomic_uint_fast64_t send_completions[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    atomic_uint_fast64_t completion_errors[ROCE_MAX_PEERS][ROCE_MAX_PATHS];
    int direct_peer_by_hca[ROCE_MAX_HCAS];
    int direct_path_by_hca[ROCE_MAX_HCAS];
    int opposite_peer_by_hca[ROCE_MAX_HCAS];
    int opposite_path_by_hca[ROCE_MAX_HCAS];
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    uint32_t last_seq;
    uint64_t ops_posted;
    uint64_t writes_completed;
    uint32_t two_wave_threshold_bytes;
    uint32_t wave_mode;
    atomic_uint_fast64_t two_wave_activations;
    // Forwarded-path flow control; a zero window disables chunking.
    uint32_t forward_window_bytes;
    uint32_t forward_chunk_bytes;
    atomic_uint_fast64_t forward_chunks;
    // Forwarded stripes registered by post_path and completed by pump_streams
    // before post_op returns.
    struct {
        uint32_t seq;
        uint32_t slot;
        uint8_t *send;
        int peer;
        int path;
        uint32_t next;
        uint32_t end;
    } streams[ROCE_MAX_STREAMS];
    int n_streams;
    // Peer-wait supervision, owned by the proxy thread; the atomics are
    // diagnostic counters that other threads sample.
    uint64_t wait_timeout_ns;
    uint64_t stall_report_ns;
    uint64_t posted_ns;
    uint64_t last_tick_ns;
    uint32_t watched_seq;
    int watching;
    int stall_reported;
    uint64_t watch_posted_ns;
    uint64_t next_probe_ns;
    uint32_t notice_seen[ROCE_MAX_PEERS];
    atomic_uint_fast64_t stalls;
    atomic_uint_fast64_t stalls_resolved;
    atomic_uint_fast64_t longest_stall_ns;
    atomic_uint_fast64_t notices_posted;
    atomic_uint_fast64_t notices_received;
    atomic_uint_fast64_t longest_loop_gap_ns;
    char err[512];
} roce_ctx_t;

void roce_destroy(roce_ctx_t *c);

static void set_err(roce_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

static uint64_t roce_now_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void roce_log(const roce_ctx_t *c, const char *format, ...) {
    char line[768];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    fprintf(stderr, "RoCEnante rank %d: %s\n", c->rank, line);
}

static double roce_seconds(uint64_t ns) { return (double)ns / 1e9; }

static void roce_raise_max(atomic_uint_fast64_t *value, uint64_t candidate) {
    uint_fast64_t current = atomic_load(value);
    while (candidate > current &&
           !atomic_compare_exchange_weak(value, &current, candidate)) {
    }
}

// Pure peer-wait decisions. The tests compile this section on its own.
// ROCE_SUPERVISION_BEGIN
#define ROCE_FLAG_ARRIVED 0
#define ROCE_FLAG_PENDING 1
#define ROCE_FLAG_INCONSISTENT 2

// Classifies one path's flags for the awaited sequence ``seq``: ``current``
// is the flag of slot seq & 1 and ``other`` the flag of the other slot. One
// reliable connection delivers a path's flags in posting order, and a sender
// posts ``seq`` only after it received this rank's ``seq - 1``. A sender that
// has not posted ``seq`` therefore leaves ``current`` at seq - 2 (0 while the
// slot is unused) and ``other`` at seq - 1; any other pair means the two
// ranks disagree about the collective sequence.
static int roce_flag_state(uint32_t seq, uint32_t current, uint32_t other) {
    if (current == seq) {
        return other == seq - 1u || other == seq + 1u ? ROCE_FLAG_ARRIVED
                                                      : ROCE_FLAG_INCONSISTENT;
    }
    if ((current == seq - 2u || (seq == 1u && current == 0u)) && other == seq - 1u) {
        return ROCE_FLAG_PENDING;
    }
    return ROCE_FLAG_INCONSISTENT;
}

// Recovers the full sequence of a notice from its low 31 bits, choosing the
// value nearest ``reference``; exact while the two differ by less than 2^30.
static uint32_t roce_notice_seq(uint32_t notice, uint32_t reference) {
    uint32_t delta = (notice - reference) & ROCE_NOTICE_SEQ;
    return delta >= 0x40000000u ? reference - (ROCE_NOTICE_SEQ + 1u - delta)
                                : reference + delta;
}

// Stall report threshold for a wait timeout: 5 s, or half a shorter timeout.
static uint64_t roce_stall_report_ns(uint64_t timeout_ns) {
    return timeout_ns / 2u < ROCE_STALL_REPORT_NS ? timeout_ns / 2u
                                                  : ROCE_STALL_REPORT_NS;
}
// ROCE_SUPERVISION_END

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

int roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
    // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes,
    //        flag_stride, slots, paths}
    if (world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0) {
        return -1;
    }
    // Reject a layout whose arithmetic would wrap; the caller sizes slots from
    // configuration, so a wrapped region must fail here rather than at the NIC.
    uint64_t recv_bytes, send_bytes, flag_bytes, flag_off, send_off, ctrl_off, total;
    if (slot_bytes > ((uint64_t)1 << 40) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
        __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS * ROCE_LAYOUT_PATHS,
                               (uint64_t)ROCE_FLAG_STRIDE, &flag_bytes) ||
        __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
        __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
        __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_FLAG_STRIDE, &total)) {
        return -1;
    }
    uint64_t recv_off = 0;
    flag_off = recv_off + recv_bytes;
    send_off = flag_off + flag_bytes;
    out[0] = recv_off;
    out[1] = flag_off;
    out[2] = send_off;
    out[3] = ctrl_off;
    out[4] = total;
    out[5] = ROCE_FLAG_STRIDE;
    out[6] = ROCE_SLOTS;
    out[7] = ROCE_LAYOUT_PATHS;
    return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_blob_t); }

static int open_hca(roce_ctx_t *c, int h, const char *name) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (list == NULL) {
        set_err(c, "ibv_get_device_list", errno);
        return -1;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (dev == NULL) {
        ibv_free_device_list(list);
        snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
        return -1;
    }
    roce_hca_t *hca = &c->hca[h];
    hca->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (hca->ctx == NULL) {
        set_err(c, "ibv_open_device", errno);
        return -1;
    }
    struct ibv_port_attr port;
    if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
        set_err(c, "ibv_query_port", errno);
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
        return -1;
    }
    hca->lid = port.lid;
    hca->mtu = port.active_mtu;
    if (ibv_query_gid(hca->ctx, ROCE_PORT, c->gid_index, &hca->gid) != 0) {
        set_err(c, "ibv_query_gid", errno);
        return -1;
    }
    hca->pd = ibv_alloc_pd(hca->ctx);
    if (hca->pd == NULL) {
        set_err(c, "ibv_alloc_pd", errno);
        return -1;
    }
    hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (hca->mr == NULL) {
        set_err(c, "ibv_reg_mr(pinned region)", errno);
        return -1;
    }
    hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH * ROCE_MAX_PEERS, NULL, NULL, 0);
    if (hca->cq == NULL) {
        set_err(c, "ibv_create_cq", errno);
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = hca->cq;
        attr.recv_cq = hca->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = ROCE_SEND_DEPTH;
        attr.cap.max_recv_wr = 1;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.cap.max_inline_data = 16;
        hca->qp[p] = ibv_create_qp(hca->pd, &attr);
        if (hca->qp[p] == NULL) {
            set_err(c, "ibv_create_qp", errno);
            return -1;
        }
        struct ibv_qp_attr init;
        memset(&init, 0, sizeof(init));
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = ROCE_PORT;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        int rc = ibv_modify_qp(hca->qp[p], &init,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc != 0) {
            set_err(c, "ibv_modify_qp(INIT)", rc);
            return -1;
        }
    }
    return 0;
}

roce_ctx_t *roce_create(int world, int rank, const char *const *hca_names, int n_hca,
                        int gid_index, void *region, uint64_t region_bytes,
                        uint64_t slot_bytes, int opposite_paths,
                        const uint8_t *peer_hca_map,
                        uint64_t peer_hca_count, char *err, uint64_t err_len) {
    uint64_t layout[8];
    if (roce_layout(world, slot_bytes, layout) != 0 || layout[4] > region_bytes ||
        rank < 0 || rank >= world || n_hca < ROCE_LAYOUT_PATHS ||
        n_hca > ROCE_MAX_HCAS ||
        (opposite_paths != ROCE_LAYOUT_PATHS && opposite_paths != ROCE_MAX_PATHS) ||
        (opposite_paths == ROCE_MAX_PATHS &&
         ((world != 2 && world != 4) || n_hca != 4)) ||
        peer_hca_map == NULL ||
        peer_hca_count != (uint64_t)world * ROCE_MAX_PATHS) {
        snprintf(err, err_len, "invalid roce runtime geometry");
        return NULL;
    }
    roce_ctx_t *c = calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    c->world = world;
    c->rank = rank;
    c->n_hca = n_hca;
    c->gid_index = gid_index;
    c->region = region;
    c->region_bytes = region_bytes;
    c->slot_bytes = slot_bytes;
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    c->opposite_paths = (uint32_t)opposite_paths;
    c->two_wave_threshold_bytes = ROCE_DEFAULT_TWO_WAVE_THRESHOLD_BYTES;
    c->wave_mode = opposite_paths == ROCE_MAX_PATHS
                       ? ROCE_WAVE_MODE_BALANCED32
                       : ROCE_WAVE_MODE_TWO;
    const char *wave_mode_text = getenv("B12X_ROCE_WAVE_MODE");
    if (wave_mode_text != NULL && wave_mode_text[0] != '\0') {
        if (opposite_paths == ROCE_MAX_PATHS &&
            strcmp(wave_mode_text, "balanced32") == 0) {
            c->wave_mode = ROCE_WAVE_MODE_BALANCED32;
        } else if (opposite_paths == ROCE_MAX_PATHS) {
            snprintf(err, err_len,
                     "B12X_ROCE_OPPOSITE_PATHS=4 requires "
                     "B12X_ROCE_WAVE_MODE=balanced32 when the wave mode is set");
            roce_destroy(c);
            return NULL;
        } else if (strcmp(wave_mode_text, "two") == 0) {
            c->wave_mode = ROCE_WAVE_MODE_TWO;
        } else if (strcmp(wave_mode_text, "mixed2") == 0) {
            c->wave_mode = ROCE_WAVE_MODE_MIXED_TWO;
        } else if (strcmp(wave_mode_text, "opposite_first") == 0) {
            c->wave_mode = ROCE_WAVE_MODE_OPPOSITE_FIRST;
        } else if (strcmp(wave_mode_text, "strict3") == 0) {
            c->wave_mode = ROCE_WAVE_MODE_STRICT_THREE;
        } else {
            snprintf(err, err_len,
                     "B12X_ROCE_WAVE_MODE must be 'two', 'mixed2', "
                     "'opposite_first', or 'strict3'");
            roce_destroy(c);
            return NULL;
        }
    }
    const char *threshold_text = getenv("B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES");
    if (threshold_text != NULL && threshold_text[0] != '\0') {
        char *end = NULL;
        errno = 0;
        unsigned long long value = strtoull(threshold_text, &end, 10);
        if (errno != 0 || end == threshold_text || *end != '\0' ||
            value > UINT32_MAX || (value != 0 && value % 16u != 0)) {
            snprintf(err, err_len,
                     "B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES must be zero or a "
                     "16-byte-aligned integer no larger than %u",
                     UINT32_MAX);
            roce_destroy(c);
            return NULL;
        }
        c->two_wave_threshold_bytes = (uint32_t)value;
    }
    c->wait_timeout_ns = ROCE_DEFAULT_WAIT_TIMEOUT_NS;
    c->stall_report_ns = roce_stall_report_ns(ROCE_DEFAULT_WAIT_TIMEOUT_NS);
    c->forward_window_bytes = ROCE_DEFAULT_FORWARD_WINDOW_BYTES;
    c->forward_chunk_bytes = ROCE_DEFAULT_FORWARD_CHUNK_BYTES;
    const char *const forward_names[2] = {"B12X_ROCE_FORWARD_WINDOW_BYTES",
                                          "B12X_ROCE_FORWARD_CHUNK_BYTES"};
    uint32_t *const forward_values[2] = {&c->forward_window_bytes,
                                         &c->forward_chunk_bytes};
    for (int i = 0; i < 2; i++) {
        const char *text = getenv(forward_names[i]);
        if (text == NULL || text[0] == '\0') {
            continue;
        }
        char *end = NULL;
        errno = 0;
        unsigned long long value = strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX ||
            value % 16u != 0) {
            snprintf(err, err_len,
                     "%s must be a 16-byte-aligned integer no larger than %u",
                     forward_names[i], UINT32_MAX);
            roce_destroy(c);
            return NULL;
        }
        *forward_values[i] = (uint32_t)value;
    }
    if (c->forward_window_bytes != 0 &&
        (c->forward_chunk_bytes == 0 ||
         c->forward_chunk_bytes > c->forward_window_bytes ||
         c->forward_window_bytes / c->forward_chunk_bytes >= ROCE_SEND_DEPTH / 4)) {
        snprintf(err, err_len,
                 "B12X_ROCE_FORWARD_CHUNK_BYTES must be nonzero, no larger than "
                 "the forward window, and allow fewer than %u chunks per window",
                 ROCE_SEND_DEPTH / 4);
        roce_destroy(c);
        return NULL;
    }
    for (int h = 0; h < ROCE_MAX_HCAS; h++) {
        c->direct_peer_by_hca[h] = -1;
        c->direct_path_by_hca[h] = -1;
        c->opposite_peer_by_hca[h] = -1;
        c->opposite_path_by_hca[h] = -1;
    }
    for (int p = 0; p < world; p++) {
        for (int path = 0; path < ROCE_MAX_PATHS; path++) {
            c->peer_hca[p][path] = -1;
            c->remote_hca[p][path] = -1;
        }
        if (p == rank) {
            continue;
        }
        int distance = (p - rank + world) % world;
        int count = world == 2
                        ? opposite_paths
                        : (world == 4 && distance == 2
                               ? opposite_paths
                               : ROCE_LAYOUT_PATHS);
        c->peer_path_count[p] = (uint32_t)count;
        uint32_t seen_hcas = 0;
        for (int path = 0; path < count; path++) {
            int h = (int)peer_hca_map[p * ROCE_MAX_PATHS + path];
            if (h < 0 || h >= n_hca || (seen_hcas & (1u << h)) != 0) {
                snprintf(err, err_len,
                         "rank %d peer %d needs %d distinct HCA indices in [0,%d)",
                         rank, p, count, n_hca);
                roce_destroy(c);
                return NULL;
            }
            seen_hcas |= 1u << h;
            c->peer_hca[p][path] = h;
        }
        for (int path = count; path < ROCE_MAX_PATHS; path++) {
            if (peer_hca_map[p * ROCE_MAX_PATHS + path] != UINT8_MAX) {
                snprintf(err, err_len,
                         "rank %d peer %d publishes an inactive path %d", rank, p, path);
                roce_destroy(c);
                return NULL;
            }
        }
        c->physical_hops[p] = world == 4 && distance == 2 ? 2u : 1u;
    }
    if (opposite_paths == ROCE_MAX_PATHS && world == 2) {
        static const char *const canonical_hcas[4] = {
            "rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1"};
        int canonical = n_hca == 4;
        int peer = 1 - rank;
        for (int h = 0; canonical && h < 4; h++) {
            canonical = strcmp(hca_names[h], canonical_hcas[h]) == 0 &&
                        c->peer_hca[peer][h] == h;
        }
        if (!canonical) {
            snprintf(err, err_len,
                     "four-path TP2 requires the canonical HCA order and "
                     "reciprocal peer-path mapping");
            roce_destroy(c);
            return NULL;
        }
    } else if (opposite_paths == ROCE_MAX_PATHS ||
               c->wave_mode != ROCE_WAVE_MODE_TWO) {
        static const char *const canonical_hcas[4] = {
            "rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1"};
        static const int canonical_map[4][4][ROCE_MAX_PATHS] = {
            {{-1, -1, -1, -1}, {0, 2, -1, -1}, {0, 3, 2, 1}, {1, 3, -1, -1}},
            {{1, 3, -1, -1}, {-1, -1, -1, -1}, {0, 2, -1, -1}, {0, 3, 2, 1}},
            {{1, 2, 3, 0}, {1, 3, -1, -1}, {-1, -1, -1, -1}, {0, 2, -1, -1}},
            {{0, 2, -1, -1}, {1, 2, 3, 0}, {1, 3, -1, -1}, {-1, -1, -1, -1}},
        };
        int canonical = world == 4 && n_hca == 4;
        for (int h = 0; canonical && h < 4; h++) {
            canonical = strcmp(hca_names[h], canonical_hcas[h]) == 0;
        }
        for (int p = 0; canonical && p < world; p++) {
            int count = c->peer_path_count[p];
            for (int path = 0; canonical && path < ROCE_MAX_PATHS; path++) {
                int expected = path < count ? canonical_map[rank][p][path] : -1;
                canonical = c->peer_hca[p][path] == expected;
            }
        }
        if (!canonical) {
            snprintf(err, err_len,
                     "the selected RoCEnante path schedule requires the canonical "
                     "four-rank HCA order and peer-path mapping");
            roce_destroy(c);
            return NULL;
        }
    }
    if (opposite_paths == ROCE_MAX_PATHS && world == 4) {
        int opposite = (rank + 2) % world;
        for (int p = 0; p < world; p++) {
            if (p == rank) {
                continue;
            }
            int is_opposite = p == opposite;
            for (int path = 0; path < (int)c->peer_path_count[p]; path++) {
                int h = c->peer_hca[p][path];
                int *peer_slot = is_opposite ? &c->opposite_peer_by_hca[h]
                                             : &c->direct_peer_by_hca[h];
                int *path_slot = is_opposite ? &c->opposite_path_by_hca[h]
                                             : &c->direct_path_by_hca[h];
                if (*peer_slot != -1) {
                    snprintf(err, err_len,
                             "rank %d HCA %d has multiple %s paths", rank, h,
                             is_opposite ? "opposite" : "direct");
                    roce_destroy(c);
                    return NULL;
                }
                *peer_slot = p;
                *path_slot = path;
            }
        }
        for (int h = 0; h < n_hca; h++) {
            if (c->direct_peer_by_hca[h] < 0 || c->opposite_peer_by_hca[h] < 0) {
                snprintf(err, err_len,
                         "rank %d HCA %d lacks one direct or opposite path", rank, h);
                roce_destroy(c);
                return NULL;
            }
        }
    }
    for (int h = 0; h < n_hca; h++) {
        if (open_hca(c, h, hca_names[h]) != 0) {
            snprintf(err, err_len, "%s", c->err);
            roce_destroy(c);
            return NULL;
        }
    }
    return c;
}

int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {
    if (out_len < sizeof(roce_blob_t)) {
        return -1;
    }
    roce_blob_t blob;
    memset(&blob, 0, sizeof(blob));
    blob.abi_version = ROCE_ABI_VERSION;
    blob.world = (uint32_t)c->world;
    blob.rank = (uint32_t)c->rank;
    blob.n_hca = (uint32_t)c->n_hca;
    blob.layout_paths = ROCE_LAYOUT_PATHS;
    blob.opposite_paths = c->opposite_paths;
    blob.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int h = 0; h < c->n_hca; h++) {
        blob.rkey[h] = c->hca[h].mr->rkey;
        blob.lid[h] = c->hca[h].lid;
        blob.mtu[h] = (uint32_t)c->hca[h].mtu;
        memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
        for (int p = 0; p < c->world; p++) {
            blob.qp_num[h][p] = (p == c->rank) ? 0 : c->hca[h].qp[p]->qp_num;
        }
    }
    memset(blob.peer_hca, UINT8_MAX, sizeof(blob.peer_hca));
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        for (int path = 0; path < (int)c->peer_path_count[p]; path++) {
            blob.peer_hca[p][path] = (uint8_t)c->peer_hca[p][path];
        }
    }
    memcpy(out, &blob, sizeof(blob));
    return 0;
}

static int connect_qp(roce_ctx_t *c, int local_h, int remote_h, int p,
                      const roce_blob_t *peer) {
    roce_hca_t *hca = &c->hca[local_h];
    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[remote_h] < (uint32_t)hca->mtu
                                      ? peer->mtu[remote_h]
                                      : (uint32_t)hca->mtu);
    rtr.dest_qp_num = peer->qp_num[remote_h][c->rank];
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer->lid[remote_h];
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[remote_h], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = 0;
    // A four-rank cycle has no physical link between opposite ranks.
    // Hardware-forwarding rules for opposite-rank paths match UDP source port
    // 65535, which mlx5 derives from flow label 16383, and rewrite only the
    // Ethernet header at the intermediate ConnectX device.  Neighbor QPs use
    // flow label zero and do not match those rules.
    rtr.ah_attr.grh.flow_label =
        (c->world == 4 && ((p - c->rank + c->world) % c->world) == 2)
            ? 16383
            : 0;
    int rc = ibv_modify_qp(hca->qp[p], &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTR)", rc);
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
    rc = ibv_modify_qp(hca->qp[p], &rts,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTS)", rc);
        return -1;
    }
    return 0;
}

int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (blobs_len < sizeof(roce_blob_t) * (uint64_t)c->world) {
        snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
        return -1;
    }
    const roce_blob_t *all = (const roce_blob_t *)blobs;
    for (int p = 0; p < c->world; p++) {
        if (all[p].abi_version != ROCE_ABI_VERSION ||
            all[p].world != (uint32_t)c->world || all[p].rank != (uint32_t)p ||
            all[p].layout_paths != ROCE_LAYOUT_PATHS ||
            all[p].opposite_paths != c->opposite_paths ||
            all[p].n_hca < ROCE_LAYOUT_PATHS || all[p].n_hca > ROCE_MAX_HCAS) {
            snprintf(c->err, sizeof(c->err),
                     "rank %d published incompatible RoCE connection metadata", p);
            return -1;
        }
        if (p == c->rank) {
            continue;
        }
        c->peer_addr[p] = all[p].region_addr;
        uint32_t remote_seen_hcas = 0;
        for (int path = 0; path < (int)c->peer_path_count[p]; path++) {
            int local_h = c->peer_hca[p][path];
            int remote_h = (int)all[p].peer_hca[c->rank][path];
            if (remote_h < 0 || remote_h >= (int)all[p].n_hca ||
                (remote_seen_hcas & (1u << remote_h)) != 0 ||
                all[p].region_addr == 0 || all[p].rkey[remote_h] == 0 ||
                all[p].qp_num[remote_h][c->rank] == 0) {
                snprintf(c->err, sizeof(c->err),
                         "rank %d path %d published incomplete RoCE connection metadata",
                         p, path);
                return -1;
            }
            remote_seen_hcas |= 1u << remote_h;
            c->remote_hca[p][path] = remote_h;
            c->peer_rkey[p][path] = all[p].rkey[remote_h];
            c->remote_qp[p][path] = all[p].qp_num[remote_h][c->rank];
            if (connect_qp(c, local_h, remote_h, p, &all[p]) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int drain_cq(roce_ctx_t *c, int h) {
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
    if (n < 0) {
        set_err(c, "ibv_poll_cq", errno);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        int notice = (wc[i].wr_id & ROCE_NOTICE_WR) != 0;
        uint64_t wr_id = wc[i].wr_id & ~ROCE_NOTICE_WR;
        int peer = (int)(wr_id / ROCE_MAX_PATHS);
        int path = (int)(wr_id % ROCE_MAX_PATHS);
        if (peer < 0 || peer >= c->world || peer == c->rank ||
            path < 0 || path >= (int)c->peer_path_count[peer] ||
            c->peer_hca[peer][path] != h) {
            snprintf(c->err, sizeof(c->err),
                     "RDMA completion has invalid path identifier %llu",
                     (unsigned long long)wc[i].wr_id);
            return -1;
        }
        if (wc[i].status != IBV_WC_SUCCESS) {
            atomic_fetch_add(&c->completion_errors[peer][path], 1);
            if (notice) {
                snprintf(c->err, sizeof(c->err),
                         "rank %d did not acknowledge a peer check on path %d (%s, "
                         "vendor_err 0x%x) while this rank waited at sequence %u; its "
                         "queue pair is gone or unreachable",
                         peer, path, ibv_wc_status_str(wc[i].status),
                         wc[i].vendor_err, c->last_seq);
            } else {
                snprintf(c->err, sizeof(c->err),
                         "RDMA write to rank %d path %d failed: %s (vendor_err 0x%x, seq %u)",
                         peer, path, ibv_wc_status_str(wc[i].status), wc[i].vendor_err,
                         c->last_seq);
            }
            return -1;
        }
        c->hca[h].outstanding[peer] -= 1;
        if (notice) {
            continue;
        }
        atomic_fetch_add(&c->send_completions[peer][path], 1);
        c->writes_completed += 1;
    }
    return 0;
}

// Posts one signaled 4-byte notice into ``peer``'s control record on
// ``path``. Its completion confirms that the peer's queue pair still
// acknowledges writes. A failed post leaves c->err unchanged, so a caller
// that is already failing keeps its original error.
static int post_notice(roce_ctx_t *c, int peer, int path, uint32_t value) {
    int h = c->peer_hca[peer][path];
    if (h < 0 || h >= c->n_hca) {
        return -1;
    }
    roce_hca_t *hca = &c->hca[h];
    if (hca->qp[peer] == NULL || hca->outstanding[peer] >= ROCE_SEND_DEPTH - 1) {
        return -1;
    }
    uint32_t value_copy = value;
    struct ibv_sge sge = {
        .addr = (uint64_t)(uintptr_t)&value_copy,
        .length = 4,
        .lkey = 0,
    };
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = ROCE_NOTICE_WR | ((uint64_t)peer * ROCE_MAX_PATHS + (uint64_t)path);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    wr.wr.rdma.remote_addr = c->peer_addr[peer] + c->ctrl_off +
                             4u * (uint64_t)(ROCE_CTRL_NOTICE + c->rank);
    wr.wr.rdma.rkey = c->peer_rkey[peer][path];
    struct ibv_send_wr *bad = NULL;
    if (ibv_post_send(hca->qp[peer], &wr, &bad) != 0) {
        return -1;
    }
    hca->outstanding[peer] += 1;
    atomic_fetch_add(&c->notices_posted, 1);
    return 0;
}

static uint64_t flag_remote_addr(const roce_ctx_t *c, int peer, int path,
                                 uint32_t slot) {
    uint32_t flag_row = path < ROCE_LAYOUT_PATHS ? (uint32_t)c->rank
                                                  : (uint32_t)peer;
    uint32_t flag_column = path < ROCE_LAYOUT_PATHS
                               ? (uint32_t)path
                               : (uint32_t)(path - ROCE_LAYOUT_PATHS);
    return c->peer_addr[peer] + c->flag_off +
           (((uint64_t)flag_row * ROCE_SLOTS + slot) * ROCE_LAYOUT_PATHS +
            flag_column) * ROCE_FLAG_STRIDE;
}

static int forward_windowed(const roce_ctx_t *c, int peer, uint32_t stripe_bytes) {
    return c->forward_window_bytes != 0 && c->physical_hops[peer] > 1 &&
           stripe_bytes > c->forward_chunk_bytes;
}

// Posts one signaled payload chunk of a forwarded stripe.
static int post_chunk(roce_ctx_t *c, uint32_t slot, uint8_t *send, int peer,
                      int path, uint32_t offset, uint32_t bytes) {
    int h = c->peer_hca[peer][path];
    roce_hca_t *hca = &c->hca[h];
    struct ibv_sge sge = {
        .addr = (uint64_t)(uintptr_t)(send + offset),
        .length = bytes,
        .lkey = hca->mr->lkey,
    };
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)peer * ROCE_MAX_PATHS + (uint64_t)path;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr =
        c->peer_addr[peer] + c->recv_off +
        ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes + offset;
    wr.wr.rdma.rkey = c->peer_rkey[peer][path];
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(hca->qp[peer], &wr, &bad);
    if (rc != 0) {
        atomic_fetch_add(&c->completion_errors[peer][path], 1);
        set_err(c, "ibv_post_send", rc);
        return -1;
    }
    atomic_fetch_add(&c->payload_writes[peer][path], 1);
    atomic_fetch_add(&c->payload_bytes[peer][path], bytes);
    atomic_fetch_add(&c->forward_chunks, 1);
    hca->outstanding[peer] += 1;
    return 0;
}

// Posts the stripe's sequence flag after its last chunk on the same QP.
static int post_stream_flag(roce_ctx_t *c, uint32_t seq, uint32_t slot,
                            int peer, int path) {
    int h = c->peer_hca[peer][path];
    roce_hca_t *hca = &c->hca[h];
    uint32_t seq_copy = seq;
    struct ibv_sge sge = {
        .addr = (uint64_t)(uintptr_t)&seq_copy,
        .length = 4,
        .lkey = 0,
    };
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)peer * ROCE_MAX_PATHS + (uint64_t)path;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    wr.wr.rdma.remote_addr = flag_remote_addr(c, peer, path, slot);
    wr.wr.rdma.rkey = c->peer_rkey[peer][path];
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(hca->qp[peer], &wr, &bad);
    if (rc != 0) {
        atomic_fetch_add(&c->completion_errors[peer][path], 1);
        set_err(c, "ibv_post_send", rc);
        return -1;
    }
    atomic_fetch_add(&c->flag_writes[peer][path], 1);
    hca->outstanding[peer] += 1;
    return 0;
}

// Completes every registered forwarded stripe. Streams advance round-robin so
// both forwarded paths of an opposite peer stay busy; a QP posts another chunk
// only while its unacknowledged chunks fit the forward window.
static int pump_streams(roce_ctx_t *c) {
    uint32_t window_chunks = c->forward_window_bytes / c->forward_chunk_bytes;
    int active = c->n_streams;
    while (active > 0) {
        int progressed = 0;
        for (int i = 0; i < c->n_streams; i++) {
            int peer = c->streams[i].peer;
            int path = c->streams[i].path;
            if (peer < 0) {
                continue;
            }
            roce_hca_t *hca = &c->hca[c->peer_hca[peer][path]];
            if (hca->outstanding[peer] >= window_chunks) {
                continue;
            }
            if (c->streams[i].next < c->streams[i].end) {
                uint32_t bytes = c->streams[i].end - c->streams[i].next;
                if (bytes > c->forward_chunk_bytes) {
                    bytes = c->forward_chunk_bytes;
                }
                if (post_chunk(c, c->streams[i].slot, c->streams[i].send, peer,
                               path, c->streams[i].next, bytes) != 0) {
                    return -1;
                }
                c->streams[i].next += bytes;
            } else {
                if (post_stream_flag(c, c->streams[i].seq, c->streams[i].slot,
                                     peer, path) != 0) {
                    return -1;
                }
                c->streams[i].peer = -1;
                active -= 1;
            }
            progressed = 1;
        }
        if (progressed) {
            continue;
        }
        for (int i = 0; i < c->n_streams; i++) {
            if (c->streams[i].peer >= 0 &&
                drain_cq(c, c->peer_hca[c->streams[i].peer][c->streams[i].path]) != 0) {
                return -1;
            }
        }
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            snprintf(c->err, sizeof(c->err),
                     "RoCE proxy stopped with %d forwarded stripes pending", active);
            return -1;
        }
    }
    c->n_streams = 0;
    return 0;
}

static int post_path(roce_ctx_t *c, uint32_t seq, uint32_t slot, uint8_t *send,
                     int peer, int path, uint32_t stripe_offset,
                     uint32_t stripe_bytes) {
    int h = c->peer_hca[peer][path];
    roce_hca_t *hca = &c->hca[h];
    if (forward_windowed(c, peer, stripe_bytes)) {
        if (c->n_streams >= ROCE_MAX_STREAMS) {
            snprintf(c->err, sizeof(c->err), "forwarded stripe table is full");
            return -1;
        }
        c->streams[c->n_streams].seq = seq;
        c->streams[c->n_streams].slot = slot;
        c->streams[c->n_streams].send = send;
        c->streams[c->n_streams].peer = peer;
        c->streams[c->n_streams].path = path;
        c->streams[c->n_streams].next = stripe_offset;
        c->streams[c->n_streams].end = stripe_offset + stripe_bytes;
        c->n_streams += 1;
        return 0;
    }
    // Each QP gets one signaled completion per operation.  Keep its queue
    // below one quarter of the configured depth.
    while (hca->outstanding[peer] >= ROCE_SEND_DEPTH / 4) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            snprintf(c->err, sizeof(c->err),
                     "RoCE proxy stopped with %u writes outstanding to rank %d path %d",
                     hca->outstanding[peer], peer, path);
            return -1;
        }
    }
    uint32_t seq_copy = seq;
    uint64_t remote = c->peer_addr[peer];
    struct ibv_sge data_sge = {
        .addr = (uint64_t)(uintptr_t)(send + stripe_offset),
        .length = stripe_bytes,
        .lkey = hca->mr->lkey,
    };
    struct ibv_sge flag_sge = {
        .addr = (uint64_t)(uintptr_t)&seq_copy,
        .length = 4,
        .lkey = 0,
    };
    struct ibv_send_wr flag_wr;
    memset(&flag_wr, 0, sizeof(flag_wr));
    flag_wr.wr_id = (uint64_t)peer * ROCE_MAX_PATHS + (uint64_t)path;
    flag_wr.sg_list = &flag_sge;
    flag_wr.num_sge = 1;
    flag_wr.opcode = IBV_WR_RDMA_WRITE;
    flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    flag_wr.wr.rdma.remote_addr = flag_remote_addr(c, peer, path, slot);
    flag_wr.wr.rdma.rkey = c->peer_rkey[peer][path];
    struct ibv_send_wr data_wr;
    memset(&data_wr, 0, sizeof(data_wr));
    data_wr.wr_id = flag_wr.wr_id;
    data_wr.next = &flag_wr;
    data_wr.sg_list = &data_sge;
    data_wr.num_sge = 1;
    data_wr.opcode = IBV_WR_RDMA_WRITE;
    data_wr.send_flags = 0;
    data_wr.wr.rdma.remote_addr =
        remote + c->recv_off +
        ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes + stripe_offset;
    data_wr.wr.rdma.rkey = c->peer_rkey[peer][path];
    struct ibv_send_wr *first_wr = stripe_bytes == 0 ? &flag_wr : &data_wr;
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(hca->qp[peer], first_wr, &bad);
    if (rc != 0) {
        atomic_fetch_add(&c->completion_errors[peer][path], 1);
        set_err(c, "ibv_post_send", rc);
        return -1;
    }
    if (stripe_bytes != 0) {
        atomic_fetch_add(&c->payload_writes[peer][path], 1);
        atomic_fetch_add(&c->payload_bytes[peer][path], stripe_bytes);
    }
    atomic_fetch_add(&c->flag_writes[peer][path], 1);
    hca->outstanding[peer] += 1;
    return 0;
}

static int post_peer(roce_ctx_t *c, uint32_t seq, uint32_t slot, uint8_t *send,
                     int peer, const uint32_t stripe_offset[ROCE_MAX_PATHS],
                     const uint32_t stripe_bytes[ROCE_MAX_PATHS]) {
    for (int path = 0; path < (int)c->peer_path_count[peer]; path++) {
        if (post_path(c, seq, slot, send, peer, path, stripe_offset[path],
                      stripe_bytes[path]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int drain_path(roce_ctx_t *c, int peer, int path) {
    int h = c->peer_hca[peer][path];
    while (c->hca[h].outstanding[peer] != 0) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            snprintf(c->err, sizeof(c->err),
                     "RoCE proxy stopped while draining rank %d path %d", peer, path);
            return -1;
        }
    }
    return 0;
}

static int drain_peer_paths(roce_ctx_t *c, int peer) {
    for (;;) {
        int pending = 0;
        for (int path = 0; path < (int)c->peer_path_count[peer]; path++) {
            int h = c->peer_hca[peer][path];
            if (c->hca[h].outstanding[peer] != 0) {
                pending = 1;
                if (drain_cq(c, h) != 0) {
                    return -1;
                }
            }
        }
        if (!pending) {
            return 0;
        }
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            snprintf(c->err, sizeof(c->err),
                     "RoCE proxy stopped while draining direct paths to rank %d", peer);
            return -1;
        }
    }
}

static int mixed_two_wave(roce_ctx_t *c, uint32_t seq, uint32_t slot,
                          uint8_t *send,
                          const uint32_t stripe_offset[ROCE_MAX_PATHS],
                          const uint32_t stripe_bytes[ROCE_MAX_PATHS]) {
    // Bits enumerate (peer, path) in peer-rank order while omitting the local
    // rank.  Each mask contains three origin QPs and balances every directed
    // physical edge across the two waves for the canonical four-rank mapping.
    static const uint8_t masks[4][2] = {
        {0x0bu, 0x34u},
        {0x2cu, 0x13u},
        {0x31u, 0x0eu},
        {0x07u, 0x38u},
    };
    for (int wave = 0; wave < 2; wave++) {
        int bit = 0;
        for (int peer = 0; peer < c->world; peer++) {
            if (peer == c->rank) {
                continue;
            }
            for (int path = 0; path < ROCE_LAYOUT_PATHS; path++, bit++) {
                if ((masks[c->rank][wave] & (1u << bit)) != 0 &&
                    post_path(c, seq, slot, send, peer, path,
                              stripe_offset[path], stripe_bytes[path]) != 0) {
                    return -1;
                }
            }
        }
        bit = 0;
        for (int peer = 0; peer < c->world; peer++) {
            if (peer == c->rank) {
                continue;
            }
            for (int path = 0; path < ROCE_LAYOUT_PATHS; path++, bit++) {
                if ((masks[c->rank][wave] & (1u << bit)) != 0 &&
                    drain_path(c, peer, path) != 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int strict_three_wave(roce_ctx_t *c, uint32_t seq, uint32_t slot,
                             uint8_t *send,
                             const uint32_t stripe_offset[ROCE_MAX_PATHS],
                             const uint32_t stripe_bytes[ROCE_MAX_PATHS]) {
    // Wave zero contains all four direct-link origins.  Waves one and two each
    // contain one reciprocal opposite-rank path, with the path assignment
    // reversed between the two diagonal rank pairs.
    static const uint8_t masks[4][3] = {
        {0x33u, 0x04u, 0x08u},
        {0x0fu, 0x20u, 0x10u},
        {0x3cu, 0x01u, 0x02u},
        {0x33u, 0x08u, 0x04u},
    };
    for (int wave = 0; wave < 3; wave++) {
        int bit = 0;
        for (int peer = 0; peer < c->world; peer++) {
            if (peer == c->rank) {
                continue;
            }
            for (int path = 0; path < ROCE_LAYOUT_PATHS; path++, bit++) {
                if ((masks[c->rank][wave] & (1u << bit)) != 0 &&
                    post_path(c, seq, slot, send, peer, path,
                              stripe_offset[path], stripe_bytes[path]) != 0) {
                    return -1;
                }
            }
        }
        bit = 0;
        for (int peer = 0; peer < c->world; peer++) {
            if (peer == c->rank) {
                continue;
            }
            for (int path = 0; path < ROCE_LAYOUT_PATHS; path++, bit++) {
                if ((masks[c->rank][wave] & (1u << bit)) != 0 &&
                    drain_path(c, peer, path) != 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

static void split_stripes(uint32_t nbytes, int count,
                          uint32_t stripe_offset[ROCE_MAX_PATHS],
                          uint32_t stripe_bytes[ROCE_MAX_PATHS]) {
    uint32_t packs = nbytes / 16u;
    uint32_t base = packs / (uint32_t)count;
    uint32_t remainder = packs % (uint32_t)count;
    uint32_t offset = 0;
    for (int path = 0; path < ROCE_MAX_PATHS; path++) {
        uint32_t path_packs = path < count
                                  ? base + ((uint32_t)path < remainder ? 1u : 0u)
                                  : 0u;
        stripe_offset[path] = offset;
        stripe_bytes[path] = path_packs * 16u;
        offset += stripe_bytes[path];
    }
}

static int balanced32_post(roce_ctx_t *c, uint32_t seq, uint32_t slot,
                           uint8_t *send,
                           const uint32_t half_offset[ROCE_MAX_PATHS],
                           const uint32_t half_bytes[ROCE_MAX_PATHS],
                           const uint32_t quarter_offset[ROCE_MAX_PATHS],
                           const uint32_t quarter_bytes[ROCE_MAX_PATHS]) {
    // Queue one path on every HCA before queueing a second. Rank and generation
    // rotate the first HCA; generation parity alternates direct and opposite
    // priority without a completion boundary or feedback loop.
    int start_hca = (c->rank + (int)(seq & 3u)) % ROCE_MAX_HCAS;
    int direct_first = (seq & 1u) == 0;
    for (int round = 0; round < 2; round++) {
        int post_direct = round == 0 ? direct_first : !direct_first;
        for (int ordinal = 0; ordinal < ROCE_MAX_HCAS; ordinal++) {
            int h = (start_hca + ordinal) % ROCE_MAX_HCAS;
            int peer = post_direct ? c->direct_peer_by_hca[h]
                                   : c->opposite_peer_by_hca[h];
            int path = post_direct ? c->direct_path_by_hca[h]
                                   : c->opposite_path_by_hca[h];
            const uint32_t *offset = post_direct ? half_offset : quarter_offset;
            const uint32_t *bytes = post_direct ? half_bytes : quarter_bytes;
            if (peer < 0 || path < 0 || c->peer_hca[peer][path] != h) {
                snprintf(c->err, sizeof(c->err),
                         "balanced path schedule is incomplete for HCA %d", h);
                return -1;
            }
            if (post_path(c, seq, slot, send, peer, path, offset[path],
                          bytes[path]) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int post_op(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    uint32_t slot = seq & 1u;
    uint8_t *send = c->region + c->send_off + (size_t)slot * c->slot_bytes;
    if (nbytes == 0 || nbytes > c->slot_bytes || (nbytes % 16u) != 0) {
        snprintf(c->err, sizeof(c->err),
                 "RoCE payload bytes must be a positive 16-byte multiple within the slot");
        return -1;
    }
    uint32_t stripe_offset[ROCE_MAX_PATHS] = {0};
    uint32_t stripe_bytes[ROCE_MAX_PATHS] = {0};
    uint32_t quarter_offset[ROCE_MAX_PATHS] = {0};
    uint32_t quarter_bytes[ROCE_MAX_PATHS] = {0};
    split_stripes(nbytes, ROCE_LAYOUT_PATHS, stripe_offset, stripe_bytes);
    split_stripes(nbytes, ROCE_MAX_PATHS, quarter_offset, quarter_bytes);
    if (c->world == 2 && c->opposite_paths == ROCE_MAX_PATHS) {
        int peer = 1 - c->rank;
        if (post_peer(c, seq, slot, send, peer, quarter_offset,
                      quarter_bytes) != 0) {
            return -1;
        }
        goto posted;
    }
    if (c->opposite_paths == ROCE_MAX_PATHS) {
        if (balanced32_post(c, seq, slot, send, stripe_offset, stripe_bytes,
                            quarter_offset, quarter_bytes) != 0) {
            return -1;
        }
        goto posted;
    }
    int two_wave = c->world == 4 && c->n_hca == 4 &&
                   c->two_wave_threshold_bytes != 0 &&
                   nbytes >= c->two_wave_threshold_bytes;
    if (two_wave) {
        atomic_fetch_add(&c->two_wave_activations, 1);
        if (c->wave_mode == ROCE_WAVE_MODE_MIXED_TWO) {
            if (mixed_two_wave(c, seq, slot, send, stripe_offset,
                               stripe_bytes) != 0) {
                return -1;
            }
        } else if (c->wave_mode == ROCE_WAVE_MODE_STRICT_THREE) {
            if (strict_three_wave(c, seq, slot, send, stripe_offset,
                                  stripe_bytes) != 0) {
                return -1;
            }
        } else if (c->wave_mode == ROCE_WAVE_MODE_OPPOSITE_FIRST) {
            int opposite = (c->rank + 2) % c->world;
            if (post_peer(c, seq, slot, send, opposite, stripe_offset,
                          stripe_bytes) != 0 ||
                drain_peer_paths(c, opposite) != 0) {
                return -1;
            }
            for (int peer = 0; peer < c->world; peer++) {
                int distance = (peer - c->rank + c->world) % c->world;
                if ((distance == 1 || distance == 3) &&
                    post_peer(c, seq, slot, send, peer, stripe_offset,
                              stripe_bytes) != 0) {
                    return -1;
                }
            }
        } else {
            // Direct-link QPs complete before the two hardware-forwarded QPs
            // are submitted, preventing both traffic classes from competing
            // in the same ConnectX reliability window for larger payloads.
            for (int peer = 0; peer < c->world; peer++) {
                int distance = (peer - c->rank + c->world) % c->world;
                if (distance == 1 || distance == 3) {
                    if (post_peer(c, seq, slot, send, peer, stripe_offset,
                                  stripe_bytes) != 0) {
                        return -1;
                    }
                }
            }
            for (int peer = 0; peer < c->world; peer++) {
                int distance = (peer - c->rank + c->world) % c->world;
                if ((distance == 1 || distance == 3) &&
                    drain_peer_paths(c, peer) != 0) {
                    return -1;
                }
            }
            int opposite = (c->rank + 2) % c->world;
            if (post_peer(c, seq, slot, send, opposite, stripe_offset,
                          stripe_bytes) != 0) {
                return -1;
            }
        }
    } else {
        for (int peer = 0; peer < c->world; peer++) {
            if (peer != c->rank &&
                post_peer(c, seq, slot, send, peer, stripe_offset,
                          stripe_bytes) != 0) {
                return -1;
            }
        }
    }
posted:
    if (c->n_streams != 0 && pump_streams(c) != 0) {
        return -1;
    }
    c->ops_posted += 1;
    for (int h = 0; h < c->n_hca; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return 0;
}

// Flag of ``peer``'s ``path`` for ``slot`` in this rank's region, at the
// address the kernel polls: opposite-path lanes 2/3 of four-path mode live in
// the receiver-local row.
static uint32_t local_flag(const roce_ctx_t *c, int peer, int path, uint32_t slot) {
    uint32_t row = path < ROCE_LAYOUT_PATHS ? (uint32_t)peer : (uint32_t)c->rank;
    uint32_t column = path < ROCE_LAYOUT_PATHS ? (uint32_t)path
                                               : (uint32_t)(path - ROCE_LAYOUT_PATHS);
    const uint32_t *flag = (const uint32_t *)(c->region + c->flag_off +
                                              (((uint64_t)row * ROCE_SLOTS + slot) *
                                                   ROCE_LAYOUT_PATHS +
                                               column) *
                                                  ROCE_FLAG_STRIDE);
    return __atomic_load_n(flag, __ATOMIC_ACQUIRE);
}

// Handles notices that peers wrote into this rank's control record. A peer
// that stopped its runtime stops this one; a waiting peer is logged with this
// rank's doorbell, posting and completion state, which tells whether this
// rank's GPU work, its proxy or the network held the peer up.
static int read_notices(roce_ctx_t *c, volatile uint32_t *ctrl) {
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        uint32_t notice = __atomic_load_n(&ctrl[ROCE_CTRL_NOTICE + p], __ATOMIC_ACQUIRE);
        if (notice == c->notice_seen[p]) {
            continue;
        }
        c->notice_seen[p] = notice;
        atomic_fetch_add(&c->notices_received, 1);
        uint32_t doorbell = ctrl[0];
        uint32_t seq = roce_notice_seq(notice & ROCE_NOTICE_SEQ, doorbell);
        if ((notice & ROCE_NOTICE_ABORT) != 0) {
            snprintf(c->err, sizeof(c->err),
                     "rank %d stopped its RoCE runtime at sequence %u while this rank's "
                     "doorbell was %u",
                     p, seq, doorbell);
            return -1;
        }
        roce_log(c,
                 "rank %d reports waiting for this rank at sequence %u; this rank's "
                 "doorbell is %u, posted %u, completed %u",
                 p, seq, doorbell, c->last_seq, ctrl[ROCE_CTRL_DONE]);
    }
    return 0;
}

static void end_watch(roce_ctx_t *c, uint64_t ended_ns) {
    if (c->watching && c->stall_reported) {
        uint64_t waited = ended_ns - c->watch_posted_ns;
        atomic_fetch_add(&c->stalls_resolved, 1);
        roce_raise_max(&c->longest_stall_ns, waited);
        roce_log(c, "the wait at sequence %u ended within %.1f s; serving continues",
                 c->watched_seq, roce_seconds(waited));
    }
    c->watching = 0;
}

// Records the time since the proxy loop last made progress (a supervision
// tick or a completed post). A long gap means this thread was not scheduled
// or one post waited for completions, either of which delays the peers.
static void note_progress(roce_ctx_t *c, volatile uint32_t *ctrl, uint64_t now) {
    if (c->last_tick_ns != 0 && now > c->last_tick_ns) {
        uint64_t gap = now - c->last_tick_ns;
        roce_raise_max(&c->longest_loop_gap_ns, gap);
        if (gap >= ROCE_LOOP_GAP_REPORT_NS) {
            roce_log(c, "the proxy loop made no progress for %.1f s (doorbell %u, posted %u)",
                     roce_seconds(gap), ctrl[0], c->last_seq);
        }
    }
    c->last_tick_ns = now;
}

// Supervises the collective whose doorbell this thread posted last. Returns
// -1 with c->err set when every wait on this rank must stop.
static int supervise(roce_ctx_t *c, volatile uint32_t *ctrl, uint64_t now) {
    note_progress(c, ctrl, now);
    if (read_notices(c, ctrl) != 0) {
        return -1;
    }
    uint32_t seq = c->last_seq;
    if (c->watching && c->watched_seq != seq) {
        // A later doorbell means the watched collective completed before it.
        end_watch(c, c->posted_ns);
    }
    if (ctrl[ROCE_CTRL_FAILED] != 0) {
        // Only the device poll bound stops a wait without the abort word; the
        // runtime is poisoned either way, so the peers are told to stop too.
        snprintf(c->err, sizeof(c->err),
                 "the kernel stopped waiting for rank %u at sequence %u after its device "
                 "poll bound (B12X_ROCE_SPIN_LIMIT) without a host verdict",
                 ctrl[ROCE_CTRL_MISSING_PEER], ctrl[ROCE_CTRL_STOPPED_SEQ]);
        return -1;
    }
    if (__atomic_load_n(&ctrl[ROCE_CTRL_DONE], __ATOMIC_ACQUIRE) == seq) {
        end_watch(c, now);
        return 0;
    }
    if (!c->watching) {
        c->watching = 1;
        c->watched_seq = seq;
        c->stall_reported = 0;
        c->watch_posted_ns = c->posted_ns;
        c->next_probe_ns = 0;
    }
    uint64_t elapsed = now > c->watch_posted_ns ? now - c->watch_posted_ns : 0;
    if (elapsed < c->stall_report_ns) {
        return 0;
    }
    char pending[384];
    size_t used = 0;
    pending[0] = '\0';
    uint32_t late_peers = 0;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        for (int path = 0; path < (int)c->peer_path_count[p]; path++) {
            uint32_t current = local_flag(c, p, path, seq & 1u);
            uint32_t other = local_flag(c, p, path, (seq + 1u) & 1u);
            int state = roce_flag_state(seq, current, other);
            if (state == ROCE_FLAG_INCONSISTENT) {
                snprintf(c->err, sizeof(c->err),
                         "rank %d path %d flags hold %u and %u while this rank waits at "
                         "sequence %u; the ranks disagree about the collective sequence",
                         p, path, current, other, seq);
                return -1;
            }
            if (state == ROCE_FLAG_PENDING) {
                late_peers |= 1u << p;
                if (used < sizeof(pending)) {
                    int wrote = snprintf(pending + used, sizeof(pending) - used,
                                         "%srank %d path %d (flag %u)", used ? ", " : "",
                                         p, path, current);
                    used += wrote > 0 ? (size_t)wrote : 0;
                }
            }
        }
    }
    if (!c->stall_reported) {
        c->stall_reported = 1;
        atomic_fetch_add(&c->stalls, 1);
        if (late_peers != 0) {
            roce_log(c,
                     "waited %.1f s at sequence %u for %s; waiting up to %.0f s while "
                     "their queue pairs acknowledge checks",
                     roce_seconds(elapsed), seq, pending, roce_seconds(c->wait_timeout_ns));
        } else {
            roce_log(c,
                     "waited %.1f s at sequence %u although every peer flag for it is in "
                     "host memory; the kernel has not observed them",
                     roce_seconds(elapsed), seq);
        }
    }
    if (elapsed >= c->wait_timeout_ns) {
        if (late_peers != 0) {
            snprintf(c->err, sizeof(c->err),
                     "waited %.1f s at sequence %u, the B12X_ROCE_PEER_TIMEOUT_S limit, for "
                     "%s whose queue pairs still acknowledge checks",
                     roce_seconds(elapsed), seq, pending);
        } else {
            snprintf(c->err, sizeof(c->err),
                     "waited %.1f s at sequence %u, the B12X_ROCE_PEER_TIMEOUT_S limit, "
                     "although every peer flag for it is in host memory",
                     roce_seconds(elapsed), seq);
        }
        return -1;
    }
    if (late_peers != 0 && now >= c->next_probe_ns) {
        c->next_probe_ns = now + ROCE_PROBE_INTERVAL_NS;
        for (int p = 0; p < c->world; p++) {
            if ((late_peers & (1u << p)) != 0 &&
                post_notice(c, p, 0, seq & ROCE_NOTICE_SEQ) != 0) {
                snprintf(c->err, sizeof(c->err),
                         "could not post a peer check to rank %d while waiting at "
                         "sequence %u; its queue pair is in an error state",
                         p, seq);
                return -1;
            }
        }
    }
    return 0;
}

// Completes finished writes and supervises the outstanding wait.
static int proxy_tick(roce_ctx_t *c, volatile uint32_t *ctrl, uint64_t now) {
    for (int h = 0; h < c->n_hca; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return supervise(c, ctrl, now);
}

// Stops every wait on this rank and tells each peer, on every path, that this
// runtime stopped, so that no rank keeps waiting for it. The thread exits.
static void *fail_proxy(roce_ctx_t *c, volatile uint32_t *ctrl) {
    uint32_t seq = c->last_seq;
    __atomic_store_n(&ctrl[ROCE_CTRL_ABORT], seq != 0u ? seq : UINT32_MAX,
                     __ATOMIC_RELEASE);
    atomic_store(&c->failed, 1);
    roce_log(c, "%s; every RoCE wait on this rank stops and its runtime is poisoned", c->err);
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        for (int path = 0; path < (int)c->peer_path_count[p]; path++) {
            (void)post_notice(c, p, path, ROCE_NOTICE_ABORT | (seq & ROCE_NOTICE_SEQ));
        }
    }
    return NULL;
}

static void *proxy_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    // Spin while ops are flowing.  After ROCE_IDLE_SPINS polls without a
    // doorbell, request a short nanosleep between polls (the OS decides the
    // actual delay) so an idle runtime does not hold a core next to the
    // serving process.  The missed-doorbell catch-up below keeps the protocol
    // correct however long the thread is away.
    uint64_t idle = 0;
    const struct timespec nap = {0, 20000};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
        if (seq == c->last_seq) {
            idle++;
            if (idle % 64 == 0 && proxy_tick(c, ctrl, roce_now_ns()) != 0) {
                return fail_proxy(c, ctrl);
            }
            if (idle >= ROCE_IDLE_SPINS) {
                nanosleep(&nap, NULL);
            }
            continue;
        }
        idle = 0;
        // The doorbell holds only the newest sequence.  Our kernel for op N
        // completes on the peers' payloads alone, so op N+1 can ring before
        // this thread has seen op N (it slept, or the scheduler moved it).
        // Peers cannot get further than one op ahead of us, so at most
        // ROCE_SLOTS doorbells are pending and every send slot is intact:
        // post each missed sequence in order using its per-slot byte count.
        uint32_t pending = seq - c->last_seq;
        if (pending > ROCE_SLOTS) {
            snprintf(c->err, sizeof(c->err),
                     "doorbell skipped %u ops (last %u, now %u)", pending, c->last_seq, seq);
            return fail_proxy(c, ctrl);
        }
        for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
            uint32_t nbytes = ctrl[4 + (s & 1u)];
            if (post_op(c, s, nbytes) != 0) {
                return fail_proxy(c, ctrl);
            }
            c->last_seq = s;
        }
        // The host-side wait for the newest sequence starts once it is posted.
        c->posted_ns = roce_now_ns();
        note_progress(c, ctrl, c->posted_ns);
    }
    return NULL;
}

int roce_set_wait_timeout(roce_ctx_t *c, uint64_t timeout_ns) {
    if (timeout_ns == 0 || atomic_load(&c->running)) {
        return -1;
    }
    c->wait_timeout_ns = timeout_ns;
    c->stall_report_ns = roce_stall_report_ns(timeout_ns);
    return 0;
}

int roce_start(roce_ctx_t *c) {
    if (atomic_load(&c->running)) {
        return 0;
    }
    if (!c->started) {
        // A restart continues from the last posted sequence so ops that rang
        // the doorbell while the thread was stopped are still posted.
        volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
        c->last_seq = ctrl[0];
        c->started = 1;
    }
    c->posted_ns = roce_now_ns();
    c->last_tick_ns = c->posted_ns;
    c->watching = 0;
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc != 0) {
        atomic_store(&c->running, 0);
        set_err(c, "pthread_create", rc);
        return -1;
    }
    return 0;
}

void roce_stop(roce_ctx_t *c) {
    if (atomic_exchange(&c->running, 0)) {
        pthread_join(c->thread, NULL);
    }
}

int roce_failed(roce_ctx_t *c) { return atomic_load(&c->failed); }

const char *roce_error(roce_ctx_t *c) { return c->err; }

uint64_t roce_stat(roce_ctx_t *c, int which) {
    switch (which) {
    case 0:
        return c->ops_posted;
    case 1:
        return c->writes_completed;
    case 2:
        return c->last_seq;
    case 3:
        return atomic_load(&c->two_wave_activations);
    case 4:
        return atomic_load(&c->forward_chunks);
    case 5:
        return atomic_load(&c->stalls);
    case 6:
        return atomic_load(&c->stalls_resolved);
    case 7:
        return atomic_load(&c->longest_stall_ns);
    case 8:
        return atomic_load(&c->notices_posted);
    case 9:
        return atomic_load(&c->notices_received);
    case 10:
        return atomic_load(&c->longest_loop_gap_ns);
    case 11:
        return c->wait_timeout_ns;
    default:
        return 0;
    }
}

uint64_t roce_two_wave_threshold_bytes(roce_ctx_t *c) {
    return c->two_wave_threshold_bytes;
}

uint64_t roce_wave_mode(roce_ctx_t *c) {
    return c->wave_mode;
}

int roce_peer_hca(roce_ctx_t *c, int peer, int path) {
    if (peer < 0 || peer >= c->world || path < 0 ||
        path >= (int)c->peer_path_count[peer]) {
        return -1;
    }
    return c->peer_hca[peer][path];
}

uint64_t roce_path_stat(roce_ctx_t *c, int peer, int path, int which) {
    if (peer < 0 || peer >= c->world || peer == c->rank ||
        path < 0 || path >= (int)c->peer_path_count[peer]) {
        return UINT64_MAX;
    }
    switch (which) {
    case 0:
        return atomic_load(&c->payload_writes[peer][path]);
    case 1:
        return atomic_load(&c->payload_bytes[peer][path]);
    case 2:
        return atomic_load(&c->payload_bytes[peer][path]) * c->physical_hops[peer];
    case 3:
        return atomic_load(&c->flag_writes[peer][path]);
    case 4:
        return atomic_load(&c->send_completions[peer][path]);
    case 5:
        return atomic_load(&c->completion_errors[peer][path]);
    case 6:
        return c->hca[c->peer_hca[peer][path]].qp[peer]->qp_num;
    case 7:
        return c->remote_qp[peer][path];
    case 8:
        return (uint64_t)c->peer_hca[peer][path];
    case 9:
        return (uint64_t)c->remote_hca[peer][path];
    case 10:
        return c->physical_hops[peer];
    default:
        return UINT64_MAX;
    }
}

void roce_destroy(roce_ctx_t *c) {
    if (c == NULL) {
        return;
    }
    roce_stop(c);
    for (int h = 0; h < ROCE_MAX_HCAS; h++) {
        roce_hca_t *hca = &c->hca[h];
        for (int p = 0; p < ROCE_MAX_PEERS; p++) {
            if (hca->qp[p] != NULL) {
                ibv_destroy_qp(hca->qp[p]);
            }
        }
        if (hca->cq != NULL) {
            ibv_destroy_cq(hca->cq);
        }
        if (hca->mr != NULL) {
            ibv_dereg_mr(hca->mr);
        }
        if (hca->pd != NULL) {
            ibv_dealloc_pd(hca->pd);
        }
        if (hca->ctx != NULL) {
            ibv_close_device(hca->ctx);
        }
    }
    free(c);
}
