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
//   ctrl             (FLAG_STRIDE)                 {u32 seq, u32 nbytes, u32 error,
//                                                   u32 missing_peer} doorbell; the
//                                                  last two are set by the kernel
//                                                  when a wait times out
//
// The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq
// in ctrl, then spins on every active path flag for every peer. The proxy uses
// two half-payload paths per neighbor. Research-only TP4 mode uses four
// quarter-payload paths to the opposite rank. Each stripe is followed by its
// 4-byte sequence flag on the same reliable QP, so a path flag cannot become
// visible before its stripe. Nothing on the receive path involves the host.
//
// This file is compiled by b12x.comm.roce._proxy at first use with the host
// gcc and libibverbs; it must stay plain C with no CUDA dependency.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
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
#define ROCE_ABI_VERSION 4
#define ROCE_IDLE_SPINS 200000
#define ROCE_DEFAULT_TWO_WAVE_THRESHOLD_BYTES 131072u
#define ROCE_WAVE_MODE_TWO 0u
#define ROCE_WAVE_MODE_MIXED_TWO 1u
#define ROCE_WAVE_MODE_OPPOSITE_FIRST 2u
#define ROCE_WAVE_MODE_STRICT_THREE 3u
#define ROCE_WAVE_MODE_BALANCED32 4u

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
    char err[512];
} roce_ctx_t;

void roce_destroy(roce_ctx_t *c);

static void set_err(roce_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

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
        (opposite_paths == ROCE_MAX_PATHS && (world != 4 || n_hca != 4)) ||
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
        int count = world == 4 && distance == 2
                        ? opposite_paths
                        : ROCE_LAYOUT_PATHS;
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
    if (opposite_paths == ROCE_MAX_PATHS || c->wave_mode != ROCE_WAVE_MODE_TWO) {
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
    if (opposite_paths == ROCE_MAX_PATHS) {
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
        uint64_t wr_id = wc[i].wr_id;
        int peer = (int)(wr_id / ROCE_MAX_PATHS);
        int path = (int)(wr_id % ROCE_MAX_PATHS);
        if (peer < 0 || peer >= c->world || peer == c->rank ||
            path < 0 || path >= (int)c->peer_path_count[peer] ||
            c->peer_hca[peer][path] != h) {
            snprintf(c->err, sizeof(c->err),
                     "RDMA completion has invalid path identifier %llu",
                     (unsigned long long)wr_id);
            return -1;
        }
        if (wc[i].status != IBV_WC_SUCCESS) {
            atomic_fetch_add(&c->completion_errors[peer][path], 1);
            snprintf(c->err, sizeof(c->err),
                     "RDMA write to rank %d path %d failed: %s (vendor_err 0x%x, seq %u)",
                     peer, path, ibv_wc_status_str(wc[i].status), wc[i].vendor_err,
                     c->last_seq);
            return -1;
        }
        c->hca[h].outstanding[peer] -= 1;
        atomic_fetch_add(&c->send_completions[peer][path], 1);
        c->writes_completed += 1;
    }
    return 0;
}

static int post_path(roce_ctx_t *c, uint32_t seq, uint32_t slot, uint8_t *send,
                     int peer, int path, uint32_t stripe_offset,
                     uint32_t stripe_bytes) {
    int h = c->peer_hca[peer][path];
    roce_hca_t *hca = &c->hca[h];
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
    uint32_t flag_row = path < ROCE_LAYOUT_PATHS ? (uint32_t)c->rank
                                                  : (uint32_t)peer;
    uint32_t flag_column = path < ROCE_LAYOUT_PATHS
                               ? (uint32_t)path
                               : (uint32_t)(path - ROCE_LAYOUT_PATHS);
    flag_wr.wr.rdma.remote_addr =
        remote + c->flag_off +
        (((uint64_t)flag_row * ROCE_SLOTS + slot) * ROCE_LAYOUT_PATHS +
         flag_column) * ROCE_FLAG_STRIDE;
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
    c->ops_posted += 1;
    for (int h = 0; h < c->n_hca; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return 0;
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
            if (idle % 64 == 0) {
                for (int h = 0; h < c->n_hca; h++) {
                    if (drain_cq(c, h) != 0) {
                        atomic_store(&c->failed, 1);
                        return NULL;
                    }
                }
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
            atomic_store(&c->failed, 1);
            return NULL;
        }
        for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
            uint32_t nbytes = ctrl[4 + (s & 1u)];
            if (post_op(c, s, nbytes) != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
            c->last_seq = s;
        }
    }
    return NULL;
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
