/*
 * In-memory verbs stand-in for the CPU tests of SIRCL's native layer.
 *
 * Implements the libibverbs subset declared in infiniband/verbs.h over
 * process memory; fake_verbs.h describes the fabric model and the control
 * interface. Every rank of a simulated session lives in this process and
 * registers its own arena; an RDMA write copies bytes from the sender's
 * registered memory into the receiver's after checking keys, bounds, the
 * route and that the destination queue pair is connected back to the sender.
 *
 * Thread safety: one mutex serializes every entry point, so progress threads,
 * kernel threads and the scheduler may call in concurrently. Writes are
 * applied with a full fence first, and 4-byte aligned writes (flag lines) are
 * release stores, so a reader that sees a flag with an acquire load also sees
 * the payload written before it on the same queue pair.
 */

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fake_verbs.h"

#define FV_MAX_DEVICES 256
#define FV_MAX_CABLES 256
#define FV_MAX_NODES 64
#define FV_MAX_MRS 1024
#define FV_MAX_QPS 4096
#define FV_MAX_CQS 512
#define FV_MAX_TAGS 4096
#define FV_MAX_INJECTIONS 64
#define FV_INLINE 16

static pthread_mutex_t fv_lock = PTHREAD_MUTEX_INITIALIZER;
#define FV_LOCK() pthread_mutex_lock(&fv_lock)
#define FV_UNLOCK() pthread_mutex_unlock(&fv_lock)

typedef struct {
    struct ibv_device dev;
    int used;
    int node, port, function;
    uint8_t gid[16];
    int failed;
} fv_device_rec;

typedef struct {
    int node_a, port_a, node_b, port_b;
    uint32_t mask;
} fv_cable_rec;

typedef struct {
    int src_device;
    uint8_t gid[16];
    uint32_t relays;
} fv_tag_rec;

typedef struct {
    struct ibv_mr mr;
    int used;
    int device;
    int access;
} fv_mr_rec;

typedef struct {
    struct ibv_cq cq;
    int used;
    struct ibv_wc *entries;
    uint64_t *visible; /* per entry: when ibv_poll_cq may return it (fv_set_latency) */
    int capacity, head, count;
} fv_cq_rec;

typedef struct {
    uint64_t wr_id;
    int signaled, is_inline;
    uint8_t inline_data[FV_INLINE];
    uint64_t local_addr;
    uint32_t length, lkey;
    uint64_t remote_addr;
    uint32_t rkey;
    uint64_t ready_ns; /* earliest execution (fv_set_latency); 0: at once */
} fv_wr;

typedef struct {
    struct ibv_qp qp;
    int device;
    int errored;
    uint32_t dest_qp_num;
    uint8_t dgid[16];
    fv_route_t route;
    fv_wr *queue;
    int capacity, head, count;
    /* Bytes posted, bytes executed, bytes retired by a completion (every write
     * up to an executed signaled one), and the largest posted-minus-retired. */
    uint64_t posted_bytes, executed_bytes, retired_bytes, max_inflight;
    uint64_t tx_free_ns;  /* fv_set_rate: when the queue pair's previous write has been sent */
} fv_qp_rec;

static fv_device_rec devices[FV_MAX_DEVICES];
static int device_count;
static fv_cable_rec cables[FV_MAX_CABLES];
static int cable_count;
static fv_tag_rec tags[FV_MAX_TAGS];
static int tag_count;
static int relay_allowed[FV_MAX_NODES];
static int relay_depth = 16;
static int ideal;
static int recording = 1;
static int teardown_failures;  /* fv_fail_teardown */
/* Path latency of a write and of its completion: base plus a share per relay (0, 0: none). */
static uint64_t latency_base_ns, latency_relay_ns;
/* Sending rate of each queue pair in bytes per microsecond (0: no sending time). */
static uint64_t rate_bytes_per_us;
/* Extra time a completion takes to return after its write executed (fv_set_ack_delay). */
static uint64_t ack_delay_ns;
/* 0: writes of more than four bytes move no payload (timing runs whose kernels check no data). */
static int payloads = 1;

static uint64_t fv_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t path_latency(const fv_qp_rec *q) {
    return latency_base_ns + latency_relay_ns * (uint64_t)q->route.relays;
}
static struct { uint32_t qp_num, value; int used; } injections[FV_MAX_INJECTIONS];
static fv_mr_rec mrs[FV_MAX_MRS];
static fv_cq_rec cqs[FV_MAX_CQS];
static fv_qp_rec *qps[FV_MAX_QPS];
static uint32_t next_key = 0x100;
static uint32_t next_qpn = 0x40;
static fv_event_t *events;
static uint64_t event_count, event_capacity, event_sequence, executed;
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

/* -- helpers (callers hold the lock) ------------------------------------------ */

static void record(fv_event_t *event) {
    event->sequence = event_sequence++;
    if (!recording) return;
    if (event_count == event_capacity) {
        uint64_t capacity = event_capacity ? event_capacity * 2 : 4096;
        fv_event_t *grown = (fv_event_t *)realloc(events, (size_t)capacity * sizeof(*grown));
        if (grown == NULL) return;
        events = grown;
        event_capacity = capacity;
    }
    events[event_count++] = *event;
}

static const fv_cable_rec *cable_at(int node, int port) {
    for (int i = 0; i < cable_count; i++) {
        if ((cables[i].node_a == node && cables[i].port_a == port) ||
            (cables[i].node_b == node && cables[i].port_b == port)) {
            return &cables[i];
        }
    }
    return NULL;
}

static void cross(const fv_cable_rec *cable, int node, int port, int *to_node, int *to_port) {
    if (cable->node_a == node && cable->port_a == port) {
        *to_node = cable->node_b;
        *to_port = cable->port_b;
    } else {
        *to_node = cable->node_a;
        *to_port = cable->port_a;
    }
}

static int device_by_gid(const uint8_t *gid) {
    for (int i = 0; i < device_count; i++) {
        if (devices[i].used && memcmp(devices[i].gid, gid, 16) == 0) return i;
    }
    return -1;
}

static int device_at(int node, int port, int function) {
    for (int i = 0; i < device_count; i++) {
        if (devices[i].used && devices[i].node == node && devices[i].port == port &&
            devices[i].function == function) {
            return i;
        }
    }
    return -1;
}

static uint32_t tag_for(int src_device, const uint8_t *gid) {
    for (int i = 0; i < tag_count; i++) {
        if (tags[i].src_device == src_device && memcmp(tags[i].gid, gid, 16) == 0) return tags[i].relays;
    }
    return 0;
}

static void resolve(fv_qp_rec *q) {
    fv_route_t route;
    memset(&route, 0, sizeof(route));
    route.dst_device = -1;
    int target = device_by_gid(q->dgid);
    if (ideal) {
        route.routable = target >= 0;
        route.dst_device = target;
        route.relays = tag_for(q->device, q->dgid);   /* for fv_set_latency; delivery ignores it */
        q->route = route;
        return;
    }
    fv_device_rec *src = &devices[q->device];
    int node = src->node, port = src->port, function = src->function;
    uint32_t left = tag_for(q->device, q->dgid);
    route.relays = left;
    const fv_cable_rec *cable = cable_at(node, port);
    if (cable == NULL || !(cable->mask & (1u << function))) {
        q->route = route;
        return;
    }
    cross(cable, node, port, &node, &port);
    while (left > 0) {
        if (node >= FV_MAX_NODES || !relay_allowed[node] || (int)left > relay_depth) {
            q->route = route;
            return;
        }
        route.relay_nodes |= 1ull << (uint64_t)node;
        int out_port = 1 - port;
        cable = cable_at(node, out_port);
        if (cable == NULL || !(cable->mask & (1u << function))) {
            q->route = route;
            return;
        }
        cross(cable, node, out_port, &node, &port);
        left--;
    }
    if (target >= 0 && device_at(node, port, function) == target) {
        route.routable = 1;
        route.dst_device = target;
    }
    q->route = route;
}

static int path_failed(const fv_qp_rec *q) {
    if (devices[q->device].failed) return 1;
    if (q->route.dst_device >= 0 && devices[q->route.dst_device].failed) return 1;
    for (int i = 0; i < device_count; i++) {
        if (devices[i].failed && devices[i].node < 64 &&
            (q->route.relay_nodes & (1ull << (uint64_t)devices[i].node))) {
            return 1;
        }
    }
    return 0;
}

static fv_mr_rec *find_mr(int device, uint32_t key, int remote, uint64_t addr, uint64_t length) {
    for (int i = 0; i < FV_MAX_MRS; i++) {
        fv_mr_rec *m = &mrs[i];
        if (!m->used || m->device != device) continue;
        if ((remote ? m->mr.rkey : m->mr.lkey) != key) continue;
        uint64_t base = (uint64_t)(uintptr_t)m->mr.addr;
        if (addr < base || addr + length > base + m->mr.length || addr + length < addr) return NULL;
        if (remote && !(m->access & IBV_ACCESS_REMOTE_WRITE)) return NULL;
        return m;
    }
    return NULL;
}

/* The queue pair on the destination device that this queue pair talks to, if
 * it exists and is connected back to it. */
static int paired(const fv_qp_rec *q, int *reason) {
    for (int i = 0; i < FV_MAX_QPS; i++) {
        fv_qp_rec *r = qps[i];
        if (r == NULL || r->device != q->route.dst_device || r->qp.qp_num != q->dest_qp_num) continue;
        if ((r->qp.state != IBV_QPS_RTR && r->qp.state != IBV_QPS_RTS) ||
            r->dest_qp_num != q->qp.qp_num || memcmp(r->dgid, devices[q->device].gid, 16) != 0) {
            *reason = FV_NOT_PAIRED;
            return 0;
        }
        return 1;
    }
    *reason = FV_NO_REMOTE_QP;
    return 0;
}

static void complete(fv_qp_rec *q, uint64_t wr_id, enum ibv_wc_status status) {
    fv_cq_rec *cq = &cqs[q->qp.send_cq->fv_index];
    if (cq->count == cq->capacity) return; /* tests size completion queues generously */
    int index = (cq->head + cq->count) % cq->capacity;
    struct ibv_wc *wc = &cq->entries[index];
    cq->visible[index] = latency_base_ns | latency_relay_ns | ack_delay_ns
                             ? fv_now_ns() + path_latency(q) + ack_delay_ns : 0;
    memset(wc, 0, sizeof(*wc));
    wc->wr_id = wr_id;
    wc->status = status;
    wc->opcode = IBV_WC_RDMA_WRITE;
    wc->qp_num = q->qp.qp_num;
    cq->count++;
}

static void apply(uint64_t remote_addr, const void *source, uint32_t length) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (length == 4 && remote_addr % 4 == 0) {
        uint32_t word;
        memcpy(&word, source, 4);
        __atomic_store_n((uint32_t *)(uintptr_t)remote_addr, word, __ATOMIC_RELEASE);
    } else if (length != 0 && payloads) {
        memmove((void *)(uintptr_t)remote_addr, source, length);
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static int take_injection(uint32_t qp_num, const fv_wr *w) {
    if (!w->is_inline || w->length < 4) return 0;
    uint32_t value;
    memcpy(&value, w->inline_data, 4);
    for (int i = 0; i < FV_MAX_INJECTIONS; i++) {
        if (injections[i].used && (injections[i].qp_num == 0 || injections[i].qp_num == qp_num) &&
            injections[i].value == value) {
            injections[i].used = 0;
            return 1;
        }
    }
    return 0;
}

static void execute_one(fv_qp_rec *q) {
    fv_wr w = q->queue[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    enum ibv_wc_status status = IBV_WC_SUCCESS;
    int reason = FV_OK;
    if (q->errored) {
        status = IBV_WC_WR_FLUSH_ERR;
        reason = FV_FLUSHED;
    } else if (!q->route.routable) {
        status = IBV_WC_RETRY_EXC_ERR;
        reason = FV_UNROUTABLE;
    } else if (path_failed(q)) {
        status = IBV_WC_RETRY_EXC_ERR;
        reason = FV_DEVICE_FAILED;
    } else if (!paired(q, &reason)) {
        status = IBV_WC_RETRY_EXC_ERR;
    } else if (take_injection(q->qp.qp_num, &w)) {
        status = IBV_WC_REM_ACCESS_ERR;
        reason = FV_INJECTED;
    } else {
        fv_mr_rec *dst = find_mr(q->route.dst_device, w.rkey, 1, w.remote_addr, w.length);
        const void *source = w.inline_data;
        if (!w.is_inline) {
            fv_mr_rec *src = find_mr(q->device, w.lkey, 0, w.local_addr, w.length);
            source = src != NULL ? (const void *)(uintptr_t)w.local_addr : NULL;
        }
        if (dst == NULL) {
            status = IBV_WC_REM_ACCESS_ERR;
            reason = FV_REMOTE_ACCESS;
        } else if (source == NULL) {
            status = IBV_WC_LOC_PROT_ERR;
            reason = FV_LOCAL_PROTECTION;
        } else {
            apply(w.remote_addr, source, w.length);
        }
    }
    if (status != IBV_WC_SUCCESS) q->errored = 1;
    executed++;
    q->executed_bytes += w.length;
    if (w.signaled || status != IBV_WC_SUCCESS) q->retired_bytes = q->executed_bytes;
    fv_event_t event;
    memset(&event, 0, sizeof(event));
    event.phase = 1;
    event.status = (uint32_t)status;
    event.reason = (uint32_t)reason;
    event.src_device = q->device;
    event.dst_device = q->route.routable ? q->route.dst_device : -1;
    event.qp_num = q->qp.qp_num;
    event.flags = (uint32_t)(w.signaled ? 1 : 0) | (uint32_t)(w.is_inline ? 2 : 0);
    event.wr_id = w.wr_id;
    event.local_addr = w.local_addr;
    event.remote_addr = w.remote_addr;
    event.length = w.length;
    if (w.is_inline) memcpy(&event.inline_word, w.inline_data, 4);
    event.relays = q->route.relays;
    event.relay_nodes = q->route.relay_nodes;
    record(&event);
    if (w.signaled || status != IBV_WC_SUCCESS) complete(q, w.wr_id, status);
}

static uint64_t next_random(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* -- control interface ---------------------------------------------------------- */

FV_API void fv_reset(void) {
    FV_LOCK();
    for (int i = 0; i < FV_MAX_QPS; i++) {
        if (qps[i] != NULL) {
            free(qps[i]->queue);
            free(qps[i]);
            qps[i] = NULL;
        }
    }
    for (int i = 0; i < FV_MAX_CQS; i++) {
        free(cqs[i].entries);
        free(cqs[i].visible);
    }
    memset(cqs, 0, sizeof(cqs));
    memset(mrs, 0, sizeof(mrs));
    memset(devices, 0, sizeof(devices));
    memset(cables, 0, sizeof(cables));
    memset(tags, 0, sizeof(tags));
    memset(relay_allowed, 0, sizeof(relay_allowed));
    memset(injections, 0, sizeof(injections));
    device_count = cable_count = tag_count = 0;
    next_key = 0x100;
    next_qpn = 0x40;
    relay_depth = 16;
    ideal = 0;
    recording = 1;
    teardown_failures = 0;
    latency_base_ns = latency_relay_ns = 0;
    rate_bytes_per_us = 0;
    ack_delay_ns = 0;
    payloads = 1;
    free(events);
    events = NULL;
    event_count = event_capacity = event_sequence = executed = 0;
    FV_UNLOCK();
}

FV_API int fv_add_device(const char *name, int node, int port, int function, const uint8_t *gid) {
    FV_LOCK();
    if (device_count >= FV_MAX_DEVICES || name == NULL || strlen(name) >= 64 || node < 0 ||
        node >= FV_MAX_NODES || port < 0 || port > 1 || function < 0 || function > 1) {
        FV_UNLOCK();
        return -1;
    }
    fv_device_rec *d = &devices[device_count];
    memset(d, 0, sizeof(*d));
    snprintf(d->dev.name, sizeof(d->dev.name), "%s", name);
    d->dev.fv_index = device_count;
    d->used = 1;
    d->node = node;
    d->port = port;
    d->function = function;
    memcpy(d->gid, gid, 16);
    int index = device_count++;
    FV_UNLOCK();
    return index;
}

FV_API int fv_add_cable(int node_a, int port_a, int node_b, int port_b, uint32_t function_mask) {
    FV_LOCK();
    if (cable_count >= FV_MAX_CABLES) {
        FV_UNLOCK();
        return -1;
    }
    cables[cable_count].node_a = node_a;
    cables[cable_count].port_a = port_a;
    cables[cable_count].node_b = node_b;
    cables[cable_count].port_b = port_b;
    cables[cable_count].mask = function_mask;
    int index = cable_count++;
    FV_UNLOCK();
    return index;
}

FV_API void fv_set_relay(int node, int allowed) {
    FV_LOCK();
    if (node >= 0 && node < FV_MAX_NODES) relay_allowed[node] = allowed != 0;
    FV_UNLOCK();
}

FV_API void fv_set_relay_depth(int depth) {
    FV_LOCK();
    relay_depth = depth;
    FV_UNLOCK();
}

FV_API int fv_set_dest_tag(int src_device, const uint8_t *dgid, uint32_t relays) {
    FV_LOCK();
    for (int i = 0; i < tag_count; i++) {
        if (tags[i].src_device == src_device && memcmp(tags[i].gid, dgid, 16) == 0) {
            tags[i].relays = relays;
            FV_UNLOCK();
            return 0;
        }
    }
    if (tag_count >= FV_MAX_TAGS) {
        FV_UNLOCK();
        return -1;
    }
    tags[tag_count].src_device = src_device;
    memcpy(tags[tag_count].gid, dgid, 16);
    tags[tag_count].relays = relays;
    tag_count++;
    FV_UNLOCK();
    return 0;
}

FV_API void fv_set_latency(uint64_t base_ns, uint64_t per_relay_ns) {
    FV_LOCK();
    latency_base_ns = base_ns;
    latency_relay_ns = per_relay_ns;
    FV_UNLOCK();
}

FV_API void fv_set_ack_delay(uint64_t ns) {
    FV_LOCK();
    ack_delay_ns = ns;
    FV_UNLOCK();
}

FV_API void fv_set_rate(uint64_t bytes_per_us) {
    FV_LOCK();
    rate_bytes_per_us = bytes_per_us;
    FV_UNLOCK();
}

FV_API void fv_set_payloads(int on) {
    FV_LOCK();
    payloads = on != 0;
    FV_UNLOCK();
}

FV_API void fv_set_ideal(int on) {
    FV_LOCK();
    ideal = on != 0;
    FV_UNLOCK();
}

FV_API void fv_set_recording(int on) {
    FV_LOCK();
    recording = on != 0;
    FV_UNLOCK();
}

FV_API void fv_inject_failure(uint32_t qp_num, uint32_t inline_value) {
    FV_LOCK();
    for (int i = 0; i < FV_MAX_INJECTIONS; i++) {
        if (!injections[i].used) {
            injections[i].used = 1;
            injections[i].qp_num = qp_num;
            injections[i].value = inline_value;
            break;
        }
    }
    FV_UNLOCK();
}

FV_API void fv_fail_teardown(int calls) {
    FV_LOCK();
    teardown_failures = calls > 0 ? calls : 0;
    FV_UNLOCK();
}

FV_API void fv_fail_device(int device) {
    FV_LOCK();
    if (device >= 0 && device < device_count) devices[device].failed = 1;
    FV_UNLOCK();
}

FV_API uint64_t fv_progress(uint64_t max_work_requests, uint64_t seed) {
    static int ready[FV_MAX_QPS];
    uint64_t done = 0;
    FV_LOCK();
    if (seed != 0) rng_state = seed;
    uint64_t now = latency_base_ns | latency_relay_ns | rate_bytes_per_us ? fv_now_ns() : 0;
    while (done < max_work_requests) {
        int count = 0;
        for (int i = 0; i < FV_MAX_QPS; i++) {
            if (qps[i] != NULL && qps[i]->count > 0 && qps[i]->queue[qps[i]->head].ready_ns <= now) {
                ready[count++] = i;
            }
        }
        if (count == 0) break;
        int pick = (int)(next_random() % (uint64_t)count);
        execute_one(qps[ready[pick]]);
        done++;
    }
    FV_UNLOCK();
    return done;
}

FV_API uint64_t fv_pending(void) {
    uint64_t total = 0;
    FV_LOCK();
    for (int i = 0; i < FV_MAX_QPS; i++) {
        if (qps[i] != NULL) total += (uint64_t)qps[i]->count;
    }
    FV_UNLOCK();
    return total;
}

FV_API uint64_t fv_executed(void) {
    FV_LOCK();
    uint64_t value = executed;
    FV_UNLOCK();
    return value;
}

FV_API uint64_t fv_event_count(void) {
    FV_LOCK();
    uint64_t count = event_count;
    FV_UNLOCK();
    return count;
}

FV_API int fv_event(uint64_t index, fv_event_t *out) {
    FV_LOCK();
    if (index >= event_count || out == NULL) {
        FV_UNLOCK();
        return -1;
    }
    *out = events[index];
    FV_UNLOCK();
    return 0;
}

FV_API void fv_clear_events(void) {
    FV_LOCK();
    event_count = 0;
    FV_UNLOCK();
}

FV_API uint64_t fv_qp_max_inflight(uint32_t qp_num) {
    FV_LOCK();
    for (int i = 0; i < FV_MAX_QPS; i++) {
        if (qps[i] != NULL && qps[i]->qp.qp_num == qp_num) {
            uint64_t value = qps[i]->max_inflight;
            FV_UNLOCK();
            return value;
        }
    }
    FV_UNLOCK();
    return UINT64_MAX;
}

FV_API int fv_qp_route(uint32_t qp_num, fv_route_t *out) {
    FV_LOCK();
    for (int i = 0; i < FV_MAX_QPS; i++) {
        if (qps[i] != NULL && qps[i]->qp.qp_num == qp_num) {
            *out = qps[i]->route;
            FV_UNLOCK();
            return 0;
        }
    }
    FV_UNLOCK();
    return -1;
}

/* -- verbs ---------------------------------------------------------------------- */

FV_API struct ibv_device **ibv_get_device_list(int *num_devices) {
    FV_LOCK();
    struct ibv_device **list = (struct ibv_device **)calloc((size_t)device_count + 1, sizeof(*list));
    int count = 0;
    if (list != NULL) {
        for (int i = 0; i < device_count; i++) {
            if (devices[i].used) list[count++] = &devices[i].dev;
        }
    }
    FV_UNLOCK();
    if (num_devices != NULL) *num_devices = count;
    if (list == NULL) errno = ENOMEM;
    return list;
}

FV_API void ibv_free_device_list(struct ibv_device **list) { free(list); }

FV_API const char *ibv_get_device_name(struct ibv_device *device) { return device->name; }

FV_API struct ibv_context *ibv_open_device(struct ibv_device *device) {
    struct ibv_context *context = (struct ibv_context *)calloc(1, sizeof(*context));
    if (context == NULL) {
        errno = ENOMEM;
        return NULL;
    }
    context->device = device;
    context->fv_index = device->fv_index;
    return context;
}

FV_API int ibv_close_device(struct ibv_context *context) {
    free(context);
    return 0;
}

FV_API int ibv_query_port(struct ibv_context *context, uint8_t port_num,
                          struct ibv_port_attr *port_attr) {
    if (port_num != 1 || port_attr == NULL) return EINVAL;
    memset(port_attr, 0, sizeof(*port_attr));
    FV_LOCK();
    int failed = devices[context->fv_index].failed;
    FV_UNLOCK();
    port_attr->state = failed ? IBV_PORT_DOWN : IBV_PORT_ACTIVE;
    port_attr->max_mtu = IBV_MTU_4096;
    port_attr->active_mtu = IBV_MTU_4096;
    port_attr->gid_tbl_len = 256;
    port_attr->lid = 0;
    return 0;
}

FV_API int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index,
                         union ibv_gid *gid) {
    if (port_num != 1 || index < 0 || index > 255) return EINVAL;
    FV_LOCK();
    memcpy(gid->raw, devices[context->fv_index].gid, 16);
    FV_UNLOCK();
    return 0;
}

FV_API struct ibv_pd *ibv_alloc_pd(struct ibv_context *context) {
    struct ibv_pd *pd = (struct ibv_pd *)calloc(1, sizeof(*pd));
    if (pd == NULL) {
        errno = ENOMEM;
        return NULL;
    }
    pd->context = context;
    return pd;
}

FV_API int ibv_dealloc_pd(struct ibv_pd *pd) {
    free(pd);
    return 0;
}

FV_API struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
    FV_LOCK();
    for (int i = 0; i < FV_MAX_MRS; i++) {
        if (!mrs[i].used) {
            fv_mr_rec *m = &mrs[i];
            memset(m, 0, sizeof(*m));
            m->used = 1;
            m->device = pd->context->fv_index;
            m->access = access;
            m->mr.context = pd->context;
            m->mr.pd = pd;
            m->mr.addr = addr;
            m->mr.length = length;
            m->mr.handle = (uint32_t)i;
            m->mr.lkey = next_key++;
            m->mr.rkey = next_key++;
            FV_UNLOCK();
            return &m->mr;
        }
    }
    FV_UNLOCK();
    errno = ENOMEM;
    return NULL;
}

FV_API int ibv_dereg_mr(struct ibv_mr *mr) {
    FV_LOCK();
    mrs[mr->handle].used = 0;
    FV_UNLOCK();
    return 0;
}

FV_API struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context,
                                    struct ibv_comp_channel *channel, int comp_vector) {
    (void)cq_context;
    (void)channel;
    (void)comp_vector;
    FV_LOCK();
    for (int i = 0; i < FV_MAX_CQS; i++) {
        if (!cqs[i].used) {
            fv_cq_rec *c = &cqs[i];
            free(c->entries);
            memset(c, 0, sizeof(*c));
            c->entries = (struct ibv_wc *)calloc((size_t)cqe, sizeof(struct ibv_wc));
            c->visible = (uint64_t *)calloc((size_t)cqe, sizeof(uint64_t));
            if (c->entries == NULL || c->visible == NULL) {
                free(c->entries);
                free(c->visible);
                c->entries = NULL;
                c->visible = NULL;
                FV_UNLOCK();
                errno = ENOMEM;
                return NULL;
            }
            c->used = 1;
            c->capacity = cqe;
            c->cq.context = context;
            c->cq.cqe = cqe;
            c->cq.fv_index = i;
            FV_UNLOCK();
            return &c->cq;
        }
    }
    FV_UNLOCK();
    errno = ENOMEM;
    return NULL;
}

FV_API int ibv_destroy_cq(struct ibv_cq *cq) {
    FV_LOCK();
    fv_cq_rec *c = &cqs[cq->fv_index];
    free(c->entries);
    free(c->visible);
    memset(c, 0, sizeof(*c));
    FV_UNLOCK();
    return 0;
}

FV_API struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr) {
    if (attr->qp_type != IBV_QPT_RC || attr->send_cq == NULL) {
        errno = EINVAL;
        return NULL;
    }
    FV_LOCK();
    for (int i = 0; i < FV_MAX_QPS; i++) {
        if (qps[i] == NULL) {
            fv_qp_rec *q = (fv_qp_rec *)calloc(1, sizeof(*q));
            if (q == NULL) {
                FV_UNLOCK();
                errno = ENOMEM;
                return NULL;
            }
            q->capacity = (int)attr->cap.max_send_wr;
            q->queue = (fv_wr *)calloc((size_t)q->capacity, sizeof(fv_wr));
            if (q->queue == NULL) {
                free(q);
                FV_UNLOCK();
                errno = ENOMEM;
                return NULL;
            }
            q->device = pd->context->fv_index;
            q->route.dst_device = -1;
            q->qp.context = pd->context;
            q->qp.pd = pd;
            q->qp.send_cq = attr->send_cq;
            q->qp.recv_cq = attr->recv_cq;
            q->qp.qp_num = next_qpn++;
            q->qp.state = IBV_QPS_RESET;
            q->qp.fv_index = i;
            qps[i] = q;
            FV_UNLOCK();
            return &q->qp;
        }
    }
    FV_UNLOCK();
    errno = ENOMEM;
    return NULL;
}

FV_API int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
    if (!(attr_mask & IBV_QP_STATE)) return EINVAL;
    FV_LOCK();
    fv_qp_rec *q = qps[qp->fv_index];
    int rc = 0;
    switch (attr->qp_state) {
    case IBV_QPS_INIT:
        rc = q->qp.state == IBV_QPS_RESET ? 0 : EINVAL;
        break;
    case IBV_QPS_RTR:
        if (q->qp.state != IBV_QPS_INIT || !(attr_mask & IBV_QP_AV) ||
            !(attr_mask & IBV_QP_DEST_QPN) || !attr->ah_attr.is_global) {
            rc = EINVAL;
            break;
        }
        q->dest_qp_num = attr->dest_qp_num;
        memcpy(q->dgid, attr->ah_attr.grh.dgid.raw, 16);
        resolve(q);
        break;
    case IBV_QPS_RTS:
        rc = q->qp.state == IBV_QPS_RTR ? 0 : EINVAL;
        break;
    case IBV_QPS_ERR:
    case IBV_QPS_RESET:
        break;
    default:
        rc = EINVAL;
        break;
    }
    if (rc == 0) q->qp.state = attr->qp_state;
    FV_UNLOCK();
    return rc;
}

FV_API int ibv_destroy_qp(struct ibv_qp *qp) {
    FV_LOCK();
    if (teardown_failures > 0) {
        teardown_failures--;
        FV_UNLOCK();
        return EBUSY;
    }
    fv_qp_rec *q = qps[qp->fv_index];
    qps[qp->fv_index] = NULL;
    free(q->queue);
    free(q);
    FV_UNLOCK();
    return 0;
}

FV_API int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr) {
    FV_LOCK();
    fv_qp_rec *q = qps[qp->fv_index];
    for (struct ibv_send_wr *w = wr; w != NULL; w = w->next) {
        int rc = 0;
        if (q->qp.state != IBV_QPS_RTS || w->opcode != IBV_WR_RDMA_WRITE || w->num_sge != 1) {
            rc = EINVAL;
        } else if ((w->send_flags & IBV_SEND_INLINE) && w->sg_list[0].length > FV_INLINE) {
            rc = EINVAL;
        } else if (q->count == q->capacity) {
            rc = ENOMEM;
        }
        if (rc != 0) {
            if (bad_wr != NULL) *bad_wr = w;
            FV_UNLOCK();
            return rc;
        }
        fv_wr *slot = &q->queue[(q->head + q->count) % q->capacity];
        memset(slot, 0, sizeof(*slot));
        if (latency_base_ns | latency_relay_ns | rate_bytes_per_us) {
            uint64_t now = fv_now_ns();
            uint64_t start = q->tx_free_ns > now ? q->tx_free_ns : now;
            uint64_t sent = start + (rate_bytes_per_us ? (uint64_t)slot->length * 1000u / rate_bytes_per_us : 0);
            q->tx_free_ns = sent;
            slot->ready_ns = sent + path_latency(q);
        }
        slot->wr_id = w->wr_id;
        slot->signaled = (w->send_flags & IBV_SEND_SIGNALED) != 0;
        slot->is_inline = (w->send_flags & IBV_SEND_INLINE) != 0;
        slot->local_addr = w->sg_list[0].addr;
        slot->length = w->sg_list[0].length;
        slot->lkey = w->sg_list[0].lkey;
        slot->remote_addr = w->wr.rdma.remote_addr;
        slot->rkey = w->wr.rdma.rkey;
        if (slot->is_inline) {
            memcpy(slot->inline_data, (const void *)(uintptr_t)slot->local_addr, slot->length);
        }
        q->count++;
        q->posted_bytes += slot->length;
        if (q->posted_bytes - q->retired_bytes > q->max_inflight) {
            q->max_inflight = q->posted_bytes - q->retired_bytes;
        }
        fv_event_t event;
        memset(&event, 0, sizeof(event));
        event.phase = 0;
        event.src_device = q->device;
        event.dst_device = q->route.routable ? q->route.dst_device : -1;
        event.qp_num = q->qp.qp_num;
        event.flags = (uint32_t)(slot->signaled ? 1 : 0) | (uint32_t)(slot->is_inline ? 2 : 0);
        event.wr_id = slot->wr_id;
        event.local_addr = slot->local_addr;
        event.remote_addr = slot->remote_addr;
        event.length = slot->length;
        if (slot->is_inline) memcpy(&event.inline_word, slot->inline_data, 4);
        event.relays = q->route.relays;
        event.relay_nodes = q->route.relay_nodes;
        record(&event);
    }
    FV_UNLOCK();
    return 0;
}

FV_API int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    FV_LOCK();
    fv_cq_rec *c = &cqs[cq->fv_index];
    int n = 0;
    if (latency_base_ns | latency_relay_ns | ack_delay_ns) {
        /* Completions become visible in the order of their paths' latencies, so a nearer queue
         * pair's may pass a farther one's; each queue pair's stay in order. */
        uint64_t now = fv_now_ns();
        int kept = 0, count = c->count;
        for (int i = 0; i < count; i++) {
            int index = (c->head + i) % c->capacity;
            if (n < num_entries && c->visible[index] <= now) {
                wc[n++] = c->entries[index];
                continue;
            }
            int to = (c->head + kept) % c->capacity;
            if (to != index) {
                c->entries[to] = c->entries[index];
                c->visible[to] = c->visible[index];
            }
            kept++;
        }
        c->count = kept;
    } else {
        while (n < num_entries && c->count > 0) {
            wc[n++] = c->entries[c->head];
            c->head = (c->head + 1) % c->capacity;
            c->count--;
        }
    }
    FV_UNLOCK();
    return n;
}

FV_API const char *ibv_wc_status_str(enum ibv_wc_status status) {
    switch (status) {
    case IBV_WC_SUCCESS: return "success";
    case IBV_WC_LOC_LEN_ERR: return "local length error";
    case IBV_WC_LOC_QP_OP_ERR: return "local QP operation error";
    case IBV_WC_LOC_PROT_ERR: return "local protection error";
    case IBV_WC_WR_FLUSH_ERR: return "Work Request Flushed Error";
    case IBV_WC_REM_INV_REQ_ERR: return "remote invalid request error";
    case IBV_WC_REM_ACCESS_ERR: return "remote access error";
    case IBV_WC_REM_OP_ERR: return "remote operation error";
    case IBV_WC_RETRY_EXC_ERR: return "transport retry counter exceeded";
    default: return "other error";
    }
}
