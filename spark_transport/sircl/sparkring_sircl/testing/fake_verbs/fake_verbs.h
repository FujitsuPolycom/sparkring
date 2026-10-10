/*
 * Control interface of the in-memory verbs stand-in (fake_verbs.c).
 *
 * The stand-in implements the libibverbs subset that SIRCL's native layer
 * (sparkring_sircl/oneshot/_roce_proxy.c) uses, for several ranks in one
 * process. Tests describe a fabric: one device per (node, port, function),
 * cables between node ports with the functions they carry, which nodes may
 * relay, and the relay tag of every (sending device, destination GID) pair,
 * which is how a site's relay plan marks relayed lanes (destination-matched,
 * flow label 0). A queue pair is routable when a frame that leaves its device
 * with that tag, forwarded by every relay out of its other port on the same
 * function, arrives at the device of its destination GID. With the ideal
 * switch on, frames go straight to the device of the destination GID and no
 * cable is consulted.
 *
 * Work requests run in fv_progress, one at a time, in a seeded interleaving
 * that keeps each queue pair's order (the only order RC gives). A write is
 * applied only when the destination queue pair exists and is connected back
 * to the sender (the pair is real); otherwise it fails as it would on a
 * fabric that drops it, and the event records why.
 */
#ifndef SIRCL_FAKE_VERBS_H
#define SIRCL_FAKE_VERBS_H

#include <stdint.h>

#include "infiniband/verbs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Why an executed write failed (fv_event_t.reason). */
enum {
    FV_OK = 0,
    FV_UNROUTABLE = 1,       /* frames do not reach the device of the destination GID */
    FV_DEVICE_FAILED = 2,    /* the sender, the receiver or a relay is down */
    FV_NO_REMOTE_QP = 3,     /* no queue pair with the destination number on that device */
    FV_NOT_PAIRED = 4,       /* the remote queue pair is connected to someone else */
    FV_REMOTE_ACCESS = 5,    /* remote key or bounds */
    FV_LOCAL_PROTECTION = 6, /* local key or bounds */
    FV_INJECTED = 7,         /* fv_inject_failure */
    FV_FLUSHED = 8,          /* an earlier write of the queue pair failed */
};

typedef struct {
    uint64_t sequence;     /* global order of events */
    uint32_t phase;        /* 0 posted, 1 executed */
    uint32_t status;       /* ibv_wc_status at execution */
    int32_t src_device;
    int32_t dst_device;    /* -1 when the queue pair is not routable */
    uint32_t qp_num;
    uint32_t flags;        /* bit 0 signaled, bit 1 inline */
    uint64_t wr_id;
    uint64_t local_addr;
    uint64_t remote_addr;
    uint32_t length;
    uint32_t inline_word;  /* first four bytes of inline data */
    uint32_t relays;
    uint32_t reason;       /* FV_* */
    uint64_t relay_nodes;  /* bit n set when node n relayed the frame */
} fv_event_t;

typedef struct {
    int32_t routable;
    int32_t dst_device;
    uint32_t relays;
    uint32_t reserved;
    uint64_t relay_nodes;
} fv_route_t;

FV_API void fv_reset(void);
FV_API int fv_add_device(const char *name, int node, int port, int function, const uint8_t *gid);
FV_API int fv_add_cable(int node_a, int port_a, int node_b, int port_b, uint32_t function_mask);
FV_API void fv_set_relay(int node, int allowed);
FV_API void fv_set_relay_depth(int depth);
FV_API int fv_set_dest_tag(int src_device, const uint8_t *dgid, uint32_t relays);
FV_API void fv_set_ideal(int ideal);
/* Path latency: a write executes no earlier than base + per_relay * relays after it was posted, and its
 * completion becomes visible that long after it executed (0, 0: none, the default). */
FV_API void fv_set_latency(uint64_t base_ns, uint64_t per_relay_ns);
/* Off: writes of more than four bytes move no payload (flags still land); on by default. */
FV_API void fv_set_payloads(int on);
/* Each queue pair sends its writes one after another at `bytes_per_us` before their path latency
 * (0, the default: no sending time). */
FV_API void fv_set_rate(uint64_t bytes_per_us);
/* Every completion returns `ns` later than its path latency alone gives (0, the default). */
FV_API void fv_set_ack_delay(uint64_t ns);
FV_API void fv_set_recording(int on);
/* The next inline write of `inline_value` on queue pair `qp_num` (0: any) fails. */
FV_API void fv_inject_failure(uint32_t qp_num, uint32_t inline_value);
/* The next `calls` calls of ibv_destroy_qp fail with EBUSY and leave their queue pair in place
 * (fv_reset releases it); 0 ends the failures. */
FV_API void fv_fail_teardown(int calls);
FV_API void fv_fail_device(int device);
FV_API uint64_t fv_progress(uint64_t max_work_requests, uint64_t seed);
FV_API uint64_t fv_pending(void);
FV_API uint64_t fv_executed(void);
FV_API uint64_t fv_event_count(void);
FV_API int fv_event(uint64_t index, fv_event_t *out);
FV_API void fv_clear_events(void);
FV_API int fv_qp_route(uint32_t qp_num, fv_route_t *out);
/* Largest number of bytes the queue pair had posted and not yet retired by a
 * completion (UINT64_MAX for an unknown queue pair). */
FV_API uint64_t fv_qp_max_inflight(uint32_t qp_num);

#ifdef __cplusplus
}
#endif

#endif /* SIRCL_FAKE_VERBS_H */
