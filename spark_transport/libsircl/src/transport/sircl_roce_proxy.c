/*
 * SIRCL ring sessions: verbs setup and the native progress thread.
 *
 * One context serves one rank of one session, a group or subgroup of 2 to 16
 * ranks. The rank owns one arena: pinned host memory that the GPU addresses
 * at its host pointer and that every opened RDMA device registers for local
 * and remote writes. Its areas, in order:
 *
 *   recv[source][slot]  world * SLOTS * slot_bytes             written by peers
 *   flag lines          world * SLOTS * FLAG_LINES * FLAG_STRIDE
 *   send[slot]          SLOTS * slot_bytes                     staged by the kernel
 *   control             FLAG_STRIDE                            the device command ring
 *
 * The kernels and this thread talk through the command ring (32-bit words):
 *   0       doorbell: newest op whose first network phase the kernel released
 *   1       byte count of that op
 *   2, 3, 6 written by a kernel whose flag wait timed out: sequence, peer, lane
 *   4, 5    op word of slot 0 and slot 1: op code in bits 30-31, bytes below
 *   10-16   phase doorbells: word 9 + k is the newest op whose phase k (1-7)
 *           the kernel released
 *   17-24   descriptors: word 17 + k describes phase k of a described op
 *   31      lane-check landing word, written by peers during setup only
 *
 * Every peer is reached over `lane_count` (1 or 2) lanes. A lane is a
 * reliable-connected queue pair between one local device and one device of
 * the peer, named by the route map; a device holds one queue pair per peer it
 * carries a lane to. Members without a shared cable are joined through NIC
 * relays that site tooling installs (flow label 0; the relays match the
 * destination address). A payload of P packs (16 bytes each) is split over a
 * peer's lanes: lane l starts l * floor(P / L) + min(l, P mod L) packs in and
 * holds floor(P / L) packs, plus one when l < P mod L. Each lane posts its
 * stripe as one RDMA write and then, on the same queue pair, a 4-byte inline
 * write of the op's sequence into the lane's flag line, so a flag that shows
 * the sequence proves its bytes have landed. Flag line of namespace ns,
 * source s, slot t and lane l: ns * world * SLOTS * L + (s * SLOTS + t) * L + l.
 * A lane with a forward window (roce_set_forward; lanes through relays) posts
 * a stripe longer than one chunk, or one that does not fit the window, as
 * signaled chunks instead, keeping its bytes in flight within the window, and
 * its flag after the last chunk.
 *
 * Op codes: 0 one-shot (whole payload to every peer, namespace 0; also the
 * all-gather); 1 two-shot all-reduce (phase 0: chunk p to peer p, namespace
 * 0; phase 1, after the kernel reduced its own chunk into the same offsets of
 * send[slot]: the own chunk to every peer, namespace 1); 2 described op (each
 * phase k writes the chunk range of descriptor k to its peer, in its
 * namespace); 3 scatter op (phase 0 of the two-shot posting only). Chunk j of
 * P packs is [floor(j * P / W), floor((j + 1) * P / W)). When the session
 * cannot run multi-phase ops (slots of 2^30 bytes or more), op words are plain
 * byte counts of one-shot ops.
 *
 * The doorbell holds only the newest sequence. A rank cannot get more than
 * one op ahead of its peers, so at most SLOTS ops are pending; missed ones are
 * posted in order from their op words. A phase is posted only after the
 * previous phase of the same op.
 *
 * Chain schedule (roce_set_chain): a pipelined all-reduce between cable
 * neighbors only, for groups whose ranks form a chain (a path, or a ring used
 * as one). Half A of a message reduces from the chain's first rank to its
 * last, and the last rank's result travels back; half B mirrors it. Four
 * streams of chunks run between neighbors, each continuous across ops:
 *
 *   0  A reduction   toward the next rank      partials, staged by the kernel
 *   1  A broadcast   toward the previous rank  results: staged by the kernel of
 *                                              the last rank, forwarded by
 *                                              this thread elsewhere
 *   2  B reduction   toward the previous rank  partials, staged by the kernel
 *   3  B broadcast   toward the next rank      results: staged by the kernel of
 *                                              the first rank, forwarded elsewhere
 *
 * Chunk g of a stream (counted from 0 over the session) uses slot g % K of
 * the stream's rings, and its flag lines and progress words carry g + 1. The
 * chain area after the control line holds, per stream: K receive slots
 * written by the upstream neighbor, K send slots staged by the kernel, a flag
 * line per (slot, lane), a line of K ready words (the kernel writes g + 1 when
 * send slot g % K holds chunk g), a line of K consumed words (the kernel
 * writes g + 1 when it finished reading receive slot g % K), a sent word (this
 * thread: chunks whose writes completed, so the kernel may restage their
 * slots) and a credit word (the downstream neighbor: chunks it released, so
 * their slots may be rewritten). The chain control line holds the chain
 * doorbell (word 0) and two parameter slots (words 1 + 4p: bytes of half A,
 * bytes of half B, chunk bytes). This thread posts a chunk when its source is
 * ready and the downstream slot is free, and returns credit for an inbound
 * chunk once the kernel consumed it and, for a forwarded stream, once its
 * forward completed.
 *
 * Chain links (roce_set_links): the chain all-gather and reduce-scatter on
 * the same chains, and the ring collectives over the chain closed by its last
 * rank's lanes to its first (through relays on a path). Link 0 carries items
 * toward the next rank in chain order and link 1 toward the previous one;
 * links 2 (ring partial sums) and 3 (ring results) carry items toward the
 * ring's next rank. Items are counted per link and direction over the
 * session (outbound, inbound and own counts). An op of P pieces per rank runs
 * P rounds per link; a round carries one piece of each owner the link serves
 * at this rank: first the own items (staged by the kernel in the link's own
 * slots), then items forwarded from the inbound round in its order (the chain
 * all-gather forwards every piece it receives, the ring all-gather all but the
 * last of a round; the reduce-scatters' kernels add their own values and
 * stage every outbound item). A link toward a peer reached through relays
 * posts each lane's stripe in forward-window chunks, keeping at most the
 * lane's window unacknowledged, and the lane's flag after its last chunk. The link area after
 * the chain area holds, per link: K receive slots, K own slots, a flag line
 * per (receive slot, lane), a line of ready words (own slots), a line of
 * consumed words (receive slots), a sent word (own items whose writes
 * completed) and a credit word (inbound items the downstream neighbor
 * released); its control line holds the link doorbell (word 0) and two
 * parameter slots (words 1 + 4p: op, bytes per rank, piece bytes). An
 * inbound item is released once the kernel consumed it and, when it is
 * forwarded, once its forward completed.
 *
 * Plain C over libibverbs and POSIX threads, built by sparkring_sircl's
 * oneshot._proxy with the host compiler. The CPU simulator builds it against
 * the test-only verbs stand-in in sparkring_sircl/testing/fake_verbs.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <sys/socket.h>
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

/* SIRCL's own native ABI counter; changes with the record, layout, ring words,
 * op codes or any exported signature. */
#define ROCE_ABI_VERSION 9
/* First word of every connection record ("SRCL" in byte order). */
#define ROCE_RECORD_MAGIC 0x4c435253u
#define ROCE_MAX_PEERS 16
#define ROCE_MAX_DEVICES 4
#define ROCE_MAX_LANES 2
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_FLAG_LINES 4
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 256
/* Signaled work requests in flight per queue pair. Each retires at most two
 * work requests (a payload write and its flag), so the send queue stays at most
 * half full. */
#define ROCE_CREDIT (ROCE_SEND_DEPTH / 4)
/* Arena op codes remembered for delivery proofs (a peer is at most one op ahead of this rank). */
#define ROCE_OP_HISTORY 8
/* ack_mark bits: an arena write, its namespace is 1, proven delivered. */
#define ROCE_ACK_ARENA 1u
#define ROCE_ACK_NS1 2u
#define ROCE_ACK_PROVEN 4u
/* Smallest part of a forward-window chunk posted when the window has no room for the whole chunk. */
#define ROCE_FWD_MIN_PIECE 4096u
#define ROCE_MAX_STREAMS (ROCE_MAX_PEERS * ROCE_MAX_LANES)
#define ROCE_CQ_DEPTH (ROCE_SEND_DEPTH * ROCE_MAX_PEERS)
#ifndef ROCE_IDLE_SPINS
#define ROCE_IDLE_SPINS 20000000ull
#endif
#define ROCE_NAP_NS 20000L
#define ROCE_NO_DEVICE 0xFFu
#define ROCE_MAX_PHASES 8
#define ROCE_OP_SHIFT 30
#define ROCE_OP_BYTES_MASK ((1u << ROCE_OP_SHIFT) - 1u)

enum {
    ROCE_CTRL_DOORBELL = 0,
    ROCE_CTRL_NBYTES = 1,
    ROCE_CTRL_ERROR_SEQ = 2,
    ROCE_CTRL_MISSING_PEER = 3,
    ROCE_CTRL_OP_WORD = 4,
    ROCE_CTRL_MISSING_LANE = 6,
    ROCE_CTRL_PHASE = 9,
    ROCE_CTRL_WAIT_LIMIT_US = 7,
    ROCE_CTRL_DESC = 17,
    ROCE_CTRL_LANE_CHECK = 31,
};

enum { ROCE_OP_ONESHOT = 0, ROCE_OP_TWOSHOT = 1, ROCE_OP_DESCRIBED = 2, ROCE_OP_SCATTER = 3 };

/* Work-request ids: peer in bits 0-7, lane in 8-15, kind in 16-23, chain stream in
 * 24-31, sequence (or global chain chunk) in 32-63. */
enum { ROCE_WR_OP = 0, ROCE_WR_CHECK = 1, ROCE_WR_CHAIN = 2, ROCE_WR_CREDIT = 3, ROCE_WR_LINK = 4,
       ROCE_WR_LINK_CREDIT = 5, ROCE_WR_LINK_DATA = 6 };

#define ROCE_CHAIN_STREAMS 4
#define ROCE_CHAIN_MAX_SLOTS 32
#define ROCE_CHAIN_OPS 8
#define ROCE_CHAIN_LAYOUT_WORDS 9
enum { ROCE_CHAIN_NONE = 0, ROCE_CHAIN_KERNEL = 1, ROCE_CHAIN_FORWARD = 2 };
/* Chain area offsets (roce_chain_layout): receive, send, flag lines, ready, consumed,
 * sent, credit, control, total. */
enum { CH_RECV = 0, CH_SEND = 1, CH_RFLAG = 2, CH_READY = 3, CH_CONSUMED = 4, CH_SENT = 5,
       CH_CREDIT = 6, CH_CTRL = 7, CH_TOTAL = 8 };

#define ROCE_LINKS 4
/* Chunk of a ring link's stripe toward a peer through relays. */
#define ROCE_LINK_WINDOW_CHUNK 32768u
#define ROCE_LINK_MAX_SLOTS 32
#define ROCE_LINK_OPS 8
enum { ROCE_LINK_GATHER = 1, ROCE_LINK_SCATTER = 2, ROCE_RING_GATHER = 3, ROCE_RING_SCATTER = 4,
       ROCE_RING_REDUCE = 5 };
/* A ring reduce-scatter's stagger D (link 2): bits 8-15 of its op word (protocol.RING_STAGGER_SHIFT);
 * a ring all-gather's stagger D3 (link 3): bits 16-23 (protocol.RING_GATHER_STAGGER_SHIFT). */
#define ROCE_STAGGER_SHIFT 8
#define ROCE_GATHER_STAGGER_SHIFT 16
#define ROCE_MAX_STAGGER 4u
/* Link area offsets (roce_link_layout): receive, own, flag lines, ready, consumed, sent,
 * credit, control, total. */
enum { LK_RECV = 0, LK_OWN = 1, LK_RFLAG = 2, LK_READY = 3, LK_CONSUMED = 4, LK_SENT = 5, LK_CREDIT = 6,
       LK_CTRL = 7, LK_TOTAL = 8 };

/* Event trace (roce_set_trace): one record per event of a chain stream (streams 0-3)
 * or link (stream 4 + link), in a ring of the trace's capacity. */
typedef struct {
    uint64_t ns;              /* CLOCK_REALTIME */
    uint32_t value;           /* chunk or item tag, credit, or op sequence */
    uint16_t event;
    uint16_t stream;
} roce_trace_rec_t;
/* OP: an op taken from its doorbell (stream 0 chain ops, 4 link ops; value: its sequence).
 * READY: the progress thread first saw an outbound chunk or item ready (the kernel staged it, or
 * every lane flag of the inbound one it forwards arrived). POSTED: its writes posted on every
 * lane. DONE: its writes completed on every lane. CONSUMED: the kernel finished an inbound
 * chunk or item. CREDIT_OUT: credit written upstream. CREDIT_IN: the credit word from
 * downstream changed (value: the credit). */
enum { ROCE_EV_OP = 1, ROCE_EV_READY = 2, ROCE_EV_POSTED = 3, ROCE_EV_DONE = 4, ROCE_EV_CONSUMED = 5,
       ROCE_EV_CREDIT_OUT = 6, ROCE_EV_CREDIT_IN = 7 };
#define ROCE_TRACE_LINK_STREAM 4
#define ROCE_TRACE_MAX_RECORDS (1u << 24)

/* Test hook points (compiled in with SIRCL_PROXY_TEST_HOOKS only). */
enum { ROCE_HOOK_DOORBELL = 0, ROCE_HOOK_PEER = 1, ROCE_HOOK_LANE = 2, ROCE_HOOK_PHASE = 3 };

typedef struct {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t world;
    uint32_t rank;
    uint32_t lane_count;
    uint32_t n_devices;
    uint64_t slot_bytes;
    uint64_t region_addr;
    uint32_t rkey[ROCE_MAX_DEVICES];
    uint32_t mtu[ROCE_MAX_DEVICES];
    uint16_t lid[ROCE_MAX_DEVICES];
    uint8_t gid[ROCE_MAX_DEVICES][16];
    uint32_t qp_num[ROCE_MAX_DEVICES][ROCE_MAX_PEERS];
    uint8_t lane_device[ROCE_MAX_PEERS][ROCE_MAX_LANES];
} roce_record_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[ROCE_MAX_PEERS];
    /* Per queue pair (toward each peer): the bytes each signaled work request
     * in flight retires, oldest first, and their sum. Completions of one queue
     * pair arrive in posting order. An arena write's entry also names its op
     * and namespace (ack_seq, ack_mark); unacked leaves out the entries the op
     * order proved delivered (fwd_prove). */
    uint32_t ack_bytes[ROCE_MAX_PEERS][ROCE_SEND_DEPTH];
    uint32_t ack_seq[ROCE_MAX_PEERS][ROCE_SEND_DEPTH];
    uint8_t ack_mark[ROCE_MAX_PEERS][ROCE_SEND_DEPTH];
    uint16_t ack_head[ROCE_MAX_PEERS];
    uint16_t ack_count[ROCE_MAX_PEERS];
    uint64_t unacked[ROCE_MAX_PEERS];
    _Atomic uint64_t writes_completed;
    _Atomic uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
    int gid_index;
    char name[64];
} roce_dev_t;

/* A stripe on a lane with a forward window, posted in chunks by pump_streams. */
typedef struct {
    int peer;
    int lane;
    int dev;
    uint32_t seq;
    const uint8_t *source;
    uint64_t remote_data;
    uint64_t remote_flag;
    uint32_t rkey;
    uint32_t length;
    uint32_t done;
    int flag_posted;
    int ns;                 /* flag namespace of the stripe's phase */
} roce_stream_t;

/* One chain op: bytes, chunk size and the global chunk range of each half. */
typedef struct {
    uint32_t seq;
    uint32_t bytes[2];
    uint32_t chunk;
    uint32_t first[2];
    uint32_t count[2];
} roce_chain_op_t;

typedef struct {
    int in_peer;              /* upstream neighbor of the stream, -1: none */
    int out_peer;             /* downstream neighbor, -1: none */
    int out_kind;             /* ROCE_CHAIN_NONE, _KERNEL or _FORWARD */
    int half;                 /* 0: half A, 1: half B */
    uint32_t next_post;       /* next chunk to write downstream */
    uint32_t lane_done[ROCE_MAX_LANES];  /* chunks whose downstream writes completed, per lane */
    uint32_t done;            /* the smallest of lane_done */
    uint32_t consumed;        /* inbound chunks the kernel finished, contiguous */
    uint32_t credited;        /* credit last written upstream */
    uint32_t ready_seen;      /* tag of the newest chunk traced as ready */
    uint32_t credit_seen;     /* credit word last traced */
} roce_chain_stream_t;

/* One link op: its kind and pieces, and per link the items per round and the first
 * outbound, inbound and own item. */
typedef struct {
    uint32_t seq;
    uint32_t op;              /* ROCE_LINK_GATHER or ROCE_LINK_SCATTER; 0: free entry */
    uint32_t bytes;           /* bytes per rank */
    uint32_t piece;           /* piece bytes */
    uint32_t pieces;
    uint32_t stagger;         /* ring reduce-scatter stagger D (link 2) */
    uint32_t stagger3;        /* ring all-gather stagger D3 (link 3) */
    uint32_t own_flags;       /* op word bit 24: every own item goes out as its flags only */
    uint32_t rounds[ROCE_LINKS];  /* rounds per link: pieces, plus (W - 2) D or (W - 2) D3 on a staggered link */
    uint32_t out_round[ROCE_LINKS], in_round[ROCE_LINKS], own_round[ROCE_LINKS];
    uint32_t first_out[ROCE_LINKS], first_in[ROCE_LINKS], first_own[ROCE_LINKS];
} roce_link_op_t;

typedef struct {
    int out_peer;             /* downstream neighbor, -1: none */
    int in_peer;              /* upstream neighbor, -1: none */
    uint32_t next_post;       /* next outbound item */
    uint32_t lane_done[ROCE_MAX_LANES];  /* outbound items whose writes completed, per lane */
    uint32_t done;            /* outbound items accounted as completed */
    uint32_t own_done;        /* own items among them (the sent word) */
    uint32_t fwd_done;        /* inbound items whose forwards completed (index + 1 of the newest) */
    uint32_t consumed;        /* inbound items the kernel finished, contiguous */
    uint32_t released;        /* inbound items released (consumed, and forwarded when forwarded) */
    uint32_t credited;        /* credit last written upstream */
    uint32_t known_out, known_in, known_own;  /* items of every taken op */
    /* Through relays (forward windows toward out_peer): per lane, the item being posted and its
     * stripe bytes posted so far. */
    int windowed;
    uint32_t lane_item[ROCE_MAX_LANES];
    uint32_t lane_bytes[ROCE_MAX_LANES];
    uint32_t ready_seen;      /* tag of the newest item traced as ready */
    uint32_t credit_seen;     /* credit word last traced */
    uint64_t orphan_ns;       /* since when a finished inbound item waits for its op to be taken; 0: none */
} roce_link_t;

typedef struct roce_ctx {
    int world;
    int rank;
    int n_dev;
    int lane_count;
    int traffic_class;
    int multi_phase;
    roce_dev_t dev[ROCE_MAX_DEVICES];
    int lane_device[ROCE_MAX_PEERS][ROCE_MAX_LANES];
    uint32_t peer_rkey[ROCE_MAX_LANES][ROCE_MAX_PEERS];
    uint8_t peer_gid[ROCE_MAX_PEERS][ROCE_MAX_LANES][16];
    uint64_t peer_addr[ROCE_MAX_PEERS];
    uint8_t *region;
    uint64_t region_bytes;
    uint64_t slot_bytes;
    uint64_t recv_off;
    uint64_t flag_off;
    uint64_t send_off;
    uint64_t ctrl_off;
    int post_order[ROCE_MAX_PEERS];
    int connected;
    int started;
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    uint32_t last_seq;
    uint32_t posting_seq;      /* the op being posted, for error messages */
    uint32_t last_phase_seq[ROCE_MAX_PHASES];
    _Atomic uint32_t posted_seq;
    _Atomic uint64_t ops_posted;
    _Atomic uint64_t writes_completed;
    _Atomic uint64_t phases_posted;
    _Atomic uint64_t cpu_migrations;
    atomic_int last_cpu;
    /* Forward windows (bytes in flight per lane; 0: none) and their chunk size. */
    uint32_t fwd_window[ROCE_MAX_PEERS][ROCE_MAX_LANES];
    uint32_t fwd_chunk;
    roce_stream_t streams[ROCE_MAX_STREAMS];
    int n_streams;
    _Atomic uint64_t fwd_chunks_posted;
    _Atomic uint64_t fwd_max_unacked;
    /* SIRCL_FORWARD_PROOF (default on): bytes proven delivered leave the forward windows. The op
     * codes of the newest posted arena ops, and per peer the newest flags a proof last used. */
    int fwd_proof;
    uint32_t op_seq[ROCE_OP_HISTORY];
    uint8_t op_code[ROCE_OP_HISTORY];
    uint32_t proof_flags[ROCE_MAX_PEERS][2];
    /* Ops posted (the current op's index, from 1) and, per flag namespace and slot, the indices of the
     * two newest ops that had every peer write that flag line (`line_every`) or that may have written
     * it (`line_any`, described ops included); 0: none. fwd_prove trusts a line through them. */
    uint64_t op_index;
    uint64_t line_every[2][ROCE_SLOTS][2];
    uint64_t line_any[2][ROCE_SLOTS][2];
    _Atomic uint64_t fwd_proven_bytes;
    /* Waits of windowed stripes for their window: count, total and longest nanoseconds. */
    _Atomic uint64_t fwd_waits, fwd_wait_ns, fwd_wait_max_ns;
    /* Chain schedule (roce_set_chain); chain_slots 0: none. */
    int chain_slots;
    uint64_t chain_slot_bytes;
    uint64_t chain_off;
    uint64_t chain_layout[ROCE_CHAIN_LAYOUT_WORDS];
    roce_chain_stream_t chain[ROCE_CHAIN_STREAMS];
    roce_chain_op_t chain_ops[ROCE_CHAIN_OPS];
    uint32_t chain_last_seq;
    uint32_t chain_known[2];
    _Atomic uint64_t chain_ops_seen;
    _Atomic uint64_t chain_chunks_posted;
    _Atomic uint64_t chain_credits_sent;
    _Atomic uint64_t chain_bytes_posted;
    /* Chain links (roce_set_links); link_slots 0: none. */
    int link_slots;
    int link_index;           /* this rank's chain index */
    uint64_t link_slot_bytes;
    uint64_t link_off;
    uint64_t link_layout[ROCE_CHAIN_LAYOUT_WORDS];
    roce_link_t link[ROCE_LINKS];
    roce_link_op_t link_ops[ROCE_LINK_OPS];
    uint32_t link_last_seq;
    uint32_t link_ring_window;  /* bytes in flight per lane on a ring link through relays; 0: direct */
    _Atomic uint64_t link_ops_seen;
    _Atomic uint64_t link_items_posted;
    _Atomic uint64_t link_credits_sent;
    _Atomic uint64_t link_bytes_posted;
    _Atomic uint64_t link_window_chunks;  /* windowed chunks of ring links through relays */
    /* Event trace (roce_set_trace); trace_cap 0: off. The progress thread appends and
     * publishes trace_written; roce_trace_take hands records out from trace_taken. */
    roce_trace_rec_t *trace;
    uint32_t trace_cap;
    _Atomic uint64_t trace_written;
    uint64_t trace_taken;
    uint64_t test_link_delay_ns;  /* tests: a link op is taken this long after its doorbell is seen */
    uint64_t test_link_seen_ns;   /* tests: when the pending link doorbell was first seen */
    char err[512];
} roce_ctx_t;

int roce_destroy(roce_ctx_t *c);

#ifdef SIRCL_PROXY_TEST_HOOKS
static void (*test_hook)(void *arg, int point, uint32_t seq, int peer);
static void *test_hook_arg;
void roce_test_set_hook(void (*fn)(void *, int, uint32_t, int), void *arg) {
    test_hook = fn;
    test_hook_arg = arg;
}
void roce_test_disable_multi_phase(roce_ctx_t *c) { c->multi_phase = 0; }
/* Take every link op `delay_us` after its doorbell is first seen (0: at once), so a test can
 * make the kernel finish inbound items before their op is taken. */
void roce_test_delay_link_ops(roce_ctx_t *c, uint32_t delay_us) {
    c->test_link_delay_ns = (uint64_t)delay_us * 1000ull;
    c->test_link_seen_ns = 0;
}
uint32_t roce_test_qp_num(roce_ctx_t *c, int d, int p) {
    return c->dev[d].qp[p] != NULL ? c->dev[d].qp[p]->qp_num : 0;
}
#define HOOK(point, seq, peer) \
    do { if (test_hook) test_hook(test_hook_arg, (point), (seq), (peer)); } while (0)
#else
#define HOOK(point, seq, peer) do { } while (0)
#endif

#define FAIL(c, ...) snprintf((c)->err, sizeof((c)->err), __VA_ARGS__)

static void trace_event(roce_ctx_t *c, int event, int stream, uint32_t value) {
    if (c->trace_cap == 0) return;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t n = atomic_load_explicit(&c->trace_written, memory_order_relaxed);
    roce_trace_rec_t *r = &c->trace[n % c->trace_cap];
    r->ns = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    r->value = value;
    r->event = (uint16_t)event;
    r->stream = (uint16_t)stream;
    atomic_store_explicit(&c->trace_written, n + 1, memory_order_release);
}

/* Trace a credit word that changed since the last trace of the stream. */
static void trace_credit_in(roce_ctx_t *c, int stream, uint32_t credit, uint32_t *seen) {
    if (c->trace_cap == 0 || credit == *seen) return;
    *seen = credit;
    trace_event(c, ROCE_EV_CREDIT_IN, stream, credit);
}

/* Trace an outbound chunk or item first seen ready (tags only grow along a stream). */
static void trace_ready(roce_ctx_t *c, int stream, uint32_t tag, uint32_t *seen) {
    if (c->trace_cap == 0 || (int32_t)(tag - *seen) <= 0) return;
    *seen = tag;
    trace_event(c, ROCE_EV_READY, stream, tag);
}

static volatile uint32_t *ctrl_words(const roce_ctx_t *c) {
    return (volatile uint32_t *)(c->region + c->ctrl_off);
}

/* -- arithmetic shared with the kernels (checked by tests/data/numeric.json) - */

/* Stripe of lane `lane` among `lanes` lanes for a payload of `packs` packs. */
static void lane_split(uint32_t packs, int lanes, int lane, uint32_t *first, uint32_t *count) {
    uint32_t base = packs / (uint32_t)lanes;
    uint32_t rest = packs % (uint32_t)lanes;
    *first = (uint32_t)lane * base + ((uint32_t)lane < rest ? (uint32_t)lane : rest);
    *count = base + ((uint32_t)lane < rest ? 1u : 0u);
}

/* Chunk j of `packs` packs over `world` ranks: [j*P/W, (j+1)*P/W). */
static void chunk_range(uint32_t packs, int world, int chunk, uint32_t *first, uint32_t *count) {
    uint32_t lo = (uint32_t)((uint64_t)chunk * packs / (uint32_t)world);
    uint32_t hi = (uint32_t)((uint64_t)(chunk + 1) * packs / (uint32_t)world);
    *first = lo;
    *count = hi - lo;
}

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

/* out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots} */
int roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
    uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, total;
    if (out == NULL || world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 ||
        slot_bytes % 4096u != 0 || slot_bytes > ((uint64_t)1 << 40)) {
        return -1;
    }
    if (__builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS * ROCE_FLAG_LINES,
                               (uint64_t)ROCE_FLAG_STRIDE, &flag_bytes) ||
        __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
        __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
        __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
        __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_FLAG_STRIDE, &total)) {
        return -1;
    }
    out[0] = 0;
    out[1] = recv_bytes;
    out[2] = send_off;
    out[3] = ctrl_off;
    out[4] = total;
    out[5] = ROCE_FLAG_STRIDE;
    out[6] = ROCE_SLOTS;
    return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_record_t); }

int roce_flag_lines(void) { return ROCE_FLAG_LINES; }

/* Host-side stand-ins for the kernels' ordered accesses (the CPU harness and
 * tools that play the kernel's part): a release store after a full fence, so
 * staged bytes are visible before the word, and an acquire load, so bytes
 * read after a flag are those written before it. */
void roce_store_release_u32(volatile uint32_t *address, uint32_t value) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

uint32_t roce_load_acquire_u32(const volatile uint32_t *address) {
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

/* -- settings read at creation ------------------------------------------------ */

/* SIRCL_POST_ORDER: rank (default), ring-farthest or an explicit peer list. */
static int set_post_order(roce_ctx_t *c, char *err, uint64_t err_len) {
    const char *text = getenv("SIRCL_POST_ORDER");
    int n = 0;
    if (text == NULL || *text == '\0' || strcmp(text, "rank") == 0) {
        for (int p = 0; p < c->world; p++) {
            if (p != c->rank) c->post_order[n++] = p;
        }
        return 0;
    }
    if (strcmp(text, "ring-farthest") == 0) {
        for (int d = c->world / 2; d >= 1; d--) {
            int cw = (c->rank + d) % c->world;
            int ccw = (c->rank + c->world - d) % c->world;
            c->post_order[n++] = cw;
            if (ccw != cw) c->post_order[n++] = ccw;
        }
        return 0;
    }
    uint32_t seen = 0;
    const char *p = text;
    while (*p != '\0') {
        while (*p == ' ') p++;
        char *end = NULL;
        long peer = strtol(p, &end, 10);
        if (end == p || peer < 0 || peer >= c->world || peer == c->rank ||
            (seen & (1u << peer)) != 0 || n >= c->world - 1) {
            snprintf(err, (size_t)err_len,
                     "SIRCL_POST_ORDER=%s is not rank, ring-farthest or a list naming every peer "
                     "of rank %d exactly once", text, c->rank);
            return -1;
        }
        seen |= 1u << peer;
        c->post_order[n++] = (int)peer;
        p = end;
        while (*p == ' ') p++;
        if (*p == ',') {
            p++;
        } else if (*p != '\0') {
            snprintf(err, (size_t)err_len, "SIRCL_POST_ORDER=%s is malformed", text);
            return -1;
        }
    }
    if (n != c->world - 1) {
        snprintf(err, (size_t)err_len, "SIRCL_POST_ORDER=%s must name all %d peers of rank %d",
                 text, c->world - 1, c->rank);
        return -1;
    }
    return 0;
}

/* SIRCL_POST_MODE: this build posts with ibv_post_send only. */
static int check_post_mode(char *err, uint64_t err_len) {
    const char *mode = getenv("SIRCL_POST_MODE");
    if (mode == NULL || *mode == '\0' || strcmp(mode, "verbs") == 0) return 0;
    if (strcmp(mode, "direct") == 0) {
        snprintf(err, (size_t)err_len,
                 "SIRCL_POST_MODE=direct (mlx5 send-queue entries) is unsupported by this "
                 "SIRCL build; use verbs");
    } else {
        snprintf(err, (size_t)err_len, "SIRCL_POST_MODE=%s is not verbs or direct", mode);
    }
    return -1;
}

/* SIRCL_PROGRESS_CPU: CPU list such as 9 or 5-9,15-19. */
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

/* -- setup ---------------------------------------------------------------------- */

static int open_device(roce_ctx_t *c, int d, const char *name, int gid_index) {
    roce_dev_t *dev = &c->dev[d];
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
    if (ibv_query_port(dev->ctx, ROCE_PORT, &port) != 0) {
        FAIL(c, "ibv_query_port(%s): %s", name, strerror(errno));
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        FAIL(c, "RDMA device %s port %d is not active", name, ROCE_PORT);
        return -1;
    }
    dev->lid = port.lid;
    dev->mtu = port.active_mtu;
    if (ibv_query_gid(dev->ctx, ROCE_PORT, gid_index, &dev->gid) != 0) {
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
    dev->cq = ibv_create_cq(dev->ctx, ROCE_CQ_DEPTH, NULL, NULL, 0);
    if (dev->cq == NULL) {
        FAIL(c, "ibv_create_cq(%s): %s", name, strerror(errno));
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        int used = 0;
        for (int l = 0; l < c->lane_count; l++) used |= c->lane_device[p][l] == d;
        if (!used) continue;
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = dev->cq;
        attr.recv_cq = dev->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = ROCE_SEND_DEPTH;
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
        init.port_num = ROCE_PORT;
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

/* Route table rules of one rank as far as the native layer sees them: every
 * other rank has lane_count lanes, the own rank none, and every lane names an
 * opened device. */
static int check_route_table(int world, int rank, int n_dev, const int *lane_devices,
                             int lane_count, char *err, uint64_t err_len) {
    for (int p = 0; p < world; p++) {
        for (int l = 0; l < lane_count; l++) {
            int d = lane_devices[p * lane_count + l];
            if (p == rank) {
                if (d != -1) {
                    snprintf(err, (size_t)err_len, "rank %d names a device for its own lane %d", rank, l);
                    return -1;
                }
                continue;
            }
            if (d < 0 || d >= n_dev) {
                snprintf(err, (size_t)err_len,
                         "rank %d lane %d toward rank %d names device index %d outside 0-%d",
                         rank, l, p, d, n_dev - 1);
                return -1;
            }
            for (int k = 0; k < l; k++) {
                if (lane_devices[p * lane_count + k] == d) {
                    snprintf(err, (size_t)err_len,
                             "rank %d uses device index %d for two lanes toward rank %d", rank, d, p);
                    return -1;
                }
            }
        }
    }
    return 0;
}

roce_ctx_t *roce_create(int world, int rank, const char *const *device_names, int n_devices,
                        const int *lane_devices, int lane_count, const int *gid_indices,
                        int traffic_class, void *region, uint64_t region_bytes,
                        uint64_t slot_bytes, char *err, uint64_t err_len) {
    uint64_t layout[7];
    if (err == NULL || err_len == 0) return NULL;
    err[0] = '\0';
    if (roce_layout(world, slot_bytes, layout) != 0 || rank < 0 || rank >= world ||
        layout[4] > region_bytes || region == NULL) {
        snprintf(err, (size_t)err_len,
                 "invalid session geometry: world %d, rank %d, slot bytes %llu, arena %llu bytes",
                 world, rank, (unsigned long long)slot_bytes, (unsigned long long)region_bytes);
        return NULL;
    }
    if (n_devices < 1 || n_devices > ROCE_MAX_DEVICES || device_names == NULL) {
        snprintf(err, (size_t)err_len, "a session opens 1 to %d RDMA devices, got %d",
                 ROCE_MAX_DEVICES, n_devices);
        return NULL;
    }
    if (lane_count < 1 || lane_count > ROCE_MAX_LANES || lane_devices == NULL) {
        snprintf(err, (size_t)err_len, "lane count must be 1 or 2, got %d", lane_count);
        return NULL;
    }
    if (traffic_class < 0 || traffic_class > 255) {
        snprintf(err, (size_t)err_len, "traffic class %d is outside 0-255", traffic_class);
        return NULL;
    }
    for (int d = 0; d < n_devices; d++) {
        if (device_names[d] == NULL || device_names[d][0] == '\0' || gid_indices == NULL ||
            gid_indices[d] < 0 || gid_indices[d] > 255) {
            snprintf(err, (size_t)err_len, "device %d needs a name and a GID index in 0-255", d);
            return NULL;
        }
    }
    if (check_route_table(world, rank, n_devices, lane_devices, lane_count, err, err_len) != 0) {
        return NULL;
    }
    if (check_post_mode(err, err_len) != 0) return NULL;
    roce_ctx_t *c = (roce_ctx_t *)calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, (size_t)err_len, "out of memory");
        return NULL;
    }
    c->world = world;
    c->rank = rank;
    c->n_dev = n_devices;
    c->lane_count = lane_count;
    c->traffic_class = traffic_class;
    c->region = (uint8_t *)region;
    c->region_bytes = region_bytes;
    c->slot_bytes = slot_bytes;
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    c->multi_phase = 2 * lane_count <= ROCE_FLAG_LINES && slot_bytes <= ROCE_OP_BYTES_MASK;
    atomic_store(&c->last_cpu, -1);
    for (int p = 0; p < ROCE_MAX_PEERS; p++) {
        for (int l = 0; l < ROCE_MAX_LANES; l++) {
            c->lane_device[p][l] = (p < world && l < lane_count) ? lane_devices[p * lane_count + l] : -1;
        }
    }
    if (set_post_order(c, err, err_len) != 0) {
        free(c);
        return NULL;
    }
    const char *proof = getenv("SIRCL_FORWARD_PROOF");
    c->fwd_proof = proof == NULL || proof[0] == '\0' || strcmp(proof, "0") != 0;
    for (int d = 0; d < n_devices; d++) {
        if (open_device(c, d, device_names[d], gid_indices[d]) != 0) {
            snprintf(err, (size_t)err_len, "%s", c->err);
            roce_destroy(c);
            return NULL;
        }
    }
    return c;
}

int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {
    roce_record_t r;
    if (c == NULL || out == NULL || out_len < sizeof(r)) return -1;
    memset(&r, 0, sizeof(r));
    r.magic = ROCE_RECORD_MAGIC;
    r.abi_version = ROCE_ABI_VERSION;
    r.world = (uint32_t)c->world;
    r.rank = (uint32_t)c->rank;
    r.lane_count = (uint32_t)c->lane_count;
    r.n_devices = (uint32_t)c->n_dev;
    r.slot_bytes = c->slot_bytes;
    r.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int d = 0; d < c->n_dev; d++) {
        r.rkey[d] = c->dev[d].mr->rkey;
        r.mtu[d] = (uint32_t)c->dev[d].mtu;
        r.lid[d] = c->dev[d].lid;
        memcpy(r.gid[d], c->dev[d].gid.raw, 16);
        for (int p = 0; p < c->world; p++) {
            r.qp_num[d][p] = c->dev[d].qp[p] != NULL ? c->dev[d].qp[p]->qp_num : 0;
        }
    }
    for (int p = 0; p < ROCE_MAX_PEERS; p++) {
        for (int l = 0; l < ROCE_MAX_LANES; l++) {
            int d = c->lane_device[p][l];
            r.lane_device[p][l] = d < 0 ? ROCE_NO_DEVICE : (uint8_t)d;
        }
    }
    memcpy(out, &r, sizeof(r));
    return 0;
}

/* Every record, before any queue pair moves: magic, ABI, geometry and the
 * lane devices each rank claims for this rank. */
static int validate_records(roce_ctx_t *c, const roce_record_t *all) {
    for (int p = 0; p < c->world; p++) {
        const roce_record_t *r = &all[p];
        if (r->magic != ROCE_RECORD_MAGIC) {
            FAIL(c, "rank %d published a record of another protocol (magic 0x%08x)", p, r->magic);
            return -1;
        }
        if (r->abi_version != ROCE_ABI_VERSION || r->world != (uint32_t)c->world ||
            r->rank != (uint32_t)p || r->slot_bytes != c->slot_bytes ||
            r->lane_count != (uint32_t)c->lane_count) {
            FAIL(c, "rank %d record differs: ABI %u/%d, world %u/%d, rank %u, slot bytes %llu/%llu, "
                    "lanes %u/%d", p, r->abi_version, ROCE_ABI_VERSION, r->world, c->world, r->rank,
                 (unsigned long long)r->slot_bytes, (unsigned long long)c->slot_bytes,
                 r->lane_count, c->lane_count);
            return -1;
        }
        if (r->n_devices < 1 || r->n_devices > ROCE_MAX_DEVICES || r->region_addr == 0) {
            FAIL(c, "rank %d record names %u devices (1 to %d needed) or no arena", p,
                 r->n_devices, ROCE_MAX_DEVICES);
            return -1;
        }
        for (int q = 0; q < c->world; q++) {
            for (int l = 0; l < c->lane_count; l++) {
                uint8_t d = r->lane_device[q][l];
                if (q == p) {
                    if (d != ROCE_NO_DEVICE) {
                        FAIL(c, "rank %d record names a device for its own lane %d", p, l);
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

static int connect_lane(roce_ctx_t *c, int p, int l, const roce_record_t *peer) {
    int a = c->lane_device[p][l];
    int b = peer->lane_device[c->rank][l];
    roce_dev_t *dev = &c->dev[a];
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
    rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[b], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)dev->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = (uint8_t)c->traffic_class;
    rtr.ah_attr.grh.flow_label = 0;
    int rc = ibv_modify_qp(qp, &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
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
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        FAIL(c, "ibv_modify_qp(RTS) of lane %d toward rank %d on %s: %s", l, p, dev->name, strerror(rc));
        return -1;
    }
    c->peer_rkey[l][p] = peer->rkey[b];
    memcpy(c->peer_gid[p][l], peer->gid[b], 16);
    return 0;
}

int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (c->connected) {
        FAIL(c, "session is already connected");
        return -1;
    }
    if (blobs == NULL || blobs_len < sizeof(roce_record_t) * (uint64_t)c->world) {
        FAIL(c, "connection records: %llu bytes for %d ranks of %zu bytes",
             (unsigned long long)blobs_len, c->world, sizeof(roce_record_t));
        return -1;
    }
    const roce_record_t *all = (const roce_record_t *)blobs;
    if (validate_records(c, all) != 0) return -1;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) continue;
        c->peer_addr[p] = all[p].region_addr;
        for (int l = 0; l < c->lane_count; l++) {
            if (connect_lane(c, p, l, &all[p]) != 0) return -1;
        }
    }
    c->connected = 1;
    return 0;
}

/* -- completions and credit ----------------------------------------------------- */

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

static uint64_t make_wr_id(int peer, int lane, int kind, uint32_t seq) {
    return (uint64_t)(uint8_t)peer | ((uint64_t)(uint8_t)lane << 8) | ((uint64_t)(uint8_t)kind << 16) |
           ((uint64_t)seq << 32);
}

static uint64_t make_chain_wr_id(int peer, int lane, int kind, int stream, uint32_t chunk) {
    return make_wr_id(peer, lane, kind, chunk) | ((uint64_t)(uint8_t)stream << 24);
}

/* A signaled work request toward `p` on device `d` that retires `bytes`; `mark` and `seq` name an
 * arena write's op and namespace (mark 0: any other write). */
static void ack_push_tagged(roce_ctx_t *c, int d, int p, uint32_t bytes, uint8_t mark, uint32_t seq) {
    roce_dev_t *dev = &c->dev[d];
    uint32_t tail = (uint32_t)(dev->ack_head[p] + dev->ack_count[p]) % ROCE_SEND_DEPTH;
    dev->ack_bytes[p][tail] = bytes;
    dev->ack_mark[p][tail] = mark;
    dev->ack_seq[p][tail] = seq;
    dev->ack_count[p] += 1;
    dev->unacked[p] += bytes;
}

static void ack_push(roce_ctx_t *c, int d, int p, uint32_t bytes) {
    ack_push_tagged(c, d, p, bytes, 0, 0);
}

static void ack_push_arena(roce_ctx_t *c, int d, int p, uint32_t bytes, uint32_t seq, int ns) {
    ack_push_tagged(c, d, p, bytes, (uint8_t)(ROCE_ACK_ARENA | (ns ? ROCE_ACK_NS1 : 0u)), seq);
}

static void ack_pop(roce_ctx_t *c, int d, int p) {
    roce_dev_t *dev = &c->dev[d];
    if (dev->ack_count[p] == 0) return;
    if (!(dev->ack_mark[p][dev->ack_head[p]] & ROCE_ACK_PROVEN)) dev->unacked[p] -= dev->ack_bytes[p][dev->ack_head[p]];
    dev->ack_head[p] = (uint16_t)((dev->ack_head[p] + 1) % ROCE_SEND_DEPTH);
    dev->ack_count[p] -= 1;
}

static int drain_device(roce_ctx_t *c, int d) {
    roce_dev_t *dev = &c->dev[d];
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
        int stream = (int)((wc[i].wr_id >> 24) & 0xFFu);
        uint32_t seq = (uint32_t)(wc[i].wr_id >> 32);
        if (wc[i].status != IBV_WC_SUCCESS) {
            if (kind == ROCE_WR_CHAIN || kind == ROCE_WR_CREDIT) {
                FAIL(c, "RDMA write of chain %s %u of stream %d to rank %d lane %d on %s failed: %s "
                        "(vendor_err 0x%x)", kind == ROCE_WR_CHAIN ? "chunk" : "credit", seq, stream, peer,
                     lane, dev->name, ibv_wc_status_str(wc[i].status), wc[i].vendor_err);
            } else if (kind == ROCE_WR_LINK || kind == ROCE_WR_LINK_CREDIT || kind == ROCE_WR_LINK_DATA) {
                FAIL(c, "RDMA write of link %s %u of link %d to rank %d lane %d on %s failed: %s "
                        "(vendor_err 0x%x)", kind == ROCE_WR_LINK_CREDIT ? "credit" : "item", seq, stream, peer,
                     lane, dev->name, ibv_wc_status_str(wc[i].status), wc[i].vendor_err);
            } else {
                FAIL(c, "RDMA write of sequence %u to rank %d lane %d on %s failed: %s (vendor_err 0x%x); "
                        "newest posted sequence %u", seq, peer, lane, dev->name,
                     ibv_wc_status_str(wc[i].status), wc[i].vendor_err, c->posting_seq);
            }
            return -1;
        }
        if (kind == ROCE_WR_CHECK || peer >= c->world) continue;
        ack_pop(c, d, peer);
        if (kind == ROCE_WR_CHAIN && stream < ROCE_CHAIN_STREAMS && lane < ROCE_MAX_LANES) {
            c->chain[stream].lane_done[lane]++;
        }
        if (kind == ROCE_WR_LINK && stream < ROCE_LINKS && lane < ROCE_MAX_LANES) {
            c->link[stream].lane_done[lane]++;
        }
        /* ROCE_WR_LINK_DATA: a windowed chunk; its lane's flag completion counts the item. */
        atomic_fetch_add_explicit(&dev->writes_completed, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&c->writes_completed, 1, memory_order_relaxed);
    }
    return 0;
}

static int drain_all(roce_ctx_t *c) {
    for (int d = 0; d < c->n_dev; d++) {
        if (drain_device(c, d) != 0) return -1;
    }
    return 0;
}

static int wait_credit(roce_ctx_t *c, int d, int p) {
    roce_dev_t *dev = &c->dev[d];
    while (dev->ack_count[p] >= ROCE_CREDIT - 1) {
        if (drain_device(c, d) != 0) return -1;
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            FAIL(c, "progress thread stopped with %u signaled writes in flight to rank %d on %s",
                 (unsigned)dev->ack_count[p], p, dev->name);
            return -1;
        }
    }
    return 0;
}

/* -- forward windows -----------------------------------------------------------------
 *
 * A lane whose packets cross NIC relays passes every relay's hairpin queue,
 * which holds a fixed number of bytes and cannot pause its sender. Such a lane
 * has a forward window: its stripe is posted as signaled chunks of `fwd_chunk`
 * bytes, and a chunk is posted only while the bytes in flight on the lane's
 * queue pair (posted, not yet completed) stay within the window. The flag
 * follows the last chunk on the same queue pair. pump_streams advances all
 * windowed stripes of one phase round-robin, so they progress together. */

/* Lane windows (world * lane_count entries, 0: none) and the chunk size; set
 * before the progress thread starts. A NULL table removes every window. */
int roce_set_forward(roce_ctx_t *c, const uint32_t *lane_window_bytes, uint32_t chunk_bytes) {
    if (atomic_load(&c->running)) {
        FAIL(c, "forward windows are set before the progress thread starts");
        return -1;
    }
    memset(c->fwd_window, 0, sizeof(c->fwd_window));
    c->fwd_chunk = 0;
    if (lane_window_bytes == NULL) return 0;
    if (chunk_bytes == 0 || chunk_bytes % 16u != 0) {
        FAIL(c, "forward chunk of %u bytes is not a positive multiple of 16", chunk_bytes);
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        for (int l = 0; l < c->lane_count; l++) {
            uint32_t w = lane_window_bytes[p * c->lane_count + l];
            if (w == 0) continue;
            if (p == c->rank) {
                FAIL(c, "rank %d sets a forward window on its own lane %d", c->rank, l);
                memset(c->fwd_window, 0, sizeof(c->fwd_window));
                return -1;
            }
            if (w < chunk_bytes || w / chunk_bytes > ROCE_CREDIT - 4) {
                FAIL(c, "forward window of %u bytes on lane %d toward rank %d must hold 1 to %d chunks of "
                        "%u bytes", w, l, p, ROCE_CREDIT - 4, chunk_bytes);
                memset(c->fwd_window, 0, sizeof(c->fwd_window));
                return -1;
            }
            c->fwd_window[p][l] = w;
        }
    }
    c->fwd_chunk = chunk_bytes;
    return 0;
}

/* The newer of two flag sequences of peer p, or `none` when neither is one of this rank's recent ops
 * (a peer is at most one op ahead of this rank; older words hold earlier ops' flags). */
static uint32_t newest_flag(const roce_ctx_t *c, uint32_t a, uint32_t b, uint32_t none, int *found) {
    uint32_t best = none;
    *found = 0;
    uint32_t words[2] = {a, b};
    for (int i = 0; i < 2; i++) {
        int32_t ahead = (int32_t)(words[i] - c->last_seq);
        if (ahead > 1 || ahead < -(int32_t)ROCE_OP_HISTORY) continue;
        if (!*found || (int32_t)(words[i] - best) > 0) best = words[i];
        *found = 1;
    }
    return best;
}

static int op_is_twoshot(const roce_ctx_t *c, uint32_t seq) {
    return c->op_seq[seq % ROCE_OP_HISTORY] == seq && c->op_code[seq % ROCE_OP_HISTORY] == ROCE_OP_TWOSHOT;
}

static void note_line(uint64_t at[2], uint64_t index) {
    at[1] = at[0];
    at[0] = index;
}

/* Records an op for the flag lines its peers write: every peer writes its namespace-0 flag of a
 * one-shot, two-shot or scatter op to every rank, and its namespace-1 flag of a two-shot op; a
 * described op writes either namespace to its phase partners only. */
static void note_op_lines(roce_ctx_t *c, uint32_t seq, uint8_t code) {
    uint32_t slot = seq & 1u;
    c->op_index++;
    for (int ns = 0; ns < 2; ns++) {
        int every = code == ROCE_OP_TWOSHOT || (ns == 0 && code != ROCE_OP_DESCRIBED);
        if (every) note_line(c->line_every[ns][slot], c->op_index);
        if (every || code == ROCE_OP_DESCRIBED) note_line(c->line_any[ns][slot], c->op_index);
    }
}

static uint64_t newest_before(const uint64_t at[2], uint64_t current) {
    return at[0] == current ? at[1] : at[0];
}

/* Whether flag line (ns, slot) of a peer holds a recent flag. A peer's writes of an op have landed once
 * this rank's kernel finished that op (it waits for every write it is sent), and ops alternate between
 * two send slots, so at most two are in flight: a line that an op within the last 2^31 ops had every
 * peer write holds a sequence at most about 2^31 ops old, never a value 2^32 ops old that only looks
 * recent. A line that no earlier op may have written holds zero until a peer writes the current op's
 * or the next op's flag. */
static int flag_line_trusted(const roce_ctx_t *c, int ns, int slot, uint32_t word) {
    uint64_t every = newest_before(c->line_every[ns][slot], c->op_index);
    if (every != 0) return c->op_index - every < (1ull << 31);
    return word != 0 && newest_before(c->line_any[ns][slot], c->op_index) == 0;
}

/* Bytes of arena writes to peer p that the op order proves delivered leave the window of every queue
 * pair toward p: a flag of op F from p proves every write of ops before F (p's kernel finished them,
 * and it waits for every write it is sent), and a two-shot phase-1 flag of op F proves the phase-0
 * writes of F (p reduced them before its phase 1). Reads p's lane-0 flags of both slots and both
 * namespaces, each only while flag_line_trusted holds; a peer whose flags did not change since the
 * last proof is skipped. */
static void fwd_prove(roce_ctx_t *c, int p) {
    if (!c->fwd_proof) return;
    const uint8_t *flags = c->region + c->flag_off;
    uint64_t ns_lines = (uint64_t)c->world * ROCE_SLOTS * (uint64_t)c->lane_count;
    uint32_t word[2][2];
    for (int ns = 0; ns < 2; ns++) {
        for (int slot = 0; slot < ROCE_SLOTS; slot++) {
            uint64_t line = ns * ns_lines + ((uint64_t)p * ROCE_SLOTS + (uint64_t)slot) * (uint64_t)c->lane_count;
            word[ns][slot] = __atomic_load_n((const uint32_t *)(flags + line * ROCE_FLAG_STRIDE), __ATOMIC_ACQUIRE);
            /* A line that may hold an old or never-written value proves nothing: newest_flag rejects a
             * sequence two ahead of this rank's. */
            if (!flag_line_trusted(c, ns, slot, word[ns][slot])) word[ns][slot] = c->last_seq + 2u;
        }
    }
    int found0, found1;
    uint32_t newest0 = newest_flag(c, word[0][0], word[0][1], 0, &found0);
    uint32_t newest1 = newest_flag(c, word[1][0], word[1][1], 0, &found1);
    if (!found0 && !found1) return;
    /* Ops before F are proven by namespace 0 alone: a peer writes its namespace-0 flag of an op before
     * its namespace-1 flag of that op on the same queue pair. */
    uint32_t key0 = found0 ? newest0 : c->last_seq + 2u, key1 = found1 ? newest1 : c->last_seq + 2u;
    if (c->proof_flags[p][0] == key0 && c->proof_flags[p][1] == key1) return;
    c->proof_flags[p][0] = key0;
    c->proof_flags[p][1] = key1;
    int phase1 = found1 && op_is_twoshot(c, newest1);
    for (int lane = 0; lane < c->lane_count; lane++) {
        int d = c->lane_device[p][lane];
        if (d < 0 || (lane > 0 && d == c->lane_device[p][0])) continue;
        roce_dev_t *dev = &c->dev[d];
        for (uint32_t i = 0; i < dev->ack_count[p]; i++) {
            uint32_t index = (dev->ack_head[p] + i) % ROCE_SEND_DEPTH;
            uint8_t mark = dev->ack_mark[p][index];
            if (!(mark & ROCE_ACK_ARENA) || (mark & ROCE_ACK_PROVEN)) continue;
            uint32_t seq = dev->ack_seq[p][index];
            int proven = (found0 && (int32_t)(newest0 - seq) > 0) ||
                         (phase1 && seq == newest1 && !(mark & ROCE_ACK_NS1));
            if (!proven) continue;
            dev->ack_mark[p][index] = (uint8_t)(mark | ROCE_ACK_PROVEN);
            dev->unacked[p] -= dev->ack_bytes[p][index];
            atomic_fetch_add_explicit(&c->fwd_proven_bytes, dev->ack_bytes[p][index], memory_order_relaxed);
        }
    }
}

static int post_stream_write(roce_ctx_t *c, roce_stream_t *s, uint32_t bytes, int flag) {
    roce_dev_t *dev = &c->dev[s->dev];
    uint32_t value = s->seq;
    struct ibv_sge sge;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = make_wr_id(s->peer, s->lane, ROCE_WR_OP, s->seq);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr.rdma.rkey = s->rkey;
    if (flag) {
        sge.addr = (uint64_t)(uintptr_t)&value;
        sge.length = 4;
        sge.lkey = 0;
        wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        wr.wr.rdma.remote_addr = s->remote_flag;
    } else {
        sge.addr = (uint64_t)(uintptr_t)(s->source + s->done);
        sge.length = bytes;
        sge.lkey = dev->mr->lkey;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = s->remote_data + s->done;
    }
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(dev->qp[s->peer], &wr, &bad);
    if (rc != 0) {
        FAIL(c, "ibv_post_send of lane %d to rank %d on %s at sequence %u: %s", s->lane, s->peer,
             dev->name, s->seq, strerror(rc));
        return -1;
    }
    ack_push_arena(c, s->dev, s->peer, flag ? 4u : bytes, s->seq, s->ns);
    if (flag) {
        s->flag_posted = 1;
    } else {
        s->done += bytes;
        atomic_fetch_add_explicit(&dev->bytes_posted, bytes, memory_order_relaxed);
        atomic_fetch_add_explicit(&c->fwd_chunks_posted, 1, memory_order_relaxed);
    }
    uint64_t now = dev->unacked[s->peer];
    if (now > atomic_load_explicit(&c->fwd_max_unacked, memory_order_relaxed)) {
        atomic_store_explicit(&c->fwd_max_unacked, now, memory_order_relaxed);
    }
    return 0;
}

static uint64_t steady_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* One wait of the phase's windowed stripes for room in their windows. */
static void note_fwd_wait(roce_ctx_t *c, uint64_t started) {
    uint64_t waited = steady_ns() - started;
    atomic_fetch_add_explicit(&c->fwd_waits, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&c->fwd_wait_ns, waited, memory_order_relaxed);
    if (waited > atomic_load_explicit(&c->fwd_wait_max_ns, memory_order_relaxed)) {
        atomic_store_explicit(&c->fwd_wait_max_ns, waited, memory_order_relaxed);
    }
}

static int pump_streams(roce_ctx_t *c) {
    int active = c->n_streams;
    uint64_t waiting = 0;   /* when the stripes last stopped for their windows; 0: not waiting */
    while (active > 0) {
        int progressed = 0;
        for (int i = 0; i < c->n_streams; i++) {
            roce_stream_t *s = &c->streams[i];
            if (s->flag_posted) continue;
            roce_dev_t *dev = &c->dev[s->dev];
            uint32_t window = c->fwd_window[s->peer][s->lane];
            while (s->done < s->length && dev->ack_count[s->peer] < ROCE_CREDIT - 1) {
                uint32_t bytes = s->length - s->done;
                if (bytes > c->fwd_chunk) bytes = c->fwd_chunk;
                if (dev->unacked[s->peer] + bytes > window) fwd_prove(c, s->peer);
                if (dev->unacked[s->peer] + bytes > window) {
                    /* The part of the chunk the window holds now goes at once (whole packs, at least
                     * ROCE_FWD_MIN_PIECE bytes); the rest waits for room. */
                    uint64_t room = dev->unacked[s->peer] < window ? (window - dev->unacked[s->peer]) / 16u * 16u : 0;
                    if (room < ROCE_FWD_MIN_PIECE) break;
                    bytes = (uint32_t)room;
                }
                if (post_stream_write(c, s, bytes, 0) != 0) return -1;
                progressed = 1;
            }
            if (s->done == s->length && dev->ack_count[s->peer] < ROCE_CREDIT - 1) {
                if (post_stream_write(c, s, 0, 1) != 0) return -1;
                active--;
                progressed = 1;
            }
        }
        if (progressed) {
            if (waiting != 0) note_fwd_wait(c, waiting);
            waiting = 0;
            continue;
        }
        if (waiting == 0) waiting = steady_ns();
        int drained = 0;
        for (int i = 0; i < c->n_streams; i++) {
            if (c->streams[i].flag_posted || (drained & (1 << c->streams[i].dev))) continue;
            drained |= 1 << c->streams[i].dev;
            if (drain_device(c, c->streams[i].dev) != 0) return -1;
        }
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            FAIL(c, "progress thread stopped with %d windowed stripes of sequence %u pending", active,
                 c->posting_seq);
            return -1;
        }
    }
    if (waiting != 0) note_fwd_wait(c, waiting);
    c->n_streams = 0;
    return 0;
}

/* -- chain schedule ----------------------------------------------------------------- */

/* Offsets within the chain area of a session with `lanes` lanes, `slots` slots per
 * stream and `slot_bytes` per slot: {recv, send, rflag, ready, consumed, sent, credit,
 * control, total}. */
int roce_chain_layout(int lanes, int slots, uint64_t slot_bytes, uint64_t *out) {
    uint64_t ring;
    if (out == NULL || lanes < 1 || lanes > ROCE_MAX_LANES || slots < 2 || slots > ROCE_CHAIN_MAX_SLOTS ||
        slot_bytes == 0 || slot_bytes % 4096u != 0 || slot_bytes > ((uint64_t)1 << 31)) {
        return -1;
    }
    ring = (uint64_t)ROCE_CHAIN_STREAMS * (uint64_t)slots * slot_bytes;
    out[CH_RECV] = 0;
    out[CH_SEND] = ring;
    out[CH_RFLAG] = 2 * ring;
    out[CH_READY] = out[CH_RFLAG] + (uint64_t)ROCE_CHAIN_STREAMS * slots * lanes * ROCE_FLAG_STRIDE;
    out[CH_CONSUMED] = out[CH_READY] + ROCE_CHAIN_STREAMS * ROCE_FLAG_STRIDE;
    out[CH_SENT] = out[CH_CONSUMED] + ROCE_CHAIN_STREAMS * ROCE_FLAG_STRIDE;
    out[CH_CREDIT] = out[CH_SENT] + ROCE_CHAIN_STREAMS * ROCE_FLAG_STRIDE;
    out[CH_CTRL] = out[CH_CREDIT] + ROCE_CHAIN_STREAMS * ROCE_FLAG_STRIDE;
    out[CH_TOTAL] = out[CH_CTRL] + ROCE_FLAG_STRIDE;
    return 0;
}

/* Configure the chain schedule before the progress thread starts: `prev` and `next`
 * are this rank's neighbors in chain order (-1 at an end), the chain area starts
 * `chain_off` bytes into every rank's arena. `slots` 0 removes the schedule. */
int roce_set_chain(roce_ctx_t *c, int prev, int next, int slots, uint64_t slot_bytes, uint64_t chain_off) {
    if (atomic_load(&c->running)) {
        FAIL(c, "the chain schedule is set before the progress thread starts");
        return -1;
    }
    memset(c->chain, 0, sizeof(c->chain));
    memset(c->chain_ops, 0, sizeof(c->chain_ops));
    c->chain_slots = 0;
    c->chain_known[0] = c->chain_known[1] = 0;
    if (slots == 0) return 0;
    if ((prev < -1 || prev >= c->world || prev == c->rank) || (next < -1 || next >= c->world || next == c->rank) ||
        (prev < 0 && next < 0) || (prev >= 0 && prev == next)) {
        FAIL(c, "chain neighbors %d and %d of rank %d in a session of %d", prev, next, c->rank, c->world);
        return -1;
    }
    if (roce_chain_layout(c->lane_count, slots, slot_bytes, c->chain_layout) != 0) {
        FAIL(c, "chain geometry: %d slots of %llu bytes (2 to %d slots, a multiple of 4096 bytes up to 2^31)",
             slots, (unsigned long long)slot_bytes, ROCE_CHAIN_MAX_SLOTS);
        return -1;
    }
    if (chain_off % 4096u != 0 || chain_off < c->ctrl_off + ROCE_FLAG_STRIDE ||
        chain_off + c->chain_layout[CH_TOTAL] > c->region_bytes) {
        FAIL(c, "chain area of %llu bytes at offset %llu does not fit after the control line in an arena of "
                "%llu bytes", (unsigned long long)c->chain_layout[CH_TOTAL], (unsigned long long)chain_off,
             (unsigned long long)c->region_bytes);
        return -1;
    }
    c->chain_slots = slots;
    c->chain_slot_bytes = slot_bytes;
    c->chain_off = chain_off;
    roce_chain_stream_t *st = c->chain;
    st[0] = (roce_chain_stream_t){.in_peer = prev, .out_peer = next, .half = 0,
                                  .out_kind = next >= 0 ? ROCE_CHAIN_KERNEL : ROCE_CHAIN_NONE};
    st[1] = (roce_chain_stream_t){.in_peer = next, .out_peer = prev, .half = 0,
                                  .out_kind = prev < 0 ? ROCE_CHAIN_NONE
                                              : next < 0 ? ROCE_CHAIN_KERNEL : ROCE_CHAIN_FORWARD};
    st[2] = (roce_chain_stream_t){.in_peer = next, .out_peer = prev, .half = 1,
                                  .out_kind = prev >= 0 ? ROCE_CHAIN_KERNEL : ROCE_CHAIN_NONE};
    st[3] = (roce_chain_stream_t){.in_peer = prev, .out_peer = next, .half = 1,
                                  .out_kind = next < 0 ? ROCE_CHAIN_NONE
                                              : prev < 0 ? ROCE_CHAIN_KERNEL : ROCE_CHAIN_FORWARD};
    c->chain_last_seq = __atomic_load_n((volatile uint32_t *)(c->region + chain_off + c->chain_layout[CH_CTRL]),
                                        __ATOMIC_ACQUIRE);
    return 0;
}

static volatile uint32_t *chain_word(const roce_ctx_t *c, int area, uint64_t offset) {
    return (volatile uint32_t *)(c->region + c->chain_off + c->chain_layout[area] + offset);
}

static uint64_t chain_slot_off(const roce_ctx_t *c, int area, int s, uint32_t m) {
    return c->chain_off + c->chain_layout[area] + ((uint64_t)s * (uint64_t)c->chain_slots + m) * c->chain_slot_bytes;
}

static uint64_t chain_flag_off(const roce_ctx_t *c, int s, uint32_t m, int lane) {
    return c->chain_off + c->chain_layout[CH_RFLAG] +
           (((uint64_t)s * (uint64_t)c->chain_slots + m) * (uint64_t)c->lane_count + (uint64_t)lane) *
               ROCE_FLAG_STRIDE;
}

/* Bytes of chunk g of half h, or 0 when no known op holds it. */
static uint32_t chain_chunk_bytes(const roce_ctx_t *c, int h, uint32_t g) {
    for (int i = 0; i < ROCE_CHAIN_OPS; i++) {
        const roce_chain_op_t *op = &c->chain_ops[i];
        if (op->chunk == 0 || op->count[h] == 0) continue;
        uint32_t index = g - op->first[h];
        if (index < op->count[h]) {
            uint32_t left = op->bytes[h] - index * op->chunk;
            return left < op->chunk ? left : op->chunk;
        }
    }
    return 0;
}

/* Append the chain ops whose doorbell rang. */
static int chain_take_ops(roce_ctx_t *c, int *work) {
    volatile uint32_t *ctrl = chain_word(c, CH_CTRL, 0);
    uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
    if (seq != c->chain_last_seq && seq - c->chain_last_seq > 2u) {
        FAIL(c, "chain doorbell skipped %u ops (newest taken %u, doorbell %u)", seq - c->chain_last_seq,
             c->chain_last_seq, seq);
        return -1;
    }
    while (seq != c->chain_last_seq) {
        uint32_t next = c->chain_last_seq + 1;
        volatile uint32_t *params = &ctrl[1 + 4 * (next & 1u)];
        uint32_t bytes_a = params[0], bytes_b = params[1], chunk = params[2];
        if (chunk == 0 || chunk % 16u != 0 || (uint64_t)chunk > c->chain_slot_bytes || bytes_a % 16u != 0 ||
            bytes_b % 16u != 0 || (uint64_t)bytes_a + bytes_b == 0) {
            FAIL(c, "chain op %u: halves of %u and %u bytes in chunks of %u bytes (multiples of 16, chunks of at "
                    "most %llu bytes)", next, bytes_a, bytes_b, chunk, (unsigned long long)c->chain_slot_bytes);
            return -1;
        }
        roce_chain_op_t *entry = &c->chain_ops[next % ROCE_CHAIN_OPS];
        if (entry->chunk != 0) {
            for (int s = 0; s < ROCE_CHAIN_STREAMS; s++) {
                const roce_chain_stream_t *st = &c->chain[s];
                int h = st->half;
                if (st->out_kind != ROCE_CHAIN_NONE &&
                    (int32_t)(st->next_post - (entry->first[h] + entry->count[h])) < 0) {
                    FAIL(c, "chain op %u would replace op %u, whose stream %d still has chunks to write", next,
                         entry->seq, s);
                    return -1;
                }
            }
        }
        entry->seq = next;
        entry->bytes[0] = bytes_a;
        entry->bytes[1] = bytes_b;
        entry->chunk = chunk;
        for (int h = 0; h < 2; h++) {
            entry->first[h] = c->chain_known[h];
            entry->count[h] = (entry->bytes[h] + chunk - 1) / chunk;
            c->chain_known[h] += entry->count[h];
        }
        c->chain_last_seq = next;
        atomic_fetch_add_explicit(&c->chain_ops_seen, 1, memory_order_relaxed);
        trace_event(c, ROCE_EV_OP, 0, next);
        *work = 1;
    }
    return 0;
}

/* Write chunk g of stream s downstream: its stripe and flag on every lane. */
static int chain_post_chunk(roce_ctx_t *c, int s, uint32_t g, uint32_t m, uint32_t bytes) {
    roce_chain_stream_t *st = &c->chain[s];
    int p = st->out_peer;
    uint64_t src = chain_slot_off(c, st->out_kind == ROCE_CHAIN_KERNEL ? CH_SEND : CH_RECV, s, m);
    uint64_t dst = chain_slot_off(c, CH_RECV, s, m);
    uint32_t packs = bytes / 16u, tag = g + 1u;
    for (int lane = 0; lane < c->lane_count; lane++) {
        int d = c->lane_device[p][lane];
        roce_dev_t *dev = &c->dev[d];
        uint32_t first, count;
        lane_split(packs, c->lane_count, lane, &first, &count);
        struct ibv_sge flag_sge = {.addr = (uint64_t)(uintptr_t)&tag, .length = 4, .lkey = 0};
        struct ibv_send_wr flag_wr;
        memset(&flag_wr, 0, sizeof(flag_wr));
        flag_wr.wr_id = make_chain_wr_id(p, lane, ROCE_WR_CHAIN, s, g);
        flag_wr.sg_list = &flag_sge;
        flag_wr.num_sge = 1;
        flag_wr.opcode = IBV_WR_RDMA_WRITE;
        flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        flag_wr.wr.rdma.remote_addr = c->peer_addr[p] + chain_flag_off(c, s, m, lane);
        flag_wr.wr.rdma.rkey = c->peer_rkey[lane][p];
        struct ibv_send_wr data_wr;
        struct ibv_sge data_sge;
        struct ibv_send_wr *head = &flag_wr;
        if (count != 0) {
            data_sge.addr = (uint64_t)(uintptr_t)(c->region + src + (uint64_t)first * 16u);
            data_sge.length = count * 16u;
            data_sge.lkey = dev->mr->lkey;
            memset(&data_wr, 0, sizeof(data_wr));
            data_wr.wr_id = flag_wr.wr_id;
            data_wr.next = &flag_wr;
            data_wr.sg_list = &data_sge;
            data_wr.num_sge = 1;
            data_wr.opcode = IBV_WR_RDMA_WRITE;
            data_wr.wr.rdma.remote_addr = c->peer_addr[p] + dst + (uint64_t)first * 16u;
            data_wr.wr.rdma.rkey = c->peer_rkey[lane][p];
            head = &data_wr;
        }
        struct ibv_send_wr *bad = NULL;
        int rc = ibv_post_send(dev->qp[p], head, &bad);
        if (rc != 0) {
            FAIL(c, "ibv_post_send of chain chunk %u of stream %d, lane %d to rank %d on %s: %s", g, s, lane, p,
                 dev->name, strerror(rc));
            return -1;
        }
        ack_push(c, d, p, count * 16u + 4u);
        atomic_fetch_add_explicit(&dev->bytes_posted, (uint64_t)count * 16u, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&c->chain_chunks_posted, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&c->chain_bytes_posted, bytes, memory_order_relaxed);
    return 0;
}

static int chain_room(const roce_ctx_t *c, int p, int lanes) {
    for (int lane = 0; lane < lanes; lane++) {
        if (c->dev[c->lane_device[p][lane]].ack_count[p] >= ROCE_CREDIT - 1) return 0;
    }
    return 1;
}

/* Tell the upstream neighbor of stream s that `value` chunks of its slots are free. */
static int chain_post_credit(roce_ctx_t *c, int s, uint32_t value) {
    int p = c->chain[s].in_peer;
    int d = c->lane_device[p][0];
    roce_dev_t *dev = &c->dev[d];
    struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = make_chain_wr_id(p, 0, ROCE_WR_CREDIT, s, value);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    wr.wr.rdma.remote_addr = c->peer_addr[p] + c->chain_off + c->chain_layout[CH_CREDIT] +
                             (uint64_t)s * ROCE_FLAG_STRIDE;
    wr.wr.rdma.rkey = c->peer_rkey[0][p];
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(dev->qp[p], &wr, &bad);
    if (rc != 0) {
        FAIL(c, "ibv_post_send of chain credit %u of stream %d to rank %d on %s: %s", value, s, p, dev->name,
             strerror(rc));
        return -1;
    }
    ack_push(c, d, p, 4u);
    atomic_fetch_add_explicit(&c->chain_credits_sent, 1, memory_order_relaxed);
    return 0;
}

/* Advance every chain stream as far as it can go without waiting; 1 when anything
 * moved, 0 when nothing did, -1 on a failure. */
static int chain_service(roce_ctx_t *c) {
    int work = 0;
    if (chain_take_ops(c, &work) != 0) return -1;
    uint32_t slots = (uint32_t)c->chain_slots;
    int pending = 0;
    for (int s = 0; s < ROCE_CHAIN_STREAMS; s++) {
        roce_chain_stream_t *st = &c->chain[s];
        if (st->out_kind == ROCE_CHAIN_NONE) continue;
        uint32_t credit = __atomic_load_n(chain_word(c, CH_CREDIT, (uint64_t)s * ROCE_FLAG_STRIDE), __ATOMIC_ACQUIRE);
        trace_credit_in(c, s, credit, &st->credit_seen);
        while (st->next_post != c->chain_known[st->half]) {
            uint32_t g = st->next_post, m = g % slots, tag = g + 1u;
            int ready = 1;
            if (st->out_kind == ROCE_CHAIN_KERNEL) {
                ready = __atomic_load_n(chain_word(c, CH_READY, (uint64_t)s * ROCE_FLAG_STRIDE + 4u * m),
                                        __ATOMIC_ACQUIRE) == tag;
            } else {
                for (int lane = 0; lane < c->lane_count && ready; lane++) {
                    volatile uint32_t *flag = (volatile uint32_t *)(c->region + chain_flag_off(c, s, m, lane));
                    ready = __atomic_load_n(flag, __ATOMIC_ACQUIRE) == tag;
                }
            }
            if (!ready) break;
            trace_ready(c, s, tag, &st->ready_seen);
            if ((int32_t)(credit - (tag - slots)) < 0 || !chain_room(c, st->out_peer, c->lane_count)) break;
            uint32_t bytes = chain_chunk_bytes(c, st->half, g);
            if (bytes == 0 && (uint64_t)g < (uint64_t)c->chain_known[st->half]) {
                /* An empty chunk only ends an empty half, which has no chunks. */
                FAIL(c, "chain chunk %u of stream %d has no op", g, s);
                return -1;
            }
            if (chain_post_chunk(c, s, g, m, bytes) != 0) return -1;
            trace_event(c, ROCE_EV_POSTED, s, tag);
            st->next_post++;
            work = 1;
        }
        if (st->done != st->next_post) pending = 1;
    }
    for (int s = 0; s < ROCE_CHAIN_STREAMS; s++) {
        if (c->chain[s].in_peer >= 0 && c->chain[s].credited != c->chain[s].consumed) pending = 1;
    }
    if (pending && drain_all(c) != 0) return -1;
    for (int s = 0; s < ROCE_CHAIN_STREAMS; s++) {
        roce_chain_stream_t *st = &c->chain[s];
        if (st->out_kind != ROCE_CHAIN_NONE) {
            uint32_t done = st->lane_done[0];
            for (int lane = 1; lane < c->lane_count; lane++) {
                if ((int32_t)(st->lane_done[lane] - done) < 0) done = st->lane_done[lane];
            }
            if (done != st->done) {
                for (uint32_t t = st->done + 1u; c->trace_cap && t != done + 1u; t++) {
                    trace_event(c, ROCE_EV_DONE, s, t);
                }
                st->done = done;
                if (st->out_kind == ROCE_CHAIN_KERNEL) {
                    __atomic_store_n(chain_word(c, CH_SENT, (uint64_t)s * ROCE_FLAG_STRIDE), done, __ATOMIC_RELEASE);
                }
                work = 1;
            }
        }
        if (st->in_peer < 0) continue;
        while (__atomic_load_n(chain_word(c, CH_CONSUMED, (uint64_t)s * ROCE_FLAG_STRIDE + 4u * (st->consumed % slots)),
                               __ATOMIC_ACQUIRE) == st->consumed + 1u) {
            st->consumed++;
            trace_event(c, ROCE_EV_CONSUMED, s, st->consumed);
        }
        uint32_t released = st->consumed;
        if (st->out_kind == ROCE_CHAIN_FORWARD && (int32_t)(st->done - released) < 0) released = st->done;
        if (released != st->credited && chain_room(c, st->in_peer, 1)) {
            if (chain_post_credit(c, s, released) != 0) return -1;
            trace_event(c, ROCE_EV_CREDIT_OUT, s, released);
            st->credited = released;
            work = 1;
        }
    }
    return work;
}


/* -- chain links: the chain all-gather and reduce-scatter -------------------------- */

int roce_link_layout(int lanes, int slots, uint64_t slot_bytes, uint64_t *out) {
    uint64_t ring;
    if (out == NULL || lanes < 1 || lanes > ROCE_MAX_LANES || slots < 2 || slots > ROCE_LINK_MAX_SLOTS ||
        slot_bytes == 0 || slot_bytes % 4096u != 0 || slot_bytes > ((uint64_t)1 << 31)) {
        return -1;
    }
    ring = (uint64_t)ROCE_LINKS * (uint64_t)slots * slot_bytes;
    out[LK_RECV] = 0;
    out[LK_OWN] = ring;
    out[LK_RFLAG] = 2 * ring;
    out[LK_READY] = out[LK_RFLAG] + (uint64_t)ROCE_LINKS * slots * lanes * ROCE_FLAG_STRIDE;
    out[LK_CONSUMED] = out[LK_READY] + ROCE_LINKS * ROCE_FLAG_STRIDE;
    out[LK_SENT] = out[LK_CONSUMED] + ROCE_LINKS * ROCE_FLAG_STRIDE;
    out[LK_CREDIT] = out[LK_SENT] + ROCE_LINKS * ROCE_FLAG_STRIDE;
    out[LK_CTRL] = out[LK_CREDIT] + ROCE_LINKS * ROCE_FLAG_STRIDE;
    out[LK_TOTAL] = out[LK_CTRL] + ROCE_FLAG_STRIDE;
    return 0;
}

/* Configure the chain links before the progress thread starts: `prev` and `next` are
 * this rank's neighbors in chain order (-1 at an end), `index` its chain index (the
 * chain holds every rank of the session), `ring_prev` and `ring_next` its neighbors on
 * the ring that closes the chain (-1: no ring links), `ring_window` the bytes each lane
 * toward `ring_next` keeps unacknowledged when those lanes run through relays (0: a
 * direct cable), and the link area starts `link_off` bytes into every rank's arena.
 * `slots` 0 removes the links. */
int roce_set_links(roce_ctx_t *c, int prev, int next, int index, int ring_prev, int ring_next,
                   uint32_t ring_window, int slots, uint64_t slot_bytes, uint64_t link_off) {
    if (atomic_load(&c->running)) {
        FAIL(c, "the chain links are set before the progress thread starts");
        return -1;
    }
    memset(c->link, 0, sizeof(c->link));
    memset(c->link_ops, 0, sizeof(c->link_ops));
    c->link_slots = 0;
    if (slots == 0) return 0;
    if ((prev < -1 || prev >= c->world || prev == c->rank) || (next < -1 || next >= c->world || next == c->rank) ||
        (prev < 0 && next < 0) || (prev >= 0 && prev == next) || index < 0 || index >= c->world ||
        (index == 0) != (prev < 0) || (index == c->world - 1) != (next < 0)) {
        FAIL(c, "link neighbors %d and %d of rank %d at chain index %d in a session of %d", prev, next, c->rank,
             index, c->world);
        return -1;
    }
    if ((ring_prev < 0) != (ring_next < 0) || ring_prev >= c->world || ring_next >= c->world ||
        ring_prev == c->rank || ring_next == c->rank ||
        (ring_next >= 0 && ((next >= 0 && ring_next != next) || (prev >= 0 && ring_prev != prev)))) {
        FAIL(c, "ring neighbors %d and %d of rank %d (chain neighbors %d and %d)", ring_prev, ring_next, c->rank,
             prev, next);
        return -1;
    }
    if (ring_window != 0 && (ring_next < 0 || ring_window < ROCE_LINK_WINDOW_CHUNK ||
                             ring_window / ROCE_LINK_WINDOW_CHUNK > ROCE_CREDIT - 4)) {
        FAIL(c, "ring window of %u bytes must hold 1 to %d chunks of %u bytes on a ring link", ring_window,
             ROCE_CREDIT - 4, ROCE_LINK_WINDOW_CHUNK);
        return -1;
    }
    if (roce_link_layout(c->lane_count, slots, slot_bytes, c->link_layout) != 0) {
        FAIL(c, "link geometry: %d slots of %llu bytes (2 to %d slots, a multiple of 4096 bytes up to 2^31)",
             slots, (unsigned long long)slot_bytes, ROCE_LINK_MAX_SLOTS);
        return -1;
    }
    if (link_off % 4096u != 0 || link_off < c->ctrl_off + ROCE_FLAG_STRIDE ||
        link_off + c->link_layout[LK_TOTAL] > c->region_bytes ||
        (c->chain_slots && link_off < c->chain_off + c->chain_layout[CH_TOTAL] &&
         c->chain_off < link_off + c->link_layout[LK_TOTAL])) {
        FAIL(c, "link area of %llu bytes at offset %llu does not fit after the control line and the chain "
                "area in an arena of %llu bytes", (unsigned long long)c->link_layout[LK_TOTAL],
             (unsigned long long)link_off, (unsigned long long)c->region_bytes);
        return -1;
    }
    c->link_slots = slots;
    c->link_slot_bytes = slot_bytes;
    c->link_off = link_off;
    c->link_index = index;
    c->link[0].out_peer = next;
    c->link[0].in_peer = prev;
    c->link[1].out_peer = prev;
    c->link[1].in_peer = next;
    for (int l = 2; l < ROCE_LINKS; l++) {
        c->link[l].out_peer = ring_next;
        c->link[l].in_peer = ring_prev;
    }
    c->link_ring_window = ring_window;
    c->link_last_seq = __atomic_load_n((volatile uint32_t *)(c->region + link_off + c->link_layout[LK_CTRL]),
                                       __ATOMIC_ACQUIRE);
    return 0;
}

static volatile uint32_t *link_word(const roce_ctx_t *c, int area, uint64_t offset) {
    return (volatile uint32_t *)(c->region + c->link_off + c->link_layout[area] + offset);
}

static uint64_t link_slot_off(const roce_ctx_t *c, int area, int l, uint32_t m) {
    return c->link_off + c->link_layout[area] + ((uint64_t)l * (uint64_t)c->link_slots + m) * c->link_slot_bytes;
}

static uint64_t link_flag_off(const roce_ctx_t *c, int l, uint32_t m, int lane) {
    return c->link_off + c->link_layout[LK_RFLAG] +
           (((uint64_t)l * (uint64_t)c->link_slots + m) * (uint64_t)c->lane_count + (uint64_t)lane) *
               ROCE_FLAG_STRIDE;
}

/* Items per round of link `l` at chain index `index` (protocol.link_rounds). */
static void link_rounds(int op, int world, int index, int l, uint32_t *out, uint32_t *in, uint32_t *own) {
    int j = l == 0 ? index : world - 1 - index;
    int sends = j < world - 1;
    if (op >= ROCE_RING_GATHER) {
        /* Ring ops: link 2 carries the partial sums of the ring reduce-scatter (every item
         * staged by the kernel), link 3 the ring all-gather (own piece, then W - 2 forwards). */
        int scatter = op == ROCE_RING_SCATTER || op == ROCE_RING_REDUCE;
        int gather = op == ROCE_RING_GATHER || op == ROCE_RING_REDUCE;
        int used = (l == 2 && scatter) || (l == 3 && gather);
        *out = used ? (uint32_t)(world - 1) : 0u;
        *in = *out;
        *own = used ? (l == 2 ? (uint32_t)(world - 1) : 1u) : 0u;
        return;
    }
    if (l >= 2) {
        *out = *in = *own = 0u;
        return;
    }
    if (op == ROCE_LINK_GATHER) {
        *out = sends ? (uint32_t)(j + 1) : 0u;
        *in = (uint32_t)j;
        *own = sends ? 1u : 0u;
    } else {
        *out = sends ? (uint32_t)(world - 1 - j) : 0u;
        *in = j > 0 ? (uint32_t)(world - j) : 0u;
        *own = *out;
    }
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Whether a taken link op still has outbound items not done or inbound items not released. */
static int link_op_busy(const roce_ctx_t *c, const roce_link_op_t *op) {
    for (int l = 0; l < ROCE_LINKS; l++) {
        const roce_link_t *lk = &c->link[l];
        uint32_t out_end = op->first_out[l] + op->rounds[l] * op->out_round[l];
        uint32_t in_end = op->first_in[l] + op->rounds[l] * op->in_round[l];
        if (lk->out_peer >= 0 && (int32_t)(lk->done - out_end) < 0) return 1;
        if (lk->in_peer >= 0 && (int32_t)(lk->released - in_end) < 0) return 1;
    }
    return 0;
}

/* Append the link ops whose doorbell rang. */
static int link_take_ops(roce_ctx_t *c, int *work) {
    volatile uint32_t *ctrl = link_word(c, LK_CTRL, 0);
    uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
    if (seq != c->link_last_seq && seq - c->link_last_seq > 2u) {
        FAIL(c, "link doorbell skipped %u ops (newest taken %u, doorbell %u)", seq - c->link_last_seq,
             c->link_last_seq, seq);
        return -1;
    }
    while (seq != c->link_last_seq) {
        uint32_t next = c->link_last_seq + 1;
#ifdef SIRCL_PROXY_TEST_HOOKS
        if (c->test_link_delay_ns != 0) {
            uint64_t now = monotonic_ns();
            if (c->test_link_seen_ns == 0) c->test_link_seen_ns = now;
            if (now - c->test_link_seen_ns < c->test_link_delay_ns) break;
            c->test_link_seen_ns = 0;
        }
#endif
        volatile uint32_t *params = &ctrl[1 + 4 * (next & 1u)];
        uint32_t word = params[0], bytes = params[1], piece = params[2];
        uint32_t op = word & 0xFFu, stagger = (word >> ROCE_STAGGER_SHIFT) & 0xFFu;
        uint32_t stagger3 = (word >> ROCE_GATHER_STAGGER_SHIFT) & 0xFFu;
        if ((word >> 25) != 0u) {
            FAIL(c, "link op %u: op word 0x%08x sets bits 25-31, which hold nothing", next, word);
            return -1;
        }
        if (op < ROCE_LINK_GATHER || op > ROCE_RING_REDUCE || bytes % 16u != 0 || piece == 0 ||
            piece % 16u != 0 || (uint64_t)piece > c->link_slot_bytes) {
            FAIL(c, "link op %u: op %u of %u bytes per rank in pieces of %u bytes (multiples of 16, pieces of "
                    "at most %llu bytes)", next, op, bytes, piece, (unsigned long long)c->link_slot_bytes);
            return -1;
        }
        int staggered = op == ROCE_RING_SCATTER || op == ROCE_RING_REDUCE;
        int gathered = op == ROCE_RING_GATHER || op == ROCE_RING_REDUCE;
        if ((stagger != 0 && !staggered) || stagger > ROCE_MAX_STAGGER ||
            (stagger != 0 && (uint32_t)c->link_slots < stagger * (uint32_t)(c->world - 1) + 2u)) {
            FAIL(c, "link op %u: op %u with stagger %u (ring reduce-scatter and all-reduce only, at most %u, "
                    "and %d link slots hold at most stagger %d)", next, op, stagger, ROCE_MAX_STAGGER,
                 c->link_slots, c->world > 1 ? (c->link_slots - 2) / (c->world - 1) : 0);
            return -1;
        }
        if ((stagger3 != 0 && !gathered) || stagger3 > ROCE_MAX_STAGGER ||
            (stagger3 != 0 && (uint32_t)c->link_slots < stagger3 * (uint32_t)(c->world - 1) + 2u)) {
            FAIL(c, "link op %u: op %u with all-gather stagger %u (ring all-gather and all-reduce only, at most "
                    "%u, and %d link slots hold at most stagger %d)", next, op, stagger3, ROCE_MAX_STAGGER,
                 c->link_slots, c->world > 1 ? (c->link_slots - 2) / (c->world - 1) : 0);
            return -1;
        }
        roce_link_op_t *entry = &c->link_ops[next % ROCE_LINK_OPS];
        /* The entry is reused once the op it holds has every outbound item done and every
         * inbound item released; until then the newer op waits in the doorbell. */
        if (entry->op != 0 && link_op_busy(c, entry)) break;
        entry->seq = next;
        entry->op = op;
        entry->bytes = bytes;
        entry->piece = piece;
        entry->pieces = (bytes + piece - 1) / piece;
        entry->stagger = stagger;
        entry->stagger3 = stagger3;
        entry->own_flags = (word >> 24) & 1u;
        for (int l = 0; l < ROCE_LINKS; l++) {
            roce_link_t *lk = &c->link[l];
            link_rounds((int)op, c->world, c->link_index, l, &entry->out_round[l], &entry->in_round[l],
                        &entry->own_round[l]);
            entry->rounds[l] = entry->pieces + (l == 2 && staggered ? (uint32_t)(c->world - 2) * stagger : 0u) +
                               (l == 3 && gathered ? (uint32_t)(c->world - 2) * stagger3 : 0u);
            entry->first_out[l] = lk->known_out;
            entry->first_in[l] = lk->known_in;
            entry->first_own[l] = lk->known_own;
            lk->known_out += entry->rounds[l] * entry->out_round[l];
            lk->known_in += entry->rounds[l] * entry->in_round[l];
            lk->known_own += entry->rounds[l] * entry->own_round[l];
        }
        c->link_last_seq = next;
        atomic_fetch_add_explicit(&c->link_ops_seen, 1, memory_order_relaxed);
        trace_event(c, ROCE_EV_OP, ROCE_TRACE_LINK_STREAM, next);
        *work = 1;
    }
    return 0;
}

/* The op and op-local index of outbound item g of link l, or NULL. */
static const roce_link_op_t *link_find_out(const roce_ctx_t *c, int l, uint32_t g, uint32_t *q) {
    for (int i = 0; i < ROCE_LINK_OPS; i++) {
        const roce_link_op_t *op = &c->link_ops[i];
        if (op->op == 0) continue;
        uint32_t index = g - op->first_out[l];
        if (index < op->rounds[l] * op->out_round[l]) {
            *q = index;
            return op;
        }
    }
    return NULL;
}

/* The op of inbound item g of link l, or NULL. */
static const roce_link_op_t *link_find_in(const roce_ctx_t *c, int l, uint32_t g) {
    for (int i = 0; i < ROCE_LINK_OPS; i++) {
        const roce_link_op_t *op = &c->link_ops[i];
        if (op->op == 0) continue;
        if (g - op->first_in[l] < op->rounds[l] * op->in_round[l]) return op;
    }
    return NULL;
}

/* Source of op-local outbound item q: own item (*own = 1), the inbound item it forwards (*own = 0),
 * or none (*own = -1: a forward of link 3 in the first D3 rounds of a staggered op, which is empty);
 * and its bytes (0: an empty item of a staggered link, posted as flags only). Round t of a link carries
 * piece t, except on a staggered link (link 2 with D, link 3 with D3), where its item of type r carries
 * piece t - r D (protocol.ring_partial); link 3 forwards as type r the inbound item of type r - 1 of
 * round t - D3, which carries the same piece (protocol.ring_forward_source). */
static void link_source(const roce_link_op_t *op, int l, uint32_t q, int *own, uint32_t *item, uint32_t *bytes) {
    uint32_t t = q / op->out_round[l], r = q % op->out_round[l];
    uint32_t stagger = l == 2 ? op->stagger : l == 3 ? op->stagger3 : 0u;
    int64_t p = (int64_t)t - (int64_t)r * (int64_t)stagger;
    if (p < 0 || p >= (int64_t)op->pieces) {
        *bytes = 0;
    } else {
        uint32_t left = op->bytes - (uint32_t)p * op->piece;
        *bytes = left < op->piece ? left : op->piece;
    }
    if (r < op->own_round[l]) {
        *own = 1;
        *item = op->first_own[l] + t * op->own_round[l] + r;
        /* Op word bit 24: the peer has no use for this rank's own items (it discards them), so they go
         * out as their flags only. */
        if (op->own_flags) *bytes = 0;
    } else if (l == 3 && t < op->stagger3) {
        *own = -1;
        *item = 0u;
    } else {
        uint32_t round = l == 3 ? t - op->stagger3 : t;
        *own = 0;
        *item = op->first_in[l] + round * op->in_round[l] + (r - op->own_round[l]);
    }
}

/* Write outbound item g of link l downstream: its stripe and flag on every lane. */
static int link_post_item(roce_ctx_t *c, int l, uint32_t g, uint64_t src, uint32_t bytes) {
    roce_link_t *lk = &c->link[l];
    int p = lk->out_peer;
    uint32_t m = g % (uint32_t)c->link_slots;
    uint64_t dst = link_slot_off(c, LK_RECV, l, m);
    uint32_t packs = bytes / 16u, tag = g + 1u;
    for (int lane = 0; lane < c->lane_count; lane++) {
        int d = c->lane_device[p][lane];
        roce_dev_t *dev = &c->dev[d];
        uint32_t first, count;
        lane_split(packs, c->lane_count, lane, &first, &count);
        struct ibv_sge flag_sge = {.addr = (uint64_t)(uintptr_t)&tag, .length = 4, .lkey = 0};
        struct ibv_send_wr flag_wr;
        memset(&flag_wr, 0, sizeof(flag_wr));
        flag_wr.wr_id = make_chain_wr_id(p, lane, ROCE_WR_LINK, l, g);
        flag_wr.sg_list = &flag_sge;
        flag_wr.num_sge = 1;
        flag_wr.opcode = IBV_WR_RDMA_WRITE;
        flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        flag_wr.wr.rdma.remote_addr = c->peer_addr[p] + link_flag_off(c, l, m, lane);
        flag_wr.wr.rdma.rkey = c->peer_rkey[lane][p];
        struct ibv_send_wr data_wr;
        struct ibv_sge data_sge;
        struct ibv_send_wr *head = &flag_wr;
        if (count != 0) {
            data_sge.addr = (uint64_t)(uintptr_t)(c->region + src + (uint64_t)first * 16u);
            data_sge.length = count * 16u;
            data_sge.lkey = dev->mr->lkey;
            memset(&data_wr, 0, sizeof(data_wr));
            data_wr.wr_id = flag_wr.wr_id;
            data_wr.next = &flag_wr;
            data_wr.sg_list = &data_sge;
            data_wr.num_sge = 1;
            data_wr.opcode = IBV_WR_RDMA_WRITE;
            data_wr.wr.rdma.remote_addr = c->peer_addr[p] + dst + (uint64_t)first * 16u;
            data_wr.wr.rdma.rkey = c->peer_rkey[lane][p];
            head = &data_wr;
        }
        struct ibv_send_wr *bad = NULL;
        int rc = ibv_post_send(dev->qp[p], head, &bad);
        if (rc != 0) {
            FAIL(c, "ibv_post_send of link item %u of link %d, lane %d to rank %d on %s: %s", g, l, lane, p,
                 dev->name, strerror(rc));
            return -1;
        }
        ack_push(c, d, p, count * 16u + 4u);
        atomic_fetch_add_explicit(&dev->bytes_posted, (uint64_t)count * 16u, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&c->link_items_posted, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&c->link_bytes_posted, bytes, memory_order_relaxed);
    return 0;
}

/* Post one chunk of lane `lane`'s stripe of outbound item g of link l, or (flag) its tag. */
static int link_post_lane(roce_ctx_t *c, int l, int lane, uint32_t g, uint64_t src, uint64_t dst, uint32_t bytes,
                          int flag) {
    roce_link_t *lk = &c->link[l];
    int p = lk->out_peer;
    int d = c->lane_device[p][lane];
    roce_dev_t *dev = &c->dev[d];
    uint32_t tag = g + 1u;
    struct ibv_sge sge;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr.rdma.rkey = c->peer_rkey[lane][p];
    if (flag) {
        sge.addr = (uint64_t)(uintptr_t)&tag;
        sge.length = 4;
        sge.lkey = 0;
        wr.wr_id = make_chain_wr_id(p, lane, ROCE_WR_LINK, l, g);
        wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        wr.wr.rdma.remote_addr = c->peer_addr[p] + dst;
    } else {
        sge.addr = (uint64_t)(uintptr_t)(c->region + src);
        sge.length = bytes;
        sge.lkey = dev->mr->lkey;
        wr.wr_id = make_chain_wr_id(p, lane, ROCE_WR_LINK_DATA, l, g);
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = c->peer_addr[p] + dst;
    }
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(dev->qp[p], &wr, &bad);
    if (rc != 0) {
        FAIL(c, "ibv_post_send of link item %u of link %d, lane %d to rank %d on %s: %s", g, l, lane, p, dev->name,
             strerror(rc));
        return -1;
    }
    ack_push(c, d, p, flag ? 4u : bytes);
    if (!flag) {
        atomic_fetch_add_explicit(&dev->bytes_posted, bytes, memory_order_relaxed);
        atomic_fetch_add_explicit(&c->link_window_chunks, 1, memory_order_relaxed);
        uint64_t now = dev->unacked[p];
        if (now > atomic_load_explicit(&c->fwd_max_unacked, memory_order_relaxed)) {
            atomic_store_explicit(&c->fwd_max_unacked, now, memory_order_relaxed);
        }
    }
    return 0;
}

/* Source, destination and bytes of outbound item g of link l, and whether it is ready; -1 on a failure. */
static int link_item(roce_ctx_t *c, int l, uint32_t g, uint64_t *src, uint32_t *bytes, int *ready) {
    uint32_t slots = (uint32_t)c->link_slots, q, item;
    int own;
    const roce_link_op_t *op = link_find_out(c, l, g, &q);
    if (op == NULL) {
        FAIL(c, "link item %u of link %d has no op", g, l);
        return -1;
    }
    link_source(op, l, q, &own, &item, bytes);
    *ready = 1;
    if (own < 0) {
        *src = 0;
    } else if (own) {
        *ready = *bytes == 0 ||
                 __atomic_load_n(link_word(c, LK_READY, (uint64_t)l * ROCE_FLAG_STRIDE + 4u * (item % slots)),
                                 __ATOMIC_ACQUIRE) == item + 1u;
        *src = link_slot_off(c, LK_OWN, l, item % slots);
    } else {
        for (int lane = 0; lane < c->lane_count && *ready; lane++) {
            volatile uint32_t *flag = (volatile uint32_t *)(c->region + link_flag_off(c, l, item % slots, lane));
            *ready = __atomic_load_n(flag, __ATOMIC_ACQUIRE) == item + 1u;
        }
        *src = link_slot_off(c, LK_RECV, l, item % slots);
    }
    return 0;
}

/* Advance a link toward a peer through relays: every lane posts its stripe of each item in
 * forward-window chunks on its own, and an item is posted once every lane's flag is. */
static int link_pump_windowed(roce_ctx_t *c, int l, uint32_t credit, int *work) {
    roce_link_t *lk = &c->link[l];
    int p = lk->out_peer;
    uint32_t slots = (uint32_t)c->link_slots;
    for (int lane = 0; lane < c->lane_count; lane++) {
        roce_dev_t *dev = &c->dev[c->lane_device[p][lane]];
        uint32_t window = c->link_ring_window;
        while (lk->lane_item[lane] != lk->known_out) {
            uint32_t g = lk->lane_item[lane], bytes, first, count;
            uint64_t src;
            int ready;
            if ((int32_t)(credit - (g + 1u - slots)) < 0) break;
            if (link_item(c, l, g, &src, &bytes, &ready) != 0) return -1;
            if (!ready) break;
            trace_ready(c, ROCE_TRACE_LINK_STREAM + l, g + 1u, &lk->ready_seen);
            lane_split(bytes / 16u, c->lane_count, lane, &first, &count);
            uint64_t dst = link_slot_off(c, LK_RECV, l, g % slots) + (uint64_t)first * 16u;
            uint32_t stripe = count * 16u;
            while (lk->lane_bytes[lane] < stripe && dev->ack_count[p] < ROCE_CREDIT - 1) {
                uint32_t n = stripe - lk->lane_bytes[lane];
                if (n > ROCE_LINK_WINDOW_CHUNK) n = ROCE_LINK_WINDOW_CHUNK;
                if (dev->unacked[p] + n > window) break;
                if (link_post_lane(c, l, lane, g, src + (uint64_t)first * 16u + lk->lane_bytes[lane],
                                   dst + lk->lane_bytes[lane], n, 0) != 0) return -1;
                lk->lane_bytes[lane] += n;
                *work = 1;
            }
            if (lk->lane_bytes[lane] < stripe || dev->ack_count[p] >= ROCE_CREDIT - 1) break;
            if (link_post_lane(c, l, lane, g, 0, link_flag_off(c, l, g % slots, lane), 4u, 1) != 0) return -1;
            lk->lane_item[lane]++;
            lk->lane_bytes[lane] = 0;
            *work = 1;
        }
    }
    uint32_t posted = lk->lane_item[0];
    for (int lane = 1; lane < c->lane_count; lane++) {
        if ((int32_t)(lk->lane_item[lane] - posted) < 0) posted = lk->lane_item[lane];
    }
    while (lk->next_post != posted) {
        uint64_t src;
        uint32_t bytes;
        int ready;
        if (link_item(c, l, lk->next_post, &src, &bytes, &ready) != 0) return -1;
        atomic_fetch_add_explicit(&c->link_items_posted, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&c->link_bytes_posted, bytes, memory_order_relaxed);
        lk->next_post++;
        trace_event(c, ROCE_EV_POSTED, ROCE_TRACE_LINK_STREAM + l, lk->next_post);
    }
    return 0;
}

/* Tell the upstream neighbor of link l that `value` inbound items are released. */
static int link_post_credit(roce_ctx_t *c, int l, uint32_t value) {
    int p = c->link[l].in_peer;
    int d = c->lane_device[p][0];
    roce_dev_t *dev = &c->dev[d];
    struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = make_chain_wr_id(p, 0, ROCE_WR_LINK_CREDIT, l, value);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    wr.wr.rdma.remote_addr = c->peer_addr[p] + c->link_off + c->link_layout[LK_CREDIT] + (uint64_t)l * ROCE_FLAG_STRIDE;
    wr.wr.rdma.rkey = c->peer_rkey[0][p];
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(dev->qp[p], &wr, &bad);
    if (rc != 0) {
        FAIL(c, "ibv_post_send of link credit %u of link %d to rank %d on %s: %s", value, l, p, dev->name,
             strerror(rc));
        return -1;
    }
    ack_push(c, d, p, 4u);
    atomic_fetch_add_explicit(&c->link_credits_sent, 1, memory_order_relaxed);
    return 0;
}

/* Advance both links as far as they go without waiting; 1 when anything moved, 0 when
 * nothing did, -1 on a failure. */
static int link_service(roce_ctx_t *c) {
    int work = 0;
    if (link_take_ops(c, &work) != 0) return -1;
    uint32_t slots = (uint32_t)c->link_slots;
    int pending = 0;
    for (int l = 0; l < ROCE_LINKS; l++) {
        roce_link_t *lk = &c->link[l];
        if (lk->out_peer < 0) continue;
        uint32_t credit = __atomic_load_n(link_word(c, LK_CREDIT, (uint64_t)l * ROCE_FLAG_STRIDE), __ATOMIC_ACQUIRE);
        trace_credit_in(c, ROCE_TRACE_LINK_STREAM + l, credit, &lk->credit_seen);
        if (lk->windowed) {
            if (link_pump_windowed(c, l, credit, &work) != 0) return -1;
            if (lk->done != lk->next_post || lk->next_post != lk->known_out) pending = 1;
            continue;
        }
        while (lk->next_post != lk->known_out) {
            uint32_t g = lk->next_post, q, item, bytes;
            int own;
            const roce_link_op_t *op = link_find_out(c, l, g, &q);
            if (op == NULL) {
                FAIL(c, "link item %u of link %d has no op", g, l);
                return -1;
            }
            link_source(op, l, q, &own, &item, &bytes);
            int ready = 1;
            uint64_t src = 0;
            if (own < 0) {
                /* An empty forward of link 3 before round D3: flags only, no source. */
            } else if (own) {
                ready = bytes == 0 ||
                        __atomic_load_n(link_word(c, LK_READY, (uint64_t)l * ROCE_FLAG_STRIDE + 4u * (item % slots)),
                                        __ATOMIC_ACQUIRE) == item + 1u;
                src = link_slot_off(c, LK_OWN, l, item % slots);
            } else {
                for (int lane = 0; lane < c->lane_count && ready; lane++) {
                    volatile uint32_t *flag = (volatile uint32_t *)(c->region + link_flag_off(c, l, item % slots, lane));
                    ready = __atomic_load_n(flag, __ATOMIC_ACQUIRE) == item + 1u;
                }
                src = link_slot_off(c, LK_RECV, l, item % slots);
            }
            if (!ready) break;
            trace_ready(c, ROCE_TRACE_LINK_STREAM + l, g + 1u, &lk->ready_seen);
            if ((int32_t)(credit - (g + 1u - slots)) < 0 || !chain_room(c, lk->out_peer, c->lane_count)) break;
            if (link_post_item(c, l, g, src, bytes) != 0) return -1;
            trace_event(c, ROCE_EV_POSTED, ROCE_TRACE_LINK_STREAM + l, g + 1u);
            lk->next_post++;
            work = 1;
        }
        if (lk->done != lk->next_post) pending = 1;
    }
    for (int l = 0; l < ROCE_LINKS; l++) {
        const roce_link_t *lk = &c->link[l];
        if (lk->in_peer >= 0 && (lk->credited != lk->released || lk->released != lk->known_in)) pending = 1;
    }
    if (pending && drain_all(c) != 0) return -1;
    for (int l = 0; l < ROCE_LINKS; l++) {
        roce_link_t *lk = &c->link[l];
        if (lk->out_peer >= 0) {
            uint32_t done = lk->lane_done[0];
            for (int lane = 1; lane < c->lane_count; lane++) {
                if ((int32_t)(lk->lane_done[lane] - done) < 0) done = lk->lane_done[lane];
            }
            if (done != lk->done) {
                uint32_t own_done = lk->own_done;
                while (lk->done != done) {
                    uint32_t q, item, bytes;
                    int own;
                    const roce_link_op_t *op = link_find_out(c, l, lk->done, &q);
                    if (op == NULL) {
                        FAIL(c, "completed link item %u of link %d has no op", lk->done, l);
                        return -1;
                    }
                    link_source(op, l, q, &own, &item, &bytes);
                    if (own > 0) {
                        lk->own_done++;
                    } else if (own == 0) {
                        lk->fwd_done = item + 1u;
                    }
                    lk->done++;
                    trace_event(c, ROCE_EV_DONE, ROCE_TRACE_LINK_STREAM + l, lk->done);
                }
                if (lk->own_done != own_done) {
                    __atomic_store_n(link_word(c, LK_SENT, (uint64_t)l * ROCE_FLAG_STRIDE), lk->own_done,
                                     __ATOMIC_RELEASE);
                }
                work = 1;
            }
        }
        if (lk->in_peer < 0) continue;
        while (__atomic_load_n(link_word(c, LK_CONSUMED, (uint64_t)l * ROCE_FLAG_STRIDE + 4u * (lk->consumed % slots)),
                               __ATOMIC_ACQUIRE) == lk->consumed + 1u) {
            lk->consumed++;
            trace_event(c, ROCE_EV_CONSUMED, ROCE_TRACE_LINK_STREAM + l, lk->consumed);
        }
        while (lk->released != lk->consumed) {
            const roce_link_op_t *op = link_find_in(c, l, lk->released);
            if (op == NULL) {
                /* The kernel finished the item before this thread took its op from the doorbell
                 * (a block of the launch may run before the one that rings it): the item is
                 * released once the op is taken, within the session's wait limit. */
                uint64_t now = monotonic_ns();
                if (lk->orphan_ns == 0) lk->orphan_ns = now;
                uint32_t limit_us = ctrl_words(c)[ROCE_CTRL_WAIT_LIMIT_US];
                uint64_t limit_ns = limit_us != 0 ? (uint64_t)limit_us * 1000ull : 20000000000ull;
                if (now - lk->orphan_ns > limit_ns) {
                    FAIL(c, "inbound link item %u of link %d was finished by the kernel, and no op that holds it "
                            "was rung within %llu ms (newest op taken %u)", lk->released, l,
                         (unsigned long long)(limit_ns / 1000000ull), c->link_last_seq);
                    return -1;
                }
                break;
            }
            lk->orphan_ns = 0;
            /* The first out - own items of an inbound round are forwarded downstream, on link 3
             * D3 rounds later and so only up to round rounds - D3. */
            uint32_t t = (lk->released - op->first_in[l]) / op->in_round[l];
            uint32_t r = (lk->released - op->first_in[l]) % op->in_round[l];
            int forwarded = r < op->out_round[l] - op->own_round[l] &&
                            (l != 3 || t + op->stagger3 < op->rounds[l]);
            if (forwarded && (int32_t)(lk->fwd_done - (lk->released + 1u)) < 0) break;
            lk->released++;
        }
        if (lk->released != lk->credited && chain_room(c, lk->in_peer, 1)) {
            if (link_post_credit(c, l, lk->released) != 0) return -1;
            trace_event(c, ROCE_EV_CREDIT_OUT, ROCE_TRACE_LINK_STREAM + l, lk->released);
            lk->credited = lk->released;
            work = 1;
        }
    }
    return work;
}

/* -- posting ---------------------------------------------------------------------- */

/* One lane: `bytes` from send + byte_offset to the peer's recv[rank][slot] at
 * the same offset (skipped when empty), then `seq` into flag line
 * (rank * SLOTS + slot) * L + flag_line, where flag_line includes the
 * namespace offset. Both on the lane's queue pair. */
static int post_lane_write(roce_ctx_t *c, uint32_t seq, int p, int lane, uint8_t *send,
                           uint64_t byte_offset, uint32_t bytes, int flag_line) {
    int d = c->lane_device[p][lane];
    roce_dev_t *dev = &c->dev[d];
    uint32_t slot = seq & 1u;
    uint64_t remote = c->peer_addr[p];
    uint64_t data_addr = remote + c->recv_off +
                         ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes + byte_offset;
    uint64_t line = ((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->lane_count +
                    (uint64_t)flag_line;
    uint64_t flag_addr = remote + c->flag_off + line * ROCE_FLAG_STRIDE;
    uint32_t rkey = c->peer_rkey[lane][p];
    uint32_t window = c->fwd_window[p][lane];
    int ns = flag_line / (c->world * ROCE_SLOTS * c->lane_count);
    if (window != 0 && bytes <= c->fwd_chunk && dev->unacked[p] + bytes + 4u > window) fwd_prove(c, p);
    if (window != 0 && (bytes > c->fwd_chunk || dev->unacked[p] + bytes + 4u > window)) {
        /* Posted by pump_streams after the lanes without windows. A stripe of at
         * most one chunk that fits the window goes out at once like any lane.
         * A queue pair carries at most one stripe per phase, and pump_streams
         * finishes the phase's stripes before the next phase, so each queue
         * pair keeps its order. */
        roce_stream_t *s = &c->streams[c->n_streams++];
        s->peer = p;
        s->lane = lane;
        s->dev = d;
        s->seq = seq;
        s->source = send + byte_offset;
        s->remote_data = data_addr;
        s->remote_flag = flag_addr;
        s->rkey = rkey;
        s->length = bytes;
        s->done = 0;
        s->flag_posted = 0;
        s->ns = ns;
        return 0;
    }
    if (wait_credit(c, d, p) != 0) return -1;
    HOOK(ROCE_HOOK_LANE, seq, p);
    uint32_t value = seq;
    struct ibv_sge flag_sge = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
    struct ibv_send_wr flag_wr;
    memset(&flag_wr, 0, sizeof(flag_wr));
    flag_wr.wr_id = make_wr_id(p, lane, ROCE_WR_OP, seq);
    flag_wr.sg_list = &flag_sge;
    flag_wr.num_sge = 1;
    flag_wr.opcode = IBV_WR_RDMA_WRITE;
    flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    flag_wr.wr.rdma.remote_addr = flag_addr;
    flag_wr.wr.rdma.rkey = rkey;
    struct ibv_send_wr data_wr;
    struct ibv_sge data_sge;
    struct ibv_send_wr *first = &flag_wr;
    if (bytes != 0) {
        data_sge.addr = (uint64_t)(uintptr_t)(send + byte_offset);
        data_sge.length = bytes;
        data_sge.lkey = dev->mr->lkey;
        memset(&data_wr, 0, sizeof(data_wr));
        data_wr.wr_id = flag_wr.wr_id;
        data_wr.next = &flag_wr;
        data_wr.sg_list = &data_sge;
        data_wr.num_sge = 1;
        data_wr.opcode = IBV_WR_RDMA_WRITE;
        data_wr.wr.rdma.remote_addr = data_addr;
        data_wr.wr.rdma.rkey = rkey;
        first = &data_wr;
    }
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(dev->qp[p], first, &bad);
    if (rc != 0) {
        FAIL(c, "ibv_post_send of lane %d to rank %d on %s at sequence %u: %s", lane, p, dev->name,
             seq, strerror(rc));
        return -1;
    }
    ack_push_arena(c, d, p, bytes + 4u, seq, ns);
    atomic_fetch_add_explicit(&dev->bytes_posted, bytes, memory_order_relaxed);
    return 0;
}

/* Packs [first, first + count) of send[slot] to the same offsets at peer p,
 * striped over its lanes, with flags in namespace ns. */
static int post_range(roce_ctx_t *c, uint32_t seq, int p, uint8_t *send, uint32_t first,
                      uint32_t count, int ns) {
    int base = ns * c->world * ROCE_SLOTS * c->lane_count;
    for (int lane = 0; lane < c->lane_count; lane++) {
        uint32_t stripe_first, stripe_count;
        lane_split(count, c->lane_count, lane, &stripe_first, &stripe_count);
        if (post_lane_write(c, seq, p, lane, send, (uint64_t)(first + stripe_first) * 16u,
                            stripe_count * 16u, base + lane) != 0) {
            return -1;
        }
    }
    return 0;
}

static uint8_t *send_slot(const roce_ctx_t *c, uint32_t seq) {
    return c->region + c->send_off + (uint64_t)(seq & 1u) * c->slot_bytes;
}

static int check_payload(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    if (nbytes == 0 || nbytes % 16u != 0 || (uint64_t)nbytes > c->slot_bytes) {
        FAIL(c, "op at sequence %u has %u bytes; payloads are positive multiples of 16 bytes up to "
                "the slot size %llu", seq, nbytes, (unsigned long long)c->slot_bytes);
        return -1;
    }
    return 0;
}

static int post_oneshot(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    if (check_payload(c, seq, nbytes) != 0) return -1;
    uint8_t *send = send_slot(c, seq);
    for (int i = 0; i < c->world - 1; i++) {
        int p = c->post_order[i];
        HOOK(ROCE_HOOK_PEER, seq, p);
        if (post_range(c, seq, p, send, 0, nbytes / 16u, 0) != 0) return -1;
    }
    if (pump_streams(c) != 0) return -1;
    atomic_fetch_add_explicit(&c->ops_posted, 1, memory_order_relaxed);
    return drain_all(c);
}

/* Phase 0 (chunk p to every peer p, namespace 0) or phase 1 (own chunk to every
 * peer, namespace 1) of a two-shot or scatter op. */
static int post_chunks(roce_ctx_t *c, uint32_t seq, uint32_t nbytes, int phase) {
    if (check_payload(c, seq, nbytes) != 0) return -1;
    uint8_t *send = send_slot(c, seq);
    uint32_t packs = nbytes / 16u;
    for (int i = 0; i < c->world - 1; i++) {
        int p = c->post_order[i];
        uint32_t first, count;
        chunk_range(packs, c->world, phase == 0 ? p : c->rank, &first, &count);
        HOOK(ROCE_HOOK_PEER, seq, p);
        if (post_range(c, seq, p, send, first, count, phase) != 0) return -1;
    }
    if (pump_streams(c) != 0) return -1;
    if (phase == 0) {
        atomic_fetch_add_explicit(&c->ops_posted, 1, memory_order_relaxed);
    } else {
        atomic_fetch_add_explicit(&c->phases_posted, 1, memory_order_relaxed);
    }
    return drain_all(c);
}

/* Phase k of a described op: the chunk range of descriptor k to its peer. */
static int post_described(roce_ctx_t *c, uint32_t seq, uint32_t nbytes, int phase) {
    if (check_payload(c, seq, nbytes) != 0) return -1;
    uint32_t d = __atomic_load_n(&ctrl_words(c)[ROCE_CTRL_DESC + phase], __ATOMIC_ACQUIRE);
    uint32_t first = d & 31u, end = (d >> 5) & 31u, peer = (d >> 10) & 15u, ns = (d >> 14) & 1u;
    if ((d >> 31) == 0 || first > end || end > (uint32_t)c->world || peer >= (uint32_t)c->world ||
        (int)peer == c->rank) {
        FAIL(c, "invalid phase %d descriptor 0x%08x at sequence %u", phase, d, seq);
        return -1;
    }
    uint32_t packs = nbytes / 16u, lo, hi, unused;
    chunk_range(packs, c->world, (int)first, &lo, &unused);
    chunk_range(packs, c->world, (int)end, &hi, &unused);
    if (end == (uint32_t)c->world) hi = packs;
    HOOK(ROCE_HOOK_PEER, seq, (int)peer);
    if (post_range(c, seq, (int)peer, send_slot(c, seq), lo, hi - lo, (int)ns) != 0) return -1;
    if (pump_streams(c) != 0) return -1;
    if (phase == 0) {
        atomic_fetch_add_explicit(&c->ops_posted, 1, memory_order_relaxed);
    } else {
        atomic_fetch_add_explicit(&c->phases_posted, 1, memory_order_relaxed);
    }
    return drain_all(c);
}

static int op_phases(uint32_t op) {
    return op == ROCE_OP_TWOSHOT ? 2 : op == ROCE_OP_DESCRIBED ? ROCE_MAX_PHASES : 1;
}

static int post_first_phase(roce_ctx_t *c, uint32_t seq, uint32_t word) {
    c->op_seq[seq % ROCE_OP_HISTORY] = seq;
    c->op_code[seq % ROCE_OP_HISTORY] = (uint8_t)(c->multi_phase ? word >> ROCE_OP_SHIFT : ROCE_OP_ONESHOT);
    note_op_lines(c, seq, c->op_code[seq % ROCE_OP_HISTORY]);
    if (!c->multi_phase) return post_oneshot(c, seq, word);
    uint32_t op = word >> ROCE_OP_SHIFT, nbytes = word & ROCE_OP_BYTES_MASK;
    switch (op) {
    case ROCE_OP_ONESHOT:
        return post_oneshot(c, seq, nbytes);
    case ROCE_OP_TWOSHOT:
    case ROCE_OP_SCATTER:
        return post_chunks(c, seq, nbytes, 0);
    default:
        return post_described(c, seq, nbytes, 0);
    }
}

/* Later phases whose doorbell rang once the previous phase of the same op was
 * posted. Returns 1 when something was posted, 0 when nothing was due. */
static int service_phases(roce_ctx_t *c) {
    volatile uint32_t *ctrl = ctrl_words(c);
    int posted = 0;
    for (int k = 1; k < ROCE_MAX_PHASES; k++) {
        uint32_t seq = __atomic_load_n(&ctrl[ROCE_CTRL_PHASE + k], __ATOMIC_ACQUIRE);
        if (seq == c->last_phase_seq[k]) continue;
        uint32_t before = k == 1 ? c->last_seq : c->last_phase_seq[k - 1];
        if ((int32_t)(seq - before) > 0) continue;
        uint32_t word = __atomic_load_n(&ctrl[ROCE_CTRL_OP_WORD + (seq & 1u)], __ATOMIC_ACQUIRE);
        uint32_t op = word >> ROCE_OP_SHIFT, nbytes = word & ROCE_OP_BYTES_MASK;
        if (k >= op_phases(op)) {
            FAIL(c, "phase %d doorbell at sequence %u for op word 0x%08x, whose op code %u has %d "
                    "phase(s)", k, seq, word, op, op_phases(op));
            return -1;
        }
        HOOK(ROCE_HOOK_PHASE, seq, k);
        c->posting_seq = seq;
        int rc = op == ROCE_OP_TWOSHOT ? post_chunks(c, seq, nbytes, 1)
                                       : post_described(c, seq, nbytes, k);
        if (rc != 0) return -1;
        c->last_phase_seq[k] = seq;
        posted = 1;
    }
    return posted;
}

static void note_cpu(roce_ctx_t *c) {
    int cpu = sched_getcpu();
    int last = atomic_load_explicit(&c->last_cpu, memory_order_relaxed);
    if (cpu != last) {
        if (last >= 0) atomic_fetch_add_explicit(&c->cpu_migrations, 1, memory_order_relaxed);
        atomic_store_explicit(&c->last_cpu, cpu, memory_order_relaxed);
    }
}

static void *progress_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    volatile uint32_t *ctrl = ctrl_words(c);
    uint64_t idle = 0;
    const struct timespec nap = {0, ROCE_NAP_NS};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        int work = 0;
        if (c->chain_slots) {
            int moved = chain_service(c);
            if (moved < 0) goto failed;
            work |= moved;
        }
        if (c->link_slots) {
            int moved = link_service(c);
            if (moved < 0) goto failed;
            work |= moved;
        }
        if (c->multi_phase) {
            int later = service_phases(c);
            if (later < 0) goto failed;
            work |= later;
        }
        uint32_t seq = __atomic_load_n(&ctrl[ROCE_CTRL_DOORBELL], __ATOMIC_ACQUIRE);
        if (seq != c->last_seq) {
            uint32_t pending = seq - c->last_seq;
            if (pending > ROCE_SLOTS) {
                FAIL(c, "doorbell skipped %u ops (newest posted %u, doorbell %u)", pending,
                     c->last_seq, seq);
                goto failed;
            }
            for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
                uint32_t word = __atomic_load_n(&ctrl[ROCE_CTRL_OP_WORD + (s & 1u)], __ATOMIC_ACQUIRE);
                HOOK(ROCE_HOOK_DOORBELL, s, -1);
                c->posting_seq = s;
                if (post_first_phase(c, s, word) != 0) goto failed;
                c->last_seq = s;
                atomic_store_explicit(&c->posted_seq, s, memory_order_release);
                note_cpu(c);
                if (c->multi_phase && service_phases(c) < 0) goto failed;
            }
            work = 1;
        }
        if (work) {
            idle = 0;
            continue;
        }
        idle++;
        if (idle % 64 == 0 && drain_all(c) != 0) goto failed;
        if (idle >= ROCE_IDLE_SPINS) nanosleep(&nap, NULL);
    }
    return NULL;
failed:
    atomic_store(&c->failed, 1);
    return NULL;
}

/* -- lane check: one small write per lane before the session is ready ------------- */

static int64_t monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* One signaled 4-byte write per (peer, lane) into the peer's lane-check word;
 * every completion must arrive successfully within `timeout_ms`. Runs after
 * every rank connected and before the progress thread starts. */
int roce_lane_check(roce_ctx_t *c, int timeout_ms) {
    int pending[ROCE_MAX_PEERS][ROCE_MAX_LANES];
    int remaining = 0;
    uint32_t value = ROCE_RECORD_MAGIC;
    if (!c->connected || atomic_load(&c->running)) {
        FAIL(c, "the lane check runs on a connected session before its progress thread starts");
        return -1;
    }
    memset(pending, 0, sizeof(pending));
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) continue;
        for (int l = 0; l < c->lane_count; l++) {
            int d = c->lane_device[p][l];
            roce_dev_t *dev = &c->dev[d];
            struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
            struct ibv_send_wr wr;
            memset(&wr, 0, sizeof(wr));
            wr.wr_id = make_wr_id(p, l, ROCE_WR_CHECK, 0);
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            wr.wr.rdma.remote_addr = c->peer_addr[p] + c->ctrl_off + 4u * ROCE_CTRL_LANE_CHECK;
            wr.wr.rdma.rkey = c->peer_rkey[l][p];
            struct ibv_send_wr *bad = NULL;
            int rc = ibv_post_send(dev->qp[p], &wr, &bad);
            if (rc != 0) {
                FAIL(c, "lane check: posting lane %d of rank %d toward rank %d on %s: %s", l,
                     c->rank, p, dev->name, strerror(rc));
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
                char where[64];
                format_address(c->peer_gid[p][l], where, sizeof(where));
                if (wc[i].status != IBV_WC_SUCCESS) {
                    FAIL(c, "lane check: lane %d of rank %d toward rank %d (local device %s, destination "
                            "%s) failed: %s (vendor_err 0x%x)", l, c->rank, p, c->dev[d].name, where,
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
                    FAIL(c, "lane check: lane %d of rank %d toward rank %d (local device %s, destination "
                            "%s) did not complete within %d ms", l, c->rank, p,
                         c->dev[c->lane_device[p][l]].name, where, timeout_ms > 0 ? timeout_ms : 2000);
                    return -1;
                }
            }
        }
    }
    return 0;
}

/* -- lifecycle ------------------------------------------------------------------------ */

static void link_set_windowed(roce_ctx_t *c) {
    for (int l = 0; l < ROCE_LINKS; l++) {
        roce_link_t *lk = &c->link[l];
        lk->windowed = l >= 2 && c->link_slots && lk->out_peer >= 0 && c->link_ring_window != 0;
    }
}

int roce_start(roce_ctx_t *c) {
    if (!c->connected) {
        FAIL(c, "the progress thread needs a connected session");
        return -1;
    }
    if (atomic_load(&c->running)) return 0;
    link_set_windowed(c);
    pthread_attr_t attr;
    pthread_attr_t *use = NULL;
    const char *cpus = getenv("SIRCL_PROGRESS_CPU");
    if (cpus != NULL && *cpus != '\0') {
        cpu_set_t set;
        if (parse_cpu_list(cpus, &set) != 0) {
            FAIL(c, "SIRCL_PROGRESS_CPU=%s is not a CPU list such as 9 or 5-9,15-19", cpus);
            return -1;
        }
        pthread_attr_init(&attr);
        int rc = pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
        if (rc != 0) {
            pthread_attr_destroy(&attr);
            FAIL(c, "pinning the progress thread to SIRCL_PROGRESS_CPU=%s: %s", cpus, strerror(rc));
            return -1;
        }
        use = &attr;
    }
    if (!c->started) {
        /* A restart keeps the newest posted sequences, so ops that rang while
         * the thread was stopped are posted when it resumes. */
        volatile uint32_t *ctrl = ctrl_words(c);
        c->last_seq = ctrl[ROCE_CTRL_DOORBELL];
        c->posting_seq = c->last_seq;
        for (int k = 1; k < ROCE_MAX_PHASES; k++) c->last_phase_seq[k] = ctrl[ROCE_CTRL_PHASE + k];
        atomic_store(&c->posted_seq, c->last_seq);
        c->started = 1;
    }
    atomic_store(&c->failed, 0);
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

void roce_stop(roce_ctx_t *c) {
    if (c != NULL && atomic_exchange(&c->running, 0)) pthread_join(c->thread, NULL);
}

int roce_failed(roce_ctx_t *c) { return atomic_load(&c->failed); }

const char *roce_error(roce_ctx_t *c) { return c->err; }

/* Keep the newest `capacity` event records of chain streams and links (0: no trace).
 * Set before the progress thread starts. */
int roce_set_trace(roce_ctx_t *c, uint32_t capacity) {
    if (atomic_load(&c->running)) {
        FAIL(c, "the event trace is set before the progress thread starts");
        return -1;
    }
    if (capacity > ROCE_TRACE_MAX_RECORDS) {
        FAIL(c, "event trace of %u records (at most %u)", capacity, ROCE_TRACE_MAX_RECORDS);
        return -1;
    }
    free(c->trace);
    c->trace = NULL;
    c->trace_cap = 0;
    atomic_store(&c->trace_written, 0);
    c->trace_taken = 0;
    if (capacity == 0) return 0;
    c->trace = calloc(capacity, sizeof(roce_trace_rec_t));
    if (c->trace == NULL) {
        FAIL(c, "no memory for an event trace of %u records", capacity);
        return -1;
    }
    c->trace_cap = capacity;
    return 0;
}

/* Copy up to `max_records` event records written since the last take, oldest first, two
 * words each (CLOCK_REALTIME nanoseconds; value | event << 32 | stream << 48). Returns
 * the records copied; `lost` receives the records overwritten before they were taken. */
int64_t roce_trace_take(roce_ctx_t *c, uint64_t *out, uint64_t max_records, uint64_t *lost) {
    *lost = 0;
    if (c->trace_cap == 0) return 0;
    uint64_t cap = c->trace_cap;
    uint64_t written = atomic_load_explicit(&c->trace_written, memory_order_acquire);
    uint64_t from = c->trace_taken;
    if (written - from > cap) {
        *lost = written - cap - from;
        from = written - cap;
    }
    uint64_t n = written - from;
    if (n > max_records) n = max_records;
    for (uint64_t i = 0; i < n; i++) {
        const roce_trace_rec_t *r = &c->trace[(from + i) % cap];
        out[2 * i] = r->ns;
        out[2 * i + 1] = (uint64_t)r->value | ((uint64_t)r->event << 32) | ((uint64_t)r->stream << 48);
    }
    /* Records the progress thread reused while they were copied are dropped as lost. */
    uint64_t after = atomic_load_explicit(&c->trace_written, memory_order_acquire);
    uint64_t valid = after > cap ? after - cap : 0;
    uint64_t torn = valid > from ? valid - from : 0;
    if (torn > n) torn = n;
    if (torn != 0) {
        memmove(out, out + 2 * torn, (size_t)(2 * (n - torn)) * sizeof(uint64_t));
        *lost += torn;
    }
    c->trace_taken = from + n;
    return (int64_t)(n - torn);
}

/* Phase tracing is unsupported by this build. */
int roce_tracing(roce_ctx_t *c) {
    (void)c;
    return 0;
}

int roce_trace_read(roce_ctx_t *c, uint64_t *out, uint64_t words) {
    (void)c;
    (void)out;
    (void)words;
    return 0;
}

uint64_t roce_stat(roce_ctx_t *c, int which) {
    switch (which) {
    case 0: return atomic_load_explicit(&c->ops_posted, memory_order_relaxed);
    case 1: return atomic_load_explicit(&c->writes_completed, memory_order_relaxed);
    case 2: return atomic_load_explicit(&c->posted_seq, memory_order_acquire);
    case 3: return (uint64_t)c->lane_count;
    case 5: return atomic_load_explicit(&c->phases_posted, memory_order_relaxed);
    case 6: return (uint64_t)c->multi_phase;
    case 7: return (uint64_t)(atomic_load_explicit(&c->last_cpu, memory_order_relaxed) + 1);
    case 8: return atomic_load_explicit(&c->cpu_migrations, memory_order_relaxed);
    case 9: return 0;
    case 10: return atomic_load_explicit(&c->fwd_chunks_posted, memory_order_relaxed);
    case 11: return atomic_load_explicit(&c->fwd_max_unacked, memory_order_relaxed);
    case 12: return atomic_load_explicit(&c->chain_ops_seen, memory_order_relaxed);
    case 13: return atomic_load_explicit(&c->chain_chunks_posted, memory_order_relaxed);
    case 14: return atomic_load_explicit(&c->chain_credits_sent, memory_order_relaxed);
    case 15: return atomic_load_explicit(&c->chain_bytes_posted, memory_order_relaxed);
    case 16: return atomic_load_explicit(&c->link_ops_seen, memory_order_relaxed);
    case 17: return atomic_load_explicit(&c->link_items_posted, memory_order_relaxed);
    case 18: return atomic_load_explicit(&c->link_credits_sent, memory_order_relaxed);
    case 19: return atomic_load_explicit(&c->link_bytes_posted, memory_order_relaxed);
    case 24: return atomic_load_explicit(&c->link_window_chunks, memory_order_relaxed);
    case 25: return atomic_load_explicit(&c->fwd_waits, memory_order_relaxed);
    case 26: return atomic_load_explicit(&c->fwd_wait_ns, memory_order_relaxed);
    case 27: return atomic_load_explicit(&c->fwd_wait_max_ns, memory_order_relaxed);
    case 28: return atomic_load_explicit(&c->fwd_proven_bytes, memory_order_relaxed);
    case 29: return (uint64_t)c->fwd_proof;
    default: return 0;
    }
}

uint64_t roce_hca_stat(roce_ctx_t *c, int device, int which) {
    if (device < 0 || device >= c->n_dev) return UINT64_MAX;
    switch (which) {
    case 0: return atomic_load_explicit(&c->dev[device].writes_completed, memory_order_relaxed);
    case 1: return atomic_load_explicit(&c->dev[device].bytes_posted, memory_order_relaxed);
    default: return 0;
    }
}

/* Stops the progress thread and releases every verbs object; returns the number of verbs calls that
 * failed. A queue pair or memory registration that could not be released can still let a peer's write
 * reach the arena, so the caller keeps the arena allocated when the count is not zero. */
int roce_destroy(roce_ctx_t *c) {
    if (c == NULL) return 0;
    roce_stop(c);
    free(c->trace);
    c->trace = NULL;
    c->trace_cap = 0;
    int failed = 0;
    for (int d = 0; d < ROCE_MAX_DEVICES; d++) {
        roce_dev_t *dev = &c->dev[d];
        for (int p = 0; p < ROCE_MAX_PEERS; p++) {
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
