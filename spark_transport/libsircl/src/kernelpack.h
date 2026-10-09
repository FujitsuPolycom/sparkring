/* The ahead-of-time CUDA kernel packs (the CUDA C++ sources in kernels/,
 * each embedded as one fatbin): the transport pack (ring-session kernels), the
 * fold pack (local rank-ordered reductions), the link pack (the chain
 * all-reduce and the link collectives) and the point-to-point pack (the send
 * and receive kernels of SIRCL's point-to-point channels). Loading per CUDA
 * context, entry lookup and launches from C. */
#ifndef SCCL_KERNELPACK_H
#define SCCL_KERNELPACK_H
#include "cuda_api.h"

enum { SCCL_DT_F32 = 0, SCCL_DT_F16 = 1, SCCL_DT_BF16 = 2, SCCL_DT_COUNT = 3 };
/* Kernel kinds; the first two are the all-reduce algorithms. The all-gather and the all-to-all
 * move bytes and have one entry per group size (dtype index 0). */
enum {
  SCCL_ALG_ONESHOT = 0,
  SCCL_ALG_TWOSHOT = 1,
  SCCL_ALG_COUNT = 2,
  SCCL_K_SCATTER = 2,
  SCCL_K_ALLGATHER = 3,
  SCCL_K_ALLTOALL = 4,
  SCCL_K_COUNT = 5
};
enum { SCCL_KP_MIN_WORLD = 2, SCCL_KP_MAX_WORLD = 8 };
/* Fold entries: one per NCCL datatype (ncclInt8 ... ncclFloat8e5m2) and built-in op (ncclSum ... ncclAvg). */
enum { SCCL_FOLD_DTYPES = 12, SCCL_FOLD_OPS = 5 };
/* Chain all-reduce entries: one per dtype (SCCL_DT_*) and unroll (packs per thread per pass, 1-8). The
 * link pack's kernels (chain all-reduce and link collectives) are built for at most SCCL_LINK_MAX_THREADS
 * threads per block, so each has up to 128 registers per thread and keeps every load of a pass in flight. */
enum { SCCL_CHAIN_MAX_UNROLL = 8, SCCL_LINK_MAX_THREADS = 512, SCCL_LINK_MIN_THREADS = 64 };
/* Link collectives (kernels/sircl_links.cu): the all-gathers move bytes (dtype ignored). Each kind's grid is
 * its roles times blocks_per_role: chain all-gather 4, chain reduce-scatter 2, ring all-gather 2, ring
 * reduce-scatter 2, ring all-reduce 3. */
enum {
  SCCL_LINK_GATHER = 0,
  SCCL_LINK_SCATTER = 1,
  SCCL_RING_GATHER = 2,
  SCCL_RING_SCATTER = 3,
  SCCL_RING_REDUCE = 4,
  SCCL_LINK_KINDS = 5,
  SCCL_LINK_MAX_WORLD = 8,
  SCCL_LINK_PIECE_COUNTERS = 65536,
  SCCL_LINK_DECISION_WORDS = 256,
  SCCL_LINK_COUNTER_WORDS = 16
};

/* Launch arguments of the one-shot and two-shot all-reduce kernels. Addresses are device
 * addresses; the arena's equal its host addresses where the GPU addresses pinned memory at its
 * host pointer. phase_counter is used by the two-shot kernel only. */
typedef struct {
  uint64_t input, output;
  int32_t size_packs, nbytes;
  uint64_t recv_base, flag_base, send_base, ctrl_base, slot_bytes;
  uint64_t epoch, stage_counter, phase_counter, tail_counter, poison, arrival;
  uint32_t spin_limit;
  int32_t rank, lanes, one_block;
} sccl_allreduce_args;

/* One all-gather tile: strides in packs (kernels/sircl_kernels.cu). */
typedef struct {
  uint64_t input, output;
  int32_t shard_packs, nbytes, tile_cols, reserved;
  int64_t in_row_stride, out_row_stride, out_src_stride;
  uint64_t recv_base, flag_base, send_base, ctrl_base, slot_bytes;
  uint64_t epoch, stage_counter, tail_counter, poison, arrival;
  uint32_t spin_limit;
  int32_t rank, lanes, one_block;
} sccl_allgather_args;

/* One scatter op (reduce-scatter or all-to-all): strides in bytes. */
typedef struct {
  uint64_t input, output;
  int32_t size_packs, nbytes, chunk_packs, reserved;
  int64_t src_stride, dst_stride;
  uint64_t recv_base, flag_base, send_base, ctrl_base, slot_bytes;
  uint64_t epoch, stage_counter, tail_counter, poison;
  uint32_t spin_limit;
  int32_t rank, lanes;
} sccl_scatter_args;

/* One chain all-reduce op (kernels/sircl_links.cu): the session's chain area and counters, the chain
 * position and geometry. Launched with 4 * blocks_per_role blocks. */
typedef struct {
  uint64_t input, output;
  int32_t a_packs, b_packs, chunk_packs, reserved;
  uint64_t chain_base, counters, ctrl_base, poison;
  uint32_t spin_limit, trace_capacity;
  uint64_t trace_base;
  int32_t world, index, prev, next, rank, lanes, slots, blocks_per_role;
  uint64_t slot_bytes;
} sccl_chain_args;

/* One link collective op: the session's link area, its 16 link counter words and (chain reduce-scatter)
 * its piece counters (SCCL_LINK_PIECE_COUNTERS words, then SCCL_LINK_DECISION_WORDS decision words), the
 * chain position and rank order (order[i]: the rank at chain index i). chunk_packs is the packs of one
 * rank's block (all-gathers) or chunk (reduce-scatters, ring all-reduce); stride_packs the packs between
 * chunks of the input (reduce-scatters) or of the message (ring all-reduce); piece_packs the packs of one
 * link item. stagger and gather_stagger are the ring staggers of links 2 and 3. Mirrors the device-side
 * LinkParams of kernels/sircl_links.cu (176 bytes). */
typedef struct {
  uint64_t input, output, scratch;
  int32_t chunk_packs, stride_packs, piece_packs, stagger;
  int32_t gather_stagger, world, index, prev, next, rank, lanes, slots, blocks_per_role, reserved;
  uint64_t link_base, counters, piece_counters, ctrl_base, poison, trace_base, slot_bytes;
  uint32_t spin_limit, trace_capacity;
  int32_t order[SCCL_LINK_MAX_WORLD];
} sccl_link_args;
_Static_assert(sizeof(sccl_link_args) == 176, "sccl_link_args must match the link pack's LinkParams");

/* The point-to-point pack (kernels/sircl_p2p.cu): sircl_p2p_send_u<U> and sircl_p2p_recv_u<U>, U 1-8, built
 * for at most SCCL_P2P_MAX_THREADS threads per block. */
enum { SCCL_P2P_MAX_UNROLL = 8, SCCL_P2P_MAX_THREADS = 512, SCCL_P2P_MIN_THREADS = 64 };
/* One point-to-point launch: the message (`data`, a 16-byte-aligned device address; `packs` 16-byte packs;
 * `tail` its bytes modulo 16), the channel's first item and the message's items, the peer's block and the
 * control line of the channel arena (device addresses), the peer, the slot geometry and the block offsets of
 * the native layer's p2p_layout (receive, send, flag, desc, ready, consumed, sent). Mirrors the device-side
 * P2PParams (120 bytes). */
typedef struct {
  uint64_t data, block_base, ctrl_base, slot_bytes;
  int32_t packs, tail, items, peer;
  uint32_t first, slots, lanes, reserved;
  uint64_t recv_off, send_off, flag_off, desc_off, ready_off, consumed_off, sent_off;
} sccl_p2p_args;
_Static_assert(sizeof(sccl_p2p_args) == 120, "sccl_p2p_args must match the point-to-point pack's P2PParams");

/* Load every pack into `ctx` once (pushing it current for the load); later
 * calls for the same context return at once. 0 on success. */
int sccl_kp_load(sccl_CUcontext ctx);
/* The entry of (kind, dtype, world) in a context the pack is loaded in. */
int sccl_kp_function(sccl_CUcontext ctx, int kind, int dtype, int world, sccl_CUfunction *out);
/* One launch of `grid` blocks of `threads` threads on `stream`, from the
 * calling thread's current context. Returns the driver result. */
sccl_CUresult sccl_kp_launch_allreduce(sccl_CUfunction function, int algorithm, const sccl_allreduce_args *args,
                                       unsigned grid, unsigned threads, sccl_CUstream stream);
sccl_CUresult sccl_kp_launch_allgather(sccl_CUfunction function, const sccl_allgather_args *args, unsigned grid,
                                       unsigned threads, sccl_CUstream stream);
sccl_CUresult sccl_kp_launch_scatter(sccl_CUfunction function, const sccl_scatter_args *args, unsigned grid,
                                     unsigned threads, sccl_CUstream stream);
/* The fold entry of an NCCL datatype and built-in op, in a context the packs are loaded in. */
int sccl_kp_fold_function(sccl_CUcontext ctx, int datatype, int op, sccl_CUfunction *out);
/* Fold `world` rows of `count` elements, rows `pitch` bytes apart from `rows`, into `out`. */
sccl_CUresult sccl_kp_launch_fold(sccl_CUfunction function, uint64_t rows, int64_t pitch, uint64_t out, int64_t count,
                                  int32_t world, unsigned grid, unsigned threads, sccl_CUstream stream);
/* The chain all-reduce entry of a dtype and unroll, in a context the packs are loaded in. */
int sccl_kp_chain_function(sccl_CUcontext ctx, int dtype, int unroll, sccl_CUfunction *out);
sccl_CUresult sccl_kp_launch_chain(sccl_CUfunction function, const sccl_chain_args *args, unsigned grid,
                                   unsigned threads, sccl_CUstream stream);
/* The link collective entry of a kind (SCCL_LINK_*, SCCL_RING_*), dtype and unroll (1-8). */
int sccl_kp_link_function(sccl_CUcontext ctx, int kind, int dtype, int unroll, sccl_CUfunction *out);
/* The ring all-reduce whose relay stores its result to link 3's slot, then copies the slot to the output
 * (sircl_ring_reduce_two_pass_<dtype>_u<U>): the same protocol and bytes as SCCL_RING_REDUCE's one-pass
 * entry; LIBSIRCL_RING_REDUCE_PASSES=2 selects it, for comparisons. */
int sccl_kp_ring_reduce_two_pass_function(sccl_CUcontext ctx, int dtype, int unroll, sccl_CUfunction *out);
/* The pair exchange (sircl_ring_exchange_u<U>): on the wire the ring all-gather of a ring of two, carrying
 * one block each way (input to the peer, the peer's block to output + peer * stride, the local block from
 * scratch to output + rank * stride; input 0 sends nothing meaningful; output 0, or flag 1 in `reserved`,
 * discards the peer's block). Launched with sccl_kp_launch_link, 3 * blocks_per_role blocks. */
int sccl_kp_ring_exchange_function(sccl_CUcontext ctx, int unroll, sccl_CUfunction *out);
/* The pair exchange's reduce-to-root form (sircl_ring_exchange_reduce_<dtype>_u<U>): the peer's block is
 * added to this rank's own values (scratch) into output, the dtype rounding of the float32 sum. */
int sccl_kp_ring_exchange_reduce_function(sccl_CUcontext ctx, int dtype, int unroll, sccl_CUfunction *out);
sccl_CUresult sccl_kp_launch_link(sccl_CUfunction function, const sccl_link_args *args, unsigned grid,
                                  unsigned threads, sccl_CUstream stream);
/* The point-to-point entry of a direction (send 1: sircl_p2p_send_u<U>; 0: sircl_p2p_recv_u<U>) and unroll. */
int sccl_kp_p2p_function(sccl_CUcontext ctx, int send, int unroll, sccl_CUfunction *out);
sccl_CUresult sccl_kp_launch_p2p(sccl_CUfunction function, const sccl_p2p_args *args, unsigned grid,
                                 unsigned threads, sccl_CUstream stream);
/* SHA-256 of the embedded transport, fold, link and point-to-point fatbins, hexadecimal; part of the setup
 * agreement. */
const char *sccl_kp_hash(void);
const char *sccl_kp_fold_hash(void);
const char *sccl_kp_links_hash(void);
const char *sccl_kp_p2p_hash(void);
const char *sccl_kp_error(void);

#endif
