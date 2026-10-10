/*
 * The link pack: SIRCL's chain all-reduce and link collectives, ported to CUDA C++ from SIRCL's CuTe DSL
 * kernels (sparkring_sircl/oneshot/_chain_cute.py and _links_cute.py), speaking the same chain and link
 * protocols with SIRCL's native progress thread. SIRCL specializes each kernel on the chain position,
 * neighbors and rank order at compile time; these entries take them as launch parameters, so one compiled
 * entry per dtype and unroll serves every rank.
 *
 * Chain all-reduce. The ranks form a chain of cable neighbors (chain index 0 to W-1; `prev` and `next`
 * are the neighbors' ranks). A message's first a_packs 16-byte packs are half A, the rest half B. Half A
 * reduces along the chain from index 0 to W-1: index 0 sends its own values, every later index adds its
 * own values to the partial it received (float32 addition of the two dtype values, rounded once to the
 * dtype) and sends the result on; index W-1 holds the final values, stores them and sends them back
 * along the chain. Half B runs the same way from W-1 to 0. Every rank stores the end rank's values.
 *
 * Each half travels in chunks of chunk_packs packs through the chain area (ChainLayout of SIRCL's
 * protocol.py: per stream `slots` receive and send slots, flag lines, ready, consumed, sent and credit
 * words, then the control line with the chain doorbell and two parameter slots). Streams: 0 half-A
 * partials toward next, 1 half-A results toward prev, 2 half-B partials toward prev, 3 half-B results
 * toward next. A chunk's tag is its global index on its half plus one. The grid has four roles of
 * `blocks_per_role` blocks (A reduce, B reduce, A results, B results); block `sub` of a role takes
 * chunks sub, sub + blocks_per_role, ...
 *
 * Reduce role: wait for the inbound chunk's lane flags (threads 0 .. lanes-1; skipped at the half's
 * first rank) and for the send slot's previous chunk to have been written downstream (the block's last
 * thread: sent >= tag - slots); add or copy into the send slot (and the output at the half's last rank);
 * publish consumed (inbound) and ready (outbound). Result role: wait for the result chunk, copy it to the
 * output, publish consumed. A wait longer than the wait limit (command ring word 7) records the peer,
 * lane, error kind (1: chain chunk, 2: chain slot) and tag (words 3, 6, 8, 2) and sets the poison word.
 * The device counters (chunks of half A before this op, of half B, the chain sequence, tail arrivals)
 * advance when the last block finishes, unless a wait timed out.
 *
 * Link collectives run over the session's link area (LinkLayout of SIRCL's protocol.py: the chain area's
 * geometry with own slots in place of send slots) and its 16 link counter words; link 0 carries items toward
 * higher chain indices, link 1 toward lower ones, links 2 and 3 toward the next rank of the ring that closes
 * the chain. An item's tag is its index on its link plus one, counted over the session.
 *  - Chain all-gather: rank j stages its own piece on each link it sends on; the progress thread forwards the
 *    pieces of owners j-1 .. 0 (link 0) and j+1 .. W-1 (link 1) straight from the receive slots; the kernel
 *    copies every received piece to its owner's place (rank order). Bytes are copied unchanged.
 *  - Chain reduce-scatter: the partials of every owner's chunk travel toward it from both chain ends, each hop
 *    the dtype rounding of the float32 sum of the received partial and the rank's values; the owner stores
 *    round((L + x) + R).
 *  - Ring all-gather, reduce-scatter and all-reduce (links 2 and 3), with the staggers of the op word
 *    (bits 8-15: link 2, bits 16-23: link 3): owner k's piece is the dtype rounding at every hop of the
 *    values of ring indices k+1, k+2, ..., k added in that order; finished pieces travel on link 3.
 *
 * Inbound slots live in pinned host memory; a pass issues all of its loads together (the result of every
 * load is gated by a word the compiler cannot prove zero, as SIRCL's _cute_batch.py does), so a pass pays
 * one host-memory latency.
 */
#include <stdint.h>

#include "sircl_common.cuh"

namespace sircl_links {

using namespace sircl;

constexpr int kChainStreams = 4;
constexpr int kCtrlErrorKind = 8;
constexpr uint32_t kErrorChainChunk = 1;
constexpr uint32_t kErrorChainSlot = 2;
constexpr int kTraceHeaderBytes = 16;
constexpr int kTraceRecordBytes = 16;
constexpr uint32_t kTraceKernelFlag = 16, kTraceKernelReady = 17, kTraceKernelConsumed = 18;

/* Spin until the counter reaches `target` ((int32)(value - target) >= 0), limits as spin_until_eq_timed_sys. */
__device__ __forceinline__ uint32_t spin_until_ge_timed_sys(uint64_t addr, uint32_t target, uint32_t poll_limit,
                                                            uint32_t limit_us) {
  const bool timed = limit_us != 0;
  const uint64_t budget = (uint64_t)limit_us * 1000u;
  const uint64_t start = globaltimer();
  uint32_t polls = 0;
  for (;;) {
    if ((int32_t)(ld_acquire_sys(addr) - target) >= 0) return 0;
    polls += 1;
    if (!timed) {
      if (polls >= poll_limit) return 1;
    } else if ((polls & (kPollsPerClockCheck - 1)) == 0) {
      if (globaltimer() - start >= budget) return 1;
    }
  }
}

/* Chain area geometry (protocol.ChainLayout). */
struct ChainGeometry {
  uint64_t base;
  int lanes, slots;
  uint64_t slot_bytes;
  __device__ __forceinline__ uint64_t ring_bytes() const { return (uint64_t)kChainStreams * slots * slot_bytes; }
  __device__ __forceinline__ uint64_t rflag_off() const { return 2 * ring_bytes(); }
  __device__ __forceinline__ uint64_t ready_off() const {
    return rflag_off() + (uint64_t)kChainStreams * slots * lanes * kFlagStride;
  }
  __device__ __forceinline__ uint64_t consumed_off() const { return ready_off() + kChainStreams * kFlagStride; }
  __device__ __forceinline__ uint64_t sent_off() const { return consumed_off() + kChainStreams * kFlagStride; }
  __device__ __forceinline__ uint64_t ctrl_off() const {
    return sent_off() + 2 * kChainStreams * kFlagStride;  // sent words, then credit words
  }
  __device__ __forceinline__ uint64_t recv_slot(int stream, uint32_t m) const {
    return base + ((uint64_t)stream * slots + m) * slot_bytes;
  }
  __device__ __forceinline__ uint64_t send_slot(int stream, uint32_t m) const {
    return base + ring_bytes() + ((uint64_t)stream * slots + m) * slot_bytes;
  }
  __device__ __forceinline__ uint64_t flag(int stream, uint32_t m, int lane) const {
    return base + rflag_off() + (((uint64_t)stream * slots + m) * lanes + lane) * kFlagStride;
  }
  __device__ __forceinline__ uint64_t ready_word(int stream, uint32_t m) const {
    return base + ready_off() + (uint64_t)stream * kFlagStride + 4u * m;
  }
  __device__ __forceinline__ uint64_t consumed_word(int stream, uint32_t m) const {
    return base + consumed_off() + (uint64_t)stream * kFlagStride + 4u * m;
  }
  __device__ __forceinline__ uint64_t sent_word(int stream) const {
    return base + sent_off() + (uint64_t)stream * kFlagStride;
  }
};

struct Trace {
  uint64_t base;
  uint32_t capacity;
  __device__ __forceinline__ void record(int tid, uint32_t event, int stream, uint32_t tag) const {
    if (capacity == 0 || tid != 0) return;
    const uint32_t index = atom_add_relaxed_gpu(base, 1u);
    if (index < capacity) {
      const uint64_t t = globaltimer();
      st_global_v4(base + kTraceHeaderBytes + (uint64_t)index * kTraceRecordBytes,
                   make_uint4((uint32_t)t, (uint32_t)(t >> 32), tag, event | ((uint32_t)stream << 16)));
    }
  }
  /* After the lane-flag waits: warp 0 converges, then thread 0 records KERNEL_FLAG. */
  __device__ __forceinline__ void flags(int tid, int stream, uint32_t tag) const {
    if (capacity == 0) return;
    if (tid < 32) __syncwarp();
    record(tid, kTraceKernelFlag, stream, tag);
  }
};

__device__ __forceinline__ void fail(uint64_t ctrl_base, uint64_t poison, uint32_t peer, uint32_t lane,
                                     uint32_t kind, uint32_t tag) {
  st_relaxed_sys(ctrl_base + 4u * kCtrlMissingPeer, peer);
  st_relaxed_sys(ctrl_base + 4u * kCtrlMissingLane, lane);
  st_relaxed_sys(ctrl_base + 4u * kCtrlErrorKind, kind);
  fence_sc_sys();
  st_relaxed_sys(ctrl_base + 4u * kCtrlErrorSeq, tag);
  st_release_gpu(poison, 1u);
}

/* One 16-byte load, at system scope (NIC-written pinned memory) or plain. */
__device__ __forceinline__ uint4 load_pack(uint64_t a, bool system) {
  uint4 r;
  if (system)
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(a));
  else
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(a));
  return r;
}

/* The dtype rounding of float32(partial) + float32(own), per element. */
template <class D>
__device__ __forceinline__ uint4 add_packs(const uint4 &partial, const uint4 &own);
template <>
__device__ __forceinline__ uint4 add_packs<F32>(const uint4 &p, const uint4 &o) {
  return make_uint4(__float_as_uint(__fadd_rn(__uint_as_float(p.x), __uint_as_float(o.x))),
                    __float_as_uint(__fadd_rn(__uint_as_float(p.y), __uint_as_float(o.y))),
                    __float_as_uint(__fadd_rn(__uint_as_float(p.z), __uint_as_float(o.z))),
                    __float_as_uint(__fadd_rn(__uint_as_float(p.w), __uint_as_float(o.w))));
}
template <bool kBf16>
__device__ __forceinline__ uint32_t add_halves(uint32_t p, uint32_t o) {
  float plo, phi, olo, ohi;
  if (kBf16) {
    unpack_bf16x2(p, plo, phi);
    unpack_bf16x2(o, olo, ohi);
    return pack_bf16x2(__fadd_rn(plo, olo), __fadd_rn(phi, ohi));
  }
  unpack_f16x2(p, plo, phi);
  unpack_f16x2(o, olo, ohi);
  return pack_f16x2(__fadd_rn(plo, olo), __fadd_rn(phi, ohi));
}
template <>
__device__ __forceinline__ uint4 add_packs<Half2x4<true>>(const uint4 &p, const uint4 &o) {
  return make_uint4(add_halves<true>(p.x, o.x), add_halves<true>(p.y, o.y), add_halves<true>(p.z, o.z),
                    add_halves<true>(p.w, o.w));
}
template <>
__device__ __forceinline__ uint4 add_packs<Half2x4<false>>(const uint4 &p, const uint4 &o) {
  return make_uint4(add_halves<false>(p.x, o.x), add_halves<false>(p.y, o.y), add_halves<false>(p.z, o.z),
                    add_halves<false>(p.w, o.w));
}

/* Store, for each of the first `packs` packs, inbound + own (HAS_IN and HAS_OWN), own (HAS_OWN only) or
 * inbound (HAS_IN only) to dst and, with to_dst2, to dst2. UNROLL packs per thread per pass, every load of
 * a pass issued together. */
template <class D, int UNROLL, bool HAS_IN, bool HAS_OWN>
__device__ __forceinline__ void chain_pass(int thread, int threads, uint64_t inbound, uint64_t own, uint64_t dst,
                                           uint64_t dst2, bool to_dst2, int32_t packs, uint32_t zero) {
  const int32_t last = packs - 1;
  for (int32_t base = 0; base < packs; base += UNROLL * threads) {
    uint4 in[UNROLL], mine[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int32_t index = base + u * threads + thread;
      index = index < last ? index : last;
      const uint64_t off = (uint64_t)index * kPackBytes;
      if (HAS_IN) in[u] = load_pack(inbound + off, true);
      if (HAS_OWN) mine[u] = load_pack(own + off, false);
    }
    uint32_t gate = 0;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      if (HAS_IN) gate ^= in[u].w;
      if (HAS_OWN) gate ^= mine[u].w;
    }
    gate &= zero;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int32_t index = base + u * threads + thread;
      if (index < packs) {
        uint4 packed;
        if (HAS_IN && HAS_OWN) {
          const uint4 a = make_uint4(in[u].x ^ gate, in[u].y ^ gate, in[u].z ^ gate, in[u].w ^ gate);
          const uint4 b = make_uint4(mine[u].x ^ gate, mine[u].y ^ gate, mine[u].z ^ gate, mine[u].w ^ gate);
          packed = add_packs<D>(a, b);
        } else if (HAS_IN) {
          packed = make_uint4(in[u].x ^ gate, in[u].y ^ gate, in[u].z ^ gate, in[u].w ^ gate);
        } else {
          packed = make_uint4(mine[u].x ^ gate, mine[u].y ^ gate, mine[u].z ^ gate, mine[u].w ^ gate);
        }
        const uint64_t off = (uint64_t)index * kPackBytes;
        st_global_v4(dst + off, packed);
        if (to_dst2) st_global_v4(dst2 + off, packed);
      }
    }
  }
}

struct ChainArgs {
  uint64_t input, output;
  int32_t a_packs, b_packs, chunk_packs;
  uint64_t counters, ctrl_base, poison;
  uint32_t spin_limit, limit_us;
  int32_t world, index, prev, next, rank, blocks_per_role;
  ChainGeometry chain;
  Trace trace;
};

template <class D, int UNROLL>
__device__ __forceinline__ void chain_reduce(const ChainArgs &a, int half, int sub, int32_t half_packs,
                                             int32_t half_offset, uint32_t base) {
  const int thread = (int)threadIdx.x, threads = (int)blockDim.x;
  const int position = half == 0 ? a.index : a.world - 1 - a.index;
  const bool has_in = position > 0, last = position == a.world - 1;
  const int in_stream = half == 0 ? 0 : 2;
  const int out_stream = last ? (half == 0 ? 1 : 3) : in_stream;
  const int32_t in_peer = half == 0 ? a.prev : a.next;
  const int32_t n_chunks = (half_packs + a.chunk_packs - 1) / a.chunk_packs;
  const uint32_t zero = (uint32_t)a.chunk_packs >> 31;
  for (int32_t j = sub; j < n_chunks; j += a.blocks_per_role) {
    const uint32_t g = base + (uint32_t)j;
    const uint32_t m = g % (uint32_t)a.chain.slots;
    const uint32_t tag = g + 1u;
    if (has_in) {
      if (thread < a.chain.lanes &&
          spin_until_eq_timed_sys(a.chain.flag(in_stream, m, thread), tag, a.spin_limit, a.limit_us))
        fail(a.ctrl_base, a.poison, (uint32_t)in_peer, (uint32_t)thread, kErrorChainChunk, tag);
      a.trace.flags(thread, in_stream, tag);
    }
    if (thread == threads - 1 &&
        spin_until_ge_timed_sys(a.chain.sent_word(out_stream), tag - (uint32_t)a.chain.slots, a.spin_limit,
                                a.limit_us))
      fail(a.ctrl_base, a.poison, (uint32_t)a.rank, 255u, kErrorChainSlot, tag);
    __syncthreads();
    if (ld_relaxed_gpu(a.poison) != 0) break;
    const int32_t remaining = half_packs - j * a.chunk_packs;
    const int32_t count = remaining < a.chunk_packs ? remaining : a.chunk_packs;
    const int32_t first = half_offset + j * a.chunk_packs;
    const uint64_t inbound = a.chain.recv_slot(in_stream, m), outbound = a.chain.send_slot(out_stream, m);
    const uint64_t own = a.input + (uint64_t)first * kPackBytes, out = a.output + (uint64_t)first * kPackBytes;
    if (has_in)
      chain_pass<D, UNROLL, true, true>(thread, threads, inbound, own, outbound, out, last, count, zero);
    else
      chain_pass<D, UNROLL, false, true>(thread, threads, inbound, own, outbound, out, last, count, zero);
    __syncthreads();
    if (thread == 0) {
      fence_sc_sys();
      if (has_in) st_relaxed_sys(a.chain.consumed_word(in_stream, m), tag);
      st_relaxed_sys(a.chain.ready_word(out_stream, m), tag);
    }
    if (has_in) a.trace.record(thread, kTraceKernelConsumed, in_stream, tag);
    a.trace.record(thread, kTraceKernelReady, out_stream, tag);
  }
}

template <class D, int UNROLL>
__device__ __forceinline__ void chain_results(const ChainArgs &a, int half, int sub, int32_t half_packs,
                                              int32_t half_offset, uint32_t base) {
  const int thread = (int)threadIdx.x, threads = (int)blockDim.x;
  const int stream = half == 0 ? 1 : 3;
  const int32_t peer = half == 0 ? a.next : a.prev;
  const int32_t n_chunks = (half_packs + a.chunk_packs - 1) / a.chunk_packs;
  const uint32_t zero = (uint32_t)a.chunk_packs >> 31;
  for (int32_t j = sub; j < n_chunks; j += a.blocks_per_role) {
    const uint32_t g = base + (uint32_t)j;
    const uint32_t m = g % (uint32_t)a.chain.slots;
    const uint32_t tag = g + 1u;
    if (thread < a.chain.lanes &&
        spin_until_eq_timed_sys(a.chain.flag(stream, m, thread), tag, a.spin_limit, a.limit_us))
      fail(a.ctrl_base, a.poison, (uint32_t)peer, (uint32_t)thread, kErrorChainChunk, tag);
    a.trace.flags(thread, stream, tag);
    __syncthreads();
    if (ld_relaxed_gpu(a.poison) != 0) break;
    const int32_t remaining = half_packs - j * a.chunk_packs;
    const int32_t count = remaining < a.chunk_packs ? remaining : a.chunk_packs;
    const int32_t first = half_offset + j * a.chunk_packs;
    chain_pass<D, UNROLL, true, false>(thread, threads, a.chain.recv_slot(stream, m), 0,
                                       a.output + (uint64_t)first * kPackBytes, 0, false, count, zero);
    __syncthreads();
    if (thread == 0) {
      fence_sc_sys();
      st_relaxed_sys(a.chain.consumed_word(stream, m), tag);
    }
    a.trace.record(thread, kTraceKernelConsumed, stream, tag);
  }
}

template <class D, int UNROLL>
__device__ __forceinline__ void chain_all_reduce(ChainArgs a) {
  const int tid = (int)threadIdx.x, bid = (int)blockIdx.x, gdim = (int)gridDim.x;
  const int role = bid / a.blocks_per_role, sub = bid - role * a.blocks_per_role;
  if (ld_relaxed_gpu(a.poison) != 0) return;
  /* Every block reads the counters before any block can advance them: the advance happens only after
   * every block arrived at the tail. */
  const uint32_t base_a = ld_relaxed_gpu(a.counters), base_b = ld_relaxed_gpu(a.counters + 4);
  const uint32_t seq = ld_relaxed_gpu(a.counters + 8) + 1u;
  if (bid == 0 && tid == 0) {
    const uint64_t params = a.chain.base + a.chain.ctrl_off() + 4u + (uint64_t)(seq & 1u) * 16u;
    st_relaxed_sys(params, (uint32_t)(a.a_packs * kPackBytes));
    st_relaxed_sys(params + 4, (uint32_t)(a.b_packs * kPackBytes));
    st_relaxed_sys(params + 8, (uint32_t)(a.chunk_packs * kPackBytes));
    fence_sc_sys();
    st_relaxed_sys(a.chain.base + a.chain.ctrl_off(), seq);
  }
  a.limit_us = ld_relaxed_sys(a.ctrl_base + 4u * kCtrlWaitLimitUs);
  if (role == 0) chain_reduce<D, UNROLL>(a, 0, sub, a.a_packs, 0, base_a);
  if (role == 1) chain_reduce<D, UNROLL>(a, 1, sub, a.b_packs, a.a_packs, base_b);
  if (role == 2 && a.index < a.world - 1) chain_results<D, UNROLL>(a, 0, sub, a.a_packs, 0, base_a);
  if (role == 3 && a.index > 0) chain_results<D, UNROLL>(a, 1, sub, a.b_packs, a.a_packs, base_b);
  /* The last block to finish advances the chunk bases and the sequence. It is the block whose arrival
   * brings the tail word to the grid; it returns the word to 0 for the next launch, which stream order
   * starts after this one ends, so launches with different grids never mistake an earlier block for the
   * last one. */
  fence_sc_gpu();
  __syncthreads();
  if (tid == 0) {
    const uint32_t prior = atom_add_relaxed_gpu(a.counters + 12, 1u);
    if (prior + 1u == (uint32_t)gdim) {
      st_release_gpu(a.counters + 12, 0u);
      fence_sc_gpu();
      if (ld_relaxed_sys(a.ctrl_base + 4u * kCtrlErrorSeq) == 0) {
        const int32_t n_a = (a.a_packs + a.chunk_packs - 1) / a.chunk_packs;
        const int32_t n_b = (a.b_packs + a.chunk_packs - 1) / a.chunk_packs;
        st_release_gpu(a.counters, base_a + (uint32_t)n_a);
        st_release_gpu(a.counters + 4, base_b + (uint32_t)n_b);
        st_release_gpu(a.counters + 8, seq);
      }
    }
  }
}

/* -- link collectives (SIRCL's _links_cute.py) ---------------------------------------------------------- */

constexpr int kTraceLinkStream = 4;
constexpr uint32_t kTraceKernelSlot = 19;
constexpr int kPieceCounters = 65536;
constexpr int kLinkMaxWorld = 8;
constexpr uint32_t kOpAllGather = 1, kOpReduceScatter = 2, kOpRingGather = 3, kOpRingScatter = 4, kOpRingReduce = 5;
constexpr int kRingStaggerShift = 8, kRingGatherStaggerShift = 16;
/* Link counter words (16 per session, shared by every link collective): own and inbound items of links 0
 * and 1 (words 0-3), the link op sequence (4), tail arrivals of the chain all-gather (5), the chain
 * reduce-scatter (6) and the ring reduce-scatter (7), own and inbound items of links 2 and 3 (8-11), tail
 * arrivals of the ring all-gather (12) and all-reduce (13). */
constexpr int kWordSeq = 4, kWordGatherTail = 5, kWordScatterTail = 6, kWordRingScatterTail = 7;
constexpr int kWordOwn2 = 8, kWordIn2 = 9, kWordOwn3 = 10, kWordIn3 = 11;
constexpr int kWordRingGatherTail = 12, kWordRingReduceTail = 13;
enum RingMode { kRingGather = 0, kRingScatter = 1, kRingReduce = 2 };

/* Launch parameters of every link collective (one struct, passed by value; src/kernelpack.h mirrors it as
 * sccl_link_args). The all-gathers take chunk_packs as the packs of one rank's block; stride_packs is the
 * distance between chunks of the input (reduce-scatters) or of the message (the ring all-reduce). */
struct LinkParams {
  uint64_t input, output, scratch;
  int32_t chunk_packs, stride_packs, piece_packs, stagger;
  int32_t gather_stagger, world, index, prev, next, rank, lanes, slots, blocks_per_role, reserved;
  uint64_t link_base, counters, piece_counters, ctrl_base, poison, trace_base, slot_bytes;
  uint32_t spin_limit, trace_capacity;
  int32_t order[kLinkMaxWorld];
};
static_assert(sizeof(LinkParams) == 176, "LinkParams must match sccl_link_args");

/* The link area has the chain area's geometry (protocol.LinkLayout: own slots where the chain area has
 * send slots), so ChainGeometry addresses it; its streams are the four links. */
struct LinkCtx {
  const LinkParams *p;
  ChainGeometry g;
  Trace trace;
  uint32_t limit_us, zero;
  int thread, threads;
  __device__ __forceinline__ int32_t pieces() const { return (p->chunk_packs + p->piece_packs - 1) / p->piece_packs; }
  __device__ __forceinline__ int32_t piece_count(int32_t first) const {
    const int32_t rest = p->chunk_packs - first;
    return rest < p->piece_packs ? rest : p->piece_packs;
  }
  /* The rank at chain (ring) index index + offset, modulo the world. */
  __device__ __forceinline__ int32_t place(int offset) const {
    int k = (p->index + offset) % p->world;
    if (k < 0) k += p->world;
    return p->order[k];
  }
  __device__ __forceinline__ uint64_t pack(uint64_t base, int64_t packs) const {
    return base + (uint64_t)packs * kPackBytes;
  }
  /* The block's last thread waits until the own slot of `item` on `link` was written downstream. */
  __device__ __forceinline__ void wait_own_slot(int link, uint32_t item) const {
    const uint32_t tag = item + 1u;
    if (thread == threads - 1 &&
        spin_until_ge_timed_sys(g.sent_word(link), tag - (uint32_t)g.slots, p->spin_limit, limit_us))
      fail(p->ctrl_base, p->poison, (uint32_t)p->rank, 255u, kErrorChainSlot, tag);
  }
  /* Threads 0 .. lanes-1 (warp 0) wait for every lane flag of inbound item `item` of `link` from `peer`. */
  __device__ __forceinline__ void wait_inbound(int link, uint32_t item, int32_t peer) const {
    const uint32_t tag = item + 1u;
    if (thread < g.lanes &&
        spin_until_eq_timed_sys(g.flag(link, item % (uint32_t)g.slots, thread), tag, p->spin_limit, limit_us))
      fail(p->ctrl_base, p->poison, (uint32_t)peer, (uint32_t)thread, kErrorChainChunk, tag);
  }
  /* Thread 0 publishes `item` in the ready (own slots) or consumed (receive slots) words of `link`. */
  __device__ __forceinline__ void publish_ready(int link, uint32_t item) const {
    if (thread == 0) {
      fence_sc_sys();
      st_relaxed_sys(g.ready_word(link, item % (uint32_t)g.slots), item + 1u);
    }
  }
  __device__ __forceinline__ void publish_consumed(int link, uint32_t item) const {
    if (thread == 0) {
      fence_sc_sys();
      st_relaxed_sys(g.consumed_word(link, item % (uint32_t)g.slots), item + 1u);
    }
  }
  __device__ __forceinline__ bool poisoned() const { return ld_relaxed_gpu(p->poison) != 0; }
};

/* The dtype rounding of the float32 sum of three packs, added in order: round((a + b) + c). */
template <class D>
__device__ __forceinline__ uint4 add3_packs(const uint4 &a, const uint4 &b, const uint4 &c);
template <>
__device__ __forceinline__ uint4 add3_packs<F32>(const uint4 &a, const uint4 &b, const uint4 &c) {
  return make_uint4(
      __float_as_uint(__fadd_rn(__fadd_rn(__uint_as_float(a.x), __uint_as_float(b.x)), __uint_as_float(c.x))),
      __float_as_uint(__fadd_rn(__fadd_rn(__uint_as_float(a.y), __uint_as_float(b.y)), __uint_as_float(c.y))),
      __float_as_uint(__fadd_rn(__fadd_rn(__uint_as_float(a.z), __uint_as_float(b.z)), __uint_as_float(c.z))),
      __float_as_uint(__fadd_rn(__fadd_rn(__uint_as_float(a.w), __uint_as_float(b.w)), __uint_as_float(c.w))));
}
template <bool kBf16>
__device__ __forceinline__ uint32_t add3_halves(uint32_t a, uint32_t b, uint32_t c) {
  float alo, ahi, blo, bhi, clo, chi;
  if (kBf16) {
    unpack_bf16x2(a, alo, ahi);
    unpack_bf16x2(b, blo, bhi);
    unpack_bf16x2(c, clo, chi);
    return pack_bf16x2(__fadd_rn(__fadd_rn(alo, blo), clo), __fadd_rn(__fadd_rn(ahi, bhi), chi));
  }
  unpack_f16x2(a, alo, ahi);
  unpack_f16x2(b, blo, bhi);
  unpack_f16x2(c, clo, chi);
  return pack_f16x2(__fadd_rn(__fadd_rn(alo, blo), clo), __fadd_rn(__fadd_rn(ahi, bhi), chi));
}
template <>
__device__ __forceinline__ uint4 add3_packs<Half2x4<true>>(const uint4 &a, const uint4 &b, const uint4 &c) {
  return make_uint4(add3_halves<true>(a.x, b.x, c.x), add3_halves<true>(a.y, b.y, c.y),
                    add3_halves<true>(a.z, b.z, c.z), add3_halves<true>(a.w, b.w, c.w));
}
template <>
__device__ __forceinline__ uint4 add3_packs<Half2x4<false>>(const uint4 &a, const uint4 &b, const uint4 &c) {
  return make_uint4(add3_halves<false>(a.x, b.x, c.x), add3_halves<false>(a.y, b.y, c.y),
                    add3_halves<false>(a.z, b.z, c.z), add3_halves<false>(a.w, b.w, c.w));
}

__device__ __forceinline__ uint4 gated(const uint4 &v, uint32_t gate) {
  return make_uint4(v.x ^ gate, v.y ^ gate, v.z ^ gate, v.w ^ gate);
}

/* dst (and dst2 when `two`) = for each of the first `packs` packs the copy of source 0 (COUNT 1) or the
 * dtype rounding of the float32 sum of the COUNT sources in source order. Yk reads source k at system
 * scope (pinned host memory, or another block's writes). UNROLL packs per thread per pass, every load of a
 * pass issued together. */
template <class D, int UNROLL, int COUNT, bool Y0, bool Y1, bool Y2>
__device__ __forceinline__ void link_pass(int thread, int threads, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t dst,
                                          uint64_t dst2, bool two, int32_t packs, uint32_t zero) {
  const int32_t last = packs - 1;
  for (int32_t base = 0; base < packs; base += UNROLL * threads) {
    uint4 w0[UNROLL], w1[UNROLL], w2[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int32_t index = base + u * threads + thread;
      index = index < last ? index : last;
      const uint64_t off = (uint64_t)index * kPackBytes;
      w0[u] = load_pack(s0 + off, Y0);
      if (COUNT > 1) w1[u] = load_pack(s1 + off, Y1);
      if (COUNT > 2) w2[u] = load_pack(s2 + off, Y2);
    }
    uint32_t gate = 0;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      gate ^= w0[u].w;
      if (COUNT > 1) gate ^= w1[u].w;
      if (COUNT > 2) gate ^= w2[u].w;
    }
    gate &= zero;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int32_t index = base + u * threads + thread;
      if (index < packs) {
        uint4 packed;
        if (COUNT == 1)
          packed = gated(w0[u], gate);
        else if (COUNT == 2)
          packed = add_packs<D>(gated(w0[u], gate), gated(w1[u], gate));
        else
          packed = add3_packs<D>(gated(w0[u], gate), gated(w1[u], gate), gated(w2[u], gate));
        const uint64_t off = (uint64_t)index * kPackBytes;
        st_global_v4(dst + off, packed);
        if (two) st_global_v4(dst2 + off, packed);
      }
    }
  }
}

__device__ __forceinline__ LinkCtx link_context(const LinkParams &p) {
  LinkCtx c;
  c.p = &p;
  c.g.base = p.link_base;
  c.g.lanes = p.lanes;
  c.g.slots = p.slots;
  c.g.slot_bytes = p.slot_bytes;
  c.trace.base = p.trace_base;
  c.trace.capacity = p.trace_capacity;
  c.limit_us = 0;
  c.zero = (uint32_t)p.chunk_packs >> 31;
  c.thread = (int)threadIdx.x;
  c.threads = (int)blockDim.x;
  return c;
}

/* Block 0's thread 0 posts the link op (parameter slot seq & 1: op word, bytes per chunk, piece bytes) and
 * rings the link doorbell with the op's sequence. */
__device__ __forceinline__ void post_link_op(const LinkCtx &c, uint32_t seq, uint32_t op_word) {
  if (blockIdx.x == 0 && c.thread == 0) {
    const uint64_t params = c.g.base + c.g.ctrl_off() + 4u + (uint64_t)(seq & 1u) * 16u;
    st_relaxed_sys(params, op_word);
    st_relaxed_sys(params + 4, (uint32_t)(c.p->chunk_packs * kPackBytes));
    st_relaxed_sys(params + 8, (uint32_t)(c.p->piece_packs * kPackBytes));
    fence_sc_sys();
    st_relaxed_sys(c.g.base + c.g.ctrl_off(), seq);
  }
}

/* The block's tail arrival on counter word `tail`; true in thread 0 of the op's last block when no wait
 * of the session timed out (that thread then advances the item bases and the sequence). The last block is
 * the one whose arrival brings the word to the grid; it returns the word to 0 for the next launch of the
 * kernel type, which stream order starts after this one ends, so launches with different grids (blocks
 * per role chosen per op) never mistake an earlier block for the last one. */
__device__ __forceinline__ bool last_block(const LinkCtx &c, int tail) {
  fence_sc_gpu();
  __syncthreads();
  if (c.thread != 0) return false;
  const uint32_t prior = atom_add_relaxed_gpu(c.p->counters + 4u * tail, 1u);
  if (prior + 1u != gridDim.x) return false;
  st_release_gpu(c.p->counters + 4u * tail, 0u);
  fence_sc_gpu();
  return ld_relaxed_sys(c.p->ctrl_base + 4u * kCtrlErrorSeq) == 0;
}

/* Chain all-gather, own role of `link`: stage every own piece in the link's own slots (and copy it to the
 * output's place of this rank). */
template <int UNROLL>
__device__ __forceinline__ void gather_own(const LinkCtx &c, int link, bool to_output, int sub, uint32_t base) {
  const LinkParams &p = *c.p;
  const int32_t pieces = c.pieces(), own_pos = p.order[p.index];
  for (int32_t q = sub; q < pieces; q += p.blocks_per_role) {
    const uint32_t g = base + (uint32_t)q;
    c.wait_own_slot(link, g);
    __syncthreads();
    if (c.poisoned()) break;
    const int32_t first = q * p.piece_packs, count = c.piece_count(first);
    link_pass<F32, UNROLL, 1, false, false, false>(
        c.thread, c.threads, c.pack(p.input, first), 0, 0, c.g.send_slot(link, g % (uint32_t)c.g.slots),
        c.pack(p.output, (int64_t)own_pos * p.chunk_packs + first), to_output, count, c.zero);
    __syncthreads();
    c.publish_ready(link, g);
  }
}

/* Chain all-gather, inbound role of `link`: copy every received piece to its owner's place. Link 0 brings
 * the owners at chain indices index-1 down to 0 in each round, link 1 those at index+1 up to W-1. */
template <int UNROLL>
__device__ __forceinline__ void gather_inbound(const LinkCtx &c, int link, int sub, uint32_t base) {
  const LinkParams &p = *c.p;
  const int32_t per_round = link == 0 ? p.index : p.world - 1 - p.index;
  const int32_t peer = link == 0 ? p.prev : p.next;
  const int32_t items = c.pieces() * per_round;
  for (int32_t i = sub; i < items; i += p.blocks_per_role) {
    const uint32_t g = base + (uint32_t)i;
    c.wait_inbound(link, g, peer);
    __syncthreads();
    if (c.poisoned()) break;
    const int32_t q = i / per_round, r = i - q * per_round;
    const int32_t first = q * p.piece_packs, count = c.piece_count(first);
    const int32_t owner = link == 0 ? p.index - 1 - r : p.index + 1 + r;
    const uint64_t out = c.pack(p.output, (int64_t)p.order[owner] * p.chunk_packs + first);
    link_pass<F32, UNROLL, 1, true, false, false>(c.thread, c.threads, c.g.recv_slot(link, g % (uint32_t)c.g.slots),
                                                  0, 0, out, 0, false, count, c.zero);
    __syncthreads();
    c.publish_consumed(link, g);
  }
}

/* Chain all-gather: four roles of blocks_per_role blocks (own link 0, own link 1, inbound link 0, inbound
 * link 1). The output holds the block of chain index i at rank order[i]'s place. */
template <int UNROLL>
__device__ __forceinline__ void chain_gather(const LinkParams &p) {
  LinkCtx c = link_context(p);
  const int bid = (int)blockIdx.x, role = bid / p.blocks_per_role, sub = bid - role * p.blocks_per_role;
  const bool sends_next = p.index < p.world - 1, sends_prev = p.index > 0;
  if (c.poisoned()) return;
  /* Every block reads the counters before any block can advance them: the advance happens only after
   * every block arrived at the tail. */
  const uint32_t own0 = ld_relaxed_gpu(p.counters), own1 = ld_relaxed_gpu(p.counters + 4);
  const uint32_t in0 = ld_relaxed_gpu(p.counters + 8), in1 = ld_relaxed_gpu(p.counters + 12);
  const uint32_t seq = ld_relaxed_gpu(p.counters + 4u * kWordSeq) + 1u;
  post_link_op(c, seq, kOpAllGather);
  c.limit_us = ld_relaxed_sys(p.ctrl_base + 4u * kCtrlWaitLimitUs);
  if (sends_next && role == 0) gather_own<UNROLL>(c, 0, true, sub, own0);
  if (sends_prev && role == 1) gather_own<UNROLL>(c, 1, !sends_next, sub, own1);
  if (sends_prev && role == 2) gather_inbound<UNROLL>(c, 0, sub, in0);
  if (sends_next && role == 3) gather_inbound<UNROLL>(c, 1, sub, in1);
  if (last_block(c, kWordGatherTail)) {
    const uint32_t pieces = (uint32_t)c.pieces();
    if (sends_next) st_release_gpu(p.counters, own0 + pieces);
    if (sends_prev) st_release_gpu(p.counters + 4, own1 + pieces);
    st_release_gpu(p.counters + 8, in0 + pieces * (uint32_t)p.index);
    st_release_gpu(p.counters + 12, in1 + pieces * (uint32_t)(p.world - 1 - p.index));
    st_release_gpu(p.counters + 4u * kWordSeq, seq);
  }
}

/* Chain reduce-scatter, the first rank of `link` (chain index 0 for link 0, W-1 for link 1): stage its own
 * values of every other owner's piece as the partials, farthest owner first. */
template <class D, int UNROLL>
__device__ __forceinline__ void scatter_own_partials(const LinkCtx &c, int link, int sub, uint32_t base_own) {
  const LinkParams &p = *c.p;
  const int32_t per_round = p.world - 1, items = c.pieces() * per_round;
  for (int32_t q = sub; q < items; q += p.blocks_per_role) {
    const uint32_t g = base_own + (uint32_t)q;
    c.wait_own_slot(link, g);
    __syncthreads();
    if (c.poisoned()) break;
    const int32_t piece = q / per_round, r = q - piece * per_round;
    const int32_t first = piece * p.piece_packs, count = c.piece_count(first);
    const int32_t owner = link == 0 ? p.world - 1 - r : r;
    const uint64_t src = c.pack(p.input, (int64_t)p.order[owner] * p.stride_packs + first);
    link_pass<D, UNROLL, 1, false, false, false>(c.thread, c.threads, src, 0, 0,
                                                 c.g.send_slot(link, g % (uint32_t)c.g.slots), 0, false, count, c.zero);
    __syncthreads();
    c.publish_ready(link, g);
  }
}

/* Chain reduce-scatter, inbound role of `link`: add the own values to every inbound partial and stage it for
 * the next rank, or keep the partial of this rank's own chunk (the last item of each round). An end of the
 * chain completes its chunk from the one partial (L + x, or x + R); a middle rank copies L to the output and
 * R to the scratch buffer, and the block that arrives second at the piece (the piece counter's parity)
 * stores round((L + x) + R). */
template <class D, int UNROLL>
__device__ __forceinline__ void scatter_inbound(const LinkCtx &c, int link, int sub, uint32_t base_in,
                                                uint32_t base_own) {
  const LinkParams &p = *c.p;
  const int32_t j = p.index;
  const int32_t per_round = link == 0 ? p.world - j : j + 1, out_round = per_round - 1;
  const bool both_sides = 0 < j && j < p.world - 1;
  const int32_t peer = link == 0 ? p.prev : p.next;
  const int32_t items = c.pieces() * per_round, own_pos = p.order[j];
  const uint64_t decisions = p.piece_counters + 4u * (uint64_t)kPieceCounters;
  for (int32_t i = sub; i < items; i += p.blocks_per_role) {
    const uint32_t g = base_in + (uint32_t)i;
    const int32_t piece = i / per_round, r = i - piece * per_round;
    const bool forward = r < out_round;
    const uint32_t g_own = base_own + (uint32_t)(piece * out_round + r);
    c.wait_inbound(link, g, peer);
    if (forward) c.wait_own_slot(link, g_own);
    __syncthreads();
    if (c.poisoned()) break;
    const int32_t first = piece * p.piece_packs, count = c.piece_count(first);
    const uint64_t slot = c.g.recv_slot(link, g % (uint32_t)c.g.slots);
    const uint64_t piece_out = c.pack(p.output, first);
    const uint64_t own_values = c.pack(p.input, (int64_t)own_pos * p.stride_packs + first);
    if (forward) {
      const int32_t owner = link == 0 ? p.world - 1 - r : r;
      const uint64_t src = c.pack(p.input, (int64_t)p.order[owner] * p.stride_packs + first);
      link_pass<D, UNROLL, 2, true, false, false>(c.thread, c.threads, slot, src, 0,
                                                  c.g.send_slot(link, g_own % (uint32_t)c.g.slots), 0, false, count,
                                                  c.zero);
      __syncthreads();
      if (c.thread == 0) {
        fence_sc_sys();
        st_relaxed_sys(c.g.ready_word(link, g_own % (uint32_t)c.g.slots), g_own + 1u);
        st_relaxed_sys(c.g.consumed_word(link, g % (uint32_t)c.g.slots), g + 1u);
      }
    } else if (!both_sides) {
      if (link == 0)
        link_pass<D, UNROLL, 2, true, false, false>(c.thread, c.threads, slot, own_values, 0, piece_out, 0, false,
                                                    count, c.zero);
      else
        link_pass<D, UNROLL, 2, false, true, false>(c.thread, c.threads, own_values, slot, 0, piece_out, 0, false,
                                                    count, c.zero);
      __syncthreads();
      c.publish_consumed(link, g);
    } else {
      const uint64_t piece_scratch = c.pack(p.scratch, first);
      link_pass<D, UNROLL, 1, true, false, false>(c.thread, c.threads, slot, 0, 0,
                                                  link == 0 ? piece_out : piece_scratch, 0, false, count, c.zero);
      __syncthreads();
      const uint64_t decision = decisions + 4u * (uint64_t)blockIdx.x;
      if (c.thread == 0) {
        /* The kept copy is complete before the slot is released and before the other link's block can see
         * this arrival. */
        fence_sc_sys();
        st_relaxed_sys(c.g.consumed_word(link, g % (uint32_t)c.g.slots), g + 1u);
        const uint32_t prior = atom_add_relaxed_gpu(p.piece_counters + 4u * (uint64_t)piece, 1u);
        fence_sc_gpu();
        st_release_gpu(decision, prior & 1u);
      }
      __syncthreads();
      if (ld_relaxed_gpu(decision) == 1u) {
        fence_sc_gpu();
        link_pass<D, UNROLL, 3, true, false, true>(c.thread, c.threads, piece_out, own_values, piece_scratch,
                                                   piece_out, 0, false, count, c.zero);
      }
      __syncthreads();
    }
  }
}

/* Chain reduce-scatter: two roles of blocks_per_role blocks, one per link. */
template <class D, int UNROLL>
__device__ __forceinline__ void chain_scatter(const LinkParams &p) {
  LinkCtx c = link_context(p);
  const int bid = (int)blockIdx.x, role = bid / p.blocks_per_role, sub = bid - role * p.blocks_per_role;
  const int32_t j = p.index, last = p.world - 1;
  if (c.poisoned()) return;
  const uint32_t own0 = ld_relaxed_gpu(p.counters), own1 = ld_relaxed_gpu(p.counters + 4);
  const uint32_t in0 = ld_relaxed_gpu(p.counters + 8), in1 = ld_relaxed_gpu(p.counters + 12);
  const uint32_t seq = ld_relaxed_gpu(p.counters + 4u * kWordSeq) + 1u;
  post_link_op(c, seq, kOpReduceScatter);
  c.limit_us = ld_relaxed_sys(p.ctrl_base + 4u * kCtrlWaitLimitUs);
  if (role == 0) {
    if (j == 0)
      scatter_own_partials<D, UNROLL>(c, 0, sub, own0);
    else
      scatter_inbound<D, UNROLL>(c, 0, sub, in0, own0);
  }
  if (role == 1) {
    if (j == last)
      scatter_own_partials<D, UNROLL>(c, 1, sub, own1);
    else
      scatter_inbound<D, UNROLL>(c, 1, sub, in1, own1);
  }
  if (last_block(c, kWordScatterTail)) {
    const uint32_t pieces = (uint32_t)c.pieces();
    if (j < last) {
      st_release_gpu(p.counters, own0 + pieces * (uint32_t)(last - j));
      st_release_gpu(p.counters + 12, in1 + pieces * (uint32_t)(j + 1));
    }
    if (j > 0) {
      st_release_gpu(p.counters + 4, own1 + pieces * (uint32_t)j);
      st_release_gpu(p.counters + 8, in0 + pieces * (uint32_t)(p.world - j));
    }
    st_release_gpu(p.counters + 4u * kWordSeq, seq);
  }
}

/* Ring, start role (link 2, outbound item 0 of every round): this rank's values of the piece of the owner
 * at ring index i-1. */
template <class D, int UNROLL>
__device__ __forceinline__ void ring_start(const LinkCtx &c, int sub, uint32_t own2) {
  const LinkParams &p = *c.p;
  const int32_t pieces = c.pieces(), per_round = p.world - 1, place = c.place(-1);
  for (int32_t q = sub; q < pieces; q += p.blocks_per_role) {
    const uint32_t item = own2 + (uint32_t)(q * per_round);
    c.wait_own_slot(2, item);
    __syncthreads();
    if (c.poisoned()) break;
    c.trace.record(c.thread, kTraceKernelSlot, kTraceLinkStream + 2, item + 1u);
    const int32_t first = q * p.piece_packs, count = c.piece_count(first);
    link_pass<D, UNROLL, 1, false, false, false>(c.thread, c.threads,
                                                 c.pack(p.input, (int64_t)place * p.stride_packs + first), 0, 0,
                                                 c.g.send_slot(2, item % (uint32_t)c.g.slots), 0, false, count, c.zero);
    __syncthreads();
    c.publish_ready(2, item);
    c.trace.record(c.thread, kTraceKernelReady, kTraceLinkStream + 2, item + 1u);
  }
}

/* Ring, relay role (link 2 inbound): inbound type r of round t carries piece t - r * stagger; add this
 * rank's values and pass the partial on (it leaves `stagger` rounds later as outbound type r + 1), or, for
 * the round's last type, complete this rank's own piece (the reduce-scatter's output; the all-reduce's
 * result, also staged as link 3's own item). Items outside the pieces carry flags only. The all-reduce
 * stores each pack of its result to link 3's own slot and to the output in one pass; TWO_PASS stores the
 * slot, then copies the slot to the output (the same bytes, one more read of the slot). */
template <class D, int UNROLL, int MODE, bool TWO_PASS>
__device__ __forceinline__ void ring_relay(const LinkCtx &c, int sub, uint32_t own2, uint32_t in2, uint32_t own3,
                                           uint32_t in3) {
  const LinkParams &p = *c.p;
  const int32_t per_round = p.world - 1, pieces = c.pieces(), stagger = p.stagger;
  const int32_t items = (pieces + stagger * (per_round - 1)) * per_round;
  const int32_t own_place = c.place(0), peer = c.place(-1);
  for (int32_t i = sub; i < items; i += p.blocks_per_role) {
    const uint32_t g = in2 + (uint32_t)i;
    const int32_t t = i / per_round, r = i - t * per_round;
    const int32_t piece = t - r * stagger;
    const bool full = (uint32_t)piece < (uint32_t)pieces;
    const bool last = r == per_round - 1;
    const uint32_t out_item = own2 + (uint32_t)(i + stagger * per_round + 1);
    c.wait_inbound(2, g, peer);
    c.trace.flags(c.thread, kTraceLinkStream + 2, g + 1u);
    if (full) {
      if (last) {
        if (MODE == kRingReduce) c.wait_own_slot(3, own3 + (uint32_t)piece);
      } else {
        c.wait_own_slot(2, out_item);
      }
    }
    __syncthreads();
    if (c.poisoned()) break;
    if (full) {
      /* The link-3 outbound index of this rank's own item of the piece (link 3's outbound and inbound
       * items advance together, so in3 is its first outbound item of the op). */
      const uint32_t tag3 = in3 + (uint32_t)(piece * per_round) + 1u;
      if (last) {
        if (MODE == kRingReduce) c.trace.record(c.thread, kTraceKernelSlot, kTraceLinkStream + 3, tag3);
      } else {
        c.trace.record(c.thread, kTraceKernelSlot, kTraceLinkStream + 2, out_item + 1u);
      }
      const int32_t first = piece * p.piece_packs, count = c.piece_count(first);
      const uint64_t slot = c.g.recv_slot(2, g % (uint32_t)c.g.slots);
      if (last) {
        const uint64_t own_values = c.pack(p.input, (int64_t)own_place * p.stride_packs + first);
        if (MODE == kRingReduce) {
          /* The result goes to its place in the message and, as link 3's own item, around the ring. */
          const uint64_t result = c.pack(p.output, (int64_t)own_place * p.stride_packs + first);
          const uint32_t item3 = own3 + (uint32_t)piece;
          const uint64_t staged = c.g.send_slot(3, item3 % (uint32_t)c.g.slots);
          if (TWO_PASS) {
            link_pass<D, UNROLL, 2, true, false, false>(c.thread, c.threads, slot, own_values, 0, staged, 0, false,
                                                        count, c.zero);
            __syncthreads();
            link_pass<D, UNROLL, 1, true, false, false>(c.thread, c.threads, staged, 0, 0, result, 0, false, count,
                                                        c.zero);
          } else {
            link_pass<D, UNROLL, 2, true, false, false>(c.thread, c.threads, slot, own_values, 0, staged, result,
                                                        true, count, c.zero);
          }
          /* Every thread's stores before thread 0's system fence and ready flag (publish_ready). */
          __syncthreads();
          c.publish_ready(3, item3);
          c.trace.record(c.thread, kTraceKernelReady, kTraceLinkStream + 3, tag3);
        } else {
          link_pass<D, UNROLL, 2, true, false, false>(c.thread, c.threads, slot, own_values, 0,
                                                      c.pack(p.output, first), 0, false, count, c.zero);
          __syncthreads();
        }
      } else {
        /* Inbound type r carries the partial of the owner at ring index i-2-r. */
        const uint64_t values = c.pack(p.input, (int64_t)c.place(-2 - r) * p.stride_packs + first);
        link_pass<D, UNROLL, 2, true, false, false>(c.thread, c.threads, slot, values, 0,
                                                    c.g.send_slot(2, out_item % (uint32_t)c.g.slots), 0, false, count,
                                                    c.zero);
        __syncthreads();
        c.publish_ready(2, out_item);
        c.trace.record(c.thread, kTraceKernelReady, kTraceLinkStream + 2, out_item + 1u);
      }
    }
    c.publish_consumed(2, g);
    c.trace.record(c.thread, kTraceKernelConsumed, kTraceLinkStream + 2, g + 1u);
  }
}

/* Ring all-gather, own role (link 3, outbound item 0 of every round): stage the own piece, and copy it to
 * this rank's place in the output. */
template <int UNROLL>
__device__ __forceinline__ void ring_own(const LinkCtx &c, int sub, uint32_t own3, uint32_t in3) {
  const LinkParams &p = *c.p;
  const int32_t pieces = c.pieces(), own_place = c.place(0);
  for (int32_t q = sub; q < pieces; q += p.blocks_per_role) {
    const uint32_t item = own3 + (uint32_t)q;
    c.wait_own_slot(3, item);
    __syncthreads();
    if (c.poisoned()) break;
    const uint32_t tag3 = in3 + (uint32_t)(q * (p.world - 1)) + 1u;
    c.trace.record(c.thread, kTraceKernelSlot, kTraceLinkStream + 3, tag3);
    const int32_t first = q * p.piece_packs, count = c.piece_count(first);
    link_pass<F32, UNROLL, 1, false, false, false>(c.thread, c.threads, c.pack(p.input, first), 0, 0,
                                                   c.g.send_slot(3, item % (uint32_t)c.g.slots),
                                                   c.pack(p.output, (int64_t)own_place * p.chunk_packs + first), true,
                                                   count, c.zero);
    __syncthreads();
    c.publish_ready(3, item);
    c.trace.record(c.thread, kTraceKernelReady, kTraceLinkStream + 3, tag3);
  }
}

/* Ring, copy role (link 3 inbound): inbound type r of round t carries the finished piece t - r * D3 of the
 * owner at ring index i-1-r; copy it to the owner's place. Items outside the pieces carry flags only. */
template <int UNROLL>
__device__ __forceinline__ void ring_copy(const LinkCtx &c, int sub, int32_t stride_packs, uint32_t in3) {
  const LinkParams &p = *c.p;
  const int32_t per_round = p.world - 1, pieces = c.pieces(), stagger = p.gather_stagger;
  const int32_t items = (pieces + stagger * (per_round - 1)) * per_round;
  const int32_t peer = c.place(-1);
  for (int32_t i = sub; i < items; i += p.blocks_per_role) {
    const uint32_t g = in3 + (uint32_t)i;
    const int32_t t = i / per_round, r = i - t * per_round;
    const int32_t piece = t - r * stagger;
    const bool full = (uint32_t)piece < (uint32_t)pieces;
    c.wait_inbound(3, g, peer);
    c.trace.flags(c.thread, kTraceLinkStream + 3, g + 1u);
    __syncthreads();
    if (c.poisoned()) break;
    if (full) {
      const int32_t first = piece * p.piece_packs, count = c.piece_count(first);
      const uint64_t out = c.pack(p.output, (int64_t)c.place(-1 - r) * stride_packs + first);
      link_pass<F32, UNROLL, 1, true, false, false>(c.thread, c.threads, c.g.recv_slot(3, g % (uint32_t)c.g.slots), 0,
                                                    0, out, 0, false, count, c.zero);
      __syncthreads();
    }
    c.publish_consumed(3, g);
    c.trace.record(c.thread, kTraceKernelConsumed, kTraceLinkStream + 3, g + 1u);
  }
}

/* Ring collective over the chain closed by its last rank's lanes to its first. Roles of blocks_per_role
 * blocks: gather (own, copy), scatter (start, relay), reduce (start, relay, copy). TWO_PASS: the all-reduce
 * relay's two-pass result (ring_relay). */
template <class D, int UNROLL, int MODE, bool TWO_PASS = false>
__device__ __forceinline__ void ring_collective(const LinkParams &p) {
  LinkCtx c = link_context(p);
  const int bid = (int)blockIdx.x, role = bid / p.blocks_per_role, sub = bid - role * p.blocks_per_role;
  if (c.poisoned()) return;
  const uint32_t own2 = ld_relaxed_gpu(p.counters + 4u * kWordOwn2), in2 = ld_relaxed_gpu(p.counters + 4u * kWordIn2);
  const uint32_t own3 = ld_relaxed_gpu(p.counters + 4u * kWordOwn3), in3 = ld_relaxed_gpu(p.counters + 4u * kWordIn3);
  const uint32_t seq = ld_relaxed_gpu(p.counters + 4u * kWordSeq) + 1u;
  uint32_t word = MODE == kRingGather ? kOpRingGather : MODE == kRingScatter ? kOpRingScatter : kOpRingReduce;
  /* The all-gather has no partials to stagger, the reduce-scatter no finished pieces to forward. */
  if (MODE != kRingGather) word |= (uint32_t)p.stagger << kRingStaggerShift;
  if (MODE != kRingScatter) word |= (uint32_t)p.gather_stagger << kRingGatherStaggerShift;
  post_link_op(c, seq, word);
  c.limit_us = ld_relaxed_sys(p.ctrl_base + 4u * kCtrlWaitLimitUs);
  if (MODE == kRingGather) {
    if (role == 0) ring_own<UNROLL>(c, sub, own3, in3);
    if (role == 1) ring_copy<UNROLL>(c, sub, p.chunk_packs, in3);
  } else {
    if (role == 0) ring_start<D, UNROLL>(c, sub, own2);
    if (role == 1) ring_relay<D, UNROLL, MODE, TWO_PASS>(c, sub, own2, in2, own3, in3);
    if (MODE == kRingReduce && role == 2) ring_copy<UNROLL>(c, sub, p.stride_packs, in3);
  }
  const int tail = MODE == kRingGather ? kWordRingGatherTail
                   : MODE == kRingScatter ? kWordRingScatterTail : kWordRingReduceTail;
  if (last_block(c, tail)) {
    const uint32_t pieces = (uint32_t)c.pieces(), steps = (uint32_t)(p.world - 1);
    if (MODE != kRingGather) {
      const uint32_t rounds = pieces + (uint32_t)p.stagger * (uint32_t)(p.world - 2);
      st_release_gpu(p.counters + 4u * kWordOwn2, own2 + rounds * steps);
      st_release_gpu(p.counters + 4u * kWordIn2, in2 + rounds * steps);
    }
    if (MODE != kRingScatter) {
      const uint32_t rounds3 = pieces + (uint32_t)p.gather_stagger * (uint32_t)(p.world - 2);
      st_release_gpu(p.counters + 4u * kWordOwn3, own3 + rounds3);
      st_release_gpu(p.counters + 4u * kWordIn3, in3 + rounds3 * steps);
    }
    st_release_gpu(p.counters + 4u * kWordSeq, seq);
  }
}

/* Pair exchange (a ring of two): the ring all-gather's items, op word and counters, carrying one block
 * each way, or one way, instead of every rank's shard. Three roles of blocks_per_role blocks:
 *   own   (link 3 outbound): piece q of `input` to link 3's own slot (input 0: the op word's bit 24 makes
 *         the proxy post the item as its flags only, and the peer discards it);
 *   copy  (link 3 inbound): piece q of the peer's block to output + peer place * stride, or, in the reduce
 *         form, the dtype rounding of the float32 sum of `scratch` (this rank's own values) and the peer's
 *         piece to output; discarded when output is 0 or flags has kExchangeDiscard;
 *   local: piece q of `scratch` (when not 0, and not the reduce form) to output + own place * stride, in
 *         parallel with the other two.
 * `reserved` carries the flags. */
constexpr int32_t kExchangeDiscard = 1;
/* Op word bit 24 (SIRCL's protocol.RING_OWN_FLAGS): every own item of the op goes out as its flags only. */
constexpr uint32_t kOpOwnFlagsOnly = 1u << 24;

template <int UNROLL>
__device__ __forceinline__ void exchange_own(const LinkCtx &c, int sub, uint32_t own3, uint32_t in3) {
  const LinkParams &p = *c.p;
  const int32_t pieces = c.pieces();
  for (int32_t q = sub; q < pieces; q += p.blocks_per_role) {
    const uint32_t item = own3 + (uint32_t)q;
    c.wait_own_slot(3, item);
    __syncthreads();
    if (c.poisoned()) break;
    const uint32_t tag3 = in3 + (uint32_t)q + 1u;
    c.trace.record(c.thread, kTraceKernelSlot, kTraceLinkStream + 3, tag3);
    const int32_t first = q * p.piece_packs, count = c.piece_count(first);
    if (p.input)
      link_pass<F32, UNROLL, 1, false, false, false>(c.thread, c.threads, c.pack(p.input, first), 0, 0,
                                                     c.g.send_slot(3, item % (uint32_t)c.g.slots), 0, false, count,
                                                     c.zero);
    __syncthreads();
    c.publish_ready(3, item);
    c.trace.record(c.thread, kTraceKernelReady, kTraceLinkStream + 3, tag3);
  }
}

template <class D, int UNROLL, bool REDUCE>
__device__ __forceinline__ void exchange_copy(const LinkCtx &c, int sub, uint32_t in3) {
  const LinkParams &p = *c.p;
  const int32_t pieces = c.pieces(), peer = c.place(-1);
  const bool keep = p.output != 0 && !(p.reserved & kExchangeDiscard);
  for (int32_t i = sub; i < pieces; i += p.blocks_per_role) {
    const uint32_t g = in3 + (uint32_t)i;
    c.wait_inbound(3, g, peer);
    c.trace.flags(c.thread, kTraceLinkStream + 3, g + 1u);
    __syncthreads();
    if (c.poisoned()) break;
    if (keep) {
      const int32_t first = i * p.piece_packs, count = c.piece_count(first);
      const uint64_t slot = c.g.recv_slot(3, g % (uint32_t)c.g.slots);
      if (REDUCE)
        link_pass<D, UNROLL, 2, true, false, false>(c.thread, c.threads, slot, c.pack(p.scratch, first), 0,
                                                    c.pack(p.output, first), 0, false, count, c.zero);
      else
        link_pass<F32, UNROLL, 1, true, false, false>(c.thread, c.threads, slot, 0, 0,
                                                      c.pack(p.output, (int64_t)peer * p.stride_packs + first), 0,
                                                      false, count, c.zero);
      __syncthreads();
    }
    c.publish_consumed(3, g);
    c.trace.record(c.thread, kTraceKernelConsumed, kTraceLinkStream + 3, g + 1u);
  }
}

template <int UNROLL>
__device__ __forceinline__ void exchange_local(const LinkCtx &c, int sub) {
  const LinkParams &p = *c.p;
  if (!p.scratch || !p.output) return;
  const int32_t pieces = c.pieces(), own_place = c.place(0);
  for (int32_t q = sub; q < pieces; q += p.blocks_per_role) {
    if (c.poisoned()) break;
    const int32_t first = q * p.piece_packs, count = c.piece_count(first);
    link_pass<F32, UNROLL, 1, false, false, false>(c.thread, c.threads, c.pack(p.scratch, first), 0, 0,
                                                   c.pack(p.output, (int64_t)own_place * p.stride_packs + first), 0,
                                                   false, count, c.zero);
  }
}

/* The pair exchange op: world 2 only (the engine's choice), 3 * blocks_per_role blocks. On the wire it is
 * the ring all-gather of chunk_packs per rank (op word kOpRingGather with the gather stagger, which on two
 * ranks staggers nothing: one inbound item per round). REDUCE: the reduce-to-root form, whose copy role
 * adds this rank's own values (dtype D). */
template <class D, int UNROLL, bool REDUCE>
__device__ __forceinline__ void ring_exchange(const LinkParams &p) {
  LinkCtx c = link_context(p);
  const int bid = (int)blockIdx.x, role = bid / p.blocks_per_role, sub = bid - role * p.blocks_per_role;
  if (c.poisoned()) return;
  const uint32_t own3 = ld_relaxed_gpu(p.counters + 4u * kWordOwn3), in3 = ld_relaxed_gpu(p.counters + 4u * kWordIn3);
  const uint32_t seq = ld_relaxed_gpu(p.counters + 4u * kWordSeq) + 1u;
  post_link_op(c, seq, kOpRingGather | (uint32_t)p.gather_stagger << kRingGatherStaggerShift |
                           (p.input ? 0u : kOpOwnFlagsOnly));
  c.limit_us = ld_relaxed_sys(p.ctrl_base + 4u * kCtrlWaitLimitUs);
  if (role == 0) exchange_own<UNROLL>(c, sub, own3, in3);
  if (role == 1) exchange_copy<D, UNROLL, REDUCE>(c, sub, in3);
  if (role == 2 && !REDUCE) exchange_local<UNROLL>(c, sub);
  if (last_block(c, kWordRingGatherTail)) {
    const uint32_t pieces = (uint32_t)c.pieces();
    st_release_gpu(p.counters + 4u * kWordOwn3, own3 + pieces);
    st_release_gpu(p.counters + 4u * kWordIn3, in3 + pieces);
    st_release_gpu(p.counters + 4u * kWordSeq, seq);
  }
}

}  // namespace sircl_links

/* Link collective entries, unroll 1-8, each taking one sccl_link_args (src/kernelpack.h):
 *   sircl_link_gather_u<U>               chain all-gather, 4 * blocks_per_role blocks
 *   sircl_link_scatter_<dtype>_u<U>      chain reduce-scatter, 2 * blocks_per_role blocks
 *   sircl_ring_gather_u<U>               ring all-gather, 2 * blocks_per_role blocks
 *   sircl_ring_exchange_u<U>             pair exchange (the ring all-gather on the wire), 3 * blocks_per_role
 *   sircl_ring_exchange_reduce_<dtype>_u<U>  the pair exchange's reduce-to-root form, 3 * blocks_per_role
 *   sircl_ring_scatter_<dtype>_u<U>      ring reduce-scatter, 2 * blocks_per_role blocks
 *   sircl_ring_reduce_<dtype>_u<U>       ring all-reduce, 3 * blocks_per_role blocks
 *   sircl_ring_reduce_two_pass_<dtype>_u<U>  the same with the relay's two-pass result (same bytes, same
 *                                        protocol), for comparisons
 * Blocks of at least 64 threads (multiples of 32), so the slot wait runs in a warp apart from the lane
 * waits; at most 128 blocks per role (one decision word per block). */
#define SIRCL_LINK_BYTES_ENTRIES(U)                                                                              \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_link_gather_u##U(const __grid_constant__ sircl_links::LinkParams p) {                               \
    sircl_links::chain_gather<U>(p);                                                                             \
  }                                                                                                              \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_ring_gather_u##U(const __grid_constant__ sircl_links::LinkParams p) {                               \
    sircl_links::ring_collective<sircl::F32, U, sircl_links::kRingGather>(p);                                   \
  }                                                                                                              \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_ring_exchange_u##U(const __grid_constant__ sircl_links::LinkParams p) {                             \
    sircl_links::ring_exchange<sircl::F32, U, false>(p);                                                        \
  }
#define SIRCL_LINK_DTYPE_ENTRIES(TAG, D, U)                                                                      \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_link_scatter_##TAG##_u##U(const __grid_constant__ sircl_links::LinkParams p) {                      \
    sircl_links::chain_scatter<D, U>(p);                                                                         \
  }                                                                                                              \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_ring_scatter_##TAG##_u##U(const __grid_constant__ sircl_links::LinkParams p) {                      \
    sircl_links::ring_collective<D, U, sircl_links::kRingScatter>(p);                                           \
  }                                                                                                              \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_ring_reduce_##TAG##_u##U(const __grid_constant__ sircl_links::LinkParams p) {                       \
    sircl_links::ring_collective<D, U, sircl_links::kRingReduce>(p);                                            \
  }                                                                                                              \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_ring_reduce_two_pass_##TAG##_u##U(const __grid_constant__ sircl_links::LinkParams p) {              \
    sircl_links::ring_collective<D, U, sircl_links::kRingReduce, true>(p);                                      \
  }                                                                                                              \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                            \
      sircl_ring_exchange_reduce_##TAG##_u##U(const __grid_constant__ sircl_links::LinkParams p) {              \
    sircl_links::ring_exchange<D, U, true>(p);                                                                  \
  }
#define SIRCL_LINK_UNROLL_ENTRIES(U)                                                                             \
  SIRCL_LINK_BYTES_ENTRIES(U)                                                                                    \
  SIRCL_LINK_DTYPE_ENTRIES(f32, sircl::F32, U)                                                                   \
  SIRCL_LINK_DTYPE_ENTRIES(f16, sircl::Half2x4<false>, U)                                                        \
  SIRCL_LINK_DTYPE_ENTRIES(bf16, sircl::Half2x4<true>, U)

SIRCL_LINK_UNROLL_ENTRIES(1)
SIRCL_LINK_UNROLL_ENTRIES(2)
SIRCL_LINK_UNROLL_ENTRIES(3)
SIRCL_LINK_UNROLL_ENTRIES(4)
SIRCL_LINK_UNROLL_ENTRIES(5)
SIRCL_LINK_UNROLL_ENTRIES(6)
SIRCL_LINK_UNROLL_ENTRIES(7)
SIRCL_LINK_UNROLL_ENTRIES(8)

/* Entry points sircl_chain_<dtype>_u<unroll>, unroll 1-8. Launch with 4 * blocks_per_role blocks of at least
 * 64 threads. The parameter order is the C launcher's in src/kernelpack.c. */
#define SIRCL_CHAIN_ENTRY(TAG, D, U)                                                                             \
  extern "C" __global__ void __launch_bounds__(512, 1) sircl_chain_##TAG##_u##U(                                   \
      uint64_t input, uint64_t output, int32_t a_packs, int32_t b_packs, int32_t chunk_packs, uint64_t chain_base, \
      uint64_t counters, uint64_t ctrl_base, uint64_t poison, uint32_t spin_limit, uint64_t trace_base,           \
      uint32_t trace_capacity, int32_t world, int32_t index, int32_t prev, int32_t next, int32_t rank,             \
      int32_t lanes, int32_t slots, uint64_t slot_bytes, int32_t blocks_per_role) {                               \
    sircl_links::ChainArgs a;                                                                                    \
    a.input = input;                                                                                             \
    a.output = output;                                                                                           \
    a.a_packs = a_packs;                                                                                         \
    a.b_packs = b_packs;                                                                                         \
    a.chunk_packs = chunk_packs;                                                                                 \
    a.counters = counters;                                                                                       \
    a.ctrl_base = ctrl_base;                                                                                     \
    a.poison = poison;                                                                                           \
    a.spin_limit = spin_limit;                                                                                   \
    a.limit_us = 0;                                                                                              \
    a.world = world;                                                                                             \
    a.index = index;                                                                                             \
    a.prev = prev;                                                                                               \
    a.next = next;                                                                                               \
    a.rank = rank;                                                                                               \
    a.blocks_per_role = blocks_per_role;                                                                         \
    a.chain.base = chain_base;                                                                                   \
    a.chain.lanes = lanes;                                                                                       \
    a.chain.slots = slots;                                                                                       \
    a.chain.slot_bytes = slot_bytes;                                                                             \
    a.trace.base = trace_base;                                                                                   \
    a.trace.capacity = trace_capacity;                                                                           \
    sircl_links::chain_all_reduce<D, U>(a);                                                                      \
  }
#define SIRCL_CHAIN_UNROLLS(TAG, D)                                                                              \
  SIRCL_CHAIN_ENTRY(TAG, D, 1)                                                                                   \
  SIRCL_CHAIN_ENTRY(TAG, D, 2)                                                                                   \
  SIRCL_CHAIN_ENTRY(TAG, D, 3)                                                                                   \
  SIRCL_CHAIN_ENTRY(TAG, D, 4)                                                                                   \
  SIRCL_CHAIN_ENTRY(TAG, D, 5)                                                                                   \
  SIRCL_CHAIN_ENTRY(TAG, D, 6)                                                                                   \
  SIRCL_CHAIN_ENTRY(TAG, D, 7)                                                                                   \
  SIRCL_CHAIN_ENTRY(TAG, D, 8)

SIRCL_CHAIN_UNROLLS(f32, sircl::F32)
SIRCL_CHAIN_UNROLLS(f16, sircl::Half2x4<false>)
SIRCL_CHAIN_UNROLLS(bf16, sircl::Half2x4<true>)
