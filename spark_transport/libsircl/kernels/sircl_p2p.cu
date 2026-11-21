/*
 * The point-to-point pack: the send and receive kernels of SIRCL's point-to-point channels, ported to CUDA
 * C++ from SIRCL's CuTe DSL kernels (sparkring_sircl/p2p/_kernels.py) and speaking their protocol with the
 * native progress thread of SIRCL's point-to-point library (src/transport/sircl_p2p_proxy.c). SIRCL
 * specializes each kernel on its threads, lanes, slots and slot bytes at compile time; these entries take
 * them, and the block offsets the native layer reports (p2p_layout), as launch parameters, so one compiled
 * entry per direction and unroll serves every geometry.
 *
 * Channel. Every ordered pair of a group's ranks with a channel has `slots` slots of `slot_bytes` bytes on
 * both ends, in the peer's block of the rank's arena (pinned host memory the GPU and every RDMA device
 * address). Item g of the channel (counted from 0 over the communicator's life, 32 bits, wrapping) uses slot
 * g % slots and carries tag g + 1. A message of n bytes is items(n) = max(1, ceil(padded(n) / slot_bytes))
 * items; item i carries packs [i * slot_packs, min(packs, (i + 1) * slot_packs)) and its header holds its
 * byte count, and on the last item bit 31 and n % 16 in bits 0-3, so a receive of another size fails on the
 * first item that differs.
 *
 * One launch per message. Block b of the grid takes items b, b + gridDim.x, ...
 *  - Send: the block's last thread waits until the channel's sent word (the items toward the peer whose
 *    writes completed, written by this rank's progress thread) reaches tag - slots, so the slot's previous
 *    item has left it; the block copies the item from the source into the send slot; thread 0 writes the
 *    header into the desc line, then the tag into the ready line, each after a system-scope fence. The
 *    progress thread then posts the item's stripes, header and lane flags.
 *  - Receive: threads 0 .. lanes - 1 wait until their lane flag of the slot holds the tag; thread 0 compares
 *    the header (byte 4 of lane 0's flag line, written before lane 0's flag on the same queue pair) with the
 *    one the receive expects; the block copies the slot into the output; thread 0 writes the tag into the
 *    consumed line after a system-scope fence, and the progress thread returns the slot to the sender.
 *
 * Every wait is bounded by the context's wait limit (control word 0, in microseconds, read when the launch
 * starts; 0 means none) on the GPU's nanosecond clock, checked every 1,024 polls, and watches the poison
 * word (control word 7, checked every 64 polls). A timeout or a header that differs records the peer, lane
 * (255: none), kind (1: lane flag, 2: send slot, 3: size), expected and received header (words 2-6), then
 * the tag (word 1) and sets the poison word; every later item and launch of the context returns at once and
 * the progress thread stops every rank's channels.
 *
 * A pass of the copy issues all of its 16-byte loads before its stores (the loads of the inbound slot at
 * system scope), so a pass pays one host-memory latency.
 */
#include <stdint.h>

#include "sircl_common.cuh"

namespace sircl_p2p {

using namespace sircl;

constexpr int kLine = 128;
constexpr int kWaitLimitUs = 0;
constexpr int kErrorTag = 1;
constexpr int kErrorPeer = 2;
constexpr int kErrorLane = 3;
constexpr int kErrorKind = 4;
constexpr int kErrorExpected = 5;
constexpr int kErrorGot = 6;
constexpr int kPoison = 7;
constexpr uint32_t kKindFlag = 1;
constexpr uint32_t kKindSlot = 2;
constexpr uint32_t kKindSize = 3;
constexpr uint32_t kNoLane = 255;
constexpr uint32_t kLast = 1u << 31;
constexpr uint32_t kPollsPerPoisonCheck = 64;

/* One launch: a message's packs, n % 16, the message's items, the peer's block, the control line and the
 * channel's device counter pair ({seq, arrival}, 8 bytes), and the block offsets of the native layer's
 * p2p_layout. Mirrors sccl_p2p_args of src/kernelpack.h (120 bytes). */
struct P2PParams {
  uint64_t data, block_base, ctrl_base, slot_bytes, counter_ptr;
  int32_t packs, tail, items, peer;
  uint32_t slots, lanes;
  uint64_t recv_off, send_off, flag_off, desc_off, ready_off, consumed_off, sent_off;
};
static_assert(sizeof(P2PParams) == 120, "P2PParams must match sccl_p2p_args");

/* Spin until the word equals `expected` (system-scope acquire loads): 0 on a match, 1 after `limit_us`
 * microseconds (0: no limit), 2 as soon as the poison word is nonzero. */
__device__ __forceinline__ uint32_t wait_eq_or_poison(uint64_t addr, uint32_t expected, uint64_t poison,
                                                      uint32_t limit_us) {
  const bool timed = limit_us != 0;
  const uint64_t budget = (uint64_t)limit_us * 1000u;
  const uint64_t start = globaltimer();
  uint32_t polls = 0;
  for (;;) {
    if (ld_acquire_sys(addr) == expected) return 0;
    polls += 1;
    if (polls & (kPollsPerPoisonCheck - 1)) continue;
    if (ld_relaxed_sys(poison) != 0) return 2;
    if ((polls & (kPollsPerClockCheck - 1)) || !timed) continue;
    if (globaltimer() - start >= budget) return 1;
  }
}

/* Spin until the counter reaches `target` ((int32)(value - target) >= 0, so counters may wrap), with the
 * results and limits of wait_eq_or_poison. */
__device__ __forceinline__ uint32_t wait_ge_or_poison(uint64_t addr, uint32_t target, uint64_t poison,
                                                      uint32_t limit_us) {
  const bool timed = limit_us != 0;
  const uint64_t budget = (uint64_t)limit_us * 1000u;
  const uint64_t start = globaltimer();
  uint32_t polls = 0;
  for (;;) {
    if ((int32_t)(ld_acquire_sys(addr) - target) >= 0) return 0;
    polls += 1;
    if (polls & (kPollsPerPoisonCheck - 1)) continue;
    if (ld_relaxed_sys(poison) != 0) return 2;
    if ((polls & (kPollsPerClockCheck - 1)) || !timed) continue;
    if (globaltimer() - start >= budget) return 1;
  }
}

/* The failure record: every detail word, then the tag, then the poison word, each step after a fence. */
__device__ __forceinline__ void fail(uint64_t ctrl, int32_t peer, uint32_t lane, uint32_t kind, uint32_t tag,
                                     uint32_t expected, uint32_t got) {
  st_relaxed_sys(ctrl + 4 * kErrorPeer, (uint32_t)peer);
  st_relaxed_sys(ctrl + 4 * kErrorLane, lane);
  st_relaxed_sys(ctrl + 4 * kErrorKind, kind);
  st_relaxed_sys(ctrl + 4 * kErrorExpected, expected);
  st_relaxed_sys(ctrl + 4 * kErrorGot, got);
  fence_sc_sys();
  st_relaxed_sys(ctrl + 4 * kErrorTag, tag);
  fence_sc_sys();
  st_relaxed_sys(ctrl + 4 * kPoison, 1u);
}

/* Copy `packs` packs from `src` to `dst`, U packs per thread per pass with the loads of a pass issued
 * together; kSystem reads pinned host memory at system scope. */
template <int U, bool kSystem>
__device__ __forceinline__ void copy_packs(uint64_t src, uint64_t dst, int32_t packs) {
  const int32_t threads = (int32_t)blockDim.x;
  const int32_t thread = (int32_t)threadIdx.x;
  const int32_t last = packs - 1;
  for (int32_t base = 0; base < packs; base += U * threads) {
    uint4 words[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int32_t index = min(base + u * threads + thread, last);
      const uint64_t at = src + (uint64_t)index * kPackBytes;
      words[u] = kSystem ? ld_relaxed_sys_v4(at) : ld_global_v4(at);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int32_t index = base + u * threads + thread;
      if (index < packs) st_global_v4(dst + (uint64_t)index * kPackBytes, words[u]);
    }
  }
}

__device__ __forceinline__ uint32_t header_of(int32_t item, int32_t items, int32_t count, int32_t tail) {
  uint32_t header = (uint32_t)count * kPackBytes;
  if (item == items - 1) header |= kLast | (uint32_t)tail;
  return header;
}

/* Advance the channel's device sequence counter once every block of this launch has finished reading it and
 * completed its items, unless the context is poisoned. Each channel runs on its own stream, so successive
 * launches (eager or CUDA-graph replayed) execute strictly in order. */
__device__ __forceinline__ void advance_counter(int32_t thread, uint64_t counter_ptr, uint32_t first, int32_t items,
                                                uint64_t poison) {
  __syncthreads();
  if (thread == 0) {
    fence_sc_gpu();
    const uint32_t done = atom_add_relaxed_gpu(counter_ptr + 4u, 1u) + 1u;
    if (done == gridDim.x) {
      st_release_gpu(counter_ptr + 4u, 0u);
      if (ld_relaxed_sys(poison) == 0) st_release_gpu(counter_ptr, first + (uint32_t)items);
    }
  }
}

template <int U>
__device__ __forceinline__ void send(const P2PParams &a) {
  __shared__ uint32_t poisoned;
  __shared__ uint32_t s_first;
  const int32_t thread = (int32_t)threadIdx.x;
  const int32_t last_thread = (int32_t)blockDim.x - 1;
  if (thread == 0) s_first = ld_acquire_gpu(a.counter_ptr);
  __syncthreads();
  const uint32_t first = s_first;
  const uint64_t poison = a.ctrl_base + 4 * kPoison;
  const uint32_t limit_us = ld_relaxed_sys(a.ctrl_base + 4 * kWaitLimitUs);
  const int32_t slot_packs = (int32_t)(a.slot_bytes / kPackBytes);
  int32_t item = (int32_t)blockIdx.x;
  while (item < a.items) {
    const uint32_t g = first + (uint32_t)item;
    const uint32_t m = g % a.slots;
    const uint32_t tag = g + 1u;
    if (thread == last_thread &&
        wait_ge_or_poison(a.block_base + a.sent_off, tag - a.slots, poison, limit_us) == 1)
      fail(a.ctrl_base, a.peer, kNoLane, kKindSlot, tag, 0, 0);
    __syncthreads();
    if (thread == 0) poisoned = ld_relaxed_sys(poison);
    __syncthreads();
    const bool healthy = poisoned == 0;
    const int32_t start = item * slot_packs;
    const int32_t count = min(slot_packs, a.packs - start);
    if (healthy)
      copy_packs<U, false>(a.data + (uint64_t)start * kPackBytes, a.block_base + a.send_off + (uint64_t)m * a.slot_bytes,
                           count);
    __syncthreads();
    if (!healthy) break;
    if (thread == 0) {
      fence_sc_sys();
      st_relaxed_sys(a.block_base + a.desc_off + 4ull * m, header_of(item, a.items, count, a.tail));
      fence_sc_sys();
      st_relaxed_sys(a.block_base + a.ready_off + 4ull * m, tag);
    }
    item += (int32_t)gridDim.x;
  }
  advance_counter(thread, a.counter_ptr, first, a.items, poison);
}

template <int U>
__device__ __forceinline__ void recv(const P2PParams &a) {
  __shared__ uint32_t poisoned;
  __shared__ uint32_t s_first;
  const int32_t thread = (int32_t)threadIdx.x;
  if (thread == 0) s_first = ld_acquire_gpu(a.counter_ptr);
  __syncthreads();
  const uint32_t first = s_first;
  const uint64_t poison = a.ctrl_base + 4 * kPoison;
  const uint32_t limit_us = ld_relaxed_sys(a.ctrl_base + 4 * kWaitLimitUs);
  const int32_t slot_packs = (int32_t)(a.slot_bytes / kPackBytes);
  int32_t item = (int32_t)blockIdx.x;
  while (item < a.items) {
    const uint32_t g = first + (uint32_t)item;
    const uint32_t m = g % a.slots;
    const uint32_t tag = g + 1u;
    const uint64_t line = a.block_base + a.flag_off + (uint64_t)m * a.lanes * kLine;
    if (thread < (int32_t)a.lanes &&
        wait_eq_or_poison(line + (uint64_t)thread * kLine, tag, poison, limit_us) == 1)
      fail(a.ctrl_base, a.peer, (uint32_t)thread, kKindFlag, tag, 0, 0);
    __syncthreads();
    const int32_t start = item * slot_packs;
    const int32_t count = min(slot_packs, a.packs - start);
    if (thread == 0) {
      /* Thread 0 waited for lane 0's flag, which follows the header on the same queue pair. */
      if (ld_relaxed_sys(poison) == 0) {
        const uint32_t expected = header_of(item, a.items, count, a.tail);
        const uint32_t got = ld_relaxed_sys(line + 4);
        if (got != expected) fail(a.ctrl_base, a.peer, kNoLane, kKindSize, tag, expected, got);
      }
      poisoned = ld_relaxed_sys(poison);
    }
    __syncthreads();
    const bool healthy = poisoned == 0;
    if (healthy)
      copy_packs<U, true>(a.block_base + a.recv_off + (uint64_t)m * a.slot_bytes, a.data + (uint64_t)start * kPackBytes,
                          count);
    __syncthreads();
    if (!healthy) break;
    if (thread == 0) {
      fence_sc_sys();
      st_relaxed_sys(a.block_base + a.consumed_off + 4ull * m, tag);
    }
    item += (int32_t)gridDim.x;
  }
  advance_counter(thread, a.counter_ptr, first, a.items, poison);
}

}  // namespace sircl_p2p

/* Entry points sircl_p2p_send_u<unroll> and sircl_p2p_recv_u<unroll>, unroll 1-8: launched with up to
 * SCCL_P2P_MAX_THREADS (512) threads per block, a multiple of 32 from 64, and min(blocks, items) blocks. */
#define SIRCL_P2P_ENTRIES(U)                                                                                    \
  extern "C" __global__ void __launch_bounds__(512, 1) sircl_p2p_send_u##U(const sircl_p2p::P2PParams a) {    \
    sircl_p2p::send<U>(a);                                                                                     \
  }                                                                                                            \
  extern "C" __global__ void __launch_bounds__(512, 1) sircl_p2p_recv_u##U(const sircl_p2p::P2PParams a) {    \
    sircl_p2p::recv<U>(a);                                                                                     \
  }

SIRCL_P2P_ENTRIES(1)
SIRCL_P2P_ENTRIES(2)
SIRCL_P2P_ENTRIES(3)
SIRCL_P2P_ENTRIES(4)
SIRCL_P2P_ENTRIES(5)
SIRCL_P2P_ENTRIES(6)
SIRCL_P2P_ENTRIES(7)
SIRCL_P2P_ENTRIES(8)

/* ======================================================================================================
 * SIRCL-EP: GPU-driven dynamic variable-length MoE token Dispatch & Combine over SIRCL's P2P channels.
 *
 * Uses the exact same wire protocol and unmodified sircl_p2p_proxy.c:
 *   - desc_off[m] holds `(chunk_bytes & kBytesMask) | (is_last ? kLast : 0)`
 *   - sircl_p2p_proxy.c RDMA-writes ONLY `chunk_bytes` (0 data bytes when chunk_bytes == 0!) plus
 *     the 4-byte header into peer's flag_line + 4 and the 4-byte tag into every lane's flag line.
 *   - Zero CPU synchronization, zero wasted RDMA bytes, 100% CUDA-graph capturable, single fused launch
 *     across all peers (no per-channel stream fork/join overhead).
 * ====================================================================================================== */
namespace sircl_ep {

using namespace sircl;
using namespace sircl_p2p;

constexpr int kMaxWorld = 8;
constexpr int kEpThreads = 256;
constexpr uint32_t kBytesMask = ((1u << 30) - 1u) & ~15u;

struct EpParams {
  uint64_t hidden_in;
  uint64_t topk_ids_in;
  uint64_t topk_weights_in;
  uint64_t hidden_out;
  uint64_t topk_ids_out;
  uint64_t topk_weights_out;
  uint64_t recv_counts_out;
  uint64_t workspace;
  uint64_t p2p_dev;
  uint64_t p2p_counters;
  uint64_t layout_control;
  uint64_t layout_block;
  uint64_t slot_bytes;
  uint64_t recv_off;
  uint64_t send_off;
  uint64_t flag_off;
  uint64_t desc_off;
  uint64_t ready_off;
  uint64_t consumed_off;
  uint64_t sent_off;
  int32_t world;
  int32_t rank;
  int32_t num_tokens;
  int32_t max_tokens;
  int32_t hidden_bytes;
  int32_t top_k;
  int32_t num_experts;
  uint32_t slots;
  uint32_t lanes;
  uint32_t wire_row_bytes;
};
static_assert(sizeof(EpParams) == 200, "EpParams must match sccl_ep_args");

__device__ __forceinline__ uint64_t align64(uint64_t v) { return (v + 63ull) & ~63ull; }
__device__ __forceinline__ uint64_t ws_recv_counts(const EpParams &a) { return a.workspace; }
__device__ __forceinline__ uint64_t ws_send_counts(const EpParams &a) { return a.workspace + 32ull; }
__device__ __forceinline__ uint64_t ws_send_indices(const EpParams &a) { return a.workspace + 64ull; }
__device__ __forceinline__ uint64_t ws_send_slot_of_tok(const EpParams &a) {
  return ws_send_indices(a) + align64((uint64_t)a.world * (uint64_t)a.max_tokens * 4ull);
}
__device__ __forceinline__ uint64_t ws_combine_recv(const EpParams &a) {
  return ws_send_slot_of_tok(a) + align64((uint64_t)a.world * (uint64_t)a.max_tokens * 4ull);
}

__device__ __forceinline__ uint64_t peer_block(const EpParams &a, int32_t p) {
  return a.p2p_dev + a.layout_control + (uint64_t)p * a.layout_block;
}
__device__ __forceinline__ uint64_t chan_counter(const EpParams &a, int32_t send_dir, int32_t p) {
  return a.p2p_counters + (uint64_t)(send_dir * kMaxWorld + p) * 8ull;
}

__device__ __forceinline__ void mask_unused_slots(const EpParams &a, int32_t p, int32_t start_k, int32_t dummy_e) {
  const int32_t thread = (int32_t)threadIdx.x;
  const int32_t threads = (int32_t)blockDim.x;
  const int32_t unused = a.max_tokens - start_k;
  if (unused <= 0) return;
  const int32_t total_elems = unused * a.top_k;
  int32_t *ids_out = (int32_t *)a.topk_ids_out + (int64_t)(p * a.max_tokens + start_k) * a.top_k;
  float *wts_out = (float *)a.topk_weights_out + (int64_t)(p * a.max_tokens + start_k) * a.top_k;
  for (int32_t idx = thread; idx < total_elems; idx += threads) {
    ids_out[idx] = dummy_e;
    wts_out[idx] = 0.0f;
  }
}

__device__ __forceinline__ void ep_dispatch_impl(const EpParams &a) {
  const int32_t bid = (int32_t)blockIdx.x;
  const int32_t thread = (int32_t)threadIdx.x;
  const int32_t last_thread = (int32_t)blockDim.x - 1;
  const uint64_t ctrl = a.p2p_dev;
  const uint64_t poison = ctrl + 4 * kPoison;
  const uint32_t limit_us = (a.world > 1) ? ld_relaxed_sys(ctrl + 4 * kWaitLimitUs) : 0u;
  const int32_t local_experts = a.num_experts / a.world;
  const int32_t dummy_e = (a.world > 1) ? (((a.rank + 1) % a.world) * local_experts) : -1;
  const int32_t hidden_packs = a.hidden_bytes / kPackBytes;

  if (bid < a.world) {
    /* Sender + self-copy block for destination peer p = bid. */
    const int32_t p = bid;
    const int32_t e_lo = p * local_experts;
    const int32_t e_hi = e_lo + local_experts;
    int32_t *send_indices = (int32_t *)ws_send_indices(a) + (int64_t)p * a.max_tokens;
    int32_t *send_slot_of_tok = (int32_t *)ws_send_slot_of_tok(a) + (int64_t)p * a.max_tokens;

    __shared__ int32_t s_prefix[kEpThreads];
    __shared__ int32_t s_total;
    __shared__ uint32_t s_poisoned;
    __shared__ uint32_t s_seq;
    if (thread == 0) {
      s_total = 0;
      s_poisoned = 0;
    }
    __syncthreads();

    for (int32_t base = 0; base < a.num_tokens; base += kEpThreads) {
      const int32_t t = base + thread;
      int32_t hit = 0;
      if (t < a.num_tokens) {
        const int32_t *ids = (const int32_t *)a.topk_ids_in + (int64_t)t * a.top_k;
        for (int32_t k = 0; k < a.top_k; ++k) {
          const int32_t e = ids[k];
          if (e >= e_lo && e < e_hi) {
            hit = 1;
            break;
          }
        }
      }
      s_prefix[thread] = hit;
      __syncthreads();
      for (int32_t stride = 1; stride < kEpThreads; stride <<= 1) {
        const int32_t val = (thread >= stride) ? s_prefix[thread - stride] : 0;
        __syncthreads();
        s_prefix[thread] += val;
        __syncthreads();
      }
      if (t < a.num_tokens) {
        if (hit) {
          const int32_t idx = s_total + s_prefix[thread] - 1;
          send_indices[idx] = t;
          send_slot_of_tok[t] = idx;
        } else {
          send_slot_of_tok[t] = -1;
        }
      }
      __syncthreads();
      if (thread == 0) s_total += s_prefix[kEpThreads - 1];
      __syncthreads();
    }

    const int32_t n_send = s_total;
    if (thread == 0) ((int32_t *)ws_send_counts(a))[p] = n_send;

    if (p == a.rank) {
      /* Self copy: direct GPU-to-GPU copy into slice [rank * max_tokens ..] + mask unused slots. */
      for (int32_t i = 0; i < n_send; ++i) {
        const int32_t t = send_indices[i];
        const int32_t out_row = p * a.max_tokens + i;
        copy_packs<2, false>(a.hidden_in + (uint64_t)t * (uint64_t)a.hidden_bytes,
                             a.hidden_out + (uint64_t)out_row * (uint64_t)a.hidden_bytes, hidden_packs);
        if (thread < a.top_k) {
          ((int32_t *)a.topk_ids_out)[(int64_t)out_row * a.top_k + thread] =
              ((const int32_t *)a.topk_ids_in)[(int64_t)t * a.top_k + thread];
          ((float *)a.topk_weights_out)[(int64_t)out_row * a.top_k + thread] =
              ((const float *)a.topk_weights_in)[(int64_t)t * a.top_k + thread];
        }
      }
      mask_unused_slots(a, p, n_send, dummy_e);
      if (thread == 0) {
        ((int32_t *)ws_recv_counts(a))[p] = n_send;
        if (a.recv_counts_out) ((int32_t *)a.recv_counts_out)[p] = n_send;
      }
      return;
    }

    /* Remote send to peer p over P2P channel (dir = 1, peer = p). */
    const uint64_t bbase = peer_block(a, p);
    const uint64_t cptr = chan_counter(a, 1, p);
    if (thread == 0) s_seq = ld_acquire_gpu(cptr);
    __syncthreads();
    uint32_t g = s_seq;
    const uint64_t pipe_bytes = (a.slot_bytes < 131072ull) ? a.slot_bytes : 131072ull;
    const int32_t toks_per_slot = max(1, (int32_t)(pipe_bytes / (uint64_t)a.wire_row_bytes));

    if (n_send == 0) {
      const uint32_t m = g % a.slots;
      const uint32_t tag = g + 1u;
      if (thread == last_thread && wait_ge_or_poison(bbase + a.sent_off, tag - a.slots, poison, limit_us) == 1)
        fail(ctrl, p, kNoLane, kKindSlot, tag, 0, 0);
      __syncthreads();
      if (thread == 0) s_poisoned = ld_relaxed_sys(poison);
      __syncthreads();
      if (s_poisoned == 0) {
        if (thread == 0) {
          fence_sc_sys();
          st_relaxed_sys(bbase + a.desc_off + 4ull * m, kLast);
          fence_sc_sys();
          st_relaxed_sys(bbase + a.ready_off + 4ull * m, tag);
        }
        g += 1u;
      }
    } else {
      for (int32_t offset = 0; offset < n_send; offset += toks_per_slot) {
        const int32_t chunk = min(toks_per_slot, n_send - offset);
        const bool is_last = (offset + chunk) == n_send;
        const uint32_t m = g % a.slots;
        const uint32_t tag = g + 1u;
        if (thread == last_thread && wait_ge_or_poison(bbase + a.sent_off, tag - a.slots, poison, limit_us) == 1)
          fail(ctrl, p, kNoLane, kKindSlot, tag, 0, 0);
        __syncthreads();
        if (thread == 0) s_poisoned = ld_relaxed_sys(poison);
        __syncthreads();
        if (s_poisoned != 0) break;

        const uint64_t slot_base = bbase + a.send_off + (uint64_t)m * a.slot_bytes;
        int32_t i = 0;
        for (; i + 4 <= chunk; i += 4) {
          const int32_t t0 = send_indices[offset + i];
          const int32_t t1 = send_indices[offset + i + 1];
          const int32_t t2 = send_indices[offset + i + 2];
          const int32_t t3 = send_indices[offset + i + 3];
          const uint64_t s0 = a.hidden_in + (uint64_t)t0 * (uint64_t)a.hidden_bytes;
          const uint64_t s1 = a.hidden_in + (uint64_t)t1 * (uint64_t)a.hidden_bytes;
          const uint64_t s2 = a.hidden_in + (uint64_t)t2 * (uint64_t)a.hidden_bytes;
          const uint64_t s3 = a.hidden_in + (uint64_t)t3 * (uint64_t)a.hidden_bytes;
          const uint64_t d0 = slot_base + (uint64_t)(i + 0) * (uint64_t)a.wire_row_bytes;
          const uint64_t d1 = slot_base + (uint64_t)(i + 1) * (uint64_t)a.wire_row_bytes;
          const uint64_t d2 = slot_base + (uint64_t)(i + 2) * (uint64_t)a.wire_row_bytes;
          const uint64_t d3 = slot_base + (uint64_t)(i + 3) * (uint64_t)a.wire_row_bytes;
          for (int32_t pk = thread; pk < hidden_packs; pk += kEpThreads) {
            const uint64_t off = (uint64_t)pk * kPackBytes;
            const uint4 v0 = ld_global_v4(s0 + off);
            const uint4 v1 = ld_global_v4(s1 + off);
            const uint4 v2 = ld_global_v4(s2 + off);
            const uint4 v3 = ld_global_v4(s3 + off);
            st_global_v4(d0 + off, v0);
            st_global_v4(d1 + off, v1);
            st_global_v4(d2 + off, v2);
            st_global_v4(d3 + off, v3);
          }
          if (thread < a.top_k) {
            const int32_t ts[4] = {t0, t1, t2, t3};
            const uint64_t ds[4] = {d0, d1, d2, d3};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const uint64_t meta = ds[r] + (uint64_t)a.hidden_bytes;
              ((int32_t *)meta)[thread] = ((const int32_t *)a.topk_ids_in)[(int64_t)ts[r] * a.top_k + thread];
              ((float *)meta)[a.top_k + thread] = ((const float *)a.topk_weights_in)[(int64_t)ts[r] * a.top_k + thread];
            }
          }
        }
        for (; i < chunk; ++i) {
          const int32_t t = send_indices[offset + i];
          const uint64_t dst = slot_base + (uint64_t)i * (uint64_t)a.wire_row_bytes;
          copy_packs<2, false>(a.hidden_in + (uint64_t)t * (uint64_t)a.hidden_bytes, dst, hidden_packs);
          if (thread < a.top_k) {
            const uint64_t meta = dst + (uint64_t)a.hidden_bytes;
            const int32_t eid = ((const int32_t *)a.topk_ids_in)[(int64_t)t * a.top_k + thread];
            const float wt = ((const float *)a.topk_weights_in)[(int64_t)t * a.top_k + thread];
            ((int32_t *)meta)[thread] = eid;
            ((float *)meta)[a.top_k + thread] = wt;
          }
        }
        __syncthreads();
        if (thread == 0) {
          const uint32_t hdr = ((uint32_t)chunk * a.wire_row_bytes) | (is_last ? kLast : 0u);
          fence_sc_sys();
          st_relaxed_sys(bbase + a.desc_off + 4ull * m, hdr);
          fence_sc_sys();
          st_relaxed_sys(bbase + a.ready_off + 4ull * m, tag);
        }
        g += 1u;
      }
    }
    __syncthreads();
    if (thread == 0 && ld_relaxed_sys(poison) == 0) st_release_gpu(cptr, g);
  } else {
    /* Receiver block for source peer p = bid - world. */
    const int32_t p = bid - a.world;
    if (p == a.rank) return;

    __shared__ uint32_t s_poisoned;
    __shared__ uint32_t s_seq;
    __shared__ uint32_t s_hdr;
    const uint64_t bbase = peer_block(a, p);
    const uint64_t cptr = chan_counter(a, 0, p);
    if (thread == 0) {
      s_seq = ld_acquire_gpu(cptr);
      s_poisoned = 0;
    }
    __syncthreads();
    uint32_t g = s_seq;
    int32_t received = 0;

    for (;;) {
      const uint32_t m = g % a.slots;
      const uint32_t tag = g + 1u;
      const uint64_t line = bbase + a.flag_off + (uint64_t)m * a.lanes * kLine;
      if (thread < (int32_t)a.lanes && wait_eq_or_poison(line + (uint64_t)thread * kLine, tag, poison, limit_us) == 1)
        fail(ctrl, p, (uint32_t)thread, kKindFlag, tag, 0, 0);
      __syncthreads();
      if (thread == 0) {
        if (ld_relaxed_sys(poison) == 0) {
          const uint32_t got = ld_relaxed_sys(line + 4);
          const uint32_t cbytes = got & kBytesMask;
          if (cbytes % a.wire_row_bytes != 0 || received + (int32_t)(cbytes / a.wire_row_bytes) > a.max_tokens) {
            fail(ctrl, p, kNoLane, kKindSize, tag, (uint32_t)a.max_tokens * a.wire_row_bytes, got);
          }
          s_hdr = got;
        }
        s_poisoned = ld_relaxed_sys(poison);
      }
      __syncthreads();
      if (s_poisoned != 0) break;

      const uint32_t cbytes = s_hdr & kBytesMask;
      const bool is_last = (s_hdr & kLast) != 0;
      const int32_t chunk = (int32_t)(cbytes / a.wire_row_bytes);
      const uint64_t slot_base = bbase + a.recv_off + (uint64_t)m * a.slot_bytes;
      int32_t i = 0;
      for (; i + 4 <= chunk; i += 4) {
        const uint64_t s0 = slot_base + (uint64_t)(i + 0) * (uint64_t)a.wire_row_bytes;
        const uint64_t s1 = slot_base + (uint64_t)(i + 1) * (uint64_t)a.wire_row_bytes;
        const uint64_t s2 = slot_base + (uint64_t)(i + 2) * (uint64_t)a.wire_row_bytes;
        const uint64_t s3 = slot_base + (uint64_t)(i + 3) * (uint64_t)a.wire_row_bytes;
        const int32_t r0 = p * a.max_tokens + received + i;
        const uint64_t d0 = a.hidden_out + (uint64_t)(r0 + 0) * (uint64_t)a.hidden_bytes;
        const uint64_t d1 = a.hidden_out + (uint64_t)(r0 + 1) * (uint64_t)a.hidden_bytes;
        const uint64_t d2 = a.hidden_out + (uint64_t)(r0 + 2) * (uint64_t)a.hidden_bytes;
        const uint64_t d3 = a.hidden_out + (uint64_t)(r0 + 3) * (uint64_t)a.hidden_bytes;
        for (int32_t pk = thread; pk < hidden_packs; pk += kEpThreads) {
          const uint64_t off = (uint64_t)pk * kPackBytes;
          const uint4 v0 = ld_relaxed_sys_v4(s0 + off);
          const uint4 v1 = ld_relaxed_sys_v4(s1 + off);
          const uint4 v2 = ld_relaxed_sys_v4(s2 + off);
          const uint4 v3 = ld_relaxed_sys_v4(s3 + off);
          st_global_v4(d0 + off, v0);
          st_global_v4(d1 + off, v1);
          st_global_v4(d2 + off, v2);
          st_global_v4(d3 + off, v3);
        }
        if (thread < a.top_k) {
          const uint64_t ss[4] = {s0, s1, s2, s3};
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const uint64_t meta = ss[r] + (uint64_t)a.hidden_bytes;
            const int32_t out_row = r0 + r;
            ((int32_t *)a.topk_ids_out)[(int64_t)out_row * a.top_k + thread] =
                (int32_t)ld_relaxed_sys(meta + 4ull * (uint64_t)thread);
            ((float *)a.topk_weights_out)[(int64_t)out_row * a.top_k + thread] =
                __uint_as_float(ld_relaxed_sys(meta + 4ull * (uint64_t)(a.top_k + thread)));
          }
        }
      }
      for (; i < chunk; ++i) {
        const uint64_t src = slot_base + (uint64_t)i * (uint64_t)a.wire_row_bytes;
        const int32_t out_row = p * a.max_tokens + received + i;
        copy_packs<2, true>(src, a.hidden_out + (uint64_t)out_row * (uint64_t)a.hidden_bytes, hidden_packs);
        if (thread < a.top_k) {
          const uint64_t meta = src + (uint64_t)a.hidden_bytes;
          const uint32_t eid_u = ld_relaxed_sys(meta + 4ull * (uint64_t)thread);
          const uint32_t wt_u = ld_relaxed_sys(meta + 4ull * (uint64_t)(a.top_k + thread));
          ((int32_t *)a.topk_ids_out)[(int64_t)out_row * a.top_k + thread] = (int32_t)eid_u;
          ((float *)a.topk_weights_out)[(int64_t)out_row * a.top_k + thread] = __uint_as_float(wt_u);
        }
      }
      __syncthreads();
      if (thread == 0) {
        fence_sc_sys();
        st_relaxed_sys(bbase + a.consumed_off + 4ull * m, tag);
      }
      received += chunk;
      g += 1u;
      if (is_last) break;
    }

    if (s_poisoned == 0) {
      mask_unused_slots(a, p, received, dummy_e);
      __syncthreads();
      if (thread == 0 && ld_relaxed_sys(poison) == 0) {
        ((int32_t *)ws_recv_counts(a))[p] = received;
        if (a.recv_counts_out) ((int32_t *)a.recv_counts_out)[p] = received;
        st_release_gpu(cptr, g);
      }
    }
  }
}

__device__ __forceinline__ void ep_combine_xfer_impl(const EpParams &a) {
  const int32_t bid = (int32_t)blockIdx.x;
  const int32_t thread = (int32_t)threadIdx.x;
  const int32_t last_thread = (int32_t)blockDim.x - 1;
  const uint64_t ctrl = a.p2p_dev;
  const uint64_t poison = ctrl + 4 * kPoison;
  const uint32_t limit_us = (a.world > 1) ? ld_relaxed_sys(ctrl + 4 * kWaitLimitUs) : 0u;
  const int32_t hidden_packs = a.hidden_bytes / kPackBytes;

  if (bid < a.world) {
    /* Return processed tokens from slice [p * max_tokens .. p * max_tokens + recv_counts[p] - 1] back to peer p. */
    const int32_t p = bid;
    if (p == a.rank) return;

    __shared__ uint32_t s_poisoned;
    __shared__ uint32_t s_seq;
    __shared__ int32_t s_n_send;
    const uint64_t bbase = peer_block(a, p);
    const uint64_t cptr = chan_counter(a, 1, p);
    if (thread == 0) {
      s_seq = ld_acquire_gpu(cptr);
      s_n_send = ((const int32_t *)ws_recv_counts(a))[p];
      s_poisoned = 0;
    }
    __syncthreads();
    uint32_t g = s_seq;
    const int32_t n_send = s_n_send;
    const uint64_t pipe_bytes = (a.slot_bytes < 131072ull) ? a.slot_bytes : 131072ull;
    const int32_t toks_per_slot = max(1, (int32_t)(pipe_bytes / (uint64_t)a.hidden_bytes));

    if (n_send == 0) {
      const uint32_t m = g % a.slots;
      const uint32_t tag = g + 1u;
      if (thread == last_thread && wait_ge_or_poison(bbase + a.sent_off, tag - a.slots, poison, limit_us) == 1)
        fail(ctrl, p, kNoLane, kKindSlot, tag, 0, 0);
      __syncthreads();
      if (thread == 0) s_poisoned = ld_relaxed_sys(poison);
      __syncthreads();
      if (s_poisoned == 0) {
        if (thread == 0) {
          fence_sc_sys();
          st_relaxed_sys(bbase + a.desc_off + 4ull * m, kLast);
          fence_sc_sys();
          st_relaxed_sys(bbase + a.ready_off + 4ull * m, tag);
        }
        g += 1u;
      }
    } else {
      for (int32_t offset = 0; offset < n_send; offset += toks_per_slot) {
        const int32_t chunk = min(toks_per_slot, n_send - offset);
        const bool is_last = (offset + chunk) == n_send;
        const uint32_t m = g % a.slots;
        const uint32_t tag = g + 1u;
        if (thread == last_thread && wait_ge_or_poison(bbase + a.sent_off, tag - a.slots, poison, limit_us) == 1)
          fail(ctrl, p, kNoLane, kKindSlot, tag, 0, 0);
        __syncthreads();
        if (thread == 0) s_poisoned = ld_relaxed_sys(poison);
        __syncthreads();
        if (s_poisoned != 0) break;

        const uint64_t src = a.hidden_in + (uint64_t)(p * a.max_tokens + offset) * (uint64_t)a.hidden_bytes;
        const uint64_t dst = bbase + a.send_off + (uint64_t)m * a.slot_bytes;
        copy_packs<4, false>(src, dst, chunk * hidden_packs);
        __syncthreads();
        if (thread == 0) {
          const uint32_t hdr = ((uint32_t)chunk * (uint32_t)a.hidden_bytes) | (is_last ? kLast : 0u);
          fence_sc_sys();
          st_relaxed_sys(bbase + a.desc_off + 4ull * m, hdr);
          fence_sc_sys();
          st_relaxed_sys(bbase + a.ready_off + 4ull * m, tag);
        }
        g += 1u;
      }
    }
    __syncthreads();
    if (thread == 0 && ld_relaxed_sys(poison) == 0) st_release_gpu(cptr, g);
  } else {
    /* Receive returned partials from peer p into workspace.combine_recv[p * max_tokens ..]. */
    const int32_t p = bid - a.world;
    if (p == a.rank) return;

    __shared__ uint32_t s_poisoned;
    __shared__ uint32_t s_seq;
    __shared__ uint32_t s_hdr;
    const uint64_t bbase = peer_block(a, p);
    const uint64_t cptr = chan_counter(a, 0, p);
    const uint64_t dst_base = ws_combine_recv(a) + (uint64_t)p * (uint64_t)a.max_tokens * (uint64_t)a.hidden_bytes;
    if (thread == 0) {
      s_seq = ld_acquire_gpu(cptr);
      s_poisoned = 0;
    }
    __syncthreads();
    uint32_t g = s_seq;
    int32_t received = 0;

    for (;;) {
      const uint32_t m = g % a.slots;
      const uint32_t tag = g + 1u;
      const uint64_t line = bbase + a.flag_off + (uint64_t)m * a.lanes * kLine;
      if (thread < (int32_t)a.lanes && wait_eq_or_poison(line + (uint64_t)thread * kLine, tag, poison, limit_us) == 1)
        fail(ctrl, p, (uint32_t)thread, kKindFlag, tag, 0, 0);
      __syncthreads();
      if (thread == 0) {
        if (ld_relaxed_sys(poison) == 0) {
          const uint32_t got = ld_relaxed_sys(line + 4);
          const uint32_t cbytes = got & kBytesMask;
          if (cbytes % (uint32_t)a.hidden_bytes != 0 ||
              received + (int32_t)(cbytes / (uint32_t)a.hidden_bytes) > a.max_tokens) {
            fail(ctrl, p, kNoLane, kKindSize, tag, (uint32_t)a.max_tokens * (uint32_t)a.hidden_bytes, got);
          }
          s_hdr = got;
        }
        s_poisoned = ld_relaxed_sys(poison);
      }
      __syncthreads();
      if (s_poisoned != 0) break;

      const uint32_t cbytes = s_hdr & kBytesMask;
      const bool is_last = (s_hdr & kLast) != 0;
      const int32_t chunk = (int32_t)(cbytes / (uint32_t)a.hidden_bytes);
      if (chunk > 0) {
        const uint64_t src = bbase + a.recv_off + (uint64_t)m * a.slot_bytes;
        const uint64_t dst = dst_base + (uint64_t)received * (uint64_t)a.hidden_bytes;
        copy_packs<4, true>(src, dst, chunk * hidden_packs);
      }
      __syncthreads();
      if (thread == 0) {
        fence_sc_sys();
        st_relaxed_sys(bbase + a.consumed_off + 4ull * m, tag);
      }
      received += chunk;
      g += 1u;
      if (is_last) break;
    }

    if (s_poisoned == 0 && thread == 0 && ld_relaxed_sys(poison) == 0) st_release_gpu(cptr, g);
  }
}

template <typename DType>
__device__ __forceinline__ void ep_combine_reduce_impl(const EpParams &a) {
  const int32_t thread = (int32_t)threadIdx.x;
  const int32_t threads = (int32_t)blockDim.x;
  const int32_t hidden_packs = a.hidden_bytes / kPackBytes;
  const int32_t *send_slot_of_tok = (const int32_t *)ws_send_slot_of_tok(a);
  const uint64_t combine_recv = ws_combine_recv(a);

  __shared__ int32_t s_slot[kMaxWorld];

  for (int32_t t = (int32_t)blockIdx.x; t < a.num_tokens; t += (int32_t)gridDim.x) {
    if (thread < a.world) s_slot[thread] = send_slot_of_tok[(int64_t)thread * a.max_tokens + t];
    __syncthreads();

    const uint64_t out_row = a.hidden_out + (uint64_t)t * (uint64_t)a.hidden_bytes;
    for (int32_t pack = thread; pack < hidden_packs; pack += threads) {
      float acc[DType::kElems];
#pragma unroll
      for (int i = 0; i < DType::kElems; ++i) acc[i] = 0.0f;
      bool first = true;
      for (int32_t p = 0; p < a.world; ++p) {
        const int32_t slot = s_slot[p];
        if (slot < 0) continue;
        const uint64_t base = (p == a.rank) ? a.hidden_in : combine_recv;
        const uint64_t src =
            base + (uint64_t)(p * a.max_tokens + slot) * (uint64_t)a.hidden_bytes + (uint64_t)pack * kPackBytes;
        DType::accumulate(acc, ld_global_v4(src), first);
        first = false;
      }
      st_global_v4(out_row + (uint64_t)pack * kPackBytes, DType::store(acc));
    }
    __syncthreads();
  }
}

}  // namespace sircl_ep

extern "C" __global__ void __launch_bounds__(256, 1) sircl_ep_dispatch(const sircl_ep::EpParams a) {
  sircl_ep::ep_dispatch_impl(a);
}
extern "C" __global__ void __launch_bounds__(256, 1) sircl_ep_combine_xfer(const sircl_ep::EpParams a) {
  sircl_ep::ep_combine_xfer_impl(a);
}
extern "C" __global__ void __launch_bounds__(256, 1) sircl_ep_combine_reduce_f32(const sircl_ep::EpParams a) {
  sircl_ep::ep_combine_reduce_impl<sircl::F32>(a);
}
extern "C" __global__ void __launch_bounds__(256, 1) sircl_ep_combine_reduce_f16(const sircl_ep::EpParams a) {
  sircl_ep::ep_combine_reduce_impl<sircl::F16>(a);
}
extern "C" __global__ void __launch_bounds__(256, 1) sircl_ep_combine_reduce_bf16(const sircl_ep::EpParams a) {
  sircl_ep::ep_combine_reduce_impl<sircl::BF16>(a);
}
