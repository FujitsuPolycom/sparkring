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

/* One launch: a message's packs, n % 16, the channel's first item, the message's items, the peer's block
 * and the control line (device addresses), and the block offsets of the native layer's p2p_layout. Mirrors
 * sccl_p2p_args of src/kernelpack.h (120 bytes). */
struct P2PParams {
  uint64_t data, block_base, ctrl_base, slot_bytes;
  int32_t packs, tail, items, peer;
  uint32_t first, slots, lanes, reserved;
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

template <int U>
__device__ __forceinline__ void send(const P2PParams &a) {
  __shared__ uint32_t poisoned;
  const int32_t thread = (int32_t)threadIdx.x;
  const int32_t last_thread = (int32_t)blockDim.x - 1;
  const uint64_t poison = a.ctrl_base + 4 * kPoison;
  const uint32_t limit_us = ld_relaxed_sys(a.ctrl_base + 4 * kWaitLimitUs);
  const int32_t slot_packs = (int32_t)(a.slot_bytes / kPackBytes);
  int32_t item = (int32_t)blockIdx.x;
  while (item < a.items) {
    const uint32_t g = a.first + (uint32_t)item;
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
}

template <int U>
__device__ __forceinline__ void recv(const P2PParams &a) {
  __shared__ uint32_t poisoned;
  const int32_t thread = (int32_t)threadIdx.x;
  const uint64_t poison = a.ctrl_base + 4 * kPoison;
  const uint32_t limit_us = ld_relaxed_sys(a.ctrl_base + 4 * kWaitLimitUs);
  const int32_t slot_packs = (int32_t)(a.slot_bytes / kPackBytes);
  int32_t item = (int32_t)blockIdx.x;
  while (item < a.items) {
    const uint32_t g = a.first + (uint32_t)item;
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
