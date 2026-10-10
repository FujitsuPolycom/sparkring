/*
 * libsircl's kernel pack: SIRCL ring-session collectives in CUDA C++.
 *
 * These kernels speak the wire protocol of SIRCL's ring sessions (the SIRCL
 * package's protocol.py, oneshot/_roce_proxy.c) byte for byte, so a rank that
 * launches them interoperates with ranks that launch SIRCL's CuTe DSL kernels
 * (oneshot/_oneshot_cute.py, _twoshot_cute.py, _allgather_cute.py,
 * _scatter_cute.py) and with SIRCL's native progress thread. They are
 * compiled ahead of time; the group size and dtype are template parameters,
 * the rank, the lane count and the polling mode are launch arguments.
 *
 * Arena (pinned host memory the GPU reads and writes with system-scope
 * accesses; offsets from protocol.ArenaLayout):
 *   recv[source][slot]  world * SLOTS * slot_bytes     written by peers' NICs
 *   flag lines          world * SLOTS * 4 * 128 bytes  written by peers' NICs
 *   send[slot]          SLOTS * slot_bytes             staged by this kernel
 *   control             128 bytes                      the command ring
 * The slot of op `seq` is `seq & 1`. Flag line of (namespace, source, slot,
 * lane): ns * W * SLOTS * L + (source * SLOTS + slot) * L + lane.
 *
 * Device words (protocol.CounterLayout, int32): the epoch (newest completed
 * sequence), one stage and one tail arrival counter per power-of-two grid
 * class, the poison word, one phase-1 counter per grid class; and two arrival
 * words for one-block polling (namespace 0, namespace 1).
 *
 * One-shot all-reduce (op code 0): stage the input into send[slot]; the last
 * block to finish staging writes the byte count (command ring word 1 and the
 * slot's op word) and then `seq` into the doorbell; wait for every lane flag
 * of every peer in namespace 0; sum the own input and every peer's slot in
 * rank order 0..W-1 in float32 and round once; the last block to finish
 * publishes the epoch unless a wait timed out.
 *
 * Two-shot all-reduce (op code 1): chunk j of P packs is
 * [floor(j P / W), floor((j + 1) P / W)). Stage every chunk but the own one;
 * doorbell with op code 1; wait for namespace-0 flags; reduce the own chunk in
 * rank order into the output and into send[slot]; the last block to finish
 * writes `seq` into the phase-1 doorbell (word 10); wait for namespace-1 flags
 * and copy every peer's reduced chunk into the output; publish the epoch.
 *
 * Both algorithms sum the same values in the same order with one rounding,
 * so their bits are equal and independent of how a message is split.
 *
 * All-gather (op code 0, one tile): pack i of an op is column i % tile_cols
 * of tile row i / tile_cols; it is read from the input at
 * row * in_row_stride + col packs, travels compactly at pack i of the slot,
 * and shard s lands in the output at row * out_row_stride + s * out_src_stride
 * + col packs. A contiguous shard of P packs is the tile of one row:
 * tile_cols = in_row_stride = out_src_stride = P, out_row_stride = W P.
 * Bytes are copied unchanged.
 *
 * Scatter op (op code 3; the reduce-scatter and the all-to-all): W chunks of
 * chunk_packs packs, chunk j of the input at j * src_stride bytes. Stage
 * every chunk but the own one into its pack range of send[slot]; doorbell;
 * the progress thread writes chunk p to rank p; every block waits for every
 * peer's lane flags (namespace 0); then reduce the own chunk in rank order in
 * float32, rounded once (the all-reduce's bits for those elements), into the
 * output, or copy the chunk of source s to output + s * dst_stride; publish
 * the epoch. No later network phase.
 *
 * A flag wait lasts at most the wait limit in command ring word 7
 * (microseconds of %globaltimer, checked every 1,024 polls; 0 selects a budget
 * of `spin_limit` polls). A timed-out wait writes the missing peer and lane
 * (words 3 and 6), then the sequence (word 2), and sets the poison word; later
 * launches on a poisoned session do nothing.
 *
 * With one-block polling (all-reduce and all-gather) only block 0 polls the
 * flags in host memory; it then stores `seq` into the arrival word of the
 * namespace (the one-shot all-reduce and the all-gather store both words) and
 * the other blocks wait for that word in device memory.
 *
 * Derived from the SIRCL ring-session kernels, which derive from b12x
 * RoCEnante (Apache-2.0, Local Inference Lab); see NOTICE.
 */
#include <type_traits>

#include "sircl_common.cuh"

namespace sircl {

constexpr uint32_t kScatterCode = 3u << 30;

template <class D, int W>
__device__ __forceinline__ void oneshot(uint64_t input, uint64_t output, int32_t size_packs, int32_t nbytes,
                                        uint64_t recv_base, uint64_t flag_base, uint64_t send_base,
                                        uint64_t ctrl_base, uint64_t slot_bytes, uint64_t epoch_ptr,
                                        uint64_t stage_counter, uint64_t tail_counter, uint64_t poison,
                                        uint64_t arrival, uint32_t spin_limit, int32_t rank, int32_t lanes,
                                        int32_t one_block) {
  const int tid = (int)threadIdx.x, bid = (int)blockIdx.x, gdim = (int)gridDim.x;
  /* Every block reads the epoch before any block can advance it: the advance
   * happens only after every block arrived at the tail counter. */
  const uint32_t seq = ld_relaxed_gpu(epoch_ptr) + 1u;
  const uint64_t slot = seq & (uint32_t)(kSlots - 1);
  const uint64_t send_slot = send_base + slot * slot_bytes;
  const int32_t index = bid * (int32_t)blockDim.x + tid;
  const int32_t stride = gdim * (int32_t)blockDim.x;
  if (ld_relaxed_gpu(poison) != 0) return;

  for (int32_t i = index; i < size_packs; i += stride) {
    st_global_v4(send_slot + (uint64_t)i * kPackBytes, ld_global_v4(input + (uint64_t)i * kPackBytes));
  }
  __syncthreads();
  ring_doorbell(tid, gdim, stage_counter, ctrl_base, slot, (uint32_t)nbytes, (uint32_t)nbytes, seq);
  wait_lanes<W>(tid, bid, flag_base, ctrl_base, poison, arrival, slot, seq, spin_limit, rank, lanes, one_block, 0,
                true);
  __syncthreads();
  /* A timed-out wait leaves a peer slot unreliable: nothing derived from it is stored. */
  if (ld_relaxed_gpu(poison) == 0) {
    for (int32_t i = index; i < size_packs; i += stride) {
      float acc[D::kElems];
      const uint64_t offset = (uint64_t)i * kPackBytes;
#pragma unroll
      for (int src = 0; src < W; ++src) {
        const uint4 words = src == rank
                                ? ld_global_v4(input + offset)
                                : ld_relaxed_sys_v4(recv_base + ((uint64_t)src * kSlots + slot) * slot_bytes + offset);
        D::accumulate(acc, words, src == 0);
      }
      st_global_v4(output + offset, D::store(acc));
    }
  }
  publish_epoch(tid, gdim, tail_counter, ctrl_base, epoch_ptr, seq);
}

template <class D, int W>
__device__ __forceinline__ void twoshot(uint64_t input, uint64_t output, int32_t size_packs, int32_t nbytes,
                                        uint64_t recv_base, uint64_t flag_base, uint64_t send_base,
                                        uint64_t ctrl_base, uint64_t slot_bytes, uint64_t epoch_ptr,
                                        uint64_t stage_counter, uint64_t phase_counter, uint64_t tail_counter,
                                        uint64_t poison, uint64_t arrival, uint32_t spin_limit, int32_t rank,
                                        int32_t lanes, int32_t one_block) {
  const int tid = (int)threadIdx.x, bid = (int)blockIdx.x, gdim = (int)gridDim.x;
  const uint32_t seq = ld_relaxed_gpu(epoch_ptr) + 1u;
  const uint64_t slot = seq & (uint32_t)(kSlots - 1);
  const uint64_t send_slot = send_base + slot * slot_bytes;
  const int32_t index = bid * (int32_t)blockDim.x + tid;
  const int32_t stride = gdim * (int32_t)blockDim.x;
  const int32_t own_first = (int32_t)((int64_t)rank * size_packs / W);
  const int32_t own_end = (int32_t)((int64_t)(rank + 1) * size_packs / W);
  const int32_t own_packs = own_end - own_first;
  if (ld_relaxed_gpu(poison) != 0) return;

  /* 1. stage every chunk but the own one */
  for (int32_t i = index; i < own_first; i += stride) {
    st_global_v4(send_slot + (uint64_t)i * kPackBytes, ld_global_v4(input + (uint64_t)i * kPackBytes));
  }
  for (int32_t i = own_end + index; i < size_packs; i += stride) {
    st_global_v4(send_slot + (uint64_t)i * kPackBytes, ld_global_v4(input + (uint64_t)i * kPackBytes));
  }
  __syncthreads();
  /* 2. doorbell with op code 1 */
  ring_doorbell(tid, gdim, stage_counter, ctrl_base, slot, (uint32_t)nbytes, kTwoshotCode | (uint32_t)nbytes, seq);
  /* 3. every peer's contribution to the own chunk (namespace 0) */
  wait_lanes<W>(tid, bid, flag_base, ctrl_base, poison, arrival, slot, seq, spin_limit, rank, lanes, one_block, 0,
                false);
  __syncthreads();
  if (ld_relaxed_gpu(poison) == 0) {
    /* 4. reduce the own chunk in rank order into the output and send[slot] */
    for (int32_t i = index; i < own_packs; i += stride) {
      float acc[D::kElems];
      const uint64_t offset = (uint64_t)(own_first + i) * kPackBytes;
#pragma unroll
      for (int src = 0; src < W; ++src) {
        const uint4 words = src == rank
                                ? ld_global_v4(input + offset)
                                : ld_relaxed_sys_v4(recv_base + ((uint64_t)src * kSlots + slot) * slot_bytes + offset);
        D::accumulate(acc, words, src == 0);
      }
      const uint4 packed = D::store(acc);
      st_global_v4(output + offset, packed);
      st_global_v4(send_slot + offset, packed);
    }
    __syncthreads();
    /* 5. the last block to finish reducing releases phase 1 */
    if (tid == 0) {
      fence_sc_sys();
      const uint32_t prior = atom_add_relaxed_gpu(phase_counter, 1u);
      if ((prior + 1u) % (uint32_t)gdim == 0) {
        fence_sc_sys();
        st_relaxed_sys(ctrl_base + 4 * kCtrlPhase1, seq);
      }
    }
    /* 6. every peer's reduced chunk (namespace 1) */
    wait_lanes<W>(tid, bid, flag_base, ctrl_base, poison, arrival, slot, seq, spin_limit, rank, lanes, one_block, 1,
                  false);
    __syncthreads();
    if (ld_relaxed_gpu(poison) == 0) {
#pragma unroll 1
      for (int src = 0; src < W; ++src) {
        if (src == rank) continue;
        const int32_t first = (int32_t)((int64_t)src * size_packs / W);
        const int32_t end = (int32_t)((int64_t)(src + 1) * size_packs / W);
        const uint64_t peer_slot = recv_base + ((uint64_t)src * kSlots + slot) * slot_bytes;
        for (int32_t i = index; i < end - first; i += stride) {
          const uint64_t offset = (uint64_t)(first + i) * kPackBytes;
          st_global_v4(output + offset, ld_relaxed_sys_v4(peer_slot + offset));
        }
      }
    }
  }
  /* 7. the last block to finish publishes the epoch */
  publish_epoch(tid, gdim, tail_counter, ctrl_base, epoch_ptr, seq);
}


/* One tile of an all-gather. */
template <int W>
__device__ __forceinline__ void allgather(uint64_t input, uint64_t output, int32_t shard_packs, int32_t nbytes,
                                          int32_t tile_cols, int64_t in_row_stride, int64_t out_row_stride,
                                          int64_t out_src_stride, uint64_t recv_base, uint64_t flag_base,
                                          uint64_t send_base, uint64_t ctrl_base, uint64_t slot_bytes,
                                          uint64_t epoch_ptr, uint64_t stage_counter, uint64_t tail_counter,
                                          uint64_t poison, uint64_t arrival, uint32_t spin_limit, int32_t rank,
                                          int32_t lanes, int32_t one_block) {
  const int tid = (int)threadIdx.x, bid = (int)blockIdx.x, gdim = (int)gridDim.x;
  const uint32_t seq = ld_relaxed_gpu(epoch_ptr) + 1u;
  const uint64_t slot = seq & (uint32_t)(kSlots - 1);
  const uint64_t send_slot = send_base + slot * slot_bytes;
  const int32_t index = bid * (int32_t)blockDim.x + tid;
  const int32_t stride = gdim * (int32_t)blockDim.x;
  if (ld_relaxed_gpu(poison) != 0) return;

  /* 1. stage the local tile compactly into the pinned send slot */
  for (int32_t i = index; i < shard_packs; i += stride) {
    const int32_t row = i / tile_cols, col = i - row * tile_cols;
    st_global_v4(send_slot + (uint64_t)i * kPackBytes,
                 ld_global_v4(input + (uint64_t)((int64_t)row * in_row_stride + col) * kPackBytes));
  }
  __syncthreads();
  ring_doorbell(tid, gdim, stage_counter, ctrl_base, slot, (uint32_t)nbytes, (uint32_t)nbytes, seq);
  wait_lanes<W>(tid, bid, flag_base, ctrl_base, poison, arrival, slot, seq, spin_limit, rank, lanes, one_block, 0,
                true);
  __syncthreads();
  if (ld_relaxed_gpu(poison) == 0) {
    /* 4. shard s of tile row r lands at r * out_row_stride + s * out_src_stride */
#pragma unroll 1
    for (int src = 0; src < W; ++src) {
      const uint64_t peer_slot = recv_base + ((uint64_t)src * kSlots + slot) * slot_bytes;
      for (int32_t i = index; i < shard_packs; i += stride) {
        const int32_t row = i / tile_cols, col = i - row * tile_cols;
        const uint64_t dest =
            output + (uint64_t)((int64_t)row * out_row_stride + (int64_t)src * out_src_stride + col) * kPackBytes;
        const uint4 words =
            src == rank ? ld_global_v4(input + (uint64_t)((int64_t)row * in_row_stride + col) * kPackBytes)
                        : ld_relaxed_sys_v4(peer_slot + (uint64_t)i * kPackBytes);
        st_global_v4(dest, words);
      }
    }
  }
  publish_epoch(tid, gdim, tail_counter, ctrl_base, epoch_ptr, seq);
}

/* One scatter op: reduce (D a dtype) or copy (D = void). Every block polls the flags. */
template <class D, int W>
__device__ __forceinline__ void scatter(uint64_t input, uint64_t output, int32_t size_packs, int32_t nbytes,
                                        int32_t chunk_packs, int64_t src_stride, int64_t dst_stride,
                                        uint64_t recv_base, uint64_t flag_base, uint64_t send_base,
                                        uint64_t ctrl_base, uint64_t slot_bytes, uint64_t epoch_ptr,
                                        uint64_t stage_counter, uint64_t tail_counter, uint64_t poison,
                                        uint32_t spin_limit, int32_t rank, int32_t lanes) {
  const int tid = (int)threadIdx.x, bid = (int)blockIdx.x, gdim = (int)gridDim.x;
  const uint32_t seq = ld_relaxed_gpu(epoch_ptr) + 1u;
  const uint64_t slot = seq & (uint32_t)(kSlots - 1);
  const uint64_t send_slot = send_base + slot * slot_bytes;
  const int32_t index = bid * (int32_t)blockDim.x + tid;
  const int32_t stride = gdim * (int32_t)blockDim.x;
  const int32_t own_lo = rank * chunk_packs, own_hi = own_lo + chunk_packs;
  /* The own chunk's pack q is at input + rank * src_stride + q * 16. */
  const uint64_t input_own = input + (uint64_t)((int64_t)rank * src_stride);
  if (ld_relaxed_gpu(poison) != 0) return;

  /* 1. stage every chunk but the own one into its pack range of the send slot */
  for (int32_t i = index; i < size_packs; i += stride) {
    const int32_t chunk = i / chunk_packs, within = i - chunk * chunk_packs;
    if (chunk != rank)
      st_global_v4(send_slot + (uint64_t)i * kPackBytes,
                   ld_global_v4(input + (uint64_t)((int64_t)chunk * src_stride) + (uint64_t)within * kPackBytes));
  }
  __syncthreads();
  ring_doorbell(tid, gdim, stage_counter, ctrl_base, slot, (uint32_t)nbytes, kScatterCode | (uint32_t)nbytes, seq);
  /* 3. every peer's stripes of the own chunk (namespace 0), polled by every block */
  poll_lanes<W>(tid, flag_base, ctrl_base, poison, slot, seq, spin_limit, rank, lanes, 0);
  __syncthreads();
  if (ld_relaxed_gpu(poison) == 0) {
    if constexpr (!std::is_void<D>::value) {
      for (int32_t i = own_lo + index; i < own_hi; i += stride) {
        float acc[D::kElems];
        const uint64_t q = (uint64_t)(i - own_lo) * kPackBytes;
#pragma unroll
        for (int src = 0; src < W; ++src) {
          const uint4 words = src == rank ? ld_global_v4(input_own + q)
                                          : ld_relaxed_sys_v4(recv_base + ((uint64_t)src * kSlots + slot) * slot_bytes +
                                                              (uint64_t)i * kPackBytes);
          D::accumulate(acc, words, src == 0);
        }
        st_global_v4(output + q, D::store(acc));
      }
    } else {
      for (int32_t i = index; i < chunk_packs; i += stride) {
#pragma unroll
        for (int src = 0; src < W; ++src) {
          const uint4 words =
              src == rank ? ld_global_v4(input_own + (uint64_t)i * kPackBytes)
                          : ld_relaxed_sys_v4(recv_base + ((uint64_t)src * kSlots + slot) * slot_bytes +
                                              (uint64_t)(own_lo + i) * kPackBytes);
          st_global_v4(output + (uint64_t)((int64_t)src * dst_stride) + (uint64_t)i * kPackBytes, words);
        }
      }
    }
  }
  publish_epoch(tid, gdim, tail_counter, ctrl_base, epoch_ptr, seq);
}

}  // namespace sircl

/* Entry points, W = 2..8, dtype f32, f16 or bf16: sircl_oneshot_<dtype>_w<W>,
 * sircl_twoshot_<dtype>_w<W>, sircl_allgather_w<W>, sircl_scatter_<dtype>_w<W>
 * (reduce-scatter) and sircl_alltoall_w<W>. The C launcher in src/kernelpack.c
 * builds their parameters in the order of the argument lists below. */
#define SIRCL_ONESHOT_ENTRY(NAME, D, W)                                                                          \
  extern "C" __global__ void __launch_bounds__(1024) NAME(                                                       \
      uint64_t input, uint64_t output, int32_t size_packs, int32_t nbytes, uint64_t recv_base, uint64_t flag_base, \
      uint64_t send_base, uint64_t ctrl_base, uint64_t slot_bytes, uint64_t epoch_ptr, uint64_t stage_counter,    \
      uint64_t tail_counter, uint64_t poison, uint64_t arrival, uint32_t spin_limit, int32_t rank, int32_t lanes, \
      int32_t one_block) {                                                                                       \
    sircl::oneshot<D, W>(input, output, size_packs, nbytes, recv_base, flag_base, send_base, ctrl_base,          \
                         slot_bytes, epoch_ptr, stage_counter, tail_counter, poison, arrival, spin_limit, rank,  \
                         lanes, one_block);                                                                      \
  }
#define SIRCL_TWOSHOT_ENTRY(NAME, D, W)                                                                          \
  extern "C" __global__ void __launch_bounds__(1024) NAME(                                                       \
      uint64_t input, uint64_t output, int32_t size_packs, int32_t nbytes, uint64_t recv_base, uint64_t flag_base, \
      uint64_t send_base, uint64_t ctrl_base, uint64_t slot_bytes, uint64_t epoch_ptr, uint64_t stage_counter,    \
      uint64_t phase_counter, uint64_t tail_counter, uint64_t poison, uint64_t arrival, uint32_t spin_limit,     \
      int32_t rank, int32_t lanes, int32_t one_block) {                                                          \
    sircl::twoshot<D, W>(input, output, size_packs, nbytes, recv_base, flag_base, send_base, ctrl_base,          \
                         slot_bytes, epoch_ptr, stage_counter, phase_counter, tail_counter, poison, arrival,     \
                         spin_limit, rank, lanes, one_block);                                                    \
  }
#define SIRCL_ENTRIES_FOR(D, TAG, W)                         \
  SIRCL_ONESHOT_ENTRY(sircl_oneshot_##TAG##_w##W, D, W)      \
  SIRCL_TWOSHOT_ENTRY(sircl_twoshot_##TAG##_w##W, D, W)
#define SIRCL_ENTRIES(W)                       \
  SIRCL_ENTRIES_FOR(sircl::F32, f32, W)        \
  SIRCL_ENTRIES_FOR(sircl::F16, f16, W)        \
  SIRCL_ENTRIES_FOR(sircl::BF16, bf16, W)

SIRCL_ENTRIES(2)
SIRCL_ENTRIES(3)
SIRCL_ENTRIES(4)
SIRCL_ENTRIES(5)
SIRCL_ENTRIES(6)
SIRCL_ENTRIES(7)
SIRCL_ENTRIES(8)

#define SIRCL_ALLGATHER_ENTRY(W)                                                                                   \
  extern "C" __global__ void __launch_bounds__(1024) sircl_allgather_w##W(                                         \
      uint64_t input, uint64_t output, int32_t shard_packs, int32_t nbytes, int32_t tile_cols, int64_t in_row_stride, \
      int64_t out_row_stride, int64_t out_src_stride, uint64_t recv_base, uint64_t flag_base, uint64_t send_base,  \
      uint64_t ctrl_base, uint64_t slot_bytes, uint64_t epoch_ptr, uint64_t stage_counter, uint64_t tail_counter,  \
      uint64_t poison, uint64_t arrival, uint32_t spin_limit, int32_t rank, int32_t lanes, int32_t one_block) {    \
    sircl::allgather<W>(input, output, shard_packs, nbytes, tile_cols, in_row_stride, out_row_stride,             \
                        out_src_stride, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_ptr,         \
                        stage_counter, tail_counter, poison, arrival, spin_limit, rank, lanes, one_block);        \
  }
#define SIRCL_SCATTER_ENTRY(NAME, D, W)                                                                            \
  extern "C" __global__ void __launch_bounds__(1024) NAME(                                                         \
      uint64_t input, uint64_t output, int32_t size_packs, int32_t nbytes, int32_t chunk_packs, int64_t src_stride, \
      int64_t dst_stride, uint64_t recv_base, uint64_t flag_base, uint64_t send_base, uint64_t ctrl_base,          \
      uint64_t slot_bytes, uint64_t epoch_ptr, uint64_t stage_counter, uint64_t tail_counter, uint64_t poison,     \
      uint32_t spin_limit, int32_t rank, int32_t lanes) {                                                          \
    sircl::scatter<D, W>(input, output, size_packs, nbytes, chunk_packs, src_stride, dst_stride, recv_base,       \
                         flag_base, send_base, ctrl_base, slot_bytes, epoch_ptr, stage_counter, tail_counter,     \
                         poison, spin_limit, rank, lanes);                                                         \
  }
#define SIRCL_GATHER_SCATTER_ENTRIES(W)                              \
  SIRCL_ALLGATHER_ENTRY(W)                                           \
  SIRCL_SCATTER_ENTRY(sircl_scatter_f32_w##W, sircl::F32, W)         \
  SIRCL_SCATTER_ENTRY(sircl_scatter_f16_w##W, sircl::F16, W)         \
  SIRCL_SCATTER_ENTRY(sircl_scatter_bf16_w##W, sircl::BF16, W)       \
  SIRCL_SCATTER_ENTRY(sircl_alltoall_w##W, void, W)

SIRCL_GATHER_SCATTER_ENTRIES(2)
SIRCL_GATHER_SCATTER_ENTRIES(3)
SIRCL_GATHER_SCATTER_ENTRIES(4)
SIRCL_GATHER_SCATTER_ENTRIES(5)
SIRCL_GATHER_SCATTER_ENTRIES(6)
SIRCL_GATHER_SCATTER_ENTRIES(7)
SIRCL_GATHER_SCATTER_ENTRIES(8)
