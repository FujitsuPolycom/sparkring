/*
 * Shared device code of libsircl's kernel pack: SIRCL's wire-protocol
 * constants, the ordered arena accesses, the timed flag waits, the lane-flag
 * waits with one-block polling, the doorbell and epoch steps, and the
 * float32 pack arithmetic. Every kernel of sircl_kernels.cu is built from
 * these, so all of them follow one protocol (kernels/sircl_kernels.cu
 * describes it).
 */
#ifndef SIRCL_COMMON_CUH
#define SIRCL_COMMON_CUH
#include <stdint.h>

namespace sircl {

constexpr int kPackBytes = 16;
constexpr int kSlots = 2;
constexpr int kFlagStride = 128;
constexpr int kCtrlNbytes = 1;
constexpr int kCtrlErrorSeq = 2;
constexpr int kCtrlMissingPeer = 3;
constexpr int kCtrlOpWord = 4;
constexpr int kCtrlMissingLane = 6;
constexpr int kCtrlWaitLimitUs = 7;
constexpr int kCtrlPhase1 = 10;
constexpr uint32_t kTwoshotCode = 1u << 30;
constexpr uint32_t kPollsPerClockCheck = 1024;

/* -- ordered accesses: every access to the arena is spelled out with its scope */

__device__ __forceinline__ uint32_t ld_relaxed_gpu(uint64_t a) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(uint64_t a) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(uint64_t a) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_acquire_gpu(uint64_t a) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint64_t a, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint64_t a, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint64_t a, uint32_t v) {
  uint32_t old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_global_v4(uint64_t a) {
  uint4 r;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(a) : "memory");
  return r;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(uint64_t a) {
  uint4 r;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(a) : "memory");
  return r;
}
__device__ __forceinline__ void st_global_v4(uint64_t a, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

/* Spin until the word equals `expected` (system-scope acquire loads). 0 on a
 * match; 1 after `limit_us` microseconds (checked every 1,024 polls), or, with
 * a limit of 0, after max(1, poll_limit) polls. */
__device__ __forceinline__ uint32_t spin_until_eq_timed_sys(uint64_t addr, uint32_t expected, uint32_t poll_limit,
                                                            uint32_t limit_us) {
  const bool timed = limit_us != 0;
  const uint64_t budget = (uint64_t)limit_us * 1000u;
  const uint64_t start = globaltimer();
  uint32_t polls = 0;
  for (;;) {
    if (ld_acquire_sys(addr) == expected) return 0;
    polls += 1;
    if (!timed) {
      if (polls >= poll_limit) return 1;
    } else if ((polls & (kPollsPerClockCheck - 1)) == 0) {
      if (globaltimer() - start >= budget) return 1;
    }
  }
}

/* Spin until the device word equals `expected` or the poison word is set. */
__device__ __forceinline__ uint32_t spin_until_eq_or_poison_gpu(uint64_t addr, uint32_t expected, uint64_t poison) {
  for (;;) {
    if (ld_acquire_gpu(addr) == expected) return 0;
    if (ld_relaxed_gpu(poison) != 0) return 1;
  }
}

/* -- the arithmetic of one 16-byte pack: float32 accumulation, one rounding */

__device__ __forceinline__ void unpack_bf16x2(uint32_t w, float &lo, float &hi) {
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.bf16 %0, l;\n\tcvt.f32.bf16 %1, h;\n\t}"
      : "=f"(lo), "=f"(hi) : "r"(w));
}
__device__ __forceinline__ void unpack_f16x2(uint32_t w, float &lo, float &hi) {
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, h;\n\t}"
      : "=f"(lo), "=f"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("{\n\t.reg .b16 l, h;\n\tcvt.rn.bf16.f32 l, %1;\n\tcvt.rn.bf16.f32 h, %2;\n\tmov.b32 %0, {l, h};\n\t}"
      : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ uint32_t word_of(const uint4 &v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

struct F32 {
  static constexpr int kElems = 4;
  static __device__ __forceinline__ void accumulate(float *acc, const uint4 &v, bool initialize) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float x = __uint_as_float(word_of(v, w));
      acc[w] = initialize ? x : __fadd_rn(acc[w], x);
    }
  }
  static __device__ __forceinline__ uint4 store(const float *acc) {
    return make_uint4(__float_as_uint(acc[0]), __float_as_uint(acc[1]), __float_as_uint(acc[2]),
                      __float_as_uint(acc[3]));
  }
};

template <bool kBf16>
struct Half2x4 {
  static constexpr int kElems = 8;
  static __device__ __forceinline__ void accumulate(float *acc, const uint4 &v, bool initialize) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float lo, hi;
      if (kBf16) {
        unpack_bf16x2(word_of(v, w), lo, hi);
      } else {
        unpack_f16x2(word_of(v, w), lo, hi);
      }
      acc[2 * w] = initialize ? lo : __fadd_rn(acc[2 * w], lo);
      acc[2 * w + 1] = initialize ? hi : __fadd_rn(acc[2 * w + 1], hi);
    }
  }
  static __device__ __forceinline__ uint4 store(const float *acc) {
    uint32_t p[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) p[w] = kBf16 ? pack_bf16x2(acc[2 * w], acc[2 * w + 1])
                                             : pack_f16x2(acc[2 * w], acc[2 * w + 1]);
    return make_uint4(p[0], p[1], p[2], p[3]);
  }
};
using BF16 = Half2x4<true>;
using F16 = Half2x4<false>;

/* -- flag waits ------------------------------------------------------------------------ */

/* Thread t < W * L waits for lane t % L of rank t / L in namespace `ns`. */
template <int W>
__device__ __forceinline__ void poll_lanes(int tid, uint64_t flag_base, uint64_t ctrl_base, uint64_t poison,
                                           uint64_t slot, uint32_t seq, uint32_t spin_limit, int rank, int lanes,
                                           int ns) {
  if (tid < W * lanes) {
    const int source = tid / lanes;
    const int lane = tid - source * lanes;
    if (source != rank) {
      const uint64_t line = (uint64_t)ns * W * kSlots * lanes + ((uint64_t)source * kSlots + slot) * lanes + lane;
      const uint64_t flag = flag_base + line * kFlagStride;
      const uint32_t limit_us = ld_relaxed_sys(ctrl_base + 4 * kCtrlWaitLimitUs);
      if (spin_until_eq_timed_sys(flag, seq, spin_limit, limit_us) != 0) {
        st_relaxed_sys(ctrl_base + 4 * kCtrlMissingPeer, (uint32_t)source);
        st_relaxed_sys(ctrl_base + 4 * kCtrlMissingLane, (uint32_t)lane);
        fence_sc_sys();
        st_relaxed_sys(ctrl_base + 4 * kCtrlErrorSeq, seq);
        st_release_gpu(poison, 1u);
      }
    }
  }
}

/* Every lane flag of every peer in `ns`: polled by this block, or by block 0
 * alone, which then stores `seq` into the arrival word(s) for the others. */
template <int W>
__device__ __forceinline__ void wait_lanes(int tid, int bid, uint64_t flag_base, uint64_t ctrl_base, uint64_t poison,
                                           uint64_t arrival, uint64_t slot, uint32_t seq, uint32_t spin_limit,
                                           int rank, int lanes, int one_block, int ns, bool store_both) {
  if (one_block) {
    if (bid == 0) {
      poll_lanes<W>(tid, flag_base, ctrl_base, poison, slot, seq, spin_limit, rank, lanes, ns);
      __syncthreads();
      if (tid == 0) {
        if (store_both) {
          st_release_gpu(arrival + 4, seq);
          st_release_gpu(arrival, seq);
        } else {
          st_release_gpu(arrival + 4 * ns, seq);
        }
      }
    } else if (tid == 0) {
      spin_until_eq_or_poison_gpu(arrival + 4 * ns, seq, poison);
    }
  } else {
    poll_lanes<W>(tid, flag_base, ctrl_base, poison, slot, seq, spin_limit, rank, lanes, ns);
  }
}

/* The last block to arrive publishes the epoch unless a wait timed out. */
__device__ __forceinline__ void publish_epoch(int tid, int gdim, uint64_t tail_counter, uint64_t ctrl_base,
                                              uint64_t epoch, uint32_t seq) {
  fence_sc_gpu();
  __syncthreads();
  if (tid == 0) {
    const uint32_t prior = atom_add_relaxed_gpu(tail_counter, 1u);
    if ((prior + 1u) % (uint32_t)gdim == 0) {
      fence_sc_gpu();
      if (ld_relaxed_sys(ctrl_base + 4 * kCtrlErrorSeq) == 0) st_release_gpu(epoch, seq);
    }
  }
}

/* The last block to finish staging rings the doorbell with `op_word`. */
__device__ __forceinline__ void ring_doorbell(int tid, int gdim, uint64_t stage_counter, uint64_t ctrl_base,
                                              uint64_t slot, uint32_t nbytes, uint32_t op_word, uint32_t seq) {
  if (tid == 0) {
    fence_sc_sys();
    const uint32_t prior = atom_add_relaxed_gpu(stage_counter, 1u);
    if ((prior + 1u) % (uint32_t)gdim == 0) {
      st_relaxed_sys(ctrl_base + 4 * kCtrlNbytes, nbytes);
      st_relaxed_sys(ctrl_base + 4 * kCtrlOpWord + slot * 4, op_word);
      fence_sc_sys();
      st_relaxed_sys(ctrl_base, seq);
    }
  }
}


}  // namespace sircl
#endif
