/* Notices of the CUDA floating-point headers this file includes (cuda_fp16.h, cuda_bf16.h,
 * cuda_fp8.h), which ask software using them to reproduce them in code comments; see also
 * LICENSES/CUDA-NOTICE.txt. The object code nvcc generates from those headers is distributed under
 * the NVIDIA CUDA Toolkit End User License Agreement, not under the Apache License.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 */
/*
 * The fold pack: a local, rank-ordered reduction of W gathered rows, for every NCCL datatype and
 * built-in op that the transport kernels (sircl_kernels.cu) do not reduce themselves. The rows arrive
 * through the transport pack's all-gather (all-reduce, reduce) or all-to-all (reduce-scatter); row r
 * holds rank r's elements, rows `pitch` bytes apart. The fold pack moves no data between ranks.
 *
 * Arithmetic, element by element, in rank order 0..W-1:
 * - integers: in the element type with two's-complement wrap-around (sum, prod), max, min; avg is the
 *   wrapped sum divided by W, truncated toward zero;
 * - float16, bfloat16, float8 e4m3 and e5m2: in float32, each step rounded to float32, one rounding
 *   to the element type at the end (float8 saturates to the largest finite value); avg is the float32
 *   sum divided by W in float32;
 * - float32 in float32 and float64 in float64; avg divides the sum by W in the same type.
 * Max and min compare with `>` and `<` and keep the earlier rank's value on ties.
 * Every rank folds the same rows in the same order, so every rank stores the same bits.
 *
 * Built ahead of time into kernels/prebuilt/sircl_fold.fatbin (Makefile target `kernels`), separately
 * from the transport pack so that each prebuilt file is checked against its own sources.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace sircl_fold {

enum { kSum = 0, kProd = 1, kMax = 2, kMin = 3, kAvg = 4 };

/* Integer element T folded in its unsigned twin U (defined wrap-around). */
template <class T, class U, int OP>
__device__ __forceinline__ void fold_int(const uint8_t *rows, int64_t pitch, uint8_t *out, int64_t count,
                                         int world) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
    T acc = ((const T *)rows)[i];
    for (int r = 1; r < world; ++r) {
      const T v = ((const T *)(rows + (int64_t)r * pitch))[i];
      if (OP == kSum || OP == kAvg) acc = (T)((U)acc + (U)v);
      else if (OP == kProd) acc = (T)((U)acc * (U)v);
      else if (OP == kMax) acc = v > acc ? v : acc;
      else acc = v < acc ? v : acc;
    }
    if (OP == kAvg) acc = (T)(acc / (T)world);
    ((T *)out)[i] = acc;
  }
}

template <class T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) { return (float)v; }
template <> __device__ __forceinline__ float to_f32<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) { return (float)v; }
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }

template <class T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __nv_fp8_e4m3 from_f32<__nv_fp8_e4m3>(float v) { return __nv_fp8_e4m3(v); }
template <> __device__ __forceinline__ __nv_fp8_e5m2 from_f32<__nv_fp8_e5m2>(float v) { return __nv_fp8_e5m2(v); }
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }

/* Floating element T folded in float32. */
template <class T, int OP>
__device__ __forceinline__ void fold_f32(const uint8_t *rows, int64_t pitch, uint8_t *out, int64_t count,
                                         int world) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
    float acc = to_f32<T>(((const T *)rows)[i]);
    for (int r = 1; r < world; ++r) {
      const float v = to_f32<T>(((const T *)(rows + (int64_t)r * pitch))[i]);
      if (OP == kSum || OP == kAvg) acc = __fadd_rn(acc, v);
      else if (OP == kProd) acc = __fmul_rn(acc, v);
      else if (OP == kMax) acc = v > acc ? v : acc;
      else acc = v < acc ? v : acc;
    }
    if (OP == kAvg) acc = __fdiv_rn(acc, (float)world);
    ((T *)out)[i] = from_f32<T>(acc);
  }
}

template <int OP>
__device__ __forceinline__ void fold_f64(const uint8_t *rows, int64_t pitch, uint8_t *out, int64_t count,
                                         int world) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
    double acc = ((const double *)rows)[i];
    for (int r = 1; r < world; ++r) {
      const double v = ((const double *)(rows + (int64_t)r * pitch))[i];
      if (OP == kSum || OP == kAvg) acc = __dadd_rn(acc, v);
      else if (OP == kProd) acc = __dmul_rn(acc, v);
      else if (OP == kMax) acc = v > acc ? v : acc;
      else acc = v < acc ? v : acc;
    }
    if (OP == kAvg) acc = __ddiv_rn(acc, (double)world);
    ((double *)out)[i] = acc;
  }
}

}  // namespace sircl_fold

/* Entry points sircl_fold_d<datatype>_o<op>: datatype and op are NCCL's enum values (ncclInt8 = 0 ...
 * ncclFloat8e5m2 = 11; ncclSum = 0 ... ncclAvg = 4). Parameters: the device address of row 0, the row
 * pitch in bytes, the output address, the element count and the number of rows. Rows and output are
 * aligned to the element size. */
#define SIRCL_FOLD_ENTRY(DT, OPN, BODY)                                                                         \
  extern "C" __global__ void __launch_bounds__(1024) sircl_fold_d##DT##_o##OPN(uint64_t rows, int64_t pitch,   \
                                                                             uint64_t out, int64_t count,      \
                                                                             int32_t world) {                   \
    BODY((const uint8_t *)rows, pitch, (uint8_t *)out, count, world);                                          \
  }
#define SIRCL_FOLD_OPS(DT, FOLD)                    \
  SIRCL_FOLD_ENTRY(DT, 0, FOLD(sircl_fold::kSum))  \
  SIRCL_FOLD_ENTRY(DT, 1, FOLD(sircl_fold::kProd)) \
  SIRCL_FOLD_ENTRY(DT, 2, FOLD(sircl_fold::kMax))  \
  SIRCL_FOLD_ENTRY(DT, 3, FOLD(sircl_fold::kMin))  \
  SIRCL_FOLD_ENTRY(DT, 4, FOLD(sircl_fold::kAvg))
#define SIRCL_FOLD_I8(OP) sircl_fold::fold_int<int8_t, uint8_t, OP>
#define SIRCL_FOLD_U8(OP) sircl_fold::fold_int<uint8_t, uint8_t, OP>
#define SIRCL_FOLD_I32(OP) sircl_fold::fold_int<int32_t, uint32_t, OP>
#define SIRCL_FOLD_U32(OP) sircl_fold::fold_int<uint32_t, uint32_t, OP>
#define SIRCL_FOLD_I64(OP) sircl_fold::fold_int<int64_t, uint64_t, OP>
#define SIRCL_FOLD_U64(OP) sircl_fold::fold_int<uint64_t, uint64_t, OP>
#define SIRCL_FOLD_F16(OP) sircl_fold::fold_f32<__half, OP>
#define SIRCL_FOLD_F32(OP) sircl_fold::fold_f32<float, OP>
#define SIRCL_FOLD_F64(OP) sircl_fold::fold_f64<OP>
#define SIRCL_FOLD_BF16(OP) sircl_fold::fold_f32<__nv_bfloat16, OP>
#define SIRCL_FOLD_E4M3(OP) sircl_fold::fold_f32<__nv_fp8_e4m3, OP>
#define SIRCL_FOLD_E5M2(OP) sircl_fold::fold_f32<__nv_fp8_e5m2, OP>

SIRCL_FOLD_OPS(0, SIRCL_FOLD_I8)
SIRCL_FOLD_OPS(1, SIRCL_FOLD_U8)
SIRCL_FOLD_OPS(2, SIRCL_FOLD_I32)
SIRCL_FOLD_OPS(3, SIRCL_FOLD_U32)
SIRCL_FOLD_OPS(4, SIRCL_FOLD_I64)
SIRCL_FOLD_OPS(5, SIRCL_FOLD_U64)
SIRCL_FOLD_OPS(6, SIRCL_FOLD_F16)
SIRCL_FOLD_OPS(7, SIRCL_FOLD_F32)
SIRCL_FOLD_OPS(8, SIRCL_FOLD_F64)
SIRCL_FOLD_OPS(9, SIRCL_FOLD_BF16)
SIRCL_FOLD_OPS(10, SIRCL_FOLD_E4M3)
SIRCL_FOLD_OPS(11, SIRCL_FOLD_E5M2)
