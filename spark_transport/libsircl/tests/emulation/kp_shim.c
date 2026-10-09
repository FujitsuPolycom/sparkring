/* Test entry points to the kernel pack for the mixed-group check
 * (tests/emulation/mixed_group.py): load into the calling thread's current
 * context and launch with explicit arguments. Built into a test-only library;
 * the production library launches through its engine instead. */
#include "kernelpack.h"

#include <stdint.h>

__attribute__((visibility("default"))) int sirclkp_load(void) {
  const sccl_cuda *cu = sccl_cuda_get();
  if (!cu) return -1;
  sccl_CUcontext ctx = NULL;
  if (cu->CtxGetCurrent(&ctx) != SCCL_CUDA_SUCCESS || !ctx) return -2;
  return sccl_kp_load(ctx);
}

__attribute__((visibility("default"))) const char *sirclkp_error(void) { return sccl_kp_error(); }

__attribute__((visibility("default"))) const char *sirclkp_hash(void) { return sccl_kp_hash(); }

__attribute__((visibility("default"))) int sirclkp_launch(int algorithm, int dtype, int world, unsigned grid,
                                                          unsigned threads, uint64_t stream,
                                                          const sccl_allreduce_args *args) {
  const sccl_cuda *cu = sccl_cuda_get();
  if (!cu) return -1;
  sccl_CUcontext ctx = NULL;
  if (cu->CtxGetCurrent(&ctx) != SCCL_CUDA_SUCCESS || !ctx) return -2;
  sccl_CUfunction fn;
  if (sccl_kp_function(ctx, algorithm, dtype, world, &fn) != 0) return -3;
  return (int)sccl_kp_launch_allreduce(fn, algorithm, args, grid, threads, (sccl_CUstream)(uintptr_t)stream);
}

static int current_function(int kind, int dtype, int world, sccl_CUfunction *fn) {
  const sccl_cuda *cu = sccl_cuda_get();
  if (!cu) return -1;
  sccl_CUcontext ctx = NULL;
  if (cu->CtxGetCurrent(&ctx) != SCCL_CUDA_SUCCESS || !ctx) return -2;
  return sccl_kp_function(ctx, kind, dtype, world, fn) ? -3 : 0;
}

__attribute__((visibility("default"))) int sirclkp_launch_allgather(int world, unsigned grid, unsigned threads,
                                                                    uint64_t stream, const sccl_allgather_args *args) {
  sccl_CUfunction fn;
  int rc = current_function(SCCL_K_ALLGATHER, 0, world, &fn);
  return rc ? rc : (int)sccl_kp_launch_allgather(fn, args, grid, threads, (sccl_CUstream)(uintptr_t)stream);
}

__attribute__((visibility("default"))) const char *sirclkp_links_hash(void) { return sccl_kp_links_hash(); }

/* One chain all-reduce op of a dtype and unroll, 4 * blocks_per_role blocks. */
__attribute__((visibility("default"))) int sirclkp_launch_chain(int dtype, int unroll, unsigned grid, unsigned threads,
                                                                uint64_t stream, const sccl_chain_args *args) {
  const sccl_cuda *cu = sccl_cuda_get();
  if (!cu) return -1;
  sccl_CUcontext ctx = NULL;
  if (cu->CtxGetCurrent(&ctx) != SCCL_CUDA_SUCCESS || !ctx) return -2;
  sccl_CUfunction fn;
  if (sccl_kp_chain_function(ctx, dtype, unroll, &fn) != 0) return -3;
  return (int)sccl_kp_launch_chain(fn, args, grid, threads, (sccl_CUstream)(uintptr_t)stream);
}

/* One link collective op of a kind (SCCL_LINK_*, SCCL_RING_*; SCCL_LINK_KINDS: the two-pass ring
 * all-reduce), dtype and unroll. */
__attribute__((visibility("default"))) int sirclkp_launch_link(int kind, int dtype, int unroll, unsigned grid,
                                                               unsigned threads, uint64_t stream,
                                                               const sccl_link_args *args) {
  const sccl_cuda *cu = sccl_cuda_get();
  if (!cu) return -1;
  sccl_CUcontext ctx = NULL;
  if (cu->CtxGetCurrent(&ctx) != SCCL_CUDA_SUCCESS || !ctx) return -2;
  sccl_CUfunction fn;
  if (kind == SCCL_LINK_KINDS ? sccl_kp_ring_reduce_two_pass_function(ctx, dtype, unroll, &fn) != 0
                               : sccl_kp_link_function(ctx, kind, dtype, unroll, &fn) != 0)
    return -3;
  return (int)sccl_kp_launch_link(fn, args, grid, threads, (sccl_CUstream)(uintptr_t)stream);
}

/* dtype -1: the all-to-all (copy); else the reduce-scatter of that dtype. */
__attribute__((visibility("default"))) int sirclkp_launch_scatter(int dtype, int world, unsigned grid, unsigned threads,
                                                                  uint64_t stream, const sccl_scatter_args *args) {
  sccl_CUfunction fn;
  int rc = current_function(dtype < 0 ? SCCL_K_ALLTOALL : SCCL_K_SCATTER, dtype < 0 ? 0 : dtype, world, &fn);
  return rc ? rc : (int)sccl_kp_launch_scatter(fn, args, grid, threads, (sccl_CUstream)(uintptr_t)stream);
}
