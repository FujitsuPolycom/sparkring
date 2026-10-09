/* Host cost of one kernel-pack launch from C (driver API), measured on a poisoned session so every
 * launch returns at once: N launches back to back on one stream, then N in a burst after a sync.
 * Build against the library's kernelpack.c, cuda_api.c and the embedded fatbin. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernelpack.h"

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 20000;
  const sccl_cuda *cu = sccl_cuda_get();
  if (!cu) { fprintf(stderr, "%s\n", sccl_cuda_error()); return 1; }
  sccl_CUdevice dev; sccl_CUcontext ctx;
  cu->DeviceGet(&dev, 0);
  cu->DevicePrimaryCtxRetain(&ctx, dev);
  cu->CtxSetCurrent(ctx);
  if (sccl_kp_load(ctx)) { fprintf(stderr, "%s\n", sccl_kp_error()); return 1; }
  sccl_CUfunction fn[2];
  sccl_kp_function(ctx, SCCL_ALG_ONESHOT, SCCL_DT_BF16, 2, &fn[0]);
  sccl_kp_function(ctx, SCCL_ALG_TWOSHOT, SCCL_DT_BF16, 2, &fn[1]);
  sccl_CUdeviceptr words;
  cu->MemAlloc(&words, 4096);
  unsigned zeros[1024]; memset(zeros, 0, sizeof zeros);
  zeros[13] = 1; /* the poison word of a 6-class counter layout: 1 + 2 * 6 */
  cu->MemcpyHtoD(words, zeros, sizeof zeros);
  sccl_allreduce_args a; memset(&a, 0, sizeof a);
  a.input = a.output = words + 2048; a.size_packs = 512; a.nbytes = 8192;
  a.recv_base = a.flag_base = a.send_base = a.ctrl_base = words + 1024; a.slot_bytes = 4096;
  a.epoch = words; a.stage_counter = words + 4; a.tail_counter = words + 28; a.poison = words + 52;
  a.phase_counter = words + 60; a.arrival = words + 3000; a.spin_limit = 1000; a.rank = 0; a.lanes = 1; a.one_block = 1;
  for (int alg = 0; alg < 2; ++alg) {
    for (int i = 0; i < 200; ++i) sccl_kp_launch_allreduce(fn[alg], alg, &a, 1, 512, NULL);
    cu->CtxSynchronize();
    double best = 1e30, total = 0; int batches = n / 100;
    for (int b = 0; b < batches; ++b) {
      double t0 = now_us();
      for (int i = 0; i < 100; ++i)
        if (sccl_kp_launch_allreduce(fn[alg], alg, &a, 1, 512, NULL) != 0) { fprintf(stderr, "launch failed\n"); return 1; }
      double t = (now_us() - t0) / 100;
      total += t; if (t < best) best = t;
      cu->CtxSynchronize();
    }
    printf("%s bf16 W=2: %d launches in batches of 100: mean %.2f us, best batch %.2f us per launch\n",
           alg ? "two-shot" : "one-shot", batches * 100, total / batches, best);
  }
  return 0;
}
