/* The kernel-pack cache (src/kernelpack.c) over the stand-in driver tests/fake_cuda.c: an entry per context,
 * held by references. Packs loaded and released in turn in 20 distinct contexts (more than the 16 entries the
 * cache holds at once) all load, and each last release unloads the context's four modules; a reference still
 * held keeps the entry; a context address reused after its last release loads the packs afresh.
 *
 *   LD_LIBRARY_PATH=<directory of the stand-in libcuda.so.1> test_kernelpack_cache
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernelpack.h"

static unsigned checks;
#define CHECK(expr) do { checks++; if (!(expr)) { \
  fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #expr, sccl_kp_error()); exit(1); } } while (0)

static void (*stats)(int *, int *, int *);
static int live_modules(void) {
  int modules = 0, cubins = 0, lookups = 0;
  stats(&modules, &cubins, &lookups);
  return modules;
}
static int loaded(sccl_CUcontext ctx) {
  sccl_CUfunction function;
  return sccl_kp_function(ctx, SCCL_ALG_ONESHOT, SCCL_DT_F32, 2, &function) == 0;
}

int main(void) {
  void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (driver) *(void **)(&stats) = dlsym(driver, "sccl_fake_cuda_stats");
  if (!driver || !stats) {
    fprintf(stderr, "test_kernelpack_cache: put the stand-in driver's directory first on LD_LIBRARY_PATH\n");
    return 2;
  }
  static int contexts[20];
  for (int i = 0; i < 20; ++i) {
    sccl_CUcontext ctx = (sccl_CUcontext)(void *)&contexts[i];
    CHECK(sccl_kp_load(ctx) == 0);
    CHECK(loaded(ctx) && live_modules() == 4);
    sccl_kp_release(ctx, 1);
    CHECK(!loaded(ctx) && live_modules() == 0);
  }
  /* Two references: the first release keeps the entry, the second removes it. */
  sccl_CUcontext a = (sccl_CUcontext)(void *)&contexts[0], b = (sccl_CUcontext)(void *)&contexts[1];
  CHECK(sccl_kp_load(a) == 0 && sccl_kp_load(b) == 0 && sccl_kp_load(a) == 0);
  CHECK(live_modules() == 8);
  sccl_kp_release(a, 1);
  CHECK(loaded(a) && loaded(b) && live_modules() == 8);
  sccl_kp_release(a, 1);
  CHECK(!loaded(a) && loaded(b) && live_modules() == 4);
  /* A reused address loads afresh; a release without unloading leaves the modules to the context. */
  CHECK(sccl_kp_load(a) == 0 && loaded(a) && live_modules() == 8);
  sccl_kp_release(a, 0);
  CHECK(!loaded(a) && live_modules() == 8);
  sccl_kp_release(b, 1);
  CHECK(!loaded(b));
  /* A pack the driver cannot load (no cubin for the GPU) is refused, naming the architectures it holds. */
  setenv("FAKE_CUDA_FAIL_LOAD", "1", 1);
  CHECK(sccl_kp_load(a) != 0);
  CHECK(strstr(sccl_kp_error(), "built for sm_120 sm_121") != NULL);
  unsetenv("FAKE_CUDA_FAIL_LOAD");
  printf("kernel-pack cache: %u checks passed\n", checks);
  return 0;
}
