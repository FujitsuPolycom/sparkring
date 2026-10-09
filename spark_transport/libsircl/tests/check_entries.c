/* The kernel-entry check. It runs the library's own pack loader (src/kernelpack.c, sccl_kp_load) on the
 * embedded transport, fold, link and point-to-point fatbins through the stand-in driver tests/fake_cuda.c, which reads the
 * fatbins' cubins offline: every entry name the loader asks for must be a defined function in every cubin
 * (every GPU architecture) of its pack. The build runs it before it links the library, so a loader that
 * names an entry its prebuilt pack lacks fails the build, without a GPU. Exit 0 when every entry is
 * present, 1 when one is missing (the loader's message names it), 2 when the stand-in driver is not the
 * libcuda.so.1 the loader would bind.
 *
 *   LD_LIBRARY_PATH=<directory of the stand-in libcuda.so.1> check_entries <architectures per pack>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernelpack.h"

int main(int argc, char **argv) {
  int architectures = argc > 1 ? atoi(argv[1]) : 1;
  void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  void (*stats)(int *, int *, int *) = NULL;
  if (driver) *(void **)(&stats) = dlsym(driver, "sccl_fake_cuda_stats");
  if (!driver || !dlsym(driver, "sccl_fake_cuda") || !stats) {
    fprintf(stderr, "check_entries: the libcuda.so.1 found first is not the stand-in driver (tests/fake_cuda.c); "
                    "put its directory first on LD_LIBRARY_PATH\n");
    return 2;
  }
  static int token;
  if (sccl_kp_load((sccl_CUcontext)(void *)&token) != 0) {
    fprintf(stderr, "check_entries: %s\n", sccl_kp_error());
    return 1;
  }
  int modules = 0, cubins = 0, lookups = 0;
  stats(&modules, &cubins, &lookups);
  if (modules != 4 || cubins < architectures) {
    fprintf(stderr, "check_entries: %d packs loaded with at least %d cubins each; expected 4 packs of %d\n", modules,
            cubins, architectures);
    return 1;
  }
  printf("kernel entries: all %d the loader names are defined in every cubin of the 4 packs (%d or more "
         "architectures each)\n", lookups, cubins);
  return 0;
}
