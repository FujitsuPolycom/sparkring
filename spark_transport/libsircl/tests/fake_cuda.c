/* A stand-in for the CUDA driver (libcuda.so.1) for the kernel-entry check (tests/check_entries.c): it
 * exports every entry point src/cuda_api.c binds, and its module functions read the cubins of a fatbin
 * offline. cuModuleLoadData finds every ELF cubin in the fatbin image and collects the defined function
 * symbols of each; cuModuleGetFunction succeeds only for a name every cubin of the module defines (one
 * cubin per GPU architecture, so an entry missing for one architecture fails), else returns
 * CUDA_ERROR_NOT_FOUND (500) as the driver does. FAKE_CUDA_HIDE=<name> hides one symbol, for negative
 * controls. Every other entry point succeeds and does nothing. Test infrastructure only. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))
enum { NOT_FOUND = 500, INVALID_IMAGE = 200, MAX_MODULES = 16, MAX_CUBINS = 8 };

typedef struct {
  int cubins;
  uint32_t flags[MAX_CUBINS];
  char **names[MAX_CUBINS];
  int counts[MAX_CUBINS];
} module_t;

static module_t modules[MAX_MODULES];
static int nmodules, lookups;
static int context_token;

static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t u32(const uint8_t *p) { return (uint32_t)u16(p) | (uint32_t)u16(p + 2) << 16; }
static uint64_t u64(const uint8_t *p) { return (uint64_t)u32(p) | (uint64_t)u32(p + 4) << 32; }

/* The defined function symbols of the ELF64 cubin at `elf` (at most `room` bytes); its extent in *extent,
 * 0 when it is not a whole ELF64 little-endian image. */
static int read_cubin(const uint8_t *elf, uint64_t room, module_t *m, uint64_t *extent) {
  *extent = 0;
  if (room < 64 || elf[4] != 2 || elf[5] != 1) return -1;
  uint64_t shoff = u64(elf + 0x28);
  uint16_t shentsize = u16(elf + 0x3a), shnum = u16(elf + 0x3c);
  if (shentsize != 64 || shoff + (uint64_t)shnum * 64 > room) return -1;
  uint64_t end = shoff + (uint64_t)shnum * 64;
  for (int i = 0; i < shnum; ++i) {
    const uint8_t *sh = elf + shoff + (uint64_t)i * 64;
    uint64_t off = u64(sh + 0x18), size = u64(sh + 0x20);
    if (u32(sh + 4) != 8 /* SHT_NOBITS */ && off + size > end) end = off + size;
  }
  if (end > room || m->cubins == MAX_CUBINS) return -1;
  int c = m->cubins;
  m->flags[c] = u32(elf + 0x30);
  m->names[c] = NULL;
  m->counts[c] = 0;
  for (int i = 0; i < shnum; ++i) {
    const uint8_t *sh = elf + shoff + (uint64_t)i * 64;
    if (u32(sh + 4) != 2 /* SHT_SYMTAB */) continue;
    uint64_t off = u64(sh + 0x18), size = u64(sh + 0x20), entsize = u64(sh + 0x38);
    uint32_t link = u32(sh + 0x28);
    if (entsize != 24 || link >= shnum) return -1;
    const uint8_t *strsh = elf + shoff + (uint64_t)link * 64;
    const char *strtab = (const char *)elf + u64(strsh + 0x18);
    uint64_t strsize = u64(strsh + 0x20);
    for (uint64_t s = 0; s + 24 <= size; s += 24) {
      const uint8_t *sym = elf + off + s;
      uint32_t name = u32(sym);
      if ((sym[4] & 0xf) != 2 /* STT_FUNC */ || u16(sym + 6) == 0 || name >= strsize) continue;
      char **grown = realloc(m->names[c], sizeof(char *) * (size_t)(m->counts[c] + 1));
      if (!grown) return -1;
      m->names[c] = grown;
      m->names[c][m->counts[c]++] = strdup(strtab + name);
    }
  }
  m->cubins = c + 1;
  *extent = end;
  return 0;
}

EXPORT int sccl_fake_cuda(void) { return 1; }

/* Modules loaded, the fewest cubins of any module, and symbol lookups so far. */
EXPORT void sccl_fake_cuda_stats(int *loaded, int *fewest_cubins, int *looked_up) {
  *loaded = nmodules;
  *fewest_cubins = 0;
  for (int i = 0; i < nmodules; ++i)
    if (!i || modules[i].cubins < *fewest_cubins) *fewest_cubins = modules[i].cubins;
  *looked_up = lookups;
}

EXPORT int cuModuleLoadData(void **module, const void *image) {
  const uint8_t *p = image;
  if (nmodules == MAX_MODULES || u32(p) != 0xBA55ED50u) return INVALID_IMAGE;
  uint64_t total = (uint64_t)u16(p + 6) + u64(p + 8);
  module_t *m = &modules[nmodules];
  memset(m, 0, sizeof *m);
  for (uint64_t at = 0; at + 4 <= total;) {
    if (memcmp(p + at, "\x7f" "ELF", 4) == 0) {
      uint64_t extent;
      if (read_cubin(p + at, total - at, m, &extent) == 0) {
        at += extent;
        continue;
      }
    }
    ++at;
  }
  if (!m->cubins) return INVALID_IMAGE;
  *module = m;
  ++nmodules;
  return 0;
}

EXPORT int cuModuleGetFunction(void **function, void *module, const char *name) {
  module_t *m = module;
  const char *hidden = getenv("FAKE_CUDA_HIDE");
  ++lookups;
  if (hidden && !strcmp(hidden, name)) return NOT_FOUND;
  for (int c = 0; c < m->cubins; ++c) {
    int found = 0;
    for (int i = 0; i < m->counts[c] && !found; ++i) found = !strcmp(m->names[c][i], name);
    if (!found) return NOT_FOUND;
  }
  *function = (void *)name;
  return 0;
}

EXPORT int cuGetErrorString(int error, const char **text) {
  *text = error == NOT_FOUND ? "named symbol not found" : error == INVALID_IMAGE ? "device kernel image is invalid"
                                                                                 : "no error";
  return 0;
}
EXPORT int cuCtxGetCurrent(void **ctx) {
  *ctx = &context_token;
  return 0;
}
EXPORT int cuCtxPopCurrent_v2(void **ctx) {
  if (ctx) *ctx = &context_token;
  return 0;
}
EXPORT int cuInit(unsigned flags) { return (int)(flags & 0); }
EXPORT int cuCtxPushCurrent_v2(void *ctx) { return ctx ? 0 : 201; }
EXPORT int cuModuleUnload(void *module) { return module ? 0 : 400; }

/* Every other entry point src/cuda_api.c binds: present, never used by the check. */
#define UNUSED_ENTRY(name) \
  EXPORT int name(void) { return 0; }
UNUSED_ENTRY(cuCtxSetCurrent)
UNUSED_ENTRY(cuCtxGetDevice)
UNUSED_ENTRY(cuCtxSynchronize)
UNUSED_ENTRY(cuDeviceGet)
UNUSED_ENTRY(cuDeviceGetAttribute)
UNUSED_ENTRY(cuDevicePrimaryCtxRetain)
UNUSED_ENTRY(cuLaunchKernel)
UNUSED_ENTRY(cuMemAlloc_v2)
UNUSED_ENTRY(cuMemFree_v2)
UNUSED_ENTRY(cuMemsetD32Async)
UNUSED_ENTRY(cuMemsetD8Async)
UNUSED_ENTRY(cuMemcpyDtoDAsync_v2)
UNUSED_ENTRY(cuMemcpyHtoD_v2)
UNUSED_ENTRY(cuMemcpyDtoH_v2)
UNUSED_ENTRY(cuMemHostAlloc)
UNUSED_ENTRY(cuMemFreeHost)
UNUSED_ENTRY(cuMemHostGetDevicePointer_v2)
UNUSED_ENTRY(cuMemHostRegister_v2)
UNUSED_ENTRY(cuMemHostUnregister)
UNUSED_ENTRY(cuStreamGetCaptureInfo)
UNUSED_ENTRY(cuStreamSynchronize)
UNUSED_ENTRY(cuStreamQuery)
UNUSED_ENTRY(cuStreamWaitEvent)
UNUSED_ENTRY(cuEventCreate)
UNUSED_ENTRY(cuEventRecord)
UNUSED_ENTRY(cuEventDestroy_v2)
UNUSED_ENTRY(cuEventSynchronize)
