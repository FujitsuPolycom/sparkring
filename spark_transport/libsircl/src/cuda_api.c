/* Run-time loading of the CUDA driver API (cuda_api.h). */
#define _GNU_SOURCE
#include "cuda_api.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static sccl_cuda table;
static const sccl_cuda *loaded;
static char load_error[256];
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void load(void) {
  const char *path = "libcuda.so.1";
  void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    snprintf(load_error, sizeof load_error, "cannot load %s: %s", path, dlerror());
    return;
  }
#define SYM(field, name)                                                              \
  do {                                                                                \
    *(void **)(&table.field) = dlsym(handle, name);                                   \
    if (!table.field) {                                                               \
      snprintf(load_error, sizeof load_error, "%s has no entry point %s", path, name); \
      return;                                                                         \
    }                                                                                 \
  } while (0)
  SYM(Init, "cuInit");
  SYM(GetErrorString, "cuGetErrorString");
  SYM(CtxGetCurrent, "cuCtxGetCurrent");
  SYM(CtxSetCurrent, "cuCtxSetCurrent");
  SYM(CtxPushCurrent, "cuCtxPushCurrent_v2");
  SYM(CtxPopCurrent, "cuCtxPopCurrent_v2");
  SYM(CtxGetDevice, "cuCtxGetDevice");
  SYM(CtxSynchronize, "cuCtxSynchronize");
  SYM(DeviceGet, "cuDeviceGet");
  SYM(DeviceGetAttribute, "cuDeviceGetAttribute");
  SYM(DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
  SYM(ModuleLoadData, "cuModuleLoadData");
  SYM(ModuleUnload, "cuModuleUnload");
  SYM(ModuleGetFunction, "cuModuleGetFunction");
  SYM(LaunchKernel, "cuLaunchKernel");
  SYM(MemAlloc, "cuMemAlloc_v2");
  SYM(MemFree, "cuMemFree_v2");
  SYM(MemsetD32Async, "cuMemsetD32Async");
  SYM(MemsetD8Async, "cuMemsetD8Async");
  SYM(MemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
  SYM(MemcpyHtoD, "cuMemcpyHtoD_v2");
  SYM(MemcpyDtoH, "cuMemcpyDtoH_v2");
  SYM(MemHostAlloc, "cuMemHostAlloc");
  SYM(MemFreeHost, "cuMemFreeHost");
  SYM(MemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2");
  SYM(MemHostRegister, "cuMemHostRegister_v2");
  SYM(MemHostUnregister, "cuMemHostUnregister");
  SYM(StreamGetCaptureInfo, "cuStreamGetCaptureInfo");
  SYM(StreamSynchronize, "cuStreamSynchronize");
  SYM(StreamQuery, "cuStreamQuery");
  SYM(StreamWaitEvent, "cuStreamWaitEvent");
  SYM(EventCreate, "cuEventCreate");
  SYM(EventRecord, "cuEventRecord");
  SYM(EventDestroy, "cuEventDestroy_v2");
  SYM(EventSynchronize, "cuEventSynchronize");
#undef SYM
  *(void **)(&table.MemAllocAsync) = dlsym(handle, "cuMemAllocAsync");
  *(void **)(&table.MemFreeAsync) = dlsym(handle, "cuMemFreeAsync");
  if (!table.MemAllocAsync || !table.MemFreeAsync) {
    table.MemAllocAsync = NULL;
    table.MemFreeAsync = NULL;
  }
  sccl_CUresult result = table.Init(0);
  if (result != SCCL_CUDA_SUCCESS) {
    snprintf(load_error, sizeof load_error, "cuInit failed: %d", result);
    return;
  }
  loaded = &table;
}

const sccl_cuda *sccl_cuda_get(void) {
  pthread_once(&once, load);
  return loaded;
}

const char *sccl_cuda_error(void) { return load_error[0] ? load_error : "no error"; }

const char *sccl_cuda_result_text(sccl_CUresult result) {
  static _Thread_local char text[160];
  const char *message = NULL;
  if (loaded && loaded->GetErrorString(result, &message) != SCCL_CUDA_SUCCESS) message = NULL;
  snprintf(text, sizeof text, "CUDA error %d (%s)", result, message ? message : "unknown");
  return text;
}
