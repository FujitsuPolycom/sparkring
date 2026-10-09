/* The CUDA driver API subset libsircl uses, loaded from libcuda.so.1 at run
 * time. The library has no link-time CUDA dependency: it loads in processes
 * without a GPU (the CPU tests), and in a process with a GPU it shares the
 * application's driver and contexts whatever CUDA runtime the application
 * carries. Types are the driver ABI's opaque handles and integers. */
#ifndef SCCL_CUDA_API_H
#define SCCL_CUDA_API_H
#include <stddef.h>
#include <stdint.h>

typedef int sccl_CUresult;
typedef int sccl_CUdevice;
typedef unsigned long long sccl_CUdeviceptr;
typedef struct sccl_CUctx_st *sccl_CUcontext;
typedef struct sccl_CUmod_st *sccl_CUmodule;
typedef struct sccl_CUfunc_st *sccl_CUfunction;
typedef struct sccl_CUstream_st *sccl_CUstream;
typedef struct sccl_CUevent_st *sccl_CUevent;

enum {
  SCCL_CUDA_SUCCESS = 0,
  SCCL_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  SCCL_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  SCCL_CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
  SCCL_CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
  SCCL_CU_MEMHOSTALLOC_PORTABLE = 1,
  SCCL_CU_MEMHOSTALLOC_DEVICEMAP = 2,
  SCCL_CU_MEMHOSTREGISTER_PORTABLE = 1,
  SCCL_CU_MEMHOSTREGISTER_DEVICEMAP = 2,
  SCCL_CU_EVENT_DISABLE_TIMING = 2,
  SCCL_CU_STREAM_CAPTURE_STATUS_NONE = 0,
  SCCL_CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
  SCCL_CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
};

typedef struct sccl_cuda {
  sccl_CUresult (*Init)(unsigned);
  sccl_CUresult (*GetErrorString)(sccl_CUresult, const char **);
  sccl_CUresult (*CtxGetCurrent)(sccl_CUcontext *);
  sccl_CUresult (*CtxSetCurrent)(sccl_CUcontext);
  sccl_CUresult (*CtxPushCurrent)(sccl_CUcontext);
  sccl_CUresult (*CtxPopCurrent)(sccl_CUcontext *);
  sccl_CUresult (*CtxGetDevice)(sccl_CUdevice *);
  sccl_CUresult (*CtxSynchronize)(void);
  sccl_CUresult (*DeviceGet)(sccl_CUdevice *, int);
  sccl_CUresult (*DeviceGetAttribute)(int *, int, sccl_CUdevice);
  sccl_CUresult (*DevicePrimaryCtxRetain)(sccl_CUcontext *, sccl_CUdevice);
  sccl_CUresult (*ModuleLoadData)(sccl_CUmodule *, const void *);
  sccl_CUresult (*ModuleUnload)(sccl_CUmodule);
  sccl_CUresult (*ModuleGetFunction)(sccl_CUfunction *, sccl_CUmodule, const char *);
  sccl_CUresult (*LaunchKernel)(sccl_CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                unsigned, sccl_CUstream, void **, void **);
  sccl_CUresult (*MemAlloc)(sccl_CUdeviceptr *, size_t);
  sccl_CUresult (*MemFree)(sccl_CUdeviceptr);
  sccl_CUresult (*MemsetD32Async)(sccl_CUdeviceptr, unsigned, size_t, sccl_CUstream);
  sccl_CUresult (*MemsetD8Async)(sccl_CUdeviceptr, unsigned char, size_t, sccl_CUstream);
  sccl_CUresult (*MemcpyDtoDAsync)(sccl_CUdeviceptr, sccl_CUdeviceptr, size_t, sccl_CUstream);
  sccl_CUresult (*MemcpyHtoD)(sccl_CUdeviceptr, const void *, size_t);
  sccl_CUresult (*MemcpyDtoH)(void *, sccl_CUdeviceptr, size_t);
  sccl_CUresult (*MemHostAlloc)(void **, size_t, unsigned);
  sccl_CUresult (*MemFreeHost)(void *);
  sccl_CUresult (*MemHostGetDevicePointer)(sccl_CUdeviceptr *, void *, unsigned);
  sccl_CUresult (*MemHostRegister)(void *, size_t, unsigned);
  sccl_CUresult (*MemHostUnregister)(void *);
  sccl_CUresult (*StreamGetCaptureInfo)(sccl_CUstream, int *, unsigned long long *);
  sccl_CUresult (*StreamSynchronize)(sccl_CUstream);
  sccl_CUresult (*StreamQuery)(sccl_CUstream);
  sccl_CUresult (*StreamWaitEvent)(sccl_CUstream, sccl_CUevent, unsigned);
  sccl_CUresult (*EventCreate)(sccl_CUevent *, unsigned);
  sccl_CUresult (*EventRecord)(sccl_CUevent, sccl_CUstream);
  sccl_CUresult (*EventDestroy)(sccl_CUevent);
  sccl_CUresult (*EventSynchronize)(sccl_CUevent);
} sccl_cuda;

/* The process's driver table, loaded once; NULL when libcuda.so.1 or one of
 * its entry points is missing (the reason is in sccl_cuda_error()). */
const sccl_cuda *sccl_cuda_get(void);
const char *sccl_cuda_error(void);
/* "name: message" of a driver result. */
const char *sccl_cuda_result_text(sccl_CUresult result);

#endif
