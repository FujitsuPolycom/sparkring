/* SIRCL host-only ABI declarations; no CUDA implementation is provided. */
#ifndef SIRCL_CUDA_TYPES_H_
#define SIRCL_CUDA_TYPES_H_
#ifndef LIBSIRCL_CPU_ONLY
#error "sccl_cuda_types.h is only for an explicit LIBSIRCL_CPU_ONLY build"
#endif
/* Public CUDA runtime handles are opaque pointers of these shapes. */
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
#endif
