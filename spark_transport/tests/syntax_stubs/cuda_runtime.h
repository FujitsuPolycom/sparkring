#pragma once

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef int cudaError_t;
typedef int cudaStreamCaptureStatus;

static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorNotReady = 1;
static constexpr cudaStreamCaptureStatus
    cudaStreamCaptureStatusActive = 1;
static constexpr int cudaDevAttrHostNativeAtomicSupported = 0;
static constexpr unsigned int cudaEventDisableTiming = 2;

const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaDeviceGetAttribute(int*, int, int);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int);
cudaError_t cudaStreamIsCapturing(
    cudaStream_t, cudaStreamCaptureStatus*);
cudaError_t cudaStreamSynchronize(cudaStream_t);
