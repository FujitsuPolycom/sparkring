#pragma once

typedef void* cudaStream_t;
typedef int cudaError_t;
typedef int cudaStreamCaptureStatus;

static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaStreamCaptureStatus
    cudaStreamCaptureStatusActive = 1;
static constexpr int cudaDevAttrHostNativeAtomicSupported = 0;

const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaDeviceGetAttribute(int*, int, int);
cudaError_t cudaStreamIsCapturing(
    cudaStream_t, cudaStreamCaptureStatus*);
cudaError_t cudaStreamSynchronize(cudaStream_t);
