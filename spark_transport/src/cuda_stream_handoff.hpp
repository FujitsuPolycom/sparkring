#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace spark_transport {

class CudaStreamHandoff {
 public:
  CudaStreamHandoff() = default;

  CudaStreamHandoff(const CudaStreamHandoff&) = delete;
  CudaStreamHandoff& operator=(const CudaStreamHandoff&) = delete;

  ~CudaStreamHandoff() {
    if (event_ == nullptr) {
      return;
    }
    const cudaError_t result = cudaEventDestroy(event_);
    if (result != cudaSuccess) {
      std::fprintf(
          stderr, "FATAL CUDA stream handoff event destruction failed: %s\n",
          cudaGetErrorString(result));
      std::fflush(stderr);
      std::abort();
    }
  }

  void order(cudaStream_t previous, cudaStream_t next) {
    ensure_created();
    cudaError_t result = cudaEventRecord(event_, previous);
    if (result != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaEventRecord stream handoff: ") +
          cudaGetErrorString(result));
    }
    result = cudaStreamWaitEvent(next, event_, 0);
    if (result != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaStreamWaitEvent stream handoff: ") +
          cudaGetErrorString(result));
    }
  }

 private:
  void ensure_created() {
    if (event_ != nullptr) {
      return;
    }
    const cudaError_t result = cudaEventCreateWithFlags(
        &event_, cudaEventDisableTiming);
    if (result != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaEventCreateWithFlags stream handoff: ") +
          cudaGetErrorString(result));
    }
  }

  cudaEvent_t event_{};
};

}  // namespace spark_transport
