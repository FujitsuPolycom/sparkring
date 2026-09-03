#include <cuda_runtime.h>
#include <cstdint>

__global__ void mark(std::uint32_t* values, std::uint32_t index) {
  if (threadIdx.x == 0 && blockIdx.x == 0) values[index] = index + 1U;
}

int main() {
  cudaStream_t streams[2]{};
  cudaEvent_t done{};
  std::uint32_t* values{};
  if (cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking) != cudaSuccess ||
      cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&done, cudaEventDisableTiming) != cudaSuccess ||
      cudaMalloc(&values, 4U * sizeof(*values)) != cudaSuccess ||
      cudaMemset(values, 0, 4U * sizeof(*values)) != cudaSuccess) {
    return 1;
  }
  for (std::uint32_t operation = 0; operation < 4; ++operation) {
    if (operation != 0 && cudaEventSynchronize(done) != cudaSuccess) return 1;
    auto stream = streams[operation & 1U];
    mark<<<1, 1, 0, stream>>>(values, operation);
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(done, stream) != cudaSuccess) {
      return 1;
    }
  }
  if (cudaEventSynchronize(done) != cudaSuccess) return 1;
  std::uint32_t host[4]{};
  if (cudaMemcpy(host, values, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return 1;
  }
  for (std::uint32_t i = 0; i < 4; ++i) {
    if (host[i] != i + 1U) return 1;
  }
  if (cudaFree(values) != cudaSuccess ||
      cudaEventDestroy(done) != cudaSuccess ||
      cudaStreamDestroy(streams[1]) != cudaSuccess ||
      cudaStreamDestroy(streams[0]) != cudaSuccess) {
    return 1;
  }
  return 0;
}
