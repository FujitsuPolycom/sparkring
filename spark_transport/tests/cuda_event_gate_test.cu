#include "cuda_event_gate.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void CUDART_CB delay_stream(void* duration_pointer) {
  const auto duration =
      *static_cast<const std::chrono::milliseconds*>(duration_pointer);
  std::this_thread::sleep_for(duration);
}

}  // namespace

int main() {
  int devices{};
  const cudaError_t device_result = cudaGetDeviceCount(&devices);
  if (device_result == cudaErrorNoDevice ||
      device_result == cudaErrorInsufficientDriver || devices == 0) {
    static_cast<void>(cudaGetLastError());
    return 0;
  }
  check_cuda(device_result, "cudaGetDeviceCount");

  cudaStream_t stream{};
  check_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "cudaStreamCreateWithFlags");
  {
    spark_transport::CudaEventGatePool gates(
        1, std::chrono::seconds(10), "test delayed stream");
    // FULL custom capture held the caller stream for 6.1--6.7 seconds. This
    // delay crosses the old five-second staging watchdog deterministically.
    std::chrono::milliseconds delay(5500);
    check_cuda(
        cudaLaunchHostFunc(stream, delay_stream, &delay),
        "cudaLaunchHostFunc");
    gates.record(1, stream);

    const auto started = std::chrono::steady_clock::now();
    gates.bind_current_thread();
    gates.wait(1);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    if (elapsed < std::chrono::seconds(5)) {
      throw std::runtime_error(
          "CUDA event gate did not wait for the delayed stream position");
    }

    // The sequence slot can be reused only after the prior gate completed.
    gates.record(2, stream);
    gates.wait(2);
  }
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
