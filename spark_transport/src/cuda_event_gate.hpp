#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace spark_transport {

// A bounded pool of reusable stream-position markers. Callers must not reuse
// a sequence slot until the corresponding operation has completed.
class CudaEventGatePool {
 public:
  CudaEventGatePool(std::size_t capacity, std::chrono::seconds timeout,
                    std::string name)
      : timeout_(timeout), name_(std::move(name)) {
    if (capacity == 0 || timeout_ <= std::chrono::seconds::zero()) {
      throw std::invalid_argument("invalid CUDA event gate pool");
    }
    const cudaError_t device_result = cudaGetDevice(&device_);
    if (device_result != cudaSuccess) {
      throw std::runtime_error(
          "cudaGetDevice " + name_ + ": " +
          cudaGetErrorString(device_result));
    }
    events_.reserve(capacity);
    try {
      for (std::size_t index = 0; index < capacity; ++index) {
        cudaEvent_t event{};
        const cudaError_t result =
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (result != cudaSuccess) {
          if (event != nullptr) {
            static_cast<void>(cudaEventDestroy(event));
          }
          throw std::runtime_error(
              "cudaEventCreateWithFlags " + name_ + ": " +
              cudaGetErrorString(result));
        }
        events_.push_back(event);
      }
    } catch (...) {
      destroy_events();
      throw;
    }
  }

  CudaEventGatePool(const CudaEventGatePool&) = delete;
  CudaEventGatePool& operator=(const CudaEventGatePool&) = delete;
  CudaEventGatePool(CudaEventGatePool&&) = delete;
  CudaEventGatePool& operator=(CudaEventGatePool&&) = delete;

  ~CudaEventGatePool() { destroy_events(); }

  void bind_current_thread() const {
    const cudaError_t device_result = cudaSetDevice(device_);
    if (device_result != cudaSuccess) {
      throw std::runtime_error(
          "cudaSetDevice " + name_ + ": " +
          cudaGetErrorString(device_result));
    }
  }

  void record(std::uint64_t sequence, cudaStream_t stream) {
    cudaEvent_t event = event_for(sequence);
    const cudaError_t result = cudaEventRecord(event, stream);
    if (result != cudaSuccess) {
      throw std::runtime_error(
          "cudaEventRecord " + name_ + " sequence " +
          std::to_string(sequence) + ": " + cudaGetErrorString(result));
    }
  }

  void wait(std::uint64_t sequence) const {
    cudaEvent_t event = event_for(sequence);
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (true) {
      const cudaError_t result = cudaEventQuery(event);
      if (result == cudaSuccess) {
        return;
      }
      if (result != cudaErrorNotReady) {
        throw std::runtime_error(
            "cudaEventQuery " + name_ + " sequence " +
            std::to_string(sequence) + ": " + cudaGetErrorString(result));
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
            "timed out waiting for " + name_ + " sequence " +
            std::to_string(sequence));
      }
      std::this_thread::yield();
    }
  }

 private:
  cudaEvent_t event_for(std::uint64_t sequence) const {
    if (sequence == 0) {
      throw std::invalid_argument(name_ + " sequence must be nonzero");
    }
    return events_[(sequence - 1) % events_.size()];
  }

  void destroy_events() noexcept {
    for (cudaEvent_t event : events_) {
      static_cast<void>(cudaEventDestroy(event));
    }
    events_.clear();
  }

  std::vector<cudaEvent_t> events_;
  std::chrono::seconds timeout_;
  std::string name_;
  int device_{};
};

}  // namespace spark_transport
