#include "spark_transport/memory_buffer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace spark_transport {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

__global__ void fill_kernel(std::uint8_t* data, std::size_t bytes,
                            std::uint8_t value) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < bytes) {
    data[index] = value;
  }
}

__global__ void verify_kernel(const std::uint8_t* data, std::size_t bytes,
                              std::uint8_t expected,
                              unsigned long long* mismatches) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < bytes && data[index] != expected) {
    atomicAdd(mismatches, 1ULL);
  }
}

class HostBuffer final : public MemoryBuffer {
 public:
  explicit HostBuffer(std::size_t bytes) : bytes_(bytes) {
    if (posix_memalign(&data_, 4096, bytes_) != 0) {
      throw std::bad_alloc();
    }
    std::memset(data_, 0, bytes_);
  }

  ~HostBuffer() override { std::free(data_); }

  void* host_data() const override { return data_; }
  void* device_data() const override { return nullptr; }
  std::size_t size() const override { return bytes_; }
  MemoryKind kind() const override { return MemoryKind::kHost; }

  void fill_from_cpu(std::uint8_t value) override {
    std::memset(data_, value, bytes_);
  }

  void fill_from_gpu(std::uint8_t) override {
    throw std::logic_error("host memory is not GPU mapped");
  }

  bool verify_on_cpu(std::uint8_t value, std::size_t bytes) const override {
    const auto* data = static_cast<const std::uint8_t*>(data_);
    return std::all_of(data, data + bytes,
                       [value](std::uint8_t byte) { return byte == value; });
  }

  bool verify_on_gpu(std::uint8_t, std::size_t) const override {
    throw std::logic_error("host memory is not GPU mapped");
  }

 private:
  void* data_{};
  std::size_t bytes_{};
};

class CudaMappedBuffer final : public MemoryBuffer {
 public:
  CudaMappedBuffer(std::size_t bytes, bool write_combined)
      : bytes_(bytes), write_combined_(write_combined) {
    unsigned int flags = cudaHostAllocMapped;
    if (write_combined_) {
      flags |= cudaHostAllocWriteCombined;
    }
    check_cuda(cudaHostAlloc(&host_, bytes_, flags), "cudaHostAlloc");
    check_cuda(cudaHostGetDevicePointer(&device_, host_, 0),
               "cudaHostGetDevicePointer");
    std::memset(host_, 0, bytes_);
  }

  ~CudaMappedBuffer() override {
    if (host_ != nullptr) {
      cudaFreeHost(host_);
    }
  }

  void* host_data() const override { return host_; }
  void* device_data() const override { return device_; }
  std::size_t size() const override { return bytes_; }
  MemoryKind kind() const override {
    return write_combined_ ? MemoryKind::kCudaWriteCombined
                           : MemoryKind::kCudaMapped;
  }

  void fill_from_cpu(std::uint8_t value) override {
    std::memset(host_, value, bytes_);
  }

  void fill_from_gpu(std::uint8_t value) override {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes_ + threads - 1) / threads);
    fill_kernel<<<blocks, threads>>>(static_cast<std::uint8_t*>(device_), bytes_,
                                     value);
    check_cuda(cudaGetLastError(), "fill_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "fill_kernel synchronize");
  }

  bool verify_on_cpu(std::uint8_t value, std::size_t bytes) const override {
    const auto* data = static_cast<const std::uint8_t*>(host_);
    return std::all_of(data, data + bytes,
                       [value](std::uint8_t byte) { return byte == value; });
  }

  bool verify_on_gpu(std::uint8_t value, std::size_t bytes) const override {
    unsigned long long* mismatches{};
    check_cuda(cudaMalloc(&mismatches, sizeof(*mismatches)), "cudaMalloc");
    check_cuda(cudaMemset(mismatches, 0, sizeof(*mismatches)), "cudaMemset");
    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes + threads - 1) / threads);
    verify_kernel<<<blocks, threads>>>(
        static_cast<const std::uint8_t*>(device_), bytes, value, mismatches);
    check_cuda(cudaGetLastError(), "verify_kernel launch");
    unsigned long long host_mismatches{};
    check_cuda(cudaMemcpy(&host_mismatches, mismatches, sizeof(host_mismatches),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy mismatch result");
    cudaFree(mismatches);
    return host_mismatches == 0;
  }

 private:
  void* host_{};
  void* device_{};
  std::size_t bytes_{};
  bool write_combined_{};
};

class CudaAllocationBuffer final : public MemoryBuffer {
 public:
  CudaAllocationBuffer(std::size_t bytes, bool managed)
      : bytes_(bytes), managed_(managed) {
    if (managed_) {
      check_cuda(cudaMallocManaged(&data_, bytes_, cudaMemAttachGlobal),
                 "cudaMallocManaged");
    } else {
      check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
    }
    check_cuda(cudaMemset(data_, 0, bytes_), "cudaMemset allocation");
    check_cuda(cudaDeviceSynchronize(), "cudaMemset allocation synchronize");
  }

  ~CudaAllocationBuffer() override {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void* host_data() const override { return data_; }
  void* device_data() const override { return data_; }
  std::size_t size() const override { return bytes_; }
  MemoryKind kind() const override {
    return managed_ ? MemoryKind::kCudaManaged : MemoryKind::kCudaDevice;
  }

  void fill_from_cpu(std::uint8_t value) override {
    if (!managed_) {
      throw std::logic_error("cuda-device memory is not CPU accessible");
    }
    check_cuda(cudaDeviceSynchronize(), "managed CPU fill synchronize");
    std::memset(data_, value, bytes_);
  }

  void fill_from_gpu(std::uint8_t value) override {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes_ + threads - 1) / threads);
    fill_kernel<<<blocks, threads>>>(static_cast<std::uint8_t*>(data_), bytes_,
                                     value);
    check_cuda(cudaGetLastError(), "allocation fill_kernel launch");
    check_cuda(cudaDeviceSynchronize(),
               "allocation fill_kernel synchronize");
  }

  bool verify_on_cpu(std::uint8_t value, std::size_t bytes) const override {
    if (!managed_) {
      throw std::logic_error("cuda-device memory is not CPU accessible");
    }
    check_cuda(cudaDeviceSynchronize(), "managed CPU verify synchronize");
    const auto* data = static_cast<const std::uint8_t*>(data_);
    return std::all_of(data, data + bytes,
                       [value](std::uint8_t byte) { return byte == value; });
  }

  bool verify_on_gpu(std::uint8_t value, std::size_t bytes) const override {
    unsigned long long* mismatches{};
    check_cuda(cudaMalloc(&mismatches, sizeof(*mismatches)),
               "cudaMalloc mismatches");
    check_cuda(cudaMemset(mismatches, 0, sizeof(*mismatches)),
               "cudaMemset mismatches");
    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes + threads - 1) / threads);
    verify_kernel<<<blocks, threads>>>(
        static_cast<const std::uint8_t*>(data_), bytes, value, mismatches);
    check_cuda(cudaGetLastError(), "allocation verify_kernel launch");
    unsigned long long host_mismatches{};
    check_cuda(cudaMemcpy(&host_mismatches, mismatches, sizeof(*mismatches),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy allocation mismatch result");
    cudaFree(mismatches);
    return host_mismatches == 0;
  }

 private:
  void* data_{};
  std::size_t bytes_{};
  bool managed_{};
};

}  // namespace

MemoryKind parse_memory_kind(std::string_view value) {
  if (value == "host") {
    return MemoryKind::kHost;
  }
  if (value == "cuda-mapped") {
    return MemoryKind::kCudaMapped;
  }
  if (value == "cuda-write-combined") {
    return MemoryKind::kCudaWriteCombined;
  }
  if (value == "cuda-device") {
    return MemoryKind::kCudaDevice;
  }
  if (value == "cuda-managed") {
    return MemoryKind::kCudaManaged;
  }
  throw std::invalid_argument("unknown memory kind: " + std::string(value));
}

const char* memory_kind_name(MemoryKind kind) {
  switch (kind) {
    case MemoryKind::kHost:
      return "host";
    case MemoryKind::kCudaMapped:
      return "cuda-mapped";
    case MemoryKind::kCudaWriteCombined:
      return "cuda-write-combined";
    case MemoryKind::kCudaDevice:
      return "cuda-device";
    case MemoryKind::kCudaManaged:
      return "cuda-managed";
  }
  return "unknown";
}

std::unique_ptr<MemoryBuffer> MemoryBuffer::allocate(MemoryKind kind,
                                                     std::size_t bytes) {
  if (bytes == 0) {
    throw std::invalid_argument("buffer size must be nonzero");
  }
  switch (kind) {
    case MemoryKind::kHost:
      return std::make_unique<HostBuffer>(bytes);
    case MemoryKind::kCudaMapped:
      return std::make_unique<CudaMappedBuffer>(bytes, false);
    case MemoryKind::kCudaWriteCombined:
      return std::make_unique<CudaMappedBuffer>(bytes, true);
    case MemoryKind::kCudaDevice:
      return std::make_unique<CudaAllocationBuffer>(bytes, false);
    case MemoryKind::kCudaManaged:
      return std::make_unique<CudaAllocationBuffer>(bytes, true);
  }
  throw std::invalid_argument("unsupported memory kind");
}

}  // namespace spark_transport
