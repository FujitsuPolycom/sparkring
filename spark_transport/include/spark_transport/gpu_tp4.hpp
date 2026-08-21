#pragma once

#include <cstddef>
#include <cstdint>

#include "spark_transport/gpu_doorbell.hpp"

namespace spark_transport {

class GpuTp4Worker {
 public:
  GpuTp4Worker(const GpuTp4Worker&) = delete;
  GpuTp4Worker& operator=(const GpuTp4Worker&) = delete;

  explicit GpuTp4Worker(std::size_t payload_bytes);
  ~GpuTp4Worker();

  void launch(void* round0_mapped_device_buffer,
              const ExchangeBufferLayout& round0_layout,
              void* round1_mapped_device_buffer,
              const ExchangeBufferLayout& round1_layout, std::uint32_t rank,
              std::uint64_t final_sequence);
  void synchronize();

 private:
  void* input_{};
  void* intermediate_{};
  void* output_{};
  std::size_t payload_bytes_{};
};

}  // namespace spark_transport
