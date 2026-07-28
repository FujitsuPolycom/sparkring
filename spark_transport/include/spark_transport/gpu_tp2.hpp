#pragma once

#include <cstddef>
#include <cstdint>

namespace spark_transport {

struct Tp2BufferLayout {
  std::size_t send_offset{};
  std::size_t receive_offset{};
  std::size_t control_offset{};
  std::size_t total_bytes{};
};

Tp2BufferLayout make_tp2_buffer_layout(std::size_t payload_bytes);

class GpuTp2Worker {
 public:
  GpuTp2Worker(const GpuTp2Worker&) = delete;
  GpuTp2Worker& operator=(const GpuTp2Worker&) = delete;

  explicit GpuTp2Worker(std::size_t payload_bytes);
  ~GpuTp2Worker();

  void launch(void* mapped_device_buffer, const Tp2BufferLayout& layout,
              std::uint32_t rank, std::uint64_t final_sequence);
  void synchronize();

 private:
  void* input_{};
  void* output_{};
  std::size_t payload_bytes_{};
};

}  // namespace spark_transport
