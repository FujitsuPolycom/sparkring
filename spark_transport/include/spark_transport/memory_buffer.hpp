#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace spark_transport {

enum class MemoryKind {
  kHost,
  kCudaMapped,
  kCudaWriteCombined,
  kCudaDevice,
  kCudaManaged,
};

MemoryKind parse_memory_kind(std::string_view value);
const char* memory_kind_name(MemoryKind kind);

class MemoryBuffer {
 public:
  MemoryBuffer(const MemoryBuffer&) = delete;
  MemoryBuffer& operator=(const MemoryBuffer&) = delete;
  virtual ~MemoryBuffer() = default;

  static std::unique_ptr<MemoryBuffer> allocate(MemoryKind kind,
                                                std::size_t bytes);

  virtual void* host_data() const = 0;
  virtual void* device_data() const = 0;
  virtual std::size_t size() const = 0;
  virtual MemoryKind kind() const = 0;

  virtual void fill_from_cpu(std::uint8_t value) = 0;
  virtual void fill_from_gpu(std::uint8_t value) = 0;
  virtual bool verify_on_cpu(std::uint8_t value, std::size_t bytes) const = 0;
  virtual bool verify_on_gpu(std::uint8_t value, std::size_t bytes) const = 0;

 protected:
  MemoryBuffer() = default;
};

}  // namespace spark_transport
