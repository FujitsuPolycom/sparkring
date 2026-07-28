#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include <infiniband/verbs.h>

#include "spark_transport/memory_buffer.hpp"

namespace spark_transport {

constexpr std::uint32_t kEndpointMagic = 0x5350524b;  // "SPRK"
constexpr std::uint16_t kEndpointVersion = 1;

struct EndpointInfo {
  std::uint32_t magic{kEndpointMagic};
  std::uint16_t version{kEndpointVersion};
  std::uint16_t reserved{};
  std::uint32_t qp_number{};
  std::uint32_t rkey{};
  std::uint64_t address{};
  std::uint64_t buffer_bytes{};
  std::uint16_t lid{};
  std::uint8_t gid[16]{};
};

static_assert(std::is_trivially_copyable_v<EndpointInfo>);

class VerbsEndpoint {
 public:
  VerbsEndpoint(const VerbsEndpoint&) = delete;
  VerbsEndpoint& operator=(const VerbsEndpoint&) = delete;
  ~VerbsEndpoint();

  VerbsEndpoint(const std::string& device_name, std::uint8_t port,
                std::uint8_t gid_index, MemoryBuffer& buffer);

  EndpointInfo local_info() const;
  void connect(const EndpointInfo& remote);

  void write(std::size_t local_offset, std::size_t remote_offset,
             std::size_t bytes, std::uint64_t work_id,
             bool signaled = true);
  void wait_for_send(std::uint64_t expected_work_id);

 private:
  void cleanup() noexcept;

  std::uint8_t port_{};
  std::uint8_t gid_index_{};
  MemoryBuffer& buffer_;
  EndpointInfo remote_{};

  ibv_context* context_{};
  ibv_pd* protection_domain_{};
  ibv_mr* memory_region_{};
  ibv_cq* completion_queue_{};
  ibv_qp* queue_pair_{};
  std::uint32_t max_inline_data_{};
};

}  // namespace spark_transport
