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
  // A session family may carry a setup-only compatibility tag here. The
  // generic endpoint validates magic/version and otherwise leaves it opaque.
  std::uint16_t reserved{};
  std::uint32_t qp_number{};
  std::uint32_t rkey{};
  std::uint64_t address{};
  std::uint64_t buffer_bytes{};
  std::uint16_t lid{};
  std::uint8_t gid[16]{};
};

static_assert(std::is_trivially_copyable_v<EndpointInfo>);

enum class SendCompletionPollState : std::uint8_t {
  kPending,
  kComplete,
};

namespace detail {

enum class SendCompletionDisposition : std::uint8_t {
  kRetire,
  kComplete,
  kUnexpected,
};

// Through polling uses ordinary unsigned order. Callers that permit older
// completions must assign monotonic work IDs and must not wrap while a
// completion remains outstanding on the endpoint.
constexpr SendCompletionDisposition classify_send_completion(
    std::uint64_t expected_work_id, std::uint64_t completed_work_id,
    bool allow_older) noexcept {
  if (completed_work_id == expected_work_id) {
    return SendCompletionDisposition::kComplete;
  }
  if (allow_older && completed_work_id < expected_work_id) {
    return SendCompletionDisposition::kRetire;
  }
  return SendCompletionDisposition::kUnexpected;
}

}  // namespace detail

class VerbsEndpoint {
 public:
  VerbsEndpoint(const VerbsEndpoint&) = delete;
  VerbsEndpoint& operator=(const VerbsEndpoint&) = delete;
  ~VerbsEndpoint();

  VerbsEndpoint(const std::string& device_name, std::uint8_t port,
                std::uint8_t gid_index, MemoryBuffer& buffer);

  EndpointInfo local_info() const;
  void connect(const EndpointInfo& remote,
               std::uint16_t expected_version = kEndpointVersion);

  void write(std::size_t local_offset, std::size_t remote_offset,
             std::size_t bytes, std::uint64_t work_id,
             bool signaled = true);
  void wait_for_send(std::uint64_t expected_work_id);
  // Waits for expected_work_id while retiring successful older completions.
  // Deferred-credit TP4 uses this to reap the preceding signaled credit WR
  // without putting that credit on the operation's output-ready path.
  void wait_for_send_through(std::uint64_t expected_work_id);
  // Performs a nonblocking form of wait_for_send_through for a signaled write.
  // Successful older CQEs are retired; kPending means the CQ was empty after
  // any available older CQEs were retired; and kComplete consumes
  // expected_work_id. Poll failures, failed CQEs, and newer work IDs throw.
  // The caller must serialize all completion methods and must not poll an ID
  // again after kComplete.
  SendCompletionPollState poll_send_through(
      std::uint64_t expected_work_id);

 private:
  void cleanup() noexcept;
  SendCompletionPollState poll_send_completion(
      std::uint64_t expected_work_id, bool allow_older);
  void wait_for_send_completion(std::uint64_t expected_work_id,
                                bool allow_older);

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
