#pragma once

#include <cstddef>
#include <cstdint>

namespace spark_transport {

struct alignas(64) DoorbellControl {
  std::uint64_t command_sequence{};
  std::uint64_t producer_sequence{};
  std::uint64_t remote_sequence{};
  std::uint64_t consumer_sequence{};
  std::uint64_t acknowledgement_sequence{};
  std::uint64_t observed_sequence{};
  std::uint64_t mismatch_count{};
  // The graph-only two-slot all-reduce stages its raw logical credit here
  // before an inline RDMA write. Other doorbell protocols leave it unused.
  std::uint64_t reserved{};
};

static_assert(sizeof(DoorbellControl) == 64);

std::size_t aligned_control_offset(std::size_t payload_bytes);

void launch_sender_doorbell(void* device_buffer, std::size_t payload_bytes,
                            std::size_t control_offset,
                            std::uint64_t final_sequence);

void launch_receiver_doorbell(void* device_buffer, std::size_t payload_bytes,
                              std::size_t control_offset,
                              std::uint64_t final_sequence,
                              bool verify_payload);

void synchronize_doorbell();

}  // namespace spark_transport
