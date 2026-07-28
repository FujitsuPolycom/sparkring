#pragma once

#include "spark_transport/tp4_graph_command.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace spark_transport::gpu_graph_command {

using AtomicWord = unsigned long long;
static_assert(sizeof(AtomicWord) == sizeof(std::uint64_t));

__device__ __forceinline__ AtomicWord* atomic_word(
    std::uint64_t* address) {
  return reinterpret_cast<AtomicWord*>(address);
}

__device__ __forceinline__ std::uint64_t load_system(
    const std::uint64_t* address) {
  auto* mutable_address = const_cast<std::uint64_t*>(address);
  AtomicWord current =
      atomicCAS_system(atomic_word(mutable_address), 0ULL, 0ULL);
  return static_cast<std::uint64_t>(current);
}

__device__ __forceinline__ void store_system(
    std::uint64_t* address, std::uint64_t value) {
  atomicExch_system(atomic_word(address), static_cast<AtomicWord>(value));
}

__device__ __forceinline__ void publish_overflow(
    Tp4GraphCommandRing* ring, std::uint64_t sequence) {
  atomicCAS_system(
      atomic_word(&ring->producer.overflow_sequence), 0ULL,
      static_cast<AtomicWord>(sequence));
}

__device__ __forceinline__ void fatal_wait() {
  while (true) {
    __nanosleep(4096);
  }
}

__device__ __forceinline__ std::uint64_t claim_sequence(
    Tp4GraphCommandRing* ring) {
  while (true) {
    if (load_system(&ring->producer.overflow_sequence) != 0) {
      fatal_wait();
    }
    const std::uint64_t claimed =
        load_system(&ring->producer.claimed_sequence);
    const std::uint64_t completed =
        load_system(&ring->consumer.completed_sequence);
    __threadfence_system();
    if (claimed >= kTp4GraphMaximumDoorbellSequence ||
        completed > claimed) {
      publish_overflow(
          ring,
          claimed >= kTp4GraphMaximumDoorbellSequence
              ? claimed
              : claimed + 1);
      fatal_wait();
    }
    if (claimed - completed >= kTp4GraphCommandCapacity) {
      __nanosleep(128);
      continue;
    }
    const AtomicWord previous = atomicCAS_system(
        atomic_word(&ring->producer.claimed_sequence),
        static_cast<AtomicWord>(claimed),
        static_cast<AtomicWord>(claimed + 1));
    if (previous == claimed) {
      return claimed + 1;
    }
  }
}

__device__ __forceinline__ std::uint64_t publish_command(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes,
    Tp4GraphCommandKind kind = Tp4GraphCommandKind::kLegacy,
    std::uint32_t parameter = 0) {
  const std::uint64_t sequence = claim_sequence(ring);
  auto& command =
      ring->commands[(sequence - 1) % kTp4GraphCommandCapacity];
  command.trace = trace ? 1U : 0U;
  command.q = q;
  command.payload_bytes = payload_bytes;
  command.kind = kind;
  command.parameter = parameter;
  __threadfence_system();
  store_system(&command.sequence, sequence);
  __threadfence_system();

  const std::uint64_t expected = sequence - 1;
  const AtomicWord previous = atomicCAS_system(
      atomic_word(&ring->producer.published_sequence),
      static_cast<AtomicWord>(expected),
      static_cast<AtomicWord>(sequence));
  if (previous != expected) {
    publish_overflow(ring, sequence);
    fatal_wait();
  }
  return sequence;
}

__device__ __forceinline__ void wait_for_sequence_block(
    const std::uint64_t* address, std::uint64_t expected,
    Tp4GraphCommandRing* graph_commands,
    std::uint64_t graph_sequence) {
  if (threadIdx.x == 0) {
    while (true) {
      const std::uint64_t observed =
          reinterpret_cast<const volatile std::uint64_t*>(address)[0];
      if (observed == expected ||
          (graph_commands == nullptr && observed > expected)) {
        break;
      }
      if (graph_commands != nullptr && observed > expected) {
        publish_overflow(graph_commands, graph_sequence);
        fatal_wait();
      }
      __nanosleep(64);
    }
  }
  __syncthreads();
}

}  // namespace spark_transport::gpu_graph_command
