#include "fused_prefill_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace spark_transport::tiled_prefill_research {
namespace {

constexpr std::size_t kPacketBytes = 16;

struct U32Packet {
  std::uint32_t words[4];
};

struct alignas(16) Bf16Packet {
  __nv_bfloat162 pairs[4];
};

static_assert(sizeof(Bf16Packet) == kPacketBytes);

__device__ std::uint64_t load_acquire_system_u64(const void* address) {
  std::uint64_t value{};
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ U32Packet load_relaxed_system_packet(const void* address) {
  U32Packet value{};
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.words[0]), "=r"(value.words[1]),
                 "=r"(value.words[2]), "=r"(value.words[3])
               : "l"(address)
               : "memory");
  return value;
}

__device__ void store_release_system_u64(void* address,
                                         std::uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}

__device__ std::uint32_t load_acquire_device_u32(const void* address) {
  std::uint32_t value{};
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ void store_release_device_u32(void* address,
                                         std::uint32_t value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;"
               :
               : "l"(address), "r"(value)
               : "memory");
}

__device__ void publish_poison(const FusedPrefillDescriptor& descriptor,
                               std::uint32_t stage) {
  atomicExch(&descriptor.device_sync->poison, stage + 1U);
  if (threadIdx.x == 0) {
    const std::uint64_t token =
        fused_prefill_stage_token(descriptor.operation_sequence, stage);
    store_release_system_u64(&descriptor.host_control->poison_sequence,
                             token);
  }
}

__device__ bool wait_exact_system(const std::uint64_t* address,
                                  std::uint64_t expected,
                                  const FusedPrefillDescriptor& descriptor,
                                  std::uint32_t stage) {
  for (std::uint32_t spin = 0; spin < descriptor.spin_limit; ++spin) {
    if (load_acquire_device_u32(&descriptor.device_sync->poison) != 0) {
      return false;
    }
    const std::uint64_t observed = load_acquire_system_u64(address);
    if (observed == expected) return true;
    // A future token means the CPU proxy reused a parity slot before this
    // consumer. That is a protocol violation, not a catch-up opportunity.
    if (observed > expected) {
      publish_poison(descriptor, stage);
      return false;
    }
    __nanosleep(64);
  }
  publish_poison(descriptor, stage);
  return false;
}

__device__ bool wait_inbound_stage(
    const FusedPrefillDescriptor& descriptor, std::uint32_t stage) {
  __shared__ std::uint32_t admitted;
  if (threadIdx.x == 0) {
    const std::uint32_t parity = fused_prefill_parity(stage);
    const std::uint64_t token =
        fused_prefill_stage_token(descriptor.operation_sequence, stage);
    bool ready = true;
    // Both rails must be visible before any CTA reads either payload. Reliable
    // QP ordering makes each acquire cover the preceding RDMA payload write.
    if (ready) {
      ready = wait_exact_system(
          &descriptor.host_control->primary_doorbell[parity], token,
          descriptor, stage);
    }
    if (ready) {
      ready = wait_exact_system(
          &descriptor.host_control->secondary_doorbell[parity], token,
          descriptor, stage);
    }
    admitted = ready ? 1U : 0U;
  }
  __syncthreads();
  return admitted != 0;
}

__device__ bool wait_previous_operation_reuse(
    const FusedPrefillDescriptor& descriptor) {
  if (descriptor.operation_sequence < descriptor.operation_slots) return true;
  __shared__ std::uint32_t reusable;
  if (threadIdx.x == 0) {
    const std::uint64_t previous =
        descriptor.operation_sequence - descriptor.operation_slots;
    const std::uint64_t parity_zero_token =
        fused_prefill_stage_token(previous, 4);
    const std::uint64_t parity_one_token =
        fused_prefill_stage_token(previous, 5);
    bool ready = wait_exact_system(&descriptor.host_control->reuse[0],
                                   parity_zero_token, descriptor, 0);
    if (ready) {
      ready = wait_exact_system(&descriptor.host_control->reuse[1],
                                parity_one_token, descriptor, 0);
    }
    reusable = ready ? 1U : 0U;
  }
  __syncthreads();
  return reusable != 0;
}

__device__ bool wait_final_reuse(
    const FusedPrefillDescriptor& descriptor) {
  __shared__ std::uint32_t retired;
  if (threadIdx.x == 0) {
    const std::uint64_t parity_zero_token = fused_prefill_stage_token(
        descriptor.operation_sequence, 4);
    const std::uint64_t parity_one_token = fused_prefill_stage_token(
        descriptor.operation_sequence, 5);
    bool ready = wait_exact_system(&descriptor.host_control->reuse[0],
                                   parity_zero_token, descriptor, 5);
    if (ready) {
      ready = wait_exact_system(&descriptor.host_control->reuse[1],
                                parity_one_token, descriptor, 5);
    }
    retired = ready ? 1U : 0U;
  }
  __syncthreads();
  return retired != 0;
}

__device__ bool flow_barrier(const FusedPrefillDescriptor& descriptor,
                             std::uint32_t phase,
                             std::uint32_t poison_stage) {
  __shared__ std::uint32_t arrived;
  if (threadIdx.x == 0) {
    const std::uint32_t prior = atomicAdd(
        &descriptor.device_sync->arrivals[phase], 1U);
    if (prior + 1U == kFusedPrefillCtasPerFlow) {
      __threadfence();
      atomicExch(&descriptor.device_sync->arrivals[phase], 0U);
      store_release_device_u32(
          &descriptor.device_sync->sense[phase],
          static_cast<std::uint32_t>(descriptor.operation_sequence + 1U));
    }
    arrived = 0;
    for (std::uint32_t spin = 0; spin < descriptor.spin_limit; ++spin) {
      const std::uint32_t observed =
          load_acquire_device_u32(&descriptor.device_sync->sense[phase]);
      const std::uint32_t expected =
          static_cast<std::uint32_t>(descriptor.operation_sequence + 1U);
      if (observed == expected) {
        arrived = 1;
        break;
      }
      if (observed > expected) break;
      if (load_acquire_device_u32(&descriptor.device_sync->poison) != 0) {
        break;
      }
      __nanosleep(64);
    }
    if (arrived == 0) publish_poison(descriptor, poison_stage);
  }
  __syncthreads();
  return arrived != 0;
}

__device__ Bf16Packet as_bf16(U32Packet value) {
  union PacketBits {
    U32Packet words;
    Bf16Packet bf16;
  } bits{};
  bits.words = value;
  return bits.bf16;
}

__device__ Bf16Packet reduce_bf16(const Bf16Packet& remote,
                                  const Bf16Packet& local) {
  Bf16Packet result{};
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    // Preserve the qualified cyclic association: incoming partial first,
    // then this rank's BF16 contribution.
    result.pairs[pair] = __hadd2(remote.pairs[pair], local.pairs[pair]);
  }
  return result;
}

__device__ bool descriptor_valid(const FusedPrefillDescriptor& descriptor,
                                 std::uint32_t flow) {
  const bool payload_valid =
      descriptor.payload_bytes != 0 &&
      descriptor.payload_bytes <= kFusedPrefillPayloadBytes &&
      descriptor.payload_bytes %
              (kFusedPrefillElementsPerRow * sizeof(__nv_bfloat16)) ==
          0;
  const std::uint64_t tile_bytes =
      descriptor.payload_bytes /
      (kFusedPrefillDirections * kFusedPrefillRanks *
       kFusedPrefillTilesPerShard);
  if (!payload_valid || tile_bytes == 0) return false;
  const std::uintptr_t planes[] = {
      reinterpret_cast<std::uintptr_t>(descriptor.primary_incoming),
      reinterpret_cast<std::uintptr_t>(descriptor.secondary_incoming),
      reinterpret_cast<std::uintptr_t>(descriptor.primary_outgoing),
      reinterpret_cast<std::uintptr_t>(descriptor.secondary_outgoing)};
  bool aligned_and_disjoint = true;
#pragma unroll
  for (std::uint32_t left = 0; left < 4; ++left) {
    aligned_and_disjoint &=
        planes[left] % kFusedPrefillPlaneAlignment == 0 &&
        planes[left] <= UINTPTR_MAX - kFusedPrefillRailPlaneBytes;
#pragma unroll
    for (std::uint32_t right = left + 1; right < 4; ++right) {
      const std::uintptr_t left_end =
          planes[left] + kFusedPrefillRailPlaneBytes;
      const std::uintptr_t right_end =
          planes[right] + kFusedPrefillRailPlaneBytes;
      aligned_and_disjoint &=
          left_end <= planes[right] || right_end <= planes[left];
    }
  }
  bool tensor_offsets_valid =
      descriptor.initial_tensor_offset_bytes % tile_bytes == 0;
#pragma unroll
  for (std::uint32_t stage = 0; stage < kFusedPrefillStages; ++stage) {
    tensor_offsets_valid &=
        descriptor.tensor_offset_bytes[stage] % tile_bytes == 0 &&
        descriptor.tensor_offset_bytes[stage] <=
            descriptor.payload_bytes - tile_bytes;
  }
  return descriptor.input != nullptr && descriptor.output != nullptr &&
         descriptor.primary_incoming != nullptr &&
         descriptor.secondary_incoming != nullptr &&
         descriptor.primary_outgoing != nullptr &&
         descriptor.secondary_outgoing != nullptr &&
         descriptor.host_control != nullptr &&
         descriptor.device_sync != nullptr &&
         aligned_and_disjoint &&
         tensor_offsets_valid &&
         reinterpret_cast<std::uintptr_t>(descriptor.host_control) %
                 alignof(FusedPrefillHostControl) ==
             0 &&
         reinterpret_cast<std::uintptr_t>(descriptor.device_sync) %
                 alignof(FusedPrefillDeviceSync) ==
             0 &&
         descriptor.rank < kFusedPrefillRanks &&
         (descriptor.direction == 1 || descriptor.direction == -1) &&
         descriptor.tile < kFusedPrefillTilesPerShard &&
         flow == (descriptor.direction == 1 ? 0U : 1U) *
                         kFusedPrefillTilesPerShard +
                     descriptor.tile &&
         descriptor.spin_limit != 0 &&
         descriptor.operation_slots != 0 &&
         descriptor.operation_slots <= 8 &&
         descriptor.initial_tensor_offset_bytes <=
             descriptor.payload_bytes - tile_bytes &&
         descriptor.operation_sequence <=
             (UINT64_MAX - kFusedPrefillStages) /
                 kFusedPrefillStages &&
         descriptor.operation_sequence < UINT32_MAX;
}

__global__ void fused_prefill_q8192_n4_kernel(
    const FusedPrefillDescriptor* descriptors) {
  const std::uint32_t flow = blockIdx.x / kFusedPrefillCtasPerFlow;
  const std::uint32_t lane = blockIdx.x % kFusedPrefillCtasPerFlow;
  if (flow >= kFusedPrefillFlows) return;
  const FusedPrefillDescriptor descriptor = descriptors[flow];
  if (!descriptor_valid(descriptor, flow)) {
    if (descriptor.host_control != nullptr &&
        descriptor.device_sync != nullptr) {
      publish_poison(descriptor, 0);
    }
    return;
  }

  const bool primary = lane < 2U;
  const std::uint32_t rail_lane = lane & 1U;
  const std::size_t tile_bytes = descriptor.payload_bytes /
      (kFusedPrefillDirections * kFusedPrefillRanks *
       kFusedPrefillTilesPerShard);
  const std::size_t cta_bytes =
      tile_bytes / kFusedPrefillCtasPerFlow;
  const std::uint8_t* incoming_base =
      primary ? descriptor.primary_incoming : descriptor.secondary_incoming;
  std::uint8_t* outgoing_base =
      primary ? descriptor.primary_outgoing : descriptor.secondary_outgoing;

  // Cross-operation reuse is distinct from same-operation parity reuse. A
  // an operation kernel may not seed parity zero or overwrite parity one until
  // the proxy has retired stage four and stage five from the previous op.
  if (!wait_previous_operation_reuse(descriptor)) return;

  // The GPU, not the CPU proxy, originates exchange zero. This initial phase
  // fills both outgoing rails before the flow leader publishes producer[0].
  const std::size_t initial_endpoint_offset =
      static_cast<std::size_t>(rail_lane) * cta_bytes;
  const std::size_t initial_tensor_offset =
      descriptor.initial_tensor_offset_bytes +
      static_cast<std::size_t>(lane) * cta_bytes;
  for (std::size_t byte = threadIdx.x * kPacketBytes;
       byte < cta_bytes;
       byte += blockDim.x * kPacketBytes) {
    reinterpret_cast<Bf16Packet*>(
        outgoing_base + initial_endpoint_offset + byte)[0] =
        reinterpret_cast<const Bf16Packet*>(
            descriptor.input + initial_tensor_offset + byte)[0];
  }
  __threadfence_system();
  if (!flow_barrier(descriptor, 0, 0)) return;
  if (lane == 0 && threadIdx.x == 0) {
    store_release_system_u64(
        &descriptor.host_control->producer[0],
        fused_prefill_stage_token(descriptor.operation_sequence, 0));
  }

  for (std::uint32_t stage = 0; stage < kFusedPrefillStages; ++stage) {
    if (!wait_inbound_stage(descriptor, stage)) return;
    const std::uint32_t parity = fused_prefill_parity(stage);
    const std::size_t endpoint_offset =
        static_cast<std::size_t>(parity) * kFusedPrefillRailBytes +
        static_cast<std::size_t>(rail_lane) * cta_bytes;
    const std::size_t tensor_offset =
        descriptor.tensor_offset_bytes[stage] +
        static_cast<std::size_t>(lane) * cta_bytes;
    const std::uint8_t* incoming = incoming_base + endpoint_offset;
    // Consuming stage s produces exchange s+1. Before overwriting that
    // parity bank, the proxy must have observed both local rail CQEs and the
    // peer's consumption credit for its previous use at s-1.
    if (stage + 1U < kFusedPrefillStages &&
        stage + 1U >= kFusedPrefillParitySlots) {
      __shared__ std::uint32_t reusable;
      if (threadIdx.x == 0) {
        const std::uint32_t next_parity = fused_prefill_parity(stage + 1U);
        const std::uint64_t prior_token = fused_prefill_stage_token(
            descriptor.operation_sequence,
            stage + 1U - kFusedPrefillParitySlots);
        reusable = wait_exact_system(
                       &descriptor.host_control->reuse[next_parity],
                       prior_token, descriptor, stage)
                       ? 1U
                       : 0U;
      }
      __syncthreads();
      if (reusable == 0) return;
    }

    const std::uint32_t next_parity = fused_prefill_parity(stage + 1U);
    const std::size_t next_endpoint_offset =
        static_cast<std::size_t>(next_parity) * kFusedPrefillRailBytes +
        static_cast<std::size_t>(rail_lane) * cta_bytes;
    std::uint8_t* next_outgoing = outgoing_base + next_endpoint_offset;

    for (std::size_t byte = threadIdx.x * kPacketBytes;
         byte < cta_bytes;
         byte += blockDim.x * kPacketBytes) {
      const U32Packet remote_words =
          load_relaxed_system_packet(incoming + byte);
      if (stage < 2U) {
        const auto local = reinterpret_cast<const Bf16Packet*>(
            descriptor.input + tensor_offset + byte)[0];
        reinterpret_cast<Bf16Packet*>(next_outgoing + byte)[0] =
            reduce_bf16(as_bf16(remote_words), local);
      } else if (stage == 2U) {
        const auto local = reinterpret_cast<const Bf16Packet*>(
            descriptor.input + tensor_offset + byte)[0];
        const Bf16Packet reduced =
            reduce_bf16(as_bf16(remote_words), local);
        reinterpret_cast<Bf16Packet*>(
            descriptor.output + tensor_offset + byte)[0] = reduced;
        reinterpret_cast<Bf16Packet*>(next_outgoing + byte)[0] = reduced;
      } else {
        reinterpret_cast<U32Packet*>(
            descriptor.output + tensor_offset + byte)[0] = remote_words;
        if (stage < kFusedPrefillStages - 1U) {
          reinterpret_cast<U32Packet*>(next_outgoing + byte)[0] = remote_words;
        }
      }
    }
    __threadfence_system();
    if (!flow_barrier(descriptor, stage + 1U, stage)) return;
    if (lane == 0 && threadIdx.x == 0) {
      const std::uint64_t token =
          fused_prefill_stage_token(descriptor.operation_sequence, stage);
      store_release_system_u64(
          &descriptor.host_control->consumer[parity], token);
      if (stage + 1U < kFusedPrefillStages) {
        const std::uint64_t next_token = fused_prefill_stage_token(
            descriptor.operation_sequence, stage + 1U);
        store_release_system_u64(
            &descriptor.host_control->producer[next_parity], next_token);
      }
    }
  }
  // Do not expose kernel completion until both parity banks have retired their
  // final use. This keeps caller-stream completion equivalent to safe MR and
  // slot reuse even when no subsequent operation is launched.
  (void)wait_final_reuse(descriptor);
}

}  // namespace

cudaError_t launch_fused_prefill_q8192_n4(
    const FusedPrefillDescriptor* device_descriptors,
    cudaStream_t stream) {
  if (device_descriptors == nullptr) return cudaErrorInvalidValue;
  int device{};
  int multiprocessors{};
  int active_blocks_per_multiprocessor{};
  cudaError_t result = cudaGetDevice(&device);
  if (result != cudaSuccess) return result;
  result = cudaDeviceGetAttribute(&multiprocessors,
                                  cudaDevAttrMultiProcessorCount, device);
  if (result != cudaSuccess) return result;
  result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_multiprocessor,
      fused_prefill_q8192_n4_kernel, kFusedPrefillThreads, 0);
  if (result != cudaSuccess) return result;
  constexpr int requested_blocks =
      kFusedPrefillFlows * kFusedPrefillCtasPerFlow;
  if (active_blocks_per_multiprocessor * multiprocessors <
      requested_blocks) {
    return cudaErrorNotSupported;
  }
  void* arguments[] = {
      const_cast<FusedPrefillDescriptor**>(&device_descriptors)};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(fused_prefill_q8192_n4_kernel),
      dim3(requested_blocks), dim3(kFusedPrefillThreads), arguments, 0,
      stream);
}

}  // namespace spark_transport::tiled_prefill_research
