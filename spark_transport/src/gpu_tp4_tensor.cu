#include "spark_transport/gpu_tp4_tensor.hpp"

#include "spark_transport/gpu_doorbell.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

using GraphAtomicWord = unsigned long long;
static_assert(sizeof(GraphAtomicWord) == sizeof(std::uint64_t));

constexpr std::size_t kSplitGraphTileBytes = 64U * 1024U;

struct SplitGraphOperationState {
  std::uint64_t sequence{};
  std::uint64_t doorbell_sequence{};
  std::size_t payload_bytes{};
  std::size_t slot{};
};

struct StripedGraphOperationState {
  std::uint64_t sequence{};
  std::uint64_t doorbell_sequence{};
  std::size_t payload_bytes{};
  std::size_t stripe_bytes{};
  std::size_t generation{};
};

// Every split node is captured on the session's one stable caller stream.
// Completion of each bulk grid therefore precedes its control publisher; the
// publisher's system fence is the payload-before-doorbell release boundary.

constexpr bool split_graph_q_supported(std::uint32_t q) noexcept {
  return q >= 1 && q <= kTp4GraphAllreduceMaximumQ;
}

__device__ GraphAtomicWord* graph_atomic_word(
    std::uint64_t* address) {
  return reinterpret_cast<GraphAtomicWord*>(address);
}

__device__ std::uint64_t graph_load_system(
    const std::uint64_t* address) {
  auto* mutable_address = const_cast<std::uint64_t*>(address);
  return atomicCAS_system(graph_atomic_word(mutable_address), 0ULL, 0ULL);
}

__device__ void graph_store_system(std::uint64_t* address,
                                   std::uint64_t value) {
  auto* atomic_address = graph_atomic_word(address);
  GraphAtomicWord current = atomicCAS_system(atomic_address, 0ULL, 0ULL);
  while (atomicCAS_system(atomic_address, current, value) != current) {
    current = atomicCAS_system(atomic_address, 0ULL, 0ULL);
  }
}

__device__ void graph_publish_overflow(Tp4GraphCommandRing* ring,
                                       std::uint64_t sequence) {
  atomicCAS_system(
      graph_atomic_word(&ring->producer.overflow_sequence), 0ULL,
      sequence);
}

__device__ __forceinline__ void graph_fatal_wait() {
  while (true) {
    __nanosleep(1024);
  }
}

__device__ std::uint64_t graph_claim_sequence(
    Tp4GraphCommandRing* ring) {
  while (true) {
    if (graph_load_system(&ring->producer.overflow_sequence) != 0) {
      graph_fatal_wait();
    }
    const std::uint64_t claimed =
        graph_load_system(&ring->producer.claimed_sequence);
    const std::uint64_t completed =
        graph_load_system(&ring->consumer.completed_sequence);
    __threadfence_system();
    if (claimed >= kTp4GraphMaximumDoorbellSequence ||
        completed > claimed) {
      graph_publish_overflow(
          ring,
          claimed >= kTp4GraphMaximumDoorbellSequence
              ? claimed
              : claimed + 1);
      graph_fatal_wait();
    }
    if (claimed - completed >= kTp4GraphCommandCapacity) {
      __nanosleep(64);
      continue;
    }
    const std::uint64_t next = claimed + 1;
    if (atomicCAS_system(
            graph_atomic_word(&ring->producer.claimed_sequence), claimed,
            next) == claimed) {
      return next;
    }
  }
}

__device__ std::uint64_t graph_publish_command(
    Tp4GraphCommandRing* ring, bool trace, std::uint32_t q,
    std::uint32_t payload_bytes) {
  const std::uint64_t sequence = graph_claim_sequence(ring);
  auto& command =
      ring->commands[(sequence - 1) % kTp4GraphCommandCapacity];
  command.trace = trace ? 1U : 0U;
  command.q = q;
  command.payload_bytes = payload_bytes;
  command.kind = Tp4GraphCommandKind::kLegacy;
  command.parameter = 0;
  __threadfence_system();
  graph_store_system(&command.sequence, sequence);
  __threadfence_system();

  while (true) {
    const std::uint64_t expected = sequence - 1;
    const std::uint64_t published =
        atomicCAS_system(
            graph_atomic_word(&ring->producer.published_sequence),
            expected, sequence);
    if (published == expected) {
      __threadfence_system();
      return sequence;
    }
    if (published >= sequence) {
      graph_publish_overflow(ring, sequence);
      graph_fatal_wait();
    }
    __nanosleep(64);
  }
}

__device__ void wait_for_sequence_block(const std::uint64_t* address,
                                        std::uint64_t sequence,
                                        Tp4GraphCommandRing* graph_commands,
                                        std::uint64_t graph_sequence) {
  if (threadIdx.x == 0) {
    while (true) {
      const std::uint64_t observed =
          reinterpret_cast<const volatile std::uint64_t*>(address)[0];
      if (observed == sequence ||
          (graph_commands == nullptr && observed > sequence)) {
        break;
      }
      if (graph_commands != nullptr && observed > sequence) {
        graph_publish_overflow(graph_commands, graph_sequence);
        graph_fatal_wait();
      }
      __nanosleep(64);
    }
  }
  __syncthreads();
}

__device__ void publish_sequence_block(std::uint64_t* address,
                                       std::uint64_t sequence) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    reinterpret_cast<volatile std::uint64_t*>(address)[0] = sequence;
  }
  __syncthreads();
}

__global__ void tp4_tensor_all_reduce(
    std::uint8_t* round0_buffer, ExchangeBufferLayout round0_layout,
    std::uint8_t* round1_buffer, ExchangeBufferLayout round1_layout,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    std::size_t payload_bytes, std::uint64_t fixed_sequence,
    Tp4GraphCommandRing* graph_commands, std::uint32_t graph_q,
    bool graph_trace, Tp4AllreduceProtocol protocol) {
  __shared__ std::uint64_t graph_sequence;
  if (graph_commands != nullptr) {
    if (threadIdx.x == 0) {
      graph_sequence =
          graph_publish_command(
              graph_commands, graph_trace, graph_q,
              static_cast<std::uint32_t>(payload_bytes));
    }
    __syncthreads();
  }
  const std::uint64_t sequence =
      graph_commands == nullptr ? fixed_sequence : graph_sequence;
  const std::uint64_t doorbell_sequence =
      graph_commands == nullptr
          ? sequence
          : (sequence << kTp4GraphDoorbellQBits) | graph_q;
  const std::size_t slot = tp4_payload_slot_index(sequence, protocol);
  round0_buffer += slot * round0_layout.total_bytes;
  round1_buffer += slot * round1_layout.total_bytes;

  auto* send0 = reinterpret_cast<__nv_bfloat16*>(
      round0_buffer + round0_layout.send_offset);
  const auto* receive0 = reinterpret_cast<const __nv_bfloat16*>(
      round0_buffer + round0_layout.receive_offset);
  auto* control0 = reinterpret_cast<DoorbellControl*>(
      round0_buffer + round0_layout.control_offset);

  auto* send1 = reinterpret_cast<__nv_bfloat16*>(
      round1_buffer + round1_layout.send_offset);
  const auto* receive1 = reinterpret_cast<const __nv_bfloat16*>(
      round1_buffer + round1_layout.receive_offset);
  auto* control1 = reinterpret_cast<DoorbellControl*>(
      round1_buffer + round1_layout.control_offset);
  const std::size_t elements = payload_bytes / sizeof(__nv_bfloat16);
  const std::size_t pairs = elements / 2;

  if (tp4_protocol_uses_deferred_ack(protocol)) {
    const std::uint64_t prior_sequence =
        tp4_expected_reuse_credit(sequence, protocol);
    wait_for_sequence_block(&control0->acknowledgement_sequence,
                            prior_sequence, graph_commands, sequence);
  }

  for (std::size_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    send0[index] = input[index];
  }
  publish_sequence_block(&control0->producer_sequence, doorbell_sequence);

  wait_for_sequence_block(&control0->remote_sequence, doorbell_sequence,
                          graph_commands, sequence);
  const auto* send0_pairs =
      reinterpret_cast<const __nv_bfloat162*>(send0);
  const auto* receive0_pairs =
      reinterpret_cast<const __nv_bfloat162*>(receive0);
  auto* send1_pairs = reinterpret_cast<__nv_bfloat162*>(send1);
  if (tp4_protocol_uses_deferred_ack(protocol)) {
    const std::uint64_t prior_sequence =
        tp4_expected_reuse_credit(sequence, protocol);
    wait_for_sequence_block(&control1->acknowledgement_sequence,
                            prior_sequence, graph_commands, sequence);
  }
  for (std::size_t index = threadIdx.x; index < pairs;
       index += blockDim.x) {
    send1_pairs[index] =
        __hadd2(send0_pairs[index], receive0_pairs[index]);
  }
  if (elements % 2 != 0 && threadIdx.x == 0) {
    send1[elements - 1] =
        __hadd(send0[elements - 1], receive0[elements - 1]);
  }
  publish_sequence_block(&control0->consumer_sequence,
                         doorbell_sequence);
  if (!tp4_protocol_uses_deferred_ack(protocol)) {
    wait_for_sequence_block(&control0->acknowledgement_sequence,
                            doorbell_sequence, graph_commands, sequence);
  }

  wait_for_sequence_block(&control1->remote_sequence, doorbell_sequence,
                          graph_commands, sequence);
  const auto* round1_send_pairs =
      reinterpret_cast<const __nv_bfloat162*>(send1);
  const auto* round1_receive_pairs =
      reinterpret_cast<const __nv_bfloat162*>(receive1);
  auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output);
  for (std::size_t index = threadIdx.x; index < pairs;
       index += blockDim.x) {
    output_pairs[index] =
        __hadd2(round1_send_pairs[index], round1_receive_pairs[index]);
  }
  if (elements % 2 != 0 && threadIdx.x == 0) {
    output[elements - 1] =
        __hadd(send1[elements - 1], receive1[elements - 1]);
  }
  publish_sequence_block(&control1->consumer_sequence,
                         doorbell_sequence);
  if (!tp4_protocol_uses_deferred_ack(protocol)) {
    wait_for_sequence_block(&control1->acknowledgement_sequence,
                            doorbell_sequence, graph_commands, sequence);
  }
  publish_sequence_block(&control1->observed_sequence,
                         doorbell_sequence);
}

__global__ void tp4_split_claim_command(
    Tp4GraphCommandRing* graph_commands,
    std::uint8_t* round0_buffer, ExchangeBufferLayout round0_layout,
    SplitGraphOperationState* state, std::uint32_t graph_q,
    std::uint32_t payload_bytes, bool graph_trace,
    Tp4AllreduceProtocol protocol) {
  if (threadIdx.x == 0) {
    const std::uint64_t sequence = graph_publish_command(
        graph_commands, graph_trace, graph_q, payload_bytes);
    state->sequence = sequence;
    state->doorbell_sequence =
        (sequence << kTp4GraphDoorbellQBits) | graph_q;
    state->payload_bytes = payload_bytes;
    state->slot = tp4_payload_slot_index(sequence, protocol);
  }
  __syncthreads();
  if (tp4_protocol_uses_deferred_ack(protocol)) {
    // acknowledgement_sequence is a raw logical sequence in deferred mode,
    // not the q-encoded transport token. This exact credit retires the prior
    // generation of the selected round-0 slot. Stream ordering prevents the
    // following staging grid from overwriting that slot until this returns.
    round0_buffer += state->slot * round0_layout.total_bytes;
    const auto* control0 = reinterpret_cast<const DoorbellControl*>(
        round0_buffer + round0_layout.control_offset);
    const std::uint64_t prior_sequence =
        tp4_expected_reuse_credit(state->sequence, protocol);
    wait_for_sequence_block(&control0->acknowledgement_sequence,
                            prior_sequence, graph_commands,
                            state->sequence);
  }
}

__global__ void tp4_split_stage_round0(
    std::uint8_t* round0_buffer, ExchangeBufferLayout round0_layout,
    const __nv_bfloat16* input,
    const SplitGraphOperationState* state) {
  round0_buffer += state->slot * round0_layout.total_bytes;
  auto* send0 = reinterpret_cast<__nv_bfloat16*>(
      round0_buffer + round0_layout.send_offset);
  const std::size_t begin_byte =
      static_cast<std::size_t>(blockIdx.x) * kSplitGraphTileBytes;
  const std::size_t candidate_end = begin_byte + kSplitGraphTileBytes;
  const std::size_t end_byte =
      candidate_end < state->payload_bytes
          ? candidate_end
          : state->payload_bytes;
  const std::size_t begin = begin_byte / sizeof(__nv_bfloat16);
  const std::size_t end = end_byte / sizeof(__nv_bfloat16);
  for (std::size_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
    send0[index] = input[index];
  }
}

__global__ void tp4_split_publish_round0(
    std::uint8_t* round0_buffer, ExchangeBufferLayout round0_layout,
    const SplitGraphOperationState* state) {
  round0_buffer += state->slot * round0_layout.total_bytes;
  auto* control0 = reinterpret_cast<DoorbellControl*>(
      round0_buffer + round0_layout.control_offset);
  publish_sequence_block(&control0->producer_sequence,
                         state->doorbell_sequence);
}

__global__ void tp4_split_wait_round0(
    std::uint8_t* round0_buffer, ExchangeBufferLayout round0_layout,
    std::uint8_t* round1_buffer, ExchangeBufferLayout round1_layout,
    const SplitGraphOperationState* state,
    Tp4GraphCommandRing* graph_commands,
    Tp4AllreduceProtocol protocol) {
  round0_buffer += state->slot * round0_layout.total_bytes;
  const auto* control0 = reinterpret_cast<const DoorbellControl*>(
      round0_buffer + round0_layout.control_offset);
  wait_for_sequence_block(&control0->remote_sequence,
                          state->doorbell_sequence, graph_commands,
                          state->sequence);
  if (tp4_protocol_uses_deferred_ack(protocol)) {
    // The reduce grid writes round-1 send storage. Its stream dependency on
    // this exact raw credit is the round-1 slot-reuse ownership barrier.
    round1_buffer += state->slot * round1_layout.total_bytes;
    const auto* control1 = reinterpret_cast<const DoorbellControl*>(
        round1_buffer + round1_layout.control_offset);
    const std::uint64_t prior_sequence =
        tp4_expected_reuse_credit(state->sequence, protocol);
    wait_for_sequence_block(&control1->acknowledgement_sequence,
                            prior_sequence, graph_commands,
                            state->sequence);
  }
}

__global__ void tp4_split_reduce_round0(
    std::uint8_t* round0_buffer, ExchangeBufferLayout round0_layout,
    std::uint8_t* round1_buffer, ExchangeBufferLayout round1_layout,
    const SplitGraphOperationState* state) {
  round0_buffer += state->slot * round0_layout.total_bytes;
  round1_buffer += state->slot * round1_layout.total_bytes;
  const auto* send0 = reinterpret_cast<const __nv_bfloat162*>(
      round0_buffer + round0_layout.send_offset);
  const auto* receive0 = reinterpret_cast<const __nv_bfloat162*>(
      round0_buffer + round0_layout.receive_offset);
  auto* send1 = reinterpret_cast<__nv_bfloat162*>(
      round1_buffer + round1_layout.send_offset);
  const std::size_t begin_byte =
      static_cast<std::size_t>(blockIdx.x) * kSplitGraphTileBytes;
  const std::size_t candidate_end = begin_byte + kSplitGraphTileBytes;
  const std::size_t end_byte =
      candidate_end < state->payload_bytes
          ? candidate_end
          : state->payload_bytes;
  const std::size_t begin = begin_byte / sizeof(__nv_bfloat162);
  const std::size_t end = end_byte / sizeof(__nv_bfloat162);
  for (std::size_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
    send1[index] = __hadd2(send0[index], receive0[index]);
  }
}

__global__ void tp4_split_handoff_round1(
    std::uint8_t* round0_buffer, ExchangeBufferLayout round0_layout,
    std::uint8_t* round1_buffer, ExchangeBufferLayout round1_layout,
    const SplitGraphOperationState* state,
    Tp4GraphCommandRing* graph_commands,
    Tp4AllreduceProtocol protocol) {
  round0_buffer += state->slot * round0_layout.total_bytes;
  round1_buffer += state->slot * round1_layout.total_bytes;
  auto* control0 = reinterpret_cast<DoorbellControl*>(
      round0_buffer + round0_layout.control_offset);
  const auto* control1 = reinterpret_cast<const DoorbellControl*>(
      round1_buffer + round1_layout.control_offset);
  publish_sequence_block(&control0->consumer_sequence,
                         state->doorbell_sequence);
  if (!tp4_protocol_uses_deferred_ack(protocol)) {
    wait_for_sequence_block(&control0->acknowledgement_sequence,
                            state->doorbell_sequence, graph_commands,
                            state->sequence);
  }
  // The CPU progress worker observes round-0 consumption and publishes the
  // round-1 producer doorbell before its ordered payload RDMA. The GPU must not
  // create a second round-1 producer.
  wait_for_sequence_block(&control1->remote_sequence,
                          state->doorbell_sequence, graph_commands,
                          state->sequence);
}

__global__ void tp4_split_reduce_round1(
    std::uint8_t* round1_buffer, ExchangeBufferLayout round1_layout,
    __nv_bfloat16* output,
    const SplitGraphOperationState* state) {
  round1_buffer += state->slot * round1_layout.total_bytes;
  const auto* send1 = reinterpret_cast<const __nv_bfloat162*>(
      round1_buffer + round1_layout.send_offset);
  const auto* receive1 = reinterpret_cast<const __nv_bfloat162*>(
      round1_buffer + round1_layout.receive_offset);
  auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output);
  const std::size_t begin_byte =
      static_cast<std::size_t>(blockIdx.x) * kSplitGraphTileBytes;
  const std::size_t candidate_end = begin_byte + kSplitGraphTileBytes;
  const std::size_t end_byte =
      candidate_end < state->payload_bytes
          ? candidate_end
          : state->payload_bytes;
  const std::size_t begin = begin_byte / sizeof(__nv_bfloat162);
  const std::size_t end = end_byte / sizeof(__nv_bfloat162);
  for (std::size_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
    output_pairs[index] = __hadd2(send1[index], receive1[index]);
  }
}

__global__ void tp4_split_finish(
    std::uint8_t* round1_buffer, ExchangeBufferLayout round1_layout,
    const SplitGraphOperationState* state,
    Tp4GraphCommandRing* graph_commands,
    Tp4AllreduceProtocol protocol) {
  round1_buffer += state->slot * round1_layout.total_bytes;
  auto* control1 = reinterpret_cast<DoorbellControl*>(
      round1_buffer + round1_layout.control_offset);
  publish_sequence_block(&control1->consumer_sequence,
                         state->doorbell_sequence);
  if (!tp4_protocol_uses_deferred_ack(protocol)) {
    wait_for_sequence_block(&control1->acknowledgement_sequence,
                            state->doorbell_sequence, graph_commands,
                            state->sequence);
  }
  publish_sequence_block(&control1->observed_sequence,
                         state->doorbell_sequence);
}

__device__ std::size_t tp4_striped_lane_offset(
    const Tp4StripedEndpointLayout& layout, std::size_t generation,
    Tp4TensorStripe stripe) {
  return generation * layout.generation_stride +
         static_cast<std::size_t>(stripe) * layout.lane_stride;
}

__device__ DoorbellControl* tp4_striped_control(
    std::uint8_t* endpoint_buffer,
    const Tp4StripedEndpointLayout& layout, std::size_t generation,
    Tp4TensorStripe stripe) {
  return reinterpret_cast<DoorbellControl*>(
      endpoint_buffer +
      tp4_striped_lane_offset(layout, generation, stripe) +
      layout.lane_control_offset);
}

__global__ void tp4_striped_claim_and_gate(
    Tp4GraphCommandRing* graph_commands,
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout,
    StripedGraphOperationState* state, std::uint32_t graph_q,
    std::uint32_t payload_bytes, bool graph_trace,
    Tp4AllreduceProtocol protocol) {
  if (threadIdx.x == 0) {
    const std::uint64_t sequence = graph_publish_command(
        graph_commands, graph_trace, graph_q, payload_bytes);
    state->sequence = sequence;
    state->doorbell_sequence =
        (sequence << kTp4GraphDoorbellQBits) | graph_q;
    state->payload_bytes = payload_bytes;
    state->stripe_bytes = payload_bytes / 2U;
    state->generation = static_cast<std::size_t>(
        (sequence - 1U) % kTp4StripedGenerationCount);
  }
  __syncthreads();

  // This is one generation-ownership transaction. No lane may be overwritten
  // until the peer has returned the exact raw S-2 credit for all four lane
  // controls in the selected generation.
  const std::uint64_t prior_sequence =
      tp4_expected_reuse_credit(state->sequence, protocol);
  const auto* endpoint0_lower = tp4_striped_control(
      endpoint0_buffer, layout, state->generation,
      Tp4TensorStripe::kLowerHalf);
  const auto* endpoint0_upper = tp4_striped_control(
      endpoint0_buffer, layout, state->generation,
      Tp4TensorStripe::kUpperHalf);
  const auto* endpoint1_lower = tp4_striped_control(
      endpoint1_buffer, layout, state->generation,
      Tp4TensorStripe::kLowerHalf);
  const auto* endpoint1_upper = tp4_striped_control(
      endpoint1_buffer, layout, state->generation,
      Tp4TensorStripe::kUpperHalf);
  wait_for_sequence_block(
      &endpoint0_lower->acknowledgement_sequence, prior_sequence,
      graph_commands, state->sequence);
  wait_for_sequence_block(
      &endpoint0_upper->acknowledgement_sequence, prior_sequence,
      graph_commands, state->sequence);
  wait_for_sequence_block(
      &endpoint1_lower->acknowledgement_sequence, prior_sequence,
      graph_commands, state->sequence);
  wait_for_sequence_block(
      &endpoint1_upper->acknowledgement_sequence, prior_sequence,
      graph_commands, state->sequence);
}

__global__ void tp4_striped_stage_phase1(
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout, const __nv_bfloat16* input,
    const StripedGraphOperationState* state) {
  const std::size_t block = static_cast<std::size_t>(blockIdx.x);
  const std::size_t stripe_index = block & 1U;
  const std::size_t tile = block >> 1U;
  const Tp4TensorStripe stripe =
      stripe_index == 0 ? Tp4TensorStripe::kLowerHalf
                        : Tp4TensorStripe::kUpperHalf;
  std::uint8_t* endpoint =
      stripe == Tp4TensorStripe::kLowerHalf
          ? endpoint0_buffer
          : endpoint1_buffer;
  auto* send = reinterpret_cast<__nv_bfloat16*>(
      endpoint + tp4_striped_lane_offset(
                     layout, state->generation, stripe));
  const std::size_t begin_byte = tile * kSplitGraphTileBytes;
  const std::size_t candidate_end = begin_byte + kSplitGraphTileBytes;
  const std::size_t end_byte =
      candidate_end < state->stripe_bytes
          ? candidate_end
          : state->stripe_bytes;
  const std::size_t begin = begin_byte / sizeof(__nv_bfloat16);
  const std::size_t end = end_byte / sizeof(__nv_bfloat16);
  const std::size_t input_offset =
      stripe_index * (state->stripe_bytes / sizeof(__nv_bfloat16));
  for (std::size_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
    send[index] = input[input_offset + index];
  }
}

__global__ void tp4_striped_publish_phase1(
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout,
    const StripedGraphOperationState* state) {
  auto* endpoint0_lower = tp4_striped_control(
      endpoint0_buffer, layout, state->generation,
      Tp4TensorStripe::kLowerHalf);
  auto* endpoint1_upper = tp4_striped_control(
      endpoint1_buffer, layout, state->generation,
      Tp4TensorStripe::kUpperHalf);
  publish_sequence_block(&endpoint0_lower->producer_sequence,
                         state->doorbell_sequence);
  publish_sequence_block(&endpoint1_upper->producer_sequence,
                         state->doorbell_sequence);
}

__global__ void tp4_striped_wait_phase1(
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout,
    const StripedGraphOperationState* state,
    Tp4GraphCommandRing* graph_commands) {
  const auto* endpoint0_lower = tp4_striped_control(
      endpoint0_buffer, layout, state->generation,
      Tp4TensorStripe::kLowerHalf);
  const auto* endpoint1_upper = tp4_striped_control(
      endpoint1_buffer, layout, state->generation,
      Tp4TensorStripe::kUpperHalf);
  wait_for_sequence_block(&endpoint0_lower->remote_sequence,
                          state->doorbell_sequence, graph_commands,
                          state->sequence);
  wait_for_sequence_block(&endpoint1_upper->remote_sequence,
                          state->doorbell_sequence, graph_commands,
                          state->sequence);
}

__global__ void tp4_striped_reduce_phase1(
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout,
    const StripedGraphOperationState* state) {
  const std::size_t block = static_cast<std::size_t>(blockIdx.x);
  const std::size_t stripe_index = block & 1U;
  const std::size_t tile = block >> 1U;
  const Tp4TensorStripe stripe =
      stripe_index == 0 ? Tp4TensorStripe::kLowerHalf
                        : Tp4TensorStripe::kUpperHalf;
  std::uint8_t* source_endpoint =
      stripe == Tp4TensorStripe::kLowerHalf
          ? endpoint0_buffer
          : endpoint1_buffer;
  std::uint8_t* destination_endpoint =
      stripe == Tp4TensorStripe::kLowerHalf
          ? endpoint1_buffer
          : endpoint0_buffer;
  const std::size_t source_lane = tp4_striped_lane_offset(
      layout, state->generation, stripe);
  const std::size_t destination_lane = tp4_striped_lane_offset(
      layout, state->generation, stripe);
  const auto* send = reinterpret_cast<const __nv_bfloat162*>(
      source_endpoint + source_lane);
  const auto* receive = reinterpret_cast<const __nv_bfloat162*>(
      source_endpoint + source_lane + layout.lane_receive_offset);
  auto* phase2_send = reinterpret_cast<__nv_bfloat162*>(
      destination_endpoint + destination_lane);
  const std::size_t begin_byte = tile * kSplitGraphTileBytes;
  const std::size_t candidate_end = begin_byte + kSplitGraphTileBytes;
  const std::size_t end_byte =
      candidate_end < state->stripe_bytes
          ? candidate_end
          : state->stripe_bytes;
  const std::size_t begin = begin_byte / sizeof(__nv_bfloat162);
  const std::size_t end = end_byte / sizeof(__nv_bfloat162);
  for (std::size_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
    phase2_send[index] = __hadd2(send[index], receive[index]);
  }
}

__global__ void tp4_striped_handoff_phase2(
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout,
    const StripedGraphOperationState* state,
    Tp4GraphCommandRing* graph_commands) {
  auto* endpoint0_lower = tp4_striped_control(
      endpoint0_buffer, layout, state->generation,
      Tp4TensorStripe::kLowerHalf);
  auto* endpoint1_upper = tp4_striped_control(
      endpoint1_buffer, layout, state->generation,
      Tp4TensorStripe::kUpperHalf);
  const auto* endpoint1_lower = tp4_striped_control(
      endpoint1_buffer, layout, state->generation,
      Tp4TensorStripe::kLowerHalf);
  const auto* endpoint0_upper = tp4_striped_control(
      endpoint0_buffer, layout, state->generation,
      Tp4TensorStripe::kUpperHalf);
  publish_sequence_block(&endpoint0_lower->consumer_sequence,
                         state->doorbell_sequence);
  publish_sequence_block(&endpoint1_upper->consumer_sequence,
                         state->doorbell_sequence);
  // The host progress engine owns both phase-2 producer publications after it
  // observes the phase-1 consumers. The GPU only waits for the ordered remote
  // phase-2 payload+doorbell writes.
  wait_for_sequence_block(&endpoint1_lower->remote_sequence,
                          state->doorbell_sequence, graph_commands,
                          state->sequence);
  wait_for_sequence_block(&endpoint0_upper->remote_sequence,
                          state->doorbell_sequence, graph_commands,
                          state->sequence);
}

__global__ void tp4_striped_reduce_phase2(
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout, __nv_bfloat16* output,
    const StripedGraphOperationState* state) {
  const std::size_t block = static_cast<std::size_t>(blockIdx.x);
  const std::size_t stripe_index = block & 1U;
  const std::size_t tile = block >> 1U;
  const Tp4TensorStripe stripe =
      stripe_index == 0 ? Tp4TensorStripe::kLowerHalf
                        : Tp4TensorStripe::kUpperHalf;
  std::uint8_t* endpoint =
      stripe == Tp4TensorStripe::kLowerHalf
          ? endpoint1_buffer
          : endpoint0_buffer;
  const std::size_t lane = tp4_striped_lane_offset(
      layout, state->generation, stripe);
  const auto* send = reinterpret_cast<const __nv_bfloat162*>(
      endpoint + lane);
  const auto* receive = reinterpret_cast<const __nv_bfloat162*>(
      endpoint + lane + layout.lane_receive_offset);
  auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output);
  const std::size_t begin_byte = tile * kSplitGraphTileBytes;
  const std::size_t candidate_end = begin_byte + kSplitGraphTileBytes;
  const std::size_t end_byte =
      candidate_end < state->stripe_bytes
          ? candidate_end
          : state->stripe_bytes;
  const std::size_t begin = begin_byte / sizeof(__nv_bfloat162);
  const std::size_t end = end_byte / sizeof(__nv_bfloat162);
  const std::size_t output_offset =
      stripe_index * (state->stripe_bytes / sizeof(__nv_bfloat162));
  for (std::size_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
    output_pairs[output_offset + index] =
        __hadd2(send[index], receive[index]);
  }
}

__global__ void tp4_striped_finish(
    std::uint8_t* endpoint0_buffer, std::uint8_t* endpoint1_buffer,
    Tp4StripedEndpointLayout layout,
    const StripedGraphOperationState* state) {
  auto* endpoint1_lower = tp4_striped_control(
      endpoint1_buffer, layout, state->generation,
      Tp4TensorStripe::kLowerHalf);
  auto* endpoint0_upper = tp4_striped_control(
      endpoint0_buffer, layout, state->generation,
      Tp4TensorStripe::kUpperHalf);
  publish_sequence_block(&endpoint1_lower->consumer_sequence,
                         state->doorbell_sequence);
  publish_sequence_block(&endpoint0_upper->consumer_sequence,
                         state->doorbell_sequence);
  publish_sequence_block(&endpoint1_lower->observed_sequence,
                         state->doorbell_sequence);
  publish_sequence_block(&endpoint0_upper->observed_sequence,
                         state->doorbell_sequence);
}

}  // namespace

GpuTp4TensorWorker::GpuTp4TensorWorker(
    std::size_t payload_bytes, std::uint32_t bytes_per_row,
    void* round0_mapped_device_buffer,
    const ExchangeBufferLayout& round0_layout,
    void* round1_mapped_device_buffer,
    const ExchangeBufferLayout& round1_layout,
    Tp4AllreduceProtocol protocol,
    Tp4GraphKernelStrategy graph_kernel_strategy,
    Tp4AllreduceSchedule schedule)
    : round0_buffer_(round0_mapped_device_buffer),
      round1_buffer_(round1_mapped_device_buffer),
      round0_layout_(round0_layout),
      round1_layout_(round1_layout),
      payload_bytes_(payload_bytes),
      bytes_per_row_(bytes_per_row),
      protocol_(protocol),
      graph_kernel_strategy_(graph_kernel_strategy),
      schedule_(schedule) {
  if (payload_bytes_ == 0 || bytes_per_row_ == 0 ||
      payload_bytes_ % sizeof(__nv_bfloat16) != 0) {
    throw std::invalid_argument(
        "TP4 tensor size must be nonzero BF16 data");
  }
  if (round0_buffer_ == nullptr || round1_buffer_ == nullptr) {
    throw std::invalid_argument("TP4 tensor worker received a null buffer");
  }
  if (!tp4_graph_kernel_strategy_valid(graph_kernel_strategy_)) {
    throw std::invalid_argument("invalid graph TP4 kernel strategy");
  }
  if (!tp4_allreduce_schedule_valid(schedule_)) {
    throw std::invalid_argument("invalid TP4 all-reduce schedule");
  }
  if (schedule_ == Tp4AllreduceSchedule::kDualPortStriped) {
    if (!tp4_protocol_uses_deferred_ack(protocol_) ||
        graph_kernel_strategy_ != Tp4GraphKernelStrategy::kFused) {
      throw std::invalid_argument(
          "dual_port_striped graph TP4 requires fused kernel selection "
          "and two_slot_deferred_ack");
    }
    static_cast<void>(
        make_tp4_striped_endpoint_layout(payload_bytes_));
    check_cuda(cudaMalloc(&striped_graph_state_,
                          sizeof(StripedGraphOperationState)),
               "cudaMalloc striped graph TP4 state");
  } else if (tp4_graph_kernel_strategy_is_graph_only(
                 graph_kernel_strategy_)) {
    check_cuda(cudaMalloc(&split_graph_state_,
                          sizeof(SplitGraphOperationState)),
               "cudaMalloc split graph TP4 state");
  }
}

GpuTp4TensorWorker::~GpuTp4TensorWorker() {
  if (split_graph_state_ != nullptr) {
    (void)cudaFree(split_graph_state_);
  }
  if (striped_graph_state_ != nullptr) {
    (void)cudaFree(striped_graph_state_);
  }
}

void GpuTp4TensorWorker::enqueue(const void* external_input,
                                 void* external_output, void* cuda_stream,
                                 std::uint64_t sequence) {
  if (external_input == nullptr || external_output == nullptr ||
      sequence == 0) {
    throw std::invalid_argument("invalid TP4 tensor operation");
  }
  if (schedule_ != Tp4AllreduceSchedule::kSequential) {
    throw std::logic_error(
        "dual_port_striped TP4 is restricted to graph all-reduce");
  }
  if (tp4_protocol_uses_deferred_ack(protocol_)) {
    throw std::logic_error(
        "two-slot deferred ACK is restricted to graph TP4 all-reduce");
  }
  if (graph_kernel_strategy_ != Tp4GraphKernelStrategy::kFused) {
    throw std::logic_error(
        "research TP4 kernel strategies are restricted to graph all-reduce");
  }
  const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
  constexpr int threads = 256;
  check_cuda(
      cudaGetLastError(),
      "prior CUDA error before tp4_tensor_all_reduce launch");
  tp4_tensor_all_reduce<<<1, threads, 0, caller_stream>>>(
      static_cast<std::uint8_t*>(round0_buffer_), round0_layout_,
      static_cast<std::uint8_t*>(round1_buffer_), round1_layout_,
      static_cast<const __nv_bfloat16*>(external_input),
      static_cast<__nv_bfloat16*>(external_output), payload_bytes_, sequence,
      nullptr, 0, false, protocol_);
  check_cuda(cudaGetLastError(), "tp4_tensor_all_reduce launch");
}

void GpuTp4TensorWorker::enqueue_graph(
    const void* external_input, void* external_output, std::uint32_t q,
    void* cuda_stream, Tp4GraphCommandRing* command_ring, bool trace) {
  const std::uint32_t active_payload_bytes =
      tp4_graph_payload_bytes(q, bytes_per_row_);
  if (!tp4_graph_command_layout_valid(
          q, active_payload_bytes, bytes_per_row_,
          kTp4GraphAllreduceMaximumQ) ||
      active_payload_bytes > payload_bytes_) {
    throw std::invalid_argument(
        "graph TP4 all-reduce requires the configured BF16 [q, width], "
        "q in [1, 512], "
        "within session capacity");
  }
  if (external_input == nullptr || external_output == nullptr ||
      command_ring == nullptr) {
    throw std::invalid_argument("invalid graph TP4 tensor operation");
  }
  const auto caller_stream = static_cast<cudaStream_t>(cuda_stream);
  if (schedule_ == Tp4AllreduceSchedule::kDualPortStriped) {
    if (active_payload_bytes != payload_bytes_ ||
        !tp4_protocol_uses_deferred_ack(protocol_) ||
        graph_kernel_strategy_ != Tp4GraphKernelStrategy::kFused ||
        striped_graph_state_ == nullptr) {
      throw std::invalid_argument(
          "dual_port_striped graph TP4 requires one fixed-Q fused/deferred "
          "session with initialized striped state");
    }
    const Tp4StripedEndpointLayout striped_layout =
        make_tp4_striped_endpoint_layout(active_payload_bytes);
    constexpr int control_threads = 32;
    constexpr int bulk_threads = 256;
    const int stripe_blocks = static_cast<int>(
        (striped_layout.stripe_bytes + kSplitGraphTileBytes - 1) /
        kSplitGraphTileBytes);
    const int bulk_blocks = 2 * stripe_blocks;
    auto* const state = static_cast<StripedGraphOperationState*>(
        striped_graph_state_);
    auto* const endpoint0 = static_cast<std::uint8_t*>(round0_buffer_);
    auto* const endpoint1 = static_cast<std::uint8_t*>(round1_buffer_);

    tp4_striped_claim_and_gate<<<1, control_threads, 0, caller_stream>>>(
        command_ring, endpoint0, endpoint1, striped_layout, state, q,
        active_payload_bytes, trace, protocol_);
    check_cuda(cudaGetLastError(), "striped graph TP4 claim/gate launch");
    tp4_striped_stage_phase1<<<bulk_blocks, bulk_threads, 0, caller_stream>>>(
        endpoint0, endpoint1, striped_layout,
        static_cast<const __nv_bfloat16*>(external_input), state);
    check_cuda(cudaGetLastError(), "striped graph TP4 phase-1 stage launch");
    tp4_striped_publish_phase1<<<1, control_threads, 0, caller_stream>>>(
        endpoint0, endpoint1, striped_layout, state);
    check_cuda(cudaGetLastError(),
               "striped graph TP4 phase-1 publish launch");
    tp4_striped_wait_phase1<<<1, control_threads, 0, caller_stream>>>(
        endpoint0, endpoint1, striped_layout, state, command_ring);
    check_cuda(cudaGetLastError(), "striped graph TP4 phase-1 wait launch");
    tp4_striped_reduce_phase1<<<bulk_blocks, bulk_threads, 0, caller_stream>>>(
        endpoint0, endpoint1, striped_layout, state);
    check_cuda(cudaGetLastError(),
               "striped graph TP4 phase-1 reduce launch");
    tp4_striped_handoff_phase2<<<1, control_threads, 0, caller_stream>>>(
        endpoint0, endpoint1, striped_layout, state, command_ring);
    check_cuda(cudaGetLastError(),
               "striped graph TP4 phase-2 handoff launch");
    tp4_striped_reduce_phase2<<<bulk_blocks, bulk_threads, 0, caller_stream>>>(
        endpoint0, endpoint1, striped_layout,
        static_cast<__nv_bfloat16*>(external_output), state);
    check_cuda(cudaGetLastError(),
               "striped graph TP4 phase-2 reduce launch");
    tp4_striped_finish<<<1, control_threads, 0, caller_stream>>>(
        endpoint0, endpoint1, striped_layout, state);
    check_cuda(cudaGetLastError(), "striped graph TP4 finish launch");
    return;
  }
  if (tp4_graph_kernel_uses_split(graph_kernel_strategy_, q)) {
    if (!split_graph_q_supported(q) || split_graph_state_ == nullptr) {
      throw std::invalid_argument(
          "split_64k graph TP4 requires Q1 through Q512 and initialized "
          "split state");
    }
    constexpr int control_threads = 32;
    constexpr int bulk_threads = 256;
    const int bulk_blocks = static_cast<int>(
        (active_payload_bytes + kSplitGraphTileBytes - 1) /
        kSplitGraphTileBytes);
    auto* const state = static_cast<SplitGraphOperationState*>(
        split_graph_state_);
    auto* const round0 = static_cast<std::uint8_t*>(round0_buffer_);
    auto* const round1 = static_cast<std::uint8_t*>(round1_buffer_);

    tp4_split_claim_command<<<1, control_threads, 0, caller_stream>>>(
        command_ring, round0, round0_layout_, state, q,
        active_payload_bytes, trace, protocol_);
    check_cuda(cudaGetLastError(), "split graph TP4 claim launch");
    tp4_split_stage_round0<<<bulk_blocks, bulk_threads, 0, caller_stream>>>(
        round0, round0_layout_,
        static_cast<const __nv_bfloat16*>(external_input), state);
    check_cuda(cudaGetLastError(), "split graph TP4 round-0 stage launch");
    tp4_split_publish_round0<<<1, control_threads, 0, caller_stream>>>(
        round0, round0_layout_, state);
    check_cuda(cudaGetLastError(), "split graph TP4 round-0 publish launch");
    tp4_split_wait_round0<<<1, control_threads, 0, caller_stream>>>(
        round0, round0_layout_, round1, round1_layout_, state,
        command_ring, protocol_);
    check_cuda(cudaGetLastError(), "split graph TP4 round-0 wait launch");
    tp4_split_reduce_round0<<<bulk_blocks, bulk_threads, 0, caller_stream>>>(
        round0, round0_layout_, round1, round1_layout_, state);
    check_cuda(cudaGetLastError(), "split graph TP4 round-0 reduce launch");
    tp4_split_handoff_round1<<<1, control_threads, 0, caller_stream>>>(
        round0, round0_layout_, round1, round1_layout_, state,
        command_ring, protocol_);
    check_cuda(cudaGetLastError(), "split graph TP4 round-1 handoff launch");
    tp4_split_reduce_round1<<<bulk_blocks, bulk_threads, 0, caller_stream>>>(
        round1, round1_layout_,
        static_cast<__nv_bfloat16*>(external_output), state);
    check_cuda(cudaGetLastError(), "split graph TP4 round-1 reduce launch");
    tp4_split_finish<<<1, control_threads, 0, caller_stream>>>(
        round1, round1_layout_, state, command_ring, protocol_);
    check_cuda(cudaGetLastError(), "split graph TP4 finish launch");
    return;
  }
  constexpr int threads = 256;
  tp4_tensor_all_reduce<<<1, threads, 0, caller_stream>>>(
      static_cast<std::uint8_t*>(round0_buffer_), round0_layout_,
      static_cast<std::uint8_t*>(round1_buffer_), round1_layout_,
      static_cast<const __nv_bfloat16*>(external_input),
      static_cast<__nv_bfloat16*>(external_output), active_payload_bytes, 0,
      command_ring, q, trace, protocol_);
  check_cuda(cudaGetLastError(), "graph tp4_tensor_all_reduce launch");
}

}  // namespace spark_transport
