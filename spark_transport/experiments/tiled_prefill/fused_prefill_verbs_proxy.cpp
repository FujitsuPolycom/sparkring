#include "fused_prefill_verbs_proxy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace spark_transport::tiled_prefill_research {
namespace {

constexpr std::uint32_t kCompletionBatch = 32;

std::uint64_t control_field_offset(std::uint32_t flow,
                                   std::size_t field_offset,
                                   std::uint32_t parity) {
  return fused_prefill_control_offset(flow) + field_offset +
         static_cast<std::uint64_t>(parity) * sizeof(std::uint64_t);
}

std::uint64_t load_acquire(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_release(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

enum class TokenState : std::uint8_t { kPast, kExact, kFuture };

TokenState classify_token(std::uint64_t observed, std::uint64_t expected) {
  if (observed == expected) return TokenState::kExact;
  return observed < expected ? TokenState::kPast : TokenState::kFuture;
}

std::int32_t flow_direction(std::uint32_t flow) {
  return flow < kFusedPrefillTilesPerShard ? 1 : -1;
}

Tp4PrefillDirection as_direction(std::int32_t direction) {
  return direction == 1 ? Tp4PrefillDirection::kClockwise
                        : Tp4PrefillDirection::kCounterClockwise;
}

std::uint64_t tensor_offset(std::uint32_t rank, std::int32_t direction,
                            std::uint32_t tile, std::uint32_t stage,
                            std::uint64_t payload_bytes) {
  const auto transfer = tp4_prefill_stage(rank, as_direction(direction), stage);
  const std::uint64_t half_bytes = payload_bytes / 2U;
  const std::uint64_t shard_bytes = half_bytes / kFusedPrefillRanks;
  const std::uint64_t tile_bytes =
      shard_bytes / kFusedPrefillTilesPerShard;
  const std::uint64_t half = direction == 1 ? 0 : half_bytes;
  return half +
         static_cast<std::uint64_t>(transfer.receive_shard) *
             shard_bytes +
         static_cast<std::uint64_t>(tile) * tile_bytes;
}

std::uint64_t initial_offset(std::uint32_t rank, std::int32_t direction,
                             std::uint32_t tile,
                             std::uint64_t payload_bytes) {
  const auto transfer = tp4_prefill_stage(rank, as_direction(direction), 0);
  const std::uint64_t half_bytes = payload_bytes / 2U;
  const std::uint64_t shard_bytes = half_bytes / kFusedPrefillRanks;
  const std::uint64_t tile_bytes =
      shard_bytes / kFusedPrefillTilesPerShard;
  const std::uint64_t half = direction == 1 ? 0 : half_bytes;
  return half + static_cast<std::uint64_t>(transfer.send_shard) *
                    shard_bytes +
         static_cast<std::uint64_t>(tile) * tile_bytes;
}

void pin_current_thread(std::int32_t cpu) {
  if (cpu < 0) return;
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<unsigned>(cpu), &set);
  const int result = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (result != 0) {
    throw std::runtime_error("failed to pin fused prefill proxy thread");
  }
#else
  static_cast<void>(cpu);
  throw std::runtime_error("proxy CPU pinning requires Linux");
#endif
}

void cpu_relax() noexcept {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace

FusedPrefillHostControl* FusedPrefillArenaView::host_control(
    std::uint32_t flow) const {
  return reinterpret_cast<FusedPrefillHostControl*>(
      host + fused_prefill_control_offset(flow));
}

FusedPrefillHostControl* FusedPrefillArenaView::device_control(
    std::uint32_t flow) const {
  return reinterpret_cast<FusedPrefillHostControl*>(
      device + fused_prefill_control_offset(flow));
}

std::uint8_t* FusedPrefillArenaView::device_plane(
    std::uint32_t flow, FusedPrefillPlane plane) const {
  return device + fused_prefill_plane_offset(flow, plane);
}

FusedPrefillArenaView make_fused_prefill_arena_view(
    MemoryBuffer& arena, std::uint32_t operation_slot) {
  const std::uint64_t offset =
      static_cast<std::uint64_t>(operation_slot) * kFusedPrefillArenaBytes;
  if ((arena.kind() != MemoryKind::kCudaMapped &&
       arena.kind() != MemoryKind::kCudaWriteCombined) ||
      arena.host_data() == nullptr || arena.device_data() == nullptr ||
      offset > arena.size() ||
      arena.size() - offset < kFusedPrefillArenaBytes ||
      reinterpret_cast<std::uintptr_t>(arena.host_data()) %
              kFusedPrefillPlaneAlignment !=
          0 ||
      reinterpret_cast<std::uintptr_t>(arena.device_data()) %
              kFusedPrefillPlaneAlignment !=
          0) {
    throw std::invalid_argument(
        "fused prefill requires one aligned CUDA-mapped global arena");
  }
  auto* registered_host = static_cast<std::uint8_t*>(arena.host_data());
  auto* registered_device = static_cast<std::uint8_t*>(arena.device_data());
  return {registered_host, registered_device,
          registered_host + offset, registered_device + offset,
          kFusedPrefillArenaBytes, arena.size(), offset};
}

FusedPrefillDescriptor make_fused_prefill_descriptor(
    const FusedPrefillArenaView& arena, FusedPrefillDeviceSync* device_sync,
    const void* input, void* output, std::uint32_t rank,
    std::int32_t direction, std::uint32_t tile,
    std::uint64_t operation_sequence, std::uint32_t spin_limit,
    std::uint64_t payload_bytes, std::uint32_t operation_slots) {
  if (arena.device == nullptr || arena.bytes < kFusedPrefillArenaBytes ||
      device_sync == nullptr || input == nullptr || output == nullptr ||
      rank >= kFusedPrefillRanks || spin_limit == 0 || payload_bytes == 0 ||
      payload_bytes > kFusedPrefillPayloadBytes ||
      payload_bytes % (kFusedPrefillElementsPerRow * 2U) != 0 ||
      operation_slots == 0 || operation_slots > 8) {
    throw std::invalid_argument("invalid fused prefill descriptor input");
  }
  const std::uint32_t flow = fused_prefill_flow(direction, tile);
  FusedPrefillDescriptor descriptor{};
  descriptor.input = static_cast<const std::uint8_t*>(input);
  descriptor.output = static_cast<std::uint8_t*>(output);
  descriptor.primary_incoming =
      arena.device_plane(flow, FusedPrefillPlane::kIncomingPrimary);
  descriptor.secondary_incoming =
      arena.device_plane(flow, FusedPrefillPlane::kIncomingSecondary);
  descriptor.primary_outgoing =
      arena.device_plane(flow, FusedPrefillPlane::kOutgoingPrimary);
  descriptor.secondary_outgoing =
      arena.device_plane(flow, FusedPrefillPlane::kOutgoingSecondary);
  descriptor.host_control = arena.device_control(flow);
  descriptor.device_sync = device_sync;
  descriptor.initial_tensor_offset_bytes =
      initial_offset(rank, direction, tile, payload_bytes);
  for (std::uint32_t stage = 0; stage < kFusedPrefillStages; ++stage) {
    descriptor.tensor_offset_bytes[stage] =
        tensor_offset(rank, direction, tile, stage, payload_bytes);
  }
  descriptor.operation_sequence = operation_sequence;
  descriptor.payload_bytes = payload_bytes;
  descriptor.operation_slots = operation_slots;
  descriptor.rank = rank;
  descriptor.direction = direction;
  descriptor.tile = tile;
  descriptor.spin_limit = spin_limit;
  return descriptor;
}

class FusedPrefillVerbsProxy::Impl {
 public:
  Impl(FusedPrefillArenaView arena, FusedPrefillVerbsProxyConfig config)
      : arena_(arena), config_(config) {
    if (arena_.host == nullptr || arena_.device == nullptr ||
        arena_.bytes < kFusedPrefillArenaBytes ||
        arena_.registered_host == nullptr ||
        arena_.registered_device == nullptr ||
        arena_.registered_bytes < kFusedPrefillArenaBytes ||
        config_.rank >= kFusedPrefillRanks ||
        config_.timeout_milliseconds == 0) {
      throw std::invalid_argument("invalid fused prefill proxy config");
    }
    for (std::uint32_t left = 0; left < config_.endpoints.size(); ++left) {
      for (std::uint32_t right = left + 1; right < config_.endpoints.size();
           ++right) {
        if (config_.endpoints[left] == config_.endpoints[right]) {
          throw std::invalid_argument(
              "fused prefill requires four distinct QPs");
        }
      }
    }
    for (std::uint32_t endpoint = 0; endpoint < config_.endpoints.size();
         ++endpoint) {
      if (config_.endpoints[endpoint] == nullptr) {
        throw std::invalid_argument("fused prefill endpoint is null");
      }
      const EndpointInfo local = config_.endpoints[endpoint]->local_info();
      if (config_.endpoints[endpoint]->maximum_send_work_requests() < 4U ||
          local.address !=
              reinterpret_cast<std::uint64_t>(arena_.registered_host) ||
          local.buffer_bytes < arena_.registered_bytes) {
        throw std::invalid_argument(
            "all four QPs must register the complete fused prefill arena");
      }
      maximum_sq_[endpoint] =
          config_.endpoints[endpoint]->maximum_send_work_requests();
    }
  }

  ~Impl() noexcept {
    if (total_outstanding_wqes() == 0) return;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    try {
      while (total_outstanding_wqes() != 0 &&
             std::chrono::steady_clock::now() < deadline) {
        poll_all_cqs();
        cpu_relax();
      }
    } catch (...) {
      std::terminate();
    }
    // Fail-stop rather than permit the owner to deregister mapped storage
    // while a QP can still DMA from it.
    if (total_outstanding_wqes() != 0) std::terminate();
  }

  FusedPrefillVerbsProxyReceipt run_operation(std::uint64_t sequence,
                                               std::uint64_t rail_bytes,
                                               std::uint32_t operation_slot) {
    if (poisoned_) throw std::runtime_error("fused prefill proxy poisoned");
    if (!have_sequence_ && sequence != 0) {
      fail("first fused prefill operation sequence must be zero");
    }
    if (have_sequence_ && sequence != last_sequence_ + 1U) {
      fail("operation sequence is not monotonic");
    }
    if (sequence > (UINT64_MAX - kFusedPrefillStages) /
                       kFusedPrefillStages) {
      fail("operation sequence exhausts token space");
    }
    if (rail_bytes == 0 || rail_bytes > kFusedPrefillRailBytes ||
        rail_bytes % 16U != 0) {
      fail("fused prefill rail bytes are invalid");
    }
    const std::uint64_t operation_offset =
        static_cast<std::uint64_t>(operation_slot) *
        kFusedPrefillArenaBytes;
    if (operation_offset > arena_.registered_bytes ||
        arena_.registered_bytes - operation_offset <
            kFusedPrefillArenaBytes) {
      fail("fused prefill operation slot is invalid");
    }
    arena_.host = arena_.registered_host + operation_offset;
    arena_.device = arena_.registered_device + operation_offset;
    arena_.registered_offset = operation_offset;
    active_rail_bytes_ = rail_bytes;
    pin_current_thread(config_.cpu);
    stages_ = {};
    receipt_ = {};
    receipt_.operation_sequence = sequence;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(
                              config_.timeout_milliseconds);

    while (!operation_complete()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        fail("fused prefill proxy operation timed out");
      }
      ++receipt_.spin_passes;
      poll_all_cqs();
      for (std::uint32_t flow = 0; flow < kFusedPrefillFlows; ++flow) {
        for (std::uint32_t stage = 0; stage < kFusedPrefillStages; ++stage) {
          advance(flow, stage, sequence);
        }
      }
      if (config_.cpu >= 0 || (receipt_.spin_passes & 0xfffU) != 0) {
        cpu_relax();
      } else {
        std::this_thread::yield();
      }
    }
    // Explicitly drain all four CQs before an owner may release the arena.
    while (total_outstanding_wqes() != 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        fail("timed out draining fused prefill QPs");
      }
      poll_all_cqs();
    }
    have_sequence_ = true;
    last_sequence_ = sequence;
    return receipt_;
  }

  bool poisoned() const noexcept { return poisoned_; }

  void poison_after_exception() noexcept {
    if (poisoned_) return;
    poisoned_ = true;
    const std::uint64_t token =
        receipt_.operation_sequence == 0
            ? 1U
            : receipt_.operation_sequence * kFusedPrefillStages + 1U;
    for (std::uint32_t flow = 0; flow < kFusedPrefillFlows; ++flow) {
      store_release(&arena_.host_control(flow)->poison_sequence, token);
    }
  }

 private:
  enum class CompletionKind : std::uint8_t {
    kExchangePrimary,
    kExchangeSecondary,
    kCredit,
  };

  struct StageState {
    bool exchange_posted{};
    std::array<bool, 2> exchange_cqe{};
    bool consumer_observed{};
    bool credit_posted{};
    bool credit_cqe{};
    bool reuse_published{};
  };

  struct PendingCompletion {
    std::uint64_t work_id{};
    std::uint32_t flow{};
    std::uint32_t stage{};
    std::uint32_t wqe_span{};
    CompletionKind kind{};
  };

  [[noreturn]] void fail(const char* message) {
    poison_after_exception();
    throw std::runtime_error(message);
  }

  std::uint64_t next_work_id(std::uint32_t endpoint) {
    if (next_work_id_[endpoint] == UINT64_MAX) {
      fail("fused prefill work ID exhausted");
    }
    return ++next_work_id_[endpoint];
  }

  std::uint64_t token(std::uint64_t sequence, std::uint32_t stage) const {
    return fused_prefill_stage_token(sequence, stage);
  }

  bool observe_exact(const std::uint64_t* address, std::uint64_t expected,
                     const char* future_message) {
    switch (classify_token(load_acquire(address), expected)) {
      case TokenState::kPast: return false;
      case TokenState::kExact: return true;
      case TokenState::kFuture: fail(future_message);
    }
    return false;
  }

  void require_sq(std::uint32_t endpoint, std::uint32_t needed) {
    if (needed > maximum_sq_[endpoint] - outstanding_wqes_[endpoint]) {
      fail("fused prefill SQ capacity exhausted");
    }
  }

  void post_exchange(std::uint32_t flow, std::uint32_t stage,
                     std::uint64_t sequence) {
    const std::int32_t direction = flow_direction(flow);
    const std::uint32_t parity = fused_prefill_parity(stage);
    const std::uint64_t expected = token(sequence, stage);
    // Transactional preflight prevents a primary half-post when secondary SQ
    // capacity is unavailable.
    const std::uint32_t primary_endpoint =
        fused_prefill_endpoint_index(direction, 0);
    const std::uint32_t secondary_endpoint =
        fused_prefill_endpoint_index(direction, 1);
    require_sq(primary_endpoint, 2);
    require_sq(secondary_endpoint, 2);

    for (std::uint32_t rail = 0; rail < kFusedPrefillRailCount; ++rail) {
      const std::uint32_t endpoint =
          fused_prefill_endpoint_index(direction, rail);
      const auto outgoing = fused_prefill_outgoing_plane(rail);
      const auto incoming = fused_prefill_incoming_plane(rail);
      const std::uint64_t local_payload =
          arena_.registered_offset +
          fused_prefill_slot_offset(flow, outgoing, parity);
      const std::uint64_t remote_payload =
          arena_.registered_offset +
          fused_prefill_slot_offset(flow, incoming, parity);
      const std::uint64_t payload_id = next_work_id(endpoint);
      config_.endpoints[endpoint]->write(
          local_payload, remote_payload, active_rail_bytes_,
          payload_id, false);
      const std::uint64_t local_doorbell = control_field_offset(
          flow, offsetof(FusedPrefillHostControl, producer), parity) +
          arena_.registered_offset;
      const std::size_t remote_field =
          rail == 0 ? offsetof(FusedPrefillHostControl, primary_doorbell)
                    : offsetof(FusedPrefillHostControl, secondary_doorbell);
      const std::uint64_t remote_doorbell =
          control_field_offset(flow, remote_field, parity) +
          arena_.registered_offset;
      const std::uint64_t doorbell_id = next_work_id(endpoint);
      config_.endpoints[endpoint]->write(local_doorbell, remote_doorbell,
                                         sizeof(expected), doorbell_id, true);
      outstanding_wqes_[endpoint] += 2U;
      pending_[endpoint].push_back(
          {doorbell_id, flow, stage, 2U,
           rail == 0 ? CompletionKind::kExchangePrimary
                     : CompletionKind::kExchangeSecondary});
      ++receipt_.payload_writes;
      ++receipt_.doorbell_writes;
      receipt_.payload_bytes[endpoint] += active_rail_bytes_;
      receipt_.doorbell_bytes[endpoint] += sizeof(expected);
    }
    stages_[flow][stage].exchange_posted = true;
  }

  bool try_post_credit(std::uint32_t flow, std::uint32_t stage,
                       std::uint64_t sequence) {
    const std::int32_t reverse_direction = -flow_direction(flow);
    const std::uint32_t endpoint =
        fused_prefill_endpoint_index(reverse_direction, 0);
    if (outstanding_wqes_[endpoint] == maximum_sq_[endpoint]) return false;
    require_sq(endpoint, 1);
    const std::uint32_t parity = fused_prefill_parity(stage);
    const std::uint64_t expected = token(sequence, stage);
    const std::uint64_t local_credit = control_field_offset(
        flow, offsetof(FusedPrefillHostControl, consumer), parity) +
        arena_.registered_offset;
    const std::uint64_t remote_credit = control_field_offset(
        flow, offsetof(FusedPrefillHostControl, peer_credit), parity) +
        arena_.registered_offset;
    const std::uint64_t credit_id = next_work_id(endpoint);
    config_.endpoints[endpoint]->write(local_credit, remote_credit,
                                       sizeof(expected), credit_id, true);
    ++outstanding_wqes_[endpoint];
    pending_[endpoint].push_back(
        {credit_id, flow, stage, 1U, CompletionKind::kCredit});
    stages_[flow][stage].credit_posted = true;
    ++receipt_.credit_writes;
    receipt_.credit_bytes[endpoint] += sizeof(expected);
    return true;
  }

  void advance(std::uint32_t flow, std::uint32_t stage,
               std::uint64_t sequence) {
    StageState& state = stages_[flow][stage];
    auto* control = arena_.host_control(flow);
    const std::uint32_t parity = fused_prefill_parity(stage);
    const std::uint64_t expected = token(sequence, stage);
    if (!state.exchange_posted &&
        observe_exact(&control->producer[parity], expected,
                      "future producer token")) {
      post_exchange(flow, stage, sequence);
    }
    if (!state.consumer_observed &&
        observe_exact(&control->consumer[parity], expected,
                      "future consumer token")) {
      state.consumer_observed = true;
    }
    if (state.consumer_observed && !state.credit_posted) {
      static_cast<void>(try_post_credit(flow, stage, sequence));
    }
    // credit_cqe also protects the mapped consumer word used as the credit
    // RDMA source on providers that decline the requested inline capacity.
    if (!state.reuse_published && state.exchange_cqe[0] &&
        state.exchange_cqe[1] && state.credit_cqe &&
        observe_exact(&control->peer_credit[parity], expected,
                      "future peer-credit token")) {
      store_release(&control->reuse[parity], expected);
      state.reuse_published = true;
    }
  }

  void poll_all_cqs() {
    std::array<SendCompletion, kCompletionBatch> completions{};
    for (std::uint32_t endpoint = 0; endpoint < config_.endpoints.size();
         ++endpoint) {
      const std::size_t count = config_.endpoints[endpoint]
                                    ->poll_send_completions(
                                        completions.data(), completions.size());
      if (count != 0) ++receipt_.cq_batches;
      for (std::size_t index = 0; index < count; ++index) {
        if (pending_[endpoint].empty() ||
            completions[index].work_id != pending_[endpoint].front().work_id) {
          fail("stale, unknown, or future fused prefill completion ID");
        }
        const PendingCompletion completion = pending_[endpoint].front();
        pending_[endpoint].pop_front();
        if (outstanding_wqes_[endpoint] < completion.wqe_span) {
          fail("fused prefill SQ accounting underflow");
        }
        outstanding_wqes_[endpoint] -= completion.wqe_span;
        StageState& state = stages_[completion.flow][completion.stage];
        switch (completion.kind) {
          case CompletionKind::kExchangePrimary:
            state.exchange_cqe[0] = true;
            break;
          case CompletionKind::kExchangeSecondary:
            state.exchange_cqe[1] = true;
            break;
          case CompletionKind::kCredit: state.credit_cqe = true; break;
        }
        ++receipt_.completions;
        ++receipt_.cq_completions[endpoint];
      }
    }
  }

  std::uint64_t total_outstanding_wqes() const {
    std::uint64_t result{};
    for (const auto outstanding : outstanding_wqes_) result += outstanding;
    return result;
  }

  bool operation_complete() const {
    for (const auto& flow : stages_) {
      for (const auto& stage : flow) {
        if (!stage.exchange_posted || !stage.exchange_cqe[0] ||
            !stage.exchange_cqe[1] || !stage.consumer_observed ||
            !stage.credit_posted || !stage.credit_cqe ||
            !stage.reuse_published) {
          return false;
        }
      }
    }
    return true;
  }

  FusedPrefillArenaView arena_{};
  FusedPrefillVerbsProxyConfig config_{};
  std::array<std::array<StageState, kFusedPrefillStages>,
             kFusedPrefillFlows>
      stages_{};
  std::array<std::deque<PendingCompletion>, kFusedPrefillEndpointCount>
      pending_{};
  std::array<std::uint64_t, kFusedPrefillEndpointCount> next_work_id_{};
  std::array<std::uint32_t, kFusedPrefillEndpointCount> outstanding_wqes_{};
  std::array<std::uint32_t, kFusedPrefillEndpointCount> maximum_sq_{};
  FusedPrefillVerbsProxyReceipt receipt_{};
  std::uint64_t active_rail_bytes_{kFusedPrefillRailBytes};
  std::uint64_t last_sequence_{};
  bool have_sequence_{};
  bool poisoned_{};
};

FusedPrefillVerbsProxy::FusedPrefillVerbsProxy(
    FusedPrefillArenaView arena, FusedPrefillVerbsProxyConfig config)
    : impl_(new Impl(arena, config)) {}

FusedPrefillVerbsProxy::~FusedPrefillVerbsProxy() { delete impl_; }

FusedPrefillVerbsProxyReceipt FusedPrefillVerbsProxy::run_operation(
    std::uint64_t operation_sequence, std::uint64_t rail_bytes,
    std::uint32_t operation_slot) {
  try {
    return impl_->run_operation(operation_sequence, rail_bytes,
                                operation_slot);
  } catch (...) {
    impl_->poison_after_exception();
    throw;
  }
}

bool FusedPrefillVerbsProxy::poisoned() const noexcept {
  return impl_->poisoned();
}

}  // namespace spark_transport::tiled_prefill_research
