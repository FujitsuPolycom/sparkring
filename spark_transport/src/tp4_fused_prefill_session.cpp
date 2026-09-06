#include "spark_transport/tp4_fused_prefill_session.hpp"

#include "../experiments/tiled_prefill/fused_prefill_kernels.cuh"
#include "../experiments/tiled_prefill/fused_prefill_verbs_proxy.hpp"
#include "spark_transport/control_channel.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace spark_transport {
namespace research = tiled_prefill_research;
namespace {

constexpr std::uint16_t kVersion = 9;
constexpr std::uint16_t kTag = 0xf804;
constexpr std::uint32_t kOperationSlots = 2;

std::uint64_t load_mapped_poison(const std::uint64_t* address) noexcept {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void check_cuda(cudaError_t value, const char* operation) {
  if (value != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(value));
}

ControlChannel open_channel(const Tp4RoundPlan& plan, const std::string& peer,
                            std::uint16_t port) {
  return plan.server ? ControlChannel::listen_and_accept(port)
                     : ControlChannel::connect(peer, port);
}

std::int32_t direction_for_plan(std::uint32_t rank, std::uint32_t link) {
  return static_cast<std::uint32_t>(tp4_prefill_outgoing_endpoint(
             rank, Tp4PrefillDirection::kClockwise)) == link
             ? 1
             : -1;
}

struct Wire {
  std::uint32_t magic{0x46555333};
  std::uint16_t version{1};
  std::uint16_t reserved{};
  std::uint32_t rank{};
  std::uint32_t peer{};
  std::int32_t direction{};
  std::uint32_t rail{};
  std::uint32_t world_size{4};
  std::uint32_t query_rows{8192};
  std::uint32_t elements_per_row{4096};
  std::uint32_t flows{8};
  std::uint32_t stages{6};
  std::uint32_t tiles_per_shard{4};
  std::uint32_t parity_slots{2};
  std::uint32_t operation_slots{kOperationSlots};
  std::uint32_t reserved2{};
  std::uint64_t arena_bytes{
      kOperationSlots * research::kFusedPrefillArenaBytes};
  std::uint64_t payload_bytes{research::kFusedPrefillPayloadBytes};
  std::uint64_t tile_bytes{research::kFusedPrefillTileBytes};
  std::uint64_t rail_bytes{research::kFusedPrefillRailBytes};
  std::uint64_t flow_stride{research::kFusedPrefillFlowStride};
};

void connect(ControlChannel& channel, VerbsEndpoint& endpoint,
             const Tp4RoundPlan& plan, std::uint32_t rank,
             std::int32_t direction, std::uint32_t rail) {
  if (endpoint.active_mtu_bytes() != 4096 ||
      endpoint.maximum_send_work_requests() < 3)
    throw std::runtime_error("fused prefill MTU/SQ preflight failed");
  auto local = endpoint.local_info();
  local.version = kVersion;
  local.reserved = kTag;
  const auto remote = channel.exchange(local);
  if (remote.version != kVersion || remote.reserved != kTag ||
      remote.buffer_bytes !=
          kOperationSlots * research::kFusedPrefillArenaBytes)
    throw std::runtime_error("fused prefill endpoint handshake mismatch");
  Wire wire{}; wire.rank=rank; wire.peer=plan.peer_rank;
  wire.direction=direction; wire.rail=rail;
  const auto peer = channel.exchange(wire);
  if (peer.magic != wire.magic || peer.version != wire.version ||
      peer.rank != plan.peer_rank || peer.peer != rank ||
      peer.direction != -direction || peer.rail != rail ||
      peer.world_size != wire.world_size || peer.query_rows != wire.query_rows ||
      peer.elements_per_row != wire.elements_per_row ||
      peer.flows != wire.flows || peer.stages != wire.stages ||
      peer.tiles_per_shard != wire.tiles_per_shard ||
      peer.parity_slots != wire.parity_slots ||
      peer.operation_slots != wire.operation_slots ||
      peer.arena_bytes != wire.arena_bytes ||
      peer.payload_bytes != wire.payload_bytes ||
      peer.tile_bytes != wire.tile_bytes || peer.rail_bytes != wire.rail_bytes ||
      peer.flow_stride != wire.flow_stride)
    throw std::runtime_error("fused prefill exact handshake mismatch");
  endpoint.connect(remote, kVersion);
}

class ProxyWorker {
 public:
  explicit ProxyWorker(research::FusedPrefillVerbsProxy& proxy)
      : proxy_(proxy), thread_([this] { loop(); }) {}
  ~ProxyWorker() {
    wait_idle();
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  void wait_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] {
      return queue_.empty() && !running_ && !slot_busy_[0] && !slot_busy_[1];
    });
  }
  void wait_slot_idle(std::uint32_t slot) {
    if (slot >= kOperationSlots)
      throw std::invalid_argument("fused proxy slot is invalid");
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this, slot] { return !slot_busy_[slot]; });
  }
  void enqueue(std::uint64_t sequence, std::uint64_t rail_bytes,
               std::uint32_t slot) {
    if (slot >= kOperationSlots)
      throw std::invalid_argument("fused proxy slot is invalid");
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_busy_[slot])
      throw std::logic_error("fused proxy slot is still busy");
    slot_busy_[slot] = true;
    submitted_sequence_.store(sequence, std::memory_order_release);
    queue_.push_back({sequence, rail_bytes, slot});
    cv_.notify_one();
  }
  Tp4FusedPrefillHealthStatus health_status() const noexcept {
    const bool running = thread_running_.load(std::memory_order_acquire);
    return {
        running,
        false,
        running,
        submitted_sequence_.load(std::memory_order_acquire),
        completed_sequence_.load(std::memory_order_acquire)};
  }
 private:
  struct Work {
    std::uint64_t sequence{};
    std::uint64_t rail_bytes{};
    std::uint32_t slot{};
  };
  void loop() noexcept {
    thread_running_.store(true, std::memory_order_release);
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
      if (stop_) {
        thread_running_.store(false, std::memory_order_release);
        return;
      }
      const Work work = queue_.front();
      queue_.pop_front();
      running_ = true;
      lock.unlock();
      try {
        (void)proxy_.run_operation(work.sequence, work.rail_bytes, work.slot);
      }
      catch (...) {
        std::fprintf(stderr, "SIRCL_FUSED_FATAL sequence=%llu stage=proxy\n",
                     static_cast<unsigned long long>(work.sequence));
        std::fflush(stderr);
        std::_Exit(70);
      }
      lock.lock();
      completed_sequence_.store(work.sequence, std::memory_order_release);
      running_ = false;
      slot_busy_[work.slot] = false;
      done_.notify_all();
    }
  }
  research::FusedPrefillVerbsProxy& proxy_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_, done_;
  std::deque<Work> queue_;
  std::array<bool, kOperationSlots> slot_busy_{};
  bool running_{}, stop_{};
  std::atomic<bool> thread_running_{false};
  std::atomic<std::uint64_t> submitted_sequence_{};
  std::atomic<std::uint64_t> completed_sequence_{};
};

}  // namespace

class Tp4FusedPrefillSession::Impl {
 public:
  explicit Impl(Tp4BidirectionalPrefillOptions options)
      : options_(std::move(options)) {
    if (options_.rail_count != 2 || options_.query_rows != 8192 ||
        options_.elements_per_row != 4096)
      throw std::invalid_argument("fused prefill requires Q8192 width4096 rail2");
    arena_buffer_ = MemoryBuffer::allocate(
        MemoryKind::kCudaMapped,
        kOperationSlots * research::kFusedPrefillArenaBytes);
    arena_buffer_->fill_from_cpu(0);
    for (std::uint32_t slot = 0; slot < kOperationSlots; ++slot) {
      arenas_[slot] = research::make_fused_prefill_arena_view(
          *arena_buffer_, slot);
    }
    const std::array<Tp4RoundPlan, 2> plans{
        make_tp4_round_plan(options_.rank, 0),
        make_tp4_round_plan(options_.rank, 1)};
    const std::array<std::string, 2> peers{options_.peer0, options_.peer1};
    const std::array<std::string, 2> peers2{options_.secondary_peer0,
                                            options_.secondary_peer1};
    const std::array<std::string, 2> devices{options_.device0, options_.device1};
    const std::array<std::string, 2> devices2{options_.secondary_device0,
                                              options_.secondary_device1};
    const std::array<std::uint8_t, 2> gids{options_.gid0, options_.gid1};
    const std::array<std::uint8_t, 2> gids2{options_.secondary_gid0,
                                            options_.secondary_gid1};
    const std::array<std::uint16_t, 2> ports{options_.control_port0,
                                             options_.control_port1};
    const std::array<std::uint16_t, 2> ports2{
        options_.secondary_control_port0, options_.secondary_control_port1};
    for (std::uint32_t link = 0; link < 2; ++link) {
      const auto direction = direction_for_plan(options_.rank, link);
      channels_[link] = std::make_unique<ControlChannel>(
          open_channel(plans[link], peers[link], ports[link]));
      endpoints_[link] = std::make_unique<VerbsEndpoint>(
          devices[link], 1, gids[link], *arena_buffer_);
      connect(*channels_[link], *endpoints_[link], plans[link], options_.rank,
              direction, 0);
      channels_[link + 2] = std::make_unique<ControlChannel>(
          open_channel(plans[link], peers2[link], ports2[link]));
      endpoints_[link + 2] = std::make_unique<VerbsEndpoint>(
          devices2[link], 1, gids2[link], *arena_buffer_);
      connect(*channels_[link + 2], *endpoints_[link + 2], plans[link],
              options_.rank, direction, 1);
    }
    const auto cw = static_cast<std::uint32_t>(tp4_prefill_outgoing_endpoint(
        options_.rank, Tp4PrefillDirection::kClockwise));
    const auto ccw = 1U - cw;
    proxy_ = std::make_unique<research::FusedPrefillVerbsProxy>(
        arenas_[0], research::FusedPrefillVerbsProxyConfig{
                    options_.rank, options_.fused_proxy_cpu,
                    options_.timeout_seconds * 1000U,
                    {endpoints_[cw].get(), endpoints_[cw + 2].get(),
                     endpoints_[ccw].get(), endpoints_[ccw + 2].get()}});
    worker_ = std::make_unique<ProxyWorker>(*proxy_);
    check_cuda(cudaMalloc(&device_sync_,
                          sizeof(*device_sync_) * 8 * kOperationSlots),
               "allocate fused sync");
    check_cuda(cudaMemset(device_sync_, 0,
                          sizeof(*device_sync_) * 8 * kOperationSlots),
               "initialize fused sync");
    check_cuda(cudaMalloc(&device_descriptors_,
                          sizeof(*device_descriptors_) * 8 * kOperationSlots),
               "allocate fused descriptors");
    for (auto& event : kernel_done_) {
      check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                 "create fused kernel completion event");
    }
    for (auto& channel : channels_) channel->barrier();
  }

  ~Impl() {
    worker_.reset();
    for (std::uint32_t slot = 0; slot < kOperationSlots; ++slot) {
      if (kernel_done_armed_[slot])
        (void)cudaEventSynchronize(kernel_done_[slot]);
      if (kernel_done_[slot]) (void)cudaEventDestroy(kernel_done_[slot]);
    }
    if (device_descriptors_) (void)cudaFree(device_descriptors_);
    if (device_sync_) (void)cudaFree(device_sync_);
    proxy_.reset();
    for (auto& endpoint : endpoints_) endpoint.reset();
    arena_buffer_.reset();
  }

  void all_reduce(const void* input, void* output, void* stream_pointer,
                  std::uint32_t query_rows) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    if (!input || !output) throw std::invalid_argument("fused tensor is null");
    if (query_rows == 0 || query_rows > research::kFusedPrefillQueryRows)
      throw std::invalid_argument("fused query rows exceed Q8192 capacity");
    const std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(query_rows) *
        research::kFusedPrefillElementsPerRow * 2U;
    const std::uint64_t rail_bytes =
        payload_bytes /
        (research::kFusedPrefillDirections * research::kFusedPrefillRanks *
         research::kFusedPrefillTilesPerShard * 2U);
    const std::uint32_t slot =
        static_cast<std::uint32_t>(sequence_ % kOperationSlots);
    worker_->wait_slot_idle(slot);
    if (kernel_done_armed_[slot]) {
      check_cuda(cudaEventSynchronize(kernel_done_[slot]),
                 "retire prior fused caller-stream kernel");
      kernel_done_armed_[slot] = false;
    }
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (sequence_ >= UINT32_MAX)
      throw std::overflow_error("fused prefill sequence exhausted");
    cudaStreamCaptureStatus capture{};
    check_cuda(cudaStreamIsCapturing(stream, &capture), "query fused capture");
    if (capture != cudaStreamCaptureStatusNone)
      throw std::logic_error("fused prefill rejects CUDA capture");
    for (std::uint32_t flow = 0; flow < 8; ++flow) {
      const std::int32_t direction = flow < 4 ? 1 : -1;
      host_descriptors_[slot][flow] =
          research::make_fused_prefill_descriptor(
              arenas_[slot], &device_sync_[slot * 8 + flow], input, output,
              options_.rank, direction, flow % 4, sequence_, UINT32_MAX,
              payload_bytes, kOperationSlots);
    }
    auto* const descriptors = device_descriptors_ + slot * 8;
    check_cuda(cudaMemcpyAsync(descriptors, host_descriptors_[slot].data(),
                               sizeof(host_descriptors_[slot]),
                               cudaMemcpyHostToDevice, stream),
               "upload fused descriptors");
    worker_->enqueue(sequence_, rail_bytes, slot);
    const auto launched = research::launch_fused_prefill_q8192_n4(
        descriptors, stream);
    if (launched != cudaSuccess) {
      std::fprintf(stderr, "SIRCL_FUSED_FATAL sequence=%llu stage=launch\n",
                   static_cast<unsigned long long>(sequence_));
      std::fflush(stderr);
      std::_Exit(70);
    }
    if (cudaEventRecord(kernel_done_[slot], stream) != cudaSuccess) {
      std::fprintf(stderr,
                   "SIRCL_FUSED_FATAL sequence=%llu stage=kernel_event\n",
                   static_cast<unsigned long long>(sequence_));
      std::fflush(stderr);
      std::_Exit(70);
    }
    kernel_done_armed_[slot] = true;
    ++sequence_;
  }

  Tp4FusedPrefillHealthStatus health_status() const noexcept {
    auto status =
        worker_ ? worker_->health_status() : Tp4FusedPrefillHealthStatus{};
    std::uint64_t poison_token{};
    for (const auto& arena : arenas_) {
      for (std::uint32_t flow = 0; flow < research::kFusedPrefillFlows;
           ++flow) {
        const std::uint64_t observed =
            load_mapped_poison(&arena.host_control(flow)->poison_sequence);
        if (observed != 0 &&
            (poison_token == 0 || observed < poison_token)) {
          poison_token = observed;
        }
      }
    }
    if (poison_token != 0) {
      status.healthy = false;
      status.poisoned = true;
      status.failing_sequence =
          (poison_token - 1U) / research::kFusedPrefillStages;
      status.failing_stage = static_cast<std::int32_t>(
          (poison_token - 1U) % research::kFusedPrefillStages);
      status.error_code = 1;
    }
    return status;
  }

 private:
  Tp4BidirectionalPrefillOptions options_;
  std::unique_ptr<MemoryBuffer> arena_buffer_;
  std::array<research::FusedPrefillArenaView, kOperationSlots> arenas_{};
  std::array<std::unique_ptr<ControlChannel>, 4> channels_;
  std::array<std::unique_ptr<VerbsEndpoint>, 4> endpoints_;
  std::unique_ptr<research::FusedPrefillVerbsProxy> proxy_;
  std::unique_ptr<ProxyWorker> worker_;
  research::FusedPrefillDeviceSync* device_sync_{};
  research::FusedPrefillDescriptor* device_descriptors_{};
  std::array<std::array<research::FusedPrefillDescriptor, 8>,
             kOperationSlots>
      host_descriptors_{};
  std::mutex submit_mutex_;
  std::uint64_t sequence_{};
  std::array<cudaEvent_t, kOperationSlots> kernel_done_{};
  std::array<bool, kOperationSlots> kernel_done_armed_{};
};

Tp4FusedPrefillSession::Tp4FusedPrefillSession(
    Tp4BidirectionalPrefillOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
Tp4FusedPrefillSession::~Tp4FusedPrefillSession() = default;
void Tp4FusedPrefillSession::all_reduce_fused(
    const void* input, void* output, void* cuda_stream,
    std::uint32_t query_rows) {
  impl_->all_reduce(input, output, cuda_stream, query_rows);
}

Tp4FusedPrefillHealthStatus Tp4FusedPrefillSession::health_status()
    const noexcept {
  return impl_->health_status();
}

}  // namespace spark_transport
