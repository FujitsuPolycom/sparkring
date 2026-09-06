// Research-only four-node Q8192 fused-ring verbs qualification probe.
// This executable is deliberately absent from the serving session and C ABI.

#include "fused_prefill_kernels.cuh"
#include "fused_prefill_verbs_proxy.hpp"

#include "spark_transport/control_channel.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/statistics.hpp"
#include "spark_transport/tp4_bidirectional_prefill.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;

namespace {

constexpr std::uint16_t kEndpointVersion = 8;
constexpr std::uint16_t kEndpointTag = 0x4633;  // "F3"
constexpr std::uint32_t kGeometryMagic = 0x46335247;  // "F3RG"
constexpr std::uint16_t kGeometryVersion = 1;
constexpr std::uint64_t kGuardBytes = 4096;
constexpr std::uint8_t kInputGuard = 0xa5;
constexpr std::uint8_t kOutputGuard = 0x5a;
constexpr std::uint32_t kRequiredMtu = 4096;

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string secondary_peer0;
  std::string secondary_peer1;
  std::string primary_device0{"rocep1s0f0"};
  std::string primary_device1{"rocep1s0f1"};
  std::string secondary_device0;
  std::string secondary_device1;
  std::uint8_t primary_gid0{3};
  std::uint8_t primary_gid1{3};
  std::uint8_t secondary_gid0{3};
  std::uint8_t secondary_gid1{3};
  std::uint16_t primary_port0{19300};
  std::uint16_t primary_port1{19301};
  std::uint16_t secondary_port0{19302};
  std::uint16_t secondary_port1{19303};
  std::int32_t proxy_cpu{-1};
  std::uint32_t warmup{5};
  std::uint32_t iterations{20};
  std::uint32_t timeout_seconds{120};
  std::uint32_t spin_limit{std::numeric_limits<std::uint32_t>::max()};
};

struct GeometryHandshake {
  std::uint32_t magic{kGeometryMagic};
  std::uint16_t version{kGeometryVersion};
  std::uint16_t reserved{};
  std::uint32_t rank{};
  std::uint32_t peer_rank{};
  std::int32_t outgoing_direction{};
  std::uint32_t rail{};
  std::uint32_t world_size{research::kFusedPrefillRanks};
  std::uint32_t query_rows{research::kFusedPrefillQueryRows};
  std::uint32_t elements_per_row{research::kFusedPrefillElementsPerRow};
  std::uint32_t flows{research::kFusedPrefillFlows};
  std::uint32_t stages{research::kFusedPrefillStages};
  std::uint32_t tiles_per_shard{research::kFusedPrefillTilesPerShard};
  std::uint32_t parity_slots{research::kFusedPrefillParitySlots};
  std::uint32_t active_mtu{kRequiredMtu};
  std::uint64_t payload_bytes{research::kFusedPrefillPayloadBytes};
  std::uint64_t tile_bytes{research::kFusedPrefillTileBytes};
  std::uint64_t rail_bytes{research::kFusedPrefillRailBytes};
  std::uint64_t flow_stride{research::kFusedPrefillFlowStride};
  std::uint64_t arena_bytes{research::kFusedPrefillArenaBytes};
};

static_assert(std::is_trivially_copyable_v<GeometryHandshake>);

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank R --peer0 IP --peer1 IP"
         " [--secondary-peer0 IP] [--secondary-peer1 IP]"
         " --primary-device0 HCA --primary-device1 HCA"
         " --secondary-device0 HCA --secondary-device1 HCA"
         " [--primary-gid0 N] [--primary-gid1 N]"
         " [--secondary-gid0 N] [--secondary-gid1 N]"
         " [--primary-port0 N] [--primary-port1 N]"
         " [--secondary-port0 N] [--secondary-port1 N]"
         " [--proxy-cpu N] [--warmup N] [--iterations N]"
         " [--timeout-seconds N] [--spin-limit N]\n";
  std::exit(2);
}

std::uint64_t unsigned_value(const char* value, const char* name) {
  std::size_t consumed{};
  const std::string text(value);
  const auto parsed = std::stoull(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto take = [&]() -> const char* {
      if (++index >= argc) usage(argv[0]);
      return argv[index];
    };
    if (argument == "--rank") {
      options.rank = static_cast<std::uint32_t>(unsigned_value(take(), "rank"));
    } else if (argument == "--peer0") {
      options.peer0 = take();
    } else if (argument == "--peer1") {
      options.peer1 = take();
    } else if (argument == "--secondary-peer0") {
      options.secondary_peer0 = take();
    } else if (argument == "--secondary-peer1") {
      options.secondary_peer1 = take();
    } else if (argument == "--primary-device0") {
      options.primary_device0 = take();
    } else if (argument == "--primary-device1") {
      options.primary_device1 = take();
    } else if (argument == "--secondary-device0") {
      options.secondary_device0 = take();
    } else if (argument == "--secondary-device1") {
      options.secondary_device1 = take();
    } else if (argument == "--primary-gid0") {
      options.primary_gid0 = static_cast<std::uint8_t>(
          unsigned_value(take(), "primary gid0"));
    } else if (argument == "--primary-gid1") {
      options.primary_gid1 = static_cast<std::uint8_t>(
          unsigned_value(take(), "primary gid1"));
    } else if (argument == "--secondary-gid0") {
      options.secondary_gid0 = static_cast<std::uint8_t>(
          unsigned_value(take(), "secondary gid0"));
    } else if (argument == "--secondary-gid1") {
      options.secondary_gid1 = static_cast<std::uint8_t>(
          unsigned_value(take(), "secondary gid1"));
    } else if (argument == "--primary-port0") {
      options.primary_port0 = static_cast<std::uint16_t>(
          unsigned_value(take(), "primary port0"));
    } else if (argument == "--primary-port1") {
      options.primary_port1 = static_cast<std::uint16_t>(
          unsigned_value(take(), "primary port1"));
    } else if (argument == "--secondary-port0") {
      options.secondary_port0 = static_cast<std::uint16_t>(
          unsigned_value(take(), "secondary port0"));
    } else if (argument == "--secondary-port1") {
      options.secondary_port1 = static_cast<std::uint16_t>(
          unsigned_value(take(), "secondary port1"));
    } else if (argument == "--proxy-cpu") {
      options.proxy_cpu = static_cast<std::int32_t>(
          unsigned_value(take(), "proxy cpu"));
    } else if (argument == "--warmup") {
      options.warmup = static_cast<std::uint32_t>(unsigned_value(take(), "warmup"));
    } else if (argument == "--iterations") {
      options.iterations = static_cast<std::uint32_t>(
          unsigned_value(take(), "iterations"));
    } else if (argument == "--timeout-seconds") {
      options.timeout_seconds = static_cast<std::uint32_t>(
          unsigned_value(take(), "timeout seconds"));
    } else if (argument == "--spin-limit") {
      options.spin_limit = static_cast<std::uint32_t>(
          unsigned_value(take(), "spin limit"));
    } else {
      usage(argv[0]);
    }
  }
  const std::array<std::uint16_t, 4> ports{
      options.primary_port0, options.primary_port1,
      options.secondary_port0, options.secondary_port1};
  if (options.rank >= research::kFusedPrefillRanks ||
      options.peer0.empty() || options.peer1.empty() ||
      options.primary_device0.empty() || options.primary_device1.empty() ||
      options.secondary_device0.empty() || options.secondary_device1.empty() ||
      std::any_of(ports.begin(), ports.end(), [](auto port) { return port == 0; }) ||
      options.primary_port0 == options.primary_port1 ||
      options.primary_port0 == options.secondary_port0 ||
      options.primary_port0 == options.secondary_port1 ||
      options.primary_port1 == options.secondary_port0 ||
      options.primary_port1 == options.secondary_port1 ||
      options.secondary_port0 == options.secondary_port1 ||
      options.iterations == 0 || options.timeout_seconds == 0 ||
      options.spin_limit == 0) {
    usage(argv[0]);
  }
  if (options.secondary_peer0.empty()) options.secondary_peer0 = options.peer0;
  if (options.secondary_peer1.empty()) options.secondary_peer1 = options.peer1;
  if (options.primary_device0 == options.secondary_device0 ||
      options.primary_device1 == options.secondary_device1) {
    throw std::invalid_argument(
        "primary and secondary rails on each link require distinct HCAs");
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result{};
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t bf16_add(std::uint16_t left, std::uint16_t right) {
  return float_to_bf16(bf16_to_float(left) + bf16_to_float(right));
}

std::uint16_t input_value(bool noninteger, std::uint32_t rank,
                          std::size_t element) {
  if (!noninteger) return float_to_bf16(static_cast<float>(rank + 1U));
  const std::int32_t centered = static_cast<std::int32_t>(
      (element * 37U + static_cast<std::size_t>(rank) * 53U) % 257U) - 128;
  const float scale = 0.0078125F * static_cast<float>(rank + 1U);
  const float offset =
      static_cast<float>((element + 3U * rank) % 11U) * 0.001953125F;
  return float_to_bf16(static_cast<float>(centered) * scale + offset);
}

std::uint16_t expected_value(bool noninteger, std::size_t element) {
  const bool clockwise =
      element * sizeof(std::uint16_t) < research::kFusedPrefillHalfBytes;
  const std::int32_t direction = clockwise ? 1 : -1;
  const std::size_t half_element =
      clockwise ? element
                : element - research::kFusedPrefillHalfBytes /
                                sizeof(std::uint16_t);
  const std::uint32_t shard = static_cast<std::uint32_t>(
      (half_element * sizeof(std::uint16_t)) /
      research::kFusedPrefillShardBytes);
  std::uint16_t result = input_value(noninteger, shard, element);
  for (std::uint32_t contributor = 1;
       contributor < research::kFusedPrefillRanks; ++contributor) {
    const auto rank = spark_transport::tp4_prefill_wrap_rank(
        static_cast<std::int32_t>(shard) +
        direction * static_cast<std::int32_t>(contributor));
    result = bf16_add(result, input_value(noninteger, rank, element));
  }
  return result;
}

spark_transport::ControlChannel open_channel(
    const spark_transport::Tp4RoundPlan& plan, const std::string& peer,
    std::uint16_t port) {
  return plan.server
             ? spark_transport::ControlChannel::listen_and_accept(port)
             : spark_transport::ControlChannel::connect(peer, port);
}

std::int32_t direction_for_plan(std::uint32_t rank, std::uint32_t plan_index) {
  const auto clockwise = spark_transport::tp4_prefill_outgoing_endpoint(
      rank, spark_transport::Tp4PrefillDirection::kClockwise);
  return static_cast<std::uint32_t>(clockwise) == plan_index ? 1 : -1;
}

void connect_endpoint(spark_transport::ControlChannel& channel,
                      spark_transport::VerbsEndpoint& endpoint,
                      const spark_transport::Tp4RoundPlan& plan,
                      std::uint32_t rank, std::int32_t direction,
                      std::uint32_t rail) {
  if (endpoint.active_mtu_bytes() != kRequiredMtu) {
    throw std::runtime_error("fused prefill requires active MTU 4096 on every rail");
  }
  auto local = endpoint.local_info();
  local.version = kEndpointVersion;
  local.reserved = kEndpointTag;
  const auto remote = channel.exchange(local);
  if (remote.version != local.version || remote.reserved != local.reserved ||
      remote.buffer_bytes != research::kFusedPrefillArenaBytes) {
    throw std::runtime_error("fused prefill endpoint handshake mismatch");
  }
  const GeometryHandshake geometry{kGeometryMagic,
                                   kGeometryVersion,
                                   0,
                                   rank,
                                   plan.peer_rank,
                                   direction,
                                   rail};
  const auto peer = channel.exchange(geometry);
  if (peer.magic != geometry.magic || peer.version != geometry.version ||
      peer.reserved != 0 || peer.rank != plan.peer_rank ||
      peer.peer_rank != rank || peer.outgoing_direction != -direction ||
      peer.rail != rail || peer.world_size != geometry.world_size ||
      peer.query_rows != geometry.query_rows ||
      peer.elements_per_row != geometry.elements_per_row ||
      peer.flows != geometry.flows || peer.stages != geometry.stages ||
      peer.tiles_per_shard != geometry.tiles_per_shard ||
      peer.parity_slots != geometry.parity_slots ||
      peer.active_mtu != kRequiredMtu ||
      peer.payload_bytes != geometry.payload_bytes ||
      peer.tile_bytes != geometry.tile_bytes ||
      peer.rail_bytes != geometry.rail_bytes ||
      peer.flow_stride != geometry.flow_stride ||
      peer.arena_bytes != geometry.arena_bytes) {
    throw std::runtime_error("fused prefill exact geometry handshake mismatch");
  }
  endpoint.connect(remote, local.version);
}

struct Validation {
  std::uint64_t output_mismatches{};
  std::uint64_t input_mismatches{};
  std::uint64_t input_guard_corruptions{};
  std::uint64_t output_guard_corruptions{};
  double max_abs{};
};

Validation validate(std::uint8_t* guarded_input, std::uint8_t* guarded_output,
                    bool noninteger, std::uint32_t rank) {
  const std::uint64_t total =
      research::kFusedPrefillPayloadBytes + 2U * kGuardBytes;
  std::vector<std::uint8_t> input(total);
  std::vector<std::uint8_t> output(total);
  check_cuda(cudaMemcpy(input.data(), guarded_input, total,
                        cudaMemcpyDeviceToHost),
             "copy guarded input");
  check_cuda(cudaMemcpy(output.data(), guarded_output, total,
                        cudaMemcpyDeviceToHost),
             "copy guarded output");
  Validation result{};
  for (std::uint64_t index = 0; index < kGuardBytes; ++index) {
    result.input_guard_corruptions += input[index] != kInputGuard;
    result.output_guard_corruptions += output[index] != kOutputGuard;
    result.input_guard_corruptions +=
        input[kGuardBytes + research::kFusedPrefillPayloadBytes + index] !=
        kInputGuard;
    result.output_guard_corruptions +=
        output[kGuardBytes + research::kFusedPrefillPayloadBytes + index] !=
        kOutputGuard;
  }
  const auto* input_words = reinterpret_cast<const std::uint16_t*>(
      input.data() + kGuardBytes);
  const auto* output_words = reinterpret_cast<const std::uint16_t*>(
      output.data() + kGuardBytes);
  const std::size_t words =
      research::kFusedPrefillPayloadBytes / sizeof(std::uint16_t);
  for (std::size_t element = 0; element < words; ++element) {
    const std::uint16_t expected_input = input_value(noninteger, rank, element);
    const std::uint16_t expected_output = expected_value(noninteger, element);
    result.input_mismatches += input_words[element] != expected_input;
    result.output_mismatches += output_words[element] != expected_output;
    result.max_abs = std::max(
        result.max_abs,
        static_cast<double>(std::abs(bf16_to_float(output_words[element]) -
                                     bf16_to_float(expected_output))));
  }
  return result;
}

class ProxyWorker {
 public:
  explicit ProxyWorker(research::FusedPrefillVerbsProxy& proxy)
      : proxy_(proxy), thread_([this] { loop(); }) {}
  ProxyWorker(const ProxyWorker&) = delete;
  ProxyWorker& operator=(const ProxyWorker&) = delete;
  ~ProxyWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    request_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  void start(std::uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_ || running_) throw std::logic_error("proxy worker already busy");
    sequence_ = sequence;
    pending_ = true;
    error_ = nullptr;
    request_.notify_one();
  }

  research::FusedPrefillVerbsProxyReceipt wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [this] { return !pending_ && !running_; });
    if (error_ != nullptr) std::rethrow_exception(error_);
    return receipt_;
  }

 private:
  void loop() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      request_.wait(lock, [this] { return stop_ || pending_; });
      if (stop_) return;
      const std::uint64_t sequence = sequence_;
      pending_ = false;
      running_ = true;
      lock.unlock();
      research::FusedPrefillVerbsProxyReceipt receipt{};
      std::exception_ptr error;
      try {
        receipt = proxy_.run_operation(sequence);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      receipt_ = receipt;
      error_ = error;
      running_ = false;
      completed_.notify_one();
    }
  }

  research::FusedPrefillVerbsProxy& proxy_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable request_;
  std::condition_variable completed_;
  std::uint64_t sequence_{};
  research::FusedPrefillVerbsProxyReceipt receipt_{};
  std::exception_ptr error_;
  bool pending_{};
  bool running_{};
  bool stop_{};
};

struct CaseReceipt {
  std::vector<double> enqueue_us;
  std::vector<double> device_us;
  std::vector<double> full_proxy_retirement_us;
  std::uint64_t payload_writes{};
  std::uint64_t doorbell_writes{};
  std::uint64_t credit_writes{};
  std::uint64_t completions{};
  std::uint64_t cq_batches{};
  std::uint64_t spin_passes{};
  std::array<std::uint64_t, research::kFusedPrefillEndpointCount>
      endpoint_payload_bytes{};
  std::array<std::uint64_t, research::kFusedPrefillEndpointCount>
      endpoint_doorbell_bytes{};
  std::array<std::uint64_t, research::kFusedPrefillEndpointCount>
      endpoint_credit_bytes{};
  std::array<std::uint64_t, research::kFusedPrefillEndpointCount>
      endpoint_cq_completions{};
  Validation validation{};
};

CaseReceipt run_case(bool noninteger, const Options& options,
                     const research::FusedPrefillArenaView& arena,
                     ProxyWorker& proxy_worker, std::uint8_t* guarded_input,
                     std::uint8_t* guarded_output,
                     research::FusedPrefillDeviceSync* device_sync,
                     research::FusedPrefillDescriptor* device_descriptors,
                     cudaStream_t stream, cudaEvent_t device_start,
                     cudaEvent_t device_stop, std::uint64_t& sequence) {
  const std::size_t words =
      research::kFusedPrefillPayloadBytes / sizeof(std::uint16_t);
  std::vector<std::uint16_t> host_input(words);
  for (std::size_t element = 0; element < words; ++element) {
    host_input[element] = input_value(noninteger, options.rank, element);
  }
  auto* input = guarded_input + kGuardBytes;
  auto* output = guarded_output + kGuardBytes;
  check_cuda(cudaMemcpy(input, host_input.data(),
                        research::kFusedPrefillPayloadBytes,
                        cudaMemcpyHostToDevice),
             "upload BF16 input");

  CaseReceipt result;
  result.enqueue_us.reserve(options.iterations);
  result.device_us.reserve(options.iterations);
  result.full_proxy_retirement_us.reserve(options.iterations);
  const std::uint64_t total =
      static_cast<std::uint64_t>(options.warmup) + options.iterations;
  for (std::uint64_t operation = 0; operation < total; ++operation, ++sequence) {
    check_cuda(cudaMemsetAsync(output, kOutputGuard,
                               research::kFusedPrefillPayloadBytes, stream),
               "poison output payload");
    std::array<research::FusedPrefillDescriptor,
               research::kFusedPrefillFlows>
        descriptors{};
    for (std::uint32_t flow = 0; flow < research::kFusedPrefillFlows; ++flow) {
      const std::int32_t direction =
          flow < research::kFusedPrefillTilesPerShard ? 1 : -1;
      descriptors[flow] = research::make_fused_prefill_descriptor(
          arena, &device_sync[flow], input, output, options.rank, direction,
          flow % research::kFusedPrefillTilesPerShard, sequence,
          options.spin_limit);
    }
    check_cuda(cudaMemcpyAsync(device_descriptors, descriptors.data(),
                               sizeof(descriptors), cudaMemcpyHostToDevice,
                               stream),
               "upload fused descriptors");
    const double full_start = now_us();
    proxy_worker.start(sequence);
    check_cuda(cudaEventRecord(device_start, stream), "record device start");
    const double enqueue_start = now_us();
    check_cuda(research::launch_fused_prefill_q8192_n4(device_descriptors,
                                                        stream),
               "launch fused prefill kernel");
    check_cuda(cudaEventRecord(device_stop, stream), "record device stop");
    const double enqueue_stop = now_us();
    check_cuda(cudaEventSynchronize(device_stop), "wait fused device completion");
    float device_milliseconds{};
    check_cuda(cudaEventElapsedTime(&device_milliseconds, device_start,
                                    device_stop),
               "measure fused device duration");
    const auto proxy_receipt = proxy_worker.wait();
    const double full_stop = now_us();
    if (operation >= options.warmup) {
      result.enqueue_us.push_back(enqueue_stop - enqueue_start);
      result.device_us.push_back(
          static_cast<double>(device_milliseconds) * 1000.0);
      result.full_proxy_retirement_us.push_back(full_stop - full_start);
      result.payload_writes += proxy_receipt.payload_writes;
      result.doorbell_writes += proxy_receipt.doorbell_writes;
      result.credit_writes += proxy_receipt.credit_writes;
      result.completions += proxy_receipt.completions;
      result.cq_batches += proxy_receipt.cq_batches;
      result.spin_passes += proxy_receipt.spin_passes;
      for (std::uint32_t endpoint = 0;
           endpoint < research::kFusedPrefillEndpointCount; ++endpoint) {
        result.endpoint_payload_bytes[endpoint] +=
            proxy_receipt.payload_bytes[endpoint];
        result.endpoint_doorbell_bytes[endpoint] +=
            proxy_receipt.doorbell_bytes[endpoint];
        result.endpoint_credit_bytes[endpoint] +=
            proxy_receipt.credit_bytes[endpoint];
        result.endpoint_cq_completions[endpoint] +=
            proxy_receipt.cq_completions[endpoint];
      }
    }
  }
  check_cuda(cudaStreamSynchronize(stream), "finish fused case");
  result.validation =
      validate(guarded_input, guarded_output, noninteger, options.rank);
  if (result.validation.output_mismatches != 0 ||
      result.validation.input_mismatches != 0 ||
      result.validation.input_guard_corruptions != 0 ||
      result.validation.output_guard_corruptions != 0) {
    throw std::runtime_error("fused prefill correctness or guard gate failed");
  }
  const std::uint64_t endpoint_payload =
      static_cast<std::uint64_t>(options.iterations) *
      research::kFusedPrefillTilesPerShard *
      research::kFusedPrefillStages * research::kFusedPrefillRailBytes;
  const std::uint64_t endpoint_doorbell =
      static_cast<std::uint64_t>(options.iterations) *
      research::kFusedPrefillTilesPerShard *
      research::kFusedPrefillStages * sizeof(std::uint64_t);
  const std::uint64_t endpoint_credit = endpoint_doorbell;
  const std::uint64_t exchange_cqes =
      static_cast<std::uint64_t>(options.iterations) *
      research::kFusedPrefillTilesPerShard * research::kFusedPrefillStages;
  for (std::uint32_t endpoint = 0;
       endpoint < research::kFusedPrefillEndpointCount; ++endpoint) {
    const bool primary = endpoint % research::kFusedPrefillRailCount == 0;
    if (result.endpoint_payload_bytes[endpoint] != endpoint_payload ||
        result.endpoint_doorbell_bytes[endpoint] != endpoint_doorbell ||
        result.endpoint_credit_bytes[endpoint] !=
            (primary ? endpoint_credit : 0U) ||
        result.endpoint_cq_completions[endpoint] !=
            exchange_cqes * (primary ? 2U : 1U)) {
      throw std::runtime_error("fused prefill per-endpoint accounting mismatch");
    }
  }
  return result;
}

void print_receipt(bool noninteger, const Options& options,
                   const CaseReceipt& receipt, int multiprocessors) {
  const auto enqueue = spark_transport::summarize_latencies(receipt.enqueue_us);
  const auto device = spark_transport::summarize_latencies(receipt.device_us);
  const auto full =
      spark_transport::summarize_latencies(receipt.full_proxy_retirement_us);
  std::cout << std::setprecision(12)
            << "TP4_FUSED_PREFILL_RECEIPT {"
            << "\"schema\":\"sparkring-tp4-fused-prefill/v1\""
            << ",\"rank\":" << options.rank
            << ",\"case\":\"" << (noninteger ? "noninteger" : "exact")
            << "\",\"query_rows\":" << research::kFusedPrefillQueryRows
            << ",\"elements_per_row\":"
            << research::kFusedPrefillElementsPerRow
            << ",\"payload_bytes\":" << research::kFusedPrefillPayloadBytes
            << ",\"arena_bytes\":" << research::kFusedPrefillArenaBytes
            << ",\"active_mtu\":" << kRequiredMtu
            << ",\"rails\":2,\"flows\":" << research::kFusedPrefillFlows
            << ",\"cooperative_ctas\":"
            << research::kFusedPrefillFlows *
                   research::kFusedPrefillCtasPerFlow
            << ",\"multiprocessors\":" << multiprocessors
            << ",\"warmup_operations\":" << options.warmup
            << ",\"measured_operations\":" << options.iterations
            << ",\"passed\":true"
            << ",\"output_mismatches\":"
            << receipt.validation.output_mismatches
            << ",\"input_mismatches\":" << receipt.validation.input_mismatches
            << ",\"input_guard_corruptions\":"
            << receipt.validation.input_guard_corruptions
            << ",\"output_guard_corruptions\":"
            << receipt.validation.output_guard_corruptions
            << ",\"max_abs\":" << receipt.validation.max_abs
            << ",\"enqueue_us_p50\":" << enqueue.p50_us
            << ",\"enqueue_us_p95\":" << enqueue.p95_us
            << ",\"device_us_p50\":" << device.p50_us
            << ",\"device_us_p95\":" << device.p95_us
            << ",\"full_proxy_retirement_us_p50\":" << full.p50_us
            << ",\"full_proxy_retirement_us_p95\":" << full.p95_us
            << ",\"payload_writes\":" << receipt.payload_writes
            << ",\"payload_wire_bytes\":"
            << receipt.payload_writes * research::kFusedPrefillRailBytes
            << ",\"doorbell_writes\":" << receipt.doorbell_writes
            << ",\"doorbell_wire_bytes\":"
            << receipt.doorbell_writes * sizeof(std::uint64_t)
            << ",\"credit_writes\":" << receipt.credit_writes
            << ",\"credit_wire_bytes\":"
            << receipt.credit_writes * sizeof(std::uint64_t)
            << ",\"cq_completions\":" << receipt.completions
            << ",\"cq_batches\":" << receipt.cq_batches
            << ",\"proxy_spin_passes\":" << receipt.spin_passes;
  constexpr std::array<const char*, research::kFusedPrefillEndpointCount>
      endpoint_names{"clockwise_primary", "clockwise_secondary",
                     "counterclockwise_primary",
                     "counterclockwise_secondary"};
  for (std::uint32_t endpoint = 0;
       endpoint < research::kFusedPrefillEndpointCount; ++endpoint) {
    std::cout << ",\"" << endpoint_names[endpoint]
              << "_payload_bytes\":"
              << receipt.endpoint_payload_bytes[endpoint]
              << ",\"" << endpoint_names[endpoint]
              << "_doorbell_bytes\":"
              << receipt.endpoint_doorbell_bytes[endpoint]
              << ",\"" << endpoint_names[endpoint]
              << "_credit_bytes\":"
              << receipt.endpoint_credit_bytes[endpoint]
              << ",\"" << endpoint_names[endpoint]
              << "_cq_completions\":"
              << receipt.endpoint_cq_completions[endpoint];
  }
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    check_cuda(cudaSetDevice(0), "select CUDA device");
    int cooperative_launch{};
    int multiprocessors{};
    check_cuda(cudaDeviceGetAttribute(&cooperative_launch,
                                      cudaDevAttrCooperativeLaunch, 0),
               "query cooperative launch support");
    check_cuda(cudaDeviceGetAttribute(&multiprocessors,
                                      cudaDevAttrMultiProcessorCount, 0),
               "query multiprocessor count");
    if (cooperative_launch == 0) {
      throw std::runtime_error("CUDA device does not support cooperative launch");
    }

    auto arena_buffer = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped,
        research::kFusedPrefillArenaBytes);
    arena_buffer->fill_from_cpu(0);
    const auto arena = research::make_fused_prefill_arena_view(*arena_buffer);
    const auto plan0 = spark_transport::make_tp4_round_plan(options.rank, 0);
    const auto plan1 = spark_transport::make_tp4_round_plan(options.rank, 1);
    std::array<spark_transport::Tp4RoundPlan, 2> plans{plan0, plan1};
    const std::array<std::string, 2> peers{options.peer0, options.peer1};
    const std::array<std::string, 2> secondary_peers{
        options.secondary_peer0, options.secondary_peer1};
    const std::array<std::string, 2> primary_devices{
        options.primary_device0, options.primary_device1};
    const std::array<std::string, 2> secondary_devices{
        options.secondary_device0, options.secondary_device1};
    const std::array<std::uint8_t, 2> primary_gids{
        options.primary_gid0, options.primary_gid1};
    const std::array<std::uint8_t, 2> secondary_gids{
        options.secondary_gid0, options.secondary_gid1};
    const std::array<std::uint16_t, 2> primary_ports{
        options.primary_port0, options.primary_port1};
    const std::array<std::uint16_t, 2> secondary_ports{
        options.secondary_port0, options.secondary_port1};

    std::array<std::unique_ptr<spark_transport::VerbsEndpoint>, 2> primary;
    std::array<std::unique_ptr<spark_transport::VerbsEndpoint>, 2> secondary;
    std::array<std::unique_ptr<spark_transport::ControlChannel>, 2>
        primary_channels;
    std::array<std::unique_ptr<spark_transport::ControlChannel>, 2>
        secondary_channels;
    for (std::uint32_t link = 0; link < 2; ++link) {
      const std::int32_t direction =
          direction_for_plan(options.rank, link);
      primary_channels[link] = std::make_unique<spark_transport::ControlChannel>(
          open_channel(plans[link], peers[link], primary_ports[link]));
      primary[link] = std::make_unique<spark_transport::VerbsEndpoint>(
          primary_devices[link], 1, primary_gids[link], *arena_buffer);
      connect_endpoint(*primary_channels[link], *primary[link], plans[link],
                       options.rank, direction, 0);
      secondary_channels[link] =
          std::make_unique<spark_transport::ControlChannel>(open_channel(
              plans[link], secondary_peers[link], secondary_ports[link]));
      secondary[link] = std::make_unique<spark_transport::VerbsEndpoint>(
          secondary_devices[link], 1, secondary_gids[link], *arena_buffer);
      connect_endpoint(*secondary_channels[link], *secondary[link], plans[link],
                       options.rank, direction, 1);
    }

    const std::uint32_t clockwise_link =
        static_cast<std::uint32_t>(
            spark_transport::tp4_prefill_outgoing_endpoint(
                options.rank,
                spark_transport::Tp4PrefillDirection::kClockwise));
    const std::uint32_t counterclockwise_link = 1U - clockwise_link;
    research::FusedPrefillVerbsProxyConfig proxy_config{
        options.rank,
        options.proxy_cpu,
        options.timeout_seconds * 1000U,
        {primary[clockwise_link].get(), secondary[clockwise_link].get(),
         primary[counterclockwise_link].get(),
         secondary[counterclockwise_link].get()}};
    research::FusedPrefillVerbsProxy proxy(arena, proxy_config);
    ProxyWorker proxy_worker(proxy);

    const std::uint64_t guarded_bytes =
        research::kFusedPrefillPayloadBytes + 2U * kGuardBytes;
    std::uint8_t* guarded_input{};
    std::uint8_t* guarded_output{};
    research::FusedPrefillDeviceSync* device_sync{};
    research::FusedPrefillDescriptor* device_descriptors{};
    cudaStream_t stream{};
    cudaEvent_t device_start{};
    cudaEvent_t device_stop{};
    check_cuda(cudaMalloc(&guarded_input, guarded_bytes), "allocate guarded input");
    check_cuda(cudaMalloc(&guarded_output, guarded_bytes),
               "allocate guarded output");
    check_cuda(cudaMemset(guarded_input, kInputGuard, guarded_bytes),
               "initialize input guards");
    check_cuda(cudaMemset(guarded_output, kOutputGuard, guarded_bytes),
               "initialize output guards");
    check_cuda(cudaMalloc(&device_sync,
                          sizeof(*device_sync) * research::kFusedPrefillFlows),
               "allocate device synchronization");
    check_cuda(cudaMemset(device_sync, 0,
                          sizeof(*device_sync) * research::kFusedPrefillFlows),
               "initialize device synchronization");
    check_cuda(cudaMalloc(&device_descriptors,
                          sizeof(*device_descriptors) *
                              research::kFusedPrefillFlows),
               "allocate fused descriptors");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create caller stream");
    check_cuda(cudaEventCreate(&device_start), "create device start event");
    check_cuda(cudaEventCreate(&device_stop), "create device stop event");

    for (const auto& channel : primary_channels) channel->barrier();
    for (const auto& channel : secondary_channels) channel->barrier();
    std::uint64_t sequence{};
    const auto exact = run_case(
        false, options, arena, proxy_worker, guarded_input, guarded_output,
        device_sync, device_descriptors, stream, device_start, device_stop,
        sequence);
    print_receipt(false, options, exact, multiprocessors);
    const auto noninteger = run_case(
        true, options, arena, proxy_worker, guarded_input, guarded_output,
        device_sync, device_descriptors, stream, device_start, device_stop,
        sequence);
    print_receipt(true, options, noninteger, multiprocessors);

    check_cuda(cudaEventDestroy(device_stop), "destroy device stop event");
    check_cuda(cudaEventDestroy(device_start), "destroy device start event");
    check_cuda(cudaStreamDestroy(stream), "destroy caller stream");
    check_cuda(cudaFree(device_descriptors), "free fused descriptors");
    check_cuda(cudaFree(device_sync), "free device synchronization");
    check_cuda(cudaFree(guarded_output), "free guarded output");
    check_cuda(cudaFree(guarded_input), "free guarded input");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TP4_FUSED_PREFILL_ERROR " << error.what() << '\n';
    return 1;
  }
}
