#include "fused_prefill_kernels.cuh"

#include "spark_transport/tp4_bidirectional_prefill.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace research = spark_transport::tiled_prefill_research;

namespace {

constexpr std::uint32_t kRank = 0;
constexpr std::size_t kElements =
    research::kFusedPrefillPayloadBytes / sizeof(std::uint16_t);
constexpr std::size_t kTileElements =
    research::kFusedPrefillTileBytes / sizeof(std::uint16_t);

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
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
      (element * 37U + static_cast<std::size_t>(rank) * 53U) % 257U) -
      128;
  const float scale = 0.0078125F * static_cast<float>(rank + 1U);
  const float offset =
      static_cast<float>((element + 3U * rank) % 11U) * 0.001953125F;
  return float_to_bf16(static_cast<float>(centered) * scale + offset);
}

std::uint32_t wrap_rank(std::int32_t rank) {
  constexpr std::int32_t ranks = research::kFusedPrefillRanks;
  const std::int32_t remainder = rank % ranks;
  return static_cast<std::uint32_t>(remainder < 0 ? remainder + ranks
                                                  : remainder);
}

struct MappedBytes {
  std::uint8_t* host{};
  std::uint8_t* device{};
  std::size_t bytes{};

  explicit MappedBytes(std::size_t allocation_bytes) : bytes(allocation_bytes) {
    check_cuda(cudaHostAlloc(&host, bytes, cudaHostAllocMapped),
               "cudaHostAlloc mapped proxy region");
    check_cuda(cudaHostGetDevicePointer(&device, host, 0),
               "cudaHostGetDevicePointer proxy region");
    std::memset(host, 0, bytes);
  }
  MappedBytes(const MappedBytes&) = delete;
  MappedBytes& operator=(const MappedBytes&) = delete;
  ~MappedBytes() {
    if (host != nullptr) cudaFreeHost(host);
  }
};

struct Flow {
  MappedBytes primary_in{research::kFusedPrefillRailPlaneBytes};
  MappedBytes secondary_in{research::kFusedPrefillRailPlaneBytes};
  MappedBytes primary_out{research::kFusedPrefillRailPlaneBytes};
  MappedBytes secondary_out{research::kFusedPrefillRailPlaneBytes};
  MappedBytes control{sizeof(research::FusedPrefillHostControl)};
  research::FusedPrefillDeviceSync* sync{};

  Flow() {
    check_cuda(cudaMalloc(&sync, sizeof(*sync)), "cudaMalloc device sync");
    check_cuda(cudaMemset(sync, 0, sizeof(*sync)), "clear device sync");
  }
  Flow(const Flow&) = delete;
  Flow& operator=(const Flow&) = delete;
  ~Flow() {
    if (sync != nullptr) cudaFree(sync);
  }
  auto* host_control() {
    return reinterpret_cast<research::FusedPrefillHostControl*>(control.host);
  }
  auto* device_control() {
    return reinterpret_cast<research::FusedPrefillHostControl*>(control.device);
  }
};

struct ProxyConfig {
  std::uint32_t operations{100};
  std::uint32_t primary_delay_us{3};
  std::uint32_t secondary_delay_us{11};
  std::uint32_t cqe_delay_us{5};
  std::uint32_t credit_delay_us{7};
};

void delay_us(std::uint32_t delay) {
  if (delay != 0) std::this_thread::sleep_for(std::chrono::microseconds(delay));
}

std::uint64_t load_acquire(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_release(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

void wait_exact(const std::uint64_t* address, std::uint64_t expected,
                const char* name) {
  for (std::uint64_t spins = 0; spins < 1000000000ULL; ++spins) {
    const std::uint64_t observed = load_acquire(address);
    if (observed == expected) return;
    if (observed > expected) {
      throw std::runtime_error(std::string(name) + " skipped token");
    }
    if ((spins & 0x3fffU) == 0) std::this_thread::yield();
  }
  throw std::runtime_error(std::string(name) + " timed out");
}

using Tile = std::vector<std::uint16_t>;
using StageTiles = std::array<Tile, research::kFusedPrefillStages>;
using ProxyPayloads = std::array<StageTiles, research::kFusedPrefillFlows>;

std::uint32_t flow_direction(std::uint32_t flow) {
  return flow < research::kFusedPrefillTilesPerShard ? 1U : UINT32_MAX;
}

std::int32_t direction_step(std::uint32_t flow) {
  return flow_direction(flow) == 1U ? 1 : -1;
}

std::uint32_t flow_tile(std::uint32_t flow) {
  return flow % research::kFusedPrefillTilesPerShard;
}

std::uint32_t receive_shard(std::int32_t direction, std::uint32_t stage) {
  if (stage < 3U) {
    return wrap_rank(-direction * static_cast<std::int32_t>(stage + 1U));
  }
  return wrap_rank(-direction * static_cast<std::int32_t>(stage - 3U));
}

std::size_t tensor_offset(std::uint32_t flow, std::uint32_t stage) {
  const std::int32_t direction = direction_step(flow);
  const std::size_t half = direction == 1 ? 0 : research::kFusedPrefillHalfBytes;
  return half +
         static_cast<std::size_t>(receive_shard(direction, stage)) *
             research::kFusedPrefillShardBytes +
         static_cast<std::size_t>(flow_tile(flow)) *
             research::kFusedPrefillTileBytes;
}

std::size_t initial_offset(std::uint32_t flow) {
  const std::size_t half =
      direction_step(flow) == 1 ? 0 : research::kFusedPrefillHalfBytes;
  return half + static_cast<std::size_t>(flow_tile(flow)) *
                    research::kFusedPrefillTileBytes;
}

std::uint16_t reduce_ranks(bool noninteger, std::size_t element,
                           const std::array<std::uint32_t, 4>& ranks,
                           std::uint32_t count) {
  std::uint16_t result = input_value(noninteger, ranks[0], element);
  for (std::uint32_t index = 1; index < count; ++index) {
    result = bf16_add(result, input_value(noninteger, ranks[index], element));
  }
  return result;
}

ProxyPayloads make_proxy_payloads(bool noninteger) {
  ProxyPayloads payloads{};
  for (std::uint32_t flow = 0; flow < research::kFusedPrefillFlows; ++flow) {
    const std::int32_t direction = direction_step(flow);
    for (std::uint32_t stage = 0; stage < research::kFusedPrefillStages;
         ++stage) {
      auto& tile = payloads[flow][stage];
      tile.resize(kTileElements);
      const std::size_t base = tensor_offset(flow, stage);
      std::array<std::uint32_t, 4> ranks{};
      std::uint32_t count{};
      if (stage < 3U) {
        count = stage + 1U;
        for (std::uint32_t contributor = 0; contributor < count;
             ++contributor) {
          ranks[contributor] = wrap_rank(
              -direction * static_cast<std::int32_t>(count - contributor));
        }
      } else {
        count = research::kFusedPrefillRanks;
        const std::uint32_t shard = receive_shard(direction, stage);
        for (std::uint32_t contributor = 0; contributor < count;
             ++contributor) {
          ranks[contributor] = wrap_rank(
              static_cast<std::int32_t>(shard) +
              direction * static_cast<std::int32_t>(contributor));
        }
      }
      for (std::size_t index = 0; index < kTileElements; ++index) {
        tile[index] = reduce_ranks(noninteger, base / 2U + index, ranks, count);
      }
    }
  }
  return payloads;
}

ProxyPayloads make_expected_outgoing(bool noninteger) {
  ProxyPayloads payloads{};
  for (std::uint32_t flow = 0; flow < research::kFusedPrefillFlows; ++flow) {
    const std::int32_t direction = direction_step(flow);
    for (std::uint32_t stage = 0; stage < research::kFusedPrefillStages;
         ++stage) {
      auto& tile = payloads[flow][stage];
      tile.resize(kTileElements);
      const std::size_t base =
          stage == 0 ? initial_offset(flow) : tensor_offset(flow, stage - 1U);
      std::array<std::uint32_t, 4> ranks{};
      std::uint32_t count{};
      if (stage == 0) {
        ranks[0] = kRank;
        count = 1;
      } else if (stage < 3U) {
        count = stage + 1U;
        for (std::uint32_t contributor = 0; contributor < stage;
             ++contributor) {
          ranks[contributor] = wrap_rank(
              -direction * static_cast<std::int32_t>(stage - contributor));
        }
        ranks[stage] = kRank;
      } else {
        count = research::kFusedPrefillRanks;
        const std::uint32_t shard = receive_shard(direction, stage - 1U);
        for (std::uint32_t contributor = 0; contributor < count;
             ++contributor) {
          ranks[contributor] = wrap_rank(
              static_cast<std::int32_t>(shard) +
              direction * static_cast<std::int32_t>(contributor));
        }
      }
      for (std::size_t index = 0; index < kTileElements; ++index) {
        tile[index] = reduce_ranks(noninteger, base / 2U + index, ranks, count);
      }
    }
  }
  return payloads;
}

std::vector<std::uint16_t> make_input(bool noninteger) {
  std::vector<std::uint16_t> input(kElements);
  for (std::size_t element = 0; element < input.size(); ++element) {
    input[element] = input_value(noninteger, kRank, element);
  }
  return input;
}

std::vector<std::uint16_t> make_expected(bool noninteger) {
  std::vector<std::uint16_t> expected(kElements);
  for (std::size_t element = 0; element < expected.size(); ++element) {
    const bool clockwise = element * 2U < research::kFusedPrefillHalfBytes;
    const std::int32_t direction = clockwise ? 1 : -1;
    const std::size_t half_element = clockwise
                                         ? element
                                         : element - research::kFusedPrefillHalfBytes / 2U;
    const std::uint32_t shard = static_cast<std::uint32_t>(
        (half_element * 2U) / research::kFusedPrefillShardBytes);
    std::array<std::uint32_t, 4> ranks{};
    for (std::uint32_t contributor = 0;
         contributor < research::kFusedPrefillRanks; ++contributor) {
      ranks[contributor] = wrap_rank(
          static_cast<std::int32_t>(shard) +
          direction * static_cast<std::int32_t>(contributor));
    }
    expected[element] = reduce_ranks(noninteger, element, ranks, 4);
  }
  return expected;
}

void proxy_one_operation(
    std::array<Flow, research::kFusedPrefillFlows>& flows,
    const ProxyPayloads& payloads, const ProxyPayloads& expected_outgoing,
    std::uint64_t sequence, const ProxyConfig& config,
    std::atomic<std::uint64_t>& outgoing_mismatches) {
  // Reverse selected tiles on alternating stages. This ensures the fused
  // kernel tolerates independent flow progress instead of accidentally
  // relying on lockstep CPU service.
  for (std::uint32_t stage = 0; stage < research::kFusedPrefillStages;
       ++stage) {
    for (std::uint32_t order = 0; order < research::kFusedPrefillFlows;
         ++order) {
      const std::uint32_t flow_index =
          (stage & 1U) == 0 ? order
                            : research::kFusedPrefillFlows - 1U - order;
      auto* control = flows[flow_index].host_control();
      const std::uint32_t parity = research::fused_prefill_parity(stage);
      const std::uint64_t token =
          research::fused_prefill_stage_token(sequence, stage);
      wait_exact(&control->producer[parity], token, "producer");

      const auto* expected = reinterpret_cast<const std::uint8_t*>(
          expected_outgoing[flow_index][stage].data());
      const std::uint8_t* observed_primary =
          flows[flow_index].primary_out.host +
          parity * research::kFusedPrefillRailBytes;
      const std::uint8_t* observed_secondary =
          flows[flow_index].secondary_out.host +
          parity * research::kFusedPrefillRailBytes;
      if (std::memcmp(observed_primary, expected,
                      research::kFusedPrefillRailBytes) != 0) {
        outgoing_mismatches.fetch_add(1, std::memory_order_relaxed);
      }
      if (std::memcmp(observed_secondary,
                      expected + research::kFusedPrefillRailBytes,
                      research::kFusedPrefillRailBytes) != 0) {
        outgoing_mismatches.fetch_add(1, std::memory_order_relaxed);
      }

      const auto* source = reinterpret_cast<const std::uint8_t*>(
          payloads[flow_index][stage].data());
      std::uint8_t* primary = flows[flow_index].primary_in.host +
                              parity * research::kFusedPrefillRailBytes;
      std::uint8_t* secondary = flows[flow_index].secondary_in.host +
                                parity * research::kFusedPrefillRailBytes;
      std::memcpy(primary, source, research::kFusedPrefillRailBytes);
      delay_us(config.primary_delay_us);
      store_release(&control->primary_doorbell[parity], token);
      std::memcpy(secondary, source + research::kFusedPrefillRailBytes,
                  research::kFusedPrefillRailBytes);
      // Tile three is deliberately the slow secondary rail. It verifies that
      // neither the consumer nor reuse token can run ahead of the late half.
      delay_us(config.secondary_delay_us +
               (flow_tile(flow_index) == 3U ? config.secondary_delay_us : 0U));
      store_release(&control->secondary_doorbell[parity], token);

      wait_exact(&control->consumer[parity], token, "consumer");
      // Model independent local-CQE and reciprocal-credit retirement. Reuse
      // is the AND gate and is never published after just one condition.
      delay_us(config.cqe_delay_us);
      const bool local_cqe_retired = true;
      delay_us(config.credit_delay_us);
      const bool peer_credit_observed = true;
      if (!local_cqe_retired || !peer_credit_observed) {
        throw std::runtime_error("proxy retirement gate failed");
      }
      store_release(&control->reuse[parity], token);
    }
  }
}

void reset_case(std::array<Flow, research::kFusedPrefillFlows>& flows) {
  for (auto& flow : flows) {
    std::memset(flow.control.host, 0, flow.control.bytes);
    check_cuda(cudaMemset(flow.sync, 0, sizeof(*flow.sync)),
               "reset device sync");
  }
}

research::FusedPrefillDescriptor make_descriptor(
    Flow& flow, const std::uint8_t* input, std::uint8_t* output,
    std::uint32_t flow_index, std::uint64_t sequence) {
  research::FusedPrefillDescriptor descriptor{};
  descriptor.input = input;
  descriptor.output = output;
  descriptor.primary_incoming = flow.primary_in.device;
  descriptor.secondary_incoming = flow.secondary_in.device;
  descriptor.primary_outgoing = flow.primary_out.device;
  descriptor.secondary_outgoing = flow.secondary_out.device;
  descriptor.host_control = flow.device_control();
  descriptor.device_sync = flow.sync;
  descriptor.initial_tensor_offset_bytes = initial_offset(flow_index);
  for (std::uint32_t stage = 0; stage < research::kFusedPrefillStages;
       ++stage) {
    descriptor.tensor_offset_bytes[stage] = tensor_offset(flow_index, stage);
  }
  descriptor.operation_sequence = sequence;
  descriptor.rank = kRank;
  descriptor.direction = direction_step(flow_index);
  descriptor.tile = flow_tile(flow_index);
  descriptor.spin_limit = std::numeric_limits<std::uint32_t>::max();
  return descriptor;
}

void run_case(bool noninteger, const ProxyConfig& config,
              std::array<Flow, research::kFusedPrefillFlows>& flows,
              cudaStream_t stream) {
  reset_case(flows);
  const auto input = make_input(noninteger);
  const auto expected = make_expected(noninteger);
  const auto payloads = make_proxy_payloads(noninteger);
  const auto expected_outgoing = make_expected_outgoing(noninteger);
  std::uint8_t* device_input{};
  std::uint8_t* device_output{};
  research::FusedPrefillDescriptor* device_descriptors{};
  check_cuda(cudaMalloc(&device_input, research::kFusedPrefillPayloadBytes),
             "cudaMalloc input");
  check_cuda(cudaMalloc(&device_output, research::kFusedPrefillPayloadBytes),
             "cudaMalloc output");
  check_cuda(cudaMalloc(&device_descriptors,
                        sizeof(research::FusedPrefillDescriptor) *
                            research::kFusedPrefillFlows),
             "cudaMalloc descriptors");
  check_cuda(cudaMemcpy(device_input, input.data(),
                        research::kFusedPrefillPayloadBytes,
                        cudaMemcpyHostToDevice),
             "copy input");

  const auto started = std::chrono::steady_clock::now();
  std::atomic<std::uint64_t> outgoing_mismatches{};
  for (std::uint64_t sequence = 0; sequence < config.operations; ++sequence) {
    check_cuda(cudaMemsetAsync(device_output, 0xa5,
                               research::kFusedPrefillPayloadBytes, stream),
               "poison output");
    std::array<research::FusedPrefillDescriptor,
               research::kFusedPrefillFlows>
        descriptors{};
    for (std::uint32_t flow = 0; flow < research::kFusedPrefillFlows;
         ++flow) {
      descriptors[flow] = make_descriptor(
          flows[flow], device_input, device_output, flow, sequence);
    }
    check_cuda(cudaMemcpyAsync(device_descriptors, descriptors.data(),
                               sizeof(descriptors), cudaMemcpyHostToDevice,
                               stream),
               "copy descriptors");
    check_cuda(research::launch_fused_prefill_q8192_n4(device_descriptors,
                                                        stream),
               "launch fused prefill");
    std::exception_ptr proxy_error;
    std::thread proxy([&] {
      try {
        proxy_one_operation(flows, payloads, expected_outgoing, sequence,
                            config, outgoing_mismatches);
      } catch (...) {
        proxy_error = std::current_exception();
      }
    });
    check_cuda(cudaStreamSynchronize(stream), "retire fused prefill");
    proxy.join();
    if (proxy_error != nullptr) std::rethrow_exception(proxy_error);
  }
  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started);

  std::vector<std::uint16_t> observed(kElements);
  check_cuda(cudaMemcpy(observed.data(), device_output,
                        research::kFusedPrefillPayloadBytes,
                        cudaMemcpyDeviceToHost),
             "copy output");
  std::uint64_t mismatches{};
  double max_abs{};
  for (std::size_t element = 0; element < observed.size(); ++element) {
    if (observed[element] != expected[element]) ++mismatches;
    max_abs = std::max(
        max_abs, static_cast<double>(std::abs(bf16_to_float(observed[element]) -
                                              bf16_to_float(expected[element]))));
  }
  for (const auto& flow : flows) {
    if (load_acquire(&reinterpret_cast<const research::FusedPrefillHostControl*>(
                          flow.control.host)->poison_sequence) != 0) {
      throw std::runtime_error("kernel published poison");
    }
  }
  std::cout << "case=" << (noninteger ? "noninteger" : "exact")
            << " operations=" << config.operations
            << " bitwise_mismatches=" << mismatches
            << " outgoing_mismatches=" << outgoing_mismatches.load()
            << " max_abs=" << max_abs
            << " mean_us="
            << elapsed.count() * 1.0e6 / static_cast<double>(config.operations)
            << '\n';
  cudaFree(device_descriptors);
  cudaFree(device_output);
  cudaFree(device_input);
  if (mismatches != 0 || outgoing_mismatches.load() != 0) {
    throw std::runtime_error("fused prefill differs from scalar BF16 oracle");
  }
}

ProxyConfig parse_config(int argc, char** argv) {
  ProxyConfig config{};
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto parse = [&](const std::string& prefix, std::uint32_t& target) {
      if (argument.rfind(prefix, 0) != 0) return false;
      target = static_cast<std::uint32_t>(std::stoul(argument.substr(prefix.size())));
      return true;
    };
    if (parse("--operations=", config.operations) ||
        parse("--primary-delay-us=", config.primary_delay_us) ||
        parse("--secondary-delay-us=", config.secondary_delay_us) ||
        parse("--cqe-delay-us=", config.cqe_delay_us) ||
        parse("--credit-delay-us=", config.credit_delay_us)) {
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument);
  }
  if (config.operations < 100U) {
    throw std::invalid_argument("proxy smoke requires at least 100 operations");
  }
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ProxyConfig config = parse_config(argc, argv);
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    std::array<Flow, research::kFusedPrefillFlows> flows;
    cudaStream_t stream{};
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream");
    run_case(false, config, flows, stream);
    run_case(true, config, flows, stream);
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    std::cout << "fused_prefill_proxy_smoke=PASS delayed_tile=3 rails=2"
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fused_prefill_proxy_smoke=FAIL error=" << error.what()
              << '\n';
    return 1;
  }
}
