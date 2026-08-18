#include "spark_transport/tp4_session.hpp"
#include "spark_transport/tp4_graph_command.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kElements = 6144;
constexpr std::size_t kPayloadBytes =
    kElements * sizeof(__nv_bfloat16);
// This GLM qualification surface remains intentionally bounded at Q512 even
// though the versioned transport descriptor also serves DeepSeek row widths.
constexpr std::size_t kMaximumQ = 512;
static_assert(kMaximumQ <= spark_transport::kTp4GraphAllreduceMaximumQ);
constexpr std::size_t kMaximumElements = kMaximumQ * kElements;
constexpr std::size_t kMaximumPayloadBytes =
    kMaximumElements * sizeof(__nv_bfloat16);
static_assert(kMaximumPayloadBytes == 6U * 1024U * 1024U);

// A physical deferred-ACK slot is reused every two logical sequences. An odd
// input-epoch period prevents sequence S from having the same payload as S-2.
// BF16 represents every integer through 256 exactly. Each rank contributes an
// integer no larger than 32, so both two-rank partials (<=64) and the final
// four-rank sum (<=128) remain exact under the transport's BF16 additions.
constexpr unsigned long long kInputEpochPeriod = 7;
constexpr unsigned int kInputEpochStep = 4;
constexpr unsigned int kMaximumInputBase = 8;
constexpr unsigned int kMaximumInputValue =
    kMaximumInputBase +
    static_cast<unsigned int>(kInputEpochPeriod - 1) * kInputEpochStep;
constexpr unsigned int kMaximumReducedValue = 4 * kMaximumInputValue;
constexpr unsigned int kBfloat16ConsecutiveIntegerLimit = 256;
static_assert(kInputEpochPeriod % 2 != 0);
static_assert(kMaximumInputValue == 32);
static_assert(kMaximumReducedValue == 128);
static_assert(kMaximumReducedValue <= kBfloat16ConsecutiveIntegerLimit);

enum class TimingMode {
  kBurst,
  kIsolated,
};

struct Options {
  spark_transport::Tp4AllreduceOptions transport;
  int warmup{10};
  int iterations{100};
  int operations_per_graph{1};
  bool multi_graph_validation{};
  bool mixed_q_validation{};
  std::uint32_t fixed_q{1};
  std::uint32_t maximum_q{6};
  std::size_t elements{kElements};
  int graph_a_operations{3};
  int graph_b_operations{128};
  TimingMode timing_mode{TimingMode::kBurst};
  double max_graph_submit_us{};
  double max_device_us{};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n\n"
      << "Options:\n"
      << "  --device0 HCA\n"
      << "  --device1 HCA\n"
      << "  --gid0 INDEX\n"
      << "  --gid1 INDEX\n"
      << "  --control-port0 PORT\n"
      << "  --control-port1 PORT\n"
      << "  --warmup COUNT\n"
      << "  --iterations COUNT\n"
      << "  --operations-per-graph COUNT\n"
      << "  --multi-graph-validation\n"
      << "  --mixed-q-validation\n"
      << "  --fixed-q Q\n"
      << "  --elements-per-row 6144|4096\n"
      << "  --maximum-q Q\n"
      << "  --graph-a-operations COUNT\n"
      << "  --graph-b-operations COUNT\n"
      << "  --graph-submit-cpu CPU\n"
      << "  --graph-progress-cpu CPU\n"
      << "  --allreduce-protocol serial_ack|two_slot_deferred_ack\n"
      << "  --graph-kernel fused|split_64k|tiered_64k\n"
      << "  --wire-schedule sequential|dual_port_striped\n"
      << "  --timing-mode burst|isolated\n"
      << "  --max-graph-submit-us MICROSECONDS\n"
      << "  --max-device-us OUTPUT_READY_MICROSECONDS\n";
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

double positive_double(const char* value, const char* name) {
  std::size_t consumed{};
  const std::string text(value);
  const double parsed = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(parsed) || parsed <= 0.0) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  options.transport.payload_bytes = kPayloadBytes;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto take_value = [&]() -> const char* {
      if (++index >= argc) {
        usage(argv[0]);
      }
      return argv[index];
    };

    if (argument == "--rank") {
      options.transport.rank =
          static_cast<std::uint32_t>(unsigned_value(take_value(), "rank"));
    } else if (argument == "--peer0") {
      options.transport.peer0 = take_value();
    } else if (argument == "--peer1") {
      options.transport.peer1 = take_value();
    } else if (argument == "--device0") {
      options.transport.device0 = take_value();
    } else if (argument == "--device1") {
      options.transport.device1 = take_value();
    } else if (argument == "--gid0") {
      options.transport.gid0 =
          static_cast<std::uint8_t>(unsigned_value(take_value(), "GID 0"));
    } else if (argument == "--gid1") {
      options.transport.gid1 =
          static_cast<std::uint8_t>(unsigned_value(take_value(), "GID 1"));
    } else if (argument == "--control-port0") {
      options.transport.control_port0 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 0"));
    } else if (argument == "--control-port1") {
      options.transport.control_port1 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 1"));
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup count"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iteration count"));
    } else if (argument == "--operations-per-graph") {
      options.operations_per_graph = static_cast<int>(
          unsigned_value(take_value(), "operations per graph"));
    } else if (argument == "--multi-graph-validation") {
      options.multi_graph_validation = true;
    } else if (argument == "--mixed-q-validation") {
      options.mixed_q_validation = true;
    } else if (argument == "--fixed-q") {
      const std::uint64_t fixed_q =
          unsigned_value(take_value(), "fixed Q");
      if (fixed_q == 0 || fixed_q > kMaximumQ) {
        throw std::invalid_argument("fixed Q must be in [1, 512]");
      }
      options.fixed_q = static_cast<std::uint32_t>(fixed_q);
    } else if (argument == "--elements-per-row") {
      const std::uint64_t elements =
          unsigned_value(take_value(), "elements per row");
      if (elements != 6144 && elements != 4096) {
        throw std::invalid_argument(
            "elements per row must be 6144 or 4096");
      }
      options.elements = static_cast<std::size_t>(elements);
      options.transport.elements_per_row =
          static_cast<std::uint32_t>(elements);
      options.transport.bytes_per_row =
          static_cast<std::uint32_t>(elements) *
          spark_transport::kTp4GraphBytesPerElement;
    } else if (argument == "--maximum-q") {
      const std::uint64_t maximum_q =
          unsigned_value(take_value(), "maximum Q");
      if (maximum_q > kMaximumQ) {
        throw std::invalid_argument("maximum Q must be in [6, 512]");
      }
      options.maximum_q = static_cast<std::uint32_t>(maximum_q);
    } else if (argument == "--graph-a-operations") {
      options.graph_a_operations = static_cast<int>(
          unsigned_value(take_value(), "graph A operations"));
    } else if (argument == "--graph-b-operations") {
      options.graph_b_operations = static_cast<int>(
          unsigned_value(take_value(), "graph B operations"));
    } else if (argument == "--graph-submit-cpu") {
      options.transport.graph_submit_cpu = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "graph submit CPU"));
    } else if (argument == "--graph-progress-cpu") {
      options.transport.graph_progress_cpu = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "graph progress CPU"));
    } else if (argument == "--allreduce-protocol") {
      options.transport.protocol =
          spark_transport::parse_tp4_allreduce_protocol(take_value());
    } else if (argument == "--graph-kernel") {
      const std::string_view graph_kernel(take_value());
      if (graph_kernel == "fused") {
        options.transport.graph_kernel_strategy =
            spark_transport::Tp4GraphKernelStrategy::kFused;
      } else if (graph_kernel == "split_64k") {
        options.transport.graph_kernel_strategy =
            spark_transport::Tp4GraphKernelStrategy::kSplit64KiB;
      } else if (graph_kernel == "tiered_64k") {
        options.transport.graph_kernel_strategy =
            spark_transport::Tp4GraphKernelStrategy::kTiered64KiB;
      } else {
        throw std::invalid_argument(
            "graph kernel must be fused, split_64k, or tiered_64k");
      }
    } else if (argument == "--wire-schedule") {
      const std::string_view wire_schedule(take_value());
      if (wire_schedule == "sequential") {
        options.transport.schedule =
            spark_transport::Tp4AllreduceSchedule::kSequential;
      } else if (wire_schedule == "dual_port_striped") {
        options.transport.schedule =
            spark_transport::Tp4AllreduceSchedule::kDualPortStriped;
      } else {
        throw std::invalid_argument(
            "wire schedule must be sequential or dual_port_striped");
      }
    } else if (argument == "--timing-mode") {
      const std::string_view timing_mode(take_value());
      if (timing_mode == "burst") {
        options.timing_mode = TimingMode::kBurst;
      } else if (timing_mode == "isolated") {
        options.timing_mode = TimingMode::kIsolated;
      } else {
        throw std::invalid_argument(
            "timing mode must be burst or isolated");
      }
    } else if (argument == "--max-graph-submit-us") {
      options.max_graph_submit_us =
          positive_double(take_value(), "graph submit threshold");
    } else if (argument == "--max-device-us") {
      options.max_device_us =
          positive_double(take_value(), "device threshold");
    } else {
      usage(argv[0]);
    }
  }

  if (options.transport.rank >= 4 || options.transport.peer0.empty() ||
      options.transport.peer1.empty() || options.warmup < 0 ||
      options.iterations <= 0 || options.operations_per_graph <= 0 ||
      options.operations_per_graph > 4096 ||
      options.graph_a_operations <= 0 || options.graph_a_operations > 4096 ||
      options.graph_b_operations <= 0 || options.graph_b_operations > 4096) {
    usage(argv[0]);
  }
  if (options.multi_graph_validation &&
      (options.max_graph_submit_us != 0.0 || options.max_device_us != 0.0)) {
    throw std::invalid_argument(
        "multi-graph validation requires disabled performance gates");
  }
  if (options.multi_graph_validation &&
      (options.graph_a_operations > 16 ||
       options.graph_b_operations != 128)) {
    throw std::invalid_argument(
        "multi-graph validation requires graph A <= 16 nodes and "
        "graph B exactly 128 nodes");
  }
  if (options.mixed_q_validation &&
      !options.multi_graph_validation) {
    throw std::invalid_argument(
        "mixed-Q validation requires multi-graph validation");
  }
  if (options.timing_mode == TimingMode::kIsolated &&
      (options.multi_graph_validation ||
       options.mixed_q_validation ||
       options.operations_per_graph != 1)) {
    throw std::invalid_argument(
        "isolated timing requires one single-graph collective");
  }
  if (options.fixed_q != 1 &&
      (options.multi_graph_validation ||
       options.mixed_q_validation ||
       options.operations_per_graph != 1)) {
    throw std::invalid_argument(
        "fixed Q above one requires one single-graph collective");
  }
  if (options.transport.graph_kernel_strategy ==
      spark_transport::Tp4GraphKernelStrategy::kSplit64KiB) {
    if (options.multi_graph_validation ||
        options.mixed_q_validation ||
        options.operations_per_graph != 1) {
      throw std::invalid_argument(
          "split_64k graph kernel requires fixed Q1 through Q512, "
          "and one collective per graph");
    }
  }
  if (options.transport.schedule ==
      spark_transport::Tp4AllreduceSchedule::kDualPortStriped) {
    if ((options.fixed_q != 40 && options.fixed_q != 512) ||
        options.multi_graph_validation ||
        options.mixed_q_validation ||
        options.operations_per_graph != 1 ||
        options.transport.protocol !=
            spark_transport::Tp4AllreduceProtocol::kTwoSlotDeferredAck ||
        options.transport.graph_kernel_strategy !=
            spark_transport::Tp4GraphKernelStrategy::kFused) {
      throw std::invalid_argument(
          "dual_port_striped wire schedule requires fixed Q40 or Q512, "
          "two_slot_deferred_ack, fused graph kernel, and one "
          "single-graph collective");
    }
  }
  if (options.maximum_q < 6 || options.maximum_q > kMaximumQ) {
    throw std::invalid_argument("maximum Q must be in [6, 512]");
  }
  if (options.elements != kElements && options.multi_graph_validation &&
      !options.mixed_q_validation) {
    // The plain multi-graph path validates through the Q1-specific
    // kernel; the mixed-Q path is fully row-parameterized.
    throw std::invalid_argument(
        "multi-graph validation requires the default row width");
  }
  if (options.mixed_q_validation) {
    options.transport.payload_bytes =
        spark_transport::tp4_graph_payload_bytes(
            options.maximum_q, options.transport.bytes_per_row);
  } else {
    options.transport.payload_bytes =
        spark_transport::tp4_graph_payload_bytes(
            options.fixed_q, options.transport.bytes_per_row);
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

spark_transport::Tp4GraphReplayStatus wait_for_graph_completion(
    const spark_transport::Tp4AllreduceSession& session,
    std::uint64_t expected) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (true) {
    const auto status = session.graph_replay_status();
    if (status.overflow_sequence != 0 ||
        status.completed_sequence >= expected) {
      return status;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          "timed out waiting for graph command completion");
    }
    std::this_thread::yield();
  }
}

__device__ float tp4_input_value(std::uint32_t rank, std::size_t element,
                                 unsigned long long input_epoch) {
  const unsigned int epoch_offset =
      static_cast<unsigned int>(input_epoch % kInputEpochPeriod) *
      kInputEpochStep;
  return static_cast<float>(
      ((element * 3U + rank * 5U) & 7U) + 1U + epoch_offset);
}

__device__ unsigned long long mixed_node_input_epoch(
    unsigned long long replay, std::size_t node_epoch_offset,
    std::size_t operations_per_cycle) {
  // The probe always launches graph A then graph B. Two adjacent replay IDs
  // therefore form one graph cycle. node_epoch_offset is the node's ordinal
  // across A+B, making this epoch equal to its one-based logical collective
  // sequence. In particular, every slot occupant S differs from S-2.
  const unsigned long long cycle = (replay - 1ULL) / 2ULL;
  return cycle * static_cast<unsigned long long>(operations_per_cycle) +
         static_cast<unsigned long long>(node_epoch_offset) + 1ULL;
}

__global__ void prepare_replay(__nv_bfloat16* input, std::uint32_t rank,
                               unsigned long long replay,
                               unsigned long long* replay_marker,
                               std::size_t input_elements) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < input_elements) {
    input[index] =
        __float2bfloat16(tp4_input_value(rank, index, replay));
  }
  if (index == 0) {
    *replay_marker = replay;
  }
}

__global__ void prepare_mixed_node_input(
    __nv_bfloat16* input, std::uint32_t rank,
    const unsigned long long* replay_marker, std::size_t input_elements,
    std::size_t node_epoch_offset, std::size_t operations_per_cycle) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= input_elements) {
    return;
  }
  const unsigned long long input_epoch = mixed_node_input_epoch(
      *replay_marker, node_epoch_offset, operations_per_cycle);
  input[index] =
      __float2bfloat16(tp4_input_value(rank, index, input_epoch));
}

__global__ void validate_q1_output(const __nv_bfloat16* output,
                                   std::size_t output_elements,
                                   const unsigned long long* replay_marker,
                                   unsigned long long* mismatches) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= output_elements) {
    return;
  }
  const std::size_t payload_index = index % kElements;
  float expected = 0.0F;
  const unsigned long long replay = *replay_marker;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    expected += tp4_input_value(rank, payload_index, replay);
  }
  if (__bfloat162float(output[index]) != expected) {
    atomicAdd(mismatches, 1ULL);
  }
}

__global__ void validate_active_output(
    const __nv_bfloat16* output, std::size_t active_elements,
    const unsigned long long* replay_marker,
    unsigned long long* mismatches) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= active_elements) {
    return;
  }
  float expected = 0.0F;
  const unsigned long long replay = *replay_marker;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    expected += tp4_input_value(rank, index, replay);
  }
  if (__bfloat162float(output[index]) != expected) {
    atomicAdd(mismatches, 1ULL);
  }
}

__global__ void validate_mixed_node_output(
    const __nv_bfloat16* output, std::size_t active_elements,
    const unsigned long long* replay_marker,
    unsigned long long* mismatches, std::size_t node_epoch_offset,
    std::size_t operations_per_cycle) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= active_elements) {
    return;
  }
  const unsigned long long input_epoch = mixed_node_input_epoch(
      *replay_marker, node_epoch_offset, operations_per_cycle);
  float expected = 0.0F;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    expected += tp4_input_value(rank, index, input_epoch);
  }
  if (__bfloat162float(output[index]) != expected) {
    atomicAdd(mismatches, 1ULL);
  }
}

struct CapturedGraph {
  int operations{};
  std::size_t output_offset{};
  cudaGraphExec_t executable{};
};

CapturedGraph capture_graph(
    spark_transport::Tp4AllreduceSession& session,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned long long* replay_marker, unsigned long long* mismatches,
    cudaStream_t stream, int operations, std::size_t output_offset,
    bool capture_validation) {
  constexpr int threads = 256;
  const std::size_t output_elements =
      kElements * static_cast<std::size_t>(operations);
  const int validation_blocks = static_cast<int>(
      (output_elements + threads - 1) / threads);

  cudaGraph_t graph{};
  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
             "cudaStreamBeginCapture");
  for (int operation = 0; operation < operations; ++operation) {
    session.capture_q1_all_reduce(
        input,
        output + output_offset +
            static_cast<std::size_t>(operation) * kElements,
        stream);
  }
  if (capture_validation) {
    validate_q1_output<<<validation_blocks, threads, 0, stream>>>(
        output + output_offset, output_elements, replay_marker, mismatches);
    check_cuda(cudaGetLastError(), "validate_q1_output capture launch");
  }
  check_cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");

  CapturedGraph captured{operations, output_offset, nullptr};
  try {
    check_cuda(cudaGraphInstantiate(&captured.executable, graph, 0),
               "cudaGraphInstantiate");
  } catch (...) {
    cudaGraphDestroy(graph);
    throw;
  }
  check_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
  return captured;
}

CapturedGraph capture_fixed_q_graph(
    spark_transport::Tp4AllreduceSession& session,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned long long* replay_marker, unsigned long long* mismatches,
    cudaStream_t stream, std::uint32_t q, std::size_t elements,
    bool capture_validation) {
  constexpr int threads = 256;
  const std::size_t active_elements =
      static_cast<std::size_t>(q) * elements;
  const int validation_blocks = static_cast<int>(
      (active_elements + threads - 1) / threads);

  cudaGraph_t graph{};
  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
             "cudaStreamBeginCapture fixed Q");
  session.capture_all_reduce(input, output, q, stream);
  if (capture_validation) {
    validate_active_output<<<validation_blocks, threads, 0, stream>>>(
        output, active_elements, replay_marker, mismatches);
    check_cuda(cudaGetLastError(),
               "validate_active_output fixed-Q capture launch");
  }
  check_cuda(cudaStreamEndCapture(stream, &graph),
             "cudaStreamEndCapture fixed Q");

  CapturedGraph captured{1, 0, nullptr};
  try {
    check_cuda(cudaGraphInstantiate(&captured.executable, graph, 0),
               "cudaGraphInstantiate fixed Q");
  } catch (...) {
    cudaGraphDestroy(graph);
    throw;
  }
  check_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy fixed Q");
  return captured;
}

CapturedGraph capture_mixed_q_graph(
    spark_transport::Tp4AllreduceSession& session,
    __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned long long* replay_marker, unsigned long long* mismatches,
    cudaStream_t stream, std::uint32_t rank,
    const std::vector<std::uint32_t>& q_values,
    std::size_t graph_epoch_offset, std::size_t operations_per_cycle,
    std::size_t elements) {
  constexpr int threads = 256;
  cudaGraph_t graph{};
  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
             "cudaStreamBeginCapture mixed Q");
  for (std::size_t operation = 0;
       operation < q_values.size(); ++operation) {
    const std::uint32_t q = q_values[operation];
    // Each validation immediately follows its collective in the captured
    // graph, so every node can reuse one stable maximum-capacity output
    // arena instead of reserving hundreds of MiB for distinct node outputs.
    __nv_bfloat16* operation_output = output;
    const std::size_t active_elements =
        static_cast<std::size_t>(q) * elements;
    const int validation_blocks = static_cast<int>(
        (active_elements + threads - 1) / threads);
    const std::size_t node_epoch_offset = graph_epoch_offset + operation;
    prepare_mixed_node_input<<<validation_blocks, threads, 0, stream>>>(
        input, rank, replay_marker, active_elements, node_epoch_offset,
        operations_per_cycle);
    check_cuda(cudaGetLastError(),
               "prepare_mixed_node_input capture launch");
    session.capture_all_reduce(input, operation_output, q, stream);
    validate_mixed_node_output<<<validation_blocks, threads, 0, stream>>>(
        operation_output, active_elements, replay_marker, mismatches,
        node_epoch_offset, operations_per_cycle);
    check_cuda(cudaGetLastError(),
               "validate_mixed_node_output capture launch");
  }
  check_cuda(cudaStreamEndCapture(stream, &graph),
             "cudaStreamEndCapture mixed Q");

  CapturedGraph captured{
      static_cast<int>(q_values.size()), 0,
      nullptr};
  try {
    check_cuda(cudaGraphInstantiate(&captured.executable, graph, 0),
               "cudaGraphInstantiate mixed Q");
  } catch (...) {
    cudaGraphDestroy(graph);
    throw;
  }
  check_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy mixed Q");
  return captured;
}

bool attempt_post_replay_capture(
    spark_transport::Tp4AllreduceSession& session,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    cudaStream_t stream, std::uint32_t q) {
  constexpr std::string_view expected_rejection =
      "graph TP4 capture cannot add nodes after the first replay";
  cudaGraph_t rejected_graph{};
  std::string rejection;

  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
             "cudaStreamBeginCapture post-replay rejection");
  try {
    session.capture_all_reduce(input, output, q, stream);
  } catch (const std::logic_error& error) {
    rejection = error.what();
  }

  const cudaError_t end_result =
      cudaStreamEndCapture(stream, &rejected_graph);
  if (end_result == cudaSuccess) {
    if (rejected_graph != nullptr) {
      check_cuda(cudaGraphDestroy(rejected_graph),
                 "cudaGraphDestroy rejected capture");
    }
  } else if (end_result == cudaErrorStreamCaptureInvalidated) {
    // Clear the expected sticky CUDA error. The transport must still have
    // rejected before adding a node; existing executable graphs remain valid.
    (void)cudaGetLastError();
  } else {
    check_cuda(end_result, "cudaStreamEndCapture post-replay rejection");
  }

  if (rejection != expected_rejection) {
    throw std::runtime_error(
        rejection.empty()
            ? "post-replay graph capture was unexpectedly accepted"
            : "post-replay graph capture returned an unexpected rejection: " +
                  rejection);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    constexpr int threads = 256;
    const std::size_t total_output_operations =
        options.multi_graph_validation
            ? static_cast<std::size_t>(options.graph_a_operations) +
                  static_cast<std::size_t>(options.graph_b_operations)
            : static_cast<std::size_t>(options.operations_per_graph);
    const std::size_t input_elements =
        options.mixed_q_validation
            ? static_cast<std::size_t>(options.maximum_q) * options.elements
            : static_cast<std::size_t>(options.fixed_q) * options.elements;
    const int input_blocks = static_cast<int>(
        (input_elements + threads - 1) / threads);
    const std::size_t output_stride_elements =
        options.mixed_q_validation
            ? static_cast<std::size_t>(options.maximum_q) * options.elements
            : static_cast<std::size_t>(options.fixed_q) * options.elements;
    const std::size_t output_elements =
        options.mixed_q_validation
            ? output_stride_elements
            : output_stride_elements * total_output_operations;

    std::vector<std::uint32_t> graph_a_q;
    std::vector<std::uint32_t> graph_b_q;
    std::array<std::uint64_t, kMaximumQ> q_histogram{};
    std::uint64_t active_bytes_per_graph_cycle{};
    const auto account_q = [&](std::uint32_t q) {
      ++q_histogram.at(q - 1);
      active_bytes_per_graph_cycle +=
          spark_transport::tp4_graph_payload_bytes(
              q, options.transport.bytes_per_row);
    };
    if (options.mixed_q_validation) {
      constexpr std::array<std::uint32_t, 3> graph_a_pattern{
          1, 4, 6};
      graph_a_q.reserve(
          static_cast<std::size_t>(options.graph_a_operations));
      for (int operation = 0;
           operation < options.graph_a_operations; ++operation) {
        const std::uint32_t q = graph_a_pattern[
            static_cast<std::size_t>(operation) %
            graph_a_pattern.size()];
        graph_a_q.push_back(q);
        account_q(q);
      }
      graph_b_q.reserve(
          static_cast<std::size_t>(options.graph_b_operations));
      std::vector<std::uint32_t> graph_b_pattern;
      const std::uint32_t decode_maximum =
          std::min(options.maximum_q,
                   spark_transport::kTp4GraphMaximumQ);
      for (std::uint32_t q = 1; q <= decode_maximum; ++q) {
        graph_b_pattern.push_back(q);
      }
      constexpr std::array<std::uint32_t, 4> prefill_pattern{
          48U, 72U, 144U, 512U};
      for (const std::uint32_t q : prefill_pattern) {
        if (q <= options.maximum_q) {
          graph_b_pattern.push_back(q);
        }
      }
      if (graph_b_pattern.back() != options.maximum_q) {
        graph_b_pattern.push_back(options.maximum_q);
      }
      for (int operation = 0;
           operation < options.graph_b_operations; ++operation) {
        const std::uint32_t q =
            graph_b_pattern[
                static_cast<std::size_t>(operation) %
                graph_b_pattern.size()];
        graph_b_q.push_back(q);
        account_q(q);
      }
    } else {
      q_histogram[options.fixed_q - 1] = total_output_operations;
      active_bytes_per_graph_cycle =
          total_output_operations *
          spark_transport::tp4_graph_payload_bytes(
              options.fixed_q, options.transport.bytes_per_row);
    }
    std::uint64_t kernel_split_nodes{};
    for (std::size_t index = 0; index < q_histogram.size(); ++index) {
      if (spark_transport::tp4_graph_kernel_uses_split(
              options.transport.graph_kernel_strategy,
              static_cast<std::uint32_t>(index + 1))) {
        kernel_split_nodes += q_histogram[index];
      }
    }
    const std::uint64_t kernel_fused_nodes =
        static_cast<std::uint64_t>(total_output_operations) -
        kernel_split_nodes;

    __nv_bfloat16* input{};
    __nv_bfloat16* output{};
    unsigned long long* replay_marker{};
    unsigned long long* mismatches{};
    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaMalloc(
                   &input, input_elements * sizeof(__nv_bfloat16)),
               "cudaMalloc input");
    check_cuda(cudaMalloc(
                   &output, output_elements * sizeof(__nv_bfloat16)),
               "cudaMalloc output");
    check_cuda(cudaMalloc(&replay_marker, sizeof(*replay_marker)),
               "cudaMalloc replay marker");
    check_cuda(cudaMalloc(&mismatches, sizeof(*mismatches)),
               "cudaMalloc mismatches");
    check_cuda(cudaMemset(replay_marker, 0, sizeof(*replay_marker)),
               "cudaMemset replay marker");
    check_cuda(cudaMemset(mismatches, 0, sizeof(*mismatches)),
               "cudaMemset mismatches");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    unsigned long long host_mismatches{};
    {
      spark_transport::Tp4AllreduceSession session(options.transport);
      std::vector<CapturedGraph> graphs;
      if (options.mixed_q_validation) {
        const std::size_t operations_per_cycle =
            graph_a_q.size() + graph_b_q.size();
        graphs.push_back(capture_mixed_q_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.transport.rank, graph_a_q, 0,
            operations_per_cycle, options.elements));
        graphs.push_back(capture_mixed_q_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.transport.rank, graph_b_q, graph_a_q.size(),
            operations_per_cycle, options.elements));
      } else if (options.multi_graph_validation) {
        graphs.push_back(capture_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.graph_a_operations, 0, true));
        graphs.push_back(capture_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.graph_b_operations,
            static_cast<std::size_t>(options.graph_a_operations) *
                kElements,
            true));
      } else if (options.fixed_q == 1 && options.elements == kElements) {
        graphs.push_back(capture_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.operations_per_graph, 0,
            options.timing_mode == TimingMode::kBurst));
      } else {
        graphs.push_back(capture_fixed_q_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.fixed_q, options.elements,
            options.timing_mode == TimingMode::kBurst));
      }

      const std::uint64_t expected_captured_nodes =
          static_cast<std::uint64_t>(
              options.multi_graph_validation
                  ? options.graph_a_operations +
                        options.graph_b_operations
                  : options.operations_per_graph);
      const auto pre_replay_status = session.graph_replay_status();
      const bool pre_replay_capture_valid =
          pre_replay_status.captured_nodes == expected_captured_nodes &&
          pre_replay_status.published_sequence == 0 &&
          pre_replay_status.consumed_sequence == 0 &&
          pre_replay_status.completed_sequence == 0 &&
          pre_replay_status.overflow_sequence == 0;
      if (!pre_replay_capture_valid) {
        throw std::runtime_error(
            "captured graph inventory changed before first replay");
      }

      std::uint64_t replay{};
      std::uint64_t expected_sequence{};
      bool monotonic_sequences = true;
      bool post_replay_capture_rejected{};

      const bool isolated_timing =
          options.timing_mode == TimingMode::kIsolated;
      const auto launch_graph = [&](const CapturedGraph& graph,
                                    const char* phase,
                                    bool collect_isolated_sample) {
        if (replay == std::numeric_limits<std::uint64_t>::max()) {
          throw std::overflow_error("graph replay marker exhausted");
        }
        ++replay;
        prepare_replay<<<input_blocks, threads, 0, stream>>>(
            input, options.transport.rank, replay, replay_marker,
            input_elements);
        check_cuda(cudaGetLastError(), "prepare_replay launch");
        if (collect_isolated_sample) {
          check_cuda(cudaEventRecord(start, stream),
                     "cudaEventRecord isolated start");
        }
        const auto submit_start = std::chrono::steady_clock::now();
        check_cuda(cudaGraphLaunch(graph.executable, stream), phase);
        const auto submit_stop = std::chrono::steady_clock::now();
        if (collect_isolated_sample) {
          check_cuda(cudaEventRecord(stop, stream),
                     "cudaEventRecord isolated stop");
        }
        expected_sequence +=
            static_cast<std::uint64_t>(graph.operations);

        if (isolated_timing) {
          const std::size_t validation_elements =
              static_cast<std::size_t>(options.fixed_q) * options.elements;
          const int validation_blocks = static_cast<int>(
              (validation_elements + threads - 1) / threads);
          validate_active_output<<<validation_blocks, threads, 0, stream>>>(
              output + graph.output_offset, validation_elements,
              replay_marker, mismatches);
          check_cuda(cudaGetLastError(),
                     "validate_q1_output isolated launch");
        }

        if (options.multi_graph_validation) {
          check_cuda(cudaStreamSynchronize(stream),
                     "multi-graph replay synchronize");
          const auto status =
              wait_for_graph_completion(session, expected_sequence);
          const bool exact =
              status.captured_nodes == expected_captured_nodes &&
              status.published_sequence == expected_sequence &&
              status.consumed_sequence == expected_sequence &&
              status.completed_sequence == expected_sequence &&
              status.overflow_sequence == 0;
          monotonic_sequences = monotonic_sequences && exact;
          if (!exact) {
            throw std::runtime_error(
                "multi-graph replay sequence did not advance exactly");
          }
          if (replay == 1) {
            post_replay_capture_rejected =
                attempt_post_replay_capture(
                    session, input, output, stream,
                    options.mixed_q_validation
                        ? options.maximum_q
                        : 1U);
          }
        }
        return std::chrono::duration<double, std::micro>(
                   submit_stop - submit_start)
            .count();
      };

      for (int iteration = 0; iteration < options.warmup; ++iteration) {
        for (const auto& graph : graphs) {
          (void)launch_graph(graph, "cudaGraphLaunch warmup", false);
        }
        if (isolated_timing) {
          check_cuda(cudaStreamSynchronize(stream),
                     "isolated warmup synchronize");
          (void)wait_for_graph_completion(session, expected_sequence);
        }
      }
      check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

      double host_submit_us{};
      double device_us{};
      double device_us_per_collective{};
      double device_us_min{};
      double device_us_p50{};
      double device_us_p95{};
      if (isolated_timing) {
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(options.iterations));
        double submit_us_total{};
        for (int iteration = 0; iteration < options.iterations;
             ++iteration) {
          submit_us_total += launch_graph(
              graphs.front(), "cudaGraphLaunch isolated", true);
          check_cuda(cudaEventSynchronize(stop),
                     "cudaEventSynchronize isolated stop");
          float sample_ms{};
          check_cuda(cudaEventElapsedTime(&sample_ms, start, stop),
                     "cudaEventElapsedTime isolated");
          samples.push_back(static_cast<double>(sample_ms) * 1000.0);
          check_cuda(cudaStreamSynchronize(stream),
                     "isolated validation synchronize");
        }
        std::sort(samples.begin(), samples.end());
        const auto nearest_rank = [&](double quantile) {
          const std::size_t rank = static_cast<std::size_t>(
              std::ceil(quantile * static_cast<double>(samples.size())));
          return samples.at(std::max<std::size_t>(1, rank) - 1);
        };
        host_submit_us = submit_us_total / options.iterations;
        device_us_min = samples.front();
        device_us_p50 = nearest_rank(0.50);
        device_us_p95 = nearest_rank(0.95);
        device_us = device_us_p50;
        device_us_per_collective = device_us_p50;
      } else {
        check_cuda(cudaEventRecord(start, stream),
                   "cudaEventRecord start");
        const auto host_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < options.iterations;
             ++iteration) {
          for (const auto& graph : graphs) {
            (void)launch_graph(
                graph, "cudaGraphLaunch measured", false);
          }
        }
        const auto host_stop = std::chrono::steady_clock::now();
        check_cuda(cudaEventRecord(stop, stream),
                   "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop),
                   "cudaEventSynchronize stop");
        float elapsed_ms{};
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "cudaEventElapsedTime");
        host_submit_us =
            std::chrono::duration<double, std::micro>(
                host_stop - host_start)
                .count() /
            options.iterations;
        device_us =
            static_cast<double>(elapsed_ms) * 1000.0 /
            options.iterations;
        const int operations_per_iteration =
            options.multi_graph_validation
                ? options.graph_a_operations +
                      options.graph_b_operations
                : options.operations_per_graph;
        device_us_per_collective =
            device_us / operations_per_iteration;
      }
      check_cuda(cudaMemcpy(&host_mismatches, mismatches,
                            sizeof(host_mismatches),
                            cudaMemcpyDeviceToHost),
                 "copy mismatch counter");

      const auto status =
          wait_for_graph_completion(session, expected_sequence);
      const bool sequences_match =
          status.captured_nodes == expected_captured_nodes &&
          status.published_sequence == expected_sequence &&
          status.consumed_sequence == expected_sequence &&
          status.completed_sequence == expected_sequence &&
          status.overflow_sequence == 0;
      const bool protocol_status_match =
          status.two_slot_deferred_ack ==
          spark_transport::tp4_protocol_uses_deferred_ack(
              options.transport.protocol);
      const std::size_t payload_slots =
          spark_transport::tp4_payload_slot_count(
              options.transport.protocol);
      const bool slot_reuse_exercised =
          spark_transport::tp4_protocol_uses_deferred_ack(
              options.transport.protocol) &&
          status.completed_sequence >= 3;
      const bool submit_fast_enough =
          options.max_graph_submit_us == 0.0 ||
          host_submit_us <= options.max_graph_submit_us;
      const bool device_fast_enough =
          options.max_device_us == 0.0 ||
          (isolated_timing ? device_us_p95
                           : device_us_per_collective) <=
              options.max_device_us;
      const bool correct =
          host_mismatches == 0 && sequences_match &&
          protocol_status_match && pre_replay_capture_valid &&
          monotonic_sequences &&
          (!options.multi_graph_validation ||
           post_replay_capture_rejected);
      const bool passed =
          correct && submit_fast_enough && device_fast_enough;
      const std::uint64_t graph_cycles =
          static_cast<std::uint64_t>(options.warmup) +
          static_cast<std::uint64_t>(options.iterations);
      const std::uint64_t validated_active_bytes_total =
          active_bytes_per_graph_cycle * graph_cycles;
      const char* const transport_kernel_path =
          options.transport.schedule ==
                  spark_transport::Tp4AllreduceSchedule::kDualPortStriped
              ? "dual_port_striped_dag"
              : options.transport.graph_kernel_strategy ==
                        spark_transport::Tp4GraphKernelStrategy::kFused
                    ? "sequential_fused"
                    : options.transport.graph_kernel_strategy ==
                              spark_transport::
                                  Tp4GraphKernelStrategy::kSplit64KiB
                          ? "sequential_split_64k"
                          : "sequential_tiered_64k";

      std::cout << "TP4_GRAPH_Q1"
                << " rank=" << options.transport.rank
                << " publisher=device"
                << " ring_capacity="
                << spark_transport::kTp4GraphCommandCapacity
                << " mode="
                << (options.multi_graph_validation ? "multi" : "single")
                << " mixed_q="
                << (options.mixed_q_validation ? "true" : "false")
                << " fixed_q=" << options.fixed_q
                << " maximum_q=" << options.maximum_q
                << " session_capacity_bytes="
                << options.transport.payload_bytes
                << " allreduce_protocol="
                << spark_transport::tp4_allreduce_protocol_name(
                       options.transport.protocol)
                << " wire_schedule="
                << spark_transport::tp4_allreduce_schedule_name(
                       options.transport.schedule)
                << " transport_kernel_path=" << transport_kernel_path
                << " graph_kernel="
                << spark_transport::tp4_graph_kernel_strategy_name(
                       options.transport.graph_kernel_strategy)
                << " kernel_fused_nodes=" << kernel_fused_nodes
                << " kernel_split_64k_nodes=" << kernel_split_nodes
                << " payload_slots=" << payload_slots
                << " slot_reuse_exercised="
                << (slot_reuse_exercised ? "true" : "false")
                << " timing_mode="
                << (isolated_timing ? "isolated" : "burst")
                << " iterations=" << options.iterations
                << " operations_per_graph="
                << options.operations_per_graph
                << " graph_a_operations="
                << (options.multi_graph_validation
                        ? options.graph_a_operations
                        : options.operations_per_graph)
                << " graph_b_operations="
                << (options.multi_graph_validation
                        ? options.graph_b_operations
                        : 0)
                << " q1_nodes=" << q_histogram[0]
                << " q2_nodes=" << q_histogram[1]
                << " q3_nodes=" << q_histogram[2]
                << " q4_nodes=" << q_histogram[3]
                << " q5_nodes=" << q_histogram[4]
                << " q6_nodes=" << q_histogram[5]
                << " q40_nodes=" << q_histogram[39]
                << " q48_nodes=" << q_histogram[47]
                << " q72_nodes=" << q_histogram[71]
                << " q144_nodes=" << q_histogram[143]
                << " q512_nodes=" << q_histogram[511]
                << " active_bytes_per_graph_cycle="
                << active_bytes_per_graph_cycle
                << " validated_active_bytes_total="
                << validated_active_bytes_total
                << " graph_launches=" << replay
                << " input_updates=" << replay
                << " captured_nodes=" << status.captured_nodes
                << " submit_affinity_verified="
                << (status.submit_affinity_verified ? "true" : "false")
                << " progress_affinity_verified="
                << (status.progress_affinity_verified ? "true" : "false")
                << " protocol_status_match="
                << (protocol_status_match ? "true" : "false")
                << " graph_submit_cpu=" << status.graph_submit_cpu
                << " graph_progress_cpu=" << status.graph_progress_cpu
                << " pre_replay_capture_valid="
                << (pre_replay_capture_valid ? "true" : "false")
                << " graph_submit_us_per_call=" << host_submit_us;
      if (isolated_timing) {
        std::cout
            << " timing_scope=device_output_ready_single_replay"
            << " timing_samples=" << options.iterations
            << " device_output_ready_us_per_graph_min="
            << device_us_min
            << " device_output_ready_us_per_graph_p50="
            << device_us_p50
            << " device_output_ready_us_per_graph_p95="
            << device_us_p95
            << " device_gate_metric="
               "p95_device_output_ready_us_per_graph";
      } else {
        std::cout
            << " timing_scope=device_output_ready_replay_throughput"
            << " device_output_ready_us_per_graph=" << device_us
            << " device_output_ready_us_per_collective="
            << device_us_per_collective
            << " device_gate_metric="
               "mean_device_output_ready_us_per_collective";
      }
      std::cout << " published=" << status.published_sequence
                << " consumed=" << status.consumed_sequence
                << " completed=" << status.completed_sequence
                << " overflow=" << status.overflow_sequence
                << " mismatched_elements=" << host_mismatches
                << " monotonic_sequences="
                << (monotonic_sequences ? "true" : "false")
                << " post_replay_capture_rejected="
                << (post_replay_capture_rejected ? "true" : "false")
                << " submit_gate="
                << (submit_fast_enough ? "pass" : "fail")
                << " device_gate="
                << (device_fast_enough ? "pass" : "fail")
                << " correct=" << (correct ? "true" : "false")
                << " passed=" << (passed ? "true" : "false") << '\n';

      for (const auto& graph : graphs) {
        check_cuda(cudaGraphExecDestroy(graph.executable),
                   "cudaGraphExecDestroy");
      }
      if (!passed) {
        throw std::runtime_error("Q1 graph replay validation failed");
      }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(mismatches);
    cudaFree(replay_marker);
    cudaFree(output);
    cudaFree(input);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
