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
constexpr std::size_t kMaximumQ =
    spark_transport::kTp4GraphAllreduceMaximumQ;
constexpr std::size_t kMaximumElements = kMaximumQ * kElements;
constexpr std::size_t kMaximumPayloadBytes =
    kMaximumElements * sizeof(__nv_bfloat16);
static_assert(kMaximumPayloadBytes == 6U * 1024U * 1024U);

struct Options {
  spark_transport::Tp4AllreduceOptions transport;
  int warmup{10};
  int iterations{100};
  int operations_per_graph{1};
  bool multi_graph_validation{};
  bool mixed_q_validation{};
  std::uint32_t maximum_q{6};
  int graph_a_operations{3};
  int graph_b_operations{128};
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
      << "  --maximum-q Q\n"
      << "  --graph-a-operations COUNT\n"
      << "  --graph-b-operations COUNT\n"
      << "  --graph-submit-cpu CPU\n"
      << "  --graph-progress-cpu CPU\n"
      << "  --max-graph-submit-us MICROSECONDS\n"
      << "  --max-device-us MICROSECONDS\n";
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
  if (options.maximum_q < 6 || options.maximum_q > kMaximumQ) {
    throw std::invalid_argument("maximum Q must be in [6, 512]");
  }
  if (options.mixed_q_validation) {
    options.transport.payload_bytes =
        spark_transport::tp4_graph_payload_bytes(options.maximum_q);
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
                                 unsigned long long replay) {
  // Every adjacent replay has different, exactly representable BF16 inputs.
  // The alternating offset catches stale replay data without introducing
  // floating-point comparison tolerance into the correctness gate.
  const unsigned int replay_offset = (replay & 1ULL) == 0 ? 0U : 16U;
  return static_cast<float>(
      ((element * 3U + rank * 5U) & 7U) + 1U + replay_offset);
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

struct CapturedGraph {
  int operations{};
  std::size_t output_offset{};
  cudaGraphExec_t executable{};
};

CapturedGraph capture_graph(
    spark_transport::Tp4AllreduceSession& session,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned long long* replay_marker, unsigned long long* mismatches,
    cudaStream_t stream, int operations, std::size_t output_offset) {
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
  validate_q1_output<<<validation_blocks, threads, 0, stream>>>(
      output + output_offset, output_elements, replay_marker, mismatches);
  check_cuda(cudaGetLastError(), "validate_q1_output capture launch");
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

CapturedGraph capture_mixed_q_graph(
    spark_transport::Tp4AllreduceSession& session,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned long long* replay_marker, unsigned long long* mismatches,
    cudaStream_t stream, const std::vector<std::uint32_t>& q_values) {
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
    session.capture_all_reduce(input, operation_output, q, stream);
    const std::size_t active_elements =
        static_cast<std::size_t>(q) * kElements;
    const int validation_blocks = static_cast<int>(
        (active_elements + threads - 1) / threads);
    validate_active_output<<<validation_blocks, threads, 0, stream>>>(
        operation_output, active_elements, replay_marker, mismatches);
    check_cuda(cudaGetLastError(),
               "validate_active_output capture launch");
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
            ? static_cast<std::size_t>(options.maximum_q) * kElements
            : kElements;
    const int input_blocks = static_cast<int>(
        (input_elements + threads - 1) / threads);
    const std::size_t output_stride_elements =
        options.mixed_q_validation
            ? static_cast<std::size_t>(options.maximum_q) * kElements
            : kElements;
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
          spark_transport::tp4_graph_payload_bytes(q);
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
      q_histogram[0] = total_output_operations;
      active_bytes_per_graph_cycle =
          total_output_operations * kPayloadBytes;
    }

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
        graphs.push_back(capture_mixed_q_graph(
            session, input, output, replay_marker, mismatches, stream,
            graph_a_q));
        graphs.push_back(capture_mixed_q_graph(
            session, input, output, replay_marker, mismatches, stream,
            graph_b_q));
      } else if (options.multi_graph_validation) {
        graphs.push_back(capture_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.graph_a_operations, 0));
        graphs.push_back(capture_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.graph_b_operations,
            static_cast<std::size_t>(options.graph_a_operations) *
                kElements));
      } else {
        graphs.push_back(capture_graph(
            session, input, output, replay_marker, mismatches, stream,
            options.operations_per_graph, 0));
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

      const auto launch_graph = [&](const CapturedGraph& graph,
                                    const char* phase) {
        if (replay == std::numeric_limits<std::uint64_t>::max()) {
          throw std::overflow_error("graph replay marker exhausted");
        }
        ++replay;
        prepare_replay<<<input_blocks, threads, 0, stream>>>(
            input, options.transport.rank, replay, replay_marker,
            input_elements);
        check_cuda(cudaGetLastError(), "prepare_replay launch");
        check_cuda(cudaGraphLaunch(graph.executable, stream), phase);
        expected_sequence +=
            static_cast<std::uint64_t>(graph.operations);

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
      };

      for (int iteration = 0; iteration < options.warmup; ++iteration) {
        for (const auto& graph : graphs) {
          launch_graph(graph, "cudaGraphLaunch warmup");
        }
      }
      check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

      check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");
      const auto host_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        for (const auto& graph : graphs) {
          launch_graph(graph, "cudaGraphLaunch measured");
        }
      }
      const auto host_stop = std::chrono::steady_clock::now();
      check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
      check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

      float elapsed_ms{};
      check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                 "cudaEventElapsedTime");
      check_cuda(cudaMemcpy(&host_mismatches, mismatches,
                            sizeof(host_mismatches),
                            cudaMemcpyDeviceToHost),
                 "copy mismatch counter");

      const auto status =
          wait_for_graph_completion(session, expected_sequence);
      const double host_submit_us =
          std::chrono::duration<double, std::micro>(host_stop - host_start)
              .count() /
          options.iterations;
      const double device_us =
          static_cast<double>(elapsed_ms) * 1000.0 / options.iterations;
      const int operations_per_iteration =
          options.multi_graph_validation
              ? options.graph_a_operations + options.graph_b_operations
              : options.operations_per_graph;
      const double device_us_per_collective =
          device_us / operations_per_iteration;
      const bool sequences_match =
          status.captured_nodes == expected_captured_nodes &&
          status.published_sequence == expected_sequence &&
          status.consumed_sequence == expected_sequence &&
          status.completed_sequence == expected_sequence &&
          status.overflow_sequence == 0;
      const bool submit_fast_enough =
          options.max_graph_submit_us == 0.0 ||
          host_submit_us <= options.max_graph_submit_us;
      const bool device_fast_enough =
          options.max_device_us == 0.0 ||
          device_us_per_collective <= options.max_device_us;
      const bool correct =
          host_mismatches == 0 && sequences_match &&
          pre_replay_capture_valid && monotonic_sequences &&
          (!options.multi_graph_validation ||
           post_replay_capture_rejected);
      const bool passed =
          correct && submit_fast_enough && device_fast_enough;
      const std::uint64_t graph_cycles =
          static_cast<std::uint64_t>(options.warmup) +
          static_cast<std::uint64_t>(options.iterations);
      const std::uint64_t validated_active_bytes_total =
          active_bytes_per_graph_cycle * graph_cycles;

      std::cout << "TP4_GRAPH_Q1"
                << " rank=" << options.transport.rank
                << " publisher=device"
                << " ring_capacity="
                << spark_transport::kTp4GraphCommandCapacity
                << " mode="
                << (options.multi_graph_validation ? "multi" : "single")
                << " mixed_q="
                << (options.mixed_q_validation ? "true" : "false")
                << " maximum_q=" << options.maximum_q
                << " session_capacity_bytes="
                << options.transport.payload_bytes
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
                << " graph_submit_cpu=" << status.graph_submit_cpu
                << " graph_progress_cpu=" << status.graph_progress_cpu
                << " pre_replay_capture_valid="
                << (pre_replay_capture_valid ? "true" : "false")
                << " graph_submit_us_per_call=" << host_submit_us
                << " device_us_per_graph=" << device_us
                << " device_us_per_call=" << device_us_per_collective
                << " published=" << status.published_sequence
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
