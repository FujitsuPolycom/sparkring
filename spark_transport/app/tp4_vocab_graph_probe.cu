#include "spark_transport/gpu_tp4_vocab_allgather.hpp"
#include "spark_transport/tp4_vocab_allgather_c_api.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

constexpr std::uint32_t kMtp4Pattern[] = {5, 1, 1, 1, 1};
constexpr std::uint32_t kMtp5Pattern[] = {6, 1, 1, 1, 1, 1};
constexpr std::size_t kMaximumPatternNodes =
    sizeof(kMtp5Pattern) / sizeof(kMtp5Pattern[0]);

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{10110};
  std::uint16_t control_port1{10111};
  std::uint32_t submit_cpu{10};
  std::uint32_t progress_cpu{12};
  int warmup{2};
  int iterations{100};
  std::uint32_t mtp_tokens{4};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n"
      << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
      << "         --control-port0 PORT --control-port1 PORT\n"
      << "         --submit-cpu CPU --progress-cpu CPU\n"
      << "         --warmup N --iterations N --mtp-tokens 4|5\n";
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
    const auto take_value = [&]() -> const char* {
      if (++index >= argc) {
        usage(argv[0]);
      }
      return argv[index];
    };
    if (argument == "--rank") {
      options.rank = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "rank"));
    } else if (argument == "--peer0") {
      options.peer0 = take_value();
    } else if (argument == "--peer1") {
      options.peer1 = take_value();
    } else if (argument == "--device0") {
      options.device0 = take_value();
    } else if (argument == "--device1") {
      options.device1 = take_value();
    } else if (argument == "--gid0") {
      options.gid0 = static_cast<std::uint8_t>(
          unsigned_value(take_value(), "GID 0"));
    } else if (argument == "--gid1") {
      options.gid1 = static_cast<std::uint8_t>(
          unsigned_value(take_value(), "GID 1"));
    } else if (argument == "--control-port0") {
      options.control_port0 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 0"));
    } else if (argument == "--control-port1") {
      options.control_port1 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 1"));
    } else if (argument == "--submit-cpu") {
      options.submit_cpu = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "submit CPU"));
    } else if (argument == "--progress-cpu") {
      options.progress_cpu = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "progress CPU"));
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iterations"));
    } else if (argument == "--mtp-tokens") {
      options.mtp_tokens = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "MTP tokens"));
    } else {
      usage(argv[0]);
    }
  }
  if (options.rank >= spark_transport::kTp4VocabWorldSize ||
      options.peer0.empty() || options.peer1.empty() ||
      options.control_port0 == options.control_port1 ||
      options.submit_cpu == options.progress_cpu ||
      options.warmup < 0 || options.iterations <= 0 ||
      (options.mtp_tokens != 4 && options.mtp_tokens != 5)) {
    usage(argv[0]);
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

__host__ __device__ std::uint16_t expected_word(
    std::uint32_t rank, std::uint32_t query_index,
    std::size_t column, std::uint64_t replay) {
  return static_cast<std::uint16_t>(
      0x1000U + rank * 7919U + query_index * 1223U +
      static_cast<std::uint32_t>(column) * 13U +
      static_cast<std::uint32_t>(replay & 255U) * 17U);
}

__global__ void prepare_input(
    std::uint16_t* input, std::uint32_t rank, std::uint64_t replay,
    std::uint64_t* replay_marker, std::size_t elements) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *replay_marker = replay;
  }
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint32_t query_index = static_cast<std::uint32_t>(
        index / spark_transport::kTp4VocabShardElements);
    const std::size_t column =
        index % spark_transport::kTp4VocabShardElements;
    input[index] =
        expected_word(rank, query_index, column, replay);
  }
}

__global__ void validate_output(
  const std::uint16_t* output, std::uint32_t query_rows,
    const std::uint64_t* replay_marker,
    unsigned long long* mismatches) {
  const std::size_t elements =
      static_cast<std::size_t>(query_rows) *
      spark_transport::kTp4VocabWorldSize *
      spark_transport::kTp4VocabShardElements;
  const std::uint64_t replay = *replay_marker;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t elements_per_output_row =
        spark_transport::kTp4VocabWorldSize *
        spark_transport::kTp4VocabShardElements;
    const std::uint32_t query_index = static_cast<std::uint32_t>(
        index / elements_per_output_row);
    const std::size_t in_row = index % elements_per_output_row;
    const std::uint32_t source_rank = static_cast<std::uint32_t>(
        in_row / spark_transport::kTp4VocabShardElements);
    const std::size_t column =
        in_row % spark_transport::kTp4VocabShardElements;
    if (output[index] !=
        expected_word(source_rank, query_index, column, replay)) {
      atomicAdd(mismatches, 1ULL);
    }
  }
}

class VocabHandle {
 public:
  explicit VocabHandle(spark_tp4_vocab_allgather_handle handle)
      : handle_(handle) {}
  VocabHandle(const VocabHandle&) = delete;
  VocabHandle& operator=(const VocabHandle&) = delete;
  ~VocabHandle() {
    spark_tp4_vocab_allgather_destroy(handle_);
  }
  spark_tp4_vocab_allgather_handle get() const { return handle_; }

 private:
  spark_tp4_vocab_allgather_handle handle_{};
};

spark_tp4_vocab_graph_status graph_status(
    spark_tp4_vocab_allgather_handle handle) {
  spark_tp4_vocab_graph_status status{};
  char error[512]{};
  if (spark_tp4_vocab_get_graph_status(
          handle, &status, sizeof(status), error, sizeof(error)) != 0) {
    throw std::runtime_error(
        std::string("vocabulary graph status: ") + error);
  }
  if (status.struct_size != sizeof(status)) {
    throw std::runtime_error(
        "vocabulary graph status ABI size mismatch");
  }
  return status;
}

spark_tp4_vocab_graph_status wait_for_completion(
    spark_tp4_vocab_allgather_handle handle,
    std::uint64_t expected_sequence) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (true) {
    const auto status = graph_status(handle);
    if (status.overflow_sequence != 0) {
      throw std::runtime_error("vocabulary graph overflow");
    }
    if (status.completed_sequence == expected_sequence) {
      return status;
    }
    if (status.completed_sequence > expected_sequence ||
        std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          "vocabulary graph completion did not advance exactly");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    constexpr int threads = 256;
    constexpr int blocks = 256;
    const std::size_t input_elements =
        spark_transport::tp4_vocab_input_bytes(
            spark_transport::kTp4VocabMaxQueryRows) /
        sizeof(std::uint16_t);
    const std::size_t max_output_bytes =
        spark_transport::tp4_vocab_output_bytes(
            spark_transport::kTp4VocabMaxQueryRows);
    const std::uint32_t* pattern =
        options.mtp_tokens == 4 ? kMtp4Pattern : kMtp5Pattern;
    const std::size_t pattern_nodes =
        static_cast<std::size_t>(options.mtp_tokens) + 1;
    const char* pattern_text =
        options.mtp_tokens == 4 ? "5,1,1,1,1" : "6,1,1,1,1,1";

    std::uint16_t* input{};
    std::uint16_t* outputs[kMaximumPatternNodes]{};
    std::uint64_t* replay_marker{};
    unsigned long long* mismatches{};
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaEvent_t start{};
    cudaEvent_t stop{};

    check_cuda(
        cudaMalloc(&input, input_elements * sizeof(*input)),
        "cudaMalloc vocabulary graph input");
    for (auto& output : outputs) {
      check_cuda(
          cudaMalloc(&output, max_output_bytes),
          "cudaMalloc vocabulary graph output");
    }
    check_cuda(
        cudaMalloc(&replay_marker, sizeof(*replay_marker)),
        "cudaMalloc vocabulary replay marker");
    check_cuda(
        cudaMalloc(&mismatches, sizeof(*mismatches)),
        "cudaMalloc vocabulary mismatch counter");
    check_cuda(
        cudaMemset(mismatches, 0, sizeof(*mismatches)),
        "cudaMemset vocabulary mismatches");
    check_cuda(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate vocabulary graph");
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    unsigned long long host_mismatches{};
    {
      spark_tp4_vocab_graph_config config{};
      config.rank = options.rank;
      config.peer0 = options.peer0.c_str();
      config.peer1 = options.peer1.c_str();
      config.device0 = options.device0.c_str();
      config.device1 = options.device1.c_str();
      config.gid0 = options.gid0;
      config.gid1 = options.gid1;
      config.control_port0 = options.control_port0;
      config.control_port1 = options.control_port1;
      config.graph_submit_cpu_plus_one = options.submit_cpu + 1;
      config.graph_progress_cpu_plus_one = options.progress_cpu + 1;
      char error[512]{};
      const VocabHandle session(
          spark_tp4_vocab_graph_create(
              &config, error, sizeof(error)));
      if (session.get() == nullptr) {
        throw std::runtime_error(
            std::string("create vocabulary graph session: ") + error);
      }

      check_cuda(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
          "cudaStreamBeginCapture vocabulary");
      for (std::size_t node = 0; node < pattern_nodes; ++node) {
        const std::uint32_t q = pattern[node];
        if (spark_tp4_vocab_capture_allgather(
                session.get(), input, outputs[node], q, stream, error,
                sizeof(error)) != 0) {
          throw std::runtime_error(
              std::string("capture vocabulary graph node: ") + error);
        }
        validate_output<<<blocks, threads, 0, stream>>>(
            outputs[node], q, replay_marker, mismatches);
        check_cuda(
            cudaGetLastError(), "capture vocabulary validator");
      }
      check_cuda(
          cudaStreamEndCapture(stream, &graph),
          "cudaStreamEndCapture vocabulary");
      check_cuda(
          cudaGraphInstantiate(&executable, graph, 0),
          "cudaGraphInstantiate vocabulary");

      const auto before = graph_status(session.get());
      const std::uint32_t required_flags =
          SPARK_TP4_VOCAB_GRAPH_CAPTURE_CONFIGURED |
          SPARK_TP4_VOCAB_GRAPH_POLLING_ENABLED |
          SPARK_TP4_VOCAB_GRAPH_HOST_NATIVE_ATOMICS |
          SPARK_TP4_VOCAB_GRAPH_SUBMIT_AFFINITY_VERIFIED |
          SPARK_TP4_VOCAB_GRAPH_PROGRESS_AFFINITY_VERIFIED;
      if (before.captured_nodes != pattern_nodes ||
          before.published_sequence != 0 ||
          before.consumed_sequence != 0 ||
          before.completed_sequence != 0 ||
          before.overflow_sequence != 0 ||
          (before.flags & required_flags) != required_flags ||
          before.graph_submit_cpu_plus_one != options.submit_cpu + 1 ||
          before.graph_progress_cpu_plus_one !=
              options.progress_cpu + 1) {
        throw std::runtime_error(
            "invalid vocabulary graph pre-replay status");
      }

      std::uint64_t replay{};
      const auto launch = [&](const char* operation) {
        if (replay == std::numeric_limits<std::uint64_t>::max()) {
          throw std::overflow_error(
              "vocabulary replay marker exhausted");
        }
        ++replay;
        prepare_input<<<blocks, threads, 0, stream>>>(
            input, options.rank, replay, replay_marker,
            input_elements);
        check_cuda(
            cudaGetLastError(), "prepare vocabulary graph input");
        check_cuda(cudaGraphLaunch(executable, stream), operation);
      };

      for (int iteration = 0; iteration < options.warmup; ++iteration) {
        launch("cudaGraphLaunch vocabulary warmup");
      }
      check_cuda(
          cudaStreamSynchronize(stream),
          "vocabulary graph warmup synchronize");

      check_cuda(cudaEventRecord(start, stream), "record start");
      const auto host_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < options.iterations;
           ++iteration) {
        launch("cudaGraphLaunch vocabulary measured");
      }
      const auto host_stop = std::chrono::steady_clock::now();
      check_cuda(cudaEventRecord(stop, stream), "record stop");
      check_cuda(cudaEventSynchronize(stop), "synchronize stop");

      float elapsed_ms{};
      check_cuda(
          cudaEventElapsedTime(&elapsed_ms, start, stop),
          "vocabulary graph elapsed time");
      check_cuda(
          cudaMemcpy(
              &host_mismatches, mismatches, sizeof(host_mismatches),
              cudaMemcpyDeviceToHost),
          "copy vocabulary mismatch count");

      const std::uint64_t expected_sequence =
          replay * pattern_nodes;
      const auto after =
          wait_for_completion(session.get(), expected_sequence);
      const bool exact =
          after.captured_nodes == pattern_nodes &&
          after.published_sequence == expected_sequence &&
          after.consumed_sequence == expected_sequence &&
          after.completed_sequence == expected_sequence &&
          after.overflow_sequence == 0;
      const double host_submit_us =
          std::chrono::duration<double, std::micro>(
              host_stop - host_start)
              .count() /
          options.iterations;
      const double device_us =
          static_cast<double>(elapsed_ms) * 1000.0 /
          options.iterations;
      const bool passed = exact && host_mismatches == 0;

      std::cout
          << std::fixed << std::setprecision(3)
          << "TP4_VOCAB_GRAPH"
          << " rank=" << options.rank
          << " mtp_tokens=" << options.mtp_tokens
          << " pattern=" << pattern_text
          << " captured_nodes=" << after.captured_nodes
          << " warmup=" << options.warmup
          << " iterations=" << options.iterations
          << " published=" << after.published_sequence
          << " consumed=" << after.consumed_sequence
          << " completed=" << after.completed_sequence
          << " overflow=" << after.overflow_sequence
          << " submit_cpu=" << options.submit_cpu
          << " progress_cpu=" << options.progress_cpu
          << " host_submit_us_per_graph=" << host_submit_us
          << " device_us_per_graph=" << device_us
          << " device_us_per_collective="
          << device_us / pattern_nodes
          << " mismatches=" << host_mismatches
          << " passed=" << (passed ? "true" : "false") << '\n';
      check_cuda(
          cudaGraphExecDestroy(executable),
          "cudaGraphExecDestroy vocabulary");
      executable = nullptr;
      check_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy vocabulary");
      graph = nullptr;
      if (!passed) {
        return 1;
      }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(mismatches);
    cudaFree(replay_marker);
    for (auto* output : outputs) {
      cudaFree(output);
    }
    cudaFree(input);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TP4_VOCAB_GRAPH_ERROR " << error.what() << '\n';
    return 1;
  }
}
