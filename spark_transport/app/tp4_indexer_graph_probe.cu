#include "spark_transport/tp4_graph_command.hpp"
#include "spark_transport/tp4_indexer_graph.hpp"
#include "spark_transport/tp4_indexer_graph_session.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

constexpr std::array<std::uint32_t, 3> kQPattern{1U, 23U, 40U};
constexpr std::uint64_t kExpectedCapturedQMask =
    (std::uint64_t{1} << (kQPattern[0] - 1)) |
    (std::uint64_t{1} << (kQPattern[1] - 1)) |
    (std::uint64_t{1} << (kQPattern[2] - 1));
constexpr std::uint64_t kRequiredRingWraps = 2;
constexpr std::uint64_t kOutputElementsPerCycle =
    spark_transport::kTp4IndexerGraphWorldSize *
    (kQPattern[0] + kQPattern[1] + kQPattern[2]) *
    spark_transport::kTp4IndexerGraphElementsPerRow;
constexpr std::uint64_t kOutputBytesPerCycle =
    kOutputElementsPerCycle * sizeof(std::int32_t);
static_assert(kExpectedCapturedQMask == 549760008193ULL);
static_assert(kOutputBytesPerCycle == 4194304ULL);

struct Options {
  spark_transport::Tp4IndexerGraphOptions transport;
  std::uint32_t cycles{100};
  bool destructive_mismatch_q{};
  bool destructive_confirmation{};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n\n"
      << "Normal mode captures separate Q1/Q23/Q40 graph executables and\n"
      << "alternates them for at least two command-ring wraps.\n\n"
      << "Options:\n"
      << "  --device0 HCA\n"
      << "  --device1 HCA\n"
      << "  --gid0 INDEX\n"
      << "  --gid1 INDEX\n"
      << "  --control-port0 PORT\n"
      << "  --control-port1 PORT\n"
      << "  --submit-cpu CPU\n"
      << "  --progress-cpu CPU\n"
      << "  --cycles COUNT\n"
      << "  --destructive-mismatch-q\n"
      << "  --i-understand-mismatch-may-abort\n";
  std::exit(2);
}

std::uint64_t unsigned_value(const char* value, const char* name) {
  std::size_t consumed{};
  const std::string text(value);
  const std::uint64_t parsed = std::stoull(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  options.transport.graph_submit_cpu = 10;
  options.transport.graph_progress_cpu = 14;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto take_value = [&]() -> const char* {
      if (++index >= argc) {
        usage(argv[0]);
      }
      return argv[index];
    };

    if (argument == "--rank") {
      options.transport.rank = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "rank"));
    } else if (argument == "--peer0") {
      options.transport.peer0 = take_value();
    } else if (argument == "--peer1") {
      options.transport.peer1 = take_value();
    } else if (argument == "--device0") {
      options.transport.device0 = take_value();
    } else if (argument == "--device1") {
      options.transport.device1 = take_value();
    } else if (argument == "--gid0") {
      options.transport.gid0 = static_cast<std::uint8_t>(
          unsigned_value(take_value(), "GID 0"));
    } else if (argument == "--gid1") {
      options.transport.gid1 = static_cast<std::uint8_t>(
          unsigned_value(take_value(), "GID 1"));
    } else if (argument == "--control-port0") {
      options.transport.control_port0 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 0"));
    } else if (argument == "--control-port1") {
      options.transport.control_port1 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 1"));
    } else if (argument == "--submit-cpu") {
      options.transport.graph_submit_cpu =
          static_cast<std::uint32_t>(
              unsigned_value(take_value(), "submit CPU"));
    } else if (argument == "--progress-cpu") {
      options.transport.graph_progress_cpu =
          static_cast<std::uint32_t>(
              unsigned_value(take_value(), "progress CPU"));
    } else if (argument == "--cycles") {
      options.cycles = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "cycle count"));
    } else if (argument == "--destructive-mismatch-q") {
      options.destructive_mismatch_q = true;
    } else if (argument ==
               "--i-understand-mismatch-may-abort") {
      options.destructive_confirmation = true;
    } else {
      usage(argv[0]);
    }
  }

  if (options.transport.rank >=
          spark_transport::kTp4IndexerGraphWorldSize ||
      options.transport.peer0.empty() ||
      options.transport.peer1.empty() ||
      options.transport.control_port0 == 0 ||
      options.transport.control_port1 == 0 ||
      options.transport.control_port0 ==
          options.transport.control_port1 ||
      !options.transport.graph_submit_cpu.has_value() ||
      !options.transport.graph_progress_cpu.has_value() ||
      options.transport.graph_submit_cpu ==
          options.transport.graph_progress_cpu ||
      options.cycles == 0 || options.cycles > 1000000) {
    usage(argv[0]);
  }
  if (options.destructive_mismatch_q !=
      options.destructive_confirmation) {
    throw std::invalid_argument(
        "destructive mismatch-Q mode requires both explicit switches");
  }
  const std::uint64_t launches =
      static_cast<std::uint64_t>(options.cycles) *
      kQPattern.size();
  if (!options.destructive_mismatch_q &&
      launches <
          kRequiredRingWraps *
              spark_transport::kTp4GraphCommandCapacity) {
    throw std::invalid_argument(
        "normal probe requires enough cycles for two ring wraps");
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " +
        cudaGetErrorString(result));
  }
}

__device__ std::int32_t input_value(
    std::uint32_t rank, std::size_t element,
    unsigned long long replay) {
  // The positive INT32 hash is byte-stable and changes on every adjacent
  // replay, which detects stale input/output while preserving exact compare.
  const std::uint32_t value =
      (rank + 1U) * 2654435761U ^
      static_cast<std::uint32_t>(replay) * 2246822519U ^
      static_cast<std::uint32_t>(element) * 3266489917U;
  return static_cast<std::int32_t>(value & 0x7fffffffU);
}

__global__ void prepare_input(
    std::int32_t* input, std::size_t input_elements,
    std::uint32_t rank, unsigned long long replay,
    unsigned long long* replay_marker) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) +
      threadIdx.x;
  if (index < input_elements) {
    input[index] = input_value(rank, index, replay);
  }
  if (index == 0) {
    *replay_marker = replay;
  }
}

__global__ void validate_rank_ordered_output(
    const std::int32_t* output, std::size_t input_elements,
    const unsigned long long* replay_marker,
    unsigned long long* mismatches) {
  const std::size_t output_elements =
      spark_transport::kTp4IndexerGraphWorldSize *
      input_elements;
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) +
      threadIdx.x;
  if (index >= output_elements) {
    return;
  }
  const std::uint32_t source_rank =
      static_cast<std::uint32_t>(index / input_elements);
  const std::size_t source_element = index % input_elements;
  const std::int32_t expected = input_value(
      source_rank, source_element, *replay_marker);
  if (output[index] != expected) {
    atomicAdd(mismatches, 1ULL);
  }
}

struct CapturedGraph {
  std::uint32_t q{};
  cudaGraphExec_t executable{};
};

CapturedGraph capture_graph(
    spark_transport::Tp4IndexerGraphSession& session,
    const std::int32_t* input, std::int32_t* output,
    const unsigned long long* replay_marker,
    unsigned long long* mismatches, cudaStream_t stream,
    std::uint32_t q) {
  constexpr int threads = 256;
  const std::size_t input_elements =
      static_cast<std::size_t>(q) *
      spark_transport::kTp4IndexerGraphElementsPerRow;
  const std::size_t output_elements =
      spark_transport::kTp4IndexerGraphWorldSize *
      input_elements;
  const int blocks = static_cast<int>(
      (output_elements + threads - 1) / threads);

  cudaGraph_t graph{};
  check_cuda(
      cudaStreamBeginCapture(
          stream, cudaStreamCaptureModeGlobal),
      "cudaStreamBeginCapture indexer");
  session.capture_all_gather(input, output, q, stream);
  validate_rank_ordered_output<<<blocks, threads, 0, stream>>>(
      output, input_elements, replay_marker, mismatches);
  check_cuda(
      cudaGetLastError(),
      "validate_rank_ordered_output capture launch");
  check_cuda(
      cudaStreamEndCapture(stream, &graph),
      "cudaStreamEndCapture indexer");

  CapturedGraph captured{q, nullptr};
  try {
    check_cuda(
        cudaGraphInstantiate(&captured.executable, graph, 0),
        "cudaGraphInstantiate indexer");
  } catch (...) {
    cudaGraphDestroy(graph);
    throw;
  }
  check_cuda(
      cudaGraphDestroy(graph), "cudaGraphDestroy indexer");
  return captured;
}

bool exact_status(
    const spark_transport::Tp4IndexerGraphReplayStatus& status,
    std::uint64_t expected_sequence,
    std::uint32_t submit_cpu, std::uint32_t progress_cpu) {
  return status.captured_nodes == kQPattern.size() &&
         status.captured_q_mask == kExpectedCapturedQMask &&
         status.published_sequence == expected_sequence &&
         status.consumed_sequence == expected_sequence &&
         status.completed_sequence == expected_sequence &&
         status.overflow_sequence == 0 &&
         status.capture_configured && status.polling_enabled &&
         status.host_native_atomics_supported &&
         status.submit_affinity_verified &&
         status.progress_affinity_verified &&
         status.graph_submit_cpu ==
             static_cast<int>(submit_cpu) &&
         status.graph_progress_cpu ==
             static_cast<int>(progress_cpu);
}

spark_transport::Tp4IndexerGraphReplayStatus
wait_for_graph_completion(
    const spark_transport::Tp4IndexerGraphSession& session,
    std::uint64_t expected) {
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::seconds(8);
  while (true) {
    const auto status = session.graph_replay_status();
    if (status.overflow_sequence != 0 ||
        status.completed_sequence >= expected) {
      return status;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          "timed out waiting for indexer graph completion");
    }
    std::this_thread::yield();
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    constexpr int threads = 256;
    constexpr std::size_t maximum_input_elements =
        spark_transport::kTp4IndexerGraphMaximumQ *
        spark_transport::kTp4IndexerGraphElementsPerRow;
    constexpr std::size_t maximum_output_elements =
        spark_transport::kTp4IndexerGraphWorldSize *
        maximum_input_elements;

    std::int32_t* input{};
    std::int32_t* output{};
    unsigned long long* replay_marker{};
    unsigned long long* mismatches{};
    cudaStream_t stream{};
    check_cuda(
        cudaMalloc(&input, maximum_input_elements * sizeof(*input)),
        "cudaMalloc indexer input");
    check_cuda(
        cudaMalloc(&output, maximum_output_elements * sizeof(*output)),
        "cudaMalloc indexer output");
    check_cuda(
        cudaMalloc(&replay_marker, sizeof(*replay_marker)),
        "cudaMalloc indexer replay marker");
    check_cuda(
        cudaMalloc(&mismatches, sizeof(*mismatches)),
        "cudaMalloc indexer mismatch counter");
    check_cuda(
        cudaMemset(replay_marker, 0, sizeof(*replay_marker)),
        "cudaMemset indexer replay marker");
    check_cuda(
        cudaMemset(mismatches, 0, sizeof(*mismatches)),
        "cudaMemset indexer mismatch counter");
    check_cuda(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags indexer");

    unsigned long long host_mismatches{};
    std::uint64_t expected_sequence{};
    std::uint64_t validated_bytes{};
    bool monotonic_sequences = true;
    bool destructive_returned = false;
    spark_transport::Tp4IndexerGraphReplayStatus final_status{};
    {
      spark_transport::Tp4IndexerGraphSession session(
          options.transport);
      std::array<CapturedGraph, kQPattern.size()> graphs{};
      for (std::size_t index = 0;
           index < kQPattern.size(); ++index) {
        graphs[index] = capture_graph(
            session, input, output, replay_marker, mismatches,
            stream, kQPattern[index]);
      }

      const std::uint32_t submit_cpu =
          *options.transport.graph_submit_cpu;
      const std::uint32_t progress_cpu =
          *options.transport.graph_progress_cpu;
      const auto pre_replay_status =
          session.graph_replay_status();
      if (!exact_status(
              pre_replay_status, 0, submit_cpu,
              progress_cpu)) {
        throw std::runtime_error(
            "indexer graph capture census/status is not exact");
      }

      if (options.destructive_mismatch_q) {
        // Even ranks launch Q1 while odd ranks launch Q23. This intentionally
        // violates the four-rank collective order. The progress engine must
        // abort on its bounded protocol timeout; returning here is failure.
        const std::size_t selected_index =
            options.transport.rank % 2 == 0 ? 0 : 1;
        const CapturedGraph& graph = graphs[selected_index];
        std::cout
            << "TP4_INDEXER_GRAPH_MISMATCH"
            << " rank=" << options.transport.rank
            << " armed=true destructive=true"
            << " confirmation=true local_q=" << graph.q
            << " expected_outcome=bounded_transport_abort"
            << std::endl;
        const std::size_t active_elements =
            static_cast<std::size_t>(graph.q) *
            spark_transport::kTp4IndexerGraphElementsPerRow;
        prepare_input<<<
            static_cast<int>(
                (active_elements + threads - 1) / threads),
            threads, 0, stream>>>(
            input, active_elements, options.transport.rank, 1,
            replay_marker);
        check_cuda(
            cudaGetLastError(),
            "prepare destructive mismatch-Q input");
        check_cuda(
            cudaGraphLaunch(graph.executable, stream),
            "cudaGraphLaunch destructive mismatch-Q");
        check_cuda(
            cudaStreamSynchronize(stream),
            "destructive mismatch-Q synchronize");
        destructive_returned = true;
        std::cerr
            << "TP4_INDEXER_GRAPH_MISMATCH_UNEXPECTED"
            << " rank=" << options.transport.rank
            << " transport_returned=true" << std::endl;
      } else {
        for (std::uint32_t cycle = 0;
             cycle < options.cycles; ++cycle) {
          for (const CapturedGraph& graph : graphs) {
            if (expected_sequence ==
                std::numeric_limits<std::uint64_t>::max()) {
              throw std::overflow_error(
                  "indexer replay marker exhausted");
            }
            const unsigned long long replay =
                expected_sequence + 1;
            const std::size_t active_elements =
                static_cast<std::size_t>(graph.q) *
                spark_transport::kTp4IndexerGraphElementsPerRow;
            const int blocks = static_cast<int>(
                (active_elements + threads - 1) / threads);
            prepare_input<<<blocks, threads, 0, stream>>>(
                input, active_elements, options.transport.rank,
                replay, replay_marker);
            check_cuda(
                cudaGetLastError(),
                "prepare indexer graph replay input");
            check_cuda(
                cudaGraphLaunch(graph.executable, stream),
                "cudaGraphLaunch indexer");
            ++expected_sequence;
            validated_bytes +=
                spark_transport::kTp4IndexerGraphWorldSize *
                active_elements * sizeof(std::int32_t);

            check_cuda(
                cudaStreamSynchronize(stream),
                "indexer graph replay synchronize");
            const auto status =
                wait_for_graph_completion(
                    session, expected_sequence);
            const bool exact = exact_status(
                status, expected_sequence, submit_cpu,
                progress_cpu);
            monotonic_sequences =
                monotonic_sequences && exact;
            if (!exact) {
              throw std::runtime_error(
                  "indexer graph sequence/status did not "
                  "advance exactly");
            }
          }
        }
        check_cuda(
            cudaMemcpy(
                &host_mismatches, mismatches,
                sizeof(host_mismatches),
                cudaMemcpyDeviceToHost),
            "copy indexer mismatch counter");
        final_status = session.graph_replay_status();
      }

      for (const CapturedGraph& graph : graphs) {
        check_cuda(
            cudaGraphExecDestroy(graph.executable),
            "cudaGraphExecDestroy indexer");
      }
    }

    check_cuda(
        cudaStreamDestroy(stream), "cudaStreamDestroy indexer");
    check_cuda(cudaFree(mismatches), "cudaFree indexer mismatches");
    check_cuda(
        cudaFree(replay_marker),
        "cudaFree indexer replay marker");
    check_cuda(cudaFree(output), "cudaFree indexer output");
    check_cuda(cudaFree(input), "cudaFree indexer input");

    if (destructive_returned) {
      return 3;
    }

    const std::uint64_t ring_wraps =
        final_status.completed_sequence /
        spark_transport::kTp4GraphCommandCapacity;
    const bool passed =
        expected_sequence ==
            static_cast<std::uint64_t>(options.cycles) *
                kQPattern.size() &&
        validated_bytes ==
            kOutputBytesPerCycle * options.cycles &&
        ring_wraps >= kRequiredRingWraps &&
        host_mismatches == 0 && monotonic_sequences &&
        final_status.overflow_sequence == 0;

    std::cout
        << "TP4_INDEXER_GRAPH"
        << " rank=" << options.transport.rank
        << " mode=normal publisher=device"
        << " q_pattern=1,23,40"
        << " cycles=" << options.cycles
        << " ring_capacity="
        << spark_transport::kTp4GraphCommandCapacity
        << " required_ring_wraps=" << kRequiredRingWraps
        << " ring_wraps=" << ring_wraps
        << " graph_launches=" << expected_sequence
        << " captured_nodes=" << final_status.captured_nodes
        << " captured_q_mask="
        << final_status.captured_q_mask
        << " census_q1=1 census_q23=1 census_q40=1"
        << " capture_configured="
        << (final_status.capture_configured ? "true" : "false")
        << " polling_enabled="
        << (final_status.polling_enabled ? "true" : "false")
        << " host_native_atomics_supported="
        << (final_status.host_native_atomics_supported
                ? "true"
                : "false")
        << " submit_affinity_verified="
        << (final_status.submit_affinity_verified ? "true" : "false")
        << " progress_affinity_verified="
        << (final_status.progress_affinity_verified ? "true" : "false")
        << " submit_cpu=" << final_status.graph_submit_cpu
        << " progress_cpu=" << final_status.graph_progress_cpu
        << " published=" << final_status.published_sequence
        << " consumed=" << final_status.consumed_sequence
        << " completed=" << final_status.completed_sequence
        << " overflow=" << final_status.overflow_sequence
        << " validated_bytes=" << validated_bytes
        << " mismatched_int32=" << host_mismatches
        << " byte_exact="
        << (host_mismatches == 0 ? "true" : "false")
        << " monotonic_sequences="
        << (monotonic_sequences ? "true" : "false")
        << " passed=" << (passed ? "true" : "false")
        << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
