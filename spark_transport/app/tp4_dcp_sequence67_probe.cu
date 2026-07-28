#include "spark_transport/gpu_tp4_dcp_query.hpp"
#include "spark_transport/tp4_c_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

// This is the deployed C1--C8 decode padding surface. The wider PIECEWISE
// prefill buckets are intentionally absent because DCP query admission ends
// at Q40.
constexpr std::array<std::uint32_t, 18> kDecodeRows{
    1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 35, 40};
constexpr std::array<std::uint32_t, 8> kFullRows{
    5, 10, 15, 20, 25, 30, 35, 40};
constexpr std::uint32_t kSequence67Q = 40;

enum class PrefixPattern {
  kDecodeBuckets,
  kQ40,
};

enum class StreamMode {
  kShared,
  kSwitchAt67,
  kSplitCapture,
};

enum class Pre67Sync {
  kNone,
  kEagerStream,
  kDevice,
};

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t eager_port0{9910};
  std::uint16_t eager_port1{9911};
  std::uint16_t graph_port0{9912};
  std::uint16_t graph_port1{9913};
  std::uint32_t submit_cpu{10};
  std::uint32_t progress_cpu{13};
  std::uint32_t prefix_queries{66};
  PrefixPattern prefix_pattern{PrefixPattern::kDecodeBuckets};
  StreamMode stream_mode{StreamMode::kShared};
  Pre67Sync pre67_sync{Pre67Sync::kNone};
  std::uint32_t sequence67_delay_ms{};
  std::uint32_t graph_replays{1};
};

struct Graph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  std::uint32_t q{};

  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  Graph(Graph&& other) noexcept
      : graph(std::exchange(other.graph, nullptr)),
        executable(std::exchange(other.executable, nullptr)),
        q(other.q) {}

  Graph& operator=(Graph&& other) noexcept {
    if (this != &other) {
      destroy();
      graph = std::exchange(other.graph, nullptr);
      executable = std::exchange(other.executable, nullptr);
      q = other.q;
    }
    return *this;
  }

  ~Graph() { destroy(); }

 private:
  void destroy() noexcept {
    if (executable != nullptr) {
      static_cast<void>(cudaGraphExecDestroy(executable));
      executable = nullptr;
    }
    if (graph != nullptr) {
      static_cast<void>(cudaGraphDestroy(graph));
      graph = nullptr;
    }
  }
};

class DcpHandle {
 public:
  explicit DcpHandle(spark_tp4_dcp_handle handle) : handle_(handle) {}
  DcpHandle(const DcpHandle&) = delete;
  DcpHandle& operator=(const DcpHandle&) = delete;
  ~DcpHandle() { spark_tp4_dcp_destroy(handle_); }

  spark_tp4_dcp_handle get() const { return handle_; }

 private:
  spark_tp4_dcp_handle handle_{};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n"
      << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
      << "         --eager-port0 PORT --eager-port1 PORT\n"
      << "         --graph-port0 PORT --graph-port1 PORT\n"
      << "         --submit-cpu CPU --progress-cpu CPU\n"
      << "         --prefix-queries N\n"
      << "         --prefix-pattern decode-buckets|q40\n"
      << "         --stream-mode shared|switch-at-67|split-capture\n"
      << "         --pre67-sync none|eager-stream|device\n"
      << "         --sequence67-delay-ms N --graph-replays N\n";
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
    } else if (argument == "--eager-port0") {
      options.eager_port0 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "eager port 0"));
    } else if (argument == "--eager-port1") {
      options.eager_port1 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "eager port 1"));
    } else if (argument == "--graph-port0") {
      options.graph_port0 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "graph port 0"));
    } else if (argument == "--graph-port1") {
      options.graph_port1 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "graph port 1"));
    } else if (argument == "--submit-cpu") {
      options.submit_cpu = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "submit CPU"));
    } else if (argument == "--progress-cpu") {
      options.progress_cpu = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "progress CPU"));
    } else if (argument == "--prefix-queries") {
      options.prefix_queries = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "prefix queries"));
    } else if (argument == "--prefix-pattern") {
      const std::string_view value(take_value());
      if (value == "decode-buckets") {
        options.prefix_pattern = PrefixPattern::kDecodeBuckets;
      } else if (value == "q40") {
        options.prefix_pattern = PrefixPattern::kQ40;
      } else {
        usage(argv[0]);
      }
    } else if (argument == "--stream-mode") {
      const std::string_view value(take_value());
      if (value == "shared") {
        options.stream_mode = StreamMode::kShared;
      } else if (value == "switch-at-67") {
        options.stream_mode = StreamMode::kSwitchAt67;
      } else if (value == "split-capture") {
        options.stream_mode = StreamMode::kSplitCapture;
      } else {
        usage(argv[0]);
      }
    } else if (argument == "--pre67-sync") {
      const std::string_view value(take_value());
      if (value == "none") {
        options.pre67_sync = Pre67Sync::kNone;
      } else if (value == "eager-stream") {
        options.pre67_sync = Pre67Sync::kEagerStream;
      } else if (value == "device") {
        options.pre67_sync = Pre67Sync::kDevice;
      } else {
        usage(argv[0]);
      }
    } else if (argument == "--sequence67-delay-ms") {
      options.sequence67_delay_ms = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "sequence-67 delay"));
    } else if (argument == "--graph-replays") {
      options.graph_replays = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "graph replays"));
    } else {
      usage(argv[0]);
    }
  }

  const std::array<std::uint16_t, 4> ports{
      options.eager_port0, options.eager_port1,
      options.graph_port0, options.graph_port1};
  std::array<std::uint16_t, ports.size()> sorted_ports = ports;
  std::sort(sorted_ports.begin(), sorted_ports.end());
  const bool duplicate_port =
      std::adjacent_find(sorted_ports.begin(), sorted_ports.end()) !=
      sorted_ports.end();
  if (options.rank >= spark_transport::kTp4DcpQueryWorldSize ||
      options.peer0.empty() || options.peer1.empty() ||
      duplicate_port || options.submit_cpu == options.progress_cpu ||
      options.prefix_queries == 0 || options.prefix_queries > 4096 ||
      options.sequence67_delay_ms > 600000 ||
      options.graph_replays == 0 || options.graph_replays > 10000) {
    usage(argv[0]);
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

const char* prefix_pattern_name(PrefixPattern pattern) {
  return pattern == PrefixPattern::kDecodeBuckets
             ? "decode-buckets"
             : "q40";
}

const char* stream_mode_name(StreamMode mode) {
  switch (mode) {
    case StreamMode::kShared:
      return "shared";
    case StreamMode::kSwitchAt67:
      return "switch-at-67";
    case StreamMode::kSplitCapture:
      return "split-capture";
  }
  return "invalid";
}

const char* pre67_sync_name(Pre67Sync mode) {
  switch (mode) {
    case Pre67Sync::kNone:
      return "none";
    case Pre67Sync::kEagerStream:
      return "eager-stream";
    case Pre67Sync::kDevice:
      return "device";
  }
  return "invalid";
}

void phase(std::uint32_t rank, const char* name,
           std::uint64_t sequence = 0, std::uint32_t q = 0) {
  std::cout << "DCP_SEQUENCE67_PHASE"
            << " rank=" << rank
            << " phase=" << name;
  if (sequence != 0) {
    std::cout << " sequence=" << sequence;
  }
  if (q != 0) {
    std::cout << " q=" << q;
  }
  std::cout << '\n' << std::flush;
}

std::uint16_t expected_word(std::uint32_t rank,
                            std::uint32_t query_index,
                            std::uint32_t local_head,
                            std::uint32_t dimension) {
  return static_cast<std::uint16_t>(
      0x1000U + rank * 7919U + query_index * 1223U +
      local_head * 577U + dimension * 13U);
}

std::size_t input_index(std::uint32_t query_index,
                        std::uint32_t local_head,
                        std::uint32_t dimension) {
  return (static_cast<std::size_t>(query_index) *
              spark_transport::kTp4DcpQueryHeadsPerRank +
          local_head) *
             spark_transport::kTp4DcpQueryHeadDimension +
         dimension;
}

std::size_t output_index(std::uint32_t query_index,
                         std::uint32_t global_head,
                         std::uint32_t dimension) {
  constexpr std::size_t global_heads =
      spark_transport::kTp4DcpQueryHeadsPerRank *
      spark_transport::kTp4DcpQueryWorldSize;
  return (static_cast<std::size_t>(query_index) * global_heads +
          global_head) *
             spark_transport::kTp4DcpQueryHeadDimension +
         dimension;
}

std::uint32_t prefix_q(const Options& options, std::uint32_t index) {
  if (options.prefix_pattern == PrefixPattern::kQ40) {
    return kSequence67Q;
  }
  return kDecodeRows[index % kDecodeRows.size()];
}

cudaStream_t prefix_stream(
    const Options&, const std::array<cudaStream_t, 2>& streams) {
  return streams[0];
}

cudaStream_t piecewise_stream(
    const Options& options,
    const std::array<cudaStream_t, 2>& streams) {
  return options.stream_mode == StreamMode::kSplitCapture
             ? streams[1]
             : streams[0];
}

cudaStream_t sequence67_stream(
    const Options& options,
    const std::array<cudaStream_t, 2>& streams) {
  return options.stream_mode == StreamMode::kSwitchAt67
             ? streams[1]
             : streams[0];
}

cudaStream_t full_stream(
    const Options& options,
    const std::array<cudaStream_t, 2>& streams) {
  return options.stream_mode == StreamMode::kShared
             ? streams[0]
             : streams[1];
}

void CUDART_CB delay_stream(void* delay_pointer) {
  const auto delay =
      *static_cast<const std::chrono::milliseconds*>(delay_pointer);
  std::this_thread::sleep_for(delay);
}

Graph capture_query_graph(
    spark_tp4_dcp_handle session, const void* input, void* output,
    std::uint32_t q, cudaStream_t stream) {
  Graph result;
  result.q = q;
  char error[512]{};
  check_cuda(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
      "begin DCP sequence-67 graph capture");
  if (spark_tp4_dcp_capture_query_all_gather(
          session, input, output, q, stream, error, sizeof(error)) != 0) {
    throw std::runtime_error(
        std::string("capture DCP sequence-67 query: ") + error);
  }
  check_cuda(
      cudaStreamEndCapture(stream, &result.graph),
      "end DCP sequence-67 graph capture");
  check_cuda(
      cudaGraphInstantiate(&result.executable, result.graph, 0),
      "instantiate DCP sequence-67 graph");
  return result;
}

spark_tp4_dcp_graph_status graph_status(spark_tp4_dcp_handle handle) {
  spark_tp4_dcp_graph_status status{};
  char error[512]{};
  if (spark_tp4_dcp_get_graph_status(
          handle, &status, sizeof(status), error, sizeof(error)) != 0) {
    throw std::runtime_error(
        std::string("get DCP sequence-67 graph status: ") + error);
  }
  if (status.struct_size != sizeof(status)) {
    throw std::runtime_error("DCP sequence-67 graph status ABI mismatch");
  }
  return status;
}

spark_tp4_dcp_graph_status wait_for_graph(
    spark_tp4_dcp_handle handle, std::uint64_t expected_sequence) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (true) {
    const auto status = graph_status(handle);
    if (status.overflow_sequence != 0) {
      throw std::runtime_error("DCP sequence-67 graph ring overflow");
    }
    if (status.completed_sequence == expected_sequence) {
      return status;
    }
    if (status.completed_sequence > expected_sequence ||
        std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          "DCP sequence-67 graph completion did not advance exactly");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

std::size_t verify_q40_output(const void* device_output) {
  constexpr std::uint32_t q = kSequence67Q;
  const std::size_t output_bytes =
      spark_transport::tp4_dcp_query_output_bytes(q);
  std::vector<std::uint16_t> host_output(
      output_bytes / sizeof(std::uint16_t));
  check_cuda(
      cudaMemcpy(
          host_output.data(), device_output, output_bytes,
          cudaMemcpyDeviceToHost),
      "copy DCP sequence-67 output");

  std::size_t mismatches{};
  constexpr std::uint32_t global_heads =
      spark_transport::kTp4DcpQueryHeadsPerRank *
      spark_transport::kTp4DcpQueryWorldSize;
  for (std::uint32_t query_index = 0; query_index < q; ++query_index) {
    for (std::uint32_t global_head = 0; global_head < global_heads;
         ++global_head) {
      const std::uint32_t source_rank =
          global_head / spark_transport::kTp4DcpQueryHeadsPerRank;
      const std::uint32_t local_head =
          global_head % spark_transport::kTp4DcpQueryHeadsPerRank;
      for (std::uint32_t dimension = 0;
           dimension < spark_transport::kTp4DcpQueryHeadDimension;
           ++dimension) {
        const auto actual =
            host_output[output_index(query_index, global_head, dimension)];
        const auto expected = expected_word(
            source_rank, query_index, local_head, dimension);
        mismatches += actual == expected ? 0U : 1U;
      }
    }
  }
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::size_t input_words =
        spark_transport::tp4_dcp_query_input_bytes(kSequence67Q) /
        sizeof(std::uint16_t);
    const std::size_t output_bytes =
        spark_transport::tp4_dcp_query_output_bytes(kSequence67Q);
    std::vector<std::uint16_t> host_input(input_words);
    for (std::uint32_t query_index = 0; query_index < kSequence67Q;
         ++query_index) {
      for (std::uint32_t head = 0;
           head < spark_transport::kTp4DcpQueryHeadsPerRank; ++head) {
        for (std::uint32_t dimension = 0;
             dimension < spark_transport::kTp4DcpQueryHeadDimension;
             ++dimension) {
          host_input[input_index(query_index, head, dimension)] =
              expected_word(options.rank, query_index, head, dimension);
        }
      }
    }

    void* input{};
    void* eager_output{};
    void* graph_output{};
    std::array<cudaStream_t, 2> streams{};
    check_cuda(
        cudaMalloc(&input, host_input.size() * sizeof(std::uint16_t)),
        "allocate DCP sequence-67 input");
    check_cuda(
        cudaMalloc(&eager_output, output_bytes),
        "allocate DCP sequence-67 eager output");
    check_cuda(
        cudaMalloc(&graph_output, output_bytes),
        "allocate DCP sequence-67 graph output");
    check_cuda(
        cudaMemcpy(
            input, host_input.data(),
            host_input.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),
        "copy DCP sequence-67 input");
    for (cudaStream_t& stream : streams) {
      check_cuda(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create DCP sequence-67 stream");
    }

    bool passed{};
    std::size_t eager_mismatches{};
    std::size_t graph_mismatches{};
    std::uint64_t expected_graph_sequence{};
    spark_tp4_dcp_graph_status final_graph_status{};
    {
      spark_tp4_dcp_config eager_config{};
      eager_config.rank = options.rank;
      eager_config.peer0 = options.peer0.c_str();
      eager_config.peer1 = options.peer1.c_str();
      eager_config.device0 = options.device0.c_str();
      eager_config.device1 = options.device1.c_str();
      eager_config.gid0 = options.gid0;
      eager_config.gid1 = options.gid1;
      eager_config.control_port0 = options.eager_port0;
      eager_config.control_port1 = options.eager_port1;

      char error[512]{};
      const DcpHandle eager_session(
          spark_tp4_dcp_create(&eager_config, error, sizeof(error)));
      if (eager_session.get() == nullptr) {
        throw std::runtime_error(
            std::string("create eager DCP sequence-67 session: ") + error);
      }

      spark_tp4_dcp_graph_config graph_config{};
      graph_config.rank = options.rank;
      graph_config.peer0 = options.peer0.c_str();
      graph_config.peer1 = options.peer1.c_str();
      graph_config.device0 = options.device0.c_str();
      graph_config.device1 = options.device1.c_str();
      graph_config.gid0 = options.gid0;
      graph_config.gid1 = options.gid1;
      graph_config.control_port0 = options.graph_port0;
      graph_config.control_port1 = options.graph_port1;
      graph_config.graph_submit_cpu_plus_one = options.submit_cpu + 1;
      graph_config.graph_progress_cpu_plus_one = options.progress_cpu + 1;
      const DcpHandle graph_session(
          spark_tp4_dcp_graph_create(
              &graph_config, error, sizeof(error)));
      if (graph_session.get() == nullptr) {
        throw std::runtime_error(
            std::string("create graph DCP sequence-67 session: ") + error);
      }
      phase(options.rank, "sessions_ready");

      const cudaStream_t initial_stream =
          prefix_stream(options, streams);
      for (std::uint32_t index = 0; index < options.prefix_queries;
           ++index) {
        const std::uint32_t q = prefix_q(options, index);
        const std::uint64_t sequence =
            static_cast<std::uint64_t>(index) + 1;
        if (spark_tp4_dcp_query_all_gather(
                eager_session.get(), input, eager_output, q,
                initial_stream, error, sizeof(error)) != 0) {
          throw std::runtime_error(
              "submit eager DCP prefix sequence " +
              std::to_string(sequence) + " q=" + std::to_string(q) +
              ": " + error);
        }
      }
      phase(
          options.rank, "eager_prefix_submitted",
          options.prefix_queries,
          prefix_q(options, options.prefix_queries - 1));

      std::vector<Graph> piecewise_graphs;
      piecewise_graphs.reserve(kDecodeRows.size());
      const cudaStream_t piecewise =
          piecewise_stream(options, streams);
      for (const std::uint32_t q : kDecodeRows) {
        piecewise_graphs.push_back(capture_query_graph(
            graph_session.get(), input, graph_output, q, piecewise));
      }
      phase(options.rank, "piecewise_captured");

      if (options.pre67_sync == Pre67Sync::kEagerStream) {
        check_cuda(
            cudaStreamSynchronize(initial_stream),
            "pre-sequence-67 eager stream synchronize");
      } else if (options.pre67_sync == Pre67Sync::kDevice) {
        check_cuda(
            cudaDeviceSynchronize(),
            "pre-sequence-67 device synchronize");
      }
      phase(options.rank, "pre67_sync_complete");

      const cudaStream_t sequence67 =
          sequence67_stream(options, streams);
      std::chrono::milliseconds sequence67_delay(
          options.sequence67_delay_ms);
      if (sequence67_delay.count() != 0) {
        check_cuda(
            cudaLaunchHostFunc(
                sequence67, delay_stream, &sequence67_delay),
            "enqueue sequence-67 diagnostic delay");
      }
      const std::uint64_t transition_sequence =
          static_cast<std::uint64_t>(options.prefix_queries) + 1;
      phase(
          options.rank, "sequence67_submit_begin",
          transition_sequence, kSequence67Q);
      if (spark_tp4_dcp_query_all_gather(
              eager_session.get(), input, eager_output, kSequence67Q,
              sequence67, error, sizeof(error)) != 0) {
        throw std::runtime_error(
            "submit eager DCP transition sequence " +
            std::to_string(transition_sequence) + " q=40: " + error);
      }
      phase(
          options.rank, "sequence67_submitted",
          transition_sequence, kSequence67Q);

      std::vector<Graph> full_graphs;
      full_graphs.reserve(kFullRows.size() * 2);
      const cudaStream_t full = full_stream(options, streams);
      for (int pass = 0; pass < 2; ++pass) {
        for (const std::uint32_t q : kFullRows) {
          full_graphs.push_back(capture_query_graph(
              graph_session.get(), input, graph_output, q, full));
        }
      }
      phase(options.rank, "full_captured");

      const auto before_replay = graph_status(graph_session.get());
      const std::uint64_t expected_nodes =
          piecewise_graphs.size() + full_graphs.size();
      if (before_replay.captured_nodes != expected_nodes ||
          before_replay.captured_query_nodes != expected_nodes ||
          before_replay.captured_combine_nodes != 0 ||
          before_replay.published_sequence != 0 ||
          before_replay.consumed_sequence != 0 ||
          before_replay.completed_sequence != 0 ||
          before_replay.overflow_sequence != 0) {
        throw std::runtime_error(
            "invalid DCP sequence-67 graph status before replay");
      }

      for (std::uint32_t replay = 0; replay < options.graph_replays;
           ++replay) {
        for (const auto& item : piecewise_graphs) {
          check_cuda(
              cudaGraphLaunch(item.executable, piecewise),
              "launch DCP sequence-67 PIECEWISE graph");
        }
        for (const auto& item : full_graphs) {
          check_cuda(
              cudaGraphLaunch(item.executable, full),
              "launch DCP sequence-67 FULL graph");
        }
      }
      expected_graph_sequence =
          expected_nodes * options.graph_replays;
      phase(options.rank, "graphs_replayed");

      check_cuda(
          cudaStreamSynchronize(sequence67),
          "synchronize DCP sequence-67 eager stream");
      check_cuda(
          cudaStreamSynchronize(piecewise),
          "synchronize DCP sequence-67 PIECEWISE stream");
      check_cuda(
          cudaStreamSynchronize(full),
          "synchronize DCP sequence-67 FULL stream");
      final_graph_status =
          wait_for_graph(graph_session.get(), expected_graph_sequence);
      phase(
          options.rank, "sequence67_completed",
          transition_sequence, kSequence67Q);

      eager_mismatches = verify_q40_output(eager_output);
      graph_mismatches = verify_q40_output(graph_output);
      passed =
          eager_mismatches == 0 && graph_mismatches == 0 &&
          final_graph_status.published_sequence ==
              expected_graph_sequence &&
          final_graph_status.consumed_sequence ==
              expected_graph_sequence &&
          final_graph_status.completed_sequence ==
              expected_graph_sequence &&
          final_graph_status.overflow_sequence == 0;
    }

    for (cudaStream_t stream : streams) {
      static_cast<void>(cudaStreamDestroy(stream));
    }
    static_cast<void>(cudaFree(graph_output));
    static_cast<void>(cudaFree(eager_output));
    static_cast<void>(cudaFree(input));

    std::cout
        << "TP4_DCP_SEQUENCE67"
        << " rank=" << options.rank
        << " prefix_queries=" << options.prefix_queries
        << " transition_sequence=" << options.prefix_queries + 1
        << " transition_q=" << kSequence67Q
        << " prefix_pattern="
        << prefix_pattern_name(options.prefix_pattern)
        << " stream_mode=" << stream_mode_name(options.stream_mode)
        << " pre67_sync=" << pre67_sync_name(options.pre67_sync)
        << " sequence67_delay_ms=" << options.sequence67_delay_ms
        << " piecewise_graphs=" << kDecodeRows.size()
        << " full_graphs=" << kFullRows.size() * 2
        << " graph_replays=" << options.graph_replays
        << " graph_expected_sequence=" << expected_graph_sequence
        << " graph_published="
        << final_graph_status.published_sequence
        << " graph_consumed="
        << final_graph_status.consumed_sequence
        << " graph_completed="
        << final_graph_status.completed_sequence
        << " graph_overflow="
        << final_graph_status.overflow_sequence
        << " eager_mismatches=" << eager_mismatches
        << " graph_mismatches=" << graph_mismatches
        << " passed=" << (passed ? "true" : "false")
        << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TP4_DCP_SEQUENCE67_ERROR " << error.what() << '\n';
    return 1;
  }
}
