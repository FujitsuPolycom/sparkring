#include "spark_transport/gpu_tp4_dcp_combine.hpp"
#include "spark_transport/gpu_tp4_dcp_query.hpp"
#include "spark_transport/tp4_c_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::array<std::uint32_t, 14> kQueryRows{
    1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40};
constexpr std::size_t kNodesPerBucket = 2;

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9892};
  std::uint16_t control_port1{9893};
  std::uint32_t submit_cpu{10};
  std::uint32_t progress_cpu{13};
  int warmup{2};
  int iterations{100};
  std::uint32_t combine_dimension{};
  std::uint32_t single_q{};
  double max_output_abs{0.0625};
  double max_lse_abs{2.0e-5};
};

struct BucketBuffers {
  std::uint32_t q{};
  std::uint32_t head_dimension{};
  void* query_output{};
  void* combine_output{};
  void* combine_lse{};
};

struct ErrorStats {
  double max_absolute{};
  double max_relative{};
  std::size_t nonfinite_mismatches{};

  void observe(float actual, float expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
      if (!(std::isinf(actual) && std::isinf(expected) &&
            std::signbit(actual) == std::signbit(expected))) {
        ++nonfinite_mismatches;
      }
      return;
    }
    const double difference =
        std::abs(static_cast<double>(actual) - expected);
    max_absolute = std::max(max_absolute, difference);
    max_relative = std::max(
        max_relative,
        difference /
            std::max(std::abs(static_cast<double>(expected)), 1.0e-12));
  }
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n"
      << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
      << "         --control-port0 PORT --control-port1 PORT\n"
      << "         --submit-cpu CPU --progress-cpu CPU\n"
      << "         --warmup N --iterations N\n"
      << "         --combine-dimension 0|256|512 (0 alternates)\n"
      << "         --single-q 0|1|2|3|4|5|6|8|10|12|16|20|24|32|40\n"
      << "         --max-output-abs X --max-lse-abs X\n";
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

double floating_value(const char* value, const char* name) {
  std::size_t consumed{};
  const std::string text(value);
  const double parsed = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(parsed) || parsed < 0.0) {
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
    } else if (argument == "--combine-dimension") {
      options.combine_dimension = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "combine dimension"));
    } else if (argument == "--single-q") {
      options.single_q = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "single Q"));
    } else if (argument == "--max-output-abs") {
      options.max_output_abs =
          floating_value(take_value(), "maximum output error");
    } else if (argument == "--max-lse-abs") {
      options.max_lse_abs =
          floating_value(take_value(), "maximum LSE error");
    } else {
      usage(argv[0]);
    }
  }
  if (options.rank >= spark_transport::kTp4DcpQueryWorldSize ||
      options.peer0.empty() || options.peer1.empty() ||
      options.control_port0 == options.control_port1 ||
      options.submit_cpu == options.progress_cpu ||
      options.warmup < 0 || options.iterations <= 0 ||
      (
          options.combine_dimension != 0 &&
          options.combine_dimension != 256 &&
          options.combine_dimension != 512
      ) ||
      (
          options.single_q != 0 &&
          std::find(
              kQueryRows.begin(), kQueryRows.end(),
              options.single_q) == kQueryRows.end()
      )) {
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

std::uint16_t query_word(std::uint32_t rank,
                         std::uint32_t query_index,
                         std::uint32_t local_head,
                         std::uint32_t dimension) {
  return static_cast<std::uint16_t>(
      0x1000U + rank * 7919U + query_index * 1223U +
      local_head * 577U + dimension * 13U);
}

std::size_t query_input_index(std::uint32_t query_index,
                              std::uint32_t local_head,
                              std::uint32_t dimension) {
  return (static_cast<std::size_t>(query_index) *
              spark_transport::kTp4DcpQueryHeadsPerRank +
          local_head) *
             spark_transport::kTp4DcpQueryHeadDimension +
         dimension;
}

std::size_t query_output_index(std::uint32_t query_index,
                               std::uint32_t source_rank,
                               std::uint32_t local_head,
                               std::uint32_t dimension) {
  const std::uint32_t global_head =
      source_rank * spark_transport::kTp4DcpQueryHeadsPerRank +
      local_head;
  return (static_cast<std::size_t>(query_index) *
              spark_transport::kTp4DcpCombineGlobalHeads +
          global_head) *
             spark_transport::kTp4DcpQueryHeadDimension +
         dimension;
}

std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  return value ^ (value >> 16U);
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t round = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + round) >> 16U);
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result{};
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float generated_output(std::uint32_t rank,
                       std::uint32_t query_index,
                       std::uint32_t global_head,
                       std::uint32_t dimension) {
  const std::uint32_t seed =
      mix(0x9e3779b9U ^ (rank * 0x1000193U) ^
          (query_index * 0x85ebca6bU) ^
          (global_head * 0xc2b2ae35U) ^ dimension);
  const std::int32_t centered =
      static_cast<std::int32_t>(seed % 4097U) - 2048;
  return static_cast<float>(centered) / 512.0F;
}

float generated_lse(std::uint32_t rank,
                    std::uint32_t query_index,
                    std::uint32_t global_head) {
  const std::uint32_t seed =
      mix(0x243f6a88U ^ (rank * 0x9e3779b9U) ^
          (query_index * 0x85ebca6bU) ^ global_head);
  const std::int32_t centered =
      static_cast<std::int32_t>(seed % 12001U) - 6000;
  return static_cast<float>(centered) / 1000.0F;
}

float global_lse(std::uint32_t query_index,
                 std::uint32_t global_head) {
  std::array<float, spark_transport::kTp4DcpCombineWorldSize> values{};
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::uint32_t rank = 0;
       rank < spark_transport::kTp4DcpCombineWorldSize; ++rank) {
    values[rank] = generated_lse(rank, query_index, global_head);
    maximum = std::max(maximum, values[rank]);
  }
  float denominator{};
  for (const float value : values) {
    denominator += std::exp(value - maximum);
  }
  return maximum + std::log(denominator);
}

float global_output(std::uint32_t query_index,
                    std::uint32_t global_head,
                    std::uint32_t dimension,
                    float reduced_lse) {
  float result{};
  for (std::uint32_t rank = 0;
       rank < spark_transport::kTp4DcpCombineWorldSize; ++rank) {
    const float value = bf16_to_float(float_to_bf16(generated_output(
        rank, query_index, global_head, dimension)));
    const float lse = generated_lse(rank, query_index, global_head);
    result += value * std::exp(lse - reduced_lse);
  }
  return result;
}

std::size_t combine_input_index(std::uint32_t query_index,
                                std::uint32_t global_head,
                                std::uint32_t dimension,
                                std::uint32_t head_dimension) {
  return (static_cast<std::size_t>(query_index) *
              spark_transport::kTp4DcpCombineGlobalHeads +
          global_head) *
             head_dimension +
         dimension;
}

std::size_t combine_lse_index(std::uint32_t query_index,
                              std::uint32_t global_head) {
  return static_cast<std::size_t>(query_index) *
             spark_transport::kTp4DcpCombineGlobalHeads +
         global_head;
}

std::size_t reduced_output_index(std::uint32_t query_index,
                                 std::uint32_t local_head,
                                 std::uint32_t dimension,
                                 std::uint32_t head_dimension) {
  return spark_transport::tp4_dcp_combine_token_major_reduced_output_index(
      query_index, local_head, dimension, head_dimension);
}

std::size_t reduced_lse_index(std::uint32_t query_index,
                              std::uint32_t local_head) {
  return static_cast<std::size_t>(query_index) *
             spark_transport::kTp4DcpCombineHeadsPerRank +
         local_head;
}

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

spark_tp4_dcp_graph_status graph_status(spark_tp4_dcp_handle handle) {
  spark_tp4_dcp_graph_status status{};
  char error[512]{};
  if (spark_tp4_dcp_get_graph_status(
          handle, &status, sizeof(status), error, sizeof(error)) != 0) {
    throw std::runtime_error(std::string("DCP graph status: ") + error);
  }
  if (status.struct_size != sizeof(status)) {
    throw std::runtime_error("DCP graph status ABI size mismatch");
  }
  return status;
}

spark_tp4_dcp_graph_status wait_for_completion(
    spark_tp4_dcp_handle handle, std::uint64_t expected_sequence) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (true) {
    const auto status = graph_status(handle);
    if (status.overflow_sequence != 0) {
      throw std::runtime_error("DCP graph command overflow");
    }
    if (status.completed_sequence == expected_sequence) {
      return status;
    }
    if (status.completed_sequence > expected_sequence ||
        std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          "DCP graph completion did not advance exactly");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const char* poll_policy_value =
        std::getenv("SPARK_TP4_DCP_GRAPH_POLL_POLICY");
    const std::string_view poll_policy =
        poll_policy_value == nullptr
            ? std::string_view("adaptive-yield")
            : std::string_view(poll_policy_value);
    if (
        poll_policy != "adaptive-yield" &&
        poll_policy != "dedicated-spin") {
      throw std::runtime_error(
          "SPARK_TP4_DCP_GRAPH_POLL_POLICY must be "
          "adaptive-yield or dedicated-spin");
    }
    const bool expected_dedicated_spin =
        poll_policy == "dedicated-spin";
    constexpr std::uint32_t maximum_q =
        spark_transport::kTp4DcpQueryMaxQ;
    constexpr std::uint32_t maximum_dimension =
        spark_transport::kTp4DcpCombineMaxHeadDimension;
    const std::size_t captured_query_nodes =
        options.single_q == 0 ? kQueryRows.size() : 1;
    const std::size_t captured_combine_nodes =
        captured_query_nodes;
    const std::size_t captured_nodes =
        kNodesPerBucket * captured_query_nodes;

    std::vector<std::uint16_t> host_query_input(
        spark_transport::tp4_dcp_query_input_bytes(maximum_q) /
        sizeof(std::uint16_t));
    for (std::uint32_t query_index = 0; query_index < maximum_q;
         ++query_index) {
      for (std::uint32_t head = 0;
           head < spark_transport::kTp4DcpQueryHeadsPerRank; ++head) {
        for (std::uint32_t dimension = 0;
             dimension < spark_transport::kTp4DcpQueryHeadDimension;
             ++dimension) {
          host_query_input[
              query_input_index(query_index, head, dimension)] =
              query_word(options.rank, query_index, head, dimension);
        }
      }
    }

    constexpr std::array<std::uint32_t, 2> combine_dimensions{
        256, maximum_dimension};
    std::array<std::vector<std::uint16_t>, combine_dimensions.size()>
        host_combine_inputs{};
    std::vector<float> host_combine_lse(
        spark_transport::tp4_dcp_combine_input_lse_bytes(maximum_q) /
        sizeof(float));
    for (std::size_t dimension_index = 0;
         dimension_index < combine_dimensions.size(); ++dimension_index) {
      host_combine_inputs[dimension_index].resize(
          spark_transport::tp4_dcp_combine_input_output_bytes(
              maximum_q, combine_dimensions[dimension_index]) /
          sizeof(std::uint16_t));
    }
    for (std::uint32_t query_index = 0; query_index < maximum_q;
         ++query_index) {
      for (std::uint32_t global_head = 0;
           global_head < spark_transport::kTp4DcpCombineGlobalHeads;
           ++global_head) {
        host_combine_lse[
            combine_lse_index(query_index, global_head)] =
            generated_lse(options.rank, query_index, global_head);
        for (std::size_t dimension_index = 0;
             dimension_index < combine_dimensions.size();
             ++dimension_index) {
          const std::uint32_t head_dimension =
              combine_dimensions[dimension_index];
          for (std::uint32_t dimension = 0;
               dimension < head_dimension; ++dimension) {
            host_combine_inputs[dimension_index][combine_input_index(
                query_index, global_head, dimension,
                head_dimension)] =
                float_to_bf16(generated_output(
                    options.rank, query_index, global_head, dimension));
          }
        }
      }
    }

    void* query_input{};
    std::array<void*, combine_dimensions.size()> combine_inputs{};
    void* combine_lse{};
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    std::array<BucketBuffers, kQueryRows.size()> buckets{};

    check_cuda(cudaMalloc(
                   &query_input,
                   host_query_input.size() * sizeof(std::uint16_t)),
               "cudaMalloc query input");
    for (std::size_t dimension_index = 0;
         dimension_index < combine_dimensions.size(); ++dimension_index) {
      check_cuda(cudaMalloc(
                     &combine_inputs[dimension_index],
                     host_combine_inputs[dimension_index].size() *
                         sizeof(std::uint16_t)),
                 "cudaMalloc combine input");
    }
    check_cuda(cudaMalloc(
                   &combine_lse,
                   host_combine_lse.size() * sizeof(float)),
               "cudaMalloc combine LSE");
    check_cuda(cudaMemcpy(
                   query_input, host_query_input.data(),
                   host_query_input.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice),
               "copy query input");
    for (std::size_t dimension_index = 0;
         dimension_index < combine_dimensions.size(); ++dimension_index) {
      check_cuda(cudaMemcpy(
                     combine_inputs[dimension_index],
                     host_combine_inputs[dimension_index].data(),
                     host_combine_inputs[dimension_index].size() *
                         sizeof(std::uint16_t),
                     cudaMemcpyHostToDevice),
                 "copy combine input");
    }
    check_cuda(cudaMemcpy(
                   combine_lse, host_combine_lse.data(),
                   host_combine_lse.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
               "copy combine LSE");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create DCP graph stream");
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    for (std::size_t index = 0; index < buckets.size(); ++index) {
      auto& bucket = buckets[index];
      bucket.q = kQueryRows[index];
      bucket.head_dimension =
          options.combine_dimension == 0
              ? (index % 2 == 0 ? 256U : 512U)
              : options.combine_dimension;
      check_cuda(cudaMalloc(
                     &bucket.query_output,
                     spark_transport::tp4_dcp_query_output_bytes(
                         bucket.q)),
                 "cudaMalloc query output");
      check_cuda(cudaMalloc(
                     &bucket.combine_output,
                     spark_transport::tp4_dcp_combine_reduced_output_bytes(
                         bucket.q, bucket.head_dimension)),
                 "cudaMalloc combine output");
      check_cuda(cudaMalloc(
                     &bucket.combine_lse,
                     spark_transport::tp4_dcp_combine_reduced_lse_bytes(
                         bucket.q)),
                 "cudaMalloc combine reduced LSE");
    }

    bool passed{};
    {
      spark_tp4_dcp_graph_config config{};
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
      const DcpHandle session(
          spark_tp4_dcp_graph_create(&config, error, sizeof(error)));
      if (session.get() == nullptr) {
        throw std::runtime_error(
            std::string("create DCP graph session: ") + error);
      }

      check_cuda(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
          "begin DCP graph capture");
      for (const auto& bucket : buckets) {
        if (
            options.single_q != 0 &&
            bucket.q != options.single_q
        ) {
          continue;
        }
        if (spark_tp4_dcp_capture_query_all_gather(
                session.get(), query_input, bucket.query_output, bucket.q,
                stream, error, sizeof(error)) != 0) {
          throw std::runtime_error(
              std::string("capture DCP query node: ") + error);
        }
        const std::uint32_t query_stride =
            spark_transport::kTp4DcpCombineGlobalHeads *
            bucket.head_dimension;
        const std::size_t combine_input_index =
            bucket.head_dimension == combine_dimensions[0] ? 0 : 1;
        if (spark_tp4_dcp_capture_combine(
                session.get(), combine_inputs[combine_input_index],
                combine_lse,
                bucket.combine_output, bucket.combine_lse, bucket.q,
                bucket.head_dimension, query_stride,
                bucket.head_dimension, stream, error,
                sizeof(error)) != 0) {
          throw std::runtime_error(
              std::string("capture DCP combine node: ") + error);
        }
      }
      check_cuda(cudaStreamEndCapture(stream, &graph),
                 "end DCP graph capture");
      check_cuda(cudaGraphInstantiate(&executable, graph, 0),
                 "instantiate DCP graph");

      const auto before = graph_status(session.get());
      const std::uint32_t required_flags =
          SPARK_TP4_GRAPH_STATUS_CAPTURE_CONFIGURED |
          SPARK_TP4_GRAPH_STATUS_POLLING_ENABLED |
          SPARK_TP4_GRAPH_STATUS_HOST_NATIVE_ATOMICS |
          SPARK_TP4_GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED |
          SPARK_TP4_GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED;
      if (before.captured_nodes != captured_nodes ||
          before.captured_query_nodes != captured_query_nodes ||
          before.captured_combine_nodes != captured_combine_nodes ||
          before.published_sequence != 0 ||
          before.consumed_sequence != 0 ||
          before.completed_sequence != 0 ||
          before.overflow_sequence != 0 ||
          (before.flags & required_flags) != required_flags ||
          bool(
              before.flags &
              SPARK_TP4_GRAPH_STATUS_DEDICATED_SPIN) !=
              expected_dedicated_spin ||
          before.graph_submit_cpu_plus_one != options.submit_cpu + 1 ||
          before.graph_progress_cpu_plus_one !=
              options.progress_cpu + 1) {
        throw std::runtime_error(
            "invalid DCP graph pre-replay status: flags=" +
            std::to_string(before.flags) +
            " captured=" + std::to_string(before.captured_nodes) +
            " query=" +
            std::to_string(before.captured_query_nodes) +
            " combine=" +
            std::to_string(before.captured_combine_nodes) +
            " published=" +
            std::to_string(before.published_sequence) +
            " consumed=" +
            std::to_string(before.consumed_sequence) +
            " completed=" +
            std::to_string(before.completed_sequence) +
            " overflow=" +
            std::to_string(before.overflow_sequence) +
            " submit_plus_one=" +
            std::to_string(before.graph_submit_cpu_plus_one) +
            " progress_plus_one=" +
            std::to_string(before.graph_progress_cpu_plus_one));
      }

      for (int iteration = 0; iteration < options.warmup; ++iteration) {
        check_cuda(cudaGraphLaunch(executable, stream),
                   "launch DCP graph warmup");
      }
      check_cuda(cudaStreamSynchronize(stream),
                 "synchronize DCP graph warmup");

      check_cuda(cudaEventRecord(start, stream), "record start");
      const auto host_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < options.iterations;
           ++iteration) {
        check_cuda(cudaGraphLaunch(executable, stream),
                   "launch DCP graph measured");
      }
      const auto host_stop = std::chrono::steady_clock::now();
      check_cuda(cudaEventRecord(stop, stream), "record stop");
      check_cuda(cudaEventSynchronize(stop), "synchronize stop");

      float elapsed_ms{};
      check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                 "DCP graph elapsed time");
      const std::uint64_t replay_count =
          static_cast<std::uint64_t>(options.warmup + options.iterations);
      const std::uint64_t expected_sequence =
          replay_count * captured_nodes;
      const auto after =
          wait_for_completion(session.get(), expected_sequence);

      std::size_t query_mismatches{};
      ErrorStats output_error;
      ErrorStats lse_error;
      for (const auto& bucket : buckets) {
        if (
            options.single_q != 0 &&
            bucket.q != options.single_q
        ) {
          continue;
        }
        std::vector<std::uint16_t> actual_query(
            spark_transport::tp4_dcp_query_output_bytes(bucket.q) /
            sizeof(std::uint16_t));
        check_cuda(cudaMemcpy(
                       actual_query.data(), bucket.query_output,
                       actual_query.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost),
                   "copy DCP graph query output");
        for (std::uint32_t query_index = 0;
             query_index < bucket.q; ++query_index) {
          for (std::uint32_t source_rank = 0;
               source_rank < spark_transport::kTp4DcpQueryWorldSize;
               ++source_rank) {
            for (std::uint32_t local_head = 0;
                 local_head < spark_transport::kTp4DcpQueryHeadsPerRank;
                 ++local_head) {
              for (std::uint32_t dimension = 0;
                   dimension < spark_transport::kTp4DcpQueryHeadDimension;
                   ++dimension) {
                const std::size_t output_index = query_output_index(
                    query_index, source_rank, local_head, dimension);
                if (actual_query[output_index] !=
                    query_word(source_rank, query_index, local_head,
                               dimension)) {
                  ++query_mismatches;
                }
              }
            }
          }
        }

        std::vector<std::uint16_t> actual_output(
            spark_transport::tp4_dcp_combine_reduced_output_bytes(
                bucket.q, bucket.head_dimension) /
            sizeof(std::uint16_t));
        std::vector<float> actual_lse(
            spark_transport::tp4_dcp_combine_reduced_lse_bytes(bucket.q) /
            sizeof(float));
        check_cuda(cudaMemcpy(
                       actual_output.data(), bucket.combine_output,
                       actual_output.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost),
                   "copy DCP graph combine output");
        check_cuda(cudaMemcpy(
                       actual_lse.data(), bucket.combine_lse,
                       actual_lse.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "copy DCP graph combine LSE");
        for (std::uint32_t query_index = 0;
             query_index < bucket.q; ++query_index) {
          for (std::uint32_t local_head = 0;
               local_head < spark_transport::kTp4DcpCombineHeadsPerRank;
               ++local_head) {
            const std::uint32_t global_head =
                options.rank *
                    spark_transport::kTp4DcpCombineHeadsPerRank +
                local_head;
            const float expected_lse =
                global_lse(query_index, global_head);
            lse_error.observe(
                actual_lse[reduced_lse_index(query_index, local_head)],
                expected_lse);
            for (std::uint32_t dimension = 0;
                 dimension < bucket.head_dimension; ++dimension) {
              output_error.observe(
                  bf16_to_float(actual_output[reduced_output_index(
                      query_index, local_head, dimension,
                      bucket.head_dimension)]),
                  global_output(query_index, global_head, dimension,
                                expected_lse));
            }
          }
        }
      }

      const bool exact_status =
          after.captured_nodes == captured_nodes &&
          after.captured_query_nodes == captured_query_nodes &&
          after.captured_combine_nodes == captured_combine_nodes &&
          after.published_sequence == expected_sequence &&
          after.consumed_sequence == expected_sequence &&
          after.completed_sequence == expected_sequence &&
          after.overflow_sequence == 0 &&
          bool(
              after.flags &
              SPARK_TP4_GRAPH_STATUS_DEDICATED_SPIN) ==
              expected_dedicated_spin;
      passed =
          exact_status && query_mismatches == 0 &&
          output_error.nonfinite_mismatches == 0 &&
          lse_error.nonfinite_mismatches == 0 &&
          output_error.max_absolute <= options.max_output_abs &&
          lse_error.max_absolute <= options.max_lse_abs;
      const double host_submit_us =
          std::chrono::duration<double, std::micro>(
              host_stop - host_start)
              .count() /
          options.iterations;
      const double device_us =
          static_cast<double>(elapsed_ms) * 1000.0 /
          options.iterations;

      std::cout
          << std::fixed << std::setprecision(6)
          << "TP4_DCP_GRAPH"
          << " rank=" << options.rank
          << " buckets="
          << (
                 options.single_q == 0
                     ? std::string(
                           "1,2,3,4,5,6,8,10,12,16,20,24,32,40")
                     : std::to_string(options.single_q)
             )
          << " single_q="
          << (
                 options.single_q == 0
                     ? std::string("all")
                     : std::to_string(options.single_q)
             )
          << " captured_nodes=" << after.captured_nodes
          << " captured_query_nodes=" << after.captured_query_nodes
          << " captured_combine_nodes=" << after.captured_combine_nodes
          << " warmup=" << options.warmup
          << " iterations=" << options.iterations
          << " combine_dimension="
          << (
                 options.combine_dimension == 0
                     ? std::string("alternate")
                     : std::to_string(options.combine_dimension)
             )
          << " published=" << after.published_sequence
          << " consumed=" << after.consumed_sequence
          << " completed=" << after.completed_sequence
          << " overflow=" << after.overflow_sequence
          << " submit_cpu=" << options.submit_cpu
          << " progress_cpu=" << options.progress_cpu
          << " poll_policy=" << poll_policy
          << " dedicated_spin="
          << (expected_dedicated_spin ? "true" : "false")
          << " host_submit_us_per_graph=" << host_submit_us
          << " device_us_per_graph=" << device_us
          << " device_us_per_collective="
          << device_us / captured_nodes
          << " query_byte_mismatches=" << query_mismatches
          << " output_max_abs=" << output_error.max_absolute
          << " output_max_rel=" << output_error.max_relative
          << " output_nonfinite="
          << output_error.nonfinite_mismatches
          << " lse_max_abs=" << lse_error.max_absolute
          << " lse_max_rel=" << lse_error.max_relative
          << " lse_nonfinite=" << lse_error.nonfinite_mismatches
          << " passed=" << (passed ? "true" : "false") << '\n';

      check_cuda(cudaGraphExecDestroy(executable),
                 "destroy DCP graph executable");
      executable = nullptr;
      check_cuda(cudaGraphDestroy(graph), "destroy DCP graph");
      graph = nullptr;
    }

    for (auto& bucket : buckets) {
      cudaFree(bucket.combine_lse);
      cudaFree(bucket.combine_output);
      cudaFree(bucket.query_output);
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(combine_lse);
    for (void* combine_input : combine_inputs) {
      cudaFree(combine_input);
    }
    cudaFree(query_input);
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TP4_DCP_GRAPH_ERROR " << error.what() << '\n';
    return 1;
  }
}
