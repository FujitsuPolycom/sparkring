#include "spark_transport/gpu_tp4_dcp_combine.hpp"
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
#include <vector>

namespace {

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9890};
  std::uint16_t control_port1{9891};
  std::uint32_t q{};
  std::uint32_t head_dimension{256};
  bool token_major{};
  int warmup{4};
  int iterations{100};
  double max_output_abs{0.0625};
  double max_lse_abs{2.0e-5};
};

struct ErrorStats {
  double max_absolute{};
  double max_relative{};
  std::size_t nonfinite_mismatches{};
  std::size_t observations{};

  void observe(float actual, float expected) {
    ++observations;
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

struct ScalarState {
  float output{};
  float lse{};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n"
      << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
      << "         --control-port0 PORT --control-port1 PORT\n"
      << "         --q Q (1..5; omit to test all Q)\n"
      << "         --head-dimension 256|512 "
         "--layout head-major|token-major\n"
      << "         --warmup N --iterations N\n"
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
    } else if (argument == "--q") {
      options.q =
          static_cast<std::uint32_t>(unsigned_value(take_value(), "Q"));
    } else if (argument == "--head-dimension") {
      options.head_dimension = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "head dimension"));
    } else if (argument == "--layout") {
      const std::string_view layout(take_value());
      if (layout == "head-major") {
        options.token_major = false;
      } else if (layout == "token-major") {
        options.token_major = true;
      } else {
        usage(argv[0]);
      }
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iterations"));
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
  if (options.rank >= spark_transport::kTp4DcpCombineWorldSize ||
      options.peer0.empty() || options.peer1.empty() ||
      options.q > spark_transport::kTp4DcpCombineMaxQ ||
      !spark_transport::tp4_dcp_combine_head_dimension_supported(
          options.head_dimension) ||
      options.warmup < 0 || options.iterations <= 0) {
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
  const std::uint32_t round =
      0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + round) >> 16U);
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result{};
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float generated_output(std::uint32_t rank, std::uint32_t query_index,
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

float generated_lse(std::uint32_t rank, std::uint32_t query_index,
                    std::uint32_t global_head) {
  const std::uint32_t local_head =
      global_head % spark_transport::kTp4DcpCombineHeadsPerRank;
  const float infinity = std::numeric_limits<float>::infinity();
  if (local_head == 0U) {
    return -infinity;
  }
  if (local_head == 1U) {
    constexpr std::array<float, 4> finite_fallback{
        0.0F, 0.0F, 1.25F, 0.0F};
    if (rank == 0U) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    if (rank == 1U) {
      return infinity;
    }
    if (rank == 3U) {
      return -infinity;
    }
    return finite_fallback[rank] +
           static_cast<float>(query_index) * 0.125F;
  }
  if (local_head == 2U) {
    return rank < 2U ? infinity
                     : std::numeric_limits<float>::quiet_NaN();
  }
  const std::uint32_t seed =
      mix(0x243f6a88U ^ (rank * 0x9e3779b9U) ^
          (query_index * 0x85ebca6bU) ^ global_head);
  const std::int32_t centered =
      static_cast<std::int32_t>(seed % 12001U) - 6000;
  return static_cast<float>(centered) / 1000.0F;
}

float clean_lse(float value) {
  if (std::isnan(value) ||
      value == std::numeric_limits<float>::infinity()) {
    return -std::numeric_limits<float>::infinity();
  }
  return value;
}

ScalarState merge_state(ScalarState a, ScalarState b,
                        bool round_output_to_bf16) {
  a.lse = clean_lse(a.lse);
  b.lse = clean_lse(b.lse);
  const float negative_infinity =
      -std::numeric_limits<float>::infinity();
  if (a.lse == negative_infinity && b.lse == negative_infinity) {
    return {0.0F, negative_infinity};
  }
  const float maximum = std::max(a.lse, b.lse);
  const float weight_a = std::exp(a.lse - maximum);
  const float weight_b = std::exp(b.lse - maximum);
  const float denominator = weight_a + weight_b;
  const float weighted_a =
      weight_a == 0.0F ? 0.0F : weight_a * a.output;
  const float weighted_b =
      weight_b == 0.0F ? 0.0F : weight_b * b.output;
  float output =
      (weighted_a + weighted_b) / denominator;
  if (round_output_to_bf16) {
    output = bf16_to_float(float_to_bf16(output));
  }
  return {output, maximum + std::log(denominator)};
}

float direct_global_lse(std::uint32_t query_index,
                        std::uint32_t global_head) {
  std::array<float, 4> lses{};
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    lses[rank] = clean_lse(generated_lse(rank, query_index, global_head));
    maximum = std::max(maximum, lses[rank]);
  }
  if (maximum == -std::numeric_limits<float>::infinity()) {
    return maximum;
  }
  float denominator = 0.0F;
  for (float lse : lses) {
    denominator += std::exp(lse - maximum);
  }
  return maximum + std::log(denominator);
}

float direct_global_output(std::uint32_t query_index,
                           std::uint32_t global_head,
                           std::uint32_t dimension, float global_lse) {
  if (global_lse == -std::numeric_limits<float>::infinity()) {
    return 0.0F;
  }
  float result = 0.0F;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    const float value = bf16_to_float(float_to_bf16(generated_output(
        rank, query_index, global_head, dimension)));
    const float lse = clean_lse(generated_lse(
        rank, query_index, global_head));
    result += value * std::exp(lse - global_lse);
  }
  return result;
}

ScalarState pairwise_wire_reference(std::uint32_t query_index,
                                    std::uint32_t global_head,
                                    std::uint32_t dimension) {
  std::array<ScalarState, 4> states{};
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    states[rank] = {
        bf16_to_float(float_to_bf16(generated_output(
            rank, query_index, global_head, dimension))),
        generated_lse(rank, query_index, global_head),
    };
  }
  const auto pair01 = merge_state(states[0], states[1], true);
  const auto pair23 = merge_state(states[2], states[3], true);
  return merge_state(pair01, pair23, true);
}

float stock_semantic_output(std::uint32_t query_index,
                            std::uint32_t global_head,
                            std::uint32_t dimension,
                            float global_lse) {
  if (global_lse == -std::numeric_limits<float>::infinity()) {
    return 0.0F;
  }
  std::array<float, 4> corrected{};
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    const float value = bf16_to_float(float_to_bf16(generated_output(
        rank, query_index, global_head, dimension)));
    const float lse = clean_lse(generated_lse(
        rank, query_index, global_head));
    corrected[rank] = bf16_to_float(float_to_bf16(
        value * std::exp(lse - global_lse)));
  }
  const float pair01 = bf16_to_float(float_to_bf16(
      corrected[0] + corrected[1]));
  const float pair23 = bf16_to_float(float_to_bf16(
      corrected[2] + corrected[3]));
  return bf16_to_float(float_to_bf16(pair01 + pair23));
}

std::size_t input_output_index(std::uint32_t query_index,
                               std::uint32_t global_head,
                               std::uint32_t dimension,
                               std::uint32_t query_stride,
                               std::uint32_t head_stride) {
  return spark_transport::tp4_dcp_combine_strided_output_index(
      query_index, global_head, dimension, query_stride, head_stride);
}

std::size_t input_lse_index(std::uint32_t query_index,
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::uint32_t maximum_q =
        spark_transport::kTp4DcpCombineMaxQ;
    std::vector<std::uint16_t> host_output(
        spark_transport::tp4_dcp_combine_input_output_bytes(
            maximum_q, options.head_dimension) /
        sizeof(std::uint16_t));
    std::vector<float> host_lse(
        spark_transport::tp4_dcp_combine_input_lse_bytes(maximum_q) /
        sizeof(float));
    void* device_output{};
    void* device_lse{};
    void* device_reduced_output{};
    void* device_reduced_lse{};
    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaMalloc(&device_output,
                          host_output.size() * sizeof(std::uint16_t)),
               "cudaMalloc output input");
    check_cuda(cudaMalloc(&device_lse, host_lse.size() * sizeof(float)),
               "cudaMalloc LSE input");
    check_cuda(cudaMalloc(
                   &device_reduced_output,
                   spark_transport::tp4_dcp_combine_reduced_output_bytes(
                       maximum_q, options.head_dimension)),
               "cudaMalloc reduced output");
    check_cuda(cudaMalloc(
                   &device_reduced_lse,
                   spark_transport::tp4_dcp_combine_reduced_lse_bytes(
                       maximum_q)),
               "cudaMalloc reduced LSE");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream");
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    bool all_correct = true;
    {
      spark_tp4_dcp_config config{};
      config.rank = options.rank;
      config.peer0 = options.peer0.c_str();
      config.peer1 = options.peer1.c_str();
      config.device0 = options.device0.c_str();
      config.device1 = options.device1.c_str();
      config.gid0 = options.gid0;
      config.gid1 = options.gid1;
      config.control_port0 = options.control_port0;
      config.control_port1 = options.control_port1;
      char error[512]{};
      const DcpHandle session(
          spark_tp4_dcp_create(&config, error, sizeof(error)));
      if (session.get() == nullptr) {
        throw std::runtime_error(
            std::string("create DCP combine session: ") + error);
      }

      const std::uint32_t first_q = options.q == 0 ? 1 : options.q;
      const std::uint32_t last_q =
          options.q == 0 ? maximum_q : options.q;
      for (std::uint32_t q = first_q; q <= last_q; ++q) {
        const std::uint32_t query_stride =
            options.token_major
                ? spark_transport::kTp4DcpCombineGlobalHeads *
                      options.head_dimension
                : options.head_dimension;
        const std::uint32_t head_stride =
            options.token_major ? options.head_dimension
                                : q * options.head_dimension;
        for (std::uint32_t query_index = 0; query_index < q;
             ++query_index) {
          for (std::uint32_t global_head = 0;
               global_head <
               spark_transport::kTp4DcpCombineGlobalHeads;
               ++global_head) {
            host_lse[input_lse_index(query_index, global_head)] =
                generated_lse(options.rank, query_index, global_head);
            for (std::uint32_t dimension = 0;
                 dimension < options.head_dimension;
                 ++dimension) {
              host_output[input_output_index(
                  query_index, global_head, dimension, query_stride,
                  head_stride)] =
                  float_to_bf16(generated_output(
                      options.rank, query_index, global_head,
                      dimension));
            }
          }
        }
        check_cuda(cudaMemcpy(
                       device_output, host_output.data(),
                       spark_transport::tp4_dcp_combine_input_output_bytes(
                           q, options.head_dimension),
                       cudaMemcpyHostToDevice),
                   "copy head-major output input");
        check_cuda(cudaMemcpy(
                       device_lse, host_lse.data(),
                       spark_transport::tp4_dcp_combine_input_lse_bytes(q),
                       cudaMemcpyHostToDevice),
                   "copy LSE input");

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
          if (spark_tp4_dcp_combine(
                  session.get(), device_output, device_lse,
                  device_reduced_output, device_reduced_lse, q,
                  options.head_dimension, query_stride, head_stride, stream,
                  error, sizeof(error)) != 0) {
            throw std::runtime_error(
                std::string("warmup DCP combine: ") + error);
          }
        }
        check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

        const auto host_start = std::chrono::steady_clock::now();
        check_cuda(cudaEventRecord(start, stream), "record start");
        for (int iteration = 0; iteration < options.iterations;
             ++iteration) {
          if (spark_tp4_dcp_combine(
                  session.get(), device_output, device_lse,
                  device_reduced_output, device_reduced_lse, q,
                  options.head_dimension, query_stride, head_stride, stream,
                  error, sizeof(error)) != 0) {
            throw std::runtime_error(
                std::string("measure DCP combine: ") + error);
          }
        }
        check_cuda(cudaEventRecord(stop, stream), "record stop");
        const auto host_submit_stop = std::chrono::steady_clock::now();
        check_cuda(cudaEventSynchronize(stop), "measurement synchronize");
        const auto host_finish = std::chrono::steady_clock::now();

        float device_ms{};
        check_cuda(cudaEventElapsedTime(&device_ms, start, stop),
                   "elapsed time");
        std::vector<std::uint16_t> actual_output(
            spark_transport::tp4_dcp_combine_reduced_output_bytes(
                q, options.head_dimension) /
            sizeof(std::uint16_t));
        std::vector<float> actual_lse(
            spark_transport::tp4_dcp_combine_reduced_lse_bytes(q) /
            sizeof(float));
        check_cuda(cudaMemcpy(
                       actual_output.data(), device_reduced_output,
                       actual_output.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost),
                   "copy reduced output");
        check_cuda(cudaMemcpy(
                       actual_lse.data(), device_reduced_lse,
                       actual_lse.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "copy reduced LSE");

        ErrorStats direct_output_error;
        ErrorStats stock_output_error;
        ErrorStats wire_output_error;
        ErrorStats direct_lse_error;
        ErrorStats wire_lse_error;
        std::size_t wire_bf16_mismatches{};
        std::size_t exceptional_mismatches{};
        for (std::uint32_t query_index = 0; query_index < q;
             ++query_index) {
          for (std::uint32_t local_head = 0;
               local_head < spark_transport::kTp4DcpCombineHeadsPerRank;
               ++local_head) {
            const std::uint32_t global_head =
                options.rank *
                    spark_transport::kTp4DcpCombineHeadsPerRank +
                local_head;
            const float global_lse =
                direct_global_lse(query_index, global_head);
            const auto wire_lse = pairwise_wire_reference(
                query_index, global_head, 0);
            const float native_lse =
                actual_lse[reduced_lse_index(query_index, local_head)];
            direct_lse_error.observe(native_lse, global_lse);
            wire_lse_error.observe(native_lse, wire_lse.lse);
            for (std::uint32_t dimension = 0;
                 dimension < options.head_dimension;
                 ++dimension) {
              const std::size_t index = reduced_output_index(
                  query_index, local_head, dimension,
                  options.head_dimension);
              const float native_output =
                  bf16_to_float(actual_output[index]);
              const float direct_output = direct_global_output(
                  query_index, global_head, dimension, global_lse);
              const auto wire = pairwise_wire_reference(
                  query_index, global_head, dimension);
              const float stock = stock_semantic_output(
                  query_index, global_head, dimension, global_lse);
              direct_output_error.observe(native_output, direct_output);
              wire_output_error.observe(native_output, wire.output);
              stock_output_error.observe(native_output, stock);
              if (actual_output[index] != float_to_bf16(wire.output)) {
                ++wire_bf16_mismatches;
              }
              if ((local_head == 0U || local_head == 2U) &&
                  (actual_output[index] != float_to_bf16(0.0F) ||
                   native_lse !=
                       -std::numeric_limits<float>::infinity())) {
                ++exceptional_mismatches;
              }
            }
          }
        }

        const bool correct =
            direct_output_error.nonfinite_mismatches == 0 &&
            direct_lse_error.nonfinite_mismatches == 0 &&
            exceptional_mismatches == 0 &&
            direct_output_error.max_absolute <= options.max_output_abs &&
            direct_lse_error.max_absolute <= options.max_lse_abs;
        all_correct = all_correct && correct;
        const double device_us_per_call =
            static_cast<double>(device_ms) * 1000.0 /
            options.iterations;
        const double submit_us_per_call =
            std::chrono::duration<double, std::micro>(
                host_submit_stop - host_start)
                .count() /
            options.iterations;
        const double wall_us_per_call =
            std::chrono::duration<double, std::micro>(
                host_finish - host_start)
                .count() /
            options.iterations;
        std::cout
            << std::fixed << std::setprecision(6)
            << "TP4_DCP_COMBINE"
            << " rank=" << options.rank << " q=" << q
            << " head_dimension=" << options.head_dimension
            << " layout="
            << (options.token_major ? "token-major" : "head-major")
            << " round0_bytes="
            << spark_transport::tp4_dcp_combine_round0_frame(
                   q, options.head_dimension)
                   .total_bytes
            << " round1_bytes="
            << spark_transport::tp4_dcp_combine_round1_frame(
                   q, options.head_dimension)
                   .total_bytes
            << " iterations=" << options.iterations
            << " device_us_per_call=" << device_us_per_call
            << " host_submit_us_per_call=" << submit_us_per_call
            << " wall_us_per_call=" << wall_us_per_call
            << " direct_output_max_abs="
            << direct_output_error.max_absolute
            << " direct_output_max_rel="
            << direct_output_error.max_relative
            << " direct_lse_max_abs=" << direct_lse_error.max_absolute
            << " direct_lse_max_rel=" << direct_lse_error.max_relative
            << " wire_output_max_abs="
            << wire_output_error.max_absolute
            << " wire_lse_max_abs=" << wire_lse_error.max_absolute
            << " wire_bf16_mismatches=" << wire_bf16_mismatches
            << " stock_output_max_abs="
            << stock_output_error.max_absolute
            << " exceptional_mismatches=" << exceptional_mismatches
            << " nonfinite_mismatches="
            << (direct_output_error.nonfinite_mismatches +
                direct_lse_error.nonfinite_mismatches)
            << " correct=" << (correct ? "true" : "false") << '\n';
      }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(device_reduced_lse);
    cudaFree(device_reduced_output);
    cudaFree(device_lse);
    cudaFree(device_output);
    return all_correct ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TP4_DCP_COMBINE_ERROR " << error.what() << '\n';
    return 1;
  }
}
