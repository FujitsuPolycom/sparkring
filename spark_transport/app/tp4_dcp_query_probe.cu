#include "spark_transport/gpu_tp4_dcp_query.hpp"
#include "spark_transport/tp4_c_api.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
  int warmup{4};
  int iterations{100};
  bool alternate_streams{};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n"
      << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
      << "         --control-port0 PORT --control-port1 PORT\n"
      << "         --q Q (1..40; omit to test all Q)\n"
      << "         --warmup N --iterations N --alternate-streams\n";
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
    } else if (argument == "--q") {
      options.q =
          static_cast<std::uint32_t>(unsigned_value(take_value(), "Q"));
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iterations"));
    } else if (argument == "--alternate-streams") {
      options.alternate_streams = true;
    } else {
      usage(argv[0]);
    }
  }
  if (options.rank >= spark_transport::kTp4DcpQueryWorldSize ||
      options.peer0.empty() || options.peer1.empty() ||
      options.q > spark_transport::kTp4DcpQueryMaxQ ||
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

class DcpQueryHandle {
 public:
  explicit DcpQueryHandle(spark_tp4_dcp_handle handle)
      : handle_(handle) {}
  DcpQueryHandle(const DcpQueryHandle&) = delete;
  DcpQueryHandle& operator=(const DcpQueryHandle&) = delete;
  ~DcpQueryHandle() { spark_tp4_dcp_destroy(handle_); }

  spark_tp4_dcp_handle get() const { return handle_; }

 private:
  spark_tp4_dcp_handle handle_{};
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::size_t max_input_words =
        spark_transport::tp4_dcp_query_input_bytes(
            spark_transport::kTp4DcpQueryMaxQ) /
        sizeof(std::uint16_t);
    const std::size_t max_output_bytes =
        spark_transport::tp4_dcp_query_output_bytes(
            spark_transport::kTp4DcpQueryMaxQ);

    std::vector<std::uint16_t> host_input(max_input_words);
    for (std::uint32_t query_index = 0;
         query_index < spark_transport::kTp4DcpQueryMaxQ;
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
    void* output{};
    std::array<cudaStream_t, 2> streams{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaMalloc(&input, host_input.size() * sizeof(std::uint16_t)),
               "cudaMalloc input");
    check_cuda(cudaMalloc(&output, max_output_bytes), "cudaMalloc output");
    check_cuda(cudaMemcpy(input, host_input.data(),
                          host_input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy input");
    check_cuda(
        cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking),
        "create stream 0");
    check_cuda(
        cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking),
        "create stream 1");
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    std::size_t total_mismatches{};
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
      const DcpQueryHandle session(
          spark_tp4_dcp_create(&config, error, sizeof(error)));
      if (session.get() == nullptr) {
        throw std::runtime_error(
            std::string("create DCP query session: ") + error);
      }

      const std::uint32_t first_q = options.q == 0 ? 1 : options.q;
      const std::uint32_t last_q =
          options.q == 0 ? spark_transport::kTp4DcpQueryMaxQ : options.q;
      for (std::uint32_t q = first_q; q <= last_q; ++q) {
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
          const auto stream = streams[
              options.alternate_streams
                  ? static_cast<std::size_t>(iteration) % streams.size()
                  : 0];
          if (spark_tp4_dcp_query_all_gather(
                  session.get(), input, output, q, stream, error,
                  sizeof(error)) != 0) {
            throw std::runtime_error(
                std::string("warmup DCP query: ") + error);
          }
        }
        check_cuda(
            cudaStreamSynchronize(streams[0]),
            "warmup synchronize stream 0");
        check_cuda(
            cudaStreamSynchronize(streams[1]),
            "warmup synchronize stream 1");

        const auto host_start = std::chrono::steady_clock::now();
        check_cuda(cudaEventRecord(start, streams[0]), "record start");
        for (int iteration = 0; iteration < options.iterations;
             ++iteration) {
          const auto stream = streams[
              options.alternate_streams
                  ? static_cast<std::size_t>(iteration) % streams.size()
                  : 0];
          if (spark_tp4_dcp_query_all_gather(
                  session.get(), input, output, q, stream, error,
                  sizeof(error)) != 0) {
            throw std::runtime_error(
                std::string("measure DCP query: ") + error);
          }
        }
        const auto stop_stream = streams[
            options.alternate_streams
                ? static_cast<std::size_t>(options.iterations - 1) %
                      streams.size()
                : 0];
        check_cuda(cudaEventRecord(stop, stop_stream), "record stop");
        const auto host_submit_stop = std::chrono::steady_clock::now();
        check_cuda(cudaEventSynchronize(stop), "measurement synchronize");
        const auto host_finish = std::chrono::steady_clock::now();

        float device_ms{};
        check_cuda(cudaEventElapsedTime(&device_ms, start, stop),
                   "elapsed time");
        const std::size_t output_bytes =
            spark_transport::tp4_dcp_query_output_bytes(q);
        std::vector<std::uint16_t> host_output(
            output_bytes / sizeof(std::uint16_t));
        check_cuda(cudaMemcpy(host_output.data(), output, output_bytes,
                              cudaMemcpyDeviceToHost),
                   "copy output");

        std::size_t mismatches{};
        constexpr std::uint32_t global_heads =
            spark_transport::kTp4DcpQueryHeadsPerRank *
            spark_transport::kTp4DcpQueryWorldSize;
        for (std::uint32_t query_index = 0; query_index < q;
             ++query_index) {
          for (std::uint32_t global_head = 0;
               global_head < global_heads; ++global_head) {
            const std::uint32_t source_rank =
                global_head /
                spark_transport::kTp4DcpQueryHeadsPerRank;
            const std::uint32_t local_head =
                global_head %
                spark_transport::kTp4DcpQueryHeadsPerRank;
            for (std::uint32_t dimension = 0;
                 dimension <
                 spark_transport::kTp4DcpQueryHeadDimension;
                 ++dimension) {
              const auto actual =
                  host_output[output_index(query_index, global_head,
                                           dimension)];
              const auto expected = expected_word(
                  source_rank, query_index, local_head, dimension);
              if (actual != expected) {
                ++mismatches;
              }
            }
          }
        }
        total_mismatches += mismatches;

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
        std::cout << std::fixed << std::setprecision(3)
                  << "TP4_DCP_QUERY"
                  << " rank=" << options.rank << " q=" << q
                  << " input_bytes="
                  << spark_transport::tp4_dcp_query_input_bytes(q)
                  << " output_bytes=" << output_bytes
                  << " iterations=" << options.iterations
                  << " alternate_streams="
                  << (options.alternate_streams ? 1 : 0)
                  << " device_us_per_call=" << device_us_per_call
                  << " host_submit_us_per_call=" << submit_us_per_call
                  << " wall_us_per_call=" << wall_us_per_call
                  << " mismatches=" << mismatches << '\n';
      }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[0]);
    cudaFree(output);
    cudaFree(input);
    return total_mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TP4_DCP_QUERY_ERROR " << error.what() << '\n';
    return 1;
  }
}
