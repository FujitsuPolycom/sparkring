#include "spark_transport/tp4_allgather_session.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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
  std::uint16_t control_port0{9490};
  std::uint16_t control_port1{9491};
  std::size_t input_bytes{};
  int warmup{4};
  int iterations{100};
  std::uint64_t queued_delay_ms{};
  std::uint32_t queued_delay_rank{4};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP --bytes BYTES [options]\n"
      << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
      << "         --control-port0 PORT --control-port1 PORT\n"
      << "         --warmup N --iterations N\n"
      << "         --queued-delay-ms MILLISECONDS\n"
      << "         --queued-delay-rank RANK\n";
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
      options.rank =
          static_cast<std::uint32_t>(unsigned_value(take_value(), "rank"));
    } else if (argument == "--peer0") {
      options.peer0 = take_value();
    } else if (argument == "--peer1") {
      options.peer1 = take_value();
    } else if (argument == "--device0") {
      options.device0 = take_value();
    } else if (argument == "--device1") {
      options.device1 = take_value();
    } else if (argument == "--gid0") {
      options.gid0 =
          static_cast<std::uint8_t>(unsigned_value(take_value(), "GID 0"));
    } else if (argument == "--gid1") {
      options.gid1 =
          static_cast<std::uint8_t>(unsigned_value(take_value(), "GID 1"));
    } else if (argument == "--control-port0") {
      options.control_port0 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 0"));
    } else if (argument == "--control-port1") {
      options.control_port1 = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port 1"));
    } else if (argument == "--bytes") {
      options.input_bytes =
          static_cast<std::size_t>(unsigned_value(take_value(), "bytes"));
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iterations"));
    } else if (argument == "--queued-delay-ms") {
      options.queued_delay_ms =
          unsigned_value(take_value(), "queued delay");
    } else if (argument == "--queued-delay-rank") {
      options.queued_delay_rank = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "queued delay rank"));
    } else {
      usage(argv[0]);
    }
  }
  if (options.rank >= 4 || options.peer0.empty() ||
      options.peer1.empty() || options.input_bytes == 0 ||
      options.input_bytes % 16 != 0 || options.warmup < 0 ||
      options.iterations <= 0) {
    usage(argv[0]);
  }
  if (options.queued_delay_ms != 0 &&
      (options.queued_delay_ms < 5500 ||
       options.queued_delay_rank >= 4 || options.warmup < 1 ||
       options.iterations < 2)) {
    throw std::invalid_argument(
        "--queued-delay-ms must be at least 5500 and requires "
        "--queued-delay-rank 0..3, "
        "--warmup >= 1, and --iterations >= 2");
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

std::uint8_t expected_byte(std::uint32_t rank, std::size_t index) {
  return static_cast<std::uint8_t>(
      (rank * 37U + static_cast<std::uint32_t>(index % 251U)) & 0xffU);
}

void CUDART_CB delay_stream(void* duration_pointer) {
  const auto duration =
      *static_cast<const std::chrono::milliseconds*>(duration_pointer);
  std::this_thread::sleep_for(duration);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::vector<std::uint8_t> host_input(options.input_bytes);
    for (std::size_t index = 0; index < host_input.size(); ++index) {
      host_input[index] = expected_byte(options.rank, index);
    }

    void* input{};
    void* output{};
    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaMalloc(&input, options.input_bytes), "cudaMalloc input");
    check_cuda(cudaMalloc(&output, options.input_bytes * 4),
               "cudaMalloc output");
    check_cuda(cudaMemcpy(input, host_input.data(), options.input_bytes,
                          cudaMemcpyHostToDevice),
               "copy input");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream");
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    std::size_t mismatches{};
    {
      // The session retains the caller stream so it can drain outstanding
      // stream-ordered work during destruction. It must therefore die before
      // cudaStreamDestroy below.
      spark_transport::Tp4AllgatherOptions session_options{};
      session_options.rank = options.rank;
      session_options.peer0 = options.peer0;
      session_options.peer1 = options.peer1;
      session_options.device0 = options.device0;
      session_options.device1 = options.device1;
      session_options.gid0 = options.gid0;
      session_options.gid1 = options.gid1;
      session_options.control_port0 = options.control_port0;
      session_options.control_port1 = options.control_port1;
      session_options.input_bytes = options.input_bytes;
      spark_transport::Tp4AllgatherSession session(
          std::move(session_options));
      std::chrono::milliseconds queued_delay(options.queued_delay_ms);
      const std::uint64_t queued_delay_sequence =
          options.queued_delay_ms == 0
              ? 0
              : static_cast<std::uint64_t>(options.warmup) + 1;

      for (int iteration = 0; iteration < options.warmup; ++iteration) {
        session.all_gather(input, output, stream);
      }
      check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

      const auto host_start = std::chrono::steady_clock::now();
      check_cuda(cudaEventRecord(start, stream), "record start");
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        const std::uint64_t submission_sequence =
            static_cast<std::uint64_t>(options.warmup) +
            static_cast<std::uint64_t>(iteration) + 1;
        if (submission_sequence == queued_delay_sequence &&
            options.rank == options.queued_delay_rank) {
          check_cuda(
              cudaLaunchHostFunc(stream, delay_stream, &queued_delay),
              "cudaLaunchHostFunc queued delay");
        }
        session.all_gather(input, output, stream);
      }
      check_cuda(cudaEventRecord(stop, stream), "record stop");
      const auto host_submit_stop = std::chrono::steady_clock::now();
      check_cuda(cudaEventSynchronize(stop), "measurement synchronize");
      const auto host_finish = std::chrono::steady_clock::now();

      float device_ms{};
      check_cuda(cudaEventElapsedTime(&device_ms, start, stop),
                 "elapsed time");
      std::vector<std::uint8_t> host_output(options.input_bytes * 4);
      check_cuda(cudaMemcpy(host_output.data(), output, host_output.size(),
                            cudaMemcpyDeviceToHost),
                 "copy output");
      for (std::uint32_t rank = 0; rank < 4; ++rank) {
        for (std::size_t index = 0; index < options.input_bytes; ++index) {
          const auto actual =
              host_output[rank * options.input_bytes + index];
          if (actual != expected_byte(rank, index)) {
            ++mismatches;
          }
        }
      }

      const double device_us_per_call =
          static_cast<double>(device_ms) * 1000.0 / options.iterations;
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
                << "TP4_ALLGATHER"
                << " rank=" << options.rank
                << " input_bytes=" << options.input_bytes
                << " output_bytes=" << options.input_bytes * 4
                << " iterations=" << options.iterations
                << " queued_delay_ms=" << options.queued_delay_ms
                << " queued_delay_rank=" << options.queued_delay_rank
                << " queued_delay_applied="
                << (options.queued_delay_ms != 0 &&
                            options.rank == options.queued_delay_rank
                        ? "true"
                        : "false")
                << " queued_delay_sequence=" << queued_delay_sequence
                << " device_us_per_call=" << device_us_per_call
                << " host_submit_us_per_call=" << submit_us_per_call
                << " wall_us_per_call=" << wall_us_per_call
                << " mismatches=" << mismatches << '\n';
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(output);
    cudaFree(input);
    return mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TP4_ALLGATHER_ERROR " << error.what() << '\n';
    return 1;
  }
}
