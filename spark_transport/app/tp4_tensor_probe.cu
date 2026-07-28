#include "spark_transport/statistics.hpp"
#include "spark_transport/tp4_session.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct Options {
  spark_transport::Tp4AllreduceOptions transport;
  int warmup{100};
  int iterations{1000};
  bool alternate_streams{};
  std::uint64_t queued_delay_ms{};
  std::uint32_t queued_delay_rank{4};
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
      << "  --bytes BYTES\n"
      << "  --warmup COUNT\n"
      << "  --iterations COUNT\n"
      << "  --queued-delay-ms MILLISECONDS\n"
      << "  --queued-delay-rank RANK\n"
      << "  --alternate-streams\n";
  std::exit(2);
}

std::uint64_t unsigned_value(const char* value, const char* name) {
  std::size_t consumed = 0;
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
    } else if (argument == "--bytes") {
      options.transport.payload_bytes =
          unsigned_value(take_value(), "payload size");
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup count"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iteration count"));
    } else if (argument == "--queued-delay-ms") {
      options.queued_delay_ms =
          unsigned_value(take_value(), "queued delay");
    } else if (argument == "--queued-delay-rank") {
      options.queued_delay_rank = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "queued delay rank"));
    } else if (argument == "--alternate-streams") {
      options.alternate_streams = true;
    } else {
      usage(argv[0]);
    }
  }

  if (options.transport.rank >= 4 || options.transport.peer0.empty() ||
      options.transport.peer1.empty() ||
      options.transport.payload_bytes == 0 ||
      options.transport.payload_bytes % sizeof(__nv_bfloat16) != 0 ||
      options.warmup < 0 || options.iterations <= 0) {
    usage(argv[0]);
  }
  if (options.queued_delay_ms != 0 &&
      (!options.alternate_streams || options.warmup < 1 ||
       options.iterations < 2 || options.queued_delay_rank >= 4)) {
    throw std::invalid_argument(
        "--queued-delay-ms requires --queued-delay-rank 0..3, "
        "--alternate-streams, --warmup >= 1, and --iterations >= 2");
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

__device__ float input_value(std::uint64_t sequence, std::uint32_t rank,
                             std::size_t element) {
  return static_cast<float>(
      ((sequence + element * 3U + rank * 5U) & 7U) + 1U);
}

__global__ void fill_input(__nv_bfloat16* input, std::size_t elements,
                           std::uint64_t sequence, std::uint32_t rank) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < elements) {
    input[index] = __float2bfloat16(input_value(sequence, rank, index));
  }
}

__global__ void validate_output(const __nv_bfloat16* output,
                                std::size_t elements,
                                std::uint64_t sequence,
                                unsigned long long* mismatches) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= elements) {
    return;
  }
  float expected = 0.0F;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    expected += input_value(sequence, rank, index);
  }
  if (__bfloat162float(output[index]) != expected) {
    atomicAdd(mismatches, 1ULL);
  }
}

void CUDART_CB delay_stream(void* duration_pointer) {
  const auto duration =
      *static_cast<const std::chrono::milliseconds*>(duration_pointer);
  std::this_thread::sleep_for(duration);
}

double now_microseconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double, std::micro>(now).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::size_t elements =
        options.transport.payload_bytes / sizeof(__nv_bfloat16);
    constexpr int threads = 256;
    const int blocks =
        static_cast<int>((elements + threads - 1) / threads);

    __nv_bfloat16* inputs[2]{};
    __nv_bfloat16* outputs[2]{};
    unsigned long long* mismatches{};
    cudaStream_t stream0{};
    cudaStream_t stream1{};
    check_cuda(cudaMalloc(&inputs[0], options.transport.payload_bytes),
               "cudaMalloc external input 0");
    check_cuda(cudaMalloc(&outputs[0], options.transport.payload_bytes),
               "cudaMalloc external output 0");
    if (options.alternate_streams) {
      check_cuda(cudaMalloc(&inputs[1], options.transport.payload_bytes),
                 "cudaMalloc external input 1");
      check_cuda(cudaMalloc(&outputs[1], options.transport.payload_bytes),
                 "cudaMalloc external output 1");
    }
    check_cuda(cudaMalloc(&mismatches, sizeof(*mismatches)),
               "cudaMalloc mismatch counter");
    check_cuda(cudaMemset(mismatches, 0, sizeof(*mismatches)),
               "cudaMemset mismatch counter");
    check_cuda(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags stream 0");
    if (options.alternate_streams) {
      check_cuda(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags stream 1");
    }

    unsigned long long host_mismatches{};
    {
      // The session retains its current caller stream and synchronizes it
      // during destruction. Destroy the session before destroying either
      // CUDA stream.
      spark_transport::Tp4AllreduceSession session(options.transport);
      const std::uint64_t total =
          static_cast<std::uint64_t>(options.warmup) +
          static_cast<std::uint64_t>(options.iterations);
      std::vector<double> latencies;
      latencies.reserve(options.iterations);
      std::chrono::milliseconds queued_delay(options.queued_delay_ms);
      const std::uint64_t queued_delay_sequence =
          options.queued_delay_ms == 0
              ? 0
              : static_cast<std::uint64_t>(options.warmup) + 1;

      for (std::uint64_t sequence = 1; sequence <= total; ++sequence) {
        const std::size_t stream_index =
            options.alternate_streams && sequence % 2 == 0 ? 1 : 0;
        const cudaStream_t stream =
            stream_index == 0 ? stream0 : stream1;
        fill_input<<<blocks, threads, 0, stream>>>(
            inputs[stream_index], elements, sequence,
            options.transport.rank);
        check_cuda(cudaGetLastError(), "fill_input launch");
        if (sequence == queued_delay_sequence &&
            options.transport.rank == options.queued_delay_rank) {
          check_cuda(
              cudaLaunchHostFunc(stream, delay_stream, &queued_delay),
              "cudaLaunchHostFunc queued delay");
        }

        const double start = now_microseconds();
        session.all_reduce(
            inputs[stream_index], outputs[stream_index], stream);
        const double complete = now_microseconds();

        validate_output<<<blocks, threads, 0, stream>>>(
            outputs[stream_index], elements, sequence, mismatches);
        check_cuda(cudaGetLastError(), "validate_output launch");

        if (sequence > static_cast<std::uint64_t>(options.warmup)) {
          latencies.push_back(complete - start);
        }
      }

      const cudaStream_t final_stream =
          options.alternate_streams && total % 2 == 0 ? stream1 : stream0;
      check_cuda(cudaStreamSynchronize(final_stream),
                 "probe stream synchronize");
      check_cuda(cudaMemcpy(&host_mismatches, mismatches,
                            sizeof(host_mismatches), cudaMemcpyDeviceToHost),
                 "copy mismatch counter");

      const auto summary =
          spark_transport::summarize_latencies(std::move(latencies));
      std::cout << "TP4_TENSOR"
                << " rank=" << options.transport.rank
                << " bytes=" << options.transport.payload_bytes
                << " iterations=" << options.iterations
                << " alternate_streams="
                << (options.alternate_streams ? "true" : "false")
                << " queued_delay_ms=" << options.queued_delay_ms
                << " queued_delay_rank=" << options.queued_delay_rank
                << " queued_delay_applied="
                << (options.queued_delay_ms != 0 &&
                            options.transport.rank ==
                                options.queued_delay_rank
                        ? "true"
                        : "false")
                << " queued_delay_sequence=" << queued_delay_sequence
                << " mismatched_elements=" << host_mismatches
                << " correct=" << (host_mismatches == 0 ? "true" : "false")
                << " p50_us=" << summary.p50_us
                << " p95_us=" << summary.p95_us
                << " p99_us=" << summary.p99_us
                << " max_us=" << summary.maximum_us << '\n';
    }

    if (stream1 != nullptr) {
      cudaStreamDestroy(stream1);
    }
    cudaStreamDestroy(stream0);
    cudaFree(mismatches);
    if (outputs[1] != nullptr) {
      cudaFree(outputs[1]);
    }
    if (inputs[1] != nullptr) {
      cudaFree(inputs[1]);
    }
    cudaFree(outputs[0]);
    cudaFree(inputs[0]);
    return host_mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
