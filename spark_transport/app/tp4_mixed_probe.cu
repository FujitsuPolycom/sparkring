#include "spark_transport/tp4_allgather_session.hpp"
#include "spark_transport/tp4_session.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t kAllreduceBytes = 12 * 1024;
constexpr std::size_t kAllgatherBytes = 16 * 1024;

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  int prewarm{32};
  int iterations{256};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " --rank RANK --peer0 IP --peer1 IP [options]\n"
            << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
            << "         --prewarm N --iterations N\n";
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
    } else if (argument == "--prewarm") {
      options.prewarm =
          static_cast<int>(unsigned_value(take_value(), "prewarm"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iterations"));
    } else {
      usage(argv[0]);
    }
  }
  if (options.rank >= 4 || options.peer0.empty() ||
      options.peer1.empty() || options.prewarm < 0 ||
      options.iterations <= 0) {
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

__device__ std::uint8_t gather_value(std::uint32_t rank,
                                     std::uint64_t sequence,
                                     std::size_t index) {
  return static_cast<std::uint8_t>(
      (rank * 37U + sequence * 11U + index % 251U) & 0xffU);
}

__global__ void fill_gather(std::uint8_t* input, std::uint32_t rank,
                            std::uint64_t sequence) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < kAllgatherBytes) {
    input[index] = gather_value(rank, sequence, index);
  }
}

__global__ void validate_gather(const std::uint8_t* output,
                                std::uint64_t sequence,
                                unsigned long long* mismatches) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= kAllgatherBytes * 4) {
    return;
  }
  const auto rank = static_cast<std::uint32_t>(index / kAllgatherBytes);
  const auto rank_index = index % kAllgatherBytes;
  if (output[index] != gather_value(rank, sequence, rank_index)) {
    atomicAdd(mismatches, 1ULL);
  }
}

__device__ float reduce_value(std::uint32_t rank, std::uint64_t sequence,
                              std::size_t index) {
  return static_cast<float>((rank * 5U + sequence + index * 3U) % 8U + 1U);
}

__global__ void fill_reduce(__nv_bfloat16* input, std::uint32_t rank,
                            std::uint64_t sequence) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < kAllreduceBytes / sizeof(__nv_bfloat16)) {
    input[index] = __float2bfloat16(reduce_value(rank, sequence, index));
  }
}

__global__ void validate_reduce(const __nv_bfloat16* output,
                                std::uint64_t sequence,
                                unsigned long long* mismatches) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= kAllreduceBytes / sizeof(__nv_bfloat16)) {
    return;
  }
  float expected{};
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    expected += reduce_value(rank, sequence, index);
  }
  if (__bfloat162float(output[index]) != expected) {
    atomicAdd(mismatches, 1ULL);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    cudaStream_t stream{};
    std::uint8_t* gather_input{};
    std::uint8_t* gather_output{};
    __nv_bfloat16* reduce_input{};
    __nv_bfloat16* reduce_output{};
    unsigned long long* mismatches{};
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream");
    check_cuda(cudaMalloc(&gather_input, kAllgatherBytes),
               "allocate gather input");
    check_cuda(cudaMalloc(&gather_output, kAllgatherBytes * 4),
               "allocate gather output");
    check_cuda(cudaMalloc(&reduce_input, kAllreduceBytes),
               "allocate reduce input");
    check_cuda(cudaMalloc(&reduce_output, kAllreduceBytes),
               "allocate reduce output");
    check_cuda(cudaMalloc(&mismatches, sizeof(*mismatches)),
               "allocate mismatch counter");
    check_cuda(cudaMemset(mismatches, 0, sizeof(*mismatches)),
               "clear mismatch counter");

    spark_transport::Tp4AllgatherOptions gather_options{};
    gather_options.rank = options.rank;
    gather_options.peer0 = options.peer0;
    gather_options.peer1 = options.peer1;
    gather_options.device0 = options.device0;
    gather_options.device1 = options.device1;
    gather_options.gid0 = options.gid0;
    gather_options.gid1 = options.gid1;
    gather_options.control_port0 = 9590;
    gather_options.control_port1 = 9591;
    gather_options.input_bytes = kAllgatherBytes;

    {
      spark_transport::Tp4AllgatherSession gather(gather_options);
      constexpr int threads = 256;
      constexpr int gather_input_blocks =
          (kAllgatherBytes + threads - 1) / threads;
      constexpr int gather_output_blocks =
          (kAllgatherBytes * 4 + threads - 1) / threads;
      constexpr int reduce_blocks =
          (kAllreduceBytes / sizeof(__nv_bfloat16) + threads - 1) / threads;

      for (int iteration = 1; iteration <= options.prewarm; ++iteration) {
        fill_gather<<<gather_input_blocks, threads, 0, stream>>>(
            gather_input, options.rank,
            static_cast<std::uint64_t>(iteration));
        gather.all_gather(gather_input, gather_output, stream);
        validate_gather<<<gather_output_blocks, threads, 0, stream>>>(
            gather_output, static_cast<std::uint64_t>(iteration),
            mismatches);
      }
      check_cuda(cudaStreamSynchronize(stream), "gather prewarm synchronize");

      spark_transport::Tp4AllreduceOptions reduce_options{};
      reduce_options.rank = options.rank;
      reduce_options.peer0 = options.peer0;
      reduce_options.peer1 = options.peer1;
      reduce_options.device0 = options.device0;
      reduce_options.device1 = options.device1;
      reduce_options.gid0 = options.gid0;
      reduce_options.gid1 = options.gid1;
      reduce_options.control_port0 = 9580;
      reduce_options.control_port1 = 9581;
      reduce_options.payload_bytes = kAllreduceBytes;
      spark_transport::Tp4AllreduceSession reduce(reduce_options);

      for (int iteration = 1; iteration <= options.iterations; ++iteration) {
        const auto sequence = static_cast<std::uint64_t>(iteration);
        fill_reduce<<<reduce_blocks, threads, 0, stream>>>(
            reduce_input, options.rank, sequence);
        reduce.all_reduce(reduce_input, reduce_output, stream);
        validate_reduce<<<reduce_blocks, threads, 0, stream>>>(
            reduce_output, sequence, mismatches);

        fill_gather<<<gather_input_blocks, threads, 0, stream>>>(
            gather_input, options.rank, sequence);
        gather.all_gather(gather_input, gather_output, stream);
        validate_gather<<<gather_output_blocks, threads, 0, stream>>>(
            gather_output, sequence, mismatches);
      }
      check_cuda(cudaStreamSynchronize(stream), "mixed synchronize");
    }

    unsigned long long host_mismatches{};
    check_cuda(cudaMemcpy(&host_mismatches, mismatches,
                          sizeof(host_mismatches), cudaMemcpyDeviceToHost),
               "copy mismatch counter");
    std::cout << "TP4_MIXED rank=" << options.rank
              << " prewarm=" << options.prewarm
              << " iterations=" << options.iterations
              << " mismatches=" << host_mismatches
              << " correct=" << (host_mismatches == 0 ? "true" : "false")
              << '\n';

    cudaFree(mismatches);
    cudaFree(reduce_output);
    cudaFree(reduce_input);
    cudaFree(gather_output);
    cudaFree(gather_input);
    cudaStreamDestroy(stream);
    return host_mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TP4_MIXED_ERROR " << error.what() << '\n';
    return 1;
  }
}
