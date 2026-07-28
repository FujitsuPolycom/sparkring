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

struct Options {
  std::uint32_t rank{4};
  std::string peer0;
  std::string peer1;
  std::string device0{"rocep1s0f0"};
  std::string device1{"rocep1s0f1"};
  std::uint8_t gid0{3};
  std::uint8_t gid1{3};
  std::uint16_t control_port0{9990};
  std::uint16_t control_port1{9991};
  std::uint32_t query_rows{};
  int warmup{4};
  int iterations{100};
  int production_rounds{};
  bool alternate_streams{};
  std::uint64_t queued_delay_ms{};
  std::uint32_t queued_delay_rank{4};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " --rank RANK --peer0 IP --peer1 IP [options]\n"
      << "Options: --device0 HCA --device1 HCA --gid0 N --gid1 N\n"
      << "         --control-port0 PORT --control-port1 PORT\n"
      << "         --q Q (1..6; omit to test all Q)\n"
      << "         --warmup N --iterations N\n"
      << "         --alternate-streams\n"
      << "         --production-rounds ROUNDS\n"
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
      options.query_rows = static_cast<std::uint32_t>(
          unsigned_value(take_value(), "Q"));
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iterations"));
    } else if (argument == "--production-rounds") {
      options.production_rounds = static_cast<int>(
          unsigned_value(take_value(), "production rounds"));
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
  if (options.rank >= spark_transport::kTp4VocabWorldSize ||
      options.peer0.empty() || options.peer1.empty() ||
      options.query_rows > spark_transport::kTp4VocabMaxQueryRows ||
      options.control_port0 == options.control_port1 ||
      options.warmup < 0 || options.iterations <= 0 ||
      options.production_rounds < 0 ||
      (options.production_rounds > 0 &&
       (!options.alternate_streams || options.query_rows != 0))) {
    usage(argv[0]);
  }
  if (options.queued_delay_ms != 0 &&
      (options.queued_delay_ms < 5500 ||
       !options.alternate_streams || options.queued_delay_rank >= 4 ||
       options.warmup < 1 ||
       (options.production_rounds == 0 && options.iterations < 2))) {
    throw std::invalid_argument(
        "--queued-delay-ms must be at least 5500 and requires "
        "--queued-delay-rank 0..3, "
        "--alternate-streams, --warmup >= 1, and at least two measured "
        "submissions");
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
    std::size_t column, std::uint64_t submission_sequence) {
  return static_cast<std::uint16_t>(
      0x1000U + rank * 7919U + query_index * 1223U +
      static_cast<std::uint32_t>(column) * 13U +
      static_cast<std::uint32_t>(
          submission_sequence & 0xffffU) *
          17U);
}

__global__ void prepare_input(
    std::uint16_t* input, std::uint32_t rank,
    std::uint32_t query_rows, std::uint64_t submission_sequence) {
  const std::size_t elements =
      static_cast<std::size_t>(query_rows) *
      spark_transport::kTp4VocabShardElements;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint32_t query_index =
        static_cast<std::uint32_t>(
            index / spark_transport::kTp4VocabShardElements);
    const std::size_t column =
        index % spark_transport::kTp4VocabShardElements;
    input[index] = expected_word(
        rank, query_index, column, submission_sequence);
  }
}

__global__ void validate_output(
    const std::uint16_t* output, std::uint32_t query_rows,
    std::uint64_t submission_sequence,
    unsigned long long* mismatches) {
  const std::size_t elements =
      static_cast<std::size_t>(query_rows) *
      spark_transport::kTp4VocabWorldSize *
      spark_transport::kTp4VocabShardElements;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t elements_per_query =
        spark_transport::kTp4VocabWorldSize *
        spark_transport::kTp4VocabShardElements;
    const std::uint32_t query_index =
        static_cast<std::uint32_t>(index / elements_per_query);
    const std::size_t in_query = index % elements_per_query;
    const std::uint32_t source_rank =
        static_cast<std::uint32_t>(
            in_query / spark_transport::kTp4VocabShardElements);
    const std::size_t column =
        in_query % spark_transport::kTp4VocabShardElements;
    if (output[index] != expected_word(
            source_rank, query_index, column,
            submission_sequence)) {
      atomicAdd(mismatches, 1ULL);
    }
  }
}

void submit_and_validate(
    spark_tp4_vocab_allgather_handle session,
    std::uint32_t rank, std::uint32_t query_rows,
    std::uint64_t submission_sequence, std::uint16_t* input,
    std::uint16_t* output, cudaStream_t stream,
    unsigned long long* mismatches, const char* operation) {
  constexpr int threads = 256;
  const std::size_t input_elements =
      static_cast<std::size_t>(query_rows) *
      spark_transport::kTp4VocabShardElements;
  const std::size_t output_elements =
      input_elements * spark_transport::kTp4VocabWorldSize;
  const int input_blocks =
      static_cast<int>((input_elements + threads - 1) / threads);
  const int output_blocks =
      static_cast<int>((output_elements + threads - 1) / threads);

  prepare_input<<<input_blocks, threads, 0, stream>>>(
      input, rank, query_rows, submission_sequence);
  check_cuda(cudaGetLastError(), "prepare vocabulary input");

  char error[512]{};
  if (spark_tp4_vocab_allgather(
          session, input, output, query_rows, stream, error,
          sizeof(error)) != 0) {
    throw std::runtime_error(
        std::string(operation) + ": " + error);
  }

  validate_output<<<output_blocks, threads, 0, stream>>>(
      output, query_rows, submission_sequence, mismatches);
  check_cuda(cudaGetLastError(), "validate vocabulary output");
}

void CUDART_CB delay_stream(void* duration_pointer) {
  const auto duration =
      *static_cast<const std::chrono::milliseconds*>(duration_pointer);
  std::this_thread::sleep_for(duration);
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::size_t max_input_bytes =
        spark_transport::tp4_vocab_input_bytes(
            spark_transport::kTp4VocabMaxQueryRows);
    const std::size_t max_output_bytes =
        spark_transport::tp4_vocab_output_bytes(
            spark_transport::kTp4VocabMaxQueryRows);

    std::uint16_t* inputs[2]{};
    std::uint16_t* outputs[2]{};
    unsigned long long* device_mismatches{};
    cudaStream_t streams[2]{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaMalloc(&inputs[0], max_input_bytes),
               "cudaMalloc input 0");
    check_cuda(cudaMalloc(&outputs[0], max_output_bytes),
               "cudaMalloc output 0");
    check_cuda(
        cudaMalloc(&device_mismatches, sizeof(*device_mismatches)),
        "cudaMalloc mismatch counter");
    check_cuda(
        cudaMemset(
            device_mismatches, 0, sizeof(*device_mismatches)),
        "cudaMemset mismatch counter");
    check_cuda(
        cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking),
        "create stream 0");
    if (options.alternate_streams) {
      check_cuda(cudaMalloc(&inputs[1], max_input_bytes),
                 "cudaMalloc input 1");
      check_cuda(cudaMalloc(&outputs[1], max_output_bytes),
                 "cudaMalloc output 1");
      check_cuda(
          cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking),
          "create stream 1");
    }
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    unsigned long long total_mismatches{};
    unsigned long long prior_mismatches{};
    {
      spark_tp4_vocab_allgather_config config{};
      config.rank = options.rank;
      config.peer0 = options.peer0.c_str();
      config.peer1 = options.peer1.c_str();
      config.device0 = options.device0.c_str();
      config.device1 = options.device1.c_str();
      config.gid0 = options.gid0;
      config.gid1 = options.gid1;
      config.control_port0 = options.control_port0;
      config.control_port1 = options.control_port1;
      char create_error[512]{};
      const VocabHandle session(
          spark_tp4_vocab_allgather_create(
              &config, create_error, sizeof(create_error)));
      if (session.get() == nullptr) {
        throw std::runtime_error(
            std::string("create vocabulary session: ") +
            create_error);
      }

      std::uint64_t submission_sequence{};
      cudaStream_t current_stream = streams[0];
      std::chrono::milliseconds queued_delay(options.queued_delay_ms);
      constexpr std::uint64_t kProductionPatternSize = 4;
      const std::uint64_t queued_delay_sequence =
          options.queued_delay_ms == 0
              ? 0
              : static_cast<std::uint64_t>(options.warmup) *
                        (options.production_rounds > 0
                             ? kProductionPatternSize
                             : 1) +
                    1;
      const auto submit = [&](std::uint32_t query_rows,
                              const char* operation) {
        if (submission_sequence ==
            std::numeric_limits<std::uint64_t>::max()) {
          throw std::overflow_error(
              "vocabulary probe submission sequence exhausted");
        }
        ++submission_sequence;
        const std::size_t stream_index =
            options.alternate_streams &&
                    (submission_sequence % 2 == 0)
                ? 1
                : 0;
        current_stream = streams[stream_index];
        if (submission_sequence == queued_delay_sequence &&
            options.rank == options.queued_delay_rank) {
          check_cuda(
              cudaLaunchHostFunc(
                  current_stream, delay_stream, &queued_delay),
              "cudaLaunchHostFunc queued delay");
        }
        submit_and_validate(
            session.get(), options.rank, query_rows,
            submission_sequence, inputs[stream_index],
            outputs[stream_index], current_stream,
            device_mismatches, operation);
      };
      const auto next_stream = [&]() {
        const std::size_t stream_index =
            options.alternate_streams &&
                    (submission_sequence % 2 != 0)
                ? 1
                : 0;
        return streams[stream_index];
      };
      const auto copy_mismatches = [&]() {
        check_cuda(
            cudaMemcpy(
                &total_mismatches, device_mismatches,
                sizeof(total_mismatches), cudaMemcpyDeviceToHost),
            "copy mismatch count");
        const unsigned long long mismatches =
            total_mismatches - prior_mismatches;
        prior_mismatches = total_mismatches;
        return mismatches;
      };

      if (options.production_rounds > 0) {
        constexpr std::uint32_t kProductionPattern[]{5, 1, 1, 1};
        constexpr std::uint64_t pattern_size =
            sizeof(kProductionPattern) /
            sizeof(kProductionPattern[0]);

        for (int round = 0; round < options.warmup; ++round) {
          for (const std::uint32_t q : kProductionPattern) {
            submit(q, "warmup production vocabulary gather");
          }
        }
        if (options.warmup > 0) {
          check_cuda(cudaStreamSynchronize(current_stream),
                     "production warmup synchronize");
        }

        check_cuda(cudaEventRecord(start, next_stream()),
                   "record production start");
        const auto host_start = std::chrono::steady_clock::now();
        for (int round = 0; round < options.production_rounds;
             ++round) {
          for (const std::uint32_t q : kProductionPattern) {
            submit(q, "measure production vocabulary gather");
          }
        }
        check_cuda(cudaEventRecord(stop, current_stream),
                   "record production stop");
        const auto host_submit_stop = std::chrono::steady_clock::now();
        check_cuda(cudaEventSynchronize(stop),
                   "production measurement synchronize");
        const auto host_finish = std::chrono::steady_clock::now();

        float device_ms{};
        check_cuda(cudaEventElapsedTime(&device_ms, start, stop),
                   "production elapsed time");
        const unsigned long long mismatches = copy_mismatches();
        const std::uint64_t measured_submissions =
            static_cast<std::uint64_t>(options.production_rounds) *
            pattern_size;
        const double device_us_per_call =
            static_cast<double>(device_ms) * 1000.0 /
            measured_submissions;
        const double submit_us_per_call =
            std::chrono::duration<double, std::micro>(
                host_submit_stop - host_start)
                .count() /
            measured_submissions;
        const double wall_us_per_call =
            std::chrono::duration<double, std::micro>(
                host_finish - host_start)
                .count() /
            measured_submissions;

        std::cout << std::fixed << std::setprecision(3)
                  << "TP4_VOCAB_ALLGATHER"
                  << " rank=" << options.rank
                  << " pattern=5,1,1,1"
                  << " production_rounds="
                  << options.production_rounds
                  << " measured_submissions="
                  << measured_submissions
                  << " alternate_streams=true"
                  << " queued_delay_ms=" << options.queued_delay_ms
                  << " queued_delay_rank="
                  << options.queued_delay_rank
                  << " queued_delay_applied="
                  << (options.queued_delay_ms != 0 &&
                              options.rank ==
                                  options.queued_delay_rank
                          ? "true"
                          : "false")
                  << " queued_delay_sequence="
                  << queued_delay_sequence
                  << " device_us_per_call=" << device_us_per_call
                  << " host_submit_us_per_call="
                  << submit_us_per_call
                  << " wall_us_per_call=" << wall_us_per_call
                  << " mismatches=" << mismatches << '\n';
      } else {
        const std::uint32_t first_q =
            options.query_rows == 0 ? 1 : options.query_rows;
        const std::uint32_t last_q =
            options.query_rows == 0
                ? spark_transport::kTp4VocabMaxQueryRows
                : options.query_rows;
        for (std::uint32_t q = first_q; q <= last_q; ++q) {
          for (int iteration = 0; iteration < options.warmup;
               ++iteration) {
            submit(q, "warmup vocabulary gather");
          }
          if (options.warmup > 0) {
            check_cuda(cudaStreamSynchronize(current_stream),
                       "warmup synchronize");
          }

          const auto host_start = std::chrono::steady_clock::now();
          check_cuda(cudaEventRecord(start, next_stream()),
                     "record start");
          for (int iteration = 0; iteration < options.iterations;
               ++iteration) {
            submit(q, "measure vocabulary gather");
          }
          check_cuda(cudaEventRecord(stop, current_stream),
                     "record stop");
          const auto host_submit_stop =
              std::chrono::steady_clock::now();
          check_cuda(cudaEventSynchronize(stop),
                     "measurement synchronize");
          const auto host_finish = std::chrono::steady_clock::now();

          float device_ms{};
          check_cuda(cudaEventElapsedTime(&device_ms, start, stop),
                     "elapsed time");
          const std::size_t output_bytes =
              spark_transport::tp4_vocab_output_bytes(q);
          const unsigned long long mismatches = copy_mismatches();
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
                    << "TP4_VOCAB_ALLGATHER"
                    << " rank=" << options.rank << " q=" << q
                    << " input_bytes="
                    << spark_transport::tp4_vocab_input_bytes(q)
                    << " output_bytes=" << output_bytes
                    << " iterations=" << options.iterations
                    << " alternate_streams="
                    << (options.alternate_streams ? "true" : "false")
                    << " queued_delay_ms=" << options.queued_delay_ms
                    << " queued_delay_rank="
                    << options.queued_delay_rank
                    << " queued_delay_applied="
                    << (options.queued_delay_ms != 0 &&
                                options.rank ==
                                    options.queued_delay_rank
                            ? "true"
                            : "false")
                    << " queued_delay_sequence="
                    << queued_delay_sequence
                    << " device_us_per_call=" << device_us_per_call
                    << " host_submit_us_per_call="
                    << submit_us_per_call
                    << " wall_us_per_call=" << wall_us_per_call
                    << " mismatches=" << mismatches << '\n';
        }
      }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    if (streams[1] != nullptr) {
      cudaStreamDestroy(streams[1]);
    }
    cudaStreamDestroy(streams[0]);
    cudaFree(device_mismatches);
    if (outputs[1] != nullptr) {
      cudaFree(outputs[1]);
    }
    if (inputs[1] != nullptr) {
      cudaFree(inputs[1]);
    }
    cudaFree(outputs[0]);
    cudaFree(inputs[0]);
    return total_mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TP4_VOCAB_ALLGATHER_ERROR " << error.what()
              << '\n';
    return 1;
  }
}
