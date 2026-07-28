#include "spark_transport/control_channel.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/gpu_tp4.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/statistics.hpp"
#include "spark_transport/tp4_schedule.hpp"
#include "spark_transport/verbs_endpoint.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
  std::uint16_t control_port0{9460};
  std::uint16_t control_port1{9461};
  std::size_t bytes{12288};
  int warmup{1000};
  int iterations{10000};
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
      << "  --iterations COUNT\n";
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
      options.bytes = unsigned_value(take_value(), "payload size");
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup count"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iteration count"));
    } else {
      usage(argv[0]);
    }
  }

  if (options.rank >= 4 || options.peer0.empty() || options.peer1.empty() ||
      options.device0.empty() || options.device1.empty() ||
      options.bytes == 0 || options.bytes % 2 != 0 ||
      options.warmup < 0 || options.iterations <= 0) {
    usage(argv[0]);
  }
  return options;
}

double now_microseconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double, std::micro>(now).count();
}

std::uint64_t load_sequence(const std::uint64_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_sequence(std::uint64_t* address, std::uint64_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

void wait_for_sequence(const std::uint64_t* address, std::uint64_t expected,
                       const char* name) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (load_sequence(address) < expected) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(std::string("timed out waiting for ") + name);
    }
  }
}

void exchange_round(spark_transport::VerbsEndpoint& endpoint,
                    spark_transport::DoorbellControl& control,
                    const spark_transport::Tp2BufferLayout& layout,
                    std::size_t bytes, std::uint64_t sequence,
                    const char* producer_name, const char* consumer_name,
                    const char* acknowledgement_name) {
  store_sequence(&control.command_sequence, sequence);
  wait_for_sequence(&control.producer_sequence, sequence, producer_name);

  endpoint.write(layout.send_offset, layout.receive_offset, bytes, sequence,
                 false);
  endpoint.write(
      layout.control_offset +
          offsetof(spark_transport::DoorbellControl, producer_sequence),
      layout.control_offset +
          offsetof(spark_transport::DoorbellControl, remote_sequence),
      sizeof(sequence), sequence);
  endpoint.wait_for_send(sequence);

  wait_for_sequence(&control.consumer_sequence, sequence, consumer_name);
  endpoint.write(
      layout.control_offset +
          offsetof(spark_transport::DoorbellControl, consumer_sequence),
      layout.control_offset +
          offsetof(spark_transport::DoorbellControl,
                   acknowledgement_sequence),
      sizeof(sequence), sequence);
  endpoint.wait_for_send(sequence);
  wait_for_sequence(&control.observed_sequence, sequence,
                    acknowledgement_name);
}

void print_summary(const char* phase, std::vector<double> latencies) {
  const auto summary =
      spark_transport::summarize_latencies(std::move(latencies));
  std::cout << "PHASE"
            << " name=" << phase
            << " samples=" << summary.samples
            << " min_us=" << summary.minimum_us
            << " p50_us=" << summary.p50_us
            << " p95_us=" << summary.p95_us
            << " p99_us=" << summary.p99_us
            << " p999_us=" << summary.p999_us
            << " max_us=" << summary.maximum_us
            << " mean_us=" << summary.mean_us << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto plan0 =
        spark_transport::make_tp4_round_plan(options.rank, 0);
    const auto plan1 =
        spark_transport::make_tp4_round_plan(options.rank, 1);
    const auto layout =
        spark_transport::make_tp2_buffer_layout(options.bytes);

    auto channel0 =
        plan0.server
            ? spark_transport::ControlChannel::listen_and_accept(
                  options.control_port0)
            : spark_transport::ControlChannel::connect(
                  options.peer0, options.control_port0);
    auto buffer0 = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped, layout.total_bytes);
    spark_transport::VerbsEndpoint endpoint0(
        options.device0, 1, options.gid0, *buffer0);
    endpoint0.connect(channel0.exchange(endpoint0.local_info()));

    auto channel1 =
        plan1.server
            ? spark_transport::ControlChannel::listen_and_accept(
                  options.control_port1)
            : spark_transport::ControlChannel::connect(
                  options.peer1, options.control_port1);
    auto buffer1 = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped, layout.total_bytes);
    spark_transport::VerbsEndpoint endpoint1(
        options.device1, 1, options.gid1, *buffer1);
    endpoint1.connect(channel1.exchange(endpoint1.local_info()));

    const std::uint64_t final_sequence =
        static_cast<std::uint64_t>(options.warmup) +
        static_cast<std::uint64_t>(options.iterations);
    spark_transport::GpuTp4Worker worker(options.bytes);
    worker.launch(buffer0->device_data(), layout, buffer1->device_data(),
                  layout, options.rank, final_sequence);

    auto* control0 = reinterpret_cast<spark_transport::DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer0->host_data()) +
        layout.control_offset);
    auto* control1 = reinterpret_cast<spark_transport::DoorbellControl*>(
        static_cast<std::uint8_t*>(buffer1->host_data()) +
        layout.control_offset);

    channel0.barrier();
    channel1.barrier();

    std::vector<double> total_latencies;
    std::vector<double> round0_latencies;
    std::vector<double> round1_latencies;
    total_latencies.reserve(options.iterations);
    round0_latencies.reserve(options.iterations);
    round1_latencies.reserve(options.iterations);

    for (std::uint64_t sequence = 1; sequence <= final_sequence; ++sequence) {
      const double start = now_microseconds();
      exchange_round(endpoint0, *control0, layout, options.bytes, sequence,
                     "round-0 GPU production", "round-0 pair reduction",
                     "round-0 acknowledgement");
      const double paired = now_microseconds();
      exchange_round(endpoint1, *control1, layout, options.bytes, sequence,
                     "round-1 pair-sum production", "round-1 full reduction",
                     "round-1 acknowledgement");
      const double complete = now_microseconds();

      if (sequence > static_cast<std::uint64_t>(options.warmup)) {
        total_latencies.push_back(complete - start);
        round0_latencies.push_back(paired - start);
        round1_latencies.push_back(complete - paired);
      }
    }

    worker.synchronize();
    channel0.barrier();
    channel1.barrier();

    const bool local_correct =
        load_sequence(&control1->mismatch_count) == 0;
    const std::uint32_t round0_status =
        channel0.exchange(local_correct ? 1U : 0U);
    const bool pair_correct = local_correct && round0_status == 1U;
    const std::uint32_t round1_status =
        channel1.exchange(pair_correct ? 1U : 0U);
    const bool globally_correct = pair_correct && round1_status == 1U;

    std::cout << "TP4_BF16"
              << " rank=" << options.rank
              << " round0_peer=" << plan0.peer_rank
              << " round1_peer=" << plan1.peer_rank
              << " bytes=" << options.bytes
              << " iterations=" << options.iterations
              << " mismatched_iterations=" << control1->mismatch_count
              << " correct=" << (globally_correct ? "true" : "false")
              << '\n';
    print_summary("tp4_two_round_allreduce",
                  std::move(total_latencies));
    print_summary("tp4_round0_pair_sum",
                  std::move(round0_latencies));
    print_summary("tp4_round1_full_sum",
                  std::move(round1_latencies));

    if (!globally_correct) {
      throw std::runtime_error("TP4 BF16 verification failed");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
