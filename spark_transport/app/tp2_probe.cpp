#include "spark_transport/control_channel.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/gpu_tp2.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/statistics.hpp"
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
  bool server{};
  std::string peer;
  std::string device;
  std::uint8_t gid_index{3};
  std::uint16_t control_port{9440};
  std::size_t bytes{16 * 1024};
  int warmup{1000};
  int iterations{10000};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage:\n"
      << "  " << executable << " --server --device HCA [options]\n"
      << "  " << executable
      << " --client PEER_IP --device HCA [options]\n\n"
      << "Options:\n"
      << "  --gid INDEX\n"
      << "  --control-port PORT\n"
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

    if (argument == "--server") {
      options.server = true;
    } else if (argument == "--client") {
      options.peer = take_value();
    } else if (argument == "--device") {
      options.device = take_value();
    } else if (argument == "--gid") {
      options.gid_index =
          static_cast<std::uint8_t>(unsigned_value(take_value(), "GID index"));
    } else if (argument == "--control-port") {
      options.control_port = static_cast<std::uint16_t>(
          unsigned_value(take_value(), "control port"));
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

  if (options.device.empty() || options.server == !options.peer.empty() ||
      options.bytes == 0 || options.bytes % 2 != 0 || options.warmup < 0 ||
      options.iterations <= 0) {
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
    const auto layout =
        spark_transport::make_tp2_buffer_layout(options.bytes);
    auto channel =
        options.server
            ? spark_transport::ControlChannel::listen_and_accept(
                  options.control_port)
            : spark_transport::ControlChannel::connect(options.peer,
                                                       options.control_port);
    auto buffer = spark_transport::MemoryBuffer::allocate(
        spark_transport::MemoryKind::kCudaMapped, layout.total_bytes);
    spark_transport::VerbsEndpoint endpoint(
        options.device, 1, options.gid_index, *buffer);
    const auto remote = channel.exchange(endpoint.local_info());
    endpoint.connect(remote);

    const std::uint32_t rank = options.server ? 1U : 0U;
    const std::uint64_t final_sequence =
        static_cast<std::uint64_t>(options.warmup) +
        static_cast<std::uint64_t>(options.iterations);
    spark_transport::GpuTp2Worker worker(options.bytes);
    worker.launch(buffer->device_data(), layout, rank, final_sequence);

    auto* host_bytes = static_cast<std::uint8_t*>(buffer->host_data());
    auto* control = reinterpret_cast<spark_transport::DoorbellControl*>(
        host_bytes + layout.control_offset);
    channel.barrier();

    std::vector<double> total_latencies;
    std::vector<double> producer_latencies;
    std::vector<double> exchange_latencies;
    std::vector<double> reduce_and_ack_latencies;
    total_latencies.reserve(options.iterations);
    producer_latencies.reserve(options.iterations);
    exchange_latencies.reserve(options.iterations);
    reduce_and_ack_latencies.reserve(options.iterations);

    for (std::uint64_t sequence = 1; sequence <= final_sequence; ++sequence) {
      const double start = now_microseconds();
      store_sequence(&control->command_sequence, sequence);
      wait_for_sequence(&control->producer_sequence, sequence,
                        "local BF16 production");
      const double produced = now_microseconds();

      endpoint.write(layout.send_offset, layout.receive_offset, options.bytes,
                     sequence, false);
      endpoint.write(
          layout.control_offset +
              offsetof(spark_transport::DoorbellControl, producer_sequence),
          layout.control_offset +
              offsetof(spark_transport::DoorbellControl, remote_sequence),
          sizeof(sequence), sequence);
      endpoint.wait_for_send(sequence);
      const double exchanged = now_microseconds();

      wait_for_sequence(&control->consumer_sequence, sequence,
                        "fused BF16 reduction");
      endpoint.write(
          layout.control_offset +
              offsetof(spark_transport::DoorbellControl, consumer_sequence),
          layout.control_offset +
              offsetof(spark_transport::DoorbellControl,
                       acknowledgement_sequence),
          sizeof(sequence), sequence);
      endpoint.wait_for_send(sequence);
      wait_for_sequence(&control->observed_sequence, sequence,
                        "peer acknowledgement");
      const double complete = now_microseconds();

      if (sequence > static_cast<std::uint64_t>(options.warmup)) {
        total_latencies.push_back(complete - start);
        producer_latencies.push_back(produced - start);
        exchange_latencies.push_back(exchanged - produced);
        reduce_and_ack_latencies.push_back(complete - exchanged);
      }
    }

    worker.synchronize();
    channel.barrier();
    const bool local_correct = load_sequence(&control->mismatch_count) == 0;
    const std::uint32_t remote_status =
        channel.exchange(local_correct ? 1U : 0U);

    std::cout << "TP2_BF16"
              << " rank=" << rank
              << " bytes=" << options.bytes
              << " iterations=" << options.iterations
              << " mismatched_iterations=" << control->mismatch_count
              << " correct="
              << (local_correct && remote_status == 1U ? "true" : "false")
              << '\n';
    print_summary("tp2_exchange_reduce_roundtrip",
                  std::move(total_latencies));
    print_summary("gpu_produce", std::move(producer_latencies));
    print_summary("bidirectional_exchange", std::move(exchange_latencies));
    print_summary("reduce_and_ack", std::move(reduce_and_ack_latencies));

    if (!local_correct || remote_status != 1U) {
      throw std::runtime_error("TP2 BF16 verification failed");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
