#include "spark_transport/control_channel.hpp"
#include "spark_transport/gpu_doorbell.hpp"
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
#include <thread>
#include <vector>

namespace {

struct Options {
  bool server{};
  std::string peer;
  std::string device;
  std::uint8_t gid_index{3};
  std::uint16_t control_port{9410};
  std::size_t bytes{16 * 1024};
  std::size_t buffer_bytes{1024 * 1024};
  int warmup{1000};
  int iterations{10000};
  spark_transport::MemoryKind memory{
      spark_transport::MemoryKind::kHost};
  bool gpu_producer{};
  bool gpu_verifier{};
  bool gpu_roundtrip{};
  bool gpu_signal_only{};
};

[[noreturn]] void usage(const char* executable) {
  std::cerr
      << "Usage:\n"
      << "  " << executable
      << " --server --device HCA [options]\n"
      << "  " << executable
      << " --client PEER_IP --device HCA [options]\n\n"
      << "Options:\n"
      << "  --gid INDEX\n"
      << "  --control-port PORT\n"
      << "  --bytes BYTES\n"
      << "  --buffer-bytes BYTES\n"
      << "  --warmup COUNT\n"
      << "  --iterations COUNT\n"
      << "  --memory host|cuda-mapped|cuda-write-combined|cuda-device"
         "|cuda-managed\n"
      << "  --gpu-producer\n"
      << "  --gpu-verifier\n"
      << "  --gpu-roundtrip\n"
      << "  --gpu-signal-only (skip per-iteration payload scan)\n";
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
    } else if (argument == "--buffer-bytes") {
      options.buffer_bytes = unsigned_value(take_value(), "buffer size");
    } else if (argument == "--warmup") {
      options.warmup =
          static_cast<int>(unsigned_value(take_value(), "warmup count"));
    } else if (argument == "--iterations") {
      options.iterations =
          static_cast<int>(unsigned_value(take_value(), "iteration count"));
    } else if (argument == "--memory") {
      options.memory = spark_transport::parse_memory_kind(take_value());
    } else if (argument == "--gpu-producer") {
      options.gpu_producer = true;
    } else if (argument == "--gpu-verifier") {
      options.gpu_verifier = true;
    } else if (argument == "--gpu-roundtrip") {
      options.gpu_roundtrip = true;
    } else if (argument == "--gpu-signal-only") {
      options.gpu_signal_only = true;
    } else {
      usage(argv[0]);
    }
  }

  if (options.device.empty() || options.server == !options.peer.empty() ||
      options.bytes == 0 || options.bytes > options.buffer_bytes ||
      options.warmup < 0 || options.iterations <= 0) {
    usage(argv[0]);
  }
  if ((options.gpu_producer || options.gpu_verifier ||
       options.gpu_roundtrip) &&
      options.memory == spark_transport::MemoryKind::kHost) {
    throw std::invalid_argument(
        "GPU modes require a CUDA-mapped memory backend");
  }
  if (options.gpu_signal_only && !options.gpu_roundtrip) {
    throw std::invalid_argument(
        "--gpu-signal-only requires --gpu-roundtrip");
  }
  if (options.gpu_roundtrip &&
      options.memory == spark_transport::MemoryKind::kCudaDevice) {
    throw std::invalid_argument(
        "GPU round trip currently needs CPU-visible control memory; use "
        "cuda-managed for the direct-allocation probe");
  }
  const auto control_offset =
      spark_transport::aligned_control_offset(options.bytes);
  if (options.gpu_roundtrip &&
      control_offset + sizeof(spark_transport::DoorbellControl) >
          options.buffer_bytes) {
    throw std::invalid_argument(
        "buffer is too small for payload and doorbell control block");
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

bool run_gpu_roundtrip(const Options& options,
                       spark_transport::ControlChannel& channel,
                       spark_transport::MemoryBuffer& buffer,
                       spark_transport::VerbsEndpoint& endpoint) {
  const std::size_t control_offset =
      spark_transport::aligned_control_offset(options.bytes);
  auto* host_bytes = static_cast<std::uint8_t*>(buffer.host_data());
  auto* control = reinterpret_cast<spark_transport::DoorbellControl*>(
      host_bytes + control_offset);
  const std::uint64_t final_sequence =
      static_cast<std::uint64_t>(options.warmup) +
      static_cast<std::uint64_t>(options.iterations);

  if (options.server) {
    spark_transport::launch_receiver_doorbell(
        buffer.device_data(), options.bytes, control_offset, final_sequence,
        !options.gpu_signal_only);
  } else {
    spark_transport::launch_sender_doorbell(
        buffer.device_data(), options.bytes, control_offset, final_sequence);
  }
  channel.barrier();

  std::vector<double> total_latencies;
  std::vector<double> producer_latencies;
  std::vector<double> outbound_latencies;
  std::vector<double> acknowledgement_latencies;
  total_latencies.reserve(options.iterations);
  producer_latencies.reserve(options.iterations);
  outbound_latencies.reserve(options.iterations);
  acknowledgement_latencies.reserve(options.iterations);

  for (std::uint64_t sequence = 1; sequence <= final_sequence; ++sequence) {
    if (options.server) {
      wait_for_sequence(&control->consumer_sequence, sequence,
                        "receiver GPU consumption");
      endpoint.write(
          control_offset +
              offsetof(spark_transport::DoorbellControl, consumer_sequence),
          control_offset + offsetof(spark_transport::DoorbellControl,
                                    acknowledgement_sequence),
          sizeof(sequence), sequence);
      endpoint.wait_for_send(sequence);
      continue;
    }

    const double start = now_microseconds();
    store_sequence(&control->command_sequence, sequence);
    wait_for_sequence(&control->producer_sequence, sequence,
                      "sender GPU publication");
    const double produced = now_microseconds();

    endpoint.write(0, 0, options.bytes, sequence, false);
    endpoint.write(
        control_offset +
            offsetof(spark_transport::DoorbellControl, producer_sequence),
        control_offset +
            offsetof(spark_transport::DoorbellControl, remote_sequence),
        sizeof(sequence), sequence);
    endpoint.wait_for_send(sequence);
    const double outbound_complete = now_microseconds();

    wait_for_sequence(&control->observed_sequence, sequence,
                      "sender GPU acknowledgement");
    const double complete = now_microseconds();

    if (sequence > static_cast<std::uint64_t>(options.warmup)) {
      total_latencies.push_back(complete - start);
      producer_latencies.push_back(produced - start);
      outbound_latencies.push_back(outbound_complete - produced);
      acknowledgement_latencies.push_back(complete - outbound_complete);
    }
  }

  spark_transport::synchronize_doorbell();
  channel.barrier();

  bool local_correct = load_sequence(&control->mismatch_count) == 0;
  if (options.server && options.gpu_signal_only) {
    const auto expected = static_cast<std::uint8_t>(
        (final_sequence * 131U + 0xa5U) & 0xffU);
    local_correct = buffer.verify_on_gpu(expected, options.bytes);
  }
  const std::uint32_t local_status = local_correct ? 1U : 0U;
  const std::uint32_t remote_status = channel.exchange(local_status);

  if (!options.server) {
    std::cout << "GPU_ROUNDTRIP"
              << " memory="
              << spark_transport::memory_kind_name(options.memory)
              << " receiver="
              << (options.gpu_signal_only ? "signal-only" : "verify")
              << " bytes=" << options.bytes
              << " iterations=" << options.iterations << '\n';
    print_summary("gpu_visible_roundtrip", std::move(total_latencies));
    print_summary("gpu_produce", std::move(producer_latencies));
    print_summary("outbound_rdma", std::move(outbound_latencies));
    print_summary("remote_consume_and_ack",
                  std::move(acknowledgement_latencies));
  } else {
    std::cout << "VERIFY"
              << " mode=gpu-roundtrip"
              << " receiver="
              << (options.gpu_signal_only ? "signal-only" : "verify")
              << " mismatched_iterations=" << control->mismatch_count
              << " correct=" << (local_correct ? "true" : "false") << '\n';
  }
  return local_correct && remote_status == 1U;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    auto channel =
        options.server
            ? spark_transport::ControlChannel::listen_and_accept(
                  options.control_port)
            : spark_transport::ControlChannel::connect(options.peer,
                                                       options.control_port);

    auto buffer = spark_transport::MemoryBuffer::allocate(
        options.memory, options.buffer_bytes);
    spark_transport::VerbsEndpoint endpoint(
        options.device, 1, options.gid_index, *buffer);
    const auto remote = channel.exchange(endpoint.local_info());
    endpoint.connect(remote);

    if (options.gpu_roundtrip) {
      if (!run_gpu_roundtrip(options, channel, *buffer, endpoint)) {
        throw std::runtime_error("GPU round-trip verification failed");
      }
      return 0;
    }

    constexpr std::uint8_t pattern = 0xa5;
    if (!options.server) {
      if (options.gpu_producer) {
        buffer->fill_from_gpu(pattern);
      } else {
        buffer->fill_from_cpu(pattern);
      }
    }

    channel.barrier();

    if (!options.server) {
      for (int iteration = 0; iteration < options.warmup; ++iteration) {
        endpoint.write(0, 0, options.bytes,
                       static_cast<std::uint64_t>(iteration + 1));
        endpoint.wait_for_send(static_cast<std::uint64_t>(iteration + 1));
      }

      std::vector<double> latencies;
      latencies.reserve(options.iterations);
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        const auto work_id =
            static_cast<std::uint64_t>(options.warmup + iteration + 1);
        const double start = now_microseconds();
        endpoint.write(0, 0, options.bytes, work_id);
        endpoint.wait_for_send(work_id);
        latencies.push_back(now_microseconds() - start);
      }

      const auto summary =
          spark_transport::summarize_latencies(std::move(latencies));
      std::cout << "RESULT"
                << " memory="
                << spark_transport::memory_kind_name(options.memory)
                << " producer=" << (options.gpu_producer ? "gpu" : "cpu")
                << " bytes=" << options.bytes
                << " samples=" << summary.samples
                << " min_us=" << summary.minimum_us
                << " p50_us=" << summary.p50_us
                << " p95_us=" << summary.p95_us
                << " p99_us=" << summary.p99_us
                << " p999_us=" << summary.p999_us
                << " max_us=" << summary.maximum_us
                << " mean_us=" << summary.mean_us << '\n';
    }

    channel.barrier();

    bool local_correct = true;
    if (options.server) {
      local_correct =
          options.gpu_verifier
              ? buffer->verify_on_gpu(pattern, options.bytes)
              : buffer->verify_on_cpu(pattern, options.bytes);
    }
    const std::uint32_t local_status = local_correct ? 1U : 0U;
    const std::uint32_t remote_status = channel.exchange(local_status);
    if (options.server) {
      std::cout << "VERIFY"
                << " memory="
                << spark_transport::memory_kind_name(options.memory)
                << " verifier=" << (options.gpu_verifier ? "gpu" : "cpu")
                << " correct=" << (local_correct ? "true" : "false")
                << '\n';
    } else if (remote_status != 1U) {
      throw std::runtime_error("remote data verification failed");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
