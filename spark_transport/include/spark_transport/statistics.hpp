#pragma once

#include <cstddef>
#include <vector>

namespace spark_transport {

struct LatencySummary {
  std::size_t samples{};
  double minimum_us{};
  double p50_us{};
  double p95_us{};
  double p99_us{};
  double p999_us{};
  double maximum_us{};
  double mean_us{};
};

LatencySummary summarize_latencies(std::vector<double> samples);

}  // namespace spark_transport
