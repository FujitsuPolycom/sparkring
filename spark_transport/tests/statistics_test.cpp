#include "spark_transport/statistics.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
  std::vector<double> samples;
  for (int value = 1; value <= 1000; ++value) {
    samples.push_back(static_cast<double>(value));
  }
  const auto summary = spark_transport::summarize_latencies(samples);
  assert(summary.samples == 1000);
  assert(summary.minimum_us == 1.0);
  assert(summary.p50_us == 500.0);
  assert(summary.p95_us == 950.0);
  assert(summary.p99_us == 990.0);
  assert(summary.p999_us == 999.0);
  assert(summary.maximum_us == 1000.0);
  assert(std::abs(summary.mean_us - 500.5) < 0.0001);
}
