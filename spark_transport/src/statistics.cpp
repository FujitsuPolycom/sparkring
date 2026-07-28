#include "spark_transport/statistics.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace spark_transport {
namespace {

double percentile(const std::vector<double>& ordered, double fraction) {
  const auto index = static_cast<std::size_t>(
      fraction * static_cast<double>(ordered.size() - 1));
  return ordered[index];
}

}  // namespace

LatencySummary summarize_latencies(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::invalid_argument("cannot summarize an empty latency sample");
  }
  std::sort(samples.begin(), samples.end());
  const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
  return {
      samples.size(),
      samples.front(),
      percentile(samples, 0.50),
      percentile(samples, 0.95),
      percentile(samples, 0.99),
      percentile(samples, 0.999),
      samples.back(),
      total / static_cast<double>(samples.size()),
  };
}

}  // namespace spark_transport
