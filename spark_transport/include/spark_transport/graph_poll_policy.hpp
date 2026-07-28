#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace spark_transport {

enum class GraphPollPolicy {
  kAdaptiveYield,
  kDedicatedSpin,
};

inline GraphPollPolicy parse_graph_poll_policy(
    const char* value, const char* variable_name) {
  const std::string_view text =
      value == nullptr ? std::string_view{} : std::string_view(value);
  if (text.empty() || text == "adaptive-yield") {
    return GraphPollPolicy::kAdaptiveYield;
  }
  if (text == "dedicated-spin") {
    return GraphPollPolicy::kDedicatedSpin;
  }
  throw std::invalid_argument(
      std::string(variable_name) +
      " must be adaptive-yield or dedicated-spin");
}

inline const char* graph_poll_policy_name(GraphPollPolicy policy) noexcept {
  return policy == GraphPollPolicy::kDedicatedSpin
             ? "dedicated-spin"
             : "adaptive-yield";
}

inline void validate_graph_poll_policy_configuration(
    GraphPollPolicy policy, bool progress_cpu_configured,
    const char* variable_name) {
  if (
      policy == GraphPollPolicy::kDedicatedSpin &&
      !progress_cpu_configured) {
    throw std::invalid_argument(
        std::string(variable_name) +
        "=dedicated-spin requires an exclusively affined graph progress CPU");
  }
}

}  // namespace spark_transport
