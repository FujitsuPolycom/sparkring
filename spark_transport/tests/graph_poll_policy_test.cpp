#include "spark_transport/graph_poll_policy.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

int main() {
  using spark_transport::GraphPollPolicy;
  using spark_transport::graph_poll_policy_name;
  using spark_transport::parse_graph_poll_policy;
  using spark_transport::validate_graph_poll_policy_configuration;

  assert(
      parse_graph_poll_policy(nullptr, "TEST_POLICY") ==
      GraphPollPolicy::kAdaptiveYield);
  assert(
      parse_graph_poll_policy("", "TEST_POLICY") ==
      GraphPollPolicy::kAdaptiveYield);
  assert(
      parse_graph_poll_policy("adaptive-yield", "TEST_POLICY") ==
      GraphPollPolicy::kAdaptiveYield);
  assert(
      parse_graph_poll_policy("dedicated-spin", "TEST_POLICY") ==
      GraphPollPolicy::kDedicatedSpin);
  assert(
      std::string(
          graph_poll_policy_name(GraphPollPolicy::kAdaptiveYield)) ==
      "adaptive-yield");
  assert(
      std::string(
          graph_poll_policy_name(GraphPollPolicy::kDedicatedSpin)) ==
      "dedicated-spin");

  bool rejected{};
  try {
    (void)parse_graph_poll_policy("spin", "TEST_POLICY");
  } catch (const std::invalid_argument& error) {
    rejected =
        std::string(error.what()).find("TEST_POLICY") != std::string::npos;
  }
  assert(rejected);

  validate_graph_poll_policy_configuration(
      GraphPollPolicy::kAdaptiveYield, false, "TEST_POLICY");
  validate_graph_poll_policy_configuration(
      GraphPollPolicy::kDedicatedSpin, true, "TEST_POLICY");

  rejected = false;
  try {
    validate_graph_poll_policy_configuration(
        GraphPollPolicy::kDedicatedSpin, false, "TEST_POLICY");
  } catch (const std::invalid_argument& error) {
    const std::string message(error.what());
    rejected =
        message.find("TEST_POLICY=dedicated-spin") != std::string::npos &&
        message.find("progress CPU") != std::string::npos;
  }
  assert(rejected);
  return 0;
}
