#include "spark_transport/verbs_endpoint.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

using spark_transport::detail::SendCompletionDisposition;

void exact_completion_is_complete() {
  assert(spark_transport::detail::classify_send_completion(
             41, 41, false) == SendCompletionDisposition::kComplete);
  assert(spark_transport::detail::classify_send_completion(
             41, 41, true) == SendCompletionDisposition::kComplete);
}

void strict_poll_rejects_every_other_work_id() {
  assert(spark_transport::detail::classify_send_completion(
             41, 40, false) == SendCompletionDisposition::kUnexpected);
  assert(spark_transport::detail::classify_send_completion(
             41, 42, false) == SendCompletionDisposition::kUnexpected);
}

void through_poll_retires_only_older_work_ids() {
  assert(spark_transport::detail::classify_send_completion(
             41, 40, true) == SendCompletionDisposition::kRetire);
  assert(spark_transport::detail::classify_send_completion(
             41, 42, true) == SendCompletionDisposition::kUnexpected);
}

void work_id_order_does_not_wrap() {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  assert(spark_transport::detail::classify_send_completion(
             maximum, maximum - 1, true) ==
         SendCompletionDisposition::kRetire);
  assert(spark_transport::detail::classify_send_completion(
             maximum, maximum, true) ==
         SendCompletionDisposition::kComplete);
  assert(spark_transport::detail::classify_send_completion(
             0, maximum, true) ==
         SendCompletionDisposition::kUnexpected);
}

}  // namespace

int main() {
  static_assert(std::is_same_v<
                std::underlying_type_t<
                    spark_transport::SendCompletionPollState>,
                std::uint8_t>);
  static_assert(std::is_same_v<
                decltype(std::declval<spark_transport::VerbsEndpoint&>()
                             .poll_send_through(std::uint64_t{})),
                spark_transport::SendCompletionPollState>);
  static_assert(std::is_same_v<
                decltype(std::declval<spark_transport::VerbsEndpoint&>()
                             .poll_send_completions(
                                 static_cast<spark_transport::SendCompletion*>(
                                     nullptr),
                                 std::size_t{})),
                std::size_t>);
  static_assert(std::is_same_v<
                decltype(std::declval<const spark_transport::VerbsEndpoint&>()
                             .active_mtu_bytes()),
                std::uint32_t>);
  static_assert(std::is_same_v<
                decltype(std::declval<const spark_transport::VerbsEndpoint&>()
                             .maximum_send_work_requests()),
                std::uint32_t>);
  static_assert(std::is_same_v<
                decltype(&spark_transport::VerbsEndpoint::wait_for_send),
                void (spark_transport::VerbsEndpoint::*)(std::uint64_t)>);
  static_assert(std::is_same_v<
                decltype(
                    &spark_transport::VerbsEndpoint::wait_for_send_through),
                void (spark_transport::VerbsEndpoint::*)(std::uint64_t)>);
  static_assert(spark_transport::SendCompletionPollState::kPending !=
                spark_transport::SendCompletionPollState::kComplete);
  exact_completion_is_complete();
  strict_poll_rejects_every_other_work_id();
  through_poll_retires_only_older_work_ids();
  work_id_order_does_not_wrap();
}
