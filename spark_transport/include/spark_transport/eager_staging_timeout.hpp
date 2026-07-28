#pragma once

#include <charconv>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace spark_transport {

inline constexpr std::uint64_t kDefaultEagerStagingTimeoutSeconds = 300;
inline constexpr std::uint64_t kMaximumEagerStagingTimeoutSeconds = 3600;

inline std::chrono::seconds parse_eager_staging_timeout(
    const char* value, const char* setting_name) {
  if (value == nullptr || value[0] == '\0') {
    return std::chrono::seconds(kDefaultEagerStagingTimeoutSeconds);
  }

  const std::string_view text(value);
  std::uint64_t parsed{};
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      parsed < 5 || parsed > kMaximumEagerStagingTimeoutSeconds) {
    throw std::invalid_argument(
        std::string(setting_name) +
        " must be an integer in [5, 3600]");
  }
  return std::chrono::seconds(parsed);
}

}  // namespace spark_transport
