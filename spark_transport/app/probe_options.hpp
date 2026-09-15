#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace spark_transport::probe {

// Parse decimal counts before narrowing to the destination field type.
// Signs and whitespace are rejected; callers handle explicit sentinels such
// as an unpinned CPU (-1) separately from unsigned count/index arguments.
template <typename Integer = std::uint64_t>
Integer unsigned_value(const char* value, const char* name) {
  static_assert(std::is_integral_v<Integer> && !std::is_same_v<Integer, bool>);
  const std::string text(value);
  if (text.empty() || !std::all_of(text.begin(), text.end(),
                                 [](char digit) { return digit >= '0' && digit <= '9'; })) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  std::uint64_t parsed;
  try {
    parsed = std::stoull(text);
  } catch (const std::out_of_range&) {
    throw std::out_of_range(std::string(name) + " exceeds its integer range");
  }
  if (parsed > static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
    throw std::out_of_range(std::string(name) + " exceeds its integer range");
  }
  return static_cast<Integer>(parsed);
}

}  // namespace spark_transport::probe
