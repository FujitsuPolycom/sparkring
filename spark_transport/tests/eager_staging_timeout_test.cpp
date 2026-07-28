#include "spark_transport/eager_staging_timeout.hpp"

#include <cassert>
#include <chrono>
#include <stdexcept>

using spark_transport::parse_eager_staging_timeout;

int main() {
  assert(parse_eager_staging_timeout(nullptr, "TEST") ==
         std::chrono::seconds(300));
  assert(parse_eager_staging_timeout("", "TEST") ==
         std::chrono::seconds(300));
  assert(parse_eager_staging_timeout("5", "TEST") ==
         std::chrono::seconds(5));
  assert(parse_eager_staging_timeout("600", "TEST") ==
         std::chrono::seconds(600));
  assert(parse_eager_staging_timeout("3600", "TEST") ==
         std::chrono::seconds(3600));

  for (const char* invalid : {"-1", "4", "5s", "3601"}) {
    bool rejected = false;
    try {
      static_cast<void>(parse_eager_staging_timeout(invalid, "TEST"));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}
