#include "spark_transport/verbs_endpoint.hpp"

#include <cassert>
#include <cstring>
#include <type_traits>

int main() {
  static_assert(std::is_trivially_copyable_v<spark_transport::EndpointInfo>);
  spark_transport::EndpointInfo info{};
  info.qp_number = 0x12345678;
  info.rkey = 0xabcdef01;
  info.address = 0x123456789abcdef0ULL;
  info.buffer_bytes = 1024 * 1024;
  info.lid = 0x4321;
  for (std::size_t index = 0; index < sizeof(info.gid); ++index) {
    info.gid[index] = static_cast<std::uint8_t>(index);
  }

  spark_transport::EndpointInfo copy{};
  std::memcpy(&copy, &info, sizeof(info));
  assert(copy.magic == spark_transport::kEndpointMagic);
  assert(copy.version == spark_transport::kEndpointVersion);
  assert(copy.qp_number == info.qp_number);
  assert(copy.rkey == info.rkey);
  assert(copy.address == info.address);
  assert(copy.buffer_bytes == info.buffer_bytes);
  assert(copy.lid == info.lid);
  assert(std::memcmp(copy.gid, info.gid, sizeof(info.gid)) == 0);
}
