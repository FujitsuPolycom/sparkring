#include "spark_transport/control_channel.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("control-channel test condition failed");
}
}  // namespace

int main() {
  // Reserve a loopback port without listening so the client initially sees
  // refusal. No remote address, GPU, or RDMA resource is used by this test.
  const int reservation = socket(AF_INET, SOCK_STREAM, 0);
  require(reservation >= 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  require(bind(reservation, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  socklen_t size = sizeof(address);
  require(getsockname(reservation, reinterpret_cast<sockaddr*>(&address), &size) == 0);
  const auto port = ntohs(address.sin_port);
  auto client = std::async(std::launch::async, [port] {
    auto channel = spark_transport::ControlChannel::connect("127.0.0.1", port);
    require(channel.exchange<std::uint32_t>(17) == 23);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  close(reservation);
  {
    auto server = spark_transport::ControlChannel::listen_and_accept(port);
    require(server.exchange<std::uint32_t>(23) == 17);
    client.get();
  }

  // An always-refused port must exhaust the shared retry deadline rather
  // than receiving a fresh timeout budget for each connection attempt.
  const auto begin = std::chrono::steady_clock::now();
  bool timed_out = false;
  try {
    auto channel = spark_transport::ControlChannel::connect("127.0.0.1", port);
  } catch (const std::runtime_error& error) {
    timed_out = std::string(error.what()) == "timed out connecting to control peer";
  }
  require(timed_out);
  const auto elapsed = std::chrono::steady_clock::now() - begin;
  require(elapsed >= std::chrono::seconds(9));
  require(elapsed < std::chrono::seconds(12));
}
