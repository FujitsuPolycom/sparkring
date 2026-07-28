#include "spark_transport/control_channel.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace spark_transport {
namespace {

constexpr int kControlConnectTimeoutMs = 10'000;

[[noreturn]] void throw_system_error(const char* operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

void configure_io_timeout(int fd) {
  timeval timeout{};
  timeout.tv_sec = kControlConnectTimeoutMs / 1000;
  timeout.tv_usec = (kControlConnectTimeoutMs % 1000) * 1000;
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    throw_system_error("setsockopt(SO_RCVTIMEO)");
  }
  if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
    throw_system_error("setsockopt(SO_SNDTIMEO)");
  }
}

}  // namespace

ControlChannel::ControlChannel(ControlChannel&& other) noexcept
    : fd_(other.fd_) {
  other.fd_ = -1;
}

ControlChannel& ControlChannel::operator=(ControlChannel&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = other.fd_;
    other.fd_ = -1;
  }
  return *this;
}

ControlChannel::~ControlChannel() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

ControlChannel ControlChannel::listen_and_accept(std::uint16_t port) {
  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) {
    throw_system_error("socket");
  }

  int reuse = 1;
  if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) <
      0) {
    close(listener);
    throw_system_error("setsockopt(SO_REUSEADDR)");
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
      0) {
    close(listener);
    throw_system_error("bind");
  }
  if (listen(listener, 1) < 0) {
    close(listener);
    throw_system_error("listen");
  }

  pollfd ready{listener, POLLIN, 0};
  int poll_result = 0;
  do {
    poll_result = poll(&ready, 1, kControlConnectTimeoutMs);
  } while (poll_result < 0 && errno == EINTR);
  if (poll_result == 0) {
    close(listener);
    throw std::runtime_error("timed out accepting control peer");
  }
  if (poll_result < 0) {
    close(listener);
    throw_system_error("poll");
  }

  const int connection = accept(listener, nullptr, nullptr);
  const int saved_errno = errno;
  close(listener);
  errno = saved_errno;
  if (connection < 0) {
    throw_system_error("accept");
  }
  try {
    configure_io_timeout(connection);
  } catch (...) {
    close(connection);
    throw;
  }
  return ControlChannel(connection);
}

ControlChannel ControlChannel::connect(const std::string& address,
                                       std::uint16_t port) {
  sockaddr_in peer{};
  peer.sin_family = AF_INET;
  peer.sin_port = htons(port);
  if (inet_pton(AF_INET, address.c_str(), &peer.sin_addr) != 1) {
    throw std::invalid_argument("control address must be an IPv4 literal");
  }

  for (int attempt = 0; attempt < 100; ++attempt) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      throw_system_error("socket");
    }
    if (::connect(fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)) == 0) {
      try {
        configure_io_timeout(fd);
      } catch (...) {
        close(fd);
        throw;
      }
      return ControlChannel(fd);
    }
    const int saved_errno = errno;
    close(fd);
    if (saved_errno != ECONNREFUSED && saved_errno != ETIMEDOUT &&
        saved_errno != EHOSTUNREACH) {
      errno = saved_errno;
      throw_system_error("connect");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("timed out connecting to control peer");
}

void ControlChannel::send_all(const void* data, std::size_t bytes) const {
  const auto* cursor = static_cast<const std::byte*>(data);
  while (bytes > 0) {
    const ssize_t sent = send(fd_, cursor, bytes, MSG_NOSIGNAL);
    if (sent < 0 && errno == EINTR) {
      continue;
    }
    if (sent <= 0) {
      throw_system_error("send");
    }
    cursor += sent;
    bytes -= static_cast<std::size_t>(sent);
  }
}

void ControlChannel::receive_all(void* data, std::size_t bytes) const {
  auto* cursor = static_cast<std::byte*>(data);
  while (bytes > 0) {
    const ssize_t received = recv(fd_, cursor, bytes, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received == 0) {
      throw std::runtime_error("control peer closed the connection");
    }
    if (received < 0) {
      throw_system_error("recv");
    }
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
}

void ControlChannel::barrier() const {
  constexpr std::uint32_t token = 0x42525231;
  const auto remote = exchange(token);
  if (remote != token) {
    throw std::runtime_error("invalid control barrier token");
  }
}

}  // namespace spark_transport
