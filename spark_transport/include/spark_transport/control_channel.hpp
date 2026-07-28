#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace spark_transport {

class ControlChannel {
 public:
  ControlChannel(const ControlChannel&) = delete;
  ControlChannel& operator=(const ControlChannel&) = delete;
  ControlChannel(ControlChannel&& other) noexcept;
  ControlChannel& operator=(ControlChannel&& other) noexcept;
  ~ControlChannel();

  static ControlChannel listen_and_accept(std::uint16_t port);
  static ControlChannel connect(const std::string& address, std::uint16_t port);

  void send_all(const void* data, std::size_t bytes) const;
  void receive_all(void* data, std::size_t bytes) const;

  template <typename T>
  T exchange(const T& local) const {
    static_assert(std::is_trivially_copyable_v<T>);
    send_all(&local, sizeof(local));
    T remote{};
    receive_all(&remote, sizeof(remote));
    return remote;
  }

  void barrier() const;

 private:
  explicit ControlChannel(int fd) : fd_(fd) {}
  int fd_{-1};
};

}  // namespace spark_transport
