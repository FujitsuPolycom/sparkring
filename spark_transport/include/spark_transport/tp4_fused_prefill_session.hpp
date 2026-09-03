#pragma once

#include "spark_transport/tp4_bidirectional_prefill_session.hpp"

#include <memory>

namespace spark_transport {

class Tp4FusedPrefillSession {
 public:
  Tp4FusedPrefillSession(const Tp4FusedPrefillSession&) = delete;
  Tp4FusedPrefillSession& operator=(const Tp4FusedPrefillSession&) = delete;
  explicit Tp4FusedPrefillSession(Tp4BidirectionalPrefillOptions options);
  ~Tp4FusedPrefillSession();

  void all_reduce_fused(const void* input, void* output, void* cuda_stream,
                        std::uint32_t query_rows = 8192);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark_transport
