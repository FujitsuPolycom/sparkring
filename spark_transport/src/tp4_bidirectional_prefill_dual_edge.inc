class DualRailEdgePort final : public research::BidirectionalRingEdgePort {
 public:
  using EndpointMatrix =
      std::array<std::array<spark_transport::VerbsEndpoint*, 2>, 2>;

  DualRailEdgePort(EndpointMatrix endpoints,
                std::array<spark_transport::MemoryBuffer*, 2> arenas,
                spark_transport::Tp4TiledPoolLayout layout,
                std::uint32_t rail_count)
      : endpoints_(endpoints), arenas_(arenas), layout_(layout),
        rail_count_(rail_count) {
    if (rail_count_ != 1 && rail_count_ != 2) {
      throw std::invalid_argument("rail count must be one or two");
    }
    if (rail_count_ == 2) {
      for (const auto& neighbor : endpoints_) {
        if (neighbor[0] == nullptr || neighbor[1] == nullptr ||
            neighbor[0]->active_mtu_bytes() != 4096 ||
            neighbor[1]->active_mtu_bytes() != 4096 ||
            neighbor[0]->active_mtu_bytes() !=
                neighbor[1]->active_mtu_bytes()) {
          throw std::runtime_error(
              "dual rail requires equal active MTU 4096");
        }
      }
    }
  }
  research::RingSubmitState try_post_exchange(
      const research::BidirectionalRingExchangeRequest& request) override {
    if (!drain_completions()) return research::RingSubmitState::kFatal;
    const auto direction = direction_index(request.direction);
    const auto endpoint = endpoint_index(request.outgoing_endpoint);
    if (exchanges_[direction].count(request.work_id) != 0 ||
        request.span.active_bytes != layout_.tile_payload_bytes ||
        request.span.local_send_offset % 16U != 0 ||
        request.span.remote_receive_offset % 16U != 0 ||
        request.span.local_send_offset + request.span.active_bytes >
            arenas_[endpoint]->size() ||
        request.span.remote_receive_offset + request.span.active_bytes >
            arenas_[endpoint]->size()) {
      return research::RingSubmitState::kFatal;
    }
    const auto region = spark_transport::tp4_tiled_slot_region(
        layout_, request.ticket.slot, 0);
    const std::uint32_t split = rail_count_ == 2 ? 2U : 1U;
    const std::uint32_t rail_bytes = request.span.active_bytes / split;
    if (request.span.active_bytes % split != 0 || rail_bytes % 16U != 0) {
      return research::RingSubmitState::kFatal;
    }
    // Preflight every rail before posting either half; a half-post exchange is
    // unrecoverable because the peer cannot consume or credit a partial tile.
    for (std::uint32_t rail = 0; rail < rail_count_; ++rail) {
      if (outstanding_wqes_[endpoint][rail] + 2U >
          endpoints_[endpoint][rail]->maximum_send_work_requests()) {
        return research::RingSubmitState::kBackpressured;
      }
    }
    ExchangeState state{};
    state.rail_count = rail_count_;
    for (std::uint32_t rail = 0; rail < rail_count_; ++rail) {
      const std::uint64_t local_doorbell =
          region.control_offset +
          (rail == 0
               ? offsetof(spark_transport::DoorbellControl, command_sequence)
               : offsetof(spark_transport::DoorbellControl, mismatch_count));
      const std::uint64_t remote_doorbell =
          region.control_offset +
          (rail == 0
               ? offsetof(spark_transport::DoorbellControl, remote_sequence)
               : offsetof(spark_transport::DoorbellControl, reserved));
      const std::uint64_t completion_id = allocate_completion_id(endpoint, rail);
      state.completion_ids[rail] = completion_id;
      pending_cq_[endpoint][rail].insert(completion_id);
      completion_wqes_[endpoint][rail].emplace(completion_id, 2U);
      outstanding_wqes_[endpoint][rail] += 2U;
      store_release(word(endpoint, local_doorbell), request.doorbell_token);
      endpoints_[endpoint][rail]->write(
          request.span.local_send_offset + rail * rail_bytes,
          request.span.remote_receive_offset + rail * rail_bytes,
          rail_bytes, completion_id, false);
      endpoints_[endpoint][rail]->write(
          local_doorbell, remote_doorbell, sizeof(request.doorbell_token),
          completion_id, true);
      payload_bytes_[endpoint][rail] += rail_bytes;
      doorbell_bytes_[endpoint][rail] += sizeof(request.doorbell_token);
    }
    exchanges_[direction].emplace(request.work_id, state);
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_exchange(
      const research::BidirectionalRingExchangeRequest& request) override {
    if (!drain_completions()) return research::RingPollState::kFatal;
    const auto direction = direction_index(request.direction);
    const auto found = exchanges_[direction].find(request.work_id);
    if (found == exchanges_[direction].end()) {
      return research::RingPollState::kFatal;
    }
    const auto incoming = endpoint_index(request.incoming_endpoint);
    const auto outgoing = endpoint_index(request.outgoing_endpoint);
    const auto region = spark_transport::tp4_tiled_slot_region(
        layout_, request.ticket.slot, 0);
    const auto inbound_offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, remote_sequence);
    for (std::uint32_t rail = 0; rail < found->second.rail_count; ++rail) {
      const auto offset = rail == 0
                              ? inbound_offset
                              : region.control_offset +
                                    offsetof(spark_transport::DoorbellControl,
                                             reserved);
      const auto observed = load_acquire(word(incoming, offset));
      if (observed > request.doorbell_token) {
        fatal_ = true;
        return research::RingPollState::kFatal;
      }
      if (observed < request.doorbell_token ||
          pending_cq_[outgoing][rail].count(
              found->second.completion_ids[rail]) != 0) {
        return research::RingPollState::kPending;
      }
    }
    exchanges_[direction].erase(found);
    return research::RingPollState::kComplete;
  }

  research::RingSubmitState try_publish_consumed_through(
      const research::BidirectionalRingCreditRequest& request) override {
    if (!drain_completions()) return research::RingSubmitState::kFatal;
    const auto direction = direction_index(request.direction);
    if (credits_[direction].active) {
      return research::RingSubmitState::kBackpressured;
    }
    const auto endpoint = endpoint_index(request.endpoint);
    const auto region = spark_transport::tp4_tiled_slot_region(layout_, 0, 0);
    const auto local_offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, consumer_sequence);
    const auto remote_offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, acknowledgement_sequence);
    constexpr std::uint32_t rail = 0;
    if (outstanding_wqes_[endpoint][rail] + 1U >
        endpoints_[endpoint][rail]->maximum_send_work_requests()) {
      return research::RingSubmitState::kBackpressured;
    }
    const std::uint64_t completion_id = allocate_completion_id(endpoint, rail);
    credits_[direction] = {true, request.work_id, completion_id};
    pending_cq_[endpoint][rail].insert(completion_id);
    completion_wqes_[endpoint][rail].emplace(completion_id, 1U);
    ++outstanding_wqes_[endpoint][rail];
    store_release(word(endpoint, local_offset), request.wire_credit);
    endpoints_[endpoint][rail]->write(local_offset, remote_offset,
                                      sizeof(request.wire_credit), completion_id,
                                      true);
    credit_bytes_[endpoint][rail] += sizeof(request.wire_credit);
    return research::RingSubmitState::kAccepted;
  }

  research::RingPollState poll_published_consumed_through(
      const research::BidirectionalRingCreditRequest& request) override {
    if (!drain_completions()) return research::RingPollState::kFatal;
    const auto direction = direction_index(request.direction);
    auto& credit = credits_[direction];
    if (!credit.active || credit.logical_work_id != request.work_id) {
      return research::RingPollState::kFatal;
    }
    const auto endpoint = endpoint_index(request.endpoint);
    if (pending_cq_[endpoint][0].count(credit.completion_id) != 0) {
      return research::RingPollState::kPending;
    }
    credit = {};
    return research::RingPollState::kComplete;
  }

  research::RingCreditPollState poll_peer_consumed_through(
      spark_transport::Tp4PrefillDirection direction,
      std::uint64_t& wire_credit) override {
    if (!drain_completions()) return research::RingCreditPollState::kFatal;
    const auto direction_value = direction_index(direction);
    // Endpoint identity depends on rank, so use the opposite of the credit's
    // incoming endpoint as encoded by the direction-local exchange traffic.
    const auto endpoint = direction ==
                                  spark_transport::Tp4PrefillDirection::kClockwise
                              ? clockwise_outgoing_endpoint_
                              : counterclockwise_outgoing_endpoint_;
    const auto region = spark_transport::tp4_tiled_slot_region(layout_, 0, 0);
    const auto offset =
        region.control_offset +
        offsetof(spark_transport::DoorbellControl, acknowledgement_sequence);
    wire_credit = load_acquire(word(endpoint, offset));
    if (wire_credit == 0 || wire_credit == peer_credit_[direction_value]) {
      return research::RingCreditPollState::kNoUpdate;
    }
    peer_credit_[direction_value] = wire_credit;
    return research::RingCreditPollState::kUpdate;
  }

  void set_direction_endpoints(std::uint32_t rank) {
    clockwise_outgoing_endpoint_ = endpoint_index(
        spark_transport::tp4_prefill_outgoing_endpoint(
            rank, spark_transport::Tp4PrefillDirection::kClockwise));
    counterclockwise_outgoing_endpoint_ = endpoint_index(
        spark_transport::tp4_prefill_outgoing_endpoint(
            rank, spark_transport::Tp4PrefillDirection::kCounterClockwise));
  }

  bool drain_all() noexcept {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (has_pending_cq() && !fatal_) {
      if (!drain_completions()) return false;
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::yield();
    }
    return !fatal_ && exchanges_[0].empty() && exchanges_[1].empty() &&
           !credits_[0].active && !credits_[1].active;
  }

  const std::array<std::array<std::uint64_t, 2>, 2>& payload_bytes() const noexcept {
    return payload_bytes_;
  }
  const std::array<std::array<std::uint64_t, 2>, 2>& doorbell_bytes() const noexcept {
    return doorbell_bytes_;
  }
  const std::array<std::array<std::uint64_t, 2>, 2>& credit_bytes() const noexcept {
    return credit_bytes_;
  }
  const std::array<std::array<std::uint64_t, 2>, 2>& cq_completions() const noexcept {
    return cq_completions_;
  }

 private:
  struct CreditState {
    bool active{};
    std::uint64_t logical_work_id{};
    std::uint64_t completion_id{};
  };

  struct ExchangeState {
    std::uint32_t rail_count{};
    std::array<std::uint64_t, 2> completion_ids{};
  };

  std::uint64_t allocate_completion_id(std::size_t endpoint,
                                       std::size_t rail) {
    if (next_completion_id_[endpoint][rail] ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("verbs completion ID exhausted");
    }
    return next_completion_id_[endpoint][rail]++;
  }

  bool has_pending_cq() const noexcept {
    for (const auto& neighbor : pending_cq_) {
      for (const auto& rail : neighbor) {
        if (!rail.empty()) return true;
      }
    }
    return false;
  }

  bool drain_completions() noexcept {
    if (fatal_) return false;
    try {
      for (std::size_t endpoint = 0; endpoint < endpoints_.size(); ++endpoint) {
        for (std::size_t rail = 0; rail < rail_count_; ++rail) {
          spark_transport::SendCompletion completions[32]{};
          const auto count = endpoints_[endpoint][rail]->poll_send_completions(
              completions, 32);
          for (std::size_t index = 0; index < count; ++index) {
            const auto id = completions[index].work_id;
            if (pending_cq_[endpoint][rail].erase(id) != 1) {
              fatal_ = true;
              return false;
            }
            const auto found = completion_wqes_[endpoint][rail].find(id);
            if (found == completion_wqes_[endpoint][rail].end() ||
                outstanding_wqes_[endpoint][rail] < found->second) {
              fatal_ = true;
              return false;
            }
            outstanding_wqes_[endpoint][rail] -= found->second;
            completion_wqes_[endpoint][rail].erase(found);
            ++cq_completions_[endpoint][rail];
          }
        }
      }
      return true;
    } catch (...) {
      fatal_ = true;
      return false;
    }
  }

  std::uint64_t* word(std::size_t endpoint, std::uint64_t offset) const {
    if (offset + sizeof(std::uint64_t) > arenas_[endpoint]->size() ||
        offset % alignof(std::uint64_t) != 0) {
      throw std::out_of_range("bidirectional control offset outside arena");
    }
    return reinterpret_cast<std::uint64_t*>(
        static_cast<std::uint8_t*>(arenas_[endpoint]->host_data()) + offset);
  }

  EndpointMatrix endpoints_{};
  std::array<spark_transport::MemoryBuffer*, 2> arenas_{};
  spark_transport::Tp4TiledPoolLayout layout_{};
  std::uint32_t rail_count_{1};
  std::array<std::array<std::uint64_t, 2>, 2> next_completion_id_{{{1, 1}, {1, 1}}};
  std::array<std::array<std::unordered_set<std::uint64_t>, 2>, 2> pending_cq_{};
  std::array<std::array<std::unordered_map<std::uint64_t, std::uint32_t>, 2>, 2>
      completion_wqes_{};
  std::array<std::array<std::uint32_t, 2>, 2> outstanding_wqes_{};
  std::array<std::unordered_map<std::uint64_t, ExchangeState>, 2> exchanges_{};
  std::array<CreditState, 2> credits_{};
  std::array<std::uint64_t, 2> peer_credit_{};
  std::array<std::array<std::uint64_t, 2>, 2> payload_bytes_{};
  std::array<std::array<std::uint64_t, 2>, 2> doorbell_bytes_{};
  std::array<std::array<std::uint64_t, 2>, 2> credit_bytes_{};
  std::array<std::array<std::uint64_t, 2>, 2> cq_completions_{};
  std::size_t clockwise_outgoing_endpoint_{};
  std::size_t counterclockwise_outgoing_endpoint_{1};
  bool fatal_{};
};
