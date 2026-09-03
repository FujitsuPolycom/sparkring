#include "spark_transport/tp4_bidirectional_prefill.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>

int main() {
  using namespace spark_transport;

  static_assert(kTp4PrefillStageCount == 6);
  static_assert(kTp4PrefillPayloadBytes == 16U * 1024U * 1024U);
  static_assert(kTp4PrefillBytesPerDirection == 12U * 1024U * 1024U);
  static_assert(kTp4PrefillBytesPerRank == 24U * 1024U * 1024U);

  constexpr std::array<Tp4PrefillEndpoint, 4> clockwise_endpoints{
      Tp4PrefillEndpoint::kXor1, Tp4PrefillEndpoint::kXor3,
      Tp4PrefillEndpoint::kXor1, Tp4PrefillEndpoint::kXor3};
  constexpr std::array<Tp4PrefillEndpoint, 4> counter_clockwise_endpoints{
      Tp4PrefillEndpoint::kXor3, Tp4PrefillEndpoint::kXor1,
      Tp4PrefillEndpoint::kXor3, Tp4PrefillEndpoint::kXor1};

  for (std::uint32_t rank = 0; rank < kTp4PrefillRankCount; ++rank) {
    assert(tp4_prefill_successor(rank, Tp4PrefillDirection::kClockwise) ==
           (rank + 1U) % kTp4PrefillRankCount);
    assert(tp4_prefill_successor(
               rank, Tp4PrefillDirection::kCounterClockwise) ==
           (rank + kTp4PrefillRankCount - 1U) % kTp4PrefillRankCount);
    assert(tp4_prefill_outgoing_endpoint(
               rank, Tp4PrefillDirection::kClockwise) ==
           clockwise_endpoints[rank]);
    assert(tp4_prefill_outgoing_endpoint(
               rank, Tp4PrefillDirection::kCounterClockwise) ==
           counter_clockwise_endpoints[rank]);
    assert(tp4_prefill_incoming_endpoint(
               rank, Tp4PrefillDirection::kClockwise) ==
           counter_clockwise_endpoints[rank]);
    assert(tp4_prefill_incoming_endpoint(
               rank, Tp4PrefillDirection::kCounterClockwise) ==
           clockwise_endpoints[rank]);
  }

  constexpr std::array<Tp4PrefillDirection, 2> directions{
      Tp4PrefillDirection::kClockwise,
      Tp4PrefillDirection::kCounterClockwise};
  for (const auto direction : directions) {
    for (std::uint32_t rank = 0; rank < kTp4PrefillRankCount; ++rank) {
      std::size_t bytes = 0;
      for (std::uint32_t stage = 0; stage < kTp4PrefillStageCount; ++stage) {
        const auto transfer = tp4_prefill_stage(rank, direction, stage);
        assert(transfer.rank == rank);
        assert(transfer.direction == direction);
        assert(transfer.outgoing_peer == tp4_prefill_successor(rank, direction));
        assert(transfer.send_shard < kTp4PrefillRankCount);
        assert(transfer.receive_shard < kTp4PrefillRankCount);
        assert(transfer.bytes == kTp4PrefillShardBytes);
        assert(transfer.phase ==
               (stage < kTp4PrefillReduceScatterStages
                    ? Tp4PrefillPhase::kReduceScatter
                    : Tp4PrefillPhase::kAllGather));
        bytes += transfer.bytes;
      }
      assert(bytes == kTp4PrefillBytesPerDirection);
    }
  }

  // Rank zero's shard sequence is a compact oracle for the two mirrored
  // rings.  The receive shard at each stage must equal the peer's send shard.
  constexpr std::array<std::uint32_t, 6> clockwise_send{0, 3, 2, 1, 0, 3};
  constexpr std::array<std::uint32_t, 6> clockwise_receive{3, 2, 1, 0, 3, 2};
  constexpr std::array<std::uint32_t, 6> counter_send{0, 1, 2, 3, 0, 1};
  constexpr std::array<std::uint32_t, 6> counter_receive{1, 2, 3, 0, 1, 2};
  for (std::uint32_t stage = 0; stage < kTp4PrefillStageCount; ++stage) {
    const auto clockwise = tp4_prefill_stage(
        0, Tp4PrefillDirection::kClockwise, stage);
    const auto counter = tp4_prefill_stage(
        0, Tp4PrefillDirection::kCounterClockwise, stage);
    assert(clockwise.send_shard == clockwise_send[stage]);
    assert(clockwise.receive_shard == clockwise_receive[stage]);
    assert(counter.send_shard == counter_send[stage]);
    assert(counter.receive_shard == counter_receive[stage]);
  }

  for (const auto direction : directions) {
    for (std::uint32_t stage = 0; stage < kTp4PrefillStageCount; ++stage) {
      for (std::uint32_t receiver = 0; receiver < kTp4PrefillRankCount;
           ++receiver) {
        const std::uint32_t sender = tp4_prefill_successor(
            receiver,
            direction == Tp4PrefillDirection::kClockwise
                ? Tp4PrefillDirection::kCounterClockwise
                : Tp4PrefillDirection::kClockwise);
        const auto sent = tp4_prefill_stage(sender, direction, stage);
        const auto received = tp4_prefill_stage(receiver, direction, stage);
        assert(sent.send_shard == received.receive_shard);
      }
    }
  }

  constexpr auto verification = verify_tp4_bidirectional_prefill();
  static_assert(verification.complete());
  constexpr Tp4PrefillContributorMask all_ranks = 0x0fU;
  for (const auto& direction : verification.contributors) {
    for (const auto& rank : direction) {
      for (const auto shard : rank) {
        assert(shard == all_ranks);
      }
    }
  }
  for (const auto& direction : verification.transmitted_bytes) {
    for (const auto bytes : direction) {
      assert(bytes == kTp4PrefillBytesPerDirection);
    }
  }

  bool rejected_rank = false;
  try {
    static_cast<void>(tp4_prefill_stage(
        kTp4PrefillRankCount, Tp4PrefillDirection::kClockwise, 0));
  } catch (const std::out_of_range&) {
    rejected_rank = true;
  }
  assert(rejected_rank);

  bool rejected_stage = false;
  try {
    static_cast<void>(tp4_prefill_stage(
        0, Tp4PrefillDirection::kClockwise, kTp4PrefillStageCount));
  } catch (const std::out_of_range&) {
    rejected_stage = true;
  }
  assert(rejected_stage);
}
