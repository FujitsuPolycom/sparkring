#include "spark_transport/tp4_graph_command.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

int main() {
  using spark_transport::Tp4GraphCommand;
  using spark_transport::Tp4GraphCommandKind;
  using spark_transport::Tp4GraphCommandRing;
  using spark_transport::kTp4GraphAllreduceMaximumQ;
  using spark_transport::kTp4GraphCommandCapacity;
  using spark_transport::kTp4GraphDoorbellQBits;
  using spark_transport::kTp4GraphDoorbellQMask;
  using spark_transport::kTp4GraphMaximumDoorbellSequence;
  using spark_transport::kTp4GraphMaximumQ;
  using spark_transport::tp4_graph_allreduce_command_descriptor_valid;
  using spark_transport::tp4_graph_allreduce_command_publish_descriptor;
  using spark_transport::tp4_graph_allreduce_command_try_consume_descriptor;
  using spark_transport::tp4_graph_command_complete;
  using spark_transport::tp4_graph_command_claimed;
  using spark_transport::tp4_graph_command_completed;
  using spark_transport::tp4_graph_command_consumed;
  using spark_transport::tp4_graph_command_descriptor_valid;
  using spark_transport::tp4_graph_command_layout_valid;
  using spark_transport::tp4_graph_command_overflow;
  using spark_transport::tp4_graph_command_publish;
  using spark_transport::tp4_graph_command_publish_descriptor;
  using spark_transport::tp4_graph_command_publish_tagged_layout;
  using spark_transport::tp4_graph_command_published;
  using spark_transport::tp4_graph_command_try_peek;
  using spark_transport::tp4_graph_command_try_consume;
  using spark_transport::tp4_graph_command_try_consume_descriptor;
  using spark_transport::tp4_graph_command_try_consume_layout;
  using spark_transport::tp4_graph_command_try_consume_tagged_layout;
  using spark_transport::tp4_graph_doorbell_token;
  using spark_transport::tp4_graph_doorbell_token_valid;
  using spark_transport::tp4_graph_payload_bytes;

  Tp4GraphCommandRing ring{};
  Tp4GraphCommand command{};
  assert(!tp4_graph_command_publish(nullptr, false, nullptr));
  assert(!tp4_graph_command_publish_descriptor(
      nullptr, false, 1, tp4_graph_payload_bytes(1), nullptr));
  assert(!tp4_graph_command_try_consume(nullptr, 1, &command));
  assert(!tp4_graph_command_try_consume_descriptor(
      nullptr, 1, &command));
  assert(!tp4_graph_command_try_consume_layout(
      nullptr, 1, 77440, 6, &command));
  assert(!tp4_graph_command_try_consume(&ring, 1, nullptr));
  assert(!tp4_graph_command_try_consume_descriptor(
      &ring, 1, nullptr));
  assert(!tp4_graph_command_try_consume_layout(
      &ring, 1, 77440, 6, nullptr));
  tp4_graph_command_complete(nullptr, 1);
  assert(tp4_graph_command_published(&ring) == 0);
  assert(tp4_graph_command_claimed(&ring) == 0);
  assert(tp4_graph_command_consumed(&ring) == 0);
  assert(tp4_graph_command_completed(&ring) == 0);
  assert(tp4_graph_command_overflow(&ring) == 0);

  for (std::uint32_t q = 1; q <= kTp4GraphMaximumQ; ++q) {
    assert(tp4_graph_command_descriptor_valid(
        q, tp4_graph_payload_bytes(q)));
    assert(!tp4_graph_command_descriptor_valid(
        q, tp4_graph_payload_bytes(q) - 1));
    assert(!tp4_graph_command_descriptor_valid(
        q, tp4_graph_payload_bytes(q) + 1));
  }
  assert(!tp4_graph_command_descriptor_valid(
      0, tp4_graph_payload_bytes(0)));
  assert(!tp4_graph_command_descriptor_valid(
      kTp4GraphMaximumQ + 1,
      tp4_graph_payload_bytes(kTp4GraphMaximumQ + 1)));
  static_assert(kTp4GraphMaximumQ == 40);
  static_assert(kTp4GraphAllreduceMaximumQ == 512);
  static_assert(tp4_graph_payload_bytes(kTp4GraphAllreduceMaximumQ) ==
                6U * 1024U * 1024U);
  assert(tp4_graph_allreduce_command_descriptor_valid(
      kTp4GraphAllreduceMaximumQ,
      tp4_graph_payload_bytes(kTp4GraphAllreduceMaximumQ)));
  assert(!tp4_graph_allreduce_command_descriptor_valid(
      kTp4GraphAllreduceMaximumQ + 1,
      tp4_graph_payload_bytes(kTp4GraphAllreduceMaximumQ + 1)));
  assert(tp4_graph_command_layout_valid(6, 6 * 77440, 77440, 6));
  assert(!tp4_graph_command_layout_valid(0, 0, 77440, 6));
  assert(!tp4_graph_command_layout_valid(7, 7 * 77440, 77440, 6));
  assert(!tp4_graph_command_layout_valid(6, 6 * 77440 - 1, 77440, 6));
  assert(!tp4_graph_command_layout_valid(1, 1, 0, 6));
  assert(!tp4_graph_doorbell_token_valid(0, 1));
  assert(!tp4_graph_doorbell_token_valid(1, 0));
  assert(tp4_graph_doorbell_token_valid(
      1, kTp4GraphAllreduceMaximumQ));
  assert(!tp4_graph_doorbell_token_valid(
      1, kTp4GraphAllreduceMaximumQ + 1));
  assert(tp4_graph_doorbell_token_valid(
      kTp4GraphMaximumDoorbellSequence,
      kTp4GraphAllreduceMaximumQ));
  assert(!tp4_graph_doorbell_token_valid(
      kTp4GraphMaximumDoorbellSequence + 1,
      kTp4GraphAllreduceMaximumQ));
  for (std::uint64_t graph_sequence = 1;
       graph_sequence <= 32; ++graph_sequence) {
    std::uint64_t previous{};
    for (std::uint32_t q = 1; q <= kTp4GraphMaximumQ; ++q) {
      const std::uint64_t token =
          tp4_graph_doorbell_token(graph_sequence, q);
      assert(token > previous);
      assert((token >> kTp4GraphDoorbellQBits) == graph_sequence);
      assert((token & kTp4GraphDoorbellQMask) == q);
      previous = token;
    }
  }
  const auto q512_token = tp4_graph_doorbell_token(
      7, kTp4GraphAllreduceMaximumQ);
  assert((q512_token >> kTp4GraphDoorbellQBits) == 7);
  assert((q512_token & kTp4GraphDoorbellQMask) ==
         kTp4GraphAllreduceMaximumQ);
  assert(q512_token < tp4_graph_doorbell_token(8, 1));

  Tp4GraphCommandRing allreduce_q512{};
  std::uint64_t allreduce_q512_sequence{};
  assert(tp4_graph_allreduce_command_publish_descriptor(
      &allreduce_q512, true, kTp4GraphAllreduceMaximumQ,
      tp4_graph_payload_bytes(kTp4GraphAllreduceMaximumQ),
      &allreduce_q512_sequence));
  assert(allreduce_q512_sequence == 1);
  Tp4GraphCommand allreduce_q512_command{};
  assert(tp4_graph_allreduce_command_try_consume_descriptor(
      &allreduce_q512, 1, &allreduce_q512_command));
  assert(allreduce_q512_command.q == kTp4GraphAllreduceMaximumQ);
  assert(allreduce_q512_command.payload_bytes == 6U * 1024U * 1024U);

  std::uint64_t sequence{99};
  assert(!tp4_graph_command_publish_descriptor(
      &ring, false, 0, 0, &sequence));
  assert(sequence == 99);
  assert(!tp4_graph_command_publish_descriptor(
      &ring, false, 7, tp4_graph_payload_bytes(6), &sequence));
  assert(sequence == 99);
  assert(!tp4_graph_command_publish_descriptor(
      &ring, false, 5, tp4_graph_payload_bytes(4), &sequence));
  assert(sequence == 99);
  assert(tp4_graph_command_claimed(&ring) == 0);
  assert(tp4_graph_command_published(&ring) == 0);
  assert(tp4_graph_command_overflow(&ring) == 0);

  assert(tp4_graph_command_publish(&ring, true, &sequence));
  assert(sequence == 1);
  assert(tp4_graph_command_claimed(&ring) == 1);
  assert(tp4_graph_command_published(&ring) == 1);

  assert(!tp4_graph_command_try_consume(&ring, 2, &command));
  assert(tp4_graph_command_try_consume_descriptor(
      &ring, 1, &command));
  assert(command.sequence == 1);
  assert(command.trace == 1);
  assert(command.q == 1);
  assert(command.payload_bytes == tp4_graph_payload_bytes(1));
  assert(command.kind == Tp4GraphCommandKind::kLegacy);
  assert(command.parameter == 0);
  assert(tp4_graph_command_consumed(&ring) == 1);

  tp4_graph_command_complete(&ring, 1);
  assert(tp4_graph_command_completed(&ring) == 1);

  for (std::uint64_t expected = 2;
       expected <= kTp4GraphCommandCapacity + 1; ++expected) {
    assert(tp4_graph_command_publish(&ring, false, &sequence));
    assert(sequence == expected);
  }
  assert(ring.commands[0].sequence == kTp4GraphCommandCapacity + 1);
  assert(!tp4_graph_command_publish(&ring, false, &sequence));
  assert(tp4_graph_command_claimed(&ring) ==
         kTp4GraphCommandCapacity + 1);
  assert(tp4_graph_command_overflow(&ring) == 0);

  // Completion frees the oldest slot for the next replay.
  tp4_graph_command_complete(&ring, 2);
  assert(tp4_graph_command_publish(&ring, false, &sequence));
  assert(sequence == kTp4GraphCommandCapacity + 2);
  assert(tp4_graph_command_claimed(&ring) == sequence);
  assert(tp4_graph_command_published(&ring) == sequence);

  Tp4GraphCommandRing invalid_order{};
  invalid_order.consumer.completed_sequence = 1;
  assert(!tp4_graph_command_publish(
      &invalid_order, false, &sequence));
  assert(tp4_graph_command_claimed(&invalid_order) == 0);
  assert(tp4_graph_command_overflow(&invalid_order) == 1);

  Tp4GraphCommandRing exhausted{};
  exhausted.producer.claimed_sequence =
      kTp4GraphMaximumDoorbellSequence;
  assert(!tp4_graph_command_publish(&exhausted, false, &sequence));
  assert(tp4_graph_command_overflow(&exhausted) ==
         kTp4GraphMaximumDoorbellSequence);

  Tp4GraphCommandRing out_of_order_completion{};
  tp4_graph_command_complete(&out_of_order_completion, 2);
  assert(tp4_graph_command_completed(&out_of_order_completion) == 0);
  assert(tp4_graph_command_overflow(&out_of_order_completion) == 2);

  // Model the live dual-graph probe: graph A contains three commands and
  // graph B contains 128. Both executable graphs share one monotonically
  // advancing session, alternate repeatedly, and may exceed ring capacity
  // as long as each command is consumed/completed before its slot is reused.
  Tp4GraphCommandRing alternating_graphs{};
  std::uint64_t alternating_sequence{};
  const auto complete_graph =
      [&](std::size_t nodes, bool trace) {
        const std::uint64_t graph_start = alternating_sequence;
        for (std::size_t node = 0; node < nodes; ++node) {
          std::uint64_t published{};
          assert(tp4_graph_command_publish(
              &alternating_graphs, trace, &published));
          assert(published == alternating_sequence + 1);
          Tp4GraphCommand consumed{};
          assert(tp4_graph_command_try_consume(
              &alternating_graphs, published, &consumed));
          assert(consumed.sequence == published);
          assert(consumed.trace == (trace ? 1U : 0U));
          assert(consumed.q == 1);
          assert(consumed.payload_bytes ==
                 tp4_graph_payload_bytes(1));
          tp4_graph_command_complete(
              &alternating_graphs, published);
          alternating_sequence = published;
        }
        assert(alternating_sequence == graph_start + nodes);
        assert(tp4_graph_command_published(&alternating_graphs) ==
               alternating_sequence);
        assert(tp4_graph_command_consumed(&alternating_graphs) ==
               alternating_sequence);
        assert(tp4_graph_command_completed(&alternating_graphs) ==
               alternating_sequence);
        assert(tp4_graph_command_overflow(&alternating_graphs) == 0);
      };
  for (int replay_pair = 0; replay_pair < 4; ++replay_pair) {
    complete_graph(3, false);
    complete_graph(128, true);
  }
  assert(alternating_sequence == 4U * (3U + 128U));

  // Alternate Q1/Q40 descriptors across several full ring wraps. The strict
  // consumer must observe the exact descriptor stored in each reused slot.
  Tp4GraphCommandRing mixed_q{};
  constexpr std::uint64_t mixed_replays =
      3U * kTp4GraphCommandCapacity + 7U;
  for (std::uint64_t expected = 1;
       expected <= mixed_replays; ++expected) {
    const std::uint32_t q =
        (expected & 1U) == 0 ? kTp4GraphMaximumQ : 1U;
    const std::uint32_t payload_bytes =
        tp4_graph_payload_bytes(q);
    std::uint64_t published{};
    assert(tp4_graph_command_publish_descriptor(
        &mixed_q, (expected % 3U) == 0, q, payload_bytes,
        &published));
    assert(published == expected);
    const auto& slot = mixed_q.commands[
        (expected - 1) % kTp4GraphCommandCapacity];
    assert(slot.sequence == expected);
    assert(slot.q == q);
    assert(slot.payload_bytes == payload_bytes);

    Tp4GraphCommand consumed{};
    assert(tp4_graph_command_try_consume_descriptor(
        &mixed_q, expected, &consumed));
    assert(consumed.sequence == expected);
    assert(consumed.trace == ((expected % 3U) == 0 ? 1U : 0U));
    assert(consumed.q == q);
    assert(consumed.payload_bytes == payload_bytes);
    tp4_graph_command_complete(&mixed_q, expected);
  }
  assert(tp4_graph_command_claimed(&mixed_q) == mixed_replays);
  assert(tp4_graph_command_published(&mixed_q) == mixed_replays);
  assert(tp4_graph_command_consumed(&mixed_q) == mixed_replays);
  assert(tp4_graph_command_completed(&mixed_q) == mixed_replays);
  assert(tp4_graph_command_overflow(&mixed_q) == 0);

  // A corrupted descriptor may have a released sequence, but the strict
  // consumer fails closed before acknowledging it and records the sequence
  // as the fatal ordering/integrity marker.
  Tp4GraphCommandRing corrupt_descriptor{};
  corrupt_descriptor.producer.claimed_sequence = 1;
  corrupt_descriptor.producer.published_sequence = 1;
  corrupt_descriptor.commands[0].sequence = 1;
  corrupt_descriptor.commands[0].q = 5;
  corrupt_descriptor.commands[0].payload_bytes =
      tp4_graph_payload_bytes(4);
  assert(!tp4_graph_command_try_consume_descriptor(
      &corrupt_descriptor, 1, &command));
  assert(tp4_graph_command_consumed(&corrupt_descriptor) == 0);
  assert(tp4_graph_command_overflow(&corrupt_descriptor) == 1);

  // The same ring layout can serve vocabulary commands, but validation must
  // use that family's 77,440-byte row rather than the all-reduce row.
  Tp4GraphCommandRing vocab_descriptor{};
  vocab_descriptor.producer.claimed_sequence = 1;
  vocab_descriptor.producer.published_sequence = 1;
  vocab_descriptor.commands[0].sequence = 1;
  vocab_descriptor.commands[0].q = 6;
  vocab_descriptor.commands[0].payload_bytes = 6 * 77440;
  assert(tp4_graph_command_try_consume_layout(
      &vocab_descriptor, 1, 77440, 6, &command));
  assert(command.q == 6);
  assert(command.payload_bytes == 6 * 77440);
  assert(tp4_graph_command_consumed(&vocab_descriptor) == 1);

  Tp4GraphCommandRing corrupt_vocab_descriptor{};
  corrupt_vocab_descriptor.producer.claimed_sequence = 1;
  corrupt_vocab_descriptor.producer.published_sequence = 1;
  corrupt_vocab_descriptor.commands[0].sequence = 1;
  corrupt_vocab_descriptor.commands[0].q = 6;
  corrupt_vocab_descriptor.commands[0].payload_bytes = 5 * 77440;
  assert(!tp4_graph_command_try_consume_layout(
      &corrupt_vocab_descriptor, 1, 77440, 6, &command));
  assert(tp4_graph_command_consumed(&corrupt_vocab_descriptor) == 0);
  assert(tp4_graph_command_overflow(&corrupt_vocab_descriptor) == 1);

  Tp4GraphCommandRing already_failed_descriptor{};
  already_failed_descriptor.producer.claimed_sequence = 1;
  already_failed_descriptor.producer.published_sequence = 1;
  already_failed_descriptor.producer.overflow_sequence = 77;
  already_failed_descriptor.commands[0].sequence = 1;
  assert(!tp4_graph_command_try_consume_descriptor(
      &already_failed_descriptor, 1, &command));
  assert(tp4_graph_command_consumed(&already_failed_descriptor) == 0);
  assert(tp4_graph_command_overflow(&already_failed_descriptor) == 77);

  // Query and fused-combine nodes share one DCP graph ring. The tag and
  // family parameter are visible before acknowledgement, then validated
  // together with the exact family row width before consumption.
  constexpr std::uint32_t query_bytes_per_row = 16U * 576U * 2U;
  constexpr std::uint32_t combine_d512_bytes_per_row =
      32U * (512U * 2U + 4U);
  Tp4GraphCommandRing dcp_commands{};
  std::uint64_t dcp_sequence{};
  assert(tp4_graph_command_publish_tagged_layout(
      &dcp_commands, true, Tp4GraphCommandKind::kDcpQuery, 0,
      5, 5 * query_bytes_per_row, query_bytes_per_row,
      kTp4GraphMaximumQ, &dcp_sequence));
  assert(dcp_sequence == 1);
  Tp4GraphCommand dcp_peek{};
  assert(tp4_graph_command_try_peek(
      &dcp_commands, 1, &dcp_peek));
  assert(dcp_peek.kind == Tp4GraphCommandKind::kDcpQuery);
  assert(dcp_peek.parameter == 0);
  assert(dcp_peek.q == 5);
  assert(dcp_peek.payload_bytes == 5 * query_bytes_per_row);
  assert(tp4_graph_command_consumed(&dcp_commands) == 0);
  assert(tp4_graph_command_try_consume_tagged_layout(
      &dcp_commands, 1, Tp4GraphCommandKind::kDcpQuery, 0,
      query_bytes_per_row, kTp4GraphMaximumQ, &command));
  assert(command.kind == Tp4GraphCommandKind::kDcpQuery);
  tp4_graph_command_complete(&dcp_commands, 1);

  assert(tp4_graph_command_publish_tagged_layout(
      &dcp_commands, false, Tp4GraphCommandKind::kDcpCombine, 512,
      4, 4 * combine_d512_bytes_per_row,
      combine_d512_bytes_per_row, kTp4GraphMaximumQ,
      &dcp_sequence));
  assert(dcp_sequence == 2);
  assert(tp4_graph_command_try_peek(
      &dcp_commands, 2, &dcp_peek));
  assert(dcp_peek.kind == Tp4GraphCommandKind::kDcpCombine);
  assert(dcp_peek.parameter == 512);
  assert(tp4_graph_command_try_consume_tagged_layout(
      &dcp_commands, 2, Tp4GraphCommandKind::kDcpCombine, 512,
      combine_d512_bytes_per_row, kTp4GraphMaximumQ, &command));
  tp4_graph_command_complete(&dcp_commands, 2);
  assert(tp4_graph_command_completed(&dcp_commands) == 2);
  assert(tp4_graph_command_overflow(&dcp_commands) == 0);

  Tp4GraphCommandRing corrupt_dcp_kind{};
  corrupt_dcp_kind.producer.claimed_sequence = 1;
  corrupt_dcp_kind.producer.published_sequence = 1;
  corrupt_dcp_kind.commands[0].sequence = 1;
  corrupt_dcp_kind.commands[0].kind =
      Tp4GraphCommandKind::kDcpCombine;
  corrupt_dcp_kind.commands[0].parameter = 0;
  corrupt_dcp_kind.commands[0].q = 1;
  corrupt_dcp_kind.commands[0].payload_bytes = query_bytes_per_row;
  assert(!tp4_graph_command_try_consume_tagged_layout(
      &corrupt_dcp_kind, 1, Tp4GraphCommandKind::kDcpQuery, 0,
      query_bytes_per_row, kTp4GraphMaximumQ, &command));
  assert(tp4_graph_command_consumed(&corrupt_dcp_kind) == 0);
  assert(tp4_graph_command_overflow(&corrupt_dcp_kind) == 1);

  // The CUDA probe alternates this exactly representable offset before every
  // launch. Adjacent replay inputs and corresponding four-rank reductions
  // must differ, which makes stale replay data observable without tolerance.
  const auto input_value = [](std::uint32_t rank, std::size_t element,
                              std::uint64_t replay) {
    const std::uint32_t offset = (replay & 1U) == 0 ? 0U : 16U;
    return ((element * 3U + rank * 5U) & 7U) + 1U + offset;
  };
  const auto reduced_value = [&](std::size_t element,
                                 std::uint64_t replay) {
    std::uint32_t sum{};
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
      sum += input_value(rank, element, replay);
    }
    return sum;
  };
  for (std::size_t element = 0; element < 6144; ++element) {
    assert(input_value(0, element, 1) !=
           input_value(0, element, 2));
    assert(reduced_value(element, 1) !=
           reduced_value(element, 2));
  }
  return 0;
}
