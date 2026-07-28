#include "spark_transport/tp4_indexer_graph.hpp"

#include <cassert>
#include <cstdint>

int main() {
  using namespace spark_transport;

  const auto arena = tp4_indexer_graph_arena_contract();
  assert(arena.maximum_q == 40);
  assert(arena.input_capacity_bytes == 655360);
  assert(arena.round0_transfer_capacity_bytes == 655360);
  assert(arena.round1_transfer_capacity_bytes == 1310720);
  assert(arena.output_capacity_bytes == 2621440);
  assert(arena.round0_arena_bytes == 1310784);
  assert(arena.round1_arena_bytes == 2621504);
  assert(arena.command_ring_bytes == sizeof(Tp4GraphCommandRing));
  assert(arena.total_mapped_bytes ==
         arena.round0_arena_bytes + arena.round1_arena_bytes +
             arena.command_ring_bytes);

  const auto ports = tp4_indexer_graph_default_port_pair();
  assert(ports.control_port0 == 9462);
  assert(ports.control_port1 == 9463);

  // Every Q1..Q40 is formula-admitted into the same fixed arena and port
  // namespace; no exact-shape port/session expansion is needed.
  for (std::uint32_t q = 1;
       q <= kTp4IndexerGraphMaximumQ; ++q) {
    Tp4IndexerGraphDescriptor by_q{};
    Tp4IndexerGraphDescriptor by_bytes{};
    assert(tp4_indexer_graph_descriptor_from_q(q, &by_q));
    assert(by_q.q == q);
    assert(by_q.input_bytes ==
           q * kTp4IndexerGraphBytesPerRow);
    assert(by_q.output_bytes ==
           by_q.input_bytes * kTp4IndexerGraphWorldSize);
    assert(tp4_indexer_graph_descriptor_valid(by_q));
    assert(tp4_indexer_graph_descriptor_from_input_bytes(
        by_q.input_bytes, &by_bytes));
    assert(by_bytes.q == q);
    assert(by_bytes.input_bytes == by_q.input_bytes);
    assert(by_bytes.output_bytes == by_q.output_bytes);
  }

  Tp4IndexerGraphDescriptor descriptor{};
  assert(!tp4_indexer_graph_descriptor_from_q(0, &descriptor));
  assert(!tp4_indexer_graph_descriptor_from_q(41, &descriptor));
  assert(!tp4_indexer_graph_descriptor_from_input_bytes(
      kTp4IndexerGraphBytesPerRow - 1, &descriptor));
  assert(!tp4_indexer_graph_descriptor_from_input_bytes(
      kTp4IndexerGraphMaximumInputBytes +
          kTp4IndexerGraphBytesPerRow,
      &descriptor));

  Tp4GraphCommandRing ring{};
  Tp4IndexerGraphFoundation foundation(&ring);
  assert(foundation.capture_q(1));
  assert(foundation.capture_q(23));
  assert(foundation.capture_q(40));
  assert(!foundation.capture_q(41));

  auto status = foundation.status();
  assert(status.capture_configured);
  assert(status.captured_nodes == 3);
  assert(status.rejected_captures == 1);
  assert((status.captured_q_mask & (std::uint64_t{1} << 22)) != 0);
  assert(status.published_sequence == 0);

  Tp4IndexerGraphDescriptor q23{};
  assert(tp4_indexer_graph_descriptor_from_q(23, &q23));
  std::uint64_t sequence{};
  assert(foundation.publish_reference_replay(q23, true, &sequence));
  assert(sequence == 1);

  Tp4GraphCommand command{};
  assert(foundation.try_consume_next(&command));
  assert(command.sequence == 1);
  assert(command.trace == 1);
  assert(command.kind ==
         Tp4GraphCommandKind::kIndexerAllgather);
  assert(command.parameter ==
         kTp4IndexerGraphDescriptorVersion);
  assert(command.q == 23);
  assert(command.payload_bytes ==
         23 * kTp4IndexerGraphBytesPerRow);
  assert(foundation.complete(1));

  status = foundation.status();
  assert(status.published_sequence == 1);
  assert(status.consumed_sequence == 1);
  assert(status.completed_sequence == 1);
  assert(status.overflow_sequence == 0);
  assert(status.invalid_commands == 0);

  // Q2 was valid by formula but no Q2 node was captured in this model.
  // A live replay can only originate from a successfully captured node.
  Tp4IndexerGraphDescriptor q2{};
  assert(tp4_indexer_graph_descriptor_from_q(2, &q2));
  assert(!foundation.publish_reference_replay(
      q2, false, &sequence));
  assert(foundation.status().rejected_reference_replays == 1);

  // Graph definitions freeze once replay begins, even though all Q values
  // remain admissible for a new session/capture epoch.
  assert(!foundation.capture_q(2));
  assert(foundation.status().rejected_captures == 2);

  // A 49,152-byte DCP-family command collides with indexer Q3 by bytes.
  // The family tag/version guard must reject it before acknowledgement.
  Tp4GraphCommandRing collision_ring{};
  Tp4IndexerGraphFoundation collision(&collision_ring);
  assert(collision.capture_q(3));
  collision_ring.producer.claimed_sequence = 1;
  collision_ring.producer.published_sequence = 1;
  collision_ring.commands[0].sequence = 1;
  collision_ring.commands[0].kind =
      Tp4GraphCommandKind::kDcpQuery;
  collision_ring.commands[0].parameter = 0;
  collision_ring.commands[0].q = 3;
  collision_ring.commands[0].payload_bytes = 49152;
  assert(!collision.try_consume_next(&command));
  const auto collision_status = collision.status();
  assert(collision_status.consumed_sequence == 0);
  assert(collision_status.completed_sequence == 0);
  assert(collision_status.overflow_sequence == 1);
  assert(collision_status.invalid_commands == 1);

  Tp4GraphCommandRing bad_completion_ring{};
  Tp4IndexerGraphFoundation bad_completion(&bad_completion_ring);
  assert(bad_completion.capture_q(1));
  Tp4IndexerGraphDescriptor q1{};
  assert(tp4_indexer_graph_descriptor_from_q(1, &q1));
  assert(bad_completion.publish_reference_replay(
      q1, false, &sequence));
  assert(bad_completion.try_consume_next(&command));
  assert(!bad_completion.complete(2));
  assert(bad_completion.status().overflow_sequence == 2);
  assert(bad_completion.status().invalid_commands == 1);

  return 0;
}
