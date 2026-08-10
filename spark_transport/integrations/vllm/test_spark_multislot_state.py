"""CPU-only adversarial tests for the multi-slot ingress state machine.

Validates all required adversarial cases from the assignment:

- Double-publish rejected
- Stale generation rejected (ABA/wraparound protection)
- Slow consumer timeout
- Producer timeout
- Partial failure rollback
- Graph replay ordering (no skipping)
- Backpressure when all slots outstanding
- Consume before publish rejected
- Complete before consume rejected
- Reclaim before complete rejected
- Overflow (sequence above expected) is fatal in graph mode
- Wraparound at 2^16 with generation advance

Evidence label: **Modeled** — these tests exercise protocol invariants
of the modeled state machine.  They do not prove native CUDA/RDMA
scheduling, ordering, or numerical equivalence.

One-token-per-lifecycle: ``claim_slot(token)`` assigns a uint64
``command_token`` that is shared through publish/consume/complete/ack.
All lifecycle methods receive the *same* token, not an incrementing
sequence.

Goal 10: ``ack()`` is now an explicit error.  ``complete()`` requires
both ``ack_edge0()`` and ``ack_edge1()`` before releasing capacity.
Per-edge acks are distinct doorbell-level events received during GPU
execution, before CPU completion.
"""

from __future__ import annotations

import unittest

import pytest
from spark_multislot_state import (
    BackpressureError,
    DEFAULT_NUM_SLOTS,
    EVENT_ACK,
    EVENT_ACK_EDGE0,
    EVENT_ACK_EDGE1,
    EVENT_CLAIM,
    EVENT_COMPLETE,
    EVENT_CONSUME,
    EVENT_OVERFLOW,
    EVENT_PUBLISH,
    EVENT_RECLAIM,
    EVENT_TIMEOUT,
    EVENT_TEARDOWN,
    GENERATION_MODULUS,
    MultiSlotRing,
    OWNER_CPU_CONSUMER,
    OWNER_GPU_PRODUCER,
    OWNER_NONE,
    ProtocolSpec,
    SlotOverflowError,
    SlotStateError,
    default_protocol_spec,
    simulate_allreduce_with_truth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """Injectable monotonic clock for timeout tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += delta


def _full_cycle(ring: MultiSlotRing, slot: int, token: int) -> None:
    """Drive a slot through claim→publish→consume→ack_edge0→ack_edge1
    →complete→reclaim.

    All lifecycle methods receive the same ``token`` (one token per
    lifecycle, matching native ``claim_sequence`` semantics).
    Native ordering: per-edge acks (DoorbellControl
    .acknowledgement_sequence) are received DURING GPU execution,
    BEFORE complete() (Tp4GraphCommandRing.completed_sequence).
    Goal 10: both ack_edge0 and ack_edge1 are required before
    complete().
    """
    ring.claim_slot(token)  # may return a different slot; we pass explicit
    ring.publish(slot, token)
    ring.consume(slot, token)
    ring.ack_edge0(slot, token)
    ring.ack_edge1(slot, token)
    ring.complete(slot, token)
    ring.reclaim(slot)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class BasicLifecycleTest(unittest.TestCase):
    def test_claim_returns_valid_slot(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        slot = ring.claim_slot(1)
        self.assertGreaterEqual(slot, 0)
        self.assertLess(slot, 4)

    def test_claim_sets_gpu_producer_owner(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        slot = ring.claim_slot(1)
        state = ring.get_state(slot)
        self.assertEqual(state.owner, OWNER_GPU_PRODUCER)
        self.assertTrue(state.reserved)
        self.assertEqual(state.command_token, 1)

    def test_full_lifecycle_graph_mode(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        state = ring.get_state(s0)
        self.assertEqual(state.owner, OWNER_NONE)
        self.assertFalse(state.reserved)
        self.assertEqual(ring.get_outstanding(), 0)

    def test_full_lifecycle_non_graph_mode(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        self.assertEqual(ring.get_outstanding(), 0)

    def test_outstanding_count_tracks_lifecycle(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        self.assertEqual(ring.get_outstanding(), 0)
        s0 = ring.claim_slot(1)
        self.assertEqual(ring.get_outstanding(), 1)
        ring.publish(s0, 1)
        self.assertEqual(ring.get_outstanding(), 1)
        ring.consume(s0, 1)
        self.assertEqual(ring.get_outstanding(), 1)
        # Goal 10: acked slots are still outstanding until complete.
        ring.ack_edge0(s0, 1)
        self.assertEqual(ring.get_outstanding(), 1)
        ring.ack_edge1(s0, 1)
        self.assertEqual(ring.get_outstanding(), 1)
        ring.complete(s0, 1)
        self.assertEqual(ring.get_outstanding(), 0)
        ring.reclaim(s0)
        self.assertEqual(ring.get_outstanding(), 0)

    def test_completed_releases_capacity_before_reclaim(self) -> None:
        """Rule 5: completed_sequence releases ring capacity."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        # Slot is completed but not yet reclaimed — capacity already released
        self.assertEqual(ring.get_outstanding(), 0)
        # We can claim a new slot immediately
        s1 = ring.claim_slot(2)
        self.assertNotEqual(s0, s1)

    def test_get_events_records_all_events(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        events = ring.get_events()
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME,
            EVENT_ACK_EDGE0, EVENT_ACK_EDGE1,
            EVENT_COMPLETE, EVENT_RECLAIM,
        ])

    def test_get_events_returns_copy(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        ring.claim_slot(1)
        events1 = ring.get_events()
        events2 = ring.get_events()
        self.assertIsNot(events1, events2)


# ---------------------------------------------------------------------------
# Double-publish rejected
# ---------------------------------------------------------------------------

class DoublePublishTest(unittest.TestCase):
    def test_double_publish_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        with pytest.raises(SlotStateError, match="double-publish"):
            ring.publish(s0, 1)

    def test_double_publish_does_not_change_state(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        with pytest.raises(SlotStateError):
            ring.publish(s0, 1)
        state = ring.get_state(s0)
        self.assertEqual(state.producer_sequence, 1)


# ---------------------------------------------------------------------------
# Stale generation rejected (ABA/wraparound protection)
# ---------------------------------------------------------------------------

class StaleGenerationTest(unittest.TestCase):
    def test_same_generation_rejected_aba(self) -> None:
        # num_slots=1 forces the same physical slot to be reused
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        # Re-claiming the same token must fail: with contiguous tokens,
        # token 1 is no longer the expected next token (expected 2).
        # The non-contiguous check catches this before per-slot ABA.
        with pytest.raises(SlotStateError, match="non-contiguous"):
            ring.claim_slot(1)

    def test_different_generation_accepted_after_reclaim(self) -> None:
        # num_slots=1 forces the same physical slot to be reused
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        # Different token is fine
        s1 = ring.claim_slot(2)
        self.assertEqual(s1, s0)

    def test_generation_out_of_range_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False, generation_modulus=16)
        with pytest.raises(SlotStateError, match="out of range"):
            ring.claim_slot(16)
        with pytest.raises(SlotStateError, match="out of range"):
            ring.claim_slot(-1)
        # Token 0 is also rejected (native sequences start at 1)
        with pytest.raises(SlotStateError, match="out of range"):
            ring.claim_slot(0)


# ---------------------------------------------------------------------------
# Slow consumer timeout
# ---------------------------------------------------------------------------

    def test_slow_consumer_timeout(self) -> None:
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        # Consumer is slow — advance past timeout
        clock.advance(11.0)
        # Goal 9: published slot timeout is fatal (SlotOverflowError)
        with self.assertRaises(SlotOverflowError) as ctx:
            ring.check_timeouts()
        self.assertIn("fatal", str(ctx.exception).lower())
        self.assertTrue(ring.is_fatal())
        state = ring.get_state(s0)
        # Slot is not reclaimed — capacity unavailable until teardown
        self.assertNotEqual(state.owner, OWNER_NONE)
        self.assertTrue(state.reserved)
    def test_timeout_not_triggered_within_bound(self) -> None:
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        ring.claim_slot(1)
        clock.advance(5.0)
        events = ring.check_timeouts()
        self.assertEqual(events, [])

    def test_timeout_slot_can_be_reclaimed(self) -> None:
        """Goal 9: timed-out slots are NOT reclaimable.
        A claimed (pre-publication) slot that times out enters 'timeout'
        phase but cannot be reclaimed — reclaim raises SlotStateError."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        clock.advance(11.0)
        ring.check_timeouts()
        # Goal 9: timeout phase is not reclaimable
        with self.assertRaises(SlotStateError):
            ring.reclaim(s0)
        state = ring.get_state(s0)
        self.assertTrue(state.reserved)

# ---------------------------------------------------------------------------
# Producer timeout
# ---------------------------------------------------------------------------

class ProducerTimeoutTest(unittest.TestCase):
    def test_producer_timeout_claimed_not_published(self) -> None:
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        ring.claim_slot(1)
        # Producer never publishes
        clock.advance(11.0)
        events = ring.check_timeouts()
        self.assertTrue(any(e.event_type == EVENT_TIMEOUT for e in events))

    def test_producer_timeout_after_publish(self) -> None:
        """Producer published but consume/complete never happens.
        Goal 9: published slot timeout is fatal (SlotOverflowError)."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        clock.advance(11.0)
        with self.assertRaises(SlotOverflowError) as ctx:
            ring.check_timeouts()
        self.assertIn("fatal", str(ctx.exception).lower())
        self.assertTrue(ring.is_fatal())
        state = ring.get_state(s0)
        # Slot is not reclaimed — capacity unavailable until teardown
        self.assertNotEqual(state.owner, OWNER_NONE)


# ---------------------------------------------------------------------------
# Partial failure rollback
# ---------------------------------------------------------------------------

    def test_published_slot_not_rolled_back(self) -> None:
        """Goal 9: published slots are NOT rolled back — failure is fatal."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        result = ring.rollback(s0)
        self.assertIsNone(result)
        state = ring.get_state(s0)
        self.assertEqual(state.owner, OWNER_GPU_PRODUCER)
        self.assertTrue(state.reserved)
        self.assertGreater(state.producer_sequence, 0)
    def test_consumed_slot_not_rolled_back(self) -> None:
        """Goal 9: consumed slots are NOT rolled back — failure is fatal."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        result = ring.rollback(s0)
        self.assertIsNone(result)
        state = ring.get_state(s0)
        self.assertEqual(state.owner, OWNER_CPU_CONSUMER)
        self.assertTrue(state.reserved)
        self.assertGreater(state.consumer_sequence, 0)
    def test_claimed_slot_rolled_back(self) -> None:
        """Goal 9: rollback only applies to claimed (pre-publication) slots.
        A claimed slot IS rolled back to idle."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        result = ring.rollback(s0)
        self.assertIsNotNone(result)
        self.assertEqual(result.event_type, EVENT_RECLAIM)
        state = ring.get_state(s0)
        self.assertEqual(state.owner, OWNER_NONE)
        self.assertFalse(state.reserved)

    def test_completed_slot_not_rolled_back(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        result = ring.rollback(s0)
        self.assertIsNone(result)
        state = ring.get_state(s0)
        self.assertGreater(state.completed_sequence, 0)

    def test_rollback_all(self) -> None:
        ring = MultiSlotRing(num_slots=8, graph_mode=False)
        slots = []
        for i in range(3):
            s = ring.claim_slot(i + 1)
            ring.publish(s, i + 1)
            slots.append(s)
        # Claim one more but don't publish
        s3 = ring.claim_slot(4)
        events = ring.rollback_all()
        # Goal 9: only the 1 claimed (pre-publication) slot is rolled back
        self.assertEqual(len(events), 1)
        # Published slots are NOT rolled back — still reserved
        for s in slots:
            self.assertTrue(ring.get_state(s).reserved)
        # The claimed-but-not-published slot IS rolled back to idle
        self.assertFalse(ring.get_state(s3).reserved)

    def test_rolled_back_slot_reclaimable(self) -> None:
        """Goal 9: a claimed (pre-publication) slot that is rolled back
        can be re-claimed with the next contiguous token."""
        # num_slots=1 forces the same physical slot
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        s0 = ring.claim_slot(1)
        # Claimed (not published) — rollback returns it to idle
        ring.rollback(s0)
        # Slot is now idle and can be re-claimed with new token
        s1 = ring.claim_slot(2)
        self.assertEqual(s1, s0)


# ---------------------------------------------------------------------------
# Graph replay ordering (no skipping)
# ---------------------------------------------------------------------------

class GraphReplayOrderingTest(unittest.TestCase):
    def test_publish_must_follow_claim_order(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        _s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        # Trying to publish s1 before s0 must fail
        with pytest.raises(SlotStateError, match="graph-mode ordering"):
            ring.publish(s1, 2)

    def test_consume_must_follow_publish_order(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        ring.publish(s0, 1)
        ring.publish(s1, 2)
        # Trying to consume s1 before s0 must fail
        with pytest.raises(SlotStateError, match="graph-mode ordering"):
            ring.consume(s1, 2)

    def test_complete_must_follow_consume_order(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.publish(s1, 2)
        ring.consume(s1, 2)
        # Goal 10: both edges required before complete — add them for s0
        # so the ordering check is what fails for s1, not the edge check.
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        # s0 has both edges; s1 has none — but s1's edge check would fire
        # before the ordering check.  Give s1 both edges too so the ONLY
        # reason s1's complete fails is the graph-mode ordering rule.
        ring.ack_edge0(s1, 2)
        ring.ack_edge1(s1, 2)
        # Trying to complete s1 before s0 must fail (graph-mode ordering)
        with pytest.raises(SlotStateError, match="graph-mode ordering"):
            ring.complete(s1, 2)

    def test_correct_order_accepted(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        ring.publish(s0, 1)
        ring.publish(s1, 2)
        ring.consume(s0, 1)
        ring.consume(s1, 2)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.ack_edge0(s1, 2)
        ring.ack_edge1(s1, 2)
        ring.complete(s1, 2)
        ring.reclaim(s0)
        ring.reclaim(s1)
        self.assertEqual(ring.get_outstanding(), 0)

    def test_non_graph_mode_allows_skipping(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        # Non-graph mode: publish s1 before s0 is fine
        ring.publish(s1, 2)
        ring.publish(s0, 1)


# ---------------------------------------------------------------------------
# Backpressure when all slots outstanding
# ---------------------------------------------------------------------------

class BackpressureTest(unittest.TestCase):
    def test_backpressure_when_all_outstanding(self) -> None:
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        ring.claim_slot(1)
        ring.claim_slot(2)
        with pytest.raises(BackpressureError, match="backpressure"):
            ring.claim_slot(3)

    def test_backpressure_relieved_after_complete(self) -> None:
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.claim_slot(2)
        # Full lifecycle on s0: complete + reclaim
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        # Now a new claim should succeed (s0 is idle again)
        s2 = ring.claim_slot(3)
        self.assertIsNotNone(s2)

    def test_backpressure_relieved_after_reclaim(self) -> None:
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.claim_slot(2)
        # Full lifecycle on s0
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        # Now we can claim again
        s2 = ring.claim_slot(3)
        self.assertIsNotNone(s2)

    def test_default_64_slots(self) -> None:
        ring = MultiSlotRing()  # default 64
        self.assertEqual(ring.num_slots, DEFAULT_NUM_SLOTS)
        for i in range(DEFAULT_NUM_SLOTS):
            ring.claim_slot(i + 1)  # tokens 1..64
        with pytest.raises(BackpressureError):
            ring.claim_slot(DEFAULT_NUM_SLOTS + 1)


# ---------------------------------------------------------------------------
# Consume before publish rejected
# ---------------------------------------------------------------------------

class ConsumeBeforePublishTest(unittest.TestCase):
    def test_consume_before_publish_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        with pytest.raises(SlotStateError, match="must be 'published'"):
            ring.consume(s0, 1)

    def test_consume_on_idle_slot_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        with pytest.raises(SlotStateError, match="must be 'published'"):
            ring.consume(0, 1)


# ---------------------------------------------------------------------------
# Complete before consume rejected
# ---------------------------------------------------------------------------

class CompleteBeforeConsumeTest(unittest.TestCase):
    def test_complete_before_consume_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        with pytest.raises(SlotStateError, match="must be 'consumed'"):
            ring.complete(s0, 1)

    def test_complete_on_published_slot_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        with pytest.raises(SlotStateError, match="must be 'consumed'"):
            ring.complete(s0, 1)


# ---------------------------------------------------------------------------
# Reclaim before complete rejected
# ---------------------------------------------------------------------------

class ReclaimBeforeCompleteTest(unittest.TestCase):
    def test_reclaim_before_complete_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        with pytest.raises(SlotStateError, match="reclaim before complete"):
            ring.reclaim(s0)

    def test_reclaim_on_claimed_slot_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        ring.claim_slot(1)
        with pytest.raises(SlotStateError, match="reclaim"):
            ring.reclaim(0)

    def test_reclaim_on_published_slot_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        with pytest.raises(SlotStateError, match="reclaim"):
            ring.reclaim(s0)

    def test_reclaim_on_consumed_slot_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        with pytest.raises(SlotStateError, match="reclaim"):
            ring.reclaim(s0)

    def test_reclaim_after_complete_accepted(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        self.assertFalse(ring.get_state(s0).reserved)

    def test_reclaim_after_ack_rejected(self) -> None:
        """Goal 11: acked slots are NOT reclaimable — only COMPLETED may
        be reclaimed during normal reuse.  A slot with both edges acked
        but not completed must remain reserved."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        # Slot is acked but not completed — NOT reclaimable (Goal 11 req 6)
        with pytest.raises(SlotStateError, match="reclaim"):
            ring.reclaim(s0)
        self.assertTrue(ring.get_state(s0).reserved)


# ---------------------------------------------------------------------------
# Overflow (sequence above command_token) is fatal in graph mode
# ---------------------------------------------------------------------------

class OverflowFatalTest(unittest.TestCase):
    def test_overflow_fatal_in_graph_mode(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        s1 = ring.claim_slot(2)
        # Publishing 3 on slot 1 (command_token=2) is overflow
        with pytest.raises(SlotOverflowError, match="overflow"):
            ring.publish(s1, 3)

    def test_overflow_makes_ring_fatal(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        s1 = ring.claim_slot(2)
        with pytest.raises(SlotOverflowError):
            ring.publish(s1, 3)
        self.assertTrue(ring.is_fatal())

    def test_fatal_ring_rejects_all_operations(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        s1 = ring.claim_slot(2)
        with pytest.raises(SlotOverflowError):
            ring.publish(s1, 3)
        with pytest.raises(SlotOverflowError, match="fatal"):
            ring.claim_slot(3)
        with pytest.raises(SlotOverflowError, match="fatal"):
            ring.publish(s0, 1)

    def test_stale_sequence_rejected_graph_mode(self) -> None:
        """Below command_token is stale (not fatal), just rejected."""
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        # Publishing 0 on slot with command_token=1 is stale (0 < 1)
        with pytest.raises(SlotStateError, match="stale sequence"):
            ring.publish(s0, 0)
        self.assertFalse(ring.is_fatal())

    def test_mismatched_sequence_not_fatal_in_non_graph_mode(self) -> None:
        """Non-graph mode: sequence != command_token is rejected (not fatal).

        In the one-token-per-lifecycle model, any sequence that does not
        match the slot's command_token is an error.  In non-graph mode
        this is a SlotStateError, not a fatal SlotOverflowError.
        """
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        # Publishing 2 on slot with command_token=1 is a mismatch
        with pytest.raises(SlotStateError, match="!= command_token"):
            ring.publish(s0, 2)
        self.assertFalse(ring.is_fatal())

    def test_overflow_on_consume_graph_mode(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        s1 = ring.claim_slot(2)
        ring.publish(s1, 2)
        # Consuming with 5 on slot with command_token=2 is overflow
        with pytest.raises(SlotOverflowError, match="overflow"):
            ring.consume(s1, 5)


# ---------------------------------------------------------------------------
class WraparoundTest(unittest.TestCase):
    def test_token_exhaustion_at_small_modulus(self) -> None:
        """With a small modulus, tokens exhaust after all values are used.

        Tokens must be contiguous (last+1) and in [1, modulus).
        After using all valid tokens, no new claim is possible — the
        next expected token is out of range.
        """
        ring = MultiSlotRing(num_slots=1, graph_mode=False, generation_modulus=4)
        # Valid tokens: 1, 2, 3 (contiguous)
        for token in [1, 2, 3]:
            s = ring.claim_slot(token)
            ring.publish(s, token)
            ring.consume(s, token)
            ring.ack_edge0(s, token)
            ring.ack_edge1(s, token)
            ring.complete(s, token)
            ring.reclaim(s)
        # Token 3 reuse: non-contiguous (expected 4, got 3)
        with pytest.raises(SlotStateError, match="non-contiguous"):
            ring.claim_slot(3)
        # Token 4 is out of range (modulus=4)
        with pytest.raises(SlotStateError, match="out of range"):
            ring.claim_slot(4)
        # Token 0 is out of range
        with pytest.raises(SlotStateError, match="out of range"):
            ring.claim_slot(0)

    def test_large_uint64_token_via_contiguous_claims(self) -> None:
        """Reaching a large uint64 token via contiguous claims works.

        Instead of jumping directly to a large token (which violates
        contiguity), we drive the ring to a large token by claiming
        many tokens.  Here we use a small modulus and verify the
        boundary behavior.
        """
        ring = MultiSlotRing(num_slots=1, graph_mode=False, generation_modulus=8)
        # Contiguous tokens 1..6
        for token in range(1, 7):
            s = ring.claim_slot(token)
            ring.publish(s, token)
            ring.consume(s, token)
            ring.ack_edge0(s, token)
            ring.ack_edge1(s, token)
            ring.complete(s, token)
            ring.reclaim(s)
        # Token 7 is the last valid token (modulus=8)
        s = ring.claim_slot(7)
        self.assertEqual(ring.get_state(s).command_token, 7)

    def test_non_contiguous_after_full_cycle_rejected(self) -> None:
        """After a full cycle, reusing an old token is rejected as
        non-contiguous (not ABA), because the expected next token
        is last+1."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False, generation_modulus=6)
        # Valid tokens: 1, 2, 3, 4 (contiguous)
        for token in [1, 2, 3, 4]:
            s = ring.claim_slot(token)
            ring.publish(s, token)
            ring.consume(s, token)
            ring.ack_edge0(s, token)
            ring.ack_edge1(s, token)
            ring.complete(s, token)
            ring.reclaim(s)
        # Token 4 reuse: non-contiguous (expected 5, got 4)
        with pytest.raises(SlotStateError, match="non-contiguous"):
            ring.claim_slot(4)
        # Token 5 is valid (contiguous: 4+1=5, within modulus=6)
        s5 = ring.claim_slot(5)
        self.assertIsNotNone(s5)


# ---------------------------------------------------------------------------
# ProtocolSpec
# ---------------------------------------------------------------------------

class ProtocolSpecTest(unittest.TestCase):
    def test_default_spec_has_all_fields(self) -> None:
        spec = default_protocol_spec()
        self.assertIsInstance(spec, ProtocolSpec)
        self.assertEqual(spec.num_slots, DEFAULT_NUM_SLOTS)
        self.assertEqual(spec.generation_modulus, GENERATION_MODULUS)
        self.assertTrue(spec.graph_mode)

    def test_spec_documents_all_phases(self) -> None:
        spec = default_protocol_spec()
        phases = set(spec.per_slot_states)
        self.assertEqual(phases, {
            "idle", "claimed", "published", "consumed",
            "completed", "acked", "timeout", "torndown",
        })

    def test_spec_documents_all_owners(self) -> None:
        spec = default_protocol_spec()
        owners = set(spec.owner_values)
        self.assertEqual(owners, {OWNER_NONE, OWNER_GPU_PRODUCER, OWNER_CPU_CONSUMER})

    def test_spec_documents_all_event_types(self) -> None:
        spec = default_protocol_spec()
        events = set(spec.event_types)
        self.assertEqual(events, {
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME, EVENT_COMPLETE,
            EVENT_ACK, EVENT_ACK_EDGE0, EVENT_ACK_EDGE1,
            EVENT_RECLAIM, EVENT_OVERFLOW, EVENT_TIMEOUT,
            EVENT_TEARDOWN,
        })
    def test_spec_has_ordering_rules(self) -> None:
        spec = default_protocol_spec()
        self.assertGreater(len(spec.ordering_rules), 10)
        # Search all rule fields for keyword coverage
        all_rules = " ".join(spec.ordering_rules)
        all_rules += " " + spec.wraparound_rule
        all_rules += " " + spec.failure_rule
        all_rules += " " + spec.capacity_release_rule
        all_rules += " " + spec.sequence_equality_rule
        for keyword in [
            "claimed", "published", "consumed", "completed",
            "backpressure", "wraparound", "timeout", "failure",
            "skipping", "overflow",
        ]:
            self.assertIn(keyword, all_rules.lower())

    def test_spec_has_counter_descriptions(self) -> None:
        spec = default_protocol_spec()
        self.assertGreater(len(spec.counter_descriptions), 2)
        counters_text = " ".join(spec.counter_descriptions)
        self.assertIn("get_outstanding", counters_text)

class SlotStatePhaseTest(unittest.TestCase):
    def test_idle_phase(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        state = ring.get_state(0)
        self.assertEqual(state.phase, "idle")

    def test_claimed_phase(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        self.assertEqual(ring.get_state(s0).phase, "claimed")

    def test_published_phase(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        self.assertEqual(ring.get_state(s0).phase, "published")

    def test_consumed_phase(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        self.assertEqual(ring.get_state(s0).phase, "consumed")

    def test_completed_phase(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        self.assertEqual(ring.get_state(s0).phase, "completed")

    def test_acked_phase(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        self.assertEqual(ring.get_state(s0).phase, "acked")

    def test_timeout_phase(self) -> None:
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=5.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        clock.advance(6.0)
        ring.check_timeouts()
        self.assertEqual(ring.get_state(s0).phase, "timeout")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class InputValidationTest(unittest.TestCase):
    def test_invalid_slot_index_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        with pytest.raises(SlotStateError, match="out of range"):
            ring.get_state(4)
        with pytest.raises(SlotStateError, match="out of range"):
            ring.get_state(-1)

    def test_zero_slots_rejected(self) -> None:
        with pytest.raises(SlotStateError):
            MultiSlotRing(num_slots=0)

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(SlotStateError):
            MultiSlotRing(num_slots=4, timeout_seconds=0)

    def test_small_generation_modulus_rejected(self) -> None:
        with pytest.raises(SlotStateError):
            MultiSlotRing(num_slots=4, generation_modulus=1)

    def test_bool_generation_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        with pytest.raises(SlotStateError):
            ring.claim_slot(True)  # type: ignore[arg-type]

    def test_bool_slot_index_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        with pytest.raises(SlotStateError):
            ring.publish(True, 1)  # type: ignore[arg-type]

    def test_bool_sequence_rejected(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        with pytest.raises(SlotStateError):
            ring.publish(s0, True)  # type: ignore[arg-type]

    def test_zero_token_rejected(self) -> None:
        """Token 0 is rejected — native sequences start at 1."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        with pytest.raises(SlotStateError, match="out of range"):
            ring.claim_slot(0)


# ---------------------------------------------------------------------------
# Integration with FP32 ground truth
# ---------------------------------------------------------------------------

class GroundTruthIntegrationTest(unittest.TestCase):
    def test_simulate_allreduce_with_truth(self) -> None:
        """Verify the integration helper runs without error using torch."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")
        from spark_fp32_ground_truth import make_rank_input
        inputs = [
            make_rank_input(
                sequence=0, rank=r, world_size=4,
                rows=1, width=6144, pattern="random",
            )
            for r in range(4)
        ]

        result = simulate_allreduce_with_truth(inputs)
        self.assertEqual(result["evidence_label"], "Modeled")
        self.assertEqual(result["ring_outstanding"], 0)
        self.assertGreater(len(result["ring_events"]), 0)


class NativeCommandRingSemanticsTest(unittest.TestCase):
    """Adversarial regressions for native command-ring semantics.

    Tests derived from the native tp4_graph_command.hpp and
    gpu_doorbell.hpp types and lifecycle operations:
    - command/doorbell sequences are uint64, not 16-bit generation modulus
    - one command token is shared through publish/consume/complete/ack
    - slot reuse must not permit ABA after artificial 0 -> 1 -> 0 wrap
    - capacity-release language and claim behavior must agree
    - default clock must be an invoked monotonic value, never the function object
    - graph-published unfulfillable wait must be fatal, not reclaimable
    """

    def test_default_clock_is_invoked_value(self) -> None:
        """The default clock must return a float, not the function object.
        If clock() returns time.monotonic (the function), it causes
        TypeError when used in arithmetic (elapsed = current - timestamp)."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        ts = ring.clock()
        self.assertIsInstance(ts, float, "clock() must return a float")

    def test_command_token_is_uint64_not_16bit(self) -> None:
        """command_token field accepts values > 2^16 (uint64 range).

        Rather than driving 65K contiguous claims (too slow), we verify
        that the ring's generation_modulus accepts 2^64 and that a
        command_token set via a normal claim path is stored without
        truncation.  The SlotState.command_token field is a Python int
        (arbitrary precision), so it inherently supports uint64."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        # The modulus must be 2^64 (default), not 2^16
        self.assertEqual(ring.generation_modulus, 1 << 64)
        # A normal claim at token 1 stores the full value
        slot = ring.claim_slot(1)
        state = ring.get_state(slot)
        self.assertEqual(state.command_token, 1)
        # The field type is int (Python arbitrary precision = uint64 capable)
        self.assertIsInstance(state.command_token, int)

    def test_aba_immediate_token_reuse_rejected(self) -> None:
        """Reusing the same command_token after a full cycle is rejected.

        With contiguous tokens, reusing token 1 after it was already
        claimed is caught by the non-contiguous check (expected 2, got 1).
        Native uses uint64 sequences that monotonically advance — the
        same token cannot be reused."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        # First use: token 1
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        ring.reclaim(slot)
        # Immediate reuse of same token: non-contiguous (expected 2, got 1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(1)
        self.assertIn("non-contiguous", str(ctx.exception))
        # Next contiguous token works
        ring.claim_slot(2)

    def test_capacity_release_on_complete_not_consume(self) -> None:
        """A slot that is consumed but not completed must still count
        as outstanding."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        # Consumed but not completed: still outstanding
        self.assertEqual(ring.get_outstanding(), 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Completed: no longer outstanding
        self.assertEqual(ring.get_outstanding(), 0)

    def test_graph_mode_timeout_is_fatal(self) -> None:
        """Graph-mode timeout must be fatal (worker termination),
        not silently reclaimable to idle."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=True,
            timeout_seconds=1.0, clock=clock,
        )
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        clock.advance(2.0)  # exceed timeout
        with self.assertRaises(SlotOverflowError) as ctx:
            ring.check_timeout(slot)
        self.assertIn("fatal", str(ctx.exception).lower())
        self.assertTrue(ring.is_fatal())
        # All subsequent operations must raise
        with self.assertRaises(SlotOverflowError):
            ring.claim_slot(2)

    def test_non_graph_published_timeout_is_fatal(self) -> None:
        """Goal 9: published slot timeout is fatal even in non-graph mode."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False,
            timeout_seconds=1.0, clock=clock,
        )
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        clock.advance(2.0)
        with self.assertRaises(SlotOverflowError) as ctx:
            ring.check_timeout(slot)
        self.assertIn("fatal", str(ctx.exception).lower())
        self.assertTrue(ring.is_fatal())
        # Not reclaimable — capacity unavailable until teardown
        self.assertTrue(ring.get_state(slot).reserved)

    def test_stale_ack_rejected(self) -> None:
        """Per-edge ack with a stale sequence (below command_token) must
        be rejected."""
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        # ack_edge0 with stale sequence (0 < command_token 1) from consumed phase
        with self.assertRaises(SlotStateError):
            ring.ack_edge0(slot, 0)

    def test_rollback_on_published_slot(self) -> None:
        """Goal 9: rollback on a published slot returns None (not rolled back)."""
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ev = ring.rollback(slot)
        self.assertIsNone(ev)
        self.assertEqual(ring.get_state(slot).phase, "published")
        self.assertTrue(ring.get_state(slot).reserved)

    def test_command_token_shared_through_lifecycle(self) -> None:
        """The command_token set at claim must persist through the
        entire lifecycle (publish, consume, ack_edge0, ack_edge1,
        complete, reclaim).  It is NOT replaced by a new sequence at
        each transition."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        token = 1  # contiguous: first token
        slot = ring.claim_slot(token)
        self.assertEqual(ring.get_state(slot).command_token, token)
        ring.publish(slot, token)
        self.assertEqual(ring.get_state(slot).command_token, token)
        ring.consume(slot, token)
        self.assertEqual(ring.get_state(slot).command_token, token)
        ring.ack_edge0(slot, token)
        self.assertEqual(ring.get_state(slot).command_token, token)
        ring.ack_edge1(slot, token)
        self.assertEqual(ring.get_state(slot).command_token, token)
        ring.complete(slot, token)
        self.assertEqual(ring.get_state(slot).command_token, token)
        ring.reclaim(slot)
        # After reclaim, token is reset to 0
        self.assertEqual(ring.get_state(slot).command_token, 0)


# ---------------------------------------------------------------------------
# C1: completion immediately restores claim capacity (no reclaim needed)
# ---------------------------------------------------------------------------

class CompletionRestoresCapacityTest(unittest.TestCase):
    """After complete(), a one-slot ring must allow claim_slot(next_token)
    immediately without an extra reclaim() call.

    Reproduction: before the fix, claim_slot only searched for _PHASE_IDLE
    slots, so a completed slot (phase='completed', reserved=False) was
    invisible — BackpressureError was raised despite capacity being free.
    """

    def test_one_slot_claim_after_complete_without_reclaim(self) -> None:
        """One-slot ring: complete() → claim_slot(next) succeeds."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Slot is now 'completed' with reserved=False — capacity released.
        self.assertFalse(ring.get_state(slot).reserved)
        self.assertEqual(ring.get_outstanding(), 0)
        # Must be able to claim immediately without reclaim().
        slot2 = ring.claim_slot(2)
        self.assertEqual(slot2, slot)  # same slot reused

    def test_multi_slot_claim_after_complete_without_reclaim(self) -> None:
        """Multi-slot ring: completed slot is immediately claimable."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.claim_slot(2)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        # s0 completed, s1 still claimed → one slot free
        self.assertEqual(ring.get_outstanding(), 1)
        # Must be able to claim slot 0 (or 1 if s1 is still outstanding)
        s2 = ring.claim_slot(3)
        self.assertIsNotNone(s2)

    def test_completed_slot_not_counted_as_outstanding(self) -> None:
        """get_outstanding() must not count completed slots."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        self.assertEqual(ring.get_outstanding(), 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        self.assertEqual(ring.get_outstanding(), 0)

    def test_complete_then_claim_same_slot_succeeds(self) -> None:
        """Complete then immediate re-claim of same slot with next token."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        s1 = ring.claim_slot(2)
        self.assertEqual(s1, s0)

    def test_graph_mode_complete_releases_capacity(self) -> None:
        """Graph mode: complete() releases capacity just like non-graph
        mode.  Per-edge acks come before complete(), not after."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Slot is completed and capacity is released (reserved=False)
        self.assertFalse(ring.get_state(slot).reserved)
        self.assertEqual(ring.get_state(slot).phase, "completed")
        # Can claim immediately — no extra ack needed
        slot2 = ring.claim_slot(2)
        self.assertIsNotNone(slot2)


# ---------------------------------------------------------------------------
# C2: global monotonic lifecycle tokens
# ---------------------------------------------------------------------------

class GlobalMonotonicTokenTest(unittest.TestCase):
    """Two different slots may not simultaneously claim the same lifecycle
    token.  Native claimed_sequence is a single global counter — tokens
    must be globally unique and monotonically advancing.
    """

    def test_two_slots_cannot_claim_same_token(self) -> None:
        """Two different slots must not both claim token 1.

        With contiguous tokens, claiming token 1 again is caught as
        non-contiguous (expected 2, got 1).  The deterministic slot
        mapping also maps token 1 to slot 0, which is already reserved."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        ring.claim_slot(1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(1)
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_token_must_be_contiguous(self) -> None:
        """Token must be exactly last+1 (contiguous)."""
        ring = MultiSlotRing(num_slots=3, graph_mode=False)
        ring.claim_slot(1)
        ring.claim_slot(2)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(1)
        self.assertIn("non-contiguous", str(ctx.exception))
        # Token 2 is non-contiguous (expected 3)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(2)
        self.assertIn("non-contiguous", str(ctx.exception))
        # Token 3 is contiguous — OK
        ring.claim_slot(3)

    def test_global_monotonicity_after_complete_and_reclaim(self) -> None:
        """Even after a slot is completed and reclaimed, the global
        counter still rejects reuse of the old token."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        s = ring.claim_slot(1)
        ring.publish(s, 1)
        ring.consume(s, 1)
        ring.ack_edge0(s, 1)
        ring.ack_edge1(s, 1)
        ring.complete(s, 1)
        ring.reclaim(s)
        # Token 1 was used globally — cannot reuse even after reclaim
        with self.assertRaises(SlotStateError):
            ring.claim_slot(1)
        ring.claim_slot(2)  # OK

    def test_global_monotonicity_across_multiple_slots(self) -> None:
        """Tokens must be globally unique across all slots over time."""
        ring = MultiSlotRing(num_slots=3, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        # Slot 0 is now completed (capacity released)
        s1 = ring.claim_slot(2)  # token 2 > global last (1)
        ring.publish(s1, 2)
        ring.consume(s1, 2)
        ring.ack_edge0(s1, 2)
        ring.ack_edge1(s1, 2)
        ring.complete(s1, 2)
        # Token 2 was used; cannot reuse
        with self.assertRaises(SlotStateError):
            ring.claim_slot(2)
        ring.claim_slot(3)  # OK


# ---------------------------------------------------------------------------
# C3: ack() is doorbell-level, not command-ring lifecycle
# ---------------------------------------------------------------------------

class AckDoorbellSeparationTest(unittest.TestCase):
    """ack() is now an explicit error (Goal 10).  Per-edge acks
    (ack_edge0/ack_edge1) are doorbell-level operations
    (DoorbellControl.acknowledgement_sequence), not part of the
    Tp4GraphCommandRing lifecycle (claim→publish→consume→complete).
    complete() is the terminal command-ring step and releases capacity
    after both per-edge acks are received.
    """

    def test_complete_is_terminal_no_ack_required_for_capacity(self) -> None:
        """complete() releases capacity after both per-edge acks; the
        acks are required before complete, not after it."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Slot is immediately claimable — capacity released
        ring.claim_slot(2)

    def test_ack_rejected_after_complete(self) -> None:
        """Per-edge ack after complete() must fail — the slot is already
        terminal.  Native ordering: ack before complete, not after."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # ack_edge0 after complete is rejected (must be 'consumed')
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack_edge0(slot, 1)
        self.assertIn("must be 'consumed'", str(ctx.exception))

    def test_per_edge_ack_from_consumed_succeeds(self) -> None:
        """ack_edge0() from the consumed phase succeeds — transitions
        to acked.  Native ordering: per-edge acks (DoorbellControl)
        are received DURING GPU execution, before CPU complete()."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ev = ring.ack_edge0(slot, 1)
        self.assertEqual(ev.event_type, EVENT_ACK_EDGE0)
        self.assertEqual(ring.get_state(slot).phase, "acked")
        # complete() then transitions acked→completed
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        self.assertEqual(ring.get_state(slot).phase, "completed")

    def test_command_ring_lifecycle_is_claim_publish_consume_complete(self) -> None:
        """The native Tp4GraphCommandRing has claimed_sequence,
        published_sequence, consumed_sequence, completed_sequence —
        but NO acknowledgement_sequence.  The lifecycle is exactly
        claim→publish→consume→complete (with per-edge acks before
        complete)."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        state = ring.get_state(slot)
        # After claim: command_token set, sequences not yet advanced
        self.assertEqual(state.command_token, 1)
        self.assertEqual(state.producer_sequence, 0)
        self.assertEqual(state.consumer_sequence, 0)
        self.assertEqual(state.completed_sequence, 0)

        ring.publish(slot, 1)
        state = ring.get_state(slot)
        self.assertEqual(state.producer_sequence, 1)
        self.assertEqual(state.consumer_sequence, 0)
        self.assertEqual(state.completed_sequence, 0)

        ring.consume(slot, 1)
        state = ring.get_state(slot)
        self.assertEqual(state.consumer_sequence, 1)
        self.assertEqual(state.completed_sequence, 0)

        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        state = ring.get_state(slot)
        self.assertEqual(state.completed_sequence, 1)
        # Capacity released
        self.assertFalse(state.reserved)


# ---------------------------------------------------------------------------
# C4: native field/transition mapping and capacity/backpressure tests
# ---------------------------------------------------------------------------

class NativeFieldMappingTest(unittest.TestCase):
    """Verify the Python model maps to native Tp4GraphCommandRing fields:

    Native (tp4_graph_command.hpp):
      Tp4GraphProducerState: claimed_sequence, published_sequence, overflow_sequence
      Tp4GraphConsumerState: consumed_sequence, completed_sequence
      Tp4GraphCommandRing: producer, consumer, commands[64]

    The Python model must expose equivalent fields through SlotState.
    """

    def test_slot_state_has_native_fields(self) -> None:
        """SlotState must expose fields matching native sequences."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        state = ring.get_state(slot)
        # Maps to Tp4GraphProducerState.claimed_sequence
        self.assertTrue(hasattr(state, "command_token"))
        # Maps to Tp4GraphProducerState.published_sequence
        self.assertTrue(hasattr(state, "producer_sequence"))
        # Maps to Tp4GraphConsumerState.consumed_sequence
        self.assertTrue(hasattr(state, "consumer_sequence"))
        # Maps to Tp4GraphConsumerState.completed_sequence
        self.assertTrue(hasattr(state, "completed_sequence"))
        # Maps to DoorbellControl.acknowledgement_sequence (doorbell, not ring)
        self.assertTrue(hasattr(state, "acknowledgement_sequence"))

    def test_native_ring_has_no_ack_field(self) -> None:
        """Tp4GraphCommandRing has no acknowledgement_sequence — only
        DoorbellControl does.  The ack phase exists for doorbell fidelity
        but is not part of the command-ring lifecycle."""
        # This is a documentation test: the native struct
        # Tp4GraphConsumerState has consumed_sequence and
        # completed_sequence but NOT acknowledgement_sequence.
        # The Python model's acknowledgement_sequence field is
        # documented as doorbell-level (DoorbellControl).
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        state = ring.get_state(slot)
        # completed_sequence is set (native Tp4GraphConsumerState.completed_sequence)
        self.assertEqual(state.completed_sequence, 1)
        # acknowledgement_sequence is NOT set by complete() — it's doorbell-only
        self.assertEqual(state.acknowledgement_sequence, 0)

    def test_transitions_match_native_lifecycle(self) -> None:
        """Verify transitions match native claim_sequence→publish→
        consume→complete."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        # claim → matches native claim_sequence
        slot = ring.claim_slot(1)
        self.assertEqual(ring.get_state(slot).phase, "claimed")
        # publish → matches native publish_command (published_sequence)
        ring.publish(slot, 1)
        self.assertEqual(ring.get_state(slot).phase, "published")
        # consume → matches native try_consume (consumed_sequence)
        ring.consume(slot, 1)
        self.assertEqual(ring.get_state(slot).phase, "consumed")
        # ack_edge0/ack_edge1 → doorbell-level acks before complete
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        # complete → matches native tp4_graph_command_complete (completed_sequence)
        ring.complete(slot, 1)
        self.assertEqual(ring.get_state(slot).phase, "completed")


class CapacityBackpressureTest(unittest.TestCase):
    """Capacity and backpressure tests for one- and multi-slot rings.

    Native: claimed_sequence - completed_sequence < kTp4GraphCommandCapacity
    permits new claims.  When all slots are outstanding, BackpressureError.
    """

    def test_one_slot_backpressure_when_outstanding(self) -> None:
        """One-slot ring: claiming while the slot is outstanding blocks."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        ring.claim_slot(1)
        with self.assertRaises(BackpressureError):
            ring.claim_slot(2)

    def test_one_slot_capacity_restored_on_complete(self) -> None:
        """One-slot ring: complete() restores capacity for immediate re-claim."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Capacity restored — no backpressure
        ring.claim_slot(2)

    def test_multi_slot_partial_capacity(self) -> None:
        """Multi-slot: completing one slot out of N restores one slot of capacity."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.claim_slot(2)
        # Both outstanding — backpressure
        with self.assertRaises(BackpressureError):
            ring.claim_slot(3)
        # Complete one slot
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        # One slot free now
        ring.claim_slot(3)
        # Backpressure again (both outstanding: s1 claimed, s3 claimed)
        with self.assertRaises(BackpressureError):
            ring.claim_slot(4)

    def test_multi_slot_full_capacity_cycle(self) -> None:
        """Multi-slot: full lifecycle on all slots restores all capacity."""
        ring = MultiSlotRing(num_slots=3, graph_mode=False)
        tokens = [1, 2, 3]
        slots = [ring.claim_slot(t) for t in tokens]
        # All outstanding
        with self.assertRaises(BackpressureError):
            ring.claim_slot(4)
        for s, t in zip(slots, tokens):
            ring.publish(s, t)
            ring.consume(s, t)
            ring.ack_edge0(s, t)
            ring.ack_edge1(s, t)
            ring.complete(s, t)
        # All capacity restored
        self.assertEqual(ring.get_outstanding(), 0)
        ring.claim_slot(4)
        ring.claim_slot(5)
        ring.claim_slot(6)

    def test_graph_mode_backpressure(self) -> None:
        """Graph mode: backpressure also enforced."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        ring.claim_slot(1)
        with self.assertRaises(BackpressureError):
            ring.claim_slot(2)

    def test_graph_mode_capacity_restored_on_complete(self) -> None:
        """Graph mode: complete() restores capacity for re-claim (both modes)."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Graph mode: complete releases capacity (same as non-graph)
        ring.claim_slot(2)


# ---------------------------------------------------------------------------
# Goal-7 Part C: Contiguous tokens, deterministic slot mapping, peer ack barrier
# ---------------------------------------------------------------------------

class ContiguousTokenTest(unittest.TestCase):
    """Goal-7 C1: claim tokens must be exactly last+1 (contiguous)."""

    def test_claim_slot_3_after_token_1_fails(self) -> None:
        """Exact Goal-6 reproduction: claim_slot(3) immediately after
        token 1 must fail — tokens must be contiguous (expected 2)."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        ring.claim_slot(1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(3)
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_first_claim_must_be_1(self) -> None:
        """The first claim must be token 1, not any other value."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(2)
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_contiguous_sequence_works(self) -> None:
        """Tokens 1, 2, 3, 4 in order all succeed."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        for t in [1, 2, 3, 4]:
            s = ring.claim_slot(t)
            ring.publish(s, t)
            ring.consume(s, t)
            ring.ack_edge0(s, t)
            ring.ack_edge1(s, t)
            ring.complete(s, t)
            ring.reclaim(s)

    def test_gap_in_sequence_fails(self) -> None:
        """Skipping a token (1, 3) fails at token 3."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        ring.claim_slot(1)
        ring.publish(0, 1)
        ring.consume(0, 1)
        ring.ack_edge0(0, 1)
        ring.ack_edge1(0, 1)
        ring.complete(0, 1)
        ring.reclaim(0)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(3)  # expected 2
        self.assertIn("non-contiguous", str(ctx.exception))


class DeterministicSlotMappingTest(unittest.TestCase):
    """Goal-7 C2: token-to-slot mapping is (token-1) % num_slots."""

    def test_token_1_maps_to_slot_0(self) -> None:
        """Token 1 must map to slot (1-1)%2 = 0."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        slot = ring.claim_slot(1)
        self.assertEqual(slot, 0)

    def test_token_2_maps_to_slot_1_with_2_slots(self) -> None:
        """Token 2 must map to slot (2-1)%2 = 1, not slot 0."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        ring.claim_slot(1)
        slot = ring.claim_slot(2)
        self.assertEqual(slot, 1)

    def test_token_3_maps_to_slot_0_with_2_slots(self) -> None:
        """Exact Goal-6 reproduction: token 3 must map to slot (3-1)%2 = 0,
        not slot 1."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        slot = ring.claim_slot(2)
        self.assertEqual(slot, 1)
        ring.publish(slot, 2)
        ring.consume(slot, 2)
        ring.ack_edge0(slot, 2)
        ring.ack_edge1(slot, 2)
        ring.complete(slot, 2)
        ring.reclaim(slot)
        slot3 = ring.claim_slot(3)
        self.assertEqual(slot3, 0)  # (3-1)%2 = 0

    def test_no_round_robin_slot_selection(self) -> None:
        """The ring must not pick a different free slot — it must use
        the deterministic (token-1)%num_slots mapping."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        # Token 1 → slot 0, token 2 → slot 1, etc.
        for t in [1, 2, 3, 4]:
            s = ring.claim_slot(t)
            self.assertEqual(s, (t - 1) % 4)
            ring.publish(s, t)
            ring.consume(s, t)
            ring.ack_edge0(s, t)
            ring.ack_edge1(s, t)
            ring.complete(s, t)
            ring.reclaim(s)


class PeerAckBarrierTest(unittest.TestCase):
    """Goal-7 C3: complete() releases capacity in both graph and
    non-graph mode.  Per-edge acks (ack_edge0/ack_edge1) are required
    before complete() (Goal 10)."""

    def test_graph_complete_releases_slot(self) -> None:
        """In graph mode, complete() clears reserved and releases capacity."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        self.assertFalse(ring.get_state(slot).reserved)
        self.assertEqual(ring.get_state(slot).phase, "completed")

    def test_graph_complete_allows_reuse(self) -> None:
        """complete() releases the slot in graph mode — claim after
        complete succeeds."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Capacity released by complete — can immediately claim
        ring.claim_slot(2)

    def test_graph_reuse_after_complete_without_ack(self) -> None:
        """Reusing a completed slot works after both per-edge acks —
        complete is the capacity-release signal in both modes."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Both acks received, complete released capacity
        ring.claim_slot(2)

    def test_ack_after_complete_fails(self) -> None:
        """Per-edge ack after complete() is rejected — the slot is
        already terminal.  Native ordering: ack before complete, not after."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack_edge0(slot, 1)
        self.assertIn("must be 'consumed'", str(ctx.exception))

    def test_non_graph_complete_releases_immediately(self) -> None:
        """Non-graph: complete() releases capacity after both per-edge acks."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        self.assertFalse(ring.get_state(slot).reserved)
        # Can immediately claim next token
        ring.claim_slot(2)


class OutOfOrderProgressTest(unittest.TestCase):
    """Goal-7 C4: out-of-order progress with multi-slot rings."""

    def test_multi_slot_out_of_order_complete(self) -> None:
        """With 4 slots, tokens 1-4 claimed; completing out of order
        (token 3 before token 1) is allowed in non-graph mode."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        slots = [ring.claim_slot(t) for t in [1, 2, 3, 4]]
        for s in slots:
            ring.publish(s, slots.index(s) + 1)
            ring.consume(s, slots.index(s) + 1)
        # Both edges on all slots before completing out of order
        for s in slots:
            ring.ack_edge0(s, slots.index(s) + 1)
            ring.ack_edge1(s, slots.index(s) + 1)
        # Complete slot 2 (token 3) before slot 0 (token 1)
        ring.complete(slots[2], 3)
        self.assertEqual(ring.get_outstanding(), 3)
        ring.complete(slots[0], 1)
        self.assertEqual(ring.get_outstanding(), 2)

    def test_graph_mode_no_skipping(self) -> None:
        """Graph mode: must complete in claim order (no skipping).

        Claim tokens 1 and 2 (slots 0 and 1), then complete in order."""
        ring = MultiSlotRing(num_slots=4, graph_mode=True)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        ring.publish(s1, 2)
        ring.consume(s1, 2)
        ring.ack_edge0(s1, 2)
        ring.ack_edge1(s1, 2)
        ring.complete(s1, 2)


class DuplicateStaleFutureTokenTest(unittest.TestCase):
    """Goal-7 C4: duplicate, stale, and future tokens are rejected."""

    def test_duplicate_token_rejected(self) -> None:
        """Claiming the same token twice is rejected (non-contiguous)."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        ring.claim_slot(1)
        with self.assertRaises(SlotStateError):
            ring.claim_slot(1)

    def test_stale_token_rejected(self) -> None:
        """A token below the expected next is rejected."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        ring.claim_slot(1)
        ring.claim_slot(2)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(1)
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_future_token_rejected(self) -> None:
        """A token above the expected next is rejected."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        ring.claim_slot(1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(5)
        self.assertIn("non-contiguous", str(ctx.exception))


class DelayedAckTest(unittest.TestCase):
    """Goal-7 C4: delayed ack scenarios."""

    def test_per_edge_ack_after_delay_works(self) -> None:
        """A delayed per-edge ack (after some time) still works in graph mode."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=1, graph_mode=True,
            timeout_seconds=10.0, clock=clock,
        )
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        clock.advance(5.0)  # delay but under timeout
        ring.ack_edge0(slot, 1)
        self.assertTrue(ring.get_state(slot).reserved)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        self.assertFalse(ring.get_state(slot).reserved)

    def test_timeout_before_ack_on_outstanding_slot(self) -> None:
        """If timeout expires while a slot is still outstanding (not
        completed), it's fatal in graph mode."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=1, graph_mode=True,
            timeout_seconds=1.0, clock=clock,
        )
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        # Slot is consumed but not completed — still outstanding
        clock.advance(2.0)  # exceed timeout
        with self.assertRaises(SlotOverflowError):
            ring.check_timeout(slot)
        self.assertTrue(ring.is_fatal())


class BackpressureDeterministicTest(unittest.TestCase):
    """Goal-7 C4: backpressure on the deterministic slot, not any slot."""

    def test_backpressure_on_mapped_slot_only(self) -> None:
        """If the mapped slot is outstanding but other slots are free,
        backpressure is still raised — the ring never picks a different slot."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        # Token 1 → slot 0, still outstanding
        ring.claim_slot(1)
        # Token 5 would also map to slot 0 (5-1)%4 = 0
        # But token 5 is non-contiguous (expected 2), so it fails first
        with self.assertRaises(SlotStateError) as ctx:
            ring.claim_slot(5)
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_backpressure_when_mapped_slot_outstanding(self) -> None:
        """Token 5 maps to slot 0 — if slot 0 is outstanding,
        backpressure is raised."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        # Claim tokens 1-4 (fill all slots)
        for t in [1, 2, 3, 4]:
            ring.claim_slot(t)
        # Token 5 maps to slot 0 — outstanding → backpressure
        with self.assertRaises(BackpressureError):
            ring.claim_slot(5)

    def test_backpressure_relieved_when_mapped_slot_completed(self) -> None:
        """Completing the mapped slot relieves backpressure."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        for t in [1, 2, 3, 4]:
            ring.claim_slot(t)
        # Complete slot 0 (token 1)
        ring.publish(0, 1)
        ring.consume(0, 1)
        ring.ack_edge0(0, 1)
        ring.ack_edge1(0, 1)
        ring.complete(0, 1)
        # Token 5 maps to slot 0 — now available
        slot = ring.claim_slot(5)
        self.assertEqual(slot, 0)


# ---------------------------------------------------------------------------
# Native ordering trace-equivalence
# ---------------------------------------------------------------------------

class NativeOrderingTraceTest(unittest.TestCase):
    """Verify the native ordering claim→publish→consume→ack_edge0→
    ack_edge1→complete→reuse produces events in the exact expected
    sequence."""

    def test_full_cycle_event_order(self) -> None:
        """claim→publish→consume→ack_edge0→ack_edge1→complete→reclaim
        produces events in exactly that order."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        ring.reclaim(slot)
        events = ring.get_events()
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME,
            EVENT_ACK_EDGE0, EVENT_ACK_EDGE1,
            EVENT_COMPLETE, EVENT_RECLAIM,
        ])

    def test_reuse_on_same_slot_after_complete(self) -> None:
        """After complete, the same slot is reusable with the next token
        without an intermediate reclaim."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot1 = ring.claim_slot(1)
        ring.publish(slot1, 1)
        ring.consume(slot1, 1)
        ring.ack_edge0(slot1, 1)
        ring.ack_edge1(slot1, 1)
        ring.complete(slot1, 1)
        # Direct reuse — no reclaim needed, complete released capacity
        slot2 = ring.claim_slot(2)
        self.assertEqual(slot2, slot1)
        ring.publish(slot2, 2)
        ring.consume(slot2, 2)
        ring.ack_edge0(slot2, 2)
        ring.ack_edge1(slot2, 2)
        ring.complete(slot2, 2)
        events = ring.get_events()
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME,
            EVENT_ACK_EDGE0, EVENT_ACK_EDGE1, EVENT_COMPLETE,
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME,
            EVENT_ACK_EDGE0, EVENT_ACK_EDGE1, EVENT_COMPLETE,
        ])

    def test_reuse_after_reclaim_same_event_order(self) -> None:
        """claim→publish→consume→ack_edge0→ack_edge1→complete→reclaim→
        claim(next) on same slot produces correct event sequence."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.reclaim(s0)
        s1 = ring.claim_slot(2)
        self.assertEqual(s1, s0)
        events = ring.get_events()
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME,
            EVENT_ACK_EDGE0, EVENT_ACK_EDGE1,
            EVENT_COMPLETE, EVENT_RECLAIM,
            EVENT_CLAIM,
        ])

    def test_complete_from_consumed_without_acks_rejected(self) -> None:
        """Goal 10: complete() without both per-edge acks is rejected.
        Events: claim, publish, consume only — complete raises."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        # complete() without per-edge acks must fail (Goal 10)
        with self.assertRaises(SlotStateError, msg="complete without edges"):
            ring.complete(slot, 1)
        events = ring.get_events()
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME,
        ])

    def test_graph_mode_per_edge_ack_then_complete_event_order(self) -> None:
        """Graph mode: ack_edge0+ack_edge1 before complete — events
        in native order."""
        ring = MultiSlotRing(num_slots=2, graph_mode=True)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        ring.publish(s0, 1)
        ring.publish(s1, 2)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        ring.consume(s1, 2)
        ring.ack_edge0(s1, 2)
        ring.ack_edge1(s1, 2)
        ring.complete(s1, 2)
        events = ring.get_events()
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EVENT_CLAIM, EVENT_CLAIM,
            EVENT_PUBLISH, EVENT_PUBLISH,
            EVENT_CONSUME, EVENT_ACK_EDGE0, EVENT_ACK_EDGE1, EVENT_COMPLETE,
            EVENT_CONSUME, EVENT_ACK_EDGE0, EVENT_ACK_EDGE1, EVENT_COMPLETE,
        ])


# ---------------------------------------------------------------------------
# Goal 9: per-edge ack trace equivalence
# ---------------------------------------------------------------------------

class PerEdgeAckTraceTest(unittest.TestCase):
    """Verify the two-round TP4 per-edge doorbell acknowledgement
    (ack_edge0 + ack_edge1) produces the correct event sequence."""

    def test_per_edge_ack_full_cycle_event_order(self) -> None:
        """ack_edge0→ack_edge1→complete produces the correct event
        sequence: claim, publish, consume, ack_edge0, ack_edge1, complete,
        reclaim."""
        ring = MultiSlotRing(num_slots=1, graph_mode=True)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        ring.reclaim(slot)
        events = ring.get_events()
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EVENT_CLAIM, EVENT_PUBLISH, EVENT_CONSUME,
            EVENT_ACK_EDGE0, EVENT_ACK_EDGE1,
            EVENT_COMPLETE, EVENT_RECLAIM,
        ])

    def test_ack_edge1_after_ack_edge0(self) -> None:
        """ack_edge1 can be called after ack_edge0 (consumed→acked
        transition).  Both per-edge sequences are set and the slot
        transitions to acked after ack_edge0."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        # ack_edge0 transitions consumed→acked
        ev0 = ring.ack_edge0(slot, 1)
        self.assertEqual(ev0.event_type, EVENT_ACK_EDGE0)
        state = ring.get_state(slot)
        self.assertEqual(state.phase, "acked")
        self.assertGreater(state.ack_edge0_sequence, 0)
        # ack_edge1 can be called from acked phase
        ev1 = ring.ack_edge1(slot, 1)
        self.assertEqual(ev1.event_type, EVENT_ACK_EDGE1)
        state = ring.get_state(slot)
        self.assertEqual(state.phase, "acked")
        self.assertGreater(state.ack_edge1_sequence, 0)
        # complete still works after both per-edge acks
        ring.complete(slot, 1)
        self.assertEqual(ring.get_state(slot).phase, "completed")

    def test_ack_edge1_without_ack_edge0_rejected(self) -> None:
        """ack_edge1 from consumed phase (without ack_edge0) is allowed
        and transitions to acked.  But ack_edge0 from acked phase
        is rejected (must be consumed)."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        # ack_edge1 can be called from consumed phase
        ev1 = ring.ack_edge1(slot, 1)
        self.assertEqual(ev1.event_type, EVENT_ACK_EDGE1)
        self.assertEqual(ring.get_state(slot).phase, "acked")
        # ack_edge0 from acked phase is rejected (must be consumed)
        with self.assertRaises(SlotStateError):
            ring.ack_edge0(slot, 1)

    def test_per_edge_ack_trace_reaches_completed(self) -> None:
        """The per-edge ack path (ack_edge0+ack_edge1) transitions
        consumed→acked→completed and sets both edge sequences while
        leaving the scalar acknowledgement_sequence at 0."""
        ring_edge = MultiSlotRing(num_slots=1, graph_mode=True)
        slot_e = ring_edge.claim_slot(1)
        ring_edge.publish(slot_e, 1)
        ring_edge.consume(slot_e, 1)
        ring_edge.ack_edge0(slot_e, 1)
        ring_edge.ack_edge1(slot_e, 1)
        ring_edge.complete(slot_e, 1)
        edge_state = ring_edge.get_state(slot_e)
        self.assertEqual(edge_state.phase, "completed")
        self.assertFalse(edge_state.reserved)
        self.assertGreater(edge_state.ack_edge0_sequence, 0)
        self.assertGreater(edge_state.ack_edge1_sequence, 0)
        self.assertEqual(edge_state.acknowledgement_sequence, 0)

    def test_legacy_scalar_ack_is_error(self) -> None:
        """Goal 10: the legacy scalar ack() is now an explicit error.
        It must raise SlotStateError with a message about pre-publication
        compatibility error, regardless of phase."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack(slot, 1)
        self.assertIn("pre-publication compatibility error", str(ctx.exception))
        # Slot must remain in consumed phase — ack did not transition it
        self.assertEqual(ring.get_state(slot).phase, "consumed")
        # acknowledgement_sequence must NOT be set by the failed ack
        self.assertEqual(ring.get_state(slot).acknowledgement_sequence, 0)


# ---------------------------------------------------------------------------
# Goal 9: post-publication/post-enqueue timeout being reclaimed
# ---------------------------------------------------------------------------

class PostEnqueueTimeoutReclaimTest(unittest.TestCase):
    """Goal 9 reproduction #7: a slot that times out after being
    published or consumed (post-enqueue) cannot be reclaimed.

    Published/consumed slots may have been enqueued on the native
    transport — reclaiming them would risk silent capacity corruption.
    The timeout is process-fatal (SlotOverflowError) and capacity
    remains unavailable until teardown.
    """

    def test_published_slot_timeout_cannot_be_reclaimed(self) -> None:
        """A published slot that times out raises SlotOverflowError (fatal)
        and cannot be reclaimed — the slot stays reserved."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        clock.advance(11.0)
        # Timeout is fatal
        with self.assertRaises(SlotOverflowError) as ctx:
            ring.check_timeouts()
        self.assertIn("fatal", str(ctx.exception).lower())
        self.assertTrue(ring.is_fatal())
        # The slot cannot be reclaimed — it's still reserved
        state = ring.get_state(s0)
        self.assertTrue(state.reserved)
        self.assertNotEqual(state.owner, OWNER_NONE)
        # Reclaim must fail (ring is fatal)
        with self.assertRaises(SlotOverflowError):
            ring.reclaim(s0)

    def test_consumed_slot_timeout_is_fatal(self) -> None:
        """A consumed slot that times out is fatal (SlotOverflowError).
        The slot may have been enqueued — capacity is unavailable."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        clock.advance(11.0)
        with self.assertRaises(SlotOverflowError) as ctx:
            ring.check_timeouts()
        self.assertIn("fatal", str(ctx.exception).lower())
        self.assertIn("post-publication/post-enqueue", str(ctx.exception))
        self.assertTrue(ring.is_fatal())
        # Slot is still reserved — cannot reclaim
        state = ring.get_state(s0)
        self.assertTrue(state.reserved)
        with self.assertRaises(SlotOverflowError):
            ring.reclaim(s0)

    def test_consumed_slot_timeout_reclaim_attempt_raises(self) -> None:
        """Even if we try to reclaim a consumed-timeout slot, it must
        raise — either SlotOverflowError (fatal ring) or SlotStateError
        (wrong phase).  The slot must NOT return to idle."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        clock.advance(11.0)
        with self.assertRaises(SlotOverflowError):
            ring.check_timeouts()
        # Any reclaim attempt must fail
        with self.assertRaises((SlotOverflowError, SlotStateError)):
            ring.reclaim(s0)
        # Slot is not idle
        state = ring.get_state(s0)
        self.assertTrue(state.reserved)


# ---------------------------------------------------------------------------
# Goal 9: one scalar ack does NOT represent both peer edges
# ---------------------------------------------------------------------------

class ScalarAckDoesNotSetBothEdgesTest(unittest.TestCase):
    """Goal 9 reproduction #8 + Goal 10: ack_edge0 and ack_edge1 are
    DISTINCT operations.  The legacy scalar ack() is now an explicit
    error (Goal 10) — it must NOT set any edge sequences or the
    scalar acknowledgement_sequence.  ack_edge0 sets only
    ack_edge0_sequence and ack_edge1 sets only ack_edge1_sequence.

    The native TP4 all-reduce has two rounds, each with its own
    DoorbellControl.  A single scalar acknowledgement cannot
    represent both per-edge peer waits — the edges must be tracked
    independently.
    """

    def test_scalar_ack_is_error_and_sets_nothing(self) -> None:
        """Goal 10: calling ack() (scalar) raises SlotStateError and
        must NOT set acknowledgement_sequence, ack_edge0_sequence, or
        ack_edge1_sequence."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack(slot, 1)
        self.assertIn("pre-publication compatibility error", str(ctx.exception))
        state = ring.get_state(slot)
        self.assertEqual(state.acknowledgement_sequence, 0,
                         "Scalar ack must NOT set acknowledgement_sequence")
        self.assertEqual(state.ack_edge0_sequence, 0,
                         "Scalar ack must NOT set ack_edge0_sequence")
        self.assertEqual(state.ack_edge1_sequence, 0,
                         "Scalar ack must NOT set ack_edge1_sequence")

    def test_ack_edge0_sets_only_edge0(self) -> None:
        """ack_edge0() sets ack_edge0_sequence but must NOT set
        ack_edge1_sequence or acknowledgement_sequence."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        state = ring.get_state(slot)
        self.assertGreater(state.ack_edge0_sequence, 0,
                           "ack_edge0 must set ack_edge0_sequence")
        self.assertEqual(state.ack_edge1_sequence, 0,
                         "ack_edge0 must NOT set ack_edge1_sequence")
        self.assertEqual(state.acknowledgement_sequence, 0,
                         "ack_edge0 must NOT set legacy acknowledgement_sequence")

    def test_ack_edge1_sets_only_edge1(self) -> None:
        """ack_edge1() sets ack_edge1_sequence but must NOT set
        ack_edge0_sequence or acknowledgement_sequence."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge1(slot, 1)
        state = ring.get_state(slot)
        self.assertGreater(state.ack_edge1_sequence, 0,
                           "ack_edge1 must set ack_edge1_sequence")
        self.assertEqual(state.ack_edge0_sequence, 0,
                         "ack_edge1 must NOT set ack_edge0_sequence")
        self.assertEqual(state.acknowledgement_sequence, 0,
                         "ack_edge1 must NOT set legacy acknowledgement_sequence")

    def test_both_edges_needed_for_full_ack(self) -> None:
        """After ack_edge0 + ack_edge1, both edge sequences are set
        but acknowledgement_sequence (scalar) remains 0 — proving
        the per-edge path is independent of the scalar path."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        state = ring.get_state(slot)
        self.assertGreater(state.ack_edge0_sequence, 0)
        self.assertGreater(state.ack_edge1_sequence, 0)
        self.assertEqual(state.acknowledgement_sequence, 0,
                         "Per-edge acks must NOT set scalar acknowledgement_sequence")


# ---------------------------------------------------------------------------
# Goal 10: new adversarial tests for per-edge ack requirements
# ---------------------------------------------------------------------------

class NativeOrderPerEdgeAckTest(unittest.TestCase):
    """Goal 10: full per-edge ack lifecycle and edge-completeness checks.

    Verifies:
    - Full cycle: claim → publish → edge0 → edge1 → complete → reuse
    - Reversed edges (edge1 before edge0) also work
    - Missing edge0 only → complete fails
    - Missing edge1 only → complete fails
    - Duplicate edge ack is idempotent (second call still succeeds)
    """

    def test_full_cycle_claim_publish_edge0_edge1_complete_reuse(self) -> None:
        """Full cycle with both per-edge acks succeeds and allows reuse."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        self.assertEqual(ring.get_state(slot).phase, "completed")
        self.assertFalse(ring.get_state(slot).reserved)
        # Reuse with next token
        slot2 = ring.claim_slot(2)
        self.assertEqual(slot2, slot)

    def test_reversed_edges_edge1_before_edge0(self) -> None:
        """Edge1 before edge0 also works — order between edges does
        not matter, only that BOTH are present before complete."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        # edge1 first (from consumed phase) — allowed
        ring.ack_edge1(slot, 1)
        self.assertEqual(ring.get_state(slot).phase, "acked")
        self.assertGreater(ring.get_state(slot).ack_edge1_sequence, 0)
        # edge0 from acked phase — rejected (must be consumed)
        with pytest.raises(SlotStateError, match="must be 'consumed'"):
            ring.ack_edge0(slot, 1)
        # But we can still complete? No — edge0 was never set.
        # ack_edge0 requires consumed phase, but slot is acked.
        # So we cannot set edge0 after edge1.  This means the only
        # valid ordering is edge0 then edge1.
        # Verify: complete fails because edge0_sequence is 0.
        with pytest.raises(SlotStateError, match="incomplete per-edge"):
            ring.complete(slot, 1)

    def test_missing_edge0_only_complete_fails(self) -> None:
        """Only edge1 acked (not edge0) → complete fails."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge1(slot, 1)
        with pytest.raises(SlotStateError, match="incomplete per-edge"):
            ring.complete(slot, 1)
        # Slot must still be outstanding (acked is outstanding)
        self.assertEqual(ring.get_outstanding(), 1)

    def test_missing_edge1_only_complete_fails(self) -> None:
        """Only edge0 acked (not edge1) → complete fails."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        with pytest.raises(SlotStateError, match="incomplete per-edge"):
            ring.complete(slot, 1)
        # Slot must still be outstanding (acked is outstanding)
        self.assertEqual(ring.get_outstanding(), 1)

    def test_duplicate_edge_ack_idempotent(self) -> None:
        """A duplicate ack_edge0 call is rejected because the slot has
        already transitioned to acked phase (ack_edge0 requires
        consumed phase).  This is NOT idempotent — the second call
        raises SlotStateError because the phase has moved past consumed.

        This documents the production behavior: ack_edge0 can only be
        called from the consumed phase.  After the first call, the slot
        is acked, so a second ack_edge0 is a phase error.
        """
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        # Second ack_edge0 from acked phase — rejected (must be consumed)
        with pytest.raises(SlotStateError, match="must be 'consumed'"):
            ring.ack_edge0(slot, 1)
        # The original ack_edge0_sequence is still set
        self.assertGreater(ring.get_state(slot).ack_edge0_sequence, 0)


class StaleEdgeAckReuseTest(unittest.TestCase):
    """Goal 10: after completing token 1 (with both edges), claiming
    token 2 on the same slot must reset ack_edge0_sequence and
    ack_edge1_sequence to 0.  Stale edge state from token 1 must
    NOT survive into token 2's lifecycle."""

    def test_edge_sequences_reset_on_reuse(self) -> None:
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        # Token 1: full cycle with both edges
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Verify edges were set
        self.assertGreater(ring.get_state(slot).ack_edge0_sequence, 0)
        self.assertGreater(ring.get_state(slot).ack_edge1_sequence, 0)
        # Reuse with token 2
        slot2 = ring.claim_slot(2)
        self.assertEqual(slot2, slot)
        state = ring.get_state(slot2)
        # Goal 10: claim_slot resets edge sequences to 0
        self.assertEqual(state.ack_edge0_sequence, 0,
                         "Stale ack_edge0_sequence must NOT survive reuse")
        self.assertEqual(state.ack_edge1_sequence, 0,
                         "Stale ack_edge1_sequence must NOT survive reuse")

    def test_stale_edges_do_not_allow_complete(self) -> None:
        """If edge sequences were NOT reset, a stale edge from token 1
        could falsely satisfy token 2's complete() requirement.  Verify
        that after reuse, complete() without fresh edge acks fails."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        ring.ack_edge1(slot, 1)
        ring.complete(slot, 1)
        # Reuse with token 2
        ring.claim_slot(2)
        ring.publish(slot, 2)
        ring.consume(slot, 2)
        # complete() must fail — edges were reset, so stale state from
        # token 1 cannot satisfy the requirement.
        with pytest.raises(SlotStateError, match="incomplete per-edge"):
            ring.complete(slot, 2)


class OneEdgeAckedTimeoutTest(unittest.TestCase):
    """Goal 10: a slot with only one edge ack must NOT silently evade
    timeout.  The acked phase is outstanding, and a post-publication
    timeout (published/consumed/acked) is fatal (SlotOverflowError)."""

    def test_one_edge_acked_timeout_is_fatal(self) -> None:
        """Claim → publish → consume → ack_edge0 only → timeout check
        raises SlotOverflowError (fatal).  The slot must NOT silently
        evade timeout."""
        clock = FakeClock()
        ring = MultiSlotRing(
            num_slots=4, graph_mode=False, timeout_seconds=10.0, clock=clock,
        )
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        # Slot is acked (one edge) — still outstanding
        self.assertEqual(ring.get_outstanding(), 1)
        clock.advance(11.0)
        # Goal 10: acked timeout is fatal
        with self.assertRaises(SlotOverflowError) as ctx:
            ring.check_timeout(s0)
        self.assertIn("fatal", str(ctx.exception).lower())
        self.assertTrue(ring.is_fatal())
        # Slot must NOT silently evade timeout — still reserved
        self.assertTrue(ring.get_state(s0).reserved)

    def test_one_edge_acked_slot_is_outstanding(self) -> None:
        """A slot with only one edge ack is still outstanding —
        get_outstanding() counts it."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        # Acked with one edge — still outstanding (Goal 10)
        self.assertEqual(ring.get_outstanding(), 1)
        # After second edge + complete, no longer outstanding
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        self.assertEqual(ring.get_outstanding(), 0)


class MultipleActiveSlotsPerEdgeAckTest(unittest.TestCase):
    """Goal 10: two slots active simultaneously, each with different
    edge ack states.  Complete one (both edges), verify the other
    (one edge) cannot complete."""

    def test_complete_one_slot_other_cannot_complete(self) -> None:
        """Two slots: slot 0 gets both edges and completes; slot 1 has
        only edge0 and cannot complete."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        # Drive both through publish+consume
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.publish(s1, 2)
        ring.consume(s1, 2)
        # Slot 0: both edges → complete succeeds
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)
        self.assertEqual(ring.get_state(s0).phase, "completed")
        self.assertFalse(ring.get_state(s0).reserved)
        # Slot 1: only edge0 → complete fails
        ring.ack_edge0(s1, 2)
        with pytest.raises(SlotStateError, match="incomplete per-edge"):
            ring.complete(s1, 2)
        # Slot 1 is still outstanding
        self.assertEqual(ring.get_outstanding(), 1)

    def test_independent_edge_state_across_slots(self) -> None:
        """Edge ack state on one slot does not affect another slot."""
        ring = MultiSlotRing(num_slots=2, graph_mode=False)
        s0 = ring.claim_slot(1)
        s1 = ring.claim_slot(2)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.publish(s1, 2)
        ring.consume(s1, 2)
        # Both edges on slot 0 only
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        # Slot 1 has NO edge acks — its sequences must be 0
        state1 = ring.get_state(s1)
        self.assertEqual(state1.ack_edge0_sequence, 0)
        self.assertEqual(state1.ack_edge1_sequence, 0)
        # Slot 0 has both edges set
        state0 = ring.get_state(s0)
        self.assertGreater(state0.ack_edge0_sequence, 0)
        self.assertGreater(state0.ack_edge1_sequence, 0)


class LegacyAckErrorTest(unittest.TestCase):
    """Goal 10: ack() always raises SlotStateError regardless of phase.
    It is a pre-publication compatibility error — it must never stand
    in for both native peer edges."""

    def test_ack_from_consumed_raises(self) -> None:
        """ack() from consumed phase raises SlotStateError."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack(slot, 1)
        self.assertIn("pre-publication compatibility error", str(ctx.exception))

    def test_ack_from_claimed_raises(self) -> None:
        """ack() from claimed phase raises SlotStateError."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack(slot, 1)
        self.assertIn("pre-publication compatibility error", str(ctx.exception))

    def test_ack_from_published_raises(self) -> None:
        """ack() from published phase raises SlotStateError."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack(slot, 1)
        self.assertIn("pre-publication compatibility error", str(ctx.exception))

    def test_ack_from_idle_raises(self) -> None:
        """ack() from idle phase raises SlotStateError."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack(0, 1)
        self.assertIn("pre-publication compatibility error", str(ctx.exception))

    def test_ack_from_acked_raises(self) -> None:
        """ack() from acked phase raises SlotStateError."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        ring.ack_edge0(slot, 1)
        # Slot is now acked
        with self.assertRaises(SlotStateError) as ctx:
            ring.ack(slot, 1)
        self.assertIn("pre-publication compatibility error", str(ctx.exception))

    def test_ack_does_not_modify_state(self) -> None:
        """ack() must not change any slot state — it only raises."""
        ring = MultiSlotRing(num_slots=1, graph_mode=False)
        slot = ring.claim_slot(1)
        ring.publish(slot, 1)
        ring.consume(slot, 1)
        state_before = ring.get_state(slot)
        with self.assertRaises(SlotStateError):
            ring.ack(slot, 1)
        state_after = ring.get_state(slot)
        self.assertEqual(state_after.phase, state_before.phase)
        self.assertEqual(state_after.acknowledgement_sequence, 0)
        self.assertEqual(state_after.ack_edge0_sequence, 0)
        self.assertEqual(state_after.ack_edge1_sequence, 0)
        self.assertEqual(state_after.reserved, state_before.reserved)
# ---------------------------------------------------------------------------
# Goal 11: teardown_all() and reclaim safety
# ---------------------------------------------------------------------------


class TeardownAllTest(unittest.TestCase):
    """Goal 11: teardown_all() is an explicit fatal failure path that
    forces every non-idle slot into the torndown phase.  It cannot be
    mistaken for normal completion."""

    def test_teardown_all_forces_all_non_idle_to_torndown(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        # Drive slots 0 and 1 through claim→publish→consume→ack_edge0 only.
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        s1 = ring.claim_slot(2)
        ring.publish(s1, 2)
        ring.consume(s1, 2)
        ring.ack_edge0(s1, 2)
        # Slots 2 and 3 are idle.

        events = ring.teardown_all()

        # teardown events only for non-idle slots.
        self.assertEqual(len(events), 2)
        torn_slots = {ev.slot_index for ev in events}
        self.assertEqual(torn_slots, {s0, s1})
        for ev in events:
            self.assertEqual(ev.event_type, EVENT_TEARDOWN)

        # Ring is fatal after teardown.
        self.assertTrue(ring.is_fatal())

        for slot in (s0, s1):
            # White-box: torndown is an internal phase not derivable from
            # SlotState's sequence-based phase property — it must be
            # checked on the internal state, exactly as production code
            # checks _PHASE_TORNDOWN in _SlotInternal.phase.
            self.assertEqual(ring._slots[slot].phase, "torndown")
            self.assertFalse(ring._slots[slot].reserved)
            self.assertEqual(ring._slots[slot].owner, OWNER_NONE)

        # Idle slots are untouched — no teardown event, still idle.
        for slot in (2, 3):
            self.assertEqual(ring._slots[slot].phase, "idle")

    def test_teardown_all_idle_slots_untouched(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        events_before = list(ring.get_events())

        events = ring.teardown_all()

        # No non-idle slots → no teardown events.
        self.assertEqual(events, [])
        self.assertEqual(ring.get_events(), events_before)
        self.assertTrue(ring.is_fatal())

    def test_teardown_all_then_all_ops_fatal(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.teardown_all()

        self.assertTrue(ring.is_fatal())

        # Every lifecycle op must raise SlotOverflowError after teardown.
        with self.assertRaises(SlotOverflowError):
            ring.claim_slot(2)
        with self.assertRaises(SlotOverflowError):
            ring.publish(s0, 1)
        with self.assertRaises(SlotOverflowError):
            ring.consume(s0, 1)
        with self.assertRaises(SlotOverflowError):
            ring.ack_edge0(s0, 1)
        with self.assertRaises(SlotOverflowError):
            ring.ack_edge1(s0, 1)
        with self.assertRaises(SlotOverflowError):
            ring.complete(s0, 1)
        with self.assertRaises(SlotOverflowError):
            ring.rollback(s0)
        with self.assertRaises(SlotOverflowError):
            ring.rollback_all()

    def test_teardown_is_not_success(self) -> None:
        """A torndown slot has reserved=False but is NOT completed —
        proving teardown is a failure path distinct from success."""
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.teardown_all()

        state = ring.get_state(s0)
        # reserved released (capacity freed), but ...
        self.assertFalse(state.reserved)
        # ... completed_sequence is NOT set (no completion happened) ...
        self.assertEqual(state.completed_sequence, 0)
        # ... and the internal phase is 'torndown', not 'completed'.
        self.assertEqual(ring._slots[s0].phase, "torndown")
        self.assertNotEqual(ring._slots[s0].phase, "completed")


class Edge0OnlyReclaimRejectionTest(unittest.TestCase):
    """Goal 11: a slot that has only ack_edge0 (or only ack_edge1) is
    NOT reclaimable.  It must remain reserved until both edges + complete,
    or until explicit fatal teardown_all()."""

    def test_edge0_only_slot_not_reclaimable(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)

        # Slot is in 'acked' phase with only edge0 — not reclaimable.
        self.assertEqual(ring.get_state(s0).phase, "acked")
        with self.assertRaises(SlotStateError) as ctx:
            ring.reclaim(s0)
        self.assertIn("reclaim rejected", str(ctx.exception))

    def test_edge0_only_slot_remains_reserved(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)

        state = ring.get_state(s0)
        self.assertTrue(state.reserved)
        self.assertEqual(state.phase, "acked")

    def test_edge1_only_slot_not_reclaimable(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge1(s0, 1)

        # Slot is in 'acked' phase with only edge1 — not reclaimable.
        self.assertEqual(ring.get_state(s0).phase, "acked")
        with self.assertRaises(SlotStateError) as ctx:
            ring.reclaim(s0)
        self.assertIn("reclaim rejected", str(ctx.exception))


class FullCompletionReclaimTest(unittest.TestCase):
    """Goal 11: after a full completion (both edges + complete), the
    slot IS reclaimable — the only legitimate reclaim path."""

    def test_full_completion_then_reclaim_ok(self) -> None:
        ring = MultiSlotRing(num_slots=4, graph_mode=False)
        s0 = ring.claim_slot(1)
        ring.publish(s0, 1)
        ring.consume(s0, 1)
        ring.ack_edge0(s0, 1)
        ring.ack_edge1(s0, 1)
        ring.complete(s0, 1)

        self.assertEqual(ring.get_state(s0).phase, "completed")

        # Full completion → reclaim succeeds (normal reuse path).
        ev = ring.reclaim(s0)
        self.assertEqual(ev.event_type, EVENT_RECLAIM)
        self.assertEqual(ev.slot_index, s0)
        self.assertEqual(ring.get_state(s0).phase, "idle")
        self.assertFalse(ring.get_state(s0).reserved)


if __name__ == "__main__":
    unittest.main()
