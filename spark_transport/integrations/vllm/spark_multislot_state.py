"""CPU-mode modeled multi-slot ingress state machine.

This is a **Modeled** state machine — it does NOT execute native
code.  It models the protocol described in
``docs/agents/SIRCL_BASELINE.md`` under *Current TP4 all-reduce
state machine*, extending the single-buffer producer/consumer/ack/
completion vocabulary to a ring of N independent slots.

Evidence label: **Modeled**.  This module tests protocol invariants
(claim/publish/consume/complete/ack/reclaim ordering, backpressure,
wraparound, timeout, rollback, graph-capture ordering) but does not
prove native scheduling, RDMA progress, CUDA ordering, numerical
equivalence, or speed.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NUM_SLOTS = 64
# Native command/doorbell sequences are uint64 (see tp4_graph_command.hpp,
# gpu_doorbell.hpp). The doorbell token packs sequence and Q into one
# uint64: token = (sequence << 10) | q. The maximum sequence is
# uint64::max >> 10. We model wrap at this native boundary, not a
# 16-bit generation modulus.
COMMAND_TOKEN_MODULUS = 1 << 64  # 2^64 — native uint64 width
GENERATION_MODULUS = COMMAND_TOKEN_MODULUS  # alias for backward compat

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_CLAIM = "claim"
EVENT_PUBLISH = "publish"
EVENT_CONSUME = "consume"
EVENT_COMPLETE = "complete"
EVENT_ACK = "ack"  # legacy scalar ack — replaced by per-edge acks
EVENT_ACK_EDGE0 = "ack_edge0"
EVENT_ACK_EDGE1 = "ack_edge1"
EVENT_RECLAIM = "reclaim"
EVENT_OVERFLOW = "overflow"
EVENT_TIMEOUT = "timeout"
EVENT_TEARDOWN = "teardown"

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_CLAIM,
    EVENT_PUBLISH,
    EVENT_CONSUME,
    EVENT_COMPLETE,
    EVENT_ACK,
    EVENT_ACK_EDGE0,
    EVENT_ACK_EDGE1,
    EVENT_RECLAIM,
    EVENT_OVERFLOW,
    EVENT_TIMEOUT,
    EVENT_TEARDOWN,
})

# ---------------------------------------------------------------------------
# Slot owners
# ---------------------------------------------------------------------------

OWNER_NONE = "none"
OWNER_GPU_PRODUCER = "gpu_producer"
OWNER_CPU_CONSUMER = "cpu_consumer"

VALID_OWNERS: frozenset[str] = frozenset({
    OWNER_NONE,
    OWNER_GPU_PRODUCER,
    OWNER_CPU_CONSUMER,
})

# ---------------------------------------------------------------------------
# Internal phases (not exposed through SlotState — derived from sequences)
# ---------------------------------------------------------------------------

_PHASE_IDLE = "idle"
_PHASE_CLAIMED = "claimed"
_PHASE_PUBLISHED = "published"
_PHASE_CONSUMED = "consumed"
_PHASE_COMPLETED = "completed"
_PHASE_ACKED = "acked"
_PHASE_TIMEOUT = "timeout"
# Slots in these phases are "outstanding" (claimed but not completed).
# Goal 10: acked slots with incomplete per-edge acks are also outstanding —
# a one-edge-acked slot must remain outstanding and timeout fatally after
# publication; it cannot silently evade timeout while other slots continue.
_OUTSTANDING_PHASES = frozenset({
    _PHASE_CLAIMED,
    _PHASE_PUBLISHED,
    _PHASE_CONSUMED,
    _PHASE_ACKED,
})

# Slots in these phases can be reclaimed (returned to idle) during normal
# reuse.  Goal 11 requirement 6: _PHASE_ACKED must NEVER be normally
# reclaimable.  A slot with only one edge acknowledged remains reserved
# until the second edge, GPU completion, and CPU completed sequence
# finish, or until explicit fatal teardown.  Only COMPLETED may be
# reclaimed during normal reuse.
_RECLAIMABLE_PHASES = frozenset({
    _PHASE_COMPLETED,
})

# Explicit teardown phase — a separate failure path that cannot be
# mistaken for success.  Only reachable via teardown_all().
_PHASE_TORNDOWN = "torndown"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SlotStateError(ValueError):
    """Raised on an invalid slot state transition."""


class SlotOverflowError(SlotStateError):
    """Raised on overflow — fatal in graph mode."""


class BackpressureError(SlotStateError):
    """Raised when all slots are outstanding and a new claim is attempted."""


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class SlotState:
    """Public snapshot of one slot's state.

    The phase is *derived* from the sequence fields and ``reserved``
    rather than stored explicitly, matching the native doorbell layout
    where the control words themselves encode ownership.
    """

    slot_index: int
    owner: str  # 'gpu_producer' | 'cpu_consumer' | 'none'
    command_token: int  # uint64, shared through lifecycle
    producer_sequence: int
    consumer_sequence: int
    completed_sequence: int
    acknowledgement_sequence: int  # legacy scalar ack
    ack_edge0_sequence: int  # per-edge: round 0 ack
    ack_edge1_sequence: int  # per-edge: round 1 ack
    reserved: bool

    @property
    def phase(self) -> str:
        """Derive the lifecycle phase from the public fields.

        Mapping (0 = unset, sequences start at 1):

        - ``idle``      — no sequences set, not reserved
        - ``claimed``    — gpu_producer, reserved, no producer_seq
        - ``published``   — gpu_producer, reserved, producer_seq set
        - ``consumed``    — cpu_consumer, reserved, consumer_seq, no completed/ack
        - ``acked``        — ack_seq set, reserved (peer ack during GPU exec, before complete)
        - ``completed``   — completed_seq set, reserved cleared (capacity released)
        - ``timeout``      — owner none, reserved, no completed_seq

        Native ordering: ack (DoorbellControl.acknowledgement_sequence)
        is received DURING GPU execution, BEFORE complete()
        (Tp4GraphCommandRing.completed_sequence).  So acked precedes
        completed.  The phase derivation checks completed *first* (it
        is terminal), then acked (intermediate), then reserved states.
        """
        if self.completed_sequence > 0:
            return _PHASE_COMPLETED
        if self.acknowledgement_sequence > 0 or self.ack_edge0_sequence > 0 or self.ack_edge1_sequence > 0:
            return _PHASE_ACKED
        if not self.reserved:
            return _PHASE_IDLE
        if self.owner == OWNER_CPU_CONSUMER and self.consumer_sequence > 0:
            return _PHASE_CONSUMED
        if self.owner == OWNER_GPU_PRODUCER and self.producer_sequence > 0:
            return _PHASE_PUBLISHED
        if self.owner == OWNER_GPU_PRODUCER:
            return _PHASE_CLAIMED
        # owner == 'none' but reserved → timeout
        return _PHASE_TIMEOUT


@dataclass(frozen=True)
class SlotEvent:
    """One recorded state-machine event."""

    slot_index: int
    event_type: str  # one of VALID_EVENT_TYPES
    sequence: int
    timestamp: float


@dataclass(frozen=True)
class ProtocolSpec:
    """Documents the multi-slot ingress protocol.

    Every field is a human-readable description or an exact tuple of
    allowed values.  This dataclass is **documentation**, not runtime
    configuration — the ring's runtime parameters are passed to
    ``MultiSlotRing.__init__``.
    """

    num_slots: int
    generation_modulus: int
    graph_mode: bool
    timeout_seconds: float

    per_slot_states: tuple[str, ...]
    owner_values: tuple[str, ...]
    event_types: tuple[str, ...]
    sequence_fields: tuple[str, ...]

    transition_rules: tuple[str, ...]
    ordering_rules: tuple[str, ...]
    backpressure_rule: str
    wraparound_rule: str
    aba_protection_rule: str
    timeout_rule: str
    failure_rule: str
    graph_capture_rule: str
    rollback_rule: str
    capacity_release_rule: str
    sequence_equality_rule: str
    reservation_rule: str
    counter_descriptions: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal per-slot state
# ---------------------------------------------------------------------------

@dataclass
class _SlotInternal:
    """Mutable per-slot state tracked by the ring.

    Native command-ring semantics (tp4_graph_command.hpp):
    - One uint64 ``command_token`` is assigned at claim time and
      shared through publish/consume/complete/ack — it is NOT
      replaced by a new sequence at each lifecycle transition.
    - The token encodes both sequence and Q: token = (seq << 10) | q.
    - ``last_token`` preserves ABA history: a slot cannot be reused
      with the same token (uint64, not 16-bit generation).
    """

    phase: str = _PHASE_IDLE
    owner: str = OWNER_NONE
    command_token: int = 0  # uint64, shared through lifecycle
    producer_sequence: int = 0  # native producer_sequence (uint64)
    consumer_sequence: int = 0  # native consumer_sequence (uint64)
    completed_sequence: int = 0  # native completed_sequence (uint64)
    acknowledgement_sequence: int = 0  # legacy scalar ack (uint64)
    ack_edge0_sequence: int = 0  # per-edge: round 0 DoorbellControl.acknowledgement_sequence
    ack_edge1_sequence: int = 0  # per-edge: round 1 DoorbellControl.acknowledgement_sequence
    reserved: bool = False
    last_token: int = -1  # ABA: previous command_token (-1 = no prior)
    claim_timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Multi-slot ring state machine
# ---------------------------------------------------------------------------

@dataclass
class MultiSlotRing:
    """Modeled ring of N slots with the TP4 doorbell protocol.

    All operations are CPU-only and synchronous.  "Blocking" (rule 9)
    is modeled by raising ``BackpressureError``; the caller decides
    whether to retry.

    Native lifecycle (tp4_graph_command.hpp, gpu_graph_command.cuh):
    - ``claim_sequence(ring)`` atomically advances ``claimed_sequence``
      and returns the new uint64 sequence.  One sequence is the
      command's lifecycle token — it is published, consumed, and
      completed with the *same* value, not a new value at each step.
    - ``publish_command`` writes the descriptor and CAS-publishes
      ``sequence`` into ``published_sequence``.
    - The consumer waits for ``expected_sequence``, acquires the
      command, and calls ``tp4_graph_command_complete(ring, sequence)``
      which advances ``completed_sequence``.
    - Capacity is released by ``completed_sequence``, not by
      ``consumed_sequence``: a new claim is permitted while
      ``claimed_sequence - completed_sequence < kTp4GraphCommandCapacity``.

    This model aligns with that lifecycle: ``claim_slot(token)`` assigns
    one uint64 ``command_token`` to the slot.  ``publish``,
    ``consume``, ``complete``, and ``ack`` all receive the *same*
    token and it is verified against the slot's ``command_token``.

    Parameters
    ----------
    num_slots
        Ring width (default 64, matching ``kTp4GraphCommandCapacity``).
    graph_mode
        If True, exact token equality is required and slot
        operations must follow claim order (no skipping).
    generation_modulus
        Generation wraparound bound (default 2^64, native uint64).
    timeout_seconds
        Bounded timeout for outstanding slots.
    clock
        Injectable monotonic clock for testing (default ``time.monotonic``).
    """

    num_slots: int = DEFAULT_NUM_SLOTS
    graph_mode: bool = True
    generation_modulus: int = GENERATION_MODULUS
    timeout_seconds: float = 30.0
    clock: Callable[[], float] = field(default=time.monotonic)

    # Internal state — initialised in __post_init__
    _slots: list[_SlotInternal] = field(default_factory=list, init=False, repr=False)
    _events: list[SlotEvent] = field(default_factory=list, init=False, repr=False)
    _fatal: bool = field(default=False, init=False, repr=False)

    # FIFOs for graph-mode ordering (no skipping)
    _claim_fifo: deque[int] = field(default_factory=deque, init=False, repr=False)
    _publish_fifo: deque[int] = field(default_factory=deque, init=False, repr=False)
    _consume_fifo: deque[int] = field(default_factory=deque, init=False, repr=False)

    # Global monotonic sequence: native claimed_sequence is a single
    # uint64 counter shared across all slots.  No two slots may
    # simultaneously hold the same lifecycle token.  This tracks the
    # highest token ever claimed across the entire ring.
    _global_last_token: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.num_slots, int) or isinstance(self.num_slots, bool):
            raise SlotStateError("num_slots must be a positive int")
        if self.num_slots < 1:
            raise SlotStateError("num_slots must be >= 1")
        if not isinstance(self.generation_modulus, int) or isinstance(
            self.generation_modulus, bool
        ):
            raise SlotStateError("generation_modulus must be an int >= 2")
        if self.generation_modulus < 2:
            raise SlotStateError("generation_modulus must be >= 2")
        if self.timeout_seconds <= 0:
            raise SlotStateError("timeout_seconds must be > 0")
        self._slots = [_SlotInternal() for _ in range(self.num_slots)]

    # -- private helpers --------------------------------------------------

    def _check_fatal(self) -> None:
        if self._fatal:
            raise SlotOverflowError(
                "ring is in fatal overflow state — no further operations"
            )

    def _validate_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise SlotStateError(f"slot index must be int, got {type(slot).__name__}")
        if slot < 0 or slot >= self.num_slots:
            raise SlotStateError(
                f"slot {slot} out of range [0, {self.num_slots})"
            )

    def _record(self, slot: int, event_type: str, sequence: int) -> SlotEvent:
        event = SlotEvent(
            slot_index=slot,
            event_type=event_type,
            sequence=sequence,
            timestamp=self.clock(),
        )
        self._events.append(event)
        return event

    def _check_token(self, slot: int, sequence: int) -> None:
        """Verify that ``sequence`` matches the slot's ``command_token``.

        In the native protocol, one uint64 sequence is claimed,
        published, consumed, and completed — it is the same value
        at every lifecycle step, not a new counter at each step.
        """
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise SlotStateError(f"sequence must be int, got {type(sequence).__name__}")
        s = self._slots[slot]
        if sequence != s.command_token:
            if sequence < s.command_token:
                raise SlotStateError(
                    f"stale sequence {sequence} < command_token "
                    f"{s.command_token} on slot {slot}"
                )
            # sequence > command_token: overflow in graph mode
            if self.graph_mode:
                self._fatal = True
                raise SlotOverflowError(
                    f"overflow: sequence {sequence} > command_token "
                    f"{s.command_token} on slot {slot} (graph mode "
                    f"requires exact token equality)"
                )
            # Non-graph mode: accept if >= token (forward progress)
            # but still record the token mismatch
            raise SlotStateError(
                f"sequence {sequence} != command_token "
                f"{s.command_token} on slot {slot}"
            )

    # -- public API --------------------------------------------------------

    def claim_slot(self, token: int) -> int:
        """Claim the slot for the given command token.

        **Contiguous tokens (C1):** the token must be exactly
        ``_global_last_token + 1``.  Non-contiguous tokens (e.g.
        ``claim_slot(3)`` immediately after token 1) are rejected.
        Native ``claimed_sequence`` advances by exactly 1 per claim.

        **Deterministic slot mapping (C2):** the slot is
        ``(token - 1) % num_slots``.  If that exact slot is not
        reusable, ``BackpressureError`` is raised — the ring
        never picks a different free slot.

        **Capacity release (C3):** in both graph and non-graph mode,
        ``complete()`` releases ring capacity by clearing
        ``reserved`` and advancing ``completed_sequence``.  This
        matches the native invariant
        ``claimed_sequence - completed_sequence < capacity``.  A
        slot is reusable after it is *completed* (or *acked* then
        *completed*), not after ack alone.  Peer doorbell acks are
        per-edge events received during GPU execution, before CPU
        completion — they do not govern ring capacity.
        """
        self._check_fatal()
        if not isinstance(token, int) or isinstance(token, bool):
            raise SlotStateError(
                f"token must be int, got {type(token).__name__}"
            )
        if token < 1 or token >= self.generation_modulus:
            raise SlotStateError(
                f"command_token {token} out of range "
                f"[1, {self.generation_modulus})"
            )

        # Contiguous tokens: must be exactly last + 1.
        expected = self._global_last_token + 1
        if token != expected:
            raise SlotStateError(
                f"non-contiguous token {token}: expected {expected} "
                f"(last token {self._global_last_token} + 1)"
            )

        # Deterministic slot mapping: (token - 1) % num_slots.
        slot = (token - 1) % self.num_slots
        s = self._slots[slot]

        # Determine whether this slot is reusable.
        # Goal 11 requirement 6: only COMPLETED slots are normally
        # reclaimable.  An acked slot (one or both edges acked but not
        # completed) must remain reserved — it cannot be reused until
        # complete() is called (which requires both edges), or until
        # explicit fatal teardown_all().
        if s.phase not in (_PHASE_IDLE, _PHASE_COMPLETED):
            raise BackpressureError(
                f"backpressure: slot {slot} (token {token}) is in "
                f"'{s.phase}' phase — not reusable"
            )

        if s.reserved:
            raise BackpressureError(
                f"slot {slot} (token {token}) is reserved "
                f"(phase '{s.phase}') — backpressure"
            )

        # Per-slot ABA protection: token must exceed last_token.
        if token <= s.last_token:
            raise SlotStateError(
                f"ABA violation: token {token} <= last_token "
                f"{s.last_token} on slot {slot}"
            )

        s.phase = _PHASE_CLAIMED
        s.owner = OWNER_GPU_PRODUCER
        s.reserved = True
        s.command_token = token
        s.last_token = token
        s.claim_timestamp = self.clock()
        s.producer_sequence = 0
        s.consumer_sequence = 0
        s.completed_sequence = 0
        s.acknowledgement_sequence = 0
        s.ack_edge0_sequence = 0
        s.ack_edge1_sequence = 0

        self._global_last_token = token
        self._claim_fifo.append(slot)
        self._record(slot, EVENT_CLAIM, token)
        return slot

    def publish(self, slot: int, sequence: int) -> SlotEvent:
        """GPU publishes ``producer_sequence`` on the slot.

        The slot must be in *claimed* phase.  ``sequence`` must equal
        the slot's ``command_token`` (one token per lifecycle).
        In graph mode the slot must be the oldest claimed-but-unpublished
        slot (no skipping).
        """
        self._check_fatal()
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase != _PHASE_CLAIMED:
            if s.phase == _PHASE_PUBLISHED:
                raise SlotStateError(
                    f"double-publish rejected: slot {slot} already published"
                )
            raise SlotStateError(
                f"publish rejected: slot {slot} is in '{s.phase}' phase, "
                f"must be 'claimed'"
            )

        # Graph-mode ordering: must publish oldest claimed slot first
        if self.graph_mode and self._claim_fifo:
            if self._claim_fifo[0] != slot:
                raise SlotStateError(
                    f"graph-mode ordering: must publish slot "
                    f"{self._claim_fifo[0]} before slot {slot} (no skipping)"
                )

        self._check_token(slot, sequence)

        s.producer_sequence = sequence
        s.phase = _PHASE_PUBLISHED
        if self.graph_mode and self._claim_fifo:
            self._claim_fifo.popleft()
        self._publish_fifo.append(slot)
        return self._record(slot, EVENT_PUBLISH, sequence)

    def consume(self, slot: int, sequence: int) -> SlotEvent:
        """CPU consumes the slot's published data.

        The slot must be in *published* phase.  ``sequence`` must equal
        the slot's ``command_token`` (one token per lifecycle).
        In graph mode the slot must be the oldest published-but-unconsumed
        slot.
        """
        self._check_fatal()
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase != _PHASE_PUBLISHED:
            raise SlotStateError(
                f"consume rejected: slot {slot} is in '{s.phase}' phase, "
                f"must be 'published' (consume before publish rejected)"
            )

        if self.graph_mode and self._publish_fifo:
            if self._publish_fifo[0] != slot:
                raise SlotStateError(
                    f"graph-mode ordering: must consume slot "
                    f"{self._publish_fifo[0]} before slot {slot} (no skipping)"
                )

        self._check_token(slot, sequence)

        s.consumer_sequence = sequence
        s.owner = OWNER_CPU_CONSUMER
        s.phase = _PHASE_CONSUMED
        if self.graph_mode and self._publish_fifo:
            self._publish_fifo.popleft()
        self._consume_fifo.append(slot)
        return self._record(slot, EVENT_CONSUME, sequence)

    def complete(self, slot: int, sequence: int) -> SlotEvent:
        """CPU publishes ``completed_sequence`` — releases ring capacity.

        The slot must be in *acked* phase with **both** per-edge
        acknowledgements received.  ``sequence`` must equal the slot's
        ``command_token`` (one token per lifecycle).

        **Goal 10 requirement 4:** ``complete()`` requires both edge
        acknowledgements (``ack_edge0`` and ``ack_edge1``) for the
        current token before GPU/CPU completion or reuse.  Edge1-only
        and legacy-scalar-only completion must fail closed.  A slot with
        only one edge ack remains outstanding (in *acked* phase) and
        cannot be completed until both edges are acknowledged.

        **Both graph and non-graph mode:** completion releases ring
        capacity — ``reserved`` is cleared so the slot becomes
        available for new claims, matching the native
        ``claimed_sequence - completed_sequence < capacity`` invariant.
        The native ``completed_sequence`` is published by the CPU
        progress thread after the GPU kernel finishes; it is the
        capacity-release signal, not the peer ack.

        Peer doorbell acknowledgements (``acknowledgement_sequence`` on
        ``DoorbellControl``) are received DURING the GPU kernel
        execution — before the kernel finishes and before CPU
        completion.  Both per-edge acks (``ack_edge0`` and
        ``ack_edge1``) must be received before ``complete()``.
        """
        self._check_fatal()
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase not in (_PHASE_CONSUMED, _PHASE_ACKED):
            raise SlotStateError(
                f"complete rejected: slot {slot} is in '{s.phase}' phase, "
                f"must be 'consumed' or 'acked'"
            )

        # Goal 10 requirement 4: both per-edge acks required before completion.
        # Edge1-only, edge0-only, and legacy-scalar-only completion must
        # fail closed.  A one-edge-acked slot remains outstanding.
        if s.ack_edge0_sequence == 0 or s.ack_edge1_sequence == 0:
            raise SlotStateError(
                f"complete rejected: slot {slot} has incomplete per-edge "
                f"acks (edge0={s.ack_edge0_sequence}, "
                f"edge1={s.ack_edge1_sequence}) — both edges required "
                f"before completion (Goal 10 requirement 4)"
            )

        if self.graph_mode and self._consume_fifo:
            if self._consume_fifo[0] != slot:
                raise SlotStateError(
                    f"graph-mode ordering: must complete slot "
                    f"{self._consume_fifo[0]} before slot {slot} (no skipping)"
                )

        self._check_token(slot, sequence)

        s.completed_sequence = sequence
        s.phase = _PHASE_COMPLETED
        s.reserved = False  # Release capacity (both modes).
        if self.graph_mode and self._consume_fifo:
            self._consume_fifo.popleft()
        return self._record(slot, EVENT_COMPLETE, sequence)

    def ack(self, slot: int, sequence: int) -> SlotEvent:
        """Legacy scalar peer doorbell acknowledgement — now an error.

        **Goal 10 requirement 4:** The legacy scalar ``ack()`` is now
        an explicitly pre-publication compatibility error.  It must
        never stand in for both native peer edges.  Use ``ack_edge0()``
        and ``ack_edge1()`` to model the actual two-round TP4 per-edge
        acknowledgement structure.

        This method always raises ``SlotStateError`` — it is retained
        as an explicit error site so callers that still use the old
        scalar ack get a clear failure instead of silently setting only
        one edge.
        """
        self._check_fatal()
        self._validate_slot(slot)
        raise SlotStateError(
            "ack() is a pre-publication compatibility error (Goal 10): "
            "use ack_edge0() and ack_edge1() instead — scalar ack must "
            "never stand in for both native peer edges"
        )


    def ack_edge0(self, slot: int, sequence: int) -> SlotEvent:
        """Per-edge doorbell acknowledgement for round 0.

        The native TP4 all-reduce (gpu_tp4_tensor.cu) has two rounds,
        each with its own ``DoorbellControl``.  Round 0 publishes
        ``control0->producer_sequence``, waits for
        ``control0->remote_sequence``, then waits for
        ``control0->acknowledgement_sequence`` after publishing
        ``control0->consumer_sequence``.

        This method models the round-0 peer acknowledgement.  It may
        be called from the *consumed* phase, BEFORE ``complete()``.
        Both per-edge acks must be received before completion in a
        full-fidelity trace.
        """
        self._check_fatal()
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase != _PHASE_CONSUMED:
            raise SlotStateError(
                f"ack_edge0 rejected: slot {slot} is in '{s.phase}' "
                f"phase, must be 'consumed'"
            )

        self._check_token(slot, sequence)

        s.ack_edge0_sequence = sequence
        # Transition to acked once at least one edge ack is received.
        if s.phase != _PHASE_ACKED:
            s.phase = _PHASE_ACKED
        return self._record(slot, EVENT_ACK_EDGE0, sequence)

    def ack_edge1(self, slot: int, sequence: int) -> SlotEvent:
        """Per-edge doorbell acknowledgement for round 1.

        Round 1 waits for ``control1->remote_sequence``, then waits
        for ``control1->acknowledgement_sequence`` after publishing
        ``control1->consumer_sequence``.  This is the second
        per-edge peer wait, distinct from round 0.

        This method models the round-1 peer acknowledgement.  It may
        be called from the *consumed* or *acked* phase (after
        ack_edge0), BEFORE ``complete()``.
        """
        self._check_fatal()
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase not in (_PHASE_CONSUMED, _PHASE_ACKED):
            raise SlotStateError(
                f"ack_edge1 rejected: slot {slot} is in '{s.phase}' "
                f"phase, must be 'consumed' or 'acked'"
            )

        self._check_token(slot, sequence)

        s.ack_edge1_sequence = sequence
        if s.phase != _PHASE_ACKED:
            s.phase = _PHASE_ACKED
        return self._record(slot, EVENT_ACK_EDGE1, sequence)

    def rollback(self, slot: int) -> SlotEvent | None:
        """Roll back a **pre-publication** slot to idle (partial failure).

        Goal 9: only demonstrably pre-publication/pre-enqueue
        reservations (claimed slots that have NOT been published) may
        be safely abandoned.  Published or consumed slots may have
        been enqueued on the native transport and must NOT be rolled
        back — their failure is process-fatal.

        Slots in 'claimed' phase are rolled back to idle.  Slots in
        'published', 'consumed', or any other phase are untouched
        (returns ``None``).
        """
        self._check_fatal()
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase != _PHASE_CLAIMED:
            return None

        token = s.command_token
        self._reset_slot(slot)
        return self._record(slot, EVENT_RECLAIM, token)

    def reclaim(self, slot: int) -> SlotEvent:
        """Return a completed, acked, or timed-out slot to idle.

        Rule 4: a slot must be completed before reclaim (normal path).
        Timed-out slots may also be reclaimed (error-recovery path).
        """
        self._check_fatal()
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase not in _RECLAIMABLE_PHASES:
            raise SlotStateError(
                f"reclaim rejected: slot {slot} is in '{s.phase}' phase, "
                f"must be one of {sorted(_RECLAIMABLE_PHASES)} "
                f"(reclaim before complete rejected)"
            )

        token = s.command_token
        self._reset_slot(slot)
        return self._record(slot, EVENT_RECLAIM, token)


    def rollback_all(self) -> list[SlotEvent]:
        """Roll back all pre-publication (claimed) slots (bulk partial failure).

        Goal 9: only claimed (pre-publication) slots are rolled back.
        Published/consumed slots are NOT rolled back — their failure
        is process-fatal.
        """
        self._check_fatal()
        events: list[SlotEvent] = []
        for slot in range(self.num_slots):
            ev = self.rollback(slot)
            if ev is not None:
                events.append(ev)
        return events

    def teardown_all(self) -> list[SlotEvent]:
        """Explicit fatal teardown — force all slots to torndown state.

        Goal 11 requirement 6: teardown is a separate, explicit failure
        path that cannot be mistaken for success.  All slots are forced
        to ``_PHASE_TORNDOWN`` regardless of their current phase.
        This is the ONLY way to reclaim acked/published/consumed slots
        outside the normal completion path.  After teardown, the ring is
        fatal and no further operations are permitted.

        This is distinct from ``reclaim()`` (normal reuse of completed
        slots) and ``rollback_all()`` (pre-publication partial failure).
        Teardown means the worker is terminating — all in-flight work
        is abandoned, not completed.
        """
        events: list[SlotEvent] = []
        self._fatal = True
        for slot in range(self.num_slots):
            s = self._slots[slot]
            if s.phase == _PHASE_IDLE:
                continue
            token = s.command_token
            s.phase = _PHASE_TORNDOWN
            s.owner = OWNER_NONE
            s.reserved = False
            self._remove_from_fifos(slot)
            events.append(self._record(slot, "teardown", token))
        return events

    def check_timeout(self, slot: int, now: float | None = None) -> SlotEvent | None:
        """Check whether a single slot has exceeded the bounded timeout.

        Goal 9: a published or consumed slot that times out is
        **process-fatal** — the slot may have been enqueued on the
        native transport and cannot be safely reclaimed.  Capacity
        remains unavailable until teardown.  Only a claimed
        (pre-publication) slot that times out may be reclaimed in
        non-graph mode.

        Returns the ``SlotEvent`` if the slot timed out, ``None``
        otherwise.  Raises ``SlotOverflowError`` for fatal timeouts.
        """
        self._validate_slot(slot)
        s = self._slots[slot]

        if s.phase not in _OUTSTANDING_PHASES:
            return None

        current = now if now is not None else self.clock()
        elapsed = current - s.claim_timestamp
        if elapsed <= self.timeout_seconds:
            return None

        # Published, consumed, or acked slots: timeout is FATAL (Goal 9/10).
        # The slot may have been enqueued on the native transport;
        # reclaiming it would risk silent capacity corruption.
        # Goal 10: a one-edge-acked slot must timeout fatally after
        # publication; it cannot silently evade timeout.
        if s.phase in (_PHASE_PUBLISHED, _PHASE_CONSUMED, _PHASE_ACKED):
            self._fatal = True
            gen = s.command_token
            self._remove_from_fifos(slot)
            self._record(slot, EVENT_TIMEOUT, gen)
            raise SlotOverflowError(
                f"fatal timeout on slot {slot} in '{s.phase}' phase: "
                f"post-publication/post-enqueue timeout — capacity "
                f"unavailable until teardown"
            )

        # Graph mode: any timeout is fatal — the native worker
        # terminates because a graph-published unfulfillable wait
        # cannot safely reclaim to idle.
        if self.graph_mode:
            self._fatal = True
            gen = s.command_token
            self._remove_from_fifos(slot)
            self._record(slot, EVENT_TIMEOUT, gen)
            raise SlotOverflowError(
                f"graph-mode timeout on slot {slot}: unfulfillable "
                f"wait — worker termination boundary (fatal)"
            )

        # Non-graph mode, claimed (pre-publication) slot: transition to
        # timeout (reclaimable).  This is safe because the slot has
        # not been published or enqueued.
        gen = s.command_token
        s.phase = _PHASE_TIMEOUT
        s.owner = OWNER_NONE
        # reserved stays True — slot is not available until reclaimed
        self._remove_from_fifos(slot)
        return self._record(slot, EVENT_TIMEOUT, gen)

    def check_timeouts(self, now: float | None = None) -> list[SlotEvent]:
        """Check all slots for timeout.  Returns events for newly timed-out slots."""
        events: list[SlotEvent] = []
        for slot in range(self.num_slots):
            ev = self.check_timeout(slot, now=now)
            if ev is not None:
                events.append(ev)
        return events

    def get_state(self, slot: int) -> SlotState:
        """Return a public snapshot of the slot's current state."""
        self._validate_slot(slot)
        s = self._slots[slot]
        return SlotState(
            slot_index=slot,
            owner=s.owner,
            command_token=s.command_token,
            producer_sequence=s.producer_sequence,
            consumer_sequence=s.consumer_sequence,
            completed_sequence=s.completed_sequence,
            acknowledgement_sequence=s.acknowledgement_sequence,
            ack_edge0_sequence=s.ack_edge0_sequence,
            ack_edge1_sequence=s.ack_edge1_sequence,
            reserved=s.reserved,
        )

    def get_outstanding(self) -> int:
        """Count of outstanding (claimed but not completed) slots."""
        return sum(
            1 for s in self._slots if s.phase in _OUTSTANDING_PHASES
        )

    def get_events(self) -> list[SlotEvent]:
        """Return all recorded events (oldest first)."""
        return list(self._events)

    def is_fatal(self) -> bool:
        """True after a graph-mode overflow has put the ring in fatal state."""
        return self._fatal

    # -- private helpers --------------------------------------------------

    def _reset_slot(self, slot: int) -> None:
        """Reset a slot to idle and remove it from all FIFOs.

        ``last_token`` is NOT reset — it preserves ABA history so the
        same command_token cannot be reused on this slot.  Monotonic
        advance is enforced: a new claim must use a token strictly
        greater than ``last_token``.
        """
        s = self._slots[slot]
        s.phase = _PHASE_IDLE
        s.owner = OWNER_NONE
        s.command_token = 0
        s.completed_sequence = 0
        s.acknowledgement_sequence = 0
        s.ack_edge0_sequence = 0
        s.ack_edge1_sequence = 0
        s.reserved = False
        # last_token is NOT reset — preserves ABA history
        self._remove_from_fifos(slot)

    def _remove_from_fifos(self, slot: int) -> None:
        """Remove a slot from all graph-mode FIFOs."""
        try:
            self._claim_fifo.remove(slot)
        except ValueError:
            pass
        try:
            self._publish_fifo.remove(slot)
        except ValueError:
            pass
        try:
            self._consume_fifo.remove(slot)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Protocol specification
# ---------------------------------------------------------------------------

def default_protocol_spec(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    graph_mode: bool = True,
    generation_modulus: int = GENERATION_MODULUS,
    timeout_seconds: float = 30.0,
) -> ProtocolSpec:
    """Return the standard protocol specification document."""
    return ProtocolSpec(
        num_slots=num_slots,
        generation_modulus=generation_modulus,
        graph_mode=graph_mode,
        timeout_seconds=timeout_seconds,
        per_slot_states=(
            "idle",
            "claimed",
            "published",
            "consumed",
            "completed",
            "acked",
            "timeout",
            "torndown",
        ),
        owner_values=(
            OWNER_NONE,
            OWNER_GPU_PRODUCER,
            OWNER_CPU_CONSUMER,
        ),
        event_types=tuple(sorted(VALID_EVENT_TYPES)),
        sequence_fields=(
            "producer_sequence",
            "consumer_sequence",
            "completed_sequence",
            "acknowledgement_sequence",
            "ack_edge0_sequence",
            "ack_edge1_sequence",
        ),
        transition_rules=(
            "idle → claimed:   claim_slot(token) sets command_token, owner=gpu_producer, reserved=True",
            "claimed → published: publish(slot, token) sets producer_sequence=token",
            "published → consumed: consume(slot, token) sets consumer_sequence=token, owner=cpu_consumer",
            "consumed → acked: ack_edge0/ack_edge1(slot, token) sets per-edge acknowledgement_sequence — peer doorbell event DURING GPU execution (before complete)",
            "consumed/acked → completed: complete(slot, token) requires BOTH ack_edge0 AND ack_edge1, then sets completed_sequence, clears reserved — releases capacity (both modes)",
            "ack() is now an explicit error — scalar ack must never stand in for both native peer edges (Goal 10)",
            "completed → idle: complete() releases capacity; slot immediately claimable without reclaim()",
            "[doorbell] ack_edge0/ack_edge1 are per-edge DoorbellControl.acknowledgement_sequence, NOT Tp4GraphCommandRing lifecycle",
            "Goal 11: only COMPLETED is reclaimable during normal reuse. acked is NOT reclaimable — must complete() first or teardown_all().",
            "Goal 11: teardown_all() forces all non-idle slots to torndown — explicit failure path, not success.",
            "published/consumed → FATAL: check_timeout(slot) raises SlotOverflowError — capacity unavailable until teardown",
            "claimed → timeout: check_timeout(slot) in non-graph mode enters 'timeout' phase (reclaimable)",
        ),
        ordering_rules=(
            "1. A slot must be claimed before publish.",
            "2. A slot must be published before consume.",
            "3. A slot must be consumed before ack_edge0/ack_edge1 or complete.",
            "4. ack_edge0/ack_edge1 may be called from consumed phase (peer ack during GPU execution, before complete).",
            "5. complete() requires BOTH ack_edge0 AND ack_edge1 before releasing capacity (Goal 10 requirement 4).",
            "6. One token per lifecycle: publish/consume/complete/ack_edge0/ack_edge1 all receive the command_token set at claim.",
            "7. Graph mode: exact token equality required; mismatched token = error.",
            "8. Non-graph mode: also requires token equality (one token per lifecycle).",
            "9. Contiguous tokens: claim_slot(token) requires token == last_token + 1 (globally).",
            "10. Deterministic slot mapping: slot = (token - 1) % num_slots. No round-robin.",
            "11. Per-slot ABA: command_token must also strictly exceed last_token on the same slot.",
            "12. Backpressure: if the deterministic slot is not reusable, claim blocks.",
            "13. Graph-mode timeout: fatal (worker termination), not reclaimable.",
            "14. Non-graph timeout: published/consumed timeout is FATAL (capacity unavailable). Only claimed (pre-publication) timeout is reclaimable.",
            "15. Partial failure: rollback applies ONLY to claimed (pre-publication) slots. Published/consumed slots CANNOT be rolled back — their failure is process-fatal.",
            "16. Graph-capture: slots replayed in order — no skipping.",
            "17. ack_edge0/ack_edge1 are doorbell-level (DoorbellControl.acknowledgement_sequence), received during GPU execution before complete().",
            "18. No post-completion ack barrier — ack_edge0/ack_edge1 before complete is the native ordering.",
        ),
        backpressure_rule=(
            "claim_slot(token) maps to slot (token-1)%num_slots.  If that "
            "slot is not reusable (outstanding or reserved), "
            "BackpressureError is raised.  The ring never picks "
            "a different free slot — the caller must progress the mapped "
            "slot before claiming the next token."
        ),
        wraparound_rule=(
            "Command tokens are uint64 (native tp4_graph_command.hpp).  "
            "The doorbell token packs sequence and Q: token = (seq << 10) | q.  "
            "Maximum sequence is 2^64 >> 10.  Tokens must be contiguous "
            "(last+1); at the modulus boundary, no wraparound is possible "
            "— the next token is out of range and the ring cannot accept "
            "further claims.  Native relies on uint64 never wrapping in "
            "practice."
        ),
        aba_protection_rule=(
            "Each slot records last_token (uint64, sentinel -1).  "
            "claim_slot(token) rejects token <= last_token for that "
            "slot.  With contiguous tokens, reuse of any old token is "
            "caught first by the contiguity check (expected last+1).  "
            "Per-slot ABA provides defense-in-depth for the case where "
            "contiguity is satisfied but the token was previously used "
            "on the same slot."
        ),
        timeout_rule=(
            f"A slot in claimed/published/consumed/acked phase for more than "
            f"{timeout_seconds}s without completion enters timeout.  "
            f"In graph mode, timeout is FATAL (worker termination) — "
            f"the slot cannot be reclaimed.  In non-graph mode, "
            f"published/consumed/acked timeout is also FATAL "
            f"(SlotOverflowError) — capacity remains unavailable until "
            f"teardown.  Goal 10: a one-edge-acked slot must timeout "
            f"fatally after publication; it cannot silently evade timeout.  "
            f"Only claimed (pre-publication) slots enter the "
            f"'timeout' phase and are reclaimable."
        ),
        failure_rule=(
            "Native failure after the CUDA stream has acquired an "
            "unfulfillable wait is fatal — the vLLM adapter terminates "
            "the worker.  In this model, graph-mode overflow sets a "
            "fatal flag (is_fatal()) and all subsequent operations raise "
            "SlotOverflowError."
        ),
        graph_capture_rule=(
            "In graph mode, slots are replayed in the exact order they "
            "were claimed.  publish, consume, and complete must each "
            "operate on the oldest slot in the corresponding FIFO.  "
            "Skipping a slot is rejected (no skipping)."
        ),
        rollback_rule=(
            "rollback(slot) resets a slot in 'claimed' (pre-publication) "
            "phase to idle (owner='none', reserved=False, sequences=0).  "
            "rollback_all() applies this to every qualifying slot.  "
            "Slots in 'published', 'consumed', 'completed', 'acked', "
            "or 'idle' are NOT touched by rollback — published/consumed "
            "failures are process-fatal and capacity remains unavailable "
            "until teardown."
        ),
        capacity_release_rule=(
            "Both graph and non-graph: complete() clears reserved, "
            "making the slot available for new claims immediately.  "
            "Native: claimed_sequence - completed_sequence < "
            "kTp4GraphCommandCapacity.  The completed_sequence is "
            "published by the CPU progress thread after the GPU kernel "
            "finishes.  Peer doorbell acks (acknowledgement_sequence on "
            "DoorbellControl) are per-edge events during GPU execution; "
            "they do not govern ring capacity.  get_outstanding() "
            "counts only slots in claimed/published/consumed phases."
        ),
        sequence_equality_rule=(
            "One uint64 command_token is assigned at claim time and "
            "shared through publish/consume/complete/ack_edge0/ack_edge1 "
            "— it is the same value at every lifecycle step.  "
            "publish(slot, seq) requires seq == slot.command_token.  "
            "A mismatched sequence is rejected (stale if below, "
            "overflow if above in graph mode)."
        ),
        reservation_rule=(
            "reserved=True means the slot is not available for a new "
            "claim.  reserved is set on claim.  Both modes: cleared on "
            "complete (releasing capacity, matching native "
            "completed_sequence).  Both modes: cleared on reclaim.  "
            "Published/consumed timed-out slots remain reserved (fatal) "
            "until teardown.  Only claimed timed-out slots can be "
            "reclaimed."
        ),
        counter_descriptions=(
            "get_outstanding(): count of claimed-but-not-completed slots",
            "get_events(): chronological list of all SlotEvent records",
            "is_fatal(): True after graph-mode overflow",
            "command_token: uint64 lifecycle token, globally unique and monotonically advancing across all slots",
            "_global_last_token: highest token ever claimed on any slot (global monotonicity)",
        ),
    )


# ---------------------------------------------------------------------------
# Integration helper — connects the state machine to FP32 ground truth
# ---------------------------------------------------------------------------

def simulate_allreduce_with_truth(
    inputs: Sequence[object],
    ring: MultiSlotRing | None = None,
) -> dict[str, object]:
    """Demonstrate the state machine alongside FP32 ground truth.

    This is a **Modeled** integration helper.  It drives the ring
    through a claim→publish→consume→complete cycle for each rank
    input and compares the FP32 ring-reduction truth with the
    sequential BF16 sum.  It does NOT prove native correctness.

    The command-ring lifecycle is claim→publish→consume→complete.
    ``ack_edge0()``/``ack_edge1()`` are doorbell-level per-edge peer
    acknowledgements (not part of the command-ring lifecycle) and are
    not called here.

    Parameters
    ----------
    inputs
        Sequence of tensors (one per rank).
    ring
        Optional pre-configured ring.  If None, a default 4-slot
        non-graph ring is created.
    """
    # Imported here to avoid a hard dependency at module load time
    # and to keep the state-machine module importable without torch.
    from spark_fp32_ground_truth import (
        fp32_ground_truth,
        fp32_truth_rounded_to_dtype,
        sequential_bf16_sum,
        tp4_ring_reduce_all_ranks,
    )

    if ring is None:
        ring = MultiSlotRing(
            num_slots=max(len(inputs), 4),
            graph_mode=False,
        )
    results: dict[str, object] = {}
    events: list[SlotEvent] = []
    for rank_idx in range(len(inputs)):
        token = rank_idx + 1  # one token per lifecycle, monotonically advancing
        slot = ring.claim_slot(token)
        events.append(ring.publish(slot, token))
        events.append(ring.consume(slot, token))
        # Goal 10: both per-edge acks required before complete().
        events.append(ring.ack_edge0(slot, token))
        events.append(ring.ack_edge1(slot, token))
        events.append(ring.complete(slot, token))
        # complete() already released capacity; reclaim() is optional.

    # Numerical truth comparison (Modeled — not native proof)
    fp32_truth = fp32_ground_truth(inputs)
    bf16_truth = fp32_truth_rounded_to_dtype(fp32_truth, inputs[0].dtype)
    ring_result = tp4_ring_reduce_all_ranks(inputs)
    sequential = sequential_bf16_sum(inputs)
    results["bf16_truth_rounded"] = bf16_truth
    results["tp4_ring_result"] = ring_result
    results["sequential_bf16"] = sequential
    results["ring_events"] = events
    results["ring_outstanding"] = ring.get_outstanding()
    results["evidence_label"] = "Modeled"

    return results
