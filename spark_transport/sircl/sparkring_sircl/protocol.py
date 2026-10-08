"""Wire protocol of SIRCL ring sessions: constants and shared arithmetic.

A ring session's GPU kernels, its native progress thread
(``oneshot/_roce_proxy.c``) and kernels outside the transport that use a
session's arena all compute the same quantities. This module is their
reference in Python, with no torch or CUDA dependency:

- arena geometry: ``SLOTS`` transport slots (the slot of op ``seq`` is
  ``seq & 1``), flag lines of ``FLAG_STRIDE`` bytes, ``FLAG_LINES`` per
  (source, slot): two namespaces of up to two lanes;
- the device command ring (:class:`Ctrl`): doorbell, byte count, timeout
  words, per-slot op words, phase doorbells, phase descriptors;
- op words (op code in bits 30-31, byte count below) and phase descriptors;
- the stripe split of a payload over a peer's lanes, the chunk split of a
  message over the group, and the flag-line index;
- posting orders, the Swing schedule and the all-reduce algorithm choice;
- the launch-grid rule and the device counter array.

Every function raises :class:`ProtocolError` for values the wire format
cannot carry, so a caller fails before a kernel launch. Status: implemented;
checked against the reference vectors of the tests (``tests/data/numeric.json``).
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence

PACK_BYTES = 16
SLOTS = 2
FLAG_STRIDE = 128
FLAG_LINES = 4
MAX_LANES = 2
MAX_DEVICES = 4
MAX_PHASES = 8
SLOT_ALIGNMENT = 4096
MAX_WORLD = 16
OP_SHIFT = 30
OP_BYTES_MASK = (1 << OP_SHIFT) - 1
CTRL_WORDS = FLAG_STRIDE // 4
ALGORITHMS = ("oneshot", "twoshot", "swing")
ALGORITHM_CHOICES = ("auto",) + ALGORITHMS
LARGE_ALGORITHMS = ("twoshot", "swing")
DEFAULT_PACKS_PER_THREAD = 2


class ProtocolError(ValueError):
    """A value the wire protocol cannot represent."""


class Op(enum.IntEnum):
    """Op codes of the per-slot op word."""

    ONESHOT = 0      # one network phase to every peer; also the all-gather
    TWOSHOT = 1      # chunk scatter, then gather of the reduced own chunk
    DESCRIBED = 2    # phases described by descriptors (the Swing all-reduce)
    SCATTER = 3      # the two-shot scatter phase only (reduce-scatter, all-to-all)


class Ctrl(enum.IntEnum):
    """32-bit words of the control area."""

    DOORBELL = 0       # newest op whose first network phase the kernel released
    NBYTES = 1         # byte count of that op
    ERROR_SEQ = 2      # sequence whose flag wait timed out (0: none)
    MISSING_PEER = 3   # the peer whose flag did not arrive
    OP_WORD = 4        # op word of slot 0; slot 1 is word 5
    MISSING_LANE = 6   # the lane whose flag did not arrive
    WAIT_LIMIT_US = 7  # flag-wait limit in microseconds, written by the host, read by kernels at launch
    ERROR_KIND = 8     # what a timed-out wait waited for (ErrorKind), written with words 2, 3 and 6
    PHASE = 9          # word 9 + k: phase-k doorbell, k = 1..7
    DESC = 17          # word 17 + k: phase-k descriptor, k = 0..7
    ECHO_SEEN = 28     # trace echo words (written only with tracing)
    ECHO_POSTED = 29
    LANE_CHECK = 31    # written by peers during the setup lane check


class ErrorKind(enum.IntEnum):
    """What the wait recorded in command ring words 2, 3 and 6 waited for."""

    SLOT_OP = 0        # a peer's lane flag of a slot op; word 2 holds the sequence
    CHAIN_CHUNK = 1    # a chain chunk from a neighbor; word 2 holds the chunk tag
    CHAIN_SLOT = 2     # a free chain send slot (the own progress thread); word 2 holds the chunk tag


def phase_doorbell_word(phase: int) -> int:
    if not 1 <= phase < MAX_PHASES:
        raise ProtocolError(f"phase doorbells exist for phases 1-{MAX_PHASES - 1}, got {phase}")
    return int(Ctrl.PHASE) + phase


def descriptor_word(phase: int) -> int:
    if not 0 <= phase < MAX_PHASES:
        raise ProtocolError(f"descriptors exist for phases 0-{MAX_PHASES - 1}, got {phase}")
    return int(Ctrl.DESC) + phase


# -- op words and descriptors ---------------------------------------------------


def op_word(op: int, nbytes: int) -> int:
    """Op word of a message: op code above bit 30, byte count below."""
    op = Op(op)
    if type(nbytes) is not int or not 0 < nbytes <= OP_BYTES_MASK or nbytes % PACK_BYTES:
        raise ProtocolError(
            f"an op word carries a positive multiple of {PACK_BYTES} bytes below 2^{OP_SHIFT}, "
            f"got {nbytes}"
        )
    return (int(op) << OP_SHIFT) | nbytes


def decode_op_word(word: int) -> tuple[Op, int]:
    if not 0 <= word < 1 << 32:
        raise ProtocolError(f"op word {word} is not a 32-bit value")
    return Op(word >> OP_SHIFT), word & OP_BYTES_MASK


@dataclasses.dataclass(frozen=True)
class Descriptor:
    """One phase of a described op: chunk positions [first, end) to ``peer``."""

    first: int
    end: int
    peer: int
    namespace: int

    def word(self, world: int | None = None) -> int:
        if not (0 <= self.first <= self.end <= 31 and 0 <= self.peer <= 15
                and self.namespace in (0, 1)):
            raise ProtocolError(f"descriptor fields out of range: {self}")
        if world is not None and (self.end > world or self.peer >= world):
            raise ProtocolError(f"descriptor {self} does not fit a group of {world}")
        return (1 << 31) | (self.namespace << 14) | (self.peer << 10) | (self.end << 5) | self.first


def descriptor(first: int, end: int, peer: int, namespace: int) -> int:
    return Descriptor(first, end, peer, namespace).word()


def decode_descriptor(word: int) -> Descriptor | None:
    """The descriptor in ``word``, or None when its valid bit is clear."""
    if not word >> 31 & 1:
        return None
    return Descriptor(word & 31, word >> 5 & 31, word >> 10 & 15, word >> 14 & 1)


# -- stripes, chunks, flag lines --------------------------------------------------


def stripe(count: int, lanes: int, lane: int) -> tuple[int, int]:
    """``(first pack, packs)`` of lane ``lane`` when ``count`` packs go over ``lanes`` lanes."""
    if not 1 <= lanes <= MAX_LANES or not 0 <= lane < lanes or count < 0:
        raise ProtocolError(f"stripe of {count} packs, lane {lane} of {lanes}")
    base, rest = divmod(count, lanes)
    return lane * base + min(lane, rest), base + (1 if lane < rest else 0)


def stripes(count: int, lanes: int) -> tuple[tuple[int, int], ...]:
    return tuple(stripe(count, lanes, lane) for lane in range(lanes))


def chunk(packs: int, world: int, index: int) -> tuple[int, int]:
    """``(first pack, packs)`` of chunk ``index``: ``[floor(j*P/W), floor((j+1)*P/W))``."""
    if not 2 <= world <= MAX_WORLD or not 0 <= index < world or packs < 0:
        raise ProtocolError(f"chunk {index} of {packs} packs over {world} ranks")
    lo = index * packs // world
    hi = (index + 1) * packs // world
    return lo, hi - lo


def chunks(packs: int, world: int) -> tuple[tuple[int, int], ...]:
    return tuple(chunk(packs, world, index) for index in range(world))


def flag_index(namespace: int, source: int, slot: int, lane: int, world: int, lanes: int) -> int:
    """Flag line of (namespace, source, slot, lane): ``ns*W*SLOTS*L + (source*SLOTS + slot)*L + lane``."""
    if (namespace not in (0, 1) or not 0 <= source < world or slot not in (0, 1)
            or not 1 <= lanes <= MAX_LANES or not 0 <= lane < MAX_LANES):
        raise ProtocolError(f"flag line ns={namespace} source={source} slot={slot} lane={lane}")
    return namespace * world * SLOTS * lanes + (source * SLOTS + slot) * lanes + lane


def slot_of(seq: int) -> int:
    return seq & (SLOTS - 1)


# -- arena geometry ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArenaLayout:
    """Byte offsets of one rank's arena (recv, flag lines, send, control)."""

    world: int
    slot_bytes: int

    def __post_init__(self) -> None:
        if not 2 <= self.world <= MAX_WORLD:
            raise ProtocolError(f"a session has 2 to {MAX_WORLD} ranks, got {self.world}")
        if (type(self.slot_bytes) is not int or self.slot_bytes <= 0
                or self.slot_bytes % SLOT_ALIGNMENT or self.slot_bytes > 1 << 40):
            raise ProtocolError(
                f"slot bytes must be a positive multiple of {SLOT_ALIGNMENT} up to 2^40, "
                f"got {self.slot_bytes}"
            )

    @property
    def recv_off(self) -> int:
        return 0

    @property
    def flag_off(self) -> int:
        return self.world * SLOTS * self.slot_bytes

    @property
    def send_off(self) -> int:
        return self.flag_off + self.world * SLOTS * FLAG_LINES * FLAG_STRIDE

    @property
    def ctrl_off(self) -> int:
        return self.send_off + SLOTS * self.slot_bytes

    @property
    def total_bytes(self) -> int:
        return self.ctrl_off + FLAG_STRIDE

    def as_tuple(self) -> tuple[int, ...]:
        """The values the native layout function reports, in its order."""
        return (self.recv_off, self.flag_off, self.send_off, self.ctrl_off, self.total_bytes,
                FLAG_STRIDE, SLOTS)


def slot_bytes_for(capacity: int, gather_capacity: int) -> int:
    """Slot size: the larger capacity rounded up to 4,096 bytes."""
    largest = max(int(capacity), int(gather_capacity))
    return (largest + SLOT_ALIGNMENT - 1) // SLOT_ALIGNMENT * SLOT_ALIGNMENT


def multi_phase_available(lane_count: int, slot_bytes: int) -> bool:
    """The progress thread can post multi-phase and scatter ops in this geometry."""
    return 2 * lane_count <= FLAG_LINES and slot_bytes <= OP_BYTES_MASK


# -- chain schedule ----------------------------------------------------------------------

CHAIN_STREAMS = 4
CHAIN_MAX_SLOTS = 32


class ChainStream(enum.IntEnum):
    """Streams of the chain schedule: (half, direction) between chain neighbors."""

    A_REDUCE = 0     # half A partials toward the next rank
    A_RESULT = 1     # half A results toward the previous rank
    B_REDUCE = 2     # half B partials toward the previous rank
    B_RESULT = 3     # half B results toward the next rank


@dataclasses.dataclass(frozen=True)
class ChainLayout:
    """Byte offsets of the chain area (relative to its start) and its size.

    Per stream: ``slots`` receive slots and ``slots`` send slots of
    ``slot_bytes``, a flag line per (slot, lane), a line of ready words and a
    line of consumed words (one per slot), a sent word and a credit word (each
    on its own line); then the control line: the chain doorbell (word 0) and
    two parameter slots (words ``1 + 4 p``: bytes of half A, bytes of half B,
    chunk bytes).
    """

    lanes: int
    slots: int
    slot_bytes: int

    def __post_init__(self) -> None:
        if (not 1 <= self.lanes <= MAX_LANES or not 2 <= self.slots <= CHAIN_MAX_SLOTS
                or type(self.slot_bytes) is not int or self.slot_bytes <= 0
                or self.slot_bytes % SLOT_ALIGNMENT or self.slot_bytes > 1 << 31):
            raise ProtocolError(f"chain geometry: {self.lanes} lanes, {self.slots} slots of {self.slot_bytes} "
                                f"bytes (2 to {CHAIN_MAX_SLOTS} slots, multiples of {SLOT_ALIGNMENT} bytes)")

    @property
    def ring_bytes(self) -> int:
        return CHAIN_STREAMS * self.slots * self.slot_bytes

    @property
    def recv_off(self) -> int:
        return 0

    @property
    def send_off(self) -> int:
        return self.ring_bytes

    @property
    def rflag_off(self) -> int:
        return 2 * self.ring_bytes

    @property
    def ready_off(self) -> int:
        return self.rflag_off + CHAIN_STREAMS * self.slots * self.lanes * FLAG_STRIDE

    @property
    def consumed_off(self) -> int:
        return self.ready_off + CHAIN_STREAMS * FLAG_STRIDE

    @property
    def sent_off(self) -> int:
        return self.consumed_off + CHAIN_STREAMS * FLAG_STRIDE

    @property
    def credit_off(self) -> int:
        return self.sent_off + CHAIN_STREAMS * FLAG_STRIDE

    @property
    def ctrl_off(self) -> int:
        return self.credit_off + CHAIN_STREAMS * FLAG_STRIDE

    @property
    def total_bytes(self) -> int:
        return self.ctrl_off + FLAG_STRIDE

    def as_tuple(self) -> tuple[int, ...]:
        """The values the native chain layout function reports, in its order."""
        return (self.recv_off, self.send_off, self.rflag_off, self.ready_off, self.consumed_off, self.sent_off,
                self.credit_off, self.ctrl_off, self.total_bytes)

    def recv_slot(self, stream: int, slot: int) -> int:
        return self.recv_off + (stream * self.slots + slot) * self.slot_bytes

    def send_slot(self, stream: int, slot: int) -> int:
        return self.send_off + (stream * self.slots + slot) * self.slot_bytes

    def flag_line(self, stream: int, slot: int, lane: int) -> int:
        return self.rflag_off + ((stream * self.slots + slot) * self.lanes + lane) * FLAG_STRIDE

    def ready_word(self, stream: int, slot: int) -> int:
        return self.ready_off + stream * FLAG_STRIDE + 4 * slot

    def consumed_word(self, stream: int, slot: int) -> int:
        return self.consumed_off + stream * FLAG_STRIDE + 4 * slot

    def sent_word(self, stream: int) -> int:
        return self.sent_off + stream * FLAG_STRIDE

    def credit_word(self, stream: int) -> int:
        return self.credit_off + stream * FLAG_STRIDE

    def param_word(self, slot: int) -> int:
        """First of the three parameter words of chain op parameter slot ``slot`` (seq & 1)."""
        return self.ctrl_off + 4 * (1 + 4 * slot)


def chain_offset(arena_total_bytes: int) -> int:
    """Offset of the chain area in an arena: after the control line, on a 4,096-byte boundary."""
    return -(-int(arena_total_bytes) // SLOT_ALIGNMENT) * SLOT_ALIGNMENT


def chain_halves(packs: int) -> tuple[int, int]:
    """``(packs of half A, packs of half B)`` of a chain op of ``packs`` 16-byte packs."""
    if packs < 0:
        raise ProtocolError(f"a message has a non-negative pack count, got {packs}")
    return packs // 2, packs - packs // 2


def chain_chunks(packs: int, chunk_packs: int) -> int:
    """Chunks of a half of ``packs`` packs in chunks of ``chunk_packs``."""
    if chunk_packs <= 0:
        raise ProtocolError(f"chain chunks hold at least one pack, got {chunk_packs}")
    return -(-packs // chunk_packs)


# -- chain links: the chain and ring collectives ---------------------------------------------

# Link 0 carries items toward the next rank in chain order and link 1 toward the previous
# one; links 2 (ring partial sums) and 3 (ring results) toward the next rank of the ring that
# closes the chain.
LINKS = 4
LINK_MAX_SLOTS = 32
# The fewest link slots a session takes when neither SIRCL_LINK_SLOTS nor its tuning table sets them
# (default_link_slots).
MIN_DEFAULT_LINK_SLOTS = 8
# A ring link through relays posts each lane's stripe in chunks of this many bytes (the native
# ROCE_LINK_WINDOW_CHUNK); its window is a whole number of them.
LINK_WINDOW_CHUNK = 32768


class TraceEvent(enum.IntEnum):
    """Events of the event trace (``SIRCL_EVENT_TRACE``). The native progress thread records
    1-7 (``_roce_proxy.c``), the chain kernel 16-18, the ring kernels 16-21; a record's stream is a
    chain stream (0-3, :class:`ChainStream`) or ``TRACE_LINK_STREAM + link``, its value a chunk or
    item tag (index on its stream plus one), a credit, an op sequence, or (``KERNEL_START``) a block
    index."""

    OP = 1              # an op taken from its doorbell (stream 0: chain op, TRACE_LINK_STREAM: link op)
    READY = 2           # the progress thread first saw an outbound chunk or item ready
    POSTED = 3          # its writes posted on every lane
    DONE = 4            # its writes completed on every lane
    CONSUMED = 5        # the progress thread saw the kernel finish an inbound chunk or item
    CREDIT_OUT = 6      # credit written upstream
    CREDIT_IN = 7       # the credit word from downstream changed
    KERNEL_FLAG = 16    # the kernel saw every lane flag of an inbound chunk
    KERNEL_READY = 17   # the kernel published an outbound chunk ready (staged)
    KERNEL_CONSUMED = 18  # the kernel published an inbound chunk consumed
    KERNEL_SLOT = 19    # a ring kernel block had its outbound item's own slot free and began staging it
    KERNEL_START = 20   # a ring kernel block began (stream TRACE_LINK_STREAM, value: the block's index)
    KERNEL_BELL = 21    # the ring kernel's block 0 rang the link doorbell (stream TRACE_LINK_STREAM, value: op)


TRACE_LINK_STREAM = 4


class LinkOp(enum.IntEnum):
    """Collectives that run over the links (word 0 of a link op's parameter slot)."""

    ALL_GATHER = 1          # chain all-gather (links 0 and 1)
    REDUCE_SCATTER = 2      # chain reduce-scatter (links 0 and 1)
    RING_GATHER = 3         # ring all-gather (link 3)
    RING_SCATTER = 4        # ring reduce-scatter (link 2)
    RING_REDUCE = 5         # ring all-reduce (links 2 and 3)


@dataclasses.dataclass(frozen=True)
class LinkLayout:
    """Byte offsets of the link area (relative to its start) and its size.

    Per link: ``slots`` receive slots written by the upstream neighbor and
    ``slots`` own slots staged by the kernel, of ``slot_bytes`` each, a flag line
    per (receive slot, lane), a line of ready words (one per own slot), a line
    of consumed words (one per receive slot), a sent word and a credit word
    (each on its own line); then the control line: the link doorbell (word 0)
    and two parameter slots (words ``1 + 4 p``: op, bytes per rank, piece bytes).
    """

    lanes: int
    slots: int
    slot_bytes: int

    def __post_init__(self) -> None:
        if (not 1 <= self.lanes <= MAX_LANES or not 2 <= self.slots <= LINK_MAX_SLOTS
                or type(self.slot_bytes) is not int or self.slot_bytes <= 0
                or self.slot_bytes % SLOT_ALIGNMENT or self.slot_bytes > 1 << 31):
            raise ProtocolError(f"link geometry: {self.lanes} lanes, {self.slots} slots of {self.slot_bytes} "
                                f"bytes (2 to {LINK_MAX_SLOTS} slots, multiples of {SLOT_ALIGNMENT} bytes)")

    @property
    def ring_bytes(self) -> int:
        return LINKS * self.slots * self.slot_bytes

    @property
    def recv_off(self) -> int:
        return 0

    @property
    def own_off(self) -> int:
        return self.ring_bytes

    @property
    def rflag_off(self) -> int:
        return 2 * self.ring_bytes

    @property
    def ready_off(self) -> int:
        return self.rflag_off + LINKS * self.slots * self.lanes * FLAG_STRIDE

    @property
    def consumed_off(self) -> int:
        return self.ready_off + LINKS * FLAG_STRIDE

    @property
    def sent_off(self) -> int:
        return self.consumed_off + LINKS * FLAG_STRIDE

    @property
    def credit_off(self) -> int:
        return self.sent_off + LINKS * FLAG_STRIDE

    @property
    def ctrl_off(self) -> int:
        return self.credit_off + LINKS * FLAG_STRIDE

    @property
    def total_bytes(self) -> int:
        return self.ctrl_off + FLAG_STRIDE

    def as_tuple(self) -> tuple[int, ...]:
        """The values the native link layout function reports, in its order."""
        return (self.recv_off, self.own_off, self.rflag_off, self.ready_off, self.consumed_off, self.sent_off,
                self.credit_off, self.ctrl_off, self.total_bytes)

    def recv_slot(self, link: int, slot: int) -> int:
        return self.recv_off + (link * self.slots + slot) * self.slot_bytes

    def own_slot(self, link: int, slot: int) -> int:
        return self.own_off + (link * self.slots + slot) * self.slot_bytes

    def flag_line(self, link: int, slot: int, lane: int) -> int:
        return self.rflag_off + ((link * self.slots + slot) * self.lanes + lane) * FLAG_STRIDE

    def ready_word(self, link: int, slot: int) -> int:
        return self.ready_off + link * FLAG_STRIDE + 4 * slot

    def consumed_word(self, link: int, slot: int) -> int:
        return self.consumed_off + link * FLAG_STRIDE + 4 * slot

    def sent_word(self, link: int) -> int:
        return self.sent_off + link * FLAG_STRIDE

    def credit_word(self, link: int) -> int:
        return self.credit_off + link * FLAG_STRIDE

    def param_word(self, slot: int) -> int:
        """First of the three parameter words of link op parameter slot ``slot`` (seq & 1)."""
        return self.ctrl_off + 4 * (1 + 4 * slot)


@dataclasses.dataclass(frozen=True)
class LinkRounds:
    """Items per round of one link at one chain index (a round carries piece ``p`` of each owner).

    Outbound round: ``own`` items staged by the kernel, then ``out - own``
    items forwarded from the inbound round in its order: the first
    ``out - own`` inbound items of a round are also written downstream.
    """

    out: int
    inbound: int
    own: int

    @property
    def forwards(self) -> bool:
        return self.out > self.own

    def forwarded(self, r: int) -> bool:
        """Whether inbound item ``r`` of a round is forwarded."""
        return r < self.out - self.own


def link_rounds(op: int, world: int, index: int, link: int) -> LinkRounds:
    """Round sizes of ``link`` at chain index ``index`` for a link op.

    All-gather, link 0: rank ``j`` sends its own piece and then the pieces of
    owners ``j - 1`` down to 0 it received; link 1 mirrors it (own, then owners
    ``j + 1`` up to ``W - 1``). Reduce-scatter, link 0: rank ``j`` sends the
    partial sums for owners ``W - 1`` down to ``j + 1`` (farthest first); link 1
    the partial sums for owners 0 up to ``j - 1``. The last rank of a link's
    direction sends nothing on it, and its first rank receives nothing. Ring
    ops: link 2 (ring reduce-scatter and all-reduce) sends and receives
    ``W - 1`` partial sums per round, every one staged by the kernel; link 3
    (ring all-gather and all-reduce) sends the own piece and forwards the
    first ``W - 2`` of the ``W - 1`` pieces it receives.
    """
    if world < 2 or not 0 <= index < world or link not in range(LINKS):
        raise ProtocolError(f"link {link} at chain index {index} of a chain of {world}")
    if op in (LinkOp.RING_GATHER, LinkOp.RING_SCATTER, LinkOp.RING_REDUCE):
        partials = op in (LinkOp.RING_SCATTER, LinkOp.RING_REDUCE)
        results = op in (LinkOp.RING_GATHER, LinkOp.RING_REDUCE)
        if (link == 2 and partials) or (link == 3 and results):
            return LinkRounds(out=world - 1, inbound=world - 1, own=world - 1 if link == 2 else 1)
        return LinkRounds(out=0, inbound=0, own=0)
    if op in (LinkOp.ALL_GATHER, LinkOp.REDUCE_SCATTER) and link >= 2:
        return LinkRounds(out=0, inbound=0, own=0)
    j = index if link == 0 else world - 1 - index      # distance from the link's first rank
    sends = j < world - 1
    if op == LinkOp.ALL_GATHER:
        return LinkRounds(out=j + 1 if sends else 0, inbound=j, own=1 if sends else 0)
    if op == LinkOp.REDUCE_SCATTER:
        out = world - 1 - j if sends else 0
        return LinkRounds(out=out, inbound=world - j if j > 0 else 0, own=out)
    raise ProtocolError(f"unknown link op {op}")


# The ring reduce-scatter's stagger D (link 2) travels in bits 8-15 of its link op word, the ring
# all-gather's stagger D3 (link 3) in bits 16-23; bits 24-31 are zero.
RING_STAGGER_SHIFT = 8
RING_GATHER_STAGGER_SHIFT = 16
MAX_RING_STAGGER = 4


def default_link_slots(world: int) -> int:
    """Link slots of a session of ``world`` ranks when neither ``SIRCL_LINK_SLOTS`` nor its tuning table sets
    them: two rounds of a ring link's items in flight, ``2 W``, at least :data:`MIN_DEFAULT_LINK_SLOTS` and at
    most :data:`LINK_MAX_SLOTS` (16 on the cycle of eight, 8 on a path of four or a pair). On the cycle of
    eight, ring all-gathers in slots of 512 KiB took 788 and 1,399 us at 2 and 4 MiB shards with 12 slots,
    696 and 1,305 us with 16 and 693 and 1,302 us with 24 (ring harness, eager p50, staggers 1)."""
    return min(LINK_MAX_SLOTS, max(MIN_DEFAULT_LINK_SLOTS, 2 * int(world)))


# The link kernels, each running one collective under one schedule, and that collective (gather, scatter,
# reduce). The chain all-reduce is the chain kernel (SIRCL_CHAIN_BLOCKS), not a link kernel.
LINK_BLOCK_KERNELS: dict[str, str] = {"ring_reduce": "reduce", "ring_gather": "gather", "ring_scatter": "scatter",
                                      "chain_gather": "gather", "chain_scatter": "scatter"}
DEFAULT_LINK_BLOCKS = 4
# Blocks per role by group shape (tuning.shape_of) and link kernel, where measured: a role's blocks share the
# GPU's path to pinned host memory, so fewer blocks finish each item sooner without lowering the op's rate,
# and the launch's block 0, which rings the link doorbell, starts its own items sooner. Ring harness, eager
# periods at 4 / 2 / 1 blocks per role:
# - pair (two DGX Sparks cabled port to port): the ring all-reduce in 256 KiB pieces 245.8 / 221.3 / 212.0 us
#   at 4 MiB, 419.4 / 399.4 / 387.8 us at 8 MiB and 2,857.5 / 2,823.9 / 2,812.5 us at 64 MiB; the ring
#   all-gather of 16 MiB shards in 512 KiB pieces 801.8 / 771.8 / 814.1 us with two kernel passes per own
#   piece, 757.7 us at 1 block with one; the ring reduce-scatter in 256 KiB pieces within 2-4 % of each other
#   at 4 / 2 / 1 blocks (8-64 MiB inputs);
# - path:4 (Sparks 0-3, the ring closed through two relays), 512 KiB pieces: the ring all-reduce 395.1 /
#   375.9 / 366.8 us at 4 MiB, 1,379.4 / 1,356.3 / 1,339.9 us at 16 MiB and 5,163.5 / 5,185.1 / 5,184.0 us
#   at 64 MiB; the ring all-gather 231.0 / 214.2 / 205.1 us for 1 MiB shards and 2,622.5 / 2,597.3 /
#   2,586.9 us for 16 MiB shards.
# Not measured, so DEFAULT_LINK_BLOCKS: the chain all-gather and reduce-scatter on a pair, the ring
# reduce-scatter and the chain all-gather and reduce-scatter on a path of four, every kernel on other paths,
# cycles and strided groups.
LINK_BLOCKS_BY_SHAPE: dict[str, dict[str, int]] = {"pair": {"ring_reduce": 1, "ring_gather": 1, "ring_scatter": 1},
                                                   "path:4": {"ring_reduce": 1, "ring_gather": 1}}


def link_blocks(shape: str | None, world: int, overall: int = 0, own: Mapping[str, int] | None = None) -> dict[str, int]:
    """Blocks per role of each link kernel (:data:`LINK_BLOCK_KERNELS`): its collective's own value in ``own``
    (``gather``, ``scatter``, ``reduce`` from ``SIRCL_GATHER_LINK_BLOCKS``, ``SIRCL_SCATTER_LINK_BLOCKS``,
    ``SIRCL_REDUCE_LINK_BLOCKS``), else ``overall`` (``SIRCL_LINK_BLOCKS``), else :data:`LINK_BLOCKS_BY_SHAPE`
    for the group shape ``shape`` (``tuning.shape_of``; without one, a world of two is a pair), else
    :data:`DEFAULT_LINK_BLOCKS`. 0 leaves a value unset; every value set is 1 to 64."""
    measured = LINK_BLOCKS_BY_SHAPE.get(shape or ("pair" if int(world) == 2 else ""), {})
    own = dict(own or {})
    for name, value in (("SIRCL_LINK_BLOCKS", overall), *((f"blocks of {c}", v) for c, v in own.items())):
        if value and not 1 <= int(value) <= 64:
            raise ValueError(f"{name} must be 1 to 64, got {value}")
    return {kernel: int(own.get(collective) or overall or measured.get(kernel, DEFAULT_LINK_BLOCKS))
            for kernel, collective in LINK_BLOCK_KERNELS.items()}


def ring_stagger_slots(world: int, stagger: int) -> int:
    """Link slots a staggered ring link needs: on link 2 a relay leaves ``stagger`` rounds
    (``stagger * (world - 1) + 1`` items) after the partial it extends arrived, on link 3 a forward
    leaves as long after the piece it passes on arrived, so the receive and own slots must hold that many
    items and one more."""
    return int(stagger) * (int(world) - 1) + 2 if int(stagger) else 2


def ring_rounds(pieces: int, world: int, stagger: int) -> int:
    """Rounds of a staggered ring link (link 2 with the stagger D, link 3 with D3) in a ring op of
    ``pieces`` pieces per rank."""
    return int(pieces) + (int(world) - 2) * int(stagger)


def ring_op_word(op: int, stagger: int = 0, gather_stagger: int = 0) -> int:
    """The link op word of a ring op: its code, the stagger D of link 2 (ring reduce-scatter and
    all-reduce) and the stagger D3 of link 3 (ring all-gather and all-reduce)."""
    if not (0 <= int(stagger) <= MAX_RING_STAGGER and 0 <= int(gather_stagger) <= MAX_RING_STAGGER):
        raise ProtocolError(f"ring staggers {stagger} and {gather_stagger}: 0 to {MAX_RING_STAGGER} rounds")
    if int(stagger) and op not in (LinkOp.RING_SCATTER, LinkOp.RING_REDUCE):
        raise ProtocolError(f"link op {op} has no partials to stagger")
    if int(gather_stagger) and op not in (LinkOp.RING_GATHER, LinkOp.RING_REDUCE):
        raise ProtocolError(f"link op {op} forwards no finished pieces")
    return int(op) | int(stagger) << RING_STAGGER_SHIFT | int(gather_stagger) << RING_GATHER_STAGGER_SHIFT


def ring_forward_source(item: int, world: int, stagger: int) -> int | None:
    """The op-local inbound item that outbound item ``item`` of link 3 forwards, with the ring
    all-gather's stagger D3 ``stagger``: type ``r >= 1`` of round ``t`` (piece ``t - r D3``, as
    :func:`ring_partial` gives) forwards inbound type ``r - 1`` of round ``t - D3``, which carries the same
    piece. None for the own item (type 0) and for the forwards of the first D3 rounds, which are empty and
    forward nothing."""
    per_round = int(world) - 1
    round_, kind = divmod(int(item), per_round)
    if kind == 0 or round_ < int(stagger):
        return None
    return (round_ - int(stagger)) * per_round + kind - 1


def ring_partial(item: int, world: int, pieces: int, stagger: int) -> tuple[int | None, int]:
    """Piece and type of op-local item ``item`` of a staggered ring link (the same at its sender and its
    receiver). Link 2: type 0 is the partial the sender starts, type ``r + 1`` its relay of the inbound
    partial of type ``r`` (the receiver's last type, ``world - 2``, completes the receiver's own chunk).
    Link 3 with D3: type 0 is the sender's own finished piece, type ``r + 1`` its forward of the inbound
    piece of type ``r``. The piece is None for an empty item."""
    per_round = int(world) - 1
    round_, kind = divmod(int(item), per_round)
    piece = round_ - kind * int(stagger)
    return (piece if 0 <= piece < int(pieces) else None), kind


def link_pieces(nbytes: int, piece_bytes: int) -> int:
    """Pieces of a per-rank block of ``nbytes`` in pieces of ``piece_bytes`` (multiples of 16)."""
    if nbytes < 0 or nbytes % PACK_BYTES or piece_bytes <= 0 or piece_bytes % PACK_BYTES:
        raise ProtocolError(f"link op of {nbytes} bytes per rank in pieces of {piece_bytes} bytes "
                            f"(multiples of {PACK_BYTES})")
    return -(-nbytes // piece_bytes)


def link_piece_bytes(nbytes: int, piece_bytes: int, piece: int) -> int:
    """Bytes of piece ``piece``: ``piece_bytes`` except the last."""
    return min(piece_bytes, nbytes - piece * piece_bytes)


def link_source(rounds: LinkRounds, item: int) -> tuple[str, int]:
    """Source of outbound item ``item`` of an op: ``("own", own index)`` or ``("in", inbound index)``."""
    if rounds.out <= 0:
        raise ProtocolError("this link sends nothing at this rank")
    piece, r = divmod(item, rounds.out)
    if r < rounds.own:
        return "own", piece * rounds.own + r
    return "in", piece * rounds.inbound + r - rounds.own


# -- posting order ---------------------------------------------------------------------


def ring_farthest(rank: int, world: int) -> tuple[int, ...]:
    """Folded distance from ``world // 2`` down to 1: clockwise peer first, then counter-clockwise."""
    order: list[int] = []
    for distance in range(world // 2, 0, -1):
        cw = (rank + distance) % world
        ccw = (rank - distance) % world
        order.append(cw)
        if ccw != cw:
            order.append(ccw)
    return tuple(order)


def post_order(rank: int, world: int, text: str | None) -> tuple[int, ...]:
    """Peers in posting order for ``SIRCL_POST_ORDER`` (rank, ring-farthest or an explicit list)."""
    if text is None or text.strip() in ("", "rank"):
        return tuple(peer for peer in range(world) if peer != rank)
    text = text.strip()
    if text == "ring-farthest":
        return ring_farthest(rank, world)
    try:
        peers = tuple(int(item) for item in text.split(","))
    except ValueError:
        raise ProtocolError(f"SIRCL_POST_ORDER={text} is not rank, ring-farthest or a peer list") from None
    expected = {peer for peer in range(world) if peer != rank}
    if len(peers) != len(expected) or set(peers) != expected:
        raise ProtocolError(f"SIRCL_POST_ORDER={text} must name every peer of rank {rank} exactly once")
    return peers


# -- Swing schedule ------------------------------------------------------------------------


def _steps(world: int) -> int:
    if world < 2 or world > MAX_WORLD or world & (world - 1):
        raise ProtocolError(f"the Swing schedule needs a power-of-two group of 2-16 ranks, got {world}")
    return world.bit_length() - 1


def swing_offset(step: int) -> int:
    """``rho(k) = (1 - (-2)^(k+1)) / 3``: 1, -1, 3, -5, ..."""
    return (1 - (-2) ** (step + 1)) // 3


def swing_peer(world: int, rank: int, step: int) -> int:
    """The peer of ``rank`` at step ``step``: ``r + rho`` for even ``r``, ``r - rho`` for odd."""
    _steps(world)
    offset = swing_offset(step)
    return (rank + offset if rank % 2 == 0 else rank - offset) % world


def _reach(world: int, steps: int, rank: int, step: int) -> frozenset[int]:
    if step >= steps:
        return frozenset({rank})
    return _reach(world, steps, rank, step + 1) | _reach(world, steps, swing_peer(world, rank, step), step + 1)


def swing_chunk_owners(world: int) -> tuple[int, ...]:
    """Owner rank of every chunk position.

    The ranks a step's peer is responsible for form one contiguous range for
    every rank and step: a depth-first walk over the reach sets, whose two
    children at every step are ordered by their lowest rank.
    """
    steps = _steps(world)
    order: list[int] = []

    def walk(rank: int, step: int) -> None:
        if step >= steps:
            order.append(rank)
            return
        mine = _reach(world, steps, rank, step + 1)
        theirs = _reach(world, steps, swing_peer(world, rank, step), step + 1)
        for part in sorted((mine, theirs), key=min):
            walk(min(part), step + 1)

    walk(0, 0)
    return tuple(order)


def swing_phases(world: int, rank: int) -> tuple[Descriptor, ...]:
    """Phases of ``rank``: reduce-scatter steps (namespace 0), then all-gather steps in reverse (1)."""
    steps = _steps(world)
    owners = swing_chunk_owners(world)
    position = {owner: index for index, owner in enumerate(owners)}

    def span(ranks: frozenset[int]) -> tuple[int, int]:
        places = sorted(position[r] for r in ranks)
        if places != list(range(places[0], places[-1] + 1)):
            raise ProtocolError("Swing ranges must be contiguous")
        return places[0], places[-1] + 1

    phases = []
    for step in range(steps):
        peer = swing_peer(world, rank, step)
        first, end = span(_reach(world, steps, peer, step + 1))
        phases.append(Descriptor(first, end, peer, 0))
    for step in range(steps - 1, -1, -1):
        peer = swing_peer(world, rank, step)
        first, end = span(_reach(world, steps, rank, step + 1))
        phases.append(Descriptor(first, end, peer, 1))
    return tuple(phases)


# -- algorithm choice and launch geometry ------------------------------------------------------


def select_algorithm(
    nbytes: int,
    *,
    algorithm: str = "auto",
    large_algorithm: str = "twoshot",
    oneshot_max_bytes: int = 131072,
    swing_above_bytes: int = 0,
    available: Mapping[str, bool] | None = None,
) -> str:
    """The all-reduce algorithm for ``nbytes`` from the agreed settings only."""
    available = dict(available or {"oneshot": True, "twoshot": True, "swing": True})
    if algorithm != "auto":
        if algorithm not in ALGORITHMS:
            raise ProtocolError(f"unknown all-reduce algorithm {algorithm!r}")
        return algorithm
    if nbytes <= oneshot_max_bytes or not available.get(large_algorithm, False):
        return "oneshot"
    if swing_above_bytes and nbytes > swing_above_bytes and available.get("swing", False):
        return "swing"
    return large_algorithm


def grid_blocks(size_packs: int, threads: int, max_blocks: int,
                packs_per_thread: int = DEFAULT_PACKS_PER_THREAD) -> int:
    """Smallest power of two of at least ``ceil(packs / (packs_per_thread * threads))``, capped."""
    per_block = int(packs_per_thread) * int(threads)
    required = max(1, (int(size_packs) + per_block - 1) // per_block)
    return min(1 << (required - 1).bit_length(), int(max_blocks))


@dataclasses.dataclass(frozen=True)
class CounterLayout:
    """Word indices of the session's int32 device counter array.

    Word 0 epoch (newest completed sequence); one stage and one tail arrival
    counter per power-of-two grid class; the poison word; then one arrival
    counter per grid class for each later phase 1-7.
    """

    blocks: int

    def __post_init__(self) -> None:
        if self.blocks < 1 or self.blocks & (self.blocks - 1):
            raise ProtocolError("blocks must be a positive power of two")

    @property
    def classes(self) -> int:
        return self.blocks.bit_length()

    @property
    def words(self) -> int:
        return 2 + (1 + MAX_PHASES) * self.classes

    epoch_word = 0

    @staticmethod
    def grid_class(grid: int) -> int:
        if grid < 1 or grid & (grid - 1):
            raise ProtocolError(f"grid sizes are powers of two, got {grid}")
        return grid.bit_length() - 1

    def stage_word(self, grid: int) -> int:
        return 1 + self.grid_class(grid)

    def tail_word(self, grid: int) -> int:
        return 1 + self.classes + self.grid_class(grid)

    @property
    def poison_word(self) -> int:
        return 1 + 2 * self.classes

    def phase_word(self, grid: int, phase: int) -> int:
        if not 1 <= phase < MAX_PHASES:
            raise ProtocolError(f"phase arrival counters exist for phases 1-{MAX_PHASES - 1}")
        return 2 + 2 * self.classes + (phase - 1) * self.classes + self.grid_class(grid)


def parse_sizes(values: Sequence[int]) -> tuple[int, ...]:
    """Validate message sizes for a collective: positive multiples of 16 bytes."""
    result = []
    for value in values:
        if type(value) is not int or value <= 0 or value % PACK_BYTES:
            raise ProtocolError(f"message sizes are positive multiples of {PACK_BYTES} bytes, got {value}")
        result.append(value)
    return tuple(result)
