"""Wire protocol of SIRCL's point-to-point channels: arena layout, items and headers (torch-free).

A group's point-to-point context (:mod:`sparkring_sircl.p2p`) gives every
ordered pair of ranks a channel: ``slots`` slots of ``slot_bytes`` on both
ends. The GPU kernels (``p2p/_kernels.py``), the native progress thread
(``p2p/_p2p_proxy.c``) and the host-side emulations of the tests compute the
same offsets and words; this module is their reference in Python.

Arena of one rank (pinned host memory the GPU addresses at its host pointer):

- the control line (:class:`Control`) in the first ``CONTROL_BYTES``;
- one block of :attr:`P2PLayout.block_bytes` per rank of the group, block
  ``p`` at ``CONTROL_BYTES + p * block_bytes`` (the own rank's block is
  unused). Block ``p`` holds the channel from ``p`` (receive slots, flag
  lines, the consumed line) and the channel toward ``p`` (send slots, the
  desc and ready lines, the sent word, and the credit word that ``p``
  writes).

A message of ``n`` bytes travels as :func:`items` items of the channel; item
``g`` (counted from 0 over the session, 32 bits, wrapping) uses slot
``g % slots`` and carries tag ``g + 1`` in its flag lines and in the ready,
consumed, sent and credit words. Its header (:func:`header`) holds its byte
count, whether it ends its message and, on the last item, ``n % 16``, so a
receiver detects a message of another size on the first item that differs.

Status: implemented; the native layout equals :class:`P2PLayout` by CPU test.
"""

from __future__ import annotations

import dataclasses
import enum

from ..protocol import FLAG_STRIDE, MAX_DEVICES, MAX_LANES, PACK_BYTES, ProtocolError, stripe

ABI_VERSION = 1
API_VERSION = 1
CONTROL_BYTES = 4096
LINE = FLAG_STRIDE
MAX_WORLD = 16
MIN_SLOTS = 2
MAX_SLOTS = LINE // 4          # one word per slot in a 128-byte line
SLOT_ALIGNMENT = 4096
MAX_SLOT_BYTES = 1 << 29       # an item's byte count fits header bits 4-29
LAST = 1 << 31                 # header bit: the item ends its message
TAIL_MASK = PACK_BYTES - 1     # header bits 0-3 of a last item: message bytes modulo 16
BYTES_MASK = ((1 << 30) - 1) & ~TAIL_MASK
DEFAULT_SLOTS = 8
DEFAULT_SLOT_BYTES = 512 << 10
DEFAULT_BLOCKS = 4
DEFAULT_THREADS = 512
DEFAULT_UNROLL = 4
DEFAULT_WINDOW_BYTES = 131072
DEFAULT_CHUNK_BYTES = 32768
MIN_WINDOW_PIECE = 4096        # smallest write of a windowed stripe when the window has room for less than a chunk
LAYOUT_WORDS = 11


class Control(enum.IntEnum):
    """32-bit words of the control line."""

    WAIT_LIMIT_US = 0   # flag-wait limit in microseconds (host), read by kernels when a wait starts
    ERROR_TAG = 1       # tag of the item a failed wait or check concerned; written last (0: no kernel error)
    ERROR_PEER = 2      # the channel's peer
    ERROR_LANE = 3      # the lane whose flag did not arrive (255: none)
    ERROR_KIND = 4      # ErrorKind
    ERROR_EXPECTED = 5  # header the receiver expected (size mismatch)
    ERROR_GOT = 6       # header that arrived (size mismatch)
    POISON = 7          # nonzero: every kernel returns at once (a kernel failure, or the native layer's)
    ABORT = 8           # written by a peer's progress thread: that peer's rank + 1
    LANE_CHECK = 31     # written by peers during the setup lane check


class ErrorKind(enum.IntEnum):
    """What a failed kernel wait or check concerned (control word ``ERROR_KIND``)."""

    FLAG = 1      # a receive waited for an item's lane flag beyond the wait limit
    SLOT = 2      # a send waited for a free send slot beyond the wait limit
    SIZE = 3      # an item's header differs from the receive's (another message size or boundary)


@dataclasses.dataclass(frozen=True)
class P2PLayout:
    """Byte offsets of a point-to-point arena (``p2p_layout`` of the native layer)."""

    world: int
    lanes: int
    slots: int
    slot_bytes: int

    def __post_init__(self) -> None:
        if not 2 <= self.world <= MAX_WORLD:
            raise ProtocolError(f"a point-to-point group has 2 to {MAX_WORLD} ranks, got {self.world}")
        if not 1 <= self.lanes <= MAX_LANES:
            raise ProtocolError(f"a channel has 1 to {MAX_LANES} lanes, got {self.lanes}")
        if not MIN_SLOTS <= self.slots <= MAX_SLOTS or self.slots & (self.slots - 1):
            raise ProtocolError(f"a channel has a power of two of {MIN_SLOTS} to {MAX_SLOTS} slots, got {self.slots}")
        if self.slot_bytes <= 0 or self.slot_bytes % SLOT_ALIGNMENT or self.slot_bytes > MAX_SLOT_BYTES:
            raise ProtocolError(f"slot bytes {self.slot_bytes} must be a positive multiple of {SLOT_ALIGNMENT} "
                                f"up to {MAX_SLOT_BYTES}")

    @property
    def ring_bytes(self) -> int:
        return self.slots * self.slot_bytes

    @property
    def recv_off(self) -> int:
        return 0

    @property
    def send_off(self) -> int:
        return self.ring_bytes

    @property
    def flag_off(self) -> int:
        return 2 * self.ring_bytes

    @property
    def desc_off(self) -> int:
        return self.flag_off + self.slots * self.lanes * LINE

    @property
    def ready_off(self) -> int:
        return self.desc_off + LINE

    @property
    def consumed_off(self) -> int:
        return self.ready_off + LINE

    @property
    def sent_off(self) -> int:
        return self.consumed_off + LINE

    @property
    def credit_off(self) -> int:
        return self.sent_off + LINE

    @property
    def block_bytes(self) -> int:
        end = self.credit_off + LINE
        return -(-end // SLOT_ALIGNMENT) * SLOT_ALIGNMENT

    @property
    def total_bytes(self) -> int:
        return CONTROL_BYTES + self.world * self.block_bytes

    def block(self, peer: int) -> int:
        """Offset of the block of ``peer``."""
        if not 0 <= peer < self.world:
            raise ProtocolError(f"peer {peer} outside a group of {self.world}")
        return CONTROL_BYTES + peer * self.block_bytes

    def recv_slot(self, peer: int, slot: int) -> int:
        return self.block(peer) + self.recv_off + slot * self.slot_bytes

    def send_slot(self, peer: int, slot: int) -> int:
        return self.block(peer) + self.send_off + slot * self.slot_bytes

    def flag(self, peer: int, slot: int, lane: int) -> int:
        """Flag word of ``slot`` and ``lane`` of the channel from ``peer``; lane 0's line holds the header
        at byte 4."""
        return self.block(peer) + self.flag_off + (slot * self.lanes + lane) * LINE

    def header(self, peer: int, slot: int) -> int:
        return self.flag(peer, slot, 0) + 4

    def word(self, peer: int, offset: int, slot: int = 0) -> int:
        return self.block(peer) + offset + 4 * slot

    def as_tuple(self) -> tuple[int, ...]:
        """The native ``p2p_layout`` words: control bytes, block bytes, the block offsets of the receive
        slots, send slots, flag lines, desc, ready, consumed, sent and credit lines, and the total."""
        return (CONTROL_BYTES, self.block_bytes, self.recv_off, self.send_off, self.flag_off, self.desc_off,
                self.ready_off, self.consumed_off, self.sent_off, self.credit_off, self.total_bytes)

    def per_peer_bytes(self) -> int:
        return self.block_bytes


def padded(nbytes: int) -> int:
    """``nbytes`` rounded up to whole 16-byte packs."""
    if nbytes < 0:
        raise ProtocolError(f"a message has a non-negative size, got {nbytes}")
    return -(-nbytes // PACK_BYTES) * PACK_BYTES


def items(nbytes: int, slot_bytes: int) -> int:
    """Items of a message of ``nbytes`` bytes: one per slot of its padded size, at least one."""
    if slot_bytes <= 0 or slot_bytes % PACK_BYTES:
        raise ProtocolError(f"slot bytes {slot_bytes} must be a positive multiple of 16")
    return max(1, -(-padded(nbytes) // slot_bytes))


def item_bytes(nbytes: int, slot_bytes: int, index: int) -> int:
    """Wire bytes of item ``index`` of a message of ``nbytes`` bytes (a multiple of 16)."""
    count = items(nbytes, slot_bytes)
    if not 0 <= index < count:
        raise ProtocolError(f"item {index} of a message of {count} items")
    return min(slot_bytes, padded(nbytes) - index * slot_bytes)


def header(nbytes: int, slot_bytes: int, index: int) -> int:
    """Header word of item ``index`` of a message of ``nbytes`` bytes."""
    word = item_bytes(nbytes, slot_bytes, index)
    if index == items(nbytes, slot_bytes) - 1:
        word |= LAST | (nbytes & TAIL_MASK)
    return word


def describe_header(word: int) -> str:
    """A header word in words, for error messages."""
    word &= 0xFFFFFFFF
    text = f"{word & BYTES_MASK} bytes"
    if word & LAST:
        tail = word & TAIL_MASK
        text += ", last item" + (f" of a message of 16k+{tail} bytes" if tail else "")
    return text


def tag(item: int) -> int:
    """Tag of item ``item`` (32-bit)."""
    return (item + 1) & 0xFFFFFFFF


def item_stripes(nbytes: int, lanes: int) -> tuple[tuple[int, int], ...]:
    """``(first pack, packs)`` of every lane for an item of ``nbytes`` wire bytes."""
    if nbytes % PACK_BYTES:
        raise ProtocolError(f"an item carries whole packs, got {nbytes} bytes")
    return tuple(stripe(nbytes // PACK_BYTES, lanes, lane) for lane in range(lanes))


def check_devices(names: tuple[str, ...]) -> None:
    if not 1 <= len(names) <= MAX_DEVICES:
        raise ProtocolError(f"a point-to-point context opens 1 to {MAX_DEVICES} RDMA devices, got {len(names)}")


__all__ = [
    "ABI_VERSION", "API_VERSION", "BYTES_MASK", "CONTROL_BYTES", "Control", "DEFAULT_BLOCKS", "DEFAULT_CHUNK_BYTES",
    "DEFAULT_SLOTS", "DEFAULT_SLOT_BYTES", "DEFAULT_THREADS", "DEFAULT_UNROLL", "DEFAULT_WINDOW_BYTES", "ErrorKind",
    "LAST", "LAYOUT_WORDS", "LINE", "MAX_SLOTS", "MAX_SLOT_BYTES", "MAX_WORLD", "MIN_SLOTS", "MIN_WINDOW_PIECE",
    "P2PLayout", "TAIL_MASK", "check_devices", "describe_header", "header", "item_bytes", "item_stripes", "items",
    "padded", "tag",
]
