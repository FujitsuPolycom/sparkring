"""Edit SIRCL's package for flags-only own items: bit 24 of a link op word makes every own item of that op go
out as its flags only (request FO from libsircl).

python edit_FO.py <package root: spark_transport/sircl>

Changes, each an exact replacement that must find its old text once (line endings kept as found):

- oneshot/_roce_proxy.c: link_take_ops accepts bit 24 of the op word (bits 25-31 still hold nothing and
  are refused) and keeps it as the op's `own_flags`; link_source sizes every own item of such an op 0
  bytes, so the item goes out as its flags only, as a staggered link's empty items already do. Nothing
  else changes: not the wire (a flags-only item is a form the links already carry), not the connection
  record, not the native ABI version (9).
- protocol.py: `RING_OWN_FLAGS` (bit 24) and `ring_op_word(..., own_flags=False)`.
- README.md: the link op word's bit 24 in the protocol module's description.
- tests/test_ring_links.py: the refusal case of the op word's high bits uses bit 25, since bit 24 now
  holds the flag.
- tests/test_link_own_flags.py (new): a pair on the verbs stand-in, one rank's ring all-gather with bit 24
  and the other's without, 12 pieces in 8 slots: the flags-only rank's data is never written to its
  peer's receive slots (a sentinel stays) and its link bytes posted are 0, while every item's flags
  arrive and the other direction is exact; and the same op without the bit, both directions exact.

Why: the libsircl library's pair exchange (one op that is SIRCL's ring all-gather on the wire) carries
one-way collectives (broadcast, scatter, gather, reduce to the root, a one-way send) by discarding the
empty direction's slots at the receiver. Without this bit the empty direction still posts full pieces of
stale slot bytes. On ConnectX-7 (two Sparks, nccl-tests v2.21.1, bfloat16, out of place, 512 KiB to
256 MiB), libsircl with this proxy change and its kernel setting the bit moved 133.1 GB per direction on
five one-way rows against 232.7 GB without it (NVIDIA NCCL 2.32.3: 133.2 GB), and ran broadcast at
256 MiB in 11.00 ms against 11.96 ms (NVIDIA 11.02 ms), and at 4 MiB in 178 us against 218 us (NVIDIA
191-193 us).
"""

import sys
from pathlib import Path


def replace_once(path: Path, pairs) -> None:
    raw = path.read_bytes()
    crlf = b"\r\n" in raw and raw.count(b"\r\n") == raw.count(b"\n")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{path}: found {count} of {old[:90]!r}")
        text = text.replace(old, new)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


PROXY_C = [
    ("    uint32_t stagger3;        /* ring all-gather stagger D3 (link 3) */\n",
     "    uint32_t stagger3;        /* ring all-gather stagger D3 (link 3) */\n"
     "    uint32_t own_flags;       /* op word bit 24: every own item goes out as its flags only */\n"),
    ('''        if ((word >> 24) != 0u) {
            FAIL(c, "link op %u: op word 0x%08x sets bits 24-31, which hold nothing", next, word);
            return -1;
        }''', '''        if ((word >> 25) != 0u) {
            FAIL(c, "link op %u: op word 0x%08x sets bits 25-31, which hold nothing", next, word);
            return -1;
        }'''),
    ("        entry->stagger3 = stagger3;\n",
     "        entry->stagger3 = stagger3;\n        entry->own_flags = (word >> 24) & 1u;\n"),
    ('''    if (r < op->own_round[l]) {
        *own = 1;
        *item = op->first_own[l] + t * op->own_round[l] + r;''', '''    if (r < op->own_round[l]) {
        *own = 1;
        *item = op->first_own[l] + t * op->own_round[l] + r;
        /* Op word bit 24: the peer has no use for this rank's own items (it discards them), so they go
         * out as their flags only. */
        if (op->own_flags) *bytes = 0;'''),
]

PROTOCOL_PY = [
    ("""# The ring reduce-scatter's stagger D (link 2) travels in bits 8-15 of its link op word, the ring
# all-gather's stagger D3 (link 3) in bits 16-23; bits 24-31 are zero.
RING_STAGGER_SHIFT = 8
RING_GATHER_STAGGER_SHIFT = 16
""", """# The ring reduce-scatter's stagger D (link 2) travels in bits 8-15 of its link op word, the ring
# all-gather's stagger D3 (link 3) in bits 16-23; bit 24 (RING_OWN_FLAGS) makes every own item of the op
# go out as its flags only (its peer discards that rank's own items); bits 25-31 are zero.
RING_STAGGER_SHIFT = 8
RING_GATHER_STAGGER_SHIFT = 16
RING_OWN_FLAGS = 1 << 24
"""),
    ('''def ring_op_word(op: int, stagger: int = 0, gather_stagger: int = 0) -> int:
    """The link op word of a ring op: its code, the stagger D of link 2 (ring reduce-scatter and
    all-reduce) and the stagger D3 of link 3 (ring all-gather and all-reduce)."""''',
     '''def ring_op_word(op: int, stagger: int = 0, gather_stagger: int = 0, own_flags: bool = False) -> int:
    """The link op word of a ring op: its code, the stagger D of link 2 (ring reduce-scatter and
    all-reduce), the stagger D3 of link 3 (ring all-gather and all-reduce) and, with ``own_flags``, the
    bit that sends every own item of the op as its flags only."""'''),
    ('''    return int(op) | int(stagger) << RING_STAGGER_SHIFT | int(gather_stagger) << RING_GATHER_STAGGER_SHIFT
''', '''    return (int(op) | int(stagger) << RING_STAGGER_SHIFT | int(gather_stagger) << RING_GATHER_STAGGER_SHIFT
            | (RING_OWN_FLAGS if own_flags else 0))
'''),
]

README = [
    ("""`RING_GATHER_STAGGER_SHIFT` (a link op word carries the op in bits 0-7, the
stagger of link 2 in bits 8-15 and the stagger of link 3 in bits 16-23;
`ring_op_word` builds it),""", """`RING_GATHER_STAGGER_SHIFT` (a link op word carries the op in bits 0-7, the
stagger of link 2 in bits 8-15 and the stagger of link 3 in bits 16-23, and
bit 24, `RING_OWN_FLAGS`, sends every own item of the op as its flags only;
`ring_op_word` builds it),"""),
]

RING_LINKS_TEST = [
    ("""    (proto.LinkOp.RING_GATHER, 0, 0, 1, "op word"),                     # bits 24-31 hold nothing""",
     """    (proto.LinkOp.RING_GATHER, 0, 0, 2, "op word"),                     # bits 25-31 hold nothing"""),
]

TEST = '''"""Flags-only own items: bit 24 of a link op word (``protocol.RING_OWN_FLAGS``) sends every own item of the op
as its flags only, for a rank whose peer discards those items.

A pair on the in-memory verbs stand-in (``testing/ring_links.py``), the kernel's part played by threads,
runs one ring all-gather of 12 pieces through 8 link slots, so the slots wrap. With the bit on rank 1
only, rank 1 stages a pattern it must not send: rank 0 receives every item's flags, its receive slots keep
their sentinel bytes, and rank 1 posts no link bytes; rank 1 receives rank 0's shard exact. Without the
bit both directions are exact.
"""

import ctypes
import threading

from sparkring_sircl import protocol as proto
from sparkring_sircl import routes
from sparkring_sircl.testing import ring_links
from sparkring_sircl.testing.fabric import payload

PIECE = 1024
PIECES = 12


def _gather(links, flags_only):
    """One ring all-gather of PIECES pieces on both ranks; rank r's own items flags only when r is in
    ``flags_only``. Returns what each rank received, piece by piece."""
    nbytes = PIECE * PIECES
    shards = [payload(0, 41, nbytes), bytes([0x55]) * nbytes]
    received = [[None] * PIECES for _ in range(2)]
    errors = []

    def rank_part(rank):
        try:
            start = dict(links.counters[rank])
            seq = start["seq"] + 1
            params = links.links.param_word(seq & 1)
            links._store(rank, params, proto.ring_op_word(proto.LinkOp.RING_GATHER, own_flags=rank in flags_only))
            links._store(rank, params + 4, nbytes)
            links._store(rank, params + 8, PIECE)
            links._store(rank, links.links.ctrl_off, seq)

            def own():
                for p in range(PIECES):
                    item = start["own3"] + p
                    links._wait_own_slot(rank, 3, item)
                    links._stage(rank, 3, item, shards[rank][p * PIECE:(p + 1) * PIECE])

            def copy():
                for i in range(PIECES):
                    g = start["in3"] + i
                    links._wait_inbound(rank, 3, g)
                    received[rank][i] = links._received(rank, 3, g, PIECE)
                    links._consume(rank, 3, g)

            roles = [threading.Thread(target=own), threading.Thread(target=copy)]
            for role in roles:
                role.start()
            for role in roles:
                role.join()
            counters = links.counters[rank]
            counters["own3"] += PIECES
            counters["in3"] += PIECES
            counters["seq"] += 1
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(f"rank {rank}: {type(error).__name__}: {error}")

    threads = [threading.Thread(target=rank_part, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    return shards, received


def test_flags_only_own_items_carry_no_payload(simulator_library):
    links = ring_links.RingLinks(str(simulator_library), routes.Layout.parse("path:0-1"), lanes=2, slots=8,
                                 slot_bytes=4096)
    try:
        links.connect()
        sentinel = bytes([0xEE]) * PIECE
        for m in range(links.slots):
            ctypes.memmove(links._at(0, links.links.recv_slot(3, m)), sentinel, PIECE)
        shards, received = _gather(links, flags_only={1})
        assert b"".join(received[1]) == shards[0]
        assert all(piece == sentinel for piece in received[0])
        stats = [proxy.stats() for proxy in links.proxies]
        assert stats[1]["link_bytes_posted"] == 0
        assert stats[0]["link_bytes_posted"] == PIECE * PIECES
        assert stats[0]["link_items_posted"] == stats[1]["link_items_posted"] == PIECES
        assert not any(proxy.failed() for proxy in links.proxies)
    finally:
        links.close()


def test_without_the_bit_both_directions_are_exact(simulator_library):
    links = ring_links.RingLinks(str(simulator_library), routes.Layout.parse("path:0-1"), lanes=2, slots=8,
                                 slot_bytes=4096)
    try:
        links.connect()
        shards, received = _gather(links, flags_only=set())
        assert b"".join(received[1]) == shards[0]
        assert b"".join(received[0]) == shards[1]
        assert all(proxy.stats()["link_bytes_posted"] == PIECE * PIECES for proxy in links.proxies)
    finally:
        links.close()
'''


def main() -> int:
    root = Path(sys.argv[1])
    replace_once(root / "sparkring_sircl" / "oneshot" / "_roce_proxy.c", PROXY_C)
    replace_once(root / "sparkring_sircl" / "protocol.py", PROTOCOL_PY)
    replace_once(root / "README.md", README)
    replace_once(root / "tests" / "test_ring_links.py", RING_LINKS_TEST)
    # The package's sources use CRLF line endings; the new test follows them.
    (root / "tests" / "test_link_own_flags.py").write_bytes(TEST.replace("\n", "\r\n").encode("utf-8"))
    print("edited", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
