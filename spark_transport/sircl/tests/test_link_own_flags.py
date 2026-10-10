"""Flags-only own items: bit 24 of a link op word (``protocol.RING_OWN_FLAGS``) sends every own item of the op
as its flags only, for a rank whose peer discards those items.

A pair on the in-memory verbs stand-in (``testing/ring_links.py``), the kernel's part played by threads,
runs one ring all-gather of 12 pieces through 8 link slots, so the slots wrap. With the bit on rank 1
only, rank 1 stages a pattern it must not send: rank 0 receives every item's flags, its receive slots keep
their sentinel bytes, and rank 1 posts no link bytes; rank 1 receives rank 0's shard exact. Without the
bit both directions are exact. On one pair of contexts, ops whose flags-only rank changes (rank 1, then rank
0, then neither), each of 12 pieces through the 8 slots, keep the slots, items and credits in step: every
useful direction is exact, every absent one leaves its receive slots unchanged, and only the rank that sends
payload posts link bytes.
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


def test_fo_direction_changes_after_wrap(simulator_library):
    links = ring_links.RingLinks(str(simulator_library), routes.Layout.parse("path:0-1"), lanes=2, slots=8,
                                 slot_bytes=4096)
    try:
        links.connect()
        sentinel = bytes([0xEE]) * PIECE
        posted = [0, 0]
        for flags_only in ({1}, {0}, set(), {1}):
            # The absent direction's receive slots hold the sentinel: a flags-only item must leave it there.
            for rank in flags_only:
                peer = 1 - rank
                for m in range(links.slots):
                    ctypes.memmove(links._at(peer, links.links.recv_slot(3, m)), sentinel, PIECE)
            shards, received = _gather(links, flags_only=flags_only)
            for rank in range(2):
                peer = 1 - rank
                if peer in flags_only:
                    assert all(piece == sentinel for piece in received[rank]), (flags_only, rank)
                else:
                    assert b"".join(received[rank]) == shards[peer], (flags_only, rank)
                if rank not in flags_only:
                    posted[rank] += PIECE * PIECES
            stats = [proxy.stats() for proxy in links.proxies]
            assert [stat["link_bytes_posted"] for stat in stats] == posted, (flags_only, stats)
            assert not any(proxy.failed() for proxy in links.proxies), [proxy.error() for proxy in links.proxies]
        assert stats[0]["link_items_posted"] == stats[1]["link_items_posted"] == 4 * PIECES
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
