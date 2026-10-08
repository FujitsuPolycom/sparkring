"""CPU simulator of the ring collectives' links (``testing/ring_links.py``): the native link layer of every
rank on the verbs stand-in, the ring kernel's part played by threads. Ring all-gathers, reduce-scatters
and all-reduces back to back under every link-2 stagger D and link-3 stagger D3 the link slots hold, with
blocks of one piece, of as many pieces as a stagger adds rounds and of more: every output exact and every
rank's link items equal to the staggered rounds (empty items included); once with a rank taking its link
ops late; and the native refusals of staggers its op word or its slots do not allow."""

from __future__ import annotations

import time

import pytest

from sparkring_sircl import protocol as proto
from sparkring_sircl import routes
from sparkring_sircl.testing import ring_links

PIECE = 256


def _steps(world: int, staggers: list[int]) -> list[tuple[str, int, int, int, int]]:
    steps = []
    for stagger in staggers:
        for gather_stagger in staggers:
            as_many = max(1, (world - 2) * gather_stagger)
            steps += [("gather", PIECE, PIECE, 0, gather_stagger),
                      ("gather", PIECE * as_many, PIECE, 0, gather_stagger),
                      ("gather", 4 * PIECE + 16, PIECE, 0, gather_stagger),
                      ("scatter", 3 * PIECE + 32, PIECE, stagger, 0),
                      ("reduce", 2 * PIECE + 48, PIECE, stagger, gather_stagger),
                      ("reduce", PIECE * max(1, (world - 2) * stagger), PIECE, stagger, gather_stagger)]
    return steps


@pytest.mark.parametrize("layout_text, lanes, slots", [("ring:8", 2, 9), ("ring:3", 1, 6), ("path:0-3", 2, 8),
                                                       ("ring:4", 2, 5)])
def test_staggered_ring_ops_through_the_native_links(simulator_library, layout_text, lanes, slots):
    layout = routes.Layout.parse(layout_text)
    links = ring_links.RingLinks(str(simulator_library), layout, lanes=lanes, slots=slots, slot_bytes=4096)
    try:
        links.connect()
        world = layout.world
        staggers = [d for d in range(proto.MAX_RING_STAGGER + 1) if slots >= proto.ring_stagger_slots(world, d)]
        assert len(staggers) >= 2
        ring_links.run_ring_ops(links, _steps(world, staggers))
        assert not any(proxy.failed() for proxy in links.proxies)
    finally:
        links.close()


def test_staggered_ring_ops_with_a_rank_taking_its_link_ops_late(simulator_library):
    layout = routes.Layout.parse("ring:4")
    links = ring_links.RingLinks(str(simulator_library), layout, lanes=2, slots=8, slot_bytes=4096)
    try:
        links.connect()
        links.delay_link_ops(3, 2000)
        ring_links.run_ring_ops(links, [("gather", 5 * PIECE, PIECE, 0, 2), ("reduce", 3 * PIECE, PIECE, 1, 2),
                                        ("gather", PIECE, PIECE, 0, 0), ("reduce", 4 * PIECE, PIECE, 2, 1),
                                        ("scatter", 2 * PIECE, PIECE, 2, 0)])
    finally:
        links.close()


@pytest.mark.parametrize("op, stagger, gather_stagger, high, message", [
    (proto.LinkOp.RING_GATHER, 0, 1, 0, "all-gather stagger 1"),        # 3 slots hold no stagger on 3 ranks
    (proto.LinkOp.RING_SCATTER, 0, 1, 0, "all-gather stagger 1"),       # no finished pieces to forward
    (proto.LinkOp.RING_GATHER, 1, 0, 0, "with stagger 1"),              # no partials to stagger
    (proto.LinkOp.RING_GATHER, 0, 0, 1, "op word"),                     # bits 24-31 hold nothing
])
def test_native_refuses_staggers_its_op_word_or_slots_do_not_allow(simulator_library, op, stagger, gather_stagger,
                                                                    high, message):
    layout = routes.Layout.parse("ring:3")
    links = ring_links.RingLinks(str(simulator_library), layout, lanes=1, slots=3, slot_bytes=4096)
    try:
        links.connect()
        word = int(op) | stagger << proto.RING_STAGGER_SHIFT | gather_stagger << proto.RING_GATHER_STAGGER_SHIFT
        word |= high << 24
        params = links.links.param_word(1)
        links._store(0, params, word)
        links._store(0, params + 4, PIECE)
        links._store(0, params + 8, PIECE)
        links._store(0, links.links.ctrl_off, 1)
        deadline = time.monotonic() + 5
        while not links.proxies[0].failed() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert links.proxies[0].failed() and message in links.proxies[0].error()
    finally:
        links.close()
