"""Wire-protocol arithmetic against the reference vectors in ``tests/data/numeric.json``."""

from __future__ import annotations

import pytest

from sparkring_sircl import protocol as p

from vectors import load


def _params(key: str) -> dict[str, int]:
    return {name: int(value) for name, value in (item.split("=") for item in key.split(","))}


def test_stripe_vectors():
    for key, expected in load("numeric.json")["stripe"].items():
        args = _params(key)
        assert [list(item) for item in p.stripes(args["count"], args["lanes"])] == expected, key


def test_chunk_vectors():
    for key, expected in load("numeric.json")["chunk"].items():
        args = _params(key)
        assert [list(item) for item in p.chunks(args["packs"], args["world"])] == expected, key


def test_flag_index_vectors():
    for key, expected in load("numeric.json")["flag_index"].items():
        a = _params(key)
        assert p.flag_index(a["ns"], a["src"], a["slot"], a["lane"], a["world"], a["lanes"]) == expected, key


def test_ring_farthest_vectors():
    for key, per_rank in load("numeric.json")["ring_farthest"].items():
        world = _params(key)["world"]
        for rank, expected in per_rank.items():
            assert list(p.ring_farthest(int(rank), world)) == expected, (key, rank)
            assert list(p.post_order(int(rank), world, "ring-farthest")) == expected


def test_op_word_and_descriptor_vectors():
    numeric = load("numeric.json")
    for key, expected in numeric["op_word"].items():
        a = _params(key)
        assert p.op_word(a["op"], a["bytes"]) == expected, key
        assert p.decode_op_word(expected) == (p.Op(a["op"]), a["bytes"])
    for key, expected in numeric["descriptor"].items():
        a = _params(key)
        assert p.descriptor(a["first"], a["end"], a["peer"], a["ns"]) == expected, key
        assert p.decode_descriptor(expected) == p.Descriptor(a["first"], a["end"], a["peer"], a["ns"])


def test_swing_vectors():
    for key, entry in load("numeric.json")["swing"].items():
        world = _params(key)["world"]
        assert list(p.swing_chunk_owners(world)) == entry["chunk_owners"], key
        for rank, phases in entry["phases"].items():
            got = [[d.peer, d.first, d.end, d.namespace] for d in p.swing_phases(world, int(rank))]
            assert got == phases, (key, rank)


def test_select_algorithm_vectors():
    for key, expected in load("numeric.json")["select_algorithm"].items():
        a = _params(key)
        world = a["world"]
        available = {"oneshot": True, "twoshot": True, "swing": world & (world - 1) == 0}
        got = p.select_algorithm(a["bytes"], oneshot_max_bytes=28672, swing_above_bytes=a["swing_above"],
                                 available=available)
        assert got == expected, key


@pytest.mark.parametrize("world", range(2, 9))
@pytest.mark.parametrize("lanes", [1, 2])
def test_exhaustive_small_cases(world, lanes):
    for packs in range(0, 257):
        parts = p.chunks(packs, world)
        assert sum(count for _, count in parts) == packs
        assert all(parts[j][0] + parts[j][1] == parts[j + 1][0] for j in range(world - 1))
        assert max(c for _, c in parts) - min(c for _, c in parts) <= 1
        split = p.stripes(packs, lanes)
        assert sum(count for _, count in split) == packs
        assert split[0][0] == 0
    lines = {p.flag_index(ns, s, slot, lane, world, lanes)
             for ns in (0, 1) for s in range(world) for slot in (0, 1) for lane in range(lanes)}
    assert lines == set(range(2 * world * p.SLOTS * lanes))
    assert max(lines) < world * p.SLOTS * p.FLAG_LINES


def test_op_word_rejects_what_the_wire_cannot_carry():
    for bad in (0, 8, 1 << 30, -16):
        with pytest.raises(p.ProtocolError):
            p.op_word(1, bad)
    with pytest.raises(p.ProtocolError):
        p.Descriptor(3, 2, 1, 0).word()
    with pytest.raises(p.ProtocolError):
        p.Descriptor(0, 9, 1, 0).word(world=8)
    assert p.decode_descriptor(0) is None


def test_post_order_rules():
    assert p.post_order(2, 4, None) == (0, 1, 3)
    assert p.post_order(2, 4, "3, 0,1".replace(" ", "")) == (3, 0, 1)
    for bad in ("1,2,3", "0,1", "0,1,1", "x"):
        with pytest.raises(p.ProtocolError):
            p.post_order(2, 4, bad)


def test_launch_geometry():
    for packs in (0, 1, 511, 1024, 1025, 4096, 1 << 20):
        for blocks in (1, 2, 8):
            grid = p.grid_blocks(packs, 512, blocks)
            assert grid & (grid - 1) == 0 and 1 <= grid <= blocks
    layout = p.CounterLayout(8)
    assert layout.classes == 4 and layout.words == 2 + 9 * 4
    words = {layout.stage_word(g) for g in (1, 2, 4, 8)} | {layout.tail_word(g) for g in (1, 2, 4, 8)}
    assert len(words) == 8 and layout.poison_word == 9 and 0 not in words
    assert layout.phase_word(1, 1) == 10 and layout.phase_word(8, 7) == 2 + 8 + 6 * 4 + 3


def test_arena_layout_matches_the_wire_format():
    layout = p.ArenaLayout(8, 155648)
    assert layout.flag_off == 8 * 2 * 155648
    assert layout.send_off - layout.flag_off == 8 * 2 * 4 * 128
    assert layout.total_bytes == layout.ctrl_off + 128
    with pytest.raises(p.ProtocolError):
        p.ArenaLayout(8, 1000)
    assert p.slot_bytes_for(131072, 155648) == 155648
    assert p.multi_phase_available(2, 1 << 20) and not p.multi_phase_available(2, 1 << 30)


def test_chain_layout_and_halves():
    from sparkring_sircl import protocol as p

    layout = p.ChainLayout(2, 4, 1 << 20)
    assert layout.send_off == 4 * 4 * (1 << 20) and layout.rflag_off == 2 * layout.send_off
    assert layout.ready_off == layout.rflag_off + 4 * 4 * 2 * 128
    assert layout.ctrl_off == layout.ready_off + 4 * 4 * 128 and layout.total_bytes == layout.ctrl_off + 128
    assert layout.flag_line(3, 2, 1) == layout.rflag_off + ((3 * 4 + 2) * 2 + 1) * 128
    assert layout.param_word(1) == layout.ctrl_off + 4 * 5
    assert p.chain_offset(4097) == 8192 and p.chain_offset(8192) == 8192
    assert p.chain_halves(7) == (3, 4) and p.chain_halves(1) == (0, 1) and p.chain_chunks(65, 32) == 3
    for bad in ((3, 4, 4096), (2, 1, 4096), (2, 33, 4096), (2, 4, 1000)):
        with pytest.raises(p.ProtocolError, match="chain geometry"):
            p.ChainLayout(*bad)


def test_link_layout_rounds_and_sources():
    from sparkring_sircl import protocol as p

    layout = p.LinkLayout(2, 8, 512 << 10)
    assert p.LINKS == 4
    assert layout.own_off == 4 * 8 * (512 << 10) and layout.rflag_off == 2 * layout.own_off
    assert layout.ready_off == layout.rflag_off + 4 * 8 * 2 * 128
    assert layout.ctrl_off == layout.ready_off + 4 * 4 * 128 and layout.total_bytes == layout.ctrl_off + 128
    assert layout.flag_line(3, 3, 1) == layout.rflag_off + ((3 * 8 + 3) * 2 + 1) * 128
    for bad in ((3, 4, 4096), (2, 1, 4096), (2, 33, 4096), (2, 4, 1000)):
        with pytest.raises(p.ProtocolError, match="link geometry"):
            p.LinkLayout(*bad)
    gather, scatter = p.LinkOp.ALL_GATHER, p.LinkOp.REDUCE_SCATTER
    for world in range(2, 9):
        for index in range(world):
            for link in (0, 1):
                for op in (gather, scatter):
                    rounds = p.link_rounds(op, world, index, link)
                    # What a rank sends on a link per round, its downstream neighbor receives.
                    downstream = index + 1 if link == 0 else index - 1
                    if rounds.out:
                        assert p.link_rounds(op, world, downstream, link).inbound == rounds.out
                    else:
                        assert downstream in (-1, world)
                total = sum(p.link_rounds(gather, world, i, 0).out for i in range(world))
                assert total == world * (world - 1) // 2      # cable i -> i + 1 carries owners 0..i
    rounds = p.link_rounds(gather, 4, 2, 0)                   # own piece, then owners 1 and 0
    assert (rounds.out, rounds.inbound, rounds.own, rounds.forwards) == (3, 2, 1, True)
    assert [p.link_source(rounds, q) for q in range(6)] == [
        ("own", 0), ("in", 0), ("in", 1), ("own", 1), ("in", 2), ("in", 3)]
    rounds = p.link_rounds(scatter, 4, 1, 0)                  # partials for owners 3 and 2; owners 3..1 arrive
    assert (rounds.out, rounds.inbound, rounds.own, rounds.forwards) == (2, 3, 2, False)
    assert p.link_rounds(gather, 4, 3, 0).out == 0 and p.link_rounds(gather, 4, 0, 1).out == 0
    # Ring ops: link 2 carries W - 1 partial sums per round, link 3 the own piece and W - 2 forwards.
    for world in range(2, 9):
        for index in range(world):
            partials = p.link_rounds(p.LinkOp.RING_SCATTER, world, index, 2)
            assert (partials.out, partials.inbound, partials.own) == (world - 1, world - 1, world - 1)
            assert not any(partials.forwarded(r) for r in range(world - 1))
            results = p.link_rounds(p.LinkOp.RING_GATHER, world, index, 3)
            assert (results.out, results.inbound, results.own) == (world - 1, world - 1, 1)
            assert [results.forwarded(r) for r in range(world - 1)] == [r < world - 2 for r in range(world - 1)]
            reduce2 = p.link_rounds(p.LinkOp.RING_REDUCE, world, index, 2)
            reduce3 = p.link_rounds(p.LinkOp.RING_REDUCE, world, index, 3)
            assert reduce2 == partials and reduce3 == results
            for link in (0, 1):
                assert p.link_rounds(p.LinkOp.RING_REDUCE, world, index, link).out == 0
            assert p.link_rounds(p.LinkOp.ALL_GATHER, world, index, 2).out == 0
    assert [p.link_source(p.link_rounds(p.LinkOp.RING_GATHER, 4, 1, 3), q) for q in range(6)] == [
        ("own", 0), ("in", 0), ("in", 1), ("own", 1), ("in", 3), ("in", 4)]
    assert p.link_pieces(4112, 4096) == 2 and p.link_piece_bytes(4112, 4096, 1) == 16
    assert p.link_pieces(0, 4096) == 0
    with pytest.raises(p.ProtocolError):
        p.link_pieces(40, 4096)


def test_staggered_ring_partials():
    """Link 2 of a ring reduce-scatter with stagger D: every (piece, type) pair once, the relay of inbound
    type r leaving D rounds (D (W - 1) + 1 items) after it as type r + 1, empty items only outside a type's
    pieces, and the slots that order needs."""
    for world in range(2, 9):
        per_round = world - 1
        for stagger in range(0, 4):
            for pieces in range(1, 6):
                items = p.ring_rounds(pieces, world, stagger) * per_round
                found = [p.ring_partial(item, world, pieces, stagger) for item in range(items)]
                full = [pair for pair in found if pair[0] is not None]
                assert sorted(full) == [(piece, kind) for piece in range(pieces) for kind in range(per_round)]
                assert len(found) - len(full) == (world - 2) * stagger * per_round
                for item, (piece, kind) in enumerate(found):
                    if piece is not None and kind < per_round - 1:
                        assert p.ring_partial(item + stagger * per_round + 1, world, pieces, stagger) == (piece,
                                                                                                         kind + 1)
            assert p.ring_stagger_slots(world, stagger) == (stagger * per_round + 2 if stagger else 2)
    assert p.ring_partial(0, 4, 3, 1) == (0, 0) and p.ring_partial(1, 4, 3, 1) == (None, 1)
    assert p.ring_partial(4, 4, 3, 1) == (0, 1) and p.RING_STAGGER_SHIFT == 8


def test_staggered_ring_forwards():
    """Link 3 of a ring all-gather with stagger D3: every forward (type r >= 1) passes on the inbound item of
    type r - 1 that carries its piece, D3 rounds after it, each inbound item but the last type forwarded
    once; the forwards of the first D3 rounds forward nothing and are empty."""
    for world in range(2, 9):
        per_round = world - 1
        for stagger in range(0, 4):
            for pieces in range(1, 6):
                items = p.ring_rounds(pieces, world, stagger) * per_round
                sources = []
                for item in range(items):
                    piece, kind = p.ring_partial(item, world, pieces, stagger)
                    source = p.ring_forward_source(item, world, stagger)
                    if kind == 0 or item // per_round < stagger:
                        assert source is None
                        assert kind == 0 or piece is None
                        continue
                    assert 0 <= source < item
                    assert p.ring_partial(source, world, pieces, stagger) == (piece, kind - 1)
                    sources.append(source)
                assert sorted(sources) == sources and len(set(sources)) == len(sources)
                assert set(sources) == {i for i in range(items) if i % per_round < per_round - 1
                                        and i + stagger * per_round + 1 < items}
    assert p.ring_op_word(p.LinkOp.RING_REDUCE, 1, 2) == 5 | 1 << 8 | 2 << 16 and p.RING_GATHER_STAGGER_SHIFT == 16
    for op, stagger, gather_stagger in ((p.LinkOp.RING_GATHER, 1, 0), (p.LinkOp.RING_SCATTER, 0, 1),
                                        (p.LinkOp.RING_REDUCE, 5, 0), (p.LinkOp.ALL_GATHER, 0, 1)):
        with pytest.raises(p.ProtocolError):
            p.ring_op_word(op, stagger, gather_stagger)
