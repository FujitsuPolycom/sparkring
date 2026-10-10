"""Piece plans of the large-message collectives, checked by emulating the ops on byte arrays."""

from __future__ import annotations

import random

import pytest

from sparkring_sircl import pieces


@pytest.mark.parametrize("total, piece", [
    (0, 16), (16, 16), (48, 32), (100, 32), (2 << 20, 2 << 20), ((64 << 20) + 6, 2 << 20), (14, 4096),
])
def test_reduce_pieces_cover_the_message_once(total, piece):
    plan = pieces.reduce_plan(total, piece)
    covered = 0
    for item in plan:
        assert item.offset == covered
        assert 0 < item.nbytes <= piece
        if item.padded:
            assert item is plan[-1] and item.nbytes < 16
        else:
            assert item.nbytes % 16 == 0
        covered += item.nbytes
    assert covered == total


def test_reduce_pieces_refuse_unusable_sizes():
    for piece in (0, 8, 24):
        with pytest.raises(pieces.PieceError):
            pieces.reduce_plan(64, piece)


def _emulate_gather(shards: list[bytes], outer: int, inner: int, piece: int) -> bytes:
    """Run the plan's ops the way all_gather_large moves bytes."""
    world = len(shards)
    output = bytearray(outer * world * inner)
    for tile in pieces.gather_plan(outer, inner, piece):
        assert tile.nbytes <= piece
        assert tile.rows == 1 or tile.cols == inner          # the source bytes are contiguous
        for source, shard in enumerate(shards):
            for row in range(tile.rows):
                start = (tile.row + row) * inner + tile.col
                destination = tile.output_offset(inner, world, source) + row * world * inner
                output[destination:destination + tile.cols] = shard[start:start + tile.cols]
    return bytes(output)


@pytest.mark.parametrize("shape, dim, itemsize, piece", [
    ((1024,), 0, 2, 256),             # flat shard, one row in column tiles
    ((64, 32), 1, 2, 256),            # rows of 64 bytes, four per op
    ((64, 32), 0, 2, 256),            # whole shard is one row
    ((5, 7, 3), 1, 2, 48),            # rows of 42 bytes (padded path), one per op
    ((3, 1000), -1, 4, 512),          # rows longer than a piece: column tiles
    ((2, 3, 4, 5), 2, 1, 16),
])
def test_gather_plan_reproduces_the_concatenation(shape, dim, itemsize, piece):
    world = 3
    rng = random.Random(7)
    outer, inner = pieces.gather_view(shape, dim, itemsize)
    shards = [bytes(rng.randrange(256) for _ in range(outer * inner)) for _ in range(world)]
    expected = b"".join(b"".join(shard[row * inner:(row + 1) * inner] for shard in shards) for row in range(outer))
    assert _emulate_gather(shards, outer, inner, piece) == expected


def test_gather_view_and_padding():
    assert pieces.gather_view((4, 6, 8), 0, 2) == (1, 384)
    assert pieces.gather_view((4, 6, 8), -1, 2) == (24, 16)
    assert pieces.gather_view((4, 6, 8), 1, 4) == (4, 192)
    assert pieces.padded(1) == 16 and pieces.padded(32) == 32 and pieces.padded(33) == 48
    assert pieces.gather_plan(0, 64, 64) == () and pieces.gather_plan(4, 0, 64) == ()
    with pytest.raises(pieces.PieceError):
        pieces.gather_view((), 0, 2)


def test_relay_safe_bytes_follow_the_queue_rule():
    # Ring of eight, two lanes: six lane paths through the busiest queue, load factor 3.
    assert pieces.relay_safe_bytes(6, 2, 512 << 10, 0.75) == 128 << 10
    # Path of four, two lanes: two lane paths, load factor 1.
    assert pieces.relay_safe_bytes(2, 2, 512 << 10, 0.75) == 384 << 10
    assert pieces.relay_safe_bytes(0, 2, 512 << 10, 0.75) is None


def test_reduce_plan_with_a_chain_op():
    plan = pieces.reduce_plan((64 << 20) + 6, 4 << 20, chain_from=2 << 20)
    assert plan == (pieces.ReducePiece(0, 64 << 20, chain=True), pieces.ReducePiece(64 << 20, 6, padded=True))
    small = pieces.reduce_plan(1 << 20, 4 << 20, chain_from=2 << 20)
    assert small == (pieces.ReducePiece(0, 1 << 20),)
    assert pieces.reduce_plan(48, 4 << 20, chain_from=16) == (pieces.ReducePiece(0, 48, chain=True),)
    assert pieces.reduce_plan(14, 4 << 20, chain_from=16) == (pieces.ReducePiece(0, 14, padded=True),)
    assert pieces.chain_reference_halves(48) == (16, 32) and pieces.chain_reference_halves(64) == (32, 32)


def test_references_follow_the_chain_order():
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - a torch build for another platform raises OSError
        pytest.skip(f"torch is unavailable: {error}")
    from sparkring_sircl import references

    generator = torch.Generator().manual_seed(5)
    inputs = [torch.randn(48, generator=generator).to(torch.bfloat16) for _ in range(4)]
    order = (0, 1, 2, 3)
    chained = references.chain_sum(torch, inputs, order)
    a = inputs[0][:24].clone()
    for rank in (1, 2, 3):
        a = (a.float() + inputs[rank][:24].float()).to(torch.bfloat16)
    b = inputs[3][24:].clone()
    for rank in (2, 1, 0):
        b = (b.float() + inputs[rank][24:].float()).to(torch.bfloat16)
    assert torch.equal(chained.view(torch.int16), torch.cat([a, b]).view(torch.int16))
    plan = pieces.reduce_plan(96, 4 << 20, chain_from=16)
    assert torch.equal(references.large_all_reduce(torch, inputs, plan, order).view(torch.int16),
                       chained.view(torch.int16))
    flat = references.large_all_reduce(torch, inputs, pieces.reduce_plan(96, 32), None)
    assert torch.equal(flat.view(torch.int16), references.rank_order_sum(torch, inputs).view(torch.int16))


def test_chain_reduce_scatter_reference_sums_from_both_ends():
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - a torch build for another platform raises OSError
        pytest.skip(f"torch is unavailable: {error}")
    from sparkring_sircl import references

    generator = torch.Generator().manual_seed(9)
    inputs = [torch.randn(4 * 6, generator=generator).to(torch.bfloat16) for _ in range(4)]
    order = (2, 0, 3, 1)                       # rank 2 at chain index 0, rank 0 at index 1, ...

    def chunk(rank, owner):
        return inputs[rank][owner * 6:(owner + 1) * 6]

    def fold(owner, ranks):
        total = chunk(ranks[0], owner).clone()
        for rank in ranks[1:]:
            total = (total.float() + chunk(rank, owner).float()).to(torch.bfloat16)
        return total

    outputs = references.chain_reduce_scatter(torch, inputs, order)
    # Rank 0 sits at chain index 1: L from rank 2, R folded from rank 1 then rank 3.
    want = ((fold(0, [2]).float() + chunk(0, 0).float()) + fold(0, [1, 3]).float()).to(torch.bfloat16)
    assert torch.equal(outputs[0].view(torch.int16), want.view(torch.int16))
    # Rank 2 is the first chain index: x + R, R folded from the far end.
    want = (chunk(2, 2).float() + fold(2, [1, 3, 0]).float()).to(torch.bfloat16)
    assert torch.equal(outputs[2].view(torch.int16), want.view(torch.int16))
    # Rank 1 is the last chain index: L + x, L folded from the first.
    want = (fold(1, [2, 0, 3]).float() + chunk(1, 1).float()).to(torch.bfloat16)
    assert torch.equal(outputs[1].view(torch.int16), want.view(torch.int16))
    strided = references.chain_reduce_scatter(torch, inputs, order, chunk_elements=4, stride_elements=6)
    assert [part.numel() for part in strided] == [4] * 4
    assert torch.equal(strided[1].view(torch.int16), outputs[1][:4].view(torch.int16))


def test_ring_references_fold_once_around_the_ring():
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - a torch build for another platform raises OSError
        pytest.skip(f"torch is unavailable: {error}")
    from sparkring_sircl import references

    generator = torch.Generator().manual_seed(11)
    inputs = [torch.randn(4 * 5, generator=generator).to(torch.bfloat16) for _ in range(4)]
    order = (1, 3, 0, 2)
    outputs = references.ring_reduce_scatter(torch, inputs, order)
    # Rank 0 sits at ring index 2: its partial starts at index 3 (rank 2), then ranks 1 and 3, then rank 0.
    chunk = [tensor[0:5] for tensor in inputs]
    total = chunk[2].clone()
    for rank in (1, 3, 0):
        total = (total.float() + chunk[rank].float()).to(torch.bfloat16)
    assert torch.equal(outputs[0].view(torch.int16), total.view(torch.int16))
    reduced = references.ring_all_reduce(torch, inputs, order)
    assert torch.equal(reduced.view(torch.int16), torch.cat(outputs).view(torch.int16))


def test_a_ring_plan_covers_whole_packs_per_rank_then_pieces():
    plan = pieces.reduce_plan(4 * 16 * 3 + 32 + 6, 4096, ring_from=64, ring_world=4)
    assert plan == (pieces.ReducePiece(0, 192, ring=True), pieces.ReducePiece(192, 32),
                    pieces.ReducePiece(224, 6, padded=True))
    assert pieces.reduce_plan(48, 4096, ring_from=64, ring_world=4) == (pieces.ReducePiece(0, 48),)
    assert pieces.reduce_plan(1 << 20, 4096, 16, ring_from=64, ring_world=4) == (
        pieces.ReducePiece(0, 1 << 20, ring=True),)
    with pytest.raises(pieces.PieceError):
        pieces.reduce_plan(256, 4096, ring_from=64)
