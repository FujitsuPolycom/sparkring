"""Geometry and op plans of the scatter collectives (``sparkring_sircl.scatter_plan``)."""

from __future__ import annotations

import pytest

from sparkring_sircl import scatter_plan as sp


@pytest.mark.parametrize("world", [2, 3, 4, 6, 8])
def test_contiguous_chunks_split_the_input_evenly(world):
    total = world * 16 * 5
    geometry = sp.scatter_geometry(total, world, capacity=total)
    assert geometry == sp.ScatterGeometry(total // world, total // world)
    assert geometry.message_bytes(world) == total
    assert [geometry.chunk_offset(j) for j in range(world)] == [j * total // world for j in range(world)]
    # Not a multiple of W packs: declined, the caller routes the message elsewhere.
    assert sp.scatter_geometry(total + 16, world, capacity=4 * total) is None
    assert sp.scatter_geometry(world * 8, world, capacity=4 * total) is None


@pytest.mark.parametrize("world", [2, 3, 4, 6, 8])
def test_strided_chunks_follow_the_rules(world):
    chunk, stride = 64, 96
    total = (world - 1) * stride + chunk
    assert sp.scatter_geometry(total, world, 1 << 20, chunk, stride) == sp.ScatterGeometry(chunk, stride)
    # The default source stride is the chunk.
    assert sp.scatter_geometry(world * chunk, world, 1 << 20, chunk) == sp.ScatterGeometry(chunk, chunk)
    # The last chunk must lie inside the input.
    assert sp.scatter_geometry(total - 16, world, 1 << 20, chunk, stride) is None
    # Strides below the chunk, and sizes that are not whole packs, are declined.
    assert sp.scatter_geometry(total, world, 1 << 20, chunk, chunk - 16) is None
    assert sp.scatter_geometry(total, world, 1 << 20, chunk + 8, stride) is None
    assert sp.scatter_geometry(total, world, 1 << 20, chunk, stride + 8) is None
    assert sp.scatter_geometry(total, world, 1 << 20, 0, stride) is None


@pytest.mark.parametrize("world", [2, 4, 8])
def test_the_message_must_fit_the_capacity(world):
    chunk = 4096
    total = world * chunk
    assert sp.scatter_geometry(total, world, capacity=world * chunk) is not None
    assert sp.scatter_geometry(total, world, capacity=world * chunk - 16) is None
    assert sp.scatter_geometry(2 * total, world, world * chunk, chunk, 2 * chunk) is not None
    assert sp.scatter_geometry(2 * total, world, world * chunk - 16, chunk, 2 * chunk) is None


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_without_a_capacity_every_size_is_eligible(world):
    total = world * (64 << 20) // 4
    assert sp.scatter_geometry(total, world) == sp.ScatterGeometry(total // world, total // world)
    assert sp.scatter_geometry(total, world, None, 4096, total // world) == sp.ScatterGeometry(4096, total // world)
    assert sp.scatter_geometry(total, world, 1 << 20) is None


def test_large_messages_travel_in_ops_of_the_large_piece():
    # A 64 MiB BF16 reduce-scatter over four ranks with 4 MiB ops: 16 ops of 1 MiB per peer.
    world, total, op = 4, 64 << 20, 4 << 20
    geometry = sp.scatter_geometry(total, world)
    per_peer = sp.op_peer_bytes(op, world)
    assert per_peer == 1 << 20
    piece = sp.piece_bytes(geometry.chunk_bytes, None, per_peer)
    plan = sp.scatter_plan(geometry.chunk_bytes, piece)
    assert len(plan) == 16 and all(part.nbytes == 1 << 20 for part in plan)
    assert all(world * part.nbytes <= op for part in plan)
    # The relay-safe size caps the pieces further when it is smaller.
    assert sp.piece_bytes(geometry.chunk_bytes, 131072, per_peer) == 131072
    assert sp.piece_bytes(4096, 131072, per_peer) == 4096
    # Ops of three ranks round the per-peer piece down to whole packs.
    assert sp.op_peer_bytes(1 << 20, 3) == (1 << 20) // 3 // 16 * 16
    with pytest.raises(sp.ScatterError, match="cannot carry"):
        sp.op_peer_bytes(48, 4)
    with pytest.raises(sp.ScatterError, match="op limit"):
        sp.piece_bytes(4096, None, 8)


def test_a_single_rank_or_an_empty_input_is_declined():
    assert sp.scatter_geometry(64, 1, 1 << 20) is None
    assert sp.scatter_geometry(0, 4, 1 << 20) is None


def test_destination_strides():
    assert sp.destination_stride(64, 4, 4 * 64) == 64
    assert sp.destination_stride(64, 4, 3 * 80 + 64, 80) == 80
    with pytest.raises(sp.ScatterError, match="at least"):
        sp.destination_stride(64, 4, 1024, 48)
    with pytest.raises(sp.ScatterError, match="multiple of 16"):
        sp.destination_stride(64, 4, 1024, 72)
    with pytest.raises(sp.ScatterError, match="cannot hold"):
        sp.destination_stride(64, 4, 3 * 80 + 48, 80)


def test_piece_bytes_follow_the_relay_safe_size():
    assert sp.piece_bytes(4096, None) == 4096
    assert sp.piece_bytes(4096, 4096) == 4096
    assert sp.piece_bytes(4096, 8192) == 4096
    assert sp.piece_bytes(4096, 1000) == 992
    with pytest.raises(sp.ScatterError, match="below one"):
        sp.piece_bytes(4096, 8)


@pytest.mark.parametrize("chunk, piece", [(16, None), (16, 16), (4096, 1024), (4096, 4096), (4112, 1024),
                                          (262144, 131072), (48, 32)])
def test_plans_cover_every_chunk_byte_once(chunk, piece):
    plan = sp.scatter_plan(chunk, piece)
    offset = 0
    for part in plan:
        assert part.offset == offset and 16 <= part.nbytes <= (piece or chunk) and part.nbytes % 16 == 0
        offset += part.nbytes
    assert offset == chunk
    assert len(plan) == -(-chunk // (piece or chunk))


def test_plans_refuse_partial_packs():
    with pytest.raises(sp.ScatterError):
        sp.scatter_plan(40)
    with pytest.raises(sp.ScatterError):
        sp.scatter_plan(64, 24)
