"""Scatter ops (op code 3) of the reduce-scatter and all-to-all kernels through the real native layer.

Each case builds every rank of one group over the in-memory verbs stand-in
(``testing.fabric.LocalSession``), plays the scatter kernel's part of the
command ring with ``testing.collective_models.scatter_op`` and compares every
rank's result with the numpy reference: chunk ``rank`` of the rank-ordered
float32 sum, rounded once (reduce-scatter), or the chunks of every source
unchanged (all-to-all).
"""

from __future__ import annotations

import numpy as np
import pytest

from sparkring_sircl import routes
from sparkring_sircl import scatter_plan as sp
from sparkring_sircl.testing import collective_models as models
from sparkring_sircl.testing import fabric

# Groups of 2, 3, 4, 6 and 8 ranks on cycles and paths, and decode-context-parallel
# subgroups of the ring of eight, adjacent and not (members relay through others).
LAYOUTS = [
    ("ring:2", 2), ("ring:2", 1), ("ring:3", 2), ("ring:4", 1), ("path:0-3", 2), ("ring:6", 2), ("ring:8", 2),
    ("ring:8:0,1,2,3", 2), ("ring:8:4,5,6,7", 2), ("ring:8:0,2,4,6", 2), ("ring:8:0,2,4,6", 1),
    ("ring:8:0,4", 2), ("ring:8:1,3,5,7", 2),
]


def _inputs(rng, world, dtype, count):
    return [models.random_values(rng, dtype, count) for _ in range(world)]


def _check_reduce(outputs, inputs, dtype, geometry):
    for rank, got in enumerate(outputs):
        want = models.reduce_scatter_reference(inputs, dtype, geometry, rank)
        assert models.same_bits(got, want), f"rank {rank}"


def _check_copy(outputs, inputs, geometry):
    for rank, got in enumerate(outputs):
        for source, (chunk, want) in enumerate(zip(got, models.all_to_all_reference(inputs, geometry, rank))):
            assert models.same_bits(chunk, want), f"rank {rank} source {source}"


@pytest.mark.parametrize("layout_text, lanes", LAYOUTS)
def test_reduce_scatter_and_all_to_all(simulator_library, layout_text, lanes):
    layout = routes.Layout.parse(layout_text)
    world = layout.world
    session = fabric.LocalSession(str(simulator_library), layout, lanes=lanes, slot_bytes=65536)
    rng = np.random.default_rng(world * 10 + lanes)
    seq = 1
    try:
        session.connect()
        for dtype in models.DTYPES:
            item = np.dtype(models.storage(dtype)).itemsize
            # Contiguous chunks of one pack, of an odd pack count and of a large size.
            for chunk in (16, 16 * 37, 4096):
                inputs = _inputs(rng, world, dtype, world * chunk // item)
                geometry = sp.scatter_geometry(world * chunk, world, 65536)
                _check_reduce(models.scatter_op(session, seq, "reduce", inputs, dtype, geometry), inputs, dtype,
                              geometry)
                seq += 1
            # Strided chunks: a sub-range of every source row, as for heads of a [W * H, rows, D] tensor.
            chunk, stride = 96, 160
            inputs = _inputs(rng, world, dtype, ((world - 1) * stride + chunk + 32) // item)
            geometry = sp.scatter_geometry(inputs[0].nbytes, world, 65536, chunk, stride)
            _check_reduce(models.scatter_op(session, seq, "reduce", inputs, dtype, geometry), inputs, dtype,
                          geometry)
            seq += 1
        for chunk, stride in ((16, 16), (272, 272), (64, 112)):
            inputs = [rng.integers(0, 256, (world - 1) * stride + chunk, dtype=np.uint8) for _ in range(world)]
            geometry = sp.scatter_geometry(inputs[0].nbytes, world, 65536, chunk, stride)
            outputs = models.scatter_op(session, seq, "copy", inputs, "float32", geometry)
            _check_copy([[c.view(np.uint8) for c in out] for out in outputs], inputs, geometry)
            seq += 1
        stats = session.proxies[0].stats()
        assert stats["ops_posted"] == seq - 1 and stats["later_phases_posted"] == 0
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()


@pytest.mark.parametrize("layout_text", ["ring:8", "ring:8:0,2,4,6", "path:0-3"])
def test_split_messages_give_the_same_bits(simulator_library, layout_text):
    """A message whose chunks exceed the per-peer op size travels as several strided ops."""
    layout = routes.Layout.parse(layout_text)
    world = layout.world
    session = fabric.LocalSession(str(simulator_library), layout, lanes=2, slot_bytes=32768)
    rng = np.random.default_rng(7)
    try:
        session.connect()
        chunk, stride = 3 * 1024 + 48, 4096
        seq = 1
        for dtype in ("bfloat16", "float32"):
            item = np.dtype(models.storage(dtype)).itemsize
            inputs = _inputs(rng, world, dtype, ((world - 1) * stride + chunk) // item)
            geometry = sp.scatter_geometry(inputs[0].nbytes, world, 1 << 20, chunk, stride)
            piece = sp.piece_bytes(chunk, 1000)
            whole, seq = models.scatter_message(session, seq, "reduce", inputs, dtype, geometry)
            split, seq = models.scatter_message(session, seq, "reduce", inputs, dtype, geometry, piece)
            for rank in range(world):
                want = models.reduce_scatter_reference(inputs, dtype, geometry, rank)
                assert models.same_bits(whole[rank], want) and models.same_bits(split[rank], want)
        raw = [rng.integers(0, 256, (world - 1) * stride + chunk, dtype=np.uint8) for _ in range(world)]
        geometry = sp.scatter_geometry(raw[0].nbytes, world, 1 << 20, chunk, stride)
        split, seq = models.scatter_message(session, seq, "copy", [r.view(np.float32) for r in raw], "float32",
                                            geometry, 512)
        for rank in range(world):
            for source in range(world):
                assert split[rank][source].tobytes() == models.chunk_of(raw[source], geometry, rank).tobytes()
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()


@pytest.mark.parametrize("layout_text", ["path:0-3", "ring:4", "ring:3"])
def test_messages_far_above_the_slot_travel_in_slot_sized_ops(simulator_library, layout_text):
    """A reduce-scatter of several slots per peer: ops of at most one slot, same bits as the reference."""
    layout = routes.Layout.parse(layout_text)
    world = layout.world
    slot = 16384
    session = fabric.LocalSession(str(simulator_library), layout, lanes=2, slot_bytes=slot)
    rng = np.random.default_rng(13)
    try:
        session.connect()
        chunk = 5 * slot + 4096          # every chunk exceeds one slot on its own
        inputs = _inputs(rng, world, "bfloat16", world * chunk // 2)
        geometry = sp.scatter_geometry(world * chunk, world)
        piece = sp.piece_bytes(chunk, None, sp.op_peer_bytes(slot, world))
        assert world * piece <= slot
        outputs, seq = models.scatter_message(session, 1, "reduce", inputs, "bfloat16", geometry, piece)
        _check_reduce(outputs, inputs, "bfloat16", geometry)
        assert session.proxies[0].stats()["ops_posted"] == seq - 1 == len(sp.scatter_plan(chunk, piece))
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()


def test_forward_windows_pace_relayed_scatter_lanes(simulator_library):
    """On a path of four, lanes through relays keep at most their window unacknowledged."""
    layout = routes.Layout.parse("path:0-3")
    session = fabric.LocalSession(str(simulator_library), layout, lanes=2, slot_bytes=65536,
                                  forward_window=16384, forward_chunk=4096)
    rng = np.random.default_rng(11)
    try:
        session.connect()
        geometry = sp.scatter_geometry(65536, 4, 65536)
        inputs = _inputs(rng, 4, "bfloat16", 65536 // 2)
        _check_reduce(models.scatter_op(session, 1, "reduce", inputs, "bfloat16", geometry), inputs, "bfloat16",
                      geometry)
        stats = session.proxies[0].stats()
        assert stats["forward_chunks_posted"] > 0
        assert 0 < stats["forward_max_unacked_bytes"] <= 16384 + 4
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()
