"""Described ops (op code 2) of the Swing all-reduce kernel through the real native layer.

Each case builds every rank of one group over the in-memory verbs stand-in,
plays the Swing kernel's part of the command ring with
``testing.collective_models.swing_op`` (stage, descriptors, phase doorbells,
namespace-0 reduce-scatter waits, namespace-1 all-gather waits) and compares
every rank's result with ``swing_reference``, which follows the schedule from
the protocol's peer and chunk-owner functions alone. Every rank must hold
identical bits.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparkring_sircl import routes
from sparkring_sircl import scatter_plan as sp
from sparkring_sircl import swing_plan
from sparkring_sircl.testing import collective_models as models
from sparkring_sircl.testing import fabric

# Power-of-two groups on cycles and paths, and subgroups of the ring of eight.
LAYOUTS = [
    ("ring:2", 2), ("ring:2", 1), ("ring:4", 2), ("ring:4", 1), ("path:0-3", 2), ("ring:8", 2), ("ring:8", 1),
    ("ring:8:0,2,4,6", 2), ("ring:8:0,1,2,3", 2), ("ring:8:0,4", 2),
]


def _run(session, seq, rng, dtype, nbytes):
    world = session.world
    item = np.dtype(models.storage(dtype)).itemsize
    inputs = [models.random_values(rng, dtype, nbytes // item) for _ in range(world)]
    outputs = models.swing_op(session, seq, inputs, dtype)
    want = models.swing_reference(inputs, dtype)
    for rank, got in enumerate(outputs):
        assert models.same_bits(got, want), f"rank {rank} of {world}, {dtype}, {nbytes} bytes"
    return inputs, want


@pytest.mark.parametrize("layout_text, lanes", LAYOUTS)
def test_swing_ops(simulator_library, layout_text, lanes):
    layout = routes.Layout.parse(layout_text)
    world = layout.world
    session = fabric.LocalSession(str(simulator_library), layout, lanes=lanes, slot_bytes=32768)
    rng = np.random.default_rng(world * 100 + lanes)
    seq = 1
    try:
        session.connect()
        # One pack (empty position ranges and flag-only lanes), a few packs, odd pack counts, a whole slot.
        for dtype in models.DTYPES:
            for nbytes in (16, 48, 4096, 16000, 32768):
                _run(session, seq, rng, dtype, nbytes)
                seq += 1
        stats = session.proxies[0].stats()
        ops = seq - 1
        assert stats["ops_posted"] == ops
        assert stats["later_phases_posted"] == ops * (swing_plan.phase_count(world) - 1)
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()


def test_swing_differs_from_the_rank_ordered_sum_only_by_rounding():
    """Swing rounds between steps; float32 inputs of exactly representable sums agree bit for bit."""
    rng = np.random.default_rng(3)
    world = 8
    integers = [rng.integers(-64, 64, 4096).astype(np.float32) for _ in range(world)]
    assert models.same_bits(models.swing_reference(integers, "float32"), models.rank_order_sum(integers, "float32"))
    values = [models.random_values(rng, "bfloat16", 4096) for _ in range(world)]
    swing = models.to_f32(models.swing_reference(values, "bfloat16"), "bfloat16")
    exact = models.to_f32(models.rank_order_sum(values, "bfloat16"), "bfloat16")
    assert np.max(np.abs(swing - exact)) <= 0.08 * (np.max(np.abs(exact)) + 1)


@pytest.mark.parametrize("layout_text", ["ring:8", "ring:4", "path:0-3"])
def test_mixed_swing_scatter_and_oneshot_ops(simulator_library, layout_text):
    """Described, scatter and one-shot ops interleave on one session without crossing flags."""
    layout = routes.Layout.parse(layout_text)
    world = layout.world
    session = fabric.LocalSession(str(simulator_library), layout, lanes=2, slot_bytes=16384)
    rng = np.random.default_rng(5)
    try:
        session.connect()
        seq = 1
        for round_ in range(6):
            _run(session, seq, rng, "bfloat16", 16 * (1 + 97 * round_))
            seq += 1
            geometry = sp.scatter_geometry(world * 512, world, 16384)
            inputs = [models.random_values(rng, "float32", world * 128) for _ in range(world)]
            outputs = models.scatter_op(session, seq, "reduce", inputs, "float32", geometry)
            for rank in range(world):
                assert models.same_bits(outputs[rank],
                                        models.reduce_scatter_reference(inputs, "float32", geometry, rank))
            seq += 1
            fabric.run_oneshot_ops(session, [16 * (round_ + 1)], first_seq=seq)
            seq += 1
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()
