"""The fused all-reduce + residual add + RMSNorm through the real native layer.

Each case plays the fused kernels' part of the command ring
(``testing.collective_models.fused_norm_op``: op code 0 or 1, per-row flag
waits from ``fused_norm._geometry``, the gather doorbell of the two-shot
algorithm) for every rank of a group over the in-memory verbs stand-in, with
some ranks playing the plain all-reduce instead, and compares every rank's
result with ``fused_norm._reference``: plain ranks hold the rank-ordered sum,
fused ranks the normalized rows and the new residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparkring_sircl import routes
from sparkring_sircl.fused_norm import _reference as ref
from sparkring_sircl.testing import collective_models as models
from sparkring_sircl.testing import fabric

CASES = [
    ("ring:2", 2, 256), ("ring:2", 1, 256), ("ring:3", 2, 768), ("ring:4", 2, 512), ("path:0-3", 1, 256),
    ("ring:6", 2, 768), ("ring:8", 2, 256), ("ring:8", 1, 6144),
]


@pytest.mark.parametrize("layout_text, lanes, hidden", CASES)
@pytest.mark.parametrize("algorithm", ["oneshot", "twoshot"])
def test_fused_norm_ops(simulator_library, layout_text, lanes, hidden, algorithm):
    layout = routes.Layout.parse(layout_text)
    world = layout.world
    session = fabric.LocalSession(str(simulator_library), layout, lanes=lanes, slot_bytes=262144)
    rng = np.random.default_rng(world * 31 + lanes + hidden)
    eps = 1e-6
    seq = 1
    try:
        session.connect()
        for rows, plain in ((1, ()), (3, (1,)), (4, tuple(range(0, world, 2)))):
            partials = [ref.random_bf16(rng, (rows, hidden)) for _ in range(world)]
            residual = ref.random_bf16(rng, (rows, hidden))
            weight = ref.random_bf16(rng, (hidden,), scale=0.7)
            results = models.fused_norm_op(session, seq, algorithm, partials, [residual] * world, weight, eps,
                                           plain=plain)
            seq += 1
            normed, z = ref.fused_add_rms_norm(partials, residual, weight, eps)
            reduced = ref.rank_order_sum(partials)
            for rank, result in enumerate(results):
                if rank in plain:
                    assert models.same_bits(result, reduced), f"plain rank {rank}"
                else:
                    assert models.same_bits(result[0], normed) and models.same_bits(result[1], z), f"rank {rank}"
        stats = session.proxies[0].stats()
        assert stats["ops_posted"] == seq - 1
        assert stats["later_phases_posted"] == (seq - 1 if algorithm == "twoshot" else 0)
        assert not any(proxy.failed() for proxy in session.proxies)
    finally:
        session.close()
