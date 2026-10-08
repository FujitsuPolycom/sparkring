"""Blocks per role of the link kernels by group shape and kernel (protocol.link_blocks, torch-free)."""

from __future__ import annotations

import pytest

from sparkring_sircl import protocol as proto

FOUR = {kernel: 4 for kernel in proto.LINK_BLOCK_KERNELS}


def test_measured_shapes_take_one_block_for_their_measured_ring_kernels():
    pair = {**FOUR, "ring_reduce": 1, "ring_gather": 1, "ring_scatter": 1}
    assert proto.link_blocks("pair", 2) == pair
    assert proto.link_blocks(None, 2) == pair
    assert proto.link_blocks("path:4", 4) == {**FOUR, "ring_reduce": 1, "ring_gather": 1}


def test_unmeasured_kernels_and_shapes_take_the_default():
    for shape, world in (("path:3", 3), ("cycle:8", 8), ("cycle:4", 4), ("strided:cycle:8:0,2", 2), (None, 4)):
        assert proto.link_blocks(shape, world) == FOUR
    assert set(proto.LINK_BLOCK_KERNELS) == {"ring_reduce", "ring_gather", "ring_scatter", "chain_gather",
                                             "chain_scatter"}


def test_variables_override_the_shape_for_both_schedules_of_a_collective():
    assert proto.link_blocks("pair", 2, 3) == {kernel: 3 for kernel in proto.LINK_BLOCK_KERNELS}
    assert proto.link_blocks("pair", 2, 3, {"reduce": 2}) == {**{k: 3 for k in proto.LINK_BLOCK_KERNELS},
                                                             "ring_reduce": 2}
    assert proto.link_blocks("pair", 2, 0, {"gather": 2}) == {**FOUR, "ring_reduce": 1, "ring_gather": 2,
                                                             "chain_gather": 2, "ring_scatter": 1}
    assert proto.link_blocks("cycle:8", 8, 0, {"scatter": 1, "gather": 0}) == {**FOUR, "ring_scatter": 1,
                                                                              "chain_scatter": 1}
    for overall, own in ((65, {}), (0, {"reduce": 65}), (-1, {})):
        with pytest.raises(ValueError, match="1 to 64"):
            proto.link_blocks("pair", 2, overall, own)
