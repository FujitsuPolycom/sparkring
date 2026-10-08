"""Geometry and reference arithmetic of the fused all-reduce + residual add + RMSNorm (torch-free)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sparkring_sircl import protocol as proto
from sparkring_sircl.fused_norm import _geometry as geo
from sparkring_sircl.fused_norm import _reference as ref
from sparkring_sircl.testing import collective_models as models
from vectors import load

# -- geometry -----------------------------------------------------------------------------


def test_geometry_equals_the_vectors_and_the_protocol():
    numeric = load("numeric.json")
    for key, expected in numeric["stripe"].items():
        fields = dict(item.split("=") for item in key.split(","))
        count, lanes = int(fields["count"]), int(fields["lanes"])
        for lane, (first, length) in enumerate(expected):
            assert geo.stripe_bounds(0, count, lanes, lane) == (first, first + length)
    for key, expected in numeric["chunk"].items():
        fields = dict(item.split("=") for item in key.split(","))
        packs, world = int(fields["packs"]), int(fields["world"])
        for owner, (first, length) in enumerate(expected):
            assert geo.chunk_bounds(packs, world, owner) == (first, first + length)
    for key, expected in numeric["flag_index"].items():
        fields = {name: int(value) for name, value in (item.split("=") for item in key.split(","))}
        assert geo.flag_index(fields["ns"], fields["src"], fields["slot"], fields["lane"], fields["world"],
                              proto.SLOTS, fields["lanes"]) == expected


def test_geometry_equals_the_protocol_on_small_cases():
    for count in range(0, 257):
        for lanes in (1, 2):
            for lane in range(lanes):
                first, length = proto.stripe(count, lanes, lane)
                assert geo.stripe_bounds(5, count, lanes, lane) == (5 + first, 5 + first + length)
    for packs in range(1, 257):
        for world in range(2, 9):
            for owner in range(world):
                first, length = proto.chunk(packs, world, owner)
                assert geo.chunk_bounds(packs, world, owner) == (first, first + length)
    for world, lanes, ns, source, slot, lane in itertools.product((2, 3, 8), (1, 2), (0, 1), range(3), (0, 1), (0, 1)):
        if source < world and lane < lanes:
            assert (geo.flag_index(ns, source, slot, lane, world, proto.SLOTS, lanes)
                    == proto.flag_index(ns, source, slot, lane, world, lanes))


def _lane_of(start: int, count: int, lanes: int, pack: int) -> int:
    for lane in range(lanes):
        lo, hi = geo.stripe_bounds(start, count, lanes, lane)
        if lo <= pack < hi:
            return lane
    raise AssertionError(f"pack {pack} is in no stripe of [{start}, {start + count})")


@pytest.mark.parametrize("world", range(2, 9))
@pytest.mark.parametrize("lanes", [1, 2])
def test_waits_are_exactly_the_stripes_a_row_reads(world, lanes):
    """Every row waits for every stripe that holds one of the packs it reads, and for no other."""
    for row_packs in (32, 96):
        for rows in (1, 2, 3, 5):
            packs = rows * row_packs
            if packs % world:
                continue
            geo.check_launch_geometry(rows, row_packs, world, lanes)
            for rank in range(world):
                own_lo, own_hi = geo.chunk_bounds(packs, world, rank)
                for row in range(rows):
                    r_lo, r_hi = geo.row_bounds(row, row_packs)
                    # One-shot: every peer's stripe of the whole message that overlaps the row.
                    need = {(peer, _lane_of(0, packs, lanes, pack)) for peer in range(world) if peer != rank
                            for pack in range(r_lo, r_hi)}
                    waited = {(peer, lane) for peer in range(world) for lane in range(lanes)
                              if geo.oneshot_wait(row, row_packs, packs, peer, lane, rank, lanes)}
                    assert waited == need
                    # Two-shot scatter: the stripes of the own chunk that overlap the row.
                    need = {(peer, _lane_of(own_lo, own_hi - own_lo, lanes, pack)) for peer in range(world)
                            if peer != rank for pack in range(max(r_lo, own_lo), min(r_hi, own_hi))}
                    waited = {(peer, lane) for peer in range(world) for lane in range(lanes)
                              if geo.twoshot_scatter_wait(row, row_packs, packs, peer, lane, rank, world, lanes)}
                    assert waited == need
                    # Two-shot gather: every other owner's chunk stripe that overlaps the row.
                    need = set()
                    for pack in range(r_lo, r_hi):
                        owner = pack // (packs // world)
                        lo, hi = geo.chunk_bounds(packs, world, owner)
                        assert lo <= pack < hi
                        if owner != rank:
                            need.add((owner, _lane_of(lo, hi - lo, lanes, pack)))
                    waited = {(owner, lane) for owner in range(world) for lane in range(lanes)
                              if geo.twoshot_gather_wait(row, row_packs, packs, owner, lane, rank, world, lanes)}
                    assert waited == need


def test_launch_geometry_refusals():
    with pytest.raises(ValueError, match="whole warps"):
        geo.check_launch_geometry(1, 40, 2, 1)
    with pytest.raises(ValueError, match="equal two-shot chunks"):
        geo.check_launch_geometry(1, 32, 3, 1)
    with pytest.raises(ValueError, match="one or two lanes"):
        geo.check_launch_geometry(1, 32, 2, 3)
    with pytest.raises(ValueError, match="flag waiters"):
        geo.check_launch_geometry(2, 32, 32, 2)


# -- reference arithmetic -----------------------------------------------------------------


def test_bf16_rounding_is_round_to_nearest_even():
    values = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5, np.nan], dtype=np.float32)
    bits = ref.f32_to_bf16(values)
    assert list(bits[:4]) == [0x3F80, 0x3F80, 0x3F82, 0xC020]
    assert bits[4] == 0x7FFF
    assert models.same_bits(ref.bf16_to_f32(bits[:4]), np.array([1.0, 1.0, 1.0 + 2 ** -6, -2.5], dtype=np.float32))


def test_rank_order_sum_agrees_with_the_collective_reference():
    rng = np.random.default_rng(2)
    partials = [ref.random_bf16(rng, (3, 256)) for _ in range(8)]
    assert models.same_bits(ref.rank_order_sum(partials).reshape(-1),
                            models.rank_order_sum([p.reshape(-1) for p in partials], "bfloat16"))
    # The order is part of the result: 1 added to 2^25 is lost in float32, added to 0 it is not.
    big = ref.f32_to_bf16(np.full((1, 8), 2.0 ** 25, dtype=np.float32))
    small = ref.f32_to_bf16(np.full((1, 8), 1.0, dtype=np.float32))
    neg = ref.f32_to_bf16(np.full((1, 8), -(2.0 ** 25), dtype=np.float32))
    assert not models.same_bits(ref.rank_order_sum([big, small, small, neg]),
                                ref.rank_order_sum([big, neg, small, small]))


def test_sum_of_squares_follows_the_block_reduction_order():
    rng = np.random.default_rng(4)
    z = ref.random_bf16(rng, (4, 6144), scale=3.0)
    got = ref.block_sum_of_squares(z)
    values = ref.bf16_to_f32(z).astype(np.float32)
    # The same order written out: per-thread pack sums, a pairwise tree per warp, warps in order.
    for row in range(4):
        squares = values[row].reshape(-1, 8) ** 2
        threads = [np.float32(np.float32(np.float32(np.float32(s[0] + s[1]) + np.float32(s[2] + s[3]))
                                         + np.float32(s[4] + s[5])) + np.float32(s[6] + s[7])) for s in squares]
        threads += [np.float32(0.0)] * (1024 - len(threads))
        warps = []
        for w in range(32):
            lane = threads[32 * w:32 * w + 32]
            while len(lane) > 1:
                lane = [np.float32(lane[2 * i] + lane[2 * i + 1]) for i in range(len(lane) // 2)]
            warps.append(lane[0])
        total = warps[0]
        for w in range(1, 32):
            total = np.float32(total + warps[w])
        assert np.float32(got[row]) == total
        assert abs(float(got[row]) - float(np.sum(values[row].astype(np.float64) ** 2))) <= 1e-4 * float(got[row])


def test_fused_equals_the_unfused_operation_on_the_reduced_message():
    rng = np.random.default_rng(6)
    partials = [ref.random_bf16(rng, (5, 1024)) for _ in range(4)]
    residual = ref.random_bf16(rng, (5, 1024))
    weight = ref.random_bf16(rng, (1024,), scale=0.5)
    normed, z = ref.fused_add_rms_norm(partials, residual, weight, 1e-5)
    normed2, z2 = ref.unfused_reference(ref.rank_order_sum(partials), residual, weight, 1e-5)
    assert models.same_bits(normed, normed2) and models.same_bits(z, z2)
    # The new residual is the BF16 sum of the reduced message and the residual.
    exact = ref.bf16_to_f32(ref.rank_order_sum(partials)) + ref.bf16_to_f32(residual)
    assert models.same_bits(z, ref.f32_to_bf16(exact.astype(np.float32)))
    # Normalized rows have a root mean square close to |weight| scaling.
    rms = np.sqrt(np.mean((ref.bf16_to_f32(normed) / ref.bf16_to_f32(weight)) ** 2, axis=1))
    assert np.all(np.abs(rms - 1.0) < 0.02)


def test_rows_beyond_one_pack_per_thread_are_refused():
    with pytest.raises(ValueError):
        ref.block_sum_of_squares(np.zeros((1, 8 * 1056), dtype=np.uint16))
