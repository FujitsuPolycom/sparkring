"""The owned-row forward equals the image's full-row forward on simulated ranks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fakes
import tp_world
from ds41_row_shard import core
from ds41_row_shard.local_query import _full_weight, _gather_kv, _gather_projection_records

HIDDEN = 8
LAYERS = 7
BOUNDARY = 4  # the CED decoder start
ENGRAM = {1: 0, 3: 1}  # layer -> hash column, as V4.1's Engram layers 1 and 14


@pytest.mark.parametrize("axis", [0, 1])
def test_full_query_weights_reassemble_rank_shards_bit_for_bit(axis):
    original = torch.arange(64, dtype=torch.uint8).view(8, 8).view(torch.float8_e4m3fn)

    def run_rank(rank):
        shard = original.chunk(4, dim=axis)[rank]
        return _full_weight(tp_world.group(), shard, axis).view(torch.uint8)

    outputs, _ = tp_world.run(4, run_rank)
    assert all(torch.equal(output, original.view(torch.uint8)) for output in outputs)


def test_local_query_kv_gather_writes_every_rows_latent():
    rows, world = 16, 4
    expected = torch.arange(rows * 2).reshape(rows, 2)
    attention = SimpleNamespace(_ds41_query_replica=object())

    def run_rank(rank):
        geometry = core.row_geometry(rows, world, rank, min_owned=1)
        own = core.Ownership(
            geometry, tp_world.group(), None, {}, BOUNDARY,
            local_queries=True, gather_kv_latents=True,
        )
        core.STATE.ownership = own
        try:
            with own.unreduced_attention():
                result = _gather_kv(attention, own.local(expected))
            assert own.kv_latent_gathers == own.all_gathers == 1
            return result
        finally:
            core.STATE.ownership = None

    outputs, group = tp_world.run(world, run_rank)
    assert all(torch.equal(output, expected) for output in outputs)
    assert group.calls == [("all_gather", (4, 2))]


def test_compressor_record_gather_preserves_mixed_precision_bits():
    rows, world = 16, 4
    bf16 = torch.arange(rows * 2, dtype=torch.int16).reshape(rows, 2).view(torch.bfloat16)
    fp32 = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3).view(torch.float32)
    inputs = (bf16, fp32)

    def run_rank(rank):
        geometry = core.row_geometry(rows, world, rank, min_owned=1)
        own = core.Ownership(geometry, tp_world.group(), None, {}, BOUNDARY)
        output = _gather_projection_records(own, tuple(own.local(tensor) for tensor in inputs))
        return all(torch.equal(got.view(torch.uint8), want.view(torch.uint8))
                   for got, want in zip(output, inputs, strict=True))

    outputs, group = tp_world.run(world, run_rank)
    assert all(outputs)
    assert group.calls == [("all_gather", (4, 16))]


def test_engram_feature_bands_preserve_reduce_scattered_rows():
    rows, world = 16, 4
    full = torch.arange(rows * 24, dtype=torch.float32).reshape(rows, 24)

    def run_rank(rank):
        geometry = core.row_geometry(rows, world, rank, min_owned=1)
        own = core.Ownership(geometry, tp_world.group(), None, {}, BOUNDARY, band_engram=True)
        partial = full * (rank + 1)
        output = core.reduce_engram_lookup(own, partial, rows * 4 * 8)
        assert own.engram_reduce_scatters == own.reduce_scatters == 3
        return output

    outputs, group = tp_world.run(world, run_rank)
    assert torch.equal(torch.cat(outputs), full * 10)
    assert group.calls == [("reduce_scatter", (16, 8))] * 3


def test_compact_routes_matches_full_selection_with_token_ids():
    rows, world = 32, 4
    hidden = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    tokens = torch.arange(rows)

    def select_experts(*, hidden_states, router_logits, topk_indices_dtype, input_ids):
        assert hidden_states.shape == router_logits.shape
        weights = router_logits[:, :2].clone()
        weights[:, 0] = -0.0
        ids = torch.stack((input_ids % 7, input_ids % 5), dim=-1).to(topk_indices_dtype)
        return weights, ids

    def gate(states):
        return states + 1, None

    runner = SimpleNamespace(
        gate=gate, is_monolithic=False, do_naive_dispatch_combine=False,
        moe_config=SimpleNamespace(pcp_size=1),
        router=SimpleNamespace(select_experts=select_experts),
    )
    kwargs = {"hidden_states": hidden, "topk_indices_dtype": torch.int64, "input_ids": tokens}
    weights, ids = core.select_owned_routes(runner, router_logits=gate(hidden)[0], **kwargs)

    def run_rank(rank):
        geometry = core.row_geometry(rows, world, rank, min_owned=1)
        core.STATE.route_owner = core.Ownership(geometry, tp_world.group(), None, {}, BOUNDARY)
        try:
            logits = core.gate_owned_routes(runner, hidden)
            assert logits.shape[0] == rows // world
            got_weights, got_ids = core.select_owned_routes(runner, router_logits=logits, **kwargs)
            assert core.STATE.route_owner.route_gathers == 1
            assert torch.equal(got_weights.view(torch.int32), weights.view(torch.int32))
            assert got_ids.dtype == ids.dtype and torch.equal(got_ids, ids)
        finally:
            core.STATE.route_owner = None

    _, group = tp_world.run(world, run_rank)
    assert group.calls == [("all_gather", (8, 4))]


def test_indexer_scores_same_tiles_and_gathers_full_table():
    rows, tile, world = 32, 4, 4
    expected = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)

    def run_rank(rank):
        geometry = core.row_geometry(rows, world, rank, min_owned=1)
        ownership = core.Ownership(geometry, tp_world.group(), None, {}, BOUNDARY)
        attention = SimpleNamespace(
            layer_id=2,
            candidate_source_layer=BOUNDARY,
            is_draft=False,
            topk_indices_buffer=torch.full_like(expected, -1),
        )
        core.STATE.ownership = ownership
        try:
            offsets = core.indexer_score_offsets(attention, rows, tile)
            assert list(offsets) == [rank * 8, rank * 8 + 4]
            for first in offsets:
                attention.topk_indices_buffer[first:first + tile] = expected[first:first + tile]
            core.publish_index_rows(attention, rows, tile)
            assert ownership.indexer_gathers == 1
            assert torch.equal(attention.topk_indices_buffer, expected)
            assert core.indexer_score_offsets(attention, rows, 3) == range(0, rows, 3)
            attention.candidate_source_layer = attention.layer_id
            assert core.indexer_score_offsets(attention, rows, tile) == range(0, rows, tile)
            core.publish_index_rows(attention, rows, tile)
            assert ownership.indexer_gathers == 1
        finally:
            core.STATE.ownership = None

    _, group = tp_world.run(world, run_rank)
    assert group.calls == [("all_gather", (8, 3))]


def _model(rows: int, rank: int, seed: int, world: int | None = None):
    """Layers whose weights are shared, except the per-rank TP shares."""
    shared = torch.Generator().manual_seed(seed)
    def reduce(local):
        return core.attention_reduce(local, tp_world.group().all_reduce)

    layers = []
    for index in range(LAYERS):
        g = torch.Generator().manual_seed(seed * 1000 + index)
        layer = fakes.FakeLayer(HIDDEN, rows, g, rank=0, reduce=reduce,
                                engram_hash_index=ENGRAM.get(index))
        # Re-draw the rank-specific shares with a rank-dependent seed.
        rg = torch.Generator().manual_seed(seed * 1000 + index * 37 + rank + 1)
        layer.attn = fakes.FakeAttention(HIDDEN, rg, rank, reduce)
        layer.ffn = fakes.FakeMoE(HIDDEN, rg, rank)
        if layer.engram is not None:
            eg = torch.Generator().manual_seed(seed * 1000 + index + 500)
            layer.engram = fakes.FakeEngram(rows, HIDDEN, ENGRAM[index], rg, rank, shared=eg)
            if world is not None:  # projection_tp: column-parallel, gathered wkv
                layer.engram.wkv = fakes.ColumnParallelLinear(
                    layer.engram.wkv.weight, world, rank
                )
        layers.append(layer)
    embed = torch.randn(rows, HIDDEN, generator=shared, dtype=torch.float64)
    return layers, embed


def _inputs(rows: int, seed: int, masked: bool):
    g = torch.Generator().manual_seed(seed + 7)
    positions = torch.arange(rows) + 100
    input_ids = torch.randint(0, 1000, (rows,), generator=g)
    hashes = torch.randint(0, 50, (rows, 2), generator=g)
    mask = (torch.rand(rows, generator=g) > 0.25) if masked else None
    # Decoder replay rows: the last rows of two requests plus graph padding.
    half = rows // 2
    ced = torch.tensor(
        list(range(half - 3, half)) + list(range(rows - 5, rows)) + [-1, -1]
    )
    return positions, input_ids, hashes, mask, ced


def _loop(layers, embed, positions, input_ids, hashes, mask, ced, layer_call):
    """The image's ``DeepseekV4Model.forward`` loop, reduced to what layers see."""
    hidden, residual, post_mix, res_mix, pre_mix = embed, None, None, None, None
    for index, layer in enumerate(layers):
        boundary = ced if index == BOUNDARY else None
        hidden, residual, post_mix, res_mix, pre_mix = layer_call(
            index, layer, hidden, positions, input_ids, pre_mix, post_mix,
            res_mix, residual, hashes, mask, boundary,
        )
        if boundary is not None:
            positions = fakes.gather_rows(positions, ced)
            input_ids = fakes.gather_rows(input_ids, ced)
    last = layers[-1]._b12x_mhc.post(hidden, residual, post_mix, res_mix)
    return (pre_mix[:, :, None] * last).sum(1)


def _run(world: int, rows: int, owned: bool, seed: int = 3, masked: bool = True,
         projection_tp: bool = False):
    def body(rank: int):
        layers, embed = _model(rows, rank, seed, world if projection_tp else None)
        positions, input_ids, hashes, mask, ced = _inputs(rows, seed, masked)
        own = None
        if owned:
            own = core.Ownership(
                geometry=core.row_geometry(rows, world, rank, min_owned=1),
                group=tp_world.group(),
                gather_rows=fakes.gather_rows,
                layer_index={id(layer): i for i, layer in enumerate(layers)},
                boundary=BOUNDARY,
            )
            core.STATE.ownership = own

        def call(index, layer, *args):
            if own is not None and index <= BOUNDARY:
                return core.owned_layer_forward(layer, own, index, *args)
            *head, boundary = args
            return layer.forward(*head, ced_indices=boundary)

        try:
            out = _loop(layers, embed, positions, input_ids, hashes, mask, ced, call)
            if own is not None:
                own.finish()
        finally:
            core.STATE.ownership = None
        return out, layers, own

    return tp_world.run(world, body)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("projection_tp", [False, True])
def test_owned_forward_equals_full_row_forward(world, masked, projection_tp):
    rows = 8 * world
    reference, ref_group = _run(world, rows, owned=False, masked=masked,
                                projection_tp=projection_tp)
    owned, own_group = _run(world, rows, owned=True, masked=masked,
                            projection_tp=projection_tp)
    for rank in range(world):
        assert torch.equal(owned[rank][0], reference[rank][0]), f"rank {rank}"
        # The reference itself is replicated: the fakes model a valid TP layout.
        assert torch.equal(reference[rank][0], reference[0][0])
        # Every rank's output is identical, as in the image.
        assert torch.equal(owned[rank][0], owned[0][0])
        # The global KV write at the boundary saw the same full rows.
        ref_kv = reference[rank][1][BOUNDARY].attn.global_kv
        own_kv = owned[rank][1][BOUNDARY].attn.global_kv
        assert len(ref_kv) == len(own_kv) == 1
        assert torch.equal(ref_kv[0], own_kv[0])
    kinds = [kind for kind, _ in own_group.calls]
    engram_scattered = 0 if projection_tp else len(ENGRAM)
    assert kinds.count("reduce_scatter") == 2 * BOUNDARY + engram_scattered
    assert kinds.count("all_gather") == 2 * BOUNDARY + 1
    engram_reduced = len(ENGRAM) if projection_tp else 0
    assert kinds.count("all_reduce") == engram_reduced + 4 + 2 * (LAYERS - BOUNDARY)
    own = owned[0][2]
    assert own.engram_layers == len(ENGRAM)
    assert own.engram_reduce_scatters == engram_scattered
    assert own.deferred_attention == BOUNDARY


def test_owned_layers_carry_owned_rows_only():
    world, rows = 4, 32
    shapes = []

    def body(rank):
        layers, embed = _model(rows, rank, 5)
        positions, input_ids, hashes, mask, ced = _inputs(rows, 5, True)
        own = core.Ownership(core.row_geometry(rows, world, rank, 1), tp_world.group(),
                             fakes.gather_rows, {}, BOUNDARY)
        x, residual, post_mix, res_mix, pre_mix = core.owned_layer_forward(
            layers[0], own, 0, embed, positions, input_ids, None, None, None, None,
            hashes, mask, None)
        if rank == 0:
            shapes.extend([x.shape, residual.shape, post_mix.shape, res_mix.shape])
        return None

    tp_world.run(world, body)
    owned = rows // world
    assert shapes == [(owned, HIDDEN), (owned, 4, HIDDEN), (owned, 4), (owned, 4, 4)]


def test_compact_matches_full_gather():
    world, rows = 4, 16
    full = torch.randn(rows, 3, 2, dtype=torch.float64)
    indices = torch.tensor([15, 0, 7, -1, 8, 3])

    def body(rank):
        own = core.Ownership(core.row_geometry(rows, world, rank, 1), tp_world.group(),
                             fakes.gather_rows, {}, 1)
        return own.compact(own.local(full), indices)

    results, _ = tp_world.run(world, body)
    expected = fakes.gather_rows(full, indices)
    for result in results:
        assert torch.equal(result, expected)


def test_attention_reduce_defers_only_inside_owned_scope():
    calls = []
    def reduce(value):
        calls.append(value)
        return value + 1

    t = torch.zeros(2)
    own = core.Ownership(core.Geometry(8, 2, 0), None, fakes.gather_rows, {}, 1)
    core.STATE.ownership = own
    try:
        assert torch.equal(core.attention_reduce(t, reduce), t + 1)
        with own.unreduced_attention():
            assert core.attention_reduce(t, reduce) is t
        assert torch.equal(core.attention_reduce(t, reduce), t + 1)
    finally:
        core.STATE.ownership = None
    assert len(calls) == 2 and own.deferred_attention == 1


@pytest.mark.parametrize(
    "rows,world,rank,minimum,expected",
    [
        (8192, 4, 3, 512, (2048, 6144)),
        (8192, 8, 7, 512, (1024, 7168)),
        (8192, 2, 1, 512, (4096, 4096)),
        (8190, 4, 0, 1, None),  # does not divide
        (1024, 4, 0, 512, None),  # 256 rows per rank, below the minimum
        (8192, 3, 0, 1, None),  # unsupported TP size
        (8192, 4, 4, 1, None),  # rank out of range
    ],
)
def test_row_geometry(rows, world, rank, minimum, expected):
    geometry = core.row_geometry(rows, world, rank, minimum)
    if expected is None:
        assert geometry is None
    else:
        assert (geometry.owned, geometry.start) == expected


def test_owned_layers_for_the_v41_flash_config():
    # DeepSeek-V4.1-Flash: 40 layers, CED decoder start 20, DSpark reads 37-39.
    assert core.owned_layers(20, 40, (37, 38, 39)) == range(21)
    with pytest.raises(core.RowShardRefused):
        core.owned_layers(None, 40, ())
    with pytest.raises(core.RowShardRefused):
        core.owned_layers(20, 40, (20,))  # read after layer 19: owned rows
    # Read after the boundary layer 20: compacted rows, identical on all ranks.
    assert core.owned_layers(20, 40, (21,)) == range(21)


def test_finish_rejects_missing_collectives():
    own = core.Ownership(core.Geometry(8, 2, 0), None, fakes.gather_rows, {}, 2)
    own.layers_seen = [0, 1, 2]
    with pytest.raises(RuntimeError, match="accounting"):
        own.finish()


def test_layer_order_and_boundary_are_enforced():
    own = core.Ownership(core.Geometry(8, 2, 0), None, fakes.gather_rows, {}, 2)
    with pytest.raises(RuntimeError, match="out of order"):
        core.owned_layer_forward(object(), own, 1, None, None, None)
    with pytest.raises(RuntimeError, match="boundary"):
        core.owned_layer_forward(object(), own, 0, None, None, None, ced_indices=torch.zeros(1))


def test_mega_moe_is_refused():
    ffn = type("F", (), {"use_mega_moe": True})()
    with pytest.raises(core.RowShardRefused):
        core.moe_unreduced(ffn, torch.zeros(2, 2), None)
