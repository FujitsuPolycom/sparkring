"""Calls the split does not take: each runs the image's ``forward`` unchanged and is counted.

Also the construction-time refusals: a single KV copy (DCP 8 at TP 8), a DCP
group that is not a run of consecutive TP ranks, and a ``forward`` that is no
longer this plugin's wrapper.
"""

from __future__ import annotations

import dataclasses
import functools

import pytest
import torch
import vllm.v1.attention.backends.mla.b12x_indexer as bi
from glm_dsa_indexer_split import PatchRefused, check_forward, runtime

import harness as h
from harness import Case, Request

IMAGE_FORWARD = bi.B12xSparseIndexer.forward.__wrapped__
LONG = Case(dcp=4, requests=(Request(context=2400, rows=120),), seed=70)


def both_paths(case: Case, settings: runtime.Settings, *, metadata=None, edit=None):
    """Image and wrapper on every rank; returns per rank (image rows, wrapper rows, TP calls of the wrapper)."""
    h.install_stubs()
    problem = h.build_problem(case)
    world = h.make_world(case.dcp)
    runtime.configure(settings)

    def body(rank: int):
        dcp_rank = rank % case.dcp
        h.CONTEXT.jitter = False
        h.set_forward(metadata(problem, dcp_rank) if metadata else h.rank_metadata(problem, dcp_rank))
        indexer = h.make_indexer(problem, dcp_rank)
        if edit:
            edit(indexer)
        runtime.attach(indexer)
        IMAGE_FORWARD(indexer, None, problem.q, None, problem.weights)
        image = indexer.topk_indices_buffer.clone()
        indexer.topk_indices_buffer.fill_(h.SENTINEL)
        before = len(h.CONTEXT.tp.calls)
        bi.B12xSparseIndexer.forward(indexer, None, problem.q, None, problem.weights)
        return image, indexer.topk_indices_buffer.clone(), h.CONTEXT.tp.calls[before:]

    return h.run_ranks(world, body)


def assert_image_path(results, rows):
    for image, wrapped, tp_calls in results:
        assert torch.equal(image[:rows], wrapped[:rows])
        assert tp_calls == []


def test_decode_only_step_takes_the_image_path():
    case = Case(dcp=4, requests=(Request(context=900, rows=3, decode=True), Request(context=700, rows=3, decode=True)),
                seed=71)
    assert_image_path(both_paths(case, runtime.Settings(min_rows=1)), case.rows)
    stats = runtime.stats()
    assert stats["decode"] == h.TP and stats["split_calls"] == 0
    assert all(stats[f"fallback_{name}"] == 0 for name in runtime.FALLBACKS)


def test_short_step_takes_the_image_path():
    assert_image_path(both_paths(LONG, runtime.Settings(min_rows=121)), LONG.rows)
    assert runtime.stats()["fallback_short"] == h.TP


def test_default_minimum_is_512_rows():
    assert runtime.Settings().min_rows == 512
    assert_image_path(both_paths(LONG, runtime.Settings()), LONG.rows)
    assert runtime.stats()["fallback_short"] == h.TP


def test_chunks_with_a_gap_take_the_image_path():
    def metadata(problem, dcp_rank):
        md = h.rank_metadata(problem, dcp_rank)
        first = md.prefill.chunks[0]
        md.prefill.chunks[0] = dataclasses.replace(first, token_start=first.token_start + 1,
                                                   cu_seqlen_ks=first.cu_seqlen_ks[1:],
                                                   cu_seqlen_ke=first.cu_seqlen_ke[1:])
        return md

    results = both_paths(LONG, runtime.Settings(min_rows=1), metadata=metadata)
    assert_image_path(results, LONG.rows)
    assert runtime.stats()["fallback_layout"] == h.TP


def test_multi_request_chunk_takes_the_image_path_and_its_error():
    def metadata(problem, dcp_rank):
        md = h.rank_metadata(problem, dcp_rank)
        md.prefill.chunks[0] = dataclasses.replace(md.prefill.chunks[0], num_reqs=2)
        return md

    with pytest.raises(RuntimeError, match="single-request chunks"):
        both_paths(LONG, runtime.Settings(min_rows=1), metadata=metadata)


def test_capture_takes_the_image_path(monkeypatch):
    monkeypatch.setattr(runtime, "_capturing", lambda tensor: True)
    assert_image_path(both_paths(LONG, runtime.Settings(min_rows=1)), LONG.rows)
    assert runtime.stats()["fallback_capturing"] == h.TP


def test_unprepared_plan_takes_the_image_path():
    problem = h.build_problem(LONG)
    _single_rank_context(problem)
    runtime.configure(runtime.Settings(min_rows=1))
    indexer = h.make_indexer(problem, 0)
    runtime.attach(indexer)
    del indexer._prepared_plans[("prefill", LONG.cap)]
    assert runtime.plan_step(indexer, problem.q, None, problem.weights) == "unprepared"
    indexer._prepared_plans[("prefill", 16)] = h.Plan("prefill", 16, False)  # below the image's launches
    assert runtime.plan_step(indexer, problem.q, None, problem.weights) == "unprepared"


def _single_rank_context(problem, rank: int = 0):
    """Thread-local groups and forward context for plan_step on this thread (no collective is issued)."""
    h.install_stubs()
    world = h.make_world(problem.case.dcp)
    h.CONTEXT.tp, h.CONTEXT.dcp = world.tp[rank], world.dcp[rank]
    h.set_forward(h.rank_metadata(problem, rank % problem.case.dcp))


def test_shapes_and_arguments_the_split_does_not_take():
    problem = h.build_problem(LONG)
    _single_rank_context(problem)
    runtime.configure(runtime.Settings(min_rows=1))
    indexer = h.make_indexer(problem, 0)
    runtime.attach(indexer)
    assert isinstance(runtime.plan_step(indexer, problem.q, None, problem.weights), runtime.Step)
    assert runtime.plan_step(indexer, problem.q[:50], None, problem.weights) == "shape"
    assert runtime.plan_step(indexer, problem.q, None, problem.weights[:50]) == "shape"
    assert runtime.plan_step(indexer, (problem.q, problem.q), None, problem.weights) == "other"
    assert runtime.plan_step(indexer, problem.q, problem.q, problem.weights) == "other"
    wide = h.make_indexer(problem, 0)
    wide.topk_indices_buffer = torch.zeros(256, h.TOPK + 1, dtype=torch.int32)
    runtime.attach(wide)
    assert runtime.plan_step(wide, problem.q, None, problem.weights) == "shape"
    short = h.make_indexer(problem, 0, buffer_rows=100)
    runtime.attach(short)
    assert runtime.plan_step(short, problem.q, None, problem.weights) == "shape"


def test_missing_metadata_takes_the_image_path():
    problem = h.build_problem(LONG)
    _single_rank_context(problem)
    runtime.configure(runtime.Settings(min_rows=1))
    indexer = h.make_indexer(problem, 0)
    runtime.attach(indexer)
    h.CONTEXT.forward = h.SimpleNamespace(attn_metadata=None)
    out = bi.B12xSparseIndexer.forward(indexer, None, problem.q, None, problem.weights)
    assert out is indexer.topk_indices_buffer and bool((out == h.SENTINEL).all())
    h.CONTEXT.forward = h.SimpleNamespace(attn_metadata={"another.layer": None})
    with pytest.raises(KeyError):
        bi.B12xSparseIndexer.forward(indexer, None, problem.q, None, problem.weights)
    assert runtime.stats()["fallback_no_metadata"] == 2


def test_tuple_query_takes_the_image_path_and_its_error():
    problem = h.build_problem(LONG)
    _single_rank_context(problem)
    indexer = h.make_indexer(problem, 0)
    with pytest.raises(ValueError, match="FP8 index queries"):
        bi.B12xSparseIndexer.forward(indexer, None, (problem.q, problem.q), None, problem.weights)
    assert runtime.stats()["fallback_other"] >= 1


@pytest.mark.parametrize("dcp", [8])
def test_single_kv_copy_refuses(dcp):
    problem = h.build_problem(Case(dcp=4, requests=(Request(context=100, rows=8),)))
    _single_rank_context(problem)
    indexer = h.make_indexer(problem, 0)
    indexer.dcp_world_size = dcp
    with pytest.raises(runtime.SplitRefused, match="single KV copy"):
        runtime.attach(indexer)


def test_dcp_group_that_is_not_consecutive_refuses():
    problem = h.build_problem(LONG)
    _single_rank_context(problem, rank=5)
    h.CONTEXT.dcp.ranks = [1, 3, 5, 7]
    indexer = h.make_indexer(problem, 1)
    with pytest.raises(runtime.SplitRefused, match="consecutive TP ranks"):
        runtime.attach(indexer)
    _single_rank_context(problem, rank=5)
    indexer = h.make_indexer(problem, 2)  # TP rank 5 is DCP rank 1 of group 1, not 2
    with pytest.raises(runtime.SplitRefused, match="consecutive TP ranks"):
        runtime.attach(indexer)


def test_forward_replaced_or_wrapped_after_install_refuses():
    cls = bi.B12xSparseIndexer
    installed = cls.forward
    try:
        def foreign(self, hidden_states, q_quant, k, weights):
            return None

        cls.forward = foreign
        with pytest.raises(PatchRefused, match="replaced this plugin's wrapper"):
            check_forward(cls)

        @functools.wraps(installed)
        def outer(self, *args):
            return installed(self, *args)

        cls.forward = outer
        with pytest.raises(PatchRefused, match="wraps this plugin's wrapper"):
            check_forward(cls)
    finally:
        cls.forward = installed
    check_forward(cls)

    class Sub(cls):
        def forward(self, hidden_states, q_quant, k, weights):
            return None

    with pytest.raises(PatchRefused):
        check_forward(Sub)


def test_gathered_key_prefill_takes_the_image_path(monkeypatch):
    """An indexer built with VLLM_DCP_INDEXER_KEY_GATHER=1 never takes the split.

    The image's gathered-key kernel is out of scope here; the stub records
    that the image's ``forward``, which the wrapper falls back to, chose it.
    """
    gathered = []

    def record(self, request, q_quant, weights):
        gathered.append(sum(int(chunk.token_end) - int(chunk.token_start) for chunk in request))

    monkeypatch.setattr(bi.B12xSparseIndexer, "_run_gathered_prefill", record)

    def edit(indexer):
        indexer.dcp_key_gather = True

    results = both_paths(LONG, runtime.Settings(min_rows=1), edit=edit)
    assert_image_path(results, LONG.rows)
    assert runtime.stats()["fallback_key_gather"] == h.TP
    assert gathered  # the fallback ran the image's gathered-key path
