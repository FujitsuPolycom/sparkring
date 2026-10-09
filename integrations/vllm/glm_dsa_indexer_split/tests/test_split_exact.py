"""The split against the image's own ``B12xSparseIndexer.forward`` on eight thread-emulated TP ranks.

Every rank runs one step twice: the image's method (``forward.__wrapped__``)
and the installed wrapper. With the deterministic reference kernel
(``harness.reference_paged_topk``, jitter off) the two results must be equal
word for word on every rank for every row of the step, the rows beyond the
step must stay untouched, every rank must hold the same rows, and the
prefill rows must equal an independent top-k over the global keys. The
kernel calls show the work: the image scores every prefill row on every rank,
the split only the rank's group block; one TP all-gather of ``per_rank``
int32 rows per call.
"""

from __future__ import annotations

import pytest
import torch
import vllm.v1.attention.backends.mla.b12x_indexer as bi
from glm_dsa_indexer_split import layout, runtime

import harness as h
from harness import Case, Request

IMAGE_FORWARD = bi.B12xSparseIndexer.forward.__wrapped__


def run(case: Case, settings: runtime.Settings, *, jitter: bool = False):
    h.install_stubs()
    problem = h.build_problem(case)
    world = h.make_world(case.dcp)
    runtime.configure(settings)

    def body(rank: int):
        dcp_rank = rank % case.dcp
        h.CONTEXT.jitter = jitter
        h.set_forward(h.rank_metadata(problem, dcp_rank))
        indexer = h.make_indexer(problem, dcp_rank)
        runtime.attach(indexer)
        log = h.KernelLog()
        h.CONTEXT.kernel_log = log
        IMAGE_FORWARD(indexer, None, problem.q, None, problem.weights)
        image = indexer.topk_indices_buffer.clone()
        image_calls = list(log.calls)
        log.calls.clear()
        indexer.topk_indices_buffer.fill_(h.SENTINEL)
        before = len(h.CONTEXT.tp.calls)
        bi.B12xSparseIndexer.forward(indexer, None, problem.q, None, problem.weights)
        split = indexer.topk_indices_buffer.clone()
        return image, split, image_calls, list(log.calls), h.CONTEXT.tp.calls[before:]

    return problem, h.run_ranks(world, body)


def check(case: Case, settings: runtime.Settings, *, expect_split: bool = True):
    problem, results = run(case, settings)
    rows, offset, prefill = case.rows, case.decode_rows, case.prefill_rows
    expected = h.expected_rows(problem)
    reference = results[0][0]
    for rank, (image, split, image_calls, split_calls, tp_calls) in enumerate(results):
        assert torch.equal(image[:rows], split[:rows]), f"rank {rank}: split differs from the image"
        assert bool((split[rows:] == h.SENTINEL).all()), f"rank {rank}: rows beyond the step were written"
        assert torch.equal(image[:rows], reference[:rows]), f"rank {rank}: ranks disagree"
        got = image[offset:rows].to(torch.int64).sort(dim=1).values
        assert torch.equal(got, expected), f"rank {rank}: prefill rows differ from the global top-k"
        image_prefill = sum(n for mode, n in image_calls if mode == "prefill")
        split_prefill = sum(n for mode, n in split_calls if mode == "prefill")
        assert image_prefill == prefill
        if expect_split:
            plan = layout.row_layout(prefill, h.TP, case.dcp, rank)
            assert split_prefill == plan.block_rows
            assert tp_calls == [("all_gather", (plan.per_rank, h.TOPK), "torch.int32", 0)]
            assert sum(n for mode, n in split_calls if mode == "decode") == sum(
                n for mode, n in image_calls if mode == "decode")
        else:
            assert split_calls == image_calls and tp_calls == []
    if expect_split:
        total = sum(sum(n for mode, n in calls if mode == "prefill") for _, _, _, calls, _ in results)
        assert total == prefill * case.dcp  # every row scored by the d members of exactly one group
    return results


SPLIT = runtime.Settings(min_rows=1)
FULL = runtime.Settings(min_rows=1, full_launches=True)

ONE_REQUEST = {1: Request(context=600, rows=200), 2: Request(context=1200, rows=300),
               4: Request(context=2400, rows=300)}


@pytest.mark.parametrize("settings", [SPLIT, FULL], ids=["image-launches", "full-launches"])
@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_one_long_request(dcp, settings):
    check(Case(dcp=dcp, requests=(ONE_REQUEST[dcp],), seed=dcp), settings)


@pytest.mark.parametrize("settings", [SPLIT, FULL], ids=["image-launches", "full-launches"])
@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_several_requests_odd_rows(dcp, settings):
    requests = (Request(context=2000, rows=37), Request(context=0, rows=211), Request(context=900, rows=5))
    check(Case(dcp=dcp, requests=requests, seed=10 + dcp), settings)


@pytest.mark.parametrize("rows", [1, 2, 7, 9, 15])
@pytest.mark.parametrize("dcp", [1, 4])
def test_steps_shorter_than_two_rows_per_rank(dcp, rows):
    check(Case(dcp=dcp, requests=(Request(context=1500, rows=rows),), seed=20 + rows), SPLIT)


@pytest.mark.parametrize("dcp", [2, 4])
def test_launches_follow_the_logits_budget_or_the_plan(dcp):
    """A tiny logits budget gives 8-row image launches; full launches use the 96-row plan."""
    case = Case(dcp=dcp, requests=(Request(context=2000, rows=192),), logits_budget=8 * 2200 * 4, seed=30)
    image_sizes = {hi.stop - hi.start for _, hi in h.chunk_specs(h.build_problem(case))}
    assert max(image_sizes) <= 8
    results = check(case, FULL)
    for _, _, _, split_calls, _ in results:
        launches = [n for mode, n in split_calls if mode == "prefill"]
        assert max(launches) > 8  # the block runs in plan-sized launches, not the image's 8-row ones
    results = check(case, SPLIT)
    for _, _, _, split_calls, _ in results:
        assert max(n for mode, n in split_calls if mode == "prefill") <= 8


@pytest.mark.parametrize("dcp", [1, 4])
def test_mixed_step_splits_prefill_rows_and_keeps_the_image_decode(dcp):
    requests = (Request(context=800, rows=3, decode=True), Request(context=1300, rows=3, decode=True),
                Request(context=1000, rows=150))
    case = Case(dcp=dcp, requests=requests, seed=40)
    check(case, runtime.Settings(min_rows=1, mixed=True))
    runtime.reset_stats()
    check(case, SPLIT, expect_split=False)  # mixed steps take the image's path by default
    assert runtime.stats()["fallback_mixed"] == h.TP


def test_counters_after_a_split():
    runtime.reset_stats()
    case = Case(dcp=4, requests=(Request(context=2400, rows=300),), seed=50)
    check(case, SPLIT)
    stats = runtime.stats()
    assert stats["engaged"] and stats["split_calls"] == h.TP
    assert stats["split_rows"] == 300 * 4
    assert "layer-chunks split" in runtime.describe_stats()
