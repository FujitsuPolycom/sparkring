"""Verify mode (``GLM_DSA_INDEXER_SPLIT_VERIFY``) on eight thread-emulated TP ranks.

With ``jitter`` the reference kernel behaves as the b12x selector does on a
GPU (``tests/gpu_local``): each call returns a row's selection in a random
order and resolves exact ties at the row's lowest selected score at random.
Verify mode must then report no difference beyond ties, count tie-only
differences apart, keep the image's result in the buffer, stop after its
budget, and catch a deliberately wrong split.
"""

from __future__ import annotations

import pytest
import torch
import vllm.v1.attention.backends.mla.b12x_indexer as bi
from glm_dsa_indexer_split import runtime

import harness as h
from harness import Case, Request

IMAGE_FORWARD = bi.B12xSparseIndexer.forward.__wrapped__


def run_steps(case: Case, settings: runtime.Settings, steps: int = 1, *, jitter: bool = True):
    """Run ``steps`` identical layer-chunks through the installed wrapper on every rank."""
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
        outputs = []
        for _ in range(steps):
            indexer.topk_indices_buffer.fill_(h.SENTINEL)
            before = len(h.CONTEXT.tp.calls)
            bi.B12xSparseIndexer.forward(indexer, None, problem.q, None, problem.weights)
            outputs.append((indexer.topk_indices_buffer.clone(), len(h.CONTEXT.tp.calls) - before))
        return outputs

    return problem, h.run_ranks(world, body)


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_verify_with_a_deterministic_kernel_reports_no_difference(dcp):
    case = Case(dcp=dcp, requests=(Request(context=600 * dcp, rows=160),), seed=60)
    problem, results = run_steps(case, runtime.Settings(min_rows=1, verify=2), steps=3, jitter=False)
    totals = runtime.stats()["verify"]
    assert totals["layer_chunks"] == 2 * h.TP  # two per rank
    assert totals["rows"] == 2 * h.TP * case.prefill_rows
    assert totals["split_different_set"] == 0 and totals["image_different_set"] == 0
    assert totals["local_different"] == 0 and totals["local_tie_only"] == 0
    assert totals["local_rows"] == (2 * case.prefill_rows * dcp if dcp > 1 else 0)
    expected = h.expected_rows(problem)
    for outputs in results:
        for buffer, _ in outputs:
            assert torch.equal(buffer[:case.rows].to(torch.int64).sort(dim=1).values, expected)
        # Verified steps issue the split's all-gather; the third step runs the plain split (one all-gather).
        assert [calls for _, calls in outputs] == [1, 1, 1]
    assert runtime.stats()["split_calls"] == 3 * h.TP


@pytest.mark.parametrize("dcp", [2, 4])
def test_verify_with_the_gpu_selector_behavior_reports_only_ties(dcp):
    """Random order and random tie resolution, as on the GPU: nothing differs beyond ties."""
    case = Case(dcp=dcp, requests=(Request(context=600 * dcp, rows=160),), seed=65)
    run_steps(case, runtime.Settings(min_rows=1, verify=1))
    totals = runtime.stats()["verify"]
    assert totals["local_rows"] == case.prefill_rows * dcp
    assert totals["local_different"] == 0
    assert totals["local_same"] + totals["local_tie_only"] == totals["local_rows"]


def test_verify_counts_tie_only_differences_apart():
    """Most keys are zero: every row's lowest selected score is a tie among many positions."""
    case = Case(dcp=4, requests=(Request(context=3200, rows=96),), seed=61, tie_keys=0.85)
    run_steps(case, runtime.Settings(min_rows=1, verify=1))
    totals = runtime.stats()["verify"]
    assert totals["local_different"] == 0
    assert totals["local_tie_only"] > 0
    # Image against image differs as much: the selector's own tie resolution.
    assert totals["image_different_set"] > 0


def test_verify_catches_a_wrong_split(monkeypatch):
    case = Case(dcp=4, requests=(Request(context=2400, rows=128),), seed=62)
    original = runtime.score_block

    def wrong(indexer, step, q_quant, weights, record=None):
        tampered = weights.clone()
        tampered[step.decode_rows + step.plan.block_start] *= -1  # the first row of every block
        return original(indexer, step, q_quant, tampered, record)

    monkeypatch.setattr(runtime, "score_block", wrong)
    run_steps(case, runtime.Settings(min_rows=1, verify=1), jitter=False)
    totals = runtime.stats()["verify"]
    assert totals["local_different"] >= 1
    assert totals["split_different_set"] >= 1
    assert totals["image_different_set"] == 0


def test_verify_min_context_selects_long_steps():
    case = Case(dcp=4, requests=(Request(context=600, rows=64),), seed=63)
    run_steps(case, runtime.Settings(min_rows=1, verify=4, verify_min_context=10_000), steps=2)
    assert runtime.stats()["verify"]["layer_chunks"] == 0
    run_steps(case, runtime.Settings(min_rows=1, verify=4, verify_min_context=600), steps=1)
    assert runtime.stats()["verify"]["layer_chunks"] == h.TP  # one per rank


def test_verify_mixed_step_compares_prefill_rows():
    requests = (Request(context=900, rows=3, decode=True), Request(context=1500, rows=100))
    case = Case(dcp=4, requests=requests, seed=64)
    run_steps(case, runtime.Settings(min_rows=1, mixed=True, verify=1))
    totals = runtime.stats()["verify"]
    assert totals["rows"] == h.TP * 100
    assert totals["split_different_set"] == 0 and totals["local_different"] == 0


def test_compare_candidates_classification():
    index_a = torch.tensor([[5, 3, 9, -1], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.int32)
    score_a = torch.tensor([[2.0, 1.0, 1.0, float("-inf")], [4.0, 3.0, 1.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    index_b = torch.tensor([[3, 5, 9, -1], [1, 2, 3, 7], [1, 2, 3, 7]], dtype=torch.int32)
    score_b = torch.tensor([[1.0, 2.0, 1.0, float("-inf")], [4.0, 3.0, 1.0, 1.0], [4.0, 3.0, 2.0, 1.5]])
    same, tie_only = runtime.compare_candidates(index_a, score_a, index_b, score_b)
    assert same.tolist() == [True, False, False]
    assert tie_only.tolist() == [False, True, False]
