"""CUDA graph capture safety rules of the adapter.

A captured graph replays exactly the session ops recorded while capturing, so:

1. a plan may not depend on anything that differs between capture and replay
   or between ranks; under capture it may only differ from the eager plan by
   staying on SIRCL where the eager plan would use NCCL;
2. nothing compiles or prepares inside a capture: every dtype a group may
   all-reduce under capture is prepared when the group is built;
3. every op of one capture is issued on the capture stream; the adapter never
   switches streams;
4. a group NCCL may not run never gets an NCCL plan, captured or not.
"""

from __future__ import annotations

import itertools

import pytest
torch = pytest.importorskip("torch")

from sparkring_sircl.vllm import emulation, executor, planner  # noqa: E402
from sparkring_sircl.vllm.emulation import CaptureCompileError, EmulatedFabric, EmulatedRingSession, run_ranks  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout, NcclPolicy, describe_group  # noqa: E402
from sparkring_sircl.vllm.planner import NCCL, SIRCL, Policy, SessionLimits, TensorMeta  # noqa: E402

PATH = Policy(describe_group(Layout.ring(8), range(4)), policy_override=NcclPolicy.NONE)
RING = Policy(describe_group(Layout.ring(8), range(8)), policy_override=NcclPolicy.RING)
PAIR = Policy(describe_group(Layout.ring(8), [6, 7]), policy_override=NcclPolicy.ALL)
SIZES = [1, 7, 8, 1023, 1024, 4096, 65536, 1 << 20, 3 << 20]
DTYPES = [("bfloat16", 2), ("float16", 2), ("float32", 4), ("int64", 8)]


def _limits(world: int) -> SessionLimits:
    return SessionLimits(world=world, capacity=2 << 20, dispatch=2 << 20, gather=155648,
                         reduce_dtypes=("bfloat16", "float16", "float32"), per_peer_op_bytes=131072)


@pytest.mark.parametrize("policy,world", [(PATH, 4), (RING, 8), (PAIR, 2)])
def test_captured_plans_never_use_nccl_where_cabling_forbids_it_and_match_eager_op_counts(policy, world):
    limits = _limits(world)
    for (dtype, item), elements in itertools.product(DTYPES, SIZES):
        meta = TensorMeta((elements,), dtype, item)
        for plan_fn in (lambda m, c: planner.plan_all_reduce(m, limits, policy, capturing=c),
                        lambda m, c: planner.plan_all_gather(m, 0, limits, policy, capturing=c)):
            eager, captured = plan_fn(meta, False), plan_fn(meta, True)
            if policy.nccl_policy is NcclPolicy.NONE:
                assert NCCL not in (eager.backend, captured.backend)
            if eager.backend == SIRCL:
                assert captured == eager
            elif captured.backend == SIRCL and captured.method not in ("gather_sum",):
                # Large messages the eager plan hands to NCCL stay on SIRCL when captured.
                assert captured.method in ("chunked", "rows", "tiles", "direct")


def test_plans_are_identical_on_every_rank_for_every_call():
    limits = _limits(4)
    metas = [TensorMeta((rows, 4096), "bfloat16", 2) for rows in (1, 8, 512, 8192)]

    def body(rank):
        return [planner.plan_all_reduce(meta, limits, PATH, capturing=capturing)
                for meta in metas for capturing in (False, True)]

    plans = run_ranks(4, body)
    assert all(rank_plans == plans[0] for rank_plans in plans)


def test_prepare_is_refused_inside_a_capture_and_unprepared_launchers_raise():
    fabric = EmulatedFabric(2)
    session = EmulatedRingSession(fabric, 0)
    session.prepare((torch.bfloat16,))
    with session.capture():
        with pytest.raises(CaptureCompileError, match="refused inside a CUDA graph capture"):
            session.prepare((torch.float32,))
        with pytest.raises(CaptureCompileError, match="not prepared"):
            session.all_reduce(torch.ones(4, dtype=torch.float32))


def test_a_path_group_prepares_every_dtype_it_may_all_reduce_under_capture():
    world = 4
    limits = _limits(world)
    fabric = EmulatedFabric(world)
    stream = object()
    sessions = [EmulatedRingSession(fabric, rank, max_size=limits.capacity,
                                    dispatch_limit_bytes=limits.dispatch,
                                    max_gather_bytes=limits.gather, current_stream=lambda: stream)
                for rank in range(world)]
    for session in sessions:
        # What GroupAdapter prepares for a group NCCL may not run (slot + extra dtypes).
        session.prepare((torch.bfloat16,), padded_gather=True)
        session.prepare((torch.float16, torch.float32), padded_gather=True)
    inputs = {
        "bf16-prefill": [torch.randn(1024, 4096).to(torch.bfloat16) for _ in range(world)],
        "fp32": [torch.randn(64, 128) for _ in range(world)],
        "int64": [torch.arange(6, dtype=torch.int64) * (r + 1) for r in range(world)],
    }

    def body(rank):
        outputs = {}
        with sessions[rank].capture(stream=stream):
            for name, tensors in inputs.items():
                meta = TensorMeta.of(tensors[rank])
                plan = planner.plan_all_reduce(meta, limits, PATH, capturing=True)
                assert plan.backend == SIRCL
                outputs[name] = executor.all_reduce(sessions[rank], plan, tensors[rank], limits, PATH,
                                                    capturing=True)
        return outputs

    results = run_ranks(world, body)
    for name, tensors in inputs.items():
        expected = (emulation.reference_sum(tensors) if tensors[0].is_floating_point()
                    else sum(tensors))
        assert all(torch.equal(result[name], expected) for result in results)
    captured = [record for record in fabric.records if record.capturing]
    assert len(captured) == len(fabric.records) > 0


def test_ops_issued_off_the_capture_stream_are_refused_by_the_session():
    fabric = EmulatedFabric(2)
    streams = iter([object(), object()])
    session = EmulatedRingSession(fabric, 0, current_stream=lambda: next(streams))
    session.prepare((torch.bfloat16,))
    with session.capture():
        with pytest.raises(emulation.EmulationError, match="one stream"):
            session.all_reduce(torch.ones(8, dtype=torch.bfloat16))


def test_the_executor_never_switches_streams():
    source = executor.__loader__.get_source(executor.__name__)
    for forbidden in ("torch.cuda.stream(", "torch.cuda.Stream(", "synchronize(", ".item()", "wait_stream"):
        assert forbidden not in source
