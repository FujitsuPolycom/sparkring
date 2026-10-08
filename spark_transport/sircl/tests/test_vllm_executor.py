"""Plans carried out on emulated ranks give exactly the collective's result.

Each test runs every rank of a group as a thread with an
:class:`~sparkring_sircl.vllm.emulation.EmulatedRingSession`, plans the call on
each rank independently, executes it, and compares against the reference: a
rank-ordered float32 sum rounded once or a byte-exact
concatenation. Split and composed plans must give the same bits as one session
op over the whole message.
"""

from __future__ import annotations

import dataclasses

import pytest
torch = pytest.importorskip("torch")

from sparkring_sircl.vllm import executor, planner  # noqa: E402
from sparkring_sircl.vllm.emulation import EmulatedFabric, EmulatedRingSession, reference_sum, run_ranks  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout, NcclPolicy, describe_group  # noqa: E402
from sparkring_sircl.vllm.planner import Policy, SessionLimits, TensorMeta  # noqa: E402

WORLD = 4
POLICY = Policy(describe_group(Layout.ring(8), range(WORLD)), policy_override=NcclPolicy.NONE)


def _limits(**overrides) -> SessionLimits:
    values = dict(world=WORLD, capacity=1 << 20, dispatch=64 << 10, gather=48 << 10,
                  reduce_dtypes=("bfloat16", "float16", "float32"), per_peer_op_bytes=32 << 10)
    values.update(overrides)
    return SessionLimits(**values)


def _sessions(limits: SessionLimits, scatter: bool = True) -> list[EmulatedRingSession]:
    fabric = EmulatedFabric(limits.world)
    return [EmulatedRingSession(fabric, rank, max_size=limits.capacity,
                                dispatch_limit_bytes=limits.dispatch,
                                max_gather_bytes=limits.gather, scatter_available=scatter)
            for rank in range(limits.world)]


def _inputs(shape, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return [torch.randn(shape, generator=generator).to(dtype) for _ in range(WORLD)]
    if dtype == torch.bool:
        return [torch.randint(0, 2, shape, generator=generator).bool() for _ in range(WORLD)]
    return [torch.randint(-50, 50, shape, generator=generator).to(dtype) for _ in range(WORLD)]


def _run(limits, body, *, scatter=True):
    sessions = _sessions(limits, scatter)
    return run_ranks(limits.world, lambda rank: body(rank, sessions[rank])), sessions


@pytest.mark.parametrize("shape,dtype,method", [
    ((300, 1000), torch.bfloat16, "chunked"),     # 600,000 bytes over 64 KiB ops
    ((64, 512), torch.bfloat16, "direct"),
    ((3,), torch.bfloat16, "padded"),
    ((1025, 33), torch.float32, "chunked"),       # 135,300 bytes: last piece not 16-byte whole
    ((5, 7), torch.int64, "gather_sum"),
    ((100,), torch.int8, "gather_sum"),
])
def test_all_reduce_matches_the_rank_ordered_sum(shape, dtype, method):
    limits = _limits()
    inputs = _inputs(shape, dtype)

    def body(rank, session):
        plan = planner.plan_all_reduce(TensorMeta.of(inputs[rank]), limits, POLICY, capturing=False)
        assert plan.method == method
        return executor.all_reduce(session, plan, inputs[rank], limits, POLICY, capturing=False)

    results, _ = _run(limits, body)
    if dtype.is_floating_point:
        expected = reference_sum(inputs)
    else:
        expected = sum(t.to(torch.int64) for t in inputs).to(dtype)
    for result in results:
        assert torch.equal(result.view(torch.uint8) if dtype == torch.bool else result, expected)


def test_chunked_all_reduce_is_bit_identical_to_one_op():
    inputs = _inputs((300, 1000), torch.bfloat16, seed=3)
    chunked = _limits()
    whole = _limits(dispatch=1 << 20)

    def run(limits):
        def body(rank, session):
            plan = planner.plan_all_reduce(TensorMeta.of(inputs[rank]), limits, POLICY, capturing=False)
            return executor.all_reduce(session, plan, inputs[rank], limits, POLICY, capturing=False)
        return _run(limits, body)

    (pieces, sessions), (single, _) = run(chunked), run(whole)
    assert sessions[0].fabric.count("all_reduce") == WORLD * 10
    for a, b in zip(pieces, single):
        assert torch.equal(a, b)


def test_in_place_all_reduce_writes_the_input():
    limits = _limits()
    inputs = _inputs((200, 1000), torch.bfloat16, seed=5)
    expected = reference_sum(inputs)
    copies = [t.clone() for t in inputs]

    def body(rank, session):
        plan = planner.plan_all_reduce(TensorMeta.of(copies[rank]), limits, POLICY, capturing=False)
        return executor.all_reduce(session, plan, copies[rank], limits, POLICY, capturing=False,
                                   out=copies[rank])

    results, _ = _run(limits, body)
    for rank, result in enumerate(results):
        assert result.data_ptr() == copies[rank].data_ptr()
        assert torch.equal(copies[rank], expected)


@pytest.mark.parametrize("shape,dtype,dim", [
    ((6, 5, 7), torch.bfloat16, 0),
    ((6, 5, 7), torch.bfloat16, 1),
    ((6, 5, 7), torch.bfloat16, -1),
    ((8, 38720), torch.bfloat16, -1),       # logits rows: one row per op
    ((40000,), torch.float32, 0),           # one row larger than an op: tiles
    ((3, 20000), torch.float32, 1),         # rows larger than an op: tiles per row
    ((16, 4, 2), torch.float32, 1),         # indexer top-k merge [R, K, 2] along dim 1
    ((9, 3), torch.bool, -1),               # byte view
    ((0, 7), torch.bfloat16, 0),            # empty shard
])
def test_all_gather_matches_concatenation(shape, dtype, dim):
    limits = _limits()
    inputs = _inputs(shape, dtype, seed=7)

    def body(rank, session):
        plan = planner.plan_all_gather(TensorMeta.of(inputs[rank]), dim, limits, POLICY, capturing=False)
        assert plan.backend == planner.SIRCL
        return executor.all_gather(session, plan, inputs[rank], dim, WORLD)

    results, _ = _run(limits, body)
    expected = torch.cat(inputs, dim=dim)
    for result in results:
        assert result.shape == expected.shape and torch.equal(result, expected)


@pytest.mark.parametrize("scatter,shape,dim", [
    (False, (8, 3000), 0),
    (False, (5, 8, 64), 1),
    (True, (8, 64), 0),
    (True, (16, 4096), 0),                  # several strided scatter ops
    (True, (3, 16, 512), 1),
])
def test_reduce_scatter_matches_the_chunk_of_the_sum(scatter, shape, dim):
    limits = _limits(scatter_dtypes=("bfloat16",) if scatter else (), scatter_op_override=64 << 10)
    inputs = _inputs(shape, torch.bfloat16, seed=11)
    total = reference_sum(inputs)
    chunk = shape[dim] // WORLD

    def body(rank, session):
        plan = planner.plan_reduce_scatter(TensorMeta.of(inputs[rank]), dim, limits, POLICY,
                                           capturing=False)
        assert plan.method == ("scatter" if scatter else "allreduce_slice")
        return executor.reduce_scatter(session, plan, inputs[rank], dim, rank, limits, POLICY,
                                       capturing=False)

    results, sessions = _run(limits, body, scatter=scatter)
    for rank, result in enumerate(results):
        assert torch.equal(result, total.narrow(dim, rank * chunk, chunk))
    if scatter and shape == (16, 4096):
        assert sessions[0].fabric.count("reduce_scatter") > WORLD


def test_uneven_gatherv_and_reduce_scatterv():
    limits = _limits()
    sizes = [3, 1, 2, 2]
    shards = [torch.randn(size, 33).to(torch.bfloat16) for size in sizes]
    full = _inputs((8, 33), torch.bfloat16, seed=13)

    def body(rank, session):
        gplan = planner.plan_all_gatherv(TensorMeta.of(shards[rank]), sizes, limits, POLICY,
                                         capturing=False)
        gathered = executor.all_gatherv(session, gplan, shards[rank], sizes, rank, WORLD)
        rplan = planner.plan_reduce_scatterv(TensorMeta.of(full[rank]), sizes, limits, POLICY,
                                             capturing=False)
        scattered = executor.reduce_scatterv(session, rplan, full[rank], 0, sizes, rank, limits,
                                             POLICY, capturing=False)
        return gathered, scattered

    results, _ = _run(limits, body)
    expected_gather = torch.cat(shards)
    total = reference_sum(full)
    offsets = [0, 3, 4, 6]
    for rank, (gathered, scattered) in enumerate(results):
        assert torch.equal(gathered, expected_gather)
        assert torch.equal(scattered, total[offsets[rank]: offsets[rank] + sizes[rank]])


def test_broadcast_and_gather_copy_bytes_exactly():
    limits = _limits()
    tensors = _inputs((7, 9), torch.float32, seed=17)
    tensors[2][0, 0] = -0.0
    expected_src = tensors[2].clone()

    def body(rank, session):
        meta = TensorMeta.of(tensors[rank])
        bplan = planner.plan_bytes_gather("broadcast", meta, limits, POLICY, capturing=False,
                                          nccl_operation="broadcast")
        copy = tensors[rank].clone()
        executor.broadcast(session, bplan, copy, 2, limits, POLICY, capturing=False)
        gplan = planner.plan_bytes_gather("gather", meta, limits, POLICY, capturing=False,
                                          nccl_operation="gather")
        gathered = executor.gather(session, gplan, tensors[rank], 1, 0, rank, limits, POLICY,
                                   capturing=False)
        return copy, gathered

    results, _ = _run(limits, body)
    for rank, (copy, gathered) in enumerate(results):
        assert torch.equal(copy.view(torch.int32), expected_src.view(torch.int32))
        assert (gathered is None) == (rank != 1)
    assert torch.equal(results[1][1], torch.cat(tensors, dim=0))


@pytest.mark.parametrize("scatter", [True, False])
def test_all_to_all_single_exchanges_chunks(scatter):
    limits = _limits(all_to_all=scatter, scatter_op_override=1024)
    inputs = [torch.arange(WORLD * 96, dtype=torch.float32) + 1000 * rank for rank in range(WORLD)]

    def body(rank, session):
        plan = planner.plan_all_to_all(TensorMeta.of(inputs[rank]), limits, POLICY, capturing=False)
        assert plan.method == ("scatter" if scatter else "gather_pick")
        output = torch.empty_like(inputs[rank])
        return executor.all_to_all_single(session, plan, output, inputs[rank], rank, limits, POLICY,
                                          capturing=False)

    results, sessions = _run(limits, body, scatter=scatter)
    for rank, output in enumerate(results):
        expected = torch.cat([inputs[source].view(WORLD, -1)[rank] for source in range(WORLD)])
        assert torch.equal(output, expected)
    if scatter:
        assert sessions[0].fabric.count("all_to_all") == WORLD * 2


def test_a_session_with_native_large_message_operations_is_preferred():
    limits = _limits()
    inputs = _inputs((300, 1000), torch.bfloat16, seed=19)
    shards = _inputs((8192, 32), torch.bfloat16, seed=23)
    calls = []

    class LargeSession(EmulatedRingSession):
        def all_reduce_large(self, inp, *, out=None, stream=None):
            calls.append(("all_reduce_large", self.rank))
            plan = planner.plan_all_reduce(TensorMeta.of(inp), limits, POLICY.internal(), capturing=False)
            return executor.all_reduce(_Plain(self), plan, inp, limits, POLICY, capturing=False)

        def all_gather_large(self, inp, *, dim=-1, out=None, stream=None):
            calls.append(("all_gather_large", self.rank))
            plan = planner.plan_all_gather(TensorMeta.of(inp), dim, limits, POLICY.internal(), capturing=False)
            return executor.all_gather(_Plain(self), plan, inp, dim, WORLD)

    class _Plain:
        """The same session without the large-message operations (the reference path)."""

        def __init__(self, session):
            self._session = session

        def __getattr__(self, name):
            if name in ("all_reduce_large", "all_gather_large"):
                raise AttributeError(name)
            return getattr(self._session, name)

    fabric = EmulatedFabric(WORLD)
    sessions = [LargeSession(fabric, rank, max_size=limits.capacity, dispatch_limit_bytes=limits.dispatch,
                             max_gather_bytes=limits.gather) for rank in range(WORLD)]

    def body(rank):
        reduce_plan = planner.plan_all_reduce(TensorMeta.of(inputs[rank]), limits, POLICY, capturing=False)
        gather_plan = planner.plan_all_gather(TensorMeta.of(shards[rank]), -1, limits, POLICY, capturing=False)
        assert (reduce_plan.method, gather_plan.method) == ("chunked", "rows")
        return (executor.all_reduce(sessions[rank], reduce_plan, inputs[rank], limits, POLICY, capturing=False),
                executor.all_gather(sessions[rank], gather_plan, shards[rank], -1, WORLD))

    results = run_ranks(WORLD, body)
    assert sorted(calls) == sorted([("all_reduce_large", r) for r in range(WORLD)]
                                   + [("all_gather_large", r) for r in range(WORLD)])
    for reduced, gathered in results:
        assert torch.equal(reduced, reference_sum(inputs))
        assert torch.equal(gathered, torch.cat(shards, dim=-1))


def test_a_session_that_states_its_large_pieces_gets_large_plans_in_its_own_pieces():
    reference = _limits()
    inputs = _inputs((8192, 64), torch.bfloat16, seed=29)          # 1 MiB: 16 dispatch ceilings
    shards = _inputs((2048, 64), torch.bfloat16, seed=31)          # 256 KiB shards
    calls = []

    class LargeSession(EmulatedRingSession):
        large_piece_bytes = 256 << 10
        gather_piece_bytes = 32 << 10

        def all_reduce_large(self, inp, *, out=None, stream=None):
            calls.append("all_reduce_large")
            plan = planner.plan_all_reduce(TensorMeta.of(inp), reference, POLICY.internal(), capturing=False)
            assert plan.method == "chunked"                         # the reference split on the host
            return executor.all_reduce(_Plain(self), plan, inp, reference, POLICY, capturing=False)

        def all_gather_large(self, inp, *, dim=-1, out=None, stream=None):
            calls.append("all_gather_large")
            plan = planner.plan_all_gather(TensorMeta.of(inp), dim, reference, POLICY.internal(), capturing=False)
            return executor.all_gather(_Plain(self), plan, inp, dim, WORLD)

    class _Plain:
        def __init__(self, session):
            self._session = session

        def __getattr__(self, name):
            if name in ("all_reduce_large", "all_gather_large"):
                raise AttributeError(name)
            return getattr(self._session, name)

    fabric = EmulatedFabric(WORLD)
    sessions = [LargeSession(fabric, rank, max_size=reference.capacity, dispatch_limit_bytes=reference.dispatch,
                             max_gather_bytes=reference.gather) for rank in range(WORLD)]
    limits = SessionLimits.of(sessions[0], reduce_dtypes=("bfloat16",), per_peer_op_bytes=32 << 10)
    assert (limits.large_piece, limits.gather_piece) == (256 << 10, 32 << 10)
    assert SessionLimits.of(EmulatedRingSession(EmulatedFabric(WORLD), 0), reduce_dtypes=("bfloat16",)).large_piece is None

    def body(rank):
        reduce_plan = planner.plan_all_reduce(TensorMeta.of(inputs[rank]), limits, POLICY, capturing=False)
        gather_plan = planner.plan_all_gather(TensorMeta.of(shards[rank]), 0, limits, POLICY, capturing=False)
        assert (reduce_plan.method, reduce_plan.ops, reduce_plan.piece) == ("large", 4, 256 << 10)
        assert "all_reduce_large" in reduce_plan.reason
        assert (gather_plan.method, gather_plan.ops, gather_plan.piece) == ("large", 8, 32 << 10)
        return (executor.all_reduce(sessions[rank], reduce_plan, inputs[rank], limits, POLICY, capturing=False),
                executor.all_gather(sessions[rank], gather_plan, shards[rank], 0, WORLD))

    results = run_ranks(WORLD, body)
    assert sorted(calls) == ["all_gather_large"] * WORLD + ["all_reduce_large"] * WORLD
    for reduced, gathered in results:
        assert torch.equal(reduced, reference_sum(inputs))
        assert torch.equal(gathered, torch.cat(shards, dim=0))
    no_native = planner.Plan("all_gather", planner.SIRCL, "large", 8, 32 << 10)
    with pytest.raises(executor.PlanError, match="all_gather_large"):
        executor.all_gather(EmulatedRingSession(EmulatedFabric(WORLD), 0), no_native, shards[0], 0, WORLD)


def test_a_session_that_cuts_its_own_reduce_scatter_gets_the_whole_message_in_one_call():
    inputs = _inputs((512, 64), torch.bfloat16, seed=37)            # 64 KiB per rank
    calls = []

    class CuttingSession(EmulatedRingSession):
        scatter_op_bytes = 16 << 10                                   # the session's own op size

        def reduce_scatter(self, inp, *, out=None, stream=None, chunk_bytes=None, src_stride_bytes=None):
            calls.append((self.rank, inp.numel() * inp.element_size(), chunk_bytes))
            return super().reduce_scatter(inp, out=out, stream=stream, chunk_bytes=chunk_bytes,
                                          src_stride_bytes=src_stride_bytes)

    fabric = EmulatedFabric(WORLD)
    sessions = [CuttingSession(fabric, rank, max_size=1 << 20) for rank in range(WORLD)]
    limits = SessionLimits.of(sessions[0], reduce_dtypes=("bfloat16",), scatter_dtypes=("bfloat16",),
                              per_peer_op_bytes=4 << 10)
    assert limits.scatter_piece == 16 << 10 and limits.scatter_op_bytes == 16 << 10
    plan = planner.plan_reduce_scatter(TensorMeta.of(inputs[0]), 0, limits, POLICY, capturing=False)
    assert (plan.method, plan.ops) == ("scatter", 1) and "whole message" in plan.reason

    def body(rank):
        return executor.reduce_scatter(sessions[rank], plan, inputs[rank], 0, rank, limits, POLICY,
                                       capturing=False)

    results = run_ranks(WORLD, body)
    total = reference_sum(inputs)
    for rank, result in enumerate(results):
        assert torch.equal(result, total[rank * 128:(rank + 1) * 128])
    assert sorted(calls) == [(rank, 64 << 10, None) for rank in range(WORLD)]      # one call each, whole
    strided = planner.plan_reduce_scatter(TensorMeta.of(inputs[0]), 0, dataclasses.replace(limits, scatter_piece=None),
                                          POLICY, capturing=False)
    assert (strided.method, strided.ops) == ("scatter", 4)                        # host-cut strided calls
