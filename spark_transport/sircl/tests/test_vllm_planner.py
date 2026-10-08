"""Dispatch decisions per collective: backend, method and op count, at every size boundary.

A plan may depend only on the group's cabling, the session's agreed limits and
the call's dtype, shape, contiguity, size and capture state, so
all ranks take the same plan and never mix backends within one collective.
"""

from __future__ import annotations

import dataclasses

import pytest
torch = pytest.importorskip("torch")

from sparkring_sircl.vllm import planner  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout, NcclPolicy, describe_group  # noqa: E402
from sparkring_sircl.vllm.planner import (NCCL, REFUSE, SIRCL, Policy, SessionLimits, TensorMeta,  # noqa: E402
                                          plan_all_gather, plan_all_gatherv, plan_all_reduce,
                                          plan_all_to_all, plan_bytes_gather, plan_point_to_point,
                                          plan_reduce_scatter, plan_reduce_scatterv)

PATH4 = describe_group(Layout.ring(8), [0, 1, 2, 3])        # TP4 on Sparks 0-3: NCCL never
RING8 = describe_group(Layout.ring(8), range(8))            # TP8: NCCL ring only
PAIR = describe_group(Layout.ring(8), [6, 7])               # TP2: NCCL anything
D = 2 * 1024 * 1024
G = 155648
LIMITS4 = SessionLimits(world=4, capacity=D, dispatch=D, gather=G,
                        reduce_dtypes=("bfloat16", "float16", "float32"), per_peer_op_bytes=262144)
LIMITS8 = SessionLimits(world=8, capacity=D, dispatch=D, gather=G, reduce_dtypes=("bfloat16",),
                        per_peer_op_bytes=131072)
LIMITS2 = SessionLimits(world=2, capacity=D, dispatch=D, gather=G, reduce_dtypes=("bfloat16",))
NONE = Policy(PATH4, policy_override=NcclPolicy.NONE)
RING = Policy(RING8, policy_override=NcclPolicy.RING)
ALL = Policy(PAIR, policy_override=NcclPolicy.ALL)


def bf16(*shape: int) -> TensorMeta:
    return TensorMeta(tuple(shape), "bfloat16", 2)


@pytest.mark.parametrize("rows,expected", [
    (1, ("direct", 1)),
    (D // 8192 - 1, ("direct", 1)),        # one row below the ceiling
    (D // 8192, ("direct", 1)),            # exactly the ceiling
    (D // 8192 + 1, ("chunked", 2)),       # one row above
    (8192, ("chunked", 32)),               # a prefill step of 8192 tokens, hidden 4096
])
def test_all_reduce_on_a_path_never_uses_nccl(rows, expected):
    for capturing in (False, True):
        plan = plan_all_reduce(bf16(rows, 4096), LIMITS4, NONE, capturing=capturing)
        assert plan.backend == SIRCL
        assert (plan.method, plan.ops) == expected


def test_all_reduce_on_a_ring_sends_large_eager_messages_to_nccl_only():
    small = plan_all_reduce(bf16(32, 4096), LIMITS8, RING, capturing=False)
    assert (small.backend, small.method) == (SIRCL, "direct")
    large = plan_all_reduce(bf16(8192, 4096), LIMITS8, RING, capturing=False)
    assert large.backend == NCCL
    captured = plan_all_reduce(bf16(8192, 4096), LIMITS8, RING, capturing=True)
    assert (captured.backend, captured.method) == (SIRCL, "chunked")
    forced = plan_all_reduce(bf16(8192, 4096), LIMITS8, Policy(RING8, large="sircl",
                                                                  policy_override=NcclPolicy.RING),
                             capturing=False)
    assert forced.backend == SIRCL
    refused = plan_all_reduce(bf16(8192, 4096), LIMITS4, Policy(PATH4, large="nccl",
                                                                   policy_override=NcclPolicy.NONE),
                              capturing=False)
    assert refused.backend == REFUSE and "SIRCL_LARGE_ALLREDUCE=nccl" in refused.reason


@pytest.mark.parametrize("meta,policy,limits,expected", [
    (TensorMeta((64, 4096), "float32", 4), NONE, LIMITS4, (SIRCL, "direct")),
    (TensorMeta((64, 4096), "float32", 4), RING, LIMITS8, (NCCL, "nccl")),
    (TensorMeta((64, 4096), "float32", 4), Policy(RING8, "never"), LIMITS8, (SIRCL, "gather_sum")),
    (TensorMeta((64,), "int64", 8), NONE, LIMITS4, (SIRCL, "gather_sum")),
    (TensorMeta((64,), "int8", 1), NONE, LIMITS4, (SIRCL, "gather_sum")),
    (TensorMeta((3,), "bfloat16", 2), NONE, LIMITS4, (SIRCL, "padded")),
    (TensorMeta((0, 4096), "bfloat16", 2), NONE, LIMITS4, (SIRCL, "empty")),
    (TensorMeta((4,), "bool", 1), NONE, LIMITS4, (REFUSE, "refuse")),
])
def test_all_reduce_dtype_and_shape_cases(meta, policy, limits, expected):
    plan = plan_all_reduce(meta, limits, policy, capturing=False)
    assert (plan.backend, plan.method) == expected


def test_all_reduce_without_a_session_needs_cabled_ranks():
    assert plan_all_reduce(bf16(4, 4), None, ALL, capturing=False).backend == NCCL
    refused = plan_all_reduce(bf16(4, 4), None, NONE, capturing=False)
    assert refused.backend == REFUSE and "no cable" in refused.reason


def test_non_contiguous_inputs_are_copied_not_declined():
    meta = TensorMeta((64, 4096), "bfloat16", 2, contiguous=False)
    plan = plan_all_reduce(meta, LIMITS4, NONE, capturing=False)
    assert plan.backend == SIRCL and plan.contiguous_copy


# The gather op size on a path of four is min(G, 256 KiB) = G = 155,648 bytes.
@pytest.mark.parametrize("shape,dim,expected", [
    ((8, 38720), -1, ("rows", 4)),            # logits of 8 rows: 2 rows of 77,440 bytes per op
    ((1, 38720), -1, ("direct", 1)),          # 77,440 bytes
    ((G // 2,), 0, ("direct", 1)),            # exactly the capacity
    ((G // 2 + 8,), 0, ("tiles", 2)),         # one pack above: one row split in tiles
    ((128, 16, 576), 1, ("rows", 16)),        # MLA query gather along heads: 8 rows per op
    ((8192, 1024), -1, ("rows", 108)),        # prefill [T, H/4] shard: 76 rows per op
])
def test_all_gather_pieces_follow_the_gather_op_size(shape, dim, expected):
    plan = plan_all_gather(bf16(*shape), dim, LIMITS4, NONE, capturing=False)
    assert plan.backend == SIRCL
    assert (plan.method, plan.ops) == expected


def test_all_gather_relay_load_limits_the_op_below_the_capacity():
    # Ring of eight: 128 KiB per peer per op even though G is 152 KiB.
    plan = plan_all_gather(bf16(70000,), 0, LIMITS8, Policy(RING8, "never"), capturing=False)
    assert (plan.method, plan.ops) == ("tiles", 2)
    eager_on_ring = plan_all_gather(bf16(70000,), 0, LIMITS8, RING, capturing=False)
    assert eager_on_ring.backend == NCCL


def test_all_gather_dtypes_and_disabled_gathers():
    assert plan_all_gather(TensorMeta((64, 2), "bool", 1), -1, LIMITS4, NONE,
                           capturing=False).method == "bytes_direct"
    disabled = SessionLimits(world=4, capacity=D, dispatch=D, gather=0, reduce_dtypes=("bfloat16",))
    assert plan_all_gather(bf16(4, 4), -1, disabled, NONE, capturing=False).backend == REFUSE
    assert plan_all_gather(bf16(4, 4), -1, disabled, ALL, capturing=False).backend == NCCL


def test_reduce_scatter_uses_the_scatter_op_where_prepared_and_otherwise_all_reduce():
    plain = plan_reduce_scatter(bf16(64, 4096), 0, LIMITS4, NONE, capturing=False)
    assert (plain.backend, plain.method) == (SIRCL, "allreduce_slice")
    dcp = SessionLimits(world=4, capacity=D, dispatch=D, gather=G, reduce_dtypes=("bfloat16",),
                        scatter_dtypes=("bfloat16",), all_to_all=True, per_peer_op_bytes=262144)
    small = plan_reduce_scatter(bf16(64, 512), 0, dcp, NONE, capturing=False)
    assert (small.method, small.ops) == ("scatter", 1)
    large = plan_reduce_scatter(bf16(64, 4096, 4), 0, dcp, NONE, capturing=False)
    assert large.method == "scatter" and large.ops == 2
    odd = plan_reduce_scatter(bf16(6, 4096), 0, dcp, NONE, capturing=False)
    assert odd.backend == REFUSE and "equal chunks" in odd.reason


def test_uneven_sizes_are_padded_or_composed():
    gatherv = plan_all_gatherv(bf16(3, 4096), [3, 1, 2, 2], LIMITS4, NONE, capturing=False)
    assert gatherv.backend == SIRCL and gatherv.method.startswith("padded_")
    even = plan_all_gatherv(bf16(2, 4096), [2, 2, 2, 2], LIMITS4, NONE, capturing=False)
    assert even.method == "direct"
    scatterv = plan_reduce_scatterv(bf16(8, 4096), [3, 1, 2, 2], LIMITS4, NONE, capturing=False)
    assert scatterv.method == "allreduce_slice"
    assert plan_reduce_scatterv(bf16(8, 4096), [3, 1, 2, 2], LIMITS8, RING,
                                capturing=False).backend == NCCL


def test_broadcast_gather_and_all_to_all():
    for name, operation in (("broadcast", "broadcast"), ("gather", "gather")):
        plan = plan_bytes_gather(name, bf16(16, 16), LIMITS4, NONE, capturing=False,
                                 nccl_operation=operation)
        assert (plan.backend, plan.method) == (SIRCL, "gather_bytes")
    assert plan_bytes_gather("broadcast", bf16(16, 16), LIMITS8, RING, capturing=False,
                             nccl_operation="broadcast").backend == NCCL
    # NCCL's gather sends every rank's tensor to one rank: not a ring operation.
    assert plan_bytes_gather("gather", bf16(16, 16), LIMITS8, RING, capturing=False,
                             nccl_operation="gather").backend == SIRCL
    a2a_ring = plan_all_to_all(bf16(8, 64), LIMITS8, RING, capturing=False)
    assert (a2a_ring.backend, a2a_ring.method) == (SIRCL, "gather_pick")
    assert plan_all_to_all(bf16(8, 64), LIMITS2, ALL, capturing=False).backend == NCCL


def test_point_to_point_only_between_cabled_ranks_of_a_group_nccl_may_run():
    assert plan_point_to_point("send", NONE, 0, 1).backend == REFUSE
    assert plan_point_to_point("send", RING, 0, 1).backend == NCCL
    assert plan_point_to_point("send", RING, 0, 4).backend == REFUSE
    assert plan_point_to_point("recv", ALL, 1, 0).backend == NCCL


def test_dcp_thresholds_move_only_eager_calls():
    dcp8 = describe_group(Layout.ring(8), range(8), parent=range(8))
    policy = Policy(dcp8, nccl_above=(("all_gather", 458752),), policy_override=NcclPolicy.RING)
    limits = SessionLimits(world=8, capacity=D, dispatch=D, gather=D, reduce_dtypes=("bfloat16",),
                           per_peer_op_bytes=131072)
    below = plan_all_gather(bf16(200000,), 0, limits, policy, capturing=False)
    assert below.backend == SIRCL and below.method == "tiles"
    above = plan_all_gather(bf16(300000,), 0, limits, policy, capturing=False)
    assert above.backend == NCCL
    assert plan_all_gather(bf16(300000,), 0, limits, policy, capturing=True).backend == SIRCL


def test_plans_ignore_pointer_values_and_storage_offsets():
    base = torch.zeros(2 * 4096 + 7, dtype=torch.bfloat16)
    aligned = base[: 4096 * 2].view(2, 4096)
    shifted = base[7: 7 + 4096 * 2].view(2, 4096)
    assert aligned.data_ptr() % 16 != shifted.data_ptr() % 16
    assert TensorMeta.of(aligned) == TensorMeta.of(shifted)
    plans = {plan_all_reduce(TensorMeta.of(t), LIMITS4, NONE, capturing=False) for t in (aligned, shifted)}
    assert len(plans) == 1


def test_every_refusal_states_its_reason():
    refused = [
        plan_all_reduce(TensorMeta((4,), "bool", 1), LIMITS4, NONE, capturing=False),
        plan_reduce_scatter(bf16(6, 8), 0, LIMITS4, NONE, capturing=False),
        plan_point_to_point("send", NONE, 0, 3),
    ]
    for plan in refused:
        assert plan.backend == REFUSE and len(plan.reason) > 10
    assert planner.SUMMED_DTYPES == ("float16", "bfloat16", "float32")


def test_no_tuning_table_routes_a_call_and_the_rules_decide_what_nccl_carries():
    """The planner takes no table input, so a table's NCCL marks route no call in any SIRCL_NCCL mode. On a pair
    under the opt-in auto the rules send eager calls above the dispatch ceiling to NCCL; under never nothing."""
    from sparkring_sircl.vllm import sessionapi

    assert "tuned" not in {field.name for field in dataclasses.fields(Policy)}
    for mode, above in (("auto", NCCL), ("never", SIRCL)):
        pair = Policy(PAIR, nccl_mode=mode, policy_override=NcclPolicy.ALL)
        assert plan_all_reduce(bf16(8, 4096), LIMITS2, pair, capturing=False).backend == SIRCL, mode   # within D
        assert plan_all_reduce(bf16(8192, 4096), LIMITS2, pair, capturing=False).backend == above, mode
        assert plan_all_reduce(bf16(8192, 4096), LIMITS2, pair, capturing=True).backend == SIRCL, mode
    # --large-allreduce sircl keeps every all-reduce on SIRCL; on a path NCCL runs nothing.
    for policy, limits in ((Policy(PAIR, large="sircl", policy_override=NcclPolicy.ALL), LIMITS2),
                           (Policy(PATH4, policy_override=NcclPolicy.NONE), LIMITS4)):
        assert plan_all_reduce(bf16(8192, 4096), limits, policy, capturing=False).backend == SIRCL
    # The session's side: a table's hash from its statistics, and its NCCL mark where the table decides, a
    # measurement that no plan reads.

    class Session:
        def __init__(self, table):
            self.table = table

        def stats(self):
            return {"tuning": {"table": self.table}} if self.table else {}

        def tuned_choice(self, collective, nbytes, mode=None):
            assert mode == "eager"
            return object() if nbytes >= 4096 else None

        def tuned_backend(self, collective, nbytes, mode=None):
            return "nccl" if collective == "all_reduce" else "sircl"

    decide = sessionapi.tuned_backend(Session("a" * 16))
    assert (decide("all_reduce", 8192), decide("all_gather", 8192), decide("all_reduce", 16)) == ("nccl", "sircl", None)
    assert sessionapi.tuning_table(Session("a" * 16)) == "a" * 16
    assert sessionapi.tuned_backend(Session(None)) is None and sessionapi.tuning_table(object()) is None
