"""Point-to-point channels of vLLM groups on emulated ranks: plans, dispatch, carriers, sharing and refusals.

Every rank of a group runs as a thread; setup uses the production code path
(placement, the instance plan of :mod:`sparkring_sircl.vllm.p2p`, the voted
construction) with the emulated channels of
:mod:`sparkring_sircl.vllm.p2p_emulation` and the emulated ring sessions.
"""

from __future__ import annotations

import sys
import types

import pytest
torch = pytest.importorskip("torch")

from sparkring_sircl import routes  # noqa: E402
from sparkring_sircl.p2p import budget  # noqa: E402
from sparkring_sircl.vllm import adapter as adapter_module  # noqa: E402
from sparkring_sircl.vllm import emulation, guard, p2p_emulation, planner  # noqa: E402
from sparkring_sircl.vllm import p2p as p2p_mod  # noqa: E402
from sparkring_sircl.vllm.adapter import (AdapterConfig, GroupAdapter, GroupPlacement, SirclDispatchError,  # noqa: E402
                                          )
from sparkring_sircl.vllm.emulation import emulated_groups, run_ranks  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout, NcclPolicy  # noqa: E402
from sparkring_sircl.vllm.planner import NCCL, REFUSE, SIRCL, P2PLimits, Policy  # noqa: E402

SHAPE_2X4 = p2p_mod.Shape(8, tp=4, pp=2)
SHAPE_4X2 = p2p_mod.Shape(8, tp=2, pp=4)


@pytest.fixture
def channels(monkeypatch):
    sessions = emulation.session_module("sircl_emulated_p2p_sessions")
    module = p2p_emulation.p2p_module("sircl_emulated_p2p")
    monkeypatch.setitem(sys.modules, "sircl_emulated_p2p_sessions", sessions)
    monkeypatch.setitem(sys.modules, "sircl_emulated_p2p", module)
    monkeypatch.setenv("SIRCL_SESSION_MODULE", "sircl_emulated_p2p_sessions")
    monkeypatch.setenv("SIRCL_ALLREDUCE_CAPACITY_BYTES", str(1 << 20))
    for name in ("SIRCL_PEER_ROUTES", "NCCL_ALGO", "NCCL_SKIP_TREE_CONNECT", "SIRCL_P2P_GROUPS"):
        monkeypatch.delenv(name, raising=False)
    yield module
    for live in adapter_module.live_adapters():
        live.close()
    guard.reset()


def config_for(layout: str, positions, *, groups=("tp", "dcp"), nccl="topology", p2p_groups=("pp", "tp", "dcp"),
               module="sircl_emulated_p2p"):
    return AdapterConfig(Layout.parse(layout), tuple(positions), tuple(groups), nccl, "auto", None, None,
                         tuple(p2p_groups), module)


def build(kind, global_ranks, config, *, shape=None, parent=None, capturing=None, environ=None):
    groups = emulated_groups(global_ranks)
    device_groups = [object() for _ in global_ranks]

    def body(rank):
        placement = GroupPlacement.of(f"{kind}:0", global_ranks, rank, config, parent_ranks=parent, environ={})
        return GroupAdapter(placement, config=config, cpu_group=groups[rank], device_group=device_groups[rank],
                            device=torch.device("cpu"), nccl=None, single_node=False, environ=dict(environ or {}),
                            capturing=capturing or (lambda: False), shape=shape, parent_ranks=parent)

    return run_ranks(len(global_ranks), body), device_groups


def _fail():
    raise AssertionError("NCCL was reached")


def test_a_pipeline_pair_across_two_tp4_groups_gets_relayed_channels_and_carries_send_and_recv(channels):
    config = config_for("ring:8", range(8))
    adapters, _ = build("pp", [0, 4], config, shape=SHAPE_2X4)
    assert adapters[0].placement.policy is NcclPolicy.NONE
    assert adapters[0].p2p is not None and adapters[0].p2p_plan.basis == "instance"
    # Spark 5's queue toward Spark 6 holds tp[4..7]'s lanes 4->6 and 4->7 (256 KiB) and the PP lanes 2->6 and 3->7.
    assert {w for row in adapters[0].p2p_plan.windows for lanes in row for w in lanes} == {0, 65536}
    payload = torch.arange(3000, dtype=torch.int32)

    def exchange(rank):
        a = adapters[rank]
        if rank == 0:
            a.send(payload, 1, _fail)
            return None
        out = torch.empty(3000, dtype=torch.int32)
        return a.recv(out, 0, _fail)

    received = run_ranks(2, exchange)[1]
    assert torch.equal(received, payload)
    assert adapters[0].counters.snapshot()[0]["collective"] == "send"
    rows = {(row["collective"], row["backend"], row["method"]) for a in adapters for row in a.counters.snapshot()}
    assert rows == {("send", "sircl", "relayed"), ("recv", "sircl", "relayed")}
    record = adapters[1].report(stats=False)
    assert record["p2p"] == "peers=1,slots=8x524288,windows=65536"
    assert record["p2p_detail"]["relayed"] == [0]
    assert "p2p=peers=1" in adapter_module.receipt.line(record)


def test_torch_point_to_point_calls_and_batches_reach_the_channels_through_the_carrier(channels):
    config = config_for("ring:8", range(8), nccl="never")
    adapters, device_groups = build("pp", [1, 5], config, shape=SHAPE_2X4)
    data = torch.randn(17, 9)

    def exchange(rank):
        a = adapters[rank]
        if rank == 0:
            work = a.carry_p2p("isend", {"tensor": data, "dst": 5, "group": device_groups[0], "tag": 0})
            work.wait()
            sent = a.carry_p2p("send", {"tensor": data * 2, "dst": 5, "group": device_groups[0], "tag": 0})
            return sent
        first, second = torch.empty(17, 9), torch.empty(17, 9)
        work = a.carry_p2p("irecv", {"tensor": first, "src": 1, "group": device_groups[1], "tag": 0})
        source = a.carry_p2p("recv", {"tensor": second, "src": 1, "group": device_groups[1], "tag": 0})
        work.wait()
        return first, second, source

    results = run_ranks(2, exchange)
    first, second, source = results[1]
    assert results[0] is None and source == 1
    assert torch.equal(first, data) and torch.equal(second, data * 2)
    op = types.SimpleNamespace(op=types.SimpleNamespace(__name__="isend"), tensor=data, peer=5, group=None)
    assert adapters[0].carry_p2p("isend", {"tensor": data, "dst": 5, "tag": 3}) is guard.NOT_CARRIED   # a tag

    def batch(rank):
        a = adapters[rank]
        if rank == 0:
            return [w.wait() for w in a.carry_p2p("batch_isend_irecv", {"p2p_op_list": [op]})]
        out = torch.empty(17, 9)
        receive = types.SimpleNamespace(op=types.SimpleNamespace(__name__="irecv"), tensor=out, group_peer=0)
        for work in a.carry_p2p("batch_isend_irecv", {"p2p_op_list": [receive]}):
            work.wait()
        return out

    assert torch.equal(run_ranks(2, batch)[1], data)
    keys = {row["collective"] for row in adapters[0].counters.snapshot()}
    assert {"torch.isend", "torch.send", "torch.batch_isend_irecv"} <= keys


def test_a_broadcast_on_a_pipeline_group_runs_as_sends_from_the_source(channels):
    config = config_for("ring:8", range(8), nccl="never")
    adapters, groups = build("pp", [0, 2, 4, 6], config, shape=SHAPE_4X2)
    tokens = torch.tensor([[11], [22], [33]], dtype=torch.int32)

    def broadcast(rank):
        tensor = tokens.clone() if rank == 3 else torch.zeros(3, 1, dtype=torch.int32)
        work = adapters[rank].carry_p2p("broadcast", {"tensor": tensor, "src": 6, "group": groups[rank],
                                                       "async_op": True})
        work.wait()
        return tensor

    for tensor in run_ranks(4, broadcast):
        assert torch.equal(tensor, tokens)
    assert ("torch.broadcast", "sircl", "p2p") in {(r["collective"], r["backend"], r["method"])
                                                  for r in adapters[0].counters.snapshot()}


def test_device_communicator_batches_carry_group_peers_and_refuse_mixed_or_captured_calls(channels):
    config = config_for("ring:8", range(8))
    adapters, _ = build("pp", [2, 6], config, shape=SHAPE_2X4)
    a, b = torch.arange(64, dtype=torch.float32), torch.arange(64, dtype=torch.float32) * -1

    def swap(rank):
        mine, out = (a, torch.empty(64)) if rank == 0 else (b, torch.empty(64))
        adapters[rank].batch_isend_irecv([("recv", out, 1 - rank), ("send", mine, 1 - rank)], _fail)
        return out

    first, second = run_ranks(2, swap)
    assert torch.equal(first, b) and torch.equal(second, a)
    capturing, _ = build("pp", [3, 7], config, shape=SHAPE_2X4, capturing=lambda: True)
    with pytest.raises(SirclDispatchError, match="outside CUDA graph capture"):
        capturing[0].send(torch.ones(4), 1, _fail)


def test_a_receive_of_another_size_poisons_the_channels_on_every_rank(channels):
    config = config_for("ring:8", range(8))
    adapters, _ = build("pp", [0, 4], config, shape=SHAPE_2X4)

    def mismatch(rank):
        if rank == 0:
            adapters[0].send(torch.ones(16, dtype=torch.uint8), 1, _fail)
            return None
        with pytest.raises(p2p_emulation.ChannelsPoisoned, match="different sizes"):
            adapters[1].recv(torch.empty(8, dtype=torch.uint8), 0, _fail)
        return True

    run_ranks(2, mismatch)
    for a in adapters:
        with pytest.raises(p2p_emulation.ChannelsPoisoned):
            a.check_health()


def test_a_pipeline_group_nccl_may_not_run_fails_setup_without_channels(channels, monkeypatch):
    monkeypatch.setitem(sys.modules, "sircl_p2p_unsupported", p2p_emulation.p2p_module("u", supported=False))
    config = config_for("ring:8", range(8), module="sircl_p2p_unsupported")
    with pytest.raises(RuntimeError, match="point-to-point channels of pp:0 .*is_supported"):
        build("pp", [0, 4], config, shape=SHAPE_2X4)
    # A cabled pipeline pair NCCL may connect keeps NCCL for its transfers.
    pair = config_for("ring:8", [3, 4], module="sircl_p2p_unsupported")
    adapters, _ = build("pp", [0, 1], pair, shape=p2p_mod.Shape(2, tp=1, pp=2))
    assert adapters[0].p2p is None and "is_supported" in adapters[0].p2p_reason
    plan = adapters[0]._plan_p2p("send", 1)
    assert plan.backend == NCCL


def test_a_pipeline_group_nccl_may_not_run_fails_setup_when_a_pair_has_no_window(channels):
    config = config_for("ring:8", range(8))
    with pytest.raises(RuntimeError, match=r"pairs without a channel: 0->1: .*SIRCL_P2P_WINDOW_BYTES=0"):
        build("pp", [0, 4], config, shape=SHAPE_2X4, environ={"SIRCL_P2P_WINDOW_BYTES": "0"})


def test_a_tensor_parallel_group_keeps_its_session_and_refuses_send_without_channels(channels, monkeypatch):
    monkeypatch.setitem(sys.modules, "sircl_p2p_unsupported", p2p_emulation.p2p_module("u", supported=False))
    config = config_for("ring:8", range(4), module="sircl_p2p_unsupported")
    adapters, _ = build("tp", [0, 1, 2, 3], config)
    assert adapters[0].session is not None and adapters[0].p2p is None
    with pytest.raises(SirclDispatchError, match="point-to-point channels could not be built"):
        adapters[0].send(torch.ones(4), 3, _fail)
    assert adapters[0].report(stats=False)["p2p"] == "none"


def test_a_tensor_parallel_groups_channels_are_shared_by_the_expert_parallel_group(channels):
    config = config_for("ring:8", range(4), nccl="never")
    tp, _ = build("tp", [0, 1, 2, 3], config)
    assert tp[0].p2p is not None and tp[0].p2p_plan.basis == "group"
    ep, _ = build("ep", [0, 1, 2, 3], config)
    assert ep[0].p2p is tp[0].p2p and ep[0].p2p_shared_from == "tp:0"
    assert ep[0].report(stats=False)["p2p"] == "shared:tp:0"
    data = torch.randn(40)

    def exchange(rank):
        if rank == 0:
            ep[0].send(data, 3, _fail)
            return None
        if rank == 3:
            return ep[3].recv(torch.empty(40), 0, _fail)
        return None

    assert torch.equal(run_ranks(4, exchange)[3], data)
    assert ("send", "sircl", "relayed") in {(r["collective"], r["backend"], r["method"])
                                            for r in ep[0].counters.snapshot()}


def test_session_groups_without_channels_in_the_settings_refuse_with_the_setting(channels):
    config = config_for("ring:8", range(4), p2p_groups=("pp",))
    adapters, _ = build("tp", [0, 1, 2, 3], config)
    assert adapters[0].p2p is None and "SIRCL_P2P_GROUPS" in adapters[0].p2p_reason
    with pytest.raises(SirclDispatchError, match="SIRCL_P2P_GROUPS"):
        adapters[0].send(torch.ones(2), 2, _fail)


def test_the_planner_carries_point_to_point_on_channels_and_names_why_not():
    path = Policy(None, policy_override=NcclPolicy.NONE, reason="no cable between ranks 3-0")
    limits = P2PLimits(frozenset({1, 3}), relayed=frozenset({3}), problems=((2, "relay 1 queue full"),))
    assert planner.plan_point_to_point("send", path, 0, 1, p2p=limits).key() == ("send", SIRCL, "direct")
    assert planner.plan_point_to_point("recv", path, 0, 3, p2p=limits).key() == ("recv", SIRCL, "relayed")
    refused = planner.plan_point_to_point("send", path, 0, 2, p2p=limits)
    assert refused.backend == REFUSE and "relay 1 queue full" in refused.reason
    assert planner.plan_point_to_point("send", path, 0, 1, p2p=limits, capturing=True).backend == REFUSE
    ring = Policy(None, policy_override=NcclPolicy.ALL)
    assert planner.plan_point_to_point("send", ring, 0, 2, p2p=limits).backend == NCCL
    assert planner.plan_point_to_point("send", ring, 0, 1, p2p=limits).backend == SIRCL   # channels first
    never = Policy(None, "never", policy_override=NcclPolicy.ALL)
    assert "SIRCL_NCCL=never" in planner.plan_point_to_point("send", never, 0, 2, p2p=limits).reason


def test_the_tripwire_asks_the_point_to_point_carrier_first_and_finds_sibling_groups(monkeypatch):
    import torch.distributed as dist

    guard.reset()
    guard.install_tripwire()
    try:
        calls = []

        def carrier(operation, arguments):
            calls.append((operation, arguments.get("dst", arguments.get("src"))))
            return "carried" if operation != "all_reduce" else guard.NOT_CARRIED

        group, sibling = object(), object()
        cabled = guard.GuardedGroup("pp:0", (0, 1), NcclPolicy.ALL, "a pair")
        guard.register(group, cabled)
        guard.register_p2p_carrier(group, (0, 1), carrier)
        assert dist.isend(torch.ones(2), 1, group=group) == "carried"            # NCCL may run, channels first
        assert calls == [("isend", 1)]
        monkeypatch.setattr(guard, "_is_nccl", lambda g: True)
        monkeypatch.setattr(dist, "get_process_group_ranks", lambda g: [0, 1])
        guard.configure(lambda ranks: guard.GuardedGroup("sibling", tuple(ranks), NcclPolicy.NONE, "relayed"))
        assert dist.broadcast(torch.ones(2), 1, group=sibling) == "carried"     # refused, then carried by ranks
        assert calls[-1] == ("broadcast", 1)
        guard.unregister(group)
        with pytest.raises(guard.NcclAcrossRelayError):
            dist.broadcast(torch.ones(2), 1, group=sibling)
    finally:
        guard.uninstall_tripwire()
        guard.reset()


def test_the_instance_plan_budgets_pipeline_lanes_before_session_channels():
    plan = p2p_mod.InstancePlan(Layout.parse("ring:8"), range(8), ("tp", "dcp"), ("pp", "tp", "dcp"), SHAPE_2X4, {})
    pp = plan.channels_for("pp", (0, 4))
    tp = plan.channels_for("tp", (0, 1, 2, 3))
    assert pp is not None and tp is not None and plan.basis == "instance"
    assert not pp.unavailable
    share = budget.queue_share(routes.DEFAULT_HAIRPIN_QUEUE)
    used = dict(plan.allocation.reserved)
    for group in plan.groups():
        lanes = p2p_mod._lane_set(group.name, group.topology)
        for key, members in lanes.queues().items():
            used[key] = used.get(key, 0) + sum(plan.allocation.windows.get((group.name, r, p, lane), 0)
                                               for r, p, lane in members)
    assert max(used.values()) <= share
    # The session lanes of tp[0..3] and the PP lanes fill the relay queues of Sparks 1 and 2: the TP group's own
    # channels between its end ranks find less than a chunk there.
    assert (0, 3) in tp.unavailable and "relay" in tp.unavailable[(0, 3)]
    assert plan.channels_for("pp", (1, 2)) is None
