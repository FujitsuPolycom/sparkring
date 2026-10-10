"""Group adapters on emulated ranks: placements of the eight-Spark ring end to end.

Every rank of a group runs as a thread. Setup uses the production code path:
placement, the voted route maps, the tensor-parallel slot or the DCP
collectives, the session package (an emulated one with ``API_VERSION``,
``is_supported`` and ``AllReduce``), extra dtype preparation, receipts. Collectives then go through the
adapter's plans. A recorder stands in for vLLM's stock NCCL paths and fails
the test if a group whose cabling forbids NCCL ever reaches it.
"""

from __future__ import annotations

import json
import sys

import pytest
torch = pytest.importorskip("torch")

from sparkring_sircl.vllm import adapter as adapter_module  # noqa: E402
from sparkring_sircl.vllm import emulation, guard  # noqa: E402
from sparkring_sircl.vllm.adapter import (AdapterConfig, GroupAdapter, GroupPlacement, NcclPaths,  # noqa: E402
                                          SirclDispatchError, SirclSetupError)
from sparkring_sircl.vllm.emulation import emulated_groups, reference_sum, run_ranks  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout, NcclPolicy  # noqa: E402

RING_NCCL = {"NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"}


@pytest.fixture
def sessions(monkeypatch):
    module = emulation.session_module("sircl_emulated_test")
    monkeypatch.setitem(sys.modules, "sircl_emulated_test", module)
    monkeypatch.setenv("SIRCL_SESSION_MODULE", "sircl_emulated_test")
    monkeypatch.setenv("SIRCL_ALLREDUCE_CAPACITY_BYTES", str(1 << 20))
    monkeypatch.setenv("SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES", str(64 << 10))
    monkeypatch.setenv("SIRCL_ALLGATHER_MAX_BYTES", str(48 << 10))
    for name in ("SIRCL_PEER_ROUTES", "NCCL_ALGO", "NCCL_SKIP_TREE_CONNECT", "SIRCL_TOPOLOGY"):
        monkeypatch.delenv(name, raising=False)
    yield module
    for live in adapter_module.live_adapters():
        live.close()
    guard.reset()


class NcclRecorder:
    """vLLM's stock paths: records every call and returns a recognisable result."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def paths(self) -> NcclPaths:
        def record(name):
            def call(*args):
                self.calls.append(name)
                tensor = args[1] if name == "all_to_all_single" else args[0]
                return tensor.clone()
            return call
        names = ("all_reduce", "all_reduce_in_place", "all_gather", "all_gatherv", "reduce_scatter",
                 "reduce_scatterv", "gather", "broadcast", "all_to_all_single")
        return NcclPaths(**{name: record(name) for name in names})


def build(kind, global_ranks, config, *, parent=None, environ=None, capturing=None, per_rank_env=None):
    groups = emulated_groups(global_ranks)
    recorders = [NcclRecorder() for _ in global_ranks]

    def body(rank):
        env = dict(environ or {})
        if per_rank_env:
            env.update(per_rank_env.get(rank, {}))
        placement = GroupPlacement.of(f"{kind}:0", global_ranks, rank, config, parent_ranks=parent,
                                      environ=env)
        return GroupAdapter(placement, config=config, cpu_group=groups[rank],
                            device_group=object(), device=torch.device("cpu"),
                            nccl=recorders[rank].paths(), single_node=False, environ=env,
                            capturing=capturing or (lambda: False))

    adapters = run_ranks(len(global_ranks), body)
    return adapters, recorders


def config_for(layout: str, positions, *, groups=("tp", "dcp"), receipt_dir=None, nccl="topology"):
    return AdapterConfig(Layout.parse(layout), tuple(positions), tuple(groups), nccl, "auto", None,
                         receipt_dir)


def test_tp4_on_four_sparks_of_a_ring_of_eight_never_reaches_nccl(sessions, tmp_path):
    config = config_for("ring:8", range(4), receipt_dir=str(tmp_path))
    adapters, recorders = build("tp", [0, 1, 2, 3], config)
    first = adapters[0]
    assert first.placement.policy is NcclPolicy.NONE and first.placement.suppress_pynccl
    assert first.prepared_extra == ("float16", "float32")
    routes = {session.rank: session.peer_routes for session in sessions.created}
    assert routes[0] == {
        1: ("rocep1s0f0", "roceP2p1s0f0"), 2: ("rocep1s0f0", "roceP2p1s0f0"),
        3: ("rocep1s0f0", "roceP2p1s0f0")}
    assert routes[3][0] == ("rocep1s0f1", "roceP2p1s0f1")

    hidden = [torch.randn(512, 4096).to(torch.bfloat16) for _ in range(4)]        # 4 MiB prefill
    decode = [torch.randn(8, 4096).to(torch.bfloat16) for _ in range(4)]
    logits = [torch.randn(8, 38720).to(torch.bfloat16) for _ in range(4)]
    counts = [torch.tensor([rank + 1, 7], dtype=torch.int64) for rank in range(4)]
    norms = [torch.randn(16, 128, dtype=torch.float32) for _ in range(4)]

    def serve(rank):
        a = adapters[rank]
        return (a.all_reduce(hidden[rank]), a.all_reduce(decode[rank]), a.all_gather(logits[rank], -1),
                a.all_reduce(counts[rank]), a.all_reduce(norms[rank]),
                a.reduce_scatter(decode[rank], 0), a.broadcast(decode[rank].clone(), 0))

    results = run_ranks(4, serve)
    for rank, (big, small, gathered, count, norm, scattered, copied) in enumerate(results):
        assert torch.equal(big, reference_sum(hidden))
        assert torch.equal(small, reference_sum(decode))
        assert torch.equal(gathered, torch.cat(logits, dim=-1))
        assert count.tolist() == [10, 28]
        assert torch.equal(norm, reference_sum(norms))
        assert torch.equal(scattered, reference_sum(decode)[rank * 2: rank * 2 + 2])
        assert torch.equal(copied, decode[0])
    assert all(recorder.calls == [] for recorder in recorders)
    for a in adapters:
        assert a.counters.total(backend="nccl") == 0
        assert a.counters.total(backend="sircl") == 7
    a.write_receipt()
    record = json.loads((tmp_path / "rank3-tp-0.json").read_text())
    assert record["nccl"] == "none" and record["pynccl"] == "skipped"
    assert record["fabric"] == "path:0-1-2-3" and record["op_per_peer"] == 262144
    assert all(row["backend"] == "sircl" for row in record["decisions"])
    line = adapter_module.receipt.line(record)
    assert line.startswith("SIRCL receipt group=tp:0 global_rank=3 rank=3 world=4 layout=ring:8")
    assert "nccl=none pynccl=skipped" in line


def test_receipts_are_rewritten_after_a_new_decision_row_and_periodically(sessions, tmp_path, monkeypatch):
    config = config_for("ring:8", range(4), receipt_dir=str(tmp_path))
    adapters, _ = build("tp", [0, 1, 2, 3], config)
    receipt_file = tmp_path / "rank0-tp-0.json"
    assert json.loads(receipt_file.read_text())["decisions"] == []          # written at setup
    small = [torch.randn(8, 4096).to(torch.bfloat16) for _ in range(4)]

    def step():
        run_ranks(4, lambda rank: adapters[rank].all_reduce(small[rank]))
        adapters[0].check_health()                     # the worker's post-step health check

    def rows():
        return [(row["collective"], row["backend"], row["method"], row["calls"])
                for row in json.loads(receipt_file.read_text())["decisions"]]

    def ops():
        return json.loads(receipt_file.read_text())["session_stats"]["ops"]

    step()                                             # a new row: rewritten with fresh statistics
    assert rows() == [("all_reduce", "sircl", "direct", 1)] and ops() == 4
    step()                                             # a known row within a second: not rewritten
    assert rows() == [("all_reduce", "sircl", "direct", 1)]
    monkeypatch.setattr(adapter_module.receipt, "COUNT_REFRESH_SECONDS", 0.0)
    adapters[0].check_health()                         # counts changed: rewritten, statistics kept
    assert rows() == [("all_reduce", "sircl", "direct", 2)] and ops() == 4
    adapters[0].check_health()                         # nothing changed: not rewritten
    receipt_file.write_text(json.dumps({"decisions": [], "session_stats": {"ops": -1}}))
    adapters[0].check_health()
    assert rows() == [] and ops() == -1
    monkeypatch.setattr(adapter_module.receipt, "REFRESH_SECONDS", 0.0)
    adapters[0].check_health()                         # the statistics period elapsed: fresh statistics
    assert rows() == [("all_reduce", "sircl", "direct", 2)] and ops() == 8


def test_the_second_tp4_group_routes_over_sparks_four_to_seven(sessions):
    config = config_for("ring:8", [4, 5, 6, 7])
    adapters, _ = build("tp", [0, 1, 2, 3], config)
    assert adapters[0].placement.positions == (4, 5, 6, 7)
    rank0 = next(session for session in sessions.created if session.rank == 0)
    assert rank0.peer_routes[3] == ("rocep1s0f0", "roceP2p1s0f0")
    assert adapters[0].record["fabric"] == "path:4-5-6-7"


def test_a_route_map_that_contradicts_the_layout_fails_every_rank(sessions):
    config = config_for("ring:8", range(4))
    wrong = {3: {"SIRCL_PEER_ROUTES": "0=rocep1s0f0/roceP2p1s0f0,1=rocep1s0f1/roceP2p1s0f1,"
                                      "2=rocep1s0f1/roceP2p1s0f1"}}
    with pytest.raises(SirclSetupError, match="rank 3: route map of rank 3 sends rank 0"):
        build("tp", [0, 1, 2, 3], config, per_rank_env=wrong)
    assert sessions.created == []


def test_tp8_ring_hands_large_eager_all_reduces_to_nccl_only_with_its_ring_contract(sessions):
    config = config_for("ring:8", range(8))
    adapters, recorders = build("tp", list(range(8)), config, environ=RING_NCCL)
    assert adapters[0].placement.policy is NcclPolicy.RING
    assert not adapters[0].placement.suppress_pynccl
    big = [torch.randn(256, 4096).to(torch.bfloat16) for _ in range(8)]
    run_ranks(8, lambda rank: adapters[rank].all_reduce(big[rank]))
    assert all(recorder.calls == ["all_reduce"] for recorder in recorders)

    for a in adapters:
        a.close()
    captured, recorders = build("tp", list(range(8)), config, environ=RING_NCCL,
                                capturing=lambda: True)
    with adapter_module.capture_all():
        results = run_ranks(8, lambda rank: captured[rank].all_reduce(big[rank]))
    assert all(recorder.calls == [] for recorder in recorders)
    assert torch.equal(results[5], reference_sum(big))


def test_tp8_ring_without_the_ring_contract_keeps_nccl_off(sessions):
    config = config_for("ring:8", range(8))
    adapters, recorders = build("tp", list(range(8)), config, environ={})
    placement = adapters[0].placement
    assert placement.policy is NcclPolicy.NONE and "NCCL_ALGO" in placement.reason
    big = [torch.randn(256, 4096).to(torch.bfloat16) for _ in range(8)]
    run_ranks(8, lambda rank: adapters[rank].all_reduce(big[rank]))
    assert all(recorder.calls == [] for recorder in recorders)


def test_dcp4_inside_tp8_carries_gathers_scatters_and_all_to_all(sessions, monkeypatch):
    monkeypatch.setenv("GLM_DCP_RDMA_CAPACITY_BYTES", str(1 << 20))
    monkeypatch.setenv("GLM_DCP_RDMA_GATHER_MAX_BYTES", str(1 << 20))
    config = config_for("ring:8", range(8))
    adapters, recorders = build("dcp", [0, 1, 2, 3], config, parent=list(range(8)), environ=RING_NCCL)
    a0 = adapters[0]
    assert a0.placement.policy is NcclPolicy.NONE and a0.dcp is not None
    assert a0.limits.scatter_dtypes == ("bfloat16",) and a0.limits.all_to_all
    assert a0.dcp.gather_chunk == 262144          # relay-safe size of a path of four
    query = [torch.randn(64, 4, 576).to(torch.bfloat16) for _ in range(4)]
    lse = [torch.randn(64, 16, dtype=torch.float32) for _ in range(4)]
    heads = [torch.randn(16, 64, 512).to(torch.bfloat16) for _ in range(4)]
    send = [torch.arange(4 * 64 * 514, dtype=torch.float32).to(torch.bfloat16) + rank for rank in range(4)]

    def serve(rank):
        a = adapters[rank]
        out = torch.empty_like(send[rank])
        return (a.all_gather(query[rank], 1), a.all_gather(lse[rank], 0), a.reduce_scatter(heads[rank], 0),
                a.all_to_all_single(out, send[rank]))

    results = run_ranks(4, serve)
    for rank, (q, gathered_lse, scattered, exchanged) in enumerate(results):
        assert torch.equal(q, torch.cat(query, dim=1))
        assert torch.equal(gathered_lse, torch.cat(lse, dim=0))
        assert torch.equal(scattered, reference_sum(heads)[rank * 4: rank * 4 + 4])
        expected = torch.cat([send[source].view(4, -1)[rank] for source in range(4)])
        assert torch.equal(exchanged, expected)
    assert all(recorder.calls == [] for recorder in recorders)


def test_groups_without_a_session_refuse_collectives_where_nccl_may_not_run(sessions):
    config = config_for("ring:8", range(4))
    adapters, recorders = build("ep", [0, 1, 2, 3], config)
    assert adapters[0].session is None and adapters[0].placement.suppress_pynccl
    with pytest.raises(SirclDispatchError, match="no cable between ranks 3-0"):
        adapters[0].all_reduce(torch.ones(4, dtype=torch.bfloat16))
    assert recorders[0].calls == []

    pair = config_for("ring:8", [6, 7])
    adapters, recorders = build("ep", [0, 1], pair)
    adapters[1].all_reduce(torch.ones(4, dtype=torch.bfloat16))
    assert recorders[1].calls == ["all_reduce"]


def test_four_tp2_groups_use_sircl_within_the_ceiling_and_nccl_above(sessions):
    results = []
    for first in (0, 2, 4, 6):
        config = config_for("ring:8", [first, first + 1])
        adapters, recorders = build("tp", [0, 1], config)
        assert adapters[0].placement.policy is NcclPolicy.ALL
        small = [torch.randn(4, 4096).to(torch.bfloat16) for _ in range(2)]
        big = [torch.randn(64, 4096).to(torch.bfloat16) for _ in range(2)]
        out = run_ranks(2, lambda rank: (adapters[rank].all_reduce(small[rank]),
                                         adapters[rank].all_reduce(big[rank])))
        assert torch.equal(out[0][0], reference_sum(small))
        results.append([recorder.calls for recorder in recorders])
        for a in adapters:
            a.close()
    assert results == [[["all_reduce"], ["all_reduce"]]] * 4


def test_a_dead_rank_poisons_every_session_and_stops_serving(sessions):
    config = config_for("ring:8", range(4))
    adapters, _ = build("tp", [0, 1, 2, 3], config)
    data = sessions.created[0].fabric
    data.timeout = 2.0
    data.fail(2)
    tensors = [torch.ones(8, 4096, dtype=torch.bfloat16) for _ in range(4)]
    failures = []

    def step(rank):
        try:
            adapters[rank].all_reduce(tensors[rank])
        except emulation.SessionPoisoned as exc:
            failures.append((rank, str(exc)))

    run_ranks(4, step)
    assert sorted(rank for rank, _ in failures) == [0, 1, 2, 3]
    with pytest.raises(emulation.SessionPoisoned):
        adapter_module.check_all_health()


def test_capture_fan_out_enters_every_live_session(sessions, monkeypatch):
    monkeypatch.setenv("GLM_DCP_RDMA_CAPACITY_BYTES", str(1 << 20))
    tp, _ = build("tp", [0, 1, 2, 3], config_for("ring:8", range(4)))
    assert all(not session.capturing() for session in sessions.created)
    with adapter_module.capture_all(stream="graph-stream"):
        assert all(session.capturing() for session in sessions.created)
    assert all(not session.capturing() for session in sessions.created)
    adapter_module.check_all_health()
