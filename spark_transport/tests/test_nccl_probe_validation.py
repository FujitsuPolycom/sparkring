"""CPU checks that the NCCL bridge probes cannot accept stale collective output.

The probes run on four Sparks with NCCL and CUDA. These tests replace CUDA
events, streams, graphs and the collectives with fakes on CPU tensors to prove
that a timed loop or graph replay performing no collective fails validation,
that input preparation stays outside the timed span, and that launch
validation rejects bad ranks before touching CUDA.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spark_transport.nccl import probe_dcp2_collectives as dcp2  # noqa: E402
from spark_transport.nccl import probe_dcp4_collectives as dcp4  # noqa: E402


class FakeEvent:
    def __init__(self, *, enable_timing: bool, log: list[str]) -> None:
        assert enable_timing
        self.log = log

    def record(self) -> None:
        self.log.append("event")

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other: "FakeEvent") -> float:
        del other
        return 0.001


class FakeGraph:
    def __init__(self, *, stale: bool) -> None:
        self.captured: list = []
        self.stale = stale
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        if not self.stale:
            for operation in self.captured:
                operation()


class FakeCuda:
    """Replaces the CUDA and torch.distributed calls the probes make."""

    def __init__(self, monkeypatch, module, *, working_iterations: int | None,
                 stale_graph: bool = False) -> None:
        self.log: list[str] = []
        self.collective_calls = 0
        self.working_iterations = working_iterations
        self.capturing: FakeGraph | None = None
        self.graph = FakeGraph(stale=stale_graph)
        monkeypatch.setattr(module, "DEVICE", "cpu")
        cuda = SimpleNamespace(
            Event=lambda enable_timing: FakeEvent(enable_timing=enable_timing, log=self.log),
            synchronize=lambda: None,
            Stream=lambda: SimpleNamespace(wait_stream=lambda stream: self.log.append("wait_input_stream")),
            current_stream=lambda: object(),
            stream=lambda stream: contextlib.nullcontext(),
            CUDAGraph=lambda: self.graph,
            graph=self._capture,
            set_device=self._forbidden,
        )
        monkeypatch.setattr(module.torch, "cuda", cuda)
        dist = SimpleNamespace(
            barrier=lambda group=None: None,
            all_gather_into_tensor=self._all_gather,
            all_reduce=self._all_reduce,
            reduce_scatter_tensor=self._reduce_scatter,
            init_process_group=self._forbidden,
            destroy_process_group=self._forbidden,
        )
        monkeypatch.setattr(module, "dist", dist)

    @contextlib.contextmanager
    def _capture(self, graph: FakeGraph, stream=None):
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("CUDA/NCCL setup must not run in CPU tests")

    def _perform(self, work) -> None:
        if self.capturing is not None:
            self.capturing.captured.append(work)
            return
        self.collective_calls += 1
        self.log.append("collective")
        if self.working_iterations is None or self.collective_calls <= self.working_iterations:
            work()

    def _all_gather(self, output, source, group=None) -> None:
        def work() -> None:
            ranks = output.shape[0] // source.shape[0]
            peers = group if group is not None else range(ranks)
            output.copy_(torch.cat(
                [torch.full_like(source, peer + 1) for peer in peers], dim=0))
        self._perform(work)

    def _all_reduce(self, tensor, group=None) -> None:
        self._perform(lambda: tensor.fill_(10))

    def _reduce_scatter(self, output, input_tensor, group=None) -> None:
        self._perform(lambda: output.fill_(10))


SMALL_GATHER_DCP2 = dcp2.Case("query_q1", (1, 4, 8), torch.bfloat16, 5)
SMALL_GATHER_DCP4 = dcp4.GatherCase("query_q1", (1, 4, 8), torch.bfloat16, 5)
SMALL_SCATTER_DCP4 = dcp4.ReduceScatterCase("output_q1_d8", 1, 8, 5)


@pytest.mark.parametrize("pair_ranks,rank", [([0, 1], 1), ([2, 3], 3)])
def test_dcp2_pair_gather_accepts_only_fresh_output(monkeypatch, pair_ranks, rank):
    fake = FakeCuda(monkeypatch, dcp2, working_iterations=None)
    row = dcp2.timed_pair_all_gather(SMALL_GATHER_DCP2, rank=rank, pair_ranks=pair_ranks, pair_group=pair_ranks)
    assert row["correct"] is True and row["iterations"] == 5

    stale = FakeCuda(monkeypatch, dcp2, working_iterations=5)  # warmup only
    row = dcp2.timed_pair_all_gather(SMALL_GATHER_DCP2, rank=rank, pair_ranks=pair_ranks, pair_group=pair_ranks)
    assert stale.collective_calls == 10
    assert row["correct"] is False
    del fake


def test_dcp2_graph_replay_without_collective_fails_validation(monkeypatch):
    working = FakeCuda(monkeypatch, dcp2, working_iterations=None)
    row = dcp2.graph_pair_all_gather(SMALL_GATHER_DCP2, rank=0, pair_ranks=[0, 1], pair_group=None, replays=3)
    assert row["correct"] is True and working.graph.replays == 3

    stale = FakeCuda(monkeypatch, dcp2, working_iterations=None, stale_graph=True)
    row = dcp2.graph_pair_all_gather(SMALL_GATHER_DCP2, rank=0, pair_ranks=[0, 1], pair_group=None, replays=3)
    assert stale.graph.replays == 3
    assert row["correct"] is False


@pytest.mark.parametrize("module", [dcp2, dcp4])
def test_graph_warmup_waits_for_input_producer_stream(monkeypatch, module):
    fake = FakeCuda(monkeypatch, module, working_iterations=None)
    if module is dcp2:
        module.graph_pair_all_gather(SMALL_GATHER_DCP2, rank=0, pair_ranks=[0, 1], pair_group=None, replays=2)
    else:
        module.graph_world_all_gather(SMALL_GATHER_DCP4, 0, replays=2)
    assert fake.log[0] == "wait_input_stream"


def test_dcp4_gather_and_reduce_scatter_reject_stale_results(monkeypatch):
    FakeCuda(monkeypatch, dcp4, working_iterations=None)
    assert dcp4.timed_world_all_gather(SMALL_GATHER_DCP4, 2)["correct"] is True
    assert dcp4.timed_world_reduce_scatter(SMALL_SCATTER_DCP4, 2)["correct"] is True
    assert dcp4.graph_world_all_gather(SMALL_GATHER_DCP4, 2, replays=2)["correct"] is True
    assert dcp4.graph_world_reduce_scatter(SMALL_SCATTER_DCP4, 2, replays=2)["correct"] is True

    FakeCuda(monkeypatch, dcp4, working_iterations=5)
    assert dcp4.timed_world_all_gather(SMALL_GATHER_DCP4, 2)["correct"] is False
    FakeCuda(monkeypatch, dcp4, working_iterations=5)
    assert dcp4.timed_world_reduce_scatter(SMALL_SCATTER_DCP4, 2)["correct"] is False
    FakeCuda(monkeypatch, dcp4, working_iterations=None, stale_graph=True)
    assert dcp4.graph_world_all_gather(SMALL_GATHER_DCP4, 2, replays=2)["correct"] is False
    FakeCuda(monkeypatch, dcp4, working_iterations=None, stale_graph=True)
    assert dcp4.graph_world_reduce_scatter(SMALL_SCATTER_DCP4, 2, replays=2)["correct"] is False


def test_dcp4_input_preparation_runs_before_the_start_event(monkeypatch):
    fake = FakeCuda(monkeypatch, dcp4, working_iterations=None)
    tensor = torch.zeros(4)
    samples = dcp4._time_cuda(
        lambda: fake._all_reduce(tensor),
        warmups=1,
        iterations=2,
        prepare=lambda: fake.log.append("prepare"),
        result=tensor,
    )
    assert len(samples) == 2
    timed = fake.log[fake.log.index("event"):]
    assert timed == ["event", "collective", "event", "prepare", "event", "collective", "event"]


def test_all_reduce_refills_its_input_before_each_timed_iteration(monkeypatch):
    fake = FakeCuda(monkeypatch, dcp2, working_iterations=None)

    def check_input(tensor):
        assert bool(torch.all(tensor == 4).item())
        fake._all_reduce(tensor)

    monkeypatch.setattr(dcp2.dist, "all_reduce", check_input)
    row = dcp2.timed_world_all_reduce(3)
    assert row["correct"] is True
    assert fake.collective_calls == 520


@pytest.mark.parametrize("module", [dcp2, dcp4])
@pytest.mark.parametrize("environ,message", [
    ({"RANK": "4", "WORLD_SIZE": "4", "HEAD_IP": "10.0.0.1"}, "RANK must be in 0..3"),
    ({"RANK": "-1", "WORLD_SIZE": "4", "HEAD_IP": "10.0.0.1"}, "RANK must be in 0..3"),
    ({"RANK": "0", "WORLD_SIZE": "2", "HEAD_IP": "10.0.0.1"}, "WORLD_SIZE=4"),
])
def test_launch_validation_rejects_bad_ranks_before_cuda(monkeypatch, module, environ, message):
    FakeCuda(monkeypatch, module, working_iterations=None)
    with pytest.raises(ValueError, match=message):
        module.validate_launch(environ)
    monkeypatch.setattr(module.os, "environ", environ)
    with pytest.raises(ValueError, match=message):
        module.main()


@pytest.mark.parametrize("module", [dcp2, dcp4])
def test_launch_validation_requires_head_ip(module):
    with pytest.raises(RuntimeError, match="HEAD_IP"):
        module.validate_launch({"RANK": "0", "WORLD_SIZE": "4"})
    assert module.validate_launch({"RANK": "3", "WORLD_SIZE": "4", "HEAD_IP": "h"}) == (3, 4, "h")
    for port in ("1", "65535"):
        assert module.validate_launch({"RANK": "3", "HEAD_IP": "h", "MASTER_PORT": port}) == (3, 4, "h")


@pytest.mark.parametrize("module", [dcp2, dcp4])
@pytest.mark.parametrize("port", ["0", "65536", "invalid", " 29641"])
def test_invalid_rendezvous_port_is_rejected_before_cuda(monkeypatch, module, port):
    FakeCuda(monkeypatch, module, working_iterations=None)
    monkeypatch.setattr(module.os, "environ", {
        "RANK": "0", "HEAD_IP": "h", "MASTER_PORT": port,
    })
    with pytest.raises(ValueError, match="MASTER_PORT"):
        module.main()


@pytest.mark.parametrize("module", [dcp2, dcp4])
def test_failed_row_still_destroys_the_process_group(monkeypatch, module):
    calls: list[str] = []
    dist = SimpleNamespace(
        init_process_group=lambda **kwargs: calls.append("init"),
        destroy_process_group=lambda: calls.append("destroy"),
    )
    monkeypatch.setattr(module, "dist", dist)
    monkeypatch.setattr(module.torch, "cuda", SimpleNamespace(set_device=lambda index: None))
    monkeypatch.setattr(module.os, "environ", {"RANK": "0", "WORLD_SIZE": "4", "HEAD_IP": "h"})

    def failing_cases(rank: int) -> None:
        raise RuntimeError("collective validation failed")

    monkeypatch.setattr(module, "_run_cases", failing_cases)
    with pytest.raises(RuntimeError, match="validation failed"):
        module.main()
    assert calls == ["init", "destroy"]


@pytest.mark.parametrize("module", [dcp2, dcp4])
def test_graph_timer_waits_for_asynchronous_output_invalidation(monkeypatch, module):
    FakeCuda(monkeypatch, module, working_iterations=None)
    original = module.invalidate
    pending = False
    clock_calls = 0

    def invalidate(tensor):
        nonlocal pending
        original(tensor)
        pending = True

    def synchronize():
        nonlocal pending
        pending = False

    def clock():
        nonlocal clock_calls
        assert not pending, "output reset must finish before host replay timing"
        clock_calls += 1
        return clock_calls * 1000

    monkeypatch.setattr(module, "invalidate", invalidate)
    monkeypatch.setattr(module.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(module.time, "perf_counter_ns", clock)
    if module is dcp2:
        module.graph_pair_all_gather(SMALL_GATHER_DCP2, rank=0, pair_ranks=[0, 1], pair_group=None, replays=2)
    else:
        module.graph_world_all_gather(SMALL_GATHER_DCP4, 2, replays=2)
    assert clock_calls == 2
