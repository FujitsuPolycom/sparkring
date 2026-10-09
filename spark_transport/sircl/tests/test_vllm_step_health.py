"""The start-of-step failure check: the ``step_health`` shim and :func:`adapter.check_all_failures`.

A flag wait that times out is recorded by the kernel after its op returned to
Python; the shim makes each of vLLM's worker step methods (``execute_model``,
``sample_tokens``, ``execute_dummy_batch``) check every SIRCL session and
point-to-point channel set of the process first, so a failure recorded during
an earlier step raises before the step's SIRCL ops launch. These tests run the
adapter against the stand-in vLLM package (:mod:`vllm_stub`) and SIRCL's
emulated sessions, every rank of a group a thread, as
``test_vllm_registration.py`` does. They cover:

- with a pinned worker the shim installs with the first group that gets a
  session, and each step method runs its body while every session is healthy;
- a failure recorded by any session of the process raises when each step
  method starts, before the method's body runs;
- a failure of a group's point-to-point channels raises there too;
- the check reads host state only: no device synchronization, no session
  statistics, no receipt, no flag-wait regime change;
- on an unpinned worker the shim refuses with a warning, the step methods run
  unchecked, and the failure raises at the worker's post-step check.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import vllm_stub  # noqa: E402
from sparkring_sircl.vllm import adapter as adapter_module  # noqa: E402
from sparkring_sircl.vllm import catalog, emulation, guard, hooks, pins, plugin, shims  # noqa: E402
from sparkring_sircl.vllm.emulation import SessionPoisoned, emulated_groups, run_ranks  # noqa: E402

STEP_CALLS = (("execute_model", (None,)), ("sample_tokens", (None,)), ("execute_dummy_batch", ()))


@pytest.fixture
def stub(tmp_path, monkeypatch):
    vllm_stub.purge()
    root = vllm_stub.write(tmp_path / "site")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    module = emulation.session_module("sircl_emulated_step_health")
    monkeypatch.setitem(sys.modules, "sircl_emulated_step_health", module)
    environ = vllm_stub.ThreadEnviron(dict(os.environ))
    monkeypatch.setattr(os, "environ", environ)
    for name in ("VLLM_ENABLE_ROCE_ALLREDUCE", "VLLM_DISABLE_PYNCCL", "VLLM_SPARK_TP4_MODE",
                 "VLLM_SPARK_TP4_VOCAB_MODE", "SIRCL_PEER_ROUTES", "NCCL_ALGO", "SIRCL_VLLM_SHIMS",
                 "SIRCL_RANK_POSITIONS", "SIRCL_TOPOLOGY", "SIRCL_LAYOUT"):
        environ.pop(name, None)
    environ.update({"SIRCL_MODE": "custom", "SIRCL_FABRIC": "ring:8", "SIRCL_NCCL": "topology",
                    "SIRCL_SESSION_MODULE": "sircl_emulated_step_health",
                    "SIRCL_ALLREDUCE_CAPACITY_BYTES": str(1 << 20),
                    "SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES": str(64 << 10),
                    "SIRCL_ALLGATHER_MAX_BYTES": str(48 << 10)})
    plugin.reset_for_tests()
    guard.reset()
    shims._INSTALLED.clear()
    adapter_module.reset_regimes_for_tests()
    yield types.SimpleNamespace(root=root, sessions=module, environ=environ)
    from vllm.v1.worker.gpu_worker import Worker

    Worker.DURING = None
    for live in adapter_module.live_adapters():
        live.close()
    if guard.tripwire_installed():
        guard.uninstall_tripwire()
    guard.reset()
    shims._INSTALLED.clear()
    adapter_module.reset_regimes_for_tests()
    plugin.reset_for_tests()
    vllm_stub.purge()


def _pin_worker(monkeypatch, root) -> None:
    build = pins.VllmBuild("stub", "0", "test stand-in", pins.hashes(root, pins.WORKER_FILES))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))


def _communicators(unique_name, global_ranks):
    from sparkring_sircl.vllm.communicator import SirclCudaCommunicator

    groups = emulated_groups(global_ranks)

    def body(rank):
        return SirclCudaCommunicator(groups[rank], torch.device("cpu"), object(), unique_name,
                                     list(global_ranks), len(global_ranks))

    return run_ranks(len(global_ranks), body)


def _worker_recording(seen):
    from vllm.v1.worker.gpu_worker import Worker

    Worker.DURING = seen.append
    return Worker()


def test_the_shim_is_pinned_cataloged_and_anchored_on_the_worker_file():
    assert shims.SHIMS["step_health"].files == pins.WORKER_FILES
    assert shims.WORKER_STEP_METHODS == ("execute_model", "sample_tokens", "execute_dummy_batch")
    [hook] = [hook for hook in hooks.HOOKS if hook.shim == "step_health"]
    assert {anchor.text for anchor in hook.anchors} == {
        "def sample_tokens(", "def execute_model(", "def execute_dummy_batch(self) -> None:"}
    record = {item["name"]: item for item in catalog.document()["shims"]}["step_health"]
    assert [code["name"] for code in record["code"]] == [f"Worker.{name}" for name in shims.WORKER_STEP_METHODS]


def test_a_recorded_failure_raises_when_each_step_method_starts_before_its_body(stub, monkeypatch):
    _pin_worker(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    assert shims.installed()["step_health"] == "stub"
    seen: list[str] = []
    worker = _worker_recording(seen)
    for name, arguments in STEP_CALLS:
        assert getattr(worker, name)(*arguments) == name      # healthy: the step's body runs
    assert seen == [name for name, _ in STEP_CALLS]
    seen.clear()
    comms[2].sircl.session._failure = "rank 2 timed out waiting for link item 7 from rank 3 lane 0"
    for name, arguments in STEP_CALLS:
        with pytest.raises(SessionPoisoned, match="rank 2 timed out"):
            getattr(worker, name)(*arguments)
    assert seen == []                                         # no step body ran, so no SIRCL op launched


def test_a_failure_of_a_groups_point_to_point_channels_raises_at_the_step_start(stub, monkeypatch):
    _pin_worker(monkeypatch, stub.root)
    plugin.register()
    _communicators("tp:0", [0, 1, 2, 3])
    seen: list[str] = []
    worker = _worker_recording(seen)

    class FailedChannels:
        def check_health(self):
            raise RuntimeError("SIRCL point-to-point channels failed on rank 1: a stopped progress thread")

    live = adapter_module.live_adapters()[0]
    monkeypatch.setattr(live, "p2p", FailedChannels())
    monkeypatch.setattr(live, "p2p_shared_from", None)
    with pytest.raises(RuntimeError, match="point-to-point channels failed"):
        worker.execute_model(None)
    assert seen == []


def test_the_start_of_step_check_reads_host_state_only(stub, monkeypatch):
    _pin_worker(monkeypatch, stub.root)
    plugin.register()
    comms = _communicators("tp:0", [0, 1, 2, 3])
    regimes = [comm.sircl.session.wait_regime for comm in comms]

    def forbidden(*args, **kwargs):
        raise AssertionError("the start-of-step check must not synchronize, read statistics or write receipts")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    for comm in comms:
        monkeypatch.setattr(comm.sircl.session, "stats", forbidden)
    for live in adapter_module.live_adapters():
        monkeypatch.setattr(live, "write_receipt", forbidden)
        monkeypatch.setattr(live, "_refresh_receipt", forbidden)
    with adapter_module.startup_all("a test", then_serve=True):
        pass                                                  # armed: a post-step check would enter serving
    seen: list[str] = []
    worker = _worker_recording(seen)
    adapter_module.check_all_failures()
    worker.execute_model(None)
    assert seen == ["execute_model"]
    assert [comm.sircl.session.wait_regime for comm in comms] == regimes


def test_an_unpinned_worker_warns_and_the_failure_raises_at_the_post_step_check(stub, caplog):
    plugin.register()
    with caplog.at_level(logging.WARNING, logger="sircl.vllm.communicator"):
        comms = _communicators("tp:0", [0, 1, 2, 3])
    assert "step_health" not in shims.installed()
    assert any("step_health refuses to load" in record.getMessage() for record in caplog.records)
    comms[1].sircl.session._failure = "rank 1 timed out waiting for chain chunk 4 from rank 0 lane 0"
    seen: list[str] = []
    worker = _worker_recording(seen)
    worker.execute_model(None)                               # unchecked: the step's body runs
    assert seen == ["execute_model"]
    with pytest.raises(SessionPoisoned, match="rank 1 timed out"):
        comms[0].b12x_ar_comm.check_health()                 # the worker's post-step check
