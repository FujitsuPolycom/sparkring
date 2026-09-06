from __future__ import annotations

from types import SimpleNamespace

import pytest

import spark_tp4_backend as backend
import spark_tp4_health_gate as gate


def test_output_is_checked_after_existing_synchronization() -> None:
    events = []

    class Output:
        def get_output(self):
            events.append("synchronized")
            return "tokens"

    gate.wrap_get_output(
        Output,
        check=lambda: events.append("health"),
        abort=lambda _code: events.append("abort"),
    )

    assert Output().get_output() == "tokens"
    assert events == ["synchronized", "health"]


def test_unhealthy_output_aborts_before_returning() -> None:
    class Output:
        def get_output(self):
            return "must-not-escape"

    def fail() -> None:
        raise RuntimeError("poisoned")

    def abort(code: int) -> None:
        raise SystemExit(code)

    gate.wrap_get_output(Output, check=fail, abort=abort)

    with pytest.raises(SystemExit, match="70"):
        Output().get_output()


def test_synchronous_execute_model_is_checked_before_return() -> None:
    events = []

    class Runner:
        def execute_model(self):
            events.append("execute")
            return "tokens"

    gate.wrap_worker_output(
        Runner,
        "execute_model",
        check=lambda: events.append("health"),
        abort=lambda _code: events.append("abort"),
    )

    assert Runner().execute_model() == "tokens"
    assert events == ["execute", "health"]


def test_asynchronous_execute_model_defers_check_to_get_output() -> None:
    events = []

    class Deferred:
        def get_output(self):
            return "tokens"

    class Runner:
        def execute_model(self):
            events.append("execute")
            return Deferred()

    gate.wrap_worker_output(
        Runner,
        "execute_model",
        check=lambda: events.append("health"),
        abort=lambda _code: events.append("abort"),
    )

    assert isinstance(Runner().execute_model(), Deferred)
    assert events == ["execute"]


def test_synchronous_sample_aborts_before_output_escapes() -> None:
    class Worker:
        def sample_tokens(self):
            return "must-not-escape"

    def fail() -> None:
        raise RuntimeError("poisoned")

    def abort(code: int) -> None:
        raise SystemExit(code)

    gate.wrap_worker_output(Worker, "sample_tokens", check=fail, abort=abort)

    with pytest.raises(SystemExit, match="70"):
        Worker().sample_tokens()


def test_none_worker_output_does_not_publish_or_check() -> None:
    events = []

    class Worker:
        def execute_model(self):
            events.append("execute")
            return None

    gate.wrap_worker_output(
        Worker,
        "execute_model",
        check=lambda: events.append("health"),
        abort=lambda _code: events.append("abort"),
    )

    assert Worker().execute_model() is None
    assert events == ["execute"]


def test_install_wraps_pinned_worker_and_async_output_boundaries(monkeypatch) -> None:
    class AsyncOutput:
        def get_output(self):
            return "tokens"

    class Worker:
        def execute_model(self):
            return "tokens"

        def sample_tokens(self):
            return "tokens"

    modules = {
        "vllm.v1.worker.gpu.async_utils": SimpleNamespace(
            AsyncOutput=AsyncOutput
        ),
        "vllm.v1.worker.gpu_worker": SimpleNamespace(Worker=Worker),
    }

    def import_module(name: str):
        module = modules.get(name)
        if module is None:
            raise ImportError(name)
        return module

    monkeypatch.setattr(gate.importlib, "import_module", import_module)
    monkeypatch.setattr(gate, "_installed", False)

    gate.install()

    assert AsyncOutput.get_output._sparkring_health_gate is True
    assert Worker.execute_model._sparkring_health_gate is True
    assert Worker.sample_tokens._sparkring_health_gate is True


def test_health_snapshot_includes_every_native_session_family(monkeypatch) -> None:
    healthy = backend.NativeHealthStatus(
        healthy=True,
        poisoned=False,
        progress_thread_running=False,
        stopping=False,
        submitted_sequence=1,
        completed_sequence=1,
        failing_sequence=0,
        error_code=0,
        failing_stage=-1,
        failing_rail=-1,
        failing_peer=-1,
    )
    session = SimpleNamespace(health_status=lambda: healthy)
    instance = SimpleNamespace(
        native_sessions={8192: session},
        bidirectional_prefill_sessions={(1024, 4096): session},
        fused_prefill_sessions={(8192, 4096): session},
    )
    monkeypatch.setattr(backend, "_graph_q1_sessions", {})
    monkeypatch.setattr(backend, "_graph_dual_port_q40_sessions", {})
    monkeypatch.setattr(backend, "_graph_width4096_sessions", {})
    monkeypatch.setattr(backend, "_backends", {0: instance})

    snapshot = backend.native_health_snapshot()

    assert set(snapshot) == {
        "eager-rank-0-bytes-8192",
        "bidirectional-prefill-rank-0-q1024x4096",
        "fused-prefill-rank-0-q8192x4096",
    }
