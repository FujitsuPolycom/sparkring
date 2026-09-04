from __future__ import annotations

import pytest

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

    gate.wrap_execute_model(
        Runner,
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

    gate.wrap_execute_model(
        Runner,
        check=lambda: events.append("health"),
        abort=lambda _code: events.append("abort"),
    )

    assert isinstance(Runner().execute_model(), Deferred)
    assert events == ["execute"]
