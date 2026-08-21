from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from . import spark_q2r_probe_bridge as bridge_module


@dataclass
class FakeDrain:
    still_pending: int = 0
    errors: int = 0


class Calls:
    def __init__(self) -> None:
        self.values: list[tuple[str, Any]] = []

    def functions(self) -> bridge_module.ProbeFunctions:
        return bridge_module.ProbeFunctions(
            phase_install=lambda: self.values.append(("phase_install", None)),
            phase_arm=lambda epoch: self.values.append(("phase_arm", epoch)),
            phase_disarm=lambda: self.values.append(("phase_disarm", None)),
            phase_drain=lambda: (
                self.values.append(("phase_drain", None)) or FakeDrain()
            ),
            phase_snapshot=lambda: {
                "phase_timing": {"enabled": True, "pending": 0}
            },
            route_arm=lambda **kwargs: self.values.append(
                ("route_arm", kwargs)
            ),
            route_disarm=lambda **kwargs: self.values.append(
                ("route_disarm", kwargs)
            ),
            route_counters=lambda: (
                self.values.append(("route_counters", None))
                or {
                    "rounds_claimed": 0,
                    "rounds_completed": 0,
                    "wrong_phase": 0,
                }
            ),
            route_drain=lambda *args, **kwargs: (
                self.values.append(
                    ("route_drain", (args, kwargs))
                )
                or {"rounds_completed": 125}
            ),
            provenance_factory=lambda **kwargs: kwargs,
            dcp_graph_report=lambda: (
                self.values.append(("dcp_graph_report", None))
                or {
                    "passed": True,
                    "ranks": {
                        2: {
                            "contract_passed": True,
                            "correctness_passed": True,
                        }
                    },
                }
            ),
        )


def write_command(path: Path, sequence: int, action: str, **extra: Any) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "sparkring-q2r-control/v1",
                "sequence": sequence,
                "action": action,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def make_bridge(
    tmp_path: Path, calls: Calls
) -> tuple[bridge_module.ProbeControlBridge, Path]:
    control = tmp_path / "control.json"
    bridge = bridge_module.ProbeControlBridge(
        functions=calls.functions(),
        control_path=control,
        route_output_path=tmp_path / "routes.jsonl",
        rank=2,
    )
    bridge.mark_worker_ready()
    return bridge, control


def test_arm_next_request_disarm_sequence(tmp_path: Path) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    write_command(
        control,
        1,
        "arm",
        epoch="q2r",
        request_slot=0,
        request_key="a",
    )
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(
        control, 2, "next_request", request_slot=1, request_key="b"
    )
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(control, 3, "disarm")
    snapshot = bridge.snapshot()
    assert snapshot["last_sequence"] == 3
    assert [item[0] for item in calls.values] == [
        "phase_arm",
        "route_arm",
        "route_disarm",
        "route_arm",
        "route_disarm",
        "phase_disarm",
    ]


def test_dcp_graph_report_is_one_shot_and_published(
    tmp_path: Path,
) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    write_command(control, 1, "dcp_graph_report")
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "ok"
    assert snapshot["dcp_graph_report"] == {
        "state": "complete",
        "passed": True,
        "ranks": {
            2: {
                "contract_passed": True,
                "correctness_passed": True,
            }
        },
    }
    assert [item[0] for item in calls.values] == ["dcp_graph_report"]

    # Re-reading the same sequence must not synchronize or report twice.
    assert bridge.snapshot()["dcp_graph_report"]["passed"] is True
    assert [item[0] for item in calls.values] == ["dcp_graph_report"]


def test_dcp_graph_report_fails_closed_when_unavailable(
    tmp_path: Path,
) -> None:
    calls = Calls()
    functions = calls.functions()
    functions = bridge_module.ProbeFunctions(
        **{
            **functions.__dict__,
            "dcp_graph_report": None,
        }
    )
    control = tmp_path / "control.json"
    bridge = bridge_module.ProbeControlBridge(
        functions=functions,
        control_path=control,
        route_output_path=tmp_path / "routes.jsonl",
        rank=2,
    )
    bridge.mark_worker_ready()
    write_command(control, 1, "dcp_graph_report")
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert "unavailable" in snapshot["last_error"]
    assert snapshot["dcp_graph_report"] == {"state": "not_run"}


def test_phase_only_epoch_never_arms_or_drains_routes(
    tmp_path: Path,
) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    provenance = bridge.snapshot()["provenance"]
    write_command(
        control,
        1,
        "arm",
        epoch="adaptive",
        request_slot=0,
        request_key="a",
        capture_routes=False,
    )
    armed = bridge.snapshot()
    assert armed["last_result"] == "ok"
    assert armed["route"]["capture_enabled"] is False
    assert armed["route"]["armed"] is False
    write_command(
        control,
        2,
        "next_request",
        request_slot=1,
        request_key="b",
        capture_routes=False,
    )
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(control, 3, "disarm")
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(control, 4, "drain", provenance=provenance)
    drained = bridge.snapshot()
    assert drained["last_result"] == "ok"
    assert drained["route"]["drain"]["state"] == "skipped-phase-only"
    names = [item[0] for item in calls.values]
    assert "route_arm" not in names
    assert "route_drain" not in names
    assert names.index("route_counters") < names.index("phase_drain")


def test_phase_only_drain_synchronizes_before_polling_events(
    tmp_path: Path,
) -> None:
    calls = Calls()
    synchronized = False

    def synchronize() -> dict[str, int]:
        nonlocal synchronized
        synchronized = True
        calls.values.append(("route_counters", None))
        return {"rounds_claimed": 0}

    def drain_after_sync() -> FakeDrain:
        calls.values.append(("phase_drain", None))
        return FakeDrain(still_pending=0 if synchronized else 1)

    functions = calls.functions()
    functions = bridge_module.ProbeFunctions(
        **{
            **functions.__dict__,
            "route_counters": synchronize,
            "phase_drain": drain_after_sync,
        }
    )
    control = tmp_path / "control.json"
    bridge = bridge_module.ProbeControlBridge(
        functions=functions,
        control_path=control,
        route_output_path=tmp_path / "routes.jsonl",
        rank=0,
    )
    bridge.mark_worker_ready()
    provenance = bridge.snapshot()["provenance"]
    write_command(
        control,
        1,
        "arm",
        epoch="adaptive",
        request_slot=0,
        request_key="a",
        capture_routes=False,
    )
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(control, 2, "disarm")
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(control, 3, "drain", provenance=provenance)
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "ok"
    assert snapshot["route"]["drain"]["state"] == "skipped-phase-only"


@pytest.mark.parametrize("capture_routes", ["false", 0, None])
def test_phase_only_mode_requires_an_exact_boolean(
    tmp_path: Path, capture_routes: Any
) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    write_command(
        control,
        1,
        "arm",
        epoch="adaptive",
        request_slot=0,
        request_key="a",
        capture_routes=capture_routes,
    )
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert "capture_routes must be a boolean" in snapshot["last_error"]


def test_capture_mode_cannot_change_mid_epoch(tmp_path: Path) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    write_command(
        control,
        1,
        "arm",
        epoch="adaptive",
        request_slot=0,
        request_key="a",
        capture_routes=False,
    )
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(
        control,
        2,
        "next_request",
        request_slot=1,
        request_key="b",
        capture_routes=True,
    )
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert "cannot change" in snapshot["last_error"]


def test_phase_only_drain_fails_on_any_route_activity(
    tmp_path: Path,
) -> None:
    calls = Calls()
    functions = calls.functions()
    functions = bridge_module.ProbeFunctions(
        **{
            **functions.__dict__,
            "route_counters": lambda: {"rounds_claimed": 1},
        }
    )
    control = tmp_path / "control.json"
    bridge = bridge_module.ProbeControlBridge(
        functions=functions,
        control_path=control,
        route_output_path=tmp_path / "routes.jsonl",
        rank=0,
    )
    bridge.mark_worker_ready()
    provenance = bridge.snapshot()["provenance"]
    write_command(
        control,
        1,
        "arm",
        epoch="adaptive",
        request_slot=0,
        request_key="a",
        capture_routes=False,
    )
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(control, 2, "disarm")
    assert bridge.snapshot()["last_result"] == "ok"
    write_command(control, 3, "drain", provenance=provenance)
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert "dirty route counters" in snapshot["last_error"]


def test_drain_orders_sync_before_phase_poll(tmp_path: Path) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    provenance = bridge.snapshot()["provenance"]
    (tmp_path / "routes.jsonl").write_text(
        '{"round":0}\n', encoding="utf-8"
    )
    write_command(control, 1, "drain", provenance=provenance)
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "ok"
    assert snapshot["route"]["counters"] == {"rounds_completed": 125}
    assert snapshot["route"]["drain"]["records"] == 1
    names = [item[0] for item in calls.values]
    assert names.index("route_counters") < names.index("phase_drain")
    assert names.index("phase_drain") < names.index("route_drain")


def test_verify_clean_synchronizes_and_accepts_zero_counters(
    tmp_path: Path,
) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    write_command(control, 1, "verify_clean", stream_slot=0)
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "ok"
    assert snapshot["route"]["counters"]["wrong_phase"] == 0
    assert [item[0] for item in calls.values] == [
        "route_disarm",
        "phase_disarm",
        "route_counters",
    ]


def test_verify_clean_fails_closed_on_prearm_counter(
    tmp_path: Path,
) -> None:
    calls = Calls()
    functions = calls.functions()
    functions = bridge_module.ProbeFunctions(
        **{
            **functions.__dict__,
            "route_counters": lambda: {"wrong_phase": 75},
        }
    )
    control = tmp_path / "control.json"
    bridge = bridge_module.ProbeControlBridge(
        functions=functions,
        control_path=control,
        route_output_path=tmp_path / "routes.jsonl",
        rank=0,
    )
    bridge.mark_worker_ready()
    write_command(control, 1, "verify_clean", stream_slot=0)
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert "not clean" in snapshot["last_error"]
    assert snapshot["route"]["counters"] == {"wrong_phase": 75}


def test_sequence_gap_fails_without_executing(tmp_path: Path) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    write_command(control, 2, "disarm")
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert "sequence gap" in snapshot["last_error"]
    assert calls.values == []


def test_repeated_sequence_is_idempotent(tmp_path: Path) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    write_command(control, 1, "disarm")
    bridge.snapshot()
    first = list(calls.values)
    bridge.snapshot()
    assert calls.values == first


def test_arm_rolls_back_phase_if_route_fails(tmp_path: Path) -> None:
    calls = Calls()
    functions = calls.functions()

    def fail_route(**kwargs: Any) -> None:
        del kwargs
        raise RuntimeError("route failure")

    functions = bridge_module.ProbeFunctions(
        **{
            **functions.__dict__,
            "route_arm": fail_route,
        }
    )
    control = tmp_path / "control.json"
    bridge = bridge_module.ProbeControlBridge(
        functions=functions,
        control_path=control,
        route_output_path=tmp_path / "routes.jsonl",
        rank=0,
    )
    bridge.mark_worker_ready()
    write_command(
        control,
        1,
        "arm",
        epoch="q2r",
        request_slot=0,
        request_key="a",
    )
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert calls.values[-1] == ("phase_disarm", None)


@pytest.mark.parametrize(
    "command",
    [
        {"schema": "wrong", "sequence": 1, "action": "disarm"},
        {
            "schema": "sparkring-q2r-control/v1",
            "sequence": 0,
            "action": "disarm",
        },
        {
            "schema": "sparkring-q2r-control/v1",
            "sequence": 1,
            "action": "unknown",
        },
    ],
)
def test_malformed_commands_fail_closed(
    tmp_path: Path, command: dict[str, Any]
) -> None:
    calls = Calls()
    bridge, control = make_bridge(tmp_path, calls)
    control.write_text(json.dumps(command), encoding="utf-8")
    snapshot = bridge.snapshot()
    assert snapshot["last_result"] == "failed"
    assert calls.values == []
