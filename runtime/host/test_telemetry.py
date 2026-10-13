import json
import os

import pytest

from runtime.host import dashboard, node, telemetry


def snapshot(at=1000):
    return {"schema": "sparkring-stats/v1", "at": at, "monotonic": at - 100, "run_id": "run-one",
            "deployment_id": "owned", "prompt": 100, "gen": 30, "accepted": 20, "drafts": 10,
            "draft_tokens": 30, "steps": 10, "step_seconds": .25, "ttft_sum": .2, "ttft_count": 1,
            "running": 1, "waiting": 0, "kv": .25}


def test_agent_carries_bounded_local_file_and_ignores_bad_inputs(tmp_path):
    directory = tmp_path / telemetry.ROOT.lstrip("/") / "owned"
    directory.mkdir(parents=True)
    path = directory / "snapshot.json"
    value = snapshot()
    path.write_text(json.dumps(value))
    assert telemetry.read_snapshots(root=tmp_path, now=lambda: 1001)["owned"]["read_at"] == 1001
    for bad in ({**value, "deployment_id": "another"}, {**value, "at": 0},
                {**value, "gen": float("nan")}, {**value, "gen": -1}, []):
        path.write_text(json.dumps(bad))
        assert telemetry.read_snapshots(root=tmp_path, now=lambda: 1001) == {}
    path.write_text("x" * 5000)
    assert telemetry.read_snapshots(root=tmp_path, now=lambda: 1001) == {}


def test_node_snapshot_and_dashboard_recovery_file_forward_raw_values(tmp_path, monkeypatch):
    value = snapshot()
    monkeypatch.setattr(telemetry, "read_snapshots", lambda **kwargs: {"owned": value})
    assert node.snapshot(root=tmp_path)["telemetry"]["owned"] == value
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(dict(schema="sparkring-recovery/v1", deployments={
        "/deployment/model": dict(finding=dict(state="serving", telemetry=value), checked_at=1001)})))
    assert dashboard.recovery_findings(path)["model"]["telemetry"] == value


def test_recovery_reuses_only_its_existing_head_status_call(monkeypatch):
    from runtime.host import recovery
    value = snapshot()
    calls = []
    monkeypatch.setattr(recovery, "node_documents", lambda hosts, **kwargs:
                        calls.append(tuple(hosts)) or {"head": {"telemetry": {"owned": value}}})
    monkeypatch.setattr(recovery, "probe_containers", lambda *args, **kwargs: [{"running": True}])
    monkeypatch.setattr(recovery, "assess", lambda *args, **kwargs: {"state": "serving"})
    lock = {"id": "owned", "site": {"ranks": [{"host": "head"}, {"host": "worker"}]}}
    finding = recovery.observe("unused", lock, {"api_url": "http://unused"}, {}, invoke=None,
                               api=lambda url: {"ok": True})
    assert finding["telemetry"] == value and calls == [("head", "worker")]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires a POSIX FIFO")
def test_agent_does_not_follow_symlinks_or_block_on_fifo(tmp_path):
    directory = tmp_path / telemetry.ROOT.lstrip("/") / "owned"
    directory.mkdir(parents=True)
    path = directory / "snapshot.json"
    os.mkfifo(path)
    assert telemetry.read_snapshots(root=tmp_path) == {}
    path.unlink()
    other = tmp_path / "other.json"
    other.write_text(json.dumps(snapshot()))
    path.symlink_to(other)
    assert telemetry.read_snapshots(root=tmp_path, now=lambda: 1001) == {}
