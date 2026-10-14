"""State durations survive record rewrites and collector restarts without extra reads."""
import json

import pytest

from runtime.host import dashboard, dashboard_admin, recovery, startup
from runtime.host.test_dashboard import PAIRS, collector
from runtime.host.test_dashboard_admin import records  # noqa: F401


def member(found):
    return found.snapshot()["groups"][0]["members"][0]


@pytest.fixture
def down_record(records):  # noqa: F811
    directory = records / "deployments" / "whole"
    receipt = directory / "operations" / "0002-down.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({"complete": True, "completed_at": 1_790_000_000.0}))
    (directory / "state.json").write_text(json.dumps({
        "generation": 2, "operation": "down", "complete": True, "receipt": "operations/0002-down.json"}))
    return directory


def test_down_since_survives_recover_off_refresh_and_restart(down_record, tmp_path, monkeypatch):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    clock.now = 1_790_003_600.0
    monkeypatch.setattr(recovery, "path", lambda: tmp_path / "recovery.json")
    monkeypatch.setattr(recovery, "supported", lambda lock: True)

    def export():
        site = dashboard_admin.site_from_records(down_record.parent.parent)
        site["deployments"] = site["deployments"][:1]
        found.sources[0].site_path.write_text(json.dumps(site))
        found.request_reload()
        found.cycle()

    export()
    assert member(found)["state_since"] == 1_790_000_000.0
    recovery.set_enabled(False, directory=down_record)
    recovery.update(down_record, checked_at=clock.now)
    # Status writers may replace the same record; its mtime is not a stop event.
    state = down_record / "state.json"
    state.write_text(state.read_text())
    clock.now += 54
    export()
    assert member(found)["state_since"] == 1_790_000_000.0
    restarted = dashboard.Collector(found.sources, found.client, clock=clock,
                                   node_status_path=None, node_reports_path=None, recovery_path=None,
                                   quiet_path=tmp_path / "quiet")
    restarted.cycle()
    assert member(restarted)["state_since"] == 1_790_000_000.0
    assert api.requests == []


def test_serving_since_uses_process_start_without_runtime_status(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    key = (PAIRS[0][1], PAIRS[0][2])
    api.bodies[(*key, "/v1/sparkring/status")] = b"{}"
    found.mark_viewed()
    found.cycle()
    assert member(found)["state_since"] == 1_790_000_000.0
    clock.now += 60
    found.request_reload()
    found.request_runtime(member(found)["id"])
    found.mark_viewed()
    found.cycle()
    assert member(found)["state_since"] == 1_790_000_000.0
    # A real model restart establishes a new start; a collector restart does not.
    api.started[key] = clock.now
    clock.now += 2
    found.cycle()
    assert member(found)["state_since"] == api.started[key]
    restarted = dashboard.Collector(found.sources, found.client, clock=clock,
                                   node_status_path=None, node_reports_path=None, recovery_path=None,
                                   quiet_path=tmp_path / "quiet")
    restarted.mark_viewed()
    restarted.cycle()
    assert member(restarted)["state_since"] == api.started[key]


@pytest.mark.parametrize("timestamp", [None, 0, -1, True, "yesterday", float("nan"), float("inf")])
def test_missing_or_invalid_stop_time_is_not_observation_time(tmp_path, timestamp):
    found, _, _ = collector(tmp_path, rows=PAIRS[:1])
    site = json.loads(found.sources[0].site_path.read_text())
    site["deployments"][0].update(stopped=True, stopped_at=timestamp)
    found.sources[0].site_path.write_text(json.dumps(site))
    found.cycle()
    assert member(found)["state_since"] is None


@pytest.mark.parametrize("operation,field", [("up", "serving_at"), ("down", "stopped_at")])
def test_state_completion_and_incomplete_operation(down_record, operation, field):
    state = down_record / "state.json"
    state.write_text(json.dumps({"operation": operation, "complete": True, "completed_at": 1234.0}))
    assert dashboard_admin.operation_times(down_record) == {field: 1234.0}
    state.write_text(json.dumps({"operation": operation, "complete": False, "completed_at": 1234.0}))
    assert dashboard_admin.operation_times(down_record) == {}


def test_legacy_or_incomplete_receipt_has_no_invented_time(down_record):
    receipt = down_record / "operations" / "0002-down.json"
    for document in ({"complete": True}, {"complete": False, "completed_at": 1234.0}):
        receipt.write_text(json.dumps(document))
        assert dashboard_admin.operation_times(down_record) == {}
    receipt.unlink()
    assert dashboard_admin.operation_times(down_record) == {}


def test_serving_uses_completed_up_when_api_has_no_start_time(tmp_path):
    found, api, _ = collector(tmp_path, rows=PAIRS[:1])
    site = json.loads(found.sources[0].site_path.read_text())
    site["deployments"][0]["serving_at"] = 1_790_000_000.0
    found.sources[0].site_path.write_text(json.dumps(site))
    key = (PAIRS[0][1], PAIRS[0][2])
    api.bodies[(*key, "/metrics")] = b"vllm:num_requests_running 0\n"
    api.bodies[(*key, "/v1/sparkring/status")] = b"{}"
    found.mark_viewed()
    found.cycle()
    assert member(found)["state_since"] == 1_790_000_000.0


def test_serving_does_not_reuse_an_older_runtime_start(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    api.started[(PAIRS[0][1], PAIRS[0][2])] = clock.now - 10
    found.mark_viewed()
    found.cycle()
    assert member(found)["state_since"] == clock.now - 10


@pytest.mark.parametrize("source", ["container", "operation"])
def test_startup_since_survives_phases_refresh_and_collector_restart(tmp_path, source):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    found.live_path = tmp_path / "live"
    site_path = found.sources[0].site_path
    site = json.loads(site_path.read_text())
    row = site["deployments"][0]
    row["deployment_id"] = "one"
    site_path.write_text(json.dumps(site))
    api.behaviour[tuple(PAIRS[0][1:3])] = "refused"
    start = clock.now - 3600

    def publish(phase):
        observation = dict(schema=startup.SCHEMA, phase=phase, at=clock.now,
                           started_at=start, rank=0, run_id=str(start))
        live = {"at": clock.now}
        if source == "container":
            live["startup"] = {"one": [observation]}
        else:
            live["operations"] = [{**observation, "deployment": row}]
        found.live_path.write_text(json.dumps(live))

    for phase in ("starting", "loading", "graphs"):
        publish(phase)
        site_path.write_text(json.dumps(site))
        found.request_reload()
        found.mark_viewed()
        found.cycle()
        assert member(found)["state"] == phase
        assert member(found)["state_since"] == start
        clock.now += 2

    publish("loading")
    restarted = dashboard.Collector(found.sources, found.client, clock=clock,
                                   node_status_path=None, node_reports_path=None, recovery_path=None,
                                   quiet_path=tmp_path / "quiet", live_path=found.live_path,
                                   viewer_path=tmp_path / "viewer")
    restarted.mark_viewed()
    restarted.cycle()
    assert member(restarted)["state"] == "loading"
    assert member(restarted)["state_since"] == start
    # A new operation or container run establishes a new event time.
    start = clock.now
    clock.now += 2
    publish("starting")
    restarted.mark_viewed()
    restarted.cycle()
    assert member(restarted)["state_since"] == start


def test_starting_without_a_recorded_event_has_no_duration(tmp_path):
    found, api, _ = collector(tmp_path, rows=PAIRS[:1])
    api.behaviour[tuple(PAIRS[0][1:3])] = "starting"
    found.mark_viewed()
    found.cycle()
    assert member(found)["state"] == "not_ready"
    assert member(found)["state_since"] is None
