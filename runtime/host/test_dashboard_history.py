"""CPU evidence for bounded raw history, source selection and incremental delivery."""
import copy
import gzip
import json

from runtime.host import dashboard
from runtime.host.dashboard_rates import ViewerRates
from runtime.host.test_dashboard import PAIRS, call, collector, paths_of, run_for, server  # noqa: F401
from runtime.host.test_dashboard_rates import document, member


def rows(found):
    return next(iter(found.history.values()))


def test_sample_and_deployment_bounds_eviction_and_departure(tmp_path):
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    found.mark_viewed()
    found.cycle()
    model = next(iter(found.models.values()))
    start = clock.now
    for i in range(dashboard.HISTORY_SAMPLES + 100):
        model.observe_metrics({"gen": i, "prompt": i, "running": 1}, start + i * 5)
        found.remember(model, start + i * 5)
    assert len(rows(found)) == dashboard.HISTORY_SAMPLES
    assert "output_tps" not in rows(found)[-1]["counters"]["values"]
    for i in range(dashboard.HISTORY_DEPLOYMENTS + 5):
        other = copy.copy(model)
        other.id = f"deployment-{i}"
        found.models[other.id] = other
        found.remember(other, start + i)
    assert len(found.history) == dashboard.HISTORY_DEPLOYMENTS
    assert sum(map(len, found.history.values())) <= dashboard.HISTORY_DEPLOYMENTS * dashboard.HISTORY_SAMPLES
    found.prune_history(start + dashboard.HISTORY_SECONDS + 500)
    assert rows(found)[0]["counters"]["at"] == start + 500
    assert all(row["counters"]["at"] >= start + 500 for row in rows(found))
    found.models.clear()
    found.prune_history(clock.now)
    assert found.history == {}


def test_plugin_choice_deduplication_and_stale_local_remote_fallback(tmp_path):
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    found.mark_viewed()
    found.cycle()
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "one"
    plugin = member(document(clock.now, 100, 200, plugin=True))["plugin_counters"]
    found.node_status = {"telemetry": {"one": plugin}}
    found.remember(model, clock.now)
    assert rows(found)[-1]["source"] == "plugin"
    assert rows(found)[-1]["counters"] == {**plugin, "source": "plugin"}
    count = len(rows(found))
    run_for(found, clock, 10, viewer=True)
    assert len(rows(found)) == count  # repeated plugin delivery and metrics reads add no plugin point
    remote = {**plugin, "at": clock.now, "monotonic": plugin["monotonic"] + 100}
    found.node_status["telemetry"]["one"] = {**plugin, "at": clock.now - 181}
    found.recovery = {model.row["id"]: {"telemetry": remote}}
    found.remember(model, clock.now)
    assert rows(found)[-1]["counters"]["at"] == clock.now
    clock.now += 181
    found.mark_viewed()
    found.cycle()
    assert rows(found)[-1]["source"] == "metrics"
    assert "values" in rows(found)[-1]["counters"]


def test_plugin_row_stores_no_metrics_values_and_windows_never_mix(tmp_path):
    """A plugin sample carries plugin counters only; the /metrics read of the same
    cycle is either absent or lives in its own metrics-labeled sample."""
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    found.mark_viewed()
    found.cycle()
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "one"
    model.observe_metrics({"gen": 4000, "prompt": 993889, "running": 1, "started": 7}, clock.now)
    plugin = member(document(clock.now, 522849, 100, plugin=True))["plugin_counters"]
    found.node_status = {"telemetry": {"one": plugin}}
    found.remember(model, clock.now)
    row = rows(found)[-1]
    assert row["source"] == "plugin"
    assert row["counters"]["prompt"] == 522849
    assert "values" not in row["counters"] and "plugin_counters" not in row
    viewer = ViewerRates()
    viewer.ingest(model.id, row)
    state = viewer.members[model.id]
    assert state["key"] == ("plugin", plugin["run_id"])
    assert all(sample["source"] == "plugin" for sample in state["samples"])
    clock.now += 10
    found.node_status = {}
    model.observe_metrics({"gen": 4010, "prompt": 993899, "running": 1, "started": 7}, clock.now)
    found.remember(model, clock.now)
    row = rows(found)[-1]
    assert row["source"] == "metrics"
    assert row["counters"]["values"]["prompt"] == 993899
    viewer.ingest(model.id, row)
    state = viewer.members[model.id]
    assert state["key"] == ("metrics", 7)
    assert all(sample["source"] == "metrics" for sample in state["samples"])
    assert state["history"][-2][1:] == [None] * 6  # the source switch closes the rate window


def test_unviewed_plugin_and_expired_plugin_make_no_requests(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    found.reload_sources()
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "one"
    found.node_status = {"telemetry": {"one": member(document(clock.now, 10, 10, plugin=True))["plugin_counters"]}}
    run_for(found, clock, 180)
    assert api.requests == []
    api.requests.clear()
    run_for(found, clock, 61)
    assert api.requests == []


def test_quiet_and_measurement_holds_preserve_a_gap_and_stop_requests(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    run_for(found, clock, 20, viewer=True)
    count = len(rows(found))
    api.requests.clear()
    hold = dashboard.quiet_hold(found.quiet_path, "history", now=clock.now)
    found.quiet_path.write_text("operator pause")
    run_for(found, clock, 20, viewer=True)
    dashboard.quiet_release(found.quiet_path, hold["holder"])
    run_for(found, clock, 20, viewer=True)
    assert api.requests == [] and len(rows(found)) == count
    found.quiet_path.unlink()
    run_for(found, clock, 10, viewer=True)
    assert list(rows(found))[count]["boundary"]
    viewer = ViewerRates()
    for row in rows(found):
        viewer.ingest("one", row)
    history = viewer.members["one"]["history"]
    assert any(row[1:] == [None] * 6 for row in history)
    assert history[-1][1] == 30
    assert history[-2][1] is None
    clock.now += dashboard.HISTORY_SECONDS + 1
    found.quiet_path.write_text("pause")
    found.mark_viewed()
    found.cycle()
    assert not rows(found)


def test_restart_decrease_and_source_boundaries_survive_replay(tmp_path):
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    found.mark_viewed()
    found.cycle()
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "one"
    for i, (gen, run, plugin) in enumerate([(100, 1, False), (200, 1, False), (1, 1, False),
                                           (1000, 2, False), (2000, 2, True), (2100, 2, True)]):
        clock.now += 10
        raw = member(document(clock.now, gen, gen, run=run, plugin=plugin))
        model.observe_metrics({**raw["counters"]["values"], "started": run}, clock.now)
        found.node_status = {"telemetry": {"one": raw.get("plugin_counters")}}
        found.remember(model, clock.now)
    viewer = ViewerRates()
    for row in rows(found):
        viewer.ingest(model.id, row)
    history = viewer.members[model.id]["history"]
    assert sum(row[1:] == [None] * 6 for row in history) == 4
    assert history[-1][1] == 10
    assert all(row[1] is None or row[1] <= 10 for row in history)


def test_history_http_compression_cursor_and_no_extra_upstream_request(request):
    port, found = request.getfixturevalue("server")()
    before = found.client.counters()["requests"].copy()
    response, body = call(port, "GET", "/api/history", headers={"Accept-Encoding": "gzip"})
    assert response.getheader("Content-Encoding") == "gzip"
    history = json.loads(gzip.decompress(body))
    cursor = history["history_cursor"]
    for _ in range(3):
        response, body = call(port, "GET", f"/api/status?since={cursor}")
        assert all(not m["counter_samples"] for g in json.loads(body)["groups"] for m in g["members"])
    assert found.client.counters()["requests"] == before
    found.clock.now += 5
    found.mark_viewed()
    found.cycle()
    response, body = call(port, "GET", f"/api/status?since={cursor}")
    assert all(len(m["counter_samples"]) == 1 for g in json.loads(body)["groups"] for m in g["members"])
    assert call(port, "GET", "/api/status?since=bad")[0].status == 400
    assert call(port, "GET", "/api/history", host="untrusted.example")[0].status == 421


def test_delta_recovery_is_bounded_and_marks_missing_samples(tmp_path):
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    run_for(found, clock, 1000, viewer=True)
    delta = member(found.snapshot(since=0))["counter_samples"]
    assert len(delta) == dashboard.HISTORY_DELTA_SAMPLES and delta[0]["boundary"]
    assert not rows(found)[-dashboard.HISTORY_DELTA_SAMPLES]["boundary"]


def test_backward_wall_clock_drops_future_records_and_marks_boundary(tmp_path):
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    run_for(found, clock, 100, viewer=True)
    clock.now -= 40
    found.mark_viewed()
    found.cycle()
    assert all(row["observed_at"] <= clock.now for row in rows(found))
    clock.now += 5
    found.mark_viewed()
    found.cycle()
    assert rows(found)[-1]["boundary"]


def test_remote_plugin_cache_with_recovery_off_and_quiet_mode(tmp_path):
    from runtime.host import dashboard_admin
    found, api, clock = collector(tmp_path, rows=PAIRS[3:])
    found.reload_sources()
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "remote-deployment"
    plugin = member(document(clock.now, 100, 200, plugin=True))["plugin_counters"]
    source = {"node_id": "remote-api-head", "observed_at": clock.now,
              "telemetry": {"remote-deployment": {**plugin, "private": "not public"}}}
    found.node_reports_path = tmp_path / "nodes.json"
    cached = dashboard_admin.cached_node_rows(tmp_path, found.node_reports_path, {"remote": source}, clock.now)
    found.node_reports_path.write_text(json.dumps({"schema": "sparkring-dashboard-nodes/v1", "nodes": cached}))
    assert "not public" not in found.node_reports_path.read_text()
    assert found.recovery == {}
    found.mark_viewed()
    found.cycle()
    assert member(found.snapshot())["plugin_counters"] == plugin
    assert rows(found)[-1]["source"] == "plugin"
    # Cached telemetry changes neither the endpoints nor their request budget.
    assert sorted(paths_of(api)) == ["/metrics", "/v1/models", "/v1/sparkring/status"]
    count = len(rows(found))
    api.requests.clear()
    found.quiet_path.write_text("operator pause")
    run_for(found, clock, 60, viewer=True)
    assert api.requests == [] and len(rows(found)) == count
    assert member(found.snapshot())["plugin_counters"] is None
    clock.now = plugin["at"] + 181
    found.mark_viewed()
    found.cycle()
    assert member(found.snapshot())["plugin_counters"] is None
    assert api.requests == []
    found.quiet_path.unlink()
    found.mark_viewed()
    found.cycle()
    assert rows(found)[-1]["source"] == "metrics"
