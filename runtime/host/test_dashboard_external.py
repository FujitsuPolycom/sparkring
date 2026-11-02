"""Endpoints served outside the installer (``sparkring dashboard external``): the list file, Flare's
metrics, the collector's read-only polling of each endpoint, the status document's ``external`` cards
and the admin commands. No host is contacted."""
import json
from pathlib import Path
import types

import pytest

from runtime.host import dashboard, dashboard_admin
from runtime.host.test_dashboard import PAIRS, Clock, FakeAPI, fabric_file, run_for, site_file
from runtime.host.test_dashboard_unmanaged import install_reports

FLARE_HOST, FLARE_PORT = "192.0.2.60", 8000


def flare_metrics(generated=12000.0, uptime=600.0):
    return "\n".join([
        "# HELP flare_http_requests_total HTTP generation or token-count requests by API.",
        "# TYPE flare_http_requests_total counter",
        'flare_http_requests_total{api="chat"} 167',
        'flare_http_requests_total{api="completions"} 2',
        f"flare_generated_tokens_total {generated}",
        "flare_prefill_tokens_total 48000",
        "flare_requests_completed_total 150",
        "flare_requests_failed_total 1",
        "flare_speculative_offered_drafts_total 1000",
        "flare_speculative_accepted_drafts_total 650",
        "flare_engram_row_cache_hits_total 5055004",
        "flare_engram_row_cache_misses_total 1447717",
        "flare_cache_lookups_total 40",
        "flare_cache_hits_total 10",
        "flare_active_requests 2",
        "flare_pending_requests 0",
        "flare_http_request_capacity 8",
        f"flare_uptime_seconds {uptime}",
        "flare_generated_tokens_last_minute 2400",
        "flare_total_kv_tokens 1900544",
        "flare_free_kv_tokens 1425408",
        "flare_cached_kv_tokens 300000",
        "flare_unknown_series 7",
    ]) + "\n"


def external_file(tmp_path, rows=None):
    path = tmp_path / "external.json"
    rows = rows if rows is not None else [{"id": "flare-spark-6", "label": "Flare · Qwen3.8-Flash-Next (Spark 6)",
                                           "kind": "flare", "host": FLARE_HOST, "port": FLARE_PORT}]
    path.write_text(json.dumps({"schema": dashboard.EXTERNAL_SCHEMA, "endpoints": rows}), encoding="utf-8")
    return path


def flare_collector(tmp_path, rows=None, pairs=PAIRS):
    clock = Clock()
    api = FakeAPI(clock)
    for _, host, port, model in pairs:
        api.models[(host, port)] = model
    api.bodies[(FLARE_HOST, FLARE_PORT, "/metrics")] = flare_metrics().encode()
    api.bodies[(FLARE_HOST, FLARE_PORT, "/v1/models")] = json.dumps({"data": [
        {"id": "qwen3.8-flash", "root": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD",
         "max_model_len": 262144}]}).encode()
    client = dashboard.ReadOnlyClient(opener=api)
    sources = [dashboard.Source(fabric_file(tmp_path), site_file(tmp_path, rows=pairs))]
    found = dashboard.Collector(sources, client, quiet_path=tmp_path / "quiet", node_status_path=None,
                                external_path=external_file(tmp_path, rows), clock=clock)
    return found, api, clock


def test_flare_metrics_are_read_by_short_name_and_summed_over_labels():
    metrics = dashboard.parse_flare_metrics(flare_metrics())
    assert metrics["http_requests"] == 169 and metrics["generated_tokens"] == 12000
    assert metrics["active_requests"] == 2 and metrics["request_capacity"] == 8
    assert metrics["kv_tokens_total"] == 1900544 and metrics["kv_tokens_free"] == 1425408
    assert "unknown_series" not in metrics and "flare_unknown_series" not in metrics
    assert dashboard.parse_flare_metrics("") == {}


def test_the_external_list_is_validated(tmp_path):
    assert dashboard.read_external(tmp_path / "missing.json") == []
    rows = dashboard.read_external(external_file(tmp_path))
    assert rows == [{"id": "flare-spark-6", "label": "Flare · Qwen3.8-Flash-Next (Spark 6)", "kind": "flare",
                     "host": FLARE_HOST, "port": FLARE_PORT}]
    good = rows[0]
    for bad, match in (({**good, "id": "Flare 6"}, "distinct short lowercase name"),
                       ({**good, "label": ""}, "label"),
                       ({**good, "kind": "vllm"}, "kind"),
                       ({**good, "host": "spark-6"}, "IP address"),
                       ({**good, "port": 70000}, "port"),
                       ({**good, "port": True}, "port"),
                       ({**good, "position": -1}, "position"),
                       ({**good, "position": "6"}, "position"),
                       ({**good, "position": True}, "position")):
        with pytest.raises(ValueError, match=match):
            dashboard.read_external(external_file(tmp_path, [bad]))
    with pytest.raises(ValueError, match="distinct"):
        dashboard.read_external(external_file(tmp_path, [good, good]))
    assert dashboard.read_external(external_file(tmp_path, [{**good, "position": 6}]))[0]["position"] == 6
    (tmp_path / "other.json").write_text(json.dumps({"schema": "x", "endpoints": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected sparkring-dashboard-external/v1"):
        dashboard.read_external(tmp_path / "other.json")


def test_a_serving_flare_endpoint_gets_a_read_only_card_with_its_figures(tmp_path):
    found, api, clock = flare_collector(tmp_path)
    run_for(found, clock, 6, viewer=True)
    assert set(found.client.allowed) >= {FLARE_HOST}
    paths = {(r.get_full_url()) for r in api.requests if FLARE_HOST in r.get_full_url()}
    assert paths == {f"http://{FLARE_HOST}:{FLARE_PORT}/metrics", f"http://{FLARE_HOST}:{FLARE_PORT}/v1/models"}
    assert all(r.get_method() == "GET" and r.data is None for r in api.requests)
    [card] = found.snapshot()["external"]
    assert card["id"] == "external:flare-spark-6" and card["state"] == "serving" and card["state_label"] == "Serving"
    assert card["endpoint"] == f"http://{FLARE_HOST}:{FLARE_PORT}/v1" and card["kind"] == "flare"
    assert (card["model"], card["root"], card["max_model_len"]) == (
        "qwen3.8-flash", "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD", 262144)
    stats = card["stats"]
    assert stats["active_requests"] == 2 and stats["request_capacity"] == 8 and stats["tokens_last_minute"] == 2400
    assert stats["kv_in_use"] == pytest.approx(0.25) and stats["uptime_seconds"] == 600
    assert stats["requests_completed"] == 150 and stats["requests_failed"] == 1 and stats["http_requests"] == 169
    assert stats["draft_acceptance"] == pytest.approx(0.65) and stats["prefix_hit_rate"] == pytest.approx(0.25)
    assert stats["row_cache_hit_rate"] == pytest.approx(5055004 / (5055004 + 1447717))
    # The counter does not move in this fake, so no rate yet; a moving counter gives tokens per second.
    assert stats["output_tps"] == 0
    api.bodies[(FLARE_HOST, FLARE_PORT, "/metrics")] = flare_metrics(generated=12000.0 + 300.0).encode()
    run_for(found, clock, 2, viewer=True)
    [card] = found.snapshot()["external"]
    assert 100 <= card["stats"]["output_tps"] <= 300
    # The event log names the endpoint by its label.
    assert any(e["stack"] == "Flare · Qwen3.8-Flash-Next (Spark 6)" and e["kind"] == "serving" for e in found.events)
    # The ring's own models are untouched by the external list.
    assert len(found.models) == 4 and all(m.state == "serving" for m in found.models.values())
    assert card["position"] is None and all(len(g["members"]) == 4 for g in found.snapshot()["groups"])


def test_an_endpoint_listed_with_a_position_joins_the_ring_figure(tmp_path):
    """With a position, the endpoint is a one-Spark member of that ring: on the figure, in the legend and
    among the model cards, with its address as the head and its figures as live rates; it leaves the
    Other endpoints list."""
    row = {"id": "flare-spark-6", "label": "Flare · Qwen3.8-Flash-Next (Spark 6)", "kind": "flare",
           "host": FLARE_HOST, "port": FLARE_PORT, "position": 6}
    found, api, clock = flare_collector(tmp_path, rows=[row], pairs=PAIRS[:3])
    run_for(found, clock, 3, viewer=True)
    api.bodies[(FLARE_HOST, FLARE_PORT, "/metrics")] = flare_metrics(generated=12000.0 + 400.0).encode()
    run_for(found, clock, 2, viewer=True)
    document = found.snapshot()
    assert document["external"] == []
    [group] = document["groups"]
    [member] = [m for m in group["members"] if m["id"] == "external:flare-spark-6"]
    assert member["external"] is True and member["kind"] == "flare" and member["name"] == "tp1-6-6"
    assert member["arc"]["positions"] == [6] and member["arc"]["source"] == "external" and member["arc"]["color"] == 3
    assert member["head"]["address"] == FLARE_HOST and member["head"]["port"] == FLARE_PORT
    assert member["head"]["name"] == document["ring"]["positions"][6]["name"]
    assert (member["host"], member["port"], member["position"]) == (FLARE_HOST, FLARE_PORT, 6)
    assert member["state"] == "serving" and member["model"] == "qwen3.8-flash" and member["model_short"]
    assert member["layout"] == {"label": "Flare", "count": 1, "tp": 1, "dcp": None}
    assert member["live"]["running"] == 2 and 100 <= member["live"]["output_tps"] <= 400
    assert member["live"]["output_tps"] == member["stats"]["output_tps"]
    assert member["totals"]["output_tokens"] == 12400 and member["counters"] is None and member["counter_samples"] == []
    assert [m["arc"]["first"] for m in group["members"]] == [0, 2, 4, 6]
    # The ring figure's position 6 belongs to it; its port is among the discovered API ports.
    position = document["ring"]["positions"][6]
    assert position["member"] == "external:flare-spark-6" and position["up"] is True
    assert FLARE_PORT in document["ring"]["discovery"]["ports"]
    # The host agents' report of an engine-unknown container on that Spark is the endpoint's own server:
    # no separate "GPU in use" row, and the stopped install record that row carried for the Spark is named
    # on the endpoint's card instead. A report from a Spark without a listed endpoint keeps its row.
    fabric = found.sources[0].fabric
    site = json.loads((tmp_path / "site.json").read_text(encoding="utf-8"))
    site["deployments"].append({"id": "glm-on-6-7", "profile": "glm", "model": "GLM-5.3-Flash-TP2", "positions": [6, 7],
                                "api_host": FLARE_HOST, "api_port": 8000, "stopped": True, "containers": []})
    (tmp_path / "site.json").write_text(json.dumps(site), encoding="utf-8")
    install_reports(found, [{"node_id": fabric["positions"][p]["node_id"], "observed_at": clock.now,
                             "model_containers": [{"managed": False, "engine": "unknown", "name": "flare",
                                                   "image": "ghcr.io/local-inference-lab/flare:latest",
                                                   "image_id": "4eab4fbd1234"}]} for p in (6, 7)], tmp_path)
    clock.now += dashboard.FILES_SECONDS + 1
    run_for(found, clock, 2, viewer=True)
    members = found.snapshot()["groups"][0]["members"]
    assert [m["arc"]["positions"] for m in members if m["gpu_job"]] == [[7]]
    [flare] = [m for m in members if m.get("external")]
    assert flare["arc"]["positions"] == [6] and [r["id"] for r in flare["stopped_records"]] == ["glm-on-6-7"]
    assert [m["arc"]["first"] for m in members] == [0, 2, 4, 6, 7]
    # A position beyond the fabric keeps the endpoint under Other endpoints.
    clock.now += dashboard.FILES_SECONDS + 1
    external_file(tmp_path, [{**row, "position": 8}])
    run_for(found, clock, 2, viewer=True)
    document = found.snapshot()
    assert [e["id"] for e in document["external"]] == ["external:flare-spark-6"]
    assert all(m["id"] != "external:flare-spark-6" for m in document["groups"][0]["members"])
    assert [m["arc"]["positions"] for m in document["groups"][0]["members"] if m["gpu_job"]] == [[6], [7]]
    assert all(r["id"] == "glm-on-6-7" for m in document["groups"][0]["members"] if m["gpu_job"]
               for r in m["stopped_records"])


def test_an_endpoint_that_does_not_answer_is_stopped_or_offline_and_reads_nothing_more(tmp_path):
    found, api, clock = flare_collector(tmp_path)
    api.behaviour[(FLARE_HOST, FLARE_PORT)] = "refused"
    run_for(found, clock, 4, viewer=True)
    [card] = found.snapshot()["external"]
    assert card["state"] == "stopped" and card["stats"] is None and card["model"] is None
    api.behaviour[(FLARE_HOST, FLARE_PORT)] = "timeout"
    run_for(found, clock, dashboard.FAILED_SECONDS + 2, viewer=True)
    assert found.snapshot()["external"][0]["state"] == "offline"
    api.behaviour[(FLARE_HOST, FLARE_PORT)] = "starting"
    run_for(found, clock, dashboard.FAILED_SECONDS + 2, viewer=True)
    assert found.snapshot()["external"][0]["state"] == "not_ready"
    # Once the viewer's lease has lapsed, nothing is asked.
    run_for(found, clock, dashboard.VIEWER_SECONDS + 2, viewer=False)
    before = len(api.requests)
    run_for(found, clock, 30, viewer=False)
    assert len(api.requests) == before


def test_the_list_is_re_read_when_the_file_changes_and_a_bad_file_is_reported(tmp_path):
    found, api, clock = flare_collector(tmp_path)
    run_for(found, clock, 3, viewer=True)
    assert [e["id"] for e in found.snapshot()["external"]] == ["external:flare-spark-6"]
    clock.now += dashboard.FILES_SECONDS + 1
    external_file(tmp_path, [])
    run_for(found, clock, 2, viewer=True)
    assert found.snapshot()["external"] == [] and found.snapshot()["external_problem"] is None
    clock.now += dashboard.FILES_SECONDS + 1
    (tmp_path / "external.json").write_text("{not json", encoding="utf-8")
    run_for(found, clock, 2, viewer=True)
    assert found.snapshot()["external"] == [] and found.snapshot()["external_problem"]


def test_the_admin_commands_add_list_and_remove_endpoints(tmp_path, capsys):
    path = tmp_path / "external.json"
    args = types.SimpleNamespace(file=str(path), action="add", id="flare-spark-6", host=FLARE_HOST, port=FLARE_PORT,
                                 label="Flare · Qwen3.8-Flash-Next (Spark 6)", kind="flare")
    assert dashboard_admin.external(args) == 0
    assert dashboard.read_external(path)[0]["host"] == FLARE_HOST
    assert dashboard.main(["external", "--file", str(path), "add", "flare-spark-7", "--host", "192.0.2.61", "--port", "8000",
                           "--label", "Flare (Spark 7)", "--position", "7"]) == 0
    assert [r["id"] for r in dashboard.read_external(path)] == ["flare-spark-6", "flare-spark-7"]
    assert dashboard.read_external(path)[1]["position"] == 7
    assert "flare-spark-7: Flare (Spark 7) (flare) at http://192.0.2.61:8000 on position 7" in capsys.readouterr().out
    assert dashboard.main(["external", "--file", str(path), "add", "flare-spark-7", "--host", "192.0.2.62", "--port", "8001",
                           "--label", "Flare (Spark 7)", "--kind", "openai"]) == 0
    rows = dashboard.read_external(path)
    assert len(rows) == 2 and rows[1] == {"id": "flare-spark-7", "label": "Flare (Spark 7)", "kind": "openai",
                                          "host": "192.0.2.62", "port": 8001}
    assert dashboard.main(["external", "--file", str(path), "list"]) == 0
    assert "flare-spark-7: Flare (Spark 7) (openai) at http://192.0.2.62:8001" in capsys.readouterr().out
    assert dashboard.main(["external", "--file", str(path), "remove", "flare-spark-7"]) == 0
    assert [r["id"] for r in dashboard.read_external(path)] == ["flare-spark-6"]
    assert dashboard.main(["external", "--file", str(path), "remove", "flare-spark-7"]) == 2
    assert "no endpoint flare-spark-7" in capsys.readouterr().err
    assert dashboard.main(["external", "--file", str(path), "add", "Bad Id", "--host", "x", "--port", "1",
                           "--label", "x"]) == 2
    assert Path(path).stat().st_mode & 0o777 in (0o644, 0o666)
