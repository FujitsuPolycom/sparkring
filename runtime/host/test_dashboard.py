"""``sparkring dashboard``: the read-only collector, its request guard, its status document and its web
server, against recorded vLLM and runtime-status payloads and fake model APIs. No host is contacted;
the server tests bind a loopback port.

The guard tests fail if the collector would send anything but a bodyless GET to /metrics,
/v1/models or /v1/sparkring/status, or start a process (an SSH session, docker or ping).
"""
import ast
import base64
import http.client
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from runtime.common import fabric_layout
from runtime.host import dashboard
from runtime.host.test_fabric import document_of

FIXTURES = Path(__file__).with_name("fixtures") / "dashboard"
ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings", "/tokenize",
                   "/detokenize", "/pooling", "/score", "/rerank", "/v1/audio/transcriptions", "/invocations",
                   "/generate", "/v1/messages")
NODE_A_LAN = "192.0.2.50"
# Four pairs on a ring of eight: (positions, API address, port, served model name).
PAIRS = (([0, 1], NODE_A_LAN, 8000, "GLM-5.3-Flash-NVFP4-Spark-TP2"),
         ([2, 3], "192.0.2.12", 8015, "Qwen3.8-Flash-Next-NVFP4-QAD-TP2"),
         ([4, 5], "192.0.2.14", 8020, "MiMo-V2.6-Flash-MOPD-TP2"),
         ([6, 7], "192.0.2.16", 8000, "DeepSeek-V4.1-Flash-TP2"))


# -- fixtures ------------------------------------------------------------------------------
def fabric_file(tmp_path, size=8):
    _, document, _ = document_of(fabric_layout.layout("cycle", size), api_address=NODE_A_LAN)
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def site_file(tmp_path, rows=PAIRS, **extra):
    deployments = []
    for positions, host, port, model in rows:
        deployments.append({"id": f"profile-on-{positions[0]}-{positions[-1]}", "profile": "profile", "model": model,
                            "positions": positions, "api_host": host, "api_port": port, "stopped": False,
                            "containers": [{"position": p, "name": f"sr-{positions[0]}-r{i}"}
                                           for i, p in enumerate(positions)], **extra})
    path = tmp_path / "site.json"
    path.write_text(json.dumps({"schema": dashboard.SITE_SCHEMA, "deployments": deployments}), encoding="utf-8")
    return path


def metrics_text(model, generated, started, running=1.0):
    """A vLLM /metrics excerpt with the series the dashboard reads."""
    labels = f'engine="0",model_name="{model}"'
    return "\n".join([
        "# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.",
        f"process_start_time_seconds {started}",
        f"vllm:num_requests_running{{{labels}}} {running}",
        f"vllm:num_requests_waiting{{{labels}}} 0.0",
        f"vllm:kv_cache_usage_perc{{{labels}}} 0.25",
        f"vllm:generation_tokens_total{{{labels}}} {generated}",
        f"vllm:prompt_tokens_total{{{labels}}} {generated * 4}",
        f"vllm:spec_decode_num_drafts_total{{{labels}}} {generated / 3}",
        f"vllm:spec_decode_num_accepted_tokens_total{{{labels}}} {generated / 2}",
        f'vllm:request_success_total{{{labels},finished_reason="stop"}} 3.0',
        'vllm:cache_config_info{block_size="16",engine="0",gpu_memory_utilization="0.85",'
        'kv_cache_max_concurrency="4.5",kv_cache_memory_bytes="25769803776",kv_cache_size_tokens="1179648"} 1.0',
    ]) + "\n"


def runtime_document(tp=2, nodes=("spark0", "spark1")):
    """A sparkring-runtime-status/v1 document as runtime-status 0.3.6 reports it, trimmed."""
    ranks = [{"identity": {"rank": {"state": "known", "value": rank}},
              "effective": {"tp_collective_transport": {"state": "known", "value": "sircl"},
                            "tp_sircl_version": {"state": "known", "value": "0.3.2"}},
              "resources": {"node": node, "memory": {"total_bytes": 128e9, "available_bytes": 20e9},
                            "filesystems": [{"paths": ["/"], "total_bytes": 4e12, "available_bytes": 1e12}]}}
             for rank, node in enumerate(nodes)]
    return {"schema": "sparkring-runtime-status/v1", "startup_snapshot_at_unix_ns": 1_790_000_000 * 10**9,
            "effective": {"tensor_parallel_size": {"state": "known", "value": tp},
                          "max_num_seqs": {"state": "known", "value": 16}},
            "provenance": {"b12x_comm_bundle": {"name": {"state": "known", "value": "tp2-prepared"}}},
            "workers": {"expected_count": tp, "received_count": tp, "ranks": ranks}}


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeAPI:
    """Stands in for urllib's opener: answers each request from the per-endpoint `behaviour`,
    and records every request it is handed."""

    def __init__(self, clock):
        self.clock = clock
        self.requests = []
        self.behaviour = {}           # (host, port) -> "serving", "refused", "timeout", "starting"
        self.started = {}             # (host, port) -> process start time of its model run
        self.models = {}              # (host, port) -> served model name
        self.bodies = {}              # (host, port, path) -> fixed body

    def open(self, request, timeout=None):
        self.requests.append(request)
        parts = urllib.request.urlparse(request.full_url)
        key = (parts.hostname, parts.port)
        behaviour = self.behaviour.get(key, "serving")
        if behaviour == "refused":
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        if behaviour == "timeout":
            raise urllib.error.URLError(TimeoutError("timed out"))
        if behaviour == "starting":
            raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, None)
        if (key + (parts.path,)) in self.bodies:
            return Response(self.bodies[key + (parts.path,)])
        model = self.models.get(key, "M")
        if parts.path == "/metrics":
            now = self.clock()
            started = self.started.get(key, 1_790_000_000.0)
            return Response(metrics_text(model, generated=max(0.0, now - started) * 30, started=started).encode())
        if parts.path == "/v1/models":
            return Response(json.dumps({"data": [{"id": model, "max_model_len": 131072}]}).encode())
        if parts.path == "/v1/sparkring/status":
            return Response(json.dumps(runtime_document()).encode())
        raise AssertionError(f"unexpected request {request.get_method()} {request.full_url}")


class Clock:
    def __init__(self, now=1_790_000_100.0):
        self.now = now

    def __call__(self):
        return self.now


def collector(tmp_path, rows=PAIRS, *, quiet=None, recovery=None):
    clock = Clock()
    api = FakeAPI(clock)
    for _, host, port, model in rows:
        api.models[(host, port)] = model
    client = dashboard.ReadOnlyClient(opener=api)
    sources = [dashboard.Source(fabric_file(tmp_path), site_file(tmp_path, rows))]
    found = dashboard.Collector(sources, client, quiet_path=quiet or tmp_path / "quiet", node_status_path=None,
                                recovery_path=recovery, clock=clock)
    return found, api, clock


def run_for(found, clock, seconds, *, viewer=False, step=1.0):
    end = clock.now + seconds
    while clock.now < end:
        if viewer and int(clock.now) % 5 == 0:
            found.mark_viewed()
        found.cycle()
        clock.now += step


def paths_of(api):
    return [urllib.request.urlparse(r.full_url).path for r in api.requests]


# -- the request guard ---------------------------------------------------------------------
@pytest.mark.parametrize("path", [*INFERENCE_PATHS, "/v1/models?limit=1", "/metrics/", "/v1/sparkring/status/view",
                                  "/v1/sparkring/status.txt", "/../v1/completions", "", "/health"])
def test_only_the_three_read_paths_are_admitted(path):
    with pytest.raises(dashboard.RequestRefused):
        dashboard.request_url("192.0.2.12", 8000, path, {"192.0.2.12"})


def test_requests_go_only_to_addresses_the_host_files_name():
    assert dashboard.request_url("192.0.2.12", 8000, "/metrics", {"192.0.2.12"}) == "http://192.0.2.12:8000/metrics"
    assert dashboard.request_url("2001:db8::5", 8000, "/v1/models", {"2001:db8::5"}) == "http://[2001:db8::5]:8000/v1/models"
    for host, port in (("192.0.2.99", 8000), ("192.0.2.12", 0), ("192.0.2.12", "8000"), ("192.0.2.12", 70000)):
        with pytest.raises(dashboard.RequestRefused):
            dashboard.request_url(host, port, "/metrics", {"192.0.2.12"})


def test_the_client_sends_bodyless_get_requests_and_counts_them():
    api = FakeAPI(Clock())
    client = dashboard.ReadOnlyClient(opener=api)
    client.allow({"192.0.2.12"})
    assert client.get("192.0.2.12", 8015, "/v1/models").ok
    with pytest.raises(dashboard.RequestRefused):
        client.get("192.0.2.12", 8015, "/v1/chat/completions")
    [request] = api.requests
    assert (request.get_method(), request.data, request.full_url) == ("GET", None, "http://192.0.2.12:8015/v1/models")
    assert client.counters()["requests"] == {"/v1/models": 1}


def test_collection_sends_only_get_requests_to_read_paths_and_starts_no_process(tmp_path, monkeypatch):
    """Two simulated hours: a viewer for half an hour, a model restart, a stopped model, a model
    that starts, an unreachable Spark and on-demand settings reads."""
    def no_process(*args, **kwargs):
        raise AssertionError(f"the dashboard started a process: {args!r}")
    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(os, "system", no_process)
    found, api, clock = collector(tmp_path)
    run_for(found, clock, 1800, viewer=True)
    api.started[("192.0.2.12", 8015)] = clock.now           # the model on 2-3 restarts
    api.behaviour[("192.0.2.14", 8020)] = "refused"          # the model on 4-5 stops
    run_for(found, clock, 600)
    api.behaviour[("192.0.2.14", 8020)] = "starting"
    api.behaviour[("192.0.2.16", 8000)] = "timeout"          # Spark 6 is unreachable
    run_for(found, clock, 600)
    api.behaviour.clear()
    for model_id in list(found.models):
        found.request_runtime(model_id)
    run_for(found, clock, 4200)
    assert api.requests
    for request in api.requests:
        path = urllib.request.urlparse(request.full_url).path
        assert request.get_method() == "GET" and request.data is None
        assert path in dashboard.READ_PATHS
        assert not any(path.startswith(p) for p in INFERENCE_PATHS)
    assert found.snapshot()["collector"]["requests"] == {p: paths_of(api).count(p) for p in set(paths_of(api))}


def test_metrics_are_read_every_five_seconds_while_viewed_and_every_minute_otherwise(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    run_for(found, clock, 600, viewer=True)
    viewed = paths_of(api).count("/metrics")
    assert 118 <= viewed <= 122                       # 600 s / 5 s
    api.requests.clear()
    clock.now += dashboard.VIEWER_SECONDS             # the browser went away
    run_for(found, clock, 600)
    assert 9 <= paths_of(api).count("/metrics") <= 11  # 600 s / 60 s
    assert paths_of(api).count("/v1/models") <= 11
    api.requests.clear()
    found.mark_viewed()                               # a viewer comes back: the next cycle reads at once
    found.cycle()
    assert paths_of(api)[0] == "/metrics"


def test_runtime_status_is_read_once_per_model_run_and_on_a_rate_limited_request(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[1:2])
    run_for(found, clock, 1200, viewer=True)
    assert paths_of(api).count("/v1/sparkring/status") == 1
    api.started[("192.0.2.12", 8015)] = clock.now      # a new model run: its settings are read again
    run_for(found, clock, 60, viewer=True)
    assert paths_of(api).count("/v1/sparkring/status") == 2
    [model_id] = found.models
    assert found.request_runtime(model_id)
    assert not found.request_runtime(model_id)         # within RUNTIME_REFRESH_SECONDS
    run_for(found, clock, 10, viewer=True)
    assert paths_of(api).count("/v1/sparkring/status") == 3
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["transport"]["label"] == "SIRCL" and member["transport"]["sircl_version"] == "0.3.2"
    assert member["settings"]["max_num_seqs"] == 16 and member["capacity"]["max_num_seqs"] == 16


def test_the_quiet_file_stops_every_request(tmp_path):
    quiet = tmp_path / "dashboard.quiet"
    quiet.write_text("measuring\n")
    found, api, clock = collector(tmp_path, quiet=quiet)
    run_for(found, clock, 300, viewer=True)
    assert api.requests == []
    assert found.snapshot()["collector"]["quiet"] is True
    quiet.unlink()
    run_for(found, clock, 10, viewer=True)
    assert "/metrics" in paths_of(api)
    api.requests.clear()
    quiet.write_text("measuring\n")                  # takes effect on the next cycle
    run_for(found, clock, 120, viewer=True)
    assert api.requests == []


def test_states_follow_the_api_and_a_silent_api_is_asked_less_often(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    key = (NODE_A_LAN, 8000)
    states = []
    for behaviour in ("serving", "refused", "starting", "timeout", "serving"):
        api.behaviour[key] = behaviour
        api.requests.clear()
        run_for(found, clock, 60, viewer=True)
        states.append(next(iter(found.models.values())).state)
        if behaviour in ("refused", "timeout"):
            assert paths_of(api).count("/metrics") <= 60 / dashboard.FAILED_SECONDS + 1
            assert "/v1/sparkring/status" not in paths_of(api)
    assert states == ["serving", "stopped", "not_ready", "offline", "serving"]
    texts = [event["text"] for event in found.snapshot()["events"]][::-1]
    assert texts[0] == "Serving: GLM-5.3-Flash-NVFP4-Spark-TP2" and texts[1:4] == ["No model running", "Not ready", "Offline"]


def test_a_deployment_recorded_as_stopped_reads_as_stopped(tmp_path):
    clock = Clock()
    api = FakeAPI(clock)
    api.behaviour[("192.0.2.14", 8020)] = "timeout"
    source = dashboard.Source(fabric_file(tmp_path), site_file(tmp_path, PAIRS[2:3], stopped=True))
    found = dashboard.Collector([source], dashboard.ReadOnlyClient(opener=api), node_status_path=None,
                                recovery_path=None, quiet_path=tmp_path / "quiet", clock=clock)
    run_for(found, clock, 120, viewer=True)
    [member] = found.snapshot()["groups"][0]["members"]
    assert member["state"] == "stopped" and paths_of(api).count("/metrics") <= 3   # every minute while stopped


def test_a_deployment_whose_port_serves_another_model_is_idle(tmp_path):
    rows = (([0, 1], NODE_A_LAN, 8000, "Model-A"), ([0, 1, 2, 3], NODE_A_LAN, 8000, "Model-B"))
    found, api, clock = collector(tmp_path, rows=rows)
    api.models[(NODE_A_LAN, 8000)] = "Model-A"
    run_for(found, clock, 30, viewer=True)
    states = {m["model"]: m["state"] for m in found.snapshot()["groups"][0]["members"]}
    assert states == {"Model-A": "serving", "Model-B": "idle"}


# -- payloads --------------------------------------------------------------------------------
def test_metrics_of_a_vllm_server():
    """A /metrics response recorded from a TP2 server (vLLM with speculative decoding)."""
    metrics = dashboard.parse_metrics((FIXTURES / "vllm-metrics-tp2.prom").read_text(encoding="utf-8"))
    assert metrics["model_name"] == "Qwen3.8-Flash-Next-NVFP4-QAD-TP2"
    assert (metrics["running"], metrics["waiting"], metrics["kv"]) == (0.0, 0.0, 0.0)
    assert metrics["finished"]["stop"] == 7.0 and metrics["finished"]["length"] == 18.0
    assert metrics["started"] == pytest.approx(1790266951.99)
    assert metrics["drafts"] == 7386.0
    capacity = dashboard.capacity_from_cache_info(metrics["cache_info"])
    assert (capacity["kv_tokens"], capacity["kv_bytes"], capacity["block_size"]) == (2877721, 25769803776, 2848)
    assert capacity["concurrency"] == pytest.approx(10.9776, abs=1e-4)


def test_runtime_status_of_a_tp2_server():
    """A /v1/sparkring/status document recorded from a TP2 server, and one as 0.3.6 reports it."""
    recorded = dashboard.read_runtime_status(json.loads((FIXTURES / "runtime-status-tp2.json").read_text("utf-8")))
    assert (recorded["tp"], recorded["dp"], recorded["pp"]) == (2, 1, 1)
    assert recorded["settings"]["max_model_len"] == 262144 and recorded["settings"]["cache_dtype"] == "fp8"
    assert recorded["workers"] == {"expected": 2, "received": 2}
    current = dashboard.read_runtime_status(runtime_document())
    assert [(n["rank"], n["node"], n["memory_available"]) for n in current["nodes"]] == [
        (0, "spark0", 20e9), (1, "spark1", 20e9)]
    assert current["transport"]["bundle"] == "tp2-prepared" and current["started_at"] == 1_790_000_000


def test_label_escapes_and_short_model_names():
    assert dashboard.parse_labels('a="x",b="q\\"z",c="\\\\",d="l1\\nl2"') == {"a": "x", "b": 'q"z', "c": "\\",
                                                                                "d": "l1\nl2"}
    assert dashboard.short_model("GLM-5.3-Flash-NVFP4-Spark-TP2") == "GLM-5.3-Flash"
    assert dashboard.capacity_from_cache_info({"kv_cache_size_tokens": "None"}) is None


# -- host files --------------------------------------------------------------------------------
def test_the_fabric_document_that_setup_writes_is_read(tmp_path):
    fabric = dashboard.fabric_from_document(json.loads(fabric_file(tmp_path).read_text("utf-8")))
    assert (fabric["shape"], fabric["size"], fabric["closed"]) == ("cycle", 8, True)
    assert [p["hostname"] for p in fabric["positions"]] == [f"spark{i}" for i in range(8)]
    assert fabric["positions"][0]["lan_address"] == NODE_A_LAN and fabric["positions"][3]["address"] == "192.0.2.13"


def test_unusable_records_are_reported_and_poll_nothing(tmp_path):
    source = dashboard.Source(tmp_path / "missing.json", tmp_path / "site.json")
    source.refresh()
    assert "missing.json" in source.problem and source.deployments == []
    fabric = fabric_file(tmp_path)
    source = dashboard.Source(fabric, tmp_path / "no-site.json")
    source.refresh()
    assert "sparkring dashboard export" in source.problem
    found = dashboard.Collector([source], dashboard.ReadOnlyClient(opener=FakeAPI(Clock())), node_status_path=None,
                                recovery_path=None, quiet_path=tmp_path / "quiet")
    found.cycle()
    snapshot = found.snapshot()
    assert snapshot["topology"]["detail"] == source.problem and snapshot["rings"][0]["size"] == 8


def test_site_rows_that_do_not_fit_the_fabric_are_left_out():
    good = {"id": "a", "positions": [0, 1], "api_host": "192.0.2.10", "api_port": 8000,
            "containers": [{"position": 0, "name": "sr-a-r0"}, {"position": 1, "name": "x; reboot"}]}
    rows = [good, {**good, "positions": [7, 8]}, {**good, "positions": [1, 1]}, {**good, "api_host": "node-a"},
            {**good, "api_port": 0}, {**good, "api_port": "8000"}, "not a row"]
    [kept] = dashboard.deployments_from_site({"schema": dashboard.SITE_SCHEMA, "deployments": rows}, 8)
    assert kept["containers"] == [{"position": 0, "name": "sr-a-r0"}]
    with pytest.raises(ValueError):
        dashboard.deployments_from_site({"schema": "other"}, 8)


def test_recovery_findings_are_shown_without_a_check_of_its_own(tmp_path):
    recovery = tmp_path / "recovery.json"
    recovery.write_text(json.dumps({"schema": "sparkring-recovery/v1", "deployments": {
        "/var/lib/sparkring/controller/deployments/profile-on-0-1": {
            "enabled": True, "checked_at": 1_790_000_000.0,
            "finding": {"state": "serving", "summary": "The model serves", "next_action": None}}}}))
    found, api, clock = collector(tmp_path, rows=PAIRS[:1], recovery=recovery)
    run_for(found, clock, 10, viewer=True)
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["recovery"]["summary"] == "The model serves" and member["recovery"]["state"] == "serving"
    assert dashboard.recovery_findings(tmp_path / "absent.json") == {}


# -- the status document and the page -----------------------------------------------------------
# The member fields page.html reads (runtime/host/dashboard.html: card, tiles, settings, totals, ringSvg).
MEMBER_FIELDS = {"id", "name", "state", "state_label", "state_since", "model", "model_short", "max_model_len",
                 "endpoint", "details_url", "layout", "live", "totals", "settings", "capacity", "nodes", "workers",
                 "image", "transport", "sparks", "history", "arc", "ring", "head", "logs"}


def test_the_status_document_has_the_fields_the_page_reads(tmp_path):
    found, api, clock = collector(tmp_path)
    run_for(found, clock, 120, viewer=True)
    snapshot = json.loads(json.dumps(found.snapshot()))
    lab = json.loads((FIXTURES / "lab-status-sample.json").read_text(encoding="utf-8"))
    assert set(lab) - {"ring"} <= set(snapshot)
    assert set(lab["ring"]) <= set(snapshot["ring"])
    ssh_only = {"ssh", "ssh_user", "containers"}            # container scans over SSH, which this dashboard drops
    assert set(lab["ring"]["positions"][0]) - ssh_only <= set(snapshot["ring"]["positions"][0])
    assert set(lab["events"][0]) == set(snapshot["events"][0])
    members = snapshot["groups"][0]["members"]
    assert [m["name"] for m in members] == ["tp2-0-1", "tp2-2-3", "tp2-4-5", "tp2-6-7"]
    for member in members:
        assert MEMBER_FIELDS <= set(member)
        assert member["state"] == "serving" and member["live"]["output_tps"] == pytest.approx(30.0, rel=0.05)
        assert member["arc"]["source"] == "deployment" and not member["arc"]["inferred"]
        assert member["capacity"]["kv_tokens"] == 1179648
    assert snapshot["ring"]["positions"][2]["member"] == members[1]["id"]
    assert members[0]["logs"]["commands"][0] == "sudo docker logs --tail 200 sr-0-r0"
    assert members[0]["head"] == {"position": 0, "name": "spark0", "address": NODE_A_LAN, "port": 8000,
                                  "container": "sr-0-r0"}


@pytest.mark.skipif(not os.environ.get("SPARKRING_DASHBOARD_SAMPLES"),
                    reason="set SPARKRING_DASHBOARD_SAMPLES to a directory holding the lab dashboard's "
                           "api-status-sample.json to compare against the unsanitized sample")
def test_the_status_document_covers_an_unsanitized_lab_sample(tmp_path):
    lab = json.loads((Path(os.environ["SPARKRING_DASHBOARD_SAMPLES"]) / "api-status-sample.json").read_text("utf-8"))
    found, api, clock = collector(tmp_path)
    run_for(found, clock, 30, viewer=True)
    snapshot = found.snapshot()
    assert set(lab) - {"ring"} <= set(snapshot) and set(lab["ring"]) <= set(snapshot["ring"])
    assert set(lab["ring"]["positions"][0]) - {"ssh", "ssh_user", "containers"} <= set(snapshot["ring"]["positions"][0])


def test_the_page_loads_nothing_from_elsewhere_and_its_script_is_admitted_by_hash():
    page, csp = dashboard.page_bytes()
    text = page.decode("utf-8")
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp and "'unsafe-inline'" not in csp.replace(
        "style-src-attr 'unsafe-inline'", "")
    assert not any(marker in text for marker in ('src="http', "src='http", 'href="http', "@import", "<link"))
    assert set(__import__("re").findall(r"fetch\(\s*[`\"']([^`\"'?$]*)", text)) == {"/api/status", "/api/refresh"}
    assert "ssh" not in text.lower().replace("no ssh", "").replace("ssh key", "")


def test_log_viewing_is_a_command_for_the_operator():
    assert dashboard.log_commands("spark2", None) == {"enabled": False, "host": "spark2", "commands": []}
    assert dashboard.log_commands("spark2", "sr-2-r0")["commands"][0] == "sudo docker logs --tail 200 sr-2-r0"


# -- the web server ------------------------------------------------------------------------------
@pytest.fixture
def server(tmp_path):
    found, api, clock = collector(tmp_path)
    found.cycle()
    page, csp = dashboard.page_bytes()
    servers = []

    def start(password=None, names=frozenset()):
        httpd = dashboard.ThreadingHTTPServer(("127.0.0.1", 0),
                                              dashboard.make_handler(found, page, csp, names, password))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return httpd.server_address[1], found
    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def call(port, method, path, host="127.0.0.1", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, headers={"Host": host, **(headers or {})})
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response, body


def test_the_server_answers_reads_only_and_only_to_its_own_address(server):
    port, found = server(names=frozenset({"dashboard.example"}))
    response, body = call(port, "GET", "/")
    assert response.status == 200 and response.getheader("Content-Security-Policy").startswith("default-src 'none'")
    assert response.getheader("X-Frame-Options") == "DENY" and b"<title>SparkRing</title>" in body
    response, body = call(port, "GET", "/api/status", host="dashboard.example:8090")
    assert response.status == 200 and json.loads(body)["schema"] == dashboard.SCHEMA and found.last_view is not None
    assert call(port, "GET", "/api/status", host="attacker.example")[0].status == 421
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        assert call(port, method, "/api/status")[0].status == 405
    assert call(port, "GET", "/api/logs")[0].status == 404


def test_a_password_file_requires_basic_authentication(server):
    port, _ = server(password="correct horse")
    response, _ = call(port, "GET", "/api/status")
    assert response.status == 401 and "Basic" in response.getheader("WWW-Authenticate")
    token = base64.b64encode(b"sparkring:correct horse").decode()
    assert call(port, "GET", "/api/status", headers={"Authorization": "Basic " + token})[0].status == 200


def test_settings_reads_are_requested_through_a_rate_limited_route(server):
    port, found = server()
    model_id = next(iter(found.models))
    assert call(port, "GET", "/api/refresh?model=" + urllib.request.quote(model_id))[0].status == 202
    assert call(port, "GET", "/api/refresh?model=" + urllib.request.quote(model_id))[0].status == 429


# -- placement, the module's own imports, packaging -----------------------------------------------
def test_the_efficiency_cores_of_a_gb10(tmp_path):
    """CPUs 0-4 and 10-14 are Cortex-A725 and 5-9 and 15-19 Cortex-X925, as SIRCL's CPU tests lay
    out a GB10; one X925 reports a higher capacity than its siblings."""
    root = tmp_path / "cpu"
    root.mkdir()
    (root / "online").write_text("0-19\n")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("")                                 # no part numbers: the capacities decide
    performance = {*range(5, 10), *range(15, 20)}
    for cpu in range(20):
        (root / f"cpu{cpu}").mkdir()
        capacity = (1024 if cpu == 19 else 1017) if cpu in performance else 580
        (root / f"cpu{cpu}" / "cpu_capacity").write_text(f"{capacity}\n")
    assert dashboard.efficiency_cpus(root, cpuinfo) == {*range(0, 5), *range(10, 15)}
    for cpu in range(20):
        (root / f"cpu{cpu}" / "cpu_capacity").write_text("1024\n")
    assert dashboard.efficiency_cpus(root, cpuinfo) is None


def test_the_module_can_start_no_process_and_open_no_ssh_session():
    tree = ast.parse(Path(dashboard.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"subprocess", "pty", "paramiko", "asyncssh", "fabric", "pexpect", "multiprocessing"}
    calls = {f"{node.func.value.id}.{node.func.attr}" for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)}
    assert not {c for c in calls if c.startswith("os.") and c.split(".")[1].startswith(
        ("system", "popen", "exec", "spawn", "fork", "posix_spawn"))}
    assert dashboard.READ_PATHS == {"/metrics", "/v1/models", "/v1/sparkring/status"}


def test_the_command_is_registered():
    source = (ROOT / "scripts" / "sparkring.py").read_text(encoding="utf-8")
    assert 'subcommands.add_parser("dashboard"' in source
    assert 'raw[0] == "dashboard"' in source and "from runtime.host.dashboard import main" in source
    with pytest.raises(SystemExit) as stopped:
        dashboard.main(["--help"])
    assert stopped.value.code == 0


def test_the_package_launcher_runs_the_dashboard_without_the_operator_cli():
    launcher = (ROOT / "packaging" / "debian" / "sparkring").read_text(encoding="utf-8")
    program = ("import sys\n"
               "sys.argv = ['sparkring', 'dashboard', '--help']\n"
               "try:\n"
               f"    exec(compile({launcher.replace('/usr/lib/sparkring', ROOT.as_posix())!r}, 'sparkring', 'exec'))\n"
               "except SystemExit:\n"
               "    pass\n"
               "print('operator-cli' if 'scripts.sparkring' in sys.modules else 'dashboard-only')\n")
    result = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=120)
    assert result.stdout.strip().splitlines()[-1] == "dashboard-only", result.stderr


def test_the_service_runs_unprivileged_at_idle_priority():
    unit = (ROOT / "packaging" / "debian" / "sparkring-dashboard.service").read_text(encoding="utf-8")
    for line in ("ExecStart=/usr/bin/sparkring dashboard serve", "DynamicUser=yes", "Nice=19",
                 "CPUSchedulingPolicy=idle", "IOSchedulingClass=idle", "MemoryMax=", "NoNewPrivileges=yes",
                 "CapabilityBoundingSet=\n", "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX"):
        assert line in unit
    assert not [line for line in unit.splitlines() if "ssh" in line.lower() and not line.startswith("#")]
    assert "sparkring-dashboard.service" in (ROOT / "packaging" / "debian" / "prerm").read_text(encoding="utf-8")


# -- quiet holders -------------------------------------------------------------------------------------
def test_overlapping_measurements_hold_quiet_mode_until_the_last_one_releases_it(tmp_path):
    quiet = tmp_path / "run" / "dashboard.quiet"
    now = 1_790_000_000.0
    first = dashboard.quiet_hold(quiet, "serving_ab", now=now)
    assert first["was_quiet"] is False and dashboard.quiet_state(quiet, now)["quiet"]
    second = dashboard.quiet_hold(quiet, "accept_profile", now=now + 1)
    assert second["was_quiet"] is True
    dashboard.quiet_release(quiet, first["holder"])
    state = dashboard.quiet_state(quiet, now + 2)
    assert state["quiet"] and [h["label"] for h in state["holders"]] == ["accept_profile"]
    dashboard.quiet_release(quiet, second["holder"])
    assert dashboard.quiet_state(quiet, now + 3) == {"quiet": False, "operator": False, "holders": []}


def test_an_operators_pause_outlasts_every_measurement_and_holders_lapse(tmp_path):
    quiet = tmp_path / "run" / "dashboard.quiet"
    quiet.parent.mkdir(parents=True)
    quiet.write_text("paused\n")
    now = 1_790_000_000.0
    held = dashboard.quiet_hold(quiet, "measure_api", ttl=60, now=now)
    assert held["was_quiet"] is True
    dashboard.quiet_release(quiet, held["holder"])
    assert dashboard.quiet_state(quiet, now)["quiet"] is True          # the operator's pause stays
    quiet.unlink()
    dashboard.quiet_hold(quiet, "measure_api", ttl=60, now=now)
    assert dashboard.quiet_state(quiet, now + 59)["quiet"] is True
    assert dashboard.quiet_state(quiet, now + 61)["quiet"] is False    # its time to live ended
    # A holder whose process exited (a harness killed outright) lapses at once.
    dashboard.quiet_hold(quiet, "killed", pid=4242, now=now + 61)
    assert dashboard.quiet_state(quiet, now + 62, alive=lambda pid: pid != 4242)["quiet"] is False
    assert dashboard.quiet_state(quiet, now + 62, alive=lambda pid: True)["quiet"] is True


def test_a_holder_stops_the_collector_on_its_next_cycle(tmp_path):
    quiet = tmp_path / "quiet"
    found, api, clock = collector(tmp_path, quiet=quiet)
    run_for(found, clock, 10, viewer=True)
    api.requests.clear()
    held = dashboard.quiet_hold(quiet, "serving_ab", now=clock.now)
    run_for(found, clock, 60, viewer=True)
    assert api.requests == []
    dashboard.quiet_release(quiet, held["holder"])
    run_for(found, clock, 10, viewer=True)
    assert "/metrics" in paths_of(api)


@pytest.mark.parametrize("label", ["", "../etc", "a" * 41, "x y"])
def test_holder_labels_are_plain_names(tmp_path, label):
    with pytest.raises(ValueError):
        dashboard.quiet_hold(tmp_path / "quiet", label)


def test_the_quiet_command_holds_and_releases(tmp_path, capsys):
    quiet = tmp_path / "dashboard.quiet"
    assert dashboard.main(["quiet", "hold", "--label", "serving_ab", "--quiet-file", str(quiet)]) == 0
    holder = json.loads(capsys.readouterr().out)["holder"]
    assert dashboard.main(["quiet", "status", "--quiet-file", str(quiet)]) == 0
    assert json.loads(capsys.readouterr().out)["quiet"] is True
    assert dashboard.main(["quiet", "release", holder, "--quiet-file", str(quiet)]) == 0
    assert json.loads(capsys.readouterr().out) == {"released": holder, "still_quiet": False}
    assert dashboard.main(["quiet", "hold", "--quiet-file", str(quiet)]) == 2
