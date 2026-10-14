"""Offline startup parsers, agent lifecycle, collector transitions and browser rendering."""
import json
import os
from pathlib import Path

import pytest

from runtime.host import dashboard, progress, startup, telemetry_live
from runtime.host.test_dashboard import PAIRS, collector
from runtime.host.test_dashboard_charts import smooth_page, update_page  # noqa: F401
from runtime.host.test_dashboard_layout import GEOMETRY

FIXTURES = Path(__file__).with_name("fixtures") / "startup"


def observation(phase="loading", at=1000, rank=0, **extra):
    return dict(schema=startup.SCHEMA, phase=phase, label=dashboard.LABELS.get(phase, "Starting"),
                at=at, started_at=at - 100, rank=rank, run_id="container:run", **extra)


@pytest.mark.parametrize("name,phase", [("vllm-weights", "loading"), ("b12x", "tuning"),
                                        ("vllm-graphs", "graphs"), ("installer", "starting")])
def test_recorded_excerpts(name, phase):
    value = startup.parse((FIXTURES / (name + ".log")).read_text(encoding="utf-8"), installer=name == "installer")
    assert value["phase"] == phase
    if name == "vllm-weights":
        assert (value["done"], value["total"]) == (3, 41)
    if name == "vllm-graphs":
        assert 0 <= value["done"] <= value["total"]


def test_known_patterns_only_carriage_returns_and_bounded_input():
    weights = startup.parse("\x1b[32mLoading safetensors checkpoint shards: 1/43\rLoading safetensors checkpoint shards: 12/43")
    assert weights == dict(phase="loading", label="Loading weights", done=12, total=43)
    for unknown in ("torch.compile disabled", "Skipping FlashInfer autotune because it is disabled.",
                    "application ready", "loading some unrelated file"):
        assert startup.parse(unknown, weights) == weights
    assert startup.parse("Loading safetensors checkpoint shards: 99/43")["phase"] == "loading"
    assert "done" not in startup.parse("Loading safetensors checkpoint shards: 99/43")
    assert startup.parse("Compiling a graph for dynamic shape") == {"phase": "compiling", "label": "Compiling"}
    assert startup.parse("Loading safetensors checkpoint shards: 2/43" + "x" * startup.LIMIT) == {}


@pytest.mark.parametrize("line,label", [
    ("Node 0: Prepare checkpoint and verify all shards", "Checkpoint check"),
    ("Node 0: Download missing checkpoint files from huggingface.co", "Downloading checkpoint"),
    ("Node 1: Copy checkpoint files from Node 0 over the fabric", "Copying checkpoint"),
    ("Node 1: Load the image from its verified layers", "Loading image"),
    ("Node 0: Derive the checkpoint in the installer image", "Deriving checkpoint"),
    ("Node 0: Wait for API readiness", "Starting"),
])
def test_installer_phases(line, label):
    assert startup.parse(line, installer=True)["label"] == label


def test_reader_health_errors_restart_and_bounded_requests():
    requests = []
    identity = "a" * 64
    state = dict(Status="running", StartedAt="2026-10-14T10:00:00Z", Health={"Status": "starting"})
    log = [b"Loading safetensors checkpoint shards: 12/43\r"]
    def get(path):
        requests.append(path)
        if path.startswith("/containers/json?"):
            return json.dumps([{"Id": identity, "Labels": {"io.sparkring.deployment": "one", "io.sparkring.rank": "0"}}]).encode()
        if path.endswith("/json"):
            return json.dumps({"State": state}).encode()
        assert "tail=200" in path and "since=2026-10-14T10%3A00" in path
        return b"\x01\0\0\0" + len(log[0]).to_bytes(4, "big") + log[0]
    reader = startup.Reader(get)
    at = startup.stamp("2026-10-14T10:10:00Z")
    assert reader.read(at)["one"][0]["done"] == 12
    log[0] = b"unrelated diagnostic"
    assert reader.read(at)["one"][0]["done"] == 12
    state["StartedAt"] = "2026-10-14T10:00:01Z"
    assert reader.read(at)["one"][0]["phase"] == "starting"
    log[0] = b"ERROR first\nRuntimeError: <bad> & last\n"
    state["Status"] = "exited"
    value = reader.read(at)["one"][0]
    assert value["phase"] == "failed" and value["error"] == "RuntimeError: <bad> & last"
    state.update(Status="running", Health={"Status": "healthy"})
    requests.clear()
    assert reader.read(at)["one"][0]["phase"] == "ready"
    assert not any("/logs?" in request for request in requests)
    state["Health"] = {"Status": "unhealthy", "Log": [{"Output": "probe\nconnection refused\n"}]}
    log[0] = b""
    assert reader.read(at)["one"][0]["error"] == "connection refused"


def test_projection_rejects_stale_bad_ranks_and_unrequested_deployments():
    good = observation()
    rows = [good, {**good, "at": 900}, {**good, "rank": -1}, {**good, "phase": "invented"},
            {**good, "at": float("nan")}, {**good, "started_at": float("inf")}]
    assert startup.project({"one": rows, "other": [good]}, ["one"], 1000) == {"one": [good]}
    assert startup.project({"one": rows}, ["one"], 1010) == {}


def test_forwarder_merges_rank_detail_and_idle_does_no_reads(tmp_path):
    reads, closed = [], []
    class Channel:
        def __init__(self, host):
            pass
        def sample(self):
            return {"__startup__": {"one": [observation("graphs", rank=1)]}}
        def close(self):
            closed.append(True)
    def local():
        reads.append(True)
        return {"__startup__": {"one": [observation(done=12, total=43)]}}
    forwarder = telemetry_live.Forwarder(channel=Channel, read=local, output=tmp_path / "live")
    forwarder.cycle({"local": ["one"], "worker": ["one"]}, "local", 1000)
    value = json.loads(forwarder.output.read_text())
    combined = startup.group(value["startup"]["one"], [4, 5])
    assert combined["phase"] == "loading"
    assert [r["position"] for r in combined["ranks"]] == [4, 5]
    forwarder.cycle({}, "local", 1001)
    forwarder.cycle({}, "local", 1002)
    assert reads == [True] and closed == [True] and not forwarder.output.exists()


def test_installer_markers_complete_and_dead_processes_are_ignored(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    from runtime.host import dashboard_admin
    monkeypatch.setattr(dashboard_admin, "deployment_row", lambda *a: dict(deployment_id="one", positions=[4, 5]))
    with progress.run("install"):
        progress.bind_slice([4, 5])
        progress.bind_deployment(tmp_path, [4, 5])
        progress.say("Node 0: Download missing checkpoint files from huggingface.co")
        progress.say("Node 1: Copy checkpoint files from Node 0 over the fabric")
        values = startup.operations(tmp_path, alive=lambda pid: True)
        assert len(values) == 1 and values[0]["phase"] == "installing"
        assert [r["position"] for r in values[0]["ranks"]] == [4, 5]
        assert startup.operations(tmp_path, alive=lambda pid: False) == []
    assert startup.operations(tmp_path, alive=lambda pid: True) == []
    capsys.readouterr()


def test_collector_transitions_elapsed_ready_failed_and_expiry(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    found.live_path = tmp_path / "live"
    site = found.sources[0].site_path
    doc = json.loads(site.read_text())
    doc["deployments"][0]["deployment_id"] = "one"
    site.write_text(json.dumps(doc))
    host, port = PAIRS[0][1:3]
    api.behaviour[(host, port)] = "refused"
    start = clock.now - 100
    for phase in ("installing", "starting", "loading", "graphs", "failed"):
        sample = observation(phase, clock.now)
        sample["started_at"] = start
        found.live_path.write_text(json.dumps({"at": clock.now, "startup": {"one": [sample]}}))
        found.mark_viewed()
        found.cycle()
        member = found.snapshot()["groups"][0]["members"][0]
        assert member["state"] == phase and member["state_since"] == start
        clock.now += 2
    api.behaviour.pop((host, port))
    model = next(iter(found.models.values()))
    model.next_metrics = 0
    found.mark_viewed()
    found.cycle()
    # A frontend can still answer metrics after a worker has failed.
    assert found.member(model, clock.now, 0)["state"] == "failed"
    sample = observation("ready", clock.now)
    found.live_path.write_text(json.dumps({"at": clock.now, "startup": {"one": [sample]}}))
    found.cycle()
    member = found.member(model, clock.now, 0)
    assert member["state"] == "serving" and member["startup"] is None
    api.behaviour[(host, port)] = "refused"
    clock.now += 20
    found.mark_viewed()
    found.cycle()
    assert found.member(model, clock.now, 0)["state"] == "stopped"


def test_pending_install_is_visible_before_site_export(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    found.live_path = tmp_path / "live"
    row = json.loads(found.sources[0].site_path.read_text())["deployments"][0]
    row.update(id="pending", deployment_id="new", stopped=False)
    op = dict(phase="installing", label="Loading image", started_at=clock.now - 90, at=clock.now, deployment=row)
    found.live_path.write_text(json.dumps({"at": clock.now, "operations": [op]}))
    found.mark_viewed()
    found.cycle()
    members = found.snapshot()["groups"][0]["members"]
    assert len(members) == 1 and members[0]["state"] == "installing"
    assert members[0]["startup"]["label"] == "Loading image"


@pytest.mark.parametrize("width", [1400, 375])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_startup_browser_progress_escape_transitions(smooth_page, width, theme):  # noqa: F811
    page, value, requests = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    members = value["groups"][0]["members"]
    now = value["generated_at"]
    value["rings"][0]["name"] = "SparkRing"
    value["rings"][0]["discovery"]["at"] = now
    value["events"] = []
    for i, phase in enumerate(("loading", "graphs", "installing", "failed")):
        p = observation(phase, now, done=12, total=43) if i < 2 else observation(phase, now)
        p["label"] = "Loading image" if i == 2 else dashboard.LABELS[phase]
        p["started_at"] = now - 183
        if i == 0:
            p["ranks"] = [dict(observation("graphs", now, rank=1), position=1)]
        if i == 3:
            p["error"] = 'RuntimeError: <img src=x onerror="window.injected=1"> & failed'
        members[i].update(state=phase, state_label=dashboard.LABELS[phase], state_since=now - 183, startup=p, live={})
    update_page(page)
    cards = page.locator("article.card")
    assert cards.first.locator(".pill").inner_text() == "Loading weights 12/43"
    assert "Spark 1" in cards.first.inner_text()
    assert "for 3 min" in cards.first.inner_text()
    assert page.locator(".ringfig .lstartup").count() == 4
    assert "Weights 12/43" in page.locator(".ringfig").inner_text()
    assert cards.locator(".start-command").count() == 0
    assert cards.locator("img").count() == 0 and page.evaluate("window.injected") is None
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.evaluate(GEOMETRY) == []
    output = os.environ.get("SPARKRING_LOADING_SCREENSHOTS")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        members[3]["startup"]["error"] = "RuntimeError: CUDA out of memory"
        update_page(page)
        page.screenshot(path=str(Path(output) / f"loading-{theme}-{width}.png"), full_page=True)
    members[0].update(state="serving", state_label="Serving", startup=None, live={"output_tps": 42})
    update_page(page)
    assert cards.first.locator(".pill").inner_text() == "Serving"
    assert cards.first.locator(".startup-detail").count() == 0


def test_installer_failure_persists_until_next_operation(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    from runtime.host import dashboard_admin
    monkeypatch.setattr(dashboard_admin, "deployment_row", lambda *a: dict(deployment_id="one", positions=[4, 5]))
    with progress.run("install"):
        progress.bind_deployment(tmp_path, [4, 5])
        progress.failure("Download failed: missing shard")
    values = startup.operations(tmp_path, alive=lambda pid: False)
    assert values[0]["phase"] == "failed" and "missing shard" in values[0]["error"]
    with progress.run("down"):
        progress.bind_slice([4, 5])
    assert startup.operations(tmp_path, alive=lambda pid: True) == []
    capsys.readouterr()


def test_installer_transfer_progress_and_rank_restart_selection():
    value = startup.parse("Still working: Node 1: Copy checkpoint files from Node 0 (30s) - 121 of 166 GiB", installer=True)
    assert (value["done"], value["total"], value["unit"]) == (73, 100, "%")
    older, newer = observation("failed"), observation("loading")
    older["started_at"] = 800
    values = startup.project({"one": [older, newer]}, ["one"], 1000)
    assert values["one"] == [newer]


def test_new_pending_install_adds_only_configured_channels(tmp_path):
    controller = tmp_path / "controller"
    controller.mkdir()
    (tmp_path / "dashboard").mkdir()
    hosts = [{"host": f"root@192.0.2.{n+10}"} for n in range(8)]
    (controller / "cluster.json").write_text(json.dumps({"plan": {"spec": {"hosts": hosts}}}))
    (tmp_path / "dashboard/site.json").write_text(json.dumps({"deployments": []}))
    wanted, local = telemetry_live.heads(controller, pending=[{"positions": [4, 5], "deployment_id": "new"}])
    assert wanted == {hosts[n]["host"]: ["new"] for n in (4, 5)}
    assert local == hosts[0]["host"]


def test_worker_progress_cannot_replace_head_counters(tmp_path):
    from runtime.host.test_telemetry_live import snapshot
    head = snapshot()
    worker = {**head, "gen": 999}
    class Channel:
        def __init__(self, host):
            pass
        def sample(self):
            return {"one": worker, "__startup__": {"one": [observation(rank=1)]}}
        def close(self):
            pass
    forwarder = telemetry_live.Forwarder(channel=Channel, read=lambda: {"one": head}, output=tmp_path / "live")
    forwarder.cycle({"head": ["one"], "worker": ["one"]}, "head", 1000, metrics={"one": "head"})
    value = json.loads(forwarder.output.read_text())
    assert value["telemetry"]["one"]["gen"] == head["gen"]
    assert value["startup"]["one"][0]["rank"] == 1


def test_created_container_uses_creation_time_before_first_start():
    def get(path):
        if path.startswith("/containers/json?"):
            return json.dumps([{"Id": "a" * 64, "Labels": {"io.sparkring.deployment": "one", "io.sparkring.rank": "0"}}]).encode()
        if path.endswith("/json"):
            return json.dumps({"Created": "2026-10-14T10:00:00Z", "State": {"Status": "created", "StartedAt": "0001-01-01T00:00:00Z"}}).encode()
        return b""
    value = startup.Reader(get).read(startup.stamp("2026-10-14T10:01:00Z"))["one"][0]
    assert value["phase"] == "starting" and value["started_at"] == startup.stamp("2026-10-14T10:00:00Z")


def test_docker_response_byte_cap_and_socket_deadline(monkeypatch):
    reads, requests, closed = [], [], []
    class Socket:
        def settimeout(self, seconds):
            assert seconds == .2
        def connect(self, path):
            assert path.endswith("docker.sock")
        def close(self):
            closed.append("socket")
    class Response:
        status = 200
        def read1(self, count):
            reads.append(count)
            return b"x" * count
    class Connection:
        def __init__(self, host, timeout):
            assert host == "localhost" and timeout == .2
        def request(self, method, path):
            requests.append((method, path))
        def getresponse(self):
            return Response()
        def close(self):
            closed.append("http")
    monkeypatch.setattr(startup.socket, "socket", lambda *a: Socket())
    monkeypatch.setattr(startup.socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(startup.http.client, "HTTPConnection", Connection)
    assert startup.docker_get("/containers/json", limit=12) == b""
    assert sum(reads) == 13 and requests == [("GET", "/containers/json")]
    assert closed == ["http", "socket"]
    reads.clear()
    times = iter([0, 0, .5])
    monkeypatch.setattr(startup.time, "monotonic", lambda: next(times))
    assert startup.docker_get("/containers/json") == b"" and len(reads) == 1


def test_missing_ranks_are_unobserved_not_ready():
    group = startup.group([observation()], [4, 5])
    assert group["phase"] == "loading" and group["unobserved"] == [5]
