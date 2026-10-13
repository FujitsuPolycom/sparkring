"""Agent launch facts and dashboard discovery without Docker, SSH or a Spark."""
import json
from urllib.parse import urlparse

import pytest

from runtime.host import dashboard, dashboard_tui, node_containers as nc
from runtime.host.test_dashboard import collector, run_for


def argv(engine="vllm", size=2, rank=0, **sizes):
    command = (["vllm", "serve", "Model"] if engine == "vllm" else
               ["python", "-m", "sglang.launch_server", "--model-path", "Model"])
    command += ["--nnodes", str(size), "--node-rank", str(rank), "--port", "9123", "--served-model-name", "Served"]
    command += (["--master-addr", "192.0.2.10", "--master-port", "29501"] if engine == "vllm" else
                ["--dist-init-addr", "192.0.2.10:29501"])
    for key, value in {"tp": size, **sizes}.items():
        command += [nc.INTEGER_FLAGS[key][0], str(value)]
    if engine == "vllm" and rank:
        command += ["--headless"]
    return command


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("size,sizes", [(2, {}), (4, {}), (6, {}), (11, {}),
                                      (8, {"tp": 4, "pp": 2}), (2, {"tp": 1, "dp": 2}), (2, {"dcp": 2})])
def test_parallel_launch_facts(engine, size, sizes):
    for rank in range(size):
        facts = nc.launch_facts(argv(engine, size, rank, **sizes))
        assert facts["engine"] == engine
        assert (facts["nnodes"], facts["node_rank"]) == (size, rank)
        assert (facts["master_addr"], facts["master_port"]) == ("192.0.2.10", 29501)
        assert facts["served_model"] == "Served"
        assert facts["tp"] == sizes.get("tp", size)
        for key, val in sizes.items():
            assert facts[key] == val
        assert ("api_port" in facts) == (engine == "sglang" or rank == 0)


def test_bounds_shell_equals_aliases_and_no_environment_values(tmp_path):
    assert nc.launch_facts(["vllm", "serve", "x", "x" * nc.MAX_ARGV_BYTES]) == {"engine": "unknown"}
    assert nc.launch_facts(["vllm"] * (nc.MAX_ARGS + 1)) == {"engine": "unknown"}
    parsed = nc.launch_facts(["sh", "-c", "exec vllm serve Model --port=9123 -tp 4 -pp 2 -dcp 2"])
    assert (parsed["api_port"], parsed["tp"], parsed["pp"], parsed["dcp"]) == (9123, 4, 2, 2)
    rows = [{"id": f"{i:064x}", "name": f"manual-{i}", "state": "running"} for i in range(50)]
    calls = []
    def inspect(cid, **kwargs):
        calls.append(cid)
        return {"Path": "vllm", "Args": argv()[1:] + ["--api-key", "PRIVATE"], "Image": "sha256:" + "a" * 64,
                "Config": {"Image": "model:tag", "Env": ["TOKEN=PRIVATE"], "Labels": {"private": "PRIVATE"}}}
    observed = nc.observations({"docker": {"containers": rows}}, root=tmp_path, inspect=inspect)
    assert len(observed) == len(calls) == nc.MAX_CONTAINERS
    assert "PRIVATE" not in json.dumps(observed) and "Env" not in json.dumps(observed)
    assert observed[0]["image_id"] == "a" * 12


def reports(found, clock, positions=(0, 2), size=None, engine="vllm"):
    found.reload_sources()
    fabric = found.sources[0].fabric
    result = []
    for rank, position in enumerate(positions):
        facts = nc.launch_facts(argv(engine, size or len(positions), rank))
        result.append({"node_id": fabric["positions"][position]["node_id"], "observed_at": clock.now,
                       "model_containers": [{**facts, "managed": False, "name": f"unrelated-{position}",
                                             "image": "model:tag", "image_id": "abcdef123456"}]})
    return result


def install_reports(found, documents, tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps({"schema": "sparkring-dashboard-nodes/v1", "nodes": documents}), encoding="utf-8")
    found.node_reports_path = path
    found.next_files = 0


@pytest.mark.parametrize("positions,size", [([0], 1), ([0, 2], 2), ([0, 2, 3, 4, 6, 7], 6), ([0, 2], 4)])
def test_grouping_head_endpoint_incomplete_and_single(tmp_path, positions, size):
    found, api, clock = collector(tmp_path, rows=())
    documents = reports(found, clock, positions, size)
    install_reports(found, documents, tmp_path)
    found.cycle()
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["arc"]["positions"] == positions
    assert bool(member["group_note"]) == (len(positions) < size)
    if len(positions) < size:
        assert member["group_note"] == f"incomplete: {len(positions)} of {size} Sparks reporting"
    assert member["unmanaged"] and member["state"] == "serving"
    assert {urlparse(r.full_url).port for r in api.requests} == {9123}
    assert {urlparse(r.full_url).hostname for r in api.requests} == {found.sources[0].fabric["positions"][0]["address"]}
    assert set(urlparse(r.full_url).path for r in api.requests) == dashboard.READ_PATHS
    assert len(found.history[member["id"]]) == 1


def test_master_keys_split_groups_even_with_same_names_and_adjacent_positions(tmp_path):
    found, _, clock = collector(tmp_path, rows=())
    docs = reports(found, clock, (0, 1, 2, 3), size=2)
    for rank, doc in enumerate(docs):
        c = doc["model_containers"][0]
        c.update(name="identical", node_rank=rank % 2)
        if rank >= 2:
            c["master_addr"] = "192.0.2.12"
    grouped = dashboard.unmanaged_groups(found.sources[0].fabric, docs, [], clock.now)
    assert [g["positions"] for g in grouped] == [[0, 1], [2, 3]]


def test_stopped_record_is_retained_in_live_card_details_and_quiet_blocks_all_reads(tmp_path):
    found, api, clock = collector(tmp_path, rows=(([0, 1], "192.0.2.10", 8000, "Prior"),))
    site = found.sources[0].site_path
    value = json.loads(site.read_text())
    value["deployments"][0]["stopped"] = True
    site.write_text(json.dumps(value))
    install_reports(found, reports(found, clock), tmp_path)
    found.quiet_path.write_text("")
    found.cycle()
    assert api.requests == []
    found.quiet_path.unlink()
    found.cycle()
    members = found.snapshot()["groups"][0]["members"]
    assert len(members) == 1 and members[0]["stopped_records"][0]["model"] == "Prior"
    assert all(urlparse(r.full_url).port == 9123 for r in api.requests)
    details = "\n".join("".join(text for text, _ in line) for line in dashboard_tui.model_details(members[0], clock.now))
    assert "Not installed by SparkRing: no automatic recovery, no install record" in details
    assert "Prior" in details and "positions 0, 2" in details and "abcdef123456" in details


@pytest.mark.parametrize("case", ["managed", "idle", "stale", "missing-head", "bad-port", "refused"])
def test_no_discovery_scans_and_busy_without_api(tmp_path, case):
    found, api, clock = collector(tmp_path, rows=())
    docs = reports(found, clock)
    if case == "managed":
        for doc in docs:
            doc["model_containers"][0]["managed"] = True
    elif case == "idle":
        docs = []
    elif case == "stale":
        for doc in docs:
            doc["observed_at"] -= 91
    elif case == "missing-head":
        docs = docs[1:]
    elif case == "bad-port":
        docs[0]["model_containers"][0]["api_port"] = 65536
    elif case == "refused":
        api.behaviour[(found.sources[0].fabric["positions"][0]["address"], 9123)] = "refused"
    install_reports(found, docs, tmp_path)
    found.cycle()
    if case != "refused":
        assert api.requests == []
    members = found.snapshot()["groups"][0]["members"]
    if case in ("missing-head", "bad-port", "refused"):
        assert members[0]["state"] == "busy"
        assert members[0]["containers"]
    else:
        assert not members


def test_facts_expire_and_polled_api_uses_shared_cadence(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    install_reports(found, reports(found, clock), tmp_path)
    run_for(found, clock, 21)
    assert [urlparse(r.full_url).path for r in api.requests].count("/metrics") == 3
    clock.now += 100
    found.cycle()
    assert not found.models


def test_eleven_sparks_and_rank_zero_not_lowest_position(tmp_path):
    found, _, clock = collector(tmp_path, rows=())
    found.reload_sources()
    fabric = found.sources[0].fabric
    fabric["positions"] = [{"position": i, "node_id": str(i), "address": f"192.0.2.{10+i}"} for i in range(11)]
    docs = [{"node_id": str((rank + 4) % 11), "observed_at": clock.now, "model_containers": [
        {**nc.launch_facts(argv(size=11, rank=rank)), "master_addr": "192.0.2.14",
         "managed": False, "name": str(rank)}]} for rank in range(11)]
    group = dashboard.unmanaged_groups(fabric, docs, [], clock.now)[0]
    assert group["positions"] == list(range(11)) and group["head_position"] == 4
    assert group["api_host"] == "192.0.2.14" and group["expected_sparks"] == 11
    assert group["group_note"] is None


def test_gpu_child_finds_docker_exec_launch_without_environment(tmp_path):
    cid = "a" * 64
    for pid, parent, args in ((50, 40, ["VLLM::EngineCore"]), (40, 1, argv())):
        path = tmp_path / "proc" / str(pid)
        path.mkdir(parents=True)
        (path / "cgroup").write_text(f"0::/system.slice/docker-{cid}.scope")
        (path / "status").write_text(f"Name:\tpython\nPPid:\t{parent}\n")
        (path / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")
        (path / "environ").write_bytes(b"SECRET=PRIVATE")
    observed = nc.observations({"docker": {"containers": [{"id": cid, "state": "running", "name": "manual"}]},
                                "gpu": {"compute_processes": [50]}}, root=tmp_path,
                               inspect=lambda *a, **kw: {"Path": "sleep", "Args": ["infinity"]})
    assert observed[0]["engine"] == "vllm" and observed[0]["api_port"] == 9123
    assert "PRIVATE" not in json.dumps(observed)


@pytest.mark.parametrize("label,name", [("io.sparkring.deployment", "manual"), ("org.sparkring.profile", "manual"),
                                       (None, "sr-example-abcdef-r0")])
def test_agent_recognizes_managed_ownership(tmp_path, label, name):
    observed = nc.observations({"docker": {"containers": [{"id": "a" * 64, "state": "running", "name": name}]}},
        root=tmp_path, inspect=lambda *a, **kw: {"Path": "vllm", "Args": argv()[1:],
                                                "Config": {"Labels": {label: "record"} if label else {}}})
    assert observed[0]["managed"] is True


def test_agent_status_includes_facts_at_existing_cycle_without_extra_commands(tmp_path, monkeypatch):
    from runtime.host import node
    from runtime.host.test_fabric_routes import member
    _, facts, spark = member(tmp_path)
    seen = []
    monkeypatch.setattr(nc, "observations", lambda f, **kw: seen.append(f) or [{"name": "model", "engine": "vllm"}])
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert status["state"] == "network-configured"
    assert seen == [facts] and status["model_containers"] == [{"name": "model", "engine": "vllm"}]
    assert not any(c[0] == "docker" for c in spark.calls)


def test_sglang_metrics_use_shared_counter_contract():
    metrics = dashboard.parse_metrics('\n'.join([
        'sglang:prompt_tokens_total{model_name="Served"} 120',
        'sglang:generation_tokens_total{model_name="Served"} 30',
        'sglang:num_running_reqs{model_name="Served"} 2',
        'sglang:num_queue_reqs{model_name="Served"} 1',
        'sglang:token_usage{model_name="Served"} .25',
        'sglang:time_to_first_token_seconds_sum{model_name="Served"} 3',
        'sglang:time_to_first_token_seconds_count{model_name="Served"} 2']))
    assert {k: metrics[k] for k in ("prompt", "gen", "running", "waiting", "kv", "ttft_sum", "ttft_count")} == {
        "prompt": 120, "gen": 30, "running": 2, "waiting": 1, "kv": .25, "ttft_sum": 3, "ttft_count": 2}


def test_api_without_metrics_is_serving_with_unknown_counters(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    install_reports(found, reports(found, clock, engine="sglang"), tmp_path)
    original = api.open
    def no_metrics(request, **kwargs):
        if urlparse(request.full_url).path == "/metrics":
            from urllib.error import HTTPError
            raise HTTPError(request.full_url, 404, "metrics disabled", {}, None)
        return original(request, **kwargs)
    api.open = no_metrics
    found.cycle()
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["state"] == "serving" and member["counters"] is None
    clock.now += 20
    found.cycle()
    assert found.snapshot()["groups"][0]["members"][0]["state"] == "serving"


def test_remote_cache_projects_only_public_facts_without_active_deployments(tmp_path, monkeypatch):
    from runtime.host import dashboard_admin, recovery
    found, _, clock = collector(tmp_path, rows=())
    docs = reports(found, clock)
    docs[0]["environment"] = "PRIVATE"
    docs[0]["model_containers"][0]["environment"] = "PRIVATE"
    docs[0]["model_containers"][0]["default_chat_template_kwargs"] = {
        "reasoning_effort": "high", "unrelated": "PRIVATE"}
    state = tmp_path / "controller"
    state.mkdir()
    hosts = ["root@192.0.2.10", "root@192.0.2.12"]
    (state / "cluster.json").write_text(json.dumps({"plan": {"spec": {"hosts": [{"host": h} for h in hosts]}}}))
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "disabled")
    calls = []
    def read(hosts, **kwargs):
        calls.append(list(hosts))
        return dict(zip(hosts, docs))
    monkeypatch.setattr(recovery, "node_documents", read)
    output = tmp_path / "nodes.json"
    arguments = dict(state_root=state, output=output, quiet_path=found.quiet_path, now=clock)
    assert dashboard_admin.refresh_nodes(**arguments) == dict(zip(hosts, docs))
    assert calls == [hosts] and "PRIVATE" not in output.read_text()
    found.node_reports_path = output
    found.cycle()
    assert len(found.models) == 1
    assert found.snapshot()["groups"][0]["members"][0]["thinking"] == "on · high"
    found.quiet_path.write_text("")
    assert dashboard_admin.refresh_nodes(**arguments) == {} and calls == [hosts]
    found.quiet_path.unlink()
    dashboard_admin.DISABLED_PATH.write_text("")
    assert dashboard_admin.refresh_nodes(**arguments) == {} and calls == [hosts]


def test_timer_reuses_cached_node_status_reads(tmp_path, monkeypatch):
    from runtime.host import controller, dashboard_admin, host_load, recovery
    from pathlib import Path
    monkeypatch.setattr(controller, "STATE", tmp_path)
    docs = {"root@192.0.2.12": {"node_id": "example", "model_containers": []}}
    monkeypatch.setattr(dashboard_admin, "refresh_nodes", lambda **kw: docs)
    commands = []
    monkeypatch.setattr(host_load, "recovery_invoke", lambda: lambda *a, **kw: commands.append(a) or "done")
    def check(**kw):
        expected = dict(docs["root@192.0.2.12"])
        assert json.loads(kw["invoke"]("root@192.0.2.12", ["/usr/bin/sparkring", "node", "status"])) == expected
        assert kw["invoke"]("root@192.0.2.12", ["/usr/bin/sparkring", "node", "status"]) == "done"
        assert kw["invoke"]("root@192.0.2.12", ["recorded-command"]) == "done"
        return {"state": "idle", "summary": "no managed deployment"}
    monkeypatch.setattr(recovery, "check", check)
    assert recovery.main(["--auto", "--json"]) == 0
    assert len(commands) == 2
    unit = (Path(__file__).resolve().parents[2] / "packaging/debian/sparkring-dashboard.service").read_text()
    assert "Wants=network-online.target sparkring-recover.timer" in unit


@pytest.mark.parametrize("bind,allowed", [("203.0.113.99", False), ("127.0.0.1", False), ("192.0.2.10", True)])
def test_api_bind_must_belong_to_head_spark_not_rendezvous(tmp_path, bind, allowed):
    found, api, clock = collector(tmp_path, rows=())
    docs = reports(found, clock)
    docs[0]["model_containers"][0]["api_bind"] = bind
    docs[0]["model_containers"][0]["master_addr"] = "203.0.113.99"
    docs[1]["model_containers"][0]["master_addr"] = "203.0.113.99"
    install_reports(found, docs, tmp_path)
    found.cycle()
    assert bool(api.requests) == allowed
    assert not any("203.0.113.99" in r.full_url for r in api.requests)


def test_http_redirect_cannot_widen_read_paths_or_reported_port():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/v1/chat/completions")
            self.end_headers()

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        client = dashboard.ReadOnlyClient()
        client.allow({"127.0.0.1"})
        result = client.get("127.0.0.1", server.server_port, "/metrics")
        assert result.failure == "http" and result.code == 302
        assert received == ["/metrics"]
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
