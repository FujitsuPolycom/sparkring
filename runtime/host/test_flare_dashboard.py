"""Flare processes that SparkRing did not install: launch facts, grouping and cards."""
import json
from urllib.parse import urlparse

from runtime.host import dashboard, node_containers as nc
from runtime.host.test_dashboard import collector


def flare_argv(rank=None, rendezvous=None, host=None, port=None, model=None):
    argv = ["/srv/flare/artifacts/bin/flare", "run", "model.safetensors", "--gpus", "0",
            "--dangerously-approve-key", "SECRET"]
    if rank is not None:
        argv += ["--rank", str(rank)]
    if rendezvous is not None:
        argv += ["--rendezvous", rendezvous]
    if host is not None:
        argv += ["--host", host]
    if port is not None:
        argv += ["--port", str(port)]
    if model is not None:
        argv += ["--served-model-name", model]
    return argv


def flare_row(argv):
    facts = nc.launch_facts(argv)
    return {**facts, "managed": False, "name": nc.flare_row_name(facts), "image": "", "image_id": ""}


def flare_metrics(generated=500, running=2, total=1000, free=250, uptime=100):
    return "\n".join([
        "# TYPE flare_active_requests gauge",
        f"flare_active_requests {running}",
        "flare_pending_requests 1",
        f"flare_total_kv_tokens {total}",
        f"flare_free_kv_tokens {free}",
        "flare_cached_kv_tokens 100",
        f"flare_generated_tokens_total {generated}",
        "flare_prefill_tokens_total 2000",
        "flare_requests_completed_total 10",
        "flare_requests_failed_total 1",
        "flare_requests_cancelled_total 2",
        "flare_preemptions_total 3",
        "flare_speculative_offered_drafts_total 40",
        "flare_speculative_accepted_drafts_total 30",
        "flare_verification_draft_tokens_total 50",
        "flare_cache_lookups_total 10",
        "flare_cache_hits_total 6",
        f"flare_uptime_seconds {uptime}",
    ]) + "\n"


def node_doc(fabric, position, rows, now):
    return {"node_id": fabric["positions"][position]["node_id"], "observed_at": now,
            "model_containers": rows}


def install_reports(found, documents, tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps({"schema": "sparkring-dashboard-nodes/v1", "nodes": documents}), encoding="utf-8")
    found.node_reports_path = path
    found.next_files = 0


def served(found, api, clock, documents, tmp_path, host, port, model="Flare-Model"):
    install_reports(found, documents, tmp_path)
    api.models[(host, port)] = model
    api.bodies[(host, port, "/metrics")] = flare_metrics().encode()
    found.mark_viewed()
    found.cycle()
    return found.snapshot()["groups"][0]["members"][0]


def test_flare_launch_facts_cover_orbit_manual_and_default_binds():
    rank0 = nc.launch_facts(flare_argv(0, "198.18.1.1:29836", "0.0.0.0", 18985, "DeepSeek-V4.1-Flash"))
    assert rank0 == {"engine": "flare", "node_rank": 0, "headless": False,
                     "master_addr": "198.18.1.1", "master_port": 29836,
                     "api_bind": "0.0.0.0", "api_port": 18985,
                     "served_model": "DeepSeek-V4.1-Flash"}
    rank1 = nc.launch_facts(flare_argv(1, "198.18.1.1:29836"))
    assert rank1["node_rank"] == 1 and rank1["headless"] and "api_port" not in rank1
    single = nc.launch_facts(flare_argv(host="0.0.0.0", port=8001))
    assert (single["node_rank"], single["headless"], single["api_port"]) == (0, False, 8001)
    default = nc.launch_facts(["flare-8ff0a231f54c37fccf3250d2", "run", "model.safetensors"])
    assert (default["api_bind"], default["api_port"]) == ("127.0.0.1", 8000)
    assert "SECRET" not in json.dumps(rank0)
    assert nc.launch_facts(["/x/flare", "agent", "--listen", "0.0.0.0:29900"]) == {"engine": "unknown"}
    assert nc.launch_facts(["/x/flare", "orbit", "run", "model"]) == {"engine": "unknown"}


def test_unparsable_flare_host_is_not_published_as_loopback():
    facts = nc.launch_facts(flare_argv(host="spark-0.lan", port=8001))
    assert facts["engine"] == "flare" and "api_bind" not in facts and "api_port" not in facts


def test_bare_flare_process_is_observed_without_moving_its_argv(tmp_path):
    proc = tmp_path / "proc" / "50"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/system.slice/sshd.service")
    (proc / "status").write_text("Name:\tflare\nPPid:\t1\n")
    argv = flare_argv(0, "198.18.1.1:29836", "0.0.0.0", 18985, "DeepSeek-V4.1-Flash")
    (proc / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (proc / "environ").write_bytes(b"TOKEN=PRIVATE")
    observed = nc.observations({"gpu": {"compute_processes": [50]}, "docker": {"containers": []}}, root=tmp_path)
    assert len(observed) == 1
    row = observed[0]
    assert row["engine"] == "flare" and row["api_port"] == 18985 and row["managed"] is False
    assert row["name"].startswith("flare-r0-")
    assert "SECRET" not in json.dumps(row) and "PRIVATE" not in json.dumps(row)


def test_flare_agent_and_orbit_commands_are_not_ranks(tmp_path):
    proc = tmp_path / "proc" / "51"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/system.slice/sshd.service")
    (proc / "status").write_text("Name:\tflare\nPPid:\t1\n")
    (proc / "cmdline").write_bytes(b"\0".join(a.encode() for a in
        ["/x/flare", "agent", "--listen", "0.0.0.0:29900"]) + b"\0")
    assert nc.observations({"gpu": {"compute_processes": [51]}, "docker": {"containers": []}},
                           root=tmp_path) == []


def test_containerized_flare_is_observed_from_its_entrypoint(tmp_path):
    cid = "a" * 64
    observed = nc.observations(
        {"docker": {"containers": [{"id": cid, "state": "running", "name": "flare"}]}},
        root=tmp_path,
        inspect=lambda *a, **kw: {"Path": "/usr/local/bin/flare",
                                  "Args": flare_argv(0, "198.18.1.1:29836", "0.0.0.0", 8001, "Qwen"),
                                  "Image": "sha256:" + "b" * 64,
                                  "Config": {"Image": "flare:tag"}})
    assert observed[0]["engine"] == "flare" and observed[0]["image"] == "flare:tag"
    assert observed[0]["api_port"] == 8001 and observed[0]["managed"] is False


def test_single_flare_group_without_rendezvous_is_one_spark(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    found.reload_sources()
    fabric = found.sources[0].fabric
    row = flare_row(flare_argv(host="0.0.0.0", port=8001))
    documents = [node_doc(fabric, 5, [row], clock.now)]
    member = served(found, api, clock, documents, tmp_path, fabric["positions"][5]["address"], 8001,
                    "Qwen3.8-Flash")
    assert member["flare"] and member["layout"]["label"] == "Flare"
    assert member["arc"]["positions"] == [5] and member["head"]["position"] == 5
    assert member["state"] == "serving" and member["model"] == "Qwen3.8-Flash"
    assert member["live"] and member["live"]["running"] == 2.0 and member["live"]["kv"] == 0.75
    assert member["totals"]["finished"] == {"stop": 10.0, "error": 1.0, "abort": 2.0}
    assert member["group_note"] is None
    assert len(found.history[member["id"]]) == 1
    assert {urlparse(r.full_url).port for r in api.requests} == {8001}


def test_multi_spark_flare_group_uses_rendezvous_and_only_head_api(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    found.reload_sources()
    fabric = found.sources[0].fabric
    documents = [node_doc(fabric, rank, [flare_row(flare_argv(
        rank, "198.18.1.1:29836", "0.0.0.0" if rank == 0 else None, 18985 if rank == 0 else None,
        "DeepSeek-V4.1-Flash" if rank == 0 else None))], clock.now) for rank in range(4)]
    member = served(found, api, clock, documents, tmp_path, "192.0.2.10", 18985,
                    "DeepSeek-V4.1-Flash")
    assert member["arc"]["positions"] == [0, 1, 2, 3]
    assert member["head"]["position"] == 0 and member["head"]["port"] == 18985
    assert member["group_note"] is None and member["layout"]["count"] == 4
    assert member["workers"] == {"received": 4, "expected": 4}
    assert {urlparse(r.full_url).port for r in api.requests} == {18985}
    assert not any(urlparse(r.full_url).port == 8000 for r in api.requests)


def test_missing_flare_head_is_named_not_counted_as_complete(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    found.reload_sources()
    fabric = found.sources[0].fabric
    documents = [node_doc(fabric, rank, [flare_row(flare_argv(rank, "198.18.1.1:29836"))], clock.now)
                 for rank in (1, 2)]
    install_reports(found, documents, tmp_path)
    found.mark_viewed()
    found.cycle()
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["group_note"] == "head Spark not reporting"
    assert member["workers"] == {"received": 2, "expected": 3}
    assert member["arc"]["positions"] == [1, 2] and member["state"] == "busy"
    assert api.requests == []


def test_lone_flare_peer_names_the_missing_head(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    found.reload_sources()
    fabric = found.sources[0].fabric
    documents = [node_doc(fabric, 1, [flare_row(flare_argv(1, "198.18.1.1:29836"))], clock.now)]
    install_reports(found, documents, tmp_path)
    found.mark_viewed()
    found.cycle()
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["group_note"] == "head Spark not reporting"
    assert member["workers"] == {"received": 1, "expected": 2}
    assert member["state"] == "busy" and api.requests == []


def test_loopback_bound_flare_group_shows_process_without_live_counters(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    found.reload_sources()
    fabric = found.sources[0].fabric
    documents = [node_doc(fabric, 6, [flare_row(flare_argv(host="127.0.0.1", port=8000))], clock.now)]
    install_reports(found, documents, tmp_path)
    found.mark_viewed()
    found.cycle()
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["api_local"] is True and member["endpoint"] is None
    assert member["state"] == "busy" and member["head"]["port"] == 8000
    assert member["flare"] is True
    assert api.requests == []


def test_flare_metric_adapter_feeds_the_shared_history_keys():
    metrics = dashboard.parse_flare_as_metrics(flare_metrics())
    assert {k: metrics[k] for k in ("gen", "prompt", "running", "waiting", "kv", "accepted",
                                    "drafts", "draft_tokens", "pc_hits", "pc_queries",
                                    "preemptions", "uptime_seconds")} == {
        "gen": 500.0, "prompt": 2000.0, "running": 2.0, "waiting": 1.0, "kv": 0.75,
        "accepted": 30.0, "drafts": 40.0, "draft_tokens": 50.0, "pc_hits": 6.0,
        "pc_queries": 10.0, "preemptions": 3.0, "uptime_seconds": 100.0}
    assert metrics["finished"] == {"stop": 10.0, "error": 1.0, "abort": 2.0}


def test_discovered_flare_reports_empty_metrics_then_recovers_with_a_json_snapshot(tmp_path):
    found, api, clock = collector(tmp_path, rows=())
    found.reload_sources()
    fabric = found.sources[0].fabric
    row = flare_row(flare_argv(host="0.0.0.0", port=8001))
    install_reports(found, [node_doc(fabric, 5, [row], clock.now)], tmp_path)
    host = fabric["positions"][5]["address"]
    api.models[(host, 8001)] = "Flare-Model"
    api.bodies[(host, 8001, "/metrics")] = b"{}"
    found.mark_viewed()
    found.cycle()
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["flare"] and member["state"] == "serving"
    assert member["metrics_problem"] == "No recognized numeric Flare metrics in /metrics response."
    assert member["counters"]["values"] == {}
    clock.now += dashboard.METRICS_VIEWED_SECONDS
    api.bodies[(host, 8001, "/metrics")] = json.dumps({
        "generated_tokens": 500, "prefill_tokens": 2000, "active": 2, "pending": 1,
        "total_pages": 100, "free_pages": 25, "page_tokens": 128, "completed": 10,
    }).encode()
    found.mark_viewed()
    found.cycle()
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["metrics_problem"] is None and member["model"] == "Flare-Model"
    assert member["counters"]["values"]["gen"] == 500
    assert member["live"]["running"] == 2 and member["live"]["kv"] == 0.75
    assert member["totals"]["finished"] == {"stop": 10}
    assert {urlparse(r.full_url).path for r in api.requests} == {"/metrics", "/v1/models"}
