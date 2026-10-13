"""Offline replay of the wrapper inspect and incomplete cache captured on ring8c."""
import copy
import json
import os
from pathlib import Path

import pytest

from runtime.host import dashboard, dashboard_admin, node_containers as nc, recovery
from runtime.host.test_dashboard import collector
from runtime.host.test_dashboard_unmanaged import install_reports

FIXTURES = Path(__file__).with_name("fixtures") / "dashboard_unmanaged"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def ring_observation(tmp_path, shared=True):
    found, api, clock = collector(tmp_path, rows=(([6, 7], "192.0.2.16", 8000, "Prior DCP2"),))
    site = json.loads(found.sources[0].site_path.read_text())
    site["deployments"][0]["stopped"] = True
    found.sources[0].site_path.write_text(json.dumps(site))
    fabric = json.loads(found.sources[0].fabric_path.read_text())
    docs = fixture("nodes-missing-six.json")["nodes"] + [fixture("status-six.json")]
    docs = {d["node_id"]: d for d in docs}
    for p in fabric["positions"]:
        i = p["position"]
        p["node_id"] = f"fixture-node-{i}"
        # Each pair shares a cable; every cable may reuse the same subnet.
        p["ports"] = {"0": {"peer": {"position": i ^ 1, "port": 0}, "functions": {
            "primary": {"address": f"198.18.{0 if shared else (i // 2) * 4}.{1 + i % 2}/24"}}}}
        inspect = copy.deepcopy(fixture("inspect-wrapper.json")[i % 2])
        args = inspect["Args"]
        args[args.index("--master-addr") + 1] = f"198.18.{0 if shared else (i // 2) * 4}.1"
        args[args.index("--served-model-name") + 1] = f"Research-pair-{i // 2}"
        observed = nc.observations({"docker": {"containers": [
            {"id": "a" * 64, "name": f"research-{i}", "state": "running"}]}},
            root=tmp_path, inspect=lambda *a, **kw: inspect)
        docs[p["node_id"]].update(observed_at=clock.now, model_containers=observed)
        api.models[(p["management"]["address"], 8000)] = f"Research-pair-{i // 2}"
    found.sources[0].fabric_path.write_text(json.dumps(fabric))
    install_reports(found, list(reversed(list(docs.values()))), tmp_path)
    return found, api, clock, fabric, [docs[f"fixture-node-{i}"] for i in range(8)]


@pytest.mark.parametrize("config_only", [False, True])
def test_real_wrapper_inspect_without_gpu_pids(tmp_path, config_only):
    for rank, doc in enumerate(fixture("inspect-wrapper.json")):
        if config_only:
            doc.pop("Path")
            doc.pop("Args")
        observed = nc.observations({"docker": {"containers": [
            {"id": "a" * 64, "name": f"manual-{rank}", "state": "running"}]}},
            root=tmp_path, inspect=lambda *a, **kw: doc)
        assert len(observed) == 1
        c = observed[0]
        assert (c["engine"], c["tp"], c["dcp"], c["node_rank"]) == ("vllm", 2, 2, rank)
        assert c["served_model"] == "GLM-5.3-Flash-CSF-DCP2-TP2"
        assert c["master_addr"] == "198.18.12.1"
        if rank == 0:
            assert c["api_port"] == 8000


@pytest.mark.parametrize("command", [
    ["python3", "-m", "vllm.entrypoints.cli.main", "serve", "Model"],
    ["vllm", "serve", "Model"],
    ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "Model"],
    ["python3", "/opt/sparkring/toolchain/toolchain.py", "serve", "Model"],
    ["python3", "-m", "sglang.launch_server", "--model-path", "Model"],
    ["sglang", "serve", "--model-path", "Model"],
])
def test_server_entrypoints(command):
    facts = nc.launch_facts(command)
    assert facts["engine"] == ("sglang" if any("sglang" in a for a in command) else "vllm")
    assert facts["served_model"] == "Model"


@pytest.mark.parametrize("shared", [False, True])
def test_four_pairs_have_distinct_heads_apis_and_stopped_record_details(tmp_path, shared):
    found, api, _, fabric, _ = ring_observation(tmp_path, shared)
    found.cycle()
    members = found.snapshot()["groups"][0]["members"]
    assert len(members) == 4
    members.sort(key=lambda m: m["arc"]["positions"])
    for pair, member in enumerate(members):
        assert member["arc"]["positions"] == [pair * 2, pair * 2 + 1]
        assert member["state"] == "serving" and member["unmanaged"]
        assert member["launch"]["tp"] == member["launch"]["dcp"] == 2
        assert member["model"] == f"Research-pair-{pair}"
        assert member["group_note"] is None
    assert members[3]["stopped_records"][0]["model"] == "Prior DCP2"
    from urllib.parse import urlparse
    assert {(urlparse(r.full_url).hostname, urlparse(r.full_url).port) for r in api.requests} == {
        (p["management"]["address"], 8000) for p in fabric["positions"][::2]}


def test_tp4_master_reached_over_relay_path(tmp_path):
    from runtime.common import fabric_layout
    from runtime.host.test_fabric import document_of
    _, _, clock, _, docs = ring_observation(tmp_path, shared=False)
    _, fabric, _ = document_of(fabric_layout.layout("cycle", 8))
    for p in fabric["positions"]:
        p["node_id"] = f"fixture-node-{p['position']}"
    master = fabric["positions"][0]["ports"]["0"]["functions"]["primary"]["address"].split("/")[0]
    docs = docs[:4]
    for rank, doc in enumerate(docs):
        doc["model_containers"][0].update(master_addr=master, nnodes=4, tp=4, node_rank=rank)
    groups = dashboard.unmanaged_groups(dashboard.fabric_from_document(fabric), docs, [], clock.now)
    assert len(groups) == 1 and groups[0]["positions"] == [0, 1, 2, 3]
    assert groups[0]["api_host"] == fabric["positions"][0]["management"]["address"]


def test_ambiguous_workers_do_not_merge_heads_and_missing_head_keeps_pair(tmp_path):
    _, _, clock, fabric, docs = ring_observation(tmp_path)
    parsed = dashboard.fabric_from_document(fabric)
    docs[0]["model_containers"] = []
    groups = dashboard.unmanaged_groups(parsed, docs, [], clock.now)
    worker = next(g for g in groups if 1 in g["positions"])
    assert worker["positions"] == [1] and worker["api_host"] is None
    assert len(groups) == 4
    parsed["positions"][1]["neighbors"] = []
    groups = dashboard.unmanaged_groups(parsed, docs, [], clock.now)
    assert next(g for g in groups if 1 in g["positions"])["positions"] == [1]


def test_failed_status_read_keeps_position_six_and_does_not_shift_seven(tmp_path, monkeypatch):
    found, _, clock, fabric, docs = ring_observation(tmp_path)
    state = tmp_path / "controller"
    state.mkdir()
    (state / "fabric.json").write_text(json.dumps(fabric))
    hosts = ["root@" + p["management"]["address"] for p in fabric["positions"]]
    (state / "cluster.json").write_text(json.dumps({"plan": {"spec": {"hosts": [{"host": h} for h in hosts]}}}))
    replies = dict(zip(hosts, docs))
    output = tmp_path / "public" / "nodes.json"
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "disabled")
    monkeypatch.setattr(recovery, "node_documents", lambda *a, **kw: replies)
    args = dict(state_root=state, output=output, quiet_path=found.quiet_path, now=clock)
    dashboard_admin.refresh_nodes(**args)
    if os.name == "posix":
        assert output.parent.stat().st_mode & 0o005 == 0o005
    observed = docs[6]["observed_at"]
    replies[hosts[6]] = {"state": "unreachable", "error": "timeout"}
    clock.now += 10
    dashboard_admin.refresh_nodes(**args)
    cached = json.loads(output.read_text())["nodes"]
    assert len(cached) == 8
    indexed = {d["node_id"]: d for d in cached}
    assert indexed["fixture-node-6"]["observed_at"] == observed
    assert indexed["fixture-node-6"]["status_read"] == "unreachable"
    assert indexed["fixture-node-7"]["model_containers"][0]["node_rank"] == 1
    clock.now += 100
    assert not dashboard.unmanaged_groups(dashboard.fabric_from_document(fabric), cached, [], clock.now)
    # A first failed read still publishes the setup identity, never invented facts.
    output.unlink()
    dashboard_admin.refresh_nodes(**args)
    indexed = {d["node_id"]: d for d in json.loads(output.read_text())["nodes"]}
    assert indexed["fixture-node-6"]["position"] == 6
    assert indexed["fixture-node-6"]["model_containers"] == []


def test_overlapping_cache_refresh_does_not_replace_another_writer(tmp_path, monkeypatch):
    from runtime.common import process_lock
    state = tmp_path / "controller"
    state.mkdir()
    (state / "cluster.json").write_text(json.dumps({"plan": {"spec": {"hosts": [{"host": "root@192.0.2.10"}]}}}))
    output = tmp_path / "nodes.json"
    output.write_text(json.dumps(fixture("nodes-missing-six.json")))
    original = output.read_bytes()
    monkeypatch.setattr(dashboard_admin, "DISABLED_PATH", tmp_path / "disabled")
    monkeypatch.setattr(recovery, "node_documents", lambda *a, **kw: pytest.fail("competing writer collected facts"))
    with process_lock.hold(output.with_suffix(".lock")):
        assert dashboard_admin.refresh_nodes(state_root=state, output=output, quiet_path=tmp_path / "quiet") == {}
    assert output.read_bytes() == original


def test_cache_rejects_wrong_identity_and_older_observations(tmp_path):
    _, _, clock, fabric, docs = ring_observation(tmp_path)
    state = tmp_path / "controller"
    state.mkdir()
    (state / "fabric.json").write_text(json.dumps(fabric))
    output = tmp_path / "prior.json"
    output.write_text(json.dumps({"nodes": list(reversed(docs))}))
    host = "root@" + fabric["positions"][6]["management"]["address"]
    wrong = dashboard_admin.cached_node_rows(state, output, {host: docs[7]}, clock.now)
    indexed = {d["node_id"]: d for d in wrong}
    assert indexed["fixture-node-6"]["status_read"] == "identity-mismatch"
    assert indexed["fixture-node-6"]["model_containers"] == docs[6]["model_containers"]
    old = copy.deepcopy(docs[6])
    old.update(observed_at=clock.now - 100, model_containers=[])
    rows = dashboard_admin.cached_node_rows(state, output, {host: old}, clock.now)
    assert next(d for d in rows if d["node_id"] == "fixture-node-6")["observed_at"] == clock.now


def test_agent_runtime_directory_preserves_quiet_on_package_restart(tmp_path):
    root = Path(__file__).resolve().parents[2]
    unit = (root / "packaging/debian/sparkring-agent.service").read_text()
    assert "RuntimeDirectoryPreserve=yes" in unit
    postinst = (root / "packaging/debian/postinst").read_text()
    assert "systemctl restart sparkring-agent.service" in postinst
    quiet = tmp_path / "sparkring" / "dashboard.quiet"
    quiet.parent.mkdir()
    quiet.touch()
    held = dashboard.quiet_hold(quiet, "research", now=100, ttl=600)
    state = dashboard.quiet_state(quiet, 101)
    assert state["operator"] and state["holders"][0]["id"] == held["holder"]


@pytest.mark.skipif(os.environ.get("SPARKRING_TEST_SYSTEMD") != "1",
                    reason="opt-in local root systemd lifecycle test")
def test_systemd_stop_start_preserves_both_quiet_owners():
    import shutil
    import subprocess
    import time
    import uuid
    root = Path(__file__).resolve().parents[2]
    packaged = (root / "packaging/debian/sparkring-agent.service").read_text()
    preserve = next(line for line in packaged.splitlines() if line.startswith("RuntimeDirectoryPreserve="))
    name = "sparkring-quiet-test-" + uuid.uuid4().hex
    unit = Path("/run/systemd/system") / (name + ".service")
    runtime = Path("/run") / name

    def systemctl(*args):
        subprocess.run(["systemctl", *args], check=True, capture_output=True)

    try:
        unit.write_text(f"[Service]\nExecStart=/usr/bin/sleep infinity\nRuntimeDirectory={name}\n{preserve}\n")
        systemctl("daemon-reload")
        systemctl("start", unit.name)
        quiet = runtime / "dashboard.quiet"
        quiet.touch()
        held = dashboard.quiet_hold(quiet, "research", ttl=600)
        holder = runtime / "dashboard-quiet" / (held["holder"] + ".json")
        original = holder.read_bytes()
        for action in ("restart", "stop", "start"):
            systemctl(action, unit.name)
            assert quiet.exists() and holder.read_bytes() == original
            status = dashboard.quiet_state(quiet, time.time())
            assert status["operator"] and status["holders"][0]["id"] == held["holder"]
    finally:
        subprocess.run(["systemctl", "stop", unit.name], capture_output=True)
        unit.unlink(missing_ok=True)
        systemctl("daemon-reload")
        if runtime.exists():
            shutil.rmtree(runtime)
