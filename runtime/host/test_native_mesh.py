"""Fresh/adopted mesh plans preserve admission barriers and exact identities."""
import copy
import base64
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from runtime.common import installer
from runtime.host import controller, models, native_mesh, topology
from runtime.host.test_appliance import nodes

PROFILE = "qwen38-flash-next-qad-tp4"


def cluster():
    found = nodes()
    return {"name": "home", "plan": topology.build_spec(found, found[0]["node_id"])}


def fresh_site():
    value = cluster()
    return native_mesh.select(controller.model_site(value, PROFILE), value, PROFILE,
                              invoke=lambda *a, **k: '{"mesh":null}')


def test_catalog_distinguishes_model_size_version_and_automation_support():
    rows = {r["profile"]: r for r in models.catalog()}
    assert rows[PROFILE]["automated"]
    assert not rows["qwen38-27b-exl3-k5k6"]["automated"]
    assert "27B" in rows["qwen38-27b-exl3-k5k6"]["title"]
    with pytest.raises(ValueError, match="family"):
        models.select("qwen", 4)
    with pytest.raises(ValueError, match="requires 4"):
        models.select(PROFILE, 2)
    assert models.select(PROFILE, 4) == PROFILE


def test_new_mesh_is_fully_planned_without_manual_site_hashes_or_host_changes():
    site = fresh_site()
    lock = installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)
    assert lock["site_input"]["native_mesh"]["mode"] == "create"
    assert all(row["host_ip"] == row["management_ip"] for row in lock["site"]["ranks"])
    names = [p["id"] for p in installer.operation_plan(lock, "up")["phases"]]
    assert names.index("model") < names.index("create") < names.index("mesh-install")
    assert names.index("mesh-up") < names.index("mesh-gate") < names.index("preflight") < names.index("start-workers") < names.index("start-api")
    assert "mesh-replace" in names
    assert len(installer.rendered(lock)) == 8


def test_existing_mesh_is_verified_and_uses_its_bootstrap_addresses():
    value = cluster()
    responses = iter({"mesh": {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json",
                                              "site_sha256": "c" * 64, "plan_sha256": "d" * 64},
                              "host_ip": f"192.0.2.{110 + rank}", "interface": "eth0", "unit": "sparkring-mesh.service"}} for rank in range(4))
    site = native_mesh.select(controller.model_site(value, PROFILE), value, PROFILE,
                              invoke=lambda *a, **k: json.dumps(next(responses)))
    assert "native_mesh" not in site
    lock = installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)
    assert lock["site"]["ranks"][0]["host_ip"] == "192.0.2.110"
    assert lock["site"]["ranks"][0]["interface"] == "eth0"
    names = [p["id"] for p in installer.operation_plan(lock, "up")["phases"]]
    assert "mesh-install" not in names
    assert names.index("preflight") < names.index("create")


def test_mixed_or_disagreeing_meshes_do_not_silently_reconfigure():
    value = cluster()
    responses = iter([{"mesh": {"reference": {"site_sha256": "c" * 64, "plan_sha256": "d" * 64}}}, *[{"mesh": None}] * 3])
    with pytest.raises(ValueError, match="part"):
        native_mesh.select(controller.model_site(value, PROFILE), value, PROFILE,
                           invoke=lambda *a, **k: json.dumps(next(responses)))


@pytest.mark.parametrize("mutate", [
    lambda s: s["native_mesh"]["site"].update(marker_binary="/tmp/foreign-program"),
    lambda s: s["native_mesh"]["site"].update(topology_file="../../foreign.json"),
    lambda s: s["native_mesh"]["reference"].update(site_path="/etc/foreign.json"),
    lambda s: s["native_mesh"]["site"].update(marker_binary_sha256="f" * 64),
])
def test_native_lock_rejects_unplanned_paths_and_artifacts(mutate):
    site = fresh_site()
    mutate(site)
    with pytest.raises(ValueError):
        installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)


def test_fresh_rehearsal_keeps_previous_deployment_workspace_distinct():
    value = cluster()
    before = controller.model_site(value, PROFILE)
    fresh = controller.model_site(value, PROFILE, "fresh")
    assert before["name"] != fresh["name"] and before["workspace"] != fresh["workspace"]
    assert len(before["name"]) <= 40 and len(fresh["name"]) <= 40


def test_read_only_native_admission_uses_root_for_all_launch_phases(monkeypatch):
    from scripts import installer_runner
    site = fresh_site()
    runner = installer_runner.Runner.__new__(installer_runner.Runner)
    runner.lock = installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)
    calls = []
    monkeypatch.setattr(installer_runner, "ssh", lambda host, argv, **kw: calls.append(argv) or '{"ok":true}')
    for action in ("preflight", "create", "start", "mesh-install-local"):
        runner.remote(0, action)
        assert calls[-1][:2] == ["sudo", "-n"]


def test_live_lldp_direct_chassis_without_hostname_is_normalized():
    document = {"lldp": {"interface": [{"port0": {"chassis": {"id": {"type": "mac", "value": "02:00:00:00:00:01"}},
                                                   "port": {"id": {"type": "mac", "value": "02:00:00:00:00:02"}}}}]}}
    original = copy.deepcopy(document)
    rows = topology.lldp_rows(document)
    assert rows[0]["chassis"] == "02:00:00:00:00:01" and rows[0]["hostname"] == ""
    assert document == original


def test_managed_source_snapshot_and_units_work_without_a_glm_model(tmp_path):
    site = fresh_site()
    lock = installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)
    planned = site["native_mesh"]
    owner, units = native_mesh.modules()
    owner.install_code(owner.source_payloads(), tmp_path / "source")
    subprocess.run([sys.executable, "-B", str(tmp_path / "source/runtime/glm53-spark-mtp3-mesh/managed_service.py"), "--help"],
                   capture_output=True, text=True, check=True)
    (tmp_path / "site.json").write_text(installer.compose.encoded(planned["site"]))
    (tmp_path / "fabric.json").write_text(installer.compose.encoded(planned["topology"]))
    specs = installer.specifications(lock)
    containers = [{"Id": str(n + 1) * 64, "Image": lock["selection"]["image_id"], "Name": "/" + specs[n].name,
                   "State": {"Running": False}, "HostConfig": {"RestartPolicy": {"Name": "no"}}, "Config": {"Env": []}} for n in range(4)]
    layout = native_mesh.managed_deployment.layout(site["name"])
    units.render(tmp_path / "site.json", containers, tmp_path / "units", layout["code_dir"], layout["config_dir"],
                 "c" * 32, 9976, deployment_name=site["name"])
    config = json.loads((tmp_path / "units/rank0/service.json").read_text())
    assert config["container_id"] == containers[0]["Id"]
    assert config["container_image"] == lock["selection"]["image_id"]
    assert (tmp_path / "units/rank0" / layout["mesh_unit"]).is_file()


def test_discovery_keeps_configured_links_with_existing_rdma_users(monkeypatch):
    from runtime.host import seed
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "nmcli":
            return SimpleNamespace(returncode=0, stdout="12345678-1234-1234-1234-123456789012", stderr="")
        if argv[:3] == ["ip", "-j", "-6"]:
            return SimpleNamespace(returncode=0, stdout='[{"addr_info":[{"scope":"link","local":"fe80::1"}]}]', stderr="")
        assert argv[0] not in ("rdma", "docker", "nvidia-smi", "ip")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(seed.control_node, "write", lambda *a, **k: None)
    algorithm = b"ssh-ed25519"
    encoded = len(algorithm).to_bytes(4, "big") + algorithm + (32).to_bytes(4, "big") + bytes(range(32))
    public = algorithm.decode() + " " + base64.b64encode(encoded).decode() + " fixture"
    assert seed.prepare(public, interfaces=["p0", "p1", "p2", "p3"], run=run)["prepared"]
    assert not any(a[0] == "rdma" for a in calls)


@pytest.mark.parametrize("approve", [None, "yes", "no"])
def test_port_preparation_ignores_tool_containers_and_stops_approved_gpu_containers(monkeypatch, approve):
    from runtime.host import seed
    state = {"gpu_running": True, "stopped": []}

    def run(argv, **kw):
        if argv[:2] == ["nmcli", "-g"]:
            return SimpleNamespace(returncode=0, stdout="--", stderr="")
        if argv[:3] == ["ip", "-j", "-6"]:
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        if argv[:3] == ["ip", "-j", "-4"]:
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        if argv[:2] == ["docker", "ps"]:
            return SimpleNamespace(returncode=0, stdout="tool\nmodel\n" if state["gpu_running"] else "tool\n", stderr="")
        if argv[:2] == ["docker", "inspect"]:
            rows = [{"Id": "tool", "Name": "/netadm", "HostConfig": {}}]
            if "model" in argv:
                rows.append({"Id": "model", "Name": "/qwen-r0", "HostConfig": {"DeviceRequests": [{"Driver": "nvidia"}]}})
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        if argv[:2] == ["docker", "stop"]:
            state["stopped"].append(argv[-1])
            state["gpu_running"] = False
        if argv[0] == "rdma":
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(seed.control_node, "write", lambda *a, **k: None)
    monkeypatch.setattr(seed.control, "netdev", lambda name: name)
    monkeypatch.setattr(seed.Path, "read_text", lambda self: "1", raising=False)
    algorithm = b"ssh-ed25519"
    encoded = len(algorithm).to_bytes(4, "big") + algorithm + (32).to_bytes(4, "big") + bytes(range(32))
    public = algorithm.decode() + " " + base64.b64encode(encoded).decode() + " fixture"
    asked = []

    def stop(names):
        asked.append(names)
        if approve == "no":
            raise ValueError("Cancelled; no further changes")
    if approve == "yes":
        assert seed.prepare(public, interfaces=["p0", "p1", "p2", "p3"], run=run, stop=stop)["prepared"]
        assert asked == [["qwen-r0"]] and state["stopped"] == ["model"]
    else:
        with pytest.raises(ValueError, match="qwen-r0" if approve is None else "Cancelled"):
            seed.prepare(public, interfaces=["p0", "p1", "p2", "p3"], run=run, stop=stop if approve else None)
        assert state["stopped"] == []


@pytest.mark.parametrize("approved", [False, True])
def test_existing_fabric_connection_gets_link_local_only_with_approval(monkeypatch, approved):
    from runtime.host import seed
    state = {"link_local": False, "calls": []}

    def run(argv, **kw):
        state["calls"].append(argv)
        if argv[:3] == ["nmcli", "-g", "GENERAL.CON-UUID"]:
            return SimpleNamespace(returncode=0, stdout="uuid-" + argv[-1], stderr="")
        if argv[:3] == ["nmcli", "-g", "connection.id"]:
            return SimpleNamespace(returncode=0, stdout="Wired connection 4", stderr="")
        if argv[:3] == ["nmcli", "device", "reapply"]:
            state["link_local"] = True
        if argv[:3] == ["ip", "-j", "-6"]:
            info = [{"scope": "link", "local": "fe80::1"}] if state["link_local"] else []
            return SimpleNamespace(returncode=0, stdout=json.dumps([{"addr_info": info}]), stderr="")
        if argv[:2] == ["docker", "ps"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[0] == "rdma":
            rows = [{"ifname": "rocep1s0f0", "comm": "ib_core", "type": "GSI"}]
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(seed.control_node, "write", lambda *a, **k: None)
    monkeypatch.setattr(seed.control, "netdev", lambda name: name)
    monkeypatch.setattr(seed.Path, "read_text", lambda self: "1", raising=False)
    algorithm = b"ssh-ed25519"
    encoded = len(algorithm).to_bytes(4, "big") + algorithm + (32).to_bytes(4, "big") + bytes(range(32))
    public = algorithm.decode() + " " + base64.b64encode(encoded).decode() + " fixture"
    approvals = []
    if approved:
        assert seed.prepare(public, interfaces=["p0", "p1", "p2", "p3"], run=run, link_local=approvals.append)["prepared"]
        assert approvals[0] == "Wired connection 4 (p0)"
        assert ["nmcli", "connection", "modify", "uuid-p0", "ipv6.method", "link-local"] in state["calls"]
        assert not any(argv[:3] == ["nmcli", "connection", "add"] for argv in state["calls"])
    else:
        with pytest.raises(ValueError, match="no IPv6 link-local"):
            seed.prepare(public, interfaces=["p0", "p1", "p2", "p3"], run=run)
        assert not any(argv[:3] == ["nmcli", "connection", "modify"] for argv in state["calls"])


def test_port_preparation_names_a_foreign_service_on_the_ssh_port(monkeypatch):
    from runtime.host import seed

    def run(argv, **kw):
        if argv[:3] == ["nmcli", "-g", "GENERAL.CON-UUID"]:
            return SimpleNamespace(returncode=0, stdout="uuid", stderr="")
        if argv[:3] == ["ip", "-j", "-6"]:
            return SimpleNamespace(returncode=0, stdout='[{"addr_info":[{"scope":"link","local":"fe80::1"}]}]', stderr="")
        if argv[0] == "ss":
            return SimpleNamespace(returncode=0, stdout='LISTEN 0 1000 0.0.0.0:2222 0.0.0.0:* users:(("dropbear",pid=1965,fd=3))\n', stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(seed.control_node, "write", lambda *a, **k: pytest.fail("configured before the port check"))
    algorithm = b"ssh-ed25519"
    encoded = len(algorithm).to_bytes(4, "big") + algorithm + (32).to_bytes(4, "big") + bytes(range(32))
    public = algorithm.decode() + " " + base64.b64encode(encoded).decode() + " fixture"
    with pytest.raises(ValueError, match="dropbear"):
        seed.prepare(public, interfaces=["p0", "p1", "p2", "p3"], run=run)
