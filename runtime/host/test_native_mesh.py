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
    assert "mesh-replace" in names and "ring-serve" not in names
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
    # The reused mesh is started and awaited on every rank before the read-only ring check.
    assert names.index("ring-serve") < names.index("preflight") < names.index("create")
    from scripts import deploy_engine
    plan = installer.operation_plan(lock, "up")
    deploy_engine.validate_plan(plan)
    serve = next(p for p in plan["phases"] if p["id"] == "ring-serve")
    assert all(a["risk"] == "mutates-host" and a["verify"]["argv"][1] == "ring-check" for a in serve["actions"])


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
    for action in ("preflight", "create", "start", "ring-serve", "ring-check", "mesh-install-local"):
        runner.remote(0, action)
        assert calls[-1][:2] == ["sudo", "-n"]


REFERENCE = {"site_path": "/etc/sparkring/managed-mesh/site.json", "site_sha256": "c" * 64, "plan_sha256": "d" * 64}
UNIT = "sparkring-mesh.service"


def test_mesh_units_follow_the_managed_layout_of_their_site():
    assert native_mesh.mesh_unit("/etc/sparkring/managed-mesh/site.json") == UNIT
    assert native_mesh.mesh_unit("/etc/sparkring/deployments/home/site.json") == "sparkring-home-mesh.service"
    with pytest.raises(ValueError, match="not the site"):
        native_mesh.mesh_unit("/etc/foreign/site.json")


def test_a_created_mesh_is_started_like_a_reused_one(monkeypatch):
    lock = installer.make_lock(PROFILE, fresh_site(), "a" * 40, "b" * 64)
    read_json = native_mesh.profiles.read_json
    monkeypatch.setattr(native_mesh.profiles, "read_json", lambda path: {"deployment": lock["id"]}
                        if str(path).endswith("installer-owner.json") else read_json(path))
    calls = []
    monkeypatch.setattr(native_mesh, "serve_ring", lambda *args: calls.append(args) or {"ok": True, "action": "started"})
    assert native_mesh.operate_local(lock, 1, "mesh-up")["action"] == "started"
    row = lock["site"]["ranks"][1]
    assert calls == [(row["fabric"], 1, row["hcas"], row["gid"], row["host_ip"])]
    # The served unit is the one this deployment installed.
    unit = native_mesh.managed_deployment.layout(lock["site_input"]["native_mesh"]["name"])["mesh_unit"]
    assert native_mesh.mesh_unit(row["fabric"]["site_path"]) == unit


class Units:
    """systemctl on one Spark: the mesh units that run and those enabled at boot."""

    def __init__(self, active=(), enabled=()):
        self.active, self.enabled = set(active), set(enabled)

    def call(self, argv, accepted=(0,)):
        unit = argv[-1]
        if argv[1] == "is-active":
            return SimpleNamespace(returncode=0 if unit in self.active else 3, stdout="")
        state = "enabled" if unit in self.enabled else "disabled"
        return SimpleNamespace(returncode=0 if unit in self.enabled else 1, stdout=state + "\n")


def installed(tmp_path, monkeypatch, units, *, names=(None,), failure=None):
    """Mesh service configurations of ``names`` (None: the default layout) on one Spark."""
    paths = []
    for name in names:
        layout = native_mesh.managed_deployment.layout(name)
        path = tmp_path / (name or "default") / "service.json"
        path.parent.mkdir()
        site = path.parent / "site.json"
        site.write_text("{}")
        config = {"rank": 1, "site_path": str(site)}
        if name:
            config.update(deployment_name=name, site_path=layout["config_dir"] + "/site.json",
                          key_file=layout["config_dir"] + "/health.key", state_dir=layout["state_dir"])
        path.write_text(json.dumps(config))
        paths.append(path)
    manager = SimpleNamespace(plan=SimpleNamespace(sha256="d" * 64, roce_gid_index=3),
                              site={"management_addresses": [f"192.0.2.{110 + rank}" for rank in range(4)]},
                              local=SimpleNamespace(management_netdev="eth0",
                                                    port=lambda direction, function: SimpleNamespace(rdma_device="mlx5_0")))
    checks = []
    def check(*args):
        checks.append(args)
        if failure:
            raise ValueError(failure)
        return {"ok": True}
    monkeypatch.setattr(native_mesh, "_service_configs", lambda: paths)
    monkeypatch.setattr(native_mesh.node, "call", units.call)
    monkeypatch.setattr(native_mesh.qwen_mesh, "_network", lambda: SimpleNamespace(NetworkManager=lambda *a, **k: manager))
    monkeypatch.setattr(native_mesh.qwen_mesh, "check", check)
    return checks


def test_a_mesh_that_failed_when_a_neighbor_restarted_is_reported_stopped(tmp_path, monkeypatch):
    checks = installed(tmp_path, monkeypatch, Units(enabled=[UNIT]))
    mesh = native_mesh.inspect_local(1)["mesh"]
    assert mesh["active"] is False and mesh["snapshot"] is None and "problem" not in mesh
    assert (mesh["unit"], mesh["host_ip"], mesh["interface"]) == (UNIT, "192.0.2.111", "eth0")
    assert mesh["reference"]["plan_sha256"] == "d" * 64 and checks == []


def test_a_running_mesh_that_fails_its_ring_check_is_reported_with_the_failure(tmp_path, monkeypatch):
    installed(tmp_path, monkeypatch, Units(active=[UNIT], enabled=[UNIT]),
              failure="GID index 3 does not match enp1s0f1np1 IPv4 address")
    mesh = native_mesh.inspect_local(1)["mesh"]
    assert mesh["active"] is True and mesh["snapshot"] is None
    assert mesh["problem"] == "GID index 3 does not match enp1s0f1np1 IPv4 address"


def test_the_running_mesh_is_reported_before_an_enabled_one(tmp_path, monkeypatch):
    installed(tmp_path, monkeypatch, Units(active=[UNIT], enabled=["sparkring-home-mesh.service"]),
              names=(None, "home"))
    mesh = native_mesh.inspect_local(1)["mesh"]
    assert mesh["unit"] == UNIT and mesh["active"] and mesh["snapshot"] == {"ok": True}


def test_several_enabled_meshes_without_a_running_one_are_not_chosen_between(tmp_path, monkeypatch):
    installed(tmp_path, monkeypatch, Units(enabled=[UNIT, "sparkring-home-mesh.service"]), names=(None, "home"))
    with pytest.raises(ValueError, match="Several mesh services are enabled"):
        native_mesh.inspect_local(1)


def test_a_disabled_stopped_mesh_is_not_reported(tmp_path, monkeypatch):
    installed(tmp_path, monkeypatch, Units())
    assert native_mesh.inspect_local(1) == {"mesh": None}


def test_meshes_stopped_when_a_neighbor_restarted_are_reused():
    value = cluster()
    responses = iter({"mesh": {"reference": REFERENCE, "host_ip": f"192.0.2.{110 + rank}", "interface": "eth0",
                               "unit": UNIT, "active": rank == 3, "snapshot": None}} for rank in range(4))
    site = native_mesh.select(controller.model_site(value, PROFILE), value, PROFILE,
                              invoke=lambda *a, **k: json.dumps(next(responses)))
    assert "native_mesh" not in site and all(row["fabric"] == REFERENCE for row in site["hosts"])
    lock = installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)
    assert "ring-serve" in [p["id"] for p in installer.operation_plan(lock, "up")["phases"]]


class Spark:
    """systemctl, ip and the ring check of one simulated Spark."""

    def __init__(self, active=(), failures=0, stale=()):
        self.active, self.failures, self.stale_ports = set(active), failures, list(stale)
        self.commands, self.checks = [], 0

    def call(self, argv, **kwargs):
        self.commands.append(argv)
        if argv[:2] == ["systemctl", "list-units"]:
            return SimpleNamespace(stdout="".join(f"{unit} loaded active running mesh\n" for unit in sorted(self.active)))
        if argv[:4] == ["ip", "-j", "-4", "addr"]:
            info = [{"local": "198.51.100.7", "prefixlen": 31, "noprefixroute": True}]
            return SimpleNamespace(stdout=json.dumps([{"addr_info": info}]))
        return SimpleNamespace(stdout="")

    def check(self, *args):
        self.checks += 1
        if self.checks <= self.failures:
            raise ValueError("Missing mesh network objects: ['route:rank1-to-rank3']")

    def stale(self, reference, rank):
        return self.stale_ports

    def serve(self, clock=lambda: 0.0):
        return native_mesh.serve_ring(REFERENCE, 1, ["mlx5_0"], 3, "192.0.2.111", call=self.call, check=self.check,
                                      stale=self.stale, sleep=lambda seconds: None, clock=clock)

    def changes(self):
        return [argv for argv in self.commands if argv[:2] != ["systemctl", "list-units"] and "show" not in argv]


def test_a_mesh_stopped_by_a_reboot_is_enabled_started_and_awaited():
    spark = Spark(failures=2)
    assert spark.serve()["action"] == "started"
    assert spark.changes() == [["systemctl", "enable", "--now", UNIT]]
    assert spark.checks == 3


def test_a_healthy_mesh_is_only_enabled():
    spark = Spark(active=[UNIT])
    assert spark.serve()["action"] == "checked"
    assert spark.changes() == [["systemctl", "enable", UNIT]]


def test_a_running_mesh_with_missing_routes_restarts():
    spark = Spark(active=[UNIT], failures=1)
    assert spark.serve()["action"] == "restarted"
    assert spark.changes() == [["systemctl", "restart", UNIT], ["systemctl", "enable", "--now", UNIT]]


def test_a_stale_gid_slot_is_rebuilt_with_the_mesh_stopped():
    spark = Spark(active=[UNIT], stale=[("enp1s0f1np1", "198.51.100.7")])
    result = spark.serve()
    assert result == {"ok": True, "unit": UNIT, "action": "repaired", "repaired": ["enp1s0f1np1"]}
    assert spark.changes() == [
        ["systemctl", "stop", UNIT],
        ["ip", "addr", "del", "198.51.100.7/31", "dev", "enp1s0f1np1"],
        ["ip", "addr", "add", "198.51.100.7/31", "noprefixroute", "dev", "enp1s0f1np1"],
        ["systemctl", "enable", "--now", UNIT]]


def test_another_active_mesh_is_left_for_the_ring_check_to_report():
    spark = Spark(active=["sparkring-home-mesh.service"], stale=[("enp1s0f1np1", "198.51.100.7")])
    assert spark.serve()["action"] == "none"
    assert spark.changes() == []


def test_a_ring_that_never_becomes_ready_fails_with_the_last_check():
    times = iter([0.0, 10.0, 250.0])
    spark = Spark(failures=99)
    with pytest.raises(ValueError, match="still fails after 240 s: Missing mesh network objects"):
        spark.serve(clock=lambda: next(times))


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
        assert ["nmcli", "connection", "modify", "uuid-p0", "ipv6.method", "link-local",
                "ipv6.addr-gen-mode", "eui64"] in state["calls"]
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
