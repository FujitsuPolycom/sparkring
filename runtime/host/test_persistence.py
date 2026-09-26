"""Lifecycle boundary regressions for root services and control-preserving setup."""
import copy
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from types import SimpleNamespace

import pytest

from runtime.host import controller, topology
from runtime.host.test_appliance import configured, nodes
from scripts.deploy_network_run import _guard, _network_local


def test_control_network_modifies_original_uuid_without_bouncing_ipv6():
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"], reset=True, fabric_cidr="198.18.32.0/21", preserve_control=True)
    for host in plan["network"]["hosts"]:
        commands = [c["argv"] for c in host["apply"]]
        assert any("modify" in c for c in commands)
        assert any("reapply" in c for c in commands)
        assert all("up" not in c and "down" not in c and "add" not in c for c in commands)
        assert all("ipv6.addr-gen-mode" not in c and "connection.uuid" not in c for c in commands)
        assert host["rollback"]


def network_runner(host, argv, timeout):
    is_check = "'check'" in argv[-1]
    return {"returncode": 0, "stderr": "", "stdout": '{"checked":true}' if is_check else '{"complete":true}'}


def test_reset_address_choice_survives_the_hairpin_step(tmp_path):
    found = nodes()
    found[0]["facts"]["rdma"][0]["devlink"]["parameters"]["hairpin_num_queues"]["value"] = 0
    initial = topology.build_spec(found, found[0]["node_id"], reset=True, fabric_cidr="198.18.40.0/21", preserve_control=True)
    # The first pass leaves addressing unconverged; the second converges it
    # while rank 0 still lacks the hairpin setting.
    ready = configured(initial)
    applied = copy.deepcopy(ready)
    applied[0]["facts"]["rdma"][0]["devlink"]["parameters"]["hairpin_num_queues"]["value"] = 4
    observations = iter((found, ready))
    reviews, received = [], []

    def ensure(plan, *, approved, inspect, rebuild, **options):
        received.append(plan)
        # The hairpin step re-inspects the ring and rebuilds the plan with the setup's choices.
        return rebuild(applied)

    result = controller.apply(initial, tmp_path, run=network_runner, invoke=lambda *a, **k: "{}",
                              inspect_nodes=lambda _: next(observations), approved=True, review=reviews.append,
                              ensure=ensure)
    assert len(reviews) == 1 and len(received) == 1
    assert received[0]["network"]["hosts"][0]["driver_action"] == "apply"
    assert result["spec"]["hosts"][0]["data_interfaces"][0]["address"] == "198.18.40.1/24"
    assert result["reset_requested"]


def test_hairpin_step_follows_converged_addressing_and_verification_uses_its_plan(tmp_path, monkeypatch):
    found = nodes()
    plan = topology.build_spec(found, found[0]["node_id"])
    returned, verified = {}, []

    def ensure(value, *, approved, record, **options):
        assert all(host["action"] == "none" for host in value["network"]["hosts"]) and approved is True
        record.update(state="kept")
        returned["plan"] = copy.deepcopy(value)
        return returned["plan"]
    monkeypatch.setattr(controller.deploy_network, "verify_network", lambda spec, inventory, **k: verified.append(spec))
    controller.apply(plan, tmp_path, run=network_runner, invoke=lambda *a, **k: "{}", inspect_nodes=lambda _: found,
                     approved=True, ensure=ensure)
    assert len(verified) == 1 and verified[0] is returned["plan"]["spec"]
    steps = json.loads((tmp_path / "setup.json").read_text())["steps"]
    assert {"hairpin": "kept"} in steps
    assert steps.index({"hairpin": "kept"}) == 1


def test_addressing_stops_after_four_passes(tmp_path):
    found = nodes(blank=True)
    plan = topology.build_spec(found, found[0]["node_id"])
    inspected = []
    with pytest.raises(ValueError, match="did not converge"):
        controller.apply(plan, tmp_path, run=network_runner, invoke=lambda *a, **k: "{}",
                         inspect_nodes=lambda _: inspected.append(1) or found, approved=True,
                         ensure=lambda *a, **k: pytest.fail("hairpin step before addressing converged"))
    assert len(inspected) == controller.ADDRESSING_PASSES == 4


def test_gpu_compute_blocks_fabric_mutation_before_journal_creation(tmp_path):
    found = nodes(2, blank=True)
    plan = topology.build_spec(found, found[0]["node_id"])
    host = plan["spec"]["hosts"][0]
    host["backup_dir"] = str(tmp_path / "backup")
    facts = found[0]["facts"]
    payload = {"host": host, "request": {}, "guard": _guard(facts, host), "commands": [{}], "require_idle_gpu": True}
    with pytest.raises(ValueError, match="GPU compute"):
        _network_local(payload, "check", collect=lambda _: facts,
                       run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="1234\n", stderr=""))
    assert not (tmp_path / "backup").exists()


def test_saved_model_api_address_uses_node_a_ethernet():
    from runtime.common import installer
    found = nodes(2)
    cluster = {"name": "home", "api_address": "192.0.2.55", "plan": topology.build_spec(found, found[0]["node_id"])}
    raw = controller.model_site(cluster, "qwen")
    lock = installer.make_lock(installer.DEFAULTS["qwen38", 2], raw, "a" * 40, "b" * 64)
    assert installer.connection(lock)["api_url"].startswith("http://192.0.2.55:")


def test_administrative_services_do_not_start_models_or_rewrite_ssh_policy():
    root = Path(__file__).resolve().parents[2] / "packaging/debian"
    postinst = (root / "postinst").read_text()
    assert "node initialize" in postinst and "sparkring up" not in postinst
    # Reinstalling restores the units prerm recorded; a recorded fabric unit is
    # enabled for the next boot but never started by the package, so package
    # installation does not change the data network.
    fabric_rules = [line for line in postinst.splitlines() if "sparkring-fabric" in line]
    assert fabric_rules and all("systemctl enable \"$unit\"" in line for line in fabric_rules)
    assert not any(word in line for line in fabric_rules for word in ("--now", "start", "restart"))
    assert "/var/lib/sparkring/package-enabled-units" in postinst
    assert "/var/lib/sparkring/package-enabled-units" in (root / "prerm").read_text()
    assert "sshd_config" not in postinst
    for unit in root.glob("*.service"):
        assert "installer_host" not in unit.read_text()
    assert "weights" in (root / "postrm").read_text()


def test_plan_only_named_model_does_not_construct_runner(tmp_path, monkeypatch, capsys):
    from runtime.common import installer
    from scripts import installer_runner
    monkeypatch.setattr(controller, "STATE", tmp_path)
    (tmp_path / "active.json").write_text(json.dumps({"path": str(tmp_path / "model")}))
    monkeypatch.setattr(installer, "apply", lambda *a, **kw: {"profile": "fixture", "hosts": ["a", "b"], "phases": ["read"]})
    monkeypatch.setattr(installer_runner, "Runner", lambda *a: pytest.fail("Plan attempted SSH runner"))
    assert controller.lifecycle(["up", "--plan"]) == 0
    assert "--execute" in capsys.readouterr().out


def test_status_text_names_the_sparks_that_need_attention(tmp_path, monkeypatch, capsys):
    from runtime.common import installer
    monkeypatch.setattr(controller, "STATE", tmp_path)
    found = nodes()
    installer.write(tmp_path / "cluster.json", {"plan": topology.build_spec(found, found[0]["node_id"])})
    monkeypatch.setattr(controller.node, "status", lambda: {"state": "network-configured", "next_action": "sparkring models"})
    worker = {"state": "needs-attention", "hostname": "spark2", "next_action": "on Node A: sudo sparkring hairpin",
              "error": "ConnectX hairpin setting not in effect on enp1s0f0np0: hairpin_queue_size 1024, required 8192",
              "warnings": ["boot restarts suspended after a failed restart in boot b1"]}
    healthy = {"state": "network-configured", "hostname": "spark", "next_action": "sparkring models"}
    def ssh(host, argv):
        if host == "root@192.0.2.11":
            raise RuntimeError("root@192.0.2.11: ssh: connect to host port 22: Connection timed out")
        return json.dumps(worker if host == "root@192.0.2.12" else healthy)
    monkeypatch.setattr(controller.discovery, "ssh", ssh)
    assert controller.lifecycle(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "network-configured | next: sparkring models"
    at = lines.index("  rank 2 (spark2) root@192.0.2.12: needs-attention — " + worker["error"])
    assert "  rank 0 (spark) root@192.0.2.10: network-configured" in lines
    assert lines[at + 1:at + 3] == ["    next: on Node A: sudo sparkring hairpin",
                                    "    warning: boot restarts suspended after a failed restart in boot b1"]
    # A Spark that cannot be read has no hostname; its SSH target names it once.
    assert ("  rank 1 (root@192.0.2.11): unreachable — root@192.0.2.11: ssh: connect to host port 22: Connection "
            "timed out") in lines
    assert "Sparks that need attention: rank 1 (root@192.0.2.11), rank 2 (spark2)" in lines
    assert not any(line == "    next: sparkring models" for line in lines)
    assert controller.lifecycle(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "network-configured"


def test_up_refuses_to_start_while_a_spark_lacks_the_hairpin_setting(tmp_path, monkeypatch, capsys):
    from runtime.common import installer
    from runtime.host import retained_source
    from runtime.host.test_hairpin_ring import document, ring_plan
    from scripts import hairpin_setting
    monkeypatch.setattr(controller, "STATE", tmp_path)
    plan = ring_plan()
    installer.write(tmp_path / "cluster.json", {"plan": plan})
    installer.write(tmp_path / "active.json", {"path": str(tmp_path / "model")})
    monkeypatch.setattr(installer, "apply", lambda *a, **kw: {"profile": "fixture", "hosts": ["a"], "phases": ["read"]})
    monkeypatch.setattr(retained_source, "apply", lambda *a, **k: pytest.fail("a model action ran"))
    statuses = {host["host"]: document(plan, rank) for rank, host in enumerate(plan["spec"]["hosts"])}
    statuses["root@192.0.2.12"] = document(plan, 2, [hairpin_setting.DEFAULT] * 2 + [hairpin_setting.IN_EFFECT] * 2)
    statuses["root@192.0.2.13"] = document(plan, 3, [hairpin_setting.FAILED] + [hairpin_setting.IN_EFFECT] * 3)
    monkeypatch.setattr(controller.discovery, "ssh", lambda host, argv: json.dumps(statuses[host]))
    first, second = (row["netdev"] for row in statuses["root@192.0.2.12"]["functions"][:2])
    failed = statuses["root@192.0.2.13"]["functions"][0]["netdev"]
    # One line per Spark; functions that share a shortfall are named together.
    expected = [f"rank 2 (spark2): the ConnectX hairpin setting is not in effect on 2 of 4 functions: {first}, "
                f"{second}: hairpin_queue_size 1024, required 8192.",
                f"  rank 3 (spark3): the ConnectX hairpin setting is not in effect on 1 of 4 functions: {failed}: "
                "last driver restart failed.",
                "  On Node A, sudo sparkring hairpin applies it after asking."]
    with pytest.raises(ValueError) as caught:
        controller.lifecycle(["up", "--execute"])
    assert str(caught.value) == "\n".join(expected)
    assert controller.lifecycle(["up", "--plan"]) == 0
    assert "Warning: " + "\n".join(expected) in capsys.readouterr().out


def test_adoption_records_facts_without_running_network_commands(tmp_path, monkeypatch):
    from runtime.host import node
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, 0)
    config.update(ownership="observed", routes=[], forwarding=[])
    node.save(tmp_path, "/etc/sparkring/node.json", {"node_id": config["node_id"]})
    monkeypatch.setattr(node, "call", lambda *a, **k: pytest.fail("Adoption changed host services/network"))
    assert node.adopt(config, root=tmp_path, collect=lambda _: found[0]["facts"])["network_changed"] is False
    with pytest.raises(ValueError, match="existing service"):
        node.restore(config)


PACKAGING = Path(__file__).resolve().parents[2] / "packaging/debian"


def unit_settings(text):
    """Values of each (section, directive) of a systemd unit; repeated directives accumulate."""
    settings, section = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        key, _, value = line.partition("=")
        settings.setdefault((section, key.strip()), []).extend(value.split())
    return settings


def test_hairpin_service_runs_before_network_management_and_every_sparkring_unit():
    from runtime.host import hairpin
    unit = unit_settings((PACKAGING / "sparkring-hairpin.service").read_text())
    assert unit[("Unit", "DefaultDependencies")] == ["no"]
    assert unit[("Unit", "Wants")] == ["network-pre.target"]
    assert {"network-pre.target", "NetworkManager.service", "sparkring-control.service",
            "sparkring-fabric.service", "sparkring-mesh.service"} <= set(unit[("Unit", "Before")])
    assert "systemd-udev-trigger.service" in unit[("Unit", "After")]
    assert unit[("Unit", "ConditionPathExists")] == ["/etc/sparkring/hairpin.json"]
    assert unit[("Unit", "ConditionKernelCommandLine")] == ["!sparkring.hairpin=off"]
    # Ordering only: NetworkManager starts after a failed or timed-out run too.
    assert ("Unit", "Requires") not in unit and ("Unit", "BindsTo") not in unit
    assert unit[("Service", "Type")] == ["oneshot"]
    assert unit[("Service", "RemainAfterExit")] == ["yes"]
    assert unit[("Service", "ExecStart")] == ["/usr/bin/sparkring", "node", "hairpin", "apply"]
    assert ("Service", "Restart") not in unit
    assert unit[("Service", "RuntimeDirectory")] == ["sparkring-hairpin"]
    assert unit[("Service", "RuntimeDirectoryPreserve")] == ["yes"]
    assert unit[("Service", "TimeoutStopSec")] == ["60"]
    # A stop signals the run only, never a devlink child in the middle of a reload.
    assert unit[("Service", "KillMode")] == ["mixed"]
    assert unit[("Install", "WantedBy")] == ["multi-user.target"]
    # systemd never stops a run within its budget: the start timeout covers the
    # wait for the functions, the budget and 35 s for the steps after it, and
    # every restart that starts fits the budget's reserve. A stop lets the
    # current function finish its reload and return.
    start = int(unit[("Service", "TimeoutStartSec")][0])
    assert start >= hairpin.PRESENT + hairpin.BUDGET["live"] + 35
    assert start >= hairpin.PRESENT + hairpin.UDEV + 5 + hairpin.BUDGET["boot"]
    assert hairpin.RESERVE["boot"] == hairpin.RELOAD + hairpin.RETURN
    assert hairpin.RESERVE["live"] == (hairpin.RELOAD + hairpin.RETURN + hairpin.ADDRESSES + hairpin.ADDRESSES_AFTER_UP
                                       + hairpin.TUNNEL)
    assert int(unit[("Service", "TimeoutStopSec")][0]) >= hairpin.RELOAD + hairpin.RETURN
    # The ring procedure waits for a run through its start and stop timeouts.
    from runtime.host import hairpin_ring
    assert hairpin_ring.REPORT > start + int(unit[("Service", "TimeoutStopSec")][0])


def test_fabric_and_control_units_start_after_the_hairpin_service():
    for name in ("sparkring-fabric.service", "sparkring-control.service"):
        assert "sparkring-hairpin.service" in unit_settings((PACKAGING / name).read_text())[("Unit", "After")]


def test_administration_access_starts_whenever_the_administration_interface_exists():
    # sparkring-control.service fails while one administration link fails, but
    # still creates sr-control over the healthy links; SSH access must follow.
    unit = unit_settings((PACKAGING / "sparkring-access.service").read_text())
    assert ("Unit", "Requires") not in unit and ("Unit", "BindsTo") not in unit
    assert unit[("Unit", "Wants")] == ["sparkring-control.service"]
    assert unit[("Unit", "After")] == ["sparkring-control.service"]
    assert unit[("Unit", "ConditionPathExists")] == ["/etc/sparkring/sshd_config", "/sys/class/net/sr-control"]
    refresh = unit_settings((PACKAGING / "sparkring-control-refresh.service").read_text())
    assert "sparkring-access.service" in refresh[("Service", "ExecStartPost")]


def test_package_enables_the_hairpin_service_only_for_the_next_boot():
    postinst = (PACKAGING / "postinst").read_text()
    rules = [line for line in postinst.splitlines() if "sparkring-hairpin" in line]
    assert rules and all("systemctl enable \"$unit\"" in line for line in rules)
    assert not any(word in line for line in rules for word in ("--now", "start", "restart"))
    assert "sparkring-hairpin" not in postinst.replace(rules[0], "")
    units = next(line for line in (PACKAGING / "prerm").read_text().splitlines() if line.startswith("UNITS="))
    assert "sparkring-hairpin.service" in units.split("=", 1)[1].strip('"').split()


@pytest.mark.skipif(os.name != "posix", reason="file modes need POSIX")
def test_package_installs_the_mesh_check_generator_executable(tmp_path, monkeypatch):
    import tarfile
    from scripts import build_deb
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as tar:
        tar.add(PACKAGING, arcname="packaging/debian")
    placed = {}

    def run(argv, **kwargs):
        if argv[:2] == ["git", "archive"]:
            shutil.copyfile(source, argv[argv.index("-o") + 1])
        elif "bundle" in argv:
            Path(argv[argv.index("create") + 1]).write_bytes(b"bundle")
        elif argv[0] == "dpkg-deb":
            package = Path(argv[-2])
            for relative in ("usr/lib/systemd/system-generators/sparkring-hairpin-mesh-check",
                             "usr/lib/systemd/system/sparkring-hairpin.service"):
                path = package / relative
                placed[relative] = (stat.S_IMODE(path.stat().st_mode), path.read_bytes())
            Path(argv[-1]).write_bytes(b"deb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_deb.distribution, "identity", lambda root: "a" * 40)
    monkeypatch.setattr(build_deb, "subprocess", SimpleNamespace(run=run, check_output=lambda *a, **k: "1790000000\n"))
    build_deb.build(tmp_path, tmp_path / "dist", version="1.0")
    mode, content = placed["usr/lib/systemd/system-generators/sparkring-hairpin-mesh-check"]
    assert mode == 0o755
    assert content == (PACKAGING / "sparkring-hairpin-mesh-check").read_bytes()
    assert placed["usr/lib/systemd/system/sparkring-hairpin.service"][0] == 0o644


def run_generator(tmp_path, names, *, masked=()):
    units = tmp_path / "units"
    units.mkdir()
    for name in names:
        (units / name).write_text("[Service]\nExecStart=/bin/true\n")
    for name in masked:
        (units / name).symlink_to("/dev/null")
    # Checkouts may carry CRLF line endings; the packaged file is LF.
    script = tmp_path / "generator"
    script.write_bytes((PACKAGING / "sparkring-hairpin-mesh-check").read_bytes().replace(b"\r\n", b"\n"))
    output = tmp_path / "normal"
    output.mkdir()
    subprocess.run(["sh", str(script), str(output), str(tmp_path / "early"), str(tmp_path / "late")], check=True,
                   env={**os.environ, "SPARKRING_UNIT_DIRECTORY": str(units)})
    return output


@pytest.mark.skipif(os.name != "posix" or shutil.which("sh") is None, reason="needs a POSIX shell")
def test_generator_adds_the_start_check_to_mesh_units_only(tmp_path):
    output = run_generator(tmp_path, ["sparkring-mesh.service", "sparkring-x-mesh.service", "sparkring-mesh-model.service",
                                      "sparkring-x-model.service", "sparkring-agent.service"],
                           masked=["sparkring-masked-mesh.service"])
    created = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    assert created == ["sparkring-mesh.service.d/10-sparkring-hairpin.conf",
                       "sparkring-x-mesh.service.d/10-sparkring-hairpin.conf"]
    dropin = unit_settings((output / created[0]).read_text())
    assert dropin[("Unit", "After")] == ["sparkring-hairpin.service"]
    assert dropin[("Service", "ExecStartPre")] == ["/usr/bin/sparkring", "node", "hairpin", "require", "--unit", "%n"]


@pytest.mark.skipif(os.name != "posix" or shutil.which("sh") is None, reason="needs a POSIX shell")
def test_generator_writes_nothing_without_mesh_units(tmp_path):
    output = run_generator(tmp_path, ["sparkring-agent.service", "sparkring-mesh-model.service"])
    assert list(output.iterdir()) == []


def test_mesh_unit_text_carries_no_hairpin_check():
    # The generator adds the check, so the standalone installer's comparison of
    # installed units with unit_text keeps matching.
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "glm53-spark-mtp3-mesh/managed_units.py"
    spec = importlib.util.spec_from_file_location("managed_units_persistence_subject", path)
    units = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(units)
    for deployment in (None, "fixture"):
        code, config = ("/opt/sparkring/managed-mesh", "/etc/sparkring/managed-mesh") if deployment is None else (
            "/opt/sparkring/deployments/fixture", "/etc/sparkring/deployments/fixture")
        rendered = units.unit_text(code, config, "a" * 64, deployment_name=deployment)
        assert rendered and all("hairpin" not in text and "sparkring node" not in text for text in rendered.values())


@pytest.mark.skipif(shutil.which("systemd-analyze") is None or shutil.which("true") is None,
                    reason="needs systemd-analyze")
def test_hairpin_service_passes_systemd_verification(tmp_path):
    # The command is replaced with one that exists on the test host; everything
    # else, including dependencies, ordering and conditions, is verified as shipped.
    unit = tmp_path / "sparkring-hairpin.service"
    unit.write_text((PACKAGING / "sparkring-hairpin.service").read_text().replace(
        "/usr/bin/sparkring node hairpin apply", shutil.which("true")))
    unit.chmod(0o644)
    result = subprocess.run(["systemd-analyze", "verify", "--man=no", "--recursive-errors=no", str(unit)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and not result.stderr.strip() and not result.stdout.strip(), result.stderr
