"""Node status of the ConnectX hairpin setting, with fake facts, systemd and root; no host access."""
import copy
import subprocess
from types import SimpleNamespace

import pytest

from runtime.common import qwen_mesh
from runtime.host import hairpin, node, topology
from runtime.host.test_appliance import nodes
from scripts import hairpin_setting

PAIR_KEYS = {"schema", "observed_at", "hostname", "node_id", "boot_id", "identity_errors", "source",
             "hardware_qualified", "state", "next_action", "rank", "size", "cluster_id", "containers", "model_ready"}
DROP_IN = "/run/systemd/generator/{}.d/" + node.MESH_START_CHECK


class Host:
    """Answers lldpctl, systemctl, sysctl and iptables queries; refuses devlink and ethtool."""

    def __init__(self, drop_ins=None, *, armed=False):
        self.calls, self.drop_ins, self.armed = [], dict(drop_ins or {}), armed

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[0] in ("devlink", "ethtool"):
            pytest.fail("status read the driver again instead of using the collected facts")
        stdout = ""
        if argv[:2] == ["sysctl", "-n"]:
            stdout = "1\n"
        elif argv[:2] == ["systemctl", "show"] and "DropInPaths" in argv:
            stdout = " ".join(self.drop_ins.get(argv[-1], [])) + "\n"
        elif argv[0] == "lldpctl":
            stdout = '{"lldp": {"interface": []}}'
        elif argv[:2] == ["systemctl", "is-enabled"]:
            stdout = "enabled\n" if self.armed else "disabled\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    def drop_in_queries(self):
        return [call[-1] for call in self.calls if "DropInPaths" in call]


def ring(tmp_path, size=4, rank=0):
    """A configured Spark of a ring whose facts pass observe and verify_persistence."""
    found = nodes(size)
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, rank)
    facts = copy.deepcopy(found[rank]["facts"])
    facts["routes"] += [{"dst": r["destination"], "dev": r["dev"], "gateway": r["via"]} for r in config["routes"]]
    for row in facts["rdma"]:
        row["devlink"]["reload"] = {"driver_reinit": 1, "failed": False}
    node.save(tmp_path, "/etc/sparkring/fabric.json", config)
    return config, facts


def mesh_unit(root, name, *, masked=False):
    directory = root / "etc/systemd/system"
    directory.mkdir(parents=True, exist_ok=True)
    if masked:
        try:
            (directory / name).symlink_to("/dev/null")
        except OSError:
            pytest.skip("this platform cannot create symbolic links")
    else:
        (directory / name).write_text("[Service]\n")


def approve(root, config, facts):
    """Write this Spark's identity and a valid approval record for the record's four functions."""
    node.save(root, "/etc/sparkring/node.json", {"node_id": config["node_id"]})
    pci = {row["device"]: row["pci_address"] for row in facts["rdma"]}
    node.save(root, hairpin.APPROVAL, {
        "schema": "sparkring-hairpin-approval/v1", "node_id": config["node_id"],
        "parameters": dict(hairpin_setting.PARAMETERS), "hw_tc_offload": True,
        "functions": [{"role": p["role"], "rdma_device": p["rdma_device"], "pci_address": pci[p["rdma_device"]],
                       "netdev": p["netdev"], "mac": p["mac"]} for p in config["interfaces"]]})


def function(facts, port):
    rdma = next(row for row in facts["rdma"] if row["device"] == port["rdma_device"])
    nic = next(row for row in facts["interfaces"] if row["name"] == port["netdev"])
    return rdma["devlink"], nic


@pytest.fixture
def report(monkeypatch):
    """Replace hairpin.status with a recorded sparkring-hairpin-status/v1 document without warnings."""
    state = SimpleNamespace(document={"schema": "sparkring-hairpin-status/v1", "warnings": []}, calls=[])

    def status(*, facts=None, root="/", run=subprocess.run):
        state.calls.append(facts)
        return copy.deepcopy(state.document)

    monkeypatch.setattr(hairpin, "status", status)
    return state


def test_inspect_carries_the_hairpin_status_of_the_collected_facts(tmp_path, monkeypatch):
    found = nodes()
    facts, management = found[0]["facts"], found[0]["facts"]["management"]
    node.save(tmp_path, "/etc/sparkring/node.json", {"node_id": found[0]["node_id"]})
    real, seen = hairpin.status, []

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(hairpin, "status", spy)
    host = Host()
    document = node.inspect(0, facts["ssh_target"], management["address"], management["controller_address"],
                            root=tmp_path, collect=lambda _: facts, run=host)
    assert len(seen) == 1 and seen[0]["facts"] is facts and seen[0]["run"] is host
    assert document["hairpin"]["schema"] == "sparkring-hairpin-status/v1"
    assert not any(call[0] in ("devlink", "ethtool") for call in host.calls)


def test_four_spark_snapshot_approved_armed_and_in_effect_has_no_warnings(tmp_path):
    config, facts = ring(tmp_path)
    approve(tmp_path, config, facts)
    mesh_unit(tmp_path, "sparkring-mesh.service")
    host = Host({"sparkring-mesh.service": [DROP_IN.format("sparkring-mesh.service")]}, armed=True)
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=host)
    assert result["state"] == "network-configured" and "warnings" not in result and "error" not in result
    assert ["systemctl", "is-enabled", hairpin.UNIT] in host.calls


def test_function_at_1024_needs_attention_before_the_mesh_check(tmp_path, report, monkeypatch):
    config, facts = ring(tmp_path)
    config.update(ownership="observed", routes=[], forwarding=[],
                  native_mesh={"reference": "x", "hcas": [], "host_ip": "198.18.1.1"})
    node.save(tmp_path, "/etc/sparkring/fabric.json", config)
    monkeypatch.setattr(qwen_mesh, "check", lambda *a, **k: pytest.fail("mesh checked before the hairpin setting"))
    port = config["interfaces"][1]
    devlink, _ = function(facts, port)
    devlink["parameters"]["hairpin_queue_size"]["value"] = 1024
    devlink["reload"] = {"driver_reinit": 0, "failed": False}
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host())
    assert report.calls == [facts] and result["state"] == "needs-attention"
    assert result["next_action"] == "on Node A: sudo sparkring hairpin"
    assert result["error"] == (f"ConnectX hairpin setting not in effect on {port['netdev']}: "
                               "hairpin_queue_size 1024, required 8192")


@pytest.mark.parametrize("change, detail", [
    ("pending", "hairpin_queue_size 8192 set, applied only by a driver restart (none since boot)"),
    ("failed", "last driver restart failed"),
    ("offload", "hw-tc-offload off"),
    ("statistics", "driver restarts since boot unavailable, restart failure flag unavailable"),
    ("queues", "hairpin_num_queues 2, required 4"),
])
def test_every_function_that_is_not_in_effect_is_named(tmp_path, report, change, detail):
    config, facts = ring(tmp_path)
    ports = config["interfaces"][2:]
    for port in ports:
        devlink, nic = function(facts, port)
        if change == "pending":
            devlink["reload"]["driver_reinit"] = 0
        elif change == "failed":
            devlink["reload"]["failed"] = True
        elif change == "offload":
            nic.update(hw_tc_offload=False, hw_tc_offload_fixed=False)
        elif change == "statistics":
            del devlink["reload"]
        else:
            devlink["parameters"]["hairpin_num_queues"]["value"] = 2
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host())
    assert result["state"] == "needs-attention" and result["next_action"] == node.HAIRPIN_REMEDY
    # Both functions share the shortfall, so it is named once.
    assert result["error"] == ("ConnectX hairpin setting not in effect on "
                               + ", ".join(p["netdev"] for p in ports) + ": " + detail)
    rows = node.hairpin_rows(config, facts)
    assert [row["state"] != hairpin_setting.IN_EFFECT for row in rows] == [False, False, True, True]


@pytest.mark.parametrize("approved, armed", [(False, False), (True, False), (False, True)])
def test_setting_in_effect_but_not_applied_at_boot_warns(tmp_path, approved, armed):
    config, facts = ring(tmp_path)
    if approved:
        approve(tmp_path, config, facts)
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host(armed=armed))
    assert result["state"] == "network-configured"
    assert result["warnings"] == ["ConnectX hairpin setting is in effect but not applied at boot; "
                                  "on Node A: sudo sparkring hairpin"]


def test_unarmed_function_at_1024_is_an_error_not_an_arming_warning(tmp_path):
    config, facts = ring(tmp_path)
    function(facts, config["interfaces"][0])[0]["parameters"]["hairpin_queue_size"]["value"] = 1024
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host())
    assert result["state"] == "needs-attention" and "warnings" not in result


def test_suspended_boot_restarts_warn_with_the_boot(tmp_path):
    config, facts = ring(tmp_path)
    approve(tmp_path, config, facts)
    port = config["interfaces"][0]
    pci = next(row["pci_address"] for row in facts["rdma"] if row["device"] == port["rdma_device"])
    node.save(tmp_path, hairpin.ATTEMPTS, {"schema": hairpin.ATTEMPTS_SCHEMA, "records": [
        {"boot_id": "b0", "mode": "boot", "pci_address": pci, "netdev": port["netdev"], "state": "failed",
         "class": "restart", "error": "reload timed out", "time": 1}]})
    function(facts, port)[0]["parameters"]["hairpin_queue_size"]["value"] = 1024
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host(armed=True))
    assert result["state"] == "needs-attention"
    assert result["warnings"] == [f"boot restarts suspended since {port['netdev']} failed to restart at "
                                  "1970-01-01 00:00:01 UTC (reload timed out); on Node A: sudo sparkring hairpin "
                                  "retries it live"]
    # The record belongs to an earlier boot, so no reboot is needed.
    assert result["next_action"] == node.HAIRPIN_REMEDY


def test_a_failed_child_facing_restart_of_this_boot_names_the_spark_to_reboot(tmp_path):
    config, facts = ring(tmp_path, rank=1)
    approve(tmp_path, config, facts)
    port = config["interfaces"][0]
    pci = next(row["pci_address"] for row in facts["rdma"] if row["device"] == port["rdma_device"])
    boot = "3b1f0c9e-2b7d-4e3f-8a5b-1c0d9e8f7a6b"
    (tmp_path / "proc/sys/kernel/random").mkdir(parents=True)
    (tmp_path / "proc/sys/kernel/random/boot_id").write_text(boot + "\n")
    # This Spark carries the administration tunnel to a child over the failed function.
    node.save(tmp_path, hairpin.CONTROL, {"schema": "sparkring-control/v1", "head": False,
                                          "head_address": "10.253.255.1",
                                          "peers": [{"id": "child", "netdev": port["netdev"],
                                                     "allowed_ips": ["10.253.255.4/32"]}]})
    node.save(tmp_path, hairpin.ATTEMPTS, {"schema": hairpin.ATTEMPTS_SCHEMA, "records": [
        {"boot_id": boot, "mode": "boot", "pci_address": pci, "netdev": port["netdev"], "state": "failed",
         "class": "restart", "cause": "driver restart failed: busy", "time": 1}]})
    devlink, _ = function(facts, port)
    devlink["parameters"]["hairpin_queue_size"]["value"] = 1024
    devlink["reload"]["driver_reinit"] = 0
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host(armed=True))
    assert result["state"] == "needs-attention"
    assert result["next_action"] == (f"reboot this Spark ({port['netdev']} failed to restart and cut the Sparks "
                                     "behind it off; the next boot restarts no ConnectX function), then on Node A: "
                                     "sudo sparkring hairpin")


def test_mesh_units_without_the_start_check_warn(tmp_path, report):
    _, facts = ring(tmp_path)
    for name in ("sparkring-mesh.service", "sparkring-site-mesh.service", "sparkring-other-mesh.service",
                 "sparkring-mesh-model.service", "sparkring-site-model.service"):
        mesh_unit(tmp_path, name)
    host = Host({"sparkring-mesh.service": [DROP_IN.format("sparkring-mesh.service")],
                 # A drop-in with the right name for another unit does not count.
                 "sparkring-other-mesh.service": [DROP_IN.format("sparkring-mesh.service")]})
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=host)
    assert sorted(host.drop_in_queries()) == ["sparkring-mesh.service", "sparkring-other-mesh.service",
                                              "sparkring-site-mesh.service"]
    assert result["state"] == "network-configured"
    assert result["warnings"] == [
        "mesh unit sparkring-other-mesh.service has no hairpin start check; run sudo systemctl daemon-reload",
        "mesh unit sparkring-site-mesh.service has no hairpin start check; run sudo systemctl daemon-reload"]


def test_unreadable_hairpin_status_is_a_warning_and_the_rule_still_applies(tmp_path, monkeypatch):
    config, facts = ring(tmp_path)

    def broken(**kwargs):
        raise ValueError("attempt journal is not JSON")

    monkeypatch.setattr(hairpin, "status", broken)
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host())
    assert result["state"] == "network-configured"
    assert result["warnings"] == ["ConnectX hairpin approval and boot state unavailable: attempt journal is not JSON"]
    function(facts, config["interfaces"][0])[0]["reload"]["driver_reinit"] = 0
    assert node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host())["state"] == "needs-attention"


def test_pair_snapshot_is_unchanged_and_warns_only_about_a_leftover_approval(tmp_path, monkeypatch):
    _, facts = ring(tmp_path, size=2)
    monkeypatch.setattr(hairpin, "status", lambda **k: pytest.fail("pairs read no hairpin status"))
    mesh_unit(tmp_path, "sparkring-mesh.service")
    for rdma in facts["rdma"]:
        rdma["devlink"]["parameters"]["hairpin_queue_size"]["value"] = 1024
    host = Host()
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=host)
    assert set(result) == PAIR_KEYS and result["state"] == "network-configured"
    assert host.drop_in_queries() == []
    node.save(tmp_path, "/etc/sparkring/hairpin.json", {"schema": "sparkring-hairpin-approval/v1"})
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host())
    assert result["state"] == "network-configured"
    assert result["warnings"] == ["hairpin approval on a Spark that is not in a four-Spark ring: "
                                  "sudo sparkring node hairpin revoke"]


def test_other_failures_keep_the_generic_next_action(tmp_path, report):
    config, facts = ring(tmp_path)
    facts["routes"] = [r for r in facts["routes"] if r.get("dst") != config["routes"][0]["destination"]]
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=Host())
    assert result["state"] == "needs-attention" and result["next_action"] == "sparkring status --refresh"
    assert "Approved fabric route is missing" in result["error"]


def test_masked_mesh_unit_is_not_checked(tmp_path, report):
    _, facts = ring(tmp_path)
    mesh_unit(tmp_path, "sparkring-masked-mesh.service", masked=True)
    host = Host()
    result = node.snapshot(root=tmp_path, collect=lambda _: facts, run=host)
    assert host.drop_in_queries() == [] and "warnings" not in result
