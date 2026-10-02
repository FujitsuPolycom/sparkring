"""Re-forming Sparks of other SparkRing clusters: detection, plan, retirement on a fake file system, setup's step."""
import argparse
import copy
import json
import os
from pathlib import Path

import pytest

from runtime.host import cabling, reform, single_uplink, survey
from runtime.host.test_survey import LAN, Transport, captured, link_local_neighbors, observation

STAMP = "20261002T143000Z-dgx4-2"
PORTS = {"enp1s0f0np0": "enp1s0f1np1", "enP2p1s0f0np0": "enP2p1s0f1np1",
         "enp1s0f1np1": "enp1s0f0np0", "enP2p1s0f1np1": "enP2p1s0f0np0"}


def swapped(sparks, name):
    """The captured Sparks after the two cables of ``name`` swap ports."""
    sparks = copy.deepcopy(sparks)
    macs = {row["ifname"]: row["address"] for row in sparks[name]["addresses"]}
    by_mac = {macs[a]: macs[b] for a, b in PORTS.items()}
    for entry in sparks[name]["lldp"]["lldp"]["interface"]:
        local = next(iter(entry))
        if next(iter(entry[local]["chassis"])) != name:
            entry[PORTS[local]] = entry.pop(local)
    for other, spark in sparks.items():
        if other == name:
            continue
        for entry in spark["lldp"]["lldp"]["interface"]:
            row = next(iter(entry.values()))
            if row["port"]["id"]["value"] in by_mac:
                row["port"]["id"]["value"] = by_mac[row["port"]["id"]["value"]]
                row["port"]["descr"] = PORTS[row["port"]["descr"]]
    return sparks


def fabric_ipv4(spark):
    return {row["ifname"]: [f"{a['local']}/{a['prefixlen']}" for a in row["addr_info"] if a["family"] == "inet"]
            for row in spark["addresses"] if row["ifname"] != "enP7s7"}


def state(spark, *, cluster=None, nodes=(), address, rank, active=None):
    """The ``prior_state`` document of one captured Spark of a former pair."""
    ipv4 = fabric_ipv4(spark)
    recorded = [{"netdev": n, "address": ipv4[n][0]} for n in ("enp1s0f0np0", "enP2p1s0f0np0")]
    units = {name: {"enabled": "disabled", "active": "inactive"} for name in ("sparkring-recover.timer",)}
    if cluster:
        units["sparkring-recover.timer"] = {"enabled": "enabled", "active": "active"}
    return {"root": True, "unreadable": [], "control_interface": True, "files": {}, "mesh": [], "containers": [],
            "controller": {"name": cluster, "nodes": list(nodes), "entries": ["cluster.json"],
                           "active_deployment": active} if cluster else None,
            "control": {"address": address, "head": bool(cluster), "subnet": "10.253.255.0/29"},
            "fabric": {"cluster_id": "c" * 64, "rank": rank, "size": 2, "interfaces": recorded, "routes": 0,
                       "forwarding": 0},
            "units": units, "fabric_ipv4": ipv4}


def surveyed(*, fixed=True, containers=None):
    """A survey of the four captured Sparks from spark-3286; ``fixed`` swaps spark-aa42's cables first."""
    sparks = swapped(captured(), "spark-aa42") if fixed else captured()
    near = ("spark-aa42", 1, "spark-931e", 0) if fixed else ("spark-aa42", 0, "spark-931e", 0)
    documents = {
        "local": observation(sparks["spark-3286"], root=True),
        LAN["spark-0a0f"]: observation(sparks["spark-0a0f"], root=True),
        LAN["spark-aa42"]: observation(sparks["spark-aa42"], root=True,
                                       neighbors=link_local_neighbors(sparks, *near)),
        LAN["spark-931e"]: observation(sparks["spark-931e"], root=True,
                                       neighbors=link_local_neighbors(sparks, near[2], near[3], near[0], near[1])
                                       + link_local_neighbors(sparks, "spark-931e", 1, "spark-0a0f", 0)),
    }
    documents["local"]["state"] = state(sparks["spark-3286"], cluster="sparkring", nodes=["spark-3286", "spark-0a0f"],
                                        address="10.253.255.1", rank=0, active="deployment-b")
    documents[LAN["spark-0a0f"]]["state"] = state(sparks["spark-0a0f"], address="10.253.255.2", rank=1)
    documents[LAN["spark-aa42"]]["state"] = state(sparks["spark-aa42"], cluster="tp2", nodes=["spark-aa42", "spark-931e"],
                                                  address="10.253.255.1", rank=0, active="deployment-a")
    documents[LAN["spark-931e"]]["state"] = state(sparks["spark-931e"], address="10.253.255.2", rank=1)
    if containers:
        documents[LAN["spark-931e"]]["state"]["containers"] = containers
    table = {next(r["address"] for r in s["addresses"] if r["ifname"] == "enP7s7"): LAN[n] for n, s in sparks.items()}
    transport = Transport(documents)
    here = survey.Reach("local", "this Spark")
    here.run = lambda t, argv, data=None, ssh=None: json.dumps(documents["local"])
    found = survey.survey(transport, recorded=(), user="code", say=lambda line: None, state=reform.prior_state,
                          arp=lambda interface: dict(table), sweep=lambda interface: None, here=here)
    return found, transport


def test_record_mismatch_names_the_unrecorded_neighbor():
    record = {"name": "sparkring", "plan": {"nodes": [{"hostname": "spark-3286"}, {"hostname": "spark-0a0f"}]}}
    sparks = captured()
    rows = [row for row in cabling.lldp_rows(sparks["spark-3286"]["lldp"])]
    assert reform.record_mismatch(record, rows, "spark-3286") == (
        "spark-aa42 is cabled to this Spark but not part of its cluster \"sparkring\" (2 Sparks)")
    pair = [row for row in rows if row["hostname"] in ("spark-0a0f", "spark-3286")]
    assert reform.record_mismatch(record, pair, "spark-3286") is None
    ring = {"name": "ring", "plan": {"nodes": [{"hostname": h} for h in ("spark-3286", "spark-0a0f", "a", "b")]}}
    both = pair + [dict(row, netdev="enp1s0f1np1") for row in pair if row["hostname"] == "spark-0a0f"]
    assert "not cabled as a ring" in reform.record_mismatch(ring, both, "spark-3286")
    assert reform.record_mismatch({"targets": ["a", "b"]}, rows, "spark-3286") == (
        "3 Sparks are cabled here, but setup enrolled 2")


def test_record_reason_reads_cluster_record_first(tmp_path):
    (tmp_path / "enrolled.json").write_text(json.dumps({"targets": ["a", "b"]}))
    (tmp_path / "cluster.json").write_text(json.dumps(
        {"name": "sparkring", "plan": {"nodes": [{"hostname": "spark-3286"}, {"hostname": "spark-aa42"}]}}))
    rows = cabling.lldp_rows(captured()["spark-3286"]["lldp"])
    assert single_uplink.record_reason(tmp_path, lldp=lambda: rows, hostname="spark-3286").startswith("spark-0a0f is")
    assert single_uplink.record_reason(tmp_path / "none", lldp=lambda: rows) is None


def test_plan_lists_each_sparks_state_foreign_addresses_and_the_ring():
    found, _ = surveyed()
    diagnosis = cabling.diagnose(survey.records(found), found["head"])
    assert diagnosis["ready"] and diagnosis["order_names"] == ["spark-3286", "spark-0a0f", "spark-931e", "spark-aa42"]
    value = reform.plan(found, diagnosis, name="dgx4-2", reason="spark-aa42 is cabled to this Spark",
                        now=lambda: (2026, 10, 2, 14, 30, 0, 4, 275, 0))
    assert value["stamp"] == STAMP and value["blockers"] == []
    lines = reform.plan_lines(value)
    assert lines[:2] == ["The cabled Sparks differ from this Spark's cluster record: spark-aa42 is cabled to this Spark.",
                         "Re-form: setup moves aside what these Sparks keep from other SparkRing clusters:"]
    aa42 = lines[lines.index("  spark-aa42:") + 1:]
    assert aa42[:6] == [
        "    - Node A of cluster \"tp2\" (2 Sparks): its records move aside",
        "    - automatic recovery: turned off",
        "    - admin network 10.253.255.1 of its cluster: stopped, turned off and its configuration moved aside",
        "    - fabric record (rank 0 of 2 Sparks): moved aside; its boot service is turned off",
        "    - fabric address 198.18.200.5/30 on enp1s0f1np1, not set by SparkRing: replaced after a backup of its "
        "NetworkManager connection",
        "    - fabric address 198.18.200.13/30 on enP2p1s0f1np1, not set by SparkRing: replaced after a backup of its "
        "NetworkManager connection"]
    assert "  spark-3286 (Node A):" in lines
    assert "    - Node A of cluster \"sparkring\" (2 Sparks): its records move aside, except Node A's SSH key" in lines
    assert [a["address"] for s in value["sparks"] for a in s["foreign_addresses"]] == [
        "198.18.200.6/30", "198.18.200.14/30", "198.18.200.5/30", "198.18.200.13/30"]
    assert lines[-1] == ("Then setup sets up the ring spark-3286 → spark-0a0f → spark-931e → spark-aa42 like a first setup "
                         "and renumbers its fabric addresses.")


def test_running_model_container_blocks_with_the_stop_command():
    found, _ = surveyed(containers=[{"name": "tp2-rank1", "deployment": "deployment-a", "profile": None},
                                    {"name": "other", "deployment": "elsewhere", "profile": None}])
    value = reform.plan(found, cabling.diagnose(survey.records(found), found["head"]), name="dgx4-2")
    assert value["blockers"] == [
        {"spark": "spark-931e", "container": "tp2-rank1", "command": "on spark-aa42: sudo sparkring down --execute"},
        {"spark": "spark-931e", "container": "other", "command": "on spark-931e: sudo docker stop other"}]
    assert reform.plan_lines(value)[-2:] == [
        "  spark-931e: tp2-rank1; stop it on spark-aa42: sudo sparkring down --execute",
        "  spark-931e: other; stop it on spark-931e: sudo docker stop other"]
    with pytest.raises(ValueError, match="Stop them, then repeat setup: tp2-rank1 on spark-931e"):
        reform.execute(value, found, None, archive=None, transfer=None, root_command=None, root="unused")


class Host:
    """A Spark's services, containers, firewall and routes as ``retire`` sees them through its command runner."""

    def __init__(self, root, *, units=None, containers="", recovering=False):
        self.root = Path(root)
        self.units = units or {}
        self.containers, self.recovering = containers, recovering
        self.rules = {("iptables", "filter"): ["-A INPUT -i sr-control -s 10.253.255.0/29 -p tcp -m tcp --dport 2222 "
                                               "-m comment --comment sparkring-control -j ACCEPT",
                                               "-A INPUT -i eth0 -j ACCEPT"],
                      ("iptables", "nat"): [], ("ip6tables", "filter"): []}
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[:2] == ["systemctl", "is-enabled"]:
            return self.units.get(argv[2], ("disabled", "inactive"))[0]
        if argv[:2] == ["systemctl", "is-active"]:
            if argv[2] == "sparkring-recover.service" and self.recovering:
                return "active"
            return self.units.get(argv[2], ("disabled", "inactive"))[1]
        if argv[:3] == ["systemctl", "disable", "--now"]:
            self.units[argv[3]] = ("disabled", "inactive")
            return ""
        if argv[:2] == ["docker", "ps"]:
            return self.containers
        if argv[-1] == "-S":
            return "\n".join(self.rules[(argv[0], argv[3])])
        if "-D" in argv:
            rule = "-A " + " ".join(argv[argv.index("-D") + 1:])
            self.rules[(argv[0], argv[3])].remove(rule)
            return ""
        if argv[:3] == ["ip", "link", "delete"] or argv[:2] == ["wg-quick", "down"]:
            (self.root / "sys/class/net/sr-control").rmdir()
        return ""


def write(root, path, text="{}"):
    target = Path(root) / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def former_node_a(root):
    """The files of a Spark that was Node A of a two-Spark cluster."""
    for path in ("/etc/sparkring/control.json", "/etc/sparkring/controller_keys", "/etc/sparkring/sshd_config",
                 "/etc/wireguard/sr-control.conf", "/etc/sparkring/node.json", "/etc/sparkring/control.key",
                 "/var/lib/sparkring/recovery.json", "/root/.ssh/sparkring_config",
                 "/var/lib/sparkring/controller/cluster.json", "/var/lib/sparkring/controller/enrolled.json",
                 "/var/lib/sparkring/controller/active.json", "/var/lib/sparkring/controller/install.lock",
                 "/var/lib/sparkring/controller/controller_ed25519", "/var/lib/sparkring/controller/controller_ed25519.pub",
                 "/var/lib/sparkring/controller/ssh/known_hosts", "/var/lib/sparkring/controller/deployments/a/lock.json",
                 "/var/lib/sparkring/controller/setups/1/setup.json",
                 "/var/lib/sparkring/controller/setups/2/worker-bundle.tar", "/srv/sparkring/tp2/checkpoint/x"):
        write(root, path)
    write(root, "/etc/sparkring/fabric.json", json.dumps({"cluster_id": "c" * 64, "routes": [
        {"destination": "198.18.4.0/24", "via": "198.18.2.2", "dev": "enp1s0f0np0"}]}))
    (Path(root) / "sys/class/net/sr-control").mkdir(parents=True)


ENABLED = {"sparkring-recover.timer": ("enabled", "active"), "sparkring-control.service": ("enabled", "active"),
           "sparkring-access.service": ("enabled", "active"), "sparkring-control-refresh.timer": ("enabled", "active"),
           "sparkring-fabric.service": ("enabled", "active")}


def test_retire_on_node_a_moves_cluster_state_and_keeps_the_key_lock_and_this_setup(tmp_path):
    former_node_a(tmp_path)
    host = Host(tmp_path, units=dict(ENABLED))
    keep = [*reform.NODE_A_KEEPS, "setups/2"]
    receipt = reform.retire({"stamp": STAMP, "node_a": True, "keep": keep}, call=host, root=tmp_path)
    controller = tmp_path / "var/lib/sparkring/controller"
    assert sorted(p.name for p in controller.iterdir()) == ["controller_ed25519", "controller_ed25519.pub",
                                                           "install.lock", "setups", "ssh"]
    assert [p.name for p in (controller / "setups").iterdir()] == ["2"]
    retired = tmp_path / "var/lib/sparkring/retired" / STAMP
    for path in ("etc/sparkring/control.json", "etc/sparkring/fabric.json", "etc/wireguard/sr-control.conf",
                 "var/lib/sparkring/controller/cluster.json", "var/lib/sparkring/controller/setups/1/setup.json",
                 "var/lib/sparkring/controller/deployments/a/lock.json", "root/.ssh/sparkring_config",
                 "var/lib/sparkring/recovery.json"):
        assert (retired / path).is_file(), path
        assert not (tmp_path / path).exists(), path
    # The Spark's own identity, its WireGuard key and the checkpoints stay.
    for path in ("etc/sparkring/node.json", "etc/sparkring/control.key", "srv/sparkring/tp2/checkpoint/x"):
        assert (tmp_path / path).exists()
    assert receipt["disabled"] == ["sparkring-recover.timer", "sparkring-control-refresh.timer", "sparkring-access.service",
                                   "sparkring-control.service", "sparkring-fabric.service"]
    assert "admin network interface sr-control" in receipt["removed"]
    assert "route 198.18.4.0/24 via 198.18.2.2 dev enp1s0f0np0" in receipt["removed"]
    assert host.rules[("iptables", "filter")] == ["-A INPUT -i eth0 -j ACCEPT"]
    assert {"from": "/etc/sparkring/control.json", "to": f"/var/lib/sparkring/retired/{STAMP}/etc/sparkring/control.json"} in receipt["moved"]
    assert (f"sudo mkdir -p /etc/sparkring && sudo mv -T /var/lib/sparkring/retired/{STAMP}/etc/sparkring/control.json "
            "/etc/sparkring/control.json") in receipt["restore"]
    assert "sudo systemctl enable sparkring-control.service" in receipt["restore"]
    written = json.loads((retired / "receipt.json").read_text(encoding="utf-8"))
    assert written == receipt
    if hasattr(os, "geteuid"):
        assert oct((retired / "receipt.json").stat().st_mode & 0o777) == "0o644"


def test_retire_repeats_safely_after_an_interruption(tmp_path):
    former_node_a(tmp_path)
    host = Host(tmp_path, units=dict(ENABLED))
    order = {"stamp": STAMP, "node_a": True, "keep": list(reform.NODE_A_KEEPS)}
    first = reform.retire(order, call=host, root=tmp_path)
    moved = len(first["moved"])
    host.calls.clear()
    again = reform.retire(order, call=host, root=tmp_path)
    assert len(again["moved"]) == moved and again["disabled"] == first["disabled"]
    assert not [argv for argv in host.calls if argv[:3] == ["systemctl", "disable", "--now"] or "-D" in argv]
    # A run interrupted after the services stopped moves the remaining files.
    other = tmp_path / "other"
    former_node_a(other)
    reform.retire({"stamp": STAMP, "node_a": False}, call=Host(other, units=dict(ENABLED)), root=other)
    assert not (other / "var/lib/sparkring/controller").exists()
    assert (other / "var/lib/sparkring/retired" / STAMP / "var/lib/sparkring/controller/controller_ed25519").exists()


@pytest.mark.parametrize("containers, recovering, message", [
    ("tp2-rank0\tdeployment-a\n", False, "SparkRing model containers run on this Spark: tp2-rank0"),
    ("", True, "automatic recovery attempt is running"),
])
def test_retire_refuses_before_any_change_while_a_model_or_recovery_runs(tmp_path, containers, recovering, message):
    former_node_a(tmp_path)
    host = Host(tmp_path, units=dict(ENABLED), containers=containers, recovering=recovering)
    with pytest.raises(RuntimeError, match=message):
        reform.retire({"stamp": STAMP, "node_a": False}, call=host, root=tmp_path)
    assert (tmp_path / "etc/sparkring/control.json").exists() and (tmp_path / "var/lib/sparkring/controller").exists()
    # Only automatic recovery's timer is turned off first, so no new attempt starts.
    assert [argv for argv in host.calls if argv[:2] == ["systemctl", "disable"]] == [
        ["systemctl", "disable", "--now", "sparkring-recover.timer"]]


def test_retire_rejects_a_stamp_that_is_not_a_plain_name(tmp_path):
    with pytest.raises(ValueError, match="Invalid retirement stamp"):
        reform.retire({"stamp": "../../etc"}, call=Host(tmp_path), root=tmp_path)


def test_execute_retires_workers_first_then_node_a_and_writes_the_receipt(tmp_path, monkeypatch):
    found, transport = surveyed()
    value = reform.plan(found, cabling.diagnose(survey.records(found), found["head"]), name="dgx4-2",
                        now=lambda: (2026, 10, 2, 14, 30, 0, 4, 275, 0))
    events = []

    def transfer(t, route, archive, destination):
        events.append(("transfer", route[-1]["address"], archive))

    def root_command(t, route, argv, *, data=None):
        events.append(("root", route[-1]["address"], "retire(json.loads(" in argv[-1] and "--prepare" in argv[-1]))

    def command(route, argv, *, data=None, tty=False):
        events.append(("read", route[-1]["address"], argv[-1]))
        return json.dumps({"schema": "sparkring-retired/v1", "moved": [{"from": "/etc/sparkring/control.json"}]})
    monkeypatch.setattr(transport, "command", command)

    def here(order):
        events.append(("node a", order["keep"]))
        return {"moved": []}
    monkeypatch.setattr(reform, "RETIRED", "/retired")
    receipts = reform.execute(value, found, transport, archive=lambda: "bundle.tar", transfer=transfer,
                              root_command=root_command, keep=["setups/7"], here=here, say=lambda line: None,
                              root=tmp_path)
    workers = [e[1] for e in events if e[0] == "root"]
    assert sorted(workers) == sorted([LAN["spark-0a0f"], LAN["spark-931e"], LAN["spark-aa42"]])
    assert all(e[2] for e in events if e[0] == "root")
    assert events[-1] == ("node a", [*reform.NODE_A_KEEPS, "setups/7"])
    assert set(receipts) == {"spark-3286", "spark-0a0f", "spark-931e", "spark-aa42"}
    document = json.loads((tmp_path / "retired" / STAMP / "reform.json").read_text(encoding="utf-8"))
    assert document["complete"] and document["plan"]["order"][0] == "spark-3286"
    assert "/var/lib/sparkring/backups/dgx4-2/rankN/" in document["network_backups"]


def test_worker_script_is_self_contained():
    code = reform.worker_script({"stamp": STAMP, "node_a": False, "keep": []}, "/var/tmp/sparkring-enroll-1/install.py")
    compile(code, "worker", "exec")
    assert code.rstrip().endswith("'--apply', '--prepare', '--yes'], check=True)")


def arguments(**values):
    return argparse.Namespace(**{"ssh_user": "code", "ssh_port": 22, "name": "dgx4-2", "plan": False,
                                 "reset_links": False, **values})


def test_setup_step_plans_without_changes(monkeypatch, capsys):
    found, transport = surveyed()
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (found, cabling.diagnose(survey.records(found), found["head"])))
    executed = []
    args = arguments(plan=True)
    assert single_uplink.reform_step(args, transport, None, reason="spark-aa42 is cabled", run=executed.append) == "planned"
    out = capsys.readouterr().out
    assert executed == [] and "Re-form: setup moves aside" in out and single_uplink.HAIRPIN_SCOPE[0] in out
    assert args.ssh_port == 22 and not args.reset_links


def test_setup_step_reforms_then_continues_over_the_preparation_service(monkeypatch):
    found, transport = surveyed()
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (found, cabling.diagnose(survey.records(found), found["head"])))
    calls = []
    args = arguments()
    result = single_uplink.reform_step(args, transport, "archive", reason=None, keep=["setups/1"], say=lambda line: None,
                                       run=lambda value, found, transport, **options: calls.append((value, options)))
    assert result == "done" and calls[0][1]["keep"] == ["setups/1"]
    assert (args.ssh_port, args.ssh_user, args.reset_links, transport.trust_new) == (2222, "root", True, True)


def test_setup_step_stops_at_the_cabling_fix(monkeypatch):
    found, transport = surveyed(fixed=False)
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (found, cabling.diagnose(survey.records(found), found["head"])))
    with pytest.raises(cabling.CablingError, match="On spark-aa42, swap its two cables"):
        single_uplink.reform_step(arguments(), transport, None, reason="x", say=lambda line: None)


def test_setup_step_leaves_sparks_without_cluster_state_to_ordinary_setup(monkeypatch):
    found, transport = surveyed()
    for spark in found["sparks"].values():
        spark["data"]["state"] = {"units": {}, "mesh": []}
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (found, None))
    assert single_uplink.reform_step(arguments(), transport, None, reason=None, say=lambda line: None) is None
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (_ for _ in ()).throw(RuntimeError("no LAN")))
    notes = []
    assert single_uplink.reform_step(arguments(), transport, None, reason=None, say=notes.append) is None
    assert notes == ["Note: reading the cabled Sparks before setup failed (no LAN); setup continues"]
    with pytest.raises(ValueError, match="Setup could not read the cabled Sparks"):
        single_uplink.reform_step(arguments(), transport, None, reason="x", say=notes.append)


def test_fresh_scope_lists_the_reform_step():
    args = argparse.Namespace(ssh_user="code", ssh_port=22, no_share_internet=False)
    assert any("keeps setup from another SparkRing cluster" in line for line in single_uplink.scope_lines(args, fresh=True))
    assert not any("another SparkRing cluster" in line for line in single_uplink.scope_lines(args, fresh=False))
    assert single_uplink.ring_state(Path("unused"), "spark-aa42 is cabled") == (True, True)


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="reads a Linux host")
def test_prior_state_reads_this_host_without_changing_it():
    result = reform.prior_state()
    assert {"controller", "control", "fabric", "units", "mesh", "containers", "fabric_ipv4", "unreadable"} <= set(result)


def test_setup_plan_reforms_when_the_record_names_fewer_sparks_than_are_cabled(tmp_path, monkeypatch, capsys):
    from runtime.host import controller
    base = tmp_path / "state"
    base.mkdir()
    (base / "cluster.json").write_text(json.dumps({"name": "sparkring", "plan": {
        "nodes": [{"hostname": "spark-3286"}, {"hostname": "spark-0a0f"}],
        "spec": {"hosts": [{"host": "root@198.51.100.1"}, {"host": "root@198.51.100.2"}]}}}))
    monkeypatch.setattr(controller, "STATE", base)
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(single_uplink.distribution, "installed", lambda root: True)
    monkeypatch.setattr(single_uplink, "identity_key", lambda directory: (tmp_path / "key", "controller public key"))
    monkeypatch.setattr(reform, "local_lldp", lambda: cabling.lldp_rows(captured()["spark-3286"]["lldp"]))
    found, _ = surveyed()
    seen = {}

    def survey_cabled(transport, **options):
        seen.update(options)
        return found, cabling.diagnose(survey.records(found), found["head"])
    monkeypatch.setattr(reform, "survey_cabled", survey_cabled)

    class SSH:
        def __init__(self, directory, *, identity=None, trust_new=False):
            self.identity, self.trust_new = identity, trust_new

        def trusted(self):
            return []
    monkeypatch.setattr(single_uplink.bootstrap, "SSH", SSH)
    monkeypatch.setattr(single_uplink.controller, "collect", lambda targets: pytest.fail("planned from the record"))
    monkeypatch.setattr(single_uplink.bootstrap, "discover", lambda *a, **k: pytest.fail("discovered before re-forming"))
    assert single_uplink.main(["--plan", "--name", "dgx4-2"]) == 0
    out = capsys.readouterr().out
    assert ("The cabled Sparks differ from this Spark's cluster record: spark-aa42 is cabled to this Spark but not "
            "part of its cluster \"sparkring\" (2 Sparks).") in out
    assert "Then setup sets up the ring spark-3286 → spark-0a0f → spark-931e → spark-aa42 like a first setup" in out
    assert seen["port"] == 22 and (base / "cluster.json").exists()


def test_setup_step_refuses_to_reform_a_spark_it_could_not_sign_in_to(monkeypatch):
    found, transport = surveyed()
    gone = next(k for k, s in found["sparks"].items() if s["data"]["inventory"]["hostname"] == "spark-931e")
    del found["sparks"][gone]
    found["notes"].append("Sign-in over the LAN at 192.0.2.31 as code failed: Permission denied")
    diagnosis = cabling.diagnose(survey.records(found), found["head"])
    assert diagnosis["ready"]
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (found, diagnosis))
    with pytest.raises(ValueError, match="Setup could not sign in to spark-931e, so it cannot re-form it"):
        single_uplink.reform_step(arguments(), transport, None, reason="x", say=lambda line: None)


def test_survey_program_with_state_is_self_contained():
    code = survey.program(reform.prior_state)
    compile(code, "observe", "exec")
    assert "def prior_state" in code and code.rstrip().endswith("print(json.dumps(observe(True)))")
