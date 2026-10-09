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
    """The ``prior_state`` document of one captured Spark of a former pair.

    Port 0 carries SparkRing's connections (EUI-64 link-local addresses). A
    port 1 with hand-made /30 addresses carries connections that generate
    their link-local addresses in NetworkManager's default mode.
    """
    ipv4 = fabric_ipv4(spark)
    names = {"enp1s0f1np1": "Wired connection 5", "enP2p1s0f1np1": "Wired connection 2"}
    link_local = {netdev: {"uuid": f"uuid-{netdev}", "addresses": [],
                           "connection": names.get(netdev, "sparkring-" + netdev),
                           "mode": "default" if netdev in names and any(a.endswith("/30") for a in ipv4[netdev])
                           else "eui64"}
                  for netdev in ipv4}
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
            "units": units, "fabric_ipv4": ipv4, "link_local": link_local}


def surveyed(*, fixed=True, containers=None):
    """A survey of the four captured Sparks from spark-b; ``fixed`` swaps spark-e's cables first."""
    sparks = swapped(captured(), "spark-e") if fixed else captured()
    near = ("spark-e", 1, "spark-d", 0) if fixed else ("spark-e", 0, "spark-d", 0)
    documents = {
        "local": observation(sparks["spark-b"], root=True),
        LAN["spark-a"]: observation(sparks["spark-a"], root=True),
        LAN["spark-e"]: observation(sparks["spark-e"], root=True,
                                       neighbors=link_local_neighbors(sparks, *near)),
        LAN["spark-d"]: observation(sparks["spark-d"], root=True,
                                       neighbors=link_local_neighbors(sparks, near[2], near[3], near[0], near[1])
                                       + link_local_neighbors(sparks, "spark-d", 1, "spark-a", 0)),
    }
    documents["local"]["state"] = state(sparks["spark-b"], cluster="sparkring", nodes=["spark-b", "spark-a"],
                                        address="10.253.255.1", rank=0, active="deployment-b")
    documents[LAN["spark-a"]]["state"] = state(sparks["spark-a"], address="10.253.255.2", rank=1)
    documents[LAN["spark-e"]]["state"] = state(sparks["spark-e"], cluster="tp2", nodes=["spark-e", "spark-d"],
                                                  address="10.253.255.1", rank=0, active="deployment-a")
    documents[LAN["spark-d"]]["state"] = state(sparks["spark-d"], address="10.253.255.2", rank=1)
    if containers:
        documents[LAN["spark-d"]]["state"]["containers"] = containers
    table = {next(r["address"] for r in s["addresses"] if r["ifname"] == "enP7s7"): LAN[n] for n, s in sparks.items()}
    transport = Transport(documents)
    here = survey.Reach("local", "this Spark")
    here.run = lambda t, argv, data=None, ssh=None: json.dumps(documents["local"])
    found = survey.survey(transport, recorded=(), user="operator", say=lambda line: None, state=reform.prior_state,
                          arp=lambda interface: dict(table), sweep=lambda interface: None, here=here)
    return found, transport


def test_record_mismatch_names_the_unrecorded_neighbor():
    record = {"name": "sparkring", "plan": {"nodes": [{"hostname": "spark-b"}, {"hostname": "spark-a"}]}}
    sparks = captured()
    rows = [row for row in cabling.lldp_rows(sparks["spark-b"]["lldp"])]
    assert reform.record_mismatch(record, rows, "spark-b") == (
        "spark-e is cabled to this Spark but not part of its cluster \"sparkring\" (2 Sparks)")
    pair = [row for row in rows if row["hostname"] in ("spark-a", "spark-b")]
    assert reform.record_mismatch(record, pair, "spark-b") is None
    ring = {"name": "ring", "plan": {"nodes": [{"hostname": h} for h in ("spark-b", "spark-a", "a", "b")]}}
    both = pair + [dict(row, netdev="enp1s0f1np1") for row in pair if row["hostname"] == "spark-a"]
    assert "not cabled as a ring" in reform.record_mismatch(ring, both, "spark-b")
    assert reform.record_mismatch({"targets": ["a", "b"]}, rows, "spark-b") == (
        "3 Sparks are cabled here, but setup enrolled 2")


def test_record_reason_reads_cluster_record_first(tmp_path):
    (tmp_path / "enrolled.json").write_text(json.dumps({"targets": ["a", "b"]}))
    (tmp_path / "cluster.json").write_text(json.dumps(
        {"name": "sparkring", "plan": {"nodes": [{"hostname": "spark-b"}, {"hostname": "spark-e"}]}}))
    rows = cabling.lldp_rows(captured()["spark-b"]["lldp"])
    assert single_uplink.record_reason(tmp_path, lldp=lambda: rows, hostname="spark-b").startswith("spark-a is")
    assert single_uplink.record_reason(tmp_path / "none", lldp=lambda: rows) is None


def test_plan_lists_each_sparks_state_foreign_addresses_and_the_ring():
    found, _ = surveyed()
    diagnosis = cabling.diagnose(survey.records(found), found["head"])
    assert diagnosis["ready"] and diagnosis["order_names"] == ["spark-b", "spark-a", "spark-d", "spark-e"]
    value = reform.plan(found, diagnosis, name="dgx4-2", reason="spark-e is cabled to this Spark",
                        now=lambda: (2026, 10, 2, 14, 30, 0, 4, 275, 0))
    assert value["stamp"] == STAMP and value["blockers"] == []
    lines = reform.plan_lines(value)
    assert lines[:2] == ["The cabled Sparks differ from this Spark's cluster record: spark-e is cabled to this Spark.",
                         "Re-form: setup moves aside what these Sparks keep from other SparkRing clusters:"]
    aa42 = lines[lines.index("  spark-e:") + 1:]
    assert aa42[:8] == [
        "    - Node A of cluster \"tp2\" (2 Sparks): its records move aside",
        "    - automatic recovery: turned off",
        "    - admin network 10.253.255.1 of its cluster: stopped, turned off and its configuration moved aside",
        "    - fabric record (rank 0 of 2 Sparks): moved aside; its boot service is turned off",
        "    - IPv6 link-local addresses of enp1s0f1np1 (Wired connection 5): set to the hardware-derived form "
        "SparkRing uses, so RoCE GID index 3 holds the port's IPv4 address",
        "    - IPv6 link-local addresses of enP2p1s0f1np1 (Wired connection 2): set to the hardware-derived form "
        "SparkRing uses, so RoCE GID index 3 holds the port's IPv4 address",
        "    - fabric address 198.18.200.5/30 on enp1s0f1np1, not set by SparkRing: replaced after a backup of its "
        "NetworkManager connection",
        "    - fabric address 198.18.200.13/30 on enP2p1s0f1np1, not set by SparkRing: replaced after a backup of its "
        "NetworkManager connection"]
    # SparkRing's own connections already use the hardware-derived form.
    assert not any("link-local" in line for line in lines[lines.index("  spark-a:"):lines.index("  spark-d:")])
    assert "  spark-b (Node A):" in lines
    assert "    - Node A of cluster \"sparkring\" (2 Sparks): its records move aside, except Node A's SSH key" in lines
    assert [a["address"] for s in value["sparks"] for a in s["foreign_addresses"]] == [
        "198.18.200.6/30", "198.18.200.14/30", "198.18.200.5/30", "198.18.200.13/30"]
    assert lines[-1] == ("Then setup sets up the ring spark-b → spark-a → spark-d → spark-e like a first setup "
                         "and renumbers its fabric addresses.")


def test_running_model_container_blocks_with_the_stop_command():
    found, _ = surveyed(containers=[{"name": "tp2-rank1", "deployment": "deployment-a", "profile": None},
                                    {"name": "other", "deployment": "elsewhere", "profile": None}])
    value = reform.plan(found, cabling.diagnose(survey.records(found), found["head"]), name="dgx4-2")
    assert value["blockers"] == [
        {"spark": "spark-d", "container": "tp2-rank1", "command": "on spark-e: sudo sparkring down --execute"},
        {"spark": "spark-d", "container": "other", "command": "on spark-d: sudo docker stop other"}]
    assert reform.plan_lines(value)[-2:] == [
        "  spark-d: tp2-rank1; stop it on spark-e: sudo sparkring down --execute",
        "  spark-d: other; stop it on spark-d: sudo docker stop other"]
    with pytest.raises(ValueError, match="Stop them, then repeat setup: tp2-rank1 on spark-d"):
        reform.execute(value, found, None, archive=None, transfer=None, root_command=None, root="unused")


class Host:
    """A Spark's services, containers, firewall and routes as ``retire`` sees them through its command runner."""

    def __init__(self, root, *, units=None, containers="", recovering=False, connections=None):
        self.root = Path(root)
        self.units = units or {}
        self.containers, self.recovering = containers, recovering
        # netdev -> {"uuid", "name", "mode", "file", "link_local": [addresses]}
        self.connections = connections or {}
        for netdev in self.connections:
            (self.root / "sys/class/net" / netdev).mkdir(parents=True, exist_ok=True)
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
        by_uuid = {row["uuid"]: row for row in self.connections.values()}
        if argv == ["nmcli", "-t", "-f", "UUID,FILENAME", "connection", "show"]:
            return "\n".join(f"{row['uuid']}:{row['file']}" for row in self.connections.values())
        if argv[:5] == ["nmcli", "-g", "GENERAL.CON-UUID", "device", "show"]:
            return self.connections.get(argv[5], {}).get("uuid", "--")
        fields = {"ipv6.addr-gen-mode": "mode", "connection.id": "name"}
        if argv[:2] == ["nmcli", "-g"] and argv[2] in fields and argv[3:5] == ["connection", "show"]:
            return by_uuid[argv[5]][fields[argv[2]]]
        if argv[:3] == ["nmcli", "connection", "modify"]:
            by_uuid[argv[3]]["mode"] = argv[5]
        if argv[:3] == ["nmcli", "connection", "up"]:
            row = by_uuid[argv[3]]
            if row["mode"] == "eui64":
                row["link_local"] = row["link_local"][:1]
        if argv[:5] == ["ip", "-j", "-6", "address", "show"]:
            addresses = self.connections[argv[-1]]["link_local"]
            return json.dumps([{"addr_info": [{"family": "inet6", "local": a, "scope": "link"} for a in addresses]}])
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


def test_retire_stops_the_fabric_service_before_it_removes_the_fabric_routes(tmp_path):
    # The host agent restores approved routes while the fabric service is active.
    former_node_a(tmp_path)
    host = Host(tmp_path, units=dict(ENABLED))
    reform.retire({"stamp": STAMP, "node_a": True, "keep": list(reform.NODE_A_KEEPS)}, call=host, root=tmp_path)
    stop = host.calls.index(["systemctl", "disable", "--now", "sparkring-fabric.service"])
    removals = [index for index, argv in enumerate(host.calls) if argv[:3] == ["ip", "route", "del"]]
    assert removals and stop < min(removals)


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
        events.append(("root", route[-1]["address"], "retire(dict(order, link_local=False))" in argv[-1]
                       and "--prepare" in argv[-1]))

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
    assert sorted(workers) == sorted([LAN["spark-a"], LAN["spark-d"], LAN["spark-e"]])
    assert all(e[2] for e in events if e[0] == "root")
    assert events[-1] == ("node a", [*reform.NODE_A_KEEPS, "setups/7"])
    assert set(receipts) == {"spark-b", "spark-a", "spark-d", "spark-e"}
    document = json.loads((tmp_path / "retired" / STAMP / "reform.json").read_text(encoding="utf-8"))
    assert document["complete"] and document["plan"]["order"][0] == "spark-b"
    assert "/var/lib/sparkring/backups/dgx4-2/rankN/" in document["network_backups"]


def test_worker_script_is_self_contained():
    code = reform.worker_script({"stamp": STAMP, "node_a": False, "keep": []}, "/var/tmp/sparkring-enroll-1/install.py")
    compile(code, "worker", "exec")
    # Retirement, then installation and preparation, then the link-local step, which ignores a lost session.
    tail = code[code.index("order = json.loads("):].splitlines()
    assert tail[1:] == ["retire(dict(order, link_local=False))",
                        "subprocess.run(['python3', '-I', '/var/tmp/sparkring-enroll-1/install.py', '--apply', "
                        "'--prepare', '--yes'], check=True)",
                        "signal.signal(signal.SIGHUP, signal.SIG_IGN)", "retire(order)"]


def arguments(**values):
    return argparse.Namespace(**{"ssh_user": "operator", "ssh_port": 22, "name": "dgx4-2", "plan": False,
                                 "reset_links": False, **values})


def test_setup_step_plans_without_changes(monkeypatch, capsys):
    found, transport = surveyed()
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (found, cabling.diagnose(survey.records(found), found["head"])))
    executed = []
    args = arguments(plan=True)
    assert single_uplink.reform_step(args, transport, None, reason="spark-e is cabled", run=executed.append) == "planned"
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
    with pytest.raises(cabling.CablingError, match="On spark-e, swap its two cables"):
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
    args = argparse.Namespace(ssh_user="operator", ssh_port=22, no_share_internet=False)
    assert any("keeps setup from another SparkRing cluster" in line for line in single_uplink.scope_lines(args, fresh=True))
    assert not any("another SparkRing cluster" in line for line in single_uplink.scope_lines(args, fresh=False))
    assert single_uplink.ring_state(Path("unused"), "spark-e is cabled") == (True, True)


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="reads a Linux host")
def test_prior_state_reads_this_host_without_changing_it():
    result = reform.prior_state()
    assert {"controller", "control", "fabric", "units", "mesh", "containers", "fabric_ipv4", "unreadable"} <= set(result)


def test_setup_plan_reforms_when_the_record_names_fewer_sparks_than_are_cabled(tmp_path, monkeypatch, capsys):
    from runtime.host import controller
    base = tmp_path / "state"
    base.mkdir()
    (base / "cluster.json").write_text(json.dumps({"name": "sparkring", "plan": {
        "nodes": [{"hostname": "spark-b"}, {"hostname": "spark-a"}],
        "spec": {"hosts": [{"host": "root@198.51.100.1"}, {"host": "root@198.51.100.2"}]}}}))
    monkeypatch.setattr(controller, "STATE", base)
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(single_uplink.distribution, "installed", lambda root: True)
    monkeypatch.setattr(single_uplink, "identity_key", lambda directory: (tmp_path / "key", "controller public key"))
    monkeypatch.setattr(reform, "local_lldp", lambda: cabling.lldp_rows(captured()["spark-b"]["lldp"]))
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
    assert ("The cabled Sparks differ from this Spark's cluster record: spark-e is cabled to this Spark but not "
            "part of its cluster \"sparkring\" (2 Sparks).") in out
    assert "Then setup sets up the ring spark-b → spark-a → spark-d → spark-e like a first setup" in out
    assert seen["port"] == 22 and (base / "cluster.json").exists()


def test_setup_step_refuses_to_reform_a_spark_it_could_not_sign_in_to(monkeypatch):
    found, transport = surveyed()
    gone = next(k for k, s in found["sparks"].items() if s["data"]["inventory"]["hostname"] == "spark-d")
    del found["sparks"][gone]
    found["notes"].append("Sign-in over the LAN at 192.0.2.31 as operator failed: Permission denied")
    diagnosis = cabling.diagnose(survey.records(found), found["head"])
    assert diagnosis["ready"]
    monkeypatch.setattr(reform, "survey_cabled", lambda t, **o: (found, diagnosis))
    with pytest.raises(ValueError, match="Setup could not sign in to spark-d, so it cannot re-form it"):
        single_uplink.reform_step(arguments(), transport, None, reason="x", say=lambda line: None)


def test_survey_program_with_state_is_self_contained():
    code = survey.program(reform.prior_state)
    compile(code, "observe", "exec")
    assert "def prior_state" in code and code.rstrip().endswith("print(json.dumps(observe(True)))")


def foreign_connections():
    """spark-d's fabric connections: SparkRing's on port 0, hand-made ones on port 1 in the default mode."""
    rows = {}
    for netdev, name, mode in (("enp1s0f0np0", "sparkring-a", "eui64"), ("enP2p1s0f0np0", "sparkring-b", "eui64"),
                               ("enp1s0f1np1", "Wired connection 5", "default"),
                               ("enP2p1s0f1np1", "Wired connection 2", "default")):
        uuid = f"00000000-0000-4000-8000-{len(rows):012d}"
        rows[netdev] = {"uuid": uuid, "name": name, "mode": mode,
                        "file": f"/run/NetworkManager/system-connections/netplan-NM-{uuid}.nmconnection",
                        "link_local": ["fe80::ff:fe3d:a431"] + (["fe80::a8c1:5eff:4d2b:91f0"] if mode != "eui64" else [])}
    return rows


def network_files(root, connections):
    for row in connections.values():
        write(root, row["file"], "[connection]\nid=" + row["name"] + "\n")
        write(root, f"/etc/netplan/90-NM-{row['uuid']}.yaml", "network: {}\n")


def test_retire_sets_the_hardware_derived_link_local_form_after_a_backup(tmp_path):
    connections = foreign_connections()
    network_files(tmp_path, connections)
    host = Host(tmp_path, connections=connections)
    receipt = reform.retire({"stamp": STAMP, "node_a": False}, call=host, root=tmp_path)
    assert {row["mode"] for row in connections.values()} == {"eui64"}
    assert all(len(row["link_local"]) == 1 for row in connections.values())
    changed = [row for row in receipt["link_local"] if row["changed"]]
    assert [(row["netdev"], row["connection"], row["mode"]) for row in changed] == [
        ("enp1s0f1np1", "Wired connection 5", "default"), ("enP2p1s0f1np1", "Wired connection 2", "default")]
    uuid = connections["enp1s0f1np1"]["uuid"]
    assert changed[0]["backup"] == [
        f"/var/lib/sparkring/retired/{STAMP}/network/run/NetworkManager/system-connections/netplan-NM-{uuid}.nmconnection",
        f"/var/lib/sparkring/retired/{STAMP}/network/etc/netplan/90-NM-{uuid}.yaml"]
    retired = tmp_path / "var/lib/sparkring/retired" / STAMP / "network"
    assert (retired / f"etc/netplan/90-NM-{uuid}.yaml").is_file()
    # Copies: NetworkManager's own files stay where they are.
    assert (tmp_path / f"etc/netplan/90-NM-{uuid}.yaml").is_file()
    modify = host.calls.index(["nmcli", "connection", "modify", uuid, "ipv6.addr-gen-mode", "eui64"])
    assert host.calls[modify + 1:].index(["nmcli", "connection", "up", uuid]) >= 0
    assert (f"sudo nmcli connection modify {uuid} ipv6.addr-gen-mode default && sudo nmcli connection up {uuid}"
            in receipt["restore"])
    # SparkRing's own connections are left alone, and a repeated run changes nothing.
    assert not [argv for argv in host.calls if argv[:4] == ["nmcli", "connection", "modify", connections["enp1s0f0np0"]["uuid"]]]
    host.calls.clear()
    reform.retire({"stamp": STAMP, "node_a": False}, call=host, root=tmp_path)
    assert not [argv for argv in host.calls if argv[1:3] in (["connection", "modify"], ["connection", "up"])]


def test_retire_reactivates_a_connection_left_between_modify_and_up(tmp_path):
    connections = foreign_connections()
    connections["enp1s0f1np1"]["mode"] = "eui64"
    host = Host(tmp_path, connections=connections)
    receipt = reform.retire({"stamp": STAMP, "node_a": False}, call=host, root=tmp_path)
    uuid = connections["enp1s0f1np1"]["uuid"]
    assert ["nmcli", "connection", "up", uuid] in host.calls
    assert not [argv for argv in host.calls if argv[:4] == ["nmcli", "connection", "modify", uuid]]
    assert any(row.get("reactivated") and row["netdev"] == "enp1s0f1np1" for row in receipt["link_local"])
    assert connections["enp1s0f1np1"]["link_local"] == ["fe80::ff:fe3d:a431"]


def test_retire_keeps_the_link_local_form_of_the_function_setup_reaches_through(tmp_path):
    connections = foreign_connections()
    host = Host(tmp_path, connections=connections)
    receipt = reform.retire({"stamp": STAMP, "node_a": False, "skip_link_local": ["enp1s0f1np1"]}, call=host,
                            root=tmp_path)
    assert connections["enp1s0f1np1"]["mode"] == "default" and connections["enP2p1s0f1np1"]["mode"] == "eui64"
    skipped = next(row for row in receipt["link_local"] if row["netdev"] == "enp1s0f1np1")
    assert skipped["skipped"].endswith(f"ipv6.addr-gen-mode eui64 && sudo nmcli connection up {skipped['uuid']}")
    # The first retire of a worker leaves the link-local form for the run after installation.
    other = tmp_path / "other"
    host = Host(other, connections=foreign_connections())
    assert reform.retire({"stamp": STAMP, "link_local": False}, call=host, root=other).get("link_local") is None
    assert not [argv for argv in host.calls if argv[:3] == ["nmcli", "connection", "modify"]]


def test_route_over_a_fabric_hop_names_the_function_it_ends_at():
    reach = survey.Reach("route", "a cable", route=[{"user": "operator", "address": "192.0.2.42", "interface": None, "port": 22},
                                                   {"user": "operator", "address": "fe80::2", "interface": "enp1s0f1np1",
                                                    "port": 22}])
    found = {"sparks": {"x": {"reach": reach, "data": {"inventory": {"functions": [
        {"netdev": "enp1s0f0np0", "addresses": ["fe80::1"]}, {"netdev": "enp1s0f1np1", "addresses": ["fe80::2"]}]}}}}}
    assert reform.through(found, "x") == ["enp1s0f1np1"]
    found["sparks"]["x"]["reach"] = survey.Reach("route", "the LAN", route=reach.route[:1])
    assert reform.through(found, "x") == []
