"""CPU tests of the node-local ConnectX hairpin commands against a simulated Spark.

The simulated Spark answers the commands that runtime/host/hairpin.py runs
(devlink, ethtool, systemctl, tc, rdma, nmcli, ip, ping, wg) from an in-memory
model of four ConnectX functions, keeps sysfs and /etc below a temporary root,
and advances a fake clock only when a command or a wait would take time.
"""
import base64
import fnmatch
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from runtime.host import hairpin, topology
from scripts import deploy_network
from scripts import hairpin_setting as rule

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the hairpin lock and file modes need POSIX")

ROLES = hairpin.ROLES
BDF = {"cw_primary": "0000:01:00.0", "cw_secondary": "0002:01:00.0",
       "ccw_primary": "0000:01:00.1", "ccw_secondary": "0002:01:00.1"}
NETDEV = {"cw_primary": "enp1s0f0np0", "cw_secondary": "enP2p1s0f0np0",
          "ccw_primary": "enp1s0f1np1", "ccw_secondary": "enP2p1s0f1np1"}
NODE_ID = "5b0c6d2e-8f3a-4c1d-9e7b-2a4f6c8d0e1f"
BOOT_ID = "fc21e622-c9b7-47cd-a1db-1fa89aef0ca1"
EARLIER_BOOT = "2070ca77-28ad-4909-975d-fa8834ff2295"
INVOCATION = "6a1f0c9e2b7d4e3f8a5b1c0d9e8f7a6b"
KEY = "A" * 43 + "="
CHANGING = (("devlink", "dev", "param", "set"), ("devlink", "dev", "reload"), ("ethtool", "-K"),
            ("systemctl", "enable"), ("systemctl", "disable"), ("systemctl", "--no-block"),
            ("systemctl", "start"), ("systemctl", "restart"), ("nmcli", "--wait"), ("wg", "set"))


def ok(stdout=""):
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def fail(stderr="error", code=1):
    return SimpleNamespace(returncode=code, stdout="", stderr=stderr)


class Spark:
    """A DGX Spark with four ConnectX functions, as hairpin.py observes and changes it."""

    def __init__(self, root, *, size=1024, counter=0, offload="on", booting=False, approved=True, armed=False,
                 fabric=4):
        self.root = Path(root)
        self.time = 100.0
        self.calls, self.lines, self.requests = [], [], []
        self.booting = booting
        self.invocation = INVOCATION
        self.enabled = {hairpin.UNIT} if armed else set()
        self.units = {}
        self.tc, self.qp, self.pd, self.gpu = {}, [], [], ""
        self.reload = {}
        self.on_reload = None
        self.drop_addresses = False
        # By PCI address: how many address polls after a restart report the RDMA
        # port down and GID index 3 unmapped; -1 keeps them so.
        self.rdma_down = {}
        self.ping = True
        self.unreachable = set()
        self.stats = True
        self.hairpin_unit = {"ActiveState": "activating", "Result": "success"}
        self.functions = {}
        for index, role in enumerate(ROLES):
            pci = BDF[role]
            address = f"198.18.{index}.1"
            self.functions[pci] = {
                "role": role, "netdev": NETDEV[role], "rdma": topology.DEVICES[role],
                "mac": f"4c:bb:47:e6:ed:{index:02x}", "values": {"hairpin_queue_size": size, "hairpin_num_queues": 4},
                "pending": {}, "counter": counter, "failed": False, "offload": offload, "ipv4": address,
                "link_local": f"fe80::4ebb:47ff:fee6:ed{index:02x}", "addressed": not booting,
                "connection": f"uuid-{index}", "autoconnect": "yes", "bound": NETDEV[role]}
            self.sysfs(pci)
        self.write("/etc/sparkring/node.json", {"schema": "sparkring-node/v1", "node_id": NODE_ID})
        for name, text in (("proc/sys/kernel/random/boot_id", BOOT_ID), ("proc/sys/kernel/osrelease", "6.17.0-1029-nvidia"),
                           ("proc/cmdline", "BOOT_IMAGE=/vmlinuz ro quiet")):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(text + "\n")
        if fabric:
            self.write(hairpin.FABRIC, {"schema": "sparkring-fabric-state/v1", "size": fabric, "interfaces": [
                {"role": f["role"], "rdma_device": f["rdma"], "netdev": f["netdev"], "mac": f["mac"],
                 "address": f["ipv4"] + "/24"} for f in self.functions.values()][:fabric]})
        if approved:
            self.write(hairpin.APPROVAL, self.approval())

    # State below the root.

    def write(self, name, value):
        path = self.root / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def read(self, name):
        return json.loads((self.root / name.lstrip("/")).read_text())

    def exists(self, name):
        return (self.root / name.lstrip("/")).exists()

    def approval(self):
        return {"schema": hairpin.APPROVAL_SCHEMA, "node_id": NODE_ID, "parameters": dict(rule.PARAMETERS),
                "hw_tc_offload": True,
                "functions": [{"role": f["role"], "rdma_device": f["rdma"], "pci_address": pci, "netdev": f["netdev"],
                               "mac": f["mac"]} for pci, f in self.functions.items()]}

    def sysfs(self, pci):
        f = self.functions[pci]
        rdma = self.root / "sys/class/infiniband" / f["rdma"]
        (rdma / "device").mkdir(parents=True, exist_ok=True)
        (rdma / "device/uevent").write_text(f"DRIVER=mlx5_core\nPCI_SLOT_NAME={pci}\n")
        (rdma / "fw_ver").write_text("28.45.4028\n")
        (rdma / "ports/1/gids").mkdir(parents=True, exist_ok=True)
        (rdma / "ports/1/state").write_text("4: ACTIVE\n")
        (rdma / "ports/1/gids/3").write_text(ipaddress.IPv6Address("::ffff:" + f["ipv4"]).exploded + "\n")
        device = self.root / "sys/bus/pci/devices" / pci
        (device / "net" / f["netdev"]).mkdir(parents=True, exist_ok=True)
        (device / "net" / f["netdev"] / "address").write_text(f["mac"] + "\n")
        (device / "infiniband" / f["rdma"]).mkdir(parents=True, exist_ok=True)
        (self.root / "sys/class/net" / f["netdev"]).mkdir(parents=True, exist_ok=True)
        (self.root / "sys/class/net" / f["netdev"] / "address").write_text(f["mac"] + "\n")

    def remove(self, pci):
        self.functions[pci]["absent"] = True
        shutil.rmtree(self.root / "sys/bus/pci/devices" / pci)

    def control(self, *, head=False):
        """Administration tree: a child behind cw_primary and, on a worker, the parent behind ccw_secondary."""
        child, parent = self.functions[BDF["cw_primary"]], self.functions[BDF["ccw_secondary"]]
        peers = [{"id": "child", "key": KEY, "allowed_ips": ["10.253.255.3/32", "10.253.255.4/32"],
                  "endpoint": f"[fe80::1%{child['netdev']}]:51871", "netdev": child["netdev"]}]
        links = [{"netdev": child["netdev"], "mac": child["mac"], "address": child["link_local"]}]
        if not head:
            peers.append({"id": "parent", "key": KEY.replace("A", "B", 1), "allowed_ips": ["0.0.0.0/0"],
                          "endpoint": f"[fe80::2%{parent['netdev']}]:51871", "netdev": parent["netdev"]})
            links.append({"netdev": parent["netdev"], "mac": parent["mac"], "address": parent["link_local"]})
        self.write(hairpin.CONTROL, {"schema": "sparkring-control/v1", "id": "x", "address": "10.253.255.1" if head else "10.253.255.2",
                                     "head": head, "head_address": "10.253.255.1", "subnet": "10.253.255.0/29",
                                     "share_uplink": True, "peers": peers, "links": links, "uplink": "enP7s7" if head else None})

    def journal(self, *records):
        self.write(hairpin.ATTEMPTS, {"schema": hairpin.ATTEMPTS_SCHEMA, "records": list(records)})

    # The host interface.

    def host(self, **overrides):
        # Under systemd JOURNAL_STREAM names the journal stream, and error
        # lines then start with <3>.
        settings = dict(root=self.root, run=self.run, clock=lambda: self.time, sleep=self.sleep,
                        now=lambda: 1_790_000_000 + self.time,
                        environ={"INVOCATION_ID": self.invocation, "JOURNAL_STREAM": "8:4242"},
                        output=self.lines.append, hostname="spark-test")
        settings.update(overrides)
        return hairpin.Host(**settings)

    def sleep(self, seconds):
        self.time += seconds

    def by_netdev(self, netdev):
        return next(f for f in self.functions.values() if f["netdev"] == netdev)

    def device(self, handle):
        function = self.functions.get(handle.removeprefix("pci/"))
        return None if function is None or function.get("absent") else function

    def run(self, argv, *, timeout=None, **_):
        assert timeout and timeout > 0, argv
        a = list(argv)
        self.calls.append(a)
        if a[0] == "systemctl":
            return self.systemctl(a[1:])
        if a[:5] == ["devlink", "-j", "dev", "param", "show"]:
            f = self.device(a[5])
            if not f:
                return fail("kernel answers: No such device")
            shown = f["pending"].get(a[7], f["values"][a[7]])
            return ok(json.dumps({"param": {a[5]: [{"name": a[7], "type": "driver-specific",
                                                    "values": [{"cmode": "driverinit", "value": shown}]}]}}))
        if a[:5] == ["devlink", "-s", "-j", "dev", "show"]:
            f = self.device(a[5])
            if not f or not self.stats:
                return fail("kernel answers: Operation not supported")
            entry = {"stats": {"reload": {"driver_reinit": {"unspecified": f["counter"]},
                                          "fw_activate": {"unspecified": 0, "no_reset": 0}},
                               "remote_reload": {"driver_reinit": {"unspecified": 0},
                                                 "fw_activate": {"unspecified": 0, "no_reset": 0}}}}
            if f["failed"]:
                entry["reload_failed"] = True
            return ok(json.dumps({"dev": {a[5]: entry}}))
        if a[:4] == ["devlink", "-j", "dev", "show"]:
            return ok(json.dumps({"dev": {a[4]: {}}})) if self.device(a[4]) else fail("kernel answers: No such device")
        if a[:4] == ["devlink", "dev", "param", "set"]:
            self.device(a[4])["pending"][a[6]] = int(a[8])
            return ok()
        if a[:3] == ["devlink", "dev", "reload"]:
            return self.restart(a, timeout)
        if a[:2] == ["ethtool", "-k"]:
            return ok(f"Features for {a[2]}:\nrx-checksumming: on\nhw-tc-offload: {self.by_netdev(a[2])['offload']}\n")
        if a[:2] == ["ethtool", "-K"]:
            f = self.by_netdev(a[2])
            if f["offload"] == "off [fixed]":
                return fail("Could not change any device features")
            f["offload"] = "on"
            return ok()
        if a[:2] == ["udevadm", "settle"]:
            return ok()
        if a[:3] == ["tc", "-j", "filter"]:
            return ok(json.dumps(self.tc.get(a[5], [])))
        if a[:4] == ["rdma", "-j", "resource", "show"]:
            return ok(json.dumps(self.qp if a[4] == "qp" else self.pd))
        if a[0] == "nvidia-smi":
            return ok(self.gpu)
        if a[:3] == ["ip", "-j", "address"]:
            f = self.by_netdev(a[5])
            self.poll_rdma(f)
            info = [{"family": "inet6", "local": f["link_local"], "prefixlen": 64, "scope": "link"}]
            if f["addressed"]:
                info.insert(0, {"family": "inet", "local": f["ipv4"], "prefixlen": 24, "scope": "global"})
            return ok(json.dumps([{"ifname": a[5], "operstate": "UP", "addr_info": info}]))
        if a[:3] == ["ip", "-j", "-6"]:
            f = self.by_netdev(a[6])
            return ok(json.dumps([{"ifname": a[6], "addr_info": [{"family": "inet6", "local": f["link_local"], "scope": "link"}]}]))
        if a[:3] == ["nmcli", "-g", "GENERAL.CON-UUID"]:
            f = self.by_netdev(a[5])
            return ok(f["connection"] + "\n" if f["addressed"] else "\n")
        if a[:2] == ["nmcli", "-g"] and a[3:5] == ["connection", "show"]:
            f = next(f for f in self.functions.values() if f["connection"] == a[6])
            return ok(f"mesh-{f['role']}\n{f['autoconnect']}\n{f['bound']}\n\n")
        if a[:3] == ["nmcli", "--wait", "10"]:
            next(f for f in self.functions.values() if f["connection"] == a[-1])["addressed"] = True
            return ok()
        if a[0] == "ping":
            self.time += 1
            return ok() if self.ping and a[-1] not in self.unreachable else fail("1 packets transmitted, 0 received")
        if a[:2] == ["wg", "set"]:
            return ok()
        raise AssertionError("unexpected command: " + " ".join(a))

    def restart(self, argv, timeout):
        pci = argv[3].removeprefix("pci/")
        f = self.functions[pci]
        behavior = self.reload.get(pci, "ok")
        if self.on_reload:
            self.on_reload(pci)
        if behavior == "hang":
            self.time += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        if behavior == "refused":
            self.time += 1
            return fail("Error: devlink: reload failed.\nkernel answers: Device or resource busy")
        self.time += 6
        f["values"].update(f["pending"])
        f["pending"] = {}
        if behavior == "broken":
            f["failed"] = True
            return fail("kernel answers: Input/output error")
        # A reload that returns 0 but does not count, counts twice, or sets reload_failed.
        f["counter"] += {"uncounted": 0, "double": 2}.get(behavior, 1)
        f["failed"] = behavior == "flagged"
        if self.drop_addresses:
            f["addressed"] = False
        if behavior == "vanish":
            shutil.rmtree(self.root / "sys/bus/pci/devices" / pci / "net" / f["netdev"])
        if behavior == "signal":
            # The kernel finished the restart; a signal then ended devlink.
            return SimpleNamespace(returncode=-15, stdout="", stderr="")
        if pci in self.rdma_down:
            rdma = self.root / "sys/class/infiniband" / f["rdma"]
            (rdma / "ports/1/state").write_text("1: DOWN\n")
            (rdma / "ports/1/gids/3").write_text("0000:0000:0000:0000:0000:0000:0000:0000\n")
            f["rdma_polls"] = self.rdma_down[pci]
        return ok()

    def poll_rdma(self, f):
        remaining = f.get("rdma_polls")
        if remaining is None or remaining < 0:
            return
        if remaining:
            f["rdma_polls"] = remaining - 1
            return
        rdma = self.root / "sys/class/infiniband" / f["rdma"]
        (rdma / "ports/1/state").write_text("4: ACTIVE\n")
        (rdma / "ports/1/gids/3").write_text(ipaddress.IPv6Address("::ffff:" + f["ipv4"]).exploded + "\n")
        f["rdma_polls"] = None

    def unit(self, name):
        if name == hairpin.UNIT:
            return {"Id": name, "InvocationID": self.invocation, **self.hairpin_unit}
        if name == "NetworkManager.service":
            return {"Id": name, "ActiveState": "failed", "InactiveExitTimestampMonotonic": "0" if self.booting else "5791234"}
        return {"Id": name, "ActiveState": "inactive", "ControlGroup": "", **self.units.get(name, {})}

    def systemctl(self, args):
        if args[0] == "show":
            properties = args[args.index("-p") + 1].split(",")
            names = [x for x in args[1:] if not x.startswith("-") and x != args[args.index("-p") + 1]]
            if "--value" in args:
                return ok("".join(f"{self.unit(n).get(p, '')}\n" for n in names for p in properties))
            return ok("\n".join("".join(f"{p}={self.unit(n).get(p, '')}\n" for p in properties) for n in names))
        if args[0] == "list-units":
            patterns = [x for x in args[1:] if not x.startswith("-")]
            return ok("".join(f"{name} loaded {values.get('ActiveState', 'inactive')} dead Fixture unit\n"
                              for name, values in sorted(self.units.items())
                              if any(fnmatch.fnmatchcase(name, p) for p in patterns)))
        if args[0] == "is-enabled":
            return ok("enabled\n") if args[1] in self.enabled else SimpleNamespace(returncode=1, stdout="disabled\n", stderr="")
        if args[0] == "enable":
            self.enabled.add(args[1])
            return ok()
        if args[0] == "disable":
            self.enabled.discard(args[1])
            return ok()
        if args[0] == "--no-block":
            self.requests.append(args[1:])
            return ok()
        if args[:2] == ["restart", "--no-block"] and args[2] == hairpin.UNIT:
            self.requests.append(args)
            return ok()
        raise AssertionError("unexpected systemctl " + " ".join(args))

    # Observations for assertions.

    def changes(self):
        return [c for c in self.calls if any(c[:len(p)] == list(p) for p in CHANGING)]

    def restarted(self):
        return [c[3].removeprefix("pci/") for c in self.calls if c[:3] == ["devlink", "dev", "reload"]]

    def parameter_sets(self):
        return [(c[4].removeprefix("pci/"), c[6], c[8]) for c in self.calls if c[:4] == ["devlink", "dev", "param", "set"]]

    def records(self):
        return self.read(hairpin.ATTEMPTS)["records"]

    def apply(self, **overrides):
        return hairpin.apply(host=self.host(**overrides))


def in_effect(spark):
    for f in spark.functions.values():
        f["values"]["hairpin_queue_size"] = rule.HAIRPIN_QUEUE_SIZE
        f["counter"] = max(f["counter"], 1)


# approve

def test_approve_records_the_four_functions_and_changes_nothing_else(tmp_path):
    spark = Spark(tmp_path, approved=False)
    record = hairpin.approve(host=spark.host())
    assert record == spark.approval()
    assert spark.read(hairpin.APPROVAL) == record
    assert (tmp_path / hairpin.APPROVAL.lstrip("/")).stat().st_mode & 0o777 == 0o644
    assert not spark.changes() and not spark.calls


def test_approve_accepts_the_same_record_and_refuses_other_functions(tmp_path):
    spark = Spark(tmp_path)
    assert hairpin.approve(host=spark.host()) == spark.approval()
    other = spark.approval()
    other["functions"][0]["mac"] = "02:00:00:00:00:01"
    spark.write(hairpin.APPROVAL, other)
    with pytest.raises(hairpin.HairpinError, match="already approves.*cw_primary: mac 02:00:00:00:00:01") as error:
        hairpin.approve(host=spark.host())
    assert error.value.kind == "M13"
    assert spark.read(hairpin.APPROVAL) == other


def test_approve_replaces_an_approval_of_other_values_for_the_same_functions(tmp_path):
    spark = Spark(tmp_path)
    older = spark.approval()
    older["parameters"] = {"hairpin_num_queues": 4, "hairpin_queue_size": 1024}
    spark.write(hairpin.APPROVAL, older)
    assert hairpin.approve(host=spark.host())["parameters"] == rule.PARAMETERS
    assert spark.read(hairpin.APPROVAL) == spark.approval()


def test_approve_refuses_a_pair_and_accepts_a_missing_fabric_record(tmp_path):
    pair = Spark(tmp_path / "pair", approved=False, fabric=2)
    with pytest.raises(hairpin.HairpinError, match="describes a 2-Spark setup") as error:
        hairpin.approve(host=pair.host())
    assert error.value.kind == "M17" and not pair.exists(hairpin.APPROVAL)
    fresh = Spark(tmp_path / "fresh", approved=False, fabric=None)
    assert hairpin.approve(host=fresh.host())["functions"] == fresh.approval()["functions"]


def test_approve_requires_every_function_on_mlx5(tmp_path):
    spark = Spark(tmp_path, approved=False)
    (tmp_path / "sys/class/infiniband/roceP2p1s0f1/device/uevent").write_text("DRIVER=other\nPCI_SLOT_NAME=0002:01:00.1\n")
    with pytest.raises(hairpin.HairpinError, match="roceP2p1s0f1.*not mlx5_core"):
        hairpin.approve(host=spark.host())


# Invocation, mode and approval

def test_apply_runs_only_under_its_unit(tmp_path):
    spark = Spark(tmp_path)
    assert spark.apply(environ={}) == 2
    assert spark.apply(environ={"INVOCATION_ID": "f" * 32}) == 2
    assert all(c[0] == "systemctl" for c in spark.calls) and not spark.changes()
    assert not spark.exists(hairpin.STATE) and "runs only as sparkring-hairpin.service" in spark.lines[-1]


def test_dry_run_prints_the_plan_and_changes_nothing(tmp_path):
    spark = Spark(tmp_path)
    spark.control()
    plan = hairpin.preview(host=spark.host(environ={}))
    assert plan["mode"] == "live" and plan["suspended"] is None and plan["restart_needed"]
    assert plan["order"] == [NETDEV[r] for r in ("cw_secondary", "ccw_primary", "cw_primary", "ccw_secondary")]
    assert {f["action"] for f in plan["functions"]} == {"set-and-restart"} and plan["busy"] == []
    in_effect(spark)
    # A boot runs the unit only when it is enabled.
    boot = hairpin.preview(boot=True, host=spark.host(environ={}))
    assert boot["armed"] is False and boot["restart_needed"] is False
    assert {f["action"] for f in boot["functions"]} == {"none (sparkring-hairpin.service is not enabled)"}
    spark.enabled.add(hairpin.UNIT)
    spark.journal({"boot_id": BOOT_ID, "pci_address": BDF["cw_primary"], "netdev": NETDEV["cw_primary"],
                   "state": "failed", "class": "restart", "error": "x", "time": 1.0})
    boot = hairpin.preview(boot=True, host=spark.host(environ={}))
    # The next boot treats this boot's failed record as an earlier boot's.
    assert boot["mode"] == "boot" and boot["suspended"]["pci_address"] == BDF["cw_primary"]
    assert {f["action"] for f in boot["functions"]} == {"none (boot restarts suspended)"}
    assert not spark.changes() and not spark.exists(hairpin.STATE)
    spark.journal()
    boot = hairpin.preview(boot=True, host=spark.host(environ={}))
    assert boot["armed"] is True
    assert {f["action"] for f in boot["functions"]} == {"set-and-restart"} and boot["restart_needed"]


@pytest.mark.parametrize("booting", [True, False])
def test_mode_follows_whether_networkmanager_ever_started_in_this_boot(tmp_path, booting):
    spark = Spark(tmp_path, booting=booting)
    # The fixture reports NetworkManager failed; only the timestamp decides.
    assert hairpin.detect_mode(spark.host()) == ("boot" if booting else "live")
    assert spark.calls == [["systemctl", "show", "-p", "InactiveExitTimestampMonotonic", "--value", "NetworkManager.service"]]


def test_approval_of_other_values_stops_before_any_change(tmp_path):
    spark = Spark(tmp_path, booting=True)
    older = spark.approval()
    older["parameters"] = {"hairpin_num_queues": 4, "hairpin_queue_size": 4096}
    spark.write(hairpin.APPROVAL, older)
    assert spark.apply() == 2
    assert not spark.changes()
    state = spark.read(hairpin.STATE)
    assert state["class"] == "M11" and "approves hairpin_queue_size 4096" in state["error"]


def test_approval_on_a_pair_stops_with_the_revoke_advice(tmp_path):
    spark = Spark(tmp_path, booting=True, fabric=2)
    assert spark.apply() == 2 and not spark.changes()
    assert spark.read(hairpin.STATE)["class"] == "M17"


# Boot runs

def test_boot_run_sets_and_restarts_one_function_at_a_time_then_arms(tmp_path):
    spark = Spark(tmp_path, booting=True)
    assert spark.apply() == 0
    assert spark.parameter_sets() == [(BDF[r], "hairpin_queue_size", "8192") for r in ROLES]
    assert spark.restarted() == [BDF[r] for r in ROLES]
    # Each restart waits for its function before the next one's parameters are set.
    changing = [c for c in spark.calls if c[:2] == ["devlink", "dev"] or c[:4] == ["devlink", "-j", "dev", "show"]]
    for index, role in enumerate(ROLES):
        position = changing.index(["devlink", "dev", "reload", "pci/" + BDF[role], "action", "driver_reinit"])
        assert changing[position + 1] == ["devlink", "-j", "dev", "show", "pci/" + BDF[role]]
        if index + 1 < len(ROLES):
            assert changing.index(["devlink", "dev", "param", "set", "pci/" + BDF[ROLES[index + 1]], "name",
                                   "hairpin_queue_size", "value", "8192", "cmode", "driverinit"]) > position
    assert hairpin.UNIT in spark.enabled
    assert all(f["counter"] == 1 and f["values"]["hairpin_queue_size"] == 8192 for f in spark.functions.values())
    assert [r["state"] for r in spark.records()] == ["applied"] * 4
    assert spark.calls.count(["udevadm", "settle", "--timeout=10"]) == 1


def test_boot_run_checks_no_addresses_gpu_or_networkmanager_and_requests_nothing(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.control()
    assert spark.apply() == 0
    used = {c[0] for c in spark.calls}
    assert not used & {"ip", "nmcli", "nvidia-smi", "ping", "wg"}
    assert spark.requests == []


# Restart outcomes

def test_a_function_in_effect_is_only_read(tmp_path):
    spark = Spark(tmp_path, armed=True)
    in_effect(spark)
    assert spark.apply() == 0
    assert not spark.changes()
    assert not spark.exists(hairpin.ATTEMPTS)


def test_a_pending_value_is_restarted_without_setting_it(tmp_path):
    spark = Spark(tmp_path, booting=True)
    for f in spark.functions.values():
        f["pending"]["hairpin_queue_size"] = 8192
    assert spark.apply() == 0
    assert spark.parameter_sets() == [] and len(spark.restarted()) == 4


def test_a_refused_restart_sets_the_prior_value_back_and_suspends_boot_runs(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.reload[BDF["cw_secondary"]] = "refused"
    assert spark.apply() == 2
    assert spark.restarted() == [BDF["cw_primary"], BDF["cw_secondary"]]
    assert spark.parameter_sets()[-1] == (BDF["cw_secondary"], "hairpin_queue_size", "1024")
    assert spark.functions[BDF["cw_secondary"]]["pending"] == {"hairpin_queue_size": 1024}
    record = spark.records()[-1]
    assert record["state"] == "failed" and record["class"] == "restart" and "Device or resource busy" in record["error"]
    assert "driver restart failed" in spark.read(hairpin.STATE)["error"]
    assert hairpin.UNIT not in spark.enabled
    assert any(line.startswith("<3>SparkRing hairpin: enP2p1s0f0np0 (pci/0002:01:00.0): driver restart failed")
               for line in spark.lines)


def test_a_failed_reload_sets_nothing_back(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.reload[BDF["cw_primary"]] = "broken"
    assert spark.apply() == 2
    assert spark.parameter_sets() == [(BDF["cw_primary"], "hairpin_queue_size", "8192")]
    assert spark.restarted() == [BDF["cw_primary"]] and spark.records()[-1]["state"] == "failed"


def test_a_reload_beyond_its_limit_is_abandoned_and_ends_the_run(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.reload[BDF["cw_primary"]] = "hang"
    started = spark.time
    assert spark.apply() == 2
    assert spark.restarted() == [BDF["cw_primary"]]
    assert spark.time - started < hairpin.RELOAD + 5
    record = spark.records()[-1]
    assert record["state"] == "failed" and record["cause"] == "driver restart did not finish within 30 s"
    assert "driver restart did not finish within 30 s" in record["error"] and "power-cycle it" in record["error"]


def test_a_function_that_does_not_return_gives_m2(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.reload[BDF["cw_primary"]] = "vanish"
    assert spark.apply() == 2
    state = spark.read(hairpin.STATE)
    assert state["class"] == "restart" and "not back within 20 s after its driver restart" in state["error"]
    assert state["cause"] == "not back within 20 s after its driver restart"
    assert state["function"] == "enp1s0f0np0 (pci/0000:01:00.0)"
    assert spark.restarted() == [BDF["cw_primary"]]


def test_an_absent_function_gives_m3_and_a_changed_one_m14(tmp_path):
    spark = Spark(tmp_path / "absent", booting=True)
    spark.remove(BDF["ccw_secondary"])
    assert spark.apply() == 2 and not spark.changes()
    assert spark.read(hairpin.STATE)["class"] == "M3"
    assert "pci/0002:01:00.1 (roceP2p1s0f1) is not present after 15 s" in spark.read(hairpin.STATE)["error"]
    changed = Spark(tmp_path / "changed", booting=True)
    (tmp_path / "changed/sys/bus/pci/devices/0000:01:00.0/net/enp1s0f0np0/address").write_text("02:00:00:00:00:09\n")
    assert changed.apply() == 2 and not changed.changes()
    assert changed.read(hairpin.STATE)["class"] == "M14"
    assert "now carries enp1s0f0np0, MAC 02:00:00:00:00:09, rocep1s0f0" in changed.read(hairpin.STATE)["error"]


def test_unreadable_reload_statistics_restart_nothing(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.stats = False
    assert spark.apply() == 2 and not spark.changes()
    state = spark.read(hairpin.STATE)
    assert state["class"] == "M16" and "devlink reload statistics are unavailable" in state["error"]


def test_the_time_budget_stops_further_restarts(tmp_path):
    spark = Spark(tmp_path, booting=True)
    original = spark.restart

    def slow(argv, timeout):
        spark.time += 19
        return original(argv, timeout)

    spark.restart = slow
    assert spark.apply() == 2
    # 120 s budget, 25 s per restart: the fourth would start with 45 s left.
    assert len(spark.restarted()) == 3
    assert spark.read(hairpin.STATE)["class"] == "budget"


def test_a_signal_that_ends_devlink_after_the_restart_is_not_a_refusal(tmp_path):
    spark = Spark(tmp_path, armed=True)
    spark.reload[BDF["cw_primary"]] = "signal"
    assert spark.apply() == 0
    function = spark.functions[BDF["cw_primary"]]
    # The kernel finished the restart, so nothing is set back and the checks decide.
    assert function["counter"] == 1 and function["values"]["hairpin_queue_size"] == 8192 and function["pending"] == {}
    assert spark.parameter_sets() == [(BDF[r], "hairpin_queue_size", "8192") for r in ROLES]
    assert [r["state"] for r in spark.records()] == ["applied"] * 4
    assert any("devlink reported exit status -15, but the driver restart completed" in line for line in spark.lines)


@pytest.mark.parametrize("behavior,detail", [
    ("uncounted", "driver restarts since boot 0"),
    ("double", "driver restarts since boot 2"),
    ("flagged", "restart failure flag True")])
def test_a_reload_that_succeeds_without_applying_the_setting_fails(tmp_path, behavior, detail):
    spark = Spark(tmp_path, booting=True)
    spark.reload[BDF["cw_primary"]] = behavior
    assert spark.apply() == 2
    record = spark.records()[-1]
    assert record["state"] == "failed" and record["class"] == "restart"
    assert record["cause"].startswith("not in effect after its driver restart (") and detail in record["cause"]
    assert "expected driver restarts since boot 1" in record["cause"]
    assert "Later functions were not restarted" in record["error"]
    assert spark.restarted() == [BDF["cw_primary"]] and hairpin.UNIT not in spark.enabled


# Suspension

@pytest.mark.parametrize("state", ["started", "failed"])
def test_an_unfinished_or_failed_restart_of_an_earlier_boot_suspends_boot_runs(tmp_path, state):
    spark = Spark(tmp_path, booting=True, offload="off")
    spark.journal({"boot_id": EARLIER_BOOT, "mode": "boot", "pci_address": BDF["ccw_primary"],
                   "netdev": NETDEV["ccw_primary"], "state": state, "class": "restart" if state == "failed" else None,
                   "error": "kernel answers: Input/output error" if state == "failed" else None, "time": 1.0})
    assert spark.apply() == 2
    assert spark.restarted() == [] and spark.parameter_sets() == []
    assert all(f["offload"] == "on" for f in spark.functions.values())
    document = spark.read(hairpin.STATE)
    assert document["class"] == "M12" and document["suspended"]["boot_id"] == EARLIER_BOOT
    outcome = "did not finish" if state == "started" else "failed"
    assert f"restart of enp1s0f1np1 (pci/0000:01:00.1) {outcome} during an earlier boot" in document["error"]


def test_a_check_failure_does_not_suspend_boot_runs(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.journal({"boot_id": EARLIER_BOOT, "mode": "live", "pci_address": BDF["cw_primary"],
                   "netdev": NETDEV["cw_primary"], "state": "check-failed", "class": "check", "error": "x", "time": 1.0})
    assert spark.apply() == 0 and len(spark.restarted()) == 4


def test_a_successful_live_run_resolves_earlier_failures(tmp_path):
    spark = Spark(tmp_path)
    spark.journal({"boot_id": EARLIER_BOOT, "mode": "boot", "pci_address": BDF["cw_primary"],
                   "netdev": NETDEV["cw_primary"], "state": "failed", "class": "restart", "error": "x", "time": 1.0},
                  {"boot_id": EARLIER_BOOT, "mode": "boot", "pci_address": BDF["cw_secondary"],
                   "netdev": NETDEV["cw_secondary"], "state": "started", "class": None, "error": None, "time": 2.0})
    assert spark.apply() == 0
    assert [r["state"] for r in spark.records()] == ["resolved", "resolved", "applied", "applied", "applied", "applied"]
    status = hairpin.status(host=spark.host())
    assert status["suspended"] is None and status["armed"] and status["in_effect"]


def test_a_live_run_whose_restart_failed_resolves_nothing_and_does_not_arm(tmp_path):
    spark = Spark(tmp_path)
    spark.journal({"boot_id": EARLIER_BOOT, "mode": "boot", "pci_address": BDF["cw_primary"],
                   "netdev": NETDEV["cw_primary"], "state": "failed", "class": "restart", "error": "x", "time": 1.0})
    last = "pci/" + BDF["ccw_secondary"]
    original = spark.restart

    def restart(argv, timeout):
        result = original(argv, timeout)
        if argv[3] == last:
            # The kernel finished this restart only after the limit.
            spark.time += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        return result

    spark.restart = restart
    assert spark.apply() == 2
    state = spark.read(hairpin.STATE)
    assert state["class"] == "restart" and state["in_effect"] and state["armed"] is None
    assert [r["state"] for r in spark.records()] == ["failed", "applied", "applied", "applied", "failed"]
    assert hairpin.UNIT not in spark.enabled
    status = hairpin.status(host=spark.host())
    assert status["suspended"]["pci_address"] == BDF["ccw_secondary"] and not status["armed"]
    # A Spark armed earlier restarts nothing at its next boot.
    spark.enabled.add(hairpin.UNIT)
    spark.restart, spark.booting, spark.invocation = original, True, "f" * 32
    (tmp_path / "proc/sys/kernel/random/boot_id").write_text("11111111-2222-4333-8444-555555555555\n")
    restarts = len(spark.restarted())
    assert spark.apply() == 2
    assert spark.read(hairpin.STATE)["class"] == "M12" and len(spark.restarted()) == restarts


def test_failures_keep_a_short_cause_that_the_summary_and_m12_reuse(tmp_path):
    spark = Spark(tmp_path, booting=True, armed=True)
    spark.reload[BDF["cw_secondary"]] = "refused"
    assert spark.apply() == 2
    cause = "driver restart failed: Error: devlink: reload failed.; kernel answers: Device or resource busy"
    assert spark.records()[-1]["cause"] == cause
    state = spark.read(hairpin.STATE)
    assert state["cause"] == cause and state["function"] == "enP2p1s0f0np0 (pci/0002:01:00.0)"
    assert spark.lines[-1].startswith("<3>SparkRing hairpin: boot run finished in ")
    assert spark.lines[-1].endswith("; first failure: enP2p1s0f0np0 (pci/0002:01:00.0): " + cause)
    # The next boot's M12 names that cause once, with one remedy.
    (tmp_path / "proc/sys/kernel/random/boot_id").write_text("11111111-2222-4333-8444-555555555555\n")
    spark.invocation = "f" * 32
    assert spark.apply() == 2
    m12 = spark.read(hairpin.STATE)["error"]
    assert m12.count("SparkRing hairpin:") == 1 and m12.count("sudo sparkring hairpin") == 1 and ".." not in m12
    assert "failed during an earlier boot (" in m12 and f"): {cause}. Nothing was restarted" in m12
    assert spark.read(hairpin.STATE)["cause"].startswith("boot restarts suspended since enP2p1s0f0np0 failed to "
                                                         "restart at ")


def test_each_function_line_follows_its_restart_and_precedes_arming(tmp_path):
    spark = Spark(tmp_path, booting=True)
    seen = []
    spark.on_reload = lambda pci: seen.append(sum(": set-and-restart (" in line for line in spark.lines))
    assert spark.apply() == 0
    assert seen == [0, 1, 2, 3]
    enabled = next(i for i, line in enumerate(spark.lines) if "enabled; this Spark applies" in line)
    functions = [i for i, line in enumerate(spark.lines) if ": set-and-restart (" in line]
    assert len(functions) == 4 and max(functions) < enabled == len(spark.lines) - 2


def test_journal_records_are_synced_before_the_restart_and_carry_their_context(tmp_path):
    spark = Spark(tmp_path, booting=True)
    seen = []
    spark.on_reload = lambda pci: seen.append(spark.records()[-1])
    assert spark.apply() == 0
    assert [r["state"] for r in seen] == ["started"] * 4
    record = seen[0]
    assert record["boot_id"] == BOOT_ID and record["mode"] == "boot" and record["driver_reinit_before"] == 0
    assert record["kernel"] == "6.17.0-1029-nvidia" and record["firmware"] == "28.45.4028"
    assert (tmp_path / hairpin.ATTEMPTS.lstrip("/")).stat().st_mode & 0o777 == 0o600


# Order

def test_tunnel_functions_restart_last_with_the_parent_facing_one_at_the_end(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.control()
    assert spark.apply() == 0
    assert spark.restarted() == [BDF[r] for r in ("cw_secondary", "ccw_primary", "cw_primary", "ccw_secondary")]
    head = Spark(tmp_path / "head", booting=True)
    head.control(head=True)
    assert head.apply() == 0
    assert head.restarted() == [BDF[r] for r in ("cw_secondary", "ccw_primary", "ccw_secondary", "cw_primary")]


def test_a_failure_before_the_tunnel_functions_leaves_them_untouched(tmp_path):
    spark = Spark(tmp_path, booting=True)
    spark.control()
    spark.reload[BDF["ccw_primary"]] = "refused"
    assert spark.apply() == 2
    touched = {pci for pci, _, _ in spark.parameter_sets()} | set(spark.restarted())
    assert touched == {BDF["cw_secondary"], BDF["ccw_primary"]}
    for role in ("cw_primary", "ccw_secondary"):
        assert spark.functions[BDF[role]]["values"]["hairpin_queue_size"] == 1024


# Busy check

def busy_case(tmp_path, change, *, booting=False):
    spark = Spark(tmp_path, booting=booting)
    change(spark)
    code = spark.apply()
    return spark, code


@pytest.mark.parametrize("change,expected", [
    (lambda s: s.units.update({"sparkring-mesh.service": {"ActiveState": "activating"}}),
     "sparkring-mesh.service is activating"),
    (lambda s: s.units.update({"sparkring-x-model.service": {"ActiveState": "failed",
                                                             "ControlGroup": "/system.slice/sparkring-x-model.service"}})
     or s.write("/sys/fs/cgroup/system.slice/sparkring-x-model.service/cgroup.procs", 4242),
     "sparkring-x-model.service is failed but its processes 4242 still run"),
    (lambda s: s.tc.update({NETDEV["cw_secondary"]: [{"protocol": "all", "kind": "flower"}]}),
     "1 forwarding rule(s) on enP2p1s0f0np0"),
    (lambda s: s.qp.append({"ifname": "rocep1s0f1", "port": 1, "lqpn": 17, "pid": 3131, "comm": "marker"}),
     "RDMA user marker (PID 3131) on rocep1s0f1"),
    (lambda s: s.pd.append({"ifname": "roceP2p1s0f0", "pid": 77, "comm": "verbs-tool"}),
     "RDMA user verbs-tool (PID 77) on roceP2p1s0f0"),
    (lambda s: setattr(s, "gpu", "5150\n"), "GPU compute process PID 5150"),
    (lambda s: s.functions[BDF["cw_primary"]].update(autoconnect="no"),
     "connection 'mesh-cw_primary' on enp1s0f0np0 would not come back after a restart: autoconnect is not yes"),
    (lambda s: s.functions[BDF["ccw_primary"]].update(bound=""),
     "bound neither to the interface name nor to its MAC"),
])
def test_the_busy_check_stops_a_live_run_before_any_change(tmp_path, change, expected):
    spark, code = busy_case(tmp_path, change)
    assert code == 2 and not spark.changes()
    state = spark.read(hairpin.STATE)
    assert state["class"] == "M4" and expected in state["error"]
    assert state["error"].startswith("SparkRing hairpin: not restarting ConnectX drivers on spark-test: ")


def test_kernel_rdma_objects_and_an_idle_ring_are_not_busy(tmp_path):
    spark, code = busy_case(tmp_path, lambda s: s.qp.append({"ifname": "rocep1s0f0", "type": "GSI", "comm": "ib_core"}))
    assert code == 0 and len(spark.restarted()) == 4


def test_boot_runs_query_no_gpu_and_no_networkmanager_profiles(tmp_path):
    spark, code = busy_case(tmp_path, lambda s: (setattr(s, "gpu", "5150\n"),
                                                 s.functions[BDF["cw_primary"]].update(autoconnect="no")), booting=True)
    assert code == 0 and len(spark.restarted()) == 4


# Live follow-up

def test_a_live_restart_waits_for_addresses_and_brings_the_connection_up_once(tmp_path):
    spark = Spark(tmp_path)
    spark.drop_addresses = True
    assert spark.apply() == 0
    ups = [c for c in spark.calls if c[:3] == ["nmcli", "--wait", "10"]]
    assert [c[-1] for c in ups] == ["uuid-0", "uuid-1", "uuid-2", "uuid-3"]
    assert spark.requests == [["try-restart", "sparkring-fabric.service"]]


def test_addresses_that_do_not_return_give_a_check_failure(tmp_path):
    spark = Spark(tmp_path)
    spark.drop_addresses = True
    spark.functions[BDF["cw_primary"]]["connection"] = "uuid-gone"
    original = spark.run

    def run(argv, **kwargs):
        if argv[:3] == ["nmcli", "--wait", "10"]:
            spark.calls.append(list(argv))
            return fail("Error: Connection activation failed")
        return original(argv, **kwargs)

    assert spark.apply(run=run) == 2
    state = spark.read(hairpin.STATE)
    assert state["class"] == "check" and "but its fabric addresses did not return within 30 s" in state["error"]
    assert spark.records()[-1]["state"] == "check-failed" and spark.restarted() == [BDF["cw_primary"]]
    # The restart removed routes, so the follow-up still runs.
    assert spark.requests == [["try-restart", "sparkring-fabric.service"]]


def test_a_live_restart_waits_for_the_rdma_port_and_gid_index_3(tmp_path):
    spark = Spark(tmp_path, armed=True)
    spark.rdma_down[BDF["cw_primary"]] = 3
    assert spark.apply() == 0
    polls = [c for c in spark.calls if c[:3] == ["ip", "-j", "address"] and c[-1] == NETDEV["cw_primary"]]
    # The busy check, the snapshot, three polls with the port down and one with it back.
    assert len(polls) == 6
    assert [r["state"] for r in spark.records()] == ["applied"] * 4
    assert not [c for c in spark.calls if c[:3] == ["nmcli", "--wait", "10"]]


def test_a_gid_that_does_not_return_is_a_check_failure(tmp_path):
    spark = Spark(tmp_path)
    spark.rdma_down[BDF["cw_primary"]] = -1
    assert spark.apply() == 2
    state = spark.read(hairpin.STATE)
    assert state["class"] == "check" and "restarted, but its fabric addresses did not return within 30 s" in state["error"]
    assert [c[-1] for c in spark.calls if c[:3] == ["nmcli", "--wait", "10"]] == ["uuid-0"]
    assert spark.restarted() == [BDF["cw_primary"]] and spark.records()[-1]["state"] == "check-failed"


def test_the_endpoint_refresh_is_limited_to_the_read_limit(tmp_path):
    spark = Spark(tmp_path)
    spark.control()
    limits = []

    def run(argv, **kwargs):
        if argv[:2] in (["wg", "set"], ["ip", "-j"]):
            limits.append((argv[0], kwargs["timeout"]))
        return spark.run(argv, **kwargs)

    spark.host(run=run).refresh_endpoint(NETDEV["cw_primary"])
    assert limits == [("ip", hairpin.READ), ("wg", hairpin.READ)]


def test_the_follow_up_runs_only_after_a_restart_and_refreshes_control_when_recorded(tmp_path):
    idle = Spark(tmp_path / "idle", armed=True)
    in_effect(idle)
    idle.control()
    assert idle.apply() == 0 and idle.requests == []
    spark = Spark(tmp_path / "restart")
    spark.control()
    assert spark.apply() == 0
    assert spark.requests == [["try-restart", "sparkring-fabric.service"], ["start", "sparkring-control-refresh.service"]]


def test_a_tunnel_function_refreshes_its_own_peer_and_pings_across_it(tmp_path):
    spark = Spark(tmp_path)
    spark.control()
    assert spark.apply() == 0
    wg = [c for c in spark.calls if c[:2] == ["wg", "set"]]
    assert wg == [["wg", "set", "sr-control", "peer", KEY, "endpoint", "[fe80::1%enp1s0f0np0]:51871"],
                  ["wg", "set", "sr-control", "peer", KEY.replace("A", "B", 1), "endpoint", "[fe80::2%enP2p1s0f1np1]:51871"]]
    pings = [c[-1] for c in spark.calls if c[0] == "ping"]
    assert pings == ["10.253.255.3", "10.253.255.1"]
    assert all(c[:8] == ["ping", "-n", "-c", "1", "-W", "1", "-I", "sr-control"] for c in spark.calls if c[0] == "ping")


def test_a_tunnel_that_does_not_answer_gives_m18_naming_the_peer(tmp_path):
    spark = Spark(tmp_path)
    spark.control()
    spark.ping = False
    assert spark.apply() == 2
    state = spark.read(hairpin.STATE)
    assert "the administration tunnel to 10.253.255.3 did not return within 30 s" in state["error"]
    assert state["class"] == "check"
    # The parent-facing function is never restarted after the failure.
    assert BDF["ccw_secondary"] not in spark.restarted()


def test_a_check_failure_on_the_last_function_arms_but_fails_and_starts_no_mesh(tmp_path):
    spark = Spark(tmp_path)
    spark.control()
    spark.unreachable.add("10.253.255.1")
    spark.write(hairpin.BLOCKED + "/sparkring-mesh.service", {"unit": "sparkring-mesh.service"})
    spark.enabled.add("sparkring-mesh.service")
    assert spark.apply() == 2
    state = spark.read(hairpin.STATE)
    assert state["in_effect"] and state["armed"] and hairpin.UNIT in spark.enabled
    assert state["class"] == "check" and "administration tunnel to 10.253.255.1" in state["error"]
    assert spark.records()[-1]["state"] == "check-failed"
    assert not [r for r in spark.requests if r[0] == "start" and r[1] == "sparkring-mesh.service"]
    assert spark.exists(hairpin.BLOCKED + "/sparkring-mesh.service")


def test_a_failed_child_facing_restart_advises_a_reboot_of_this_spark(tmp_path):
    worker = Spark(tmp_path / "worker", booting=True)
    worker.control()
    worker.reload[BDF["cw_primary"]] = "refused"
    assert worker.apply() == 2
    state = worker.read(hairpin.STATE)
    assert state["class"] == "restart" and state["cut_off"] == NETDEV["cw_primary"]
    assert "The Sparks behind enp1s0f0np0 are cut off from the administration network" in state["error"]
    assert ("reboot this Spark (power-cycle it if the reboot hangs), then run sudo sparkring hairpin on Node A"
            in state["error"])
    assert hairpin.reboot_advice(root=tmp_path / "worker").startswith("reboot this Spark (enp1s0f0np0 failed to restart")
    head = Spark(tmp_path / "head", booting=True)
    head.control(head=True)
    head.reload[BDF["cw_primary"]] = "refused"
    assert head.apply() == 2
    assert "The workers behind enp1s0f0np0 are cut off" in head.read(hairpin.STATE)["error"]
    assert "reboot Node A" in head.read(hairpin.STATE)["error"]
    assert hairpin.reboot_advice(root=tmp_path / "head").startswith("reboot Node A (enp1s0f0np0 failed")
    # A failed function without an administration tunnel keeps the retry advice.
    plain = Spark(tmp_path / "plain", booting=True)
    plain.control()
    plain.reload[BDF["cw_secondary"]] = "refused"
    assert plain.apply() == 2
    assert plain.read(hairpin.STATE)["cut_off"] is None
    assert plain.read(hairpin.STATE)["error"].endswith("On Node A, sudo sparkring hairpin retries it.")
    assert hairpin.reboot_advice(root=tmp_path / "plain") is None
    # After the reboot the failed record belongs to an earlier boot.
    (tmp_path / "worker/proc/sys/kernel/random/boot_id").write_text(EARLIER_BOOT + "\n")
    assert hairpin.reboot_advice(root=tmp_path / "worker") is None


def test_a_child_facing_failure_at_boot_keeps_the_administration_path_to_node_a(tmp_path):
    from runtime.host import control_node
    spark = Spark(tmp_path, booting=True, armed=True)
    spark.control()
    pci, netdev = BDF["cw_primary"], NETDEV["cw_primary"]

    def gone(reloaded):
        if reloaded == pci:
            # reload_failed: the driver removed the netdev and did not add it again.
            shutil.rmtree(tmp_path / "sys/bus/pci/devices" / pci / "net" / netdev)
            shutil.rmtree(tmp_path / "sys/class/net" / netdev)

    spark.on_reload = gone
    spark.reload[pci] = "broken"
    assert spark.apply() == 2
    assert BDF["ccw_secondary"] not in spark.restarted()
    # Later in this boot, sparkring-control.service runs control-up before sr-control exists.
    (tmp_path / "etc/sparkring/control.key").write_text(base64.b64encode(b"k" * 32).decode() + "\n")
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:4] == ["ip", "-j", "link", "show"]:
            return subprocess.CompletedProcess(argv, 1, "", 'Device "sr-control" does not exist.')
        if argv[:3] == ["ip", "-j", "-6"]:
            return spark.run(argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(ValueError, match="enp1s0f0np0: interface is not present"):
        control_node.up(root=tmp_path, run=run)
    [up] = [call for call in calls if call[0] == "wg-quick"]
    path = Path(up[2])
    assert up[:2] == ["wg-quick", "up"] and path == tmp_path / "run/sparkring-control/sr-control.conf"
    config = path.read_text()
    assert "Endpoint = [fe80::2%enP2p1s0f1np1]:51871" in config and "%enp1s0f0np0" not in config
    assert config.count("[Peer]") == 2 and path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    # The Spark stays reachable: forwarding and firewall rules are applied before the failure is raised.
    assert ["sysctl", "-w", "net.ipv4.conf.sr-control.forwarding=1"] in calls
    assert any(call[:2] == ["iptables", "-w"] for call in calls)


def test_an_unreadable_control_record_is_a_busy_finding(tmp_path):
    spark = Spark(tmp_path)
    (tmp_path / hairpin.CONTROL.lstrip("/")).write_text("{")
    findings = hairpin.busy_findings(spark.host(), list(spark.approval()["functions"]), "live")
    assert [f["kind"] for f in findings] == ["network"] and "administration network record" in findings[0]["detail"]


# Arming and resuming refused mesh units

def test_arming_happens_only_when_every_function_is_in_effect(tmp_path):
    spark = Spark(tmp_path, offload="off [fixed]")
    assert spark.apply() == 2
    assert hairpin.UNIT not in spark.enabled
    assert spark.read(hairpin.STATE)["class"] == "M20"
    assert any("hardware TC offload is off and fixed on enp1s0f0np0" in line for line in spark.lines)


def test_a_run_starts_no_refused_mesh_unit(tmp_path):
    spark = Spark(tmp_path)
    spark.write(hairpin.BLOCKED + "/sparkring-mesh.service", {"unit": "sparkring-mesh.service"})
    spark.enabled.add("sparkring-mesh.service")
    assert spark.apply() == 0
    # Only the ring procedure starts refused units, once every Spark's run succeeded.
    assert not [r for r in spark.requests if r[0] == "start" and r[1] == "sparkring-mesh.service"]
    assert spark.exists(hairpin.BLOCKED + "/sparkring-mesh.service")
    assert "started_units" not in spark.read(hairpin.STATE)


def test_resume_starts_refused_units_only_when_enabled_and_inactive(tmp_path):
    spark = Spark(tmp_path)
    for name in ("sparkring-mesh.service", "sparkring-x-mesh.service", "sparkring-y-mesh.service",
                 "sparkring-agent.service"):
        spark.write(hairpin.BLOCKED + "/" + name, {"unit": name})
    spark.enabled |= {"sparkring-mesh.service", "sparkring-y-mesh.service", "sparkring-agent.service"}
    spark.units["sparkring-y-mesh.service"] = {"ActiveState": "active"}
    result = hairpin.resume(host=spark.host())
    assert result["started"] == ["sparkring-mesh.service"]
    assert [entry["unit"] for entry in result["skipped"]] == ["sparkring-x-mesh.service", "sparkring-y-mesh.service"]
    assert [r for r in spark.requests if r[0] == "start"] == [["start", "sparkring-mesh.service"]]
    assert list((tmp_path / hairpin.BLOCKED.lstrip("/")).iterdir()) == []
    assert not [c for c in spark.changes() if c[:1] == ["devlink"]]


# Starting the unit from the ring procedure

@pytest.mark.parametrize("unit,after,started", [
    ({"ActiveState": "inactive"}, "", True),
    ({"ActiveState": "failed"}, INVOCATION, True),
    ({"ActiveState": "activating"}, INVOCATION, False),
    ({"ActiveState": "active"}, "0" * 32, False)])
def test_start_restarts_the_unit_only_while_no_newer_run_exists(tmp_path, unit, after, started):
    spark = Spark(tmp_path)
    spark.hairpin_unit = {**unit, "Result": "success"}
    if unit["ActiveState"] == "inactive":
        spark.invocation = ""
    result = hairpin.start(after, host=spark.host())
    assert result["started"] is started
    assert ([r for r in spark.requests if r[:2] == ["restart", "--no-block"]] != []) is started


def test_offload_is_turned_on_where_it_can_be(tmp_path):
    spark = Spark(tmp_path, offload="off", armed=True)
    in_effect(spark)
    assert spark.apply() == 0
    assert [c for c in spark.changes()] == [["ethtool", "-K", NETDEV[r], "hw-tc-offload", "on"] for r in ROLES]


# Signals

def test_a_stop_request_finishes_the_current_function_and_starts_no_other(tmp_path):
    spark = Spark(tmp_path, booting=True)
    host = spark.host()
    spark.on_reload = lambda pci: host.stop()
    assert hairpin.apply(host=host) == 2
    assert spark.restarted() == [BDF["cw_primary"]]
    assert spark.records()[-1]["state"] == "applied"
    state = spark.read(hairpin.STATE)
    assert state["class"] == "signal" and state["stopped_on_request"] and hairpin.UNIT not in spark.enabled


def test_a_stop_during_the_last_live_restart_still_checks_once_and_requests_the_follow_up(tmp_path):
    spark = Spark(tmp_path, armed=True)
    spark.control()
    host = spark.host()
    spark.on_reload = lambda pci: host.stop() if pci == BDF["ccw_secondary"] else None
    assert hairpin.apply(host=host) == 0
    assert "[fe80::2%enP2p1s0f1np1]:51871" in [c[-1] for c in spark.calls if c[:2] == ["wg", "set"]]
    assert spark.requests == [["try-restart", "sparkring-fabric.service"],
                              ["start", "sparkring-control-refresh.service"]]
    state = spark.read(hairpin.STATE)
    assert state["stopped_on_request"] and state["error"] is None and state["armed"]


def test_a_stop_that_ends_a_check_is_the_runs_error(tmp_path):
    spark = Spark(tmp_path, armed=True)
    spark.control()
    spark.unreachable.add("10.253.255.1")
    host = spark.host()
    spark.on_reload = lambda pci: host.stop() if pci == BDF["ccw_secondary"] else None
    assert hairpin.apply(host=host) == 2
    state = spark.read(hairpin.STATE)
    assert state["class"] == "signal" and state["in_effect"]
    assert ("enP2p1s0f1np1 was restarted, but the stop ended the check of the administration tunnel to "
            "10.253.255.1") in state["error"]
    assert spark.records()[-1]["state"] == "applied"
    # The endpoint was still set again, and the tunnel was pinged once instead of for 30 s.
    assert "[fe80::2%enP2p1s0f1np1]:51871" in [c[-1] for c in spark.calls if c[:2] == ["wg", "set"]]
    assert len([c for c in spark.calls if c[0] == "ping" and c[-1] == "10.253.255.1"]) == 1
    assert spark.requests == [["try-restart", "sparkring-fabric.service"],
                              ["start", "sparkring-control-refresh.service"]]


def test_a_stop_during_the_address_wait_records_the_unchecked_addresses(tmp_path):
    spark = Spark(tmp_path)
    spark.drop_addresses = True
    host = spark.host()
    spark.on_reload = lambda pci: host.stop()
    assert hairpin.apply(host=host) == 2
    assert spark.restarted() == [BDF["cw_primary"]]
    assert not [c for c in spark.calls if c[:3] == ["nmcli", "--wait", "10"]]
    state = spark.read(hairpin.STATE)
    assert state["class"] == "signal" and "the stop ended the check of its fabric addresses" in state["error"]
    assert spark.requests == [["try-restart", "sparkring-fabric.service"]]


def test_a_stop_before_arming_leaves_the_unit_disabled_and_fails(tmp_path):
    spark = Spark(tmp_path, booting=True)
    host = spark.host()
    original = spark.systemctl

    def systemctl(args):
        if args[0] == "is-enabled":
            host.stop()
        return original(args)

    spark.systemctl = systemctl
    assert hairpin.apply(host=host) == 2
    state = spark.read(hairpin.STATE)
    assert hairpin.UNIT not in spark.enabled and state["in_effect"] and state["armed"] is False
    assert state["class"] == "signal" and "sparkring-hairpin.service was not enabled" in state["error"]


def test_the_unit_entry_point_turns_sigterm_into_a_stop_request(monkeypatch):
    import signal
    installed = []
    monkeypatch.setattr(hairpin.signal, "signal", lambda number, handler: installed.append((number, handler)))
    monkeypatch.setattr(hairpin, "apply", lambda *, host: host.stopping or installed[0][1](signal.SIGTERM, None) or host.stopping)
    assert hairpin.apply_unit() is True
    assert installed[0][0] == signal.SIGTERM and installed[-1][0] == signal.SIGTERM


def test_a_hung_child_is_killed_without_waiting_for_its_output():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        hairpin.execute([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert time.monotonic() - started < 10
    assert hairpin.execute([sys.executable, "-c", "print('x')"], timeout=30).stdout == "x\n"


# require

def test_require_passes_without_a_four_spark_record(tmp_path):
    missing = Spark(tmp_path / "missing", fabric=None)
    assert hairpin.require("sparkring-mesh.service", host=missing.host()) == 0 and missing.calls == []
    pair = Spark(tmp_path / "pair", fabric=2)
    assert hairpin.require("sparkring-mesh.service", host=pair.host()) == 0 and pair.calls == []


def test_require_refuses_a_function_at_the_default_and_records_the_unit(tmp_path):
    spark = Spark(tmp_path, approved=False)
    in_effect(spark)
    spark.functions[BDF["cw_secondary"]]["values"]["hairpin_queue_size"] = 1024
    assert hairpin.require("sparkring-x-mesh.service", host=spark.host()) == 2
    marker = tmp_path / hairpin.BLOCKED.lstrip("/") / "sparkring-x-mesh.service"
    assert json.loads(marker.read_text())["functions"] == [NETDEV["cw_secondary"]]
    line = spark.lines[-1]
    assert line.startswith("<3>SparkRing hairpin: sparkring-x-mesh.service not started: the ConnectX hairpin setting "
                           "is not in effect on enP2p1s0f0np0 (pci/0002:01:00.0): hairpin_queue_size 1024, required "
                           "8192.")
    assert "This Spark has no SparkRing approval for the setting." in line
    assert line.endswith("this unit then starts by itself if it is enabled.")
    assert not spark.changes()
    spark.functions[BDF["cw_secondary"]]["values"]["hairpin_queue_size"] = 8192
    assert hairpin.require("sparkring-x-mesh.service", host=spark.host()) == 0 and not marker.exists()


def test_require_names_a_suspension_or_a_failed_boot_run(tmp_path):
    spark = Spark(tmp_path)
    spark.journal({"boot_id": EARLIER_BOOT, "pci_address": BDF["cw_primary"], "netdev": NETDEV["cw_primary"],
                   "state": "started", "time": 1.0})
    assert hairpin.require("sparkring-mesh.service", host=spark.host()) == 2
    assert "Boot restarts are suspended after a failed restart in an earlier boot." in spark.lines[-1]
    spark.journal()
    spark.hairpin_unit["Result"] = "exit-code"
    assert hairpin.require("sparkring-mesh.service", host=spark.host()) == 2
    assert "Its boot run failed; see journalctl -b -u sparkring-hairpin.service." in spark.lines[-1]


def test_require_fails_closed_when_devlink_cannot_be_read(tmp_path):
    spark = Spark(tmp_path)
    in_effect(spark)
    spark.stats = False
    assert hairpin.require("sparkring-mesh.service", host=spark.host()) == 2
    assert "driver restarts since boot unavailable" in spark.lines[-1]
    assert (tmp_path / hairpin.BLOCKED.lstrip("/") / "sparkring-mesh.service").exists()


def test_a_boot_with_the_kernel_option_names_it_in_m5_and_status(tmp_path):
    spark = Spark(tmp_path)
    (tmp_path / "proc/cmdline").write_text("BOOT_IMAGE=/vmlinuz ro quiet sparkring.hairpin=off\n")
    assert hairpin.require("sparkring-mesh.service", host=spark.host()) == 2
    line = spark.lines[-1]
    assert line.endswith("This boot was started with sparkring.hairpin=off, so sparkring-hairpin.service does not run "
                         "in it; reboot without it.")
    assert "sudo sparkring hairpin applies it" not in line
    status = hairpin.status(host=spark.host())
    assert status["boot_disabled_by_kernel_command_line"]
    assert ("sparkring-hairpin.service does not run in this boot (sparkring.hairpin=off on the kernel command line); "
            "reboot without the option to apply the ConnectX hairpin setting") in status["warnings"]


def test_error_lines_carry_the_journal_priority_only_under_systemd(tmp_path):
    lines = []
    terminal = hairpin.Host(root=tmp_path, output=lines.append, environ={}, hostname="x")
    terminal.error("SparkRing hairpin: refused")
    service = hairpin.Host(root=tmp_path, output=lines.append, environ={"JOURNAL_STREAM": "8:1"}, hostname="x")
    service.error("SparkRing hairpin: refused")
    assert lines == ["SparkRing hairpin: refused", "<3>SparkRing hairpin: refused"]
    # With the default output, stdout itself must be the journal stream.
    assert hairpin._journal_stream({"JOURNAL_STREAM": "0:0"}, None) is False


def test_require_rejects_a_unit_name_that_is_not_a_service():
    with pytest.raises(hairpin.HairpinError):
        hairpin.require("../../etc/passwd", host=SimpleNamespace())


# state.json, status and revoke

def test_the_state_record_names_the_boot_the_invocation_and_each_function(tmp_path):
    spark = Spark(tmp_path, booting=True)
    assert spark.apply() == 0
    state = spark.read(hairpin.STATE)
    assert state["schema"] == hairpin.STATE_SCHEMA and state["boot_id"] == BOOT_ID and state["invocation_id"] == INVOCATION
    assert state["mode"] == "boot" and state["in_effect"] and state["armed"] and state["error"] is None
    assert [f["netdev"] for f in state["functions"]] == [NETDEV[r] for r in ROLES]
    assert all(f["action"] == "set-and-restart" and f["after"]["state"] == "in-effect" for f in state["functions"])
    lines = [line for line in spark.lines if "in effect, 4 restarted" in line]
    assert lines and lines[0].startswith("SparkRing hairpin: boot run finished in ")


def test_status_reports_approval_arming_units_and_function_states(tmp_path):
    spark = Spark(tmp_path)
    spark.functions[BDF["ccw_primary"]]["pending"]["hairpin_queue_size"] = 8192
    spark.write(hairpin.BLOCKED + "/sparkring-mesh.service", {})
    status = hairpin.status(host=spark.host(), busy=True)
    assert status["schema"] == hairpin.STATUS_SCHEMA and status["approval"] == {"present": True, "valid": True, "error": None}
    assert status["armed"] is False and status["in_effect"] is False and status["suspended"] is None
    assert status["unit"] == {"ActiveState": "activating", "Result": "success", "InvocationID": INVOCATION}
    assert [row["state"] for row in status["functions"]] == ["default", "default", "pending", "default"]
    assert status["functions"][0]["values"] == {"hairpin_num_queues": 4, "hairpin_queue_size": 1024}
    assert status["blocked_units"] == ["sparkring-mesh.service"] and status["busy"] == []
    assert status["boot_disabled_by_kernel_command_line"] is False and status["last_run"] is None
    assert not spark.changes()


def test_status_from_inventory_facts_runs_no_devlink(tmp_path):
    spark = Spark(tmp_path, armed=True)
    facts = {"rdma": [], "interfaces": []}
    for pci, f in spark.functions.items():
        facts["rdma"].append({"device": f["rdma"], "netdev": f["netdev"], "pci_address": pci, "devlink": {
            "parameters": {"hairpin_queue_size": {"value": 8192}, "hairpin_num_queues": {"value": 4}},
            "reload": {"driver_reinit": 1, "failed": False}}})
        facts["interfaces"].append({"name": f["netdev"], "hw_tc_offload": True, "hw_tc_offload_fixed": False})
    status = hairpin.status(facts=facts, host=spark.host())
    assert status["in_effect"] and status["armed"] and status["warnings"] == []
    assert not any(c[0] in ("devlink", "ethtool") for c in spark.calls)
    del facts["rdma"][0]["devlink"]["reload"]
    assert hairpin.status(facts=facts, host=spark.host())["functions"][0]["state"] == "unknown"


def test_status_warns_about_a_setting_that_is_not_armed_and_a_suspension(tmp_path):
    spark = Spark(tmp_path)
    in_effect(spark)
    spark.journal({"boot_id": EARLIER_BOOT, "pci_address": BDF["cw_primary"], "state": "failed", "time": 1.0})
    status = hairpin.status(host=spark.host())
    assert status["warnings"] == [
        "ConnectX hairpin setting is in effect but not applied at boot; on Node A: sudo sparkring hairpin",
        "boot restarts suspended since pci/0000:01:00.0 failed to restart at 1970-01-01 00:00:01 UTC; on Node A: "
        "sudo sparkring hairpin retries it live"]
    spark.journal({"boot_id": EARLIER_BOOT, "pci_address": BDF["cw_primary"], "netdev": NETDEV["cw_primary"],
                   "state": "failed", "cause": "driver restart failed: busy", "time": 1.0})
    assert hairpin.status(host=spark.host())["warnings"][1] == (
        "boot restarts suspended since enp1s0f0np0 failed to restart at 1970-01-01 00:00:01 UTC (driver restart "
        "failed: busy); on Node A: sudo sparkring hairpin retries it live")
    pair = Spark(tmp_path / "pair", fabric=2)
    assert hairpin.status(host=pair.host())["warnings"] == [
        "hairpin approval on a Spark that is not in a four-Spark ring: sudo sparkring node hairpin revoke"]


def test_status_without_approval_reads_the_fabric_record_functions(tmp_path):
    spark = Spark(tmp_path, approved=False)
    status = hairpin.status(host=spark.host())
    assert status["approval"] == {"present": False, "valid": False, "error": None}
    assert status["function_source"] == "fabric" and [row["pci_address"] for row in status["functions"]] == [BDF[r] for r in ROLES]


def test_revoke_disables_the_service_and_removes_the_records(tmp_path):
    spark = Spark(tmp_path, armed=True)
    spark.journal()
    result = hairpin.revoke(host=spark.host())
    assert result["removed"] == [hairpin.APPROVAL, hairpin.ATTEMPTS]
    assert hairpin.UNIT not in spark.enabled and not spark.exists(hairpin.APPROVAL)
    assert not any(c[:3] == ["devlink", "dev", "reload"] for c in spark.calls)


# Interfaces shared with other modules

def test_roles_and_devices_match_the_planner_and_topology():
    assert hairpin.ROLES == deploy_network.ROLES
    assert set(topology.DEVICES) == set(hairpin.ROLES)


def test_node_cli_dispatches_hairpin_commands(monkeypatch, capsys):
    from scripts import sparkring_node
    monkeypatch.setattr(sparkring_node.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(hairpin, "status", lambda busy=False: {"busy_requested": busy})
    assert sparkring_node.main(["hairpin", "status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"busy_requested": False}
    assert sparkring_node.main(["hairpin", "status", "--busy"]) == 2
    assert "requires sudo" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        sparkring_node.main(["hairpin", "apply", "--boot"])
    monkeypatch.setattr(sparkring_node.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(hairpin, "require", lambda unit: 2 if unit == "sparkring-mesh.service" else 0)
    assert sparkring_node.main(["hairpin", "require", "--unit", "sparkring-mesh.service"]) == 2
    monkeypatch.setattr(hairpin, "apply_unit", lambda: 0)
    monkeypatch.setattr(hairpin, "preview", lambda boot=False: {"boot": boot})
    assert sparkring_node.main(["hairpin", "apply"]) == 0
    assert sparkring_node.main(["hairpin", "apply", "--dry-run", "--boot"]) == 0
    assert json.loads(capsys.readouterr().out) == {"boot": True}
    monkeypatch.setattr(hairpin, "start", lambda after: {"after": after})
    assert sparkring_node.main(["hairpin", "start", "--after", ""]) == 0
    assert json.loads(capsys.readouterr().out) == {"after": ""}
    monkeypatch.setattr(hairpin, "resume", lambda: {"started": ["sparkring-mesh.service"]})
    assert sparkring_node.main(["hairpin", "resume"]) == 0
    assert json.loads(capsys.readouterr().out) == {"started": ["sparkring-mesh.service"]}

    def refused():
        raise hairpin.HairpinError(hairpin.PREFIX + "this Spark's fabric record describes a 2-Spark setup")
    monkeypatch.setattr(hairpin, "approve", refused)
    assert sparkring_node.main(["hairpin", "approve"]) == 2
    # The hairpin message keeps its own prefix only.
    assert capsys.readouterr().err == "SparkRing hairpin: this Spark's fabric record describes a 2-Spark setup\n"


def test_package_launcher_runs_node_commands_without_the_operator_cli(tmp_path):
    launcher = (Path(__file__).resolve().parents[2] / "packaging/debian/sparkring").read_text()
    repository = str(Path(__file__).resolve().parents[2])
    program = (
        "import sys\n"
        f"sys.argv = ['sparkring', 'node', 'hairpin', 'require', '--help']\n"
        "try:\n"
        f"    exec(compile({launcher.replace('/usr/lib/sparkring', repository)!r}, 'sparkring', 'exec'))\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('operator-cli' if 'scripts.sparkring' in sys.modules else 'node-only')\n")
    result = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=120)
    assert result.stdout.strip().splitlines()[-1] == "node-only", result.stderr


def test_node_commands_that_print_json_keep_log_lines_off_stdout(monkeypatch, capsys):
    from scripts import sparkring_node
    monkeypatch.setattr(sparkring_node.os, "geteuid", lambda: 0, raising=False)

    def resume():
        # As Host.log does with its default output.
        print(hairpin.PREFIX + "started sparkring-mesh.service, which its start check refused earlier in this boot")
        return {"started": ["sparkring-mesh.service"], "skipped": []}
    monkeypatch.setattr(hairpin, "resume", resume)
    assert sparkring_node.main(["hairpin", "resume"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == {"started": ["sparkring-mesh.service"], "skipped": []}
    assert "started sparkring-mesh.service" in out.err
