"""CPU contracts for ownership-safe mesh networking; no host commands run."""
import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


network = module("managed_network_test_subject", "managed_network.py")
example = module("managed_network_examples", "make_example.py")


def rule_entry(rule):
    mac = bytes.fromhex(rule.rewrite_ethernet_destination.replace(":", ""))
    def edit(offset, val, mask):
        return {"htype": "eth", "cmd": "set", "offset": offset, "val": val, "mask": mask}
    return {"pref": rule.preference, "kind": "flower", "options": {
        "handle": rule.handle, "skip_sw": True, "in_hw": True,
        "keys": {"src_mac": rule.match_ethernet_source, "dst_mac": rule.match_ethernet_destination,
                 "eth_type": "88b5"}, "actions": [
            {"kind": "pedit", "keys": [edit(12, 0x08000000, 0xFFFF)]},
            {"kind": "pedit", "keys": [edit(0, int.from_bytes(mac[:4], "big"), 0),
                                        edit(4, int.from_bytes(mac[4:] + b"\0\0", "big"), 0xFFFF)]},
            {"kind": "mirred", "to_dev": rule.egress_netdev, "mirred_action": "redirect"}]}}


class FakeHost:
    def __init__(self):
        self.inventory = {}
        self.commands = []
        self.fail_add = None
        self.qdisc_extra = False
        self.mtu = 4096

    def attach(self, manager):
        self.manager = manager
        self.by_add = {tuple(manager._argv(kind, obj, True)): key
                       for key, (kind, obj) in manager.objects.items()}
        self.by_del = {tuple(manager._argv(kind, obj, False)): key
                       for key, (kind, obj) in manager.objects.items()}

    def value(self, kind, obj):
        if kind == "route":
            result = {"dst": obj.destination_ipv4, "dev": obj.source_netdev, "prefsrc": obj.source_ipv4}
            if obj.gateway_ipv4:
                result["gateway"] = obj.gateway_ipv4
            else:
                result["scope"] = "link"
            return result
        if kind == "neighbor":
            return {"lladdr": obj.next_hop_mac, "state": ["PERMANENT"]}
        if kind == "qdisc":
            return {"kind": "clsact", "handle": "ffff:"}
        return rule_entry(obj)

    def __call__(self, argv):
        self.commands.append(argv)
        if argv[0] == "ibv_devinfo":
            return f"\tactive_mtu: {self.mtu} (5)\n"
        key = self.by_add.get(tuple(argv))
        if key:
            if key == self.fail_add:
                raise RuntimeError("Injected add failure")
            assert key not in self.inventory
            self.inventory[key] = self.value(*self.manager.objects[key])
            return ""
        key = self.by_del.get(tuple(argv))
        if key:
            del self.inventory[key]
            return ""
        if "addr" in argv:
            device = argv[-1]
            if device == self.manager.local.management_netdev:
                return json.dumps([{"addr_info": [{"local": self.manager.site["management_addresses"][0]}]}])
            port = next(p for p in self.manager.local.ports if p.netdev == device)
            return json.dumps([{"mtu": 9000, "address": port.mac, "addr_info": [{"local": port.ipv4}]}])
        rows = []
        for key, (kind, obj) in self.manager.objects.items():
            if key not in self.inventory:
                continue
            if (("route" in argv and kind == "route" and argv[-1] == obj.destination_ipv4 + "/32")
                    or ("neigh" in argv and kind == "neighbor" and obj.destination_ipv4 in argv)
                    or ("qdisc" in argv and kind == "qdisc" and argv[-1] == obj)
                    or ("filter" in argv and kind == "rule" and argv[-2:] == [obj.ingress_netdev, "ingress"])):
                rows.append(self.inventory[key])
        if self.qdisc_extra and "filter" in argv and argv[-1] == "egress":
            rows.append({"pref": 99, "kind": "bpf"})
        return json.dumps(rows)

    def read_text(self, path):
        port = next(p for p in self.manager.local.ports if p.rdma_device in path.parts)
        if "gids" in path.parts:
            return "::ffff:" + port.ipv4
        if "ndevs" in path.parts:
            return port.netdev
        if path.name == "active_mtu":
            return "4096 (5)"
        raise AssertionError(path)


@pytest.fixture
def rig(tmp_path):
    (tmp_path / "fabric.example.json").write_text(json.dumps(example.topology_example()))
    site = tmp_path / "site.json"
    site.write_text(json.dumps(example.site_example()))
    host = FakeHost()
    manager = network.NetworkManager(site, 0, tmp_path / "state", runner=host,
                                     read_text=host.read_text, require_root=False)
    host.attach(manager)
    return manager, host


def test_create_verify_remove_only_owned(rig):
    manager, host = rig
    assert manager.up()["ready"]
    assert set(manager.journal["objects"].values()) == {"owned"}
    assert manager.check()["ready"]
    assert manager.down()["clean"]
    assert not host.inventory
    assert not any("flush" in argv or "replace" in argv for argv in host.commands)


def test_adopted_objects_survive_down(rig):
    manager, host = rig
    host.inventory = {key: host.value(*value) for key, value in manager.objects.items()}
    before = dict(host.inventory)
    assert set(manager.up()["ownership"].values()) == {"adopted"}
    assert manager.down()["clean"]
    assert host.inventory == before


def test_second_up_preserves_ownership(rig):
    manager, host = rig
    manager.up()
    manager.up()
    assert set(manager.journal["objects"].values()) == {"owned"}
    assert manager.down()["clean"]
    assert not host.inventory


def test_partial_failure_keeps_created_objects_journaled(rig):
    manager, host = rig
    host.fail_add = list(manager.objects)[1]
    with pytest.raises(RuntimeError, match="Injected"):
        manager.up()
    saved = json.loads(manager.journal_path.read_text())["objects"]
    assert saved[list(manager.objects)[0]] == "owned"
    assert saved[host.fail_add] == "pending"
    result = manager.down()
    assert not host.inventory
    assert result["warnings"]


def test_pending_exact_object_is_adopted_not_deleted(rig):
    manager, host = rig
    manager.up()
    key = next(iter(manager.objects))
    data = json.loads(manager.journal_path.read_text())
    data["objects"][key] = "pending"
    manager.journal_path.write_text(json.dumps(data))
    assert manager.up()["ownership"][key] == "adopted"
    manager.down()
    assert key in host.inventory


def test_changed_owned_route_is_retained(rig):
    manager, host = rig
    manager.up()
    key = next(key for key in host.inventory if key.startswith("route:"))
    host.inventory[key]["gateway"] = "192.0.2.254"
    assert not manager.down()["clean"]
    assert key in host.inventory


def test_conflicting_route_refuses_before_mutation(rig):
    manager, host = rig
    key = next(iter(manager.objects))
    host.inventory[key] = {"dev": "wrong"}
    with pytest.raises(ValueError, match="Conflicting route"):
        manager.up()
    assert not any(tuple(argv) in host.by_add for argv in host.commands)


@pytest.mark.parametrize("field", ["pref", "handle"])
def test_conflicting_tc_identity_refuses_before_mutation(rig, field):
    manager, host = rig
    key = next(k for k in manager.objects if k.startswith("rule:"))
    row = host.value(*manager.objects[key])
    if field == "pref":
        row["pref"] += 1
    else:
        row["options"]["handle"] += 1
    host.inventory[key] = row
    with pytest.raises(ValueError):
        manager.up()
    assert not any(tuple(argv) in host.by_add for argv in host.commands)


def test_nonempty_qdisc_retained(rig):
    manager, host = rig
    manager.up()
    host.qdisc_extra = True
    result = manager.down()
    assert not result["clean"]
    assert all(k.startswith("qdisc:") for k in host.inventory)


def test_plan_mismatch_rejected(rig):
    manager, host = rig
    manager.up()
    data = json.loads(manager.journal_path.read_text())
    data["plan_sha256"] = "0" * 64
    manager.journal_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="canonical plan"):
        manager.down()
    assert host.inventory


def test_journal_cannot_supply_commands(rig):
    manager, host = rig
    manager.up()
    data = json.loads(manager.journal_path.read_text())
    data["objects"]["arbitrary-command"] = "owned"
    manager.journal_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="canonical plan"):
        manager.down()


@pytest.mark.parametrize("target", ["gids", "ndevs"])
def test_device_gid_netdev_mtu_required(rig, target):
    manager, host = rig
    manager.read_text = lambda path: "0" if target in path.parts else host.read_text(path)
    with pytest.raises(ValueError):
        manager.up()
    assert not host.inventory


def test_roce_mtu_required(rig):
    manager, host = rig
    host.mtu = 1024
    with pytest.raises(ValueError, match="RoCE MTU"):
        manager.up()
    assert not host.inventory


def test_check_does_not_create_state_or_call_marker(rig):
    manager, host = rig
    host.inventory = {key: host.value(*value) for key, value in manager.objects.items()}
    assert manager.check()["ready"]
    assert not manager.state_dir.exists()
    assert all(argv[0] in ("tc", "ip", "ibv_devinfo") for argv in host.commands)


def test_missing_object_fails_check(rig):
    manager, _ = rig
    with pytest.raises(ValueError, match="Missing mesh"):
        manager.check()


def test_fast_check_skips_only_verbs_probe_and_detects_tc_drift(rig):
    manager, host = rig
    manager.up()
    host.commands.clear()
    assert manager.check(verify_rdma_mtu=False)["ready"]
    assert not any(argv[0] == "ibv_devinfo" for argv in host.commands)
    assert any("addr" in argv for argv in host.commands)
    assert any("route" in argv for argv in host.commands)
    assert any("filter" in argv for argv in host.commands)
    key = next(k for k in manager.objects if k.startswith("rule:"))
    host.inventory[key]["options"]["actions"][2]["to_dev"] = "wrong-device"
    with pytest.raises(ValueError, match="redirect"):
        manager.check(verify_rdma_mtu=False)


@pytest.mark.parametrize("target", ["gids", "ndevs"])
def test_fast_check_still_verifies_sysfs_identity(rig, target):
    manager, host = rig
    manager.up()
    manager.read_text = lambda path: "0" if target in path.parts else host.read_text(path)
    with pytest.raises(ValueError):
        manager.check(verify_rdma_mtu=False)


def test_full_check_still_detects_verbs_mtu_drift(rig):
    manager, host = rig
    manager.up()
    host.mtu = 1024
    assert manager.check(verify_rdma_mtu=False)["ready"]
    with pytest.raises(ValueError, match="RoCE MTU"):
        manager.check()


def test_absent_owned_object_cleanup_idempotent(rig):
    manager, host = rig
    manager.up()
    host.inventory.clear()
    assert manager.down()["clean"]
    assert manager.down()["clean"]


@pytest.mark.parametrize("preexisting", [True, False])
def test_permanent_neighbor_ownership(rig, preexisting):
    manager, host = rig
    route = next(obj for kind, obj in manager.objects.values() if kind == "route")
    route = replace(route, permanent_final_neighbor=True)
    key = "neighbor:" + route.path_name
    manager.objects[key] = ("neighbor", route)
    host.attach(manager)
    if preexisting:
        host.inventory[key] = host.value("neighbor", route)
    manager.up()
    assert manager.down()["clean"]
    assert (key in host.inventory) == preexisting


def test_dynamic_neighbor_is_not_overwritten(rig):
    manager, host = rig
    route = next(obj for kind, obj in manager.objects.values() if kind == "route")
    route = replace(route, permanent_final_neighbor=True)
    key = "neighbor:" + route.path_name
    manager.objects[key] = ("neighbor", route)
    host.attach(manager)
    host.inventory[key] = {"lladdr": route.next_hop_mac, "state": ["REACHABLE"]}
    with pytest.raises(ValueError, match="Conflicting permanent neighbor"):
        manager.up()
    assert not any(tuple(argv) in host.by_add for argv in host.commands)


def test_ingress_qdisc_conflict_not_replaced(rig):
    manager, host = rig
    key = next(k for k in manager.objects if k.startswith("qdisc:"))
    host.inventory[key] = {"kind": "ingress", "handle": "ffff:"}
    with pytest.raises(ValueError, match="Conflicting ingress"):
        manager.up()
    assert not any(tuple(argv) in host.by_add for argv in host.commands)


def test_counter_failure_retains_identity_but_allows_owned_cleanup(rig):
    manager, host = rig
    manager.up()
    key = next(k for k in manager.objects if k.startswith("rule:"))
    host.inventory[key]["options"]["actions"][0]["stats"] = {"drops": 1}
    with pytest.raises(ValueError, match="drops"):
        manager.check()
    assert manager.down()["clean"]


def test_failed_deletion_keeps_owned_journal_for_retry(rig):
    manager, host = rig
    manager.up()
    key = next(k for k in manager.objects if k.startswith("route:"))
    delete = manager._argv(*manager.objects[key], False)
    def runner(argv):
        if argv == delete:
            raise RuntimeError("Injected delete failure")
        return host(argv)
    manager.runner = runner
    result = manager.down()
    assert not result["clean"]
    assert result["ownership"][key] == "owned"
    manager.runner = host
    assert manager.down()["clean"]


def test_root_required_by_default(rig, monkeypatch):
    manager, _ = rig
    manager.require_root = True
    monkeypatch.setattr(network.os, "geteuid", lambda: 1000, raising=False)
    with pytest.raises(PermissionError, match="root"):
        manager.up()
