"""CPU simulations: no SSH sessions, service changes, or GPU/RDMA operations."""
import copy
import ipaddress
import json
import uuid
from types import SimpleNamespace

import pytest

from runtime.host import controller, discovery, node, topology
from scripts import deploy_network
from scripts.test_deploy_suite import inventory


def nodes(size=4, *, blank=False):
    rows = list(inventory()["hosts"].values())[:size]
    result = []
    for rank, facts in enumerate(rows):
        facts["ssh_target"] = "root@192.0.2." + str(10 + rank)
        facts["management"]["controller_address"] = "192.0.2." + str(11 if rank == 0 else 10)
        facts["management"]["route_to_controller"]["dst"] = facts["management"]["controller_address"]
        for device, port in topology.endpoints({"facts": facts, "hostname": "fixture"}).items():
            interface = next(i for i in facts["interfaces"] if i["name"] == port["netdev"])
            if blank:
                interface["ipv4"] = []
                interface["mtu"] = 1500
            elif size == 2 and device.startswith("cw_"):
                address = f"198.18.{int(device.endswith('secondary'))}.{rank + 1}/24"
                interface["ipv4"] = [address]
                interface["network_manager"]["ipv4_addresses"] = [address]
                next(r for r in facts["rdma"] if r["netdev"] == port["netdev"])["gid"] = "::ffff:" + address.split("/")[0]
        result.append({"node_id": str(uuid.UUID(int=rank + 1)), "hostname": f"spark{rank}", "revision": "a" * 40,
                       "facts": facts, "lldp": {"lldp": {"interface": []}}})
    for rank, current in enumerate(result):
        for role, port in topology.endpoints(current).items():
            if size == 2 and not role.startswith("cw_"):
                continue
            clockwise = role.startswith("cw_")
            peer = result[(rank + (1 if clockwise else -1)) % size]
            peer_role = role if size == 2 else ("ccw_" if clockwise else "cw_") + role.split("_")[1]
            other = topology.endpoints(peer)[peer_role]
            current["lldp"]["lldp"]["interface"].append({port["netdev"]: {
                "chassis": {peer["hostname"]: {"id": {"type": "mac", "value": other["mac"]}}},
                "port": {"id": {"type": "mac", "value": other["mac"]}}}})
    return result


def configured(plan):
    result = copy.deepcopy(plan["nodes"])
    for row, host in zip(result, plan["spec"]["hosts"], strict=True):
        for port in host["data_interfaces"]:
            interface = next(i for i in row["facts"]["interfaces"] if i["name"] == port["netdev"])
            interface.update(ipv4=[port["address"]], mtu=9000)
            interface["network_manager"].update(ipv4_addresses=[port["address"]], ethernet_mtu=9000)
            next(r for r in row["facts"]["rdma"] if r["device"] == port["rdma_device"])["gid"] = "::ffff:" + port["address"].split("/")[0]
    return result


@pytest.mark.parametrize("size", [2, 4])
def test_discovery_orders_cables_from_selected_head_and_preserves_addresses(size):
    found = nodes(size)
    for n in found:
        n["facts"]["interfaces"].append({"name": "tailscale0", "mac": None, "ipv4": ["100.64.0.1/32"]})
        ports = topology.endpoints(n)
        for role, p in ports.items():
            sibling = ports[role.replace("primary", "secondary") if role.endswith("primary") else role.replace("secondary", "primary")]
            n["lldp"]["lldp"]["interface"].append({p["netdev"]: {
                "chassis": {n["hostname"]: {"id": {"type": "mac", "value": sibling["mac"]}}},
                "port": {"id": {"type": "mac", "value": sibling["mac"]}}}})
    plan = topology.build_spec(list(reversed(found)), found[0]["node_id"])
    assert [n["node_id"] for n in plan["nodes"]] == [n["node_id"] for n in found]
    assert all(h["action"] == "none" for h in plan["network"]["hosts"])
    assert deploy_network.verify_network(plan["spec"], plan["inventory"]["hosts"])["data_functions"] == size * (2 if size == 2 else 4)
    for rank in range(size):
        config = topology.persistent_config(plan, rank)
        node.validate(config)
        assert len(config["routes"]) == (4 if size == 4 else 0)
        for route in config["routes"]:
            subnet = ipaddress.ip_network(route["destination"])
            peer = next(h for h in plan["spec"]["hosts"] if any(ipaddress.ip_interface(p["address"]).ip == ipaddress.ip_address(route["via"]) for p in h["data_interfaces"]))
            assert any(ipaddress.ip_interface(p["address"]).network == subnet for p in peer["data_interfaces"])


@pytest.mark.parametrize("size", [2, 4])
def test_blank_nics_have_reviewable_changes_and_preserve_management(size):
    found = nodes(size, blank=True)
    plan = topology.build_spec(found, found[0]["node_id"])
    assert not plan["preserve_existing_addresses"]
    for host, proposed in zip(plan["spec"]["hosts"], plan["network"]["hosts"], strict=True):
        assert proposed["action"] == "configure"
        assert all(host["management_netdev"] not in action["argv"] for action in proposed["apply"])
    refreshed = configured(plan)
    after = topology.build_spec(refreshed, found[0]["node_id"])
    assert deploy_network.verify_network(after["spec"], after["inventory"]["hosts"])["ready"]


@pytest.mark.parametrize("fault", ["missing", "disagreement", "duplicate", "partial", "management"])
def test_unsafe_topology_stops_before_any_runner(fault):
    found = nodes()
    if fault == "missing":
        found[0]["lldp"]["lldp"]["interface"] = []
    elif fault == "disagreement":
        rows = found[0]["lldp"]["lldp"]["interface"]
        list(rows[1].values())[0]["port"]["id"]["value"] = topology.endpoints(found[2])["ccw_secondary"]["mac"]
    elif fault == "duplicate":
        found[1]["node_id"] = found[0]["node_id"]
    elif fault == "partial":
        found[0]["facts"]["interfaces"][1]["ipv4"] = []
    else:
        found[0]["facts"]["management"]["interface"] = found[0]["facts"]["rdma"][0]["netdev"]
    with pytest.raises(ValueError):
        topology.build_spec(found, found[0]["node_id"])


def test_avahi_never_authenticates_advertisements():
    value = discovery.avahi_candidates("=;eth0;IPv4;Spark;_sparkring._tcp;local;spark.local;192.0.2.10;22;\n" * 2)
    assert value == [{"address": "192.0.2.10", "hostname": "spark.local", "authenticated": False}]
    with pytest.raises(ValueError):
        discovery.target("-oProxyCommand=evil")


def test_installed_identity_must_match_ssh_target():
    response = {"facts": {"management": {"address": "192.0.2.50"}, "ssh_target": "root@192.0.2.50"}}
    with pytest.raises(ValueError, match="differs"):
        discovery.inspect_node("root@192.0.2.10", 0, "192.0.2.11", invoke=lambda *a: json.dumps(response))


@pytest.mark.parametrize("size", [2, 4])
def test_setup_simulation_checks_every_host_and_completes_receipts(tmp_path, size):
    found = nodes(size)
    plan = topology.build_spec(found, found[0]["node_id"])
    # Four-Spark nodes report the ConnectX hairpin setting kept, so the hairpin
    # step has nothing to do; the plan's nodes are the dicts inspect_nodes returns.
    from runtime.host.test_hairpin_ring import kept
    kept(plan)
    remote = []

    def invoke(host, argv, **kwargs):
        remote.append((host, argv))
        return "{}"

    def runner(*args):
        return {"returncode": 0, "stdout": '{"checked":true}', "stderr": ""}

    result = controller.apply(plan, tmp_path / "setup", run=runner, invoke=invoke, inspect_nodes=lambda _: found)
    assert result["id"] == plan["id"]
    assert json.loads((tmp_path / "setup/setup.json").read_text())["complete"]
    operations = [argv[4] for _, argv in remote if argv[0] == "sudo"]
    assert operations[:size] == ["verify"] * size
    assert operations.count("configure") == size
    with pytest.raises(ValueError, match="receipt exists"):
        controller.apply(plan, tmp_path / "setup", run=runner, invoke=invoke)


def test_setup_unknown_outcome_stops_before_persistence(tmp_path):
    found = nodes(2, blank=True)
    plan = topology.build_spec(found, found[0]["node_id"])
    calls = []

    def failed(*args):
        if "'apply'" in args[1][-1]:
            return {"returncode": 124, "stdout": "", "stderr": "connection lost", "uncertain": True}
        return {"returncode": 0, "stdout": '{"checked":true}', "stderr": ""}

    with pytest.raises(RuntimeError):
        controller.apply(plan, tmp_path, run=failed, invoke=lambda *a, **k: calls.append(a))
    assert calls == []
    assert not json.loads((tmp_path / "setup.json").read_text())["complete"]


def test_boot_restore_refuses_conflicting_routes_before_mutation():
    found = nodes()
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, 0)
    facts = found[0]["facts"]
    facts["routes"].append({"dst": config["routes"][0]["destination"], "dev": "management", "gateway": "192.0.2.1"})
    calls = []
    with pytest.raises(ValueError, match="Conflicting"):
        node.restore(config, collect=lambda _: facts, run=lambda *a, **k: calls.append(a))
    assert not calls


def test_status_preserves_stale_and_not_configured_states(tmp_path):
    assert node.snapshot(root=tmp_path)["state"] == "not-configured"
    node.save(tmp_path, "/run/sparkring/status.json", {"observed_at": 10, "state": "network-configured"})
    assert node.status(root=tmp_path, now=lambda: 500)["state"] == "stale"


def test_changed_mac_cannot_reconfigure_or_restore(tmp_path):
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, 0)
    facts = found[0]["facts"]
    facts["interfaces"][1]["mac"] = "00:00:00:00:00:00"
    with pytest.raises(ValueError, match="interface changed"):
        node.restore(config, collect=lambda _: facts)


def test_restore_uses_only_fabric_interfaces():
    found = nodes()
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, 0)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if "-C" in argv else 0, stdout="", stderr="")

    node.restore(config, collect=lambda _: found[0]["facts"], run=run)
    assert len([a for a in calls if a[:3] == ["ip", "route", "add"]]) == 4
    assert all(config["management"]["interface"] not in " ".join(a) for a in calls)
    assert not any("net.ipv4.ip_forward" in " ".join(a) for a in calls)


def test_discovery_ignores_this_hosts_own_sibling_functions():
    from runtime.host import bootstrap
    head = {"id": "a", "hostname": "a", "architecture": "aarch64",
            "functions": [{"netdev": "p0", "mac": "aa:aa:aa:aa:aa:01", "addresses": ["fe80::1"]},
                          {"netdev": "p0b", "mac": "aa:aa:aa:aa:aa:02", "addresses": ["fe80::2"]}],
            "neighbors": [{"dst": "fe80::2", "dev": "p0", "lladdr": "aa:aa:aa:aa:aa:02"}]}

    class Transport:
        def inventory(self, route):
            if route:
                raise AssertionError("logged into its own sibling function")
            return head

        def login(self, route):
            raise AssertionError("logged into its own sibling function")
    with pytest.raises(ValueError, match="setup needs two to eight"):
        bootstrap.discover(Transport())


def stable_privacy(facts, netdev, *, mode="default"):
    """A function whose connection also generates a stable-privacy link-local address, which moves its IPv4 GID."""
    interface = next(i for i in facts["interfaces"] if i["name"] == netdev)
    interface["network_manager"].update(connection_name="Wired connection 5", ipv6_addr_gen_mode=mode)
    facts["ipv6_link_local"] = {netdev: ["fe80::4ebb:47ff:fe2c:9320", "fe80::1037:222a:cf8e:5d35"]}
    function = next(r for r in facts["rdma"] if r["netdev"] == netdev)
    function["gid"] = "fe80::1037:222a:cf8e:5d35"


HINT = ("; enp1s0f1np1 has 2 IPv6 link-local addresses (fe80::4ebb:47ff:fe2c:9320, fe80::1037:222a:cf8e:5d35) and "
        "its NetworkManager connection 'Wired connection 5' uses ipv6.addr-gen-mode default, which moves the IPv4 "
        "RoCE v2 GID past index 3. Fix: nmcli connection modify 'Wired connection 5' ipv6.addr-gen-mode eui64, then "
        "nmcli connection up 'Wired connection 5'")


def test_gid_index_refusals_name_the_link_local_cause_and_the_nmcli_fix():
    from scripts.deploy_inventory import gid_index_hint
    plan = topology.build_spec(nodes(blank=True), nodes(blank=True)[0]["node_id"])
    found = configured(plan)
    facts = found[1]["facts"]
    stable_privacy(facts, "enp1s0f1np1")
    assert gid_index_hint(facts, "enp1s0f1np1") == HINT
    after = topology.build_spec(found, found[0]["node_id"])
    with pytest.raises(deploy_network.NetworkPlanError) as error:
        deploy_network.verify_network(after["spec"], after["inventory"]["hosts"], hairpin=False)
    assert str(error.value) == f"{after['spec']['hosts'][1]['host']}: GID index 3 does not match enp1s0f1np1 IPv4 address" + HINT
    config = topology.persistent_config(after, 1)
    with pytest.raises(ValueError) as error:
        node.observe(config, collect=lambda request: facts)
    assert str(error.value) == "GID3/RoCE link is unavailable: enp1s0f1np1" + HINT


def test_gid_index_hint_is_silent_for_the_hardware_derived_form():
    from scripts.deploy_inventory import gid_index_hint
    facts = nodes()[0]["facts"]
    assert gid_index_hint(facts, "enp1s0f1np1") == ""
    next(i for i in facts["interfaces"] if i["name"] == "enp1s0f1np1")["network_manager"]["ipv6_addr_gen_mode"] = "eui64"
    facts["ipv6_link_local"] = {"enp1s0f1np1": ["fe80::4ebb:47ff:fe2c:9320"]}
    assert gid_index_hint(facts, "enp1s0f1np1") == ""
    stable_privacy(facts, "enp1s0f1np1", mode="eui64")
    assert gid_index_hint(facts, "enp1s0f1np1").startswith("; enp1s0f1np1 has 2 IPv6 link-local addresses")
