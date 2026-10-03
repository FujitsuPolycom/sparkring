"""The host agent's restoration of approved fabric routes, with a fake routing table, sysfs and systemd; no host access."""
import ipaddress
import json
import subprocess
from pathlib import Path

import pytest

from runtime.host import native_mesh, node
from runtime.host.test_node_status import FAILED_MESH, PAIR_KEYS, Answers, approve, ring
from runtime.host.test_persistence import unit_settings

PACKAGING = Path(__file__).resolve().parents[2] / "packaging/debian"
# Commands that would change a route; the agent may run only "ip route add".
ROUTE_CHANGES = ("del", "delete", "replace", "change", "flush", "append", "prepend")


class Spark(Answers):
    """A ring member's routing table, IPv4 addresses, link carrier and fabric unit state.

    ``ip route add`` appends to the same route list that the status facts
    read, as the kernel does. Every fabric function starts with carrier, its
    approved address and that address's subnet route.
    """

    def __init__(self, root, config, facts, *, active=True, fail=None, answers=None, **kwargs):
        super().__init__(answers or {}, **kwargs)
        self.root, self.routes, self.active, self.fail = Path(root), facts["routes"], active, fail
        self.addresses = {row["name"]: list(row["ipv4"]) for row in facts["interfaces"]}
        for port in config["interfaces"]:
            self.routes.append({"dst": str(ipaddress.ip_interface(port["address"]).network), "dev": port["netdev"],
                                "protocol": "kernel", "scope": "link"})
            self.link(port["netdev"], True)

    def link(self, netdev, carrier):
        path = self.root / "sys/class/net" / netdev
        path.mkdir(parents=True, exist_ok=True)
        (path / "carrier").write_text("1\n" if carrier else "0\n")
        (path / "operstate").write_text("up\n" if carrier else "down\n")

    def lose(self, netdev):
        """Remove every route through ``netdev`` except its subnet route, as a lost address does once it returns."""
        self.routes[:] = [row for row in self.routes if row.get("dev") != netdev or not row.get("gateway")]

    def added(self):
        return [argv for argv in self.calls if argv[:3] == ["ip", "route", "add"]]

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        if argv[:3] == ["systemctl", "is-active", node.FABRIC_UNIT]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 0 if self.active else 3,
                                               "active\n" if self.active else "inactive\n", "")
        if argv[:5] == ["ip", "-j", "-4", "route", "show"]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.routes), "")
        if argv[:5] == ["ip", "-j", "-4", "address", "show"]:
            self.calls.append(argv)
            rows = [{"ifname": name, "addr_info": [{"family": "inet", "local": value.split("/")[0],
                                                    "prefixlen": int(value.split("/")[1])} for value in values]}
                    for name, values in self.addresses.items()]
            return subprocess.CompletedProcess(argv, 0, json.dumps(rows), "")
        if argv[:3] == ["ip", "route", "add"]:
            self.calls.append(argv)
            if self.fail:
                return subprocess.CompletedProcess(argv, 2, "", self.fail)
            self.routes.append({"dst": argv[3], "gateway": argv[5], "dev": argv[7], "protocol": "static"})
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "ip":
            pytest.fail("unexpected ip command: " + " ".join(argv))
        return super().__call__(argv, **kwargs)


def member(tmp_path, rank=0, **options):
    """An approved, armed ring member whose fabric record, facts and fake Spark agree."""
    config, facts = ring(tmp_path, rank=rank)
    approve(tmp_path, config, facts)
    spark = Spark(tmp_path, config, facts, armed=True, **options)
    return config, facts, spark


def through(config, role):
    """The approved fabric function of ``role`` and the approved routes through it."""
    port = next(p for p in config["interfaces"] if p["role"] == role)
    return port["netdev"], [r for r in config["routes"] if r["dev"] == port["netdev"]]


def states(result):
    return {row["destination"]: row["state"] for row in result["routes"]}


def test_only_the_routes_of_a_returned_link_are_added_and_a_repeat_changes_nothing(tmp_path):
    config, _, spark = member(tmp_path)
    netdev, lost = through(config, "ccw_primary")
    spark.lose(netdev)
    logged = []
    result = node.restore_routes(config, root=tmp_path, run=spark, log=logged.append)
    assert result["active"] is True
    assert spark.added() == [["ip", "route", "add", r["destination"], "via", r["via"], "dev", r["dev"], "proto", "static"]
                             for r in lost]
    assert states(result) == {r["destination"]: "restored" if r in lost else "present" for r in config["routes"]}
    assert logged == ["restored approved fabric route " + node.route_text(r) for r in lost]
    again = node.restore_routes(config, root=tmp_path, run=spark, log=logged.append)
    assert set(states(again).values()) == {"present"}
    assert len(spark.added()) == len(lost) and len(logged) == len(lost)


def test_restoration_changes_no_other_route_setting_or_rule(tmp_path):
    config, _, spark = member(tmp_path)
    for role in ("cw_primary", "ccw_secondary"):
        spark.lose(through(config, role)[0])
    # An operator's routes, one through a fabric function whose approved route is missing.
    unrelated = [{"dst": "203.0.113.0/24", "gateway": "192.0.2.1", "dev": "enP7s7", "protocol": "static"},
                 {"dst": "198.18.200.0/24", "gateway": "198.18.1.9", "dev": "enp1s0f0np0", "protocol": "static"}]
    spark.routes.extend(unrelated)
    before = list(spark.routes)
    node.restore_routes(config, root=tmp_path, run=spark, log=lambda text: None)
    assert all(row in spark.routes for row in unrelated)
    assert all(row in spark.routes for row in before)
    mutations = [argv for argv in spark.calls if argv[0] != "systemctl" and argv[:3] != ["ip", "-j", "-4"]]
    assert mutations == spark.added() and len(mutations) == 2
    assert {argv[3] for argv in mutations} <= {r["destination"] for r in config["routes"]}
    assert not any(word in argv for argv in spark.calls for word in ROUTE_CHANGES)
    assert not any(argv[0] in ("sysctl", "iptables", "nmcli") for argv in spark.calls)


def test_another_route_to_an_approved_destination_is_left_in_place_and_named(tmp_path):
    config, facts, spark = member(tmp_path)
    netdev, lost = through(config, "ccw_primary")
    spark.lose(netdev)
    other = {"dst": lost[0]["destination"], "gateway": "198.18.1.9", "dev": "enp1s0f0np0", "protocol": "static"}
    spark.routes.append(other)
    result = node.restore_routes(config, root=tmp_path, run=spark, log=lambda text: None)
    assert states(result)[lost[0]["destination"]] == "conflict" and spark.added() == []
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert status["state"] == "needs-attention"
    assert status["error"] == (f"Approved fabric route is missing: {node.route_text(lost[0])}; another route to "
                               f"{lost[0]['destination']} is in its place, and SparkRing does not replace it")
    assert other in spark.routes and spark.added() == []


def test_a_route_waits_for_carrier_then_for_the_address_and_its_subnet_route(tmp_path):
    config, _, spark = member(tmp_path)
    netdev, lost = through(config, "cw_secondary")
    port = next(p for p in config["interfaces"] if p["netdev"] == netdev)
    subnet = {"dst": str(ipaddress.ip_interface(port["address"]).network), "dev": netdev,
              "protocol": "kernel", "scope": "link"}
    spark.lose(netdev)
    spark.routes.remove(subnet)
    spark.addresses[netdev] = []
    spark.link(netdev, False)

    def run():
        return states(node.restore_routes(config, root=tmp_path, run=spark, log=lambda text: None))

    assert run()[lost[0]["destination"]] == "no-link"
    # NetworkManager has not configured the returned link yet.
    spark.link(netdev, True)
    assert run()[lost[0]["destination"]] == "unconfigured"
    # The address is back; NetworkManager adds the subnet route of a noprefixroute address after it.
    spark.addresses[netdev] = [port["address"]]
    assert run()[lost[0]["destination"]] == "unconfigured"
    assert spark.added() == []
    spark.routes.append(subnet)
    assert run()[lost[0]["destination"]] == "restored"
    assert len(spark.added()) == 1


def test_status_names_the_rank_behind_a_link_without_carrier_ahead_of_a_failed_mesh(tmp_path):
    config, facts, spark = member(tmp_path, answers=FAILED_MESH)
    lost = []
    for role in ("ccw_primary", "ccw_secondary"):
        netdev, routes = through(config, role)
        spark.lose(netdev)
        spark.link(netdev, False)
        lost += routes
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert status["state"] == "needs-attention" and spark.added() == []
    assert status["error"] == ("No link to rank 3 on enp1s0f1np1, enP2p1s0f1np1: rank 3 is down or restarting the "
                               "link, or the cable is out; the fabric routes through them ("
                               + ", ".join(r["destination"] for r in lost) + ") return with the link")
    assert status["next_action"] == "start rank 3 or reconnect the cable"
    assert status["mesh"]["failed"][0]["unit"] == "sparkring-mesh.service"


def test_status_names_both_ranks_when_both_neighbors_are_down(tmp_path):
    config, facts, spark = member(tmp_path, rank=1)
    for port in config["interfaces"]:
        spark.link(port["netdev"], False)
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark)
    assert status["error"].startswith("No link to rank 0 on enp1s0f1np1, enP2p1s0f1np1 and to rank 2 on "
                                      "enp1s0f0np0, enP2p1s0f0np0: rank 0 and rank 2 are down or restarting")
    assert status["next_action"] == "start rank 0 and rank 2 or reconnect the cable"


def test_a_pair_names_the_other_spark_when_its_cable_has_no_link(tmp_path):
    config, facts = ring(tmp_path, size=2, rank=1)
    spark = Spark(tmp_path, config, facts)
    for port in config["interfaces"]:
        spark.link(port["netdev"], False)
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert status["error"] == ("No link to rank 0 on " + ", ".join(p["netdev"] for p in config["interfaces"])
                               + ": rank 0 is down or restarting the link, or the cable is out")
    assert status["next_action"] == "start rank 0 or reconnect the cable" and spark.added() == []


def test_the_agent_snapshot_restores_a_route_and_reports_the_spark_configured(tmp_path):
    config, facts, spark = member(tmp_path)
    netdev, lost = through(config, "ccw_primary")
    spark.lose(netdev)
    # A refreshed status, which any operator may request, reads only.
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark)
    assert status["state"] == "needs-attention" and spark.added() == []
    assert status["error"] == (f"Approved fabric route is missing: {node.route_text(lost[0])}; "
                               "sparkring-agent adds it again within 30 seconds")
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert status["state"] == "network-configured" and "error" not in status and "warnings" not in status
    assert len(spark.added()) == len(lost)


def test_nothing_is_restored_while_the_fabric_service_is_not_active(tmp_path):
    config, facts, spark = member(tmp_path, active=False)
    netdev, lost = through(config, "ccw_primary")
    spark.lose(netdev)
    assert node.restore_routes(config, root=tmp_path, run=spark) == {"active": False, "routes": []}
    assert [argv for argv in spark.calls if argv[0] == "ip"] == []
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert status["error"] == (f"Approved fabric route is missing: {node.route_text(lost[0])}; "
                               "sparkring-agent adds it only while sparkring-fabric.service is active")
    assert spark.added() == []


def test_a_failed_addition_is_logged_and_named_by_the_status(tmp_path):
    config, facts, spark = member(tmp_path, fail="Error: Nexthop has invalid gateway.")
    netdev, lost = through(config, "ccw_primary")
    spark.lose(netdev)
    logged = []
    result = node.restore_routes(config, root=tmp_path, run=spark, log=logged.append)
    failed = [row for row in result["routes"] if row["state"] == "failed"]
    assert [row["destination"] for row in failed] == [r["destination"] for r in lost]
    assert failed[0]["error"] == f"ip route add {lost[0]['destination']}: Error: Nexthop has invalid gateway."
    assert logged[0].startswith("could not restore approved fabric route " + node.route_text(lost[0]))
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert status["error"] == (f"Approved fabric route is missing: {node.route_text(lost[0])}; adding it failed: "
                               f"ip route add {lost[0]['destination']}: Error: Nexthop has invalid gateway.")


def test_a_pair_and_an_adopted_fabric_have_no_routes_to_restore(tmp_path):
    config, facts = ring(tmp_path, size=2)
    spark = Spark(tmp_path, config, facts)
    assert config["routes"] == [] and node.restore_routes(config, root=tmp_path, run=spark) is None
    assert spark.calls == []
    status = node.snapshot(root=tmp_path, collect=lambda _: facts, run=spark, repair=True)
    assert set(status) == PAIR_KEYS and status["state"] == "network-configured"
    assert spark.added() == []
    observed = dict(config, ownership="observed", routes=[], forwarding=[])
    assert node.restore_routes(observed, root=tmp_path, run=spark) is None


def test_routes_are_kept_while_two_spark_models_serve_on_the_halves(tmp_path):
    config, facts, spark = member(tmp_path, rank=2)
    native_mesh._save_parked(["sparkring-mesh.service"], root=tmp_path)
    netdev, lost = through(config, "cw_primary")
    spark.lose(netdev)
    result = node.restore_routes(config, root=tmp_path, run=spark, log=lambda text: None)
    assert [row["destination"] for row in result["routes"] if row["state"] == "restored"] == \
        [r["destination"] for r in lost]


def test_the_agent_records_the_snapshot_that_restores_routes(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(node, "snapshot", lambda **kwargs: seen.append(kwargs) or {"observed_at": 1})
    node.agent(root=tmp_path, once=True)
    assert seen == [{"root": tmp_path, "repair": True}]
    assert node.read(tmp_path, "/run/sparkring/status.json") == {"observed_at": 1}


def test_the_agent_unit_can_add_routes_and_package_installation_restarts_it():
    unit = unit_settings((PACKAGING / "sparkring-agent.service").read_text())
    # Root in the host network namespace keeps CAP_NET_ADMIN for "ip route add".
    for key in ("User", "CapabilityBoundingSet", "AmbientCapabilities", "PrivateNetwork", "RestrictAddressFamilies",
                "NetworkNamespacePath", "PrivateUsers"):
        assert ("Service", key) not in unit, key
    assert unit[("Service", "ExecStart")] == ["/usr/bin/sparkring", "node", "agent"]
    assert "systemctl restart sparkring-agent.service" in (PACKAGING / "postinst").read_text()
