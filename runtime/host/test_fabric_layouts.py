"""Cabling, addressing, routes and setup plans of pairs, paths and cycles of 2 to 8 Sparks; no host is contacted.

``sparks`` builds synthetic inspection records (``sparkring node inspect``)
for any layout: every Spark has the four ConnectX functions with the DGX OS
names, locally administered MACs ``02:00:<position>:<port>:<function>:01``
and a management address ``192.0.2.<10 + position>``, and its LLDP shows the
far end of each cable. Other tests import it.
"""
import copy
import ipaddress
import json
from pathlib import Path
import uuid

import pytest

from runtime.common import fabric_layout
from runtime.host import cabling, node, topology
from runtime.host.test_cabling import synthetic
from scripts import deploy_network
from scripts.test_deploy_suite import inventory

LEGACY = json.loads((Path(__file__).parent / "fixtures/fabric/pair-cycle4-plans.json").read_text(encoding="utf-8"))
LAYOUTS = [fabric_layout.layout("pair", 2)] + [fabric_layout.layout(shape, size)
                                                for size in range(3, 9) for shape in ("path", "cycle")]


def mac(position, port, function):
    return f"02:00:{position:02x}:{port:02x}:{function:02x}:01"


def sparks(layout, *, blank=True, order=None):
    """Inspection records of a fabric with ``layout``, Node A first; ``order`` lists them in another order."""
    size = layout["size"]
    template = list(inventory()["hosts"].values())[0]
    result = []
    for position in range(size):
        facts = copy.deepcopy(template)
        address = f"192.0.2.{10 + position}"
        facts["ssh_target"] = "root@" + address
        facts["rank"] = position
        facts["management"]["address"] = address
        facts["management"]["controller_address"] = "192.0.2.11" if position == 0 else "192.0.2.10"
        facts["management"]["route_to_controller"]["dst"] = facts["management"]["controller_address"]
        management = facts["management"]["interface"]
        for interface in facts["interfaces"]:
            if interface["name"] == management:
                interface["ipv4"] = [address + "/24"]
                continue
            role = next(role for role, name in fabric_layout.NETDEVS.items() if name == interface["name"])
            port, function = fabric_layout.role_port(role), fabric_layout.FUNCTIONS.index(fabric_layout.role_function(role))
            interface["mac"] = mac(position, port, function)
            identity = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sparkring-test/{position}/{role}"))
            interface["network_manager"]["connection_uuid"] = identity
            if blank:
                interface["ipv4"] = []
                interface["mtu"] = 1500
            if fabric_layout.peer(layout, position, port) is None or role not in fabric_layout.roles(layout, position):
                # A port without a fabric cable has no carrier (a pair's port 1 here).
                interface["operstate"] = "DOWN"
        for connection in facts["network"]["connections"]:
            role = next((role for role, name in fabric_layout.NETDEVS.items() if name == connection["interface"]), None)
            if role:
                connection["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sparkring-test/{position}/{role}"))
        result.append({"node_id": str(uuid.UUID(int=position + 1)), "hostname": f"spark{position}",
                       "revision": "a" * 40, "facts": facts, "lldp": {"lldp": {"interface": []}}})
    for _, first, second in fabric_layout.cables(layout):
        for (here, port), (there, far_port) in ((first, second), (second, first)):
            for function in fabric_layout.FUNCTIONS:
                netdev = fabric_layout.NETDEVS[fabric_layout.port_role(port, function)]
                far = mac(there, far_port, fabric_layout.FUNCTIONS.index(function))
                result[here]["lldp"]["lldp"]["interface"].append({netdev: {
                    "chassis": {f"spark{there}": {"id": {"type": "mac", "value": far}}},
                    "port": {"id": {"type": "mac", "value": far}}}})
    if order is not None:
        result = [result[index] for index in order]
    return result


def configured(plan):
    """The plan's inspection records after its addresses were applied."""
    result = copy.deepcopy(plan["nodes"])
    for row, host in zip(result, plan["spec"]["hosts"], strict=True):
        for port in host["data_interfaces"]:
            interface = next(i for i in row["facts"]["interfaces"] if i["name"] == port["netdev"])
            interface.update(ipv4=[port["address"]], mtu=9000)
            interface["network_manager"].update(ipv4_addresses=[port["address"]], ethernet_mtu=9000)
            next(r for r in row["facts"]["rdma"] if r["device"] == port["rdma_device"])["gid"] = (
                "::ffff:" + port["address"].split("/")[0])
    return result


def plan_of(layout, **options):
    found = sparks(layout)
    return topology.build_spec(found, found[0]["node_id"], **options)


# Cabling: the port map of every supported layout.

def ring_cables(size, *, closed):
    return [((i, 0), ((i + 1) % size, 1)) for i in range(size if closed else size - 1)]


@pytest.mark.parametrize("layout", LAYOUTS, ids=fabric_layout.name)
def test_cabling_names_every_layout_and_maps_each_port(layout):
    size = layout["size"]
    cables = [((0, 0), (1, 0))] if layout["shape"] == "pair" else ring_cables(size, closed=layout["shape"] == "cycle")
    result = cabling.diagnose(synthetic(size, cables), "spark-a", strict=True, whole=True)
    assert result["ready"] and result["schema"] == "sparkring-cabling/v2"
    assert (result["shape"], result["layout_size"], result["layout_name"]) == (
        layout["shape"], size, fabric_layout.name(layout))
    assert result["layout"] == {"pair": "pair", "path": "path", "cycle": "ring"}[layout["shape"]]
    assert result["order_names"] == [f"spark-{chr(97 + i)}" for i in range(size)]
    for row in result["positions"]:
        for port in (0, 1):
            far = fabric_layout.peer(layout, row["position"], port)
            entry = row["ports"][str(port)]
            if far is None:
                assert entry["peer"] is None and entry["cable"] is None
            else:
                assert entry["peer"] == {"position": far[0], "spark": f"spark-{chr(97 + far[0])}", "port": far[1]}
                assert entry["cable"] == fabric_layout.cable_of(layout, row["position"], port)
    assert sorted(c["cable"] for c in result["cables"]) == list(range(fabric_layout.cable_count(layout)))
    lines = cabling.lines(result)
    assert f"Layout: {fabric_layout.name(layout)}" in lines and "Ports:" in lines
    if layout["shape"] == "path":
        assert result["free"] == ["spark-a port 1", f"spark-{chr(96 + size)} port 0"]
        assert any(f"makes a cycle-{size}" in note for note in result["notes"])
        assert lines[-2].startswith("Line order: spark-a →")


@pytest.mark.parametrize("size", range(3, 9))
def test_a_line_with_one_swapped_spark_names_the_swap_and_the_order(size):
    cables = ring_cables(size, closed=False)
    swapped = size // 2
    cables = [tuple((s, p ^ (s == swapped)) for s, p in cable) for cable in cables]
    result = cabling.diagnose(synthetic(size, cables), "spark-a", strict=True, whole=True)
    assert result["layout"] == "path" and not result["ready"]
    assert result["fix"] == [f"On spark-{chr(97 + swapped)}, swap its two cables (port 0 ↔ port 1)."]
    assert "2 cables do not run from port 0 to the next Spark's port 1" in result["summary"]
    assert result["positions"] is None


def test_a_line_cabled_the_other_way_says_where_setup_runs_as_it_is():
    cables = [((i, 1), (i + 1, 0)) for i in range(4)]
    result = cabling.diagnose(synthetic(5, cables), "spark-a")
    assert result["layout"] == "path" and not result["ready"]
    assert "From spark-e, at the other end, every cable already runs that way" in result["summary"]


def test_node_a_inside_a_line_is_told_to_close_it_or_run_setup_at_an_end():
    result = cabling.diagnose(synthetic(6, ring_cables(6, closed=False)), "spark-c", strict=True, whole=True)
    assert result["layout"] == "ring" and not result["ready"]
    assert result["fix"][0] == "Connect a cable from spark-f port 0 to spark-a port 1."
    assert "run setup on spark-a or spark-f" in result["summary"]
    with pytest.raises(cabling.CablingError, match="A line must start at Node A"):
        raise cabling.CablingError(result)


@pytest.mark.parametrize("size", [3, 5, 8])
def test_a_ring_with_an_unseen_cable_end_is_incomplete(size):
    sparks_ = synthetic(size, ring_cables(size, closed=False), carrier={(size - 1, 0): True})
    survey = cabling.diagnose(sparks_, "spark-a")
    assert survey["layout"] == "incomplete" and "so the last cable is unknown" in survey["summary"]
    setup = cabling.diagnose(sparks_, "spark-a", strict=True, whole=True)
    assert setup["layout"] == "incomplete" and not setup["ready"] and cabling.MISSING in setup["summary"]


def test_nine_sparks_are_more_than_setup_supports():
    result = cabling.diagnose(synthetic(9, ring_cables(9, closed=True)), "spark-a")
    assert result["layout"] == "unsupported"
    assert result["summary"] == "9 Sparks are cabled together. SparkRing supports two to eight Sparks: a pair, a line or a ring."


def test_eight_ring_that_lost_a_cable_reads_as_a_line_with_its_free_ports():
    result = cabling.diagnose(synthetic(8, ring_cables(8, closed=False)), "spark-a")
    assert result["ready"] and result["layout_name"] == "path-8"
    assert result["notes"] == ["spark-h port 0 and spark-a port 1 are free; a cable from the first to the second "
                               "makes a cycle-8."]


# Topology: addresses, routes and forwarding as functions of the layout.

@pytest.mark.parametrize("layout", LAYOUTS, ids=fabric_layout.name)
def test_addresses_follow_the_cable_rule_and_the_supernet_grows_above_four_cables(layout):
    plan = plan_of(layout)
    cables = fabric_layout.cable_count(layout)
    assert plan["layout"] == layout
    assert plan["fabric_cidr"] == ("198.18.0.0/21" if cables <= 4 else "198.18.0.0/20")
    assert ("layout" in plan["spec"]) == (layout not in (fabric_layout.layout("pair", 2),
                                                         fabric_layout.layout("cycle", 4)))
    for rank, host in enumerate(plan["spec"]["hosts"]):
        assert [p["role"] for p in host["data_interfaces"]] == fabric_layout.roles(layout, rank)
        for port in host["data_interfaces"]:
            cable = fabric_layout.cable_of(layout, rank, fabric_layout.role_port(port["role"]))
            subnet = ipaddress.IPv4Network(plan["fabric_cidr"]).network_address + 256 * (
                2 * cable + fabric_layout.FUNCTIONS.index(fabric_layout.role_function(port["role"])))
            first = fabric_layout.cables(layout)[cable][1]
            host_part = 1 if first == (rank, fabric_layout.role_port(port["role"])) else 2
            assert port["address"] == f"{subnet + host_part}/24"
    after = topology.build_spec(configured(plan), plan["nodes"][0]["node_id"])
    assert deploy_network.verify_network(after["spec"], after["inventory"]["hosts"], hairpin=False)["ready"]
    assert after["id"] == plan["id"]


def test_a_narrow_supernet_is_refused_above_four_cables():
    found = sparks(fabric_layout.layout("cycle", 5))
    with pytest.raises(ValueError, match=r"198\.18\.0\.0/21 holds 4 cables, but this cycle-5 has 5; use "
                                         r"--fabric-cidr 198\.18\.0\.0/20"):
        topology.build_spec(found, found[0]["node_id"], fabric_cidr="198.18.0.0/21")
    assert topology.build_spec(found, found[0]["node_id"], fabric_cidr="198.18.0.0/20")["fabric_cidr"] == "198.18.0.0/20"


def route_target(plan, route):
    for rank, host in enumerate(plan["spec"]["hosts"]):
        for port in host["data_interfaces"]:
            if ipaddress.ip_interface(port["address"]).ip == ipaddress.ip_address(route["via"]):
                return rank
    raise AssertionError(route)


@pytest.mark.parametrize("layout", LAYOUTS, ids=fabric_layout.name)
def test_routes_reach_every_other_cable_subnet_through_the_neighbor_on_the_shorter_way(layout):
    plan = plan_of(layout)
    size = layout["size"]
    subnets = {str(ipaddress.ip_interface(p["address"]).network) for h in plan["spec"]["hosts"]
               for p in h["data_interfaces"]}
    for rank, host in enumerate(plan["spec"]["hosts"]):
        config = topology.persistent_config(plan, rank)
        node.validate(config)
        own = {str(ipaddress.ip_interface(p["address"]).network) for p in host["data_interfaces"]}
        assert {route["destination"] for route in config["routes"]} == subnets - own
        for route in config["routes"]:
            neighbor = route_target(plan, route)
            assert neighbor in fabric_layout.neighbors(layout, rank)
            port = next(p for p in host["data_interfaces"] if p["netdev"] == route["dev"])
            assert neighbor == fabric_layout.peer(layout, rank, fabric_layout.role_port(port["role"]))[0]
        if layout["shape"] == "cycle":
            # The far end of each routed subnet is no farther the way the route goes than the other way.
            for route in config["routes"]:
                cable = next(c for c, first, second in fabric_layout.cables(layout)
                             if str(ipaddress.ip_interface(plan["spec"]["hosts"][first[0]]["data_interfaces"][0]["address"]).network)
                             in (route["destination"],) or any(
                                 str(ipaddress.ip_interface(p["address"]).network) == route["destination"]
                                 for p in plan["spec"]["hosts"][first[0]]["data_interfaces"]
                                 if fabric_layout.role_port(p["role"]) == first[1]))
                first, second = fabric_layout.cables(layout)[cable][1:]
                forward = min((first[0] - rank) % size, (second[0] - rank) % size)
                backward = min((rank - first[0]) % size, (rank - second[0]) % size)
                going = route_target(plan, route) == (rank + 1) % size
                assert (forward <= backward) == going
        expected = 4 if fabric_layout.forwards(layout, rank) else 0
        assert len(config["forwarding"]) == expected


@pytest.mark.parametrize("case", sorted(LEGACY))
def test_pair_and_four_cycle_plans_equal_those_of_records_that_name_only_their_count(case):
    """The fixture holds what the topology code produced for these fixtures before it handled other layouts."""
    from runtime.host import fabric_bandwidth, fabric_ssh, hairpin_ring
    from runtime.host.test_appliance import nodes
    from runtime.host.test_hairpin_ring import kept
    import hashlib
    expected = LEGACY[case]
    size = 2 if case.startswith("pair") else 4
    found = nodes(size, blank=case.endswith("blank"))
    plan = topology.build_spec(found, found[0]["node_id"])
    if size == 4:
        kept(plan)

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    assert plan["id"] == expected["plan_id"] and plan["fabric_cidr"] == expected["fabric_cidr"]
    assert digest(plan["spec"]) == expected["spec_sha256"] and digest(plan["network"]) == expected["network_sha256"]
    configs = [topology.persistent_config(plan, rank) for rank in range(size)]
    assert [c["routes"] for c in configs] == expected["routes"]
    assert [c["forwarding"] for c in configs] == expected["forwarding"]
    # Records add only the layout's name.
    assert [digest({k: v for k, v in c.items() if k != "layout"}) for c in configs] == expected["persistent_sha256"]
    assert [[{k: p[k] for k in ("role", "netdev", "rdma_device", "address")} for p in h["data_interfaces"]]
            for h in plan["spec"]["hosts"]] == expected["interfaces"]
    assert fabric_ssh.routes({"name": "test", "plan": plan}) == expected["bulk_routes"]
    rows = fabric_bandwidth.cables(plan)
    assert [c["ends"] for c in rows] == expected["cable_ends"]
    assert [[[f["function"], f["server"]["rank"], f["server"]["netdev"], f["server"]["address"],
              f["client"]["rank"], f["client"]["netdev"], f["client"]["address"]] for f in c["functions"]]
            for c in rows] == expected["cable_functions"]
    assert [[r["rank"], r["state"]] for r in hairpin_ring.requirement(plan)] == expected["hairpin"]


@pytest.mark.parametrize("layout", LAYOUTS, ids=fabric_layout.name)
def test_hairpin_and_its_driver_steps_belong_to_the_sparks_that_relay(layout):
    from runtime.host import hairpin_ring
    plan = plan_of(layout)
    relaying = [rank for rank in range(layout["size"])
                if fabric_layout.relayed(layout) and fabric_layout.forwards(layout, rank)]
    assert hairpin_ring.ranks(plan) == relaying
    assert [row["rank"] for row in hairpin_ring.requirement(plan)] == relaying
    assert [rank for rank, host in enumerate(plan["network"]["hosts"]) if host["hairpin"]] == relaying


def test_setup_refuses_sparks_that_do_not_form_one_layout():
    found = sparks(fabric_layout.layout("cycle", 6))
    # Remove the LLDP evidence of one Spark: setup's check cannot confirm its cables.
    found[3]["lldp"]["lldp"]["interface"] = []
    with pytest.raises(cabling.CablingError, match="Missing reciprocal LLDP cable evidence"):
        topology.build_spec(found, found[0]["node_id"])
    with pytest.raises(ValueError, match="Select two to eight distinct authenticated Sparks"):
        topology.build_spec(sparks(fabric_layout.layout("path", 3))[:1], str(uuid.UUID(int=1)))


def test_spark_order_comes_from_the_cables_not_from_the_list():
    layout = fabric_layout.layout("cycle", 7)
    found = sparks(layout, order=[3, 6, 0, 5, 1, 4, 2])
    plan = topology.build_spec(found, str(uuid.UUID(int=1)))
    assert [host["node_id"] for host in plan["spec"]["hosts"]] == [str(uuid.UUID(int=i + 1)) for i in range(7)]


def test_legacy_records_derive_their_layout_from_their_count():
    plan = plan_of(fabric_layout.layout("cycle", 4))
    del plan["layout"]
    assert topology.layout_of(plan) == fabric_layout.layout("cycle", 4)
    config = topology.persistent_config(plan, 0)
    del config["layout"]
    assert node.record_layout(config) == fabric_layout.layout("cycle", 4)
    with pytest.raises(ValueError, match="must name its fabric layout"):
        fabric_layout.legacy(6)
