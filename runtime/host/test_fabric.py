"""The fabric document (``sparkring-fabric/v1``) and ``sparkring fabric show|verify``; no host is contacted.

Documents come from synthetic setup plans (``test_fabric_layouts``); each
Spark's checks run against a fake ``ip``, ``tc``, ``sysctl``, ``systemctl``
and ``ping``.
"""
import copy
import ipaddress
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.common import compose, fabric_document, fabric_layout
from runtime.host import fabric, node, relays, topology
from runtime.host.test_fabric_layouts import LAYOUTS, configured, plan_of

MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}


def document_of(layout, **options):
    plan = plan_of(layout)
    document, relay_plan = fabric.prepare(plan, cluster="test", marker=MARKER, **options)
    return plan, document, relay_plan


@pytest.mark.parametrize("layout", LAYOUTS, ids=fabric_layout.name)
def test_setup_documents_validate_and_list_both_ports_of_every_spark(layout):
    _, document, _ = document_of(layout, api_address="192.0.2.50")
    assert fabric_document.validate(document) is document
    assert (document["shape"], document["size"], document["head"]) == (layout["shape"], layout["size"], 0)
    assert document["fabric_cidr"] == fabric_layout.default_cidr(layout) and document["addressing"] == "planned"
    assert document["positions"][0]["lan_address"] == "192.0.2.50"
    for row in document["positions"]:
        assert sorted(row["ports"]) == ["0", "1"]
        cabled = {fabric_layout.role_port(role) for role in fabric_layout.roles(layout, row["position"])}
        for port, entry in row["ports"].items():
            assert (entry["cable"] is None) == (int(port) not in cabled)
            for function, value in entry["functions"].items():
                role = fabric_layout.port_role(int(port), function)
                assert (value["rdma"], value["netdev"], value["role"]) == (
                    fabric_layout.DEVICES[role], fabric_layout.NETDEVS[role], role)
    assert [cable["cable"] for cable in document["cables"]] == list(range(fabric_layout.cable_count(layout)))
    assert document["hairpin"]["required"] == fabric_layout.relayed(layout)
    # The canonical serialization is the one Compose documents use.
    assert fabric_document.encoded(document) == compose.encoded(document)


@pytest.mark.parametrize("layout", LAYOUTS, ids=fabric_layout.name)
def test_sircl_reads_the_device_names_of_every_document(layout):
    from spark_transport.sircl.sparkring_sircl import routes
    _, document, _ = document_of(layout)
    assert routes.roles_from_fabric_document(document) == routes.DEFAULT_ROLES
    assert fabric_document.uniform_names(document) and fabric_document.default_names(document)


def test_the_device_map_names_each_role_and_the_neighbor_it_serves():
    _, document, _ = document_of(fabric_layout.layout("path", 4))
    end = fabric_document.devices(document, 0)
    assert list(end) == ["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"]
    assert end["rocep1s0f0"] == {"netdev": "enp1s0f0np0", "role": "cw_primary", "port": 0, "function": "primary",
                                 "cable": 0, "address": "198.18.0.1",
                                 "neighbor": {"position": 1, "port": 1, "rdma": "rocep1s0f1",
                                              "netdev": "enp1s0f1np1", "address": "198.18.0.2"}}
    assert end["rocep1s0f1"]["neighbor"] is None and end["rocep1s0f1"]["cable"] is None
    middle = fabric_document.devices(document, 2)
    assert middle["roceP2p1s0f1"]["neighbor"] == {"position": 1, "port": 0, "rdma": "roceP2p1s0f0",
                                                  "netdev": "enP2p1s0f0np0", "address": "198.18.3.1"}
    assert fabric_document.position_of(document, document["positions"][3]["node_id"]) == 3


def test_identity_covers_sparks_ports_and_cables_but_not_addresses_names_or_health():
    _, document, _ = document_of(fabric_layout.layout("cycle", 6))
    changed = copy.deepcopy(document)
    changed["positions"][2]["hostname"] = "renamed"
    changed["positions"][0]["management"]["address"] = "192.0.2.99"
    changed["cables"][1]["health"] = {"state": "degraded"}
    changed["cables"][1]["seen_from"] = []
    assert fabric_document.identity(changed) == document["id"]
    replaced = copy.deepcopy(document)
    replaced["positions"][4]["node_id"] = "00000000-0000-0000-0000-000000000099"
    assert fabric_document.identity(replaced) != document["id"]
    with pytest.raises(fabric_document.FabricDocumentError, match="id does not match"):
        fabric_document.validate(replaced)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["positions"][1]["ports"]["0"]["functions"]["secondary"].update(rdma="rocep1s0f0"),
     "RDMA device rocep1s0f0 appears more than once"),
    (lambda d: d["positions"][2]["ports"].pop("1"), r'ports "0" and "1" are required'),
    (lambda d: d["positions"][0]["ports"]["1"]["functions"]["primary"].update(address="198.18.9.9/24"),
     "a free port has no fabric address"),
    (lambda d: d["positions"][1]["ports"]["0"].update(cable=3), "cable differs from the layout"),
    (lambda d: d["positions"][1]["ports"]["0"]["functions"]["primary"].update(address="198.18.7.1/24"),
     "is not in its cable's subnet"),
    (lambda d: d.update(transports=["nccl"]), "transports may list"),
    (lambda d: d.update(head=2), "head must be position 0"),
    (lambda d: d.update(fabric_cidr="198.18.0.0/22"), "holds fewer than 3 cables"),
])
def test_invalid_documents_are_refused_naming_the_rule(change, message):
    _, document, _ = document_of(fabric_layout.layout("path", 4))
    broken = copy.deepcopy(document)
    change(broken)
    broken["id"] = fabric_document.identity(broken)
    with pytest.raises(fabric_document.FabricDocumentError, match=message):
        fabric_document.validate(broken)


def test_documents_with_names_that_differ_between_sparks_are_valid_but_not_for_sircl():
    from spark_transport.sircl.sparkring_sircl import routes
    _, document, _ = document_of(fabric_layout.layout("cycle", 4))
    other = copy.deepcopy(document)
    other["positions"][2]["ports"]["0"]["functions"]["primary"].update(rdma="mlx5_0")
    other["id"] = fabric_document.identity(other)
    fabric_document.validate(other)
    assert not fabric_document.uniform_names(other) and not fabric_document.default_names(other)
    with pytest.raises(routes.RouteError, match="differently from the first position"):
        routes.roles_from_fabric_document(other)


def test_preserved_addresses_keep_one_subnet_per_cable_function():
    from runtime.host.test_appliance import nodes
    found = nodes(4)
    plan = topology.build_spec(found, found[0]["node_id"])
    document, relay_plan = fabric.prepare(plan, cluster="test", marker=MARKER)
    assert plan["preserve_existing_addresses"] and document["addressing"] == "preserved"
    assert document["cables"][0]["subnets"] == {"primary": "198.18.1.0/24", "secondary": "198.18.101.0/24"}
    assert relay_plan["positions"][0]["routes"]


def test_load_reads_a_spark_copy(tmp_path):
    _, document, _ = document_of(fabric_layout.layout("pair", 2))
    path = tmp_path / "topology.json"
    path.write_text(fabric_document.encoded(document), encoding="utf-8")
    assert fabric_document.load(path) == document
    with pytest.raises(fabric_document.FabricDocumentError, match="does not exist; sudo sparkring setup writes it"):
        fabric_document.load(tmp_path / "missing.json")


def test_install_document_writes_the_canonical_bytes_for_a_spark_it_names(tmp_path):
    _, document, _ = document_of(fabric_layout.layout("cycle", 3))
    node.save(tmp_path, "/etc/sparkring/node.json", {"schema": "sparkring-node/v1",
                                                     "node_id": document["positions"][1]["node_id"]})
    text = fabric_document.encoded(document)
    result = fabric.install_document(text, root=tmp_path)
    assert result["position"] == 1 and (tmp_path / fabric_document.HOST_PATH.lstrip("/")).read_text() == text
    with pytest.raises(ValueError, match="canonical form"):
        fabric.install_document(json.dumps(document), root=tmp_path)
    node.save(tmp_path, "/etc/sparkring/node.json", {"schema": "sparkring-node/v1",
                                                     "node_id": "00000000-0000-0000-0000-000000000099"})
    with pytest.raises(fabric_document.FabricDocumentError, match="is not part of fabric"):
        fabric.install_document(text, root=tmp_path)


def test_plan_lines_show_the_layout_ports_relays_and_transports():
    _, document, relay_plan = document_of(fabric_layout.layout("cycle", 8))
    lines = fabric.plan_lines(document, relay_plan)
    assert lines[0] == "Layout: cycle-8; fabric addresses from 198.18.0.0/20 (8 cables, two /24 subnets each)"
    assert lines[2] == ("  position 0 spark0: port 0 → position 1 spark1 port 1 (cable 0); port 1 → position 7 "
                        "spark7 port 0 (cable 7)")
    assert lines[10].startswith("Relays: one table for every Spark, at most 3 relays on a route; 96 routes")
    assert lines[-1] == "Transports this fabric can carry: sircl"
    _, document, relay_plan = document_of(fabric_layout.layout("cycle", 4))
    assert any("prepared transport's two-hop traffic" in line for line in fabric.plan_lines(document, relay_plan))
    _, document, relay_plan = document_of(fabric_layout.layout("path", 6))
    assert any("research-only" in line for line in fabric.plan_lines(document, relay_plan))


# Each Spark's checks and the verification on Node A.

class Host:
    """A Spark after setup: its records, kernel state, units and the answers to ping."""

    def __init__(self, root, plan, document, relay_plan, rank, *, unreachable=(), disabled=(), missing_route=None):
        self.root, self.plan, self.rank = Path(root), plan, rank
        self.config = topology.persistent_config(plan, rank, relays=relays.section(relay_plan, rank)
                                                 if relay_plan else None)
        node.save(root, "/etc/sparkring/node.json", {"schema": "sparkring-node/v1",
                                                     "node_id": self.config["node_id"]})
        node.save(root, "/etc/sparkring/fabric.json", self.config)
        copy_path = self.root / fabric_document.HOST_PATH.lstrip("/")
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(fabric_document.encoded(document), encoding="utf-8", newline="\n")
        self.facts = configured(plan)[rank]["facts"]
        self.unreachable, self.disabled = set(unreachable), set(disabled)
        for port in self.config["interfaces"]:
            path = self.root / "sys/class/net" / port["netdev"]
            path.mkdir(parents=True, exist_ok=True)
            (path / "carrier").write_text("1\n")
        self.routes = [{"dst": r["destination"], "gateway": r["via"], "dev": r["dev"]} for r in self.config["routes"]
                       if r["destination"] != missing_route]
        self.routes += [{"dst": str(ipaddress.ip_interface(p["address"]).network), "dev": p["netdev"]}
                        for p in self.config["interfaces"]]
        self.tc, self.neighbours, self.qdiscs = {}, [], {}
        table = self.config.get("relays") or {}
        for route in table.get("routes", []):
            self.routes.append({"dst": route["dst"].split("/")[0], "dev": route["dev"], "prefsrc": route["src"],
                                "protocol": "82"})
        for neighbour in table.get("neighbours", []):
            self.neighbours.append({"dst": neighbour["addr"], "dev": neighbour["dev"],
                                    "lladdr": neighbour["lladdr"], "state": ["PERMANENT"]})
        for rule in table.get("filters", []):
            self.tc.setdefault(rule["dev"], []).append({"pref": rule["pref"], "options": {
                "handle": rule["handle"], "keys": {"eth_type": rule["protocol"][2:]}, "skip_sw": True, "in_hw": True,
                "actions": [{"kind": "pedit"}, {"kind": "mirred", "to_dev": rule["actions"][1]["mirred"]["redirect"]}]}})
        for pid, row in enumerate(table.get("markers", []), 300):
            path = self.root / "proc" / str(pid)
            path.mkdir(parents=True)
            (path / "cmdline").write_bytes(b"\0".join(p.encode() for p in relays.marker_argv(table, row)) + b"\0")
        self.settings = {key: value for _, key, value in node.approved_settings(self.config)}

    def __call__(self, argv, **kwargs):
        output, code = "", 0
        if argv[:5] == ["ip", "-j", "-4", "route", "show"]:
            output = json.dumps(self.routes)
        elif argv[:5] == ["ip", "-j", "-4", "address", "show"]:
            output = json.dumps([{"ifname": p["netdev"], "addr_info": [
                {"family": "inet", "local": p["address"].split("/")[0], "prefixlen": 24}]}
                for p in self.config["interfaces"]])
        elif argv[:4] == ["ip", "-j", "-4", "neigh"]:
            output = json.dumps(self.neighbours)
        elif argv[:4] == ["tc", "-j", "filter", "show"]:
            output = json.dumps(self.tc.get(argv[5], []))
        elif argv[:2] == ["sysctl", "-n"]:
            output = self.settings.get(argv[2], "")
        elif argv[:2] == ["systemctl", "is-enabled"]:
            output = "disabled" if argv[2] in self.disabled else "enabled"
        elif argv[0] == "ping":
            code = 1 if argv[-1] in self.unreachable else 0
        return SimpleNamespace(returncode=code, stdout=output + "\n", stderr="")


def test_a_spark_that_matches_its_record_reports_every_check_ok(tmp_path):
    plan, document, relay_plan = document_of(fabric_layout.layout("cycle", 4))
    host = Host(tmp_path, plan, document, relay_plan, 1)
    hairpin_rows = [{"netdev": p["netdev"], "rdma_device": p["rdma_device"], "state": "in-effect"}
                    for p in host.config["interfaces"]]
    original = node.hairpin_rows
    node.hairpin_rows = lambda config, facts: hairpin_rows
    try:
        result = fabric.check_local(document, root=tmp_path, run=host, collect=lambda request: host.facts)
    finally:
        node.hairpin_rows = original
    assert result["position"] == 1 and {row["state"] for row in result["rows"]} == {"ok"}
    kinds = {row["kind"] for row in result["rows"]}
    assert kinds >= {"interfaces", "hairpin", "route", "setting", "relay-route", "relay-neighbour", "relay-filter",
                     "relay-marker", "unit", "document", "reach"}
    assert sum(row["kind"] == "reach" for row in result["rows"]) == 12


def test_a_spark_check_names_each_problem(tmp_path):
    plan, document, relay_plan = document_of(fabric_layout.layout("path", 4))
    destination = topology.persistent_config(plan, 0)["routes"][0]["destination"]
    host = Host(tmp_path, plan, document, relay_plan, 0, unreachable={"198.18.4.2"},
                disabled={relays.MARKER_UNIT}, missing_route=destination)
    host.tc.clear()
    result = fabric.check_local(document, root=tmp_path, run=host, collect=lambda request: host.facts)
    problems = {(row["kind"], row["what"], row["state"]) for row in result["rows"] if row["state"] != "ok"}
    assert ("unit", relays.MARKER_UNIT, "disabled") in problems
    assert ("reach", "position 3 198.18.4.2", "unreachable") in problems
    assert any(kind == "route" and state == "missing" for kind, _, state in problems)
    report = {"schema": fabric.VERIFY_SCHEMA, "fabric": document["id"], "layout": fabric_document.layout(document),
              "traffic": "none", "lanes": [], "result": "problems", "problems": len(problems),
              "sparks": [dict(result, hostname="spark0")]}
    lines = fabric.report_lines(report)
    assert lines[0] == f"Fabric verification found {len(problems)} problems:"
    assert "  spark0 (position 0) unit sparkring-relay-marker.service: disabled; disabled" in lines


class Fleet:
    """``fabric.Access`` over ``Host`` fakes: one per position."""

    def __init__(self, hosts):
        self.hosts, self.calls = hosts, []

    def node(self, rank, argv, *, data=None):
        self.calls.append((rank, argv))
        host = self.hosts[rank]
        if argv == ["fabric-check"]:
            return json.dumps(fabric.check_local(json.loads(data), root=host.root, run=host,
                                                 collect=lambda request: host.facts))
        if argv == ["fabric-document"]:
            return json.dumps(fabric.install_document(data, root=host.root))
        raise AssertionError(argv)


def test_setup_records_the_document_on_node_a_and_every_spark_and_verify_repeats_the_check(tmp_path, capsys):
    plan, document, relay_plan = document_of(fabric_layout.layout("path", 3))
    state = tmp_path / "controller"
    state.mkdir()
    hosts = [Host(tmp_path / f"spark{rank}", plan, document, relay_plan, rank) for rank in range(3)]
    hairpin = node.hairpin_rows
    node.hairpin_rows = lambda config, facts: [{"netdev": p["netdev"], "rdma_device": p["rdma_device"],
                                                "state": "in-effect"} for p in config["interfaces"]]
    try:
        for host in hosts:
            (host.root / fabric_document.HOST_PATH.lstrip("/")).unlink()
        cluster = {"name": "test", "plan": plan}
        fabric.save_prepared(tmp_path / "setup", document, relay_plan)
        bandwidth = {"measured_at": 1790000000, "cables": [
            {"ends": [{"rank": 0, "port": 0}, {"rank": 1, "port": 1}], "verdict": "healthy"},
            {"ends": [{"rank": 1, "port": 0}, {"rank": 2, "port": 1}], "verdict": "degraded"}]}
        recorded = fabric.finish_setup(state, cluster, tmp_path / "setup", bandwidth=bandwidth,
                                       access=Fleet(hosts), say=print)
        assert recorded["verified"]["result"] == "healthy" and recorded["id"] == document["id"]
        assert [c["health"]["state"] for c in recorded["cables"]] == ["healthy", "degraded"]
        text = (state / "fabric.json").read_text()
        assert all((host.root / fabric_document.HOST_PATH.lstrip("/")).read_text() == text for host in hosts)
        assert json.loads(relays.plan_path(state).read_text()) == relay_plan
        out = capsys.readouterr().out
        assert "Fabric verified: 2 cables on 3 Sparks (path-3), relays 4 rules and 4 routes, reboot-persistent." in out
        (state / "cluster.json").write_text(json.dumps(cluster))
        assert fabric.verify(state, access=Fleet(hosts)) == 0
        assert "Fabric verified: 2 cables on 3 Sparks (path-3)" in capsys.readouterr().out
        hosts[2].disabled.add("sparkring-fabric.service")
        assert fabric.verify(state, access=Fleet(hosts)) == 1
        assert "spark2 (position 2) unit sparkring-fabric.service: disabled" in capsys.readouterr().out
        assert len(list((state / fabric.REPORTS).glob("fabric-verify-*.json"))) >= 2
        assert fabric.show(state) == 0
        shown = capsys.readouterr().out
        assert "position 0 spark0: port 0 → position 1 spark1 port 1 (cable 0); port 1 free" in shown
        assert "cable 1: position 1 port 0 ↔ position 2 port 1; 198.18.2.0/24, 198.18.3.0/24; degraded" in shown
    finally:
        node.hairpin_rows = hairpin


def test_show_explains_a_cluster_without_a_document(tmp_path, capsys):
    plan = plan_of(fabric_layout.layout("cycle", 4))
    (tmp_path / "cluster.json").write_text(json.dumps({"name": "test", "plan": plan}))
    assert fabric.show(tmp_path) == 1
    assert "per-deployment mesh service" in capsys.readouterr().out


def test_relayed_lanes_pair_each_route_with_its_way_back():
    _, document, relay_plan = document_of(fabric_layout.layout("cycle", 8))
    lanes = fabric.relayed_lanes(document, relay_plan)
    assert len(lanes) == sum(len(p["routes"]) for p in relay_plan["positions"]) // 2 == 48
    first = lanes[0]
    assert first["client"]["rank"] == 0 and first["relays"] >= 1
    assert first["server"]["rdma_device"] in fabric_layout.DEVICES.values()


def test_status_names_the_fabric_its_relays_and_the_last_verification(tmp_path):
    plan, document, relay_plan = document_of(fabric_layout.layout("cycle", 4))
    cluster = {"name": "test", "plan": plan}
    value = fabric.summary(tmp_path, cluster)
    assert value == {"layout": "cycle-4", "document": None, "relays": "mesh-service", "verified": None}
    assert fabric.status_line(value) == ("Fabric: cycle-4, relays: mesh service (run sudo sparkring setup to move to "
                                         "the fabric relay table); no fabric document (sudo sparkring setup records it)")
    recorded = dict(document, verified={"at": "2026-10-07T10:05:00Z", "result": "healthy", "report": "r.json",
                                        "after_boot": False})
    (tmp_path / "fabric.json").write_text(fabric_document.encoded(recorded), encoding="utf-8")
    value = fabric.summary(tmp_path, cluster)
    assert value["relays"] == "table" and value["document"] == document["id"]
    assert fabric.status_line(value) == ("Fabric: cycle-4, relay table restored at every boot; last verified healthy "
                                         "at 2026-10-07T10:05:00Z; sudo sparkring fabric verify checks it")


def test_kept_addresses_that_follow_the_rule_count_as_planned():
    plan = plan_of(fabric_layout.layout("cycle", 8))
    again = topology.build_spec(configured(plan), plan["nodes"][0]["node_id"])
    assert again["preserve_existing_addresses"]
    document, _ = fabric.prepare(again, cluster="test", marker=MARKER)
    assert document["addressing"] == "planned" and document["id"] == fabric.prepare(plan, cluster="test",
                                                                                     marker=MARKER)[0]["id"]
