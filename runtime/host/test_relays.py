"""The universal relay table: plans of paths and cycles, its persistent records, restore, checks and markers.

Plans come from synthetic fabric documents (``test_fabric_layouts.sparks``);
commands run against fakes. No host is contacted.
"""
import copy
import hashlib
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from runtime.common import fabric_document, fabric_layout
from runtime.host import fabric, node, relays, topology
from runtime.host.test_fabric_layouts import plan_of

MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}
RECORDED = json.loads((Path(__file__).parent / "fixtures/fabric/cycle8-relay-table.json").read_text(encoding="utf-8"))


def prepared(shape, size):
    plan = plan_of(fabric_layout.layout(shape, size))
    document, relay_plan = fabric.prepare(plan, cluster="test", marker=MARKER)
    return plan, document, relay_plan


def labels(document):
    return {function["mac"]: f"p{row['position']}:{function['netdev']}" for row in document["positions"]
            for entry in row["ports"].values() for function in entry["functions"].values() if function["mac"]}


@pytest.mark.parametrize("shape, size, counts", [
    # (routes, relay filters, marker processes) per position.
    ("path", 4, [(4, 0, 2), (2, 6, 2), (2, 6, 2), (4, 0, 2)]),
    ("cycle", 4, [(4, 4, 4)] * 4),
    ("path", 6, [(8, 0, 2), (6, 10, 2), (6, 10, 4), (6, 10, 4), (6, 10, 2), (8, 0, 2)]),
    ("cycle", 8, [(12, 12, 4)] * 8),
])
def test_relay_plans_list_every_object_per_position(shape, size, counts):
    _, document, plan = prepared(shape, size)
    assert [(len(p["routes"]), len(p["filters"]), len(p["markers"])) for p in plan["positions"]] == counts
    assert all(len(p["neighbours"]) == len(p["routes"]) for p in plan["positions"])
    layout = fabric_layout.layout(shape, size)
    assert plan["max_relays"] == fabric_layout.max_relays(layout) and plan["fabric"] == document["id"]
    assert plan["sha256"] == relays.sha256(plan) and document["relays"]["plan_sha256"] == plan["sha256"]
    for row in plan["positions"]:
        for route in row["routes"]:
            assert route["hops"] == fabric_layout.hops(layout, row["position"], route["peer"]) >= 2
        for rule in row["filters"]:
            k = rule["handle"]
            assert rule["pref"] == 10 + k and rule["protocol"] == relays.hex16(relays.tag(k))
            assert rule["actions"][0]["pedit"]["eth_type"] == relays.hex16(relays.tag(k - 1))
    # SIRCL's route maps of the same group are carried by the plan.
    assert len(plan["sircl"]["route_maps"]) == size and plan["sircl"]["relay_egress"] == "same"


def test_cycle8_plan_equals_the_recorded_table_object_for_object():
    """The fixture is the table installed on an eight-Spark cycle, with MACs replaced by position labels."""
    _, document, plan = prepared("cycle", 8)
    names = labels(document)

    def key(value):
        return json.dumps(value, sort_keys=True)

    for mine, recorded in zip(plan["positions"], RECORDED["positions"], strict=True):
        assert sorted(key({"dst": r["dst"], "dev": r["dev"], "src": r["src"]}) for r in mine["routes"]) == sorted(
            key(r) for r in recorded["routes"])
        assert sorted(key({"addr": n["addr"], "dev": n["dev"], "lladdr": names[n["lladdr"]]})
                      for n in mine["neighbours"]) == sorted(key(n) for n in recorded["neighbours"])
        assert sorted(key({"dev": f["dev"], "protocol": f["protocol"], "pref": f["pref"], "handle": f["handle"],
                           "next": names[f["actions"][0]["pedit"]["eth_dst"]],
                           "eth_type": f["actions"][0]["pedit"]["eth_type"],
                           "redirect": f["actions"][1]["mirred"]["redirect"]}) for f in mine["filters"]) == sorted(
            key({k: v for k, v in f.items() if k != "in_hw"}) for f in recorded["filters"])


def test_four_cycle_marks_the_prepared_transports_devices_by_source_port_too():
    from spark_transport.fabric.cx7_hairpin_diagonal import fabric as cx7
    topology_ = cx7.load_topology(Path(__file__).resolve().parents[1] / "glm53-spark-mtp3-mesh/fabric.example.json")
    mesh = cx7.build_rocenante_plan(cx7.build_plan(topology_))
    expected = {}
    for marker in mesh.markers:
        expected.setdefault(marker.source_rank, set()).add(marker.rdma_device)
        assert (marker.udp_source_port, marker.marked_ether_type) == (65535, 0x88b5)
    _, _, plan = prepared("cycle", 4)
    for row in plan["positions"]:
        marked = {m["rdma"] for m in row["markers"] if "source_port" in m}
        assert marked == expected[row["position"]]
        assert all(m["source_port"] == {"port": 65535, "ethertype": "0x88b5"} for m in row["markers"]
                   if "source_port" in m)
        # The mesh rule's tag is the table's tag for one relay left, whose relay rule restores IPv4.
        assert {f["actions"][0]["pedit"]["eth_type"] for f in row["filters"]} == {"0x0800"}
    assert plan["marker"]["modes"] == {"by_destination": True, "source_port": 65535}
    for shape, size in (("cycle", 8), ("path", 4)):
        assert all("source_port" not in m for row in prepared(shape, size)[2]["positions"] for m in row["markers"])


def test_layouts_without_relays_have_an_empty_table_and_need_no_marker():
    for layout in (fabric_layout.layout("pair", 2), fabric_layout.layout("cycle", 3)):
        document, plan = fabric.prepare(plan_of(layout), cluster="test", marker=None)
        assert plan["marker"] is None and plan["sircl"] is None
        assert all(not (p["routes"] or p["filters"] or p["markers"]) for p in plan["positions"])
        assert document["relays"]["max_relays"] == 0 and "sircl" in document["transports"]


def test_without_the_marker_a_relaying_fabric_records_no_table():
    document, plan = fabric.prepare(plan_of(fabric_layout.layout("cycle", 4)), cluster="test", marker=None)
    assert plan is None and document["relays"] is None and document["transports"] == ["prepared"]
    lines = fabric.relay_lines(document, None)
    assert "built without the relay marker" in lines[0] and "per-deployment mesh service" in lines[1]


def test_each_spark_record_carries_its_section_and_validates_it():
    plan, document, relay_plan = prepared("cycle", 8)
    for rank in range(8):
        section = relays.section(relay_plan, rank)
        config = topology.persistent_config(plan, rank, relays=section)
        assert node.validate(config)["relays"]["plan_sha256"] == relay_plan["sha256"]
        broken = copy.deepcopy(config)
        broken["relays"]["routes"][0]["src"] = "198.18.200.1"
        with pytest.raises(ValueError, match="own source address"):
            node.validate(broken)
        broken = copy.deepcopy(config)
        broken["relays"]["filters"][0]["pref"] = 49152
        with pytest.raises(ValueError, match="SparkRing's preferences"):
            node.validate(broken)
        broken = copy.deepcopy(config)
        broken["relays"]["position"] = (rank + 1) % 8
        with pytest.raises(ValueError, match="another position"):
            node.validate(broken)


def test_marker_command_lines():
    _, _, plan = prepared("cycle", 4)
    section = relays.section(plan, 0)
    rows = {row["rdma"]: relays.marker_argv(section, row) for row in section["markers"]}
    assert rows["rocep1s0f0"] == [relays.MARKER_BINARY, "--device", "rocep1s0f0", "--rule", "198.18.2.2=0x88b5",
                                  "--source-port", "65535=0x88b5", "--managed"]
    assert rows["rocep1s0f1"] == [relays.MARKER_BINARY, "--device", "rocep1s0f1", "--rule", "198.18.4.1=0x88b5",
                                  "--managed"]


# Restoring and observing the table with a fake kernel.

class Kernel:
    """``ip`` and ``tc`` of one Spark: routes, permanent neighbours, qdiscs and filter preferences."""

    def __init__(self, root, netdevs, *, carrier=True):
        self.routes, self.neighbours, self.qdiscs, self.filters, self.calls = [], [], {}, {}, []
        for netdev in netdevs:
            path = Path(root) / "sys/class/net" / netdev
            path.mkdir(parents=True, exist_ok=True)
            (path / "carrier").write_text("1\n" if carrier else "0\n")

    def __call__(self, argv):
        self.calls.append(argv)
        output = ""
        if argv[:5] == ["ip", "-j", "-4", "route", "show"]:
            output = json.dumps(self.routes)
        elif argv[:4] == ["ip", "-j", "-4", "neigh"]:
            output = json.dumps(self.neighbours)
        elif argv[:4] == ["tc", "-j", "qdisc", "show"]:
            output = json.dumps(self.qdiscs.get(argv[5], []))
        elif argv[:4] == ["tc", "-j", "filter", "show"]:
            output = json.dumps(self.filters.get(argv[5], []))
        elif argv[:3] == ["tc", "qdisc", "add"]:
            self.qdiscs[argv[4]] = [{"kind": argv[5]}]
        elif argv[:3] == ["tc", "filter", "replace"]:
            pref, handle = int(argv[argv.index("pref") + 1]), int(argv[argv.index("handle") + 1])
            self.filters.setdefault(argv[4], []).append({"pref": pref, "kind": "flower"})
            self.filters[argv[4]].append({"pref": pref, "kind": "flower", "options": {
                "handle": handle, "keys": {"eth_type": argv[argv.index("protocol") + 1].removeprefix("0x")},
                "skip_sw": True, "in_hw": True,
                "actions": [{"kind": "pedit"}, {"kind": "pedit"}, {"kind": "mirred", "to_dev": argv[-1]}]}})
        elif argv[:3] == ["ip", "route", "replace"]:
            self.routes.append({"dst": argv[3].split("/")[0], "dev": argv[5], "prefsrc": argv[7], "scope": "link",
                                "protocol": argv[-1]})
        elif argv[:3] == ["ip", "neigh", "replace"]:
            self.neighbours.append({"dst": argv[3], "dev": argv[7], "lladdr": argv[5], "state": ["PERMANENT"]})
        return SimpleNamespace(returncode=0, stdout=output, stderr="")


def section_of(rank=1, shape="cycle", size=8):
    _, _, plan = prepared(shape, size)
    return relays.section(plan, rank)


def test_restore_adds_what_is_missing_and_then_finds_everything_present(tmp_path):
    section = section_of()
    netdevs = sorted({r["dev"] for r in section["routes"]} | {f["dev"] for f in section["filters"]})
    kernel = Kernel(tmp_path, netdevs)
    rows = relays.restore(section, call=kernel, root=tmp_path)
    assert {row["state"] for row in rows} == {"restored"}
    assert sum(argv[:3] == ["tc", "qdisc", "add"] for argv in kernel.calls) == 4
    assert all(argv[-1] == "clsact" for argv in kernel.calls if argv[:3] == ["tc", "qdisc", "add"])
    assert all(argv[-2:] == ["proto", "82"] for argv in kernel.calls if argv[:3] in (["ip", "route", "replace"],
                                                                                    ["ip", "neigh", "replace"]))
    again = relays.restore(section, call=kernel, root=tmp_path)
    assert {row["state"] for row in again} == {"present"}
    assert {row["state"] for row in relays.observe(section, call=kernel, root=tmp_path)} == {"present"}


def test_restore_keeps_an_existing_ingress_queue_and_never_replaces_foreign_routes(tmp_path):
    section = section_of()
    netdevs = sorted({r["dev"] for r in section["routes"]} | {f["dev"] for f in section["filters"]})
    kernel = Kernel(tmp_path, netdevs)
    for netdev in netdevs:
        kernel.qdiscs[netdev] = [{"kind": "ingress"}]
    foreign = section["routes"][0]
    kernel.routes.append({"dst": foreign["dst"].split("/")[0], "dev": "enP7s7", "prefsrc": "192.0.2.10",
                          "protocol": "static"})
    rows = relays.restore(section, call=kernel, root=tmp_path)
    assert not any(argv[:3] == ["tc", "qdisc", "add"] for argv in kernel.calls)
    assert next(row for row in rows if row.get("dst") == foreign["dst"])["state"] == "conflict"
    assert relays.missing(rows) == [next(row for row in rows if row.get("dst") == foreign["dst"])]


def test_objects_on_a_function_without_carrier_wait_for_the_link(tmp_path):
    section = section_of(rank=0, shape="path", size=4)
    netdevs = sorted({r["dev"] for r in section["routes"]})
    kernel = Kernel(tmp_path, netdevs, carrier=False)
    rows = relays.restore(section, call=kernel, root=tmp_path)
    assert {row["state"] for row in rows} == {"no-link"}
    assert not any(argv[:3] in (["ip", "route", "replace"], ["ip", "neigh", "replace"]) for argv in kernel.calls)


def test_observe_names_filters_that_differ_or_are_not_in_hardware(tmp_path):
    section = section_of()
    netdevs = sorted({r["dev"] for r in section["routes"]} | {f["dev"] for f in section["filters"]})
    kernel = Kernel(tmp_path, netdevs)
    relays.restore(section, call=kernel, root=tmp_path)
    first = section["filters"][0]
    entry = next(row for row in kernel.filters[first["dev"]] if row.get("pref") == first["pref"] and "options" in row)
    entry["options"]["in_hw"] = False
    second = next(f for f in section["filters"] if f["dev"] != first["dev"])
    entry = next(row for row in kernel.filters[second["dev"]] if row.get("pref") == second["pref"] and "options" in row)
    entry["options"]["actions"][-1]["to_dev"] = "enP7s7"
    states = {(row["dev"], row["pref"]): row["state"] for row in relays.observe(section, call=kernel, root=tmp_path)
              if row["kind"] == "filter"}
    assert states[(first["dev"], first["pref"])] == "not-in-hardware"
    assert states[(second["dev"], second["pref"])] == "different"


def proc(root, pid, argv):
    path = Path(root) / "proc" / str(pid)
    path.mkdir(parents=True)
    (path / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")


def test_markers_are_present_only_with_their_planned_command_line(tmp_path):
    section = section_of(rank=0, shape="cycle", size=4)
    rows = section["markers"]
    proc(tmp_path, 100, relays.marker_argv(section, rows[0]))
    proc(tmp_path, 101, [relays.MARKER_BINARY, "--device", rows[1]["rdma"], "--managed"])
    states = {row["rdma"]: row["state"] for row in relays.check_markers(section, root=tmp_path)}
    assert states == {rows[0]["rdma"]: "present", rows[1]["rdma"]: "different", rows[2]["rdma"]: "missing",
                      rows[3]["rdma"]: "missing"}


class Child:
    def __init__(self, argv, exit_after=None):
        self.argv, self.code, self.signals, self.exit_after = argv, None, [], exit_after

    def poll(self):
        if self.exit_after is not None:
            self.exit_after -= 1
            if self.exit_after < 0:
                self.code = 1
        return self.code

    def send_signal(self, number):
        self.signals.append(number)
        self.code = 0

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


def test_the_supervisor_starts_one_marker_per_device_and_exits_when_one_stops():
    section = section_of(rank=0, shape="cycle", size=4)
    children = []

    def popen(argv, **kwargs):
        children.append(Child(argv, exit_after=2 if not children else None))
        return children[-1]

    assert relays.supervise(section, popen=popen, digest=lambda path: "ab" * 32, poll=0) == 1
    assert [child.argv for child in children] == [relays.marker_argv(section, row) for row in section["markers"]]
    assert all(child.signals == [signal.SIGTERM] for child in children[1:])
    with pytest.raises(ValueError, match="differs from the relay marker that setup recorded"):
        relays.supervise(section, popen=popen, digest=lambda path: "cd" * 32, poll=0)


# The prepared transport over the table.

def write_state(root, document, config):
    path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    path.parent.mkdir(parents=True)
    path.write_text(fabric_document.encoded(document), encoding="utf-8", newline="\n")
    node.save(root, "/etc/sparkring/fabric.json", config)


def test_a_four_cycle_with_a_persistent_table_gives_deployments_a_fabric_reference(tmp_path):
    plan, document, relay_plan = prepared("cycle", 4)
    cluster = {"name": "test", "plan": plan}
    (tmp_path / "fabric.json").write_text(fabric_document.encoded(document), encoding="utf-8", newline="\n")
    reference = relays.persistent_reference(tmp_path, cluster)
    assert reference == {"site_path": fabric_document.HOST_PATH,
                         "site_sha256": hashlib.sha256(fabric_document.encoded(document).encode()).hexdigest(),
                         "plan_sha256": relay_plan["sha256"]}
    assert relays.is_reference(reference)
    for shape, size in (("cycle", 8), ("pair", 2)):
        other_plan, other, _ = prepared(shape, size)
        (tmp_path / "fabric.json").write_text(fabric_document.encoded(other), encoding="utf-8", newline="\n")
        assert relays.persistent_reference(tmp_path, {"name": "test", "plan": other_plan}) is None
    (tmp_path / "fabric.json").unlink()
    assert relays.persistent_reference(tmp_path, cluster) is None


def test_the_ring_check_over_the_table_checks_devices_addresses_and_every_object(tmp_path):
    plan, document, relay_plan = prepared("cycle", 4)
    rank = 2
    config = topology.persistent_config(plan, rank, relays=relays.section(relay_plan, rank))
    write_state(tmp_path, document, config)
    section = config["relays"]
    netdevs = sorted({r["dev"] for r in section["routes"]} | {f["dev"] for f in section["filters"]})
    kernel = Kernel(tmp_path, netdevs)
    for pid, row in enumerate(section["markers"], 200):
        proc(tmp_path, pid, relays.marker_argv(section, row))
    reference = {"site_path": fabric_document.HOST_PATH,
                 "site_sha256": hashlib.sha256(fabric_document.encoded(document).encode()).hexdigest(),
                 "plan_sha256": relay_plan["sha256"]}
    hcas = ["rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1"]
    host_ip = document["positions"][rank]["management"]["address"]
    with pytest.raises(ValueError, match="Relay table incomplete on this Spark: filter"):
        relays.check_reference(reference, rank, hcas, 3, host_ip, root=tmp_path, call=lambda argv: kernel(argv))
    relays.restore(section, call=kernel, root=tmp_path)
    rows = relays.check_reference(reference, rank, hcas, 3, host_ip, root=tmp_path, call=lambda argv: kernel(argv))
    assert {row["state"] for row in rows} == {"present"}
    with pytest.raises(ValueError, match="HCA order"):
        relays.check_reference(reference, rank, hcas[::-1], 3, host_ip, root=tmp_path, call=kernel)
    with pytest.raises(ValueError, match="bootstrap address"):
        relays.check_reference(reference, rank, hcas, 3, "192.0.2.99", root=tmp_path, call=kernel)
    with pytest.raises(ValueError, match="differs from the deployment's"):
        relays.check_reference(dict(reference, plan_sha256="0" * 64), rank, hcas, 3, host_ip, root=tmp_path,
                               call=kernel)


def test_a_sircl_group_refers_to_the_fabric_document_on_any_layout_with_a_persistent_table(tmp_path):
    for shape, size in (("cycle", 8), ("path", 5), ("cycle", 4)):
        plan, document, relay_plan = prepared(shape, size)
        cluster = {"name": "test", "plan": plan}
        (tmp_path / "fabric.json").write_text(fabric_document.encoded(document), encoding="utf-8", newline="\n")
        assert relays.group_reference(tmp_path, cluster) == {
            "site_path": fabric_document.HOST_PATH,
            "site_sha256": hashlib.sha256(fabric_document.encoded(document).encode()).hexdigest(),
            "plan_sha256": relay_plan["sha256"]}
    # On a four-Spark cycle it is the prepared transport's reference too.
    assert relays.group_reference(tmp_path, cluster) == relays.persistent_reference(tmp_path, cluster)
    # The recorded document of four Sparks does not describe a cluster of eight.
    other_plan, _, _ = prepared("cycle", 8)
    with pytest.raises(ValueError, match="describes other Sparks"):
        relays.group_reference(tmp_path, {"name": "test", "plan": other_plan})
    plan = plan_of(fabric_layout.layout("cycle", 8))
    without, _ = fabric.prepare(plan, cluster="test", marker=None)
    (tmp_path / "fabric.json").write_text(fabric_document.encoded(without), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="relay table is not installed"):
        relays.group_reference(tmp_path, {"name": "test", "plan": plan})
    (tmp_path / "fabric.json").unlink()
    with pytest.raises(ValueError, match="no fabric document"):
        relays.group_reference(tmp_path, cluster)


def test_a_sircl_rank_checks_the_relay_table_at_its_own_fabric_position(tmp_path):
    plan, document, relay_plan = prepared("cycle", 8)
    position = 5
    config = topology.persistent_config(plan, position, relays=relays.section(relay_plan, position))
    write_state(tmp_path, document, config)
    section = config["relays"]
    netdevs = sorted({r["dev"] for r in section["routes"]} | {f["dev"] for f in section["filters"]})
    kernel = Kernel(tmp_path, netdevs)
    for pid, row in enumerate(section["markers"], 200):
        proc(tmp_path, pid, relays.marker_argv(section, row))
    reference = {"site_path": fabric_document.HOST_PATH,
                 "site_sha256": hashlib.sha256(fabric_document.encoded(document).encode()).hexdigest(),
                 "plan_sha256": relay_plan["sha256"]}
    host_ip = document["positions"][position]["management"]["address"]
    relays.restore(section, call=kernel, root=tmp_path)
    rows = relays.check_position(reference, position, 3, host_ip, root=tmp_path, call=kernel)
    assert {row["state"] for row in rows} == {"present"}
    # The position, not the deployment rank, names the Spark's record: rank 1 of a group on 4-7 is position 5.
    with pytest.raises(ValueError, match="relay table differs"):
        relays.check_position(reference, 1, 3, host_ip, root=tmp_path, call=kernel)
    with pytest.raises(ValueError, match="bootstrap address"):
        relays.check_position(reference, position, 3, "192.0.2.99", root=tmp_path, call=kernel)
    assert relays.position_devices(position, root=tmp_path) == ["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1",
                                                                  "roceP2p1s0f1"]


def test_restore_at_boot_installs_the_table_after_routes_and_settings(tmp_path):
    plan, document, relay_plan = prepared("path", 4)
    config = topology.persistent_config(plan, 1, relays=relays.section(relay_plan, 1))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if "-C" in argv else 0, stdout="[]", stderr="")

    from runtime.host.test_fabric_layouts import configured
    facts = configured(plan)[1]["facts"]
    for port in config["interfaces"]:
        path = tmp_path / "sys/class/net" / port["netdev"]
        path.mkdir(parents=True)
        (path / "carrier").write_text("1\n")
    result = node.restore(config, collect=lambda request: facts, run=run, root=tmp_path)
    kinds = [argv[:3] for argv in calls]
    assert ["ip", "route", "add"] in kinds and ["tc", "qdisc", "add"] in kinds and ["tc", "filter", "replace"] in kinds
    assert kinds.index(["ip", "route", "add"]) < kinds.index(["tc", "qdisc", "add"])
    assert result["relays"]["missing"] == [] and result["relays"]["restored"] == len(
        config["relays"]["routes"]) * 2 + len(config["relays"]["filters"]) + len({f["dev"] for f in config["relays"]["filters"]})


def test_configure_enables_the_marker_unit_only_with_markers(tmp_path):
    plan, document, relay_plan = prepared("cycle", 4)
    node.save(tmp_path, "/etc/sparkring/node.json", {"schema": "sparkring-node/v1", "node_id": plan["spec"]["hosts"][0]["node_id"]})
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    from runtime.host.test_fabric_layouts import configured
    facts = configured(plan)[0]["facts"]
    with_table = topology.persistent_config(plan, 0, relays=relays.section(relay_plan, 0))
    assert node.configure(with_table, root=tmp_path, collect=lambda r: facts, run=run)["relay_markers"]
    assert ["systemctl", "enable", relays.MARKER_UNIT] in calls and ["systemctl", "restart", relays.MARKER_UNIT] in calls
    # A setup of the same Sparks may refresh the layout and relay fields of the record.
    calls.clear()
    without = topology.persistent_config(plan, 0)
    assert not node.configure(without, root=tmp_path, collect=lambda r: facts, run=run)["relay_markers"]
    assert ["systemctl", "disable", "--now", relays.MARKER_UNIT] in calls
    changed = copy.deepcopy(without)
    changed["routes"] = []
    with pytest.raises(ValueError, match="another approved configuration"):
        node.configure(changed, root=tmp_path, collect=lambda r: facts, run=run)


def test_configure_refuses_relay_markers_beside_a_running_mesh_service(tmp_path):
    plan, document, relay_plan = prepared("cycle", 4)
    node.save(tmp_path, "/etc/sparkring/node.json", {"schema": "sparkring-node/v1",
                                                     "node_id": plan["spec"]["hosts"][0]["node_id"]})

    def run(argv, **kwargs):
        listed = "sparkring-mesh.service loaded active running Mesh\n" if "list-units" in argv else ""
        return SimpleNamespace(returncode=0, stdout=listed, stderr="")

    from runtime.host.test_fabric_layouts import configured
    facts = configured(plan)[0]["facts"]
    with pytest.raises(ValueError, match=r"mesh service runs on this Spark \(sparkring-mesh.service\)"):
        node.configure(topology.persistent_config(plan, 0, relays=relays.section(relay_plan, 0)), root=tmp_path,
                       collect=lambda r: facts, run=run)
    assert not (tmp_path / "etc/sparkring/fabric.json").exists()


def test_relay_markers_entry_runs_the_records_markers(tmp_path):
    plan, document, relay_plan = prepared("cycle", 4)
    config = topology.persistent_config(plan, 3, relays=relays.section(relay_plan, 3))
    node.save(tmp_path, "/etc/sparkring/fabric.json", config)
    seen = []

    def supervise(value, popen):
        seen.append(value)
        return 0

    original = relays.supervise
    relays.supervise = supervise
    try:
        assert node.relay_markers(root=tmp_path) == 0
    finally:
        relays.supervise = original
    assert seen == [config["relays"]]
    node.save(tmp_path, "/etc/sparkring/fabric.json", topology.persistent_config(plan, 3))
    assert node.relay_markers(root=tmp_path) == 0


MUTATIONS = (["ip", "route", "add"], ["ip", "route", "replace"], ["ip", "route", "del"], ["ip", "neigh", "replace"],
             ["ip", "neigh", "add"], ["ip", "neigh", "del"], ["tc", "filter", "replace"], ["tc", "filter", "add"],
             ["tc", "filter", "del"], ["tc", "qdisc", "add"], ["tc", "qdisc", "del"], ["ip", "addr", "add"],
             ["ip", "addr", "del"], ["ip", "link", "set"])
NETWORK_TOOLS = ("nmcli", "netplan", "devlink", "modprobe", "rmmod", "ethtool", "iptables", "sysctl")


def adopted_spark(tmp_path, rank=1):
    """A Spark of an eight-Spark ring whose relay table is already in place, and its observed record."""
    plan, document, relay_plan = prepared("cycle", 8)
    section = relays.section(relay_plan, rank)
    config = topology.persistent_config(plan, rank, relays=section)
    config.update(ownership="observed", routes=[], forwarding=[])
    netdevs = sorted({r["dev"] for r in section["routes"]} | {f["dev"] for f in section["filters"]})
    kernel = Kernel(tmp_path, netdevs)
    relays.restore(section, call=kernel, root=tmp_path)
    kernel.calls.clear()
    node.save(tmp_path, "/etc/sparkring/node.json", {"schema": "sparkring-node/v1",
                                                     "node_id": plan["spec"]["hosts"][rank]["node_id"]})
    from runtime.host.test_fabric_layouts import configured
    return config, kernel, configured(plan)[rank]["facts"]


def test_adoption_keeps_a_present_relay_table_as_it_is_and_changes_no_network(tmp_path):
    config, kernel, facts = adopted_spark(tmp_path)
    before = copy.deepcopy((kernel.routes, kernel.neighbours, kernel.qdiscs, kernel.filters))
    # The Spark records another cluster's setup, which adoption moves aside.
    node.save(tmp_path, "/etc/sparkring/fabric.json", {"schema": "another-record"})
    services = []

    def run(argv, **kwargs):
        if argv[0] == "systemctl":
            services.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return kernel(argv)
    with pytest.raises(ValueError, match="records another setup"):
        node.adopt(config, root=tmp_path, collect=lambda request: facts, run=run)
    result = node.adopt(config, root=tmp_path, collect=lambda request: facts, run=run, retire=True)
    assert result["network_changed"] is False and result["relay_markers"] is True
    assert result["relays"]["restored"] == 0 and result["relays"]["missing"] == []
    # Nothing that changes an interface, address, connection, route, neighbour, filter or driver ran.
    commands = [argv for argv in kernel.calls]
    assert not [argv for argv in commands if argv[:3] in MUTATIONS or argv[0] in NETWORK_TOOLS]
    assert (kernel.routes, kernel.neighbours, kernel.qdiscs, kernel.filters) == before
    assert [argv for argv in services if argv[1] != "list-units"] == [
        ["systemctl", "enable", node.FABRIC_UNIT], ["systemctl", "restart", node.FABRIC_UNIT],
        ["systemctl", "enable", relays.MARKER_UNIT], ["systemctl", "restart", relays.MARKER_UNIT]]
    # The other record is kept with a receipt that says how to put it back.
    retired = tmp_path / result["retired"].lstrip("/")
    assert json.loads((retired / "etc/sparkring/fabric.json").read_text()) == {"schema": "another-record"}
    receipt = json.loads((retired / "receipt.json").read_text())
    assert receipt["restore"].startswith("sudo mv /var/lib/sparkring/retired/")
    assert node.read(tmp_path, "/etc/sparkring/fabric.json") == config


def test_an_adopted_record_restores_only_its_relay_table_at_boot(tmp_path):
    config, kernel, facts = adopted_spark(tmp_path)
    kernel.routes.clear()
    kernel.neighbours.clear()
    result = node.restore(config, collect=lambda request: facts, run=lambda argv, **kwargs: kernel(argv), root=tmp_path)
    assert result["relays"]["missing"] == [] and result["relays"]["restored"] == len(config["relays"]["routes"]) + len(
        config["relays"]["neighbours"])
    assert not [argv for argv in kernel.calls if argv[0] in NETWORK_TOOLS or argv[:3] == ["ip", "route", "add"]]
    # Without a relay table an adopted record still restores nothing.
    bare = dict(config, relays=None)
    with pytest.raises(ValueError, match="existing service"):
        node.restore(bare, collect=lambda request: facts, run=lambda argv, **kwargs: kernel(argv), root=tmp_path)


def test_the_agent_keeps_an_adopted_relay_table_and_nothing_else(tmp_path):
    config, kernel, _ = adopted_spark(tmp_path)
    kernel.neighbours.clear()

    def run(argv, **kwargs):
        if argv[0] == "systemctl":
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        return kernel(argv)
    result = node.restore_fabric(config, root=tmp_path, run=run, log=lambda line: None)
    assert result["active"] is True and result["routes"] == [] and result["settings"] == []
    assert {row["state"] for row in result["relays"] if row["kind"] == "neighbour"} == {"restored"}
    assert not [argv for argv in kernel.calls if argv[0] in NETWORK_TOOLS or argv[:3] == ["ip", "route", "add"]]
