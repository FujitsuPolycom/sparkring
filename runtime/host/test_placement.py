"""Placements on arcs of the fabric: parsing, groups, slots, per-rank fabric selection and API addresses."""
import pytest

from runtime.common import fabric_layout, installer
from runtime.host import controller, placement
from runtime.host.test_fabric_ssh import cluster

QWEN = "qwen38-flash-next-tp2"
GLM = "glm53-flash-nvfp4-spark-tp2"
TP4 = "qwen38-flash-next-qad-tp4"


CYCLE4 = fabric_layout.layout("cycle", 4)
CYCLE8 = fabric_layout.layout("cycle", 8)
PATH6 = fabric_layout.layout("path", 6)


@pytest.mark.parametrize("text, layout, expected", [
    ("0,1", CYCLE4, (0, 1)), ("2,3", CYCLE4, (2, 3)), (" 2, 3", CYCLE4, (2, 3)), ("1,2", CYCLE4, (1, 2)),
    ("3,0", CYCLE4, (3, 0)), ("0-3", CYCLE8, (0, 1, 2, 3)), ("4-7", CYCLE8, (4, 5, 6, 7)),
    ("6-1", CYCLE8, (6, 7, 0, 1)), ("6,7,0,1", CYCLE8, (6, 7, 0, 1)), ("2-4", PATH6, (2, 3, 4)),
    ("0-3", CYCLE4, None), ("0,1,2,3", CYCLE4, None), ("0-7", CYCLE8, None)])
def test_on_names_an_arc_in_cable_order_and_every_spark_from_node_a_is_the_whole_fabric(text, layout, expected):
    assert placement.parse(text, layout) == expected


@pytest.mark.parametrize("text, layout, message", [
    ("1,0", CYCLE4, "names no consecutive Sparks of this cycle-4"),
    ("0,2", CYCLE8, "names no consecutive Sparks"),
    ("0", CYCLE4, "names no consecutive Sparks"), ("a,b", CYCLE8, "names no consecutive Sparks"),
    ("", CYCLE4, "names no consecutive Sparks"), ("4-1", PATH6, "names no consecutive Sparks of this path-6"),
    ("6-8", CYCLE8, "names no consecutive Sparks"), ("3-3", CYCLE8, "names no consecutive Sparks"),
    ("2-1", CYCLE4, "every Spark from position 2 on; a model on every Spark runs with Node A as rank 0"),
    ("0,1", fabric_layout.layout("pair", 2), "this cluster is a pair")])
def test_other_rank_sets_are_refused_with_the_forms_the_fabric_accepts(text, layout, message):
    with pytest.raises(ValueError, match=message):
        placement.parse(text, layout)


def test_an_arc_has_a_group_shape_a_flag_a_text_and_a_slot():
    assert placement.group(CYCLE8, (6, 7, 0, 1)) == {"shape": "path", "size": 4, "name": "path-4",
                                                     "positions": [6, 7, 0, 1]}
    assert placement.group(CYCLE8, None)["name"] == "cycle-8" and placement.group(CYCLE4, (2, 3))["name"] == "pair"
    assert placement.group(PATH6, None)["name"] == "path-6"
    assert [placement.flag(arc) for arc in ((2, 3), (4, 5, 6, 7), (6, 7, 0, 1), None)] == [
        "--on 2,3", "--on 4-7", "--on 6-1", ""]
    assert [placement.text(arc, 8) for arc in ((2, 3), (4, 5, 6, 7), (6, 7, 0, 1), None)] == [
        "Sparks 2 and 3", "Sparks 4-7", "Sparks 6, 7, 0 and 1", "all eight Sparks"]
    assert placement.text(None, 4) == "all four Sparks" and placement.text(None, 2) == "both Sparks"
    assert placement.instance_label((6, 7, 0, 1)) == "on-6-7-0-1" and placement.instance_label((2, 3)) == "on-2-3"
    assert placement.slot_directory("state", (4, 5, 6, 7)).as_posix() == "state/slots/4-5-6-7"
    assert placement.forwarding_positions(CYCLE8, (6, 7, 0, 1)) == [7, 0]
    assert placement.forwarding_positions(CYCLE8, (2, 3)) == []
    assert placement.forwarding_positions(CYCLE8, None) == list(range(8))


def test_lines_of_six_or_more_sparks_are_unsupported_and_cycles_of_eight_are_not():
    assert placement.unsupported(CYCLE8, (0, 1, 2, 3, 4)) is None and placement.unsupported(CYCLE8, None) is None
    for arc in ((0, 1, 2, 3, 4, 5), (2, 3, 4, 5, 6, 7, 0)):
        assert "SIRCL ring sessions cross at most three relays" in placement.unsupported(CYCLE8, arc)
    assert "line of six Sparks" in placement.unsupported(PATH6, None)


def test_an_arc_needs_the_profiles_size():
    assert placement.check((2, 3), layout=CYCLE4, profile_nodes=2, profile=QWEN) == (2, 3)
    assert placement.check((4, 5, 6, 7), layout=CYCLE8, profile_nodes=4, profile=TP4) == (4, 5, 6, 7)
    assert placement.check(None, layout=CYCLE4, profile_nodes=4, profile=TP4) is None
    with pytest.raises(ValueError, match=f"{TP4} uses all four Sparks; --on applies to two-Spark profiles"):
        placement.check((0, 1), layout=CYCLE4, profile_nodes=4, profile=TP4)
    with pytest.raises(ValueError, match=f"{TP4} serves four Sparks; Sparks 0-5 are six"):
        placement.check((0, 1, 2, 3, 4, 5), layout=CYCLE8, profile_nodes=4, profile=TP4)
    with pytest.raises(ValueError, match="cross at most three relays"):
        placement.check((0, 1, 2, 3, 4, 5), layout=CYCLE8, profile_nodes=6, profile="six")
    with pytest.raises(ValueError, match="serves four Sparks and this cycle-8 has eight"):
        placement.check(None, layout=CYCLE8, profile_nodes=4, profile=TP4)


def test_slots_on_disjoint_arcs_do_not_conflict_and_overlapping_ones_do(tmp_path):
    for slot in ((0, 1, 2, 3), (4, 5), (6, 7)):
        placement.record(tmp_path, slot, tmp_path / ("d" + placement.slot_name(slot)))
    assert placement.slots(tmp_path) == [None, (0, 1, 2, 3), (4, 5), (6, 7)]
    assert placement.conflicting(tmp_path, (4, 5, 6, 7)) == [None, (4, 5), (6, 7)]
    assert placement.conflicting(tmp_path, (2, 3)) == [None, (0, 1, 2, 3)]
    assert placement.conflicting(tmp_path, None) == [(0, 1, 2, 3), (4, 5), (6, 7)]
    (tmp_path / "slots" / "unrelated").mkdir()
    assert placement.slots(tmp_path)[1:] == [(0, 1, 2, 3), (4, 5), (6, 7)]


def test_each_rank_of_a_half_uses_the_port_functions_facing_its_partner():
    value = cluster(4)
    # The fixture ring keeps its recorded addresses: edge e uses 198.18.(e+1).0/24 and 198.18.(101+e).0/24.
    assert placement.fabric_rows(value, (0, 1)) == [
        {"host": "root@192.0.2.10", "management_ip": "192.0.2.10", "fabric_ip": "198.18.1.1",
         "interface": "enp1s0f0np0", "hcas": ["rocep1s0f0", "roceP2p1s0f0"]},
        {"host": "root@192.0.2.11", "management_ip": "192.0.2.11", "fabric_ip": "198.18.1.2",
         "interface": "enp1s0f1np1", "hcas": ["rocep1s0f1", "roceP2p1s0f1"]}]
    assert placement.fabric_rows(value, (2, 3)) == [
        {"host": "root@192.0.2.12", "management_ip": "192.0.2.12", "fabric_ip": "198.18.3.1",
         "interface": "enp1s0f0np0", "hcas": ["rocep1s0f0", "roceP2p1s0f0"]},
        {"host": "root@192.0.2.13", "management_ip": "192.0.2.13", "fabric_ip": "198.18.3.2",
         "interface": "enp1s0f1np1", "hcas": ["rocep1s0f1", "roceP2p1s0f1"]}]


def test_a_half_whose_recorded_ports_share_no_subnet_is_refused():
    value = cluster(4)
    port = next(p for p in value["plan"]["spec"]["hosts"][3]["data_interfaces"] if p["role"] == "ccw_secondary")
    port["address"] = "198.18.77.2/24"
    with pytest.raises(ValueError, match="Sparks 2 and 3 do not share the cable"):
        placement.fabric_rows(value, (2, 3))
    assert placement.fabric_rows(value, (0, 1))[1]["fabric_ip"] == "198.18.1.2"


def routes(plan, rank, rows, interfaces=None):
    host = plan["spec"]["hosts"][rank]["host"]
    facts = plan["inventory"]["hosts"][host]
    facts["routes"] = rows
    if interfaces is not None:
        facts["interfaces"] += interfaces


def test_a_sparks_lan_address_is_on_its_default_route_outside_the_fabric_and_the_admin_network():
    value = cluster(4)
    plan = value["plan"]
    assert placement.lan_address(plan, 2) == "192.0.2.12"
    routes(plan, 2, [{"dst": "default", "dev": "sr-control", "gateway": "10.42.0.1"}],
           [{"name": "sr-control", "ipv4": ["10.42.0.3/24"]}])
    assert placement.lan_address(plan, 2) is None
    routes(plan, 2, [{"dst": "default", "dev": "enp1s0f0np0", "gateway": "198.18.3.2"}])
    assert placement.lan_address(plan, 2) is None
    routes(plan, 2, [{"dst": "default", "dev": "sr-control", "metric": 50},
                     {"dst": "default", "dev": "enP7s7", "metric": 100, "prefsrc": "192.0.2.12"},
                     {"dst": "default", "dev": "enP7s7", "table": "local"}])
    assert placement.lan_address(plan, 2) == "192.0.2.12"


def test_half_two_three_advertises_spark_twos_own_lan_address():
    value = cluster(4)
    assert placement.api_address(value, (2, 3)) == (None, None)
    value["api_address"] = "192.0.2.10"
    plan = value["plan"]
    routes(plan, 2, [{"dst": "default", "dev": "enP7s7"}],
           [{"name": "enP7s8", "ipv4": ["169.254.3.1/16"]}])
    plan["inventory"]["hosts"]["root@192.0.2.12"]["interfaces"][0]["ipv4"] = ["198.51.100.12/24"]
    assert placement.api_address(value, (2, 3)) == ("198.51.100.12", None)
    assert placement.api_address(value, (0, 1)) == (None, None)
    routes(plan, 2, [{"dst": "default", "dev": "sr-control"}])
    address, note = placement.api_address(value, (2, 3))
    assert address is None and note.startswith("Spark 2 has no LAN connection of its own")


def test_a_half_site_records_its_placement_fabric_and_api_address():
    value = cluster(4)
    value["api_address"] = "198.51.100.10"
    plan = value["plan"]
    plan["inventory"]["hosts"]["root@192.0.2.12"]["interfaces"][0]["ipv4"] = ["198.51.100.12/24"]
    site = controller.model_site(value, GLM, "iabc", (2, 3))
    assert site["placement"] == [2, 3] and site["api_address"] == "198.51.100.12"
    assert site["controller_address"] == "192.0.2.10"
    assert [(row["host"], row["fabric_ip"], row["interface"], row["hcas"], row["node_id"]) for row in site["hosts"]] == [
        ("root@192.0.2.12", "198.18.3.1", "enp1s0f0np0", ["rocep1s0f0", "roceP2p1s0f0"], plan["nodes"][2]["node_id"]),
        ("root@192.0.2.13", "198.18.3.2", "enp1s0f1np1", ["rocep1s0f1", "roceP2p1s0f1"], plan["nodes"][3]["node_id"])]
    first = controller.model_site(value, QWEN, "iabd", (0, 1))
    assert first["api_address"] == "198.51.100.10" and first["placement"] == [0, 1]


def environment(lock, rank):
    return installer.specifications(lock, only_rank=rank)[0].environment


def command(lock, rank):
    return list(installer.specifications(lock, only_rank=rank)[0].command)


@pytest.mark.parametrize("half, addresses, master", [((0, 1), ("198.18.1.1", "198.18.1.2"), "198.18.1.1"),
                                                     ((2, 3), ("198.18.3.1", "198.18.3.2"), "198.18.3.1")])
def test_each_rank_of_a_half_serves_on_the_ports_facing_its_partner(half, addresses, master):
    value = cluster(4)
    site = controller.model_site(value, QWEN, "itest", half)
    for row in site["hosts"]:
        row["model"] = "/srv/sparkring/test/checkpoints/model"
    lock = installer.make_lock(QWEN, site, "1" * 40, "2" * 64)
    expected = [("=rocep1s0f0,roceP2p1s0f0", "rocep1s0f0,roceP2p1s0f0", "1=0/1", "enp1s0f0np0"),
                ("=rocep1s0f1,roceP2p1s0f1", "rocep1s0f1,roceP2p1s0f1", "0=0/1", "enp1s0f1np1")]
    for rank, (nccl, b12x, peers, netdev) in enumerate(expected):
        env = environment(lock, rank)
        assert (env["NCCL_IB_HCA"], env["B12X_ROCE_HCA"], env["B12X_ROCE_PEER_HCA_MAP"]) == (nccl, b12x, peers)
        assert (env["VLLM_HOST_IP"], env["NCCL_SOCKET_IFNAME"], env["GLOO_SOCKET_IFNAME"]) == (
            addresses[rank], netdev, netdev)
        assert env["NCCL_IB_GID_INDEX"] == "3"
        arguments = command(lock, rank)
        assert arguments[arguments.index("--master-addr") + 1] == master
    assert lock["site"]["placement"] == list(half)


def test_pair_sites_and_containers_are_unchanged():
    value = cluster(2)
    site = controller.model_site(value, QWEN)
    nodes = value["plan"]["nodes"]
    # The pair's raw site, exactly as installations before ring halves recorded it.
    assert site == {"schema": "sparkring-install-site/v1", "name": "test-qwen38-flash-next--8f81f6",
                    "workspace": "/srv/sparkring/test/qwen38-flash-next-tp2",
                    "hosts": [{"host": "root@192.0.2.10", "management_ip": "192.0.2.10", "fabric_ip": "198.18.0.1",
                               "interface": "enp1s0f0np0", "node_id": nodes[0]["node_id"]},
                              {"host": "root@192.0.2.11", "management_ip": "192.0.2.11", "fabric_ip": "198.18.0.2",
                               "interface": "enp1s0f0np0", "node_id": nodes[1]["node_id"]}],
                    "controller_address": "192.0.2.10"}
    lock = installer.make_lock(QWEN, site, "1" * 40, "2" * 64)
    assert "placement" not in lock["site"] and "placement" not in lock["site_input"]
    assert [row["hcas"] for row in lock["site"]["ranks"]] == [installer.PAIR_HCAS] * 2
    for rank, peers in enumerate(("1=0/1", "0=0/1")):
        env = environment(lock, rank)
        assert (env["NCCL_IB_HCA"], env["B12X_ROCE_PEER_HCA_MAP"]) == ("=rocep1s0f0,roceP2p1s0f0", peers)
        assert (env["VLLM_HOST_IP"], env["NCCL_SOCKET_IFNAME"]) == (f"198.18.0.{rank + 1}", "enp1s0f0np0")
    phases = [phase["id"] for phase in installer.operation_plan(lock, "up")["phases"]]
    assert phases == ["prerequisites", "source", "model", "image", "gid-serve", "preflight", "create",
                      "start-workers", "start-api", "ready", "smoke", "model-settled"]
    # The lock records the raw site as given: no device list or placement enters a pair's identity.
    assert lock["site_input"] == site and not any("hcas" in row for row in lock["site_input"]["hosts"])


def test_a_half_parks_the_ring_mesh_before_restoring_its_gid():
    value = cluster(4)
    lock = installer.make_lock(QWEN, controller.model_site(value, QWEN, "itest", (2, 3)), "1" * 40, "2" * 64)
    phases = [phase["id"] for phase in installer.operation_plan(lock, "up")["phases"]]
    assert phases[phases.index("image") + 1:phases.index("preflight")] == ["ring-park", "gid-serve"]
    assert [action["host"] for action in installer.operation_plan(lock, "up")["phases"][phases.index("ring-park")]
            ["actions"]] == ["root@192.0.2.12", "root@192.0.2.13"]


@pytest.mark.parametrize("change, message", [
    (lambda site: site.update(placement=[1, 2]), "one half of a four-Spark ring"),
    (lambda site: site.update(placement=[2]), "one half of a four-Spark ring"),
    (lambda site: site["hosts"][1].update(hcas=["rocep1s0f1"]), "two distinct RDMA devices"),
    (lambda site: site["hosts"][1].update(hcas=["rocep1s0f1", "rocep1s0f1"]), "two distinct RDMA devices"),
    (lambda site: site["hosts"][1].update(hcas=["rocep1s0f1", "bad device"]), "two distinct RDMA devices"),
])
def test_a_half_site_with_another_placement_or_device_list_is_refused(change, message):
    site = controller.model_site(cluster(4), QWEN, "itest", (2, 3))
    change(site)
    with pytest.raises(ValueError, match=message):
        installer.make_lock(QWEN, site, "1" * 40, "2" * 64)


def test_a_four_spark_site_takes_no_placement_or_device_list():
    from runtime.common.test_installer import site
    raw = site(4)
    raw["placement"] = [0, 1]
    with pytest.raises(ValueError, match="for a two-Spark profile"):
        installer.site_document(raw, installer.setup.selection("glm53-flash-spark-tp4-dcp1-nocache"), "1" * 40)
    raw = site(4)
    raw["hosts"][0]["hcas"] = ["rocep1s0f0", "roceP2p1s0f0"]
    with pytest.raises(ValueError, match="two distinct RDMA devices"):
        installer.site_document(raw, installer.setup.selection("glm53-flash-spark-tp4-dcp1-nocache"), "1" * 40)
