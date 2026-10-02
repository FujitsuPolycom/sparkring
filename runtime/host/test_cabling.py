"""Cabling diagnosis from captured and synthetic LLDP and neighbor observations; no host is contacted."""
import json
from pathlib import Path

import pytest

from runtime.host import cabling, topology

FIXTURES = Path(__file__).parent / "fixtures" / "cabling"
NETDEVS = {(0, 0): "enp1s0f0np0", (0, 1): "enP2p1s0f0np0", (1, 0): "enp1s0f1np1", (1, 1): "enP2p1s0f1np1"}


def captured():
    """Four Sparks of two former pairs, recabled into one loop: spark-aa42 and spark-931e (pair "tp2"),
    spark-3286 and spark-0a0f (pair "sparkring"). Two cables join equal ports."""
    document = json.loads((FIXTURES / "recabled-pairs.json").read_text(encoding="utf-8"))
    return {s["hostname"]: cabling.from_capture(s["hostname"], s["lldp"], s["addresses"]) for s in document["sparks"]}


def mac(spark, port, function):
    return f"02:00:{spark:02x}:{port:02x}:{function:02x}:01"


def synthetic(count, cables, *, lldp=True, carrier=None, names=None):
    """Spark records whose LLDP shows ``cables``: [((spark, port), (spark, port)), ...].

    Every function also sees its sibling on the same port, as Socket Direct
    delivers it. ``carrier`` maps (spark, port) to a carrier value; a port
    without a cable has no carrier by default.
    """
    names = names or [f"spark-{chr(97 + i)}" for i in range(count)]
    far = {}
    for a, b in cables:
        far.setdefault(a, []).append(b)
        far.setdefault(b, []).append(a)
    sparks = []
    for i in range(count):
        functions, rows = [], []
        for (port, function), netdev in NETDEVS.items():
            state = (carrier or {}).get((i, port), (i, port) in far)
            functions.append({"netdev": netdev, "port": port, "mac": mac(i, port, function), "carrier": state})
            rows.append({netdev: {"chassis": {names[i]: {"id": {"type": "mac", "value": f"02:00:{i:02x}:ff:00:01"}}},
                                  "port": {"id": {"type": "mac", "value": mac(i, port, 1 - function)},
                                           "descr": NETDEVS[(port, 1 - function)]}}})
            for j, q in far.get((i, port), []):
                for g in (0, 1):
                    rows.append({netdev: {"chassis": {names[j]: {"id": {"type": "mac", "value": f"02:00:{j:02x}:ff:00:01"}}},
                                          "port": {"id": {"type": "mac", "value": mac(j, q, g)}, "descr": NETDEVS[(q, g)]}}})
        sparks.append({"key": names[i], "name": names[i], "reached": True, "functions": functions,
                       "macs": [f"02:00:{i:02x}:ff:00:01"] + [f["mac"] for f in functions],
                       "lldp": cabling.lldp_rows({"lldp": {"interface": rows}}) if lldp else None, "neighbors": None})
    return sparks


def ring(order_ports):
    """Cables around a loop: ``order_ports[i]`` is (port at Spark i, port at Spark i+1)."""
    count = len(order_ports)
    return [((i, a), ((i + 1) % count, b)) for i, (a, b) in enumerate(order_ports)]


def test_captured_loop_names_the_one_swap_and_the_ring_order():
    sparks = captured()
    result = cabling.diagnose(list(sparks.values()), "spark-3286")
    assert result["layout"] == "ring" and not result["ready"]
    assert result["fix"] == ["On spark-aa42, swap its two cables (port 0 ↔ port 1)."]
    assert result["order_names"] == ["spark-3286", "spark-0a0f", "spark-931e", "spark-aa42"]
    assert cabling.lines(result) == [
        "Cables:",
        "  spark-3286 port 0 ↔ spark-0a0f port 1",
        "  spark-0a0f port 0 ↔ spark-931e port 1",
        "  spark-931e port 0 ↔ spark-aa42 port 0",
        "  spark-aa42 port 1 ↔ spark-3286 port 1",
        "The four Sparks form a loop, but 2 cables join the same port number at both ends. "
        "In a ring, every cable runs from port 0 of one Spark to port 1 of the next.",
        "To fix:",
        "  On spark-aa42, swap its two cables (port 0 ↔ port 1).",
        "Ring order after the fix: spark-3286 → spark-0a0f → spark-931e → spark-aa42",
    ]
    # Every cable is confirmed from both ends, and same-chassis sibling observations are ignored.
    assert all(len(cable["seen_from"]) == 2 for cable in result["cables"])
    assert len(result["cables"]) == 4 and not result["notes"] and not result["problems"]


def test_captured_loop_from_another_node_a_keeps_the_swap_off_node_a():
    result = cabling.diagnose(list(captured().values()), "spark-aa42")
    # Swapping spark-aa42 alone fixes the loop; the alternative swaps the other three.
    assert result["fix"] == ["On spark-aa42, swap its two cables (port 0 ↔ port 1)."]
    assert result["order_names"] == ["spark-aa42", "spark-3286", "spark-0a0f", "spark-931e"]


def test_setup_check_refuses_the_captured_loop_with_the_physical_fix():
    result = cabling.diagnose(list(captured().values()), "spark-3286", strict=True, whole=True)
    assert result["fix"] == ["On spark-aa42, swap its two cables (port 0 ↔ port 1)."] and not result["ready"]
    error = cabling.CablingError(result)
    assert str(error) == (
        "Fabric cabling: The four Sparks form a loop, but 2 cables join the same port number at both ends. "
        "In a ring, every cable runs from port 0 of one Spark to port 1 of the next. To fix: On spark-aa42, swap its "
        "two cables (port 0 ↔ port 1). Then the ring order is spark-3286 → spark-0a0f → spark-931e → spark-aa42. "
        "sparkring cabling shows the cables.")
    # The lines below the error add the cables it describes.
    assert error.details["lines"] == ["Cables:", "  spark-3286 port 0 ↔ spark-0a0f port 1",
                                      "  spark-0a0f port 0 ↔ spark-931e port 1", "  spark-931e port 0 ↔ spark-aa42 port 0",
                                      "  spark-aa42 port 1 ↔ spark-3286 port 1"]


def test_captured_pair_on_port_one_of_the_worker_moves_one_cable_end():
    sparks = captured()
    pair = [sparks["spark-3286"], sparks["spark-0a0f"]]
    for spark in pair:
        spark["lldp"] = [row for row in spark["lldp"] if row["hostname"] in ("spark-3286", "spark-0a0f")]
    result = cabling.diagnose(pair, "spark-3286")
    assert result["layout"] == "pair" and not result["ready"]
    assert result["summary"] == ("Pair: spark-0a0f port 1 ↔ spark-3286 port 0, but no cable joins the two ports 0. "
                                 "Pair models use port 0 on both Sparks.")
    assert result["fix"] == ["On spark-0a0f, move the cable from port 1 to port 0."]
    assert result["order_names"] == ["spark-3286", "spark-0a0f"]


@pytest.mark.parametrize("cables, fix, ready", [
    ([((0, 0), (1, 0))], [], True),
    ([((0, 0), (1, 0)), ((0, 1), (1, 1))], [], True),
    ([((0, 0), (1, 1)), ((0, 1), (1, 0))], ["On spark-b, swap its two cables (port 0 ↔ port 1)."], False),
    ([((0, 1), (1, 1))], ["On spark-a, move the cable from port 1 to port 0.",
                          "On spark-b, move the cable from port 1 to port 0."], False),
    ([((0, 1), (1, 0))], ["On spark-a, move the cable from port 1 to port 0."], False),
])
def test_pair_needs_a_cable_between_both_ports_zero(cables, fix, ready):
    result = cabling.diagnose(synthetic(2, cables), "spark-a")
    assert (result["layout"], result["fix"], result["ready"]) == ("pair", fix, ready)
    if len(cables) == 2 and ready:
        assert result["notes"] == ["The cable between the ports 1 carries only the admin network's fallback path."]


def test_correct_ring_is_ready_in_port_zero_order_from_node_a():
    sparks = synthetic(4, ring([(0, 1)] * 4))
    result = cabling.diagnose(sparks, "spark-c", strict=True, whole=True)
    assert result["ready"] and result["layout"] == "ring" and result["fix"] == []
    assert result["order_names"] == ["spark-c", "spark-d", "spark-a", "spark-b"]
    assert cabling.lines(result)[-1] == "Ring order: spark-c → spark-d → spark-a → spark-b"


def test_same_port_cables_on_different_sparks_need_two_swaps_away_from_node_a():
    # spark-a p0-p0 spark-b and spark-c p1-p1 spark-d: swapping {b, c} or {a, d} fixes it; Node A stays untouched.
    cables = ring([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = cabling.diagnose(synthetic(4, cables), "spark-a")
    assert result["layout"] == "ring"
    assert result["fix"] == ["On spark-b, swap its two cables (port 0 ↔ port 1).",
                             "On spark-c, swap its two cables (port 0 ↔ port 1)."]
    assert result["order_names"] == ["spark-a", "spark-b", "spark-c", "spark-d"]
    # Applying the fix yields a ready ring in the announced order.
    fixed = [tuple((i, p ^ (i in (1, 2))) for i, p in cable) for cable in cables]
    after = cabling.diagnose(synthetic(4, fixed), "spark-a")
    assert after["ready"] and after["order_names"] == result["order_names"]


def test_a_port_in_two_cables_is_a_disagreement_not_a_layout():
    result = cabling.diagnose(synthetic(4, ring([(0, 0), (1, 0), (1, 1), (0, 0)])), "spark-a")
    assert result["layout"] == "unsupported" and not result["fix"]
    assert "spark-a port 0 sees more than one far port" in result["summary"]


def test_loose_cable_names_the_two_free_ports():
    # The loop spark-a → b → c → d → a with the d-to-a cable unplugged: both ends report no carrier.
    sparks = synthetic(4, ring([(0, 1)] * 4)[:3])
    result = cabling.diagnose(sparks, "spark-a")
    assert result["layout"] == "ring" and not result["ready"]
    assert result["summary"].startswith("Four Sparks are cabled in a line: spark-d port 0 and spark-a port 1 have no "
                                        "cable (missing or loose).")
    assert result["fix"] == ["Connect a cable from spark-d port 0 to spark-a port 1."]
    assert result["order_names"] == ["spark-a", "spark-b", "spark-c", "spark-d"]


def test_line_with_a_link_whose_far_end_is_unseen_is_incomplete():
    sparks = synthetic(4, ring([(0, 1)] * 4)[:3], carrier={(3, 0): True})
    result = cabling.diagnose(sparks, "spark-a")
    assert result["layout"] == "incomplete" and not result["fix"]


def test_three_sparks_are_not_supported():
    sparks = synthetic(3, [((0, 0), (1, 1)), ((1, 0), (2, 1))])
    result = cabling.diagnose(sparks, "spark-a")
    assert result["layout"] == "unsupported"
    assert result["summary"] == ("Three Sparks are cabled together (spark-a, spark-b, spark-c). SparkRing needs two "
                                 "Sparks (a pair) or four (a ring). Free ports: spark-a port 1, spark-c port 0.")


def test_duplicate_cable_in_a_four_spark_set():
    sparks = synthetic(4, [((0, 0), (1, 1)), ((0, 1), (1, 0)), ((2, 0), (3, 1)), ((2, 1), (3, 0))])
    whole = cabling.diagnose(sparks, "spark-a", whole=True)
    assert whole["layout"] == "unsupported"
    assert whole["summary"].startswith("spark-a and spark-b are joined by two cables, so they cannot also reach "
                                       "spark-c and spark-d.")
    with pytest.raises(cabling.CablingError, match="joined by two cables"):
        topology_free_check(sparks)
    # Without a fixed set, Node A's group is a pair and the other Sparks are noted.
    group = cabling.diagnose(sparks, "spark-a")
    assert group["layout"] == "pair"
    assert "spark-c was reached but is not cabled to spark-a's Sparks" in group["notes"]


def topology_free_check(sparks):
    result = cabling.diagnose(sparks, sparks[0]["key"], strict=True, whole=True)
    if not result["ready"]:
        raise cabling.CablingError(result)


def test_spark_cabled_to_itself():
    sparks = synthetic(1, [((0, 0), (0, 1))])
    result = cabling.diagnose(sparks, "spark-a")
    assert result["summary"] == "spark-a is cabled to itself (port 0 to port 1). Each cable must join two different Sparks."


def test_one_sided_cable_is_a_note_for_the_survey_and_blocks_setup():
    sparks = synthetic(4, ring([(0, 1)] * 4))
    sparks[2]["lldp"] = [row for row in sparks[2]["lldp"] if not row["netdev"].endswith("f0np0")]
    tolerant = cabling.diagnose(sparks, "spark-a")
    assert tolerant["ready"]
    assert tolerant["notes"] == ["spark-c port 0 ↔ spark-d port 1 was seen only from spark-d "
                                 "(spark-c reports nothing on port 0)"]
    strict = cabling.diagnose(sparks, "spark-a", strict=True, whole=True)
    assert not strict["ready"] and strict["layout"] == "incomplete"
    assert cabling.MISSING in strict["summary"]


def test_unknown_neighbor_is_a_seen_only_spark_for_the_survey_and_a_problem_for_setup():
    sparks = synthetic(4, ring([(0, 1)] * 4))
    # spark-d is not signed in to: only its neighbors' LLDP names it.
    survey = cabling.diagnose(sparks[:3] + [dict(sparks[3], reached=False, functions=[], macs=[], lldp=None)], "spark-a")
    assert survey["ready"] and survey["order_names"] == ["spark-a", "spark-b", "spark-c", "spark-d"]
    assert any("spark-d was not reached" in note for note in survey["notes"])
    strict = cabling.diagnose(sparks[:2], "spark-a", strict=True, whole=True)
    assert not strict["ready"]
    assert any("which is not one of the Sparks setup signed in to" in p for p in strict["problems"])


def test_pair_tolerates_an_unknown_device_on_port_one():
    sparks = synthetic(3, [((0, 0), (1, 0)), ((0, 1), (2, 1))])
    result = cabling.diagnose(sparks[:2], "spark-a", strict=True, whole=True)
    assert result["ready"] and result["layout"] == "pair"
    assert any("spark-c" in note for note in result["notes"])


def test_neighbor_caches_identify_cables_without_lldp():
    sparks = synthetic(4, ring([(0, 1)] * 4), lldp=False)
    for spark in sparks:
        spark["neighbors"] = []
    for (a, p), (b, q) in ring([(0, 1)] * 4):
        for (x, xp), (y, yp) in (((a, p), (b, q)), ((b, q), (a, p))):
            for f in (0, 1):
                sparks[x]["neighbors"] += [{"dev": NETDEVS[(xp, f)], "lladdr": mac(y, yp, g), "answered": True}
                                           for g in (0, 1)]
                # A stale entry that did not answer is ignored.
                sparks[x]["neighbors"].append({"dev": NETDEVS[(xp, f)], "lladdr": mac((y + 1) % 4, 0, 0),
                                               "answered": False})
    result = cabling.diagnose(sparks, "spark-a")
    assert result["ready"] and result["order_names"] == ["spark-a", "spark-b", "spark-c", "spark-d"]
    # Setup's check uses LLDP only.
    assert not cabling.diagnose(sparks, "spark-a", strict=True, whole=True)["ready"]


def test_probe_inventory_ports_come_from_rdma_device_names():
    inventory = {"id": "x", "hostname": "spark-x", "neighbors": [],
                 "functions": [{"device": "rocep1s0f1", "netdev": "enp1s0f1np1", "mac": "02:00:00:00:00:01",
                                "carrier": True, "addresses": []},
                               {"device": "mlx5_9", "netdev": "eth9", "mac": "02:00:00:00:00:02"}]}
    record = cabling.from_probe(inventory, macs=["02:00:00:00:ff:01"])
    assert record["functions"] == [{"netdev": "enp1s0f1np1", "port": 1, "mac": "02:00:00:00:00:01", "carrier": True}]
    assert record["macs"] == ["02:00:00:00:00:01", "02:00:00:00:ff:01"]
    assert [cabling.port_of(n) for n in ("enP2p1s0f0np0", "roceP2p1s0f1", "enP7s7")] == [0, 1, None]


def test_setup_ordering_reports_a_miscabled_ring():
    from runtime.host.test_appliance import nodes
    found = nodes()
    ports = topology.endpoints(found[1])
    flip = {"cw_primary": "ccw_primary", "cw_secondary": "ccw_secondary",
            "ccw_primary": "cw_primary", "ccw_secondary": "cw_secondary"}
    # Swap rank 1's two cables: its own observations move to the other port, and
    # its neighbors see its other port.
    by_netdev = {ports[role]["netdev"]: ports[flip[role]]["netdev"] for role in flip}
    found[1]["lldp"]["lldp"]["interface"] = [{by_netdev[netdev]: row} for entry in found[1]["lldp"]["lldp"]["interface"]
                                             for netdev, row in entry.items()]
    by_mac = {ports[role]["mac"]: ports[flip[role]]["mac"] for role in flip}
    for node in (found[0], found[2]):
        for entry in node["lldp"]["lldp"]["interface"]:
            for row in entry.values():
                row["port"]["id"]["value"] = by_mac.get(row["port"]["id"]["value"], row["port"]["id"]["value"])
    with pytest.raises(cabling.CablingError, match="To fix: On spark1, swap its two cables"):
        topology.ordered_nodes(found, found[0]["node_id"])


def test_setup_prints_the_cabling_fix_and_the_cables(tmp_path, monkeypatch, capsys):
    from runtime.host import controller
    from runtime.host.test_appliance import nodes
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    found = nodes(2)
    # Move rank 1's end of the pair cable to its port 1.
    ports = topology.endpoints(found[1])
    swap = {ports["cw_primary"]["mac"]: ports["ccw_primary"]["mac"], ports["cw_secondary"]["mac"]: ports["ccw_secondary"]["mac"]}
    for entry in found[0]["lldp"]["lldp"]["interface"]:
        for row in entry.values():
            row["port"]["id"]["value"] = swap[row["port"]["id"]["value"]]
    netdevs = {ports["cw_primary"]["netdev"]: ports["ccw_primary"]["netdev"],
               ports["cw_secondary"]["netdev"]: ports["ccw_secondary"]["netdev"]}
    found[1]["lldp"]["lldp"]["interface"] = [{netdevs[k]: v} for entry in found[1]["lldp"]["lldp"]["interface"]
                                             for k, v in entry.items()]
    # Rank 1's port 0 has no cable now, so no carrier.
    for interface in found[1]["facts"]["interfaces"]:
        if interface["name"] in netdevs:
            interface["operstate"] = "DOWN"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(found))
    assert controller.main(["setup", "--inventory", str(inventory), "--head-id", found[0]["node_id"], "--plan",
                            "--output", str(tmp_path / "plan")]) == 2
    err = capsys.readouterr().err
    assert ("SparkRing: Fabric cabling: Pair: spark0 port 0 ↔ spark1 port 1, but no cable joins the two ports 0. "
            "Pair models use port 0 on both Sparks. To fix: On spark1, move the cable from port 1 to port 0.") in err
    assert "  Cables:\n    spark0 port 0 ↔ spark1 port 1\n" in err
