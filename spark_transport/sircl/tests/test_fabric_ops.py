"""The relay plan installer against simulated Sparks: show, diff, up, down, adoption and group isolation."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from sparkring_sircl import routes
from sparkring_sircl.fabric import cli, commands, diff, layouts, ops, state
from sparkring_sircl.fabric import plan as plan_mod
from sparkring_sircl.testing.relay_hosts import SimRing, SimulatorError

import fabric_reference

MARKER = commands.MarkerConfig()


@pytest.fixture
def ring() -> SimRing:
    return SimRing(8)


@pytest.fixture
def site_file(tmp_path, ring):
    site = ring.site()
    document = {"schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "enP7s7",
                "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
                "ring": [{"name": h.name, "ssh": h.ssh, "lan_address": h.lan_address} for h in site.ring]}
    path = tmp_path / "site.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _cli(ring: SimRing, site_file: str, *argv: str) -> int:
    command, rest = argv[0], list(argv[1:])
    return cli.main([command, "--site", site_file, *rest], executor=ring.executor)


def _lane_destinations(ring: SimRing, members) -> list[tuple[int, str, int]]:
    """(origin position, destination address, hops) of every SIRCL lane of a group on the simulated ring."""
    group = layouts.Group(tuple(members), ring.size)
    layout = group.layout()
    derived = routes.derive_routes(layout, 2)
    found = []
    for rank, peers in enumerate(derived.ranks):
        for peer, lanes in peers.items():
            for lane in lanes:
                peer_position = layout.positions[peer]
                found.append((layout.positions[rank], ring.sparks[peer_position].netdevs[lane.remote.netdev].address,
                              lane.hops))
    return found


def _assert_lanes_deliver(ring: SimRing, members) -> None:
    for origin, destination, hops in _lane_destinations(ring, members):
        outcome, path = ring.deliver(origin, destination)
        assert outcome == "delivered", (origin, destination, outcome, path)
        assert len(path) == hops and {position for position, _ in path} <= set(members), (origin, destination, path)


# -- fresh ring ---------------------------------------------------------------------------------------


def test_up_ring8_installs_the_plan_and_a_second_up_changes_nothing(ring, site_file, capsys):
    assert _cli(ring, site_file, "up", "--layout", "ring8") == 0           # dry run
    assert all(not m for m in ring.mutations().values())
    assert _cli(ring, site_file, "up", "--layout", "ring8", "--apply") == 0
    out = capsys.readouterr().out
    assert "verified: 8 Spark(s)" in out
    _assert_lanes_deliver(ring, range(8))
    for spark in ring.sparks:
        assert sum(len(f) for f in spark.filters.values()) == 12 and len(spark.marker_processes()) == 4
        assert all(route["proto"] == commands.ROUTE_PROTOCOL for route in spark.routes if route["plen"] == 32)
        record = json.loads(spark.files[commands.RECORD_PATH])
        assert record["group"] == "cycle:0-1-2-3-4-5-6-7" and record["layout"] == "ring8"
        assert sorted(record["qdiscs"]) == sorted(spark.netdevs)
    ring.clear_mutations()
    assert _cli(ring, site_file, "up", "--layout", "ring8", "--apply") == 0
    assert "nothing to apply" in capsys.readouterr().out
    assert all(not m for m in ring.mutations().values())
    assert _cli(ring, site_file, "diff", "--layout", "ring8") == 0
    assert "ring8 up: no host changes on 8 Spark(s)" in capsys.readouterr().out


def test_down_removes_exactly_the_installed_objects(ring, site_file, capsys):
    before = [ring.snapshot(p) for p in range(8)]
    ring.sparks[2].neighbours.append({"dst": "198.18.4.1", "dev": "enp1s0f0np0", "lladdr": "02:5a:03:01:00:01",
                                      "state": "PERMANENT", "proto": None})     # an on-link pin: not a relay object
    ring.sparks[2].routes.append({"dst": "10.9.9.9", "plen": 32, "dev": "enP7s7", "src": None, "scope": "link",
                                  "proto": 4, "metric": 0, "gateway": None})     # a route of other tooling
    before[2] = ring.snapshot(2)
    assert _cli(ring, site_file, "up", "--layout", "ring8", "--apply") == 0
    assert _cli(ring, site_file, "down", "--layout", "ring8", "--apply") == 0
    assert "verified" in capsys.readouterr().out
    assert [ring.snapshot(p) for p in range(8)] == before


# -- a table installed by other tooling ------------------------------------------------------------------


def test_diff_of_ring8_against_the_reference_table_shows_no_host_changes(ring, site_file, capsys):
    fabric_reference.install(ring)
    assert _cli(ring, site_file, "diff", "--layout", "ring8") == 0
    out = capsys.readouterr().out
    assert ("ring8 up: no host changes on 8 Spark(s); 192 unowned object(s) equal to the plan to mark in place "
            "(--adopt); 8 record change(s); --apply needs --adopt on 8 Spark(s)") in out
    assert out.count("24 mark, record write; 16 unchanged") == 8       # 12 routes + 12 neighbours to mark
    assert "~ replace" not in out and "+ add" not in out and "BLOCKER" not in out


def test_show_labels_the_reference_table_unowned_and_resolves_addresses(ring, site_file, capsys):
    fabric_reference.install(ring)
    assert _cli(ring, site_file, "show") == 0
    out = capsys.readouterr().out
    assert out.count("no group record") == 8
    assert "routes: 12 relay routes (12 unowned)" in out
    assert "198.18.4.2/32 dev enp1s0f0np0 src 198.18.0.1 proto boot -> spark3 enp1s0f1np1; neighbour " \
           "02:5a:01:01:00:01 (spark1 enp1s0f1np1); tag 0x88b6 [unowned]" in out
    assert "relay filters: 12" in out and "markers: 4" in out


def test_up_adopts_the_reference_table_in_place_only_with_adopt(ring, site_file, capsys):
    fabric_reference.install(ring)
    ring.clear_mutations()
    assert _cli(ring, site_file, "up", "--layout", "ring8", "--apply") == 3
    assert "--adopt takes them over" in capsys.readouterr().out
    assert all(not m for m in ring.mutations().values())
    processes = [dict(spark.processes) for spark in ring.sparks]
    assert _cli(ring, site_file, "up", "--layout", "ring8", "--adopt", "--apply") == 0
    for spark, before in zip(ring.sparks, processes):
        kinds = {line.split()[2] + " " + line.split()[3] for line in spark.mutations if line.startswith("sudo -n")}
        assert kinds <= {"ip route", "ip neigh"}, kinds
        assert all(" replace " in line for line in spark.mutations if line.startswith("sudo -n ip"))
        assert spark.processes == before
    _assert_lanes_deliver(ring, range(8))
    assert _cli(ring, site_file, "down", "--layout", "ring8", "--apply") == 0
    for spark in ring.sparks:
        assert not [r for r in spark.routes if r["plen"] == 32] and not spark.neighbours
        assert not spark.processes and not any(spark.filters.values())
        assert all(spark.qdiscs.values())        # the reference table's ingress qdiscs are not the installer's


def test_down_of_the_reference_table_needs_adopt(ring, site_file, capsys):
    fabric_reference.install(ring)
    ring.clear_mutations()
    assert _cli(ring, site_file, "down", "--layout", "ring8", "--apply") == 3
    assert all(not m for m in ring.mutations().values())
    assert _cli(ring, site_file, "down", "--layout", "ring8", "--adopt", "--apply") == 0
    assert all(not spark.processes and not [r for r in spark.routes if r["plen"] == 32] for spark in ring.sparks)


def test_two_tp4_dry_run_from_the_reference_table_lists_the_transition(ring, site_file, capsys):
    fabric_reference.install(ring)
    ring.clear_mutations()
    assert _cli(ring, site_file, "up", "--layout", "2xTP4") == 3
    assert "--adopt takes them over" in capsys.readouterr().out
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--adopt") == 0
    out = capsys.readouterr().out
    assert "dry run: nothing was changed" in out and "BLOCKER" not in out
    assert all(not m for m in ring.mutations().values())
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--adopt", "--apply") == 0
    _assert_lanes_deliver(ring, (0, 1, 2, 3))
    _assert_lanes_deliver(ring, (4, 5, 6, 7))
    for spark in ring.sparks:
        assert not [r for r in spark.routes if r["plen"] == 32 and r["proto"] != commands.ROUTE_PROTOCOL]


# -- groups -----------------------------------------------------------------------------------------------


def _installed_isolation_problems(ring: SimRing, layout: layouts.FabricLayout) -> list[str]:
    """Section R3 checked on what the Sparks hold, independently of the plan module."""
    problems = []
    owner = {member: group for group in layout.groups for member in group.members}
    address_owner = {netdev.address: spark.position for spark in ring.sparks for netdev in spark.netdevs.values()}
    for spark in ring.sparks:
        group = owner.get(spark.position)
        relay_routes = [r for r in spark.routes if r["plen"] == 32]
        filters = [f for entries in spark.filters.values() for f in entries]
        if group is None:
            if relay_routes or filters or spark.processes or spark.neighbours:
                problems.append(f"{spark.name} is in no group but holds relay objects")
            continue
        fabric = group.layout().fabric
        for route in relay_routes:
            if address_owner.get(route["dst"]) not in group.members:
                problems.append(f"{spark.name} routes {route['dst']} outside {group.label}")
        for process in spark.marker_processes():
            if any(address_owner.get(address) not in group.members for address in process["rules"]):
                problems.append(f"{spark.name} tags a destination outside {group.label}")
        for relay_filter in filters:
            for netdev in (relay_filter.netdev, relay_filter.out_dev):
                if fabric.cable_at(spark.position, spark.netdevs[netdev].port) is None:
                    problems.append(f"{spark.name} relays through {netdev}, whose cable {group.label} does not own")
    return problems


@pytest.mark.parametrize("spec", [("--layout", "ring8"), ("--layout", "2xTP4"), ("--layout", "4xTP2"),
                                  ("--groups", "7,0,1,2")])
def test_each_layout_installs_diffs_clean_delivers_and_stays_isolated(ring, site_file, spec):
    assert _cli(ring, site_file, "up", *spec, "--apply") == 0
    assert _cli(ring, site_file, "diff", *spec) == 0
    layout = layouts.resolve(ring_size=8, **{spec[0][2:]: spec[1]})
    assert _installed_isolation_problems(ring, layout) == []
    for group in layout.groups:
        _assert_lanes_deliver(ring, group.members)
        for spark in ring.sparks:
            if spark.position in group.members:
                continue
            for member in group.members:
                for netdev in spark.netdevs.values():
                    outcome, path = ring.deliver(member, netdev.address)
                    assert outcome != "delivered" or len(path) == 1, (group.label, member, netdev.address, path)
        for member in group.members:
            for port in (0, 1):
                cable = group.layout().fabric.cable_at(member, port)
                if cable is not None:
                    continue
                for netdev in ring.sparks[member].netdevs.values():
                    if netdev.port == port:
                        for k in range(1, 8):
                            assert ring.inject(member, netdev.name, plan_mod.tag(k))[0] == "dropped"
    assert _cli(ring, site_file, "down", *spec, "--apply") == 0
    assert _installed_isolation_problems(ring, layout) == []
    assert all(not [r for r in s.routes if r["plen"] == 32] and not s.processes for s in ring.sparks)


def test_two_tp4_groups_are_isolated_on_the_installed_state(ring, site_file):
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--apply") == 0
    _assert_lanes_deliver(ring, (0, 1, 2, 3))
    _assert_lanes_deliver(ring, (4, 5, 6, 7))
    for origin in range(8):
        for target in range(8):
            if (origin < 4) != (target < 4):
                for netdev in ring.sparks[target].netdevs.values():
                    outcome, path = ring.deliver(origin, netdev.address)
                    assert outcome != "delivered" or path == [(target, netdev.name)], (origin, target, path)
    for end in (0, 3, 4, 7):
        assert not any(ring.sparks[end].filters.values())
    for position, netdev in ((3, "enp1s0f0np0"), (3, "enP2p1s0f0np0"), (4, "enp1s0f1np1"), (4, "enP2p1s0f1np1"),
                             (0, "enp1s0f1np1"), (7, "enp1s0f0np0")):
        for k in range(1, 8):
            outcome, path = ring.inject(position, netdev, plan_mod.tag(k))
            assert outcome == "dropped" and path == [(position, netdev)], (position, netdev, k)


def test_applying_one_group_never_touches_the_other(ring, site_file, capsys):
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "0", "--apply") == 0
    first_group = [ring.snapshot(p) for p in range(4)]
    ring.clear_mutations()
    calls_before = len(ring.calls)
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "path:4-5-6-7", "--apply") == 0
    assert {target for target, _ in ring.calls[calls_before:]} == {f"op@192.0.2.{10 + p}" for p in range(4, 8)}
    assert [ring.snapshot(p) for p in range(4)] == first_group
    assert all(not ring.mutations()[p] for p in range(4))
    second_group = [ring.snapshot(p) for p in range(4, 8)]
    ring.clear_mutations()
    assert _cli(ring, site_file, "down", "--layout", "2xTP4", "--group", "0", "--apply") == 0
    assert all(not ring.sparks[p].processes and not any(ring.sparks[p].filters.values()) for p in range(4))
    assert [ring.snapshot(p) for p in range(4, 8)] == second_group
    assert all(not ring.mutations()[p] for p in range(4, 8))
    _assert_lanes_deliver(ring, (4, 5, 6, 7))
    assert _cli(ring, site_file, "diff", "--layout", "2xTP4", "--group", "1") == 0
    assert _cli(ring, site_file, "diff", "--layout", "2xTP4", "--group", "0") == 1


def test_a_spark_of_another_group_blocks_up_and_down(ring, site_file, capsys):
    assert _cli(ring, site_file, "up", "--layout", "ring8", "--apply") == 0
    ring.clear_mutations()
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--apply") == 3
    out = capsys.readouterr().out
    assert out.count("belongs to group cycle:0-1-2-3-4-5-6-7 (layout ring8)") == 8
    assert _cli(ring, site_file, "down", "--groups", "0-3", "--apply") == 3
    assert _cli(ring, site_file, "up", "--groups", "2-5", "--apply") == 3
    assert all(not m for m in ring.mutations().values())


def test_wrap_around_group_installs_on_its_members_only(ring, site_file):
    others = [ring.snapshot(p) for p in (3, 4, 5, 6)]
    assert _cli(ring, site_file, "up", "--groups", "7,0,1,2", "--name", "wrap4", "--apply") == 0
    assert [ring.snapshot(p) for p in (3, 4, 5, 6)] == others
    _assert_lanes_deliver(ring, (7, 0, 1, 2))
    assert {p for p in range(8) if any(ring.sparks[p].filters.values())} == {0, 1}
    record = json.loads(ring.sparks[7].files[commands.RECORD_PATH])
    assert (record["group"], record["layout"], record["relay_egress"]) == ("path:7-0-1-2", "wrap4", "sibling")
    for position, netdev in ((2, "enp1s0f0np0"), (7, "enp1s0f1np1")):
        assert ring.inject(position, netdev, plan_mod.tag(1))[0] == "dropped"
    assert _cli(ring, site_file, "diff", "--groups", "7-2") == 0


def test_adjacent_pairs_write_only_records(ring, site_file):
    assert _cli(ring, site_file, "up", "--layout", "4xTP2", "--apply") == 0
    for spark in ring.sparks:
        assert [line for line in spark.mutations if "relay-state.json" not in line] == []
        assert json.loads(spark.files[commands.RECORD_PATH])["group"].startswith("pair:")
    for members in ((0, 1), (2, 3), (4, 5), (6, 7)):
        _assert_lanes_deliver(ring, members)


# -- safety ------------------------------------------------------------------------------------------------


def test_dry_runs_and_reads_never_mutate(ring, site_file):
    fabric_reference.install(ring)
    ring.clear_mutations()
    for argv in (("show",), ("diff", "--layout", "2xTP4"), ("up", "--layout", "2xTP4", "--adopt"),
                 ("down", "--layout", "ring8", "--adopt"), ("up", "--groups", "7-2", "--adopt"), ("facts",),
                 ("marker",)):
        _cli(ring, site_file, *argv)
    assert all(not m for m in ring.mutations().values())
    for _, script in ring.calls:
        assert "sudo" not in script


def test_hostname_mismatch_unreachable_spark_and_missing_marker_block(ring, site_file, capsys):
    ring.sparks[5].hostname = "someone-else"
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--apply") == 3
    assert "is named someone-else" in capsys.readouterr().out
    assert all(not m for m in ring.mutations().values())
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "0", "--apply") == 0
    ring.sparks[5].hostname = ring.sparks[5].name
    ring.sparks[6].reachable = False
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "1", "--apply") == 3
    assert "group path:4-5-6-7 cannot be planned" in capsys.readouterr().out
    ring.sparks[6].reachable = True
    del ring.sparks[4].files[MARKER.path]
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "1", "--apply") == 3
    assert "marker executable /var/tmp/ring8-mesh-marker is missing" in capsys.readouterr().out
    assert _cli(ring, site_file, "marker", "--sparks", "4-7") == 0
    out = capsys.readouterr().out
    assert "spark4: /var/tmp/ring8-mesh-marker missing" in out
    assert "dry run: --apply compiles mesh_marker.c and installs /var/tmp/ring8-mesh-marker on spark4" in out
    assert MARKER.path not in ring.sparks[4].files
    assert _cli(ring, site_file, "marker", "--sparks", "4-7", "--apply") == 0
    assert ring.sparks[4].files[MARKER.path].startswith("\x7fELF built ")
    assert not any(line.startswith("install ") for p in (5, 6, 7) for line in ring.sparks[p].mutations)
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "1", "--apply") == 0


def test_cabling_that_disagrees_with_the_site_order_blocks(ring, site_file, tmp_path):
    document = json.loads(open(site_file, encoding="utf-8").read())
    document["ring"][1], document["ring"][2] = document["ring"][2], document["ring"][1]
    swapped = tmp_path / "swapped.json"
    swapped.write_text(json.dumps(document), encoding="utf-8")
    assert cli.main(["up", "--site", str(swapped), "--layout", "2xTP4", "--group", "0", "--apply",
                     "--skip-hostname-check"], executor=ring.executor) == 3
    assert all(not m for m in ring.mutations().values())


def test_a_foreign_route_to_a_planned_destination_is_never_touched(ring, site_file, capsys):
    ring.sparks[0].routes.append({"dst": "198.18.4.2", "plen": 32, "dev": "enP7s7", "src": None, "scope": "global",
                                  "proto": 4, "metric": 0, "gateway": "192.0.2.254"})
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "0", "--apply") == 3
    assert "is held by a route that is not a relay route" in capsys.readouterr().out
    assert not ring.sparks[0].mutations


def test_stale_owned_objects_are_removed_and_changed_ones_replaced(ring, site_file):
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--apply") == 0
    spark = ring.sparks[1]
    spark.routes.append({"dst": "198.18.9.9", "plen": 32, "dev": "enp1s0f0np0", "src": "198.18.2.1", "scope": "link",
                         "proto": commands.ROUTE_PROTOCOL, "metric": 0, "gateway": None})
    spark.filters["enp1s0f1np1"][0].dst_mac = "02:5a:0f:0f:0f:0f"
    spark.filters["enp1s0f0np0"].append(spark.filters["enp1s0f0np0"][0].__class__(
        "enp1s0f0np0", 13, 3, 0x88B7, "02:5a:0f:0f:0f:0f", 0x88B6, "enp1s0f1np1"))
    victim = next(iter(spark.processes))
    spark.processes[victim]["argv"] = spark.processes[victim]["argv"][:-1]       # a marker without --managed
    ring.clear_mutations()
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--group", "0", "--apply") == 0
    assert any("ip route del 198.18.9.9/32 dev enp1s0f0np0 proto 82" in line for line in spark.mutations)
    assert sum("tc filter add" in line for line in spark.mutations) == 1          # the changed filter
    assert sum("tc filter del" in line for line in spark.mutations) == 2          # it and the stale pref 13
    assert sum(f"sudo -n kill {victim}" in line for line in spark.mutations) == 1
    assert sum("setsid" in line for line in spark.mutations) == 1
    assert victim not in spark.processes and len(spark.marker_processes()) == 2
    assert all(not ring.sparks[p].mutations for p in (0, 2, 3))
    _assert_lanes_deliver(ring, (0, 1, 2, 3))


def test_a_failing_command_stops_the_spark_and_up_converges_afterwards(ring, site_file, capsys):
    ring.sparks[2].fail_on.add("tc filter add dev enp1s0f1np1")
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--apply") == 1
    out = capsys.readouterr().out
    assert "spark2: exit 2" in out and "verification failed" in out
    ring.sparks[2].fail_on.clear()
    assert _cli(ring, site_file, "up", "--layout", "2xTP4", "--apply") == 0
    _assert_lanes_deliver(ring, (0, 1, 2, 3))


def test_both_iproute2_spellings_parse_alike(site_file):
    for style in ("iproute2-6", "iproute2-5"):
        ring = SimRing(8, style=style)
        fabric_reference.install(ring)
        site = ring.site()
        states = ops.gather(site, range(8), ring.executor, MARKER)
        filters = [entry for netdev in plan_mod.FABRIC_NETDEVS for entry in states[1].reserved_filters(netdev)]
        assert len(filters) == 12 and all(entry.problem is None and entry.handle == entry.pref - 10 for entry in filters)
        layout = layouts.resolve(ring_size=8, layout="ring8")
        outcome = ops.evaluate(site, layout, layout.groups, states, mode="up", marker=MARKER, adopt=True)
        assert outcome.host_changes == 0 and sum(d.count("mark") for d in outcome.diffs) == 192


def test_filter_decoder_accepts_both_mask_conventions_and_rejects_other_actions():
    entry = {"protocol": "0x88b6", "pref": 12, "kind": "flower", "chain": 0, "options": {
        "handle": 2, "skip_sw": True, "in_hw": True, "actions": [
            # Masks that name the bytes written, the other convention from the one iproute2 prints.
            {"kind": "pedit", "keys": [
                {"htype": "eth", "offset": 0, "cmd": "set", "val": "0x025a0100", "mask": "0xffffffff"},
                {"htype": "eth", "offset": 4, "cmd": "set", "val": "0x00010000", "mask": "0xffff0000"}]},
            {"kind": "pedit", "keys": [
                {"htype": "eth", "offset": 12, "cmd": "set", "val": "0x88b50000", "mask": "0xffff0000"}]},
            {"kind": "mirred", "mirred_action": "redirect", "direction": "egress", "to_dev": "enp1s0f0np0"}]}}
    decoded = state.decode_filter("enp1s0f1np1", entry)
    assert decoded.problem is None and decoded.dst_mac == "02:5a:01:00:00:01" and decoded.new_type == 0x88B5
    assert decoded.protocol == 0x88B6 and decoded.handle == 2
    entry["options"]["actions"].insert(0, {"kind": "gact"})
    assert "other actions" in state.decode_filter("enp1s0f1np1", entry).problem
    assert state.decode_filter("x", {"protocol": "[34997]", "pref": 11, "kind": "flower", "chain": 0}) is None
    assert state.ethertype("[34997]") == 0x88B5 and state.ethertype("ip") == 0x0800
    assert state.protocol_number("boot") == 3 and state.protocol_number("82") == 82


def test_unparseable_output_and_unknown_script_lines(ring):
    parsed = state.parse(0, "spark0", "op@x", returncode=255, stdout="", stderr="ssh: connect to host x: refused")
    assert not parsed.reachable and "refused" in parsed.error
    truncated = state.parse(0, "spark0", "op@x", returncode=0, stdout="@@hostname\nspark0\n@@links\n[", stderr="")
    assert truncated.reachable and truncated.fact_problems()
    with pytest.raises(SimulatorError):
        ring.sparks[0].run("sudo -n ip link set enp1s0f0np0 down\n")


def test_generated_scripts_are_valid_bash(ring, site_file):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash on this host")
    site = ring.site()
    layout = layouts.resolve(ring_size=8, layout="ring8")
    states = ops.gather(site, range(8), ring.executor, MARKER)
    up = ops.evaluate(site, layout, layout.groups, states, mode="up", marker=MARKER, adopt=False)
    fabric_reference.install(ring)
    states = ops.gather(site, range(8), ring.executor, MARKER)
    down = ops.evaluate(site, layout, layout.groups, states, mode="down", marker=MARKER, adopt=True)
    scripts = [commands.read_script(MARKER), up.diffs[0].script, down.diffs[0].script,
               commands.marker_build_script("spark0", MARKER, cli.MARKER_SOURCE.read_text(encoding="utf-8"))]
    for script in scripts:
        # Bytes, not text: a text-mode pipe on Windows would hand bash CRLF line endings.
        checked = subprocess.run([bash, "-n"], input=script.encode("utf-8"), capture_output=True)
        assert checked.returncode == 0, (script[:300], checked.stderr.decode("utf-8", "replace"))


def test_commands_refuse_placeholders_and_unsafe_values():
    with pytest.raises(layouts.FabricError):
        commands.route_replace("<spark2:enp1s0f1np1>", "enp1s0f0np0", "198.18.0.1")
    with pytest.raises(layouts.FabricError):
        commands.neigh_replace("198.18.4.2", "02:5a:01:01:00:01; reboot", "enp1s0f0np0")
    with pytest.raises(layouts.FabricError):
        commands.qdisc_add("eth0")
    with pytest.raises(layouts.FabricError):
        commands.MarkerConfig("/var/tmp/marker; rm -rf /")
    with pytest.raises(layouts.FabricError):
        commands.guard("spark0$(reboot)")
    assert commands.MarkerConfig().comm == "ring8-mesh-mark"
    assert commands.MarkerConfig().log("rocep1s0f0") == "/tmp/ring8-mesh-marker-rocep1s0f0.log"


def test_plan_and_facts_commands(ring, site_file, tmp_path, capsys):
    assert cli.main(["plan", "--layout", "2xTP4", "--brief"]) == 0
    out = capsys.readouterr().out
    assert "group path:0-1-2-3" in out and "placeholders" in out
    facts_path = tmp_path / "facts.json"
    assert _cli(ring, site_file, "facts", "--output", str(facts_path)) == 0
    assert cli.main(["plan", "--layout", "ring8", "--facts", str(facts_path), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["groups"][0]["sparks"][0]["routes"][0]["destination"] == "198.18.2.2"
    assert cli.main(["plan", "--layout", "nonsense"]) == 2


def test_diff_reports_changes_with_exit_code_one(ring, site_file, capsys):
    assert _cli(ring, site_file, "diff", "--layout", "2xTP4") == 1
    out = capsys.readouterr().out
    assert "2xTP4 up: 104 host change(s) on 8 Spark(s): 104 to add" in out
    # Only the direct lanes are checked before the origin routes exist: 3 neighbour pairs per group, two lanes.
    assert "route check: 24 of 24 lane destination(s) resolve" in out


def test_route_check_resolves_every_lane_like_the_harness(ring, site_file, capsys):
    fabric_reference.install(ring)
    assert _cli(ring, site_file, "diff", "--layout", "ring8") == 0
    assert "route check: 128 of 128 lane destination(s) resolve over their lane's device" in capsys.readouterr().out
    ring.sparks[3].policy_routes["198.18.10.2"] = "enP7s7"
    assert _cli(ring, site_file, "diff", "--layout", "ring8") == 3
    out = capsys.readouterr().out
    assert "ROUTE CHECK spark3: ip route get 198.18.10.2 uses enP7s7, the lane needs enp1s0f0np0" in out
    assert "route check: 127 of 128" in out
    assert _cli(ring, site_file, "up", "--layout", "ring8", "--adopt", "--apply") == 1
    assert "verification failed" in capsys.readouterr().out


def test_spark_diff_counts_and_script_order(ring):
    site = ring.site()
    layout = layouts.resolve(ring_size=8, layout="2xTP4")
    states = ops.gather(site, range(4), ring.executor, MARKER)
    outcome = ops.evaluate(site, layout, layout.groups[:1], states, mode="up", marker=MARKER, adopt=False)
    inner = next(d for d in outcome.diffs if d.position == 1)
    assert isinstance(inner, diff.SparkDiff) and inner.count("add") == 16
    lines = inner.commands
    assert lines[0] == commands.HEADER and lines[1].startswith('test "$(hostname)" = spark1')
    kinds = [line.split()[2:4] for line in lines if line.startswith("sudo -n") and "relay-state" not in line]
    order = [k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k]
    assert order[:4] == [["tc", "qdisc"], ["tc", "filter"], ["ip", "neigh"], ["ip", "route"]]
    assert lines[-1] == commands.SETTLE
