"""The relay plan offline: derivation from the route module, isolation, traces and relay queue load."""

from __future__ import annotations

import json

import pytest

from sparkring_sircl import routes
from sparkring_sircl.fabric import layouts, plan
from sparkring_sircl.fabric.layouts import FabricError, Group
from sparkring_sircl.testing.relay_hosts import SimRing

import fabric_reference

RING = SimRing(8)
NAMES = [spark.name for spark in RING.sparks]


def _facts() -> plan.Facts:
    return plan.Facts(plan.SparkFacts(spark.position, spark.name,
                                      tuple(plan.Port(n.name, n.mac, n.address, n.prefixlen)
                                            for n in spark.netdevs.values()))
                      for spark in RING.sparks)


FACTS = _facts()


def _plan(**spec) -> plan.LayoutPlan:
    return plan.build_layout_plan(layouts.resolve(ring_size=8, **spec), FACTS)


def _mac(rank: int, netdev: str) -> str:
    return RING.sparks[rank].netdevs[netdev].mac


def test_ring8_plan_is_the_universal_relay_table():
    (group,) = _plan(layout="ring8").groups
    assert group.relay_egress == "same" and group.group.label == "cycle:0-1-2-3-4-5-6-7"
    for rank in range(8):
        routes_ref, relays_ref = fabric_reference.plan(rank, _mac)
        spark = group.spark(rank)
        assert {(r.destination, r.netdev, r.source, r.next_mac, r.tag) for r in spark.routes} == \
            {(r["dest"], r["dev"], r["src"], r["via_mac"], r["tag"]) for r in routes_ref}
        assert {(f.in_netdev, f.out_netdev, f.next_mac, f.k) for f in spark.filters} == \
            {(r["in"], r["out"], r["next_mac"], r["k"]) for r in relays_ref}
        markers_ref: dict[str, set] = {}
        for r in routes_ref:
            markers_ref.setdefault(r["rdma"], set()).add((r["dest"], r["tag"]))
        assert {m.device: set(m.rules) for m in spark.markers} == markers_ref
        assert (len(spark.routes), len(spark.filters), len(spark.markers)) == (12, 12, 4)


def test_ring8_routes_carry_every_sircl_lane_and_the_other_class_of_opposite_paths():
    (group,) = _plan(layout="ring8").groups
    spark = group.spark(0)
    carried = [route for route in spark.routes if route.lanes]
    extra = [route for route in spark.routes if not route.lanes]
    # Distances 2 and 3 in both directions on both classes are SIRCL lanes; the opposite Spark is reached on
    # lane 0 (primary, clockwise from the smaller end) and lane 1 (secondary, counter-clockwise).
    assert len(carried) == 10 and len(extra) == 2
    assert {(r.netdev, r.peer) for r in extra} == {("enP2p1s0f0np0", 4), ("enp1s0f1np1", 4)}
    lane_routes = {lane: route for route in carried for lane in route.lanes}
    assert lane_routes["rank 0->4 lane 0"].netdev == "enp1s0f0np0"
    assert lane_routes["rank 0->4 lane 1"].netdev == "enP2p1s0f1np1"
    assert group.sircl.route_text(0) == routes.derive_routes(routes.Layout.parse("ring:8")).route_text(0)


def test_two_tp4_groups_relay_only_inside_their_paths():
    built = _plan(layout="2xTP4")
    assert built.layout.relay_egress == "sibling"
    first, second = built.groups
    assert (first.group.label, second.group.label) == ("path:0-1-2-3", "path:4-5-6-7")
    for group, (a, b, c, d) in ((first, (0, 1, 2, 3)), (second, (4, 5, 6, 7))):
        counts = {s.position: (len(s.routes), len(s.filters), len(s.markers)) for s in group.sparks}
        assert counts == {a: (4, 0, 2), b: (2, 6, 2), c: (2, 6, 2), d: (4, 0, 2)}
        inner = group.spark(b)
        assert {(f.in_netdev, f.k) for f in inner.filters} == {
            ("enp1s0f1np1", 1), ("enp1s0f1np1", 2), ("enP2p1s0f1np1", 1), ("enP2p1s0f1np1", 2),
            ("enp1s0f0np0", 1), ("enP2p1s0f0np0", 1)}
        # Sibling egress: in on the primary function of port 1, out of the secondary function of port 0, with
        # the next Spark's primary MAC, so the frame arrives on the primary function.
        cw = inner.filter_for("enp1s0f1np1", 2)
        assert cw.out_netdev == "enP2p1s0f0np0" and cw.next_mac == _mac(c, "enp1s0f1np1")
        assert all(route.peer in (a, b, c, d) for spark in group.sparks for route in spark.routes)
    assert first.load_factor == 1 and first.busiest_queue == 2 and first.per_peer_bytes == 393216


def test_relay_egress_same_keeps_the_function_class():
    built = _plan(layout="2xTP4", relay_egress="same")
    inner = built.groups[0].spark(1)
    for relay_filter in inner.filters:
        assert plan.role_of_netdev(relay_filter.in_netdev).secondary == \
            plan.role_of_netdev(relay_filter.out_netdev).secondary
        assert plan.role_of_netdev(relay_filter.in_netdev).port != plan.role_of_netdev(relay_filter.out_netdev).port


def test_adjacent_pairs_need_no_relay_objects():
    built = _plan(layout="4xTP2")
    assert [g.group.label for g in built.groups] == ["pair:0-1", "pair:2-3", "pair:4-5", "pair:6-7"]
    assert all(not s.routes and not s.filters and not s.markers for g in built.groups for s in g.sparks)
    assert all(g.per_peer_bytes is None for g in built.groups)


def test_wrap_around_group_relays_on_its_inner_members_only():
    (group,) = _plan(groups="7,0,1,2").groups
    assert group.group.label == "path:7-0-1-2" and group.group.cables == (
        "7.port0-0.port1", "0.port0-1.port1", "1.port0-2.port1")
    assert [s.position for s in group.sparks] == [7, 0, 1, 2]
    assert {s.position for s in group.sparks if s.filters} == {0, 1}
    for spark in group.sparks:
        for relay_filter in spark.filters:
            assert relay_filter.next_position in (7, 0, 1, 2)
        for route in spark.routes:
            assert route.peer in (7, 0, 1, 2) and set(route.relays) <= {0, 1}
    assert layouts.resolve(ring_size=8, groups="7-2").groups[0].members == (7, 0, 1, 2)


def test_every_sircl_lane_is_carried_and_delivered_inside_its_group():
    for spec in (dict(layout="ring8"), dict(layout="2xTP4"), dict(layout="4xTP2"), dict(groups="7,0,1,2"),
                 dict(groups="7-2;3-6"), dict(layout="ring8", relay_egress="sibling"),
                 dict(layout="2xTP4", relay_egress="same")):
        built = _plan(**spec)
        for group in built.groups:
            sparks = {s.position: s for s in group.sparks}
            for rank, peers in enumerate(group.sircl.ranks):
                origin = group.layout.positions[rank]
                for peer, lanes in peers.items():
                    for lane in lanes:
                        if lane.hops < 2:
                            continue
                        destination = FACTS.port(group.layout.positions[peer], lane.remote.netdev).address
                        result = plan.trace(sparks, FACTS, 8, origin, destination)
                        assert result.outcome == "delivered", (spec, rank, peer, lane.lane, result)
                        assert result.hops[-1] == (group.layout.positions[peer], lane.remote.netdev)
                        assert {position for position, _ in result.hops} <= set(group.group.members)


def test_isolation_checks_catch_a_filter_toward_a_foreign_cable():
    built = _plan(layout="2xTP4")
    assert plan.isolation_problems(built.groups) == []
    first = built.groups[0]
    end = first.spark(3)
    leaking = plan.RelayFilter("enp1s0f1np1", 1, "enp1s0f0np0", _mac(4, "enp1s0f1np1"), 4, "enp1s0f1np1", "cw",
                               "primary")
    broken = plan.GroupPlan(first.group, first.layout, first.sircl, first.relay_egress, first.max_relays,
                            tuple(s if s.position != 3 else plan.SparkPlan(3, end.name, end.group, end.routes,
                                                                           (leaking,)) for s in first.sparks),
                            first.queues, first.busiest_queue, first.load_factor, first.digest)
    problems = plan.isolation_problems([broken, built.groups[1]])
    assert any("does not own" in p for p in problems) and any("outside the group" in p for p in problems)


def test_layouts_refuse_shared_sparks_long_paths_and_non_consecutive_groups():
    with pytest.raises(FabricError, match="belongs to groups"):
        layouts.resolve(ring_size=8, groups="0-3;3-5")
    with pytest.raises(FabricError, match="consecutive"):
        layouts.resolve(ring_size=8, groups="0,2,4")
    with pytest.raises(FabricError, match="qualified limit is 3"):
        plan.build_group_plan(Group((0, 1, 2, 3, 4, 5), 8), FACTS, relay_egress="sibling")
    wide = plan.build_group_plan(Group((0, 1, 2, 3, 4, 5), 8), FACTS, relay_egress="sibling", max_relays=4)
    assert max(route.hops for spark in wide.sparks for route in spark.routes) == 5
    with pytest.raises(FabricError, match="ring of 8"):
        layouts.resolve(ring_size=6, layout="2xTP4")
    with pytest.raises(FabricError, match="unknown layout"):
        layouts.resolve(ring_size=8, layout="3xTP3")


def test_group_selection_and_parsing():
    layout = layouts.resolve(ring_size=8, layout="2xTP4")
    assert layout.select("1")[0].members == (4, 5, 6, 7)
    assert layout.select("path:0-1-2-3")[0].members == (0, 1, 2, 3)
    assert layout.select("4-7")[0].members == (4, 5, 6, 7)
    with pytest.raises(FabricError, match="has no group"):
        layout.select("2-5")
    assert layouts.parse_groups("6-7,0-1;2-5", 8) == ((6, 7, 0, 1), (2, 3, 4, 5))
    with pytest.raises(FabricError, match="outside the ring"):
        layouts.parse_groups("0-9", 8)


def test_the_plan_digest_names_the_group_not_the_layout():
    named = _plan(layout="2xTP4").groups[1]
    custom = _plan(groups="4-7", relay_egress="sibling").groups[0]
    assert named.digest == custom.digest
    assert _plan(groups="4-7", relay_egress="same").groups[0].digest != named.digest


def test_relay_queue_report_follows_the_route_module():
    (ring8,) = _plan(layout="ring8").groups
    busiest, load = routes.relay_load(ring8.sircl)
    assert (ring8.busiest_queue, ring8.load_factor, ring8.per_peer_bytes) == (busiest, load, 131072) == (6, 3.0, 131072)
    queue = next(q for q in ring8.queues if q.position == 1 and q.in_netdev == "enp1s0f1np1")
    assert queue.out_netdev == "enp1s0f0np0" and queue.lanes
    sibling = _plan(layout="2xTP4").groups[0]
    assert {(q.position, q.in_netdev, q.out_netdev) for q in sibling.queues} >= {(1, "enp1s0f1np1", "enP2p1s0f0np0")}


def test_symbolic_and_file_facts_and_json():
    symbolic = plan.Facts.symbolic_for(NAMES)
    built = plan.build_layout_plan(layouts.resolve(ring_size=8, layout="2xTP4"), symbolic)
    text = plan.render_text(built, symbolic)
    assert "placeholders" in text and "<spark1:enp1s0f1np1:mac>" in text
    assert "SIRCL_PEER_ROUTES=1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f0/roceP2p1s0f0" in text
    round_trip = plan.Facts.from_json(json.loads(json.dumps(FACTS.to_json())))
    assert round_trip.port(3, "enP2p1s0f1np1") == FACTS.port(3, "enP2p1s0f1np1")
    document = _plan(layout="ring8").to_json()
    assert document["schema"] == plan.PLAN_SCHEMA and len(document["constraints"]) == 4
    assert document["groups"][0]["per_peer_bytes"] == 131072
    with pytest.raises(FabricError, match="not a MAC"):
        plan.port_from_text("enp1s0f0np0", "zz", "198.18.0.1/24", owner="x")


def test_tags_and_preferences():
    assert [plan.tag(k) for k in range(4)] == [0x0800, 0x88B5, 0x88B6, 0x88B7]
    assert plan.relays_left(0x88B7) == 3 and plan.relays_left(0x0800) == 0 and plan.relays_left(0x86DD) is None
    relay_filter = _plan(layout="ring8").groups[0].spark(0).filters[0]
    assert (relay_filter.pref, relay_filter.handle) == (10 + relay_filter.k, relay_filter.k)
    assert list(plan.RESERVED_PREFS) == list(range(11, 18))
