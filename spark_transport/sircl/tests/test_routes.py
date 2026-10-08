"""Route maps against the reference route vectors (``tests/data/routes.json``) and invalid maps."""

from __future__ import annotations

import pytest

from sparkring_sircl import routes as r

from vectors import load


def _layout(entry) -> r.Layout:
    return r.Layout(r.Fabric.parse(entry["cables"]), tuple(entry["rank_positions"]))


def test_every_layout_vector_is_derived_exactly():
    vectors = load("routes.json")
    assert len(vectors) >= 20
    for key, entry in vectors.items():
        layout = _layout(entry)
        derived = r.derive_routes(layout, entry["lanes"])
        for rank_entry in entry["ranks"]:
            rank = rank_entry["rank"]
            expected_map = {int(peer): tuple(devices) for peer, devices in rank_entry["route_map"].items()}
            assert derived.route_map(rank) == expected_map, (key, rank)
            assert derived.route_text(rank) == rank_entry["route_text"], (key, rank)
            for peer, lanes in rank_entry["lanes"].items():
                got = [lane.to_json() for lane in derived.lanes_to(rank, int(peer))]
                assert got == lanes, (key, rank, peer)


def test_vectors_parse_validate_and_pair():
    for key, entry in load("routes.json").items():
        layout = _layout(entry)
        maps = [r.parse_peer_routes(rank["route_text"]) for rank in entry["ranks"]]
        limit = 16 if key == "ring-of-8-tp6-0-5" else r.DEFAULT_MAX_RELAYS
        for rank, routes in enumerate(maps):
            assert r.validate_route_map(rank, layout.world, routes, layout=layout, max_relays=limit) == entry["lanes"]
        assert r.check_complementary(layout, maps) == [], key


def test_the_six_rank_group_exceeds_the_relay_limit():
    entry = load("routes.json")["ring-of-8-tp6-0-5"]
    layout = _layout(entry)
    routes = r.parse_peer_routes(entry["ranks"][0]["route_text"])
    with pytest.raises(r.RouteError, match="qualified limit is 3"):
        r.validate_route_map(0, 6, routes, layout=layout)


def test_independent_groups_share_nothing():
    vectors = load("routes.json")
    for pair in (("ring-of-8-tp4-0-3", "ring-of-8-tp4-4-7"), ("ring-of-4-half-0-1", "ring-of-4-half-2-3"),
                 ("ring-of-8-tp6-0-5", "ring-of-8-tp2-6-7")):
        groups = [r.derive_routes(_layout(vectors[name]), 2) for name in pair]
        assert r.isolation_problems(groups) == [], pair


def test_relay_load_factor_matches_the_measured_layouts():
    vectors = load("routes.json")
    assert r.relay_load(r.derive_routes(_layout(vectors["ring-of-8"]), 2))[1] == 3
    assert r.relay_load(r.derive_routes(_layout(vectors["ring-of-8-sub-0-3"]), 2))[1] == 1
    assert r.relay_load(r.derive_routes(_layout(vectors["ring-of-8-tp2-6-7"]), 2))[1] == 0


RING4 = "1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f1,3=rocep1s0f1/roceP2p1s0f1"


@pytest.mark.parametrize("text, message", [
    ("1=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f1/roceP2p1s0f1", "has no entry for rank 2"),
    ("0=rocep1s0f0/roceP2p1s0f0," + RING4, "names its own rank"),
    ("1=rocep1s0f0,1=roceP2p1s0f0,2=rocep1s0f0,3=rocep1s0f1", "names rank 1 twice"),
    ("1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0,3=rocep1s0f1/roceP2p1s0f1", "different lane counts"),
    ("1=rocep1s0f0/roceP2p1s0f0/rocep1s0f1,2=rocep1s0f0/roceP2p1s0f1,3=rocep1s0f1/roceP2p1s0f1",
     "different lane counts"),
    ("1=rocep1s0f0/rocep1s0f0,2=rocep1s0f0/roceP2p1s0f1,3=rocep1s0f1/roceP2p1s0f1", "names device rocep1s0f0 twice"),
    ("1=mlx5_9/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f1,3=rocep1s0f1/roceP2p1s0f1", "5 devices; 1 to 4 are supported"),
], ids=["missing-peer", "own-rank", "duplicate-peer", "lane-counts-differ", "three-lanes",
        "device-twice", "too-many-devices"])
def test_invalid_route_maps(text, message):
    known = [role.device for role in r.ROLES]
    with pytest.raises(r.RouteError, match=message):
        r.validate_route_map(0, 4, r.parse_peer_routes(text), available_devices=known)


def test_opposite_ranks_listing_the_same_devices_fail_the_pairing_check():
    layout = r.Layout(r.Fabric.ring(8), (0, 4))
    maps = [{1: ("rocep1s0f0", "roceP2p1s0f1")}, {0: ("rocep1s0f0", "roceP2p1s0f1")}]
    problems = r.check_complementary(layout, maps)
    assert problems and "lane 0 of rank 0 leaves through port 0 and arrives on rank 1's port 1" in problems[0]
    assert "but rank 1 lists rocep1s0f0 for lane 0" in problems[0]


def test_a_lane_through_another_groups_spark_is_rejected():
    layout = r.Layout(r.Fabric.path(range(6)), tuple(range(6)))
    routes = r.derive_routes(layout, 2).route_map(0)
    routes[5] = ("rocep1s0f1", "roceP2p1s0f1")       # counter-clockwise, through Sparks 7 and 6
    with pytest.raises(r.RouteError, match="outside the group's fabric"):
        r.validate_route_map(0, 6, routes, layout=layout, max_relays=16)


def test_layout_text_forms():
    assert r.Layout.parse("ring:8").positions == tuple(range(8))
    sub = r.Layout.parse("ring:8:0,4")
    assert sub.fabric.kind == "cycle" and sub.positions == (0, 4)
    path = r.Layout.parse("path:4-7")
    assert path.fabric.kind == "path" and path.positions == (4, 5, 6, 7)
    explicit = r.Layout.parse("cables=0.port0-1.port0,0.port1-1.port1;positions=0,1")
    assert r.derive_routes(explicit).route_text(1) == "0=rocep1s0f0/roceP2p1s0f1"
    for bad in ("star:4", "ring:x", "path:0-3:0,0,1,2", "cables=0.port0-1.port0"):
        with pytest.raises(r.RouteError):
            r.Layout.parse(bad)


def test_route_text_round_trip_and_whitespace():
    parsed = r.parse_peer_routes(" 1 = rocep1s0f0 / roceP2p1s0f0 , 2=rocep1s0f1 ")
    assert parsed == {1: ("rocep1s0f0", "roceP2p1s0f0"), 2: ("rocep1s0f1",)}
    assert r.parse_peer_routes(r.format_peer_routes({1: ("a", "b"), 2: ("c", "d")})) == {1: ("a", "b"), 2: ("c", "d")}
    for bad in ("", "1", "x=rocep1s0f0", "1=a,,2=b"):
        with pytest.raises(r.RouteError):
            r.parse_peer_routes(bad)


@pytest.mark.parametrize("text, busiest", [("path:0-3", 2), ("ring:8", 6), ("ring:4", 1), ("ring:2", 0)])
def test_forward_windows_keep_every_relay_queue_within_its_share(text, busiest):
    layout = r.Layout.parse(text)
    derived = r.derive_routes(layout, 2)
    maps = [derived.route_map(rank) for rank in range(layout.world)]
    queues = r.relay_queues(layout, maps)
    assert max((len(members) for members in queues.values()), default=0) == busiest
    tables = [r.forward_windows(layout, maps, rank) for rank in range(layout.world)]
    for rank in range(layout.world):
        assert tables[rank][rank] == [0, 0]
        for peer, lanes in derived.ranks[rank].items():
            for lane in lanes:
                window = tables[rank][peer][lane.lane]
                if lane.relays:
                    assert window % r.DEFAULT_FORWARD_CHUNK == 0 and 0 < window <= r.DEFAULT_FORWARD_WINDOW
                else:
                    assert window == 0
    for members in queues.values():
        held = sum(tables[rank][peer][lane] for rank, peer, lane in members)
        assert held <= r.RELAY_QUEUE_SHARE * r.DEFAULT_HAIRPIN_QUEUE


def test_forward_windows_on_the_measured_layouts():
    path = r.Layout.parse("path:0-3")
    derived = r.derive_routes(path, 2)
    maps = [derived.route_map(rank) for rank in range(4)]
    assert r.forward_windows(path, maps, 0) == [[0, 0], [0, 0], [131072, 131072], [131072, 131072]]
    ring = r.Layout.parse("ring:8")
    derived = r.derive_routes(ring, 2)
    maps = [derived.route_map(rank) for rank in range(8)]
    table = r.forward_windows(ring, maps, 0)
    assert table[1] == table[7] == [0, 0]
    assert table[4] == [65536, 65536]
    assert r.forward_windows(ring, maps, 0, max_window=0) == [[0, 0]] * 8
    with pytest.raises(r.RouteError, match="forward chunk"):
        r.forward_windows(ring, maps, 0, chunk=1000)


@pytest.mark.parametrize("text, order", [
    ("path:0-3", (0, 1, 2, 3)), ("ring:8", tuple(range(8))), ("ring:2", (0, 1)), ("path:0-1", (0, 1)),
    ("cables=0.port0-1.port1,1.port0-2.port1;positions=2,1,0", (2, 1, 0)), ("ring:8:0,2,4,6", None),
])
def test_chain_order_follows_cable_neighbors(text, order):
    layout = r.Layout.parse(text)
    derived = r.derive_routes(layout, 2)
    maps = [derived.route_map(rank) for rank in range(layout.world)]
    assert r.chain_order(layout, maps) == order


def test_chain_order_needs_direct_lanes_between_neighbors():
    layout = r.Layout.parse("path:0-3")
    derived = r.derive_routes(layout, 2)
    maps = [dict(derived.route_map(rank)) for rank in range(4)]
    maps[1][2] = ("rocep1s0f1", "roceP2p1s0f1")          # toward rank 2 the long way: off the path
    assert r.chain_order(layout, maps) is None


def test_bounds_of_the_schedules_on_a_path_and_a_ring():
    from sparkring_sircl import bounds

    mib64, mib96 = 64 << 20, 96 << 20
    path, ring8 = r.Layout.parse("path:0-3"), r.Layout.parse("ring:8")
    order4, order8 = (0, 1, 2, 3), tuple(range(8))

    def ms(layout, collective, schedule, nbytes, order):
        return round(bounds.bound_seconds(layout, 2, collective, schedule, nbytes, order) * 1e3, 2)

    # The host interface (24 GB/s sending, 26.8 GB/s receiving): the chain all-reduce sends 2M at a middle rank.
    assert ms(path, "all_reduce", "chain", mib64, order4) == 5.59
    # Two-shot pieces: 1.5M per host, but the middle cable carries 2M each way.
    assert ms(path, "all_reduce", "twoshot", mib64, order4) == 5.59
    # The ring closed through the relays: 1.5M per host and per cable direction.
    assert ms(path, "all_reduce", "ring", mib64, order4) == 4.19
    assert ms(path, "reduce_scatter", "chain", mib64, order4) == 3.13     # a middle rank receives 1.25M
    assert ms(path, "all_gather", "chain", mib64, order4) == 3.5          # a middle rank sends 1.25M
    for collective in ("reduce_scatter", "all_gather"):
        assert ms(path, collective, "scatter", mib64, order4) == 2.8       # the middle cable carries M
        assert ms(path, collective, "ring", mib64, order4) == 2.1          # 0.75M everywhere
    assert ms(ring8, "all_reduce", "ring", mib96, order8) == 7.34       # 1.75M per host on the 8-cycle
    loads = bounds.loads(r.derive_routes(path, 2), bounds.flows("reduce_scatter", "chain", 4, mib64, order4))
    assert [round(value / mib64, 2) for value in loads.egress] == [0.75] * 4
    assert [round(value / mib64, 2) for value in loads.ingress] == [0.25, 1.25, 1.25, 0.25]
    assert loads.binding() == "receive" and loads.limit() == "host"
    # One host rate for both directions (HOST_CAP_GBPS, the sending rate): the middle rank's 1.25M binds.
    assert loads.binding(bounds.HOST_CAP_GBPS) == "receive"
    assert round(loads.seconds(bounds.HOST_CAP_GBPS) * 1e3, 2) == 3.5
    assert round(bounds.bound_seconds(path, 2, "reduce_scatter", "chain", mib64, order4,
                                      host_gbps=bounds.HOST_CAP_GBPS) * 1e3, 2) == 3.5
    with pytest.raises(bounds.BoundError):
        bounds.flows("all_reduce", "scatter", 4, mib64)
    with pytest.raises(bounds.BoundError):
        bounds.flows("all_gather", "chain", 4, mib64, None)


def test_the_ring_over_a_path_closes_through_one_lane_per_relay_queue():
    path = r.Layout.parse("path:0-3")
    derived = r.derive_routes(path, 2)
    maps = [derived.route_map(rank) for rank in range(4)]
    order = r.chain_order(path, maps)
    queues = r.ring_queues(path, maps, order)
    # Rank 3's two lanes to rank 0 cross Sparks 2 and 1, each lane through its own function's queues.
    assert sorted(queues) == [(1, False, 1), (1, True, 1), (2, False, 1), (2, True, 1)]
    assert all(members in ([(3, 0, 0)], [(3, 0, 1)]) for members in queues.values())
    assert r.ring_window(path, maps, order) == (393216, [])
    ring8 = r.Layout.parse("ring:8")
    derived8 = r.derive_routes(ring8, 2)
    maps8 = [derived8.route_map(rank) for rank in range(8)]
    assert r.ring_queues(ring8, maps8, tuple(range(8))) == {} and r.ring_window(ring8, maps8, tuple(range(8))) == (0, [])
    # Two lanes on one function would share every relay queue: the ring cannot run.
    shared = [dict(m) for m in maps]
    shared[3][0] = (maps[3][0][0], maps[3][0][0])
    window, problems = r.ring_window(path, shared, order)
    assert problems and "carries ring lanes [(3, 0, 0), (3, 0, 1)]" in problems[0]
