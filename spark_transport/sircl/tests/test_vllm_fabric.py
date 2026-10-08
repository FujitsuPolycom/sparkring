"""The vLLM adapter's fabric model: route maps, NCCL policy and relay load.

Route maps are checked against every layout of the reference vectors in
``tests/data/routes.json``; the NCCL policy table states,
for the placements the eight-Spark ring serves, which groups may use NCCL at
all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkring_sircl.vllm import fabric, guard
from sparkring_sircl.vllm.fabric import FabricError, Layout, NcclPolicy, describe_group

VECTORS = Path(__file__).resolve().parent / "data" / "routes.json"


def _vector_layout(key: str) -> tuple[Layout, list[int] | None]:
    if key == "pair-port0-port0":
        return Layout.pair(1), None
    if key == "pair-two-cables-same-ports":
        return Layout.pair(2), None
    if key == "pair-ring-of-2":
        return Layout.ring(2), None
    if key == "triangle":
        return Layout.ring(3), None
    size = int(key.split("ring-of-")[1].split("-")[0])
    return Layout.ring(size), (list(range(size)) if "-sub-" in key else None)


@pytest.mark.skipif(not VECTORS.is_file(), reason="the reference vectors are not present")
def test_route_maps_equal_every_reference_vector():
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    checked = 0
    for key, vector in vectors.items():
        layout, parent = _vector_layout(key)
        if key == "ring-of-8-tp6-0-5":
            # Its end-to-end lanes cross four relays, above the limit; the vector shows the map anyway.
            with pytest.raises(FabricError, match="4 relays"):
                describe_group(layout, vector["rank_positions"], parent=parent,
                               lane_count=vector["lanes"])
            topology = describe_group(layout, vector["rank_positions"], parent=parent,
                                      lane_count=vector["lanes"], max_relays=4)
        else:
            topology = describe_group(layout, vector["rank_positions"], parent=parent,
                                      lane_count=vector["lanes"])
        for entry in vector["ranks"]:
            expected = {int(peer): tuple(devices) for peer, devices in entry["route_map"].items()}
            assert topology.route_map(entry["rank"]) == expected, (key, entry["rank"])
            assert fabric.format_routes(expected) == entry["route_text"]
            for peer, lanes in entry["lanes"].items():
                derived = topology.lanes(entry["rank"], int(peer))
                assert [lane.hops for lane in derived] == [lane["hops"] for lane in lanes]
                assert [list(lane.relays) for lane in derived] == [lane["relays"] for lane in lanes]
                assert [lane.remote_device() for lane in derived] == [lane["remote"] for lane in lanes]
            checked += 1
    assert checked >= 80


@pytest.mark.parametrize("text,reason", [
    ("1=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f1/roceP2p1s0f1", "must name ranks"),
    ("0=rocep1s0f0,1=rocep1s0f0,2=rocep1s0f0,3=rocep1s0f1", "names its own rank"),
    ("1=rocep1s0f0,1=roceP2p1s0f0,2=rocep1s0f0,3=rocep1s0f1", "names rank 1 twice"),
    ("1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0,3=rocep1s0f1/roceP2p1s0f1", "different lane counts"),
    ("1=rocep1s0f0/roceP2p1s0f0/rocep1s0f1,2=a/b,3=c/d", "distinct device names"),
    ("1=rocep1s0f0/rocep1s0f0,2=a/b,3=c/d", "distinct device names"),
    ("1-rocep1s0f0,2=a,3=b", "is not peer=device"),
])
def test_negative_route_maps_are_refused_with_their_reason(text, reason):
    with pytest.raises(FabricError, match=reason):
        fabric.parse_routes(text, world=4, rank=0)


def test_a_given_map_must_match_the_layout():
    ring = describe_group(Layout.ring(8), range(8))
    # Opposite ranks 0 and 4 both listing port-0 primary for lane 0.
    wrong = dict(ring.route_map(4))
    wrong[0] = ("rocep1s0f0", "roceP2p1s0f1")
    with pytest.raises(FabricError, match="needs"):
        fabric.check_routes(ring, 4, wrong)
    path = describe_group(Layout.ring(8), range(4))
    given = path.route_map(0)
    fabric.check_routes(path, 0, given)
    # A TP4 group must not route rank 0 to rank 3 counter-clockwise through Sparks 7-4.
    given[3] = ("rocep1s0f1", "roceP2p1s0f1")
    with pytest.raises(FabricError, match="needs"):
        fabric.check_routes(path, 0, given)


PLACEMENTS = [
    # (layout, group positions, parent, policy)
    ("ring:8", [0, 1, 2, 3], None, NcclPolicy.NONE),          # TP4 on Sparks 0-3
    ("ring:8", [4, 5, 6, 7], None, NcclPolicy.NONE),          # second TP4 of 2xTP4
    ("ring:8", list(range(8)), None, NcclPolicy.RING),        # TP8
    ("ring:8", [6, 7], None, NcclPolicy.ALL),                 # one TP2 of 4xTP2
    ("ring:8", [7, 0], None, NcclPolicy.ALL),                 # a pair across the ring seam
    ("ring:8", [0, 1, 2, 3], list(range(8)), NcclPolicy.NONE),  # DCP4 inside TP8
    ("ring:8", [2, 3], list(range(8)), NcclPolicy.ALL),       # DCP2 inside TP8
    ("ring:8", [0, 4], list(range(8)), NcclPolicy.NONE),      # opposite members
    ("ring:4", [0, 1, 2, 3], None, NcclPolicy.RING),          # the four-Spark ring
    ("ring:3", [0, 1, 2], None, NcclPolicy.ALL),              # triangle
    ("pair:2", [0, 1], None, NcclPolicy.ALL),
]


@pytest.mark.parametrize("layout,members,parent,policy", PLACEMENTS)
def test_nccl_policy_follows_the_cabling(layout, members, parent, policy):
    topology = describe_group(Layout.parse(layout), members, parent=parent)
    assert topology.nccl_policy is policy
    if policy is NcclPolicy.NONE:
        assert "no cable between" in topology.nccl_reason
    shape_free, _ = fabric.nccl_policy_of(Layout.parse(layout), members)
    assert shape_free is policy


def test_tp4_on_a_ring_of_eight_names_the_relayed_pair():
    topology = describe_group(Layout.ring(8), [0, 1, 2, 3])
    assert topology.fabric.kind == "path"
    assert topology.uncabled_pairs() == ((3, 0),)
    assert "ranks 3-0" in topology.nccl_reason and "2 relays" in topology.nccl_reason
    assert not NcclPolicy.NONE.allows("all_reduce")
    assert topology.max_relays() == 2


def test_groups_that_do_not_own_a_path_or_cycle_are_refused():
    with pytest.raises(FabricError, match="consecutive"):
        describe_group(Layout.ring(8), [0, 2, 4, 6])
    with pytest.raises(FabricError, match="not inside the parent"):
        describe_group(Layout.ring(8), [0, 5], parent=[0, 1, 2, 3])
    # The same members are a valid subgroup of the full ring, routed over its cables.
    assert describe_group(Layout.ring(8), [0, 2, 4, 6], parent=range(8)).nccl_policy is NcclPolicy.NONE


def test_relay_factor_and_per_peer_op_size():
    ring = describe_group(Layout.ring(8), range(8))
    path = describe_group(Layout.ring(8), range(4))
    pair = describe_group(Layout.ring(8), [6, 7])
    assert ring.relay_factor() == 3.0 and ring.per_peer_op_bytes()[0] == 131072
    assert path.relay_factor() == 1.0 and path.per_peer_op_bytes()[0] == 262144
    assert pair.relay_factor() == 0 and pair.per_peer_op_bytes() == (None, "no relayed lane")
    six = describe_group(Layout.ring(6), range(6))
    value, basis = six.per_peer_op_bytes()
    assert value == int(512 * 1024 * 0.75 / 1.5) // 16 * 16 and "unmeasured" in basis
    assert path.per_peer_op_bytes(override=65536) == (65536, "SIRCL_RELAY_PER_PEER_BYTES")


def test_ring_groups_need_nccl_restricted_to_its_ring_algorithm():
    ring = describe_group(Layout.ring(8), range(8))
    policy, reason = guard.effective_policy(ring.nccl_policy, ring.nccl_reason, environ={})
    assert policy is NcclPolicy.NONE and "NCCL_ALGO" in reason
    environ = {"NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"}
    policy, _ = guard.effective_policy(ring.nccl_policy, ring.nccl_reason, environ=environ)
    assert policy is NcclPolicy.RING
    policy, _ = guard.effective_policy(NcclPolicy.ALL, "pair", nccl_mode="never")
    assert policy is NcclPolicy.NONE
    assert NcclPolicy.RING.allows("all_gather") and not NcclPolicy.RING.allows("all_to_all")
    assert not NcclPolicy.RING.allows("send")


def test_layout_text_and_identity():
    assert Layout.parse("ring:8").describe() == "ring:8"
    assert Layout.parse("pair").describe() == "pair:1"
    with pytest.raises(FabricError):
        Layout.parse("mesh:8")
    topology = describe_group(Layout.ring(8), [4, 5, 6, 7])
    assert topology.identity() == {"layout": "ring:8", "fabric": "path", "fabric_order": [4, 5, 6, 7],
                                   "positions": [4, 5, 6, 7], "lanes": 2}
