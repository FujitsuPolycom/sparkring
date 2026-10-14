"""Active subset chain policy plus retained, inactive ring budget/progress helpers."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import pytest
from sparkring_sircl import protocol, routes, subset
from sparkring_sircl.p2p import budget
from sparkring_sircl.testing import ring_links
from sparkring_sircl.vllm import fabric
from sparkring_sircl.vllm.serve import plan


def layout_maps(n, positions, lanes=2):
    layout = routes.Layout.parse(f"ring:{n}:" + ",".join(map(str, positions)))
    derived = routes.derive_routes(layout, lanes)
    return layout, [derived.route_map(rank) for rank in range(layout.world)]


@pytest.mark.parametrize("n", range(4, 9))
@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("lanes", (1, 2))
def test_all_subsets_admit_a_bounded_envelope_or_refuse(n, active, lanes):
    for k in range(3, min(n, 8)):
        for positions in combinations(range(n), k):
            layout, maps = layout_maps(n, positions, lanes)
            outside = sorted(set(range(n)) - set(positions))
            arc = sum((p - 1) % n not in positions for p in positions) == 1
            admitted = arc and len(outside) <= 2
            if not admitted:
                with pytest.raises(routes.RouteError):
                    routes.chain_order(layout, maps, subset_ring=True)
                with pytest.raises(routes.RouteError):
                    routes.forward_windows(layout, maps, 0, chunk=4096, subset_ring=True)
                continue
            order = routes.chain_order(layout, maps, subset_ring=True)
            window, problems = routes.ring_window(layout, maps, order, subset_ring=True)
            assert not problems and window >= protocol.LINK_WINDOW_CHUNK
            b = subset.payload_budget(layout, maps)
            queues = routes.relay_queues(layout, maps)
            rings = routes.ring_queues(layout, maps, order)
            tables = [routes.forward_windows(layout, maps, rank, chunk=4096, subset_ring=True)
                      for rank in range(k)]
            other = {}
            if active and len(outside) > 1:
                other_layout, other_maps = layout_maps(n, outside)
                other = routes.relay_queues(other_layout, other_maps)
                # TP, DCP and any number of sessions confined to this pair use no hairpin queue.
                assert other == {}
            for (p, c, e), members in queues.items():
                reverse = queues.get((p, c, 1 - e), ())
                forward = sum(tables[r][peer][lane] for r, peer, lane in members)
                back = sum(tables[r][peer][lane] for r, peer, lane in reverse)
                ring = len(rings.get((p, c, e), ())) * window
                ring_back = len(rings.get((p, c, 1 - e), ())) * window
                assert max(forward, back, ring, ring_back) <= b
                # Saturate both operation families, both directions and every control credit.
                data = forward + ring
                ack_payload = back + ring_back
                controls = (len(members) + len(reverse)) * subset.CONTROL_BYTES_PER_LANE
                wire = data + (data + ack_payload) // 2 + controls
                assert wire <= routes.DEFAULT_HAIRPIN_QUEUE
                assert (p, c, e) not in other


def test_six_of_eight_lanes_are_carried_by_the_universal_table():
    from sparkring_sircl.fabric import plan as fabric_plan
    from test_fabric_plan import FACTS, _plan

    universal = _plan(layout="ring8").groups[0]
    layout, maps = layout_maps(8, range(6))
    derived = routes.derive_routes(layout)
    for rank, peers in enumerate(derived.ranks):
        for peer, lanes in peers.items():
            for lane in lanes:
                if lane.hops == 1:
                    assert layout.fabric.cross(rank, lane.local.port).dst == peer
                    continue
                destination = FACTS.port(peer, lane.remote.netdev).address
                actual = universal.spark(rank).route_to(destination)
                assert actual.netdev == lane.local.netdev and actual.cables == lane.cables
                trace = fabric_plan.trace({s.position: s for s in universal.sparks}, FACTS, 8, rank, destination)
                assert trace.outcome == "delivered"
    assert derived.lanes_to(5, 0)[0].relays == (6, 7)
    assert routes.ring_window(layout, maps, tuple(range(6)), subset_ring=True) == (65536, [])


def test_wrapped_and_permuted_rank_order():
    layout, maps = layout_maps(8, (1, 7, 2, 0, 5, 6))
    assert routes.chain_order(layout, maps, subset_ring=True) == (4, 5, 1, 3, 0, 2)


def test_retained_budget_helpers_reject_unsafe_overrides():
    layout, maps = layout_maps(8, range(6))
    with pytest.raises(routes.RouteError, match="forward windows"):
        routes.forward_windows(layout, maps, 0, max_window=0, subset_ring=True)
    with pytest.raises(routes.RouteError, match="cannot hold"):
        routes.forward_windows(layout, maps, 0, chunk=32768, subset_ring=True)
    with pytest.raises(routes.RouteError, match="capacity"):
        routes.ring_window(layout, maps, tuple(range(6)), queue_bytes=1048576, subset_ring=True)
    maps[0][5] = tuple(reversed(maps[0][5]))
    with pytest.raises(routes.RouteError, match="universal"):
        routes.chain_order(layout, maps, subset_ring=True)
    with pytest.raises(routes.RouteError, match="0 or 1"):
        subset.enabled({subset.VARIABLE: "yes"})


def test_planner_and_p2p_reservation():
    group = fabric.describe_group(fabric.Layout.ring(8), list(range(6)), parent=list(range(8)))
    off = plan.session_settings(group, name="tp", groups="", scoped=True)
    on = plan.session_settings(group, name="tp", groups="", scoped=True, subset_ring=True,
                               schedules={"large_schedule": "ring", "gather_schedule": "ring"})
    assert not off.chain_area and not off.link_area
    assert not on.chain_area and on.link_area and not on.problems
    assert on.chain_order == tuple(range(6)) and "disabled" in on.ring
    assert on.schedules == {"large_schedule": "pieces", "gather_schedule": "auto", "scatter_schedule": "pieces"}
    layout, _ = layout_maps(8, range(6))
    lanes = budget.LaneSet.of("tp", layout)
    reserved = budget.session_reservation(lanes, budget.SessionSettings.from_env({subset.VARIABLE: "1"}))
    allocation = budget.allocate([[lanes]], reserved=reserved, max_window=131072, chunk=4096)
    assert not allocation.windows and allocation.unavailable


def test_bundle_explicit_option_reaches_the_rank_environment():
    from test_vllm_serve import make_bundle

    bundle = make_bundle(positions=tuple(range(6)), subset_ring=True,
                         large_schedule="ring", gather_schedule="ring")
    assert not bundle.sessions()[0].problems
    assert bundle.sessions()[0].ranks_on_fabric == (6, 8)
    assert all(r.environment["SIRCL_SUBSET_RING"] == "1" and
               r.environment["SIRCL_ROUTE_SCOPE"] == "fabric" for r in bundle.ranks)


def test_harness_on_and_off_share_the_universal_fabric():
    from sparkring_sircl.ring import plan as harness
    from test_ring_harness import _site

    records = []
    for enabled in (False, True):
        options = harness.Options(session_env=(("SIRCL_ROUTE_SCOPE", "fabric"),
                                                ("SIRCL_SUBSET_RING", str(int(enabled)))))
        built = harness.build_plan(_site(), "subset", "test", groups=[tuple(range(6))],
                                    options=options, digest="fixed")
        records.append(built)
    assert records[0].groups[0].layout == records[1].groups[0].layout
    assert records[0].groups[0].route_texts == records[1].groups[0].route_texts
    assert "no ring" in records[0].groups[0].ring
    assert "no ring" in records[1].groups[0].ring
    assert "direct chain" in records[1].groups[0].ring
    assert records[0].groups[0].lanes_detail == records[1].groups[0].lanes_detail


def test_subset_chain_runtime_selection_and_pieces_parity():
    pytest.importorskip("torch")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIRCL_")}
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("subset_chain_driver.py"))],
                            env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("large", ("auto", "pieces", "chain", "ring"))
@pytest.mark.parametrize("gather", ("auto", "pieces", "chain", "ring"))
def test_subset_planner_limits_explicit_schedules(large, gather):
    group = fabric.describe_group(fabric.Layout.ring(8), list(range(6)), parent=list(range(8)))
    settings = plan.session_settings(group, name="tp", groups="", scoped=True, subset_ring=True,
                                     schedules={"large_schedule": large, "gather_schedule": gather,
                                                "scatter_schedule": large})
    assert not settings.problems and not settings.chain_area
    assert settings.link_area == (gather != "pieces")
    assert settings.schedules == {"large_schedule": "pieces", "scatter_schedule": "pieces",
                                  "gather_schedule": "auto" if gather == "ring" else gather}
    assert "disabled" in settings.ring


def test_subset_tuner_only_offers_measurable_chain_gathers():
    from test_op_blocks import harness

    built = harness()
    built.world = 6
    built.session.subset_ring = True
    built.session.chain_available = built.session.ring_available = False
    for collective, size, expected in (("all_reduce", 8 << 20, {"pieces"}),
                                       ("reduce_scatter", 6 << 20, {"pieces"}),
                                       ("all_gather", 1 << 20, {"pieces"}),
                                       ("all_gather", 4 << 20, {"pieces", "chain"})):
        assert {choice.get("schedule") for choice, *_ in built.tune_candidates(collective, size)} == expected


@pytest.mark.parametrize("n", range(4, 9))
@pytest.mark.parametrize("lanes", (1, 2))
def test_active_chain_adds_zero_relay_bytes_to_legacy_pieces_bound(n, lanes):
    for k in range(max(3, n - 2), n):
        for start in range(n):
            positions = tuple((start + i) % n for i in range(k))
            layout, maps = layout_maps(n, positions, lanes)
            order = subset.validate(layout, maps)
            derived = routes.derive_routes(layout, lanes)
            # Both directions include data and returning ACKs. No closing link exists.
            for a, b in zip(order, order[1:]):
                assert all(not lane.relays for lane in (*derived.lanes_to(a, b), *derived.lanes_to(b, a)))
            outside = sorted(set(range(n)) - set(positions))
            if len(outside) == 2:
                other_layout, other_maps = layout_maps(n, outside, lanes)
                assert not routes.relay_queues(other_layout, other_maps)
            tables = [routes.forward_windows(layout, maps, r) for r in range(k)]
            for members in routes.relay_queues(layout, maps).values():
                payload = sum(tables[r][peer][lane] for r, peer, lane in members)
                assert payload <= int(routes.RELAY_QUEUE_SHARE * routes.DEFAULT_HAIRPIN_QUEUE)


def test_subset_setting_must_agree_even_when_rank_zero_omits_it():
    from sparkring_sircl.agreement import agreement_failures

    records = [(None, b"", {"lane_count": 1, "lane_counts": [0, 1], "route_map": {1: ("a",)}}),
               (None, b"", {"lane_count": 1, "lane_counts": [1, 0], "route_map": {0: ("a",)},
                              "subset_ring": True})]
    assert "rank 1 subset ring setting differs from rank 0" in agreement_failures(records, None)


@pytest.mark.parametrize("value", (None, "0"))
def test_off_plan_and_setup_record_bytes_match_origin_main(value):
    pytest.importorskip("torch")
    tests = Path(__file__).parent
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIRCL_")}
    env["PYTHONPATH"] = str(tests.parent)
    if value is not None:
        env[subset.VARIABLE] = value
    result = subprocess.run([sys.executable, str(tests / "subset_compat_driver.py")],
                             env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads((tests / "data/subset-off-baseline.json").read_text())


@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("n,k", [(n, k) for n in range(4, 9) for k in range(max(3, n - 2), n)])
def test_retained_subset_ring_helper_native_progress(simulator_library, n, k, active):
    from sparkring_sircl.testing import fabric as sim

    layout, _ = layout_maps(n, range(k))
    links = ring_links.RingLinks(str(simulator_library), layout, subset_ring=True,
                                 slots=12, slot_bytes=65536, timeout=30)
    pair = None
    try:
        if active and n - k == 2:
            other, _ = layout_maps(n, range(k, n))
            pair = sim.LocalSession(str(simulator_library), other, node_base=k, reset=False)
            pair.connect()
        links.connect()
        links.delay_link_ops(k - 1, 2000)
        # More than one ring window, changing operations and repeated sequence slots.
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(sim.run_oneshot_ops, pair, [16384] * 100) if pair else None
            ring_links.run_ring_ops(links, [("gather", 262160, 65536, 0, 1),
                                            ("reduce", 262176, 65536, 1, 1),
                                            ("gather", 131088, 65536, 0, 0),
                                            ("reduce", 196640, 65536, 0, 0)])
            if pending:
                pending.result(timeout=30)
                assert not any(proxy.failed() for proxy in pair.proxies)
        assert not any(proxy.failed() for proxy in links.proxies)
        assert all(proxy.stats()["forward_max_unacked_bytes"] <= links._ring_window + 256
                   for proxy in links.proxies)
    finally:
        if pair:
            pair.close()
        links.close()
