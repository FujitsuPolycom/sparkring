"""Offline selection must not turn an unbounded candidate into a recommendation."""
import copy
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest

from sparkring_sircl import routes, schedule_selection as select, subset


@pytest.mark.parametrize("kind", ("path", "ring"))
@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("collective", select.COLLECTIVES)
def test_layouts_and_representative_placements(kind, n, collective):
    fabric = select.fabric_named(f"{kind}{n}")
    placements = {tuple(range(n)), tuple(range(max(2, n - 2))), tuple(range(max(2, n // 2))),
                  (0, n - 1), tuple(range(0, n, 2))}
    for members in placements:
        if len(members) < 2:
            continue
        outside = tuple(p for p in range(n) if p not in members)
        result = select.plan(fabric, members, co_resident=[outside] if outside else [],
                             sizes=[4096, 4 << 20], collective=collective)
        assert result["advisory_only"]
        for row in result["sizes"]:
            if row["chosen"] is None:
                assert len(row["rejected"]) == 10
                continue
            assert row["chosen"] in row["candidates"]
            for candidate in row["candidates"]:
                assert candidate["congested_us"] >= candidate["predicted_us"] > 0
                for queue in candidate["queues"]:
                    assert (3 * queue["data_bytes"] + queue["reverse_data_bytes"]) / 2 + queue["control_bytes"] <= queue["capacity_bytes"]
                for path in candidate["paths"]:
                    assert bool(path["window_bytes"]) == (len(path["path"]) > 2)
                    assert len(path["path"]) <= 5


def test_subsets_fixed_arena_ring_bound_is_maximal_in_its_envelope():
    for n in range(4, 9):
        for count in range(max(3, n - 2), n):
            for members in combinations(range(n), count):
                layout = routes.Layout(routes.Fabric.ring(n), members)
                derived = routes.derive_routes(layout)
                maps = [derived.route_map(r) for r in range(count)]
                try:
                    subset.order(layout)
                except routes.RouteError:
                    continue
                window, rows = subset.ring_packet_bound(layout, maps)
                assert all(row["bound_bytes"] <= select.QUEUE for row in rows)
                if window:
                    assert any(row["next_chunk_bound_bytes"] > select.QUEUE for row in rows)
                outside = tuple(p for p in range(n) if p not in members)
                if len(outside) == 2:
                    other = routes.Layout(layout.fabric, outside)
                    other_routes = routes.derive_routes(other)
                    assert routes.relay_queues(other, [other_routes.route_map(r) for r in range(2)]) == {}
    layout = routes.Layout.parse("ring:8:0,1,2,3,4,5")
    derived = routes.derive_routes(layout)
    window, rows = subset.ring_packet_bound(layout, [derived.route_map(r) for r in range(6)])
    assert window == 196608
    closing = [row for row in rows if row["queue"] == [6, True, 0]][0]
    assert closing["arena_bytes"] == closing["reverse_arena_bytes"] == 86016
    assert closing["control_bytes"] == 98304
    assert closing["bound_bytes"] == 522240


def test_ack_only_queues_and_joint_reservations():
    fabric = routes.Fabric.path(range(4))
    lane = routes.derive_lanes(fabric, 3, 0, 1)[0]
    windows, proof = select.packet_windows({(0, 0, 1, 0): lane}, chunk=32768)
    assert windows[(0, 0, 1, 0)] == 327680
    assert any(q["data_bytes"] == 0 and q["reverse_data_bytes"] == 327680 for q in proof)
    crowded = {(g, 0, 1, 0): lane for g in range(40)}
    with pytest.raises(ValueError, match="packet envelope"):
        select.packet_windows(crowded)
    for options in ({"queue_bytes": 1048576}, {"chunk": 0}, {"chunk": 15}):
        with pytest.raises(ValueError):
            select.packet_windows({(0, 0, 1, 0): lane}, **options)


def test_measurement_precedence_is_exact_and_cannot_bypass_safety():
    fabric = select.fabric_named("ring8")
    kwargs = dict(co_resident=[(6, 7)], sizes=[4 << 20], conditions={
        "image": "fixture", "drivers": "fixture", "dtype": "bfloat16", "shape": "flat",
        "rotate_buffers": 8, "co_workload": "fixed decode fixture"})
    original = select.plan(fabric, tuple(range(6)), **kwargs)
    slower = max(original["sizes"][0]["candidates"], key=lambda c: c["predicted_us"])
    measured = {"schema": select.MEASUREMENTS, "key": original["key"], "rows": [
        {"bytes": 4 << 20, "candidate": slower["id"], "period_us": 123.0, "exact": True,
         "drops": 0, "counters_valid": True, "isolation_passed": True, "evidence": "offline test fixture"}]}
    answer = select.plan(fabric, tuple(range(6)), measurements=measured, **kwargs)
    assert answer["sizes"][0]["chosen"]["id"] == slower["id"]
    assert answer["sizes"][0]["source"] == "measured"
    for field, value in (("exact", False), ("drops", 1), ("period_us", float("nan")),
                         ("counters_valid", False), ("isolation_passed", False)):
        bad = copy.deepcopy(measured)
        bad["rows"][0][field] = value
        with pytest.raises(ValueError, match="measurement needs"):
            select.plan(fabric, tuple(range(6)), measurements=bad, **kwargs)
    bad = copy.deepcopy(measured)
    bad["key"]["co_resident"] = []
    with pytest.raises(ValueError, match="does not match"):
        select.plan(fabric, tuple(range(6)), measurements=bad, **kwargs)
    for rows in ({}, [None], [{"bytes": -16, "candidate": slower["id"]}]):
        bad = {**measured, "rows": rows}
        with pytest.raises(ValueError, match="measurement"):
            select.plan(fabric, tuple(range(6)), measurements=bad, **kwargs)
    measured["rows"][0]["candidate"] = "unsafe/nonexistent"
    answer = select.plan(fabric, tuple(range(6)), measurements=measured, **kwargs)
    assert answer["sizes"][0]["source"] == "model"


def test_nonflat_and_missing_relay_plan():
    result = select.plan(select.fabric_named("ring8"), tuple(range(6)), collective="all_gather",
                         sizes=[4 << 20], flat=False, fabric_record={"relay_plan": None})
    assert not result["sizes"][0]["candidates"]
    assert all("chain" not in c["schedule"] for c in result["sizes"][0]["candidates"])


@pytest.mark.parametrize("members,other", [((0, 0), []), ((0, 9), []), ((0, 1), [(1, 2)]), ((0,), [])])
def test_bad_placements_refused(members, other):
    with pytest.raises(ValueError):
        select.plan(select.fabric_named("ring8"), members, co_resident=other)


def test_cli_is_offline_and_deterministic():
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "sparkring_sircl.schedule_selection", "--fabric", "ring7",
               "--placement", "6,0-3", "--co-resident", "4-5", "--sizes", "4096"]
    runs = [subprocess.run(command, cwd=root, capture_output=True, text=True, check=True) for _ in range(2)]
    assert runs[0].stdout == runs[1].stdout
    assert json.loads(runs[0].stdout)["advisory_only"]


def test_rtt_calibration_and_sensitivity():
    assert select.rtt_us(65536, 2) == pytest.approx(2 * 65536 / 9400)
    assert select.rtt_us(393216, 2) == pytest.approx(2 * 393216 / 21110)
    assert select.rtt_us(245760, 2) > select.rtt_us(163840, 2)
    assert select.rtt_us(163840, 3) > select.rtt_us(163840, 2)
    assert select.rtt_us(163840, 2, congested=True) > select.rtt_us(163840, 2)


def test_setup_fabric_document_and_invalid_identity(tmp_path):
    from runtime.common import fabric_document, fabric_layout
    from runtime.host.test_fabric import document_of

    _, document, _ = document_of(fabric_layout.layout("cycle", 8))
    path = tmp_path / "fabric.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    fabric, record = select.document_fabric(path)
    assert fabric == select.fabric_named("ring8")
    assert record["relay_plan"] == document["relays"]
    assert record["document_sha256"] == fabric_document.digest(document)
    result = select.plan(fabric, tuple(range(6)), co_resident=[(6, 7)],
                         sizes=[4096], fabric_record=record)
    assert result["key"]["fabric_record"] == record
    document["cables"][0]["ends"][0]["port"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        select.document_fabric(path)


@pytest.mark.parametrize("active", (False, True))
def test_larger_bound_native_progress_with_direct_complement(simulator_library, monkeypatch, active):
    from concurrent.futures import ThreadPoolExecutor
    from sparkring_sircl.testing import fabric as sim, ring_links

    monkeypatch.setenv("SIRCL_FORWARD_PROOF", "0")
    layout = routes.Layout.parse("ring:8:0,1,2,3,4,5")
    derived = routes.derive_routes(layout)
    window, _ = subset.ring_packet_bound(layout, [derived.route_map(r) for r in range(6)])
    links = ring_links.RingLinks(str(simulator_library), layout, subset_ring=True,
                                 slots=12, slot_bytes=65536, timeout=30)
    # Only the in-memory link harness uses the offline bound; runtime admission stays unchanged.
    links._ring_window = window
    pair = None
    try:
        if active:
            pair = sim.LocalSession(str(simulator_library), routes.Layout.parse("ring:8:6,7"),
                                    node_base=6, reset=False)
            pair.connect()
        links.connect()
        links.delay_link_ops(5, 2000)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(sim.run_oneshot_ops, pair, [16384] * 100) if pair else None
            ring_links.run_ring_ops(links, [("gather", 524304, 65536, 0, 1),
                                            ("reduce", 524352, 65536, 1, 1)])
            if pending:
                pending.result(timeout=30)
                assert not any(proxy.failed() for proxy in pair.proxies)
        assert not any(proxy.failed() for proxy in links.proxies)
        assert all(proxy.stats()["forward_max_unacked_bytes"] <= window + 256 for proxy in links.proxies)
    finally:
        if pair:
            pair.close()
        links.close()
