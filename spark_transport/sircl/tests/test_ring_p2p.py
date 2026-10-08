"""The point-to-point ring cases offline: plans, launch commands and result merging."""

from __future__ import annotations

import json

import pytest

from sparkring_sircl.ring import p2p, remote
from sparkring_sircl.ring.site import Site


def _site(size: int = 8) -> Site:
    return Site.from_json({
        "schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "lan0",
        "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"spark{i}", "ssh": f"op@192.0.2.{10 + i}", "lan_address": f"192.0.2.{10 + i}"}
                 for i in range(size)]})


def test_the_ring_of_eight_plan_covers_every_relay_count_and_the_shifts():
    plan = p2p.build(_site(), "run1", digest="d" * 16)
    assert plan.world == 8
    assert [case.get("ranks") or case["distance"] for case in plan.cases] == [[0, 1], [0, 2], [0, 3], [0, 4], 1, 2, 4]
    assert [plan.hops[0][d] for d in (1, 2, 3, 4)] == [1, 2, 3, 4]
    assert not plan.unavailable
    # Every pair of the ring has a channel, so the relayed lanes of all 56 ordered pairs share the relay queues:
    # direct lanes have no window, relayed lanes 64 to 128 KiB.
    assert plan.windows[0][1] == (0, 0) and plan.windows[0][2] == (98304, 65536)
    assert plan.windows[0][4] == (65536, 65536)
    assert {w for row in plan.windows for lanes in row for w in lanes} == {0, 65536, 98304, 131072}
    text = p2p.render_text(plan)
    assert "pair 0-4 (Sparks 0-4): 3 relay(s); windows 0->4 [65536 B, 65536 B]" in text
    assert "shift 4: every rank sends to rank r+4" in text
    document = plan.to_json()
    assert document["schema"] == p2p.SCHEMA and document["p2p"]["hops"][0][4] == 4
    assert json.loads(json.dumps(document))["p2p"]["options"]["sizes"][-1] == 64 << 20


def test_a_path_of_four_and_chosen_cases():
    plan = p2p.build(_site(), "run1", positions=(2, 3, 4, 5), cases="0-3,shift1", digest="d" * 16)
    assert plan.configuration.ranks[0].position == 2 and plan.hops[0][3] == 3
    assert plan.cases == ({"kind": "pair", "ranks": [0, 3]}, {"kind": "shift", "distance": 1})
    assert [c.get("ranks") for c in p2p.build(_site(), "run1", positions=(0, 1, 2, 3), digest="d" * 16).cases] == [
        [0, 1], [0, 2], [0, 3], None]
    for text, problem in (("0-0", "does not join"), ("0-9", "does not join"), ("shift8", "distance from 1 to 7"),
                          ("x", "neither")):
        with pytest.raises(p2p.P2PPlanError, match=problem):
            p2p.parse_cases(text, 8)
    with pytest.raises(p2p.P2PPlanError, match="SIRCL_P2P_"):
        p2p.P2POptions(session_env=(("SIRCL_FORWARD_WINDOW_BYTES", "0"),))


def test_windows_off_leave_relayed_pairs_without_channels_in_the_plan():
    plan = p2p.build(_site(), "run1", options=p2p.P2POptions(window_bytes=0), digest="d" * 16)
    assert (0, 2) in {(a, b) for a, b, _ in plan.unavailable}
    assert "NO CHANNEL 0->2" in p2p.render_text(plan)


def test_the_launch_runs_the_point_to_point_worker_with_the_harness_label():
    plan = p2p.build(_site(), "run1", digest="d" * 16)
    command = p2p.docker_run(plan, plan.configuration.ranks[3])
    assert "sparkring_sircl.ring.p2p_worker --plan /sircl/runs/run1/p2p/p2p-plan.json --global-rank 3" in command
    assert f"{remote.LABEL}=run1" in command and "SIRCL_BUILD_CACHE_DIR=/sircl/build-cache" in command
    assert p2p.main(["run", "--site", "missing.json", "--print"]) == 2


def test_sizes_set_the_rounds_and_bursts():
    assert p2p.DEFAULT_SIZES[0] == 4096 and p2p.DEFAULT_SIZES[-1] == 64 << 20
    assert p2p.latency_rounds(4096, 200) == 200 and p2p.latency_rounds(64 << 20, 200) == 10
    assert p2p.burst_messages(4096, 256 << 20) == 256 and p2p.burst_messages(64 << 20, 256 << 20) == 4


def test_results_merge_into_rows_and_a_status():
    plan = p2p.build(_site(2), "run1", positions=(0, 1), digest="d" * 16).to_json()
    rank0 = {"records": [{"case": "pair 0-1", "bytes": 4096, "hops": 1, "checked": 3, "mismatches": 0,
                          "latency": {"p50_us": 6.5, "p99_us": 9.0}, "bandwidth_gbps": 1.25}],
             "channels": {"rank": 0}}
    rank1 = {"records": [{"case": "pair 0-1", "bytes": 4096, "hops": 1, "checked": 3, "mismatches": 0}],
             "channels": {"rank": 1}}
    merged = p2p.merge(plan, [rank0, rank1])
    assert merged["status"] == "passed" and merged["rows"][0]["checked"] == 6
    assert "6.5/9.0" in p2p.table(merged) and "1.25" in p2p.table(merged)
    rank1["records"][0]["mismatches"] = 1
    assert p2p.merge(plan, [rank0, rank1])["status"] == "failed"
    assert p2p.merge(plan, [rank0, None])["problems"] == ["rank 1 wrote no result"]
