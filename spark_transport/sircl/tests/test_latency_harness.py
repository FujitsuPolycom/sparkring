"""The latency model and the ring harness's ring-latency configuration offline."""

from __future__ import annotations

import json

import pytest

from sparkring_sircl import latency_model as lm
from sparkring_sircl import routes
from sparkring_sircl.ring import cli, plan, remote, summary, worker
from sparkring_sircl.ring.site import Site


def _site_document(size: int = 8) -> dict:
    return {
        "schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "lan0",
        "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"spark{i}", "ssh": f"op@192.0.2.{10 + i}", "lan_address": f"192.0.2.{10 + i}"}
                 for i in range(size)],
    }


def _ring_latency(name: str = "ring-latency", size: int = 8, **overrides) -> plan.ConfigurationPlan:
    options = plan.configuration_options(name, plan.Options(**overrides))
    return plan.build_plan(Site.from_json(_site_document(size)), name, "run1", options=options, digest="d" * 16)


# -- the model -------------------------------------------------------------------------------------


def test_farthest_first_saves_the_relays_of_the_longest_path_on_the_ring_of_eight():
    layout = routes.Layout.parse("ring:8")
    rank_order = lm.orders_for(layout, 2, "rank")
    farthest = lm.orders_for(layout, 2, "ring-farthest")
    assert farthest == lm.orders_for(layout, 2, "farthest")
    parameters = lm.Parameters()
    by_rank = lm.allreduce(layout, 2, 8192, "oneshot", rank_order, parameters)
    by_distance = lm.allreduce(layout, 2, 8192, "oneshot", farthest, parameters)
    # 14 lanes at 0.3 us each; the lane posted last carries 4 KiB (0.17 us at 24 GB/s). In rank order it
    # crosses three relays (rank 3 to rank 7); posted farthest first it is a direct lane.
    last = 14 * 0.3 + 4096 / 24e3
    assert by_rank.us == pytest.approx(last + 3 * 0.75)
    assert by_distance.us == pytest.approx(last)
    critical = by_rank.phases[0]
    assert (critical.sender, critical.receiver, critical.relays) == (3, 7, 3)
    assert by_distance.phases[0].relays == 0
    # Two-shot: two phases of 512-byte stripes (a 1 KiB chunk per peer over two lanes).
    twoshot = lm.allreduce(layout, 2, 8192, "twoshot", farthest, parameters)
    assert len(twoshot.phases) == 2 and twoshot.us == pytest.approx(2 * (14 * 0.3 + 512 / 24e3))
    # Measured posting time per rank replaces the assumption; relays scale with relay_us.
    slower = lm.allreduce(layout, 2, 8192, "oneshot", rank_order, lm.Parameters(relay_us=0.9), post_us=[0.5] * 8)
    assert slower.us == pytest.approx(14 * 0.5 + 4096 / 24e3 + 3 * 0.9)


def test_bytes_dominate_large_one_shot_messages_and_windows_defer_long_stripes():
    layout = routes.Layout.parse("ring:8")
    farthest = lm.orders_for(layout, 2, "farthest")
    estimate = lm.allreduce(layout, 2, 64 << 10, "oneshot", farthest)
    # 7 x 64 KiB leave every rank: the host interface sets the time once stripes outlast posting.
    assert estimate.us >= 7 * (64 << 10) / 24e3
    # One-shot stripes of 48 KiB exceed the 32 KiB forward chunk: relayed lanes go after the direct ones.
    deferred = lm.allreduce(layout, 2, 96 << 10, "oneshot", farthest)
    windowless = lm.allreduce(layout, 2, 96 << 10, "oneshot", farthest, lm.Parameters(forward_window=0))
    assert deferred.phases[0].relays > 0 and windowless.phases[0].relays == 0
    with pytest.raises(ValueError, match="covers oneshot, twoshot"):
        lm.allreduce(layout, 2, 4096, "swing", farthest)
    with pytest.raises(ValueError, match="multiples of 16"):
        lm.allreduce(layout, 2, 4008, "oneshot", farthest)
    with pytest.raises(ValueError, match="every peer once"):
        lm.allreduce(layout, 2, 4096, "oneshot", [(1, 2)] + list(farthest[1:]))
    with pytest.raises(ValueError, match="per-rank value needs 8"):
        lm.allreduce(layout, 2, 4096, "oneshot", farthest, post_us=[0.3] * 4)
    with pytest.raises(ValueError, match="non-negative"):
        lm.Parameters(relay_us=-1)


# -- the configuration -----------------------------------------------------------------------------


def test_ring_latency_plans_both_algorithms_for_every_posting_order():
    built = _ring_latency()
    assert [group.positions for group in built.groups] == [tuple(range(8))]
    assert built.options.latency_sizes == plan.RING_LATENCY_SIZES
    assert built.options.post_orders == ("rank", "ring-farthest") and built.options.large_only
    assert _ring_latency("ring8-latency").options.latency_sizes == plan.RING_LATENCY_SIZES
    assert "posting orders ['rank', 'ring-farthest'] back to back" in plan.render_text(built)
    harness = worker.Harness.__new__(worker.Harness)
    harness.options = json.loads(json.dumps(built.to_json()["options"]))
    harness.result = {}
    harness.session = type("Session", (), {"set_post_order": lambda self, order: ()})()
    harness.latency_orders = harness._latency_orders()
    cases = harness.latency_cases()
    assert cases[:4] == [("all_reduce_oneshot", (2048,), 0, "rank"), ("all_reduce_oneshot", (2048,), 0, "ring-farthest"),
                         ("all_reduce_twoshot", (2048,), 0, "rank"), ("all_reduce_twoshot", (2048,), 0, "ring-farthest")]
    assert len(cases) == 4 * len(plan.RING_LATENCY_SIZES)
    # A session class without set_post_order runs the first order and lists the others.
    harness.session = object()
    assert harness._latency_orders() == ("rank",)
    assert harness.result["latency"]["skipped"] == [{"order": "ring-farthest",
                                                     "reason": "the session class has no set_post_order"}]


def test_ring_latency_refusals_and_cli(tmp_path, monkeypatch, capsys):
    with pytest.raises(plan.PlanError, match="transport-only"):
        plan.configuration_options("ring-latency", plan.Options(transport_only=True))
    with pytest.raises(plan.PlanError, match="--post-orders"):
        plan.configuration_options("ring-latency", plan.Options(session_env=(("SIRCL_POST_ORDER", "rank"),)))
    with pytest.raises(plan.PlanError, match="exceed the capacity"):
        _ring_latency(latency_sizes=(4 << 20,))
    with pytest.raises(plan.PlanError, match="distinct names"):
        plan.Options(post_orders=("rank", "nearest"))
    with pytest.raises(plan.PlanError, match="distinct names"):
        plan.Options(post_orders=("rank", "rank"))
    with pytest.raises(plan.PlanError, match="multiples of 16"):
        plan.Options(latency_sizes=(1000,))
    with pytest.raises(plan.PlanError, match="non-negative"):
        plan.Options(latency_post_us=-0.1)
    monkeypatch.setattr(remote, "ssh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("contacted a host")))
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-latency", "--json",
                     "--latency-sizes", "8192,28672", "--post-orders", "farthest,rank", "--relay-us", "0.9",
                     "--post-us", "0.25", "--write-us", "1.1"]) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert options["latency_sizes"] == [8192, 28672] and options["post_orders"] == ["farthest", "rank"]
    assert (options["latency_relay_us"], options["latency_post_us"], options["latency_write_us"]) == (0.9, 0.25, 1.1)


# -- the summary -----------------------------------------------------------------------------------


def _runs(order_times: dict, posting_us: float | None, rank: int, orders: dict) -> list[dict]:
    runs = []
    for (algorithm, nbytes, order), p50 in order_times.items():
        record = {"collective": f"all_reduce_{algorithm}", "mode": "graph", "shape": [nbytes // 2], "dim": 0,
                  "bytes": nbytes, "correct": True, "mismatched_calls": 0, "checked": 4, "counters": {},
                  "traffic": algorithm, "algorithm": algorithm, "times_us": [p50] * 3, "post_order": order,
                  "post_order_peers": list(orders[order][rank])}
        if posting_us is not None:
            record["posting"] = {"lanes": 14000, "ns": int(posting_us * 14e6), "us_per_lane": posting_us}
        runs.append(record)
    return runs


def test_the_summary_gives_the_crossover_the_order_comparison_and_the_model():
    built = _ring_latency(latency_sizes=(8192, 16384, 32768, 65536)).to_json()
    layout = routes.Layout.parse(built["groups"][0]["layout"])
    orders = {name: lm.orders_for(layout, 2, name) for name in ("rank", "ring-farthest")}
    times = {}
    for nbytes, oneshot, twoshot in ((8192, 24.0, 30.0), (16384, 26.0, 30.5), (32768, 33.0, 31.0),
                                     (65536, 52.0, 34.0)):
        for order, faster in (("rank", 0.0), ("ring-farthest", 2.0)):
            times[("oneshot", nbytes, order)] = oneshot - faster
            times[("twoshot", nbytes, order)] = twoshot - faster / 2
    results = [{"global_rank": rank, "group": 0, "host": f"spark{rank}", "error": None, "exit_code": 0,
                "runs": _runs(times, 0.4, rank, orders)} for rank in range(8)]
    merged = summary.merge(built, results)
    case = next(item for item in merged["cases"]
                if item["collective"] == "all_reduce_oneshot" and item["bytes"] == 8192 and item["post_order"] == "rank")
    expected = lm.allreduce(layout, 2, 8192, "oneshot", orders["rank"], lm.parameters_from(built["options"]),
                            post_us=0.4)
    assert case["latency_model_us"] == pytest.approx(expected.us, abs=1e-3) and case["posting_measured"]
    assert case["of_latency_model"] == pytest.approx(expected.us / 24.0, abs=1e-3)
    crossover = {item["post_order"]: item for item in merged["crossover"]}
    assert crossover["rank"]["first_twoshot_bytes"] == 32768 and crossover["rank"]["oneshot_max_bytes"] == 16384
    assert crossover["ring-farthest"]["oneshot_faster_above"] == []
    compared = [item for item in merged["post_orders"] if item["bytes"] == 8192 and item["algorithm"] == "oneshot"]
    assert compared[0]["p50_us"] == {"rank": 24.0, "ring-farthest": 22.0}
    text = summary.table(merged)
    assert "SIRCL_ONESHOT_MAX_BYTES=16384" in text and "[oneshot, posting order rank]" in text
    assert "[latency model" in text and "posting orders, group 0 graph oneshot 8192 bytes" in text
    # Without measured posting times the plan's assumption serves, and the row says so.
    unmeasured = summary.merge(built, [{**result, "runs": _runs(times, None, result["global_rank"], orders)}
                                       for result in results])
    row = next(item for item in unmeasured["cases"] if item["collective"] == "all_reduce_twoshot")
    assert not row["posting_measured"] and row["posting_us_per_lane"] == [0.3] * 8
    assert "(posting assumed)" in summary.table(unmeasured)


# -- calibration and the one-shot limit ------------------------------------------------------------


def test_the_calibrated_model_gives_the_measured_crossovers_on_both_layouts():
    ring = routes.Layout.parse("ring:8")
    path = routes.Layout.parse("path:0-3")
    # Measured on the ring of eight: one-shot faster up to 28 KiB farthest first and up to 36 KiB in rank order.
    assert lm.oneshot_limit(ring, 2, "farthest") == 28672 == lm.oneshot_limit(ring, 2, "ring-farthest")
    assert lm.oneshot_limit(ring, 2, "rank") == 36864
    # Measured on the path of four: one-shot faster at 64 KiB, two-shot at 96 KiB, in both orders.
    assert lm.oneshot_limit(path, 2, "farthest") == 73728 and lm.oneshot_limit(path, 2, "rank") == 81920
    assert lm.oneshot_limit(routes.Layout.parse("ring:2"), 2, "farthest") == 131072
    assert lm.oneshot_limit(ring, 2, "farthest", cap=8192) == 8192
    assert lm.oneshot_limit(ring, 2, "farthest", cap=1024) == 1024
    assert lm.calibration_for(4) is lm.PATH4_GRAPH is lm.calibration_for(2)
    assert lm.calibration_for(8) is lm.RING8_GRAPH is lm.calibration_for(6) is lm.calibration_for(16)
    for layout, order, nbytes, algorithm, measured in (
            (ring, "farthest", 8192, "oneshot", 21.6), (ring, "rank", 8192, "oneshot", 25.6),
            (ring, "farthest", 8192, "twoshot", 32.0), (ring, "rank", 8192, "twoshot", 36.0),
            (ring, "farthest", 131072, "oneshot", 85.3), (ring, "farthest", 131072, "twoshot", 50.2),
            (path, "farthest", 65536, "oneshot", 30.0), (path, "farthest", 65536, "twoshot", 31.8),
            (path, "farthest", 98304, "oneshot", 42.2), (path, "farthest", 98304, "twoshot", 40.0),
            (path, "rank", 131072, "oneshot", 52.4), (path, "rank", 131072, "twoshot", 44.1)):
        orders = lm.orders_for(layout, 2, order)
        assert lm.predicted_us(layout, 2, nbytes, algorithm, orders) == pytest.approx(measured, abs=2.5)
    # One-shot on the path is cable-bound at 128 KiB: four times the message on the middle cable.
    estimate = lm.allreduce(path, 2, 131072, "oneshot", lm.orders_for(path, 2, "farthest"))
    assert estimate.phases[0].cable_us == pytest.approx(4 * 131072 / 24e3)
    assert estimate.us >= 4 * 131072 / 24e3
    with pytest.raises(ValueError, match="multiple of 16"):
        lm.oneshot_limit(ring, 2, step=1000)


def test_fit_recovers_a_calibration_and_leaves_out_noisy_and_outlying_rows():
    ring = routes.Layout.parse("ring:8")
    truth = lm.Calibration(world=8, oneshot_us=15.0, oneshot_us_per_kib=0.3, oneshot_us_per_kib_above=0.15,
                           twoshot_us=22.0, twoshot_us_per_kib=0.12, twoshot_us_per_kib_above=0.2)
    cases = []
    for order in ("rank", "ring-farthest"):
        orders = lm.orders_for(ring, 2, order)
        for kib in (4, 8, 16, 32, 48, 64, 96, 128):
            for algorithm in ("oneshot", "twoshot"):
                median = lm.predicted_us(ring, 2, kib << 10, algorithm, orders, calibration=truth)
                cases.append({"collective": f"all_reduce_{algorithm}", "mode": "graph", "post_order": order,
                              "bytes": kib << 10, "slowest_p50_us": median, "slowest_p90_us": 1.05 * median})
    template = dict(cases[0])
    cases += [{**template, "bytes": 49152, "slowest_p50_us": 93.0, "slowest_p90_us": 99.0},   # outlier
              {**template, "slowest_p50_us": 40.0, "slowest_p90_us": 120.0},                  # noisy
              {**template, "mode": "eager", "slowest_p50_us": 50.0, "slowest_p90_us": 51.0}]
    fitted = lm.fit(cases, ring, 2)
    assert (fitted.oneshot_us, fitted.oneshot_us_per_kib, fitted.oneshot_us_per_kib_above) == pytest.approx(
        (15.0, 0.3, 0.15), abs=0.01)
    assert (fitted.twoshot_us, fitted.twoshot_us_per_kib, fitted.twoshot_us_per_kib_above) == pytest.approx(
        (22.0, 0.12, 0.2), abs=0.01)
    # Rows below the knee only: one slope for both sides.
    below = lm.fit(cases, ring, 2, max_bytes=65536)
    assert below.oneshot_us_per_kib == below.oneshot_us_per_kib_above == pytest.approx(0.3, abs=0.01)
    with pytest.raises(ValueError, match="at least two sizes"):
        lm.fit(cases[:1], ring, 2)


def test_path4_latency_runs_rank_and_farthest_on_sparks_0_to_3():
    built = _ring_latency("path4-latency")
    assert [group.positions for group in built.groups] == [(0, 1, 2, 3)]
    assert built.options.post_orders == ("rank", "farthest")
    assert built.options.latency_sizes == plan.RING_LATENCY_SIZES
    document = built.to_json()
    harness = worker.Harness.__new__(worker.Harness)
    harness.options = dict(document["options"], post_orders=["farthest", "rank"])
    harness.group, harness.me = document["groups"][0], document["ranks"][0]
    # A session class without run-time orders still builds farthest first: as the rank's explicit list.
    assert harness._construction_order() == "3,2,1" and harness._built_order_peers == (3, 2, 1)
    harness.options = dict(document["options"])
    assert harness._construction_order() == "rank" and harness._built_order_peers == ()
    with pytest.raises(plan.PlanError, match="--post-orders"):
        plan.configuration_options("path4-latency", plan.Options(session_env=(("SIRCL_POST_ORDER", "rank"),)))
