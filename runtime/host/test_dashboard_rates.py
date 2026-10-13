"""Both viewers consume the same counter cases; no serving process is needed."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from runtime.host.dashboard_rates import ViewerRates
from runtime.host.test_dashboard import PAIRS, collector, paths_of, run_for


def document(at, prompt, gen, *, run=1, plugin=False, quiet=False, **values):
    counters = dict(prompt=prompt, gen=gen, accepted=gen, drafts=gen / 2,
                    draft_tokens=gen * 2, ttft_sum=gen / 10, ttft_count=gen,
                    running=1, waiting=0, kv=.25, steps=gen / 3, step_seconds=gen / 30,
                    pc_hits=gen, pc_queries=gen * 2, **values)
    m = dict(id="deployment", state="serving", live={},
             counters=dict(source="metrics", at=at, run_id=run, values=counters))
    if plugin:
        m["plugin_counters"] = dict(schema="sparkring-stats/v1", at=at, monotonic=at - 100,
                                    run_id=str(run), step_timing=True, **counters)
    return dict(generated_at=at, groups=[dict(members=[m])], collector=dict(quiet=quiet))


def member(doc):
    return doc["groups"][0]["members"][0]


CASES = [document(1000, 10, 10), document(1005, 510, 160), document(1010, 510, 310),
         document(1010, 510, 310), document(1015, 1, 3), document(1020, 1000, 1000, run=2),
         document(1025, 1100, 1030, run=2), document(900, 1200, 1100, run=2),
         document(1100, 10, 10, plugin=True), document(1160, 6010, 1810, plugin=True),
         document(1220, 6010, 3610, plugin=True), document(1225, 6010, 3700, quiet=True),
         document(1230, 6010, 3800)]


def test_prefill_idle_reset_restart_backward_clock_and_source_change():
    viewer = ViewerRates()
    got = [copy.deepcopy(member(viewer.apply(copy.deepcopy(d)))) for d in CASES]
    assert [m["live"]["prefill_tps"] for m in got] == [None, 100, None, None, None, None, 20,
                                                          None, None, 100, None, None, None]
    assert got[1]["live"]["output_tps"] == 30
    assert len(got[3]["history"]) == 3  # duplicate read has no fabricated point
    assert got[9]["live"]["steps_ps"] == 10
    assert got[9]["live"]["acceptance"] == .5
    assert got[9]["live"]["accept_length_5m"] == 3


def test_stale_plugin_and_previous_run_use_metrics():
    viewer = ViewerRates()
    doc = document(1000, 10, 10, plugin=True)
    viewer.apply(doc)
    doc = document(1010, 30, 310, plugin=True)
    viewer.apply(doc)
    assert member(doc)["live"]["source"] == "plugin"
    later = document(1200, 100, 500)
    member(later)["plugin_counters"] = member(doc)["plugin_counters"]
    viewer.apply(later)
    assert member(later)["live"]["source"] == "metrics"
    assert member(later)["live"]["output_tps"] is None
    member(later)["counters"]["run_id"] = 1190
    member(later)["plugin_counters"]["at"] = 1180
    viewer.apply(later)
    assert member(later)["live"]["source"] == "metrics"


def test_browser_and_terminal_match_all_counter_cases():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser counter tests")
    code = Path(__file__).with_name("dashboard_rates.js").read_text()
    code += "\nconst v = new ViewerRates(); console.log(JSON.stringify(" + json.dumps(CASES) + ".map(d => JSON.parse(JSON.stringify(v.apply(d))))));"
    actual = json.loads(subprocess.check_output([node, "-e", code], text=True))
    viewer = ViewerRates()
    expected = [copy.deepcopy(viewer.apply(copy.deepcopy(d))) for d in CASES]
    assert actual == expected


def test_plugin_presence_changes_no_upstream_request_or_quiet_cadence(tmp_path):
    calls = []
    for has_plugin in (False, True):
        directory = tmp_path / str(has_plugin)
        directory.mkdir()
        found, api, clock = collector(directory, rows=PAIRS[:1])
        found.reload_sources()
        next(iter(found.models.values())).row["deployment_id"] = "one"
        if has_plugin:
            found.node_status = {"telemetry": {"one": member(document(clock.now, 1, 1, plugin=True))["plugin_counters"]}}
        run_for(found, clock, 600, viewer=True)
        snapshot = found.snapshot()
        assert bool(member(snapshot)["plugin_counters"]) == has_plugin
        assert "output_tps" not in member(snapshot)["live"]
        assert member(snapshot)["counters"]["at"] <= clock.now
        assert not hasattr(next(iter(found.models.values())), "samples")
        calls.append(paths_of(api))
        found.quiet_path.write_text("test")
        api.requests.clear()
        run_for(found, clock, 30, viewer=True)
        assert api.requests == []
    assert calls[0] == calls[1]


def test_history_bounded_and_departed_deployments_forgotten():
    viewer = ViewerRates()
    for at in range(1000, 24000, 5):
        doc = viewer.apply(document(at, at, at))
    assert len(member(doc)["history"]) <= 4321
    assert len(viewer.members["deployment"]["samples"]) <= 100
    viewer.apply(dict(generated_at=25000, groups=[]))
    assert not viewer.members


def test_idle_gauge_and_malformed_plugin_never_show_prefill():
    viewer = ViewerRates()
    viewer.apply(document(1000, 10, 10))
    doc = document(1005, 510, 160)
    member(doc)["counters"]["values"]["running"] = 0
    viewer.apply(doc)
    assert member(doc)["live"]["prefill_tps"] is None
    bad = document(1010, 710, 310, plugin=True)
    member(bad)["plugin_counters"]["prompt"] = "broken"
    viewer.apply(bad)
    assert member(bad)["live"]["source"] == "metrics"


def test_plugin_restart_with_larger_counters_still_needs_baseline():
    viewer = ViewerRates()
    viewer.apply(document(1000, 100, 100, plugin=True))
    doc = viewer.apply(document(1005, 600, 250, plugin=True))
    assert member(doc)["live"]["prefill_tps"] == 100
    doc = viewer.apply(document(1010, 1000, 1000, plugin=True, run=2))
    assert member(doc)["live"]["prefill_tps"] is None
    assert len(member(doc)["history"]) == 1


def test_metrics_acceptance_has_a_verified_token_denominator():
    from runtime.host.dashboard import parse_metrics
    counters = parse_metrics("vllm:spec_decode_num_draft_tokens_total 6\nvllm:spec_decode_num_accepted_tokens_total 3\n")
    assert counters["draft_tokens"] == 6 and counters["accepted"] == 3
