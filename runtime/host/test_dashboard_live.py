"""Viewer-driven collection and short-window regression cases."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from runtime.host import dashboard
from runtime.host.dashboard_rates import ViewerRates
from runtime.host.test_dashboard import PAIRS, collector, paths_of, run_for
from runtime.host.test_dashboard_rates import document, member


def test_no_metrics_without_viewer_and_two_second_reads_with_viewer(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    run_for(found, clock, 30)
    assert api.requests == []
    run_for(found, clock, 12, viewer=True)
    assert paths_of(api).count("/metrics") == 6
    clock.now += dashboard.VIEWER_SECONDS + 1
    api.requests.clear()
    run_for(found, clock, 30)
    assert api.requests == []


def live_cases(plugin):
    cases = [document(1000 + i * 2, 10, g, plugin=plugin)
             for i, g in enumerate([0, 20, 40, 60, 120, 180, 240])]
    idle = document(1014, 10, 240, plugin=plugin)
    member(idle)["counters"]["values"].update(running=0, waiting=0)
    if plugin:
        member(idle)["plugin_counters"].update(running=0, waiting=0)
    cases.append(idle)
    stopped = copy.deepcopy(idle)
    stopped["generated_at"] = 1016
    member(stopped)["state"] = "stopped"
    cases.append(stopped)
    cases.extend([document(1018, 10, 10000, run=2, plugin=plugin),
                  document(1020, 10, 2, run=2, plugin=plugin)])
    return cases


@pytest.mark.parametrize("plugin", [False, True])
def test_short_window_idle_stop_restart_and_reset_in_both_viewers(plugin):
    cases = live_cases(plugin)
    viewer = ViewerRates()
    expected = [copy.deepcopy(viewer.apply(copy.deepcopy(d))) for d in cases]
    live = [member(d)["live"] for d in expected]
    assert live[6]["output_tps"] == 30
    assert live[7]["output_tps"] == 0
    assert live[7]["steps_ps"] == (0 if plugin else None)
    assert live[8]["output_tps"] is None
    assert live[9]["output_tps"] is None
    assert live[10]["output_tps"] is None
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required")
    code = Path(dashboard.__file__).with_name("dashboard_rates.js").read_text()
    code += "\nconst v=new ViewerRates(); console.log(JSON.stringify(" + json.dumps(cases) + ".map(d=>JSON.parse(JSON.stringify(v.apply(d))))));"
    assert json.loads(subprocess.check_output([node, "-e", code], text=True)) == expected


@pytest.mark.parametrize("tp,dcp", [(2, 1), (1, 2), (4, 1)])
def test_group_rates_are_not_multiplied_by_parallel_ranks(tp, dcp):
    cases = []
    for at, gen, steps, accepted in [(1000, 100, 10, 20), (1002, 160, 30, 50)]:
        doc = document(at, 100, gen, plugin=True)
        m = member(doc)
        m["layout"] = {"count": tp * dcp, "tp": tp}
        m["settings"] = {"decode_context_parallel_size": dcp}
        m["sparks"] = [{"position": rank} for rank in range(tp * dcp)]
        m["plugin_counters"].update(steps=steps, step_accepted=accepted)
        cases.append(doc)
    viewer = ViewerRates()
    expected = [copy.deepcopy(viewer.apply(copy.deepcopy(d))) for d in cases]
    result = member(expected[-1])["live"]
    assert result["output_tps"] == 30 and result["steps_ps"] == 10
    assert result["accepted_per_step"] == 1.5
    code = Path(dashboard.__file__).with_name("dashboard_rates.js").read_text()
    code += "\nconst v=new ViewerRates(); console.log(JSON.stringify(" + json.dumps(cases) + ".map(d=>JSON.parse(JSON.stringify(v.apply(d))))));"
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required")
    assert json.loads(subprocess.check_output([node, "-e", code], text=True)) == expected


def test_stopped_legacy_payload_cannot_retain_derived_rates():
    value = {"id": "one", "state": "stopped", "live": {"output_tps": 35, "steps_ps": 12}}
    ViewerRates().member(value, 1000)
    assert value["live"]["output_tps"] is None and value["live"]["steps_ps"] is None



def test_group_metrics_requests_are_concurrent_and_multiple_viewers_share_samples(tmp_path):
    import threading
    from urllib.parse import urlparse
    found, api, clock = collector(tmp_path)
    barrier = threading.Barrier(len(PAIRS))
    original = api.open
    def concurrent(request, timeout=None):
        if urlparse(request.full_url).path == "/metrics":
            barrier.wait(timeout=3)
        return original(request, timeout=timeout)
    api.open = concurrent
    found.mark_viewed()
    found.mark_viewed()
    found.cycle()
    assert paths_of(api).count("/metrics") == len(PAIRS)
    clock.now += 1
    found.mark_viewed()
    found.cycle()
    assert paths_of(api).count("/metrics") == len(PAIRS)



def test_slow_drip_response_cannot_hold_sample_forever(monkeypatch):
    from runtime.host.test_dashboard import Response
    clock = [0.0]
    class Drip(Response):
        def read1(self, count):
            clock[0] += .6
            return b"x"
    class API:
        def open(self, request, timeout=None):
            return Drip()
    client = dashboard.ReadOnlyClient(opener=API())
    client.allow({"192.0.2.1"})
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: clock[0])
    assert client.get("192.0.2.1", 8000, "/metrics").failure == "timeout"
    assert clock[0] < 3



def test_archive_thinning_preserves_reset_between_stored_samples(tmp_path):
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    found.mark_viewed()
    found.cycle()
    model = next(iter(found.models.values()))
    found.history.clear()
    start = clock.now
    for offset, gen in [(0, 50), (2, 100), (4, 70), (6, 100), (10, 140)]:
        model.observe_metrics({"started": 1, "gen": gen, "running": 1}, start + offset)
        found.remember(model, start + offset)
    rows = list(found.history[model.id])
    assert [row["counters"]["at"] - start for row in rows] == [0, 4, 10]
    assert rows[1]["boundary"]
    # The two-second observations are always available in the live envelope.
    assert found.member(model, start + 10, 0)["counters"]["values"]["gen"] == 140
