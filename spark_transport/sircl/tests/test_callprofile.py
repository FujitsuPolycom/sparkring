"""The eager call profile (``sparkring_sircl.callprofile``): stage medians from the marks of each call, the
caller's gaps, device times, the newest calls kept, the summary file, and the harness's profile lines."""

from __future__ import annotations

import json

from sparkring_sircl import callprofile
from sparkring_sircl.ring import summary


def _call(profile, op, base, steps, detail="oneshot", nbytes=8192):
    call = profile.begin(op)
    call.marks = {"entry": base}
    clock = base
    for name, step in zip(callprofile.MARKS[1:-1], steps[:-1]):
        clock += step
        call.marks[name] = clock
    profile.end(call, detail, nbytes)
    call.marks["exit"] = clock + steps[-1]
    return call


def test_stage_medians_gaps_and_device_times(tmp_path):
    profile = callprofile.CallProfile(2, path=str(tmp_path / "profile"), rank=3)
    first = _call(profile, "all_reduce", 0, [2000, 5000, 1000, 20000, 1000, 2000])
    first.events = ("a", "b")
    second = _call(profile, "all_reduce", 100000, [4000, 5000, 1000, 22000, 1000, 2000])
    second.events = ("a", "b")
    second.gap_ns = second.marks["entry"] - first.marks["exit"]
    row = profile.summary(lambda start, end: 0.025)["rows"][0]
    assert row["op"] == "all_reduce" and row["calls"] == 2 and row["path"] == "oneshot"
    assert row["checks"] == 3.0 and row["launch"] == 21.0 and row["total"] == 33.0 and row["gpu"] == 25.0
    assert row["gap"] == 69.0
    written = json.loads((tmp_path / "profile.rank3.json").read_text())
    assert written["rank"] == 3 and written["recorded"] == 2
    _call(profile, "all_gather", 300000, [1000] * 6, detail="direct", nbytes=4096)
    assert profile.summary()["kept"] == 2
    assert [row["op"] for row in profile.summary()["rows"]] == ["all_gather", "all_reduce"]
    lines = callprofile.render(profile.summary())
    assert lines[0].startswith("all_gather 4096 bytes (direct, 1 calls): checks 1, dispatch 1")
    profile.reset()
    assert profile.summary()["rows"] == []


def test_profile_lines_of_a_result():
    rows = [{"op": "all_reduce", "path": "oneshot", "bytes": 8192, "calls": 40, "checks": 2.0, "launch": 20.0,
             "total": 33.0}]
    slow = [dict(rows[0], total=41.0)]
    result = {"cases": [{"group": 0, "collective": "all_reduce", "bytes": 8192, "slowest_p50_us": 50.4,
                         "call_profile": {"0": rows, "1": slow},
                         "adapter_profile": {"0": {"plan_us": 3.5, "execute_us": 40.0, "methods": ["direct"]}}}]}
    lines = summary.profile_lines(result)
    assert lines[0] == "eager profile, group 0 all_reduce 8192 bytes (event p50 50.4 us):"
    assert lines[1].startswith("  rank 0: all_reduce 8192 bytes (oneshot, 40 calls): checks 2, launch 20, total 33")
    assert lines[2].startswith("  rank 1:") and "total 41" in lines[2]
    assert lines[3] == "  rank 0 adapter: plan 3.5 us, execute 40 us (direct)"
