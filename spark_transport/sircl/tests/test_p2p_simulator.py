"""The point-to-point simulator: the real native layer, real progress threads, simulated verbs.

One run executes every case of ``sparkring_sircl/testing/sim/p2p_sim.c``:
random traffic between every pair of groups of 2, 3, 4 and 8 ranks on cycles
and paths with one and two lanes (messages of one to several items, every
size class, exact payload checks); two and 32 slots; channels between adjacent
ranks only and between ranks two apart; relayed lanes paced by forward windows
on a simulated wire whose completions return long after delivery (no queue
pair holds more than its window unacknowledged, and proofs of delivery free
it); item counters that wrap; a slow receiver that holds every sender at its
slot credit without overwriting a slot; a receive of another size, which stops
both ranks' channels and names the rank that found it; an injected failed
write; and native refusals.
"""

from __future__ import annotations

import subprocess

import pytest


@pytest.fixture(scope="session")
def p2p_simulator():
    from conftest import WORK, _require_compiler

    _require_compiler()
    from sparkring_sircl.testing import p2p_build

    return p2p_build.build_simulator(WORK / ".build" / "sim")


def _run(binary, *cases):
    process = subprocess.run([str(binary), *cases], capture_output=True, text=True, timeout=900)
    return process.returncode, process.stdout


def test_every_point_to_point_simulator_case_passes(p2p_simulator):
    code, output = _run(p2p_simulator)
    failures = [line for line in output.splitlines() if line.startswith("FAIL")]
    assert code == 0 and not failures, "\n".join(failures) or output[-2000:]
    passed = [line.split()[1].rstrip(":") for line in output.splitlines() if line.startswith("PASS")]
    assert len(passed) >= 30
    for name in ("pairs/cycle8/lanes2", "pairs/path8/lanes2", "pairs/cycle2/lanes1", "pairs/path3/lanes2",
                 "slots2/path4/lanes2", "slots32/cycle3/lanes2", "adjacent-only/cycle8/lanes2",
                 "two-apart/cycle8/lanes2", "relayed-windows/cycle8/lanes2", "relayed-windows/path4/lanes2",
                 "relayed-windows/path4/lanes1", "wrap/cycle4/lanes2", "wrap/path3/lanes1",
                 "back-pressure/path4/lanes2", "back-pressure/cycle8/lanes2", "size-mismatch/path4/lanes2",
                 "injected-failure/cycle4/lanes2", "refusals"):
        assert any(case.startswith(name) for case in passed), name


def test_one_case_runs_alone(p2p_simulator):
    code, output = _run(p2p_simulator, "size-mismatch/path4/lanes2")
    assert code == 0
    assert [line.split()[1].rstrip(":") for line in output.splitlines() if line.startswith("PASS")] == [
        "size-mismatch/path4/lanes2"]
