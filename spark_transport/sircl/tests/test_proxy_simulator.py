"""The proxy simulator: the real native layer, real progress threads, simulated verbs.

One run executes every case of ``sparkring_sircl/testing/sim/proxy_sim.c``:
3,000 one-shot ops per rank with random sizes and mixed one-shot, two-shot,
described (Swing) and scatter ops for groups of 2, 3, 4, 6 and 8 on cycles
and paths with one and two lanes; a sequence wrap; random progress-thread
pauses; missed doorbells; a later phase released before the earlier one was
posted; malformed op words, phase doorbells and descriptors; an injected
failed completion; the three posting orders; two concurrent sessions; a
route map whose lanes do not pair; and forward windows whose completions return
long after their writes landed, freed by proofs of delivery, also across the
sequence wrap.
"""

from __future__ import annotations

import subprocess

from vectors import load


def _run(binary, *cases):
    process = subprocess.run([str(binary), *cases], capture_output=True, text=True, timeout=900)
    return process.returncode, process.stdout


def test_every_simulator_case_passes(simulator_binary):
    code, output = _run(simulator_binary)
    failures = [line for line in output.splitlines() if line.startswith("FAIL")]
    assert code == 0 and not failures, "\n".join(failures) or output[-2000:]
    passed = [line for line in output.splitlines() if line.startswith("PASS")]
    assert len(passed) >= 57
    for name in ("oneshot-3000/cycle8/lanes2", "mixed-ops/path8/lanes2", "pauses/cycle8/lanes2",
                 "missed-doorbell/cycle4/lanes2", "phase-order/cycle4/lanes2", "injected-failure",
                 "post-order/ring-farthest", "two-sessions", "unpaired-lanes", "forward-windows/path4",
                 "wait-regimes/path4", "chain/path4/lanes2", "chain/cycle8/lanes2", "chain-pauses/path4",
                 "chain-trace/path4", "chain/refusals", "forward-proof/cycle8/lanes2", "forward-proof/path4/lanes2",
                 "forward-proof-wrap/cycle8/lanes2", "forward-proof-wrap/path4/lanes2"):
        assert any(line.split()[1].startswith(name) for line in passed), name


def test_simulator_swing_schedule_matches_the_vectors(simulator_binary):
    code, output = _run(simulator_binary, "swing-schedule")
    assert code == 0
    got = {}
    for line in output.splitlines():
        if line.startswith("SWING"):
            parts = line.split()
            world, rank = int(parts[1].split("=")[1]), int(parts[2].split("=")[1])
            got[(world, rank)] = [[int(v) for v in item.split(",")] for item in parts[3:]]
    for key, entry in load("numeric.json")["swing"].items():
        world = int(key.split("=")[1])
        for rank, phases in entry["phases"].items():
            assert got[(world, int(rank))] == phases
