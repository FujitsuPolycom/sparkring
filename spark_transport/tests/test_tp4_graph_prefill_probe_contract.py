from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
PROBE = (ROOT / "app" / "tp4_graph_q1_probe.cu").read_text(encoding="utf-8")
RUNNER = (ROOT / "scripts" / "run_tp4_graph_q1_probe.ps1").read_text(
    encoding="utf-8"
)


def test_probe_has_explicit_q512_capacity_and_prefill_shapes() -> None:
    assert "kTp4GraphAllreduceMaximumQ" in PROBE
    assert "--maximum-q" in PROBE
    assert "6U * 1024U * 1024U" in PROBE
    for q in (48, 72, 144, 512):
        assert f"{q}U" in PROBE
        assert f"q{q}_nodes=" in PROBE
    assert "validate_active_output" in PROBE
    assert "mismatched_elements=" in PROBE
    assert "monotonic_sequences=" in PROBE
    assert "post_replay_capture_rejected=" in PROBE


def test_runner_forwards_and_attests_q512_without_changing_default() -> None:
    assert "[ValidateRange(6, 512)]" in RUNNER
    assert "[int]$MaximumQ = 6" in RUNNER
    assert '"--maximum-q $MaximumQ"' in RUNNER
    assert "maximum_q=$MaximumQ" in RUNNER
    for q in (48, 72, 144, 512):
        assert f"q{q}_nodes=" in RUNNER
    assert "mismatched_elements=0" in RUNNER
    assert "monotonic_sequences=true" in RUNNER
    assert "post_replay_capture_rejected=true" in RUNNER
