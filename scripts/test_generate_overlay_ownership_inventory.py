"""Offline freshness and determinism tests for the overlay inventory."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_overlay_ownership_inventory.py"


def run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_committed_inventory_is_fresh() -> None:
    result = run_generator("--check")
    assert result.returncode == 0, result.stderr
    assert "is fresh" in result.stdout


def test_generation_is_deterministic_and_check_detects_drift(tmp_path: Path) -> None:
    output = tmp_path / "inventory.json"
    first = run_generator("--output", str(output))
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()

    second = run_generator("--output", str(output))
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes

    fresh = run_generator("--output", str(output), "--check")
    assert fresh.returncode == 0, fresh.stderr

    output.write_bytes(first_bytes + b"\n")
    stale = run_generator("--output", str(output), "--check")
    assert stale.returncode != 0
    assert "inventory is stale" in stale.stderr
