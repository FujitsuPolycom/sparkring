"""Behavioral tests for the public runtime requirements closure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


RUNTIME = Path(__file__).resolve().parent
SCRIPT = RUNTIME / "prepare-public-requirements.py"
VERIFY_SCRIPT = RUNTIME / "verify-frozen-packages.py"


def _run(freeze: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--freeze",
            str(freeze),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_repository_freeze_closes_without_machine_local_urls(tmp_path):
    output = tmp_path / "public-requirements.txt"

    result = _run(RUNTIME / "pip-freeze.txt", output)

    assert result.returncode == 0, result.stderr
    requirements = output.read_text(encoding="utf-8")
    assert "file://" not in requirements
    assert "torch==2.12.0+cu132" in requirements
    assert "excluded 6 source-provided distribution(s)" in result.stdout


def test_final_package_verifier_rejects_resolver_drift(tmp_path):
    freeze = tmp_path / "freeze.txt"
    freeze.write_text("alpha==1.2.3\nvllm @ file:///opt/vllm\n", encoding="utf-8")
    installed = tmp_path / "installed.json"
    installed.write_text('{"alpha": "9.9.9", "vllm": "0.11.2"}', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--freeze",
            str(freeze),
            "--installed-json",
            str(installed),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "alpha: installed 9.9.9 != frozen 1.2.3" in result.stderr
