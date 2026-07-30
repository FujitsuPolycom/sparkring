"""Offline behavior tests for the public runtime capability gate."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("public-capability-gate.py")
SPEC = importlib.util.spec_from_file_location("public_capability_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gate_rejects_runtime_without_configured_parser_flags(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "capabilities.py").write_text(
        "B12X_MLA_SPARSE = 'nvfp4_ds_mla'\nglm47 = True\n",
        encoding="utf-8",
    )
    help_text = " ".join(
        (
            "--attention-backend",
            "--decode-context-parallel-size",
            "--dcp-comm-backend",
            "--kv-cache-dtype",
            "--speculative-config",
        )
    )
    monkeypatch.setattr(MODULE, "vllm_root", lambda: tmp_path)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, help_text, ""
        ),
    )

    report = MODULE.evaluate("vllm")

    assert "CLI flag absent: --enable-auto-tool-choice" in report["failures"]
    assert "CLI flag absent: --reasoning-parser" in report["failures"]
    assert "CLI flag absent: --tool-call-parser" in report["failures"]
