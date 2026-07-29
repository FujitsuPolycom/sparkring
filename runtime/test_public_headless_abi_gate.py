from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("public-headless-abi-gate.py")
SPEC = importlib.util.spec_from_file_location("public_headless_abi_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PINNED_VLLM = Path(
    r"C:\Users\Cody\AppData\Local\Temp\sparkring-vllm-fcc6141\vllm"
)


def test_unmodified_upstream_is_not_the_post_overlay_target() -> None:
    if not PINNED_VLLM.is_dir():
        return
    report = MODULE.audit(PINNED_VLLM)
    assert report["ok"] is False
    assert all("source hash mismatch" in failure
               for failure in report["failures"])


def test_source_mutation_fails_closed(tmp_path: Path) -> None:
    for relative in MODULE.EXPECTED_SHA256:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("# deliberately not pinned\n", encoding="utf-8")
    report = MODULE.audit(tmp_path)
    assert report["ok"] is False
    assert len(report["failures"]) == 2
