from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("generate-manifest.py")
SPEC = importlib.util.spec_from_file_location("generate_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_patch_log_accepts_patch_and_addition_rows(tmp_path: Path) -> None:
    log = tmp_path / "apply.jsonl"
    records = [
        {
            "patch": "change.patch",
            "target": "vllm/change.py",
            "preimage_sha256": "1" * 64,
            "postimage_sha256": "2" * 64,
            "action": "rebased",
        },
        {
            "addition": "backend.py",
            "target": "vllm/backend.py",
            "sha256": "3" * 64,
            "action": "inherited",
        },
    ]
    log.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    files = MODULE.load_patch_log(str(log))

    assert files["vllm/change.py"] == {
        "preimage": "1" * 64,
        "postimage": "2" * 64,
    }
    assert files["vllm/backend.py"] == {
        "preimage": None,
        "postimage": "3" * 64,
    }


def test_patch_log_rejects_row_without_postimage(tmp_path: Path) -> None:
    log = tmp_path / "apply.jsonl"
    log.write_text(
        json.dumps({"target": "vllm/broken.py"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no postimage hash"):
        MODULE.load_patch_log(str(log))
