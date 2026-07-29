from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("apply-patches.py")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run(patches: Path, site: Path, log: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--patches-root",
            str(patches),
            "--site-packages",
            str(site),
            "--log",
            str(log),
            "--fail-closed",
        ],
        capture_output=True,
        text=True,
    )


def test_hash_pinned_addition_is_installed(tmp_path: Path) -> None:
    component = tmp_path / "patches" / "reference"
    source = component / "added" / "new_backend.py"
    source.parent.mkdir(parents=True)
    payload = b"CAPABILITY = 'sm121-sparse-mla'\n"
    source.write_bytes(payload)
    (component / "additions.json").write_text(
        json.dumps(
            {
                "new_backend.py": {
                    "target_path": "vllm/new_backend.py",
                    "sha256": _sha256(payload),
                }
            }
        ),
        encoding="utf-8",
    )
    site = tmp_path / "site"
    site.mkdir()
    log = tmp_path / "apply.jsonl"

    result = _run(tmp_path / "patches", site, log)

    assert result.returncode == 0, result.stderr
    assert (site / "vllm" / "new_backend.py").read_bytes() == payload
    assert json.loads(log.read_text(encoding="utf-8"))["addition"] == (
        "new_backend.py"
    )


def test_addition_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    component = tmp_path / "patches" / "reference"
    source = component / "added" / "new_backend.py"
    source.parent.mkdir(parents=True)
    source.write_text("changed\n", encoding="utf-8")
    (component / "additions.json").write_text(
        json.dumps(
            {
                "new_backend.py": {
                    "target_path": "vllm/new_backend.py",
                    "sha256": "0" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    site = tmp_path / "site"
    site.mkdir()

    result = _run(tmp_path / "patches", site, tmp_path / "apply.jsonl")

    assert result.returncode != 0
    assert "addition hash mismatch" in result.stderr
    assert not (site / "vllm" / "new_backend.py").exists()


def test_addition_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    component = tmp_path / "patches" / "reference"
    source = component / "added" / "new_backend.py"
    source.parent.mkdir(parents=True)
    payload = b"new\n"
    source.write_bytes(payload)
    (component / "additions.json").write_text(
        json.dumps(
            {
                "new_backend.py": {
                    "target_path": "vllm/new_backend.py",
                    "sha256": _sha256(payload),
                }
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "site" / "vllm" / "new_backend.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original\n")

    result = _run(
        tmp_path / "patches",
        tmp_path / "site",
        tmp_path / "apply.jsonl",
    )

    assert result.returncode != 0
    assert "refuses to overwrite" in result.stderr
    assert target.read_bytes() == b"original\n"
