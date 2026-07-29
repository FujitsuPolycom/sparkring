from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("apply-patches.py")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run(
    patches: Path,
    site: Path,
    log: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
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
            *extra,
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


def test_compatible_base_accepts_identical_inherited_addition(
    tmp_path: Path,
) -> None:
    component = tmp_path / "patches" / "reference"
    source = component / "added" / "backend.py"
    source.parent.mkdir(parents=True)
    payload = b"VALUE = 1\n"
    source.write_bytes(payload)
    (component / "additions.json").write_text(
        json.dumps(
            {
                "backend.py": {
                    "target_path": "vllm/backend.py",
                    "sha256": _sha256(payload),
                }
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "site" / "vllm" / "backend.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    log = tmp_path / "apply.jsonl"

    result = _run(
        tmp_path / "patches",
        tmp_path / "site",
        log,
        "--compatible-base",
    )

    assert result.returncode == 0, result.stderr
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["action"] == "inherited"


def test_compatible_base_uses_only_exact_supplement(tmp_path: Path) -> None:
    component = tmp_path / "patches" / "reference"
    component.mkdir(parents=True)
    original = b"alpha\n"
    drifted = b"renamed alpha\nbase feature\n"
    target = tmp_path / "site" / "vllm" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(drifted)
    patch_name = "module.patch"
    (component / patch_name).write_bytes(
        b"--- a/vllm/module.py\n"
        b"+++ b/vllm/module.py\n"
        b"@@ -1 +1,2 @@\n"
        b" alpha\n"
        b"+recovered feature\n"
    )
    (component / "preimages.json").write_text(
        json.dumps(
            {
                patch_name: {
                    "target_path": "vllm/module.py",
                    "preimage_sha256": _sha256(original),
                }
            }
        ),
        encoding="utf-8",
    )
    supplements = tmp_path / "supplements"
    supplements.mkdir()
    (supplements / patch_name).write_bytes(
        b"--- a/vllm/module.py\n"
        b"+++ b/vllm/module.py\n"
        b"@@ -1,2 +1,3 @@\n"
        b" renamed alpha\n"
        b" base feature\n"
        b"+recovered feature\n"
    )
    (supplements / "preimages.json").write_text(
        json.dumps(
            {
                patch_name: {
                    "target_path": "vllm/module.py",
                    "preimage_sha256": _sha256(drifted),
                }
            }
        ),
        encoding="utf-8",
    )
    log = tmp_path / "apply.jsonl"

    result = _run(
        tmp_path / "patches",
        tmp_path / "site",
        log,
        "--compatible-base",
        "--supplemental-root",
        str(supplements),
    )

    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8").splitlines() == [
        "renamed alpha",
        "base feature",
        "recovered feature",
    ]
    assert json.loads(log.read_text(encoding="utf-8"))["action"] == "supplemental"
