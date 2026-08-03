#!/usr/bin/env python3
"""Resume, adopt, and fail-closed verify the pinned 81-shard EXL3 model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm52-exl3-tr3-3.25bpw.json"
HEADROOM_BYTES = 16 * 1024**3
sys.path.insert(0, str(ROOT / "runtime/exl3"))

from model_manifest import (  # noqa: E402
    manifest_entries,
    sha256,
    stderr_progress,
    verify_model,
)


def contract() -> dict:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if recipe.get("recipe_id") != "glm52-exl3-tr3-3.25bpw":
        raise RuntimeError("wrong EXL3 recipe")
    return recipe["model"]


def verify(path: Path, model: dict) -> dict:
    return verify_model(path, model, progress=stderr_progress)


def require_capacity(path: Path, model: dict) -> None:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    existing = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0
    remaining = max(0, int(model["repository_bytes"]) - existing)
    required = remaining + HEADROOM_BYTES
    if free < required:
        raise RuntimeError(f"insufficient model disk space: need {required}, have {free}")


def download_manifested_snapshot(path: Path, model: dict, snapshot_download) -> None:
    """Download only receipt-owned bytes, never arbitrary repository sidecars."""
    common = {
        "repo_id": model["repository"],
        "revision": model["revision"],
        "local_dir": path,
        "token": os.environ.get("HF_TOKEN"),
    }
    snapshot_download(**common, allow_patterns=["MANIFEST.sha256"])
    manifest_path = path / "MANIFEST.sha256"
    observed = sha256(manifest_path)
    if observed != model["manifest_sha256"]:
        raise RuntimeError(
            "MANIFEST.sha256 hash mismatch: expected "
            f"{model['manifest_sha256']}, got {observed}"
        )
    owned = sorted(manifest_entries(manifest_path))
    snapshot_download(
        **common,
        allow_patterns=["MANIFEST.sha256", *owned],
    )


def download(path: Path, model: dict) -> None:
    try:
        report = verify(path, model)
    except (OSError, ValueError, RuntimeError):
        pass
    else:
        print("PASS: exact EXL3 model already exists; download skipped")
        print(json.dumps(report, sort_keys=True))
        return
    require_capacity(path, model)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_HOME", str(path.parent / ".huggingface"))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install it in the rank-0 Python environment") from exc
    path.mkdir(parents=True, exist_ok=True)
    download_manifested_snapshot(path, model, snapshot_download)
    print(json.dumps(verify(path, model), sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        model = contract()
        if args.action == "download":
            download(args.model_path.resolve(), model)
        else:
            print(json.dumps(verify(args.model_path.resolve(), model), sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
