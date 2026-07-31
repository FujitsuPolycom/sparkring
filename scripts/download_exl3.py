#!/usr/bin/env python3
"""Resume, adopt, and fail-closed verify the pinned 81-shard EXL3 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm52-exl3-tr3-3.25bpw.json"
HEADROOM_BYTES = 16 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract() -> dict:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if recipe.get("recipe_id") != "glm52-exl3-tr3-3.25bpw":
        raise RuntimeError("wrong EXL3 recipe")
    return recipe["model"]


def manifest_entries(path: Path) -> dict[str, str]:
    entries = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError(f"invalid MANIFEST.sha256 line {number}")
        digest, name = fields
        name = name.strip().removeprefix("./")
        if name in entries:
            raise RuntimeError(f"duplicate MANIFEST.sha256 entry: {name}")
        entries[name] = digest
    return entries


def verify(path: Path, model: dict) -> dict:
    metadata = {
        "config.json": model["config_sha256"],
        "model.safetensors.index.json": model["index_sha256"],
        "tier_bitmap.json": model["tier_bitmap_sha256"],
        "MANIFEST.sha256": model["manifest_sha256"],
    }
    for name, expected in metadata.items():
        target = path / name
        if not target.is_file():
            raise RuntimeError(f"model metadata is missing: {target}")
        observed = sha256(target)
        if observed != expected:
            raise RuntimeError(f"{name} hash mismatch: expected {expected}, got {observed}")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if len(shards) != model["shard_count"]:
        raise RuntimeError(f"expected {model['shard_count']} shards, got {len(shards)}")
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"model shard is missing: {missing[0]}")
    manifest = manifest_entries(path / "MANIFEST.sha256")
    missing_hashes = [name for name in shards if name not in manifest]
    if missing_hashes:
        raise RuntimeError(f"model shard hash is missing from manifest: {missing_hashes[0]}")
    for index, name in enumerate(shards, 1):
        observed = sha256(path / name)
        if observed != manifest[name]:
            raise RuntimeError(
                f"model shard hash mismatch for {name}: expected {manifest[name]}, got {observed}"
            )
        if index % 8 == 0 or index == len(shards):
            print(f"verified EXL3 shards: {index}/{len(shards)}", file=sys.stderr)
    weight_bytes = sum((path / name).stat().st_size for name in shards)
    if weight_bytes != model["weight_bytes"]:
        raise RuntimeError(f"model weight bytes mismatch: expected {model['weight_bytes']}, got {weight_bytes}")
    return {
        "schema": "sparkring-exl3-model-verification/v1",
        "repository": model["repository"],
        "revision": model["revision"],
        "shard_count": len(shards),
        "weight_bytes": weight_bytes,
        "status": "pass",
    }


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
    snapshot_download(
        repo_id=model["repository"],
        revision=model["revision"],
        local_dir=path,
        token=os.environ.get("HF_TOKEN"),
    )
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
