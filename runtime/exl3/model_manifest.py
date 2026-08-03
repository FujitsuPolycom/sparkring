#!/usr/bin/env python3
"""Path-safe verification for the pinned public EXL3 model manifest."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Callable


ProgressCallback = Callable[[int, int], None]

# Files vLLM/Transformers may consume while loading or serving this checkpoint.
# Weight files are added from model.safetensors.index.json.  Release notes,
# compose examples, licenses, and calibration provenance remain covered by the
# pinned MANIFEST.sha256 identity but are deliberately not runtime inputs.
RUNTIME_METADATA_FILES = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tier_bitmap.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise RuntimeError(f"invalid MANIFEST.sha256 line {number}")
        digest, name = fields
        name = name.strip().removeprefix("./")
        pure = PurePosixPath(name)
        if (
            not name
            or pure.is_absolute()
            or "\\" in name
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise RuntimeError(
                f"unsafe MANIFEST.sha256 path on line {number}: {name!r}"
            )
        if name in entries:
            raise RuntimeError(f"duplicate MANIFEST.sha256 entry: {name}")
        entries[name] = digest
    if not entries:
        raise RuntimeError("MANIFEST.sha256 is empty")
    return entries


def manifest_target(root: Path, name: str) -> Path:
    target = root.joinpath(*PurePosixPath(name).parts)
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"model manifest path escapes model root: {name}") from exc
    return target


def runtime_manifest_entries(manifest: dict[str, str]) -> dict[str, str]:
    """Select the receipt-owned files that can affect model execution."""
    missing = sorted(RUNTIME_METADATA_FILES - set(manifest))
    if missing:
        raise RuntimeError(f"runtime model file is missing from manifest: {missing[0]}")
    return {
        name: digest
        for name, digest in manifest.items()
        if name in RUNTIME_METADATA_FILES or name.endswith(".safetensors")
    }


def verify_model(
    path: Path,
    model: dict,
    *,
    progress: ProgressCallback | None = None,
) -> dict:
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
            raise RuntimeError(
                f"{name} hash mismatch: expected {expected}, got {observed}"
            )

    index = json.loads(
        (path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index.get("weight_map", {}).values()))
    if len(shards) != model["shard_count"]:
        raise RuntimeError(f"expected {model['shard_count']} shards, got {len(shards)}")
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"model shard is missing: {missing[0]}")

    manifest = manifest_entries(path / "MANIFEST.sha256")
    runtime_manifest = runtime_manifest_entries(manifest)
    missing_hashes = [name for name in shards if name not in runtime_manifest]
    if missing_hashes:
        raise RuntimeError(
            f"model shard hash is missing from manifest: {missing_hashes[0]}"
        )
    observed_files = {
        candidate.relative_to(path).as_posix()
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.relative_to(path).as_posix() != "MANIFEST.sha256"
        and candidate.relative_to(path).parts[:1] != (".cache",)
    }
    unmanifested = sorted(observed_files - set(manifest))
    if unmanifested:
        raise RuntimeError(f"unmanifested model files: {unmanifested}")
    shard_names = set(shards)
    verified_shards = 0
    for name, expected in runtime_manifest.items():
        target = manifest_target(path, name)
        if not target.is_file():
            raise RuntimeError(f"model manifest file is missing: {name}")
        observed = sha256(target)
        if observed != expected:
            kind = "model shard" if name in shard_names else "model file"
            raise RuntimeError(
                f"{kind} hash mismatch for {name}: expected {expected}, got {observed}"
            )
        if name in shard_names:
            verified_shards += 1
            if progress is not None:
                progress(verified_shards, len(shards))

    weight_bytes = sum((path / name).stat().st_size for name in shards)
    if weight_bytes != model["weight_bytes"]:
        raise RuntimeError(
            f"model weight bytes mismatch: expected {model['weight_bytes']}, "
            f"got {weight_bytes}"
        )
    return {
        "schema": "sparkring-exl3-model-verification/v1",
        "repository": model["repository"],
        "revision": model["revision"],
        "shard_count": len(shards),
        "runtime_file_count": len(runtime_manifest),
        "ignored_release_file_count": len(manifest) - len(runtime_manifest),
        "weight_bytes": weight_bytes,
        "status": "pass",
    }


def stderr_progress(done: int, total: int) -> None:
    if done % 8 == 0 or done == total:
        print(f"verified EXL3 shards: {done}/{total}", file=sys.stderr)
