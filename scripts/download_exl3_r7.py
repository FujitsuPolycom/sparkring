#!/usr/bin/env python3
"""Download and fail-closed verify the immutable GLM-5.2 R7 checkpoint.

The repository's MANIFEST files describe assembly provenance. They do not seal
the serving payload. Runtime ownership therefore comes from the pinned index,
and every referenced LFS object is checked against metadata at the pinned Git
revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path


REPOSITORY = "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
REVISION = "9ab9579774cc432df91567a36f6e9e863e0d4c9f"
HEADROOM_BYTES = 16 * 1024**3
EXPECTED_INDEX_TOTAL_SIZE = 346_218_639_128
STALE_INDEX_TOTAL_SIZE = EXPECTED_INDEX_TOTAL_SIZE - 22_130_880
EXPECTED_SHARD_COUNT = 157
EXPECTED_WEIGHT_COUNT = 186_905

# These identities were resolved at REVISION. MANIFEST.json is pinned only so
# it cannot silently change; it remains provenance rather than a payload seal.
PINNED_FILES = {
    "MANIFEST.json": (10_719, "df2f4c87b22c21c5234ef216149f5b5adc556820bb97ae4bc6dd7f4f0647b8db"),
    "MANIFEST.sha256": (80, "c81f3129e418683b6e37c17b8198681c11324f3918fdbd844c1be346114c387b"),
    "chat_template.jinja": (5_076, "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679"),
    "config.json": (150_620, "fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126"),
    "generation_config.json": (194, "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55"),
    "model.safetensors.index.json": (16_284_633, "9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd"),
    "model-sharedbf16.safetensors": (5_737_837_256, "ee1e7d9b2adb5d49c0895dc2f4b7d6d424b108cdb796879eee4c55d040408c6a"),
    "tier_bitmap.json": (514_221, "e0b03bead848272a3ae1a335c24ebc55632dea4916bfa8b4bac742fc802e7a3f"),
    "tokenizer.json": (20_217_442, "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"),
    "tokenizer_config.json": (761, "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(api) -> dict[str, tuple[int, str]]:
    info = api.model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"repository resolved to {info.sha}, expected {REVISION}")
    result = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            digest = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            result[sibling.rfilename] = (int(sibling.size), digest)
    return result


def indexed_shards(path: Path) -> set[str]:
    data = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    total = data.get("metadata", {}).get("total_size")
    if total == STALE_INDEX_TOTAL_SIZE:
        raise RuntimeError(
            "index has the qualified checkpoint's stale payload total; "
            f"expected corrected total_size {EXPECTED_INDEX_TOTAL_SIZE}"
        )
    if total != EXPECTED_INDEX_TOTAL_SIZE:
        raise RuntimeError(f"index total_size is {total}, expected {EXPECTED_INDEX_TOTAL_SIZE}")
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_WEIGHT_COUNT:
        raise RuntimeError(f"index weight count is {len(weight_map or {})}, expected {EXPECTED_WEIGHT_COUNT}")
    shards = set(weight_map.values())
    if len(shards) != EXPECTED_SHARD_COUNT:
        raise RuntimeError(f"index shard count is {len(shards)}, expected {EXPECTED_SHARD_COUNT}")
    if "model-sharedbf16.safetensors" not in shards:
        raise RuntimeError("index does not reference the pinned BF16 shared-expert shard")
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise RuntimeError("index contains an unsafe or non-safetensors shard name")
    return shards


def check_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing model file: {path.name}")
    if path.stat().st_size != size:
        raise RuntimeError(f"model file size mismatch for {path.name}")
    observed = sha256(path)
    if observed != digest:
        raise RuntimeError(f"model file hash mismatch for {path.name}: expected {digest}, got {observed}")


def verify(path: Path, remote_inventory: dict[str, tuple[int, str]]) -> dict:
    for name, (size, digest) in PINNED_FILES.items():
        check_file(path / name, size, digest)
    shards = indexed_shards(path)
    missing_metadata = sorted(shards - remote_inventory.keys())
    if missing_metadata:
        raise RuntimeError(f"pinned revision lacks LFS SHA-256 metadata for: {', '.join(missing_metadata)}")
    for number, name in enumerate(sorted(shards), 1):
        print(f"verify {number}/{len(shards)} {name}", file=sys.stderr)
        check_file(path / name, *remote_inventory[name])
    return {
        "status": "pass",
        "repository": REPOSITORY,
        "revision": REVISION,
        "runtime_shard_count": len(shards),
        "runtime_weight_bytes": sum(remote_inventory[name][0] for name in shards),
        "index_total_size": EXPECTED_INDEX_TOTAL_SIZE,
    }


def quarantine(path: Path, root: Path) -> None:
    destination = root / ".cache" / "sparkring-replaced" / f"{path.name}.replaced-{uuid.uuid4().hex}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)


def download(path: Path, api, snapshot_download) -> dict:
    remote_inventory = inventory(api)
    path.mkdir(parents=True, exist_ok=True)
    common = {"repo_id": REPOSITORY, "revision": REVISION, "local_dir": path, "token": os.environ.get("HF_TOKEN")}
    metadata_names = sorted(name for name in PINNED_FILES if name != "model-sharedbf16.safetensors")
    for name in metadata_names:
        target = path / name
        expected = PINNED_FILES[name]
        if target.is_file() and (target.stat().st_size != expected[0] or sha256(target) != expected[1]):
            quarantine(target, path)
    snapshot_download(**common, allow_patterns=metadata_names, force_download=True)
    for name in metadata_names:
        check_file(path / name, *PINNED_FILES[name])
    shards = indexed_shards(path)
    missing_metadata = sorted(shards - remote_inventory.keys())
    if missing_metadata:
        raise RuntimeError(f"pinned revision lacks LFS SHA-256 metadata for: {', '.join(missing_metadata)}")
    needed = []
    for name in sorted(shards):
        target = path / name
        expected = remote_inventory[name]
        if target.is_file() and (target.stat().st_size != expected[0] or sha256(target) != expected[1]):
            quarantine(target, path)
        if not target.is_file():
            needed.append(name)
    remaining = sum(remote_inventory[name][0] for name in needed)
    if shutil.disk_usage(path).free < remaining + HEADROOM_BYTES:
        raise RuntimeError(f"insufficient model disk space: need {remaining + HEADROOM_BYTES}")
    if needed:
        snapshot_download(**common, allow_patterns=needed, force_download=True)
    return verify(path, remote_inventory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        remote_inventory = inventory(api)
        report = (
            download(args.model_path.resolve(), api, snapshot_download)
            if args.action == "download"
            else verify(args.model_path.resolve(), remote_inventory)
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
