#!/usr/bin/env python3
"""Read-only structural preflight for the pinned sharded safetensors model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class ModelPreflightError(ValueError):
    pass


def _inside(root: Path, relative: str, label: str) -> Path:
    if not relative or "\x00" in relative:
        raise ModelPreflightError(f"{label}: empty or invalid path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelPreflightError(f"{label}: path escapes model root: {relative}") from exc
    return path


def inspect_model(model_path: Path, draft_relative: str) -> dict:
    root = model_path.resolve()
    if not root.is_dir():
        raise ModelPreflightError(f"model root is not a directory: {root}")
    config = root / "config.json"
    if not config.is_file():
        raise ModelPreflightError(f"model config is missing: {config}")
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ModelPreflightError(
            f"expected exactly one root safetensors index, found {len(indexes)}"
        )
    try:
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPreflightError(f"cannot read {indexes[0]}: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelPreflightError("safetensors index has no non-empty weight_map")
    if not all(isinstance(name, str) and name for name in weight_map):
        raise ModelPreflightError("safetensors index contains an invalid tensor name")
    shard_values = list(weight_map.values())
    if not all(
        isinstance(name, str) and name.endswith(".safetensors")
        for name in shard_values
    ):
        raise ModelPreflightError("safetensors index contains an invalid shard path")
    shard_names = sorted(set(shard_values))
    total_bytes = 0
    for name in shard_names:
        shard = _inside(root, name, "weight_map")
        if not shard.is_file():
            raise ModelPreflightError(f"referenced shard is missing: {name}")
        size = shard.stat().st_size
        if size <= 0:
            raise ModelPreflightError(f"referenced shard is empty: {name}")
        total_bytes += size

    draft = _inside(root, draft_relative, "draft")
    if not (draft / "config.json").is_file():
        raise ModelPreflightError(f"draft config is missing: {draft / 'config.json'}")
    draft_weights = sorted(draft.glob("*.safetensors"))
    draft_indexes = sorted(draft.glob("*.safetensors.index.json"))
    if not draft_weights and not draft_indexes:
        raise ModelPreflightError(f"draft safetensors are missing under {draft}")

    return {
        "schema": "sparkring-public-model-preflight/v1",
        "passed": True,
        "model_path": str(root),
        "index": indexes[0].name,
        "tensor_entries": len(weight_map),
        "unique_shards": len(shard_names),
        "weight_bytes": total_bytes,
        "draft_path": str(draft),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--draft-relative", default="mtp-draft")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = inspect_model(args.model_path, args.draft_relative)
    except (OSError, ModelPreflightError) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema": "sparkring-public-model-preflight/v1",
                        "passed": False,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"FAIL: {exc}")
        return 78
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
