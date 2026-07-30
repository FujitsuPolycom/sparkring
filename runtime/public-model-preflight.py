#!/usr/bin/env python3
"""Read-only structural preflight for the pinned sharded safetensors model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class ModelPreflightError(ValueError):
    pass


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ModelPreflightError(f"{label}: expected hash is malformed")
    if not path.is_file():
        raise ModelPreflightError(f"{label}: missing file: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ModelPreflightError(
            f"{label}: sha256 mismatch: expected {expected}, got {actual}"
        )


def _inside(root: Path, relative: str, label: str) -> Path:
    if not relative or "\x00" in relative:
        raise ModelPreflightError(f"{label}: empty or invalid path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelPreflightError(f"{label}: path escapes model root: {relative}") from exc
    return path


def inspect_model(
    model_path: Path,
    draft_relative: str = "mtp-draft",
    *,
    draft_path: Path | None = None,
    index_sha256: str | None = None,
    draft_config_sha256: str | None = None,
    draft_index_sha256: str | None = None,
    draft_weight_sha256: str | None = None,
) -> dict:
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
    if index_sha256 is not None:
        _require_sha256(indexes[0], index_sha256, "model index")
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

    if draft_path is None:
        draft = _inside(root, draft_relative, "draft")
    else:
        draft = draft_path.resolve()
        if not draft.is_dir():
            raise ModelPreflightError(f"draft root is not a directory: {draft}")
    if not (draft / "config.json").is_file():
        raise ModelPreflightError(f"draft config is missing: {draft / 'config.json'}")
    draft_weights = sorted(draft.glob("*.safetensors"))
    draft_indexes = sorted(draft.glob("*.safetensors.index.json"))
    if not draft_weights and not draft_indexes:
        raise ModelPreflightError(f"draft safetensors are missing under {draft}")
    if draft_config_sha256 is not None:
        _require_sha256(
            draft / "config.json",
            draft_config_sha256,
            "draft config",
        )
    if draft_index_sha256 is not None:
        _require_sha256(
            draft / "model.safetensors.index.json",
            draft_index_sha256,
            "draft index",
        )
    if draft_weight_sha256 is not None:
        _require_sha256(
            draft / "model-mtp.safetensors",
            draft_weight_sha256,
            "draft weight",
        )

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
    parser.add_argument("--draft-path", type=Path)
    parser.add_argument("--index-sha256")
    parser.add_argument("--draft-config-sha256")
    parser.add_argument("--draft-index-sha256")
    parser.add_argument("--draft-weight-sha256")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = inspect_model(
            args.model_path,
            args.draft_relative,
            draft_path=args.draft_path,
            index_sha256=args.index_sha256,
            draft_config_sha256=args.draft_config_sha256,
            draft_index_sha256=args.draft_index_sha256,
            draft_weight_sha256=args.draft_weight_sha256,
        )
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
