#!/usr/bin/env python3
"""Verify or adopt the pinned public NF3 target and MTP draft.

Exit codes are deliberately part of the shell-bootstrap contract:

* 0: the payload is complete and its identity markers are valid (or were
  created by ``--adopt``);
* 10: the unmarked payload is incomplete and may be resumed by the downloader;
* 20: existing content violates the pinned identity and must not be overwritten
  automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping

INCOMPLETE = 10
INTEGRITY_FAILURE = 20


class IncompletePayload(RuntimeError):
    """An unmarked download is missing one or more expected files."""


class IntegrityFailure(RuntimeError):
    """Existing bytes contradict the pinned checkpoint identity."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_pinned_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise IncompletePayload(f"missing pinned file: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise IntegrityFailure(
            f"hash mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def load_index(path: Path) -> tuple[set[str], int]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        weight_map = document["weight_map"]
        total_size = document["metadata"]["total_size"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise IntegrityFailure(f"invalid safetensors index {path}: {error}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise IntegrityFailure(f"empty or invalid weight_map in {path}")
    if not isinstance(total_size, int) or total_size <= 0:
        raise IntegrityFailure(f"invalid metadata.total_size in {path}")

    names: set[str] = set()
    for raw_name in weight_map.values():
        if not isinstance(raw_name, str) or not raw_name:
            raise IntegrityFailure(f"invalid shard name in {path}: {raw_name!r}")
        parsed = PurePosixPath(raw_name)
        if parsed.is_absolute() or parsed.name != raw_name or ".." in parsed.parts:
            raise IntegrityFailure(f"unsafe shard name in {path}: {raw_name!r}")
        names.add(raw_name)
    return names, total_size


def verify_indexed_payload(
    root: Path,
    index_path: Path,
    *,
    expected_file_count: int | None = None,
    expected_names: set[str] | None = None,
) -> None:
    names, tensor_bytes = load_index(index_path)
    if expected_file_count is not None and len(names) != expected_file_count:
        raise IntegrityFailure(
            f"{index_path} names {len(names)} files; expected {expected_file_count}"
        )
    if expected_names is not None and names != expected_names:
        raise IntegrityFailure(
            f"{index_path} names {sorted(names)!r}; expected {sorted(expected_names)!r}"
        )

    missing = [name for name in sorted(names) if not (root / name).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise IncompletePayload(f"missing checkpoint files: {preview}{suffix}")

    observed_bytes = sum((root / name).stat().st_size for name in names)
    if observed_bytes < tensor_bytes:
        raise IntegrityFailure(
            f"checkpoint files under {root} contain {observed_bytes} bytes; "
            f"index requires at least {tensor_bytes}"
        )


def parse_marker(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise IntegrityFailure(f"cannot read identity marker {path}: {error}") from error
    for line in lines:
        if not line or "=" not in line:
            raise IntegrityFailure(f"invalid identity marker line in {path}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise IntegrityFailure(f"invalid duplicate marker key in {path}: {key!r}")
        values[key] = value
    return values


def marker_text(values: Mapping[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in values.items())


def write_marker_atomic(path: Path, expected: Mapping[str, str]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".sparkring-model.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(marker_text(expected))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_markers(arguments: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    model = {
        "repository": arguments.model_repository,
        "revision": arguments.model_revision,
        "config_sha256": arguments.model_config_sha256,
        "index_sha256": arguments.model_index_sha256,
        "shards": str(arguments.model_shards),
    }
    draft = {
        "repository": arguments.draft_repository,
        "revision": arguments.draft_revision,
        "subdirectory": "mtp-draft",
        "config_sha256": arguments.draft_config_sha256,
        "index_sha256": arguments.draft_index_sha256,
        "weight_sha256": arguments.draft_weight_sha256,
        "inputscales_sha256": arguments.draft_inputscales_sha256,
    }
    return model, draft


def legacy_draft_marker(expected: Mapping[str, str]) -> dict[str, str]:
    """Return the marker schema written before input scales were pinned."""
    return {
        key: value
        for key, value in expected.items()
        if key != "inputscales_sha256"
    }


def verify_payload(arguments: argparse.Namespace) -> None:
    model_root = arguments.model_dir.resolve()
    draft_root = arguments.draft_dir.resolve()
    model_index = model_root / "model.safetensors.index.json"
    draft_index = draft_root / "model.safetensors.index.json"

    require_pinned_file(model_root / "config.json", arguments.model_config_sha256)
    require_pinned_file(model_index, arguments.model_index_sha256)
    require_pinned_file(draft_root / "config.json", arguments.draft_config_sha256)
    require_pinned_file(draft_index, arguments.draft_index_sha256)
    require_pinned_file(
        draft_root / "model-mtp.safetensors", arguments.draft_weight_sha256
    )
    require_pinned_file(
        draft_root / "model-mtp-inputscales.safetensors",
        arguments.draft_inputscales_sha256,
    )

    verify_indexed_payload(
        model_root,
        model_index,
        expected_file_count=arguments.model_shards,
    )
    verify_indexed_payload(
        draft_root,
        draft_index,
        expected_names={
            "model-mtp.safetensors",
            "model-mtp-inputscales.safetensors",
        },
    )


def verify_or_adopt(arguments: argparse.Namespace) -> None:
    model_marker = arguments.model_dir / ".sparkring-model.txt"
    draft_marker = arguments.draft_dir / ".sparkring-model.txt"
    marker_paths = (model_marker, draft_marker)
    markers_present = any(path.exists() for path in marker_paths)
    model_expected, draft_expected = expected_markers(arguments)

    try:
        verify_payload(arguments)
    except IncompletePayload:
        if markers_present:
            raise IntegrityFailure(
                "a SparkRing identity marker exists but the checkpoint payload is incomplete"
            )
        raise

    for path, expected, legacy in (
        (model_marker, model_expected, None),
        (draft_marker, draft_expected, legacy_draft_marker(draft_expected)),
    ):
        if path.exists():
            actual = parse_marker(path)
            if actual == dict(expected):
                continue
            if arguments.adopt and legacy is not None and actual == legacy:
                # The full payload, including the newly pinned input-scales
                # file, was verified before this marker schema upgrade.
                write_marker_atomic(path, expected)
                continue
            raise IntegrityFailure(
                f"identity marker mismatch for {path}; "
                "expected pinned checkpoint metadata"
            )
        elif arguments.adopt:
            write_marker_atomic(path, expected)
        else:
            raise IncompletePayload(f"missing identity marker: {path}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--draft-dir", type=Path, required=True)
    result.add_argument("--model-repository", required=True)
    result.add_argument("--model-revision", required=True)
    result.add_argument("--model-config-sha256", required=True)
    result.add_argument("--model-index-sha256", required=True)
    result.add_argument("--model-shards", type=int, required=True)
    result.add_argument("--draft-repository", required=True)
    result.add_argument("--draft-revision", required=True)
    result.add_argument("--draft-config-sha256", required=True)
    result.add_argument("--draft-index-sha256", required=True)
    result.add_argument("--draft-weight-sha256", required=True)
    result.add_argument("--draft-inputscales-sha256", required=True)
    result.add_argument(
        "--adopt",
        action="store_true",
        help="atomically create missing identity markers after full verification",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        verify_or_adopt(arguments)
    except IncompletePayload as error:
        print(f"INCOMPLETE: {error}", file=sys.stderr)
        return INCOMPLETE
    except IntegrityFailure as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return INTEGRITY_FAILURE
    print("PASS: pinned NF3 target and MTP draft payload verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
