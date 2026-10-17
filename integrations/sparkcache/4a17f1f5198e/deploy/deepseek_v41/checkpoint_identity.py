"""Derive or verify the DeepSeek-V4.1-Flash checkpoint identity from pins.

The identity binds the checkpoint repository revision, the complete artifact
manifest, the served B12X representation, the exact config bytes, and the
embedded DSpark draft behavior via :mod:`sparkcache.checkpoint_identity`.
The pins ship beside this script and come from SparkRing's installer profile
``profiles/deepseek-v41-flash-tp4`` (revision, ``config_sha256``,
``index_sha256``, speculative configuration, B12X load format and backends)
and its 77-file ``SHA256SUMS`` checkpoint manifest. Deriving a digest hashes
supplied pins only; it does not attest that local weights match them.

``derive`` recomputes the digest from the pinned inputs and compares it with
the recorded receipt. ``verify-root`` recomputes per-file SHA-256 digests of
a local checkpoint directory and compares them with the pinned
``checkpoint-sha256sums.txt`` manifest, so an operator can prove a downloaded
checkpoint matches the pins before serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparkcache.checkpoint_identity import (
    SCHEMA as IDENTITY_SCHEMA,
    derive_checkpoint_identity,
)

_DIRECTORY = Path(__file__).resolve().parent
INPUTS_PATH = _DIRECTORY / "checkpoint-identity-inputs.json"
RECEIPT_PATH = _DIRECTORY / "checkpoint-identity-receipt.json"
CHECKSUMS_PATH = _DIRECTORY / "checkpoint-sha256sums.txt"


def load_inputs() -> dict[str, Any]:
    """Return the pinned identity inputs."""

    return json.loads(INPUTS_PATH.read_text(encoding="utf-8"))


def derive() -> dict[str, Any]:
    """Recompute the identity from the pins and compare the recorded receipt."""

    inputs = load_inputs()
    checkpoint_sha256 = derive_checkpoint_identity(inputs)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    if receipt.get("schema") != IDENTITY_SCHEMA:
        raise RuntimeError(
            "checkpoint identity receipt schema differs:"
            f" {receipt.get('schema')!r}"
        )
    if receipt.get("inputs") != inputs:
        raise RuntimeError("checkpoint identity receipt inputs differ from pins")
    if receipt.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(
            "checkpoint identity digest differs from the recorded receipt:"
            f" derived {checkpoint_sha256}, recorded"
            f" {receipt.get('checkpoint_sha256')}"
        )
    return receipt


def _parse_sha256sums(text: str) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, path = line.partition("  ")
        digest = digest.strip()
        path = path.strip()
        if len(digest) != 64 or not path or path in pinned:
            raise RuntimeError(f"pinned checksum manifest line is invalid: {line!r}")
        pinned[path] = digest
    return pinned


def verify_root(root: Path, *, workers: int = 4) -> dict[str, Any]:
    """Hash a local checkpoint root and compare it with the pinned manifest.

    Every pinned path must exist with exactly the pinned digest and no extra
    files may appear. Symlinks and special files are refused.
    """

    import concurrent.futures

    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"checkpoint root is absent: {root}")
    pinned = _parse_sha256sums(CHECKSUMS_PATH.read_text(encoding="utf-8"))

    present: dict[str, int] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file():
            if path.is_dir():
                continue
            raise RuntimeError(f"checkpoint root holds a non-regular file: {relative}")
        present[relative] = path.stat().st_size
    missing = sorted(set(pinned) - set(present))
    extra = sorted(set(present) - set(pinned))
    if missing or extra:
        raise RuntimeError(
            "checkpoint root does not match the pinned manifest:"
            f" missing={missing[:8]} extra={extra[:8]}"
        )

    def _hash(relative: str) -> tuple[str, str]:
        digest = hashlib.sha256()
        with (root / relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return relative, digest.hexdigest()

    mismatches: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for relative, observed in pool.map(_hash, sorted(pinned)):
            if observed != pinned[relative]:
                mismatches.append(relative)
    if mismatches:
        raise RuntimeError(
            "checkpoint files disagree with the pinned manifest:"
            f" {mismatches[:8]}"
        )
    return {
        "schema": "sparkcache-deepseek-v41-checkpoint-verify/v1",
        "repository": str(load_inputs()["repository"]),
        "revision": str(load_inputs()["revision"]),
        "files_verified": len(pinned),
        "bytes_verified": sum(present.values()),
        "checkpoint_sha256": derive()["checkpoint_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("derive", help="recompute the pinned identity")
    verify = commands.add_parser(
        "verify-root", help="hash a local checkpoint against the pins"
    )
    verify.add_argument("root", type=Path)
    verify.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "derive":
        print(json.dumps(derive(), sort_keys=True))
    else:
        print(json.dumps(verify_root(args.root, workers=args.workers), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
