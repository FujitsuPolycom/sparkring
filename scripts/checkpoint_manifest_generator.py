#!/usr/bin/env python3
"""Canonical checkpoint manifest generator for SparkCache identity.

Produces a deterministic, versioned JSON receipt inventorying every
regular file under an explicit artifact root.  The checkpoint identity
is a domain-separated SHA-256 over the canonical serialization of the
complete inventory — not merely revision/config/index pins.

The generator is **read-only**: it never downloads, mutates, or creates
files in the artifact root.  It streams file contents in chunks (default
1 MiB) and never loads a whole file into memory.

Fail-closed on:
  - Symlink root (the explicitly supplied root is itself a symlink)
  - Symlink files or symlink directories anywhere under the root
  - Non-regular files (devices, sockets, FIFOs, etc.)
  - Unreadable files
  - Duplicate/colliding normalized paths (Unicode NFC + POSIX)
  - Files changing during the read (size/inode/mtime_ns/ctime_ns/dev)
  - Empty inventory

Path normalization contract:
  Relative paths are normalized to POSIX separators (``/``) and
  Unicode NFC form.  Two paths that normalize to the same string are
  a collision and cause fail-closed rejection.  This makes the
  identity reproducible regardless of filesystem enumeration order,
  root directory name, or platform path separators.

Usage (offline, read-only):

    # Target artifact root — stdout receipt (includes identity)
    python scripts/checkpoint_manifest_generator.py \
        --artifact-root /path/to/target-model

    # Draft artifact root — write to exclusive local file
    python scripts/checkpoint_manifest_generator.py \
        --artifact-root /path/to/mtp-draft \
        --output draft_receipt.json

Actually hashing the deployed mounts is READ-ONLY REMOTE and may be
expensive (the target model has ~184 weight shards).  The generator
is designed to be supplied via stdin to a container running the same
image — see docs/history/SPARKCACHE_DCP2_LIVE_RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_VERSION = 2
CHUNK_SIZE = 1 << 20  # 1 MiB
DOMAIN_SEPARATOR = "sparkcache-checkpoint-manifest-v2"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ManifestError(ValueError):
    """Raised when the artifact root is incompatible with manifest generation."""


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileEntry:
    rel_path: str   # normalized POSIX + NFC relative path
    byte_size: int
    content_sha256: str


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------

def _normalize_rel_path(root: Path, abs_path: Path) -> str:
    """Return a normalized POSIX + NFC relative path from root to abs_path."""
    rel = abs_path.relative_to(root)
    posix = PurePosixPath(*rel.parts).as_posix()
    return unicodedata.normalize("NFC", posix)


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def _hash_file(path: Path, chunk_size: int = CHUNK_SIZE) -> tuple[str, int]:
    """Stream-hash a file in chunks.  Returns (sha256_hexdigest, byte_size)."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# ---------------------------------------------------------------------------
# Stable metadata for change detection
# ---------------------------------------------------------------------------

def _stable_stat(path: Path) -> tuple[int, int, int, int, int]:
    """Return (st_size, st_ino, st_dev, st_mtime_ns, st_ctime_ns) from stat.

    These five fields together detect in-place content changes that
    preserve size and inode (e.g. ``dd conv=notrunc``) via mtime/ctime
    changes, and file replacement via dev/ino changes.
    """
    st = path.stat()
    return (st.st_size, st.st_ino, st.st_dev, st.st_mtime_ns, st.st_ctime_ns)


# ---------------------------------------------------------------------------
# Core inventory
# ---------------------------------------------------------------------------

def inventory_artifact_root(
    artifact_root: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    _hasher=_hash_file,
    _stat_fn=_stable_stat,
) -> list[FileEntry]:
    """Recursively inventory all regular files under artifact_root.

    Returns a list of FileEntry sorted by normalized relative path.

    Raises ManifestError on symlink root, symlinks (file or directory),
    non-regular files, unreadable files, duplicate normalized paths,
    files changing during read, or empty inventory.
    """
    supplied = Path(artifact_root)

    # Reject symlink root BEFORE resolve — the caller supplied a path,
    # and if that path is a symlink it must not be silently dereferenced.
    try:
        if supplied.is_symlink():
            raise ManifestError(
                f"artifact root is a symlink, refusing to dereference: {supplied}"
            )
    except OSError as error:
        raise ManifestError(
            f"cannot stat artifact root: {error}"
        ) from error

    root = supplied.resolve()

    try:
        if not root.is_dir():
            raise ManifestError(f"artifact root is not a directory: {root}")
    except OSError as error:
        raise ManifestError(
            f"cannot stat artifact root: {error}"
        ) from error

    entries: list[FileEntry] = []
    seen_paths: set[str] = set()
    seen_dirs: set[str] = set()

    def _on_walk_error(error: OSError) -> None:
        raise ManifestError(
            f"traversal error: {error}"
        ) from error

    # Walk in sorted order for deterministic enumeration.  followlinks=False
    # (the default) ensures os.walk does not traverse symlinked dirs,
    # but we still need to reject them explicitly (no implicit exclusion).
    # onerror converts any os.walk traversal error (e.g. permission
    # denied) into a fail-closed ManifestError.
    try:
        for dirpath, dirnames, filenames in os.walk(
            root, followlinks=False, onerror=_on_walk_error
        ):
            dirnames.sort()

            # Track and validate directory paths for NFC collisions.
            current_dir_rel = _normalize_rel_path(root, Path(dirpath))
            if current_dir_rel and current_dir_rel not in seen_dirs:
                seen_dirs.add(current_dir_rel)

            # Inspect every directory entry — reject symlinked directories
            # and other non-directory types in dirnames.
            kept_dirs: list[str] = []
            for dname in dirnames:
                dpath = Path(dirpath) / dname
                drel = _normalize_rel_path(root, dpath)
                try:
                    if dpath.is_symlink():
                        raise ManifestError(
                            f"symlink directory rejected: {drel}"
                        )
                    # is_dir() follows symlinks; we already rejected
                    # symlinks above, so this checks the real entry.
                    if not dpath.is_dir():
                        raise ManifestError(
                            f"non-directory entry in dirnames rejected: {drel}"
                        )
                except OSError as error:
                    raise ManifestError(
                        f"cannot inspect directory entry: {error}"
                    ) from error

                # Reject duplicate normalized directory paths (NFC collision)
                if drel in seen_dirs:
                    raise ManifestError(
                        f"duplicate normalized directory path: {drel}"
                    )
                seen_dirs.add(drel)

                kept_dirs.append(dname)
            dirnames[:] = kept_dirs

            for filename in sorted(filenames):
                abs_path = Path(dirpath) / filename
                rel_path = _normalize_rel_path(root, abs_path)

                # Reject symlink files
                try:
                    if abs_path.is_symlink():
                        raise ManifestError(
                            f"symlink rejected: {rel_path}"
                        )
                except OSError as error:
                    raise ManifestError(
                        f"cannot stat {rel_path}: {error}"
                    ) from error

                # Reject non-regular files (devices, sockets, FIFOs, etc.)
                try:
                    if not abs_path.is_file():
                        raise ManifestError(
                            f"non-regular file rejected: {rel_path}"
                        )
                except OSError as error:
                    raise ManifestError(
                        f"cannot stat {rel_path}: {error}"
                    ) from error

                # Reject duplicate normalized paths (NFC + POSIX collision)
                if rel_path in seen_paths:
                    raise ManifestError(
                        f"duplicate normalized path: {rel_path}"
                    )
                seen_paths.add(rel_path)

                # Stat before read
                try:
                    meta_before = _stat_fn(abs_path)
                except OSError as error:
                    raise ManifestError(
                        f"cannot stat {rel_path}: {error}"
                    ) from error

                # Hash (stream)
                try:
                    content_sha256, byte_size = _hasher(abs_path, chunk_size)
                except OSError as error:
                    raise ManifestError(
                        f"cannot read {rel_path}: {error}"
                    ) from error

                # Stat after read — detect change during read
                try:
                    meta_after = _stat_fn(abs_path)
                except OSError as error:
                    raise ManifestError(
                        f"cannot re-stat {rel_path}: {error}"
                    ) from error

                if meta_before != meta_after:
                    raise ManifestError(
                        f"file changed during read: {rel_path}"
                    )

                if byte_size != meta_before[0]:
                    raise ManifestError(
                        f"size mismatch after read: {rel_path}"
                        f" (expected {meta_before[0]}, got {byte_size})"
                    )

                entries.append(FileEntry(
                    rel_path=rel_path,
                    byte_size=byte_size,
                    content_sha256=content_sha256,
                ))
    except ManifestError:
        raise
    except OSError as error:
        raise ManifestError(
            f"traversal error: {error}"
        ) from error
    if not entries:
        raise ManifestError(
            f"empty inventory: no regular files under {root}"
        )

    # Deterministic sort by normalized path (after NFC normalization)
    entries.sort(key=lambda e: e.rel_path)
    return entries


# ---------------------------------------------------------------------------
# Receipt and identity
# ---------------------------------------------------------------------------

def build_receipt(
    entries: list[FileEntry],
    *,
    artifact_root_name: str = "",
) -> dict:
    """Build a versioned canonical JSON receipt from file entries."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "path_normalization": "POSIX separators + Unicode NFC",
        "artifact_root_name": artifact_root_name,
        "file_count": len(entries),
        "files": [
            {
                "rel_path": e.rel_path,
                "byte_size": e.byte_size,
                "content_sha256": e.content_sha256,
            }
            for e in entries
        ],
    }


def compute_identity(receipt: dict) -> str:
    """Compute the checkpoint identity from the complete receipt.

    The identity is a domain-separated SHA-256 over the canonical JSON
    serialization of the receipt, excluding ``artifact_root_name`` (for
    root-directory-name independence) and ``checkpoint_identity_sha256``
    (to avoid recursion).  Domain separation prevents collisions with
    other uses of SHA-256 in the cache system.
    """
    identity_input = {
        k: v for k, v in receipt.items()
        if k not in ("artifact_root_name", "checkpoint_identity_sha256")
    }
    canonical = json.dumps(
        identity_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    h = hashlib.sha256()
    h.update(DOMAIN_SEPARATOR.encode("ascii"))
    h.update(b"\x00")
    h.update(canonical)
    return h.hexdigest()


def generate_manifest(
    artifact_root: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    _hasher=_hash_file,
    _stat_fn=_stable_stat,
) -> tuple[dict, str]:
    """Generate a receipt and checkpoint identity for an artifact root.

    Returns (receipt_dict, checkpoint_identity_sha256).

    The supplied path is passed through to ``inventory_artifact_root``
    unchanged — it must NOT be resolved here, because ``.resolve()``
    dereferences symlinks and would bypass the symlink-root rejection
    in ``inventory_artifact_root``.  The display name for the receipt
    is derived separately from the resolved path's final component.
    """
    entries = inventory_artifact_root(
        artifact_root, chunk_size=chunk_size, _hasher=_hasher, _stat_fn=_stat_fn
    )
    # Derive display name without affecting validation — resolve only
    # for the name, not for the traversal.
    display_name = Path(artifact_root).resolve().name
    receipt = build_receipt(
        entries, artifact_root_name=display_name
    )
    identity = compute_identity(receipt)
    return receipt, identity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical checkpoint manifest generator for SparkCache identity."
            " Read-only: never downloads, mutates, or creates files in"
            " the artifact root."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Explicit path to the artifact root (model checkpoint directory).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Write receipt JSON to this path using exclusive creation."
            " Refuses to overwrite an existing file.  If omitted,"
            " prints receipt (including checkpoint_identity_sha256) to"
            " stdout."
        ),
    )
    args = parser.parse_args(argv)

    try:
        receipt, identity = generate_manifest(args.artifact_root)
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    receipt["checkpoint_identity_sha256"] = identity

    if args.output:
        output = Path(args.output)
        try:
            # Exclusive creation — atomically fails if the file already
            # exists, eliminating the TOCTOU race between exists() and
            # write_text().
            with open(output, "x", encoding="utf-8") as f:
                f.write(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
                )
        except FileExistsError:
            print(
                f"error: output file already exists, refusing to overwrite: {output}",
                file=sys.stderr,
            )
            return 1
        except OSError as error:
            print(
                f"error: cannot write output file {output}: {error}",
                file=sys.stderr,
            )
            return 1
        print(f"receipt written to {output}", file=sys.stderr)
        print(identity)
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
