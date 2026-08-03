#!/usr/bin/env python3
"""Connector bundle manifest and identity builder for SparkCache staging.

Produces a deterministic SHA-256 identity over the complete inventory of
regular, non-symlink files in a connector staging directory.  The
identity pins the exact connector code that gets bind-mounted read-only
into target containers, preventing silent drift between plan, prepare,
and cutover.

The required-file allowlist ensures the bundle is internally consistent:
the staging directory must contain the complete connector import
closure under ``sparkcache/`` — the three top-level connector
modules, the persistent-context-cache storage engine, the streaming
package (eagerly imported via ``streaming/__init__.py``), and the
feature-gate module.

Canonical staging layout (everything beneath ``<staging-root>/sparkcache/``):

    sparkcache/
      spark_context_cache_connector.py
      spark_context_cache_codec.py
      spark_context_cache_store.py
      sparkcache/
        streaming/            (all non-test .py modules)
        persistent_context_cache/
          cache_manifest.py

Usage (OFFLINE, on the operator's staging directory):

    python scripts/connector_bundle_manifest.py \\
        --staging-root /opt/sparkcache-host-staging

The ``connector_bundle_identity_sha256`` field in the receipt JSON is
the value passed to ``--connector-bundle-identity`` in
``live_dcp2_cutover.py``.

Fail-closed on:
  - Symlink root, symlink files, or symlink directories
  - Non-regular files
  - Missing required files from the allowlist
  - Extra files not in the allowlist
  - Unreadable files
  - Files changing during the read
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Never

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUNDLE_DOMAIN_SEPARATOR = "sparkcache-connector-bundle-v1"
CHUNK_SIZE = 1 << 20  # 1 MiB

# ---------------------------------------------------------------------------
# Canonical required-files allowlist
#
# The connector's module-load import closure, expressed as paths relative
# to the staging root.  Every file is beneath ``sparkcache/``:
#
#   spark_context_cache_{connector,codec,store}.py  — top-level modules
#     importable via ``<dest>/sparkcache`` on PYTHONPATH.
#   sparkcache/streaming/*.py  — eagerly imported by streaming/__init__.py
#     when the connector imports ``sparkcache.streaming.feature_gate``.
#   sparkcache/persistent_context_cache/cache_manifest.py  — storage engine
#     imported by publisher.py and dynamically loaded by store.py.
#
# Tests and generated caches are excluded.  This list is single-sourced:
# the cutover script imports it to build its remote verifier, so the
# offline builder and remote verifier can never drift.
# ---------------------------------------------------------------------------

# Non-test runtime .py files in sparkcache/streaming/ (eagerly imported
# by streaming/__init__.py plus lazy-loaded runtime modules).
_STREAMING_RUNTIME_FILES = [
    "__init__.py",
    "feature_gate.py",
    "block_lease.py",
    "planner.py",
    "native_ring.py",
    "preemption.py",
    "publisher.py",
    "factory.py",
    "runtime.py",
    "timing.py",
]

# Runtime patches package (lease-contract verifier + JSON contract).
_RUNTIME_PATCHES_FILES = [
    "__init__.py",
    "verify_lease_contract.py",
    "vllm-kv-block-lease-contract.json",
]

# Required files — the staging directory must contain exactly these
# regular files (no symlinks, no extra files).  This is the complete
# local import closure of spark_context_cache_connector.py when staged
# under <root>/sparkcache/ with PYTHONPATH=<root>:<root>/sparkcache.
REQUIRED_FILES = (
    frozenset(
        # Top-level connector modules.
        f"sparkcache/spark_context_cache_{name}.py"
        for name in ("connector", "codec", "store")
    )
    | frozenset({
        # Persistent-context-cache storage engine (dynamically loaded by store).
        "sparkcache/persistent_context_cache/cache_manifest.py",
    })
    | frozenset(
        # Streaming package — __init__.py eagerly imports 5 submodules.
        # Plus runtime modules (lazy-loaded by factory when enabled).
        f"sparkcache/streaming/{name}" for name in _STREAMING_RUNTIME_FILES
    )
    | frozenset(
        # Runtime patches package (lease-contract verifier + JSON contract).
        f"sparkcache/runtime_patches/{name}" for name in _RUNTIME_PATCHES_FILES
    )
)


class BundleManifestError(ValueError):
    """Raised when the staging directory is incompatible with manifest generation."""


@dataclass(frozen=True)
class BundleFileEntry:
    rel_path: str
    byte_size: int
    content_sha256: str


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------


def _normalize_rel_path(root: Path, abs_path: Path) -> str:
    rel = abs_path.relative_to(root)
    posix = PurePosixPath(*rel.parts).as_posix()
    return unicodedata.normalize("NFC", posix)


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def _hash_file(
    path: Path,
    chunk_size: int = CHUNK_SIZE,
    st_before: os.stat_result | None = None,
) -> tuple[str, int]:
    """Hash a file in chunks, returning (sha256-hex, bytes-read).

    If ``st_before`` is provided, it is compared against a fresh
    ``os.lstat`` after the read completes.  If the stable metadata
    (size, ino, dev, mtime_ns, ctime_ns) differs, ``OSError`` is raised
    to signal that the file changed during the read.
    """
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    if st_before is not None:
        st_after = os.lstat(path)
        if (st_before.st_size != st_after.st_size or
                st_before.st_ino != st_after.st_ino or
                st_before.st_dev != st_after.st_dev or
                st_before.st_mtime_ns != st_after.st_mtime_ns or
                st_before.st_ctime_ns != st_after.st_ctime_ns):
            raise OSError(f"file changed during read: {path}")
        if size != st_before.st_size:
            raise OSError(
                f"bytes read differ from pre-read size for {path}: "
                f"read {size}, expected {st_before.st_size}"
            )
    return h.hexdigest(), size


# ---------------------------------------------------------------------------
# Core inventory
# ---------------------------------------------------------------------------


def inventory_staging_root(
    staging_root: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    _hasher=_hash_file,
) -> list[BundleFileEntry]:
    """Inventory all required regular files under staging_root.

    Raises BundleManifestError on symlink root, symlinks, missing
    required files, extra files, non-regular files, or unreadable files.
    """
    supplied = Path(staging_root)

    # --- Root validation: lstat, reject symlink, require directory ---
    try:
        root_st = os.lstat(supplied)
    except OSError as error:
        raise BundleManifestError(
            f"cannot lstat staging root: {error}"
        ) from error
    if stat.S_ISLNK(root_st.st_mode):
        raise BundleManifestError(
            f"staging root is a symlink, refusing to dereference: {supplied}"
        )
    if not stat.S_ISDIR(root_st.st_mode):
        raise BundleManifestError(f"staging root is not a directory: {supplied}")
    root = supplied.resolve()

    entries: list[BundleFileEntry] = []
    seen_paths: set[str] = set()

    for required_rel in sorted(REQUIRED_FILES):
        abs_path = root / required_rel
        try:
            st = os.lstat(abs_path)
        except OSError as error:
            raise BundleManifestError(
                f"cannot lstat required file {required_rel}: {error}"
            ) from error
        if stat.S_ISLNK(st.st_mode):
            raise BundleManifestError(
                f"required file is a symlink: {required_rel}"
            )
        if not stat.S_ISREG(st.st_mode):
            raise BundleManifestError(
                f"required file missing or not regular: {required_rel}"
            )

        norm = _normalize_rel_path(root, abs_path)
        if norm in seen_paths:
            raise BundleManifestError(
                f"duplicate normalized path: {norm}"
            )
        seen_paths.add(norm)

        try:
            content_sha, size = _hasher(abs_path, chunk_size=chunk_size, st_before=st)
        except OSError as error:
            raise BundleManifestError(
                f"cannot read file {required_rel}: {error}"
            ) from error

        entries.append(BundleFileEntry(
            rel_path=norm, byte_size=size, content_sha256=content_sha,
        ))

    # Reject extra files not in the allowlist (fail-closed walk errors).
    def _walk_fail(error: OSError) -> Never:
        raise BundleManifestError(f"walk error: {error}") from error

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=_walk_fail):
        for dirname in list(dirnames):
            full = Path(dirpath) / dirname
            try:
                dst = os.lstat(full)
            except OSError as error:
                raise BundleManifestError(
                    f"cannot lstat directory {full}: {error}"
                ) from error
            if stat.S_ISLNK(dst.st_mode):
                raise BundleManifestError(
                    f"symlink directory found: {full}"
                )
        for filename in filenames:
            abs_file = Path(dirpath) / filename
            rel = _normalize_rel_path(root, abs_file)
            if rel not in REQUIRED_FILES:
                raise BundleManifestError(
                    f"extra file not in allowlist: {rel}"
                )

    return sorted(entries, key=lambda e: e.rel_path)


def compute_bundle_identity(entries: list[BundleFileEntry]) -> str:
    """Compute the deterministic bundle identity SHA-256."""
    h = hashlib.sha256()
    h.update(BUNDLE_DOMAIN_SEPARATOR.encode("utf-8"))
    for entry in entries:
        h.update(b"\x00")
        h.update(entry.rel_path.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(entry.byte_size).encode("utf-8"))
        h.update(b"\x00")
        h.update(entry.content_sha256.encode("utf-8"))
    return h.hexdigest()


def generate_receipt(staging_root: str | Path) -> dict:
    """Generate a complete receipt dict with inventory and identity."""
    entries = inventory_staging_root(staging_root)
    identity = compute_bundle_identity(entries)
    return {
        "connector_bundle_identity_sha256": identity,
        "domain_separator": BUNDLE_DOMAIN_SEPARATOR,
        "file_count": len(entries),
        "total_bytes": sum(e.byte_size for e in entries),
        "files": [
            {
                "rel_path": e.rel_path,
                "byte_size": e.byte_size,
                "content_sha256": e.content_sha256,
            }
            for e in entries
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root", required=True,
        help="Connector staging directory to inventory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write receipt JSON to this file (default: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        receipt = generate_receipt(args.staging_root)
    except BundleManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
