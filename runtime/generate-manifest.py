#!/usr/bin/env python3
"""Generate an immutable runtime manifest from inside the built SparkRing image.

Emits EXACTLY the schema consumed by runtime/verify-runtime.py:

    {
      "runtime_id":  "<id>",
      "components":  [ {"name": ..., "verification": {"mode": "tree"|
                        "patched_files", "root": ..., "files": {...}}} ],
      "native_libs": [ {"path": ..., "sha256": ...} ],
      "image":       {"digest": ...},
      "self_hash":   "<sha256 over the canonical serialization sans self_hash>"
    }

The canonical serialization matches verify-runtime.py's
canonical_manifest_bytes byte-for-byte: self_hash removed, json.dumps with
sort_keys=True, separators=(",", ":"), ensure_ascii=True, encoded UTF-8.
Extra provenance fields (lock, packages) are allowed; the verifier ignores
them but they are covered by the self-hash.

Stdlib only. Python 3.10+.

Usage:
    generate-manifest.py --lock runtime-lock.json --output manifest.json \
        [--runtime-id ID] [--tree name=/path ...] [--native-lib /path ...] \
        [--patch-log patch-apply.jsonl --patch-root /path] \
        [--image-digest sha256:...]
    generate-manifest.py --verify-self manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

SELF_HASH_FIELD = "self_hash"
_CHUNK = 1 << 20


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: str) -> dict[str, str]:
    """{relpath: sha256} over every regular file under root (symlinks by target)."""
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if os.path.islink(full):
                target = os.readlink(full)
                files[rel] = "symlink:" + hashlib.sha256(target.encode()).hexdigest()
            elif os.path.isfile(full):
                files[rel] = sha256_file(full)
    return files


def load_patch_log(path: str) -> dict[str, dict]:
    """Parse apply-patches.py's JSONL log into the verifier's patched_files map.

    Patch lines contain ``preimage_sha256`` and ``postimage_sha256``.
    Addition lines contain ``sha256`` and may be newly installed or inherited
    from an attested compatible base.
    Returns {target: {"preimage": ..., "postimage": ...}}.
    """
    files: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            postimage = rec.get("postimage_sha256", rec.get("sha256"))
            if not isinstance(postimage, str):
                raise ValueError(
                    f"{path}:{lineno}: audit row has no postimage hash"
                )
            files[rec["target"]] = {
                "preimage": rec.get("preimage_sha256"),
                "postimage": postimage,
            }
    return files


def installed_packages() -> dict[str, str]:
    from importlib import metadata

    pkgs: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            pkgs[name] = dist.version
    return dict(sorted(pkgs.items()))


def canonical_manifest_bytes(manifest: dict) -> bytes:
    """Byte-identical to verify-runtime.py's canonical_manifest_bytes."""
    stripped = {k: v for k, v in manifest.items() if k != SELF_HASH_FIELD}
    return json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def compute_self_hash(doc: dict) -> str:
    return hashlib.sha256(canonical_manifest_bytes(doc)).hexdigest()


def parse_kv(pairs: list[str], flag: str) -> dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"{flag} expects name=path, got: {pair}")
        name, path = pair.split("=", 1)
        out[name] = path
    return out


def build_manifest(args: argparse.Namespace) -> dict:
    components: list[dict] = []
    for name, path in parse_kv(args.tree, "--tree").items():
        components.append(
            {
                "name": name,
                "verification": {
                    "mode": "tree",
                    "root": path,
                    "files": hash_tree(path),
                },
            }
        )

    if args.patch_log:
        components.append(
            {
                "name": "patch-overlay",
                "verification": {
                    "mode": "patched_files",
                    "root": args.patch_root or "",
                    "files": load_patch_log(args.patch_log),
                },
            }
        )

    native_libs = [
        {"path": path, "sha256": sha256_file(path)} for path in args.native_lib
    ]

    doc: dict = {
        "runtime_id": args.runtime_id,
        "components": components,
        "native_libs": native_libs,
        "image": {"digest": args.image_digest},
    }
    if args.lock:
        with open(args.lock, "r", encoding="utf-8") as f:
            doc["lock"] = json.load(f)
    if args.with_packages:
        doc["packages"] = installed_packages()
    doc[SELF_HASH_FIELD] = compute_self_hash(doc)
    return doc


def verify_self(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    recorded = doc.get(SELF_HASH_FIELD)
    if not recorded:
        print(f"FAIL: {path} has no {SELF_HASH_FIELD} field")
        return 1
    computed = compute_self_hash(doc)
    if computed != recorded:
        print(f"FAIL: self-hash mismatch\n  recorded: {recorded}\n  computed: {computed}")
        return 1
    print(f"OK: {path} self-hash verified ({recorded})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-self", metavar="MANIFEST",
                        help="verify an existing manifest's self-hash and exit")
    parser.add_argument("--lock", help="path to runtime-lock.json to embed")
    parser.add_argument("--output", help="output manifest path (default: stdout)")
    parser.add_argument("--runtime-id", default="glm52-gb10-rc1")
    parser.add_argument("--tree", action="append", default=[], metavar="NAME=PATH",
                        help="source tree component to hash (repeatable)")
    parser.add_argument("--native-lib", action="append", default=[], metavar="PATH",
                        help="native library file to hash into native_libs (repeatable)")
    parser.add_argument("--patch-log", metavar="JSONL",
                        help="apply-patches.py log with pre/postimage hashes")
    parser.add_argument("--patch-root", default="",
                        help="root the patch-log target paths are relative to")
    parser.add_argument("--image-digest", default="pending",
                        help="registry digest of the image (default: pending)")
    parser.add_argument("--with-packages", action="store_true",
                        help="embed installed package versions as provenance")
    args = parser.parse_args(argv)

    if args.verify_self:
        return verify_self(args.verify_self)

    doc = build_manifest(args)
    text = json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
