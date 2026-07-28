#!/usr/bin/env python3
"""SparkRing fail-closed runtime startup verifier.

Verifies that the runtime a rank is about to start inside matches the audited,
frozen runtime manifest. Stdlib only; safe to run before any heavy import.

Consumed via:
    SPARKRING_RUNTIME_MANIFEST=/opt/sparkring/runtime-manifest.json \
        python verify-runtime.py [--json] [--expect-runtime-id ID] [--emit-attestation]

Verification order (each is a named check; first hard failure wins, but all
checks are executed so --json reports the full picture):

  1. manifest_self_hash   - sha256 over the canonical serialization of the
                            manifest with the "self_hash" field removed must
                            equal the recorded "self_hash".
  2. installed_source     - each component's installed files must match the
                            manifest. Two modes per component:
                              "tree":          {relpath: sha256} over a root
                                               (full tree or sampled subset).
                              "patched_files": {relpath: {"preimage": h,
                                               "postimage": h}} - the
                                               POSTIMAGE must match what is
                                               installed (preimage is
                                               provenance metadata only).
  3. native_libs          - sha256 of each listed native library file.
  4. image_digest         - compared against SPARKRING_IMAGE_DIGEST injected
                            by the launcher. A container cannot always see its
                            own image digest, so when the env var is absent
                            this check SKIPS WITH A LOUD WARNING instead of
                            failing; when present, mismatch is a hard failure.
                            A manifest digest of "pending" (pre-first-build
                            lock state) also skips with a warning.
  5. runtime_id           - --expect-runtime-id compares against the
                            manifest's runtime_id (cross-rank identity).
                            --emit-attestation prints a single JSON line
                            {"runtime_id": ..., "manifest_self_hash": ...} to
                            stdout so the launcher can compare across ranks.

Exit codes: 0 ok, 2 verification failure, 3 manifest unreadable/unparseable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

MANIFEST_ENV = "SPARKRING_RUNTIME_MANIFEST"
IMAGE_DIGEST_ENV = "SPARKRING_IMAGE_DIGEST"

EXIT_OK = 0
EXIT_VERIFY_FAILED = 2
EXIT_MANIFEST_UNREADABLE = 3

_CHUNK = 1024 * 1024


class ManifestError(Exception):
    """Manifest missing, unreadable, or not valid JSON (exit 3)."""


class CheckResult:
    __slots__ = ("name", "status", "detail")

    def __init__(self, name: str, status: str, detail: str = "") -> None:
        # status: "pass" | "fail" | "skip"
        self.name = name
        self.status = status
        self.detail = detail

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_manifest_bytes(manifest: dict) -> bytes:
    """Canonical serialization used for the self-hash: the manifest with
    'self_hash' removed, dumped with sorted keys and compact separators."""
    stripped = {k: v for k, v in manifest.items() if k != "self_hash"}
    return json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def manifest_self_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def load_manifest(path: str) -> dict:
    if not path:
        raise ManifestError(
            f"{MANIFEST_ENV} is not set and no --manifest given; refusing to start"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path!r}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestError(f"manifest {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"manifest {path!r} top level must be a JSON object")
    return manifest


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_self_hash(manifest: dict) -> CheckResult:
    name = "manifest_self_hash"
    recorded = manifest.get("self_hash")
    if not isinstance(recorded, str) or not recorded:
        return CheckResult(name, "fail", "manifest has no 'self_hash' field")
    actual = manifest_self_hash(manifest)
    if actual != recorded:
        return CheckResult(
            name,
            "fail",
            f"manifest self-hash mismatch: recorded {recorded} != computed {actual}"
            " (manifest was modified after freezing)",
        )
    return CheckResult(name, "pass", f"self_hash {recorded} verified")


def _verify_hash_map(root: str, files: dict, label: str) -> list:
    """Return list of error strings for {relpath: expected_sha256} under root."""
    errors = []
    for relpath in sorted(files):
        expected = files[relpath]
        full = os.path.join(root, relpath) if root else relpath
        if not os.path.isfile(full):
            errors.append(f"{label}: missing file {full!r}")
            continue
        actual = _sha256_file(full)
        if actual != expected:
            errors.append(
                f"{label}: {full!r} sha256 {actual} != expected {expected}"
            )
    return errors


def check_installed_source(manifest: dict) -> CheckResult:
    name = "installed_source"
    components = manifest.get("components", [])
    if not isinstance(components, list):
        return CheckResult(name, "fail", "'components' must be a list")
    errors = []
    checked = 0
    for comp in components:
        cname = comp.get("name", "<unnamed>")
        ver = comp.get("verification")
        if not isinstance(ver, dict):
            errors.append(f"component {cname!r}: missing 'verification' block")
            continue
        mode = ver.get("mode")
        root = ver.get("root", "")
        files = ver.get("files", {})
        if mode == "tree":
            errors.extend(_verify_hash_map(root, files, f"component {cname!r}"))
            checked += len(files)
        elif mode == "patched_files":
            for relpath in sorted(files):
                entry = files[relpath]
                post = entry.get("postimage") if isinstance(entry, dict) else None
                if not post:
                    errors.append(
                        f"component {cname!r}: {relpath!r} has no 'postimage' hash"
                    )
                    continue
                full = os.path.join(root, relpath) if root else relpath
                if not os.path.isfile(full):
                    errors.append(f"component {cname!r}: missing file {full!r}")
                    continue
                actual = _sha256_file(full)
                if actual != post:
                    errors.append(
                        f"component {cname!r}: {full!r} sha256 {actual} != "
                        f"expected postimage {post}"
                    )
                checked += 1
        else:
            errors.append(
                f"component {cname!r}: unknown verification mode {mode!r} "
                "(expected 'tree' or 'patched_files')"
            )
    if errors:
        return CheckResult(name, "fail", "; ".join(errors))
    return CheckResult(name, "pass", f"{checked} source file(s) verified")


def check_native_libs(manifest: dict) -> CheckResult:
    name = "native_libs"
    libs = manifest.get("native_libs", [])
    if not isinstance(libs, list):
        return CheckResult(name, "fail", "'native_libs' must be a list")
    errors = []
    for lib in libs:
        path = lib.get("path")
        expected = lib.get("sha256")
        if not path or not expected:
            errors.append(f"native lib entry {lib!r} needs 'path' and 'sha256'")
            continue
        if not os.path.isfile(path):
            errors.append(f"native lib missing: {path!r}")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            errors.append(
                f"native lib {path!r} sha256 {actual} != expected {expected}"
            )
    if errors:
        return CheckResult(name, "fail", "; ".join(errors))
    return CheckResult(name, "pass", f"{len(libs)} native lib(s) verified")


def check_image_digest(manifest: dict, env: dict) -> CheckResult:
    name = "image_digest"
    image = manifest.get("image", {}) or {}
    expected = image.get("digest")
    observed = env.get(IMAGE_DIGEST_ENV)
    if expected in (None, "", "pending"):
        return CheckResult(
            name,
            "skip",
            "WARNING: manifest image digest is pending (pre-first-build lock); "
            "image identity NOT verified",
        )
    if not observed:
        return CheckResult(
            name,
            "skip",
            f"WARNING: {IMAGE_DIGEST_ENV} not injected by launcher; a container "
            "cannot always discover its own image digest, so this check is "
            "SKIPPED - image identity NOT verified",
        )
    if observed != expected:
        return CheckResult(
            name,
            "fail",
            f"image digest mismatch: running {observed} != manifest {expected}",
        )
    return CheckResult(name, "pass", f"image digest {expected} verified")


def check_runtime_id(manifest: dict, expected_id) -> CheckResult:
    name = "runtime_id"
    rid = manifest.get("runtime_id")
    if not isinstance(rid, str) or not rid:
        return CheckResult(name, "fail", "manifest has no 'runtime_id'")
    if expected_id is None:
        return CheckResult(name, "pass", f"runtime_id {rid} (no expectation given)")
    if rid != expected_id:
        return CheckResult(
            name,
            "fail",
            f"runtime_id mismatch: manifest {rid!r} != expected {expected_id!r} "
            "(ranks are not running the same frozen runtime)",
        )
    return CheckResult(name, "pass", f"runtime_id {rid} matches expectation")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_checks(manifest: dict, expect_runtime_id=None, env=None) -> list:
    env = os.environ if env is None else env
    return [
        check_self_hash(manifest),
        check_installed_source(manifest),
        check_native_libs(manifest),
        check_image_digest(manifest, env),
        check_runtime_id(manifest, expect_runtime_id),
    ]


def attestation_line(manifest: dict) -> str:
    return json.dumps(
        {
            "runtime_id": manifest.get("runtime_id"),
            "manifest_self_hash": manifest.get("self_hash"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify-runtime", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--manifest",
        default=os.environ.get(MANIFEST_ENV, ""),
        help=f"manifest path (default: ${MANIFEST_ENV})",
    )
    parser.add_argument(
        "--expect-runtime-id",
        default=None,
        help="fail unless the manifest runtime_id equals this value",
    )
    parser.add_argument(
        "--emit-attestation",
        action="store_true",
        help="print one JSON line {runtime_id, manifest_self_hash} to stdout "
        "for cross-rank comparison by the launcher",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable check results"
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "exit_code": 3}))
        else:
            print(f"verify-runtime: FATAL: {exc}", file=sys.stderr)
        return EXIT_MANIFEST_UNREADABLE

    results = run_checks(manifest, expect_runtime_id=args.expect_runtime_id)
    failed = [r for r in results if r.status == "fail"]
    ok = not failed

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "exit_code": EXIT_OK if ok else EXIT_VERIFY_FAILED,
                    "checks": [r.as_dict() for r in results],
                },
                indent=2,
            )
        )
    else:
        for r in results:
            tag = {"pass": "OK  ", "fail": "FAIL", "skip": "SKIP"}[r.status]
            stream = sys.stderr if r.status != "pass" else sys.stdout
            print(f"verify-runtime: [{tag}] {r.name}: {r.detail}", file=stream)

    if args.emit_attestation and ok:
        print(attestation_line(manifest))

    return EXIT_OK if ok else EXIT_VERIFY_FAILED


if __name__ == "__main__":
    sys.exit(main())
