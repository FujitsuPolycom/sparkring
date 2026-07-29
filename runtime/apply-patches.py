#!/usr/bin/env python3
"""Fail-closed patch overlay applier for the SparkRing runtime build.

Reads runtime/patches/<component>/ directories. Each component directory that
contains *.patch files MUST also contain a preimages.json of the form:

    { "<patch_filename>": { "target_path": "<relpath under --site-packages>",
                            "preimage_sha256": "<sha256 of the target BEFORE>" } }

For every patch, in sorted order per component:
  1. the target file's sha256 must equal the recorded preimage (else abort),
  2. the patch is applied with `git apply --whitespace=nowarn` (no fuzz),
  3. {patch, target, preimage_sha256, postimage_sha256} is appended as one
     JSON line to the --log file.

Any mismatch, missing file, missing preimages.json entry, or git-apply
failure aborts the whole build with a non-zero exit. A component directory
may also contain an ``additions.json`` mapping source files under its
``added/`` directory to new paths under ``--site-packages``. Additions are
content-addressed, may not overwrite an existing file, and are installed only
after every patch in that component succeeds. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

_CHUNK = 1 << 20


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fatal(msg: str) -> None:
    print(f"apply-patches: FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def install_additions(comp_dir: str, site_packages: str, log_fh) -> int:
    additions_path = os.path.join(comp_dir, "additions.json")
    if not os.path.isfile(additions_path):
        return 0
    with open(additions_path, "r", encoding="utf-8") as f:
        additions = json.load(f)
    if not isinstance(additions, dict):
        fatal(f"{additions_path} must contain a JSON object")

    installed = 0
    site_root = os.path.realpath(site_packages)
    for source_rel, entry in sorted(additions.items()):
        if not isinstance(entry, dict):
            fatal(f"additions.json entry for {source_rel!r} must be an object")
        target_rel = entry.get("target_path")
        expected_sha = entry.get("sha256")
        if not isinstance(target_rel, str) or not isinstance(expected_sha, str):
            fatal(
                f"additions.json entry for {source_rel!r} needs "
                "'target_path' and 'sha256'"
            )
        source = os.path.realpath(os.path.join(comp_dir, "added", source_rel))
        added_root = os.path.realpath(os.path.join(comp_dir, "added"))
        if os.path.commonpath((source, added_root)) != added_root:
            fatal(f"addition source escapes added directory: {source_rel}")
        if not os.path.isfile(source):
            fatal(f"addition source missing: {source_rel}")
        actual_sha = sha256_file(source)
        if actual_sha != expected_sha:
            fatal(
                f"addition hash mismatch for {source_rel}: "
                f"{actual_sha} != expected {expected_sha}"
            )

        target = os.path.realpath(os.path.join(site_packages, target_rel))
        if os.path.commonpath((target, site_root)) != site_root:
            fatal(f"addition target escapes site-packages: {target_rel}")
        if os.path.exists(target):
            fatal(f"addition refuses to overwrite existing path: {target_rel}")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
        record = {
            "addition": source_rel,
            "target": target_rel,
            "sha256": expected_sha,
        }
        log_fh.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"apply-patches: added {source_rel} -> {target_rel}")
        installed += 1
    return installed


def apply_component(comp_dir: str, site_packages: str, log_fh) -> int:
    patches = sorted(f for f in os.listdir(comp_dir) if f.endswith(".patch"))
    if not patches:
        additions = install_additions(comp_dir, site_packages, log_fh)
        if not additions:
            print(f"apply-patches: {comp_dir}: no patches (verified no-op)")
        return additions
    preimages_path = os.path.join(comp_dir, "preimages.json")
    if not os.path.isfile(preimages_path):
        fatal(f"{comp_dir} has patches but no preimages.json")
    with open(preimages_path, "r", encoding="utf-8") as f:
        preimages = json.load(f)

    applied = 0
    for patch_name in patches:
        entry = preimages.get(patch_name)
        if not isinstance(entry, dict):
            fatal(f"preimages.json has no entry for {patch_name!r}")
        target_rel = entry.get("target_path")
        expected_pre = entry.get("preimage_sha256")
        if not target_rel or not expected_pre:
            fatal(
                f"preimages.json entry for {patch_name!r} needs "
                "'target_path' and 'preimage_sha256'"
            )
        target = os.path.join(site_packages, target_rel)
        if not os.path.isfile(target):
            fatal(f"{patch_name}: target missing: {target}")
        actual_pre = sha256_file(target)
        if actual_pre != expected_pre:
            fatal(
                f"{patch_name}: preimage mismatch for {target_rel}: "
                f"{actual_pre} != expected {expected_pre}"
            )

        patch_path = os.path.join(comp_dir, patch_name)
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", os.path.abspath(patch_path)],
            cwd=site_packages,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            fatal(f"{patch_name}: git apply failed:\n{proc.stderr.strip()}")

        record = {
            "patch": patch_name,
            "target": target_rel,
            "preimage_sha256": expected_pre,
            "postimage_sha256": sha256_file(target),
        }
        log_fh.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"apply-patches: applied {patch_name} -> {target_rel}")
        applied += 1
    return applied + install_additions(comp_dir, site_packages, log_fh)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--patches-root",
        required=True,
        help="root dir containing per-component patch dirs",
    )
    parser.add_argument(
        "--site-packages",
        required=True,
        help="installed tree that patch target paths are relative to",
    )
    parser.add_argument(
        "--log",
        default="/var/log/sparkring/patch-apply.jsonl",
        help="JSONL log of applied patches",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="accepted for explicitness; this tool is always fail-closed",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.patches_root):
        fatal(f"patches root missing: {args.patches_root}")
    if not os.path.isdir(args.site_packages):
        fatal(f"site-packages missing: {args.site_packages}")

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    total = 0
    with open(args.log, "w", encoding="utf-8") as log_fh:
        for comp in sorted(os.listdir(args.patches_root)):
            comp_dir = os.path.join(args.patches_root, comp)
            if os.path.isdir(comp_dir):
                total += apply_component(comp_dir, args.site_packages, log_fh)
    print(f"apply-patches: done, {total} patch(es) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
