#!/usr/bin/env python3
"""Classify a vLLM tree against SparkRing's ordered patch series.

The input tree is never modified. Target files are copied to a temporary
working tree, then each patch is classified and applied there in order:

* apply_exact: current hash equals the published preimage;
* already_applied: reverse-apply succeeds;
* apply_rebased: forward-apply succeeds despite unrelated source drift;
* conflict: neither direction applies cleanly.

Additions are similarly classified as add_exact, already_added, or conflict.
The JSON report is suitable for reviewing a base-specific fail-closed plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_apply(root: Path, patch: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *flags, str(patch.resolve())],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(
    site_packages: Path,
    patches_root: Path,
    snapshot_dir: Path | None = None,
) -> dict:
    components = [
        path
        for path in sorted(patches_root.iterdir())
        if path.is_dir()
        and (
            any(path.glob("*.patch"))
            or (path / "additions.json").is_file()
        )
    ]

    target_paths: set[str] = set()
    for component in components:
        preimages = component / "preimages.json"
        if preimages.is_file():
            for record in load_json(preimages).values():
                target_paths.add(record["target_path"])
        additions = component / "additions.json"
        if additions.is_file():
            for record in load_json(additions).values():
                target_paths.add(record["target_path"])

    if snapshot_dir is not None:
        for relative in sorted(target_paths):
            source = site_packages / relative
            if source.is_file():
                target = snapshot_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="sparkring-classify-") as raw:
        work = Path(raw)
        for relative in sorted(target_paths):
            source = site_packages / relative
            if source.is_file():
                target = work / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        for component in components:
            preimage_path = component / "preimages.json"
            preimages = load_json(preimage_path) if preimage_path.is_file() else {}
            for patch in sorted(component.glob("*.patch")):
                expected = preimages[patch.name]
                relative = expected["target_path"]
                target = work / relative
                record = {
                    "component": component.name,
                    "kind": "patch",
                    "name": patch.name,
                    "target_path": relative,
                    "published_preimage_sha256": expected["preimage_sha256"],
                }
                if not target.is_file():
                    record.update(action="conflict", detail="target missing")
                    records.append(record)
                    continue

                before = sha256(target)
                record["base_or_prior_sha256"] = before
                if before == expected["preimage_sha256"]:
                    result = git_apply(work, patch, "--whitespace=nowarn")
                    action = "apply_exact" if result.returncode == 0 else "conflict"
                else:
                    reverse = git_apply(work, patch, "--reverse", "--check")
                    if reverse.returncode == 0:
                        action = "already_applied"
                        result = reverse
                    else:
                        forward = git_apply(work, patch, "--check")
                        if forward.returncode == 0:
                            result = git_apply(work, patch, "--whitespace=nowarn")
                            action = (
                                "apply_rebased"
                                if result.returncode == 0
                                else "conflict"
                            )
                        else:
                            action = "conflict"
                            result = forward
                record["action"] = action
                record["detail"] = result.stderr.strip()
                record["result_sha256"] = sha256(target)
                records.append(record)

            additions_path = component / "additions.json"
            if not additions_path.is_file():
                continue
            for source_rel, expected in sorted(load_json(additions_path).items()):
                source = component / "added" / source_rel
                relative = expected["target_path"]
                target = work / relative
                source_hash = sha256(source)
                record = {
                    "component": component.name,
                    "kind": "addition",
                    "name": source_rel,
                    "target_path": relative,
                    "published_source_sha256": expected["sha256"],
                }
                if source_hash != expected["sha256"]:
                    record.update(action="conflict", detail="addition source hash drift")
                elif not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    record.update(action="add_exact", detail="")
                elif sha256(target) == source_hash:
                    record.update(action="already_added", detail="")
                else:
                    record.update(
                        action="conflict",
                        detail="addition target exists with different content",
                    )
                if target.is_file():
                    record["result_sha256"] = sha256(target)
                records.append(record)

    counts: dict[str, int] = {}
    for record in records:
        action = record["action"]
        counts[action] = counts.get(action, 0) + 1
    return {
        "schema": "sparkring-faststart-base-classification/v1",
        "site_packages": str(site_packages),
        "patches_root": str(patches_root),
        "counts": dict(sorted(counts.items())),
        "passed": counts.get("conflict", 0) == 0,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--patches-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="optionally copy every base target file here for review",
    )
    args = parser.parse_args()
    report = classify(
        args.site_packages.resolve(),
        args.patches_root.resolve(),
        args.snapshot_dir.resolve() if args.snapshot_dir else None,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(
        "PASS" if report["passed"] else "FAIL",
        report["counts"],
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
