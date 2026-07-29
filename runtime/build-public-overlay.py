#!/usr/bin/env python3
"""Build the reviewable public SparkRing Python overlay bundle.

Only files explicitly named in ``public-overlay-files.json`` are admitted.
The output contains a content manifest so the runtime manifest can attest the
exact bundle copied into the serving image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

SCHEMA = "sparkring-public-overlay/v1"
MANIFEST = "sparkring-overlay-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    parts = path.parts
    integration = ("spark_transport", "integrations", "vllm")
    experiments = ("spark_transport", "experiments")
    if parts[:3] == integration and len(parts) == 4:
        return PurePosixPath(parts[-1])
    if parts[:2] == experiments and len(parts) == 4:
        return PurePosixPath(parts[2], parts[3])
    raise ValueError(f"unsupported public-overlay source layout: {relative}")


def build(repo: Path, spec_path: Path, output: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if set(spec) != {"schema", "files"} or spec.get("schema") != SCHEMA:
        raise ValueError(f"{spec_path}: expected exact {SCHEMA} schema")
    files = spec.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{spec_path}: files must be a non-empty list")
    if len(files) != len(set(files)):
        raise ValueError(f"{spec_path}: duplicate source path")

    records: list[dict[str, str]] = []
    destinations: set[str] = set()
    output.mkdir(parents=True, exist_ok=False)
    for relative in files:
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{spec_path}: every file must be a non-empty string")
        source = (repo / Path(relative)).resolve()
        try:
            source.relative_to(repo.resolve())
        except ValueError as exc:
            raise ValueError(f"source escapes repository: {relative}") from exc
        if not source.is_file():
            raise ValueError(f"public-overlay source missing: {relative}")
        target_relative = destination(relative)
        target_key = target_relative.as_posix()
        if target_key in destinations:
            raise ValueError(f"duplicate output path: {target_key}")
        destinations.add(target_key)
        target = output / Path(*target_relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append(
            {
                "source": relative,
                "path": target_key,
                "sha256": sha256_file(target),
            }
        )

    manifest = {"schema": SCHEMA, "files": records}
    (output / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.repo, args.spec, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"public overlay bundled: {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
