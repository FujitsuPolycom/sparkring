#!/usr/bin/env python3
"""Capture exact installed files for every non-inherited locked wheel."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path


INHERITED: set[str] = set()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closure = json.loads(args.closure.read_text())["downloads"]
    files = {}
    distributions = {}
    for raw_name, expected in sorted(closure.items()):
        name = raw_name.lower().replace("_", "-")
        if name in INHERITED:
            continue
        distribution = metadata.distribution(raw_name)
        if distribution.version != expected["version"]:
            raise RuntimeError(f"installed version differs: {raw_name}")
        count = 0
        for member in distribution.files or ():
            path = Path(distribution.locate_file(member)).resolve()
            if path.suffix == ".pyc" or "__pycache__" in path.parts or not path.is_file():
                continue
            try:
                relative = path.relative_to("/opt/venv").as_posix()
            except ValueError as error:
                raise RuntimeError(f"locked payload escaped /opt/venv: {raw_name}: {path}") from error
            observed = digest(path)
            previous = files.setdefault(relative, observed)
            if previous != observed:
                raise RuntimeError(f"installed distributions disagree on {relative}")
            count += 1
        distributions[name] = {"version": distribution.version, "files": count}
    result = {
        "schema": "sparkring-r33-installed-python-files/v1",
        "status": "captured-from-offline-locked-wheel-install",
        "distributions": distributions,
        "files": files,
    }
    import transformers
    import xgrammar
    result["xgrammar_transformers5_import"] = {
        "passed": True,
        "transformers_version": transformers.__version__,
        "xgrammar_version": metadata.version("xgrammar"),
        "xgrammar_origin": xgrammar.__file__,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"distributions": len(distributions), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
