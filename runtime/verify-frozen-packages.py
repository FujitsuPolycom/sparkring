#!/usr/bin/env python3
"""Verify the final environment against the audited distribution freeze.

Exact ``name==version`` entries must match. Reviewed local-source entries must
be installed, except ``instanttensor``: it belongs to the disabled reference
loader and is intentionally absent from the public runtime.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import re
import sys


INTENTIONALLY_ABSENT = {"instanttensor"}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def expected_packages(text: str) -> tuple[dict[str, str], set[str]]:
    exact: dict[str, str] = {}
    present: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            exact[canonical(name)] = version
        elif " @ file://" in line:
            name = canonical(line.split("@", 1)[0].strip())
            if name not in INTENTIONALLY_ABSENT:
                present.add(name)
    return exact, present


def installed_packages(path: Path | None) -> dict[str, str]:
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {canonical(str(name)): str(version) for name, version in raw.items()}
    result: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result[canonical(name)] = dist.version
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--installed-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        exact, present = expected_packages(
            args.freeze.read_text(encoding="utf-8")
        )
        installed = installed_packages(args.installed_json)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    failures: list[str] = []
    for name, expected in sorted(exact.items()):
        observed = installed.get(name)
        if observed is None:
            failures.append(f"{name}: missing (frozen {expected})")
        elif observed != expected:
            failures.append(f"{name}: installed {observed} != frozen {expected}")
    for name in sorted(present):
        if name not in installed:
            failures.append(f"{name}: source-provided distribution is missing")

    if failures:
        for failure in failures:
            print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    print(
        f"verified {len(exact)} exact package version(s) and "
        f"{len(present)} source-provided distribution(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
