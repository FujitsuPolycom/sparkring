#!/usr/bin/env python3
"""Remove one exact Python distribution after proving module ownership."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "sparkring-python-distribution-removal/v1"


class RemovalError(RuntimeError):
    """The installed module ownership or removal result differs from the receipt."""


def load_receipt(path: Path) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise RemovalError(f"removal receipt must use schema {SCHEMA}")
    expected = {
        "schema",
        "status",
        "module",
        "distribution",
        "version",
        "postcondition",
        "reason",
    }
    if set(document) != expected:
        raise RemovalError("removal receipt fields differ from the contract")
    if document.get("status") != "implemented":
        raise RemovalError("removal receipt status must be implemented")
    if document.get("postcondition") != "module-absent":
        raise RemovalError("removal receipt postcondition must be module-absent")
    for name in ("module", "distribution", "version", "reason"):
        if not isinstance(document.get(name), str) or not document[name]:
            raise RemovalError(f"removal receipt {name} must be a non-empty string")
    return document


def verify_unique_owner(receipt: dict[str, str]) -> None:
    module = receipt["module"]
    distribution = receipt["distribution"]
    owners = importlib.metadata.packages_distributions().get(module) or []
    if owners != [distribution]:
        raise RemovalError(
            f"module {module} must have exactly one owner {distribution}; got {owners}"
        )
    installed = importlib.metadata.distribution(distribution)
    observed_name = installed.metadata.get("Name")
    if observed_name != distribution or installed.version != receipt["version"]:
        raise RemovalError(
            f"distribution identity differs: expected {distribution}=={receipt['version']}, "
            f"got {observed_name}=={installed.version}"
        )
    if importlib.util.find_spec(module) is None:
        raise RemovalError(f"owned module is not importable before removal: {module}")


def verify_absent(receipt: dict[str, str]) -> None:
    importlib.invalidate_caches()
    module = receipt["module"]
    distribution = receipt["distribution"]
    if importlib.util.find_spec(module) is not None:
        raise RemovalError(f"module remains importable after removal: {module}")
    owners = importlib.metadata.packages_distributions().get(module) or []
    if owners:
        raise RemovalError(f"module ownership remains after removal: {owners}")
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return
    raise RemovalError(f"distribution metadata remains after removal: {distribution}")


def remove_distribution(receipt: dict[str, str]) -> dict[str, Any]:
    verify_unique_owner(receipt)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "--yes",
            receipt["distribution"],
        ],
        check=True,
    )
    verify_absent(receipt)
    return {
        "schema": SCHEMA,
        "module": receipt["module"],
        "distribution": receipt["distribution"],
        "version": receipt["version"],
        "postcondition": "module-absent",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = remove_distribution(load_receipt(args.receipt))
    except (
        OSError,
        json.JSONDecodeError,
        RemovalError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
