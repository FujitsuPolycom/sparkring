"""Protected pytest receipt adapter for source and installed-image gates."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--component", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    # Source-only oracles may need the foundation's compiled extension modules.
    overlay = Path("/tmp/upgrade-oracle-packages")
    package = overlay / args.component
    shutil.copytree(args.source / args.component, package)
    spec = importlib.util.find_spec(args.component)
    if spec and spec.submodule_search_locations:
        installed = Path(next(iter(spec.submodule_search_locations)))
        for original in installed.rglob("*"):
            if original.is_file() and (
                ".so" in original.name or original.name == "_version.py"
            ):
                target = package / original.relative_to(installed)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(original)
    paths = []
    for name in args.tests:
        path = (args.baseline / name).resolve()
        if not path.is_relative_to(args.baseline.resolve()) or not path.is_file():
            raise ValueError("Missing protected baseline test: " + name)
        paths.append(str(path))
    junit = Path("/tmp/upgrade-oracle.xml")
    env = dict(
        os.environ,
        PYTHONPATH=str(overlay),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--import-mode=importlib",
            "-q",
            "--junitxml=" + str(junit),
            *paths,
        ],
        env=env,
    )
    tests = [] if not junit.exists() else list(ET.parse(junit).iter("testcase"))
    skipped = sum(test.find("skipped") is not None for test in tests)
    receipt = {
        "schema": "sparkring-upgrade-gate/v1",
        "gate": os.environ["SPARKRING_UPGRADE_GATE"],
        "subject_sha256": os.environ["SPARKRING_UPGRADE_SUBJECT"],
        "variant": os.environ["SPARKRING_UPGRADE_VARIANT"],
        "input_sha256": os.environ["SPARKRING_UPGRADE_INPUT"],
        "outcome": "passed" if result.returncode == 0 else "failed",
        "assertions": len(tests),
        "skipped": skipped,
    }
    args.result.write_text(json.dumps(receipt), encoding="utf-8")


if __name__ == "__main__":
    main()
