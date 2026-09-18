"""Protected pytest receipt adapter for source and installed-image gates."""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def protected_test_path(baseline, name):
    filename, *selectors = name.split("::")
    path = (Path(baseline) / filename).resolve()
    if not path.is_relative_to(Path(baseline).resolve()) or not path.is_file():
        raise ValueError("Missing protected baseline test: " + name)
    if any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", selector) for selector in selectors
    ):
        raise ValueError("Protected test selector must name a test/class explicitly")
    return "::".join((str(path), *selectors))


def oracle_environment(source, overlay, baseline):
    """Point runtime-aware protected tests at the selected source variant."""
    return dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(overlay), str(baseline))),
        XDG_CACHE_HOME="/tmp/upgrade-oracle-cache",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SPARKRING_TEST_SOURCE_ROOT=str(source),
    )


def bind_test_roots(items, selected, baseline, source):
    """Bind explicitly declared AST source probes to the runtime under test."""
    modules = {str(Path(item.module.__file__).resolve()): item.module for item in items}
    for path in selected:
        module = modules.get(str(Path(path).resolve()))
        if (
            module is None
            or Path(getattr(module, "ROOT", "")).resolve() != Path(baseline).resolve()
        ):
            raise ValueError(
                "Protected AST probe has no declared baseline ROOT: " + path
            )
        module.ROOT = Path(source)


def pytest_collection_modifyitems(items):
    selected = json.loads(os.environ.get("SPARKRING_AST_SOURCE_TESTS", "[]"))
    if selected:
        bind_test_roots(
            items,
            selected,
            os.environ["SPARKRING_ORACLE_BASELINE"],
            os.environ["SPARKRING_TEST_SOURCE_ROOT"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--component", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--source-root-test", action="append", default=[])
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
        paths.append(protected_test_path(args.baseline, name))
    junit = Path("/tmp/upgrade-oracle.xml")
    env = oracle_environment(args.source, overlay, args.baseline)
    for name in args.source_root_test:
        if name not in {value.split("::")[0] for value in args.tests}:
            raise ValueError("AST source-root probe must also be a selected test")
    root_tests = [
        protected_test_path(args.baseline, name) for name in args.source_root_test
    ]
    env["SPARKRING_AST_SOURCE_TESTS"] = json.dumps(root_tests)
    env["SPARKRING_ORACLE_BASELINE"] = str(args.baseline)
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env["PYTHONPATH"]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_gate",
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
