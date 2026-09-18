"""Run hash-bound CPU oracles without importing a source checkout's conftest.

Common behavior checks run on every source variant. Interface-specific checks
are reported separately and cannot substitute for the common assertions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_paths(suite_path, variant):
    suite_path = Path(suite_path).resolve()
    root = suite_path.parent
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != "sparkring-protected-source-suite/v1":
        raise ValueError("Unknown protected source suite")
    if suite.get("component") not in ("vllm", "b12x"):
        raise ValueError("Unsupported protected source component")
    if variant not in ("baseline", "upstream", "candidate"):
        raise ValueError("Unknown source variant")
    if not suite.get("common") or not suite.get("files"):
        raise ValueError(
            "A source suite requires nonempty common tests and an inventory"
        )
    for name, expected in suite["files"].items():
        unresolved = root / name
        path = unresolved.resolve()
        if (
            not path.is_relative_to(root)
            or Path(name).is_absolute()
            or any(part.is_symlink() for part in (unresolved, *unresolved.parents))
            or not path.is_file()
            or digest(path) != expected
        ):
            raise ValueError("Protected source test differs: " + name)
    groups = {"common": suite["common"]}
    scope = "baseline_only" if variant == "baseline" else "candidate_only"
    if suite.get(scope):
        groups[scope] = suite[scope]
    result = {}
    for name, selections in groups.items():
        paths = []
        for selection in selections:
            filename, *selectors = selection.split("::")
            if (
                filename not in suite["files"]
                or not filename.endswith(".py")
                or any(
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item)
                    for item in selectors
                )
            ):
                raise ValueError("Invalid protected test selection: " + selection)
            paths.append("::".join((str(root / filename), *selectors)))
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate protected test selection")
        result[name] = paths
    return suite, result


def source_environment(source, baseline, protected_root):
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    }
    return dict(
        inherited,
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=os.pathsep.join((str(source), str(protected_root))),
        SPARKRING_TEST_SOURCE_ROOT=str(source),
        SPARKRING_B12X_SOURCE_ROOT=str(source),
        SPARKRING_ORACLE_BASELINE=str(baseline),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        NVIDIA_VISIBLE_DEVICES="void",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )


def run(source, baseline, suite_path, result_path, *, variant=None):
    variant = variant or os.environ["SPARKRING_UPGRADE_VARIANT"]
    source, baseline, suite_path = map(
        lambda p: Path(p).resolve(), (source, baseline, suite_path)
    )
    suite, groups = selected_paths(suite_path, variant)
    if (
        not (source / suite["component"]).is_dir()
        or not (baseline / suite["component"]).is_dir()
    ):
        raise ValueError("Selected component source or protected baseline is missing")
    receipts = {}
    with tempfile.TemporaryDirectory(prefix="sparkring-source-oracle-") as temporary:
        temporary = Path(temporary)
        config = temporary / "pytest.ini"
        config.write_text(
            "[pytest]\nmarkers =\n    cpu_test: GPU-free source contract\n"
        )
        env = source_environment(source, baseline, suite_path.parent)
        for scope, paths in groups.items():
            junit = temporary / (scope + ".xml")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--noconftest",
                    "--import-mode=importlib",
                    "-p",
                    "no:cacheprovider",
                    "-c",
                    str(config),
                    "--rootdir",
                    str(suite_path.parent),
                    "--junitxml",
                    str(junit),
                    "-q",
                    *paths,
                ],
                cwd=temporary,
                env=env,
                check=False,
            )
            cases = list(ET.parse(junit).iter("testcase")) if junit.exists() else []
            skipped = sum(item.find("skipped") is not None for item in cases)
            failures = sum(
                item.find("failure") is not None or item.find("error") is not None
                for item in cases
            )
            receipts[scope] = {
                "tests": len(cases),
                "skipped": skipped,
                "failures": failures,
                "pytest_returncode": completed.returncode,
                "outcome": "passed"
                if completed.returncode == 0 and cases and not skipped and not failures
                else "failed",
                "selections": suite[scope],
            }
    # Tests cannot mutate a protected oracle and then claim success.
    selected_paths(suite_path, variant)
    receipt = {
        "schema": "sparkring-upgrade-gate/v1",
        "gate": os.environ["SPARKRING_UPGRADE_GATE"],
        "subject_sha256": os.environ["SPARKRING_UPGRADE_SUBJECT"],
        "variant": variant,
        "input_sha256": os.environ["SPARKRING_UPGRADE_INPUT"],
        "outcome": "passed"
        if all(item["outcome"] == "passed" for item in receipts.values())
        else "failed",
        "assertions": sum(item["tests"] for item in receipts.values()),
        "skipped": sum(item["skipped"] for item in receipts.values()),
        "scopes": receipts,
        "suite_sha256": digest(suite_path),
        "protected_files": suite["files"],
        "scope": "Common CPU behavior plus separately identified API-specific checks; no GPU or serving qualification.",
    }
    Path(result_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "suite", "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.baseline, args.suite, args.result)))
