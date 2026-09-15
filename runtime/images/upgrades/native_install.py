"""Install source-built wheels while preserving the foundation's other payload.

The descriptor and compiler receipt are trusted build inputs. Installation does
not infer serving qualification or rewrite feature/cache compatibility contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SITE = Path("/opt/venv/lib/python3.12/site-packages")
ROOT = Path("/opt/sparkring")
RECEIPT = ROOT / "receipts/candidate-installed.json"
NATIVE = ROOT / "receipts/native-installed.json"
ENTRYPOINT = ROOT / "bin/native-image.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(files):
    for name, expected in files.items():
        path = Path(name)
        require(
            path.is_absolute()
            and ".." not in path.parts
            and path.is_file()
            and sha(path) == expected,
            "Installed payload differs: " + name,
        )


def distribution_files(name):
    result = {}
    distribution = metadata.distribution(name)
    for entry in distribution.files or []:
        path = Path(distribution.locate_file(entry)).resolve()
        require(
            path.is_relative_to(SITE.parent.parent.parent),
            "Distribution RECORD escapes the virtual environment",
        )
        if path.is_file() and path.suffix not in (".pyc", ".pyo"):
            result[str(path)] = sha(path)
    require(result, "Distribution has no installed file inventory: " + name)
    return result


def package_owned(path, package_names, metadata_roots):
    path = Path(path)
    if any(path.is_relative_to(SITE / name) for name in package_names):
        return True
    if any(path.is_relative_to(root) for root in metadata_roots):
        return True
    return path in (Path("/opt/venv/bin/vllm"),)


def selected_boundary_identity(parent):
    """Use an owned versioned cache identity when the foundation declares one."""
    selected = parent.get("cache_extension", {}).get("boundary_runtime")
    if selected is None:
        return ROOT / "contracts/boundary-runtime.json"
    path = Path(selected["path"])
    require(
        path.parent == ROOT / "contracts" and not path.is_symlink(),
        "Selected boundary identity escapes its owner",
    )
    require(
        parent["files"].get(str(path)) == selected["sha256"] == sha(path),
        "Selected boundary identity differs from the installed receipt",
    )
    return path


def install(context):
    context = Path(context)
    descriptor = read(context / "descriptor.json")
    compiler = read(context / "compiler-result.json")
    require(
        descriptor.get("schema") == "sparkring-native-install/v1",
        "Unknown native installation descriptor",
    )
    require(
        compiler.get("schema") == "sparkring-native-wheel-result/v1"
        and compiler["descriptor_sha256"] == descriptor["compiler_descriptor_sha256"]
        and compiler["source_trees"] == descriptor["source_trees"],
        "Compiler receipt does not bind accepted source",
    )
    require(
        sha(RECEIPT) == descriptor["parent_installed_sha256"],
        "Foundation installed receipt differs",
    )
    parent = read(RECEIPT)
    verify_files(parent["files"])
    boundary_path = selected_boundary_identity(parent)
    boundary = read(boundary_path) if boundary_path.exists() else None
    if boundary is not None:
        require(
            boundary.get("schema") == "sparkcache-boundary-runtime/v1",
            "Unknown boundary runtime identity schema",
        )
        verify_files(
            {str(SITE / name): expected for name, expected in boundary["files"].items()}
        )
    require(
        metadata.version("torch") == compiler["torch_version"],
        "Native compiler and runtime Torch ABI differ",
    )
    protected_torch = {
        name: metadata.version(name) for name in ("torch", "torchvision", "torchaudio")
    }
    before = dict(parent["files"])
    packages = ["vllm", "b12x"]
    selected = []
    metadata_roots = []
    for name in packages:
        before.update(distribution_files(name))
        metadata_roots.extend(SITE.glob(name + "-*.dist-info"))
        record = compiler["wheels"][name]
        filename = record["file"]
        require(
            Path(filename).name == filename and filename.endswith(".whl"),
            "Wheel is not a contained context file",
        )
        path = context / "wheels" / filename
        require(sha(path) == record["sha256"], "Compiled wheel bytes differ")
        selected.append(str(path))
    prior_errors = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    ).stdout.splitlines()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            *selected,
        ],
        check=True,
    )
    require(
        {name: metadata.version(name) for name in protected_torch} == protected_torch,
        "Wheel installation changed protected Torch versions",
    )
    after_errors = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    ).stdout.splitlines()
    require(
        set(after_errors) <= set(prior_errors),
        "Native wheel introduces unsatisfied dependencies: "
        + str(sorted(set(after_errors) - set(prior_errors))),
    )
    files, removed = {}, []
    for path, expected in before.items():
        if package_owned(path, packages, metadata_roots):
            if not Path(path).exists():
                removed.append(path)
        else:
            require(
                Path(path).is_file() and sha(path) == expected,
                "Unrelated foundation bytes changed: " + path,
            )
            files[path] = expected
    for name in packages:
        require(
            metadata.version(name) == compiler["wheels"][name]["version"],
            "Installed distribution version differs",
        )
        files.update(distribution_files(name))
        # Include package additions that a retained foundation RECORD omitted.
        for path in (SITE / name).rglob("*"):
            if path.is_file() and path.suffix not in (".pyc", ".pyo"):
                files[str(path)] = sha(path)
    # Optional cache/feature additions remain unchanged and receive explicit hashes.
    for directory in (
        SITE / "sparkcache",
        ROOT / "contracts",
        ROOT / "features",
        ROOT / "transports",
    ):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix not in (".pyc", ".pyo"):
                    files[str(path)] = sha(path)
    for path in SITE.glob("*.pth"):
        files[str(path)] = sha(path)
    ENTRYPOINT.parent.mkdir(parents=True, exist_ok=True)
    ENTRYPOINT.write_bytes(Path(__file__).read_bytes())
    files[str(ENTRYPOINT)] = sha(ENTRYPOINT)
    boundary_record = None
    if boundary is not None:
        # A compiled runtime gets a distinct attestation. This is identity
        # regeneration, not evidence that its connector interfaces are compatible.
        bound_files = {
            str(Path(name).relative_to(SITE)): expected
            for name, expected in files.items()
            if Path(name).is_relative_to(SITE)
            and Path(name).relative_to(SITE).parts[0] in ("vllm", "b12x", "sparkcache")
        }
        boundary_native = ROOT / (
            "contracts/boundary-native-" + descriptor["input_sha256"][:16] + ".json"
        )
        require(
            not boundary_native.exists(),
            "Boundary attestation destination already exists",
        )
        boundary_native.write_text(
            json.dumps(
                {
                    "schema": boundary["schema"],
                    "files": bound_files,
                    "external": boundary["external"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        files[str(boundary_native)] = sha(boundary_native)
        boundary_record = {
            "path": str(boundary_native),
            "sha256": sha(boundary_native),
            "cache_namespace_suffix": descriptor["input_sha256"][:16],
            "serving_qualified": False,
        }
    receipt = {
        "schema": "sparkring-native-installed/v1",
        "input_sha256": descriptor["input_sha256"],
        "parent_image_id": descriptor["parent_image_id"],
        "parent_installed_sha256": descriptor["parent_installed_sha256"],
        "compiler": compiler,
        "files": files,
        "removed_files": removed,
        "versions": {
            **parent["versions"],
            **{name: metadata.version(name) for name in packages},
        },
        "foundation_dependency_exceptions": prior_errors,
        "active_contracts": sorted(
            descriptor.get("active_contracts", parent.get("integration_contracts", {}))
        ),
        "boundary_runtime": boundary_record,
        "serving_qualified": False,
    }
    NATIVE.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return verify()


def verify():
    value = read(NATIVE)
    require(
        value.get("schema") == "sparkring-native-installed/v1" and value.get("files"),
        "Missing native installed inventory",
    )
    verify_files(value["files"])
    for path in value["removed_files"]:
        require(not Path(path).exists(), "Removed package file reappeared: " + path)
    for name, expected in value["versions"].items():
        require(
            metadata.version(name) == expected, "Installed version differs: " + name
        )
    features = {}
    capabilities = ROOT / "features/capabilities.json"
    if capabilities.exists():
        manifest = read(capabilities)
        require(
            manifest.get("schema") == "sparkring-image-capabilities/v1",
            "Unknown feature manifest schema",
        )
        for name, feature in manifest.get("features", {}).items():
            paths = {}
            for relative, expected in feature["files"].items():
                path = (ROOT / "features" / relative).resolve()
                require(
                    path.is_relative_to((ROOT / "features").resolve()),
                    "Feature inventory escapes its directory",
                )
                paths[str(path)] = expected
            require(paths, "Feature has no source inventory")
            verify_files(paths)
            features[name] = paths
    return {
        "schema": "sparkring-native-verification/v1",
        "receipt_sha256": sha(NATIVE),
        "source_trees": value["compiler"]["source_trees"],
        "files_verified": len(value["files"]),
        "input_sha256": value["input_sha256"],
        "serving_qualified": False,
        "features": sorted(features),
    }


def main():
    if sys.argv[1:2] == ["install"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("action")
        parser.add_argument("--context", type=Path, required=True)
        print(json.dumps(install(parser.parse_args().context)))
        return
    if sys.argv[1:2] == ["verify"]:
        print(json.dumps(verify()))
        return
    require(sys.argv[1:2] == ["serve"], "Expected install, verify or serve")
    verify()
    argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *sys.argv[2:]]
    os.execve(argv[0], argv, os.environ)


if __name__ == "__main__":
    main()
