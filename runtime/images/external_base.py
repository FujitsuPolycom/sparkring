"""Install and verify declared SparkRing overlays on a pinned external runtime.

The external image owns framework binaries. The composition owns explicit source
edits, integration assets and package RECORD updates. Serving qualification is a
separate receipt; successful installation is not a performance claim.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys


ROOT = Path("/opt/sparkring")
RECEIPT = ROOT / "receipts/external-base-installed.json"
ENTRYPOINT = ROOT / "bin/external-base.py"
PYTHON_ROOT = ROOT / "python"
SITE = "/usr/local/lib/python3.12/dist-packages/"
PACKAGE_NAMES = frozenset(("vllm", "b12x", "sparkcache"))
HOOK_NAMES = frozenset(("sparkring_features.pth", "sparkring_transport.pth", "sparkring_runtime_status.pth"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def owned_path(name):
    path = PurePosixPath(name)
    require(
        path.is_absolute() and str(path) == name
        and ".." not in path.parts and "\\" not in name,
        "Invalid installed path: " + name,
    )
    allowed = any(name.startswith(SITE + package + "/") for package in PACKAGE_NAMES)
    allowed |= name in {SITE + hook for hook in HOOK_NAMES}
    allowed |= name.startswith("/opt/sparkring/")
    allowed |= name.startswith("/opt/local-inference/nccl/lib/")
    require(allowed, "Overlay path is outside its owners: " + name)
    return Path(name)


def writable_path(name):
    path = owned_path(name)
    require(not any(part.is_symlink() for part in (path, *path.parents)),
            "Overlay write traverses a symlink: " + name)
    return path


def inventory_package(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix != ".pyc":
            require(not path.is_symlink(), "Package contains a symlink: " + str(path))
            result[str(path)] = sha(path)
    return result


def inventory_python_root(root):
    """Inventory every file, including importable standalone bytecode."""
    require(not root.is_symlink(), "Owned Python root is a symlink")
    require(not root.exists() or root.is_dir(), "Owned Python root is not a directory")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Owned Python root contains a symlink: " + str(path))
        if path.is_file():
            result[str(path)] = sha(path)
        else:
            require(path.is_dir(), "Owned Python root contains a special file: " + str(path))
    return result


def admit_python_roots(descriptor):
    roots = descriptor.get("python_roots", {})
    status = descriptor["capabilities"].get("runtime_status")
    require(set(roots) == ({str(PYTHON_ROOT)} if status is not None else set()),
            "Owned Python roots do not match the selected status artifact")
    require(not PYTHON_ROOT.exists(), "Owned Python root already exists")
    changes = descriptor["files"]
    actual = {name: row["after"] for name, row in changes.items()
              if Path(name).is_relative_to(PYTHON_ROOT)}
    expected = {}
    for relative, digest in roots.get(str(PYTHON_ROOT), {}).items():
        path = PurePosixPath(relative)
        require(path.parts and str(path) == relative and not path.is_absolute()
                and ".." not in path.parts and "\\" not in relative,
                "Invalid owned Python file path")
        require(path.parts[0] in {"sparkring_runtime_status", f"sparkring_runtime_status-{status['version']}.dist-info"},
                "Owned Python file has an unexpected owner")
        require(not (path.suffix in {".pyc", ".pth", ".so", ".dll", ".pyd", ".a", ".o"}
                     or ".so." in path.name), "Owned Python payload must be pure source and metadata")
        expected[str(PYTHON_ROOT / relative)] = digest
    require(actual == expected and (status is None or bool(expected)),
            "Owned Python inventory differs from its declared payload")
    hook = SITE + "sparkring_runtime_status.pth"
    if status is None:
        require(hook not in changes, "Status path hook requires a selected artifact")
    else:
        require(status.get("python_root") == str(PYTHON_ROOT)
                and changes.get(hook, {}).get("after") == hashlib.sha256((str(PYTHON_ROOT) + "\n").encode()).hexdigest()
                and changes[hook].get("before") is None,
                "Status path hook or Python root differs")
        for group in ("vllm.endpoint_plugins", "vllm.general_plugins"):
            require(not any(entry.name == "sparkring_status" for entry in metadata.entry_points(group=group)),
                    "External base already declares the status plugin: " + group)
    return roots


def record_distribution(name, root):
    distribution = metadata.distribution(name)
    records = [item for item in distribution.files or ()
               if str(item).endswith(".dist-info/RECORD")]
    require(len(records) == 1, "Distribution RECORD is ambiguous: " + name)
    relative_record = str(records[0])
    record = Path(distribution.locate_file(records[0]))
    require(record.parent.parent == root.parent and not record.is_symlink(),
            "Distribution RECORD is outside its Python site")
    with record.open(newline="") as stream:
        rows = [row for row in csv.reader(stream)
                if not row[0].startswith(name + "/") and row[0] != relative_record]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        digest = base64.urlsafe_b64encode(bytes.fromhex(sha(path))).rstrip(b"=").decode()
        rows.append((name + "/" + path.relative_to(root).as_posix(),
                     "sha256=" + digest, str(path.stat().st_size)))
    rows.append((relative_record, "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return str(record), sha(record)


def install(context):
    require(platform.system() == "Linux" and platform.machine() == "aarch64",
            "This composition requires Linux ARM64")
    require(not RECEIPT.exists(), "An external composition is already installed")
    descriptor = read(context / "composition.json")
    require(descriptor.get("schema") == "sparkring-external-composition/v1",
            "Unknown composition schema")
    require(descriptor.get("platform") == "linux/arm64", "Platform differs")
    require(sha(Path(__file__)) == descriptor["installer_sha256"], "Installer differs")
    baseline = read(context / "base-inventory.json")
    require(sha(context / "base-inventory.json") == descriptor["base_inventory_sha256"],
            "External image inventory differs")
    for name, expected in baseline["versions"].items():
        require(metadata.version(name) == expected, "External package version differs: " + name)
    native = {}
    for name in ("vllm", "b12x"):
        package = baseline["packages"][name]
        root = Path(package["root"])
        require(str(root) == SITE + name, "Unexpected external package root")
        actual = inventory_package(root)
        expected = {str(root / relative): row["sha256"]
                    for relative, row in package["files"].items()}
        require(actual == expected, "External package inventory differs: " + name)
        native.update({path: digest for path, digest in expected.items()
                       if ".so" in Path(path).name})
    changes = descriptor["files"]
    python_roots = admit_python_roots(descriptor)
    for name, row in changes.items():
        path = writable_path(name)
        require(str(path) not in native, "Framework native payload cannot be replaced")
        framework = any(name.startswith(SITE + package + "/")
                        for package in ("vllm", "b12x"))
        require(not (framework and (
            path.name.endswith((".so", ".dll", ".pyd", ".a", ".o"))
            or ".so." in path.name)), "Source overlay cannot add framework binaries")
        require((sha(path) if path.exists() else None) == row["before"],
                "Overlay preimage differs: " + name)
        if row["after"] is not None:
            relative = PurePosixPath(row["payload"])
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "Invalid payload path")
            payload = context / "payload" / str(relative)
            require(not payload.is_symlink() and sha(payload) == row["after"],
                    "Overlay payload differs: " + name)
    for name, target in descriptor.get("symlinks", {}).items():
        path = writable_path(name)
        require(not path.exists() and PurePosixPath(target).name == target,
                "Integration symlink is not a new sibling reference")
        destinations = descriptor.get("symlinks", {})
        end = str(path.parent / target)
        seen = {name}
        while end in destinations:
            require(end not in seen, "Integration symlinks contain a cycle")
            seen.add(end)
            end = str(Path(end).parent / destinations[end])
        require(end in changes and changes[end]["after"] is not None,
                "Integration symlink target is not owned")
    for name, row in changes.items():
        path = writable_path(name)
        if row["after"] is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(context / "payload" / row["payload"], path)
            path.chmod(row.get("mode", 0o644))
    for name, target in descriptor.get("symlinks", {}).items():
        path = writable_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    files = {}
    records = {}
    for name in PACKAGE_NAMES:
        root = Path(SITE + name)
        if not root.exists():
            continue
        for bytecode in root.rglob("*.pyc"):
            require(not bytecode.is_symlink(), "Unexpected bytecode symlink")
            bytecode.unlink()
        files.update(inventory_package(root))
        if name in baseline["packages"]:
            record, digest = record_distribution(name, root)
            records[record] = digest
    for name, row in changes.items():
        if row["after"] is not None:
            files[name] = sha(name)
    for name, expected in native.items():
        require(sha(name) == expected, "Framework native payload changed: " + name)
    ENTRYPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, ENTRYPOINT)
    files[str(ENTRYPOINT)] = sha(ENTRYPOINT)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "sparkring-external-installed/v1",
        "base": descriptor["base"], "composition_sha256": sha(context / "composition.json"),
        "sources": descriptor["sources"], "versions": baseline["versions"],
        "files": files, "distribution_records": records, "native_files": native,
        "removed_files": [name for name, row in changes.items() if row["after"] is None],
        "symlinks": descriptor.get("symlinks", {}),
        "capabilities": descriptor["capabilities"], "serving_qualified": False,
        "python_roots": python_roots,
    }
    RECEIPT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return verify()


def verify():
    record = read(RECEIPT)
    require(record.get("schema") == "sparkring-external-installed/v1", "Unknown receipt")
    other_files = dict(record["files"])
    python_roots = record.get("python_roots", {})
    require(set(python_roots) <= {str(PYTHON_ROOT)}, "Unknown owned Python root in receipt")
    expected_python = {str(PYTHON_ROOT / relative): digest
                       for relative, digest in python_roots.get(str(PYTHON_ROOT), {}).items()}
    require(inventory_python_root(PYTHON_ROOT) == expected_python,
            "Installed owned Python inventory differs")
    recorded_python = {name: digest for name, digest in record["files"].items()
                       if Path(name).is_relative_to(PYTHON_ROOT)}
    require(recorded_python == expected_python, "Owned Python files differ between receipt inventories")
    for name in recorded_python:
        other_files.pop(name)
    for package in PACKAGE_NAMES:
        root = Path(SITE + package)
        expected = {name: digest for name, digest in record["files"].items()
                    if Path(name).is_relative_to(root)}
        require(inventory_package(root) == expected,
                "Installed package inventory differs: " + package)
        for name in expected:
            other_files.pop(name)
    for name, expected in {**other_files, **record["distribution_records"]}.items():
        require(sha(name) == expected, "Installed composition differs: " + name)
    for name in record["removed_files"]:
        require(not Path(name).exists(), "Removed source path exists: " + name)
    for name, target in record["symlinks"].items():
        require(Path(name).is_symlink() and os.readlink(name) == target,
                "Integration symlink differs: " + name)
    for name, expected in record["versions"].items():
        require(metadata.version(name) == expected, "Installed dependency differs: " + name)
    return {
        "schema": "sparkring-external-verification/v1",
        "composition_sha256": record["composition_sha256"], "sources": record["sources"],
        "files_verified": len(record["files"]), "framework_native_rebuilt": False,
        "owned_python_roots": sorted(python_roots),
        "serving_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "verify", "serve"))
    parser.add_argument("--context", type=Path)
    args, extra = parser.parse_known_args()
    result = install(args.context) if args.action == "install" else verify()
    print(json.dumps(result), flush=True)
    if args.action == "serve":
        argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *extra]
        os.execve(argv[0], argv, os.environ)


if __name__ == "__main__":
    main()
