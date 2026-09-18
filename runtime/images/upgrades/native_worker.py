"""Compile pinned vLLM/B12X wheels inside a resource-bounded builder container.

This script is a trusted build adapter, not an agent-proposed command. Source
mounts are read-only; generated files and wheels remain in the owned work mount.
No model weights, Docker socket or host credentials are mounted into the build.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import struct
import time
import zipfile
from email.parser import BytesParser


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def wheel_record(path, expected_name):
    """Validate wheel geometry and metadata before authorizing installation."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Wheel contains duplicate paths")
        for name in names:
            p = PurePosixPath(name.rstrip("/"))
            require(
                not p.is_absolute()
                and ".." not in p.parts
                and "\\" not in name
                and ":" not in name
                and p.parts,
                "Wheel path escapes installation",
            )
            info = archive.getinfo(name)
            require(
                ((info.external_attr >> 16) & 0o170000) != 0o120000,
                "Wheel symlink is not admitted",
            )
        meta_paths = [
            name for name in names if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)
        ]
        require(len(meta_paths) == 1, "Wheel must identify one distribution")
        value = BytesParser().parsebytes(archive.read(meta_paths[0]))

        def normalize(name):
            return re.sub(r"[-_.]+", "-", name).lower()

        require(
            normalize(value["Name"]) == normalize(expected_name),
            "Wheel distribution identity differs",
        )
        package = expected_name.replace("-", "_")
        metadata_root = meta_paths[0].split("/")[0]
        require(
            not any(name.split("/")[0].endswith(".data") for name in names),
            "Wheel data relocations require a separately reviewed installer",
        )
        require(
            not any(
                name.split("/")[0] in ("torch", "torchvision", "torchaudio")
                for name in names
            ),
            "Wheel may not replace the foundation Torch ABI",
        )
        if expected_name in ("vllm", "b12x"):
            require(
                all(name.split("/")[0] in (package, metadata_root) for name in names),
                "Runtime wheel modifies another package",
            )
        native = [name for name in names if re.search(r"\.so(?:\.|$)", name)]
        for name in native:
            head = archive.read(name)[:64]
            require(
                len(head) == 64
                and head[:6] == b"\x7fELF\x02\x01"
                and struct.unpack_from("<H", head, 18)[0] == 183,
                "Native wheel member is not ARM64 ELF: " + name,
            )
        if expected_name == "vllm":
            require(native, "vLLM wheel contains no compiled extensions")
        return {
            "file": path.name,
            "sha256": sha(path),
            "name": value["Name"],
            "version": value["Version"],
            "requires_dist": value.get_all("Requires-Dist", []),
            "native_members": native,
            "native_hashes": {
                name: hashlib.sha256(archive.read(name)).hexdigest() for name in native
            },
        }


def source_digest(root):
    inventory = {}
    for path in sorted(Path(root).rglob("*")):
        name = path.relative_to(root)
        if ".git" in name.parts:
            continue
        require(not path.is_symlink(), "Source snapshot contains a symlink")
        if path.is_file():
            value = {
                "bytes": sha(path),
                "executable": bool(path.stat().st_mode & 0o111),
            }
            inventory[name.as_posix()] = hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_checked(argv, *, cwd=None, env=None):
    print(json.dumps({"command": argv, "cwd": str(cwd) if cwd else None}), flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def cmake_build_evidence(source, expected):
    """Require the generated vLLM CMake configuration to match the recipe."""
    source = Path(source)
    records = []
    for path in sorted(source.rglob("CMakeCache.txt")):
        values = {}
        for line in path.read_text(errors="replace").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
        if values.get("CMAKE_PROJECT_NAME") != "vllm_extensions":
            continue
        require(
            values.get("CMAKE_BUILD_TYPE") == expected,
            "Generated vLLM CMake build mode differs from the recipe",
        )
        records.append(
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": sha(path),
                "build_type": expected,
                "flags": {
                    key: value
                    for key, value in values.items()
                    if key.startswith(("CMAKE_CXX_FLAGS", "CMAKE_CUDA_FLAGS"))
                },
            }
        )
    require(records, "Missing generated vLLM CMake build-mode evidence")
    return records


def build(descriptor_path, source_root, work):
    descriptor = json.loads(Path(descriptor_path).read_text())
    require(
        descriptor.get("schema") == "sparkring-native-wheel-build/v1",
        "Unknown native build descriptor",
    )
    require(
        set(descriptor["sources"]) == {"vllm", "b12x"},
        "The native wheel adapter supports vLLM and B12X",
    )
    require(
        descriptor["architecture"] in ("12.1", "12.1a"),
        "This compiler recipe targets GB10",
    )
    require(
        descriptor.get("build_type") in ("Release", "RelWithDebInfo"),
        "An explicit supported native build_type is required",
    )
    require(
        type(descriptor["jobs"]) is int and 1 <= descriptor["jobs"] <= 20,
        "Invalid native parallelism",
    )
    require(
        re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", descriptor["distribution_version"]),
        "Invalid wheel version",
    )
    require(
        metadata.version("torch") == descriptor["torch_version"],
        "Compiler Torch ABI differs from approved foundation",
    )
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    marker = work / "input-descriptor.sha256"
    identity = sha(descriptor_path)
    if marker.exists():
        require(
            marker.read_text() == identity,
            "Build work belongs to another source/policy identity",
        )
    else:
        require(not any(work.iterdir()), "Build work directory is not empty")
        marker.write_text(identity)
    completed = work / "result.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        require(
            result["descriptor_sha256"] == identity, "Completed build identity differs"
        )
        for record in result["wheels"].values():
            require(
                sha(work / "wheels" / record["file"]) == record["sha256"],
                "Completed wheel changed",
            )
        return result
    source_root = Path(source_root)
    before = {}
    for name, record in descriptor["sources"].items():
        before[name] = source_digest(source_root / name)
        require(
            before[name] == record["tree_sha256"],
            "Read-only accepted source differs: " + name,
        )
        target = work / "source" / name
        if not target.exists():
            shutil.copytree(
                source_root / name, target, ignore=shutil.ignore_patterns(".git")
            )
    wheels = work / "wheels"
    wheels.mkdir(exist_ok=True)
    # Optional offline wheels are operator-pinned inputs, never agent choices.
    dependencies = descriptor.get("dependency_wheels", [])
    if dependencies:
        selected = []
        for item in dependencies:
            filename = item["file"]
            require(
                Path(filename).name == filename and filename.endswith(".whl"),
                "Invalid dependency wheel name",
            )
            path = Path(descriptor_path).parent / "dependencies" / filename
            require(sha(path) == item["sha256"], "Dependency wheel hash differs")
            record = wheel_record(path, item["name"])
            require(
                record["version"] == item["version"]
                and item["name"].lower() not in ("torch", "torchvision", "torchaudio"),
                "Dependency migration changes the foundation Torch ABI",
            )
            selected.append(str(path))
        run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                *selected,
            ]
        )
    env = dict(
        os.environ,
        MAX_JOBS=str(descriptor["jobs"]),
        CMAKE_BUILD_PARALLEL_LEVEL=str(descriptor["jobs"]),
        CMAKE_BUILD_TYPE=descriptor["build_type"],
        NVCC_THREADS="1",
        TORCH_CUDA_ARCH_LIST=descriptor["architecture"],
        VLLM_TARGET_DEVICE="cuda",
        VLLM_USE_PRECOMPILED="0",
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=descriptor["distribution_version"],
        SETUPTOOLS_SCM_PRETEND_VERSION=descriptor["distribution_version"],
        CMAKE_ARGS="-DVLLM_BUILD_CUTLASS_SCALED_MM_C2X=OFF -DFETCHCONTENT_BASE_DIR="
        + str(work / "fetchcontent"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SPARKRING_FEATURES="",
        SPARKRING_TRANSPORT_PROFILE="",
        CUDA_VISIBLE_DEVICES="",
        NVIDIA_VISIBLE_DEVICES="void",
        PYTHONDONTWRITEBYTECODE="1",
        XDG_CACHE_HOME=str(work / "cache"),
        TORCH_EXTENSIONS_DIR=str(work / "torch-extensions"),
        CARGO_HOME=str(work / "cargo"),
        CARGO_BUILD_JOBS=str(descriptor["jobs"]),
        B12X_COMPILE_CACHE_DIR=str(work / "cache/b12x"),
    )
    started = time.time()
    records = {}
    cached = descriptor.get("native_cache")
    if cached:
        require(
            cached["architecture"] == descriptor["architecture"]
            and cached["torch_version"] == descriptor["torch_version"]
            and cached.get("build_type") == descriptor["build_type"],
            "Cached native ABI or build mode differs",
        )
        cached_wheel = Path("/native-cache") / cached["wheel"]["file"]
        require(
            cached_wheel.is_file()
            and not cached_wheel.is_symlink()
            and sha(cached_wheel) == cached["wheel"]["sha256"],
            "Cached native wheel identity differs",
        )
        cached_record = wheel_record(cached_wheel, "vllm")
    for name in ("b12x", "vllm"):
        selected_env = dict(env)
        if cached and name == "vllm":
            selected_env.update(
                VLLM_USE_PRECOMPILED="1",
                VLLM_PRECOMPILED_WHEEL_LOCATION=str(cached_wheel),
            )
        run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-cache-dir",
                "--wheel-dir",
                str(wheels),
                str(work / "source" / name),
            ],
            env=selected_env,
        )
        files = list(wheels.glob(name + "-*.whl"))
        require(len(files) == 1, "Build must produce exactly one wheel: " + name)
        records[name] = wheel_record(files[0], name)
        if cached and name == "vllm":
            require(
                records[name]["native_hashes"] == cached_record["native_hashes"],
                "Repackaged native payload differs from verified compiled bytes",
            )
    for name, expected in before.items():
        require(
            source_digest(source_root / name) == expected,
            "Compiler changed its read-only source inputs",
        )
    mode_evidence = (
        {"source": "verified-native-cache", "build_type": cached["build_type"]}
        if cached
        else {
            "source": "generated-cmake",
            "configurations": cmake_build_evidence(
                work / "source/vllm", descriptor["build_type"]
            ),
        }
    )
    result = {
        "schema": "sparkring-native-wheel-result/v1",
        "descriptor_sha256": identity,
        "input_sha256": descriptor["input_sha256"],
        "source_trees": before,
        "wheels": records,
        "torch_version": metadata.version("torch"),
        "architecture": descriptor["architecture"],
        "build_type": descriptor["build_type"],
        "build_mode_evidence": mode_evidence,
        "jobs": descriptor["jobs"],
        "elapsed_seconds": time.time() - started,
        "serving_qualified": False,
        "native_rebuilt": not bool(cached),
        "native_cache": cached,
        "native_inputs": descriptor.get("native_inputs"),
    }
    with (work / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.descriptor, args.sources, args.work)), flush=True)
