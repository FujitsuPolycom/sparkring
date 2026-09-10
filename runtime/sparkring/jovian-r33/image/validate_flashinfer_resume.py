#!/usr/bin/env python3
"""Prove the completed FlashInfer resume used pristine tracked/submodule inputs and generated-only additions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


EXPECTED_HEAD = "803c4664f4771ddc418f20a57f752469a237a825"
EXPECTED_TREE = "8728cfffc16c122e3536b63dc3456fdcdb9db272"
ALLOWED_UNTRACKED = {
    "LICENSE.cutlass.txt", "LICENSE.flashattention3.txt", "LICENSE.fmt.txt", "LICENSE.spdlog.txt"
}
ALLOWED_GENERATED_ROOTS = {"build", "flashinfer", "flashinfer-jit-cache", "flashinfer_python.egg-info", "__pycache__"}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def file_map_digest(root: Path, names: list[str]) -> str:
    value = hashlib.sha256(b"sparkring-flashinfer-generated-files/v1\0")
    for name in sorted(names):
        path = root.joinpath(*PurePosixPath(name).parts)
        if path.is_symlink():
            content_hash = hashlib.sha256(path.readlink().as_posix().encode()).hexdigest()
        elif path.is_file():
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            continue
        value.update(name.encode() + b"\0" + content_hash.encode() + b"\n")
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    head = git(source, "rev-parse", "HEAD").decode().strip()
    tree = git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    if head != EXPECTED_HEAD or tree != EXPECTED_TREE:
        raise RuntimeError(f"FlashInfer source identity differs: {head} {tree}")
    if git(source, "diff", "--binary") or git(source, "diff", "--cached", "--binary"):
        raise RuntimeError("FlashInfer tracked files changed during the resumed build")
    submodules = git(source, "submodule", "status", "--recursive").decode().splitlines()
    if any(not line.startswith(" ") for line in submodules):
        raise RuntimeError("FlashInfer recursive submodule differs or is dirty")
    untracked = git(source, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    if set(untracked) != ALLOWED_UNTRACKED:
        raise RuntimeError(f"unexpected untracked FlashInfer inputs: {untracked}")
    generated = git(source, "ls-files", "--others", "--ignored", "--exclude-standard").decode().splitlines()
    roots = {PurePosixPath(name).parts[0] for name in generated}
    if not generated or not roots <= ALLOWED_GENERATED_ROOTS:
        raise RuntimeError(f"unexpected resumed-build output roots: {sorted(roots)}")
    sums = {}
    for line in (args.artifacts / "SHA256SUMS").read_text().splitlines():
        expected, raw_path = line.split(maxsplit=1)
        path = args.artifacts / Path(raw_path).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"FlashInfer artifact differs from terminal receipt: {path.name}")
        sums[path.name] = expected
    if set(sums) != {
        "flashinfer_python-0.6.18+cu133-py3-none-any.whl",
        "flashinfer_jit_cache-0.6.18+cu133-cp39-abi3-manylinux_2_28_aarch64.whl",
    }:
        raise RuntimeError("FlashInfer terminal wheel set differs")
    result = {
        "schema": "sparkring-r33-flashinfer-resume-receipt/v1",
        "status": "tracked-and-recursive-submodule-inputs-pristine-generated-only-output-validated",
        "source_head": head,
        "source_tree": tree,
        "recursive_submodules": submodules,
        "untracked_source_files": sorted(untracked),
        "generated_roots": sorted(roots),
        "generated_file_count": len(generated),
        "generated_file_map_sha256": file_map_digest(source, generated),
        "wheel_sha256": sums,
        "tracked_input_pristine": True,
        "generated_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
