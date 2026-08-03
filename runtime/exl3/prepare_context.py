#!/usr/bin/env python3
"""Fetch, attest, and stage the public EXL3 ARM64 build context."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run_git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(repo), *args],
        capture_output=True,
        check=False,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {stderr.strip()}")
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {observed}")


def clone_if_missing(repo: Path, repository: str) -> bool:
    if repo.exists():
        if not (repo / ".git").is_dir():
            raise RuntimeError(f"source cache path is not a git checkout: {repo}")
        origin = str(run_git(repo, "remote", "get-url", "origin")).strip()
        if origin != repository:
            raise RuntimeError(f"source origin mismatch for {repo}: expected {repository}, got {origin}")
        return False
    repo.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--filter=blob:none", repository, str(repo)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"git clone failed for {repository}: {result.stderr.strip()}")
    return True


def materialize_source(source_root: Path, name: str, record: dict[str, str]) -> tuple[Path, str]:
    repo = source_root / f"{name}-{record['base_commit'][:12]}"
    fresh = clone_if_missing(repo, record["repository"])
    status = str(run_git(repo, "status", "--porcelain")).strip()
    if status and not fresh:
        current_head = str(run_git(repo, "rev-parse", "HEAD")).strip()
        current_tree = str(run_git(repo, "write-tree")).strip()
        unstaged = str(run_git(repo, "diff", "--name-only")).strip()
        untracked = str(run_git(repo, "ls-files", "--others", "--exclude-standard")).strip()
        if current_head == record["base_commit"] and current_tree == record["tree"] and not unstaged and not untracked:
            return repo, current_tree
        raise RuntimeError(
            f"source cache is dirty or has an unexpected staged tree: {repo}; "
            "remove only this cache directory and retry"
        )
    run_git(repo, "fetch", "--force", "--depth=1", "origin", record["base_commit"])
    run_git(repo, "checkout", "--detach", record["base_commit"])
    patch_path = HERE / record["patch"]
    verify_file(patch_path, record["patch_sha256"], f"{name} source patch")
    run_git(repo, "apply", "--index", "--whitespace=nowarn", str(patch_path))
    run_git(repo, "diff", "--cached", "--check")
    tree = str(run_git(repo, "write-tree")).strip()
    if tree != record["tree"]:
        raise RuntimeError(f"{name} patched tree mismatch: expected {record['tree']}, got {tree}")
    return repo, tree


def materialize_lmcache(source_root: Path, record: dict) -> tuple[Path, str]:
    """Compose the exact reviewed LMCache PR heads and local-server patch."""
    repo = source_root / f"lmcache-{record['composed_tree'][:12]}"
    fresh = clone_if_missing(repo, record["repository"])
    if not fresh:
        head_tree = str(run_git(repo, "rev-parse", "HEAD^{tree}")).strip()
        index_tree = str(run_git(repo, "write-tree")).strip()
        unstaged = str(run_git(repo, "diff", "--name-only")).strip()
        untracked = str(
            run_git(repo, "ls-files", "--others", "--exclude-standard")
        ).strip()
        if (
            head_tree == record["integration_tree"]
            and index_tree == record["composed_tree"]
            and not unstaged
            and not untracked
        ):
            return repo, index_tree
        raise RuntimeError(
            f"LMCache source cache has an unexpected tree or status: {repo}; "
            "remove only this cache directory and retry"
        )

    commits = [record["base_commit"], *record["integration_heads"]]
    for commit in commits:
        run_git(repo, "fetch", "--force", "origin", commit)
    run_git(repo, "checkout", "--detach", record["base_commit"])
    run_git(repo, "config", "user.email", "agent@sparkring.local")
    run_git(repo, "config", "user.name", "SparkRing Agent")
    for commit in record["integration_heads"]:
        run_git(repo, "merge", "--no-edit", "--no-ff", commit)
    integration_tree = str(run_git(repo, "rev-parse", "HEAD^{tree}")).strip()
    if integration_tree != record["integration_tree"]:
        raise RuntimeError(
            "LMCache integration tree mismatch: expected "
            f"{record['integration_tree']}, got {integration_tree}"
        )

    patch_path = HERE / record["topology_patch"]
    verify_file(
        patch_path,
        record["topology_patch_sha256"],
        "LMCache four-local-server topology patch",
    )
    run_git(repo, "apply", "--index", "--whitespace=nowarn", str(patch_path))
    run_git(repo, "diff", "--cached", "--check")
    composed_tree = str(run_git(repo, "write-tree")).strip()
    if composed_tree != record["composed_tree"]:
        raise RuntimeError(
            "LMCache composed tree mismatch: expected "
            f"{record['composed_tree']}, got {composed_tree}"
        )
    return repo, composed_tree


def archive_tree(repo: Path, tree: str, destination: Path) -> None:
    archive = run_git(repo, "archive", "--format=tar", tree, binary=True)
    assert isinstance(archive, bytes)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(destination, filter="data")


def verify_overlay(pins: dict) -> None:
    for relative, expected in pins["overlay_files"].items():
        verify_file(HERE / "overlay" / relative, expected, f"vLLM overlay {relative}")
    observed = {
        path.relative_to(HERE / "overlay").as_posix()
        for path in (HERE / "overlay").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    expected = set(pins["overlay_files"])
    if observed != expected:
        raise RuntimeError(f"unattested vLLM overlay files: {sorted(observed ^ expected)}")


def verify_runtime_overlay(pins: dict) -> None:
    for relative, record in pins["spark_runtime_overlay_files"].items():
        verify_file(HERE / "runtime-overlay" / relative, record["postimage"], f"runtime overlay {relative}")


def build_manifest(output: Path, pins: dict) -> dict:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "context-manifest.json":
            files[path.relative_to(output).as_posix()] = sha256(path)
    return {
        "schema": "sparkring-public-exl3-build-context/v1",
        "profile_id": pins["profile_id"],
        "sources": pins["sources"],
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    if pins.get("schema") != "sparkring-public-exl3-pins/v1":
        raise RuntimeError("wrong public EXL3 pins schema")
    verify_file(
        HERE / "cutlass-requirements.txt",
        pins["cutlass_python_lock"]["requirements_sha256"],
        "CUTLASS Python wheel lock",
    )
    verify_file(
        HERE / "lmcache-requirements.txt",
        pins["lmcache"]["requirements_sha256"],
        "LMCache runtime wheel lock",
    )
    verify_overlay(pins)
    verify_runtime_overlay(pins)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists; use a new disposable directory: {output}")

    materialized = {}
    for name in ("sparkinfer", "exllamav3"):
        repo, tree = materialize_source(args.source_root.resolve(), name, pins["sources"][name])
        materialized[name] = (repo, tree)
    materialized["lmcache"] = materialize_lmcache(
        args.source_root.resolve(), pins["lmcache"]
    )

    output.mkdir(parents=True)
    (output / "sources").mkdir()
    for name, (repo, tree) in materialized.items():
        archive_tree(repo, tree, output / "sources" / name)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(HERE / "overlay", output / "overlay", ignore=ignored)
    shutil.copytree(HERE / "runtime-overlay", output / "runtime-overlay", ignore=ignored)
    for name in (
        "Containerfile",
        "cutlass-requirements.txt",
        "lmcache-requirements.txt",
        "pins.json",
        "model_manifest.py",
        "verify_exl3_model.py",
        "verify_exl3_runtime.py",
        "compose_runtime_manifest.py",
        "patch_sparse_profile_capacity.py",
        "entrypoint.sh",
        "build-image.sh",
        "verify_build_context.py",
    ):
        shutil.copy2(HERE / name, output / name)
    manifest = build_manifest(output, pins)
    (output / "context-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
