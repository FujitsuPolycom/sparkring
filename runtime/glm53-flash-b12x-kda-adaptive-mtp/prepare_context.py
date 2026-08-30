#!/usr/bin/env python3
"""Prepare and verify public source inputs for the GLM-5.3 ARM64 image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_PINS = HERE / "pins.json"
PINS_SCHEMA = "sparkring-glm53-flash-runtime-lock/v1"
RECEIPT_SCHEMA = "sparkring-glm53-public-build-context/v1"


class PrepareError(RuntimeError):
    """Raised when a public build input differs from its immutable pin."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pins(path: Path = DEFAULT_PINS) -> dict[str, Any]:
    pins = json.loads(path.read_text(encoding="utf-8"))
    if pins.get("schema") != PINS_SCHEMA:
        raise PrepareError(f"unsupported pins schema: {pins.get('schema')!r}")
    return pins


def run(argv: Iterable[str], *, cwd: Path | None = None) -> str:
    arguments = list(argv)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PrepareError(f"command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout.strip()


def git_value(repository: Path, revision: str) -> str:
    return run(("git", "-C", str(repository), "rev-parse", revision))


def clone_detached(
    destination: Path,
    *,
    repository: str,
    commit: str,
    tree: str,
    fetch_depth: int = 1,
) -> None:
    destination.mkdir(parents=True)
    run(("git", "init", "--quiet", str(destination)))
    run(("git", "-C", str(destination), "config", "core.autocrlf", "false"))
    run(("git", "-C", str(destination), "remote", "add", "origin", repository))
    run(
        (
            "git",
            "-C",
            str(destination),
            "fetch",
            "--quiet",
            "--depth",
            str(fetch_depth),
            "origin",
            commit,
        )
    )
    run(("git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"))
    verify_git_tree(destination, expected_commit=commit, expected_tree=tree)


def verify_source_lineage(repository: Path, lineage: dict[str, Any]) -> None:
    base = lineage["base_commit"]
    expected = lineage["included_commits"]
    observed = run(
        (
            "git",
            "-C",
            str(repository),
            "rev-list",
            "--first-parent",
            "--reverse",
            f"{base}..HEAD",
        )
    ).splitlines()
    if observed != expected:
        raise PrepareError(
            "vLLM first-parent lineage differs from the pinned commit sequence"
        )
    adaptive_commit = lineage["adaptive_mtp_commit"]
    observed_tree = git_value(repository, f"{adaptive_commit}^{{tree}}")
    if observed_tree != lineage["adaptive_mtp_tree"]:
        raise PrepareError(
            "vLLM adaptive-MTP source boundary differs from its pinned tree"
        )
    if expected[-3:] != lineage["kda_live_tensor_commits"]:
        raise PrepareError(
            "vLLM live-tensor KDA commits are not the final source-lineage entries"
        )


def verify_git_tree(
    repository: Path,
    *,
    expected_commit: str,
    expected_tree: str,
    indexed: bool = False,
) -> None:
    observed_commit = git_value(repository, "HEAD")
    if observed_commit != expected_commit:
        raise PrepareError(
            f"{repository.name} commit drift: expected {expected_commit}, "
            f"got {observed_commit}"
        )
    tree_revision = None if indexed else "HEAD^{tree}"
    observed_tree = (
        run(("git", "-C", str(repository), "write-tree"))
        if indexed
        else git_value(repository, tree_revision)
    )
    if observed_tree != expected_tree:
        raise PrepareError(
            f"{repository.name} tree drift: expected {expected_tree}, "
            f"got {observed_tree}"
        )
    if run(("git", "-C", str(repository), "diff", "--name-only")):
        raise PrepareError(f"{repository.name} has unstaged changes")


def require_hash(path: Path, expected: str, description: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise PrepareError(
            f"{description} SHA-256 drift: expected {expected}, got {observed}"
        )


def download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "SparkRing-builder/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    require_hash(destination, expected_sha256, destination.name)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(output: Path, *, repository_root: Path, pins_path: Path) -> dict[str, Any]:
    if output.exists():
        raise PrepareError(f"output already exists: {output}")
    output.mkdir(parents=True)
    sources = output / "bundle" / "sources"
    runtime = output / "bundle" / "runtime"
    sources.mkdir(parents=True)
    runtime.mkdir(parents=True)

    pins = load_pins(pins_path)
    build = pins["public_image_build"]
    source_pins = build["sources"]

    vllm_pin = source_pins["vllm"]
    vllm_lineage = vllm_pin["source_lineage"]
    clone_detached(
        sources / "vllm",
        repository=vllm_pin["repository"],
        commit=vllm_pin["commit"],
        tree=vllm_pin["tree"],
        fetch_depth=len(vllm_lineage["included_commits"]) + 1,
    )
    verify_source_lineage(sources / "vllm", vllm_lineage)
    b12x_pin = source_pins["b12x"]
    clone_detached(
        sources / "b12x",
        repository=b12x_pin["repository"],
        commit=b12x_pin["commit"],
        tree=b12x_pin["tree"],
    )

    nccl_pin = source_pins["nccl"]
    nccl = sources / "nccl"
    clone_detached(
        nccl,
        repository=nccl_pin["repository"],
        commit=nccl_pin["commit"],
        tree=nccl_pin["base_tree"],
    )
    for patch_pin in nccl_pin["patches"]:
        patch_path = repository_root / patch_pin["path"]
        require_hash(patch_path, patch_pin["sha256"], patch_pin["path"])
        run(("git", "-C", str(nccl), "apply", "--check", "--binary", str(patch_path)))
        run(("git", "-C", str(nccl), "apply", "--binary", "--index", str(patch_path)))
    verify_git_tree(
        nccl,
        expected_commit=nccl_pin["commit"],
        expected_tree=nccl_pin["patched_tree"],
        indexed=True,
    )

    instanttensor = build["instanttensor"]
    instanttensor_sdist = sources / f"instanttensor-{instanttensor['version']}.tar.gz"
    download(
        instanttensor["sdist_url"],
        instanttensor_sdist,
        instanttensor["sdist_sha256"],
    )

    copy_file(pins_path, runtime / "pins.json")
    copy_file(HERE / "verify_image.py", runtime / "verify_image.py")
    copy_file(HERE / "Containerfile", runtime / "Containerfile")
    copy_file(HERE / "Containerfile.seed", runtime / "Containerfile.seed")
    copy_file(HERE / "build-image.sh", runtime / "build-image.sh")
    copy_file(HERE / "LICENSES" / "README.md", runtime / "LICENSES.md")
    copy_file(repository_root / "LICENSE", runtime / "SparkRing-LICENSE")

    receipt_files = (
        "bundle/runtime/pins.json",
        "bundle/runtime/verify_image.py",
        "bundle/runtime/Containerfile",
        "bundle/runtime/Containerfile.seed",
        "bundle/runtime/build-image.sh",
        "bundle/runtime/LICENSES.md",
        "bundle/runtime/SparkRing-LICENSE",
        f"bundle/sources/instanttensor-{instanttensor['version']}.tar.gz",
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "pins_sha256": sha256_file(pins_path),
        "sources": {
            "vllm": {
                "commit": vllm_pin["commit"],
                "tree": vllm_pin["tree"],
                "source_lineage": vllm_lineage,
            },
            "b12x": {"commit": b12x_pin["commit"], "tree": b12x_pin["tree"]},
            "nccl": {
                "commit": nccl_pin["commit"],
                "tree": nccl_pin["patched_tree"],
            },
        },
        "files": {
            relative: sha256_file(output / relative) for relative in receipt_files
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_context(output, pins_path=pins_path)
    return receipt


def verify_context(context: Path, *, pins_path: Path) -> dict[str, Any]:
    pins = load_pins(pins_path)
    receipt_path = context / "receipt.json"
    if not receipt_path.is_file():
        raise PrepareError(f"missing prepared-context receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise PrepareError(f"unsupported receipt schema: {receipt.get('schema')!r}")
    if receipt.get("pins_sha256") != sha256_file(pins_path):
        raise PrepareError("prepared context was produced from different pins")
    for relative, expected in receipt["files"].items():
        require_hash(context / relative, expected, relative)

    build = pins["public_image_build"]
    source_pins = build["sources"]
    sources = context / "bundle" / "sources"
    source_contracts = {
        "vllm": (
            source_pins["vllm"]["commit"],
            source_pins["vllm"]["tree"],
            False,
        ),
        "b12x": (
            source_pins["b12x"]["commit"],
            source_pins["b12x"]["tree"],
            False,
        ),
        "nccl": (
            source_pins["nccl"]["commit"],
            source_pins["nccl"]["patched_tree"],
            True,
        ),
    }
    for name, (commit, tree, indexed) in source_contracts.items():
        verify_git_tree(
            sources / name,
            expected_commit=commit,
            expected_tree=tree,
            indexed=indexed,
        )
    verify_source_lineage(sources / "vllm", source_pins["vllm"]["source_lineage"])
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify:
            receipt = verify_context(args.output.resolve(), pins_path=args.pins.resolve())
        else:
            receipt = prepare(
                args.output.resolve(),
                repository_root=args.repo_root.resolve(),
                pins_path=args.pins.resolve(),
            )
    except (OSError, KeyError, json.JSONDecodeError, PrepareError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
