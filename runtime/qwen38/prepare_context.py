#!/usr/bin/env python3
"""Prepare and verify the public source context for the Qwen3.8 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_PINS = HERE / "pins.json"
SCHEMA = "sparkring-qwen38-runtime-pins/v1"
RECEIPT_SCHEMA = "sparkring-qwen38-prepared-context/v1"


class PrepareError(RuntimeError):
    """Raised when a public build input differs from its pin."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pins(path: Path = DEFAULT_PINS) -> dict[str, Any]:
    pins = json.loads(path.read_text(encoding="utf-8"))
    if pins.get("schema") != SCHEMA:
        raise PrepareError(f"unsupported pins schema: {pins.get('schema')!r}")
    return pins


def run(argv: Iterable[str], *, cwd: Path | None = None) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PrepareError(f"command failed ({' '.join(argv)}): {detail}")
    return completed.stdout.strip()


def git_value(repo: Path, revision: str) -> str:
    return run(("git", "-C", str(repo), "rev-parse", revision))


def verify_git_tree(
    repo: Path,
    *,
    expected_commit: str,
    expected_tree: str,
) -> None:
    observed_commit = git_value(repo, "HEAD")
    if observed_commit != expected_commit:
        raise PrepareError(
            f"{repo.name} commit drift: expected {expected_commit}, "
            f"got {observed_commit}"
        )
    observed_tree = git_value(repo, "HEAD^{tree}")
    if observed_tree != expected_tree:
        raise PrepareError(
            f"{repo.name} base tree drift: expected {expected_tree}, "
            f"got {observed_tree}"
        )
    if run(("git", "-C", str(repo), "status", "--porcelain")):
        raise PrepareError(f"{repo.name} checkout is not clean")


def clone_detached(
    destination: Path,
    *,
    repository: str,
    commit: str,
    tree: str,
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
            "1",
            "origin",
            commit,
        )
    )
    run(("git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"))
    verify_git_tree(
        destination,
        expected_commit=commit,
        expected_tree=tree,
    )


def require_hash(path: Path, expected: str, description: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise PrepareError(
            f"{description} SHA-256 drift: expected {expected}, got {observed}"
        )


def apply_patch(repo: Path, patch: Path, *, expected_tree: str) -> None:
    run(("git", "-C", str(repo), "apply", "--check", "--binary", str(patch)))
    run(("git", "-C", str(repo), "apply", "--binary", "--index", str(patch)))
    observed = git_value(repo, "HEAD^{tree}")
    if observed == expected_tree:
        raise PrepareError(f"{repo.name} patch did not change the indexed tree")
    observed = run(("git", "-C", str(repo), "write-tree"))
    if observed != expected_tree:
        raise PrepareError(
            f"{repo.name} patched tree drift: expected {expected_tree}, got {observed}"
        )
    if run(("git", "-C", str(repo), "diff", "--name-only")):
        raise PrepareError(f"{repo.name} patch left unstaged working-tree changes")


def sanitize_requirements(text: str, excluded: list[str]) -> str:
    excluded_exact = set(excluded)
    output: list[str] = []
    removed: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line in excluded_exact:
            removed.add(line)
            continue
        if line.startswith("-e ") or " @ file:" in line or "git+" in line:
            raise PrepareError(f"unapproved non-public requirement: {line}")
        output.append(raw_line)
    if removed != excluded_exact:
        missing = sorted(excluded_exact - removed)
        raise PrepareError(f"expected local requirements were not found: {missing}")
    return "\n".join(output) + "\n"


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(output: Path, *, repo_root: Path, pins_path: Path) -> dict[str, Any]:
    if output.exists():
        raise PrepareError(f"output already exists: {output}")
    output.mkdir(parents=True)
    sources = output / "bundle" / "sources"
    runtime = output / "bundle" / "runtime"
    sources.mkdir(parents=True)
    runtime.mkdir(parents=True)

    pins = load_pins(pins_path)
    companion_pin = pins["companion"]
    companion = sources / "qwen38-spark-pair"
    clone_detached(
        companion,
        repository=companion_pin["repository"],
        commit=companion_pin["commit"],
        tree=companion_pin["tree"],
    )

    requirements = companion / companion_pin["requirements_freeze_path"]
    require_hash(
        requirements,
        companion_pin["requirements_freeze_sha256"],
        "companion requirements freeze",
    )
    public_requirements = sanitize_requirements(
        requirements.read_text(encoding="utf-8"),
        pins["python_packages"]["excluded_from_public_freeze"],
    )
    (runtime / "requirements-public.txt").write_text(
        public_requirements,
        encoding="utf-8",
        newline="\n",
    )

    chat_template = companion / companion_pin["chat_template_path"]
    require_hash(
        chat_template,
        companion_pin["chat_template_sha256"],
        "Qwen chat template",
    )
    copy_file(chat_template, runtime / "chat_template_agentic.jinja")

    vllm_pin = pins["sources"]["vllm"]
    clone_detached(
        sources / "vllm",
        repository=vllm_pin["repository"],
        commit=vllm_pin["commit"],
        tree=vllm_pin["tree"],
    )

    exllamav3_pin = pins["sources"]["exllamav3"]
    exllamav3 = sources / "exllamav3"
    clone_detached(
        exllamav3,
        repository=exllamav3_pin["repository"],
        commit=exllamav3_pin["commit"],
        tree=exllamav3_pin["base_tree"],
    )
    arm_patch = companion / companion_pin["exllamav3_arm_patch_path"]
    require_hash(
        arm_patch,
        companion_pin["exllamav3_arm_patch_sha256"],
        "ExLlamaV3 ARM64 patch",
    )
    apply_patch(
        exllamav3,
        arm_patch,
        expected_tree=exllamav3_pin["patched_tree"],
    )

    nccl_pin = pins["sources"]["nccl"]
    nccl = sources / "nccl"
    clone_detached(
        nccl,
        repository=nccl_pin["repository"],
        commit=nccl_pin["commit"],
        tree=nccl_pin["base_tree"],
    )
    for patch_pin in pins["nccl"]["patches"]:
        patch = repo_root / patch_pin["path"]
        require_hash(patch, patch_pin["sha256"], patch_pin["path"])
        run(("git", "-C", str(nccl), "apply", "--check", "--binary", str(patch)))
        run(("git", "-C", str(nccl), "apply", "--binary", "--index", str(patch)))
    observed_nccl_tree = run(("git", "-C", str(nccl), "write-tree"))
    if observed_nccl_tree != nccl_pin["patched_tree"]:
        raise PrepareError(
            "NCCL patched tree drift: expected "
            f"{nccl_pin['patched_tree']}, got {observed_nccl_tree}"
        )
    if run(("git", "-C", str(nccl), "diff", "--name-only")):
        raise PrepareError("NCCL patches left unstaged working-tree changes")

    copy_file(pins_path, runtime / "pins.json")
    copy_file(HERE / "verify_runtime.py", runtime / "verify_runtime.py")
    copy_file(repo_root / "scripts" / "qwen38_dgx2_serve.sh", runtime / "qwen38_dgx2_serve.sh")
    copy_file(repo_root / "scripts" / "qwen38_dgx4_serve.sh", runtime / "qwen38_dgx4_serve.sh")

    receipt_files = (
        "bundle/runtime/pins.json",
        "bundle/runtime/verify_runtime.py",
        "bundle/runtime/qwen38_dgx2_serve.sh",
        "bundle/runtime/qwen38_dgx4_serve.sh",
        "bundle/runtime/chat_template_agentic.jinja",
        "bundle/runtime/requirements-public.txt",
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "pins_sha256": sha256_file(pins_path),
        "sources": {
            "companion": {
                "commit": companion_pin["commit"],
                "tree": companion_pin["tree"],
            },
            "vllm": {"commit": vllm_pin["commit"], "tree": vllm_pin["tree"]},
            "exllamav3": {
                "commit": exllamav3_pin["commit"],
                "tree": exllamav3_pin["patched_tree"],
            },
            "nccl": {"commit": nccl_pin["commit"], "tree": nccl_pin["patched_tree"]},
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

    sources = context / "bundle" / "sources"
    source_pins = {
        "qwen38-spark-pair": (
            pins["companion"]["commit"],
            pins["companion"]["tree"],
        ),
        "vllm": (
            pins["sources"]["vllm"]["commit"],
            pins["sources"]["vllm"]["tree"],
        ),
        "exllamav3": (
            pins["sources"]["exllamav3"]["commit"],
            pins["sources"]["exllamav3"]["patched_tree"],
        ),
        "nccl": (
            pins["sources"]["nccl"]["commit"],
            pins["sources"]["nccl"]["patched_tree"],
        ),
    }
    for name, (commit, tree) in source_pins.items():
        repo = sources / name
        if git_value(repo, "HEAD") != commit:
            raise PrepareError(f"prepared {name} commit drift")
        if run(("git", "-C", str(repo), "write-tree")) != tree:
            raise PrepareError(f"prepared {name} tree drift")
        if run(("git", "-C", str(repo), "diff", "--name-only")):
            raise PrepareError(f"prepared {name} has unstaged changes")
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
                repo_root=args.repo_root.resolve(),
                pins_path=args.pins.resolve(),
            )
    except (OSError, KeyError, json.JSONDecodeError, PrepareError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
