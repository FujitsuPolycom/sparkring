#!/usr/bin/env python3
"""Verify installed Qwen3.8 runtime identities without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PINS = Path("/ws/runtime/pins.json")


class VerificationError(RuntimeError):
    """Raised when installed runtime bytes differ from the public pins."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: Iterable[str]) -> str:
    completed = subprocess.run(
        list(argv),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise VerificationError(f"command failed ({' '.join(argv)}): {detail}")
    return completed.stdout.strip()


def require_equal(description: str, observed: str, expected: str) -> None:
    if observed != expected:
        raise VerificationError(
            f"{description} drift: expected {expected}, got {observed}"
        )


def verify_git_source(
    path: Path,
    *,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, str]:
    commit = run(("git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"))
    tree = run(("git", "-c", f"safe.directory={path}", "-C", str(path), "write-tree"))
    require_equal(f"{path.name} commit", commit, expected_commit)
    require_equal(f"{path.name} tree", tree, expected_tree)
    if run(("git", "-c", f"safe.directory={path}", "-C", str(path), "diff", "--name-only")):
        raise VerificationError(f"{path.name} has unstaged source changes")
    return {"commit": commit, "tree": tree}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise VerificationError(f"required package is not installed: {name}") from exc


def verify(pins_path: Path, *, imports: bool = False) -> dict[str, Any]:
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    if pins.get("schema") != "sparkring-qwen38-runtime-pins/v1":
        raise VerificationError(f"unsupported pins schema: {pins.get('schema')!r}")

    layout = pins["layout"]
    vllm_pin = pins["sources"]["vllm"]
    exllamav3_pin = pins["sources"]["exllamav3"]
    nccl_pin = pins["sources"]["nccl"]
    sources = {
        "vllm": verify_git_source(
            Path(layout["vllm_source"]),
            expected_commit=vllm_pin["commit"],
            expected_tree=vllm_pin["tree"],
        ),
        "exllamav3": verify_git_source(
            Path(layout["exllamav3_source"]),
            expected_commit=exllamav3_pin["commit"],
            expected_tree=exllamav3_pin["patched_tree"],
        ),
        "nccl": verify_git_source(
            Path(layout["nccl_source"]),
            expected_commit=nccl_pin["commit"],
            expected_tree=nccl_pin["patched_tree"],
        ),
    }

    nccl_path = Path(layout["patched_nccl"])
    nccl_sha = sha256_file(nccl_path)
    require_equal("patched NCCL library SHA-256", nccl_sha, pins["nccl"]["library_sha256"])
    chat_path = Path(layout["chat_template"])
    chat_sha = sha256_file(chat_path)
    require_equal(
        "chat template SHA-256",
        chat_sha,
        pins["companion"]["chat_template_sha256"],
    )

    versions = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "b12x": package_version("b12x"),
        "ray": package_version("ray"),
    }
    require_equal("torch version", versions["torch"], pins["toolchain"]["torch"])
    require_equal(
        "torchvision version",
        versions["torchvision"],
        pins["toolchain"]["torchvision"],
    )
    require_equal("B12X version", versions["b12x"], pins["python_packages"]["b12x"])
    require_equal("Ray version", versions["ray"], pins["python_packages"]["ray"])

    nvcc = run(("nvcc", "--version"))
    if f"V{pins['toolchain']['nvcc_version']}" not in nvcc:
        raise VerificationError(
            "CUDA compiler drift: expected nvcc "
            f"{pins['toolchain']['nvcc_version']}"
        )

    imported: list[str] = []
    if imports:
        for module_name in ("torch", "vllm", "exllamav3_ext"):
            module = importlib.import_module(module_name)
            imported.append(module_name)
            if module_name == "exllamav3_ext" and not hasattr(module, "exl3_gemm"):
                raise VerificationError("exllamav3_ext does not export exl3_gemm")

    return {
        "schema": "sparkring-qwen38-runtime-verification/v1",
        "passed": True,
        "sources": sources,
        "patched_nccl_sha256": nccl_sha,
        "chat_template_sha256": chat_sha,
        "packages": versions,
        "imports": imported,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument(
        "--imports",
        action="store_true",
        help="also import torch, vLLM, and the ExLlamaV3 extension",
    )
    args = parser.parse_args()
    try:
        result = verify(args.pins, imports=args.imports)
    except (OSError, KeyError, json.JSONDecodeError, VerificationError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
