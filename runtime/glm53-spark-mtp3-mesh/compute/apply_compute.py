"""Install the prepared CUDA, B12X, and vLLM compute composition offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map_sha256(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load(prepared: Path) -> tuple[dict, dict]:
    lock_path = prepared / "source-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads((prepared / "prepared-manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != manifest[
        "source_lock_sha256"
    ]:
        raise ValueError("prepared source lock hash mismatch")
    return lock, manifest


def _install_cuda(prepared: Path, lock: dict, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"CUDA destination already exists: {destination}")
    destination.mkdir(parents=True)
    for relative, expected in lock["cuda"]["components"].items():
        archive = prepared / "cuda-archives" / Path(relative).name
        if _sha256(archive) != expected:
            raise ValueError(f"CUDA archive hash mismatch: {archive.name}")
        with tempfile.TemporaryDirectory(prefix="sparkring-cuda-") as temporary:
            unpack = Path(temporary)
            with tarfile.open(archive) as source:
                source.extractall(unpack, filter="data")
            entries = list(unpack.iterdir())
            if len(entries) != 1 or not entries[0].is_dir():
                raise ValueError(f"invalid CUDA archive root: {archive.name}")
            shutil.copytree(entries[0], destination, dirs_exist_ok=True, symlinks=True)
    (destination / "sparkring-component-manifest.json").write_text(
        json.dumps(lock["cuda"]["components"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def _install_vllm(prepared: Path, site: Path, lock: dict) -> dict[str, str]:
    entries = lock["vllm"]["files"]
    patch = prepared / lock["vllm"]["patch"]
    if _sha256(patch) != lock["vllm"]["patch_sha256"]:
        raise ValueError("vLLM patch hash mismatch")
    archive = prepared / lock["vllm"]["replacement_archive"]
    if _sha256(archive) != lock["vllm"]["replacement_archive_sha256"]:
        raise ValueError("vLLM replacement archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="sparkring-vllm-") as temporary:
        work = Path(temporary)
        for relative, base_hash, _ in entries:
            installed = site / relative
            actual = _sha256(installed)
            if actual != base_hash:
                raise ValueError(f"vLLM base hash mismatch for {relative}: {actual}")
        with tarfile.open(archive) as source:
            names = set(source.getnames())
            expected_names = {entry[0] for entry in entries}
            if names != expected_names:
                raise ValueError("vLLM replacement archive has an unexpected file set")
            source.extractall(work, filter="data")
        result: dict[str, str] = {}
        for relative, _, expected in entries:
            staged = work / relative
            actual = _sha256(staged)
            if actual != expected:
                raise ValueError(f"vLLM result hash mismatch for {relative}: {actual}")
            result[relative] = actual
        # Reject any malformed replacement before modifying installed sources.
        for relative in result:
            shutil.copyfile(work / relative, site / relative)
    return result


def _package_map(root: Path) -> dict[str, str]:
    package = root / "b12x"
    files = sorted(
        path
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"no installed B12X package files under {package}")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in files}


def apply(
    prepared: Path,
    site_packages: Path,
    receipt: Path,
    cuda_destination: Path,
) -> Path:
    prepared = prepared.resolve()
    site_packages = site_packages.resolve()
    lock, manifest = _load(prepared)
    _install_cuda(prepared, lock, cuda_destination)
    vllm_files = _install_vllm(prepared, site_packages, lock)
    subprocess.run(
        [
            "python3",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            str(prepared / "b12x-source"),
        ],
        check=True,
    )
    b12x_files = _package_map(site_packages)
    if b12x_files != manifest["b12x_files"]:
        raise ValueError("installed B12X package differs from prepared source")
    if _map_sha256(b12x_files) != lock["b12x"]["package_files_sha256"]:
        raise ValueError("installed B12X package differs from source-lock.json")
    output = {
        "schema": "sparkring-glm53-compute-installed/v1",
        "source_lock_sha256": manifest["source_lock_sha256"],
        "vllm_revision": lock["vllm"]["base_revision"],
        "vllm_overrides": vllm_files,
        "b12x_revision": lock["b12x"]["revision"],
        "b12x_tree": lock["b12x"]["tree"],
        "b12x_files": b12x_files,
        "b12x_package_files_sha256": lock["b12x"]["package_files_sha256"],
        "cuda_components": lock["cuda"]["components"],
        "environment": lock["environment"],
        "target_head_quantization": False,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/opt/sparkring/receipts/glm53-compute-installed.json"),
    )
    parser.add_argument("--cuda-destination", type=Path, default=Path("/opt/cuda-13.3"))
    args = parser.parse_args()
    print(apply(args.prepared, args.site_packages, args.receipt, args.cuda_destination))


if __name__ == "__main__":
    main()
