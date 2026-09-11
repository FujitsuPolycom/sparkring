"""Prepare the network-fetched compute payload for an offline image build."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "source-lock.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map_sha256(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _download(url: str, expected: str, destination: Path) -> None:
    if destination.is_file() and _sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        urllib.request.urlretrieve(url, temporary)
        actual = _sha256(temporary)
        if actual != expected:
            raise ValueError(f"download hash mismatch for {url}: {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_single_root(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    entries = list(destination.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        raise ValueError(f"expected one source root in {archive.name}")
    return entries[0]


def _package_map(root: Path, package: str) -> dict[str, str]:
    package_root = root / package
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"no package files found under {package_root}")
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in files
    }


def _normalize_b12x_bytes(root: Path) -> None:
    # The locked installed package hashes use CRLF for these text suffixes.
    # Normalize LF and CRLF inputs alike; changing this requires a new byte lock.
    suffixes = set(LOCK["b12x"]["normalized_text_suffixes"])
    for path in sorted((root / "b12x").rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            data = path.read_bytes().replace(b"\r\n", b"\n")
            path.write_bytes(data.replace(b"\n", b"\r\n"))


def _apply_b12x_overrides(root: Path, archive: Path, contract: dict) -> None:
    """Apply checksum-bound selector files only over their expected source bytes."""
    if _sha256(archive) != contract["archive_sha256"]:
        raise ValueError("B12X override archive hash mismatch")
    entries = contract["files"]
    expected = {name for name, _, _ in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate B12X override path")
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if (len(members) != len(expected) or {item.name for item in members} != expected
                or any(not item.isfile() for item in members)):
            raise ValueError("B12X override archive has an unexpected file set")
        replacements = {}
        for name, base_hash, result_hash in entries:
            path = root / name
            if not name.startswith("b12x/") or not path.resolve().is_relative_to((root / "b12x").resolve()):
                raise ValueError("B12X override escapes package directory")
            if _sha256(path) != base_hash:
                raise ValueError(f"B12X override base hash mismatch: {name}")
            data = source.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != result_hash:
                raise ValueError(f"B12X override result hash mismatch: {name}")
            replacements[path] = data
        for path, data in replacements.items():
            path.write_bytes(data)


def prepare(destination: Path, cache: Path | None = None) -> Path:
    """Create a complete, checksum-bound context for a network-disabled build."""
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"compute destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    cache = (cache or destination.parent / ".compute-downloads").resolve()
    cache.mkdir(parents=True, exist_ok=True)

    for name in (
        "source-lock.json",
        "vllm-e02-to-compute.patch",
        "vllm-compute-files.tar.gz",
        "apply_compute.py",
        "verify_compute.py",
    ):
        shutil.copy2(HERE / name, destination / name)

    lock_hash = _sha256(destination / "source-lock.json")
    patch = destination / LOCK["vllm"]["patch"]
    if _sha256(patch) != LOCK["vllm"]["patch_sha256"]:
        raise ValueError("vLLM patch does not match source-lock.json")
    replacement = destination / LOCK["vllm"]["replacement_archive"]
    if _sha256(replacement) != LOCK["vllm"]["replacement_archive_sha256"]:
        raise ValueError("vLLM replacement archive does not match source-lock.json")

    b12x = LOCK["b12x"]
    b12x_archive = cache / "b12x.tar.gz"
    _download(b12x["archive_url"], b12x["archive_sha256"], b12x_archive)
    unpack = destination / ".b12x-unpack"
    source_root = _extract_single_root(b12x_archive, unpack)
    b12x_destination = destination / "b12x-source"
    shutil.move(str(source_root), b12x_destination)
    shutil.rmtree(unpack)
    _normalize_b12x_bytes(b12x_destination)
    if b12x.get("overrides"):
        overrides = b12x["overrides"]
        archive = destination / overrides["archive"]
        shutil.copy2(HERE / overrides["archive"], archive)
        _apply_b12x_overrides(b12x_destination, archive, overrides)
    b12x_files = _package_map(b12x_destination, "b12x")
    if _map_sha256(b12x_files) != b12x["package_files_sha256"]:
        raise ValueError("normalized B12X package does not match source-lock.json")

    cuda_archives: dict[str, str] = {}
    cuda_dir = destination / "cuda-archives"
    cuda_dir.mkdir()
    for relative, expected in LOCK["cuda"]["components"].items():
        archive = cache / Path(relative).name
        _download(LOCK["cuda"]["base_url"] + relative, expected, archive)
        target = cuda_dir / archive.name
        shutil.copy2(archive, target)
        cuda_archives[f"cuda-archives/{target.name}"] = expected

    prepared = {
        "schema": "sparkring-glm53-compute-prepared/v1",
        "source_lock_sha256": lock_hash,
        "vllm_patch_sha256": LOCK["vllm"]["patch_sha256"],
        "vllm_replacement_archive_sha256": LOCK["vllm"][
            "replacement_archive_sha256"
        ],
        "b12x_revision": b12x["revision"],
        "b12x_tree": b12x["tree"],
        "b12x_archive_sha256": b12x["archive_sha256"],
        "b12x_package_files_sha256": b12x["package_files_sha256"],
        "b12x_files": b12x_files,
        "cuda_archives": cuda_archives,
    }
    manifest = destination / "prepared-manifest.json"
    manifest.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    print(prepare(args.output, args.cache))


if __name__ == "__main__":
    main()
