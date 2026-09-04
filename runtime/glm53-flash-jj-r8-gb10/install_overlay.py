#!/usr/bin/env python3
"""Install and verify exact vLLM, B12X, and SparkCache source overlays."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, expected_commit: str | None = None) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if expected_commit is not None and document.get("commit") != expected_commit:
        raise RuntimeError(f"source manifest commit mismatch: {path}")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"source manifest is empty: {path}")
    return {str(name): str(digest) for name, digest in files.items()}


def verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"installed source is missing: {relative}")
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"installed source mismatch: {relative}: {actual} != {expected}"
            )


def verify_package_file_set(
    package_root: Path, files: dict[str, str], package: str
) -> None:
    expected: set[str] = set()
    for relative in files:
        path = Path(relative)
        if not path.parts or path.parts[0] != package:
            raise RuntimeError(f"source manifest escaped {package}: {relative}")
        expected.add(Path(*path.parts[1:]).as_posix())
    observed = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError(f"installed {package} package file set differs from manifest")


def native_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*.so"))
    }


def remove_parent_sources(site_root: Path, receipt: Path, prefix: str) -> None:
    for relative in load_manifest(receipt):
        path = Path(relative)
        if not path.parts or path.parts[0] != prefix:
            raise RuntimeError(f"parent manifest escaped {prefix}: {relative}")
        target = site_root / path
        if target.is_file() or target.is_symlink():
            target.unlink()


def copy_source(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise RuntimeError(f"source overlay contains a symlink: {relative}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def replace_python_package(source: Path, destination: Path) -> None:
    """Replace one inherited package tree so no stale module can survive."""
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
    copy_source(source, destination)


def install(root: Path, site_root: Path) -> None:
    pins = json.loads((root / "receipts/pins.json").read_text(encoding="utf-8"))
    vllm_manifest = load_manifest(
        root / "receipts/vllm-source-manifest.json", pins["vllm"]["commit"]
    )
    b12x_manifest = load_manifest(
        root / "receipts/b12x-source-manifest.json", pins["b12x"]["commit"]
    )
    sparkcache_manifest = load_manifest(
        root / "receipts/sparkcache-source-manifest.json",
        pins["sparkcache"]["commit"],
    )
    vllm_root = site_root / "vllm"
    before_native = native_manifest(vllm_root)
    if not before_native:
        raise RuntimeError("the parent has no retained vLLM native extensions")

    remove_parent_sources(
        site_root,
        Path(
            "/opt/sparkring/receipts/jj-r7-arm64/vllm-source-manifest.json"
        ),
        "vllm",
    )
    remove_parent_sources(
        site_root,
        Path(
            "/opt/sparkring/receipts/jj-r7-sparkcache-arm64/"
            "sparkcache-source-manifest.json"
        ),
        "sparkcache",
    )
    remove_parent_sources(
        Path("/opt/sparkcache-src"),
        Path(
            "/opt/sparkring/receipts/jj-r7-sparkcache-arm64/"
            "sparkcache-source-manifest.json"
        ),
        "sparkcache",
    )
    copy_source(root / "sources/vllm", vllm_root)
    replace_python_package(root / "sources/b12x", site_root / "b12x")
    copy_source(root / "sources/sparkcache", site_root / "sparkcache")
    copy_source(root / "sources/sparkcache", Path("/opt/sparkcache-src/sparkcache"))

    snapshot_source = root / "native/libspark_cache_snapshot.so"
    snapshot_destination = Path(
        "/opt/sparkcache-src/sparkcache/native/build-cuda/"
        "libspark_cache_snapshot.so"
    )
    expected_snapshot = pins["sparkcache"]["cuda_snapshot_sha256"]
    if file_sha256(snapshot_source) != expected_snapshot:
        raise RuntimeError("SparkCache CUDA snapshot library input differs")
    snapshot_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(snapshot_source, snapshot_destination)

    verify_files(site_root, vllm_manifest)
    verify_files(site_root, b12x_manifest)
    verify_package_file_set(site_root / "b12x", b12x_manifest, "b12x")
    verify_files(site_root, sparkcache_manifest)
    after_native = native_manifest(vllm_root)
    if after_native != before_native:
        raise RuntimeError("vLLM native extensions changed during Python overlay")

    placement = Path(
        "/opt/sparkcache-src/sparkcache/native/build-cuda/"
        "libspark_cache_placement.so"
    )
    if file_sha256(placement) != (
        "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c"
    ):
        raise RuntimeError("SparkCache CUDA placement library identity changed")
    if file_sha256(snapshot_destination) != expected_snapshot:
        raise RuntimeError("SparkCache CUDA snapshot library identity changed")
    nccl = Path("/opt/sparkring/nccl/libnccl.so.2.30.7")
    if file_sha256(nccl) != (
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3"
    ):
        raise RuntimeError("switchless NCCL library identity changed")
    if importlib.util.find_spec("fastsafetensors") is None:
        raise RuntimeError("fastsafetensors is absent from the inherited runtime")
    for candidate in site_root.glob("deep_ep*"):
        raise RuntimeError(f"unused DeepEP package remains installed: {candidate.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--site-root",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    install(args.root, args.site_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
