"""Verify the installed compute source against its build receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map_sha256(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify(site_packages: Path, receipt: Path, source_lock: Path) -> dict:
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    installed = json.loads(receipt.read_text(encoding="utf-8"))
    lock_hash = _sha256(source_lock)
    if installed["source_lock_sha256"] != lock_hash:
        raise ValueError("installed receipt uses a different compute source lock")
    expected_vllm = {path: result for path, _, result in lock["vllm"]["files"]}
    if installed["vllm_overrides"] != expected_vllm:
        raise ValueError("installed receipt omits or changes vLLM overrides")
    for relative, expected in expected_vllm.items():
        path = PurePosixPath(relative)
        target = site_packages / path
        if (path.is_absolute() or str(path) != relative or ".." in path.parts
                or "\\" in relative or not relative.startswith("vllm/")
                or not target.resolve().is_relative_to((site_packages / "vllm").resolve())
                or any((site_packages / Path(*path.parts[:index])).is_symlink()
                       for index in range(1, len(path.parts) + 1))):
            raise ValueError(f"Unsafe vLLM override path: {relative}")
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"installed vLLM hash mismatch for {relative}: {actual}")
    expected_b12x = installed.get("b12x_files")
    if not expected_b12x:
        raise ValueError("installed receipt has no B12X package map")
    actual_b12x = {
        path.relative_to(site_packages).as_posix(): _sha256(path)
        for path in sorted((site_packages / "b12x").rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual_b12x != expected_b12x:
        raise ValueError("installed B12X package map mismatch")
    expected_b12x_hash = lock["b12x"]["package_files_sha256"]
    if installed.get("b12x_package_files_sha256") != expected_b12x_hash:
        raise ValueError("installed receipt has a different B12X package-map hash")
    if _map_sha256(actual_b12x) != expected_b12x_hash:
        raise ValueError("installed B12X package differs from source-lock.json")
    if installed["environment"] != lock["environment"]:
        raise ValueError("installed compute environment differs from source lock")
    if installed.get("target_head_quantization") is not False:
        raise ValueError("target LM head must remain unquantized")
    return {
        "checks_passed": True,
        "source_lock_sha256": lock_hash,
        "vllm_override_count": len(expected_vllm),
        "b12x_file_count": len(expected_b12x),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.site_packages, args.receipt, args.source_lock)))


if __name__ == "__main__":
    main()
