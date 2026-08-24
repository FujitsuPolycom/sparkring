#!/usr/bin/env python3
"""Patch the exact retained GB10 source that owns _C_stable_libtorch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from apply_runtime_overlay import (
    DEFAULT_CONTRACT,
    FileContract,
    OverlayError,
    apply_patch_set,
    load_contract,
    sha256_file,
)


def _run(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise OverlayError(
            f"command failed ({result.returncode}): {' '.join(argv)}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def build_receipt(source_root: Path, contract_path: Path) -> dict[str, object]:
    contract = load_contract(contract_path)
    native = contract["native_patch"]
    expected_commit = str(native["source_commit"])
    observed_commit = _run(
        ["git", "-c", f"safe.directory={source_root}", "-C", str(source_root), "rev-parse", "HEAD"]
    )
    if observed_commit != expected_commit:
        raise OverlayError(
            f"retained vLLM commit differs: expected {expected_commit}, got {observed_commit}"
        )

    build_directory = Path(str(native["build_directory"])).resolve(strict=True)
    expected_build_directory = (
        source_root / "build" / "temp.linux-aarch64-cpython-312"
    ).resolve(strict=True)
    if build_directory != expected_build_directory:
        raise OverlayError(
            f"native build directory escapes the retained source: {build_directory}"
        )
    build_inputs = {}
    expected_build_inputs = native["build_inputs_sha256"]
    for name in ("CMakeCache.txt", "build.ninja"):
        path = build_directory / name
        if not path.is_file():
            raise OverlayError(f"retained native build input is missing: {path}")
        observed = sha256_file(path)
        expected = str(expected_build_inputs[name])
        if observed != expected:
            raise OverlayError(
                f"retained {name} differs: expected {expected}, got {observed}"
            )
        build_inputs[name] = observed

    installed = Path(str(native["installed_path"])).resolve(strict=True)
    observed_installed = sha256_file(installed)
    expected_installed = str(native["installed_preimage_sha256"])
    if observed_installed != expected_installed:
        raise OverlayError(
            f"installed native preimage differs: expected {expected_installed}, "
            f"got {observed_installed}"
        )

    patch_path = (contract_path.parent / str(native["path"])).resolve(strict=True)
    records = tuple(
        FileContract(
            path=str(value["path"]),
            preimage_sha256=str(value["preimage_sha256"]),
            result_sha256=str(value["result_sha256"]),
        )
        for value in native["files"]
    )
    status = apply_patch_set(
        source_root,
        patch_path,
        records,
        str(native["sha256"]),
    )
    return {
        "schema": "sparkring-deepseek-v4-gb10-native-source/v1",
        "status": status,
        "base_image": contract["base_image"],
        "source_root": str(source_root),
        "source_commit": observed_commit,
        "build_directory": str(build_directory),
        "build_inputs": build_inputs,
        "installed_preimage": {
            "path": str(installed),
            "sha256": observed_installed,
        },
        "patch_sha256": sha256_file(patch_path),
        "headers": [
            {
                "path": str((source_root / item.path).resolve()),
                "sha256": item.result_sha256,
            }
            for item in records
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_receipt(
            args.source_root.resolve(strict=True),
            args.contract.resolve(strict=True),
        )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.receipt is None:
            print(payload, end="")
        else:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
    except (FileExistsError, OSError, OverlayError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
