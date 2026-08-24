#!/usr/bin/env python3
"""Attest the rebuilt GB10 native component and its final installed copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from apply_runtime_overlay import DEFAULT_CONTRACT, OverlayError, load_contract, sha256_file


def _run(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise OverlayError(
            f"command failed ({result.returncode}): {' '.join(argv)}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _digest_lines(values: list[str]) -> str:
    payload = "\n".join(sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def elf_profile(path: Path) -> dict[str, Any]:
    header = _run(["readelf", "-hW", str(path)])
    dynamic = _run(["readelf", "-dW", str(path)])
    versions = _run(["readelf", "--version-info", "-W", str(path)])
    notes = _run(["readelf", "-nW", str(path)])
    symbols = _run(["nm", "-D", "--defined-only", "--format=posix", str(path)])
    imports = _run(["nm", "-D", "--undefined-only", "--format=posix", str(path)])

    def header_value(label: str) -> str:
        match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", header, re.MULTILINE)
        if match is None:
            raise OverlayError(f"ELF header omits {label}: {path}")
        return match.group(1).strip()

    needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic)
    rpath = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]*)\]", dynamic)
    version_needs: list[str] = []
    in_needs = False
    for line in versions.splitlines():
        if line.startswith("Version needs section"):
            in_needs = True
            continue
        if in_needs and line.startswith("Version definition section"):
            in_needs = False
        if in_needs:
            file_match = re.search(r"File:\s*(\S+)", line)
            name_match = re.search(r"Name:\s*(\S+)", line)
            if file_match:
                version_needs.append(f"file:{file_match.group(1)}")
            if name_match:
                version_needs.append(f"name:{name_match.group(1)}")
    symbol_names = [
        line.split()[0]
        for line in symbols.splitlines()
        if line.strip() and len(line.split()) >= 2
    ]
    import_names = [
        line.split()[0]
        for line in imports.splitlines()
        if line.strip() and len(line.split()) >= 2
    ]
    build_id = re.search(r"Build ID:\s*([0-9a-fA-F]+)", notes)
    if build_id is None:
        raise OverlayError(f"ELF build ID is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "class": header_value("Class"),
        "data": header_value("Data"),
        "type": header_value("Type"),
        "machine": header_value("Machine"),
        "needed": sorted(set(needed)),
        "rpath_runpath": sorted(set(rpath)),
        "gnu_version_needs": sorted(set(version_needs)),
        "gnu_version_needs_sha256": _digest_lines(version_needs),
        "exported_symbol_count": len(set(symbol_names)),
        "exported_symbol_set_sha256": _digest_lines(symbol_names),
        "imported_symbol_count": len(set(import_names)),
        "imported_symbol_set_sha256": _digest_lines(import_names),
        "build_id": build_id.group(1).lower(),
    }


def compare_abi(old: dict[str, Any], new: dict[str, Any]) -> None:
    for field in (
        "class",
        "data",
        "type",
        "machine",
        "needed",
        "rpath_runpath",
        "gnu_version_needs_sha256",
        "exported_symbol_count",
        "exported_symbol_set_sha256",
        "imported_symbol_count",
        "imported_symbol_set_sha256",
    ):
        if old[field] != new[field]:
            raise OverlayError(
                f"native ABI surface differs for {field}: "
                f"old={old[field]!r}, new={new[field]!r}"
            )
    if old["build_id"] == new["build_id"]:
        raise OverlayError("rebuilt native component retained the old ELF build ID")


def create_receipt(
    old_path: Path,
    new_path: Path,
    component_root: Path,
    source_receipt_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    native = contract["native_patch"]
    files = sorted(
        str(path.relative_to(component_root))
        for path in component_root.rglob("*")
        if path.is_file()
    )
    if files != ["vllm/_C_stable_libtorch.abi3.so"]:
        raise OverlayError(f"native component install contains unexpected files: {files}")

    old = elf_profile(old_path)
    new = elf_profile(new_path)
    if old["sha256"] != native["installed_preimage_sha256"]:
        raise OverlayError("installed native library is not the contracted 6fc preimage")
    if new["size_bytes"] != native["result_size_bytes"]:
        raise OverlayError("rebuilt native library size differs from the qualified artifact")
    compare_abi(old, new)
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if source_receipt.get("schema") != "sparkring-deepseek-v4-gb10-native-source/v1":
        raise OverlayError("native source receipt has an unsupported schema")
    return {
        "schema": "sparkring-deepseek-v4-gb10-native-artifact/v1",
        "status": "built-and-abi-attested",
        "base_image": contract["base_image"],
        "source_receipt": source_receipt,
        "component_files": files,
        "reference_artifact": {
            "sha256": native["reference_result_sha256"],
            "matches_this_build": new["sha256"]
            == native["reference_result_sha256"],
        },
        "old": old,
        "new": new,
    }


def verify_installed(
    installed_path: Path,
    source_root: Path,
    receipt_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    native = contract["native_patch"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "sparkring-deepseek-v4-gb10-native-artifact/v1":
        raise OverlayError("native artifact receipt has an unsupported schema")
    installed = elf_profile(installed_path)
    expected = receipt["new"]
    for field in (
        "sha256",
        "size_bytes",
        "class",
        "data",
        "type",
        "machine",
        "needed",
        "rpath_runpath",
        "gnu_version_needs_sha256",
        "exported_symbol_count",
        "exported_symbol_set_sha256",
        "imported_symbol_count",
        "imported_symbol_set_sha256",
        "build_id",
    ):
        if installed[field] != expected[field]:
            raise OverlayError(f"final installed native field differs: {field}")
    headers = []
    for record in native["files"]:
        path = (source_root / record["path"]).resolve(strict=True)
        observed = sha256_file(path)
        if observed != record["result_sha256"]:
            raise OverlayError(f"final native source header differs: {path}")
        headers.append({"path": str(path), "sha256": observed})
    return {
        "schema": "sparkring-deepseek-v4-gb10-final-native/v1",
        "status": "pass",
        "base_image": contract["base_image"],
        "installed": installed,
        "source_headers": headers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--new", type=Path)
    parser.add_argument("--component-root", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--verify-installed", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifact-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        contract_path = args.contract.resolve(strict=True)
        if args.verify_installed is not None:
            if args.source_root is None or args.artifact_receipt is None:
                raise OverlayError(
                    "--verify-installed requires --source-root and --artifact-receipt"
                )
            result = verify_installed(
                args.verify_installed.resolve(strict=True),
                args.source_root.resolve(strict=True),
                args.artifact_receipt.resolve(strict=True),
                contract_path,
            )
        else:
            if any(
                value is None
                for value in (
                    args.old,
                    args.new,
                    args.component_root,
                    args.source_receipt,
                )
            ):
                raise OverlayError(
                    "receipt creation requires --old, --new, --component-root, and --source-receipt"
                )
            result = create_receipt(
                args.old.resolve(strict=True),
                args.new.resolve(strict=True),
                args.component_root.resolve(strict=True),
                args.source_receipt.resolve(strict=True),
                contract_path,
            )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
    except (FileExistsError, OSError, OverlayError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
