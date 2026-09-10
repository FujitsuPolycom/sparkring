#!/usr/bin/env python3
"""Verify an R33 ARM64 vLLM wheel without importing GPU-dependent modules."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import struct
import subprocess
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--native-tree", required=True)
    parser.add_argument("--native-install", type=Path, required=True)
    parser.add_argument("--native-modules", type=Path, required=True)
    parser.add_argument("--flash-attn-source-dir", type=Path, required=True)
    parser.add_argument("--flash-attn-commit", required=True)
    parser.add_argument("--rust-bin-sha256", required=True)
    parser.add_argument("--rust-parser-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tracked_flash_attn_python(
    source_dir: Path, commit: str
) -> dict[str, tuple[str, bytes]]:
    """Map commit-pinned external helpers to their vLLM wheel destinations."""
    output = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={source_dir}",
            "-C",
            str(source_dir),
            "ls-tree",
            "-r",
            commit,
            "--",
            "vllm_flash_attn",
        ],
        text=True,
    )
    files = {}
    excluded = {
        "vllm_flash_attn/__init__.py",
        "vllm_flash_attn/flash_attn_interface.py",
    }
    for line in output.splitlines():
        metadata_part, source_name = line.split("\t", 1)
        mode, object_type, _ = metadata_part.split()
        if (
            object_type != "blob"
            or mode == "120000"
            or not source_name.endswith(".py")
            or source_name in excluded
        ):
            continue
        relative = source_name.removeprefix("vllm_flash_attn/")
        destination = f"vllm/vllm_flash_attn/{relative}"
        data = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={source_dir}",
                "-C",
                str(source_dir),
                "show",
                f"{commit}:{source_name}",
            ]
        )
        files[destination] = (source_name, data)
    return files


def main() -> None:
    args = parse_args()
    expected_modules = {
        "vllm/_C_stable_libtorch.abi3.so",
        "vllm/cumem_allocator.abi3.so",
        "vllm/fs_io_C.abi3.so",
        "vllm/_flashkda_C.abi3.so",
        "vllm/_moe_C_stable_libtorch.abi3.so",
        "vllm/_qutlass_C.abi3.so",
        "vllm/spinloop.abi3.so",
        "vllm/third_party/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so",
        "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
        "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
        "vllm/_rust_tool_parser.abi3.so",
    }
    expected_name = re.compile(r"^vllm-.+-cp312-cp312-linux_aarch64\.whl$")
    assert expected_name.match(args.wheel.name), args.wheel.name

    with zipfile.ZipFile(args.wheel) as archive:
        member_names = archive.namelist()
        names = set(member_names)
        assert len(names) == len(member_names), "duplicate wheel members"
        assert expected_modules <= names, sorted(expected_modules - names)
        assert "vllm/vllm-rs" in names
        flash_attn_python = tracked_flash_attn_python(
            args.flash_attn_source_dir, args.flash_attn_commit
        )
        assert "vllm/vllm_flash_attn/layers/rotary.py" in flash_attn_python
        for destination, (_, source_data) in flash_attn_python.items():
            assert destination in names, destination
            assert archive.read(destination) == source_data, destination
        dist_info = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(dist_info) == 1, dist_info
        metadata = BytesParser(policy=default).parsebytes(archive.read(dist_info[0]))
        assert metadata["Name"] == "vllm"
        assert metadata["Version"] == args.version
        requirements = metadata.get_all("Requires-Dist", [])
        assert any(req.startswith("torch==2.13.0") for req in requirements)
        assert any(req.startswith("flashinfer-python==0.6.18") for req in requirements)
        assert not any("flashinfer-cubin" in req for req in requirements)

        wheel_meta_name = dist_info[0].replace("METADATA", "WHEEL")
        wheel_meta = archive.read(wheel_meta_name).decode()
        assert "Root-Is-Purelib: false" in wheel_meta
        assert "Tag: cp312-cp312-linux_aarch64" in wheel_meta

        native_sources = {
            "vllm/_C_stable_libtorch.abi3.so": args.native_modules / "_C_stable_libtorch.abi3.so",
            "vllm/cumem_allocator.abi3.so": args.native_modules / "cumem_allocator.abi3.so",
            "vllm/fs_io_C.abi3.so": args.native_modules / "fs_io_C.abi3.so",
            "vllm/_flashkda_C.abi3.so": args.native_modules / "_flashkda_C.abi3.so",
            "vllm/_moe_C_stable_libtorch.abi3.so": args.native_modules / "_moe_C_stable_libtorch.abi3.so",
            "vllm/_qutlass_C.abi3.so": args.native_modules / "_qutlass_C.abi3.so",
            "vllm/spinloop.abi3.so": args.native_modules / "spinloop.abi3.so",
            "vllm/third_party/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so": args.native_modules / "deepgemm_C_cpython-312-aarch64-linux-gnu/_C.cpython-312-aarch64-linux-gnu.so",
            "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so": args.native_modules / "vllm-flash-attn/_vllm_fa2_C.abi3.so",
            "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so": args.native_modules / "vllm-flash-attn/_vllm_fa3_C.abi3.so",
        }
        elf = {}
        for name in sorted(expected_modules):
            data = archive.read(name)
            assert data[:4] == b"\x7fELF", name
            assert data[4] == 2 and data[5] == 1, name  # ELF64, little-endian
            machine = struct.unpack_from("<H", data, 18)[0]
            assert machine == 183, (name, machine)  # EM_AARCH64
            if name in native_sources:
                assert sha256(data) == sha256(native_sources[name].read_bytes()), name
            elf[name] = {"sha256": sha256(data), "size": len(data), "machine": "AArch64"}

        assert sha256(archive.read("vllm/vllm-rs")) == args.rust_bin_sha256
        assert sha256(archive.read("vllm/_rust_tool_parser.abi3.so")) == args.rust_parser_sha256

        record_name = dist_info[0].replace("METADATA", "RECORD")
        rows = list(csv.reader(archive.read(record_name).decode().splitlines()))
        row_map = {row[0]: row[1:] for row in rows}
        assert set(row_map) == names
        for name in sorted(names - {record_name}):
            data = archive.read(name)
            digest, size = row_map[name]
            assert digest == record_digest(data), name
            assert size == str(len(data)), name
        assert row_map[record_name] == ["", ""]

        install_files = {
            PurePosixPath(path).as_posix()
            for path in (
                item.relative_to(args.native_install)
                for item in args.native_install.rglob("*")
                if item.is_file()
            )
        }
        missing_install_files = sorted(install_files - names)
        assert not missing_install_files, missing_install_files[:20]

        tracked_output = subprocess.check_output(
            ["git", "-C", str(args.source_dir), "ls-tree", "-r", args.source_tree, "--", "vllm"],
            text=True,
        )
        source_files_checked = 0
        for line in tracked_output.splitlines():
            metadata_part, name = line.split("\t", 1)
            mode, object_type, _ = metadata_part.split()
            if object_type != "blob" or mode == "120000":
                continue
            assert name in names, name
            source_data = subprocess.check_output(
                ["git", "-C", str(args.source_dir), "show", f"{args.source_tree}:{name}"]
            )
            if name == "vllm/_version.py":
                # setuptools-scm intentionally materializes the exact override here.
                continue
            assert archive.read(name) == source_data, name
            source_files_checked += 1

        result = {
            "status": "package-structure-and-integrity-qualified-runtime-pending",
            "wheel": args.wheel.name,
            "wheel_sha256": sha256(args.wheel.read_bytes()),
            "wheel_size": args.wheel.stat().st_size,
            "version": args.version,
            "source_tree": args.source_tree,
            "native_source_tree": args.native_tree,
            "flash_attn_source_commit": args.flash_attn_commit,
            "flash_attn_python_files_byte_checked": len(flash_attn_python),
            "native_to_package_diff": ["requirements/cuda.txt"],
            "tag": "cp312-cp312-linux_aarch64",
            "requires_dist": requirements,
            "file_count": len(names),
            "cmake_install_file_count": len(install_files),
            "tracked_source_files_byte_checked": source_files_checked,
            "native_and_rust_elf": elf,
            "record_entries": len(rows),
            "record_valid": True,
            "runtime_qualification": "pending",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
