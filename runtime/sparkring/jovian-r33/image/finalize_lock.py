#!/usr/bin/env python3
"""Finalize the immutable candidate source and dependency closure lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
import zipfile

from validate_receipts import load_json, validate as validate_receipts


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if len(Path(name).parts) == 2
            and Path(name).parts[0].endswith(".dist-info")
            and Path(name).parts[1] == "METADATA"
        ]
        if len(members) != 1:
            raise RuntimeError(f"wheel metadata is ambiguous: {path}")
        metadata = BytesParser(policy=default).parsebytes(archive.read(members[0]))
        return metadata["Name"], metadata["Version"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    context = args.context.resolve()
    target = context / "source-lock.json"
    if target.exists():
        raise RuntimeError(f"source lock already exists: {target}")
    partial = json.loads((context / "source-lock.partial.json").read_text())
    closure = json.loads((context / "closure-downloads.json").read_text())

    wheels = {}
    for path in sorted((context / "wheelhouse").glob("*.whl")):
        name, version = wheel_metadata(path)
        key = name.lower().replace("_", "-")
        if key in wheels and wheels[key]["sha256"] != digest(path):
            raise RuntimeError(f"multiple wheel artifacts for {key}")
        wheels[key] = {"version": version, "filename": path.name, "sha256": digest(path), "size": path.stat().st_size}

    closure_versions = {
        name.lower().replace("_", "-"): item["version"]
        for name, item in closure["downloads"].items()
    }
    inherited_distributions: set[str] = set()
    closure_install_wheels = (context / "closure-install-wheels.txt").read_text().splitlines()
    installed_python = json.loads((context / "installed-python-files.json").read_text())
    if installed_python.get("schema") != "sparkring-r33-installed-python-files/v1":
        raise RuntimeError("installed Python payload manifest is invalid")
    direct = []
    for filename in partial["install_wheels"]:
        name, _ = wheel_metadata(context / "wheelhouse" / filename)
        direct.append(name.lower().replace("_", "-"))

    installed_files = {
        "opt/local-inference/nccl/lib/libnccl.so.2.31.2": partial["inputs"]["nccl-2.31.2-sparkring-routing"]["sha256"],
        "opt/lmcache/lib/liblmcache_cumem_shareable.so": partial["inputs"]["lmcache-cumem-interposer"]["sha256"],
        "opt/sparkring/sircl/libspark_transport_capi.so": partial["inputs"]["sircl"]["sha256"],
        "opt/sparkring/sparkcache/lib/libspark_cache_placement.so": partial["inputs"]["sparkcache-placement"]["sha256"],
        "opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so": partial["inputs"]["sparkcache-snapshot"]["sha256"],
        "opt/sparkring/bin/sparkring-r33": digest(context / "entrypoint.py"),
        "opt/sparkring/bin/verify-candidate": digest(context / "verify_candidate.py"),
        "opt/sparkring/image/artifact-lock.json": digest(context / "receipts/artifact-lock.json"),
    }
    for directory, installed_prefix in (("profile-contract", "opt/sparkring/profile-contract"), ("profile-assets", "opt/sparkring/runtime"), ("transports", "opt/sparkring/transports"), ("sircl-python", "opt/sparkring/sircl/python"), ("contracts", "opt/sparkring/contracts"), ("receipts", "opt/sparkring/receipts")):
        for path in sorted((context / directory).rglob("*")):
            if path.is_file():
                installed_files[f"{installed_prefix}/{path.relative_to(context / directory).as_posix()}"] = digest(path)

    context_files = {}
    for path in sorted(context.rglob("*")):
        if path.is_file() and path != target:
            context_files[path.relative_to(context).as_posix()] = digest(path)
    component_receipts = {
        path.name: digest(path)
        for path in sorted((context / "receipts").iterdir())
        if path.is_file()
    }
    receipt_semantics = validate_receipts(
        load_json(context / "receipts/artifact-lock.json"), partial["inputs"], context / "receipts"
    )
    if receipt_semantics != load_json(context / "receipts/semantic-validation.json"):
        raise RuntimeError("semantic receipt validation changed after context preparation")
    result = {
        "schema": "sparkring-r33-candidate-source-lock/v1",
        "status": "context-and-python-closure-locked-image-build-pending",
        "foundation": partial["foundation"],
        "media_runtime": partial["media_runtime"],
        "inputs": partial["inputs"],
        "component_receipts": component_receipts,
        "receipt_semantics": receipt_semantics,
        "wheelhouse": wheels,
        "python_closure": closure_versions,
        "closure_install_wheels": closure_install_wheels,
        "installed_python_manifest_sha256": digest(context / "installed-python-files.json"),
        "installed_python_file_count": len(installed_python["files"]),
        "venv_direct_distributions": sorted(set(direct)),
        "inherited_distributions": sorted(inherited_distributions),
        "installed_files": installed_files,
        "context_files": context_files,
        "identities": {
            "component_sources": partial["source_identities"],
            "vllm_source_tree": "667ee2f6652efa065c57a7adc0193991f6cde6ac",
            "nccl_patched_tree": "aa7028b2b2a55af4817f8d742e17717dd4509ee7",
            "nccl_sha256": "84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47",
            "sircl_source_tree": "18e13dad5071b0a153f2c33c9c1720a3eca50318",
            "sircl_sha256": "bea00f2ba6051c2c0bcd2853aae894672aa7f1fe5a1d905edaa9120aabf74246"
        },
        "qualification": {"image_built": False, "gpu": False, "model": False, "tp2": False, "tp4": False, "cache_recovery": False}
    }
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_lock": str(target), "wheelhouse_files": len(wheels), "closure_distributions": len(closure_versions)}, indent=2))


if __name__ == "__main__":
    main()
