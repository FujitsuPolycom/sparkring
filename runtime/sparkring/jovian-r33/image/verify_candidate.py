#!/opt/venv/bin/python
"""Verify the installed R33 candidate closure without initializing CUDA."""
from __future__ import annotations

import ctypes
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path("/opt/sparkring")
LOCK = ROOT / "receipts/source-lock.json"
NCCL = Path("/opt/local-inference/nccl/lib/libnccl.so.2")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    lock = json.loads(LOCK.read_text())
    if lock.get("schema") != "sparkring-r33-candidate-source-lock/v1":
        raise RuntimeError("unsupported candidate source lock")
    if sys.prefix != "/opt/venv" or sys.base_prefix == sys.prefix:
        raise RuntimeError("candidate verifier is outside /opt/venv")

    checked_files = {}
    for relative, expected in lock["installed_files"].items():
        path = Path("/") / relative
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"installed file identity mismatch: /{relative}")
        checked_files["/" + relative] = actual

    installed = {}
    for name, expected in lock["python_closure"].items():
        distribution = metadata.distribution(name)
        if distribution.version != expected:
            raise RuntimeError(f"distribution mismatch: {name} {distribution.version} != {expected}")
        installed[name] = distribution.version
    for name in lock["venv_direct_distributions"]:
        distribution = metadata.distribution(name)
        if not str(distribution._path).startswith("/opt/venv/"):
            raise RuntimeError(f"direct artifact is not installed in /opt/venv: {name}")

    payload_manifest_path = ROOT / "receipts/installed-python-files.json"
    if digest(payload_manifest_path) != lock["installed_python_manifest_sha256"]:
        raise RuntimeError("installed Python payload manifest changed")
    payload_manifest = json.loads(payload_manifest_path.read_text())
    if (payload_manifest.get("schema") != "sparkring-r33-installed-python-files/v1"
            or len(payload_manifest.get("files", {})) != lock["installed_python_file_count"]):
        raise RuntimeError("installed Python payload manifest is incomplete")
    compatibility = payload_manifest.get("xgrammar_transformers5_import", {})
    if compatibility.get("passed") is not True or not compatibility.get("transformers_version", "").startswith("5."):
        raise RuntimeError("XGrammar was not import-verified with Transformers 5")
    payload_files = 0
    for relative, expected in payload_manifest["files"].items():
        path = Path("/opt/venv") / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"installed Python payload differs: {relative}")
        payload_files += 1

    expected_nccl = lock["identities"]["nccl_sha256"]
    if NCCL.resolve() != Path("/opt/local-inference/nccl/lib/libnccl.so.2.31.2"):
        raise RuntimeError("NCCL SONAME does not resolve to the candidate library")
    if digest(NCCL.resolve()) != expected_nccl:
        raise RuntimeError("candidate NCCL hash mismatch")
    loaded = ctypes.CDLL(str(NCCL), mode=ctypes.RTLD_GLOBAL)
    version = ctypes.c_int()
    if loaded.ncclGetVersion(ctypes.byref(version)) != 0 or version.value != 23102:
        raise RuntimeError(f"unexpected NCCL runtime version: {version.value}")
    mapped = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libnccl.so" in line
    }
    if mapped != {str(NCCL.resolve())}:
        raise RuntimeError(f"NCCL loaded from more than one path: {sorted(mapped)}")

    ffmpeg = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    result = {
        "schema": "sparkring-r33-candidate-verification/v1",
        "status": "source-closure-verified-runtime-qualification-pending",
        "source_lock_sha256": digest(LOCK),
        "checked_files": checked_files,
        "python_distributions": installed,
        "installed_python_payload_files": payload_files,
        "nccl_version": version.value,
        "nccl_loaded_path": next(iter(mapped)),
        "ffmpeg": ffmpeg,
        "cuda_initialized": False,
        "model_loaded": False,
        "gpu_qualified": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
