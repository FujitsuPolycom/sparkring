#!/opt/venv/bin/python
"""Model-neutral entrypoint consuming the canonical externally rendered R33 profile."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys


VERIFY = "/opt/sparkring/bin/verify-candidate"
PYTHON = "/opt/venv/bin/python"
PROFILE_ROOT = Path("/opt/sparkring/profile-contract")
ALLOWED_PROFILES = {
    "tp2-dcp1",
    "tp2-dcp1-sparkcache",
    "tp4-dcp1",
    "tp4-dcp1-sparkcache",
    "tp4-dcp4",
    "tp4-dcp4-sparkcache",
}
PLACEHOLDER_MARKERS = ("<", ">", "${")


def usage() -> str:
    return (
        "Usage: sparkring-r33 verify\n"
        "       sparkring-r33 serve MODEL [vLLM options...]\n"
        "Serving requires a canonical externally rendered SOURCE_IMAGE_PROFILE.\n"
        "TP4 containers must be created by the managed mesh renderer.\n"
        "The image contains no model and starts no service by default."
    )


def concrete(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or any(marker in value for marker in PLACEHOLDER_MARKERS):
        raise RuntimeError(f"external profile did not provide a concrete {name}")
    return value


def verification_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "SPARKRING_TRANSPORT_PROFILE", "SPARKRING_TRANSPORT_MANIFEST_SHA256"):
        environment.pop(name, None)
    return environment

CONTRACT_MOUNT_LOCK = Path("/opt/sparkring/receipts/source-lock.json")


def verify_installed_files() -> None:
    """Verify installed files, tolerating a mounted profile-contract overlay.

    With the contract overlay active, the mounted directory supersedes the
    baked contract files, so the source-lock identity check excludes that
    directory and verifies every other installed file itself. Without the
    overlay the baked verify-candidate runs unchanged.
    """
    if not os.environ.get("R33_PROFILE_CONTRACT_HOST_ROOT", ""):
        subprocess.run([VERIFY], check=True, env=verification_environment())
        return
    lock = json.loads(CONTRACT_MOUNT_LOCK.read_text())
    if lock.get("schema") != "sparkring-r33-candidate-source-lock/v1":
        raise RuntimeError("unsupported candidate source lock")
    if sys.prefix != "/opt/venv" or sys.base_prefix == sys.prefix:
        raise RuntimeError("candidate verifier is outside /opt/venv")
    for relative, expected in lock["installed_files"].items():
        if relative.startswith("opt/sparkring/profile-contract/"):
            continue
        digest = hashlib.sha256((Path("/") / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"installed file identity mismatch: /{relative}")


def validate_external_profile(root: Path = PROFILE_ROOT) -> dict:
    profile_name = concrete("SOURCE_IMAGE_PROFILE")
    if profile_name not in ALLOWED_PROFILES:
        raise RuntimeError(f"unsupported canonical profile: {profile_name}")
    if os.environ.get("SPARKRING_PROFILE_MODE") != "custom":
        raise RuntimeError("SPARKRING_PROFILE_MODE=custom is required")
    contract = json.loads((root / "profile-contract.json").read_text())
    selected = contract["profiles"].get(profile_name)
    if not selected:
        raise RuntimeError(f"profile is absent from the canonical contract: {profile_name}")
    subprocess.run(
        [PYTHON, str(root / "verify_profile.py"), "template", "--profile", profile_name,
         "--asset-root", str(root.parent / "runtime")],
        check=True,
        env=verification_environment(),
    )
    common = dict(contract["common_environment"])
    if selected.get("load_format"):
        common.update(LOAD_FORMAT=selected["load_format"], VLLM_PLUGINS=selected["plugins"])
    for key, expected in common.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"external profile requires {key}={expected}")
    if concrete("NODE_RANK") not in {str(rank) for rank in range(selected["node_count"])}:
        raise RuntimeError(f"NODE_RANK is outside {profile_name}")
    concrete("MASTER_ADDR")
    concrete("NCCL_IB_HCA")
    if profile_name.startswith("tp2-"):
        cache = selected["sparkcache"]
        required = {
            "SPARKRING_TRANSPORT_PROFILE": "tp2-rocenante-adaptive",
            "VLLM_B12X_KDA_PREFILL_COALESCING": "1" if cache else "0",
            "VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT": "4" if cache else "0",
            "VLLM_SPARK_TP4_MODE": "",
            "VLLM_SPARK_TP4_VOCAB_MODE": "",
            "SPARKCACHE_ENABLED": "1" if cache else "0",
        }
        for key, expected in required.items():
            if os.environ.get(key, "") != expected:
                raise RuntimeError(f"TP2 external profile requires {key}={expected}")
        concrete("B12X_ROCE_PEER_HCA_MAP")
        if cache:
            capability = root / selected["capability_file"]
            if not capability.is_file():
                raise RuntimeError("TP2 SparkCache is unavailable: missing source-bound runtime capability evidence")
            subprocess.run([PYTHON, str(root / "verify_profile.py"), "capability", "--profile", profile_name,
                            "--receipt", str(capability)], check=True, env=verification_environment())
            concrete("SPARKCACHE_CACHE_NAMESPACE")
            native = contract["sparkcache_native"]
            for key, expected in {
                "SPARKCACHE_PLACEMENT_LIBRARY_PATH": native["placement_path"],
                "SPARKCACHE_PLACEMENT_LIBRARY_SHA256": native["placement_sha256"],
                "SPARKCACHE_SNAPSHOT_LIBRARY_PATH": native["snapshot_path"],
                "SPARKCACHE_SNAPSHOT_LIBRARY_SHA256": native["snapshot_sha256"],
                "SPARKCACHE_VLLM_ROOT": native["vllm_root"],
                "SPARKCACHE_SOURCE_LEASE_CONTRACT": native["lease_contract"],
            }.items():
                if os.environ.get(key) != expected:
                    raise RuntimeError(f"TP2 SparkCache requires {key}={expected}")
    else:
        required = {
            "SPARKRING_MANAGED_MESH_RENDERED": "1",
            "VLLM_B12X_KDA_PREFILL_COALESCING": "1",
            "VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT": "4",
            "SIRCL_ENABLED": "1",
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
            "SPARK_TP4_LIBRARY": "/opt/sparkring/sircl/libspark_transport_capi.so",
            "NCCL_SWITCHLESS_RING_ONLY": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise RuntimeError(f"managed TP4 profile requires {key}={expected}")
        for key in (
            "SPARK_TP4_PEER0", "SPARK_TP4_PEER1",
            "SPARK_TP4_DEVICE0", "SPARK_TP4_DEVICE1",
            "SPARK_TP4_GID0", "SPARK_TP4_GID1",
            "SPARK_TP4_CONTROL_PORT0", "SPARK_TP4_CONTROL_PORT1",
        ):
            concrete(key)
        if profile_name in ("tp4-dcp1-sparkcache", "tp4-dcp4-sparkcache"):
            concrete("SPARKCACHE_CACHE_NAMESPACE")
            native = contract["sparkcache_native"]
            for key, expected in {
                "SPARKCACHE_PLACEMENT_LIBRARY_PATH": native["placement_path"],
                "SPARKCACHE_PLACEMENT_LIBRARY_SHA256": native["placement_sha256"],
                "SPARKCACHE_SNAPSHOT_LIBRARY_PATH": native["snapshot_path"],
                "SPARKCACHE_SNAPSHOT_LIBRARY_SHA256": native["snapshot_sha256"],
                "SPARKCACHE_VLLM_ROOT": native["vllm_root"],
                "SPARKCACHE_SOURCE_LEASE_CONTRACT": native["lease_contract"],
            }.items():
                if os.environ.get(key) != expected:
                    raise RuntimeError(f"managed SparkCache profile requires {key}={expected}")
        elif os.environ.get("SPARKCACHE_ENABLED") != "0":
            raise RuntimeError("cache-disabled TP4 profile requires SPARKCACHE_ENABLED=0")
    return selected


def serving_argv(argv: list[str]) -> list[str]:
    if argv == ["verify"]:
        return [VERIFY]
    if argv and argv[0] == "serve":
        if len(argv) < 2:
            raise RuntimeError("serve requires a model path or repository")
        validate_external_profile()
        return [PYTHON, "-m", "vllm.entrypoints.cli.main", *argv]
    raise RuntimeError(usage())


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        return 0
    argv = serving_argv(sys.argv[1:])
    verify_environment = verification_environment()
    if argv == [VERIFY]:
        os.execve(argv[0], argv, verify_environment)
    verify_installed_files()
    os.execve(argv[0], argv, os.environ)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"sparkring-r33: {error}", file=sys.stderr)
        raise SystemExit(2)
