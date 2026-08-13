#!/usr/bin/env python3
"""Build a hash-bound tiered/deferred SIRCL profile and staging bundle.

The source profile must already satisfy the 262K dynamic-NVFP4 and full-CKV
contracts.  The output mounts one locally compiled native library and three
public adapter modules, attests every mounted byte at container startup, and
enables the operator-qualified two-slot deferred-ack/tiered-64K selector.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import prepare_exl3_r7_mtp4_ckv_gather as ckv


class ContractError(ValueError):
    """The source or derived SIRCL profile violates its bounded contract."""


REMOTE_ROOT = "/var/tmp/sparkring-sircl-tiered-v1"
LIBRARY_CONTAINER = "/opt/sparkring/spark_transport/libspark_transport_capi.so"
BACKEND_CONTAINER = "/opt/spark-vllm/spark_tp4_backend.py"
PORT_NAMESPACE_CONTAINER = "/opt/spark-vllm/spark_tp4_port_namespace.py"
CAPACITY_POOL_CONTAINER = "/opt/spark-vllm/spark_tp4_prefill_capacity_pool.py"
ARTIFACTS = {
    "transport_library": ("libspark_transport_capi.so", LIBRARY_CONTAINER),
    "backend": ("spark_tp4_backend.py", BACKEND_CONTAINER),
    "port_namespace": ("spark_tp4_port_namespace.py", PORT_NAMESPACE_CONTAINER),
    "capacity_pool": ("spark_tp4_prefill_capacity_pool.py", CAPACITY_POOL_CONTAINER),
}
ENVIRONMENT_DELTA = {
    "SPARK_TP4_CONTROL_PORT0": "11100",
    "SPARK_TP4_CONTROL_PORT1": "11101",
    "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL": "two_slot_deferred_ack",
    "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY": "tiered_64k",
}
LABEL_DELTA = {
    "org.sparkring.sircl-graph-protocol": "two-slot-deferred-ack",
    "org.sparkring.sircl-graph-kernel": "tiered-64k",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum_line(digest: str, path: str) -> str:
    return f"{digest}  {path}"


def _expected_sha256(value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ContractError("expected base profile SHA-256 must be 64 lowercase hex")
    return value


def validate_source_profile(profile: dict[str, Any]) -> None:
    """Require the full-CKV derivative before enabling native transport."""

    environment = profile.get("environment")
    labels = profile.get("extra_labels")
    volumes = profile.get("extra_volumes")
    if not isinstance(environment, dict) or not isinstance(labels, dict):
        raise ContractError("source environment or labels are malformed")
    if not isinstance(volumes, list):
        raise ContractError("source extra_volumes is malformed")
    ckv_source = copy.deepcopy(profile)
    ckv_source["profile_id"] = ckv.SOURCE_PROFILE_ID
    ckv_source["environment"].pop("VLLM_B12X_MLA_CKV_GATHER", None)
    ckv_source["environment"].pop("VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS", None)
    ckv_source["extra_labels"].pop(ckv.CKV_LABEL, None)
    try:
        ckv.validate_source_profile(ckv_source)
    except ckv.ContractError as exc:
        raise ContractError(f"source pre-CKV contract drifted: {exc}") from exc
    if profile.get("profile_id") != ckv.CANDIDATE_PROFILE_ID:
        raise ContractError("source profile_id is not the full-CKV derivative")
    if environment.get("VLLM_B12X_MLA_CKV_GATHER") != "1":
        raise ContractError("source full-CKV gather must be enabled")
    if environment.get("VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS") != "262144":
        raise ContractError("source CKV gather ceiling must be 262144")
    if labels.get(ckv.CKV_LABEL) != ckv.CKV_LABEL_VALUE:
        raise ContractError("source CKV label is missing")
    for key in ENVIRONMENT_DELTA:
        if key in environment:
            raise ContractError(f"source profile already declares {key}")
    for _name, (_filename, container) in ARTIFACTS.items():
        if any(volume.get("container") == container for volume in volumes):
            raise ContractError(f"source profile already mounts {container}")


def _inject_attestation(command: str, digests: dict[str, str]) -> str:
    marker = " | sha256sum --check --strict -"
    if command.count(marker) != 1:
        raise ContractError("source attestation checksum marker is absent or non-unique")
    additions = []
    for name, (_filename, container) in ARTIFACTS.items():
        additions.append(repr(_checksum_line(digests[name], container)))
    return command.replace(marker, " " + " ".join(additions) + marker, 1)


def derive_candidate(
    source: dict[str, Any], *, artifact_digests: dict[str, str]
) -> dict[str, Any]:
    """Return the exact tiered/deferred SIRCL derivative."""

    validate_source_profile(source)
    if set(artifact_digests) != set(ARTIFACTS):
        raise ContractError("SIRCL artifact digest inventory is incomplete")
    for name, digest in artifact_digests.items():
        if not _SHA256_RE.fullmatch(digest):
            raise ContractError(f"{name} SHA-256 is invalid")
    candidate = copy.deepcopy(source)
    candidate["profile_id"] = f"{source['profile_id']}-sircl-tiered"
    candidate["container_name"] = "glm52-sparkring-sircl-tiered"
    candidate["confirmation"] = "START-SIRCL-TIERED-ALL-FOUR"
    candidate["environment"].update(ENVIRONMENT_DELTA)
    candidate["extra_labels"].update(LABEL_DELTA)
    for name, (filename, container) in ARTIFACTS.items():
        candidate["extra_volumes"].append(
            {
                "host": f"{REMOTE_ROOT}/{filename}",
                "container": container,
                "mode": "ro",
            }
        )
    hook = candidate.get("attestation_hook")
    if not isinstance(hook, list) or len(hook) != 3 or not isinstance(hook[2], str):
        raise ContractError("source attestation hook shape drifted")
    hook[2] = _inject_attestation(hook[2], artifact_digests)
    validate_candidate(source, candidate, artifact_digests=artifact_digests)
    return candidate


def validate_candidate(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    artifact_digests: dict[str, str],
) -> None:
    """Require exactly the declared identity, flag, mount, and hash delta."""

    validate_source_profile(source)
    expected = copy.deepcopy(source)
    expected["profile_id"] = f"{source['profile_id']}-sircl-tiered"
    expected["container_name"] = "glm52-sparkring-sircl-tiered"
    expected["confirmation"] = "START-SIRCL-TIERED-ALL-FOUR"
    expected["environment"].update(ENVIRONMENT_DELTA)
    expected["extra_labels"].update(LABEL_DELTA)
    for name, (filename, container) in ARTIFACTS.items():
        expected["extra_volumes"].append(
            {
                "host": f"{REMOTE_ROOT}/{filename}",
                "container": container,
                "mode": "ro",
            }
        )
    expected["attestation_hook"][2] = _inject_attestation(
        expected["attestation_hook"][2], artifact_digests
    )
    if candidate != expected:
        raise ContractError("candidate differs outside the tiered SIRCL allowlist")


def prepare(
    *,
    base_profile: Path,
    expected_base_profile_sha256: str,
    artifact_paths: dict[str, Path],
    bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bundle.exists():
        raise ContractError(f"refusing to replace bundle {bundle}")
    if sha256(base_profile) != _expected_sha256(expected_base_profile_sha256):
        raise ContractError("base profile SHA-256 mismatch")
    if set(artifact_paths) != set(ARTIFACTS):
        raise ContractError("SIRCL artifact path inventory is incomplete")
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise ContractError(f"{name} is not a regular file")
    digests = {name: sha256(path) for name, path in artifact_paths.items()}
    source = json.loads(base_profile.read_text(encoding="utf-8"))
    candidate = derive_candidate(source, artifact_digests=digests)

    bundle.mkdir(parents=True)
    files: dict[str, dict[str, str | int]] = {}
    for name, source_path in artifact_paths.items():
        filename, _container = ARTIFACTS[name]
        destination = bundle / filename
        shutil.copyfile(source_path, destination)
        files[name] = {
            "path": str(destination.resolve()),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
    manifest = {
        "schema": "sparkring-r7-sircl-tiered-bundle/v1",
        "maturity": "offline-validated",
        "remote_root": REMOTE_ROOT,
        "base_profile": {
            "path": str(base_profile.resolve()),
            "sha256": expected_base_profile_sha256,
        },
        "candidate_profile_id": candidate["profile_id"],
        "files": files,
        "policy": {
            "graph_protocol": "two_slot_deferred_ack",
            "kernel_strategy": "tiered_64k",
            "dual_port": False,
            "prefill_capacity_pool": False,
        },
        "rollback": {
            "profile": str(base_profile.resolve()),
            "sha256": expected_base_profile_sha256,
        },
    }
    return candidate, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--expected-base-profile-sha256", required=True)
    parser.add_argument("--transport-library", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--port-namespace", type=Path, required=True)
    parser.add_argument("--capacity-pool", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_profile, args.output_manifest):
        if output.exists():
            parser.error(f"refusing to replace {output}")
    try:
        candidate, manifest = prepare(
            base_profile=args.base_profile.resolve(),
            expected_base_profile_sha256=args.expected_base_profile_sha256,
            artifact_paths={
                "transport_library": args.transport_library.resolve(),
                "backend": args.backend.resolve(),
                "port_namespace": args.port_namespace.resolve(),
                "capacity_pool": args.capacity_pool.resolve(),
            },
            bundle=args.bundle.resolve(),
        )
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        parser.error(str(exc))
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"profile_sha256={sha256(args.output_profile)}")
    print(f"manifest_sha256={sha256(args.output_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
