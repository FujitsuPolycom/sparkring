#!/usr/bin/env python3
"""Emit a non-executing GLM-5.3 virtual-diagonal overlay and cleanup plan.

Status: research-only. This program reads local configuration and writes JSON.
It never invokes SSH, Docker, ip, tc, RDMA, or a model process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spark_transport.experiments.cx7_hairpin_diagonal import fabric
from spark_transport.experiments.glm53_rocenante_overlay.rocenante_vllm_overlay import (
    load_contract,
)


PLAN_SCHEMA = "sparkring.glm53-rocenante-full-model-plan/v1"


class PlanError(ValueError):
    """The private bundle, topology, or sidecar evidence is incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"{label} cannot be read: {error}") from error


def _bind_marker_lifetime(
    manifest: dict[str, object], runtime_seconds: int
) -> dict[str, object]:
    """Attest, without changing, the topology-bound marker lifetime."""

    result = json.loads(json.dumps(manifest))
    marker_commands = 0
    for phase in result["apply_phases"]:
        if phase["name"] != "source_markers":
            continue
        for command in phase["commands"]:
            argv = command["argv"]
            index = argv.index("--run-seconds")
            if argv[index + 1] != str(runtime_seconds):
                raise PlanError(
                    "topology marker lifetime differs from the overlay contract"
                )
            marker_commands += 1
            command["required_helper_contract"] = {
                "signal_safe_cleanup": True,
                "maximum_runtime_seconds": runtime_seconds,
                "binary_identity_verified": True,
            }
    if marker_commands != 8:
        raise PlanError("source-marker phase must contain eight commands")
    return result


def require_full_topology_gate(
    selected: fabric.FabricPlan, gate: object
) -> dict[str, object]:
    """Require the six-QP-per-rank topology and its exact qualified gate."""

    inventory = fabric.rocenante_inventory(selected)
    expected_per_rank = {str(rank): 6 for rank in range(4)}
    if (
        inventory.get("total_origin_qps") != 24
        or inventory.get("direct_origin_qps") != 16
        or inventory.get("forwarded_origin_qps") != 8
        or inventory.get("origin_qps_per_rank") != expected_per_rank
        or selected.shared_diagonal_flow_label is not True
        or len(selected.markers) != 8
        or {marker.flow_label for marker in selected.markers} != {16383}
        or {marker.udp_source_port for marker in selected.markers} != {65535}
        or len(selected.routes) != 8
        or len(selected.tc_rules) != 8
    ):
        raise PlanError(
            "overlay topology must contain 24 origin QPs, six origin QPs per "
            "rank, eight forwarded paths, eight source markers, eight endpoint "
            "routes, eight hardware restore rules, and shared diagonal marker "
            "flow label 16383 with UDP source port 65535"
        )
    fabric.require_hardware_gate(selected, gate)
    return inventory


def build_plan(
    bundle: Path,
    topology: Path,
    hardware_gate: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, object]:
    """Build one exact mount, rank environment, sidecar, and cleanup document."""

    bundle = bundle.resolve()
    manifest_path = bundle / "sparkring-overlay-manifest.json"
    config_path = bundle / "rocenante-overlay-config.json"
    manifest = _load_json(manifest_path, "private bundle manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") != (
        "sparkring.glm53-rocenante-private-bundle/v1"
    ):
        raise PlanError("private bundle manifest has an unsupported schema")
    contract = load_contract(contract_path or config_path)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise PlanError("private bundle manifest files must be an array")
    config_records = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("path") == "rocenante-overlay-config.json"
    ]
    if len(config_records) != 1 or _sha256(config_path) != config_records[0].get(
        "sha256"
    ):
        raise PlanError("mounted overlay contract differs from the bundle manifest")

    complete = fabric.build_plan(fabric.load_topology(topology))
    selected = fabric.build_rocenante_plan(complete)
    gate = _load_json(hardware_gate, "hardware gate")
    if not isinstance(gate, dict):
        raise PlanError("hardware gate must be an object")
    inventory = require_full_topology_gate(selected, gate)
    base_apply = fabric.build_apply_manifest(
        selected,
        gate,
        authorization_token=fabric.AUTHORIZATION_TOKEN,
    )
    runtime_seconds = int(contract["sidecars"]["source_marker_runtime_seconds"])
    if selected.bounded_runtime_seconds != runtime_seconds:
        raise PlanError(
            "topology marker lifetime differs from the overlay contract"
        )
    sidecar_apply = _bind_marker_lifetime(base_apply, runtime_seconds)
    cleanup = fabric.build_cleanup_manifest(selected)

    rank_environment = {}
    hcas = ",".join(contract["canonical_hca_order"])
    for rank in range(4):
        rank_environment[str(rank)] = {
            "B12X_ROCE_HCA": hcas,
            "B12X_ROCE_GID_INDEX": "3",
            "B12X_ROCE_OPPOSITE_PATHS": "2",
            "B12X_ROCE_PEER_HCA_MAP": contract["peer_hca_maps"][str(rank)],
            "B12X_ROCE_WAVE_MODE": "two",
            "B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES": "196608",
            "ROCENANTE_PROXY_CPU": "13",
        }

    plan = {
        "schema": PLAN_SCHEMA,
        "status": "research-only",
        "execution_authorized": False,
        "purpose": (
            "Describe a reversible GLM-5.3 TP4/DCP4 model test without "
            "executing any host or model operation."
        ),
        "source": {
            "bundle": str(bundle),
            "bundle_manifest_sha256": _sha256(manifest_path),
            "overlay_contract_sha256": _sha256(config_path),
            "topology_sha256": selected.topology_sha256,
            "topology_plan_sha256": selected.sha256,
            "hardware_gate_sha256": hashlib.sha256(
                json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "runtime_mount": {
            "operator_environment_assignment": (f"SIRCL_BUNDLE_HOST_ROOT={bundle}"),
            "container_destination": "/opt/spark-sircl",
            "read_only": True,
            "launcher": "runtime/glm53-flash-jj-r8-gb10/launch-rank.sh",
            "base_runtime_values_unchanged": True,
        },
        "rank_environment": rank_environment,
        "dispatch": {
            "candidate": "contiguous CUDA BF16 TP4 [Q,4096], Q1 through Q32",
            "fallback": "the saved SIRCL/NCCL all-reduce chain",
            "q64_and_larger": "the saved SIRCL/NCCL all-reduce chain",
        },
        "metadata": {
            "group": "TP cpu_group",
            "backend": "gloo",
            "additional_nccl_communicators": 0,
        },
        "cpu_affinity": {
            "rocenante_proxy": 13,
            "sircl_graph_submit": 10,
            "sircl_graph_progress": 11,
            "overlap": False,
        },
        "fabric_inventory": inventory,
        "sidecar_apply": sidecar_apply,
        "cleanup": cleanup,
        "operator_sequence": [
            "verify bundle, topology, hardware gate, and free ports",
            "apply sidecar phases in order",
            "start four ranks with the unchanged GLM launcher and private bundle assignment",
            "run deterministic and concurrency test cells",
            "stop only the test containers",
            "run cleanup phases in order even after test failure",
            "verify marker, rule, qdisc, route, neighbor, and MTU restoration",
        ],
        "limitations": [
            "The plan does not execute a command or establish model correctness.",
            "Every marker helper must prove signal-safe cleanup and an exact configured runtime of 7200 seconds.",
            "Hardware QoS and four-opposite-path mesh32 are rejected research arms and are absent.",
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--hardware-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        plan = build_plan(arguments.bundle, arguments.topology, arguments.hardware_gate)
    except (PlanError, fabric.FabricError, OSError) as error:
        parser.error(str(error))
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
