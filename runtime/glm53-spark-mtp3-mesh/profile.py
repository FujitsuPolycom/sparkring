#!/usr/bin/env python3
"""Compose a pinned MTP3 transport bundle and render non-executing site plans."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from spark_transport.experiments.cx7_hairpin_diagonal import fabric  # noqa: E402
from spark_transport.experiments.glm53_rocenante_overlay import build_bundle  # noqa: E402

PINS = json.loads((HERE / "pins.json").read_text())
BASE = HERE.parent / "glm53-flash-jj-r8-gb10"
IMAGE = json.loads((BASE / "pins.json").read_text())
ASSIGNMENT = re.compile(r"([A-Z][A-Z0-9_]*)=(.*)")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose(base_sircl: Path, output: Path) -> dict:
    if sha(base_sircl / "sparkring-overlay-manifest.json") != IMAGE["sircl"]["overlay_manifest_sha256"]:
        raise ValueError("Base SIRCL manifest does not match the pinned operator image")
    if sha(base_sircl / "libspark_transport_capi.so") != IMAGE["sircl"]["native_sha256"]:
        raise ValueError("Base SIRCL native library does not match the pinned image")
    result = build_bundle.build(base_sircl, ROOT / "third_party/b12x_roce", output,
                                captured_sircl_rows=tuple(PINS["captured_sircl_query_rows"]))
    actual = sha(output / "sparkring-overlay-manifest.json")
    if actual != PINS["canonical_bundle_manifest_sha256"]:
        raise ValueError(f"Composed bundle differs from its pin: {actual}")
    return result


def defaults(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        match = ASSIGNMENT.fullmatch(line)
        if match:
            words = shlex.split(match[2], comments=True)
            if len(words) > 1:
                raise ValueError(f"Nonliteral default for {match[1]}")
            result[match[1]] = words[0] if words else ""
    return result


def absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value == "/":
        raise ValueError(f"{label} must be an absolute non-root Linux path")
    if any(char in value for char in ("\n", "\r", ":", "\x00")) or "REPLACE" in value:
        raise ValueError(f"{label} contains an unsafe or unresolved value")
    if ".." in Path(value).parts:
        raise ValueError(f"{label} must not contain parent traversal")
    return value


def load_site(path: Path):
    data = json.loads(path.read_text())
    expected = {"schema", "topology_file", "management_addresses", "model_roots", "cache_roots",
                "bundle_root", "container_prefix", "marker_binary", "marker_binary_sha256", "state_root"}
    if set(data) != expected or data["schema"] != "sparkring-glm53-mtp3-mesh-site/v1":
        raise ValueError("Site fields do not match sparkring-glm53-mtp3-mesh-site/v1")
    for name in ("management_addresses", "model_roots", "cache_roots"):
        if not isinstance(data[name], list) or len(data[name]) != 4:
            raise ValueError(f"{name} must contain four rank-ordered values")
    addresses = [str(ipaddress.IPv4Address(value)) for value in data["management_addresses"]]
    if len(set(addresses)) != 4:
        raise ValueError("Management addresses must be distinct")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", data["container_prefix"]):
        raise ValueError("Invalid container prefix")
    if not re.fullmatch(r"[0-9a-f]{64}", data["marker_binary_sha256"]):
        raise ValueError("Marker binary must have an exact lowercase SHA-256")
    for key in ("model_roots", "cache_roots"):
        for value in data[key]:
            absolute(value, key)
    for key in ("bundle_root", "marker_binary", "state_root"):
        absolute(data[key], key)
    topology_path = path.parent / data["topology_file"]
    topology = fabric.load_topology(topology_path)
    for node in topology.ranks:
        for direction, physical in (("clockwise", 0), ("counter_clockwise", 1)):
            for function in (0, 1):
                expected_hca = f"roce{'P2' if function else ''}p1s0f{physical}"
                if node.port(direction, function).rdma_device != expected_hca:
                    raise ValueError("Topology does not match the pinned RoCEnante HCA order and peer map")
    plan = fabric.build_rocenante_plan(fabric.build_plan(topology))
    inventory = fabric.rocenante_inventory(plan)
    if (inventory["total_origin_qps"] != 24 or len(plan.tc_rules) != 8
            or len(plan.markers) != 8 or not plan.shared_diagonal_flow_label
            or {m.udp_source_port for m in plan.markers} != {65535}
            or topology.bounded_runtime_seconds != 7200):
        raise ValueError("The profile requires the two-path, six-QP-per-rank bounded fabric")
    return data, topology, plan


def load_image_receipt(path: Path) -> dict:
    document = json.loads(path.read_text())
    expected = PINS["canonical_bundle_manifest_sha256"]
    image_id = document.get("image_id", "")
    inside = document.get("inside_image", {})
    source_sha = document.get("source_receipt_sha256", "")
    if (document.get("schema") != "sparkring-mtp3-mesh-image-receipt/v1"
            or document.get("checks_passed") is not True
            or document.get("platform") != "linux/arm64"
            or document.get("parent_image_id") != IMAGE["operator_image"]["image_id"]
            or not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or document.get("image_reference") != image_id
            or document.get("bundle_manifest_sha256") != expected
            or not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
            or not isinstance(inside, dict) or inside.get("checks_passed") is not True
            or inside.get("bundle_manifest_sha256") != expected
            or inside.get("source_receipt_sha256") != source_sha
            or inside.get("cuda_initialized") is not False or inside.get("model_loaded") is not False):
        raise ValueError("Image receipt does not verify the pinned parent and mesh bundle")
    warmup = inside.get("readiness_warmup")
    if warmup is not None and warmup != {
        "environment": "SPARKRING_WARMUP_TEMPERATURE",
        "helper_sha256": sha(BASE / "warmup_dflash.py"), "temperature": 1.0,
    }:
        raise ValueError("Image receipt does not verify the sampling warmup helper")
    return document


def manifest_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Bundle path must be a nonempty relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError("Bundle path escapes or aliases its root")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Bundle symlink escapes its root")
    return target


def render(site_path: Path, bundle: Path, output: Path, image_receipt: Path | None = None) -> dict:
    if output.exists():
        raise ValueError("Output directory exists; use an absent directory")
    if sha(bundle / "sparkring-overlay-manifest.json") != PINS["canonical_bundle_manifest_sha256"]:
        raise ValueError("Bundle manifest does not match the MTP3 mesh profile")
    manifest = json.loads((bundle / "sparkring-overlay-manifest.json").read_text())
    for item in manifest["files"]:
        if sha(manifest_file(bundle, item["path"])) != item["sha256"]:
            raise ValueError("Bundle entry is unsafe or differs from its manifest")
    site, topology, plan = load_site(site_path)
    values = defaults(BASE / "runtime.env.example")
    values.update(defaults(BASE / "sircl-fused.env.example"))
    values.pop("DFLASH_MODEL_HOST_PATH", None)
    values.update({
        "TARGET_MODEL_VARIANT": "nvfp4-spark", "SPECULATION_METHOD": "mtp",
        "NUM_SPECULATIVE_TOKENS": "3", "MAX_CUDAGRAPH_CAPTURE_SIZE": "64",
        "SERVED_MODEL_NAME": "glm-5.3-flash-spark", "CONTAINER_PREFIX": site["container_prefix"],
        "SIRCL_ENABLED": "1", "SIRCL_BUNDLE_HOST_ROOT": site["bundle_root"],
        "SPARKCACHE_CACHE_NAMESPACE": PINS["cache_identity"]["namespace"],
        "SPARKCACHE_ASYNC_PAGE_CAPTURE": "1", "SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES": "3221225472",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT": "2", "KV_CACHE_MEMORY_BYTES": "25769803776",
        "MASTER_ADDR": site["management_addresses"][0], "DFLASH_WARMUP": "1",
        "SPARKRING_WARMUP_TEMPERATURE": "0",
    })
    image_record = load_image_receipt(image_receipt) if image_receipt else None
    if image_record is not None:
        values["IMAGE_ID"] = image_record["image_id"]
        values["IMAGE_REF"] = image_record["image_reference"]
        if image_record["inside_image"].get("readiness_warmup") is not None:
            values["SPARKRING_WARMUP_TEMPERATURE"] = "1"
    output.mkdir(parents=True)
    ranks = []
    for rank in range(4):
        env = dict(values)
        env.update(HOST_IP=site["management_addresses"][rank], TARGET_MODEL_HOST_PATH=site["model_roots"][rank],
                   CACHE_HOST_ROOT=site["cache_roots"][rank], SOCKET_IFNAME=topology.rank(rank).management_netdev,
                   NCCL_IB_HCA=",".join(topology.rank(rank).port(direction, 0).rdma_device
                                         for direction in ("clockwise", "counter_clockwise")))
        # The native SIRCL endpoint order is rank XOR 1, then rank XOR 3.
        # Odd ranks therefore reverse physical direction order; RoCEnante's
        # HCA inventory remains clockwise f0, counter-clockwise f1.
        directions = ("clockwise", "counter_clockwise") if rank % 2 == 0 else ("counter_clockwise", "clockwise")
        for slot, direction in enumerate(directions):
            for function in (0, 1):
                local = topology.rank(rank).port(direction, function)
                if local.peer_rank != rank ^ (1 if slot == 0 else 3):
                    raise ValueError("SIRCL endpoint does not match the native XOR peer ordering")
                peer = topology.rank(local.peer_rank).port(local.peer_direction, local.peer_function)
                prefix = "SPARK_TP4_" if function == 0 else "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_"
                env[prefix + f"PEER{slot}"] = peer.ipv4
                env[prefix + f"DEVICE{slot}"] = local.rdma_device
                env[prefix + f"GID{slot}"] = "3"
        if any("REPLACE" in value for value in env.values()):
            raise ValueError("Rendered runtime still contains unresolved values")
        text = "# Native MTP3 mesh profile. Review before sourcing.\n"
        text += "\n".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + "\n"
        (output / f"rank{rank}.env").write_text(text, newline="\n")
        rank_plan = {"rank": rank, "ssh_alias": topology.rank(rank).ssh_alias,
                     "management_netdev": topology.rank(rank).management_netdev,
                     "ports": [p.__dict__ for p in topology.rank(rank).ports],
                     "routes": [fabric.route_command(r, add=True) for r in plan.routes if r.source_rank == rank],
                     "neighbors": [fabric.neighbor_command(r, add=True) for r in plan.routes
                                   if r.source_rank == rank and r.permanent_final_neighbor],
                     "tc_rules": [fabric.tc_rule_command(r, add=True) for r in plan.tc_rules if r.intermediate_rank == rank],
                     "markers": [{"device": m.rdma_device, "argv": [site["marker_binary"], "--device", m.rdma_device,
                                 "--source-port", "65535", "--replacement-ethertype", "0x88b5", "--attach", "--run-seconds", "7200"]}
                                 for m in plan.markers if m.source_rank == rank]}
        ranks.append(rank_plan)
    shutil.copyfile(BASE / "launch-rank.sh", output / "launch-rank.sh")
    rendered_site = dict(site, topology_file="fabric.json")
    (output / "site.json").write_text(json.dumps(rendered_site, indent=2) + "\n", newline="\n")
    shutil.copyfile(topology.source_path, output / "fabric.json")
    result = {"schema": "sparkring-mtp3-mesh-render/v1", "status": "research-only", "execution_authorized": False,
              "site_sha256": sha(site_path), "topology_sha256": topology.sha256,
              "bundle_manifest_sha256": PINS["canonical_bundle_manifest_sha256"],
              "image": IMAGE["operator_image"], "marker_binary": site["marker_binary"],
              "marker_binary_sha256": site["marker_binary_sha256"], "state_root": site["state_root"],
              "marker_scope": "All RDMA-TX packets with reserved UDP source port 65535 on each selected function; not an IP/QPN-scoped rule.",
              "ranks": ranks, "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    if image_record is not None:
        result["image"] = image_record
        result["image_receipt_sha256"] = sha(image_receipt)
    (output / "fabric-plan.json").write_text(json.dumps(result, indent=2) + "\n", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--base-sircl", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("render")
    run.add_argument("--site", type=Path, required=True)
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--image-receipt", type=Path,
                     help="Optional verified private child image; omission retains the pinned public parent")
    args = parser.parse_args()
    result = compose(args.base_sircl, args.output) if args.action == "bundle" else render(args.site, args.bundle, args.output, args.image_receipt)
    print(json.dumps({"status": "research-only", "output": str(args.output), "files": len(result.get("files", []))}))


if __name__ == "__main__":
    main()
