#!/usr/bin/env python3
"""Compose a pinned MTP3 transport bundle and render non-executing site plans."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from spark_transport.fabric.cx7_hairpin_diagonal import fabric  # noqa: E402
from integrations.vllm.rocenante import build_bundle  # noqa: E402

PINS = json.loads((HERE / "pins.json").read_text())
BASE = HERE.parent / "glm53-flash-jj-r8-gb10"
IMAGE = json.loads((BASE / "pins.json").read_text())
ASSIGNMENT = re.compile(r"([A-Z][A-Z0-9_]*)=(.*)")
SOURCE_LOCK = ROOT / "runtime/sparkring/source_image/glm53-tp4-lock.json"


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
    required = {"schema", "topology_file", "management_addresses", "model_roots", "cache_roots",
                "bundle_root", "container_prefix", "marker_binary", "marker_binary_sha256", "state_root"}
    optional = {"api_keys_file", "liveness_output_seconds", "runtime_profile", "cache_diagnostics", "nccl_debug", "r33_profile_contract_roots"}
    if (not required <= set(data) <= required | optional
            or data["schema"] != "sparkring-glm53-mtp3-mesh-site/v1"):
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
    if "r33_profile_contract_roots" in data:
        roots = data["r33_profile_contract_roots"]
        if (data.get("runtime_profile") not in
                ("tp4-dcp1", "tp4-dcp1-sparkcache", "tp4-dcp4", "tp4-dcp4-sparkcache")
                or not isinstance(roots, list) or len(roots) != 4):
            raise ValueError("r33_profile_contract_roots requires an R33 TP4 selection and four rank-ordered paths")
        for root in roots:
            absolute(root, "r33_profile_contract_roots")
    if "api_keys_file" in data:
        absolute(data["api_keys_file"], "api_keys_file")
    if "liveness_output_seconds" in data:
        timeout = data["liveness_output_seconds"]
        if type(timeout) is not int or not 0 < timeout <= 2147483647:
            raise ValueError("liveness_output_seconds must be an integer from 1 to 2147483647")
    if "nccl_debug" in data and (
            data.get("runtime_profile")
            not in ("tp4-dcp1", "tp4-dcp1-sparkcache", "tp4-dcp4", "tp4-dcp4-sparkcache")
            or data["nccl_debug"] != "INFO"):
        raise ValueError("nccl_debug diagnostic mode requires an R33 TP4 profile and INFO")
    if "cache_diagnostics" in data:
        diagnostic = data["cache_diagnostics"]
        if (data.get("runtime_profile") not in ("tp4-dcp1-sparkcache", "tp4-dcp4-sparkcache")
                or not isinstance(diagnostic, dict)
                or set(diagnostic) != {"namespace", "access_mode", "trace_reuse"}
                or not isinstance(diagnostic.get("namespace"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", diagnostic["namespace"])
                or "REPLACE" in diagnostic["namespace"]
                or diagnostic["access_mode"] != "restore-only"
                or type(diagnostic["trace_reuse"]) is not int or diagnostic["trace_reuse"] != 1):
            raise ValueError("cache_diagnostics requires an R33 tp4-dcp1-sparkcache or tp4-dcp4-sparkcache profile, a concrete safe namespace, restore-only access and trace_reuse=1")
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
    return validate_image_receipt(document)


def _source_receipt_contract():
    """Load the verifier's sibling dependencies without depending on caller imports."""
    directory = ROOT / "runtime/sparkring/source_image"
    names = ("archive_utils", "native_files", "source_image_receipt_contract")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name, filename in zip(names, ("archive_utils.py", "native_files.py", "receipt_contract.py")):
            spec = importlib.util.spec_from_file_location(name, directory / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, saved in previous.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def _r33_profile_verifier():
    path = ROOT / "runtime/sparkring/jovian-r33/profiles/verify_profile.py"
    spec = importlib.util.spec_from_file_location("sparkring_r33_mesh_profile_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_image_receipt(document: dict) -> dict:
    if not isinstance(document, dict):
        raise ValueError("Image receipt must be a JSON object")
    if document.get("schema") == "sparkring-r33-image-receipt/v1":
        _r33_profile_verifier().validate_image_receipt(document)
        expected = document.get("bundle_manifest_sha256")
        verification = document.get("verification")
        checked = verification.get("checked_files") if isinstance(verification, dict) else None
        manifest_path = "/opt/sparkring/sircl/python/sparkring-overlay-manifest.json"
        if (not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or not isinstance(checked, dict)
                or checked.get(manifest_path) != expected):
            raise ValueError(
                "R33 image receipt does not bind its mesh manifest to verified image bytes")
        return dict(document, r33_candidate=True)
    if document.get("schema") == "sparkring-source-image-receipt/v1":
        lock = json.loads(SOURCE_LOCK.read_text())
        selected = lock["profiles"].get(document.get("profile"), {})
        if selected.get("tp_size", 4) != 4 or selected.get("topology") == "switched":
            raise ValueError("Use the selected profile's launcher instead of the TP4 mesh renderer")
        contract = _source_receipt_contract()
        contract.validate_receipt(document, lock)
        inside = document.get("inside_image", {})
        image_id = document.get("image_id", "")
        runtime = lock["runtime"]
        if (lock.get("schema") != "sparkring-source-image-lock/v1"
                or document.get("source_lock_sha256") != sha(SOURCE_LOCK)
                or document.get("profile") not in lock["profiles"]
                or not isinstance(image_id, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
                or document.get("image_reference") != image_id
                or document.get("platform") != "linux/arm64"
                or document.get("checks_passed") is not True
                or not isinstance(inside, dict) or inside.get("checks_passed") is not True
                or inside.get("source_lock_sha256") != sha(SOURCE_LOCK)
                or inside.get("cuda_initialized") is not False
                or inside.get("model_loaded") is not False):
            raise ValueError("Local image receipt does not match the selected source composition")
        for key in ("bundle_manifest_sha256", "marker_binary_sha256", "marker_source_sha256",
                    "nccl_sha256"):
            if inside.get(key) != runtime[key]:
                raise ValueError(f"Local image runtime identity differs: {key}")
        if document.get("bundle_manifest_sha256", inside["bundle_manifest_sha256"]) != inside["bundle_manifest_sha256"]:
            raise ValueError("Local image bundle identity differs")
        if inside.get("nccl_path", runtime["nccl_path"]) != runtime["nccl_path"]:
            raise ValueError("Local image NCCL path contradicts its source lock")
        if inside.get("readiness_warmup") != runtime.get("readiness_warmup") or not inside.get("readiness_warmup"):
            raise ValueError("Local image lacks source-bound sampling warmup")
        return dict(document, bundle_manifest_sha256=inside["bundle_manifest_sha256"])
    if document.get("schema") == "sparkring-mtp3-performance-public-image/v1":
        pinned = json.loads((HERE / "performance/public-image.json").read_text())
        if document != pinned or not document.get("anonymous_manifest_verified"):
            raise ValueError("Performance image receipt differs from the repository pin")
        return dict(document, inside_image=document["verification"])
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
    expected_warmup = {
        "environment": "SPARKRING_WARMUP_TEMPERATURE",
        "helper_sha256": sha(BASE / "warmup_dflash.py"), "temperature": 1.0,
    }
    public = json.loads((HERE / "public-image.json").read_text())
    if image_id == public["config_image_id"]:
        # Immutable images retain their packaged helper when checkout sources change.
        # The public identity only authorizes its complete canonical content receipt.
        canonical = json.loads((HERE / "image-receipt.json").read_text())
        if canonical["image_id"] != public["config_image_id"] or document != canonical:
            raise ValueError("Published image receipt differs from the repository pin")
        expected_warmup = canonical["inside_image"].get("readiness_warmup")
    if warmup is not None and warmup != expected_warmup:
        raise ValueError("Image receipt does not verify the sampling warmup helper")
    compute = inside.get("compute")
    required = PINS.get("compute", {})
    if required:
        if not isinstance(compute, dict):
            raise ValueError("Image receipt lacks the required compute attestation")
        for field in ("source_lock_sha256", "b12x_revision", "b12x_tree", "cuda_version"):
            if compute.get(field) != required[field]:
                raise ValueError(f"Image receipt compute identity differs: {field}")
        lock = json.loads((HERE / required["source_lock"]).read_text())
        environment = compute.get("environment", {})
        if (not isinstance(environment, dict)
                or any(environment.get(k) != v for k, v in lock["environment"].items())
                or compute.get("proposal_head_nvfp4") is not True
                or compute.get("target_head_quantization") is not False):
            raise ValueError("Image receipt does not attest the required proposal and verifier paths")
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


def verify_bundle(bundle: Path, image_record: dict | None = None) -> str:
    """Verify bundle bytes and return the manifest digest from pins or a validated receipt."""
    expected_bundle = image_record["bundle_manifest_sha256"] if image_record else PINS["canonical_bundle_manifest_sha256"]
    if sha(bundle / "sparkring-overlay-manifest.json") != expected_bundle:
        raise ValueError("Bundle manifest does not match the MTP3 mesh profile")
    manifest = json.loads((bundle / "sparkring-overlay-manifest.json").read_text())
    for item in manifest["files"]:
        expected_hashes = {item["sha256"]}
        if image_record and image_record.get("schema") == "sparkring-r33-image-receipt/v1":
            # R33 retains the overlay lineage manifest and separately attests its
            # rebuilt native library and packaged Python files. Rendering and
            # installation accept the lineage bundle or verified image files.
            image_path = "/opt/sparkring/sircl/python/" + item["path"]
            if item["path"] == "libspark_transport_capi.so":
                image_path = "/opt/sparkring/sircl/libspark_transport_capi.so"
            rebuilt_hash = image_record["verification"]["checked_files"].get(image_path)
            if rebuilt_hash is not None:
                expected_hashes.add(rebuilt_hash)
        if sha(manifest_file(bundle, item["path"])) not in expected_hashes:
            raise ValueError("Bundle entry is unsafe or differs from its manifest")
    return expected_bundle


def render(site_path: Path, bundle: Path, output: Path, image_receipt: Path | None = None) -> dict:
    if output.exists():
        raise ValueError("Output directory exists; use an absent directory")
    image_record = load_image_receipt(image_receipt) if image_receipt else None
    expected_bundle = verify_bundle(bundle, image_record)
    site, topology, plan = load_site(site_path)
    source_composition = image_record and image_record.get("schema") == "sparkring-source-image-receipt/v1"
    r33_composition = image_record and image_record.get("schema") == "sparkring-r33-image-receipt/v1"
    if "r33_profile_contract_roots" in site and not r33_composition:
        raise ValueError("r33_profile_contract_roots requires an R33 image receipt")
    if "cache_diagnostics" in site and not r33_composition:
        raise ValueError("cache_diagnostics requires an R33 image receipt")
    if "nccl_debug" in site and not r33_composition:
        raise ValueError("nccl_debug diagnostic mode requires an R33 image receipt")
    if r33_composition:
        runtime_profile = site.get("runtime_profile")
        if runtime_profile not in (
                "tp4-dcp1", "tp4-dcp1-sparkcache", "tp4-dcp4", "tp4-dcp4-sparkcache"):
            raise ValueError(
                "R33 managed site must select tp4-dcp1 or tp4-dcp4, with or without sparkcache")
    elif site.get("runtime_profile") != (image_record["profile"] if source_composition else None):
        raise ValueError("Site runtime profile differs from the explicit image receipt")
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
    if "liveness_output_seconds" in site:
        values["SPARKRING_LIVENESS_OUTPUT_SECONDS"] = str(site["liveness_output_seconds"])
    if "nccl_debug" in site:
        values["NCCL_DEBUG"] = site["nccl_debug"]
    if site.get("api_keys_file"):
        values["API_KEYS_FILE"] = site["api_keys_file"]
    if image_record is not None:
        values["IMAGE_ID"] = image_record["image_id"]
        values["IMAGE_REF"] = image_record["image_reference"]
        if image_record.get("inside_image", {}).get("readiness_warmup") is not None:
            values["SPARKRING_WARMUP_TEMPERATURE"] = "1"
        if image_record.get("schema") == "sparkring-mtp3-performance-public-image/v1":
            values["SPARKCACHE_PLACEMENT_LIBRARY_SHA256"] = image_record["native_placement_sha256"]
            values["SPARKCACHE_CACHE_NAMESPACE"] = image_record["cache_namespace"]
        if source_composition:
            lock = json.loads(SOURCE_LOCK.read_text())
            selected = lock["profiles"][image_record["profile"]]
            values.update(selected["environment"])
            values.update(NCCL_LIBRARY_PATH=lock["runtime"]["nccl_path"],
                          NCCL_LIBRARY_SHA256=lock["runtime"]["nccl_sha256"])
        elif r33_composition:
            verifier = _r33_profile_verifier()
            contract = verifier.load_contract()
            selected = contract["profiles"][runtime_profile]
            profile_values = verifier.parse_template(
                ROOT / "runtime/sparkring/jovian-r33/profiles" / selected["template"]
            )
            if selected.get("inherits"):
                parent = contract["profiles"][selected["inherits"]]
                profile_values = {
                    **verifier.parse_template(ROOT / "runtime/sparkring/jovian-r33/profiles" / parent["template"]),
                    **profile_values,
                }
            for unresolved in ("NODE_RANK", "MASTER_ADDR", "NCCL_IB_HCA"):
                profile_values.pop(unresolved, None)
            values.update(contract["common_environment"])
            values.update(profile_values)
            sparkcache_enabled = "1" if selected["sparkcache"] else "0"
            values.update({
                "SPARKCACHE_ENABLED": sparkcache_enabled,
                "SPARKCACHE_ASYNC_PAGE_CAPTURE": sparkcache_enabled,
            })
            values.update({
                "SOURCE_IMAGE_PROFILE": runtime_profile,
                "SPARKRING_PROFILE_MODE": "custom",
                "SPARKRING_MANAGED_MESH_RENDERED": "1",
                "PYTHONPATH": "/opt/sparkring/sircl/python",
                "SPARK_TP4_LIBRARY": "/opt/sparkring/sircl/libspark_transport_capi.so",
                "SIRCL_BUNDLE_HOST_ROOT": "",
                "SPARKRING_DECLARED_SIRCL_NATIVE_SHA256": image_record["verification"]["checked_files"]["/opt/sparkring/sircl/libspark_transport_capi.so"],
                "SPARKRING_DECLARED_SIRCL_MANIFEST_SHA256": image_record["bundle_manifest_sha256"],
                "VLLM_SPARK_TP4_MODE": "custom",
                "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
                "NCCL_LIBRARY_PATH": "/opt/local-inference/nccl/lib/libnccl.so.2",
                "VLLM_NCCL_SO_PATH": "/opt/local-inference/nccl/lib/libnccl.so.2",
                "NCCL_LOCAL_INFERENCE_PATH": "/opt/local-inference/nccl/lib/libnccl.so.2",
                "LD_PRELOAD": "/opt/local-inference/nccl/lib/libnccl.so.2",
                "NCCL_LIBRARY_SHA256": image_record["verification"]["checked_files"]["/opt/local-inference/nccl/lib/libnccl.so.2.31.2"],
            })
            if selected["sparkcache"]:
                native = contract["sparkcache_native"]
                checked = image_record.get("verification", {}).get("checked_files", {})
                if (checked.get(native["placement_path"]) != native["placement_sha256"]
                        or checked.get(native["snapshot_path"]) != native["snapshot_sha256"]):
                    raise ValueError("R33 image receipt does not bind SparkCache native libraries")
                values.update({
                    "SPARKCACHE_CACHE_NAMESPACE": f"sparkring-r33-{image_record['image_id'][7:19]}-{runtime_profile}",
                    "SPARKCACHE_PLACEMENT_LIBRARY_PATH": native["placement_path"],
                    "SPARKCACHE_PLACEMENT_LIBRARY_SHA256": native["placement_sha256"],
                    "SPARKCACHE_SNAPSHOT_LIBRARY_PATH": native["snapshot_path"],
                    "SPARKCACHE_SNAPSHOT_LIBRARY_SHA256": native["snapshot_sha256"],
                    "SPARKCACHE_VLLM_ROOT": native["vllm_root"],
                    "SPARKCACHE_SOURCE_LEASE_CONTRACT": native["lease_contract"],
                })
    if "cache_diagnostics" in site:
        diagnostic = site["cache_diagnostics"]
        if diagnostic["namespace"] == values["SPARKCACHE_CACHE_NAMESPACE"]:
            raise ValueError("cache_diagnostics requires an isolated namespace distinct from the image default")
        values.update(SPARKCACHE_CACHE_NAMESPACE=diagnostic["namespace"],
                      SPARKCACHE_ACCESS_MODE="restore-only", SPARKCACHE_ASYNC_PAGE_CAPTURE="0",
                      SPARK_CONTEXT_CACHE_TRACE_REUSE="1", SPARKCACHE_CLEAR_ONCE="")
    output.mkdir(parents=True)
    ranks = []
    for rank in range(4):
        env = dict(values)
        if "r33_profile_contract_roots" in site:
            env["R33_PROFILE_CONTRACT_HOST_ROOT"] = site["r33_profile_contract_roots"][rank]
        env.update(HOST_IP=site["management_addresses"][rank], TARGET_MODEL_HOST_PATH=site["model_roots"][rank],
                   CACHE_HOST_ROOT=site["cache_roots"][rank], SOCKET_IFNAME=topology.rank(rank).management_netdev,
                   NODE_RANK=str(rank), SPARKRING_NODE_RANK=str(rank),
                   NCCL_IB_HCA=",".join(topology.rank(rank).port(direction, 0).rdma_device
                                         for direction in ("clockwise", "counter_clockwise")))
        if (source_composition or r33_composition) and selected["host_domains"] == "dual":
            env["NCCL_IB_HCA"] = "=" + ",".join(
                topology.rank(rank).port(direction, function).rdma_device + ":1"
                for function in (0, 1) for direction in ("clockwise", "counter_clockwise")
            )
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
        if r33_composition:
            env["SPARK_TP4_CONTROL_PORT0"] = env["SPARK_TP4_GRAPH_CONTROL_PORT0"]
            env["SPARK_TP4_CONTROL_PORT1"] = env["SPARK_TP4_GRAPH_CONTROL_PORT1"]
        if any("REPLACE" in value for value in env.values()):
            raise ValueError("Rendered runtime still contains unresolved values")
        text = "# MTP3 mesh profile. Review before sourcing.\n"
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
              "bundle_manifest_sha256": expected_bundle,
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
