#!/usr/bin/env python3
"""Validate an R33 profile template, immutable image receipt, or live activation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "profile-contract.json"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
REGISTRY_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


def load_contract() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text())
    lock = (HERE / contract["image"]["artifact_lock"]).resolve()
    if hashlib.sha256(lock.read_bytes()).hexdigest() != contract["image"]["artifact_lock_sha256"]:
        raise ValueError("The R33 artifact lock differs from the profile contract")
    return contract


def profile(name: str) -> tuple[dict, dict]:
    contract = load_contract()
    try:
        selected = contract["profiles"][name]
    except KeyError as error:
        raise ValueError(f"Unknown R33 profile: {name}") from error
    return contract, selected


def parse_template(path: Path) -> dict[str, str]:
    result = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in result:
            raise ValueError(f"Invalid environment assignment at {path}:{number}")
        result[key] = value
    return result


def validate_template(name: str) -> dict:
    contract, selected = profile(name)
    values = parse_template(HERE / selected["template"])
    if selected.get("inherits"):
        _, parent = profile(selected["inherits"])
        values = {**parse_template(HERE / parent["template"]), **values}
    expected = {
        "SOURCE_IMAGE_PROFILE": name,
        "TENSOR_PARALLEL_SIZE": str(selected["tensor_parallel_size"]),
        "DECODE_CONTEXT_PARALLEL_SIZE": str(selected["decode_context_parallel_size"]),
        "NODE_COUNT": str(selected["node_count"]),
        "MAX_MODEL_LEN": str(contract["model"]["max_model_len"]),
        "KV_CACHE_MEMORY_BYTES": str(selected["kv_cache_memory_bytes"]),
        "NUM_SPECULATIVE_TOKENS": str(contract["model"]["speculation"]["num_speculative_tokens"]),
        "CUDAGRAPH_CAPTURE_SIZES": ",".join(map(str, selected["cudagraph_capture_sizes"])),
        "VLLM_B12X_KDA_PREFILL_COALESCING": "1",
        "VLLM_GLM53_MHC_PREFILL_SHARD": "1",
        "SPARKCACHE_ENABLED": "1" if selected["sparkcache"] else "0",
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"{name} requires {key}={value}")
    if name == "tp2-dcp1":
        manifest = (HERE / selected["transport_manifest"]).resolve()
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != selected["transport_manifest_sha256"]:
            raise ValueError("The TP2 RoCEnante manifest differs from the profile contract")
        if values["NCCL_IB_HCA"] != "=rocep1s0f0,roceP2p1s0f0" or values["B12X_ROCE_PAIR_PATHS"] != "2":
            raise ValueError("TP2 must use both PCI domains of one physical DAC")
        if values.get("PYTHONPATH"):
            raise ValueError("TP2 must clear the inherited TP4 mesh overlay")
    else:
        pins = (HERE / selected["mesh_pins"]).resolve()
        if hashlib.sha256(pins.read_bytes()).hexdigest() != selected["mesh_pins_sha256"]:
            raise ValueError("The managed TP4 mesh pins differ from the profile contract")
        required = {"SIRCL_ENABLED": "1", "VLLM_SPARK_TP4_MODE": "custom",
                    "VLLM_SPARK_TP4_VOCAB_MODE": "custom", "NCCL_SWITCHLESS_RING_ONLY": "1"}
        if any(values.get(key) != value for key, value in required.items()):
            raise ValueError("TP4 must select the custom mesh and switchless ring")
    return {"profile": name, "template": selected["template"], "checks_passed": True}


def validate_image_receipt(document: dict) -> dict:
    contract = load_contract()
    required = contract["image"]
    if (document.get("schema") != "sparkring-r33-image-receipt/v1"
            or document.get("checks_passed") is not True
            or document.get("platform") != required["required_platform"]
            or not IMAGE_ID.fullmatch(document.get("image_id", ""))
            or not (REGISTRY_DIGEST.fullmatch(document.get("image_reference", ""))
                    or document.get("image_reference") == document.get("image_id"))
            or document.get("artifact_lock_sha256") != required["artifact_lock_sha256"]
            or document.get("sources") != required["required_sources"]
            or document.get("nccl_version") != required["required_nccl_version"]):
        raise ValueError("Image receipt does not identify the exact qualified R33 ARM64 composition")
    if document.get("source_locks_match") is not True or document.get("package_checks_passed") is not True:
        raise ValueError("Image receipt lacks passing source-lock and package checks")
    return {"image_id": document["image_id"], "image_reference": document["image_reference"]}


def _positive(value: object, label: str) -> None:
    if type(value) not in (int, float) or value <= 0:
        raise ValueError(f"Activation receipt requires positive {label}")


def validate_activation(document: dict) -> dict:
    name = document.get("profile")
    contract, selected = profile(name)
    validate_image_receipt(document.get("image", {}))
    expected_hash = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    if (document.get("schema") != "sparkring-r33-activation-receipt/v1"
            or document.get("checks_passed") is not True
            or document.get("profile_contract_sha256") != expected_hash):
        raise ValueError("Activation receipt is not bound to this exact profile contract")
    ranks = document.get("ranks")
    if not isinstance(ranks, list) or {item.get("rank") for item in ranks} != set(range(selected["node_count"])):
        raise ValueError("Activation receipt must contain every rank exactly once")
    expected_graphs = selected["cudagraph_capture_sizes"]
    image_id = document["image"]["image_id"]
    for rank in ranks:
        if rank.get("image_id") != image_id or rank.get("nccl_version") != "2.31.2":
            raise ValueError("Every rank must report the selected image and NCCL 2.31.2")
        if set(rank.get("nccl_host_domains", [])) != {"primary", "secondary"}:
            raise ValueError("Every rank must activate NCCL HCAs from both host PCIe domains")
        if rank.get("captured_graph_sizes") != expected_graphs:
            raise ValueError("Every rank must capture the profile's exact CUDA graph sizes")
        for key in ("instanttensor_allocations", "mtp_draft_tokens", "continuation_coalesced_groups",
                    "mhc_sharded_prefill_calls"):
            _positive(rank.get(key), key)
        if 2048 not in rank.get("mhc_owner_rows", []):
            raise ValueError("Every rank must report a 2,048-row mHC owner execution")
        counter = "rocenante_collectives" if name == "tp2-dcp1" else "sircl_collectives"
        _positive(rank.get(counter), counter)
    serving = document.get("serving", {})
    if (serving.get("max_model_len") != contract["model"]["max_model_len"]
            or serving.get("prefill_decode_passed") is not True
            or serving.get("correctness_passed") is not True):
        raise ValueError("Serving receipt lacks 1M admission, prefill/decode, or correctness evidence")
    _positive(serving.get("kv_capacity_tokens"), "kv_capacity_tokens")
    if name.startswith("tp4"):
        regression = document.get("long_prefill_sample_tokens", {})
        if (regression.get("bounded") is not True or regression.get("prompt_tokens", 0) < 32768
                or regression.get("completed_requests", 0) < 2 or regression.get("timeouts") != 0
                or regression.get("fatal_engine_errors") != 0):
            raise ValueError("TP4 requires a bounded repeated long-prefill sample_tokens regression")
    cache = document.get("sparkcache", {})
    if selected["sparkcache"]:
        for key in ("capture_jobs_completed", "restores_completed", "recoveries_after_fault"):
            _positive(cache.get(key), key)
        if cache.get("payload_correctness_passed") is not True:
            raise ValueError("SparkCache restore payload correctness was not proven")
    elif cache.get("enabled") is not False:
        raise ValueError("The cache-disabled profile must prove that SparkCache stayed disabled")
    return {"profile": name, "ranks": len(ranks), "checks_passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("template", "image", "activation"))
    parser.add_argument("--profile", choices=("tp2-dcp1", "tp4-dcp1", "tp4-dcp1-sparkcache"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.kind == "template":
        if not args.profile:
            parser.error("template validation requires --profile")
        result = validate_template(args.profile)
    else:
        if args.receipt is None:
            parser.error(f"{args.kind} validation requires --receipt")
        document = json.loads(args.receipt.read_text())
        result = validate_image_receipt(document) if args.kind == "image" else validate_activation(document)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
