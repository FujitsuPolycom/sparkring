"""Select and admit reviewed source extensions without changing public releases."""

import copy
import hashlib
import json
from pathlib import Path
import re

from runtime.common import candidate, feature_candidate

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = "lil-r37-qwen-prefill"
DESCRIPTOR = ROOT / "runtime/images/compositions" / IDENTITY / "descriptor.json"
ENTRYPOINT = "/opt/sparkring/bin/source-extension.py"
PARENT_RECEIPT = "/opt/sparkring/receipts/source-parent-installed.json"
INSTALLED_DESCRIPTOR = "/opt/sparkring/receipts/source-extension.json"
PATCH = "/opt/sparkring/receipts/source-extension.patch"
SITE = "/opt/venv/lib/python3.12/site-packages/"
LEASE_CONTRACT = "/opt/sparkring/contracts/vllm-connector-jobs-r37-qwen-prefill.json"


def descriptor(identity=IDENTITY):
    if identity != IDENTITY:
        raise ValueError("Unregistered source extension")
    result = candidate._read(DESCRIPTOR.read_bytes())
    if result.get("id") != IDENTITY or result.get("schema") != "sparkring-source-extension/v1":
        raise ValueError("Source extension descriptor differs from its registered identity")
    return result


def publication(identity=IDENTITY, *, image_id=None):
    """Bind a published source image to its registry digest and reviewed descriptor."""
    contract = descriptor(identity)
    path = DESCRIPTOR.with_name("publication.json")
    if not path.is_file():
        raise ValueError("Source extension has no registered publication")
    result = candidate._read(path.read_bytes())
    if (result.get("schema") != "sparkring-image-publication/v1"
            or result.get("platform") != "linux/arm64"
            or result.get("anonymous_pull_verified") is not True
            or not isinstance(result.get("image_reference"), str)
            or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", result["image_reference"])
            or not isinstance(result.get("image_id"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", result["image_id"])
            or result["image_id"] == contract["parent"]["image_id"]
            or result.get("descriptor_sha256") != hashlib.sha256(DESCRIPTOR.read_bytes()).hexdigest()):
        raise ValueError("Source publication must bind an immutable ARM64 image and the reviewed descriptor")
    if image_id is not None and image_id != result["image_id"]:
        raise ValueError("Image differs from the registered source publication")
    return result


def release_publication(release, identity=IDENTITY):
    """Require the release to pin both publication bytes and source descriptor."""
    result = publication(identity)
    prefix = "runtime/images/compositions/" + identity + "/"
    inputs = release.get("inputs", [])
    if not isinstance(inputs, list) or not inputs or inputs[0].get("path") != prefix + "publication.json":
        raise ValueError("Source release must select its registered publication input")
    records = {item.get("path"): item.get("sha256") for item in inputs}
    required = {prefix + "publication.json": DESCRIPTOR.with_name("publication.json"),
                prefix + "descriptor.json": DESCRIPTOR}
    if (release.get("selection") != "registered-source-extension-image"
            or release.get("image") != result["image_reference"]
            or len(records) != len(inputs)
            or any(records.get(name) != hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in required.items())):
        raise ValueError("Source release does not pin the publication, descriptor and registry image together")
    return result


def image_reference(identity, image_id):
    descriptor(identity)
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Local source extension requires an exact --local-image-id")
    if image_id == descriptor(identity)["parent"]["image_id"]:
        raise ValueError("Local source extension cannot select its unchanged parent image")
    fingerprint = hashlib.sha256(DESCRIPTOR.read_bytes()).hexdigest()[:12]
    return f"sparkring-local:{identity}-{fingerprint}"


def profile_nodes(profile):
    """Require a matching pair/ring topology, TP width and node count."""
    arguments = profile["vllm_args"]
    nodes = {"direct-pair-2": 2, "direct-cycle-4": 4}.get(profile.get("topology"))
    if nodes is None or any(arguments[arguments.index(flag) + 1] != str(nodes)
                            for flag in ("--tensor-parallel-size", "--nnodes")):
        raise ValueError("Qwen source extension requires matching TP2 pair or TP4 ring settings")
    return nodes


def profile_settings(profile, identity, kv_cache_gib=None, master_port=None):
    """Select bounded local options; TP2 coalescing does not enable TP4 HC kernels."""
    descriptor(identity)
    nodes = profile_nodes(profile)
    kv_alternative = {2: 33, 4: 40}[nodes]
    if kv_cache_gib is not None and (type(kv_cache_gib) is not int or kv_cache_gib != kv_alternative):
        raise ValueError(f"The local TP{nodes} KV alternative is {kv_alternative} GiB per rank")
    if master_port is not None and (type(master_port) is not int or not 1 <= master_port <= 65535):
        raise ValueError("Local master port must be an integer from 1 to 65535")
    result = copy.deepcopy(profile)
    result["environment"].update(
        VLLM_QWEN3_8_HC_PREFILL_MODE="off" if nodes == 2 else "shard",
        VLLM_QWEN3_8_PREFILL_COALESCE="1",
    )
    if nodes == 2:
        # An image's defaults must not activate a feature absent from the pair profile.
        result["environment"].setdefault("SPARKRING_FEATURES", "")
    arguments = result["vllm_args"]
    if kv_cache_gib is not None:
        arguments[arguments.index("--kv-cache-memory-bytes") + 1] = str(kv_cache_gib * 1024 ** 3)
    if master_port is not None:
        arguments[arguments.index("--master-port") + 1] = str(master_port)
    if "--kv-transfer-config" in arguments:
        index = arguments.index("--kv-transfer-config") + 1
        transfer = json.loads(arguments[index])
        extra = transfer["kv_connector_extra_config"]
        extra["spark_cache_async_page_capture_lease_contract"] = LEASE_CONTRACT
        # Different source layouts must not consume the public profile's entries.
        extra["spark_cache_root"] = f"/cache/persistent/qwen38-flash-next-qad-tp{nodes}-" + identity
        arguments[index] = json.dumps(transfer, separators=(",", ":"))
    validate_profile_contract(result, identity)
    return result


def validate_profile_contract(profile, identity=IDENTITY):
    """Preserve TP eligibility and bind persistent capture to installed sources."""
    contract = descriptor(identity)
    if profile_nodes(profile) == 2:
        environment = profile["environment"]
        features = {part.strip() for part in environment.get("SPARKRING_FEATURES", "").split(",") if part.strip()}
        if (environment.get("VLLM_QWEN3_8_HC_PREFILL_MODE", "off") != "off"
                or not features <= {"qwen-collectives"}):
            raise ValueError("Local TP2 source selection does not support HC sharding or TP4 prefill features")
    arguments = profile["vllm_args"]
    if "--kv-transfer-config" in arguments:
        transfer = json.loads(arguments[arguments.index("--kv-transfer-config") + 1])
        selected = transfer.get("kv_connector_extra_config", {}).get("spark_cache_async_page_capture_lease_contract")
        if selected != LEASE_CONTRACT or selected not in contract["integration_contracts"]:
            raise ValueError("Source-image SparkCache profile must select its packaged lease contract")


def validate(image_id, installed_bytes, parent_bytes, cache_parent_bytes, base_bytes,
             verification, identity=IDENTITY):
    """Check the complete source delta and preserved feature/cache inventory chain.

    The caller reads receipts and verification from the same inspected image.
    Parent verifier records below are structural projections of the child's
    complete checked inventory; they are not observations of running parents.
    """
    contract = descriptor(identity)
    if any(not isinstance(value, bytes) for value in (
        installed_bytes, parent_bytes, cache_parent_bytes, base_bytes,
    )):
        raise ValueError("Raw child and retained parent receipt bytes are required")
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Exact source-extension image ID required")
    fingerprint = hashlib.sha256(DESCRIPTOR.read_bytes()).hexdigest()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    if parent_hash != contract["parent"]["receipt_sha256"]:
        raise ValueError("Source extension parent receipt is not the pinned shared image")
    parent = candidate._read(parent_bytes)
    installed = candidate._read(installed_bytes)
    if "source_extension" in parent or not isinstance(parent.get("files"), dict):
        raise ValueError("Source extension requires an unextended shared parent receipt")
    expected = copy.deepcopy(parent)
    additions = {}
    if not isinstance(contract.get("sources"), dict) or not contract["sources"]:
        raise ValueError("Trusted source replacement inventory is required")
    for name, record in contract["sources"].items():
        target = SITE + name
        if (not isinstance(name, str) or not re.fullmatch(r"(?:b12x|vllm)/[A-Za-z0-9_./-]+\.py", name)
                or not candidate._path(target) or not isinstance(record, dict)
                or set(record) != {"sha256", "parent_sha256"}
                or not candidate._hash(record["sha256"])):
            raise ValueError("Invalid trusted Python source replacement")
        before = record["parent_sha256"]
        if target in parent.get("removed_authored_files", []):
            raise ValueError("Source addition conflicts with an inherited removal")
        if before is None:
            if target in parent["files"]:
                raise ValueError("Added source already belongs to the parent")
        elif not candidate._hash(before) or parent["files"].get(target) != before:
            raise ValueError("Source replacement preimage differs from the retained parent")
        additions[target] = record["sha256"]
    contracts = contract.get("integration_contracts")
    if not isinstance(contracts, dict):
        raise ValueError("Trusted integration contracts are required")
    for target, record in contracts.items():
        if (not isinstance(target, str)
                or not re.fullmatch(r"/opt/sparkring/contracts/[A-Za-z0-9][A-Za-z0-9_.-]*\.json", target)
                or target in parent["files"] or not isinstance(record, dict)
                or not candidate._hash(record.get("sha256"))):
            raise ValueError("Source integration contract must be a distinct pinned addition")
        additions[target] = record["sha256"]
    reserved = {
        ENTRYPOINT: contract.get("installer_sha256"),
        PARENT_RECEIPT: parent_hash,
        INSTALLED_DESCRIPTOR: fingerprint,
        PATCH: contract.get("patch", {}).get("sha256"),
    }
    if any(path in parent["files"] or not candidate._hash(value) for path, value in reserved.items()):
        raise ValueError("Source extension metadata must be distinct and hash-pinned")
    additions.update(reserved)
    expected["files"].update(additions)
    expected["source_extension"] = {
        "id": contract["id"],
        "descriptor_sha256": fingerprint,
        "parent_image_id": contract["parent"]["image_id"],
        "parent_receipt_sha256": parent_hash,
        "provenance": copy.deepcopy(contract["provenance"]),
        "qualification": "Source inventory verified; serving requires profile validation.",
    }
    if installed != expected:
        raise ValueError("Source receipt differs from the complete reviewed parent and source delta")
    receipt_hash = hashlib.sha256(installed_bytes).hexdigest()
    if (not isinstance(verification, dict)
            or type(verification.get("files_verified")) is not int
            or verification.get("serving_qualified") is not False
            or verification != {
                "schema": "sparkring-source-verification/v1",
                "descriptor_sha256": fingerprint,
                "receipt_sha256": receipt_hash,
                "files_verified": len(installed["files"]),
                "serving_qualified": False,
            }):
        raise ValueError("Source verification does not bind the complete unqualified child inventory")
    parent_verification = {
        "schema": "sparkring-candidate-verification/v1",
        "receipt_sha256": parent_hash,
        "files_verified": len(parent["files"]),
        "source_components": parent["components"],
        "serving_qualified": False,
    }
    feature_verification = {
        "schema": "sparkring-feature-verification/v1",
        "descriptor_sha256": parent["feature_extension"]["descriptor_sha256"],
        "capabilities": parent["feature_extension"]["capabilities"],
        "files_verified": len(parent["files"]),
        "serving_qualified": False,
    }
    feature_candidate.validate(contract["parent"]["image_id"], parent_bytes, cache_parent_bytes,
                               base_bytes, parent_verification, feature_verification)
    return {
        "schema": "sparkring-source-admission/v1", "image_id": image_id,
        "receipt_sha256": receipt_hash, "source_extension": expected["source_extension"],
        "feature_extension": copy.deepcopy(parent["feature_extension"]),
        "serving_qualified": False,
    }
