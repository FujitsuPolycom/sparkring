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
        raise ValueError("Unregistered local source extension")
    result = candidate._read(DESCRIPTOR.read_bytes())
    if result.get("id") != IDENTITY or result.get("schema") != "sparkring-source-extension/v1":
        raise ValueError("Local source extension descriptor differs from its registered identity")
    return result


def image_reference(identity, image_id):
    descriptor(identity)
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Local source extension requires an exact --local-image-id")
    if image_id == descriptor(identity)["parent"]["image_id"]:
        raise ValueError("Local source extension cannot select its unchanged parent image")
    fingerprint = hashlib.sha256(DESCRIPTOR.read_bytes()).hexdigest()[:12]
    return f"sparkring-local:{identity}-{fingerprint}"


def profile_settings(profile, identity, kv_cache_gib=None, master_port=None):
    """Apply reviewed TP4 options after the caller admits the canonical profile."""
    descriptor(identity)
    arguments = profile["vllm_args"]
    if profile.get("topology") != "direct-cycle-4" or arguments[arguments.index("--tensor-parallel-size") + 1] != "4":
        raise ValueError("Qwen source-extension selection is restricted to TP4 profiles")
    if kv_cache_gib is not None and (type(kv_cache_gib) is not int or kv_cache_gib != 40):
        raise ValueError("The local KV alternative is 40 GiB per rank")
    if master_port is not None and (type(master_port) is not int or not 1 <= master_port <= 65535):
        raise ValueError("Local master port must be an integer from 1 to 65535")
    result = copy.deepcopy(profile)
    result["environment"].update(
        VLLM_QWEN3_8_HC_PREFILL_MODE="shard",
        VLLM_QWEN3_8_PREFILL_COALESCE="1",
    )
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
        extra["spark_cache_root"] = "/cache/persistent/qwen38-flash-next-qad-tp4-" + identity
        arguments[index] = json.dumps(transfer, separators=(",", ":"))
    return result


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
