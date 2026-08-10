#!/usr/bin/env python3
"""Generate the exact, manifest-linked SparkRing overlay ownership inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/configurations/vllm-overlay-ownership-v1.json"
REF_ROOT = ROOT / "runtime/patches/00-reference-vllm"
SPARKCACHE_ROOT = ROOT / "runtime/patches/vllm"


class InventoryError(RuntimeError):
    """Raised when inventory sources or the committed artifact drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


# This is an ownership-planning classification, not an authorship claim. Paths
# omitted here remain unknown instead of being inferred from names or markers.
DESTINATIONS = {
    # GB10 kernels, low-bit formats, and sparse-MLA implementation.
    "model_executor/layers/fused_moe/runner/shared_experts.py": "SparkInfer",
    "model_executor/layers/quantization/__init__.py": "SparkInfer",
    "model_executor/warmup/deepseek_v4_mhc_warmup.py": "SparkInfer",
    "model_executor/warmup/flashinfer_sparse_mla_warmup.py": "SparkInfer",
    "model_executor/warmup/kernel_warmup.py": "SparkInfer",
    "models/deepseek_v4/nvidia/flashinfer_sparse.py": "SparkInfer",
    "utils/torch_utils.py": "SparkInfer",
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": "SparkInfer",
    "model_executor/layers/fp8_lm_head.py": "SparkInfer",
    "model_executor/layers/quantization/hybrid_mxfp4_ct.py": "SparkInfer",
    "models/deepseek_v4/nvidia/ops/dspark_sparse_attn_tilelang.py": "SparkInfer",
    # Generic scheduler, DCP, MTP, graph, cache, and model semantics.
    "config/compilation.py": "vLLM",
    "config/speculative.py": "vLLM",
    "config/vllm.py": "vLLM",
    "engine/arg_utils.py": "vLLM",
    "model_executor/model_loader/base_loader.py": "vLLM",
    "model_executor/model_loader/sharded_state_loader.py": "vLLM",
    "model_executor/model_loader/weight_utils.py": "vLLM",
    "model_executor/models/qwen3_dflash.py": "vLLM",
    "model_executor/models/registry.py": "vLLM",
    "models/deepseek_v4/__init__.py": "vLLM",
    "models/deepseek_v4/common/rope.py": "vLLM",
    "models/deepseek_v4/sparse_mla.py": "vLLM",
    "v1/attention/backends/mla/flashmla_sparse.py": "vLLM",
    "v1/attention/backends/mla/indexer.py": "vLLM",
    "v1/attention/backends/utils.py": "vLLM",
    "v1/attention/ops/common.py": "vLLM",
    "v1/core/kv_cache_coordinator.py": "vLLM",
    "v1/core/kv_cache_manager.py": "vLLM",
    "v1/core/sched/scheduler.py": "vLLM",
    "v1/core/single_type_kv_cache_manager.py": "vLLM",
    "v1/cudagraph_dispatcher.py": "vLLM",
    "v1/sample/ops/topk_topp_sampler.py": "vLLM",
    "v1/spec_decode/dflash.py": "vLLM",
    "v1/spec_decode/utils.py": "vLLM",
    "v1/worker/gpu/block_table.py": "vLLM",
    "v1/worker/gpu/cudagraph_utils.py": "vLLM",
    "v1/worker/gpu/model_runner.py": "vLLM",
    "v1/worker/gpu/pp_utils.py": "vLLM",
    "v1/worker/gpu/sample/gumbel.py": "vLLM",
    "v1/worker/gpu/spec_decode/__init__.py": "vLLM",
    "v1/worker/gpu/spec_decode/autoregressive/speculator.py": "vLLM",
    "v1/worker/gpu/spec_decode/dflash/speculator.py": "vLLM",
    "v1/worker/gpu/spec_decode/eagle/utils.py": "vLLM",
    "v1/worker/gpu/spec_decode/utils.py": "vLLM",
    "v1/worker/gpu/warmup.py": "vLLM",
    "v1/spec_decode/dynamic/acceptance_length.py": "vLLM",
    "v1/worker/gpu/spec_decode/dspark/__init__.py": "vLLM",
    "v1/worker/gpu/spec_decode/dspark/utils.py": "vLLM",
}


MIXED_SUBCLASSIFICATIONS = {
    "distributed/parallel_state.py": [
        {"destination": "vLLM", "scope": "CUDA-graph capture lifecycle and group interface"},
        {"destination": "SIRCL adapter/plugin", "scope": "Spark custom-transport shared capture stream"},
    ],
    "config/cache.py": [
        {"destination": "vLLM", "scope": "KV-cache dtype configuration interface"},
        {"destination": "SparkInfer", "scope": "nvfp4_ds_mla format implementation"},
    ],
    "envs.py": [
        {"destination": "vLLM", "scope": "environment registry integration"},
        {"destination": "SparkInfer", "scope": "B12X DCP controls"},
    ],
    "model_executor/layers/attention/mla_attention.py": [
        {"destination": "vLLM", "scope": "generic MLA and DCP dispatch semantics"},
        {"destination": "SparkInfer", "scope": "B12X and low-bit MLA implementation"},
    ],
    "model_executor/layers/mla.py": [
        {"destination": "vLLM", "scope": "generic MLA cache lifecycle and calibration hooks"},
        {"destination": "SparkInfer", "scope": "low-bit MLA record scaling and B12X consumers"},
    ],
    "model_executor/layers/sparse_attn_indexer.py": [
        {"destination": "vLLM", "scope": "DCP index ownership and scheduling semantics"},
        {"destination": "SparkInfer", "scope": "B12X sparse-indexer kernels and buffers"},
    ],
    "model_executor/layers/vocab_parallel_embedding.py": [
        {"destination": "vLLM", "scope": "vocabulary-parallel layer lifecycle"},
        {"destination": "SparkInfer", "scope": "FP8 LM-head quantization implementation"},
    ],
    "model_executor/models/deepseek_v2.py": [
        {"destination": "vLLM", "scope": "model-level slot-coordinate semantics"},
        {"destination": "SparkInfer", "scope": "B12X fused indexer integration"},
    ],
    "models/deepseek_v4/attention.py": [
        {"destination": "vLLM", "scope": "sequence-parallel model semantics"},
        {"destination": "SparkInfer", "scope": "fused indexer kernel integration"},
    ],
    "models/deepseek_v4/nvidia/model.py": [
        {"destination": "vLLM", "scope": "DSpark model and auxiliary-state semantics"},
        {"destination": "SparkInfer", "scope": "NVIDIA/B12X model implementation"},
    ],
    "v1/attention/backend.py": [
        {"destination": "vLLM", "scope": "KV-cache writer interface and dispatch"},
        {"destination": "SparkInfer", "scope": "nvfp4 MLA writer implementation"},
    ],
    "v1/attention/backends/mla/b12x_mla_sparse.py": [
        {"destination": "vLLM", "scope": "attention-backend interface and DCP orchestration"},
        {"destination": "SparkInfer", "scope": "B12X sparse-MLA kernels and workspaces"},
    ],
    "v1/attention/backends/mla/sparse_swa.py": [
        {"destination": "vLLM", "scope": "DSpark/SWA model and speculative thresholds"},
        {"destination": "SparkInfer", "scope": "SM120 sparse-attention kernel integration"},
    ],
    "v1/attention/ops/dcp_alltoall.py": [
        {"destination": "vLLM", "scope": "generic DCP collective and reduction semantics"},
        {"destination": "SparkInfer", "scope": "B12X PCIe DCP implementation"},
    ],
    "v1/kv_cache_interface.py": [
        {"destination": "vLLM", "scope": "replicated-DCP KV specification semantics"},
        {"destination": "SparkInfer", "scope": "nvfp4_ds_mla record geometry"},
    ],
    "v1/worker/gpu_worker.py": [
        {"destination": "vLLM", "scope": "pipeline-local warmup sequencing"},
        {"destination": "SparkInfer", "scope": "B12X allocator-cache peak shaving"},
    ],
    "models/deepseek_v4/nvidia/dspark.py": [
        {"destination": "vLLM", "scope": "DSpark model and MTP semantics"},
        {"destination": "SparkInfer", "scope": "GB10/B12X/DeepGEMM implementation"},
    ],
    "model_executor/models/qwen3_dspark.py": [
        {"destination": "vLLM", "scope": "Qwen/DSpark model semantics and registration"},
        {"destination": "SparkInfer", "scope": "Spark-specialized execution implementation"},
    ],
    "models/deepseek_v4/nvidia/dspark_v2.py": [
        {"destination": "vLLM", "scope": "DSpark v2 model semantics"},
        {"destination": "SparkInfer", "scope": "FlashInfer/DeepGEMM implementation"},
    ],
    "v1/worker/gpu/spec_decode/dspark/utils_v2.py": [
        {"destination": "vLLM", "scope": "DSpark speculative-decoding semantics"},
        {"destination": "SparkInfer", "scope": "B12X/FlashInfer/DeepGEMM integration"},
    ],
    "v1/worker/gpu/spec_decode/dspark/speculator.py": [
        {"destination": "vLLM", "scope": "speculator lifecycle and rolling-KV semantics"},
        {"destination": "SparkInfer", "scope": "Spark/FlashInfer execution implementation"},
    ],
    "v1/worker/gpu/spec_decode/dspark/speculator_v2.py": [
        {"destination": "vLLM", "scope": "paged DSpark speculative-decoding semantics"},
        {"destination": "SparkInfer", "scope": "Spark-specialized execution implementation"},
    ],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination_for(path: str) -> str:
    if path in MIXED_SUBCLASSIFICATIONS:
        return "mixed/requires-decomposition"
    return DESTINATIONS.get(path, "unknown")


def missing_seam(destination: str) -> dict:
    if destination == "unknown":
        return {
            "status": "unknown",
            "reason": (
                "The offline source audit has not established semantic ownership, so it "
                "cannot yet determine whether an extension seam is required."
            ),
        }
    if destination == "mixed/requires-decomposition":
        return {
            "status": "unknown",
            "reason": (
                "The operation must first be decomposed by destination; the required "
                "interface and seam can differ for each extracted behavior."
            ),
        }
    return {
        "status": "none",
        "justification": (
            "No missing extension seam was established by this offline audit. Direct "
            "upstreaming or relocation remains a candidate and must be tested before "
            "introducing a new seam."
        ),
    }


def destination_basis(destination: str) -> str:
    if destination == "mixed/requires-decomposition":
        return (
            "Patch/source review found behavior belonging to two or more listed "
            "destinations; retirement requires hunk-level decomposition."
        )
    if destination == "vLLM":
        return (
            "Patch/source review found generic model, scheduler, DCP, MTP, cache, graph, "
            "or loader semantics in the vLLM interface."
        )
    if destination == "SparkInfer":
        return (
            "Patch/source review found GB10, quantization, sparse-MLA, warmup, or backend "
            "implementation without a separately identified generic vLLM behavior."
        )
    return "The current offline source evidence does not establish a single destination."


def retirement_proof(destination: str) -> dict:
    replacement = {
        "vLLM": "accepted upstream vLLM commit or released public vLLM seam",
        "SparkInfer": "pinned SparkInfer commit containing equivalent implementation",
        "mixed/requires-decomposition": (
            "separately pinned replacements for every destination subclassification"
        ),
        "unknown": "reviewed ownership decision followed by a pinned replacement",
    }.get(destination, "pinned replacement in the selected owner module")
    return {
        "replacement_identity": replacement,
        "semantic_equivalence": (
            "Map every changed behavior in the operation to replacement code and review "
            "the map with no unexplained residual hunk."
        ),
        "numerical_correctness": (
            "Pass the applicable fixed-seed token comparison and, where numerics are "
            "touched, token-level logit/KLD or FP32-reference audit on the same stack."
        ),
        "distributed_and_graph_behavior": (
            "Pass applicable four-rank DCP, CUDA-graph capture/replay, native-execution, "
            "fallback-accounting, startup, and restart gates."
        ),
        "performance_and_reliability": (
            "Show no unacceptable C1/C8 regression and no new topology-dependent hang, "
            "fallback, health, or rollback failure."
        ),
        "artifact_attestation": (
            "Build the identical pinned image on all four ranks, attest the replacement "
            "identity, and prove this operation is absent from the patch-apply receipt."
        ),
        "rollback": (
            "Retain and exercise the last known-good locked image until replacement gates pass."
        ),
        "owner_acceptance": (
            "Record upstream merge/release evidence or an explicit SparkRing owner review "
            "for plugin/distribution ownership before marking retired."
        ),
    }


def consumption(provenance: str) -> dict:
    reference_status = (
        "recovered-from-runtime"
        if provenance == "recovered-reference"
        else "equivalent-semantics-reported"
    )
    reference_evidence = (
        "runtime/patches/00-reference-vllm/provenance.json"
        if provenance == "recovered-reference"
        else "runtime/patches/vllm/README.md"
    )
    reference_notes = (
        "This exact operation was recovered from the historical reference runtime artifact."
        if provenance == "recovered-reference"
        else (
            "The README reports equivalent semantics exercised against different "
            "surrounding file hashes, not application of this exact public patch."
        )
    )
    return {
        "generic_overlay_builders": [
            {
                "path": "runtime/Containerfile",
                "status": "declared-input",
                "evidence": "copies runtime/patches and invokes apply-patches.py",
            },
            {
                "path": "runtime/Containerfile.faststart",
                "status": "declared-input",
                "evidence": (
                    "copies runtime/patches and invokes apply-patches.py in compatible-base mode"
                ),
            },
        ],
        "configurations": [
            {
                "configuration_id": "reference-glm52-runtime",
                "lane": "reference",
                "maturity": "live-validated",
                "consumption_status": reference_status,
                "evidence": reference_evidence,
                "notes": reference_notes,
            },
            {
                "configuration_id": "glm52-exl3-tr3-3.25bpw-lmcache-cs512",
                "lane": "public-functional",
                "maturity": "live-validated",
                "consumption_status": "unknown",
                "evidence": None,
                "notes": (
                    "The current source tree does not provide an exact image/build-receipt "
                    "chain from this lock to the named live EXL3 image."
                ),
            },
            {
                "configuration_id": "glm52-nf3-hybrid",
                "lane": "public-functional",
                "maturity": "accepted",
                "consumption_status": "unknown",
                "evidence": None,
                "notes": (
                    "The accepted NF3 recipe uses a separate bootstrap/image lineage; this "
                    "inventory does not infer consumption from shared repository presence."
                ),
            },
        ],
    }


def validate_row_schema(operations: list[dict]) -> None:
    required = {
        "operation_id",
        "build_order",
        "path",
        "operation_kind",
        "provenance_class",
        "attribution_confidence",
        "attribution_basis",
        "currently_consumed_by",
        "natural_destination",
        "destination_basis",
        "destination_confidence",
        "destination_subclassifications",
        "missing_extension_seam",
        "evidence_needed_to_retire",
        "migration_status",
        "uncertainty_notes",
        "manifest_linkage",
    }
    proof_fields = {
        "replacement_identity",
        "semantic_equivalence",
        "numerical_correctness",
        "distributed_and_graph_behavior",
        "performance_and_reliability",
        "artifact_attestation",
        "rollback",
        "owner_acceptance",
    }
    for operation in operations:
        missing = required - set(operation)
        require(not missing, f"{operation.get('operation_id')} missing fields: {sorted(missing)}")
        require(
            operation["attribution_confidence"] in {"high", "medium", "low", "unknown"},
            f"invalid attribution confidence for {operation['operation_id']}",
        )
        require(
            operation["destination_confidence"] in {"high", "medium", "low", "unknown"},
            f"invalid destination confidence for {operation['operation_id']}",
        )
        seam = operation["missing_extension_seam"]
        require(isinstance(seam, dict), f"missing seam record for {operation['operation_id']}")
        require(
            seam.get("status") in {"none", "known", "unknown"},
            f"invalid seam status for {operation['operation_id']}",
        )
        if seam["status"] == "none":
            require(bool(seam.get("justification")), f"none seam needs justification: {operation['operation_id']}")
        elif seam["status"] == "unknown":
            require(bool(seam.get("reason")), f"unknown seam needs reason: {operation['operation_id']}")
        else:
            require(bool(seam.get("description")), f"known seam needs description: {operation['operation_id']}")
        proof = operation["evidence_needed_to_retire"]
        require(isinstance(proof, dict), f"retirement proof must be an object: {operation['operation_id']}")
        require(
            set(proof) == proof_fields,
            f"retirement proof fields drifted for {operation['operation_id']}",
        )
        consumed = operation["currently_consumed_by"]
        require(
            set(consumed) == {"generic_overlay_builders", "configurations"},
            f"consumption fields drifted for {operation['operation_id']}",
        )
        require(
            len(consumed["generic_overlay_builders"]) == 2,
            f"generic builder count drifted for {operation['operation_id']}",
        )
        config_ids = {item["configuration_id"] for item in consumed["configurations"]}
        require(
            config_ids
            == {
                "reference-glm52-runtime",
                "glm52-exl3-tr3-3.25bpw-lmcache-cs512",
                "glm52-nf3-hybrid",
            },
            f"per-configuration coverage drifted for {operation['operation_id']}",
        )
        for configuration in consumed["configurations"]:
            require(
                configuration["consumption_status"]
                in {"recovered-from-runtime", "equivalent-semantics-reported", "unknown"},
                f"invalid configuration consumption status for {operation['operation_id']}",
            )
            require(
                configuration["maturity"]
                in {"planned", "candidate", "offline-validated", "live-validated", "accepted"},
                f"noncanonical configuration maturity for {operation['operation_id']}",
            )
        subclasses = operation["destination_subclassifications"]
        if operation["natural_destination"] == "mixed/requires-decomposition":
            require(len(subclasses) >= 2, f"mixed operation lacks decomposition: {operation['operation_id']}")
        else:
            require(not subclasses, f"single-owner operation has subclasses: {operation['operation_id']}")


def build_document() -> dict:
    lock = load_json(ROOT / "runtime/runtime-lock.json")
    lock_hashes = {entry["path"]: entry["sha256"] for entry in lock["overlays"]}
    provenance = load_json(REF_ROOT / "provenance.json")
    ref_preimages = load_json(REF_ROOT / "preimages.json")
    additions = load_json(REF_ROOT / "additions.json")
    sparkcache_preimages = load_json(SPARKCACHE_ROOT / "preimages.json")
    provenance_entries = {entry["path"]: entry for entry in provenance["entries"]}

    # Verify every overlay input pinned by the lock, not just the 73 operation
    # artifacts. This also links the four operation manifests used below.
    for relative, expected in lock_hashes.items():
        observed = sha256(ROOT / relative)
        require(observed == expected, f"lock hash mismatch for {relative}")
    require(
        provenance["counts"] == {"modified": 59, "added": 12, "files": 71},
        "reference provenance counts drifted",
    )
    require(len(provenance_entries) == 71, "reference provenance paths are not unique")

    operations: list[dict] = []
    for patch_name, preimage in sorted(ref_preimages.items()):
        path = preimage["target_path"].removeprefix("vllm/")
        source = f"runtime/patches/00-reference-vllm/{patch_name}"
        marker = provenance_entries[path]
        destination = destination_for(path)
        operations.append(
            {
                "operation_id": f"recovered-reference:modify:{path}",
                "path": preimage["target_path"],
                "operation_kind": "modify",
                "provenance_class": "recovered-reference",
                "attribution_confidence": "high",
                "attribution_basis": (
                    "Exact path is present in the hash-linked recovered provenance and "
                    "preimage manifests; this establishes recovery class, not authorship."
                ),
                "currently_consumed_by": consumption("recovered-reference"),
                "natural_destination": destination,
                "destination_basis": destination_basis(destination),
                "destination_confidence": (
                    "unknown" if destination == "unknown" else "medium"
                ),
                "destination_subclassifications": MIXED_SUBCLASSIFICATIONS.get(path, []),
                "missing_extension_seam": missing_seam(destination),
                "evidence_needed_to_retire": retirement_proof(destination),
                "migration_status": "inventory-classified-not-migrated",
                "uncertainty_notes": (
                    f"Recovered byte delta; provenance marker classification "
                    f"'{marker['classification']}' and markers {marker['markers']} are "
                    "best-effort signals, not authorship evidence. Destination is a "
                    "planning judgment that still requires owner review."
                ),
                "manifest_linkage": {
                    "source_artifact": source,
                    "artifact_sha256": lock_hashes[source],
                    "operation_manifest": "runtime/patches/00-reference-vllm/preimages.json",
                    "preimage_sha256": preimage["preimage_sha256"],
                    "provenance_manifest": "runtime/patches/00-reference-vllm/provenance.json",
                    "changed_lines": marker["changed_lines"],
                },
            }
        )

    for source_rel, addition in sorted(additions.items()):
        path = addition["target_path"].removeprefix("vllm/")
        source = f"runtime/patches/00-reference-vllm/added/{source_rel}"
        marker = provenance_entries[path]
        destination = destination_for(path)
        operations.append(
            {
                "operation_id": f"recovered-reference:add:{path}",
                "path": addition["target_path"],
                "operation_kind": "add",
                "provenance_class": "recovered-reference",
                "attribution_confidence": "high",
                "attribution_basis": (
                    "Exact path is present in the hash-linked recovered provenance and "
                    "addition manifests; this establishes recovery class, not authorship."
                ),
                "currently_consumed_by": consumption("recovered-reference"),
                "natural_destination": destination,
                "destination_basis": destination_basis(destination),
                "destination_confidence": (
                    "unknown" if destination == "unknown" else "medium"
                ),
                "destination_subclassifications": MIXED_SUBCLASSIFICATIONS.get(path, []),
                "missing_extension_seam": missing_seam(destination),
                "evidence_needed_to_retire": retirement_proof(destination),
                "migration_status": "inventory-classified-not-migrated",
                "uncertainty_notes": (
                    f"Recovered fork-only file; provenance marker classification "
                    f"'{marker['classification']}' and markers {marker['markers']} are "
                    "best-effort signals, not authorship evidence. Destination is a "
                    "planning judgment that still requires owner review."
                ),
                "manifest_linkage": {
                    "source_artifact": source,
                    "artifact_sha256": lock_hashes[source],
                    "operation_manifest": "runtime/patches/00-reference-vllm/additions.json",
                    "declared_content_sha256": addition["sha256"],
                    "provenance_manifest": "runtime/patches/00-reference-vllm/provenance.json",
                    "changed_lines": marker["changed_lines"],
                },
            }
        )

    seams = {
        "010-sparkcache-async-rollback.patch": (
            "A KV-connector restore-failure rollback hook that restores speculative "
            "request state before rescheduling."
        ),
        "020-sparkcache-vmm-exemption.patch": (
            "A connector capability declaring whether it registers KV-cache GPU "
            "memory with an external device."
        ),
    }
    patch_notes = {
        "010-sparkcache-async-rollback.patch": (
            "Independently authored SparkCache compatibility behavior. It modifies the "
            "same target path as one recovered reference operation, so target paths are "
            "not globally unique even though operation IDs are."
        ),
        "020-sparkcache-vmm-exemption.patch": (
            "Independently authored SparkCache compatibility behavior. It modifies the "
            "same target path as one recovered reference operation, so target paths are "
            "not globally unique even though operation IDs are."
        ),
    }
    for patch_name, preimage in sorted(sparkcache_preimages.items()):
        path = preimage["target_path"]
        source = f"runtime/patches/vllm/{patch_name}"
        operations.append(
            {
                "operation_id": f"independently-authored-sparkcache:modify:{patch_name}",
                "path": path,
                "operation_kind": "modify",
                "provenance_class": "independently-authored-sparkcache",
                "attribution_confidence": "high",
                "attribution_basis": (
                    "runtime/patches/vllm/README.md explicitly identifies both public "
                    "patches as independently written SparkCache compatibility patches."
                ),
                "currently_consumed_by": consumption("independently-authored-sparkcache"),
                "natural_destination": "SparkCache connector",
                "destination_basis": (
                    "The independently authored patches exist only to satisfy SparkCache "
                    "connector semantics described by runtime/patches/vllm/README.md."
                ),
                "destination_confidence": "high",
                "destination_subclassifications": [],
                "missing_extension_seam": {
                    "status": "known",
                    "description": seams[patch_name],
                    "evidence": "runtime/patches/vllm/README.md and patch content",
                },
                "evidence_needed_to_retire": retirement_proof("SparkCache connector"),
                "migration_status": "temporary-patch-awaiting-public-connector-seam",
                "uncertainty_notes": patch_notes[patch_name],
                "manifest_linkage": {
                    "source_artifact": source,
                    "artifact_sha256": lock_hashes[source],
                    "operation_manifest": "runtime/patches/vllm/preimages.json",
                    "preimage_sha256": preimage["preimage_sha256"],
                },
            }
        )

    # Preserve executable component order: recovered modifications (sorted by
    # patch name), recovered additions, then the SparkCache compatibility
    # component. apply-patches.py sorts component and patch names this way.
    for build_order, operation in enumerate(operations, start=1):
        operation["build_order"] = build_order
    validate_row_schema(operations)
    counts = {
        "recovered_modified": sum(
            item["provenance_class"] == "recovered-reference"
            and item["operation_kind"] == "modify"
            for item in operations
        ),
        "recovered_added": sum(
            item["provenance_class"] == "recovered-reference"
            and item["operation_kind"] == "add"
            for item in operations
        ),
        "sparkcache_compatibility_modified": sum(
            item["provenance_class"] == "independently-authored-sparkcache"
            for item in operations
        ),
        "operations": len(operations),
        "unique_operation_ids": len({item["operation_id"] for item in operations}),
        "unique_target_paths": len({item["path"] for item in operations}),
    }
    expected_counts = {
        "recovered_modified": 59,
        "recovered_added": 12,
        "sparkcache_compatibility_modified": 2,
        "operations": 73,
        "unique_operation_ids": 73,
        "unique_target_paths": 71,
    }
    require(counts == expected_counts, f"operation count/uniqueness drift: {counts}")
    recovered_paths = {
        item["path"].removeprefix("vllm/")
        for item in operations
        if item["provenance_class"] == "recovered-reference"
    }
    require(
        recovered_paths == set(provenance_entries),
        "inventory does not exactly cover recovered provenance paths",
    )
    classified_paths = set(DESTINATIONS) | set(MIXED_SUBCLASSIFICATIONS)
    require(
        classified_paths <= recovered_paths,
        "destination classification names a path outside the recovered overlay",
    )
    require(
        not (set(DESTINATIONS) & set(MIXED_SUBCLASSIFICATIONS)),
        "a path cannot have both a single and mixed destination",
    )

    document = {
        "schema": "sparkring-overlay-ownership-inventory/v1",
        "title": "vLLM overlay ownership and provenance inventory",
        "scope": {
            "lane": ["reference", "public-functional"],
            "maturity": "offline-validated",
            "artifact_status": "draft",
            "hardware": "no hardware contacted; offline source audit",
            "runtime_id": lock["runtime_id"],
            "upstream_vllm": lock["vllm"],
            "classification_policy": (
                "Natural destination describes proposed long-term ownership, not "
                "authorship or present maturity. Unknown is retained where the source "
                "does not establish a destination."
            ),
            "consumption_policy": (
                "Only the two generic overlay builders are recorded as declared consumers. "
                "Each named configuration is evaluated separately and remains unknown "
                "unless an exact artifact chain proves consumption."
            ),
        },
        "allowed_values": {
            "operation_kind": ["modify", "add"],
            "attribution_confidence": ["high", "medium", "low", "unknown"],
            "destination_confidence": ["high", "medium", "low", "unknown"],
            "missing_extension_seam.status": ["none", "known", "unknown"],
            "configuration_consumption_status": [
                "recovered-from-runtime",
                "equivalent-semantics-reported",
                "unknown",
            ],
            "provenance_class": [
                "recovered-reference",
                "independently-authored-sparkcache",
            ],
            "natural_destination": [
                "vLLM",
                "SparkInfer",
                "NCCL",
                "SIRCL adapter/plugin",
                "SparkCache connector",
                "SparkRing distribution",
                "historical-only",
                "mixed/requires-decomposition",
                "unknown",
            ],
        },
        "source_manifests": {
            "runtime_lock": {
                "path": "runtime/runtime-lock.json",
                "sha256": sha256(ROOT / "runtime/runtime-lock.json"),
            },
            "reference_preimages": {
                "path": "runtime/patches/00-reference-vllm/preimages.json",
                "sha256": sha256(REF_ROOT / "preimages.json"),
            },
            "reference_additions": {
                "path": "runtime/patches/00-reference-vllm/additions.json",
                "sha256": sha256(REF_ROOT / "additions.json"),
            },
            "reference_provenance": {
                "path": "runtime/patches/00-reference-vllm/provenance.json",
                "sha256": sha256(REF_ROOT / "provenance.json"),
            },
            "sparkcache_preimages": {
                "path": "runtime/patches/vllm/preimages.json",
                "sha256": sha256(SPARKCACHE_ROOT / "preimages.json"),
            },
        },
        "validation": {
            "expected_counts": {
                "recovered_modified": 59,
                "recovered_added": 12,
                "sparkcache_compatibility_modified": 2,
                "operations": 73,
            },
            "observed_counts": counts,
            "pinning_contract": (
                "All 61 modifications (59 recovered plus 2 SparkCache) are "
                "preimage-pinned. All 12 additions are content-addressed and refuse to "
                "overwrite an existing target. All 73 ordered operations fail closed."
            ),
            "target_path_collision_note": (
                "The two SparkCache patches intentionally follow recovered operations "
                "on vllm/config/vllm.py and vllm/v1/core/sched/scheduler.py."
            ),
        },
        "operations": operations,
    }
    return document


def serialize(document: dict) -> str:
    serialized = json.dumps(document, indent=2, sort_keys=False) + "\n"
    require(str(ROOT) not in serialized, "workspace path leaked into inventory")
    require(
        not re.search(
        r"(?<![0-9])(?:10\.[0-9]{1,3}(?:\.[0-9]{1,3}){2}|"
        r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9])",
        serialized,
        ),
        "private address found in inventory",
    )
    return serialized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="inventory path (default: committed docs/configurations artifact)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the output is missing or differs from deterministic generation",
    )
    args = parser.parse_args(argv)

    try:
        serialized = serialize(build_document())
        output = args.output.resolve()
        if args.check:
            require(output.is_file(), f"inventory is missing: {output}")
            observed = output.read_text(encoding="utf-8")
            require(observed == serialized, f"inventory is stale: {output}")
            print(f"overlay ownership inventory is fresh: {output}")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8", newline="\n")
        print(f"wrote overlay ownership inventory: {output}")
    except (InventoryError, KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"overlay ownership inventory: ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
