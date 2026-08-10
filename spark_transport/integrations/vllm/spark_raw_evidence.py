"""Raw per-rank/per-iteration evidence producer and validator for TP4.

This module produces a ``tp4_raw_evidence/v2`` artifact that records,
for every rank and every iteration:

- Per-rank input identity (SHA-256 of the input tensor's raw BF16 bytes).
- Per-rank observed output identity (SHA-256 of the output tensor's raw
  BF16 bytes) **and** a bounded sample of actual output values (first
  ``OUTPUT_SAMPLE_SIZE`` floats per iteration).
- Per-rank FP32 truth identity (SHA-256 of the unrounded FP32 truth
  tensor's raw bytes).
- Per-rank metrics: MAE, RMSE, max_abs_error (vs **unrounded** FP32),
  mismatch_count (vs FP32 truth rounded once to BF16),
  outside_tolerance_count (vs unrounded FP32, using
  ``TOLERANCE_THRESHOLD_BF16``).

The validator **recomputes** every metric, hash, and count from the
deterministic input generation — it does NOT trust caller-supplied
aggregates. Since inputs are deterministically seeded by
``make_rank_input(sequence, rank, world_size, rows, width, pattern)``,
the validator can regenerate all inputs, recompute FP32 truth and the
TP4 ring reduction, verify all stored hashes, and recompute all metrics.

The artifact carries a SHA-256 binding of the entire raw payload.

All computation is CPU-only.  No CUDA, RDMA, NCCL, or live cluster.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import torch

from spark_fp32_ground_truth import (
    TOLERANCE_POLICY_NAME,
    TOLERANCE_THRESHOLD_BF16,
    fp32_ground_truth,
    fp32_truth_rounded_to_dtype,
    make_rank_input,
    tp4_ring_reduce_all_ranks,
)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

RAW_EVIDENCE_SCHEMA_V2 = "tp4_raw_evidence/v2"
REDUCED_EVIDENCE_SCHEMA = "tp4_reduced_evidence/v1"

# Evidence type labels.
# - "modeled": outputs computed locally by tp4_ring_reduce_all_ranks.
#   Cannot prove actual SIRCL or NCCL numerical output.
# - "observed": DISABLED — no real offline runtime output seam exists.
#   The tp4_numerical_audit.py probe requires CUDA, RDMA, and a live
#   4-rank process group; it cannot run in the public checkout.  An
#   ObservedEvidenceReceipt accepts arbitrary caller-supplied hashes,
#   which is caller-fabricated — it cannot prove live numerical
#   output.  The observed type is retained for documentation but is
#   NOT a valid evidence type.  Any artifact claiming evidence_type=
#   "observed" is rejected by both the producer and the validator.
EVIDENCE_TYPE_MODELED = "modeled"
EVIDENCE_TYPE_OBSERVED = "observed"  # retained for error messages only
_VALID_EVIDENCE_TYPES = frozenset({EVIDENCE_TYPE_MODELED})

# Transport selectors bound into observed evidence.
SELECTOR_CUSTOM = "custom"
SELECTOR_DISABLED = "disabled"
_VALID_SELECTORS = frozenset({SELECTOR_CUSTOM, SELECTOR_DISABLED})

# Bounded sample size: first N output values per iteration to keep
# the raw artifact under 64 MiB for reasonable iteration counts.
OUTPUT_SAMPLE_SIZE = 64

# Maximum artifact size: 64 MiB.
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

# SHA-256 pattern: exactly 64 lowercase hex characters.
_SHA256_RE = "^[0-9a-f]{64}$"

# Workload parameters bound into the artifact.
_WORKLOAD_PATTERN = "random"
_WORKLOAD_ROWS = 1


class RawEvidenceError(ValueError):
    """Raised when a raw evidence artifact is invalid or tampered."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute SHA-256 of a tensor's raw bytes via int16 view.

    Uses ``tensor.view(torch.int16).numpy().tobytes()`` so the hash
    covers the exact bit pattern, not a stringified representation.
    """
    raw = tensor.view(torch.int16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _is_int_not_bool(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_float(val: Any) -> bool:
    """True if *val* is a finite float (or int), not bool, not NaN/Inf."""
    if isinstance(val, bool):
        return False
    if isinstance(val, int):
        return True
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return False
        return True
    return False


def _is_finite_number(val: Any) -> bool:
    """True for finite int or float, excluding bool/NaN/Inf."""
    if isinstance(val, bool):
        return False
    if isinstance(val, int):
        return True
    if isinstance(val, float):
        return not (math.isnan(val) or math.isinf(val))
    return False


def _validate_sha256(val: Any, name: str) -> str:
    if not isinstance(val, str):
        raise RawEvidenceError(f"{name} must be a string, got {type(val).__name__}")
    if not re.match(_SHA256_RE, val):
        raise RawEvidenceError(
            f"{name} must be 64 lowercase hex chars, got '{val[:16]}...'"
        )
    return val


def _require_keys(data: Any, required: set[str], name: str) -> None:
    if not isinstance(data, dict):
        raise RawEvidenceError(f"{name} must be a dict")
    keys = set(data.keys())
    missing = required - keys
    extra = keys - required
    if missing:
        raise RawEvidenceError(f"{name} missing keys: {sorted(missing)}")
    if extra:
        raise RawEvidenceError(f"{name} has extra keys: {sorted(extra)}")


# ---------------------------------------------------------------------------
# Canonical serialization and binding
# ---------------------------------------------------------------------------

def compute_artifact_sha256(artifact: dict) -> str:
    """Compute SHA-256 of the canonical JSON serialization of *artifact*.

    The ``artifact_sha256`` field itself (if present) is excluded from
    the hash so the binding is self-consistent (the hash does not
    include the field that stores the hash).
    """
    a = copy.deepcopy(artifact)
    a.pop("artifact_sha256", None)
    canonical = json.dumps(a, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compute_reduced_sha256(artifact: dict) -> str:
    """Compute SHA-256 of the reduced (aggregated) metrics.

    The reduced evidence is the per-rank aggregate metrics derived
    from the per-iteration raw data.  This hash cross-references the
    raw artifact to the reduced view.

    The reduced metrics are **recomputed** by the validator, not
    copied from the artifact.
    """
    reduced = _reduce_artifact(artifact)
    canonical = json.dumps(reduced, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _reduce_artifact(artifact: dict) -> dict:
    """Reduce per-iteration raw data into per-rank aggregate metrics.

    Metrics are recomputed from the per-iteration entries, not trusted
    from caller-supplied aggregates.
    """
    reduced_ranks = []
    for rank_entry in artifact["per_rank_raw"]:
        iterations = rank_entry["per_iteration"]
        n = len(iterations)
        agg_mae = sum(it["mae"] for it in iterations) / n if n else 0.0
        agg_rmse = (
            (sum(it["rmse"] ** 2 for it in iterations) / n) ** 0.5
            if n else 0.0
        )
        agg_max = max(it["max_abs_error"] for it in iterations) if n else 0.0
        agg_mismatch = sum(it["mismatch_count"] for it in iterations)
        agg_outside = sum(it["outside_tolerance_count"] for it in iterations)
        reduced_ranks.append({
            "rank": rank_entry["rank"],
            "iterations": n,
            "mae": agg_mae,
            "rmse": agg_rmse,
            "max_abs_error": agg_max,
            "mismatch_count": agg_mismatch,
            "outside_tolerance_count": agg_outside,
        })
    reduced = {
        "schema": REDUCED_EVIDENCE_SCHEMA,
        "iterations": artifact["iterations"],
        "elements": artifact["elements"],
        "ranks": artifact["ranks"],
        "tolerance_policy": artifact["tolerance_policy"],
        "per_rank_reduced": reduced_ranks,
    }
    canonical = json.dumps(reduced, sort_keys=True, separators=(",", ":"))
    reduced["reduced_sha256"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    return reduced

# ---------------------------------------------------------------------------
# Observed evidence receipt — execution-derived commitments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservedEvidenceReceipt:
    """One rank's execution-derived evidence for one iteration.

    **DISABLED** — no real offline runtime output seam exists in the
    public checkout.  The ``tp4_numerical_audit.py`` probe requires
    CUDA, RDMA, and a live 4-rank process group; it cannot run
    offline.  This class accepted arbitrary caller-supplied output
    hashes, which is caller-fabricated — it cannot prove live
    numerical output.

    The class is retained for documentation and error messaging but
    ``evidence_type="observed"`` is rejected by the producer and
    validator.  A modeled artifact must remain explicitly modeled and
    cannot satisfy live numerical proof.
    """

    rank: int
    iteration: int
    output_hash: str
    selector: str
    custom_collectives: int
    fallback_collectives: int
    unsupported_bypassed_collectives: int
    unclassified_collectives: int

    def __post_init__(self) -> None:
        if not _is_int_not_bool(self.rank) or self.rank < 0:
            raise RawEvidenceError(f"rank must be a non-negative int, got {self.rank}")
        if not _is_int_not_bool(self.iteration) or self.iteration < 0:
            raise RawEvidenceError(
                f"iteration must be a non-negative int, got {self.iteration}"
            )
        _validate_sha256(self.output_hash, "output_hash")
        if self.selector not in _VALID_SELECTORS:
            raise RawEvidenceError(
                f"selector must be one of {sorted(_VALID_SELECTORS)}, "
                f"got '{self.selector}'"
            )
        for name, val in [
            ("custom_collectives", self.custom_collectives),
            ("fallback_collectives", self.fallback_collectives),
            ("unsupported_bypassed_collectives", self.unsupported_bypassed_collectives),
            ("unclassified_collectives", self.unclassified_collectives),
        ]:
            if not _is_int_not_bool(val) or val < 0:
                raise RawEvidenceError(f"{name} must be a non-negative int, got {val}")


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class RawEvidenceProducer:
    """Produce a ``tp4_raw_evidence/v2`` raw evidence artifact.

    All computation is CPU-only, using the modeled TP4 ring reduction
    from ``spark_fp32_ground_truth.py``.  No CUDA, RDMA, NCCL, or
    live cluster is involved.

    For ``evidence_type="modeled"`` (default and only valid type),
    the artifact records modeled outputs and is labeled as such —
    it cannot prove actual SIRCL or NCCL numerical output.

    ``evidence_type="observed"`` is DISABLED — no real offline
    runtime output seam exists.  The producer rejects it.  A modeled
    artifact must remain explicitly modeled and cannot satisfy live
    numerical proof.
    """

    def produce(
        self,
        record_iterations: int,
        world_size: int = 4,
        elements: int = 6144,
        *,
        evidence_type: str = EVIDENCE_TYPE_MODELED,
        observed_receipts: dict[int, list[ObservedEvidenceReceipt]] | None = None,
    ) -> dict:
        """Produce the raw evidence artifact as a dict.

        Parameters
        ----------
        record_iterations : int
            Number of iterations to record (must be >= 1).
        world_size : int
            Number of ranks (must be 4 for TP4 ring reduce).
        elements : int
            Number of elements per tensor (default 6144).
        evidence_type : str
            Must be ``"modeled"`` (the only valid type).
            ``"observed"`` is DISABLED and will be rejected.
        observed_receipts : dict[int, list[ObservedEvidenceReceipt]] | None
            Deprecated — ``evidence_type="observed"`` is disabled.
            If provided, the producer still rejects observed evidence.

        Returns
        -------
        dict
            The raw evidence artifact dict with schema
            ``tp4_raw_evidence/v2``.
        """
        if not _is_int_not_bool(record_iterations) or record_iterations < 1:
            raise RawEvidenceError(
                f"record_iterations must be a positive int, got {record_iterations}"
            )
        if not _is_int_not_bool(world_size) or world_size != 4:
            raise RawEvidenceError(
                f"world_size must be 4 for TP4 ring reduce, got {world_size}"
            )
        if not _is_int_not_bool(elements) or elements < 1:
            raise RawEvidenceError(
                f"elements must be a positive int, got {elements}"
            )
        if evidence_type not in _VALID_EVIDENCE_TYPES:
            raise RawEvidenceError(
                f"evidence_type must be one of {sorted(_VALID_EVIDENCE_TYPES)}, "
                f"got '{evidence_type}'"
            )

        if evidence_type == EVIDENCE_TYPE_OBSERVED:
            if observed_receipts is None:
                raise RawEvidenceError(
                    "observed_receipts is required for "
                    "evidence_type='observed'"
                )
            # Validate receipt coverage: every rank, every iteration.
            for rank in range(world_size):
                if rank not in observed_receipts:
                    raise RawEvidenceError(
                        f"observed_receipts missing rank {rank}"
                    )
                rank_receipts = observed_receipts[rank]
                if len(rank_receipts) != record_iterations:
                    raise RawEvidenceError(
                        f"observed_receipts[{rank}] must have "
                        f"{record_iterations} entries, got "
                        f"{len(rank_receipts)}"
                    )
                for it_idx, rcpt in enumerate(rank_receipts):
                    if rcpt.rank != rank:
                        raise RawEvidenceError(
                            f"observed_receipts[{rank}][{it_idx}].rank "
                            f"must be {rank}, got {rcpt.rank}"
                        )
                    if rcpt.iteration != it_idx:
                        raise RawEvidenceError(
                            f"observed_receipts[{rank}][{it_idx}].iteration "
                            f"must be {it_idx}, got {rcpt.iteration}"
                        )

        per_rank_raw: list[dict[str, Any]] = []

        for rank in range(world_size):
            per_iteration: list[dict[str, Any]] = []
            for iteration in range(record_iterations):
                entry = self._produce_one_iteration(
                    iteration, rank, world_size, elements,
                )
                if evidence_type == EVIDENCE_TYPE_OBSERVED:
                    # Replace the modeled output_hash with the
                    # execution-derived output hash from the receipt.
                    rcpt = observed_receipts[rank][iteration]
                    entry["output_hash"] = rcpt.output_hash
                    # Add transport binding fields.
                    entry["transport_selector"] = rcpt.selector
                    entry["transport_custom"] = rcpt.custom_collectives
                    entry["transport_fallback"] = rcpt.fallback_collectives
                    entry["transport_unsupported"] = rcpt.unsupported_bypassed_collectives
                    entry["transport_unclassified"] = rcpt.unclassified_collectives
                    # Remove the modeled output sample — observed
                    # evidence does not include modeled output values.
                    # The validator must compare the observed output
                    # hash, not a recomputed modeled sample.
                    del entry["output_sample"]
                per_iteration.append(entry)
            per_rank_raw.append({
                "rank": rank,
                "per_iteration": per_iteration,
            })

        artifact: dict[str, Any] = {
            "schema": RAW_EVIDENCE_SCHEMA_V2,
            "evidence_type": evidence_type,
            "iterations": record_iterations,
            "elements": elements,
            "ranks": world_size,
            "workload_pattern": _WORKLOAD_PATTERN,
            "workload_rows": _WORKLOAD_ROWS,
            "tolerance_policy": TOLERANCE_POLICY_NAME,
            "per_rank_raw": per_rank_raw,
        }

        # Compute and attach the artifact SHA-256 binding.
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        return artifact

    def _produce_one_iteration(
        self,
        iteration: int,
        rank: int,
        world_size: int,
        elements: int,
    ) -> dict[str, Any]:
        """Produce one per-iteration entry for one rank.

        Generates the full set of rank inputs for this iteration,
        computes the TP4 ring reduction, and records per-rank
        identity hashes, a bounded output sample, and metrics.
        """
        # Generate all rank inputs for this iteration.
        inputs = [
            make_rank_input(
                sequence=iteration,
                rank=r,
                world_size=world_size,
                rows=_WORKLOAD_ROWS,
                width=elements,
                pattern=_WORKLOAD_PATTERN,
            )
            for r in range(world_size)
        ]

        # Compute the modeled TP4 ring reduction.
        outputs = tp4_ring_reduce_all_ranks(inputs)

        # Compute unrounded FP32 ground truth.
        truth_fp32 = fp32_ground_truth(inputs)

        # Compute FP32 truth rounded once to BF16 (for mismatch count).
        truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)

        # This rank's output.
        output = outputs[rank]
        input_tensor = inputs[rank]

        # Per-rank identity hashes.
        input_hash = _tensor_sha256(input_tensor)
        output_hash = _tensor_sha256(output)
        fp32_truth_hash = _tensor_sha256(truth_fp32)

        # Bounded output sample: first OUTPUT_SAMPLE_SIZE floats.
        output_sample = output.float()[:OUTPUT_SAMPLE_SIZE].tolist()

        # Metrics: compare this rank's output to unrounded FP32 truth.
        err = (output.float() - truth_fp32).abs()
        mae = float(err.mean().item())
        rmse = float((err.square().mean()).sqrt().item())
        max_abs_error = float(err.max().item())

        # mismatch_count: output vs FP32 truth rounded once to BF16.
        mismatch_count = int(
            torch.count_nonzero(
                output.view(torch.int16) != truth_bf16.view(torch.int16)
            ).item()
        )

        # outside_tolerance_count: abs error vs unrounded FP32 exceeds threshold.
        outside_tolerance_count = int(
            torch.count_nonzero(err > TOLERANCE_THRESHOLD_BF16).item()
        )

        return {
            "iteration": iteration,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "fp32_truth_hash": fp32_truth_hash,
            "output_sample": output_sample,
            "mae": mae,
            "rmse": rmse,
            "max_abs_error": max_abs_error,
            "mismatch_count": mismatch_count,
            "outside_tolerance_count": outside_tolerance_count,
        }


# ---------------------------------------------------------------------------
# Validator — recomputes everything from deterministic inputs
# ---------------------------------------------------------------------------

# Required keys for the top-level artifact.
_ARTIFACT_REQUIRED_KEYS = {
    "schema", "evidence_type", "iterations", "elements", "ranks",
    "workload_pattern", "workload_rows", "tolerance_policy",
    "per_rank_raw", "artifact_sha256",
}

# Required keys for each per-iteration entry (modeled).
_ITERATION_REQUIRED_KEYS_MODELED = {
    "iteration", "input_hash", "output_hash", "fp32_truth_hash",
    "output_sample", "mae", "rmse", "max_abs_error",
    "mismatch_count", "outside_tolerance_count",
}

# Required keys for each per-iteration entry (observed).
# Observed entries have transport bindings instead of output_sample.
_ITERATION_REQUIRED_KEYS_OBSERVED = {
    "iteration", "input_hash", "output_hash", "fp32_truth_hash",
    "mae", "rmse", "max_abs_error",
    "mismatch_count", "outside_tolerance_count",
    "transport_selector", "transport_custom", "transport_fallback",
    "transport_unsupported", "transport_unclassified",
}

# Required keys for each per-rank entry.
_RANK_REQUIRED_KEYS = {"rank", "per_iteration"}


def validate_raw_evidence(
    artifact: dict,
    expected_binding: str | None = None,
) -> dict:
    """Validate a raw evidence artifact by recomputing everything.

    The validator does NOT trust caller-supplied metrics or hashes.
    It regenerates all inputs deterministically, recomputes FP32 truth
    and the TP4 ring reduction, verifies all stored hashes, recomputes
    all metrics (MAE, RMSE, max error, mismatch count, tolerance count),
    and verifies the output sample matches the recomputed output.

    Parameters
    ----------
    artifact : dict
        The raw evidence artifact dict to validate.
    expected_binding : str | None
        If provided, the artifact's ``artifact_sha256`` must match
        this value.

    Returns
    -------
    dict
        Derived/reduced metrics including the raw artifact binding,
        the reduced evidence binding, and per-rank aggregates.

    Raises
    ------
    RawEvidenceError
        If the artifact is missing keys, has duplicate/reordered
        rank or iteration data, contains NaN/Inf/bool values,
        has a truncated output sample, has a tampered hash, or
        if any recomputed metric/hash does not match the stored value.
    """
    if not isinstance(artifact, dict):
        raise RawEvidenceError(
            f"artifact must be a dict, got {type(artifact).__name__}"
        )

    # Exact-key validation at top level.
    _require_keys(artifact, _ARTIFACT_REQUIRED_KEYS, "artifact")

    # Validate schema.
    if artifact["schema"] != RAW_EVIDENCE_SCHEMA_V2:
        raise RawEvidenceError(
            f"schema must be '{RAW_EVIDENCE_SCHEMA_V2}', "
            f"got '{artifact['schema']}'"
        )

    # Validate evidence_type.
    evidence_type = artifact["evidence_type"]
    if evidence_type not in _VALID_EVIDENCE_TYPES:
        raise RawEvidenceError(
            f"evidence_type must be one of {sorted(_VALID_EVIDENCE_TYPES)}, "
            f"got '{evidence_type}'"
        )
    is_observed = evidence_type == EVIDENCE_TYPE_OBSERVED

    # Select iteration keys based on evidence_type.
    iter_required_keys = (
        _ITERATION_REQUIRED_KEYS_OBSERVED
        if is_observed
        else _ITERATION_REQUIRED_KEYS_MODELED
    )

    # Validate scalar fields.
    iterations = artifact["iterations"]
    if not _is_int_not_bool(iterations) or iterations < 1:
        raise RawEvidenceError(
            f"iterations must be a positive int, got {iterations}"
        )

    elements = artifact["elements"]
    if not _is_int_not_bool(elements) or elements < 1:
        raise RawEvidenceError(
            f"elements must be a positive int, got {elements}"
        )

    ranks = artifact["ranks"]
    if not _is_int_not_bool(ranks) or ranks < 1:
        raise RawEvidenceError(
            f"ranks must be a positive int, got {ranks}"
        )

    if artifact["tolerance_policy"] != TOLERANCE_POLICY_NAME:
        raise RawEvidenceError(
            f"tolerance_policy must be '{TOLERANCE_POLICY_NAME}', "
            f"got '{artifact['tolerance_policy']}'"
        )

    # Validate workload binding fields.
    workload_pattern = artifact["workload_pattern"]
    if workload_pattern != _WORKLOAD_PATTERN:
        raise RawEvidenceError(
            f"workload_pattern must be '{_WORKLOAD_PATTERN}', "
            f"got '{workload_pattern}'"
        )

    workload_rows = artifact["workload_rows"]
    if not _is_int_not_bool(workload_rows) or workload_rows != _WORKLOAD_ROWS:
        raise RawEvidenceError(
            f"workload_rows must be {_WORKLOAD_ROWS}, got {workload_rows}"
        )

    # Validate artifact_sha256 format.
    stored_sha = _validate_sha256(artifact["artifact_sha256"], "artifact_sha256")

    # Recompute the binding and check for tampering.
    recomputed_sha = compute_artifact_sha256(artifact)
    if stored_sha != recomputed_sha:
        raise RawEvidenceError(
            f"artifact_sha256 tampered: stored '{stored_sha[:16]}...' "
            f"but recomputed '{recomputed_sha[:16]}...'"
        )

    # If an expected binding is provided, verify it matches.
    if expected_binding is not None and stored_sha != expected_binding:
        raise RawEvidenceError(
            f"artifact_sha256 mismatch: expected '{expected_binding[:16]}...' "
            f"but got '{stored_sha[:16]}...'"
        )

    # Validate per_rank_raw structure.
    per_rank_raw = artifact["per_rank_raw"]
    if not isinstance(per_rank_raw, list) or len(per_rank_raw) != ranks:
        raise RawEvidenceError(
            f"per_rank_raw must be a list of {ranks} entries, "
            f"got len={len(per_rank_raw) if isinstance(per_rank_raw, list) else 'N/A'}"
        )

    seen_ranks: set[int] = set()
    for rank_idx, rank_entry in enumerate(per_rank_raw):
        if not isinstance(rank_entry, dict):
            raise RawEvidenceError(
                f"per_rank_raw[{rank_idx}] must be a dict"
            )
        _require_keys(rank_entry, _RANK_REQUIRED_KEYS,
                      f"per_rank_raw[{rank_idx}]")

        rank_val = rank_entry["rank"]
        if not _is_int_not_bool(rank_val):
            raise RawEvidenceError(
                f"per_rank_raw[{rank_idx}].rank must be an int, "
                f"got {rank_val}"
            )
        if rank_val in seen_ranks:
            raise RawEvidenceError(
                f"duplicate rank {rank_val} in per_rank_raw"
            )
        if rank_val != rank_idx:
            raise RawEvidenceError(
                f"per_rank_raw[{rank_idx}].rank must be {rank_idx}, "
                f"got {rank_val}"
            )
        seen_ranks.add(rank_val)

        per_iteration = rank_entry["per_iteration"]
        if not isinstance(per_iteration, list):
            raise RawEvidenceError(
                f"per_rank_raw[{rank_idx}].per_iteration must be a list"
            )
        if len(per_iteration) != iterations:
            raise RawEvidenceError(
                f"per_rank_raw[{rank_idx}].per_iteration must have "
                f"{iterations} entries, got {len(per_iteration)}"
            )

        seen_iterations: set[int] = set()
        for it_idx, it_entry in enumerate(per_iteration):
            if not isinstance(it_entry, dict):
                raise RawEvidenceError(
                    f"per_rank_raw[{rank_idx}].per_iteration[{it_idx}] "
                    f"must be a dict"
                )
            _require_keys(
                    it_entry, iter_required_keys,
                    f"per_rank_raw[{rank_idx}].per_iteration[{it_idx}]",
            )

            it_val = it_entry["iteration"]
            if not _is_int_not_bool(it_val):
                raise RawEvidenceError(
                    f"per_rank_raw[{rank_idx}].per_iteration[{it_idx}]"
                    f".iteration must be an int, got {it_val}"
                )
            if it_val in seen_iterations:
                raise RawEvidenceError(
                    f"duplicate iteration {it_val} in rank {rank_val}"
                )
            if it_val != it_idx:
                raise RawEvidenceError(
                    f"per_rank_raw[{rank_idx}].per_iteration[{it_idx}]"
                    f".iteration must be {it_idx}, got {it_val}"
                )
            seen_iterations.add(it_val)

            # Validate field types and values.
            _validate_iteration_entry(
                it_entry, rank_idx, it_idx, elements, is_observed,
            )

            # Recompute from deterministic inputs.
            _recompute_and_verify_iteration(
                it_entry, rank_idx, it_idx, rank_val,
                iterations, elements, ranks, workload_pattern, workload_rows,
                is_observed,
            )

    # Compute reduced evidence and its binding.
    reduced = _reduce_artifact(artifact)
    reduced_sha = reduced["reduced_sha256"]

    return {
        "valid": True,
        "schema": artifact["schema"],
        "evidence_type": evidence_type,
        "artifact_sha256": stored_sha,
        "reduced_sha256": reduced_sha,
        "reduced_evidence": reduced,
        "iterations": iterations,
        "elements": elements,
        "ranks": ranks,
        "per_rank_reduced": reduced["per_rank_reduced"],
    }


def _validate_iteration_entry(
    entry: dict,
    rank_idx: int,
    it_idx: int,
    elements: int,
    is_observed: bool = False,
) -> None:
    """Validate one per-iteration entry's field types and values."""
    label = f"per_rank_raw[{rank_idx}].per_iteration[{it_idx}]"

    # Validate hash fields.
    for key in ("input_hash", "output_hash", "fp32_truth_hash"):
        _validate_sha256(entry[key], f"{label}.{key}")

    if not is_observed:
        # Validate output_sample: list of finite floats, not bools.
        sample = entry["output_sample"]
        if not isinstance(sample, list):
            raise RawEvidenceError(f"{label}.output_sample must be a list")
        if len(sample) > OUTPUT_SAMPLE_SIZE:
            raise RawEvidenceError(
                f"{label}.output_sample has {len(sample)} elements, "
                f"max allowed is {OUTPUT_SAMPLE_SIZE}"
            )
        if len(sample) == 0:
            raise RawEvidenceError(
                f"{label}.output_sample is empty (truncated)"
            )
        # Sample size must be min(OUTPUT_SAMPLE_SIZE, elements).
        expected_sample_size = min(OUTPUT_SAMPLE_SIZE, elements)
        if len(sample) != expected_sample_size:
            raise RawEvidenceError(
                f"{label}.output_sample has {len(sample)} elements, "
                f"expected {expected_sample_size}"
            )
        for s_idx, s_val in enumerate(sample):
            if isinstance(s_val, bool):
                raise RawEvidenceError(
                    f"{label}.output_sample[{s_idx}] is a bool, "
                    f"not a float"
                )
            if not _is_finite_number(s_val):
                raise RawEvidenceError(
                    f"{label}.output_sample[{s_idx}] is NaN/Inf or "
                    f"not a number: {s_val}"
                )
    else:
        # Observed: validate transport binding fields.
        selector = entry["transport_selector"]
        if selector not in _VALID_SELECTORS:
            raise RawEvidenceError(
                f"{label}.transport_selector must be one of "
                f"{sorted(_VALID_SELECTORS)}, got '{selector}'"
            )
        for key in (
            "transport_custom", "transport_fallback",
            "transport_unsupported", "transport_unclassified",
        ):
            val = entry[key]
            if not _is_int_not_bool(val) or val < 0:
                raise RawEvidenceError(
                    f"{label}.{key} must be a non-negative int, "
                    f"got {val}"
                )

    # Validate metric fields: must be finite non-negative floats.
    for key in ("mae", "rmse", "max_abs_error"):
        val = entry[key]
        if isinstance(val, bool):
            raise RawEvidenceError(
                f"{label}.{key} is a bool, not a float"
            )
        if not _is_finite_float(val) or val < 0:
            raise RawEvidenceError(
                f"{label}.{key} must be a finite non-negative "
                f"float, got {val}"
            )

    # Validate count fields: non-negative ints, not bools.
    for key in ("mismatch_count", "outside_tolerance_count"):
        val = entry[key]
        if not _is_int_not_bool(val) or val < 0:
            raise RawEvidenceError(
                f"{label}.{key} must be a non-negative int, "
                f"got {val}"
            )
        # Counts cannot exceed total elements.
        if val > elements:
            raise RawEvidenceError(
                f"{label}.{key} ({val}) exceeds elements ({elements})"
            )


def _recompute_and_verify_iteration(
    entry: dict,
    rank_idx: int,
    it_idx: int,
    rank_val: int,
    iterations: int,
    elements: int,
    ranks: int,
    workload_pattern: str,
    workload_rows: int,
    is_observed: bool = False,
) -> None:
    """Recompute all hashes, metrics, and sample from deterministic inputs.

    This is the core anti-forgery check: since inputs are deterministically
    seeded, the validator can regenerate everything and verify the stored
    values match. Any discrepancy means the artifact is forged or corrupt.

    For observed evidence, the output_hash is execution-derived and is
    NOT compared to the modeled output.  The validator verifies input
    hashes and FP32 truth (which are deterministic), and recomputes
    metrics using the modeled output as a reference — but does not
    treat the modeled output hash as the ground truth for the
    observed output hash.  Transport bindings are checked for presence
    and consistency, not recomputed.
    """
    label = f"per_rank_raw[{rank_idx}].per_iteration[{it_idx}]"

    # Regenerate inputs for this iteration.
    inputs = [
        make_rank_input(
            sequence=it_idx,
            rank=r,
            world_size=ranks,
            rows=workload_rows,
            width=elements,
            pattern=workload_pattern,
        )
        for r in range(ranks)
    ]

    # Recompute TP4 ring reduction (modeled reference).
    outputs = tp4_ring_reduce_all_ranks(inputs)

    # Recompute FP32 ground truth.
    truth_fp32 = fp32_ground_truth(inputs)
    truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)

    # This rank's output and input.
    output = outputs[rank_val]
    input_tensor = inputs[rank_val]

    # Verify input hash — always recomputed from deterministic inputs.
    recomputed_input_hash = _tensor_sha256(input_tensor)
    if recomputed_input_hash != entry["input_hash"]:
        raise RawEvidenceError(
            f"{label}.input_hash tampered: stored "
            f"'{entry['input_hash'][:16]}...' but recomputed "
            f"'{recomputed_input_hash[:16]}...'"
        )

    if not is_observed:
        # Modeled: verify output hash matches recomputed modeled output.
        recomputed_output_hash = _tensor_sha256(output)
        if recomputed_output_hash != entry["output_hash"]:
            raise RawEvidenceError(
                f"{label}.output_hash tampered: stored "
                f"'{entry['output_hash'][:16]}...' but recomputed "
                f"'{recomputed_output_hash[:16]}...'"
            )

        # Verify output sample matches recomputed output.
        recomputed_sample = output.float()[:len(entry["output_sample"])].tolist()
        for s_idx, (stored_val, recomputed_val) in enumerate(
            zip(entry["output_sample"], recomputed_sample)
        ):
            if stored_val != recomputed_val:
                raise RawEvidenceError(
                    f"{label}.output_sample[{s_idx}] tampered: stored "
                    f"{stored_val} but recomputed {recomputed_val}"
                )
    else:
        # Observed: output_hash is execution-derived — do NOT compare
        # to modeled output.  The output_hash presence and format are
        # already validated by _validate_iteration_entry.  We verify
        # transport bindings are present (already checked by key
        # validation).  We do not manufacture expected bindings from
        # the modeled computation and return them without comparison.
        pass

    # Verify FP32 truth hash — always recomputed (deterministic).
    recomputed_truth_hash = _tensor_sha256(truth_fp32)
    if recomputed_truth_hash != entry["fp32_truth_hash"]:
        raise RawEvidenceError(
            f"{label}.fp32_truth_hash tampered: stored "
            f"'{entry['fp32_truth_hash'][:16]}...' but recomputed "
            f"'{recomputed_truth_hash[:16]}...'"
        )

    # Recompute metrics using modeled output as reference.
    # For observed evidence, these metrics compare the observed output
    # (whose hash we cannot recompute) against FP32 truth — but since
    # we don't have the observed output tensor, we skip metric
    # verification for observed artifacts.  The metrics in observed
    # artifacts are informational, not anti-forgery checks.
    if not is_observed:
        err = (output.float() - truth_fp32).abs()
        recomputed_mae = float(err.mean().item())
        recomputed_rmse = float((err.square().mean()).sqrt().item())
        recomputed_max = float(err.max().item())
        recomputed_mismatch = int(
            torch.count_nonzero(
                output.view(torch.int16) != truth_bf16.view(torch.int16)
            ).item()
        )
        recomputed_outside = int(
            torch.count_nonzero(err > TOLERANCE_THRESHOLD_BF16).item()
        )

        # Verify metrics match (with small float tolerance).
        if abs(recomputed_mae - entry["mae"]) > 1e-12:
            raise RawEvidenceError(
                f"{label}.mae tampered: stored {entry['mae']} but "
                f"recomputed {recomputed_mae}"
            )
        if abs(recomputed_rmse - entry["rmse"]) > 1e-12:
            raise RawEvidenceError(
                f"{label}.rmse tampered: stored {entry['rmse']} but "
                f"recomputed {recomputed_rmse}"
            )
        if abs(recomputed_max - entry["max_abs_error"]) > 1e-12:
            raise RawEvidenceError(
                f"{label}.max_abs_error tampered: stored "
                f"{entry['max_abs_error']} but recomputed {recomputed_max}"
            )
        if recomputed_mismatch != entry["mismatch_count"]:
            raise RawEvidenceError(
                f"{label}.mismatch_count tampered: stored "
                f"{entry['mismatch_count']} but recomputed "
                f"{recomputed_mismatch}"
            )
        if recomputed_outside != entry["outside_tolerance_count"]:
            raise RawEvidenceError(
                f"{label}.outside_tolerance_count tampered: stored "
                f"{entry['outside_tolerance_count']} but recomputed "
                f"{recomputed_outside}"
            )
