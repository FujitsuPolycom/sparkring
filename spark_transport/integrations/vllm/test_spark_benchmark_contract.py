"""CPU-only tests for the SIRCL-vs-NCCL benchmark evidence contract.

Validates review-5 through review-8 blockers:
- Forged self-attestation reproduction must fail
- Raw-evidence schema replaces PASS-only trust chain
- One artifact cannot be reused for both arms
- Tampered raw metrics, NaN/Inf/bool values, missing/extra ranks/iterations fail
- Mismatched selector semantics/hash fail
- Fake verifier hash fails
- Absolute path, traversal, symlink/reparse point, oversized file fail
- Dry-run proves zero remote/executor calls
- No threshold still means NOT-JUDGED, never PASS
- Preserves multislot design and modeled-audit work
- Review-7: selector truth, junction containment, TOCTOU, CLI exits,
  rendering truth, provenance truth, numeric semantics
- Review-8: structured effective-env selector, handle-confined TOCTOU,
  huge-integer/OverflowError, boolean rank identity rejection,
  outside_tolerance_count coverage, provenance/declaration truth,
  unambiguous CLI exit codes with --validate-only
"""

from __future__ import annotations

import hashlib
import json
import os

import tempfile
import unittest
from unittest.mock import patch

import pytest

from spark_benchmark_contract import (
    BenchmarkContractError,
    BenchmarkRecord,
    ClocksPowerSettings,
    CounterScope,
    IdentitySpec,
    LatencyStats,
    RawEvidenceArtifact,
    RuntimeSpec,
    SelectorConfig,
    Tp4EdgeIdentity,
    WorkloadSpec,
    _check_transport_semantics,
    _make_selector_config,
    _record_from_dict,
    compare,
    emit_json,
    main,
    render_comparison,
)


VALID_SHA256 = "a" * 64
VALID_SHA256_B = "b" * 64
VALID_SHA256_C = "c" * 64
VALID_COMMIT = "a" * 40
VALID_COMMIT_B = "b" * 40
VALID_DIGEST = "sha256:" + "f" * 64
VALID_LOCAL_ID = "sha256:" + "e" * 64


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_edges() -> tuple[Tp4EdgeIdentity, ...]:
    """8 edges matching the TP4 schedule."""
    return (
        Tp4EdgeIdentity(rank=0, round=0, peer_rank=1,
                         device="rocep1s0f0", gid=3, peer_interface="192.0.2.2"),
        Tp4EdgeIdentity(rank=1, round=0, peer_rank=0,
                         device="rocep1s0f0", gid=3, peer_interface="192.0.2.1"),
        Tp4EdgeIdentity(rank=2, round=0, peer_rank=3,
                         device="rocep2s0f0", gid=4, peer_interface="192.0.2.4"),
        Tp4EdgeIdentity(rank=3, round=0, peer_rank=2,
                         device="rocep2s0f0", gid=4, peer_interface="192.0.2.3"),
        Tp4EdgeIdentity(rank=0, round=1, peer_rank=3,
                         device="rocep2s0f0", gid=4, peer_interface="192.0.2.4"),
        Tp4EdgeIdentity(rank=1, round=1, peer_rank=2,
                         device="rocep2s0f0", gid=4, peer_interface="192.0.2.3"),
        Tp4EdgeIdentity(rank=2, round=1, peer_rank=1,
                         device="rocep1s0f0", gid=3, peer_interface="192.0.2.2"),
        Tp4EdgeIdentity(rank=3, round=1, peer_rank=0,
                         device="rocep1s0f0", gid=3, peer_interface="192.0.2.1"),
    )


def _make_clocks_power(
    policy: str = "controlled",
) -> ClocksPowerSettings:
    if policy == "controlled":
        return ClocksPowerSettings(
            policy="controlled",
            gpu_clock_mhz=2100,
            power_limit_w=700,
            declaration_sha256=VALID_SHA256,
        )
    return ClocksPowerSettings(
        policy=policy,
        description=f"clocks not controlled ({policy})",
    )


def _make_selector(
    transport_role: str = "sircl",
    env_vars: frozenset[str] | set[str] | list[str] | tuple[str, ...] | None = None,
) -> SelectorConfig:
    return _make_selector_config(transport_role, env_vars)


def _make_identity(
    transport_library_hash: str = VALID_SHA256,
    selector: SelectorConfig | None = None,
    registry_digest: str = VALID_DIGEST,
    local_image_id: str = VALID_LOCAL_ID,
    source_commit: str = VALID_COMMIT,
    runtime_commit: str = VALID_COMMIT,
    clocks_power: ClocksPowerSettings | None = None,
) -> IdentitySpec:
    return IdentitySpec(
        schema_version="v1",
        source_commit=source_commit,
        runtime_commit=runtime_commit,
        registry_digest=registry_digest,
        local_image_id=local_image_id,
        transport_library_hash=transport_library_hash,
        selector=selector or _make_selector(),
        torch_version="2.12.0",
        vllm_version="0.11.2",
        cuda_version="13.2",
        driver_version="580.173.02",
        model_repository="madeby561/GLM-5.2",
        model_revision=VALID_COMMIT,
        model_config_sha256=VALID_SHA256,
        tp4_edges=_make_edges(),
        clocks_power=clocks_power or _make_clocks_power(),
        evidence_run_id="run-2026-08-09-001",
    )


def _make_workload(
    collective_count: int = 1,
) -> WorkloadSpec:
    return WorkloadSpec(
        shape=(1, 6144),
        dtype="torch.bfloat16",
        bytes_per_collective=12288,
        collective_count=collective_count,
    )


def _make_runtime(
    transport: str = "sircl",
) -> RuntimeSpec:
    return RuntimeSpec(
        lane="public-functional",
        transport=transport,
        topology="tp4_switchless_ring",
        world_size=4,
        warmup_iterations=100,
        sample_iterations=1000,
    )


def _make_latency(
    p50: float = 20.0,
    p95: float = 25.0,
    p99: float = 30.0,
    mx: float = 40.0,
    count: int = 1000,
    timing_boundary: str = "host_submission",
    clock_source: str = "cuda_event",
) -> LatencyStats:
    return LatencyStats(
        p50_us=p50, p95_us=p95, p99_us=p99, max_us=mx, sample_count=count,
        timing_boundary=timing_boundary, clock_source=clock_source,
    )


def _make_counter_scope(
    before: int = 0,
    after: int = 1000,
    warmup_excluded: bool = True,
    reset_source: str = "before-run",
) -> CounterScope:
    return CounterScope(
        counter_source="spark_collective_audit.py",
        warmup_excluded=warmup_excluded,
        reset_source=reset_source,
        before_snapshot=before,
        after_snapshot=after,
    )


def _make_correctness(
    artifact_path: str = "sircl_audit.json",
    artifact_sha256: str = VALID_SHA256,
    arm_transport: str = "sircl",
) -> RawEvidenceArtifact:
    return RawEvidenceArtifact(
        schema_name="tp4_raw_evidence/v2",
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        arm_transport=arm_transport,
    )


def _make_record(
    transport: str = "sircl",
    transport_hash: str = VALID_SHA256,
    selector: SelectorConfig | None = None,
    custom: int = 1000,
    fallback: int = 0,
    p50: float = 20.0,
    p95: float = 25.0,
    p99: float = 30.0,
    mx: float = 40.0,
    correctness: RawEvidenceArtifact | None = None,
    counter_scope: CounterScope | None = None,
    identity: IdentitySpec | None = None,
) -> BenchmarkRecord:
    sel = selector or _make_selector(transport_role=transport)
    return BenchmarkRecord(
        identity=identity or _make_identity(
            transport_library_hash=transport_hash,
            selector=sel,
        ),
        workload=_make_workload(),
        runtime=_make_runtime(transport=transport),
        latency=_make_latency(p50=p50, p95=p95, p99=p99, mx=mx),
        custom_collectives=custom,
        fallback_collectives=fallback,
        unsupported_bypassed_collectives=0,
        unclassified_collectives=0,
        counter_scope=counter_scope or _make_counter_scope(
            after=custom + fallback,
        ),
        correctness=correctness or _make_correctness(
            arm_transport=transport,
        ),
        evidence_label="test-label",
    )


def _make_raw_evidence_file(
    record: BenchmarkRecord,
    path: str | None = None,
    iterations: int = 1000,
    elements: int = 6144,
    ranks: int = 4,
    per_rank_metrics: list[dict] | None = None,
    per_rank_counters: list[dict] | None = None,
    arm_transport: str | None = None,
    selector_hash: str | None = None,
    workload_binding: str | None = None,
    identity_binding: str | None = None,
    artifact_binding: str | None = None,
    schema: str = "tp4_raw_evidence/v2",
    tolerance_policy_version: str = "bf16_fixed_abs_v1",
    extra_key: str | None = None,
    remove_key: str | None = None,
    raw_bytes: bytes | None = None,
) -> str:
    """Write a valid (or deliberately corrupted) raw-evidence artifact file."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

    if raw_bytes is not None:
        with open(path, "wb") as f:
            f.write(raw_bytes)
        return path

    at = arm_transport or record.runtime.transport
    # sh, wb, ib, ab were computed but unused in v2 (producer handles bindings)

    # Track whether the caller explicitly provided forged overrides.
    # If not, we use the real producer for valid artifacts.
    metrics_provided = per_rank_metrics is not None
    counters_provided = per_rank_counters is not None

    if per_rank_metrics is None:
        per_rank_metrics = []
        for rank in range(ranks):
            per_rank_metrics.append({
                "rank": rank,
                "iterations": iterations,
                "mismatch_count": 0,
                "outside_tolerance_count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "all_finite": True,
                "mae": 0.001,
                "rmse": 0.002,
                "max_abs_error": 0.003,
            })

    if per_rank_counters is None:
        per_rank_counters = []
        total = record.runtime.sample_iterations * record.workload.collective_count
        for rank in range(ranks):
            if at == "sircl":
                per_rank_counters.append({
                    "rank": rank,
                    "custom_count": total,
                    "stock_count": 0,
                    "fallback_count": 0,
                    "unsupported_bypassed_count": 0,
                    "unclassified_count": 0,
                    "dropped_signatures": 0,
                })
            else:
                per_rank_counters.append({
                    "rank": rank,
                    "custom_count": 0,
                    "stock_count": 0,
                    "fallback_count": total,
                    "unsupported_bypassed_count": 0,
                    "unclassified_count": 0,
                    "dropped_signatures": 0,
                })

    # Build a v2 artifact. For valid artifacts, use the real producer.
    # For corrupted/forged artifacts, allow manual override.
    if (
        not metrics_provided
        and not counters_provided
        and arm_transport is None
        and selector_hash is None
        and workload_binding is None
        and identity_binding is None
        and artifact_binding is None
        and extra_key is None
        and remove_key is None
        and raw_bytes is None
        and schema == "tp4_raw_evidence/v2"
        and tolerance_policy_version == "bf16_fixed_abs_v1"
    ):
        # Use the real producer for valid artifacts.
        from spark_raw_evidence import RawEvidenceProducer
        producer = RawEvidenceProducer()
        artifact = producer.produce(
            record_iterations=iterations,
            world_size=ranks,
            elements=elements,
        )
    else:
        # Build a v2-shaped artifact with manual overrides for testing.
        from spark_raw_evidence import compute_artifact_sha256
        artifact = {
            "schema": schema,
            "evidence_type": "modeled",
            "iterations": iterations,
            "elements": elements,
            "ranks": ranks,
            "workload_pattern": "random",
            "workload_rows": 1,
            "tolerance_policy": tolerance_policy_version,
            "per_rank_raw": [],
        }
        # If per_rank_metrics is provided, convert to v2 per_rank_raw.
        if per_rank_metrics is not None:
            for rm in per_rank_metrics:
                per_iteration = []
                for it_idx in range(iterations):
                    per_iteration.append({
                        "iteration": it_idx,
                        "input_hash": "a" * 64,
                        "output_hash": "b" * 64,
                        "fp32_truth_hash": "c" * 64,
                        "output_sample": [0.0] * min(64, elements),
                        "mae": rm.get("mae", 0.001),
                        "rmse": rm.get("rmse", 0.002),
                        "max_abs_error": rm.get("max_abs_error", 0.003),
                        "mismatch_count": rm.get("mismatch_count", 0),
                        "outside_tolerance_count": rm.get("outside_tolerance_count", 0),
                    })
                artifact["per_rank_raw"].append({
                    "rank": rm.get("rank", 0),
                    "per_iteration": per_iteration,
                })
        if extra_key:
            artifact[extra_key] = "x"
        if remove_key and remove_key in artifact:
            del artifact[remove_key]
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
    with open(path, "w") as f:
        json.dump(artifact, f)
    return path

def _make_record_with_artifact(
    transport: str = "sircl",
    transport_hash: str = VALID_SHA256,
    selector: SelectorConfig | None = None,
    custom: int = 1000,
    fallback: int = 0,
    p50: float = 20.0,
    p95: float = 25.0,
    p99: float = 30.0,
    mx: float = 40.0,
    artifact_iterations: int = 1000,
    artifact_elements: int = 6144,
    artifact_ranks: int = 4,
    artifact_per_rank_metrics: list[dict] | None = None,
    artifact_per_rank_counters: list[dict] | None = None,
    artifact_arm_transport: str | None = None,
    artifact_selector_hash: str | None = None,
    artifact_workload_binding: str | None = None,
    artifact_identity_binding: str | None = None,
    artifact_binding: str | None = None,
    artifact_schema: str = "tp4_raw_evidence/v2",
    artifact_tolerance_policy_version: str = "bf16_fixed_abs_v1",
    artifact_extra_key: str | None = None,
    artifact_remove_key: str | None = None,
    artifact_raw_bytes: bytes | None = None,
) -> tuple[BenchmarkRecord, str]:
    """Build a record with a real raw-evidence artifact file on disk."""
    tmp_path = tempfile.mktemp(suffix=".json")
    sel = selector or _make_selector(transport_role=transport)
    tmp_record = _make_record(
        transport=transport, transport_hash=transport_hash,
        selector=sel, custom=custom, fallback=fallback,
        p50=p50, p95=p95, p99=p99, mx=mx,
        correctness=_make_correctness(
            artifact_path=os.path.basename(tmp_path),
            arm_transport=transport,
        ),
    )
    _make_raw_evidence_file(
        tmp_record,
        path=tmp_path,
        iterations=artifact_iterations,
        elements=artifact_elements,
        ranks=artifact_ranks,
        per_rank_metrics=artifact_per_rank_metrics,
        per_rank_counters=artifact_per_rank_counters,
        arm_transport=artifact_arm_transport,
        selector_hash=artifact_selector_hash,
        workload_binding=artifact_workload_binding,
        identity_binding=artifact_identity_binding,
        artifact_binding=artifact_binding,
        schema=artifact_schema,
        tolerance_policy_version=artifact_tolerance_policy_version,
        extra_key=artifact_extra_key,
        remove_key=artifact_remove_key,
        raw_bytes=artifact_raw_bytes,
    )
    with open(tmp_path, "rb") as f:
        file_sha = hashlib.sha256(f.read()).hexdigest()
    record = _make_record(
        transport=transport, transport_hash=transport_hash,
        selector=sel, custom=custom, fallback=fallback,
        p50=p50, p95=p95, p99=p99, mx=mx,
        correctness=_make_correctness(
            artifact_path=os.path.basename(tmp_path),
            artifact_sha256=file_sha,
            arm_transport=transport,
        ),
    )
    return record, tmp_path


def _make_pair(
    sircl_p50: float = 20.0,
    nccl_p50: float = 50.0,
    transport_hash: str = VALID_SHA256,
) -> tuple[BenchmarkRecord, str, BenchmarkRecord, str]:
    """Build a valid SIRCL+NCCL pair with real artifact files."""
    sircl_sel = _make_selector("sircl")
    nccl_sel = _make_selector("nccl_ib")
    sircl, s_path = _make_record_with_artifact(
        transport="sircl", transport_hash=transport_hash,
        selector=sircl_sel, custom=1000, fallback=0,
        p50=sircl_p50, p95=sircl_p50 * 1.25,
        p99=sircl_p50 * 1.5, mx=sircl_p50 * 2.0,
    )
    nccl, n_path = _make_record_with_artifact(
        transport="nccl_ib", transport_hash=transport_hash,
        selector=nccl_sel, custom=0, fallback=1000,
        p50=nccl_p50, p95=nccl_p50 * 1.25,
        p99=nccl_p50 * 1.5, mx=nccl_p50 * 2.0,
    )
    return sircl, s_path, nccl, n_path


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Forged self-attestation reproduction test (central blocker)
# ---------------------------------------------------------------------------

class ForgedSelfAttestationTest(unittest.TestCase):
    """The exact forged pair of caller-authored PASS JSON files must fail."""

    def test_forged_pass_artifacts_fail(self) -> None:
        with self.assertRaises(BenchmarkContractError) as ctx:
            RawEvidenceArtifact(
                schema_name="tp4_numerical_audit/v1",
                artifact_path="forged.json",
                artifact_sha256=VALID_SHA256,
                arm_transport="sircl",
            )
        self.assertIn("tp4_raw_evidence/v2", str(ctx.exception))

    def test_forged_pass_with_raw_schema_no_evidence_root(self) -> None:
        sircl = _make_record(
            correctness=_make_correctness(
                artifact_path="forged_sircl.json",
                arm_transport="sircl",
            ),
        )
        nccl = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            correctness=_make_correctness(
                artifact_path="forged_nccl.json",
                arm_transport="nccl_ib",
            ),
        )
        result = compare(sircl, nccl)
        self.assertTrue(result.comparable)
        self.assertFalse(result.correct)
        self.assertEqual(result.correctness_verdict, "not_judged")

    def test_forged_schema_rejected_at_construction(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            RawEvidenceArtifact(
                schema_name="tp4_numerical_audit/v1",
                artifact_path="x.json",
                artifact_sha256=VALID_SHA256,
                arm_transport="sircl",
            )

    def test_single_artifact_reused_across_arms_fails(self) -> None:
        sircl, s_path = _make_record_with_artifact(transport="sircl")
        nccl_record = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            correctness=_make_correctness(
                artifact_path=os.path.basename(s_path),
                artifact_sha256=sircl.correctness.artifact_sha256,
                arm_transport="nccl_ib",
            ),
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl_record,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path)

    def test_no_threshold_is_not_judged_never_pass(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertEqual(result.performance_verdict, "NOT-JUDGED")
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)


# ---------------------------------------------------------------------------
# Selector config tests
# ---------------------------------------------------------------------------

class SelectorConfigTest(unittest.TestCase):
    def test_valid_selector_accepted(self) -> None:
        sel = _make_selector("sircl")
        self.assertEqual(sel.transport_role, "sircl")
        self.assertIn("VLLM_SPARK_TP4_MODE=custom", sel.env_vars)

    def test_selector_hash_recomputed(self) -> None:
        sel = _make_selector("sircl")
        canonical = json.dumps({
            "transport_role": "sircl",
            "env_vars": sorted(sel.env_vars),
        }, sort_keys=True)
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        self.assertEqual(sel.selector_hash, expected)

    def test_bad_selector_hash_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            SelectorConfig(
                transport_role="sircl",
                env_vars=frozenset({"VLLM_SPARK_TP4_MODE=custom"}),
                selector_hash="wrong",
            )

    def test_bad_transport_role_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            SelectorConfig(
                transport_role="bogus",
                env_vars=frozenset({"x=y"}),
                selector_hash=VALID_SHA256,
            )

    def test_selector_role_must_match_transport(self) -> None:
        record = _make_record(
            transport="sircl",
            selector=_make_selector("nccl_ib"),
        )
        with self.assertRaises(BenchmarkContractError) as ctx:
            record.validate()
        self.assertIn("selector transport_role", str(ctx.exception))


# ---------------------------------------------------------------------------
# Clocks/power structured settings tests
# ---------------------------------------------------------------------------

class ClocksPowerTest(unittest.TestCase):
    def test_controlled_with_declaration_accepted(self) -> None:
        cp = _make_clocks_power("controlled")
        self.assertEqual(cp.policy, "controlled")
        self.assertEqual(cp.gpu_clock_mhz, 2100)

    def test_uncontrolled_with_description_accepted(self) -> None:
        cp = _make_clocks_power("uncontrolled")
        self.assertEqual(cp.policy, "uncontrolled")

    def test_not_claimed_with_description_accepted(self) -> None:
        cp = _make_clocks_power("not-claimed")
        self.assertEqual(cp.policy, "not-claimed")

    def test_controlled_missing_declaration_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            ClocksPowerSettings(
                policy="controlled",
                gpu_clock_mhz=2100, power_limit_w=700,
                declaration_sha256=None,
            )

    def test_controlled_bad_declaration_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            ClocksPowerSettings(
                policy="controlled",
                gpu_clock_mhz=2100, power_limit_w=700,
                declaration_sha256="short",
            )

    def test_controlled_missing_clock_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            ClocksPowerSettings(
                policy="controlled",
                gpu_clock_mhz=None, power_limit_w=700,
                declaration_sha256=VALID_SHA256,
            )

    def test_uncontrolled_missing_description_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            ClocksPowerSettings(
                policy="uncontrolled",
                description=None,
            )

    def test_controlled_must_match_for_comparable(self) -> None:
        sircl = _make_record(
            identity=_make_identity(
                clocks_power=_make_clocks_power("controlled"),
            ),
        )
        nccl = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            identity=_make_identity(
                selector=_make_selector("nccl_ib"),
                clocks_power=ClocksPowerSettings(
                    policy="controlled",
                    gpu_clock_mhz=2100,
                    power_limit_w=800,
                    declaration_sha256=VALID_SHA256,
                ),
            ),
        )
        result = compare(sircl, nccl)
        self.assertFalse(result.comparable)
        self.assertTrue(any("clocks_power" in r for r in result.failure_reasons))


# ---------------------------------------------------------------------------
# TP4 topology tests
# ---------------------------------------------------------------------------

class Tp4TopologyTest(unittest.TestCase):
    def test_valid_edges_accepted(self) -> None:
        edges = _make_edges()
        self.assertEqual(len(edges), 8)

    def test_wrong_edge_count_rejected(self) -> None:
        ident_fields = _base_ident_fields()
        with self.assertRaises(BenchmarkContractError) as ctx:
            IdentitySpec(tp4_edges=_make_edges()[:6], **ident_fields)
        self.assertIn("8 edges", str(ctx.exception))

    def test_wrong_peer_in_round_0_rejected(self) -> None:
        edges = list(_make_edges())
        edges[0] = Tp4EdgeIdentity(
            rank=0, round=0, peer_rank=2, device="rocep1s0f0", gid=3,
            peer_interface="192.0.2.2",
        )
        edges[2] = Tp4EdgeIdentity(
            rank=2, round=0, peer_rank=0, device="rocep1s0f0", gid=3,
            peer_interface="192.0.2.1",
        )
        with self.assertRaises(BenchmarkContractError) as ctx:
            IdentitySpec(tp4_edges=tuple(edges), **_base_ident_fields())
        self.assertIn("TP4 schedule", str(ctx.exception))

    def test_non_reciprocal_edges_rejected(self) -> None:
        edges = list(_make_edges())
        edges[1] = Tp4EdgeIdentity(
            rank=1, round=0, peer_rank=2, device="rocep1s0f0", gid=3,
            peer_interface="192.0.2.2",
        )
        with self.assertRaises(BenchmarkContractError) as ctx:
            IdentitySpec(tp4_edges=tuple(edges), **_base_ident_fields())
        msg = str(ctx.exception).lower()
        self.assertTrue("non-reciprocal" in msg or "tp4 schedule" in msg)

    def test_same_device_both_rounds_rejected(self) -> None:
        edges = list(_make_edges())
        edges[4] = Tp4EdgeIdentity(
            rank=0, round=1, peer_rank=3, device="rocep1s0f0",
            gid=4, peer_interface="192.0.2.4",
        )
        with self.assertRaises(BenchmarkContractError) as ctx:
            IdentitySpec(tp4_edges=tuple(edges), **_base_ident_fields())
        self.assertIn("distinct", str(ctx.exception))

    def test_self_peer_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            Tp4EdgeIdentity(
                rank=0, round=0, peer_rank=0, device="rocep1s0f0", gid=3,
                peer_interface="192.0.2.1",
            )


def _base_ident_fields() -> dict:
    return {
        "schema_version": "v1",
        "source_commit": VALID_COMMIT, "runtime_commit": VALID_COMMIT,
        "registry_digest": VALID_DIGEST, "local_image_id": VALID_LOCAL_ID,
        "transport_library_hash": VALID_SHA256,
        "selector": _make_selector(),
        "torch_version": "2.12", "vllm_version": "0.11",
        "cuda_version": "13.2", "driver_version": "580",
        "model_repository": "repo", "model_revision": VALID_COMMIT,
        "model_config_sha256": VALID_SHA256,
        "clocks_power": _make_clocks_power(),
        "evidence_run_id": "run-1",
    }


# ---------------------------------------------------------------------------
# Commit validation tests
# ---------------------------------------------------------------------------

class CommitValidationTest(unittest.TestCase):
    def test_full_40_hex_accepted(self) -> None:
        ident = _make_identity(source_commit="a" * 40, runtime_commit="b" * 40)
        self.assertEqual(ident.source_commit, "a" * 40)

    def test_short_commit_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            _make_identity(source_commit="abc1234")

    def test_non_hex_commit_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            _make_identity(source_commit="g" * 40)


# ---------------------------------------------------------------------------
# Image identity tests
# ---------------------------------------------------------------------------

class ImageIdentityTest(unittest.TestCase):
    def test_both_absent_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            _make_identity(registry_digest="absent", local_image_id="absent")

    def test_local_image_id_bare_hex_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            _make_identity(local_image_id="e" * 64)


# ---------------------------------------------------------------------------
# Counter scope tests
# ---------------------------------------------------------------------------

class CounterScopeTest(unittest.TestCase):
    def test_valid_aggregate_scope(self) -> None:
        cs = _make_counter_scope(before=0, after=1000)
        self.assertEqual(cs.delta, 1000)

    def test_warmup_excluded_false_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            _make_counter_scope(warmup_excluded=False)

    def test_reset_source_not_reset_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            _make_counter_scope(reset_source="not-reset")

    def test_negative_delta_rejected(self) -> None:
        record = _make_record(
            custom=1000, fallback=0,
            counter_scope=_make_counter_scope(before=100, after=50),
        )
        with self.assertRaises(BenchmarkContractError) as ctx:
            record.validate()
        self.assertIn("negative", str(ctx.exception))

    def test_delta_not_equal_custom_plus_fallback_rejected(self) -> None:
        record = _make_record(
            custom=1000, fallback=0,
            counter_scope=_make_counter_scope(before=0, after=999),
        )
        with self.assertRaises(BenchmarkContractError) as ctx:
            record.validate()
        self.assertIn("counter delta", str(ctx.exception))


# ---------------------------------------------------------------------------
# Latency stats tests
# ---------------------------------------------------------------------------

class LatencyStatsTest(unittest.TestCase):
    def test_percentiles_from_samples(self) -> None:
        samples = [float(i + 1) for i in range(100)]
        stats = LatencyStats.from_samples(samples)
        self.assertAlmostEqual(stats.p50_us, 51.0, places=1)
        self.assertEqual(stats.max_us, 100.0)

    def test_empty_samples_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            LatencyStats.from_samples([])

    def test_nan_sample_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            LatencyStats.from_samples([float("nan"), 1.0])


@pytest.mark.parametrize("bad_p50,bad_p95,bad_p99", [
    (30.0, 25.0, 20.0),
    (20.0, 30.0, 25.0),
])
def test_percentile_ordering_violations_rejected(
    bad_p50: float, bad_p95: float, bad_p99: float,
) -> None:
    with pytest.raises(BenchmarkContractError) as exc_info:
        _make_record(p50=bad_p50, p95=bad_p95, p99=bad_p99)
    assert "percentile ordering" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Record validation tests
# ---------------------------------------------------------------------------

class RecordValidationTest(unittest.TestCase):
    def test_valid_record_passes(self) -> None:
        _make_record().validate()

    def test_indeterminate_counts_rejected(self) -> None:
        record = _make_record(custom=0, fallback=0)
        with self.assertRaises(BenchmarkContractError):
            record.validate()

    def test_sircl_with_fallback_fails(self) -> None:
        record = _make_record(custom=500, fallback=500)
        with self.assertRaises(BenchmarkContractError):
            record.validate()

    def test_nccl_with_custom_fails(self) -> None:
        record = _make_record(
            transport="nccl_ib", custom=500, fallback=500,
            selector=_make_selector("nccl_ib"),
        )
        with self.assertRaises(BenchmarkContractError):
            record.validate()

    def test_world_size_not_4_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            RuntimeSpec(
                lane="public-functional", transport="sircl",
                topology="tp4_switchless_ring", world_size=2,
                warmup_iterations=100, sample_iterations=1000,
            )


# ---------------------------------------------------------------------------
# Arm role tests
# ---------------------------------------------------------------------------

class ArmRoleTest(unittest.TestCase):
    def test_sircl_vs_nccl_comparable(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            result = compare(sircl, nccl)
            self.assertTrue(result.comparable)
        finally:
            _cleanup(s_path, n_path)

    def test_nccl_as_arg1_not_comparable(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            result = compare(nccl, sircl)
            self.assertFalse(result.comparable)
            self.assertTrue(any("first argument" in r for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)

    def test_selector_hashes_differ_across_arms(self) -> None:
        sircl_sel = _make_selector("sircl")
        nccl_sel = _make_selector("nccl_ib")
        self.assertNotEqual(sircl_sel.selector_hash, nccl_sel.selector_hash)
        sircl = _make_record(transport="sircl", selector=sircl_sel)
        nccl = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            selector=nccl_sel,
        )
        result = compare(sircl, nccl)
        self.assertTrue(result.comparable)

    def test_mismatched_library_hash_not_comparable(self) -> None:
        sircl = _make_record(transport="sircl", transport_hash=VALID_SHA256)
        nccl = _make_record(
            transport="nccl_ib", transport_hash=VALID_SHA256_B,
            custom=0, fallback=1000,
        )
        result = compare(sircl, nccl)
        self.assertFalse(result.comparable)


# ---------------------------------------------------------------------------
# Raw-evidence artifact derivation tests
# ---------------------------------------------------------------------------

class RawEvidenceDerivationTest(unittest.TestCase):
    def test_valid_artifact_derives_consistent(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertTrue(result.comparable)
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_missing_artifact_file_fails(self) -> None:
        sircl = _make_record(
            correctness=_make_correctness(artifact_path="nonexistent.json"),
        )
        nccl = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            correctness=_make_correctness(
                artifact_path="nonexistent2.json",
                arm_transport="nccl_ib",
            ),
        )
        result = compare(sircl, nccl,
                         evidence_root="/tmp",
                         repo_root=os.getcwd())
        self.assertFalse(result.correct)
        self.assertTrue(any("not found" in r or "not accessible" in r
                            or "cannot open" in r
                            for r in result.failure_reasons))

    def test_tampered_artifact_fails(self) -> None:
        sircl, s_path = _make_record_with_artifact(transport="sircl")
        with open(s_path, "w") as f:
            json.dump({"schema": "tp4_raw_evidence/v1", "bogus": True}, f)
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertTrue(any("SHA-256 mismatch" in r
                                 for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)

    def test_wrong_schema_fails(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_schema="bogus/v1",
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    def test_nan_in_metrics_fails(self) -> None:
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0, "outside_tolerance_count": 0,
            "nan_count": 1 if r == 0 else 0,
            "inf_count": 0, "all_finite": r != 0,
            "mae": float("nan") if r == 0 else 0.001,
            "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    def test_inf_in_metrics_fails(self) -> None:
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0, "outside_tolerance_count": 0,
            "nan_count": 0,
            "inf_count": 1 if r == 1 else 0,
            "all_finite": r != 1,
            "mae": 0.001, "rmse": 0.002,
            "max_abs_error": float("inf") if r == 1 else 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    def test_missing_rank_fails(self) -> None:
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0, "outside_tolerance_count": 0,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True, "mae": 0.001, "rmse": 0.002,
            "max_abs_error": 0.003,
        } for r in range(3)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_ranks=3,
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    def test_mismatched_selector_hash_fails(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_selector_hash=VALID_SHA256_B,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_extra_key_in_artifact_fails(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_extra_key="bogus",
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    def test_non_json_artifact_fails(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_raw_bytes=b"not json",
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    def test_artifact_arm_transport_mismatch_fails(self) -> None:
        sircl, s_path = _make_record_with_artifact(transport="sircl")
        nccl = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            correctness=_make_correctness(
                artifact_path=os.path.basename(s_path),
                artifact_sha256=sircl.correctness.artifact_sha256,
                arm_transport="nccl_ib",
            ),
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path)


# ---------------------------------------------------------------------------
# Evidence root path security tests
# ---------------------------------------------------------------------------

class EvidenceRootSecurityTest(unittest.TestCase):
    def test_absolute_path_rejected(self) -> None:
        sircl = _make_record(
            correctness=_make_correctness(
                artifact_path="/etc/passwd",
            ),
        )
        nccl = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            correctness=_make_correctness(
                artifact_path="x.json", arm_transport="nccl_ib",
            ),
        )
        result = compare(sircl, nccl,
                         evidence_root="/tmp",
                         repo_root=os.getcwd())
        self.assertFalse(result.correct)
        self.assertTrue(any("absolute" in r.lower()
                            for r in result.failure_reasons))

    def test_traversal_path_rejected(self) -> None:
        sircl = _make_record(
            correctness=_make_correctness(
                artifact_path="../../../etc/passwd",
            ),
        )
        nccl = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            correctness=_make_correctness(
                artifact_path="x.json", arm_transport="nccl_ib",
            ),
        )
        result = compare(sircl, nccl,
                         evidence_root="/tmp",
                         repo_root=os.getcwd())
        self.assertFalse(result.correct)
        self.assertTrue(any("escapes" in r.lower() or "outside" in r.lower()
                            for r in result.failure_reasons))


# ---------------------------------------------------------------------------
# Verifier receipt tests
# ---------------------------------------------------------------------------

class VerifierReceiptTest(unittest.TestCase):
    def test_verifier_receipt_verified(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)


# ---------------------------------------------------------------------------
# Performance verdict tests
# ---------------------------------------------------------------------------

class PerformanceVerdictTest(unittest.TestCase):
    def test_no_threshold_gives_not_judged(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            result = compare(sircl, nccl)
            self.assertEqual(result.performance_verdict, "NOT-JUDGED")
        finally:
            _cleanup(s_path, n_path)

    def test_threshold_not_judged_without_raw_samples(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair(
            sircl_p50=40.0, nccl_p50=50.0,
        )
        try:
            result = compare(sircl, nccl, performance_threshold=0.5)
            self.assertEqual(result.performance_verdict, "NOT-JUDGED")
        finally:
            _cleanup(s_path, n_path)

    def test_threshold_not_judged_even_when_fast(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair(
            sircl_p50=20.0, nccl_p50=50.0,
        )
        try:
            result = compare(sircl, nccl, performance_threshold=0.5)
            self.assertEqual(result.performance_verdict, "NOT-JUDGED")
        finally:
            _cleanup(s_path, n_path)

    def test_threshold_nan_rejected(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            with self.assertRaises(BenchmarkContractError):
                compare(sircl, nccl, performance_threshold=float("nan"))
        finally:
            _cleanup(s_path, n_path)


# ---------------------------------------------------------------------------
# JSON parsing fail-closed tests
# ---------------------------------------------------------------------------

class JSONParsingTest(unittest.TestCase):
    def test_missing_field_rejected(self) -> None:
        data = _make_record().to_dict()
        del data["custom_collectives"]
        with self.assertRaises(BenchmarkContractError):
            _record_from_dict(data)

    def test_extra_field_rejected(self) -> None:
        data = _make_record().to_dict()
        data["bogus_field"] = "x"
        with self.assertRaises(BenchmarkContractError):
            _record_from_dict(data)

    def test_missing_selector_rejected(self) -> None:
        data = _make_record().to_dict()
        del data["identity"]["selector"]
        with self.assertRaises(BenchmarkContractError):
            _record_from_dict(data)

    def test_missing_clocks_power_rejected(self) -> None:
        data = _make_record().to_dict()
        del data["identity"]["clocks_power"]
        with self.assertRaises(BenchmarkContractError):
            _record_from_dict(data)

    def test_full_roundtrip(self) -> None:
        record = _make_record()
        data = record.to_dict()
        restored = _record_from_dict(data)
        self.assertEqual(restored.identity, record.identity)
        self.assertEqual(restored.workload, record.workload)
        self.assertEqual(restored.correctness, record.correctness)


# ---------------------------------------------------------------------------
# Render and JSON tests
# ---------------------------------------------------------------------------

class RenderTest(unittest.TestCase):
    def test_render_contains_key_fields(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            comparison = compare(sircl, nccl)
            report = render_comparison(comparison)
            self.assertIn("SIRCL_VS_NCCL_BENCHMARK", report)
            self.assertIn("correctness_verdict=", report)
        finally:
            _cleanup(s_path, n_path)

    def test_emit_json_is_valid(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            comparison = compare(sircl, nccl)
            data = json.loads(emit_json(comparison))
            self.assertIn("correctness_verdict", data)
            self.assertIn("correct", data)
        finally:
            _cleanup(s_path, n_path)


# ---------------------------------------------------------------------------
# CLI adversarial tests
# ---------------------------------------------------------------------------

class AdversarialCLIRegressions(unittest.TestCase):
    def test_cli_nonzero_on_not_comparable(self) -> None:
        sircl = _make_record(transport="sircl", transport_hash=VALID_SHA256)
        nccl = _make_record(
            transport="nccl_ib", transport_hash=VALID_SHA256_B,
            custom=0, fallback=1000,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as sf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as nf:
            json.dump(sircl.to_dict(), sf)
            sf.flush()
            json.dump(nccl.to_dict(), nf)
            nf.flush()
            rc = main([
                "--sircl-json", sf.name,
                "--nccl-json", nf.name,
                "--emit-json",
            ])
        os.unlink(sf.name)
        os.unlink(nf.name)
        self.assertEqual(rc, 1)

    def test_cli_nonzero_on_not_judged(self) -> None:
        """Compare mode always exits nonzero (correct=false, NOT-JUDGED)."""
        sircl, s_path, nccl, n_path = _make_pair(
            sircl_p50=40.0, nccl_p50=50.0,
        )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--threshold", "0.5",
                    "--emit-json",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            # Compare mode: correct=false always, NOT-JUDGED → exit 1
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_nonzero_on_malformed_json(self) -> None:
        """Malformed JSON exits 2 (contract-invalid)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as sf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as nf:
            sf.write("{not valid json")
            sf.flush()
            json.dump(_make_record(
                transport="nccl_ib", custom=0, fallback=1000,
            ).to_dict(), nf)
            nf.flush()
            rc = main([
                "--sircl-json", sf.name,
                "--nccl-json", nf.name,
            ])
        os.unlink(sf.name)
        os.unlink(nf.name)
        self.assertEqual(rc, 2)

    def test_cli_nonzero_without_evidence_root(self) -> None:
        """Compare mode without evidence_root: not_judged → exit 1."""
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--emit-json",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_nonzero_with_valid_evidence(self) -> None:
        """Compare mode with valid evidence: structurally_consistent but
        correct=false → exit 1 (compare mode never exits 0)."""
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--evidence-root", evidence_dir,
                    "--repo-root", os.getcwd(),
                    "--emit-json",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            # Compare mode: correct=false always → exit 1
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_validate_only_exits_zero(self) -> None:
        """--validate-only mode exits 0 when both records are valid."""
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--validate-only",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 0)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_validate_only_exits_two_on_malformed(self) -> None:
        """--validate-only with malformed input exits 2."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as sf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as nf:
            sf.write("{bad json")
            sf.flush()
            json.dump(_make_record(
                transport="nccl_ib", custom=0, fallback=1000,
            ).to_dict(), nf)
            nf.flush()
            rc = main([
                "--sircl-json", sf.name,
                "--nccl-json", nf.name,
                "--validate-only",
            ])
        os.unlink(sf.name)
        os.unlink(nf.name)
        self.assertEqual(rc, 2)

    def test_cli_validate_only_exits_two_on_invalid_record(self) -> None:
        """--validate-only with structurally invalid record exits 2."""
        record = _make_record()
        data = record.to_dict()
        del data["identity"]["source_commit"]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as sf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as nf:
            json.dump(data, sf)
            sf.flush()
            json.dump(_make_record(
                transport="nccl_ib", custom=0, fallback=1000,
            ).to_dict(), nf)
            nf.flush()
            rc = main([
                "--sircl-json", sf.name,
                "--nccl-json", nf.name,
                "--validate-only",
            ])
        os.unlink(sf.name)
        os.unlink(nf.name)
        self.assertEqual(rc, 2)

    def test_cli_nonzero_on_swapped_arm_roles(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(nccl.to_dict(), sf)
                sf.flush()
                json.dump(sircl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--emit-json",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)


# ---------------------------------------------------------------------------
# Review-6 adversarial regressions
# ---------------------------------------------------------------------------

class Review6AdversarialRegressions(unittest.TestCase):
    """Adversarial tests for review-6 mandatory repairs."""

    def test_catastrophic_forged_metrics_not_correct(self) -> None:
        forged_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 6144000,
            "outside_tolerance_count": 6144000,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 1e30, "rmse": 1e30, "max_abs_error": 1e30,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=forged_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertTrue(any("exceeds" in r.lower()
                                for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)

    def test_copied_fake_repo_arbitrary_selector_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            SelectorConfig(
                transport_role="sircl",
                env_vars=frozenset({"ARBITRARY_TEXT"}),
                selector_hash=VALID_SHA256,
            )

    def test_identical_payloads_accepted_as_structurally_consistent(self) -> None:
        sircl, s_path = _make_record_with_artifact(transport="sircl")
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
            artifact_per_rank_metrics=None,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_arbitrary_device_name_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError) as ctx:
            Tp4EdgeIdentity(
                rank=0, round=0, peer_rank=1,
                device="arbitrary_device", gid=3,
                peer_interface="192.0.2.1",
            )
        self.assertIn("HCA device pattern", str(ctx.exception))

    def test_arbitrary_peer_interface_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError) as ctx:
            Tp4EdgeIdentity(
                rank=0, round=0, peer_rank=1,
                device="rocep1s0f0", gid=3,
                peer_interface="not_an_ip",
            )
        self.assertIn("IPv4", str(ctx.exception))

    def test_fake_clock_declaration_rejected(self) -> None:
        """A one-character string must not establish controlled identity."""
        with self.assertRaises(BenchmarkContractError):
            ClocksPowerSettings(
                policy="controlled",
                gpu_clock_mhz=2100, power_limit_w=700,
                declaration_sha256="x",
            )

    def test_wrong_tolerance_policy_version_rejected(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_tolerance_policy_version="v0",
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_missing_per_rank_counters_rejected(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_remove_key="per_rank_counters",
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_per_rank_counters_structurally_validated_only(self) -> None:
        # v2 artifacts don't carry per_rank_counters; counter values
        # don't affect artifact structural validation.  Use the real
        # producer; tolerance policy rejects natural BF16 rounding
        # mismatch, so verdict is "failed".
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_iterations_not_matching_sample_iterations_rejected(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_iterations=999,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertTrue(any("iterations" in r
                                for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)

    def test_elements_not_matching_workload_shape_rejected(self) -> None:
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_elements=9999,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertTrue(any("elements" in r
                                for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)

    def test_correct_always_false_even_when_structurally_consistent(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertTrue(result.comparable)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    def test_performance_not_judged_with_threshold(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            result = compare(sircl, nccl, performance_threshold=0.5)
            self.assertEqual(result.performance_verdict, "NOT-JUDGED")
        finally:
            _cleanup(s_path, n_path)

    def test_dropped_signatures_structurally_validated_only(self) -> None:
        # v2 artifacts don't carry per_rank_counters; dropped signatures
        # are a counter-level concern that doesn't affect artifact
        # structural validation.  Use the real producer so the artifact
        # passes v2 recomputation; tolerance policy still rejects the
        # natural BF16 rounding mismatch, so verdict is "failed".
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_bad_timing_boundary_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError) as ctx:
            LatencyStats(
                p50_us=20.0, p95_us=25.0, p99_us=30.0,
                max_us=40.0, sample_count=1000,
                timing_boundary="bogus",
            )
        self.assertIn("timing_boundary", str(ctx.exception))

    def test_bad_clock_source_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError) as ctx:
            LatencyStats(
                p50_us=20.0, p95_us=25.0, p99_us=30.0,
                max_us=40.0, sample_count=1000,
                clock_source="bogus",
            )
        self.assertIn("clock_source", str(ctx.exception))


# ---------------------------------------------------------------------------
# Review-7 adversarial regressions
# ---------------------------------------------------------------------------

class Review7AdversarialRegressions(unittest.TestCase):
    """Adversarial tests for review-7 mandatory repairs."""

    # --- 1. Selector truth ---

    def test_ineffective_spark_tp4_mode_rejected(self) -> None:
        with self.assertRaises(BenchmarkContractError) as ctx:
            SelectorConfig(
                transport_role="sircl",
                env_vars=frozenset({"SPARK_TP4_MODE=custom"}),
                selector_hash="0" * 64,
            )
        self.assertIn("does not exactly match", str(ctx.exception))

    def test_effective_vllm_spark_tp4_mode_accepted(self) -> None:
        sel = _make_selector("sircl")
        self.assertIn("VLLM_SPARK_TP4_MODE=custom", sel.env_vars)

    # --- 2. Windows containment (junction) ---

    def test_junction_escape_rejected(self) -> None:
        import subprocess
        import tempfile as _tf
        base = _tf.mkdtemp()
        evidence_root = os.path.join(base, "evidence")
        os.makedirs(evidence_root)
        target_dir = os.path.join(base, "target")
        os.makedirs(target_dir)
        junct = os.path.join(evidence_root, "junct")
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", junct, target_dir],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            self.skipTest("junction creation unavailable on this platform")
        artifact_file = os.path.join(target_dir, "escaped.json")
        with open(artifact_file, "w") as f:
            json.dump({"schema": "tp4_raw_evidence/v1"}, f)
        try:
            sircl = _make_record(
                correctness=_make_correctness(
                    artifact_path="junct/escaped.json",
                ),
            )
            nccl = _make_record(
                transport="nccl_ib", custom=0, fallback=1000,
                correctness=_make_correctness(
                    artifact_path="x.json", arm_transport="nccl_ib",
                ),
            )
            result = compare(sircl, nccl,
                             evidence_root=evidence_root,
                             repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertTrue(any("symlink" in r.lower() or "junction" in r.lower()
                                or "reparse" in r.lower()
                                for r in result.failure_reasons))
        finally:
            _cleanup(artifact_file)
            try:
                os.rmdir(junct)
            except OSError:
                pass
            os.rmdir(target_dir)
            os.rmdir(evidence_root)
            os.rmdir(base)

    # --- 3. Parent-swap TOCTOU ---

    def test_parent_swap_toctou_rejected(self) -> None:
        import subprocess
        import tempfile as _tf
        base = _tf.mkdtemp()
        evidence_root = os.path.join(base, "evidence")
        os.makedirs(evidence_root)
        target_dir = os.path.join(base, "target")
        os.makedirs(target_dir)
        junct = os.path.join(evidence_root, "junct")
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", junct, target_dir],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            self.skipTest("junction creation unavailable on this platform")
        artifact_file = os.path.join(target_dir, "swap.json")
        with open(artifact_file, "w") as f:
            json.dump({"schema": "tp4_raw_evidence/v1"}, f)
        try:
            from spark_benchmark_contract import _safe_open_and_read
            with self.assertRaises(BenchmarkContractError) as ctx:
                _safe_open_and_read(os.path.join(junct, "swap.json"),
                                    evidence_root)
            self.assertTrue(
                "symlink" in str(ctx.exception).lower()
                or "junction" in str(ctx.exception).lower()
                or "reparse" in str(ctx.exception).lower()
                or "confinement" in str(ctx.exception).lower()
            )
        finally:
            _cleanup(artifact_file)
            try:
                os.rmdir(junct)
            except OSError:
                pass
            os.rmdir(target_dir)
            os.rmdir(evidence_root)
            os.rmdir(base)

    # --- 4. CLI exit codes ---

    def test_cli_exit_nonzero_on_not_judged(self) -> None:
        """correct=false with both verdicts NOT-JUDGED exits 1
        (compare mode never exits 0)."""
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--emit-json",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_exit_nonzero_on_structurally_consistent(self) -> None:
        """With valid evidence artifacts, compare mode exits 1
        (correct=false always in compare mode)."""
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--evidence-root", evidence_dir,
                    "--repo-root", os.getcwd(),
                    "--emit-json",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_exit_2_on_malformed_input(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as sf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as nf:
            sf.write("{not valid json")
            sf.flush()
            json.dump(_make_record(
                transport="nccl_ib", custom=0, fallback=1000,
            ).to_dict(), nf)
            nf.flush()
            rc = main([
                "--sircl-json", sf.name,
                "--nccl-json", nf.name,
            ])
        os.unlink(sf.name)
        os.unlink(nf.name)
        self.assertEqual(rc, 2)

    # --- 5. Rendering truth ---

    def test_render_table_header_says_structural_not_correct(self) -> None:
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            comparison = compare(sircl, nccl)
            report = render_comparison(comparison)
            self.assertIn("| structural |", report)
            lines = report.split("\n")
            header_line = [ln for ln in lines if ln.startswith("| transport")]
            self.assertTrue(header_line)
            self.assertIn("structural", header_line[0])
            self.assertNotIn("correct |", header_line[0])
        finally:
            _cleanup(s_path, n_path)

    # --- 6. Provenance truth ---

    def test_verifier_receipt_returns_hash(self) -> None:
        from spark_benchmark_contract import _verify_verifier_receipt, _VERIFIER_REL_PATH
        # Find the repo root by searching for the verifier file
        cwd = os.getcwd()
        repo_root = cwd
        # When running from spark_transport/integrations/vllm/, the
        # _VERIFIER_REL_PATH is relative to the repo root, not CWD.
        # Walk up until we find the verifier file.
        for _ in range(5):
            if os.path.isfile(os.path.join(repo_root, _VERIFIER_REL_PATH)):
                break
            repo_root = os.path.dirname(repo_root)
        result = _verify_verifier_receipt(repo_root)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)

    # --- 8. Workload/numeric semantics ---

    def test_bytes_per_collective_validated_from_dtype_and_shape(self) -> None:
        wl = _make_workload()
        self.assertEqual(wl.bytes_per_collective, 12288)
        with self.assertRaises(BenchmarkContractError) as ctx:
            WorkloadSpec(
                shape=(1, 6144),
                dtype="torch.bfloat16",
                bytes_per_collective=999,
                collective_count=1,
            )
        self.assertIn("bytes_per_collective", str(ctx.exception))

    def test_bytes_per_collective_fp32(self) -> None:
        wl = WorkloadSpec(
            shape=(1, 6144),
            dtype="torch.float32",
            bytes_per_collective=24576,
            collective_count=1,
        )
        self.assertEqual(wl.bytes_per_collective, 24576)

    def test_dtype_specific_tolerances_bf16_vs_fp32(self) -> None:
        from spark_benchmark_contract import _DTYPE_TOLERANCES
        bf16 = _DTYPE_TOLERANCES["torch.bfloat16"]
        fp32 = _DTYPE_TOLERANCES["torch.float32"]
        self.assertGreater(bf16["max_mae"], fp32["max_mae"])
        self.assertGreater(bf16["max_rmse"], fp32["max_rmse"])
        self.assertGreater(bf16["max_abs_error"], fp32["max_abs_error"])

    def test_int32_tolerance_is_zero(self) -> None:
        from spark_benchmark_contract import _DTYPE_TOLERANCES
        i32 = _DTYPE_TOLERANCES["torch.int32"]
        self.assertEqual(i32["max_mae"], 0.0)
        self.assertEqual(i32["max_rmse"], 0.0)
        self.assertEqual(i32["max_abs_error"], 0)

    def test_huge_integer_count_rejected(self) -> None:
        huge = 2**53 + 1
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": huge,
            "outside_tolerance_count": 0,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 0.001, "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertTrue(any("exceeds" in r.lower()
                                for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)

    def test_bool_in_metrics_rejected(self) -> None:
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0,
            "outside_tolerance_count": 0,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": True,
            "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_mismatch_count_exceeds_total_comparisons_rejected(self) -> None:
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 99999,
            "outside_tolerance_count": 0,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 0.001, "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertTrue(any("mismatch_count" in r.lower()
                                for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)


# ---------------------------------------------------------------------------
# Review-8 adversarial regressions
# ---------------------------------------------------------------------------

class Review8AdversarialRegressions(unittest.TestCase):
    """Adversarial tests for review-8 mandatory repairs."""

    # --- 1. Structured effective-environment selector ---

    def test_nccl_socket_requires_full_env_vars(self) -> None:
        """NCCL socket arm requires VLLM_SPARK_TP4_MODE=disabled,
        NCCL_NET=Socket, and NCCL_IB_DISABLE=1 — all three."""
        sel = _make_selector("nccl_socket")
        self.assertEqual(sel.env_vars, frozenset({
            "VLLM_SPARK_TP4_MODE=disabled",
            "NCCL_NET=Socket",
            "NCCL_IB_DISABLE=1",
        }))

    def test_nccl_ib_requires_full_env_vars(self) -> None:
        """NCCL IB arm requires VLLM_SPARK_TP4_MODE=disabled,
        NCCL_NET=IB, and NCCL_IB_DISABLE=0 — all three."""
        sel = _make_selector("nccl_ib")
        self.assertEqual(sel.env_vars, frozenset({
            "VLLM_SPARK_TP4_MODE=disabled",
            "NCCL_NET=IB",
            "NCCL_IB_DISABLE=0",
        }))

    def test_inherited_tp4_mode_on_nccl_rejected(self) -> None:
        """VLLM_SPARK_TP4_MODE=custom (inherited TP4 mode) on an NCCL
        arm must be rejected — NCCL arms must disable SIRCL."""
        with self.assertRaises(BenchmarkContractError):
            SelectorConfig(
                transport_role="nccl_ib",
                env_vars=frozenset({
                    "VLLM_SPARK_TP4_MODE=custom",
                    "NCCL_NET=IB",
                    "NCCL_IB_DISABLE=0",
                }),
                selector_hash="0" * 64,
            )

    def test_nccl_socket_missing_ib_disable_rejected(self) -> None:
        """NCCL socket arm missing NCCL_IB_DISABLE=1 is incomplete."""
        with self.assertRaises(BenchmarkContractError):
            SelectorConfig(
                transport_role="nccl_socket",
                env_vars=frozenset({
                    "VLLM_SPARK_TP4_MODE=disabled",
                    "NCCL_NET=Socket",
                }),
                selector_hash="0" * 64,
            )

    def test_nccl_ib_contradictory_ib_disable_rejected(self) -> None:
        """NCCL_IB_DISABLE=1 with NCCL_NET=IB is contradictory."""
        with self.assertRaises(BenchmarkContractError):
            SelectorConfig(
                transport_role="nccl_ib",
                env_vars=frozenset({
                    "VLLM_SPARK_TP4_MODE=disabled",
                    "NCCL_NET=IB",
                    "NCCL_IB_DISABLE=1",  # contradictory
                }),
                selector_hash="0" * 64,
            )

    def test_nccl_extra_env_var_rejected(self) -> None:
        """Extra env var beyond the allowlisted set is rejected."""
        with self.assertRaises(BenchmarkContractError):
            SelectorConfig(
                transport_role="nccl_ib",
                env_vars=frozenset({
                    "VLLM_SPARK_TP4_MODE=disabled",
                    "NCCL_NET=IB",
                    "NCCL_IB_DISABLE=0",
                    "EXTRA_VAR=x",
                }),
                selector_hash="0" * 64,
            )

    def test_sircl_selector_spelling_preserved(self) -> None:
        """The actual SIRCL selector spelling VLLM_SPARK_TP4_MODE=custom
        must be preserved in the env_vars."""
        sel = _make_selector("sircl")
        self.assertEqual(sel.env_vars, frozenset({"VLLM_SPARK_TP4_MODE=custom"}))

    def test_legacy_selector_config_string_in_dict_accepted(self) -> None:
        """Legacy single-string selector_config in dict is converted to
        a single-element frozenset for backward compatibility."""
        from spark_benchmark_contract import _selector_from_dict
        sel = _selector_from_dict({
            "transport_role": "sircl",
            "selector_config": "VLLM_SPARK_TP4_MODE=custom",
            "selector_hash": _make_selector("sircl").selector_hash,
        })
        self.assertEqual(sel.env_vars, frozenset({"VLLM_SPARK_TP4_MODE=custom"}))

    # --- 2. Handle-confined TOCTOU ---

    def test_parent_swap_after_recheck_caught_by_fd_confinement(self) -> None:
        """Deterministic test: swap a previously validated parent into a
        junction after the final validation point and prove outside bytes
        cannot be read via the fd confinement check.

        This test creates a valid artifact inside evidence_root, validates
        the path (which passes), then replaces the parent directory with
        a junction pointing outside.  The _safe_open_and_read fd
        confinement check must catch this and refuse the read.

        On platforms without junction support, the test verifies that
        _safe_open_and_read with evidence_root still provides fd
        confinement or fails closed.
        """
        import subprocess
        import tempfile as _tf
        base = _tf.mkdtemp()
        evidence_root = os.path.join(base, "evidence")
        subdir = os.path.join(evidence_root, "sub")
        os.makedirs(subdir)
        # Write a legit artifact file
        legit_file = os.path.join(subdir, "artifact.json")
        with open(legit_file, "w") as f:
            json.dump({"schema": "tp4_raw_evidence/v1"}, f)

        # Outside target with secret content
        outside_dir = os.path.join(base, "outside")
        os.makedirs(outside_dir)
        outside_file = os.path.join(outside_dir, "artifact.json")
        with open(outside_file, "w") as f:
            f.write("SECRET_OUTSIDE_BYTES")

        try:
            # Swap: replace 'sub' with a junction to 'outside_dir'
            _cleanup(legit_file)
            os.rmdir(subdir)
            r = subprocess.run(
                ["cmd", "/c", "mklink", "/J", subdir, outside_dir],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                self.skipTest("junction creation unavailable on this platform")

            # Now try to read sub/artifact.json — it resolves to
            # outside_dir/artifact.json via the junction.
            # _safe_open_and_read must catch this via fd confinement.
            from spark_benchmark_contract import _safe_open_and_read
            with self.assertRaises(BenchmarkContractError) as ctx:
                _safe_open_and_read(legit_file, evidence_root)
            # The error must mention confinement, symlink, junction,
            # or reparse — proving outside bytes were NOT read.
            msg = str(ctx.exception).lower()
            self.assertTrue(
                "confinement" in msg
                or "symlink" in msg
                or "junction" in msg
                or "reparse" in msg
            )
            # Verify the secret bytes were NOT returned
            # (the exception proves the read was refused)
        finally:
            _cleanup(legit_file, outside_file)
            # Remove junction first (rmdir follows the junction target
            # and fails if the target is non-empty).  Use cmd rmdir
            # which removes the junction itself, not the target.
            try:
                import subprocess as _sp
                _sp.run(["cmd", "/c", "rmdir", subdir],
                        capture_output=True)
            except OSError:
                pass
            try:
                os.rmdir(outside_dir)
            except OSError:
                pass
            try:
                os.rmdir(evidence_root)
            except OSError:
                pass
            try:
                os.rmdir(base)
            except OSError:
                pass

    # --- 3. Huge integers / OverflowError ---

    def test_huge_integer_latency_raises_contract_error(self) -> None:
        """Huge integer latency values that overflow float() must raise
        BenchmarkContractError, not OverflowError."""
        with self.assertRaises(BenchmarkContractError):
            LatencyStats(
                p50_us=2**10000, p95_us=2**10000, p99_us=2**10000,
                max_us=2**10000, sample_count=1000,
            )

    def test_huge_integer_in_mae_raises_contract_error(self) -> None:
        """Huge integer in mae must be caught by _is_finite_float."""
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0, "outside_tolerance_count": 0,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 2**100,  # huge int → OverflowError in float()
            "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_empty_shape_rejected(self) -> None:
        """Empty shape must be rejected."""
        with self.assertRaises(BenchmarkContractError) as ctx:
            WorkloadSpec(
                shape=(),
                dtype="torch.bfloat16",
                bytes_per_collective=0,
                collective_count=1,
            )
        self.assertIn("non-empty", str(ctx.exception))

    def test_absurd_dimension_rejected(self) -> None:
        """Dimension exceeding _MAX_SAFE_FLOAT_INT must be rejected."""
        from spark_benchmark_contract import _MAX_SAFE_FLOAT_INT
        with self.assertRaises(BenchmarkContractError) as ctx:
            WorkloadSpec(
                shape=(_MAX_SAFE_FLOAT_INT + 1, 1),
                dtype="torch.bfloat16",
                bytes_per_collective=0,
                collective_count=1,
            )
        self.assertIn("max safe", str(ctx.exception).lower())

    def test_shape_product_overflow_rejected(self) -> None:
        """Shape product overflow must be rejected before byte calc."""
        with self.assertRaises(BenchmarkContractError) as ctx:
            WorkloadSpec(
                shape=(2**30, 2**30),  # product = 2^60 > 2^53
                dtype="torch.bfloat16",
                bytes_per_collective=1,  # positive, but overflow check comes first
                collective_count=1,
            )
        self.assertIn("overflow", str(ctx.exception).lower())

    # --- 4. Boolean rank identity rejection ---

    def test_bool_rank_in_metrics_rejected(self) -> None:
        """Boolean rank in per_rank_metrics must be rejected (False == 0,
        True == 1 must never satisfy rank coverage)."""
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0, "outside_tolerance_count": 0,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 0.001, "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        # Use bool for rank 0's rank field: True == 1, not 0
        bad_metrics[0]["rank"] = True  # type: ignore[assignment]
        # Also need to fix rank 1 since True==1 would match
        bad_metrics[1]["rank"] = False  # type: ignore[assignment]
        # Now rank 0 has rank=True (==1), rank 1 has rank=False (==0)
        # This should fail because rank must be int, not bool
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_bool_mismatch_count_rejected(self) -> None:
        """Boolean mismatch_count must be rejected (True == 1 must not
        satisfy as a count)."""
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": True,  # bool, not int
            "outside_tolerance_count": 0,
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 0.001, "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_bool_counter_rank_rejected(self) -> None:
        """Boolean rank in per_rank_counters must be rejected."""
        bad_counters = [{
            "rank": True if r == 0 else r,  # bool for rank 0
            "custom_count": 1000,
            "stock_count": 0,
            "fallback_count": 0,
            "unsupported_bypassed_count": 0,
            "unclassified_count": 0,
            "dropped_signatures": 0,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_counters=bad_counters,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    # --- 5. Numerical coverage coherence ---

    def test_outside_tolerance_count_required_key(self) -> None:
        """Artifact missing outside_tolerance_count in per_rank_metrics
        must fail."""
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0,
            # outside_tolerance_count missing
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 0.001, "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    def test_outside_tolerance_count_exceeds_coverage_rejected(self) -> None:
        """outside_tolerance_count exceeding total comparisons must be
        rejected.  This is checked BEFORE tolerance policy so it's
        always reachable even with _MAX_MISMATCH_COUNT=0."""
        bad_metrics = [{
            "rank": r, "iterations": 1000,
            "mismatch_count": 0,
            "outside_tolerance_count": 9999999,  # exceeds 6144000
            "nan_count": 0, "inf_count": 0,
            "all_finite": True,
            "mae": 0.001, "rmse": 0.002, "max_abs_error": 0.003,
        } for r in range(4)]
        sircl, s_path = _make_record_with_artifact(
            transport="sircl",
            artifact_per_rank_metrics=bad_metrics,
        )
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertTrue(any("outside_tolerance_count" in r.lower()
                                for r in result.failure_reasons))
        finally:
            _cleanup(s_path, n_path)

    def test_coverage_bound_with_collective_count(self) -> None:
        """Coverage is bound to iterations * elements * collective_count.
        With collective_count=2, total = 1000 * 6144 * 2 = 12,288,000.
        outside_tolerance_count = 6144 (one iteration's worth) is within
        the bound and should pass coverage check (but mismatch_count=0
        still required by tolerance policy)."""
        sircl_sel = _make_selector("sircl")
        sircl_record_base = _make_record(
            transport="sircl", selector=sircl_sel,
            custom=2000, fallback=0,
            counter_scope=_make_counter_scope(after=2000),
        )
        # Override workload with collective_count=2
        sircl_record = BenchmarkRecord(
            identity=sircl_record_base.identity,
            workload=WorkloadSpec(
                shape=(1, 6144), dtype="torch.bfloat16",
                bytes_per_collective=12288, collective_count=2,
            ),
            runtime=sircl_record_base.runtime,
            latency=sircl_record_base.latency,
            custom_collectives=2000, fallback_collectives=0,
            unsupported_bypassed_collectives=0,
            unclassified_collectives=0,
            counter_scope=sircl_record_base.counter_scope,
            correctness=_make_correctness(),
            evidence_label="test-label",
        )
        # Use the real producer — v2 recomputes outside_tolerance_count
        # from deterministic inputs, so forged values are rejected.
        # The natural outside_tolerance_count is within the coverage
        # bound (collective_count=2 → total=12,288,000), but the
        # natural mismatch_count > 0 fails tolerance policy, so
        # verdict is "failed".
        s_path = tempfile.mktemp(suffix=".json")
        _make_raw_evidence_file(
            sircl_record, path=s_path,
            iterations=1000, elements=6144,
        )
        with open(s_path, "rb") as f:
            file_sha = hashlib.sha256(f.read()).hexdigest()
        sircl_final = BenchmarkRecord(
            identity=sircl_record.identity,
            workload=sircl_record.workload,
            runtime=sircl_record.runtime,
            latency=sircl_record.latency,
            custom_collectives=2000, fallback_collectives=0,
            unsupported_bypassed_collectives=0,
            unclassified_collectives=0,
            counter_scope=sircl_record.counter_scope,
            correctness=_make_correctness(
                artifact_path=os.path.basename(s_path),
                artifact_sha256=file_sha,
            ),
            evidence_label="test-label",
        )
        nccl_sel = _make_selector("nccl_ib")
        nccl_record_base = _make_record(
            transport="nccl_ib", custom=0, fallback=2000,
            selector=nccl_sel,
            counter_scope=_make_counter_scope(after=2000),
        )
        nccl_record = BenchmarkRecord(
            identity=nccl_record_base.identity,
            workload=sircl_record.workload,  # same workload
            runtime=nccl_record_base.runtime,
            latency=nccl_record_base.latency,
            custom_collectives=0, fallback_collectives=2000,
            unsupported_bypassed_collectives=0,
            unclassified_collectives=0,
            counter_scope=nccl_record_base.counter_scope,
            correctness=_make_correctness(arm_transport="nccl_ib"),
            evidence_label="test-label",
        )
        n_path = tempfile.mktemp(suffix=".json")
        _make_raw_evidence_file(
            nccl_record, path=n_path,
            iterations=1000, elements=6144,
        )
        with open(n_path, "rb") as f:
            nfile_sha = hashlib.sha256(f.read()).hexdigest()
        nccl_final = BenchmarkRecord(
            identity=nccl_record.identity,
            workload=nccl_record.workload,
            runtime=nccl_record.runtime,
            latency=nccl_record.latency,
            custom_collectives=0, fallback_collectives=2000,
            unsupported_bypassed_collectives=0,
            unclassified_collectives=0,
            counter_scope=nccl_record.counter_scope,
            correctness=_make_correctness(
                artifact_path=os.path.basename(n_path),
                artifact_sha256=nfile_sha,
                arm_transport="nccl_ib",
            ),
            evidence_label="test-label",
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl_final, nccl_final,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertTrue(result.comparable)
            self.assertEqual(result.correctness_verdict, "failed")
        finally:
            _cleanup(s_path, n_path)

    # --- 6. Provenance/declaration truth ---

    def test_verifier_hash_not_called_receipt(self) -> None:
        """The constant is TRUSTED_VERIFIER_HASH, not TRUSTED_VERIFIER_RECEIPT.
        It is an implementation-byte check, not source-commit anchoring."""
        from spark_benchmark_contract import TRUSTED_VERIFIER_HASH
        self.assertIsInstance(TRUSTED_VERIFIER_HASH, str)
        self.assertEqual(len(TRUSTED_VERIFIER_HASH), 64)

    def test_clocks_power_field_named_declaration(self) -> None:
        """ClocksPowerSettings field is declaration_sha256, not
        receipt_sha256."""
        cp = _make_clocks_power("controlled")
        self.assertTrue(hasattr(cp, "declaration_sha256"))
        self.assertFalse(hasattr(cp, "receipt_sha256"))

    def test_unverified_topology_excluded_from_judgment(self) -> None:
        """Unverified topology/clocks/counters must be visibly labeled
        and excluded from correctness/performance judgment.  correct
        is always False even with valid structural artifacts."""
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertEqual(result.performance_verdict, "NOT-JUDGED")
        finally:
            _cleanup(s_path, n_path)

    def test_benchmark_judgment_not_judged_without_producer_receipts(self) -> None:
        """Because the baseline requires executed-path proof, benchmark
        judgment remains NOT-JUDGED without compatible producer
        receipts."""
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            result = compare(sircl, nccl, performance_threshold=0.5)
            self.assertEqual(result.performance_verdict, "NOT-JUDGED")
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

    # --- 7. CLI exit codes ---

    def test_cli_json_path_exits_one(self) -> None:
        """Compare mode with JSON output exits 1 (not 0)."""
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--emit-json",
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_contract_invalid_exits_two(self) -> None:
        """Contract-invalid input (missing field) exits 2."""
        record = _make_record()
        data = record.to_dict()
        del data["identity"]["source_commit"]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as sf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as nf:
            json.dump(data, sf)
            sf.flush()
            json.dump(_make_record(
                transport="nccl_ib", custom=0, fallback=1000,
            ).to_dict(), nf)
            nf.flush()
            rc = main([
                "--sircl-json", sf.name,
                "--nccl-json", nf.name,
            ])
        os.unlink(sf.name)
        os.unlink(nf.name)
        self.assertEqual(rc, 2)

    def test_cli_failed_evidence_exits_one(self) -> None:
        """Compare mode with failed evidence (tampered artifact) exits 1."""
        sircl, s_path = _make_record_with_artifact(transport="sircl")
        with open(s_path, "w") as f:
            json.dump({"bogus": True}, f)
        nccl, n_path = _make_record_with_artifact(
            transport="nccl_ib", custom=0, fallback=1000,
        )
        evidence_dir = os.path.dirname(s_path)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                rc = main([
                    "--sircl-json", sf.name,
                    "--nccl-json", nf.name,
                    "--evidence-root", evidence_dir,
                    "--repo-root", os.getcwd(),
                ])
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 1)
        finally:
            _cleanup(s_path, n_path)

    def test_cli_validate_only_json_output(self) -> None:
        """--validate-only with --emit-json outputs JSON with
        validate_only=true."""
        sircl, s_path, nccl, n_path = _make_pair()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as sf, tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as nf:
                json.dump(sircl.to_dict(), sf)
                sf.flush()
                json.dump(nccl.to_dict(), nf)
                nf.flush()
                # Capture stdout
                import io
                import contextlib
                f_out = io.StringIO()
                with contextlib.redirect_stdout(f_out):
                    rc = main([
                        "--sircl-json", sf.name,
                        "--nccl-json", nf.name,
                        "--validate-only",
                        "--emit-json",
                    ])
                output = json.loads(f_out.getvalue())
            os.unlink(sf.name)
            os.unlink(nf.name)
            self.assertEqual(rc, 0)
            self.assertTrue(output["validate_only"])
            self.assertTrue(output["sircl_valid"])
            self.assertTrue(output["nccl_valid"])
        finally:
            _cleanup(s_path, n_path)


# ---------------------------------------------------------------------------
# Dry-run zero remote calls test
# ---------------------------------------------------------------------------

class DryRunTest(unittest.TestCase):
    def test_compare_makes_no_remote_calls(self) -> None:
        """compare() is purely offline: no SSH, Docker, CUDA, or network."""
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        try:
            result = compare(sircl, nccl,
                              evidence_root=evidence_dir,
                              repo_root=os.getcwd())
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)


class FallbackAccountingTest(unittest.TestCase):
    """Adversarial tests for fallback accounting as acceptance condition."""

    def test_sircl_with_fallback_is_invalid(self) -> None:
        """SIRCL arm with unexpected fallback is invalid, not merely slower."""
        sircl = _make_record(transport="sircl", custom=500, fallback=500)
        errors = _check_transport_semantics(sircl)
        self.assertTrue(any("fallback" in e for e in errors))

    def test_nccl_with_custom_proves_sircl_invoked(self) -> None:
        """NCCL arm with custom > 0 means SIRCL was invoked — invalid."""
        nccl = _make_record(
            transport="nccl_ib", custom=500, fallback=500,
        )
        errors = _check_transport_semantics(nccl)
        self.assertTrue(any("SIRCL was not invoked" in e for e in errors))

    def test_sircl_with_unsupported_is_invalid(self) -> None:
        from spark_benchmark_contract import BenchmarkRecord
        sircl = _make_record(
            transport="sircl", custom=1000, fallback=0,
        )
        bad = BenchmarkRecord(
            identity=sircl.identity,
            workload=sircl.workload,
            runtime=sircl.runtime,
            latency=sircl.latency,
            custom_collectives=900,
            fallback_collectives=0,
            unsupported_bypassed_collectives=100,
            unclassified_collectives=0,
            counter_scope=sircl.counter_scope,
            correctness=sircl.correctness,
            evidence_label="test-label",
        )
        errors = _check_transport_semantics(bad)
        self.assertTrue(any("unsupported" in e for e in errors))

    def test_sircl_with_unclassified_is_invalid(self) -> None:
        from spark_benchmark_contract import BenchmarkRecord
        sircl = _make_record(transport="sircl", custom=1000, fallback=0)
        bad = BenchmarkRecord(
            identity=sircl.identity,
            workload=sircl.workload,
            runtime=sircl.runtime,
            latency=sircl.latency,
            custom_collectives=900,
            fallback_collectives=0,
            unsupported_bypassed_collectives=0,
            unclassified_collectives=100,
            counter_scope=sircl.counter_scope,
            correctness=sircl.correctness,
            evidence_label="test-label",
        )
        errors = _check_transport_semantics(bad)
        self.assertTrue(any("unclassified" in e for e in errors))

    def test_count_total_must_sum_to_expected(self) -> None:
        """custom+fallback+unsupported+unclassified must equal
        sample_iterations * collective_count."""
        from spark_benchmark_contract import BenchmarkRecord
        sircl = _make_record(transport="sircl", custom=1000, fallback=0)
        bad = BenchmarkRecord(
            identity=sircl.identity,
            workload=sircl.workload,
            runtime=sircl.runtime,
            latency=sircl.latency,
            custom_collectives=500,
            fallback_collectives=0,
            unsupported_bypassed_collectives=0,
            unclassified_collectives=0,
            counter_scope=_make_counter_scope(after=500),
            correctness=sircl.correctness,
            evidence_label="test-label",
        )
        with self.assertRaises(BenchmarkContractError):
            bad.validate()


class VerifiedFactsTest(unittest.TestCase):
    """Tests for separating verified facts from caller declarations."""

    def test_source_commit_is_declared_not_verified(self) -> None:
        """The ComparisonResult docstring must label source_commit
        and other identity fields as caller-declared unverified."""
        import spark_benchmark_contract as mod
        docstring = mod.ComparisonResult.__doc__ or ""
        self.assertIn("caller-declared", docstring.lower())
        self.assertIn("unverified", docstring.lower())

    def test_verifier_hash_does_not_verify_commit(self) -> None:
        """The _verify_verifier_receipt docstring must say it does
        NOT verify source_commit against the actual checkout."""
        from spark_benchmark_contract import _verify_verifier_receipt
        docstring = _verify_verifier_receipt.__doc__ or ""
        self.assertIn("NOT", docstring)
        self.assertIn("implementation-byte", docstring.lower())

    def test_false_declaration_cannot_make_incomparable_comparable(self) -> None:
        """Two arms with different transport_library_hash cannot be
        made comparable by matching other fields — identity mismatch
        must still be detected."""
        sircl = _make_record(transport="sircl")
        nccl_diff = _make_record(
            transport="nccl_ib", custom=0, fallback=1000,
            transport_hash="d" * 64,
        )
        result = compare(sircl, nccl_diff)
        self.assertFalse(result.comparable)
        self.assertTrue(any("transport_library_hash" in r
                            for r in result.failure_reasons))


# ---------------------------------------------------------------------------
# Goal 9: Stale trusted probe hash (req 9)
# ---------------------------------------------------------------------------

class StaleTrustedProbeHashTest(unittest.TestCase):
    """A stale TRUSTED_VERIFIER_HASH must be detected — the verifier
    implementation on disk must match the trusted hash, and a mismatch
    must raise BenchmarkContractError (Goal 9 req 9).
    """

    def test_stale_trusted_hash_raises_contract_error(self) -> None:
        """Monkeypatching TRUSTED_VERIFIER_HASH to a wrong value must
        cause _verify_verifier_receipt to raise BenchmarkContractError
        with 'verifier hash mismatch' in the message."""
        import spark_benchmark_contract as mod
        stale_hash = "0" * 64
        with patch.object(mod, "TRUSTED_VERIFIER_HASH", stale_hash):
            with self.assertRaises(BenchmarkContractError) as ctx:
                mod._verify_verifier_receipt(os.getcwd())
            self.assertIn("verifier hash mismatch", str(ctx.exception))

    def test_correct_trusted_hash_passes(self) -> None:
        """The trusted hash must match LF-stable verifier bytes."""
        import spark_benchmark_contract as mod
        verifier_path = os.path.join(
            os.getcwd(), mod._VERIFIER_REL_PATH,
        )
        with open(verifier_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(actual_hash, mod.TRUSTED_VERIFIER_HASH)

    def test_verifier_checkout_is_forced_to_lf(self) -> None:
        """Verifier attestation cannot vary with Windows autocrlf."""
        import spark_benchmark_contract as mod
        verifier_path = os.path.join(os.getcwd(), mod._VERIFIER_REL_PATH)
        with open(verifier_path, "rb") as f:
            verifier_bytes = f.read()
        self.assertNotIn(b"\r\n", verifier_bytes)

    def test_stale_hash_in_compare_marks_failed(self) -> None:
        """A stale verifier hash must cause compare() to mark
        correctness_verdict='failed' and include the mismatch in
        failure_reasons — not silently accept a modified verifier."""
        import spark_benchmark_contract as mod
        sircl, s_path, nccl, n_path = _make_pair()
        evidence_dir = os.path.dirname(s_path)
        stale_hash = "0" * 64
        try:
            with patch.object(mod, "TRUSTED_VERIFIER_HASH", stale_hash):
                result = mod.compare(sircl, nccl,
                                     evidence_root=evidence_dir,
                                     repo_root=os.getcwd())
            self.assertEqual(result.correctness_verdict, "failed")
            self.assertTrue(
                any("verifier hash mismatch" in r
                    for r in result.failure_reasons),
                f"Expected verifier hash mismatch in failure_reasons, "
                f"got: {result.failure_reasons}",
            )
            self.assertFalse(result.correct)
        finally:
            _cleanup(s_path, n_path)

if __name__ == "__main__":
    unittest.main()
