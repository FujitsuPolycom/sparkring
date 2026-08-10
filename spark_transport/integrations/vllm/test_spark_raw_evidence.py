"""CPU-only tests for the raw per-rank/per-iteration evidence producer.

Covers:
- Correct SHA-256 binding of the raw artifact.
- Raw/reduced cross-reference (reduced SHA-256 derived from raw data).
- Missing rank rejected.
- Duplicate iteration rejected.
- Reordered iterations rejected.
- NaN/Inf in output sample rejected.
- Tampered artifact hash rejected.
- Truncated output_sample rejected.
- Bool in metrics rejected.
- Recomputation equality: load artifact, recompute metrics, verify match.
"""

from __future__ import annotations

import copy
import unittest

import torch

from spark_fp32_ground_truth import (
    TOLERANCE_THRESHOLD_BF16,
    fp32_ground_truth,
    fp32_truth_rounded_to_dtype,
    make_rank_input,
    tp4_ring_reduce_all_ranks,
)
from spark_raw_evidence import (
    OUTPUT_SAMPLE_SIZE,
    RAW_EVIDENCE_SCHEMA_V2,
    REDUCED_EVIDENCE_SCHEMA,
    EVIDENCE_TYPE_OBSERVED,
    ObservedEvidenceReceipt,
    RawEvidenceError,
    RawEvidenceProducer,
    compute_artifact_sha256,
    validate_raw_evidence,
)


def _produce_small(
    iterations: int = 2,
    world_size: int = 4,
    elements: int = 128,
) -> dict:
    """Produce a small valid artifact for testing."""
    return RawEvidenceProducer().produce(
        record_iterations=iterations,
        world_size=world_size,
        elements=elements,
    )


def _deep_copy_artifact(artifact: dict) -> dict:
    return copy.deepcopy(artifact)


class TestRawEvidenceSHA256Binding(unittest.TestCase):
    """Verify the artifact SHA-256 binding is correct and self-consistent."""

    def test_artifact_sha256_matches_recompute(self) -> None:
        artifact = _produce_small()
        stored = artifact["artifact_sha256"]
        recomputed = compute_artifact_sha256(artifact)
        self.assertEqual(stored, recomputed)

    def test_artifact_sha256_excludes_self(self) -> None:
        """The hash must not include the artifact_sha256 field itself."""
        artifact = _produce_small()
        artifact_copy = _deep_copy_artifact(artifact)
        # Change the stored hash and verify recompute still gives the
        # original value (proving the field is excluded).
        original = artifact["artifact_sha256"]
        artifact_copy["artifact_sha256"] = "0" * 64
        recomputed = compute_artifact_sha256(artifact_copy)
        self.assertEqual(recomputed, original)

    def test_artifact_sha256_is_64_hex(self) -> None:
        artifact = _produce_small()
        self.assertRegex(
            artifact["artifact_sha256"], r"^[0-9a-f]{64}$",
        )


class TestRawReducedCrossReference(unittest.TestCase):
    """Verify raw and reduced evidence cross-reference via SHA-256."""

    def test_validate_returns_reduced_sha256(self) -> None:
        artifact = _produce_small()
        result = validate_raw_evidence(artifact)
        self.assertIn("reduced_sha256", result)
        self.assertRegex(result["reduced_sha256"], r"^[0-9a-f]{64}$")

    def test_reduced_schema_is_correct(self) -> None:
        artifact = _produce_small()
        result = validate_raw_evidence(artifact)
        self.assertEqual(result["reduced_evidence"]["schema"],
                         REDUCED_EVIDENCE_SCHEMA)

    def test_reduced_evidence_has_per_rank(self) -> None:
        artifact = _produce_small(iterations=3, elements=128)
        result = validate_raw_evidence(artifact)
        per_rank = result["per_rank_reduced"]
        self.assertEqual(len(per_rank), 4)
        for entry in per_rank:
            self.assertIn("rank", entry)
            self.assertIn("mae", entry)
            self.assertIn("rmse", entry)
            self.assertIn("max_abs_error", entry)
            self.assertIn("mismatch_count", entry)
            self.assertIn("outside_tolerance_count", entry)

    def test_reduced_sha256_changes_on_tamper(self) -> None:
        """Tampering with a metric and re-hashing must be REJECTED.

        The recomputing validator catches the tampered metric even when
        the artifact SHA-256 is fixed — it does not trust caller-
        supplied aggregates.
        """
        artifact = _produce_small()
        artifact_tampered = _deep_copy_artifact(artifact)
        # Tamper with a metric value and fix the artifact hash.
        artifact_tampered["per_rank_raw"][0]["per_iteration"][0]["mae"] = 999.0
        artifact_tampered["artifact_sha256"] = compute_artifact_sha256(
            artifact_tampered
        )
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact_tampered)
        self.assertIn("mae tampered", str(ctx.exception))


class TestMissingRankRejected(unittest.TestCase):
    """Missing rank data must be rejected."""

    def test_missing_rank_rejected(self) -> None:
        artifact = _produce_small()
        # Remove the last rank entry.
        artifact["per_rank_raw"].pop()
        # Fix the artifact hash so we test the structural check, not the hash.
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("per_rank_raw must be a list of", str(ctx.exception))

    def test_ranks_mismatch_rejected(self) -> None:
        artifact = _produce_small()
        # Declare ranks=4 but provide only 3.
        artifact["per_rank_raw"].pop()
        artifact["ranks"] = 4
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("per_rank_raw must be a list of", str(ctx.exception))


class TestDuplicateIterationRejected(unittest.TestCase):
    """Duplicate iteration numbers must be rejected."""

    def test_duplicate_iteration_rejected(self) -> None:
        artifact = _produce_small(iterations=2)
        # Duplicate iteration 0 in rank 0 (replace iteration 1 with
        # another iteration 0).
        rank0 = artifact["per_rank_raw"][0]
        rank0["per_iteration"][1] = copy.deepcopy(
            rank0["per_iteration"][0]
        )
        rank0["per_iteration"][1]["iteration"] = 0
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("duplicate iteration", str(ctx.exception))

    def test_duplicate_rank_rejected(self) -> None:
        artifact = _produce_small()
        # Duplicate rank 0 (replace rank 1 with a copy of rank 0).
        artifact["per_rank_raw"][1] = copy.deepcopy(
            artifact["per_rank_raw"][0]
        )
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("duplicate rank", str(ctx.exception))


class TestReorderedIterationsRejected(unittest.TestCase):
    """Reordered iteration indices must be rejected."""

    def test_reordered_iterations_rejected(self) -> None:
        artifact = _produce_small(iterations=3)
        # Swap iterations 0 and 1 in rank 0.
        rank0 = artifact["per_rank_raw"][0]
        rank0["per_iteration"][0], rank0["per_iteration"][1] = (
            rank0["per_iteration"][1], rank0["per_iteration"][0]
        )
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("iteration must be", str(ctx.exception))

    def test_reordered_ranks_rejected(self) -> None:
        artifact = _produce_small()
        # Swap ranks 0 and 1.
        artifact["per_rank_raw"][0], artifact["per_rank_raw"][1] = (
            artifact["per_rank_raw"][1], artifact["per_rank_raw"][0]
        )
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("rank must be", str(ctx.exception))


class TestNaNInfRejected(unittest.TestCase):
    """NaN/Inf in output sample or metrics must be rejected."""

    def test_nan_in_output_sample_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0][
            "output_sample"
        ][0] = float("nan")
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_inf_in_output_sample_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0][
            "output_sample"
        ][0] = float("inf")
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_nan_in_mae_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0]["mae"] = float("nan")
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("finite non-negative", str(ctx.exception))

    def test_inf_in_rmse_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0]["rmse"] = float("inf")
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("finite non-negative", str(ctx.exception))


class TestTamperedHashRejected(unittest.TestCase):
    """Tampered artifact SHA-256 must be rejected."""

    def test_tampered_hash_rejected(self) -> None:
        artifact = _produce_small()
        # Tamper with the stored hash.
        artifact["artifact_sha256"] = "a" * 64
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("tampered", str(ctx.exception))

    def test_tampered_data_rejected(self) -> None:
        """Change data but don't update the hash — must be caught."""
        artifact = _produce_small()
        original_hash = artifact["artifact_sha256"]
        artifact["per_rank_raw"][0]["per_iteration"][0]["mae"] = 42.0
        # Don't update the hash.
        artifact["artifact_sha256"] = original_hash
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("tampered", str(ctx.exception))

    def test_expected_binding_mismatch_rejected(self) -> None:
        artifact = _produce_small()
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact, expected_binding="b" * 64)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_expected_binding_match_accepted(self) -> None:
        artifact = _produce_small()
        result = validate_raw_evidence(
            artifact, expected_binding=artifact["artifact_sha256"],
        )
        self.assertTrue(result["valid"])


class TestTruncatedOutputSampleRejected(unittest.TestCase):
    """Truncated (empty or oversized) output_sample must be rejected."""

    def test_empty_sample_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0][
            "output_sample"
        ] = []
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("truncated", str(ctx.exception))

    def test_oversized_sample_rejected(self) -> None:
        artifact = _produce_small(elements=256)
        entry = artifact["per_rank_raw"][0]["per_iteration"][0]
        entry["output_sample"] = [0.0] * (OUTPUT_SAMPLE_SIZE + 1)
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("max allowed", str(ctx.exception))


class TestBoolInMetricsRejected(unittest.TestCase):
    """Boolean values in metric or count fields must be rejected."""

    def test_bool_in_mae_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0]["mae"] = True
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("bool", str(ctx.exception))

    def test_bool_in_mismatch_count_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0][
            "mismatch_count"
        ] = True
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("non-negative int", str(ctx.exception))

    def test_bool_in_output_sample_rejected(self) -> None:
        artifact = _produce_small()
        artifact["per_rank_raw"][0]["per_iteration"][0][
            "output_sample"
        ][0] = True
        artifact["artifact_sha256"] = compute_artifact_sha256(artifact)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(artifact)
        self.assertIn("bool", str(ctx.exception))


class TestRecomputationEquality(unittest.TestCase):
    """Load artifact, recompute metrics from hashes, verify they match."""

    def test_recomputed_metrics_match_artifact(self) -> None:
        """Produce an artifact, then independently recompute the metrics
        from the same inputs and verify they match the stored values."""
        iterations = 2
        world_size = 4
        elements = 128
        artifact = _produce_small(
            iterations=iterations, elements=elements,
        )

        # Independently recompute metrics for each rank/iteration.
        for rank in range(world_size):
            for it in range(iterations):
                inputs = [
                    make_rank_input(
                        sequence=it, rank=r, world_size=world_size,
                        rows=1, width=elements, pattern="random",
                    )
                    for r in range(world_size)
                ]
                outputs = tp4_ring_reduce_all_ranks(inputs)
                truth_fp32 = fp32_ground_truth(inputs)
                truth_bf16 = fp32_truth_rounded_to_dtype(
                    truth_fp32, torch.bfloat16,
                )
                output = outputs[rank]
                err = (output.float() - truth_fp32).abs()
                exp_mae = float(err.mean().item())
                exp_rmse = float((err.square().mean()).sqrt().item())
                exp_max = float(err.max().item())
                exp_mismatch = int(
                    torch.count_nonzero(
                        output.view(torch.int16) != truth_bf16.view(torch.int16)
                    ).item()
                )
                exp_outside = int(
                    torch.count_nonzero(
                        err > TOLERANCE_THRESHOLD_BF16
                    ).item()
                )

                stored = artifact["per_rank_raw"][rank]["per_iteration"][it]
                self.assertAlmostEqual(stored["mae"], exp_mae, places=6)
                self.assertAlmostEqual(stored["rmse"], exp_rmse, places=6)
                self.assertAlmostEqual(stored["max_abs_error"], exp_max, places=6)
                self.assertEqual(stored["mismatch_count"], exp_mismatch)
                self.assertEqual(
                    stored["outside_tolerance_count"], exp_outside,
                )

    def test_output_sample_matches_recompute(self) -> None:
        """The stored output_sample must match the first N recomputed values."""
        iterations = 1
        world_size = 4
        elements = 128
        artifact = _produce_small(
            iterations=iterations, elements=elements,
        )

        for rank in range(world_size):
            it = 0
            inputs = [
                make_rank_input(
                    sequence=it, rank=r, world_size=world_size,
                    rows=1, width=elements, pattern="random",
                )
                for r in range(world_size)
            ]
            outputs = tp4_ring_reduce_all_ranks(inputs)
            expected_sample = outputs[rank].float()[
                :OUTPUT_SAMPLE_SIZE
            ].tolist()
            stored_sample = artifact["per_rank_raw"][rank][
                "per_iteration"
            ][it]["output_sample"]
            self.assertEqual(len(stored_sample), len(expected_sample))
            for s_idx, (s, e) in enumerate(
                zip(stored_sample, expected_sample)
            ):
                self.assertAlmostEqual(s, e, places=6,
                                       msg=f"rank={rank} sample[{s_idx}]")

    def test_validate_then_reproduce_consistent(self) -> None:
        """Validate an artifact, produce a second one with the same params,
        and verify the hashes match (determinism)."""
        a1 = _produce_small(iterations=3, elements=256)
        a2 = _produce_small(iterations=3, elements=256)
        self.assertEqual(a1["artifact_sha256"], a2["artifact_sha256"])
        r1 = validate_raw_evidence(a1)
        r2 = validate_raw_evidence(a2)
        self.assertEqual(r1["reduced_sha256"], r2["reduced_sha256"])

    def test_hash_fields_correct_length(self) -> None:
        """All hash fields must be 64-char hex strings."""
        artifact = _produce_small()
        for rank_entry in artifact["per_rank_raw"]:
            for it_entry in rank_entry["per_iteration"]:
                for key in ("input_hash", "output_hash", "fp32_truth_hash"):
                    self.assertRegex(it_entry[key], r"^[0-9a-f]{64}$")

    def test_fp32_truth_hash_same_across_ranks(self) -> None:
        """FP32 truth is the same for all ranks in one iteration."""
        artifact = _produce_small(iterations=2)
        for it in range(2):
            hashes = set()
            for rank in range(4):
                h = artifact["per_rank_raw"][rank][
                    "per_iteration"
                ][it]["fp32_truth_hash"]
                hashes.add(h)
            self.assertEqual(len(hashes), 1,
                             f"FP32 truth hash differs across ranks at it={it}")

    def test_output_hash_same_across_ranks(self) -> None:
        """In TP4 ring reduce, all ranks get the same output (all-reduce)."""
        artifact = _produce_small(iterations=2)
        for it in range(2):
            hashes = set()
            for rank in range(4):
                h = artifact["per_rank_raw"][rank][
                    "per_iteration"
                ][it]["output_hash"]
                hashes.add(h)
            self.assertEqual(len(hashes), 1,
                             f"Output hash differs across ranks at it={it}")

    def test_input_hash_differs_across_ranks(self) -> None:
        """Each rank has a different input."""
        artifact = _produce_small(iterations=1)
        hashes = set()
        for rank in range(4):
            h = artifact["per_rank_raw"][rank][
                "per_iteration"
            ][0]["input_hash"]
            hashes.add(h)
        self.assertEqual(len(hashes), 4,
                         "Input hashes should differ across ranks")


class TestSchemaAndStructure(unittest.TestCase):
    """Verify schema constant and artifact structure."""

    def test_schema_constant(self) -> None:
        self.assertEqual(RAW_EVIDENCE_SCHEMA_V2, "tp4_raw_evidence/v2")

    def test_reduced_schema_constant(self) -> None:
        self.assertEqual(REDUCED_EVIDENCE_SCHEMA, "tp4_reduced_evidence/v1")

    def test_artifact_has_required_top_level_keys(self) -> None:
        artifact = _produce_small()
        for key in ("schema", "iterations", "elements", "ranks",
                     "tolerance_policy", "per_rank_raw", "artifact_sha256"):
            self.assertIn(key, artifact)

    def test_per_rank_raw_structure(self) -> None:
        artifact = _produce_small(iterations=2, elements=128)
        self.assertEqual(len(artifact["per_rank_raw"]), 4)
        for rank_idx, rank_entry in enumerate(artifact["per_rank_raw"]):
            self.assertEqual(rank_entry["rank"], rank_idx)
            self.assertEqual(len(rank_entry["per_iteration"]), 2)
            for it_idx, it_entry in enumerate(rank_entry["per_iteration"]):
                self.assertEqual(it_entry["iteration"], it_idx)
                for key in ("input_hash", "output_hash", "fp32_truth_hash",
                            "output_sample", "mae", "rmse", "max_abs_error",
                            "mismatch_count", "outside_tolerance_count"):
                    self.assertIn(key, it_entry)

    def test_output_sample_bounded(self) -> None:
        artifact = _produce_small(iterations=1, elements=6144)
        for rank_entry in artifact["per_rank_raw"]:
            for it_entry in rank_entry["per_iteration"]:
                self.assertLessEqual(
                    len(it_entry["output_sample"]), OUTPUT_SAMPLE_SIZE,
                )

    def test_tolerance_policy_correct(self) -> None:
        artifact = _produce_small()
        self.assertEqual(artifact["tolerance_policy"], "bf16_fixed_abs_v1")


class TestProducerValidation(unittest.TestCase):
    """Verify the producer rejects invalid parameters."""

    def test_zero_iterations_rejected(self) -> None:
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(record_iterations=0)

    def test_negative_iterations_rejected(self) -> None:
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(record_iterations=-1)

    def test_non_int_iterations_rejected(self) -> None:
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(record_iterations=2.0)  # type: ignore[arg-type]

    def test_bool_iterations_rejected(self) -> None:
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(record_iterations=True)  # type: ignore[arg-type]

    def test_wrong_world_size_rejected(self) -> None:
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(record_iterations=1, world_size=2)

    def test_zero_elements_rejected(self) -> None:
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(record_iterations=1, elements=0)



class TestForgeryRegression(unittest.TestCase):
    """Reproduce the exact forgery from the adversarial review:
    change all hashes, reduce each sample to one bogus value, zero
    all metrics, recompute the self-hash, and verify validation REJECTS.
    """

    def test_forged_hashes_samples_metrics_rejected(self) -> None:
        """Change all hashes, reduce samples, zero metrics, rehash — must fail."""
        artifact = _produce_small(iterations=2, elements=128)
        forged = _deep_copy_artifact(artifact)
        # Change all hashes to bogus values.
        for rank_entry in forged["per_rank_raw"]:
            for it in rank_entry["per_iteration"]:
                it["input_hash"] = "d" * 64
                it["output_hash"] = "e" * 64
                it["fp32_truth_hash"] = "f" * 64
                # Reduce each sample to one bogus value.
                it["output_sample"] = [42.0]
                # Zero all metrics.
                it["mae"] = 0.0
                it["rmse"] = 0.0
                it["max_abs_error"] = 0.0
                it["mismatch_count"] = 0
                it["outside_tolerance_count"] = 0
        # Recompute the self-hash so the binding is self-consistent.
        forged["artifact_sha256"] = compute_artifact_sha256(forged)
        # Validation must REJECT this — the recomputing validator catches
        # the forged hashes, bogus samples, and zeroed metrics.
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(forged)
        # The error must mention a specific tampered field.
        self.assertTrue(
            "tampered" in str(ctx.exception).lower()
            or "expected" in str(ctx.exception).lower(),
            f"Error should mention tampering: {ctx.exception}",
        )

    def test_valid_artifact_independently_recomputed(self) -> None:
        """A valid artifact's complete published metrics must be
        independently recomputed by the validator."""
        artifact = _produce_small(iterations=3, elements=256)
        result = validate_raw_evidence(artifact)
        self.assertTrue(result["valid"])
        # The validator recomputed everything — verify per-rank reduced
        # metrics match what we compute independently.
        for rank_idx, rm in enumerate(result["per_rank_reduced"]):
            self.assertEqual(rm["rank"], rank_idx)
            self.assertEqual(rm["iterations"], 3)
            # Recompute expected metrics independently.
            total_mae = 0.0
            total_sq = 0.0
            total_max = 0.0
            total_mm = 0
            total_outside = 0
            for seq in range(3):
                inputs = [
                    make_rank_input(seq, r, 4, 1, 256, "random")
                    for r in range(4)
                ]
                outputs = tp4_ring_reduce_all_ranks(inputs)
                truth_fp32 = fp32_ground_truth(inputs)
                truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)
                output = outputs[rank_idx]
                err = (output.float() - truth_fp32).abs()
                total_mae += float(err.mean().item())
                total_sq += float(err.square().sum().item())
                total_max = max(total_max, float(err.max().item()))
                total_mm += int(
                    torch.count_nonzero(
                        output.view(torch.int16) != truth_bf16.view(torch.int16)
                    ).item()
                )
                total_outside += int(
                    torch.count_nonzero(err > TOLERANCE_THRESHOLD_BF16).item()
                )
            expected_mae = total_mae / 3
            expected_rmse = (total_sq / (3 * 256)) ** 0.5
            self.assertAlmostEqual(rm["mae"], expected_mae, places=10)
            self.assertAlmostEqual(rm["rmse"], expected_rmse, places=10)
            self.assertAlmostEqual(rm["max_abs_error"], total_max, places=10)


class TestObservedEvidence(unittest.TestCase):
    """Tests for the DISABLED observed evidence type.

    No real offline runtime output seam exists in the public checkout.
    The tp4_numerical_audit.py probe requires CUDA, RDMA, and a live
    4-rank process group; it cannot run offline.  ObservedEvidenceReceipt
    accepted arbitrary caller-supplied hashes, which is caller-fabricated.

    The observed evidence type is DISABLED.  The producer and validator
    must reject any artifact claiming evidence_type='observed'.  A modeled
    artifact must remain explicitly modeled and cannot satisfy live
    numerical proof.
    """

    def _make_observed_receipts(
        self, iterations: int, world_size: int,
    ) -> dict[int, list[ObservedEvidenceReceipt]]:
        """Create observed receipts with arbitrary output hashes."""
        receipts: dict[int, list[ObservedEvidenceReceipt]] = {}
        for rank in range(world_size):
            rank_receipts = []
            for it in range(iterations):
                rank_receipts.append(ObservedEvidenceReceipt(
                    rank=rank,
                    iteration=it,
                    output_hash="a" * 64,  # arbitrary caller-supplied
                    selector="custom",
                    custom_collectives=iterations,
                    fallback_collectives=0,
                    unsupported_bypassed_collectives=0,
                    unclassified_collectives=0,
                ))
            receipts[rank] = rank_receipts
        return receipts

    def test_observed_evidence_type_rejected_by_producer(self) -> None:
        """The producer must reject evidence_type='observed'."""
        receipts = self._make_observed_receipts(2, 4)
        with self.assertRaises(RawEvidenceError) as ctx:
            RawEvidenceProducer().produce(
                record_iterations=2, world_size=4, elements=128,
                evidence_type=EVIDENCE_TYPE_OBSERVED,
                observed_receipts=receipts,
            )
        self.assertIn("evidence_type", str(ctx.exception).lower())
        self.assertIn("modeled", str(ctx.exception))

    def test_observed_evidence_type_rejected_by_validator(self) -> None:
        """The validator must reject evidence_type='observed' artifacts."""
        artifact = _produce_small(iterations=2, elements=128)
        forged = _deep_copy_artifact(artifact)
        forged["evidence_type"] = "observed"
        forged["artifact_sha256"] = compute_artifact_sha256(forged)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(forged)
        self.assertIn("evidence_type", str(ctx.exception).lower())

    def test_modeled_artifact_has_evidence_type(self) -> None:
        """A modeled artifact must include evidence_type='modeled'."""
        artifact = _produce_small(iterations=2, elements=128)
        self.assertEqual(artifact["evidence_type"], "modeled")

    def test_modeled_artifact_relabeled_observed_fails(self) -> None:
        """A modeled artifact with evidence_type changed to 'observed'
        must fail — observed type is disabled."""
        artifact = _produce_small(iterations=2, elements=128)
        forged = _deep_copy_artifact(artifact)
        forged["evidence_type"] = "observed"
        forged["artifact_sha256"] = compute_artifact_sha256(forged)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(forged)
        self.assertIn("evidence_type", str(ctx.exception).lower())

    def test_observed_relabeled_modeled_fails(self) -> None:
        """An observed artifact relabeled as 'modeled' must fail —
        it has transport keys and lacks output_sample."""
        # First produce a modeled artifact, then forge it to look observed
        # by adding transport keys and removing output_sample.
        artifact = _produce_small(iterations=2, elements=128)
        forged = _deep_copy_artifact(artifact)
        for rank_entry in forged["per_rank_raw"]:
            for it in rank_entry["per_iteration"]:
                del it["output_sample"]
                it["transport_selector"] = "custom"
                it["transport_custom"] = 2
                it["transport_fallback"] = 0
                it["transport_unsupported"] = 0
                it["transport_unclassified"] = 0
        forged["artifact_sha256"] = compute_artifact_sha256(forged)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(forged)
        # Must fail because modeled entries must have output_sample
        self.assertTrue(
            "missing keys" in str(ctx.exception).lower()
            or "output_sample" in str(ctx.exception).lower(),
            f"Should fail on missing output_sample: {ctx.exception}",
        )

    def test_observed_without_receipts_rejected(self) -> None:
        """Observed evidence without receipts must fail at production."""
        with self.assertRaises(RawEvidenceError) as ctx:
            RawEvidenceProducer().produce(
                record_iterations=2, world_size=4, elements=128,
                evidence_type=EVIDENCE_TYPE_OBSERVED,
            )
        self.assertIn("evidence_type", str(ctx.exception).lower())

    def test_observed_with_wrong_selector_receipt_rejected(self) -> None:
        """ObservedEvidenceReceipt with invalid selector must fail
        (the class itself still validates its fields)."""
        with self.assertRaises(RawEvidenceError):
            ObservedEvidenceReceipt(
                rank=0, iteration=0, output_hash="a" * 64,
                selector="invalid",
                custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0,
                unclassified_collectives=0,
            )

    def test_observed_invalid_evidence_type_fails(self) -> None:
        """Invalid evidence_type must fail."""
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(
                record_iterations=2, world_size=4, elements=128,
                evidence_type="hybrid",  # type: ignore[arg-type]
            )

    def test_arbitrary_hashes_not_accepted_as_observed(self) -> None:
        """Arbitrary 'a'*64 hashes cannot be accepted as observed evidence.
        The producer must reject evidence_type='observed' regardless of
        receipt content."""
        receipts = self._make_observed_receipts(2, 4)
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(
                record_iterations=2, world_size=4, elements=128,
                evidence_type=EVIDENCE_TYPE_OBSERVED,
                observed_receipts=receipts,
            )

    def test_relabeling_sircl_as_nccl_observed_rejected(self) -> None:
        """Relabeling SIRCL evidence as NCCL via observed type is rejected."""
        receipts = self._make_observed_receipts(2, 4)
        # Change selector to 'disabled' (NCCL)
        for rank in receipts:
            for rcpt in receipts[rank]:
                object.__setattr__(rcpt, "selector", "disabled")
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(
                record_iterations=2, world_size=4, elements=128,
                evidence_type=EVIDENCE_TYPE_OBSERVED,
                observed_receipts=receipts,
            )

    def test_mixed_selectors_across_ranks_observed_rejected(self) -> None:
        """Mixed selectors across ranks in observed evidence is rejected
        — the entire observed type is disabled."""
        receipts = self._make_observed_receipts(2, 4)
        receipts[1] = [
            ObservedEvidenceReceipt(
                rank=1, iteration=it, output_hash="b" * 64,
                selector="disabled",
                custom_collectives=0, fallback_collectives=2,
                unsupported_bypassed_collectives=0,
                unclassified_collectives=0,
            )
            for it in range(2)
        ]
        with self.assertRaises(RawEvidenceError):
            RawEvidenceProducer().produce(
                record_iterations=2, world_size=4, elements=128,
                evidence_type=EVIDENCE_TYPE_OBSERVED,
                observed_receipts=receipts,
            )

    def test_self_rehashed_observed_artifact_rejected(self) -> None:
        """A modeled artifact relabeled as observed with a recomputed
        self-hash must still be rejected by the validator."""
        artifact = _produce_small(iterations=2, elements=128)
        forged = _deep_copy_artifact(artifact)
        forged["evidence_type"] = "observed"
        # Add transport keys to make it look observed
        for rank_entry in forged["per_rank_raw"]:
            for it in rank_entry["per_iteration"]:
                del it["output_sample"]
                it["transport_selector"] = "custom"
                it["transport_custom"] = 2
                it["transport_fallback"] = 0
                it["transport_unsupported"] = 0
                it["transport_unclassified"] = 0
        # Recompute self-hash so the binding is self-consistent
        forged["artifact_sha256"] = compute_artifact_sha256(forged)
        with self.assertRaises(RawEvidenceError) as ctx:
            validate_raw_evidence(forged)
        # Must fail on evidence_type, not on hash
        self.assertIn("evidence_type", str(ctx.exception).lower())

    def test_modeled_artifact_still_validates(self) -> None:
        """A modeled artifact must still validate after disabling observed."""
        artifact = _produce_small(iterations=2, elements=128)
        result = validate_raw_evidence(artifact)
        self.assertTrue(result["valid"])
        self.assertEqual(result["evidence_type"], "modeled")

if __name__ == "__main__":
    unittest.main()
