"""CPU-only tests for the modeled BF16 reduction-order analysis.

Validates input generation, reduction-order invariants, finiteness
checking (per-iteration), all-rank output identity, self-consistency,
input hardening, and that metrics are reported without a numerical
quality pass/fail gate.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from spark_fp32_ground_truth import (
    AuditCase,
    AuditInputError,
    ELEMENTS,
    Q1_BYTES,
    TOLERANCE_POLICY_NAME,
    default_cases,
    fp32_ground_truth,
    fp32_truth_rounded_to_dtype,
    make_rank_input,
    run_audit,
    run_case,
    sequential_bf16_sum,
    tp2_ring_reduce_all_ranks,
    tp4_ring_reduce_all_ranks,
)


class InputGenerationTest(unittest.TestCase):
    def test_inputs_are_deterministic_bfloat16(self) -> None:
        first = make_rank_input(7, 2, 4, 1, ELEMENTS, "random")
        second = make_rank_input(7, 2, 4, 1, ELEMENTS, "random")
        self.assertEqual(first.shape, (ELEMENTS,))
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.isfinite(first).all()))

    def test_uniform_pattern_produces_constant_tensors(self) -> None:
        tensor = make_rank_input(3, 1, 4, 1, ELEMENTS, "uniform")
        self.assertEqual(tensor.dtype, torch.bfloat16)
        unique = tensor.unique()
        self.assertEqual(unique.numel(), 1)

    def test_cancellation_pattern_has_finite_fp32_truth(self) -> None:
        inputs = [
            make_rank_input(1, rank, 4, 1, ELEMENTS, "cancellation")
            for rank in range(4)
        ]
        truth = torch.stack([t.float() for t in inputs]).sum(dim=0)
        self.assertTrue(bool(torch.isfinite(truth).all()))
        self.assertGreater(float(truth.abs().max()), 0.0)

    def test_sequence_or_rank_changes_input(self) -> None:
        baseline = make_rank_input(0, 0, 4, 1, ELEMENTS, "random")
        self.assertFalse(
            torch.equal(baseline, make_rank_input(1, 0, 4, 1, ELEMENTS, "random"))
        )
        self.assertFalse(
            torch.equal(baseline, make_rank_input(0, 1, 4, 1, ELEMENTS, "random"))
        )


class InputHardeningTest(unittest.TestCase):
    def test_bool_as_sequence_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            make_rank_input(True, 0, 4, 1, ELEMENTS, "random")  # type: ignore[arg-type]

    def test_bool_as_rank_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            make_rank_input(0, True, 4, 1, ELEMENTS, "random")  # type: ignore[arg-type]

    def test_invalid_pattern_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            make_rank_input(0, 0, 4, 1, ELEMENTS, "bogus")

    def test_rank_exceeds_world_size_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            make_rank_input(0, 4, 4, 1, ELEMENTS, "random")

    def test_zero_rows_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            make_rank_input(0, 0, 4, 0, ELEMENTS, "random")

    def test_world_size_1_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            make_rank_input(0, 0, 1, 1, ELEMENTS, "random")

    def test_case_validation_rejects_bad_world_size(self) -> None:
        with self.assertRaises(AuditInputError):
            AuditCase(name="bad", rows=1, width=ELEMENTS, world_size=1, pattern="random")

    def test_case_validation_rejects_bad_pattern(self) -> None:
        with self.assertRaises(AuditInputError):
            AuditCase(name="bad", rows=1, width=ELEMENTS, world_size=4, pattern="nope")

    def test_case_validation_rejects_bool_rows(self) -> None:
        with self.assertRaises(AuditInputError):
            AuditCase(name="bad", rows=True, width=ELEMENTS, world_size=4, pattern="random")  # type: ignore[arg-type]


class ReductionOrderTest(unittest.TestCase):
    def test_tp4_ring_all_ranks_identical_for_uniform(self) -> None:
        inputs = [
            torch.full((ELEMENTS,), float(r + 1), dtype=torch.bfloat16)
            for r in range(4)
        ]
        outputs = tp4_ring_reduce_all_ranks(inputs)
        for i in range(1, 4):
            self.assertTrue(torch.equal(outputs[0], outputs[i]))

    def test_tp4_ring_matches_fp32_for_uniform(self) -> None:
        inputs = [
            torch.full((ELEMENTS,), float(r + 1), dtype=torch.bfloat16)
            for r in range(4)
        ]
        outputs = tp4_ring_reduce_all_ranks(inputs)
        truth = fp32_ground_truth(inputs)
        self.assertTrue(torch.equal(outputs[0], truth))

    def test_tp2_ring_matches_fp32_for_uniform(self) -> None:
        inputs = [
            torch.full((ELEMENTS,), float(r + 1), dtype=torch.bfloat16)
            for r in range(2)
        ]
        outputs = tp2_ring_reduce_all_ranks(inputs)
        truth = fp32_ground_truth(inputs)
        self.assertTrue(torch.equal(outputs[0], truth))

    def test_tp4_ring_uses_pairwise_grouping_order(self) -> None:
        """Verify the ring groups as (r0+r1)+(r2+r3), not some other order.

        This is NOT a 'self-consistent across permutations' test — it
        verifies the ring's specific grouping order matches the
        expected pairwise structure. BF16 addition is non-associative,
        so a different grouping (e.g. ((r0+r1)+r2)+r3) can produce
        a different result, which is correct and expected.
        """
        inputs = [
            make_rank_input(3, r, 4, 1, ELEMENTS, "random")
            for r in range(4)
        ]
        outputs = tp4_ring_reduce_all_ranks(inputs)
        pairwise = (inputs[0] + inputs[1]) + (inputs[2] + inputs[3])
        self.assertTrue(torch.equal(outputs[0], pairwise))

    def test_tp4_wrong_input_count_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            tp4_ring_reduce_all_ranks([torch.zeros(1, dtype=torch.bfloat16)] * 3)

    def test_tp2_wrong_input_count_rejected(self) -> None:
        with self.assertRaises(AuditInputError):
            tp2_ring_reduce_all_ranks([torch.zeros(1, dtype=torch.bfloat16)] * 3)


class GroundTruthTest(unittest.TestCase):
    def test_fp32_truth_preserves_fp32_dtype(self) -> None:
        """fp32_ground_truth must return unrounded FP32, not BF16.

        Regression: the old implementation rounded to BF16 at the end,
        causing all error metrics to compare against a BF16-rounded truth
        and produce a falsely small MAE (0.0121087 instead of ~0.0178331).
        """
        inputs = [
            make_rank_input(0, r, 4, 1, ELEMENTS, "random")
            for r in range(4)
        ]
        truth = fp32_ground_truth(inputs)
        self.assertEqual(truth.dtype, torch.float32)
        fp32_sum = torch.stack([t.float() for t in inputs]).sum(dim=0)
        self.assertTrue(torch.equal(truth, fp32_sum))

    def test_fp32_truth_mae_matches_actual_fp32(self) -> None:
        """Reproduce the false-MAE regression: BF16-rounded truth
        gives MAE 0.0121087, but actual FP32 truth gives ~0.0178331.

        This is a regression test for the prior false MAE that compared
        against BF16-rounded truth instead of unrounded FP32.
        """
        inputs = [
            make_rank_input(0, r, 4, 1, ELEMENTS, "random")
            for r in range(4)
        ]
        truth_fp32 = fp32_ground_truth(inputs)
        truth_bf16 = truth_fp32.to(torch.bfloat16)
        candidate = tp4_ring_reduce_all_ranks(inputs)[0]

        mae_vs_bf16 = float(
            (candidate.float() - truth_bf16.float()).abs().mean()
        )
        mae_vs_fp32 = float(
            (candidate.float() - truth_fp32).abs().mean()
        )
        # The BF16-rounded truth produces a falsely small MAE
        self.assertAlmostEqual(mae_vs_bf16, 0.0121087, places=5)
        # The actual FP32 truth produces a larger, correct MAE
        self.assertAlmostEqual(mae_vs_fp32, 0.0178331, places=5)
        # They must differ — this is the regression
        self.assertNotAlmostEqual(mae_vs_bf16, mae_vs_fp32, places=5)

    def test_fp32_truth_differs_from_sequential_bf16_sum(self) -> None:
        inputs = [
            make_rank_input(1, r, 4, 1, ELEMENTS, "cancellation")
            for r in range(4)
        ]
        truth_fp32 = fp32_ground_truth(inputs)
        bf16_sum = sequential_bf16_sum(inputs)
        # FP32 truth (unrounded) differs from BF16 sequential sum
        self.assertFalse(torch.equal(truth_fp32, bf16_sum))
        # FP32 truth rounded to BF16 also differs from sequential BF16 sum
        # because BF16 addition is non-associative
        truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)
        self.assertFalse(torch.equal(truth_bf16, bf16_sum))

    def test_sequential_sum_disclaims_nccl_modeling(self) -> None:
        """The sequential_bf16_sum docstring must disclaim NCCL modeling."""
        import spark_fp32_ground_truth as mod
        docstring = mod.sequential_bf16_sum.__doc__ or ""
        self.assertIn("not", docstring.lower())
        self.assertIn("model", docstring.lower())
        self.assertIn("NCCL", docstring)  # word present but disclaimed


class AuditCaseTest(unittest.TestCase):
    def test_default_cases_cover_tp2_and_tp4(self) -> None:
        cases = default_cases()
        world_sizes = {case.world_size for case in cases}
        self.assertIn(2, world_sizes)
        self.assertIn(4, world_sizes)

    def test_default_cases_include_hot_decode_shape(self) -> None:
        cases = default_cases()
        hot = [
            c for c in cases
            if c.rows == 1 and c.width == ELEMENTS and c.world_size == 4
        ]
        self.assertGreater(len(hot), 0)

    def test_default_cases_include_cancellation_pattern(self) -> None:
        cases = default_cases()
        cancel = [c for c in cases if c.pattern == "cancellation"]
        self.assertGreater(len(cancel), 0)

    def test_q1_bytes_matches_glm_decode(self) -> None:
        self.assertEqual(Q1_BYTES, 12288)


class AuditExecutionTest(unittest.TestCase):
    def test_uniform_case_invariants_pass_with_zero_error(self) -> None:
        case = AuditCase(
            name="tp4_uniform", rows=1, width=ELEMENTS,
            world_size=4, pattern="uniform",
        )
        result = run_case(case, iterations=10)
        self.assertTrue(result.invariants.passed)
        self.assertEqual(result.metrics.candidate_mae, 0.0)
        self.assertEqual(result.metrics.candidate_max_abs, 0.0)

    def test_random_case_invariants_pass_and_reports_metrics(self) -> None:
        case = AuditCase(
            name="tp4_q1_random", rows=1, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        result = run_case(case, iterations=20)
        self.assertTrue(result.invariants.passed)
        self.assertGreater(result.compared_elements, 0)
        self.assertGreaterEqual(result.metrics.candidate_exact_agreement, 0)

    def test_cancellation_case_invariants_pass(self) -> None:
        case = AuditCase(
            name="tp4_cancel", rows=1, width=ELEMENTS,
            world_size=4, pattern="cancellation",
        )
        result = run_case(case, iterations=20)
        self.assertTrue(result.invariants.passed, result.invariants.failure_reason)

    def test_tp2_case_invariants_pass(self) -> None:
        case = AuditCase(
            name="tp2_q1", rows=1, width=ELEMENTS,
            world_size=2, pattern="random",
        )
        result = run_case(case, iterations=20)
        self.assertTrue(result.invariants.passed)

    def test_multi_row_case_invariants_pass(self) -> None:
        case = AuditCase(
            name="tp4_q5", rows=5, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        result = run_case(case, iterations=10)
        self.assertTrue(result.invariants.passed)
        self.assertEqual(result.compared_elements, 5 * ELEMENTS * 10)

    def test_full_audit_summary_aggregates_correctly(self) -> None:
        summary = run_audit(iterations=5)
        self.assertGreater(summary.total_passed, 0)
        self.assertEqual(summary.total_failed, 0)
        self.assertTrue(summary.all_invariants_passed)
        self.assertGreater(summary.total_compared, 0)

    def test_report_renders_without_error(self) -> None:
        from spark_fp32_ground_truth import render_report
        summary = run_audit(iterations=3)
        report = render_report(summary)
        self.assertIn("MODELED_BF16_REDUCTION_ORDER_ANALYSIS", report)
        self.assertIn("does not execute SIRCL", report)
        self.assertIn("invariants", report)

    def test_no_numerical_quality_pass_gate(self) -> None:
        """Metrics must be reported but not used for pass/fail."""
        case = AuditCase(
            name="tp4_cancel", rows=1, width=ELEMENTS,
            world_size=4, pattern="cancellation",
        )
        result = run_case(case, iterations=50)
        # Invariants pass even if MAE is nonzero
        self.assertTrue(result.invariants.passed)
        # There is no 'passed' field based on MAE threshold
        self.assertFalse(hasattr(result, "failure_reason"))

    def test_invalid_iterations_rejected(self) -> None:
        case = AuditCase(
            name="tp4_q1", rows=1, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        with self.assertRaises(AuditInputError):
            run_case(case, iterations=0)
        with self.assertRaises(AuditInputError):
            run_case(case, iterations=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_iterations", [0, -1, True, 1.5, "10"])
def test_invalid_iterations_types_rejected(bad_iterations: object) -> None:
    case = AuditCase(
        name="tp4_q1", rows=1, width=ELEMENTS,
        world_size=4, pattern="random",
    )
    with pytest.raises(AuditInputError):
        run_case(case, iterations=bad_iterations)  # type: ignore[arg-type]


class FP32TruthAdversarialTest(unittest.TestCase):
    """Adversarial tests for the FP32-vs-BF16 truth distinction.

    These tests defend against the prior false-MAE bug where
    fp32_ground_truth returned BF16-rounded values, causing all
    continuous error metrics to be computed against the wrong
    reference.
    """

    def test_fp32_truth_dtype_is_float32(self) -> None:
        inputs = [
            make_rank_input(0, r, 4, 1, ELEMENTS, "random")
            for r in range(4)
        ]
        truth = fp32_ground_truth(inputs)
        self.assertEqual(truth.dtype, torch.float32)

    def test_fp32_truth_not_equal_to_bf16_rounded(self) -> None:
        """FP32 truth must differ from its own BF16 rounding for
        random inputs — otherwise the rounding is a no-op and the
        bug is undetectable."""
        inputs = [
            make_rank_input(0, r, 4, 1, ELEMENTS, "random")
            for r in range(4)
        ]
        truth_fp32 = fp32_ground_truth(inputs)
        truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)
        self.assertFalse(torch.equal(truth_fp32, truth_bf16))

    def test_rounded_helper_preserves_bf16_dtype(self) -> None:
        truth = torch.randn(100, dtype=torch.float32)
        rounded = fp32_truth_rounded_to_dtype(truth, torch.bfloat16)
        self.assertEqual(rounded.dtype, torch.bfloat16)

    def test_continuous_metrics_use_unrounded_fp32(self) -> None:
        """run_case MAE must match the MAE computed against actual
        FP32 truth, not against BF16-rounded truth."""
        case = AuditCase(
            name="tp4_q1_random", rows=1, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        result = run_case(case, iterations=1)
        # Recompute manually against FP32 truth
        inputs = [
            make_rank_input(0, r, 4, 1, ELEMENTS, "random")
            for r in range(4)
        ]
        candidate = tp4_ring_reduce_all_ranks(inputs)[0]
        truth_fp32 = fp32_ground_truth(inputs)
        expected_mae = float((candidate.float() - truth_fp32).abs().mean())
        self.assertAlmostEqual(
            result.metrics.candidate_mae, expected_mae, places=6,
        )

    def test_exact_agreement_uses_bf16_rounded_truth(self) -> None:
        """Exact agreement count must compare candidate against FP32
        truth rounded once to BF16, not against unrounded FP32
        (which would always be zero for BF16 candidates)."""
        case = AuditCase(
            name="tp4_q1_random", rows=1, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        result = run_case(case, iterations=1)
        inputs = [
            make_rank_input(0, r, 4, 1, ELEMENTS, "random")
            for r in range(4)
        ]
        candidate = tp4_ring_reduce_all_ranks(inputs)[0]
        truth_fp32 = fp32_ground_truth(inputs)
        truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)
        expected_exact = int(torch.count_nonzero(candidate == truth_bf16).item())
        self.assertEqual(
            result.metrics.candidate_exact_agreement, expected_exact,
        )
        # Exact agreement against BF16 truth should be > 0 for random
        # (some elements will round the same way)
        # but not 100% (some will differ due to BF16 addition order)

    def test_nonzero_bf16_mismatch_exists(self) -> None:
        """For cancellation inputs, the TP4 ring and sequential BF16
        sum must produce nonzero mismatches — BF16 addition is
        non-associative and different orders produce different bits."""
        case = AuditCase(
            name="tp4_cancel", rows=1, width=ELEMENTS,
            world_size=4, pattern="cancellation",
        )
        result = run_case(case, iterations=50)
        self.assertGreater(
            result.metrics.candidate_sequential_mismatches, 0,
            "Cancellation pattern must produce nonzero BF16 mismatches",
        )

    def test_outside_tolerance_count_is_explicit(self) -> None:
        """outside_tolerance_count must be a real count governed by
        an explicit policy, not implicitly zero or NOT-JUDGED."""
        case = AuditCase(
            name="tp4_q1_random", rows=1, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        result = run_case(case, iterations=10)
        self.assertGreaterEqual(result.metrics.outside_tolerance_count, 0)
        self.assertEqual(
            result.metrics.tolerance_policy, TOLERANCE_POLICY_NAME,
        )
        # For random inputs, some elements should be outside BF16 ULP
        self.assertGreater(result.metrics.outside_tolerance_count, 0)

    def test_outside_tolerance_bounded_by_compared(self) -> None:
        """outside_tolerance_count cannot exceed compared_elements."""
        case = AuditCase(
            name="tp4_q1_random", rows=1, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        result = run_case(case, iterations=10)
        self.assertLessEqual(
            result.metrics.outside_tolerance_count,
            result.compared_elements,
        )

    def test_truth_vs_sequential_mismatches_differ(self) -> None:
        """Reproduce the 1938-vs-1823 case: rounded-truth disagreements
        were 1938 but the reported field (candidate_reference_mismatches,
        which compared vs sequential BF16) was 1823.

        The fix separates the two counts:
        - candidate_truth_mismatches: vs FP32 truth rounded once to BF16
        - candidate_sequential_mismatches: vs sequential BF16 sum

        These must be independently reported and may differ.
        """
        case = AuditCase(
            name="tp4_q1_cancellation", rows=1, width=ELEMENTS,
            world_size=4, pattern="cancellation",
        )
        result = run_case(case, iterations=10)
        # Both fields must exist and be independently computed.
        self.assertGreaterEqual(
            result.metrics.candidate_truth_mismatches, 0,
        )
        self.assertGreaterEqual(
            result.metrics.candidate_sequential_mismatches, 0,
        )
        # Recompute across all 10 iterations (run_case accumulates).
        total_truth_mm = 0
        total_seq_mm = 0
        for seq in range(10):
            inputs = [
                make_rank_input(seq, r, 4, 1, ELEMENTS, "cancellation")
                for r in range(4)
            ]
            candidate = tp4_ring_reduce_all_ranks(inputs)[0]
            truth_fp32 = fp32_ground_truth(inputs)
            truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)
            reference = sequential_bf16_sum(inputs)
            total_truth_mm += int(
                torch.count_nonzero(
                    candidate.view(torch.int16) != truth_bf16.view(torch.int16)
                ).item()
            )
            total_seq_mm += int(
                torch.count_nonzero(
                    candidate.view(torch.int16) != reference.view(torch.int16)
                ).item()
            )
        self.assertEqual(
            result.metrics.candidate_truth_mismatches, total_truth_mm,
        )
        self.assertEqual(
            result.metrics.candidate_sequential_mismatches, total_seq_mm,
        )
        # The counts CAN differ (this is the 1938-vs-1823 scenario).
        # They measure different comparisons. The contract is now
        # unambiguous: each field name says exactly what it compares.
        self.assertNotEqual(
            result.metrics.candidate_truth_mismatches,
            result.metrics.candidate_sequential_mismatches,
            "Truth mismatches and sequential mismatches measure "
            "different things and should differ for cancellation inputs",
        )

    def test_tolerance_policy_name_is_truthful(self) -> None:
        """The tolerance policy name must be 'bf16_fixed_abs_v1', not
        'bf16_ulp_v1' — the threshold is a fixed absolute 2^{-7},
        not a magnitude-dependent BF16 ULP."""
        self.assertEqual(TOLERANCE_POLICY_NAME, "bf16_fixed_abs_v1")
        case = AuditCase(
            name="tp4_q1_random", rows=1, width=ELEMENTS,
            world_size=4, pattern="random",
        )
        result = run_case(case, iterations=1)
        self.assertEqual(
            result.metrics.tolerance_policy, "bf16_fixed_abs_v1",
        )


if __name__ == "__main__":
    unittest.main()
