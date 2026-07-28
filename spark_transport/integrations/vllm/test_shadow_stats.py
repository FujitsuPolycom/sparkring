"""CPU tests for the real-model shadow correctness metrics."""

from __future__ import annotations

import unittest

import torch

from spark_tp4_backend import _ShadowStats


class ShadowStatsTest(unittest.TestCase):
    def test_exact_bfloat16_values_report_zero_error(self) -> None:
        reference = torch.tensor(
            [0.0, 1.0, -1.0, float("nan"), float("inf"), -float("inf")],
            dtype=torch.bfloat16,
        )
        stats = _ShadowStats()

        stats.observe(reference.clone(), reference)

        self.assertEqual(stats.report(), (0, 0, 0, 0.0, 0, 0, 0, 0))

    def test_one_bfloat16_ulp_is_measured(self) -> None:
        reference = torch.tensor([1.0], dtype=torch.bfloat16)
        candidate = reference.clone()
        candidate.view(torch.int16)[0] += 1
        stats = _ShadowStats()

        stats.observe(candidate, reference)

        (
            exact,
            outside,
            nonfinite,
            maximum,
            max_ulp,
            ulp_gt1,
            ulp_gt2,
            ulp_gt4,
        ) = stats.report()
        self.assertEqual(exact, 1)
        self.assertEqual(outside, 0)
        self.assertEqual(nonfinite, 0)
        self.assertAlmostEqual(maximum, 0.0078125)
        self.assertEqual(max_ulp, 1)
        self.assertEqual((ulp_gt1, ulp_gt2, ulp_gt4), (0, 0, 0))

    def test_nonfinite_disagreement_cannot_hide_in_tolerance(self) -> None:
        reference = torch.tensor([float("inf")], dtype=torch.bfloat16)
        candidate = torch.tensor([float("nan")], dtype=torch.bfloat16)
        stats = _ShadowStats()

        stats.observe(candidate, reference)

        (
            exact,
            outside,
            nonfinite,
            maximum,
            max_ulp,
            ulp_gt1,
            ulp_gt2,
            ulp_gt4,
        ) = stats.report()
        self.assertEqual(exact, 1)
        self.assertEqual(outside, 1)
        self.assertEqual(nonfinite, 1)
        self.assertEqual(maximum, 0.0)
        self.assertEqual(max_ulp, 0)
        self.assertEqual((ulp_gt1, ulp_gt2, ulp_gt4), (0, 0, 0))

    def test_ulp_tail_counters_describe_error_distribution(self) -> None:
        reference = torch.tensor([1.0, 1.0, 1.0], dtype=torch.bfloat16)
        candidate = reference.clone()
        candidate.view(torch.int16)[0] += 2
        candidate.view(torch.int16)[1] += 3
        candidate.view(torch.int16)[2] += 5
        stats = _ShadowStats()

        stats.observe(candidate, reference)

        report = stats.report()
        self.assertEqual(report[4], 5)
        self.assertEqual(report[5:], (3, 2, 1))


if __name__ == "__main__":
    unittest.main()
