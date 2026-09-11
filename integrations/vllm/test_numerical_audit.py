"""CPU-only tests for deterministic TP4 numerical-audit inputs."""

from __future__ import annotations

import unittest

import torch

from tp4_numerical_audit import ELEMENTS, make_rank_input


class NumericalAuditInputTest(unittest.TestCase):
    def test_inputs_are_deterministic_bfloat16_vectors(self) -> None:
        first = make_rank_input(7, 2)
        second = make_rank_input(7, 2)

        self.assertEqual(first.shape, (ELEMENTS,))
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.isfinite(first).all()))

    def test_sequence_or_rank_changes_the_input(self) -> None:
        baseline = make_rank_input(0, 0)

        self.assertFalse(torch.equal(baseline, make_rank_input(1, 0)))
        self.assertFalse(torch.equal(baseline, make_rank_input(0, 1)))

    def test_cancellation_case_has_a_finite_fp32_ground_truth(self) -> None:
        inputs = [make_rank_input(1, rank) for rank in range(4)]
        truth = torch.stack([tensor.float() for tensor in inputs]).sum(dim=0)

        self.assertTrue(bool(torch.isfinite(truth).all()))
        self.assertGreater(float(truth.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
