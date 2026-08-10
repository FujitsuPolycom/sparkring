"""CPU-only tests for the multi-slot ingress feasibility contract.

Validates that the feasibility contract correctly identifies the
unimplemented requirements, cites source locations, and does not
model or estimate performance.

This is a **feasibility contract**, not a performance simulator.
No speedup values are tested or produced.
"""

from __future__ import annotations

import unittest

from spark_multislot_overlap_experiment import (
    ExperimentInputError,
    FeasibilityContract,
    FeasibilityRequirement,
    build_contract,
    default_requirements,
    render_report,
)


class RequirementTest(unittest.TestCase):
    def test_all_requirements_unimplemented(self) -> None:
        """All default requirements must be unimplemented."""
        reqs = default_requirements()
        self.assertGreater(len(reqs), 0)
        for req in reqs:
            self.assertEqual(req.status, "unimplemented")

    def test_interleavable_round_progress_requirement_exists(self) -> None:
        reqs = default_requirements()
        identifiers = [r.identifier for r in reqs]
        self.assertIn("interleavable_round_progress", identifiers)

    def test_gpu_execution_concurrency_requirement_exists(self) -> None:
        reqs = default_requirements()
        identifiers = [r.identifier for r in reqs]
        self.assertIn("gpu_execution_concurrency", identifiers)

    def test_safe_per_slot_ownership_requirement_exists(self) -> None:
        reqs = default_requirements()
        identifiers = [r.identifier for r in reqs]
        self.assertIn("safe_per_slot_ownership", identifiers)

    def test_each_requirement_cites_source(self) -> None:
        reqs = default_requirements()
        for req in reqs:
            self.assertTrue(req.source_citation)
            # Must reference a real source file
            self.assertTrue(
                ".cpp" in req.source_citation
                or ".cu" in req.source_citation
                or ".hpp" in req.source_citation,
                f"requirement {req.identifier} must cite a source file",
            )

    def test_each_requirement_has_description(self) -> None:
        reqs = default_requirements()
        for req in reqs:
            self.assertTrue(req.description)
            self.assertGreater(len(req.description), 20)



class ContractTest(unittest.TestCase):
    def test_build_contract_has_all_requirements(self) -> None:
        contract = build_contract()
        self.assertEqual(len(contract.requirements), 3)

    def test_contract_any_unimplemented(self) -> None:
        contract = build_contract()
        self.assertTrue(contract.any_unimplemented)

    def test_contract_not_all_satisfied(self) -> None:
        contract = build_contract()
        self.assertFalse(contract.all_satisfied)

    def test_contract_all_satisfied_when_all_satisfied(self) -> None:
        reqs = [
            FeasibilityRequirement(
                identifier="test",
                description="test description here",
                source_citation="test.py:1",
                status="satisfied",
            )
        ]
        contract = FeasibilityContract(requirements=reqs)
        self.assertTrue(contract.all_satisfied)
        self.assertFalse(contract.any_unimplemented)

    def test_requirement_accepts_empty_identifier(self) -> None:
        """FeasibilityRequirement is a simple dataclass — no
        validation on identifier. This is acceptable because the
        contract builder controls all requirements."""
        req = FeasibilityRequirement(
            identifier="",
            description="test description here",
            source_citation="test.py:1",
            status="unimplemented",
        )
        self.assertEqual(req.identifier, "")


class ReportTest(unittest.TestCase):
    def test_report_contains_feasibility_title(self) -> None:
        contract = build_contract()
        report = render_report(contract)
        self.assertIn("MULTISLOT_INGRESS_FEASIBILITY_CONTRACT", report)

    def test_report_states_not_performance_model(self) -> None:
        contract = build_contract()
        report = render_report(contract)
        self.assertIn("not a performance model", report.lower())

    def test_report_states_no_performance_estimate(self) -> None:
        contract = build_contract()
        report = render_report(contract)
        self.assertIn("No performance estimate", report)

    def test_report_no_speedup_ratio(self) -> None:
        """Report must not contain speedup_ratio or any performance estimate."""
        contract = build_contract()
        report = render_report(contract)
        self.assertNotIn("speedup_ratio", report)
        self.assertNotIn("speedup", report.lower())
        self.assertIn("CONCLUSION", report)
        self.assertIn("NOT feasible", report)

    def test_report_lists_each_requirement(self) -> None:
        contract = build_contract()
        report = render_report(contract)
        for req in contract.requirements:
            self.assertIn(req.identifier, report)
            self.assertIn(req.status, report)



class InputHardeningTest(unittest.TestCase):
    def test_validate_positive_int_rejects_bool(self) -> None:
        from spark_multislot_overlap_experiment import _validate_positive_int
        with self.assertRaises(ExperimentInputError):
            _validate_positive_int(True, "test")  # type: ignore[arg-type]

    def test_validate_positive_int_rejects_zero(self) -> None:
        from spark_multislot_overlap_experiment import _validate_positive_int
        with self.assertRaises(ExperimentInputError):
            _validate_positive_int(0, "test")

    def test_validate_positive_int_rejects_negative(self) -> None:
        from spark_multislot_overlap_experiment import _validate_positive_int
        with self.assertRaises(ExperimentInputError):
            _validate_positive_int(-1, "test")

    def test_validate_positive_int_rejects_float(self) -> None:
        from spark_multislot_overlap_experiment import _validate_positive_int
        with self.assertRaises(ExperimentInputError):
            _validate_positive_int(1.5, "test")  # type: ignore[arg-type]

    def test_validate_positive_int_accepts_valid(self) -> None:
        from spark_multislot_overlap_experiment import _validate_positive_int
        self.assertEqual(_validate_positive_int(5, "test"), 5)


if __name__ == "__main__":
    unittest.main()
