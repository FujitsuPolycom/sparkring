"""GPU-free tests for the collective critical-path attribution arithmetic.

Everything here runs on a machine with no CUDA device, no fabric, and no
capture. The module under test takes no measurement of its own, so every
behavior it has is reachable from synthetic documents: the arithmetic, the
schema refusals, the comparability contract, the threshold floor, and the
exit codes.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import collective_attribution as attribution


def _ranks(
    *,
    residency: float,
    gated: bool,
    world_size: int = 4,
    skewed: bool = True,
) -> dict[str, dict[str, list[float]]]:
    """Four ranks with a known median residency and a known gate pattern."""

    out: dict[str, dict[str, list[float]]] = {}
    for rank in range(world_size):
        record: dict[str, list[float]] = {
            "residency_us": [residency - 1.0, residency, residency + 1.0]
        }
        if gated:
            first = 10.0 + (rank * 10.0 if skewed else 0.0)
            record["gate_first_us"] = [first - 1.0, first, first + 1.0]
            record["gate_second_us"] = [9.0, 10.0, 11.0]
        out[str(rank)] = record
    return out


def _instance(**overrides) -> dict:
    base = {
        "family": "tp4_all_reduce",
        "communicator": "tp:0",
        "world_size": 4,
        "shape": [512, 4096],
        "element_bytes": 2,
        "step": 1,
        "ordinal": 1,
        "wire_bytes_multiplier": 1.5,
        "multiplier_basis": "ring all-reduce, 2(N-1)/N per rank at N=4",
        "occurrences": 2,
        "ranks": _ranks(residency=400.0, gated=True),
    }
    base.update(overrides)
    return base


def _capture(arm: str, **overrides) -> dict:
    gated = arm == attribution.GATED_ARM
    document = {
        "schema": attribution.CAPTURE_SCHEMA,
        "arm": arm,
        "layer": attribution.DEVICE_LAYER,
        "session": f"{arm}-session",
        "link": {"rate_gbit_per_second": 200.0, "rate_basis": "nameplate"},
        "instances": [
            _instance(
                ranks=_ranks(residency=400.0 if gated else 520.0, gated=gated)
            )
        ],
    }
    document.update(overrides)
    return document


class SummaryTest(unittest.TestCase):
    def test_quantile_returns_an_observed_sample(self) -> None:
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        self.assertEqual(attribution.quantile(values, 0.0), 1.0)
        self.assertEqual(attribution.quantile(values, 1.0), 5.0)
        self.assertIn(attribution.quantile(values, 0.5), values)

    def test_quantile_refuses_an_empty_sample_set(self) -> None:
        with self.assertRaises(ValueError):
            attribution.quantile([], 0.5)

    def test_quantile_refuses_a_fraction_outside_the_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            attribution.quantile([1.0], 1.5)

    def test_summary_reports_count_median_and_spread(self) -> None:
        summary = attribution.summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary.count, 4)
        self.assertEqual(summary.median, 2.5)
        self.assertEqual(summary.minimum, 1.0)
        self.assertEqual(summary.maximum, 4.0)
        self.assertEqual(summary.iqr, summary.q3 - summary.q1)

    def test_summary_dictionary_carries_the_spread_not_only_the_median(self) -> None:
        payload = attribution.summarize([1.0, 2.0, 3.0, 4.0]).to_dict()
        for field in ("count", "median", "q1", "q3", "iqr", "min", "max"):
            self.assertIn(field, payload)

    def test_relative_percent_refuses_a_zero_reference(self) -> None:
        with self.assertRaises(ValueError):
            attribution.relative_percent(1.0, 0.0)

    def test_spread_percent_is_peak_to_peak_over_the_median(self) -> None:
        self.assertAlmostEqual(attribution.spread_percent([90.0, 100.0, 110.0]), 20.0)


class FloorTest(unittest.TestCase):
    def test_payload_bytes_multiplies_shape_by_element_width(self) -> None:
        self.assertEqual(attribution.payload_bytes([512, 4096], 2), 4_194_304)

    def test_payload_bytes_refuses_a_nonpositive_element_width(self) -> None:
        with self.assertRaises(ValueError):
            attribution.payload_bytes([4], 0)

    def test_payload_bytes_refuses_an_empty_shape(self) -> None:
        with self.assertRaises(ValueError):
            attribution.payload_bytes([], 2)

    def test_floor_is_wire_bits_over_the_stated_rate(self) -> None:
        # 1,000,000 bytes at multiplier 1 over 8 Gb/s is exactly 1,000 us.
        self.assertAlmostEqual(
            attribution.floor_microseconds(1_000_000, 1.0, 8.0), 1_000.0
        )

    def test_floor_scales_with_the_wire_multiplier(self) -> None:
        single = attribution.floor_microseconds(1_000_000, 1.0, 8.0)
        doubled = attribution.floor_microseconds(1_000_000, 2.0, 8.0)
        self.assertAlmostEqual(doubled, 2.0 * single)

    def test_floor_refuses_a_nonpositive_rate_or_multiplier(self) -> None:
        with self.assertRaises(ValueError):
            attribution.floor_microseconds(1, 1.0, 0.0)
        with self.assertRaises(ValueError):
            attribution.floor_microseconds(1, 0.0, 8.0)


class RepetitionPlanTest(unittest.TestCase):
    def test_count_grows_with_the_square_of_the_dispersion_ratio(self) -> None:
        # 1.96 * 15.8 / 10 is 3.097; its square is 9.59, which ceilings to 10.
        self.assertEqual(attribution.required_repetitions(15.8, 10.0), 10)

    def test_a_wider_dispersion_requires_more_repetitions(self) -> None:
        self.assertGreater(
            attribution.required_repetitions(30.0, 10.0),
            attribution.required_repetitions(15.0, 10.0),
        )

    def test_count_never_falls_below_the_module_minimum(self) -> None:
        self.assertEqual(
            attribution.required_repetitions(0.1, 50.0),
            attribution.MINIMUM_REPETITIONS,
        )

    def test_plan_refuses_a_nonpositive_target(self) -> None:
        with self.assertRaises(ValueError):
            attribution.required_repetitions(10.0, 0.0)


class VerdictTest(unittest.TestCase):
    def test_a_difference_below_the_threshold_is_indeterminate(self) -> None:
        label, percent = attribution.verdict(2.0, 100.0, 5.0)
        self.assertEqual(label, "indeterminate")
        self.assertAlmostEqual(percent, 2.0)

    def test_a_difference_above_the_threshold_keeps_its_direction(self) -> None:
        self.assertEqual(attribution.verdict(9.0, 100.0, 5.0)[0], "higher")
        self.assertEqual(attribution.verdict(-9.0, 100.0, 5.0)[0], "lower")

    def test_a_difference_exactly_at_the_threshold_is_reportable(self) -> None:
        self.assertEqual(attribution.verdict(5.0, 100.0, 5.0)[0], "higher")


class ThresholdFloorTest(unittest.TestCase):
    def test_each_layer_defaults_to_its_own_floor(self) -> None:
        self.assertEqual(
            attribution.resolve_detect_percent(attribution.DEVICE_LAYER, None), 5.0
        )
        self.assertEqual(
            attribution.resolve_detect_percent(attribution.END_TO_END_LAYER, None),
            10.0,
        )

    def test_a_threshold_may_be_raised(self) -> None:
        self.assertEqual(
            attribution.resolve_detect_percent(attribution.DEVICE_LAYER, 12.0), 12.0
        )

    def test_a_threshold_below_the_floor_is_refused(self) -> None:
        with self.assertRaises(attribution.ThresholdRefused):
            attribution.resolve_detect_percent(attribution.DEVICE_LAYER, 1.0)
        with self.assertRaises(attribution.ThresholdRefused):
            attribution.resolve_detect_percent(attribution.END_TO_END_LAYER, 6.0)


class SkewTest(unittest.TestCase):
    def test_skew_subtracts_the_gate_from_itself(self) -> None:
        self.assertAlmostEqual(
            attribution.skew_microseconds([40.0, 41.0, 42.0], [9.0, 10.0, 11.0]),
            31.0,
        )

    def test_skew_never_goes_negative(self) -> None:
        self.assertEqual(
            attribution.skew_microseconds([5.0], [10.0]),
            0.0,
        )


class ExposureFitTest(unittest.TestCase):
    def test_a_fully_exposed_region_fits_a_slope_of_one(self) -> None:
        points = [(0.0, 1_000_000.0), (100_000.0, 1_100_000.0), (200_000.0, 1_200_000.0)]
        fit = attribution.fit_exposure(points, 10.0)
        self.assertAlmostEqual(fit.slope, 1.0)
        self.assertTrue(fit.determinate)

    def test_a_fully_overlapped_region_fits_a_slope_of_zero(self) -> None:
        points = [(0.0, 1_000_000.0), (100_000.0, 1_000_000.0), (200_000.0, 1_000_000.0)]
        fit = attribution.fit_exposure(points, 10.0)
        self.assertAlmostEqual(fit.slope, 0.0)

    def test_a_sweep_inside_the_noise_floor_is_indeterminate(self) -> None:
        points = [(0.0, 1_000_000.0), (100.0, 1_000_100.0), (200.0, 1_000_200.0)]
        fit = attribution.fit_exposure(points, 10.0)
        self.assertFalse(fit.determinate)
        self.assertIn("below", fit.reason)

    def test_a_fit_requires_three_points(self) -> None:
        with self.assertRaises(ValueError):
            attribution.fit_exposure([(0.0, 1.0), (1.0, 2.0)], 10.0)

    def test_a_fit_refuses_a_negative_injected_delay(self) -> None:
        with self.assertRaises(ValueError):
            attribution.fit_exposure(
                [(-1.0, 1.0), (0.0, 2.0), (1.0, 3.0)], 10.0
            )

    def test_identical_delays_have_no_slope(self) -> None:
        with self.assertRaises(ValueError):
            attribution.fit_exposure([(5.0, 1.0), (5.0, 2.0), (5.0, 3.0)], 10.0)


class CaptureParsingTest(unittest.TestCase):
    def test_a_valid_gated_capture_parses(self) -> None:
        capture = attribution.parse_capture(_capture(attribution.GATED_ARM))
        self.assertEqual(capture.arm, attribution.GATED_ARM)
        self.assertEqual(len(capture.instances), 1)
        self.assertEqual(capture.instances[0].payload_bytes, 4_194_304)

    def test_a_wrong_schema_is_refused(self) -> None:
        document = _capture(attribution.GATED_ARM, schema="something-else/v1")
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)

    def test_an_unknown_layer_is_refused(self) -> None:
        document = _capture(attribution.GATED_ARM, layer="wall")
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)

    def test_a_rate_basis_must_state_nameplate_or_measured(self) -> None:
        document = _capture(attribution.GATED_ARM)
        document["link"]["rate_basis"] = "assumed"
        with self.assertRaises(attribution.DocumentInvalid) as raised:
            attribution.parse_capture(document)
        self.assertIn("nameplate", str(raised.exception))

    def test_a_missing_wire_multiplier_is_refused_rather_than_guessed(self) -> None:
        instance = _instance()
        del instance["wire_bytes_multiplier"]
        document = _capture(attribution.GATED_ARM, instances=[instance])
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)

    def test_a_multiplier_without_a_stated_basis_is_refused(self) -> None:
        instance = _instance(multiplier_basis="")
        document = _capture(attribution.GATED_ARM, instances=[instance])
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)

    def test_rank_count_must_match_the_declared_world_size(self) -> None:
        instance = _instance(ranks=_ranks(residency=200.0, gated=True, world_size=3))
        document = _capture(attribution.GATED_ARM, instances=[instance])
        with self.assertRaises(attribution.DocumentInvalid) as raised:
            attribution.parse_capture(document)
        self.assertIn("world_size", str(raised.exception))

    def test_a_gated_arm_requires_both_gate_regions(self) -> None:
        ranks = _ranks(residency=400.0, gated=True)
        del ranks["0"]["gate_second_us"]
        document = _capture(attribution.GATED_ARM, instances=[_instance(ranks=ranks)])
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)

    def test_a_naked_arm_may_not_carry_gate_timings(self) -> None:
        document = _capture(attribution.NAKED_ARM)
        document["instances"][0]["ranks"]["0"]["gate_first_us"] = [1.0]
        with self.assertRaises(attribution.DocumentInvalid) as raised:
            attribution.parse_capture(document)
        self.assertIn("no gate to time", str(raised.exception))

    def test_gate_and_collective_sample_counts_must_agree(self) -> None:
        ranks = _ranks(residency=400.0, gated=True)
        ranks["0"]["gate_first_us"] = [1.0]
        document = _capture(attribution.GATED_ARM, instances=[_instance(ranks=ranks)])
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)

    def test_a_repeated_instance_key_is_refused(self) -> None:
        document = _capture(
            attribution.GATED_ARM, instances=[_instance(), _instance()]
        )
        with self.assertRaises(attribution.DocumentInvalid) as raised:
            attribution.parse_capture(document)
        self.assertIn("occurrences", str(raised.exception))

    def test_a_nonfinite_sample_is_refused(self) -> None:
        ranks = _ranks(residency=400.0, gated=True)
        ranks["0"]["residency_us"] = [float("inf"), 1.0, 2.0]
        document = _capture(attribution.GATED_ARM, instances=[_instance(ranks=ranks)])
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)

    def test_a_negative_sample_is_refused(self) -> None:
        ranks = _ranks(residency=400.0, gated=True)
        ranks["0"]["residency_us"] = [-1.0, 1.0, 2.0]
        document = _capture(attribution.GATED_ARM, instances=[_instance(ranks=ranks)])
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_capture(document)


class ExposureParsingTest(unittest.TestCase):
    def test_a_valid_sweep_reduces_each_point_to_its_median(self) -> None:
        layer, points = attribution.parse_exposure(
            attribution.example_documents()["exposure"]
        )
        self.assertEqual(layer, attribution.END_TO_END_LAYER)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0][0], 0.0)

    def test_fewer_than_three_distinct_delays_are_refused(self) -> None:
        document = {
            "schema": attribution.EXPOSURE_SCHEMA,
            "layer": attribution.END_TO_END_LAYER,
            "samples": [
                {"injected_delay_us": 0.0, "wall_us": [1.0]},
                {"injected_delay_us": 0.0, "wall_us": [2.0]},
                {"injected_delay_us": 1.0, "wall_us": [3.0]},
            ],
        }
        with self.assertRaises(attribution.DocumentInvalid):
            attribution.parse_exposure(document)


class ComparabilityTest(unittest.TestCase):
    def _pair(self) -> tuple[attribution.Capture, attribution.Capture]:
        return (
            attribution.parse_capture(_capture(attribution.GATED_ARM)),
            attribution.parse_capture(_capture(attribution.NAKED_ARM)),
        )

    def test_a_matched_pair_is_comparable(self) -> None:
        gated, naked = self._pair()
        attribution.require_comparable(gated, naked)

    def test_two_gated_arms_are_not_a_pair(self) -> None:
        gated = attribution.parse_capture(_capture(attribution.GATED_ARM))
        with self.assertRaises(attribution.NotComparable):
            attribution.require_comparable(gated, gated)

    def test_one_session_cannot_supply_both_arms(self) -> None:
        gated, _ = self._pair()
        naked = attribution.parse_capture(
            _capture(attribution.NAKED_ARM, session=gated.session)
        )
        with self.assertRaises(attribution.NotComparable) as raised:
            attribution.require_comparable(gated, naked)
        self.assertIn("two collections", str(raised.exception))

    def test_different_inventories_are_not_comparable(self) -> None:
        gated, _ = self._pair()
        naked = attribution.parse_capture(
            _capture(
                attribution.NAKED_ARM,
                instances=[
                    _instance(
                        ordinal=7, ranks=_ranks(residency=520.0, gated=False)
                    )
                ],
            )
        )
        with self.assertRaises(attribution.NotComparable) as raised:
            attribution.require_comparable(gated, naked)
        self.assertIn("inventories", str(raised.exception))

    def test_different_link_rates_are_not_comparable(self) -> None:
        gated, _ = self._pair()
        naked_document = _capture(attribution.NAKED_ARM)
        naked_document["link"]["rate_gbit_per_second"] = 100.0
        naked = attribution.parse_capture(naked_document)
        with self.assertRaises(attribution.NotComparable):
            attribution.require_comparable(gated, naked)


class ReportTest(unittest.TestCase):
    def _report(self, detect_percent: float = 5.0) -> dict:
        gated = attribution.parse_capture(_capture(attribution.GATED_ARM))
        naked = attribution.parse_capture(_capture(attribution.NAKED_ARM))
        return attribution.build_report(gated, naked, detect_percent, None)

    def test_totals_order_floor_below_transport_below_residency(self) -> None:
        totals = self._report()["totals_seconds"]
        self.assertLess(totals["floor"], totals["transport_ceiling"])
        self.assertLess(totals["transport_ceiling"], totals["residency_ceiling"])

    def test_totals_count_each_instance_by_its_occurrences(self) -> None:
        report = self._report()
        row = report["instances"][0]
        self.assertAlmostEqual(
            report["totals_seconds"]["transport_ceiling"] * 1e6,
            row["occurrences"] * row["slowest_rank_transport_us"],
            places=3,
        )

    def test_gating_removes_the_difference_between_the_two_ceilings(self) -> None:
        totals = self._report()["totals_seconds"]
        self.assertAlmostEqual(
            totals["removed_by_gating"],
            totals["residency_ceiling"] - totals["transport_ceiling"],
            places=6,
        )

    def test_a_removal_above_the_threshold_keeps_its_direction(self) -> None:
        # 520 us naked against 400 us gated is a 23% removal, well above 5%.
        self.assertEqual(
            self._report()["removed_by_gating"]["verdict"], "higher"
        )

    def test_a_removal_below_the_threshold_is_indeterminate(self) -> None:
        gated = attribution.parse_capture(_capture(attribution.GATED_ARM))
        naked = attribution.parse_capture(
            _capture(
                attribution.NAKED_ARM,
                instances=[_instance(ranks=_ranks(residency=404.0, gated=False))],
            )
        )
        report = attribution.build_report(gated, naked, 5.0, None)
        self.assertEqual(report["removed_by_gating"]["verdict"], "indeterminate")

    def test_per_rank_skew_is_reported_and_rises_with_rank(self) -> None:
        ranks = self._report()["instances"][0]["ranks"]
        self.assertAlmostEqual(ranks["0"]["skew_us"], 0.0)
        self.assertAlmostEqual(ranks["3"]["skew_us"], 30.0)

    def test_cross_rank_agreement_passes_when_gated_medians_agree(self) -> None:
        report = self._report()
        self.assertTrue(report["validity"]["cross_rank_agreement"])
        self.assertEqual(report["validity"]["cross_rank_failures"], [])

    def test_cross_rank_disagreement_is_reported_as_a_validity_failure(self) -> None:
        ranks = _ranks(residency=400.0, gated=True)
        ranks["3"]["residency_us"] = [900.0, 901.0, 902.0]
        gated = attribution.parse_capture(
            _capture(attribution.GATED_ARM, instances=[_instance(ranks=ranks)])
        )
        naked = attribution.parse_capture(_capture(attribution.NAKED_ARM))
        report = attribution.build_report(gated, naked, 5.0, None)
        self.assertFalse(report["validity"]["cross_rank_agreement"])

    def test_a_gate_that_costs_too_much_is_reported_as_a_validity_failure(
        self,
    ) -> None:
        small = {"shape": [16, 576]}
        gated = attribution.parse_capture(
            _capture(
                attribution.GATED_ARM,
                instances=[
                    _instance(ranks=_ranks(residency=20.0, gated=True), **small)
                ],
            )
        )
        naked = attribution.parse_capture(
            _capture(
                attribution.NAKED_ARM,
                instances=[
                    _instance(ranks=_ranks(residency=30.0, gated=False), **small)
                ],
            )
        )
        report = attribution.build_report(gated, naked, 5.0, None)
        self.assertFalse(report["validity"]["gate_cost_acceptable"])

    def test_a_nameplate_rate_is_flagged_as_not_measured(self) -> None:
        self.assertFalse(self._report()["validity"]["rate_basis_measured"])

    def test_an_exposure_section_states_its_extrapolation(self) -> None:
        gated = attribution.parse_capture(_capture(attribution.GATED_ARM))
        naked = attribution.parse_capture(_capture(attribution.NAKED_ARM))
        fit = attribution.fit_exposure(
            [(0.0, 1_000_000.0), (100_000.0, 1_050_000.0), (200_000.0, 1_100_000.0)],
            10.0,
        )
        report = attribution.build_report(gated, naked, 5.0, fit)
        self.assertAlmostEqual(report["exposure"]["slope"], 0.5)
        self.assertIn("linear", report["exposure"]["extrapolation_note"])
        self.assertAlmostEqual(
            report["exposure"]["exposed_transport_seconds"],
            0.5 * report["totals_seconds"]["transport_ceiling"],
            places=6,
        )

    def test_an_indeterminate_exposure_reports_no_exposed_seconds(self) -> None:
        gated = attribution.parse_capture(_capture(attribution.GATED_ARM))
        naked = attribution.parse_capture(_capture(attribution.NAKED_ARM))
        fit = attribution.fit_exposure(
            [(0.0, 1_000_000.0), (100.0, 1_000_050.0), (200.0, 1_000_100.0)], 10.0
        )
        report = attribution.build_report(gated, naked, 5.0, fit)
        self.assertIsNone(report["exposure"]["exposed_transport_seconds"])

    def test_the_rendered_report_states_its_scope(self) -> None:
        text = attribution.render_report(self._report())
        self.assertIn("does not identify a critical path", text)
        self.assertIn(attribution.FLOOR_FORMULA, text)


class ExampleDocumentTest(unittest.TestCase):
    def test_every_example_document_validates(self) -> None:
        examples = attribution.example_documents()
        attribution.parse_capture(examples["gated"])
        attribution.parse_capture(examples["naked"])
        attribution.parse_exposure(examples["exposure"])

    def test_the_example_pair_is_comparable(self) -> None:
        examples = attribution.example_documents()
        attribution.require_comparable(
            attribution.parse_capture(examples["gated"]),
            attribution.parse_capture(examples["naked"]),
        )


class CommandLineTest(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = attribution.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _write(self, directory: Path, name: str, document: dict) -> str:
        path = directory / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return str(path)

    def test_print_schema_emits_documents_that_validate(self) -> None:
        code, out, _ = self._run(["--print-schema"])
        self.assertEqual(code, attribution.EXIT_OK)
        parsed = json.loads(out)
        attribution.parse_capture(parsed["gated"])
        attribution.parse_capture(parsed["naked"])

    def test_plan_prints_a_repetition_count_and_its_formula(self) -> None:
        code, out, _ = self._run(
            ["--plan", "--dispersion-percent", "15.8", "--detect-percent", "10"]
        )
        self.assertEqual(code, attribution.EXIT_OK)
        self.assertIn("required repetitions", out)
        self.assertIn("10", out)

    def test_a_missing_input_returns_the_input_exit_code(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gated = self._write(root, "g.json", _capture(attribution.GATED_ARM))
            code, _, err = self._run(
                ["--gated", gated, "--naked", str(root / "absent.json")]
            )
        self.assertEqual(code, attribution.EXIT_INPUT_MISSING)
        self.assertIn("input unavailable", err)

    def test_malformed_json_returns_the_invalid_exit_code(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gated = self._write(root, "g.json", _capture(attribution.GATED_ARM))
            broken = root / "n.json"
            broken.write_text("{not json", encoding="utf-8")
            code, _, err = self._run(["--gated", gated, "--naked", str(broken)])
        self.assertEqual(code, attribution.EXIT_INVALID_DOCUMENT)
        self.assertIn("invalid document", err)

    def test_swapped_arms_return_the_not_comparable_exit_code(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gated = self._write(root, "g.json", _capture(attribution.GATED_ARM))
            naked = self._write(root, "n.json", _capture(attribution.NAKED_ARM))
            code, _, err = self._run(["--gated", naked, "--naked", gated])
        self.assertEqual(code, attribution.EXIT_NOT_COMPARABLE)
        self.assertIn("arm", err)

    def test_a_threshold_below_the_floor_is_refused_at_the_command_line(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gated = self._write(root, "g.json", _capture(attribution.GATED_ARM))
            naked = self._write(root, "n.json", _capture(attribution.NAKED_ARM))
            code, _, err = self._run(
                ["--gated", gated, "--naked", naked, "--detect-percent", "0.5"]
            )
        self.assertEqual(code, attribution.EXIT_NOT_COMPARABLE)
        self.assertIn("floor", err)

    def test_a_matched_pair_reports_and_writes_json(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gated = self._write(root, "g.json", _capture(attribution.GATED_ARM))
            naked = self._write(root, "n.json", _capture(attribution.NAKED_ARM))
            destination = root / "report.json"
            code, out, _ = self._run(
                ["--gated", gated, "--naked", naked, "--json", str(destination)]
            )
            written = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(code, attribution.EXIT_OK)
        self.assertIn("Per-request totals", out)
        self.assertEqual(written["schema"], attribution.REPORT_SCHEMA)

    def test_plan_without_its_inputs_is_rejected_by_the_parser(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                attribution.parse_args(["--plan"])

    def test_an_analysis_without_both_arms_is_rejected_by_the_parser(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                attribution.parse_args(["--gated", "only.json"])


class SafetyClassTest(unittest.TestCase):
    def test_the_module_docstring_declares_a_safety_class(self) -> None:
        self.assertIn("Safety class: OFFLINE", attribution.__doc__ or "")

    def test_the_module_docstring_names_the_design_document(self) -> None:
        self.assertIn(
            "docs/COLLECTIVE_CRITICAL_PATH_MEASUREMENT.md",
            attribution.__doc__ or "",
        )


if __name__ == "__main__":
    unittest.main()
