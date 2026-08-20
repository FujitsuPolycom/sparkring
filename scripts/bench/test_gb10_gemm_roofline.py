"""GPU-free tests for the dense BF16 GEMM rate measurement.

Everything here runs on a machine with no CUDA device. The device path is
covered only where it can be reached without one: the refusal that fires
when no device is visible. The timing loop itself is a manual gate on a
GB10 host.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import gb10_gemm_roofline as bench


def _samples(count: int = 8, base: float = 1.0) -> list[float]:
    """A timing sample with a known median and a known interquartile range."""

    return [base + index for index in range(count)]


def _report(results=None) -> dict:
    """A report built from synthetic timings, with no device involved."""

    if results is None:
        results = [bench.shape_result(shape, _samples()) for shape in bench.SHAPES]
    return bench.build_report(
        environment={
            "torch_version": "2.11.0",
            "torch_cuda_version": "13.2",
            "device_index": 0,
            "device_name": "NVIDIA GB10",
            "compute_capability": [12, 1],
            "multi_processor_count": 48,
            "total_memory_bytes": 128 * 1024**3,
            "dtype": "torch.bfloat16",
            "element_bytes": 2,
            "allow_bf16_reduced_precision_reduction": True,
            "platform": "Linux-aarch64",
            "python_version": "3.12.10",
        },
        clocks_before={"read": False, "reason": "nvidia-smi is not on PATH"},
        clocks_after={"read": False, "reason": "nvidia-smi is not on PATH"},
        floor_timing=bench.summarize([0.01, 0.02, 0.03, 0.04]),
        results=results,
        warmup=50,
        iterations=200,
    )


class ShapeArithmeticTest(unittest.TestCase):
    def test_flop_count_is_two_m_n_k(self) -> None:
        shape = bench.Shape("case", 40, 512, 6144, "specified")

        self.assertEqual(bench.flop_count(shape), 2 * 40 * 512 * 6144)

    def test_compulsory_bytes_counts_each_element_once_at_two_bytes(self) -> None:
        shape = bench.Shape("case", 40, 512, 6144, "specified")

        expected = 2 * (40 * 6144 + 6144 * 512 + 40 * 512)

        self.assertEqual(bench.compulsory_bytes(shape), expected)

    def test_arithmetic_intensity_is_flop_over_bytes(self) -> None:
        shape = bench.Shape("case", 128, 512, 6144, "specified")

        self.assertAlmostEqual(
            bench.arithmetic_intensity(shape),
            bench.flop_count(shape) / bench.compulsory_bytes(shape),
        )

    def test_skinny_shapes_have_far_lower_intensity_than_the_control(self) -> None:
        skinny = next(s for s in bench.SHAPES if s.m == 40)
        control = next(s for s in bench.SHAPES if s.role == "control")

        self.assertLess(
            bench.arithmetic_intensity(skinny),
            bench.arithmetic_intensity(control) / 10,
        )

    def test_output_tiles_round_up_in_both_dimensions(self) -> None:
        self.assertEqual(bench.output_tiles(bench.Shape("c", 40, 512, 6144, "x")), 4)
        self.assertEqual(bench.output_tiles(bench.Shape("c", 129, 512, 6144, "x")), 8)

    def test_tflops_converts_flop_and_milliseconds(self) -> None:
        # 1e12 FLOP in 1 ms is 1000 TFLOP/s.
        self.assertAlmostEqual(bench.tflops(10**12, 1.0), 1000.0)

    def test_gigabytes_per_second_converts_bytes_and_milliseconds(self) -> None:
        # 1e9 bytes in 1000 ms is 1 GB/s.
        self.assertAlmostEqual(bench.gigabytes_per_second(10**9, 1000.0), 1.0)

    def test_a_non_positive_elapsed_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bench.tflops(10**12, 0.0)
        with self.assertRaises(ValueError):
            bench.gigabytes_per_second(10**9, -1.0)


class SpecifiedShapesTest(unittest.TestCase):
    def test_the_three_specified_shapes_hold_k_6144_and_n_512(self) -> None:
        specified = [s for s in bench.SHAPES if s.role == "specified"]

        self.assertEqual(sorted(s.m for s in specified), [40, 128, 512])
        self.assertEqual({s.k for s in specified}, {6144})
        self.assertEqual({s.n for s in specified}, {512})

    def test_exactly_one_square_control_shape_accompanies_them(self) -> None:
        control = [s for s in bench.SHAPES if s.role == "control"]

        self.assertEqual(len(control), 1)
        self.assertEqual((control[0].m, control[0].n, control[0].k), (4096, 4096, 4096))

    def test_the_control_is_not_labelled_as_a_specified_shape(self) -> None:
        self.assertNotIn("control", {s.role for s in bench.SHAPES if s.m != 4096})

    def test_the_per_call_floor_is_the_smallest_possible_gemm(self) -> None:
        floor = bench.LAUNCH_FLOOR

        self.assertEqual((floor.m, floor.n, floor.k), (1, 1, 1))
        self.assertNotIn(floor, bench.SHAPES)


class DistributionTest(unittest.TestCase):
    def test_percentile_interpolates_between_neighbours(self) -> None:
        self.assertAlmostEqual(bench.percentile([0.0, 10.0], 0.5), 5.0)
        self.assertAlmostEqual(bench.percentile([0.0, 10.0], 0.25), 2.5)

    def test_percentile_ignores_input_order(self) -> None:
        self.assertAlmostEqual(bench.percentile([3.0, 1.0, 2.0], 0.5), 2.0)

    def test_percentile_endpoints_are_the_extremes(self) -> None:
        values = [4.0, 1.0, 9.0]

        self.assertAlmostEqual(bench.percentile(values, 0.0), 1.0)
        self.assertAlmostEqual(bench.percentile(values, 1.0), 9.0)

    def test_percentile_rejects_an_empty_sample_and_a_bad_fraction(self) -> None:
        with self.assertRaises(ValueError):
            bench.percentile([], 0.5)
        with self.assertRaises(ValueError):
            bench.percentile([1.0], 1.5)

    def test_summarize_reports_median_iqr_and_extremes(self) -> None:
        summary = bench.summarize([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(summary["samples"], 5)
        self.assertAlmostEqual(summary["min_ms"], 1.0)
        self.assertAlmostEqual(summary["max_ms"], 5.0)
        self.assertAlmostEqual(summary["median_ms"], 3.0)
        self.assertAlmostEqual(summary["p25_ms"], 2.0)
        self.assertAlmostEqual(summary["p75_ms"], 4.0)
        self.assertAlmostEqual(summary["iqr_ms"], 2.0)

    def test_summarize_reports_a_dispersion_measure_not_only_a_centre(self) -> None:
        summary = bench.summarize(_samples())

        for key in ("min_ms", "max_ms", "iqr_ms", "p25_ms", "p75_ms"):
            self.assertIn(key, summary)

    def test_summarize_rejects_an_empty_sample(self) -> None:
        with self.assertRaises(ValueError):
            bench.summarize([])


class ShapeResultTest(unittest.TestCase):
    def test_the_median_rate_is_the_flop_count_over_the_median_time(self) -> None:
        shape = bench.Shape("case", 40, 512, 6144, "specified")
        result = bench.shape_result(shape, [1.0, 2.0, 3.0])

        self.assertAlmostEqual(
            result["rate"]["tflops_at_median"],
            bench.flop_count(shape) / (2.0 * 1e-3) / 1e12,
        )

    def test_the_minimum_time_gives_the_highest_reported_rate(self) -> None:
        result = bench.shape_result(bench.SHAPES[0], _samples())

        self.assertGreater(
            result["rate"]["tflops_at_min"], result["rate"]["tflops_at_median"]
        )

    def test_the_result_carries_the_role_so_the_control_stays_labelled(self) -> None:
        control = next(s for s in bench.SHAPES if s.role == "control")

        self.assertEqual(bench.shape_result(control, _samples())["role"], "control")


class ClockStateTest(unittest.TestCase):
    def test_an_absent_nvidia_smi_is_reported_with_a_reason(self) -> None:
        state = bench.read_clock_state(0, which=lambda _name: None)

        self.assertFalse(state["read"])
        self.assertIn("PATH", state["reason"])

    def test_a_nonzero_exit_is_reported_with_the_last_line_of_output(self) -> None:
        def runner(_command, **_kwargs):
            return SimpleNamespace(returncode=9, stdout="", stderr="GPU is lost\n")

        state = bench.read_clock_state(
            0, which=lambda _name: "/usr/bin/nvidia-smi", runner=runner
        )

        self.assertFalse(state["read"])
        self.assertEqual(state["reason"], "GPU is lost")

    def test_a_failure_to_execute_is_reported_rather_than_raised(self) -> None:
        def runner(_command, **_kwargs):
            raise OSError("exec format error")

        state = bench.read_clock_state(
            0, which=lambda _name: "/usr/bin/nvidia-smi", runner=runner
        )

        self.assertFalse(state["read"])
        self.assertIn("exec format error", state["reason"])

    def test_pinned_application_clocks_are_reported_as_pinned(self) -> None:
        row = "NVIDIA GB10, 1400, 1400, 1400, Not Active, Enabled, 61.2, 140.0, 48"

        def runner(_command, **_kwargs):
            return SimpleNamespace(returncode=0, stdout=row + "\n", stderr="")

        state = bench.read_clock_state(
            0, which=lambda _name: "/usr/bin/nvidia-smi", runner=runner
        )

        self.assertTrue(state["read"])
        self.assertTrue(state["application_clocks_pinned"])
        self.assertEqual(state["fields"]["clocks.sm"], "1400")

    def test_unset_application_clocks_are_reported_as_not_pinned(self) -> None:
        row = "NVIDIA GB10, 1400, 1400, [N/A], Not Active, Enabled, 61.2, 140.0, 48"

        def runner(_command, **_kwargs):
            return SimpleNamespace(returncode=0, stdout=row + "\n", stderr="")

        state = bench.read_clock_state(
            0, which=lambda _name: "/usr/bin/nvidia-smi", runner=runner
        )

        self.assertTrue(state["read"])
        self.assertFalse(state["application_clocks_pinned"])
        self.assertIn("not pinned", state["lock_note"])

    def test_an_unexpected_field_count_is_reported_rather_than_misparsed(self) -> None:
        def runner(_command, **_kwargs):
            return SimpleNamespace(returncode=0, stdout="NVIDIA GB10, 1400\n", stderr="")

        state = bench.read_clock_state(
            0, which=lambda _name: "/usr/bin/nvidia-smi", runner=runner
        )

        self.assertFalse(state["read"])
        self.assertIn("fields", state["reason"])


class ReportShapeTest(unittest.TestCase):
    def test_the_report_names_its_schema_and_both_formulas(self) -> None:
        report = _report()

        self.assertEqual(report["schema"], bench.SCHEMA)
        self.assertEqual(report["measurement"]["flop_formula"], "flop = 2 * M * N * K")
        self.assertEqual(
            report["measurement"]["bytes_formula"], "bytes = 2 * (M*K + K*N + M*N)"
        )

    def test_the_report_records_the_conditions_a_reader_needs(self) -> None:
        report = _report()

        for key in (
            "torch_version",
            "torch_cuda_version",
            "device_name",
            "dtype",
            "compute_capability",
        ):
            self.assertIn(key, report["environment"])
        self.assertIn("before", report["clocks"])
        self.assertIn("after", report["clocks"])

    def test_an_unread_clock_state_states_a_reason_rather_than_being_absent(
        self,
    ) -> None:
        report = _report()

        self.assertFalse(report["clocks"]["before"]["read"])
        self.assertTrue(report["clocks"]["before"]["reason"])

    def test_the_report_states_it_is_single_process_and_not_distributed(self) -> None:
        report = _report()

        self.assertTrue(report["measurement"]["single_process"])
        self.assertFalse(report["measurement"]["distributed"])

    def test_the_report_carries_every_shape_and_the_per_call_floor(self) -> None:
        report = _report()

        self.assertEqual(len(report["shapes"]), len(bench.SHAPES))
        self.assertEqual(report["per_call_floor"]["m"], 1)
        self.assertIn("median_ms", report["per_call_floor"]["timing"])

    def test_the_report_is_json_serializable_and_round_trips(self) -> None:
        report = _report()

        restored = json.loads(json.dumps(report, sort_keys=True))

        self.assertEqual(restored["schema"], bench.SCHEMA)
        self.assertEqual(len(restored["shapes"]), len(bench.SHAPES))

    def test_emit_json_writes_the_document_to_a_path(self) -> None:
        report = _report()

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            bench.emit_json(report, str(destination))
            restored = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(restored["schema"], bench.SCHEMA)

    def test_emit_json_writes_to_stdout_for_a_dash(self) -> None:
        stream = io.StringIO()

        with redirect_stdout(stream):
            bench.emit_json(_report(), "-")

        self.assertEqual(json.loads(stream.getvalue())["schema"], bench.SCHEMA)


class TextRenderTest(unittest.TestCase):
    def test_the_text_report_prints_the_flop_formula_with_the_numbers(self) -> None:
        rendered = bench.render_text(_report())

        self.assertIn("flop = 2 * M * N * K", rendered)
        self.assertIn("bytes = 2 * (M*K + K*N + M*N)", rendered)

    def test_the_text_report_labels_the_control_row(self) -> None:
        rendered = bench.render_text(_report())

        self.assertIn("m4096_k4096_n4096", rendered)
        self.assertIn("control", rendered)

    def test_the_text_report_says_when_the_clock_state_was_not_read(self) -> None:
        rendered = bench.render_text(_report())

        self.assertIn("NOT READ", rendered)

    def test_the_text_report_prints_both_a_median_and_a_dispersion_column(
        self,
    ) -> None:
        rendered = bench.render_text(_report())

        self.assertIn("median ms", rendered)
        self.assertIn("IQR ms", rendered)
        self.assertIn("min ms", rendered)
        self.assertIn("max ms", rendered)

    def test_the_text_report_prints_arithmetic_intensity_per_shape(self) -> None:
        rendered = bench.render_text(_report())

        self.assertIn("AI f/B", rendered)

    def test_the_shape_table_renders_without_a_device(self) -> None:
        rendered = bench.render_shape_table()

        self.assertIn("m40_k6144_n512", rendered)
        self.assertIn("per_call_floor", rendered)
        self.assertIn("flop = 2 * M * N * K", rendered)


class ArgumentTest(unittest.TestCase):
    def test_defaults_measure_device_zero_with_warmup_and_many_trials(self) -> None:
        arguments = bench.parse_args([])

        self.assertEqual(arguments.device, 0)
        self.assertGreaterEqual(arguments.warmup, 1)
        self.assertGreaterEqual(arguments.iterations, 100)
        self.assertIsNone(arguments.json)
        self.assertFalse(arguments.list_shapes)

    def test_json_and_device_and_counts_are_accepted(self) -> None:
        arguments = bench.parse_args(
            ["--device", "1", "--warmup", "5", "--iterations", "20", "--json", "-"]
        )

        self.assertEqual(arguments.device, 1)
        self.assertEqual(arguments.warmup, 5)
        self.assertEqual(arguments.iterations, 20)
        self.assertEqual(arguments.json, "-")

    def test_zero_warmup_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["--warmup", "0"])

        self.assertEqual(raised.exception.code, 2)

    def test_too_few_iterations_for_an_interquartile_range_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["--iterations", "3"])

        self.assertEqual(raised.exception.code, 2)

    def test_list_shapes_measures_nothing_and_succeeds(self) -> None:
        def refuse() -> None:
            raise AssertionError("--list-shapes must not import torch")

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = bench.main(["--list-shapes"], load_torch=refuse)

        self.assertEqual(code, bench.EXIT_OK)
        self.assertIn("m4096_k4096_n4096", stream.getvalue())


class NoDeviceTest(unittest.TestCase):
    """The path this repository's development machines actually take."""

    def test_require_cuda_device_refuses_when_no_device_is_visible(self) -> None:
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.require_cuda_device(torch_module, 0)

        self.assertIn("no CUDA device", str(raised.exception))

    def test_require_cuda_device_refuses_an_index_that_does_not_exist(self) -> None:
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        )

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.require_cuda_device(torch_module, 3)

        self.assertIn("does not exist", str(raised.exception))

    def test_main_exits_non_zero_and_explains_when_no_device_is_visible(self) -> None:
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )
        errors = io.StringIO()

        with redirect_stdout(io.StringIO()), redirect_stderr(errors):
            code = bench.main([], load_torch=lambda: torch_module)

        self.assertEqual(code, bench.EXIT_UNAVAILABLE)
        self.assertNotEqual(code, 0)
        self.assertIn("no measurement taken", errors.getvalue())
        self.assertIn("no CUDA device", errors.getvalue())

    def test_main_exits_non_zero_and_explains_when_torch_is_absent(self) -> None:
        def refuse() -> None:
            raise bench.MeasurementUnavailable("torch is not importable")

        errors = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errors):
            code = bench.main([], load_torch=refuse)

        self.assertEqual(code, bench.EXIT_UNAVAILABLE)
        self.assertIn("torch is not importable", errors.getvalue())

    def test_no_report_is_printed_to_stdout_when_no_measurement_was_taken(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            bench.main(
                [],
                load_torch=lambda: SimpleNamespace(
                    cuda=SimpleNamespace(
                        is_available=lambda: False, device_count=lambda: 0
                    )
                ),
            )

        self.assertEqual(output.getvalue(), "")


class SafetyClassTest(unittest.TestCase):
    def test_the_module_docstring_declares_a_safety_class(self) -> None:
        self.assertIn("Safety class: OFFLINE", bench.__doc__ or "")

    def test_the_module_docstring_states_the_flop_formula(self) -> None:
        self.assertIn("2 * M * N * K", (bench.__doc__ or "") + bench.FLOP_FORMULA)


if __name__ == "__main__":
    unittest.main()
