"""GPU-free tests for the mixed-bitrate Trellis MoE measurement and sweep.

Everything here runs on a machine with no CUDA device and without the b12x
kernel package installed. Covered: the flop and compressed-byte arithmetic,
the timing statistics, sweep enumeration and its cap, the skip path that
keeps a failing configuration from ending a sweep, signature discovery and
its refusals, the module-path check, report shape and rendering, argument
parsing, and the refusal that fires when no device is visible.

The timed loop is driven here against a stand-in for Torch, which fixes the
loop's structure - warmup count, pool cycling, one event pair per call, and
the output checks - without asserting anything about kernel performance.
Timing the kernel itself is a manual gate on a GB10 host.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import mixed_trellis_roofline as bench

GEOMETRY = bench.DEPLOYED_GEOMETRY


def _samples(count: int = 8, base: float = 1.0) -> list[float]:
    """A timing sample with a known median and a known interquartile range."""

    return [base + index for index in range(count)]


def _plan(route_slots: int = 320, max_m_blocks: int = 40) -> dict[str, int]:
    return {"route_slots": route_slots, "max_m_blocks": max_m_blocks}


def _size_result(
    size_m: int = 40,
    configuration: bench.Configuration = bench.DEPLOYED_CONFIGURATION,
    samples: list[float] | None = None,
) -> dict:
    return bench.size_result(
        geometry=GEOMETRY,
        size_m=size_m,
        configuration=configuration,
        samples_ms=samples or _samples(),
        selected_tier0=190,
        selected_tier1=60,
        plan=_plan(),
    )


def _arguments(*extra: str) -> object:
    return bench.parse_args(["measure", *extra])


def _environment() -> dict:
    return {
        "torch_version": "2.11.0",
        "torch_cuda_version": "13.2",
        "device_index": 0,
        "device_name": "NVIDIA GB10",
        "compute_capability": [12, 1],
        "multi_processor_count": 48,
        "shared_memory_per_block_optin": 232448,
        "total_memory_bytes": 128 * 1024**3,
        "platform": "Linux-aarch64",
        "python_version": "3.12.10",
    }


def _module_record() -> dict:
    return {
        "expected": bench.KERNEL_MODULE_PATH,
        "resolved": bench.KERNEL_MODULE_PATH,
        "resolved_realpath": bench.KERNEL_MODULE_PATH,
        "matches": True,
        "reason": "",
    }


def _discovery() -> dict:
    return {
        "kernel_module": bench.KERNEL_MODULE,
        "prepare_module": bench.PREPARE_MODULE,
        "host_module": bench.HOST_MODULE,
        "signatures": {},
        "rotations_source": "combine_trellis_rotations",
    }


def _map_record() -> dict:
    return {
        "identifier_form_accepted": "list[int]",
        "rejected_forms": [],
        "tier0_identifiers": [0, 191],
        "tier1_identifiers": [192, 255],
    }


def _pool_record(pool_size: int = 3) -> dict:
    return {
        "pool_size": pool_size,
        "bytes_per_set": bench.resident_weight_bytes(GEOMETRY),
        "predicted_bytes": bench.weight_pool_bytes(GEOMETRY, pool_size),
        "fraction_of_device_memory": 0.02,
        "note": "packed expert weights only",
    }


def _measure_report(sizes=None) -> dict:
    if sizes is None:
        sizes = [_size_result(size_m) for size_m in bench.DEPLOYED_SIZES_M]
    return bench.build_report(
        mode="measure",
        geometry=GEOMETRY,
        environment=_environment(),
        module_record=_module_record(),
        discovery=_discovery(),
        map_record=_map_record(),
        pool_record=_pool_record(),
        clocks_before={"read": False, "reason": "nvidia-smi is not on PATH"},
        clocks_after={"read": False, "reason": "nvidia-smi is not on PATH"},
        arguments=_arguments(),
        sizes=sizes,
    )


def _sweep_entries() -> list[dict]:
    baseline = bench.DEPLOYED_CONFIGURATION
    faster = bench.Configuration((128, 128, 64, 256), 8)
    slower = bench.Configuration((128, 128, 128, 128), 16)
    broken = bench.Configuration((128, 128, 32, 128), 16)
    return [
        bench.measured_entry(_size_result(configuration=baseline, samples=[2.0] * 8)),
        bench.measured_entry(_size_result(configuration=faster, samples=[1.0] * 8)),
        bench.measured_entry(_size_result(configuration=slower, samples=[4.0] * 8)),
        bench.skipped_entry(broken, ValueError("unsupported tile config"), "measure"),
    ]


def _tune_report(entries=None, enumeration=None) -> dict:
    entries = _sweep_entries() if entries is None else entries
    if enumeration is None:
        enumeration = {
            "enumerated": 5,
            "cap": 4,
            "measured": 4,
            "dropped": 1,
            "truncated": True,
        }
    return bench.build_report(
        mode="tune",
        geometry=GEOMETRY,
        environment=_environment(),
        module_record=_module_record(),
        discovery=_discovery(),
        map_record=_map_record(),
        pool_record=_pool_record(),
        clocks_before={"read": False, "reason": "nvidia-smi is not on PATH"},
        clocks_after={"read": False, "reason": "nvidia-smi is not on PATH"},
        arguments=_arguments(),
        configurations=entries,
        enumeration=enumeration,
        ranking=bench.rank_configurations(entries),
        tuned_size_m=40,
    )


class DenseEquivalentFlopTest(unittest.TestCase):
    def test_a_routed_token_costs_gate_and_up_and_down(self) -> None:
        expected = (
            2
            * (40 * GEOMETRY.top_k)
            * (
                2 * GEOMETRY.hidden_size * GEOMETRY.intermediate_size
                + GEOMETRY.intermediate_size * GEOMETRY.hidden_size
            )
        )

        self.assertEqual(bench.dense_equivalent_flop(GEOMETRY, 40), expected)

    def test_the_closed_form_matches_the_expanded_form(self) -> None:
        self.assertEqual(
            bench.dense_equivalent_flop(GEOMETRY, 128),
            6
            * 128
            * GEOMETRY.top_k
            * GEOMETRY.hidden_size
            * GEOMETRY.intermediate_size,
        )

    def test_flop_scales_linearly_with_the_token_count(self) -> None:
        self.assertEqual(
            bench.dense_equivalent_flop(GEOMETRY, 512),
            4 * bench.dense_equivalent_flop(GEOMETRY, 128),
        )

    def test_flop_scales_linearly_with_top_k(self) -> None:
        wider = bench.Geometry(top_k=GEOMETRY.top_k * 2)

        self.assertEqual(
            bench.dense_equivalent_flop(wider, 40),
            2 * bench.dense_equivalent_flop(GEOMETRY, 40),
        )

    def test_a_token_count_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bench.dense_equivalent_flop(GEOMETRY, 0)


class CompressedByteTest(unittest.TestCase):
    def test_an_expert_holds_three_projections_at_its_tier_bit_width(self) -> None:
        coefficients = 3 * GEOMETRY.hidden_size * GEOMETRY.intermediate_size

        self.assertEqual(
            bench.expert_compressed_bytes(GEOMETRY, 3), coefficients * 3 // 8
        )
        self.assertEqual(
            bench.expert_compressed_bytes(GEOMETRY, 4), coefficients * 4 // 8
        )

    def test_a_four_bit_expert_costs_a_third_more_than_a_three_bit_expert(
        self,
    ) -> None:
        self.assertAlmostEqual(
            bench.expert_compressed_bytes(GEOMETRY, 4)
            / bench.expert_compressed_bytes(GEOMETRY, 3),
            4 / 3,
        )

    def test_weight_stream_bytes_sums_both_tiers_at_their_own_widths(self) -> None:
        expected = 10 * bench.expert_compressed_bytes(
            GEOMETRY, 3
        ) + 5 * bench.expert_compressed_bytes(GEOMETRY, 4)

        self.assertEqual(
            bench.weight_stream_bytes(GEOMETRY, tier0_experts=10, tier1_experts=5),
            expected,
        )

    def test_selecting_no_expert_streams_no_weight(self) -> None:
        self.assertEqual(
            bench.weight_stream_bytes(GEOMETRY, tier0_experts=0, tier1_experts=0), 0
        )

    def test_a_selected_count_above_the_tier_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bench.weight_stream_bytes(
                GEOMETRY, tier0_experts=GEOMETRY.tier0_num_experts + 1, tier1_experts=0
            )

    def test_a_resident_set_covers_every_expert_in_both_tiers(self) -> None:
        self.assertEqual(
            bench.resident_weight_bytes(GEOMETRY),
            bench.weight_stream_bytes(
                GEOMETRY,
                tier0_experts=GEOMETRY.tier0_num_experts,
                tier1_experts=GEOMETRY.tier1_num_experts,
            ),
        )

    def test_the_deployed_geometry_costs_under_a_gibibyte_per_weight_set(self) -> None:
        self.assertLess(bench.resident_weight_bytes(GEOMETRY), 1 << 30)

    def test_a_weight_pool_costs_its_size_times_one_set(self) -> None:
        self.assertEqual(
            bench.weight_pool_bytes(GEOMETRY, 4),
            4 * bench.resident_weight_bytes(GEOMETRY),
        )

    def test_an_empty_weight_pool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bench.weight_pool_bytes(GEOMETRY, 0)


class SelectedExpertTest(unittest.TestCase):
    def test_identifiers_split_at_the_tier_boundary(self) -> None:
        self.assertEqual(bench.count_selected_experts([0, 191, 192, 255], 192, 256), (2, 2))

    def test_repeated_identifiers_are_counted_once(self) -> None:
        self.assertEqual(bench.count_selected_experts([5, 5, 5, 200, 200], 192, 256), (1, 1))

    def test_an_identifier_outside_the_expert_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bench.count_selected_experts([256], 192, 256)


class RoutePlanTest(unittest.TestCase):
    def test_max_m_blocks_is_the_ceiling_of_slots_over_the_block_size(self) -> None:
        api = SimpleNamespace(max_packed_route_slots=lambda *_args: 321)

        plan = bench.route_slot_plan(api, GEOMETRY, size_m=40, moe_block_size=8)

        self.assertEqual(plan["route_slots"], 321)
        self.assertEqual(plan["max_m_blocks"], 41)

    def test_the_slot_count_is_asked_for_the_drawn_slots_and_expert_total(
        self,
    ) -> None:
        seen = []

        def slots(drawn, block, experts):
            seen.append((drawn, block, experts))
            return 320

        bench.route_slot_plan(
            SimpleNamespace(max_packed_route_slots=slots),
            GEOMETRY,
            size_m=40,
            moe_block_size=8,
        )

        self.assertEqual(seen, [(40 * GEOMETRY.top_k, 8, GEOMETRY.total_experts)])

    def test_a_non_positive_slot_count_is_refused_with_a_reason(self) -> None:
        api = SimpleNamespace(max_packed_route_slots=lambda *_args: 0)

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.route_slot_plan(api, GEOMETRY, size_m=40, moe_block_size=8)

        self.assertIn("max_packed_route_slots", str(raised.exception))


class DistributionTest(unittest.TestCase):
    def test_percentile_interpolates_between_neighbours(self) -> None:
        self.assertAlmostEqual(bench.percentile([0.0, 10.0], 0.5), 5.0)
        self.assertAlmostEqual(bench.percentile([0.0, 10.0], 0.25), 2.5)

    def test_percentile_ignores_input_order(self) -> None:
        self.assertAlmostEqual(bench.percentile([3.0, 1.0, 2.0], 0.5), 2.0)

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
        self.assertAlmostEqual(summary["iqr_ms"], 2.0)

    def test_summarize_rejects_an_empty_sample(self) -> None:
        with self.assertRaises(ValueError):
            bench.summarize([])

    def test_rate_conversions(self) -> None:
        # 1e12 flop in 1 ms is 1000 TFLOP/s; 1e9 bytes in 1000 ms is 1 GB/s.
        self.assertAlmostEqual(bench.tflops(10**12, 1.0), 1000.0)
        self.assertAlmostEqual(bench.gigabytes_per_second(10**9, 1000.0), 1.0)

    def test_a_non_positive_elapsed_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bench.tflops(10**12, 0.0)
        with self.assertRaises(ValueError):
            bench.gigabytes_per_second(10**9, -1.0)


class SizeResultTest(unittest.TestCase):
    def test_the_median_rate_is_the_flop_count_over_the_median_time(self) -> None:
        result = bench.size_result(
            geometry=GEOMETRY,
            size_m=40,
            configuration=bench.DEPLOYED_CONFIGURATION,
            samples_ms=[1.0, 2.0, 3.0],
            selected_tier0=100,
            selected_tier1=50,
            plan=_plan(),
        )

        self.assertAlmostEqual(
            result["rate"]["dense_equivalent_tflops_at_median"],
            bench.dense_equivalent_flop(GEOMETRY, 40) / (2.0 * 1e-3) / 1e12,
        )

    def test_the_weight_rate_uses_compressed_bytes_of_selected_experts(self) -> None:
        result = _size_result()
        expected = bench.weight_stream_bytes(
            GEOMETRY, tier0_experts=190, tier1_experts=60
        )

        self.assertEqual(result["weight_stream_bytes"], expected)
        self.assertAlmostEqual(
            result["rate"]["weight_stream_gigabytes_per_second_at_median"],
            expected / (result["timing"]["median_ms"] * 1e-3) / 1e9,
        )

    def test_the_tier_split_of_the_traffic_is_reported(self) -> None:
        result = _size_result()

        self.assertEqual(
            result["weight_stream_bytes_tier0"] + result["weight_stream_bytes_tier1"],
            result["weight_stream_bytes"],
        )
        self.assertEqual(
            result["weight_stream_bytes_tier0"],
            190 * bench.expert_compressed_bytes(GEOMETRY, GEOMETRY.tier0_bits),
        )

    def test_the_result_carries_the_routing_and_launch_plan(self) -> None:
        result = _size_result()

        self.assertEqual(result["routing"]["max_m_blocks"], 40)
        self.assertEqual(result["routing"]["route_slots_drawn"], 40 * GEOMETRY.top_k)
        self.assertEqual(result["routing"]["selected_experts"], 250)

    def test_the_minimum_time_gives_the_highest_reported_rate(self) -> None:
        result = _size_result()

        self.assertGreater(
            result["rate"]["dense_equivalent_tflops_at_min"],
            result["rate"]["dense_equivalent_tflops_at_median"],
        )


class SweepEnumerationTest(unittest.TestCase):
    def test_the_default_space_holds_fc1_at_128_and_sweeps_fc2(self) -> None:
        configurations, _ = bench.enumerate_configurations()

        self.assertEqual({item.tile_config[:2] for item in configurations}, {(128, 128)})
        self.assertEqual(
            {item.tile_config[2] for item in configurations}, {32, 64, 128}
        )
        self.assertEqual(
            {item.tile_config[3] for item in configurations}, {128, 256, 512}
        )
        self.assertEqual({item.moe_block_size for item in configurations}, {8, 16})

    def test_the_default_space_contains_all_three_backend_branches(self) -> None:
        configurations, _ = bench.enumerate_configurations()
        tiles = {item.tile_config for item in configurations}

        for branch in ((128, 128, 32, 512), (128, 128, 64, 256), (128, 128, 128, 128)):
            self.assertIn(branch, tiles)

    def test_the_baseline_comes_first_and_is_labelled(self) -> None:
        configurations, _ = bench.enumerate_configurations()

        self.assertEqual(configurations[0], bench.DEPLOYED_CONFIGURATION)
        self.assertEqual(configurations[0].role, bench.ROLE_BASELINE)
        self.assertEqual(configurations[0].tile_config, (128, 128, 32, 512))
        self.assertEqual(configurations[0].moe_block_size, 8)

    def test_the_baseline_is_not_enumerated_twice_as_a_candidate(self) -> None:
        configurations, enumeration = bench.enumerate_configurations()
        keys = [(item.tile_config, item.moe_block_size) for item in configurations]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(enumeration["enumerated"], len(configurations))
        self.assertEqual(
            sum(1 for item in configurations if item.role == bench.ROLE_BASELINE), 1
        )

    def test_configurations_are_grouped_by_tile_config(self) -> None:
        configurations, _ = bench.enumerate_configurations()
        groups = bench.tile_config_groups(configurations)

        self.assertEqual(len(groups), len({item.tile_config for item in configurations}))
        self.assertEqual(groups[0][0], bench.DEPLOYED_TILE_CONFIG)
        for tile, group in groups:
            self.assertTrue(all(item.tile_config == tile for item in group))

    def test_the_cap_truncates_and_records_what_was_dropped(self) -> None:
        configurations, enumeration = bench.enumerate_configurations(cap=5)

        self.assertEqual(len(configurations), 5)
        self.assertEqual(enumeration["measured"], 5)
        self.assertEqual(enumeration["enumerated"], 18)
        self.assertEqual(enumeration["dropped"], 13)
        self.assertTrue(enumeration["truncated"])

    def test_the_baseline_survives_the_tightest_cap(self) -> None:
        configurations, enumeration = bench.enumerate_configurations(cap=1)

        self.assertEqual(configurations, [bench.DEPLOYED_CONFIGURATION])
        self.assertTrue(enumeration["truncated"])

    def test_an_untruncated_sweep_reports_nothing_dropped(self) -> None:
        _, enumeration = bench.enumerate_configurations(cap=1000)

        self.assertFalse(enumeration["truncated"])
        self.assertEqual(enumeration["dropped"], 0)

    def test_a_control_tile_is_enumerated_and_labelled_a_control(self) -> None:
        configurations, _ = bench.enumerate_configurations(
            controls=(bench.FC1_CONTROL_TILE_CONFIG,)
        )
        controls = [
            item for item in configurations if item.role == bench.ROLE_CONTROL
        ]

        self.assertTrue(controls)
        self.assertEqual({item.tile_config for item in controls}, {(64, 256, 64, 256)})

    def test_a_cap_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bench.enumerate_configurations(cap=0)

    def test_a_configuration_names_the_tile_widths_weight_preparation_needs(
        self,
    ) -> None:
        configuration = bench.Configuration((128, 128, 32, 512), 8)

        self.assertEqual(configuration.fc1_tile_n, 128)
        self.assertEqual(configuration.fc2_tile_n, 512)
        self.assertEqual(configuration.name, "tile128x128x32x512_block8")


class SkipAndRankingTest(unittest.TestCase):
    def test_a_failed_configuration_is_recorded_with_its_exception_type(self) -> None:
        entry = bench.skipped_entry(
            bench.Configuration((128, 128, 32, 128), 16),
            RuntimeError("shared memory request exceeds the device limit"),
            "compile",
        )

        self.assertEqual(entry["status"], "skipped")
        self.assertEqual(entry["exception_type"], "RuntimeError")
        self.assertEqual(entry["stage"], "compile")
        self.assertIn("shared memory", entry["reason"])
        self.assertEqual(entry["configuration"]["moe_block_size"], 16)

    def test_an_exception_with_no_message_still_records_a_reason(self) -> None:
        entry = bench.skipped_entry(
            bench.DEPLOYED_CONFIGURATION, ValueError(), "measure"
        )

        self.assertTrue(entry["reason"])

    def test_ranking_orders_by_median_and_leaves_skips_out_of_the_order(self) -> None:
        ranking = bench.rank_configurations(_sweep_entries())

        self.assertEqual(len(ranking["ordered"]), 3)
        self.assertEqual(ranking["skipped"], 1)
        self.assertEqual(
            [row["median_ms"] for row in ranking["ordered"]], [1.0, 2.0, 4.0]
        )

    def test_speedup_is_the_baseline_median_over_the_row_median(self) -> None:
        ranking = bench.rank_configurations(_sweep_entries())
        rows = {row["name"]: row for row in ranking["ordered"]}

        self.assertAlmostEqual(
            rows["tile128x128x64x256_block8"]["speedup_over_baseline"], 2.0
        )
        self.assertAlmostEqual(
            rows["tile128x128x128x128_block16"]["speedup_over_baseline"], 0.5
        )
        self.assertAlmostEqual(ranking["baseline"]["median_ms"], 2.0)

    def test_a_faster_configuration_is_named_in_the_verdict(self) -> None:
        ranking = bench.rank_configurations(_sweep_entries())

        self.assertEqual(ranking["faster_than_baseline"], 1)
        self.assertIn("tile128x128x64x256_block8", ranking["verdict"])

    def test_the_verdict_states_plainly_when_nothing_beat_the_baseline(self) -> None:
        entries = [
            bench.measured_entry(
                _size_result(
                    configuration=bench.DEPLOYED_CONFIGURATION, samples=[1.0] * 8
                )
            ),
            bench.measured_entry(
                _size_result(
                    configuration=bench.Configuration((128, 128, 64, 256), 8),
                    samples=[3.0] * 8,
                )
            ),
        ]

        ranking = bench.rank_configurations(entries)

        self.assertEqual(ranking["faster_than_baseline"], 0)
        self.assertIn("no swept configuration beat the baseline", ranking["verdict"])

    def test_a_baseline_measured_alone_states_nothing_about_alternatives(self) -> None:
        entries = [
            bench.measured_entry(
                _size_result(configuration=bench.DEPLOYED_CONFIGURATION)
            )
        ]

        self.assertIn("states nothing", bench.rank_configurations(entries)["verdict"])

    def test_a_ranking_without_a_measured_baseline_is_refused(self) -> None:
        entries = [
            bench.measured_entry(
                _size_result(configuration=bench.Configuration((128, 128, 64, 256), 8))
            )
        ]

        with self.assertRaises(ValueError):
            bench.rank_configurations(entries)


# --------------------------------------------------------------------------
# Signature discovery.
# --------------------------------------------------------------------------


@dataclass
class FakeRotations:
    intermediate: object
    gate_suh: object
    up_suh: object
    down_svh: object


def _fake_build_tiered_maps(tier0, tier1, *, device=None):
    return (["global_to_combined", tier0, tier1], ["descriptor_map", device])


def _fake_compile(
    *,
    size_m,
    hidden_size,
    intermediate_size,
    tier0_num_experts,
    tier1_num_experts,
    top_k,
    max_m_blocks,
    sms,
    max_shared_mem,
    force_tile_config,
    tier0_bits=3,
    tier1_bits=4,
    trellis_codebook="mcg",
    moe_block_size=8,
    rotation_input_dtype="bf16",
    route_ids_dtype=None,
    broadcast_suh=False,
    broadcast_svh=False,
    route_num_experts=None,
):
    return SimpleNamespace(size_m=size_m, tile=force_tile_config)


def _fake_make_buffers(launch, *, device, sms):
    return SimpleNamespace(launch=launch, device=device, sms=sms)


def _fake_run(
    x,
    tier0,
    tier1,
    topk_weights,
    topk_ids,
    global_to_combined,
    descriptor_map,
    rotations,
    launch,
    buffers,
    gate_experts=None,
    up_experts=None,
):
    return x


def _fake_prepare(
    w13=None,
    w2=None,
    *,
    hidden_size,
    intermediate_size,
    num_experts,
    activation,
    fc1_tile_n,
    fc2_tile_n,
    device=None,
    seed=0,
    params_dtype=None,
    w13_layout="packed",
    trellis_bits=None,
    dummy_scale=None,
    codebook="mcg",
    gate_suh=None,
    up_suh=None,
    intermediate_rotations=None,
    down_svh=None,
    tile_config=None,
    workspace=None,
):
    return SimpleNamespace(
        num_experts=num_experts,
        trellis_bits=trellis_bits,
        tile_config=tile_config,
        intermediate_rotations=f"intermediate{num_experts}",
        gate_suh=f"gate{num_experts}",
        up_suh=f"up{num_experts}",
        down_svh=f"down{num_experts}",
    )


def _fake_slots(route_slots, block_size, num_experts):
    return route_slots + num_experts


def _fake_modules(**overrides):
    kernel = SimpleNamespace(
        build_tiered_maps=_fake_build_tiered_maps,
        compile_mixed_trellis=_fake_compile,
        make_mixed_trellis_buffers=_fake_make_buffers,
        run_mixed_trellis=_fake_run,
        MixedTrellisRotations=FakeRotations,
    )
    for name, value in overrides.items():
        setattr(kernel, name, value)
    prepare = SimpleNamespace(prepare_trellis256_moe_weights=_fake_prepare)
    host = SimpleNamespace(max_packed_route_slots=_fake_slots)
    return kernel, prepare, host


class SignatureRecordTest(unittest.TestCase):
    def test_positional_and_keyword_parameters_are_separated(self) -> None:
        record = bench.signature_record(_fake_make_buffers, "make_buffers")

        # A positional-or-keyword parameter appears in both lists, because
        # either way of passing it is accepted.
        self.assertEqual(record["positional"], ["launch"])
        self.assertEqual(record["keywords"], ["launch", "device", "sms"])
        self.assertFalse(record["var_positional"])
        self.assertFalse(record["var_keyword"])

    def test_the_recorded_text_shows_the_signature_a_reader_can_check(self) -> None:
        record = bench.signature_record(_fake_run, "run_mixed_trellis")

        self.assertTrue(record["text"].startswith("run_mixed_trellis(x, tier0"))

    def test_variadic_parameters_are_recorded_as_such(self) -> None:
        def variadic(*args, **kwargs):
            return args, kwargs

        record = bench.signature_record(variadic, "variadic")

        self.assertTrue(record["var_positional"])
        self.assertTrue(record["var_keyword"])

    def test_positional_order_is_accepted_when_the_names_match(self) -> None:
        record = bench.signature_record(_fake_run, "run_mixed_trellis")

        bench.require_positional_order(record, bench.EXPECTED_RUN_POSITIONAL)

    def test_a_reordered_positional_signature_is_refused_naming_what_was_found(
        self,
    ) -> None:
        def reordered(tier0, tier1, x, topk_ids, topk_weights):
            return x

        record = bench.signature_record(reordered, "run_mixed_trellis")

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.require_positional_order(record, bench.EXPECTED_RUN_POSITIONAL)

        message = str(raised.exception)
        self.assertIn("run_mixed_trellis(tier0, tier1, x", message)
        self.assertIn("not", message)

    def test_too_few_positional_parameters_are_refused(self) -> None:
        record = bench.signature_record(lambda only: only, "build_tiered_maps")

        with self.assertRaises(bench.MeasurementUnavailable):
            bench.require_positional_count(record, 2)

    def test_variadic_positional_parameters_satisfy_any_count(self) -> None:
        record = bench.signature_record(lambda *args: args, "combine")

        bench.require_positional_count(record, 10)

    def test_a_missing_keyword_is_refused_and_named(self) -> None:
        record = bench.signature_record(_fake_make_buffers, "make_buffers")

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.require_keywords(record, ("device", "streams"))

        self.assertIn("streams", str(raised.exception))

    def test_a_variadic_keyword_signature_accepts_any_keyword(self) -> None:
        record = bench.signature_record(lambda **kwargs: kwargs, "flexible")

        bench.require_keywords(record, ("anything",))


class ResolveApiTest(unittest.TestCase):
    def test_a_matching_module_set_binds_and_records_every_signature(self) -> None:
        api, discovery = bench.resolve_api(*_fake_modules())

        self.assertIs(api.run_mixed_trellis, _fake_run)
        self.assertIs(api.max_packed_route_slots, _fake_slots)
        for name in (
            "build_tiered_maps",
            "compile_mixed_trellis",
            "make_mixed_trellis_buffers",
            "run_mixed_trellis",
            "prepare_trellis256_moe_weights",
            "max_packed_route_slots",
        ):
            self.assertIn(name, discovery["signatures"])

    def test_an_absent_entry_point_is_refused_and_the_module_contents_named(
        self,
    ) -> None:
        kernel, prepare, host = _fake_modules()
        del kernel.build_tiered_maps

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.resolve_api(kernel, prepare, host)

        message = str(raised.exception)
        self.assertIn("build_tiered_maps", message)
        self.assertIn("run_mixed_trellis", message)

    def test_a_build_tiered_maps_without_a_device_keyword_is_refused(self) -> None:
        kernel, prepare, host = _fake_modules(
            build_tiered_maps=lambda tier0, tier1: (tier0, tier1)
        )

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.resolve_api(kernel, prepare, host)

        self.assertIn("device", str(raised.exception))

    def test_a_compile_missing_a_keyword_is_refused_before_any_call(self) -> None:
        def compile_without_tile(*, size_m, hidden_size):
            return size_m + hidden_size

        kernel, prepare, host = _fake_modules(
            compile_mixed_trellis=compile_without_tile
        )

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.resolve_api(kernel, prepare, host)

        self.assertIn("force_tile_config", str(raised.exception))

    def test_a_two_argument_route_slot_helper_is_refused(self) -> None:
        kernel, prepare, host = _fake_modules()
        host.max_packed_route_slots = lambda slots, block: slots

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.resolve_api(kernel, prepare, host)

        self.assertIn("max_packed_route_slots", str(raised.exception))

    def test_the_rotation_source_is_recorded_when_the_module_combines_them(
        self,
    ) -> None:
        kernel, prepare, host = _fake_modules(
            combine_trellis_rotations=lambda *tiers: FakeRotations(*range(4))
        )

        _, discovery = bench.resolve_api(kernel, prepare, host)

        self.assertEqual(discovery["rotations_source"], "combine_trellis_rotations")
        self.assertIn("combine_trellis_rotations", discovery["signatures"])

    def test_the_fallback_rotation_source_is_recorded_when_it_is_absent(self) -> None:
        _, discovery = bench.resolve_api(*_fake_modules())

        self.assertIn("MixedTrellisRotations", discovery["rotations_source"])


class RotationAssemblyTest(unittest.TestCase):
    class _Tensor:
        def __init__(self, value):
            self.value = value

        def contiguous(self):
            return self

    def _torch(self):
        return SimpleNamespace(
            cat=lambda parts, dim=0: self._Tensor(
                "|".join(str(part) for part in parts)
            )
        )

    def test_the_module_helper_is_used_when_it_exists(self) -> None:
        api, _ = bench.resolve_api(
            *_fake_modules(
                combine_trellis_rotations=lambda *tiers: FakeRotations(
                    len(tiers), 0, 0, 0
                )
            )
        )

        rotations = bench.build_rotations(self._torch(), api, ["a", "b"])

        self.assertEqual(rotations.intermediate, 2)

    def test_tier_rotations_are_concatenated_when_no_helper_exists(self) -> None:
        api, _ = bench.resolve_api(*_fake_modules())
        tiers = [_fake_prepare(hidden_size=1, intermediate_size=1, num_experts=n,
                               activation="silu", fc1_tile_n=128, fc2_tile_n=512)
                 for n in (192, 64)]

        rotations = bench.build_rotations(self._torch(), api, tiers)

        self.assertEqual(rotations.intermediate.value, "intermediate192|intermediate64")
        self.assertEqual(rotations.gate_suh.value, "gate192|gate64")
        self.assertEqual(rotations.down_svh.value, "down192|down64")

    def test_a_tier_missing_a_rotation_is_refused_with_a_reason(self) -> None:
        api, _ = bench.resolve_api(*_fake_modules())

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.build_rotations(self._torch(), api, [SimpleNamespace()])

        self.assertIn("intermediate_rotations", str(raised.exception))


class ModulePathTest(unittest.TestCase):
    def test_a_module_at_the_expected_path_matches(self) -> None:
        module = SimpleNamespace(__file__=bench.KERNEL_MODULE_PATH)

        record = bench.module_path_record(
            module, bench.KERNEL_MODULE_PATH, realpath=lambda path: path
        )

        self.assertTrue(record["matches"])
        self.assertEqual(record["reason"], "")

    def test_a_module_bound_elsewhere_is_reported_as_a_mismatch(self) -> None:
        module = SimpleNamespace(__file__="/tmp/shim/mixed_trellis.py")

        record = bench.module_path_record(
            module, bench.KERNEL_MODULE_PATH, realpath=lambda path: path
        )

        self.assertFalse(record["matches"])
        self.assertIn("import hook", record["reason"])
        self.assertEqual(record["resolved"], "/tmp/shim/mixed_trellis.py")

    def test_a_symlinked_module_matches_through_its_real_path(self) -> None:
        module = SimpleNamespace(__file__="/opt/link/mixed_trellis.py")

        record = bench.module_path_record(
            module,
            bench.KERNEL_MODULE_PATH,
            realpath=lambda path: bench.KERNEL_MODULE_PATH,
        )

        self.assertTrue(record["matches"])

    def test_a_module_without_a_file_cannot_be_identified(self) -> None:
        record = bench.module_path_record(
            SimpleNamespace(), bench.KERNEL_MODULE_PATH, realpath=lambda path: path
        )

        self.assertFalse(record["matches"])
        self.assertIn("__file__", record["reason"])


class KernelImportTest(unittest.TestCase):
    def test_an_unimportable_kernel_module_is_refused_with_its_reason(self) -> None:
        def importer(name):
            raise ModuleNotFoundError(f"No module named {name!r}")

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.import_kernel_modules(importer=importer)

        self.assertIn("ModuleNotFoundError", str(raised.exception))
        self.assertIn(bench.KERNEL_MODULE, str(raised.exception))

    def test_all_three_companion_modules_are_imported(self) -> None:
        seen = []

        def importer(name):
            seen.append(name)
            return SimpleNamespace(name=name)

        bench.import_kernel_modules(importer=importer)

        self.assertEqual(
            seen, [bench.KERNEL_MODULE, bench.PREPARE_MODULE, bench.HOST_MODULE]
        )


# --------------------------------------------------------------------------
# The timed loop, driven against a stand-in for Torch.
# --------------------------------------------------------------------------


class FakeEvent:
    def __init__(self, clock):
        self.clock = clock
        self.stamp = None

    def record(self):
        self.clock[0] += 1
        self.stamp = self.clock[0]

    def elapsed_time(self, other):
        return float(other.stamp - self.stamp)


class FakeBool:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def any(self):
        return self

    def item(self):
        return self.value


class FakeOutput:
    def __init__(self, finite=True, nonzero=True):
        self.finite = finite
        self.nonzero = nonzero

    def __ne__(self, other):
        return FakeBool(self.nonzero)


class FakeTorch:
    def __init__(self):
        self.clock = [0]
        self.synchronizations = 0
        self.cuda = SimpleNamespace(
            Event=lambda enable_timing=False: FakeEvent(self.clock),
            synchronize=self._synchronize,
            empty_cache=lambda: None,
        )

    def _synchronize(self, _device=None):
        self.synchronizations += 1

    def isfinite(self, output):
        return FakeBool(output.finite)


class TimedLoopTest(unittest.TestCase):
    def _drive(self, torch_module, output, warmup=3, iterations=6, pool_size=2):
        calls = []

        def run(x, tier0, tier1, *_rest):
            calls.append(tier0)
            return output

        api = SimpleNamespace(run_mixed_trellis=run)
        pool = [(f"tier0_{index}", f"tier1_{index}", f"rot{index}")
                for index in range(pool_size)]
        samples = bench.time_calls(
            torch_module,
            api,
            x="x",
            topk_weights="weights",
            topk_ids="ids",
            global_to_combined="map",
            descriptor_map="descriptors",
            pool=pool,
            launch="launch",
            buffers="buffers",
            device="cuda:0",
            warmup=warmup,
            iterations=iterations,
            label="case",
        )
        return samples, calls

    def test_one_sample_is_produced_per_timed_call(self) -> None:
        torch_module = FakeTorch()

        samples, _ = self._drive(torch_module, FakeOutput(), iterations=6)

        self.assertEqual(len(samples), 6)
        self.assertTrue(all(sample > 0 for sample in samples))

    def test_warmup_runs_before_the_timed_calls_and_is_not_measured(self) -> None:
        torch_module = FakeTorch()

        samples, calls = self._drive(
            torch_module, FakeOutput(), warmup=5, iterations=4
        )

        self.assertEqual(len(calls), 9)
        self.assertEqual(len(samples), 4)

    def test_the_device_is_synchronized_after_warmup_and_after_the_loop(self) -> None:
        torch_module = FakeTorch()

        self._drive(torch_module, FakeOutput())

        self.assertEqual(torch_module.synchronizations, 2)

    def test_the_weight_pool_is_cycled_one_set_per_timed_call(self) -> None:
        torch_module = FakeTorch()

        _, calls = self._drive(
            torch_module, FakeOutput(), warmup=0, iterations=5, pool_size=2
        )

        self.assertEqual(
            calls,
            ["tier0_0", "tier0_1", "tier0_0", "tier0_1", "tier0_0"],
        )

    def test_a_non_finite_output_is_refused_rather_than_reported_as_fast(self) -> None:
        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            self._drive(FakeTorch(), FakeOutput(finite=False))

        self.assertIn("non-finite", str(raised.exception))

    def test_an_all_zero_output_is_refused(self) -> None:
        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            self._drive(FakeTorch(), FakeOutput(nonzero=False))

        self.assertIn("all-zero", str(raised.exception))


class PoolFitTest(unittest.TestCase):
    def test_a_pool_within_the_fraction_is_accepted_and_described(self) -> None:
        record = bench.require_pool_fits(GEOMETRY, 3, 128 * 1024**3, 0.25)

        self.assertEqual(record["pool_size"], 3)
        self.assertEqual(
            record["predicted_bytes"], bench.weight_pool_bytes(GEOMETRY, 3)
        )
        self.assertLess(record["fraction_of_device_memory"], 0.25)

    def test_a_pool_above_the_fraction_is_refused_with_the_cost_stated(self) -> None:
        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.require_pool_fits(GEOMETRY, 64, 128 * 1024**3, 0.25)

        self.assertIn("--pool-size", str(raised.exception))

    def test_an_unknown_device_memory_size_does_not_block_the_run(self) -> None:
        record = bench.require_pool_fits(GEOMETRY, 3, 0, 0.25)

        self.assertIsNone(record["fraction_of_device_memory"])


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
    def test_both_modes_name_the_schema(self) -> None:
        self.assertEqual(_measure_report()["schema"], "gb10-mixed-trellis-roofline/v1")
        self.assertEqual(_tune_report()["schema"], bench.SCHEMA)

    def test_the_report_states_both_formulas_and_labels_the_flop_count(self) -> None:
        measurement = _measure_report()["measurement"]

        self.assertEqual(measurement["flop_formula"], bench.FLOP_FORMULA)
        self.assertEqual(measurement["weight_bytes_formula"], bench.WEIGHT_BYTES_FORMULA)
        self.assertIn("dense-equivalent", measurement["flop_note"])

    def test_the_report_records_the_module_it_measured(self) -> None:
        report = _measure_report()

        self.assertEqual(report["module"]["resolved"], bench.KERNEL_MODULE_PATH)
        self.assertTrue(report["module"]["matches"])
        self.assertIn("signatures", report["api_discovery"])

    def test_the_report_records_the_weight_pool_and_its_cost(self) -> None:
        pool = _measure_report()["measurement"]["weight_pool"]

        self.assertEqual(pool["pool_size"], 3)
        self.assertGreater(pool["predicted_bytes"], 0)

    def test_the_report_records_the_conditions_a_reader_needs(self) -> None:
        report = _measure_report()

        for key in (
            "torch_version",
            "torch_cuda_version",
            "device_name",
            "compute_capability",
            "multi_processor_count",
        ):
            self.assertIn(key, report["environment"])
        self.assertIn("before", report["clocks"])
        self.assertIn("after", report["clocks"])

    def test_the_report_states_it_is_single_process_and_not_distributed(self) -> None:
        measurement = _measure_report()["measurement"]

        self.assertTrue(measurement["single_process"])
        self.assertFalse(measurement["distributed"])

    def test_measure_mode_carries_one_entry_per_token_count(self) -> None:
        report = _measure_report()

        self.assertEqual(
            [entry["size_m"] for entry in report["sizes"]], [40, 128, 512]
        )
        self.assertEqual(report["configuration"]["tile_config"], [128, 128, 32, 512])

    def test_tune_mode_carries_the_baseline_the_entries_and_the_ranking(self) -> None:
        report = _tune_report()

        self.assertEqual(report["baseline_configuration"]["role"], bench.ROLE_BASELINE)
        self.assertEqual(len(report["configurations"]), 4)
        self.assertEqual(report["tuned_size_m"], 40)
        self.assertIn("verdict", report["ranking"])
        self.assertEqual(report["enumeration"]["dropped"], 1)

    def test_both_modes_round_trip_through_json(self) -> None:
        for report in (_measure_report(), _tune_report()):
            restored = json.loads(json.dumps(report, sort_keys=True, default=str))

            self.assertEqual(restored["schema"], bench.SCHEMA)

    def test_emit_json_writes_the_document_to_a_path(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            bench.emit_json(_tune_report(), str(destination))
            restored = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(restored["mode"], "tune")

    def test_emit_json_writes_to_stdout_for_a_dash(self) -> None:
        stream = io.StringIO()

        with redirect_stdout(stream):
            bench.emit_json(_measure_report(), "-")

        self.assertEqual(json.loads(stream.getvalue())["schema"], bench.SCHEMA)


class TextRenderTest(unittest.TestCase):
    def test_the_measure_report_prints_both_formulas_with_the_numbers(self) -> None:
        rendered = bench.render_text(_measure_report())

        self.assertIn(bench.FLOP_FORMULA, rendered)
        self.assertIn(bench.WEIGHT_BYTES_FORMULA, rendered)

    def test_the_measure_report_prints_a_median_and_a_dispersion_column(self) -> None:
        rendered = bench.render_text(_measure_report())

        for column in ("median ms", "IQR ms", "min ms", "max ms", "TFLOP/s med", "GB/s med"):
            self.assertIn(column, rendered)

    def test_the_measure_report_says_when_the_clock_state_was_not_read(self) -> None:
        self.assertIn("NOT READ", bench.render_text(_measure_report()))

    def test_the_tune_report_ranks_and_states_the_verdict(self) -> None:
        rendered = bench.render_text(_tune_report())

        self.assertIn("RANKING BY MEDIAN", rendered)
        self.assertIn("speedup", rendered)
        self.assertIn("beat the baseline", rendered)

    def test_the_tune_report_lists_skipped_configurations_with_their_type(self) -> None:
        rendered = bench.render_text(_tune_report())

        self.assertIn("SKIPPED", rendered)
        self.assertIn("ValueError", rendered)
        self.assertIn("tile128x128x32x128_block16", rendered)

    def test_the_tune_report_states_when_the_cap_dropped_configurations(self) -> None:
        self.assertIn("cap was reached", bench.render_text(_tune_report()))

    def test_the_tune_report_explains_the_control_row_when_one_was_measured(
        self,
    ) -> None:
        entries = _sweep_entries()
        entries.append(
            bench.measured_entry(
                _size_result(
                    configuration=bench.Configuration(
                        (64, 256, 64, 256), 8, bench.ROLE_CONTROL
                    )
                )
            )
        )

        rendered = bench.render_text(_tune_report(entries=entries))

        self.assertIn("CONTROL", rendered)
        self.assertIn("partial reductions", rendered)

    def test_the_configuration_table_renders_without_a_device(self) -> None:
        configurations, enumeration = bench.enumerate_configurations()

        rendered = bench.render_configuration_table(
            configurations, enumeration, GEOMETRY, 40
        )

        self.assertIn("tile128x128x32x512_block8", rendered)
        self.assertIn("baseline", rendered)
        self.assertIn(bench.FLOP_FORMULA, rendered)


class ArgumentTest(unittest.TestCase):
    def test_defaults_measure_the_deployed_geometry_on_device_zero(self) -> None:
        arguments = bench.parse_args(["measure"])

        self.assertEqual(arguments.mode, "measure")
        self.assertEqual(arguments.device, 0)
        self.assertGreaterEqual(arguments.warmup, 1)
        self.assertGreaterEqual(arguments.iterations, 4)
        self.assertEqual(arguments.sizes, (40, 128, 512))
        self.assertEqual(arguments.size_m, 40)
        self.assertEqual(arguments.pool_size, 3)
        self.assertEqual(arguments.dtype, "bf16")
        self.assertEqual(arguments.module, bench.KERNEL_MODULE)
        self.assertEqual(arguments.module_path, bench.KERNEL_MODULE_PATH)
        self.assertIsNone(arguments.json)

    def test_the_default_geometry_is_the_deployed_one(self) -> None:
        geometry = bench.geometry_from_arguments(bench.parse_args(["measure"]))

        self.assertEqual(geometry, bench.DEPLOYED_GEOMETRY)
        self.assertEqual(geometry.hidden_size, 6144)
        self.assertEqual(geometry.intermediate_size, 512)
        self.assertEqual(geometry.tier0_num_experts, 192)
        self.assertEqual(geometry.tier1_num_experts, 64)
        self.assertEqual(geometry.total_experts, 256)
        self.assertEqual(geometry.top_k, 8)
        self.assertEqual(geometry.sms, 48)

    def test_the_default_baseline_is_the_tile_config_hidden_6144_selects(self) -> None:
        baseline = bench.baseline_from_arguments(bench.parse_args(["tune"]))

        self.assertEqual(baseline.tile_config, (128, 128, 32, 512))
        self.assertEqual(baseline.moe_block_size, 8)
        self.assertEqual(baseline.role, bench.ROLE_BASELINE)

    def test_geometry_overrides_are_accepted(self) -> None:
        geometry = bench.geometry_from_arguments(
            bench.parse_args(
                ["measure", "--hidden-size", "4096", "--top-k", "4", "--sms", "24"]
            )
        )

        self.assertEqual(geometry.hidden_size, 4096)
        self.assertEqual(geometry.top_k, 4)
        self.assertEqual(geometry.sms, 24)

    def test_the_sweep_space_is_taken_from_the_command_line(self) -> None:
        arguments = bench.parse_args(
            [
                "tune",
                "--fc2-tile-k",
                "32 64",
                "--fc2-tile-n",
                "256,512",
                "--moe-block-sizes",
                "8",
            ]
        )

        configurations, enumeration = bench.configurations_from_arguments(arguments)

        self.assertEqual(enumeration["enumerated"], 4)
        self.assertEqual(
            {item.tile_config for item in configurations},
            {
                (128, 128, 32, 256),
                (128, 128, 32, 512),
                (128, 128, 64, 256),
                (128, 128, 64, 512),
            },
        )

    def test_the_control_tile_is_opt_in(self) -> None:
        without, _ = bench.configurations_from_arguments(bench.parse_args(["tune"]))
        with_control, _ = bench.configurations_from_arguments(
            bench.parse_args(["tune", "--include-fc1-control"])
        )

        self.assertNotIn(
            bench.ROLE_CONTROL, {item.role for item in without}
        )
        self.assertIn(bench.ROLE_CONTROL, {item.role for item in with_control})

    def test_the_configuration_cap_is_applied_from_the_command_line(self) -> None:
        configurations, enumeration = bench.configurations_from_arguments(
            bench.parse_args(["tune", "--max-configurations", "3"])
        )

        self.assertEqual(len(configurations), 3)
        self.assertTrue(enumeration["truncated"])

    def test_a_baseline_override_is_accepted_as_four_integers(self) -> None:
        baseline = bench.baseline_from_arguments(
            bench.parse_args(["tune", "--baseline-tile-config", "128 128 64 256"])
        )

        self.assertEqual(baseline.tile_config, (128, 128, 64, 256))

    def test_a_tile_config_of_the_wrong_length_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["tune", "--baseline-tile-config", "128 128"])

        self.assertEqual(raised.exception.code, 2)

    def test_a_missing_mode_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                bench.parse_args([])

    def test_zero_warmup_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["measure", "--warmup", "0"])

        self.assertEqual(raised.exception.code, 2)

    def test_too_few_iterations_for_an_interquartile_range_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["measure", "--iterations", "3"])

    def test_an_empty_weight_pool_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["measure", "--pool-size", "0"])

    def test_a_pool_fraction_outside_the_unit_interval_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["measure", "--max-pool-fraction", "1.5"])

    def test_a_cap_below_one_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                bench.parse_args(["tune", "--max-configurations", "0"])

    def test_list_configurations_measures_nothing_and_succeeds(self) -> None:
        def refuse_torch():
            raise AssertionError("--list-configurations must not import torch")

        def refuse_kernel(**_kwargs):
            raise AssertionError("--list-configurations must not import the kernel")

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = bench.main(
                ["tune", "--list-configurations"],
                load_torch=refuse_torch,
                load_kernel=refuse_kernel,
            )

        self.assertEqual(code, bench.EXIT_OK)
        self.assertIn("tile128x128x32x512_block8", stream.getvalue())


class NoDeviceTest(unittest.TestCase):
    """The path this repository's development machines actually take."""

    def _torch_without_cuda(self):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )

    def test_require_cuda_device_refuses_when_no_device_is_visible(self) -> None:
        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.require_cuda_device(self._torch_without_cuda(), 0)

        self.assertIn("no CUDA device", str(raised.exception))

    def test_require_cuda_device_refuses_an_index_that_does_not_exist(self) -> None:
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        )

        with self.assertRaises(bench.MeasurementUnavailable) as raised:
            bench.require_cuda_device(torch_module, 3)

        self.assertIn("does not exist", str(raised.exception))

    def test_main_exits_non_zero_and_explains_when_no_device_is_visible(self) -> None:
        errors = io.StringIO()

        with redirect_stdout(io.StringIO()), redirect_stderr(errors):
            code = bench.main(
                ["measure"],
                load_torch=self._torch_without_cuda,
                load_kernel=lambda **_kwargs: _fake_modules(),
            )

        self.assertEqual(code, bench.EXIT_UNAVAILABLE)
        self.assertIn("no measurement taken", errors.getvalue())
        self.assertIn("no CUDA device", errors.getvalue())

    def test_no_report_is_printed_to_stdout_when_no_measurement_was_taken(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            bench.main(["tune"], load_torch=self._torch_without_cuda)

        self.assertEqual(output.getvalue(), "")

    def test_main_exits_non_zero_and_explains_when_torch_is_absent(self) -> None:
        def refuse():
            raise bench.MeasurementUnavailable("torch is not importable")

        errors = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errors):
            code = bench.main(["measure"], load_torch=refuse)

        self.assertEqual(code, bench.EXIT_UNAVAILABLE)
        self.assertIn("torch is not importable", errors.getvalue())

    def test_main_refuses_when_the_kernel_module_resolves_elsewhere(self) -> None:
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                set_device=lambda _device: None,
                get_device_properties=lambda _index: SimpleNamespace(
                    multi_processor_count=48,
                    shared_memory_per_block_optin=232448,
                    total_memory=128 * 1024**3,
                ),
                get_device_capability=lambda _index: (12, 1),
                get_device_name=lambda _index: "NVIDIA GB10",
            ),
            device=lambda kind, index: f"{kind}:{index}",
            version=SimpleNamespace(cuda="13.2"),
            __version__="2.11.0",
        )
        kernel, prepare, host = _fake_modules()
        kernel.__file__ = "/somewhere/else/mixed_trellis.py"

        errors = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errors):
            code = bench.main(
                ["measure"],
                load_torch=lambda: torch_module,
                load_kernel=lambda **_kwargs: (kernel, prepare, host),
            )

        self.assertEqual(code, bench.EXIT_UNAVAILABLE)
        self.assertIn("/somewhere/else/mixed_trellis.py", errors.getvalue())
        self.assertIn("--module-path", errors.getvalue())


class SafetyClassTest(unittest.TestCase):
    def test_the_module_docstring_declares_a_safety_class(self) -> None:
        self.assertIn("Safety class: OFFLINE", bench.__doc__ or "")

    def test_the_module_docstring_states_the_dense_equivalent_flop_formula(
        self,
    ) -> None:
        self.assertIn("6 * size_m * top_k * hidden_size * intermediate_size", bench.__doc__ or "")

    def test_the_module_docstring_states_why_weights_are_cycled(self) -> None:
        self.assertIn("WEIGHT-POOL CYCLING", bench.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
