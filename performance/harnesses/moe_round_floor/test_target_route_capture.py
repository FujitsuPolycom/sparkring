from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "target_route_capture", HERE / "target_route_capture.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CaptureConfig = MODULE.CaptureConfig
CaptureError = MODULE.CaptureError
CaptureProvenance = MODULE.CaptureProvenance
CaptureSnapshot = MODULE.CaptureSnapshot
TargetRouteCapture = MODULE.TargetRouteCapture
records_from_snapshot = MODULE.records_from_snapshot
write_jsonl = MODULE.write_jsonl


def provenance() -> CaptureProvenance:
    return CaptureProvenance(
        image="sha256:image",
        checkpoint="aidendle94/GLM-5.2-MXFP4-Experts-GPTQ@revision",
        config_sha256="a" * 64,
        source_sha256={"b12x_moe.py": "b" * 64},
        rank=0,
    )


def valid_snapshot(
    width: int = 5,
    rounds: int = 1,
    *,
    rejected_tokens: list[int] | None = None,
) -> CaptureSnapshot:
    config = CaptureConfig()
    routes = []
    metadata = []
    masks = []
    for round_index in range(rounds):
        round_routes = []
        for layer in range(config.num_layers):
            positions = []
            for position in range(config.max_width):
                first = (layer * 13 + position * 9 + round_index) % 256
                positions.append([(first + offset) % 256 for offset in range(8)])
            round_routes.append(positions)
        routes.append(round_routes)
        metadata.append([0, round_index, width, 0, 0])
        masks.append([(1 << 64) - 1, (1 << 11) - 1])
    return CaptureSnapshot(
        routes=routes,
        metadata=metadata,
        layer_masks=masks,
        counters=[rounds, rounds, *([0] * (len(MODULE.COUNTER_NAMES) - 2))],
        rejected_tokens=(
            rejected_tokens
            if rejected_tokens is not None
            else [1 for _ in range(rounds)]
        ),
    )


class TargetRouteCaptureTests(unittest.TestCase):
    def test_device_control_rows_start_in_canonical_disarmed_state(self) -> None:
        class FakeTensor:
            def __init__(self, value=None) -> None:
                self.value = value

        class FakeTorch:
            int16 = "int16"
            int32 = "int32"
            int64 = "int64"

            @staticmethod
            def empty(*_args, **_kwargs):
                return FakeTensor()

            @staticmethod
            def zeros(*_args, **_kwargs):
                return FakeTensor()

            @staticmethod
            def full(shape, fill_value, **_kwargs):
                if len(shape) == 1:
                    return FakeTensor([fill_value for _ in range(shape[0])])
                return FakeTensor(
                    [
                        [fill_value for _ in range(shape[1])]
                        for _ in range(shape[0])
                    ]
                )

            @staticmethod
            def tensor(value, **_kwargs):
                return FakeTensor(value)

        capture = TargetRouteCapture(
            torch_module=FakeTorch(),
            device="cuda:0",
        )
        expected = [
            [-1, MODULE.MODEL_ROLE_TARGET, -1, 0]
            for _ in range(CaptureConfig().max_stream_slots)
        ]
        self.assertEqual(capture.stream_control.value, expected)
        self.assertEqual(
            capture.rejected_tokens.value,
            [-1 for _ in range(CaptureConfig().capacity_rounds)],
        )

    def test_q5_snapshot_emits_canonical_all_layer_record(self) -> None:
        records = records_from_snapshot(
            valid_snapshot(width=5, rejected_tokens=[2]),
            CaptureConfig(),
            {0: "salted-request"},
            provenance(),
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["schema"], "glm52-target-expert-routes/v1")
        self.assertEqual(record["phase"], "target")
        self.assertEqual(record["width"], 5)
        self.assertEqual(record["accepted_prefix_tokens"], 2)
        self.assertEqual(record["rejected_tokens"], 2)
        self.assertEqual(len(record["layers"]), 75)
        self.assertEqual(len(record["layers"][74]["positions"]), 5)
        self.assertEqual(
            len(record["layers"][74]["positions"][4]["expert_ids"]), 8
        )

    def test_q6_multiple_rounds_preserve_request_and_round_metadata(self) -> None:
        records = records_from_snapshot(
            valid_snapshot(width=6, rounds=2, rejected_tokens=[0, 5]),
            CaptureConfig(),
            {0: "salted-request"},
            provenance(),
        )
        self.assertEqual([record["round"] for record in records], [0, 1])
        self.assertTrue(all(record["width"] == 6 for record in records))
        self.assertEqual(
            [record["accepted_prefix_tokens"] for record in records],
            [5, 0],
        )
        self.assertEqual(
            [record["rejected_tokens"] for record in records],
            [0, 5],
        )
        self.assertTrue(
            all(
                len(layer["positions"]) == 6
                for record in records
                for layer in record["layers"]
            )
        )

    def test_any_overflow_or_drop_counter_fails_closed(self) -> None:
        for counter_index in range(2, len(MODULE.COUNTER_NAMES)):
            values = [1, 1, *([0] * (len(MODULE.COUNTER_NAMES) - 2))]
            values[counter_index] = 1
            snapshot = valid_snapshot()
            snapshot = CaptureSnapshot(
                routes=snapshot.routes,
                metadata=snapshot.metadata,
                layer_masks=snapshot.layer_masks,
                counters=values,
                rejected_tokens=snapshot.rejected_tokens,
            )
            with self.subTest(counter_index=counter_index):
                with self.assertRaisesRegex(CaptureError, "fatal"):
                    records_from_snapshot(
                        snapshot,
                        CaptureConfig(),
                        {0: "salted-request"},
                        provenance(),
                    )

    def test_incomplete_layer_mask_fails_closed(self) -> None:
        snapshot = valid_snapshot()
        masks = [[snapshot.layer_masks[0][0], 0]]
        damaged = CaptureSnapshot(
            routes=snapshot.routes,
            metadata=snapshot.metadata,
            layer_masks=masks,
            counters=snapshot.counters,
            rejected_tokens=snapshot.rejected_tokens,
        )
        with self.assertRaisesRegex(CaptureError, "75-layer mask"):
            records_from_snapshot(
                damaged,
                CaptureConfig(),
                {0: "salted-request"},
                provenance(),
            )

    def test_complete_int64_layer_mask_accepts_signed_low_word(self) -> None:
        snapshot = valid_snapshot()
        signed_masks = [[-1, (1 << 11) - 1]]
        materialized = CaptureSnapshot(
            routes=snapshot.routes,
            metadata=snapshot.metadata,
            layer_masks=signed_masks,
            counters=snapshot.counters,
            rejected_tokens=snapshot.rejected_tokens,
        )
        records = records_from_snapshot(
            materialized,
            CaptureConfig(),
            {0: "salted-request"},
            provenance(),
        )
        self.assertEqual(len(records), 1)

    def test_production_artifact_enforces_100_round_floor(self) -> None:
        with self.assertRaisesRegex(CaptureError, "requires 100"):
            records_from_snapshot(
                valid_snapshot(),
                CaptureConfig(),
                {0: "salted-request"},
                provenance(),
                minimum_rounds=100,
            )

    def test_draft_phase_and_unknown_request_slot_fail_closed(self) -> None:
        snapshot = valid_snapshot()
        draft = CaptureSnapshot(
            routes=snapshot.routes,
            metadata=[[0, 0, 5, MODULE.MODEL_ROLE_DRAFT, 0]],
            layer_masks=snapshot.layer_masks,
            counters=snapshot.counters,
            rejected_tokens=snapshot.rejected_tokens,
        )
        with self.assertRaisesRegex(CaptureError, "non-target"):
            records_from_snapshot(
                draft, CaptureConfig(), {0: "salted-request"}, provenance()
            )
        with self.assertRaisesRegex(CaptureError, "unknown request slot"):
            records_from_snapshot(
                snapshot, CaptureConfig(), {1: "different-slot"}, provenance()
            )

    def test_duplicate_or_out_of_range_expert_fails_closed(self) -> None:
        for replacement, message in (([3] * 8, "duplicate"), ([300] * 8, "outside")):
            snapshot = valid_snapshot()
            routes = [
                [
                    [list(position) for position in layer]
                    for layer in snapshot.routes[0]
                ]
            ]
            routes[0][0][0] = replacement
            damaged = CaptureSnapshot(
                routes=routes,
                metadata=snapshot.metadata,
                layer_masks=snapshot.layer_masks,
                counters=snapshot.counters,
                rejected_tokens=snapshot.rejected_tokens,
            )
            with self.subTest(message=message):
                with self.assertRaisesRegex(CaptureError, message):
                    records_from_snapshot(
                        damaged,
                        CaptureConfig(),
                        {0: "salted-request"},
                        provenance(),
                    )

    def test_missing_or_invalid_rejection_association_fails_closed(self) -> None:
        for rejected_tokens, message in (
            ([-1], "missing rejection"),
            ([5], "outside"),
        ):
            with self.subTest(rejected_tokens=rejected_tokens):
                with self.assertRaisesRegex(CaptureError, message):
                    records_from_snapshot(
                        valid_snapshot(
                            width=5,
                            rejected_tokens=rejected_tokens,
                        ),
                        CaptureConfig(),
                        {0: "salted-request"},
                        provenance(),
                    )

    def test_rejection_hot_path_dispatches_gpu_tensors_without_readback(
        self,
    ) -> None:
        calls: list[tuple[object, ...]] = []

        class FakeRecord:
            @staticmethod
            def record_rejection(*args):
                calls.append(args)

        capture = object.__new__(TargetRouteCapture)
        capture.config = CaptureConfig()
        capture.rejected_tokens = object()
        capture.metadata = object()
        capture.stream_control = object()
        capture.counters = object()
        capture._torch = type(
            "FakeTorch",
            (),
            {
                "ops": type(
                    "FakeOps",
                    (),
                    {"sparkring_target_route_capture": FakeRecord()},
                )()
            },
        )()
        sampled = object()
        rejected = object()
        capture.record_rejection(
            sampled,
            rejected,
            stream_slot=2,
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], sampled)
        self.assertIs(calls[0][1], rejected)
        self.assertEqual(calls[0][-1], 2)

    def test_jsonl_writer_is_atomic_and_analyzer_compatible(self) -> None:
        records = records_from_snapshot(
            valid_snapshot(),
            CaptureConfig(),
            {0: "salted-request"},
            provenance(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.jsonl"
            write_jsonl(path, records)
            parsed = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(parsed[0]["schema"], MODULE.SCHEMA)
            self.assertFalse((path.parent / f".{path.name}.tmp").exists())

    def test_hot_path_contains_no_sync_copy_or_file_io(self) -> None:
        source = (HERE / "target_route_capture.py").read_text(encoding="utf-8")
        start = source.index("    def record_target_routes(")
        end = source.index("\n    def drain_jsonl(", start)
        hot_path = source[start:end]
        for forbidden in (
            ".item(",
            ".cpu(",
            ".tolist(",
            "synchronize(",
            "torch.empty(",
            "torch.zeros(",
            "open(",
            "json.",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, hot_path)
        self.assertIn(
            "sparkring_target_route_capture.record",
            hot_path,
        )
        self.assertIn(
            "sparkring_target_route_capture.record_rejection",
            hot_path,
        )
        cuda_source = (HERE / "target_route_capture_cuda.cu").read_text(
            encoding="utf-8"
        )
        for forbidden in ("cudaMalloc", "cudaMemcpy", "cudaDeviceSynchronize"):
            with self.subTest(cuda_forbidden=forbidden):
                self.assertNotIn(forbidden, cuda_source)
        self.assertIn("kControlArmed", cuda_source)
        self.assertIn("atomicAdd(counters_u64 + kCounterClaimed", cuda_source)
        self.assertIn("num_sampled.numel() == 1", cuda_source)
        self.assertIn("num_rejected.numel() == 1", cuda_source)
        self.assertIn("atomicCAS(rejected_tokens + capture_slot", cuda_source)
        self.assertIn("kCounterRejectionOrder", cuda_source)

    def test_configuration_is_glm52_specific_and_bounded(self) -> None:
        CaptureConfig(capacity_rounds=100).validate()
        CaptureConfig(capacity_rounds=500).validate()
        for config in (
            CaptureConfig(capacity_rounds=99),
            CaptureConfig(capacity_rounds=501),
            CaptureConfig(num_layers=74),
            CaptureConfig(topk=4),
            CaptureConfig(num_experts=255),
        ):
            with self.assertRaises(ValueError):
                config.validate()

    def test_extension_is_loaded_as_a_torch_library_not_python_module(
        self,
    ) -> None:
        source = (HERE / "target_route_capture.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("is_python_module=False", source)

    def test_existing_base_router_callback_filters_non_q5_q6(self) -> None:
        class Probe(TargetRouteCapture):
            def record_target_routes(
                self, topk_ids, *, layer_index, width, stream_slot=0
            ) -> None:
                self.calls.append((topk_ids, layer_index, width, stream_slot))

        class ShapeOnly:
            def __init__(self, width: int) -> None:
                self.shape = (width, 8)

        capture = object.__new__(Probe)
        capture.config = CaptureConfig()
        capture.calls = []
        callback = capture.make_base_router_callback(
            routed_layer_index=9, model_role="target", stream_slot=2
        )
        callback(ShapeOnly(1))
        callback(ShapeOnly(5))
        callback(ShapeOnly(6))
        self.assertEqual(
            [(call[2], call[1], call[3]) for call in capture.calls],
            [(5, 9, 2), (6, 9, 2)],
        )
        with self.assertRaisesRegex(CaptureError, "MTP/draft"):
            capture.make_base_router_callback(
                routed_layer_index=9, model_role="draft"
            )


if __name__ == "__main__":
    unittest.main()
