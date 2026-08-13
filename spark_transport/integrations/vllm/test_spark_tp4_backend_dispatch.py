"""GPU-free behavioral tests for the Spark TP4 vLLM adapter."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

import spark_collective_audit
import spark_tp4_backend


class _FakeCuda:
    def __init__(self) -> None:
        self.capturing = False

    def is_current_stream_capturing(self) -> bool:
        return self.capturing


class _FakeTensor:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (1, 6144),
        dtype: str = "torch.bfloat16",
        is_cuda: bool = True,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


def _make_communicator_type() -> type:
    class FakeCudaCommunicator:
        def __init__(
            self,
            *,
            world_size: int = 4,
            unique_name: str = "tp:0",
            rank_in_group: int = 2,
        ) -> None:
            self.world_size = world_size
            self.unique_name = unique_name
            self.rank_in_group = rank_in_group
            self.original_inputs: list[object] = []

        def all_reduce(self, input_: object) -> tuple[str, object]:
            self.original_inputs.append(input_)
            return ("reference", input_)

    return FakeCudaCommunicator


def _fake_vllm_modules(
    communicator_type: type,
) -> dict[str, types.ModuleType]:
    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    device_communicators = types.ModuleType("vllm.distributed.device_communicators")
    cuda_communicator = types.ModuleType(
        "vllm.distributed.device_communicators.cuda_communicator"
    )
    cuda_communicator.CudaCommunicator = communicator_type

    vllm.distributed = distributed
    distributed.device_communicators = device_communicators
    device_communicators.cuda_communicator = cuda_communicator

    return {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.device_communicators": device_communicators,
        "vllm.distributed.device_communicators.cuda_communicator": (cuda_communicator),
    }


class _FakeShadowStats:
    def __init__(
        self,
        report: tuple[int, int, int, float, int, int, int, int] | None = None,
    ) -> None:
        self.count = 0
        self.validated = False
        self.observations: list[tuple[object, object]] = []
        self._report = report or (0, 0, 0, 0.0, 0, 0, 0, 0)

    def observe(self, candidate: object, reference: object) -> None:
        self.observations.append((candidate, reference))
        self.count += 1

    def report(self) -> tuple[int, int, int, float, int, int, int, int]:
        return self._report


class _FakeNative:
    def __init__(self, payload_bytes: int, *, graph_only: bool = False) -> None:
        self.payload_bytes = payload_bytes
        self.graph_only = graph_only
        self.inputs: list[object] = []
        self.capture_inputs: list[object] = []

    def all_reduce(self, tensor: object) -> tuple[str, int, object]:
        self.inputs.append(tensor)
        return ("candidate", self.payload_bytes, tensor)

    def capture_q1(self, tensor: object) -> tuple[str, int, object]:
        return self.capture(tensor)

    def capture(self, tensor: object) -> tuple[str, int, object]:
        self.capture_inputs.append(tensor)
        return ("graph-candidate", self.payload_bytes, tensor.shape[0], tensor)


class _FailingNative:
    def all_reduce(self, tensor: object) -> object:
        raise RuntimeError("injected native failure")


class _FakeBackend:
    created: list["_FakeBackend"] = []

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.native_sessions: dict[int, _FakeNative] = {}
        self.graph_q1_session: _FakeNative | None = None
        self.graph_dual_port_q40_session: _FakeNative | None = None
        self.shadow_stats: dict[object, _FakeShadowStats] = {}
        self.created.append(self)

    def native_for(self, payload_bytes: int) -> _FakeNative:
        native = self.native_sessions.get(payload_bytes)
        if native is None:
            native = _FakeNative(payload_bytes)
            self.native_sessions[payload_bytes] = native
        return native

    def prepare_graph_q1(self) -> _FakeNative:
        native = self.graph_q1_session
        if native is None:
            native = _FakeNative(
                spark_tp4_backend._graph_capacity_bytes(),
                graph_only=True,
            )
            self.graph_q1_session = native
        if (
            spark_tp4_backend._graph_dual_port_q40_enabled()
            and self.graph_dual_port_q40_session is None
        ):
            self.graph_dual_port_q40_session = _FakeNative(
                40 * 6144 * 2,
                graph_only=True,
            )
        return native

    def graph_session_for_capture(self, tensor: object) -> _FakeNative | None:
        if (
            tensor.shape[0] == 40
            and self.graph_dual_port_q40_session is not None
        ):
            return self.graph_dual_port_q40_session
        return self.graph_q1_session

    def shadow_for(self, signature: object) -> _FakeShadowStats:
        shadow = self.shadow_stats.get(signature)
        if shadow is None:
            shadow = _FakeShadowStats()
            self.shadow_stats[signature] = shadow
        return shadow


class _FakeFunction:
    def __init__(self, implementation=None) -> None:
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        if self.implementation is None:
            return None
        return self.implementation(*args)


class _FakeLibrary:
    def __init__(self) -> None:
        self.configs: list[dict[str, object]] = []
        self.protocols: list[int] = []
        self.graph_kernels: list[int] = []
        self.wire_schedules: list[int] = []
        self.spark_tp4_create = _FakeFunction(self._create)
        self.spark_tp4_create_with_protocol = _FakeFunction(
            self._create_with_protocol
        )
        self.spark_tp4_create_with_protocol_and_graph_kernel = _FakeFunction(
            self._create_with_protocol_and_graph_kernel
        )
        self.spark_tp4_create_with_protocol_graph_kernel_and_schedule = (
            _FakeFunction(
                self._create_with_protocol_graph_kernel_and_schedule
            )
        )
        self.spark_tp4_all_reduce = _FakeFunction()
        self.spark_tp4_capture_q1_all_reduce = _FakeFunction()
        self.capture_calls: list[dict[str, object]] = []
        self.spark_tp4_capture_all_reduce = _FakeFunction(self._capture)
        self.spark_tp4_get_graph_status = _FakeFunction(self._graph_status)
        self.spark_tp4_destroy = _FakeFunction()

    def _create(self, config_pointer, error, error_size) -> int:
        del error, error_size
        config = config_pointer._obj
        self.configs.append(
            {
                name: getattr(config, name)
                for name, _field_type in spark_tp4_backend._NativeConfig._fields_
            }
        )
        return 1

    def _create_with_protocol(
        self, config_pointer, protocol, error, error_size
    ) -> int:
        self.protocols.append(int(protocol))
        return self._create(config_pointer, error, error_size)

    def _create_with_protocol_and_graph_kernel(
        self,
        config_pointer,
        protocol,
        graph_kernel,
        error,
        error_size,
    ) -> int:
        self.protocols.append(int(protocol))
        self.graph_kernels.append(int(graph_kernel))
        return self._create(config_pointer, error, error_size)

    def _create_with_protocol_graph_kernel_and_schedule(
        self,
        config_pointer,
        protocol,
        graph_kernel,
        wire_schedule,
        error,
        error_size,
    ) -> int:
        self.protocols.append(int(protocol))
        self.graph_kernels.append(int(graph_kernel))
        self.wire_schedules.append(int(wire_schedule))
        return self._create(config_pointer, error, error_size)

    def _capture(
        self, handle, input_pointer, output_pointer, q, stream, error, error_size
    ) -> int:
        del error, error_size
        self.capture_calls.append(
            {
                "handle": handle,
                "input": input_pointer,
                "output": output_pointer,
                "q": q,
                "stream": stream,
            }
        )
        return 0

    def _graph_status(
        self, handle, status_pointer, status_bytes, error, error_size
    ) -> int:
        del handle, status_bytes, error, error_size
        status = status_pointer._obj
        status.struct_size = spark_tp4_backend.ctypes.sizeof(
            spark_tp4_backend._NativeGraphStatus
        )
        status.flags = (
            spark_tp4_backend._GRAPH_STATUS_CAPTURE_CONFIGURED
            | spark_tp4_backend._GRAPH_STATUS_POLLING_ENABLED
            | spark_tp4_backend._GRAPH_STATUS_HOST_NATIVE_ATOMICS
            | spark_tp4_backend._GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED
            | spark_tp4_backend._GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED
        )
        if self.protocols and self.protocols[-1] == 1:
            status.flags |= (
                spark_tp4_backend._GRAPH_STATUS_TWO_SLOT_DEFERRED_ACK
            )
        if self.graph_kernels == [1]:
            status.flags |= spark_tp4_backend._GRAPH_STATUS_SPLIT_64K
        elif self.graph_kernels == [2]:
            status.flags |= spark_tp4_backend._GRAPH_STATUS_TIERED_64K
        if self.wire_schedules == [1]:
            status.flags |= (
                spark_tp4_backend._GRAPH_STATUS_DUAL_PORT_STRIPED
            )
        status.captured_nodes = 128
        status.published_sequence = 1024
        status.consumed_sequence = 1024
        status.completed_sequence = 1023
        status.overflow_sequence = 0
        status.graph_submit_cpu_plus_one = 11
        status.graph_progress_cpu_plus_one = 12
        return 0


class SparkTp4BackendDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.communicator_type = _make_communicator_type()
        self.modules = _fake_vllm_modules(self.communicator_type)
        self.torch_module = types.ModuleType("torch")
        self.torch_module.cuda = _FakeCuda()
        self.modules["torch"] = self.torch_module
        self.original_all_reduce = self.communicator_type.all_reduce
        spark_tp4_backend._installed = False
        spark_tp4_backend._graph_q1_sessions.clear()
        spark_tp4_backend._graph_dual_port_q40_sessions.clear()
        spark_tp4_backend._graph_event_counts.clear()
        spark_collective_audit._reset_for_tests()
        _FakeBackend.created.clear()

    def _install(
        self,
        mode: str | None,
        extra_environment: dict[str, str] | None = None,
    ) -> None:
        environment = dict(extra_environment or {})
        if mode is not None:
            environment["VLLM_SPARK_TP4_MODE"] = mode
        patchers = (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(sys.modules, self.modules),
            patch.object(spark_tp4_backend, "_Backend", _FakeBackend),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        spark_tp4_backend.install()

    def test_unset_mode_leaves_vllm_communicator_unchanged(self) -> None:
        self._install(None)

        self.assertIs(self.communicator_type.all_reduce, self.original_all_reduce)
        self.assertFalse(spark_tp4_backend._installed)

    def test_invalid_mode_is_rejected_before_vllm_is_patched(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "VLLM_SPARK_TP4_MODE must be 'shadow', 'custom', 'disabled', or unset",
        ):
            self._install("fastest")

        self.assertIs(self.communicator_type.all_reduce, self.original_all_reduce)
        self.assertFalse(spark_tp4_backend._installed)

    def test_deferred_ack_protocol_requires_custom_graph_transport(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "two_slot_deferred_ack requires custom graph all-reduce",
        ):
            self._install(
                "custom",
                {
                    "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL": (
                        "two_slot_deferred_ack"
                    )
                },
            )

        self.assertIs(
            self.communicator_type.all_reduce,
            self.original_all_reduce,
        )
        self.assertFalse(spark_tp4_backend._installed)

    def test_invalid_graph_allreduce_protocol_is_rejected_before_patch(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL",
        ):
            self._install(
                "custom",
                {
                    "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL": "fast",
                },
            )

        self.assertIs(
            self.communicator_type.all_reduce,
            self.original_all_reduce,
        )
        self.assertFalse(spark_tp4_backend._installed)

    def test_invalid_graph_kernel_strategy_is_rejected_before_patch(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY",
        ):
            self._install(
                "custom",
                {
                    "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY": "automatic",
                },
            )

        self.assertIs(
            self.communicator_type.all_reduce,
            self.original_all_reduce,
        )
        self.assertFalse(spark_tp4_backend._installed)

    def test_tiered_graph_kernel_requires_custom_graph_transport(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "research graph kernel strategies require custom graph all-reduce",
        ):
            self._install(
                "custom",
                {
                    "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY": "tiered_64k",
                },
            )

        self.assertFalse(spark_tp4_backend._installed)

    def test_custom_mode_routes_only_mtp_rows_one_through_six(self) -> None:
        self._install("custom")
        communicator = self.communicator_type(rank_in_group=3)
        tensors = [_FakeTensor(shape=(rows, 6144)) for rows in range(1, 7)]

        results = [communicator.all_reduce(tensor) for tensor in tensors]

        expected_payloads = [rows * 12288 for rows in range(1, 7)]
        self.assertEqual(
            results,
            [
                ("candidate", payload_bytes, tensor)
                for payload_bytes, tensor in zip(expected_payloads, tensors)
            ],
        )
        self.assertEqual(communicator.original_inputs, [])
        self.assertEqual(len(_FakeBackend.created), 1)
        backend = _FakeBackend.created[0]
        self.assertEqual(backend.rank, 3)
        self.assertEqual(list(backend.native_sessions), expected_payloads)
        for payload_bytes, tensor in zip(expected_payloads, tensors):
            self.assertEqual(backend.native_sessions[payload_bytes].inputs, [tensor])

    def test_prefill_q512_requires_exact_runtime_opt_in(self) -> None:
        self._install("custom")
        communicator = self.communicator_type(rank_in_group=3)
        tensors = [
            _FakeTensor(shape=(rows, 6144))
            for rows in (48, 72, 144, 512)
        ]
        for tensor in tensors:
            self.assertEqual(
                communicator.all_reduce(tensor),
                ("reference", tensor),
            )

        self.assertEqual(communicator.original_inputs, tensors)
        self.assertEqual(_FakeBackend.created, [])

    def test_prefill_q512_opt_in_routes_observed_shapes(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_PREFILL_Q512"] = "1"
        communicator = self.communicator_type(rank_in_group=3)
        tensors = [
            _FakeTensor(shape=(rows, 6144))
            for rows in (48, 72, 144, 512)
        ]

        results = [communicator.all_reduce(tensor) for tensor in tensors]

        self.assertEqual(
            results,
            [
                ("candidate", rows * 12288, tensor)
                for rows, tensor in zip((48, 72, 144, 512), tensors)
            ],
        )
        self.assertEqual(communicator.original_inputs, [])

    def test_prefill_q512_opt_in_admits_every_boundary_fail_closed(self) -> None:
        self._install(
            "custom",
            {"VLLM_SPARK_TP4_PREFILL_Q512": "1"},
        )
        communicator = self.communicator_type(rank_in_group=3)
        admitted = [
            _FakeTensor(shape=(rows, 6144))
            for rows in (1, 40, 41, 511, 512)
        ]

        for tensor in admitted:
            result = communicator.all_reduce(tensor)
            self.assertEqual(result[0], "candidate")

        rejected = _FakeTensor(shape=(513, 6144))
        self.assertEqual(
            communicator.all_reduce(rejected),
            ("reference", rejected),
        )
        self.assertEqual(communicator.original_inputs, [rejected])

    def test_invalid_prefill_q512_opt_in_fails_before_patch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "VLLM_SPARK_TP4_PREFILL_Q512 must be '0', '1', or unset",
        ):
            self._install(
                "custom",
                {"VLLM_SPARK_TP4_PREFILL_Q512": "yes"},
            )
        self.assertIs(
            self.communicator_type.all_reduce,
            self.original_all_reduce,
        )

    def test_capture_delegates_without_touching_native_or_shadow_state(
        self,
    ) -> None:
        self._install("custom")
        self.torch_module.cuda.capturing = True
        communicator = self.communicator_type(rank_in_group=1)
        backend = _FakeBackend(communicator.rank_in_group)
        communicator._spark_tp4_native = backend
        _FakeBackend.created.clear()
        tensors = [_FakeTensor(), _FakeTensor(shape=(5, 6144))]

        results = []
        for mode, tensor in zip(("custom", "shadow"), tensors):
            os.environ["VLLM_SPARK_TP4_MODE"] = mode
            results.append(communicator.all_reduce(tensor))

        self.assertEqual(
            results,
            [
                ("reference", tensors[0]),
                ("reference", tensors[1]),
            ],
        )
        self.assertEqual(communicator.original_inputs, tensors)
        self.assertIs(communicator._spark_tp4_native, backend)
        self.assertEqual(backend.native_sessions, {})
        self.assertEqual(backend.shadow_stats, {})
        self.assertEqual(_FakeBackend.created, [])

    def test_custom_q1_capture_uses_prepared_graph_session(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        warmup = _FakeTensor()
        captured = _FakeTensor()

        warmup_result = communicator.all_reduce(warmup)
        backend = _FakeBackend.created[0]
        self.torch_module.cuda.capturing = True
        captured_result = communicator.all_reduce(captured)

        self.assertEqual(warmup_result, ("candidate", 12288, warmup))
        self.assertEqual(
            captured_result,
            ("graph-candidate", 73728, 1, captured),
        )
        self.assertIsNotNone(backend.graph_q1_session)
        self.assertEqual(backend.graph_q1_session.capture_inputs, [captured])
        self.assertEqual(communicator._spark_tp4_graph_q1_captured_nodes, 1)
        self.assertEqual(communicator.original_inputs, [])

    def test_custom_mixed_q_capture_uses_one_q6_capacity_session(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        warmup = _FakeTensor(shape=(3, 6144))
        captured = [
            _FakeTensor(shape=(rows, 6144)) for rows in range(1, 7)
        ]

        communicator.all_reduce(warmup)
        backend = _FakeBackend.created[0]
        self.torch_module.cuda.capturing = True
        results = [communicator.all_reduce(tensor) for tensor in captured]

        self.assertIsNotNone(backend.graph_q1_session)
        self.assertEqual(backend.graph_q1_session.payload_bytes, 73728)
        self.assertEqual(backend.graph_q1_session.capture_inputs, captured)
        self.assertEqual(
            results,
            [
                ("graph-candidate", 73728, rows, tensor)
                for rows, tensor in zip(range(1, 7), captured)
            ],
        )
        self.assertEqual(
            communicator._spark_tp4_graph_q1_captured_nodes,
            6,
        )
        self.assertEqual(
            spark_tp4_backend.graph_q1_diagnostic_snapshot()["events"],
            {"captured_nodes": 6},
        )
        self.assertEqual(communicator.original_inputs, [])

    def test_exact_q40_capture_uses_dedicated_dual_port_session(self) -> None:
        with patch.object(
            spark_tp4_backend,
            "_TARGET_SHAPES",
            frozenset({(39, 6144), (40, 6144)}),
        ), patch.object(
            spark_tp4_backend,
            "MAX_QUERY_ROWS",
            40,
        ):
            self._install(
                "custom",
                {
                    "VLLM_SPARK_TP4_GRAPH_Q1": "1",
                    "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40": "1",
                    "SPARK_TP4_GRAPH_DUAL_PORT_Q40_PROGRESS_CPU": "15",
                },
            )
            communicator = self.communicator_type(rank_in_group=1)
            warmup = _FakeTensor(shape=(39, 6144))
            q39 = _FakeTensor(shape=(39, 6144))
            q40 = _FakeTensor(shape=(40, 6144))

            communicator.all_reduce(warmup)
            backend = _FakeBackend.created[0]
            self.torch_module.cuda.capturing = True
            q39_result = communicator.all_reduce(q39)
            q40_result = communicator.all_reduce(q40)

        self.assertEqual(
            q39_result,
            ("graph-candidate", 40 * 6144 * 2, 39, q39),
        )
        self.assertEqual(
            q40_result,
            ("graph-candidate", 40 * 6144 * 2, 40, q40),
        )
        self.assertEqual(backend.graph_q1_session.capture_inputs, [q39])
        self.assertEqual(
            backend.graph_dual_port_q40_session.capture_inputs,
            [q40],
        )

    def test_prefill_graph_capture_uses_one_six_mib_session(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        os.environ["VLLM_SPARK_TP4_PREFILL_Q512"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        warmup = _FakeTensor(shape=(48, 6144))
        captured = [
            _FakeTensor(shape=(rows, 6144))
            for rows in (48, 72, 144, 512)
        ]

        communicator.all_reduce(warmup)
        backend = _FakeBackend.created[0]
        self.torch_module.cuda.capturing = True
        results = [communicator.all_reduce(tensor) for tensor in captured]

        self.assertIsNotNone(backend.graph_q1_session)
        self.assertEqual(
            backend.graph_q1_session.payload_bytes,
            6 * 1024 * 1024,
        )
        self.assertEqual(backend.graph_q1_session.capture_inputs, captured)
        self.assertEqual(
            results,
            [
                ("graph-candidate", 6 * 1024 * 1024, rows, tensor)
                for rows, tensor in zip((48, 72, 144, 512), captured)
            ],
        )

    def test_unprepared_q1_capture_falls_back_without_blocking_setup(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        self.torch_module.cuda.capturing = True
        communicator = self.communicator_type(rank_in_group=1)
        captured = _FakeTensor()

        result = communicator.all_reduce(captured)

        self.assertEqual(result, ("reference", captured))
        self.assertEqual(communicator.original_inputs, [captured])
        self.assertEqual(communicator._spark_tp4_graph_q1_fallbacks, 1)
        self.assertEqual(_FakeBackend.created, [])

    def test_unprepared_capture_is_visible_to_stock_audit(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        self.torch_module.cuda.capturing = True
        communicator = self.communicator_type(rank_in_group=1)

        communicator.all_reduce(_FakeTensor())

        self.assertEqual(
            spark_collective_audit.stock_collective_snapshot(),
            {
                "capture": {
                    "all_reduce:graph_session_unprepared": 1,
                },
                "eager": {},
                "capture_total": 1,
                "eager_total": 0,
            },
        )

    def test_unprepared_q6_capture_falls_back_without_blocking_setup(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        self.torch_module.cuda.capturing = True
        communicator = self.communicator_type(rank_in_group=1)
        captured = _FakeTensor(shape=(6, 6144))

        result = communicator.all_reduce(captured)

        self.assertEqual(result, ("reference", captured))
        self.assertEqual(communicator.original_inputs, [captured])
        self.assertEqual(communicator._spark_tp4_graph_q1_fallbacks, 1)
        self.assertEqual(_FakeBackend.created, [])

    def test_first_eligible_custom_warmup_prepares_graph_q1(self) -> None:
        self._install("custom")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        communicator = self.communicator_type(rank_in_group=1)

        communicator.all_reduce(_FakeTensor(shape=(3, 6144)))
        backend = _FakeBackend.created[0]
        self.assertIsNotNone(backend.graph_q1_session)

    def test_shadow_warmup_does_not_prepare_graph_q1(self) -> None:
        self._install("shadow")
        os.environ["VLLM_SPARK_TP4_GRAPH_Q1"] = "1"
        communicator = self.communicator_type(rank_in_group=1)

        communicator.all_reduce(_FakeTensor())
        backend = _FakeBackend.created[0]
        self.assertIsNone(backend.graph_q1_session)

    def test_native_sessions_are_reused_by_payload_bytes(self) -> None:
        self._install("custom")
        communicator = self.communicator_type(rank_in_group=1)
        row_sequence = (3, 5, 3, 1, 5, 3)
        tensors = [_FakeTensor(shape=(rows, 6144)) for rows in row_sequence]

        for tensor in tensors:
            communicator.all_reduce(tensor)

        backend = _FakeBackend.created[0]
        self.assertEqual(set(backend.native_sessions), {12288, 36864, 61440})
        self.assertEqual(
            backend.native_sessions[36864].inputs,
            [tensors[0], tensors[2], tensors[5]],
        )
        self.assertEqual(
            backend.native_sessions[61440].inputs,
            [tensors[1], tensors[4]],
        )
        self.assertEqual(backend.native_sessions[12288].inputs, [tensors[3]])

    def test_ineligible_inputs_delegate_without_creating_native_session(
        self,
    ) -> None:
        self._install("custom")
        cases = {
            "not four ranks": (
                self.communicator_type(world_size=2),
                _FakeTensor(),
            ),
            "not tensor-parallel group": (
                self.communicator_type(unique_name="pp:0"),
                _FakeTensor(),
            ),
            "zero rows": (
                self.communicator_type(),
                _FakeTensor(shape=(0, 6144)),
            ),
            "too many decode rows": (
                self.communicator_type(),
                _FakeTensor(shape=(7, 6144)),
            ),
            "warmup 15 rows": (
                self.communicator_type(),
                _FakeTensor(shape=(15, 6144)),
            ),
            "prefill 28 rows": (
                self.communicator_type(),
                _FakeTensor(shape=(28, 6144)),
            ),
            "prefill 31 rows": (
                self.communicator_type(),
                _FakeTensor(shape=(31, 6144)),
            ),
            "wrong width": (
                self.communicator_type(),
                _FakeTensor(shape=(3, 6143)),
            ),
            "wrong rank": (
                self.communicator_type(),
                _FakeTensor(shape=(3, 1, 6144)),
            ),
            "wrong dtype": (
                self.communicator_type(),
                _FakeTensor(dtype="torch.float16"),
            ),
            "not CUDA": (
                self.communicator_type(),
                _FakeTensor(is_cuda=False),
            ),
            "not contiguous": (
                self.communicator_type(),
                _FakeTensor(contiguous=False),
            ),
        }

        for name, (communicator, tensor) in cases.items():
            with self.subTest(name=name):
                result = communicator.all_reduce(tensor)
                self.assertEqual(result, ("reference", tensor))
                self.assertEqual(communicator.original_inputs, [tensor])

        self.assertEqual(_FakeBackend.created, [])

    def test_ineligible_input_records_exact_bounded_signature(self) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        communicator = self.communicator_type(
            world_size=4,
            unique_name="tp:0",
        )
        tensor = _FakeTensor(shape=(160, 6144))

        result = communicator.all_reduce(tensor)

        self.assertEqual(result, ("reference", tensor))
        snapshot = spark_collective_audit.stock_collective_snapshot()
        self.assertEqual(
            snapshot["eager"],
            {"all_reduce:ineligible_signature": 1},
        )
        self.assertEqual(
            snapshot["signatures"]["eager"],
            [
                {
                    "family": "all_reduce",
                    "reason": "ineligible_signature",
                    "count": 1,
                    "shape": [160, 6144],
                    "dtype": "torch.bfloat16",
                    "is_cuda": True,
                    "contiguous": True,
                    "world_size": 4,
                    "unique_name": "tp:0",
                }
            ],
        )

    def test_shadow_mode_compares_candidate_but_returns_reference(
        self,
    ) -> None:
        self._install("shadow")
        communicator = self.communicator_type(rank_in_group=1)
        tensor = _FakeTensor()

        result = communicator.all_reduce(tensor)

        backend = _FakeBackend.created[0]
        self.assertEqual(result, ("reference", tensor))
        self.assertEqual(communicator.original_inputs, [tensor])
        self.assertEqual(backend.native_sessions[12288].inputs, [tensor])
        self.assertEqual(
            next(iter(backend.shadow_stats.values())).observations,
            [(("candidate", 12288, tensor), ("reference", tensor))],
        )

    def test_shadow_stats_are_separate_per_payload_signature(self) -> None:
        self._install("shadow")
        communicator = self.communicator_type(rank_in_group=1)
        row3_first = _FakeTensor(shape=(3, 6144))
        row5 = _FakeTensor(shape=(5, 6144))
        row3_second = _FakeTensor(shape=(3, 6144))

        for tensor in (row3_first, row5, row3_second):
            communicator.all_reduce(tensor)

        backend = _FakeBackend.created[0]
        row3_signature = spark_tp4_backend._collective_signature(row3_first)
        row5_signature = spark_tp4_backend._collective_signature(row5)
        self.assertEqual(
            set(backend.shadow_stats),
            {
                row3_signature,
                row5_signature,
            },
        )
        self.assertEqual(backend.shadow_stats[row3_signature].count, 2)
        self.assertEqual(backend.shadow_stats[row5_signature].count, 1)
        self.assertEqual(
            backend.shadow_stats[row3_signature].observations,
            [
                (
                    ("candidate", 36864, row3_first),
                    ("reference", row3_first),
                ),
                (
                    ("candidate", 36864, row3_second),
                    ("reference", row3_second),
                ),
            ],
        )

    def test_mode_change_applies_to_the_next_collective(self) -> None:
        self._install("custom")
        communicator = self.communicator_type(rank_in_group=1)
        shadow_tensor = _FakeTensor()
        custom_tensor = _FakeTensor()

        os.environ["VLLM_SPARK_TP4_MODE"] = "shadow"
        shadow_result = communicator.all_reduce(shadow_tensor)
        os.environ["VLLM_SPARK_TP4_MODE"] = "custom"
        custom_result = communicator.all_reduce(custom_tensor)

        backend = _FakeBackend.created[0]
        self.assertEqual(shadow_result, ("reference", shadow_tensor))
        self.assertEqual(custom_result, ("candidate", 12288, custom_tensor))
        self.assertEqual(communicator.original_inputs, [shadow_tensor])
        self.assertEqual(
            backend.native_sessions[12288].inputs,
            [shadow_tensor, custom_tensor],
        )

    def test_passing_shadow_signature_promotes_on_next_call(self) -> None:
        self._install("shadow")
        os.environ["SPARK_TP4_SHADOW_COLLECTIVES"] = "2"
        os.environ["SPARK_TP4_SHADOW_PROMOTE"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        tensors = [_FakeTensor(shape=(3, 6144)) for _ in range(3)]

        results = [communicator.all_reduce(tensor) for tensor in tensors]

        self.assertEqual(
            results,
            [
                ("reference", tensors[0]),
                ("reference", tensors[1]),
                ("candidate", 36864, tensors[2]),
            ],
        )
        self.assertEqual(communicator.original_inputs, [tensors[0], tensors[1]])
        shadow = next(iter(_FakeBackend.created[0].shadow_stats.values()))
        self.assertEqual(shadow.count, 2)
        self.assertTrue(shadow.validated)

    def test_ulp_is_diagnostic_only_by_default(self) -> None:
        self._install("shadow")
        os.environ["SPARK_TP4_SHADOW_COLLECTIVES"] = "1"
        os.environ["SPARK_TP4_SHADOW_PROMOTE"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        first = _FakeTensor(shape=(3, 6144))
        second = _FakeTensor(shape=(3, 6144))
        signature = spark_tp4_backend._collective_signature(first)
        backend = _FakeBackend(communicator.rank_in_group)
        backend.shadow_stats[signature] = _FakeShadowStats(
            report=(100, 0, 0, 0.015625, 14976, 80, 40, 20)
        )
        communicator._spark_tp4_native = backend

        results = [
            communicator.all_reduce(first),
            communicator.all_reduce(second),
        ]

        self.assertEqual(
            results,
            [("reference", first), ("candidate", 36864, second)],
        )
        self.assertTrue(backend.shadow_stats[signature].validated)

    def test_explicit_ulp_gate_can_block_promotion(self) -> None:
        self._install("shadow")
        os.environ["SPARK_TP4_SHADOW_COLLECTIVES"] = "1"
        os.environ["SPARK_TP4_SHADOW_PROMOTE"] = "1"
        os.environ["SPARK_TP4_SHADOW_MAX_ULP"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        first = _FakeTensor(shape=(3, 6144))
        second = _FakeTensor(shape=(3, 6144))
        signature = spark_tp4_backend._collective_signature(first)
        backend = _FakeBackend(communicator.rank_in_group)
        backend.shadow_stats[signature] = _FakeShadowStats(
            report=(100, 0, 0, 0.015625, 14976, 80, 40, 20)
        )
        communicator._spark_tp4_native = backend

        results = [
            communicator.all_reduce(first),
            communicator.all_reduce(second),
        ]

        self.assertEqual(
            results,
            [("reference", first), ("reference", second)],
        )
        self.assertFalse(backend.shadow_stats[signature].validated)

    def test_shadow_promotion_is_signature_local(self) -> None:
        self._install("shadow")
        os.environ["SPARK_TP4_SHADOW_COLLECTIVES"] = "1"
        os.environ["SPARK_TP4_SHADOW_PROMOTE"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        row3_first = _FakeTensor(shape=(3, 6144))
        row5_first = _FakeTensor(shape=(5, 6144))
        row3_promoted = _FakeTensor(shape=(3, 6144))

        results = [
            communicator.all_reduce(row3_first),
            communicator.all_reduce(row5_first),
            communicator.all_reduce(row3_promoted),
        ]

        self.assertEqual(
            results,
            [
                ("reference", row3_first),
                ("reference", row5_first),
                ("candidate", 36864, row3_promoted),
            ],
        )
        self.assertEqual(communicator.original_inputs, [row3_first, row5_first])
        backend = _FakeBackend.created[0]
        self.assertTrue(
            backend.shadow_stats[
                spark_tp4_backend._collective_signature(row3_first)
            ].validated
        )
        self.assertTrue(
            backend.shadow_stats[
                spark_tp4_backend._collective_signature(row5_first)
            ].validated
        )

    def test_shadow_promotion_defaults_off(self) -> None:
        self._install("shadow")
        os.environ["SPARK_TP4_SHADOW_COLLECTIVES"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        first = _FakeTensor(shape=(5, 6144))
        second = _FakeTensor(shape=(5, 6144))

        results = [
            communicator.all_reduce(first),
            communicator.all_reduce(second),
        ]

        self.assertEqual(results, [("reference", first), ("reference", second)])
        self.assertEqual(communicator.original_inputs, [first, second])
        shadow = next(iter(_FakeBackend.created[0].shadow_stats.values()))
        self.assertTrue(shadow.validated)

    def test_failed_shadow_signature_never_promotes(self) -> None:
        self._install("shadow")
        os.environ["SPARK_TP4_SHADOW_COLLECTIVES"] = "1"
        os.environ["SPARK_TP4_SHADOW_PROMOTE"] = "1"
        communicator = self.communicator_type(rank_in_group=1)
        first = _FakeTensor(shape=(3, 6144))
        second = _FakeTensor(shape=(3, 6144))
        signature = spark_tp4_backend._collective_signature(first)
        backend = _FakeBackend(communicator.rank_in_group)
        backend.shadow_stats[signature] = _FakeShadowStats(
            report=(0, 1, 0, 0.02, 2, 1, 0, 0)
        )
        communicator._spark_tp4_native = backend

        results = [
            communicator.all_reduce(first),
            communicator.all_reduce(second),
        ]

        self.assertEqual(results, [("reference", first), ("reference", second)])
        self.assertEqual(communicator.original_inputs, [first, second])
        self.assertFalse(backend.shadow_stats[signature].validated)

    def test_native_failure_terminates_instead_of_falling_back(self) -> None:
        self._install("shadow")
        communicator = self.communicator_type(rank_in_group=0)
        backend = _FakeBackend.created
        self.assertEqual(backend, [])

        fatal_backend = types.SimpleNamespace(
            native_for=lambda payload_bytes: _FailingNative(),
            shadow_for=lambda signature: _FakeShadowStats(),
        )
        communicator._spark_tp4_native = fatal_backend
        with patch.object(
            spark_tp4_backend,
            "_abort_after_native_failure",
            side_effect=SystemExit(70),
        ) as abort:
            with self.assertRaisesRegex(SystemExit, "70"):
                communicator.all_reduce(_FakeTensor())

        abort.assert_called_once_with()
        self.assertEqual(communicator.original_inputs, [])


class SparkTp4NativeSessionConfigTest(unittest.TestCase):
    def test_native_config_layout_remains_legacy_64_bytes(self) -> None:
        config = spark_tp4_backend._NativeConfig
        self.assertEqual(spark_tp4_backend.ctypes.sizeof(config), 64)
        self.assertEqual(config.rank.offset, 0)
        self.assertEqual(config.peer0.offset, 8)
        self.assertEqual(config.peer1.offset, 16)
        self.assertEqual(config.device0.offset, 24)
        self.assertEqual(config.device1.offset, 32)
        self.assertEqual(config.gid0.offset, 40)
        self.assertEqual(config.gid1.offset, 41)
        self.assertEqual(config.control_port0.offset, 42)
        self.assertEqual(config.control_port1.offset, 44)
        self.assertEqual(config.payload_bytes.offset, 48)
        self.assertEqual(config.graph_submit_cpu_plus_one.offset, 56)
        self.assertEqual(config.graph_progress_cpu_plus_one.offset, 60)

    def test_two_slot_protocol_uses_versioned_create_symbol(self) -> None:
        library = _FakeLibrary()
        environment = {"SPARK_TP4_LIBRARY": "fake.dll"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
                allreduce_protocol="two_slot_deferred_ack",
            )
            status = session.graph_status()

        self.assertEqual(library.protocols, [1])
        self.assertEqual(len(library.configs), 1)
        self.assertTrue(status.two_slot_deferred_ack)
        self.assertFalse(status.command_caught_up)
        self.assertIsNone(status.payload_slots_retired)

    def test_two_slot_protocol_is_rejected_for_eager_session(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "two_slot_deferred_ack requires a graph-only session",
        ):
            spark_tp4_backend._NativeSession(
                0,
                12288,
                allreduce_protocol="two_slot_deferred_ack",
            )

    def test_dual_port_schedule_uses_additive_create_symbol(self) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                40 * 6144 * 2,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
                allreduce_protocol="two_slot_deferred_ack",
                graph_kernel_strategy="fused",
                wire_schedule="dual_port_striped",
            )
            status = session.graph_status()

        self.assertEqual(library.protocols, [1])
        self.assertEqual(library.graph_kernels, [0])
        self.assertEqual(library.wire_schedules, [1])
        self.assertEqual(status.wire_schedule, "dual_port_striped")

    def test_dual_port_schedule_rejects_incompatible_contracts(self) -> None:
        common = {
            "rank": 0,
            "payload_bytes": 40 * 6144 * 2,
            "control_ports": (14000, 14001),
            "graph_only": True,
            "graph_cpu_affinity": (10, 11),
            "allreduce_protocol": "two_slot_deferred_ack",
            "graph_kernel_strategy": "fused",
            "wire_schedule": "dual_port_striped",
        }
        for field, value, message in (
            ("graph_only", False, "graph-only"),
            ("allreduce_protocol", "serial_ack", "two_slot_deferred_ack"),
            ("graph_kernel_strategy", "tiered_64k", "fused graph kernel"),
            ("payload_bytes", 39 * 6144 * 2, "Q40 or Q512 capacity"),
        ):
            arguments = dict(common)
            arguments[field] = value
            if field == "graph_only":
                arguments["graph_cpu_affinity"] = None
            with self.assertRaisesRegex(ValueError, message):
                spark_tp4_backend._NativeSession(**arguments)

    def test_dual_port_schedule_rejects_stale_native_library(self) -> None:
        library = _FakeLibrary()
        del library.spark_tp4_create_with_protocol_graph_kernel_and_schedule
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "stale.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "lacks the wire-schedule ABI",
            ):
                spark_tp4_backend._NativeSession(
                    0,
                    40 * 6144 * 2,
                    control_ports=(14000, 14001),
                    graph_only=True,
                    graph_cpu_affinity=(10, 11),
                    allreduce_protocol="two_slot_deferred_ack",
                    wire_schedule="dual_port_striped",
                )

        self.assertEqual(library.configs, [])

    def test_graph_status_rejects_wire_schedule_attestation_mismatch(self) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                40 * 6144 * 2,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
                allreduce_protocol="two_slot_deferred_ack",
                wire_schedule="dual_port_striped",
            )
            library.wire_schedules.clear()
            with self.assertRaisesRegex(
                RuntimeError,
                "wire schedule attestation mismatch",
            ):
                session.graph_status()

    def test_tiered_graph_kernel_uses_additive_create_symbol(self) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
                allreduce_protocol="two_slot_deferred_ack",
                graph_kernel_strategy="tiered_64k",
            )
            status = session.graph_status()

        self.assertEqual(library.protocols, [1])
        self.assertEqual(library.graph_kernels, [2])
        self.assertEqual(status.graph_kernel_strategy, "tiered_64k")

    def test_tiered_graph_kernel_rejects_stale_native_library(self) -> None:
        library = _FakeLibrary()
        del library.spark_tp4_create_with_protocol_and_graph_kernel
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "stale.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "lacks the graph-kernel ABI",
            ):
                spark_tp4_backend._NativeSession(
                    0,
                    73728,
                    control_ports=(14000, 14001),
                    graph_only=True,
                    graph_cpu_affinity=(10, 11),
                    graph_kernel_strategy="tiered_64k",
                )

        self.assertEqual(library.configs, [])

    def test_graph_status_rejects_kernel_attestation_mismatch(self) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
                graph_kernel_strategy="tiered_64k",
            )
            library.graph_kernels.clear()
            with self.assertRaisesRegex(
                RuntimeError,
                "kernel attestation mismatch",
            ):
                session.graph_status()

    def test_two_slot_protocol_rejects_stale_native_library(self) -> None:
        library = _FakeLibrary()
        del library.spark_tp4_create_with_protocol
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "stale.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "lacks the deferred-ACK ABI",
            ):
                spark_tp4_backend._NativeSession(
                    0,
                    73728,
                    control_ports=(14000, 14001),
                    graph_only=True,
                    graph_cpu_affinity=(10, 11),
                    allreduce_protocol="two_slot_deferred_ack",
                )

        self.assertEqual(library.configs, [])

    def test_graph_status_rejects_protocol_attestation_mismatch(self) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
                allreduce_protocol="two_slot_deferred_ack",
            )
            library.protocols.clear()
            with self.assertRaisesRegex(
                RuntimeError,
                "protocol attestation mismatch",
            ):
                session.graph_status()

    def test_prefill_graph_session_has_six_mib_capacity(self) -> None:
        library = _FakeLibrary()
        environment = {
            "SPARK_TP4_LIBRARY": "fake.dll",
            "VLLM_SPARK_TP4_PREFILL_Q512": "1",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(spark_tp4_backend.ctypes, "CDLL", return_value=library),
        ):
            spark_tp4_backend._NativeSession(
                0,
                6 * 1024 * 1024,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
            )

        self.assertEqual(len(library.configs), 1)
        self.assertEqual(
            library.configs[0]["payload_bytes"],
            6 * 1024 * 1024,
        )

    def test_payloads_receive_distinct_ports_and_reach_native_config(
        self,
    ) -> None:
        library = _FakeLibrary()
        environment = {"SPARK_TP4_LIBRARY": "fake.dll"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(spark_tp4_backend.ctypes, "CDLL", return_value=library),
        ):
            for rows in range(1, 7):
                spark_tp4_backend._NativeSession(rows % 4, rows * 12288)

        self.assertEqual(
            [config["payload_bytes"] for config in library.configs],
            [12288, 24576, 36864, 49152, 61440, 73728],
        )
        self.assertEqual(
            [
                (config["control_port0"], config["control_port1"])
                for config in library.configs
            ],
            [
                (11000, 11001),
                (11002, 11003),
                (11004, 11005),
                (11006, 11007),
                (11008, 11009),
                (11010, 11011),
            ],
        )

    def test_k0_preserves_environment_port_compatibility(self) -> None:
        library = _FakeLibrary()
        environment = {
            "SPARK_TP4_LIBRARY": "fake.dll",
            "SPARK_TP4_CONTROL_PORT0": "12000",
            "SPARK_TP4_CONTROL_PORT1": "13000",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(spark_tp4_backend.ctypes, "CDLL", return_value=library),
        ):
            spark_tp4_backend._NativeSession(0, 12288)
            spark_tp4_backend._NativeSession(0, 61440)

        self.assertEqual(
            [
                (config["control_port0"], config["control_port1"])
                for config in library.configs
            ],
            [(12000, 13000), (12008, 13008)],
        )

    def test_graph_q1_uses_dedicated_ports(self) -> None:
        library = _FakeLibrary()
        environment = {
            "SPARK_TP4_LIBRARY": "fake.dll",
            "SPARK_TP4_GRAPH_CONTROL_PORT0": "14000",
            "SPARK_TP4_GRAPH_CONTROL_PORT1": "14001",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(spark_tp4_backend.ctypes, "CDLL", return_value=library),
        ):
            spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=spark_tp4_backend._graph_control_ports(),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
            )

        self.assertEqual(len(library.configs), 1)
        self.assertEqual(library.configs[0]["payload_bytes"], 73728)
        self.assertEqual(
            (
                library.configs[0]["control_port0"],
                library.configs[0]["control_port1"],
            ),
            (14000, 14001),
        )
        self.assertEqual(library.configs[0]["graph_submit_cpu_plus_one"], 11)
        self.assertEqual(library.configs[0]["graph_progress_cpu_plus_one"], 12)

    def test_graph_session_rejects_less_than_q6_capacity(self) -> None:
        with patch.object(spark_tp4_backend.ctypes, "CDLL") as library_loader:
            with self.assertRaisesRegex(ValueError, "requires Q6 capacity"):
                spark_tp4_backend._NativeSession(
                    0,
                    12288,
                    control_ports=(14000, 14001),
                    graph_only=True,
                    graph_cpu_affinity=(10, 11),
                )

        library_loader.assert_not_called()

    def test_graph_status_exposes_native_replay_advancement(self) -> None:
        library = _FakeLibrary()
        environment = {"SPARK_TP4_LIBRARY": "fake.dll"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(spark_tp4_backend.ctypes, "CDLL", return_value=library),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
            )
            status = session.graph_status()

        self.assertEqual(status.captured_nodes, 128)
        self.assertEqual(status.published_sequence, 1024)
        self.assertEqual(status.consumed_sequence, 1024)
        self.assertEqual(status.completed_sequence, 1023)
        self.assertEqual(status.overflow_sequence, 0)
        self.assertEqual(status.submit_cpu, 10)
        self.assertEqual(status.progress_cpu, 11)
        self.assertTrue(status.capture_configured)
        self.assertTrue(status.polling_enabled)
        self.assertTrue(status.host_native_atomics)
        self.assertTrue(status.submit_affinity_verified)
        self.assertTrue(status.progress_affinity_verified)
        self.assertFalse(status.two_slot_deferred_ack)
        self.assertFalse(status.command_caught_up)
        self.assertFalse(status.payload_slots_retired)
        self.assertTrue(status.replay_advanced)
        self.assertFalse(status.replay_caught_up)
        self.assertFalse(status.fatal)

    def test_generic_graph_capture_passes_dynamic_q_to_native_abi(self) -> None:
        class _NativeTensor:
            shape = (3, 6144)
            dtype = "torch.bfloat16"
            is_cuda = True
            device = "cuda:0"

            def is_contiguous(self) -> bool:
                return True

            def data_ptr(self) -> int:
                return 0x1234

        output = _NativeTensor()
        output.data_ptr = lambda: 0x5678
        fake_torch = types.ModuleType("torch")
        fake_torch.empty_like = lambda tensor: output
        fake_torch.cuda = types.SimpleNamespace(
            current_stream=lambda device: types.SimpleNamespace(
                cuda_stream=0x9ABC
            )
        )
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
            )
            result = session.capture(_NativeTensor())

        self.assertIs(result, output)
        self.assertEqual(len(library.capture_calls), 1)
        call = library.capture_calls[0]
        self.assertEqual(call["q"], 3)
        self.assertEqual(call["input"].value, 0x1234)
        self.assertEqual(call["output"].value, 0x5678)
        self.assertEqual(call["stream"].value, 0x9ABC)

    def test_backend_builds_a_separate_exact_q40_striped_session(self) -> None:
        created: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class _PreparedSession:
            def __init__(self, *args: object, **kwargs: object) -> None:
                created.append((args, kwargs))

        reporter = types.ModuleType("spark_graph_status_reporter")

        def ensure_status_reporter(*, rank: int) -> None:
            self.assertEqual(rank, 2)

        reporter.ensure_status_reporter = ensure_status_reporter
        with (
            patch.dict(
                os.environ,
                {"VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40": "1"},
                clear=True,
            ),
            patch.dict(
                sys.modules,
                {"spark_graph_status_reporter": reporter},
            ),
            patch.object(
                spark_tp4_backend,
                "_NativeSession",
                _PreparedSession,
            ),
            patch.object(
                spark_tp4_backend,
                "_graph_preflight",
                return_value=(10, 11),
            ),
            patch.object(
                spark_tp4_backend,
                "_graph_dual_port_q40_cpu_affinity",
                return_value=(10, 15),
            ),
            patch.object(
                spark_tp4_backend,
                "_graph_control_ports",
                return_value=(9970, 9971),
            ),
            patch.object(
                spark_tp4_backend,
                "_graph_dual_port_q40_control_ports",
                return_value=(9972, 9973),
            ),
            patch.dict(
                spark_tp4_backend._graph_q1_sessions,
                {},
                clear=True,
            ),
            patch.dict(
                spark_tp4_backend._graph_dual_port_q40_sessions,
                {},
                clear=True,
            ),
        ):
            backend = spark_tp4_backend._Backend(2)
            backend.prepare_graph_q1()

        self.assertEqual(len(created), 2)
        base_args, base_kwargs = created[0]
        striped_args, striped_kwargs = created[1]
        self.assertEqual(
            base_args,
            (2, spark_tp4_backend._graph_capacity_bytes()),
        )
        self.assertEqual(base_kwargs["control_ports"], (9970, 9971))
        self.assertNotIn("wire_schedule", base_kwargs)
        self.assertEqual(
            striped_args,
            (2, 40 * 6144 * 2),
        )
        self.assertEqual(striped_kwargs["control_ports"], (9972, 9973))
        self.assertEqual(striped_kwargs["graph_cpu_affinity"], (10, 15))
        self.assertEqual(
            striped_kwargs["allreduce_protocol"],
            "two_slot_deferred_ack",
        )
        self.assertEqual(
            striped_kwargs["wire_schedule"],
            "dual_port_striped",
        )

    def test_generic_graph_capture_rejects_non_exact_signature(self) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                73728,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 11),
            )
            for tensor in (
                _FakeTensor(shape=(7, 6144)),
                _FakeTensor(dtype="torch.float16"),
                _FakeTensor(is_cuda=False),
                _FakeTensor(contiguous=False),
            ):
                with self.subTest(tensor=tensor):
                    with self.assertRaisesRegex(
                        ValueError, "requires contiguous CUDA BF16"
                    ):
                        session.capture(tensor)

        self.assertEqual(library.capture_calls, [])

    def test_dual_port_graph_capture_requires_exact_fixed_q(self) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "fake.dll"},
                clear=True,
            ),
            patch.object(
                spark_tp4_backend.ctypes,
                "CDLL",
                return_value=library,
            ),
            patch.object(
                spark_tp4_backend,
                "_target_shape_eligible",
                return_value=True,
            ),
        ):
            session = spark_tp4_backend._NativeSession(
                0,
                40 * 6144 * 2,
                control_ports=(14000, 14001),
                graph_only=True,
                graph_cpu_affinity=(10, 15),
                allreduce_protocol="two_slot_deferred_ack",
                wire_schedule="dual_port_striped",
            )
            with self.assertRaisesRegex(
                ValueError,
                "exact fixed Q",
            ):
                session.capture(_FakeTensor(shape=(39, 6144)))

        self.assertEqual(library.capture_calls, [])

    def test_process_snapshot_is_read_only_and_rpc_serializable(self) -> None:
        class _StatusSession:
            def graph_status(self):
                return spark_tp4_backend.GraphReplayStatus(
                    captured_nodes=4,
                    published_sequence=9,
                    consumed_sequence=9,
                    completed_sequence=9,
                    overflow_sequence=0,
                    capture_configured=True,
                    polling_enabled=True,
                    host_native_atomics=True,
                    submit_affinity_verified=True,
                    progress_affinity_verified=True,
                    submit_cpu=10,
                    progress_cpu=11,
                )

        with (
            patch.dict(
                spark_tp4_backend._graph_q1_sessions,
                {2: _StatusSession()},
                clear=True,
            ),
            patch.dict(
                spark_tp4_backend._graph_dual_port_q40_sessions,
                {2: _StatusSession()},
                clear=True,
            ),
        ):
            snapshot = spark_tp4_backend.graph_q1_status_snapshot()
            diagnostic = spark_tp4_backend.graph_q1_diagnostic_snapshot()

        self.assertEqual(snapshot[2]["captured_nodes"], 4)
        self.assertEqual(snapshot[2]["completed_sequence"], 9)
        self.assertTrue(snapshot[2]["replay_advanced"])
        self.assertTrue(snapshot[2]["replay_caught_up"])
        self.assertFalse(snapshot[2]["fatal"])
        self.assertEqual(
            diagnostic["dual_port_q40_sessions"][2]["captured_nodes"],
            4,
        )

    def test_graph_preflight_requires_positive_fixed_kv_cache(self) -> None:
        environment = {
            "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
            "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
            "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(spark_tp4_backend.sys, "platform", "linux"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "--kv-cache-memory-bytes"
            ):
                spark_tp4_backend._graph_preflight(["vllm", "serve"])
            with self.assertRaisesRegex(
                RuntimeError, "must be a positive integer"
            ):
                spark_tp4_backend._graph_preflight(
                    ["vllm", "serve", "--kv-cache-memory-bytes", "0"]
                )

    def test_graph_preflight_requires_distinct_cpus_and_linux(self) -> None:
        argv = ["vllm", "serve", "--kv-cache-memory-bytes=5500000000"]
        with patch.dict(
            os.environ,
            {
                "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
                "SPARK_TP4_GRAPH_PROGRESS_CPU": "10",
                "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
            },
            clear=True,
        ):
            with (
                patch.object(spark_tp4_backend.sys, "platform", "linux"),
                self.assertRaisesRegex(RuntimeError, "must be distinct"),
            ):
                spark_tp4_backend._graph_preflight(argv)

        with patch.dict(
            os.environ,
            {
                "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
                "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
                "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
            },
            clear=True,
        ):
            with (
                patch.object(spark_tp4_backend.sys, "platform", "win32"),
                self.assertRaisesRegex(RuntimeError, "requires Linux"),
            ):
                spark_tp4_backend._graph_preflight(argv)

    def test_graph_preflight_accepts_safe_contract(self) -> None:
        argv = ["vllm", "serve", "--kv-cache-memory-bytes", "5500000000"]
        with (
            patch.dict(
                os.environ,
                {
                    "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
                    "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
                    "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
                },
                clear=True,
            ),
            patch.object(spark_tp4_backend.sys, "platform", "linux"),
        ):
            self.assertEqual(
                spark_tp4_backend._graph_preflight(argv),
                (10, 11),
            )

    def test_dual_port_graph_preflight_reserves_a_distinct_progress_cpu(
        self,
    ) -> None:
        argv = ["vllm", "serve", "--kv-cache-memory-bytes", "5500000000"]
        environment = {
            "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
            "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
            "SPARK_TP4_GRAPH_DUAL_PORT_Q40_PROGRESS_CPU": "15",
            "SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU": "12",
            "SPARK_TP4_GRAPH_DCP_PROGRESS_CPU": "13",
            "SPARK_TP4_GRAPH_INDEXER_PROGRESS_CPU": "14",
            "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
            "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
            "VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM": "1",
            "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": "1",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(spark_tp4_backend.sys, "platform", "linux"),
        ):
            self.assertEqual(
                spark_tp4_backend._graph_dual_port_q40_cpu_affinity(argv),
                (10, 15),
            )
            for occupied in (10, 11, 12, 13, 14):
                os.environ[
                    "SPARK_TP4_GRAPH_DUAL_PORT_Q40_PROGRESS_CPU"
                ] = str(occupied)
                with self.subTest(occupied=occupied), self.assertRaisesRegex(
                    RuntimeError,
                    "collides with",
                ):
                    spark_tp4_backend._graph_dual_port_q40_cpu_affinity(argv)

    def test_graph_preflight_requires_shared_capture_stream(self) -> None:
        argv = ["vllm", "serve", "--kv-cache-memory-bytes=5500000000"]
        with (
            patch.dict(
                os.environ,
                {
                    "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
                    "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
                },
                clear=True,
            ),
            patch.object(spark_tp4_backend.sys, "platform", "linux"),
            self.assertRaisesRegex(
                RuntimeError, "VLLM_SPARK_SHARED_CAPTURE_STREAM=1"
            ),
        ):
            spark_tp4_backend._graph_preflight(argv)

    def test_unsupported_payload_is_rejected_before_library_load(self) -> None:
        with patch.object(spark_tp4_backend.ctypes, "CDLL") as library_loader:
            with self.assertRaisesRegex(
                ValueError, "unsupported Spark TP4 payload size"
            ):
                spark_tp4_backend._NativeSession(0, 15 * 12288)

        library_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
