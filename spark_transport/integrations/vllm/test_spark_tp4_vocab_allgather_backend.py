"""GPU-free tests for the Spark TP4 vocabulary all-gather adapter."""

from __future__ import annotations

import os
import sys
import types
import unittest
from typing import ClassVar, Self
from unittest.mock import patch

import spark_collective_audit
import spark_tp4_vocab_allgather_backend as backend_module


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def __iadd__(self, other: Self) -> Self:
        self.value += other.value
        return self

    def item(self) -> int:
        return self.value


class _FakeDifference:
    def __init__(self, mismatches: int) -> None:
        self.mismatches = mismatches


class _FakeByteView:
    def __init__(self, tensor: _FakeTensor) -> None:
        self.tensor = tensor

    def __ne__(self, other: _FakeByteView) -> _FakeDifference:
        mismatches = 0 if self.tensor.payload == other.tensor.payload else 1
        return _FakeDifference(mismatches)


class _FakeTensor:
    _next_pointer = 0x1000

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str = "torch.bfloat16",
        *,
        is_cuda: bool = True,
        contiguous: bool = True,
        payload: int = 0,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = "cuda:0" if is_cuda else "cpu"
        self._contiguous = contiguous
        self.payload = payload
        self.pointer = self._next_pointer
        _FakeTensor._next_pointer += 0x1000

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return self.pointer

    def view(self, dtype: object) -> _FakeByteView:
        del dtype
        return _FakeByteView(self)


class _FakeStream:
    cuda_stream = 0xCAFE


class _FakeCuda:
    def __init__(self) -> None:
        self.capturing = False
        self.stream = _FakeStream()

    def current_stream(self, *, device: object) -> _FakeStream:
        if device != "cuda:0":
            raise AssertionError(f"unexpected device: {device}")
        return self.stream

    def is_current_stream_capturing(self) -> bool:
        return self.capturing


def _fake_torch_module() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.uint8 = "torch.uint8"
    module.cuda = _FakeCuda()
    module.allocations = []

    def empty(
        shape: tuple[int, ...], *, dtype: str, device: object
    ) -> _FakeTensor:
        if device != "cuda:0":
            raise AssertionError(f"unexpected device: {device}")
        tensor = _FakeTensor(shape, dtype)
        module.allocations.append(tensor)
        return tensor

    def count_nonzero(difference: _FakeDifference) -> _FakeScalar:
        return _FakeScalar(difference.mismatches)

    module.empty = empty
    module.count_nonzero = count_nonzero
    return module


def _make_group_type() -> type:
    class FakeGroupCoordinator:
        def __init__(
            self,
            *,
            unique_name: str = "tp:0",
            world_size: int = 4,
            rank_in_group: int = 2,
        ) -> None:
            self.unique_name = unique_name
            self.world_size = world_size
            self.rank_in_group = rank_in_group
            self.fail_original = False
            self.original_calls: list[tuple[object, int]] = []

        def _all_gather_out_place(
            self, input_tensor: _FakeTensor, dim: int
        ) -> _FakeTensor:
            self.original_calls.append((input_tensor, dim))
            if self.fail_original:
                raise RuntimeError("reference failed")
            q = int(input_tensor.shape[0])
            return _FakeTensor(
                (q, 154880),
                input_tensor.dtype,
                payload=input_tensor.payload,
            )

    return FakeGroupCoordinator


def _fake_modules(
    group_type: type, torch_module: types.ModuleType
) -> dict[str, types.ModuleType]:
    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.GroupCoordinator = group_type
    return {
        "torch": torch_module,
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": parallel_state,
    }


class _FakeNativeSession:
    created: ClassVar[list[Self]] = []
    fail_create = False
    fail_graph_create = False
    fail_call = False
    mismatch = False

    def __init__(
        self,
        rank: int,
        *,
        graph_only: bool = False,
        control_ports: tuple[int, int] | None = None,
        graph_cpu_affinity: tuple[int, int] | None = None,
    ) -> None:
        if self.fail_create or (graph_only and self.fail_graph_create):
            raise RuntimeError("create failed")
        self.rank = rank
        self.graph_only = graph_only
        self.control_ports = control_ports
        self.graph_cpu_affinity = graph_cpu_affinity
        self.calls: list[tuple[object, object, int, object]] = []
        self.capture_calls: list[tuple[object, object, int, object]] = []
        self.created.append(self)

    def all_gather(
        self,
        input_tensor: _FakeTensor,
        output_tensor: _FakeTensor,
        query_rows: int,
        stream: object,
    ) -> None:
        if self.fail_call:
            raise RuntimeError("native call failed")
        output_tensor.payload = (
            input_tensor.payload + 1
            if self.mismatch
            else input_tensor.payload
        )
        self.calls.append(
            (input_tensor, output_tensor, query_rows, stream)
        )

    def capture(
        self,
        input_tensor: _FakeTensor,
        output_tensor: _FakeTensor,
        query_rows: int,
        stream: object,
    ) -> None:
        if not self.graph_only:
            raise RuntimeError("not a graph session")
        self.capture_calls.append(
            (input_tensor, output_tensor, query_rows, stream)
        )

    def graph_status(self) -> dict[str, object]:
        if not self.graph_only:
            raise RuntimeError("not a graph session")
        return {"captured_nodes": len(self.capture_calls)}


class _AbortCalled(RuntimeError):
    pass


class SparkTp4VocabDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.group_type = _make_group_type()
        self.torch_module = _fake_torch_module()
        self.modules = _fake_modules(
            self.group_type, self.torch_module
        )
        self.original = self.group_type._all_gather_out_place
        backend_module._installed = False
        backend_module._vocab_graph_sessions.clear()
        backend_module._graph_event_counts.clear()
        spark_collective_audit._reset_for_tests()
        _FakeNativeSession.created.clear()
        _FakeNativeSession.fail_create = False
        _FakeNativeSession.fail_graph_create = False
        _FakeNativeSession.fail_call = False
        _FakeNativeSession.mismatch = False

    def _install(self, mode: str | None, **environment: str) -> None:
        values = dict(environment)
        if mode is not None:
            values["VLLM_SPARK_TP4_VOCAB_MODE"] = mode
        patchers = (
            patch.dict(os.environ, values, clear=True),
            patch.dict(sys.modules, self.modules),
            patch.object(
                backend_module,
                "_NativeVocabSession",
                _FakeNativeSession,
            ),
            patch.object(
                backend_module,
                "_abort_after_native_failure",
                side_effect=_AbortCalled("abort"),
            ),
            patch.object(
                backend_module,
                "_graph_preflight",
                return_value=(10, 12),
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        backend_module.install()

    def test_unset_mode_does_not_patch_group(self) -> None:
        self._install(None)
        self.assertIs(
            self.group_type._all_gather_out_place, self.original
        )
        self.assertFalse(backend_module._installed)

    def test_invalid_mode_and_shadow_limit_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "must be 'shadow', 'custom'"
        ):
            self._install("fast")
        self.assertIs(
            self.group_type._all_gather_out_place, self.original
        )

        backend_module._installed = False
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._install(
                "shadow", SPARK_TP4_VOCAB_SHADOW_COLLECTIVES="0"
            )

    def test_custom_routes_q1_through_q6_on_one_native_handle(
        self,
    ) -> None:
        self._install("custom")
        group = self.group_type(rank_in_group=3)
        for q in range(1, 7):
            input_tensor = _FakeTensor(
                (q, 38720), payload=100 + q
            )
            output = group._all_gather_out_place(input_tensor, -1)
            self.assertEqual(output.shape, (q, 154880))
            self.assertEqual(output.payload, input_tensor.payload)

        self.assertEqual(group.original_calls, [])
        self.assertEqual(len(_FakeNativeSession.created), 1)
        native = _FakeNativeSession.created[0]
        self.assertEqual(native.rank, 3)
        self.assertEqual(
            [call[2] for call in native.calls], [1, 2, 3, 4, 5, 6]
        )
        self.assertTrue(
            all(
                call[3] is self.torch_module.cuda.stream
                for call in native.calls
            )
        )

    def test_positive_and_negative_last_dim_are_equivalent(self) -> None:
        self._install("custom")
        group = self.group_type()
        for dim in (-1, 1):
            output = group._all_gather_out_place(
                _FakeTensor((3, 38720), payload=33), dim
            )
            self.assertEqual(output.shape, (3, 154880))
        self.assertEqual(group.original_calls, [])

    def test_non_exact_signatures_and_capture_use_original(self) -> None:
        self._install("custom")
        cases = (
            (
                self.group_type(unique_name="dcp:0"),
                _FakeTensor((3, 38720)),
                -1,
            ),
            (
                self.group_type(world_size=2),
                _FakeTensor((3, 38720)),
                -1,
            ),
            (self.group_type(), _FakeTensor((7, 38720)), -1),
            (self.group_type(), _FakeTensor((3, 38719)), -1),
            (
                self.group_type(),
                _FakeTensor((3, 38720), "torch.float16"),
                -1,
            ),
            (
                self.group_type(),
                _FakeTensor((3, 38720), is_cuda=False),
                -1,
            ),
            (
                self.group_type(),
                _FakeTensor((3, 38720), contiguous=False),
                -1,
            ),
            (self.group_type(), _FakeTensor((3, 38720)), 0),
        )
        for group, tensor, dim in cases:
            with self.subTest(
                group=group.unique_name, shape=tensor.shape, dim=dim
            ):
                result = group._all_gather_out_place(tensor, dim)
                self.assertEqual(result.shape, (tensor.shape[0], 154880))
                self.assertEqual(len(group.original_calls), 1)

        capture_group = self.group_type()
        self.torch_module.cuda.capturing = True
        capture_group._all_gather_out_place(
            _FakeTensor((3, 38720)), -1
        )
        self.assertEqual(len(capture_group.original_calls), 1)
        self.assertEqual(_FakeNativeSession.created, [])

    def test_dcp2_capture_miss_records_bounded_signature(self) -> None:
        self._install(
            "custom",
            SPARK_TP4_GRAPH_STATUS_PATH="/tmp/status.json",
        )
        self.torch_module.cuda.capturing = True
        group = self.group_type(
            unique_name="dcp:0",
            world_size=2,
            rank_in_group=1,
        )
        tensor = _FakeTensor((5, 2048), "torch.int32")

        group._all_gather_out_place(tensor, 0)

        snapshot = spark_collective_audit.stock_collective_snapshot()
        self.assertEqual(
            snapshot["capture"],
            {"dcp_all_gather:ineligible_signature": 1},
        )
        self.assertEqual(
            snapshot["signatures"]["capture"],
            [
                {
                    "family": "dcp_all_gather",
                    "reason": "ineligible_signature",
                    "count": 1,
                    "shape": [5, 2048],
                    "dtype": "torch.int32",
                    "is_cuda": True,
                    "contiguous": True,
                    "world_size": 2,
                    "unique_name": "dcp:0",
                }
            ],
        )

    def test_dcp4_query_and_lse_are_classified_by_real_family(self) -> None:
        self._install(
            "custom",
            SPARK_TP4_GRAPH_STATUS_PATH="/tmp/status.json",
        )
        self.torch_module.cuda.capturing = True
        group = self.group_type(
            unique_name="dcp:0",
            world_size=4,
            rank_in_group=1,
        )

        group._all_gather_out_place(
            _FakeTensor((40, 16, 576), "torch.bfloat16"),
            1,
        )
        group._all_gather_out_place(
            _FakeTensor((40, 64), "torch.float32"),
            1,
        )

        snapshot = spark_collective_audit.stock_collective_snapshot()
        # dim=1 LSE gathers classify as generic dcp_all_gather by design
        # (dcp_lse_all_gather requires dim in {0, -2}).
        self.assertEqual(
            snapshot["capture"],
            {
                "dcp_all_gather:ineligible_signature": 1,
                "dcp_query_all_gather:ineligible_signature": 1,
            },
        )
        self.assertEqual(
            {
                item["family"]
                for item in snapshot["signatures"]["capture"]
            },
            {"dcp_all_gather", "dcp_query_all_gather"},
        )

    def test_graph_mode_uses_eager_native_and_capture_graph_sessions(
        self,
    ) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_GRAPH_Q1="1",
            SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0="10110",
            SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1="10111",
            SPARK_TP4_GRAPH_STATUS_PATH="/tmp/status.json",
        )
        group = self.group_type(rank_in_group=2)

        first_warmup_stream = self.torch_module.cuda.stream
        group._all_gather_out_place(
            _FakeTensor((3, 38720), payload=30), -1
        )
        second_warmup_stream = types.SimpleNamespace(cuda_stream=0xD00D)
        self.torch_module.cuda.stream = second_warmup_stream
        group._all_gather_out_place(
            _FakeTensor((2, 38720), payload=20), -1
        )
        self.assertEqual(len(_FakeNativeSession.created), 2)
        graph, eager = _FakeNativeSession.created
        self.assertTrue(graph.graph_only)
        self.assertFalse(eager.graph_only)
        self.assertEqual(graph.control_ports, (10110, 10111))
        self.assertEqual(graph.graph_cpu_affinity, (10, 12))
        self.assertEqual(group.original_calls, [])
        self.assertEqual(
            [call[2] for call in eager.calls],
            [3, 2],
        )
        self.assertIs(
            eager.calls[0][3],
            first_warmup_stream,
        )
        self.assertIs(
            eager.calls[1][3],
            second_warmup_stream,
        )
        self.assertEqual(
            spark_collective_audit.stock_collective_snapshot()["eager"],
            {},
        )

        capture_stream = types.SimpleNamespace(cuda_stream=0xBEEF)
        self.torch_module.cuda.stream = capture_stream
        self.torch_module.cuda.capturing = True
        inputs = [
            _FakeTensor((q, 38720), payload=100 + q)
            for q in range(1, 7)
        ]
        outputs = [
            group._all_gather_out_place(input_tensor, -1)
            for input_tensor in inputs
        ]

        self.assertEqual([output.shape for output in outputs], [
            (q, 154880) for q in range(1, 7)
        ])
        self.assertEqual([call[2] for call in graph.capture_calls], [
            1, 2, 3, 4, 5, 6
        ])
        self.assertTrue(all(
            call[3] is capture_stream
            for call in graph.capture_calls
        ))
        self.assertEqual(
            group._spark_tp4_vocab_graph_captured_nodes,
            6,
        )
        self.assertEqual(
            backend_module.vocab_graph_diagnostic_snapshot()["events"],
            {"captured_nodes": 6},
        )
        self.assertFalse(
            hasattr(group, "_spark_tp4_vocab_graph_cold_fallbacks")
        )
        self.assertEqual(group.original_calls, [])

    def test_unprepared_graph_capture_aborts_without_stock_fallback(
        self,
    ) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_GRAPH_Q1="1",
            SPARK_TP4_GRAPH_STATUS_PATH="/tmp/status.json",
        )
        group = self.group_type()
        self.torch_module.cuda.capturing = True

        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(
                _FakeTensor((5, 38720), payload=50), -1
            )

        self.assertEqual(group.original_calls, [])
        self.assertEqual(_FakeNativeSession.created, [])
        self.assertEqual(
            group._spark_tp4_vocab_graph_cold_fallbacks,
            1,
        )
        self.assertFalse(
            hasattr(group, "_spark_tp4_vocab_graph_captured_nodes")
        )

    def test_unprepared_capture_never_records_a_stock_collective(self) -> None:
        self._install("custom", VLLM_SPARK_TP4_GRAPH_Q1="1")
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        group = self.group_type()
        self.torch_module.cuda.capturing = True

        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(
                _FakeTensor((5, 38720), payload=50), -1
            )

        self.assertEqual(
            spark_collective_audit.stock_collective_snapshot()["capture"],
            {},
        )

    def test_graph_session_warmup_failure_aborts_without_stock_fallback(
        self,
    ) -> None:
        _FakeNativeSession.fail_graph_create = True
        self._install(
            "custom",
            VLLM_SPARK_TP4_GRAPH_Q1="1",
            SPARK_TP4_GRAPH_STATUS_PATH="/tmp/status.json",
        )
        group = self.group_type()

        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(
                _FakeTensor((2, 38720), payload=20), -1
            )

        self.assertEqual(group.original_calls, [])
        state = group._spark_tp4_vocab_native
        self.assertIsNone(state._session)
        self.assertIsNone(state._graph_session)
        self.assertTrue(state.graph_disabled)
        self.assertEqual(
            spark_collective_audit.stock_collective_snapshot()["eager"],
            {},
        )

    def test_shadow_never_prepares_or_captures_graph_session(self) -> None:
        self._install("shadow", VLLM_SPARK_TP4_GRAPH_Q1="1")
        group = self.group_type()
        group._all_gather_out_place(_FakeTensor((1, 38720)), -1)

        self.assertEqual(len(_FakeNativeSession.created), 1)
        self.assertFalse(_FakeNativeSession.created[0].graph_only)
        self.torch_module.cuda.capturing = True
        group._all_gather_out_place(_FakeTensor((1, 38720)), -1)
        self.assertEqual(len(group.original_calls), 2)

    def test_shadow_promotes_each_q_independently(self) -> None:
        self._install(
            "shadow",
            SPARK_TP4_VOCAB_SHADOW_COLLECTIVES="2",
            SPARK_TP4_VOCAB_SHADOW_PROMOTE="1",
        )
        group = self.group_type()
        q1_input = _FakeTensor((1, 38720), payload=11)
        q3_input = _FakeTensor((3, 38720), payload=33)

        group._all_gather_out_place(q1_input, -1)
        group._all_gather_out_place(q1_input, -1)
        group._all_gather_out_place(q3_input, -1)
        promoted = group._all_gather_out_place(q1_input, -1)

        self.assertEqual(promoted.payload, 11)
        self.assertEqual(len(group.original_calls), 3)
        state = group._spark_tp4_vocab_native
        self.assertTrue(state.shadows[1].validated)
        self.assertEqual(state.shadows[3].count, 1)
        self.assertFalse(state.shadows[3].validated)
        self.assertEqual(
            [
                call[2]
                for call in _FakeNativeSession.created[0].calls
            ],
            [1, 1, 3, 1],
        )

    def test_shadow_mismatch_fails_validation(self) -> None:
        _FakeNativeSession.mismatch = True
        self._install(
            "shadow", SPARK_TP4_VOCAB_SHADOW_COLLECTIVES="1"
        )
        group = self.group_type()
        with self.assertRaisesRegex(RuntimeError, "byte mismatches"):
            group._all_gather_out_place(
                _FakeTensor((2, 38720), payload=20), -1
            )
        self.assertFalse(
            group._spark_tp4_vocab_native.shadows[2].validated
        )

    def test_custom_create_failure_aborts_without_stock_fallback(self) -> None:
        _FakeNativeSession.fail_create = True
        self._install("custom")
        group = self.group_type()
        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(
                _FakeTensor((3, 38720), payload=30), -1
            )
        self.assertEqual(group.original_calls, [])
        self.assertTrue(group._spark_tp4_vocab_native.disabled)

    def test_native_failure_after_enqueue_aborts_worker(self) -> None:
        _FakeNativeSession.fail_call = True
        self._install("custom")
        group = self.group_type()
        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(
                _FakeTensor((3, 38720)), -1
            )

    def test_reference_failure_after_enqueue_aborts_worker(self) -> None:
        self._install("shadow")
        group = self.group_type()
        group.fail_original = True
        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(
                _FakeTensor((3, 38720)), -1
            )

    def test_install_is_idempotent(self) -> None:
        self._install("custom")
        patched = self.group_type._all_gather_out_place
        backend_module.install()
        self.assertIs(
            self.group_type._all_gather_out_place, patched
        )


class _FakeFunction:
    def __init__(
        self, result: object = None, implementation: object = None
    ) -> None:
        self.result = result
        self.implementation = implementation
        self.calls: list[tuple[object, ...]] = []
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        if self.implementation is not None:
            return self.implementation(*args)
        return self.result


class _FakeLibrary:
    def __init__(self) -> None:
        self.spark_tp4_vocab_allgather_create = _FakeFunction(0x1234)
        self.spark_tp4_vocab_graph_create = _FakeFunction(0x5678)
        self.spark_tp4_vocab_allgather = _FakeFunction(0)
        self.spark_tp4_vocab_capture_allgather = _FakeFunction(0)
        self.spark_tp4_vocab_get_graph_status = _FakeFunction(
            implementation=self._graph_status
        )
        self.spark_tp4_vocab_allgather_destroy = _FakeFunction(None)

    @staticmethod
    def _graph_status(
        handle, status_pointer, status_bytes, error, error_bytes
    ) -> int:
        del handle, status_bytes, error, error_bytes
        status = status_pointer._obj
        status.struct_size = backend_module.ctypes.sizeof(
            backend_module._NativeVocabGraphStatus
        )
        status.flags = (
            backend_module._GRAPH_STATUS_CAPTURE_CONFIGURED
            | backend_module._GRAPH_STATUS_POLLING_ENABLED
            | backend_module._GRAPH_STATUS_HOST_NATIVE_ATOMICS
            | backend_module._GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED
            | backend_module._GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED
        )
        status.captured_nodes = 5
        status.published_sequence = 8
        status.consumed_sequence = 8
        status.completed_sequence = 8
        status.overflow_sequence = 0
        status.graph_submit_cpu_plus_one = 11
        status.graph_progress_cpu_plus_one = 13
        return 0


class NativeBindingTest(unittest.TestCase):
    def test_vocab_graph_uses_shared_submit_and_dedicated_progress_cpu(
        self,
    ) -> None:
        tp_backend = types.ModuleType("spark_tp4_backend")
        tp_backend._graph_preflight = lambda: (10, 11)
        with (
            patch.dict(
                sys.modules, {"spark_tp4_backend": tp_backend}
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(
                backend_module._graph_preflight(),
                (10, 12),
            )

        with (
            patch.dict(
                sys.modules, {"spark_tp4_backend": tp_backend}
            ),
            patch.dict(
                os.environ,
                {"SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU": "11"},
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "must differ"
            ):
                backend_module._graph_preflight()

    def test_binds_dynamic_q_vocabulary_call(self) -> None:
        library = _FakeLibrary()
        environment = {
            "SPARK_TP4_LIBRARY": "/tmp/libspark_transport_capi.so",
            "SPARK_TP4_VOCAB_CONTROL_PORT0": "10090",
            "SPARK_TP4_VOCAB_CONTROL_PORT1": "10091",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                backend_module.ctypes, "CDLL", return_value=library
            ),
        ):
            session = backend_module._NativeVocabSession(2)
            input_tensor = _FakeTensor((5, 38720))
            output_tensor = _FakeTensor((5, 154880))
            stream = _FakeStream()
            session.all_gather(
                input_tensor, output_tensor, 5, stream
            )

        create_args = (
            library.spark_tp4_vocab_allgather_create.calls[0]
        )
        config = create_args[0]._obj
        self.assertEqual(config.rank, 2)
        self.assertEqual(config.control_port0, 10090)
        self.assertEqual(config.control_port1, 10091)
        self.assertFalse(
            hasattr(config, "graph_submit_cpu_plus_one")
        )
        self.assertFalse(
            hasattr(config, "graph_progress_cpu_plus_one")
        )
        self.assertEqual(
            len(library.spark_tp4_vocab_graph_create.calls), 0
        )
        gather_args = library.spark_tp4_vocab_allgather.calls[0]
        self.assertEqual(gather_args[0], 0x1234)
        self.assertEqual(
            gather_args[1].value, input_tensor.data_ptr()
        )
        self.assertEqual(
            gather_args[2].value, output_tensor.data_ptr()
        )
        self.assertEqual(gather_args[3], 5)
        self.assertEqual(
            gather_args[4].value, stream.cuda_stream
        )

    def test_binds_graph_capture_status_and_affinity_contract(self) -> None:
        library = _FakeLibrary()
        environment = {"SPARK_TP4_LIBRARY": "/tmp/lib.so"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                backend_module.ctypes, "CDLL", return_value=library
            ),
        ):
            session = backend_module._NativeVocabSession(
                1,
                graph_only=True,
                control_ports=(10110, 10111),
                graph_cpu_affinity=(10, 12),
            )
            input_tensor = _FakeTensor((4, 38720))
            output_tensor = _FakeTensor((4, 154880))
            stream = _FakeStream()
            session.capture(
                input_tensor, output_tensor, 4, stream
            )

        self.assertEqual(
            len(library.spark_tp4_vocab_allgather_create.calls), 0
        )
        create_args = library.spark_tp4_vocab_graph_create.calls[0]
        config = create_args[0]._obj
        self.assertEqual(config.control_port0, 10110)
        self.assertEqual(config.control_port1, 10111)
        self.assertEqual(config.graph_submit_cpu_plus_one, 11)
        self.assertEqual(config.graph_progress_cpu_plus_one, 13)
        capture_args = (
            library.spark_tp4_vocab_capture_allgather.calls[0]
        )
        self.assertEqual(capture_args[0], 0x5678)
        self.assertEqual(
            capture_args[1].value, input_tensor.data_ptr()
        )
        self.assertEqual(
            capture_args[2].value, output_tensor.data_ptr()
        )
        self.assertEqual(capture_args[3], 4)
        self.assertEqual(
            capture_args[4].value, stream.cuda_stream
        )
        self.assertIsNotNone(
            library.spark_tp4_vocab_get_graph_status.argtypes
        )
        status = session.graph_status()
        self.assertEqual(status["captured_nodes"], 5)
        self.assertEqual(status["completed_sequence"], 8)
        self.assertTrue(status["replay_advanced"])
        self.assertTrue(status["replay_caught_up"])
        self.assertFalse(status["fatal"])
        self.assertEqual(status["submit_cpu"], 10)
        self.assertEqual(status["progress_cpu"], 12)

    def test_graph_capture_rejects_wrong_layout_and_stream_change(
        self,
    ) -> None:
        library = _FakeLibrary()
        with (
            patch.dict(
                os.environ,
                {"SPARK_TP4_LIBRARY": "/tmp/lib.so"},
                clear=True,
            ),
            patch.object(
                backend_module.ctypes, "CDLL", return_value=library
            ),
        ):
            session = backend_module._NativeVocabSession(
                1,
                graph_only=True,
                control_ports=(10110, 10111),
                graph_cpu_affinity=(10, 12),
            )
            with self.assertRaisesRegex(
                ValueError, "requires contiguous CUDA BF16"
            ):
                session.capture(
                    _FakeTensor((7, 38720)),
                    _FakeTensor((7, 154880)),
                    7,
                    _FakeStream(),
                )

            session.capture(
                _FakeTensor((1, 38720)),
                _FakeTensor((1, 154880)),
                1,
                _FakeStream(),
            )
            other_stream = types.SimpleNamespace(cuda_stream=0xBEEF)
            with self.assertRaisesRegex(
                ValueError, "one stable capture stream"
            ):
                session.capture(
                    _FakeTensor((2, 38720)),
                    _FakeTensor((2, 154880)),
                    2,
                    other_stream,
                )


if __name__ == "__main__":
    unittest.main()
