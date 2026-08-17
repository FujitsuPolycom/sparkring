"""GPU-free dispatch tests for the Spark TP4 all-gather adapter."""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import spark_collective_audit
import spark_tp4_allgather_backend as backend_module


class _FakeCuda:
    def __init__(self) -> None:
        self.capturing = False
        self.current_cuda_stream = 101

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def stream(self, stream: object) -> "_FakeStreamContext":
        return _FakeStreamContext(self, stream)

    def current_stream(self, device: object = None) -> object:
        del device
        return types.SimpleNamespace(
            cuda_stream=self.current_cuda_stream
        )


class _FakeStreamContext:
    def __init__(self, cuda: _FakeCuda, stream: object) -> None:
        self.cuda = cuda
        self.stream = stream
        self.previous = cuda.capturing

    def __enter__(self) -> None:
        self.cuda.capturing = bool(
            getattr(self.stream, "capturing", self.previous)
        )

    def __exit__(self, *args: object) -> None:
        self.cuda.capturing = self.previous


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str,
        *,
        is_cuda: bool = True,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._contiguous = contiguous
        self.device = "cuda:0"

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        product = 1
        for value in self.shape:
            product *= value
        return product


class _FakeBackend:
    created: list["_FakeBackend"] = []

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.calls: list[tuple[int, int, object, object, object]] = []
        self.streams: dict[backend_module._Signature, int] = {}
        self.created.append(self)

    def session_for_stream(
        self,
        signature: backend_module._Signature,
        stream: object,
    ) -> "_FakeBackend | None":
        stream_handle = int(getattr(stream, "cuda_stream", id(stream)))
        bound_stream = self.streams.setdefault(signature, stream_handle)
        if bound_stream != stream_handle:
            return None
        return self.session(signature)

    def session(
        self, signature: backend_module._Signature
    ) -> "_FakeBackend":
        self._pending = signature
        return self

    def all_gather(
        self, input_tensor: object, output_tensor: object, stream: object
    ) -> None:
        self.calls.append(
            (self._pending, input_tensor, output_tensor, stream)
        )


class _FakePromotedBackend(_FakeBackend):
    def __init__(self, rank: int) -> None:
        super().__init__(rank)
        self.promoted_shadow = types.SimpleNamespace(
            count=8,
            validated=True,
            candidate=object(),
        )

    def shadow(
        self,
        signature: backend_module._Signature,
        output_tensor: object,
    ) -> types.SimpleNamespace:
        return self.promoted_shadow


class _FakeIndexerGraphSession:
    def __init__(self) -> None:
        self.capture_calls: list[
            tuple[object, object, int, object]
        ] = []

    def capture(
        self,
        input_tensor: object,
        output_tensor: object,
        q: int,
        stream: object,
    ) -> None:
        self.capture_calls.append(
            (input_tensor, output_tensor, q, stream)
        )


class _FakeGraphBackend(_FakeBackend):
    def __init__(self, rank: int) -> None:
        super().__init__(rank)
        self._indexer_graph_session = _FakeIndexerGraphSession()
        self.prepare_calls: list[object] = []

    def prepare_indexer_graph(
        self, device: object
    ) -> _FakeIndexerGraphSession:
        self.prepare_calls.append(device)
        return self._indexer_graph_session


def _make_pynccl_type() -> type:
    class FakePyNcclCommunicator:
        def __init__(
            self,
            *,
            world_size: int = 4,
            rank: int = 2,
            disabled: bool = False,
            unique_name: str = "tp:0",
            group: object | None = None,
        ) -> None:
            self.world_size = world_size
            self.rank = rank
            self.disabled = disabled
            self.unique_name = unique_name
            self.group = group
            self.original_calls: list[tuple[object, object, object]] = []

        def all_gather(
            self, output_tensor: object, input_tensor: object, stream=None
        ) -> str:
            self.original_calls.append((output_tensor, input_tensor, stream))
            return "reference"

    return FakePyNcclCommunicator


def _fake_modules(pynccl_type: type) -> dict[str, types.ModuleType]:
    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    device_communicators = types.ModuleType(
        "vllm.distributed.device_communicators"
    )
    pynccl = types.ModuleType("vllm.distributed.device_communicators.pynccl")
    pynccl.PyNcclCommunicator = pynccl_type
    return {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.device_communicators": device_communicators,
        "vllm.distributed.device_communicators.pynccl": pynccl,
    }


class SparkTp4AllgatherDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pynccl_type = _make_pynccl_type()
        self.modules = _fake_modules(self.pynccl_type)
        self.torch_module = types.ModuleType("torch")
        self.torch_module.cuda = _FakeCuda()
        self.modules["torch"] = self.torch_module
        self.original = self.pynccl_type.all_gather
        backend_module._installed = False
        backend_module._trace_counts.clear()
        backend_module._reset_capture_decisions_for_tests()
        backend_module._indexer_graph_sessions.clear()
        backend_module._indexer_graph_event_counts.clear()
        spark_collective_audit._reset_for_tests()
        _FakeBackend.created.clear()

    def _install(
        self, mode: str | None, backend_type: type = _FakeBackend
    ) -> None:
        environment = {}
        if mode is not None:
            environment["VLLM_SPARK_TP4_ALLGATHER_MODE"] = mode
        patchers = (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(sys.modules, self.modules),
            patch.object(backend_module, "_Backend", backend_type),
            patch.object(
                backend_module,
                "_is_indexer_communicator",
                side_effect=lambda communicator: getattr(
                    communicator, "unique_name", ""
                )
                == "tp:0",
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        backend_module.install()

    def test_unset_mode_does_not_patch_pynccl(self) -> None:
        self._install(None)
        self.assertIs(self.pynccl_type.all_gather, self.original)
        self.assertFalse(backend_module._installed)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 'shadow', 'custom'"):
            self._install("fast")
        self.assertIs(self.pynccl_type.all_gather, self.original)

    def test_protocol_trace_is_opt_in_and_bounded_to_two_calls(self) -> None:
        fields = {
            "rank": 2,
            "signature": "indexer-k4",
            "input_bytes": 81920,
            "stream_handle": 12345,
            "state": "before-native-call",
        }
        output = StringIO()
        with (
            patch.dict(
                os.environ, {"SPARK_TP4_PROTOCOL_TRACE": "1"}, clear=True
            ),
            redirect_stderr(output),
        ):
            for call in range(1, 4):
                backend_module._protocol_trace(
                    **fields,
                    call=call,
                    result=0 if call == 2 else None,
                )

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("signature=indexer-k4", lines[0])
        self.assertIn("input_bytes=81920", lines[0])
        self.assertIn("stream=12345", lines[0])
        self.assertIn("call=1", lines[0])
        self.assertNotIn("result=", lines[0])
        self.assertIn("call=2", lines[1])
        self.assertIn("result=0", lines[1])

        disabled = StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            redirect_stderr(disabled),
        ):
            backend_module._protocol_trace(**fields, call=1)
        self.assertEqual(disabled.getvalue(), "")

    def test_capture_decision_trace_is_bounded_deduplicated_and_opt_in(
        self,
    ) -> None:
        communicator = self.pynccl_type(rank=2)
        q3_input = _FakeTensor((3, 2, 2048), "torch.int32")
        q3_output = _FakeTensor((12, 2, 2048), "torch.int32")
        q4_input = _FakeTensor((4, 2, 2048), "torch.int32")
        q4_output = _FakeTensor((16, 2, 2048), "torch.int32")
        output = StringIO()

        with (
            patch.dict(
                os.environ,
                {
                    "SPARK_TP4_CAPTURE_DECISION_TRACE": "1",
                    "SPARK_TP4_CAPTURE_DECISION_TRACE_LIMIT": "2",
                },
                clear=True,
            ),
            redirect_stderr(output),
        ):
            for _ in range(3):
                backend_module._record_capture_decision(
                    communicator=communicator,
                    input_tensor=q3_input,
                    output_tensor=q3_output,
                    graph_q=3,
                    current_capturing=False,
                    explicit_capturing=False,
                    route="custom",
                    reason="eager_native",
                )
            backend_module._record_capture_decision(
                communicator=communicator,
                input_tensor=q3_input,
                output_tensor=q3_output,
                graph_q=3,
                current_capturing=True,
                explicit_capturing=False,
                route="stock",
                reason="graph_capture_unsupported",
            )
            backend_module._record_capture_decision(
                communicator=communicator,
                input_tensor=q4_input,
                output_tensor=q4_output,
                graph_q=4,
                current_capturing=False,
                explicit_capturing=True,
                route="stock",
                reason="graph_capture_unsupported",
            )

        snapshot = backend_module.capture_decision_snapshot()
        self.assertEqual(len(snapshot["records"]), 2)
        self.assertEqual(snapshot["dropped"], 1)
        self.assertEqual(
            [record["count"] for record in snapshot["records"]],
            [3, 1],
        )
        self.assertEqual(
            [
                (
                    record["capture_state"],
                    record["route"],
                    record["reason"],
                )
                for record in snapshot["records"]
            ],
            [
                ("neither", "custom", "eager_native"),
                ("current", "stock", "graph_capture_unsupported"),
            ],
        )
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(
            all(
                line.startswith("TP4_AG_CAPTURE_DECISION ")
                for line in lines
            )
        )
        first = json.loads(lines[0].split(" ", 1)[1])
        self.assertFalse(first["current_capture"])
        self.assertFalse(first["explicit_capture"])
        self.assertEqual(first["capture_state"], "neither")
        self.assertEqual(first["route"], "custom")

        backend_module._reset_capture_decisions_for_tests()
        disabled = StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            redirect_stderr(disabled),
        ):
            backend_module._record_capture_decision(
                communicator=communicator,
                input_tensor=q3_input,
                output_tensor=q3_output,
                graph_q=3,
                current_capturing=True,
                explicit_capturing=True,
                route="stock",
                reason="disabled",
            )
        self.assertEqual(disabled.getvalue(), "")
        self.assertEqual(
            backend_module.capture_decision_snapshot(),
            {"records": [], "dropped": 0},
        )

    def test_dispatch_census_separates_eager_custom_and_capture_stock(
        self,
    ) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_CAPTURE_DECISION_TRACE"] = "1"
        communicator = self.pynccl_type(rank=1)
        input_tensor = _FakeTensor((3, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((12, 2, 2048), "torch.int32")
        stream = types.SimpleNamespace(
            cuda_stream=101,
            capturing=False,
        )

        with redirect_stderr(StringIO()):
            eager_result = communicator.all_gather(
                output_tensor, input_tensor, stream
            )
            self.torch_module.cuda.capturing = True
            capture_result = communicator.all_gather(
                output_tensor, input_tensor, stream
            )

        self.assertIsNone(eager_result)
        self.assertEqual(capture_result, "reference")
        self.assertEqual(len(_FakeBackend.created[0].calls), 1)
        self.assertEqual(len(communicator.original_calls), 1)
        records = backend_module.capture_decision_snapshot()["records"]
        self.assertEqual(
            [
                (
                    record["capture_state"],
                    record["route"],
                    record["reason"],
                )
                for record in records
            ],
            [
                ("neither", "custom", "eager_native"),
                (
                    "current+explicit",
                    "stock",
                    "graph_capture_unsupported",
                ),
            ],
        )

    def test_custom_routes_all_exact_k0_and_mtp_signatures(self) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_ALLGATHER_ENABLE_CKV"] = "1"
        communicator = self.pynccl_type(rank=3)
        stream = object()
        cases = (
            (
                (1, 2, 2048),
                "torch.int32",
                (4, 2, 2048),
                (16384, 0, "indexer"),
            ),
            (
                (1, 38720),
                "torch.bfloat16",
                (4, 38720),
                (77440, 1, "vocab"),
            ),
            (
                (753664,),
                "torch.uint8",
                (3014656,),
                (753664, 2, "ckv"),
            ),
            (
                (2, 2, 2048),
                "torch.int32",
                (8, 2, 2048),
                (32768, 3, "indexer-k1"),
            ),
            (
                (3, 2, 2048),
                "torch.int32",
                (12, 2, 2048),
                (49152, 4, "indexer-k2"),
            ),
            (
                (4, 2, 2048),
                "torch.int32",
                (16, 2, 2048),
                (65536, 5, "indexer-k3"),
            ),
            (
                (5, 2, 2048),
                "torch.int32",
                (20, 2, 2048),
                (81920, 6, "indexer-k4"),
            ),
            (
                (23552,),
                "torch.uint8",
                (94208,),
                (23552, 7, "ckv-prefill"),
            ),
        )
        for input_shape, dtype, output_shape, signature in cases:
            input_tensor = _FakeTensor(input_shape, dtype)
            output_tensor = _FakeTensor(output_shape, dtype)
            result = communicator.all_gather(
                output_tensor, input_tensor, stream
            )
            self.assertIsNone(result)
            self.assertEqual(
                _FakeBackend.created[0].calls[-1],
                (signature, input_tensor, output_tensor, stream),
            )
        self.assertEqual(communicator.original_calls, [])
        self.assertEqual(_FakeBackend.created[0].rank, 3)

    def test_ckv_signatures_default_to_original_collective(self) -> None:
        self._install("custom")
        communicator = self.pynccl_type(rank=3)
        cases = (
            (
                _FakeTensor((753664,), "torch.uint8"),
                _FakeTensor((3014656,), "torch.uint8"),
            ),
            (
                _FakeTensor((23552,), "torch.uint8"),
                _FakeTensor((94208,), "torch.uint8"),
            ),
        )

        for input_tensor, output_tensor in cases:
            result = communicator.all_gather(
                output_tensor, input_tensor, "stream"
            )
            self.assertEqual(result, "reference")

        self.assertEqual(len(communicator.original_calls), 2)
        self.assertEqual(_FakeBackend.created, [])

    def test_capture_delegates_without_touching_native_or_shadow_state(
        self,
    ) -> None:
        self._install("custom")
        self.torch_module.cuda.capturing = True
        communicator = self.pynccl_type(rank=1)
        backend = _FakeBackend(communicator.rank)
        communicator._spark_tp4_allgather_native = backend
        _FakeBackend.created.clear()
        inputs = [
            _FakeTensor((1, 2, 2048), "torch.int32"),
            _FakeTensor((5, 2, 2048), "torch.int32"),
        ]
        outputs = [
            _FakeTensor((4, 2, 2048), "torch.int32"),
            _FakeTensor((20, 2, 2048), "torch.int32"),
        ]
        stream = object()

        results = []
        for mode, input_tensor, output_tensor in zip(
            ("custom", "shadow"), inputs, outputs
        ):
            os.environ["VLLM_SPARK_TP4_ALLGATHER_MODE"] = mode
            results.append(
                communicator.all_gather(output_tensor, input_tensor, stream)
            )

        self.assertEqual(results, ["reference", "reference"])
        self.assertEqual(
            communicator.original_calls,
            [
                (outputs[0], inputs[0], stream),
                (outputs[1], inputs[1], stream),
            ],
        )
        self.assertIs(communicator._spark_tp4_allgather_native, backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(_FakeBackend.created, [])

    def test_explicit_capture_stream_delegates_when_current_stream_is_not_capturing(
        self,
    ) -> None:
        self._install("custom")
        self.torch_module.cuda.capturing = False
        communicator = self.pynccl_type(rank=1)
        input_tensor = _FakeTensor((5, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((20, 2, 2048), "torch.int32")
        capture_stream = types.SimpleNamespace(
            cuda_stream=303, capturing=True
        )

        result = communicator.all_gather(
            output_tensor, input_tensor, capture_stream
        )

        self.assertEqual(result, "reference")
        self.assertEqual(
            communicator.original_calls,
            [(output_tensor, input_tensor, capture_stream)],
        )
        self.assertEqual(_FakeBackend.created, [])

    def test_capture_detection_uses_current_or_explicit_stream(self) -> None:
        cases = (
            (False, False, False),
            (False, True, True),
            (True, False, True),
            (True, True, True),
        )

        for current_capturing, explicit_capturing, expected in cases:
            with self.subTest(
                current_capturing=current_capturing,
                explicit_capturing=explicit_capturing,
            ):
                self.torch_module.cuda.capturing = current_capturing
                explicit_stream = types.SimpleNamespace(
                    cuda_stream=303,
                    capturing=explicit_capturing,
                )

                self.assertEqual(
                    backend_module._is_stream_capturing(
                        self.torch_module,
                        explicit_stream,
                    ),
                    expected,
                )
                self.assertEqual(
                    self.torch_module.cuda.capturing,
                    current_capturing,
                )

    def test_capture_state_census_distinguishes_all_stream_states(
        self,
    ) -> None:
        cases = (
            (False, 101, 303, False, (False, False)),
            (False, 101, 303, True, (False, True)),
            (True, 101, 303, False, (True, False)),
            (True, 101, 101, True, (True, True)),
        )

        for (
            current_capturing,
            current_handle,
            explicit_handle,
            explicit_capturing,
            expected,
        ) in cases:
            with self.subTest(expected=expected):
                self.torch_module.cuda.capturing = current_capturing
                self.torch_module.cuda.current_cuda_stream = current_handle
                explicit_stream = types.SimpleNamespace(
                    cuda_stream=explicit_handle,
                    capturing=explicit_capturing,
                )

                self.assertEqual(
                    backend_module._stream_capture_states(
                        self.torch_module,
                        explicit_stream,
                    ),
                    expected,
                )

    def test_current_capture_delegates_when_explicit_stream_is_not_capturing(
        self,
    ) -> None:
        self._install("custom")
        self.torch_module.cuda.capturing = True
        communicator = self.pynccl_type(rank=1)
        input_tensor = _FakeTensor((3, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((12, 2, 2048), "torch.int32")
        noncapturing_stream = types.SimpleNamespace(
            cuda_stream=404,
            capturing=False,
        )

        result = communicator.all_gather(
            output_tensor,
            input_tensor,
            noncapturing_stream,
        )

        self.assertEqual(result, "reference")
        self.assertEqual(
            communicator.original_calls,
            [(output_tensor, input_tensor, noncapturing_stream)],
        )
        self.assertEqual(_FakeBackend.created, [])

    def test_capture_original_is_visible_to_stock_audit(self) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        self.torch_module.cuda.capturing = True
        communicator = self.pynccl_type(rank=1)
        input_tensor = _FakeTensor((1, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((4, 2, 2048), "torch.int32")

        communicator.all_gather(output_tensor, input_tensor, object())

        self.assertEqual(
            spark_collective_audit.stock_collective_snapshot()["capture"],
            {"dcp_owner_topk_all_gather:graph_capture_unsupported": 1},
        )
        self.assertEqual(
            spark_collective_audit.stock_collective_snapshot()["signatures"][
                "capture"
            ],
            [
                {
                    "family": "dcp_owner_topk_all_gather",
                    "reason": "graph_capture_unsupported",
                    "count": 1,
                    "shape": [1, 2, 2048],
                    "dtype": "torch.int32",
                    "is_cuda": True,
                    "contiguous": True,
                    "world_size": 4,
                    "unique_name": "tp:0",
                }
            ],
        )

    def test_changed_eager_stream_falls_back_before_native_call(self) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        communicator = self.pynccl_type(rank=1)
        input_tensor = _FakeTensor((5, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((20, 2, 2048), "torch.int32")
        first_stream = types.SimpleNamespace(cuda_stream=101)
        second_stream = types.SimpleNamespace(cuda_stream=202)

        first_result = communicator.all_gather(
            output_tensor, input_tensor, first_stream
        )
        second_result = communicator.all_gather(
            output_tensor, input_tensor, second_stream
        )

        self.assertIsNone(first_result)
        self.assertEqual(second_result, "reference")
        self.assertEqual(len(_FakeBackend.created[0].calls), 1)
        self.assertEqual(
            communicator.original_calls,
            [(output_tensor, input_tensor, second_stream)],
        )
        self.assertEqual(
            spark_collective_audit.stock_collective_snapshot()["eager"],
            {"dcp_owner_topk_all_gather:caller_stream_changed": 1},
        )

    def test_near_misses_delegate_to_nccl(self) -> None:
        self._install("custom")
        cases = (
            (
                self.pynccl_type(world_size=2),
                _FakeTensor((1, 2, 2048), "torch.int32"),
                _FakeTensor((2, 2, 2048), "torch.int32"),
            ),
            (
                self.pynccl_type(disabled=True),
                _FakeTensor((1, 2, 2048), "torch.int32"),
                _FakeTensor((4, 2, 2048), "torch.int32"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor((2, 1, 2048), "torch.int32"),
                _FakeTensor((8, 1, 2048), "torch.int32"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor((6, 2, 2048), "torch.int32"),
                _FakeTensor((24, 2, 2048), "torch.int32"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor((23551,), "torch.uint8"),
                _FakeTensor((94204,), "torch.uint8"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor((23552,), "torch.int32"),
                _FakeTensor((94208,), "torch.int32"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor((28,), "torch.uint8"),
                _FakeTensor((112,), "torch.uint8"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor((31,), "torch.uint8"),
                _FakeTensor((124,), "torch.uint8"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor(
                    (1, 2, 2048), "torch.int32", contiguous=False
                ),
                _FakeTensor((4, 2, 2048), "torch.int32"),
            ),
            (
                self.pynccl_type(),
                _FakeTensor((1, 2, 2048), "torch.int32"),
                _FakeTensor((3, 2, 2048), "torch.int32"),
            ),
        )
        for communicator, input_tensor, output_tensor in cases:
            with self.subTest(
                input_shape=input_tensor.shape,
                output_shape=output_tensor.shape,
            ):
                result = communicator.all_gather(
                    output_tensor, input_tensor, "stream"
                )
                self.assertEqual(result, "reference")
                self.assertEqual(len(communicator.original_calls), 1)
        self.assertEqual(_FakeBackend.created, [])

    @patch.object(
        backend_module,
        "_is_indexer_communicator",
        lambda communicator: getattr(communicator, "unique_name", "")
        == "tp:0",
    )
    def test_indexer_graph_formula_admits_every_q1_q40(self) -> None:
        communicator = self.pynccl_type()
        for q in range(1, 41):
            input_tensor = _FakeTensor(
                (q, 2, 2048), "torch.int32"
            )
            output_tensor = _FakeTensor(
                (4 * q, 2, 2048), "torch.int32"
            )
            self.assertEqual(
                backend_module._indexer_graph_q(
                    communicator,
                    input_tensor,
                    output_tensor,
                    "custom",
                ),
                q,
            )
        for q in (0, 41):
            input_tensor = _FakeTensor(
                (q, 2, 2048), "torch.int32"
            )
            output_tensor = _FakeTensor(
                (4 * q, 2, 2048), "torch.int32"
            )
            self.assertIsNone(
                backend_module._indexer_graph_q(
                    communicator,
                    input_tensor,
                    output_tensor,
                    "custom",
                )
            )
        self.assertIsNone(
            backend_module._indexer_graph_q(
                self.pynccl_type(unique_name="dcp:0"),
                _FakeTensor((3, 2, 2048), "torch.int32"),
                _FakeTensor((12, 2, 2048), "torch.int32"),
                "custom",
            )
        )

    def test_indexer_identity_accepts_only_configured_group_owner(
        self,
    ) -> None:
        cpu_group = object()
        correct = self.pynccl_type(
            unique_name="",
            group=cpu_group,
        )
        wrong_group = self.pynccl_type(
            unique_name="",
            group=object(),
        )
        unnamed_nonowner = self.pynccl_type(
            unique_name="",
            group=cpu_group,
        )
        indexer_group = types.SimpleNamespace(
            unique_name="dcp:0",
            world_size=4,
            rank_in_group=correct.rank,
            cpu_group=cpu_group,
            device_communicator=types.SimpleNamespace(
                pynccl_comm=correct,
            ),
        )
        parallel_state = types.ModuleType(
            "vllm.distributed.parallel_state"
        )

        def get_indexer_dcp_group(expected_world_size: int) -> object:
            self.assertEqual(expected_world_size, 4)
            return indexer_group

        parallel_state.get_indexer_dcp_group = get_indexer_dcp_group
        modules = dict(self.modules)
        modules["vllm.distributed.parallel_state"] = parallel_state
        input_tensor = _FakeTensor((3, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((12, 2, 2048), "torch.int32")

        with patch.dict(sys.modules, modules):
            self.assertTrue(
                backend_module._is_indexer_communicator(correct)
            )
            self.assertIsNone(
                backend_module._signature(
                    correct,
                    input_tensor,
                    output_tensor,
                    "custom",
                )
            )
            self.assertEqual(
                backend_module._indexer_graph_q(
                    correct,
                    input_tensor,
                    output_tensor,
                    "custom",
                ),
                3,
            )
            for reason, rejected in (
                ("wrong-group", wrong_group),
                ("unnamed-nonowner", unnamed_nonowner),
            ):
                with self.subTest(reason=reason):
                    self.assertFalse(
                        backend_module._is_indexer_communicator(
                            rejected
                        )
                    )
                    self.assertIsNone(
                        backend_module._signature(
                            rejected,
                            input_tensor,
                            output_tensor,
                            "custom",
                        )
                    )
                    self.assertIsNone(
                        backend_module._indexer_graph_q(
                            rejected,
                            input_tensor,
                            output_tensor,
                            "custom",
                        )
                    )

            indexer_group.unique_name = ""
            self.assertFalse(
                backend_module._is_indexer_communicator(correct)
            )
            indexer_group.unique_name = "dcp:0"
            indexer_group.world_size = 3
            self.assertFalse(
                backend_module._is_indexer_communicator(correct)
            )
            indexer_group.world_size = 4
            indexer_group.rank_in_group = correct.rank + 1
            self.assertFalse(
                backend_module._is_indexer_communicator(correct)
            )

    def test_indexer_identity_fails_closed_without_group_state(
        self,
    ) -> None:
        communicator = self.pynccl_type(unique_name="dcp:0")
        parallel_state = types.ModuleType(
            "vllm.distributed.parallel_state"
        )

        def unavailable_group(expected_world_size: int) -> object:
            del expected_world_size
            raise RuntimeError("indexer group unavailable")

        parallel_state.get_indexer_dcp_group = unavailable_group
        modules = dict(self.modules)
        modules["vllm.distributed.parallel_state"] = parallel_state
        with patch.dict(sys.modules, modules):
            self.assertFalse(
                backend_module._is_indexer_communicator(communicator)
            )

    def test_indexer_graph_uses_selected_dcp_communicator_identity(self) -> None:
        cpu_group = object()
        selected = self.pynccl_type(unique_name="", group=cpu_group)
        unrelated = self.pynccl_type(unique_name="", group=cpu_group)
        indexer_group = types.SimpleNamespace(
            unique_name="dcp:0",
            world_size=4,
            rank_in_group=selected.rank,
            cpu_group=cpu_group,
            device_communicator=types.SimpleNamespace(pynccl_comm=selected),
        )
        parallel_state = types.ModuleType("vllm.distributed.parallel_state")
        parallel_state.get_indexer_dcp_group = (
            lambda expected_world_size: indexer_group
        )
        input_tensor = _FakeTensor((3, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((12, 2, 2048), "torch.int32")

        with patch.dict(
            sys.modules,
            {"vllm.distributed.parallel_state": parallel_state},
        ):
            self.assertEqual(
                backend_module._indexer_graph_q(
                    selected,
                    input_tensor,
                    output_tensor,
                    "custom",
                ),
                3,
            )
            self.assertIsNone(
                backend_module._indexer_graph_q(
                    unrelated,
                    input_tensor,
                    output_tensor,
                    "custom",
                )
            )

    def test_indexer_graph_uses_one_noncolliding_port_pair(self) -> None:
        with patch.dict(
            os.environ,
            {"SPARK_TP4_ALLGATHER_BASE_PORT": "9490"},
            clear=True,
        ):
            self.assertEqual(
                backend_module._indexer_graph_control_ports(),
                (9462, 9463),
            )
        with (
            patch.dict(
                os.environ,
                {
                    "VLLM_SPARK_TP4_ALLGATHER_MODE": "custom",
                    "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": "1",
                    "SPARK_TP4_ALLGATHER_BASE_PORT": "9490",
                    "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT0": "9490",
                    "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT1": "9491",
                },
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "collide"),
        ):
            backend_module._indexer_graph_control_ports()
        with (
            patch.dict(
                os.environ,
                {
                    "VLLM_SPARK_TP4_MODE": "custom",
                    "VLLM_SPARK_TP4_ALLGATHER_MODE": "custom",
                    "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": "1",
                    "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT0": "11002",
                    "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT1": "11003",
                },
                clear=True,
            ),
            self.assertRaisesRegex(
                ValueError, "eager_allreduce:payload=24576"
            ),
        ):
            backend_module._indexer_graph_control_ports()

    def test_indexer_graph_prepares_before_capture_and_covers_new_q(
        self,
    ) -> None:
        self._install("custom", _FakeGraphBackend)
        os.environ["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] = "1"
        communicator = self.pynccl_type(rank=1)
        stream = types.SimpleNamespace(cuda_stream=101)

        # Q23 has no eager exact-signature session. The eager warmup remains
        # stock but prepares the one family-level graph session.
        q23_input = _FakeTensor((23, 2, 2048), "torch.int32")
        q23_output = _FakeTensor((92, 2, 2048), "torch.int32")
        self.assertEqual(
            communicator.all_gather(q23_output, q23_input, stream),
            "reference",
        )
        graph_backend = _FakeBackend.created[0]
        self.assertEqual(graph_backend.prepare_calls, ["cuda:0"])

        self.torch_module.cuda.capturing = True
        for q in (1, 23, 40):
            input_tensor = _FakeTensor(
                (q, 2, 2048), "torch.int32"
            )
            output_tensor = _FakeTensor(
                (4 * q, 2, 2048), "torch.int32"
            )
            self.assertIsNone(
                communicator.all_gather(
                    output_tensor, input_tensor, stream
                )
            )

        self.assertEqual(
            [
                call[2]
                for call in graph_backend._indexer_graph_session.capture_calls
            ],
            [1, 23, 40],
        )
        self.assertEqual(
            communicator.original_calls,
            [(q23_output, q23_input, stream)],
        )

    def test_indexer_graph_rejects_noncurrent_explicit_capture_stream(
        self,
    ) -> None:
        self._install("custom", _FakeGraphBackend)
        os.environ["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] = "1"
        communicator = self.pynccl_type(rank=1)
        graph_backend = _FakeGraphBackend(communicator.rank)
        communicator._spark_tp4_allgather_native = graph_backend
        self.torch_module.cuda.capturing = True
        stale_stream = types.SimpleNamespace(cuda_stream=202)
        input_tensor = _FakeTensor((40, 2, 2048), "torch.int32")
        output_tensor = _FakeTensor((160, 2, 2048), "torch.int32")

        with (
            patch.object(
                backend_module,
                "_abort_after_native_failure",
                side_effect=RuntimeError("abort"),
            ),
            self.assertRaisesRegex(RuntimeError, "abort"),
        ):
            communicator.all_gather(
                output_tensor, input_tensor, stale_stream
            )
        self.assertEqual(
            graph_backend._indexer_graph_session.capture_calls, []
        )
        self.assertEqual(communicator.original_calls, [])

    def test_shape_trace_records_mtp_route_and_ineligible_warmup(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "shapes.jsonl"
            self._install("custom")
            os.environ["SPARK_TP4_TRACE_ALLGATHER_SHAPES"] = "1"
            os.environ["SPARK_TP4_ALLGATHER_TRACE_PATH"] = str(output_path)
            communicator = self.pynccl_type()
            input_tensor = _FakeTensor((5, 2, 2048), "torch.int32")
            output_tensor = _FakeTensor((20, 2, 2048), "torch.int32")

            result = communicator.all_gather(
                output_tensor, input_tensor, "stream"
            )
            warmup_input = _FakeTensor((28,), "torch.uint8")
            warmup_output = _FakeTensor((112,), "torch.uint8")
            warmup_result = communicator.all_gather(
                warmup_output, warmup_input, "stream"
            )

            self.assertIsNone(result)
            self.assertEqual(warmup_result, "reference")
            records = [
                json.loads(line)
                for line in output_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [record["input_shape"] for record in records],
                [[5, 2, 2048], [28]],
            )
            self.assertEqual(
                [record["output_shape"] for record in records],
                [[20, 2, 2048], [112]],
            )
            self.assertEqual(
                [record["count"] for record in records], [1, 1]
            )

    def test_validated_shadow_can_promote_without_restart(self) -> None:
        self._install("shadow", _FakePromotedBackend)
        os.environ["SPARK_TP4_ALLGATHER_SHADOW_PROMOTE"] = "1"
        os.environ["SPARK_TP4_ALLGATHER_ENABLE_CKV"] = "1"
        communicator = self.pynccl_type(rank=1)
        stream = object()
        input_tensor = _FakeTensor((753664,), "torch.uint8")
        output_tensor = _FakeTensor((3014656,), "torch.uint8")

        result = communicator.all_gather(
            output_tensor, input_tensor, stream
        )

        self.assertIsNone(result)
        promoted = _FakeBackend.created[0]
        self.assertEqual(
            promoted.calls,
            [
                (
                    (753664, 2, "ckv"),
                    input_tensor,
                    output_tensor,
                    stream,
                )
            ],
        )
        self.assertEqual(communicator.original_calls, [])

    def test_native_sessions_and_shadows_are_isolated_by_signature(
        self,
    ) -> None:
        class FakeNativeSession:
            created: list["FakeNativeSession"] = []

            def __init__(
                self, rank: int, input_bytes: int, port_slot: int
            ) -> None:
                self.config = (rank, input_bytes, port_slot)
                self.created.append(self)

        fake_torch = types.ModuleType("torch")
        fake_torch.empty_like = lambda output: ("candidate", output)
        signature_a = (23552, 7, "ckv-mtp")
        signature_b = (23552, 8, "same-payload-distinct-signature")
        output_a = object()
        output_b = object()

        with (
            patch.object(
                backend_module,
                "_NativeAllgatherSession",
                FakeNativeSession,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            native = backend_module._Backend(rank=2)
            session_a = native.session(signature_a)
            self.assertIs(native.session(signature_a), session_a)
            session_b = native.session(signature_b)
            shadow_a = native.shadow(signature_a, output_a)
            self.assertIs(native.shadow(signature_a, object()), shadow_a)
            shadow_b = native.shadow(signature_b, output_b)

        self.assertIsNot(session_a, session_b)
        self.assertEqual(
            [session.config for session in FakeNativeSession.created],
            [(2, 23552, 7), (2, 23552, 8)],
        )
        self.assertIsNot(shadow_a, shadow_b)
        self.assertEqual(shadow_a.candidate, ("candidate", output_a))
        self.assertEqual(shadow_b.candidate, ("candidate", output_b))


if __name__ == "__main__":
    unittest.main()
