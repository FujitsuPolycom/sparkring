"""GPU-free tests for the Spark TP4 DCP query and combine adapter."""

from __future__ import annotations

import os
import sys
import types
import unittest
from typing import ClassVar, Self
from unittest.mock import patch

import spark_collective_audit
import spark_tp4_dcp_backend as backend_module


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def __iadd__(self, other: _FakeScalar) -> Self:
        self.value += other.value
        return self

    def item(self) -> float:
        return self.value

    def copy_(self, other: _FakeScalar) -> _FakeScalar:
        self.value = other.value
        return self

    def add_(self, value: int) -> _FakeScalar:
        self.value += value
        return self


class _FakeSlots:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        count = 1
        for dimension in shape:
            count *= dimension
        self._values = [_FakeScalar(0) for _ in range(count)]

    def __getitem__(self, index: int | tuple[int, ...]) -> _FakeScalar:
        indices = (index,) if isinstance(index, int) else index
        if len(indices) != len(self.shape):
            raise IndexError("fake slot index rank mismatch")
        flat = 0
        for value, dimension in zip(indices, self.shape, strict=True):
            if not 0 <= value < dimension:
                raise IndexError("fake slot index out of range")
            flat = flat * dimension + value
        return self._values[flat]


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
        strides: tuple[int, ...] | None = None,
        payload: float = 0,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = "cuda:0" if is_cuda else "cpu"
        self._contiguous = contiguous
        self._strides = strides or self._contiguous_strides(shape)
        if not contiguous and strides is None and self._strides:
            self._strides = (self._strides[0] + 1, *self._strides[1:])
        self.contiguous_calls = 0
        self.payload = payload
        self.pointer = self._next_pointer
        _FakeTensor._next_pointer += 0x1000

    def is_contiguous(self) -> bool:
        return self._contiguous

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        strides = []
        next_stride = 1
        for dimension in reversed(shape):
            strides.append(next_stride)
            next_stride *= dimension
        return tuple(reversed(strides))

    def stride(self) -> tuple[int, ...]:
        return self._strides

    def data_ptr(self) -> int:
        return self.pointer

    def contiguous(self) -> _FakeTensor:
        self.contiguous_calls += 1
        if self._contiguous:
            return self
        return _FakeTensor(
            self.shape,
            self.dtype,
            is_cuda=self.is_cuda,
            contiguous=True,
            payload=self.payload,
        )

    def view(self, dtype: object) -> _FakeByteView:
        del dtype
        return _FakeByteView(self)


def _live_combine_output(
    q: int,
    dtype: str = "torch.bfloat16",
    *,
    head_dimension: int = 256,
    is_cuda: bool = True,
    payload: float = 0,
) -> _FakeTensor:
    return _FakeTensor(
        (q, 64, head_dimension),
        dtype,
        is_cuda=is_cuda,
        contiguous=q == 1,
        strides=(head_dimension, q * head_dimension, 1),
        payload=payload,
    )


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

    def synchronize(self) -> None:
        return None


def _fake_torch_module() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.uint8 = "torch.uint8"
    module.int64 = "torch.int64"
    module.float32 = "torch.float32"
    module.cuda = _FakeCuda()
    module.allocations = []

    def empty(shape: tuple[int, ...], *, dtype: str, device: object) -> _FakeTensor:
        if device != "cuda:0":
            raise AssertionError(f"unexpected device: {device}")
        tensor = _FakeTensor(shape, dtype)
        module.allocations.append(tensor)
        return tensor

    def empty_like(tensor: _FakeTensor) -> _FakeTensor:
        result = _FakeTensor(tensor.shape, tensor.dtype)
        module.allocations.append(result)
        return result

    def count_nonzero(difference: _FakeDifference) -> _FakeScalar:
        return _FakeScalar(difference.mismatches)

    def zeros(shape: tuple[int, ...], *, dtype: str, device: object) -> _FakeSlots:
        del dtype
        if device != "cuda:0":
            raise AssertionError(f"unexpected device: {device}")
        result = _FakeSlots(shape)
        module.allocations.append(result)
        return result

    def maximum(left: _FakeScalar, right: _FakeScalar) -> _FakeScalar:
        return _FakeScalar(max(left.value, right.value))

    module.empty = empty
    module.empty_like = empty_like
    module.count_nonzero = count_nonzero
    module.zeros = zeros
    module.maximum = maximum
    return module


def _fake_numeric_sample(
    candidate: _FakeTensor,
    reference: _FakeTensor,
    rtol: float,
    atol: float,
) -> tuple[_FakeScalar, _FakeScalar, _FakeScalar, _FakeScalar]:
    difference = abs(float(candidate.payload) - float(reference.payload))
    allowed = atol + rtol * abs(float(reference.payload))
    relative = difference / max(abs(float(reference.payload)), 1.0e-12)
    return (
        _FakeScalar(int(difference > allowed)),
        _FakeScalar(0),
        _FakeScalar(difference),
        _FakeScalar(relative),
    )


def _make_group_type() -> type:
    class FakeGroupCoordinator:
        def __init__(
            self,
            *,
            unique_name: str = "dcp:0",
            world_size: int = 4,
            rank_in_group: int = 2,
        ) -> None:
            self.unique_name = unique_name
            self.world_size = world_size
            self.rank_in_group = rank_in_group
            self.fail_original = False
            self.fail_combine_original = False
            self.original_calls: list[tuple[object, int]] = []
            self.combine_original_calls: list[tuple[object, ...]] = []

        def _all_gather_out_place(
            self, input_tensor: _FakeTensor, dim: int
        ) -> _FakeTensor:
            self.original_calls.append((input_tensor, dim))
            if self.fail_original:
                raise RuntimeError("reference failed")
            q = int(input_tensor.shape[0])
            return _FakeTensor(
                (q, 64, 576),
                input_tensor.dtype,
                payload=input_tensor.payload,
            )

    return FakeGroupCoordinator


def _stock_combine(
    cp_attn_out: _FakeTensor,
    cp_attn_lse: _FakeTensor,
    cp_group: object,
    ctx: object = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> object:
    cp_group.combine_original_calls.append(
        (
            cp_attn_out,
            cp_attn_lse,
            ctx,
            return_lse,
            is_lse_base_on_e,
        )
    )
    if cp_group.fail_combine_original:
        raise RuntimeError("combine reference failed")
    q = int(cp_attn_out.shape[0])
    output = _FakeTensor(
        (q, 16, cp_attn_out.shape[2]),
        cp_attn_out.dtype,
        payload=cp_attn_out.payload,
    )
    if not return_lse:
        return output
    lse = _FakeTensor((q, 16), cp_attn_lse.dtype, payload=cp_attn_lse.payload)
    return output, lse


def _fake_modules(
    group_type: type, torch_module: types.ModuleType
) -> dict[str, types.ModuleType]:
    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.GroupCoordinator = group_type
    v1 = types.ModuleType("vllm.v1")
    attention = types.ModuleType("vllm.v1.attention")
    ops = types.ModuleType("vllm.v1.attention.ops")
    common = types.ModuleType("vllm.v1.attention.ops.common")
    common.cp_lse_ag_out_rs = _stock_combine
    return {
        "torch": torch_module,
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": parallel_state,
        "vllm.v1": v1,
        "vllm.v1.attention": attention,
        "vllm.v1.attention.ops": ops,
        "vllm.v1.attention.ops.common": common,
    }


class _FakeNativeSession:
    created: ClassVar[list[_FakeNativeSession]] = []
    fail_create = False
    fail_call = False
    fail_combine_call = False
    mismatch = False
    combine_output_delta = 0.0
    combine_lse_delta = 0.0

    def __init__(
        self,
        rank: int,
        *,
        graph_only: bool = False,
        control_ports: tuple[int, int] | None = None,
        graph_cpu_affinity: tuple[int, int] | None = None,
    ) -> None:
        if self.fail_create:
            raise RuntimeError("create failed")
        self.rank = rank
        self.graph_only = graph_only
        self.control_ports = control_ports
        self.graph_cpu_affinity = graph_cpu_affinity
        self.query_calls: list[tuple[object, object, int, object]] = []
        self.combine_calls: list[tuple[object, ...]] = []
        self.graph_query_calls: list[tuple[object, object, int, object]] = []
        self.graph_combine_calls: list[tuple[object, ...]] = []
        self.operation_order: list[tuple[str, int]] = []
        self.created.append(self)

    def query_all_gather(
        self,
        input_tensor: _FakeTensor,
        output_tensor: _FakeTensor,
        q: int,
        stream: object,
    ) -> None:
        if self.fail_call:
            raise RuntimeError("native call failed")
        output_tensor.payload = (
            input_tensor.payload + 1 if self.mismatch else input_tensor.payload
        )
        self.query_calls.append((input_tensor, output_tensor, q, stream))
        self.operation_order.append(("query", q))

    def combine(
        self,
        output_tensor: _FakeTensor,
        lse_tensor: _FakeTensor,
        reduced_output: _FakeTensor,
        reduced_lse: _FakeTensor,
        signature: tuple[int, int, int, int],
        stream: object,
    ) -> None:
        if self.fail_combine_call:
            raise RuntimeError("native combine failed")
        reduced_output.payload = output_tensor.payload + self.combine_output_delta
        reduced_lse.payload = lse_tensor.payload + self.combine_lse_delta
        q = signature[0]
        self.combine_calls.append(
            (
                output_tensor,
                lse_tensor,
                reduced_output,
                reduced_lse,
                signature,
                stream,
            )
        )
        self.operation_order.append(("combine", q))

    def capture_query_all_gather(
        self,
        input_tensor: _FakeTensor,
        output_tensor: _FakeTensor,
        q: int,
        stream: object,
    ) -> None:
        if not self.graph_only:
            raise RuntimeError("not a graph session")
        output_tensor.payload = (
            input_tensor.payload + 1 if self.mismatch else input_tensor.payload
        )
        self.graph_query_calls.append((input_tensor, output_tensor, q, stream))
        self.operation_order.append(("graph_query", q))

    def capture_combine(
        self,
        output_tensor: _FakeTensor,
        lse_tensor: _FakeTensor,
        reduced_output: _FakeTensor,
        reduced_lse: _FakeTensor,
        signature: tuple[int, int, int, int],
        stream: object,
    ) -> None:
        if not self.graph_only:
            raise RuntimeError("not a graph session")
        reduced_output.payload = output_tensor.payload + self.combine_output_delta
        reduced_lse.payload = lse_tensor.payload + self.combine_lse_delta
        self.graph_combine_calls.append(
            (
                output_tensor,
                lse_tensor,
                reduced_output,
                reduced_lse,
                signature,
                stream,
            )
        )
        self.operation_order.append(("graph_combine", signature[0]))

    def graph_status(self) -> dict[str, object]:
        query = len(self.graph_query_calls)
        combine = len(self.graph_combine_calls)
        total = query + combine
        return {
            "captured_nodes": total,
            "captured_query_nodes": query,
            "captured_combine_nodes": combine,
            "published_sequence": total,
            "consumed_sequence": total,
            "completed_sequence": total,
            "overflow_sequence": 0,
            "capture_configured": total > 0,
            "polling_enabled": total > 0,
            "host_native_atomics": True,
            "submit_affinity_verified": True,
            "progress_affinity_verified": True,
            "submit_cpu": 10,
            "progress_cpu": 13,
            "replay_advanced": total > 0,
            "replay_caught_up": total > 0,
            "fatal": False,
        }


class _AbortCalled(RuntimeError):
    pass


class SparkTp4DcpDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.group_type = _make_group_type()
        self.torch_module = _fake_torch_module()
        self.modules = _fake_modules(self.group_type, self.torch_module)
        self.original = self.group_type._all_gather_out_place
        self.flash_module = self.modules[backend_module._COMBINE_TARGET]
        self.original_combine = self.flash_module.cp_lse_ag_out_rs
        backend_module._installed = False
        backend_module._dcp_backends.clear()
        backend_module._dcp_graph_sessions.clear()
        backend_module._graph_event_counts.clear()
        spark_collective_audit._reset_for_tests()
        self._remove_combine_finders()
        self.addCleanup(self._remove_combine_finders)
        _FakeNativeSession.created.clear()
        _FakeNativeSession.fail_create = False
        _FakeNativeSession.fail_call = False
        _FakeNativeSession.fail_combine_call = False
        _FakeNativeSession.mismatch = False
        _FakeNativeSession.combine_output_delta = 0.0
        _FakeNativeSession.combine_lse_delta = 0.0

    @staticmethod
    def _remove_combine_finders() -> None:
        sys.meta_path[:] = [
            finder
            for finder in sys.meta_path
            if not isinstance(finder, backend_module._CombineFinder)
        ]

    def _install(
        self,
        mode: str | None,
        *,
        include_combine: bool = True,
        **environment: str,
    ) -> None:
        values = dict(environment)
        if mode is not None:
            values["VLLM_SPARK_TP4_DCP_MODE"] = mode
        modules = dict(self.modules)
        if not include_combine:
            modules.pop(backend_module._COMBINE_TARGET)
        patchers = (
            patch.dict(os.environ, values, clear=True),
            patch.dict(sys.modules, modules),
            patch.object(backend_module, "_NativeDcpSession", _FakeNativeSession),
            patch.object(
                backend_module,
                "_graph_preflight",
                return_value=(10, 13),
            ),
            patch.object(backend_module, "_numeric_sample", _fake_numeric_sample),
            patch.object(
                backend_module,
                "_abort_after_native_failure",
                side_effect=_AbortCalled("abort"),
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        if not include_combine:
            sys.modules.pop(backend_module._COMBINE_TARGET, None)
        backend_module.install()

    def test_unset_mode_does_not_patch_group(self) -> None:
        self._install(None)
        self.assertIs(self.group_type._all_gather_out_place, self.original)
        self.assertFalse(backend_module._installed)

    def test_invalid_mode_and_shadow_limit_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 'shadow', 'custom'"):
            self._install("fast")
        self.assertIs(self.group_type._all_gather_out_place, self.original)

        backend_module._installed = False
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._install("shadow", SPARK_TP4_DCP_SHADOW_COLLECTIVES="0")

    def test_family_flags_require_exact_booleans_even_without_mode(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "VLLM_SPARK_TP4_DCP_QUERY_ENABLED must be '0' or '1'",
        ):
            self._install(
                None,
                VLLM_SPARK_TP4_DCP_QUERY_ENABLED="true",
            )

    def test_combine_flag_requires_exact_boolean_even_without_mode(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "VLLM_SPARK_TP4_DCP_COMBINE_ENABLED must be '0' or '1'",
        ):
            self._install(
                None,
                VLLM_SPARK_TP4_DCP_COMBINE_ENABLED="true",
            )

    def test_custom_mode_rejects_both_families_disabled(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "unset aggregate mode for stock/stock",
        ):
            self._install(
                "custom",
                VLLM_SPARK_TP4_DCP_QUERY_ENABLED="0",
                VLLM_SPARK_TP4_DCP_COMBINE_ENABLED="0",
            )

    def test_graph_custom_query_only_keeps_combine_on_stock(self) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_DCP_QUERY_ENABLED="1",
            VLLM_SPARK_TP4_DCP_COMBINE_ENABLED="0",
            VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM="1",
        )
        group = self.group_type(rank_in_group=2)

        # The enabled query family prepares the shared graph session before
        # capture.  The disabled combine family must remain an ordinary stock
        # call even while global DCP graph custom mode is active.
        group._all_gather_out_place(_FakeTensor((2, 16, 576), payload=7), 1)
        self.torch_module.cuda.capturing = True
        query_result = group._all_gather_out_place(
            _FakeTensor((5, 16, 576), payload=17), 1
        )
        combine_result = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=23),
            _FakeTensor((5, 64), "torch.float32", payload=29),
            group,
            return_lse=True,
        )
        self.torch_module.cuda.capturing = False

        graph_sessions = [
            session for session in _FakeNativeSession.created if session.graph_only
        ]
        self.assertEqual(len(graph_sessions), 1)
        self.assertEqual(graph_sessions[0].operation_order, [("graph_query", 5)])
        self.assertEqual(query_result.payload, 17)
        self.assertEqual(combine_result[0].payload, 23)
        self.assertEqual(len(group.combine_original_calls), 1)
        self.assertEqual(graph_sessions[0].graph_combine_calls, [])
        self.assertEqual(
            backend_module.dcp_graph_diagnostic_snapshot()["family_selection"],
            {
                "mode": "custom",
                "query_enabled": True,
                "combine_enabled": False,
            },
        )

    def test_graph_custom_combine_only_keeps_query_on_stock(self) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_DCP_QUERY_ENABLED="0",
            VLLM_SPARK_TP4_DCP_COMBINE_ENABLED="1",
            VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM="1",
        )
        group = self.group_type(rank_in_group=2)

        # The enabled combine family prepares the same global graph session
        # during its eager warmup; no custom query node is required.
        self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(2, payload=3),
            _FakeTensor((2, 64), "torch.float32", payload=5),
            group,
            return_lse=True,
        )
        self.torch_module.cuda.capturing = True
        query_result = group._all_gather_out_place(
            _FakeTensor((5, 16, 576), payload=17), 1
        )
        combine_result = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=23),
            _FakeTensor((5, 64), "torch.float32", payload=29),
            group,
            return_lse=True,
        )
        self.torch_module.cuda.capturing = False

        graph_sessions = [
            session for session in _FakeNativeSession.created if session.graph_only
        ]
        self.assertEqual(len(graph_sessions), 1)
        self.assertEqual(graph_sessions[0].operation_order, [("graph_combine", 5)])
        self.assertEqual(query_result.payload, 17)
        self.assertEqual(combine_result[0].payload, 23)
        self.assertEqual(len(group.original_calls), 1)
        self.assertEqual(graph_sessions[0].graph_query_calls, [])
        self.assertEqual(
            backend_module.dcp_graph_diagnostic_snapshot()["family_selection"],
            {
                "mode": "custom",
                "query_enabled": False,
                "combine_enabled": True,
            },
        )

    def test_shared_capture_context_warmup_uses_stock_then_graph_native(
        self,
    ) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM="1",
            VLLM_SPARK_SHARED_CAPTURE_STREAM="1",
            SPARK_TP4_GRAPH_STATUS_PATH="/tmp/status.json",
        )
        key = (os.getpid(), 0)
        parallel_state = self.modules["vllm.distributed.parallel_state"]
        parallel_state._SPARK_ACTIVE_CAPTURE_STREAMS = {key}
        parallel_state._SPARK_SHARED_CAPTURE_STREAMS = {
            key: self.torch_module.cuda.stream
        }
        group = self.group_type(rank_in_group=2)

        query_warmup = group._all_gather_out_place(
            _FakeTensor((5, 16, 576), payload=17), 1
        )
        combine_warmup = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=23),
            _FakeTensor((5, 64), "torch.float32", payload=29),
            group,
            return_lse=True,
        )

        sessions = list(_FakeNativeSession.created)
        self.assertEqual(len(sessions), 1)
        self.assertTrue(sessions[0].graph_only)
        self.assertEqual(sessions[0].query_calls, [])
        self.assertEqual(sessions[0].combine_calls, [])
        self.assertEqual(query_warmup.payload, 17)
        self.assertEqual(combine_warmup[0].payload, 23)
        self.assertEqual(len(group.original_calls), 1)
        self.assertEqual(len(group.combine_original_calls), 1)

        self.torch_module.cuda.capturing = True
        query_capture = group._all_gather_out_place(
            _FakeTensor((5, 16, 576), payload=31), 1
        )
        combine_capture = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=37),
            _FakeTensor((5, 64), "torch.float32", payload=41),
            group,
            return_lse=True,
        )
        self.torch_module.cuda.capturing = False

        self.assertEqual(query_capture.payload, 31)
        self.assertEqual(combine_capture[0].payload, 37)
        self.assertEqual(
            sessions[0].operation_order,
            [("graph_query", 5), ("graph_combine", 5)],
        )
        snapshot = spark_collective_audit.stock_collective_snapshot()
        self.assertEqual(
            snapshot["eager"],
            {
                "dcp_combine:shared_capture_warmup_reference": 1,
                "dcp_query_all_gather:shared_capture_warmup_reference": 1,
            },
        )

    def test_inactive_shared_capture_marker_keeps_eager_custom(self) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM="1",
            VLLM_SPARK_SHARED_CAPTURE_STREAM="1",
        )
        parallel_state = self.modules["vllm.distributed.parallel_state"]
        parallel_state._SPARK_ACTIVE_CAPTURE_STREAMS = set()
        parallel_state._SPARK_SHARED_CAPTURE_STREAMS = {}
        group = self.group_type(rank_in_group=2)

        result = group._all_gather_out_place(
            _FakeTensor((5, 16, 576), payload=17), 1
        )

        graph_sessions = [
            session for session in _FakeNativeSession.created if session.graph_only
        ]
        eager_sessions = [
            session for session in _FakeNativeSession.created if not session.graph_only
        ]
        self.assertEqual(len(graph_sessions), 1)
        self.assertEqual(len(eager_sessions), 1)
        self.assertEqual(eager_sessions[0].operation_order, [("query", 5)])
        self.assertEqual(result.payload, 17)
        self.assertEqual(group.original_calls, [])

    def test_active_shared_capture_marker_rejects_wrong_current_stream(
        self,
    ) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM="1",
            VLLM_SPARK_SHARED_CAPTURE_STREAM="1",
        )
        key = (os.getpid(), 0)
        parallel_state = self.modules["vllm.distributed.parallel_state"]
        parallel_state._SPARK_ACTIVE_CAPTURE_STREAMS = {key}
        parallel_state._SPARK_SHARED_CAPTURE_STREAMS = {
            key: types.SimpleNamespace(cuda_stream=0xBAD)
        }
        group = self.group_type(rank_in_group=2)

        with self.assertRaisesRegex(
            RuntimeError,
            "warmup left its retained stream",
        ):
            group._all_gather_out_place(
                _FakeTensor((5, 16, 576), payload=17), 1
            )

        self.assertEqual(_FakeNativeSession.created, [])
        self.assertEqual(group.original_calls, [])

    def test_custom_routes_q1_through_q6_on_one_native_handle(self) -> None:
        self._install("custom")
        group = self.group_type(rank_in_group=3)
        outputs = []
        for q in range(1, 7):
            input_tensor = _FakeTensor((q, 16, 576), payload=100 + q)
            output = group._all_gather_out_place(input_tensor, 1)
            outputs.append(output)
            self.assertEqual(output.shape, (q, 64, 576))
            self.assertEqual(output.payload, input_tensor.payload)

        self.assertEqual(group.original_calls, [])
        self.assertEqual(len(_FakeNativeSession.created), 1)
        native = _FakeNativeSession.created[0]
        self.assertEqual(native.rank, 3)
        self.assertEqual(
            [call[2] for call in native.query_calls],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertTrue(
            all(call[3] is self.torch_module.cuda.stream for call in native.query_calls)
        )

    def test_non_exact_signatures_and_capture_use_original(self) -> None:
        self._install("custom")
        cases = (
            (self.group_type(unique_name="tp:0"), _FakeTensor((3, 16, 576)), 1),
            (self.group_type(world_size=2), _FakeTensor((3, 16, 576)), 1),
            (self.group_type(), _FakeTensor((7, 16, 576)), 1),
            (self.group_type(), _FakeTensor((3, 8, 576)), 1),
            (
                self.group_type(),
                _FakeTensor((3, 16, 576), "torch.float16"),
                1,
            ),
            (
                self.group_type(),
                _FakeTensor((3, 16, 576), is_cuda=False),
                1,
            ),
            (
                self.group_type(),
                _FakeTensor((3, 16, 576), contiguous=False),
                1,
            ),
            (self.group_type(), _FakeTensor((3, 16, 576)), 0),
        )
        for group, tensor, dim in cases:
            with self.subTest(group=group.unique_name, shape=tensor.shape, dim=dim):
                result = group._all_gather_out_place(tensor, dim)
                self.assertEqual(result.shape, (tensor.shape[0], 64, 576))
                self.assertEqual(len(group.original_calls), 1)

        capture_group = self.group_type()
        self.torch_module.cuda.capturing = True
        capture_group._all_gather_out_place(_FakeTensor((3, 16, 576)), 1)
        self.assertEqual(len(capture_group.original_calls), 1)
        self.assertEqual(_FakeNativeSession.created, [])

    def test_captured_stock_query_is_visible_to_signature_audit(self) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        group = self.group_type()
        self.torch_module.cuda.capturing = True

        group._all_gather_out_place(_FakeTensor((3, 16, 576)), 1)

        snapshot = spark_collective_audit.stock_collective_snapshot()
        self.assertEqual(
            snapshot["capture"],
            {"dcp_query_all_gather:original": 1},
        )
        self.assertEqual(
            snapshot["signatures"]["capture"],
            [
                {
                    "family": "dcp_query_all_gather",
                    "reason": "original",
                    "count": 1,
                    "shape": [3, 16, 576],
                    "dtype": "torch.bfloat16",
                    "is_cuda": True,
                    "contiguous": True,
                    "world_size": 4,
                    "unique_name": "dcp:0",
                }
            ],
        )

    def test_stock_all_gather_audit_labels_lse_and_unknown_abis(self) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        group = self.group_type()

        backend_module._call_stock_all_gather(
            self.original,
            group,
            _FakeTensor((3, 64), "torch.float32"),
            0,
        )
        backend_module._call_stock_all_gather(
            self.original,
            group,
            _FakeTensor((3, 64), "torch.float32"),
            1,
        )

        snapshot = spark_collective_audit.stock_collective_snapshot()
        self.assertEqual(
            snapshot["eager"],
            {
                "dcp_all_gather:original": 1,
                "dcp_lse_all_gather:original": 1,
            },
        )

    def test_graph_shadow_captures_query_and_combine_then_reports_pass(
        self,
    ) -> None:
        self._install(
            "shadow",
            VLLM_SPARK_TP4_DCP_GRAPH_SHADOW="1",
            SPARK_TP4_DCP_SHADOW_COLLECTIVES="8",
        )
        group = self.group_type(rank_in_group=2)

        # The ordinary warmup creates both peer-connected sessions before
        # CUDA capture begins.
        group._all_gather_out_place(_FakeTensor((2, 16, 576), payload=7), 1)
        self.torch_module.cuda.capturing = True
        query_reference = group._all_gather_out_place(
            _FakeTensor((5, 16, 576), payload=17), 1
        )
        combine_reference = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=23),
            _FakeTensor((5, 64), "torch.float32", payload=29),
            group,
            return_lse=True,
        )
        self.torch_module.cuda.capturing = False

        self.assertEqual(query_reference.payload, 17)
        self.assertEqual(combine_reference[0].payload, 23)
        graph_sessions = [
            session for session in _FakeNativeSession.created if session.graph_only
        ]
        self.assertEqual(len(graph_sessions), 1)
        self.assertEqual(
            graph_sessions[0].operation_order,
            [("graph_query", 5), ("graph_combine", 5)],
        )
        report = backend_module.dcp_graph_shadow_report()
        self.assertTrue(report["passed"])
        rank = report["ranks"][2]
        self.assertEqual(rank["query_nodes"], 1)
        self.assertEqual(rank["combine_nodes"], 1)
        self.assertEqual(rank["replayed_query_nodes"], 1)
        self.assertEqual(rank["replayed_combine_nodes"], 1)
        self.assertEqual(rank["unreplayed_query_nodes"], 0)
        self.assertEqual(rank["unreplayed_combine_nodes"], 0)
        self.assertEqual(rank["query"][0]["replays"], 1)
        self.assertEqual(rank["combine"][0]["replays"], 1)
        self.assertTrue(rank["contract_passed"])
        self.assertTrue(rank["correctness_passed"])

    def test_graph_shadow_detects_query_byte_mismatch(self) -> None:
        self._install(
            "shadow",
            VLLM_SPARK_TP4_DCP_GRAPH_SHADOW="1",
            SPARK_TP4_DCP_SHADOW_COLLECTIVES="8",
        )
        group = self.group_type()
        group._all_gather_out_place(_FakeTensor((1, 16, 576), payload=3), 1)
        _FakeNativeSession.mismatch = True
        self.torch_module.cuda.capturing = True
        group._all_gather_out_place(_FakeTensor((3, 16, 576), payload=11), 1)
        self.torch_module.cuda.capturing = False

        report = backend_module.dcp_graph_shadow_report()
        self.assertFalse(report["passed"])
        rank = report["ranks"][2]
        self.assertFalse(rank["correctness_passed"])
        self.assertEqual(rank["query"][0]["byte_mismatches"], 1)

    def test_graph_shadow_ignores_unreplayed_capture_nodes(self) -> None:
        self._install(
            "shadow",
            VLLM_SPARK_TP4_DCP_GRAPH_SHADOW="1",
            SPARK_TP4_DCP_SHADOW_COLLECTIVES="8",
        )
        group = self.group_type(rank_in_group=2)
        group._all_gather_out_place(_FakeTensor((2, 16, 576), payload=7), 1)
        self.torch_module.cuda.capturing = True
        group._all_gather_out_place(_FakeTensor((5, 16, 576), payload=17), 1)
        self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=23),
            _FakeTensor((5, 64), "torch.float32", payload=29),
            group,
            return_lse=True,
        )
        self.torch_module.cuda.capturing = False

        state = group._spark_tp4_dcp_native
        state.graph_query_shadows[0].replays.value = 0
        state.graph_combine_shadows[0].replays.value = 0
        report = backend_module.dcp_graph_shadow_report()
        rank = report["ranks"][2]
        self.assertFalse(report["passed"])
        self.assertTrue(rank["contract_passed"])
        self.assertFalse(rank["correctness_passed"])
        self.assertEqual(rank["query_nodes"], 1)
        self.assertEqual(rank["combine_nodes"], 1)
        self.assertEqual(rank["replayed_query_nodes"], 0)
        self.assertEqual(rank["replayed_combine_nodes"], 0)
        self.assertEqual(rank["unreplayed_query_nodes"], 1)
        self.assertEqual(rank["unreplayed_combine_nodes"], 1)
        self.assertEqual(rank["query"], [])
        self.assertEqual(rank["combine"], [])

    def test_graph_shadow_capacity_exhaustion_is_fatal(self) -> None:
        self._install(
            "shadow",
            VLLM_SPARK_TP4_DCP_GRAPH_SHADOW="1",
            SPARK_TP4_DCP_GRAPH_SHADOW_CAPACITY="1",
        )
        group = self.group_type()
        group._all_gather_out_place(_FakeTensor((1, 16, 576), payload=3), 1)
        state = group._spark_tp4_dcp_native
        self.assertIsNotNone(state.graph_shadow_arena)
        self.assertEqual(state.graph_shadow_arena.query_used, 0)
        self.torch_module.cuda.capturing = True
        group._all_gather_out_place(_FakeTensor((5, 16, 576), payload=11), 1)
        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(_FakeTensor((3, 16, 576), payload=13), 1)

    def test_graph_capture_without_prepared_session_is_fatal(self) -> None:
        self._install(
            "custom",
            VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM="1",
        )
        self.torch_module.cuda.capturing = True
        with self.assertRaises(_AbortCalled):
            self.group_type()._all_gather_out_place(_FakeTensor((3, 16, 576)), 1)

    def test_graph_modes_are_fail_closed_and_mutually_exclusive(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self._install(
                "shadow",
                VLLM_SPARK_TP4_DCP_GRAPH_SHADOW="1",
                VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM="1",
            )

    def test_shadow_promotes_each_q_independently(self) -> None:
        self._install(
            "shadow",
            SPARK_TP4_DCP_SHADOW_COLLECTIVES="2",
            SPARK_TP4_DCP_SHADOW_PROMOTE="1",
        )
        group = self.group_type()
        q1_input = _FakeTensor((1, 16, 576), payload=11)
        q3_input = _FakeTensor((3, 16, 576), payload=33)

        first = group._all_gather_out_place(q1_input, 1)
        second = group._all_gather_out_place(q1_input, 1)
        q3_reference = group._all_gather_out_place(q3_input, 1)
        promoted = group._all_gather_out_place(q1_input, 1)

        self.assertEqual(first.payload, 11)
        self.assertEqual(second.payload, 11)
        self.assertEqual(q3_reference.payload, 33)
        self.assertEqual(promoted.payload, 11)
        self.assertEqual(len(group.original_calls), 3)
        state = group._spark_tp4_dcp_native
        self.assertTrue(state.query_shadows[1].validated)
        self.assertEqual(state.query_shadows[3].count, 1)
        self.assertFalse(state.query_shadows[3].validated)
        self.assertEqual(
            [call[2] for call in _FakeNativeSession.created[0].query_calls],
            [1, 1, 3, 1],
        )

    def test_shadow_mismatch_is_fatal_to_validation(self) -> None:
        _FakeNativeSession.mismatch = True
        self._install("shadow", SPARK_TP4_DCP_SHADOW_COLLECTIVES="1")
        group = self.group_type()
        with self.assertRaisesRegex(RuntimeError, "byte mismatches"):
            group._all_gather_out_place(_FakeTensor((2, 16, 576), payload=20), 1)
        self.assertFalse(group._spark_tp4_dcp_native.query_shadows[2].validated)

    def test_custom_query_create_failure_is_fatal_without_stock_fallback(
        self,
    ) -> None:
        _FakeNativeSession.fail_create = True
        self._install("custom")
        group = self.group_type()
        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(_FakeTensor((3, 16, 576), payload=30), 1)
        self.assertEqual(group.original_calls, [])
        self.assertTrue(group._spark_tp4_dcp_native.disabled)

    def test_shadow_query_create_failure_falls_back_before_enqueue(self) -> None:
        _FakeNativeSession.fail_create = True
        self._install("shadow")
        group = self.group_type()
        result = group._all_gather_out_place(
            _FakeTensor((3, 16, 576), payload=30),
            1,
        )
        self.assertEqual(result.payload, 30)
        self.assertEqual(len(group.original_calls), 1)
        self.assertTrue(group._spark_tp4_dcp_native.disabled)

    def test_custom_combine_create_failure_is_fatal_without_stock_fallback(
        self,
    ) -> None:
        _FakeNativeSession.fail_create = True
        self._install(
            "custom",
            VLLM_SPARK_TP4_DCP_QUERY_ENABLED="0",
            VLLM_SPARK_TP4_DCP_COMBINE_ENABLED="1",
        )
        group = self.group_type()
        with self.assertRaises(_AbortCalled):
            self.flash_module.cp_lse_ag_out_rs(
                _live_combine_output(3, payload=31),
                _FakeTensor((3, 64), "torch.float32", payload=37),
                group,
                return_lse=True,
            )
        self.assertEqual(group.combine_original_calls, [])
        self.assertTrue(group._spark_tp4_dcp_native.disabled)

    def test_native_call_failure_aborts_worker(self) -> None:
        _FakeNativeSession.fail_call = True
        self._install("custom")
        group = self.group_type()
        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(_FakeTensor((3, 16, 576)), 1)

    def test_reference_failure_after_native_enqueue_aborts_worker(self) -> None:
        self._install("shadow")
        group = self.group_type()
        group.fail_original = True
        with self.assertRaises(_AbortCalled):
            group._all_gather_out_place(_FakeTensor((3, 16, 576)), 1)
        self.assertEqual(len(_FakeNativeSession.created[0].query_calls), 1)

    def test_custom_combine_q1_through_q6_shares_ordered_session(self) -> None:
        self._install("custom")
        group = self.group_type(rank_in_group=1)
        group._all_gather_out_place(_FakeTensor((2, 16, 576)), 1)

        for q in range(1, 7):
            output = _live_combine_output(q, payload=10 + q)
            lse = _FakeTensor(
                (q, 64),
                "torch.float32",
                contiguous=False,
                payload=20 + q,
            )
            reduced_output, reduced_lse = self.flash_module.cp_lse_ag_out_rs(
                output,
                lse,
                group,
                ctx={"q": q},
                return_lse=True,
                is_lse_base_on_e=True,
            )
            self.assertEqual(reduced_output.shape, (q, 16, 256))
            self.assertEqual(reduced_output.dtype, "torch.bfloat16")
            self.assertEqual(reduced_output.payload, output.payload)
            self.assertEqual(reduced_lse.shape, (q, 16))
            self.assertEqual(reduced_lse.dtype, "torch.float32")
            self.assertEqual(reduced_lse.payload, lse.payload)
            self.assertEqual(output.stride(), (256, q * 256, 1))
            self.assertEqual(output.contiguous_calls, 0)
            self.assertEqual(lse.contiguous_calls, 1)

        self.assertEqual(group.combine_original_calls, [])
        self.assertEqual(len(_FakeNativeSession.created), 1)
        native = _FakeNativeSession.created[0]
        self.assertEqual(native.rank, 1)
        self.assertEqual(
            native.operation_order,
            [("query", 2), *[("combine", q) for q in range(1, 7)]],
        )
        self.assertTrue(
            all(
                call[1].is_contiguous() and call[5] is self.torch_module.cuda.stream
                for call in native.combine_calls
            )
        )

    def test_custom_combine_without_lse_matches_live_decode_contract(self) -> None:
        self._install("custom")
        group = self.group_type(rank_in_group=1)
        output = _live_combine_output(5, payload=15)
        lse = _FakeTensor((5, 64), "torch.float32", payload=25)

        reduced_output = self.flash_module.cp_lse_ag_out_rs(
            output,
            lse,
            group,
            is_lse_base_on_e=True,
        )

        self.assertNotIsInstance(reduced_output, tuple)
        self.assertEqual(reduced_output.shape, (5, 16, 256))
        self.assertEqual(reduced_output.payload, 15)
        self.assertEqual(group.combine_original_calls, [])
        self.assertEqual(len(_FakeNativeSession.created[0].combine_calls), 1)

    def test_combine_shadow_preserves_default_output_only_contract(self) -> None:
        self._install(
            "shadow",
            SPARK_TP4_DCP_SHADOW_COLLECTIVES="1",
            SPARK_TP4_DCP_SHADOW_PROMOTE="1",
        )
        group = self.group_type(rank_in_group=1)

        shadow_result = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=15),
            _FakeTensor((5, 64), "torch.float32", payload=25),
            group,
        )
        promoted_result = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(5, payload=16),
            _FakeTensor((5, 64), "torch.float32", payload=26),
            group,
        )

        self.assertNotIsInstance(shadow_result, tuple)
        self.assertNotIsInstance(promoted_result, tuple)
        self.assertEqual(shadow_result.payload, 15)
        self.assertEqual(promoted_result.payload, 16)
        self.assertEqual(len(group.combine_original_calls), 1)
        self.assertTrue(group.combine_original_calls[0][3])
        self.assertTrue(
            group._spark_tp4_dcp_native.combine_shadows[
                (5, 256, 256, 5 * 256)
            ].validated
        )
        self.assertEqual(len(_FakeNativeSession.created[0].combine_calls), 2)

    def test_combine_accepts_both_exact_live_layouts_and_dimensions(self) -> None:
        self._install("custom")
        live_group = self.group_type()
        live_output = _live_combine_output(3)
        self.assertFalse(live_output.is_contiguous())
        self.assertEqual(live_output.stride(), (256, 3 * 256, 1))
        self.flash_module.cp_lse_ag_out_rs(
            live_output,
            _FakeTensor((3, 64), "torch.float32"),
            live_group,
            return_lse=True,
        )

        token_major_group = self.group_type()
        token_major_output = _FakeTensor((3, 64, 512))
        self.assertTrue(token_major_output.is_contiguous())
        self.assertEqual(token_major_output.stride(), (64 * 512, 512, 1))
        self.flash_module.cp_lse_ag_out_rs(
            token_major_output,
            _FakeTensor((3, 64), "torch.float32"),
            token_major_group,
            return_lse=True,
        )

        self.assertEqual(live_group.combine_original_calls, [])
        self.assertEqual(token_major_group.combine_original_calls, [])
        self.assertEqual(len(_FakeNativeSession.created), 2)
        self.assertIs(_FakeNativeSession.created[0].combine_calls[0][0], live_output)
        self.assertIs(
            _FakeNativeSession.created[1].combine_calls[0][0],
            token_major_output,
        )
        self.assertEqual(live_output.contiguous_calls, 0)

    def test_combine_non_exact_signatures_and_capture_use_stock(self) -> None:
        self._install("custom")

        def exact() -> tuple[object, _FakeTensor, _FakeTensor]:
            return (
                self.group_type(),
                _live_combine_output(3),
                _FakeTensor((3, 64), "torch.float32"),
            )

        cases: list[tuple[str, object, _FakeTensor, _FakeTensor, bool, bool]] = []
        group, output, lse = exact()
        group.unique_name = "tp:0"
        cases.append(("group", group, output, lse, True, True))
        group, output, lse = exact()
        group.world_size = 2
        cases.append(("world", group, output, lse, True, True))
        group, output, lse = exact()
        group.rank_in_group = 4
        cases.append(("rank", group, output, lse, True, True))
        cases.extend(
            (
                (
                    "q",
                    self.group_type(),
                    _live_combine_output(7),
                    _FakeTensor((7, 64), "torch.float32"),
                    True,
                    True,
                ),
                (
                    "output shape",
                    self.group_type(),
                    _FakeTensor((3, 32, 256)),
                    _FakeTensor((3, 64), "torch.float32"),
                    True,
                    True,
                ),
                (
                    "output dtype",
                    self.group_type(),
                    _live_combine_output(3, "torch.float16"),
                    _FakeTensor((3, 64), "torch.float32"),
                    True,
                    True,
                ),
                (
                    "output cpu",
                    self.group_type(),
                    _live_combine_output(3, is_cuda=False),
                    _FakeTensor((3, 64), "torch.float32"),
                    True,
                    True,
                ),
                (
                    "unsupported head dimension",
                    self.group_type(),
                    _FakeTensor((3, 64, 384)),
                    _FakeTensor((3, 64), "torch.float32"),
                    True,
                    True,
                ),
                (
                    "unsupported strides",
                    self.group_type(),
                    _FakeTensor(
                        (3, 64, 512),
                        strides=(64 * 512 + 1, 512, 1),
                    ),
                    _FakeTensor((3, 64), "torch.float32"),
                    True,
                    True,
                ),
                (
                    "lse shape",
                    self.group_type(),
                    _live_combine_output(3),
                    _FakeTensor((3, 32), "torch.float32"),
                    True,
                    True,
                ),
                (
                    "lse dtype",
                    self.group_type(),
                    _live_combine_output(3),
                    _FakeTensor((3, 64), "torch.bfloat16"),
                    True,
                    True,
                ),
                (
                    "lse cpu",
                    self.group_type(),
                    _live_combine_output(3),
                    _FakeTensor((3, 64), "torch.float32", is_cuda=False),
                    True,
                    True,
                ),
                (
                    "log base",
                    self.group_type(),
                    _live_combine_output(3),
                    _FakeTensor((3, 64), "torch.float32"),
                    True,
                    False,
                ),
            )
        )
        group, output, lse = exact()
        lse.device = "cuda:1"
        cases.append(("device", group, output, lse, True, True))

        for label, group, output, lse, return_lse, base_e in cases:
            with self.subTest(label=label):
                result = self.flash_module.cp_lse_ag_out_rs(
                    output,
                    lse,
                    group,
                    ctx=label,
                    return_lse=return_lse,
                    is_lse_base_on_e=base_e,
                )
                if return_lse:
                    self.assertIsInstance(result, tuple)
                else:
                    self.assertNotIsInstance(result, tuple)
                self.assertEqual(len(group.combine_original_calls), 1)

        capture_group, output, lse = exact()
        self.torch_module.cuda.capturing = True
        self.flash_module.cp_lse_ag_out_rs(output, lse, capture_group, return_lse=True)
        self.assertEqual(len(capture_group.combine_original_calls), 1)
        self.assertEqual(_FakeNativeSession.created, [])

    def test_captured_stock_combine_is_visible_to_audit(self) -> None:
        self._install("custom")
        os.environ["SPARK_TP4_GRAPH_STATUS_PATH"] = "/tmp/status.json"
        group = self.group_type()
        output = _live_combine_output(3)
        lse = _FakeTensor((3, 64), "torch.float32")
        self.torch_module.cuda.capturing = True

        self.flash_module.cp_lse_ag_out_rs(
            output,
            lse,
            group,
            return_lse=True,
        )

        snapshot = spark_collective_audit.stock_collective_snapshot()
        self.assertEqual(
            snapshot["capture"],
            {"dcp_combine:original": 1},
        )
        self.assertEqual(
            snapshot["signatures"]["capture"],
            [
                {
                    "family": "dcp_combine",
                    "reason": "original",
                    "count": 1,
                    "shape": [3, 64, 256],
                    "dtype": "torch.bfloat16",
                    "is_cuda": True,
                    "contiguous": False,
                    "world_size": 4,
                    "unique_name": "dcp:0",
                }
            ],
        )

    def test_combine_shadow_runs_stock_and_promotes_each_q(self) -> None:
        self._install(
            "shadow",
            SPARK_TP4_DCP_SHADOW_COLLECTIVES="2",
            SPARK_TP4_DCP_SHADOW_PROMOTE="1",
        )
        group = self.group_type()

        def invoke(q: int, payload: float) -> tuple[_FakeTensor, _FakeTensor]:
            return self.flash_module.cp_lse_ag_out_rs(
                _live_combine_output(q, payload=payload),
                _FakeTensor((q, 64), "torch.float32", payload=payload + 1),
                group,
                return_lse=True,
            )

        first = invoke(1, 11)
        second = invoke(1, 12)
        q3_reference = invoke(3, 33)
        promoted = invoke(1, 14)

        self.assertEqual(first[0].payload, 11)
        self.assertEqual(second[0].payload, 12)
        self.assertEqual(q3_reference[0].payload, 33)
        self.assertEqual(promoted[0].payload, 14)
        self.assertEqual(len(group.combine_original_calls), 3)
        backend = group._spark_tp4_dcp_native
        signature1 = (1, 256, 256, 256)
        signature3 = (3, 256, 256, 3 * 256)
        self.assertTrue(backend.combine_shadows[signature1].validated)
        self.assertEqual(backend.combine_shadows[signature3].count, 1)
        self.assertFalse(backend.combine_shadows[signature3].validated)
        self.assertEqual(
            [call[4][0] for call in _FakeNativeSession.created[0].combine_calls],
            [1, 1, 3, 1],
        )

    def test_combine_output_tolerance_failure_aborts_worker(self) -> None:
        _FakeNativeSession.combine_output_delta = 0.1
        self._install("shadow", SPARK_TP4_DCP_SHADOW_COLLECTIVES="1")
        group = self.group_type()
        with self.assertRaises(_AbortCalled):
            self.flash_module.cp_lse_ag_out_rs(
                _live_combine_output(2),
                _FakeTensor((2, 64), "torch.float32"),
                group,
                return_lse=True,
            )
        self.assertFalse(
            group._spark_tp4_dcp_native.combine_shadows[
                (2, 256, 256, 2 * 256)
            ].validated
        )

    def test_combine_lse_tolerance_failure_aborts_worker(self) -> None:
        _FakeNativeSession.combine_lse_delta = 3.0e-5
        self._install("shadow", SPARK_TP4_DCP_SHADOW_COLLECTIVES="1")
        with self.assertRaises(_AbortCalled):
            self.flash_module.cp_lse_ag_out_rs(
                _live_combine_output(2),
                _FakeTensor((2, 64), "torch.float32"),
                self.group_type(),
                return_lse=True,
            )

    def test_shadow_combine_create_failure_falls_back_before_enqueue(self) -> None:
        _FakeNativeSession.fail_create = True
        self._install("shadow")
        group = self.group_type()
        result = self.flash_module.cp_lse_ag_out_rs(
            _live_combine_output(3, payload=30),
            _FakeTensor((3, 64), "torch.float32", payload=40),
            group,
            return_lse=True,
        )
        self.assertEqual(result[0].payload, 30)
        self.assertEqual(len(group.combine_original_calls), 1)
        self.assertTrue(group._spark_tp4_dcp_native.disabled)

    def test_combine_native_failure_aborts_worker(self) -> None:
        _FakeNativeSession.fail_combine_call = True
        self._install("custom")
        with self.assertRaises(_AbortCalled):
            self.flash_module.cp_lse_ag_out_rs(
                _live_combine_output(3),
                _FakeTensor((3, 64), "torch.float32"),
                self.group_type(),
                return_lse=True,
            )

    def test_combine_stock_failure_after_enqueue_aborts_worker(self) -> None:
        self._install("shadow")
        group = self.group_type()
        group.fail_combine_original = True
        with self.assertRaises(_AbortCalled):
            self.flash_module.cp_lse_ag_out_rs(
                _live_combine_output(3),
                _FakeTensor((3, 64), "torch.float32"),
                group,
                return_lse=True,
            )
        self.assertEqual(len(_FakeNativeSession.created[0].combine_calls), 1)

    def test_invalid_combine_tolerance_is_rejected_before_patch(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            self._install("shadow", SPARK_TP4_DCP_COMBINE_LSE_ATOL="-0.1")
        self.assertIs(self.flash_module.cp_lse_ag_out_rs, self.original_combine)

    def test_deferred_combine_import_installs_one_shot_finder(self) -> None:
        self._install("custom", include_combine=False)
        finders = [
            finder
            for finder in sys.meta_path
            if isinstance(finder, backend_module._CombineFinder)
        ]
        self.assertEqual(len(finders), 1)

    def test_combine_hook_is_installed_before_first_vllm_import(self) -> None:
        real_import = __import__
        finder_present = []

        def traced_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "vllm.distributed.parallel_state":
                finder_present.append(
                    any(
                        isinstance(finder, backend_module._CombineFinder)
                        for finder in sys.meta_path
                    )
                )
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=traced_import):
            self._install("custom", include_combine=False)

        self.assertEqual(finder_present, [True])

    def test_preimported_combine_alias_is_repaired(self) -> None:
        consumer = types.ModuleType("live.mla_attention")
        consumer.cp_lse_ag_out_rs = self.original_combine
        self.modules[consumer.__name__] = consumer

        self._install("custom")

        self.assertIs(
            consumer.cp_lse_ag_out_rs,
            self.flash_module.cp_lse_ag_out_rs,
        )

    def test_install_is_idempotent(self) -> None:
        self._install("custom")
        patched = self.group_type._all_gather_out_place
        patched_combine = self.flash_module.cp_lse_ag_out_rs
        backend_module.install()
        self.assertIs(self.group_type._all_gather_out_place, patched)
        self.assertIs(self.flash_module.cp_lse_ag_out_rs, patched_combine)


class _FakeFunction:
    def __init__(self, result: object = None) -> None:
        self.result = result
        self.calls: list[tuple[object, ...]] = []
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        if self.argtypes is not None and len(args) != len(self.argtypes):
            raise TypeError(
                f"fake ABI expected {len(self.argtypes)} arguments, got {len(args)}"
            )
        self.calls.append(args)
        return self.result


class _FakeGraphStatusFunction(_FakeFunction):
    def __init__(self) -> None:
        super().__init__(0)

    def __call__(self, *args: object) -> object:
        result = super().__call__(*args)
        status = args[1]._obj
        status.struct_size = backend_module.ctypes.sizeof(
            backend_module._NativeDcpGraphStatus
        )
        status.flags = 0x5F
        status.captured_nodes = 2
        status.captured_query_nodes = 1
        status.captured_combine_nodes = 1
        status.published_sequence = 20
        status.consumed_sequence = 20
        status.completed_sequence = 20
        status.overflow_sequence = 0
        status.graph_submit_cpu_plus_one = 11
        status.graph_progress_cpu_plus_one = 14
        return result


class _FakeLibrary:
    def __init__(self) -> None:
        self.spark_tp4_dcp_create = _FakeFunction(0x1234)
        self.spark_tp4_dcp_graph_create = _FakeFunction(0x5678)
        self.spark_tp4_dcp_query_all_gather = _FakeFunction(0)
        self.spark_tp4_dcp_combine = _FakeFunction(0)
        self.spark_tp4_dcp_capture_query_all_gather = _FakeFunction(0)
        self.spark_tp4_dcp_capture_combine = _FakeFunction(0)
        self.spark_tp4_dcp_get_graph_status = _FakeGraphStatusFunction()
        self.spark_tp4_dcp_destroy = _FakeFunction(None)


class NativeBindingTest(unittest.TestCase):
    def test_binds_generic_handle_and_dynamic_q_call(self) -> None:
        library = _FakeLibrary()
        environment = {
            "SPARK_TP4_LIBRARY": "/tmp/libspark_transport_capi.so",
            "SPARK_TP4_DCP_CONTROL_PORT0": "10090",
            "SPARK_TP4_DCP_CONTROL_PORT1": "10091",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(backend_module.ctypes, "CDLL", return_value=library),
        ):
            session = backend_module._NativeDcpSession(2)
            input_tensor = _FakeTensor((5, 16, 576))
            output_tensor = _FakeTensor((5, 64, 576))
            combine_output = _live_combine_output(5)
            combine_lse = _FakeTensor((5, 64), "torch.float32")
            reduced_output = _FakeTensor((5, 16, 256))
            reduced_lse = _FakeTensor((5, 16), "torch.float32")
            stream = _FakeStream()
            session.query_all_gather(input_tensor, output_tensor, 5, stream)
            session.combine(
                combine_output,
                combine_lse,
                reduced_output,
                reduced_lse,
                (5, 256, 256, 5 * 256),
                stream,
            )

        create_args = library.spark_tp4_dcp_create.calls[0]
        config = create_args[0]._obj
        self.assertEqual(config.rank, 2)
        self.assertEqual(config.control_port0, 10090)
        self.assertEqual(config.control_port1, 10091)
        query_args = library.spark_tp4_dcp_query_all_gather.calls[0]
        self.assertEqual(len(library.spark_tp4_dcp_query_all_gather.argtypes), 7)
        self.assertEqual(query_args[0], 0x1234)
        self.assertEqual(query_args[1].value, input_tensor.data_ptr())
        self.assertEqual(query_args[2].value, output_tensor.data_ptr())
        self.assertEqual(query_args[3], 5)
        self.assertEqual(query_args[4].value, stream.cuda_stream)
        combine_args = library.spark_tp4_dcp_combine.calls[0]
        self.assertEqual(len(library.spark_tp4_dcp_combine.argtypes), 12)
        self.assertEqual(combine_args[0], 0x1234)
        self.assertEqual(combine_args[1].value, combine_output.data_ptr())
        self.assertEqual(combine_args[2].value, combine_lse.data_ptr())
        self.assertEqual(combine_args[3].value, reduced_output.data_ptr())
        self.assertEqual(combine_args[4].value, reduced_lse.data_ptr())
        self.assertEqual(combine_args[5], 5)
        self.assertEqual(combine_args[6], 256)
        self.assertEqual(combine_args[7], 256)
        self.assertEqual(combine_args[8], 5 * 256)
        self.assertEqual(combine_args[9].value, stream.cuda_stream)

    def test_binds_graph_handle_mixed_families_and_status(self) -> None:
        library = _FakeLibrary()
        environment = {
            "SPARK_TP4_LIBRARY": "/tmp/libspark_transport_capi.so",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(backend_module.ctypes, "CDLL", return_value=library),
        ):
            session = backend_module._NativeDcpSession(
                1,
                graph_only=True,
                control_ports=(10092, 10093),
                graph_cpu_affinity=(10, 13),
            )
            input_tensor = _FakeTensor((5, 16, 576))
            query_output = _FakeTensor((5, 64, 576))
            combine_output = _live_combine_output(5)
            combine_lse = _FakeTensor((5, 64), "torch.float32")
            reduced_output = _FakeTensor((5, 16, 256))
            reduced_lse = _FakeTensor((5, 16), "torch.float32")
            stream = _FakeStream()
            session.capture_query_all_gather(input_tensor, query_output, 5, stream)
            session.capture_combine(
                combine_output,
                combine_lse,
                reduced_output,
                reduced_lse,
                (5, 256, 256, 5 * 256),
                stream,
            )
            status = session.graph_status()

        create_args = library.spark_tp4_dcp_graph_create.calls[0]
        config = create_args[0]._obj
        self.assertEqual(config.rank, 1)
        self.assertEqual(config.control_port0, 10092)
        self.assertEqual(config.control_port1, 10093)
        self.assertEqual(config.graph_submit_cpu_plus_one, 11)
        self.assertEqual(config.graph_progress_cpu_plus_one, 14)
        self.assertEqual(
            len(library.spark_tp4_dcp_capture_query_all_gather.argtypes),
            7,
        )
        self.assertEqual(len(library.spark_tp4_dcp_capture_combine.argtypes), 12)
        self.assertEqual(status["captured_nodes"], 2)
        self.assertEqual(status["captured_query_nodes"], 1)
        self.assertEqual(status["captured_combine_nodes"], 1)
        self.assertTrue(status["replay_caught_up"])
        self.assertTrue(status["dedicated_spin"])
        self.assertEqual(status["submit_cpu"], 10)
        self.assertEqual(status["progress_cpu"], 13)


class NumericSampleTest(unittest.TestCase):
    def test_numeric_sample_treats_matching_infinities_as_equal(self) -> None:
        import torch

        candidate = torch.tensor([1.01, float("inf"), float("-inf"), float("nan")])
        reference = torch.tensor([1.0, float("inf"), float("-inf"), float("nan")])
        outside, nonfinite, max_abs, max_rel = backend_module._numeric_sample(
            candidate, reference, rtol=0.0, atol=0.02
        )
        self.assertEqual(outside.item(), 1)
        self.assertEqual(nonfinite.item(), 1)
        self.assertAlmostEqual(max_abs.item(), 0.01, places=5)
        self.assertAlmostEqual(max_rel.item(), 0.01, places=5)


if __name__ == "__main__":
    unittest.main()
