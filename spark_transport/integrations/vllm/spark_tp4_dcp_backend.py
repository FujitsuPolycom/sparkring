"""Exact-signature vLLM adapter for Spark TP4 DCP query and combine."""

from __future__ import annotations

import ctypes
import importlib.abc
import importlib.machinery
import logging
import math
import os
import sys
from types import ModuleType
from typing import Any

from spark_tp4_query_contract import SUPPORTED_QUERY_ROWS

logger = logging.getLogger(__name__)

_installed = False
_VALID_MODES = {"shadow", "custom"}
_DCP_GROUP = "dcp:0"
_WORLD_SIZE = 4
_QUERY_HEADS_PER_RANK = 16
_QUERY_HEAD_DIM = 576
_OUTPUT_HEADS = _QUERY_HEADS_PER_RANK * _WORLD_SIZE
_BF16_BYTES = 2
_COMBINE_HEADS_PER_RANK = _OUTPUT_HEADS // _WORLD_SIZE
_COMBINE_TARGET = "vllm.v1.attention.ops.common"
_SUPPORTED_Q = SUPPORTED_QUERY_ROWS
_DCP_GRAPH_DEFAULT_PROGRESS_CPU = 13
_DCP_GRAPH_SHADOW_DEFAULT_CAPACITY = 4096
_GRAPH_STATUS_CAPTURE_CONFIGURED = 1 << 0
_GRAPH_STATUS_POLLING_ENABLED = 1 << 1
_GRAPH_STATUS_HOST_NATIVE_ATOMICS = 1 << 2
_GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED = 1 << 3
_GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED = 1 << 4
_GRAPH_STATUS_DEDICATED_SPIN = 1 << 6
_dcp_backends: dict[int, "_Backend"] = {}
_dcp_graph_sessions: dict[int, "_NativeDcpSession"] = {}
_graph_event_counts: dict[str, int] = {}
# PLACEHOLDER ring peers (RFC 5737 TEST-NET-1): 192.0.2.N stands in for
# rank N-1's direct-cable address. These are NOT routable and MUST be
# replaced for any live run by setting SPARK_TP4_PEER0 / SPARK_TP4_PEER1
# (the authoritative per-rank overrides) or by editing this table.
_DEFAULT_PEERS = {
    0: ("192.0.2.2", "192.0.2.4"),
    1: ("192.0.2.1", "192.0.2.3"),
    2: ("192.0.2.4", "192.0.2.2"),
    3: ("192.0.2.3", "192.0.2.1"),
}

_QuerySignature = int
_CombineSignature = tuple[int, int, int, int]


def _abort_after_native_failure() -> None:
    """Terminate a worker whose CUDA stream may contain a native wait."""

    os._exit(72)


def _mode() -> str:
    mode = os.getenv("VLLM_SPARK_TP4_DCP_MODE", "").lower()
    if mode and mode not in _VALID_MODES:
        raise ValueError("VLLM_SPARK_TP4_DCP_MODE must be 'shadow', 'custom', or unset")
    return mode


def _family_enabled(name: str, mode: str) -> bool:
    value = os.getenv(name, "1" if mode else "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0' or '1'")
    return bool(mode) and value == "1"


def _query_enabled(mode: str) -> bool:
    return _family_enabled("VLLM_SPARK_TP4_DCP_QUERY_ENABLED", mode)


def _combine_enabled(mode: str) -> bool:
    return _family_enabled("VLLM_SPARK_TP4_DCP_COMBINE_ENABLED", mode)


def _validate_family_selection(mode: str) -> tuple[bool, bool]:
    query_enabled = _query_enabled(mode)
    combine_enabled = _combine_enabled(mode)
    if mode and not (query_enabled or combine_enabled):
        raise ValueError(
            "VLLM_SPARK_TP4_DCP_MODE requires at least one of "
            "VLLM_SPARK_TP4_DCP_QUERY_ENABLED or "
            "VLLM_SPARK_TP4_DCP_COMBINE_ENABLED to be '1'; "
            "unset aggregate mode for stock/stock"
        )
    return query_enabled, combine_enabled


def _positive_integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _combine_tolerances() -> tuple[float, float, float, float]:
    return (
        _nonnegative_float("SPARK_TP4_DCP_COMBINE_OUTPUT_RTOL", 0.01),
        _nonnegative_float("SPARK_TP4_DCP_COMBINE_OUTPUT_ATOL", 0.0625),
        _nonnegative_float("SPARK_TP4_DCP_COMBINE_LSE_RTOL", 2.0e-6),
        _nonnegative_float("SPARK_TP4_DCP_COMBINE_LSE_ATOL", 2.0e-5),
    )


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _stride(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.stride())


def _query_signature(
    group: Any, input_tensor: Any, dim: int, mode: str
) -> _QuerySignature | None:
    shape = _shape(input_tensor)
    if (
        not _query_enabled(mode)
        or mode not in _VALID_MODES
        or getattr(group, "unique_name", None) != _DCP_GROUP
        or int(getattr(group, "world_size", 0)) != _WORLD_SIZE
        or int(getattr(group, "rank_in_group", -1)) not in range(_WORLD_SIZE)
        or dim != 1
        or len(shape) != 3
        or shape[0] not in _SUPPORTED_Q
        or shape[1:] != (_QUERY_HEADS_PER_RANK, _QUERY_HEAD_DIM)
        or str(input_tensor.dtype) != "torch.bfloat16"
        or not bool(input_tensor.is_cuda)
        or not bool(input_tensor.is_contiguous())
    ):
        return None
    return shape[0]


def _combine_signature(
    group: Any,
    output_tensor: Any,
    lse_tensor: Any,
    return_lse: bool,
    is_lse_base_on_e: bool,
    mode: str,
) -> _CombineSignature | None:
    output_shape = _shape(output_tensor)
    output_stride = _stride(output_tensor)
    lse_shape = _shape(lse_tensor)
    if (
        not _combine_enabled(mode)
        or mode not in _VALID_MODES
        or getattr(group, "unique_name", None) != _DCP_GROUP
        or int(getattr(group, "world_size", 0)) != _WORLD_SIZE
        or int(getattr(group, "rank_in_group", -1)) not in range(_WORLD_SIZE)
        or len(output_shape) != 3
        or output_shape[0] not in _SUPPORTED_Q
        or output_shape[1] != _OUTPUT_HEADS
        or output_shape[2] not in (256, 512)
        or lse_shape != (output_shape[0], _OUTPUT_HEADS)
        or str(output_tensor.dtype) != "torch.bfloat16"
        or str(lse_tensor.dtype) != "torch.float32"
        or not bool(output_tensor.is_cuda)
        or not bool(lse_tensor.is_cuda)
        or output_tensor.device != lse_tensor.device
        or output_stride
        not in (
            (
                _OUTPUT_HEADS * output_shape[2],
                output_shape[2],
                1,
            ),
            (
                output_shape[2],
                output_shape[0] * output_shape[2],
                1,
            ),
        )
        or type(return_lse) is not bool
        or is_lse_base_on_e is not True
    ):
        return None
    return (
        output_shape[0],
        output_shape[2],
        output_stride[0],
        output_stride[1],
    )


def _is_stream_capturing(torch_module: Any) -> bool:
    checker = getattr(torch_module.cuda, "is_current_stream_capturing", None)
    return bool(checker is not None and checker())


def _device_index(torch_module: Any, device: Any) -> int:
    index = getattr(device, "index", None)
    if not isinstance(index, int):
        index = None
    if index is None:
        text = str(device)
        if text.startswith("cuda:"):
            index = int(text.removeprefix("cuda:"))
    if index is None:
        current_device = getattr(torch_module.cuda, "current_device", None)
        if current_device is None:
            raise RuntimeError(
                "Spark shared CUDA graph capture cannot resolve the device index"
            )
        index = current_device()
    return int(index)


def _stream_handle(stream: Any) -> int | None:
    handle = getattr(stream, "cuda_stream", None)
    return None if handle is None else int(handle)


def _is_shared_capture_warmup(torch_module: Any, device: Any) -> bool:
    """Identify vLLM graph-manager warmup on its retained capture stream."""

    if os.getenv("VLLM_SPARK_SHARED_CAPTURE_STREAM", "0") != "1":
        return False
    parallel_state = sys.modules.get("vllm.distributed.parallel_state")
    if parallel_state is None:
        raise RuntimeError(
            "Spark shared CUDA graph capture state is unavailable"
        )
    active = getattr(parallel_state, "_SPARK_ACTIVE_CAPTURE_STREAMS", None)
    streams = getattr(parallel_state, "_SPARK_SHARED_CAPTURE_STREAMS", None)
    if active is None or streams is None:
        raise RuntimeError(
            "Spark shared CUDA graph capture markers are unavailable"
        )
    key = (os.getpid(), _device_index(torch_module, device))
    if key not in active:
        return False
    expected = streams.get(key)
    if expected is None:
        raise RuntimeError(
            "Spark shared CUDA graph capture is active without a retained stream"
        )
    current = torch_module.cuda.current_stream(device=device)
    expected_handle = _stream_handle(expected)
    current_handle = _stream_handle(current)
    if expected_handle is None or current_handle is None:
        matches = current is expected
    else:
        matches = current_handle == expected_handle
    if not matches:
        raise RuntimeError(
            "Spark shared CUDA graph capture warmup left its retained stream"
        )
    return True


def _execution_phase(torch_module: Any, device: Any) -> str:
    if _is_stream_capturing(torch_module):
        return "capture"
    if _is_shared_capture_warmup(torch_module, device):
        return "capture_warmup"
    return "eager"


def _graph_shadow_enabled() -> bool:
    value = os.getenv("VLLM_SPARK_TP4_DCP_GRAPH_SHADOW", "0")
    if value not in {"0", "1"}:
        raise ValueError("VLLM_SPARK_TP4_DCP_GRAPH_SHADOW must be '0' or '1'")
    return value == "1"


def _graph_custom_enabled() -> bool:
    value = os.getenv("VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM", "0")
    if value not in {"0", "1"}:
        raise ValueError("VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM must be '0' or '1'")
    return value == "1"


def _graph_enabled(mode: str) -> bool:
    shadow = _graph_shadow_enabled()
    custom = _graph_custom_enabled()
    if shadow and custom:
        raise ValueError("DCP graph shadow and custom modes are mutually exclusive")
    if shadow and mode != "shadow":
        raise ValueError("VLLM_SPARK_TP4_DCP_GRAPH_SHADOW requires DCP shadow mode")
    if custom and mode != "custom":
        raise ValueError("VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM requires DCP custom mode")
    return shadow or custom


def _validate_control_ports(ports: tuple[int, int]) -> None:
    port0, port1 = ports
    if not (0 < port0 <= 65535 and 0 < port1 <= 65535):
        raise ValueError("Spark TP4 DCP graph control ports must be in [1, 65535]")
    if port0 == port1:
        raise ValueError("Spark TP4 DCP graph control ports must be distinct")


def _graph_control_ports() -> tuple[int, int]:
    ports = (
        int(os.getenv("SPARK_TP4_GRAPH_DCP_CONTROL_PORT0", "9892")),
        int(os.getenv("SPARK_TP4_GRAPH_DCP_CONTROL_PORT1", "9893")),
    )
    _validate_control_ports(ports)
    return ports


def _graph_preflight() -> tuple[int, int]:
    from spark_tp4_backend import _graph_preflight as tp_graph_preflight

    submit_cpu, tp_progress_cpu = tp_graph_preflight()
    progress_cpu = int(
        os.getenv(
            "SPARK_TP4_GRAPH_DCP_PROGRESS_CPU",
            str(_DCP_GRAPH_DEFAULT_PROGRESS_CPU),
        )
    )
    occupied = {submit_cpu, tp_progress_cpu}
    if os.getenv("VLLM_SPARK_TP4_GRAPH_Q1", "0") == "1":
        occupied.add(int(os.getenv("SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU", "12")))
    if progress_cpu < 0:
        raise RuntimeError("Spark TP4 DCP graph progress CPU must be nonnegative")
    if progress_cpu in occupied:
        raise RuntimeError(
            "Spark TP4 DCP graph progress CPU must differ from the "
            "shared submit, TP progress, and vocabulary progress CPUs"
        )
    return submit_cpu, progress_cpu


def _record_graph_event(group: Any, event: str) -> int:
    attribute = f"_spark_tp4_dcp_graph_{event}"
    count = int(getattr(group, attribute, 0)) + 1
    setattr(group, attribute, count)
    _graph_event_counts[event] = _graph_event_counts.get(event, 0) + 1
    if count == 1 or count % 128 == 0:
        logger.warning(
            "Spark TP4 DCP graph %s on rank %d: count=%d",
            event,
            int(group.rank_in_group),
            count,
        )
    return count


class _NativeDcpConfig(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_uint32),
        ("peer0", ctypes.c_char_p),
        ("peer1", ctypes.c_char_p),
        ("device0", ctypes.c_char_p),
        ("device1", ctypes.c_char_p),
        ("gid0", ctypes.c_uint8),
        ("gid1", ctypes.c_uint8),
        ("control_port0", ctypes.c_uint16),
        ("control_port1", ctypes.c_uint16),
    ]


class _NativeDcpGraphConfig(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_uint32),
        ("peer0", ctypes.c_char_p),
        ("peer1", ctypes.c_char_p),
        ("device0", ctypes.c_char_p),
        ("device1", ctypes.c_char_p),
        ("gid0", ctypes.c_uint8),
        ("gid1", ctypes.c_uint8),
        ("control_port0", ctypes.c_uint16),
        ("control_port1", ctypes.c_uint16),
        ("graph_submit_cpu_plus_one", ctypes.c_uint32),
        ("graph_progress_cpu_plus_one", ctypes.c_uint32),
    ]


class _NativeDcpGraphStatus(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("captured_nodes", ctypes.c_uint64),
        ("captured_query_nodes", ctypes.c_uint64),
        ("captured_combine_nodes", ctypes.c_uint64),
        ("published_sequence", ctypes.c_uint64),
        ("consumed_sequence", ctypes.c_uint64),
        ("completed_sequence", ctypes.c_uint64),
        ("overflow_sequence", ctypes.c_uint64),
        ("graph_submit_cpu_plus_one", ctypes.c_uint32),
        ("graph_progress_cpu_plus_one", ctypes.c_uint32),
    ]


class _NativeDcpSession:
    def __init__(
        self,
        rank: int,
        *,
        graph_only: bool = False,
        control_ports: tuple[int, int] | None = None,
        graph_cpu_affinity: tuple[int, int] | None = None,
    ) -> None:
        if rank not in _DEFAULT_PEERS:
            raise ValueError(f"DCP rank must be in [0, 3], got {rank}")
        if graph_only != (graph_cpu_affinity is not None):
            raise ValueError(
                "Spark TP4 DCP graph session requires an explicit CPU "
                "pair; eager sessions cannot set one"
            )
        self._library = ctypes.CDLL(os.environ["SPARK_TP4_LIBRARY"])
        self._library.spark_tp4_dcp_create.argtypes = [
            ctypes.POINTER(_NativeDcpConfig),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_dcp_create.restype = ctypes.c_void_p
        if graph_only:
            self._library.spark_tp4_dcp_graph_create.argtypes = [
                ctypes.POINTER(_NativeDcpGraphConfig),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_dcp_graph_create.restype = ctypes.c_void_p
        self._library.spark_tp4_dcp_query_all_gather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_dcp_query_all_gather.restype = ctypes.c_int
        self._library.spark_tp4_dcp_combine.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_dcp_combine.restype = ctypes.c_int
        if graph_only:
            self._library.spark_tp4_dcp_capture_query_all_gather.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_dcp_capture_query_all_gather.restype = ctypes.c_int
            self._library.spark_tp4_dcp_capture_combine.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_dcp_capture_combine.restype = ctypes.c_int
            self._library.spark_tp4_dcp_get_graph_status.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NativeDcpGraphStatus),
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_dcp_get_graph_status.restype = ctypes.c_int
        self._library.spark_tp4_dcp_destroy.argtypes = [ctypes.c_void_p]
        self._library.spark_tp4_dcp_destroy.restype = None
        self._graph_only = graph_only

        default_peer0, default_peer1 = _DEFAULT_PEERS[rank]
        ports = control_ports or (
            int(os.getenv("SPARK_TP4_DCP_CONTROL_PORT0", "9890")),
            int(os.getenv("SPARK_TP4_DCP_CONTROL_PORT1", "9891")),
        )
        _validate_control_ports(ports)
        port0, port1 = ports
        common_config = {
            "rank": rank,
            "peer0": os.getenv("SPARK_TP4_PEER0", default_peer0).encode(),
            "peer1": os.getenv("SPARK_TP4_PEER1", default_peer1).encode(),
            "device0": os.getenv("SPARK_TP4_DEVICE0", "rocep1s0f0").encode(),
            "device1": os.getenv("SPARK_TP4_DEVICE1", "rocep1s0f1").encode(),
            "gid0": int(os.getenv("SPARK_TP4_GID0", "3")),
            "gid1": int(os.getenv("SPARK_TP4_GID1", "3")),
            "control_port0": port0,
            "control_port1": port1,
        }
        if graph_only:
            assert graph_cpu_affinity is not None
            submit_cpu, progress_cpu = graph_cpu_affinity
            config = _NativeDcpGraphConfig(
                **common_config,
                graph_submit_cpu_plus_one=submit_cpu + 1,
                graph_progress_cpu_plus_one=progress_cpu + 1,
            )
            create = self._library.spark_tp4_dcp_graph_create
        else:
            config = _NativeDcpConfig(**common_config)
            create = self._library.spark_tp4_dcp_create
        error = ctypes.create_string_buffer(512)
        self._handle = create(ctypes.byref(config), error, len(error))
        if not self._handle:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"failed to create Spark TP4 DCP session: {message}")
        logger.warning(
            "Spark TP4 DCP %s session ready: rank=%d ports=%d/%d",
            "graph-only" if graph_only else "eager",
            rank,
            port0,
            port1,
        )

    def query_all_gather(
        self,
        input_tensor: Any,
        output_tensor: Any,
        q: int,
        stream: Any,
    ) -> None:
        if self._graph_only:
            raise RuntimeError("Spark TP4 DCP graph session rejects eager query")
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_dcp_query_all_gather(
            self._handle,
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.c_void_p(output_tensor.data_ptr()),
            q,
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 DCP query failed: {message}")

    def combine(
        self,
        output_tensor: Any,
        lse_tensor: Any,
        reduced_output: Any,
        reduced_lse: Any,
        signature: _CombineSignature,
        stream: Any,
    ) -> None:
        if self._graph_only:
            raise RuntimeError("Spark TP4 DCP graph session rejects eager combine")
        q, head_dimension, query_stride, head_stride = signature
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_dcp_combine(
            self._handle,
            ctypes.c_void_p(output_tensor.data_ptr()),
            ctypes.c_void_p(lse_tensor.data_ptr()),
            ctypes.c_void_p(reduced_output.data_ptr()),
            ctypes.c_void_p(reduced_lse.data_ptr()),
            q,
            head_dimension,
            query_stride,
            head_stride,
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 DCP combine failed: {message}")

    def capture_query_all_gather(
        self,
        input_tensor: Any,
        output_tensor: Any,
        q: int,
        stream: Any,
    ) -> None:
        if not self._graph_only:
            raise RuntimeError("Spark TP4 DCP eager session cannot capture query")
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_dcp_capture_query_all_gather(
            self._handle,
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.c_void_p(output_tensor.data_ptr()),
            q,
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 DCP graph query capture failed: {message}")

    def capture_combine(
        self,
        output_tensor: Any,
        lse_tensor: Any,
        reduced_output: Any,
        reduced_lse: Any,
        signature: _CombineSignature,
        stream: Any,
    ) -> None:
        if not self._graph_only:
            raise RuntimeError("Spark TP4 DCP eager session cannot capture combine")
        q, head_dimension, query_stride, head_stride = signature
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_dcp_capture_combine(
            self._handle,
            ctypes.c_void_p(output_tensor.data_ptr()),
            ctypes.c_void_p(lse_tensor.data_ptr()),
            ctypes.c_void_p(reduced_output.data_ptr()),
            ctypes.c_void_p(reduced_lse.data_ptr()),
            q,
            head_dimension,
            query_stride,
            head_stride,
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 DCP graph combine capture failed: {message}")

    def graph_status(self) -> dict[str, object]:
        if not self._graph_only:
            raise RuntimeError("Spark TP4 DCP eager session has no graph status")
        status = _NativeDcpGraphStatus()
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_dcp_get_graph_status(
            self._handle,
            ctypes.byref(status),
            ctypes.sizeof(status),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 DCP graph status failed: {message}")
        if status.struct_size != ctypes.sizeof(_NativeDcpGraphStatus):
            raise RuntimeError(
                "Spark TP4 DCP graph status ABI mismatch: "
                f"native={status.struct_size} python="
                f"{ctypes.sizeof(_NativeDcpGraphStatus)}"
            )
        flags = int(status.flags)
        published = int(status.published_sequence)
        consumed = int(status.consumed_sequence)
        completed = int(status.completed_sequence)
        overflow = int(status.overflow_sequence)
        return {
            "captured_nodes": int(status.captured_nodes),
            "captured_query_nodes": int(status.captured_query_nodes),
            "captured_combine_nodes": int(status.captured_combine_nodes),
            "published_sequence": published,
            "consumed_sequence": consumed,
            "completed_sequence": completed,
            "overflow_sequence": overflow,
            "capture_configured": bool(flags & _GRAPH_STATUS_CAPTURE_CONFIGURED),
            "polling_enabled": bool(flags & _GRAPH_STATUS_POLLING_ENABLED),
            "host_native_atomics": bool(flags & _GRAPH_STATUS_HOST_NATIVE_ATOMICS),
            "submit_affinity_verified": bool(
                flags & _GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED
            ),
            "progress_affinity_verified": bool(
                flags & _GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED
            ),
            "dedicated_spin": bool(flags & _GRAPH_STATUS_DEDICATED_SPIN),
            "submit_cpu": (
                int(status.graph_submit_cpu_plus_one) - 1
                if status.graph_submit_cpu_plus_one
                else None
            ),
            "progress_cpu": (
                int(status.graph_progress_cpu_plus_one) - 1
                if status.graph_progress_cpu_plus_one
                else None
            ),
            "replay_advanced": published > 0,
            "replay_caught_up": (
                published > 0 and published == consumed and published == completed
            ),
            "fatal": overflow != 0,
        }


class _QueryShadowState:
    def __init__(self, output_tensor: Any) -> None:
        self.candidate = output_tensor
        self.mismatches: Any | None = None
        self.count = 0
        self.validated = False

    def observe(self, reference: Any) -> None:
        import torch

        mismatch = torch.count_nonzero(
            self.candidate.view(torch.uint8) != reference.view(torch.uint8)
        )
        if self.mismatches is None:
            self.mismatches = mismatch
        else:
            self.mismatches += mismatch
        self.count += 1

    def mismatch_count(self) -> int:
        if self.mismatches is None:
            return 0
        return int(self.mismatches.item())


def _numeric_sample(
    candidate: Any, reference: Any, rtol: float, atol: float
) -> tuple[Any, Any, Any, Any]:
    import torch

    candidate_float = candidate.float()
    reference_float = reference.float()
    candidate_finite = torch.isfinite(candidate_float)
    reference_finite = torch.isfinite(reference_float)
    both_finite = candidate_finite & reference_finite
    same_infinity = (
        torch.isinf(candidate_float)
        & torch.isinf(reference_float)
        & (candidate_float == reference_float)
    )
    absolute = (candidate_float - reference_float).abs()
    finite_absolute = torch.where(both_finite, absolute, torch.zeros_like(absolute))
    allowed = atol + rtol * reference_float.abs()
    within = (both_finite & (absolute <= allowed)) | same_infinity
    outside = torch.count_nonzero(~within)
    nonfinite = torch.count_nonzero(~(both_finite | same_infinity))
    maximum_absolute = finite_absolute.max()
    denominator = torch.clamp(reference_float.abs(), min=1.0e-12)
    relative = torch.where(
        both_finite,
        finite_absolute / denominator,
        torch.zeros_like(finite_absolute),
    )
    return outside, nonfinite, maximum_absolute, relative.max()


class _NumericStats:
    def __init__(self) -> None:
        self.outside: Any | None = None
        self.nonfinite: Any | None = None
        self.maximum_absolute: Any | None = None
        self.maximum_relative: Any | None = None

    def observe(self, candidate: Any, reference: Any, rtol: float, atol: float) -> None:
        import torch

        outside, nonfinite, maximum_absolute, maximum_relative = _numeric_sample(
            candidate, reference, rtol, atol
        )
        if self.outside is None:
            self.outside = outside
            self.nonfinite = nonfinite
            self.maximum_absolute = maximum_absolute
            self.maximum_relative = maximum_relative
        else:
            self.outside += outside
            self.nonfinite += nonfinite
            self.maximum_absolute = torch.maximum(
                self.maximum_absolute, maximum_absolute
            )
            self.maximum_relative = torch.maximum(
                self.maximum_relative, maximum_relative
            )

    def report(self) -> tuple[int, int, float, float]:
        return (
            int(self.outside.item()),
            int(self.nonfinite.item()),
            float(self.maximum_absolute.item()),
            float(self.maximum_relative.item()),
        )


class _CombineShadowState:
    def __init__(
        self,
        output_tensor: Any,
        lse_tensor: Any,
        tolerances: tuple[float, float, float, float],
    ) -> None:
        self.candidate_output = output_tensor
        self.candidate_lse = lse_tensor
        self.tolerances = tolerances
        self.output_stats = _NumericStats()
        self.lse_stats = _NumericStats()
        self.count = 0
        self.validated = False

    def observe(self, reference_output: Any, reference_lse: Any) -> None:
        output_rtol, output_atol, lse_rtol, lse_atol = self.tolerances
        self.output_stats.observe(
            self.candidate_output,
            reference_output,
            output_rtol,
            output_atol,
        )
        self.lse_stats.observe(
            self.candidate_lse,
            reference_lse,
            lse_rtol,
            lse_atol,
        )
        self.count += 1


class _GraphQueryShadow:
    def __init__(
        self,
        signature: _QuerySignature,
        mismatches: Any,
        replays: Any,
    ) -> None:
        self.signature = signature
        self.mismatches = mismatches
        self.replays = replays

    def capture(self, mismatches: Any) -> None:
        self.mismatches.copy_(mismatches)
        self.replays.add_(1)

    def replay_count(self) -> int:
        return int(self.replays.item())

    def mismatch_count(self) -> int:
        return int(self.mismatches.item())


class _GraphCombineShadow:
    def __init__(
        self,
        signature: _CombineSignature,
        output_integer_sample: tuple[Any, Any],
        output_float_sample: tuple[Any, Any],
        lse_integer_sample: tuple[Any, Any],
        lse_float_sample: tuple[Any, Any],
        replays: Any,
    ) -> None:
        self.signature = signature
        self.output_integer_sample = output_integer_sample
        self.output_float_sample = output_float_sample
        self.lse_integer_sample = lse_integer_sample
        self.lse_float_sample = lse_float_sample
        self.replays = replays

    def capture(
        self,
        output_sample: tuple[Any, Any, Any, Any],
        lse_sample: tuple[Any, Any, Any, Any],
    ) -> None:
        for destination, source in zip(
            self.output_integer_sample, output_sample[:2], strict=True
        ):
            destination.copy_(source)
        for destination, source in zip(
            self.output_float_sample, output_sample[2:], strict=True
        ):
            destination.copy_(source)
        for destination, source in zip(
            self.lse_integer_sample, lse_sample[:2], strict=True
        ):
            destination.copy_(source)
        for destination, source in zip(
            self.lse_float_sample, lse_sample[2:], strict=True
        ):
            destination.copy_(source)
        self.replays.add_(1)

    def replay_count(self) -> int:
        return int(self.replays.item())

    def report(self) -> dict[str, object]:
        output_report = (
            *(int(value.item()) for value in self.output_integer_sample),
            *(float(value.item()) for value in self.output_float_sample),
        )
        lse_report = (
            *(int(value.item()) for value in self.lse_integer_sample),
            *(float(value.item()) for value in self.lse_float_sample),
        )
        return {
            "signature": list(self.signature),
            "replays": self.replay_count(),
            "output_outside": output_report[0],
            "output_nonfinite": output_report[1],
            "output_max_abs": output_report[2],
            "output_max_rel": output_report[3],
            "lse_outside": lse_report[0],
            "lse_nonfinite": lse_report[1],
            "lse_max_abs": lse_report[2],
            "lse_max_rel": lse_report[3],
            "passed": not (
                output_report[0] or output_report[1] or lse_report[0] or lse_report[1]
            ),
        }


class _GraphShadowArena:
    """Persistent CUDA storage for diagnostics emitted by captured graphs."""

    def __init__(
        self,
        torch_module: Any,
        device: Any,
        capacity: int,
    ) -> None:
        self.device = str(device)
        self.capacity = capacity
        self.query_mismatches = torch_module.zeros(
            (capacity,), dtype=torch_module.int64, device=device
        )
        self.query_replays = torch_module.zeros(
            (capacity,), dtype=torch_module.int64, device=device
        )
        self.combine_integers = torch_module.zeros(
            (capacity, 4), dtype=torch_module.int64, device=device
        )
        self.combine_floats = torch_module.zeros(
            (capacity, 4), dtype=torch_module.float32, device=device
        )
        self.combine_replays = torch_module.zeros(
            (capacity,), dtype=torch_module.int64, device=device
        )
        self.query_used = 0
        self.combine_used = 0

    def reserve_query(self, signature: _QuerySignature) -> _GraphQueryShadow:
        if self.query_used >= self.capacity:
            raise RuntimeError(
                "Spark TP4 DCP graph query shadow arena exhausted: "
                f"capacity={self.capacity}"
            )
        index = self.query_used
        self.query_used += 1
        return _GraphQueryShadow(
            signature,
            self.query_mismatches[index],
            self.query_replays[index],
        )

    def reserve_combine(self, signature: _CombineSignature) -> _GraphCombineShadow:
        if self.combine_used >= self.capacity:
            raise RuntimeError(
                "Spark TP4 DCP graph combine shadow arena exhausted: "
                f"capacity={self.capacity}"
            )
        index = self.combine_used
        self.combine_used += 1
        return _GraphCombineShadow(
            signature,
            (
                self.combine_integers[index, 0],
                self.combine_integers[index, 1],
            ),
            (
                self.combine_floats[index, 0],
                self.combine_floats[index, 1],
            ),
            (
                self.combine_integers[index, 2],
                self.combine_integers[index, 3],
            ),
            (
                self.combine_floats[index, 2],
                self.combine_floats[index, 3],
            ),
            self.combine_replays[index],
        )


class _Backend:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._session: _NativeDcpSession | None = None
        self._graph_session: _NativeDcpSession | None = None
        self.disabled = False
        self.graph_disabled = False
        self.query_shadows: dict[_QuerySignature, _QueryShadowState] = {}
        self.combine_shadows: dict[_CombineSignature, _CombineShadowState] = {}
        self.graph_query_shadows: list[_GraphQueryShadow] = []
        self.graph_combine_shadows: list[_GraphCombineShadow] = []
        self.graph_shadow_arena: _GraphShadowArena | None = None

    def session(self) -> _NativeDcpSession | None:
        if self.disabled:
            return None
        if self._session is None:
            try:
                self._session = _NativeDcpSession(self.rank)
            except Exception:
                self.disabled = True
                logger.exception(
                    "disabling Spark TP4 DCP before native enqueue because "
                    "session creation failed"
                )
                return None
        return self._session

    def prepare_graph(self, device: Any | None = None) -> _NativeDcpSession | None:
        if self.graph_disabled:
            return None
        try:
            if _graph_shadow_enabled():
                if device is None:
                    raise RuntimeError(
                        "DCP graph shadow preparation requires a CUDA device"
                    )
                if self.graph_shadow_arena is None:
                    import torch

                    self.graph_shadow_arena = _GraphShadowArena(
                        torch,
                        device,
                        _positive_integer(
                            "SPARK_TP4_DCP_GRAPH_SHADOW_CAPACITY",
                            _DCP_GRAPH_SHADOW_DEFAULT_CAPACITY,
                        ),
                    )
                elif self.graph_shadow_arena.device != str(device):
                    raise RuntimeError(
                        "DCP graph shadow arena changed CUDA devices: "
                        f"{self.graph_shadow_arena.device} -> {device}"
                    )
            if self._graph_session is None:
                self._graph_session = _NativeDcpSession(
                    self.rank,
                    graph_only=True,
                    control_ports=_graph_control_ports(),
                    graph_cpu_affinity=_graph_preflight(),
                )
                _dcp_graph_sessions[self.rank] = self._graph_session
                from spark_graph_status_reporter import (
                    ensure_status_reporter,
                )

                ensure_status_reporter(rank=self.rank)
        except Exception:
            self.graph_disabled = True
            logger.exception(
                "disabling Spark TP4 DCP graph capture before native "
                "enqueue because session or diagnostic arena creation failed"
            )
            return None
        return self._graph_session

    def query_shadow(
        self, signature: _QuerySignature, output_tensor: Any
    ) -> _QueryShadowState:
        state = self.query_shadows.get(signature)
        if state is None:
            state = _QueryShadowState(output_tensor)
            self.query_shadows[signature] = state
        return state

    def combine_shadow(
        self,
        signature: _CombineSignature,
        output_tensor: Any,
        lse_tensor: Any,
    ) -> _CombineShadowState:
        state = self.combine_shadows.get(signature)
        if state is None:
            state = _CombineShadowState(
                output_tensor, lse_tensor, _combine_tolerances()
            )
            self.combine_shadows[signature] = state
        return state


def _new_output(torch_module: Any, input_tensor: Any, q: int) -> Any:
    return torch_module.empty(
        (q, _OUTPUT_HEADS, _QUERY_HEAD_DIM),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


def _new_combine_outputs(
    torch_module: Any,
    output_tensor: Any,
    lse_tensor: Any,
    signature: _CombineSignature,
) -> tuple[Any, Any]:
    q, head_dimension, _, _ = signature
    return (
        torch_module.empty(
            (q, _COMBINE_HEADS_PER_RANK, head_dimension),
            dtype=output_tensor.dtype,
            device=output_tensor.device,
        ),
        torch_module.empty(
            (q, _COMBINE_HEADS_PER_RANK),
            dtype=lse_tensor.dtype,
            device=lse_tensor.device,
        ),
    )


def _backend_for(group: Any) -> _Backend:
    backend = getattr(group, "_spark_tp4_dcp_native", None)
    if backend is None:
        backend = _Backend(int(group.rank_in_group))
        group._spark_tp4_dcp_native = backend
        _dcp_backends[backend.rank] = backend
    return backend


def dcp_graph_status_snapshot() -> dict[int, dict[str, object]]:
    """Return process-local DCP graph replay status."""

    return {
        rank: session.graph_status()
        for rank, session in sorted(_dcp_graph_sessions.items())
    }


def dcp_graph_diagnostic_snapshot() -> dict[str, object]:
    mode = _mode()
    query_enabled, combine_enabled = _validate_family_selection(mode)
    return {
        "family_selection": {
            "mode": mode or "stock",
            "query_enabled": query_enabled,
            "combine_enabled": combine_enabled,
        },
        "sessions": dcp_graph_status_snapshot(),
        "events": dict(sorted(_graph_event_counts.items())),
        "shadow_nodes": {
            rank: {
                "query": len(backend.graph_query_shadows),
                "combine": len(backend.graph_combine_shadows),
            }
            for rank, backend in sorted(_dcp_backends.items())
        },
    }


def dcp_graph_shadow_report() -> dict[str, object]:
    """Synchronize and validate captured DCP shadow outputs after replay."""

    import torch

    torch.cuda.synchronize()
    mode = _mode()
    query_enabled, combine_enabled = _validate_family_selection(mode)
    rank_reports: dict[int, dict[str, object]] = {}
    all_passed = bool(_dcp_backends)
    for rank, backend in sorted(_dcp_backends.items()):
        session = backend._graph_session
        status = None if session is None else session.graph_status()
        captured_query_nodes = len(backend.graph_query_shadows)
        captured_combine_nodes = len(backend.graph_combine_shadows)
        query_mismatches = [
            {
                "q": record.signature,
                "replays": record.replay_count(),
                "byte_mismatches": record.mismatch_count(),
            }
            for record in backend.graph_query_shadows
            if record.replay_count() > 0
        ]
        combine_reports = [
            record.report()
            for record in backend.graph_combine_shadows
            if record.replay_count() > 0
        ]
        replayed_query_nodes = len(query_mismatches)
        replayed_combine_nodes = len(combine_reports)
        contract_passed = bool(
            status is not None
            and status["captured_nodes"]
            == captured_query_nodes + captured_combine_nodes
            and status["captured_query_nodes"] == captured_query_nodes
            and status["captured_combine_nodes"] == captured_combine_nodes
            and status["replay_advanced"]
            and status["replay_caught_up"]
            and not status["fatal"]
        )
        query_correctness_passed = not query_enabled or (
            replayed_query_nodes > 0
            and all(item["byte_mismatches"] == 0 for item in query_mismatches)
        )
        combine_correctness_passed = not combine_enabled or (
            replayed_combine_nodes > 0
            and all(item["passed"] for item in combine_reports)
        )
        correctness_passed = (
            query_correctness_passed and combine_correctness_passed
        )
        passed = contract_passed and correctness_passed
        all_passed = all_passed and passed
        rank_reports[rank] = {
            "status": status,
            "query_nodes": captured_query_nodes,
            "combine_nodes": captured_combine_nodes,
            "replayed_query_nodes": replayed_query_nodes,
            "replayed_combine_nodes": replayed_combine_nodes,
            "unreplayed_query_nodes": (captured_query_nodes - replayed_query_nodes),
            "unreplayed_combine_nodes": (
                captured_combine_nodes - replayed_combine_nodes
            ),
            "query": query_mismatches,
            "combine": combine_reports,
            "contract_passed": contract_passed,
            "correctness_passed": correctness_passed,
            "passed": passed,
        }
    return {
        "family_selection": {
            "mode": mode or "stock",
            "query_enabled": query_enabled,
            "combine_enabled": combine_enabled,
        },
        "ranks": rank_reports,
        "passed": all_passed,
    }


def _audit_capture_state() -> bool:
    from spark_collective_audit import enabled

    if not enabled():
        return False
    import torch

    return _is_stream_capturing(torch)


def _record_stock_path(
    family: str,
    *,
    reason: str,
    group: Any | None = None,
    tensor: Any | None = None,
    seam: str | None = None,
    dim: int | None = None,
) -> None:
    from spark_collective_audit import (
        StockCollectiveSignature,
        classify_stock_family,
        enabled,
        record_stock,
    )

    signature = None
    if enabled() and group is not None and tensor is not None:
        world_size = getattr(group, "world_size", None)
        signature = StockCollectiveSignature(
            shape=_shape(tensor),
            dtype=str(tensor.dtype),
            is_cuda=bool(tensor.is_cuda),
            contiguous=bool(tensor.is_contiguous()),
            world_size=(None if world_size is None else int(world_size)),
            unique_name=str(getattr(group, "unique_name", "")),
        )
        if seam is not None:
            family = classify_stock_family(
                seam,
                signature,
                dim=dim,
            )
    record_stock(
        family,
        capturing=_audit_capture_state(),
        reason=reason,
        signature=signature,
    )


def _call_stock_all_gather(
    original: Any,
    group: Any,
    input_tensor: Any,
    dim: int,
    *,
    reason: str = "original",
) -> Any:
    if (
        getattr(group, "unique_name", None) == _DCP_GROUP
        and int(getattr(group, "world_size", 0)) == _WORLD_SIZE
    ):
        _record_stock_path(
            "dcp_all_gather",
            reason=reason,
            group=group,
            tensor=input_tensor,
            seam="group_all_gather",
            dim=dim,
        )
    return original(group, input_tensor, dim)


def _call_stock_combine(
    original: Any,
    output_tensor: Any,
    lse_tensor: Any,
    group: Any,
    ctx: Any,
    return_lse: bool,
    is_lse_base_on_e: bool,
    *,
    reason: str = "original",
) -> Any:
    _record_stock_path(
        "dcp_combine",
        reason=reason,
        group=group,
        tensor=output_tensor,
    )
    return original(
        output_tensor,
        lse_tensor,
        group,
        ctx=ctx,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


def _combine_result(output_tensor: Any, lse_tensor: Any, return_lse: bool) -> Any:
    return (output_tensor, lse_tensor) if return_lse else output_tensor


def _patch_combine(module: ModuleType) -> None:
    current = module.cp_lse_ag_out_rs
    if getattr(current, "_spark_tp4_dcp_backend", False):
        original = current._spark_original
        replacement = current
        _patch_existing_combine_aliases(original, replacement)
        return
    original = current

    def spark_dcp_combine(
        cp_attn_out: Any,
        cp_attn_lse: Any,
        cp_group: Any,
        ctx: Any = None,
        return_lse: bool = False,
        is_lse_base_on_e: bool = True,
    ) -> Any:
        mode = _mode()
        signature = _combine_signature(
            cp_group,
            cp_attn_out,
            cp_attn_lse,
            return_lse,
            is_lse_base_on_e,
            mode,
        )
        if signature is None:
            return _call_stock_combine(
                original,
                cp_attn_out,
                cp_attn_lse,
                cp_group,
                ctx,
                return_lse,
                is_lse_base_on_e,
            )

        import torch

        execution_phase = _execution_phase(torch, cp_attn_out.device)
        if execution_phase == "capture_warmup" and _graph_enabled(mode):
            backend = _backend_for(cp_group)
            if backend.prepare_graph(cp_attn_out.device) is None:
                logger.critical(
                    "fatal Spark TP4 DCP graph session creation failed "
                    "before shared-stream combine warmup; terminating worker"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            return _call_stock_combine(
                original,
                cp_attn_out,
                cp_attn_lse,
                cp_group,
                ctx,
                return_lse,
                is_lse_base_on_e,
                reason="shared_capture_warmup_reference",
            )

        if execution_phase == "capture":
            if not _graph_enabled(mode):
                return _call_stock_combine(
                    original,
                    cp_attn_out,
                    cp_attn_lse,
                    cp_group,
                    ctx,
                    return_lse,
                    is_lse_base_on_e,
                )
            backend = getattr(cp_group, "_spark_tp4_dcp_native", None)
            graph_session = None if backend is None else backend._graph_session
            if graph_session is None:
                logger.critical(
                    "fatal Spark TP4 DCP graph session is absent during "
                    "combine capture; terminating worker to prevent a "
                    "rank-split collective"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            contiguous_lse = cp_attn_lse.contiguous()
            candidate_output, candidate_lse = _new_combine_outputs(
                torch, cp_attn_out, contiguous_lse, signature
            )
            stream = torch.cuda.current_stream(device=cp_attn_out.device)
            try:
                graph_session.capture_combine(
                    cp_attn_out,
                    contiguous_lse,
                    candidate_output,
                    candidate_lse,
                    signature,
                    stream,
                )
                _record_graph_event(cp_group, "captured_combine_nodes")
                if mode == "custom":
                    return _combine_result(candidate_output, candidate_lse, return_lse)
                reference_output, reference_lse = _call_stock_combine(
                    original,
                    cp_attn_out,
                    cp_attn_lse,
                    cp_group,
                    ctx,
                    True,
                    is_lse_base_on_e,
                )
                (
                    output_rtol,
                    output_atol,
                    lse_rtol,
                    lse_atol,
                ) = _combine_tolerances()
                arena = backend.graph_shadow_arena
                if arena is None:
                    raise RuntimeError(
                        "DCP graph combine shadow arena was not prepared"
                    )
                record = arena.reserve_combine(signature)
                record.capture(
                    _numeric_sample(
                        candidate_output,
                        reference_output,
                        output_rtol,
                        output_atol,
                    ),
                    _numeric_sample(
                        candidate_lse,
                        reference_lse,
                        lse_rtol,
                        lse_atol,
                    ),
                )
                backend.graph_combine_shadows.append(record)
                return _combine_result(reference_output, reference_lse, return_lse)
            except BaseException:
                logger.exception(
                    "fatal Spark TP4 DCP graph combine capture failure; "
                    "terminating worker because a partial native graph "
                    "cannot safely fall back"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")

        if os.getenv("SPARK_TP4_STOCK_TIMING", "0") == "1":
            from spark_tp4_stock_timing import time_original

            stream = torch.cuda.current_stream(device=cp_attn_out.device)
            return time_original(
                "combine",
                signature[0],
                stream,
                lambda: _call_stock_combine(
                    original,
                    cp_attn_out,
                    cp_attn_lse,
                    cp_group,
                    ctx,
                    return_lse,
                    is_lse_base_on_e,
                ),
                torch,
            )

        backend = _backend_for(cp_group)
        if _graph_enabled(mode) and backend.prepare_graph(cp_attn_out.device) is None:
            logger.critical(
                "fatal Spark TP4 DCP graph session creation failed "
                "before combine capture; terminating worker"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")
        session = backend.session()
        if session is None:
            if mode == "custom":
                logger.critical(
                    "fatal Spark TP4 DCP combine session is unavailable while "
                    "the combine family is armed custom; terminating worker "
                    "instead of falling back to stock"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            return _call_stock_combine(
                original,
                cp_attn_out,
                cp_attn_lse,
                cp_group,
                ctx,
                return_lse,
                is_lse_base_on_e,
            )

        shadow_limit = 0
        shadow = None
        promoted = False
        if mode == "shadow":
            shadow_limit = _positive_integer("SPARK_TP4_DCP_SHADOW_COLLECTIVES", 8)
            shadow = backend.combine_shadows.get(signature)
            if shadow is not None:
                promoted = shadow.validated and (
                    os.getenv("SPARK_TP4_DCP_SHADOW_PROMOTE", "0") == "1"
                )
                if not promoted and shadow.count >= shadow_limit:
                    return _call_stock_combine(
                        original,
                        cp_attn_out,
                        cp_attn_lse,
                        cp_group,
                        ctx,
                        return_lse,
                        is_lse_base_on_e,
                    )

        contiguous_lse = cp_attn_lse.contiguous()
        if mode == "custom" or promoted:
            candidate_output, candidate_lse = _new_combine_outputs(
                torch, cp_attn_out, contiguous_lse, signature
            )
        elif shadow is None:
            candidate_output, candidate_lse = _new_combine_outputs(
                torch, cp_attn_out, contiguous_lse, signature
            )
            shadow = backend.combine_shadow(signature, candidate_output, candidate_lse)
        else:
            candidate_output = shadow.candidate_output
            candidate_lse = shadow.candidate_lse

        stream = torch.cuda.current_stream(device=cp_attn_out.device)
        try:
            session.combine(
                cp_attn_out,
                contiguous_lse,
                candidate_output,
                candidate_lse,
                signature,
                stream,
            )
        except BaseException:
            logger.exception(
                "fatal Spark TP4 DCP combine failure; terminating worker "
                "because native enqueue may have poisoned its CUDA stream"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")

        if mode == "custom" or promoted:
            return _combine_result(candidate_output, candidate_lse, return_lse)

        try:
            # Always request LSE during the finite shadow window so both native
            # outputs are checked even when the production caller only needs
            # the reduced attention output.
            reference_output, reference_lse = _call_stock_combine(
                original,
                cp_attn_out,
                cp_attn_lse,
                cp_group,
                ctx,
                True,
                is_lse_base_on_e,
            )
            assert shadow is not None
            shadow.observe(reference_output, reference_lse)
            report = None
            if shadow.count == shadow_limit:
                report = (
                    shadow.output_stats.report(),
                    shadow.lse_stats.report(),
                )
        except BaseException:
            logger.exception(
                "fatal failure after Spark TP4 DCP combine enqueue; terminating worker"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")

        if report is not None:
            q, head_dimension, query_stride, head_stride = signature
            output_report, lse_report = report
            output_rtol, output_atol, lse_rtol, lse_atol = shadow.tolerances
            logger.warning(
                "Spark TP4 DCP combine shadow complete: q=%d dim=%d "
                "strides=(%d,%d,1) "
                "collectives=%d output_outside=%d "
                "output_nonfinite=%d output_max_abs=%g "
                "output_max_rel=%g output_rtol=%g output_atol=%g "
                "lse_outside=%d lse_nonfinite=%d lse_max_abs=%g "
                "lse_max_rel=%g lse_rtol=%g lse_atol=%g",
                q,
                head_dimension,
                query_stride,
                head_stride,
                shadow_limit,
                *output_report,
                output_rtol,
                output_atol,
                *lse_report,
                lse_rtol,
                lse_atol,
            )
            passed = not (
                output_report[0] or output_report[1] or lse_report[0] or lse_report[1]
            )
            shadow.validated = passed
            if not passed:
                logger.error(
                    "Spark TP4 DCP combine Q%d/D%d exceeded shadow tolerances; "
                    "terminating worker",
                    q,
                    head_dimension,
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            if os.getenv("SPARK_TP4_DCP_SHADOW_PROMOTE", "0") == "1":
                logger.warning(
                    "Spark TP4 DCP combine Q%d/D%d will promote to custom "
                    "on its next call",
                    q,
                    head_dimension,
                )
        return _combine_result(reference_output, reference_lse, return_lse)

    spark_dcp_combine._spark_tp4_dcp_backend = True  # type: ignore[attr-defined]
    spark_dcp_combine._spark_original = original  # type: ignore[attr-defined]
    module.cp_lse_ag_out_rs = spark_dcp_combine
    _patch_existing_combine_aliases(original, spark_dcp_combine)


def _patch_existing_combine_aliases(original: Any, replacement: Any) -> None:
    """Repair aliases bound before the defining common module was patched."""

    for loaded in tuple(sys.modules.values()):
        if not isinstance(loaded, ModuleType) or loaded is sys.modules.get(
            _COMBINE_TARGET
        ):
            continue
        if vars(loaded).get("cp_lse_ag_out_rs") is original:
            loaded.cp_lse_ag_out_rs = replacement


class _CombineLoader(importlib.abc.Loader):
    def __init__(self, delegate: importlib.abc.Loader) -> None:
        self._delegate = delegate

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._delegate, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._delegate.exec_module(module)
        _patch_combine(module)


class _CombineFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any, target: ModuleType | None = None
    ) -> Any:
        if fullname != _COMBINE_TARGET:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _CombineLoader(spec.loader)
        return spec


def _install_combine_import_hook() -> None:
    loaded = sys.modules.get(_COMBINE_TARGET)
    if loaded is not None:
        _patch_combine(loaded)
        return
    if not any(isinstance(finder, _CombineFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CombineFinder())


def install() -> None:
    global _installed
    mode = _mode()
    query_enabled, combine_enabled = _validate_family_selection(mode)
    if _installed or not mode:
        return
    _graph_enabled(mode)
    if mode == "shadow":
        _positive_integer("SPARK_TP4_DCP_SHADOW_COLLECTIVES", 8)
        _combine_tolerances()
        if _graph_shadow_enabled():
            _positive_integer(
                "SPARK_TP4_DCP_GRAPH_SHADOW_CAPACITY",
                _DCP_GRAPH_SHADOW_DEFAULT_CAPACITY,
            )

    _install_combine_import_hook()
    from vllm.distributed.parallel_state import GroupCoordinator

    original = GroupCoordinator._all_gather_out_place
    if getattr(original, "_spark_tp4_dcp_backend", False):
        _installed = True
        return

    def spark_dcp_all_gather(self: Any, input_tensor: Any, dim: int) -> Any:
        mode = _mode()
        signature = _query_signature(self, input_tensor, dim, mode)
        if signature is None:
            return _call_stock_all_gather(
                original,
                self,
                input_tensor,
                dim,
            )

        import torch

        execution_phase = _execution_phase(torch, input_tensor.device)
        if execution_phase == "capture_warmup" and _graph_enabled(mode):
            backend = _backend_for(self)
            if backend.prepare_graph(input_tensor.device) is None:
                logger.critical(
                    "fatal Spark TP4 DCP graph session creation failed "
                    "before shared-stream query warmup; terminating worker"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            return _call_stock_all_gather(
                original,
                self,
                input_tensor,
                dim,
                reason="shared_capture_warmup_reference",
            )

        if execution_phase == "capture":
            if not _graph_enabled(mode):
                return _call_stock_all_gather(
                    original,
                    self,
                    input_tensor,
                    dim,
                )
            backend = getattr(self, "_spark_tp4_dcp_native", None)
            graph_session = None if backend is None else backend._graph_session
            if graph_session is None:
                logger.critical(
                    "fatal Spark TP4 DCP graph session is absent during "
                    "query capture; terminating worker to prevent a "
                    "rank-split collective"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            candidate = _new_output(torch, input_tensor, signature)
            stream = torch.cuda.current_stream(device=input_tensor.device)
            try:
                graph_session.capture_query_all_gather(
                    input_tensor, candidate, signature, stream
                )
                _record_graph_event(self, "captured_query_nodes")
                if mode == "custom":
                    return candidate
                reference = _call_stock_all_gather(
                    original,
                    self,
                    input_tensor,
                    dim,
                )
                arena = backend.graph_shadow_arena
                if arena is None:
                    raise RuntimeError("DCP graph query shadow arena was not prepared")
                record = arena.reserve_query(signature)
                record.capture(
                    torch.count_nonzero(
                        candidate.view(torch.uint8) != reference.view(torch.uint8)
                    )
                )
                backend.graph_query_shadows.append(record)
                return reference
            except BaseException:
                logger.exception(
                    "fatal Spark TP4 DCP graph query capture failure; "
                    "terminating worker because a partial native graph "
                    "cannot safely fall back"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")

        if os.getenv("SPARK_TP4_STOCK_TIMING", "0") == "1":
            from spark_tp4_stock_timing import time_original

            stream = torch.cuda.current_stream(device=input_tensor.device)
            return time_original(
                "query",
                signature,
                stream,
                lambda: _call_stock_all_gather(
                    original,
                    self,
                    input_tensor,
                    dim,
                ),
                torch,
            )

        backend = _backend_for(self)
        if _graph_enabled(mode) and backend.prepare_graph(input_tensor.device) is None:
            logger.critical(
                "fatal Spark TP4 DCP graph session creation failed "
                "before query capture; terminating worker"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")
        session = backend.session()
        if session is None:
            if mode == "custom":
                logger.critical(
                    "fatal Spark TP4 DCP query session is unavailable while "
                    "the query family is armed custom; terminating worker "
                    "instead of falling back to stock"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            return _call_stock_all_gather(
                original,
                self,
                input_tensor,
                dim,
            )

        shadow_limit = 0
        shadow = None
        promoted = False
        if mode == "shadow":
            shadow_limit = _positive_integer("SPARK_TP4_DCP_SHADOW_COLLECTIVES", 8)
            shadow = backend.query_shadows.get(signature)
            if shadow is None:
                template = _new_output(torch, input_tensor, signature)
                shadow = backend.query_shadow(signature, template)
            promoted = shadow.validated and (
                os.getenv("SPARK_TP4_DCP_SHADOW_PROMOTE", "0") == "1"
            )
            if not promoted and shadow.count >= shadow_limit:
                return _call_stock_all_gather(
                    original,
                    self,
                    input_tensor,
                    dim,
                )

        candidate = (
            _new_output(torch, input_tensor, signature)
            if mode == "custom" or promoted
            else shadow.candidate
        )
        stream = torch.cuda.current_stream(device=input_tensor.device)
        try:
            session.query_all_gather(input_tensor, candidate, signature, stream)
        except BaseException:
            logger.exception(
                "fatal Spark TP4 DCP query failure; terminating worker "
                "because native enqueue may have poisoned its CUDA stream"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")

        if mode == "custom" or promoted:
            return candidate

        try:
            reference = _call_stock_all_gather(
                original,
                self,
                input_tensor,
                dim,
            )
            assert shadow is not None
            shadow.observe(reference)
        except BaseException:
            logger.exception(
                "fatal failure after Spark TP4 DCP native enqueue; terminating worker"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")

        if shadow.count == shadow_limit:
            mismatches = shadow.mismatch_count()
            logger.warning(
                "Spark TP4 DCP query shadow complete: q=%d input_bytes=%d "
                "collectives=%d byte_mismatches=%d",
                signature,
                signature * _QUERY_HEADS_PER_RANK * _QUERY_HEAD_DIM * _BF16_BYTES,
                shadow_limit,
                mismatches,
            )
            if mismatches:
                raise RuntimeError("Spark TP4 DCP query shadow found byte mismatches")
            shadow.validated = True
            if os.getenv("SPARK_TP4_DCP_SHADOW_PROMOTE", "0") == "1":
                logger.warning(
                    "Spark TP4 DCP query Q%d will promote to custom on its next call",
                    signature,
                )
        return reference

    spark_dcp_all_gather._spark_tp4_dcp_backend = True  # type: ignore[attr-defined]
    spark_dcp_all_gather._spark_original = original  # type: ignore[attr-defined]
    GroupCoordinator._all_gather_out_place = spark_dcp_all_gather
    _installed = True
    logger.warning(
        "installed Spark TP4 DCP backend in %s mode: query=%s combine=%s",
        _mode(),
        query_enabled,
        combine_enabled,
    )
