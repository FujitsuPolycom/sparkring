"""Exact-signature vLLM adapter for direct-cable TP4 all-gather."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_installed = False
_trace_lock = threading.Lock()
_trace_counts: dict[tuple[Any, ...], int] = defaultdict(int)
_indexer_graph_sessions: dict[int, "_NativeIndexerGraphSession"] = {}
_indexer_graph_event_counts: dict[str, int] = {}
_capture_decision_seen: set[tuple[Any, ...]] = set()
_capture_decision_index: dict[tuple[Any, ...], int] = {}
_capture_decision_records: list[dict[str, Any]] = []
_capture_decision_dropped = 0
_VALID_MODES = {"shadow", "custom"}
_INDEXER_MAX_Q = 40
_INDEXER_BYTES_PER_Q = 2 * 2048 * 4
_INDEXER_GRAPH_DEFAULT_PROGRESS_CPU = 14
_GRAPH_STATUS_CAPTURE_CONFIGURED = 1 << 0
_GRAPH_STATUS_POLLING_ENABLED = 1 << 1
_GRAPH_STATUS_HOST_NATIVE_ATOMICS = 1 << 2
_GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED = 1 << 3
_GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED = 1 << 4
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

# These are the exact single-stream decode signatures measured in the GLM-5.2
# K0 and MTP traces. Shape and dtype are part of the key so a same-sized
# unrelated collective cannot silently enter the custom transport. Existing
# port slots are stable; every added signature gets its own deterministic slot
# and therefore its own native control ports.
_Signature = tuple[int, int, str]
_SUPPORTED_SIGNATURES = {
    ((1, 2, 2048), "torch.int32"): (16384, 0, "indexer"),
    ((1, 38720), "torch.bfloat16"): (77440, 1, "vocab"),
    ((753664,), "torch.uint8"): (753664, 2, "ckv"),
    ((2, 2, 2048), "torch.int32"): (32768, 3, "indexer-k1"),
    ((3, 2, 2048), "torch.int32"): (49152, 4, "indexer-k2"),
    ((4, 2, 2048), "torch.int32"): (65536, 5, "indexer-k3"),
    ((5, 2, 2048), "torch.int32"): (81920, 6, "indexer-k4"),
    ((23552,), "torch.uint8"): (23552, 7, "ckv-prefill"),
}


def _abort_after_native_failure() -> None:
    """Terminate a worker whose CUDA stream may contain an unfulfillable wait."""
    os._exit(71)


def _trace_all_gather(
    communicator: Any, input_tensor: Any, output_tensor: Any
) -> None:
    if os.getenv("SPARK_TP4_TRACE_ALLGATHER_SHAPES", "0") != "1":
        return

    input_shape = tuple(int(value) for value in input_tensor.shape)
    output_shape = tuple(int(value) for value in output_tensor.shape)
    key = (
        getattr(communicator, "unique_name", ""),
        int(getattr(communicator, "world_size", 0)),
        input_shape,
        output_shape,
        str(input_tensor.dtype),
        str(output_tensor.dtype),
    )
    with _trace_lock:
        _trace_counts[key] += 1
        count = _trace_counts[key]
        if count != 1 and count & (count - 1):
            return
        output_path = Path(
            os.getenv(
                "SPARK_TP4_ALLGATHER_TRACE_PATH",
                f"/tmp/spark-allgather-rank{os.getenv('RANK', 'unknown')}.jsonl",
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "unix_ns": time.time_ns(),
            "pid": os.getpid(),
            "rank": os.getenv("RANK", os.getenv("LOCAL_RANK", "unknown")),
            "group": key[0],
            "world_size": key[1],
            "input_shape": input_shape,
            "output_shape": output_shape,
            "input_dtype": key[4],
            "output_dtype": key[5],
            "input_elements": int(input_tensor.numel()),
            "output_elements": int(output_tensor.numel()),
            "count": count,
        }
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


class _NativeAllgatherConfig(ctypes.Structure):
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
        ("input_bytes", ctypes.c_size_t),
    ]


class _NativeIndexerGraphConfig(ctypes.Structure):
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


class _NativeIndexerGraphStatus(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("captured_nodes", ctypes.c_uint64),
        ("captured_q_mask", ctypes.c_uint64),
        ("published_sequence", ctypes.c_uint64),
        ("consumed_sequence", ctypes.c_uint64),
        ("completed_sequence", ctypes.c_uint64),
        ("overflow_sequence", ctypes.c_uint64),
        ("graph_submit_cpu_plus_one", ctypes.c_uint32),
        ("graph_progress_cpu_plus_one", ctypes.c_uint32),
    ]


def _mode() -> str:
    mode = os.getenv("VLLM_SPARK_TP4_ALLGATHER_MODE", "").lower()
    if mode and mode not in _VALID_MODES:
        raise ValueError(
            "VLLM_SPARK_TP4_ALLGATHER_MODE must be 'shadow', 'custom', "
            "or unset"
        )
    return mode


def _indexer_graph_custom_enabled(mode: str) -> bool:
    value = os.getenv("VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM", "0")
    if value not in {"0", "1"}:
        raise ValueError(
            "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM must be '0' or '1'"
        )
    enabled = value == "1"
    if enabled and mode != "custom":
        raise ValueError(
            "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM requires all-gather "
            "custom mode"
        )
    return enabled


def _validate_control_ports(ports: tuple[int, int]) -> None:
    port0, port1 = ports
    if not (0 < port0 <= 65535 and 0 < port1 <= 65535):
        raise ValueError(
            "Spark TP4 indexer graph control ports must be in [1,65535]"
        )
    if port0 == port1:
        raise ValueError(
            "Spark TP4 indexer graph control ports must be distinct"
        )


def _indexer_graph_control_ports() -> tuple[int, int]:
    ports = (
        int(
            os.getenv(
                "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT0", "9462"
            )
        ),
        int(
            os.getenv(
                "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT1", "9463"
            )
        ),
    )
    _validate_control_ports(ports)

    # Exact eager signatures retain their existing namespace. The graph
    # family gets one audited pair, never one pair per Q.
    base = int(os.getenv("SPARK_TP4_ALLGATHER_BASE_PORT", "9490"))
    exact_ports = {
        port
        for _, slot, _ in _SUPPORTED_SIGNATURES.values()
        for port in (base + slot * 10, base + slot * 10 + 1)
    }
    allreduce_base0 = int(
        os.getenv("SPARK_TP4_CONTROL_PORT0", "9480")
    )
    allreduce_base1 = int(
        os.getenv("SPARK_TP4_CONTROL_PORT1", "9481")
    )
    allreduce_ports = {
        port
        for slot in range(512)
        for port in (
            allreduce_base0 + slot * 10,
            allreduce_base1 + slot * 10,
        )
        if port <= 65535
    }
    static_ports = {
        int(os.getenv("SPARK_TP4_GRAPH_CONTROL_PORT0", "9970")),
        int(os.getenv("SPARK_TP4_GRAPH_CONTROL_PORT1", "9971")),
        int(os.getenv("SPARK_TP4_DCP_CONTROL_PORT0", "9890")),
        int(os.getenv("SPARK_TP4_DCP_CONTROL_PORT1", "9891")),
        int(os.getenv("SPARK_TP4_GRAPH_DCP_CONTROL_PORT0", "9892")),
        int(os.getenv("SPARK_TP4_GRAPH_DCP_CONTROL_PORT1", "9893")),
        int(os.getenv("SPARK_TP4_VOCAB_CONTROL_PORT0", "9990")),
        int(os.getenv("SPARK_TP4_VOCAB_CONTROL_PORT1", "9991")),
        int(
            os.getenv(
                "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0", "10110"
            )
        ),
        int(
            os.getenv(
                "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1", "10111"
            )
        ),
    }
    reserved_ports = exact_ports | allreduce_ports | static_ports
    if set(ports) & reserved_ports:
        raise ValueError(
            "Spark TP4 indexer graph ports collide with a reserved "
            "transport namespace"
        )
    return ports


def _indexer_graph_preflight() -> tuple[int, int]:
    from spark_tp4_backend import _graph_preflight as tp_graph_preflight

    submit_cpu, tp_progress_cpu = tp_graph_preflight()
    progress_cpu = int(
        os.getenv(
            "SPARK_TP4_GRAPH_INDEXER_PROGRESS_CPU",
            str(_INDEXER_GRAPH_DEFAULT_PROGRESS_CPU),
        )
    )
    occupied = {submit_cpu, tp_progress_cpu}
    if os.getenv("VLLM_SPARK_TP4_GRAPH_Q1", "0") == "1":
        occupied.add(
            int(os.getenv("SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU", "12"))
        )
    if (
        os.getenv("VLLM_SPARK_TP4_DCP_GRAPH_SHADOW", "0") == "1"
        or os.getenv("VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM", "0") == "1"
    ):
        occupied.add(
            int(os.getenv("SPARK_TP4_GRAPH_DCP_PROGRESS_CPU", "13"))
        )
    if progress_cpu < 0 or progress_cpu in occupied:
        raise RuntimeError(
            "Spark TP4 indexer progress CPU must be nonnegative and "
            "distinct from other graph submit/progress CPUs"
        )
    return submit_cpu, progress_cpu


def _record_indexer_graph_event(
    communicator: Any, event: str
) -> int:
    attribute = f"_spark_tp4_indexer_graph_{event}"
    count = int(getattr(communicator, attribute, 0)) + 1
    setattr(communicator, attribute, count)
    _indexer_graph_event_counts[event] = (
        _indexer_graph_event_counts.get(event, 0) + 1
    )
    return count


def _protocol_trace(
    *,
    rank: int,
    signature: str,
    input_bytes: int,
    stream_handle: int,
    call: int,
    state: str,
    result: int | None = None,
) -> None:
    if (
        os.getenv("SPARK_TP4_PROTOCOL_TRACE", "0") != "1"
        or call < 1
        or call > 2
    ):
        return
    result_field = "" if result is None else f" result={result}"
    print(
        "TP4_AG_PROTOCOL_PY"
        f" rank={rank}"
        f" signature={signature}"
        f" input_bytes={input_bytes}"
        f" stream={stream_handle}"
        f" call={call}"
        f" state={state}"
        f"{result_field}",
        file=sys.stderr,
        flush=True,
    )


def _stream_capture_states(
    torch_module: Any, stream: Any | None = None
) -> tuple[bool, bool]:
    checker = getattr(torch_module.cuda, "is_current_stream_capturing", None)
    if checker is None:
        return False, False
    current_capturing = bool(checker())
    if stream is None:
        return current_capturing, False
    if current_capturing:
        # Avoid switching streams while a global capture is active. vLLM's
        # shared-capture contract permits only one active capture stream per
        # worker, so handle identity safely distinguishes the explicit stream.
        current_stream = getattr(torch_module.cuda, "current_stream", None)
        if current_stream is None:
            return True, False
        explicit_handle = getattr(stream, "cuda_stream", None)
        if explicit_handle is None:
            return True, False
        current = current_stream()
        explicit_capturing = int(current.cuda_stream) == int(
            explicit_handle
        )
        return True, explicit_capturing
    stream_context = getattr(torch_module.cuda, "stream", None)
    if stream_context is None:
        # An explicit stream whose state cannot be inspected is unsafe for the
        # eager spin-wait kernel. Route it through the stock graph-capable path.
        return False, True
    with stream_context(stream):
        return False, bool(checker())


def _is_stream_capturing(
    torch_module: Any, stream: Any | None = None
) -> bool:
    current_capturing, explicit_capturing = _stream_capture_states(
        torch_module, stream
    )
    return current_capturing or explicit_capturing


def _capture_decision_limit() -> int:
    raw = os.getenv("SPARK_TP4_CAPTURE_DECISION_TRACE_LIMIT", "64")
    try:
        limit = int(raw)
    except ValueError as error:
        raise ValueError(
            "SPARK_TP4_CAPTURE_DECISION_TRACE_LIMIT must be an integer "
            "in [1, 4096]"
        ) from error
    if not 1 <= limit <= 4096:
        raise ValueError(
            "SPARK_TP4_CAPTURE_DECISION_TRACE_LIMIT must be an integer "
            "in [1, 4096]"
        )
    return limit


def _capture_state_name(
    current_capturing: bool, explicit_capturing: bool
) -> str:
    if current_capturing and explicit_capturing:
        return "current+explicit"
    if current_capturing:
        return "current"
    if explicit_capturing:
        return "explicit"
    return "neither"


def _capture_decision_signature(
    communicator: Any,
    input_tensor: Any,
    output_tensor: Any,
    graph_q: int | None,
) -> tuple[Any, ...]:
    return (
        int(getattr(communicator, "world_size", 0)),
        tuple(int(value) for value in input_tensor.shape),
        str(input_tensor.dtype),
        tuple(int(value) for value in output_tensor.shape),
        str(output_tensor.dtype),
        graph_q,
    )


def _record_capture_decision(
    *,
    communicator: Any,
    input_tensor: Any,
    output_tensor: Any,
    graph_q: int | None,
    current_capturing: bool,
    explicit_capturing: bool,
    route: str,
    reason: str,
) -> None:
    global _capture_decision_dropped
    if os.getenv("SPARK_TP4_CAPTURE_DECISION_TRACE", "0") != "1":
        return
    if route not in {"stock", "custom"}:
        raise ValueError("capture decision route must be stock or custom")

    signature = _capture_decision_signature(
        communicator, input_tensor, output_tensor, graph_q
    )
    state = _capture_state_name(
        current_capturing, explicit_capturing
    )
    key = signature + (state, route)
    record = {
        "rank": int(getattr(communicator, "rank", -1)),
        "world_size": signature[0],
        "input_shape": list(signature[1]),
        "input_dtype": signature[2],
        "output_shape": list(signature[3]),
        "output_dtype": signature[4],
        "graph_q": graph_q,
        "current_capture": current_capturing,
        "explicit_capture": explicit_capturing,
        "capture_state": state,
        "route": route,
        "reason": reason,
        "count": 1,
    }
    with _trace_lock:
        if key in _capture_decision_seen:
            index = _capture_decision_index.get(key)
            if index is not None:
                _capture_decision_records[index]["count"] += 1
            return
        _capture_decision_seen.add(key)
        if len(_capture_decision_records) >= _capture_decision_limit():
            _capture_decision_dropped += 1
            return
        _capture_decision_index[key] = len(_capture_decision_records)
        _capture_decision_records.append(record)

    print(
        "TP4_AG_CAPTURE_DECISION "
        + json.dumps(record, separators=(",", ":"), sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def capture_decision_snapshot() -> dict[str, Any]:
    with _trace_lock:
        return {
            "records": [
                dict(record) for record in _capture_decision_records
            ],
            "dropped": _capture_decision_dropped,
        }


def _reset_capture_decisions_for_tests() -> None:
    global _capture_decision_dropped
    with _trace_lock:
        _capture_decision_seen.clear()
        _capture_decision_index.clear()
        _capture_decision_records.clear()
        _capture_decision_dropped = 0


def _capture_stream_is_current(
    torch_module: Any, input_tensor: Any, stream: Any
) -> bool:
    current_stream = getattr(torch_module.cuda, "current_stream", None)
    if current_stream is None:
        return False
    explicit_handle = getattr(stream, "cuda_stream", None)
    if explicit_handle is None:
        return False
    current = current_stream(device=input_tensor.device)
    return int(current.cuda_stream) == int(explicit_handle)


def _record_stock_path(
    *,
    capturing: bool,
    reason: str,
    communicator: Any | None = None,
    tensor: Any | None = None,
) -> None:
    from spark_collective_audit import (
        StockCollectiveSignature,
        classify_stock_family,
        enabled,
        record_stock,
    )

    signature = None
    family = "pynccl_all_gather"
    if enabled() and communicator is not None and tensor is not None:
        world_size = getattr(communicator, "world_size", None)
        signature = StockCollectiveSignature(
            shape=tuple(int(value) for value in tensor.shape),
            dtype=str(tensor.dtype),
            is_cuda=bool(tensor.is_cuda),
            contiguous=bool(tensor.is_contiguous()),
            world_size=(
                None if world_size is None else int(world_size)
            ),
            unique_name=str(getattr(communicator, "unique_name", "")),
        )
        family = classify_stock_family(
            "pynccl_all_gather",
            signature,
        )
    record_stock(
        family,
        capturing=capturing,
        reason=reason,
        signature=signature,
    )


def _signature(
    communicator: Any, input_tensor: Any, output_tensor: Any, mode: str
) -> _Signature | None:
    if (
        mode not in _VALID_MODES
        or getattr(communicator, "world_size", None) != 4
        or getattr(communicator, "unique_name", "") != "tp:0"
        or bool(getattr(communicator, "disabled", False))
        or not bool(input_tensor.is_cuda)
        or not bool(output_tensor.is_cuda)
        or not bool(input_tensor.is_contiguous())
        or not bool(output_tensor.is_contiguous())
        or str(input_tensor.dtype) != str(output_tensor.dtype)
        or int(output_tensor.numel()) != 4 * int(input_tensor.numel())
    ):
        return None
    signature = _SUPPORTED_SIGNATURES.get(
        (tuple(int(value) for value in input_tensor.shape), str(input_tensor.dtype))
    )
    if (
        signature is not None
        and signature[2].startswith("ckv")
        and os.getenv("SPARK_TP4_ALLGATHER_ENABLE_CKV", "0") != "1"
    ):
        # These layouts pass short byte-equality shadows, but the 753,664-byte
        # path has not passed sustained long-prefill sequencing. Fixed-K4 DCP
        # decode does not use this full-CKV path, so fence it independently.
        return None
    return signature


def _indexer_graph_q(
    communicator: Any,
    input_tensor: Any,
    output_tensor: Any,
    mode: str,
) -> int | None:
    shape = tuple(int(value) for value in input_tensor.shape)
    if (
        mode not in _VALID_MODES
        or getattr(communicator, "world_size", None) != 4
        or getattr(communicator, "unique_name", "") != "tp:0"
        or bool(getattr(communicator, "disabled", False))
        or len(shape) != 3
        or shape[0] not in range(1, _INDEXER_MAX_Q + 1)
        or shape[1:] != (2, 2048)
        or str(input_tensor.dtype) != "torch.int32"
        or str(output_tensor.dtype) != "torch.int32"
        or not bool(input_tensor.is_cuda)
        or not bool(output_tensor.is_cuda)
        or not bool(input_tensor.is_contiguous())
        or not bool(output_tensor.is_contiguous())
        or int(output_tensor.numel()) != 4 * int(input_tensor.numel())
    ):
        return None
    return shape[0]


class _NativeAllgatherSession:
    def __init__(self, rank: int, input_bytes: int, port_slot: int) -> None:
        if rank not in _DEFAULT_PEERS:
            raise ValueError(f"TP4 rank must be in [0, 3], got {rank}")
        library_path = os.environ["SPARK_TP4_LIBRARY"]
        self._library = ctypes.CDLL(library_path)
        self._library.spark_tp4_allgather_create.argtypes = [
            ctypes.POINTER(_NativeAllgatherConfig),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_allgather_create.restype = ctypes.c_void_p
        self._library.spark_tp4_allgather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_allgather.restype = ctypes.c_int
        self._library.spark_tp4_allgather_destroy.argtypes = [ctypes.c_void_p]
        self._library.spark_tp4_allgather_destroy.restype = None
        self.rank = rank
        self.input_bytes = input_bytes
        self.signature_name = "unknown"
        self._protocol_trace_calls = 0

        default_peer0, default_peer1 = _DEFAULT_PEERS[rank]
        base_port = int(os.getenv("SPARK_TP4_ALLGATHER_BASE_PORT", "9490"))
        port0 = base_port + port_slot * 10
        config = _NativeAllgatherConfig(
            rank=rank,
            peer0=os.getenv("SPARK_TP4_PEER0", default_peer0).encode(),
            peer1=os.getenv("SPARK_TP4_PEER1", default_peer1).encode(),
            device0=os.getenv("SPARK_TP4_DEVICE0", "rocep1s0f0").encode(),
            device1=os.getenv("SPARK_TP4_DEVICE1", "rocep1s0f1").encode(),
            gid0=int(os.getenv("SPARK_TP4_GID0", "3")),
            gid1=int(os.getenv("SPARK_TP4_GID1", "3")),
            control_port0=port0,
            control_port1=port0 + 1,
            input_bytes=input_bytes,
        )
        error = ctypes.create_string_buffer(512)
        self._handle = self._library.spark_tp4_allgather_create(
            ctypes.byref(config), error, len(error)
        )
        if not self._handle:
            message = error.value.decode(errors="replace")
            raise RuntimeError(
                f"failed to create Spark TP4 all-gather session: {message}"
            )
        logger.warning(
            "Spark TP4 all-gather session ready: rank=%d bytes=%d ports=%d/%d",
            rank,
            input_bytes,
            port0,
            port0 + 1,
        )

    def all_gather(
        self, input_tensor: Any, output_tensor: Any, stream: Any
    ) -> None:
        self._protocol_trace_calls += 1
        call = self._protocol_trace_calls
        stream_handle = int(stream.cuda_stream)
        _protocol_trace(
            rank=self.rank,
            signature=self.signature_name,
            input_bytes=self.input_bytes,
            stream_handle=stream_handle,
            call=call,
            state="before-native-call",
        )
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_allgather(
            self._handle,
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.c_void_p(output_tensor.data_ptr()),
            ctypes.c_void_p(stream_handle),
            error,
            len(error),
        )
        _protocol_trace(
            rank=self.rank,
            signature=self.signature_name,
            input_bytes=self.input_bytes,
            stream_handle=stream_handle,
            call=call,
            state="after-native-call",
            result=result,
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 all-gather failed: {message}")


class _NativeIndexerGraphSession:
    def __init__(
        self,
        rank: int,
        control_ports: tuple[int, int],
        graph_cpu_affinity: tuple[int, int],
    ) -> None:
        if rank not in _DEFAULT_PEERS:
            raise ValueError(f"TP4 rank must be in [0,3], got {rank}")
        _validate_control_ports(control_ports)
        submit_cpu, progress_cpu = graph_cpu_affinity
        if (
            submit_cpu < 0
            or progress_cpu < 0
            or submit_cpu == progress_cpu
        ):
            raise ValueError(
                "Spark TP4 indexer graph requires distinct CPU indexes"
            )
        self._library = ctypes.CDLL(os.environ["SPARK_TP4_LIBRARY"])
        self._library.spark_tp4_indexer_graph_create.argtypes = [
            ctypes.POINTER(_NativeIndexerGraphConfig),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_indexer_graph_create.restype = (
            ctypes.c_void_p
        )
        self._library.spark_tp4_indexer_capture_allgather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_indexer_capture_allgather.restype = (
            ctypes.c_int
        )
        self._library.spark_tp4_indexer_get_graph_status.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NativeIndexerGraphStatus),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_indexer_get_graph_status.restype = (
            ctypes.c_int
        )
        self._library.spark_tp4_indexer_graph_destroy.argtypes = [
            ctypes.c_void_p
        ]
        self._library.spark_tp4_indexer_graph_destroy.restype = None

        default_peer0, default_peer1 = _DEFAULT_PEERS[rank]
        config = _NativeIndexerGraphConfig(
            rank=rank,
            peer0=os.getenv(
                "SPARK_TP4_PEER0", default_peer0
            ).encode(),
            peer1=os.getenv(
                "SPARK_TP4_PEER1", default_peer1
            ).encode(),
            device0=os.getenv(
                "SPARK_TP4_DEVICE0", "rocep1s0f0"
            ).encode(),
            device1=os.getenv(
                "SPARK_TP4_DEVICE1", "rocep1s0f1"
            ).encode(),
            gid0=int(os.getenv("SPARK_TP4_GID0", "3")),
            gid1=int(os.getenv("SPARK_TP4_GID1", "3")),
            control_port0=control_ports[0],
            control_port1=control_ports[1],
            graph_submit_cpu_plus_one=submit_cpu + 1,
            graph_progress_cpu_plus_one=progress_cpu + 1,
        )
        error = ctypes.create_string_buffer(512)
        self._handle = self._library.spark_tp4_indexer_graph_create(
            ctypes.byref(config), error, len(error)
        )
        if not self._handle:
            message = error.value.decode(errors="replace")
            raise RuntimeError(
                f"failed to create Spark TP4 indexer graph: {message}"
            )
        logger.warning(
            "Spark TP4 indexer graph session ready: rank=%d ports=%d/%d",
            rank,
            control_ports[0],
            control_ports[1],
        )

    def capture(
        self,
        input_tensor: Any,
        output_tensor: Any,
        q: int,
        stream: Any,
    ) -> None:
        input_shape = tuple(int(value) for value in input_tensor.shape)
        if (
            q not in range(1, _INDEXER_MAX_Q + 1)
            or input_shape != (q, 2, 2048)
            or str(input_tensor.dtype) != "torch.int32"
            or str(output_tensor.dtype) != "torch.int32"
            or not bool(input_tensor.is_cuda)
            or not bool(output_tensor.is_cuda)
            or not bool(input_tensor.is_contiguous())
            or not bool(output_tensor.is_contiguous())
            or int(output_tensor.numel()) != 4 * int(input_tensor.numel())
        ):
            raise ValueError(
                "Spark TP4 indexer graph capture contract mismatch"
            )
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_indexer_capture_allgather(
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
            raise RuntimeError(
                f"Spark TP4 indexer graph capture failed: {message}"
            )

    def graph_status(self) -> dict[str, object]:
        status = _NativeIndexerGraphStatus()
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_indexer_get_graph_status(
            self._handle,
            ctypes.byref(status),
            ctypes.sizeof(status),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(
                f"Spark TP4 indexer graph status failed: {message}"
            )
        if status.struct_size != ctypes.sizeof(_NativeIndexerGraphStatus):
            raise RuntimeError("Spark TP4 indexer graph status ABI mismatch")
        flags = int(status.flags)
        published = int(status.published_sequence)
        consumed = int(status.consumed_sequence)
        completed = int(status.completed_sequence)
        overflow = int(status.overflow_sequence)
        return {
            "captured_nodes": int(status.captured_nodes),
            "captured_q_mask": int(status.captured_q_mask),
            "published_sequence": published,
            "consumed_sequence": consumed,
            "completed_sequence": completed,
            "overflow_sequence": overflow,
            "capture_configured": bool(
                flags & _GRAPH_STATUS_CAPTURE_CONFIGURED
            ),
            "polling_enabled": bool(
                flags & _GRAPH_STATUS_POLLING_ENABLED
            ),
            "host_native_atomics": bool(
                flags & _GRAPH_STATUS_HOST_NATIVE_ATOMICS
            ),
            "submit_affinity_verified": bool(
                flags & _GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED
            ),
            "progress_affinity_verified": bool(
                flags & _GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED
            ),
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
                published > 0
                and published == consumed
                and published == completed
            ),
            "fatal": overflow != 0,
        }


class _ShadowState:
    def __init__(self, output_tensor: Any) -> None:
        import torch

        self.candidate = torch.empty_like(output_tensor)
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


class _Backend:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.sessions: dict[_Signature, _NativeAllgatherSession] = {}
        self.shadows: dict[_Signature, _ShadowState] = {}
        self.session_streams: dict[_Signature, int] = {}
        self.stream_mismatch_logged: set[_Signature] = set()
        self._indexer_graph_session: _NativeIndexerGraphSession | None = None
        self.indexer_graph_disabled = False

    def session(self, signature: _Signature) -> _NativeAllgatherSession:
        session = self.sessions.get(signature)
        if session is None:
            input_bytes, port_slot, _ = signature
            session = _NativeAllgatherSession(
                self.rank, input_bytes, port_slot
            )
            session.signature_name = signature[2]
            self.sessions[signature] = session
        return session

    def session_for_stream(
        self, signature: _Signature, stream: Any
    ) -> _NativeAllgatherSession | None:
        stream_handle = int(getattr(stream, "cuda_stream", id(stream)))
        if signature not in self.session_streams:
            self.session_streams[signature] = stream_handle
        elif self.session_streams[signature] != stream_handle:
            if signature not in self.stream_mismatch_logged:
                logger.warning(
                    "Spark TP4 all-gather signature=%s changed caller CUDA "
                    "stream (%d -> %d); using stock collective before native "
                    "submission",
                    signature[2],
                    self.session_streams[signature],
                    stream_handle,
                )
                self.stream_mismatch_logged.add(signature)
            return None
        return self.session(signature)

    def shadow(
        self, signature: _Signature, output_tensor: Any
    ) -> _ShadowState:
        state = self.shadows.get(signature)
        if state is None:
            state = _ShadowState(output_tensor)
            self.shadows[signature] = state
        return state

    def prepare_indexer_graph(
        self, device: Any
    ) -> _NativeIndexerGraphSession | None:
        del device
        if self.indexer_graph_disabled:
            return None
        if self._indexer_graph_session is None:
            try:
                self._indexer_graph_session = _NativeIndexerGraphSession(
                    self.rank,
                    _indexer_graph_control_ports(),
                    _indexer_graph_preflight(),
                )
                _indexer_graph_sessions[
                    self.rank
                ] = self._indexer_graph_session
            except Exception:
                self.indexer_graph_disabled = True
                logger.exception(
                    "failed to prepare Spark TP4 indexer graph session"
                )
                return None
        return self._indexer_graph_session


def indexer_graph_status_snapshot() -> dict[int, dict[str, object]]:
    return {
        rank: session.graph_status()
        for rank, session in sorted(_indexer_graph_sessions.items())
    }


def indexer_graph_diagnostic_snapshot() -> dict[str, object]:
    return {
        "sessions": indexer_graph_status_snapshot(),
        "events": dict(sorted(_indexer_graph_event_counts.items())),
    }


def install() -> None:
    global _installed
    mode = _mode()
    if _installed or not mode:
        return
    _indexer_graph_custom_enabled(mode)

    from vllm.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )

    original = PyNcclCommunicator.all_gather
    if getattr(original, "_spark_tp4_allgather_backend", False):
        _installed = True
        return

    def spark_all_gather(
        self: Any,
        output_tensor: Any,
        input_tensor: Any,
        stream: Any | None = None,
    ) -> Any:
        mode = _mode()
        import torch

        if stream is None:
            stream = torch.cuda.current_stream(device=input_tensor.device)

        (
            current_capturing,
            explicit_capturing,
        ) = _stream_capture_states(torch, stream)
        capturing = current_capturing or explicit_capturing
        graph_enabled = _indexer_graph_custom_enabled(mode)
        graph_q = _indexer_graph_q(
            self, input_tensor, output_tensor, mode
        )
        if capturing:
            if not graph_enabled or graph_q is None:
                _record_capture_decision(
                    communicator=self,
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    graph_q=graph_q,
                    current_capturing=current_capturing,
                    explicit_capturing=explicit_capturing,
                    route="stock",
                    reason="graph_capture_unsupported",
                )
                _record_stock_path(
                    capturing=True,
                    reason="graph_capture_unsupported",
                    communicator=self,
                    tensor=input_tensor,
                )
                return original(
                    self, output_tensor, input_tensor, stream
                )
            if not _capture_stream_is_current(
                torch, input_tensor, stream
            ):
                logger.critical(
                    "fatal Spark TP4 indexer graph capture stream is not "
                    "the current CUDA stream; terminating worker"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            backend = getattr(
                self, "_spark_tp4_allgather_native", None
            )
            graph_session = (
                None
                if backend is None
                else backend._indexer_graph_session
            )
            if graph_session is None:
                logger.critical(
                    "fatal Spark TP4 indexer graph session is absent "
                    "during capture; terminating worker"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")
            try:
                _record_capture_decision(
                    communicator=self,
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    graph_q=graph_q,
                    current_capturing=current_capturing,
                    explicit_capturing=explicit_capturing,
                    route="custom",
                    reason="graph_native_indexer",
                )
                graph_session.capture(
                    input_tensor, output_tensor, graph_q, stream
                )
                _record_indexer_graph_event(self, "captured_nodes")
                return None
            except BaseException:
                logger.exception(
                    "fatal Spark TP4 indexer graph capture failure; "
                    "terminating worker because partial native capture "
                    "cannot fall back"
                )
                _abort_after_native_failure()
                raise AssertionError("unreachable after worker termination")

        _trace_all_gather(self, input_tensor, output_tensor)
        signature = _signature(
            self, input_tensor, output_tensor, mode
        )
        if signature is None and (graph_q is None or not graph_enabled):
            _record_capture_decision(
                communicator=self,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                graph_q=graph_q,
                current_capturing=current_capturing,
                explicit_capturing=explicit_capturing,
                route="stock",
                reason="ineligible_signature",
            )
            _record_stock_path(
                capturing=False,
                reason="ineligible_signature",
                communicator=self,
                tensor=input_tensor,
            )
            return original(self, output_tensor, input_tensor, stream)

        backend = getattr(self, "_spark_tp4_allgather_native", None)
        if backend is None:
            backend = _Backend(int(self.rank))
            self._spark_tp4_allgather_native = backend
        if (
            graph_enabled
            and graph_q is not None
            and backend.prepare_indexer_graph(
                getattr(input_tensor, "device", None)
            )
            is None
        ):
            logger.critical(
                "fatal Spark TP4 indexer graph preparation failed before "
                "capture; terminating worker"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")
        if signature is None:
            _record_capture_decision(
                communicator=self,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                graph_q=graph_q,
                current_capturing=current_capturing,
                explicit_capturing=explicit_capturing,
                route="stock",
                reason="graph_prepared_eager_reference",
            )
            _record_stock_path(
                capturing=False,
                reason="graph_prepared_eager_reference",
                communicator=self,
                tensor=input_tensor,
            )
            return original(self, output_tensor, input_tensor, stream)
        input_bytes, port_slot, name = signature

        shadow_limit = int(
            os.getenv("SPARK_TP4_ALLGATHER_SHADOW_COLLECTIVES", "8")
        )
        shadow = None
        candidate = output_tensor
        promoted = False
        if mode == "shadow":
            shadow = backend.shadow(signature, output_tensor)
            promoted = shadow.validated and (
                os.getenv("SPARK_TP4_ALLGATHER_SHADOW_PROMOTE", "0") == "1"
            )
            if promoted:
                candidate = output_tensor
            elif shadow.count >= shadow_limit:
                _record_capture_decision(
                    communicator=self,
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    graph_q=graph_q,
                    current_capturing=current_capturing,
                    explicit_capturing=explicit_capturing,
                    route="stock",
                    reason="shadow_reference_only",
                )
                _record_stock_path(
                    capturing=False,
                    reason="shadow_reference_only",
                )
                return original(self, output_tensor, input_tensor, stream)
            else:
                candidate = shadow.candidate

        native_session = backend.session_for_stream(signature, stream)
        if native_session is None:
            _record_capture_decision(
                communicator=self,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                graph_q=graph_q,
                current_capturing=current_capturing,
                explicit_capturing=explicit_capturing,
                route="stock",
                reason="caller_stream_changed",
            )
            _record_stock_path(
                capturing=False,
                reason="caller_stream_changed",
                communicator=self,
                tensor=input_tensor,
            )
            return original(self, output_tensor, input_tensor, stream)

        try:
            _record_capture_decision(
                communicator=self,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                graph_q=graph_q,
                current_capturing=current_capturing,
                explicit_capturing=explicit_capturing,
                route="custom",
                reason="eager_native",
            )
            if os.getenv("SPARK_TP4_FLIGHT_RECORDER", "0") == "1":
                from spark_tp4_flight_recorder import record_collective

                record_collective(
                    f"AG:{name}", stream, input_tensor, candidate
                )
            native_session.all_gather(input_tensor, candidate, stream)
        except BaseException:
            logger.exception(
                "fatal Spark TP4 all-gather error; terminating worker because "
                "its CUDA stream may be poisoned"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")

        if mode == "custom" or promoted:
            return None

        _record_stock_path(
            capturing=False,
            reason="shadow_reference",
        )
        result = original(self, output_tensor, input_tensor, stream)
        assert shadow is not None
        shadow.observe(output_tensor)
        if shadow.count == shadow_limit:
            mismatches = int(shadow.mismatches.item())
            logger.warning(
                "Spark TP4 all-gather shadow complete: signature=%s "
                "bytes=%d collectives=%d byte_mismatches=%d",
                name,
                input_bytes,
                shadow_limit,
                mismatches,
            )
            if mismatches:
                raise RuntimeError(
                    "Spark TP4 all-gather shadow found byte mismatches"
                )
            shadow.validated = True
            if os.getenv("SPARK_TP4_ALLGATHER_SHADOW_PROMOTE", "0") == "1":
                logger.warning(
                    "Spark TP4 all-gather signature=%s will promote to "
                    "custom on its next call",
                    name,
                )
        return result

    spark_all_gather._spark_tp4_allgather_backend = True  # type: ignore[attr-defined]
    spark_all_gather._spark_original = original  # type: ignore[attr-defined]
    PyNcclCommunicator.all_gather = spark_all_gather
    _installed = True
    logger.warning(
        "installed Spark TP4 all-gather backend in %s mode", _mode()
    )
