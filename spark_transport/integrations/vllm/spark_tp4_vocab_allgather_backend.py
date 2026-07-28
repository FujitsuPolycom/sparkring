"""Exact-signature vLLM adapter for Spark TP4 vocabulary all-gather."""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Any

from spark_tp4_query_contract import SUPPORTED_QUERY_ROWS

logger = logging.getLogger(__name__)

_installed = False
_VALID_MODES = {"shadow", "custom"}
_TP_GROUP = "tp:0"
_WORLD_SIZE = 4
_VOCAB_PER_RANK = 38720
_OUTPUT_VOCAB = _VOCAB_PER_RANK * _WORLD_SIZE
_BF16_BYTES = 2
_SUPPORTED_Q = SUPPORTED_QUERY_ROWS
_VOCAB_GRAPH_DEFAULT_PROGRESS_CPU = 12
_GRAPH_STATUS_CAPTURE_CONFIGURED = 1 << 0
_GRAPH_STATUS_POLLING_ENABLED = 1 << 1
_GRAPH_STATUS_HOST_NATIVE_ATOMICS = 1 << 2
_GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED = 1 << 3
_GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED = 1 << 4
_vocab_graph_sessions: dict[int, "_NativeVocabSession"] = {}
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

_Signature = int


def _abort_after_native_failure() -> None:
    """Terminate a worker whose CUDA stream may contain a native wait."""

    os._exit(73)


def _mode() -> str:
    mode = os.getenv("VLLM_SPARK_TP4_VOCAB_MODE", "").lower()
    if mode and mode not in _VALID_MODES:
        raise ValueError(
            "VLLM_SPARK_TP4_VOCAB_MODE must be 'shadow', 'custom', or unset"
        )
    return mode


def _positive_integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _signature(
    group: Any, input_tensor: Any, dim: int, mode: str
) -> _Signature | None:
    shape = _shape(input_tensor)
    normalized_dim = dim + len(shape) if dim < 0 else dim
    if (
        mode not in _VALID_MODES
        or getattr(group, "unique_name", None) != _TP_GROUP
        or int(getattr(group, "world_size", 0)) != _WORLD_SIZE
        or int(getattr(group, "rank_in_group", -1)) not in range(_WORLD_SIZE)
        or normalized_dim != 1
        or len(shape) != 2
        or shape[0] not in _SUPPORTED_Q
        or shape[1] != _VOCAB_PER_RANK
        or str(input_tensor.dtype) != "torch.bfloat16"
        or not bool(input_tensor.is_cuda)
        or not bool(input_tensor.is_contiguous())
    ):
        return None
    return shape[0]


def _is_stream_capturing(torch_module: Any) -> bool:
    checker = getattr(torch_module.cuda, "is_current_stream_capturing", None)
    return bool(checker is not None and checker())


def _record_stock_path(
    *,
    capturing: bool,
    reason: str,
    group: Any | None = None,
    tensor: Any | None = None,
    dim: int | None = None,
) -> None:
    from spark_collective_audit import (
        StockCollectiveSignature,
        classify_stock_family,
        enabled,
        record_stock,
    )

    signature = None
    family = "vocabulary_all_gather"
    if enabled() and group is not None and tensor is not None:
        world_size = getattr(group, "world_size", None)
        signature = StockCollectiveSignature(
            shape=_shape(tensor),
            dtype=str(tensor.dtype),
            is_cuda=bool(tensor.is_cuda),
            contiguous=bool(tensor.is_contiguous()),
            world_size=(
                None if world_size is None else int(world_size)
            ),
            unique_name=str(getattr(group, "unique_name", "")),
        )
        family = classify_stock_family(
            "group_all_gather",
            signature,
            dim=dim,
        )
    record_stock(
        family,
        capturing=capturing,
        reason=reason,
        signature=signature,
    )


def _graph_enabled() -> bool:
    value = os.getenv("VLLM_SPARK_TP4_GRAPH_Q1", "0")
    if value not in {"0", "1"}:
        raise ValueError("VLLM_SPARK_TP4_GRAPH_Q1 must be '0' or '1'")
    return value == "1"


def _graph_preflight() -> tuple[int, int]:
    from spark_tp4_backend import _graph_preflight as tp_graph_preflight

    submit_cpu, tp_progress_cpu = tp_graph_preflight()
    progress_cpu = int(
        os.getenv(
            "SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU",
            str(_VOCAB_GRAPH_DEFAULT_PROGRESS_CPU),
        )
    )
    if progress_cpu < 0:
        raise RuntimeError(
            "Spark TP4 vocabulary graph progress CPU must be nonnegative"
        )
    if progress_cpu in {submit_cpu, tp_progress_cpu}:
        raise RuntimeError(
            "Spark TP4 vocabulary graph progress CPU must differ from "
            "the shared submit and TP progress CPUs"
        )
    return submit_cpu, progress_cpu


def _graph_control_ports() -> tuple[int, int]:
    ports = (
        int(os.getenv("SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0", "10110")),
        int(os.getenv("SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1", "10111")),
    )
    _validate_control_ports(ports)
    return ports


def _validate_control_ports(ports: tuple[int, int]) -> None:
    port0, port1 = ports
    if not (0 < port0 <= 65535 and 0 < port1 <= 65535):
        raise ValueError(
            "Spark TP4 vocabulary control ports must be in [1, 65535]"
        )
    if port0 == port1:
        raise ValueError(
            "Spark TP4 vocabulary control ports must be distinct"
        )


def _record_graph_event(group: Any, event: str) -> int:
    attribute = f"_spark_tp4_vocab_graph_{event}"
    count = int(getattr(group, attribute, 0)) + 1
    setattr(group, attribute, count)
    _graph_event_counts[event] = _graph_event_counts.get(event, 0) + 1
    if count == 1 or count % 128 == 0:
        logger.warning(
            "Spark TP4 vocabulary graph %s on rank %d: count=%d",
            event,
            int(group.rank_in_group),
            count,
        )
    return count


class _NativeVocabConfig(ctypes.Structure):
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


class _NativeVocabGraphConfig(ctypes.Structure):
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


class _NativeVocabGraphStatus(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("captured_nodes", ctypes.c_uint64),
        ("published_sequence", ctypes.c_uint64),
        ("consumed_sequence", ctypes.c_uint64),
        ("completed_sequence", ctypes.c_uint64),
        ("overflow_sequence", ctypes.c_uint64),
        ("graph_submit_cpu_plus_one", ctypes.c_uint32),
        ("graph_progress_cpu_plus_one", ctypes.c_uint32),
    ]


class _NativeVocabSession:
    def __init__(
        self,
        rank: int,
        *,
        graph_only: bool = False,
        control_ports: tuple[int, int] | None = None,
        graph_cpu_affinity: tuple[int, int] | None = None,
    ) -> None:
        if rank not in _DEFAULT_PEERS:
            raise ValueError(f"TP rank must be in [0, 3], got {rank}")
        if graph_only != (graph_cpu_affinity is not None):
            raise ValueError(
                "Spark TP4 vocabulary graph session requires an explicit "
                "CPU pair; eager sessions cannot set one"
            )
        self._library = ctypes.CDLL(os.environ["SPARK_TP4_LIBRARY"])
        self._library.spark_tp4_vocab_allgather_create.argtypes = [
            ctypes.POINTER(_NativeVocabConfig),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_vocab_allgather_create.restype = (
            ctypes.c_void_p
        )
        if graph_only:
            self._library.spark_tp4_vocab_graph_create.argtypes = [
                ctypes.POINTER(_NativeVocabGraphConfig),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_vocab_graph_create.restype = (
                ctypes.c_void_p
            )
        self._library.spark_tp4_vocab_allgather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_vocab_allgather.restype = ctypes.c_int
        if graph_only:
            self._library.spark_tp4_vocab_capture_allgather.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_vocab_capture_allgather.restype = (
                ctypes.c_int
            )
            self._library.spark_tp4_vocab_get_graph_status.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NativeVocabGraphStatus),
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_vocab_get_graph_status.restype = (
                ctypes.c_int
            )
        self._library.spark_tp4_vocab_allgather_destroy.argtypes = [
            ctypes.c_void_p
        ]
        self._library.spark_tp4_vocab_allgather_destroy.restype = None
        self._graph_only = graph_only
        self._capture_stream: int | None = None

        default_peer0, default_peer1 = _DEFAULT_PEERS[rank]
        ports = control_ports or (
            int(os.getenv("SPARK_TP4_VOCAB_CONTROL_PORT0", "9990")),
            int(os.getenv("SPARK_TP4_VOCAB_CONTROL_PORT1", "9991")),
        )
        _validate_control_ports(ports)
        port0, port1 = ports
        submit_cpu, progress_cpu = graph_cpu_affinity or (-1, -1)
        common_config = {
            "rank": rank,
            "peer0": os.getenv(
                "SPARK_TP4_PEER0", default_peer0
            ).encode(),
            "peer1": os.getenv(
                "SPARK_TP4_PEER1", default_peer1
            ).encode(),
            "device0": os.getenv(
                "SPARK_TP4_DEVICE0", "rocep1s0f0"
            ).encode(),
            "device1": os.getenv(
                "SPARK_TP4_DEVICE1", "rocep1s0f1"
            ).encode(),
            "gid0": int(os.getenv("SPARK_TP4_GID0", "3")),
            "gid1": int(os.getenv("SPARK_TP4_GID1", "3")),
            "control_port0": port0,
            "control_port1": port1,
        }
        if graph_only:
            config = _NativeVocabGraphConfig(
                **common_config,
                graph_submit_cpu_plus_one=submit_cpu + 1,
                graph_progress_cpu_plus_one=progress_cpu + 1,
            )
            create = self._library.spark_tp4_vocab_graph_create
        else:
            config = _NativeVocabConfig(**common_config)
            create = self._library.spark_tp4_vocab_allgather_create
        error = ctypes.create_string_buffer(512)
        self._handle = create(
            ctypes.byref(config), error, len(error)
        )
        if not self._handle:
            message = error.value.decode(errors="replace")
            raise RuntimeError(
                "failed to create Spark TP4 vocabulary session: "
                f"{message}"
            )
        logger.warning(
            "Spark TP4 vocabulary %s session ready: rank=%d ports=%d/%d",
            "graph-only" if graph_only else "eager",
            rank,
            port0,
            port1,
        )

    def all_gather(
        self,
        input_tensor: Any,
        output_tensor: Any,
        query_rows: int,
        stream: Any,
    ) -> None:
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_vocab_allgather(
            self._handle,
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.c_void_p(output_tensor.data_ptr()),
            query_rows,
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(
                f"Spark TP4 vocabulary all-gather failed: {message}"
            )

    def capture(
        self,
        input_tensor: Any,
        output_tensor: Any,
        query_rows: int,
        stream: Any,
    ) -> None:
        if not self._graph_only:
            raise RuntimeError(
                "Spark TP4 eager vocabulary session cannot capture"
            )
        input_shape = _shape(input_tensor)
        output_shape = _shape(output_tensor)
        if (
            query_rows not in _SUPPORTED_Q
            or input_shape != (query_rows, _VOCAB_PER_RANK)
            or output_shape != (query_rows, _OUTPUT_VOCAB)
            or str(input_tensor.dtype) != "torch.bfloat16"
            or str(output_tensor.dtype) != "torch.bfloat16"
            or not bool(input_tensor.is_cuda)
            or not bool(output_tensor.is_cuda)
            or not bool(input_tensor.is_contiguous())
            or not bool(output_tensor.is_contiguous())
        ):
            raise ValueError(
                "Spark TP4 vocabulary graph capture requires contiguous "
                "CUDA BF16 [Q,38720] -> [Q,154880], Q in [1,6]"
            )
        stream_pointer = int(stream.cuda_stream)
        if (
            self._capture_stream is not None
            and stream_pointer != self._capture_stream
        ):
            raise ValueError(
                "Spark TP4 vocabulary graph session requires one stable "
                "capture stream"
            )
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_vocab_capture_allgather(
            self._handle,
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.c_void_p(output_tensor.data_ptr()),
            query_rows,
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(
                "Spark TP4 vocabulary graph capture failed: "
                f"{message}"
            )
        self._capture_stream = stream_pointer

    def graph_status(self) -> dict[str, object]:
        if not self._graph_only:
            raise RuntimeError(
                "Spark TP4 eager vocabulary session has no graph status"
            )
        status = _NativeVocabGraphStatus()
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_vocab_get_graph_status(
            self._handle,
            ctypes.byref(status),
            ctypes.sizeof(status),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(
                f"Spark TP4 vocabulary graph status failed: {message}"
            )
        if status.struct_size != ctypes.sizeof(_NativeVocabGraphStatus):
            raise RuntimeError(
                "Spark TP4 vocabulary graph status ABI mismatch: "
                f"native={status.struct_size} python="
                f"{ctypes.sizeof(_NativeVocabGraphStatus)}"
            )
        flags = int(status.flags)
        published = int(status.published_sequence)
        consumed = int(status.consumed_sequence)
        completed = int(status.completed_sequence)
        overflow = int(status.overflow_sequence)
        return {
            "captured_nodes": int(status.captured_nodes),
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


class _Backend:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._session: _NativeVocabSession | None = None
        self._graph_session: _NativeVocabSession | None = None
        self.disabled = False
        self.graph_disabled = False
        self.shadows: dict[_Signature, _ShadowState] = {}

    def session(self) -> _NativeVocabSession | None:
        if self.disabled:
            return None
        if self._session is None:
            try:
                self._session = _NativeVocabSession(self.rank)
            except Exception:
                self.disabled = True
                logger.exception(
                    "disabling Spark TP4 vocabulary before native enqueue "
                    "because session creation failed"
                )
                return None
        return self._session

    def prepare_graph(self) -> _NativeVocabSession | None:
        if self.graph_disabled:
            return None
        if self._graph_session is None:
            try:
                graph_cpu_affinity = _graph_preflight()
                self._graph_session = _NativeVocabSession(
                    self.rank,
                    graph_only=True,
                    control_ports=_graph_control_ports(),
                    graph_cpu_affinity=graph_cpu_affinity,
                )
                _vocab_graph_sessions[self.rank] = self._graph_session
                from spark_graph_status_reporter import (
                    ensure_status_reporter,
                )

                ensure_status_reporter(rank=self.rank)
            except Exception:
                self.graph_disabled = True
                logger.exception(
                    "disabling Spark TP4 vocabulary graph capture before "
                    "native enqueue because session creation failed"
                )
                return None
        return self._graph_session

    def shadow(
        self, signature: _Signature, output_tensor: Any
    ) -> _ShadowState:
        state = self.shadows.get(signature)
        if state is None:
            state = _ShadowState(output_tensor)
            self.shadows[signature] = state
        return state


def _new_output(torch_module: Any, input_tensor: Any, q: int) -> Any:
    return torch_module.empty(
        (q, _OUTPUT_VOCAB),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


def vocab_graph_status_snapshot() -> dict[int, dict[str, object]]:
    """Return process-local vocabulary graph replay status."""
    return {
        rank: session.graph_status()
        for rank, session in sorted(_vocab_graph_sessions.items())
    }


def vocab_graph_diagnostic_snapshot() -> dict[str, object]:
    return {
        "sessions": vocab_graph_status_snapshot(),
        "events": dict(sorted(_graph_event_counts.items())),
    }


def install() -> None:
    global _installed
    mode = _mode()
    if _installed or not mode:
        return
    if mode == "custom":
        _graph_enabled()
    if mode == "shadow":
        _positive_integer("SPARK_TP4_VOCAB_SHADOW_COLLECTIVES", 8)

    from vllm.distributed.parallel_state import GroupCoordinator

    original = GroupCoordinator._all_gather_out_place
    if getattr(original, "_spark_tp4_vocab_backend", False):
        _installed = True
        return

    def spark_vocab_all_gather(
        self: Any, input_tensor: Any, dim: int
    ) -> Any:
        mode = _mode()
        signature = _signature(self, input_tensor, dim, mode)
        if signature is None:
            import torch

            _record_stock_path(
                capturing=_is_stream_capturing(torch),
                reason="ineligible_signature",
                group=self,
                tensor=input_tensor,
                dim=dim,
            )
            return original(self, input_tensor, dim)

        import torch

        capturing = _is_stream_capturing(torch)
        if capturing:
            if mode == "custom" and _graph_enabled():
                backend = getattr(
                    self, "_spark_tp4_vocab_native", None
                )
                graph_session = (
                    None if backend is None else backend._graph_session
                )
                if graph_session is not None:
                    candidate = _new_output(
                        torch, input_tensor, signature
                    )
                    stream = torch.cuda.current_stream(
                        device=input_tensor.device
                    )
                    try:
                        graph_session.capture(
                            input_tensor,
                            candidate,
                            signature,
                            stream,
                        )
                        _record_graph_event(self, "captured_nodes")
                        return candidate
                    except BaseException:
                        logger.exception(
                            "fatal Spark TP4 vocabulary graph-capture "
                            "failure; terminating worker because a partial "
                            "native graph cannot safely fall back"
                        )
                        _abort_after_native_failure()
                        raise AssertionError(
                            "unreachable after worker termination"
                        )
                _record_graph_event(self, "cold_fallbacks")
                logger.critical(
                    "fatal Spark TP4 vocabulary graph session is absent "
                    "during custom capture; terminating worker to prevent "
                    "a rank-split collective"
                )
                _abort_after_native_failure()
                raise AssertionError(
                    "unreachable after worker termination"
                )
            else:
                _record_stock_path(
                    capturing=True,
                    reason="graph_transport_disabled",
                )
            return original(self, input_tensor, dim)

        if os.getenv("SPARK_TP4_STOCK_TIMING", "0") == "1":
            from spark_tp4_stock_timing import time_original

            stream = torch.cuda.current_stream(device=input_tensor.device)
            _record_stock_path(
                capturing=False,
                reason="stock_timing",
            )
            return time_original(
                "vocab",
                signature,
                stream,
                lambda: original(self, input_tensor, dim),
                torch,
            )

        backend = getattr(self, "_spark_tp4_vocab_native", None)
        if backend is None:
            backend = _Backend(int(self.rank_in_group))
            self._spark_tp4_vocab_native = backend
        if mode == "custom" and _graph_enabled():
            graph_session = backend.prepare_graph()
            if graph_session is None:
                logger.critical(
                    "fatal Spark TP4 vocabulary graph session creation "
                    "failed in custom mode; terminating worker to prevent "
                    "a rank-split collective"
                )
                _abort_after_native_failure()
                raise AssertionError(
                    "unreachable after worker termination"
                )

        session = backend.session()
        if session is None:
            if mode == "custom":
                logger.critical(
                    "fatal Spark TP4 vocabulary eager session creation "
                    "failed in custom mode; terminating worker to prevent "
                    "a rank-split collective"
                )
                _abort_after_native_failure()
                raise AssertionError(
                    "unreachable after worker termination"
                )
            _record_stock_path(
                capturing=False,
                reason="native_session_unavailable",
            )
            return original(self, input_tensor, dim)

        shadow_limit = 0
        shadow = None
        promoted = False
        if mode == "shadow":
            shadow_limit = _positive_integer(
                "SPARK_TP4_VOCAB_SHADOW_COLLECTIVES", 8
            )
            shadow = backend.shadows.get(signature)
            if shadow is None:
                template = _new_output(torch, input_tensor, signature)
                shadow = backend.shadow(signature, template)
            promoted = shadow.validated and (
                os.getenv("SPARK_TP4_VOCAB_SHADOW_PROMOTE", "0") == "1"
            )
            if not promoted and shadow.count >= shadow_limit:
                _record_stock_path(
                    capturing=False,
                    reason="shadow_reference_only",
                )
                return original(self, input_tensor, dim)

        candidate = (
            _new_output(torch, input_tensor, signature)
            if mode == "custom" or promoted
            else shadow.candidate
        )
        stream = torch.cuda.current_stream(device=input_tensor.device)
        try:
            session.all_gather(
                input_tensor, candidate, signature, stream
            )
        except BaseException:
            logger.exception(
                "fatal Spark TP4 vocabulary failure; terminating worker "
                "because native enqueue may have poisoned its CUDA stream"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")

        if mode == "custom" or promoted:
            return candidate

        try:
            _record_stock_path(
                capturing=False,
                reason="shadow_reference",
            )
            reference = original(self, input_tensor, dim)
            assert shadow is not None
            shadow.observe(reference)
        except BaseException:
            logger.exception(
                "fatal failure after Spark TP4 vocabulary enqueue; "
                "terminating worker"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")

        if shadow.count == shadow_limit:
            mismatches = shadow.mismatch_count()
            logger.warning(
                "Spark TP4 vocabulary shadow complete: q=%d "
                "input_bytes=%d collectives=%d byte_mismatches=%d",
                signature,
                signature * _VOCAB_PER_RANK * _BF16_BYTES,
                shadow_limit,
                mismatches,
            )
            if mismatches:
                raise RuntimeError(
                    "Spark TP4 vocabulary shadow found byte mismatches"
                )
            shadow.validated = True
            if (
                os.getenv("SPARK_TP4_VOCAB_SHADOW_PROMOTE", "0")
                == "1"
            ):
                logger.warning(
                    "Spark TP4 vocabulary Q%d will promote to custom on "
                    "its next call",
                    signature,
                )
        return reference

    spark_vocab_all_gather._spark_tp4_vocab_backend = True  # type: ignore[attr-defined]
    spark_vocab_all_gather._spark_original = original  # type: ignore[attr-defined]
    GroupCoordinator._all_gather_out_place = spark_vocab_all_gather
    _installed = True
    logger.warning(
        "installed Spark TP4 vocabulary backend in %s mode", _mode()
    )
