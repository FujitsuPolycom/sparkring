"""Signature-gated vLLM adapter for the direct-cable TP4 transport."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

from spark_persistent_output_ring import PersistentOutputRing
from spark_tp4_port_namespace import (
    eager_allreduce_control_ports,
    graph_allreduce_control_ports,
    graph_dual_port_q40_control_ports,
    validate_active_port_namespace,
    validate_control_port_pair,
)
from spark_tp4_prefill_capacity_pool import capacity_pool_requested
from spark_tp4_query_contract import (
    ABSOLUTE_MAX_QUERY_ROWS,
    MAX_QUERY_ROWS,
)

logger = logging.getLogger(__name__)

_installed = False
_VALID_MODES = {"shadow", "custom", "disabled"}
_TARGET_WIDTH = 6144
_BF16_BYTES = 2
_BYTES_PER_ROW = _TARGET_WIDTH * _BF16_BYTES
_ALLREDUCE_PREFILL_MAX_QUERY_ROWS = 512
_ALLREDUCE_PREFILL_CAPACITY_BYTES = (
    _ALLREDUCE_PREFILL_MAX_QUERY_ROWS * _BYTES_PER_ROW
)
_DUAL_PORT_Q40_CAPACITY_BYTES = ABSOLUTE_MAX_QUERY_ROWS * _BYTES_PER_ROW
_TARGET_SHAPES = frozenset(
    (rows, _TARGET_WIDTH) for rows in range(1, MAX_QUERY_ROWS + 1)
)
_GRAPH_CAPACITY_BYTES = MAX_QUERY_ROWS * _BYTES_PER_ROW
_GRAPH_STATUS_CAPTURE_CONFIGURED = 1 << 0
_GRAPH_STATUS_POLLING_ENABLED = 1 << 1
_GRAPH_STATUS_HOST_NATIVE_ATOMICS = 1 << 2
_GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED = 1 << 3
_GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED = 1 << 4
_GRAPH_STATUS_OVERFLOW_FATAL = 1 << 5
_GRAPH_STATUS_TWO_SLOT_DEFERRED_ACK = 1 << 7
_GRAPH_STATUS_SPLIT_64K = 1 << 8
_GRAPH_STATUS_TIERED_64K = 1 << 9
_GRAPH_STATUS_DUAL_PORT_STRIPED = 1 << 10
_SERIAL_ACK_PROTOCOL = "serial_ack"
_TWO_SLOT_DEFERRED_ACK_PROTOCOL = "two_slot_deferred_ack"
_ALLREDUCE_PROTOCOL_WIRE = {
    _SERIAL_ACK_PROTOCOL: 0,
    _TWO_SLOT_DEFERRED_ACK_PROTOCOL: 1,
}
_FUSED_GRAPH_KERNEL = "fused"
_SPLIT_64K_GRAPH_KERNEL = "split_64k"
_TIERED_64K_GRAPH_KERNEL = "tiered_64k"
_GRAPH_KERNEL_WIRE = {
    _FUSED_GRAPH_KERNEL: 0,
    _SPLIT_64K_GRAPH_KERNEL: 1,
    _TIERED_64K_GRAPH_KERNEL: 2,
}
_SEQUENTIAL_WIRE_SCHEDULE = "sequential"
_DUAL_PORT_STRIPED_WIRE_SCHEDULE = "dual_port_striped"
_WIRE_SCHEDULE_WIRE = {
    _SEQUENTIAL_WIRE_SCHEDULE: 0,
    _DUAL_PORT_STRIPED_WIRE_SCHEDULE: 1,
}
_MAX_PERSISTENT_OUTPUT_SLOTS = 4096
_graph_q1_sessions: dict[int, "_NativeSession"] = {}
_graph_dual_port_q40_sessions: dict[int, "_NativeSession"] = {}
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


def _abort_after_native_failure() -> None:
    """Terminate a worker whose CUDA stream may contain an unfulfillable wait."""
    os._exit(70)


class _NativeConfig(ctypes.Structure):
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
        ("payload_bytes", ctypes.c_size_t),
        ("graph_submit_cpu_plus_one", ctypes.c_uint32),
        ("graph_progress_cpu_plus_one", ctypes.c_uint32),
    ]


class _NativeGraphStatus(ctypes.Structure):
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


@dataclass(frozen=True)
class GraphReplayStatus:
    captured_nodes: int
    published_sequence: int
    consumed_sequence: int
    completed_sequence: int
    overflow_sequence: int
    capture_configured: bool
    polling_enabled: bool
    host_native_atomics: bool
    submit_affinity_verified: bool
    progress_affinity_verified: bool
    submit_cpu: int | None
    progress_cpu: int | None
    two_slot_deferred_ack: bool = False
    graph_kernel_strategy: str = _FUSED_GRAPH_KERNEL
    wire_schedule: str = _SEQUENTIAL_WIRE_SCHEDULE

    @property
    def replay_advanced(self) -> bool:
        """True only after a replay command has reached the native publisher."""
        return self.published_sequence > 0

    @property
    def command_caught_up(self) -> bool:
        return (
            self.replay_advanced
            and self.published_sequence == self.consumed_sequence
            and self.published_sequence == self.completed_sequence
        )

    @property
    def replay_caught_up(self) -> bool:
        """Compatibility alias for command-ring completion.

        In two-slot mode this proves both outgoing credits were posted, not
        that reciprocal credits retired both physical payload slots.
        """
        return self.command_caught_up

    @property
    def payload_slots_retired(self) -> bool | None:
        """Return retirement proof when the selected protocol exposes it."""
        if self.two_slot_deferred_ack:
            return None
        return self.command_caught_up

    @property
    def fatal(self) -> bool:
        return self.overflow_sequence != 0

    def to_dict(self) -> dict[str, object]:
        snapshot = asdict(self)
        snapshot["command_caught_up"] = self.command_caught_up
        snapshot["replay_advanced"] = self.replay_advanced
        snapshot["replay_caught_up"] = self.replay_caught_up
        snapshot["payload_slots_retired"] = self.payload_slots_retired
        snapshot["fatal"] = self.fatal
        return snapshot


def _mode() -> str:
    mode = os.getenv("VLLM_SPARK_TP4_MODE", "").lower()
    if mode and mode not in _VALID_MODES:
        raise ValueError("VLLM_SPARK_TP4_MODE must be 'shadow', 'custom', 'disabled', or unset")
    return mode


def _graph_allreduce_protocol() -> str:
    value = os.getenv(
        "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL",
        _SERIAL_ACK_PROTOCOL,
    )
    if value not in _ALLREDUCE_PROTOCOL_WIRE:
        raise ValueError(
            "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL must be "
            "'serial_ack', 'two_slot_deferred_ack', or unset"
        )
    return value


def _graph_kernel_strategy() -> str:
    value = os.getenv(
        "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY",
        _FUSED_GRAPH_KERNEL,
    )
    if value not in _GRAPH_KERNEL_WIRE:
        raise ValueError(
            "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY must be "
            "'fused', 'split_64k', 'tiered_64k', or unset"
        )
    return value


def _prefill_q512_enabled() -> bool:
    value = os.getenv("VLLM_SPARK_TP4_PREFILL_Q512", "0")
    if value not in {"0", "1"}:
        raise ValueError(
            "VLLM_SPARK_TP4_PREFILL_Q512 must be '0', '1', or unset"
        )
    return value == "1"


def _maximum_allreduce_query_rows() -> int:
    return (
        _ALLREDUCE_PREFILL_MAX_QUERY_ROWS
        if _prefill_q512_enabled()
        else MAX_QUERY_ROWS
    )


def _graph_capacity_bytes() -> int:
    if _prefill_q512_enabled():
        return _ALLREDUCE_PREFILL_CAPACITY_BYTES
    return MAX_QUERY_ROWS * _BYTES_PER_ROW


def _target_shape_eligible(shape: tuple[int, ...]) -> bool:
    return (
        len(shape) == 2
        and shape[1] == _TARGET_WIDTH
        and 1 <= shape[0] <= _maximum_allreduce_query_rows()
    )


def _shadow_result_passed(
    outside_tolerance: int,
    nonfinite_mismatches: int,
    max_ulp: int,
) -> bool:
    """Apply numerical gates without making near-zero BF16 ULP diagnostic fatal."""

    max_allowed_ulp = int(os.getenv("SPARK_TP4_SHADOW_MAX_ULP", "0"))
    if max_allowed_ulp < 0:
        raise ValueError("SPARK_TP4_SHADOW_MAX_ULP must be nonnegative")
    ulp_failed = max_allowed_ulp > 0 and max_ulp > max_allowed_ulp
    return not (outside_tolerance or nonfinite_mismatches or ulp_failed)


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(size) for size in tensor.shape)


def _eligible(communicator: Any, tensor: Any, mode: str | None = None) -> bool:
    active_mode = _mode() if mode is None else mode
    return (
        active_mode in _VALID_MODES
        and getattr(communicator, "world_size", None) == 4
        and getattr(communicator, "unique_name", "") == "tp:0"
        and _target_shape_eligible(_tensor_shape(tensor))
        and str(tensor.dtype) == "torch.bfloat16"
        and bool(tensor.is_cuda)
        and bool(tensor.is_contiguous())
    )


def _is_stream_capturing(torch_module: Any) -> bool:
    checker = getattr(torch_module.cuda, "is_current_stream_capturing", None)
    return bool(checker is not None and checker())


def _record_stock_path(
    *,
    capturing: bool,
    reason: str,
    communicator: Any | None = None,
    tensor: Any | None = None,
) -> None:
    from spark_collective_audit import (
        StockCollectiveSignature,
        enabled,
        record_stock,
    )

    signature = None
    if enabled() and communicator is not None and tensor is not None:
        world_size = getattr(communicator, "world_size", None)
        signature = StockCollectiveSignature(
            shape=_tensor_shape(tensor),
            dtype=str(tensor.dtype),
            is_cuda=bool(tensor.is_cuda),
            contiguous=bool(tensor.is_contiguous()),
            world_size=(
                None if world_size is None else int(world_size)
            ),
            unique_name=str(getattr(communicator, "unique_name", "")),
        )
    record_stock(
        "all_reduce",
        capturing=capturing,
        reason=reason,
        signature=signature,
    )


def _payload_bytes(tensor: Any) -> int:
    rows, width = _tensor_shape(tensor)
    return rows * width * _BF16_BYTES


def _collective_signature(tensor: Any) -> tuple[int, tuple[int, ...], str]:
    return (
        _payload_bytes(tensor),
        _tensor_shape(tensor),
        str(tensor.dtype),
    )


def _control_ports(payload_bytes: int) -> tuple[int, int]:
    if (
        payload_bytes <= 0
        or payload_bytes % _BYTES_PER_ROW != 0
        or payload_bytes // _BYTES_PER_ROW
        > _maximum_allreduce_query_rows()
    ):
        raise ValueError(
            f"unsupported Spark TP4 payload size: {payload_bytes} bytes"
        )
    query_rows = payload_bytes // _BYTES_PER_ROW
    return eager_allreduce_control_ports(query_rows)


def _graph_q1_enabled() -> bool:
    value = os.getenv("VLLM_SPARK_TP4_GRAPH_Q1", "0")
    if value not in {"0", "1"}:
        raise ValueError("VLLM_SPARK_TP4_GRAPH_Q1 must be '0' or '1'")
    return value == "1"


def _graph_dual_port_q40_enabled() -> bool:
    value = os.getenv("VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40", "0")
    if value not in {"0", "1"}:
        raise ValueError(
            "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40 must be '0' or '1'"
        )
    return value == "1"


def _persistent_output_slots() -> int:
    text = os.getenv("SPARK_TP4_PERSISTENT_OUTPUT_SLOTS", "0")
    try:
        slots = int(text)
    except ValueError as error:
        raise ValueError(
            "SPARK_TP4_PERSISTENT_OUTPUT_SLOTS must be an integer"
        ) from error
    if slots < 0 or slots > _MAX_PERSISTENT_OUTPUT_SLOTS:
        raise ValueError(
            "SPARK_TP4_PERSISTENT_OUTPUT_SLOTS must be in "
            f"[0, {_MAX_PERSISTENT_OUTPUT_SLOTS}]"
        )
    return slots


def _graph_control_ports() -> tuple[int, int]:
    return graph_allreduce_control_ports()


def _graph_dual_port_q40_control_ports() -> tuple[int, int]:
    return graph_dual_port_q40_control_ports()


def _fixed_kv_cache_bytes(argv: list[str] | None = None) -> int:
    arguments = sys.argv if argv is None else argv
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "--kv-cache-memory-bytes":
            if index + 1 >= len(arguments):
                raise RuntimeError(
                    "graph TP4 requires a value for --kv-cache-memory-bytes"
                )
            values.append(arguments[index + 1])
        elif argument.startswith("--kv-cache-memory-bytes="):
            values.append(argument.partition("=")[2])
    if len(values) != 1:
        raise RuntimeError(
            "graph TP4 requires exactly one positive "
            "--kv-cache-memory-bytes value so vLLM skips its throwaway "
            "memory-profiling capture stream"
        )
    try:
        parsed = int(values[0])
    except ValueError as error:
        raise RuntimeError(
            "graph TP4 --kv-cache-memory-bytes must be a positive integer"
        ) from error
    if parsed <= 0:
        raise RuntimeError(
            "graph TP4 --kv-cache-memory-bytes must be a positive integer"
        )
    return parsed


def _graph_cpu_affinity() -> tuple[int, int]:
    names = (
        "SPARK_TP4_GRAPH_SUBMIT_CPU",
        "SPARK_TP4_GRAPH_PROGRESS_CPU",
    )
    values: list[int] = []
    for name in names:
        text = os.getenv(name)
        if text is None:
            raise RuntimeError(
                f"graph TP4 requires explicit {name}; submission and "
                "transport progress must use distinct CPUs"
            )
        try:
            value = int(text)
        except ValueError as error:
            raise RuntimeError(f"graph TP4 {name} must be a nonnegative integer") from error
        if value < 0:
            raise RuntimeError(f"graph TP4 {name} must be a nonnegative integer")
        values.append(value)
    submit_cpu, progress_cpu = values
    if submit_cpu == progress_cpu:
        raise RuntimeError(
            "graph TP4 submit/progress CPUs must be distinct"
        )
    return submit_cpu, progress_cpu


def _graph_preflight(argv: list[str] | None = None) -> tuple[int, int]:
    _fixed_kv_cache_bytes(argv)
    if os.getenv("VLLM_SPARK_SHARED_CAPTURE_STREAM", "0") != "1":
        raise RuntimeError(
            "graph TP4 requires VLLM_SPARK_SHARED_CAPTURE_STREAM=1"
        )
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "graph TP4 CPU affinity requires Linux pthread affinity support"
        )
    return _graph_cpu_affinity()


def _graph_dual_port_q40_cpu_affinity(
    argv: list[str] | None = None,
) -> tuple[int, int]:
    submit_cpu, tp_progress_cpu = _graph_preflight(argv)
    name = "SPARK_TP4_GRAPH_DUAL_PORT_Q40_PROGRESS_CPU"
    text = os.getenv(name)
    if text is None:
        raise RuntimeError(
            f"exact-Q40 dual-port graph TP4 requires explicit {name}"
        )
    try:
        progress_cpu = int(text)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a nonnegative integer") from error
    if progress_cpu < 0:
        raise RuntimeError(f"{name} must be a nonnegative integer")

    occupied = {
        submit_cpu: "shared graph submit",
        tp_progress_cpu: "sequential TP graph progress",
    }
    if os.getenv("VLLM_SPARK_TP4_VOCAB_MODE", "") == "custom":
        occupied[
            int(os.getenv("SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU", "12"))
        ] = "vocabulary graph progress"
    if (
        os.getenv("VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM", "0") == "1"
        or os.getenv("VLLM_SPARK_TP4_DCP_GRAPH_SHADOW", "0") == "1"
    ):
        occupied[
            int(os.getenv("SPARK_TP4_GRAPH_DCP_PROGRESS_CPU", "13"))
        ] = "DCP graph progress"
    if os.getenv("VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM", "0") == "1":
        occupied[
            int(os.getenv("SPARK_TP4_GRAPH_INDEXER_PROGRESS_CPU", "14"))
        ] = "indexer graph progress"
    if progress_cpu in occupied:
        raise RuntimeError(
            f"{name}={progress_cpu} collides with "
            f"{occupied[progress_cpu]} CPU ownership"
        )
    return submit_cpu, progress_cpu


def _record_graph_event(communicator: Any, event: str) -> int:
    attribute = f"_spark_tp4_graph_q1_{event}"
    count = int(getattr(communicator, attribute, 0)) + 1
    setattr(communicator, attribute, count)
    _graph_event_counts[event] = _graph_event_counts.get(event, 0) + 1
    if count == 1 or count % 128 == 0:
        logger.warning(
            "Spark TP4 mixed-Q graph %s on rank %d: count=%d",
            event,
            int(communicator.rank_in_group),
            count,
        )
    return count


class _NativeSession:
    def __init__(
        self,
        rank: int,
        payload_bytes: int,
        *,
        control_ports: tuple[int, int] | None = None,
        graph_only: bool = False,
        graph_cpu_affinity: tuple[int, int] | None = None,
        allreduce_protocol: str = _SERIAL_ACK_PROTOCOL,
        graph_kernel_strategy: str = _FUSED_GRAPH_KERNEL,
        wire_schedule: str = _SEQUENTIAL_WIRE_SCHEDULE,
    ) -> None:
        if rank not in _DEFAULT_PEERS:
            raise ValueError(f"TP4 rank must be in [0, 3], got {rank}")
        expected_graph_capacity = _graph_capacity_bytes()
        if (
            graph_only
            and wire_schedule == _SEQUENTIAL_WIRE_SCHEDULE
            and payload_bytes != expected_graph_capacity
        ):
            raise ValueError(
                "Spark TP4 graph session requires "
                f"Q{expected_graph_capacity // _BYTES_PER_ROW} capacity"
            )
        if graph_only != (graph_cpu_affinity is not None):
            raise ValueError(
                "Spark TP4 graph session requires an explicit graph CPU "
                "affinity pair; eager sessions cannot set one"
            )
        if allreduce_protocol not in _ALLREDUCE_PROTOCOL_WIRE:
            raise ValueError("invalid Spark TP4 all-reduce protocol")
        if graph_kernel_strategy not in _GRAPH_KERNEL_WIRE:
            raise ValueError("invalid Spark TP4 graph kernel strategy")
        if wire_schedule not in _WIRE_SCHEDULE_WIRE:
            raise ValueError("invalid Spark TP4 wire schedule")
        if graph_kernel_strategy != _FUSED_GRAPH_KERNEL and not graph_only:
            raise ValueError(
                "research graph kernel strategies require a graph-only session"
            )
        if (
            allreduce_protocol == _TWO_SLOT_DEFERRED_ACK_PROTOCOL
            and not graph_only
        ):
            raise ValueError(
                "two_slot_deferred_ack requires a graph-only session"
            )
        if wire_schedule == _DUAL_PORT_STRIPED_WIRE_SCHEDULE:
            if not graph_only:
                raise ValueError(
                    "dual_port_striped requires a graph-only session"
                )
            if allreduce_protocol != _TWO_SLOT_DEFERRED_ACK_PROTOCOL:
                raise ValueError(
                    "dual_port_striped requires two_slot_deferred_ack"
                )
            if graph_kernel_strategy != _FUSED_GRAPH_KERNEL:
                raise ValueError(
                    "dual_port_striped requires the fused graph kernel"
                )
            if payload_bytes not in {
                _DUAL_PORT_Q40_CAPACITY_BYTES,
                _ALLREDUCE_PREFILL_CAPACITY_BYTES,
            }:
                raise ValueError(
                    "dual_port_striped requires Q40 or Q512 capacity"
                )
        if control_ports is None:
            control_port0, control_port1 = _control_ports(payload_bytes)
        else:
            control_port0, control_port1 = validate_control_port_pair(
                control_ports,
                owner=(
                    "graph all-reduce" if graph_only else "eager all-reduce"
                ),
            )
            validate_active_port_namespace()
        library_path = os.environ["SPARK_TP4_LIBRARY"]
        self._library = ctypes.CDLL(library_path)
        self._library.spark_tp4_create.argtypes = [
            ctypes.POINTER(_NativeConfig),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_create.restype = ctypes.c_void_p
        create_session = self._library.spark_tp4_create
        if wire_schedule != _SEQUENTIAL_WIRE_SCHEDULE:
            try:
                create_session = (
                    self._library
                    .spark_tp4_create_with_protocol_graph_kernel_and_schedule
                )
            except AttributeError as error:
                raise RuntimeError(
                    "Spark TP4 native library lacks the wire-schedule ABI; "
                    "rebuild and deploy a matching library"
                ) from error
            create_session.argtypes = [
                ctypes.POINTER(_NativeConfig),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            create_session.restype = ctypes.c_void_p
        elif graph_kernel_strategy != _FUSED_GRAPH_KERNEL:
            try:
                create_session = (
                    self._library.spark_tp4_create_with_protocol_and_graph_kernel
                )
            except AttributeError as error:
                raise RuntimeError(
                    "Spark TP4 native library lacks the graph-kernel ABI; "
                    "rebuild and deploy a matching library"
                ) from error
            create_session.argtypes = [
                ctypes.POINTER(_NativeConfig),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            create_session.restype = ctypes.c_void_p
        elif allreduce_protocol == _TWO_SLOT_DEFERRED_ACK_PROTOCOL:
            try:
                create_session = (
                    self._library.spark_tp4_create_with_protocol
                )
            except AttributeError as error:
                raise RuntimeError(
                    "Spark TP4 native library lacks the deferred-ACK ABI; "
                    "rebuild and deploy a matching library"
                ) from error
            create_session.argtypes = [
                ctypes.POINTER(_NativeConfig),
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            create_session.restype = ctypes.c_void_p
        self._library.spark_tp4_all_reduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.spark_tp4_all_reduce.restype = ctypes.c_int
        if graph_only:
            self._library.spark_tp4_capture_all_reduce.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_capture_all_reduce.restype = ctypes.c_int
            self._library.spark_tp4_get_graph_status.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NativeGraphStatus),
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            self._library.spark_tp4_get_graph_status.restype = ctypes.c_int
        self._library.spark_tp4_destroy.argtypes = [ctypes.c_void_p]
        self._library.spark_tp4_destroy.restype = None
        self._graph_only = graph_only
        self._allreduce_protocol = allreduce_protocol
        self._graph_kernel_strategy = graph_kernel_strategy
        self._wire_schedule = wire_schedule
        self._graph_max_query_rows = (
            payload_bytes // _BYTES_PER_ROW if graph_only else 0
        )
        persistent_slots = 0 if graph_only else _persistent_output_slots()
        self._persistent_outputs = (
            PersistentOutputRing(persistent_slots)
            if persistent_slots
            else None
        )
        if persistent_slots:
            logger.warning(
                "Spark TP4 eager persistent output ring enabled on rank %d "
                "for %d bytes with %d slots",
                rank,
                payload_bytes,
                persistent_slots,
            )

        default_peer0, default_peer1 = _DEFAULT_PEERS[rank]
        submit_cpu, progress_cpu = graph_cpu_affinity or (-1, -1)
        config = _NativeConfig(
            rank=rank,
            peer0=os.getenv("SPARK_TP4_PEER0", default_peer0).encode(),
            peer1=os.getenv("SPARK_TP4_PEER1", default_peer1).encode(),
            device0=os.getenv("SPARK_TP4_DEVICE0", "rocep1s0f0").encode(),
            device1=os.getenv("SPARK_TP4_DEVICE1", "rocep1s0f1").encode(),
            gid0=int(os.getenv("SPARK_TP4_GID0", "3")),
            gid1=int(os.getenv("SPARK_TP4_GID1", "3")),
            control_port0=control_port0,
            control_port1=control_port1,
            payload_bytes=payload_bytes,
            graph_submit_cpu_plus_one=submit_cpu + 1,
            graph_progress_cpu_plus_one=progress_cpu + 1,
        )
        error = ctypes.create_string_buffer(512)
        if wire_schedule != _SEQUENTIAL_WIRE_SCHEDULE:
            self._handle = create_session(
                ctypes.byref(config),
                _ALLREDUCE_PROTOCOL_WIRE[allreduce_protocol],
                _GRAPH_KERNEL_WIRE[graph_kernel_strategy],
                _WIRE_SCHEDULE_WIRE[wire_schedule],
                error,
                len(error),
            )
        elif graph_kernel_strategy != _FUSED_GRAPH_KERNEL:
            self._handle = create_session(
                ctypes.byref(config),
                _ALLREDUCE_PROTOCOL_WIRE[allreduce_protocol],
                _GRAPH_KERNEL_WIRE[graph_kernel_strategy],
                error,
                len(error),
            )
        elif allreduce_protocol == _SERIAL_ACK_PROTOCOL:
            self._handle = create_session(
                ctypes.byref(config), error, len(error)
            )
        else:
            self._handle = create_session(
                ctypes.byref(config),
                _ALLREDUCE_PROTOCOL_WIRE[allreduce_protocol],
                error,
                len(error),
            )
        if not self._handle:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"failed to create Spark TP4 session: {message}")
        logger.warning(
            "Spark TP4 %s session ready on rank %d for %d bytes "
            "(control ports %d/%d, protocol %s, graph kernel %s, "
            "wire schedule %s)",
            "graph-only" if graph_only else "eager",
            rank,
            payload_bytes,
            control_port0,
            control_port1,
            allreduce_protocol,
            graph_kernel_strategy,
            wire_schedule,
        )

    def all_reduce(self, tensor: Any) -> Any:
        import torch

        output = (
            torch.empty_like(tensor)
            if self._persistent_outputs is None
            else self._persistent_outputs.acquire(tensor, torch)
        )
        stream = torch.cuda.current_stream(device=tensor.device)
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_all_reduce(
            self._handle,
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 all-reduce failed: {message}")
        return output

    def graph_status(self) -> GraphReplayStatus:
        if not self._graph_only:
            raise RuntimeError(
                "Spark TP4 eager session has no graph replay status"
            )
        native = _NativeGraphStatus()
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_get_graph_status(
            self._handle,
            ctypes.byref(native),
            ctypes.sizeof(native),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 graph status failed: {message}")
        if native.struct_size != ctypes.sizeof(_NativeGraphStatus):
            raise RuntimeError(
                "Spark TP4 graph status ABI mismatch: "
                f"native={native.struct_size} python="
                f"{ctypes.sizeof(_NativeGraphStatus)}"
            )
        flags = int(native.flags)
        native_deferred = bool(
            flags & _GRAPH_STATUS_TWO_SLOT_DEFERRED_ACK
        )
        requested_deferred = (
            self._allreduce_protocol == _TWO_SLOT_DEFERRED_ACK_PROTOCOL
        )
        if native_deferred != requested_deferred:
            raise RuntimeError(
                "Spark TP4 graph protocol attestation mismatch: "
                f"requested={self._allreduce_protocol} "
                f"native_deferred={native_deferred}"
            )
        native_split = bool(flags & _GRAPH_STATUS_SPLIT_64K)
        native_tiered = bool(flags & _GRAPH_STATUS_TIERED_64K)
        if native_split and native_tiered:
            raise RuntimeError(
                "Spark TP4 graph kernel attestation is contradictory"
            )
        native_graph_kernel = (
            _TIERED_64K_GRAPH_KERNEL
            if native_tiered
            else _SPLIT_64K_GRAPH_KERNEL
            if native_split
            else _FUSED_GRAPH_KERNEL
        )
        if native_graph_kernel != self._graph_kernel_strategy:
            raise RuntimeError(
                "Spark TP4 graph kernel attestation mismatch: "
                f"requested={self._graph_kernel_strategy} "
                f"native={native_graph_kernel}"
            )
        native_wire_schedule = (
            _DUAL_PORT_STRIPED_WIRE_SCHEDULE
            if flags & _GRAPH_STATUS_DUAL_PORT_STRIPED
            else _SEQUENTIAL_WIRE_SCHEDULE
        )
        if native_wire_schedule != self._wire_schedule:
            raise RuntimeError(
                "Spark TP4 wire schedule attestation mismatch: "
                f"requested={self._wire_schedule} "
                f"native={native_wire_schedule}"
            )
        return GraphReplayStatus(
            captured_nodes=int(native.captured_nodes),
            published_sequence=int(native.published_sequence),
            consumed_sequence=int(native.consumed_sequence),
            completed_sequence=int(native.completed_sequence),
            overflow_sequence=int(native.overflow_sequence),
            capture_configured=bool(
                flags & _GRAPH_STATUS_CAPTURE_CONFIGURED
            ),
            polling_enabled=bool(flags & _GRAPH_STATUS_POLLING_ENABLED),
            host_native_atomics=bool(
                flags & _GRAPH_STATUS_HOST_NATIVE_ATOMICS
            ),
            submit_affinity_verified=bool(
                flags & _GRAPH_STATUS_SUBMIT_AFFINITY_VERIFIED
            ),
            progress_affinity_verified=bool(
                flags & _GRAPH_STATUS_PROGRESS_AFFINITY_VERIFIED
            ),
            submit_cpu=(
                int(native.graph_submit_cpu_plus_one) - 1
                if native.graph_submit_cpu_plus_one
                else None
            ),
            progress_cpu=(
                int(native.graph_progress_cpu_plus_one) - 1
                if native.graph_progress_cpu_plus_one
                else None
            ),
            two_slot_deferred_ack=native_deferred,
            graph_kernel_strategy=native_graph_kernel,
            wire_schedule=native_wire_schedule,
        )

    def capture(self, tensor: Any) -> Any:
        if not self._graph_only:
            raise RuntimeError("Spark TP4 eager session cannot define graph nodes")
        shape = _tensor_shape(tensor)
        dual_port_session = (
            self._wire_schedule == _DUAL_PORT_STRIPED_WIRE_SCHEDULE
        )
        exact_dual_port_shape = (
            not dual_port_session
            or shape == (self._graph_max_query_rows, _TARGET_WIDTH)
        )
        if (
            not _target_shape_eligible(shape)
            or shape[0] > self._graph_max_query_rows
            or not exact_dual_port_shape
            or str(tensor.dtype) != "torch.bfloat16"
            or not bool(tensor.is_cuda)
            or not bool(tensor.is_contiguous())
        ):
            query_requirement = (
                f"Q={self._graph_max_query_rows}"
                if dual_port_session
                else f"Q in [1, {self._graph_max_query_rows}]"
            )
            raise ValueError(
                "Spark TP4 graph capture requires contiguous CUDA BF16 "
                f"[Q, 6144] with {query_requirement}; dual-port striped "
                "sessions require their exact fixed Q"
            )
        q = shape[0]

        import torch

        output = torch.empty_like(tensor)
        stream = torch.cuda.current_stream(device=tensor.device)
        error = ctypes.create_string_buffer(512)
        result = self._library.spark_tp4_capture_all_reduce(
            self._handle,
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            q,
            ctypes.c_void_p(stream.cuda_stream),
            error,
            len(error),
        )
        if result != 0:
            message = error.value.decode(errors="replace")
            raise RuntimeError(f"Spark TP4 graph capture failed: {message}")
        return output

    def capture_q1(self, tensor: Any) -> Any:
        if _tensor_shape(tensor) != (1, _TARGET_WIDTH):
            raise ValueError("Spark TP4 capture_q1 requires BF16 [1, 6144]")
        return self.capture(tensor)


class _ShadowStats:
    def __init__(self) -> None:
        self.count = 0
        self.validated = False
        self.exact_mismatches: Any | None = None
        self.outside_tolerance: Any | None = None
        self.nonfinite_mismatches: Any | None = None
        self.maximum_absolute_error: Any | None = None
        self.maximum_ulp_error: Any | None = None
        self.ulp_over_one: Any | None = None
        self.ulp_over_two: Any | None = None
        self.ulp_over_four: Any | None = None

    def observe(self, candidate: Any, reference: Any) -> None:
        import torch

        candidate_raw = candidate.view(torch.int16)
        reference_raw = reference.view(torch.int16)
        exact = torch.count_nonzero(candidate_raw != reference_raw)
        close = torch.isclose(
            candidate, reference, rtol=0.01, atol=0.01, equal_nan=True
        )
        outside = torch.count_nonzero(~close)
        candidate_finite = torch.isfinite(candidate)
        reference_finite = torch.isfinite(reference)
        both_finite = candidate_finite & reference_finite
        same_nan = torch.isnan(candidate) & torch.isnan(reference)
        same_infinity = (
            torch.isinf(candidate) & torch.isinf(reference) & (candidate == reference)
        )
        nonfinite = torch.count_nonzero(~(both_finite | same_nan | same_infinity))

        absolute = (candidate.float() - reference.float()).abs()
        finite_absolute = torch.where(both_finite, absolute, torch.zeros_like(absolute))
        delta = finite_absolute.max()

        candidate_bits = candidate_raw.to(torch.int32) & 0xFFFF
        reference_bits = reference_raw.to(torch.int32) & 0xFFFF
        candidate_ordered = torch.where(
            (candidate_bits & 0x8000) != 0,
            0x8000 - (candidate_bits & 0x7FFF),
            0x8000 + candidate_bits,
        )
        reference_ordered = torch.where(
            (reference_bits & 0x8000) != 0,
            0x8000 - (reference_bits & 0x7FFF),
            0x8000 + reference_bits,
        )
        ulp_distance = torch.where(
            both_finite,
            (candidate_ordered - reference_ordered).abs(),
            torch.zeros_like(candidate_ordered),
        )
        ulp = ulp_distance.max()
        ulp_over_one = torch.count_nonzero(ulp_distance > 1)
        ulp_over_two = torch.count_nonzero(ulp_distance > 2)
        ulp_over_four = torch.count_nonzero(ulp_distance > 4)

        if self.exact_mismatches is None:
            self.exact_mismatches = exact
            self.outside_tolerance = outside
            self.nonfinite_mismatches = nonfinite
            self.maximum_absolute_error = delta
            self.maximum_ulp_error = ulp
            self.ulp_over_one = ulp_over_one
            self.ulp_over_two = ulp_over_two
            self.ulp_over_four = ulp_over_four
        else:
            self.exact_mismatches += exact
            self.outside_tolerance += outside
            self.nonfinite_mismatches += nonfinite
            self.maximum_absolute_error = torch.maximum(
                self.maximum_absolute_error, delta
            )
            self.maximum_ulp_error = torch.maximum(self.maximum_ulp_error, ulp)
            self.ulp_over_one += ulp_over_one
            self.ulp_over_two += ulp_over_two
            self.ulp_over_four += ulp_over_four
        self.count += 1

    def report(self) -> tuple[int, int, int, float, int, int, int, int]:
        return (
            int(self.exact_mismatches.item()),
            int(self.outside_tolerance.item()),
            int(self.nonfinite_mismatches.item()),
            float(self.maximum_absolute_error.item()),
            int(self.maximum_ulp_error.item()),
            int(self.ulp_over_one.item()),
            int(self.ulp_over_two.item()),
            int(self.ulp_over_four.item()),
        )


class _Backend:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.native_sessions: dict[int, _NativeSession] = {}
        self.graph_q1_session: _NativeSession | None = None
        self.graph_dual_port_q40_session: _NativeSession | None = None
        self.shadow_stats: dict[tuple[int, tuple[int, ...], str], _ShadowStats] = {}

    def native_for(self, payload_bytes: int) -> _NativeSession:
        session = self.native_sessions.get(payload_bytes)
        if session is None:
            session = _NativeSession(self.rank, payload_bytes)
            self.native_sessions[payload_bytes] = session
        return session

    def prepare_graph_q1(self) -> _NativeSession:
        session = self.graph_q1_session
        if session is None:
            graph_cpu_affinity = _graph_preflight()
            session = _NativeSession(
                self.rank,
                _graph_capacity_bytes(),
                control_ports=_graph_control_ports(),
                graph_only=True,
                graph_cpu_affinity=graph_cpu_affinity,
                allreduce_protocol=_graph_allreduce_protocol(),
                graph_kernel_strategy=_graph_kernel_strategy(),
            )
            self.graph_q1_session = session
            _graph_q1_sessions[self.rank] = session
            from spark_graph_status_reporter import (
                ensure_status_reporter,
            )

            ensure_status_reporter(rank=self.rank)
        if (
            _graph_dual_port_q40_enabled()
            and self.graph_dual_port_q40_session is None
        ):
            dual_port_session = _NativeSession(
                self.rank,
                _DUAL_PORT_Q40_CAPACITY_BYTES,
                control_ports=_graph_dual_port_q40_control_ports(),
                graph_only=True,
                graph_cpu_affinity=_graph_dual_port_q40_cpu_affinity(),
                allreduce_protocol=_TWO_SLOT_DEFERRED_ACK_PROTOCOL,
                graph_kernel_strategy=_FUSED_GRAPH_KERNEL,
                wire_schedule=_DUAL_PORT_STRIPED_WIRE_SCHEDULE,
            )
            self.graph_dual_port_q40_session = dual_port_session
            _graph_dual_port_q40_sessions[self.rank] = dual_port_session
        return session

    def graph_session_for_capture(
        self, tensor: Any
    ) -> _NativeSession | None:
        if (
            _tensor_shape(tensor) == (ABSOLUTE_MAX_QUERY_ROWS, _TARGET_WIDTH)
            and self.graph_dual_port_q40_session is not None
        ):
            return self.graph_dual_port_q40_session
        return self.graph_q1_session

    def shadow_for(self, signature: tuple[int, tuple[int, ...], str]) -> _ShadowStats:
        stats = self.shadow_stats.get(signature)
        if stats is None:
            stats = _ShadowStats()
            self.shadow_stats[signature] = stats
        return stats


def graph_q1_status_snapshot() -> dict[int, dict[str, object]]:
    """Return nonblocking native graph progress for process-local TP ranks."""
    return {
        rank: session.graph_status().to_dict()
        for rank, session in sorted(_graph_q1_sessions.items())
    }


def graph_dual_port_q40_status_snapshot() -> dict[int, dict[str, object]]:
    """Return exact-Q40 striped graph progress for process-local TP ranks."""
    return {
        rank: session.graph_status().to_dict()
        for rank, session in sorted(_graph_dual_port_q40_sessions.items())
    }


def graph_q1_diagnostic_snapshot() -> dict[str, object]:
    return {
        "sessions": graph_q1_status_snapshot(),
        "dual_port_q40_sessions": graph_dual_port_q40_status_snapshot(),
        "events": dict(sorted(_graph_event_counts.items())),
    }


def install() -> None:
    global _installed
    mode = _mode()
    graph_protocol = _graph_allreduce_protocol()
    graph_kernel = _graph_kernel_strategy()
    graph_dual_port_q40 = _graph_dual_port_q40_enabled()
    if capacity_pool_requested():
        raise RuntimeError(
            "VLLM_SPARK_TP4_PREFILL_CAPACITY_POOL is research-only: "
            "the active-bytes native ABI is unavailable, so the shared "
            "tiled engine cannot be dispatched safely"
        )
    if graph_protocol == _TWO_SLOT_DEFERRED_ACK_PROTOCOL and (
        mode != "custom" or not _graph_q1_enabled()
    ):
        raise ValueError(
            "two_slot_deferred_ack requires custom graph all-reduce"
        )
    if graph_kernel != _FUSED_GRAPH_KERNEL and (
        mode != "custom" or not _graph_q1_enabled()
    ):
        raise ValueError(
            "research graph kernel strategies require custom graph all-reduce"
        )
    if graph_dual_port_q40 and (
        mode != "custom"
        or not _graph_q1_enabled()
        or MAX_QUERY_ROWS != ABSOLUTE_MAX_QUERY_ROWS
    ):
        raise ValueError(
            "exact-Q40 dual-port graph TP4 requires custom graph all-reduce "
            f"with VLLM_SPARK_MAX_QUERY_ROWS={ABSOLUTE_MAX_QUERY_ROWS}"
        )
    if _installed or not mode or mode == "disabled":
        return
    _prefill_q512_enabled()
    validate_active_port_namespace()

    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    original = CudaCommunicator.all_reduce
    if getattr(original, "_spark_tp4_backend", False):
        _installed = True
        return

    def spark_all_reduce(self: Any, input_: Any) -> Any:
        mode = _mode()
        if not _eligible(self, input_, mode):
            import torch

            _record_stock_path(
                capturing=_is_stream_capturing(torch),
                reason="ineligible_signature",
                communicator=self,
                tensor=input_,
            )
            return original(self, input_)

        import torch

        capturing = _is_stream_capturing(torch)
        if capturing:
            if (
                mode == "custom"
                and _graph_q1_enabled()
            ):
                backend = getattr(self, "_spark_tp4_native", None)
                graph_session = (
                    None
                    if backend is None
                    else backend.graph_session_for_capture(input_)
                )
                if graph_session is not None:
                    try:
                        output = graph_session.capture(input_)
                        _record_graph_event(self, "captured_nodes")
                        return output
                    except BaseException:
                        logger.exception(
                            "fatal Spark TP4 graph-capture error; terminating "
                            "worker because a partially captured native graph "
                            "cannot safely fall back"
                        )
                        _abort_after_native_failure()
                        raise AssertionError("unreachable after worker termination")
                _record_graph_event(self, "fallbacks")
                _record_stock_path(
                    capturing=True,
                    reason="graph_session_unprepared",
                )
            else:
                _record_stock_path(
                    capturing=True,
                    reason="graph_transport_disabled",
                )
            return original(self, input_)

        payload_bytes = _payload_bytes(input_)
        signature = _collective_signature(input_)
        backend = getattr(self, "_spark_tp4_native", None)
        if backend is None:
            backend = _Backend(int(self.rank_in_group))
            self._spark_tp4_native = backend
            if os.getenv("SPARK_TP4_FLIGHT_RECORDER", "0") == "1":
                from spark_tp4_flight_recorder import activate

                activate(int(self.rank_in_group))

        shadow = None
        promoted = False
        if mode == "shadow":
            shadow = backend.shadow_for(signature)
            promoted = shadow.validated and (
                os.getenv("SPARK_TP4_SHADOW_PROMOTE", "0") == "1"
            )

        try:
            if mode == "custom" and _graph_q1_enabled():
                backend.prepare_graph_q1()
            native_session = backend.native_for(payload_bytes)
            if os.getenv("SPARK_TP4_FLIGHT_RECORDER", "0") == "1":
                import torch
                from spark_tp4_flight_recorder import record_collective

                record_collective(
                    "AR",
                    torch.cuda.current_stream(device=input_.device),
                    input_,
                )
            candidate = native_session.all_reduce(input_)
        except BaseException:
            logger.exception(
                "fatal Spark TP4 error; terminating worker because its "
                "CUDA stream may be poisoned"
            )
            _abort_after_native_failure()
            raise AssertionError("unreachable after worker termination")
        if mode == "custom" or promoted:
            return candidate

        _record_stock_path(
            capturing=False,
            reason="shadow_reference",
        )
        reference = original(self, input_)
        assert shadow is not None
        limit = int(os.getenv("SPARK_TP4_SHADOW_COLLECTIVES", "10000"))
        if shadow.count < limit:
            shadow.observe(candidate, reference)
            if shadow.count == limit:
                (
                    exact,
                    outside,
                    nonfinite,
                    maximum,
                    max_ulp,
                    ulp_gt1,
                    ulp_gt2,
                    ulp_gt4,
                ) = shadow.report()
                logger.warning(
                    "Spark TP4 shadow complete: payload_bytes=%d "
                    "shape=%s dtype=%s collectives=%d "
                    "exact_mismatches=%d outside_tolerance=%d "
                    "nonfinite_mismatches=%d max_abs=%g max_ulp=%d "
                    "ulp_gt1=%d ulp_gt2=%d ulp_gt4=%d",
                    payload_bytes,
                    _tensor_shape(input_),
                    str(input_.dtype),
                    limit,
                    exact,
                    outside,
                    nonfinite,
                    maximum,
                    max_ulp,
                    ulp_gt1,
                    ulp_gt2,
                    ulp_gt4,
                )
                passed = _shadow_result_passed(outside, nonfinite, max_ulp)
                shadow.validated = passed
                if not passed and os.getenv("SPARK_TP4_SHADOW_STRICT") == "1":
                    raise RuntimeError(
                        "Spark TP4 shadow result exceeded correctness limits"
                    )
                if passed and os.getenv("SPARK_TP4_SHADOW_PROMOTE", "0") == "1":
                    logger.warning(
                        "Spark TP4 payload_bytes=%d shape=%s dtype=%s "
                        "will promote to custom on its next call",
                        payload_bytes,
                        _tensor_shape(input_),
                        str(input_.dtype),
                    )
        return reference

    spark_all_reduce._spark_tp4_backend = True  # type: ignore[attr-defined]
    spark_all_reduce._spark_original = original  # type: ignore[attr-defined]
    CudaCommunicator.all_reduce = spark_all_reduce
    _installed = True
    logger.warning("installed Spark TP4 vLLM backend in %s mode", _mode())
