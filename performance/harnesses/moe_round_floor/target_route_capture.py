"""Bounded, CUDA-graph-safe capture of GLM-5.2 target expert routes.

The hot-path method in :class:`TargetRouteCapture` only validates tensor
metadata and dispatches a no-output custom CUDA operator.  The operator claims
round slots, copies expert IDs, and updates counters on the device.  It does
not allocate tensors, inspect device values from the CPU, or write files.

All synchronization, device-to-host transfer, validation, and JSONL output are
confined to :meth:`TargetRouteCapture.drain_jsonl`, which must be called after
the measured/model execution window.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "glm52-target-expert-routes/v1"
MODEL_ROLE_TARGET = 0
MODEL_ROLE_DRAFT = 1
SUPPORTED_WIDTHS = (5, 6)
COUNTER_NAMES = (
    "rounds_claimed",
    "rounds_completed",
    "overflow_rounds",
    "wrong_phase",
    "orphan_layer",
    "duplicate_layer",
    "incomplete_round",
    "invalid_expert",
    "rejection_order_error",
    "rejection_value_error",
)
FATAL_COUNTER_NAMES = COUNTER_NAMES[2:]
METADATA_COLUMNS = 5
META_REQUEST_SLOT = 0
META_ROUND = 1
META_WIDTH = 2
META_PHASE = 3
META_FLAGS = 4
UINT64_MASK = (1 << 64) - 1


class CaptureError(RuntimeError):
    """The capture cannot provide a complete, trustworthy route artifact."""


@dataclass(frozen=True)
class CaptureConfig:
    capacity_rounds: int = 500
    num_layers: int = 75
    max_width: int = 6
    topk: int = 8
    num_experts: int = 256
    max_request_slots: int = 64
    max_stream_slots: int = 4

    def validate(self) -> None:
        if not 100 <= self.capacity_rounds <= 500:
            raise ValueError("capacity_rounds must be in [100, 500]")
        if self.num_layers != 75:
            raise ValueError("GLM-5.2 target capture requires 75 routed layers")
        if self.max_width != 6:
            raise ValueError("max_width must be 6 for the Q5/Q6 capture")
        if self.topk != 8:
            raise ValueError("GLM-5.2 target capture requires topk=8")
        if self.num_experts != 256:
            raise ValueError("GLM-5.2 target capture requires 256 experts")
        if self.max_request_slots <= 0:
            raise ValueError("max_request_slots must be positive")
        if self.max_stream_slots <= 0:
            raise ValueError("max_stream_slots must be positive")


@dataclass(frozen=True)
class CaptureProvenance:
    image: str
    checkpoint: str
    config_sha256: str
    source_sha256: Mapping[str, str]
    rank: int

    def as_dict(self) -> dict[str, Any]:
        if not self.image or not self.checkpoint:
            raise CaptureError("image and checkpoint fingerprints are required")
        if len(self.config_sha256) != 64:
            raise CaptureError("config_sha256 must be a full SHA-256 digest")
        source = dict(sorted(self.source_sha256.items()))
        if not source or any(len(value) != 64 for value in source.values()):
            raise CaptureError("source_sha256 requires full SHA-256 digests")
        if self.rank < 0:
            raise CaptureError("rank must be non-negative")
        return {
            "image": self.image,
            "checkpoint": self.checkpoint,
            "config_sha256": self.config_sha256,
            "source_sha256": source,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class CaptureSnapshot:
    """Host materialization made only after the CUDA stream is synchronized."""

    routes: Sequence[Sequence[Sequence[Sequence[int]]]]
    metadata: Sequence[Sequence[int]]
    layer_masks: Sequence[Sequence[int]]
    counters: Sequence[int]
    rejected_tokens: Sequence[int]


def _counter_map(values: Sequence[int]) -> dict[str, int]:
    if len(values) != len(COUNTER_NAMES):
        raise CaptureError(
            f"expected {len(COUNTER_NAMES)} counters, got {len(values)}"
        )
    counters = {name: int(value) for name, value in zip(COUNTER_NAMES, values)}
    if any(value < 0 for value in counters.values()):
        raise CaptureError("capture counters cannot be negative")
    return counters


def _expected_masks(num_layers: int) -> tuple[int, int]:
    low_bits = min(num_layers, 64)
    high_bits = max(0, num_layers - 64)
    low = (1 << low_bits) - 1 if low_bits else 0
    high = (1 << high_bits) - 1 if high_bits else 0
    return low, high


def records_from_snapshot(
    snapshot: CaptureSnapshot,
    config: CaptureConfig,
    request_keys: Mapping[int, str],
    provenance: CaptureProvenance,
    *,
    minimum_rounds: int = 1,
) -> list[dict[str, Any]]:
    """Validate and convert a synchronized host snapshot to canonical records."""

    config.validate()
    counters = _counter_map(snapshot.counters)
    fatal = {name: counters[name] for name in FATAL_COUNTER_NAMES if counters[name]}
    if fatal:
        raise CaptureError(f"capture has fatal drop/overflow counters: {fatal}")
    claimed = counters["rounds_claimed"]
    completed = counters["rounds_completed"]
    if claimed != completed:
        raise CaptureError(
            f"incomplete capture: claimed={claimed}, completed={completed}"
        )
    if not 0 <= claimed <= config.capacity_rounds:
        raise CaptureError(f"claimed round count is outside capacity: {claimed}")
    if minimum_rounds < 1 or minimum_rounds > config.capacity_rounds:
        raise ValueError("minimum_rounds must be inside configured capacity")
    if claimed < minimum_rounds:
        raise CaptureError(
            f"capture has {claimed} rounds; artifact requires {minimum_rounds}"
        )
    if len(snapshot.routes) < claimed:
        raise CaptureError("route snapshot is shorter than claimed rounds")
    if len(snapshot.metadata) < claimed or len(snapshot.layer_masks) < claimed:
        raise CaptureError("metadata snapshot is shorter than claimed rounds")
    if len(snapshot.rejected_tokens) < claimed:
        raise CaptureError("rejection snapshot is shorter than claimed rounds")

    provenance_dict = provenance.as_dict()
    expected_masks = _expected_masks(config.num_layers)
    seen_rounds: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    for slot in range(claimed):
        metadata = snapshot.metadata[slot]
        if len(metadata) != METADATA_COLUMNS:
            raise CaptureError(f"slot {slot}: malformed metadata")
        request_slot, round_index, width, phase, flags = map(int, metadata)
        if phase != MODEL_ROLE_TARGET:
            raise CaptureError(f"slot {slot}: non-target phase {phase}")
        if flags != 0:
            raise CaptureError(f"slot {slot}: device validation flags={flags}")
        if width not in SUPPORTED_WIDTHS:
            raise CaptureError(f"slot {slot}: unsupported width {width}")
        if request_slot not in request_keys or not request_keys[request_slot]:
            raise CaptureError(f"slot {slot}: unknown request slot {request_slot}")
        if round_index < 0:
            raise CaptureError(f"slot {slot}: negative round index")
        rejected_tokens = int(snapshot.rejected_tokens[slot])
        if rejected_tokens == -1:
            raise CaptureError(
                f"slot {slot}: missing rejection association"
            )
        if not 0 <= rejected_tokens <= width - 1:
            raise CaptureError(
                f"slot {slot}: rejected token count {rejected_tokens} "
                f"is outside [0, {width - 1}]"
            )
        accepted_prefix_tokens = width - 1 - rejected_tokens
        identity = (request_slot, round_index)
        if identity in seen_rounds:
            raise CaptureError(f"slot {slot}: duplicate request/round {identity}")
        seen_rounds.add(identity)

        masks = snapshot.layer_masks[slot]
        # CUDA atomics update these int64 tensors as uint64 bitsets.  A full
        # low word therefore materializes through torch as signed ``-1``.
        # Compare the bit pattern, not Python's signed integer value.
        observed_masks = tuple(int(value) & UINT64_MASK for value in masks)
        if len(masks) != 2 or observed_masks != expected_masks:
            raise CaptureError(f"slot {slot}: incomplete 75-layer mask {masks}")
        round_routes = snapshot.routes[slot]
        if len(round_routes) != config.num_layers:
            raise CaptureError(f"slot {slot}: expected 75 routed layers")

        layers: list[dict[str, Any]] = []
        for layer_index, layer_routes in enumerate(round_routes):
            if len(layer_routes) < width:
                raise CaptureError(
                    f"slot {slot} layer {layer_index}: fewer than Q{width} positions"
                )
            positions: list[dict[str, list[int]]] = []
            for position_index in range(width):
                expert_ids = list(map(int, layer_routes[position_index]))
                if len(expert_ids) != config.topk:
                    raise CaptureError(
                        f"slot {slot} layer {layer_index} position "
                        f"{position_index}: expected top-8"
                    )
                if any(
                    expert < 0 or expert >= config.num_experts
                    for expert in expert_ids
                ):
                    raise CaptureError(
                        f"slot {slot} layer {layer_index} position "
                        f"{position_index}: expert outside [0, 255]"
                    )
                if len(set(expert_ids)) != config.topk:
                    raise CaptureError(
                        f"slot {slot} layer {layer_index} position "
                        f"{position_index}: duplicate expert"
                    )
                positions.append({"expert_ids": expert_ids})
            layers.append({"layer": layer_index, "positions": positions})

        records.append(
            {
                "schema": SCHEMA,
                "request_key": request_keys[request_slot],
                "request_slot": request_slot,
                "round": round_index,
                "width": width,
                "phase": "target",
                "capture_slot": slot,
                "accepted_prefix_tokens": accepted_prefix_tokens,
                "rejected_tokens": rejected_tokens,
                "provenance": provenance_dict,
                "layers": layers,
            }
        )
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write a completed host snapshot; never call this from model execution."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
    temporary.replace(path)


class TargetRouteCapture:
    """Own the fixed device arena and dispatch the graph-captured CUDA op."""

    def __init__(
        self,
        *,
        torch_module: Any,
        device: Any,
        config: CaptureConfig = CaptureConfig(),
    ) -> None:
        config.validate()
        self._torch = torch_module
        self._device = device
        self.config = config
        torch = torch_module
        self.routes = torch.empty(
            (
                config.capacity_rounds,
                config.num_layers,
                config.max_width,
                config.topk,
            ),
            dtype=torch.int16,
            device=device,
        )
        self.metadata = torch.zeros(
            (config.capacity_rounds, METADATA_COLUMNS),
            dtype=torch.int64,
            device=device,
        )
        self.layer_masks = torch.zeros(
            (config.capacity_rounds, 2), dtype=torch.int64, device=device
        )
        # [request_slot, model_role, active_capture_slot, armed].  Request
        # context is updated outside graph/timed execution; the graph reads it
        # on replay.  An unarmed graph node is an intentional no-op.
        self.stream_control = torch.tensor(
            [
                [-1, MODEL_ROLE_TARGET, -1, 0]
                for _ in range(config.max_stream_slots)
            ],
            dtype=torch.int64,
            device=device,
        )
        self.request_rounds = torch.zeros(
            config.max_request_slots, dtype=torch.int64, device=device
        )
        self.counters = torch.zeros(
            len(COUNTER_NAMES), dtype=torch.int64, device=device
        )
        # One exact rejection count is associated after sampling with every
        # claimed target-route slot. ``-1`` is an unambiguous missing sentinel.
        self.rejected_tokens = torch.full(
            (config.capacity_rounds,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._request_keys: dict[int, str] = {}

    def begin_request(
        self,
        *,
        request_slot: int,
        request_key: str,
        stream_slot: int = 0,
        model_role: str = "target",
    ) -> None:
        """Set request context outside the timed region before graph replay."""

        if model_role != "target":
            raise CaptureError("route capture must never be bound to MTP/draft")
        if not 0 <= request_slot < self.config.max_request_slots:
            raise CaptureError("request_slot is outside the preallocated table")
        if not 0 <= stream_slot < self.config.max_stream_slots:
            raise CaptureError("stream_slot is outside the preallocated table")
        if not request_key:
            raise CaptureError("request_key must be a non-empty salted identifier")
        prior = self._request_keys.get(request_slot)
        if prior is not None and prior != request_key:
            raise CaptureError(
                "request slots cannot be reused before draining/resetting capture"
            )
        self._request_keys[request_slot] = request_key
        # This host-to-device update is deliberately outside timed execution.
        context = self._torch.tensor(
            [request_slot, MODEL_ROLE_TARGET, -1, 1],
            dtype=self._torch.int64,
            device=self._device,
        )
        self.stream_control[stream_slot].copy_(context, non_blocking=False)

    def disarm(self, *, stream_slot: int = 0) -> None:
        """Disable collection outside timed execution without changing graphs."""

        if not 0 <= stream_slot < self.config.max_stream_slots:
            raise CaptureError("stream_slot is outside the preallocated table")
        context = self._torch.tensor(
            [-1, MODEL_ROLE_TARGET, -1, 0],
            dtype=self._torch.int64,
            device=self._device,
        )
        self.stream_control[stream_slot].copy_(context, non_blocking=False)

    def record_target_routes(
        self,
        topk_ids: Any,
        *,
        layer_index: int,
        width: int,
        stream_slot: int = 0,
    ) -> None:
        """Record one routed layer without allocating or reading device data.

        This method belongs immediately after target
        ``FusedMoERouter.select_experts()``.  The returned ``topk_ids`` tensor
        from that call must be passed directly, before B12X consumes it.
        """

        if width not in SUPPORTED_WIDTHS:
            raise CaptureError(f"only Q5/Q6 are supported, got Q{width}")
        if not 0 <= layer_index < self.config.num_layers:
            raise CaptureError(f"layer_index outside [0, 74]: {layer_index}")
        if not 0 <= stream_slot < self.config.max_stream_slots:
            raise CaptureError("stream_slot is outside the preallocated table")
        # Tensor/dtype/device/shape validation lives in the C++ dispatcher.
        # Keeping it there avoids Python tuple/string/tensor construction in
        # the successful model-layer path and fails before any kernel launch.
        self._torch.ops.sparkring_target_route_capture.record(
            topk_ids,
            self.routes,
            self.metadata,
            self.layer_masks,
            self.stream_control,
            self.request_rounds,
            self.counters,
            layer_index,
            width,
            stream_slot,
            self.config.num_layers,
            self.config.num_experts,
        )

    def record_rejection(
        self,
        num_sampled: Any,
        num_rejected: Any,
        *,
        stream_slot: int = 0,
    ) -> None:
        """Associate one sampler result with the latest target-route slot.

        Both inputs remain device tensors. Shape, dtype, route/sample ordering,
        and the ``sampled + rejected == width`` invariant are validated by the
        CUDA dispatcher/kernel without a host read or synchronization.
        """

        if not 0 <= stream_slot < self.config.max_stream_slots:
            raise CaptureError("stream_slot is outside the preallocated table")
        self._torch.ops.sparkring_target_route_capture.record_rejection(
            num_sampled,
            num_rejected,
            self.rejected_tokens,
            self.metadata,
            self.stream_control,
            self.counters,
            stream_slot,
        )

    def make_base_router_callback(
        self,
        *,
        routed_layer_index: int,
        model_role: str,
        stream_slot: int = 0,
    ) -> Any:
        """Make the existing BaseRouter ``capture_fn(topk_ids)`` callback."""

        if model_role != "target":
            raise CaptureError("refusing to bind route capture to MTP/draft")
        if not 0 <= routed_layer_index < self.config.num_layers:
            raise CaptureError("routed_layer_index must be in [0, 74]")
        if not 0 <= stream_slot < self.config.max_stream_slots:
            raise CaptureError("stream_slot is outside the preallocated table")

        def capture_fn(topk_ids: Any) -> None:
            width = topk_ids.shape[0]
            # BaseRouter calls the callback for profile, prefill, and Q1
            # forwards too.  Those are outside this bounded Q5/Q6 artifact and
            # must remain unaffected, including while the arena is disarmed.
            if width != 5 and width != 6:
                return
            self.record_target_routes(
                topk_ids,
                layer_index=routed_layer_index,
                width=width,
                stream_slot=stream_slot,
            )

        return capture_fn

    def drain_jsonl(
        self,
        path: Path,
        provenance: CaptureProvenance,
        *,
        timed_execution_complete: bool,
    ) -> dict[str, int]:
        """Synchronize, copy, validate, and write outside model execution."""

        if not timed_execution_complete:
            raise CaptureError("refusing to drain inside a timed execution window")
        if self._torch.cuda.is_current_stream_capturing():
            raise CaptureError("refusing to drain during CUDA graph capture")
        self._torch.cuda.synchronize(self._device)
        snapshot = CaptureSnapshot(
            routes=self.routes.cpu().tolist(),
            metadata=self.metadata.cpu().tolist(),
            layer_masks=self.layer_masks.cpu().tolist(),
            counters=self.counters.cpu().tolist(),
            rejected_tokens=self.rejected_tokens.cpu().tolist(),
        )
        records = records_from_snapshot(
            snapshot,
            self.config,
            self._request_keys,
            provenance,
            minimum_rounds=100,
        )
        write_jsonl(Path(path), records)
        return _counter_map(snapshot.counters)

    def read_counters(
        self, *, timed_execution_complete: bool
    ) -> dict[str, int]:
        """Expose overflow/drop state after, never during, measured execution."""

        if not timed_execution_complete:
            raise CaptureError("refusing to read counters in a timed window")
        if self._torch.cuda.is_current_stream_capturing():
            raise CaptureError("refusing to read counters during graph capture")
        self._torch.cuda.synchronize(self._device)
        return _counter_map(self.counters.cpu().tolist())


def salted_request_key(request_id: str, salt: bytes) -> str:
    """Create a stable non-reversible request key before capture begins."""

    if not request_id or len(salt) < 16:
        raise ValueError("request_id and at least 16 salt bytes are required")
    return hashlib.sha256(salt + request_id.encode("utf-8")).hexdigest()[:24]


def cuda_source_path() -> Path:
    return Path(__file__).with_name("target_route_capture_cuda.cu")


def load_cuda_extension(torch_module: Any, *, name: str) -> Any:
    """Build/load the isolated custom op during image preparation, not serving."""

    if not name:
        raise ValueError("an immutable extension name is required")
    cpp_extension = importlib.import_module("torch.utils.cpp_extension")
    if cpp_extension.__package__.split(".", maxsplit=1)[0] != torch_module.__name__:
        raise RuntimeError("CUDA extension loader belongs to a different torch")
    return cpp_extension.load(
        name=name,
        sources=[str(cuda_source_path())],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True,
        is_python_module=False,
        verbose=False,
    )
