"""Opt-in model-serving seam for SparkCache CUDA placement.

This module does not install itself into the vLLM connector.  It layers
artifact attestation, registered-tensor validation, and parked-request
semantics on the canonical ctypes ABI binding in ``spark_cache_native``.
The caller still owns manifest quorum, scheduler parking, block reclamation,
and verified-or-recompute handling.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
from collections.abc import Mapping, Sequence
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any, Callable

try:
    import spark_cache_native as native
except ModuleNotFoundError as error:
    if error.name != "spark_cache_native":
        raise
    # Installed-package import path. Deployment source mounts also expose the
    # package directory directly, where the top-level import above resolves.
    from sparkcache import spark_cache_native as native


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1


class NativePlacementContractError(RuntimeError):
    """The Python/CUDA integration contract is not safe to execute."""


class NativePlacementCallError(RuntimeError):
    """An attested SparkCache CUDA function returned a non-success status."""

    def __init__(self, action: str, status: int, detail: str = "") -> None:
        message = (
            f"SparkCache CUDA placement did not complete {action}: status={status}"
        )
        if detail:
            message += f": {detail}"
        super().__init__(message)
        self.action = action
        self.status = status
        self.detail = detail


class ArenaMode(IntEnum):
    MAPPED_HOST = native.ARENA_MAPPED_HOST
    MANAGED = native.ARENA_MANAGED
    STAGED_DEVICE = native.ARENA_STAGED_DEVICE


class RecordKind(IntEnum):
    TARGET_CKV = native.RECORD_TARGET_CKV
    SPARSE_INDEXER = native.RECORD_SPARSE_INDEXER
    MTP_DRAFT_KV = native.RECORD_MTP_DRAFT_KV
    BOUNDARY_HIDDEN = native.RECORD_BOUNDARY_HIDDEN


_RECORD_KINDS = {
    "target_ckv": RecordKind.TARGET_CKV,
    "sparse_indexer": RecordKind.SPARSE_INDEXER,
    "mtp_draft_kv": RecordKind.MTP_DRAFT_KV,
    "boundary_hidden": RecordKind.BOUNDARY_HIDDEN,
}


class RestoreState(Enum):
    PARKED = auto()
    FINISHED = auto()
    ABORTED = auto()


# Public aliases keep the adapter surface explicit without defining a second
# ctypes ABI that could drift from the SparkCache CUDA placement contract.
SparkCacheDestinationDescriptor = native.DestinationDescriptor
SparkCacheChunkDescriptor = native.ChunkDescriptor
SparkCachePlacementStats = native.PlacementStats
SparkCacheArenaView = native.ArenaView


def _u32(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativePlacementContractError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if not minimum <= value <= _U32_MAX:
        raise NativePlacementContractError(
            f"{label} must be in [{minimum}, {_U32_MAX}]"
        )
    return value


def _u64(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativePlacementContractError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if not minimum <= value <= _U64_MAX:
        raise NativePlacementContractError(
            f"{label} must be in [{minimum}, {_U64_MAX}]"
        )
    return value


def _tensor_descriptor(
    name: str,
    tensor: Any,
    *,
    bytes_per_token: int,
    record_kind: RecordKind,
    source_layer_ordinal: int,
) -> native.DestinationDescriptor:
    if getattr(getattr(tensor, "device", None), "type", None) != "cuda":
        raise NativePlacementContractError(
            f"registered layer {name} must be a CUDA tensor"
        )
    shape = tuple(getattr(tensor, "shape", ()))
    if len(shape) < 3 or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in shape
    ):
        raise NativePlacementContractError(
            f"registered layer {name} must have a positive rank-3+ shape"
        )
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if not callable(is_contiguous) or is_contiguous() is not True:
        raise NativePlacementContractError(
            f"registered layer {name} must be contiguous"
        )
    element_size_method = getattr(tensor, "element_size", None)
    if not callable(element_size_method):
        raise NativePlacementContractError(
            f"registered layer {name} has no element size"
        )
    element_size = _u32(
        int(element_size_method()), f"{name} element size", positive=True
    )
    row_elements = 1
    for dimension in shape[2:]:
        row_elements *= dimension
    actual_row_bytes = row_elements * element_size
    if actual_row_bytes != bytes_per_token:
        raise NativePlacementContractError(
            f"registered layer {name} bytes-per-token mismatch:"
            f" plan={bytes_per_token}, tensor={actual_row_bytes}"
        )
    stride_method = getattr(tensor, "stride", None)
    if not callable(stride_method):
        raise NativePlacementContractError(
            f"registered layer {name} has no stride metadata"
        )
    strides = tuple(stride_method())
    if len(strides) != len(shape):
        raise NativePlacementContractError(
            f"registered layer {name} has malformed stride metadata"
        )
    row_stride = _u32(
        int(strides[1]) * element_size,
        f"{name} destination row stride",
        positive=True,
    )
    if row_stride != actual_row_bytes:
        raise NativePlacementContractError(
            f"registered layer {name} row stride is not tightly packed"
        )
    data_ptr_method = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr_method):
        raise NativePlacementContractError(
            f"registered layer {name} has no data pointer"
        )
    pointer = _u64(int(data_ptr_method()), f"{name} data pointer", positive=True)
    rows = _u64(shape[0] * shape[1], f"{name} destination rows", positive=True)
    return native.DestinationDescriptor(
        pointer,
        rows,
        row_stride,
        bytes_per_token,
        int(record_kind),
        source_layer_ordinal,
    )


def build_destination_descriptors(
    plans: Sequence[Any],
    registered_tensors: Mapping[str, Any],
) -> tuple[native.DestinationDescriptor, ...]:
    """Build ABI-v1 destinations in the codec's layer-major order.

    ABI v1 derives a source layer offset as ``ordinal * rows * row_width``.
    Mixed row widths inside one record kind therefore cannot be represented
    and are rejected rather than scattered incorrectly.
    """

    plan_tuple = tuple(plans)
    if not plan_tuple:
        raise NativePlacementContractError("at least one layer plan is required")
    names = [getattr(plan, "name", None) for plan in plan_tuple]
    if any(not isinstance(name, str) or not name for name in names):
        raise NativePlacementContractError("every layer plan needs a nonempty name")
    if len(set(names)) != len(names):
        raise NativePlacementContractError("layer plan names must be unique")
    tensor_names = set(registered_tensors)
    if tensor_names != set(names):
        missing = sorted(set(names) - tensor_names)
        extra = sorted(tensor_names - set(names))
        raise NativePlacementContractError(
            "registered tensor set does not match layer plans:"
            f" missing={missing}, extra={extra}"
        )

    kind_widths: dict[str, int] = {}
    ordinals: dict[str, int] = {}
    descriptors = []
    for plan in plan_tuple:
        kind_name = getattr(plan, "record_kind", None)
        if kind_name not in _RECORD_KINDS:
            raise NativePlacementContractError(
                f"layer {plan.name} has unsupported record kind {kind_name!r}"
            )
        width = _u32(
            getattr(plan, "bytes_per_token", None),
            f"{plan.name} bytes_per_token",
            positive=True,
        )
        prior_width = kind_widths.setdefault(kind_name, width)
        if prior_width != width:
            raise NativePlacementContractError(
                f"ABI v1 requires equal bytes_per_token within {kind_name}:"
                f" saw {prior_width} and {width}"
            )
        ordinal = ordinals.get(kind_name, 0)
        descriptors.append(
            _tensor_descriptor(
                plan.name,
                registered_tensors[plan.name],
                bytes_per_token=width,
                record_kind=_RECORD_KINDS[kind_name],
                source_layer_ordinal=ordinal,
            )
        )
        ordinals[kind_name] = ordinal + 1
    return tuple(descriptors)


def build_page_destination_descriptors(
    layout: Any,
    registered_tensors: Mapping[str, Any],
) -> tuple[native.PageDestinationDescriptor, ...]:
    """Build stable opaque-page destinations in PageLayout order."""

    groups = tuple(getattr(layout, "groups", ()) or ())
    layers = tuple(layer for group in groups for layer in group.layers)
    names = {layer.name for layer in layers}
    if not groups or set(registered_tensors) != names:
        raise NativePlacementContractError(
            "registered tensor set does not match page layout"
        )
    descriptors = []
    for group_index, group in enumerate(groups):
        for layer in group.layers:
            tensor = registered_tensors[layer.name]
            if getattr(getattr(tensor, "device", None), "type", None) != "cuda":
                raise NativePlacementContractError(
                    f"registered page layer {layer.name} must be a CUDA tensor"
                )
            shape = tuple(getattr(tensor, "shape", ()) or ())
            expected_shape = tuple(layer.page_shape)
            if len(shape) < 2 or tuple(shape[1:]) != expected_shape:
                raise NativePlacementContractError(
                    f"registered page layer {layer.name} shape differs from layout"
                )
            element_size = int(tensor.element_size())
            stride = tuple(tensor.stride())
            expected_stride = 1
            for dimension, actual_stride in zip(
                reversed(shape[1:]),
                reversed(stride[1:]),
            ):
                if int(actual_stride) != expected_stride:
                    raise NativePlacementContractError(
                        f"registered page layer {layer.name} page interior"
                        " is not contiguous"
                    )
                expected_stride *= int(dimension)
            page_stride = _u32(
                int(stride[0]) * element_size,
                f"{layer.name} page stride",
                positive=True,
            )
            page_bytes = _u32(
                int(layer.bytes_per_page),
                f"{layer.name} bytes per page",
                positive=True,
            )
            if page_stride < page_bytes:
                raise NativePlacementContractError(
                    f"registered page layer {layer.name} page stride overlaps"
                )
            pointer = _u64(
                int(tensor.data_ptr()),
                f"{layer.name} data pointer",
                positive=True,
            )
            descriptors.append(
                native.PageDestinationDescriptor(
                    pointer,
                    _u64(int(shape[0]), f"{layer.name} page capacity", positive=True),
                    page_stride,
                    page_bytes,
                    group_index,
                    0,
                )
            )
    return tuple(descriptors)


class NativePlacementLibrary:
    """SHA-256 attestation in front of the canonical strict ABI loader."""

    def __init__(
        self,
        artifact_path: Path,
        artifact_sha256: str,
        cdll: Any,
        abi_info: native.AbiInfo,
    ) -> None:
        self.artifact_path = artifact_path
        self.artifact_sha256 = artifact_sha256
        self.cdll = cdll
        self.abi_info = abi_info

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        *,
        expected_sha256: str,
        binding_loader: Callable[[str], tuple[Any, native.AbiInfo]] = (
            native.load_library
        ),
    ) -> NativePlacementLibrary:
        if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
            expected_sha256
        ):
            raise NativePlacementContractError(
                "expected SHA-256 must be exactly 64 lowercase hex characters"
            )
        try:
            resolved = Path(artifact_path).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise NativePlacementContractError(
                f"SparkCache CUDA placement artifact is unavailable: {artifact_path}"
            ) from error
        if not resolved.is_file():
            raise NativePlacementContractError(
                f"SparkCache CUDA placement artifact is not a regular file: {resolved}"
            )
        actual_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise NativePlacementContractError(
                "SparkCache CUDA placement artifact SHA-256 mismatch:"
                f" expected={expected_sha256}, actual={actual_sha256}"
            )
        try:
            cdll, abi_info = binding_loader(str(resolved))
        except (OSError, native.NativePlacementError) as error:
            raise NativePlacementContractError(
                f"SparkCache CUDA binding rejected attested artifact: {error}"
            ) from error
        if abi_info.abi_version != native.ABI_VERSION:
            raise NativePlacementContractError(
                f"canonical binding returned ABI {abi_info.abi_version},"
                f" expected {native.ABI_VERSION}"
            )
        return cls(resolved, actual_sha256, cdll, abi_info)

    def detail(self, handle: ctypes.c_void_p | None) -> str:
        if handle is not None and handle.value:
            return native.handle_error(self.cdll, handle)
        return native.runtime_error(self.cdll)

    def check(
        self,
        result: int,
        action: str,
        handle: ctypes.c_void_p | None = None,
    ) -> None:
        if result == native.STATUS_OK:
            return
        try:
            native.check(self.cdll, result, action, handle)
        except native.NativePlacementError as error:
            raise NativePlacementCallError(action, result, str(error)) from error
        raise NativePlacementCallError(action, result)


class NativePlacementAdapter:
    """One CUDA placer with one serialized restore transaction."""

    def __init__(
        self,
        library: NativePlacementLibrary,
        handle: ctypes.c_void_p,
        max_destinations: int,
        max_slots: int,
        arena_mode: ArenaMode,
    ) -> None:
        self.library = library
        self._handle = handle
        self._max_destinations = max_destinations
        self._max_slots = max_slots
        self._arena_mode = arena_mode
        self._configured = False
        self._page_configured = False
        self._closed = False
        self._active: ParkedRestore | None = None
        self._descriptors: tuple[native.DestinationDescriptor, ...] = ()
        self._page_descriptors: tuple[native.PageDestinationDescriptor, ...] = ()

    @classmethod
    def create(
        cls,
        library: NativePlacementLibrary,
        *,
        arena_mode: ArenaMode,
        arena_bytes: int,
        max_destinations: int,
        max_slots: int,
        max_chunks_per_slab: int,
        device_ordinal: int,
        flags: int = 0,
    ) -> NativePlacementAdapter:
        if not isinstance(library, NativePlacementLibrary):
            raise NativePlacementContractError(
                "library must be an attested NativePlacementLibrary"
            )
        try:
            mode = ArenaMode(arena_mode)
        except (TypeError, ValueError) as error:
            raise NativePlacementContractError(
                "unsupported SparkCache CUDA placement arena mode"
            ) from error
        arena_bytes = _u64(arena_bytes, "arena_bytes", positive=True)
        max_destinations = _u32(max_destinations, "max_destinations", positive=True)
        max_slots = _u32(max_slots, "max_slots", positive=True)
        max_chunks_per_slab = _u32(
            max_chunks_per_slab, "max_chunks_per_slab", positive=True
        )
        flags = _u32(flags, "flags")
        if (
            isinstance(device_ordinal, bool)
            or not isinstance(device_ordinal, int)
            or not -(1 << 31) <= device_ordinal < (1 << 31)
        ):
            raise NativePlacementContractError(
                "device_ordinal must fit a signed 32-bit integer"
            )
        config = native.PlacementConfig(
            native.ABI_VERSION,
            int(mode),
            arena_bytes,
            max_destinations,
            max_slots,
            max_chunks_per_slab,
            device_ordinal,
            flags,
            (ctypes.c_uint32 * 3)(0, 0, 0),
        )
        handle = ctypes.c_void_p()
        result = int(
            library.cdll.spark_cache_placement_create(
                ctypes.byref(config), ctypes.byref(handle)
            )
        )
        library.check(result, "create placement")
        if not handle.value:
            raise NativePlacementContractError(
                "SparkCache CUDA placement creation returned a null handle"
            )
        return cls(library, handle, max_destinations, max_slots, mode)

    def _ensure_open(self) -> None:
        if self._closed or not self._handle.value:
            raise NativePlacementContractError("SparkCache CUDA placement is closed")

    def _call(self, action: str, function: Any, *arguments: Any) -> None:
        self._ensure_open()
        result = int(function(*arguments))
        self.library.check(result, action, self._handle)

    def configure(
        self,
        plans: Sequence[Any],
        registered_tensors: Mapping[str, Any],
    ) -> tuple[native.DestinationDescriptor, ...]:
        self._ensure_open()
        if self._active is not None:
            raise NativePlacementContractError(
                "cannot reconfigure while a restore is active"
            )
        descriptors = build_destination_descriptors(plans, registered_tensors)
        if len(descriptors) > self._max_destinations:
            raise NativePlacementContractError(
                f"destination count {len(descriptors)} exceeds configured"
                f" maximum {self._max_destinations}"
            )
        native_descriptors = (native.DestinationDescriptor * len(descriptors))(
            *descriptors
        )
        self._call(
            "configure destinations",
            self.library.cdll.spark_cache_placement_configure_destinations,
            self._handle,
            native_descriptors,
            len(descriptors),
        )
        self._descriptors = descriptors
        self._configured = True
        return descriptors

    def configure_pages(
        self,
        layout: Any,
        registered_tensors: Mapping[str, Any],
    ) -> tuple[native.PageDestinationDescriptor, ...]:
        self._ensure_open()
        if self._active is not None:
            raise NativePlacementContractError(
                "cannot reconfigure while a restore is active"
            )
        descriptors = build_page_destination_descriptors(layout, registered_tensors)
        if len(descriptors) > self._max_destinations:
            raise NativePlacementContractError(
                "page destination count exceeds configured maximum"
            )
        native_descriptors = (native.PageDestinationDescriptor * len(descriptors))(
            *descriptors
        )
        self._call(
            "configure page destinations",
            self.library.cdll.spark_cache_placement_configure_page_destinations,
            self._handle,
            native_descriptors,
            len(descriptors),
        )
        self._page_descriptors = descriptors
        self._page_configured = True
        return descriptors

    def begin_parked_restore(
        self,
        request_id: str,
        slots: Sequence[int],
    ) -> ParkedRestore:
        self._ensure_open()
        if not self._configured:
            raise NativePlacementContractError(
                "destinations must be configured before restore"
            )
        if self._active is not None:
            raise NativePlacementContractError(
                "SparkCache CUDA restore is already active for"
                f" {self._active.request_id}"
            )
        if not isinstance(request_id, str) or not request_id:
            raise NativePlacementContractError("request_id must be a nonempty string")
        slot_tuple = tuple(_u32(slot, "slot") for slot in slots)
        if not slot_tuple:
            raise NativePlacementContractError("restore slot vector must not be empty")
        if len(slot_tuple) > self._max_slots:
            raise NativePlacementContractError(
                f"slot count {len(slot_tuple)} exceeds configured maximum"
                f" {self._max_slots}"
            )
        if len(set(slot_tuple)) != len(slot_tuple):
            raise NativePlacementContractError("restore slots must be unique")
        native_slots = (ctypes.c_uint32 * len(slot_tuple))(*slot_tuple)
        self._call(
            "begin restore",
            self.library.cdll.spark_cache_placement_begin_restore,
            self._handle,
            native_slots,
            len(slot_tuple),
        )
        restore = ParkedRestore(self, request_id, slot_tuple)
        self._active = restore
        return restore

    def begin_parked_page_restore(
        self,
        request_id: str,
        group_slots: Sequence[Sequence[int]],
        *,
        snapshot_bytes: int,
    ) -> ParkedRestore:
        self._ensure_open()
        if not self._page_configured:
            raise NativePlacementContractError(
                "page destinations must be configured before restore"
            )
        if self._active is not None:
            raise NativePlacementContractError(
                "SparkCache CUDA restore is already active for"
                f" {self._active.request_id}"
            )
        if not isinstance(request_id, str) or not request_id:
            raise NativePlacementContractError("request_id must be a nonempty string")
        flattened = []
        groups = []
        for supplied in group_slots:
            slots = tuple(_u32(slot, "page slot") for slot in supplied)
            if not slots or len(set(slots)) != len(slots):
                raise NativePlacementContractError(
                    "page slots must be nonempty and unique within each group"
                )
            groups.append(native.PageGroupDescriptor(len(flattened), len(slots), 0, 0))
            flattened.extend(slots)
        if not groups or len(flattened) > self._max_slots:
            raise NativePlacementContractError(
                "page slot vector exceeds configured maximum"
            )
        snapshot_bytes = _u64(snapshot_bytes, "snapshot_bytes", positive=True)
        native_groups = (native.PageGroupDescriptor * len(groups))(*groups)
        native_slots = (ctypes.c_uint32 * len(flattened))(*flattened)
        self._call(
            "begin page restore",
            self.library.cdll.spark_cache_placement_begin_page_restore,
            self._handle,
            native_groups,
            len(groups),
            native_slots,
            len(flattened),
            snapshot_bytes,
        )
        restore = ParkedRestore(self, request_id, tuple(flattened))
        self._active = restore
        return restore

    def _release(self, restore: ParkedRestore) -> None:
        if self._active is restore:
            self._active = None

    def close(self) -> None:
        if self._closed:
            return
        abort_error = None
        if self._active is not None:
            try:
                self._active.abort()
            except NativePlacementCallError as error:
                abort_error = error
        handle = self._handle
        self._handle = ctypes.c_void_p()
        self._closed = True
        if handle.value:
            self.library.cdll.spark_cache_placement_destroy(handle)
        if abort_error is not None:
            raise abort_error

    def __enter__(self) -> NativePlacementAdapter:
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class ParkedRestore:
    """CUDA transaction whose request is resumable only after ``finish``."""

    def __init__(
        self,
        adapter: NativePlacementAdapter,
        request_id: str,
        slots: tuple[int, ...],
    ) -> None:
        self._adapter = adapter
        self.request_id = request_id
        self.slots = slots
        self.state = RestoreState.PARKED

    @property
    def can_resume(self) -> bool:
        return self.state is RestoreState.FINISHED

    @property
    def needs_recompute(self) -> bool:
        return self.state is RestoreState.ABORTED

    def _require_parked(self) -> None:
        if self.state is not RestoreState.PARKED:
            raise NativePlacementContractError(
                f"restore {self.request_id} is already {self.state.name.lower()}"
            )
        if self._adapter._active is not self:
            raise NativePlacementContractError(
                f"restore {self.request_id} no longer owns SparkCache CUDA placement"
            )

    def _abort_after_failure(self) -> None:
        if self.state is not RestoreState.PARKED:
            return
        try:
            self._adapter.library.cdll.spark_cache_placement_abort_restore(
                self._adapter._handle
            )
        finally:
            self.state = RestoreState.ABORTED
            self._adapter._release(self)

    def _call(self, action: str, function: Any, *arguments: Any) -> None:
        self._require_parked()
        try:
            self._adapter._call(action, function, *arguments)
        except NativePlacementCallError:
            self._abort_after_failure()
            raise

    def acquire_arena(self, arena_index: int) -> native.ArenaView:
        arena_index = _u32(arena_index, "arena_index")
        if arena_index >= native.ARENA_COUNT:
            raise NativePlacementContractError(
                f"arena_index must be below {native.ARENA_COUNT}"
            )
        view = native.ArenaView()
        self._call(
            "acquire arena",
            self._adapter.library.cdll.spark_cache_placement_acquire_arena_view,
            self._adapter._handle,
            arena_index,
            ctypes.byref(view),
        )
        if (
            not view.host_address
            or not view.capacity_bytes
            or view.arena_index != arena_index
            or view.arena_mode != int(self._adapter._arena_mode)
        ):
            self._abort_after_failure()
            raise NativePlacementContractError(
                "SparkCache CUDA arena view disagrees with the requested contract"
            )
        return view

    def parse_verified_chunk(
        self,
        *,
        arena: native.ArenaView,
        arena_used_bytes: int,
        arena_offset_bytes: int,
        encoded_bytes: int,
        expected_logical_start: int,
        dcp_degree: int,
        dcp_rank: int,
        first_slot_index: int,
        required_data_record_mask: int,
    ) -> native.ChunkDescriptor:
        """Parse only after the caller verified the complete encoded SHA-256."""

        self._require_parked()
        if not isinstance(arena, native.ArenaView) or not arena.host_address:
            raise NativePlacementContractError("parser requires an acquired arena view")
        arena_used_bytes = _u64(arena_used_bytes, "arena_used_bytes", positive=True)
        if arena_used_bytes > arena.capacity_bytes:
            raise NativePlacementContractError(
                "arena_used_bytes exceeds acquired arena capacity"
            )
        values = (
            _u64(arena_offset_bytes, "arena_offset_bytes"),
            _u32(encoded_bytes, "encoded_bytes", positive=True),
            _u32(expected_logical_start, "expected_logical_start"),
            _u32(dcp_degree, "dcp_degree", positive=True),
            _u32(dcp_rank, "dcp_rank"),
            _u32(first_slot_index, "first_slot_index"),
            _u32(required_data_record_mask, "required_data_record_mask"),
        )
        output = native.ChunkDescriptor()
        error = ctypes.create_string_buffer(512)
        result = int(
            self._adapter.library.cdll.spark_cache_parse_verified_v1_chunk(
                ctypes.c_void_p(arena.host_address),
                arena_used_bytes,
                *values,
                ctypes.byref(output),
                error,
                len(error),
            )
        )
        if result != native.STATUS_OK:
            detail = error.value.decode("utf-8", errors="replace")
            self._abort_after_failure()
            raise NativePlacementCallError("parse verified chunk", result, detail)
        return output

    def submit_direct_slab(
        self,
        *,
        arena_index: int,
        arena_used_bytes: int,
        chunks: Sequence[native.ChunkDescriptor],
    ) -> None:
        chunk_tuple = tuple(chunks)
        if not chunk_tuple or not all(
            isinstance(item, native.ChunkDescriptor) for item in chunk_tuple
        ):
            raise NativePlacementContractError(
                "direct CUDA slab requires SparkCache chunk descriptors"
            )
        native_chunks = (native.ChunkDescriptor * len(chunk_tuple))(*chunk_tuple)
        self._call(
            "submit direct slab",
            self._adapter.library.cdll.spark_cache_placement_submit_direct_slab,
            self._adapter._handle,
            _u32(arena_index, "arena_index"),
            _u64(arena_used_bytes, "arena_used_bytes", positive=True),
            native_chunks,
            _u32(len(chunk_tuple), "chunk_count", positive=True),
        )

    def submit_page_slab(
        self,
        *,
        arena_index: int,
        arena_used_bytes: int,
        spans: Sequence[native.PageCopySpan],
    ) -> None:
        span_tuple = tuple(spans)
        if not span_tuple or not all(
            isinstance(item, native.PageCopySpan) for item in span_tuple
        ):
            raise NativePlacementContractError("page slab requires page copy spans")
        native_spans = (native.PageCopySpan * len(span_tuple))(*span_tuple)
        self._call(
            "submit page slab",
            self._adapter.library.cdll.spark_cache_placement_submit_page_slab,
            self._adapter._handle,
            _u32(arena_index, "arena_index"),
            _u64(arena_used_bytes, "arena_used_bytes", positive=True),
            native_spans,
            _u32(len(span_tuple), "span_count", positive=True),
        )

    def finish(self) -> native.PlacementStats:
        self._require_parked()
        stats = native.PlacementStats()
        try:
            self._adapter._call(
                "finish restore",
                self._adapter.library.cdll.spark_cache_placement_finish_restore,
                self._adapter._handle,
                ctypes.byref(stats),
            )
        except NativePlacementCallError:
            self._abort_after_failure()
            raise
        self.state = RestoreState.FINISHED
        self._adapter._release(self)
        return stats

    def abort(self) -> None:
        self._require_parked()
        result = int(
            self._adapter.library.cdll.spark_cache_placement_abort_restore(
                self._adapter._handle
            )
        )
        self.state = RestoreState.ABORTED
        self._adapter._release(self)
        self._adapter.library.check(result, "abort restore", self._adapter._handle)

    def __enter__(self) -> ParkedRestore:
        self._require_parked()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.state is RestoreState.PARKED:
            self.abort()


CudaPlacementAdapter = NativePlacementAdapter
CudaPlacementCallError = NativePlacementCallError
CudaPlacementContractError = NativePlacementContractError
CudaPlacementLibrary = NativePlacementLibrary


__all__ = [
    "ArenaMode",
    "CudaPlacementAdapter",
    "CudaPlacementCallError",
    "CudaPlacementContractError",
    "CudaPlacementLibrary",
    "NativePlacementAdapter",
    "NativePlacementCallError",
    "NativePlacementContractError",
    "NativePlacementLibrary",
    "ParkedRestore",
    "RecordKind",
    "RestoreState",
    "SparkCacheArenaView",
    "SparkCacheChunkDescriptor",
    "SparkCacheDestinationDescriptor",
    "SparkCachePlacementStats",
    "build_destination_descriptors",
    "build_page_destination_descriptors",
]
