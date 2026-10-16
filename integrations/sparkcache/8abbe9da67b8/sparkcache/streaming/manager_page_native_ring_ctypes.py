"""Checksum-attested ctypes backend for CUDA manager-page capture."""

from __future__ import annotations

import ctypes
from os import PathLike

from sparkcache.native.python import spark_cache_snapshot_native as abi

from .manager_page_capture import ManagerPageSource
from .native_ring import NativeRingConfig, NativeStatus, RawReadyView, RawTicket


CONTRACT_VERSION = 1
MAX_GROUPS = 64
MAX_SOURCES = 256


class PageCaptureSource(ctypes.Structure):
    _fields_ = [
        ("source_base", ctypes.c_uint64),
        ("source_pages", ctypes.c_uint64),
        ("source_page_stride_bytes", ctypes.c_uint64),
        ("bytes_per_page", ctypes.c_uint32),
        ("group_index", ctypes.c_uint32),
        ("layer_ordinal", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class PageCaptureGroup(ctypes.Structure):
    _fields_ = [
        ("physical_page_offset", ctypes.c_uint32),
        ("page_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class PageCaptureSpan(ctypes.Structure):
    _fields_ = [
        ("destination_offset_bytes", ctypes.c_uint64),
        ("length_bytes", ctypes.c_uint64),
        ("source_index", ctypes.c_uint32),
        ("physical_page_offset", ctypes.c_uint32),
        ("page_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class PageCapturePlan(ctypes.Structure):
    _fields_ = [
        ("used_bytes", ctypes.c_uint64),
        ("span_count", ctypes.c_uint32),
        ("group_count", ctypes.c_uint32),
        ("source_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class PageCaptureSubmission(ctypes.Structure):
    _fields_ = [
        ("context_sequence", ctypes.c_uint64),
        ("logical_start", ctypes.c_uint64),
        ("physical_page_count", ctypes.c_uint32),
        ("group_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class PageCaptureAbiInfo(ctypes.Structure):
    _fields_ = [
        ("contract_version", ctypes.c_uint32),
        ("max_groups", ctypes.c_uint32),
        ("max_sources", ctypes.c_uint32),
        ("sizeof_source", ctypes.c_uint32),
        ("sizeof_group", ctypes.c_uint32),
        ("sizeof_span", ctypes.c_uint32),
        ("sizeof_plan", ctypes.c_uint32),
        ("sizeof_submission", ctypes.c_uint32),
        ("capability_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


def _bind_page_api(library: ctypes.CDLL) -> PageCaptureAbiInfo:
    handle = ctypes.c_void_p
    library.spark_cache_snapshot_query_page_capture_abi.argtypes = [
        ctypes.POINTER(PageCaptureAbiInfo)
    ]
    library.spark_cache_snapshot_query_page_capture_abi.restype = ctypes.c_int
    library.spark_cache_snapshot_configure_page_sources.argtypes = [
        handle,
        ctypes.POINTER(PageCaptureSource),
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    library.spark_cache_snapshot_configure_page_sources.restype = ctypes.c_int
    library.spark_cache_snapshot_try_submit_pages.argtypes = [
        handle,
        ctypes.POINTER(PageCaptureSubmission),
        ctypes.POINTER(PageCaptureGroup),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(abi.SnapshotTicket),
    ]
    library.spark_cache_snapshot_try_submit_pages.restype = ctypes.c_int
    library.spark_cache_snapshot_drain_context.argtypes = [
        handle,
        ctypes.c_uint64,
    ]
    library.spark_cache_snapshot_drain_context.restype = ctypes.c_int
    info = PageCaptureAbiInfo()
    result = int(
        library.spark_cache_snapshot_query_page_capture_abi(ctypes.byref(info))
    )
    if result != abi.STATUS_OK:
        raise abi.NativeSnapshotError(
            f"manager-page ABI query failed: status={result}"
        )
    expected_sizes = {
        "sizeof_source": ctypes.sizeof(PageCaptureSource),
        "sizeof_group": ctypes.sizeof(PageCaptureGroup),
        "sizeof_span": ctypes.sizeof(PageCaptureSpan),
        "sizeof_plan": ctypes.sizeof(PageCapturePlan),
        "sizeof_submission": ctypes.sizeof(PageCaptureSubmission),
    }
    mismatches = {
        name: (getattr(info, name), expected)
        for name, expected in expected_sizes.items()
        if getattr(info, name) != expected
    }
    required = (
        abi.CAP_MANAGER_PAGE_CAPTURE
        | abi.CAP_LOW_PRIORITY_CAPTURE_STREAM
    )
    if (
        info.contract_version != CONTRACT_VERSION
        or info.max_groups != MAX_GROUPS
        or info.max_sources != MAX_SOURCES
        or mismatches
        or info.capability_flags & required != required
    ):
        raise abi.NativeSnapshotError(
            "manager-page capture ABI differs from the Python binding: "
            f"sizes={mismatches} capabilities={info.capability_flags}"
        )
    return info


class CtypesManagerPageRingBackend:
    def __init__(
        self,
        library_path: str | PathLike[str],
        *,
        expected_sha256: str,
    ) -> None:
        library, _ = abi.load_library(
            library_path,
            expected_sha256=expected_sha256,
        )
        _bind_page_api(library)
        self._library = library
        self._handle: ctypes.c_void_p | None = None

    def create(self, config: NativeRingConfig) -> int:
        raw = abi.SnapshotConfig(
            abi_version=abi.ABI_VERSION,
            arena_mode=config.arena_mode,
            slot_bytes=config.slot_bytes,
            slot_count=config.slot_count,
            max_sources=config.max_sources,
            max_rows=config.max_rows,
            device_ordinal=config.device_ordinal,
            flags=0,
        )
        handle = ctypes.c_void_p()
        result = int(
            self._library.spark_cache_snapshot_create(
                ctypes.byref(raw), ctypes.byref(handle)
            )
        )
        if result == NativeStatus.OK:
            if not handle.value:
                raise abi.NativeSnapshotError(
                    "manager-page create returned no handle"
                )
            self._handle = handle
        return result

    def configure_page_sources(
        self,
        sources: tuple[ManagerPageSource, ...],
        group_count: int,
    ) -> int:
        values = (PageCaptureSource * len(sources))(
            *(
                PageCaptureSource(
                    source.source_base,
                    source.source_pages,
                    source.source_page_stride_bytes,
                    source.bytes_per_page,
                    source.group_index,
                    source.layer_ordinal,
                    0,
                )
                for source in sources
            )
        )
        return int(
            self._library.spark_cache_snapshot_configure_page_sources(
                self._require_handle(), values, len(values), group_count
            )
        )

    def submit_pages(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        groups: tuple[tuple[int, ...], ...],
        producer_stream: int,
        used_bytes: int,
    ) -> tuple[int, RawTicket | None]:
        del used_bytes
        flattened = tuple(page for group in groups for page in group)
        offset = 0
        descriptors = []
        for group in groups:
            descriptors.append(PageCaptureGroup(offset, len(group), (0, 0)))
            offset += len(group)
        raw_groups = (PageCaptureGroup * len(descriptors))(*descriptors)
        raw_pages = (ctypes.c_uint32 * len(flattened))(*flattened)
        submission = PageCaptureSubmission(
            context_sequence,
            logical_start,
            len(flattened),
            len(groups),
            0,
            0,
        )
        ticket = abi.SnapshotTicket()
        result = int(
            self._library.spark_cache_snapshot_try_submit_pages(
                self._require_handle(),
                ctypes.byref(submission),
                raw_groups,
                raw_pages,
                producer_stream,
                ctypes.byref(ticket),
            )
        )
        if result != NativeStatus.OK:
            return result, None
        return result, RawTicket(int(ticket.generation), int(ticket.slot_index))

    def poll(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        return self._view("poll", ticket)

    def claim(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        return self._view("claim", ticket)

    def release(self, ticket: RawTicket) -> int:
        raw = abi.SnapshotTicket(ticket.generation, ticket.slot_index)
        return int(
            self._library.spark_cache_snapshot_release(
                self._require_handle(), ctypes.byref(raw)
            )
        )

    def abandon_context(self, context_sequence: int) -> int:
        return int(
            self._library.spark_cache_snapshot_abandon_context(
                self._require_handle(), context_sequence
            )
        )

    def drain_context(self, context_sequence: int) -> int:
        return int(
            self._library.spark_cache_snapshot_drain_context(
                self._require_handle(), context_sequence
            )
        )

    def shutdown(self) -> int:
        return int(
            self._library.spark_cache_snapshot_shutdown(self._require_handle())
        )

    def destroy(self) -> None:
        if self._handle is not None:
            self._library.spark_cache_snapshot_destroy(self._handle)
            self._handle = None

    def status_text(self, status: int) -> str:
        value = self._library.spark_cache_snapshot_status_string(status)
        return "unknown" if value is None else value.decode(errors="replace")

    def _view(
        self, operation: str, ticket: RawTicket
    ) -> tuple[int, RawReadyView | None]:
        raw_ticket = abi.SnapshotTicket(ticket.generation, ticket.slot_index)
        raw_view = abi.SnapshotReadyView()
        function = getattr(
            self._library, f"spark_cache_snapshot_{operation}"
        )
        result = int(
            function(
                self._require_handle(),
                ctypes.byref(raw_ticket),
                ctypes.byref(raw_view),
            )
        )
        if result != NativeStatus.OK:
            return result, None
        return result, RawReadyView(
            payload=abi.ready_memoryview(raw_view),
            capacity_bytes=int(raw_view.capacity_bytes),
            used_bytes=int(raw_view.used_bytes),
            context_sequence=int(raw_view.context_sequence),
            logical_start=int(raw_view.logical_start),
            generation=int(raw_view.generation),
            row_count=int(raw_view.row_count),
            slot_index=int(raw_view.slot_index),
            record_mask=int(raw_view.record_mask),
            state=int(raw_view.state),
            record_offsets=tuple(raw_view.record_offset_bytes),
            record_lengths=tuple(raw_view.record_length_bytes),
        )

    def _require_handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise abi.NativeSnapshotError("manager-page backend has no handle")
        return self._handle


__all__ = [
    "CtypesManagerPageRingBackend",
    "PageCaptureAbiInfo",
    "PageCaptureGroup",
    "PageCapturePlan",
    "PageCaptureSource",
    "PageCaptureSpan",
    "PageCaptureSubmission",
]
