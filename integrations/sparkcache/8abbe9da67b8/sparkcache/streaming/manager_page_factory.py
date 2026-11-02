"""Explicit assembly for asynchronous opaque manager-page capture."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .manager_page_capture import ManagerPageSource
from .manager_page_native_ring import NativeManagerPageRing
from .manager_page_runtime import ManagerPageCaptureRuntime
from .native_ring import NativeRingConfig


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
LIBRARY_KEY = "spark_cache_async_page_capture_library"
LIBRARY_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_LIBRARY"
LIBRARY_SHA256_KEY = "spark_cache_async_page_capture_library_sha256"
LIBRARY_SHA256_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_LIBRARY_SHA256"
SLOT_BYTES_KEY = "spark_cache_async_page_capture_slot_bytes"
SLOT_BYTES_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_SLOT_BYTES"
SLOT_COUNT_KEY = "spark_cache_async_page_capture_slot_count"
SLOT_COUNT_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_SLOT_COUNT"
VLLM_ROOT_KEY = "spark_cache_async_page_capture_vllm_root"
VLLM_ROOT_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_VLLM_ROOT"
LEASE_CONTRACT_KEY = "spark_cache_async_page_capture_lease_contract"
LEASE_CONTRACT_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_LEASE_CONTRACT"
LEASE_MODE_KEY = "spark_cache_async_page_capture_lease_mode"
LEASE_MODE_ENV = "SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_LEASE_MODE"


def _absolute(path: Path) -> bool:
    return path.is_absolute() or bool(path.root) or PurePosixPath(str(path)).is_absolute()


def _extra(connector: Any, key: str, environment: str, default: str = "") -> str:
    value = connector._kv_transfer_config.get_from_extra_config(
        key, os.environ.get(environment, default)
    )
    return str(value or "")


@dataclass(frozen=True, slots=True)
class ManagerPageCaptureSettings:
    library_path: Path
    library_sha256: str
    slot_bytes: int
    slot_count: int = 2
    vllm_root: Path | None = None
    lease_contract: Path | None = None
    lease_mode: str = "request-finish"

    def __post_init__(self) -> None:
        if not _absolute(self.library_path):
            raise RuntimeError("manager-page capture library path must be absolute")
        if _SHA256_RE.fullmatch(self.library_sha256) is None:
            raise RuntimeError(
                "manager-page capture library requires a lowercase SHA-256"
            )
        if self.slot_bytes <= 0:
            raise RuntimeError("manager-page capture slot bytes must be positive")
        if self.slot_count not in (2, 3):
            raise RuntimeError("manager-page capture slot count must be two or three")
        if self.lease_mode not in ("request-finish", "connector-jobs"):
            raise RuntimeError("manager-page capture lease mode is unsupported")
        if self.lease_mode == "connector-jobs" and self.lease_contract is None:
            raise RuntimeError("connector-job capture requires an explicit source contract")
        for name in ("vllm_root", "lease_contract"):
            value = getattr(self, name)
            if value is not None and not _absolute(value):
                raise RuntimeError(f"manager-page capture {name} must be absolute")

    @classmethod
    def from_connector(cls, connector: Any) -> "ManagerPageCaptureSettings":
        vllm_root = _extra(connector, VLLM_ROOT_KEY, VLLM_ROOT_ENV)
        lease = _extra(connector, LEASE_CONTRACT_KEY, LEASE_CONTRACT_ENV)
        try:
            slot_bytes = int(_extra(connector, SLOT_BYTES_KEY, SLOT_BYTES_ENV, "0"))
            slot_count = int(_extra(connector, SLOT_COUNT_KEY, SLOT_COUNT_ENV, "2"))
        except ValueError as error:
            raise RuntimeError(
                "manager-page capture slot settings must be integers"
            ) from error
        return cls(
            library_path=Path(_extra(connector, LIBRARY_KEY, LIBRARY_ENV)),
            library_sha256=_extra(
                connector, LIBRARY_SHA256_KEY, LIBRARY_SHA256_ENV
            ),
            slot_bytes=slot_bytes,
            slot_count=slot_count,
            vllm_root=Path(vllm_root) if vllm_root else None,
            lease_contract=Path(lease) if lease else None,
            lease_mode=_extra(connector, LEASE_MODE_KEY, LEASE_MODE_ENV, "request-finish"),
        )


def verify_manager_page_lease_contract(
    settings: ManagerPageCaptureSettings,
) -> tuple[Path, ...]:
    from sparkcache.runtime_patches.verify_lease_contract import verify_contract

    if settings.vllm_root is None:
        import vllm

        root = Path(vllm.__file__).resolve().parent.parent
    else:
        root = settings.vllm_root
    contract = settings.lease_contract or (
        Path(__file__).resolve().parents[1]
        / "runtime_patches"
        / "vllm-manager-page-async-contract-55969c16.json"
    )
    if settings.lease_mode == "connector-jobs":
        from sparkcache.runtime_patches.generic_connector_contract import (
            verify_connector_job_contract,
        )

        return tuple(verify_connector_job_contract(root, contract))
    return tuple(verify_contract(root, contract))


def build_manager_page_runtime(
    connector: Any,
    settings: ManagerPageCaptureSettings,
    *,
    ring_builder: Callable[..., Any] = NativeManagerPageRing.from_attested,
    progress_thread_initializer: Callable[[], None] | None = None,
) -> ManagerPageCaptureRuntime:
    verify_manager_page_lease_contract(settings)
    layout = connector._page_layout
    if layout is None:
        raise RuntimeError("manager-page layout is not registered")
    sources = []
    device_indexes = set()
    group_capacities = []
    for group_index, group in enumerate(layout.groups):
        capacity = None
        for layer_ordinal, layer in enumerate(group.layers):
            tensor = connector._layer_tensors[layer.name]
            if tensor.device.type != "cuda":
                raise RuntimeError("manager-page capture source is not CUDA memory")
            device_indexes.add(int(tensor.device.index or 0))
            source_pages = int(tensor.shape[0])
            if capacity is None:
                capacity = source_pages
            elif capacity != source_pages:
                raise RuntimeError(
                    "manager-page layers in one group have different capacities"
                )
            sources.append(
                ManagerPageSource(
                    source_base=int(tensor.data_ptr()),
                    source_pages=source_pages,
                    source_page_stride_bytes=(
                        int(tensor.stride(0)) * int(tensor.element_size())
                    ),
                    bytes_per_page=layer.bytes_per_page,
                    group_index=group_index,
                    layer_ordinal=layer_ordinal,
                )
            )
        assert capacity is not None
        group_capacities.append(capacity)
    if len(device_indexes) != 1:
        raise RuntimeError("manager-page sources must share one CUDA device")
    device_ordinal = next(iter(device_indexes))
    config = NativeRingConfig(
        arena_mode=1,
        slot_bytes=settings.slot_bytes,
        slot_count=settings.slot_count,
        max_sources=len(sources),
        max_rows=sum(group_capacities),
        device_ordinal=device_ordinal,
    )
    ring = ring_builder(
        config,
        library_path=settings.library_path,
        expected_sha256=settings.library_sha256,
    )
    ring.configure_sources(sources, group_count=len(layout.groups))
    if progress_thread_initializer is None:
        import torch

        def initialize_progress_thread() -> None:
            torch.cuda.set_device(device_ordinal)

        progress_thread_initializer = initialize_progress_thread
    return ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_thread_initializer=progress_thread_initializer,
    )


__all__ = [
    "ManagerPageCaptureSettings",
    "build_manager_page_runtime",
    "verify_manager_page_lease_contract",
]
