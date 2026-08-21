"""Low-volume first-request ordering trace for TP4/B12X integration."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_TARGET = "b12x.attention.indexer.fused_indexer"
_active = False
_rank = -1
_sequence = 0


def enabled() -> bool:
    return os.getenv("SPARK_TP4_FLIGHT_RECORDER", "0") == "1"


def activate(rank: int) -> None:
    global _active, _rank, _sequence
    if not enabled() or _active:
        return
    _active = True
    _rank = rank
    _sequence = 0
    logger.warning("SPARK_FLIGHT start rank=%d", rank)


def _next_sequence() -> int:
    global _sequence
    _sequence += 1
    return _sequence


def record_collective(kind: str, stream: Any, *tensors: Any) -> None:
    if not _active:
        return
    descriptions = []
    for tensor in tensors:
        descriptions.append(
            "%s/%s/ptr=%#x"
            % (
                tuple(int(value) for value in tensor.shape),
                tensor.dtype,
                int(tensor.data_ptr()),
            )
        )
    logger.warning(
        "SPARK_FLIGHT rank=%d seq=%d op=%s stream=%#x tensors=%s",
        _rank,
        _next_sequence(),
        kind,
        int(stream.cuda_stream),
        ",".join(descriptions),
    )


def record_b12x(arguments: dict[str, Any]) -> None:
    if not _active:
        return
    names = (
        "q_bytes",
        "weights",
        "k_quant_bytes",
        "k_scales",
        "real_page_table",
        "seqlens",
        "out_indices",
        "out_values",
        "pack_values",
        "pack_indices",
        "merge_state",
    )
    descriptions = []
    for name in names:
        tensor = arguments.get(name)
        if tensor is None:
            continue
        descriptions.append(
            "%s=%s/%s/stride=%s/ptr=%#x"
            % (
                name,
                tuple(int(value) for value in tensor.shape),
                tensor.dtype,
                tuple(int(value) for value in tensor.stride()),
                int(tensor.data_ptr()),
            )
        )
    import torch

    stream = torch.cuda.current_stream()
    logger.warning(
        "SPARK_FLIGHT rank=%d seq=%d op=B12X stream=%#x "
        "ctas=%s threshold=%s tensors=%s",
        _rank,
        _next_sequence(),
        int(stream.cuda_stream),
        arguments.get("ctas_per_group"),
        arguments.get("merge_threshold"),
        ";".join(descriptions),
    )


def _wrap(module: ModuleType) -> None:
    original = module.run_fused_paged_indexer
    if getattr(original, "_spark_tp4_flight_recorder", False):
        return

    def traced_run_fused_paged_indexer(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("run_fused_paged_indexer is keyword-only")
        record_b12x(kwargs)
        return original(**kwargs)

    traced_run_fused_paged_indexer._spark_tp4_flight_recorder = True
    module.run_fused_paged_indexer = traced_run_fused_paged_indexer


class _TracingLoader(importlib.abc.Loader):
    def __init__(self, delegate: importlib.abc.Loader) -> None:
        self._delegate = delegate

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._delegate, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._delegate.exec_module(module)
        _wrap(module)


class _TracingFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any, target: ModuleType | None = None
    ) -> Any:
        if fullname != _TARGET:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _TracingLoader(spec.loader)
        return spec


def install_b12x_import_hook() -> None:
    if not enabled():
        return
    loaded = sys.modules.get(_TARGET)
    if loaded is not None:
        _wrap(loaded)
        return
    sys.meta_path.insert(0, _TracingFinder())
