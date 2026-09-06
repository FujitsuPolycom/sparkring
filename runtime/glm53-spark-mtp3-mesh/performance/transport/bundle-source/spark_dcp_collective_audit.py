"""Audit-only DCP combine/reduce-scatter accounting at vLLM seams."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any

_installed = False
_COMBINE_TARGET = "vllm.v1.attention.ops.common"


def _enabled() -> bool:
    value = os.getenv("SPARK_TP4_DCP_COLLECTIVE_AUDIT", "0")
    if value not in {"0", "1"}:
        raise ValueError(
            "SPARK_TP4_DCP_COLLECTIVE_AUDIT must be '0' or '1'"
        )
    return value == "1"


def _is_stream_capturing(torch_module: Any) -> bool:
    checker = getattr(torch_module.cuda, "is_current_stream_capturing", None)
    return bool(checker is not None and checker())


def _signature(group: Any, tensor: Any) -> Any:
    from spark_collective_audit import StockCollectiveSignature

    world_size = getattr(group, "world_size", None)
    return StockCollectiveSignature(
        shape=tuple(int(value) for value in tensor.shape),
        dtype=str(tensor.dtype),
        is_cuda=bool(tensor.is_cuda),
        contiguous=bool(tensor.is_contiguous()),
        world_size=None if world_size is None else int(world_size),
        unique_name=str(getattr(group, "unique_name", "")),
    )


def _standalone_combine_audit_enabled() -> bool:
    """Use the stock-only hook only when the native DCP adapter is absent."""

    return not bool(os.getenv("VLLM_SPARK_TP4_DCP_MODE", ""))


def _patch_existing_combine_aliases(original: Any, replacement: Any) -> None:
    """Repair aliases bound before the defining common module was patched."""

    for loaded in tuple(sys.modules.values()):
        if not isinstance(loaded, ModuleType) or loaded is sys.modules.get(
            _COMBINE_TARGET
        ):
            continue
        if vars(loaded).get("cp_lse_ag_out_rs") is original:
            loaded.cp_lse_ag_out_rs = replacement


def _patch_combine(module: ModuleType) -> None:
    """Wrap the stock combine without constructing any transport state."""

    current = module.cp_lse_ag_out_rs
    if getattr(current, "_spark_dcp_collective_audit", False):
        _patch_existing_combine_aliases(current._spark_original, current)
        return
    # A selected native DCP adapter owns this seam and records its own stock
    # calls.  Wrapping it here would falsely label custom calls as stock.
    if getattr(current, "_spark_tp4_dcp_backend", False):
        return
    original = current

    def audited_combine(
        cp_attn_out: Any,
        cp_attn_lse: Any,
        cp_group: Any,
        ctx: Any = None,
        return_lse: bool = False,
        is_lse_base_on_e: bool = True,
        head_major_output: bool = False,
    ) -> Any:
        import torch
        from spark_collective_audit import enabled, record_stock

        signature = _signature(cp_group, cp_attn_out) if enabled() else None
        record_stock(
            "dcp_combine",
            capturing=_is_stream_capturing(torch),
            reason="original",
            signature=signature,
        )
        return original(
            cp_attn_out,
            cp_attn_lse,
            cp_group,
            ctx=ctx,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
            head_major_output=head_major_output,
        )

    audited_combine._spark_dcp_collective_audit = True  # type: ignore[attr-defined]
    audited_combine._spark_original = original  # type: ignore[attr-defined]
    module.cp_lse_ag_out_rs = audited_combine
    _patch_existing_combine_aliases(original, audited_combine)


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
        self,
        fullname: str,
        path: Any,
        target: ModuleType | None = None,
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
    """Install pointer-free accounting wrappers without changing execution."""
    global _installed
    if _installed or not _enabled():
        return

    if _standalone_combine_audit_enabled():
        # Install before importing vLLM so a deferred common-module import is
        # captured.  Native/custom DCP mode uses spark_tp4_dcp_backend instead;
        # two independent import hooks on this seam would not be fail-closed.
        _install_combine_import_hook()

    from vllm.distributed.parallel_state import GroupCoordinator

    original = GroupCoordinator._reduce_scatter_out_place
    if getattr(original, "_spark_dcp_collective_audit", False):
        _installed = True
        return

    def audited_reduce_scatter(
        self: Any,
        input_tensor: Any,
        dim: int,
    ) -> Any:
        import torch
        from spark_collective_audit import (
            classify_stock_family,
            enabled,
            record_stock,
        )

        signature = _signature(self, input_tensor) if enabled() else None
        family = (
            "group_reduce_scatter"
            if signature is None
            else classify_stock_family(
                "group_reduce_scatter",
                signature,
                dim=dim,
            )
        )
        record_stock(
            family,
            capturing=_is_stream_capturing(torch),
            reason="original",
            signature=signature,
        )
        return original(self, input_tensor, dim)

    audited_reduce_scatter._spark_dcp_collective_audit = True  # type: ignore[attr-defined]
    audited_reduce_scatter._spark_original = original  # type: ignore[attr-defined]
    GroupCoordinator._reduce_scatter_out_place = audited_reduce_scatter
    _installed = True
