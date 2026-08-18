"""Bounded counters proving that live graph requests avoid stock collectives."""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass

_PHASES = ("capture", "eager")
_SIGNATURE_LIMIT_ENV = "SPARK_COLLECTIVE_AUDIT_SIGNATURE_LIMIT_PER_PHASE"
_DEFAULT_SIGNATURES_PER_PHASE = 512
_HARD_MAX_SIGNATURES_PER_PHASE = 2048
_MAX_SHAPE_RANK = 8
_MAX_DTYPE_CHARS = 64
_MAX_UNIQUE_NAME_CHARS = 128
_MAX_DIMENSION = (1 << 63) - 1


def _signature_limit_from_env() -> int:
    raw = os.getenv(_SIGNATURE_LIMIT_ENV)
    if raw is None:
        return _DEFAULT_SIGNATURES_PER_PHASE
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise RuntimeError(
            f"{_SIGNATURE_LIMIT_ENV} must be an integer between 1 and "
            f"{_HARD_MAX_SIGNATURES_PER_PHASE}; got {raw!r}"
        ) from exc
    if not 1 <= value <= _HARD_MAX_SIGNATURES_PER_PHASE:
        raise RuntimeError(
            f"{_SIGNATURE_LIMIT_ENV} must be an integer between 1 and "
            f"{_HARD_MAX_SIGNATURES_PER_PHASE}; got {raw!r}"
        )
    return value


_MAX_SIGNATURES_PER_PHASE = _signature_limit_from_env()
_lock = threading.Lock()
_counts: dict[str, dict[str, int]] = {
    phase: defaultdict(int) for phase in _PHASES
}
_signature_counts: dict[
    str,
    dict[tuple[str, str, "StockCollectiveSignature"], int],
] = {phase: {} for phase in _PHASES}
_signature_dropped_calls: dict[str, int] = {phase: 0 for phase in _PHASES}
_signature_dropped_calls_by_family_reason: dict[str, dict[str, int]] = {
    phase: defaultdict(int) for phase in _PHASES
}


@dataclass(frozen=True)
class StockCollectiveSignature:
    """Pointer-free, cardinality-safe evidence for one stock fallback."""

    shape: tuple[int, ...]
    dtype: str
    is_cuda: bool
    contiguous: bool
    world_size: int | None
    unique_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise ValueError(
                "stock collective signature shape must be a tuple"
            )
        if len(self.shape) > _MAX_SHAPE_RANK:
            raise ValueError(
                "stock collective signature shape rank exceeds "
                f"{_MAX_SHAPE_RANK}"
            )
        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            or dimension > _MAX_DIMENSION
            for dimension in self.shape
        ):
            raise ValueError(
                "stock collective signature shape dimensions must be "
                "nonnegative signed 64-bit integers"
            )
        if (
            not isinstance(self.dtype, str)
            or not self.dtype
            or len(self.dtype) > _MAX_DTYPE_CHARS
        ):
            raise ValueError(
                "stock collective signature dtype must contain between "
                f"1 and {_MAX_DTYPE_CHARS} characters"
            )
        if not isinstance(self.is_cuda, bool) or not isinstance(
            self.contiguous, bool
        ):
            raise ValueError(
                "stock collective signature CUDA and contiguous fields "
                "must be booleans"
            )
        if (
            not isinstance(self.unique_name, str)
            or len(self.unique_name) > _MAX_UNIQUE_NAME_CHARS
        ):
            raise ValueError(
                "stock collective signature unique_name exceeds "
                f"{_MAX_UNIQUE_NAME_CHARS} characters"
            )
        if self.world_size is not None and (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 0
            or self.world_size > _MAX_DIMENSION
        ):
            raise ValueError(
                "stock collective signature world_size must be a "
                "nonnegative signed 64-bit integer or None"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "is_cuda": self.is_cuda,
            "contiguous": self.contiguous,
            "world_size": self.world_size,
            "unique_name": self.unique_name,
        }


def classify_stock_family(
    seam: str,
    signature: StockCollectiveSignature,
    *,
    dim: int | None = None,
) -> str:
    """Map a pointer-free call signature to its semantic collective family."""
    shape = signature.shape
    dtype = signature.dtype
    group = signature.unique_name

    if seam == "group_all_gather":
        if group.startswith("dcp:"):
            exact_dcp4_tensor = (
                signature.world_size == 4
                and signature.is_cuda
                and signature.contiguous
                and bool(shape)
                and shape[0] > 0
            )
            if (
                exact_dcp4_tensor
                and len(shape) == 3
                and shape[1:] == (16, 576)
                and dtype == "torch.bfloat16"
                and dim in {1, -2}
            ):
                return "dcp_query_all_gather"
            if (
                exact_dcp4_tensor
                and len(shape) == 2
                and shape[1] == 64
                and dtype == "torch.float32"
                and dim in {0, -2}
            ):
                return "dcp_lse_all_gather"
            return "dcp_all_gather"
        if (
            group.startswith("tp:")
            and len(shape) == 2
            and shape[1] == 38720
            and dtype == "torch.bfloat16"
            and dim in {1, -1}
        ):
            return "vocabulary_all_gather"
        return "group_all_gather"

    if seam == "pynccl_all_gather":
        if (
            len(shape) == 3
            and shape[1:] == (2, 2048)
            and dtype == "torch.int32"
        ):
            return "dcp_owner_topk_all_gather"
        if (
            len(shape) == 1
            and shape[0] in {23552, 753664}
            and dtype == "torch.uint8"
        ):
            return "dcp_ckv_all_gather"
        if (
            shape == (1, 38720)
            and dtype == "torch.bfloat16"
        ):
            return "vocabulary_all_gather"
        return "pynccl_all_gather"

    if seam == "group_reduce_scatter":
        if (
            group.startswith("dcp:")
            and len(shape) == 3
            and shape[1] == 64
            and shape[2] in {256, 512}
            and dtype == "torch.bfloat16"
            and dim in {1, -2}
        ):
            return "dcp_output_reduce_scatter"
        return "group_reduce_scatter"

    raise ValueError(f"unknown stock collective seam: {seam}")


def enabled() -> bool:
    """Return whether the live graph status surface requested auditing."""
    return bool(os.getenv("SPARK_TP4_GRAPH_STATUS_PATH"))


def record_stock(
    family: str,
    *,
    capturing: bool,
    reason: str,
    signature: StockCollectiveSignature | None = None,
) -> None:
    """Record one call through an original/stock collective implementation."""
    if not enabled():
        return
    if not family or not reason:
        raise ValueError("stock collective family and reason must be nonempty")
    phase = "capture" if capturing else "eager"
    key = f"{family}:{reason}"
    with _lock:
        _counts[phase][key] += 1
        if signature is not None:
            signature_key = (family, reason, signature)
            phase_signatures = _signature_counts[phase]
            if signature_key in phase_signatures:
                phase_signatures[signature_key] += 1
            elif len(phase_signatures) < _MAX_SIGNATURES_PER_PHASE:
                phase_signatures[signature_key] = 1
            else:
                _signature_dropped_calls[phase] += 1
                _signature_dropped_calls_by_family_reason[phase][key] += 1


def _signature_sort_key(
    item: tuple[tuple[str, str, StockCollectiveSignature], int],
) -> tuple[object, ...]:
    (family, reason, signature), _count = item
    return (
        family,
        reason,
        signature.shape,
        signature.dtype,
        signature.is_cuda,
        signature.contiguous,
        -1 if signature.world_size is None else signature.world_size,
        signature.unique_name,
    )


def stock_collective_snapshot() -> dict[str, object]:
    """Return a stable, bounded copy for the low-rate status reporter."""
    with _lock:
        phases = {
            phase: dict(sorted(_counts[phase].items()))
            for phase in _PHASES
        }
        signatures = {
            phase: [
                {
                    "family": family,
                    "reason": reason,
                    "count": count,
                    **signature.to_dict(),
                }
                for (family, reason, signature), count in sorted(
                    _signature_counts[phase].items(),
                    key=_signature_sort_key,
                )
            ]
            for phase in _PHASES
        }
        dropped_calls = dict(_signature_dropped_calls)
        dropped_calls_by_family_reason = {
            phase: dict(
                sorted(_signature_dropped_calls_by_family_reason[phase].items())
            )
            for phase in _PHASES
        }
    snapshot: dict[str, object] = {
        "capture": phases["capture"],
        "eager": phases["eager"],
        "capture_total": sum(phases["capture"].values()),
        "eager_total": sum(phases["eager"].values()),
    }
    if any(signatures.values()) or any(dropped_calls.values()):
        snapshot.update(
            {
                "signatures": signatures,
                "signature_limit_per_phase": _MAX_SIGNATURES_PER_PHASE,
                "signature_dropped_calls": dropped_calls,
                "signature_dropped_calls_by_family_reason": (
                    dropped_calls_by_family_reason
                ),
            }
        )
    return snapshot


def _reset_for_tests() -> None:
    with _lock:
        for phase in _PHASES:
            _counts[phase].clear()
            _signature_counts[phase].clear()
            _signature_dropped_calls[phase] = 0
            _signature_dropped_calls_by_family_reason[phase].clear()
