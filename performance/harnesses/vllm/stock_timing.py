"""CUDA-event timing for one GLM-5.2 round with four speculative tokens.

This diagnostic preserves inputs, outputs and intra-stream operation order.
Reporting synchronizes the host once and may perturb later arrivals. It
records CUDA events around the original vLLM operation and reports once the
fixed call inventory in _EXPECTED has completed. Query-row counts Q5 and Q1
represent a five-row target step and single-row draft steps for MTP4. This
inventory is specific to that execution shape, not a model-independent timer.
SPARK_TP4_STOCK_TIMING=1 enables instrumentation; an arm-file run ID selects
the measured round after the three-row startup call has been observed.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_Result = TypeVar("_Result")
_EXPECTED: dict[tuple[str, int], tuple[int, int]] = {
    ("query", 5): (79, 79),
    ("query", 1): (3, 3),
    ("combine", 5): (79, 158),
    ("combine", 1): (3, 6),
    ("vocab", 5): (1, 1),
    ("vocab", 1): (4, 4),
}
_TOTAL_WRAPPER_CALLS = sum(calls for calls, _ in _EXPECTED.values())
_armed = False
_reported = False
_invalid = False
_startup_q3_seen = False
_run_id: str | None = None
_stream_id: int | None = None
_first_host_ns: int | None = None
_last_host_ns: int | None = None
_samples: dict[tuple[str, int], list[tuple[Any, Any, float]]] = {
    key: [] for key in _EXPECTED
}
_overflow: dict[tuple[str, int], int] = {key: 0 for key in _EXPECTED}
_invalid_reasons: set[str] = set()
_event_pairs: list[tuple[Any, Any]] = []
_calibration_events: tuple[Any, Any] | None = None
_first_start: Any | None = None
_last_stop: Any | None = None


def enabled() -> bool:
    return os.getenv("SPARK_TP4_STOCK_TIMING", "0") == "1"


def _complete() -> bool:
    return all(
        len(_samples[key]) == expected_calls
        for key, (expected_calls, _) in _EXPECTED.items()
    )


def _invalidate(reason: str) -> None:
    global _invalid
    _invalid = True
    if reason not in _invalid_reasons:
        _invalid_reasons.add(reason)
        logger.error(
            "SPARK_STOCK_TIMING invalid reason=%s rank=%s run_id=%s",
            reason,
            os.getenv("RANK", "unknown"),
            _run_id or "none",
        )


def _initialize_event_pool(torch_module: Any, stream: Any) -> None:
    global _calibration_events
    if _event_pairs:
        return
    calibration_start = torch_module.cuda.Event(enable_timing=True)
    calibration_stop = torch_module.cuda.Event(enable_timing=True)
    calibration_start.record(stream)
    calibration_stop.record(stream)
    pairs = []
    for _ in range(_TOTAL_WRAPPER_CALLS):
        pairs.append(
            (
                torch_module.cuda.Event(enable_timing=True),
                torch_module.cuda.Event(enable_timing=True),
            )
        )
    _event_pairs.extend(pairs)
    _calibration_events = (calibration_start, calibration_stop)


def _elapsed_ms(start: Any, stop: Any) -> float:
    value = float(start.elapsed_time(stop))
    if not math.isfinite(value) or value < 0:
        raise ValueError("CUDA event duration must be finite and nonnegative")
    return value


def _report() -> None:
    global _reported
    assert _last_stop is not None
    assert _first_start is not None
    _last_stop.synchronize()

    assert _calibration_events is not None
    calibration_ms = _elapsed_ms(*_calibration_events)
    total_device_ms = 0.0
    total_host_enqueue_us = 0.0
    total_calls = 0
    total_logical = 0
    for key in _EXPECTED:
        family, q = key
        samples = _samples[key]
        expected_calls, logical_collectives = _EXPECTED[key]
        device_ms = sum(_elapsed_ms(start, stop) for start, stop, _ in samples)
        host_enqueue_us = sum(host_us for _, _, host_us in samples)
        total_device_ms += device_ms
        total_host_enqueue_us += host_enqueue_us
        total_calls += expected_calls
        total_logical += logical_collectives
        logger.warning(
            "SPARK_STOCK_TIMING rank=%s run_id=%s family=%s q=%d "
            "wrapper_calls=%d "
            "logical_collectives=%d device_ms=%.6f device_us_per_call=%.3f "
            "host_enqueue_us=%.3f",
            os.getenv("RANK", "unknown"),
            _run_id or "none",
            family,
            q,
            expected_calls,
            logical_collectives,
            device_ms,
            device_ms * 1000.0 / expected_calls,
            host_enqueue_us,
        )

    covered_span_ms = _elapsed_ms(_first_start, _last_stop)
    logger.warning(
        "SPARK_STOCK_TIMING rank=%s run_id=%s total wrapper_calls=%d "
        "logical_collectives=%d "
        "device_ms=%.6f covered_span_ms=%.6f host_enqueue_us=%.3f "
        "event_calibration_us=%.3f stream=%s first_host_ns=%s last_host_ns=%s "
        "host_span_ms=%.6f overflow_calls=%d invalid_reasons=%s valid=%s "
        "semantics=stock-fixed-k4",
        os.getenv("RANK", "unknown"),
        _run_id or "none",
        total_calls,
        total_logical,
        total_device_ms,
        covered_span_ms,
        total_host_enqueue_us,
        calibration_ms * 1000.0,
        str(_stream_id),
        str(_first_host_ns),
        str(_last_host_ns),
        (
            0.0
            if _first_host_ns is None or _last_host_ns is None
            else (_last_host_ns - _first_host_ns) / 1_000_000.0
        ),
        sum(_overflow.values()),
        ",".join(sorted(_invalid_reasons)) or "none",
        str(not _invalid).lower(),
    )
    _reported = True


def _requested_run_id() -> str | None:
    arm_path = os.getenv("SPARK_TP4_STOCK_TIMING_ARM_PATH", "")
    if not arm_path:
        return None
    try:
        value = Path(arm_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def time_original(
    family: str,
    q: int,
    stream: Any,
    operation: Callable[[], _Result],
    torch_module: Any | None = None,
) -> _Result:
    """Run an original collective unchanged and optionally time its GPU span."""

    global _armed, _first_host_ns, _first_start, _last_host_ns, _last_stop
    global _run_id, _startup_q3_seen, _stream_id, _reported
    if not enabled() or _reported:
        return operation()

    requested_run_id = _requested_run_id()
    if q == 3 and not _armed:
        if requested_run_id is not None:
            _invalidate("operator_arm_before_startup_q3")
        _startup_q3_seen = True
        logger.warning(
            "SPARK_STOCK_TIMING startup_q3_seen rank=%s",
            os.getenv("RANK", "unknown"),
        )
        return operation()

    if not _armed and requested_run_id is not None:
        if not _startup_q3_seen:
            _invalidate("operator_arm_before_startup_q3")
            return operation()
        _armed = True
        _run_id = requested_run_id
        logger.warning(
            "SPARK_STOCK_TIMING armed rank=%s run_id=%s path=%s",
            os.getenv("RANK", "unknown"),
            _run_id,
            os.getenv("SPARK_TP4_STOCK_TIMING_ARM_PATH"),
        )
    if not _armed:
        return operation()
    if requested_run_id != _run_id:
        _invalidate("run_id_changed_or_removed")
        return operation()
    if q in (2, 3, 4):
        _invalidate(f"unexpected_q{q}")
        return operation()

    key = (family, q)
    expected = _EXPECTED.get(key)
    if expected is None:
        _invalidate(f"unexpected_family_or_q:{family}:q{q}")
        return operation()
    if len(_samples[key]) >= expected[0]:
        if not _complete():
            _overflow[key] += 1
            _invalidate(f"overflow:{family}:q{q}")
        return operation()

    if torch_module is None:
        import torch as torch_module

    current_stream_id = int(stream.cuda_stream)
    if _stream_id is None:
        _stream_id = current_stream_id
    elif _stream_id != current_stream_id:
        _invalidate("stream_changed")
        return operation()

    try:
        _initialize_event_pool(torch_module, stream)
        event_index = sum(len(samples) for samples in _samples.values())
        start, stop = _event_pairs[event_index]
        host_before_ns = time.perf_counter_ns()
        start.record(stream)
    except Exception:
        _invalidate('event_record_failed')
        return operation()
    host_start_ns = time.perf_counter_ns()
    try:
        result = operation()
    except BaseException:
        _invalidate('operation_failed')
        raise
    host_enqueue_us = (time.perf_counter_ns() - host_start_ns) / 1000.0
    try:
        stop.record(stream)
    except Exception:
        _invalidate('event_record_failed')
        return result
    host_after_ns = time.perf_counter_ns()

    if _first_start is None:
        _first_start = start
        _first_host_ns = host_before_ns
    _last_stop = stop
    _last_host_ns = host_after_ns
    _samples[key].append((start, stop, host_enqueue_us))
    try:
        max_span_ms = float(os.getenv("SPARK_TP4_STOCK_TIMING_MAX_HOST_SPAN_MS", "2000"))
    except ValueError:
        max_span_ms = float("nan")
    if (
        not 0.0 < max_span_ms <= 60_000.0
        or _first_host_ns is None
        or _last_host_ns is None
    ):
        _invalidate("invalid_host_span_limit")
    elif (_last_host_ns - _first_host_ns) / 1_000_000.0 > max_span_ms:
        _invalidate("host_span_limit_exceeded")
    if _complete():
        try:
            _report()
        except Exception:
            _invalidate("report_failed")
        finally:
            _reported = True
    return result


def snapshot_for_test() -> dict[str, Any]:
    """Return counter-only state without synchronizing CUDA."""

    return {
        "armed": _armed,
        "reported": _reported,
        "invalid": _invalid,
        "startup_q3_seen": _startup_q3_seen,
        "run_id": _run_id,
        "stream_id": _stream_id,
        "counts": {key: len(samples) for key, samples in _samples.items()},
        "overflow": dict(_overflow),
        "invalid_reasons": sorted(_invalid_reasons),
    }


def reset_for_test() -> None:
    """Reset module state for isolated unit tests."""

    global _armed, _reported, _invalid, _startup_q3_seen, _run_id
    global _stream_id, _first_host_ns, _last_host_ns
    global _calibration_events, _first_start, _last_stop
    _armed = False
    _reported = False
    _invalid = False
    _startup_q3_seen = False
    _run_id = None
    _stream_id = None
    _first_host_ns = None
    _last_host_ns = None
    _calibration_events = None
    _first_start = None
    _last_stop = None
    _event_pairs.clear()
    _invalid_reasons.clear()
    for samples in _samples.values():
        samples.clear()
    for key in _overflow:
        _overflow[key] = 0
