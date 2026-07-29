from __future__ import annotations

import json
import threading

from sparkcache.streaming.timing import (
    STAGES,
    TIMING_PREFIX,
    StreamingTimingTrace,
)


def _payload(line: str) -> dict[str, object]:
    assert line.startswith(TIMING_PREFIX)
    return json.loads(line.removeprefix(TIMING_PREFIX))


def test_registered_final_batch_emits_one_compact_timing_record() -> None:
    lines: list[str] = []
    trace = StreamingTimingTrace(enabled=True, sink=lines.append)
    trace.register_final("request-7", 3, 256)

    trace.mark("request-7", 3, "final_watermark", at_ns=100)
    trace.mark("request-7", 3, "ring_submit_begin", at_ns=110)
    trace.mark("request-7", 3, "ring_submit_end", at_ns=130)
    trace.mark("request-7", 3, "adapter_observed", at_ns=200)
    trace.mark("request-7", 3, "adapter_observed", at_ns=300)

    assert len(lines) == 1
    assert "\n" not in lines[0]
    assert " " not in lines[0].removeprefix(TIMING_PREFIX)
    assert _payload(lines[0]) == {
        "batch_index": 3,
        "request_id": "request-7",
        "span_tokens": 256,
        "stage_delta_ns": {
            "adapter_observed": 100,
            "final_watermark": 0,
            "ring_submit_begin": 10,
            "ring_submit_end": 30,
        },
        "total_ns": 100,
    }


def test_timing_is_disabled_by_default_without_touching_diagnostics() -> None:
    def explode() -> int:
        raise AssertionError("disabled tracing called a diagnostic dependency")

    trace = StreamingTimingTrace(sink=explode, clock_ns=explode)

    trace.register_final("request-7", 3, 256)
    trace.mark("request-7", 3, "adapter_observed")


def test_only_registered_final_key_and_known_stages_are_traced() -> None:
    lines: list[str] = []
    trace = StreamingTimingTrace(enabled=True, sink=lines.append)
    trace.register_final("request-7", 3, 256)

    trace.mark("other-request", 3, "adapter_observed", at_ns=1)
    trace.mark("request-7", 2, "adapter_observed", at_ns=2)
    trace.mark("request-7", 3, "payload_bytes", at_ns=3)
    trace.mark("request-7", 3, "adapter_observed", at_ns=4)

    assert len(lines) == 1
    payload = _payload(lines[0])
    assert payload["stage_delta_ns"] == {"adapter_observed": 0}
    assert set(payload) == {
        "batch_index",
        "request_id",
        "span_tokens",
        "stage_delta_ns",
        "total_ns",
    }


def test_stage_deltas_follow_pipeline_order_and_never_go_backwards() -> None:
    lines: list[str] = []
    trace = StreamingTimingTrace(enabled=True, sink=lines.append)
    trace.register_final("request-7", 3, 256)

    # One stale cross-thread timestamp must not make the timeline regress.
    timestamps = [100 + index * 10 for index in range(len(STAGES))]
    timestamps[8] = timestamps[7] - 5
    for stage, timestamp_ns in zip(STAGES, timestamps, strict=True):
        trace.mark("request-7", 3, stage, at_ns=timestamp_ns)

    deltas = _payload(lines[0])["stage_delta_ns"]
    assert list(deltas) == list(STAGES)
    assert list(deltas.values()) == sorted(deltas.values())
    assert deltas["writer_end"] == deltas["writer_start"]


def test_concurrent_adapter_observation_emits_exactly_once() -> None:
    lines: list[str] = []
    trace = StreamingTimingTrace(enabled=True, sink=lines.append)
    trace.register_final("request-7", 3, 256)
    trace.mark("request-7", 3, "final_watermark", at_ns=1)
    barrier = threading.Barrier(16)

    def observe() -> None:
        barrier.wait()
        trace.mark("request-7", 3, "adapter_observed", at_ns=2)

    threads = [threading.Thread(target=observe) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(lines) == 1


def test_diagnostic_clock_and_sink_failures_never_escape() -> None:
    def explode() -> int:
        raise RuntimeError("diagnostics failed")

    trace = StreamingTimingTrace(enabled=True, sink=explode, clock_ns=explode)
    trace.register_final("request-7", 3, 256)

    trace.mark("request-7", 3, "ring_submit_begin")
    trace.mark("request-7", 3, "adapter_observed", at_ns=2)


def test_idempotent_final_registration_preserves_existing_marks() -> None:
    lines: list[str] = []
    trace = StreamingTimingTrace(enabled=True, sink=lines.append)
    trace.register_final("request-7", 3, 256)
    trace.mark("request-7", 3, "final_watermark", at_ns=100)

    trace.register_final("request-7", 3, 256)
    trace.mark("request-7", 3, "adapter_observed", at_ns=200)

    payload = _payload(lines[0])
    assert payload["stage_delta_ns"] == {
        "final_watermark": 0,
        "adapter_observed": 100,
    }


def test_mark_final_targets_the_registered_scheduler_watermark() -> None:
    lines: list[str] = []
    trace = StreamingTimingTrace(enabled=True, sink=lines.append)
    trace.register_final("request-7", 11, 192)
    trace.mark("request-7", 11, "final_watermark", at_ns=100)

    trace.mark_final("request-7", "adapter_observed", at_ns=220)

    payload = _payload(lines[0])
    assert payload["batch_index"] == 11
    assert payload["span_tokens"] == 192
    assert payload["total_ns"] == 120
