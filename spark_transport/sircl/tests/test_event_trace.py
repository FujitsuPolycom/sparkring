"""The event trace offline: event numbering shared with the native layer and the chain kernel, stage times."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sparkring_sircl import protocol as proto
from sparkring_sircl.ring import trace

PACKAGE = Path(__file__).resolve().parents[1] / "sparkring_sircl"


def test_native_event_numbers_match_the_protocol():
    source = (PACKAGE / "oneshot" / "_roce_proxy.c").read_text(encoding="utf-8")
    native = {name: int(value) for name, value in re.findall(r"ROCE_EV_([A-Z_]+) = (\d+)", source)}
    assert native == {event.name: int(event) for event in proto.TraceEvent if event < 16}
    assert f"#define ROCE_TRACE_LINK_STREAM {proto.TRACE_LINK_STREAM}" in source
    kernel = (PACKAGE / "oneshot" / "_chain_cute.py").read_text(encoding="utf-8")
    for event in ("KERNEL_FLAG", "KERNEL_READY", "KERNEL_CONSUMED"):
        assert f"TraceEvent.{event}" in kernel


def _records():
    # Rank at a middle of half A: inbound partials and outbound partials on stream 0.
    return [
        (1000, "kernel", "KERNEL_FLAG", 0, 1), (2000, "kernel", "KERNEL_FLAG", 0, 2),
        (3000, "kernel", "KERNEL_READY", 0, 1), (3500, "native", "READY", 0, 1),
        (4000, "native", "POSTED", 0, 1), (5000, "kernel", "KERNEL_READY", 0, 2),
        (5200, "native", "READY", 0, 2), (9000, "native", "DONE", 0, 1),
        (9500, "native", "POSTED", 0, 2), (12000, "native", "DONE", 0, 2),
        # The end rank's turn: the partial on stream 0 becomes a result on stream 1.
        (20000, "kernel", "KERNEL_FLAG", 0, 7), (21000, "kernel", "KERNEL_READY", 1, 7),
        (21400, "native", "READY", 1, 7), (21500, "native", "POSTED", 1, 7),
        # A link item forwarded without the kernel.
        (30000, "native", "READY", 6, 3), (30100, "native", "POSTED", 6, 3), (30900, "native", "DONE", 6, 3),
    ]


def test_stage_times_of_chain_streams_and_links():
    found = trace.stages(_records())
    assert found[0]["kernel"] == pytest.approx([2.0, 3.0])
    assert found[0]["notice"] == pytest.approx([0.5, 0.2])
    assert found[0]["credit"] == pytest.approx([0.5, 4.3])
    assert found[0]["wire"] == pytest.approx([5.0, 2.5])
    assert found[0]["gap"] == pytest.approx([5.5])
    assert found[1]["kernel"] == pytest.approx([1.0]) and found[1]["notice"] == pytest.approx([0.4])
    assert found[1]["wire"] == []
    assert found[6]["kernel"] == [] and found[6]["wire"] == pytest.approx([0.8])
    assert trace.stream_name(0) == "A partials" and trace.stream_name(6) == "link 2"


def test_summary_and_table():
    stats = trace.summary(_records())
    assert stats["A partials"]["credit"] == {"chunks": 2, "p50_us": 4.3, "p90_us": 4.3}
    assert "kernel" not in stats["link 2"]
    merged = {"cases": [{"collective": "all_reduce_large", "bytes": 1 << 26, "algorithm": "chain",
                         "event_traces": {"1": {"offset_error_ns": 900, "lost": {"native": 0, "kernel": 0},
                                                "records": [list(r) for r in _records()]}}},
                        {"collective": "all_reduce_large", "bytes": 1 << 26, "algorithm": "pieces"}]}
    text = trace.table(merged)
    assert "rank 1 (kernel clock within 900 ns" in text
    assert "A partials: kernel 3.0/3.0, notice 0.5/0.5, credit 4.3/4.3, wire 5.0/5.0, gap 5.5/5.5" in text
    assert "SIRCL_EVENT_TRACE" in trace.table({"cases": []})
