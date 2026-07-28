from __future__ import annotations

from typing import Any

import pytest

from .phase_timing import (
    PhaseDescriptor,
    PhaseKind,
    PhaseTimingCollector,
)
from .vllm_adapter import (
    AdapterValidationError,
    FailClosedMethodAdapter,
    MethodHook,
    source_sha256,
)


class FakeEvent:
    def record(self, stream: Any) -> None:
        del stream

    def query(self) -> bool:
        return True

    def elapsed_time(self, end: Any) -> float:
        del end
        return 1.0


class FullGraph:
    def run(self, descriptor: str) -> str:
        return f"full:{descriptor}"


class DraftGraph:
    def run(self, descriptor: str) -> str:
        return f"draft:{descriptor}"


def _collector() -> PhaseTimingCollector:
    return PhaseTimingCollector(
        event_factory=FakeEvent,
        capacity=4,
        descriptors=(
            PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6"),
            PhaseDescriptor(PhaseKind.DRAFT_MULTISTEP_GRAPH, "Q1"),
        ),
    )


def _hook(
    owner: type,
    descriptor: PhaseDescriptor,
    source_hash: str | None = None,
) -> MethodHook:
    return MethodHook(
        owner=owner,
        method_name="run",
        expected_source_sha256=source_hash or source_sha256(owner.run),
        descriptor=descriptor,
        stream_for_call=lambda instance, args, kwargs: instance,
    )


def test_validation_failure_mutates_no_methods() -> None:
    full_original = FullGraph.run
    draft_original = DraftGraph.run
    adapter = FailClosedMethodAdapter(
        _collector(),
        (
            _hook(
                FullGraph,
                PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6"),
            ),
            _hook(
                DraftGraph,
                PhaseDescriptor(PhaseKind.DRAFT_MULTISTEP_GRAPH, "Q1"),
                "0" * 64,
            ),
        ),
    )

    with pytest.raises(AdapterValidationError, match="source mismatch"):
        adapter.install()
    assert FullGraph.run is full_original
    assert DraftGraph.run is draft_original


def test_installed_hooks_measure_and_uninstall_exactly() -> None:
    timing = _collector()
    full_original = FullGraph.run
    draft_original = DraftGraph.run
    adapter = FailClosedMethodAdapter(
        timing,
        (
            _hook(
                FullGraph,
                PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6"),
            ),
            _hook(
                DraftGraph,
                PhaseDescriptor(PhaseKind.DRAFT_MULTISTEP_GRAPH, "Q1"),
            ),
        ),
    )
    adapter.install()
    timing.arm("adapter-test")
    assert FullGraph().run("a") == "full:a"
    assert DraftGraph().run("b") == "draft:b"
    timing.disarm()
    assert timing.snapshot()["reserved"] == 2

    adapter.uninstall()
    assert FullGraph.run is full_original
    assert DraftGraph.run is draft_original


def test_adapter_rejects_second_wrapper() -> None:
    timing = _collector()
    hook = _hook(
        FullGraph,
        PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6"),
    )
    first = FailClosedMethodAdapter(timing, (hook,))
    first.install()
    try:
        second = FailClosedMethodAdapter(timing, (hook,))
        with pytest.raises(AdapterValidationError, match="already wrapped"):
            second.install()
    finally:
        first.uninstall()
