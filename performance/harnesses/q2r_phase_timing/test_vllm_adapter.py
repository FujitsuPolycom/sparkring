from __future__ import annotations

from builtins import BaseExceptionGroup, ExceptionGroup
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

@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_interrupted_install_restores_methods(monkeypatch, error_type) -> None:
    from . import vllm_adapter
    originals = (FullGraph.run, DraftGraph.run)
    adapter = FailClosedMethodAdapter(_collector(), (
        _hook(FullGraph, PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6")),
        _hook(DraftGraph, PhaseDescriptor(PhaseKind.DRAFT_MULTISTEP_GRAPH, "Q1")),
    ))
    interrupted = False

    def interrupt_after_write(owner, name, value):
        nonlocal interrupted
        setattr(owner, name, value)
        if owner is DraftGraph and not interrupted:
            interrupted = True
            raise error_type("installation interrupted")

    monkeypatch.setattr(vllm_adapter, "setattr", interrupt_after_write, raising=False)
    try:
        with pytest.raises(error_type):
            adapter.install()
        assert FullGraph.run is originals[0]
        assert DraftGraph.run is originals[1]
    finally:
        FullGraph.run, DraftGraph.run = originals

@pytest.mark.parametrize("descriptor_type", [staticmethod, classmethod])
def test_non_instance_methods_are_rejected(descriptor_type) -> None:
    class OtherGraph:
        run = descriptor_type(FullGraph.run)
    original = OtherGraph.__dict__["run"]
    adapter = FailClosedMethodAdapter(_collector(), (
        _hook(OtherGraph, PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6")),
    ))
    try:
        with pytest.raises(AdapterValidationError, match="instance method"):
            adapter.install()
        assert OtherGraph.__dict__["run"] is original
    finally:
        OtherGraph.run = original

def test_inherited_method_restores_inheritance() -> None:
    class ChildGraph(FullGraph):
        pass
    adapter = FailClosedMethodAdapter(_collector(), (
        _hook(ChildGraph, PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6")),
    ))
    adapter.install()
    adapter.uninstall()
    assert "run" not in ChildGraph.__dict__
    assert ChildGraph.run is FullGraph.run

def test_call_keywords_cannot_replace_the_pinned_method() -> None:
    adapter = FailClosedMethodAdapter(_collector(), (
        _hook(FullGraph, PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6")),
    ))
    adapter.install()
    try:
        with pytest.raises(TypeError):
            FullGraph().run("a", _FailClosedMethodAdapter__original=lambda *a, **k: "wrong")
    finally:
        adapter.uninstall()

def test_uninstall_checks_all_wrapper_identities_before_restoring() -> None:
    import functools
    originals = FullGraph.run, DraftGraph.run
    adapter = FailClosedMethodAdapter(_collector(), (
        _hook(FullGraph, PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6")),
        _hook(DraftGraph, PhaseDescriptor(PhaseKind.DRAFT_MULTISTEP_GRAPH, "Q1")),
    ))
    adapter.install()
    full_wrapper, draft_wrapper = FullGraph.run, DraftGraph.run
    @functools.wraps(full_wrapper)
    def foreign(*args, **kwargs):
        return full_wrapper(*args, **kwargs)
    FullGraph.run = foreign
    try:
        with pytest.raises(AdapterValidationError, match="changed after installation"):
            adapter.uninstall()
        assert FullGraph.run is foreign
        assert DraftGraph.run is draft_wrapper
        FullGraph.run = full_wrapper
        adapter.uninstall()
        assert (FullGraph.run, DraftGraph.run) == originals
    finally:
        FullGraph.run, DraftGraph.run = originals

@pytest.mark.parametrize("during_install", [False, True])
def test_restore_failure_cleans_other_hooks_and_allows_retry(monkeypatch, during_install) -> None:
    from . import vllm_adapter
    originals = FullGraph.run, DraftGraph.run
    adapter = FailClosedMethodAdapter(_collector(), (
        _hook(FullGraph, PhaseDescriptor(PhaseKind.TARGET_FULL_GRAPH, "Q6")),
        _hook(DraftGraph, PhaseDescriptor(PhaseKind.DRAFT_MULTISTEP_GRAPH, "Q1")),
    ))
    blocked = True

    def failing_setattr(owner, name, value):
        if owner is DraftGraph and value is originals[1] and blocked:
            raise RuntimeError("restore temporarily refused")
        setattr(owner, name, value)
        if during_install and owner is DraftGraph and value is not originals[1]:
            raise KeyboardInterrupt("interrupt after assignment")

    monkeypatch.setattr(vllm_adapter, "setattr", failing_setattr, raising=False)
    try:
        if during_install:
            with pytest.raises(BaseExceptionGroup):
                adapter.install()
        else:
            adapter.install()
            with pytest.raises(ExceptionGroup):
                adapter.uninstall()
        assert FullGraph.run is originals[0]
        assert DraftGraph.run is not originals[1]
        blocked = False
        adapter.uninstall()
        assert (FullGraph.run, DraftGraph.run) == originals
    finally:
        FullGraph.run, DraftGraph.run = originals
