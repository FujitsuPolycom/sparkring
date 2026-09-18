"""Execute B12X's transformed gate with CPU driver doubles and real Python GC."""

import ast
import gc
import os
from contextlib import contextmanager
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest

from integrations.b12x.patches.stream_gate_gc import apply


# This method is from b12x/preparation/_measurement.py, SHA-256 e45ca21666e5e056.
# An explicit source root runs the same tests against the complete build input.
SOURCE = """class _StreamGate:
    @contextmanager
    def hold(self, stream):
        self.streams[stream.cuda_stream] = stream
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        target = self.sequence
        self._check(self.driver.cuStreamWaitValue32(
            stream.cuda_stream, self.device_pointer, target,
            int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        ))
        try:
            yield
        finally:
            # A later release must also satisfy an earlier, still queued wait.
            self.flag.value = target
"""


def upstream_source():
    root = os.environ.get("SPARKRING_B12X_SOURCE_ROOT")
    if root:
        return (Path(root) / "b12x/preparation/_measurement.py").read_text(
            encoding="utf-8"
        )
    return SOURCE


def load(source):
    # Keep the real gate/helper AST; importing the complete B12X module needs CUDA.
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in {"_StreamGate", "_defer_automatic_gc"}
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.startswith("_gc_deferral_")
                for target in node.targets
            )
        )
    ]
    namespace = dict(contextmanager=contextmanager, gc=gc, threading=threading)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "_measurement.py", "exec"),
        namespace,
    )
    return namespace


def gate(namespace, *, wait_error=None):
    owner = namespace["_StreamGate"].__new__(namespace["_StreamGate"])
    owner.streams, owner.sequence = {}, 0
    owner.device_pointer, owner.flag = 1, NS(value=0)

    def wait(*_args):
        if wait_error:
            raise wait_error
        return (0,)

    owner.driver = NS(
        cuStreamWaitValue32=wait, CUstreamWaitValue_flags=NS(CU_STREAM_WAIT_VALUE_GEQ=0)
    )
    owner._check = lambda result: None
    return owner


@pytest.fixture(autouse=True)
def preserve_gc():
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    gc.collect()
    yield
    gc.set_threshold(*thresholds)
    gc.collect()
    gc.enable() if enabled else gc.disable()


def collect_cycle_during_hold(owner):
    finalized = []

    class Library:
        def __init__(self):
            self.cycle = self

        def __del__(self):
            finalized.append(owner.flag.value)

    gc.enable()
    gc.set_threshold(20, 1, 1)
    with owner.hold(NS(cuda_stream=1)):
        victim = Library()
        del victim
        # Retained allocations cross the automatic collection threshold.
        allocations = [[index] for index in range(1000)]
        assert len(allocations) == 1000
        inside = tuple(finalized)
    gc.collect()
    return inside, finalized


def test_baseline_finalizes_an_unreachable_library_while_gate_is_closed():
    inside, finalized = collect_cycle_during_hold(gate(load(upstream_source())))
    assert inside == (0,)
    assert finalized == [0]


def test_automatic_library_finalization_waits_for_gate_release():
    inside, finalized = collect_cycle_during_hold(gate(load(apply(upstream_source()))))
    assert inside == ()
    assert finalized == [1]
    assert gc.isenabled()


@pytest.mark.parametrize("failure", ["body", "enqueue"])
def test_error_releases_device_wait_before_restoring_gc(monkeypatch, failure):
    namespace = load(apply(upstream_source()))
    owner = gate(
        namespace, wait_error=RuntimeError("enqueue") if failure == "enqueue" else None
    )
    gc.enable()
    enabled_after_release = []
    real_enable = gc.enable

    def enable():
        enabled_after_release.append(owner.flag.value)
        real_enable()

    monkeypatch.setattr(gc, "enable", enable)
    with pytest.raises(RuntimeError, match=failure):
        with owner.hold(NS(cuda_stream=1)):
            raise RuntimeError("body")
    assert enabled_after_release == [1]
    assert namespace["_gc_deferral_count"] == 0
    assert gc.isenabled()


def test_disabled_gc_stays_disabled_after_nested_gates():
    namespace = load(apply(upstream_source()))
    outer, inner = gate(namespace), gate(namespace)
    gc.disable()
    with outer.hold(NS(cuda_stream=1)):
        with inner.hold(NS(cuda_stream=2)):
            assert not gc.isenabled()
        assert not gc.isenabled()
    assert not gc.isenabled()
    assert (outer.flag.value, inner.flag.value) == (1, 1)


def test_nested_gate_does_not_restore_gc_before_outer_release():
    namespace = load(apply(upstream_source()))
    outer, inner = gate(namespace), gate(namespace)
    gc.enable()
    with outer.hold(NS(cuda_stream=1)):
        with inner.hold(NS(cuda_stream=2)):
            assert not gc.isenabled()
        assert not gc.isenabled()
        assert outer.flag.value == 0
    assert gc.isenabled()


def test_concurrent_gate_does_not_restore_gc_while_another_gate_waits():
    namespace = load(apply(upstream_source()))
    first, second = gate(namespace), gate(namespace)
    entered, release = threading.Event(), threading.Event()
    errors = []

    def worker():
        try:
            with second.hold(NS(cuda_stream=2)):
                entered.set()
                assert release.wait(5)
                assert not gc.isenabled()
        except BaseException as error:
            errors.append(error)

    gc.enable()
    thread = threading.Thread(target=worker)
    try:
        with first.hold(NS(cuda_stream=1)):
            thread.start()
            assert entered.wait(5)
        assert not gc.isenabled()
        assert second.flag.value == 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert gc.isenabled()
    assert namespace["_gc_deferral_count"] == 0


def test_explicit_collection_is_not_blocked_by_automatic_gc_guard():
    owner = gate(load(apply(upstream_source())))
    finalized = []

    class Library:
        def __del__(self):
            finalized.append(owner.flag.value)

    with owner.hold(NS(cuda_stream=1)):
        value = Library()
        value.cycle = value
        del value
        gc.collect()
        assert finalized == [0]


def test_reference_count_finalizers_are_not_blocked_by_automatic_gc_guard():
    owner = gate(load(apply(upstream_source())))
    finalized = []

    class Library:
        def __del__(self):
            finalized.append(owner.flag.value)

    with owner.hold(NS(cuda_stream=1)):
        value = Library()
        del value
        assert finalized == [0]


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("target = self.sequence", "target = 1"),
        apply(SOURCE),
        SOURCE + SOURCE,
    ],
)
def test_unfamiliar_or_already_patched_gate_is_rejected(source):
    with pytest.raises(ValueError):
        apply(source)
