"""Execute the selected B12X stream gate and verify automatic-GC ordering on CPU."""

import ast
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest


def load_gate():
    path = Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"]) / "b12x/preparation/_measurement.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Preserve implementation bodies. Only CUDA allocation and driver calls use
    # doubles, so the protected source itself owns collection and flag ordering.
    nodes = [
        node for node in tree.body
        if (isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in {"_StreamGate", "_defer_automatic_gc"})
        or (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id.startswith("_gc_deferral_")
                    for target in node.targets))
    ]
    assert any(isinstance(node, ast.ClassDef) and node.name == "_StreamGate" for node in nodes)
    namespace = dict(contextmanager=contextmanager, gc=gc, threading=threading)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_StreamGate"]


def gate(cls, *, wait_error=None):
    owner = cls.__new__(cls)
    owner.streams, owner.sequence = {}, 0
    owner.device_pointer, owner.flag = 1, NS(value=0)

    def wait(*_args):
        if wait_error:
            raise wait_error
        return (0,)

    owner.driver = NS(cuStreamWaitValue32=wait, CUstreamWaitValue_flags=NS(CU_STREAM_WAIT_VALUE_GEQ=0))
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


def test_automatic_finalizers_run_only_after_device_gate_release():
    owner = gate(load_gate())
    finalized = []

    class Library:
        def __init__(self):
            self.cycle = self

        def __del__(self):
            finalized.append(owner.flag.value)

    gc.enable()
    gc.set_threshold(20, 1, 1)
    with owner.hold(NS(cuda_stream=1)):
        value = Library()
        del value
        allocations = [[index] for index in range(1000)]
        assert len(allocations) == 1000
        assert finalized == [], "Automatic finalization ran while CUDA work was gated"
        assert not gc.isenabled()
    gc.collect()
    assert finalized == [1]
    assert gc.isenabled()


@pytest.mark.parametrize("failure", [None, "body", "enqueue"])
def test_flag_release_precedes_gc_restoration_on_every_exit(monkeypatch, failure):
    owner = gate(load_gate(), wait_error=RuntimeError("enqueue") if failure == "enqueue" else None)
    gc.enable()
    enabled_after_release = []
    real_enable = gc.enable

    def enable():
        enabled_after_release.append(owner.flag.value)
        real_enable()

    monkeypatch.setattr(gc, "enable", enable)

    def submit():
        with owner.hold(NS(cuda_stream=1)):
            if failure == "body":
                raise RuntimeError("body")

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            submit()
    else:
        submit()
    assert enabled_after_release == [1]
    assert gc.isenabled()


def test_caller_disabled_collection_stays_disabled_after_nested_gates():
    cls = load_gate()
    outer, inner = gate(cls), gate(cls)
    gc.disable()
    with outer.hold(NS(cuda_stream=1)):
        with inner.hold(NS(cuda_stream=2)):
            assert not gc.isenabled()
        assert not gc.isenabled()
    assert not gc.isenabled()
    assert (outer.flag.value, inner.flag.value) == (1, 1)


def test_nested_gate_cannot_restore_gc_while_outer_gate_is_closed():
    cls = load_gate()
    outer, inner = gate(cls), gate(cls)
    gc.enable()
    with outer.hold(NS(cuda_stream=1)):
        with inner.hold(NS(cuda_stream=2)):
            assert not gc.isenabled()
        assert not gc.isenabled()
        assert outer.flag.value == 0
    assert gc.isenabled()


def test_concurrent_gate_cannot_restore_gc_while_another_gate_is_closed():
    cls = load_gate()
    first, second = gate(cls), gate(cls)
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
