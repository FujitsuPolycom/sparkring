"""The prebuilt-argument launch (``oneshot/_fast_launch.py``) against a stand-in of the CuTe DSL's compiled
function: slots laid out as the DSL lays out each argument, checked byte for byte, the executor's extra
entries appended, values stored per call, a failed launch raised, and every refusal keeping the DSL's call."""

from __future__ import annotations

import ctypes
import threading

import pytest

from sparkring_sircl.oneshot import _fast_launch


class Number:
    def __init__(self, width, signed):
        self.width, self.signed = width, signed


class CUstream:
    pass


class Pointer:
    pass


class Meta:
    def __init__(self, annotations):
        self.annotated_types = annotations


class Spec:
    def __init__(self, annotations):
        self._meta = Meta(annotations)


class Executor:
    def __init__(self, kernels=2, result=True):
        self.cuda_result = ctypes.c_int32(0) if result else None
        self._has_cuda_result = result
        self._cuda_result_addr = ctypes.addressof(self.cuda_result) if result else None
        self._kernel_ptrs = [ctypes.c_void_p(0xABC0 + k) for k in range(kernels)]
        self._num_extra_args = kernels + (1 if result else 0)
        self.calls = []

    def capi_func(self, packed):
        self.calls.append([packed[i] for i in range(len(packed))])


class Compiled:
    """The DSL compiled function's surface the fast launch reads: its argument annotations, its per-call
    conversion (one pointer per argument to the value's bytes) and its executor."""

    def __init__(self, annotations, executor, corrupt=None):
        self.args_spec = Spec(annotations)
        self._executor_lock = threading.Lock()
        self._default_executor = executor
        self.corrupt = corrupt
        self.keep = []

    def generate_execution_args(self, *args):
        entries = []
        for index, (annotation, value) in enumerate(zip(_fast_launch.annotations_of(self), args)):
            kind, ctype = _fast_launch.kind_of(annotation)
            raw = value if kind == "number" else value.address
            if index == self.corrupt:
                raw += 1
            box = ctype(raw)
            self.keep.append(box)
            entries.append(ctypes.c_void_p(ctypes.addressof(box)))
        return entries, []


class CompiledWithBinder(Compiled):
    """nvidia-cutlass-dsl 4.7 keeps the annotations in the argument binder ``execution_args``."""

    def __init__(self, annotations, executor):
        super().__init__(annotations, executor)
        self.execution_args = self.args_spec
        del self.args_spec


class Address:
    def __init__(self, address):
        self.address = address


ANNOTATIONS = [Pointer, Pointer, Number(32, True), Number(64, True), Number(32, False), CUstream]


def _build(compiled):
    return _fast_launch.build(compiled, make_pointer=Address, make_stream=Address, name="test launcher")


def test_slots_follow_the_dsl_layout_and_carry_each_calls_values(monkeypatch):
    monkeypatch.delenv("SIRCL_FAST_LAUNCH", raising=False)
    executor = Executor()
    fast = _build(Compiled(ANNOTATIONS, executor))
    assert fast is not None
    fast(0x1000, 0x2000, -5, 1 << 40, 0xFFFFFFFF, 0x55)
    packed = executor.calls[-1]
    assert len(packed) == len(ANNOTATIONS) + 3
    values = [ctypes.c_uint64.from_address(packed[0]).value, ctypes.c_uint64.from_address(packed[1]).value,
              ctypes.c_int32.from_address(packed[2]).value, ctypes.c_int64.from_address(packed[3]).value,
              ctypes.c_uint32.from_address(packed[4]).value, ctypes.c_uint64.from_address(packed[5]).value]
    assert values == [0x1000, 0x2000, -5, 1 << 40, 0xFFFFFFFF, 0x55]
    assert packed[6] == executor._cuda_result_addr and packed[7:] == [0xABC0, 0xABC1]
    executor.cuda_result.value = 700
    with pytest.raises(RuntimeError, match="CUDA error 700"):
        fast(0, 0, 0, 0, 0, 0)


def test_refusals_keep_the_dsl_call(monkeypatch):
    monkeypatch.delenv("SIRCL_FAST_LAUNCH", raising=False)
    assert _build(Compiled(ANNOTATIONS, Executor(), corrupt=3)) is None       # bytes differ from the DSL's
    broken = Executor()
    broken._num_extra_args += 1
    assert _build(Compiled(ANNOTATIONS, broken)) is None                      # extra entries miscounted
    assert _build(Compiled([Number(12, True)], Executor())) is None           # no slot for the width
    monkeypatch.setenv("SIRCL_FAST_LAUNCH", "0")
    assert _build(Compiled(ANNOTATIONS, Executor())) is None
    monkeypatch.setenv("SIRCL_FAST_LAUNCH", "1")
    assert _build(Compiled(ANNOTATIONS, Executor(result=False))) is not None


def test_annotations_come_from_either_binder(monkeypatch):
    monkeypatch.delenv("SIRCL_FAST_LAUNCH", raising=False)
    executor = Executor()
    fast = _build(CompiledWithBinder(ANNOTATIONS, executor))
    assert fast is not None
    fast(0x1000, 0x2000, -5, 1 << 40, 0xFFFFFFFF, 0x55)
    assert ctypes.c_int64.from_address(executor.calls[-1][3]).value == 1 << 40
    bare = Compiled(ANNOTATIONS, Executor())
    del bare.args_spec
    assert _build(bare) is None                                                # no binder: the DSL's call
