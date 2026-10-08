"""A compiled CuTe DSL launcher called with a prebuilt argument block (no torch, no DSL import).

A call through the DSL's compiled function converts every argument in Python first: it casts each number to
its annotated type, asks each pointer and stream object for the address of its value, and flattens those
addresses into the argument array of one call into the compiled host function, which packs the kernel's
parameters and launches it. :class:`FastLaunch` builds that array once per launcher: one ctypes slot per
argument, laid out as the DSL lays out the value (8 bytes for a pointer, a stream or a 64-bit number, 4 for a
32-bit number), followed by the executor's own extra entries (its result word and its kernel handles). A call
stores the new values in the slots and calls the compiled host function with the array: the same host code,
the same parameter packing, the same kernel.

Before it is used, the block is checked against the DSL's own conversion of sample arguments, byte for byte
for every argument; :func:`build` returns None, and the caller keeps the DSL's call, when the check fails,
when the DSL's executor lacks an expected attribute, or when ``SIRCL_FAST_LAUNCH`` is ``0``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import weakref
from collections.abc import Callable, Sequence
from typing import Any, Optional

logger = logging.getLogger("sircl")


def kind_of(annotation: Any) -> tuple[str, Any]:
    """``("stream", c_uint64)``, ``("number", <ctypes type>)`` or ``("pointer", c_uint64)`` for a jit
    function's argument annotation."""
    if getattr(annotation, "__name__", "") == "CUstream":
        return "stream", ctypes.c_uint64
    width, signed = getattr(annotation, "width", None), getattr(annotation, "signed", None)
    if isinstance(width, int) and isinstance(signed, bool):
        types = {(8, True): ctypes.c_int8, (8, False): ctypes.c_uint8, (16, True): ctypes.c_int16,
                 (16, False): ctypes.c_uint16, (32, True): ctypes.c_int32, (32, False): ctypes.c_uint32,
                 (64, True): ctypes.c_int64, (64, False): ctypes.c_uint64}
        if (width, signed) not in types:
            raise ValueError(f"no ctypes slot for a {width}-bit number")
        return "number", types[(width, signed)]
    return "pointer", ctypes.c_uint64


def _address(entry: Any) -> int:
    return entry if isinstance(entry, int) else int(entry.value)


def annotations_of(compiled: Any) -> list[Any]:
    """The jit function's argument annotations in argument order. CuTe DSL keeps them in the argument binder's
    metadata: ``execution_args`` (``ExecutionArgs``, nvidia-cutlass-dsl 4.7) or ``args_spec`` (earlier
    releases)."""
    binder = getattr(compiled, "execution_args", None)
    if binder is None:
        binder = getattr(compiled, "args_spec", None)
    if binder is None:
        raise AttributeError(f"{type(compiled).__name__} has neither execution_args nor args_spec")
    return list(binder._meta.annotated_types)


class FastLaunch:
    """One launcher's compiled host entry, its prebuilt argument array and its result word."""

    def __init__(self, capi: Callable[[Any], Any], slots: Sequence[Any], packed: Any, result: Any, name: str) -> None:
        self._capi = capi
        self._slots = tuple(slots)
        self._packed = packed
        self._result = result
        self._name = name
        self._lock = threading.Lock()

    def __call__(self, *values: int) -> None:
        """Launch with ``values`` in the jit function's argument order (pointers and the stream as integer
        addresses and handles)."""
        with self._lock:
            for slot, value in zip(self._slots, values):
                slot.value = value
            self._capi(self._packed)
            if self._result is not None and self._result.value != 0:
                raise RuntimeError(f"{self._name}: the launch failed with CUDA error {self._result.value}")


def default_executor(compiled: Any) -> Any:
    """The compiled function's executor for the current device, created as its first call creates it."""
    with compiled._executor_lock:
        if compiled._default_executor is None:
            compiled._default_executor = weakref.proxy(compiled).to(None)
    return compiled._default_executor


def build(compiled: Any, *, make_pointer: Callable[[int], Any], make_stream: Callable[[int], Any],
          name: str, executor: Optional[Any] = None) -> Optional[FastLaunch]:
    """The fast launch of ``compiled`` (a CuTe DSL ``JitCompiledFunction``), or None with the reason logged."""
    if os.environ.get("SIRCL_FAST_LAUNCH", "1").strip() == "0":
        return None
    try:
        kinds = [kind_of(annotation) for annotation in annotations_of(compiled)]
        samples, dsl_args = [], []
        for index, (kind, ctype) in enumerate(kinds):
            if kind == "pointer":
                value = 0x10000 * (index + 1)
                dsl_args.append(make_pointer(value))
            elif kind == "stream":
                value = 0x7F000 + index
                dsl_args.append(make_stream(value))
            else:
                bits = ctypes.sizeof(ctype) * 8
                value = (1000 + 37 * index) & ((1 << (bits - 1)) - 1)
                dsl_args.append(value)
            samples.append(value)
        exe_args, _adapted = compiled.generate_execution_args(*dsl_args)
        if len(exe_args) != len(kinds):
            raise ValueError(f"{len(exe_args)} execution arguments for {len(kinds)} jit arguments")
        slots = [ctype() for _, ctype in kinds]
        for slot, value, entry in zip(slots, samples, exe_args):
            slot.value = value
            size = ctypes.sizeof(slot)
            if ctypes.string_at(ctypes.addressof(slot), size) != ctypes.string_at(_address(entry), size):
                raise ValueError("an argument's bytes differ from the DSL's conversion")
        executor = executor if executor is not None else default_executor(compiled)
        extra = []
        if executor._has_cuda_result:
            extra.append(int(executor._cuda_result_addr))
        extra += [_address(pointer) for pointer in (executor._kernel_ptrs or ())]
        if len(extra) != executor._num_extra_args:
            raise ValueError("the executor's extra arguments differ from its count")
        packed = (ctypes.c_void_p * (len(slots) + len(extra)))()
        for index, slot in enumerate(slots):
            packed[index] = ctypes.addressof(slot)
        for index, address in enumerate(extra, start=len(slots)):
            packed[index] = address
        return FastLaunch(executor.capi_func, slots, packed,
                          executor.cuda_result if executor._has_cuda_result else None, name)
    except Exception as error:  # noqa: BLE001 - any surprise keeps the DSL's own call
        logger.info("fast launch of %s unavailable, the DSL's call stays: %s", name, error)
        return None


__all__ = ["FastLaunch", "annotations_of", "build", "default_executor", "kind_of"]
