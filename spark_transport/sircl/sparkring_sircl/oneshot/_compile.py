"""Compile helper of SIRCL's CuTe DSL kernels.

:func:`compile_launcher` compiles one kernel launch object for its example
argument types with the CUTLASS CuTe DSL and logs the kernel name, its cache
key and the compile time. The DSL's on-disk cache (``CUTE_DSL_CACHE_DIR``)
lets a later process load a compiled kernel instead of compiling it again.
:func:`current_cuda_stream` passes the caller's torch stream to a launcher as a
CUDA driver handle, :func:`current_stream_handle` as an integer. :func:`fast_launch`
gives a launcher's prebuilt-argument call (:mod:`._fast_launch`).

:func:`raise_if_kernel_resolution_frozen` honours b12x's process-wide kernel
freeze when b12x provides one (it ships either as ``b12x`` or inside FlashInfer
as ``flashinfer.experimental.b12x``) and otherwise returns. SIRCL has no freeze
switch of its own: an unprepared launcher already refuses to compile inside a
CUDA graph capture.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

from . import _fast_launch

logger = logging.getLogger("sircl")

_B12X_HOMES = ("b12x", "flashinfer.experimental.b12x")
_freeze_check: Any = None
_freeze_resolved = False


def _b12x_freeze_check():
    global _freeze_check, _freeze_resolved
    if not _freeze_resolved:
        for home in _B12X_HOMES:
            try:
                module = importlib.import_module(f"{home}._lib.runtime_control")
            except ImportError:
                continue
            _freeze_check = getattr(module, "raise_if_kernel_resolution_frozen", None)
            if _freeze_check is not None:
                break
        _freeze_resolved = True
    return _freeze_check


def raise_if_kernel_resolution_frozen(kind: str, *, target: object = None, cache_key: object = None) -> None:
    """Raise when b12x's kernel freeze is on; return otherwise."""
    check = _b12x_freeze_check()
    if check is not None:
        check(kind, target=target, cache_key=cache_key)


_pointer_factory: Any = None


def make_pointer(address: int, assumed_align: int = 16) -> Any:
    """A CuTe runtime pointer to global memory at ``address`` (``Uint32`` elements).

    Uses b12x's runtime pointer wrapper when b12x is installed (the serving
    image's kernels are compiled through it), else the CuTe DSL's own
    ``make_ptr``.
    """
    global _pointer_factory
    if _pointer_factory is None:
        factory = None
        for home in _B12X_HOMES:
            try:
                factory = getattr(importlib.import_module(f"{home}._lib.utils"), "make_ptr", None)
            except ImportError:
                continue
            if factory is not None:
                break
        if factory is None:
            from cutlass.cute.runtime import make_ptr as factory
        _pointer_factory = factory
    import cutlass

    return _pointer_factory(cutlass.Uint32, int(address), cute.AddressSpace.gmem, assumed_align=assumed_align)


def current_cuda_stream() -> cuda.CUstream:
    """The caller's current torch CUDA stream as a driver stream handle."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def current_stream_handle(device_index: int) -> int:
    """The current torch CUDA stream of device ``device_index`` as an integer handle (torch's raw-stream
    query when it has one, which builds no stream object)."""
    if _RAW_STREAM is not None:
        return int(_RAW_STREAM(device_index))
    return int(torch.cuda.current_stream(device_index).cuda_stream)


def fast_launch(compiled: Any, name: str) -> Any:
    """The prebuilt-argument call of ``compiled`` (:func:`._fast_launch.build`), or None."""
    return _fast_launch.build(compiled, make_pointer=make_pointer, make_stream=cuda.CUstream, name=name)


# The CuTe DSL's front end keeps per-compilation state that concurrent compiles
# would share, so compilations in one process run one at a time.
_COMPILE_LOCK = threading.Lock()


def compile_launcher(launch: object, *example_args: object, name: str, cache_key: tuple) -> Any:
    """Compile ``launch`` for ``example_args`` and return the executor (one compilation at a time)."""
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=cache_key)
    with _COMPILE_LOCK:
        started = time.perf_counter()
        compiled = cute.compile(launch, *example_args)
    logger.info("compiled %s %s in %.2f s", name, cache_key, time.perf_counter() - started)
    return compiled


__all__ = ["compile_launcher", "current_cuda_stream", "current_stream_handle", "fast_launch", "make_pointer",
           "raise_if_kernel_resolution_frozen"]
