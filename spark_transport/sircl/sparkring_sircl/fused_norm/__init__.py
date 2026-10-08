"""All-reduce fused with the residual add and RMSNorm (``fused_add_rms_norm``) on a SIRCL ring session.

Status: research-only. CPU tests of the geometry, the reference arithmetic
and the wire protocol, and GPU checks on an emulated group, exist; nothing
here has run on a Spark.

``bind(session, hidden=6144)`` returns a ``FusedAddRmsNorm``
(``sparkring_sircl.fused_norm.runtime``) with the kernels of ``_kernel``
compiled for a constructed ring session (``sparkring_sircl.oneshot.AllReduce``);
``fused.allreduce_add_rms_norm(x, residual, weight, eps)`` then returns
``(normed, residual)``, or None to decline: the sum of every rank's ``x``,
plus ``residual`` (updated in place), normalized by RMS and multiplied by
``weight``, the bits of the session's all-reduce followed by vLLM's
``fused_add_rms_norm``. The kernels speak the session's wire protocol
(one-shot and two-shot), so the package needs no change to the session; it
reads the session's arena addresses, epoch and poison words through one
adapter (``runtime._TransportView``).

Modules:

* ``_geometry``: pack, chunk, stripe and flag arithmetic shared by the kernel
  and the CPU tests;
* ``_reference``: the operation in NumPy, operation for operation;
* ``_ptx``: the inline PTX the kernels use;
* ``_kernel``: the CuTe DSL kernels (one CTA per row);
* ``runtime``: host glue (eligibility, preparation, launch, CUDA-graph
  capture rules, and vLLM's ``try_fused_add_rms_norm`` interface).

Only ``_geometry`` and ``_reference`` load without the CuTe DSL and torch;
the other modules import them on first use.
"""

from __future__ import annotations

from . import _geometry, _reference

API_VERSION = 1


def __getattr__(name: str):
    if name in ("FusedAddRmsNorm", "FusedNormUnavailable", "bind"):
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(name)


__all__ = ["API_VERSION", "FusedAddRmsNorm", "FusedNormUnavailable", "_geometry", "_reference", "bind"]
