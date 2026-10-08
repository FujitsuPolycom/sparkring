"""SIRCL's fused all-reduce + residual add + RMSNorm at vLLM's post-all-reduce norm sites.

Where it applies. Models built on vLLM's DeepSeek-V3.2 code
(``models/deepseek_v32/nvidia``, GLM-5.3 among them) leave the row-parallel
outputs of attention and of the MLP or MoE un-reduced and hand them, with the
residual stream and the next RMSNorm, to vLLM's eager helper
``fused_allreduce_rms_norm(hidden_states, residual, norm)``
(``models/common/ops/fused_allreduce_rms_norm.py``): at every layer's
post-attention norm, at every later layer's input norm, at the final norm and
at the MTP draft's final norm (``model.py:152``, ``:166``, ``:327`` and
``mtp.py:162`` in the image's build), ``2 L`` calls per forward of ``L``
decoder layers. On GB10 (compute capability 12.1) the helper's FlashInfer path
never applies, because FlashInfer's size table has no 12.1 entry
(``compilation/passes/fusion/allreduce_rms_fusion.py:108-130``), so every call
is the tensor-parallel all-reduce followed by ``norm(reduced, residual)``. On
a tensor-parallel group with a SIRCL session that all-reduce is the session's
rank-ordered sum (the adapter's ``direct`` plan for every message within the
dispatch ceiling).

What it does. With ``SIRCL_FUSED_NORM=1`` SIRCL's communicator binds the
kernels of :mod:`sparkring_sircl.fused_norm` (``FusedAddRmsNorm``) to the
tensor-parallel group's session for the model's hidden size and installs the
pinned ``fused_allreduce_rms_norm`` shim (:mod:`.shims`), which wraps vLLM's
helper where it is defined and in every loaded vLLM module that imported it
(modules imported later import the wrapper). A call runs as one fused
collective when the group's communicator holds the bound kernels, ``norm`` is
vLLM's plain ``RMSNorm`` (a weight, no variance-size override, no subclass
that overrides its forward) and the kernels accept the tensors: BF16,
contiguous, 16-byte aligned, the bound hidden size, 1 to ``max_rows`` rows
(the GPU's multiprocessor count, 48 on GB10, or fewer with
``SIRCL_FUSED_NORM_MAX_ROWS``), within the session's dispatch ceiling, and a
size for which the session selects the one-shot or two-shot algorithm. Any
other call runs vLLM's helper unchanged. The fused call is one collective of
the session's sequence and speaks the plain all-reduce's wire protocol, so a
rank that declines and runs the plain all-reduce for the same message still
exchanges correctly with ranks that fused it; every input of the decision is
the same on every rank in practice. Each fused call is counted in the group's
receipt as ``all_reduce/sircl/fused_rms_norm``.

Exactness. The fused kernels reproduce, bit for bit, SIRCL's all-reduce
followed by the ``vllm_c`` provider of vLLM's ``fused_add_rms_norm``
(``torch.ops._C.fused_add_rms_norm``: the vectorized
``fused_add_rms_norm_kernel`` of ``csrc/libtorch_stable/layernorm_kernels.cu``
with 1024-thread blocks below 256 rows; that file and ``type_convert.cuh`` are
identical at the image's vLLM revision ``ab86b70`` and at ``4a87c588``): the
rank-ordered float32 sum rounded once to BF16, the BF16 residual add, the
kernel's per-thread, per-warp and per-block order of the sum of squares, IEEE
division, ``rsqrt.approx.f32``, and ``(x * inv) * weight`` rounded once
(:mod:`sparkring_sircl.fused_norm._reference`). vLLM's ``RMSNorm`` reaches
that kernel only when its IR op priority puts ``vllm_c`` first for
``fused_add_rms_norm``, which is the CUDA platform's default without inductor
compilation (``platforms/cuda.py:756-778``). With inductor compilation the
default is the ``native`` provider (``ir/ops/layernorm.py:43-62``), which
normalizes the unrounded float32 sum and rounds to BF16 before the weight
product, and ``VLLM_BATCH_INVARIANT=1`` selects another kernel; neither is
bit-identical to the fused kernels. Setup therefore refuses
``SIRCL_FUSED_NORM=1`` unless vLLM dispatches ``fused_add_rms_norm`` of BF16
rows of the model's width to ``vllm_c``; the worker fixes that priority for
its lifetime before it builds any group (``v1/worker/worker_base.py:97-99``).
The evidence for bit-identity is the kernels' GPU checks on emulated groups on
an RTX 5090 against the NumPy model of vLLM's kernel with the GPU's reciprocal
square root; nothing compares the fused kernels with the compiled vLLM kernel
on GB10.

Setup. :func:`bind` runs on every rank of the tensor-parallel group, eagerly,
before any CUDA graph capture: it installs the shim, checks the provider,
reads the hidden size from vLLM's current config and compiles the kernels.
The outcome, with the hidden size, row limit, algorithms and provider, is
voted over the group's CPU group, and any failure or disagreement raises
``SirclSetupError`` on every rank: no SIRCL session on the group, SparkRing's
four-rank sessions carrying its small all-reduces, the shim refused on an
unpinned vLLM, a provider other than ``vllm_c``, a hidden size the kernels do
not take (a multiple of 256 up to 8192 that splits into one chunk per rank)
or a session they cannot use (one or two lanes per peer).

Status: research-only. The adapter side has CPU tests on emulated groups
with a stand-in kernel; the kernels have run on emulated groups on an RTX 5090
only, not on GB10 or a ring.
"""

from __future__ import annotations

import functools
import importlib
import logging
import sys
from typing import Any

import torch

from . import groupops

logger = logging.getLogger("sircl.vllm.norm_fusion")

ENV = "SIRCL_FUSED_NORM"
HELPER_MODULE = "vllm.models.common.ops.fused_allreduce_rms_norm"
HELPER = "fused_allreduce_rms_norm"
EXACT_PROVIDER = "vllm_c"
RECEIPT_KEY = ("all_reduce", "sircl", "fused_rms_norm")
RECEIPT_REASON = f"all-reduce, residual add and RMSNorm in one kernel ({ENV}=1)"


class SirclFusedNorm:
    """One tensor-parallel group's bound fused kernels, as the shim calls them.

    ``kernel`` is a ``sparkring_sircl.fused_norm.FusedAddRmsNorm`` (or an
    object with its ``allreduce_add_rms_norm``, ``supports_fused_add_rms_norm``,
    ``hidden``, ``max_rows`` and ``stats``); ``adapter`` is the group's
    :class:`.adapter.GroupAdapter`, whose receipt counts every fused call.
    """

    def __init__(self, kernel: Any, adapter: Any, *, provider: str) -> None:
        self.kernel = kernel
        self.adapter = adapter
        self.provider = provider
        self.hidden = int(kernel.hidden)
        self.max_rows = int(kernel.max_rows)

    def supports_fused_add_rms_norm(self) -> bool:
        """True while the group is open and its session healthy."""
        return not self.adapter._closed and bool(self.kernel.supports_fused_add_rms_norm())

    def allreduce_add_rms_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                               weight: torch.Tensor, epsilon: float
                               ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(normed, residual)`` from one fused collective, or None to decline (see the module docstring)."""
        if not self.supports_fused_add_rms_norm():
            return None
        result = self.kernel.allreduce_add_rms_norm(hidden_states, residual, weight, epsilon)
        if result is not None:
            self.adapter._count(RECEIPT_KEY, RECEIPT_REASON)
        return result

    def describe(self) -> dict[str, Any]:
        """The bound configuration, for the receipt."""
        try:
            stats = dict(self.kernel.stats())
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            stats = {"error": f"{type(exc).__name__}: {exc}"}
        return {**stats, "hidden": self.hidden, "max_rows": self.max_rows, "provider": self.provider}


# -- seams the CPU tests replace ------------------------------------------------------------


def model_hidden_size() -> int:
    """The model's hidden size from vLLM's current config, which the worker sets while it builds groups."""
    from vllm.config import get_current_vllm_config

    model_config = getattr(get_current_vllm_config(), "model_config", None)
    if model_config is None:
        raise RuntimeError("vLLM's current config has no model config to read the hidden size from")
    return int(model_config.get_hidden_size())


def bind_kernel(session: Any, hidden: int) -> Any:
    """The fused kernels compiled for ``session`` and rows of ``hidden`` values."""
    from ..fused_norm import runtime

    return runtime.bind(session, hidden=hidden)


def image_norm_provider(device: Any, hidden: int) -> str:
    """The implementation vLLM's ``RMSNorm`` runs for the residual add and norm of BF16 rows of ``hidden``."""
    from vllm import envs

    if getattr(envs, "VLLM_BATCH_INVARIANT", False):
        return "batch-invariant (VLLM_BATCH_INVARIANT=1)"
    from vllm import ir

    rows = torch.zeros((1, hidden), dtype=torch.bfloat16, device=device)
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    implementation = ir.ops.fused_add_rms_norm.dispatch(rows, torch.zeros_like(rows), weight, 1e-6, None)
    return str(implementation.provider)


# -- binding --------------------------------------------------------------------------------


def _bind_here(communicator: Any) -> tuple[Any, str, int]:
    """Install the shim and compile the kernels on this rank; raises on any condition the module states."""
    from . import shims

    adapter = communicator.sircl
    if adapter is None or adapter.session is None or adapter.shared_from is not None:
        raise RuntimeError("the group has no SIRCL session of its own (SIRCL_GROUPS must name tp)")
    if adapter._tp4:
        raise RuntimeError("SparkRing's four-rank sessions carry this group's small all-reduces, so the "
                           "unfused path is not SIRCL's all-reduce")
    shims.install(["fused_allreduce_rms_norm"])
    hidden = model_hidden_size()
    provider = image_norm_provider(communicator.device, hidden)
    if provider != EXACT_PROVIDER:
        raise RuntimeError(
            f"vLLM's RMSNorm runs fused_add_rms_norm with the {provider} provider here, and the fused "
            f"kernels are bit-identical only to the {EXACT_PROVIDER} provider (the CUDA default without "
            "inductor compilation; see --ir-op-priority and the compilation mode)")
    return bind_kernel(adapter.session, hidden), provider, hidden


def bind(communicator: Any) -> SirclFusedNorm:
    """Bind the fused kernels to a tensor-parallel communicator's session on every rank (voted).

    Raises ``SirclSetupError`` on every rank when any rank cannot bind or the
    ranks disagree on the configuration.
    """
    from .adapter import SirclSetupError

    adapter = communicator.sircl
    name = getattr(getattr(adapter, "placement", None), "name", None) or "the tensor-parallel group"
    kernel = None
    provider = ""
    error = None
    settings = None
    try:
        kernel, provider, hidden = _bind_here(communicator)
        settings = {"hidden": hidden, "max_rows": int(kernel.max_rows), "provider": provider,
                    "algorithms": sorted(kernel.stats().get("prepared", ()))}
    except Exception as exc:  # noqa: BLE001 - voted below
        error = f"{type(exc).__name__}: {exc}"
    cpu_group = adapter.cpu_group if adapter is not None else communicator.cpu_group
    verdict = groupops.vote(cpu_group, (error, settings), compare=True)
    if verdict is not None:
        raise SirclSetupError(f"{ENV}=1 cannot run the fused all-reduce + residual add + RMSNorm on {name}: "
                              f"{verdict}; unset {ENV} to keep vLLM's all-reduce and RMSNorm")
    fused = SirclFusedNorm(kernel, adapter, provider=provider)
    adapter.attach_fused_norm(fused)
    logger.info("SIRCL fused all-reduce + RMSNorm bound on %s: hidden %d, rows 1-%d, %s, provider %s", name,
                fused.hidden, fused.max_rows, ", ".join(settings["algorithms"]) or "-", provider)
    return fused


# -- the shim -------------------------------------------------------------------------------


def _plain_rms_norm(norm: Any, rms_norm: type) -> bool:
    """Whether ``norm`` computes exactly vLLM's ``RMSNorm`` with a weight and no variance-size override."""
    cls = type(norm)
    if not isinstance(norm, rms_norm):
        return False
    if cls is not rms_norm and (getattr(cls, "forward_native", None) is not rms_norm.forward_native
                                or getattr(cls, "forward_cuda", None) is not rms_norm.forward_cuda):
        return False
    weighted = getattr(norm, "pass_weight_add", getattr(norm, "has_weight", False))
    return bool(weighted) and getattr(norm, "variance_size_override", None) is None


def install_shim(marker: str) -> None:
    """Wrap vLLM's helper where it is defined and where loaded vLLM modules bound it (idempotent)."""
    module = importlib.import_module(HELPER_MODULE)
    original = getattr(module, HELPER)
    if getattr(original, marker, None) is not None:
        return
    parallel_state = importlib.import_module("vllm.distributed.parallel_state")
    rms_norm = importlib.import_module("vllm.model_executor.layers.layernorm").RMSNorm

    @functools.wraps(original)
    def fused_allreduce_rms_norm(hidden_states, residual, norm):
        communicator = getattr(parallel_state.get_tp_group(), "device_communicator", None)
        fused = getattr(communicator, "sircl_fused_add_rms_norm", None)
        if fused is not None and residual is not None and _plain_rms_norm(norm, rms_norm):
            result = fused.allreduce_add_rms_norm(hidden_states, residual, norm.weight.data,
                                                  norm.variance_epsilon)
            if result is not None:
                return result
        return original(hidden_states, residual, norm)

    setattr(fused_allreduce_rms_norm, marker, original)
    setattr(module, HELPER, fused_allreduce_rms_norm)
    for loaded_name, loaded in list(sys.modules.items()):
        if (loaded is not None and loaded_name.startswith("vllm.")
                and getattr(loaded, "__dict__", {}).get(HELPER) is original):
            setattr(loaded, HELPER, fused_allreduce_rms_norm)
