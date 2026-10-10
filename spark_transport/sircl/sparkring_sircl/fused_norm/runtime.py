"""Host glue of the fused all-reduce + residual add + RMSNorm.

``FusedAddRmsNorm(session, hidden=6144)`` binds the kernels of ``_kernel`` to
a constructed ring session (``sparkring_sircl.oneshot.AllReduce``) and
performs, on every rank of the session's group,

    normed, residual = fused(x, residual, weight, eps)

with ``x`` this rank's partial sum (BF16, ``[..., hidden]``), ``residual`` the
residual stream (updated in place to ``all_reduce(x) + residual``), ``weight``
the RMSNorm weight and ``normed`` a new tensor (or ``out=``). The result is
what vLLM's ``fused_add_rms_norm`` returns for ``all_reduce(x)``; see
``_reference`` for the exact arithmetic.

Contract:

* **Same wire protocol as the session's own all-reduce.** The kernels use the
  session's slots, flags, doorbells, epoch and poison words with the plain
  kernels' op codes, so the fused call is one collective of the session's
  sequence; a rank that runs the plain all-reduce for the same message and
  algorithm exchanges data with a rank that runs the fused one correctly.
* **Rank-invariant choice.** ``should_fuse`` decides from dtype, shape,
  contiguity, size and pointer alignment; the algorithm follows the
  session's ``select_algorithm`` (one-shot or two-shot; a Swing size is
  declined). Allocators return aligned tensors, so ranks agree in practice;
  if they did not, the declining rank's plain all-reduce still matches.
* **Fail-stop.** A peer wait beyond the session's wait limit poisons the
  session exactly as a plain collective does; ``check_health`` raises.
* **CUDA graphs.** ``prepare`` compiles the launchers; inside a capture an
  unprepared launcher raises, every call must use the session's capture
  stream (the session's ``_order_stream`` rule) and ``out`` is allocated
  from the graph's pool when omitted.
* **Limits.** BF16 only; ``hidden`` a multiple of 256 and at most 8192
  (``row_packs = hidden / 8`` threads per CTA); at most ``max_rows`` rows
  (default: the multiprocessor count, so every CTA of a launch is resident;
  ``SIRCL_FUSED_NORM_MAX_ROWS`` lowers it); one or two lanes per peer;
  two-shot needs the session's multi-phase support.
* **vLLM.** :func:`bind` prepares the kernels for a session (every rank,
  eagerly, before capture). ``allreduce_add_rms_norm(hidden_states,
  residual, weight, epsilon)`` is the post-all-reduce RMSNorm of a decoder
  sublayer (vLLM's ``fused_allreduce_rms_norm``): it returns ``(normed,
  residual)`` or None to decline. ``supports_fused_add_rms_norm()`` and
  ``try_fused_add_rms_norm(inp, residual, weight, epsilon)`` are the in-place
  form: the normalized result replaces ``inp`` and the new residual replaces
  ``residual``; False declines. After a decline the caller runs the session's
  all-reduce and vLLM's ``fused_add_rms_norm``.

The session fields that this module reads are listed in ``_TransportView``;
that class is the single place to change when the session's arena view
changes.

Origin: SparkRing's fused-norm kernels.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch

from . import _geometry as geo

logger = logging.getLogger("sircl")

ENV_MAX_ROWS = "SIRCL_FUSED_NORM_MAX_ROWS"
SUPPORTED_SESSION_API = 1


class FusedNormUnavailable(RuntimeError):
    """The session or the configuration cannot run the fused kernels."""


@dataclass(frozen=True)
class _TransportView:
    """What the kernels need from the session, read once from its fields.

    Fields of ``RoceOneshotAllReduce`` (``sparkring_sircl/oneshot/runtime.py``)
    used: ``device``, ``rank``, ``world_size``, ``topology``, ``lane_count``,
    ``max_size``, ``dispatch_limit_bytes``, ``spin_limit``, ``_recv_base``,
    ``_flag_base``, ``_send_base``, ``_ctrl_base``, ``_slot_bytes``,
    ``_layout.slots``, ``_layout.flag_stride``, ``_epoch_address``,
    ``_poison_address``, ``_available``; methods ``select_algorithm``,
    ``check_health``, ``_order_stream``, ``_mark_stream``; the lock ``_lock``.
    """

    runtime: Any
    device: torch.device
    rank: int
    world_size: int
    lanes: int
    max_size: int
    dispatch_limit_bytes: int
    spin_limit: int
    recv_base: int
    flag_base: int
    send_base: int
    ctrl_base: int
    slot_bytes: int
    slots: int
    flag_stride: int
    epoch_address: int
    poison_address: int
    twoshot: bool

    @classmethod
    def of(cls, runtime) -> "_TransportView":
        from ..oneshot import runtime as session_module

        api = getattr(session_module, "API_VERSION", None)
        if api != SUPPORTED_SESSION_API:
            raise FusedNormUnavailable(f"sparkring_sircl.oneshot API version {api}; the fused kernels need "
                                       f"{SUPPORTED_SESSION_API}")
        needed = ("device", "rank", "world_size", "topology", "lane_count", "max_size",
                  "dispatch_limit_bytes", "spin_limit", "_recv_base", "_flag_base", "_send_base",
                  "_ctrl_base", "_slot_bytes", "_layout", "_epoch_address", "_poison_address",
                  "_available", "_lock", "select_algorithm", "check_health", "_order_stream",
                  "_mark_stream")
        missing = [name for name in needed if not hasattr(runtime, name)]
        if missing:
            raise FusedNormUnavailable(f"the session lacks {missing}; the fused kernels need a "
                                       "sparkring_sircl ring session")
        if runtime.topology != "direct":
            raise FusedNormUnavailable(f"topology {runtime.topology!r}: the fused kernels need the direct topology")
        if int(runtime.lane_count) not in (1, 2):
            raise FusedNormUnavailable(f"{runtime.lane_count} lanes per peer: the fused kernels need 1 or 2")
        return cls(
            runtime=runtime,
            device=runtime.device,
            rank=int(runtime.rank),
            world_size=int(runtime.world_size),
            lanes=int(runtime.lane_count),
            max_size=int(runtime.max_size),
            dispatch_limit_bytes=int(runtime.dispatch_limit_bytes),
            spin_limit=int(runtime.spin_limit),
            recv_base=int(runtime._recv_base),
            flag_base=int(runtime._flag_base),
            send_base=int(runtime._send_base),
            ctrl_base=int(runtime._ctrl_base),
            slot_bytes=int(runtime._slot_bytes),
            slots=int(runtime._layout.slots),
            flag_stride=int(runtime._layout.flag_stride),
            epoch_address=int(runtime._epoch_address),
            poison_address=int(runtime._poison_address),
            twoshot=bool(runtime._available.get("twoshot", False)),
        )


def _multiprocessors(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


class FusedAddRmsNorm:
    """The fused all-reduce + residual add + RMSNorm on one ring session, for one hidden size."""

    def __init__(self, runtime, *, hidden: int, max_rows: Optional[int] = None) -> None:
        self._view = _TransportView.of(runtime)
        self.hidden = int(hidden)
        if self.hidden % (geo.WARP * geo.BF16_PER_PACK) or not 256 <= self.hidden <= 8192:
            raise FusedNormUnavailable(f"hidden {hidden}: the fused kernels need a multiple of 256 up to 8192")
        self.row_packs = self.hidden // geo.BF16_PER_PACK
        view = self._view
        if self.row_packs % view.world_size:
            raise FusedNormUnavailable(f"a row of {self.row_packs} packs does not split into {view.world_size} chunks")
        if view.world_size * view.lanes > self.row_packs:
            raise FusedNormUnavailable("world_size * lanes exceeds the threads of one CTA")
        limit = _multiprocessors(view.device)
        env = os.environ.get(ENV_MAX_ROWS, "").strip()
        wanted = int(env) if env else (limit if max_rows is None else int(max_rows))
        if wanted < 1:
            raise FusedNormUnavailable(f"{ENV_MAX_ROWS}={wanted} must be positive")
        self.max_rows = min(wanted, limit)
        with torch.cuda.device(view.device):
            # stage, mid and tail arrival counters (atom.inc returns each to 0 after every launch)
            self._counters = torch.zeros(4, dtype=torch.int32, device=view.device)
        self._launchers: dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- policy ---------------------------------------------------------------------

    def select_algorithm(self, nbytes: int, algorithm: Optional[str] = None) -> Optional[str]:
        """``oneshot`` or ``twoshot`` for a message of ``nbytes``, or None (Swing sizes, no two-shot)."""
        chosen = algorithm or self._view.runtime.select_algorithm(int(nbytes))
        if chosen == geo.TWOSHOT and not self._view.twoshot:
            return None
        return chosen if chosen in geo.ALGORITHMS else None

    def should_fuse(self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                    algorithm: Optional[str] = None) -> bool:
        """Whether ``__call__`` accepts these tensors (decided from rank-invariant properties)."""
        return self._decline_reason(inp, residual, weight, algorithm) is None

    def _decline_reason(self, inp, residual, weight, algorithm=None) -> Optional[str]:
        view = self._view
        if inp.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return "dtype is not bfloat16"
        if not (inp.is_cuda and inp.device == view.device and residual.device == view.device
                and weight.device == view.device):
            return "tensor not on the session's device"
        if inp.dim() < 1 or inp.shape[-1] != self.hidden:
            return f"last dimension is not {self.hidden}"
        if residual.shape != inp.shape or weight.dim() != 1 or weight.shape[0] != self.hidden:
            return "residual or weight shape differs"
        if not (inp.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
            return "tensor not contiguous"
        rows = inp.numel() // self.hidden
        if not 1 <= rows <= self.max_rows:
            return f"{rows} rows outside 1..{self.max_rows}"
        nbytes = inp.numel() * inp.element_size()
        if nbytes > view.dispatch_limit_bytes or nbytes > view.max_size:
            return "message above the session's dispatch limit"
        if (inp.data_ptr() | residual.data_ptr() | weight.data_ptr()) % geo.PACK_BYTES:
            return "pointer not 16-byte aligned"
        if self.select_algorithm(nbytes, algorithm) is None:
            return "size selects an algorithm without a fused kernel"
        return None

    # -- compilation ---------------------------------------------------------------

    def _key(self, algorithm: str) -> tuple:
        view = self._view
        return (algorithm, view.world_size, view.rank, self.row_packs, view.lanes, view.slots,
                view.flag_stride, view.device.index if view.device.index is not None else torch.cuda.current_device())

    def prepare(self, algorithms: Optional[tuple[str, ...]] = None) -> None:
        """Compile the launchers (one-shot, and two-shot when the session has it); refused in a capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("fused norm preparation is refused inside a CUDA graph capture")
        wanted = algorithms or ((geo.ONESHOT, geo.TWOSHOT) if self._view.twoshot else (geo.ONESHOT,))
        for name in wanted:
            if name == geo.TWOSHOT and not self._view.twoshot:
                raise FusedNormUnavailable("the session has no two-shot all-reduce")
            self._launcher(name, capturing=False)

    def _launcher(self, algorithm: str, capturing: bool):
        with self._lock:
            launcher = self._launchers.get(algorithm)
        if launcher is not None:
            return launcher
        if capturing:
            raise RuntimeError(f"fused norm launcher {algorithm!r} was not prepared before CUDA graph capture; "
                               "call prepare()")
        from . import _kernel

        with torch.cuda.device(self._view.device):
            launcher = _kernel.get_launcher(*self._key(algorithm))
        with self._lock:
            self._launchers[algorithm] = launcher
        return launcher

    # -- execution -----------------------------------------------------------------

    def __call__(self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, *,
                 out: Optional[torch.Tensor] = None, stream: object = None,
                 algorithm: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One fused collective: returns ``(normed, residual)`` with ``residual`` updated in place."""
        from ..oneshot._compile import current_cuda_stream

        view = self._view
        runtime = view.runtime
        with runtime._lock:
            runtime.check_health()
            reason = self._decline_reason(inp, residual, weight, algorithm)
            if reason is not None:
                raise ValueError(f"input is not eligible for the fused all-reduce + RMSNorm: {reason}")
            if out is not None and (out.shape != inp.shape or out.dtype != inp.dtype or out.device != inp.device
                                    or not out.is_contiguous() or out.data_ptr() % geo.PACK_BYTES
                                    or out.data_ptr() in (residual.data_ptr(), weight.data_ptr())):
                raise ValueError("out must be a separate aligned contiguous tensor shaped like the input")
            nbytes = inp.numel() * inp.element_size()
            rows = inp.numel() // self.hidden
            chosen = self.select_algorithm(nbytes, algorithm)
            context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with torch.cuda.device(view.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                launcher = self._launcher(chosen, capturing)
                if out is None:
                    out = torch.empty_like(inp)
                runtime._order_stream(capturing)
                launcher(inp.data_ptr(), residual.data_ptr(), weight.data_ptr(), out.data_ptr(), rows, nbytes,
                         view.recv_base, view.flag_base, view.send_base, view.ctrl_base, view.slot_bytes,
                         view.epoch_address, view.poison_address, self._counters.data_ptr(), view.spin_limit,
                         float(eps), current_cuda_stream())
                runtime._mark_stream(capturing)
        if not capturing:
            runtime.check_health()
        return out, residual

    # -- vLLM's fused all-reduce + RMSNorm interfaces ---------------------------------------

    def allreduce_add_rms_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                               epsilon: float, *, out: Optional[torch.Tensor] = None
                               ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """The tensor-parallel all-reduce of ``hidden_states`` with vLLM's ``fused_add_rms_norm``, or None.

        Returns ``(normed, residual)``: ``residual`` updated in place to
        ``all_reduce(hidden_states) + residual`` and ``normed`` (a new tensor,
        or ``out``) the RMSNorm of it times ``weight``, the bits of the session's
        all-reduce followed by vLLM's kernel. None declines (the reasons of
        ``should_fuse``, a negative ``epsilon``, a poisoned or closed session):
        the caller then runs the session's all-reduce and its own norm, which
        exchange data with ranks that fused the same message.
        """
        if (epsilon < 0 or not self.supports_fused_add_rms_norm()
                or not self.should_fuse(hidden_states, residual, weight)):
            return None
        return self(hidden_states, residual, weight, epsilon, out=out)

    def supports_fused_add_rms_norm(self) -> bool:
        """True while the session is healthy: vLLM may offer fused calls."""
        runtime = self._view.runtime
        return not getattr(runtime, "_closed", False) and not runtime.poisoned

    def try_fused_add_rms_norm(self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                               epsilon: float) -> bool:
        """The fused operation in place (``inp`` becomes the normalized rows), or False to decline."""
        if epsilon < 0 or inp.data_ptr() == residual.data_ptr() or not self.should_fuse(inp, residual, weight):
            return False
        self(inp, residual, weight, epsilon, out=inp)
        return True

    def stats(self) -> dict[str, Any]:
        """Configuration, for logs."""
        view = self._view
        return {"hidden": self.hidden, "row_packs": self.row_packs, "max_rows": self.max_rows,
                "world_size": view.world_size, "rank": view.rank, "lanes": view.lanes,
                "twoshot": view.twoshot, "prepared": sorted(self._launchers)}


def bind(session, *, hidden: int, max_rows: Optional[int] = None) -> FusedAddRmsNorm:
    """The fused kernels of ``hidden``-wide rows on ``session``, compiled and ready for CUDA graph capture.

    Call it on every rank of the session's group, eagerly, after the session
    is constructed and before any capture (compilation is refused inside
    one); it exchanges nothing. ``FusedNormUnavailable`` names the condition a
    session or width does not meet; the caller votes the outcome over the
    group so that every rank binds or none does.
    """
    fused = FusedAddRmsNorm(session, hidden=hidden, max_rows=max_rows)
    fused.prepare()
    return fused


__all__ = ["ENV_MAX_ROWS", "FusedAddRmsNorm", "FusedNormUnavailable", "bind"]
