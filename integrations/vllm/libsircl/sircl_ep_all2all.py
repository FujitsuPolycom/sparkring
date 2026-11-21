"""Opt-in SIRCL-EP GPU-driven all-to-all manager for vLLM DP + EP.

Activated when ``SIRCL_EP_ALL2ALL=1`` is set and the ``libsircl`` vLLM plugin
runs. Subclasses ``AgRsAll2AllManager`` so vLLM's existing
``MoEPrepareAndFinalizeNaiveDPEPModular`` pipeline runs ``sirclEpDispatch`` and
``sirclEpCombine`` directly over ``libsircl``'s point-to-point RDMA channels
(zero extra queue pairs), with an automatic fallback to ``AgRsAll2AllManager``
for unsupported calls (extra tensors, non-linear expert maps, or capacities
above ``SIRCL_EP_MAX_TOKENS``).

Two exchange paths, both on the same ``libsircl`` entry points:

``dynamic``
    The exchange capacity is ``max`` over every rank's token count, learnt from
    vLLM's ``_get_sizes`` collective once per layer, so it wastes no rows. That
    collective returns a host-visible count, which is one host synchronization
    per MoE layer and cannot be captured into a CUDA graph.

``static`` (``SIRCL_EP_STATIC_A2A=1``)
    The capacity is one constant for the whole run, ``SIRCL_EP_STATIC_A2A_MAX_TOKENS``,
    so ``dispatch`` and ``combine`` make no collective, read no device result and
    synchronize nothing: they are plain kernel launches into buffers reused from a
    grow-only arena. The manager then contributes no host synchronization and no
    allocation that a CUDA graph capture would refuse; whether vLLM captures the
    layer is vLLM's decision. Rows a peer never routes stay in the padded tail of
    every rank's slice; the kernel masks them with a non-local expert ID and a zero
    weight, so they never reach an expert.

Why the capacity is a constant rather than a per-step or per-graph value: the
capacity is also the row stride of every rank's slice, and ``combine`` strides with
it, so **all ranks of the group must pass the same value in the same layer**. Any
rule that reads this rank's own state (its token count, a bucket derived from that
count, its replay size) can differ from a peer's and breaks the channel with a size
mismatch. In particular the data-parallel ranks do not share a token count on eager steps - one
rank holds the request while the others hold none - and a rung derived from that
count made the first request fail at DP4. A constant is identical on every rank for
every eager step and for every captured graph, so graphs of different sizes can be
replayed against each other.

Eligibility is decided from state that is already identical across ranks:

- ``capturing``: vLLM captures the same sizes on every rank, so the constant
  applies; a capture size above the cap takes its own size as the capacity, and
  ``SIRCL_EP_STATIC_A2A_MAX_TOKENS`` must cover the largest capture size for every
  graph to share one stride.
- eager: the bound comes from ``ForwardContext.dp_metadata``, which vLLM
  synchronizes across the data-parallel ranks before the forward pass - the per-rank
  counts and their maximum, as CPU tensors, so reading them costs no device
  synchronization. Every rank then makes the same decision: static when the bound
  fits the cap, dynamic when it does not, and dynamic when the metadata is absent
  (a step whose bound cannot be proved common).
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import torch

_LOGGER_NAME = f"vllm.{__name__}"

try:
    from vllm.logger import init_logger as _init_logger

    logger = _init_logger(_LOGGER_NAME)
except ImportError:
    logger = logging.getLogger(_LOGGER_NAME)

try:
    from vllm.distributed.device_communicators.all2all import (
        AgRsAll2AllManager as _AgRsAll2AllManager,
    )

    ALL2ALL_BASE_IS_STUB = False
except ImportError:  # offline imports: install() refuses this base

    class _AgRsAll2AllManager:
        """Placeholder for vLLM's manager so this module imports without vLLM."""

    ALL2ALL_BASE_IS_STUB = True

ENV_FLAG = "SIRCL_EP_ALL2ALL"
ENV_MAX_TOKENS = "SIRCL_EP_MAX_TOKENS"
ENV_PREALLOC_TOKENS = "SIRCL_EP_PREALLOC_TOKENS"
ENV_STATIC_A2A = "SIRCL_EP_STATIC_A2A"
ENV_STATIC_MAX_TOKENS = "SIRCL_EP_STATIC_A2A_MAX_TOKENS"
DEFAULT_STATIC_MAX_TOKENS = 96

# One greppable summary line per manager: first static dispatch, every STATS_INTERVAL
# dispatches, and at process exit, so a benchmark can read the counters out of a container log.
STATS_TAG = "SIRCL_EP_STATS"
STATS_INTERVAL = 5000

_INSTALLED = False
_ACTIVE_META = threading.local()
_LIVE_MANAGERS: "weakref.WeakSet[Any]" = weakref.WeakSet()
_ATEXIT_REGISTERED = False

# NCCL datatype constants matching spark_transport/libsircl/include/nccl.h
_NCCL_FLOAT16 = 6
_NCCL_FLOAT32 = 7
_NCCL_BFLOAT16 = 9
_TORCH_TO_NCCL_DTYPE = {
    torch.float16: _NCCL_FLOAT16,
    torch.float32: _NCCL_FLOAT32,
    torch.bfloat16: _NCCL_BFLOAT16,
}


def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_FLAG, "").strip().lower() in ("1", "true", "on", "yes")


def static_enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_STATIC_A2A, "").strip().lower() in ("1", "true", "on", "yes")


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


def _shared_capacity(bound: int, cap: int) -> int:
    """The exchange capacity for a step whose cross-rank token bound is ``bound``.

    ``cap`` when the bound fits it, ``0`` when it does not (the caller then takes the
    dynamic path). The capacity is a constant, never a function of the bound: it is
    also the stride every rank's slice is laid out with, so a value that differs
    between ranks breaks the channel with a size mismatch.
    """
    return cap if cap > 0 and 0 < bound <= cap else 0


def _tokens_across_dp() -> int | None:
    """vLLM's token bound for this step across the data-parallel ranks, or ``None``.

    ``ForwardContext.dp_metadata`` carries the per-rank token counts and their
    maximum as CPU tensors, synchronized before the forward pass, so reading them
    costs no device synchronization and every rank reads the same numbers. Both
    fields are consulted because the DeepSeek-V4.1 CED hook rewrites the per-rank
    counts without rewriting the maximum: the larger of the two is the safe bound.
    ``None`` means this step cannot prove a bound every rank shares.
    """
    try:
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().dp_metadata
    except Exception:
        return None
    if metadata is None:
        return None
    bound = 0
    for name in ("max_tokens_across_dp_cpu", "num_tokens_across_dp_cpu"):
        value = getattr(metadata, name, None)
        if value is None:
            continue
        if getattr(getattr(value, "device", None), "type", "cpu") != "cpu":
            # A device-resident count would synchronize the device, and inside a capture it would
            # be illegal: treat the step as having no provable bound instead.
            return None
        try:
            bound = max(bound, int(value.max()) if hasattr(value, "max") else int(value))
        except Exception:
            return None
    return bound or None


def _stats_logger():
    """vLLM's logger for this module, resolved at call time so it carries vLLM's handlers.

    A process that imports this module before vLLM configures logging keeps a logger
    with no handler, so the summary would never reach the container log; resolving it
    per summary takes vLLM's configured logger.
    """
    try:
        from vllm.logger import init_logger

        return init_logger(_LOGGER_NAME)
    except Exception:
        return logger


def _log_at_exit() -> None:
    """Write one ``SIRCL_EP_STATS`` summary per live manager when the process ends."""
    for manager in list(_LIVE_MANAGERS):
        try:
            manager.log_stats("atexit")
        except Exception:
            continue


def _register_for_exit(manager: Any) -> None:
    """Track a manager for the exit summary, installing the single ``atexit`` handler once."""
    global _ATEXIT_REGISTERED
    try:
        _LIVE_MANAGERS.add(manager)
    except TypeError:  # no weak reference: the exit line is skipped, nothing else changes
        return
    if not _ATEXIT_REGISTERED:
        _ATEXIT_REGISTERED = True
        atexit.register(_log_at_exit)


def _last_error(lib: ctypes.CDLL, comm: Any) -> str:
    if not hasattr(lib, "ncclGetLastError"):
        return ""
    try:
        raw = lib.ncclGetLastError(comm)
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    except Exception:
        return ""


class _EpArena:
    """Device buffers for one exchange geometry: ``(world, hidden_dim, dtype, top_k, device)``.

    ``hidden``/``ids``/``weights`` hold the dispatch output, ``world * max_tokens``
    rows: every rank's slice, its routed rows first and its padded tail behind
    them. ``combine_out`` holds ``max_tokens`` rows, the combine output for this
    rank's own tokens. ``counts`` receives the per-peer row counts the kernel
    writes (diagnostics only; nothing on the hot path reads it).

    Both row counts grow monotonically. A growth allocates new tensors and
    returns the displaced ones so the caller can retire, not free, them: a CUDA
    graph captured earlier has the old addresses baked in, so keeping every
    retired allocation alive keeps every replay pointing at live memory.
    """

    __slots__ = (
        "world",
        "hidden_dim",
        "dtype",
        "top_k",
        "device",
        "rows",
        "combine_rows",
        "hidden",
        "ids",
        "weights",
        "combine_out",
        "counts",
    )

    def __init__(self, world: int, hidden_dim: int, dtype: torch.dtype, top_k: int, device: torch.device):
        self.world = world
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.top_k = top_k
        self.device = device
        self.rows = 0
        self.combine_rows = 0
        self.hidden: torch.Tensor | None = None
        self.ids: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.combine_out: torch.Tensor | None = None
        self.counts = torch.empty((world,), dtype=torch.int32, device=device)

    def covers(self, rows: int, combine_rows: int) -> bool:
        return rows <= self.rows and combine_rows <= self.combine_rows

    def grow(self, rows: int, combine_rows: int) -> list[torch.Tensor]:
        """Allocate buffers covering these row counts; return the displaced tensors to retire."""
        displaced: list[torch.Tensor] = []
        if rows > self.rows:
            displaced += [tensor for tensor in (self.hidden, self.ids, self.weights) if tensor is not None]
            self.hidden = torch.empty((rows, self.hidden_dim), dtype=self.dtype, device=self.device)
            self.ids = torch.empty((rows, self.top_k), dtype=torch.int32, device=self.device)
            self.weights = torch.empty((rows, self.top_k), dtype=torch.float32, device=self.device)
            self.rows = rows
        if combine_rows > self.combine_rows:
            if self.combine_out is not None:
                displaced.append(self.combine_out)
            self.combine_out = torch.empty((combine_rows, self.hidden_dim), dtype=self.dtype, device=self.device)
            self.combine_rows = combine_rows
        return displaced


def _is_linear_expert_map(expert_map: torch.Tensor, world: int, rank: int) -> bool:
    if expert_map.ndim != 1 or expert_map.dtype != torch.int32:
        return False
    num_experts = int(expert_map.numel())
    if num_experts < world or (num_experts % world) != 0:
        return False
    per_rank = num_experts // world
    expected = torch.full((num_experts,), -1, dtype=torch.int32, device=expert_map.device)
    lo = rank * per_rank
    expected[lo : lo + per_rank] = torch.arange(per_rank, dtype=torch.int32, device=expert_map.device)
    return bool(torch.equal(expert_map, expected))


@dataclass
class _ActiveEpCall:
    used_sircl: bool
    num_tokens: int = 0
    max_tokens: int = 0
    hidden_dim: int = 0
    dtype: torch.dtype = torch.bfloat16
    device: torch.device | None = None
    lib: Any = None
    comm: Any = None
    arena: _EpArena | None = None
    workspace: Any = None
    workspace_bytes: int = 0
    static: bool = False
    rank: int = 0


class SirclEpAll2AllManager(_AgRsAll2AllManager):
    """Drop-in ``AgRsAll2AllManager`` replacement backed by ``sirclEpDispatch``/``sirclEpCombine``."""

    def __init__(self, cpu_group, tcp_store_group=None, device_comm=None):
        super().__init__(cpu_group, tcp_store_group)
        self._device_comm = device_comm
        self._workspace: torch.Tensor | None = None
        self._workspace_bytes: int = 0
        self._retired: list[torch.Tensor] = []
        self._arenas: dict[tuple[int, int, torch.dtype, int, torch.device], _EpArena] = {}
        self._linear_map_cache: dict[tuple[int, int, int, int], bool] = {}
        self._active_call: _ActiveEpCall | None = None
        self._max_tokens_cap = _parse_int_env(ENV_MAX_TOKENS, 4096)
        self._prealloc_tokens = _parse_int_env(ENV_PREALLOC_TOKENS, 512)
        self._static_max_tokens = min(
            _parse_int_env(ENV_STATIC_MAX_TOKENS, DEFAULT_STATIC_MAX_TOKENS), self._max_tokens_cap
        ) if static_enabled() else 0
        self._logged_static_buckets: set[int] = set()
        self._warned_oversized_captures: set[int] = set()
        self._disabled_reason: str | None = None
        self.sircl_dispatches: int = 0
        self.sircl_combines: int = 0
        self.agrs_fallbacks: int = 0
        self.static_dispatches: int = 0
        self.dynamic_dispatches: int = 0
        self.size_exchanges: int = 0
        self.arena_allocations: int = 0
        self.last_num_tokens: int = 0
        self.last_max_tokens: int = 0
        _register_for_exit(self)

    # -- device access, overridden by offline tests that have no CUDA device ------------------

    def _capturing(self) -> bool:
        """Whether this stream is inside a CUDA graph capture: no allocation may happen there."""
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    def _shared_tokens(self) -> int | None:
        """The token bound every data-parallel rank agrees on for this step, or ``None``."""
        return _tokens_across_dp()

    def _on_device(self, tensor: torch.Tensor) -> bool:
        """Whether the tensor lives in device memory, as ``libsircl`` requires."""
        return bool(tensor.is_cuda)

    def _stream_ptr(self, device: torch.device) -> ctypes.c_void_p:
        return ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)

    def stats(self) -> dict[str, int]:
        """Counters for a benchmark run: which path each call took and what it had to allocate."""
        return {
            "dispatches": self.sircl_dispatches,
            "combines": self.sircl_combines,
            "static_dispatches": self.static_dispatches,
            "dynamic_dispatches": self.dynamic_dispatches,
            "size_exchanges": self.size_exchanges,
            "agrs_fallbacks": self.agrs_fallbacks,
            "arena_allocations": self.arena_allocations,
            "retired_buffers": len(self._retired),
            "last_num_tokens": self.last_num_tokens,
            "last_max_tokens": self.last_max_tokens,
        }

    def log_stats(self, reason: str) -> None:
        """Write one ``SIRCL_EP_STATS`` line: the counters a benchmark reads out of a log."""
        counters = " ".join(f"{key}={value}" for key, value in self.stats().items())
        _stats_logger().info(
            "%s reason=%s static_cap=%d %s", STATS_TAG, reason, self._static_max_tokens, counters
        )

    # -- setup ------------------------------------------------------------------------------

    def _resolve_sircl(self, dist_group: Any):
        if self._disabled_reason is not None:
            return None, None, 0, 0
        comm_obj = None
        if self._device_comm is not None:
            comm_obj = getattr(self._device_comm, "pynccl_comm", None)
        if comm_obj is None:
            dcomm = getattr(dist_group, "device_communicator", None)
            if dcomm is not None:
                comm_obj = getattr(dcomm, "pynccl_comm", None)
        if comm_obj is None or getattr(comm_obj, "disabled", False):
            return None, None, 0, 0
        nccl_wrap = getattr(comm_obj, "nccl", None)
        lib = getattr(nccl_wrap, "lib", None) if nccl_wrap is not None else None
        comm = getattr(comm_obj, "comm", None)
        if lib is None or comm is None:
            return None, None, 0, 0
        if not _bind_ep_prototypes(lib):
            self._disabled_reason = "libsircl lacks sirclEpDispatch/sirclEpCombine"
            logger.warning("SirclEpAll2AllManager disabled: %s", self._disabled_reason)
            return None, None, 0, 0
        world = int(getattr(comm_obj, "world_size", dist_group.world_size))
        rank = int(getattr(comm_obj, "rank", dist_group.rank_in_group))
        if world < 3 or world > 8 or world != int(dist_group.world_size):
            return None, None, 0, 0
        return lib, comm, world, rank

    def _ensure_workspace(
        self,
        lib: ctypes.CDLL,
        world: int,
        max_tokens: int,
        hidden_bytes: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        need = int(lib.sirclEpWorkspaceBytes(world, max_tokens, hidden_bytes))
        if need <= 0:
            return None
        if self._workspace is not None and self._workspace_bytes >= need and self._workspace.device == device:
            return self._workspace
        if self._capturing():
            return None
        alloc_tokens = max(max_tokens, min(self._prealloc_tokens, self._max_tokens_cap))
        alloc_bytes = int(lib.sirclEpWorkspaceBytes(world, alloc_tokens, hidden_bytes))
        if self._workspace is not None:
            self._retired.append(self._workspace)
            self.arena_allocations += 1
        self._workspace = torch.empty((alloc_bytes,), dtype=torch.uint8, device=device)
        self._workspace_bytes = alloc_bytes
        return self._workspace

    def _ensure_arena(
        self,
        world: int,
        max_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        top_k: int,
        device: torch.device,
    ) -> _EpArena | None:
        """Buffers for this geometry covering ``world * max_tokens`` and ``max_tokens`` rows."""
        rows = world * max_tokens
        key = (world, hidden_dim, dtype, top_k, device)
        arena = self._arenas.get(key)
        if arena is None:
            if self._capturing():
                return None
            arena = _EpArena(world, hidden_dim, dtype, top_k, device)
            self._arenas[key] = arena
        if arena.covers(rows, max_tokens):
            return arena
        if self._capturing():
            return None
        self._retired.extend(arena.grow(rows, max_tokens))
        self.arena_allocations += 1
        return arena

    def _check_linear_map(self, expert_map: torch.Tensor | None, world: int, rank: int) -> bool:
        if expert_map is None:
            return False
        key = (expert_map.data_ptr(), int(expert_map.numel()), world, rank)
        cached = self._linear_map_cache.get(key)
        if cached is not None:
            return cached
        if self._capturing():
            return False
        ok = _is_linear_expert_map(expert_map, world, rank)
        self._linear_map_cache[key] = ok
        return ok

    # -- the pipeline's two calls -----------------------------------------------------------

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        meta = getattr(_ACTIVE_META, "current", None)
        dist_group = self._get_comm_group(is_sequence_parallel)
        lib, comm, world, rank = self._resolve_sircl(dist_group)
        can_use = (
            lib is not None
            and extra_tensors is None
            and meta is not None
            and self._on_device(hidden_states)
            and hidden_states.ndim == 2
            and hidden_states.dtype in _TORCH_TO_NCCL_DTYPE
            and topk_ids.ndim == 2
            and topk_weights.ndim == 2
        )
        max_tokens = 0
        is_static = False
        if can_use:
            num_experts, expert_map = meta
            num_tokens = int(hidden_states.shape[0])
            hidden_dim = int(hidden_states.shape[1])
            hidden_bytes = hidden_dim * hidden_states.element_size()
            top_k = int(topk_ids.shape[1])
            capacity = 0
            if self._static_max_tokens:
                if self._capturing():
                    # Every rank captures the same sizes in lockstep, so the constant capacity
                    # applies; a capture size above it keeps its own size so that graph can run.
                    capacity = max(num_tokens, self._static_max_tokens)
                    if num_tokens > self._static_max_tokens and num_tokens not in self._warned_oversized_captures:
                        self._warned_oversized_captures.add(num_tokens)
                        logger.warning(
                            "SIRCL-EP static capacity: capture size %d exceeds %s=%d; raise it to %d so every "
                            "captured graph shares one exchange stride",
                            num_tokens,
                            ENV_STATIC_MAX_TOKENS,
                            self._static_max_tokens,
                            num_tokens,
                        )
                else:
                    # Eager: the bound comes from metadata vLLM synchronized across the DP ranks,
                    # so every rank decides alike. Absent metadata means no provable common
                    # bound, and every rank takes the dynamic path.
                    bound = self._shared_tokens()
                    if bound is not None:
                        capacity = _shared_capacity(max(int(bound), num_tokens), self._static_max_tokens)
            if capacity:
                max_tokens, is_static = capacity, True
            else:
                sizes = self._get_sizes(num_tokens, dist_group)
                self.size_exchanges += 1
                max_tokens = max(max(int(s) for s in sizes), 1)
            is_linear = self._check_linear_map(expert_map, world, rank)
            if (
                num_experts < world
                or (num_experts % world) != 0
                or (hidden_bytes % 16) != 0
                or top_k < 1
                or top_k > 16
                or not is_linear
            ):
                can_use = False
            elif max_tokens > self._max_tokens_cap:
                if self._workspace is None and not self._capturing():
                    self._ensure_workspace(
                        lib,
                        world,
                        min(self._prealloc_tokens, self._max_tokens_cap),
                        hidden_bytes,
                        hidden_states.device,
                    )
                can_use = False

        arena = None
        if can_use:
            ws = self._ensure_workspace(lib, world, max_tokens, hidden_bytes, hidden_states.device)
            arena = (
                self._ensure_arena(world, max_tokens, hidden_dim, hidden_states.dtype, top_k, hidden_states.device)
                if ws is not None
                else None
            )
            if ws is None or arena is None:
                can_use = False

        if not can_use:
            self.agrs_fallbacks += 1
            self._active_call = _ActiveEpCall(used_sircl=False)
            return super().dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel=is_sequence_parallel,
                extra_tensors=extra_tensors,
            )

        h_in = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        ids_in = topk_ids if (topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()) else topk_ids.to(torch.int32).contiguous()
        wts_in = topk_weights if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()) else topk_weights.to(torch.float32).contiguous()

        total_rows = world * max_tokens
        h_out = arena.hidden[:total_rows]
        ids_out = arena.ids[:total_rows]
        wts_out = arena.weights[:total_rows]
        stream_ptr = self._stream_ptr(hidden_states.device)

        rc = int(
            lib.sirclEpDispatch(
                ctypes.c_void_p(h_in.data_ptr() if num_tokens > 0 else 0),
                ctypes.c_void_p(ids_in.data_ptr() if num_tokens > 0 else 0),
                ctypes.c_void_p(wts_in.data_ptr() if num_tokens > 0 else 0),
                ctypes.c_void_p(h_out.data_ptr()),
                ctypes.c_void_p(ids_out.data_ptr()),
                ctypes.c_void_p(wts_out.data_ptr()),
                ctypes.c_void_p(arena.counts.data_ptr()),
                ctypes.c_void_p(ws.data_ptr()),
                ctypes.c_size_t(self._workspace_bytes),
                ctypes.c_int(num_tokens),
                ctypes.c_int(max_tokens),
                ctypes.c_size_t(hidden_bytes),
                ctypes.c_int(top_k),
                ctypes.c_int(num_experts),
                comm,
                stream_ptr,
            )
        )
        if rc != 0:
            err_text = _last_error(lib, comm)
            if self.sircl_dispatches == 0:
                self._disabled_reason = f"sirclEpDispatch returned rc={rc}" + (
                    f" ({err_text})" if err_text else ""
                )
                logger.warning(
                    "SirclEpAll2AllManager falling back to AgRs (%s)", self._disabled_reason
                )
                self.agrs_fallbacks += 1
                self._active_call = _ActiveEpCall(used_sircl=False)
                return super().dispatch(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    is_sequence_parallel=is_sequence_parallel,
                    extra_tensors=extra_tensors,
                )
            raise RuntimeError(
                f"sirclEpDispatch failed with rc={rc}"
                f" (rank={rank} num_tokens={num_tokens} max_tokens={max_tokens} static={is_static}"
                f" hidden_bytes={hidden_bytes} top_k={top_k}"
                f" num_experts={num_experts} dispatches={self.sircl_dispatches}"
                f" combines={self.sircl_combines}"
                + (f" err={err_text!r}" if err_text else "")
                + ")"
            )

        self.sircl_dispatches += 1
        self.last_num_tokens = num_tokens
        self.last_max_tokens = max_tokens
        if is_static:
            self.static_dispatches += 1
            if max_tokens not in self._logged_static_buckets:
                self._logged_static_buckets.add(max_tokens)
                logger.info(
                    "SIRCL-EP static all-to-all: capacity %d tokens (%d exchange rows) for %d routed tokens",
                    max_tokens,
                    total_rows,
                    num_tokens,
                )
                self.log_stats("first-static")
        else:
            self.dynamic_dispatches += 1
        if self.sircl_dispatches % STATS_INTERVAL == 0:
            self.log_stats("interval")
        self._active_call = _ActiveEpCall(
            used_sircl=True,
            num_tokens=num_tokens,
            max_tokens=max_tokens,
            hidden_dim=hidden_dim,
            dtype=h_in.dtype,
            device=hidden_states.device,
            lib=lib,
            comm=comm,
            arena=arena,
            workspace=ws,
            workspace_bytes=self._workspace_bytes,
            static=is_static,
            rank=rank,
        )
        return h_out, wts_out, ids_out

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        call = self._active_call
        self._active_call = None
        if call is None or not call.used_sircl or call.arena is None or call.workspace is None:
            return super().combine(hidden_states, is_sequence_parallel=is_sequence_parallel)

        h_in = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        h_out = call.arena.combine_out[: call.num_tokens]
        nccl_dt = _TORCH_TO_NCCL_DTYPE[call.dtype]
        stream_ptr = self._stream_ptr(call.device)

        rc = int(
            call.lib.sirclEpCombine(
                ctypes.c_void_p(h_in.data_ptr()),
                ctypes.c_void_p(h_out.data_ptr() if call.num_tokens > 0 else 0),
                ctypes.c_void_p(call.workspace.data_ptr()),
                ctypes.c_size_t(call.workspace_bytes),
                ctypes.c_int(call.num_tokens),
                ctypes.c_int(call.max_tokens),
                ctypes.c_size_t(call.hidden_dim),
                ctypes.c_int(nccl_dt),
                call.comm,
                stream_ptr,
            )
        )
        if rc != 0:
            err_text = _last_error(call.lib, call.comm)
            raise RuntimeError(
                f"sirclEpCombine failed with rc={rc}"
                f" (rank={call.rank} num_tokens={call.num_tokens} max_tokens={call.max_tokens}"
                f" static={call.static} hidden_dim={call.hidden_dim}"
                f" dispatches={self.sircl_dispatches} combines={self.sircl_combines}"
                + (f" err={err_text!r}" if err_text else "")
                + ")"
            )
        self.sircl_combines += 1
        return h_out


def _bind_ep_prototypes(lib: ctypes.CDLL) -> bool:
    if not (
        hasattr(lib, "sirclEpWorkspaceBytes")
        and hasattr(lib, "sirclEpDispatch")
        and hasattr(lib, "sirclEpCombine")
    ):
        return False
    if getattr(lib, "_sircl_ep_bound", False):
        return True
    lib.sirclEpWorkspaceBytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    lib.sirclEpWorkspaceBytes.restype = ctypes.c_size_t
    lib.sirclEpDispatch.argtypes = [
        ctypes.c_void_p,  # hidden_in
        ctypes.c_void_p,  # topk_ids_in
        ctypes.c_void_p,  # topk_weights_in
        ctypes.c_void_p,  # hidden_out
        ctypes.c_void_p,  # topk_ids_out
        ctypes.c_void_p,  # topk_weights_out
        ctypes.c_void_p,  # recv_counts_out
        ctypes.c_void_p,  # workspace
        ctypes.c_size_t,  # workspace_bytes
        ctypes.c_int,     # num_tokens
        ctypes.c_int,     # max_tokens
        ctypes.c_size_t,  # hidden_bytes
        ctypes.c_int,     # top_k
        ctypes.c_int,     # num_experts
        ctypes.c_void_p,  # comm
        ctypes.c_void_p,  # stream
    ]
    lib.sirclEpDispatch.restype = ctypes.c_int
    lib.sirclEpCombine.argtypes = [
        ctypes.c_void_p,  # expert_out_in
        ctypes.c_void_p,  # hidden_out
        ctypes.c_void_p,  # workspace
        ctypes.c_size_t,  # workspace_bytes
        ctypes.c_int,     # num_tokens
        ctypes.c_int,     # max_tokens
        ctypes.c_size_t,  # hidden_dim
        ctypes.c_int,     # datatype
        ctypes.c_void_p,  # comm
        ctypes.c_void_p,  # stream
    ]
    lib.sirclEpCombine.restype = ctypes.c_int
    if hasattr(lib, "ncclGetLastError"):
        lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
        lib.ncclGetLastError.restype = ctypes.c_char_p
    lib._sircl_ep_bound = True
    return True


def install_sircl_ep_all2all() -> bool:
    """Install ``SirclEpAll2AllManager`` on vLLM's ``CudaCommunicator``."""
    global _INSTALLED
    if _INSTALLED:
        return True

    if ALL2ALL_BASE_IS_STUB:
        logger.warning("SIRCL_EP_ALL2ALL requested but vllm all2all is unavailable")
        return False

    try:
        from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )
        from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
            MoEPrepareAndFinalizeNaiveDPEPModular,
        )
    except ImportError as exc:
        logger.warning("SIRCL_EP_ALL2ALL requested but vllm all2all is unavailable: %s", exc)
        return False

    orig_prepare = MoEPrepareAndFinalizeNaiveDPEPModular.prepare

    def _wrapped_prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        prev = getattr(_ACTIVE_META, "current", None)
        _ACTIVE_META.current = (int(num_experts), expert_map)
        try:
            return orig_prepare(
                self,
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant=defer_input_quant,
            )
        finally:
            _ACTIVE_META.current = prev

    MoEPrepareAndFinalizeNaiveDPEPModular.prepare = _wrapped_prepare

    orig_cc_init = CudaCommunicator.__init__

    def _wrapped_cc_init(self, *args, **kwargs):
        orig_cc_init(self, *args, **kwargs)
        if self.use_all2all and isinstance(self.all2all_manager, AgRsAll2AllManager):
            tcp_store_group = kwargs.get("tcp_store_group")
            if tcp_store_group is None and len(args) >= 7:
                tcp_store_group = args[6]
            self.all2all_manager = SirclEpAll2AllManager(
                self.cpu_group,
                tcp_store_group,
                device_comm=self,
            )
            logger.info(
                "Replaced AgRsAll2AllManager with SirclEpAll2AllManager (max_tokens=%d, prealloc=%d, "
                "static=%s)",
                self.all2all_manager._max_tokens_cap,
                self.all2all_manager._prealloc_tokens,
                self.all2all_manager._static_max_tokens or "off",
            )

    CudaCommunicator.__init__ = _wrapped_cc_init
    _INSTALLED = True
    logger.info("Installed SIRCL-EP all2all manager hook (SIRCL_EP_ALL2ALL=1)")
    return True
