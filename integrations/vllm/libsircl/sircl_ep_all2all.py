"""Opt-in SIRCL-EP GPU-driven All-to-All manager for vLLM DP + EP.

Activated when ``SIRCL_EP_ALL2ALL=1`` is set and the ``libsircl`` vLLM plugin
runs. Subclasses ``AgRsAll2AllManager`` so vLLM's existing
``MoEPrepareAndFinalizeNaiveDPEPModular`` pipeline runs ``sirclEpDispatch`` and
``sirclEpCombine`` directly over ``libsircl``'s existing P2P RDMA channels (zero
extra queue pairs, zero CPU synchronization, CUDA-graph capturable), with an
automatic fallback to ``AgRsAll2AllManager`` for unsupported calls (e.g., extra
tensors, non-linear expert maps, or capacities above ``SIRCL_EP_MAX_TOKENS``).
"""
from __future__ import annotations

import ctypes
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch

try:
    from vllm.logger import init_logger as _init_logger

    logger = _init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

ENV_FLAG = "SIRCL_EP_ALL2ALL"
ENV_MAX_TOKENS = "SIRCL_EP_MAX_TOKENS"
ENV_PREALLOC_TOKENS = "SIRCL_EP_PREALLOC_TOKENS"

_INSTALLED = False
_ACTIVE_META = threading.local()

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


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


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
    lib._sircl_ep_bound = True
    return True


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


def install_sircl_ep_all2all() -> bool:
    """Install ``SirclEpAll2AllManager`` on vLLM's ``CudaCommunicator``."""
    global _INSTALLED
    if _INSTALLED:
        return True

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

    class SirclEpAll2AllManager(AgRsAll2AllManager):
        """Drop-in ``AgRsAll2AllManager`` replacement backed by ``sirclEpDispatch``/``sirclEpCombine``."""

        def __init__(self, cpu_group, tcp_store_group=None, device_comm=None):
            super().__init__(cpu_group, tcp_store_group)
            self._device_comm = device_comm
            self._workspace: torch.Tensor | None = None
            self._workspace_bytes: int = 0
            self._linear_map_cache: dict[tuple[int, int, int, int], bool] = {}
            self._active_call: _ActiveEpCall | None = None
            self._max_tokens_cap = _parse_int_env(ENV_MAX_TOKENS, 4096)
            self._prealloc_tokens = _parse_int_env(ENV_PREALLOC_TOKENS, 512)
            self._disabled_reason: str | None = None
            self.sircl_dispatches: int = 0
            self.sircl_combines: int = 0
            self.agrs_fallbacks: int = 0

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
            if torch.cuda.is_current_stream_capturing():
                return None
            alloc_tokens = max(max_tokens, min(self._prealloc_tokens, self._max_tokens_cap))
            alloc_bytes = int(lib.sirclEpWorkspaceBytes(world, alloc_tokens, hidden_bytes))
            self._workspace = torch.empty((alloc_bytes,), dtype=torch.uint8, device=device)
            self._workspace_bytes = alloc_bytes
            return self._workspace

        def _check_linear_map(self, expert_map: torch.Tensor | None, world: int, rank: int) -> bool:
            if expert_map is None:
                return False
            key = (expert_map.data_ptr(), int(expert_map.numel()), world, rank)
            cached = self._linear_map_cache.get(key)
            if cached is not None:
                return cached
            if torch.cuda.is_current_stream_capturing():
                return False
            ok = _is_linear_expert_map(expert_map, world, rank)
            self._linear_map_cache[key] = ok
            return ok

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
                and hidden_states.is_cuda
                and hidden_states.ndim == 2
                and hidden_states.dtype in _TORCH_TO_NCCL_DTYPE
                and topk_ids.ndim == 2
                and topk_weights.ndim == 2
            )
            if can_use:
                num_experts, expert_map = meta
                num_tokens = int(hidden_states.shape[0])
                hidden_dim = int(hidden_states.shape[1])
                hidden_bytes = hidden_dim * hidden_states.element_size()
                top_k = int(topk_ids.shape[1])
                sizes = self._get_sizes(num_tokens, dist_group)
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
                    if self._workspace is None and not torch.cuda.is_current_stream_capturing():
                        self._ensure_workspace(
                            lib,
                            world,
                            min(self._prealloc_tokens, self._max_tokens_cap),
                            hidden_bytes,
                            hidden_states.device,
                        )
                    can_use = False

            if can_use:
                ws = self._ensure_workspace(lib, world, max_tokens, hidden_bytes, hidden_states.device)
                if ws is None:
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
            device = h_in.device
            h_out = torch.empty((total_rows, hidden_dim), dtype=h_in.dtype, device=device)
            ids_out = torch.empty((total_rows, top_k), dtype=torch.int32, device=device)
            wts_out = torch.empty((total_rows, top_k), dtype=torch.float32, device=device)
            stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)

            rc = int(
                lib.sirclEpDispatch(
                    ctypes.c_void_p(h_in.data_ptr() if num_tokens > 0 else 0),
                    ctypes.c_void_p(ids_in.data_ptr() if num_tokens > 0 else 0),
                    ctypes.c_void_p(wts_in.data_ptr() if num_tokens > 0 else 0),
                    ctypes.c_void_p(h_out.data_ptr()),
                    ctypes.c_void_p(ids_out.data_ptr()),
                    ctypes.c_void_p(wts_out.data_ptr()),
                    ctypes.c_void_p(0),
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
                if self.sircl_dispatches == 0:
                    self._disabled_reason = f"sirclEpDispatch returned rc={rc}"
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
                raise RuntimeError(f"sirclEpDispatch failed with rc={rc}")

            self.sircl_dispatches += 1
            self._active_call = _ActiveEpCall(
                used_sircl=True,
                num_tokens=num_tokens,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                dtype=h_in.dtype,
                device=device,
                lib=lib,
                comm=comm,
            )
            return h_out, wts_out, ids_out

        def combine(
            self,
            hidden_states: torch.Tensor,
            is_sequence_parallel: bool = False,
        ) -> torch.Tensor:
            call = self._active_call
            self._active_call = None
            if call is None or not call.used_sircl:
                return super().combine(hidden_states, is_sequence_parallel=is_sequence_parallel)

            h_in = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
            h_out = torch.empty((call.num_tokens, call.hidden_dim), dtype=call.dtype, device=call.device)
            nccl_dt = _TORCH_TO_NCCL_DTYPE[call.dtype]
            stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(call.device).cuda_stream)

            rc = int(
                call.lib.sirclEpCombine(
                    ctypes.c_void_p(h_in.data_ptr()),
                    ctypes.c_void_p(h_out.data_ptr() if call.num_tokens > 0 else 0),
                    ctypes.c_void_p(self._workspace.data_ptr()),
                    ctypes.c_size_t(self._workspace_bytes),
                    ctypes.c_int(call.num_tokens),
                    ctypes.c_int(call.max_tokens),
                    ctypes.c_size_t(call.hidden_dim),
                    ctypes.c_int(nccl_dt),
                    call.comm,
                    stream_ptr,
                )
            )
            if rc != 0:
                raise RuntimeError(f"sirclEpCombine failed with rc={rc}")
            self.sircl_combines += 1
            return h_out

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
                "Replaced AgRsAll2AllManager with SirclEpAll2AllManager (max_tokens=%d, prealloc=%d)",
                self.all2all_manager._max_tokens_cap,
                self.all2all_manager._prealloc_tokens,
            )

    CudaCommunicator.__init__ = _wrapped_cc_init
    _INSTALLED = True
    logger.info("Installed SIRCL-EP all2all manager hook (SIRCL_EP_ALL2ALL=1)")
    return True
