"""Opt-in b12x Expert-Parallel (EP) W4A8/W4A16 MoE integration for vLLM.

Activated when ``SIRCL_B12X_EP_MOE=1`` is set and the ``libsircl`` vLLM plugin
runs. Enables ``B12xExperts`` to run with ``--enable-expert-parallel`` in both:

- Stage 1 (TP + EP, replicated activations + ``expert_map`` + 1 AllReduce or
  deferred reduce-scatter for ``ds41_row_shard``)
- Stage 3 (DP + EP with ``allgather_reducescatter`` or ``SirclEpAll2AllManager``)

Supports:

- **MXFP4 W4A8 (``w4a8_mx``)**: remaps global ``topk_ids`` through ``expert_map``
  on GPU (CUDA-graph safe, clamping out-of-range / ``-1`` padding slots to
  ``-1``) and invokes ``b12x.moe.fused_moe`` directly (whose kernels skip
  ``-1`` routes and write exact zeros for tokens with zero local experts).
- **MXFP4 W4A16 BF16 (``w4a16``)**: runs ``b12x.moe.ep_moe`` /
  ``run_w4a16_moe`` with native ``expert_map``.
"""
from __future__ import annotations

import dataclasses
import inspect
import logging
import math
import os
import sys
import threading
from typing import Any

import torch

try:
    from vllm.logger import init_logger as _init_logger

    logger = _init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

ENV_FLAG = "SIRCL_B12X_EP_MOE"
TRITON_REMAP_ENV = "SIRCL_EP_TRITON_REMAP"
ROUTER_SKIP_PADDING_ENV = "SIRCL_EP_ROUTER_SKIP_PADDING"
FORCE_GROUPED_ROUTING_ENV = "SIRCL_EP_FORCE_GROUPED_ROUTING"
_INSTALLED = False
_NEG_ONE_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}
_TRITON_REMAP_WARMED: set[tuple[int, torch.dtype, torch.dtype]] = set()

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _remap_expert_ids_kernel(
        ids_ptr,
        map_ptr,
        out_ptr,
        n_elements,
        global_num_experts,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        eid = tl.load(ids_ptr + offs, mask=mask, other=-1)
        valid = (eid >= 0) & (eid < global_num_experts)
        safe_eid = tl.where(valid, eid, 0)
        mapped = tl.load(map_ptr + safe_eid, mask=mask & valid, other=-1)
        out = tl.where(valid, mapped, -1)
        tl.store(out_ptr + offs, out, mask=mask)

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    _remap_expert_ids_kernel = None
    _TRITON_AVAILABLE = False


def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_FLAG, "").strip().lower() in ("1", "true", "on", "yes")


def _triton_remap_enabled(environ=None) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    env = os.environ if environ is None else environ
    raw = env.get(TRITON_REMAP_ENV, "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _is_capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _neg_one_scalar(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dev_idx = -1 if device.index is None else int(device.index)
    key = (dev_idx, dtype)
    cached = _NEG_ONE_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    tensor = torch.tensor(-1, dtype=dtype, device=device)
    _NEG_ONE_CACHE[key] = tensor
    return tensor


def _warmup_triton_remap(device: torch.device, global_num_experts: int = 384) -> None:
    """Compile Triton remap kernels before CUDA graph capture for int32 & int64 IDs."""
    if not _triton_remap_enabled() or device.type != "cuda":
        return
    dev_idx = -1 if device.index is None else int(device.index)
    emap = torch.full((max(global_num_experts, 1),), -1, dtype=torch.int32, device=device)
    for id_dtype in (torch.int32, torch.int64):
        key = (dev_idx, id_dtype, torch.int32)
        if key in _TRITON_REMAP_WARMED:
            continue
        dummy_ids = torch.zeros((6, 6), dtype=id_dtype, device=device)
        _remap_expert_ids(dummy_ids, emap, max(global_num_experts, 1))
        _TRITON_REMAP_WARMED.add(key)


def _remap_expert_ids(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    global_num_experts: int,
) -> torch.Tensor:
    """Map global ``topk_ids`` to rank-local IDs (``-1`` for non-local/padding).

    Guards against negative or out-of-range IDs (e.g. ``-1`` padding rows from
    ``dsv4_topk`` or DP all2all dispatch) before indexing ``expert_map`` so
    negative indices never wrap around to the last expert on rank ``EP-1``.
    Uses a single fused Triton kernel on CUDA (eliminating 6-8 ATen ops/layer
    under ``compilation_config.mode=NONE``) with an identical PyTorch fallback.
    """
    if (
        _triton_remap_enabled()
        and topk_ids.is_cuda
        and expert_map.is_cuda
        and topk_ids.is_contiguous()
        and expert_map.is_contiguous()
    ):
        n_elements = int(topk_ids.numel())
        out = torch.empty(topk_ids.shape, dtype=expert_map.dtype, device=topk_ids.device)
        if n_elements > 0:
            block_size = 256
            grid = ((n_elements + block_size - 1) // block_size,)
            _remap_expert_ids_kernel[grid](
                topk_ids,
                expert_map,
                out,
                n_elements,
                int(global_num_experts),
                BLOCK_SIZE=block_size,
            )
        return out
    neg_one = _neg_one_scalar(expert_map.device, expert_map.dtype)
    valid = (topk_ids >= 0) & (topk_ids < global_num_experts)
    safe_ids = topk_ids.clamp(0, max(global_num_experts - 1, 0))
    return torch.where(valid, expert_map[safe_ids], neg_one).contiguous()


def _widen_workload_for_ep(workload: Any, ep_mult: int, cap: int = 32768) -> Any:
    """Scale a ``B12xWorkload`` by ``ep_mult`` for DP+EP all2all dispatch."""
    if ep_mult <= 1:
        return workload
    max_base = int(workload.max_tokens)
    max_target = min(max(max_base * ep_mult, max_base), cap)
    fixed_set = {int(c) for c in workload.fixed_token_counts if 0 < int(c) < max_target}
    for c in list(fixed_set):
        scaled = c * ep_mult
        if 0 < scaled < max_target:
            fixed_set.add(scaled)
    counts_set = {int(c) for c in workload.token_counts if 0 < int(c) <= max_target}
    counts_set.update(fixed_set)
    counts_set.add(max_target)
    for c in list(counts_set):
        scaled = c * ep_mult
        if 0 < scaled <= max_target:
            counts_set.add(scaled)
    return dataclasses.replace(
        workload,
        max_tokens=max_target,
        token_counts=tuple(sorted(counts_set)),
        fixed_token_counts=tuple(sorted(fixed_set)),
    )


def _is_ep_parallel_supported(moe_parallel_config: Any) -> bool:
    if getattr(moe_parallel_config, "enable_eplb", False):
        return False
    if not moe_parallel_config.use_ep:
        return (
            moe_parallel_config.ep_size == 1
            and not moe_parallel_config.use_all2all_kernels
        )
    if moe_parallel_config.ep_size <= 1:
        return False
    if not moe_parallel_config.use_all2all_kernels:
        return True
    return bool(moe_parallel_config.use_ag_rs_all2all_kernels)


def install_b12x_ep_moe() -> bool:
    """Patch vLLM's ``B12xExperts`` to support W4A8 and W4A16 EP."""
    global _INSTALLED
    if _INSTALLED:
        return True

    try:
        from vllm.model_executor.layers.fused_moe import b12x as vllm_b12x
        from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
    except ImportError as exc:
        logger.warning("SIRCL_B12X_EP_MOE requested but vllm b12x is unavailable: %s", exc)
        return False

    from b12x._lib.compile_plan import retain_compiled_programs
    from b12x.moe._shared.kernels.w4a16.host import (
        route_pack_token_capacity,
        route_pack_warmup_token_counts,
        select_route_block_size_m,
    )
    from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
    from b12x.moe.ep_moe._impl import (
        EPMoEScratchCaps,
        _map_scratch_views,
        _materialize_layout,
    )
    from b12x.moe.fused_moe._impl import _prepared_payload_for_runtime

    orig_is_supported_config = B12xExperts.is_supported_config
    orig_get_prep_units = B12xExperts.get_b12x_preparation_units
    orig_workspace_shapes = B12xExperts.workspace_shapes
    orig_apply = B12xExperts.apply
    kMxfp4Static = vllm_b12x.kMxfp4Static
    kMxfp8Dynamic = vllm_b12x.kMxfp8Dynamic
    normalize_topk_weights = vllm_b12x._normalize_topk_weights

    def _use_ep(self: Any) -> bool:
        return bool(self.moe_config.moe_parallel_config.use_ep)

    def _is_w4a8_ep(self: Any) -> bool:
        return _use_ep(self) and getattr(self, "_quant_mode", None) != "w4a16"

    def _ep_layout_for(self: Any, tokens: int, activation: Any, route_ids_workspace: bool):
        cache = getattr(self, "_sircl_ep_layout_cache", None)
        if cache is None:
            cache = {}
            self._sircl_ep_layout_cache = cache
        m = max(int(tokens), 1)
        key = (m, activation, bool(route_ids_workspace))
        entry = cache.get(key)
        if entry is not None:
            return entry
        prepared_impl = self._prepared()._impl
        limit, alpha, beta = self._swiglu_params(activation)
        caps = EPMoEScratchCaps(
            max_tokens=m,
            num_topk=int(self.moe_config.experts_per_token),
            global_num_experts=int(self.moe_config.num_experts),
            device=torch.device("cuda", torch.cuda.current_device()),
            weight_plan=prepared_impl.plan,
            apply_router_weight_on_input=bool(self._apply_router_weight_on_input),
            swiglu_limit=limit,
            swiglu_alpha=alpha,
            swiglu_beta=beta,
        )
        layout = _materialize_layout(caps, route_ids_workspace=bool(route_ids_workspace))
        entry = (caps, layout)
        cache[key] = entry
        return entry

    def _ep_runtime_payload(self: Any, dtype: torch.dtype, hidden_size: int):
        cache = getattr(self, "_sircl_ep_rt_cache", None)
        if cache is None:
            cache = {}
            self._sircl_ep_rt_cache = cache
        prepared_impl = self._prepared()._impl
        key = (id(prepared_impl), dtype, hidden_size)
        rt = cache.get(key)
        if rt is not None:
            return rt
        rt = _prepared_payload_for_runtime(
            prepared_impl,
            quant_mode="w4a16",
            source_format=prepared_impl.source_format,
            activation=prepared_impl.activation,
            w13_layout=prepared_impl.w13_layout,
            dtype=dtype,
            hidden_size=hidden_size,
        )
        if rt is None:
            raise RuntimeError("b12x EP W4A16 weight representation was not materialized")
        cache[key] = rt
        return rt

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: Any) -> bool:
        return _is_ep_parallel_supported(moe_parallel_config)

    @staticmethod
    def is_supported_config(
        cls: Any,
        moe_config: Any,
        weight_key: Any,
        activation_key: Any,
        activation_format: Any,
    ) -> tuple[bool, str | None]:
        if moe_config.moe_parallel_config.use_ep:
            if (weight_key, activation_key) not in (
                (kMxfp4Static, kMxfp8Dynamic),
                (kMxfp4Static, None),
            ):
                return (
                    False,
                    "b12x EP MoE supports only MXFP4 W4A8/W4A16 BF16",
                )
            if moe_config.in_dtype != torch.bfloat16:
                return False, "b12x EP MoE requires bfloat16 activations"
        return orig_is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    def supports_expert_map(self: Any) -> bool:
        return _use_ep(self)

    _warmed_keys: set[tuple[Any, ...]] = set()
    _retained_program_handles: list[Any] = []

    def _warmup_ep_kernels(self: Any, activation: Any, counts: set[int]) -> None:
        prepared_impl = self._prepared()._impl
        dtype = self.moe_config.in_dtype
        hidden_size = int(prepared_impl.plan.hidden_size)
        num_experts = int(self.moe_config.num_experts)
        local_experts = int(prepared_impl.plan.num_experts)
        topk = int(self.moe_config.experts_per_token)
        device = torch.device("cuda", torch.cuda.current_device())
        rt = _ep_runtime_payload(self, dtype, hidden_size)
        use_swigluoai = prepared_impl.activation == "swigluoai_uninterleave"

        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        expert_map[:local_experts] = torch.arange(local_experts, dtype=torch.int32, device=device)

        max_m = max(counts, default=0)
        if max_m <= 0:
            return
        max_nbytes = 0
        layouts_by_m: dict[int, tuple[Any, Any]] = {}
        for m in sorted(counts):
            if m <= 0:
                continue
            caps, layout = _ep_layout_for(self, m, activation, route_ids_workspace=(m <= 8))
            layouts_by_m[m] = (caps, layout)
            nbytes = max((spec.offset_bytes + spec.nbytes for spec in layout._layout), default=0)
            max_nbytes = max(max_nbytes, nbytes)

        # Check whether any (m, ids_dtype) pair still needs warming up.
        todo: list[tuple[int, torch.dtype]] = []
        for m in sorted(layouts_by_m):
            caps, _ = layouts_by_m[m]
            dtypes = (torch.int32,) if m <= 8 else (torch.int32, torch.int64)
            for ids_dt in dtypes:
                wkey = (
                    prepared_impl.plan,
                    prepared_impl.activation,
                    dtype,
                    hidden_size,
                    num_experts,
                    topk,
                    bool(self._apply_router_weight_on_input),
                    caps.swiglu_limit,
                    m,
                    ids_dt,
                )
                if wkey not in _warmed_keys:
                    todo.append((m, ids_dt))
                    _warmed_keys.add(wkey)
        if not todo:
            return

        hidden_buf = torch.zeros((max_m, hidden_size), dtype=dtype, device=device)
        output_buf = torch.empty((max_m, hidden_size), dtype=dtype, device=device)
        weights_buf = torch.full((max_m, topk), 1.0 / max(topk, 1), dtype=torch.float32, device=device)
        ids_i32_buf = torch.arange(max_m * topk, dtype=torch.int32, device=device).remainder_(num_experts).view(max_m, topk)
        ids_i64_buf = ids_i32_buf.to(torch.int64)
        scratch_buf = torch.empty((max(max_nbytes, 1),), dtype=torch.uint8, device=device)

        with retain_compiled_programs() as retained:
            _retained_program_handles.append(retained)
            for m, ids_dt in todo:
                caps, layout = layouts_by_m[m]
                views = _map_scratch_views(scratch_buf, layout._layout)
                ids_slice = ids_i32_buf[:m] if ids_dt == torch.int32 else ids_i64_buf[:m]
                run_w4a16_moe(
                    hidden_buf[:m],
                    rt,
                    weights_buf[:m],
                    ids_slice,
                    activation=prepared_impl.activation,
                    intermediate_cache13=views["intermediate_cache13"],
                    intermediate_cache2=views["intermediate_cache2"],
                    output=output_buf[:m],
                    fc1_c_tmp=views["fc1_c_tmp"],
                    fc2_c_tmp=views["fc2_c_tmp"],
                    packed_route_indices=views["packed_route_indices"],
                    block_expert_ids=views["block_expert_ids"],
                    packed_route_count=views["packed_route_count"],
                    expert_offsets=views["expert_offsets"],
                    expert_counts=views["expert_counts"],
                    expert_map=expert_map,
                    apply_router_weight_on_input=bool(self._apply_router_weight_on_input),
                    fast_math=True,
                    swiglu_limit=caps.swiglu_limit,
                    swiglu_alpha=float(caps.swiglu_alpha) if use_swigluoai else None,
                    swiglu_beta=float(caps.swiglu_beta) if use_swigluoai else None,
                    fused_launch=None,
                )
        torch.cuda.synchronize(device)
        del hidden_buf, output_buf, weights_buf, ids_i32_buf, ids_i64_buf, scratch_buf, expert_map
        torch.cuda.empty_cache()

        logger.info(
            "SIRCL b12x EP warmup complete: local_E=%d global_E=%d topk=%d warmed=%d shapes (max_m=%d)",
            local_experts,
            num_experts,
            topk,
            len(todo),
            max_m,
        )

    def _ep_warmup_token_counts(max_m: int, topk: int, num_experts: int) -> set[int]:
        """Cover every (bucket, block_size_m, m_div16, numel_div16, small_m) Triton key."""
        seen: set[tuple[int, int, bool, bool, bool]] = set()
        picked: set[int] = set(route_pack_warmup_token_counts(max_m))
        for m in range(1, max_m + 1):
            key = (
                route_pack_token_capacity(m, topk),
                select_route_block_size_m(m, topk, num_experts),
                (m % 16) == 0,
                ((m * topk) % 16) == 0,
                m <= 8,
            )
            if key not in seen:
                seen.add(key)
                picked.add(m)
        return picked

    def get_b12x_preparation_units(self: Any, layer: Any, workload: Any):
        if not _use_ep(self):
            return orig_get_prep_units(self, layer, workload)
        ep_cfg = self.moe_config.moe_parallel_config
        if int(getattr(ep_cfg, "dp_size", 1)) > 1:
            _patch_deepseek_v4_ced_dp_metadata()
            _patch_dp_sync_cudagraph_num_reqs()
            _patch_dp_speculator_propose_sync()
        if _is_w4a8_ep(self):
            # Pre-populate cached -1 scalar and compile Triton remap before CUDA graph capture.
            if torch.cuda.is_available():
                dev = torch.device("cuda", torch.cuda.current_device())
                _neg_one_scalar(dev, torch.int32)
                _warmup_triton_remap(dev, int(self.moe_config.num_experts))
            ep_mult = int(ep_cfg.ep_size) if ep_cfg.use_all2all_kernels else 1
            prep_workload = (
                _widen_workload_for_ep(workload, ep_mult)
                if getattr(workload, "stage", None) == "weights" and ep_mult > 1
                else workload
            )
            return orig_get_prep_units(self, layer, prep_workload)
        # Pre-populate layout cache and eagerly compile b12x W4A16 EP kernels
        # before vLLM freezes b12x kernel resolution for CUDA graph capture.
        ep_mult = int(ep_cfg.ep_size) if ep_cfg.use_all2all_kernels else 1
        num_experts = int(self.moe_config.num_experts)
        topk = int(self.moe_config.experts_per_token)
        counts = {int(c) for c in workload.fixed_token_counts if int(c) > 0}
        if int(workload.max_tokens) > 0:
            counts.add(int(workload.max_tokens))
        max_base = max(counts, default=96)
        max_target = max_base * ep_mult if ep_mult > 1 else max_base
        # Cap DP+EP warmup target at 32768 rows (e.g. 4 ranks x 8192 batched tokens).
        max_target = min(max_target, 32768)
        counts.update(range(1, min(max_target, 16) + 1))
        counts.update(_ep_warmup_token_counts(max_target, topk, num_experts))
        if ep_mult > 1:
            for c in list(counts):
                if c <= max_base and c * ep_mult <= max_target:
                    counts.add(int(c) * ep_mult)
        _warmup_ep_kernels(self, layer.activation, counts)
        return ()

    def workspace_shapes(
        self: Any,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Any,
        activation: Any,
    ):
        if not _use_ep(self) or _is_w4a8_ep(self):
            return orig_workspace_shapes(
                self, M, N, K, topk, global_num_experts, local_num_experts, expert_tokens_meta, activation
            )
        m = max(int(M), 1)
        _, layout = _ep_layout_for(self, m, activation, route_ids_workspace=(m <= 8))
        required_nbytes = max(
            (spec.offset_bytes + spec.nbytes for spec in layout._layout),
            default=0,
        )
        itemsize = self.moe_config.in_dtype.itemsize
        return (0,), (max(1, (required_nbytes + itemsize - 1) // itemsize),), (M, K)

    def apply(
        self: Any,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: Any,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: Any,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        if expert_map is None and not _use_ep(self):
            return orig_apply(
                self,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
        if expert_map is None:
            raise ValueError("b12x EP MoE requires an expert_map tensor")
        m = int(hidden_states.shape[0])
        if m == 0:
            return
        if _is_w4a8_ep(self):
            local_topk_ids = _remap_expert_ids(
                topk_ids, expert_map, int(global_num_experts)
            )
            return orig_apply(
                self,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                local_topk_ids,
                activation,
                global_num_experts,
                None,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
        if bool(apply_router_weight_on_input) != self._apply_router_weight_on_input:
            raise ValueError(
                "apply_router_weight_on_input does not match the prepared b12x EP MoE plan"
            )
        if (
            topk_ids.dtype not in (torch.int32, torch.int64)
            or not topk_ids.is_contiguous()
        ):
            raise TypeError("b12x EP MoE topk_ids must be contiguous int32 or int64")
        if workspace2 is None or not workspace2.is_contiguous():
            raise ValueError("b12x EP MoE requires contiguous caller-owned workspace2")
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        topk_weights = normalize_topk_weights(topk_weights)
        route_ws = m <= 8
        caps, layout = _ep_layout_for(self, m, activation, route_ids_workspace=route_ws)
        scratch = workspace2.view(-1).view(torch.uint8)
        views = _map_scratch_views(scratch, layout._layout)

        launch_topk_ids = topk_ids
        if route_ws and topk_ids.dtype == torch.int64:
            route_count = topk_ids.numel()
            i32_view = views["route_ids_i32"][:route_count].view_as(topk_ids)
            i32_view.copy_(topk_ids)
            launch_topk_ids = i32_view

        prepared_impl = self._prepared()._impl
        rt = _ep_runtime_payload(self, hidden_states.dtype, int(hidden_states.shape[1]))
        use_swigluoai = prepared_impl.activation == "swigluoai_uninterleave"

        run_w4a16_moe(
            hidden_states,
            rt,
            topk_weights,
            launch_topk_ids,
            activation=prepared_impl.activation,
            intermediate_cache13=views["intermediate_cache13"],
            intermediate_cache2=views["intermediate_cache2"],
            output=output,
            fc1_c_tmp=views["fc1_c_tmp"],
            fc2_c_tmp=views["fc2_c_tmp"],
            packed_route_indices=views["packed_route_indices"],
            block_expert_ids=views["block_expert_ids"],
            packed_route_count=views["packed_route_count"],
            expert_offsets=views["expert_offsets"],
            expert_counts=views["expert_counts"],
            expert_map=expert_map,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            fast_math=True,
            swiglu_limit=caps.swiglu_limit,
            swiglu_alpha=float(caps.swiglu_alpha) if use_swigluoai else None,
            swiglu_beta=float(caps.swiglu_beta) if use_swigluoai else None,
            fused_launch=None,
        )

    B12xExperts._supports_parallel_config = _supports_parallel_config
    B12xExperts.is_supported_config = is_supported_config
    B12xExperts.supports_expert_map = supports_expert_map
    B12xExperts.get_b12x_preparation_units = get_b12x_preparation_units
    B12xExperts.workspace_shapes = workspace_shapes
    B12xExperts.apply = apply

    _patch_b12x_dp_global_rank()
    _patch_modular_kernel_dp_workspace()
    _patch_dp_sync_cudagraph_num_reqs()
    _patch_dp_speculator_propose_sync()
    _patch_router_skip_padding()
    _patch_ep_grouped_routing()

    _INSTALLED = True
    logger.info("Installed SIRCL b12x EP W4A8/W4A16 MoE extension on B12xExperts")
    return True


def _resize_dp_workspace(
    fused_experts: Any,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    in_dtype: torch.dtype,
    device: torch.device,
    m_full: int,
    n_dim: int,
    k_dim: int,
    top_k: int,
    global_num_experts: int,
    local_num_experts: int,
    expert_tokens_meta: Any,
    activation: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand an ``M_local`` workspace to ``m_full`` without re-leasing the shared arena.

    ``prepare_workspace`` leases routed workspace and ``shared_workspace`` from
    the same ``WorkspaceManager`` arena so ``shared_experts`` can overlap on an
    aux stream. Dropping ``workspace=None`` would force ``_allocate_buffers`` to
    lease from that same arena a second time, colliding with or freeing
    ``shared_workspace`` while the aux stream is executing. Instead, reuse any
    already-large-enough routed slices (e.g. W4A8's pre-sized ``workspace2``)
    and allocate only undersized tensors directly via ``torch.empty``.
    """
    ws13, ws2, fused_out = workspace
    ws_dtype = fused_experts.workspace_dtype(in_dtype)
    out_dtype = fused_experts.output_dtype
    ws13_shape, ws2_shape, out_shape = fused_experts.workspace_shapes(
        m_full,
        n_dim,
        k_dim,
        top_k,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    )

    if ws13.shape != ws13_shape or ws13.dtype != ws_dtype:
        need13 = math.prod(ws13_shape)
        if ws13.dtype == ws_dtype and ws13.is_contiguous() and ws13.numel() >= need13:
            ws13 = ws13.view(-1)[:need13].view(ws13_shape)
        else:
            ws13 = torch.empty(ws13_shape, dtype=ws_dtype, device=device)

    if ws2.shape != ws2_shape or ws2.dtype != ws_dtype:
        need2 = math.prod(ws2_shape)
        if ws2.dtype == ws_dtype and ws2.is_contiguous() and ws2.numel() >= need2:
            ws2 = ws2.view(-1)[:need2].view(ws2_shape)
        else:
            ws2 = torch.empty(ws2_shape, dtype=ws_dtype, device=device)

    if fused_out.shape != out_shape or fused_out.dtype != out_dtype:
        fused_out = torch.empty(out_shape, dtype=out_dtype, device=device)

    return ws13, ws2, fused_out


def _patch_modular_kernel_dp_workspace() -> None:
    """Resize pre-allocated M_local workspace when DP all2all expands M_full."""
    try:
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEKernelModularImpl,
        )
    except ImportError:
        return

    if getattr(FusedMoEKernelModularImpl._fused_experts, "_sircl_dp_ws_patched", False):
        return

    orig_fe = FusedMoEKernelModularImpl._fused_experts
    sig = inspect.signature(orig_fe)
    param_names = list(sig.parameters)[1:]  # exclude self
    ws_pos = param_names.index("workspace") if "workspace" in param_names else 14

    def _get_arg(name: str, pos: int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name in kwargs:
            return kwargs[name]
        return args[pos]

    def _wrapped_fused_experts(self, *args, **kwargs):
        workspace = kwargs.get("workspace")
        ws_in_args = False
        if workspace is None and len(args) > ws_pos:
            workspace = args[ws_pos]
            ws_in_args = True
        if workspace is not None:
            a1q = _get_arg("a1q", 1, args, kwargs)
            w1 = _get_arg("w1", 3, args, kwargs)
            w2 = _get_arg("w2", 4, args, kwargs)
            topk_ids = _get_arg("topk_ids", 6, args, kwargs)
            _, m_full, n_dim, k_dim, top_k = self.fused_experts.moe_problem_size(
                a1q, w1, w2, topk_ids
            )
            if m_full > 0 and workspace[2].shape[0] != m_full:
                in_dtype = _get_arg("in_dtype", 0, args, kwargs)
                activation = _get_arg("activation", 7, args, kwargs)
                global_num_experts = _get_arg("global_num_experts", 8, args, kwargs)
                local_num_experts = _get_arg("local_num_experts", 9, args, kwargs)
                expert_tokens_meta = _get_arg("expert_tokens_meta", 12, args, kwargs)
                resized = _resize_dp_workspace(
                    self.fused_experts,
                    workspace,
                    in_dtype=in_dtype,
                    device=a1q.device,
                    m_full=m_full,
                    n_dim=n_dim,
                    k_dim=k_dim,
                    top_k=top_k,
                    global_num_experts=global_num_experts,
                    local_num_experts=local_num_experts,
                    expert_tokens_meta=expert_tokens_meta,
                    activation=activation,
                )
                if ws_in_args:
                    args = (*args[:ws_pos], resized, *args[ws_pos + 1 :])
                else:
                    kwargs["workspace"] = resized
        return orig_fe(self, *args, **kwargs)

    _wrapped_fused_experts._sircl_dp_ws_patched = True
    FusedMoEKernelModularImpl._fused_experts = _wrapped_fused_experts


def _resolve_world_global_rank(world_group: Any) -> int | None:
    if world_group is None:
        return None
    ranks = getattr(world_group, "ranks", None)
    rank_in_group = getattr(world_group, "rank_in_group", None)
    if (
        ranks is not None
        and isinstance(rank_in_group, int)
        and 0 <= rank_in_group < len(ranks)
    ):
        return int(ranks[rank_in_group])
    rank = getattr(world_group, "rank", None)
    return int(rank) if isinstance(rank, int) else None


def _patch_b12x_dp_global_rank() -> None:
    """Fix upstream b12x_prepare passing local TP worker.rank as global_rank under DP>1."""
    try:
        from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator
    except ImportError:
        B12xPreparationCoordinator = None

    if B12xPreparationCoordinator is not None and not getattr(
        B12xPreparationCoordinator.__init__, "_sircl_dp_rank_patched", False
    ):
        orig_coord_init = B12xPreparationCoordinator.__init__

        def _wrapped_coord_init(
            self,
            session,
            batches,
            *,
            global_rank: int,
            world_group,
            process_local_only: bool = False,
            **kwargs,
        ):
            if not process_local_only and world_group is not None:
                actual = _resolve_world_global_rank(world_group)
                if actual is not None and actual != int(global_rank):
                    logger.info(
                        "Fixing B12xPreparationCoordinator global_rank %d -> %d for DP world",
                        int(global_rank),
                        actual,
                    )
                    global_rank = actual
            return orig_coord_init(
                self,
                session,
                batches,
                global_rank=global_rank,
                world_group=world_group,
                process_local_only=process_local_only,
                **kwargs,
            )

        _wrapped_coord_init._sircl_dp_rank_patched = True
        B12xPreparationCoordinator.__init__ = _wrapped_coord_init

    try:
        from b12x.preparation import PreparationSession
    except ImportError:
        PreparationSession = None

    if PreparationSession is not None and not getattr(
        PreparationSession.configure_tuning_shard, "_sircl_dp_rank_patched", False
    ):
        orig_configure_shard = PreparationSession.configure_tuning_shard

        def _wrapped_configure_shard(self, rank: int, ranks):
            if int(rank) not in ranks:
                try:
                    from vllm.distributed.parallel_state import get_world_group

                    actual = _resolve_world_global_rank(get_world_group())
                    if actual is not None and actual in ranks:
                        logger.info(
                            "Fixing b12x tuning shard rank %d -> %d for DP world",
                            int(rank),
                            actual,
                        )
                        rank = actual
                except Exception:
                    pass
            return orig_configure_shard(self, rank, ranks)

        _wrapped_configure_shard._sircl_dp_rank_patched = True
        PreparationSession.configure_tuning_shard = _wrapped_configure_shard


def _patch_deepseek_v4_ced_dp_metadata() -> None:
    """Sync DPMetadata token counts across DP ranks after DeepSeek-V4.1 CED compaction."""
    mod = sys.modules.get("vllm.models.deepseek_v4_1.nvidia.model")
    if mod is None:
        return
    model_cls = getattr(mod, "DeepseekV4Model", None)
    if model_cls is None or getattr(model_cls.forward, "_sircl_ced_dp_patched", False):
        return
    try:
        from vllm.distributed.parallel_state import get_dp_group
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except ImportError:
        return

    def _ced_ffn_pre_hook(_module: Any, hook_args: tuple[Any, ...]) -> None:
        if _is_capturing():
            return
        if not hook_args or not isinstance(hook_args[0], torch.Tensor):
            return
        dp_group = get_dp_group()
        if dp_group.world_size <= 1:
            return
        if not is_forward_context_available():
            return
        dp_meta = get_forward_context().dp_metadata
        if dp_meta is None:
            return
        dtype = dp_meta.num_tokens_across_dp_cpu.dtype
        local_n = torch.tensor([int(hook_args[0].shape[0])], dtype=dtype)
        gathered = [torch.empty((1,), dtype=dtype) for _ in range(dp_group.world_size)]
        torch.distributed.all_gather(gathered, local_n, group=dp_group.cpu_group)
        dp_meta.num_tokens_across_dp_cpu = torch.cat(gathered, dim=0)

    orig_forward = model_cls.forward

    def _wrapped_model_forward(self: Any, *args: Any, **kwargs: Any):
        ced_start = getattr(self, "ced_decoder_start", None)
        if ced_start is not None and not getattr(self, "_sircl_ced_hook_installed", False):
            sync_idx = max(int(ced_start), int(getattr(self, "start_layer", 0)))
            end_layer = int(getattr(self, "end_layer", len(self.layers)))
            if sync_idx < end_layer:
                ffn = getattr(self.layers[sync_idx], "ffn", None)
                if ffn is not None:
                    ffn.register_forward_pre_hook(_ced_ffn_pre_hook)
                    logger.info(
                        "Installed DeepSeek-V4.1 CED DP metadata sync hook on layer %d",
                        sync_idx,
                    )
            self._sircl_ced_hook_installed = True

        dp_meta = None
        orig_tokens = None
        if is_forward_context_available():
            dp_meta = get_forward_context().dp_metadata
            if dp_meta is not None:
                orig_tokens = dp_meta.num_tokens_across_dp_cpu
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            if dp_meta is not None and orig_tokens is not None:
                dp_meta.num_tokens_across_dp_cpu = orig_tokens

    _wrapped_model_forward._sircl_ced_dp_patched = True
    model_cls.forward = _wrapped_model_forward
    logger.info("Patched DeepseekV4Model.forward for CED DP metadata synchronization")


def _patch_dp_sync_cudagraph_num_reqs() -> None:
    """Ensure DP>1 FULL CUDA-graph dispatch agrees on ``num_reqs`` across ranks.

    Upstream ``dp_utils.sync_cudagraph_and_dp_padding`` passes each rank's local
    ``num_reqs`` to ``cudagraph_manager.dispatch(...)``. With ``varlen_decode``
    (e.g. DSpark adaptive verification), multiple FULL graphs can share the same
    ``synced_num_tokens`` with different ``num_reqs``; ranks with different local
    ``num_reqs`` then select different descriptors and store divergent
    ``DPSyncState.num_reqs`` values, causing downstream drafters (DSpark/DFlash)
    to replay different batch sizes across DP ranks.
    """
    try:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu import dp_utils
    except ImportError:
        return

    if getattr(dp_utils.sync_cudagraph_and_dp_padding, "_sircl_dp_num_reqs_patched", False):
        return

    tls = threading.local()
    orig_dist = dp_utils.dist

    class _DistProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(orig_dist, name)

        def all_reduce(self, tensor: Any, *args: Any, **kwargs: Any) -> Any:
            res = orig_dist.all_reduce(tensor, *args, **kwargs)
            if (
                getattr(tls, "active", False)
                and isinstance(tensor, torch.Tensor)
                and tensor.ndim == 2
                and tensor.shape[0] >= 6
            ):
                tls.max_num_reqs = int(tensor[5].max().item())
            return res

    class _SyncedNumReqsManagerProxy:
        def __init__(self, manager: Any, get_min_num_reqs: Any) -> None:
            self._manager = manager
            self._get_min_num_reqs = get_min_num_reqs

        def __getattr__(self, name: str) -> Any:
            return getattr(self._manager, name)

        def dispatch(self, num_reqs: int, *args: Any, **kwargs: Any) -> Any:
            synced = self._get_min_num_reqs()
            eff = max(int(num_reqs), int(synced)) if synced is not None else num_reqs
            desc = self._manager.dispatch(eff, *args, **kwargs)
            if (
                synced is not None
                and desc.cg_mode != CUDAGraphMode.FULL
                and desc.num_reqs is not None
                and desc.num_reqs != num_reqs
            ):
                desc = dataclasses.replace(desc, num_reqs=num_reqs)
            return desc

    orig_sync = dp_utils.sync_cudagraph_and_dp_padding
    orig_dispatch = dp_utils.dispatch_cg_and_sync_dp
    dispatch_params = list(inspect.signature(orig_dispatch).parameters)
    dp_sync_pos = (
        dispatch_params.index("dp_sync") - 1 if "dp_sync" in dispatch_params else 11
    )

    def _wrapped_sync(cudagraph_manager: Any, *args: Any, **kwargs: Any):
        prev_active = getattr(tls, "active", False)
        prev_max = getattr(tls, "max_num_reqs", None)
        tls.active = True
        tls.max_num_reqs = None
        cm = (
            _SyncedNumReqsManagerProxy(
                cudagraph_manager, lambda: getattr(tls, "max_num_reqs", None)
            )
            if cudagraph_manager is not None
            else None
        )
        try:
            return orig_sync(cm, *args, **kwargs)
        finally:
            tls.active = prev_active
            tls.max_num_reqs = prev_max

    def _wrapped_dispatch(cudagraph_manager: Any, *args: Any, **kwargs: Any):
        dp_sync = kwargs.get("dp_sync")
        if dp_sync is None and len(args) > dp_sync_pos:
            dp_sync = args[dp_sync_pos]
        if dp_sync is not None and cudagraph_manager is not None:
            min_reqs = int(dp_sync.num_reqs)
            cudagraph_manager = _SyncedNumReqsManagerProxy(
                cudagraph_manager, lambda: min_reqs
            )
        return orig_dispatch(cudagraph_manager, *args, **kwargs)

    _wrapped_sync._sircl_dp_num_reqs_patched = True
    _wrapped_dispatch._sircl_dp_num_reqs_patched = True
    dp_utils.dist = _DistProxy()
    dp_utils.sync_cudagraph_and_dp_padding = _wrapped_sync
    dp_utils.dispatch_cg_and_sync_dp = _wrapped_dispatch

    for mod_name in (
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "dispatch_cg_and_sync_dp", None) is orig_dispatch:
            mod.dispatch_cg_and_sync_dp = _wrapped_dispatch

    logger.info("Patched dp_utils CUDA-graph DP sync to agree on num_reqs across ranks")


def _patch_dp_speculator_propose_sync() -> None:
    """Keep DP>1 speculator ``propose()`` in lockstep when ``num_spec_tokens_to_schedule == 0``.

    Idle DP ranks in ``execute_dummy_batch()`` unconditionally call
    ``speculator.propose(..., dummy_run=True)`` to stay synchronized on MoE
    drafter collectives, whereas ``GPUModelRunner.sample_tokens()`` skips
    ``propose()`` when ``num_spec_tokens_to_schedule == 0`` (e.g. intermediate
    chunked-prefill steps or ``max_tokens=1`` requests). Under DP>1, force a
    ``dummy_run=True`` propose call and discard its output so every DP rank
    executes the same drafter collectives before entering the next step.
    """
    mod = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if mod is None:
        return
    runner_cls = getattr(mod, "GPUModelRunner", None)
    if runner_cls is None or getattr(
        runner_cls.sample_tokens, "_sircl_dp_spec_sync_patched", False
    ):
        return

    orig_sample_tokens = runner_cls.sample_tokens

    def _wrapped_sample_tokens(self: Any, *args: Any, **kwargs: Any):
        state = getattr(self, "execute_model_state", None)
        spec = getattr(self, "speculator", None)
        dp_size = int(getattr(self, "dp_size", 1))
        if (
            dp_size > 1
            and spec is not None
            and state is not None
            and int(getattr(state, "num_spec_tokens_to_schedule", 1)) == 0
            and not getattr(state, "boundary_logits_only", False)
            and hasattr(state, "_replace")
        ):
            self.execute_model_state = state._replace(num_spec_tokens_to_schedule=1)
            orig_propose = spec.propose
            orig_adaptive = getattr(self, "adaptive_verification", None)

            def _sync_only_propose(*p_args: Any, **p_kwargs: Any):
                p_kwargs["dummy_run"] = True
                out = orig_propose(*p_args, **p_kwargs)
                return out[:, :0]

            spec.propose = _sync_only_propose
            if orig_adaptive is not None:
                self.adaptive_verification = None
            try:
                return orig_sample_tokens(self, *args, **kwargs)
            finally:
                spec.propose = orig_propose
                if orig_adaptive is not None:
                    self.adaptive_verification = orig_adaptive
        return orig_sample_tokens(self, *args, **kwargs)

    _wrapped_sample_tokens._sircl_dp_spec_sync_patched = True
    runner_cls.sample_tokens = _wrapped_sample_tokens
    logger.info("Patched GPUModelRunner.sample_tokens for DP>1 speculator propose sync")


def _patch_router_skip_padding() -> None:
    """Forward ``is_padding`` in text-only ``dsv4_topk`` when ``VLLM_MOE_SKIP_PADDING=1``."""
    raw = os.environ.get(ROUTER_SKIP_PADDING_ENV, "1").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return
    try:
        from vllm.model_executor.layers.fused_moe.router import (
            fused_topk_bias_router as router_mod,
        )
    except ImportError:
        return
    orig_dsv4 = getattr(router_mod, "dsv4_topk", None)
    get_pad = getattr(router_mod, "_get_padding_mask", None)
    if orig_dsv4 is None or get_pad is None or getattr(orig_dsv4, "_sircl_pad_patched", False):
        return

    def _wrapped_dsv4_topk(gating_output: torch.Tensor, *args: Any, **kwargs: Any):
        if kwargs.get("is_padding") is None:
            pad = get_pad(gating_output.shape[0])
            if pad is not None:
                kwargs["is_padding"] = pad
        return orig_dsv4(gating_output, *args, **kwargs)

    _wrapped_dsv4_topk._sircl_pad_patched = True
    router_mod.dsv4_topk = _wrapped_dsv4_topk
    logger.info("Patched fused_topk_bias_router.dsv4_topk to forward is_padding mask")


def _patch_ep_grouped_routing() -> None:
    """Prefer grouped over direct W4A8 dynamic routing on EP small batches (e.g. M=5).

    On EP4, 75% of routed slots are ``-1`` and consecutive speculative tokens share
    hot experts; grouped routing compacts out ``-1`` slots and coalesces duplicate
    expert loads into one tile, beating direct routing by ~0.9-1.6 ms/step at M=5.

    Overrides ``_dynamic_direct_routing_selected`` and ``_heuristic_dynamic_route_mode``
    rather than zeroing ``_DIRECT_ROUTING_MAX_ROUTED_ROWS`` so cached ``b12x``
    preparation plans that recorded ``dynamic_route_mode="direct"`` still validate
    cleanly during ``_materialize_tuning`` and compile/execute as grouped.
    """
    raw = os.environ.get(FORCE_GROUPED_ROUTING_ENV, "1").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return
    try:
        from b12x.moe.fused_moe import _impl as fm_impl
    except ImportError:
        return
    orig_sel = getattr(fm_impl, "_dynamic_direct_routing_selected", None)
    if orig_sel is not None and not getattr(orig_sel, "_sircl_grouped_patched", False):
        def _always_grouped_selected(*_args: Any, **_kwargs: Any) -> bool:
            return False

        _always_grouped_selected._sircl_grouped_patched = True
        fm_impl._dynamic_direct_routing_selected = _always_grouped_selected

        orig_heur = getattr(fm_impl, "_heuristic_dynamic_route_mode", None)
        if orig_heur is not None and not getattr(orig_heur, "_sircl_grouped_patched", False):
            def _always_grouped_heuristic(*_args: Any, **_kwargs: Any) -> str:
                return "grouped"

            _always_grouped_heuristic._sircl_grouped_patched = True
            fm_impl._heuristic_dynamic_route_mode = _always_grouped_heuristic

        logger.info(
            "Patched b12x _dynamic_direct_routing_selected -> grouped for EP routing"
        )
