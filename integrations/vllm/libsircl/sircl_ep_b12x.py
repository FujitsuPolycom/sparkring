"""Opt-in b12x Expert-Parallel (EP) W4A16 MoE integration for vLLM.

Activated when ``SIRCL_B12X_EP_MOE=1`` is set and the ``libsircl`` vLLM plugin
runs. Enables ``B12xExperts`` to run with ``--enable-expert-parallel`` in both:

- Stage 1 (TP + EP, replicated activations + ``expert_map`` + 1 AllReduce)
- Stage 3 (DP + EP with ``allgather_reducescatter`` or ``SirclEpAll2AllManager``)

using ``b12x.moe.ep_moe`` and ``b12x.moe._shared.kernels.w4a16.kernel.run_w4a16_moe``
for MXFP4 W4A16 BF16 weights.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch

try:
    from vllm.logger import init_logger as _init_logger

    logger = _init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

ENV_FLAG = "SIRCL_B12X_EP_MOE"
_INSTALLED = False


def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_FLAG, "").strip().lower() in ("1", "true", "on", "yes")


def _is_capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


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
    """Patch vLLM's ``B12xExperts`` to support EP via ``b12x.moe.ep_moe``."""
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
    normalize_topk_weights = vllm_b12x._normalize_topk_weights

    def _use_ep(self: Any) -> bool:
        return bool(self.moe_config.moe_parallel_config.use_ep)

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
            if weight_key != kMxfp4Static or activation_key is not None:
                return (
                    False,
                    "b12x EP MoE supports only MXFP4 W4A16 BF16 (kMxfp4Static, None)",
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
        _patch_deepseek_v4_ced_dp_metadata()
        # Pre-populate layout cache and eagerly compile b12x W4A16 EP kernels
        # before vLLM freezes b12x kernel resolution for CUDA graph capture.
        ep_cfg = self.moe_config.moe_parallel_config
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
        if not _use_ep(self):
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
        if bool(apply_router_weight_on_input) != self._apply_router_weight_on_input:
            raise ValueError(
                "apply_router_weight_on_input does not match the prepared b12x EP MoE plan"
            )
        m = int(hidden_states.shape[0])
        if m == 0:
            return
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
    _patch_deepseek_v4_ced_dp_metadata()

    _INSTALLED = True
    logger.info("Installed SIRCL b12x EP W4A16 MoE extension on B12xExperts")
    return True


def _patch_modular_kernel_dp_workspace() -> None:
    """Drop pre-allocated M_local workspace when DP all2all dispatch expands M_full."""
    try:
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEKernelModularImpl,
        )
    except ImportError:
        return

    if getattr(FusedMoEKernelModularImpl._fused_experts, "_sircl_dp_ws_patched", False):
        return

    orig_fe = FusedMoEKernelModularImpl._fused_experts

    def _wrapped_fused_experts(self, *args, **kwargs):
        workspace = kwargs.get("workspace")
        ws_in_args = False
        if workspace is None and len(args) >= 13:
            workspace = args[12]
            ws_in_args = True
        if workspace is not None:
            a1q = kwargs["a1q"] if "a1q" in kwargs else args[1]
            w1 = kwargs["w1"] if "w1" in kwargs else args[3]
            w2 = kwargs["w2"] if "w2" in kwargs else args[4]
            topk_ids = kwargs["topk_ids"] if "topk_ids" in kwargs else args[6]
            _, m_full, _, _, _ = self.fused_experts.moe_problem_size(
                a1q, w1, w2, topk_ids
            )
            if workspace[2].shape[0] != m_full:
                if ws_in_args:
                    args = (*args[:12], None, *args[13:])
                else:
                    kwargs["workspace"] = None
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

