# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own GLM mHC token rows within an eligible GB10 eager model forward."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

_LOG = logging.getLogger(__name__)
_REPORTS = 0


@dataclass
class PrefillDiagnostics:
    """Observe host admission and enqueue counts without reading GPU contents."""

    rank: int
    decisions: dict[str, int] = field(default_factory=dict)
    emitted: set[str] = field(default_factory=set)
    admitted: int = 0

    def decision(self, reason: str, **details: Any) -> str:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        kind = "no_context"
        if is_forward_context_available():
            context = get_forward_context()
            kind = "dummy" if getattr(context, "is_dummy_run", False) else "request"
        key = f"{kind}:{reason}"
        self.decisions[key] = self.decisions.get(key, 0) + 1
        report_key = key
        if reason == "hidden_shape":
            report_key += ":" + json.dumps(details["shape"])
        if report_key not in self.emitted and _LOG.isEnabledFor(logging.WARNING):
            _LOG.warning(
                "GLM_MHC_DIAGNOSTIC %s",
                json.dumps(
                    dict(rank=self.rank, kind=kind, reason=reason, **details),
                    sort_keys=True,
                ),
            )
            self.emitted.add(report_key)
        return kind


def metadata_diagnostics(metadata: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Describe admission counts without converting or representing tensors."""
    if not isinstance(metadata, dict):
        return {"metadata_type": type(metadata).__name__}
    counts = {}
    fields = (
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "num_spec_decodes",
        "num_spec_decode_tokens",
    )
    for name in names:
        item = metadata.get(name)
        counts[name] = {
            field: {
                "type": type(value).__name__,
                "value": value if type(value) is int else None,
            }
            for field in fields
            for value in [getattr(item, field, None)]
        }
    return {
        "metadata_counts": counts,
        "missing_names": [n for n in names if n not in metadata],
    }


def device_is_supported(properties: Any) -> bool:
    """Return whether a device matches the bounded 48-SM GB10 geometry."""
    return (
        (properties.major, properties.minor) == (12, 1)
        and properties.multi_processor_count == 48
        and "GB10" in properties.name
    )


def configure(model: Any, config: Any, enabled: bool) -> None:
    """Agree on device/configuration admission before any graph is captured.

    Args:
        model: GLM base model whose eager prefills may own token rows.
        config: Serving configuration shared by all TP ranks.
        enabled: Explicit opt-in from VLLM_GLM53_MHC_PREFILL_SHARD.

    Raises:
        RuntimeError: TP ranks disagree or an enabled device/config is unsupported.
    """
    import torch

    from vllm import envs
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    error = None
    rows = config.scheduler_config.max_num_batched_tokens
    if enabled:
        try:
            captures = config.compilation_config.cudagraph_capture_sizes or []
            if (
                group.world_size not in (2, 4)
                or rows not in (4096, 8192)
                or any(size >= rows for size in captures)
            ):
                raise RuntimeError(
                    "GLM mHC row ownership requires TP2 or TP4, a 4096/8192-token "
                    "ceiling, and graph capture sizes below that ceiling"
                )
            parallel = config.parallel_config
            if (
                (parallel.tensor_parallel_size, parallel.decode_context_parallel_size)
                not in ((2, 1), (4, 1), (4, 2), (4, 4))
                or group.world_size != parallel.tensor_parallel_size
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.enable_expert_parallel
                or parallel.enable_eplb
                or parallel.use_sequence_parallel_moe
                or config.model_config.dtype != torch.bfloat16
                or model.config.hidden_size != 4096
            ):
                raise RuntimeError(
                    "GLM mHC row ownership requires BF16 H4096 and "
                    "TP2/DCP1 or TP4/DCP1,2,4 with PP1/DP1/PCP1 and no "
                    "expert or sequence parallelism"
                )
            properties = torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            )
            if not device_is_supported(properties):
                raise RuntimeError("GLM mHC row ownership requires a 48-SM GB10 GPU")
        except (AttributeError, RuntimeError, AssertionError) as exc:
            error = str(exc)
    votes: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(
        votes, (enabled, error, rows), group=group.cpu_group
    )
    if any(vote[0] != enabled for vote in votes):
        raise RuntimeError("All TP ranks must agree on VLLM_GLM53_MHC_PREFILL_SHARD")
    errors = [vote[1] for vote in votes if vote[1] is not None]
    if errors:
        raise RuntimeError("GLM mHC configuration admission failed: " + repr(errors))
    if enabled and any(vote[2] != rows for vote in votes):
        raise RuntimeError("All TP ranks must agree on the mHC prefill token ceiling")
    model._mhc_prefill_enabled = enabled
    model._mhc_prefill_rows = rows
    model._mhc_prefill_parallel_config = config.parallel_config
    model._mhc_prefill_diagnostics = (
        PrefillDiagnostics(group.rank_in_group)
        if envs.VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS
        else None
    )
    if model._mhc_prefill_diagnostics is not None:
        model._mhc_prefill_diagnostics.decision("configured", enabled=enabled)


def pure_prefill_metadata(metadata: Any, names: tuple[str, ...], rows: int) -> bool:
    """Use host counts only; missing or ambiguous metadata cannot opt in."""
    if not isinstance(metadata, dict) or not names or rows not in (4096, 8192):
        return False
    fields = (
        "num_decodes",
        "num_decode_tokens",
        "num_spec_decodes",
        "num_spec_decode_tokens",
    )
    for name in names:
        item = metadata.get(name)
        if item is None:
            return False
        if (
            type(getattr(item, "num_prefills", None)) is not int
            or item.num_prefills <= 0
        ):
            return False
        if (
            type(getattr(item, "num_prefill_tokens", None)) is not int
            or item.num_prefill_tokens != rows
        ):
            return False
        if any(
            type(getattr(item, field, None)) is not int or getattr(item, field) != 0
            for field in fields
        ):
            return False
    return True


def validate_moe_deferral(runner: Any) -> None:
    config = runner.moe_config
    parallel = config.moe_parallel_config
    if (
        config.tp_size not in (2, 4)
        or config.dp_size != 1
        or config.ep_size != 1
        or config.pcp_size != 1
        or config.is_sequence_parallel
        or config.skip_final_all_reduce
        or parallel.use_all2all_kernels
        or runner._fused_output_is_reduced
        or runner.routed_output_transform is not None
        or runner.routed_input_transform is not None
        or type(runner.router).__name__ == "ZeroExpertRouter"
    ):
        raise RuntimeError(
            "mHC prefill requires one unreduced conventional TP2/TP4 MoE output"
        )


def validate_model(model: Any) -> tuple[str, ...]:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

    from .attention import Glm5NextMLAAttention
    from .kda import Glm5NextLinearAttention

    config = model._mhc_prefill_parallel_config
    if (
        (config.tensor_parallel_size, config.decode_context_parallel_size)
        not in ((2, 1), (4, 1), (4, 2), (4, 4))
        or config.pipeline_parallel_size != 1
        or config.data_parallel_size != 1
        or config.prefill_context_parallel_size != 1
        or config.enable_expert_parallel
        or config.enable_eplb
        or model.is_sequence_parallel
    ):
        raise RuntimeError(
            "mHC prefill requires TP2/DCP1 or TP4/DCP1,2,4 with "
            "PP1/DP1/PCP1 and no EP/EPLB"
        )
    names = []
    for layer in model._active_layers:
        if not layer.mhc or layer.is_mtp_layer or layer._b12x_mhc is None:
            raise RuntimeError("mHC prefill requires B12X mHC on every base layer")
        attn = layer.self_attn
        if type(attn) is Glm5NextLinearAttention:
            names.append(attn.prefix)
        elif type(attn) is Glm5NextMLAAttention:
            if type(attn.mla_attn) is not MultiHeadLatentAttentionWrapper:
                raise RuntimeError(
                    "Unsupported outer MLA wrapper for mHC row ownership"
                )
        else:
            raise RuntimeError(
                "Unsupported GLM attention implementation for mHC row ownership"
            )
        projection = attn.o_proj
        if (
            type(projection) is not RowParallelLinear
            or projection.tp_size != config.tensor_parallel_size
            or not projection.reduce_results
            or projection.bias is not None
        ):
            raise RuntimeError(
                "Unsupported attention projection reduction for mHC row ownership"
            )
        if layer._mlp_is_moe:
            if type(layer.mlp.experts) is not MoERunner:
                raise RuntimeError("Unsupported MoE runner for mHC row ownership")
            if layer.mlp.experts.moe_config.tp_size != config.tensor_parallel_size:
                raise RuntimeError("mHC MoE reduction differs from the TP group")
            validate_moe_deferral(layer.mlp.experts)
        elif (
            type(layer.mlp.down_proj) is not RowParallelLinear
            or layer.mlp.down_proj.tp_size != config.tensor_parallel_size
            or not layer.mlp.down_proj.reduce_results
            or layer.mlp.down_proj.bias is not None
        ):
            raise RuntimeError("Unsupported dense FFN reduction for mHC row ownership")
    if not names:
        raise RuntimeError("mHC prefill requires explicit GDN metadata owners")
    return tuple(names)


@dataclass
class PrefillOwnership:
    """Hold per-forward row ownership and return caller-owned collective outputs.

    Each collective allocates its own output. Final and auxiliary tensors escape
    the model forward; shared reusable buffers would need explicit consumer
    lifetime ownership. Stream records protect allocator reuse after release.
    """

    comm: Any
    rank: int
    rows: int = 8192
    world_size: int = 4
    rs_count: int = 0
    ag_count: int = 0
    diagnostics: PrefillDiagnostics | None = None
    diagnostic_kind: str = "request"
    diagnostic_forward: int = 0
    mhc_rows: dict[str, dict[int, int]] = field(default_factory=dict)

    def record_mhc(self, stage: str, output: Any) -> None:
        """Count the output rows of an mHC call that returned to the host."""
        if self.diagnostics is not None:
            counts = self.mhc_rows.setdefault(stage, {})
            rows = output.shape[0]
            counts[rows] = counts.get(rows, 0) + 1

    def local_view(self, tensor: Any) -> Any:
        if tensor.shape[0] != self.rows:
            raise RuntimeError("mHC full-to-owner row count mismatch")
        q = self.rows // self.world_size
        return tensor.narrow(0, self.rank * q, q)

    def _check(self, tensor: Any, expected_rows: int) -> None:
        import torch

        if not self.comm.available or self.comm.disabled:
            raise RuntimeError("mHC PyNccl communicator became unavailable")
        if (
            tensor.shape[0] != expected_rows
            or not tensor.is_cuda
            or tensor.device != self.comm.device
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                "mHC collective requires contiguous BF16 owner/full rows"
            )

    def reduce_scatter(self, partial: Any) -> Any:
        from vllm.utils.torch_utils import current_stream

        self._check(partial, self.rows)
        if tuple(partial.shape) != (self.rows, 4096):
            raise RuntimeError("mHC reduce-scatter expects a full hidden TP partial")
        output = partial.new_empty((self.rows // self.world_size, 4096))
        stream = current_stream()
        self.comm.reduce_scatter(output, partial, stream=stream)
        partial.record_stream(stream)
        output.record_stream(stream)
        self.rs_count += 1
        return output

    def all_gather(self, owned: Any) -> Any:
        from vllm.utils.torch_utils import current_stream

        self._check(owned, self.rows // self.world_size)
        output = owned.new_empty((self.rows, *owned.shape[1:]))
        stream = current_stream()
        self.comm.all_gather(output, owned, stream=stream)
        owned.record_stream(stream)
        output.record_stream(stream)
        self.ag_count += 1
        return output

    def finish(self, layers: int, auxiliary_gathers: int) -> None:
        global _REPORTS
        if (
            self.rs_count != 2 * layers
            or self.ag_count != 2 * layers + auxiliary_gathers
        ):
            raise RuntimeError(
                "mHC collective accounting mismatch: "
                f"RS={self.rs_count} AG={self.ag_count}"
            )
        if self.diagnostics is not None and _LOG.isEnabledFor(logging.WARNING):
            _LOG.warning(
                "GLM_MHC_ENQUEUE %s",
                json.dumps(
                    dict(
                        rank=self.rank,
                        kind=self.diagnostic_kind,
                        forward=self.diagnostic_forward,
                        rows=self.rows,
                        owner_rows=self.rows // self.world_size,
                        rs=self.rs_count,
                        ag=self.ag_count,
                        auxiliary_gathers=auxiliary_gathers,
                        mhc_rows=self.mhc_rows,
                        decisions=self.diagnostics.decisions,
                        gpu_completion_verified=False,
                    ),
                    sort_keys=True,
                ),
            )
        if _REPORTS < 8 and _LOG.isEnabledFor(logging.INFO):
            _LOG.info(
                "GLM_MHC_PREFILL rank=%d rows=%d owner_rows=%d rs=%d ag=%d aux=%d",
                self.rank,
                self.rows,
                self.rows // self.world_size,
                self.rs_count,
                self.ag_count,
                auxiliary_gathers,
            )
            _REPORTS += 1


def maybe_create(model: Any, hidden: Any, positions: Any) -> PrefillOwnership | None:
    diagnostics = getattr(model, "_mhc_prefill_diagnostics", None)
    if not getattr(model, "_mhc_prefill_enabled", False):
        if diagnostics is not None:
            diagnostics.decision("disabled")
        return None
    import torch

    # Avoid Python diagnostic mutation while tracing a compiled graph.
    if torch.compiler.is_compiling():
        return None
    rows = model._mhc_prefill_rows
    if tuple(hidden.shape) != (rows, 4096):
        if diagnostics is not None:
            diagnostics.decision("hidden_shape", shape=list(hidden.shape))
        return None
    if torch.cuda.is_current_stream_capturing():
        if diagnostics is not None:
            diagnostics.decision("stream_capture")
        return None
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        if diagnostics is not None:
            diagnostics.decision("missing_forward_context")
        return None
    context = get_forward_context()
    if (
        context.cudagraph_runtime_mode.name != "NONE"
        or context.ubatch_slices is not None
    ):
        if diagnostics is not None:
            diagnostics.decision(
                "execution_context",
                mode=context.cudagraph_runtime_mode.name,
                has_ubatches=context.ubatch_slices is not None,
            )
        return None
    group = get_tp_group()
    if group.world_size not in (2, 4):
        raise RuntimeError("mHC prefill TP group must contain two or four ranks")
    comm = getattr(group.device_communicator, "pynccl_comm", None)
    error = None
    try:
        if (
            comm is None
            or not comm.available
            or comm.disabled
            or comm.world_size != group.world_size
            or group.world_size
            != model._mhc_prefill_parallel_config.tensor_parallel_size
        ):
            raise RuntimeError(
                "mHC prefill requires the enabled TP PyNccl communicator"
            )
        if comm.rank != group.rank_in_group or comm.device != hidden.device:
            raise RuntimeError("mHC communicator rank/device ownership mismatch")
        names = validate_model(model)
    except (AttributeError, RuntimeError) as exc:
        names = ()
        error = str(exc)
    eligible = (
        error is None
        and positions.shape[0] == rows
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and pure_prefill_metadata(context.attn_metadata, names, rows)
    )
    if diagnostics is not None:
        details = metadata_diagnostics(context.attn_metadata, names)
        diagnostics.decision(
            "local_eligible" if eligible else "local_ineligible",
            positions_rows=positions.shape[0],
            hidden_is_cuda=hidden.is_cuda,
            hidden_is_bf16=hidden.dtype == torch.bfloat16,
            capability_error=error,
            **details,
        )
    # Small host vote before changing ownership; no GPU metadata synchronization.
    # Rank-local capability or metadata differences cannot silently choose
    # incompatible reduction paths after one rank has produced partial output.
    votes: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(votes, (eligible, error), group=group.cpu_group)
    errors = [item[1] for item in votes if item[1] is not None]
    if errors:
        raise RuntimeError("mHC prefill capability vote failed: " + repr(errors))
    if not all(item[0] for item in votes):
        if diagnostics is not None:
            diagnostics.decision("peer_ineligible", votes=votes)
        return None
    kind = "request"
    forward = 0
    if diagnostics is not None:
        kind = diagnostics.decision("admitted")
        diagnostics.admitted += 1
        forward = diagnostics.admitted
    return PrefillOwnership(
        comm=comm,
        rank=group.rank_in_group,
        rows=rows,
        world_size=group.world_size,
        diagnostics=diagnostics,
        diagnostic_kind=kind,
        diagnostic_forward=forward,
    )
