# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint plans and physical ownership for bounded KDA prefill."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.worker.gpu.input_batch import InputBatch

CheckpointPlan = tuple[int, int, tuple[int, ...]]

# DCP4 retention preserves two fine-grid and two scheduler-grid states.
COALESCED_CHECKPOINT_CAPACITY = 4


def validate_plan(
    plan: CheckpointPlan, start: int, end: int, block_size: int
) -> tuple[int, ...]:
    planned_start, planned_end, targets = plan
    if (planned_start, planned_end) != (start, end):
        raise ValueError(
            "recurrent checkpoint plan does not match the actual query span"
        )
    if (
        not isinstance(targets, tuple)
        or not 0 <= len(targets) <= COALESCED_CHECKPOINT_CAPACITY
        or targets != tuple(sorted(set(targets)))
    ):
        raise ValueError(
            "recurrent checkpoint targets must be up to four sorted unique positions"
        )
    if any(
        type(p) is not int or not start < p < end or p % block_size or (p - start) % 16
        for p in targets
    ):
        raise ValueError(
            "recurrent checkpoint target is not a representable interior state"
        )
    return targets


def prefill_checkpoint_plan(
    *,
    start: int,
    end: int,
    prompt: int,
    num_tokens: int,
    block_size: int,
    publications: tuple[int, ...],
    shared_prefix_boundary: int = 0,
) -> CheckpointPlan | None:
    """Export exact retained states inside a bounded pure-prompt chunk."""
    if (
        start < 0
        or start % block_size
        or not start < end <= prompt
        or num_tokens != prompt
        or end - start > 8192
        or end % block_size
    ):
        return None
    required = {position for position in publications if start < position < end}
    if start < shared_prefix_boundary < end:
        required.add(shared_prefix_boundary // block_size * block_size)
    required.discard(0)
    plan = (start, end, tuple(sorted(required)))
    try:
        validate_plan(plan, start, end, block_size)
    except ValueError:
        return None
    return plan


def checkpoint_metadata(
    plan: CheckpointPlan | None, start: int, end: int, block_size: int, capacity: int
) -> tuple[list[int], list[int]]:
    if capacity not in (1, 2, COALESCED_CHECKPOINT_CAPACITY):
        raise ValueError("unsupported recurrent checkpoint capacity")
    if plan is not None:
        targets = validate_plan(plan, start, end, block_size)
        if capacity < len(targets):
            raise ValueError("metadata checkpoint capacity smaller than scheduled plan")
    else:
        boundary = end // block_size * block_size
        targets = (
            (boundary,)
            if end % block_size
            and start < boundary < end
            and (boundary - start) % 16 == 0
            else ()
        )
    return (
        [p - start for p in targets] + [0] * (capacity - len(targets)),
        [p // block_size - 1 for p in targets] + [-1] * (capacity - len(targets)),
    )


def continuation_layout(
    blocks: Sequence[KVCacheBlock],
    plan: CheckpointPlan,
    block_size: int,
    speculative_blocks: int,
) -> tuple[int, tuple[int, ...]]:
    """Return retained column indices, -1 for NULL and -2 for allocation.

    Checkpoint columns retain their physical pages. Unused private speculative
    pages move to appended columns so worker block tables remain append-only.
    """
    start, end, targets = plan
    validate_plan(plan, start, end, block_size)
    if start <= 0 or start % block_size or end % block_size:
        raise ValueError(
            "continuation requires aligned nonzero source and final states"
        )
    source = start // block_size - 1
    final = end // block_size - 1
    if speculative_blocks < 0 or len(blocks) != source + 1 + speculative_blocks:
        raise ValueError("continuation table does not end at its expected reserve")
    if blocks[source].is_null or blocks[source].ref_cnt < 1:
        raise ValueError("continuation source is not retained")
    reserve_columns = range(source + 1, len(blocks))
    for column in reserve_columns:
        block = blocks[column]
        if block.is_null or block.ref_cnt != 1 or block.block_hash is not None:
            raise ValueError("continuation reserve must be private and unhashed")
    owned = [block.block_id for block in blocks if not block.is_null]
    if len(owned) != len(set(owned)):
        raise ValueError("recurrent table aliases physical ownership")
    outputs = [p // block_size - 1 for p in targets] + [final]
    reserves = list(range(final + 1, final + 1 + speculative_blocks))
    desired = set(outputs + reserves)
    if len(desired) != len(outputs) + len(reserves):
        raise ValueError("continuation outputs alias")
    layout = list(range(len(blocks))) + [-1] * (
        final + 1 + speculative_blocks - len(blocks)
    )
    movable = []
    for column in reserve_columns:
        if column not in desired:
            movable.append(column)
            layout[column] = -1
    for column in outputs:
        if column >= len(blocks):
            layout[column] = -2
    for column in reserves:
        if column >= len(blocks):
            layout[column] = movable.pop(0) if movable else -2
    if movable or layout.count(-2) != len(targets) + 1:
        raise ValueError("continuation reserve accounting differs")
    for column in desired:
        if column < len(blocks) and layout[column] != column:
            raise ValueError("continuation would replace an existing worker column")
    return source, tuple(layout)


def validate_coalescing_config(config: VllmConfig) -> bool:
    """Validate the opt-in serving contract without querying a GPU."""
    from vllm import envs

    if not envs.VLLM_B12X_KDA_PREFILL_COALESCING:
        return False
    model = config.model_config
    cache = config.cache_config
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    additional = config.additional_config
    spec = config.speculative_config
    supported = (
        model is not None
        and model.hf_text_config.model_type in ("glm5_next", "glm5_next_text")
        and str(model.dtype) == "torch.bfloat16"
        and config.use_v2_model_runner
        and config.lora_config is None
        and not model.enable_sleep_mode
        and not model.enable_return_routed_experts
        and (
            (
                parallel.tensor_parallel_size == 4
                and parallel.decode_context_parallel_size in (1, 2, 4)
            )
            or (
                parallel.tensor_parallel_size == 2
                and parallel.decode_context_parallel_size == 1
            )
        )
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and not parallel.enable_expert_parallel
        and not parallel.enable_eplb
        and scheduler.max_num_batched_tokens == 8192
        and scheduler.max_num_scheduled_tokens in (None, 8192)
        and scheduler.long_prefill_token_threshold in (0, 8192)
        and scheduler.prefill_compute_share is None
        and scheduler.max_parallel_prefills == 1
        and scheduler.enable_chunked_prefill
        and cache.enable_prefix_caching
        and cache.mamba_cache_mode == "align"
        and cache.prefix_cache_retention_interval == 0
        and not config.use_request_boundary_checkpoints
        and isinstance(additional, dict)
        and additional.get("kda_prefill_backend") == "b12x"
        and (
            spec is None
            or (
                spec.method == "mtp"
                and spec.num_speculative_tokens == 3
                and not spec.uses_dynamic_speculative_decoding()
            )
        )
    )
    if not supported:
        raise ValueError(
            "VLLM_B12X_KDA_PREFILL_COALESCING requires GLM5Next BF16, V2, "
            "TP4 with DCP1/2/4 or TP2 with DCP1, PP1/DP1, B12X KDA, "
            "an 8192-token scheduler budget, "
            "align-mode prefix caching with retention interval 0, and static "
            "MTP3 or no speculation; LoRA, EP, PCP and fairness engines are unsupported"
        )
    return True


def checkpoint_plan_rows(
    input_batch: InputBatch,
    plans: dict[str, CheckpointPlan] | None,
    num_reqs: int,
    *,
    for_capture: bool = False,
) -> list[CheckpointPlan | None] | None:
    """Map scheduler plans to the worker's packed request order."""
    if not plans:
        return None
    if for_capture:
        raise ValueError("recurrent checkpoint plans cannot enter graph capture")
    if (
        len(input_batch.req_ids) != input_batch.num_reqs
        or num_reqs < input_batch.num_reqs
    ):
        raise ValueError("checkpoint request counts differ from the worker batch")
    positions = {request_id: row for row, request_id in enumerate(input_batch.req_ids)}
    if len(positions) != input_batch.num_reqs:
        raise ValueError("checkpoint batch contains duplicate request IDs")
    rows: list[CheckpointPlan | None] = [None] * num_reqs
    for request_id, plan in plans.items():
        row = positions.get(request_id)
        if row is None:
            raise ValueError("checkpoint request is absent from the worker batch")
        start, end, targets = plan
        validate_plan(plan, start, end, 16)
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or int(input_batch.num_computed_tokens_np[row]) != start
            or int(input_batch.num_scheduled_tokens[row]) != end - start
            or int(
                input_batch.query_start_loc_np[row + 1]
                - input_batch.query_start_loc_np[row]
            )
            != end - start
        ):
            raise ValueError("checkpoint plan differs from the worker query span")
        rows[row] = (start, end, targets)
    return rows
