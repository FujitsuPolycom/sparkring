"""Explicit recurrent checkpoint positions for bounded fresh-prefill coalescing."""
from __future__ import annotations

CheckpointPlan = tuple[int, int, tuple[int, ...]]


def validate_plan(plan: CheckpointPlan, start: int, end: int, block_size: int) -> tuple[int, ...]:
    planned_start, planned_end, targets = plan
    if (planned_start, planned_end) != (start, end):
        raise ValueError('recurrent checkpoint plan does not match the actual query span')
    if not isinstance(targets, tuple) or not 1 <= len(targets) <= 2 or targets != tuple(sorted(set(targets))):
        raise ValueError('recurrent checkpoint targets must be one or two sorted unique positions')
    if any(type(p) is not int or not start < p < end or p % block_size or (p - start) % 16 for p in targets):
        raise ValueError('recurrent checkpoint target is not a representable interior state')
    return targets


def fresh_prompt_plan(*, start: int, end: int, prompt: int, num_tokens: int,
                      block_size: int, publications: tuple[int, ...],
                      shared_prefix_boundary: int = 0) -> CheckpointPlan | None:
    # Resume, partial-tail and intermediate-prefill behavior stays in the
    # existing scheduler. This first path only appends a completely new table.
    if start != 0 or end != prompt or num_tokens != prompt or end > 8192 or end % block_size:
        return None
    required = {p for p in publications if start < p < end}
    predecessor = max((num_tokens - 1) // block_size * block_size - block_size, 0)
    if start < predecessor < end:
        required.add(predecessor)
    if start < shared_prefix_boundary < end:
        required.add(shared_prefix_boundary // block_size * block_size)
    required.discard(0)
    targets = tuple(sorted(required))
    if not targets or len(targets) > 2:
        return None
    plan = (start, end, targets)
    try:
        validate_plan(plan, start, end, block_size)
    except ValueError:
        return None
    return plan


def checkpoint_metadata(plan: CheckpointPlan | None, start: int, end: int,
                        block_size: int, capacity: int) -> tuple[list[int], list[int]]:
    if capacity not in (1, 2):
        raise ValueError('unsupported recurrent checkpoint capacity')
    if plan is not None:
        targets = validate_plan(plan, start, end, block_size)
        if capacity < len(targets):
            raise ValueError('metadata checkpoint capacity smaller than scheduled plan')
    else:
        boundary = end // block_size * block_size
        targets = ((boundary,) if end % block_size and start < boundary < end and (boundary - start) % 16 == 0 else ())
    return ([p - start for p in targets] + [0] * (capacity - len(targets)),
            [p // block_size - 1 for p in targets] + [-1] * (capacity - len(targets)))
