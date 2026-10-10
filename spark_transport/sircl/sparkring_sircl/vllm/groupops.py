"""CPU-group operations the adapter's setup votes use.

Setup agreement runs over a group's CPU (gloo) process group with
``torch.distributed`` object collectives. An object that provides
``sircl_rank()``, ``sircl_size()``, ``sircl_ranks()`` and
``sircl_all_gather_object(value)`` is accepted in place of a process group;
:class:`.emulation.EmulatedGroup` is one, so the CPU tests run every rank of a
group as a thread of one process with the production setup code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def rank(group: Any) -> int:
    if hasattr(group, "sircl_rank"):
        return int(group.sircl_rank())
    import torch.distributed as dist

    return dist.get_rank(group=group)


def size(group: Any) -> int:
    if hasattr(group, "sircl_size"):
        return int(group.sircl_size())
    import torch.distributed as dist

    return dist.get_world_size(group=group)


def ranks(group: Any, global_ranks: Sequence[int] | None = None) -> list[int]:
    """Global ranks of ``group`` in group order."""
    if hasattr(group, "sircl_ranks"):
        return list(group.sircl_ranks())
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _world

    if _world.pg_map.get(group, None) is None:   # a stateless group: vLLM passes its ranks
        if global_ranks is None:
            raise ValueError("a stateless process group needs its global ranks")
        return list(global_ranks)
    return dist.get_process_group_ranks(group)


def all_gather_object(group: Any, value: Any) -> list[Any]:
    if hasattr(group, "sircl_all_gather_object"):
        return list(group.sircl_all_gather_object(value))
    import torch.distributed as dist

    gathered: list[Any] = [None] * size(group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def vote(group: Any, local: tuple[str | None, Any] | str | None, *, compare: bool,
         ignore: Sequence[str] = ()) -> str | None:
    """Gather every rank's ``local``; the text of any failure or disagreement, else None.

    With ``compare``, ``local`` is ``(reason, settings)`` and the settings must
    equal rank 0's (mapping keys in ``ignore`` excepted); without it ``local``
    is an error text or None.
    """
    votes = all_gather_object(group, local)
    reasons = [vote[0] if compare else vote for vote in votes]
    failures = [f"rank {i}: {reason}" for i, reason in enumerate(reasons) if reason]
    if failures:
        return "; ".join(failures)
    if compare:
        def shared(settings: Any) -> Any:
            if isinstance(settings, dict):
                return {k: v for k, v in settings.items() if k not in ignore}
            return settings

        reference = shared(votes[0][1])
        differing = [f"rank {i}: {vote[1]}" for i, vote in enumerate(votes)
                     if shared(vote[1]) != reference]
        if differing:
            return f"settings differ from rank 0's {reference}: " + "; ".join(differing)
    return None
