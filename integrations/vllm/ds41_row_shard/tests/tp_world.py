"""Simulated tensor-parallel ranks: one thread per rank, collectives in rank order.

Every sum adds the ranks' contributions in rank order, as a ring or two-shot
reduction over a fixed order would, so an all-reduce equals the
concatenation of the matching reduce-scatter's shards bit for bit. That is
the property the owned-row path relies on; the tests use float64 so any
difference in arithmetic, not only in summation order, is visible.
"""

from __future__ import annotations

import threading

import torch

_CONTEXT = threading.local()


def rank() -> int:
    return _CONTEXT.rank


def group() -> "Group":
    return _CONTEXT.group


class Group:
    """The subset of vLLM's ``GroupCoordinator`` the plugin calls."""

    def __init__(self, world: int):
        self.world_size = world
        self._barrier = threading.Barrier(world, timeout=60.0)
        self._slots: list = [None] * world
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    @property
    def rank_in_group(self) -> int:
        return rank()

    def _exchange(self, tensor):
        self._barrier.wait()
        self._slots[rank()] = tensor.clone()
        self._barrier.wait()
        items = list(self._slots)
        self._barrier.wait()
        return items

    def all_reduce(self, tensor):
        items = self._exchange(tensor)
        total = items[0].clone()
        for item in items[1:]:
            total = total + item
        if rank() == 0:
            self.calls.append(("all_reduce", tuple(tensor.shape)))
        return total

    def reduce_scatter(self, tensor, dim=-1):
        assert dim == 0
        items = self._exchange(tensor)
        chunk = tensor.shape[0] // self.world_size
        start = rank() * chunk
        total = items[0][start : start + chunk].clone()
        for item in items[1:]:
            total = total + item[start : start + chunk]
        if rank() == 0:
            self.calls.append(("reduce_scatter", tuple(tensor.shape)))
        return total

    def all_gather(self, tensor, dim=-1):
        assert dim == 0
        items = self._exchange(tensor)
        if rank() == 0:
            self.calls.append(("all_gather", tuple(tensor.shape)))
        return torch.cat(items, dim=0)


def run(world: int, body):
    """Run ``body(rank)`` on ``world`` threads sharing one ``Group``; return results."""
    shared = Group(world)
    results: list = [None] * world
    errors: list = []

    def target(index: int) -> None:
        _CONTEXT.rank = index
        _CONTEXT.group = shared
        try:
            results[index] = body(index)
        except BaseException as exc:  # surfaced below
            errors.append(exc)
            shared._barrier.abort()

    threads = [threading.Thread(target=target, args=(i,)) for i in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        primary = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
        raise (primary or errors)[0]
    return results, shared
