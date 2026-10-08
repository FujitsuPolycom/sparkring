"""Host references of the session's all-reduces, bit for bit.

The one-shot and two-shot all-reduce store the float32 sum of every rank's
values in rank order 0..W-1, rounded once to the dtype. A chain op
(``oneshot/_chain_cute.py``) sums half A of its message in chain order and
half B in reverse chain order, rounding to the dtype at every hop.
``all_reduce_large`` runs the ops of ``session.large_reduce_plan(nbytes)``;
:func:`large_all_reduce` reproduces them from that plan and
``session.chain_order``. A chain reduce-scatter (``oneshot/_links_cute.py``)
gives the owner at chain index ``i`` the dtype rounding of ``(L + x) + R`` in
float32: ``L`` folds the values of chain indices ``0 .. i - 1`` from index 0,
``R`` those of ``W - 1 .. i + 1`` from index ``W - 1``, each fold rounding to
the dtype at every hop; :func:`chain_reduce_scatter` reproduces it. A ring
reduce-scatter and all-reduce give the owner at ring index ``k`` the values of
indices ``k + 1``, ..., ``k`` folded once around the ring
(:func:`ring_reduce_scatter`, :func:`ring_all_reduce`). The
functions take the ``torch`` module as an argument, so this module imports
without torch.
"""

from __future__ import annotations

from collections.abc import Sequence

from .protocol import PACK_BYTES


def rank_order_sum(torch, parts):
    """Float32 sum in rank order, rounded once to the dtype of ``parts[0]``."""
    total = parts[0].float().clone()
    for part in parts[1:]:
        total += part.float()
    return total.to(parts[0].dtype)


def chain_sum(torch, parts, order: Sequence[int]):
    """A chain op over 1-D ``parts`` (one per rank) of a multiple of 16 bytes."""
    dtype = parts[0].dtype
    item = parts[0].element_size()
    half = parts[0].numel() * item // PACK_BYTES // 2 * PACK_BYTES // item
    a = parts[order[0]][:half].clone()
    for rank in order[1:]:
        a = (a.float() + parts[rank][:half].float()).to(dtype)
    b = parts[order[-1]][half:].clone()
    for rank in reversed(order[:-1]):
        b = (b.float() + parts[rank][half:].float()).to(dtype)
    return torch.cat([a, b])


def large_all_reduce(torch, inputs, plan, order: Sequence[int] | None):
    """``all_reduce_large`` of ``inputs`` (one tensor per rank) for ``plan``."""
    flat = [tensor.reshape(-1) for tensor in inputs]
    item = flat[0].element_size()
    out = []
    for piece in plan:
        first, count = piece.offset // item, piece.nbytes // item
        parts = [tensor[first:first + count] for tensor in flat]
        if piece.chain or getattr(piece, "ring", False):
            if order is None:
                raise ValueError("a chain or ring piece needs the session's chain order")
            if piece.chain:
                out.append(chain_sum(torch, parts, order))
            else:
                out.append(torch.cat(ring_reduce_scatter(torch, parts, order)))
        else:
            out.append(rank_order_sum(torch, parts))
    if not out:
        return inputs[0].clone()
    return torch.cat(out).reshape(inputs[0].shape)


def chain_reduce_scatter(torch, inputs, order: Sequence[int], *, chunk_elements: int | None = None,
                         stride_elements: int | None = None) -> list:
    """Every rank's output of a chain reduce-scatter of ``inputs`` (one tensor per rank), in rank order.

    Chunk ``k`` (rank ``k``'s) of a flattened input holds ``chunk_elements`` elements from
    ``k * stride_elements`` (defaults: ``W`` contiguous chunks).
    """
    world = len(inputs)
    flat = [tensor.reshape(-1) for tensor in inputs]
    chunk = flat[0].numel() // world if chunk_elements is None else int(chunk_elements)
    stride = chunk if stride_elements is None else int(stride_elements)
    dtype = flat[0].dtype
    outputs: list = [None] * world
    for index, owner in enumerate(order):
        part = [tensor[owner * stride:owner * stride + chunk] for tensor in flat]

        def fold(indices):
            total = None
            for i in indices:
                value = part[order[i]]
                total = value.clone() if total is None else (total.float() + value.float()).to(dtype)
            return total

        left = fold(range(index))
        right = fold(range(world - 1, index, -1))
        total = part[owner].float()
        if left is not None:
            total = left.float() + total
        if right is not None:
            total = total + right.float()
        outputs[owner] = total.to(dtype)
    return outputs


def ring_reduce_scatter(torch, inputs, order: Sequence[int], *, chunk_elements: int | None = None,
                        stride_elements: int | None = None) -> list:
    """Every rank's output of a ring reduce-scatter over ``order`` closed by its last rank to its first.

    The owner at ring index ``k`` gets the dtype rounding, at every hop, of the values of ring indices
    ``k + 1``, ``k + 2``, ..., ``k`` (indices modulo ``W``) added in that order: its partial sum starts at
    the next rank and travels once around the ring. Chunks as in :func:`chain_reduce_scatter`.
    """
    world = len(inputs)
    flat = [tensor.reshape(-1) for tensor in inputs]
    chunk = flat[0].numel() // world if chunk_elements is None else int(chunk_elements)
    stride = chunk if stride_elements is None else int(stride_elements)
    dtype = flat[0].dtype
    outputs: list = [None] * world
    for index, owner in enumerate(order):
        part = [tensor[owner * stride:owner * stride + chunk] for tensor in flat]
        total = part[order[(index + 1) % world]].clone()
        for step in range(2, world + 1):
            total = (total.float() + part[order[(index + step) % world]].float()).to(dtype)
        outputs[owner] = total
    return outputs


def ring_all_reduce(torch, inputs, order: Sequence[int]):
    """A ring all-reduce over ``order`` of ``inputs`` (one tensor per rank, a multiple of ``W`` elements):
    chunk ``r`` (rank ``r``'s) of the message is :func:`ring_reduce_scatter`'s output of rank ``r``, and
    every rank holds every chunk."""
    flat = [tensor.reshape(-1) for tensor in inputs]
    return torch.cat(ring_reduce_scatter(torch, flat, order)).reshape(inputs[0].shape)


__all__ = ["chain_reduce_scatter", "chain_sum", "large_all_reduce", "rank_order_sum", "ring_all_reduce",
           "ring_reduce_scatter"]
