"""SIRCL point-to-point channels: send, receive and batched send/receive between the ranks of a group.

Every ordered pair of a group's ranks is a first-in first-out channel of RDMA
writes into pinned host slots, paced by credits, over the lanes of SIRCL's
route maps (relayed lanes within forward windows). :class:`PointToPoint`
(:mod:`.session`) is one rank's context of one group; the vLLM adapter builds
one for every pipeline-parallel group and for every group with a collective
session (``sparkring_sircl/vllm``).

Importing this package is cheap; the first access to ``PointToPoint`` or
``P2PWork`` imports torch, CUDA Python and the CuTe DSL. Torch-free modules:
:mod:`.protocol` (arena layout, items, headers), :mod:`.settings`
(environment), :mod:`.budget` (forward windows within the relay rule),
:mod:`.build` and :mod:`._native` (the native library).
"""

from __future__ import annotations

from .protocol import ABI_VERSION, API_VERSION

__all__ = ["ABI_VERSION", "API_VERSION", "P2PWork", "PointToPoint", "is_supported"]


def is_supported() -> bool:
    """True when torch sees a CUDA device and CUDA Python imports (the channels also need RDMA devices and a GPU
    that addresses pinned host memory at its host pointer, which setup checks)."""
    try:
        import cuda.bindings  # noqa: F401
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def __getattr__(name: str):
    if name in ("PointToPoint", "P2PWork"):
        from . import session

        return getattr(session, name)
    raise AttributeError(name)
