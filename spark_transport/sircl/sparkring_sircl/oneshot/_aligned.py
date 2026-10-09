"""Aligned working buffers for collectives whose kernels need 16-byte aligned pointers.

A collective's ops (ring, chain, scatter or transport ops) follow its shared arguments only: the message size,
shape, dtype and the session's agreed settings, never a rank's pointer alignment, which is a fact of each rank's
own memory. A rank whose input or output is not 16-byte aligned runs the same ops on aligned working buffers:
:func:`aligned_input` copies the input into a fresh allocation, :func:`aligned_output` gives a fresh output to copy
back from. Fresh allocations come from the caching allocator (at least 512-byte aligned); inside a CUDA graph
capture they come from the graph's private pool and replay at the same addresses, so the staging copies are part
of the graph.
"""

from __future__ import annotations

import torch

from ..protocol import PACK_BYTES


def aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % PACK_BYTES == 0


def aligned_input(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` when its pointer is 16-byte aligned, else a contiguous copy of it in a fresh allocation."""
    return tensor if aligned(tensor) else tensor.clone(memory_format=torch.contiguous_format)


def aligned_output(tensor: torch.Tensor, *, keep: bool = False) -> torch.Tensor:
    """``tensor`` when its pointer is 16-byte aligned, else a fresh tensor of its shape and dtype to copy back into
    it with :func:`copy_back`; with ``keep``, holding ``tensor``'s values (an op that leaves bytes unchanged)."""
    if aligned(tensor):
        return tensor
    if keep:
        return tensor.clone(memory_format=torch.contiguous_format)
    return torch.empty_like(tensor, memory_format=torch.contiguous_format)


def copy_back(work: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out`` holding ``work``'s values (nothing to do when they are the same tensor)."""
    if work is not out:
        out.copy_(work)
    return out


__all__ = ["aligned", "aligned_input", "aligned_output", "copy_back"]
