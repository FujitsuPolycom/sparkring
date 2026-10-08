"""GLM-5.3-Flash's mHC prefill row ownership on tensor-parallel groups whose collectives SIRCL carries.

GLM-5.3-Flash mixes several residual streams in every decoder layer
(hyper-connections, "mHC"). With ``VLLM_GLM53_MHC_PREFILL_SHARD=1`` vLLM
(``models/glm5next/nvidia/mhc_prefill_sharding.py`` and ``model.py`` in the
image's build) splits the mixing of an eager prefill forward of exactly
``max_num_batched_tokens`` rows (8,192 in the profiles) among the
tensor-parallel ranks; each rank mixes its own ``rows / W`` rows. Per forward
of ``L`` decoder layers it issues, past the device communicator, straight to
the group's PyNccl communicator:

- ``2 L`` reduce-scatters along dimension 0 of the attention and MLP partial
  outputs, ``[rows, 4096]`` BF16 in, ``[rows / W, 4096]`` out
  (``mhc_prefill_sharding.py:339-351``, called at ``model.py:672`` and
  ``:698``);
- ``2 L`` all-gathers along dimension 0 of the owned rows, ``[rows / W, 4096]``
  BF16 in, ``[rows, 4096]`` out: before every attention but the first layer's
  (``model.py:657``), before every MLP (``:696``) and once after the last
  layer (``:1217``), plus one per auxiliary hidden-state layer (``:1198``;
  none with the MTP draft).

For GLM-5.3-Flash at TP4 (45 layers) one 8,192-token chunk carries 90
reduce-scatters of 64 MiB and 90 all-gathers of 16 MiB shards; the embedding
and the MTP draft keep their four all-reduces of ``[8192, 4096]``. Before the
ownership object exists, ``maybe_create`` (``mhc_prefill_sharding.py:408-513``)
checks the PyNccl communicator's ``available``, ``disabled``, ``world_size``,
``rank`` and ``device`` and votes over the gloo group.

On a group NCCL may not run, SIRCL's communicator builds no PyNccl, and that
check fails on every rank. The ``mhc_prefill_shard`` shim (:mod:`.shims`,
pinned to both files) wraps ``maybe_create``: while vLLM's own function runs,
the group's PyNccl slot holds :class:`SirclPrefillComm`, whose
``reduce_scatter`` and ``all_gather`` go through SIRCL's communicator, so they
follow the same rank-invariant plans as every other collective of the group:
a reduce-scatter is the session's reduce-scatter where the session has one,
otherwise an all-reduce (the rank-ordered float32 sum rounded once) and this
rank's rows; an all-gather copies bytes. The slot is restored
before ``maybe_create`` returns. Groups whose PyNccl runs (a pair of Sparks
with ``SIRCL_NCCL=auto``) keep vLLM's own path, and nothing in the forward reaches NCCL on a group NCCL
may not run.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from collections.abc import Mapping
from typing import Any

ENV = "VLLM_GLM53_MHC_PREFILL_SHARD"
MODULE = "vllm.models.glm5next.nvidia.mhc_prefill_sharding"
MODEL_MODULE = "vllm.models.glm5next.nvidia.model"
BOUND_NAME = "maybe_create_mhc_prefill_ownership"


def requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether vLLM will shard mHC prefill rows (``bool(int(value))``, as vLLM reads it)."""
    raw = (os.environ if environ is None else environ).get(ENV, "").strip()
    if not raw:
        return False
    try:
        return int(raw) != 0
    except ValueError:
        raise ValueError(f"{ENV}={raw} is not an integer (vLLM reads it with int())") from None


def _is_sum(op: Any) -> bool:
    if op is None:
        return True
    name = getattr(op, "name", None) or str(op)
    return str(name).upper().endswith("SUM")


@contextlib.contextmanager
def _on(stream: Any):
    """Run on ``stream`` when it is a CUDA stream other than the current one."""
    if stream is None:
        yield
        return
    import torch

    if not torch.cuda.is_available() or stream == torch.cuda.current_stream():
        yield
        return
    with torch.cuda.stream(stream):
        yield


class SirclPrefillComm:
    """The PyNccl surface vLLM's prefill row ownership uses, carried by SIRCL's communicator.

    GLM-5.3-Flash's mHC rows (this module) and Qwen3.8's hyper-connection rows
    (:mod:`.qwen_hc`) call these two methods with PyNccl's signatures.
    """

    available = True
    disabled = False

    def __init__(self, communicator: Any) -> None:
        self.communicator = communicator
        self.world_size = int(communicator.world_size)
        self.rank = int(communicator.rank_in_group)
        self.device = communicator.device

    @staticmethod
    def _store(output: Any, result: Any, what: str) -> None:
        if tuple(output.shape) != tuple(result.shape) or output.dtype != result.dtype:
            raise RuntimeError(f"prefill row ownership {what}: vLLM's output is {tuple(output.shape)} "
                               f"{output.dtype}, SIRCL's result {tuple(result.shape)} {result.dtype}")
        output.copy_(result)

    def reduce_scatter(self, output_tensor: Any, input_tensor: Any, op: Any = None, stream: Any = None) -> None:
        if not _is_sum(op):
            raise RuntimeError(f"prefill row ownership reduce-scatter with {op} is not a sum; SIRCL carries "
                               "sums only")
        with _on(stream):
            self._store(output_tensor, self.communicator.reduce_scatter(input_tensor, 0), "reduce-scatter")

    def all_gather(self, output_tensor: Any, input_tensor: Any, stream: Any = None) -> None:
        with _on(stream):
            self._store(output_tensor, self.communicator.all_gather(input_tensor, 0), "all-gather")


def prefill_comm(communicator: Any) -> SirclPrefillComm | None:
    """SIRCL's stand-in for ``communicator``'s PyNccl, or None when vLLM's own path applies.

    None unless SIRCL owns the group with a session and the group has no
    working PyNccl (NCCL may not run on it).
    """
    adapter = getattr(communicator, "sircl", None)
    if adapter is None or getattr(adapter, "session", None) is None:
        return None
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is not None and getattr(pynccl, "available", False) and not getattr(pynccl, "disabled", True):
        return None
    return SirclPrefillComm(communicator)


def install_shim(marker: str) -> None:
    """Wrap vLLM's ``maybe_create`` where the module and the GLM model bind it (idempotent)."""
    module = importlib.import_module(MODULE)
    original = module.maybe_create
    if getattr(original, marker, None) is not None:
        return

    def maybe_create(model: Any, hidden: Any, positions: Any) -> Any:
        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
        stand_in = prefill_comm(communicator)
        if stand_in is None:
            return original(model, hidden, positions)
        saved = communicator.pynccl_comm
        communicator.pynccl_comm = stand_in
        try:
            return original(model, hidden, positions)
        finally:
            communicator.pynccl_comm = saved

    maybe_create.__doc__ = original.__doc__
    maybe_create.__name__ = getattr(original, "__name__", "maybe_create")
    setattr(maybe_create, marker, original)
    module.maybe_create = maybe_create
    bound = sys.modules.get(MODEL_MODULE)
    if bound is not None and getattr(bound, BOUND_NAME, None) is original:
        setattr(bound, BOUND_NAME, maybe_create)
