"""Qwen3.8-Flash-Next's hyper-connection prefill row ownership on tensor-parallel groups whose collectives SIRCL carries.

Qwen3.8-Flash-Next carries ``hc`` hyper-connection streams per token
(``hc_count``, 4 by default, ``models/qwen4_exp/config.py:93``). With
``VLLM_QWEN3_8_HC_PREFILL_MODE=shard`` (the SparkRing Qwen3.8 profiles'
setting) vLLM splits the hyper-connection mixing of an eager pure prefill
among the tensor-parallel ranks: each rank keeps ``rows / W`` rows of the
multi-stream state, and the sublayers between exchange block boundaries
(``models/qwen4_exp/nvidia/hc_prefill.py`` and ``model.py`` in the image's
build):

- admission at model construction (``hc_prefill.py:17-49``): tensor
  parallelism 2 or 4 with pipeline, data, decode-context and prefill-context
  parallelism 1, no expert parallelism, EPLB, sequence-parallel MoE or
  dual-batch overlap, BF16 weights, and no hyper-connection projection
  sharding; the ranks vote over the CPU group;
- admission per forward (``hc_prefill.py:52-84``, called at
  ``model.py:785``): at least 1,024 rows, a multiple of ``W``, in an eager
  pure prefill (not compiling or capturing, not a dummy run, graph mode NONE
  or PIECEWISE, no micro-batches, every recurrent layer's metadata a prefill
  of exactly those rows and no decodes);
- ``create`` (``hc_prefill.py:123-130``, called at ``model.py:595``) takes the
  tensor-parallel group's PyNccl communicator and requires it available and
  enabled; the ownership object then calls it directly, on the current
  stream:

  - ``reduce`` (``hc_prefill.py:113-120``): ``reduce_scatter`` (sum) along
    dimension 0, ``[rows, H]`` BF16 in, ``[rows / W, H]`` out, of every
    attention and MLP output (``model.py:387`` and ``:405``), ``2 L`` per
    forward of ``L`` decoder layers;
  - ``gather`` (``hc_prefill.py:104-111``): ``all_gather`` along dimension 0
    of owned rows, ``[rows / W, H]`` in, ``[rows, H]`` out, before every
    attention and MLP (``model.py:374`` and ``:401``, ``2 L`` per forward)
    and once for the sampled hidden states (``:658``); ``[rows / W, hc H]``
    in, ``[rows, hc H]`` out for the multi-stream state before every PLE layer
    (``:353``) and once for the MTP drafter's state (``:660``).

For an 8,192-row chunk at TP4, ``H`` = 2,048 (the Qwen3-Next configuration's
default) and ``hc`` = 4 (check both against the checkpoint), each
reduce-scatter takes 32 MiB in and gives 8 MiB per rank; each block-input
gather turns 8 MiB per rank into 32 MiB, and each multi-stream gather 32 MiB
into 128 MiB.

On a group NCCL may not run, SIRCL's communicator builds no PyNccl, and
``create`` raises on every rank ("Qwen HC ownership requires the TP NCCL
communicator"). The ``qwen_hc_prefill_shard`` shim (:mod:`.shims`, pinned to
both files of the image's build) wraps ``create``: while vLLM's own function
runs, the group's PyNccl slot holds :class:`.mhc.SirclPrefillComm`, whose
``reduce_scatter`` and ``all_gather`` go through SIRCL's communicator and
follow the group's rank-invariant plans: the session's reduce-scatter on the
whole message where the session has one (otherwise an all-reduce and this
rank's rows), and the session's large-message all-gather above the gather op
size. The slot is restored before ``create`` returns; ``model.py`` calls
``hc_prefill.create`` through its module, so no other binding needs the
wrapper. Groups whose PyNccl runs (a pair of Sparks with ``SIRCL_NCCL=auto``)
keep vLLM's own path.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from typing import Any

from . import mhc

ENV = "VLLM_QWEN3_8_HC_PREFILL_MODE"
MODULE = "vllm.models.qwen4_exp.nvidia.hc_prefill"
SHARD = "shard"


def mode(environ: Mapping[str, str] | None = None) -> str:
    """The row-ownership mode as vLLM reads it (``os.getenv`` with default ``off``, no normalization)."""
    return (os.environ if environ is None else environ).get(ENV, "off")


def requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether vLLM will give Qwen3.8 prefill rows to their owning ranks (``shard``)."""
    return mode(environ) == SHARD


def install_shim(marker: str) -> None:
    """Wrap vLLM's ``create`` in its module, where the Qwen model looks it up (idempotent)."""
    module = importlib.import_module(MODULE)
    original = module.create
    if getattr(original, marker, None) is not None:
        return

    def create(model: Any, rows: int) -> Any:
        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
        stand_in = mhc.prefill_comm(communicator)
        if stand_in is None:
            return original(model, rows)
        saved = communicator.pynccl_comm
        communicator.pynccl_comm = stand_in
        try:
            return original(model, rows)
        finally:
            communicator.pynccl_comm = saved

    create.__doc__ = original.__doc__
    create.__name__ = getattr(original, "__name__", "create")
    setattr(create, marker, original)
    module.create = create
