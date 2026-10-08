"""SIRCL's four-rank native sessions, and when they may serve a group.

The four-rank sessions (``libspark_transport_capi.so``) and their vLLM adapter
(``integrations/vllm/spark_tp4_backend.py`` with its sibling modules on
``PYTHONPATH``, installed from ``sitecustomize`` when ``VLLM_SPARK_TP4_MODE`` is
set) are SparkRing's qualified implementation for a four-Spark ring. The
adapter installs itself by wrapping ``CudaCommunicator.all_reduce`` and keeps
vLLM's original method as ``_spark_original``. SIRCL's communicator overrides
``all_reduce``, so the wrapped base method is reached only when this module
says the four-rank sessions admit a call.

The sessions are offered only when:

- the four-rank adapter is installed and its source hash equals the pinned
  revision below (its eligibility functions are called directly);
- the group is the four-rank tensor-parallel group;
- both perfect matchings the sessions use, ranks (0,1),(2,3) and (1,2),(3,0),
  share cables. Four consecutive Sparks of a larger ring form a path whose
  rank 3 to rank 0 pair shares no cable, which those sessions cannot serve.

The sibling vocabulary all-gather adapter
(``spark_tp4_vocab_allgather_backend.py``) wraps
``GroupCoordinator._all_gather_out_place`` in front of every device
communicator and has the same cabling requirement. :func:`conflicts` reports
either adapter installed for a group they cannot serve, and SIRCL's
communicator refuses to start in that case instead of letting their native
sessions try to reach an uncabled peer.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from .fabric import GroupTopology

# integrations/vllm/spark_tp4_backend.py at SparkRing 6d27dd10 (LF-normalized).
PINNED_ADAPTER_SHA256 = "fa90938b8daf251bb5b5daf95c19b9311a4c50c8065edb0aabb2156632bab1ed"
MATCHINGS = ((0, 1), (2, 3), (1, 2), (3, 0))


def _adapter_module() -> Any | None:
    return sys.modules.get("spark_tp4_backend")


def installed(communicator_class: type) -> bool:
    """The four-rank all-reduce adapter wraps ``communicator_class.all_reduce``."""
    return bool(getattr(getattr(communicator_class, "all_reduce", None), "_spark_tp4_backend", False))


def vocab_installed(coordinator_class: type | None) -> bool:
    method = getattr(coordinator_class, "_all_gather_out_place", None) if coordinator_class else None
    return getattr(method, "_spark_original", None) is not None


def cycle_of_four(topology: GroupTopology | None) -> bool:
    return (topology is not None and len(topology.members) == 4
            and all(topology.pair_cabled(a, b) for a, b in MATCHINGS))


def unavailable_reason(topology: GroupTopology | None, *, kind: str,
                       communicator_class: type) -> str | None:
    """None when the four-rank sessions may serve this group, else why not."""
    if kind != "tp" or topology is None or len(topology.members) != 4:
        return "the four-rank sessions serve only a four-rank tensor-parallel group"
    module = _adapter_module()
    if module is None or not installed(communicator_class):
        return "the four-rank adapter is not installed"
    path = Path(getattr(module, "__file__", "") or "")
    try:
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return "the four-rank adapter source cannot be read"
    if digest != PINNED_ADAPTER_SHA256:
        return f"the four-rank adapter at {path} is not the pinned revision"
    for a, b in MATCHINGS:
        if not topology.pair_cabled(a, b):
            return f"group ranks {a} and {b} share no cable"
    return None


def admits(communicator: Any, tensor: Any, *, capturing: bool) -> bool:
    """Whether the installed four-rank adapter's own checks admit this all-reduce."""
    module = _adapter_module()
    if module is None:
        return False
    mode = module._mode()
    checks = [
        lambda: module._eligible(communicator, tensor, mode),
        lambda: module._bidirectional_prefill_eligible(communicator, tensor, mode=mode,
                                                       capturing=capturing),
        lambda: module._fused_prefill_eligible(communicator, tensor, mode=mode, capturing=capturing),
        lambda: (module._graph_width4096_research_enabled()
                 and module._research_graph_shape_eligible(tuple(tensor.shape))),
    ]
    for check in checks:
        try:
            if check():
                return True
        except AttributeError:
            continue
    return False


def conflicts(topology: GroupTopology | None, *, kind: str, communicator_class: type,
              coordinator_class: type | None = None, environ=None) -> list[str]:
    """Four-rank adapters that are installed for a tensor-parallel group they cannot serve."""
    env = os.environ if environ is None else environ
    if kind != "tp" or topology is None or len(topology.members) != 4 or cycle_of_four(topology):
        return []
    found = []
    tp4_mode = env.get("VLLM_SPARK_TP4_MODE", "").strip().lower()
    if installed(communicator_class) and tp4_mode not in ("", "disabled"):
        found.append("the four-rank all-reduce adapter (VLLM_SPARK_TP4_MODE="
                     f"{tp4_mode}) needs a four-Spark ring; this group is "
                     f"{topology.describe()}")
    if vocab_installed(coordinator_class) and env.get("VLLM_SPARK_TP4_VOCAB_MODE", "").strip():
        found.append("the four-rank vocabulary all-gather adapter (VLLM_SPARK_TP4_VOCAB_MODE) needs "
                     f"a four-Spark ring; this group is {topology.describe()}")
    return found
