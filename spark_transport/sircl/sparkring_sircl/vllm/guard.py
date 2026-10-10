"""Keep NCCL off every pair of Sparks that shares no cable, and fail loudly instead.

NCCL picks its RDMA devices and addresses without the relay table, so its
queue-pair setup toward a Spark that shares no cable with the caller times out
(``ibv_modify_qp`` INIT to RTR). A group of four consecutive Sparks of a larger
ring therefore has no NCCL fallback at all: NCCL's ring would need rank 3 to
connect to rank 0. This module enforces that in three places:

1. **Construction.** :func:`pynccl_suppressed` makes vLLM build a group's
   communicator without a PyNccl communicator, whose constructor otherwise
   runs a warm-up all-reduce over the whole group right after
   ``ncclCommInitRank``. It sets vLLM's own ``VLLM_DISABLE_PYNCCL`` for the
   duration of the constructor only, which vLLM reads live until its worker
   caches the environment after the model is loaded.
2. **Dispatch.** SIRCL's communicator asks :meth:`GuardedGroup.check` before
   it hands any collective to NCCL.
3. **Tripwire.** :func:`install_tripwire` wraps the ``torch.distributed``
   collective functions so that a call on an NCCL process group whose ranks
   NCCL may not connect for that operation raises
   :class:`NcclAcrossRelayError` instead of hanging. This catches vLLM code
   that bypasses the device communicator (for example
   ``GroupCoordinator.broadcast``, the DCP all-to-all combine, or the weight
   ``amax`` reductions of online quantization). A group with a SIRCL session
   registers a carrier (:func:`register_carrier`); the tripwire hands it the
   refused call, and the carrier runs the calls it supports on the session
   (:data:`CARRIED`) instead of raising. A group with point-to-point
   channels registers a point-to-point carrier (:func:`register_p2p_carrier`),
   which the tripwire asks first for ``send``, ``recv``, ``isend``, ``irecv``
   and ``batch_isend_irecv``, whatever NCCL may do on the group, and after a
   refusal for a ``broadcast``; it is also found by the group's ranks for an
   NCCL group SIRCL did not build (vLLM's sibling of the pipeline group). It
   cannot see collectives issued from C++ (functional collectives
   inside compiled graphs); the communicator refuses the compilation passes
   that create them.

A ring group (consecutive ranks share cables, last to first included) may run
NCCL's ring algorithm only. The repository's patched-NCCL cycle contract
(``spark_transport/nccl/README.md``) enforces that with ``NCCL_ALGO=Ring`` and
``NCCL_SKIP_TREE_CONNECT=1``; without both, :func:`effective_policy` treats the
group as one NCCL may not touch.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from . import settings
from .fabric import NcclPolicy


class NcclAcrossRelayError(RuntimeError):
    """A collective would make NCCL connect Sparks that share no cable."""


@dataclasses.dataclass(frozen=True)
class GuardedGroup:
    """A process group and what NCCL may do on it."""

    name: str
    ranks: tuple[int, ...]                      # global ranks in group order
    policy: NcclPolicy
    reason: str
    cabled: Callable[[int, int], bool] | None = None   # global rank pair -> shares a cable
    remedy: str = ""                            # how to let a refused call run, for the error text

    def allows(self, operation: str, *, pair: tuple[int, int] | None = None) -> bool:
        if self.policy is NcclPolicy.ALL:
            return True
        if pair is not None and operation in POINT_TO_POINT:
            return (self.policy is NcclPolicy.RING and self.cabled is not None
                    and self.cabled(*pair))
        return self.policy.allows(operation)

    def check(self, operation: str, *, pair: tuple[int, int] | None = None) -> None:
        if self.allows(operation, pair=pair):
            return
        where = f" between global ranks {pair[0]} and {pair[1]}" if pair else ""
        remedy = f" {self.remedy}" if self.remedy else ""
        raise NcclAcrossRelayError(
            f"NCCL {operation}{where} on group {self.name} (global ranks {list(self.ranks)}) is "
            f"refused: {self.reason}. NCCL cannot connect Sparks that share no cable, and SIRCL "
            f"does not carry this call.{remedy} This is a configuration or dispatch error, not a "
            "transport failure to retry"
        )


def remedies(raw: NcclPolicy, effective: NcclPolicy, *, carried: bool) -> str:
    """How to let a call the policy refuses run: the text refusal messages end with."""
    options = []
    if raw is NcclPolicy.RING and effective is NcclPolicy.NONE:
        options.append("let NCCL's ring algorithm run on this cycle: SIRCL_NCCL=auto (the launchers' "
                       "--nccl auto) with NCCL_ALGO=Ring and NCCL_SKIP_TREE_CONNECT=1")
    if carried:
        options.append("issue the call through vLLM's device communicator; on this group SIRCL also carries "
                       "the torch.distributed calls " + ", ".join(CARRIED) + " and refuses other direct "
                       "torch.distributed calls")
    else:
        options.append("give the group a SIRCL session: tensor-parallel and decode-context-parallel groups get "
                       "one through SIRCL_GROUPS, and a group whose ranks equal the tensor-parallel group's "
                       "shares that group's session")
    return "Remedies: " + "; or ".join(options) + "."


NOT_CARRIED = object()   # a carrier's answer for a call it does not run
# torch.distributed calls a group with a SIRCL session carries (GroupAdapter.carry_torch).
CARRIED = ("all_reduce (sum, max, min)", "broadcast", "all_gather", "all_gather_into_tensor",
           "reduce_scatter_tensor (sum)", "all_to_all_single (equal splits)")


POINT_TO_POINT = frozenset({"send", "recv", "isend", "irecv", "batch_isend_irecv"})

_LOCK = threading.Lock()
_REGISTRY: dict[int, tuple[object, GuardedGroup]] = {}
_CARRIERS: dict[int, tuple[object, Callable[[str, Mapping[str, Any]], Any]]] = {}
_P2P_CARRIERS: dict[int, tuple[object, Callable[[str, Mapping[str, Any]], Any]]] = {}
_P2P_BY_RANKS: dict[tuple[int, ...], Callable[[str, Mapping[str, Any]], Any]] = {}
_RESOLVER: Callable[[Sequence[int]], GuardedGroup | None] | None = None


def register(group: object, entry: GuardedGroup) -> None:
    """Record what NCCL may do on ``group`` (a device process group)."""
    if group is None:
        return
    with _LOCK:
        _REGISTRY[id(group)] = (group, entry)


def unregister(group: object) -> None:
    with _LOCK:
        _REGISTRY.pop(id(group), None)
        _CARRIERS.pop(id(group), None)
        found = _P2P_CARRIERS.pop(id(group), None)
        if found is not None:
            for ranks, carrier in list(_P2P_BY_RANKS.items()):
                if carrier is found[1]:
                    del _P2P_BY_RANKS[ranks]


def register_carrier(group: object, carrier: Callable[[str, Mapping[str, Any]], Any]) -> None:
    """Let ``carrier(operation, arguments)`` run torch.distributed calls on ``group`` the policy refuses.

    ``arguments`` are the call's bound arguments with defaults applied; the
    carrier returns the call's result, or :data:`NOT_CARRIED` to let the
    refusal stand.
    """
    if group is None:
        return
    with _LOCK:
        _CARRIERS[id(group)] = (group, carrier)


def register_p2p_carrier(group: object, ranks: Sequence[int],
                         carrier: Callable[[str, Mapping[str, Any]], Any]) -> None:
    """Let ``carrier(operation, arguments)`` run point-to-point calls (and broadcasts) on ``group``, and on any
    other NCCL group without a carrier of its own whose global ranks are ``ranks`` in the same order (vLLM's
    sibling groups of the same ranks); the first group registered for a rank tuple carries for it."""
    if group is None:
        return
    with _LOCK:
        _P2P_CARRIERS[id(group)] = (group, carrier)
        _P2P_BY_RANKS.setdefault(tuple(int(r) for r in ranks), carrier)


def p2p_carrier_for(group: object) -> Callable[[str, Mapping[str, Any]], Any] | None:
    with _LOCK:
        found = _P2P_CARRIERS.get(id(group))
        if found is not None:
            return found[1]
        if not _P2P_BY_RANKS:
            return None
    if not _is_nccl(group):
        return None
    try:
        import torch.distributed as dist

        ranks = tuple(dist.get_process_group_ranks(group))
    except Exception:  # noqa: BLE001 - the group does not include this rank
        return None
    with _LOCK:
        return _P2P_BY_RANKS.get(ranks)


def carrier_for(group: object) -> Callable[[str, Mapping[str, Any]], Any] | None:
    with _LOCK:
        found = _CARRIERS.get(id(group))
    return None if found is None else found[1]


def registered() -> list[GuardedGroup]:
    with _LOCK:
        return [entry for _, entry in _REGISTRY.values()]


def configure(resolver: Callable[[Sequence[int]], GuardedGroup | None] | None) -> None:
    """Classify NCCL groups that SIRCL's communicator did not build (the default group, siblings)."""
    global _RESOLVER
    with _LOCK:
        _RESOLVER = resolver


def reset() -> None:
    """Forget every registration (tests)."""
    global _RESOLVER
    with _LOCK:
        _REGISTRY.clear()
        _CARRIERS.clear()
        _P2P_CARRIERS.clear()
        _P2P_BY_RANKS.clear()
        _RESOLVER = None


def _is_nccl(group: object) -> bool:
    """True for a group whose only backend is NCCL.

    A group with both a CPU and a CUDA backend (vLLM's split-group CPU groups,
    ``cpu:gloo,cuda:nccl``) routes CPU-object collectives to gloo, so it is not
    classified here; SIRCL registers the device groups it builds explicitly.
    """
    try:
        import torch.distributed as dist

        backend = str(dist.get_backend(group)).lower()
    except Exception:  # noqa: BLE001 - not a torch process group of this rank
        return False
    return "nccl" in backend and "gloo" not in backend


def entry_for(group: object) -> GuardedGroup | None:
    """The registered or resolved entry of ``group`` (None: not an NCCL group SIRCL knows)."""
    with _LOCK:
        found = _REGISTRY.get(id(group))
        resolver = _RESOLVER
    if found is not None:
        return found[1]
    if resolver is None or not _is_nccl(group):
        return None
    try:
        import torch.distributed as dist

        ranks = dist.get_process_group_ranks(group)
    except Exception:  # noqa: BLE001 - the group does not include this rank
        return None
    entry = resolver(tuple(ranks))
    if entry is not None:
        register(group, entry)
    return entry


# -- construction -------------------------------------------------------------------


@contextlib.contextmanager
def pynccl_suppressed(active: bool) -> Iterator[None]:
    """Build vLLM communicators without PyNccl while active (``VLLM_DISABLE_PYNCCL``)."""
    if not active:
        yield
        return
    try:
        import vllm.envs as envs

        cached = getattr(envs, "_is_envs_cache_enabled", lambda: False)()
    except ImportError:
        cached = False
    if cached:
        raise NcclAcrossRelayError(
            "a communicator is being built after vLLM cached its environment, so PyNccl and its "
            "warm-up all-reduce cannot be suppressed for a group NCCL may not connect"
        )
    saved = os.environ.get("VLLM_DISABLE_PYNCCL")
    os.environ["VLLM_DISABLE_PYNCCL"] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("VLLM_DISABLE_PYNCCL", None)
        else:
            os.environ["VLLM_DISABLE_PYNCCL"] = saved


def ring_algorithm_enforced(environ: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """Whether NCCL is limited to its ring algorithm with tree connections skipped."""
    env = os.environ if environ is None else environ
    algorithm = env.get("NCCL_ALGO", "").strip().lower()
    entries = [item.strip() for item in algorithm.split(",") if item.strip()]
    ring_only = bool(entries) and all(item == "ring" or item.endswith(":ring") for item in entries)
    skip_tree = env.get("NCCL_SKIP_TREE_CONNECT", "").strip() == "1"
    if ring_only and skip_tree:
        return True, "NCCL_ALGO=Ring and NCCL_SKIP_TREE_CONNECT=1"
    missing = []
    if not ring_only:
        missing.append(f"NCCL_ALGO={env.get('NCCL_ALGO', '') or '(unset)'} is not Ring")
    if not skip_tree:
        missing.append("NCCL_SKIP_TREE_CONNECT is not 1")
    return False, " and ".join(missing)


def effective_policy(policy: NcclPolicy, reason: str, *, nccl_mode: str = "auto",
                     environ: Mapping[str, str] | None = None) -> tuple[NcclPolicy, str]:
    """The policy the adapter applies, after the operator's mode and NCCL's settings."""
    if policy is NcclPolicy.ALL and nccl_mode != "never":
        return policy, reason
    if nccl_mode == "never":
        return NcclPolicy.NONE, "SIRCL_NCCL=never"
    if policy is NcclPolicy.RING:
        enforced, detail = ring_algorithm_enforced(environ)
        if not enforced:
            return NcclPolicy.NONE, (f"{reason}; but {detail}, so NCCL could build tree "
                                     "connections between Sparks that share no cable")
        return policy, f"{reason} ({detail})"
    return policy, reason


def environment_problems(environ: Mapping[str, str] | None = None) -> list[str]:
    """Settings that would let NCCL connect uncabled Sparks before any communicator exists."""
    env = os.environ if environ is None else environ
    problems = []
    split_group = env.get("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0").strip() not in ("", "0")
    runtime_connect = env.get("NCCL_RUNTIME_CONNECT", "").strip()
    # The NCCL mode as the adapter resolves it: unset is never. A malformed value is the settings' own
    # refusal, so it refuses nothing here.
    try:
        nccl_mode = settings.nccl_mode(env)
    except settings.SettingError:
        nccl_mode = ""
    if split_group and nccl_mode == "never":
        problems.append(
            "VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1 binds the default process group to the GPU "
            "(torch.distributed.init_process_group with device_id), which creates an NCCL communicator over "
            "every rank at startup and one per group through split_group, and SIRCL_NCCL=never (the default "
            "when it is unset) keeps NCCL off every group; leave VLLM_DISTRIBUTED_USE_SPLIT_GROUP unset or 0"
        )
    elif split_group and runtime_connect == "0":
        problems.append(
            "VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1 creates every NCCL communicator eagerly and "
            "NCCL_RUNTIME_CONNECT=0 connects each at creation, including the world group's ring "
            "between Sparks that share no cable; leave NCCL_RUNTIME_CONNECT unset or 1"
        )
    return problems


# -- tripwire ---------------------------------------------------------------------

TRIPWIRE_OPERATIONS = (
    "all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter",
    "reduce_scatter_tensor", "all_to_all", "all_to_all_single", "broadcast", "reduce",
    "gather", "scatter", "send", "recv", "isend", "irecv", "barrier", "batch_isend_irecv",
    "broadcast_object_list", "all_gather_object", "gather_object", "scatter_object_list",
    "send_object_list", "recv_object_list", "all_reduce_coalesced", "all_gather_coalesced",
)
_PEER_ARGUMENTS = {"send": "dst", "isend": "dst", "recv": "src", "irecv": "src",
                   "send_object_list": "dst", "recv_object_list": "src"}
_ORIGINALS: dict[tuple[str, str], Callable[..., Any]] = {}


def _bound_argument(signature: inspect.Signature | None, args: tuple, kwargs: dict, name: str) -> Any:
    if name in kwargs:
        return kwargs[name]
    if signature is None:
        return None
    try:
        return signature.bind_partial(*args, **kwargs).arguments.get(name)
    except TypeError:
        return None


def _default_group() -> object | None:
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return None
        from torch.distributed.distributed_c10d import _get_default_group

        return _get_default_group()
    except Exception:  # noqa: BLE001 - no default group in this process
        return None


def _check_call(operation: str, signature: inspect.Signature | None, args: tuple, kwargs: dict) -> None:
    if operation == "batch_isend_irecv":
        ops = args[0] if args else kwargs.get("p2p_op_list", ())
        for op in ops or ():
            group = getattr(op, "group", None) or _default_group()
            entry = entry_for(group) if group is not None else None
            if entry is not None:
                entry.check(operation, pair=_pair(getattr(op, "peer", None), group))
        return
    group = _bound_argument(signature, args, kwargs, "group")
    if group is None:
        group = _default_group()
    if group is None:
        return
    entry = entry_for(group)
    if entry is None:
        return
    pair = None
    if operation in _PEER_ARGUMENTS:
        pair = _pair(_bound_argument(signature, args, kwargs, _PEER_ARGUMENTS[operation]), group)
    entry.check(operation, pair=pair)


def _carry(operation: str, signature: inspect.Signature | None, args: tuple, kwargs: dict) -> Any:
    """Hand a refused call to its group's carrier, then to its point-to-point carrier; :data:`NOT_CARRIED`
    when there is none or both decline."""
    if signature is None:
        return NOT_CARRIED
    group = _bound_argument(signature, args, kwargs, "group")
    if group is None:
        group = _default_group()
    if group is None:
        return NOT_CARRIED
    carriers = [carrier for carrier in (carrier_for(group), p2p_carrier_for(group)) if carrier is not None]
    if not carriers:
        return NOT_CARRIED
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return NOT_CARRIED
    bound.apply_defaults()
    for carrier in carriers:
        carried = carrier(operation, dict(bound.arguments))
        if carried is not NOT_CARRIED:
            return carried
    return NOT_CARRIED


def _carry_p2p(operation: str, signature: inspect.Signature | None, args: tuple, kwargs: dict) -> Any:
    """Point-to-point calls go to the group's point-to-point carrier first, whatever NCCL may do there."""
    if signature is None:
        return NOT_CARRIED
    if operation == "batch_isend_irecv":
        ops = args[0] if args else kwargs.get("p2p_op_list", ())
        group = next((getattr(op, "group", None) for op in ops or ()), None)
    else:
        group = _bound_argument(signature, args, kwargs, "group")
    if group is None:
        group = _default_group()
    carrier = p2p_carrier_for(group) if group is not None else None
    if carrier is None:
        return NOT_CARRIED
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return NOT_CARRIED
    bound.apply_defaults()
    return carrier(operation, dict(bound.arguments))


def _pair(peer: Any, group: object) -> tuple[int, int] | None:
    if peer is None:
        return None
    try:
        import torch.distributed as dist

        return dist.get_rank(), int(peer)
    except Exception:  # noqa: BLE001
        return None


def _wrap(operation: str, original: Callable[..., Any]) -> Callable[..., Any]:
    try:
        signature: inspect.Signature | None = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if operation in POINT_TO_POINT:
            carried = _carry_p2p(operation, signature, args, kwargs)
            if carried is not NOT_CARRIED:
                return carried
        try:
            _check_call(operation, signature, args, kwargs)
        except NcclAcrossRelayError:
            carried = _carry(operation, signature, args, kwargs)
            if carried is NOT_CARRIED:
                raise
            return carried
        return original(*args, **kwargs)

    guarded._sircl_original = original  # type: ignore[attr-defined]
    return guarded


def install_tripwire() -> list[str]:
    """Wrap the ``torch.distributed`` collectives (idempotent); returns the wrapped names."""
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    wrapped = []
    with _LOCK:
        for module in (dist, c10d):
            for name in TRIPWIRE_OPERATIONS:
                current = getattr(module, name, None)
                if current is None or getattr(current, "_sircl_original", None) is not None:
                    continue
                key = (module.__name__, name)
                _ORIGINALS[key] = current
                setattr(module, name, _wrap(name, current))
                wrapped.append(f"{module.__name__}.{name}")
    return wrapped


def uninstall_tripwire() -> None:
    import importlib

    with _LOCK:
        for (module_name, name), original in list(_ORIGINALS.items()):
            setattr(importlib.import_module(module_name), name, original)
            del _ORIGINALS[(module_name, name)]


def tripwire_installed() -> bool:
    with _LOCK:
        return bool(_ORIGINALS)
