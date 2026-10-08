"""Rank-invariant choice of how one collective of a vLLM group is carried.

For every collective call the adapter asks :func:`plan` for a :class:`Plan`:
which backend serves it (the group's SIRCL ring session, NCCL, or nobody) and
how (one session op, several session ops over pieces of the message, or a
composition of session ops). The answer depends only on facts every rank of
the group shares:

- the group's cabling (:class:`.fabric.GroupTopology`): whether NCCL may run
  at all, only its ring algorithm, or anything;
- the session's agreed limits (:class:`SessionLimits`): all-reduce capacity
  ``C``, dispatch ceiling ``D``, all-gather capacity ``G``, prepared dtypes,
  scatter availability and the relay-safe per-peer op size (:mod:`.fabric`);
- the call: collective, dtype, shape, contiguity, byte size, dimension,
  per-rank sizes, source or destination rank;
- whether a CUDA graph is being captured.

It never looks at pointer values or timing, so all ranks of a group take the
same plan for the same call and never mix backends within one collective.

Rules, in order:

1. A collective NCCL may not run on this group (:meth:`.fabric.NcclPolicy.allows`)
   never gets an NCCL plan; if SIRCL cannot carry it either the plan is
   ``refuse`` and the adapter raises instead of falling back.
2. Messages within the session's single-op limits are one session op.
3. Larger messages go to NCCL only when the group's cabling allows that
   collective, the call is eager (captured calls stay on SIRCL, so a graph
   replays the transport it was captured with) and ``SIRCL_LARGE_ALLREDUCE``
   is ``auto`` or ``nccl``. Under ``auto``, a session with a tuning table
   decides every eager call the table covers instead (``Policy.tuned``):
   NCCL where the table measured NCCL faster than every SIRCL candidate,
   SIRCL elsewhere, at any size. Otherwise a session with the large-message
   operations (``all_reduce_large``, ``all_gather_large``) carries them in ops
   it chooses itself (method ``large``: pieces of ``large_piece_bytes`` and
   ``gather_piece_bytes``, recorded in ``SessionLimits.large_piece`` and
   ``gather_piece``, or one chain op where the session chains), and the
   adapter splits them on the host for a session without them (methods
   ``chunked``, ``rows``, ``tiles``). Every rank gets identical bits either
   way: a piece is an independent rank-ordered float32 sum rounded once, and
   a session's chain op may differ from that sum in the last place, the same
   on every rank.
4. A reduce-scatter goes to the session's reduce-scatter when its dtype is
   prepared, and an all-to-all to the session's all-to-all when it is
   prepared: in one call on the whole message when the session states the op
   size it cuts messages into (``scatter_op_bytes``, recorded in
   ``SessionLimits.scatter_piece``), else in strided calls of at most
   ``SessionLimits.scatter_op_bytes``.
5. Point-to-point calls (send, receive, batched send/receive) go to the
   group's point-to-point channels wherever the group has a channel to the
   peer (:class:`P2PLimits`), whatever NCCL may do there; captured calls on
   such a pair are refused (the channels run outside CUDA graph capture). A
   pair without a channel goes to NCCL only where NCCL may connect it (a
   cabled pair of an ``all`` or ``ring`` group); otherwise it is refused with
   the reason the pair has no channel.
6. Shapes a session op does not accept are composed from session ops with
   rank-identical results: an all-reduce of a dtype the session does not sum
   is an all-gather followed by a rank-ordered local sum; a reduce-scatter
   without a prepared scatter kernel is an all-reduce followed by this rank's
   slice of it; a broadcast or gather is an all-gather of bytes; uneven
   gathers are padded. A composed reduce-scatter may differ in the last place
   from the session's reduce-scatter when the session's all-reduce chains
   (rounding per hop); every rank still holds the same bits.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

from ..pieces import gather_plan as _gather_ops
from .fabric import GroupTopology, NcclPolicy

PACK = 16
SUMMED_DTYPES = ("float16", "bfloat16", "float32")
EXACT_LOCAL_SUM_DTYPES = SUMMED_DTYPES + ("int8", "uint8", "int16", "int32", "int64", "float64")
BYTE_VIEW_DTYPES = ("bool", "complex64", "complex128", "float8_e4m3fn", "float8_e5m2")

SIRCL = "sircl"
NCCL = "nccl"
REFUSE = "refuse"


@dataclasses.dataclass(frozen=True)
class TensorMeta:
    """The rank-invariant facts of one tensor argument."""

    shape: tuple[int, ...]
    dtype: str
    itemsize: int
    contiguous: bool = True

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @classmethod
    def of(cls, tensor) -> "TensorMeta":
        return cls(tuple(int(s) for s in tensor.shape), str(tensor.dtype).replace("torch.", ""),
                   int(tensor.element_size()), bool(tensor.is_contiguous()))


@dataclasses.dataclass(frozen=True)
class SessionLimits:
    """Agreed limits of one group's ring session: sizes, prepared dtypes, relay-safe and piece sizes."""

    world: int
    capacity: int                       # C: largest all-reduce one op carries
    dispatch: int                       # D: largest all-reduce the adapter sends as one op
    gather: int                         # G: largest all-gather shard of one op (0: none)
    reduce_dtypes: tuple[str, ...]      # all-reduce dtypes prepared before any capture
    scatter_dtypes: tuple[str, ...] = ()  # reduce-scatter dtypes prepared (empty: no scatter op)
    all_to_all: bool = False            # the byte-copy all-to-all launcher is prepared
    per_peer_op_bytes: int | None = None  # relay-safe bytes per peer per op (None: no relays)
    scatter_op_override: int | None = None  # whole-message bytes of one scatter op, when set
    large_piece: int | None = None      # piece of the session's all_reduce_large (None: no such method)
    gather_piece: int | None = None     # piece of the session's all_gather_large (None: no such method)
    scatter_piece: int | None = None    # op size the session cuts a whole reduce-scatter into (None: one op a call)

    @property
    def gather_op_bytes(self) -> int:
        """Largest shard of one all-gather op (relay load included)."""
        if self.per_peer_op_bytes is None:
            return self.gather
        return min(self.gather, self.per_peer_op_bytes)

    @property
    def scatter_op_bytes(self) -> int:
        """Largest whole message of one reduce-scatter or all-to-all op."""
        limit = self.capacity
        if self.scatter_op_override is not None:
            limit = min(limit, self.scatter_op_override)
        elif self.per_peer_op_bytes is not None:
            limit = min(limit, self.per_peer_op_bytes * self.world)
        return limit // (self.world * PACK) * (self.world * PACK)

    @classmethod
    def of(cls, session, *, reduce_dtypes: Sequence[str], scatter_dtypes: Sequence[str] = (),
           all_to_all: bool = False, per_peer_op_bytes: int | None = None,
           gather: int | None = None, scatter_op: int | None = None) -> "SessionLimits":
        return cls(
            world=int(session.world_size),
            capacity=int(session.max_size),
            dispatch=int(session.dispatch_limit_bytes),
            gather=int(session.max_gather_bytes if gather is None else gather),
            reduce_dtypes=tuple(reduce_dtypes),
            scatter_dtypes=tuple(scatter_dtypes),
            all_to_all=all_to_all,
            per_peer_op_bytes=per_peer_op_bytes,
            scatter_op_override=scatter_op,
            large_piece=_session_piece(session, "all_reduce_large", "large_piece_bytes"),
            gather_piece=_session_piece(session, "all_gather_large", "gather_piece_bytes"),
            scatter_piece=_scatter_piece(session) if scatter_dtypes else None,
        )


def _scatter_piece(session) -> int | None:
    """The op size a session cuts a whole reduce-scatter into, when it states one.

    Read from the attribute ``scatter_op_bytes`` or, failing that, the same key
    of ``stats()`` (read once, when the group is set up).
    """
    piece = getattr(session, "scatter_op_bytes", None)
    if piece is None and callable(getattr(session, "stats", None)):
        try:
            piece = session.stats().get("scatter_op_bytes")
        except Exception:  # noqa: BLE001 - a session without statistics states no op size
            piece = None
    return int(piece) if isinstance(piece, int) and not isinstance(piece, bool) and piece >= PACK else None


def _session_piece(session, method: str, attribute: str) -> int | None:
    """The session's piece size for ``method``, when it offers the method and states its piece."""
    if not callable(getattr(session, method, None)):
        return None
    piece = getattr(session, attribute, None)
    return int(piece) if isinstance(piece, int) and piece >= PACK else None


@dataclasses.dataclass(frozen=True)
class Plan:
    """How one collective call is carried."""

    collective: str
    backend: str          # sircl, nccl or refuse
    method: str
    ops: int = 0          # session ops issued (0 for nccl and refuse)
    piece: int = 0        # bytes of one piece (rows for gathers) when split
    reason: str = ""
    contiguous_copy: bool = False

    def key(self) -> tuple[str, str, str]:
        return (self.collective, self.backend, self.method)


@dataclasses.dataclass(frozen=True)
class Policy:
    """The group-level inputs of every plan."""

    topology: GroupTopology | None   # None: a single-rank group
    nccl_mode: str = "auto"            # auto (topology is another name for it) or never
    large: str = "auto"                # auto, sircl or nccl
    # Eager calls above these bytes prefer NCCL where the cabling allows it
    # (all_reduce: whole message, all_gather: one shard, reduce_scatter and
    # all_to_all: whole message). A missing entry means "above one session op";
    # 0 means never. The DCP collectives set the measured crossovers.
    nccl_above: tuple[tuple[str, int], ...] = ()
    # Effective NCCL policy when it differs from the cabling's (guard.effective_policy).
    policy_override: NcclPolicy | None = None
    # Why NCCL may or may not run here, as stated in refusals.
    reason: str = ""
    # Whether two group ranks share a cable (point-to-point admission).
    cabled: Callable[[int, int], bool] | None = dataclasses.field(default=None, compare=False)
    # The session's tuning table for eager calls (sessionapi.tuned_backend): nccl, sircl, or None where the
    # table decides nothing. Consulted under large == "auto" where NCCL may run the collective.
    tuned: Callable[[str, int], str | None] | None = dataclasses.field(default=None, compare=False)

    def nccl_allows(self, operation: str) -> bool:
        if self.topology is None and self.policy_override is None:
            return True
        if self.nccl_mode == "never":
            return False
        return self.nccl_policy.allows(operation)

    @property
    def nccl_policy(self) -> NcclPolicy:
        if self.policy_override is not None:
            return self.policy_override
        return NcclPolicy.ALL if self.topology is None else self.topology.nccl_policy

    def prefers_nccl(self, collective: str, nbytes: int, single_op_limit: int, *,
                     capturing: bool) -> bool:
        """Eager call above its threshold, on a group whose cabling carries the NCCL collective."""
        if capturing or self.large == "sircl" or not self.nccl_allows(collective):
            return False
        if self.tuned is not None and self.large == "auto":
            verdict = self.tuned(collective, nbytes)
            if verdict is not None:
                return verdict == "nccl"
        threshold = dict(self.nccl_above).get(collective)
        if threshold is None:
            return nbytes > single_op_limit
        return threshold > 0 and nbytes > threshold

    def internal(self) -> "Policy":
        """The policy of a session op that is part of a composed collective (never NCCL)."""
        return Policy(self.topology, "never", "sircl", (), self.policy_override, self.reason,
                      self.cabled)

    def pair_cabled(self, rank: int, peer: int) -> bool:
        if self.cabled is not None:
            return bool(self.cabled(rank, peer))
        return self.topology is not None and self.topology.pair_cabled(rank, peer)


@dataclasses.dataclass(frozen=True)
class P2PLimits:
    """This rank's point-to-point channels in a group: the peers it has a channel with and why others have none."""

    peers: frozenset[int]                    # group ranks this rank has a channel with
    relayed: frozenset[int] = frozenset()    # of them, those whose lanes cross relays
    problems: tuple[tuple[int, str], ...] = ()   # peers without a channel, with the reason
    reason: str = ""                         # why the group has no channels at all ("" when it has them)

    def problem(self, peer: int) -> str:
        if self.reason:
            return self.reason
        for other, why in self.problems:
            if other == peer:
                return f"the point-to-point channel to rank {peer} is unavailable: {why}"
        return f"rank {peer} has no point-to-point channel with this rank"


def _refuse(collective: str, why: str) -> Plan:
    return Plan(collective, REFUSE, "refuse", reason=why)


def _nccl(collective: str, why: str) -> Plan:
    return Plan(collective, NCCL, "nccl", reason=why)


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _forbidden(policy: Policy, operation: str) -> str:
    if policy.nccl_mode == "never":
        return "SIRCL_NCCL=never"
    reason = policy.reason or (policy.topology.nccl_reason if policy.topology is not None else "")
    return f"NCCL may not run {operation} on this group ({policy.nccl_policy.value}): {reason}"


def plan_all_reduce(meta: TensorMeta, limits: SessionLimits | None, policy: Policy,
                    *, capturing: bool) -> Plan:
    name = "all_reduce"
    if meta.nbytes == 0:
        return Plan(name, SIRCL, "empty", reason="empty tensor")
    copy = not meta.contiguous
    if limits is None:
        if policy.nccl_allows(name):
            return _nccl(name, "the group has no SIRCL session")
        return _refuse(name, "the group has no SIRCL session and " + _forbidden(policy, name))
    if meta.dtype not in limits.reduce_dtypes:
        if policy.nccl_allows(name) and policy.large != "sircl":
            return _nccl(name, f"dtype {meta.dtype} is not summed by the prepared session")
        if meta.dtype not in EXACT_LOCAL_SUM_DTYPES:
            return _refuse(name, f"dtype {meta.dtype} has no exact rank-ordered sum; "
                                 + _forbidden(policy, name))
        gather = plan_all_gather(TensorMeta((meta.numel,), meta.dtype, meta.itemsize), 0,
                                 limits, policy.internal(),
                                 capturing=capturing)
        if gather.backend != SIRCL:
            return _refuse(name, f"dtype {meta.dtype}: the gather it is composed of is refused: "
                                 f"{gather.reason}")
        return Plan(name, SIRCL, "gather_sum", gather.ops, gather.piece,
                    f"dtype {meta.dtype} is summed locally after an all-gather", copy)
    padded = _ceil(meta.nbytes, PACK) * PACK
    if padded <= limits.dispatch and not policy.prefers_nccl(name, meta.nbytes, limits.dispatch,
                                                             capturing=capturing):
        method = "direct" if padded == meta.nbytes else "padded"
        return Plan(name, SIRCL, method, 1, padded, "within the dispatch ceiling", copy)
    if policy.prefers_nccl(name, meta.nbytes, limits.dispatch, capturing=capturing):
        return _nccl(name, f"{meta.nbytes} bytes exceed the dispatch ceiling {limits.dispatch} "
                           "on a group whose cabling carries NCCL's ring")
    if policy.large == "nccl" and not capturing:
        return _refuse(name, "SIRCL_LARGE_ALLREDUCE=nccl but " + _forbidden(policy, name))
    if limits.large_piece:
        ops = _ceil(meta.nbytes, limits.large_piece)
        return Plan(name, SIRCL, "large", ops, limits.large_piece,
                    f"{meta.nbytes} bytes through the session's all_reduce_large (at most {ops} op(s) of "
                    f"at most {limits.large_piece} bytes, or one chain op)", copy)
    piece = limits.dispatch // max(PACK, meta.itemsize) * max(PACK, meta.itemsize)
    piece = piece // PACK * PACK
    if piece < PACK:
        return _refuse(name, f"dispatch ceiling {limits.dispatch} holds no whole element")
    ops = _ceil(padded, piece)
    return Plan(name, SIRCL, "chunked", ops, piece,
                f"{meta.nbytes} bytes in {ops} ops of at most {piece} bytes", copy)


def _gather_view(meta: TensorMeta, dim: int) -> tuple[int, int, int]:
    """``(outer, inner, element bytes)`` of the ``[outer, inner]`` view gathered on its last dim."""
    dim = dim % len(meta.shape)
    outer = math.prod(meta.shape[:dim])
    inner = math.prod(meta.shape[dim:])
    return outer, inner, meta.itemsize


def plan_all_gather(meta: TensorMeta, dim: int, limits: SessionLimits | None, policy: Policy,
                    *, capturing: bool) -> Plan:
    name = "all_gather"
    if not meta.shape:
        return _refuse(name, "a scalar has no dimension to gather along")
    if meta.nbytes == 0:
        return Plan(name, SIRCL, "empty", reason="empty shard")
    copy = not meta.contiguous
    if limits is None or limits.gather <= 0:
        why = "the group has no SIRCL session" if limits is None else "SIRCL all-gather is disabled (G=0)"
        if policy.nccl_allows(name):
            return _nccl(name, why)
        return _refuse(name, why + " and " + _forbidden(policy, name))
    op_bytes = limits.gather_op_bytes
    outer, inner, item = _gather_view(meta, dim)
    method_prefix = "bytes_" if meta.dtype in BYTE_VIEW_DTYPES else ""
    if policy.prefers_nccl(name, meta.nbytes, op_bytes, capturing=capturing):
        return _nccl(name, f"{meta.nbytes}-byte shard is above the group's SIRCL gather threshold "
                           "on a group whose cabling carries NCCL's ring")
    if meta.nbytes <= op_bytes:
        return Plan(name, SIRCL, method_prefix + "direct", 1, meta.nbytes, "within the gather op limit", copy)
    row_bytes = inner * item
    if limits.gather_piece:
        ops = len(_gather_ops(outer, row_bytes, limits.gather_piece))
        return Plan(name, SIRCL, method_prefix + "large", ops, limits.gather_piece,
                    f"{outer} rows of {row_bytes} bytes in {ops} session op(s) of at most {limits.gather_piece} "
                    "bytes (all_gather_large)", copy)
    if row_bytes <= op_bytes:
        rows = op_bytes // row_bytes
        return Plan(name, SIRCL, method_prefix + "rows", _ceil(outer, rows), rows,
                    f"{outer} rows of {row_bytes} bytes, {rows} per op", copy)
    columns = max(1, op_bytes // item)
    ops = outer * _ceil(inner, columns)
    return Plan(name, SIRCL, method_prefix + "tiles", ops, columns,
                f"rows of {row_bytes} bytes exceed the op limit {op_bytes}: tiles of {columns} "
                "elements", copy)


def plan_reduce_scatter(meta: TensorMeta, dim: int, limits: SessionLimits | None, policy: Policy,
                        *, capturing: bool) -> Plan:
    name = "reduce_scatter"
    if not meta.shape:
        return _refuse(name, "a scalar cannot be scattered")
    world = limits.world if limits is not None else None
    extent = meta.shape[dim % len(meta.shape)]
    if world is not None and extent % world:
        return _refuse(name, f"dimension {dim} of extent {extent} does not split into {world} "
                             "equal chunks")
    if meta.nbytes == 0:
        return Plan(name, SIRCL, "empty", reason="empty tensor")
    if limits is None:
        if policy.nccl_allows(name):
            return _nccl(name, "the group has no SIRCL session")
        return _refuse(name, "the group has no SIRCL session and " + _forbidden(policy, name))
    if policy.prefers_nccl(name, meta.nbytes, limits.scatter_op_bytes, capturing=capturing):
        return _nccl(name, f"{meta.nbytes} bytes are above the group's SIRCL reduce-scatter "
                           "threshold on a group whose cabling carries NCCL's ring")
    rows = extent
    row_bytes = meta.nbytes // rows
    if meta.dtype in limits.scatter_dtypes and row_bytes % PACK == 0 and limits.scatter_piece:
        return Plan(name, SIRCL, "scatter", 1, meta.nbytes,
                    f"session reduce-scatter of the whole message (the session cuts ops of at most "
                    f"{limits.scatter_piece} bytes)", not meta.contiguous or dim % len(meta.shape) != 0)
    if meta.dtype in limits.scatter_dtypes and row_bytes % PACK == 0 and limits.scatter_op_bytes:
        heads = rows // limits.world
        per_op_chunk = limits.scatter_op_bytes // limits.world
        block = heads * row_bytes
        ops = 1 if meta.nbytes <= limits.scatter_op_bytes else _ceil(block, per_op_chunk)
        return Plan(name, SIRCL, "scatter", ops, min(block, per_op_chunk),
                    "session reduce-scatter" + ("" if ops == 1 else f" in {ops} strided ops"),
                    not meta.contiguous or dim % len(meta.shape) != 0)
    reduce = plan_all_reduce(meta, limits, policy.internal(),
                             capturing=capturing)
    if reduce.backend != SIRCL:
        if policy.nccl_allows(name):
            return _nccl(name, "SIRCL cannot reduce this tensor: " + reduce.reason)
        return _refuse(name, reduce.reason)
    return Plan(name, SIRCL, "allreduce_slice", reduce.ops, reduce.piece,
                "all-reduce then this rank's chunk of it", reduce.contiguous_copy)


def plan_all_gatherv(meta: TensorMeta, sizes: Sequence[int] | None, limits: SessionLimits | None,
                     policy: Policy, *, capturing: bool) -> Plan:
    if sizes is None or len(set(sizes)) == 1:
        inner = plan_all_gather(meta, 0, limits, policy, capturing=capturing)
        return dataclasses.replace(inner, collective="all_gatherv")
    longest = max(sizes)
    padded = TensorMeta((longest,) + meta.shape[1:], meta.dtype, meta.itemsize)
    inner = plan_all_gather(padded, 0, limits, policy, capturing=capturing)
    if inner.backend == NCCL:
        return dataclasses.replace(inner, collective="all_gatherv")
    if inner.backend == REFUSE:
        return dataclasses.replace(inner, collective="all_gatherv")
    return Plan("all_gatherv", SIRCL, "padded_" + inner.method, inner.ops, inner.piece,
                f"uneven sizes {list(sizes)} padded to {longest}", True)


def plan_reduce_scatterv(meta: TensorMeta, sizes: Sequence[int] | None, limits: SessionLimits | None,
                         policy: Policy, *, capturing: bool) -> Plan:
    if sizes is None or len(set(sizes)) == 1:
        inner = plan_reduce_scatter(meta, 0, limits, policy, capturing=capturing)
        return dataclasses.replace(inner, collective="reduce_scatterv")
    if policy.nccl_allows("reduce_scatter") and not capturing and policy.large != "sircl":
        return _nccl("reduce_scatterv", "uneven sizes on a group whose cabling carries NCCL's ring")
    reduce = plan_all_reduce(meta, limits, policy.internal(),
                             capturing=capturing)
    if reduce.backend != SIRCL:
        return Plan("reduce_scatterv", REFUSE, "refuse", reason=reduce.reason)
    return Plan("reduce_scatterv", SIRCL, "allreduce_slice", reduce.ops, reduce.piece,
                f"uneven sizes {list(sizes)}: all-reduce then this rank's rows", reduce.contiguous_copy)


def plan_bytes_gather(collective: str, meta: TensorMeta, limits: SessionLimits | None,
                      policy: Policy, *, capturing: bool, nccl_operation: str) -> Plan:
    """Broadcast and gather as an all-gather of the tensor's bytes."""
    if meta.nbytes == 0:
        return Plan(collective, SIRCL, "empty", reason="empty tensor")
    if policy.nccl_allows(nccl_operation) and (limits is None or (not capturing and policy.large != "sircl")):
        return _nccl(collective, "the group's cabling carries this NCCL collective")
    flat = TensorMeta((meta.nbytes,), "uint8", 1)
    inner = plan_all_gather(flat, 0, limits, policy.internal(),
                            capturing=capturing)
    if inner.backend != SIRCL:
        return _refuse(collective, inner.reason)
    return Plan(collective, SIRCL, "gather_bytes", inner.ops, inner.piece,
                "all-gather of the bytes, then the source rank's copy", not meta.contiguous)


def plan_all_to_all(meta: TensorMeta, limits: SessionLimits | None, policy: Policy,
                    *, capturing: bool) -> Plan:
    name = "all_to_all"
    if meta.nbytes == 0:
        return Plan(name, SIRCL, "empty", reason="empty tensor")
    if limits is None:
        if policy.nccl_allows(name):
            return _nccl(name, "the group has no SIRCL session")
        return _refuse(name, "the group has no SIRCL session and " + _forbidden(policy, name))
    world = limits.world
    if policy.prefers_nccl(name, meta.nbytes, limits.scatter_op_bytes, capturing=capturing):
        return _nccl(name, "above the group's SIRCL all-to-all threshold on a group whose cabling "
                           "connects every pair of ranks")
    if meta.nbytes % (world * PACK) == 0 and limits.all_to_all and limits.scatter_piece:
        return Plan(name, SIRCL, "scatter", 1, meta.nbytes,
                    f"session all-to-all of the whole message (the session cuts ops of at most "
                    f"{limits.scatter_piece} bytes)", not meta.contiguous)
    if meta.nbytes % (world * PACK) == 0 and limits.all_to_all and limits.scatter_op_bytes:
        ops = max(1, _ceil(meta.nbytes, limits.scatter_op_bytes))
        return Plan(name, SIRCL, "scatter", ops, min(meta.nbytes, limits.scatter_op_bytes),
                    "session all-to-all", not meta.contiguous)
    if policy.nccl_allows(name) and not capturing:
        return _nccl(name, "the group's cabling connects every pair of ranks")
    flat = TensorMeta((meta.nbytes,), "uint8", 1)
    inner = plan_all_gather(flat, 0, limits, policy.internal(),
                            capturing=capturing)
    if inner.backend != SIRCL:
        return _refuse(name, inner.reason)
    return Plan(name, SIRCL, "gather_pick", inner.ops, inner.piece,
                "all-gather of every rank's send buffer, then this rank's chunks", not meta.contiguous)


def plan_point_to_point(collective: str, policy: Policy, rank: int, peer: int, *, p2p: P2PLimits | None = None,
                        capturing: bool = False, nbytes: int = 0) -> Plan:
    """A send or receive between group ranks ``rank`` and ``peer`` (rule 5 of the module docstring)."""
    if p2p is not None and peer in p2p.peers:
        if capturing:
            return _refuse(collective, f"point-to-point transfers to rank {peer} run on SIRCL's channels, which "
                                       "work outside CUDA graph capture only")
        method = "relayed" if peer in p2p.relayed else "direct"
        return Plan(collective, SIRCL, method, 1, nbytes,
                    f"the group's point-to-point channel between ranks {rank} and {peer} ({method})")
    missing = ("the group has no point-to-point channels (SIRCL_P2P_GROUPS)" if p2p is None
               else p2p.problem(peer))
    if policy.nccl_mode == "never":
        return _refuse(collective, f"{missing}, and SIRCL_NCCL=never")
    if policy.nccl_policy is NcclPolicy.NONE:
        return _refuse(collective, f"{missing}, and " + _forbidden(policy, collective))
    if policy.nccl_policy is NcclPolicy.ALL or policy.pair_cabled(rank, peer):
        return _nccl(collective, f"ranks {rank} and {peer} share a cable and {missing}")
    return _refuse(collective, f"ranks {rank} and {peer} share no cable, and {missing}")
