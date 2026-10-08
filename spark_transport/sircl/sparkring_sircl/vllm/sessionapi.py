"""The ring-session interface the vLLM adapter calls, and how it is found.

The adapter does not implement transport. It drives one ring session per
SIRCL-enabled vLLM group through the session package's surface: the package
exports ``API_VERSION``, ``is_supported(device)`` and the session class
``AllReduce``; sessions are constructed collectively over the group's CPU
process group and offer the methods below. :class:`RingSession` lists exactly
the part of that surface the adapter uses, so a session implementation can be
checked against it and the adapter's CPU tests can substitute
:class:`.emulation.EmulatedRingSession`.

The package name is configurable (``SIRCL_SESSION_MODULE``, default
``sparkring_sircl.oneshot``). The default names the package by its full
name only, so an unrelated top-level ``sircl`` module is never imported.

Point-to-point channels come from a package of their own
(``SIRCL_P2P_MODULE``, default ``sparkring_sircl.p2p``) that exports
``API_VERSION``, ``is_supported()`` and the class ``PointToPoint``;
:class:`PointToPointChannels` lists the part the adapter uses and
:func:`load_p2p_module` finds it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from . import settings

REQUIRED_API_VERSION = 1


class SessionUnavailable(RuntimeError):
    """No importable ring-session package with the required API version."""


@runtime_checkable
class RingSession(Protocol):
    """The part of one rank's ring session that the adapter uses."""

    rank: int
    world_size: int
    max_size: int                 # all-reduce capacity C
    dispatch_limit_bytes: int     # dispatch ceiling D
    max_gather_bytes: int         # all-gather capacity G
    lane_count: int
    hca_names: Sequence[str]
    scatter_available: bool

    @property
    def poisoned(self) -> bool: ...

    def should_allreduce(self, inp: Any) -> bool: ...

    def all_reduce(self, inp: Any, *, out: Any = None, stream: Any = None) -> Any: ...

    def should_all_gather(self, inp: Any, dim: int = -1) -> bool: ...

    def all_gather(self, inp: Any, *, dim: int = -1, out: Any = None, stream: Any = None) -> Any: ...

    def should_reduce_scatter(self, inp: Any, *, chunk_bytes: int | None = None,
                              src_stride_bytes: int | None = None) -> bool: ...

    def reduce_scatter(self, inp: Any, *, out: Any = None, stream: Any = None,
                       chunk_bytes: int | None = None, src_stride_bytes: int | None = None) -> Any: ...

    def should_all_to_all(self, inp: Any, *, chunk_bytes: int | None = None,
                          src_stride_bytes: int | None = None) -> bool: ...

    def all_to_all(self, inp: Any, out: Any, *, stream: Any = None, chunk_bytes: int | None = None,
                   src_stride_bytes: int | None = None, dst_stride_bytes: int | None = None) -> Any: ...

    def prepare(self, dtypes: Sequence[Any] = ..., *, padded_gather: bool = False,
                algorithms: Sequence[str] | None = None, scatter: bool = False, links: bool = False) -> None: ...

    def capture(self, stream: Any = None, *, channel_id: Any = None) -> Any: ...

    def check_health(self) -> None: ...

    def stats(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class LargeMessageSession(Protocol):
    """Optional session operations for messages above one session op.

    Without them the adapter splits a message into pieces of at most the
    dispatch ceiling (all-reduce) or the relay-safe gather op size (all-gather)
    and issues one session op per piece from the host
    (:func:`.executor.all_reduce`, :func:`.executor.all_gather`), which is the
    reference behaviour. A session that implements these keeps several pieces
    in flight on its own and must give rank-identical bits: a piece is an
    independent rank-ordered float32 sum rounded once (a chain op may differ
    from that sum in the last place, the same on every rank), and gathers copy
    bytes. Both must be capturable in a CUDA graph on one stream, with
    everything compiled before the capture.
    """

    large_piece_bytes: int        # op size of all_reduce_large
    gather_piece_bytes: int       # op size of all_gather_large

    def all_reduce_large(self, inp: Any, *, out: Any = None, stream: Any = None) -> Any: ...

    def all_gather_large(self, inp: Any, *, dim: int = -1, out: Any = None, stream: Any = None) -> Any: ...


@runtime_checkable
class RegimeSession(Protocol):
    """Optional flag-wait regimes: how long a session's kernels wait for a late peer.

    ``startup`` allows minutes (compilation, warm-up, graph capture);
    ``serving`` allows seconds. The limit is read by every launch, eager or
    replayed from a graph, so a regime change applies to captured graphs
    too. The adapter selects serving after the first completed step and
    startup around worker methods that can make a rank late
    (:mod:`.adapter`).
    """

    wait_regime: str              # "startup" or "serving"
    startup_wait_s: float
    serving_wait_s: float

    @property
    def wait_limit_s(self) -> float: ...

    def enter_startup(self) -> None: ...

    def enter_serving(self) -> None: ...

    def startup(self) -> Any: ...


@runtime_checkable
class PointToPointChannels(Protocol):
    """The part of one rank's point-to-point channels that the adapter uses (:mod:`sparkring_sircl.p2p`).

    Built collectively over the group's CPU process group with
    ``PointToPoint(exchange_group=, device=, peer_routes=, layout=, windows=, unavailable=)``.
    ``isend``/``irecv`` return work objects whose ``wait()`` orders the caller's current stream after the
    transfer; ``send``/``recv`` wait at once; ``batch_isend_irecv`` takes ``(kind, tensor, peer)`` ops.
    """

    rank: int
    world_size: int
    wait_regime: str

    @property
    def poisoned(self) -> bool: ...

    def has_channel(self, peer: int) -> bool: ...

    def channel_problem(self, peer: int) -> str | None: ...

    def isend(self, tensor: Any, peer: int) -> Any: ...

    def irecv(self, tensor: Any, peer: int) -> Any: ...

    def send(self, tensor: Any, peer: int) -> None: ...

    def recv(self, tensor: Any, peer: int) -> Any: ...

    def batch_isend_irecv(self, ops: Sequence[tuple[str, Any, int]]) -> list: ...

    def prepare(self) -> None: ...

    def enter_startup(self) -> None: ...

    def enter_serving(self) -> None: ...

    def check_health(self) -> None: ...

    def stats(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


REQUIRED_P2P_API_VERSION = 1


def load_p2p_module(name: str | None = None) -> ModuleType:
    """The point-to-point package (``SIRCL_P2P_MODULE``) when its API version fits and it runs here.

    Raises :class:`SessionUnavailable` naming the module and why it is not used.
    """
    name = name or settings.p2p_module()
    try:
        module = importlib.import_module(name)
        api = getattr(module, "API_VERSION", None)
        supported = getattr(module, "is_supported", None)
        if api != REQUIRED_P2P_API_VERSION:
            raise SessionUnavailable(f"{name}: API version {api}, the adapter needs {REQUIRED_P2P_API_VERSION}")
        if not callable(supported) or not supported():
            raise SessionUnavailable(f"{name}: is_supported() is missing or False on this host")
        # Reading the class may import torch, CUDA Python and the CuTe DSL (a lazy export).
        channels = getattr(module, "PointToPoint", None)
    except SessionUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken build counts as unavailable
        raise SessionUnavailable(f"{name}: {type(exc).__name__}: {exc}") from None
    if channels is None:
        raise SessionUnavailable(f"{name}: no PointToPoint class")
    return module


def load(modules: Sequence[str] | None = None) -> ModuleType:
    """The first importable session package whose ``API_VERSION`` is the one the adapter needs.

    Raises :class:`SessionUnavailable` naming every module tried and why it
    was not used.
    """
    reasons = []
    for name in modules or settings.session_modules():
        try:
            # Packages may resolve their exports lazily (module __getattr__), so
            # reading them can import the runtime and fail like the import itself.
            module = importlib.import_module(name)
            api = getattr(module, "API_VERSION", None)
            session_class = getattr(module, "AllReduce", None)
            supported = getattr(module, "is_supported", None)
        except Exception as exc:  # noqa: BLE001 - a broken native build counts as unavailable
            reasons.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if api != REQUIRED_API_VERSION:
            reasons.append(f"{name}: API version {api}, the adapter needs {REQUIRED_API_VERSION}")
            continue
        if session_class is None or not callable(supported):
            reasons.append(f"{name}: no AllReduce class or is_supported()")
            continue
        return module
    raise SessionUnavailable("no ring-session package is usable: " + "; ".join(reasons))


def missing_methods(session: object) -> list[str]:
    """Names of the :class:`RingSession` members ``session`` lacks."""
    names = [name for name in dir(RingSession) if not name.startswith("_")]
    return sorted(name for name in names if not hasattr(session, name))


Factory = Callable[..., RingSession]


# Session attributes naming the schedules of all_reduce_large, all_gather_large and reduce_scatter
# (SIRCL_LARGE_SCHEDULE, SIRCL_GATHER_SCHEDULE, SIRCL_SCATTER_SCHEDULE: auto, chain, ring or pieces).
SCHEDULE_ATTRIBUTES = ("large_schedule", "gather_schedule", "scatter_schedule")


def link_keywords(session: Any) -> dict[str, bool]:
    """The ``prepare`` keywords that compile ``session``'s chain and ring collectives before any capture.

    Any schedule other than ``pieces`` (``auto``, ``chain`` or ``ring``) can run
    chain or ring ops, and ``prepare(..., links=True)`` compiles every link
    collective, so none compiles while peers wait. The session caches its
    launchers, so passing the keyword in each of a session's prepare calls
    compiles each kernel once. Returns ``{}`` for a session whose ``prepare``
    takes no ``links`` keyword or whose schedules are all ``pieces``.
    """
    import inspect

    try:
        parameters = inspect.signature(session.prepare).parameters
    except (AttributeError, TypeError, ValueError):
        return {}
    if "links" not in parameters:
        return {}
    schedules = [getattr(session, name, "pieces") for name in SCHEDULE_ATTRIBUTES]
    return {"links": True} if any(schedule != "pieces" for schedule in schedules) else {}


def schedule_text(session: Any) -> str | None:
    """``large:<s>,gather:<s>,scatter:<s>`` for a session that names its schedules, else None."""
    values = [getattr(session, name, None) for name in SCHEDULE_ATTRIBUTES]
    if not all(isinstance(value, str) for value in values):
        return None
    return ",".join(f"{name.split('_')[0]}:{value}" for name, value in zip(SCHEDULE_ATTRIBUTES, values))


# Session attributes holding the link collectives' slot and piece in bytes
# (SIRCL_LINK_SLOT_BYTES, SIRCL_LINK_CHUNK_BYTES).
LINK_ATTRIBUTES = (("slot", "link_slot_bytes"), ("chunk", "link_chunk_bytes"))


def oneshot_limit(session: Any) -> int | None:
    """The largest all-reduce ``session``'s auto algorithm runs one-shot (``oneshot_max_bytes``,
    SIRCL_ONESHOT_MAX_BYTES), or None for a session that does not state it."""
    value = getattr(session, "oneshot_max_bytes", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# The link collectives with a piece of their own (``link_chunk_for``; SIRCL_GATHER_LINK_CHUNK_BYTES,
# SIRCL_SCATTER_LINK_CHUNK_BYTES, SIRCL_REDUCE_LINK_CHUNK_BYTES).
LINK_COLLECTIVES = ("gather", "scatter", "reduce")


def tuning_table(session: Any) -> str | None:
    """The hash of the tuning table ``session`` decides from (``stats()["tuning"]["table"]``), or None: no
    table, or a session without measured choices or statistics."""
    stats = getattr(session, "stats", None)
    if not callable(stats) or not callable(getattr(session, "tuned_choice", None)):
        return None
    try:
        info = stats().get("tuning")
    except Exception:  # noqa: BLE001 - a session without statistics names no table
        return None
    table = info.get("table") if isinstance(info, Mapping) else None
    return table if isinstance(table, str) else None


def tuned_backend(session: Any) -> Callable[[str, int], str | None] | None:
    """For a session with a tuning table, a function of (collective, bytes) giving the backend the table
    measured fastest for an eager call (``nccl`` or ``sircl``; ``tuned_backend``), or None where the table
    decides nothing (below its smallest decision, ``tuned_choice``); None for a session without a table.
    The answer depends only on the table, which the session's setup agreement makes the same on every
    rank, so every rank of the group decides alike."""
    choice = getattr(session, "tuned_choice", None)
    backend = getattr(session, "tuned_backend", None)
    if not callable(choice) or not callable(backend) or tuning_table(session) is None:
        return None

    def decide(collective: str, nbytes: int) -> str | None:
        if choice(collective, int(nbytes), mode="eager") is None:
            return None
        return str(backend(collective, int(nbytes), mode="eager"))

    return decide


def _byte_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def link_text(session: Any) -> str | None:
    """``slot:<bytes>,chunk:<bytes>`` for a session that states its link sizes, else None; then
    ``,gather:<bytes>,scatter:<bytes>,reduce:<bytes>``, the piece each link collective uses, for a session
    with ``link_chunk_for``."""
    values = [getattr(session, attribute, None) for _, attribute in LINK_ATTRIBUTES]
    if not all(_byte_count(value) for value in values):
        return None
    text = ",".join(f"{name}:{value}" for (name, _), value in zip(LINK_ATTRIBUTES, values))
    piece_for = getattr(session, "link_chunk_for", None)
    if callable(piece_for):
        pieces = [piece_for(collective) for collective in LINK_COLLECTIVES]
        if all(_byte_count(piece) for piece in pieces):
            text += "".join(f",{collective}:{piece}" for collective, piece in zip(LINK_COLLECTIVES, pieces))
    return text


# The collectives with chain and ring minimums of their own (``chain_min_for``, ``ring_min_for``), sizes of an
# all-reduce's message, an all-gather's output and a reduce-scatter's input.
MIN_COLLECTIVES = ("reduce", "gather", "scatter")


def _minimums(session: Any, per_collective: str, shared: str) -> str | int | None:
    """``reduce:<bytes>,gather:<bytes>,scatter:<bytes>`` from ``per_collective`` (a method) for a session with
    it, else the byte count ``shared`` names for a session that states one size, else None."""
    minimum_for = getattr(session, per_collective, None)
    if callable(minimum_for):
        values = [minimum_for(collective) for collective in MIN_COLLECTIVES]
        if all(_byte_count(value) for value in values):
            return ",".join(f"{collective}:{value}" for collective, value in zip(MIN_COLLECTIVES, values))
    value = getattr(session, shared, None)
    return value if _byte_count(value) else None


def chain_min(session: Any) -> str | int | None:
    """The smallest collective ``session``'s auto schedules run as chain ops, per collective
    (``chain_min_for``; SIRCL_CHAIN_MIN_BYTES sets one size for all three), or the one size an older session
    states (``chain_min_bytes``), or None."""
    return _minimums(session, "chain_min_for", "chain_min_bytes")


def ring_min(session: Any) -> str | int | None:
    """The smallest collective ``session``'s ring schedules run as ring ops, per collective (``ring_min_for``;
    SIRCL_RING_MIN_BYTES sets one size for all three), or the one size an older session states
    (``ring_min_bytes``), or None."""
    return _minimums(session, "ring_min_for", "ring_min_bytes")
