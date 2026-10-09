"""The two-round teardown of a ring session's or point-to-point channel set's close.

A rank's close must order itself against every peer of its group: a peer's kernels may still wait for writes this
rank's progress thread posts after its own kernels completed (a forward gated on a peer's flags, the last items of
a link op), and no rank may destroy queue pairs that a peer's progress thread still submits writes to. Once the
rank's own work is done (its device synchronized, its streams idle), :func:`ordered_close` runs:

- round 1 (health): every rank posts its own failure, ``None`` when it has none (a flag wait that timed out, a
  progress thread that stopped, a device synchronization that failed). The round completes once every rank's own
  kernels have finished, so every inbound write any kernel waited for has landed, while every progress thread
  still runs;
- the rank stops its own progress thread, which submits nothing more;
- round 2 (quiet), only when round 1 completed: it completes once every progress thread of the group has stopped,
  so no new write is submitted toward this rank's queue pairs, memory registrations or arena when they are
  destroyed.

The rounds order submissions, not completions: writes a progress thread submitted before it stopped (a final
credit, the completion of an earlier write) may still be in flight when a peer destroys its queue pairs. Destroying
a queue pair before deregistering the arena's memory region revokes both, so such a write fails at the network
adapter instead of reaching the arena; neither round drains a completion queue.

The close result is this rank's own failure, else the first peer's note (``rank <i>: <note>``), else why a round
did not complete; ``None`` when every rank was healthy and both rounds completed. :func:`close_native` then
destroys the native context; when a verbs object could not be released (``roce_destroy`` and ``p2p_destroy``
return the number of failed verbs calls), a peer's write could still reach the registered arena, so the arena is
kept allocated for the rest of the process (:data:`RETAINED`) and the result says so. The arena is kept, too, when
the rank's own GPU work was not shown complete: a failed device or stream synchronization is the rank's failure,
voted in round 1, and a kernel may still read or write the arena. An exception inside the teardown (a round's
thread that does not start, a progress-thread stop or a destroy that raises) is part of the close result too, and
the native context is still stopped and destroyed, each exactly once; a destroy that raised keeps the arena. A
session or channel set that is interrupted mid-teardown (``KeyboardInterrupt``) records a failed close, keeps its
native context referenced and its arena allocated, and re-raises: every later close returns that failure.

A round is one collective over the group's CPU process group, the channel of the setup agreement: an all-gather of
a fixed-size note (:class:`Note`, :data:`NOTE_BYTES` bytes as a uint8 tensor, :func:`tensor_exchange`), so a
pending round is exactly one pending collective. A note names the close it belongs to: the object's kind (a ring
session or a channel set), its ordinal (its index among the objects this rank created on that group, which every
rank creates in the same order, :func:`register`) and the round. A round in which any rank's note names another
close (a rank closing another object, or in another round) did not complete: the ranks' inventories or histories
differ, and the result says which rank closed what. A round runs in a daemon thread and is waited for at most the
session's flag-wait limit plus :data:`SLACK_S`: a peer whose last kernel waits out the whole limit before its
synchronization returns still arrives in time. A peer that never answers leaves the thread blocked and the close
goes on.

A round that did not complete marks its group unusable for teardown rounds in this process (:func:`mark_unusable`):
a later close on that group, of another session or channel set sharing it, stops and destroys without rounds
instead of issuing a collective that would pair with the abandoned one, which also bounds a dead group at one
wait per close. An abandoned round stays pending on the group and pairs with the next collective any user issues
there (vLLM's own collectives on its CPU group). The group must not carry further sessions, channel sets or
collectives: recreate the process group or end the processes that share it. This holds in a long-running process
that closes and reopens sessions as well as at shutdown.

Every rank must close the sessions and channel sets that share a group in the same order (the vLLM adapter's
close does), because the group pairs collectives by the order in which each rank issues them; a note's ordinal
detects a rank that does not.
"""

from __future__ import annotations

import dataclasses
import struct
import threading
from collections.abc import Callable, Sequence
from typing import Any, Optional

NOTE_BYTES = 512                  # a round's note on the wire (:func:`encode`)
NOTE_FORMAT = 2                   # byte 0 of a note on the wire
SLACK_S = 5.0                     # a round waits the session's flag-wait limit plus this
KINDS = ("session", "channels")   # the objects a close tears down, by their code on the wire (1, 2)

_HEADER = struct.Struct("<BBBBI")  # format, round, kind code, failure flag, ordinal


@dataclasses.dataclass(frozen=True)
class Note:
    """One rank's note in one round: the close it belongs to (``kind``, ``ordinal``, ``phase``: round 1 or 2)
    and the rank's failure (None when it has none)."""

    kind: str
    ordinal: int
    phase: int
    failure: Optional[str] = None

    def same_close(self, other: "Note") -> bool:
        return (self.kind, self.ordinal, self.phase) == (other.kind, other.ordinal, other.phase)

    def describe(self) -> str:
        return f"{self.kind} {self.ordinal} in round {self.phase}" if self.phase else "a note of another format"


Exchange = Callable[[Note], Sequence[Note]]

_LOCK = threading.Lock()
# id(group) -> (group, why its teardown rounds stopped); the group is held so that its id is never reused.
_UNUSABLE: dict[int, tuple[Any, str]] = {}
# (id(group), rank) -> (group, objects registered so far); the group is held so that its id is never reused.
_ORDINALS: dict[tuple[int, int], tuple[Any, int]] = {}
RETAINED: list[Any] = []          # arenas kept alive because their RDMA objects could not all be released


def encode(note: Note) -> bytes:
    """``note`` as :data:`NOTE_BYTES` bytes: a header (format, round, kind code, failure flag, ordinal), then the
    failure's UTF-8 text (truncated), zero-padded."""
    text = b"" if note.failure is None else str(note.failure).encode("utf-8", "replace")[:NOTE_BYTES - _HEADER.size]
    header = _HEADER.pack(NOTE_FORMAT, note.phase, KINDS.index(note.kind) + 1, int(note.failure is not None),
                          note.ordinal & 0xFFFFFFFF)
    return header + text + bytes(NOTE_BYTES - _HEADER.size - len(text))


def decode(raw: bytes) -> Note:
    """The note :func:`encode` wrote; a note of another format decodes to a phase-0 note that names no close."""
    if len(raw) < _HEADER.size or raw[0] != NOTE_FORMAT:
        return Note("unknown", -1, 0, "a teardown note of another format")
    _, phase, kind, failed, ordinal = _HEADER.unpack_from(raw)
    name = KINDS[kind - 1] if 1 <= kind <= len(KINDS) else "unknown"
    failure = raw[_HEADER.size:].rstrip(b"\x00").decode("utf-8", "replace") or "failed" if failed else None
    return Note(name, ordinal, phase, failure)


def tensor_exchange(group: Any) -> Exchange:
    """A round's exchange over a ``torch.distributed`` CPU process group: one ``all_gather`` of every rank's
    :func:`encode`-d note as a uint8 tensor."""

    def exchange(note: Note) -> list[Note]:
        import torch
        import torch.distributed as dist

        world = dist.get_world_size(group=group)
        mine = torch.tensor(list(encode(note)), dtype=torch.uint8)
        gathered = [torch.empty(NOTE_BYTES, dtype=torch.uint8) for _ in range(world)]
        dist.all_gather(gathered, mine, group=group)
        return [decode(bytes(part.tolist())) for part in gathered]

    return exchange


def register(group: Any, rank: int) -> int:
    """The ordinal of a new session or channel set of ``rank`` on ``group``: how many this rank registered on the
    group before it in this process. Every rank creates the objects of a group in the same order, so the ordinal of
    one object is the same on every rank."""
    with _LOCK:
        _, count = _ORDINALS.get((id(group), rank), (group, 0))
        _ORDINALS[(id(group), rank)] = (group, count + 1)
    return count


def unusable(group: Any) -> Optional[str]:
    """Why ``group``'s teardown rounds stopped in this process, or None."""
    with _LOCK:
        entry = _UNUSABLE.get(id(group))
    return None if entry is None else entry[1]


def mark_unusable(group: Any, why: str) -> None:
    """Stop teardown rounds on ``group`` in this process (the first reason is kept)."""
    with _LOCK:
        _UNUSABLE.setdefault(id(group), (group, why))


def retain(arena: Any) -> None:
    """Keep ``arena`` (a registered pinned buffer) alive for the rest of the process."""
    with _LOCK:
        RETAINED.append(arena)


@dataclasses.dataclass(frozen=True)
class Round:
    arrived: bool                          # every rank answered within the limit
    notes: Sequence[Any] = ()              # every rank's note, when it arrived
    why: Optional[str] = None              # why it did not arrive


def teardown_round(exchange: Exchange, note: Any, limit_s: float, name: str) -> Round:
    """One round: ``exchange(note)`` in a daemon thread, waited for at most ``limit_s`` seconds."""
    result: list[Round] = []

    def gather() -> None:
        try:
            result.append(Round(True, tuple(exchange(note))))
        except Exception as exc:  # noqa: BLE001 - a dead group still closes
            result.append(Round(False, why=f"{name}: {type(exc).__name__}: {exc}"))

    worker = threading.Thread(target=gather, name=f"sircl-teardown {name}", daemon=True)
    try:
        worker.start()
    except Exception as exc:  # noqa: BLE001 - a round that cannot start did not complete
        return Round(False, why=f"{name}: its thread did not start: {type(exc).__name__}: {exc}")
    worker.join(limit_s)
    if worker.is_alive() or not result:
        return Round(False, why=f"{name}: not every rank arrived within {limit_s:g} s")
    return result[0]


def _matched(round_: Round, mine: Note, name: str) -> Round:
    """``round_`` as a round that did not complete when a rank's note names another close than ``mine``."""
    if not round_.arrived:
        return round_
    for rank, note in enumerate(round_.notes):
        if not isinstance(note, Note) or not mine.same_close(note):
            other = note.describe() if isinstance(note, Note) else repr(note)
            return Round(False, why=f"{name}: rank {rank} closed {other} while this rank closed {mine.describe()}")
    return round_


def ordered_close(*, group: Any, exchange: Exchange, own_failure: Optional[str], limit_s: float,
                  stop: Callable[[], None], what: str, kind: str = "session", ordinal: int = 0) -> Optional[str]:
    """Round 1 (health), ``stop()``, then round 2 (quiet) when round 1 completed; the close result (see the
    module docstring). The notes name the close (``kind``, ``ordinal``, the round); a round whose notes do not
    all name it did not complete. ``stop`` runs exactly once. On a group already marked unusable no round runs."""
    reason = unusable(group)
    if reason is not None:
        stop()
        return own_failure or f"{what}: closed without the teardown rounds ({reason})"
    mine = Note(kind, ordinal, 1, own_failure)
    first = _matched(teardown_round(exchange, mine, limit_s, f"{what} round 1"), mine, f"{what} round 1")
    if not first.arrived:
        mark_unusable(group, first.why or f"{what} round 1 did not complete")
        stop()
        return own_failure or first.why
    result = own_failure
    if result is None:
        result = next((f"rank {index}: {note.failure}" for index, note in enumerate(first.notes)
                       if note.failure is not None), None)
    stop()
    quiet = Note(kind, ordinal, 2, None)
    second = _matched(teardown_round(exchange, quiet, limit_s, f"{what} round 2"), quiet, f"{what} round 2")
    if not second.arrived:
        mark_unusable(group, second.why or f"{what} round 2 did not complete")
        return result or second.why
    return result


def close_native(native: Any, *, group: Any, exchange: Exchange, own_failure: Optional[str], limit_s: float,
                 abort: bool, arena: Any, what: str, unsettled: bool = False, kind: str = "session",
                 ordinal: int = 0) -> Optional[str]:
    """Close one rank's native context ``native`` (``stop()``, and ``close()`` returning the number of verbs
    calls that failed) once the rank's own work is done: with ``abort`` it stops at once, without rounds, else
    through :func:`ordered_close` (whose notes name ``kind`` and ``ordinal``); then it is destroyed. ``arena`` is
    retained when a verbs object could not be released, and with ``unsettled``, when the rank's own GPU work was
    not shown complete (a stream or device synchronization failed): a kernel may still read or write it. Returns
    the close result; ``native`` None (setup failed before it existed) returns ``own_failure``."""
    if native is None:
        if unsettled and arena is not None:
            retain(arena)
        return own_failure
    stopped = []

    def stop_once() -> None:
        if not stopped:
            stopped.append(True)
            native.stop()

    # Any exception from here on is part of the close result, and the context is stopped and destroyed exactly
    # once whatever happened before: a close that raised is a failed close, never a healthy one on retry.
    result = own_failure
    try:
        if abort:
            stop_once()
        else:
            result = ordered_close(group=group, exchange=exchange, own_failure=own_failure, limit_s=limit_s,
                                   stop=stop_once, what=what, kind=kind, ordinal=ordinal)
    except Exception as exc:  # noqa: BLE001 - the close's terminal failure
        result = _joined(result, f"{what}: the teardown raised {type(exc).__name__}: {exc}")
        if not stopped:
            try:
                stop_once()
            except Exception as stop_exc:  # noqa: BLE001
                result = _joined(result, f"{what}: stopping the progress thread raised {type(stop_exc).__name__}: "
                                         f"{stop_exc}")
    try:
        failed: Optional[int] = int(native.close() or 0)
    except Exception as exc:  # noqa: BLE001 - its objects may survive: the arena is kept
        failed = None
        result = _joined(result, f"{what}: destroying the native context raised {type(exc).__name__}: {exc}; the "
                                 "registered arena stays allocated for the rest of the process")
    if failed is None:
        pass
    elif failed:
        note = (f"{what}: {failed} RDMA teardown call(s) failed; the registered arena stays allocated for the rest "
                "of the process")
        result = f"{result}; {note}" if result else note
    elif unsettled:
        note = f"{what}: the arena stays allocated for the rest of the process: its GPU work was not shown complete"
        result = f"{result}; {note}" if result else note
    if (failed is None or failed or unsettled) and arena is not None:
        retain(arena)
    return result


def _joined(first: Optional[str], second: str) -> str:
    return f"{first}; {second}" if first else second


__all__ = ["KINDS", "NOTE_BYTES", "NOTE_FORMAT", "Note", "RETAINED", "Round", "SLACK_S", "close_native", "decode",
           "encode", "mark_unusable", "ordered_close", "register", "retain", "teardown_round", "tensor_exchange",
           "unusable"]
