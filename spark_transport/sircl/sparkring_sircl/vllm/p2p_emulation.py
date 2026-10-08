"""CPU reference of SIRCL's point-to-point channels, with emulated ranks in one process.

:func:`p2p_module` returns a package (``API_VERSION``, ``is_supported``,
``PointToPoint``) whose channels implement the surface the vLLM adapter uses
(:class:`.sessionapi.PointToPointChannels`) on CPU tensors, for groups of
:class:`.emulation.EmulatedGroup` ranks (threads of one process). It states,
as executable code, what the adapter relies on from the channels:

- every ordered pair with a channel is first-in first-out: the n-th receive a
  rank issues toward a peer takes the n-th message that peer issued toward it
  (receives and sends are numbered when they are issued, not when they are
  waited);
- a receive names the byte count of that message; a different count fails
  the group's channels on every rank (they are poisoned and
  ``check_health`` raises);
- ``isend`` and ``irecv`` return work objects at once; the bytes reach the
  receive's tensor when its work is waited (the CUDA channels order the
  caller's stream instead), so a rank may issue its receives before the
  matching sends exist;
- setup is collective over the group and compares the channel and window
  tables the ranks were given.

It does not model the wire, relays, credits, streams or timing.
"""

from __future__ import annotations

import threading
import types
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from .emulation import EmulatedGroup, EmulationError

TIMEOUT_S = 20.0


class ChannelsPoisoned(EmulationError):
    """The group's channels failed (a size mismatch or a missing message); every later call raises."""


class _Shared:
    """The queues of one group's channels, shared by its ranks."""

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.messages: dict[tuple[int, int], list[bytes | None]] = {}   # (source, destination) -> issued messages
        self.receives: dict[tuple[int, int], int] = {}                 # (source, destination) -> receives issued
        self.poisoned: str | None = None
        self.records: list[tuple[int, str, int, int]] = []      # (rank, kind, peer, bytes)


_SHARED_LOCK = threading.Lock()


def _shared(group: EmulatedGroup) -> _Shared:
    with _SHARED_LOCK:
        state = getattr(group.objects, "_p2p_shared", None)
        if state is None:
            state = _Shared()
            group.objects._p2p_shared = state
        return state


class _Work:
    def __init__(self, run=None) -> None:
        self._run = run
        self._done = run is None

    def wait(self, timeout: Any = None) -> bool:
        if not self._done:
            self._run()
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done

    def is_success(self) -> bool:
        return self._done

    def exception(self) -> None:
        return None


class EmulatedChannels:
    """One rank's emulated point-to-point channels."""

    wait_regime = "startup"

    def __init__(self, *, exchange_group: EmulatedGroup, device: Any = "cpu",
                 peer_routes: Mapping[int, Sequence[str]] | None = None, layout: Any = None,
                 channels: Iterable[tuple[int, int]] | None = None,
                 windows: Sequence[Sequence[Sequence[int]]] | None = None,
                 unavailable: Mapping[tuple[int, int], str] | None = None, **ignored: Any) -> None:
        self.rank = exchange_group.sircl_rank()
        self.world_size = exchange_group.sircl_size()
        self.device = torch.device(device)
        self.peer_routes = dict(peer_routes or {})
        self.layout = layout
        self.windows = windows
        self._shared = _shared(exchange_group)
        table = {(a, b) for a in range(self.world_size) for b in range(self.world_size) if a != b}
        if channels is not None:
            table = {pair for a, b in channels for pair in ((a, b), (b, a))}
        self.channel_problems: dict[int, str] = {}
        for (a, b), why in (unavailable or {}).items():
            table -= {(a, b), (b, a)}
            if self.rank in (a, b):
                self.channel_problems[b if a == self.rank else a] = why
        self._table = table
        self.closed = False
        settings = (sorted(table), None if windows is None else [[list(lanes) for lanes in row] for row in windows],
                    None if layout is None else str(layout))
        votes = exchange_group.sircl_all_gather_object(("p2p-setup", settings))
        if any(vote != votes[0] for vote in votes):
            raise RuntimeError("SIRCL point-to-point setup failed: ranks were given different channel or window "
                               "tables")

    # -- the adapter's surface -------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._shared.poisoned is not None

    def has_channel(self, peer: int) -> bool:
        return (self.rank, int(peer)) in self._table

    def channel_problem(self, peer: int) -> str | None:
        if self.has_channel(peer):
            return None
        return self.channel_problems.get(int(peer), f"ranks {self.rank} and {peer} have no point-to-point channel")

    def prepare(self) -> None:
        return None

    def enter_startup(self) -> None:
        self.wait_regime = "startup"

    def enter_serving(self) -> None:
        self.wait_regime = "serving"

    def check_health(self) -> None:
        if self._shared.poisoned is not None:
            raise ChannelsPoisoned(f"SIRCL point-to-point channels on rank {self.rank}: {self._shared.poisoned}")

    def _check(self, peer: int) -> None:
        self.check_health()
        if self.closed:
            raise EmulationError("the channels are closed")
        if not self.has_channel(peer):
            raise ValueError(f"SIRCL point-to-point toward rank {peer}: {self.channel_problem(peer)}")

    def isend(self, tensor: torch.Tensor, peer: int) -> _Work:
        self._check(peer)
        payload = tensor.detach().contiguous().view(-1).view(torch.uint8).clone().numpy().tobytes()
        state = self._shared
        with state.lock:
            state.messages.setdefault((self.rank, int(peer)), []).append(payload)
            state.records.append((self.rank, "send", int(peer), len(payload)))
            state.lock.notify_all()
        return _Work()

    def irecv(self, tensor: torch.Tensor, peer: int) -> _Work:
        self._check(peer)
        nbytes = tensor.numel() * tensor.element_size()
        state = self._shared
        key = (int(peer), self.rank)
        with state.lock:
            ticket = state.receives.get(key, 0)
            state.receives[key] = ticket + 1

        def receive() -> None:
            with state.lock:
                messages = state.messages.setdefault(key, [])
                if not state.lock.wait_for(lambda: len(messages) > ticket or state.poisoned is not None, TIMEOUT_S):
                    state.poisoned = f"rank {self.rank} waited {TIMEOUT_S} s for a message from rank {peer}"
                    state.lock.notify_all()
                self.check_health()
                payload, messages[ticket] = messages[ticket], None
                if len(payload) != nbytes:
                    state.poisoned = (f"rank {self.rank} received a message of {len(payload)} bytes from rank {peer}, "
                                      f"but the receive expected {nbytes}: the two ranks issued messages of different "
                                      "sizes on this channel")
                    state.lock.notify_all()
                    self.check_health()
                state.records.append((self.rank, "recv", int(peer), nbytes))
            if nbytes:
                source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                if tensor.is_contiguous():
                    tensor.view(-1).view(torch.uint8).copy_(source)
                else:
                    tensor.copy_(source.view(tensor.dtype).view(tensor.shape))

        return _Work(receive)

    def send(self, tensor: torch.Tensor, peer: int) -> None:
        self.isend(tensor, peer).wait()

    def recv(self, tensor: torch.Tensor, peer: int) -> torch.Tensor:
        self.irecv(tensor, peer).wait()
        return tensor

    def batch_isend_irecv(self, ops: Sequence[tuple[str, torch.Tensor, int]]) -> list[_Work]:
        return [self.isend(tensor, peer) if kind == "send" else self.irecv(tensor, peer) for kind, tensor, peer in ops]

    def stats(self) -> dict[str, Any]:
        return {"channels": sorted(peer for peer in range(self.world_size) if self.has_channel(peer)),
                "slots": 8, "slot_bytes": 524288, "wait_regime": self.wait_regime}

    def records(self) -> list[tuple[int, str, int, int]]:
        with self._shared.lock:
            return list(self._shared.records)

    def close(self) -> None:
        self.closed = True


def p2p_module(name: str = "sircl_p2p_emulated", *, supported: bool = True):
    """A point-to-point package (``API_VERSION``, ``is_supported``, ``PointToPoint``) of emulated channels; every
    constructed context is kept in the module's ``created`` list."""
    module = types.ModuleType(name)
    module.API_VERSION = 1
    module.created = []
    module.is_supported = lambda: supported

    class PointToPoint(EmulatedChannels):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            module.created.append(self)

    module.PointToPoint = PointToPoint
    return module


__all__ = ["ChannelsPoisoned", "EmulatedChannels", "p2p_module"]
