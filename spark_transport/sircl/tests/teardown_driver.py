"""The close cases of ``tests/test_teardown.py`` that need the session modules, run in a process of their own.

``python teardown_driver.py closes`` runs the ring session's and the point-to-point channel set's real
``close`` (``oneshot/runtime.py``, ``p2p/session.py``) on objects built without ``__init__``: every rank is a
thread, the native context a recording stand-in, and the teardown rounds run over
:class:`sparkring_sircl.testing.gpu_emulation.TeardownRounds`. The session modules import CUDA Python and the
CuTe DSL when they load; this process stands both in with mocks (no close path calls them), which is why it
is a process of its own. It prints one JSON object of every case's observations.

``python teardown_driver.py gloo <rank> <store file>`` is one rank of a two-rank gloo group that exchanges two
rounds of notes through :func:`sparkring_sircl.teardown.tensor_exchange` and prints them as JSON.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
import threading
import types
from unittest.mock import MagicMock


class _CudaStandIn(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Mocks for the CUDA Python and CuTe DSL packages, for this process only."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("cuda", "cutlass"):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = MagicMock(name=spec.name)
        module.__path__ = []
        module.__spec__ = spec
        return module

    def exec_module(self, module):
        pass


def gloo(rank: int, store: str) -> None:
    import torch.distributed as dist

    from sparkring_sircl import teardown

    dist.init_process_group("gloo", store=dist.FileStore(store, 2), rank=rank, world_size=2)
    exchange = teardown.tensor_exchange(dist.group.WORLD)
    first = exchange(teardown.Note("session", 3, 1, None if rank == 0 else "rank 1 timed out"))
    second = exchange(teardown.Note("channels", 1, 2, "é" * 600 if rank == 0 else None))
    dist.destroy_process_group()
    print(json.dumps({"first": [[note.kind, note.ordinal, note.phase, note.failure] for note in first],
                      "second": [[note.kind, note.ordinal, note.phase, note.failure] for note in second]}))


class Native:
    """The native surface a close touches, recording its steps."""

    def __init__(self, *, failed: bool = False, error: str = "", destroy_failures: int = 0,
                 stop_raises: BaseException | None = None, close_raises: BaseException | None = None) -> None:
        self.steps: list[str] = []
        self._failed, self._error, self.destroy_failures = failed, error, destroy_failures
        self.stop_raises, self.close_raises = stop_raises, close_raises

    def failed(self) -> bool:
        return self._failed

    def error(self) -> str:
        return self._error

    def stop(self) -> None:
        self.steps.append("stop")
        if self.stop_raises is not None:
            raise self.stop_raises

    def close(self) -> int:
        self.steps.append("destroy")
        if self.close_raises is not None:
            raise self.close_raises
        return self.destroy_failures


def closes() -> None:
    sys.meta_path.insert(0, _CudaStandIn())

    import numpy as np
    import torch

    from sparkring_sircl import protocol as proto
    from sparkring_sircl import teardown
    from sparkring_sircl.oneshot import runtime
    from sparkring_sircl.p2p import protocol as p2p_proto
    from sparkring_sircl.p2p import session as p2p_session
    from sparkring_sircl.testing.gpu_emulation import TeardownRounds

    teardown.SLACK_S = 0.1           # a round gives up 0.1 s after the wait limit
    failing_sync: set[str] = set()   # names of the rank threads whose device synchronization raises

    def synchronize(device=None) -> None:
        if threading.current_thread().name in failing_sync:
            raise RuntimeError("an illegal memory access was encountered")

    torch.cuda.synchronize = synchronize

    class Stream:
        """A channel's CUDA stream stand-in whose synchronization fails when asked to."""

        def __init__(self, fails: bool) -> None:
            self.fails = fails

        def synchronize(self) -> None:
            if self.fails:
                raise RuntimeError("an illegal memory access was encountered")

    def destroy_stream(raw: int) -> None:
        if raw == 2:
            raise RuntimeError("cuStreamDestroy failed: CUresult.CUDA_ERROR_INVALID_HANDLE")

    p2p_session._destroy_stream = destroy_stream

    def channel_on_rank_1(stream_fails: bool, raw: int):
        """A real channel direction on rank 1 whose stream is ``Stream(stream_fails)`` with handle ``raw``."""
        def prepare(objects) -> None:
            channel = p2p_session._Channel(torch.device("cpu"))
            channel.stream, channel.raw_stream = Stream(stream_fails), raw
            objects[1]._out = [channel]
        return prepare

    def session(rank: int, group: object, native, wait_s: float):
        s = runtime.RoceOneshotAllReduce.__new__(runtime.RoceOneshotAllReduce)
        s.device, s.rank, s.world_size, s._group = torch.device("cpu"), rank, 2, group
        s._closed, s.close_result, s._lock, s._proxy = False, None, threading.Lock(), native
        s._profile = s._profiled = None
        s.wait_regime, s.startup_wait_s, s.serving_wait_s = "serving", 600.0, wait_s
        s._region = bytearray(64)
        s._teardown_ordinal = 0
        s._ctrl_np = np.zeros(32, dtype=np.int32)
        return s

    def channels(rank: int, group: object, native, wait_s: float):
        c = p2p_session.PointToPoint.__new__(p2p_session.PointToPoint)
        c.device, c.rank, c.world_size, c._group = torch.device("cpu"), rank, 2, group
        c._closed, c.close_result, c._lock, c._native = False, None, threading.Lock(), native
        c._out, c._in, c.wait_regime = [], [], "serving"
        c.settings = types.SimpleNamespace(startup_wait_s=600.0, serving_wait_s=wait_s)
        c._region = bytearray(64)
        c._teardown_ordinal = 0
        c._control = np.zeros(p2p_proto.CONTROL_BYTES // 4, dtype=np.int32)
        return c

    def health(obj) -> str | None:
        try:
            obj.check_health()
        except RuntimeError as error:
            return str(error)
        return None

    def run(make, *, natives=None, prepare=None, abort=(), wait_s: float = 0.2, group=None) -> dict:
        group = object() if group is None else group
        natives = natives or [Native(), Native()]
        objects = [make(rank, group, natives[rank], wait_s) for rank in range(2)]
        rounds = TeardownRounds(objects, 2)
        if prepare is not None:
            prepare(objects)
        results: list = [None, None]

        def close(rank: int) -> None:
            results[rank] = objects[rank].close(abort=rank in abort)

        threads = [threading.Thread(target=close, args=(rank,), name=f"rank-{rank}") for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        return {"results": results, "again": [obj.close() for obj in objects],
                "kept": [obj.close_result for obj in objects], "steps": [native.steps if native else None
                                                                       for native in natives],
                "entered": rounds.entered, "unusable": teardown.unusable(group),
                "retained": [any(kept is obj._region for kept in teardown.RETAINED) for obj in objects],
                "detached": [getattr(obj, "_proxy", getattr(obj, "_native", None)) is None for obj in objects],
                "health": [health(obj) for obj in objects], "group": group}

    def poison_session(objects) -> None:
        ctrl = objects[1]._ctrl_np
        ctrl[proto.Ctrl.MISSING_PEER], ctrl[proto.Ctrl.MISSING_LANE] = 0, 1
        ctrl[proto.Ctrl.ERROR_KIND], ctrl[proto.Ctrl.ERROR_SEQ] = int(proto.ErrorKind.SLOT_OP), 3

    def poison_channels(objects) -> None:
        control = objects[1]._control
        control[p2p_proto.Control.ERROR_PEER], control[p2p_proto.Control.ERROR_LANE] = 0, 0
        control[p2p_proto.Control.ERROR_KIND] = int(p2p_proto.ErrorKind.FLAG)
        control[p2p_proto.Control.POISON], control[p2p_proto.Control.ERROR_TAG] = 1, 5

    out: dict = {}
    for kind, make, poison in (("session", session, poison_session), ("channels", channels, poison_channels)):
        out[f"{kind} healthy"] = run(make)
        out[f"{kind} poisoned"] = run(make, prepare=poison)
        out[f"{kind} native failed"] = run(make, natives=[Native(failed=True, error="RDMA write of sequence 9 to "
                                                                                    "rank 1 failed"), Native()])
        out[f"{kind} destroy failed"] = run(make, natives=[Native(), Native(destroy_failures=2)])
        out[f"{kind} no native"] = run(make, natives=[None, None])
        aborted = run(make, abort=(0,))
        out[f"{kind} aborted"] = aborted
        out[f"{kind} later close"] = run(make, group=aborted["group"])
    failing_sync.add("rank-1")
    out["session sync failed"] = run(session)
    failing_sync.clear()
    out["channels stream sync failed"] = run(channels, prepare=channel_on_rank_1(True, 1))
    out["channels stream destroy failed"] = run(channels, prepare=channel_on_rank_1(False, 2))

    def ordinals(first: int, second: int):
        """Rank 0 closes the object of ordinal ``first`` while rank 1 closes that of ordinal ``second``."""
        def prepare(objects) -> None:
            objects[0]._teardown_ordinal, objects[1]._teardown_ordinal = first, second
        return prepare

    out["session other object"] = run(session, prepare=ordinals(1, 0))
    out["channels other object"] = run(channels, prepare=ordinals(1, 0))

    class NoThread:
        """A thread whose start fails, as when the process cannot start another."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("can't start new thread")

    for kind, make in (("session", session), ("channels", channels)):
        saved = teardown.threading
        teardown.threading = types.SimpleNamespace(Thread=NoThread)
        try:
            out[f"{kind} thread start failed"] = run(make)
        finally:
            teardown.threading = saved
        out[f"{kind} native stop raised"] = run(make, natives=[Native(), Native(stop_raises=RuntimeError("stop failed"))])
        out[f"{kind} native destroy raised"] = run(
            make, natives=[Native(), Native(close_raises=RuntimeError("destroy failed"))])
        native = Native(close_raises=KeyboardInterrupt())
        lone = make(0, object(), native, 0.2)
        lone._teardown_exchange = lambda note: [note]     # an abort holds no round
        try:
            lone.close(abort=True)
            raised = None
        except KeyboardInterrupt:
            raised = "KeyboardInterrupt"
        again = lone.close()
        out[f"{kind} interrupted"] = {
            "raised": raised, "kept": lone.close_result, "again": again, "steps": native.steps,
            "attached": getattr(lone, "_proxy", getattr(lone, "_native", None)) is native,
            "retained": any(kept is lone._region for kept in teardown.RETAINED), "group": None}
    for case in out.values():
        case.pop("group")
    print(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "gloo":
        gloo(int(sys.argv[2]), sys.argv[3])
    else:
        closes()
