"""The two-round teardown of a session's and a channel set's close (:mod:`sparkring_sircl.teardown`).

- the round protocol itself: the fixed-size note, a round that arrives, raises or stays silent, the health
  vote of round 1, round 2 only after round 1 arrived, a missed round marking the group unusable so that a
  later close on it holds no round, and the destroy of a native context whose verbs objects were not all
  released keeping its arena;
- the ring session's and the point-to-point channel set's real ``close`` on objects built without
  ``__init__`` (``teardown_driver.py closes``, a process of its own whose CUDA modules are mocks): the result
  every rank gets for a healthy group, a poisoned rank, a failed progress thread, a failed device
  synchronization, a failed destroy, an abort and a later close on the abandoned group;
- one round trip of notes through a two-rank gloo group (``teardown_driver.py gloo``);
- the native destroys (``roce_destroy``, ``p2p_destroy``) returning how many verbs calls failed, on the
  simulator build with the fake verbs' ``fv_fail_teardown``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sparkring_sircl import routes, teardown

DRIVER = Path(__file__).with_name("teardown_driver.py")
PROJECT = Path(__file__).resolve().parents[1]


def _driver(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run([sys.executable, str(DRIVER), *args], capture_output=True, text=True, env=env,
                          timeout=timeout)


class Recorder:
    """A native context stand-in: stop and destroy recorded in order."""

    def __init__(self, destroy_failures: int = 0) -> None:
        self.steps: list[str] = []
        self.destroy_failures = destroy_failures

    def stop(self) -> None:
        self.steps.append("stop")

    def close(self) -> int:
        self.steps.append("destroy")
        return self.destroy_failures


def thread_exchanges(world: int):
    """Every rank's exchange over one barrier-backed all-gather, and the rounds each rank entered."""
    barrier = threading.Barrier(world)
    slots: list = [None] * world
    entered = [0] * world

    def exchange_of(rank: int):
        def exchange(note):
            entered[rank] += 1
            slots[rank] = note
            barrier.wait(timeout=10)
            out = list(slots)
            barrier.wait(timeout=10)
            return out
        return exchange

    return [exchange_of(rank) for rank in range(world)], entered


def close_all(world: int, group, own=None, limit_s: float = 5.0, abort=()):
    """Every rank's :func:`teardown.close_native` at once; results, recorders and rounds entered."""
    exchanges, entered = thread_exchanges(world)
    natives = [Recorder() for _ in range(world)]
    own = own or [None] * world
    results: list = [None] * world

    def run(rank: int) -> None:
        results[rank] = teardown.close_native(natives[rank], group=group, exchange=exchanges[rank],
                                              own_failure=own[rank], limit_s=limit_s, abort=rank in abort,
                                              arena=None, what=f"close on rank {rank}")

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    return results, natives, entered


# -- the note ------------------------------------------------------------------------------------------------

def test_a_note_is_a_fixed_size_record_that_names_its_close():
    for note in (teardown.Note("session", 0, 1), teardown.Note("channels", 7, 2, "rank 1 timed out"),
                 teardown.Note("session", 2, 1, "naïve ✓")):
        raw = teardown.encode(note)
        assert len(raw) == teardown.NOTE_BYTES and raw[0] == teardown.NOTE_FORMAT
        assert teardown.decode(raw) == note
    assert teardown.decode(teardown.encode(teardown.Note("session", 0, 1, ""))).failure == "failed"
    long = teardown.decode(teardown.encode(teardown.Note("channels", 1, 2, "x" * 2000)))
    assert long.failure == "x" * (teardown.NOTE_BYTES - 8) and long.same_close(teardown.Note("channels", 1, 2))
    other = teardown.decode(b"\x01rank 1 timed out" + bytes(teardown.NOTE_BYTES - 17))     # an earlier format
    assert other.phase == 0 and not other.same_close(teardown.Note("session", 0, 1))


def test_ordinals_count_each_ranks_objects_of_a_group():
    group, other = object(), object()
    assert [teardown.register(group, 0) for _ in range(3)] == [0, 1, 2]
    assert teardown.register(group, 1) == 0 and teardown.register(other, 0) == 0


def test_a_round_whose_notes_name_another_close_did_not_complete():
    # Rank 0 closes the object of ordinal 1 while rank 1 closes that of ordinal 0: both rounds would pair, and
    # before the notes named their close both ranks returned None. Now neither does, and the group is marked.
    group = object()
    peer = {1: teardown.Note("session", 0, 1), 2: teardown.Note("session", 0, 2)}
    stops: list[int] = []

    def exchange(note):
        return [note, peer[note.phase]]

    result = teardown.ordered_close(group=group, exchange=exchange, own_failure=None, limit_s=5.0,
                                    stop=lambda: stops.append(1), what="w", kind="session", ordinal=1)
    assert result == "w round 1: rank 1 closed session 0 in round 1 while this rank closed session 1 in round 1"
    assert stops == [1] and teardown.unusable(group) == result


# -- one round -----------------------------------------------------------------------------------------------

def test_a_round_that_arrives_returns_every_note():
    seen = []
    result = teardown.teardown_round(lambda note: seen.append(note) or [None, "late"], "mine", 5.0, "round 1")
    assert result.arrived and list(result.notes) == [None, "late"] and seen == ["mine"]


def test_a_round_whose_exchange_raises_did_not_arrive():
    def exchange(note):
        raise RuntimeError("the group is gone")

    result = teardown.teardown_round(exchange, None, 5.0, "round 1")
    assert not result.arrived and "round 1: RuntimeError: the group is gone" in result.why


def test_a_silent_round_is_bounded_by_its_limit():
    started = time.monotonic()
    result = teardown.teardown_round(lambda note: time.sleep(3600), None, 0.2, "round 1")
    elapsed = time.monotonic() - started
    assert not result.arrived and result.why == "round 1: not every rank arrived within 0.2 s"
    assert 0.15 <= elapsed < 5.0


# -- the ordered close ---------------------------------------------------------------------------------------

def test_a_healthy_group_holds_both_rounds_around_every_stop():
    group = object()
    results, natives, entered = close_all(3, group)
    assert results == [None, None, None]
    assert entered == [2, 2, 2]
    assert all(native.steps == ["stop", "destroy"] for native in natives)
    assert teardown.unusable(group) is None


def test_round_one_votes_health_and_every_rank_learns_the_first_failure():
    group = object()
    own = [None, "rank 1's flag wait timed out", "rank 2's progress thread failed"]
    results, natives, entered = close_all(3, group, own=own)
    assert results == ["rank 1: rank 1's flag wait timed out", own[1], own[2]]
    assert entered == [2, 2, 2]
    assert teardown.unusable(group) is None


def test_a_missed_round_one_skips_round_two_and_marks_the_group():
    group = object()
    results, natives, entered = close_all(2, group, limit_s=0.3, abort=(0,))
    assert results[0] is None and natives[0].steps == ["stop", "destroy"]
    assert results[1] == "close on rank 1 round 1: not every rank arrived within 0.3 s"
    assert natives[1].steps == ["stop", "destroy"] and entered == [0, 1]
    assert "round 1: not every rank arrived" in teardown.unusable(group)
    # A later close on the group (another session sharing it) holds no round.
    calls: list = []
    native = Recorder()
    later = teardown.close_native(native, group=group, exchange=lambda note: calls.append(note) or [note],
                                  own_failure=None, limit_s=5.0, abort=False, arena=None, what="later close")
    assert calls == [] and native.steps == ["stop", "destroy"]
    assert later.startswith("later close: closed without the teardown rounds (close on rank 1 round 1")
    # Its own failure takes precedence over the reason.
    assert teardown.close_native(Recorder(), group=group, exchange=lambda note: [note], own_failure="mine",
                                 limit_s=5.0, abort=False, arena=None, what="w") == "mine"


def test_a_missed_round_two_marks_the_group_and_keeps_the_vote():
    group = object()
    calls = []

    def exchange(note):
        calls.append(note)
        if len(calls) == 2:
            raise RuntimeError("peer gone")
        return [note, teardown.Note(note.kind, note.ordinal, note.phase, "rank 1 failed")]

    stops = []
    result = teardown.ordered_close(group=group, exchange=exchange, own_failure=None, limit_s=5.0,
                                    stop=lambda: stops.append(len(calls)), what="w")
    assert result == "rank 1: rank 1 failed" and stops == [1] and len(calls) == 2
    assert "w round 2: RuntimeError: peer gone" in teardown.unusable(group)


def test_a_destroy_that_leaves_verbs_objects_keeps_the_arena():
    arena = bytearray(16)
    native = Recorder(destroy_failures=3)
    result = teardown.close_native(native, group=object(), exchange=lambda note: [note], own_failure=None,
                                   limit_s=5.0, abort=False, arena=arena, what="close on rank 0")
    assert result == ("close on rank 0: 3 RDMA teardown call(s) failed; the registered arena stays allocated "
                      "for the rest of the process")
    assert any(kept is arena for kept in teardown.RETAINED)
    native = Recorder(destroy_failures=1)
    result = teardown.close_native(native, group=object(), exchange=lambda note: [note], own_failure="mine",
                                   limit_s=5.0, abort=True, arena=None, what="w")
    assert result.startswith("mine; w: 1 RDMA teardown call(s) failed") and native.steps == ["stop", "destroy"]


def test_no_native_context_holds_no_round():
    calls: list = []
    assert teardown.close_native(None, group=object(), exchange=lambda note: calls.append(note) or [note],
                                 own_failure="setup failed", limit_s=5.0, abort=False, arena=None,
                                 what="w") == "setup failed"
    assert calls == []


# -- the session's and the channel set's close ---------------------------------------------------------------

@pytest.fixture(scope="module")
def closes():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    process = _driver("closes")
    assert process.returncode == 0, process.stderr[-3000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("kind", ["session", "channels"])
def test_a_healthy_close_holds_both_rounds_and_destroys(closes, kind):
    case = closes[f"{kind} healthy"]
    assert case["results"] == [None, None] and case["again"] == [None, None] and case["kept"] == [None, None]
    assert case["steps"] == [["stop", "destroy"], ["stop", "destroy"]] and case["entered"] == [2, 2]
    assert case["detached"] == [True, True] and case["retained"] == [False, False] and case["unusable"] is None
    closed = "SIRCL session is closed" if kind == "session" else "SIRCL point-to-point channels are closed"
    assert case["health"] == [closed, closed]


def test_a_poisoned_session_rank_is_named_on_every_rank(closes):
    case = closes["session poisoned"]
    own = ("SIRCL collective on rank 1 timed out waiting for rank 0 lane 1 at sequence 3 (wait limit 0.2 s, "
           "serving regime)")
    assert case["results"][1].startswith(own)
    assert case["results"][0] == f"rank 1: {case['results'][1]}"
    assert case["again"] == case["results"] == case["kept"] and case["entered"] == [2, 2]
    assert case["health"][1].startswith(own)          # the poison outlives the close


def test_poisoned_channels_are_named_on_every_rank(closes):
    case = closes["channels poisoned"]
    assert case["results"][1].startswith("SIRCL point-to-point channels on rank 1: a kernel timed out waiting "
                                         "for item 4 from rank 0 lane 0")
    assert case["results"][0] == f"rank 1: {case['results'][1]}" and case["entered"] == [2, 2]


@pytest.mark.parametrize("kind, text", [("session", "SIRCL progress thread failed on rank 0: "),
                                        ("channels", "SIRCL point-to-point channels failed on rank 0: ")])
def test_a_failed_progress_thread_is_voted(closes, kind, text):
    case = closes[f"{kind} native failed"]
    assert case["results"][0] == text + "RDMA write of sequence 9 to rank 1 failed"
    assert case["results"][1] == f"rank 0: {case['results'][0]}"


KEPT = "the arena stays allocated for the rest of the process: its GPU work was not shown complete"


def test_a_failed_device_synchronization_is_voted_and_keeps_the_arena(closes):
    case = closes["session sync failed"]
    own = ("SIRCL session on rank 1: device synchronization failed: RuntimeError: an illegal memory access was "
           "encountered")
    assert case["results"] == [f"rank 1: {own}", f"{own}; SIRCL session teardown on rank 1: {KEPT}"]
    assert case["again"] == case["results"] and case["entered"] == [2, 2]
    assert case["retained"] == [False, True] and case["steps"] == [["stop", "destroy"], ["stop", "destroy"]]


def test_a_failed_channel_stream_synchronization_is_voted_and_keeps_the_arena(closes):
    case = closes["channels stream sync failed"]
    own = ("SIRCL point-to-point channels on rank 1: stream synchronization failed: RuntimeError: an illegal "
           "memory access was encountered")
    assert case["results"] == [f"rank 1: {own}", f"{own}; SIRCL point-to-point teardown on rank 1: {KEPT}"]
    assert case["again"] == case["results"] == case["kept"] and case["entered"] == [2, 2]
    assert case["retained"] == [False, True] and case["steps"] == [["stop", "destroy"], ["stop", "destroy"]]


@pytest.mark.parametrize("kind, what", [("session", "SIRCL session teardown on rank"),
                                        ("channels", "SIRCL point-to-point teardown on rank")])
def test_a_close_paired_with_another_objects_close_fails_on_both_ranks(closes, kind, what):
    case = closes[f"{kind} other object"]
    assert case["results"] == [f"{what} 0 round 1: rank 1 closed {kind} 0 in round 1 while this rank closed {kind} 1 "
                               "in round 1",
                               f"{what} 1 round 1: rank 0 closed {kind} 1 in round 1 while this rank closed {kind} 0 "
                               "in round 1"]
    assert case["entered"] == [1, 1] and case["unusable"] in case["results"]
    assert case["steps"] == [["stop", "destroy"], ["stop", "destroy"]]


def test_a_failed_channel_stream_destroy_is_voted(closes):
    case = closes["channels stream destroy failed"]
    own = ("SIRCL point-to-point channels on rank 1: stream destruction failed: RuntimeError: cuStreamDestroy failed: "
           "CUresult.CUDA_ERROR_INVALID_HANDLE")
    assert case["results"] == [f"rank 1: {own}", own] and case["entered"] == [2, 2]
    assert case["retained"] == [False, False]                       # the stream's work was complete


@pytest.mark.parametrize("kind, what", [("session", "SIRCL session teardown on rank 1"),
                                        ("channels", "SIRCL point-to-point teardown on rank 1")])
def test_a_failed_destroy_keeps_the_arena(closes, kind, what):
    case = closes[f"{kind} destroy failed"]
    assert case["results"] == [None, f"{what}: 2 RDMA teardown call(s) failed; the registered arena stays "
                                     "allocated for the rest of the process"]
    assert case["retained"] == [False, True] and case["detached"] == [True, True]


@pytest.mark.parametrize("kind", ["session", "channels"])
def test_without_a_native_context_no_round_runs(closes, kind):
    case = closes[f"{kind} no native"]
    assert case["results"] == [None, None] and case["entered"] == [0, 0]


@pytest.mark.parametrize("kind, what", [("session", "SIRCL session teardown on rank"),
                                        ("channels", "SIRCL point-to-point teardown on rank")])
def test_an_abort_leaves_the_peer_waiting_one_bounded_round(closes, kind, what):
    case = closes[f"{kind} aborted"]
    assert case["results"][0] is None and case["steps"][0] == ["stop", "destroy"]
    assert case["results"][1] == f"{what} 1 round 1: not every rank arrived within 0.3 s"
    assert case["entered"] == [0, 1] and case["steps"][1] == ["stop", "destroy"]
    assert case["unusable"] == case["results"][1]
    later = closes[f"{kind} later close"]
    assert later["entered"] == [0, 0] and later["steps"] == [["stop", "destroy"], ["stop", "destroy"]]
    assert later["results"] == [f"{what} {rank}: closed without the teardown rounds ({case['results'][1]})"
                                for rank in range(2)]


@pytest.mark.parametrize("kind, what", [("session", "SIRCL session teardown on rank"),
                                        ("channels", "SIRCL point-to-point teardown on rank")])
def test_a_round_whose_thread_cannot_start_is_a_terminal_failure(closes, kind, what):
    case = closes[f"{kind} thread start failed"]
    # The first rank to fail marks the group unusable; the other may then close without rounds.
    missed = [f"{what} {rank} round 1: its thread did not start: RuntimeError: can't start new thread"
              for rank in range(2)]
    assert case["unusable"] in missed
    for rank, result in enumerate(case["results"]):
        assert result in (missed[rank], f"{what} {rank}: closed without the teardown rounds ({case['unusable']})")
    assert case["again"] == case["results"] == case["kept"]
    assert case["steps"] == [["stop", "destroy"], ["stop", "destroy"]] and case["entered"] == [0, 0]
    assert case["detached"] == [True, True]


@pytest.mark.parametrize("kind, what", [("session", "SIRCL session teardown on rank 1"),
                                        ("channels", "SIRCL point-to-point teardown on rank 1")])
def test_a_progress_thread_stop_that_raises_is_a_terminal_failure(closes, kind, what):
    case = closes[f"{kind} native stop raised"]
    assert case["results"][1] == f"{what}: the teardown raised RuntimeError: stop failed"
    assert case["results"][0].endswith("round 2: not every rank arrived within 0.3 s")    # rank 1 left round 2
    assert case["again"] == case["results"] == case["kept"]
    assert case["steps"] == [["stop", "destroy"], ["stop", "destroy"]] and case["detached"] == [True, True]


@pytest.mark.parametrize("kind, what", [("session", "SIRCL session teardown on rank 1"),
                                        ("channels", "SIRCL point-to-point teardown on rank 1")])
def test_a_destroy_that_raises_is_a_terminal_failure_and_keeps_the_arena(closes, kind, what):
    case = closes[f"{kind} native destroy raised"]
    assert case["results"] == [None, f"{what}: destroying the native context raised RuntimeError: destroy failed; "
                                     "the registered arena stays allocated for the rest of the process"]
    assert case["again"] == case["results"] == case["kept"] and case["retained"] == [False, True]
    assert case["steps"] == [["stop", "destroy"], ["stop", "destroy"]] and case["entered"] == [2, 2]


@pytest.mark.parametrize("kind, what", [("session", "SIRCL session teardown on rank 0"),
                                        ("channels", "SIRCL point-to-point teardown on rank 0")])
def test_an_interrupted_close_stays_failed_and_keeps_its_context_and_arena(closes, kind, what):
    case = closes[f"{kind} interrupted"]
    assert case["raised"] == "KeyboardInterrupt"
    assert case["kept"] == case["again"] == f"{what}: interrupted: KeyboardInterrupt: "
    assert case["steps"] == ["stop", "destroy"] and case["attached"] and case["retained"]


def test_notes_round_trip_through_a_two_rank_gloo_group(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with gloo is unavailable")
    store = tmp_path / "store"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    ranks = [subprocess.Popen([sys.executable, str(DRIVER), "gloo", str(rank), str(store)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env) for rank in range(2)]
    outputs = [rank.communicate(timeout=120) for rank in ranks]
    assert all(rank.returncode == 0 for rank in ranks), [err[-2000:] for _, err in outputs]
    for out, _ in outputs:
        seen = json.loads(out.strip().splitlines()[-1])
        assert seen["first"] == [["session", 3, 1, None], ["session", 3, 1, "rank 1 timed out"]]
        assert [note[:3] for note in seen["second"]] == [["channels", 1, 2]] * 2 and seen["second"][1][3] is None
        assert seen["second"][0][3].startswith("é" * 252)
        assert len(seen["second"][0][3].encode("utf-8")) <= teardown.NOTE_BYTES - 8 + 2


# -- the native destroys -------------------------------------------------------------------------------------

def test_the_session_destroy_counts_failed_verbs_calls(simulator_library):
    from sparkring_sircl.testing import fabric

    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("ring:3"), lanes=2)
    try:
        session.connect()
        fabric.run_oneshot_ops(session, [64, 4096], first_seq=1)
        proxies, session.proxies = session.proxies, []
        for proxy in proxies:
            proxy.stop()
        session.library.fv_fail_teardown(1)
        counts = [proxy.close() for proxy in proxies]
        session.library.fv_fail_teardown(0)
        assert counts == [1, 0, 0]
        assert proxies[0].close() == 0              # idempotent: the context is gone
    finally:
        session.close()


def test_the_channel_destroy_counts_failed_verbs_calls():
    from conftest import WORK, _require_compiler

    _require_compiler()
    from sparkring_sircl.testing import p2p_build
    from sparkring_sircl.testing.p2p_fabric import LocalChannels

    library = p2p_build.build_shared_library(WORK / ".build" / "sim")
    channels = LocalChannels(str(library), routes.Layout.parse("ring:2"), lanes=1, slots=4, slot_bytes=8192)
    contexts, channels.contexts = channels.contexts, []
    try:
        channels.fabric.lib.fv_fail_teardown(1)
        counts = [context.close() for context in contexts]
        channels.fabric.lib.fv_fail_teardown(0)
        assert counts == [1, 0]
    finally:
        channels.close()
