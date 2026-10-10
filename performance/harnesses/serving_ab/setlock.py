"""Per-Spark-set locks beside the ring's global measurement lock.

The global lock (``hwlock.sh``: the directory ``/tmp/ring8-hw.lock`` on the lock host, its ``owner`` file
``NAME EPOCH``) serialises every GPU or fabric load on the ring. A runner on a subset of Sparks takes a set
lock instead: one directory per Spark position, ``/tmp/ring8-sets/<position>`` on the same lock host, each
with an ``owner`` file in the same format. Acquisition is one shell script on the lock host that creates
every directory of the set with ``mkdir`` (atomic), and removes the ones it created if any already exists,
so two runners never hold overlapping sets. It also waits while the global lock is held by anyone.

The rules, so that set locks and the global lock exclude each other:
- a set lock waits while ``/tmp/ring8-hw.lock`` exists (unless its owner is stale);
- the global lock must wait while any ``/tmp/ring8-sets/<position>`` exists. hwlock.sh does not check
  this; until it does, a global-lock holder can start beside a set lock.
- A lock whose owner time is older than ``STALE_SECONDS`` is abandoned and may be broken. Holders
  rewrite their owner time every ``HEARTBEAT_SECONDS`` (:class:`SetLock` does it from a thread).
"""

from __future__ import annotations

import shlex
import threading
import time
from collections.abc import Sequence

from . import remote

GLOBAL = "/tmp/ring8-hw.lock"
SETS = "/tmp/ring8-sets"
STALE_SECONDS = 2700
HEARTBEAT_SECONDS = 300


def acquire_script(name: str, positions: Sequence[int], stale: int = STALE_SECONDS, *, global_dir: str = GLOBAL,
                   sets_dir: str = SETS) -> str:
    """Shell that prints ACQUIRED, or BUSY and the holders, without leaving partial state."""
    GLOBAL, SETS = global_dir, sets_dir  # noqa: N806 - the paths the script names
    dirs = " ".join(f"{SETS}/{p}" for p in sorted(positions))
    q = shlex.quote(name)
    return (
        f"now=$(date +%s); mkdir -p {SETS}; "
        f"if [ -d {GLOBAL} ]; then read -r o t < {GLOBAL}/owner 2>/dev/null; "
        f"  if [ -n \"$t\" ] && [ $((now - t)) -gt {stale} ]; then rm -rf {GLOBAL}; "
        f"  else echo \"BUSY global ${{o:-unknown}}\"; exit 0; fi; fi; "
        f"for d in {dirs}; do if [ -d $d ]; then read -r o t < $d/owner 2>/dev/null; "
        f"  if [ -n \"$t\" ] && [ $((now - t)) -gt {stale} ]; then rm -rf $d; fi; fi; done; "
        f"made=''; for d in {dirs}; do if mkdir $d 2>/dev/null; then echo {q} $now > $d/owner; made=\"$made $d\"; "
        f"  else for m in $made; do rm -rf $m; done; read -r o t < $d/owner 2>/dev/null; "
        f"    echo \"BUSY $d ${{o:-unknown}}\"; exit 0; fi; done; echo ACQUIRED")


def refresh_script(name: str, positions: Sequence[int], *, sets_dir: str = SETS) -> str:
    SETS = sets_dir  # noqa: N806
    q = shlex.quote(name)
    return "; ".join(f"read -r o t < {SETS}/{p}/owner 2>/dev/null; [ \"$o\" = {q} ] && echo {q} $(date +%s) > "
                     f"{SETS}/{p}/owner" for p in sorted(positions)) + "; true"


def release_script(name: str, positions: Sequence[int], *, sets_dir: str = SETS) -> str:
    SETS = sets_dir  # noqa: N806
    q = shlex.quote(name)
    return "; ".join(f"read -r o t < {SETS}/{p}/owner 2>/dev/null; [ \"$o\" = {q} ] && rm -rf {SETS}/{p}"
                     for p in sorted(positions)) + "; true"


class SetLock:
    """Holds the set lock of ``positions`` on ``lock_host`` (a :class:`remote.Spark`) as ``name``."""

    def __init__(self, lock_host: remote.Spark, name: str, positions: Sequence[int], *, wait_s: float = 3600,
                 log=print):
        self.host, self.name, self.positions, self.wait_s, self.log = lock_host, name, tuple(positions), wait_s, log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        deadline = time.monotonic() + self.wait_s
        while True:
            answer = remote.run(self.host, acquire_script(self.name, self.positions)).strip()
            if answer.endswith("ACQUIRED"):
                break
            if time.monotonic() > deadline:
                raise remote.RemoteError(f"set lock {self.name}: still {answer} after {self.wait_s:.0f} s")
            self.log(f"set lock {self.name}: waiting ({answer})")
            time.sleep(15)
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        self.log(f"set lock {self.name} holds Sparks {list(self.positions)}")
        return self

    def _heartbeat(self):
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                remote.run(self.host, refresh_script(self.name, self.positions), check=False)
            except Exception:  # a missed refresh is retried at the next beat
                pass

    def __exit__(self, *exc):
        self._stop.set()
        remote.run(self.host, release_script(self.name, self.positions), check=False)
        self.log(f"set lock {self.name} released")
        return False


class Held:
    """No lock of its own: the caller holds the global lock (``bash hwlock.sh OWNER COMMAND``)."""

    def __init__(self, log=print):
        self.log = log

    def __enter__(self):
        self.log("lock: the caller holds the ring's global lock")
        return self

    def __exit__(self, *exc):
        return False
