"""An operator hold for a whole acceptance run: other operators' changes refuse while it runs.

``sparkring hold`` (``runtime/host/hold.py``) records on Node A that one operator holds the
cluster; while it stands, other logins' ``install``, ``up``, ``down`` and similar commands refuse
and name the holder. An entry point wraps its run in ``held(reason, ...)``:

- before the run it asks ``sparkring hold --json`` whether a hold stands. When none does, it
  takes one (``sudo -n sparkring hold --reason REASON --until 12h``), on this host or, with
  ``ssh``, on Node A through the harness's own SSH client and target. The holder is the login
  that runs ``sudo``, so commands the harness runs under the same login are not refused.
- when a hold already stands, by this login or another, it takes none and releases none, so an
  operator's own hold outlives the run. Another login's hold also refuses the harness's own
  ``sparkring install``; this module never overrides one.
- after the run, also after an error, Ctrl-C or SIGTERM, it releases the hold it took, unless the
  standing hold is no longer that one (replaced since). A harness killed outright leaves its hold
  until ``--until`` passes; ``sudo sparkring hold --release`` under the same login clears it.

``held`` yields the record field HOLD_FIELD, which every result record of the run carries: ``state`` is ``held`` (this run took the hold),
``already held`` (``holder`` and ``reason`` name the standing hold), ``unsupported`` (no SparkRing
package, or one without ``sparkring hold``, on that host) or ``not held`` (``detail`` says why, for
example that ``sudo -n`` needs a password). The run goes ahead in every case. SKIP=skip in the
environment turns holding off; the harness tests set it. Standard library only.
"""
from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import threading

from performance.harnesses.dashboard_quiet import SPARKRING

HOLD_FIELD = "cluster_hold"
UNTIL = "12h"
ABSENT = "SPARKRING-ABSENT"
SKIP = "SPARKRING_CLUSTER_HOLD"


def _script(arguments, *, sudo) -> str:
    command = shlex.join([*(("sudo", "-n") if sudo else ()), SPARKRING, "hold", *arguments, "--json"])
    return f"if [ ! -x {SPARKRING} ]; then echo {ABSENT}; exit 0; fi; {command} 2>&1"


def _run(arguments, *, sudo, ssh, runner, timeout):
    """(exit code, last JSON document or None, output) of one ``sparkring hold`` call."""
    script = _script(arguments, sudo=sudo)
    argv = [*ssh, script] if ssh else ["sh", "-c", script]
    try:
        done = runner(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return None, None, f"{type(error).__name__}: {error}"
    output = (done.stdout or "") + (done.stderr or "")
    document = None
    for line in reversed([line for line in output.splitlines() if line.strip()]):
        try:
            document = json.loads(line)
            break
        except ValueError:
            continue
    return done.returncode, document if isinstance(document, dict) else None, output


def _private(text, ssh):
    """``text`` with Node A's SSH target and host replaced, since records may be published."""
    if ssh:
        target = ssh[-1]
        for value in (target, target.rpartition("@")[2]):
            if value:
                text = text.replace(value, "NODE_A")
    return text.strip()[-300:]


def _unsupported(output):
    return ABSENT in (output or "") or "invalid choice: 'hold'" in (output or "")


def take(reason, *, ssh=None, where=None, until=UNTIL, runner=subprocess.run, timeout=60) -> dict:
    """Take a hold when none stands; returns the record field. Never raises: a failure is recorded."""
    field = {"state": "not held", "where": where or ("this host" if ssh is None else "Node A"),
             "holder": None, "reason": None, "since": None, "detail": None}
    if os.environ.get(SKIP) == "skip":
        return {**field, "detail": f"{SKIP}=skip"}
    code, document, output = _run([], sudo=True, ssh=ssh, runner=runner, timeout=timeout)
    if code is not None and _unsupported(output):
        return {**field, "state": "unsupported", "detail": "no SparkRing package with sparkring hold"}
    if code != 0 or document is None:
        return {**field, "detail": _private(output or f"exit code {code}", ssh)}
    standing = document.get("hold")
    if isinstance(standing, dict) and standing:
        return {**field, "state": "already held", "holder": standing.get("holder"),
                "reason": standing.get("reason"), "since": standing.get("since")}
    code, document, output = _run(["--reason", reason, "--until", until], sudo=True, ssh=ssh, runner=runner,
                                  timeout=timeout)
    record = (document or {}).get("hold") if code == 0 else None
    if not isinstance(record, dict) or not record:
        return {**field, "detail": _private(output or f"exit code {code}", ssh)}
    return {**field, "state": "held", "holder": record.get("holder"), "reason": record.get("reason"),
            "since": record.get("since")}


def release(field, *, ssh=None, runner=subprocess.run, timeout=60) -> dict:
    """Release the hold ``take`` took, if it still stands; returns the field with ``released``. Never raises."""
    if field.get("state") != "held":
        return field
    code, document, output = _run([], sudo=True, ssh=ssh, runner=runner, timeout=timeout)
    standing = (document or {}).get("hold") if code == 0 else None
    if not isinstance(standing, dict) or standing.get("holder") != field["holder"] or standing.get("since") != field["since"]:
        detail = "the hold was replaced, released or expired meanwhile" if code == 0 else _private(output, ssh)
        return {**field, "released": False, "release_detail": detail}
    code, document, output = _run(["--release"], sudo=True, ssh=ssh, runner=runner, timeout=timeout)
    released = code == 0 and (document or {}).get("state") == "released"
    return {**field, "released": released,
            **({} if released else {"release_detail": _private(output or f"exit code {code}", ssh)})}


@contextlib.contextmanager
def held(reason, *, ssh=None, where=None, until=UNTIL, runner=subprocess.run):
    """Hold the cluster while the block runs and yield the record field.

    SIGTERM in the main thread is turned into SystemExit for the block's duration, so the release
    runs as it does on Ctrl-C."""
    field = take(reason, ssh=ssh, where=where, until=until, runner=runner)
    previous = None
    if threading.current_thread() is threading.main_thread():
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield field
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        field.update(release(field, ssh=ssh, runner=runner))
