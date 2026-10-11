"""Dashboard quiet mode around a measurement: no request from ``sparkring dashboard`` while load runs.

The SparkRing dashboard on Node A (``runtime/host/dashboard.py``) reads each model's ``/metrics``
every few seconds. A measurement entry point wraps its load in ``held(label, ...)``:

- before the load it asks ``sparkring dashboard quiet hold`` for a holder of its own, on this host
  or, with ``ssh``, on Node A through the harness's own SSH client and target;
- after the load, also after an error, Ctrl-C or SIGTERM, it releases that holder. Quiet mode
  then returns to its previous state: it stays on while the operator (``quiet on``) or another
  measurement holds it, and the dashboard drops a holder on its own when its process exits or its
  time to live (``ttl``) ends, so a harness killed outright does not silence the page for good.

``held`` yields the record field DASHBOARD_FIELD, which every result record of the run carries:

- ``state``: ``on`` (quiet mode held during the load), ``dashboard absent`` (no SparkRing package,
  or a package without the dashboard, on that host) or ``not held`` (the hold failed; ``detail``
  says why, for example that ``sudo -n`` needs a password); the load runs in every case;
- ``where``: ``this host`` or the Spark's name; ``holder``: the holder id; ``was_on_before``:
  whether quiet mode was already on; ``service``: ``systemctl is-active`` of the dashboard unit.

The hold costs one command (one SSH login with ``ssh``) before the load and one after it, never
during it. While a block holds quiet mode, INHERITED in the environment carries its field to the
programs the harness starts, so a child measurement records the parent's hold and holds nothing
itself. SKIP=skip in the environment turns holding off (the harness tests set it, so a test run
on a Spark never silences its dashboard). This module uses only the Python standard library.

Entry points take ``--dashboard-ssh TARGET`` (``add_argument``): Node A's SSH target, defaulting
to the environment variable TARGET_VARIABLE. Without one, the hold is asked of this host.
"""
from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import threading

DASHBOARD_FIELD = "dashboard_quiet"
SPARKRING = "/usr/bin/sparkring"
UNIT = "sparkring-dashboard.service"
TTL_SECONDS = 12 * 3600
ABSENT = "SPARKRING-ABSENT"
INHERITED = "SPARKRING_DASHBOARD_QUIET_HELD"
SKIP = "SPARKRING_DASHBOARD_QUIET"
TARGET_VARIABLE = "SPARKRING_DASHBOARD_SSH"
SSH_CLIENT = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")


def add_argument(parser) -> None:
    """``--dashboard-ssh TARGET`` on an entry point's argument parser."""
    parser.add_argument("--dashboard-ssh", metavar="TARGET", default=os.environ.get(TARGET_VARIABLE),
                        help="Node A's SSH target, to hold its dashboard's quiet mode while this measures "
                             f"(default ${TARGET_VARIABLE}; without one, the hold is asked of this host)")


def ssh_for(target, client=SSH_CLIENT):
    """The ``ssh`` argument of ``held`` for an SSH target, or None for this host."""
    return [*client, target] if target else None


def _script(arguments) -> str:
    """The shell script that runs ``sparkring dashboard quiet ...`` with sudo -n and reports the unit."""
    command = shlex.join(["sudo", "-n", SPARKRING, "dashboard", "quiet", *arguments])
    return (f"if [ ! -x {SPARKRING} ]; then echo {ABSENT}; exit 0; fi; "
            f"echo \"@service $(systemctl is-active {UNIT} 2>/dev/null || true)\"; {command} 2>&1")


def _run(script, *, ssh, runner, timeout):
    """(exit code, output) of the script, here (``ssh`` None) or through ``ssh``: the SSH command and target."""
    argv = [*ssh, script] if ssh else ["sh", "-c", script]
    try:
        done = runner(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{type(error).__name__}: {error}"
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def _parse(output):
    lines = [line for line in (output or "").splitlines() if line.strip()]
    service = next((line[len("@service "):].strip() or None for line in lines if line.startswith("@service ")), None)
    body = [line for line in lines if not line.startswith("@service ")]
    document = None
    for line in reversed(body):
        try:
            document = json.loads(line)
            break
        except ValueError:
            continue
    return service, body, document


def hold(label, *, ssh=None, where=None, ttl=TTL_SECONDS, runner=subprocess.run, timeout=60) -> dict:
    """Ask for a quiet holder; returns the record field. Never raises: a failure is recorded."""
    inherited = os.environ.get(INHERITED)
    if inherited:
        try:
            return {**json.loads(inherited), "inherited": True}
        except ValueError:
            pass
    if os.environ.get(SKIP) == "skip":
        return {"state": "not held", "where": where or ("this host" if ssh is None else "Node A"), "holder": None,
                "was_on_before": None, "service": None, "detail": f"{SKIP}=skip"}
    arguments = ["hold", "--label", label, "--ttl", str(int(ttl))]
    if ssh is None:
        arguments += ["--pid", str(os.getpid())]
    field = {"state": "not held", "where": where or ("this host" if ssh is None else "Node A"), "holder": None,
             "was_on_before": None, "service": None, "detail": None}
    code, output = _run(_script(arguments), ssh=ssh, runner=runner, timeout=timeout)
    if code is None:
        field["detail"] = _private(output, ssh)
        return field
    if ABSENT in (output or ""):
        return {**field, "state": "dashboard absent", "detail": f"no SparkRing package ({SPARKRING})"}
    service, body, document = _parse(output)
    field["service"] = service
    if isinstance(document, dict) and document.get("holder"):
        return {**field, "state": "on", "holder": document["holder"], "was_on_before": bool(document.get("was_quiet"))}
    text = " ".join(body)[-300:]
    if "invalid choice: 'dashboard'" in text or "invalid choice: 'hold'" in text:
        return {**field, "state": "dashboard absent", "detail": "the SparkRing package has no dashboard quiet hold"}
    field["detail"] = _private(text or f"exit code {code}", ssh)
    return field


def _private(text, ssh):
    """``text`` with Node A's SSH target and host replaced, since records may be published."""
    if ssh:
        target = ssh[-1]
        for value in (target, target.rpartition("@")[2]):
            if value:
                text = text.replace(value, "NODE_A")
    return text


def release(field, *, ssh=None, runner=subprocess.run, timeout=60) -> dict:
    """Release the holder ``hold`` returned; returns the field with ``released``. Never raises."""
    if field.get("state") != "on" or not field.get("holder") or field.get("inherited"):
        return field
    code, output = _run(_script(["release", field["holder"]]), ssh=ssh, runner=runner, timeout=timeout)
    _, _, document = _parse(output)
    released = isinstance(document, dict) and document.get("released") == field["holder"]
    return {**field, "released": released,
            **({} if released else {"release_detail": (output or f"exit code {code}").strip()[-300:]})}


@contextlib.contextmanager
def held(label, *, ssh=None, where=None, ttl=TTL_SECONDS, runner=subprocess.run, record=None):
    """Hold quiet mode while the block runs and yield the record field.

    ``record`` is an optional dict that receives the field under DASHBOARD_FIELD before the
    block runs and again with ``released`` after it. SIGTERM in the main thread is turned into
    SystemExit for the block's duration, so the release runs as it does on Ctrl-C."""
    field = hold(label, ssh=ssh, where=where, ttl=ttl, runner=runner)
    if record is not None:
        record[DASHBOARD_FIELD] = field
    before = os.environ.get(INHERITED)
    if field["state"] == "on" and not field.get("inherited"):
        os.environ[INHERITED] = json.dumps(field)
    previous = None
    if threading.current_thread() is threading.main_thread():
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield field
    finally:
        if before is None:
            os.environ.pop(INHERITED, None)
        else:
            os.environ[INHERITED] = before
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        final = release(field, ssh=ssh, runner=runner)
        field.update(final)
        if record is not None:
            record[DASHBOARD_FIELD] = field
