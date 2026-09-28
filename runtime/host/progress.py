"""Readable, persistent setup/deployment progress with separate command details.

An optional event stream (``run(..., events=PATH)``) receives one
``sparkring-install-event/v1`` JSON object per line for every step's start,
periodic report, end and failure; see ``emit`` for its fields.
"""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from runtime.host.terminal import Console

EVENT_SCHEMA = "sparkring-install-event/v1"
# Seconds between a running step's "Still working" lines.
HEARTBEAT = 30

_log = None
_details = None
_console = None
_events = None
_lock = threading.RLock()


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def emit(state, label, *, node=None, phase=None, elapsed=None, **fields):
    """Write one event line when an event stream is open.

    Every event has ``schema``, ``time`` (ISO 8601 with offset), ``state``,
    ``label`` (the human progress text), ``node`` (the Spark's rank, or null
    for work that is not one Spark's) and ``phase`` (a stable step key such as
    ``ready`` or ``model-fetch``, or null). ``elapsed_s`` is present once a
    step has started. ``fields`` adds optional keys, for example ``detail``,
    ``bytes_done``, ``bytes_total``, ``rate_bps``, ``percent``, ``eta_s`` or
    ``message``; keys whose value is None are left out.
    """
    if _events is None:
        return
    document = {"schema": EVENT_SCHEMA, "time": _now(), "state": state, "label": label, "node": node, "phase": phase}
    if elapsed is not None:
        document["elapsed_s"] = round(elapsed, 1)
    document.update((key, value) for key, value in fields.items() if value is not None)
    with _lock:
        try:
            _events.write(json.dumps(document) + "\n")
            _events.flush()
        except (OSError, ValueError):
            pass  # A closed or failing event reader never stops the installation.


def node_of(message):
    """The Spark rank a progress label names with its ``Node N:`` prefix, or None."""
    match = re.match(r"Node (\d+): ", message)
    return int(match.group(1)) if match else None


def directory():
    configured = os.environ.get("SPARKRING_LOG_DIR")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError("SPARKRING_LOG_DIR must be absolute")
        return path
    return Path("/var/log/sparkring") if hasattr(os, "geteuid") and os.geteuid() == 0 else Path.home() / ".local/state/sparkring/logs"


def record(message):
    if _log is not None:
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with _lock:
            _log.write(stamp + "  " + message.rstrip() + "\n")
            _log.flush()


def say(message):
    with _lock:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()


def failure(message):
    if _details is not None:
        with _lock:
            _details.write(message + "\n")
            _details.flush()
    lines = [line for line in message.splitlines() if line.strip()]
    text = lines[-1][:500] if lines else "Operation failed"
    say("Error: " + text)
    emit("error", "Error", message=text)


class Tee:
    def __init__(self, stream, console):
        self.stream = stream
        self.console = console
        self.partial = ""

    def write(self, text):
        with _lock:
            self.console.write(text, stream=self.stream)
            self.partial += text
            while "\n" in self.partial:
                line, self.partial = self.partial.split("\n", 1)
                if line.strip():
                    record(line)
        return len(text)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    @property
    def encoding(self):
        return self.stream.encoding


@contextlib.contextmanager
def run(operation, *, events=None):
    """Log progress of ``operation``; ``events`` is a path that receives the event stream, replacing its content."""
    global _log, _details, _console, _events
    root = directory()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Installation log directory contains a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    paths = [root / "install.log", root / "install-details.log"]
    if any(p.is_symlink() for p in paths):
        raise ValueError("Installation log file cannot be a symlink")
    with contextlib.ExitStack() as stack:
        primary = stack.enter_context(paths[0].open("a", encoding="utf-8", buffering=1))
        details = stack.enter_context(paths[1].open("a", encoding="utf-8", buffering=1))
        for path in paths:
            path.chmod(0o600)
        _log, _details = primary, details
        try:
            if events is not None:
                _events = stack.enter_context(open(events, "w", encoding="utf-8", buffering=1))
            with Console(sys.stderr) as _console:
                with contextlib.redirect_stdout(Tee(sys.stdout, _console)), contextlib.redirect_stderr(Tee(sys.stderr, _console)):
                    print(f"SparkRing {operation}. Progress: {paths[0]}", flush=True)
                    print("Follow from another terminal: sudo sparkring logs --follow", flush=True)
                    emit("start", "SparkRing " + operation, phase=operation)
                    yield
        finally:
            _log = _details = _console = _events = None


def size_text(done, total):
    """``121 of 166 GiB``: both byte counts in the unit that suits ``total``."""
    unit, scale = ("GiB", 1024 ** 3) if total >= 1024 ** 3 else ("MiB", 1024 ** 2)
    digits = 0 if total >= 10 * scale else 1
    return f"{done / scale:.{digits}f} of {total / scale:.{digits}f} {unit}"


def rate_text(bytes_per_second):
    """A transfer rate in bits per second, as network links are rated: ``850 Mb/s`` or ``1.2 Gb/s``."""
    bits = bytes_per_second * 8
    return f"{bits / 1e9:.1f} Gb/s" if bits >= 1e9 else f"{bits / 1e6:.0f} Mb/s"


def time_left(seconds):
    """``about 7 min left``, rounded to whole minutes; hours above one hour."""
    minutes = max(1, round(seconds / 60))
    if seconds < 60:
        return "less than a minute left"
    if minutes < 60:
        return f"about {minutes} min left"
    return f"about {minutes // 60} h {minutes % 60} min left"


class Meter:
    """A step report for a transfer of ``total`` bytes; ``measure()`` returns the bytes done so far.

    Each call measures once. The rate is the change since the previous call
    (since construction for the first), so it follows the recent speed rather
    than the average. The ``detail`` reads ``121 of 166 GiB, 850 Mb/s, about 7
    min left``; rate and time left are omitted until bytes move.
    """

    def __init__(self, total, measure, *, clock=time.monotonic):
        self.total, self.measure, self.clock = total, measure, clock
        self.last = (clock(), min(measure(), total))

    def __call__(self, elapsed=None):
        now, done = self.clock(), min(self.measure(), self.total)
        then, before = self.last
        self.last = (now, done)
        rate = (done - before) / (now - then) if now > then and done > before else 0
        fields = {"bytes_done": done, "bytes_total": self.total,
                  "percent": round(100 * done / self.total, 1) if self.total else 100.0}
        parts = [size_text(done, self.total)]
        if rate > 0:
            eta = (self.total - done) / rate
            fields.update(rate_bps=int(rate * 8), eta_s=int(eta))
            parts += [rate_text(rate), time_left(eta)]
        fields["detail"] = ", ".join(parts)
        return fields


def _report(report, elapsed):
    """The fields ``report(elapsed)`` returns, or an empty mapping when it has none or fails."""
    if report is None:
        return {}
    try:
        fields = report(elapsed)
    except Exception:  # noqa: BLE001 - a report is optional; the step itself decides success
        return {}
    if not isinstance(fields, dict):
        return {}
    return {key: value for key, value in fields.items() if key not in ("node", "phase", "elapsed", "state", "label")}


@contextlib.contextmanager
def step(message, *, phase=None, report=None):
    """Show ``message`` while its block runs, with a "Still working" line every ``HEARTBEAT`` seconds.

    ``phase`` is the stable key recorded in events. ``report(elapsed)``, when
    given, runs once per heartbeat and returns a mapping of event fields; its
    ``detail`` text is appended to the "Still working" line. A report that
    fails or returns nothing leaves the line as it is.
    """
    say(message)
    console = _console
    activity = console.begin(message) if console is not None else None
    outcome = {"failed": False}
    finished = threading.Event()
    started = time.monotonic()
    where = {"node": node_of(message), "phase": phase}
    emit("start", message, elapsed=0.0, **where)

    def heartbeat():
        while not finished.wait(HEARTBEAT):
            elapsed = time.monotonic() - started
            fields = _report(report, elapsed)
            if finished.is_set():
                break
            detail = fields.get("detail")
            say(f"Still working: {message} ({int(elapsed)}s)" + (f" - {detail}" if detail else ""))
            emit("working", message, elapsed=elapsed, **where, **fields)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield outcome
    except BaseException:
        say("Failed: " + message)
        emit("failed", message, elapsed=time.monotonic() - started, **where)
        raise
    else:
        elapsed = time.monotonic() - started
        say(("Stopped: " if outcome["failed"] else "Done: ") + f"{message} ({elapsed:.1f}s)")
        emit("failed" if outcome["failed"] else "done", message, elapsed=elapsed, **where)
    finally:
        finished.set()
        thread.join(timeout=1)
        if console is not None:
            console.end(activity)


def command(argv, *, title, invoke=subprocess.run, **kwargs):
    if _details is None:
        return invoke(argv, **kwargs)
    with step(title):
        _details.write("COMMAND " + title + "\n")
        _details.flush()
        try:
            return invoke(argv, stdout=_details, stderr=_details, **kwargs)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(title + "; see " + str(directory() / "install-details.log")) from error


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring logs")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--plain", action="store_true", help="disable terminal colors and animation")
    parser.add_argument("--lines", type=int, default=40)
    args = parser.parse_args(argv)
    root = Path(os.environ.get("SPARKRING_LOG_DIR", "/var/log/sparkring"))
    path = root / ("install-details.log" if args.details else "install.log")
    if not path.exists():
        root = Path.home() / ".local/state/sparkring/logs"
        path = root / path.name
    if not path.exists():
        print("No installation log yet. Run sparkring setup or up first.")
        return 0
    if not 1 <= args.lines <= 10000:
        parser.error("--lines must be between 1 and 10000")
    try:
        with path.open(encoding="utf-8") as stream, Console(sys.stdout, plain=args.plain or args.details) as console:
            from collections import deque
            for line in deque(stream, maxlen=args.lines):
                console.write(line)
            if args.follow:
                console.begin("Waiting for new installation output", elapsed=False)
            while args.follow:
                line = stream.readline()
                if line:
                    console.write(line)
                else:
                    time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    except PermissionError:
        print("Use sudo sparkring logs to read the installation log.", file=sys.stderr)
        return 2
    return 0
