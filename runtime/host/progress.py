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
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from runtime.host import colour
from runtime.host.privilege import reader_request, share, share_directory
from runtime.host.terminal import Console

EVENT_SCHEMA = "sparkring-install-event/v1"
# Seconds between a running step's "Still working" lines.
HEARTBEAT = 30
# The root installation log; members of the sparkring group read it (mode 0640,
# group sparkring, set by share), everyone else is told how to gain access.
LOG_ROOT = "/var/log/sparkring"

_log = None
_details = None
_console = None
_events = None
_operation_id = None
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
    document = {"schema": EVENT_SCHEMA, "time": _now(), "state": state, "label": colour.plain(label), "node": node,
                "phase": phase}
    if elapsed is not None:
        document["elapsed_s"] = round(elapsed, 1)
    document.update((key, colour.plain(value) if isinstance(value, str) else value)
                    for key, value in fields.items() if value is not None)
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
    return Path(LOG_ROOT) if hasattr(os, "geteuid") and os.geteuid() == 0 else Path.home() / ".local/state/sparkring/logs"


def record(message):
    """Write plain progress text; command highlighting is reconstructed on replay."""
    if _log is not None:
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with _lock:
            _log.write(stamp + "  " + colour.plain(message).rstrip() + "\n")
            _log.flush()


class Copies:
    """Write combined and operation logs from the same existing progress events."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def bind_slice(positions):
    if _operation_id is None:
        return
    from runtime.host import node
    key = "ring" if positions is None else "-".join(map(str, sorted(positions)))
    root = directory()
    node.save(root, "slice-" + key + ".json", {"operation": _operation_id}, mode=0o640)
    share(root / ("slice-" + key + ".json"), 0o640)


def operation_log():
    return directory() / (f"operation-{_operation_id}.log" if _operation_id else "install.log")


def say(message):
    stream = sys.stderr
    with _lock:
        # Under run() the stream is the Tee: it renders the marks for its
        # console and records the plain text. Alone, say renders itself.
        stream.write((message if isinstance(stream, Tee) else colour.render(message, stream)) + "\n")
        stream.flush()


def failure(message):
    if _details is not None:
        with _lock:
            _details.write(colour.plain(message) + "\n")
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
def run(operation, *, events=None, json_output=False):
    """Log progress of ``operation``; ``events`` is a path that receives the event stream, replacing its content."""
    global _log, _details, _console, _events, _operation_id
    root = directory()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Installation log directory contains a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    share_directory(root, 0o750)
    _operation_id = uuid.uuid4().hex
    paths = [root / "install.log", root / "install-details.log",
             operation_log(), root / f"operation-{_operation_id}-details.log"]
    if any(p.is_symlink() for p in paths):
        raise ValueError("Installation log file cannot be a symlink")
    with contextlib.ExitStack() as stack:
        primary = stack.enter_context(paths[0].open("a", encoding="utf-8", buffering=1, newline="\n"))
        details = stack.enter_context(paths[1].open("a", encoding="utf-8", buffering=1))
        own = stack.enter_context(paths[2].open("a", encoding="utf-8", buffering=1))
        own_details = stack.enter_context(paths[3].open("a", encoding="utf-8", buffering=1))
        for path in paths:
            path.chmod(0o600)
            # The sparkring group follows the installation progress; the log
            # carries no key or token, but stays closed to other users.
            share(path, 0o640)
        _log, _details = Copies(primary, own), Copies(details, own_details)
        try:
            if events is not None:
                _events = stack.enter_context(open(events, "w", encoding="utf-8", buffering=1))
            with Console(sys.stderr) as _console:
                with contextlib.redirect_stdout(Tee(sys.stdout, _console)), contextlib.redirect_stderr(Tee(sys.stderr, _console)):
                    output = sys.stderr if json_output else sys.stdout
                    print(f"SparkRing {operation}. Progress: {paths[2]}", flush=True, file=output)
                    print("Follow from another terminal: " + colour.cmd("sudo sparkring logs --follow"), flush=True, file=output)
                    emit("start", "SparkRing " + operation, phase=operation)
                    yield
        finally:
            _log = _details = _console = _events = _operation_id = None


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
        _details.write("COMMAND " + colour.plain(title) + "\n")
        _details.flush()
        destination = _details.streams[-1] if isinstance(_details, Copies) else _details
        offset = destination.tell()
        try:
            return invoke(argv, stdout=destination, stderr=destination, **kwargs)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(title + "; see " + str(directory() / "install-details.log")) from error
        finally:
            if isinstance(_details, Copies):
                # The child writes directly to its own file descriptor. Copy
                # that completed segment to the compatibility log on return;
                # no tee process or background reader is needed.
                with open(destination.name, encoding="utf-8", errors="replace") as source:
                    source.seek(offset)
                    while block := source.read(65536):
                        _details.streams[0].write(block)
                _details.streams[0].flush()


def unreadable(path: Path) -> bool:
    """Whether ``path`` exists and this process may neither read nor enter it."""
    try:
        path.stat()
    except OSError:
        return False
    return not os.access(path, os.R_OK | os.X_OK)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring logs", parents=[colour.parser()])
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--plain", action="store_true", help="disable terminal colors and animation")
    parser.add_argument("--lines", type=int, default=40)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--on", help="latest operation on these fabric positions")
    select.add_argument("--operation", help="operation identifier printed at start")
    args = parser.parse_args(argv)
    colour.set_mode(args.color)
    if args.plain or args.details:
        colour.set_mode("never")
    root = Path(os.environ.get("SPARKRING_LOG_DIR", LOG_ROOT))
    path = root / ("install-details.log" if args.details else "install.log")
    ident = args.operation
    if args.on:
        from runtime.common import fabric_layout
        from runtime.host import placement
        positions = placement.parse(args.on, fabric_layout.layout("cycle", 8))
        key = "ring" if positions is None else "-".join(map(str, sorted(positions)))
        index = root / ("slice-" + key + ".json")
        if not index.exists():
            print("No operation log for these Sparks.")
            return 0
        ident = json.loads(index.read_text(encoding="utf-8"))["operation"]
    if ident:
        if not re.fullmatch(r"[0-9a-f]{32}", ident):
            parser.error("--operation must be a recorded operation identifier")
        path = root / (f"operation-{ident}" + ("-details" if args.details else "") + ".log")
    if not path.exists() and root == Path(LOG_ROOT) and unreadable(root):
        print("The installation log is root's: " + reader_request() + ".", file=sys.stderr)
        return 2
    if not path.exists():
        root = Path.home() / ".local/state/sparkring/logs"
        path = root / path.name
    if not path.exists():
        print(colour.render("No installation log yet. Run " + colour.cmd("sparkring setup") + " or up first.",
                            sys.stdout))
        return 0
    if not 1 <= args.lines <= 10000:
        parser.error("--lines must be between 1 and 10000")
    try:
        with path.open(encoding="utf-8") as stream, Console(sys.stdout, plain=args.plain or args.details) as console:
            from collections import deque
            replay = colour.CardReplay()
            def replay_line(line):
                return line if args.details else replay.line(line)
            for line in deque((replay_line(line) for line in stream), maxlen=args.lines):
                console.write(line)
            if args.follow:
                console.begin("Waiting for new installation output", elapsed=False)
            while args.follow:
                line = stream.readline()
                if line:
                    console.write(replay_line(line))
                else:
                    time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    except PermissionError:
        print("The installation log is root's: " + reader_request() + ".", file=sys.stderr)
        return 2
    return 0
