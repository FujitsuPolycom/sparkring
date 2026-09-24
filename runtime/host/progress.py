"""Readable, persistent setup/deployment progress with separate command details."""
import argparse
import contextlib
import datetime
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from runtime.host.terminal import Console

_log = None
_details = None
_console = None
_lock = threading.RLock()


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
    say("Error: " + (lines[-1][:500] if lines else "Operation failed"))


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
def run(operation):
    global _log, _details, _console
    root = directory()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Installation log directory contains a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    paths = [root / "install.log", root / "install-details.log"]
    if any(p.is_symlink() for p in paths):
        raise ValueError("Installation log file cannot be a symlink")
    with paths[0].open("a", encoding="utf-8", buffering=1) as primary, paths[1].open("a", encoding="utf-8", buffering=1) as details:
        for path in paths:
            path.chmod(0o600)
        _log, _details = primary, details
        try:
            with Console(sys.stderr) as _console:
                with contextlib.redirect_stdout(Tee(sys.stdout, _console)), contextlib.redirect_stderr(Tee(sys.stderr, _console)):
                    print(f"SparkRing {operation}. Progress: {paths[0]}", flush=True)
                    print("Follow from another terminal: sudo sparkring logs --follow", flush=True)
                    yield
        finally:
            _log = _details = _console = None


@contextlib.contextmanager
def step(message):
    say(message)
    console = _console
    activity = console.begin(message) if console is not None else None
    outcome = {"failed": False}
    finished = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not finished.wait(30):
            say(f"Still working: {message} ({int(time.monotonic() - started)}s)")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield outcome
    except BaseException:
        say("Failed: " + message)
        raise
    else:
        say(("Stopped: " if outcome["failed"] else "Done: ") + f"{message} ({time.monotonic() - started:.1f}s)")
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
