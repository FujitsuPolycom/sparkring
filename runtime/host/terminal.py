"""Optional terminal decoration; persistent progress remains plain text."""
import os
import re
import shutil
import threading
import time


class Console:
    def __init__(self, stream, *, plain=False):
        self.stream = stream
        self.enabled = not plain and self._interactive(stream)
        self.active = {}
        self.lock = threading.RLock()
        self.finished = threading.Event()
        self.thread = None
        self.drawn = False
        self.partial = False
        self.frame = 0
        try:
            "✓✗›⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏".encode(stream.encoding or "utf-8")
            self.frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            self.symbols = ("✓", "✗", "›")
        except (UnicodeError, LookupError):
            self.frames = "|/-\\"
            self.symbols = ("[ok]", "[!]", ">")

    @staticmethod
    def _interactive(stream):
        return stream.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ

    def __enter__(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.finished.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        with self.lock:
            self._clear()
            self.stream.flush()

    def begin(self, message, *, elapsed=True):
        token = object()
        with self.lock:
            self.active[token] = (message, time.monotonic() if elapsed else None)
        return token

    def end(self, token):
        with self.lock:
            self.active.pop(token, None)
            self._clear()
            self.stream.flush()

    def _clear(self):
        if self.drawn:
            self.stream.write("\r\033[2K")
            self.stream.flush()
            self.drawn = False

    def _color(self, text):
        result = []
        for line in text.splitlines(keepends=True):
            match = re.match(r"^(\d{4}-\d{2}-\d{2}T\S+  )(.*)", line)
            stamp = match[1] if match else ""
            message = line[len(stamp):]
            if message.startswith("Done:"):
                color, symbol = "32", self.symbols[0]
            elif message.startswith(("Error:", "Failed:", "Stopped:", "SparkRing:")):
                color, symbol = "31", self.symbols[1]
            elif message.startswith(("Still working:", "Warning:")):
                color, symbol = "33", self.symbols[2]
            elif message.startswith(("Node ", "SparkRing ")):
                color, symbol = "36", self.symbols[2]
            else:
                result.append(line)
                continue
            body = message.rstrip("\n")
            result.append(f"{stamp}\033[{color}m{symbol} {body}\033[0m" + ("\n" if message.endswith("\n") else ""))
        return "".join(result)

    def write(self, text, *, stream=None):
        target = self.stream if stream is None else stream
        with self.lock:
            self._clear()
            decorated = self.enabled and self._interactive(target)
            target.write(self._color(text) if decorated else text)
            target.flush()
            if text and (target is self.stream or decorated):
                self.partial = not text.endswith("\n")

    def draw(self):
        with self.lock:
            if not self.enabled or not self.active or self.partial:
                return
            message, started = next(iter(self.active.values()))
            elapsed = f" ({int(time.monotonic() - started)}s)" if started is not None else ""
            others = f" | {len(self.active)} active" if len(self.active) > 1 else ""
            width = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
            line = f"{self.frames[self.frame % len(self.frames)]} {message}{elapsed}{others}"[:width]
            self.stream.write(f"\r\033[2K\033[36m{line}\033[0m")
            self.stream.flush()
            self.drawn = True
            self.frame += 1

    def _animate(self):
        while not self.finished.wait(0.12):
            self.draw()
