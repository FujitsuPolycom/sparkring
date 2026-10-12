"""The privileged helper: one narrow, audited root path for the mutating commands.

Mutating commands (``up``, ``down``, ``recover``, ``install``) change the
host or start serving, so they run as root today, through the operator's
sudo. The helper replaces that blanket sudo with a root systemd service
(``sparkring-helper.service``, socket-activated on
``/run/sparkring/helper.sock``, group ``sparkring``, mode 0660):

- A member of the ``sparkring`` group connects and sends one JSON request
  line: ``{"argv": [...]}``.
- The service reads the peer's credentials from the kernel (root or a member
  of the ``sparkring`` group), validates the argv against a fixed allowlist
  (``COMMANDS``) — no shell, no free-form command — runs
  ``/usr/bin/sparkring <argv>`` as its child and streams the child's output
  back as JSON lines.
- Every request writes one audit line to the service's standard output,
  which the journal keeps: time, peer uid, argv, exit status.

Nothing is enabled by the package: an operator opts in with
``systemctl enable --now sparkring-helper.socket``. Without the socket the
commands keep their present behavior (sudo). The validation grammar and the
protocol run on any host, so they are CPU-tested here; the unit wiring needs
a Spark to confirm.
"""
from __future__ import annotations

import json
import re
import socket
import struct
import subprocess
import sys
import time

SOCKET_PATH = "/run/sparkring/helper.sock"
SPARKRING = "/usr/bin/sparkring"
PROTOCOL = "sparkring-helper/v1"
# One request line; a larger request is refused before anything runs.
LINE_LIMIT = 1 << 20
REQUEST_TIMEOUT = 15.0

# A profile or instance name, and an arc such as 0,1 or 2-3, as values.
PROFILE = ("re", re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"))
# The allowed argv shapes, in the order the commands document them. A request
# matches one shape exactly: every token matches, nothing is left over, and
# argparse's flag reordering is not granted. Widening this table is a
# reviewed change, because each shape grants root execution of itself.
COMMANDS = (
    ("recover", "--auto"),
    ("up", PROFILE, "--execute"),
    ("up", "--execute"),
    ("down", PROFILE, "--execute"),
    ("down", "--execute"),
)


def validate(argv) -> None:
    """Refuse any argv the allowlist does not name exactly, with the reason."""
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(item, str) and item for item in argv)):
        raise ValueError("the request's argv must be a non-empty list of strings")
    for shape in COMMANDS:
        if _matches(argv, shape):
            return
    raise ValueError("the helper runs only: " + "; ".join(_shape_text(shape) for shape in COMMANDS))


def _matches(argv, shape) -> bool:
    if len(argv) != len(shape):
        return False
    for token, pattern in zip(argv, shape):
        # A plain string matches literally; ("re", REGEX) matches one value.
        if isinstance(pattern, str):
            if token != pattern:
                return False
        elif not pattern[1].fullmatch(token):
            return False
    return True


def _shape_text(shape) -> str:
    return " ".join(token[1].pattern if isinstance(token, tuple) else token for token in shape)


def audit_line(peer_uid, argv, exit_code) -> str:
    """The one journal line every request produces."""
    return json.dumps({"schema": PROTOCOL, "time": time.time(), "peer_uid": peer_uid,
                       "argv": argv, "exit": exit_code})


def serve(connection: socket.socket, *, spawn=None, argv0=SPARKRING, out=None):
    """Handle one connection: validate, run, stream, audit; never raises to the socket.

    Each response is one JSON line: ``{"type": "output", "stream":
    "out"|"err", "text": ...}`` for the child's output, then ``{"type":
    "exit", "code": N}`` (with ``"error"`` when the request itself was
    refused or the child could not start).
    """
    stream = out or sys.stdout
    run = spawn or (lambda argv: subprocess.run([argv0, *argv], capture_output=True, text=True, timeout=3600))
    peer_uid = None
    argv = None
    try:
        connection.settimeout(REQUEST_TIMEOUT)
        request = json.loads(_readline(connection))
        argv = request["argv"]
        peer_uid = _peer_uid(connection)
        validate(argv)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        _send(connection, {"type": "exit", "code": 2, "error": str(error)})
        print(audit_line(peer_uid, argv, 2), file=stream, flush=True)
        return
    try:
        result = run(argv)
        code, error = result.returncode, None
        for name, text in (("out", result.stdout), ("err", result.stderr)):
            for start in range(0, len(text or ""), 4096):
                _send(connection, {"type": "output", "stream": name, "text": text[start:start + 4096]})
    except (OSError, subprocess.SubprocessError) as exc:
        code, error = 2, str(exc)
    _send(connection, {"type": "exit", "code": code, **({"error": error} if error else {})})
    print(audit_line(peer_uid, argv, code), file=stream, flush=True)


def _peer_uid(connection: socket.socket):
    """The connecting process's uid, from the kernel's peer credentials (Linux)."""
    try:
        uid, _, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    except OSError:
        return None
    return uid


def _readline(connection: socket.socket) -> bytes:
    data = bytearray()
    while b"\n" not in data:
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
        if len(data) > LINE_LIMIT:
            raise ValueError("the request line is too long")
    return bytes(data.split(b"\n", 1)[0])


def _send(connection: socket.socket, document: dict) -> None:
    connection.sendall(json.dumps(document).encode() + b"\n")


def request(argv, *, socket_path: str = SOCKET_PATH, timeout: float = 7200.0) -> int:
    """Run ``argv`` through the helper; return the child's exit status.

    The caller stays unprivileged. Output arrives as the helper sends it, so
    a long installation shows its progress; the client's timeout outlives a
    model load, which can take thirty minutes.
    """
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    connection.connect(socket_path)
    connection.sendall(json.dumps({"argv": list(argv)}).encode() + b"\n")
    code = 1
    while True:
        line = _readline(connection)
        if not line:
            break
        document = json.loads(line)
        if document.get("type") == "output":
            print(document["text"], end="", file=sys.stdout if document.get("stream") == "out" else sys.stderr)
        elif document.get("type") == "exit":
            code = int(document.get("code", 1))
            if document.get("error"):
                print("sparkring helper: " + document["error"], file=sys.stderr)
            break
    connection.close()
    return code
