"""Bounded subprocesses, crash-released local locks and run artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
import uuid

from .contracts import Refused, require


def clean_environment(extra=None):
    keys = (
        "PATH",
        "SystemRoot",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "ProgramData",
        "ProgramFiles",
        "ProgramFiles(x86)",
    )
    env = {key: os.environ[key] for key in keys if key in os.environ}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        PYTHONNOUSERSITE="1",
        PYTHONUTF8="1",
    )
    env.update(extra or {})
    return env


def command(
    argv, *, cwd=None, seconds=60, limit=1024 * 1024, env=None, input_bytes=None
):
    """Execute an argument vector with bounded output and an owned process group."""
    require(
        argv and all(isinstance(x, str) and "\x00" not in x for x in argv),
        "Invalid command vector",
    )
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=clean_environment(env),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        **options,
    )
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()

    def drain(stream, name):
        try:
            while data := stream.read(65536):
                remaining = limit - len(buffers[name])
                buffers[name].extend(data[: max(0, remaining)])
                if len(data) > remaining:
                    overflow.set()
        except (OSError, ValueError):
            pass

    readers = [
        threading.Thread(target=drain, args=(getattr(process, key), key), daemon=True)
        for key in buffers
    ]
    for reader in readers:
        reader.start()

    def send():
        try:
            if input_bytes:
                pending = memoryview(input_bytes)
                while pending:
                    written = process.stdin.write(pending)
                    if not written:
                        break
                    pending = pending[written:]
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    writer = threading.Thread(target=send, daemon=True)
    writer.start()
    terminated = None
    try:
        while process.poll() is None:
            if overflow.is_set() or time.monotonic() - started > seconds:
                terminated = "output_limit" if overflow.is_set() else "timeout"
                break
            time.sleep(0.02)
    except BaseException:
        terminated = "interrupted"
        raise
    finally:
        if terminated and process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=1)
        if any(reader.is_alive() for reader in readers):
            terminated = terminated or "descendant_holds_output"
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    return {
        "returncode": process.returncode,
        "stdout": bytes(buffers["stdout"]),
        "stderr": bytes(buffers["stderr"]),
        "seconds": time.monotonic() - started,
        "uncertain": terminated is not None or overflow.is_set(),
        "termination": terminated,
    }


def checked(argv, **kwargs):
    result = command(argv, **kwargs)
    if result["returncode"] or result["uncertain"]:
        detail = result["stderr"].decode(errors="replace")
        if len(detail) > 2000:
            # Build systems commonly emit their actionable exception last.
            detail = detail[:400] + "\n[intermediate output omitted]\n" + detail[-1500:]
        raise Refused("Command failed: " + str(argv[0]) + ": " + detail)
    return result["stdout"]


def write_json(path, value, *, replace=False):
    path = Path(path)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(raw)


def storage_check(root, budgets):
    """Soft checks between stages; use filesystem/Docker quotas for hard limits."""
    root = Path(root)
    if "min_free_bytes" in budgets:
        require(
            shutil.disk_usage(root).free >= budgets["min_free_bytes"],
            "Builder filesystem free-space floor reached",
        )
    if "state_bytes" in budgets:
        total = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                # CMake/compiler outputs can contain links. Budget their inode,
                # never traverse or read the target; source inputs have a
                # separate strict inventory check.
                total += path.lstat().st_size
            elif path.is_file():
                total += path.stat().st_size
            require(
                total <= budgets["state_bytes"],
                "Retained state exceeds storage budget; archive exact runs after review",
            )


@contextmanager
def lock(root):
    supplied = Path(root).absolute()
    require(
        not any(p.is_symlink() for p in (supplied, *supplied.parents)),
        "State path cannot traverse symlinks",
    )
    root = supplied.resolve()
    require(
        root not in (Path(root.anchor), Path.home(), Path.cwd()),
        "Use a dedicated upgrade state directory",
    )
    marker = root / ".sparkring-upgrade-state"
    if root.exists() and any(root.iterdir()):
        require(
            marker.is_file() and not marker.is_symlink(),
            "State directory is not owned by the upgrade runner",
        )
        require(
            marker.read_text(encoding="ascii") == "sparkring-upgrade-state/v1\n",
            "State ownership marker differs",
        )
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text("sparkring-upgrade-state/v1\n", encoding="ascii")
    require(not root.is_symlink(), "State root cannot be a symlink")
    require(
        not any(
            (root / name).is_symlink()
            for name in ("run.lock", "state.json", "mirrors", "runs")
        ),
        "State control path cannot be a symlink",
    )
    with (root / "run.lock").open("a+b") as stream:
        try:
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise Refused("Another upgrade run owns this state directory") from exc
        try:
            yield root
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
