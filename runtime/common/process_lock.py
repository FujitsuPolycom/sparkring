"""Process-scoped local locks that are released automatically after interruption."""
import contextlib
import os
from pathlib import Path


@contextlib.contextmanager
def hold(path):
    path = Path(path)
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise ValueError("Operation lock path contains a symlink")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "r+b", buffering=0) as stream:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            raise ValueError("Another operation is active; use sparkring status to follow it") from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()).encode())
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
