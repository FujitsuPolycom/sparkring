"""Bounded local stat snapshots attached to the host agent's existing report."""
import json
import math
import os
from pathlib import Path
import stat
import time

ROOT = "/run/sparkring/telemetry"
LIMIT = 4096
COUNTERS = ("prompt", "gen", "accepted", "drafts", "draft_tokens", "steps", "step_seconds",
            "ttft_sum", "ttft_count")


def valid(value):
    if not isinstance(value, dict) or value.get("schema") != "sparkring-stats/v1":
        return False
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        return False
    return all(type(value.get(key)) in (int, float) and math.isfinite(value[key]) and value[key] >= 0
               for key in (*COUNTERS, "at", "monotonic", "running", "waiting", "kv"))


def read_snapshots(*, root="/", now=time.time):
    found = {}
    directory = Path(root) / ROOT.lstrip("/")
    try:
        # Installations leave small directories behind; bound even a damaged
        # host's enumeration and payload without following special files.
        import itertools
        with os.scandir(directory) as entries:
            for entry in itertools.islice(entries, 128):
                child = Path(entry.path)
                try:
                    if child.is_symlink():
                        continue
                    fd = os.open(child / "snapshot.json", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                 | getattr(os, "O_NONBLOCK", 0))
                    with os.fdopen(fd, "rb") as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            continue
                        raw = stream.read(LIMIT + 1)
                    if len(raw) > LIMIT:
                        continue
                    value = json.loads(raw)
                    if valid(value) and value.get("deployment_id") == child.name and -5 <= now() - value["at"] <= 180:
                        found[child.name] = {**value, "read_at": now()}
                except (OSError, ValueError, TypeError):
                    continue
    except OSError:
        pass
    return found
