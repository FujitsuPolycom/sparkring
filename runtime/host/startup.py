"""Viewer-only startup observations from bounded local Docker and installer logs.

The host agent calls this through its existing telemetry lease. No API probing,
SSH, subprocess, or worker RPC is performed here. Docker access is local only.
"""
import datetime
import http.client
import itertools
import json
import os
from pathlib import Path
import re
import socket
import stat
import time
import urllib.parse

LIMIT = 64 * 1024
ROWS = 200
SCHEMA = "sparkring-startup/v1"
PHASES = {"installing", "starting", "loading", "compiling", "tuning", "graphs", "failed", "ready"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS = re.compile(r"\b(\d{1,7})/(\d{1,7})\b")
INSTALL = (
    (r"(?:Prepare checkpoint and verify all shards|Link and verify local checkpoint files|Verify checkpoint receipt)", "Checkpoint check"),
    (r"Download missing checkpoint files from huggingface.co", "Downloading checkpoint"),
    (r"Copy checkpoint files from Node \d|Spread the checkpoint from Node \d|Transfer checkpoint files", "Copying checkpoint"),
    (r"Load the image from its verified layers", "Loading image"),
    (r"Derive the checkpoint in the installer image", "Deriving checkpoint"),
    (r"Create stopped model container|Start model|Wait for API readiness|Start prepared model container", "Starting"),
)


def stamp(value):
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def clean(value):
    return "".join(c for c in ANSI.sub("", value) if c >= " " or c in "\r\n\t")


def parse(text, previous=None, *, installer=False):
    """Known log patterns only; unknown lines retain the last observation."""
    result = dict(previous or {})
    for line in clean(text[-LIMIT:]).splitlines()[-ROWS:]:
        phase = label = None
        if "Loading safetensors checkpoint shards:" in line:
            phase, label = "loading", "Loading weights"
        elif "Capturing CUDA graphs" in line:
            phase, label = "graphs", "Capturing CUDA graphs"
        elif re.search(r"\bb12x autotuning uses \d+ temporary KV blocks|\b(?:B12X|b12x):? [Aa]utotuning\b", line):
            phase, label = "tuning", "Tuning B12X"
        elif re.search(r"(?:Compiling|Compile|compilation).*torch\.compile|torch\.compile.*(?:takes|took)|\bCompiling a graph", line):
            phase, label = "compiling", "Compiling"
        elif installer:
            if re.search(r"(?:^|  )(?:Error:|Failed:|Stopped:) ", line):
                result.update(phase="failed", label="Failed", error=line.strip()[-500:])
                continue
            if "Done: " in line:
                continue
            for pattern, description in INSTALL:
                if re.search(pattern, line):
                    phase = "starting" if description == "Starting" else "installing"
                    label = description
                    break
        if phase:
            result = {"phase": phase, "label": label}
            progress = PROGRESS.search(line)
            if progress and 0 <= int(progress[1]) <= int(progress[2]) and int(progress[2]) > 0:
                result.update(done=int(progress[1]), total=int(progress[2]))
            if installer:
                transfer = re.search(r"\b(\d+(?:\.\d+)?) of (\d+(?:\.\d+)?) (?:GiB|MiB)\b", line)
                if transfer and 0 <= float(transfer[1]) <= float(transfer[2]) and float(transfer[2]) > 0:
                    result.update(done=round(100 * float(transfer[1]) / float(transfer[2])), total=100, unit="%")
        # An error line alone does not prove that a container failed.
        if not installer and re.search(r"\b(?:ERROR|[A-Za-z]*Error|[A-Za-z]*Exception):?\s", line):
            result["error"] = line.strip()[-500:]
    return result


def docker_get(path, *, root="/", limit=LIMIT):
    """A bounded, deadline-limited GET on Docker's existing Unix socket."""
    connection = http.client.HTTPConnection("localhost", timeout=.2)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + .4
    try:
        sock.settimeout(.2)
        sock.connect(str(Path(root) / "var/run/docker.sock"))
        connection.sock = sock
        connection.request("GET", path)
        response = connection.getresponse()
        body = bytearray()
        while len(body) <= limit and time.monotonic() < deadline:
            block = response.read1(min(8192, limit + 1 - len(body)))
            if not block:
                return bytes(body) if response.status == 200 else b""
            body.extend(block)
        return b""
    finally:
        connection.close()
        sock.close()


def log_text(raw):
    """Decode Docker's stdout/stderr multiplex frames, or a TTY's plain text."""
    if raw[:4] not in (b"\x01\0\0\0", b"\x02\0\0\0"):
        return raw.decode("utf-8", "replace")
    parts = []
    while len(raw) >= 8 and raw[:4] in (b"\x01\0\0\0", b"\x02\0\0\0"):
        size = int.from_bytes(raw[4:8], "big")
        if size > len(raw) - 8:
            break
        parts.append(raw[8:8 + size])
        raw = raw[8 + size:]
    return b"".join(parts).decode("utf-8", "replace")


class Reader:
    """Retain phases only for the same container ID and Docker StartedAt."""
    def __init__(self, get=docker_get):
        self.get, self.cache = get, {}

    def read(self, now=None):
        now = time.time() if now is None else now
        found, retained = {}, {}
        deadline = time.monotonic() + 1.5
        try:
            filters = urllib.parse.quote(json.dumps({"label": ["io.sparkring.deployment"]}))
            containers = json.loads(self.get("/containers/json?all=1&limit=32&filters=" + filters) or b"[]")
            for container in containers[:32]:
                if time.monotonic() >= deadline:
                    break
                labels = container.get("Labels") or {}
                deployment, rank = labels.get("io.sparkring.deployment"), labels.get("io.sparkring.rank", "")
                identity = container.get("Id", "")
                if not isinstance(deployment, str) or not re.fullmatch(r"[0-9a-f]{12,64}", identity) or not re.fullmatch(r"[0-7]", rank):
                    continue
                doc = json.loads(self.get(f"/containers/{identity}/json") or b"{}")
                state = doc.get("State") or {}
                started = stamp(state.get("StartedAt"))
                if not started or started <= 0:
                    started = stamp(doc.get("Created"))
                if not started or started <= 0 or started > now + 5:
                    continue
                key = (identity, state.get("StartedAt"))
                value = dict(self.cache.get(key) or {"phase": "starting", "label": "Starting"})
                health = (state.get("Health") or {}).get("Status")
                failed = state.get("Status") in ("exited", "dead") or health == "unhealthy"
                if value.get("phase") == "failed" and not failed:
                    value = {"phase": "starting", "label": "Starting"}
                # Healthy containers need no log reads. API readiness is decided by the dashboard.
                if failed or health != "healthy":
                    since = urllib.parse.quote(str(state.get("StartedAt") or doc.get("Created")))
                    raw = self.get(f"/containers/{identity}/logs?stdout=1&stderr=1&tail={ROWS}&since={since}")
                    value = parse(log_text(raw), value)
                if failed:
                    error = value.get("error") or state.get("Error")
                    health_log = (state.get("Health") or {}).get("Log") or []
                    if not error and health_log:
                        lines = clean(str(health_log[-1].get("Output", ""))).splitlines()
                        error = lines[-1][-500:] if lines else None
                    fallback = "Health check failed" if health == "unhealthy" else "Container exited"
                    value.update(phase="failed", label="Failed", error=error or fallback)
                elif health == "healthy":
                    value = {"phase": "ready", "label": "Starting"}
                value.update(container_state=state.get("Status"), health=health,
                             schema=SCHEMA, at=now, started_at=started,
                             run_id=identity + ":" + str(state.get("StartedAt")), rank=int(rank))
                retained[key] = value
                found.setdefault(deployment, []).append(value)
        except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException):
            pass
        self.cache = dict(list({**self.cache, **retained}.items())[-32:])
        return found


def tail(path, limit=LIMIT):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        size = os.fstat(stream.fileno())
        if not stat.S_ISREG(size.st_mode):
            return ""
        stream.seek(max(0, size.st_size - limit))
        return stream.read(limit).decode("utf-8", "replace")


def process_alive(pid):
    return Path(f"/proc/{pid}").is_dir()


def parse_operation(text, positions):
    value = parse(text, installer=True)
    ranks = {}
    for line in clean(text).splitlines()[-ROWS:]:
        match = re.search(r"Node ([0-7]): ", line)
        if match and int(match[1]) < len(positions):
            rank = int(match[1])
            row = parse(line, ranks.get(rank), installer=True)
            if row:
                ranks[rank] = {**row, "rank": rank, "position": positions[rank]}
    if ranks:
        value["ranks"] = list(ranks.values())
    return value


def operations(root=Path("/var/log/sparkring"), now=None, alive=process_alive):
    """Read at most eight active installer operations; never scan their details logs."""
    now = time.time() if now is None else now
    result = []
    try:
        with os.scandir(root) as entries:
            markers = [(e.stat().st_mtime, Path(e.path)) for e in itertools.islice(entries, 1024)
                       if e.name.startswith("slice-") and e.name.endswith(".json") and not e.is_symlink()]
            markers = [path for _, path in sorted(markers, reverse=True)[:8]]
        documents = []
        for path in markers:
            try:
                marker = json.loads(tail(path, 8192))
                if isinstance(marker, dict) and type(marker.get("started_at")) in (int, float):
                    documents.append(marker)
            except (OSError, ValueError):
                continue
        occupied = set()
        for marker in sorted(documents, key=lambda m: m["started_at"], reverse=True):
            try:
                positions = marker.get("positions")
                positions = list(range(8)) if positions is None else positions
                if not isinstance(positions, list) or any(type(p) is not int or not 0 <= p < 8 for p in positions):
                    continue
                if occupied.intersection(positions):
                    continue
                occupied.update(positions)
                operation = marker.get("operation", "")
                if (not marker.get("deployment") or (marker.get("complete") and not marker.get("failed"))
                        or not re.fullmatch(r"[0-9a-f]{32}", operation)
                        or type(marker.get("pid")) is not int or marker["pid"] <= 0):
                    continue
                if not marker.get("complete") and not alive(marker["pid"]):
                    continue
                value = parse_operation(tail(root / f"operation-{operation}.log"), marker["deployment"]["positions"])
                if not value:
                    value = {"phase": "installing", "label": "Checkpoint check"}
                if marker.get("failed"):
                    value.update(phase="failed", label="Failed")
                value.update(at=now, started_at=marker["started_at"], run_id=operation,
                             deployment=marker["deployment"])
                result.append(value)
            except (OSError, ValueError, TypeError, KeyError):
                continue
    except OSError:
        pass
    return result


_reader = Reader()


def read_live():
    from runtime.host import telemetry
    return {**telemetry.read_snapshots(), "__startup__": _reader.read()}


def project(values, ids, now):
    """Bound and validate observations across the existing agent channel."""
    result = {}
    if not isinstance(values, dict):
        return result
    for key in ids[:8]:
        ranks = values.get(key, [])
        if not isinstance(ranks, list):
            continue
        for row in ranks[:8]:
            if (not isinstance(row, dict) or row.get("schema") != SCHEMA or row.get("phase") not in PHASES
                    or type(row.get("at")) not in (int, float) or not -5 <= now - row["at"] <= 6
                    or type(row.get("started_at")) not in (int, float) or not 0 < row["started_at"] <= now + 5
                    or type(row.get("rank")) is not int or not 0 <= row["rank"] < 8):
                continue
            clean_row = {k: row[k] for k in ("phase", "at", "started_at", "rank")}
            for name in ("label", "error", "run_id", "container_state", "health"):
                if isinstance(row.get(name), str):
                    clean_row[name] = clean(row[name])[:500]
            if row.get("unit") == "%":
                clean_row["unit"] = "%"
            if type(row.get("done")) is int and type(row.get("total")) is int and 0 <= row["done"] <= row["total"] <= 10**7:
                clean_row.update(done=row["done"], total=row["total"])
            existing = result.setdefault(key, [])
            same_rank = next((r for r in existing if r["rank"] == row["rank"]), None)
            if same_rank is None or row["started_at"] > same_rank["started_at"]:
                if same_rank is not None:
                    existing.remove(same_rank)
                existing.append({"schema": SCHEMA, **clean_row})
    return result


def group(ranks, positions):
    ranks = [dict(row, position=positions[row["rank"]]) for row in ranks if row["rank"] < len(positions)]
    if not ranks:
        return None
    order = ["failed", "installing", "starting", "loading", "compiling", "tuning", "graphs", "ready"]
    row = min(ranks, key=lambda r: (order.index(r["phase"]), r.get("done", 0) / (r.get("total") or 1)))
    missing = [p for p in positions if p not in {r["position"] for r in ranks}]
    return {**row, "started_at": min(r["started_at"] for r in ranks), "ranks": ranks, "unobserved": missing}
