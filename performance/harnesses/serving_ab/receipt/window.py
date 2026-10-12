"""An exclusive measurement window: the pre-flight before a measurement and the request-source record after it.

**Pre-flight** (:func:`preflight`) refuses to start a measurement when

- the deployment is serving requests: vLLM's ``vllm:num_requests_running`` or ``vllm:num_requests_waiting``
  gauge on ``/metrics`` is above zero in any of several samples; or
- another owner holds a reservation of any requested Spark. Reservations come from pluggable sources
  (:data:`RESERVATION_SOURCES`), each a function from Node A to a list of :class:`Reservation`:
  ``hwlock`` reads the ring's measurement locks ``/tmp/ring8-hw.lock/owner`` (every Spark) and
  ``/tmp/ring8-hw-<first>-<last>.lock/owner`` (that range); ``setlock`` reads the serving A/B runner's set
  locks ``/tmp/ring8-sets/<position>/owner``; ``command:CMD`` runs ``CMD`` on Node A, which prints a JSON list
  of ``{"positions": [...], "owner": "..."}`` (for a reservation tool such as ``sparkring reserve``).
  An owner file is either ``NAME EPOCH`` (``hwlock.sh``, ``setlock.py``; abandoned after
  :data:`STALE_SECONDS` without a refresh, as ``setlock.py`` rules) or ``key: value`` lines with an
  ``owner:`` line and an optional ``sparks: <first>-<last>`` line (no refresh protocol, so never stale).
  A reservation whose owner text contains the campaign's ``--owner`` is the campaign's own.

**Window record** (:func:`summarize`) counts rank 0's access-log lines between two times by source address,
method and path, and fails when a source other than the declared clients sent a request other than the
monitoring reads in :data:`MONITORING` (``GET /metrics``, ``/health``, ``/v1/models`` and similar, which a
dashboard or health check sends without loading the GPU). With ``strict`` every request from another
source fails it.

An access log names a client by its address only, so two clients on one machine look like one. The
pre-flight is what keeps a second client on the same machine out. Inside the window two signs remain:

- ``expect_tool_starts``: llm-inference-bench reads ``GET /version`` once when it starts, so a window in
  which the declared address read it more often than the campaign started the tool holds another instance
  of the tool (a campaign that runs a warm-up and a measured start of the serving A/B runner starts it
  twice);
- ``max_concurrency`` (the benchmark's largest stream count): vLLM's periodic engine line reports more
  running plus waiting requests than the benchmark sends. That line comes every 10 s while the engine is
  busy, so an overlap shorter than that, or one whose sum stays within the limit, is not seen.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shlex
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

ENGINE_STATS = re.compile(r"^(?:(\d{4}-\d\d-\d\dT[\d:.]+Z) )?.*Running: (\d+) reqs, Waiting: (\d+) reqs")
ACCESS = re.compile(r'^(?:(\d{4}-\d\d-\d\dT[\d:.]+Z) )?.*?INFO:\s+(\[[0-9a-fA-F:.]+\]|[0-9A-Za-z.:-]+?):(\d+) - '
                    r'"([A-Z]+) (\S+) HTTP/[\d.]+" (\d{3})')
MONITORING = frozenset({("GET", "/metrics"), ("GET", "/health"), ("GET", "/v1/models"), ("GET", "/version"),
                        ("GET", "/ping"), ("GET", "/get_server_info"), ("GET", "/load"),
                        ("GET", "/server_info"), ("GET", "/sparkring/status")})
STALE_SECONDS = 2700
"""As performance/harnesses/serving_ab/setlock.py: an ``NAME EPOCH`` owner older than this is abandoned."""


def parse_time(text: str) -> dt.datetime:
    """An ISO-8601 time; nanosecond fractions are cut to microseconds; naive times are UTC."""
    text = text.strip().replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    value = dt.datetime.fromisoformat(text)
    return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)


def parse_access_log(text: str) -> list[dict]:
    """Every access-log line of a ``docker logs --timestamps`` text: time, source address, method, path, status."""
    out = []
    for line in text.splitlines():
        match = ACCESS.search(line)
        if match:
            stamp, address, _port, method, path, status = match.groups()
            out.append({"time": stamp, "address": address.strip("[]"), "method": method,
                        "path": path.split("?", 1)[0], "status": int(status)})
    return out


def parse_engine_stats(text: str) -> list[dict]:
    """vLLM's periodic engine lines of a ``docker logs --timestamps`` text: time, running, waiting."""
    return [{"time": m[1], "running": int(m[2]), "waiting": int(m[3])}
            for m in (ENGINE_STATS.search(line) for line in text.splitlines()) if m]


def summarize(entries: Iterable[Mapping], *, clients: Sequence[str], since: str | None = None,
              until: str | None = None, strict: bool = False, engine_stats: Iterable[Mapping] = (),
              max_concurrency: int | None = None, expect_tool_starts: int | None = None) -> dict:
    """Request counts in [since, until] by source, method and path, and the window's verdict."""
    start = parse_time(since) if since else None
    end = parse_time(until) if until else None
    peak = None
    for stat in engine_stats:
        when = parse_time(stat["time"]) if stat.get("time") else None
        if when and ((start and when < start) or (end and when > end)):
            continue
        total = stat["running"] + stat["waiting"]
        if peak is None or total > peak["requests"]:
            peak = {"time": stat.get("time"), "requests": total, "running": stat["running"], "waiting": stat["waiting"]}
    counts: Counter = Counter()
    untimed = 0
    for entry in entries:
        if entry.get("time"):
            when = parse_time(entry["time"])
            if (start and when < start) or (end and when > end):
                continue
        elif start or end:
            untimed += 1
            continue
        counts[(entry["address"], entry["method"], entry["path"])] += 1
    rows, outside = [], []
    for (address, method, path), n in sorted(counts.items()):
        mine = address in clients
        monitoring = (method, path) in MONITORING
        row = {"address": address, "method": method, "path": path, "requests": n,
               "source": "declared client" if mine else "monitoring" if monitoring else "OUTSIDE"}
        if not mine and (strict or not monitoring):
            row["source"] = "OUTSIDE"
            outside.append(row)
        rows.append(row)
    over = max_concurrency is not None and peak is not None and peak["requests"] > max_concurrency
    starts = sum(n for (address, method, path), n in counts.items()
                 if address in clients and (method, path) == ("GET", "/version"))
    extra_starts = expect_tool_starts is not None and starts > expect_tool_starts
    return {"since": since, "until": until, "clients": list(clients), "strict": strict, "rows": rows,
            "outside": outside, "outside_requests": sum(r["requests"] for r in outside),
            "lines_without_time": untimed, "max_concurrency": max_concurrency, "engine_peak": peak,
            "over_concurrency": over, "tool_starts": starts, "expect_tool_starts": expect_tool_starts,
            "extra_tool_starts": extra_starts, "exclusive": not outside and not over and not extra_starts}


def format_window(record: Mapping) -> str:
    lines = [f"window {record['since']} .. {record['until']}, declared clients {', '.join(record['clients'])}"]
    for row in record["rows"]:
        lines.append(f"  {row['address']:<16} {row['method']:<5} {row['path']:<28} {row['requests']:>6}  {row['source']}")
    peak = record.get("engine_peak")
    if peak:
        lines.append(f"engine peak: {peak['running']} running + {peak['waiting']} waiting at {peak['time']}"
                     + (f" (the benchmark sends at most {record['max_concurrency']})"
                        if record.get("max_concurrency") else ""))
    if record.get("over_concurrency"):
        lines.append("the engine held more requests than the benchmark sends: another client was active")
    if record.get("expect_tool_starts") is not None:
        lines.append(f"benchmark tool starts from the declared clients (GET /version): {record['tool_starts']}, "
                     f"expected {record['expect_tool_starts']}")
    if record.get("extra_tool_starts"):
        lines.append("another instance of the benchmark tool started from a declared address in the window")
    lines.append(f"outside requests: {record['outside_requests']} -> "
                 f"{'EXCLUSIVE' if record['exclusive'] else 'NOT EXCLUSIVE'}")
    return "\n".join(lines)


# Pre-flight -------------------------------------------------------------------------------------------------

def request_gauges(metrics_text: str) -> dict:
    """Sum of ``vllm:num_requests_running`` and ``..._waiting`` over engines; None where absent."""
    out: dict = {"running": None, "waiting": None}
    for line in metrics_text.splitlines():
        match = re.match(r"^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", line)
        if match:
            out[match[1]] = (out[match[1]] or 0.0) + float(match[2])
    return out


@dataclass(frozen=True)
class Reservation:
    source: str
    where: str
    owner: str
    positions: tuple[int, ...] | None  # None: every Spark of the ring
    age_seconds: float | None = None
    stale: bool = False


def parse_owner(text: str, *, now: float, where: str, source: str, positions=None) -> Reservation:
    """A reservation from an owner file (see the module docstring for its two formats)."""
    text = text.strip()
    first = text.splitlines()[0] if text else ""
    one_line = re.match(r"^(\S+)\s+(\d{9,})\s*$", first)
    if one_line and len(text.splitlines()) == 1:
        age = now - int(one_line[2])
        return Reservation(source, where, one_line[1], positions, age, age > STALE_SECONDS)
    fields = dict(re.findall(r"^([A-Za-z_]+):\s*(.*)$", text, flags=re.M))
    sparks = re.match(r"^\s*(\d+)\s*-\s*(\d+)", fields.get("sparks", ""))
    if sparks:
        positions = tuple(range(int(sparks[1]), int(sparks[2]) + 1))
    return Reservation(source, where, fields.get("owner", first) or "unknown", positions)


def hwlock_source(run: Callable[[str], str], now: float) -> list[Reservation]:
    script = ('for d in /tmp/ring8-hw.lock /tmp/ring8-hw-*.lock; do [ -d "$d" ] || continue; '
              'printf "@@LOCK %s %s\\n" "$d" "$(stat -c %Y "$d/owner" 2>/dev/null)"; cat "$d/owner" 2>/dev/null; done')
    return _lock_dirs(run(script), now, "hwlock")


def setlock_source(run: Callable[[str], str], now: float) -> list[Reservation]:
    script = ('for d in /tmp/ring8-sets/*; do [ -d "$d" ] || continue; '
              'printf "@@LOCK %s %s\\n" "$d" "$(stat -c %Y "$d/owner" 2>/dev/null)"; cat "$d/owner" 2>/dev/null; done')
    return _lock_dirs(run(script), now, "setlock")


def _lock_dirs(text: str, now: float, source: str) -> list[Reservation]:
    out = []
    for block in text.split("@@LOCK ")[1:]:
        head, _, body = block.partition("\n")
        where = head.split()[0]
        name = where.rsplit("/", 1)[-1]
        span = re.match(r"ring8-hw-(\d+)-(\d+)\.lock$", name)
        positions = (tuple(range(int(span[1]), int(span[2]) + 1)) if span
                     else (int(name),) if name.isdigit() else None)
        out.append(parse_owner(body, now=now, where=where, source=source, positions=positions))
    return out


def command_source(command: str) -> Callable[[Callable[[str], str], float], list[Reservation]]:
    def read(run: Callable[[str], str], now: float) -> list[Reservation]:
        text = run(command)
        items = json.loads(text) if text.strip() else []
        return [Reservation(f"command:{command}", command, str(item.get("owner", "unknown")),
                            tuple(item["positions"]) if item.get("positions") is not None else None)
                for item in items]
    return read


RESERVATION_SOURCES: dict[str, Callable] = {"hwlock": hwlock_source, "setlock": setlock_source}


def sources_from(names: Sequence[str]) -> list[Callable]:
    out = []
    for name in names:
        if name.startswith("command:"):
            out.append(command_source(name.split(":", 1)[1]))
        elif name in RESERVATION_SOURCES:
            out.append(RESERVATION_SOURCES[name])
        else:
            raise ValueError(f"unknown reservation source {name!r}; known: {', '.join(RESERVATION_SOURCES)}, command:CMD")
    return out


def blocking(reservations: Iterable[Reservation], positions: Sequence[int], owner: str | None) -> list[Reservation]:
    """Reservations held by someone else on any of ``positions`` (stale ones are not blocking)."""
    out = []
    for item in reservations:
        if item.stale:
            continue
        if owner and owner in item.owner:
            continue
        if item.positions is None or set(item.positions) & set(positions):
            out.append(item)
    return out


def preflight(*, positions: Sequence[int], read_metrics: Callable[[], str | None], run_on_node_a: Callable[[str], str],
              sources: Sequence[Callable], owner: str | None = None, samples: int = 5, interval: float = 2.0,
              now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> dict:
    """The pre-flight verdict (see the module docstring)."""
    readings, problems = [], []
    for i in range(samples):
        text = read_metrics()
        gauges = request_gauges(text or "")
        readings.append(gauges)
        if text is None:
            problems.append("the API's /metrics did not answer")
            break
        if gauges["running"] is None:
            problems.append("/metrics has no vllm:num_requests_running gauge")
            break
        if gauges["running"] or gauges["waiting"]:
            problems.append(f"the deployment is serving: {gauges['running']:.0f} running, "
                            f"{gauges['waiting'] or 0:.0f} waiting requests")
            break
        if i + 1 < samples:
            sleep(interval)
    reservations = []
    for source in sources:
        reservations.extend(source(run_on_node_a, now()))
    held = blocking(reservations, positions, owner)
    for item in held:
        problems.append(f"{item.where} reserves {'every Spark' if item.positions is None else list(item.positions)} "
                        f"for {item.owner!r}")
    return {"positions": list(positions), "owner": owner, "metrics": readings,
            "reservations": [asdict(r) for r in reservations], "blocking": [asdict(r) for r in held],
            "problems": problems, "clear": not problems}


def format_preflight(result: Mapping) -> str:
    lines = [f"pre-flight for positions {result['positions']}" + (f" as {result['owner']!r}" if result["owner"] else "")]
    for reading in result["metrics"]:
        lines.append(f"  /metrics: running {reading['running']}, waiting {reading['waiting']}")
    for item in result["reservations"]:
        state = "stale" if item["stale"] else "held"
        age = f", {item['age_seconds']:.0f} s old" if item.get("age_seconds") is not None else ""
        span = "every Spark" if item["positions"] is None else item["positions"]
        lines.append(f"  {item['source']}: {item['where']} {state} by {item['owner']!r} on {span}{age}")
    lines += [f"  REFUSE {p}" for p in result["problems"]]
    lines.append("CLEAR" if result["clear"] else "REFUSED")
    return "\n".join(lines)


def access_log_script(container: str, since: str, until: str | None) -> str:
    until_part = f" --until {shlex.quote(until)}" if until else ""
    return (f"docker logs --timestamps --since {shlex.quote(since)}{until_part} {shlex.quote(container)} 2>&1 "
            f"| grep -E '\"[A-Z]+ /|Running: [0-9]+ reqs'\n")
