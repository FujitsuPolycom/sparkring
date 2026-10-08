"""Collect a SIRCL deployment's receipts, scan its logs for NCCL and judge both against its transport section.

Each rank's SIRCL adapter writes one receipt per vLLM group,
``rank<global>-<group>.json`` (``sircl-vllm-receipt/v1``,
``sparkring_sircl.vllm.receipt``), into its ``SIRCL_RECEIPT_DIR``, which
``runtime/common/transport.py`` binds to ``<workspace>/sircl/receipts`` on its
Spark. A receipt names the group's NCCL policy, whether PyNccl was built, the
SIRCL session and one row per (collective, backend, method) with its calls.

``host_report`` runs on each Spark (``scripts/installer_host.py``, operation
``transport-receipts``, read-only): the receipts the running container wrote
since it started, and the lines of its log that show NCCL. A SIRCL deployment
with NCCL off runs with ``NCCL_DEBUG=INFO`` and ``NCCL_DEBUG_SUBSYS=INIT``, so
NCCL logs every communicator it creates, including those of compiled C++
paths that SIRCL's Python tripwire does not see; vLLM's PyNccl logs its own
line whatever NCCL's settings.

``evaluate`` judges the reports on Node A with the SIRCL launcher's own
checks (``sparkring_sircl.vllm.serve.checks``):

- every rank has a tensor-parallel receipt in state ``ready`` that ran at
  least one all-reduce on SIRCL, and no collective was refused;
- with NCCL off (``nccl: never``): every group of every rank shows NCCL
  policy ``none``, PyNccl ``skipped`` and no ``nccl`` row, and no log shows
  an NCCL communicator;
- with ``--nccl auto``: no group whose policy is ``none`` built PyNccl or
  sent a collective to NCCL;
- with decode-context parallelism above 1 (the section's ``group.dcp``), every
  rank also holds a decode-context-parallel receipt with a SIRCL session;
- each tensor-parallel and decode-context-parallel session decided from the
  measured tuning table the deployment's transport section matched for its
  kind of session (none: SIRCL's rules), and with a ``large_blocks`` tuning
  setting its sessions report that grid cap;
- every tensor-parallel session reports the link slots, link slot, chain slot
  and large-message piece that the tuning row sets or, where the row leaves
  them unset, the matched SIRCL table records
  (``transport.expected_session_settings``);
- every receipt that names its NCCL mode names the deployment's (receipts
  state the mode and SIRCL's NCCL rule, ``nccl_mode`` and ``nccl_rule``).

The verdict is ``as-expected``, ``differs`` (with each problem) or
``unknown`` when a Spark could not be read. ``record`` copies the receipts to
``<deployment>/receipts/<UTC time>/`` on Node A and writes the verdict there
and as ``<deployment>/transport.json``, which ``sparkring status`` reads. A
differing verdict never stops the model.
"""
from collections import deque
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import time

from runtime.common import transport

REPORT_SCHEMA = "sparkring-transport-report/v1"
VERDICT_SCHEMA = "sparkring-transport-verdict/v1"
RECEIPT_SCHEMA = "sircl-vllm-receipt/v1"
LATEST = "transport.json"
RECEIPTS = "receipts"
MAX_RECEIPTS = 64
MAX_RECEIPT_BYTES = 1 << 20
MAX_MATCHES = 50
TAIL_LINES = 200
LOG_SECONDS = 300


def _patterns():
    from spark_transport.sircl.sparkring_sircl.vllm.serve import plan
    return plan.NCCL_INIT_PATTERNS, plan.NCCL_LIBRARY_PATTERNS


def started_epoch(text):
    """Seconds since the epoch of Docker's ``State.StartedAt`` (RFC 3339 with up to nanoseconds), or None."""
    if not isinstance(text, str) or not text or text.startswith("0001-"):
        return None
    head, _, fraction = text.rstrip("Z").partition(".")
    try:
        moment = datetime.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    return moment.timestamp() + (float("0." + fraction[:6]) if fraction.isdigit() else 0.0)


def read_receipts(directory, *, since=None):
    """``(receipts, problems)``: the ``sircl-vllm-receipt/v1`` files in ``directory`` written at or after ``since``."""
    directory = Path(directory)
    receipts, problems = [], []
    if not directory.is_dir():
        return receipts, [f"{directory} does not exist"]
    paths = sorted(directory.glob("rank*-*.json"))
    if len(paths) > MAX_RECEIPTS:
        problems.append(f"{len(paths)} receipt files; the first {MAX_RECEIPTS} were read")
    for path in paths[:MAX_RECEIPTS]:
        try:
            stat = path.stat()
            if since is not None and stat.st_mtime + 1 < since:
                continue
            if stat.st_size > MAX_RECEIPT_BYTES:
                problems.append(f"{path.name} is larger than {MAX_RECEIPT_BYTES} bytes")
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append(f"{path.name} cannot be read: {error}")
            continue
        if not isinstance(record, dict) or record.get("schema") != RECEIPT_SCHEMA:
            problems.append(f"{path.name} is not a {RECEIPT_SCHEMA} receipt")
            continue
        receipts.append({"file": path.name, **record})
    return receipts, problems


def scan_log(container, *, popen=subprocess.Popen, clock=time.monotonic):
    """The NCCL lines and the last lines of a container's log, read once from its start.

    Returns ``{"init": [...], "library": [...], "tail": [...], "complete": bool}``:
    NCCL communicator lines (SIRCL's ``NCCL_INIT_PATTERNS``), other NCCL
    library lines, at most ``MAX_MATCHES`` of each, and the last
    ``TAIL_LINES`` lines. ``complete`` is false when the read stopped after
    ``LOG_SECONDS``.
    """
    init_patterns, library_patterns = _patterns()
    found = {"init": [], "library": []}
    tail = deque(maxlen=TAIL_LINES)
    complete = True
    process = popen(["docker", "--context", "default", "logs", container], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, errors="replace")
    deadline = clock() + LOG_SECONDS
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if any(pattern in line for pattern in init_patterns):
                if len(found["init"]) < MAX_MATCHES:
                    found["init"].append(line[-400:])
            elif any(pattern in line for pattern in library_patterns):
                if len(found["library"]) < MAX_MATCHES:
                    found["library"].append(line[-400:])
            if clock() > deadline:
                complete = False
                break
    finally:
        process.kill()
        process.wait()
    return {**found, "tail": list(tail), "complete": complete}


def host_report(lock, number, info, *, root="/", popen=subprocess.Popen):
    """One rank's transport report on its Spark: its receipts since the container started and its log's NCCL lines."""
    section = lock.get("transport")
    if not section:
        return {"schema": REPORT_SCHEMA, "rank": number, "backend": "prepared"}
    state = (info or {}).get("State") or {}
    started = state.get("StartedAt")
    directory = Path(root) / transport.receipt_directory(lock).lstrip("/")
    receipts, problems = read_receipts(directory, since=started_epoch(started))
    log = scan_log(info["Id"], popen=popen) if info else {"init": [], "library": [], "tail": [], "complete": False}
    if not info:
        problems.append("the model container does not exist")
    return {"schema": REPORT_SCHEMA, "rank": number, "backend": "sircl",
            "container": {"running": bool(state.get("Running")), "started_at": started},
            "receipts": receipts, "problems": problems, "log": log}


# Node A.

def gather(runner, lock, *, workers=8):
    """Every rank's ``host_report`` through the deployment's runner; a rank that cannot be read gets ``error``."""
    ranks = [row["rank"] for row in lock["site"]["ranks"]]

    def one(rank):
        try:
            return runner.remote(rank, "transport-receipts", timeout=LOG_SECONDS + 60)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            text = [line for line in str(error).splitlines() if line.strip()]
            return {"schema": REPORT_SCHEMA, "rank": rank, "error": (text[-1] if text else type(error).__name__)[:300]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(ranks))) as pool:
        return list(pool.map(one, ranks))


def expectation(section):
    if section["nccl"] == "never":
        return "SIRCL carries every collective; NCCL creates no communicator"
    return ("SIRCL's sessions carry the group's collectives; NCCL may carry what the cabling allows "
            f"({section['group']['cabling']} on this group)")


def _rows(receipts):
    """Calls per (collective, backend, method) of the tensor-parallel receipts of every rank, summed."""
    totals = {}
    for records in receipts.values():
        for record in records:
            if str(record.get("group", "")).split(":")[0] != "tp":
                continue
            for row in record.get("decisions") or []:
                key = (row.get("collective"), row.get("backend"), row.get("method"))
                totals[key] = totals.get(key, 0) + int(row.get("calls") or 0)
    return [{"collective": key[0], "backend": key[1], "method": key[2], "calls": calls}
            for key, calls in sorted(totals.items(), key=lambda item: tuple(map(str, item[0])))]


def settings_findings(receipts, expected):
    """``(lines, problems)``: each rank's tensor-parallel session against ``expected``, the statistics the
    tuning row and its SIRCL table set (``transport.expected_session_settings``)."""
    if not expected:
        return [], []
    lines, problems = [], []
    for rank, records in sorted(receipts.items()):
        record = next((item for item in records if str(item.get("group", "")).split(":")[0] == "tp"), None)
        stats = (record or {}).get("session_stats")
        if not isinstance(stats, dict) or "error" in stats:
            lines.append(f"tuning settings of rank {rank}: its receipt states no session statistics (not judged)")
            continue
        for stat, value in sorted(expected.items()):
            if stats.get(stat) != value:
                problems.append(f"rank {rank}: the session's {stat} is {stats.get(stat)}, the tuning row and its "
                                f"table set {value}")
    if not problems:
        lines.append("tuning settings: the sessions report the row's and its table's " + ", ".join(
            f"{stat} {value}" for stat, value in sorted(expected.items())))
    return lines, problems


def nccl_mode_findings(receipts, mode):
    """``(lines, problems)``: the NCCL mode each receipt names against the deployment's ``mode``."""
    lines, problems, unnamed = [], [], []
    for rank, records in sorted(receipts.items()):
        for record in records:
            named = record.get("nccl_mode")
            if named is None:
                unnamed.append(f"rank {rank} group {record.get('group')}")
            elif named != mode:
                problems.append(f"rank {rank} group {record.get('group')}: the receipt names NCCL mode {named}, the "
                                f"deployment sets {mode}")
    if unnamed:
        lines.append("NCCL mode: not stated by the receipts of " + ", ".join(unnamed[:8]) + " (not judged)")
    elif receipts and not problems:
        lines.append(f"NCCL mode: every receipt names {mode} ({transport.nccl_rule()})")
    return lines, problems


def evaluate(lock, reports, *, now=time.time):
    """The ``sparkring-transport-verdict/v1`` document of a SIRCL deployment's reports."""
    from spark_transport.sircl.sparkring_sircl.vllm.serve import checks
    section = lock["transport"]
    world = len(lock["site"]["ranks"])
    receipts, logs, problems, lines, unreadable = {}, {}, [], [], []
    for report in reports:
        rank = report.get("rank")
        if "error" in report:
            unreadable.append(f"rank {rank}: {report['error']}")
            continue
        for problem in report.get("problems") or []:
            problems.append(f"rank {rank}: {problem}")
        for record in report.get("receipts") or []:
            receipts.setdefault(int(record.get("global_rank", rank)), []).append(record)
        log = report.get("log") or {}
        logs[rank] = list(log.get("init") or []) + list(log.get("library") or [])
        if log and not log.get("complete", True):
            lines.append(f"rank {rank}: the log was read for {LOG_SECONDS} s and not to its end")
    dcp = section["group"]["dcp"]
    found, summary = checks.evaluate_receipts(receipts, world, dcp=dcp)
    problems += found
    lines += summary
    mode_lines, mode_problems = nccl_mode_findings(receipts, section["nccl"])
    lines += mode_lines
    problems += mode_problems
    if section["nccl"] == "never":
        free_lines, free_problems = checks.nccl_free_findings(receipts, world)
        log_lines, log_problems = checks.nccl_log_findings(logs, debug=True)
        lines += free_lines + log_lines
        problems += free_problems + log_problems
    # The table each kind of session takes (None: its rules choose), as the SIRCL launcher's check compares it.
    expected = {kind: (entry["hash"] if entry is not None else None)
                for kind, entry in (("tp", transport.session_table(section)),
                                    *((("dcp", transport.session_table(section, "dcp")),) if dcp > 1 else ()))}
    tuning_lines, tuning_problems = checks.tuning_findings(receipts, expected)
    lines += tuning_lines
    problems += tuning_problems
    blocks = section["tuning"]["settings"].get("large_blocks")
    if blocks is not None:
        block_lines, block_problems = checks.large_blocks_findings(receipts, blocks)
        lines += block_lines
        problems += block_problems
    setting_lines, setting_problems = settings_findings(receipts, transport.expected_session_settings(section))
    lines += setting_lines
    problems += setting_problems
    rows = _rows(receipts)
    nccl_rows = [row for row in rows if row["backend"] == "nccl"]
    log_hits = any(any(pattern in line for pattern in _patterns()[0]) for values in logs.values() for line in values)
    if unreadable:
        verdict, observed = "unknown", "unknown"
        problems = unreadable + problems
    else:
        verdict = "as-expected" if not problems else "differs"
        observed = "present" if nccl_rows or log_hits else "absent"
    return {"schema": VERDICT_SCHEMA, "backend": "sircl", "nccl": section["nccl"], "nccl_rule": transport.nccl_rule(),
            "expected": expectation(section),
            "verdict": verdict, "nccl_observed": observed, "fabric": section["fabric"]["id"],
            "group": section["group"]["name"], "checked_at": _iso(now()), "ranks_read": world - len(unreadable),
            "groups": {"tp": {"rows": rows}}, "problems": problems, "lines": lines}


def _iso(seconds):
    return datetime.datetime.fromtimestamp(float(seconds), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(seconds):
    return datetime.datetime.fromtimestamp(float(seconds), datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def record(directory, reports, verdict, *, now=time.time):
    """Copy the receipts and the verdict to ``<deployment>/receipts/<UTC time>/``; returns the verdict with its path."""
    directory = Path(directory)
    base = directory / RECEIPTS / _stamp(now())
    target, number = base, 1
    while target.exists():
        number += 1
        target = base.with_name(f"{base.name}-{number}")
    for report in reports:
        for receipt in report.get("receipts") or []:
            name = Path(str(receipt.get("file") or f"rank{report.get('rank')}.json")).name
            _write(target / name, {key: value for key, value in receipt.items() if key != "file"})
        _write(target / f"log-rank{report.get('rank')}.json",
               {key: report.get(key) for key in ("rank", "container", "problems", "error", "log")})
    result = {**verdict, "receipts": str(target)}
    _write(target / "verdict.json", result)
    _write(directory / LATEST, result)
    return result


def latest(directory):
    """The last recorded verdict of a deployment, or None."""
    try:
        value = json.loads((Path(directory) / LATEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("schema") == VERDICT_SCHEMA else None


def check(directory, lock, runner, *, now=time.time):
    """Gather, evaluate and record a SIRCL deployment's receipts; the recorded verdict."""
    reports = gather(runner, lock)
    return record(directory, reports, evaluate(lock, reports, now=now), now=now)


def text(verdict):
    """One line for the summary card and ``sparkring status``."""
    if verdict is None:
        return None
    if verdict["verdict"] == "unknown":
        return (f"Transport: sircl; receipts could not be read from every Spark ({verdict['problems'][0]}); "
                "sudo sparkring check repeats the check")
    if verdict["verdict"] == "differs":
        return f"Transport check failed: {verdict['problems'][0]}; receipts in {verdict.get('receipts', '-')}"
    if verdict["nccl"] == "never":
        return "Transport: sircl, NCCL: absent"
    return "Transport: sircl; NCCL " + ("carried collectives where the cabling allows, as expected"
                                        if verdict["nccl_observed"] == "present" else "unused")
