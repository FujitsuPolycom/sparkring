"""Decode and prefill throughput with llm-inference-bench, and repository-safe matrices.

The benchmark program, `llm_decode_bench.py` from llm-inference-bench, lives
outside this repository; the caller names its directory. Every cell runs at
temperature 1.0 with exact token targeting, 20 s after a 5 s warm-up, with up
to 2,048 output tokens.

- The standard measurement, the README profile table's figures: 1, 4, 8 and
  16 streams, each with 16K tokens of added context, plus the benchmark's
  default cold scout-only prefill prompts of 8K, 64K and 128K tokens.
- The full matrix: 1, 2, 4, 8 and 16 streams at 8K, 32K, 64K and 128K tokens
  of added context, without prefill prompts.

The benchmark reads the deployment's KV cache capacity from vLLM's metrics
(`--dcp-size 1`: installer profiles run no decode context parallelism) and
skips a cell whose streams need more, concurrency x (context + 2,048)
tokens; a cell the server queued (more streams than it runs at once) is
marked capacity-limited. Neither is a failure: both print as a dash with the
reason.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import statistics

BENCH_PROGRAM = "llm_decode_bench.py"
CONCURRENCY = (1, 4, 8, 16)
CONTEXT = 16384
FULL_CONCURRENCY = (1, 2, 4, 8, 16)
FULL_CONTEXTS = (8192, 32768, 65536, 131072)
PREFILL = (8192, 65536, 131072)
TEMPERATURE = 1.0
# Cells that did not run as measured decode, with the reason printed for them.
NOT_FITTING = "exceeds the KV cache"
QUEUED = "queued: more streams than the server runs at once"
KEEP = ("metadata", "prefill", "results", "summary_table", "burst_results", "burst_summary_table", "methodology")
SERVER_PLACEHOLDER = "http://NODE_A"
PRIVATE_ADDRESS = re.compile(
    r"(?<![\d.])(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))"
    r"\.\d{1,3}\.\d{1,3}(?![\d]|\.\d)")


def bench_command(python, bench_dir, *, host, port, model, output, full=False):
    """The benchmark invocation: the standard measurement, or the full matrix with ``full``.

    `--display-mode plain` suits output captured to a log file.
    """
    concurrency, contexts = (FULL_CONCURRENCY, FULL_CONTEXTS) if full else (CONCURRENCY, (CONTEXT,))
    command = [str(python), str(Path(bench_dir) / BENCH_PROGRAM), "--host", host, "--port", str(port), "--model", model,
               "--concurrency", ",".join(map(str, concurrency)), "--contexts", ",".join(map(str, contexts)),
               "--duration", "20", "--decode-warmup-seconds", "5", "--cell-warmup-timeout-seconds", "900",
               "--max-tokens", "2048", "--temperature", str(TEMPERATURE), "--token-targeting", "exact",
               "--dcp-size", "1", "--no-hw-monitor", "--no-resume", "--display-mode", "plain", "--output", str(output)]
    return command + (["--skip-prefill"] if full else [])


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _check_temperature(matrix):
    temperature = matrix.get("metadata", {}).get("temperature")
    if temperature is not None and float(temperature) != TEMPERATURE:
        raise ValueError(f"matrix temperature is {temperature}; acceptance matrices use {TEMPERATURE}")


def _cell(row):
    """(cell, reason): a measured cell's rates, or no rates and why."""
    rate = _number(row.get("aggregate_tps"))
    cell = {"aggregate_tps": None, "server_steps_per_s": None, "server_spec_accept_length": None,
            "num_errors": int(row.get("num_errors") or 0)}
    reason = row.get("failure_reason") or ("invalid cell" if rate is not None and rate < 0 else "")
    if reason:
        return cell, reason
    if row.get("capacity_limited"):
        return {**cell, "not_applicable": QUEUED}, ""
    cell.update(aggregate_tps=rate, server_steps_per_s=_number(row.get("server_steps_per_s")),
                server_spec_accept_length=_number(row.get("server_spec_accept_length")))
    return cell, ""


def extract(matrix):
    """Per-concurrency decode figures at 16K context and prefill rates of one standard matrix.

    A cell whose aggregate rate is negative (the benchmark's marker for an
    invalid cell, such as a detected output loop) or that names a failure
    reason contributes no rates and is listed under `invalid`. A cell the
    benchmark skipped or the server queued contributes no rates and carries
    `not_applicable` with the reason.
    """
    _check_temperature(matrix)
    metadata = matrix.get("metadata", {})
    decode, invalid = {}, []
    for row in matrix.get("results", []):
        if row.get("context_tokens") != CONTEXT or row.get("concurrency") not in CONCURRENCY:
            continue
        cell, reason = _cell(row)
        if reason:
            invalid.append({"concurrency": row["concurrency"], "reason": reason})
        decode[str(row["concurrency"])] = cell
    for level in CONCURRENCY:
        decode.setdefault(str(level), {"aggregate_tps": None, "server_steps_per_s": None,
                                       "server_spec_accept_length": None, "num_errors": 0,
                                       "not_applicable": NOT_FITTING})
    prefill = {}
    for length in PREFILL:
        entry = matrix.get("prefill", {}).get(str(length)) or {}
        prefill[str(length)] = _number(entry.get("tok_per_sec"))
    return {"version": metadata.get("version"), "decode": decode, "prefill": prefill, "invalid": invalid}


def extract_full(matrix):
    """The full matrix's decode rates as {context: {concurrency: cell}}, with its KV cache budget.

    Every grid cell is present: measured cells carry `aggregate_tps`; skipped,
    queued and failed cells carry `not_applicable` or `failure` instead.
    """
    _check_temperature(matrix)
    metadata = matrix.get("metadata", {})
    rows = {(row.get("context_tokens"), row.get("concurrency")): row for row in matrix.get("results", [])}
    table = {}
    for context in FULL_CONTEXTS:
        for level in FULL_CONCURRENCY:
            row = rows.get((context, level))
            if row is None:
                cell = {"aggregate_tps": None, "not_applicable": NOT_FITTING}
            else:
                measured, reason = _cell(row)
                cell = {key: measured[key] for key in ("aggregate_tps", "not_applicable") if key in measured}
                if reason:
                    cell["failure"] = reason
            table.setdefault(str(context), {})[str(level)] = cell
    budget = metadata.get("kv_budget") or metadata.get("max_total_tokens")
    return {"version": metadata.get("version"), "kv_budget": int(budget) if budget else None, "decode": table}


def combine(runs):
    """Median of each rate across runs; request errors are summed so a median cannot hide them.

    A cell that is not applicable in every run (skipped or queued) is expected
    and keeps its reason; it does not make the measurement incomplete.
    """
    def median(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None
    decode = {}
    for level in map(str, CONCURRENCY):
        cells = [run["decode"].get(level) for run in runs]
        present = [cell for cell in cells if cell]
        decode[level] = {key: median(cell[key] for cell in present)
                         for key in ("aggregate_tps", "server_steps_per_s", "server_spec_accept_length")}
        decode[level]["num_errors"] = sum(cell["num_errors"] for cell in present)
        decode[level]["missing_runs"] = len(cells) - len(present)
        reasons = {cell.get("not_applicable") for cell in present}
        if present and len(reasons) == 1 and None not in reasons:
            decode[level]["not_applicable"] = reasons.pop()
    prefill = {length: median(run["prefill"].get(length) for run in runs) for length in map(str, PREFILL)}
    invalid = [dict(item, run=index + 1) for index, run in enumerate(runs) for item in run["invalid"]]
    versions = sorted({run["version"] for run in runs if run["version"]})
    complete = all((cell["aggregate_tps"] is not None or cell.get("not_applicable")) and not cell["missing_runs"]
                   for cell in decode.values())
    errors = sum(cell["num_errors"] for cell in decode.values())
    return {"runs": len(runs), "versions": versions, "decode": decode, "prefill": prefill, "invalid": invalid,
            "ok": complete and not invalid and not errors}


class PrivateDataError(ValueError):
    """A document still names a private address, host or account."""


def _tokens(value):
    return re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(value) + r"(?![A-Za-z0-9_-])")


def private_findings(text, *, names=(), accounts=()):
    """Describe each private item in `text`.

    `names` (hostnames, SSH aliases) are matched as whole tokens. `accounts`
    are matched as whole tokens as well; callers pass SSH users here only for
    documents without model-generated text, where a user name such as a common
    word cannot occur by chance. `account_forms` below covers the rest.
    """
    found = [f"private address {m.group(0)}" for m in PRIVATE_ADDRESS.finditer(text)]
    found += [f"name {name}" for name in names if name and _tokens(name).search(text)]
    found += [f"account {user}" for user in accounts if user and _tokens(user).search(text)]
    return found


def account_forms(text, users):
    """SSH users written as accounts: `user@`, `/home/user`, `/Users/user`."""
    found = []
    for user in filter(None, users):
        escaped = re.escape(user)
        if re.search(rf"(?<![A-Za-z0-9_.-]){escaped}@|/(?:home|Users)/{escaped}(?![A-Za-z0-9_-])", text):
            found.append(f"account {user}")
    return found


def sanitize_matrix(matrix, *, names=(), accounts=()):
    """Keep the benchmark sections a record needs and hide the server address.

    Raises PrivateDataError, naming each finding, when a private address,
    one of `names` or one of `accounts` remains anywhere in the result.
    """
    missing = [key for key in ("metadata", "prefill", "results", "summary_table") if key not in matrix]
    if missing:
        raise ValueError("benchmark matrix lacks " + ", ".join(missing))
    result = {key: json.loads(json.dumps(matrix[key])) for key in KEEP if key in matrix}
    result["metadata"]["server"] = SERVER_PLACEHOLDER
    findings = private_findings(json.dumps(result, ensure_ascii=False), names=names, accounts=accounts)
    if findings:
        raise PrivateDataError("refusing to write the matrix; it still contains " + "; ".join(sorted(set(findings))))
    return result
