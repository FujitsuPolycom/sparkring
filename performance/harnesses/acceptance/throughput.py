"""Decode and prefill throughput with llm-inference-bench, and repository-safe matrices.

The benchmark program, `llm_decode_bench.py` from llm-inference-bench, lives
outside this repository; the caller names its directory. Every run uses the
settings of the installer profile records: temperature 1.0, exact token
targeting, 1, 8 and 16 streams without added context, 20 s cells after a 5 s
warm-up and up to 2,048 output tokens, plus the benchmark's default cold
scout-only prefill prompts of 8K, 64K and 128K tokens.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import statistics

BENCH_PROGRAM = "llm_decode_bench.py"
CONCURRENCY = (1, 8, 16)
PREFILL = (8192, 65536, 131072)
TEMPERATURE = 1.0
KEEP = ("metadata", "prefill", "results", "summary_table", "burst_results", "burst_summary_table", "methodology")
SERVER_PLACEHOLDER = "http://NODE_A"
PRIVATE_ADDRESS = re.compile(
    r"(?<![\d.])(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))"
    r"\.\d{1,3}\.\d{1,3}(?![\d]|\.\d)")


def bench_command(python, bench_dir, *, host, port, model, output):
    """The benchmark invocation; `--display-mode plain` suits output captured to a log file."""
    return [str(python), str(Path(bench_dir) / BENCH_PROGRAM), "--host", host, "--port", str(port), "--model", model,
            "--concurrency", ",".join(map(str, CONCURRENCY)), "--contexts", "0", "--duration", "20",
            "--decode-warmup-seconds", "5", "--max-tokens", "2048", "--temperature", str(TEMPERATURE),
            "--token-targeting", "exact", "--no-hw-monitor", "--no-resume", "--display-mode", "plain",
            "--output", str(output)]


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def extract(matrix):
    """Per-concurrency decode figures and prefill rates of one benchmark matrix.

    A cell whose aggregate rate is negative (the benchmark's marker for an
    invalid cell, such as a detected output loop) or that names a failure
    reason contributes no rates and is listed under `invalid`.
    """
    metadata = matrix.get("metadata", {})
    temperature = metadata.get("temperature")
    if temperature is not None and float(temperature) != TEMPERATURE:
        raise ValueError(f"matrix temperature is {temperature}; acceptance matrices use {TEMPERATURE}")
    decode, invalid = {}, []
    for row in matrix.get("results", []):
        if row.get("context_tokens") != 0 or row.get("concurrency") not in CONCURRENCY:
            continue
        rate = _number(row.get("aggregate_tps"))
        reason = row.get("failure_reason") or ("invalid cell" if rate is not None and rate < 0 else "")
        cell = {"aggregate_tps": None, "server_steps_per_s": None, "server_spec_accept_length": None,
                "num_errors": int(row.get("num_errors") or 0)}
        if reason:
            invalid.append({"concurrency": row["concurrency"], "reason": reason})
        else:
            cell.update(aggregate_tps=rate, server_steps_per_s=_number(row.get("server_steps_per_s")),
                        server_spec_accept_length=_number(row.get("server_spec_accept_length")))
        decode[str(row["concurrency"])] = cell
    prefill = {}
    for length in PREFILL:
        entry = matrix.get("prefill", {}).get(str(length)) or {}
        prefill[str(length)] = _number(entry.get("tok_per_sec"))
    return {"version": metadata.get("version"), "decode": decode, "prefill": prefill, "invalid": invalid}


def combine(runs):
    """Median of each rate across runs; request errors are summed so a median cannot hide them."""
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
    prefill = {length: median(run["prefill"].get(length) for run in runs) for length in map(str, PREFILL)}
    invalid = [dict(item, run=index + 1) for index, run in enumerate(runs) for item in run["invalid"]]
    versions = sorted({run["version"] for run in runs if run["version"]})
    complete = all(cell["aggregate_tps"] is not None and not cell["missing_runs"] for cell in decode.values())
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
