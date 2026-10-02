#!/usr/bin/env python3
"""Summarize the decode benchmark runs and acceptance probes of this record.

For each variant (a file-name label such as ``b12x`` or ``humming``), the
decode runs ``decode-<label>-r<n>.json`` (llm-inference-bench 0.6.2
``llm_decode_bench.py``) give per concurrency the aggregate tokens per second,
the server's MTP accept length, and steps per second (tokens per second
divided by accept length: verification steps per second). Probe files
``probe-<label>-*.json`` (spec_accept_probe.py) give the acceptance rate and
accept length over the fixed seeded request set. Each decode run also
records one prefill measurement per prompt length. With ``--json OUT`` it also
writes the per-run values and each source file's SHA-256 to OUT.

    python3 summarize.py DIRECTORY [--json OUT]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


def decode_rows(directory):
    rows = {}
    for path in sorted(Path(directory).glob("decode-*-r*.json")):
        label = re.fullmatch(r"decode-(.+)-r\d+\.json", path.name).group(1)
        for cell in json.loads(path.read_text(encoding="utf-8"))["results"]:
            accept = cell["server_spec_accept_length"]
            rows.setdefault((label, cell["concurrency"]), []).append(
                {"run": path.name, "tps": cell["aggregate_tps"], "accept_length": accept,
                 "steps": cell["aggregate_tps"] / accept, "accept_rate": cell["server_spec_accept_rate"]})
    return rows


def span(values, digits):
    return (f"{statistics.mean(values):.{digits}f} ({min(values):.{digits}f} – {max(values):.{digits}f})")


def compact(directory):
    """Per-run decode cells and probe summaries, with each source file's SHA-256."""
    runs = []
    for path in sorted(Path(directory).glob("decode-*-r*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "bench_version": data["metadata"]["version"], "timestamp": data["metadata"]["timestamp"],
                     "cells": [{key: cell[key] for key in (
                         "concurrency", "context_tokens", "measurement_seconds", "aggregate_tps",
                         "server_gen_throughput", "server_spec_accept_rate", "server_spec_accept_length",
                         "server_spec_drafts", "num_errors")} for cell in data["results"]],
                     "prefill": {context: {"prompt_tokens": row["prompt_tokens"], "ttft_seconds": row["ttft_seconds"],
                                           "tok_per_sec": row["tok_per_sec"]}
                                 for context, row in data["prefill"].items()}})
    probes = []
    for path in sorted(Path(directory).glob("probe-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        probes.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       **{key: value for key, value in data.items() if key != "responses"}})
    return {"decode_runs": runs, "probes": probes}


def main(directory):
    if len(sys.argv) > 3 and sys.argv[2] == "--json":
        Path(sys.argv[3]).write_text(json.dumps(compact(directory), indent=2) + "\n", encoding="utf-8")
    print("| Draft backend | Streams | tok/s mean (range) | steps/s mean (range) | accept length mean (range) | runs |")
    print("|---|---|---|---|---|---|")
    for (label, concurrency), runs in sorted(decode_rows(directory).items()):
        print(f"| {label} | {concurrency} | {span([r['tps'] for r in runs], 1)} | "
              f"{span([r['steps'] for r in runs], 1)} | {span([r['accept_length'] for r in runs], 2)} | {len(runs)} |")
    print()
    prefill = {}
    for path in sorted(Path(directory).glob("decode-*-r*.json")):
        label = re.fullmatch(r"decode-(.+)-r\d+\.json", path.name).group(1)
        for context, row in json.loads(path.read_text(encoding="utf-8"))["prefill"].items():
            prefill.setdefault((label, int(context)), []).append(row["tok_per_sec"])
    print("| Draft backend | Prompt tokens | Prefill tok/s mean (range) | runs |")
    print("|---|---|---|---|")
    for (label, context), values in sorted(prefill.items()):
        print(f"| {label} | {context} | {span(values, 0)} | {len(values)} |")
    print()
    print("| Probe | Requests | Completion tokens | Acceptance rate | Accept length | Per position |")
    print("|---|---|---|---|---|---|")
    for path in sorted(Path(directory).glob("probe-*.json")):
        probe = json.loads(path.read_text(encoding="utf-8"))
        positions = " / ".join(f"{value:.3f}" for value in probe["per_position_acceptance"].values())
        print(f"| {path.stem} | {probe['requests']} | {probe['completion_tokens']} | "
              f"{probe['acceptance_rate']:.3f} | {probe['accept_length']:.2f} | {positions} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
