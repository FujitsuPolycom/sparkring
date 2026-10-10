#!/usr/bin/env python3
"""Evaluate the pair's nccl-tests exit criteria from both ranks' outputs (RUNBOOK.md sections 3.4 and 3.5).

Reads, in each rank's output directory of ``tools/nccl_tests_pair.sh run``, the job manifest
``jobs.tsv`` (one row per line run: the job, its log file, its exit status), the logs it names and every
``receipt.rank*.json``, and prints one verdict per criterion:

- every output directory has a job manifest, every rank ran the same jobs in the same order, and, with
  ``--expect-lines FILE`` (the lines file given to the runner, or any file of "<binary> <arguments>" rows),
  every expected job ran;
- every job exited 0 on every rank;
- every job completed: one rank's log (nccl-tests prints on its main rank only) has data rows, its sweep
  reached its largest size (a row at least ``-e`` divided by ``-f``), "Out of bounds values : 0 OK" and the
  footer "Collective test concluded";
- every data row has ``#wrong`` 0, out of place and in place. The tests that define no in-place result
  (``NO_IN_PLACE``: ``alltoall_perf``, ``alltoallv_perf`` and ``sendrecv_perf``, whose in-place ``#wrong``
  nccl-tests v2.21.1 prints as ``N/A`` on every row) have their in-place ``N/A`` counted as not covered,
  never as passed or wrong; their out-of-place ``#wrong`` must be 0 and an in-place count other than
  ``N/A`` must be 0. An in-place ``N/A`` of any other test is wrong. Rows of tests without a reduction op
  (``hypercube_perf``) have that column blank; a line that starts like a data row (size and count) but does
  not parse as one fails this criterion, so no row passes unchecked;
- every receipt has ``"forwarded":0``, no refusals, ``"healthy":true``, and all-reduce ops (transport,
  fold, chain and ring) covering its all-reduce calls;
- the eager out-of-place time of the 8192-byte rows is at or below ``--eager-limit-us`` (20);
- with ``--graph-reference-us`` (the SIRCL ring harness's graph p50 at 8 KiB on the same pair), the
  graph out-of-place time of the 8192-byte rows is within ``--graph-margin-us`` (1) of it.

Exit status 0 when every evaluated criterion holds.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

# A data row starts with the size and the element count; the columns after them are the type, the reduction
# op (blank in tests without one, such as hypercube_perf), the root, and four out-of-place and four in-place
# columns (time, algorithm and bus bandwidth, #wrong).
DATA = re.compile(r"^\s*\d+\s+\d+\s")
INTEGER = re.compile(r"-?\d+")
# nccl-tests binaries without an in-place result: their in-place column has times but "N/A" for #wrong.
NO_IN_PLACE = frozenset({"alltoall_perf", "alltoallv_perf", "sendrecv_perf"})


def binary(job: str) -> str:
    """The nccl-tests binary of a job ("<binary> <arguments>"), without its directory."""
    words = job.split()
    return Path(words[0]).name if words else ""


def number(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def parse_row(line: str):
    """The fields of one data row, or None when the line is not one."""
    if not DATA.match(line):
        return None
    words = line.split()
    if len(words) < 12:
        return None
    if INTEGER.fullmatch(words[3]):  # no reduction op: the root follows the type
        op, root, rest = "", words[3], words[4:]
    else:
        op, root, rest = words[3], words[4], words[5:]
    if not INTEGER.fullmatch(root) or len(rest) < 8:
        return None
    return {"size": int(words[0]), "type": words[2], "op": op,
            "out_us": number(rest[0]), "out_wrong": rest[3], "in_us": number(rest[4]), "in_wrong": rest[7]}


def rows(log: Path):
    """(size, type, op, out-of-place time, out wrong, in-place time, in wrong) of each data row."""
    for line in log.read_text(errors="replace").splitlines():
        row = parse_row(line)
        if row is not None:
            yield row


def unread(log: Path) -> list[str]:
    """Lines that start like a data row (size and count) but do not parse as one: never passed unchecked."""
    return [line.strip() for line in log.read_text(errors="replace").splitlines()
            if DATA.match(line) and parse_row(line) is None]


def manifest(directory: Path):
    """The job rows of a directory's jobs.tsv: (job, log name, status), or None without one."""
    path = directory / "jobs.tsv"
    if not path.exists():
        return None
    jobs = []
    for line in path.read_text().splitlines():
        if line.strip():
            job, log, status = line.split("\t")
            jobs.append((job, log, int(status)))
    return jobs


def option(job: str, name: str):
    words = job.split()
    return words[words.index(name) + 1] if name in words[:-1] else None


def size_bytes(text: str) -> int:
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(float(text[:-1]) * scale[text[-1].upper()]) if text[-1].upper() in scale else int(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--expect-lines", type=Path, default=None)
    parser.add_argument("--eager-limit-us", type=float, default=20.0)
    parser.add_argument("--graph-reference-us", type=float, default=None)
    parser.add_argument("--graph-margin-us", type=float, default=1.0)
    args = parser.parse_args(argv)
    verdicts, failed = [], False

    def verdict(ok, text):
        nonlocal failed
        failed |= not ok
        verdicts.append(f"{'PASS' if ok else 'FAIL'} {text}")

    manifests = {directory: manifest(directory) for directory in args.outputs}
    missing = [str(d) for d, m in manifests.items() if m is None]
    present = {d: m for d, m in manifests.items() if m is not None}
    orders = {tuple(job for job, _, _ in m) for m in present.values()}
    verdict(not missing and len(orders) == 1 and bool(next(iter(orders), ())),
            f"job manifests in {len(present)} of {len(manifests)} directories, the same jobs on every rank"
            + (f"; no jobs.tsv in {missing}" if missing else "") + ("" if len(orders) <= 1 else "; the job lists differ"))
    jobs = list(next(iter(orders), ()))
    if args.expect_lines is not None:
        expected = [" ".join(shlex.split(row)) for row in args.expect_lines.read_text().splitlines()
                    if row.strip() and not row.lstrip().startswith("#")]
        absent = [job for job in expected if job not in jobs]
        verdict(not absent, f"{len(expected)} expected jobs ran" + (f"; not run: {absent}" if absent else ""))
    bad_exit = [(str(d), job, status) for d, m in present.items() for job, _, status in m if status != 0]
    verdict(not bad_exit, f"every job exited 0 on every rank ({sum(len(m) for m in present.values())} runs)"
            + (f"; failed: {bad_exit[:10]}" if bad_exit else ""))
    data, texts, unparsed = {}, {}, []
    for directory, m in present.items():
        for job, log, _ in m:
            path = directory / log
            text = path.read_text(errors="replace") if path.exists() else ""
            found = list(rows(path)) if path.exists() else []
            if found:
                data.setdefault(job, []).append((path, found))
            if path.exists():
                unparsed += [(path.name, line) for line in unread(path)]
            texts.setdefault(job, []).append(text)
    incomplete = []
    for job in jobs:
        logs = data.get(job, [])
        whole = "\n".join(texts.get(job, []))
        last, factor = option(job, "-e"), option(job, "-f")
        reached = True
        if logs and last and factor:
            biggest = max(r["size"] for _, found in logs for r in found)
            reached = biggest * float(factor) > size_bytes(last)
        if not logs or not reached or "Out of bounds values : 0 OK" not in whole or "Collective test concluded" not in whole:
            incomplete.append(job)
    verdict(not incomplete, f"{len(jobs) - len(incomplete)} of {len(jobs)} jobs completed their sweep"
            + (f"; incomplete: {incomplete[:10]}" if incomplete else ""))
    every = [(job, path, r) for job, logs in data.items() for path, found in logs for r in found]
    wrong, uncovered = [], {}
    for job, path, r in every:
        in_na = r["in_wrong"] == "N/A" and binary(job) in NO_IN_PLACE
        if in_na:
            uncovered[binary(job)] = uncovered.get(binary(job), 0) + 1
        if r["out_wrong"] != "0" or (r["in_wrong"] != "0" and not in_na):
            wrong.append((path.name, r["size"], r["out_wrong"], r["in_wrong"]))
    verdict(not wrong and not unparsed, f"#wrong 0 on {len(every)} rows"
            + (f"; wrong (log, size, out of place, in place): {wrong[:10]}" if wrong else "")
            + (f"; {len(unparsed)} data rows not parsed: {unparsed[:5]}" if unparsed else ""))
    if uncovered:
        counts = ", ".join(f"{name} {count}" for name, count in sorted(uncovered.items()))
        verdicts.append(f"INFO {sum(uncovered.values())} in-place rows N/A ({counts}): not covered (these tests "
                        "have no in-place result)")
    receipts = sorted(p for directory in args.outputs for p in directory.glob("receipt.rank*.json"))
    bad = []
    for path in receipts:
        receipt = json.loads(path.read_text())
        # libsircl-receipt/v1; receipts of builds named SIRCL-CCL carry sirclccl-receipt/v1.
        if receipt.get("schema", "libsircl-receipt/v1") not in ("libsircl-receipt/v1", "sirclccl-receipt/v1"):
            bad.append(f"{path.name}: schema {receipt.get('schema')!r}")
            continue
        ops = (sum(receipt["all_reduce"]["ops"].values()) + sum(receipt.get("fold", {}).get("ops", {}).values())
               + receipt.get("chain", {}).get("ops", 0) + receipt.get("links", {}).get("ops", {}).get("ring_reduce", 0))
        if (receipt["forwarded"] != 0 or any(receipt["refused"].values()) or not receipt["healthy"]
                or ops < receipt["all_reduce"]["calls"]):
            bad.append(path.name)
    verdict(bool(receipts) and not bad, f"{len(receipts)} receipts forwarded 0, no refusals, healthy, ops cover calls"
            + (f"; failing: {bad}" if bad else ""))
    named = [(path, found) for logs in data.values() for path, found in logs]
    eager = [(path.name, r["out_us"]) for path, found in named if "-G_0" in path.name
             for r in found if r["size"] == 8192 and r["out_us"] is not None]
    if eager:
        worst = max(us for _, us in eager)
        verdict(worst <= args.eager_limit_us, f"eager 8192-byte out-of-place time at most {worst:.2f} us "
                f"(limit {args.eager_limit_us} us) over {len(eager)} rows")
    graph = [(path.name, r["out_us"]) for path, found in named if "-G_20" in path.name
             for r in found if r["size"] == 8192 and r["out_us"] is not None]
    if graph:
        worst = max(us for _, us in graph)
        if args.graph_reference_us is None:
            verdicts.append(f"INFO graph 8192-byte out-of-place time at most {worst:.2f} us over {len(graph)} rows; "
                            "pass --graph-reference-us to evaluate the graph criterion")
        else:
            verdict(worst <= args.graph_reference_us + args.graph_margin_us,
                    f"graph 8192-byte time at most {worst:.2f} us against {args.graph_reference_us} us "
                    f"+ {args.graph_margin_us} us")
    print("\n".join(verdicts))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
