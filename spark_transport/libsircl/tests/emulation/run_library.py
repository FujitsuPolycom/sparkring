#!/usr/bin/env python3
"""Run the library's multi-process GPU emulation: W processes, one per rank, on one GPU.

Each rank is ``library_rank.py`` in its own process with the library's emulation transport
(``LIBSIRCL_TRANSPORT=emulation``) on a fabric segment of its own run (``SIRCL_EMU_FABRIC``). The
processes share the GPU by time slicing, so an op that waits for a peer costs a few milliseconds here;
the run checks correctness, not speed. Prints one line per check and rank, and a summary.

Usage (inside WSL, under the GPU lock):
  python run_library.py --library build/libsircl.so --world 2 [--lanes 1] [--golden DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--golden", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--json", default="")
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE for every rank")
    parser.add_argument("--work", default="", help="the work directory (rank logs, results, receipts); "
                        "default a new temporary directory")
    args = parser.parse_args(argv)
    if args.work:
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="sircl-ccl-emu-"))
    environment = dict(os.environ)
    environment.update({
        "LIBSIRCL_TRANSPORT": "emulation",
        "SIRCL_EMU_FABRIC": f"/sircl-emu-run-{os.getpid()}",
        "LIBSIRCL_EMU_LANES": str(args.lanes),
        "CUDA_DEVICE_MAX_CONNECTIONS": environment.get("CUDA_DEVICE_MAX_CONNECTIONS", "32"),
        "CUDA_MODULE_LOADING": "EAGER",
        "LIBSIRCL_RECEIPT": str(work / "receipt"),
    })
    for item in args.env:
        name, _, value = item.partition("=")
        environment[name] = value
    started = time.perf_counter()
    processes = []
    for rank in range(args.world):
        command = [args.python, str(HERE / "library_rank.py"), "--library", args.library, "--world",
                   str(args.world), "--rank", str(rank), "--id-file", str(work / "unique-id"),
                   "--out", str(work / f"rank{rank}.json")]
        if args.golden:
            command += ["--golden", args.golden]
        log = open(work / f"rank{rank}.log", "w")
        processes.append((subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT), log))
    deadline = time.monotonic() + args.timeout
    codes = []
    for process, log in processes:
        try:
            codes.append(process.wait(timeout=max(1.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            process.kill()
            codes.append(process.wait())
        log.close()
    elapsed = time.perf_counter() - started
    # Segments the ranks left (a rank that ended before destroy, or a destroy that kept its memory after a
    # failed release): every rank process has ended, so its segments are removed.
    for process, _ in processes:
        for segment in Path("/dev/shm").glob(f"sircl-emu-seg-{process.pid}-*"):
            segment.unlink(missing_ok=True)
    fabric = Path("/dev/shm") / environment["SIRCL_EMU_FABRIC"].lstrip("/")
    if fabric.exists():
        fabric.unlink()
    checks, failed = 0, 0
    summary = {"world": args.world, "lanes": args.lanes, "seconds": round(elapsed, 1), "ranks": []}
    for rank in range(args.world):
        path = work / f"rank{rank}.json"
        if not path.exists():
            print(f"FAIL rank {rank}: no result (exit {codes[rank]}); log {work / f'rank{rank}.log'}")
            print((work / f"rank{rank}.log").read_text()[-3000:])
            failed += 1
            continue
        data = json.loads(path.read_text())
        summary["ranks"].append(data)
        for name, ok, detail in data["checks"]:
            checks += 1
            failed += 0 if ok else 1
            if not ok or rank == 0:
                print(f"{'PASS' if ok else 'FAIL'} rank {rank} {name}{': ' + detail if detail else ''}")
    receipts = sorted(str(p) for p in work.glob("receipt.rank*.json"))
    summary["receipt_files"] = receipts
    # One receipt file per communicator: every process holds its communicator and the split children of its
    # checks (each rank's result says how many communicators it created), each under its own communicator
    # number, and every file is a whole receipt.
    try:
        parsed = [json.loads(Path(path).read_text()) for path in receipts]
        numbers = sorted((r["pid"], r["communicator"]) for r in parsed)
        whole = all(r.get("schema") == "libsircl-receipt/v1" for r in parsed)
        created = sum(rank.get("communicators", 2) for rank in summary["ranks"])
        ok = whole and len(parsed) == created and len(set(numbers)) == len(numbers)
        detail = f"{len(parsed)} files for {created} communicators, numbers {sorted({n for _, n in numbers})}"
    except (OSError, ValueError, KeyError) as error:
        ok, detail = False, f"{type(error).__name__}: {error}"
    checks += 1
    failed += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'} receipt files, one per communicator: {detail}")
    print(f"{checks} checks over {args.world} ranks, {failed} failed, {elapsed:.1f} s; work {work}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
