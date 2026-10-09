#!/usr/bin/env python3
"""Fail-stop (LIBSIRCL_FAIL_STOP) in GPU emulation: a late rank's peer ends its process within the wait limit.

Started without --rank, the script is the runner. It runs three cases of two ranks each on the one GPU over
the emulation transport, with a short startup wait limit (``--wait-s``, SIRCL_STARTUP_WAIT_S). In each, after
one exact all-reduce, rank 1 starts its next all-reduce ``--late-s`` seconds late; rank 0 enqueues its own at
once (the enqueue returns 0), waits for its stream as a caller that only checks enqueue codes does, keeps the
output and calls the library no more. Rank 0's flag wait times out at the wait limit.

- exit (``LIBSIRCL_FAIL_STOP=1``): the library's watcher writes the fail-stop line, naming the timed-out wait
  and the time, and ends rank 0's process with exit status 70. Passes when the line's time is at least 0.95
  of the wait limit after the enqueue (the wait ran its limit) and both the line and the process's end are
  within the wait limit plus ``--margin-s`` of it, and rank 0 never finished holding its output.
- abort (``LIBSIRCL_FAIL_STOP=abort``): the same line within the same bound, and the process ends by SIGABRT
  (after the system's core-dump handling, which this case does not bound).
- control (``LIBSIRCL_FAIL_STOP=0``): rank 0's stream wait returns at the wait limit (at least 0.95 of it)
  with a wrong output and no error from any call it made, and the process is still running past the bound:
  the silent wrong result that fail-stop ends.
- observe, destroy, commabort (``LIBSIRCL_FAIL_STOP=1`` with ``LIBSIRCL_FAIL_STOP_POLL_MS=600000``, so the
  watcher does not poll during the case): right after its stream wait rank 0 calls ncclCommGetAsyncError,
  ncclCommDestroy or ncclCommAbort, as a caller that reads the error and tears the communicator down at once
  does. The call must end the process with exit status 70 and the fail-stop line, naming the timed-out wait
  and saying that a library call found it (ncclCommAbort: "found by ncclCommDestroy or ncclCommAbort"),
  within the same bound; the call never returns to rank 0.

Every time is CLOCK_MONOTONIC (Python's time.monotonic, the line's second time): a VM's wall clock may be
stepped by its time synchronization while a rank waits (WSL2's moves back by over a second about every
30 s), which would make a full wait look short.

The runner ends rank 1 (and the control's rank 0) once rank 0's outcome is known: rank 1's own late
all-reduce can complete, since rank 0 sent its values before its wait timed out.

Status: research-only test infrastructure (the late rank models a peer that stops issuing collectives).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAIL_STOP_LINE = "LIBSIRCL_FAIL_STOP: ending the process at "
MONOTONIC_MARK = "CLOCK_MONOTONIC "
FAIL_STOP_STATUS = 70
MODES = {"exit": "1", "abort": "abort", "control": "0", "observe": "1", "destroy": "1", "commabort": "1"}
# The cases in which a library call, not the watcher, must find the error: the watcher's poll is longer than
# the case, and the call rank 0 makes right after its stream wait.
CALLS = {"observe": "ncclCommGetAsyncError", "destroy": "ncclCommDestroy", "commabort": "ncclCommAbort"}
# ncclCommDestroy's teardown asks for the error before the engine's own check at release.
FOUND_BY = {"observe": "found by a library call", "destroy": "found by a library call",
            "commabort": "found by ncclCommDestroy or ncclCommAbort"}


def run_rank(args) -> int:
    import torch

    import unique_id
    from library_rank import allreduce_reference, inputs_for, receipt_of, same_bits

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    lib = ctypes.CDLL(args.library)
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, unique_id.UniqueId,
                                     ctypes.c_int]
    lib.sirclGetReceipt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_size_t)]
    lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    world, rank, count = 2, args.rank, 4096
    out: dict = {"rank": rank}

    def save():
        # Replaced whole: fail-stop may end the process during a write, which must leave the previous result.
        partial = Path(args.out + ".partial")
        partial.write_text(json.dumps(out))
        os.replace(partial, args.out)

    uid = unique_id.share(lib, rank, world, args.id_file, "")
    comm = ctypes.c_void_p()
    code = lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank)
    if code:
        out["init"] = f"result {code}: {lib.ncclGetLastError(None).decode()}"
        save()
        return 1
    stream = torch.cuda.Stream()
    handle = ctypes.c_void_p(stream.cuda_stream)

    def all_reduce(seed):
        values = inputs_for(torch, world, count, torch.bfloat16, seed)
        x = values[rank].cuda()
        y = torch.full_like(x, 7.0)
        torch.cuda.synchronize()
        rc = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), count, 9, 0, comm,
                               handle)
        return rc, y, allreduce_reference(torch, values)

    rc, y, want = all_reduce(9500)
    torch.cuda.synchronize()
    out["healthy"] = {"result": rc, "exact": rc == 0 and same_bits(torch, y.cpu(), want)}
    out["fail_stop"] = receipt_of(lib, comm).get("fail_stop")
    save()
    if rank == args.late_rank:
        time.sleep(args.late_s)
    out["launched_at"] = time.monotonic()
    save()
    rc, y, want = all_reduce(9501)
    out["enqueue_result"] = rc
    save()
    # A caller that checks only enqueue codes: it waits for its stream, keeps the output and calls the library
    # no more; or (--after) one that reads the error or tears the communicator down right after the wait.
    torch.cuda.synchronize()
    if args.after:
        out["call_at"] = time.monotonic()
        save()
        if args.after == "ncclCommGetAsyncError":
            state = ctypes.c_int(0)
            rc = lib.ncclCommGetAsyncError(comm, ctypes.byref(state))
            out["call_returned"] = {"result": rc, "async_error": state.value}
        else:
            out["call_returned"] = {"result": getattr(lib, args.after)(comm)}
        save()
    out["consumed_at"] = time.monotonic()
    out["consumed_exact"] = same_bits(torch, y.cpu(), want)
    save()
    time.sleep(args.hold_s)
    out["held"] = True
    save()
    return 0


def run_case(args, work: Path, tag: str) -> list[str]:
    case = work / tag
    case.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "LIBSIRCL_TRANSPORT": "emulation", "SIRCL_EMU_FABRIC": f"/sircl-emu-failstop-{os.getpid()}-{tag}",
        "LIBSIRCL_EMU_LANES": "1", "CUDA_MODULE_LOADING": "EAGER",
        "CUDA_DEVICE_MAX_CONNECTIONS": env.get("CUDA_DEVICE_MAX_CONNECTIONS", "32"),
        "SIRCL_STARTUP_WAIT_S": str(args.wait_s), "LIBSIRCL_FAIL_STOP": MODES[tag],
    })
    if tag in CALLS:
        env["LIBSIRCL_FAIL_STOP_POLL_MS"] = "600000"
    processes, logs = [], []
    for rank in range(2):
        command = [args.python, str(Path(__file__).resolve()), "--rank", str(rank), "--library", args.library,
                   "--late-rank", "1", "--late-s", str(args.late_s), "--hold-s", str(args.hold_s),
                   "--id-file", str(case / "unique-id"), "--out", str(case / f"rank{rank}.json")]
        if tag in CALLS and rank == 0:
            command += ["--after", CALLS[tag]]
        log = open(case / f"rank{rank}.log", "w")
        logs.append(log)
        processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))

    def result(rank):
        try:
            return json.loads((case / f"rank{rank}.json").read_text())
        except (OSError, ValueError):
            return {}

    ended = None
    bound = args.wait_s + args.margin_s
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if processes[0].poll() is not None:
            ended = time.monotonic()
            break
        launched = result(0).get("launched_at")
        if tag == "control" and launched and time.monotonic() > launched + bound + 2:
            break
        time.sleep(0.02)
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGKILL)
        process.wait()
    for log in logs:
        log.close()
    for process in processes:
        for segment in Path("/dev/shm").glob(f"sircl-emu-seg-{process.pid}-*"):
            segment.unlink(missing_ok=True)
    fabric = Path("/dev/shm") / env["SIRCL_EMU_FABRIC"].lstrip("/")
    if fabric.exists():
        fabric.unlink()
    text = (case / "rank0.log").read_text(errors="replace")
    line = next((row for row in text.splitlines() if FAIL_STOP_LINE in row), "")
    data = [result(0), result(1)]
    problems = []
    for rank in range(2):
        if not data[rank].get("healthy", {}).get("exact"):
            problems.append(f"{tag}: rank {rank}'s first all-reduce not exact: {data[rank]}")
        if data[rank].get("fail_stop") is not (tag != "control"):
            problems.append(f"{tag}: rank {rank}'s receipt says fail_stop {data[rank].get('fail_stop')}")
    launched = data[0].get("launched_at")
    if data[0].get("enqueue_result") != 0:
        problems.append(f"{tag}: rank 0's late all-reduce enqueue returned {data[0].get('enqueue_result')}")
    code = processes[0].returncode
    if tag == "control":
        if ended is not None and launched and ended < launched + bound:
            problems.append(f"{tag}: rank 0 ended {ended - launched:.2f} s after its enqueue, exit {code}")
        if "consumed_at" not in data[0] or data[0].get("consumed_exact"):
            problems.append(f"{tag}: rank 0 did not keep a wrong output: {data[0]}")
        elif launched and data[0]["consumed_at"] - launched < 0.95 * args.wait_s:
            problems.append(f"{tag}: rank 0's wait ended {data[0]['consumed_at'] - launched:.3f} s after the "
                            f"enqueue, before the {args.wait_s} s wait limit")
        if line:
            problems.append(f"{tag}: the fail-stop line without fail-stop")
        waited = data[0]["consumed_at"] - launched if launched and "consumed_at" in data[0] else None
        print(f"{tag}: rank 0 enqueue {data[0].get('enqueue_result')}; its stream wait returned "
              f"{waited if waited is None else round(waited, 2)} s after the enqueue with the output "
              f"{'exact' if data[0].get('consumed_exact') else 'wrong'}; still running {bound + 2} s after it")
        return problems
    stamp = None
    if line:
        try:
            stamp = float(line.split(MONOTONIC_MARK, 1)[1].split(" ", 1)[0])
        except ValueError:
            stamp = None
    said = stamp - launched if stamp is not None and launched else None
    if said is None or said > bound or said < 0.95 * args.wait_s:
        problems.append(f"{tag}: fail-stop line {said} s after the enqueue (between {0.95 * args.wait_s:.2f} and "
                        f"{bound} s): {line[:200]!r}")
    if "timed out" not in line:
        problems.append(f"{tag}: the fail-stop line does not name the timed-out wait")
    if data[0].get("held"):
        problems.append(f"{tag}: rank 0 held its output to the end")
    if tag in CALLS:
        if "call_at" not in data[0]:
            problems.append(f"{tag}: rank 0 ended before it called {CALLS[tag]} (the watcher's poll is 600 s)")
        if "call_returned" in data[0]:
            problems.append(f"{tag}: {CALLS[tag]} returned {data[0]['call_returned']} to rank 0")
        if FOUND_BY[tag] not in line:
            problems.append(f"{tag}: the fail-stop line does not say {FOUND_BY[tag]!r}")
    gone = ended - launched if ended is not None and launched else None
    if tag == "exit" or tag in CALLS:
        if code != FAIL_STOP_STATUS:
            problems.append(f"{tag}: rank 0 exit {code}, not {FAIL_STOP_STATUS}")
        if gone is None or gone > bound:
            problems.append(f"{tag}: rank 0 ended {gone} s after its enqueue (bound {bound} s)")
    elif code != -signal.SIGABRT:
        problems.append(f"{tag}: rank 0 exit {code}, not SIGABRT")
    print(f"{tag}: rank 0 enqueue {data[0].get('enqueue_result')}; fail-stop line "
          f"{said if said is None else round(said, 3)} s and process end {gone if gone is None else round(gone, 2)} s "
          f"after the enqueue (bound {bound} s), exit {code}; {line[:260]}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait-s", type=float, default=2.0, help="SIRCL_STARTUP_WAIT_S of both ranks")
    parser.add_argument("--late-s", type=float, default=12.0, help="how late rank 1 starts its second all-reduce")
    parser.add_argument("--margin-s", type=float, default=3.0, help="allowed beyond the wait limit")
    parser.add_argument("--hold-s", type=float, default=30.0, help="how long rank 0 keeps running after its output")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cases", default="exit,abort,control,observe,destroy,commabort",
                        help="a comma list of exit, abort, control, observe, destroy, commabort")
    parser.add_argument("--work", default="")
    parser.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--late-rank", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--after", default="", help=argparse.SUPPRESS)
    parser.add_argument("--id-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.rank >= 0:
        return run_rank(args)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="sircl-failstop-"))
    work.mkdir(parents=True, exist_ok=True)
    problems = []
    for tag in args.cases.split(","):
        problems += run_case(args, work, tag)
    for line in problems:
        print(f"FAIL {line}")
    print(f"fail-stop: {len(problems)} problems; work {work}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
