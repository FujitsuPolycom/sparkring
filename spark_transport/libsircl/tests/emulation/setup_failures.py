#!/usr/bin/env python3
"""Communicator setup failures in GPU emulation: every rank fails promptly and names every rank's reason.

Each case starts two processes on one GPU that call ``ncclCommInitRank`` with settings that cannot form a
session, and checks that both return an error within the deadline and that the error text (from
``ncclGetLastError``) names the failing rank and its reason:

- the verbs transport without a route map;
- a route map naming an RDMA device the host does not have (libibverbs loaded at run time);
- RoCE v2 GID resolution over a sysfs tree with two candidate entries (``LIBSIRCL_SYSFS_INFINIBAND``);
- a setting that must agree differing between the ranks (``SIRCL_ONESHOT_MAX_BYTES``);
- an invalid setting on one rank (``SIRCL_THREADS``);
- a ring schedule without the layout's ring plan (``SIRCL_SCATTER_SCHEDULE=ring`` without
  ``LIBSIRCL_RING_WINDOW``, and ``SIRCL_LARGE_SCHEDULE=pieces``, which turns off the pair plan and with it
  the ring plan a cabled pair gets by default);
- a schedule differing between the ranks (``SIRCL_GATHER_SCHEDULE``).
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

RANK = r'''
import ctypes, json, sys, time
import torch
torch.cuda.set_device(0); torch.zeros(1, device="cuda")
class U(ctypes.Structure): _fields_ = [("b", ctypes.c_ubyte * 128)]
lib = ctypes.CDLL(sys.argv[1]); lib.ncclGetLastError.restype = ctypes.c_char_p; lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, U, ctypes.c_int]
rank, path = int(sys.argv[2]), sys.argv[3]
u = U()
if rank == 0:
    assert lib.ncclGetUniqueId(ctypes.byref(u)) == 0
    open(path + ".tmp", "wb").write(bytes(u.b)); __import__("os").rename(path + ".tmp", path)
else:
    while not __import__("os").path.exists(path): time.sleep(0.01)
    ctypes.memmove(u.b, open(path, "rb").read(), 128)
c = ctypes.c_void_p(); start = time.monotonic()
r = lib.ncclCommInitRank(ctypes.byref(c), 2, u, rank)
print(json.dumps({"result": r, "seconds": round(time.monotonic() - start, 2),
                  "message": lib.ncclGetLastError(None).decode()}))
'''


def fake_sysfs(root: Path, device: str, entries) -> None:
    """A sysfs RDMA tree: `entries` are (index, gid text, type)."""
    port = root / device / "ports" / "1"
    (port / "gids").mkdir(parents=True)
    (port / "gid_attrs" / "types").mkdir(parents=True)
    for index, gid, kind in entries:
        (port / "gids" / str(index)).write_text(gid + "\n")
        (port / "gid_attrs" / "types" / str(index)).write_text(kind + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="sircl-ccl-setup-"))
    script = work / "rank.py"
    script.write_text(RANK)
    sysfs = work / "sysfs"
    fake_sysfs(sysfs, "mlx5_fake0", [(0, "fe80:0000:0000:0000:0000:0000:0000:0001", "IB/RoCE v1"),
                                     (2, "0000:0000:0000:0000:0000:ffff:c612:0001", "RoCE v2"),
                                     (3, "0000:0000:0000:0000:0000:ffff:c612:0002", "RoCE v2")])
    emulation = {"LIBSIRCL_TRANSPORT": "emulation", "SIRCL_EMU_FABRIC": f"/sircl-emu-setup-{os.getpid()}"}
    verbs = {"LIBSIRCL_TRANSPORT": "verbs"}
    cases = [
        ("verbs transport without a route map", [verbs, verbs], ["rank 0:", "rank 1:", "needs a route map"]),
        ("a route map naming a device the host lacks",
         [dict(verbs, SIRCL_PEER_ROUTES="1=mlx5_missing0", SIRCL_GID_INDEX="3"),
          dict(verbs, SIRCL_PEER_ROUTES="0=mlx5_missing1", SIRCL_GID_INDEX="3")],
         ["rank 0:", "rank 1:", "mlx5_missing"]),
        ("GID resolution with two RoCE v2 IPv4 entries",
         [dict(verbs, SIRCL_PEER_ROUTES="1=mlx5_fake0", LIBSIRCL_SYSFS_INFINIBAND=str(sysfs)),
          dict(verbs, SIRCL_PEER_ROUTES="0=mlx5_fake0", LIBSIRCL_SYSFS_INFINIBAND=str(sysfs))],
         ["rank 0:", "rank 1:", "has 2 RoCE v2 IPv4 GIDs"]),
        ("SIRCL_ONESHOT_MAX_BYTES differing between ranks",
         [emulation, dict(emulation, SIRCL_ONESHOT_MAX_BYTES="4096")],
         ["rank 1: SIRCL_ONESHOT_MAX_BYTES differs from rank 0"]),
        ("an invalid SIRCL_THREADS on rank 0",
         [dict(emulation, SIRCL_THREADS="100"), emulation], ["rank 0:", "SIRCL_THREADS must be a multiple of 32"]),
        ("a ring schedule without the ring plan",
         [dict(emulation, SIRCL_SCATTER_SCHEDULE="ring", SIRCL_LARGE_SCHEDULE="pieces"),
          dict(emulation, SIRCL_SCATTER_SCHEDULE="ring", SIRCL_LARGE_SCHEDULE="pieces")],
         ["rank 0:", "rank 1:", "a ring schedule needs the ring plan"]),
        ("SIRCL_GATHER_SCHEDULE differing between ranks",
         [emulation, dict(emulation, SIRCL_GATHER_SCHEDULE="chain")],
         ["rank 1: SIRCL_GATHER_SCHEDULE differs from rank 0"]),
    ]
    failed = 0
    for index, (name, environments, expected) in enumerate(cases):
        id_path = work / f"id{index}"
        processes = []
        for rank, extra in enumerate(environments):
            env = dict(os.environ, LIBSIRCL_SETUP_TIMEOUT_MS="60000", CUDA_MODULE_LOADING="EAGER")
            env.update(extra)
            processes.append(subprocess.Popen([args.python, str(script), args.library, str(rank), str(id_path)],
                                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        started = time.monotonic()
        outcomes = []
        for process in processes:
            try:
                out, err = process.communicate(timeout=120)
                outcomes.append(json.loads(out.strip().splitlines()[-1]) if out.strip() else {"error": err[-500:]})
            except subprocess.TimeoutExpired:
                process.kill()
                outcomes.append({"error": "timed out"})
        elapsed = time.monotonic() - started
        ok = all(o.get("result", 0) != 0 and all(text in o.get("message", "") for text in expected)
                 for o in outcomes)
        failed += 0 if ok else 1
        detail = f"{elapsed:.1f} s; rank 0: {outcomes[0].get('message', outcomes[0])[:400]}"
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            print(f"     rank 1: {outcomes[1]}")
    fabric = Path("/dev/shm") / emulation["SIRCL_EMU_FABRIC"].lstrip("/")
    if fabric.exists():
        fabric.unlink()
    print(f"{len(cases)} cases, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
