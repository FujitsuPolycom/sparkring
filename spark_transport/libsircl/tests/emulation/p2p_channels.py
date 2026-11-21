#!/usr/bin/env python3
"""Point-to-point channels on communicators of more than two ranks, in GPU emulation.

Started without --rank, the script is the runner: one process per rank on the one GPU over the emulation
transport, every rank with LIBSIRCL_P2P_CHANNELS=on, so ncclSend and ncclRecv between the ranks of the
communicator run as items of SIRCL's point-to-point channels (src/transport/sircl_p2p_proxy.c and the
point-to-point pack). ``--layout`` places the ranks:

- ``none``: no layout (every lane direct), ``--world`` ranks (4 or 8); every ordered pair has a channel.
- ``ring8``: eight ranks with the settings ``tools/site_routes.py --layout ring:8 --lanes 2`` emits for
  their positions (tests/data/site_routes_ring8_l2.json). Under SIRCL's budget the layout's collective
  session leaves no relay queue room for point-to-point lanes, so only cable neighbors have channels; a
  send to a rank reached through relays is refused with ncclInvalidUsage on both ranks.
- ``ring8-alone``: the same with ``--p2p-reserve none`` (tests/data/site_routes_ring8_l2_p2p_alone.json):
  every pair has a channel and every relayed lane posts within its point-to-point window.

``--case traffic`` (default) runs, checking every received byte:

1. every ordered pair with a channel at once: one group per round in which each rank sends to and
   receives from every channel peer, in a shuffled issue order, messages of 0 bytes to 8 MiB + 16 (up to 17
   items of the 512 KiB slots, more than the 8 slots: credits), some buffers not 16-byte aligned and some
   sizes not whole packs (staged);
2. a subset of pairs while the other ranks idle: pairs (0, 1) (and (4, 5) on eight ranks) exchange in
   groups at a pace of their own, and the lower rank sends three messages outside groups that the other
   receives later (first-in first-out matching); then an all-reduce on the communicator;
3. a pipeline chain at independent paces: stage r receives from r - 1 and forwards to r + 1, calls outside
   groups as torch's isend and irecv issue them;
4. nccl-tests' sendrecv pattern: send to r + 1 and receive from r - 1 in one group, 8 MiB;
5. on ``ring8``, a send to every rank without a channel refused, naming LIBSIRCL_P2P_WINDOWS;
6. the receipt: channel peers, windows of relayed lanes (``ring8-alone``), per-peer messages and bytes as
   sent and received, every item posted and released, messages staged, refused calls counted, healthy;
   then ncclCommDestroy.

``--env NAME=VALUE`` adds a setting on every rank, for example LIBSIRCL_STREAM_ORDERED_ALLOC=off, under which
staged messages use each channel's own staging buffer instead of stream-ordered allocations.

``--case size``: four ranks; rank 0 sends 4096 bytes to rank 1, which receives 8192. Rank 1's kernel records
the size mismatch; every rank's ncclCommGetAsyncError then reports the channels' failure (rank 1 the sizes,
the others rank 1 as its origin) and ncclCommAbort returns. ``--case gone``: four ranks with
LIBSIRCL_FAIL_STOP=1 and a 3 s wait limit; rank 3 ends its process after setup, rank 0 receives from it:
its wait times out, and ranks 0, 1 and 2 end with status 70 (rank 0 by its own timeout, 1 and 2 by the abort
notice). ``--case abort``: four ranks with a 60 s wait limit; rank 1 enqueues a receive from rank 0 that no
send matches and calls ncclCommAbort 0.5 s later: the abort returns within 2 s, the receive's kernel stops
within 5 s, and ranks 0, 2 and 3 report the channels' failure naming rank 1 within 20 s, long before the wait
limit, because the abort poisons the channels and their progress thread posts abort notices. ``--case setup``: three groups of three ranks whose setup must fail on every rank, naming the
reason: LIBSIRCL_P2P_CHANNELS on rank 0 only, SIRCL_P2P_THREADS=1024 on rank 1 (the pack runs at most 512),
and a malformed LIBSIRCL_P2P_WINDOWS on rank 2. ``--case graph``: CUDA graph capture of grouped
ncclSend/ncclRecv across channel peers (aligned multi-slot and unaligned/staged), 10 replays with mutated
inputs plus eager traffic before and after, checking every byte and the receipt's captured_calls.
``--case ep``: GPU-driven dynamic MoE token dispatch and combine (sirclEpWorkspaceBytes, sirclEpDispatch,
sirclEpCombine) across asymmetric eager rounds (BF16/FP16/FP32 and all-local routing) plus CUDA graph
capture and 8 dynamic-routing replays including an all-local toggle, checking bit-exactness against a
PyTorch reference and the receipt's ep counters.

Status: research-only test infrastructure (the emulation gives every pair a shared-memory lane; relays are
modeled by the windows' chunked posting, not by relay queues).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"
LAYOUTS = {"ring8": DATA / "site_routes_ring8_l2.json", "ring8-alone": DATA / "site_routes_ring8_l2_p2p_alone.json",
           "path4": DATA / "site_routes_path0-3_l2.json"}
# The library's cycle plan per layout (the receipt's "cycle_plan"): ring:8 closes over cables on every
# rank's settings, path:0-3's closing edge (position 3 to 0) crosses relays, and without a layout no rank
# has a ring plan.
CYCLE_PLAN = {"ring8": True, "ring8-alone": True, "path4": False, "none": False}
U8, F16, F32, BF16 = 1, 6, 7, 9
SLOT_BYTES = 512 << 10
INVALID_USAGE = 5
SIZES = (0, 1, 15, 4096, 65539, 1000003, 5 << 20, (8 << 20) + 16)


def items(n: int) -> int:
    padded = -(-n // 16) * 16
    return max(1, -(-padded // SLOT_BYTES))


def bind(lib):
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    for name in ("ncclSend", "ncclRecv"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    lib.sirclGetReceipt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    if hasattr(lib, "sirclEpWorkspaceBytes"):
        lib.sirclEpWorkspaceBytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
        lib.sirclEpWorkspaceBytes.restype = ctypes.c_size_t
        lib.sirclEpDispatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.sirclEpDispatch.restype = ctypes.c_int
        lib.sirclEpCombine.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.sirclEpCombine.restype = ctypes.c_int


def channel_peers(layout: str, world: int, rank: int) -> set:
    if layout == "ring8":
        return {(rank + 1) % world, (rank - 1) % world}
    return {p for p in range(world) if p != rank}


def run_rank(args) -> int:
    import numpy as np
    import torch

    import unique_id
    from library_rank import receipt_of

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    lib = ctypes.CDLL(args.library)
    bind(lib)
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, unique_id.UniqueId, ctypes.c_int]
    rank, world = args.rank, args.world
    checks: list = []
    out = {"rank": rank, "checks": checks}

    def report(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def save():
        partial = Path(args.out + ".partial")
        partial.write_text(json.dumps(out))
        os.replace(partial, args.out)

    def last_error(handle=None):
        return lib.ncclGetLastError(handle).decode(errors="replace")

    uid = unique_id.share(lib, rank, world, args.id_file, "")
    comm = ctypes.c_void_p()
    code = lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank)
    if args.case == "setup":
        report("setup fails", code != 0, last_error())
        out["message"] = last_error()
        save()
        return 0
    if code:
        report(f"ncclCommInitRank of {world} ranks", False, last_error())
        save()
        return 1
    stream = torch.cuda.Stream()
    handle = ctypes.c_void_p(stream.cuda_stream)

    def call(send, view, peer):
        function = lib.ncclSend if send else lib.ncclRecv
        return function(ctypes.c_void_p(view.data_ptr()), view.numel(), U8, peer, comm, handle)

    if args.case == "size":
        x = torch.zeros(8192, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        code = call(True, x[:4096], 1) if rank == 0 else call(False, x, 0) if rank == 1 else 0
        report("the mismatched calls enqueue", code == 0, last_error() if code else "")
        status, message, started = ctypes.c_int(0), "", time.monotonic()
        while time.monotonic() - started < 60:
            lib.ncclCommGetAsyncError(comm, ctypes.byref(status))
            if status.value:
                message = last_error(comm)
                break
            time.sleep(0.05)
        want = "different sizes" if rank == 1 else "failure on rank 1"
        report(f"the channels' failure reported within {time.monotonic() - started:.2f} s", status.value != 0 and
               want in message, f"result {status.value}: {message}")
        report("ncclCommAbort", lib.ncclCommAbort(comm) == 0, last_error())
        save()
        return 0
    if args.case == "abort":
        x = torch.zeros(4096, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        if rank == 1:
            code = call(False, x, 0)
            report("a receive that no send matches enqueues", code == 0, last_error() if code else "")
            time.sleep(0.5)
            started = time.monotonic()
            aborted = lib.ncclCommAbort(comm)
            seconds = time.monotonic() - started
            report(f"ncclCommAbort returned within {seconds:.2f} s", aborted == 0 and seconds < 2, last_error())
            started = time.monotonic()
            stream.synchronize()
            seconds = time.monotonic() - started
            report(f"the receive's kernel stopped {seconds:.2f} s after the abort", seconds < 5)
            save()
            return 0
        status, message, started = ctypes.c_int(0), "", time.monotonic()
        while time.monotonic() - started < 20:
            lib.ncclCommGetAsyncError(comm, ctypes.byref(status))
            if status.value:
                message = last_error(comm)
                break
            time.sleep(0.05)
        report(f"rank 1's abort reported within {time.monotonic() - started:.2f} s",
               status.value != 0 and "failure on rank 1" in message, f"result {status.value}: {message}")
        report("ncclCommAbort", lib.ncclCommAbort(comm) == 0, last_error())
        save()
        return 0
    if args.case == "gone":
        save()
        if rank == 3:
            os._exit(0)
        if rank == 0:
            x = torch.zeros(4096, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            call(False, x, 3)
        time.sleep(60)  # fail-stop ends this process
        return 3

    rng = random.Random(1000 + rank)

    def payload(src, dst, tag, n):
        seed = ((src * 64 + dst) * 1000003 + tag) % (1 << 32)
        return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8))

    def placed(data, offset):
        whole = torch.zeros(data.numel() + offset + 16, dtype=torch.uint8, device="cuda")
        view = whole[offset:offset + data.numel()]
        if data.numel():
            view.copy_(data.cuda())
        return whole, view

    def empty(n, offset):
        whole = torch.full((n + offset + 16,), 7, dtype=torch.uint8, device="cuda")
        return whole, whole[offset:offset + n]

    peers = channel_peers(args.layout, world, rank)
    sent = {p: [0, 0, 0] for p in range(world)}       # messages, bytes, items
    received = {p: [0, 0, 0] for p in range(world)}

    def count(table, peer, n):
        table[peer][0] += 1
        table[peer][1] += n
        table[peer][2] += items(n)

    receipt = receipt_of(lib, comm)
    channels = receipt["channels"]
    report("channels on, context open, peers as the layout gives them", channels["on"] and channels["context"] and
           set(channels["peers"]) == peers, json.dumps({k: channels[k] for k in ("on", "context", "peers")}))

    if args.case == "graph":
        def run_eager(tag):
            keep_e, calls_e, expect_e = [], [], []
            for peer in sorted(peers):
                n_out = 4096 if (rank + peer) % 2 == 0 else 1029
                off_out = 0 if (rank + peer) % 2 == 0 else 3
                whole, view = placed(payload(rank, peer, tag, n_out), off_out)
                keep_e.append(whole)
                calls_e.append((True, view, peer))
                n_in = 4096 if (peer + rank) % 2 == 0 else 1029
                off_in = 0 if (peer + rank) % 2 == 0 else 5
                whole, view = empty(n_in, off_in)
                keep_e.append(whole)
                calls_e.append((False, view, peer))
                expect_e.append((view, payload(peer, rank, tag, n_in)))
            torch.cuda.synchronize()
            codes = [lib.ncclGroupStart()] + [call(*c) for c in calls_e] + [lib.ncclGroupEnd()]
            stream.synchronize()
            bad = [i for i, (view, want) in enumerate(expect_e) if not torch.equal(view.cpu(), want)]
            return not any(codes) and not bad, f"codes {codes}, {len(bad)} differ; {last_error() if any(codes) else ''}"

        ok, detail = run_eager(500)
        report("eager grouped exchange before CUDA graph capture", ok, detail)

        # Mix aligned multi-slot ((1 MiB) + 16), aligned small (4096), and unaligned/staged (1029 @ offset 3).
        graph_sizes = (4096, (1 << 20) + 16, 1029)
        keep_g, send_views, recv_views, graph_calls = [], [], [], []
        for peer in sorted(peers):
            idx_out = (rank * 3 + peer * 5) % len(graph_sizes)
            n_out = graph_sizes[idx_out]
            off_out = 3 if idx_out == 2 else 0
            whole, sview = placed(payload(rank, peer, 600, n_out), off_out)
            keep_g.append(whole)
            send_views.append((peer, n_out, sview))
            graph_calls.append((True, sview, peer))

            idx_in = (peer * 3 + rank * 5) % len(graph_sizes)
            n_in = graph_sizes[idx_in]
            off_in = 5 if idx_in == 2 else 0
            whole, rview = empty(n_in, off_in)
            keep_g.append(whole)
            recv_views.append((peer, n_in, rview))
            graph_calls.append((False, rview, peer))

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            codes = [lib.ncclGroupStart()] + [call(*c) for c in graph_calls] + [lib.ncclGroupEnd()]
        stream.synchronize()
        report("grouped P2P channel calls capture into a CUDA graph", not any(codes),
               f"codes {codes}; {last_error() if any(codes) else ''}")

        bad_replays = []
        for rep in range(10):
            for peer, n_out, sview in send_views:
                sview.copy_(payload(rank, peer, 1000 + rep, n_out).cuda())
            for _, _, rview in recv_views:
                rview.fill_(0x5A)
            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            for peer, n_in, rview in recv_views:
                want = payload(peer, rank, 1000 + rep, n_in)
                if not torch.equal(rview.cpu(), want):
                    bad_replays.append(f"replay {rep} from rank {peer}")
        report("10 CUDA graph replays with mutated inputs are bit-exact", not bad_replays, "; ".join(bad_replays[:4]))

        ok, detail = run_eager(700)
        report("eager grouped exchange after CUDA graph replays", ok, detail)

        barrier = torch.full((16,), float(rank + 1), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        bcode = lib.ncclAllReduce(ctypes.c_void_p(barrier.data_ptr()), ctypes.c_void_p(barrier.data_ptr()),
                                  barrier.numel(), F32, 0, comm, handle)
        stream.synchronize()
        report("all-reduce barrier after graph replays", bcode == 0, last_error() if bcode else "")

        receipt = receipt_of(lib, comm)
        captured = receipt["all_reduce"]["captured_calls"]
        report("receipt records captured P2P channel calls and healthy state",
               captured >= len(graph_calls) and receipt["healthy"],
               f"captured_calls={captured}, healthy={receipt['healthy']}, error={receipt.get('error')}")
        report("ncclCommDestroy", lib.ncclCommDestroy(comm) == 0, last_error())
        save()
        return 0 if all(ok for _, ok, _ in checks) else 1

    if args.case == "ep":
        num_experts = world * 8
        local_e = num_experts // world
        top_k = 4
        hidden_dim = 256

        def make_ep_inputs(seed, n_tok, td, all_local=False):
            g = np.random.default_rng(seed)
            if n_tok == 0:
                h = torch.empty((0, hidden_dim), dtype=td, device="cuda")
                ids = torch.empty((0, top_k), dtype=torch.int32, device="cuda")
                wts = torch.empty((0, top_k), dtype=torch.float32, device="cuda")
                return h, ids, wts
            h_np = g.normal(0.0, 1.0, (n_tok, hidden_dim)).astype(np.float32)
            h = torch.from_numpy(h_np).to(dtype=td, device="cuda").contiguous()
            if all_local:
                lo = rank * local_e
                ids_np = np.stack([g.choice(local_e, size=top_k, replace=False) + lo for _ in range(n_tok)]).astype(
                    np.int32
                )
            else:
                ids_np = np.stack([g.choice(num_experts, size=top_k, replace=False) for _ in range(n_tok)]).astype(
                    np.int32
                )
            wts_np = g.uniform(0.1, 1.0, (n_tok, top_k)).astype(np.float32)
            wts_np /= wts_np.sum(axis=1, keepdims=True)
            return h, torch.from_numpy(ids_np).cuda().contiguous(), torch.from_numpy(wts_np).cuda().contiguous()

        def ref_combine(h_in, ids_in, wts_in, td):
            n_tok = h_in.shape[0]
            if n_tok == 0:
                return torch.empty((0, hidden_dim), dtype=td, device="cuda")
            acc = torch.zeros((n_tok, hidden_dim), dtype=torch.float32, device="cuda")
            h_f = h_in.float()
            for dst in range(world):
                lo, hi = dst * local_e, (dst + 1) * local_e
                mask = (ids_in >= lo) & (ids_in < hi)
                active = mask.any(dim=1)
                if active.any():
                    scale = (wts_in * mask.to(torch.float32)).sum(dim=1, keepdim=True) * (1.0 + 0.25 * dst)
                    contrib = (h_f * scale).to(td).float()
                    acc += torch.where(active.unsqueeze(1), contrib, torch.zeros_like(contrib))
            return acc.to(td)

        def run_expert(h_out, ids_out, wts_out, td):
            lo, hi = rank * local_e, (rank + 1) * local_e
            mask = (ids_out >= lo) & (ids_out < hi)
            scale = (wts_out * mask.to(torch.float32)).sum(dim=1, keepdim=True) * (1.0 + 0.25 * rank)
            return (h_out.float() * scale).to(td).contiguous()

        # 1. Asymmetric eager rounds across BF16, FP16, FP32 + all-local round.
        eager_specs = [
            (BF16, torch.bfloat16, "bf16-asym", [0, 3, 7, 1], False),
            (F16, torch.float16, "fp16-asym", [5, 0, 2, 6], False),
            (F32, torch.float32, "fp32-asym", [4, 2, 1, 3], False),
            (BF16, torch.bfloat16, "bf16-all-local", [4, 4, 4, 4], True),
        ]
        for idx, (nccl_dt, td, label, counts, all_local) in enumerate(eager_specs):
            n_tok = counts[rank % len(counts)]
            max_tok = max(counts)
            hidden_bytes = hidden_dim * torch.empty((), dtype=td).element_size()
            ws_bytes = int(lib.sirclEpWorkspaceBytes(world, max_tok, hidden_bytes))
            ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
            h_in, ids_in, wts_in = make_ep_inputs(2000 + idx * 100 + rank, n_tok, td, all_local=all_local)
            total_rows = world * max_tok
            h_out = torch.empty((total_rows, hidden_dim), dtype=td, device="cuda")
            ids_out = torch.empty((total_rows, top_k), dtype=torch.int32, device="cuda")
            wts_out = torch.empty((total_rows, top_k), dtype=torch.float32, device="cuda")
            recv_counts = torch.full((world,), -1, dtype=torch.int32, device="cuda")
            comb_out = torch.empty((n_tok, hidden_dim), dtype=td, device="cuda")
            torch.cuda.synchronize()
            c1 = lib.sirclEpDispatch(
                ctypes.c_void_p(h_in.data_ptr() if n_tok else 0),
                ctypes.c_void_p(ids_in.data_ptr() if n_tok else 0),
                ctypes.c_void_p(wts_in.data_ptr() if n_tok else 0),
                ctypes.c_void_p(h_out.data_ptr()),
                ctypes.c_void_p(ids_out.data_ptr()),
                ctypes.c_void_p(wts_out.data_ptr()),
                ctypes.c_void_p(recv_counts.data_ptr()),
                ctypes.c_void_p(ws.data_ptr()),
                ws_bytes, n_tok, max_tok, hidden_bytes, top_k, num_experts, comm, handle,
            )
            with torch.cuda.stream(stream):
                expert_out = run_expert(h_out, ids_out, wts_out, td)
            c2 = lib.sirclEpCombine(
                ctypes.c_void_p(expert_out.data_ptr()),
                ctypes.c_void_p(comb_out.data_ptr() if n_tok else 0),
                ctypes.c_void_p(ws.data_ptr()),
                ws_bytes, n_tok, max_tok, hidden_dim, nccl_dt, comm, handle,
            )
            stream.synchronize()
            want = ref_combine(h_in, ids_in, wts_in, td)
            rc_list = recv_counts.cpu().tolist()
            local_ok = (not all_local) or all(rc_list[s] == (n_tok if s == rank else 0) for s in range(world))
            exact = (c1 == 0 and c2 == 0 and local_ok and torch.equal(comb_out, want))
            report(f"SIRCL-EP eager {label}", exact,
                   f"codes=({c1},{c2}) recv_counts={rc_list}; {last_error() if (c1 or c2) else ''}")

        # 2. CUDA graph capture of dispatch -> expert -> combine, 8 dynamic-routing replays including all-local toggle.
        g_tok = 8
        g_td, g_dt = torch.bfloat16, BF16
        g_hbytes = hidden_dim * 2
        g_ws_bytes = int(lib.sirclEpWorkspaceBytes(world, g_tok, g_hbytes))
        g_ws = torch.empty(g_ws_bytes, dtype=torch.uint8, device="cuda")
        g_hin = torch.zeros((g_tok, hidden_dim), dtype=g_td, device="cuda")
        g_ids = torch.zeros((g_tok, top_k), dtype=torch.int32, device="cuda")
        g_wts = torch.full((g_tok, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda")
        g_hout = torch.empty((world * g_tok, hidden_dim), dtype=g_td, device="cuda")
        g_idsout = torch.empty((world * g_tok, top_k), dtype=torch.int32, device="cuda")
        g_wtsout = torch.empty((world * g_tok, top_k), dtype=torch.float32, device="cuda")
        g_rc = torch.empty((world,), dtype=torch.int32, device="cuda")
        g_comb = torch.empty((g_tok, hidden_dim), dtype=g_td, device="cuda")

        torch.cuda.synchronize()
        ep_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ep_graph, stream=stream):
            gc1 = lib.sirclEpDispatch(
                ctypes.c_void_p(g_hin.data_ptr()), ctypes.c_void_p(g_ids.data_ptr()),
                ctypes.c_void_p(g_wts.data_ptr()), ctypes.c_void_p(g_hout.data_ptr()),
                ctypes.c_void_p(g_idsout.data_ptr()), ctypes.c_void_p(g_wtsout.data_ptr()),
                ctypes.c_void_p(g_rc.data_ptr()), ctypes.c_void_p(g_ws.data_ptr()),
                g_ws_bytes, g_tok, g_tok, g_hbytes, top_k, num_experts, comm, handle,
            )
            g_exp = run_expert(g_hout, g_idsout, g_wtsout, g_td)
            gc2 = lib.sirclEpCombine(
                ctypes.c_void_p(g_exp.data_ptr()), ctypes.c_void_p(g_comb.data_ptr()),
                ctypes.c_void_p(g_ws.data_ptr()), g_ws_bytes, g_tok, g_tok, hidden_dim, g_dt, comm, handle,
            )
        stream.synchronize()
        report("SIRCL-EP dispatch + expert + combine captures into a CUDA graph", gc1 == 0 and gc2 == 0,
               f"codes=({gc1},{gc2}); {last_error() if (gc1 or gc2) else ''}")

        bad_ep = []
        for rep in range(8):
            all_local = rep in (2, 5)
            h_s, ids_s, wts_s = make_ep_inputs(4000 + rep * 100 + rank, g_tok, g_td, all_local=all_local)
            g_hin.copy_(h_s)
            g_ids.copy_(ids_s)
            g_wts.copy_(wts_s)
            g_comb.zero_()
            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                ep_graph.replay()
            stream.synchronize()
            want = ref_combine(g_hin, g_ids, g_wts, g_td)
            rc_list = g_rc.cpu().tolist()
            if all_local and any(rc_list[s] != (g_tok if s == rank else 0) for s in range(world)):
                bad_ep.append(f"replay {rep} all-local recv_counts={rc_list}")
            elif not torch.equal(g_comb, want):
                bad_ep.append(f"replay {rep} mismatch")
        report("8 SIRCL-EP CUDA graph replays with dynamic routing (including all-local toggle) are bit-exact",
               not bad_ep, "; ".join(bad_ep[:4]))

        barrier = torch.full((16,), float(rank + 1), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        bcode = lib.ncclAllReduce(ctypes.c_void_p(barrier.data_ptr()), ctypes.c_void_p(barrier.data_ptr()),
                                  barrier.numel(), F32, 0, comm, handle)
        stream.synchronize()
        report("all-reduce barrier after SIRCL-EP", bcode == 0, last_error() if bcode else "")

        receipt = receipt_of(lib, comm)
        ep_rec = receipt["channels"]["ep"]
        captured = receipt["all_reduce"]["captured_calls"]
        report("receipt records SIRCL-EP dispatches, combines, captured calls and healthy state",
               ep_rec["dispatches"] >= 5 and ep_rec["combines"] >= 5 and captured >= 2 and receipt["healthy"],
               json.dumps({"ep": ep_rec, "captured_calls": captured, "healthy": receipt["healthy"]}))
        report("ncclCommDestroy", lib.ncclCommDestroy(comm) == 0, last_error())
        save()
        return 0 if all(ok for _, ok, _ in checks) else 1

    # 1. Every ordered pair with a channel at once.
    for round_ in range(args.rounds):
        def size(src, dst):
            return SIZES[(src * 3 + dst * 5 + round_) % len(SIZES)]

        def offset(src, dst, receiving):
            return (src + dst + round_ + (2 if receiving else 0)) % 4 * 3

        keep, calls, expect = [], [], []
        for peer in sorted(peers):
            data = payload(rank, peer, round_, size(rank, peer))
            whole, view = placed(data, offset(rank, peer, False))
            keep.append(whole)
            calls.append((True, view, peer))
            count(sent, peer, view.numel())
            n = size(peer, rank)
            whole, view = empty(n, offset(peer, rank, True))
            keep.append(whole)
            calls.append((False, view, peer))
            expect.append((view, payload(peer, rank, round_, n)))
            count(received, peer, n)
        rng.shuffle(calls)
        torch.cuda.synchronize()
        codes = [lib.ncclGroupStart()] + [call(*c) for c in calls] + [lib.ncclGroupEnd()]
        stream.synchronize()
        bad = [i for i, (view, want) in enumerate(expect) if not torch.equal(view.cpu(), want)]
        report(f"every ordered pair at once, round {round_}: {len(calls)} calls in a shuffled order", not any(codes)
               and not bad, f"codes {codes}, {len(bad)} receives differ; {last_error() if any(codes) else ''}")

    # 2. A subset of pairs while the other ranks idle; then an all-reduce on the communicator.
    pairs = [(0, 1)] + ([(4, 5)] if world == 8 else [])
    mine = next((pair for pair in pairs if rank in pair), None)
    if mine is None:
        time.sleep(1.0)
    else:
        peer = mine[1] if rank == mine[0] else mine[0]
        pace = random.Random(77 + min(mine))
        bad = []
        for step in range(args.rounds * 2):
            time.sleep(pace.uniform(0, 0.05))
            n_out, n_in = (3 << 20) + 16 * step + 5, (3 << 20) + 16 * step + 5
            data = payload(rank, peer, 100 + step, n_out)
            _, out_view = placed(data, 0)
            _, in_view = empty(n_in, 0)
            torch.cuda.synchronize()
            codes = [lib.ncclGroupStart(), call(True, out_view, peer), call(False, in_view, peer), lib.ncclGroupEnd()]
            stream.synchronize()
            count(sent, peer, n_out)
            count(received, peer, n_in)
            if any(codes) or not torch.equal(in_view.cpu(), payload(peer, rank, 100 + step, n_in)):
                bad.append(f"step {step}: codes {codes}")
        report(f"pair {mine} exchanging while the other ranks idle", not bad, "; ".join(bad[:3]))
        one_way = [100, 200000, 3 << 20]
        if rank == mine[0]:
            keep = []
            for k, n in enumerate(one_way):
                _, view = placed(payload(rank, peer, 200 + k, n), 0)
                keep.append(view)
                torch.cuda.synchronize()
                code = call(True, view, peer)
                count(sent, peer, n)
                if code:
                    bad.append(f"one-way send {k}: {code}")
            stream.synchronize()
        else:
            time.sleep(0.3)
            views = [empty(n, 1)[1] for n in one_way]
            torch.cuda.synchronize()
            codes = [call(False, view, peer) for view in views]
            stream.synchronize()
            for k, n in enumerate(one_way):
                count(received, peer, n)
            ok = not any(codes) and all(torch.equal(view.cpu(), payload(peer, rank, 200 + k, n))
                                        for k, (view, n) in enumerate(zip(views, one_way)))
            if not ok:
                bad.append(f"one-way receives: codes {codes}")
        report(f"pair {mine}: three sends outside groups matched in issue order", not bad, "; ".join(bad[:3]))
    x = torch.full((4096,), float(rank + 1), dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    code = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), x.numel(), F32, 0, comm,
                             handle)
    stream.synchronize()
    report("an all-reduce after the subset's channel traffic", code == 0 and bool((y == world * (world + 1) / 2).all()),
           last_error() if code else "")

    # 3. A pipeline chain at independent paces (calls outside groups).
    pace = random.Random(500 + rank)
    bad = []
    for mb in range(args.rounds * 2):
        time.sleep(pace.uniform(0, 0.04))
        n = (1 << 20) + 4096 * mb + (13 if mb % 2 else 0)
        buffer = empty(n, 0)[1] if rank > 0 else placed(payload(0, world, 300 + mb, n), 0)[1]
        torch.cuda.synchronize()
        codes = []
        if rank > 0:
            codes.append(call(False, buffer, rank - 1))
            count(received, rank - 1, n)
        if rank < world - 1:
            codes.append(call(True, buffer, rank + 1))
            count(sent, rank + 1, n)
        stream.synchronize()
        if any(codes) or not torch.equal(buffer.cpu(), payload(0, world, 300 + mb, n)):
            bad.append(f"microbatch {mb}: codes {codes}")
    report(f"pipeline chain of {world} stages, {args.rounds * 2} microbatches forwarded", not bad, "; ".join(bad[:3]))

    # 4. nccl-tests' sendrecv pattern.
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    bad = []
    for it in range(args.rounds):
        n = 8 << 20
        _, out_view = placed(payload(rank, nxt, 400 + it, n), 0)
        _, in_view = empty(n, 0)
        torch.cuda.synchronize()
        codes = [lib.ncclGroupStart(), call(True, out_view, nxt), call(False, in_view, prv), lib.ncclGroupEnd()]
        stream.synchronize()
        count(sent, nxt, n)
        count(received, prv, n)
        if any(codes) or not torch.equal(in_view.cpu(), payload(prv, rank, 400 + it, n)):
            bad.append(f"iteration {it}: codes {codes}")
    report("sendrecv ring, 8 MiB", not bad, "; ".join(bad[:3]))

    # 5. Refusals toward ranks without a channel.
    refused = []
    if args.layout == "ring8":
        x = torch.ones(64, dtype=torch.uint8, device="cuda")
        for peer in range(world):
            if peer == rank or peer in peers:
                continue
            code = call(True, x, peer)
            refused.append((peer, code, last_error()))
        report(f"sends to the {len(refused)} ranks without a channel refused, naming LIBSIRCL_P2P_WINDOWS",
               refused and all(code == INVALID_USAGE and "LIBSIRCL_P2P_WINDOWS" in text for _, code, text in refused),
               refused[0][2] if refused else "")

    # 6. The receipt, once every rank's sends were received (an all-reduce as a barrier).
    code = lib.ncclAllReduce(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(y.data_ptr()), 16, F32, 0, comm, handle)
    stream.synchronize()
    report("an all-reduce as the barrier before the receipt", code == 0, last_error() if code else "")
    want_items = sum(v[2] for v in sent.values())
    started = time.monotonic()
    while True:
        receipt = receipt_of(lib, comm)
        channels = receipt["channels"]
        if channels["native"]["items_posted"] >= want_items or time.monotonic() - started > 10:
            break
        time.sleep(0.05)

    def flows(table):
        return {str(p): {"messages": v[0], "bytes": v[1], "items": v[2]} for p, v in table.items() if v[0]}

    report("receipt: per-peer messages, bytes and items sent and received as issued",
           channels["sent"] == flows(sent) and channels["received"] == flows(received),
           json.dumps({"sent": channels["sent"], "received": channels["received"]})[:600])
    report("receipt: every sent item posted and released", channels["native"]["items_posted"] == want_items and
           channels["native"]["items_released"] == sum(v[2] for v in received.values()),
           json.dumps(channels["native"]))
    report("receipt: messages not 16-byte aligned or not whole packs went through staging", channels["staged"] > 0,
           f"{channels['staged']} staged, stream-ordered allocation {receipt['links']['graph_staging']}")
    if args.layout == "ring8-alone":
        relayed = {int(p) for p in channels["windows"]}
        report("receipt: every relayed peer's lanes keep a point-to-point window",
               relayed == peers - {(rank + 1) % world, (rank - 1) % world},
               json.dumps(channels["windows"]))
    report("receipt: the refused calls counted", channels["refused"] == len(refused), f"refused {channels['refused']}")
    report("receipt: healthy", receipt["healthy"], json.dumps(receipt.get("error")))
    report(f"receipt: the cycle plan {'on' if CYCLE_PLAN[args.layout] else 'off'} for layout {args.layout}",
           receipt.get("cycle_plan") is CYCLE_PLAN[args.layout], f"cycle_plan {receipt.get('cycle_plan')}")
    report("ncclCommDestroy", lib.ncclCommDestroy(comm) == 0, last_error())
    save()
    return 0 if all(ok for _, ok, _ in checks) else 1


def run_group(args, extra_env=None, label="") -> int:
    layout = json.loads(LAYOUTS[args.layout].read_text()) if args.layout in LAYOUTS else None
    world = len(layout["ranks"]) if layout else args.world
    work = Path(args.work) / label if args.work else Path(tempfile.mkdtemp(prefix=f"sircl-p2p-{args.case}-"))
    work.mkdir(parents=True, exist_ok=True)
    base = dict(os.environ)
    base.update({
        "LIBSIRCL_TRANSPORT": "emulation", "SIRCL_EMU_FABRIC": f"/sircl-emu-p2p-{os.getpid()}-{label or args.case}",
        "LIBSIRCL_EMU_LANES": str(layout["lanes"] if layout else args.lanes), "CUDA_MODULE_LOADING": "EAGER",
        "CUDA_DEVICE_MAX_CONNECTIONS": base.get("CUDA_DEVICE_MAX_CONNECTIONS", "32"),
        "LIBSIRCL_RECEIPT": str(work / "receipt"), "LIBSIRCL_P2P_CHANNELS": "on",
    })
    if args.case == "gone":
        base.update({"SIRCL_STARTUP_WAIT_S": "3", "LIBSIRCL_FAIL_STOP": "1"})
    if args.case == "abort":
        base.update({"SIRCL_STARTUP_WAIT_S": "60"})
    base.update(item.split("=", 1) for item in args.env)
    started = time.perf_counter()
    processes = []
    for rank in range(world):
        env = dict(base)
        if layout:
            env.update({k: v for k, v in layout["ranks"][rank]["env"].items() if k != "SIRCL_PEER_ROUTES"})
        else:
            env["LIBSIRCL_POSITION"] = str(rank)
        env.update((extra_env or {}).get(rank, {}))
        command = [args.python, str(Path(__file__).resolve()), "--rank", str(rank), "--world", str(world),
                   "--library", args.library, "--layout", args.layout, "--case", args.case, "--rounds",
                   str(args.rounds), "--id-file", str(work / "unique-id"), "--out", str(work / f"rank{rank}.json")]
        log = open(work / f"rank{rank}.log", "w")
        processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    deadline = time.monotonic() + args.timeout
    codes = []
    for process, log in processes:
        try:
            codes.append(process.wait(timeout=max(1.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            process.kill()
            codes.append(process.wait())
        log.close()
    seconds = time.perf_counter() - started
    for process, _ in processes:
        for segment in Path("/dev/shm").glob(f"sircl-emu-seg-{process.pid}-*"):
            segment.unlink(missing_ok=True)
    fabric = Path("/dev/shm") / base["SIRCL_EMU_FABRIC"].lstrip("/")
    if fabric.exists():
        fabric.unlink()
    total = failed = 0
    if args.case == "gone":
        for rank, code in enumerate(codes):
            want = 0 if rank == 3 else 70
            text = (work / f"rank{rank}.log").read_text(errors="replace")
            named = rank == 3 or "point-to-point channels" in text
            total += 1
            failed += code != want or not named
            print(f"{'PASS' if code == want and named else 'FAIL'} rank {rank} exit {code} (expected {want})"
                  f"{'' if named else ', fail-stop line does not name the channels'}")
        print(f"channels whose peer is gone: {world} ranks, {failed} failed, {seconds:.1f} s; work {work}")
        return 1 if failed else 0
    messages = []
    for rank in range(world):
        path = work / f"rank{rank}.json"
        if not path.exists():
            print(f"FAIL rank {rank}: no result (exit {codes[rank]}); {work / f'rank{rank}.log'}")
            failed += 1
            continue
        result = json.loads(path.read_text())
        messages.append(result.get("message", ""))
        for name, ok, detail in result["checks"]:
            total += 1
            failed += not ok
            if not ok or rank == 0:
                print(f"{'PASS' if ok else 'FAIL'} rank {rank} {name}{': ' + detail if detail else ''}")
        if codes[rank]:
            print(f"FAIL rank {rank} exit {codes[rank]}")
            failed += 1
    if args.case == "setup":
        return failed, messages
    print(f"point-to-point channels, case {args.case}, layout {args.layout}, {world} ranks: {total} checks, "
          f"{failed} failed, {seconds:.1f} s; work {work}")
    return 1 if failed else 0


def run_setup(args) -> int:
    args.world, args.layout = 3, "none"
    cases = [("LIBSIRCL_P2P_CHANNELS on rank 0 only", {1: {"LIBSIRCL_P2P_CHANNELS": "off"},
                                                       2: {"LIBSIRCL_P2P_CHANNELS": "off"}},
              "LIBSIRCL_P2P_CHANNELS differs from rank 0"),
             ("SIRCL_P2P_THREADS=1024 on rank 1", {1: {"SIRCL_P2P_THREADS": "1024"}}, "SIRCL_P2P_THREADS"),
             ("a malformed LIBSIRCL_P2P_WINDOWS on rank 2", {2: {"LIBSIRCL_P2P_WINDOWS": "1=100"}},
              "LIBSIRCL_P2P_WINDOWS")]
    failures = 0
    for index, (name, extra, want) in enumerate(cases):
        failed, messages = run_group(args, extra, label=f"setup{index}")
        ok = not failed and len(messages) == 3 and all(want in m for m in messages)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} setup with {name}: every rank fails naming {want!r}"
              f"{'' if ok else ': ' + ' | '.join(messages)}")
    print(f"point-to-point channel setup failures: {len(cases)} cases, {failures} failed")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--layout", choices=("none", "ring8", "ring8-alone", "path4"), default="none")
    parser.add_argument("--case", choices=("traffic", "size", "gone", "abort", "setup", "graph", "ep"),
                        default="traffic")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                        help="a setting for every rank (repeatable)")
    parser.add_argument("--work", default="")
    parser.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--id-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.rank >= 0:
        return run_rank(args)
    if args.case == "setup":
        return run_setup(args)
    if args.case in ("size", "gone"):
        args.world, args.layout = 4, "none"
    return run_group(args)


if __name__ == "__main__":
    sys.exit(main())
