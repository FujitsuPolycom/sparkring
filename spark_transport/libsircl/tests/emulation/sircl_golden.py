#!/usr/bin/env python3
"""The SIRCL Python session's outputs for the library emulation's cases (``library_rank.py``).

Runs SIRCL's own session class and CuTe DSL kernels for every rank of a group as threads of one process
on one GPU over SIRCL's in-process verbs stand-in (``sparkring_sircl.testing.gpu_emulation``, from the
SIRCL reference tree on ``PYTHONPATH``, unmodified), with the schedules fixed to pieces, tiles and scatter
ops unless ``SIRCL_GOLDEN_*_SCHEDULE`` names the chain or ring schedules the library runs, and writes
each rank's output bytes of every eager and in-place case to ``DIR/caseNNN/rank<r>.bin`` (all-reduce)
and ``DIR/gNNN/rank<r>.bin`` (all-gather through ``all_gather_large`` along dimension 0, and
reduce-scatter through the scatter ops for chunks of whole 16-byte packs). The library's
ranks compare their outputs with these files byte for byte. ``--skip-reduce-scatter`` writes no
reduce-scatter outputs, for layouts whose scatter ops SIRCL's emulation harness does not complete; the
library then checks those cases against the rank-order reference only. ``--digests FILE`` also writes the
SHA-256 of every output to a JSON file (``{"caseNNN/rank<r>": hex}``), which ``library_rank.py --golden
FILE`` accepts on hosts that do not hold the bytes, with the SHA-256 of each case's inputs
(``"caseNNN/inputs"``) so that a host can tell different inputs from different outputs; ``--from-dir DIR --digests FILE`` writes it for an
existing output directory without running SIRCL.

Usage (inside WSL, under the GPU lock):
  PYTHONPATH=<sircl reference>/spark_transport/sircl python sircl_golden.py --out DIR --layout path:0-1 --lanes 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
# SIRCL_GOLDEN_LARGE_SCHEDULE, SIRCL_GOLDEN_GATHER_SCHEDULE and SIRCL_GOLDEN_SCATTER_SCHEDULE (pieces when
# unset) set the SIRCL session's schedules, for the library's runs under the same SIRCL_*_SCHEDULE; with a
# ring schedule the library's references take the ring plan as present (LIBSIRCL_RING_WINDOW).
for name in ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE"):
    os.environ[name] = os.environ.get(name.replace("SIRCL_", "SIRCL_GOLDEN_", 1)) or "pieces"
if "ring" in (os.environ[name] for name in ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE",
                                             "SIRCL_SCATTER_SCHEDULE")):
    os.environ.setdefault("LIBSIRCL_RING_WINDOW", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import library_rank  # noqa: E402


def write_digests(directory: Path, target: Path) -> int:
    digests = {f"{p.parent.name}/{p.stem}": hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(directory.glob("*/rank*.bin"))}
    for p in sorted(directory.glob("*/inputs.sha256")):
        digests[f"{p.parent.name}/inputs"] = p.read_text().split()[0]
    target.write_text(json.dumps(digests, indent=0, sort_keys=True) + "\n")
    print(f"wrote {len(digests)} digests of {directory} to {target}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="")
    parser.add_argument("--layout", default="path:0-1")
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--max-piece", type=int, default=4 << 20)
    parser.add_argument("--skip-reduce-scatter", action="store_true")
    parser.add_argument("--digests", default="")
    parser.add_argument("--from-dir", default="")
    args = parser.parse_args(argv)
    if args.from_dir:
        if not args.digests:
            parser.error("--from-dir needs --digests")
        return write_digests(Path(args.from_dir), Path(args.digests))
    if not args.out:
        parser.error("--out is required")

    import torch
    from sparkring_sircl.testing import gpu_emulation as ge
    from sparkring_sircl.testing import native_build

    build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", "/tmp/sircl-ccl-work/sim"))
    group = ge.EmulatedGroup(args.layout, args.lanes, max_size=2 << 20, max_gather_bytes=64 << 10,
                             library=native_build.build_shared_library(build),
                             environment={"SIRCL_LARGE_PIECE_BYTES": str(args.max_piece)})
    out = Path(args.out)
    written = 0
    try:
        types = (torch.bfloat16, torch.float16, torch.float32)
        for rank, session in enumerate(group.sessions):
            with torch.cuda.stream(group.streams[rank]):
                session.prepare(types, padded_gather=True, scatter=True)
        group.load_modules(types)
        started = time.perf_counter()
        for index, (name, dtype_name, count, mode) in enumerate(library_rank.cases(args.max_piece)):
            if mode not in ("eager", "inplace"):
                continue
            dtype = getattr(torch, dtype_name)
            inputs = library_rank.inputs_for(torch, group.world, count, dtype, args.seed_base + index)
            outputs = group.each(lambda rank, session: session.all_reduce_large(inputs[rank].to(session.device)))
            reference = library_rank.allreduce_reference(torch, inputs, order=list(group.sessions[0].chain_order or ()))
            for rank, tensor in enumerate(outputs):
                tensor = tensor.cpu()
                if not library_rank.same_bits(torch, tensor, reference):
                    print(f"FAIL {name}: the SIRCL session's rank {rank} differs from the rank-order reference")
                    return 1
                path = out / f"case{index:03d}" / f"rank{rank}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                written += 1
            (out / f"case{index:03d}" / "inputs.sha256").write_text(library_rank.inputs_digest(torch, inputs) + "\n")
        world = group.world
        for index, (name, kind, dtype_name, count, mode) in enumerate(library_rank.gather_scatter_cases(args.max_piece)):
            print(f"case g{index:03d}: {name}", flush=True)
            if mode not in ("eager", "inplace"):
                continue
            dtype = getattr(torch, dtype_name)
            item = torch.empty((), dtype=dtype).element_size()
            seed = args.seed_base + 500 + index
            if kind == "all_gather":
                inputs = library_rank.inputs_for(torch, world, count, dtype, seed)
                outputs = group.each(lambda rank, session: session.all_gather_large(inputs[rank].to(session.device),
                                                                                    dim=0))
                wanted = [torch.cat(inputs)] * world
            else:
                if args.skip_reduce_scatter:
                    continue
                if (count * item) % 16:
                    continue  # SIRCL's reduce-scatter carries chunks of whole 16-byte packs
                inputs = library_rank.inputs_for(torch, world, world * count, dtype, seed)
                outputs = group.each(lambda rank, session: session.reduce_scatter(inputs[rank].to(session.device)))
                wanted = library_rank.scatter_reference(torch, inputs, list(group.sessions[0].chain_order or ()))
            for rank, tensor in enumerate(outputs):
                tensor = tensor.cpu().reshape(-1)
                if not library_rank.same_bits(torch, tensor, wanted[rank]):
                    print(f"FAIL {name}: the SIRCL session's rank {rank} differs from the reference")
                    return 1
                path = out / f"g{index:03d}" / f"rank{rank}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                written += 1
            (out / f"g{index:03d}" / "inputs.sha256").write_text(library_rank.inputs_digest(torch, inputs) + "\n")
        print(f"wrote {written} outputs of {group.world} ranks in {time.perf_counter() - started:.1f} s to {out}; "
              f"schedules {group.sessions[0].large_schedule}/{group.sessions[0].gather_schedule}/"
              f"{group.sessions[0].scatter_schedule}, ring {group.sessions[0].ring_available}, "
              f"chain order {group.sessions[0].chain_order}")
        if args.digests:
            write_digests(out, Path(args.digests))
    finally:
        group.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
