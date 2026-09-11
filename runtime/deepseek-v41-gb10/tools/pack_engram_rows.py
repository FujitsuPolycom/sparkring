#!/usr/bin/env python3
"""pack_engram_rows.py — build this rank's packed Engram shard(s) for the disk loader
(patches/engram.py, DSV41_ENGRAM_PACKED_DIR).

Per Engram layer, a SPARSE file `engram-l<layer>-packed.bin` addressed by GLOBAL row id
(offset = row * 264) holding the 256 fp8 weight bytes and the 8 ue8m0 scale bytes of a
row adjacent, so a lookup costs one pread instead of two into tensors ~24 GB apart. Only
the rows behind the hash columns this rank owns are written (the rest are holes), and a
manifest `<file>.json` records the covered ranges; the loader refuses a shard whose
manifest does not cover what the rank needs. Use a fresh output directory per
checkpoint: existing manifests do not attest checkpoint identity, so the packer
refuses reuse or replacement of any existing destination. Self-contained: rebuilds the column layout
from config.json (same prime walk as vllm's EngramLayout), reads the safetensors headers.

  python3 pack_engram_rows.py --model-dir /models/DeepSeek-V4.1-Flash --out-dir /cache/engram-packed \\
      --tp 4 --rank 0 --balanced        # or --contiguous (stock head split)
"""

import argparse
import json
import os
import struct
import time
import numpy as np


def _is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start, seen):
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def head_sizes_per_layer(cfg):
    layers = list(cfg["engram_layer_ids"])
    seen = set()
    out = []
    for _ in layers:
        flat = []
        for _ in range(cfg["engram_max_ngram_size"] - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _ in range(cfg["engram_n_heads"]):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                flat.append(cur)
        out.append(flat)
    return layers, out


def tensor_loc(model_dir, name):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))[
        "weight_map"
    ]
    path = os.path.join(model_dir, idx[name])
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    m = hdr[name]
    return path, 8 + n + m["data_offsets"][0], tuple(m["shape"]), m["dtype"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--rank", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--balanced", action="store_true")
    g.add_argument("--contiguous", action="store_true")
    ap.add_argument("--chunk-rows", type=int, default=1 << 20)
    a = ap.parse_args()
    if os.path.lexists(a.out_dir):
        raise FileExistsError(
            "packed output already exists; preserve it and select a fresh "
            "per-checkpoint destination (existing manifests do not attest checkpoint identity)"
        )
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    layers, sizes = head_sizes_per_layer(cfg)
    n_cols = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
    part = -(-n_cols // a.tp)
    if a.balanced:
        assert n_cols % a.tp == 0, "balanced needs n_hash_cols divisible by tp"
        cols = [c for c in range(n_cols) if c % a.tp == a.rank]
    else:
        cols = list(range(a.rank * part, min((a.rank + 1) * part, n_cols)))
    os.makedirs(a.out_dir, exist_ok=False)
    for layer, hs in zip(layers, sizes):
        wpath, woff, wshape, wdt = tensor_loc(
            a.model_dir, f"layers.{layer}.engram.embed.weight"
        )
        spath, soff, sshape, sdt = tensor_loc(
            a.model_dir, f"layers.{layer}.engram.embed.scale"
        )
        assert wdt == "F8_E4M3" and sdt == "F8_E8M0" and wshape[0] == sshape[0], (
            wdt,
            sdt,
            wshape,
            sshape,
        )
        rows, dim, sb = wshape[0], wshape[1], sshape[1]
        rb = dim + sb
        assert sum(hs) <= rows
        cum = [0]
        for x in hs:
            cum.append(cum[-1] + x)
        ranges = [(cum[c], cum[c + 1]) for c in cols]
        out = os.path.join(a.out_dir, f"engram-l{layer}-packed.bin")
        mf = out + ".json"
        print(
            f"layer {layer}: rows {rows} x {rb} B, cols {cols}, {sum(hi - lo for lo, hi in ranges):,} rows to pack",
            flush=True,
        )
        wfd = os.open(wpath, os.O_RDONLY)
        sfd = os.open(spath, os.O_RDONLY)
        tmp = out + ".partial"
        ofd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.ftruncate(ofd, rows * rb)
        t0 = time.time()
        done = 0
        for lo, hi in ranges:
            for s in range(lo, hi, a.chunk_rows):
                n = min(a.chunk_rows, hi - s)
                w = np.frombuffer(
                    os.pread(wfd, n * dim, woff + s * dim), dtype=np.uint8
                ).reshape(n, dim)
                sc = np.frombuffer(
                    os.pread(sfd, n * sb, soff + s * sb), dtype=np.uint8
                ).reshape(n, sb)
                buf = np.empty((n, rb), dtype=np.uint8)
                buf[:, :dim] = w
                buf[:, dim:] = sc
                off = s * rb
                mv = memoryview(buf).cast("B")
                written = 0
                while written < len(mv):
                    written += os.pwrite(ofd, mv[written:], off + written)
                done += n
            print(
                f"  range [{lo}, {hi}) done ({done:,} rows, {time.time() - t0:.0f} s)",
                flush=True,
            )
        os.fsync(ofd)
        os.close(ofd)
        os.close(wfd)
        os.close(sfd)
        os.replace(tmp, out)
        json.dump(
            {
                "layer": layer,
                "rows": rows,
                "row_bytes": rb,
                "dim": dim,
                "sb": sb,
                "tp": a.tp,
                "rank": a.rank,
                "mode": "balanced" if a.balanced else "contiguous",
                "cols": cols,
                "ranges": ranges,
                "source": os.path.basename(wpath),
                "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            open(mf, "w"),
            indent=1,
        )
        print(f"layer {layer}: wrote {out} ({time.time() - t0:.0f} s)", flush=True)
    print("PACK-OK", flush=True)


if __name__ == "__main__":
    main()
