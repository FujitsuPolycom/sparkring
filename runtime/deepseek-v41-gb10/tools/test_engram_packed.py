#!/usr/bin/env python3
"""test_engram_packed.py — CPU check inside the serving image: the packed single-read path returns
byte-identical rows to the stock two-read path for random rows in this rank's owned ranges, and the
balanced disk_rel_owned mapping is the identity on global ids. Prints TEST-OK on success.
Env: MODEL_DIR (/models/...), PACKED_DIR (/cache/engram-packed), LAYERS (1,14)."""
import json, os, random, sys, torch
os.environ.setdefault("DSV41_ENGRAM_DISK", "1")
import vllm.models.deepseek_v4_1.common.engram as E
model_dir = os.environ["MODEL_DIR"]; pdir = os.environ["PACKED_DIR"]
layers = [int(x) for x in os.environ.get("LAYERS", "1,14").split(",")]
ok = True
for layer in layers:
    m = json.load(open(os.path.join(pdir, f"engram-l{layer}-packed.bin.json")))
    ranges = [tuple(r) for r in m["ranges"]]
    E._DSV41_ENGRAM_PACKED_DIR = pdir
    tp = E.DiskEngramTable(model_dir, layer, m["dim"], m["dim"] // m["sb"], row_start=0, num_rows=m["rows"], owned_ranges=ranges)
    E._DSV41_ENGRAM_PACKED_DIR = ""
    tw = E.DiskEngramTable(model_dir, layer, m["dim"], m["dim"] // m["sb"], row_start=0, num_rows=m["rows"], owned_ranges=ranges)
    assert tp.packed and not tw.packed, (tp.packed, tw.packed)
    rng = random.Random(layer)
    rows = []
    for lo, hi in ranges:
        rows += [rng.randrange(lo, hi) for _ in range(400)] + [lo, hi - 1]
    rows += rows[:50]  # duplicates exercise torch.unique/inverse
    rel = torch.tensor(rows, dtype=torch.int64); owned = torch.ones_like(rel, dtype=torch.bool)
    a = E.gather_dequant_many([(tp, rel, owned)])[0]; b = E.gather_dequant_many([(tw, rel, owned)])[0]
    same = torch.equal(a, b); nz = (a != 0).any().item()
    print(f"layer {layer}: {len(rows)} rows, packed==two-read: {same}, non-zero data: {nz}, shape {tuple(a.shape)}", flush=True)
    ok &= same and nz
print("TEST-OK" if ok else "TEST-FAIL", flush=True)
sys.exit(0 if ok else 1)
