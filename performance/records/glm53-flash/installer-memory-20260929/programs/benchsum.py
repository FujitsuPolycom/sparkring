"""Measures throughput from llm_decode_bench JSON results: prefill tok/s by context, decode aggregate tok/s and
steps/s by (concurrency, context). Several files of the same configuration are averaged.

Usage: benchsum.py LABEL=file1[,file2,...] [LABEL=...]
"""
import collections
import json
import statistics
import sys


def load(paths):
    prefill = collections.defaultdict(list)
    decode = collections.defaultdict(list)
    steps = collections.defaultdict(list)
    for path in paths:
        data = json.load(open(path))
        for ctx, entry in (data.get("prefill") or {}).items():
            value = None
            if isinstance(entry, dict):
                for key in ("tok_per_sec", "client_tok_per_sec"):
                    if isinstance(entry.get(key), (int, float)):
                        value = entry[key]
                        break
            if value:
                prefill[int(ctx)].append(value)
        for row in data.get("results") or []:
            if row.get("aggregate_tps"):
                key = (row["concurrency"], row["context_tokens"])
                decode[key].append(row["aggregate_tps"])
                if row.get("server_steps_per_s"):
                    steps[key].append(row["server_steps_per_s"])
    return prefill, decode, steps


configs = {}
for arg in sys.argv[1:]:
    label, _, files = arg.partition("=")
    configs[label] = load(files.split(","))
labels = list(configs)
ctxs = sorted({c for p, _, _ in configs.values() for c in p})
print("Prefill tok/s (mean of runs)")
print(f"{'ctx':>8s} " + " ".join(f"{name:>14s}" for name in labels))
for ctx in ctxs:
    cells = []
    for label in labels:
        vals = configs[label][0].get(ctx)
        cells.append(f"{statistics.mean(vals):9.0f} (n={len(vals)})" if vals else f"{'-':>14s}")
    print(f"{ctx:8d} " + " ".join(cells))
keys = sorted({k for _, d, _ in configs.values() for k in d})
print("\nDecode aggregate tok/s | engine steps/s (mean of runs)")
print(f"{'conc,ctx':>12s} " + " ".join(f"{name:>22s}" for name in labels))
for key in keys:
    cells = []
    for label in labels:
        vals = configs[label][1].get(key)
        st = configs[label][2].get(key)
        if vals:
            cells.append(f"{statistics.mean(vals):7.1f} | {statistics.mean(st) if st else 0:5.1f} (n={len(vals)})")
        else:
            cells.append(f"{'-':>22s}")
    print(f"{str(key):>12s} " + " ".join(cells))
