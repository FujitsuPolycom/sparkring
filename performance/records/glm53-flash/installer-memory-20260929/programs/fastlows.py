"""Measures the lowest MemAvailable per Spark in fast-guard logs ("node epoch.fraction kB" lines).

Usage: fastlows.py LOG [LOG ...]
"""
import collections
import sys

for path in sys.argv[1:]:
    low = collections.defaultdict(lambda: (float("inf"), 0.0))
    count = collections.Counter()
    start = None
    for line in open(path):
        parts = line.split()
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        gib = int(parts[2]) / 1048576
        start = float(parts[1]) if start is None else min(start, float(parts[1]))
        count[parts[0]] += 1
        if gib < low[parts[0]][0]:
            low[parts[0]] = (gib, float(parts[1]))
    first = start or 0.0
    print(f"{path.split('/')[-1]:28s} " + "  ".join(
        f"{n} {v:5.2f} at +{t - first:5.1f} s" for n, (v, t) in sorted(low.items())))
