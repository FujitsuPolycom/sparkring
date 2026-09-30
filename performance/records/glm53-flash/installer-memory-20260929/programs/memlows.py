"""Measures the lowest MemAvailable per Spark inside named time windows of collect.sh output.

Input: a file of "node epoch kB" lines (collect.sh output).
Windows: name=start_epoch:end_epoch arguments.
Prints, per window and node, the lowest MemAvailable (GiB), when it happened, and the sample count.
"""
import collections
import sys
import time

path, *window_args = sys.argv[1:]
samples = collections.defaultdict(list)
for line in open(path):
    parts = line.split()
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        samples[parts[0]].append((int(parts[1]), int(parts[2])))
windows = []
for arg in window_args:
    name, _, span = arg.partition("=")
    start, _, end = span.partition(":")
    windows.append((name, int(start), int(end)))
nodes = sorted(samples)
print(f"{'window':10s} " + " ".join(f"{n:>22s}" for n in nodes))
for name, start, end in windows:
    cells = []
    for node in nodes:
        inside = [(kb, t) for t, kb in samples[node] if start <= t <= end]
        if not inside:
            cells.append(f"{'-':>22s}")
            continue
        kb, t = min(inside)
        cells.append(f"{kb / 1048576:6.2f} GiB @{time.strftime('%H:%M:%S', time.localtime(t))} n={len(inside):4d}")
    print(f"{name:10s} " + " ".join(cells))
