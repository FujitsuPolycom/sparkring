"""Measures NCCL host buffers: per vLLM process, anonymous /dev/zero (deleted) mappings by size, and total RSS.

NCCL keeps each connection's host-side protocol buffers (Simple + LL128 + LL = 9,633,792 bytes)
in such mappings when GPUDirect RDMA is unavailable.
"""
import collections
import os

NCCL_CONN_BYTES = 9633792
for pid in sorted(os.listdir("/proc"), key=lambda p: int(p) if p.isdigit() else 0):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")[0].decode(errors="replace")
    except OSError:
        continue
    if "VLLM::" not in cmd and "EngineCore" not in cmd:
        continue
    sizes = collections.Counter()
    try:
        for line in open(f"/proc/{pid}/maps"):
            if "/dev/zero (deleted)" in line:
                start, end = line.split()[0].split("-")
                sizes[int(end, 16) - int(start, 16)] += 1
        rss = 0
        for line in open(f"/proc/{pid}/smaps_rollup"):
            if line.startswith("Rss:"):
                rss = int(line.split()[1])
    except OSError:
        continue
    nccl = sizes.get(NCCL_CONN_BYTES, 0)
    other = {k: v for k, v in sizes.items() if k != NCCL_CONN_BYTES}
    print(f"pid {pid} {cmd[:40]:40s} rss {rss / 1048576:6.2f} GiB  nccl_conn_buffers {nccl:4d} "
          f"({nccl * NCCL_CONN_BYTES / 2**30:.2f} GiB)  other_devzero {sorted(other.items())[:8]}")
