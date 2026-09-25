"""Summarize a torch profiler trace (Chrome JSON, optionally gzipped) of a decode window.

Usage: trace_breakdown.py TRACE [TOKENS]
Prints GPU kernel time by category and by kernel name, the GPU busy fraction of
the traced window, and per-token figures when TOKENS generated tokens are given.
"""
import collections
import gzip
import json
import re
import sys

path = sys.argv[1]
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 0
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as handle:
    trace = json.load(handle)
events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not events:
    raise SystemExit("no GPU kernel events in trace")
CATEGORIES = [
    ("collective", r"all_?reduce|allgather|all_gather|reduce_scatter|nccl|roce|oneshot|twoshot|rocenante|ncclDevKernel|cross_device"),
    ("moe", r"moe|expert|grouped|fused_experts|topk_?softmax|router|b12x.*(dispatch|combine)"),
    ("attention", r"attn|attention|fmha|flash|qsa|gdn|delta|linear_attn|chunk_|conv1d|causal_conv|mamba|kv_cache|paged|indexer|rotary|rope"),
    ("sampling", r"sampl|argmax|softmax|topk|top_k|top_p|rejection|gumbel|multinomial|probs|logits"),
    ("gemm", r"gemm|matmul|mma|cutlass|sm100|sm120|sm121|cublas|nvjet|linear|blockscaled|dense|mxfp8|nvfp4"),
    ("norm/elementwise", r"norm|rms|silu|gelu|act|elementwise|vectorized|add|mul|copy|cat|index|gather|scatter|embedding|fill|reduce"),
]
by_category = collections.Counter()
by_name = collections.Counter()
count_by_name = collections.Counter()
for event in events:
    name = event.get("name", "")
    duration = float(event.get("dur", 0))
    category = next((label for label, pattern in CATEGORIES if re.search(pattern, name, re.I)), "other")
    if event["cat"] != "kernel":
        category = "memcpy/memset"
    by_category[category] += duration
    short = re.sub(r"<.*", "", name)[:90]
    by_name[short] += duration
    count_by_name[short] += 1
starts = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in events)
window = starts[-1][1] - starts[0][0]
busy, cursor = 0.0, starts[0][0]
for begin, end in starts:
    if end > cursor:
        busy += end - max(begin, cursor)
        cursor = end
total = sum(by_category.values())
print(f"window {window / 1000:.1f} ms, GPU busy {busy / 1000:.1f} ms ({100 * busy / window:.0f}%), kernel time {total / 1000:.1f} ms, kernels {len(events)}")
if tokens:
    print(f"per generated token: window {window / tokens / 1000:.2f} ms, busy {busy / tokens / 1000:.2f} ms")
print("\nby category (share of kernel time):")
for category, duration in by_category.most_common():
    print(f"  {category:18s} {duration / 1000:9.1f} ms  {100 * duration / total:5.1f}%")
print("\ntop kernels:")
for name, duration in by_name.most_common(30):
    print(f"  {duration / 1000:8.1f} ms {count_by_name[name]:6d}x  {name}")
