"""Measure vLLM's CuTe skinny GEMM against torch linear for Qwen4Exp decode shapes on this GPU.

Usage (inside the serving image, one GPU): bench_skinny.py OUTPUT_JSON [M,M,...]
Each timing replays a CUDA graph of 20 calls that cycle through enough weight
copies to exceed L2, so every call reads its weight from memory as in decode.
A configuration is correct when its error against an FP32 reference is at most
twice torch linear's BF16 error; the fastest correct configuration is reported.
"""
import itertools
import json
import sys
import time

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig, ShapeDynamicSkinnyGemm

OUTPUT = sys.argv[1]
MS = [int(m) for m in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8]
# Local (N, K) shapes of BF16 linear layers in Qwen3.8 Flash Next revision
# 629bc3218833: replicated hyper-connection, router and MTP projections, plus the
# column- and row-parallel shards on two and four ranks.
SHAPES = {
    "hc_down_inject": (336, 10240), "hc_down_inject_324": (324, 10240), "hc_down": (320, 10240),
    "hc_up": (10240, 320), "router": (512, 2560), "mtp_indexer": (640, 2560), "mtp_fc": (2560, 2560),
    "tp2_lm_head": (124160, 2560), "tp4_lm_head": (62080, 2560),
    "tp2_ple_kv": (6400, 2560), "tp4_ple_kv": (3200, 2560),
    "tp2_mtp_qkv": (6656, 2560), "tp4_mtp_qkv": (3328, 2560),
    "tp2_mtp_o": (2560, 3072), "tp4_mtp_o": (2560, 1536),
    "tp2_mtp_shared_up": (640, 2560), "tp4_mtp_shared_up": (320, 2560),
    "tp2_mtp_shared_down": (2560, 320), "tp4_mtp_shared_down": (2560, 160),
}
L2_BYTES = 64 * 1024**2
device = torch.device("cuda")
gemm = ShapeDynamicSkinnyGemm()


def candidates(m):
    for block, width, outputs, unroll in itertools.product((32, 64, 128, 160, 256), (2, 4, 8), (1, 2, 4), (1, 2, 4)):
        yield SkinnyGemmConfig(m, block, outputs, k_unroll=unroll, vector_width=width)


def valid(config, n, k):
    return k % (config.block_size * config.vector_width) == 0 and n % config.outputs_per_block == 0


def graph_time(call, copies):
    for index in range(3):
        call(index % copies)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(20):
            call(index % copies)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / 200  # microseconds per call


results = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "shapes": {}}
for name, (n, k) in SHAPES.items():
    weight_bytes = n * k * 2
    copies = max(1, min(64, -(-2 * L2_BYTES // weight_bytes)))
    weights = [torch.randn(n, k, device=device, dtype=torch.bfloat16) / 16 for _ in range(copies)]
    entry = {}
    for m in MS:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        reference = torch.nn.functional.linear(x, weights[0])
        exact = x.float() @ weights[0].float().T
        # Accept the rounding error that torch's own BF16 linear makes, doubled.
        limit = 2 * (reference.float() - exact).abs().max().item() + 1e-3 * exact.abs().max().item()
        baseline = graph_time(lambda i: torch.nn.functional.linear(x, weights[i]), copies)
        best = None
        tried = 0
        for config in candidates(m):
            if not valid(config, n, k):
                continue
            try:
                output = gemm(x, weights[0], config)
                if (output.float() - exact).abs().max().item() > limit:
                    entry.setdefault("rejected", []).append(str(config))
                    continue
                microseconds = graph_time(lambda i: gemm(x, weights[i], config), copies)
            except Exception as error:  # noqa: BLE001 - record unusable configs and continue
                entry.setdefault("errors", []).append(f"{config}: {type(error).__name__}: {str(error)[:120]}")
                continue
            tried += 1
            if best is None or microseconds < best[1]:
                best = (config, microseconds)
        row = {"torch_us": round(baseline, 2), "tried": tried,
               "ideal_us": round(weight_bytes / 273e9 * 1e6, 2)}
        if best is not None:
            config, microseconds = best
            row.update(best_us=round(microseconds, 2), speedup=round(baseline / microseconds, 3),
                       config=[config.num_rows, config.block_size, config.outputs_per_block, config.k_unroll, config.vector_width])
        entry[str(m)] = row
        print(f"{name:22s} N={n:6d} K={k:5d} M={m:2d} torch {baseline:8.2f} us  best {row.get('best_us', '-')!s:>8} us  "
              f"x{row.get('speedup', '-')!s:6}  {row.get('config', '')}", flush=True)
    results["shapes"][name] = {"n": n, "k": k, "by_m": entry}
    del weights
    torch.cuda.empty_cache()
    with open(OUTPUT, "w") as handle:
        json.dump(results, handle, indent=1)
print("done", time.strftime("%H:%M:%S"))
