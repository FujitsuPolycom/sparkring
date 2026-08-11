"""Compare one GLM-5.2 TP-local shared-expert projection across BF16,
cached online EXL3 K6, and the native B12X MXFP8 path.

This is a diagnostic harness for GB10/SM121.  It loads only one 25 MiB BF16
weight and one roughly 2.3 MiB K6 cache entry; it never constructs the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def cache_key(path: Path) -> dict[str, object]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return json.loads(handle.metadata()["cache_key"])


def find_cache(root: Path, prefix: str, tp_rank: int) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in root.rglob("*.safetensors"):
        key = cache_key(path)
        if key.get("prefix") == prefix and int(key.get("tp_rank", -1)) == tp_rank:
            matches.append((path, key))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one cache for prefix={prefix!r}, tp_rank={tp_rank}; "
            f"found {[str(path) for path, _ in matches]}"
        )
    return matches[0]


def metrics(name: str, output: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    value = output.float()
    ref = reference.float()
    finite = torch.isfinite(value)
    delta = value - ref
    ref_norm = torch.linalg.vector_norm(ref)
    rel_f = torch.linalg.vector_norm(delta) / ref_norm
    return {
        "name": name,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": int(finite.sum().item()),
        "elements": output.numel(),
        "nan": int(torch.isnan(value).sum().item()),
        "posinf": int(torch.isposinf(value).sum().item()),
        "neginf": int(torch.isneginf(value).sum().item()),
        "absmax": float(torch.where(finite, value.abs(), 0).max().item()),
        "rel_f_vs_bf16": float(rel_f.item()),
        "max_abs_vs_bf16": float(delta.abs().max().item()),
    }


def graph_replay(fn, source: torch.Tensor) -> torch.Tensor:
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            fn(source)
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()

    static_source = source.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn(static_source)
    graph.replay()
    torch.cuda.synchronize()
    return static_output.clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("/models/glm52-exl3-r7-3.5bpw/model-sharedbf16.safetensors"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/cache/exl3-online")
    )
    parser.add_argument(
        "--prefix", default="model.layers.3.mlp.shared_experts.down_proj"
    )
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32])
    args = parser.parse_args()

    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"this probe targets GB10/SM121, got capability={capability}")

    cache_path, key = find_cache(args.cache_root, args.prefix, args.tp_rank)
    if int(key["bits"]) != 6 or int(key["tp_world_size"]) != 4:
        raise RuntimeError(f"expected K6 TP4 cache, got {key}")

    tensor_name = args.prefix + ".weight"
    with safe_open(args.model_file, framework="pt", device="cpu") as handle:
        full_weight = handle.get_tensor(tensor_name)

    input_size = int(key["input_size"])
    output_size = int(key["output_size"])
    tp_world_size = int(key["tp_world_size"])
    column_start = args.tp_rank * input_size
    column_end = column_start + input_size
    if tuple(full_weight.shape) != (output_size, input_size * tp_world_size):
        raise RuntimeError(
            f"probe expects a row-parallel down_proj weight; got "
            f"{tuple(full_weight.shape)} for cache {key}"
        )
    weight = full_weight[:, column_start:column_end].contiguous().cuda()
    del full_weight

    with safe_open(cache_path, framework="pt", device="cpu") as handle:
        trellis = handle.get_tensor("trellis").cuda()
        suh = handle.get_tensor("suh").cuda()
        svh = handle.get_tensor("svh").cuda()

    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.layers.quantization.exl3 import _b12x_trellis_linear
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    quant_weight, weight_scale = mxfp8_e4m3_quantize(weight.contiguous())
    mx_layer = torch.nn.Module()
    mx_layer.prefix = "probe.shared_experts.down_proj"
    mx_layer.register_parameter(
        "weight", torch.nn.Parameter(quant_weight, requires_grad=False)
    )
    mx_layer.register_parameter(
        "weight_scale", torch.nn.Parameter(weight_scale, requires_grad=False)
    )
    mx_kernel = init_mxfp8_linear_kernel()
    if type(mx_kernel).__name__ != "B12xMxfp8LinearKernel":
        raise RuntimeError(f"expected B12X MXFP8 kernel, got {type(mx_kernel).__name__}")
    mx_kernel.process_weights_after_loading(mx_layer)

    def bf16(source: torch.Tensor) -> torch.Tensor:
        return F.linear(source, weight)

    def k6(source: torch.Tensor) -> torch.Tensor:
        result = _b12x_trellis_linear(source.half(), trellis, suh, svh)
        return result.to(source.dtype)

    def mxfp8(source: torch.Tensor) -> torch.Tensor:
        return mx_kernel.apply_weights(mx_layer, source)

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": capability,
                "cache": str(cache_path),
                "cache_key": key,
                "bf16_weight_shape": list(weight.shape),
                "mxfp8_kernel": type(mx_kernel).__name__,
            },
            sort_keys=True,
        )
    )

    torch.manual_seed(0)
    bad = False
    with torch.inference_mode():
        for rows in args.rows:
            source = torch.randn(
                rows, input_size, dtype=torch.bfloat16, device="cuda"
            )
            reference = bf16(source)
            eager_k6 = k6(source)
            eager_mx = mxfp8(source)
            torch.cuda.synchronize()
            print(json.dumps(metrics(f"bf16-eager-m{rows}", reference, reference)))
            print(json.dumps(metrics(f"k6-eager-m{rows}", eager_k6, reference)))
            print(json.dumps(metrics(f"mxfp8-eager-m{rows}", eager_mx, reference)))

            graph_k6 = graph_replay(k6, source)
            graph_mx = graph_replay(mxfp8, source)
            print(json.dumps(metrics(f"k6-graph-m{rows}", graph_k6, reference)))
            print(json.dumps(metrics(f"mxfp8-graph-m{rows}", graph_mx, reference)))
            bad |= not bool(torch.isfinite(eager_k6).all().item())
            bad |= not bool(torch.isfinite(eager_mx).all().item())
            bad |= not bool(torch.isfinite(graph_k6).all().item())
            bad |= not bool(torch.isfinite(graph_mx).all().item())
    raise SystemExit(2 if bad else 0)


if __name__ == "__main__":
    main()
