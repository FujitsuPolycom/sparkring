#!/usr/bin/env python3
"""Run a small SM121 differential test of the R7 mixed-Trellis MoE path.

The probe uses deterministic representative K3, K4, and K5 EXL3 payloads. It
compares the single-grid mixed-Trellis result with both the per-tier B12X path
and an independently decoded FP16-weight/PyTorch reference. It also verifies
repeat determinism and CUDA-graph replay without loading a model checkpoint.

The composed B12X source tree is required because its GPU tests contain the
independent scalar EXL3 decoder used as the reference oracle. The installed
B12X package supplies the candidate kernels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import torch


def _install_pytest_import_stub() -> None:
    """Let packaged test oracles import when the runtime omits pytest.

    The probe calls plain helper functions from the composed B12X tests; it
    does not execute pytest fixtures or assertions. The flattened serving
    image intentionally omits pytest, so only decorator and import-or-skip
    behavior is needed while those helper modules are defined.
    """

    try:
        __import__("pytest")
        return
    except ModuleNotFoundError:
        pass

    class _Mark:
        def __getattr__(self, _name: str):
            def marker(*_args, **_kwargs):
                return lambda function: function

            return marker

    stub = ModuleType("pytest")
    stub.mark = _Mark()
    stub.importorskip = __import__
    sys.modules["pytest"] = stub


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import probe helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b12x-source",
        type=Path,
        required=True,
        help="exact composed B12X source root containing tests/moe",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--relative-error-limit", type=float, default=2.0e-2)
    parser.add_argument("--cosine-limit", type=float, default=0.999)
    parser.add_argument("--mixed-serial-limit", type=float, default=4.0e-3)
    return parser.parse_args()


def _native_projection(tier, *, projection: str, helper: ModuleType) -> torch.Tensor:
    experts = int(tier.num_experts)
    hidden = int(tier.hidden_size)
    intermediate = int(tier.intermediate_size)
    bits = int(tier.trellis_bits)
    if projection in ("gate", "up"):
        raw = tier.w13.view(torch.int32).reshape(
            2, experts, hidden // 16, intermediate // 16, 8 * bits
        )
        raw = raw.view(torch.int16).reshape(
            2, experts, hidden // 16, intermediate // 16, 16 * bits
        )
        index = 0 if projection == "gate" else 1
        payloads = raw[index]
    elif projection == "down":
        raw = tier.w2.view(torch.int32).reshape(
            experts, intermediate // 16, hidden // 16, 8 * bits
        )
        payloads = raw.view(torch.int16).reshape(
            experts, intermediate // 16, hidden // 16, 16 * bits
        )
    else:
        raise ValueError(f"unknown projection {projection!r}")
    return torch.stack(
        [
            helper._reconstruct_native(payloads[expert], codebook="mcg")
            for expert in range(experts)
        ]
    )


def main() -> int:
    args = _parse_args()
    mixed_test = args.b12x_source / "tests/moe/test_w4a16_mixed_trellis.py"
    reference_test = args.b12x_source / "tests/moe/test_fused_moe_trellis.py"
    for path in (mixed_test, reference_test):
        if not path.is_file():
            raise FileNotFoundError(path)

    _install_pytest_import_stub()
    mixed = _load_module("_sparkring_b12x_mixed_probe", mixed_test)
    reference = _load_module("_sparkring_b12x_reference_probe", reference_test)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 1):
        raise RuntimeError(f"the Spark R7 probe requires SM121, found SM{capability[0]}{capability[1]}")

    torch.manual_seed(20260810)
    m, hidden, intermediate, topk = 2, 128, 128, 3
    tiers = tuple(
        mixed._prepared(
            experts=2,
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            seed=300 + bits,
            device=device,
            codebook="mcg",
        )
        for bits in (3, 4, 5)
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 2, 4], [5, 3, 1]], dtype=torch.int32, device=device
    )
    topk_weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    tier_maps = tuple(
        torch.tensor(values, dtype=torch.int32, device=device)
        for values in (
            [0, 1, -1, -1, -1, -1],
            [-1, -1, 0, 1, -1, -1],
            [-1, -1, -1, -1, 0, 1],
        )
    )

    serial = sum(
        (
            mixed._serial_tier(x, tier, topk_weights, topk_ids, expert_map)
            for tier, expert_map in zip(tiers, tier_maps, strict=True)
        ),
        torch.zeros((m, hidden), dtype=torch.float32, device=device),
    )

    props = torch.cuda.get_device_properties(device)
    launch = mixed.compile_mixed_trellis3(
        size_m=m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        tier2_num_experts=2,
        top_k=topk,
        max_m_blocks=8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=(128, 128, 128, 128),
        trellis_codebook="mcg",
    )
    global_to_combined, descriptor = mixed.build_projection_tiered_maps(
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        tier_slots=(2, 2, 2),
        device=device,
    )
    rotations = mixed.combine_trellis_rotations(*tiers)
    buffers = mixed.make_mixed_trellis3_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )

    def run() -> torch.Tensor:
        return mixed.run_mixed_trellis3(
            x,
            *tiers,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            rotations,
            launch,
            buffers,
        )

    actual = run().clone()
    repeated = run().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(device)
    captured = captured.clone()

    decoded = {
        projection: torch.cat(
            [
                _native_projection(tier, projection=projection, helper=reference)
                for tier in tiers
            ]
        ).to(device)
        for projection in ("gate", "up", "down")
    }
    reference_output = reference._reference_full_rotation_decoded(
        x,
        topk_ids,
        topk_weights,
        decoded["gate"],
        decoded["up"],
        decoded["down"],
        torch.cat([tier.gate_suh for tier in tiers]),
        torch.cat([tier.up_suh for tier in tiers]),
        torch.cat([tier.intermediate_rotations for tier in tiers]),
        torch.cat([tier.down_svh for tier in tiers]),
        activation="silu",
    )
    torch.cuda.synchronize(device)

    denominator = reference_output.norm().clamp_min(1.0e-9)
    relative_error = float((actual - reference_output).norm() / denominator)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten(), reference_output.flatten(), dim=0
        )
    )
    mixed_serial_error = float(
        (actual - serial).norm() / serial.norm().clamp_min(1.0e-9)
    )
    finite = bool(
        torch.isfinite(actual).all()
        and torch.isfinite(serial).all()
        and torch.isfinite(reference_output).all()
    )
    repeat_equal = bool(torch.equal(repeated, actual))
    graph_equal = bool(torch.equal(captured, actual))
    passed = (
        finite
        and repeat_equal
        and graph_equal
        and relative_error <= args.relative_error_limit
        and cosine >= args.cosine_limit
        and mixed_serial_error <= args.mixed_serial_limit
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "capability": list(capability),
                "payload": "representative EXL3 MCG K3/K4/K5, two experts per tier",
                "finite": finite,
                "relative_error_vs_decoded_fp16_reference": relative_error,
                "cosine_vs_decoded_fp16_reference": cosine,
                "relative_error_vs_serial_tiers": mixed_serial_error,
                "repeat_bitwise_equal": repeat_equal,
                "cuda_graph_replay_bitwise_equal": graph_equal,
                "limits": {
                    "relative_error": args.relative_error_limit,
                    "cosine": args.cosine_limit,
                    "mixed_serial_error": args.mixed_serial_limit,
                },
                "passed": passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
