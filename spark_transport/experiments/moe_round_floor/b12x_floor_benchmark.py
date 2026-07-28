#!/usr/bin/env python3
"""Fail-closed GLM-5.2 B12X Q5/Q6 model-down benchmark scaffold.

This prototype benchmarks the exact deployed B12X TP-MoE binding path.  It is
not a generic GEMM benchmark.  ``plan`` and ``dry-run`` work without Torch,
B12X, CUDA, or a Spark.  ``live`` is intentionally restricted to the pinned
deployed source fingerprints and an SM12.1 CUDA device.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "glm52-b12x-round-floor/v1"
SEED = 20260726
HIDDEN_SIZE = 6144
GLOBAL_INTERMEDIATE_SIZE = 2048
TP_SIZE = 4
LOCAL_INTERMEDIATE_SIZE = GLOBAL_INTERMEDIATE_SIZE // TP_SIZE
NUM_EXPERTS = 256
TOPK = 8

# These are the exact sources audited in the deployed 2026-07-26 image.
PINNED_SOURCES: dict[str, tuple[str, str]] = {
    "vllm_b12x_adapter": (
        "vllm/model_executor/layers/fused_moe/b12x_moe.py",
        "d534632c7aa8ee64334cfce51c946ebf2c805cd17e46993f6f6305df4cd2fda4",
    ),
    "b12x_tp_moe": (
        "b12x/integration/tp_moe.py",
        "98f5b8b3cea77ef71253450cca412d3f6e79b95587edb184445274545dd76b27",
    ),
    "b12x_micro": (
        "b12x/moe/fused/micro.py",
        "ca1126ba045ee82084d7abaff531186a5e111ac1b8f14656198c9ecbd6f867f4",
    ),
    "b12x_silu": (
        "b12x/moe/fused/silu.py",
        "d43f95d6ea8a12e6f6c942ba92794e7c1c363bb945274123019e9ff4726691e2",
    ),
}


class GateError(RuntimeError):
    """A deliberate, user-actionable fail-closed benchmark refusal."""


@dataclass(frozen=True)
class Case:
    name: str
    width: int
    launches_per_sample: int
    execution: str
    backend: str
    routes: str
    admission: str = "enabled"
    purpose: str = ""


CASES: tuple[Case, ...] = (
    Case(
        "5xQ1-eager",
        5,
        5,
        "eager",
        "deployed-direct-micro",
        "variable",
        purpose="five one-token launches using the same Q5 routes",
    ),
    Case(
        "Q5-direct-micro-eager",
        5,
        1,
        "eager",
        "deployed-direct-micro",
        "variable",
        purpose="deployed tiny-decode implementation and launch control",
    ),
    Case(
        "Q5-direct-micro-graph",
        5,
        1,
        "graph",
        "deployed-direct-micro",
        "variable",
        purpose="production-like Q5 graph-internal baseline",
    ),
    Case(
        "Q6-direct-micro-eager",
        6,
        1,
        "eager",
        "deployed-direct-micro",
        "variable",
        purpose="MTP5 Q6 eager control",
    ),
    Case(
        "Q6-direct-micro-graph",
        6,
        1,
        "graph",
        "deployed-direct-micro",
        "variable",
        purpose="MTP5 Q6 graph-internal baseline",
    ),
    Case(
        "Q5-forced-dynamic-eager",
        5,
        1,
        "eager",
        "forced-dynamic-grouped",
        "variable",
        purpose="existing grouped implementation launch control",
    ),
    Case(
        "Q5-forced-dynamic-graph",
        5,
        1,
        "graph",
        "forced-dynamic-grouped",
        "variable",
        purpose="existing grouping benefit versus direct-micro",
    ),
    Case(
        "Q6-forced-dynamic-eager",
        6,
        1,
        "eager",
        "forced-dynamic-grouped",
        "variable",
        purpose="existing grouped MTP5 launch control",
    ),
    Case(
        "Q6-forced-dynamic-graph",
        6,
        1,
        "graph",
        "forced-dynamic-grouped",
        "variable",
        purpose="existing grouped MTP5 graph comparison",
    ),
    Case(
        "Q5-identical-route-graph",
        5,
        1,
        "graph",
        "deployed-direct-micro",
        "identical",
        purpose="optimistic physical ceiling with perfect route coherence",
    ),
    Case(
        "Q6-identical-route-graph",
        6,
        1,
        "graph",
        "deployed-direct-micro",
        "identical",
        purpose="optimistic MTP5 physical ceiling",
    ),
    Case(
        "Q5-coherent-micro-graph",
        5,
        1,
        "graph",
        "coherent-micro",
        "variable",
        admission="blocked",
        purpose="placeholder until route reuse and memory counters pass the gate",
    ),
    Case(
        "Q6-coherent-micro-graph",
        6,
        1,
        "graph",
        "coherent-micro",
        "variable",
        admission="blocked",
        purpose="placeholder until a real coherent direct-micro kernel exists",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_site_roots(explicit: str | None) -> list[Path]:
    if explicit:
        return [Path(explicit).resolve()]
    candidates: list[Path] = []
    paths = sysconfig.get_paths()
    for name in ("purelib", "platlib"):
        value = paths.get(name)
        if value:
            candidates.append(Path(value).resolve())
    for package in ("b12x", "vllm"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                candidates.append(Path(location).resolve().parent)
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def audit_sources(source_root: str | None = None) -> dict[str, dict[str, Any]]:
    roots = _candidate_site_roots(source_root)
    result: dict[str, dict[str, Any]] = {}
    for name, (relative, expected) in PINNED_SOURCES.items():
        matches = [root / relative for root in roots if (root / relative).is_file()]
        path = matches[0] if matches else None
        actual = _sha256(path) if path else None
        result[name] = {
            "relative_path": relative,
            "path": str(path) if path else None,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "state": (
                "match" if actual == expected else "mismatch" if actual else "missing"
            ),
        }
    return result


def require_pinned_sources(audit: dict[str, dict[str, Any]]) -> None:
    failures = [
        f"{name}:{entry['state']}"
        for name, entry in audit.items()
        if entry["state"] != "match"
    ]
    if failures:
        raise GateError(
            "live mode requires every pinned deployed source fingerprint; "
            + ", ".join(failures)
        )


def deterministic_routes(
    width: int,
    *,
    style: str = "variable",
    seed: int = SEED,
    experts: int = NUM_EXPERTS,
    topk: int = TOPK,
) -> list[list[int]]:
    if width <= 0 or topk <= 0 or experts < topk:
        raise ValueError("invalid route dimensions")
    generator = random.Random(seed + width * 1009)
    common = generator.sample(range(experts), topk)
    if style == "identical":
        return [list(common) for _ in range(width)]
    if style != "variable":
        raise ValueError(f"unsupported route style {style!r}")

    # Four shared and four position-specific experts give a deterministic,
    # nontrivial reuse case without pretending to be a measured GLM trace.
    shared_count = topk // 2
    shared = common[:shared_count]
    available = [expert for expert in range(experts) if expert not in shared]
    routes: list[list[int]] = []
    for position in range(width):
        local = random.Random(seed + width * 1009 + position * 65537)
        unique = local.sample(available, topk - shared_count)
        route = shared + unique
        local.shuffle(route)
        routes.append(route)
    validate_routes(routes, width=width, experts=experts, topk=topk)
    return routes


def validate_routes(
    routes: Sequence[Sequence[int]], *, width: int, experts: int, topk: int
) -> None:
    if len(routes) != width:
        raise ValueError(f"route width {len(routes)} does not match Q{width}")
    for position, route in enumerate(routes):
        if len(route) != topk:
            raise ValueError(f"position {position} has top-k {len(route)}, expected {topk}")
        if len(set(route)) != len(route):
            raise ValueError(f"position {position} contains duplicate experts")
        if any(expert < 0 or expert >= experts for expert in route):
            raise ValueError(f"position {position} contains out-of-range expert")


def synthetic_weight_bytes() -> dict[str, int]:
    e = NUM_EXPERTS
    k = HIDDEN_SIZE
    n = LOCAL_INTERMEDIATE_SIZE
    components = {
        "w1_fp4": e * (2 * n) * (k // 2),
        "w2_fp4": e * k * (n // 2),
        "w1_blockscale_e4m3": e * (2 * n) * (k // 16),
        "w2_blockscale_e4m3": e * k * (n // 16),
        "weight_global_scales_fp32": 2 * e * 4,
        "activation_global_scales_fp32": 2 * 4,
    }
    return components | {"total": sum(components.values())}


def ensure_coherent_cases_blocked(cases: Iterable[Case] = CASES) -> None:
    bad = [
        case.name
        for case in cases
        if case.backend == "coherent-micro" and case.admission != "blocked"
    ]
    if bad:
        raise GateError("coherent-micro has no implementation: " + ", ".join(bad))


def build_plan(source_root: str | None = None, *, include_audit: bool = False) -> dict:
    ensure_coherent_cases_blocked()
    report = {
        "schema": SCHEMA,
        "mode": "plan",
        "prototype": True,
        "question": (
            "Does deployed B12X direct-micro leave enough Q5/Q6 expert-weight "
            "reuse for grouping or a coherent-micro kernel to matter?"
        ),
        "model_shape": {
            "hidden_size": HIDDEN_SIZE,
            "global_intermediate_size": GLOBAL_INTERMEDIATE_SIZE,
            "tp_size": TP_SIZE,
            "local_intermediate_size": LOCAL_INTERMEDIATE_SIZE,
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "quant_mode": "nvfp4",
            "source_format": "modelopt_nvfp4",
            "activation": "silu",
            "w13_layout": "w31",
        },
        "dispatch_contract": {
            "deployed_direct_micro_name": "static",
            "direct_condition": "num_tokens <= 8 and num_tokens * topk < 64",
            "forced_dynamic_environment": {
                "B12X_STATIC_COMPACT_CUTOVER_PAIRS": "1"
            },
            "live_assertion": "binding.implementation must equal expected backend",
        },
        "allocation_contract": {
            "weights": "allocated/prepared once per child process",
            "scratch": "caller-owned plan scratch, allocated before timing",
            "output": "caller-owned fixed-address tensor",
            "events": "all CUDA events allocated before timing",
            "timed_allocations_expected": 0,
            "synthetic_weight_bytes": synthetic_weight_bytes(),
        },
        "cases": [asdict(case) for case in CASES],
        "live_gates": [
            "Linux model-down Spark",
            "CUDA SM12.1",
            "all four pinned source SHA-256 fingerprints match",
            "exact TPMoEScratchCaps/plan/bind/b12x_moe_fp4 ABI is present",
            "coherent-micro remains blocked until implemented and admitted",
        ],
    }
    if include_audit:
        report["source_audit"] = audit_sources(source_root)
    return report


def build_dry_run(source_root: str | None = None) -> dict:
    report = build_plan(source_root, include_audit=True)
    report["mode"] = "dry-run"
    report["routes"] = {
        f"Q{width}-{style}": deterministic_routes(width, style=style)
        for width in (5, 6)
        for style in ("variable", "identical")
    }
    report["validation"] = {
        "plan_serializable": True,
        "routes_valid": True,
        "live_cuda_work_attempted": False,
        "source_mismatches_are_fatal_in_live_only": True,
    }
    return report


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    return {
        "samples": len(samples_ms),
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p90_ms": _percentile(samples_ms, 0.90),
        "p99_ms": _percentile(samples_ms, 0.99),
    }


def _encoded_output(torch: Any, tensor: Any) -> dict[str, Any]:
    cpu = tensor.detach().contiguous().cpu()
    raw = bytes(cpu.view(torch.uint8).numpy())
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bf16_base64": base64.b64encode(raw).decode("ascii"),
    }


def _decoded_output(torch: Any, encoded: dict[str, Any]) -> Any:
    if encoded["dtype"] != "torch.bfloat16":
        raise GateError(f"unsupported reference dtype {encoded['dtype']!r}")
    raw = base64.b64decode(encoded["bf16_base64"], validate=True)
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
    return tensor.reshape(tuple(encoded["shape"]))


def _check_live_platform(torch: Any) -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise GateError(f"live mode requires Linux; got {sys.platform}")
    if not torch.cuda.is_available():
        raise GateError("live mode requires CUDA")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != (12, 1):
        raise GateError(f"live mode requires DGX Spark SM12.1; got SM{capability}")
    return {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _assert_live_abi(tp_moe: Any) -> dict[str, Any]:
    required = (
        "TPMoEScratchCaps",
        "plan_b12x_fp4_moe_weights",
        "prepare_b12x_fp4_moe_weights",
        "plan_tp_moe_scratch",
        "b12x_moe_fp4",
    )
    missing = [name for name in required if not hasattr(tp_moe, name)]
    if missing:
        raise GateError("pinned B12X TP-MoE ABI is incomplete: " + ", ".join(missing))
    return {"required_symbols": list(required), "missing_symbols": []}


def _allocate_synthetic_experts(torch: Any, tp_moe: Any, seed: int) -> tuple[Any, Any]:
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    e, k, n = NUM_EXPERTS, HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE

    weight_plan = tp_moe.plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=e,
        hidden_size=k,
        intermediate_size=n,
        w13_layout="w31",
    )
    w1_fp4 = torch.randint(
        0,
        256,
        (e, 2 * n, k // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_fp4 = torch.randint(
        0, 256, (e, k, n // 2), dtype=torch.uint8, device=device, generator=generator
    )
    w1_blockscale = torch.randint(
        110,
        126,
        (e, 2 * n, k // 16),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    ).view(torch.float8_e4m3fn)
    w2_blockscale = torch.randint(
        110,
        126,
        (e, k, n // 16),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    ).view(torch.float8_e4m3fn)
    weight_scale = torch.ones(e, dtype=torch.float32, device=device)
    a1_scale = torch.full((1,), 1.7, dtype=torch.float32, device=device)
    a2_scale = torch.ones(1, dtype=torch.float32, device=device)
    experts = tp_moe.prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_global_scale=weight_scale,
        w2_global_scale=weight_scale,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_scale,
        a2_gscale=a2_scale,
        params_dtype=torch.bfloat16,
    )
    return weight_plan, experts


def _runtime(
    torch: Any,
    tp_moe: Any,
    weight_plan: Any,
    experts: Any,
    *,
    width: int,
    style: str,
    seed: int,
    expected_implementation: str,
) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + width * 7919 + (1 if style == "identical" else 0))
    plan = tp_moe.plan_tp_moe_scratch(
        tp_moe.TPMoEScratchCaps(
            max_tokens=width,
            num_topk=TOPK,
            device=device,
            weight_plan=weight_plan,
            quant_mode="nvfp4",
            core_token_counts=(width,),
            route_num_experts=0,
            apply_router_weight_on_input=False,
        )
    )
    scratch = {
        spec.name: torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    }
    activation = torch.randn(
        width,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    route_list = deterministic_routes(width, style=style, seed=seed)
    topk_ids = torch.tensor(route_list, dtype=torch.int32, device=device)
    topk_weights = torch.rand(
        width, TOPK, dtype=torch.float32, device=device, generator=generator
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    output = torch.empty_like(activation)
    binding = plan.bind(
        scratch=scratch,
        a=activation,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=True,
    )
    implementation = str(binding.implementation)
    if implementation != expected_implementation:
        raise GateError(
            f"Q{width} dispatch mismatch: expected {expected_implementation!r}, "
            f"got {implementation!r}; refusing to benchmark a mislabeled backend"
        )
    return {
        "width": width,
        "style": style,
        "plan": plan,
        "scratch": scratch,
        "activation": activation,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "output": output,
        "bindings": [binding],
        "implementation": implementation,
    }


def _q1_runtime_from_q5(
    torch: Any,
    tp_moe: Any,
    weight_plan: Any,
    experts: Any,
    q5: dict[str, Any],
    *,
    expected_implementation: str,
) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    plan = tp_moe.plan_tp_moe_scratch(
        tp_moe.TPMoEScratchCaps(
            max_tokens=1,
            num_topk=TOPK,
            device=device,
            weight_plan=weight_plan,
            quant_mode="nvfp4",
            core_token_counts=(1,),
            route_num_experts=0,
            apply_router_weight_on_input=False,
        )
    )
    scratch = {
        spec.name: torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    }
    output = torch.empty_like(q5["activation"])
    bindings = []
    for position in range(5):
        binding = plan.bind(
            scratch=scratch,
            a=q5["activation"][position : position + 1],
            experts=experts,
            topk_weights=q5["topk_weights"][position : position + 1],
            topk_ids=q5["topk_ids"][position : position + 1],
            output=output[position : position + 1],
            input_scales_static=True,
        )
        if str(binding.implementation) != expected_implementation:
            raise GateError(
                "Q1 dispatch mismatch in 5xQ1 control: "
                f"expected {expected_implementation!r}, got {binding.implementation!r}"
            )
        bindings.append(binding)
    return {
        "width": 5,
        "style": "variable",
        "plan": plan,
        "scratch": scratch,
        "activation": q5["activation"],
        "topk_ids": q5["topk_ids"],
        "topk_weights": q5["topk_weights"],
        "output": output,
        "bindings": bindings,
        "implementation": expected_implementation,
    }


def _make_launch(torch: Any, tp_moe: Any, runtime: dict[str, Any], execution: str) -> Any:
    def eager_launch() -> None:
        for binding in runtime["bindings"]:
            tp_moe.b12x_moe_fp4(binding=binding)

    if execution == "eager":
        return eager_launch, None
    if execution != "graph":
        raise GateError(f"unsupported execution mode {execution!r}")

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            eager_launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        eager_launch()
    torch.cuda.synchronize()
    return graph.replay, graph


def _time_case(
    torch: Any,
    tp_moe: Any,
    case: Case,
    runtime: dict[str, Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    launch, graph_owner = _make_launch(torch, tp_moe, runtime, case.execution)
    del graph_owner  # replay owns the captured graph; name documents lifetime intent.
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    allocated_before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.nvtx.range_push(f"glm52-moe-floor:{case.name}")
    try:
        for start, end in zip(starts, ends, strict=True):
            start.record()
            launch()
            end.record()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    allocated_after = int(torch.cuda.memory_allocated())
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    output = runtime["output"]
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise GateError(f"{case.name} produced non-finite output")
    return {
        "name": case.name,
        "width": case.width,
        "launches_per_sample": case.launches_per_sample,
        "execution": case.execution,
        "backend": case.backend,
        "routes": case.routes,
        "binding_implementation": runtime["implementation"],
        "timing": _timing_summary(samples),
        "allocator": {
            "live_bytes_before": allocated_before,
            "live_bytes_after": allocated_after,
            "live_bytes_delta": allocated_after - allocated_before,
            "peak_live_bytes": int(torch.cuda.max_memory_allocated()),
            "note": (
                "fixed caller-owned tensors; live-byte delta is not an allocation "
                "event counter"
            ),
        },
        "output": {
            "finite": finite,
            "l1": float(output.float().abs().sum().item()),
            "l2": float(output.float().norm().item()),
            "encoded": _encoded_output(torch, output),
        },
    }


def _compare_output(torch: Any, result: dict[str, Any], reference: dict[str, Any]) -> dict:
    actual = _decoded_output(torch, result["output"]["encoded"]).float()
    expected = _decoded_output(torch, reference["output"]["encoded"]).float()
    delta = (actual - expected).abs()
    denom = expected.abs().amax().clamp(min=1e-6)
    return {
        "reference_case": reference["name"],
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_max": float((delta.max() / denom).item()),
        "exact_bf16": bool(torch.equal(actual, expected)),
    }


def _run_live_child(arguments: argparse.Namespace) -> dict:
    audit = audit_sources(arguments.source_root)
    require_pinned_sources(audit)

    import torch
    import b12x.integration.tp_moe as tp_moe

    platform = _check_live_platform(torch)
    abi = _assert_live_abi(tp_moe)
    expected = "static" if arguments.backend == "direct" else "dynamic"
    weight_plan, experts = _allocate_synthetic_experts(torch, tp_moe, arguments.seed)

    runtimes: dict[tuple[int, str], dict[str, Any]] = {}
    for width, style in (
        (5, "variable"),
        (6, "variable"),
        (5, "identical"),
        (6, "identical"),
    ):
        if arguments.backend == "dynamic" and style == "identical":
            continue
        runtimes[(width, style)] = _runtime(
            torch,
            tp_moe,
            weight_plan,
            experts,
            width=width,
            style=style,
            seed=arguments.seed,
            expected_implementation=expected,
        )
    if arguments.backend == "direct":
        runtimes[(1, "q5-slices")] = _q1_runtime_from_q5(
            torch,
            tp_moe,
            weight_plan,
            experts,
            runtimes[(5, "variable")],
            expected_implementation=expected,
        )

    selected = [
        case
        for case in CASES
        if case.admission == "enabled"
        and (
            arguments.backend == "direct"
            and case.backend == "deployed-direct-micro"
            or arguments.backend == "dynamic"
            and case.backend == "forced-dynamic-grouped"
        )
    ]
    results = []
    for case in selected:
        runtime = (
            runtimes[(1, "q5-slices")]
            if case.name == "5xQ1-eager"
            else runtimes[(case.width, case.routes)]
        )
        results.append(
            _time_case(
                torch,
                tp_moe,
                case,
                runtime,
                warmup=arguments.warmup,
                iterations=arguments.iterations,
            )
        )

    by_name = {result["name"]: result for result in results}
    if arguments.backend == "direct":
        by_name["5xQ1-eager"]["correctness_vs_q5"] = _compare_output(
            torch, by_name["5xQ1-eager"], by_name["Q5-direct-micro-eager"]
        )
        by_name["Q5-direct-micro-graph"]["correctness_vs_eager"] = _compare_output(
            torch,
            by_name["Q5-direct-micro-graph"],
            by_name["Q5-direct-micro-eager"],
        )
        by_name["Q6-direct-micro-graph"]["correctness_vs_eager"] = _compare_output(
            torch,
            by_name["Q6-direct-micro-graph"],
            by_name["Q6-direct-micro-eager"],
        )
    elif arguments.reference:
        reference_report = json.loads(Path(arguments.reference).read_text(encoding="utf-8"))
        reference_by_name = {
            result["name"]: result for result in reference_report["results"]
        }
        for width in (5, 6):
            dynamic = by_name[f"Q{width}-forced-dynamic-graph"]
            direct = reference_by_name[f"Q{width}-direct-micro-graph"]
            dynamic["correctness_vs_direct_micro"] = _compare_output(
                torch, dynamic, direct
            )

    return {
        "schema": SCHEMA,
        "mode": "live-child",
        "backend_child": arguments.backend,
        "seed": arguments.seed,
        "platform": platform,
        "source_audit": audit,
        "abi": abi,
        "environment": {
            "B12X_STATIC_COMPACT_CUTOVER_PAIRS": os.environ.get(
                "B12X_STATIC_COMPACT_CUTOVER_PAIRS"
            ),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        "results": results,
    }


def _child_command(
    arguments: argparse.Namespace,
    *,
    backend: str,
    output: Path,
    reference: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "live-child",
        "--backend",
        backend,
        "--warmup",
        str(arguments.warmup),
        "--iterations",
        str(arguments.iterations),
        "--seed",
        str(arguments.seed),
        "--output",
        str(output),
    ]
    if arguments.source_root:
        command.extend(["--source-root", arguments.source_root])
    if reference:
        command.extend(["--reference", str(reference)])
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["B12X_STATIC_COMPACT_CUTOVER_PAIRS"] = (
        "64" if backend == "direct" else "1"
    )
    return command, environment


def _run_live_parent(arguments: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory(prefix="glm52-b12x-floor-") as directory:
        temp = Path(directory)
        direct_path = temp / "direct.json"
        dynamic_path = temp / "dynamic.json"
        child_reports = []
        for backend, output, reference in (
            ("direct", direct_path, None),
            ("dynamic", dynamic_path, direct_path),
        ):
            command, environment = _child_command(
                arguments,
                backend=backend,
                output=output,
                reference=reference,
            )
            completed = subprocess.run(
                command,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise GateError(
                    f"{backend} child failed ({completed.returncode}); "
                    f"stdout tail={completed.stdout[-2000:]!r}; "
                    f"stderr tail={completed.stderr[-4000:]!r}"
                )
            child_reports.append(json.loads(output.read_text(encoding="utf-8")))

    results = []
    for child in child_reports:
        for result in child["results"]:
            result["output"]["encoded"].pop("bf16_base64", None)
            results.append(result)
    return {
        "schema": SCHEMA,
        "mode": "live",
        "prototype": True,
        "seed": arguments.seed,
        "platform": child_reports[0]["platform"],
        "source_audit": child_reports[0]["source_audit"],
        "results": results,
        "blocked_cases": [
            asdict(case) for case in CASES if case.admission == "blocked"
        ],
        "interpretation_guardrails": [
            "identical-route is an optimistic physical ceiling, not a workload claim",
            "CUDA-event time does not provide LPDDR bytes; collect Nsight counters",
            "coherent-micro remains unimplemented and was not simulated",
            "only profiler-attributed MoE savings may be projected into whole rounds",
        ],
    }


def _write_or_print(report: dict, output: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run", "live", "live-child"),
        default="dry-run",
    )
    parser.add_argument("--source-root", help="site-packages root for source pins")
    parser.add_argument("--output", help="optional machine-readable JSON path")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--request-coherent",
        action="store_true",
        help="exercise the intentional coherent-micro fail-closed gate",
    )
    parser.add_argument(
        "--backend", choices=("direct", "dynamic"), help=argparse.SUPPRESS
    )
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.warmup < 1 or arguments.iterations < 2:
        parser.error("--warmup must be >=1 and --iterations must be >=2")
    if arguments.mode == "live-child" and not arguments.backend:
        parser.error("--backend is required in live-child mode")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        if arguments.request_coherent:
            raise GateError(
                "coherent-micro is a placeholder, not an implementation; run the "
                "route census and memory-counter gate first"
            )
        if arguments.mode == "plan":
            report = build_plan(arguments.source_root)
        elif arguments.mode == "dry-run":
            report = build_dry_run(arguments.source_root)
        elif arguments.mode == "live-child":
            report = _run_live_child(arguments)
        else:
            report = _run_live_parent(arguments)
        _write_or_print(report, arguments.output)
        return 0
    except GateError as error:
        report = {
            "schema": SCHEMA,
            "mode": arguments.mode,
            "passed": False,
            "error": str(error),
            "error_type": "fail_closed",
        }
        _write_or_print(report, arguments.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
