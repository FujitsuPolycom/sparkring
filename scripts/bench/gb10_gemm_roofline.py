#!/usr/bin/env python3
"""Measure the dense BF16 GEMM rate one GB10 GPU reaches through cuBLASLt.

An efficiency percentage is only defensible against a rate the same device
was observed to reach. A published peak figure does not state the clock
state, the datatype, or the sparsity assumption behind it, and published
figures for this part disagree with one another, so this measures the
achieved rate directly rather than deriving one.

WHAT IS MEASURED

    C[M, N] = A[M, K] @ B[K, N], BF16 operands and BF16 output, dispatched
    by Torch to cuBLASLt. The B operand is a view of a row-major [N, K]
    weight, which is the operand layout `torch.nn.functional.linear`
    produces. Layout is part of the result: a contiguous [K, N] B is a
    different GEMM and can reach a different rate.

    Three shapes hold K = 6144 and N = 512 with M in {40, 128, 512}. Those
    two dimensions are the GLM-5.2 hidden size and the per-rank expert
    intermediate width at TP4 that
    `spark_transport/experiments/moe_round_floor/b12x_floor_benchmark.py`
    records; M spans a decode-sized to a prefill-sized token count. This is
    a dense cuBLASLt GEMM at those dimensions, not the fused MoE binding,
    which that file measures instead.

    A 4096-cubed shape is measured in the same run and labelled `control`.
    It is not one of the three specified shapes. It exists so the skinny
    numbers can be read against a shape on the same device, in the same
    process, under the same clocks, that is large enough to approach the
    device's multiply-accumulate limit.

WHY THE THREE SPECIFIED SHAPES ARE NOT A PEAK MEASUREMENT

    All three are small-M skinny GEMMs. Two reported properties make the
    regime visible:

      * Arithmetic intensity, FLOP per compulsory byte. At M = 40 the B
        operand is most of the traffic and is read once per call, so
        intensity is bounded near 2*M and the shape is operand-traffic
        bound long before it is arithmetic bound.
      * Output tile count. With N = 512, a 128-wide output tile gives four
        tile columns, so the launched grid is a small multiple of four
        whatever M is. That is far below the device's streaming
        multiprocessor count, so much of the device is idle by
        construction.

    A per-call floor is measured on the same code path with M = N = K = 1
    and reported separately. It bounds how much of a small-M shape's
    elapsed time can be attributed to per-call dispatch rather than to
    work.

METHOD

    Each timed call is bracketed by its own pair of CUDA events. Wall clock
    is not used: the launches are asynchronous, so a host timer around them
    measures queueing rather than execution. The device is synchronized
    after the warmup calls and again after the timed loop, and event times
    are read only after that second synchronization.

    Warmup runs before every measured shape. The first call to a shape pays
    workspace allocation and the cuBLASLt heuristic and algorithm selection
    for that shape; including it would report a one-time cost as a steady
    state rate.

    Operand and output tensors are allocated once per shape and reused, so
    the caching allocator does no work inside the timed loop.

    The reported statistic is the median with its interquartile range, plus
    the observed minimum and maximum. A single number with no spread does
    not say whether the device was in a steady state.

OUTPUT

    A human-readable report on stdout always. `--json PATH` additionally
    writes the same content as JSON, and `--json -` writes that JSON to
    stdout, so a result can be recorded without retyping numbers.

    The FLOP and byte formulas are printed with the numbers so a reader can
    check the arithmetic rather than trust it.

Safety class: OFFLINE. It runs entirely on the machine that invokes it: it
contacts no configured Spark, starts and stops no service, installs
nothing, and writes only the JSON path the caller names. Two qualifications
that OFFLINE does not otherwise imply. It allocates device memory for the
duration of the run, so it must not be pointed at a GPU that is serving:
GB10 memory is unified and the control shape allocates hundreds of
megabytes. It also runs `nvidia-smi` locally to record clock and power
state, which queries the driver and mutates nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "gb10-dense-gemm-roofline/v1"

EXIT_OK = 0
# The measurement could not be taken because the environment cannot support
# it. That is distinct from a measurement that ran and produced numbers.
EXIT_UNAVAILABLE = 2

BF16_BYTES = 2

# The output tile width assumed when reporting a tile count. cuBLASLt picks
# its own tile size per shape, so this indicates the grid's scale, not the
# grid that was launched.
ASSUMED_TILE = 128

FLOP_FORMULA = "flop = 2 * M * N * K"
BYTES_FORMULA = "bytes = 2 * (M*K + K*N + M*N)"


class MeasurementUnavailable(RuntimeError):
    """The environment cannot support the measurement, so none was taken."""


@dataclass(frozen=True)
class Shape:
    """One GEMM, measured as C[M, N] = A[M, K] @ B[K, N]."""

    name: str
    m: int
    n: int
    k: int
    role: str


# `specified` shapes are the measurement this script exists to take.
# `control` is a reference point on the same device, not one of them.
SHAPES: tuple[Shape, ...] = (
    Shape("m40_k6144_n512", 40, 512, 6144, "specified"),
    Shape("m128_k6144_n512", 128, 512, 6144, "specified"),
    Shape("m512_k6144_n512", 512, 512, 6144, "specified"),
    Shape("m4096_k4096_n4096", 4096, 4096, 4096, "control"),
)

LAUNCH_FLOOR = Shape("per_call_floor", 1, 1, 1, "floor")

NVIDIA_SMI_FIELDS = (
    "name",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.applications.graphics",
    "clocks_throttle_reasons.active",
    "persistence_mode",
    "power.draw",
    "power.limit",
    "temperature.gpu",
)

# What nvidia-smi prints for a field the driver does not answer for.
UNREPORTED = ("", "N/A", "[N/A]", "[Not Supported]", "[Unknown Error]")


# --------------------------------------------------------------------------
# Shape arithmetic. Pure; exercised without a GPU.
# --------------------------------------------------------------------------


def flop_count(shape: Shape) -> int:
    """FLOP in one call: two per inner-product term, one multiply one add."""

    return 2 * shape.m * shape.n * shape.k


def compulsory_bytes(shape: Shape, element_bytes: int = BF16_BYTES) -> int:
    """Bytes moved if every operand and output element is touched once.

    A lower bound on traffic, not a measurement of it. A kernel that
    re-reads an operand because it does not fit in cache moves more.
    """

    return element_bytes * (shape.m * shape.k + shape.k * shape.n + shape.m * shape.n)


def arithmetic_intensity(shape: Shape, element_bytes: int = BF16_BYTES) -> float:
    """FLOP per compulsory byte."""

    return flop_count(shape) / compulsory_bytes(shape, element_bytes)


def output_tiles(shape: Shape, tile: int = ASSUMED_TILE) -> int:
    """Output tiles at an assumed tile size: the grid's scale, not the grid."""

    return math.ceil(shape.m / tile) * math.ceil(shape.n / tile)


def tflops(flop: int, milliseconds: float) -> float:
    """Achieved rate in TFLOP/s for `flop` completed in `milliseconds`."""

    if milliseconds <= 0:
        raise ValueError(f"elapsed time must be positive; got {milliseconds}")
    return flop / (milliseconds * 1e-3) / 1e12


def gigabytes_per_second(byte_count: int, milliseconds: float) -> float:
    """Achieved compulsory-traffic rate in GB/s, 10^9 bytes per second."""

    if milliseconds <= 0:
        raise ValueError(f"elapsed time must be positive; got {milliseconds}")
    return byte_count / (milliseconds * 1e-3) / 1e9


# --------------------------------------------------------------------------
# Distribution. Pure; exercised without a GPU.
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile over `values`, which need not be sorted."""

    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1]; got {fraction}")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples_ms: Sequence[float]) -> dict[str, float]:
    """Median, interquartile range, and observed extremes of a timing sample."""

    if not samples_ms:
        raise ValueError("cannot summarize an empty timing sample")
    ordered = sorted(float(sample) for sample in samples_ms)
    p25 = percentile(ordered, 0.25)
    p75 = percentile(ordered, 0.75)
    return {
        "samples": len(ordered),
        "min_ms": ordered[0],
        "p25_ms": p25,
        "median_ms": statistics.median(ordered),
        "p75_ms": p75,
        "max_ms": ordered[-1],
        "iqr_ms": p75 - p25,
    }


def shape_result(shape: Shape, samples_ms: Sequence[float]) -> dict[str, Any]:
    """One shape's timing distribution and the rates implied by it."""

    timing = summarize(samples_ms)
    flop = flop_count(shape)
    traffic = compulsory_bytes(shape)
    return {
        "name": shape.name,
        "role": shape.role,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "flop": flop,
        "compulsory_bytes": traffic,
        "arithmetic_intensity_flop_per_byte": arithmetic_intensity(shape),
        "output_tiles_at_128": output_tiles(shape),
        "timing": timing,
        "rate": {
            "tflops_at_median": tflops(flop, timing["median_ms"]),
            "tflops_at_min": tflops(flop, timing["min_ms"]),
            "compulsory_gigabytes_per_second_at_median": gigabytes_per_second(
                traffic, timing["median_ms"]
            ),
        },
    }


# --------------------------------------------------------------------------
# Conditions that make a result reproducible and interpretable.
# --------------------------------------------------------------------------


def read_clock_state(
    device_index: int = 0,
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Clock, throttle, and power state from nvidia-smi, or why it is absent.

    A result whose clock state is unknown is still a result; a result that
    silently omits it is not, so every failure path returns a stated reason.
    """

    binary = which("nvidia-smi")
    if binary is None:
        return {"read": False, "reason": "nvidia-smi is not on PATH"}
    command = [
        binary,
        "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS),
        "--format=csv,noheader,nounits",
        "-i",
        str(device_index),
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return {"read": False, "reason": f"nvidia-smi could not be run: {error!r}"}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return {
            "read": False,
            "reason": detail[-1] if detail else "nvidia-smi exited non-zero",
        }
    rows = (result.stdout or "").strip().splitlines()
    if not rows:
        return {"read": False, "reason": "nvidia-smi returned no rows"}
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(NVIDIA_SMI_FIELDS):
        return {
            "read": False,
            "reason": (
                f"nvidia-smi returned {len(values)} fields where "
                f"{len(NVIDIA_SMI_FIELDS)} were requested"
            ),
        }
    fields = dict(zip(NVIDIA_SMI_FIELDS, values))
    applied = fields["clocks.applications.graphics"]
    pinned = applied not in UNREPORTED
    return {
        "read": True,
        "fields": fields,
        "application_clocks_pinned": pinned,
        "lock_note": (
            f"application clocks pinned at {applied} MHz"
            if pinned
            else (
                "application clocks are not pinned; the driver is free to "
                "move the SM clock during the run, so compare the before and "
                "after readings"
            )
        ),
    }


def describe_environment(torch_module: Any, device_index: int) -> dict[str, Any]:
    """Torch, CUDA, and device facts a reader needs to interpret the numbers."""

    properties = torch_module.cuda.get_device_properties(device_index)
    capability = tuple(torch_module.cuda.get_device_capability(device_index))
    reduced = getattr(
        torch_module.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
        None,
    )
    return {
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
        "device_index": device_index,
        "device_name": torch_module.cuda.get_device_name(device_index),
        "compute_capability": list(capability),
        "multi_processor_count": int(getattr(properties, "multi_processor_count", 0)),
        "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
        "dtype": "torch.bfloat16",
        "element_bytes": BF16_BYTES,
        "allow_bf16_reduced_precision_reduction": reduced,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


# --------------------------------------------------------------------------
# Device path.
# --------------------------------------------------------------------------


def import_torch() -> Any:
    """Import Torch, or state that the measurement cannot be taken."""

    try:
        import torch
    except ImportError as error:
        raise MeasurementUnavailable(
            f"torch is not importable, so no GEMM can be dispatched: {error}"
        ) from error
    return torch


def require_cuda_device(torch_module: Any, device_index: int) -> None:
    """Refuse, with a reason, unless the named CUDA device exists."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise MeasurementUnavailable(
            "no CUDA device is visible to torch. This measures a GB10 GPU and "
            "has no CPU fallback, because a CPU BF16 GEMM rate answers a "
            "different question. Run it on the GB10 host, inside the serving "
            "runtime image, with the device visible to the container."
        )
    count = int(cuda.device_count())
    if not 0 <= device_index < count:
        raise MeasurementUnavailable(
            f"--device {device_index} does not exist; torch sees {count} CUDA "
            "device(s)"
        )


def measure_shape(
    torch_module: Any,
    shape: Shape,
    *,
    device: Any,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Return one elapsed time in milliseconds per timed call of `shape`.

    The B operand is `weight.t()`, a view of a row-major [N, K] weight, so
    the dispatched GEMM is the one `torch.nn.functional.linear` produces.
    The output tensor is preallocated and reused, keeping the caching
    allocator out of the timed loop.
    """

    left = torch_module.randn(
        shape.m, shape.k, device=device, dtype=torch_module.bfloat16
    )
    weight = torch_module.randn(
        shape.n, shape.k, device=device, dtype=torch_module.bfloat16
    )
    right = weight.t()
    out = torch_module.empty(
        shape.m, shape.n, device=device, dtype=torch_module.bfloat16
    )

    # Warmup absorbs workspace allocation and the cuBLASLt heuristic and
    # algorithm selection for this shape, both one-time costs.
    for _ in range(warmup):
        torch_module.mm(left, right, out=out)
    torch_module.cuda.synchronize(device)

    starts = [torch_module.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch_module.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        torch_module.mm(left, right, out=out)
        end.record()
    torch_module.cuda.synchronize(device)

    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    if not bool(torch_module.isfinite(out).all().item()):
        raise MeasurementUnavailable(
            f"{shape.name} produced non-finite output, so its timing is not a "
            "measurement of a completed GEMM"
        )
    del left, weight, right, out
    torch_module.cuda.empty_cache()
    return samples


def measure(
    torch_module: Any,
    *,
    device_index: int,
    warmup: int,
    iterations: int,
    shapes: Sequence[Shape] = SHAPES,
) -> dict[str, Any]:
    """Run every shape plus the per-call floor and assemble the report."""

    require_cuda_device(torch_module, device_index)
    device = torch_module.device("cuda", device_index)
    torch_module.cuda.set_device(device)

    environment = describe_environment(torch_module, device_index)
    clocks_before = read_clock_state(device_index)

    floor_samples = measure_shape(
        torch_module,
        LAUNCH_FLOOR,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    results = [
        shape_result(
            shape,
            measure_shape(
                torch_module,
                shape,
                device=device,
                warmup=warmup,
                iterations=iterations,
            ),
        )
        for shape in shapes
    ]

    clocks_after = read_clock_state(device_index)
    return build_report(
        environment=environment,
        clocks_before=clocks_before,
        clocks_after=clocks_after,
        floor_timing=summarize(floor_samples),
        results=results,
        warmup=warmup,
        iterations=iterations,
    )


# --------------------------------------------------------------------------
# Report assembly and rendering. Pure; exercised without a GPU.
# --------------------------------------------------------------------------


def build_report(
    *,
    environment: dict[str, Any],
    clocks_before: dict[str, Any],
    clocks_after: dict[str, Any],
    floor_timing: dict[str, float],
    results: Sequence[dict[str, Any]],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Assemble the machine-readable document."""

    return {
        "schema": SCHEMA,
        "measurement": {
            "operation": "dense GEMM, C[M, N] = A[M, K] @ B[K, N]",
            "dispatch": (
                "torch.mm with B a view of a row-major [N, K] weight, the "
                "operand layout torch.nn.functional.linear produces; Torch "
                "routes this to cuBLASLt"
            ),
            "dtype": "torch.bfloat16",
            "element_bytes": BF16_BYTES,
            "flop_formula": FLOP_FORMULA,
            "bytes_formula": BYTES_FORMULA,
            "bytes_note": (
                "compulsory traffic: every operand and output element counted "
                "once. A lower bound on traffic, not a measurement of it."
            ),
            "timing": (
                "one CUDA event pair per call; the device is synchronized "
                "after warmup and after the timed loop, and event times are "
                "read only after that second synchronization"
            ),
            "statistic": "median with interquartile range, plus observed min and max",
            "warmup_calls_per_shape": warmup,
            "timed_calls_per_shape": iterations,
            "tile_assumption": (
                f"output_tiles_at_128 assumes a {ASSUMED_TILE}-wide output "
                "tile; cuBLASLt selects its own tile size, so this indicates "
                "the grid's scale rather than the grid that was launched"
            ),
            "single_process": True,
            "distributed": False,
        },
        "environment": environment,
        "clocks": {"before": clocks_before, "after": clocks_after},
        "per_call_floor": {
            "m": LAUNCH_FLOOR.m,
            "n": LAUNCH_FLOOR.n,
            "k": LAUNCH_FLOOR.k,
            "timing": floor_timing,
            "note": (
                "smallest GEMM on the same code path; an upper bound on how "
                "much of a small-M shape's elapsed time is per-call dispatch "
                "rather than work"
            ),
        },
        "shapes": list(results),
    }


def _clock_line(label: str, state: dict[str, Any]) -> list[str]:
    if not state.get("read"):
        return [f"  {label:<22}NOT READ ({state.get('reason', 'no reason given')})"]
    fields = state["fields"]
    return [
        f"  {label:<22}sm {fields['clocks.sm']} MHz of max "
        f"{fields['clocks.max.sm']} MHz, {fields['power.draw']} W of "
        f"{fields['power.limit']} W, {fields['temperature.gpu']} C, "
        f"throttle: {fields['clocks_throttle_reasons.active']}"
    ]


def render_text(report: dict[str, Any]) -> str:
    """Render the report for a person reading a terminal."""

    environment = report["environment"]
    measurement = report["measurement"]
    capability = "".join(str(part) for part in environment["compute_capability"])
    lines = [
        "GB10 dense BF16 GEMM rate, single GPU, single process",
        f"schema: {report['schema']}",
        "",
        "CONDITIONS",
        f"  {'torch':<22}{environment['torch_version']} "
        f"(CUDA {environment['torch_cuda_version']})",
        f"  {'device':<22}{environment['device_name']} "
        f"(index {environment['device_index']}, SM{capability}, "
        f"{environment['multi_processor_count']} SMs)",
        f"  {'dtype':<22}{environment['dtype']} "
        f"({environment['element_bytes']} bytes/element)",
        f"  {'bf16 reduced reduce':<22}"
        f"{environment['allow_bf16_reduced_precision_reduction']}",
        f"  {'platform':<22}{environment['platform']}",
        f"  {'warmup calls/shape':<22}{measurement['warmup_calls_per_shape']}",
        f"  {'timed calls/shape':<22}{measurement['timed_calls_per_shape']}",
        f"  {'timing':<22}{measurement['timing']}",
        f"  {'operand layout':<22}{measurement['dispatch']}",
    ]
    lines += _clock_line("clocks before", report["clocks"]["before"])
    lines += _clock_line("clocks after", report["clocks"]["after"])
    before = report["clocks"]["before"]
    if before.get("read"):
        lines.append(f"  {'clock lock':<22}{before['lock_note']}")

    floor = report["per_call_floor"]
    lines += [
        "",
        "ARITHMETIC",
        f"  {FLOP_FORMULA}",
        f"  {BYTES_FORMULA}",
        f"  {measurement['bytes_note']}",
        "  arithmetic intensity = flop / bytes",
        "",
        "PER-CALL FLOOR",
        f"  M=N=K=1 on the same path: median "
        f"{floor['timing']['median_ms']:.4f} ms, "
        f"min {floor['timing']['min_ms']:.4f} ms",
        f"  {floor['note']}",
        "",
        "SHAPES",
        f"  {'name':<20}{'role':<10}{'M':>6}{'N':>6}{'K':>6}"
        f"{'median ms':>12}{'IQR ms':>10}{'min ms':>10}{'max ms':>10}"
        f"{'TFLOP/s med':>13}{'TFLOP/s min':>13}{'GB/s med':>11}"
        f"{'AI f/B':>9}{'tiles128':>10}",
    ]
    for entry in report["shapes"]:
        timing = entry["timing"]
        rate = entry["rate"]
        lines.append(
            f"  {entry['name']:<20}{entry['role']:<10}"
            f"{entry['m']:>6}{entry['n']:>6}{entry['k']:>6}"
            f"{timing['median_ms']:>12.4f}{timing['iqr_ms']:>10.4f}"
            f"{timing['min_ms']:>10.4f}{timing['max_ms']:>10.4f}"
            f"{rate['tflops_at_median']:>13.2f}{rate['tflops_at_min']:>13.2f}"
            f"{rate['compulsory_gigabytes_per_second_at_median']:>11.1f}"
            f"{entry['arithmetic_intensity_flop_per_byte']:>9.1f}"
            f"{entry['output_tiles_at_128']:>10}"
        )

    specified = [entry for entry in report["shapes"] if entry["role"] == "specified"]
    control = [entry for entry in report["shapes"] if entry["role"] == "control"]
    lines += [
        "",
        "HOW TO READ THIS",
        "  Rows marked `specified` are small-M skinny GEMMs. Their arithmetic",
        "  intensity and tile count are printed above so the regime is visible:",
        "  a low intensity means operand traffic bounds the row, and a tile",
        f"  count far below the device's {environment['multi_processor_count']} SMs",
        "  means most of the device is idle. Neither is a statement about the",
        "  device's multiply-accumulate limit.",
        "  The `control` row is not one of the specified shapes. It is a square",
        "  shape large enough to approach that limit, measured on the same",
        "  device in the same process under the same clocks, so the specified",
        "  rows can be read against it.",
    ]
    if specified and control:
        best = max(entry["rate"]["tflops_at_median"] for entry in specified)
        reference = max(entry["rate"]["tflops_at_median"] for entry in control)
        if reference > 0:
            lines.append(
                f"  Highest specified row reaches {best:.2f} TFLOP/s; the "
                f"control reaches {reference:.2f} TFLOP/s, so the specified "
                f"shapes reach {100.0 * best / reference:.1f}% of the control."
            )
    lines.append(
        "  Quote a rate with its shape. A rate from one shape does not "
        "transfer to another."
    )
    return "\n".join(lines) + "\n"


def emit_json(report: dict[str, Any], destination: str) -> None:
    """Write the report as JSON to a path, or to stdout for `-`."""

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if destination == "-":
        print(rendered)
        return
    Path(destination).write_text(rendered + "\n", encoding="utf-8")


def render_shape_table() -> str:
    """The shapes this script measures, without touching a device."""

    lines = [
        f"{'name':<20}{'role':<10}{'M':>6}{'N':>6}{'K':>6}{'flop':>16}"
        f"{'bytes':>14}{'AI f/B':>9}{'tiles128':>10}"
    ]
    for shape in (*SHAPES, LAUNCH_FLOOR):
        lines.append(
            f"{shape.name:<20}{shape.role:<10}{shape.m:>6}{shape.n:>6}"
            f"{shape.k:>6}{flop_count(shape):>16}{compulsory_bytes(shape):>14}"
            f"{arithmetic_intensity(shape):>9.1f}{output_tiles(shape):>10}"
        )
    lines += ["", FLOP_FORMULA, BYTES_FORMULA]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Command line.
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gb10_gemm_roofline",
        description=(
            "Measure the dense BF16 GEMM rate one GB10 GPU reaches through "
            "cuBLASLt, for K=6144 N=512 at M in {40, 128, 512} plus a "
            "4096-cubed control shape."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to measure (default: 0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help=(
            "untimed calls per shape before measuring, absorbing workspace "
            "allocation and cuBLASLt algorithm selection (default: 50)"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="timed calls per shape (default: 200)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write the report as JSON to PATH, or to stdout for '-'",
    )
    parser.add_argument(
        "--list-shapes",
        action="store_true",
        help="print the shapes with their FLOP and byte counts and measure nothing",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.warmup < 1:
        parser.error(
            "--warmup must be at least 1; the first call to a shape is not a "
            "steady state"
        )
    if arguments.iterations < 4:
        parser.error("--iterations must be at least 4 for an interquartile range")
    return arguments


def main(
    argv: Sequence[str] | None = None,
    *,
    load_torch: Callable[[], Any] = import_torch,
) -> int:
    arguments = parse_args(argv)

    if arguments.list_shapes:
        print(render_shape_table(), end="")
        return EXIT_OK

    try:
        torch_module = load_torch()
        report = measure(
            torch_module,
            device_index=arguments.device,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
        )
    except MeasurementUnavailable as error:
        print(f"FAIL no measurement taken: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    print(render_text(report), end="")
    if arguments.json:
        emit_json(report, arguments.json)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
