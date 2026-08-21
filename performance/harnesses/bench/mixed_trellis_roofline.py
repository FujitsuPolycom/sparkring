#!/usr/bin/env python3
"""Measure and tune the mixed-bitrate Trellis MoE kernel at one GB10 geometry.

WHAT IS MEASURED

    One call of the fused mixed-bitrate Trellis mixture-of-experts kernel
    exposed by `b12x.moe._shared.kernels.w4a16.mixed_trellis`, driven the
    way the vLLM EXL3 quantization backend drives it: two expert tiers at
    different Trellis bit widths behind a single grid, routed by a top-k
    identifier tensor, with the rotations and packed weights the companion
    `prepare` module produces.

    The default geometry is a four-Spark GB10 deployment at tensor
    parallel 4: hidden size 6144, per-rank expert intermediate width 512,
    192 experts at 3 bits and 64 experts at 4 bits, top-k 8, route block
    size 8, 48 streaming multiprocessors. Token counts 40, 128, and 512
    span the deployed decode capacity to a prefill-sized batch.

    Weights are synthesized by `prepare_trellis256_moe_weights` with
    `w13=None` and `w2=None`. No checkpoint is loaded, so the numbers
    describe the kernel at a geometry, not a particular model's values.

TWO MODES

    `measure` runs one configuration across the requested token counts and
    reports elapsed time, a dense-equivalent arithmetic rate, and an
    achieved weight-stream rate over compressed bytes.

    `tune` holds the geometry fixed, sweeps `force_tile_config` and
    `moe_block_size`, and ranks every configuration against the deployed
    one. A configuration that fails to compile or produces a non-finite
    result is recorded as a skip carrying its exception type; it does not
    end the sweep.

DENSE-EQUIVALENT ARITHMETIC

    A gated mixture-of-experts token routed to one expert passes through
    three matrix products: gate and up, each hidden_size to
    intermediate_size, and down, intermediate_size to hidden_size. With two
    flops per inner-product term,

        dense_equivalent_flop
            = 2 * (size_m * top_k)
              * (2 * hidden_size * intermediate_size
                 + intermediate_size * hidden_size)
            = 6 * size_m * top_k * hidden_size * intermediate_size

    The count is labelled dense-equivalent because it is the arithmetic a
    dense BF16 GEMM would perform for the same routed work. It is directly
    comparable with a dense GEMM rate such as the one
    `performance/harnesses/bench/gb10_gemm_roofline.py` reports. It is not a count of the
    operations this kernel issues: the kernel also decodes Trellis codes
    and applies rotations, and it performs no arithmetic for experts no
    token selected.

COMPRESSED WEIGHT TRAFFIC

    Each expert holds gate, up, and down weights, so 3 * hidden_size *
    intermediate_size coefficients, stored at its tier's bit width:

        expert_compressed_bytes = 3 * hidden_size * intermediate_size
                                  * tier_bits / 8
        weight_stream_bytes     = that, summed over the experts the routing
                                  selected, per tier

    The reported GB/s divides that by elapsed time. It assumes each
    selected expert's compressed weight is read once per call; a kernel
    that re-reads an expert across m-blocks moves more, so the figure is a
    lower bound on traffic and therefore a lower bound on the achieved
    rate. Whether the kernel is traffic-bound or decode-bound decides
    whether the 3-bit and 4-bit tier mix affects elapsed time at all, so
    the tier split of the traffic is reported alongside it.

WEIGHT-POOL CYCLING

    Compressed expert weights for this geometry are on the order of a
    gigabyte, which is small enough that repeated calls on one weight set
    can be served from cache and report a rate no deployment reaches.
    Several independent weight sets are therefore prepared and one is used
    per timed call, round-robin. `--pool-size` sets the count; the
    predicted device memory cost is computed, reported, and refused when it
    exceeds a stated fraction of device memory.

METHOD

    Each timed call is bracketed by its own pair of CUDA events. The device
    is synchronized after the warmup calls and again after the timed loop,
    and event times are read only after that second synchronization.
    Warmup absorbs the kernel's compilation, which the first call to a
    configuration pays. Input activations, routing tensors, tier maps,
    launch descriptors, and device buffers are allocated before the loop.

    The reported statistic is the median with its interquartile range plus
    the observed minimum and maximum.

API DISCOVERY

    The kernel entry points are resolved by introspection at run time, and
    every signature this harness relies on is checked before a call is made
    and recorded in the report. A signature that does not match is a stated
    refusal naming what was found. Argument order is never assumed
    silently: `run_mixed_trellis` is called positionally only after its
    first ten parameter names are confirmed in the expected order.

    The resolved `__file__` of the kernel module is compared against the
    path the report names. Import hooks can bind a different implementation
    to the same module name, and a measurement that names one module while
    timing another is not a measurement of either.

Safety class: OFFLINE. It runs entirely on the machine that invokes it: it
contacts no configured Spark, starts and stops no service, installs
nothing, and writes only the JSON path the caller names. Two qualifications
that OFFLINE does not otherwise imply. It allocates device memory for the
duration of the run, roughly a gigabyte per weight set at the default
geometry, so it must not be pointed at a GPU that is serving: GB10 memory
is unified. It also runs `nvidia-smi` locally to record clock and power
state, which queries the driver and mutates nothing.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

SCHEMA = "gb10-mixed-trellis-roofline/v1"

EXIT_OK = 0
# The measurement could not be taken because the environment cannot support
# it. That is distinct from a measurement that ran and produced numbers.
EXIT_UNAVAILABLE = 2

KERNEL_MODULE = "b12x.moe._shared.kernels.w4a16.mixed_trellis"
PREPARE_MODULE = "b12x.moe._shared.kernels.w4a16.prepare"
HOST_MODULE = "b12x.moe._shared.kernels.w4a16.host"
KERNEL_MODULE_PATH = (
    "/opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/w4a16/"
    "mixed_trellis.py"
)

FLOP_FORMULA = (
    "dense_equivalent_flop = 2 * (size_m * top_k) * (2 * hidden_size * "
    "intermediate_size + intermediate_size * hidden_size) = 6 * size_m * "
    "top_k * hidden_size * intermediate_size"
)
FLOP_NOTE = (
    "dense-equivalent: the arithmetic a dense BF16 GEMM would perform for the "
    "same routed work, comparable with a dense GEMM rate. Not a count of the "
    "operations this kernel issues; it also decodes Trellis codes and applies "
    "rotations, and does no arithmetic for unselected experts."
)
WEIGHT_BYTES_FORMULA = (
    "weight_stream_bytes = sum over selected experts of ceil(3 * hidden_size "
    "* intermediate_size * tier_bits / 8)"
)
WEIGHT_BYTES_NOTE = (
    "compressed bytes at each tier's Trellis bit width, counted once per "
    "expert the routing selected. A kernel that re-reads an expert across "
    "m-blocks moves more, so this is a lower bound on traffic and the rate "
    "derived from it is a lower bound on the achieved rate."
)

# Weight coefficients an expert holds: gate and up, each hidden_size by
# intermediate_size, plus down, intermediate_size by hidden_size.
PROJECTIONS_PER_EXPERT = 3

# Matrix products a routed token passes through: gate, up, and down.
MATMULS_PER_ROUTED_TOKEN = 3

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


class MeasurementUnavailable(RuntimeError):
    """The environment cannot support the measurement, so none was taken."""


# --------------------------------------------------------------------------
# Geometry and configuration. Pure; exercised without a GPU.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """The kernel's problem size, independent of how it is tiled."""

    hidden_size: int = 6144
    intermediate_size: int = 512
    tier0_num_experts: int = 192
    tier1_num_experts: int = 64
    tier0_bits: int = 3
    tier1_bits: int = 4
    top_k: int = 8
    sms: int = 48

    @property
    def total_experts(self) -> int:
        return self.tier0_num_experts + self.tier1_num_experts

    def as_dict(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "tier0_num_experts": self.tier0_num_experts,
            "tier1_num_experts": self.tier1_num_experts,
            "tier0_bits": self.tier0_bits,
            "tier1_bits": self.tier1_bits,
            "top_k": self.top_k,
            "total_experts": self.total_experts,
            "sms": self.sms,
        }


DEPLOYED_GEOMETRY = Geometry()

# The four-tuple is (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n). The vLLM
# EXL3 backend selects it from hidden_size alone: a hidden size divisible by
# 512 gives (128, 128, 32, 512), one divisible by 256 gives (128, 128, 64,
# 256), and any other gives (128, 128, 128, 128). Hidden size 6144 is
# divisible by 512, so the deployed value is the first of those.
DEPLOYED_TILE_CONFIG: tuple[int, int, int, int] = (128, 128, 32, 512)
DEPLOYED_MOE_BLOCK_SIZE = 8

# The tile geometry the single-bitrate rank-sliced path uses. The backend
# documents it as losing partial reductions at large prefill token counts, so
# it is swept only as a labelled control, never as a recommendation.
FC1_CONTROL_TILE_CONFIG: tuple[int, int, int, int] = (64, 256, 64, 256)

DEPLOYED_SIZES_M: tuple[int, ...] = (40, 128, 512)
# Token capacity the deployed decode path plans for.
DECODE_SIZE_M = 40

# The FC1 half of the tile config is held at (128, 128) by default: the mixed
# three-and-four-bit megakernel requires a 128-wide FC1 K tile. The FC2 half is
# the swept axis, because the 512-wide FC2 tile is the deployed choice and the
# claim behind it, that it removes a second persistent wave, is a claim about
# wave quantization against the device's 48 streaming multiprocessors.
DEFAULT_FC1_K_VALUES: tuple[int, ...] = (128,)
DEFAULT_FC1_N_VALUES: tuple[int, ...] = (128,)
DEFAULT_FC2_K_VALUES: tuple[int, ...] = (32, 64, 128)
DEFAULT_FC2_N_VALUES: tuple[int, ...] = (128, 256, 512)
DEFAULT_MOE_BLOCK_SIZES: tuple[int, ...] = (8, 16)
DEFAULT_CONFIGURATION_CAP = 32

ROLE_BASELINE = "baseline"
ROLE_CANDIDATE = "candidate"
ROLE_CONTROL = "control"

CONTROL_NOTE = (
    "labelled control: a 64-wide FC1 K tile is documented in the vLLM EXL3 "
    "backend as losing partial reductions at large token counts, so it is "
    "measured for reference and not offered as a candidate"
)


@dataclass(frozen=True)
class Configuration:
    """One tiling of the kernel at a fixed geometry."""

    tile_config: tuple[int, int, int, int]
    moe_block_size: int
    role: str = ROLE_CANDIDATE

    @property
    def name(self) -> str:
        tiles = "x".join(str(value) for value in self.tile_config)
        return f"tile{tiles}_block{self.moe_block_size}"

    @property
    def fc1_tile_n(self) -> int:
        """The FC1 output tile width the weight preparation must agree with."""

        return self.tile_config[1]

    @property
    def fc2_tile_n(self) -> int:
        """The FC2 output tile width the weight preparation must agree with."""

        return self.tile_config[3]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tile_config": list(self.tile_config),
            "moe_block_size": self.moe_block_size,
            "role": self.role,
            "fc1_tile_n": self.fc1_tile_n,
            "fc2_tile_n": self.fc2_tile_n,
        }


DEPLOYED_CONFIGURATION = Configuration(
    DEPLOYED_TILE_CONFIG, DEPLOYED_MOE_BLOCK_SIZE, ROLE_BASELINE
)


def enumerate_configurations(
    *,
    baseline: Configuration = DEPLOYED_CONFIGURATION,
    fc1_k_values: Sequence[int] = DEFAULT_FC1_K_VALUES,
    fc1_n_values: Sequence[int] = DEFAULT_FC1_N_VALUES,
    fc2_k_values: Sequence[int] = DEFAULT_FC2_K_VALUES,
    fc2_n_values: Sequence[int] = DEFAULT_FC2_N_VALUES,
    moe_block_sizes: Sequence[int] = DEFAULT_MOE_BLOCK_SIZES,
    controls: Sequence[tuple[int, int, int, int]] = (),
    cap: int = DEFAULT_CONFIGURATION_CAP,
) -> tuple[list[Configuration], dict[str, Any]]:
    """The sweep space, baseline first, deduplicated and capped.

    Configurations are grouped by tile config so that one prepared weight
    pool serves every route block size measured against it. The baseline
    survives the cap unconditionally: a ranking with no baseline states
    nothing.
    """

    if cap < 1:
        raise ValueError(f"the configuration cap must be at least 1; got {cap}")

    grouped: dict[tuple[int, int, int, int], list[Configuration]] = {
        baseline.tile_config: [baseline]
    }
    order: list[tuple[int, int, int, int]] = [baseline.tile_config]
    seen = {(baseline.tile_config, baseline.moe_block_size)}

    def add(tile: tuple[int, int, int, int], block: int, role: str) -> None:
        key = (tile, block)
        if key in seen:
            return
        seen.add(key)
        if tile not in grouped:
            grouped[tile] = []
            order.append(tile)
        grouped[tile].append(Configuration(tile, block, role))

    for fc1_k in fc1_k_values:
        for fc1_n in fc1_n_values:
            for fc2_k in fc2_k_values:
                for fc2_n in fc2_n_values:
                    for block in moe_block_sizes:
                        add((fc1_k, fc1_n, fc2_k, fc2_n), block, ROLE_CANDIDATE)
    for control in controls:
        tile = (control[0], control[1], control[2], control[3])
        for block in moe_block_sizes:
            add(tile, block, ROLE_CONTROL)

    enumerated = [configuration for tile in order for configuration in grouped[tile]]
    kept = enumerated[:cap]
    return kept, {
        "enumerated": len(enumerated),
        "cap": cap,
        "measured": len(kept),
        "dropped": len(enumerated) - len(kept),
        "truncated": len(enumerated) > cap,
    }


def tile_config_groups(
    configurations: Sequence[Configuration],
) -> list[tuple[tuple[int, int, int, int], list[Configuration]]]:
    """Configurations grouped by tile config, in first-appearance order."""

    groups: dict[tuple[int, int, int, int], list[Configuration]] = {}
    order: list[tuple[int, int, int, int]] = []
    for configuration in configurations:
        if configuration.tile_config not in groups:
            groups[configuration.tile_config] = []
            order.append(configuration.tile_config)
        groups[configuration.tile_config].append(configuration)
    return [(tile, groups[tile]) for tile in order]


# --------------------------------------------------------------------------
# Arithmetic. Pure; exercised without a GPU.
# --------------------------------------------------------------------------


def dense_equivalent_flop(geometry: Geometry, size_m: int) -> int:
    """Flops a dense BF16 GEMM would perform for the same routed work."""

    if size_m < 1:
        raise ValueError(f"size_m must be at least 1; got {size_m}")
    routed_tokens = size_m * geometry.top_k
    return (
        2
        * routed_tokens
        * MATMULS_PER_ROUTED_TOKEN
        * geometry.hidden_size
        * geometry.intermediate_size
    )


def expert_compressed_bytes(geometry: Geometry, bits: int) -> int:
    """Compressed bytes one expert's gate, up, and down weights occupy."""

    if bits < 1:
        raise ValueError(f"a tier bit width must be at least 1; got {bits}")
    coefficients = (
        PROJECTIONS_PER_EXPERT * geometry.hidden_size * geometry.intermediate_size
    )
    return (coefficients * bits + 7) // 8


def weight_stream_bytes(
    geometry: Geometry, *, tier0_experts: int, tier1_experts: int
) -> int:
    """Compressed bytes for the experts a routing selected, both tiers."""

    for count, limit, label in (
        (tier0_experts, geometry.tier0_num_experts, "tier0"),
        (tier1_experts, geometry.tier1_num_experts, "tier1"),
    ):
        if not 0 <= count <= limit:
            raise ValueError(
                f"{label} selected expert count {count} lies outside [0, {limit}]"
            )
    return tier0_experts * expert_compressed_bytes(
        geometry, geometry.tier0_bits
    ) + tier1_experts * expert_compressed_bytes(geometry, geometry.tier1_bits)


def resident_weight_bytes(geometry: Geometry) -> int:
    """Compressed bytes one complete prepared weight set occupies."""

    return weight_stream_bytes(
        geometry,
        tier0_experts=geometry.tier0_num_experts,
        tier1_experts=geometry.tier1_num_experts,
    )


def weight_pool_bytes(geometry: Geometry, pool_size: int) -> int:
    """Compressed bytes a weight pool of `pool_size` sets occupies.

    Packed expert weights only. Rotations, launch descriptors, and the
    kernel's device buffers are additional and are not counted here.
    """

    if pool_size < 1:
        raise ValueError(f"the weight pool must hold at least 1 set; got {pool_size}")
    return pool_size * resident_weight_bytes(geometry)


def count_selected_experts(
    route_ids: Iterable[int], tier0_num_experts: int, total_experts: int
) -> tuple[int, int]:
    """Distinct experts a routing selects, split at the tier boundary.

    Expert identifiers below `tier0_num_experts` belong to tier 0 and the
    rest to tier 1, which is the assignment this harness gives its
    synthetic tiers.
    """

    distinct = set()
    for identifier in route_ids:
        value = int(identifier)
        if not 0 <= value < total_experts:
            raise ValueError(
                f"route identifier {value} lies outside [0, {total_experts})"
            )
        distinct.add(value)
    tier0 = sum(1 for value in distinct if value < tier0_num_experts)
    return tier0, len(distinct) - tier0


def tflops(flop: int, milliseconds: float) -> float:
    """Achieved rate in TFLOP/s for `flop` completed in `milliseconds`."""

    if milliseconds <= 0:
        raise ValueError(f"elapsed time must be positive; got {milliseconds}")
    return flop / (milliseconds * 1e-3) / 1e12


def gigabytes_per_second(byte_count: int, milliseconds: float) -> float:
    """Achieved traffic rate in GB/s, 10^9 bytes per second."""

    if milliseconds <= 0:
        raise ValueError(f"elapsed time must be positive; got {milliseconds}")
    return byte_count / (milliseconds * 1e-3) / 1e9


def route_slot_plan(
    api: Any, geometry: Geometry, *, size_m: int, moe_block_size: int
) -> dict[str, int]:
    """Packed route slots and the m-block count the compiler is given.

    `max_m_blocks` is ceil(route_slots / moe_block_size), which is how the
    vLLM EXL3 backend derives it from `max_packed_route_slots`.
    """

    if moe_block_size < 1:
        raise ValueError(f"moe_block_size must be at least 1; got {moe_block_size}")
    route_slots = int(
        api.max_packed_route_slots(
            size_m * geometry.top_k, moe_block_size, geometry.total_experts
        )
    )
    if route_slots < 1:
        raise MeasurementUnavailable(
            f"max_packed_route_slots returned {route_slots} for size_m="
            f"{size_m}, block={moe_block_size}, experts="
            f"{geometry.total_experts}; a launch cannot be planned from it"
        )
    return {
        "route_slots": route_slots,
        "max_m_blocks": (route_slots + moe_block_size - 1) // moe_block_size,
    }


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


def size_result(
    *,
    geometry: Geometry,
    size_m: int,
    configuration: Configuration,
    samples_ms: Sequence[float],
    selected_tier0: int,
    selected_tier1: int,
    plan: dict[str, int],
) -> dict[str, Any]:
    """One token count's timing distribution and the rates implied by it."""

    timing = summarize(samples_ms)
    flop = dense_equivalent_flop(geometry, size_m)
    traffic = weight_stream_bytes(
        geometry, tier0_experts=selected_tier0, tier1_experts=selected_tier1
    )
    tier0_traffic = selected_tier0 * expert_compressed_bytes(
        geometry, geometry.tier0_bits
    )
    return {
        "size_m": size_m,
        "configuration": configuration.as_dict(),
        "routing": {
            "route_slots_planned": plan["route_slots"],
            "max_m_blocks": plan["max_m_blocks"],
            "selected_tier0_experts": selected_tier0,
            "selected_tier1_experts": selected_tier1,
            "selected_experts": selected_tier0 + selected_tier1,
            "route_slots_drawn": size_m * geometry.top_k,
        },
        "dense_equivalent_flop": flop,
        "weight_stream_bytes": traffic,
        "weight_stream_bytes_tier0": tier0_traffic,
        "weight_stream_bytes_tier1": traffic - tier0_traffic,
        "timing": timing,
        "rate": {
            "dense_equivalent_tflops_at_median": tflops(flop, timing["median_ms"]),
            "dense_equivalent_tflops_at_min": tflops(flop, timing["min_ms"]),
            "weight_stream_gigabytes_per_second_at_median": gigabytes_per_second(
                traffic, timing["median_ms"]
            ),
            "weight_stream_gigabytes_per_second_at_min": gigabytes_per_second(
                traffic, timing["min_ms"]
            ),
        },
    }


def measured_entry(result: dict[str, Any]) -> dict[str, Any]:
    """A sweep entry for a configuration that produced timings."""

    return {"status": "measured", **result}


def skipped_entry(
    configuration: Configuration, error: BaseException, stage: str
) -> dict[str, Any]:
    """A sweep entry for a configuration that could not be measured.

    The exception type is recorded rather than raised: an invalid tile
    config is a fact about that configuration, not a reason to abandon the
    remaining ones.
    """

    return {
        "status": "skipped",
        "configuration": configuration.as_dict(),
        "stage": stage,
        "exception_type": type(error).__name__,
        "reason": str(error) or repr(error),
    }


def rank_configurations(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Rank measured configurations by median against the baseline entry."""

    measured = [entry for entry in entries if entry["status"] == "measured"]
    baseline = next(
        (
            entry
            for entry in measured
            if entry["configuration"]["role"] == ROLE_BASELINE
        ),
        None,
    )
    if baseline is None:
        raise ValueError(
            "the ranking needs a measured baseline configuration to compare against"
        )
    baseline_median = baseline["timing"]["median_ms"]
    ranked = []
    for entry in sorted(measured, key=lambda item: item["timing"]["median_ms"]):
        ranked.append(
            {
                "name": entry["configuration"]["name"],
                "tile_config": entry["configuration"]["tile_config"],
                "moe_block_size": entry["configuration"]["moe_block_size"],
                "role": entry["configuration"]["role"],
                "median_ms": entry["timing"]["median_ms"],
                "iqr_ms": entry["timing"]["iqr_ms"],
                "speedup_over_baseline": baseline_median / entry["timing"]["median_ms"],
            }
        )
    candidates = [row for row in ranked if row["role"] != ROLE_BASELINE]
    faster = [row for row in candidates if row["speedup_over_baseline"] > 1.0]
    best = max(candidates, key=lambda row: row["speedup_over_baseline"], default=None)
    if best is None:
        verdict = (
            "only the baseline was measured, so the sweep states nothing about "
            "alternatives"
        )
    elif not faster:
        verdict = (
            "no swept configuration beat the baseline; the fastest alternative "
            f"({best['name']}) reached {best['speedup_over_baseline']:.3f} "
            "times the baseline rate"
        )
    else:
        verdict = (
            f"{len(faster)} configuration(s) beat the baseline; the fastest "
            f"({best['name']}) reached {best['speedup_over_baseline']:.3f} "
            "times the baseline rate"
        )
    return {
        "baseline": {
            "name": baseline["configuration"]["name"],
            "median_ms": baseline_median,
            "iqr_ms": baseline["timing"]["iqr_ms"],
        },
        "ordered": ranked,
        "faster_than_baseline": len(faster),
        "skipped": sum(1 for entry in entries if entry["status"] == "skipped"),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# Signature discovery. Pure over the recorded signature; exercised without a
# GPU or the kernel package.
# --------------------------------------------------------------------------


def signature_record(function: Any, name: str) -> dict[str, Any]:
    """Positional and keyword parameters of `function`, as a record."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        raise MeasurementUnavailable(
            f"the signature of {name} could not be read, so no call to it can "
            f"be checked before it is made: {error!r}"
        ) from error
    positional: list[str] = []
    keywords: list[str] = []
    var_positional = False
    var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional.append(parameter.name)
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                keywords.append(parameter.name)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keywords.append(parameter.name)
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword = True
    return {
        "name": name,
        "text": f"{name}{signature}",
        "positional": positional,
        "keywords": keywords,
        "var_positional": var_positional,
        "var_keyword": var_keyword,
    }


def require_positional_order(record: dict[str, Any], expected: Sequence[str]) -> None:
    """Refuse unless the leading positional parameters are exactly `expected`."""

    found = list(record["positional"])
    if found[: len(expected)] != list(expected):
        raise MeasurementUnavailable(
            f"{record['name']} does not take the positional parameters this "
            f"harness would pass. Expected the first {len(expected)} to be "
            f"{list(expected)}; found {record['text']}. Argument order is not "
            "guessed: correct the harness against the resolved implementation "
            "before measuring."
        )


def require_positional_count(record: dict[str, Any], count: int) -> None:
    """Refuse unless at least `count` positional parameters are accepted."""

    if record["var_positional"]:
        return
    if len(record["positional"]) < count:
        raise MeasurementUnavailable(
            f"{record['name']} accepts {len(record['positional'])} positional "
            f"parameter(s) where this harness passes {count}; found "
            f"{record['text']}"
        )


def require_keywords(record: dict[str, Any], expected: Sequence[str]) -> None:
    """Refuse unless every keyword this harness passes is accepted by name."""

    if record["var_keyword"]:
        return
    missing = [name for name in expected if name not in record["keywords"]]
    if missing:
        raise MeasurementUnavailable(
            f"{record['name']} does not accept the keyword argument(s) "
            f"{missing} this harness passes; found {record['text']}"
        )


def module_path_record(
    module: Any,
    expected_path: str,
    *,
    realpath: Callable[[str], str] = os.path.realpath,
) -> dict[str, Any]:
    """Compare a module's resolved file against the path the report names."""

    resolved = getattr(module, "__file__", None)
    if resolved is None:
        return {
            "expected": expected_path,
            "resolved": None,
            "resolved_realpath": None,
            "matches": False,
            "reason": (
                "the imported module has no __file__, so the implementation "
                "being timed cannot be identified"
            ),
        }
    resolved = str(resolved)
    candidates = {resolved.replace("\\", "/"), realpath(resolved).replace("\\", "/")}
    expected = {
        expected_path.replace("\\", "/"),
        realpath(expected_path).replace("\\", "/"),
    }
    matches = bool(candidates & expected)
    return {
        "expected": expected_path,
        "resolved": resolved,
        "resolved_realpath": realpath(resolved),
        "matches": matches,
        "reason": (
            ""
            if matches
            else (
                "the module bound to this name resolves to a different file "
                "than the report names. An import hook can rebind a module "
                "path, and a measurement that names one implementation while "
                "timing another states nothing about either."
            )
        ),
    }


EXPECTED_RUN_POSITIONAL = (
    "x",
    "tier0",
    "tier1",
    "topk_weights",
    "topk_ids",
    "global_to_combined",
    "descriptor_map",
    "rotations",
    "launch",
    "buffers",
)
EXPECTED_COMPILE_KEYWORDS = (
    "size_m",
    "hidden_size",
    "intermediate_size",
    "tier0_num_experts",
    "tier1_num_experts",
    "top_k",
    "max_m_blocks",
    "sms",
    "max_shared_mem",
    "force_tile_config",
    "tier0_bits",
    "tier1_bits",
    "trellis_codebook",
    "moe_block_size",
    "rotation_input_dtype",
    "route_ids_dtype",
)
EXPECTED_BUFFERS_KEYWORDS = ("device", "sms")
EXPECTED_PREPARE_KEYWORDS = (
    "w13",
    "w2",
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "activation",
    "fc1_tile_n",
    "fc2_tile_n",
    "device",
    "seed",
    "params_dtype",
    "trellis_bits",
    "codebook",
    "tile_config",
)
# The rotations a tier carries, paired as (field of MixedTrellisRotations,
# attribute of a prepared tier).
ROTATION_FIELDS = (
    ("intermediate", "intermediate_rotations"),
    ("gate_suh", "gate_suh"),
    ("up_suh", "up_suh"),
    ("down_svh", "down_svh"),
)


def _attribute(module: Any, module_name: str, attribute: str) -> Any:
    value = getattr(module, attribute, None)
    if value is None:
        exported = sorted(name for name in dir(module) if not name.startswith("_"))[:40]
        raise MeasurementUnavailable(
            f"{module_name} does not provide {attribute!r}, which this harness "
            f"calls. It exposes {exported}"
        )
    return value


def resolve_api(
    kernel_module: Any, prepare_module: Any, host_module: Any
) -> tuple[Any, dict[str, Any]]:
    """Bind the kernel entry points and check every signature relied on.

    Nothing is called before its signature is confirmed, and the confirmed
    signatures travel with the report so a reader can see what was bound.
    """

    build_tiered_maps = _attribute(kernel_module, KERNEL_MODULE, "build_tiered_maps")
    compile_mixed_trellis = _attribute(
        kernel_module, KERNEL_MODULE, "compile_mixed_trellis"
    )
    make_buffers = _attribute(
        kernel_module, KERNEL_MODULE, "make_mixed_trellis_buffers"
    )
    run_mixed_trellis = _attribute(kernel_module, KERNEL_MODULE, "run_mixed_trellis")
    prepare_weights = _attribute(
        prepare_module, PREPARE_MODULE, "prepare_trellis256_moe_weights"
    )
    max_packed_route_slots = _attribute(
        host_module, HOST_MODULE, "max_packed_route_slots"
    )
    rotations_class = _attribute(
        kernel_module, KERNEL_MODULE, "MixedTrellisRotations"
    )
    combine_rotations = getattr(kernel_module, "combine_trellis_rotations", None)

    records = {
        "build_tiered_maps": signature_record(build_tiered_maps, "build_tiered_maps"),
        "compile_mixed_trellis": signature_record(
            compile_mixed_trellis, "compile_mixed_trellis"
        ),
        "make_mixed_trellis_buffers": signature_record(
            make_buffers, "make_mixed_trellis_buffers"
        ),
        "run_mixed_trellis": signature_record(run_mixed_trellis, "run_mixed_trellis"),
        "prepare_trellis256_moe_weights": signature_record(
            prepare_weights, "prepare_trellis256_moe_weights"
        ),
        "max_packed_route_slots": signature_record(
            max_packed_route_slots, "max_packed_route_slots"
        ),
    }
    if combine_rotations is not None:
        records["combine_trellis_rotations"] = signature_record(
            combine_rotations, "combine_trellis_rotations"
        )

    # `build_tiered_maps` is passed the tier-0 and tier-1 expert identifier
    # sequences positionally and the device by keyword, and returns the pair
    # (global_to_combined, descriptor_map) in that order.
    require_positional_count(records["build_tiered_maps"], 2)
    require_keywords(records["build_tiered_maps"], ("device",))
    require_keywords(records["compile_mixed_trellis"], EXPECTED_COMPILE_KEYWORDS)
    require_positional_count(records["make_mixed_trellis_buffers"], 1)
    require_keywords(records["make_mixed_trellis_buffers"], EXPECTED_BUFFERS_KEYWORDS)
    require_positional_order(records["run_mixed_trellis"], EXPECTED_RUN_POSITIONAL)
    require_keywords(
        records["prepare_trellis256_moe_weights"], EXPECTED_PREPARE_KEYWORDS
    )
    # max_packed_route_slots(route_slot_count, block_size, num_experts).
    require_positional_count(records["max_packed_route_slots"], 3)

    api = SimpleNamespace(
        build_tiered_maps=build_tiered_maps,
        compile_mixed_trellis=compile_mixed_trellis,
        make_mixed_trellis_buffers=make_buffers,
        run_mixed_trellis=run_mixed_trellis,
        prepare_weights=prepare_weights,
        max_packed_route_slots=max_packed_route_slots,
        combine_rotations=combine_rotations,
        rotations_class=rotations_class,
    )
    discovery = {
        "kernel_module": KERNEL_MODULE,
        "prepare_module": PREPARE_MODULE,
        "host_module": HOST_MODULE,
        "signatures": records,
        "rotations_source": (
            "combine_trellis_rotations"
            if combine_rotations is not None
            else "MixedTrellisRotations built from concatenated tier rotations"
        ),
    }
    return api, discovery


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
    return {
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
        "device_index": device_index,
        "device_name": torch_module.cuda.get_device_name(device_index),
        "compute_capability": list(capability),
        "multi_processor_count": int(getattr(properties, "multi_processor_count", 0)),
        "shared_memory_per_block_optin": int(
            getattr(properties, "shared_memory_per_block_optin", 0)
        ),
        "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
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
            f"torch is not importable, so no kernel can be launched: {error}"
        ) from error
    return torch


def import_kernel_modules(
    *,
    kernel_name: str = KERNEL_MODULE,
    prepare_name: str = PREPARE_MODULE,
    host_name: str = HOST_MODULE,
    importer: Callable[[str], Any] = importlib.import_module,
) -> tuple[Any, Any, Any]:
    """Import the kernel, weight preparation, and host planning modules."""

    modules = []
    for name in (kernel_name, prepare_name, host_name):
        try:
            modules.append(importer(name))
        except Exception as error:  # noqa: BLE001 - the reason is reported
            raise MeasurementUnavailable(
                f"{name} is not importable, so the mixed-bitrate Trellis "
                f"kernel cannot be measured: {type(error).__name__}: {error}"
            ) from error
    return modules[0], modules[1], modules[2]


def require_cuda_device(torch_module: Any, device_index: int) -> None:
    """Refuse, with a reason, unless the named CUDA device exists."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise MeasurementUnavailable(
            "no CUDA device is visible to torch. This measures a compiled "
            "GB10 kernel and has no CPU fallback, because there is no CPU "
            "implementation of it to time. Run it on the GB10 host, inside "
            "the serving runtime image, with the device visible to the "
            "container."
        )
    count = int(cuda.device_count())
    if not 0 <= device_index < count:
        raise MeasurementUnavailable(
            f"--device {device_index} does not exist; torch sees {count} CUDA "
            "device(s)"
        )


def require_pool_fits(
    geometry: Geometry, pool_size: int, total_memory_bytes: int, fraction: float
) -> dict[str, Any]:
    """Refuse a weight pool that would claim too much of device memory."""

    predicted = weight_pool_bytes(geometry, pool_size)
    limit = int(total_memory_bytes * fraction)
    if total_memory_bytes and predicted > limit:
        raise MeasurementUnavailable(
            f"a weight pool of {pool_size} set(s) needs about "
            f"{predicted / (1 << 30):.1f} GiB of packed expert weights, above "
            f"the {fraction:.0%} of the device's "
            f"{total_memory_bytes / (1 << 30):.1f} GiB this harness will "
            "claim. Lower --pool-size, raise --max-pool-fraction, or run on a "
            "device with more memory. GB10 memory is unified, so a pool that "
            "fills it starves the host as well."
        )
    return {
        "pool_size": pool_size,
        "bytes_per_set": resident_weight_bytes(geometry),
        "predicted_bytes": predicted,
        "fraction_of_device_memory": (
            predicted / total_memory_bytes if total_memory_bytes else None
        ),
        "note": (
            "packed expert weights only; rotations, launch descriptors, and "
            "kernel buffers are additional"
        ),
    }


def build_routing(
    torch_module: Any,
    geometry: Geometry,
    *,
    size_m: int,
    device: Any,
    seed: int,
) -> tuple[Any, Any, tuple[int, int]]:
    """Deterministic top-k routing over every expert, and what it selects.

    Each token draws `top_k` distinct experts uniformly, which is the
    routing shape the kernel is given. A deployed router's distribution is
    skewed, so a measured elapsed time under this routing describes the
    geometry rather than a particular model's routing behaviour.
    """

    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed)
    scores = torch_module.rand(
        size_m,
        geometry.total_experts,
        generator=generator,
        dtype=torch_module.float32,
    )
    _, ids = torch_module.topk(scores, geometry.top_k, dim=1)
    weights = torch_module.rand(
        size_m, geometry.top_k, generator=generator, dtype=torch_module.float32
    )
    weights = weights / weights.sum(dim=1, keepdim=True)
    selected = count_selected_experts(
        ids.reshape(-1).tolist(), geometry.tier0_num_experts, geometry.total_experts
    )
    return (
        ids.to(device=device, dtype=torch_module.int32).contiguous(),
        weights.to(device=device).contiguous(),
        selected,
    )


def build_rotations(torch_module: Any, api: Any, tiers: Sequence[Any]) -> Any:
    """Rotations covering both tiers, in combined expert order."""

    if api.combine_rotations is not None:
        return api.combine_rotations(*tiers)
    fields = {}
    for field, attribute in ROTATION_FIELDS:
        parts = []
        for tier in tiers:
            value = getattr(tier, attribute, None)
            if value is None:
                raise MeasurementUnavailable(
                    f"a prepared tier does not carry {attribute!r}, so the "
                    "rotations this kernel needs cannot be assembled, and the "
                    "module exposes no combine_trellis_rotations to assemble "
                    "them"
                )
            parts.append(value)
        fields[field] = torch_module.cat(parts, dim=0).contiguous()
    return api.rotations_class(**fields)


def prepare_weight_pool(
    torch_module: Any,
    api: Any,
    geometry: Geometry,
    *,
    tile_config: tuple[int, int, int, int],
    device: Any,
    pool_size: int,
    seed: int,
    params_dtype: Any,
    activation: str,
) -> list[tuple[Any, Any, Any]]:
    """Independent prepared weight sets, one per timed call in the cycle.

    Weight preparation is told the same FC1 and FC2 output tile widths the
    launch is compiled with: the vLLM EXL3 backend passes
    `fc1_tile_n=tile_config[1]` and `fc2_tile_n=tile_config[3]`, so packed
    weights belong to a tile config and a pool cannot be shared across
    them.
    """

    pool = []
    for index in range(pool_size):
        tiers = []
        for tier, (experts, bits) in enumerate(
            (
                (geometry.tier0_num_experts, geometry.tier0_bits),
                (geometry.tier1_num_experts, geometry.tier1_bits),
            )
        ):
            tiers.append(
                api.prepare_weights(
                    w13=None,
                    w2=None,
                    hidden_size=geometry.hidden_size,
                    intermediate_size=geometry.intermediate_size,
                    num_experts=experts,
                    activation=activation,
                    fc1_tile_n=tile_config[1],
                    fc2_tile_n=tile_config[3],
                    device=device,
                    seed=seed + 1000 * index + tier,
                    params_dtype=params_dtype,
                    trellis_bits=bits,
                    codebook="mcg",
                    tile_config=tile_config,
                )
            )
        pool.append((tiers[0], tiers[1], build_rotations(torch_module, api, tiers)))
    return pool


def compile_launch(
    api: Any,
    geometry: Geometry,
    configuration: Configuration,
    *,
    size_m: int,
    device: Any,
    max_shared_mem: int,
    rotation_input_dtype: str,
    route_ids_dtype: Any,
) -> tuple[Any, Any, dict[str, int]]:
    """Compile one launch and allocate its device buffers."""

    plan = route_slot_plan(
        api, geometry, size_m=size_m, moe_block_size=configuration.moe_block_size
    )
    launch = api.compile_mixed_trellis(
        size_m=size_m,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        tier0_num_experts=geometry.tier0_num_experts,
        tier1_num_experts=geometry.tier1_num_experts,
        tier0_bits=geometry.tier0_bits,
        tier1_bits=geometry.tier1_bits,
        top_k=geometry.top_k,
        max_m_blocks=plan["max_m_blocks"],
        moe_block_size=configuration.moe_block_size,
        sms=geometry.sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=configuration.tile_config,
        trellis_codebook="mcg",
        rotation_input_dtype=rotation_input_dtype,
        route_ids_dtype=route_ids_dtype,
    )
    buffers = api.make_mixed_trellis_buffers(launch, device=device, sms=geometry.sms)
    return launch, buffers, plan


def time_calls(
    torch_module: Any,
    api: Any,
    *,
    x: Any,
    topk_weights: Any,
    topk_ids: Any,
    global_to_combined: Any,
    descriptor_map: Any,
    pool: Sequence[tuple[Any, Any, Any]],
    launch: Any,
    buffers: Any,
    device: Any,
    warmup: int,
    iterations: int,
    label: str,
) -> list[float]:
    """Elapsed milliseconds per timed call, cycling the weight pool.

    Every reusable tensor is allocated by the caller. The kernel returns
    its output, so that allocation stays inside the loop; only the most
    recent output is retained, and it is checked for a finite, non-zero
    result after the loop.
    """

    def call(entry: tuple[Any, Any, Any]) -> Any:
        tier0, tier1, rotations = entry
        return api.run_mixed_trellis(
            x,
            tier0,
            tier1,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor_map,
            rotations,
            launch,
            buffers,
        )

    # Warmup absorbs the kernel's compilation, which the first call to a
    # configuration pays, and touches every weight set in the pool.
    for index in range(warmup):
        call(pool[index % len(pool)])
    torch_module.cuda.synchronize(device)

    starts = [torch_module.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch_module.cuda.Event(enable_timing=True) for _ in range(iterations)]
    output = None
    for index, (start, end) in enumerate(zip(starts, ends)):
        entry = pool[index % len(pool)]
        start.record()
        output = call(entry)
        end.record()
    torch_module.cuda.synchronize(device)

    if output is None:
        raise MeasurementUnavailable(
            f"{label} ran no timed call, so there is nothing to report"
        )
    if not bool(torch_module.isfinite(output).all().item()):
        raise MeasurementUnavailable(
            f"{label} produced non-finite output, so its timing is not a "
            "measurement of a completed kernel"
        )
    if not bool((output != 0).any().item()):
        raise MeasurementUnavailable(
            f"{label} produced an all-zero output, so its timing is not a "
            "measurement of completed work"
        )
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


def measure_size(
    torch_module: Any,
    api: Any,
    geometry: Geometry,
    configuration: Configuration,
    *,
    size_m: int,
    device: Any,
    pool: Sequence[tuple[Any, Any, Any]],
    maps: tuple[Any, Any],
    max_shared_mem: int,
    dtype_name: str,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Compile, drive, and summarize one token count of one configuration."""

    torch_dtype = (
        torch_module.bfloat16 if dtype_name == "bf16" else torch_module.float16
    )
    topk_ids, topk_weights, selected = build_routing(
        torch_module, geometry, size_m=size_m, device=device, seed=seed
    )
    x = torch_module.randn(
        size_m, geometry.hidden_size, device=device, dtype=torch_module.float32
    )
    x = (x * 1.0e-2).to(torch_dtype).contiguous()
    launch, buffers, plan = compile_launch(
        api,
        geometry,
        configuration,
        size_m=size_m,
        device=device,
        max_shared_mem=max_shared_mem,
        rotation_input_dtype=dtype_name,
        route_ids_dtype=topk_ids.dtype,
    )
    samples = time_calls(
        torch_module,
        api,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_to_combined=maps[0],
        descriptor_map=maps[1],
        pool=pool,
        launch=launch,
        buffers=buffers,
        device=device,
        warmup=warmup,
        iterations=iterations,
        label=f"{configuration.name} at size_m={size_m}",
    )
    result = size_result(
        geometry=geometry,
        size_m=size_m,
        configuration=configuration,
        samples_ms=samples,
        selected_tier0=selected[0],
        selected_tier1=selected[1],
        plan=plan,
    )
    del launch, buffers, x, topk_ids, topk_weights
    torch_module.cuda.empty_cache()
    return result


def build_tier_maps(
    torch_module: Any, api: Any, geometry: Geometry, device: Any
) -> tuple[tuple[Any, Any], dict[str, Any]]:
    """Expert identifier maps for the two tiers, and how they were accepted.

    Expert identifiers 0 to tier0_num_experts-1 are assigned to tier 0 and
    the rest to tier 1. Which identifier sits in which tier is a property
    of a checkpoint's bitrate assignment; the kernel's cost depends on the
    tier sizes and the routing, so the assignment is stated here rather
    than recovered from a checkpoint.

    `build_tiered_maps` is passed the two identifier sequences positionally
    and the device by keyword. The container type its implementation
    accepts is not part of the documented interface, so a list of integers
    is tried first and a 64-bit integer tensor second; whichever form was
    accepted is recorded.
    """

    tier0_ids = list(range(geometry.tier0_num_experts))
    tier1_ids = list(range(geometry.tier0_num_experts, geometry.total_experts))
    attempts: list[str] = []
    forms = (
        ("list[int]", tier0_ids, tier1_ids),
        (
            "torch.int64 tensor",
            torch_module.tensor(tier0_ids, dtype=torch_module.int64, device=device),
            torch_module.tensor(tier1_ids, dtype=torch_module.int64, device=device),
        ),
    )
    for form, first, second in forms:
        try:
            maps = api.build_tiered_maps(first, second, device=device)
        except Exception as error:  # noqa: BLE001 - the reason is reported
            attempts.append(f"{form}: {type(error).__name__}: {error}")
            continue
        if not isinstance(maps, tuple) or len(maps) != 2:
            raise MeasurementUnavailable(
                f"build_tiered_maps returned {type(maps).__name__} where the "
                "pair (global_to_combined, descriptor_map) was expected"
            )
        return (maps[0], maps[1]), {
            "identifier_form_accepted": form,
            "rejected_forms": attempts,
            "tier0_identifiers": [tier0_ids[0], tier0_ids[-1]],
            "tier1_identifiers": [tier1_ids[0], tier1_ids[-1]],
        }
    raise MeasurementUnavailable(
        "build_tiered_maps rejected every expert identifier container this "
        "harness knows how to pass: " + "; ".join(attempts)
    )


# --------------------------------------------------------------------------
# Drivers.
# --------------------------------------------------------------------------


def open_device(
    torch_module: Any, arguments: argparse.Namespace
) -> tuple[Any, dict[str, Any]]:
    """Claim the named CUDA device and describe it."""

    require_cuda_device(torch_module, arguments.device)
    device = torch_module.device("cuda", arguments.device)
    torch_module.cuda.set_device(device)
    return device, describe_environment(torch_module, arguments.device)


def run_measure(
    torch_module: Any,
    api: Any,
    *,
    geometry: Geometry,
    configuration: Configuration,
    sizes: Sequence[int],
    device: Any,
    environment: dict[str, Any],
    discovery: dict[str, Any],
    module_record: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Time one configuration across every requested token count."""

    pool_record = require_pool_fits(
        geometry,
        arguments.pool_size,
        environment["total_memory_bytes"],
        arguments.max_pool_fraction,
    )
    clocks_before = read_clock_state(arguments.device)
    maps, map_record = build_tier_maps(torch_module, api, geometry, device)
    torch_dtype = (
        torch_module.bfloat16 if arguments.dtype == "bf16" else torch_module.float16
    )
    pool = prepare_weight_pool(
        torch_module,
        api,
        geometry,
        tile_config=configuration.tile_config,
        device=device,
        pool_size=arguments.pool_size,
        seed=arguments.seed,
        params_dtype=torch_dtype,
        activation=arguments.activation,
    )
    results = [
        measure_size(
            torch_module,
            api,
            geometry,
            configuration,
            size_m=size_m,
            device=device,
            pool=pool,
            maps=maps,
            max_shared_mem=environment["shared_memory_per_block_optin"],
            dtype_name=arguments.dtype,
            seed=arguments.seed,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
        )
        for size_m in sizes
    ]
    del pool
    torch_module.cuda.empty_cache()
    clocks_after = read_clock_state(arguments.device)
    return build_report(
        mode="measure",
        geometry=geometry,
        environment=environment,
        module_record=module_record,
        discovery=discovery,
        map_record=map_record,
        pool_record=pool_record,
        clocks_before=clocks_before,
        clocks_after=clocks_after,
        arguments=arguments,
        sizes=results,
        baseline=configuration,
    )


def run_tune(
    torch_module: Any,
    api: Any,
    *,
    geometry: Geometry,
    configurations: Sequence[Configuration],
    enumeration: dict[str, Any],
    size_m: int,
    device: Any,
    environment: dict[str, Any],
    discovery: dict[str, Any],
    module_record: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Time every configuration at one token count and rank the results.

    A candidate that fails at any stage is recorded as a skip carrying its
    exception type. The baseline is the only configuration whose failure
    ends the run, because a ranking with no baseline states nothing.
    """

    pool_record = require_pool_fits(
        geometry,
        arguments.pool_size,
        environment["total_memory_bytes"],
        arguments.max_pool_fraction,
    )
    clocks_before = read_clock_state(arguments.device)
    maps, map_record = build_tier_maps(torch_module, api, geometry, device)
    torch_dtype = (
        torch_module.bfloat16 if arguments.dtype == "bf16" else torch_module.float16
    )
    entries: list[dict[str, Any]] = []
    for tile_config, group in tile_config_groups(configurations):
        try:
            pool = prepare_weight_pool(
                torch_module,
                api,
                geometry,
                tile_config=tile_config,
                device=device,
                pool_size=arguments.pool_size,
                seed=arguments.seed,
                params_dtype=torch_dtype,
                activation=arguments.activation,
            )
        except Exception as error:  # noqa: BLE001 - recorded as a skip
            if any(item.role == ROLE_BASELINE for item in group):
                raise MeasurementUnavailable(
                    "weight preparation failed for the baseline tile config "
                    f"{tile_config}: {type(error).__name__}: {error}"
                ) from error
            entries.extend(
                skipped_entry(item, error, "prepare_weights") for item in group
            )
            torch_module.cuda.empty_cache()
            continue
        for configuration in group:
            try:
                entries.append(
                    measured_entry(
                        measure_size(
                            torch_module,
                            api,
                            geometry,
                            configuration,
                            size_m=size_m,
                            device=device,
                            pool=pool,
                            maps=maps,
                            max_shared_mem=environment[
                                "shared_memory_per_block_optin"
                            ],
                            dtype_name=arguments.dtype,
                            seed=arguments.seed,
                            warmup=arguments.warmup,
                            iterations=arguments.iterations,
                        )
                    )
                )
            except Exception as error:  # noqa: BLE001 - recorded as a skip
                if configuration.role == ROLE_BASELINE:
                    raise MeasurementUnavailable(
                        f"the baseline configuration {configuration.name} "
                        "could not be measured, so no ranking can be stated "
                        f"against it: {type(error).__name__}: {error}"
                    ) from error
                entries.append(skipped_entry(configuration, error, "measure"))
                torch_module.cuda.empty_cache()
        del pool
        torch_module.cuda.empty_cache()
    clocks_after = read_clock_state(arguments.device)
    return build_report(
        mode="tune",
        geometry=geometry,
        environment=environment,
        module_record=module_record,
        discovery=discovery,
        map_record=map_record,
        pool_record=pool_record,
        clocks_before=clocks_before,
        clocks_after=clocks_after,
        arguments=arguments,
        configurations=entries,
        enumeration=enumeration,
        ranking=rank_configurations(entries),
        tuned_size_m=size_m,
        baseline=next(
            (item for item in configurations if item.role == ROLE_BASELINE),
            DEPLOYED_CONFIGURATION,
        ),
    )


# --------------------------------------------------------------------------
# Report assembly and rendering. Pure; exercised without a GPU.
# --------------------------------------------------------------------------


def build_report(
    *,
    mode: str,
    geometry: Geometry,
    environment: dict[str, Any],
    module_record: dict[str, Any],
    discovery: dict[str, Any],
    map_record: dict[str, Any],
    pool_record: dict[str, Any],
    clocks_before: dict[str, Any],
    clocks_after: dict[str, Any],
    arguments: Any,
    sizes: Sequence[dict[str, Any]] = (),
    configurations: Sequence[dict[str, Any]] = (),
    enumeration: dict[str, Any] | None = None,
    ranking: dict[str, Any] | None = None,
    tuned_size_m: int | None = None,
    baseline: Configuration = DEPLOYED_CONFIGURATION,
) -> dict[str, Any]:
    """Assemble the machine-readable document."""

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "measurement": {
            "operation": (
                "one call of the fused mixed-bitrate Trellis mixture-of-"
                "experts kernel over two expert tiers behind a single grid"
            ),
            "weights": (
                "synthesized by prepare_trellis256_moe_weights with w13=None "
                "and w2=None; no checkpoint is loaded, so the result "
                "describes the kernel at a geometry and not a model's values"
            ),
            "flop_formula": FLOP_FORMULA,
            "flop_note": FLOP_NOTE,
            "weight_bytes_formula": WEIGHT_BYTES_FORMULA,
            "weight_bytes_note": WEIGHT_BYTES_NOTE,
            "timing": (
                "one CUDA event pair per call; the device is synchronized "
                "after warmup and after the timed loop, and event times are "
                "read only after that second synchronization"
            ),
            "statistic": (
                "median with interquartile range, plus observed min and max"
            ),
            "warmup_calls": arguments.warmup,
            "timed_calls": arguments.iterations,
            "weight_pool": pool_record,
            "weight_pool_note": (
                "one prepared weight set per timed call, round-robin, so "
                "weights stream rather than remaining resident in cache "
                "across calls"
            ),
            "activation": arguments.activation,
            "rotation_input_dtype": arguments.dtype,
            "route_ids_dtype": "torch.int32",
            "routing": (
                "each token draws top_k distinct experts uniformly from a "
                "seeded generator; a deployed router's distribution is skewed, "
                "so elapsed time here describes the geometry rather than a "
                "model's routing behaviour"
            ),
            "seed": arguments.seed,
            "single_process": True,
            "distributed": False,
        },
        "geometry": geometry.as_dict(),
        "module": module_record,
        "api_discovery": discovery,
        "tier_maps": map_record,
        "environment": environment,
        "clocks": {"before": clocks_before, "after": clocks_after},
    }
    if mode == "measure":
        report["configuration"] = baseline.as_dict() if not sizes else sizes[0][
            "configuration"
        ]
        report["sizes"] = list(sizes)
    else:
        report["tuned_size_m"] = tuned_size_m
        report["baseline_configuration"] = baseline.as_dict()
        report["enumeration"] = dict(enumeration or {})
        report["configurations"] = list(configurations)
        report["ranking"] = dict(ranking or {})
    return report


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


def _condition_lines(report: dict[str, Any]) -> list[str]:
    environment = report["environment"]
    measurement = report["measurement"]
    geometry = report["geometry"]
    module = report["module"]
    capability = "".join(str(part) for part in environment["compute_capability"])
    pool = measurement["weight_pool"]
    lines = [
        "CONDITIONS",
        f"  {'torch':<22}{environment['torch_version']} "
        f"(CUDA {environment['torch_cuda_version']})",
        f"  {'device':<22}{environment['device_name']} "
        f"(index {environment['device_index']}, SM{capability}, "
        f"{environment['multi_processor_count']} SMs)",
        f"  {'kernel module':<22}{module['resolved']}",
        f"  {'module path check':<22}"
        + ("matches the expected path" if module["matches"] else "MISMATCH"),
        f"  {'rotations from':<22}{report['api_discovery']['rotations_source']}",
        f"  {'tier ids passed as':<22}"
        f"{report['tier_maps']['identifier_form_accepted']}",
        f"  {'geometry':<22}hidden {geometry['hidden_size']}, intermediate "
        f"{geometry['intermediate_size']}, experts "
        f"{geometry['tier0_num_experts']}@{geometry['tier0_bits']}b + "
        f"{geometry['tier1_num_experts']}@{geometry['tier1_bits']}b, top_k "
        f"{geometry['top_k']}",
        f"  {'warmup calls':<22}{measurement['warmup_calls']}",
        f"  {'timed calls':<22}{measurement['timed_calls']}",
        f"  {'weight pool':<22}{pool['pool_size']} set(s), "
        f"{pool['predicted_bytes'] / (1 << 30):.2f} GiB of packed expert weights",
        f"  {'activation':<22}{measurement['activation']}",
        f"  {'timing':<22}{measurement['timing']}",
        f"  {'platform':<22}{environment['platform']}",
    ]
    lines += _clock_line("clocks before", report["clocks"]["before"])
    lines += _clock_line("clocks after", report["clocks"]["after"])
    before = report["clocks"]["before"]
    if before.get("read"):
        lines.append(f"  {'clock lock':<22}{before['lock_note']}")
    return lines


def _arithmetic_lines(report: dict[str, Any]) -> list[str]:
    measurement = report["measurement"]
    return [
        "",
        "ARITHMETIC",
        f"  {measurement['flop_formula']}",
        f"  {measurement['flop_note']}",
        f"  {measurement['weight_bytes_formula']}",
        f"  {measurement['weight_bytes_note']}",
    ]


def render_text(report: dict[str, Any]) -> str:
    """Render the report for a person reading a terminal."""

    if report["mode"] == "measure":
        return _render_measure(report)
    return _render_tune(report)


def _render_measure(report: dict[str, Any]) -> str:
    configuration = report["configuration"]
    lines = [
        "Mixed-bitrate Trellis MoE kernel rate, single GPU, single process",
        f"schema: {report['schema']}  mode: {report['mode']}",
        "",
    ]
    lines += _condition_lines(report)
    lines.append(
        f"  {'tile config':<22}{tuple(configuration['tile_config'])}, "
        f"moe_block_size {configuration['moe_block_size']}"
    )
    lines += _arithmetic_lines(report)
    lines += [
        "",
        "TOKEN COUNTS",
        f"  {'size_m':>7}{'median ms':>12}{'IQR ms':>10}{'min ms':>10}"
        f"{'max ms':>10}{'TFLOP/s med':>13}{'GB/s med':>11}{'experts':>9}"
        f"{'m_blocks':>10}",
    ]
    for entry in report["sizes"]:
        timing = entry["timing"]
        rate = entry["rate"]
        lines.append(
            f"  {entry['size_m']:>7}{timing['median_ms']:>12.4f}"
            f"{timing['iqr_ms']:>10.4f}{timing['min_ms']:>10.4f}"
            f"{timing['max_ms']:>10.4f}"
            f"{rate['dense_equivalent_tflops_at_median']:>13.2f}"
            f"{rate['weight_stream_gigabytes_per_second_at_median']:>11.1f}"
            f"{entry['routing']['selected_experts']:>9}"
            f"{entry['routing']['max_m_blocks']:>10}"
        )
    lines += [
        "",
        "HOW TO READ THIS",
        "  TFLOP/s is dense-equivalent: it counts the arithmetic a dense BF16",
        "  GEMM would perform for the same routed work, so it can be read",
        "  against a dense GEMM rate at the same hidden and intermediate",
        "  sizes. It is not this kernel's operation count.",
        "  GB/s counts compressed weight bytes for the experts the routing",
        "  selected, once each. If that rate approaches the device's memory",
        "  bandwidth the kernel is weight-traffic bound and the tier bit mix",
        "  moves elapsed time; if it sits far below, decode and launch",
        "  behaviour bound it and the mix does not.",
        "  Both rates are lower bounds to the extent the kernel re-reads",
        "  weights across m-blocks, which this harness does not observe.",
    ]
    return "\n".join(lines) + "\n"


def _render_tune(report: dict[str, Any]) -> str:
    ranking = report["ranking"]
    enumeration = report["enumeration"]
    lines = [
        "Mixed-bitrate Trellis MoE kernel tiling sweep, single GPU, single process",
        f"schema: {report['schema']}  mode: {report['mode']}  "
        f"size_m: {report['tuned_size_m']}",
        "",
    ]
    lines += _condition_lines(report)
    baseline = report["baseline_configuration"]
    lines += [
        f"  {'baseline':<22}{tuple(baseline['tile_config'])}, moe_block_size "
        f"{baseline['moe_block_size']}",
        f"  {'configurations':<22}{enumeration.get('measured', 0)} measured of "
        f"{enumeration.get('enumerated', 0)} enumerated, cap "
        f"{enumeration.get('cap', 0)}",
    ]
    if enumeration.get("truncated"):
        lines.append(
            f"  {'cap note':<22}{enumeration['dropped']} configuration(s) were "
            "not measured because the cap was reached"
        )
    lines += _arithmetic_lines(report)
    lines += [
        "",
        "RANKING BY MEDIAN",
        f"  {'rank':>5}  {'configuration':<28}{'role':<10}{'median ms':>12}"
        f"{'IQR ms':>10}{'speedup':>10}",
    ]
    for index, row in enumerate(ranking.get("ordered", []), start=1):
        lines.append(
            f"  {index:>5}  {row['name']:<28}{row['role']:<10}"
            f"{row['median_ms']:>12.4f}{row['iqr_ms']:>10.4f}"
            f"{row['speedup_over_baseline']:>10.3f}"
        )
    skipped = [
        entry for entry in report["configurations"] if entry["status"] == "skipped"
    ]
    if skipped:
        lines += ["", "SKIPPED"]
        for entry in skipped:
            lines.append(
                f"  {entry['configuration']['name']:<28}{entry['stage']:<18}"
                f"{entry['exception_type']}: {entry['reason'][:90]}"
            )
    if any(
        entry["configuration"]["role"] == ROLE_CONTROL
        for entry in report["configurations"]
    ):
        lines += ["", "CONTROL", f"  {CONTROL_NOTE}"]
    lines += [
        "",
        "VERDICT",
        f"  {ranking.get('verdict', 'no ranking was produced')}",
        "  speedup is the baseline median divided by the row's median, so a",
        "  value above 1.000 is faster than the deployed configuration. Read",
        "  it against the interquartile range in the same row: a speedup",
        "  smaller than the spread is not a difference this run resolved.",
    ]
    return "\n".join(lines) + "\n"


def emit_json(report: dict[str, Any], destination: str) -> None:
    """Write the report as JSON to a path, or to stdout for `-`."""

    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if destination == "-":
        print(rendered)
        return
    Path(destination).write_text(rendered + "\n", encoding="utf-8")


def render_configuration_table(
    configurations: Sequence[Configuration],
    enumeration: dict[str, Any],
    geometry: Geometry,
    size_m: int,
) -> str:
    """The sweep space and its arithmetic, without touching a device."""

    lines = [
        f"{'configuration':<28}{'role':<10}{'fc1_tile':>12}{'fc2_tile':>12}"
        f"{'block':>7}"
    ]
    for configuration in configurations:
        tile = configuration.tile_config
        fc1 = f"{tile[0]}x{tile[1]}"
        fc2 = f"{tile[2]}x{tile[3]}"
        lines.append(
            f"{configuration.name:<28}{configuration.role:<10}{fc1:>12}"
            f"{fc2:>12}{configuration.moe_block_size:>7}"
        )
    lines += [
        "",
        f"{enumeration['measured']} of {enumeration['enumerated']} enumerated "
        f"configuration(s), cap {enumeration['cap']}",
        "",
        f"at size_m={size_m}: dense_equivalent_flop = "
        f"{dense_equivalent_flop(geometry, size_m)}",
        f"one prepared weight set holds {resident_weight_bytes(geometry)} "
        "compressed bytes of expert weights",
        "",
        FLOP_FORMULA,
        WEIGHT_BYTES_FORMULA,
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Command line.
# --------------------------------------------------------------------------


def _int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.replace(",", " ").split())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("every value must be at least 1")
    return values


def _tile_config(text: str) -> tuple[int, int, int, int]:
    values = _int_list(text)
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "a tile config is four integers: fc1_tile_k, fc1_tile_n, "
            "fc2_tile_k, fc2_tile_n"
        )
    return (values[0], values[1], values[2], values[3])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mixed_trellis_roofline",
        description=(
            "Measure the fused mixed-bitrate Trellis MoE kernel at a GB10 "
            "deployment geometry, and sweep its tiling for a configuration "
            "faster than the deployed one."
        ),
    )
    parser.add_argument(
        "mode",
        choices=("measure", "tune"),
        help=(
            "measure: time the deployed configuration across token counts. "
            "tune: sweep force_tile_config and moe_block_size at one token "
            "count and rank against the deployed configuration."
        ),
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help=(
            "untimed calls per configuration before measuring, absorbing the "
            "kernel's compilation (default: 20)"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="timed calls per configuration (default: 100)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=3,
        help=(
            "prepared weight sets cycled one per timed call, so weights stream "
            "cold. Each set holds every expert's packed weights: about 0.91 "
            "GiB at the default geometry, so the default of 3 costs about 2.7 "
            "GiB of device memory (default: 3)"
        ),
    )
    parser.add_argument(
        "--max-pool-fraction",
        type=float,
        default=0.25,
        help=(
            "refuse a weight pool larger than this fraction of device memory "
            "(default: 0.25)"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="activation and rotation input dtype (default: bf16)",
    )
    parser.add_argument(
        "--activation",
        default="silu",
        help="gated MoE activation passed to weight preparation (default: silu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260820,
        help="seed for the synthetic weights and the routing (default: 20260820)",
    )
    parser.add_argument(
        "--sizes",
        type=_int_list,
        default=DEPLOYED_SIZES_M,
        help=(
            "token counts to measure in measure mode (default: "
            f"{' '.join(str(value) for value in DEPLOYED_SIZES_M)})"
        ),
    )
    parser.add_argument(
        "--size-m",
        type=int,
        default=DECODE_SIZE_M,
        help=(
            "token count the sweep holds fixed in tune mode, the deployed "
            f"decode capacity (default: {DECODE_SIZE_M})"
        ),
    )
    parser.add_argument(
        "--hidden-size", type=int, default=DEPLOYED_GEOMETRY.hidden_size
    )
    parser.add_argument(
        "--intermediate-size", type=int, default=DEPLOYED_GEOMETRY.intermediate_size
    )
    parser.add_argument(
        "--tier0-experts", type=int, default=DEPLOYED_GEOMETRY.tier0_num_experts
    )
    parser.add_argument(
        "--tier1-experts", type=int, default=DEPLOYED_GEOMETRY.tier1_num_experts
    )
    parser.add_argument("--tier0-bits", type=int, default=DEPLOYED_GEOMETRY.tier0_bits)
    parser.add_argument("--tier1-bits", type=int, default=DEPLOYED_GEOMETRY.tier1_bits)
    parser.add_argument("--top-k", type=int, default=DEPLOYED_GEOMETRY.top_k)
    parser.add_argument(
        "--sms",
        type=int,
        default=DEPLOYED_GEOMETRY.sms,
        help="streaming multiprocessor count the launch is planned for",
    )
    parser.add_argument(
        "--baseline-tile-config",
        type=_tile_config,
        default=DEPLOYED_TILE_CONFIG,
        help=(
            "tile config the sweep ranks against, four integers: fc1_tile_k "
            "fc1_tile_n fc2_tile_k fc2_tile_n. The vLLM EXL3 backend selects "
            "it from hidden_size, which at 6144 gives '128 128 32 512' "
            "(default: 128 128 32 512)"
        ),
    )
    parser.add_argument(
        "--baseline-moe-block-size",
        type=int,
        default=DEPLOYED_MOE_BLOCK_SIZE,
        help=(
            "route block size the sweep ranks against (default: "
            f"{DEPLOYED_MOE_BLOCK_SIZE})"
        ),
    )
    parser.add_argument(
        "--fc1-tile-k",
        type=_int_list,
        default=DEFAULT_FC1_K_VALUES,
        help=(
            "FC1 K tile widths to sweep. The mixed three-and-four-bit "
            "megakernel needs a 128-wide FC1 K tile, so widening this is "
            "opt-in (default: 128)"
        ),
    )
    parser.add_argument(
        "--fc1-tile-n",
        type=_int_list,
        default=DEFAULT_FC1_N_VALUES,
        help="FC1 output tile widths to sweep (default: 128)",
    )
    parser.add_argument(
        "--fc2-tile-k",
        type=_int_list,
        default=DEFAULT_FC2_K_VALUES,
        help="FC2 K tile widths to sweep (default: 32 64 128)",
    )
    parser.add_argument(
        "--fc2-tile-n",
        type=_int_list,
        default=DEFAULT_FC2_N_VALUES,
        help=(
            "FC2 output tile widths to sweep. The deployed 512 is chosen to "
            "remove a second persistent wave, which is a claim about wave "
            "quantization against the device's SM count (default: 128 256 512)"
        ),
    )
    parser.add_argument(
        "--moe-block-sizes",
        type=_int_list,
        default=DEFAULT_MOE_BLOCK_SIZES,
        help="route block sizes to sweep (default: 8 16)",
    )
    parser.add_argument(
        "--include-fc1-control",
        action="store_true",
        help=(
            "also measure the 64x256 FC1 geometry the single-bitrate path "
            "uses, labelled a control because it is documented as losing "
            "partial reductions at large token counts"
        ),
    )
    parser.add_argument(
        "--max-configurations",
        type=int,
        default=DEFAULT_CONFIGURATION_CAP,
        help=(
            "cap on configurations measured in one sweep; the baseline is "
            f"always among them (default: {DEFAULT_CONFIGURATION_CAP})"
        ),
    )
    parser.add_argument(
        "--module",
        default=KERNEL_MODULE,
        help=f"kernel module to import (default: {KERNEL_MODULE})",
    )
    parser.add_argument(
        "--module-path",
        default=KERNEL_MODULE_PATH,
        help=(
            "file the kernel module must resolve to; a mismatch is refused "
            "because an import hook can bind another implementation to the "
            "same name"
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write the report as JSON to PATH, or to stdout for '-'",
    )
    parser.add_argument(
        "--list-configurations",
        action="store_true",
        help="print the sweep space with its arithmetic and measure nothing",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.warmup < 1:
        parser.error(
            "--warmup must be at least 1; this kernel is compiled, so the "
            "first call to a configuration is not a steady state"
        )
    if arguments.iterations < 4:
        parser.error("--iterations must be at least 4 for an interquartile range")
    if arguments.pool_size < 1:
        parser.error("--pool-size must be at least 1")
    if not 0.0 < arguments.max_pool_fraction <= 1.0:
        parser.error("--max-pool-fraction must lie in (0, 1]")
    if arguments.max_configurations < 1:
        parser.error("--max-configurations must be at least 1")
    if arguments.size_m < 1:
        parser.error("--size-m must be at least 1")
    return arguments


def geometry_from_arguments(arguments: argparse.Namespace) -> Geometry:
    """The problem size the run measures, from defaults and overrides."""

    return replace(
        DEPLOYED_GEOMETRY,
        hidden_size=arguments.hidden_size,
        intermediate_size=arguments.intermediate_size,
        tier0_num_experts=arguments.tier0_experts,
        tier1_num_experts=arguments.tier1_experts,
        tier0_bits=arguments.tier0_bits,
        tier1_bits=arguments.tier1_bits,
        top_k=arguments.top_k,
        sms=arguments.sms,
    )


def baseline_from_arguments(arguments: argparse.Namespace) -> Configuration:
    """The configuration a sweep ranks against and measure mode times."""

    return Configuration(
        _tile_config_tuple(arguments.baseline_tile_config),
        arguments.baseline_moe_block_size,
        ROLE_BASELINE,
    )


def _tile_config_tuple(values: Sequence[int]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError(
            f"a tile config is four integers; got {len(values)}: {list(values)}"
        )
    return (values[0], values[1], values[2], values[3])


def configurations_from_arguments(
    arguments: argparse.Namespace,
) -> tuple[list[Configuration], dict[str, Any]]:
    """The sweep space the command line asks for."""

    return enumerate_configurations(
        baseline=baseline_from_arguments(arguments),
        fc1_k_values=arguments.fc1_tile_k,
        fc1_n_values=arguments.fc1_tile_n,
        fc2_k_values=arguments.fc2_tile_k,
        fc2_n_values=arguments.fc2_tile_n,
        moe_block_sizes=arguments.moe_block_sizes,
        controls=(FC1_CONTROL_TILE_CONFIG,) if arguments.include_fc1_control else (),
        cap=arguments.max_configurations,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    load_torch: Callable[[], Any] = import_torch,
    load_kernel: Callable[..., tuple[Any, Any, Any]] = import_kernel_modules,
) -> int:
    arguments = parse_args(argv)
    geometry = geometry_from_arguments(arguments)

    if arguments.list_configurations:
        configurations, enumeration = configurations_from_arguments(arguments)
        print(
            render_configuration_table(
                configurations, enumeration, geometry, arguments.size_m
            ),
            end="",
        )
        return EXIT_OK

    try:
        torch_module = load_torch()
        device, environment = open_device(torch_module, arguments)
        kernel_module, prepare_module, host_module = load_kernel(
            kernel_name=arguments.module
        )
        module_record = module_path_record(kernel_module, arguments.module_path)
        if not module_record["matches"]:
            raise MeasurementUnavailable(
                f"{arguments.module} resolves to {module_record['resolved']} "
                f"where {module_record['expected']} was expected. "
                f"{module_record['reason']} Pass --module-path to name the "
                "path this run should measure."
            )
        api, discovery = resolve_api(kernel_module, prepare_module, host_module)
        if arguments.mode == "measure":
            report = run_measure(
                torch_module,
                api,
                geometry=geometry,
                configuration=baseline_from_arguments(arguments),
                sizes=arguments.sizes,
                device=device,
                environment=environment,
                discovery=discovery,
                module_record=module_record,
                arguments=arguments,
            )
        else:
            configurations, enumeration = configurations_from_arguments(arguments)
            report = run_tune(
                torch_module,
                api,
                geometry=geometry,
                configurations=configurations,
                enumeration=enumeration,
                size_m=arguments.size_m,
                device=device,
                environment=environment,
                discovery=discovery,
                module_record=module_record,
                arguments=arguments,
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
