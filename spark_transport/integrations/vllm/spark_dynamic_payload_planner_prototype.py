"""PROTOTYPE: formula admission and restart-only capture planning.

Question: can every positive width of a known collective family be admitted
without an exact-shape whitelist, while unknown semantics, malformed layouts,
and arena overflow still fail closed?

This GPU-free module is deliberately not imported by a live adapter.  It
builds on ``spark_decode_payload_contract`` and emits planning evidence only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from spark_decode_payload_contract import PAYLOAD_FAMILY_BY_KEY


REGISTRY_VERSION = 1
WORLD_SIZE = 4
DECODE_LATENCY_HORIZON_Q = 40
REGISTERED_ARENA_HORIZON_Q = 4096
ARENA_ALIGNMENT_BYTES = 2 * 1024 * 1024
ARENA_HEADROOM_NUMERATOR = 9
ARENA_HEADROOM_DENOMINATOR = 8
MAX_DESCRIPTOR_BYTES = (1 << 63) - 1


class FamilyTag(str, Enum):
    TP_AR = "tp_ar"
    INDEXER = "indexer"
    DCP_QUERY = "dcp_query"
    DCP_LSE = "dcp_lse"
    DCP_OUTPUT = "dcp_output"
    VOCABULARY = "vocabulary"
    FUSED_COMBINE_0 = "fused_combine_0"
    FUSED_COMBINE_1 = "fused_combine_1"


class CollectiveType(str, Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    FUSED_COMBINE = "fused_combine"


class DType(str, Enum):
    BF16 = "bf16"
    FP32 = "fp32"
    INT32 = "int32"


class TransportTier(str, Enum):
    LATENCY = "latency"
    CHUNKED = "chunked"


class DecisionCode(str, Enum):
    ADMIT_LATENCY = "admit_latency"
    ADMIT_CHUNKED = "admit_chunked"
    REJECT_UNKNOWN_FAMILY = "reject_unknown_family"
    REJECT_INVALID_DESCRIPTOR = "reject_invalid_descriptor"
    REJECT_CONTRACT_MISMATCH = "reject_contract_mismatch"
    REJECT_NONCONTIGUOUS = "reject_noncontiguous"
    REJECT_OVERFLOW = "reject_overflow"


class CensusRoute(str, Enum):
    PADDED = "padded"
    EAGER = "eager"


@dataclass(frozen=True)
class BytesPerQ:
    coefficient: int
    expression: str

    def bytes_for(self, query_rows: int) -> int:
        if (
            isinstance(query_rows, bool)
            or not isinstance(query_rows, int)
            or query_rows < 1
        ):
            raise ValueError("query_rows must be a positive integer")
        if query_rows > MAX_DESCRIPTOR_BYTES // self.coefficient:
            raise OverflowError("payload byte formula exceeds int64")
        return self.coefficient * query_rows


@dataclass(frozen=True)
class OutputExpansion:
    numerator: int
    denominator: int
    expression: str

    def output_bytes(self, input_bytes: int) -> int:
        if input_bytes > MAX_DESCRIPTOR_BYTES // self.numerator:
            raise OverflowError("output expansion exceeds int64")
        return (
            input_bytes * self.numerator + self.denominator - 1
        ) // self.denominator


@dataclass(frozen=True)
class ArenaPolicy:
    latency_slot_bytes: int
    capacity_bytes: int
    registered_horizon_q: int


@dataclass(frozen=True)
class SemanticFamily:
    tag: FamilyTag
    collective: CollectiveType
    dtype: DType
    layout: str
    bytes_per_q: BytesPerQ
    output_expansion: OutputExpansion
    arena: ArenaPolicy

    def byte_geometry(self, query_rows: int) -> tuple[int, int, int]:
        input_bytes = self.bytes_per_q.bytes_for(query_rows)
        output_bytes = self.output_expansion.output_bytes(input_bytes)
        return input_bytes, output_bytes, max(input_bytes, output_bytes)


@dataclass(frozen=True)
class PayloadDescriptor:
    family_tag: str
    collective: str
    dtype: str
    layout: str
    query_rows: int
    payload_bytes: int
    contiguous: bool


@dataclass(frozen=True)
class AdmissionDecision:
    code: DecisionCode
    admitted: bool
    reason: str
    family_tag: str
    tier: TransportTier | None = None
    expected_payload_bytes: int | None = None
    output_bytes: int | None = None
    arena_bytes: int | None = None
    arena_capacity_bytes: int | None = None
    chunk_count: int | None = None


@dataclass(frozen=True)
class CensusSample:
    family_tag: str
    query_rows: int
    count: int
    route: CensusRoute


@dataclass(frozen=True)
class PaddingMetric:
    family_tag: str
    query_rows: int
    route: str
    count: int
    target_bucket: int | None
    padding_rows: int
    logical_bytes: int
    padded_bytes: int
    padding_waste_bytes: int
    padding_waste_ppm: int


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _policy(
    bytes_per_q: int,
    output_expansion: OutputExpansion,
) -> ArenaPolicy:
    per_decode = max(
        bytes_per_q,
        output_expansion.output_bytes(bytes_per_q),
    )
    latency_slot = _round_up(
        per_decode * DECODE_LATENCY_HORIZON_Q,
        ARENA_ALIGNMENT_BYTES,
    )
    horizon_bytes = per_decode * REGISTERED_ARENA_HORIZON_Q
    capacity = _round_up(
        (
            horizon_bytes * ARENA_HEADROOM_NUMERATOR
            + ARENA_HEADROOM_DENOMINATOR
            - 1
        )
        // ARENA_HEADROOM_DENOMINATOR,
        ARENA_ALIGNMENT_BYTES,
    )
    return ArenaPolicy(
        latency_slot_bytes=latency_slot,
        capacity_bytes=capacity,
        registered_horizon_q=REGISTERED_ARENA_HORIZON_Q,
    )


def _family(
    tag: FamilyTag,
    collective: CollectiveType,
    dtype: DType,
    layout: str,
    output_expansion: OutputExpansion,
) -> SemanticFamily:
    coefficient = PAYLOAD_FAMILY_BY_KEY[tag.value].bytes_per_query_row
    return SemanticFamily(
        tag=tag,
        collective=collective,
        dtype=dtype,
        layout=layout,
        bytes_per_q=BytesPerQ(
            coefficient=coefficient,
            expression=f"{coefficient} * Q",
        ),
        output_expansion=output_expansion,
        arena=_policy(coefficient, output_expansion),
    )


_IDENTITY = OutputExpansion(1, 1, "1x")
_GATHER_4X = OutputExpansion(WORLD_SIZE, 1, "world_size (4x)")
_SCATTER_4X = OutputExpansion(1, WORLD_SIZE, "1 / world_size")

_FAMILIES = (
    _family(
        FamilyTag.TP_AR,
        CollectiveType.ALL_REDUCE,
        DType.BF16,
        "contiguous-row-major:[Q,6144]",
        _IDENTITY,
    ),
    _family(
        FamilyTag.INDEXER,
        CollectiveType.ALL_GATHER,
        DType.INT32,
        "contiguous-row-major:[Q,2,2048]",
        _GATHER_4X,
    ),
    _family(
        FamilyTag.DCP_QUERY,
        CollectiveType.ALL_GATHER,
        DType.BF16,
        "contiguous-row-major:[Q,16,576]",
        _GATHER_4X,
    ),
    _family(
        FamilyTag.DCP_LSE,
        CollectiveType.ALL_GATHER,
        DType.FP32,
        "contiguous-row-major:[Q,64]",
        _GATHER_4X,
    ),
    _family(
        FamilyTag.DCP_OUTPUT,
        CollectiveType.REDUCE_SCATTER,
        DType.BF16,
        "contiguous-row-major:[Q,64,512]",
        _SCATTER_4X,
    ),
    _family(
        FamilyTag.VOCABULARY,
        CollectiveType.ALL_GATHER,
        DType.BF16,
        "contiguous-row-major:[Q,38720]",
        _GATHER_4X,
    ),
    _family(
        FamilyTag.FUSED_COMBINE_0,
        CollectiveType.FUSED_COMBINE,
        DType.BF16,
        "contiguous-packed:32-head-output-plus-lse",
        _IDENTITY,
    ),
    _family(
        FamilyTag.FUSED_COMBINE_1,
        CollectiveType.FUSED_COMBINE,
        DType.BF16,
        "contiguous-packed:16-head-output-plus-lse",
        _IDENTITY,
    ),
)

FAMILY_REGISTRY: Mapping[str, SemanticFamily] = MappingProxyType(
    {family.tag.value: family for family in _FAMILIES}
)


def descriptor_for(
    family_tag: str,
    query_rows: int,
    *,
    contiguous: bool = True,
) -> PayloadDescriptor:
    """Build an exact descriptor for a known family (test/probe convenience)."""

    family = FAMILY_REGISTRY[family_tag]
    return PayloadDescriptor(
        family_tag=family_tag,
        collective=family.collective.value,
        dtype=family.dtype.value,
        layout=family.layout,
        query_rows=query_rows,
        payload_bytes=family.bytes_per_q.bytes_for(query_rows),
        contiguous=contiguous,
    )


def _rejection(
    code: DecisionCode,
    descriptor: PayloadDescriptor,
    reason: str,
    *,
    family: SemanticFamily | None = None,
) -> AdmissionDecision:
    return AdmissionDecision(
        code=code,
        admitted=False,
        reason=reason,
        family_tag=descriptor.family_tag,
        arena_capacity_bytes=(
            family.arena.capacity_bytes if family is not None else None
        ),
    )


def admit_payload(descriptor: PayloadDescriptor) -> AdmissionDecision:
    """Admit by semantic tag and formula; never infer a family from bytes."""

    family = FAMILY_REGISTRY.get(descriptor.family_tag)
    if family is None:
        return _rejection(
            DecisionCode.REJECT_UNKNOWN_FAMILY,
            descriptor,
            "family tag is not registered; byte collisions do not admit",
        )
    if (
        isinstance(descriptor.query_rows, bool)
        or not isinstance(descriptor.query_rows, int)
        or descriptor.query_rows < 1
        or isinstance(descriptor.payload_bytes, bool)
        or not isinstance(descriptor.payload_bytes, int)
        or descriptor.payload_bytes < 1
    ):
        return _rejection(
            DecisionCode.REJECT_INVALID_DESCRIPTOR,
            descriptor,
            "query_rows and payload_bytes must be positive integers",
            family=family,
        )
    if (
        descriptor.collective != family.collective.value
        or descriptor.dtype != family.dtype.value
        or descriptor.layout != family.layout
    ):
        return _rejection(
            DecisionCode.REJECT_CONTRACT_MISMATCH,
            descriptor,
            "collective, dtype, or layout does not match the family tag",
            family=family,
        )
    if not descriptor.contiguous:
        return _rejection(
            DecisionCode.REJECT_NONCONTIGUOUS,
            descriptor,
            "formula admission requires a contiguous tensor",
            family=family,
        )
    try:
        input_bytes, output_bytes, arena_bytes = family.byte_geometry(
            descriptor.query_rows
        )
    except OverflowError as error:
        return _rejection(
            DecisionCode.REJECT_OVERFLOW,
            descriptor,
            str(error),
            family=family,
        )
    if descriptor.payload_bytes != input_bytes:
        return _rejection(
            DecisionCode.REJECT_CONTRACT_MISMATCH,
            descriptor,
            "payload_bytes does not match the registered bytes-per-Q formula",
            family=family,
        )
    if arena_bytes > family.arena.capacity_bytes:
        return AdmissionDecision(
            code=DecisionCode.REJECT_OVERFLOW,
            admitted=False,
            reason="input/output footprint exceeds the registered arena cap",
            family_tag=descriptor.family_tag,
            expected_payload_bytes=input_bytes,
            output_bytes=output_bytes,
            arena_bytes=arena_bytes,
            arena_capacity_bytes=family.arena.capacity_bytes,
        )

    if arena_bytes <= family.arena.latency_slot_bytes:
        tier = TransportTier.LATENCY
        code = DecisionCode.ADMIT_LATENCY
        chunk_count = 1
    else:
        tier = TransportTier.CHUNKED
        code = DecisionCode.ADMIT_CHUNKED
        chunk_count = (
            arena_bytes + family.arena.latency_slot_bytes - 1
        ) // family.arena.latency_slot_bytes
    return AdmissionDecision(
        code=code,
        admitted=True,
        reason="known semantic family and arena-derived size admission",
        family_tag=descriptor.family_tag,
        tier=tier,
        expected_payload_bytes=input_bytes,
        output_bytes=output_bytes,
        arena_bytes=arena_bytes,
        arena_capacity_bytes=family.arena.capacity_bytes,
        chunk_count=chunk_count,
    )


def _validated_buckets(buckets: Sequence[int]) -> tuple[int, ...]:
    values = tuple(buckets)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        )
        or tuple(sorted(set(values))) != values
    ):
        raise ValueError(
            "capture buckets must be strictly increasing positive integers"
        )
    return values


def _aggregate_census(
    census: Sequence[CensusSample],
) -> tuple[CensusSample, ...]:
    totals: dict[tuple[str, int, CensusRoute], int] = {}
    for sample in census:
        if sample.family_tag not in FAMILY_REGISTRY:
            raise ValueError(
                f"census contains unknown family: {sample.family_tag}"
            )
        if (
            isinstance(sample.query_rows, bool)
            or not isinstance(sample.query_rows, int)
            or sample.query_rows < 1
            or isinstance(sample.count, bool)
            or not isinstance(sample.count, int)
            or sample.count < 1
            or not isinstance(sample.route, CensusRoute)
        ):
            raise ValueError("census samples must be positive and typed")
        key = (sample.family_tag, sample.query_rows, sample.route)
        totals[key] = totals.get(key, 0) + sample.count
    return tuple(
        CensusSample(family_tag, query_rows, count, route)
        for (family_tag, query_rows, route), count in sorted(
            totals.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2].value,
            ),
        )
    )


def _padding_metrics(
    buckets: Sequence[int],
    census: Sequence[CensusSample],
    *,
    exact_eager_promotions: frozenset[int],
) -> tuple[PaddingMetric, ...]:
    validated_buckets = _validated_buckets(buckets)
    metrics: list[PaddingMetric] = []
    for sample in _aggregate_census(census):
        family = FAMILY_REGISTRY[sample.family_tag]
        logical = family.byte_geometry(sample.query_rows)[2]
        target = next(
            (
                bucket
                for bucket in validated_buckets
                if bucket >= sample.query_rows
            ),
            None,
        )
        if (
            sample.route is CensusRoute.EAGER
            and sample.query_rows not in exact_eager_promotions
        ) or target is None:
            target = None
            padding_rows = 0
            padded = logical
            waste = 0
            waste_ppm = 0
        else:
            padding_rows = target - sample.query_rows
            padded = family.byte_geometry(target)[2]
            waste = padded - logical
            waste_ppm = (
                (waste * 1_000_000 + logical - 1) // logical
                if waste
                else 0
            )
        metrics.append(
            PaddingMetric(
                family_tag=sample.family_tag,
                query_rows=sample.query_rows,
                route=sample.route.value,
                count=sample.count,
                target_bucket=target,
                padding_rows=padding_rows,
                logical_bytes=logical * sample.count,
                padded_bytes=padded * sample.count,
                padding_waste_bytes=waste * sample.count,
                padding_waste_ppm=waste_ppm,
            )
        )
    return tuple(metrics)


def padding_metrics(
    buckets: Sequence[int],
    census: Sequence[CensusSample],
) -> tuple[PaddingMetric, ...]:
    """Return deterministic byte-weighted padding metrics per census row."""

    return _padding_metrics(
        buckets,
        census,
        exact_eager_promotions=frozenset(),
    )


def _metric_summary(metrics: Sequence[PaddingMetric]) -> dict[str, int]:
    logical = sum(metric.logical_bytes for metric in metrics)
    padded = sum(metric.padded_bytes for metric in metrics)
    waste = sum(metric.padding_waste_bytes for metric in metrics)
    return {
        "observations": sum(metric.count for metric in metrics),
        "eager_observations": sum(
            metric.count
            for metric in metrics
            if metric.route == "eager" and metric.target_bucket is None
        ),
        "logical_bytes": logical,
        "padded_bytes": padded,
        "padding_waste_bytes": waste,
        "padding_waste_ppm": (
            (waste * 1_000_000 + logical - 1) // logical if waste else 0
        ),
    }


def propose_next_restart_plan(
    current_buckets: Sequence[int],
    census: Sequence[CensusSample],
    *,
    min_observations: int = 100,
    maximum_new_buckets: int = 4,
    maximum_padding_waste_ppm: int = 80_000,
) -> dict[str, object]:
    """Propose, but never apply, exact-Q buckets for the next restart."""

    current = _validated_buckets(current_buckets)
    if (
        isinstance(min_observations, bool)
        or not isinstance(min_observations, int)
        or min_observations < 1
        or isinstance(maximum_new_buckets, bool)
        or not isinstance(maximum_new_buckets, int)
        or maximum_new_buckets < 0
        or isinstance(maximum_padding_waste_ppm, bool)
        or not isinstance(maximum_padding_waste_ppm, int)
        or maximum_padding_waste_ppm < 0
    ):
        raise ValueError("proposal policy values are out of range")

    aggregated = _aggregate_census(census)
    before_metrics = padding_metrics(current, aggregated)
    candidates: dict[int, dict[str, int]] = {}
    for metric in before_metrics:
        if metric.query_rows in current:
            continue
        candidate = candidates.setdefault(
            metric.query_rows,
            {
                "observations": 0,
                "eager_observations": 0,
                "padding_waste_bytes": 0,
                "priority_bytes": 0,
                "maximum_padding_waste_ppm": 0,
            },
        )
        candidate["observations"] += metric.count
        candidate["padding_waste_bytes"] += metric.padding_waste_bytes
        candidate["maximum_padding_waste_ppm"] = max(
            candidate["maximum_padding_waste_ppm"],
            metric.padding_waste_ppm,
        )
        if metric.route == "eager":
            candidate["eager_observations"] += metric.count
            candidate["priority_bytes"] += metric.logical_bytes
        else:
            candidate["priority_bytes"] += metric.padding_waste_bytes

    eligible = [
        (query_rows, values)
        for query_rows, values in candidates.items()
        if values["observations"] >= min_observations
        and (
            values["eager_observations"] > 0
            or values["padding_waste_bytes"] > 0
        )
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -int(
                item[1]["eager_observations"] > 0
                or item[1]["maximum_padding_waste_ppm"]
                > maximum_padding_waste_ppm
            ),
            -item[1]["priority_bytes"],
            -item[1]["observations"],
            item[0],
        ),
    )
    selected = ranked[:maximum_new_buckets]
    additions = sorted(query_rows for query_rows, _ in selected)
    proposed = tuple(sorted(set(current).union(additions)))
    after_metrics = _padding_metrics(
        proposed,
        aggregated,
        exact_eager_promotions=frozenset(additions),
    )
    selected_by_q = dict(selected)

    return {
        "schema_version": 1,
        "registry_version": REGISTRY_VERSION,
        "activation": "next_restart_only",
        "live_capture": False,
        "policy": {
            "min_observations": min_observations,
            "maximum_new_buckets": maximum_new_buckets,
            "maximum_padding_waste_ppm": maximum_padding_waste_ppm,
        },
        "current_buckets": list(current),
        "proposed_buckets": list(proposed),
        "additions": [
            {
                "query_rows": query_rows,
                **selected_by_q[query_rows],
            }
            for query_rows in additions
        ],
        "unselected_candidate_widths": sorted(
            query_rows
            for query_rows, _ in ranked[maximum_new_buckets:]
        ),
        "before": _metric_summary(before_metrics),
        "after": _metric_summary(after_metrics),
        "metrics": [asdict(metric) for metric in before_metrics],
    }


def canonical_plan_json(plan: Mapping[str, object]) -> str:
    return json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ) + "\n"


def _demo_plan() -> dict[str, object]:
    return propose_next_restart_plan(
        (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 72, 144),
        (
            CensusSample("indexer", 23, 4_112, CensusRoute.PADDED),
            CensusSample("vocabulary", 69, 800, CensusRoute.PADDED),
            CensusSample("dcp_query", 143, 600, CensusRoute.PADDED),
            CensusSample("vocabulary", 512, 300, CensusRoute.EAGER),
        ),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-plan",
        action="store_true",
        help="print a canonical restart-only proposal",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.demo_plan:
        raise SystemExit("use --demo-plan (this prototype never applies plans)")
    print(canonical_plan_json(_demo_plan()), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
