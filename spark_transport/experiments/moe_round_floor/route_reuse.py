"""Analyze expert reuse available inside a speculative verification batch.

The analyzer is deliberately GPU-free.  It consumes the compact target-route
records described by k8_two_block_prototype/README.md and answers the first
question that gates an expert-coherent direct-micro kernel project: do
candidate positions route to enough of the same experts for weight reuse to
matter?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


SCHEMA = "glm52-target-expert-routes/v1"


@dataclass(frozen=True)
class LayerReuse:
    layer: int
    positions: int
    assignments: int
    unique_experts: int
    route_order_runs: int

    @property
    def reuse_factor(self) -> float:
        return self.assignments / self.unique_experts

    @property
    def duplicate_fraction(self) -> float:
        return 1.0 - self.unique_experts / self.assignments

    @property
    def schedule_compaction_factor(self) -> float:
        """Logical expert runs removable by an expert-grouped schedule.

        This is not a cache-miss or speedup estimate. The deployed direct-micro
        kernel distributes route chunks across resident CTAs, so hardware
        overlap and cache behavior still require GPU counters.
        """
        return self.route_order_runs / self.unique_experts


@dataclass(frozen=True)
class RoundReuse:
    request_key: str
    round: int
    layers: int
    assignments: int
    unique_expert_layer_pairs: int
    route_order_runs: int
    median_layer_reuse: float
    p90_layer_reuse: float

    @property
    def aggregate_reuse_factor(self) -> float:
        return self.assignments / self.unique_expert_layer_pairs

    @property
    def duplicate_fraction(self) -> float:
        return 1.0 - self.unique_expert_layer_pairs / self.assignments

    @property
    def schedule_compaction_factor(self) -> float:
        return self.route_order_runs / self.unique_expert_layer_pairs


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_expert_ids(value: object, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}: expert_ids must be a non-empty list")
    result = [int(item) for item in value]
    if any(item < 0 for item in result):
        raise ValueError(f"{context}: expert ids must be non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"{context}: duplicate expert id within one position")
    return result


def iter_layers(record: dict, width: int) -> Iterator[tuple[int, list[list[int]]]]:
    """Yield ``(layer, positions[expert_ids])`` from the canonical schema."""
    if record.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema: {record.get('schema')!r}")
    layers = record.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("record must contain at least one layer")

    seen_layers: set[int] = set()
    for layer_record in layers:
        layer = int(layer_record["layer"])
        if layer in seen_layers:
            raise ValueError(f"duplicate layer {layer}")
        seen_layers.add(layer)
        positions = layer_record.get("positions")
        if not isinstance(positions, list) or len(positions) < width:
            count = len(positions) if isinstance(positions, list) else 0
            raise ValueError(
                f"layer {layer}: has {count} positions, requires Q{width}"
            )
        yield layer, [
            _validate_expert_ids(
                position.get("expert_ids") if isinstance(position, dict) else None,
                f"layer {layer} position {position_index}",
            )
            for position_index, position in enumerate(positions[:width])
        ]


def analyze_layer(layer: int, positions: list[list[int]]) -> LayerReuse:
    route_order = [expert for experts in positions for expert in experts]
    assignments = len(route_order)
    unique_experts = len(set(route_order))
    route_order_runs = 1 + sum(
        left != right for left, right in zip(route_order, route_order[1:])
    )
    return LayerReuse(
        layer=layer,
        positions=len(positions),
        assignments=assignments,
        unique_experts=unique_experts,
        route_order_runs=route_order_runs,
    )


def analyze_round(record: dict, width: int = 5) -> tuple[RoundReuse, list[LayerReuse]]:
    layer_results = [
        analyze_layer(layer, positions) for layer, positions in iter_layers(record, width)
    ]
    assignments = sum(layer.assignments for layer in layer_results)
    unique_pairs = sum(layer.unique_experts for layer in layer_results)
    route_order_runs = sum(layer.route_order_runs for layer in layer_results)
    reuse_values = [layer.reuse_factor for layer in layer_results]
    summary = RoundReuse(
        request_key=str(record.get("request_key", "")),
        round=int(record.get("round", 0)),
        layers=len(layer_results),
        assignments=assignments,
        unique_expert_layer_pairs=unique_pairs,
        route_order_runs=route_order_runs,
        median_layer_reuse=statistics.median(reuse_values),
        p90_layer_reuse=_percentile(reuse_values, 0.90),
    )
    return summary, layer_results


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: {error}") from error
    if not records:
        raise ValueError("trace contains no records")
    return records


def classify(reuse_factor: float) -> str:
    if reuse_factor >= 1.8:
        return "GO: substantial expert-weight reuse is available"
    if reuse_factor >= 1.3:
        return "MEASURE: useful but not independently transformative"
    return "NO-GO: route coherence alone cannot justify the kernel project"


def expert_expansion_summary(records: Iterable[dict], width: int) -> dict:
    """Summarize E(k) across every layer-round observation.

    E(k) is the number of unique experts touched by positions ``1..k``
    divided by the unique experts touched by position 1 in the same layer and
    round.
    """

    values_by_k: list[list[float]] = [[] for _ in range(width)]
    for record in records:
        for _, positions in iter_layers(record, width):
            baseline = len(set(positions[0]))
            touched: set[int] = set()
            for position_index, experts in enumerate(positions):
                touched.update(experts)
                values_by_k[position_index].append(len(touched) / baseline)
    observations = len(values_by_k[0]) if values_by_k else 0
    return {
        "definition": (
            "E(k) = unique experts in positions 1..k divided by unique "
            "experts in position 1, measured per layer-round observation"
        ),
        "observations": observations,
        "curve": [
            {
                "k": index + 1,
                "median": statistics.median(values),
                "p10": _percentile(values, 0.10),
                "p90": _percentile(values, 0.90),
                "min": min(values),
                "max": max(values),
            }
            for index, values in enumerate(values_by_k)
        ],
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def _exact_acceptance_counts(
    record: dict,
    width: int,
    record_index: int,
) -> tuple[int, int] | None:
    accepted_present = "accepted_prefix_tokens" in record
    rejected_present = "rejected_tokens" in record
    if not accepted_present and not rejected_present:
        return None
    if not accepted_present or not rejected_present:
        raise ValueError(
            f"record {record_index}: accepted_prefix_tokens and "
            "rejected_tokens must either both be present or both be absent"
        )

    if "width" in record:
        declared_width = record["width"]
        if type(declared_width) is not int or declared_width != width:
            raise ValueError(
                f"record {record_index}: declared width "
                f"{declared_width!r} does not match analysis width {width}"
            )

    accepted = record["accepted_prefix_tokens"]
    rejected = record["rejected_tokens"]
    if type(accepted) is not int or type(rejected) is not int:
        raise ValueError(
            f"record {record_index}: accepted_prefix_tokens and "
            "rejected_tokens must be integers"
        )
    if accepted < 0 or rejected < 0:
        raise ValueError(
            f"record {record_index}: accepted_prefix_tokens and "
            "rejected_tokens must be non-negative"
        )
    if accepted + rejected != width - 1:
        raise ValueError(
            f"record {record_index}: accepted_prefix_tokens ({accepted}) + "
            f"rejected_tokens ({rejected}) must equal width - 1 "
            f"({width - 1})"
        )
    return accepted, rejected


def rejected_route_waste_summary(records: Iterable[dict], width: int) -> dict:
    """Measure exact expert routes introduced only by rejected positions.

    Position zero is retained independent of speculative acceptance. The next
    ``accepted_prefix_tokens`` positions are the accepted speculative prefix,
    and the remaining ``rejected_tokens`` positions are rejected. An expert
    is waste for a layer-round exactly when a rejected position touches it and
    neither position zero nor an accepted speculative position touches it.
    """

    record_list = list(records)
    acceptance_counts = [
        _exact_acceptance_counts(record, width, record_index)
        for record_index, record in enumerate(record_list)
    ]
    available = [counts is not None for counts in acceptance_counts]
    if any(available) and not all(available):
        raise ValueError(
            "mixed exact acceptance metadata availability: every record must "
            "contain both accepted_prefix_tokens and rejected_tokens, or no "
            "record may contain either field"
        )
    if not any(available):
        return {
            "available": False,
            "reason": (
                "exact rejected-route waste requires per-round "
                "accepted_prefix_tokens and rejected_tokens; legacy "
                "route-only records do not contain acceptance metadata"
            ),
        }

    layer_round_details: list[dict] = []
    round_details: list[dict] = []
    for record, counts in zip(record_list, acceptance_counts):
        if counts is None:  # Guarded by the all-or-none check above.
            raise AssertionError("exact acceptance metadata unexpectedly absent")
        accepted, rejected = counts
        current_round_layers: list[dict] = []
        for layer, positions in iter_layers(record, width):
            retained_positions = positions[: accepted + 1]
            rejected_positions = positions[accepted + 1 :]
            retained_experts = {
                expert
                for position in retained_positions
                for expert in position
            }
            rejected_experts = {
                expert
                for position in rejected_positions
                for expert in position
            }
            all_experts = retained_experts | rejected_experts
            rejected_only_experts = rejected_experts - retained_experts
            detail = {
                "request_key": str(record.get("request_key", "")),
                "round": int(record.get("round", 0)),
                "layer": layer,
                "accepted_prefix_tokens": accepted,
                "rejected_tokens": rejected,
                "retained_positions": len(retained_positions),
                "rejected_positions": len(rejected_positions),
                "retained_unique_experts": len(retained_experts),
                "rejected_unique_experts": len(rejected_experts),
                "all_unique_experts": len(all_experts),
                "rejected_only_unique_experts": len(rejected_only_experts),
                "rejected_only_fraction_of_all_unique_experts": (
                    len(rejected_only_experts) / len(all_experts)
                ),
            }
            current_round_layers.append(detail)
            layer_round_details.append(detail)

        all_pairs = sum(
            detail["all_unique_experts"] for detail in current_round_layers
        )
        rejected_only_pairs = sum(
            detail["rejected_only_unique_experts"]
            for detail in current_round_layers
        )
        round_details.append(
            {
                "request_key": str(record.get("request_key", "")),
                "round": int(record.get("round", 0)),
                "accepted_prefix_tokens": accepted,
                "rejected_tokens": rejected,
                "layers": len(current_round_layers),
                "retained_unique_expert_layer_pairs": sum(
                    detail["retained_unique_experts"]
                    for detail in current_round_layers
                ),
                "rejected_unique_expert_layer_pairs": sum(
                    detail["rejected_unique_experts"]
                    for detail in current_round_layers
                ),
                "all_unique_expert_layer_pairs": all_pairs,
                "rejected_only_unique_expert_layer_pairs": rejected_only_pairs,
                "rejected_only_fraction_of_all_unique_expert_layer_pairs": (
                    rejected_only_pairs / all_pairs
                ),
            }
        )

    total_all_pairs = sum(
        detail["all_unique_expert_layer_pairs"] for detail in round_details
    )
    total_rejected_only_pairs = sum(
        detail["rejected_only_unique_expert_layer_pairs"]
        for detail in round_details
    )
    return {
        "available": True,
        "exact": True,
        "definition": (
            "per layer-round, unique experts touched by rejected speculative "
            "positions but not by position 0 or the accepted speculative "
            "prefix; expert identity is layer-scoped"
        ),
        "position_convention": (
            "position 0 is retained; positions 1..accepted_prefix_tokens are "
            "accepted; the trailing rejected_tokens positions are rejected"
        ),
        "rounds": len(round_details),
        "layer_round_observations": len(layer_round_details),
        "aggregate": {
            "all_unique_expert_layer_pair_observations": total_all_pairs,
            "rejected_only_unique_expert_layer_pair_observations": (
                total_rejected_only_pairs
            ),
            "rejected_only_fraction_of_all_unique_expert_layer_pair_observations": (
                total_rejected_only_pairs / total_all_pairs
            ),
        },
        "per_layer_round": {
            "observations": len(layer_round_details),
            "rejected_only_unique_experts": _distribution(
                [
                    detail["rejected_only_unique_experts"]
                    for detail in layer_round_details
                ]
            ),
            "rejected_only_fraction_of_all_unique_experts": _distribution(
                [
                    detail["rejected_only_fraction_of_all_unique_experts"]
                    for detail in layer_round_details
                ]
            ),
        },
        "per_round": {
            "observations": len(round_details),
            "rejected_only_unique_expert_layer_pairs": _distribution(
                [
                    detail["rejected_only_unique_expert_layer_pairs"]
                    for detail in round_details
                ]
            ),
            "rejected_only_fraction_of_all_unique_expert_layer_pairs": (
                _distribution(
                    [
                        detail[
                            "rejected_only_fraction_of_all_unique_expert_layer_pairs"
                        ]
                        for detail in round_details
                    ]
                )
            ),
        },
        "layer_round_details": layer_round_details,
        "round_details": round_details,
    }


def adjacent_round_reuse_summary(records: Iterable[dict], width: int) -> dict:
    """Measure layer-local reuse between consecutive rounds of one request."""

    candidates: list[tuple[str, int, dict[int, set[int]]]] = []
    occurrence_counts: dict[tuple[str, int], int] = {}
    skipped_missing_request_key = 0
    for record in records:
        request_key = str(record.get("request_key", ""))
        if not request_key:
            skipped_missing_request_key += 1
            continue
        round_index = int(record.get("round", 0))
        key = (request_key, round_index)
        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
        candidates.append(
            (
                request_key,
                round_index,
                {
                    layer: {
                        expert
                        for position in positions
                        for expert in position
                    }
                    for layer, positions in iter_layers(record, width)
                },
            )
        )

    ambiguous_keys = {
        key for key, count in occurrence_counts.items() if count > 1
    }
    skipped_duplicate_round_records = sum(
        occurrence_counts[key] for key in ambiguous_keys
    )
    grouped: dict[str, dict[int, dict[int, set[int]]]] = {}
    for request_key, round_index, layers in candidates:
        if (request_key, round_index) in ambiguous_keys:
            continue
        grouped.setdefault(request_key, {})[round_index] = layers

    intersection_values: list[float] = []
    previous_retention_values: list[float] = []
    next_coverage_values: list[float] = []
    jaccard_values: list[float] = []
    round_pairs = 0
    skipped_layer_mismatch_pairs = 0
    for request_key, request_rounds in grouped.items():
        ordered = sorted(request_rounds.items())
        for (previous_round, previous), (next_round, current) in zip(
            ordered, ordered[1:]
        ):
            if next_round != previous_round + 1:
                continue
            if set(previous) != set(current):
                skipped_layer_mismatch_pairs += 1
                continue
            round_pairs += 1
            for layer in sorted(previous):
                before = previous[layer]
                after = current[layer]
                intersection = len(before & after)
                union = len(before | after)
                intersection_values.append(float(intersection))
                previous_retention_values.append(intersection / len(before))
                next_coverage_values.append(intersection / len(after))
                jaccard_values.append(intersection / union)

    return {
        "available": bool(round_pairs),
        "definition": (
            "observations are matching layers in rounds r and r+1 within the "
            "same request_key; each round-layer set unions positions 1..width"
        ),
        "round_pairs": round_pairs,
        "layer_observations": len(jaccard_values),
        "skipped_missing_request_key": skipped_missing_request_key,
        "skipped_duplicate_round_records": skipped_duplicate_round_records,
        "skipped_layer_mismatch_pairs": skipped_layer_mismatch_pairs,
        "intersection_experts": _distribution(intersection_values),
        "previous_round_retention": {
            "definition": "|A intersect B| / |A|",
            **_distribution(previous_retention_values),
        },
        "next_round_coverage": {
            "definition": (
                "|A intersect B| / |B|; fraction of next-round experts "
                "already touched in the previous round"
            ),
            **_distribution(next_coverage_values),
        },
        "jaccard": {
            "definition": "|A intersect B| / |A union B|",
            **_distribution(jaccard_values),
        },
    }


def _without_provenance(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_provenance(item)
            for key, item in value.items()
            if key != "provenance"
        }
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return value


def canonical_route_digest(records: Iterable[dict]) -> str:
    """Return an order-stable SHA-256 for rank comparison.

    Provenance objects are excluded because their rank and environment fields
    are expected to differ. All route, request, round, layer, position, and
    capture metadata remains covered.
    """

    canonical_records = sorted(
        json.dumps(
            _without_provenance(record),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )
    payload = "[" + ",".join(canonical_records) + "]"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize(records: Iterable[dict], width: int = 5) -> dict:
    record_list = list(records)
    rejected_route_waste = rejected_route_waste_summary(record_list, width)
    rounds = [analyze_round(record, width)[0] for record in record_list]
    reuse = [item.aggregate_reuse_factor for item in rounds]
    duplicates = [item.duplicate_fraction for item in rounds]
    compaction = [item.schedule_compaction_factor for item in rounds]
    result = {
        "schema": "glm52-expert-reuse-summary/v1",
        "width": width,
        "rounds": len(rounds),
        "aggregate_reuse": {
            "median": statistics.median(reuse),
            "p10": _percentile(reuse, 0.10),
            "p90": _percentile(reuse, 0.90),
        },
        "duplicate_fraction": {
            "median": statistics.median(duplicates),
            "p10": _percentile(duplicates, 0.10),
            "p90": _percentile(duplicates, 0.90),
        },
        "logical_schedule_compaction": {
            "median": statistics.median(compaction),
            "p10": _percentile(compaction, 0.10),
            "p90": _percentile(compaction, 0.90),
            "warning": (
                "logical route-run reduction only; not a cache-miss or speedup "
                "estimate"
            ),
        },
        "decision": classify(_percentile(reuse, 0.10)),
        "canonical_route_sha256": canonical_route_digest(record_list),
        "expert_expansion": expert_expansion_summary(record_list, width),
        "adjacent_round_reuse": adjacent_round_reuse_summary(
            record_list, width
        ),
        "rejected_route_waste": rejected_route_waste,
        "round_details": [asdict(item) | {
            "aggregate_reuse_factor": item.aggregate_reuse_factor,
            "duplicate_fraction": item.duplicate_fraction,
            "schedule_compaction_factor": item.schedule_compaction_factor,
        } for item in rounds],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help=f"{SCHEMA} JSONL trace")
    parser.add_argument("--width", type=int, default=5, help="Q width (default: 5)")
    parser.add_argument("--output", help="optional JSON output path")
    arguments = parser.parse_args()
    if arguments.width <= 0:
        parser.error("--width must be positive")

    summary = summarize(load_jsonl(arguments.trace), arguments.width)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if arguments.output:
        Path(arguments.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
