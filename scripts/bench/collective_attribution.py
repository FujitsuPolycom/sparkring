#!/usr/bin/env python3
"""Turn gated and naked collective timings into a bounded critical-path share.

A collective call's elapsed time on one rank is not that collective's cost.
It also contains the time that rank spent waiting for its slowest peer, and
that wait is produced by load imbalance elsewhere. Summing elapsed times
therefore yields a ceiling that can exceed the transport's real contribution
by an unknown factor, while dividing payload bytes by a link rate yields a
floor that no implementation can beat. The interval between the two is not a
measurement.

WHAT THIS READS

    Two capture documents describing the same collective inventory measured
    two ways, plus an optional third describing an injected-delay sweep.

    * A `naked` arm times each collective with nothing placed before it, so
      each sample carries whatever arrival skew had accumulated.
    * A `gated` arm places two back-to-back synchronizing operations on the
      same communicator and stream immediately before each timed collective,
      and times all three regions separately. The first gate absorbs arrival
      skew plus its own cost; the second, entered by every rank at nearly the
      same instant, measures its own cost alone. Their difference estimates
      skew and subtracts the gate from itself.
    * An `exposure` sweep records end-to-end wall time against a calibrated
      delay injected into the collective's stream region. Its slope is the
      fraction of a marginal collective microsecond that reaches wall time.

    `docs/COLLECTIVE_CRITICAL_PATH_MEASUREMENT.md` specifies the arms, the
    validity gates, and what each arm does and does not establish.

WHAT THIS COMPUTES

    Per rank and per collective instance, keyed by family, communicator,
    payload shape, and position in the step:

      floor      payload bytes * wire multiplier / link rate
      transport  median gated residency
      skew       median first-gate residency - median second-gate residency
      residency  median naked residency

    Per request, over the stated occurrence count of each instance, it sums
    those into a floor, a transport-limited ceiling, and a residency ceiling,
    and reports the residency ceiling that the gated arm removes.

    Every number is a median with its interquartile range and sample count.
    A single value with no spread does not say whether a rank was in a steady
    state.

DETECTION THRESHOLD

    A difference smaller than the layer's threshold is reported as
    indeterminate, never as an effect. The floors are fixed in this module:
    5% of the compared median for device-timed collectives and 10% for
    end-to-end serving quantities. `--detect-percent` may raise a threshold
    and is refused if it lowers one below its floor. The design document
    states the observations those floors come from.

    `--plan` computes the paired repetition count that a stated dispersion
    and a stated target require, so a campaign can size itself before it
    collects anything.

Safety class: OFFLINE. It reads the JSON paths named on the command line,
writes only the JSON path `--json` names, contacts no host, starts no
runtime, imports neither torch nor any distributed runtime, and allocates no
device memory. It takes no measurement of its own: every number it prints is
arithmetic over numbers some other instrument recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CAPTURE_SCHEMA = "sparkring-collective-attribution/v1"
EXPOSURE_SCHEMA = "sparkring-collective-exposure/v1"
REPORT_SCHEMA = "sparkring-collective-attribution-report/v1"

EXIT_OK = 0
# The two documents are individually valid but cannot be compared: different
# inventories, different layers, or a failed validity gate. Distinct from a
# document that is malformed.
EXIT_NOT_COMPARABLE = 2
EXIT_INPUT_MISSING = 3
EXIT_INVALID_DOCUMENT = 4

DEVICE_LAYER = "device"
END_TO_END_LAYER = "end_to_end"
LAYERS = (DEVICE_LAYER, END_TO_END_LAYER)

# The smallest relative difference each layer is permitted to call an effect.
# Both are relative to the compared median, in percent.
DETECT_FLOOR_PERCENT: Mapping[str, float] = {
    DEVICE_LAYER: 5.0,
    END_TO_END_LAYER: 10.0,
}

GATED_ARM = "gated"
NAKED_ARM = "naked"
ARMS = (GATED_ARM, NAKED_ARM)

RATE_BASES = ("nameplate", "measured")

# A gate whose own cost is a large fraction of the collective it precedes has
# changed the thing it was placed there to isolate.
GATE_COST_MAX_FRACTION = 0.10

# After a gate, every rank enters the collective at nearly the same instant,
# so their gated residencies should agree. Disagreement beyond this means the
# gate did not equalize arrival and the gated number is not a transport time.
CROSS_RANK_SPREAD_MAX_PERCENT = 25.0

# Normal-approximation multiplier for a two-sided 95% interval. The bootstrap
# over actual samples supersedes it; this sizes a campaign before samples
# exist.
Z_95 = 1.96

# A paired comparison below this many repetitions has no usable dispersion
# estimate of its own, whatever the arithmetic says.
MINIMUM_REPETITIONS = 3

BITS_PER_BYTE = 8
GBIT = 1_000_000_000.0
MICROSECONDS_PER_SECOND = 1_000_000.0

FLOOR_FORMULA = (
    "floor_us = payload_bytes * wire_bytes_multiplier * 8 "
    "/ (rate_gbit_per_second * 1e9) * 1e6"
)


class DocumentInvalid(ValueError):
    """A capture document does not satisfy the schema it declares."""


class NotComparable(ValueError):
    """Two individually valid documents do not describe the same inventory."""


# --------------------------------------------------------------------------
# Distribution summaries. Pure; exercised without any capture.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """One sample set, reported as a median with its observed spread."""

    count: int
    median: float
    q1: float
    q3: float
    minimum: float
    maximum: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "median": round(self.median, 6),
            "q1": round(self.q1, 6),
            "q3": round(self.q3, 6),
            "iqr": round(self.iqr, 6),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
        }


def quantile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank quantile: an observed sample, never an interpolation."""

    if not values:
        raise ValueError("a quantile requires at least one sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: Sequence[float]) -> Summary:
    """Summarize one sample set without discarding its spread."""

    if not values:
        raise ValueError("a summary requires at least one sample")
    return Summary(
        count=len(values),
        median=statistics.median(values),
        q1=quantile(values, 0.25),
        q3=quantile(values, 0.75),
        minimum=min(values),
        maximum=max(values),
    )


def relative_percent(value: float, reference: float) -> float:
    """`value` as a percentage of `reference`; undefined at a zero reference."""

    if reference == 0.0:
        raise ValueError("a relative percentage requires a nonzero reference")
    return 100.0 * value / reference


def spread_percent(values: Sequence[float]) -> float:
    """Peak-to-peak spread of a sample set as a percentage of its median."""

    summary = summarize(values)
    if summary.median == 0.0:
        raise ValueError("a spread percentage requires a nonzero median")
    return relative_percent(summary.maximum - summary.minimum, summary.median)


# --------------------------------------------------------------------------
# Wire-time floor. Pure.
# --------------------------------------------------------------------------


def payload_bytes(shape: Sequence[int], element_bytes: int) -> int:
    """Bytes in one collective's payload buffer on one rank."""

    if element_bytes <= 0:
        raise ValueError("element_bytes must be positive")
    if not shape:
        raise ValueError("shape must contain at least one dimension")
    total = element_bytes
    for dimension in shape:
        if dimension < 0:
            raise ValueError("shape dimensions must be nonnegative")
        total *= dimension
    return total


def floor_microseconds(
    payload: int,
    wire_bytes_multiplier: float,
    rate_gbit_per_second: float,
) -> float:
    """Time to move one instance's wire bytes at a stated link rate.

    A lower bound on that instance's transport time and nothing more. It
    assumes the multiplier the caller states is the algorithm's actual
    per-rank traffic, that the link sustains the stated rate at this payload,
    and that no per-call latency exists. All three are false at small
    payloads, where the floor is dominated by fixed cost rather than bytes.
    """

    if payload < 0:
        raise ValueError("payload bytes must be nonnegative")
    if wire_bytes_multiplier <= 0.0:
        raise ValueError("wire_bytes_multiplier must be positive")
    if rate_gbit_per_second <= 0.0:
        raise ValueError("rate_gbit_per_second must be positive")
    bits = payload * wire_bytes_multiplier * BITS_PER_BYTE
    return bits / (rate_gbit_per_second * GBIT) * MICROSECONDS_PER_SECOND


# --------------------------------------------------------------------------
# Campaign sizing. Pure.
# --------------------------------------------------------------------------


def required_repetitions(
    dispersion_percent: float,
    detect_percent: float,
    multiplier: float = Z_95,
) -> int:
    """Paired repetitions whose interval half-width reaches a target.

    A normal approximation over paired differences: the half-width of a
    two-sided interval on the mean paired difference falls as the dispersion
    over the square root of the count. It sizes a campaign that has not yet
    collected anything. Once samples exist, a bootstrap over the observed
    paired differences replaces it, and the two can disagree.
    """

    if dispersion_percent < 0.0:
        raise ValueError("dispersion_percent must be nonnegative")
    if detect_percent <= 0.0:
        raise ValueError("detect_percent must be positive")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")
    needed = math.ceil((multiplier * dispersion_percent / detect_percent) ** 2)
    return max(MINIMUM_REPETITIONS, int(needed))


def verdict(
    difference: float,
    reference: float,
    detect_percent: float,
) -> tuple[str, float]:
    """Classify a difference against a threshold, refusing anything smaller.

    Returns the verdict and the difference as a percentage of the reference.
    `indeterminate` is a result: it states that this instrument, at this
    repetition count, cannot separate the difference from run-to-run
    variation.
    """

    if detect_percent <= 0.0:
        raise ValueError("detect_percent must be positive")
    percent = relative_percent(difference, reference)
    if abs(percent) < detect_percent:
        return "indeterminate", percent
    return ("higher" if percent > 0.0 else "lower"), percent


# --------------------------------------------------------------------------
# Exposure slope. Pure.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureFit:
    """Least-squares slope of wall time against injected collective delay."""

    points: int
    delay_span_us: float
    baseline_wall_us: float
    slope: float
    intercept_us: float
    max_residual_us: float
    determinate: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "delay_span_us": round(self.delay_span_us, 6),
            "baseline_wall_us": round(self.baseline_wall_us, 6),
            "slope": round(self.slope, 6),
            "intercept_us": round(self.intercept_us, 6),
            "max_residual_us": round(self.max_residual_us, 6),
            "determinate": self.determinate,
            "reason": self.reason,
        }


def fit_exposure(
    points: Sequence[tuple[float, float]],
    detect_percent: float,
) -> ExposureFit:
    """Fit the fraction of a marginal collective microsecond that reaches wall.

    A slope near one means the delayed region is fully exposed on the
    critical path; a slope near zero means it is fully overlapped by work on
    another stream. The fit is valid only over the delays actually injected,
    which are nonnegative. Reading it as the saving that removing the
    collective entirely would produce extrapolates outside that range and
    assumes the exposure stays linear down to zero.
    """

    if len(points) < 3:
        raise ValueError("an exposure fit requires at least three delay points")
    delays = [delay for delay, _wall in points]
    walls = [wall for _delay, wall in points]
    if min(delays) < 0.0:
        raise ValueError("injected delays must be nonnegative")
    baseline_candidates = [wall for delay, wall in points if delay == min(delays)]
    baseline = statistics.median(baseline_candidates)

    span = max(delays) - min(delays)
    mean_delay = statistics.fmean(delays)
    mean_wall = statistics.fmean(walls)
    covariance = sum(
        (delay - mean_delay) * (wall - mean_wall) for delay, wall in points
    )
    variance = sum((delay - mean_delay) ** 2 for delay in delays)
    if variance == 0.0:
        raise ValueError("every delay point is identical; no slope exists")
    slope = covariance / variance
    intercept = mean_wall - slope * mean_delay
    residual = max(abs(wall - (slope * delay + intercept)) for delay, wall in points)

    # A sweep whose largest delay is inside the noise floor moves nothing the
    # instrument can see, so its slope describes noise rather than exposure.
    minimum_span = baseline * detect_percent / 100.0
    if span < minimum_span:
        return ExposureFit(
            points=len(points),
            delay_span_us=span,
            baseline_wall_us=baseline,
            slope=slope,
            intercept_us=intercept,
            max_residual_us=residual,
            determinate=False,
            reason=(
                f"delay span {span:.1f} us is below {minimum_span:.1f} us, "
                f"which is {detect_percent:.1f}% of the undelayed median wall "
                "time; the sweep cannot separate exposure from noise"
            ),
        )
    return ExposureFit(
        points=len(points),
        delay_span_us=span,
        baseline_wall_us=baseline,
        slope=slope,
        intercept_us=intercept,
        max_residual_us=residual,
        determinate=True,
        reason="delay span exceeds the detection threshold",
    )


# --------------------------------------------------------------------------
# Capture documents.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceKey:
    """What makes two records across arms and ranks the same collective."""

    family: str
    communicator: str
    world_size: int
    shape: tuple[int, ...]
    element_bytes: int
    step: int
    ordinal: int

    @property
    def label(self) -> str:
        dimensions = "x".join(str(value) for value in self.shape)
        return (
            f"{self.family}|{self.communicator}|w{self.world_size}"
            f"|{dimensions}|e{self.element_bytes}"
            f"|s{self.step}|o{self.ordinal}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "communicator": self.communicator,
            "world_size": self.world_size,
            "shape": list(self.shape),
            "element_bytes": self.element_bytes,
            "step": self.step,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class Instance:
    """One collective occurrence, timed on every rank that took part."""

    key: InstanceKey
    wire_bytes_multiplier: float
    multiplier_basis: str
    occurrences: int
    residency_us: Mapping[str, tuple[float, ...]]
    gate_first_us: Mapping[str, tuple[float, ...]]
    gate_second_us: Mapping[str, tuple[float, ...]]

    @property
    def payload_bytes(self) -> int:
        return payload_bytes(self.key.shape, self.key.element_bytes)


@dataclass(frozen=True)
class Capture:
    """One arm of one session."""

    arm: str
    layer: str
    session: str
    rate_gbit_per_second: float
    rate_basis: str
    instances: tuple[Instance, ...]

    @property
    def keys(self) -> tuple[InstanceKey, ...]:
        return tuple(instance.key for instance in self.instances)


def _require(document: Mapping[str, Any], field: str, where: str) -> Any:
    if field not in document:
        raise DocumentInvalid(f"{where} is missing required field {field!r}")
    return document[field]


def _require_number(value: Any, field: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DocumentInvalid(f"{where} field {field!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise DocumentInvalid(f"{where} field {field!r} must be finite")
    return number


def _require_positive_int(value: Any, field: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DocumentInvalid(f"{where} field {field!r} must be a positive integer")
    return value


def _samples(value: Any, field: str, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise DocumentInvalid(f"{where} field {field!r} must be a nonempty list")
    out: list[float] = []
    for entry in value:
        number = _require_number(entry, field, where)
        if number < 0.0:
            raise DocumentInvalid(f"{where} field {field!r} must be nonnegative")
        out.append(number)
    return tuple(out)


def parse_instance(document: Mapping[str, Any], arm: str, index: int) -> Instance:
    """Build one instance record, refusing anything it would have to invent."""

    where = f"instance {index}"
    if not isinstance(document, dict):
        raise DocumentInvalid(f"{where} must be an object")
    shape_field = _require(document, "shape", where)
    if not isinstance(shape_field, list) or not shape_field:
        raise DocumentInvalid(f"{where} field 'shape' must be a nonempty list")
    shape: list[int] = []
    for dimension in shape_field:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise DocumentInvalid(f"{where} shape dimensions must be integers")
        if dimension <= 0:
            raise DocumentInvalid(f"{where} shape dimensions must be positive")
        shape.append(dimension)

    family = _require(document, "family", where)
    communicator = _require(document, "communicator", where)
    for name, value in (("family", family), ("communicator", communicator)):
        if not isinstance(value, str) or not value:
            raise DocumentInvalid(f"{where} field {name!r} must be a nonempty string")

    key = InstanceKey(
        family=family,
        communicator=communicator,
        world_size=_require_positive_int(
            _require(document, "world_size", where), "world_size", where
        ),
        shape=tuple(shape),
        element_bytes=_require_positive_int(
            _require(document, "element_bytes", where), "element_bytes", where
        ),
        step=_require_positive_int(_require(document, "step", where), "step", where),
        ordinal=_require_positive_int(
            _require(document, "ordinal", where), "ordinal", where
        ),
    )

    multiplier = _require_number(
        _require(document, "wire_bytes_multiplier", where),
        "wire_bytes_multiplier",
        where,
    )
    if multiplier <= 0.0:
        raise DocumentInvalid(f"{where} field 'wire_bytes_multiplier' must be positive")
    basis = _require(document, "multiplier_basis", where)
    if not isinstance(basis, str) or not basis:
        raise DocumentInvalid(
            f"{where} field 'multiplier_basis' must name the algorithm whose "
            "per-rank traffic the multiplier states"
        )
    occurrences = _require_positive_int(
        document.get("occurrences", 1), "occurrences", where
    )

    ranks = _require(document, "ranks", where)
    if not isinstance(ranks, dict) or not ranks:
        raise DocumentInvalid(f"{where} field 'ranks' must be a nonempty object")
    if len(ranks) != key.world_size:
        raise DocumentInvalid(
            f"{where} reports {len(ranks)} ranks for world_size {key.world_size}"
        )

    residency: dict[str, tuple[float, ...]] = {}
    gate_first: dict[str, tuple[float, ...]] = {}
    gate_second: dict[str, tuple[float, ...]] = {}
    for rank, record in sorted(ranks.items()):
        rank_where = f"{where} rank {rank}"
        if not isinstance(record, dict):
            raise DocumentInvalid(f"{rank_where} must be an object")
        residency[rank] = _samples(
            _require(record, "residency_us", rank_where), "residency_us", rank_where
        )
        if arm == NAKED_ARM:
            for forbidden in ("gate_first_us", "gate_second_us"):
                if forbidden in record:
                    raise DocumentInvalid(
                        f"{rank_where} carries {forbidden!r}; a naked arm has "
                        "no gate to time"
                    )
            continue
        gate_first[rank] = _samples(
            _require(record, "gate_first_us", rank_where), "gate_first_us", rank_where
        )
        gate_second[rank] = _samples(
            _require(record, "gate_second_us", rank_where), "gate_second_us", rank_where
        )
        counts = {
            len(residency[rank]),
            len(gate_first[rank]),
            len(gate_second[rank]),
        }
        if len(counts) != 1:
            raise DocumentInvalid(
                f"{rank_where} records different sample counts for the two "
                "gates and the collective; they are recorded per iteration and "
                "must agree"
            )

    return Instance(
        key=key,
        wire_bytes_multiplier=multiplier,
        multiplier_basis=basis,
        occurrences=occurrences,
        residency_us=residency,
        gate_first_us=gate_first,
        gate_second_us=gate_second,
    )


def parse_capture(document: Any) -> Capture:
    """Validate one capture document against the schema it declares."""

    if not isinstance(document, dict):
        raise DocumentInvalid("a capture document must be a JSON object")
    schema = document.get("schema")
    if schema != CAPTURE_SCHEMA:
        raise DocumentInvalid(
            f"schema is {schema!r}; this reads {CAPTURE_SCHEMA!r} only"
        )
    arm = document.get("arm")
    if arm not in ARMS:
        raise DocumentInvalid(f"arm must be one of {ARMS}; got {arm!r}")
    layer = document.get("layer")
    if layer not in LAYERS:
        raise DocumentInvalid(f"layer must be one of {LAYERS}; got {layer!r}")
    session = document.get("session")
    if not isinstance(session, str) or not session:
        raise DocumentInvalid("session must be a nonempty string")

    link = _require(document, "link", "document")
    if not isinstance(link, dict):
        raise DocumentInvalid("link must be an object")
    rate = _require_number(
        _require(link, "rate_gbit_per_second", "link"),
        "rate_gbit_per_second",
        "link",
    )
    if rate <= 0.0:
        raise DocumentInvalid("link rate_gbit_per_second must be positive")
    basis = link.get("rate_basis")
    if basis not in RATE_BASES:
        raise DocumentInvalid(
            f"link rate_basis must be one of {RATE_BASES}; got {basis!r}. A "
            "floor derived from a nameplate rate is not a floor derived from "
            "a rate this fabric was observed to reach."
        )

    instance_field = _require(document, "instances", "document")
    if not isinstance(instance_field, list) or not instance_field:
        raise DocumentInvalid("instances must be a nonempty list")
    instances = tuple(
        parse_instance(entry, arm, index)
        for index, entry in enumerate(instance_field)
    )
    seen: set[str] = set()
    for instance in instances:
        if instance.key.label in seen:
            raise DocumentInvalid(
                f"instance {instance.key.label} appears twice; one occurrence "
                "of a collective is one instance, and repeat counts belong in "
                "'occurrences'"
            )
        seen.add(instance.key.label)
    return Capture(
        arm=arm,
        layer=layer,
        session=session,
        rate_gbit_per_second=rate,
        rate_basis=basis,
        instances=instances,
    )


def parse_exposure(document: Any) -> tuple[str, tuple[tuple[float, float], ...]]:
    """Validate an injected-delay sweep and reduce each point to its median."""

    if not isinstance(document, dict):
        raise DocumentInvalid("an exposure document must be a JSON object")
    if document.get("schema") != EXPOSURE_SCHEMA:
        raise DocumentInvalid(
            f"schema is {document.get('schema')!r}; this reads "
            f"{EXPOSURE_SCHEMA!r} only"
        )
    layer = document.get("layer")
    if layer not in LAYERS:
        raise DocumentInvalid(f"layer must be one of {LAYERS}; got {layer!r}")
    samples = _require(document, "samples", "document")
    if not isinstance(samples, list) or len(samples) < 3:
        raise DocumentInvalid("samples must list at least three delay points")
    points: list[tuple[float, float]] = []
    for index, entry in enumerate(samples):
        where = f"sample {index}"
        if not isinstance(entry, dict):
            raise DocumentInvalid(f"{where} must be an object")
        delay = _require_number(
            _require(entry, "injected_delay_us", where), "injected_delay_us", where
        )
        if delay < 0.0:
            raise DocumentInvalid(f"{where} injected_delay_us must be nonnegative")
        walls = _samples(_require(entry, "wall_us", where), "wall_us", where)
        points.append((delay, statistics.median(walls)))
    if len({delay for delay, _wall in points}) < 3:
        raise DocumentInvalid("samples must cover at least three distinct delays")
    return layer, tuple(points)


# --------------------------------------------------------------------------
# Attribution.
# --------------------------------------------------------------------------


def skew_microseconds(
    gate_first: Sequence[float], gate_second: Sequence[float]
) -> float:
    """Arrival skew a first gate absorbed, with the gate's own cost removed.

    The second gate is entered by every rank at nearly the same instant, so
    its residency is the gate's own cost carrying almost no incoming skew.
    Subtracting it from the first gate's residency removes the instrument
    from its own reading. The second gate still absorbs whatever exit skew
    the first gate produced, so it overstates the gate's cost and this
    difference is a lower bound on arrival skew, not an estimate of it.
    """

    difference = statistics.median(gate_first) - statistics.median(gate_second)
    return max(0.0, difference)


def attribute_instance(
    instance: Instance,
    rate_gbit_per_second: float,
    naked: Instance | None,
) -> dict[str, Any]:
    """Per-rank rows and one instance summary, in microseconds."""

    floor_us = floor_microseconds(
        instance.payload_bytes, instance.wire_bytes_multiplier, rate_gbit_per_second
    )
    rows: dict[str, Any] = {}
    gated_medians: list[float] = []
    for rank, samples in sorted(instance.residency_us.items()):
        transport = summarize(samples)
        gated_medians.append(transport.median)
        row: dict[str, Any] = {"transport_us": transport.to_dict()}
        if instance.gate_first_us:
            row["gate_first_us"] = summarize(instance.gate_first_us[rank]).to_dict()
            row["gate_second_us"] = summarize(instance.gate_second_us[rank]).to_dict()
            row["skew_us"] = round(
                skew_microseconds(
                    instance.gate_first_us[rank], instance.gate_second_us[rank]
                ),
                6,
            )
        if naked is not None and rank in naked.residency_us:
            residency = summarize(naked.residency_us[rank])
            row["residency_us"] = residency.to_dict()
            row["residency_minus_transport_us"] = round(
                residency.median - transport.median, 6
            )
        rows[rank] = row

    gate_note = "no gate in this arm"
    gate_ok = True
    if instance.gate_second_us:
        gate_cost = max(
            statistics.median(samples) for samples in instance.gate_second_us.values()
        )
        slowest = max(gated_medians)
        if slowest > 0.0 and gate_cost > GATE_COST_MAX_FRACTION * slowest:
            gate_ok = False
            gate_note = (
                f"gate cost {gate_cost:.3f} us exceeds "
                f"{GATE_COST_MAX_FRACTION:.0%} of the gated collective "
                f"{slowest:.3f} us; the gate is not negligible at this payload"
            )
        else:
            gate_note = f"gate cost {gate_cost:.3f} us"

    spread = spread_percent(gated_medians) if len(gated_medians) > 1 else 0.0
    ranks_agree = spread <= CROSS_RANK_SPREAD_MAX_PERCENT
    return {
        "key": instance.key.to_dict(),
        "label": instance.key.label,
        "payload_bytes": instance.payload_bytes,
        "wire_bytes_multiplier": instance.wire_bytes_multiplier,
        "multiplier_basis": instance.multiplier_basis,
        "occurrences": instance.occurrences,
        "floor_us": round(floor_us, 6),
        "slowest_rank_transport_us": round(max(gated_medians), 6),
        "cross_rank_spread_percent": round(spread, 3),
        "cross_rank_agreement": ranks_agree,
        "gate_cost_acceptable": gate_ok,
        "gate_note": gate_note,
        "ranks": rows,
    }


def build_report(
    gated: Capture,
    naked: Capture,
    detect_percent: float,
    exposure: ExposureFit | None,
) -> dict[str, Any]:
    """Assemble the narrowed interval and every gate that qualifies it."""

    naked_by_key = {instance.key: instance for instance in naked.instances}
    rows = [
        attribute_instance(
            instance, gated.rate_gbit_per_second, naked_by_key.get(instance.key)
        )
        for instance in gated.instances
    ]

    floor_us = 0.0
    transport_us = 0.0
    residency_us = 0.0
    for instance, row in zip(gated.instances, rows):
        floor_us += instance.occurrences * row["floor_us"]
        transport_us += instance.occurrences * row["slowest_rank_transport_us"]
        matched = naked_by_key[instance.key]
        residency_us += instance.occurrences * max(
            statistics.median(samples) for samples in matched.residency_us.values()
        )

    removed_us = residency_us - transport_us
    removed_verdict, removed_percent = verdict(removed_us, residency_us, detect_percent)

    failures = [row["label"] for row in rows if not row["cross_rank_agreement"]]
    gate_failures = [row["label"] for row in rows if not row["gate_cost_acceptable"]]

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "sessions": {"gated": gated.session, "naked": naked.session},
        "layer": gated.layer,
        "detect_percent": detect_percent,
        "detect_floor_percent": DETECT_FLOOR_PERCENT[gated.layer],
        "link": {
            "rate_gbit_per_second": gated.rate_gbit_per_second,
            "rate_basis": gated.rate_basis,
        },
        "floor_formula": FLOOR_FORMULA,
        "instances": rows,
        "totals_seconds": {
            "floor": round(floor_us / MICROSECONDS_PER_SECOND, 6),
            "transport_ceiling": round(transport_us / MICROSECONDS_PER_SECOND, 6),
            "residency_ceiling": round(residency_us / MICROSECONDS_PER_SECOND, 6),
            "removed_by_gating": round(removed_us / MICROSECONDS_PER_SECOND, 6),
        },
        "removed_by_gating": {
            "percent_of_residency_ceiling": round(removed_percent, 3),
            "verdict": removed_verdict,
        },
        "validity": {
            "cross_rank_agreement": not failures,
            "cross_rank_failures": failures,
            "gate_cost_acceptable": not gate_failures,
            "gate_cost_failures": gate_failures,
            "rate_basis_measured": gated.rate_basis == "measured",
        },
    }
    if exposure is not None:
        exposed_us = exposure.slope * transport_us if exposure.determinate else None
        report["exposure"] = exposure.to_dict()
        report["exposure"]["exposed_transport_seconds"] = (
            None
            if exposed_us is None
            else round(exposed_us / MICROSECONDS_PER_SECOND, 6)
        )
        report["exposure"]["extrapolation_note"] = (
            "The slope is fitted over nonnegative injected delays. Applying it "
            "to the whole transport ceiling assumes the exposure stays linear "
            "down to a collective that costs nothing, which no injected delay "
            "observes."
        )
    return report


def require_comparable(gated: Capture, naked: Capture) -> None:
    """Refuse two captures that do not describe the same inventory."""

    if gated.arm != GATED_ARM:
        raise NotComparable(f"--gated names a {gated.arm!r} arm")
    if naked.arm != NAKED_ARM:
        raise NotComparable(f"--naked names a {naked.arm!r} arm")
    if gated.layer != naked.layer:
        raise NotComparable(
            f"layers differ: {gated.layer!r} against {naked.layer!r}"
        )
    if gated.session == naked.session:
        raise NotComparable(
            "both arms declare session "
            f"{gated.session!r}; a paired comparison needs two collections"
        )
    if gated.rate_gbit_per_second != naked.rate_gbit_per_second:
        raise NotComparable("the two arms state different link rates")
    missing = sorted(
        key.label for key in set(gated.keys) - set(naked.keys)
    )
    extra = sorted(key.label for key in set(naked.keys) - set(gated.keys))
    if missing or extra:
        raise NotComparable(
            "the two arms cover different collective inventories; "
            f"only in gated: {missing or 'none'}; only in naked: "
            f"{extra or 'none'}"
        )


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------


def _line(label: str, value: str) -> str:
    return f"  {label:<34}{value}\n"


def render_report(report: Mapping[str, Any]) -> str:
    """A report that carries its own scope, not only its numbers."""

    totals = report["totals_seconds"]
    validity = report["validity"]
    out = ["Collective critical-path attribution\n\n"]
    out.append(
        _line("gated session", str(report["sessions"]["gated"]))
        + _line("naked session", str(report["sessions"]["naked"]))
        + _line("layer", str(report["layer"]))
        + _line(
            "link rate",
            f"{report['link']['rate_gbit_per_second']:.1f} Gb/s "
            f"({report['link']['rate_basis']})",
        )
        + _line("detection threshold", f"{report['detect_percent']:.1f}% of median")
    )
    out.append("\nPer-request totals, seconds\n")
    out.append(
        _line("wire-time floor", f"{totals['floor']:.6f}")
        + _line("transport ceiling (gated)", f"{totals['transport_ceiling']:.6f}")
        + _line("residency ceiling (naked)", f"{totals['residency_ceiling']:.6f}")
        + _line(
            "removed by gating",
            f"{totals['removed_by_gating']:.6f} "
            f"({report['removed_by_gating']['percent_of_residency_ceiling']:.1f}%, "
            f"{report['removed_by_gating']['verdict']})",
        )
    )
    out.append(f"\n  {FLOOR_FORMULA}\n")

    out.append("\nPer instance\n")
    for row in report["instances"]:
        out.append(f"  {row['label']}\n")
        out.append(
            f"    payload {row['payload_bytes']} B x {row['wire_bytes_multiplier']} "
            f"({row['multiplier_basis']}), {row['occurrences']} occurrence(s)\n"
        )
        out.append(
            f"    floor {row['floor_us']:.3f} us; slowest-rank transport "
            f"{row['slowest_rank_transport_us']:.3f} us; cross-rank spread "
            f"{row['cross_rank_spread_percent']:.1f}%\n"
        )
        out.append(f"    {row['gate_note']}\n")
        for rank, values in row["ranks"].items():
            transport = values["transport_us"]
            piece = (
                f"    rank {rank}: transport median {transport['median']:.3f} us "
                f"IQR {transport['iqr']:.3f} n={transport['count']}"
            )
            if "skew_us" in values:
                piece += f"; skew >= {values['skew_us']:.3f} us"
            if "residency_us" in values:
                piece += (
                    f"; naked residency {values['residency_us']['median']:.3f} us"
                )
            out.append(piece + "\n")

    if "exposure" in report:
        exposure = report["exposure"]
        out.append("\nExposure\n")
        out.append(
            _line("slope (wall us per delayed us)", f"{exposure['slope']:.4f}")
            + _line("delay span", f"{exposure['delay_span_us']:.1f} us")
            + _line("determinate", str(exposure["determinate"]))
            + _line("reason", str(exposure["reason"]))
        )
        exposed = exposure["exposed_transport_seconds"]
        out.append(
            _line(
                "exposed transport, seconds",
                "not reported" if exposed is None else f"{exposed:.6f}",
            )
        )
        out.append(f"  {exposure['extrapolation_note']}\n")

    out.append("\nValidity\n")
    out.append(
        _line("cross-rank agreement", str(validity["cross_rank_agreement"]))
        + _line("gate cost acceptable", str(validity["gate_cost_acceptable"]))
        + _line("link rate measured", str(validity["rate_basis_measured"]))
    )
    for label in validity["cross_rank_failures"]:
        out.append(f"    cross-rank spread too wide: {label}\n")
    for label in validity["gate_cost_failures"]:
        out.append(f"    gate cost not negligible: {label}\n")

    out.append(
        "\nScope: every number above is arithmetic over timings another "
        "instrument recorded. The transport ceiling is a sum of per-instance "
        "slowest-rank medians, so it bounds the transport's contribution and "
        "does not identify a critical path through the four ranks.\n"
    )
    return "".join(out)


def render_plan(
    dispersion_percent: float, detect_percent: float, repetitions: int
) -> str:
    return (
        "Paired repetition count\n\n"
        + _line("stated dispersion", f"{dispersion_percent:.2f}% of median")
        + _line("target detection threshold", f"{detect_percent:.2f}% of median")
        + _line("interval multiplier", f"{Z_95} (two-sided 95%, normal)")
        + _line("required repetitions", str(repetitions))
        + "\n  n >= (multiplier * dispersion / detect)^2, floored at "
        f"{MINIMUM_REPETITIONS}\n\n"
        "  This sizes a campaign before it has samples. Once paired "
        "differences exist, a bootstrap over them replaces this count, and "
        "the two can disagree.\n"
    )


def example_documents() -> dict[str, Any]:
    """Schema-valid documents, so a probe author can target the shape."""

    def ranks(base: float, gated: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for rank in range(4):
            record: dict[str, Any] = {
                "residency_us": [base + rank * 0.5 + step for step in (0.0, 0.4, 0.8)]
            }
            if gated:
                record["gate_first_us"] = [
                    3.0 + rank * 4.0 + step for step in (0.0, 0.2, 0.4)
                ]
                record["gate_second_us"] = [2.8, 3.0, 3.2]
            out[str(rank)] = record
        return out

    instance = {
        "family": "tp4_all_reduce",
        "communicator": "tp:0",
        "world_size": 4,
        "shape": [512, 4096],
        "element_bytes": 2,
        "step": 1,
        "ordinal": 1,
        "wire_bytes_multiplier": 1.5,
        "multiplier_basis": "ring all-reduce, 2(N-1)/N per rank at N=4",
        "occurrences": 87,
    }
    return {
        "gated": {
            "schema": CAPTURE_SCHEMA,
            "arm": GATED_ARM,
            "layer": DEVICE_LAYER,
            "session": "gated-session-label",
            "link": {"rate_gbit_per_second": 200.0, "rate_basis": "nameplate"},
            "instances": [{**instance, "ranks": ranks(310.0, True)}],
        },
        "naked": {
            "schema": CAPTURE_SCHEMA,
            "arm": NAKED_ARM,
            "layer": DEVICE_LAYER,
            "session": "naked-session-label",
            "link": {"rate_gbit_per_second": 200.0, "rate_basis": "nameplate"},
            "instances": [{**instance, "ranks": ranks(372.0, False)}],
        },
        "exposure": {
            "schema": EXPOSURE_SCHEMA,
            "layer": END_TO_END_LAYER,
            "samples": [
                {"injected_delay_us": 0.0, "wall_us": [5_687_000.0, 5_690_000.0]},
                {"injected_delay_us": 600_000.0, "wall_us": [6_003_000.0]},
                {"injected_delay_us": 1_200_000.0, "wall_us": [6_320_000.0]},
            ],
        },
    }


# --------------------------------------------------------------------------
# Command line.
# --------------------------------------------------------------------------


def load_document(path: str) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"{path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise DocumentInvalid(f"{path} is not valid JSON: {error}") from error


def emit_json(document: Mapping[str, Any], destination: str) -> None:
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if destination == "-":
        sys.stdout.write(text)
        return
    Path(destination).write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Narrow a collective's critical-path share from a gated and a "
            "naked timing arm."
        )
    )
    parser.add_argument(
        "--gated",
        metavar="PATH",
        help="capture document for the arm that gates each timed collective",
    )
    parser.add_argument(
        "--naked",
        metavar="PATH",
        help="capture document for the arm that gates nothing",
    )
    parser.add_argument(
        "--exposure",
        metavar="PATH",
        help="injected-delay sweep whose slope is the exposed fraction",
    )
    parser.add_argument(
        "--detect-percent",
        type=float,
        help=(
            "smallest difference reportable as an effect, as a percentage of "
            "the compared median; defaults to and may not fall below the "
            "layer floor"
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write the report as JSON to PATH, or to stdout for '-'",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="compute a paired repetition count and read no capture",
    )
    parser.add_argument(
        "--dispersion-percent",
        type=float,
        help="with --plan: observed paired-difference dispersion, in percent",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="print schema-valid example documents and read nothing",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.print_schema:
        return arguments
    if arguments.plan:
        if arguments.dispersion_percent is None or arguments.detect_percent is None:
            parser.error("--plan requires --dispersion-percent and --detect-percent")
        if arguments.dispersion_percent < 0.0:
            parser.error("--dispersion-percent must be nonnegative")
        if arguments.detect_percent <= 0.0:
            parser.error("--detect-percent must be positive")
        return arguments
    if not arguments.gated or not arguments.naked:
        parser.error("--gated and --naked are both required")
    return arguments


class ThresholdRefused(ValueError):
    """A requested threshold falls below the floor its layer fixes."""


def resolve_detect_percent(layer: str, requested: float | None) -> float:
    """Apply a layer's threshold floor, refusing any request below it."""

    if layer not in DETECT_FLOOR_PERCENT:
        raise ValueError(f"unknown layer {layer!r}")
    floor = DETECT_FLOOR_PERCENT[layer]
    if requested is None:
        return floor
    if requested < floor:
        raise ThresholdRefused(
            f"--detect-percent {requested} is below the {floor} floor this "
            f"module fixes for the {layer!r} layer; a smaller difference is "
            "not separable from run-to-run variation"
        )
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)

    if arguments.print_schema:
        print(json.dumps(example_documents(), indent=2, sort_keys=True))
        return EXIT_OK

    if arguments.plan:
        repetitions = required_repetitions(
            arguments.dispersion_percent, arguments.detect_percent
        )
        print(
            render_plan(
                arguments.dispersion_percent, arguments.detect_percent, repetitions
            ),
            end="",
        )
        return EXIT_OK

    try:
        gated = parse_capture(load_document(arguments.gated))
        naked = parse_capture(load_document(arguments.naked))
    except FileNotFoundError as error:
        print(f"FAIL input unavailable: {error}", file=sys.stderr)
        return EXIT_INPUT_MISSING
    except DocumentInvalid as error:
        print(f"FAIL invalid document: {error}", file=sys.stderr)
        return EXIT_INVALID_DOCUMENT

    try:
        require_comparable(gated, naked)
    except NotComparable as error:
        print(f"FAIL not comparable: {error}", file=sys.stderr)
        return EXIT_NOT_COMPARABLE

    try:
        detect_percent = resolve_detect_percent(gated.layer, arguments.detect_percent)
    except ThresholdRefused as error:
        print(f"FAIL threshold refused: {error}", file=sys.stderr)
        return EXIT_NOT_COMPARABLE

    fit: ExposureFit | None = None
    if arguments.exposure:
        try:
            exposure_layer, points = parse_exposure(load_document(arguments.exposure))
            fit = fit_exposure(points, resolve_detect_percent(exposure_layer, None))
        except FileNotFoundError as error:
            print(f"FAIL input unavailable: {error}", file=sys.stderr)
            return EXIT_INPUT_MISSING
        except (DocumentInvalid, ValueError) as error:
            print(f"FAIL invalid exposure sweep: {error}", file=sys.stderr)
            return EXIT_INVALID_DOCUMENT

    report = build_report(gated, naked, detect_percent, fit)
    print(render_report(report), end="")
    if arguments.json:
        emit_json(report, arguments.json)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
