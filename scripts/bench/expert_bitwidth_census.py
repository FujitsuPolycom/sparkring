#!/usr/bin/env python3
"""Measure the bit-width distribution a quantized checkpoint actually stores.

A mixed-precision quantizer reaches a target average bit rate by giving
different tensors different bit widths, so one declared average describes no
individual weight matrix. A cost model that multiplies a weight count by a
single uniform bit rate — a per-expert transfer size, for instance — asserts a
distribution it has not observed. This reads a checkpoint's own metadata and
reports the distribution that is stored: per tensor, per name class, and
aggregated over the MoE routed-expert tensors that dominate expert transfer.

Safety class: OFFLINE. It opens files under the named model directory for
reading, parses the safetensors header of each shard, and reads the JSON
metadata files beside them. It contacts no host, starts no runtime, imports
neither torch nor the safetensors package, and writes nothing. Tensor payload
bytes are never read: a shard is touched only for its header prefix, so a
checkpoint of hundreds of gigabytes costs kilobytes of I/O.

Two quantities are reported side by side and are not interchangeable.

- Declared bit rate: what the checkpoint's quantization metadata states.
- Measured bit rate: stored payload bits divided by the logical weight count
  implied by the stored geometry.

They are independent observations. When they disagree, that disagreement is
the result; this reports it and reconciles nothing.

Limits of a header-only census, which apply to every number it prints:

- Only geometry derivable from a tensor's name, dtype, and shape is derivable
  at all. A packing whose logical shape lives in the payload, or in metadata
  this does not recognize, is reported as undetermined rather than guessed,
  and undetermined bytes are excluded from every average.
- A measured bit rate is a storage property. It states how many bits per
  logical weight a checkpoint spends, not what precision a kernel computes in
  and not what error the quantizer incurred.
- Sidecar tensors — per-channel scales, sign vectors, codebook sentinels —
  carry no logical weights of their own but must move whenever their matrix
  moves. Expert averages are therefore reported both without and with sidecar
  bytes; a transfer-cost model wants the second.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "sparkring-expert-bitwidth-census/v1"

# The safetensors container: 8 bytes of little-endian header length, then that
# many bytes of JSON, then the tensor payload. A header is metadata for at most
# a few hundred thousand tensors, so a length beyond this bound means the file
# is not a safetensors container rather than that it is a large one.
HEADER_LENGTH_BYTES = 8
MAX_HEADER_BYTES = 256 * 1024 * 1024

# Bits occupied by one stored element of each safetensors dtype. BOOL occupies
# a whole byte in this container despite naming one bit of information.
DTYPE_BITS: Mapping[str, int] = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
}

# dtypes that store one logical weight per stored element. For these the
# element count is the product of the declared shape and no packing has to be
# reasoned about.
UNPACKED_DTYPES = frozenset({"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"})

# EXL3 stores a quantized matrix as an int16 trellis of shape
# [K/16, N/16, 16*bits] beside float16 input/output rotation vectors (suh/svh,
# or the packed sign bitfields su/sv) and an int32 codebook sentinel (mcg or
# mul1). `runtime/exl3/overlay/vllm/model_executor/layers/quantization/exl3.py`
# is the executable statement of that geometry and validates it at load time.
EXL3_TRELLIS_SUFFIX = "trellis"
EXL3_BLOCK = 16
EXL3_MIN_BITS = 1
EXL3_MAX_BITS = 8

# Final name segments that carry quantization apparatus rather than weights.
SIDECAR_SUFFIXES = frozenset(
    {
        "suh",
        "svh",
        "su",
        "sv",
        "mcg",
        "mul1",
        "scale",
        "scales",
        "scale_inv",
        "weight_scale",
        "weight_scale_inv",
        "weight_scale_2",
        "input_scale",
        "act_scale",
        "zero_point",
        "zeros",
        "g_idx",
    }
)

# Name classification, applied in a fixed order; the first rule that matches
# owns the tensor. Order matters: a shared expert is named with the same
# `experts` stem as a routed one, so it is excluded before the routed-expert
# rule is tried.
EMBEDDING_RE = re.compile(r"(?:^|\.)(embed_tokens|embed_out|lm_head|wte|wpe)(?:\.|$)")
SHARED_EXPERT_RE = re.compile(r"(?:^|\.)shared_experts?(?:\.|$)")
ROUTED_EXPERT_RE = re.compile(r"(?:^|\.)experts\.(?P<expert>\d+)(?:\.|$)")
ATTENTION_RE = re.compile(r"(?:^|\.)(self_attn|attention|attn)(?:\.|$)")
DENSE_RE = re.compile(
    r"(?:^|\.)(mlp|gate|feed_forward|ffn|[a-z_]*norm[a-z_]*)(?:\.|$)"
)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)(?:\.|$)")

CLASS_EXPERT = "expert"
CLASS_ATTENTION = "attention"
CLASS_SHARED_DENSE = "shared_dense"
CLASS_EMBEDDING = "embedding"
CLASS_OTHER = "other"
CLASS_ORDER = (
    CLASS_EXPERT,
    CLASS_ATTENTION,
    CLASS_SHARED_DENSE,
    CLASS_EMBEDDING,
    CLASS_OTHER,
)

# Printed with every report so a reader can audit which tensors were counted
# as routed-expert weights without reading this file.
CLASSIFICATION_RULES: tuple[tuple[str, str], ...] = (
    (CLASS_EMBEDDING, "name segment embed_tokens, embed_out, lm_head, wte, or wpe"),
    (
        CLASS_SHARED_DENSE,
        "name segment shared_expert or shared_experts: always-active capacity, "
        "not routed",
    ),
    (CLASS_EXPERT, "name segment experts followed by a numeric expert index"),
    (CLASS_ATTENTION, "name segment self_attn, attention, or attn"),
    (
        CLASS_SHARED_DENSE,
        "name segment mlp, gate, feed_forward, ffn, or any segment "
        "containing norm",
    ),
    (CLASS_OTHER, "no rule above matched"),
)

DERIVATION_TRELLIS = "exl3-trellis-geometry"
DERIVATION_UNPACKED = "unpacked-dtype-elements"
DERIVATION_UNDETERMINED = "undetermined"

DERIVATION_METHODS: Mapping[str, str] = {
    DERIVATION_TRELLIS: (
        "int16 rank-3 tensor whose final name segment is trellis and whose "
        "last dimension is a multiple of 16: logical shape is "
        "[shape[0]*16, shape[1]*16] and the tier is shape[2]//16, the EXL3 "
        "packing the EXL3 quantization backend validates at load time"
    ),
    DERIVATION_UNPACKED: (
        "float dtype storing one logical weight per element: the logical "
        "element count is the product of the declared shape"
    ),
    DERIVATION_UNDETERMINED: (
        "no recognized packing relates the stored bytes to a logical weight "
        "count; the tensor is listed and excluded from every average"
    ),
}

# Metadata files beside the shards that may state a bit rate.
DECLARED_SOURCES = ("config.json", "quantization_config.json", "tier_bitmap.json")

# Keys read as declared bit rates wherever they appear in those files.
DECLARED_BIT_KEYS = frozenset(
    {"bits", "bpw", "bits_per_weight", "head_bits", "weight_bits", "num_bits"}
)

# A JSON array of small integers inside a per-tensor bit-rate map is read as a
# per-unit tier vector. The value bound is the EXL3 tier range and the length
# bound keeps short unrelated arrays — token id lists, shapes — out of the
# histogram. The reading is a heuristic over files with no versioned schema and
# is labelled as one in the report. `config.json` is deliberately excluded: it
# carries many short integer arrays that have nothing to do with bit rates.
DECLARED_TIER_MIN = EXL3_MIN_BITS
DECLARED_TIER_MAX = EXL3_MAX_BITS
DECLARED_TIER_MIN_LENGTH = 8
TIER_VECTOR_SOURCES = frozenset({"quantization_config.json", "tier_bitmap.json"})

MAX_LISTED_UNDETERMINED = 20


class CensusError(Exception):
    """A condition that stops a census from being produced at all."""


@dataclass(frozen=True)
class TensorRecord:
    """One tensor as its shard's header describes it, plus what that implies."""

    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    stored_bytes: int
    name_class: str
    role: str
    derivation: str
    logical_weights: int | None = None
    bits_per_weight: float | None = None
    layer_index: int | None = None
    expert_index: int | None = None
    note: str | None = None

    @property
    def determined(self) -> bool:
        return self.derivation != DERIVATION_UNDETERMINED


@dataclass
class ClassTotals:
    """Aggregates over every tensor one classification rule claimed."""

    tensor_count: int = 0
    stored_bytes: int = 0
    weight_tensor_count: int = 0
    weight_bytes: int = 0
    sidecar_tensor_count: int = 0
    sidecar_bytes: int = 0
    logical_weights: int = 0
    undetermined_tensor_count: int = 0
    undetermined_bytes: int = 0
    bits_per_weight: list[float] = field(default_factory=list)
    histogram: dict[float, list[int]] = field(default_factory=dict)


def read_safetensors_header(path: Path) -> tuple[Mapping[str, Any], int]:
    """Return one shard's JSON header and the payload's starting offset.

    Only the header prefix is read. The declared length is checked against the
    file size before the read, so a corrupt or non-safetensors file cannot make
    this allocate an arbitrary buffer.
    """

    size = path.stat().st_size
    if size < HEADER_LENGTH_BYTES:
        raise CensusError(f"{path.name}: shorter than a safetensors header length")
    with path.open("rb") as handle:
        raw_length = handle.read(HEADER_LENGTH_BYTES)
        header_bytes = int.from_bytes(raw_length, "little", signed=False)
        if header_bytes == 0:
            raise CensusError(f"{path.name}: declares a zero-length header")
        if header_bytes > MAX_HEADER_BYTES:
            raise CensusError(
                f"{path.name}: declares a {header_bytes}-byte header, beyond "
                f"the {MAX_HEADER_BYTES}-byte bound a safetensors header has"
            )
        if HEADER_LENGTH_BYTES + header_bytes > size:
            raise CensusError(
                f"{path.name}: declares a {header_bytes}-byte header but holds "
                f"{size} bytes in total"
            )
        raw_header = handle.read(header_bytes)
    if len(raw_header) != header_bytes:
        raise CensusError(f"{path.name}: header is truncated")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CensusError(f"{path.name}: header is not JSON ({error})") from error
    if not isinstance(header, dict):
        raise CensusError(f"{path.name}: header is not a JSON object")
    return header, HEADER_LENGTH_BYTES + header_bytes


def classify_name(name: str) -> str:
    """Return the class the ordered name rules assign to one tensor."""

    if EMBEDDING_RE.search(name):
        return CLASS_EMBEDDING
    if SHARED_EXPERT_RE.search(name):
        return CLASS_SHARED_DENSE
    if ROUTED_EXPERT_RE.search(name):
        return CLASS_EXPERT
    if ATTENTION_RE.search(name):
        return CLASS_ATTENTION
    if DENSE_RE.search(name):
        return CLASS_SHARED_DENSE
    return CLASS_OTHER


def tensor_role(name: str) -> str:
    """Whether a tensor holds weights or the apparatus that decodes them."""

    segment = name.rsplit(".", 1)[-1]
    return "sidecar" if segment in SIDECAR_SUFFIXES else "weight"


def _index(pattern: re.Pattern[str], name: str, group: str) -> int | None:
    match = pattern.search(name)
    return int(match.group(group)) if match else None


def _expert_instance(name: str) -> tuple[int, int] | None:
    """The (layer, expert) pair a routed-expert tensor belongs to."""

    layer = _index(LAYER_RE, name, "layer")
    expert = _index(ROUTED_EXPERT_RE, name, "expert")
    if layer is None or expert is None:
        return None
    return layer, expert


def derive_geometry(
    name: str, dtype: str, shape: Sequence[int], stored_bytes: int
) -> tuple[str, int | None, float | None, str | None]:
    """Relate stored bytes to a logical weight count for one tensor.

    Returns the derivation method, the logical weight count, the effective bits
    per logical weight, and a note where one is needed. A tensor whose packing
    is not recognized returns the undetermined method and a reason, because a
    guessed logical shape would silently move the aggregate this measures.
    """

    element_bits = DTYPE_BITS.get(dtype)
    if element_bits is None:
        return (
            DERIVATION_UNDETERMINED,
            None,
            None,
            f"dtype {dtype!r} is not a known safetensors dtype",
        )

    elements = 1
    for extent in shape:
        elements *= int(extent)
    expected_bytes = elements * element_bits // 8
    if expected_bytes != stored_bytes:
        return (
            DERIVATION_UNDETERMINED,
            None,
            None,
            f"header byte range holds {stored_bytes} bytes but dtype {dtype} "
            f"over shape {tuple(shape)} occupies {expected_bytes}",
        )

    segment = name.rsplit(".", 1)[-1]
    if segment == EXL3_TRELLIS_SUFFIX:
        if dtype != "I16" or len(shape) != 3:
            return (
                DERIVATION_UNDETERMINED,
                None,
                None,
                f"named as an EXL3 trellis but stored as {dtype} with shape "
                f"{tuple(shape)}; the packing requires rank-3 I16",
            )
        packed = int(shape[2])
        if packed % EXL3_BLOCK:
            return (
                DERIVATION_UNDETERMINED,
                None,
                None,
                f"EXL3 trellis last dimension {packed} is not a multiple of "
                f"{EXL3_BLOCK}, so it encodes no whole bit width",
            )
        tier = packed // EXL3_BLOCK
        if not EXL3_MIN_BITS <= tier <= EXL3_MAX_BITS:
            return (
                DERIVATION_UNDETERMINED,
                None,
                None,
                f"EXL3 trellis implies a {tier}-bit tier, outside the "
                f"{EXL3_MIN_BITS}..{EXL3_MAX_BITS} range the format defines",
            )
        logical = int(shape[0]) * EXL3_BLOCK * int(shape[1]) * EXL3_BLOCK
        if logical <= 0:
            return (
                DERIVATION_UNDETERMINED,
                None,
                None,
                f"EXL3 trellis shape {tuple(shape)} implies no weights",
            )
        return (
            DERIVATION_TRELLIS,
            logical,
            stored_bytes * 8 / logical,
            f"logical shape [{int(shape[0]) * EXL3_BLOCK}, "
            f"{int(shape[1]) * EXL3_BLOCK}]",
        )

    if dtype in UNPACKED_DTYPES:
        if elements <= 0:
            return (
                DERIVATION_UNDETERMINED,
                None,
                None,
                f"shape {tuple(shape)} implies no elements",
            )
        return (DERIVATION_UNPACKED, elements, float(element_bits), None)

    return (
        DERIVATION_UNDETERMINED,
        None,
        None,
        f"{dtype} storage with no recognized packing; the logical weight count "
        "cannot be read from name, dtype, and shape alone",
    )


def read_tensor_records(path: Path) -> tuple[list[TensorRecord], list[str]]:
    """Parse every shard header under a model directory.

    Returns the records and any findings — duplicate names across shards, and
    shards whose header could not be parsed — rather than raising, so one bad
    shard does not suppress the census of the rest.
    """

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise CensusError(f"{path}: holds no *.safetensors files")

    records: list[TensorRecord] = []
    findings: list[str] = []
    seen: dict[str, str] = {}
    for shard in shards:
        try:
            header, payload_start = read_safetensors_header(shard)
        except CensusError as error:
            findings.append(f"unreadable shard: {error}")
            continue
        shard_size = shard.stat().st_size
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict):
                findings.append(f"{shard.name}: {name} has a non-object header entry")
                continue
            dtype = str(entry.get("dtype", ""))
            shape = tuple(int(value) for value in entry.get("shape", ()) or ())
            offsets = entry.get("data_offsets") or ()
            if len(offsets) != 2:
                findings.append(f"{shard.name}: {name} has no data_offsets pair")
                continue
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or payload_start + end > shard_size:
                findings.append(
                    f"{shard.name}: {name} data_offsets {start}..{end} fall "
                    f"outside the {shard_size}-byte shard"
                )
                continue
            if name in seen:
                findings.append(
                    f"{name} appears in both {seen[name]} and {shard.name}; "
                    "the duplicate is counted once"
                )
                continue
            seen[name] = shard.name
            stored_bytes = end - start
            derivation, logical, bpw, note = derive_geometry(
                name, dtype, shape, stored_bytes
            )
            instance = _expert_instance(name)
            records.append(
                TensorRecord(
                    name=name,
                    shard=shard.name,
                    dtype=dtype,
                    shape=shape,
                    stored_bytes=stored_bytes,
                    name_class=classify_name(name),
                    role=tensor_role(name),
                    derivation=derivation,
                    logical_weights=logical,
                    bits_per_weight=bpw,
                    layer_index=_index(LAYER_RE, name, "layer"),
                    expert_index=instance[1] if instance else None,
                    note=note,
                )
            )
    if not records:
        raise CensusError(
            f"{path}: {len(shards)} safetensors file(s) yielded no readable "
            "tensor records"
        )
    return records, findings


def _round_bpw(value: float) -> float:
    return round(value, 4)


def summarize_classes(records: Iterable[TensorRecord]) -> dict[str, ClassTotals]:
    """Fold per-tensor records into per-class totals and bit-width histograms."""

    totals: dict[str, ClassTotals] = {name: ClassTotals() for name in CLASS_ORDER}
    for record in records:
        bucket = totals[record.name_class]
        bucket.tensor_count += 1
        bucket.stored_bytes += record.stored_bytes
        if record.role == "sidecar":
            bucket.sidecar_tensor_count += 1
            bucket.sidecar_bytes += record.stored_bytes
            continue
        bucket.weight_tensor_count += 1
        bucket.weight_bytes += record.stored_bytes
        if not record.determined:
            bucket.undetermined_tensor_count += 1
            bucket.undetermined_bytes += record.stored_bytes
            continue
        assert record.logical_weights is not None
        assert record.bits_per_weight is not None
        bucket.logical_weights += record.logical_weights
        bucket.bits_per_weight.append(record.bits_per_weight)
        tier = _round_bpw(record.bits_per_weight)
        slot = bucket.histogram.setdefault(tier, [0, 0])
        slot[0] += 1
        slot[1] += record.logical_weights
    return totals


def distribution(bucket: ClassTotals) -> dict[str, Any] | None:
    """Min, median, max, and the tier histogram over one class's weights."""

    if not bucket.bits_per_weight:
        return None
    histogram = [
        {
            "bits_per_weight": tier,
            "tensor_count": counts[0],
            "logical_weights": counts[1],
            "share_of_logical_weights": (
                counts[1] / bucket.logical_weights if bucket.logical_weights else 0.0
            ),
        }
        for tier, counts in sorted(bucket.histogram.items())
    ]
    return {
        "min": _round_bpw(min(bucket.bits_per_weight)),
        "median": _round_bpw(statistics.median(bucket.bits_per_weight)),
        "max": _round_bpw(max(bucket.bits_per_weight)),
        "tensor_count": len(bucket.bits_per_weight),
        "histogram": histogram,
    }


def expert_aggregate(records: Sequence[TensorRecord]) -> dict[str, Any]:
    """The measured routed-expert bit rate that replaces a uniform assumption.

    Two averages come from the same logical weight count: one over the
    quantized payload alone, and one that also carries the sidecar tensors a
    matrix cannot be decoded without. A transfer-cost model wants the second,
    because moving an expert moves both.
    """

    payload_bytes = 0
    sidecar_bytes = 0
    logical_weights = 0
    undetermined_bytes = 0
    undetermined_count = 0
    for record in records:
        if record.name_class != CLASS_EXPERT:
            continue
        if record.role == "sidecar":
            sidecar_bytes += record.stored_bytes
            continue
        if not record.determined:
            undetermined_bytes += record.stored_bytes
            undetermined_count += 1
            continue
        assert record.logical_weights is not None
        payload_bytes += record.stored_bytes
        logical_weights += record.logical_weights

    return {
        "logical_weights": logical_weights,
        "payload_bytes": payload_bytes,
        "sidecar_bytes": sidecar_bytes,
        "undetermined_tensor_count": undetermined_count,
        "undetermined_bytes": undetermined_bytes,
        "average_bits_per_weight_payload": (
            _round_bpw(payload_bytes * 8 / logical_weights)
            if logical_weights
            else None
        ),
        "average_bits_per_weight_with_sidecars": (
            _round_bpw((payload_bytes + sidecar_bytes) * 8 / logical_weights)
            if logical_weights
            else None
        ),
        "covers_every_expert_tensor": undetermined_count == 0,
    }


def expert_instances(records: Sequence[TensorRecord]) -> dict[str, Any]:
    """Per (layer, expert) byte and bit-rate spread across routed experts.

    One routed expert is the unit an expert transfer moves, so its stored size
    is the quantity a transfer-time estimate needs. Reporting the spread rather
    than a single figure is the point: a uniform assumption predicts none.
    """

    per_instance: dict[tuple[int, int], list[int]] = {}
    for record in records:
        if record.name_class != CLASS_EXPERT:
            continue
        instance = _expert_instance(record.name)
        if instance is None:
            continue
        slot = per_instance.setdefault(instance, [0, 0, 0])
        slot[0] += record.stored_bytes
        if record.role == "sidecar":
            continue
        if record.determined:
            assert record.logical_weights is not None
            slot[1] += record.logical_weights
        else:
            slot[2] += record.stored_bytes

    if not per_instance:
        return {"count": 0}

    sizes = sorted(slot[0] for slot in per_instance.values())
    rates = sorted(
        slot[0] * 8 / slot[1] for slot in per_instance.values() if slot[1] > 0
    )
    summary: dict[str, Any] = {
        "count": len(per_instance),
        "stored_bytes_total": sum(sizes),
        "stored_bytes_min": sizes[0],
        "stored_bytes_median": int(statistics.median(sizes)),
        "stored_bytes_max": sizes[-1],
        "instances_with_undetermined_tensors": sum(
            1 for slot in per_instance.values() if slot[2] > 0
        ),
    }
    if rates:
        summary["bits_per_weight_with_sidecars_min"] = _round_bpw(rates[0])
        summary["bits_per_weight_with_sidecars_median"] = _round_bpw(
            statistics.median(rates)
        )
        summary["bits_per_weight_with_sidecars_max"] = _round_bpw(rates[-1])
    return summary


def _walk(document: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Yield (json path, value) for every mapping member in a document."""

    if isinstance(document, dict):
        for key, value in document.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            yield from _walk(value, path)
    elif isinstance(document, list):
        for position, value in enumerate(document):
            path = f"{prefix}[{position}]"
            yield from _walk(value, path)


def _is_tier_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= DECLARED_TIER_MIN_LENGTH
        and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and DECLARED_TIER_MIN <= item <= DECLARED_TIER_MAX
            for item in value
        )
    )


def read_declared(path: Path) -> dict[str, Any]:
    """Read what the checkpoint's own metadata states about its bit rate.

    Every field is reported with the file and JSON path it came from, so the
    declared side of the comparison is auditable without this file. Nothing is
    normalized against the measured side.
    """

    sources: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    tier_counts: dict[int, int] = {}
    tier_vector_sources: list[str] = []

    for name in DECLARED_SOURCES:
        candidate = path / name
        if not candidate.is_file():
            sources.append({"file": name, "present": False})
            continue
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            sources.append({"file": name, "present": True, "error": str(error)})
            continue
        sources.append({"file": name, "present": True})
        for json_path, value in _walk(document):
            key = json_path.rsplit(".", 1)[-1]
            if key in DECLARED_BIT_KEYS and isinstance(value, (int, float, str)):
                fields.append({"file": name, "path": json_path, "value": value})
            if name in TIER_VECTOR_SOURCES and _is_tier_vector(value):
                tier_vector_sources.append(name)
                for item in value:
                    tier_counts[item] = tier_counts.get(item, 0) + 1

    declared_average: float | None = None
    declared_average_source: str | None = None
    for entry in fields:
        if entry["path"].rsplit(".", 1)[-1] != "head_bits" and isinstance(
            entry["value"], (int, float)
        ):
            declared_average = float(entry["value"])
            declared_average_source = f"{entry['file']}:{entry['path']}"
            break

    tier_total = sum(tier_counts.values())
    return {
        "sources": sources,
        "bit_rate_fields": fields,
        "declared_average_bits_per_weight": declared_average,
        "declared_average_source": declared_average_source,
        "tier_vector_histogram": {
            "reading": (
                "heuristic: every JSON array of at least "
                f"{DECLARED_TIER_MIN_LENGTH} integers within "
                f"{DECLARED_TIER_MIN}..{DECLARED_TIER_MAX} found in "
                + ", ".join(sorted(TIER_VECTOR_SOURCES))
                + " is read as a per-unit tier vector"
            ),
            "files": sorted(set(tier_vector_sources)),
            "entry_count": tier_total,
            "counts": {str(bits): count for bits, count in sorted(tier_counts.items())},
            "mean_declared_tier": (
                _round_bpw(
                    sum(bits * count for bits, count in tier_counts.items()) / tier_total
                )
                if tier_total
                else None
            ),
        },
    }


def compare(
    declared: Mapping[str, Any], measured: float | None, tolerance: float
) -> dict[str, Any]:
    """State declared against measured without reconciling the two."""

    stated = declared.get("declared_average_bits_per_weight")
    if stated is None or measured is None:
        return {
            "declared_average_bits_per_weight": stated,
            "measured_average_bits_per_weight": measured,
            "tolerance_bits_per_weight": tolerance,
            "comparable": False,
            "agrees": None,
            "note": (
                "no declared scalar bit rate was found"
                if stated is None
                else "no expert bit rate could be measured"
            ),
        }
    difference = measured - float(stated)
    return {
        "declared_average_bits_per_weight": float(stated),
        "declared_average_source": declared.get("declared_average_source"),
        "measured_average_bits_per_weight": measured,
        "difference_bits_per_weight": _round_bpw(difference),
        "tolerance_bits_per_weight": tolerance,
        "comparable": True,
        "agrees": abs(difference) <= tolerance,
        "note": (
            "the declared rate is a quantizer target over its own tensor "
            "selection; the measured rate is stored payload bits over routed "
            "expert weights. They are different populations and a difference "
            "is a fact about the checkpoint, not an error to correct."
        ),
    }


def census(path: Path, tolerance: float) -> dict[str, Any]:
    """Produce the whole machine-readable census for one model directory."""

    if not path.exists():
        raise CensusError(f"{path}: no such path")
    if not path.is_dir():
        raise CensusError(f"{path}: not a directory")

    records, findings = read_tensor_records(path)
    totals = summarize_classes(records)
    aggregate = expert_aggregate(records)
    declared = read_declared(path)
    comparison = compare(
        declared, aggregate["average_bits_per_weight_payload"], tolerance
    )

    if comparison.get("comparable") and comparison.get("agrees") is False:
        findings.append(
            "declared and measured expert bit rates disagree by "
            f"{comparison['difference_bits_per_weight']} bpw "
            f"(declared {comparison['declared_average_bits_per_weight']}, "
            f"measured {comparison['measured_average_bits_per_weight']})"
        )
    if not aggregate["covers_every_expert_tensor"]:
        findings.append(
            f"{aggregate['undetermined_tensor_count']} expert weight tensor(s) "
            f"holding {aggregate['undetermined_bytes']} byte(s) have no "
            "derivable logical shape; every expert average excludes them"
        )

    shards = sorted(path.glob("*.safetensors"))
    undetermined = [record for record in records if not record.determined]
    return {
        "schema": SCHEMA,
        "model_path": str(path),
        "safetensors_files": [
            {"name": shard.name, "bytes": shard.stat().st_size} for shard in shards
        ],
        "classification_rules": [
            {"order": position, "class": name, "rule": rule}
            for position, (name, rule) in enumerate(CLASSIFICATION_RULES, 1)
        ],
        "derivation_methods": dict(DERIVATION_METHODS),
        "sidecar_suffixes": sorted(SIDECAR_SUFFIXES),
        "totals": {
            "tensor_count": len(records),
            "stored_bytes": sum(record.stored_bytes for record in records),
            "undetermined_tensor_count": len(undetermined),
            "undetermined_bytes": sum(
                record.stored_bytes for record in undetermined
            ),
        },
        "classes": {
            name: {
                "tensor_count": bucket.tensor_count,
                "stored_bytes": bucket.stored_bytes,
                "weight_tensor_count": bucket.weight_tensor_count,
                "weight_bytes": bucket.weight_bytes,
                "sidecar_tensor_count": bucket.sidecar_tensor_count,
                "sidecar_bytes": bucket.sidecar_bytes,
                "logical_weights": bucket.logical_weights,
                "undetermined_tensor_count": bucket.undetermined_tensor_count,
                "undetermined_bytes": bucket.undetermined_bytes,
                "bits_per_weight": distribution(bucket),
            }
            for name, bucket in totals.items()
        },
        "expert_aggregate": aggregate,
        "expert_instances": expert_instances(records),
        "declared": declared,
        "comparison": comparison,
        "undetermined": [
            {
                "name": record.name,
                "file": record.shard,
                "class": record.name_class,
                "dtype": record.dtype,
                "shape": list(record.shape),
                "stored_bytes": record.stored_bytes,
                "reason": record.note,
            }
            for record in undetermined[:MAX_LISTED_UNDETERMINED]
        ],
        "undetermined_listing_truncated": len(undetermined) > MAX_LISTED_UNDETERMINED,
        "findings": findings,
    }


def _gib(value: int) -> str:
    return f"{value / 1024 ** 3:.3f} GiB"


def render(report: Mapping[str, Any]) -> str:
    """Render the census for a reader who has to audit how it was derived."""

    lines: list[str] = []
    lines.append(f"expert bit-width census: {report['model_path']}")
    files = report["safetensors_files"]
    totals = report["totals"]
    lines.append(
        f"  {len(files)} safetensors file(s), {totals['tensor_count']} tensor(s), "
        f"{totals['stored_bytes']} bytes ({_gib(totals['stored_bytes'])})"
    )
    lines.append("")

    lines.append("classification rules, first match wins:")
    for rule in report["classification_rules"]:
        lines.append(f"  {rule['order']}. {rule['class']:<13} {rule['rule']}")
    lines.append(
        "  sidecar tensors (counted as bytes, never as weights) end in: "
        + ", ".join(report["sidecar_suffixes"])
    )
    lines.append("")

    lines.append("bits-per-weight derivation:")
    for method, description in report["derivation_methods"].items():
        lines.append(f"  {method}: {description}")
    lines.append("")

    lines.append(
        f"{'class':<13} {'tensors':>8} {'stored bytes':>16} "
        f"{'logical weights':>16} {'min':>7} {'median':>7} {'max':>7}"
    )
    for name in CLASS_ORDER:
        entry = report["classes"][name]
        if not entry["tensor_count"]:
            continue
        spread = entry["bits_per_weight"]
        if spread:
            tail = (
                f"{spread['min']:>7.3f} {spread['median']:>7.3f} "
                f"{spread['max']:>7.3f}"
            )
        else:
            tail = f"{'-':>7} {'-':>7} {'-':>7}"
        lines.append(
            f"{name:<13} {entry['tensor_count']:>8} {entry['stored_bytes']:>16} "
            f"{entry['logical_weights']:>16} {tail}"
        )
    lines.append("")

    expert = report["classes"][CLASS_EXPERT]
    spread = expert["bits_per_weight"]
    if spread:
        lines.append("routed-expert bits-per-weight tiers:")
        for bucket in spread["histogram"]:
            lines.append(
                f"  {bucket['bits_per_weight']:>7.3f} bpw  "
                f"{bucket['tensor_count']:>8} tensor(s)  "
                f"{bucket['logical_weights']:>16} weight(s)  "
                f"{bucket['share_of_logical_weights'] * 100:6.2f}% of expert weights"
            )
        lines.append("")

    aggregate = report["expert_aggregate"]
    lines.append("routed-expert aggregate (this replaces a uniform assumption):")
    lines.append(f"  logical weights           {aggregate['logical_weights']}")
    lines.append(
        f"  quantized payload bytes   {aggregate['payload_bytes']} "
        f"({_gib(aggregate['payload_bytes'])})"
    )
    lines.append(
        f"  sidecar bytes             {aggregate['sidecar_bytes']} "
        f"({_gib(aggregate['sidecar_bytes'])})"
    )
    lines.append(
        "  average bpw, payload      "
        f"{aggregate['average_bits_per_weight_payload']}"
    )
    lines.append(
        "  average bpw, incl sidecar "
        f"{aggregate['average_bits_per_weight_with_sidecars']}"
    )
    if not aggregate["covers_every_expert_tensor"]:
        lines.append(
            f"  EXCLUDED {aggregate['undetermined_tensor_count']} undetermined "
            f"expert tensor(s), {aggregate['undetermined_bytes']} byte(s); "
            "the averages above are not a complete account"
        )
    lines.append("")

    instances = report["expert_instances"]
    if instances.get("count"):
        lines.append(
            f"per routed expert across {instances['count']} (layer, expert) "
            "instance(s):"
        )
        lines.append(
            f"  stored bytes   min {instances['stored_bytes_min']}  "
            f"median {instances['stored_bytes_median']}  "
            f"max {instances['stored_bytes_max']}"
        )
        if "bits_per_weight_with_sidecars_median" in instances:
            lines.append(
                "  bpw incl sidecar   min "
                f"{instances['bits_per_weight_with_sidecars_min']}  "
                f"median {instances['bits_per_weight_with_sidecars_median']}  "
                f"max {instances['bits_per_weight_with_sidecars_max']}"
            )
        lines.append("")

    declared = report["declared"]
    lines.append("declared quantization metadata:")
    for source in declared["sources"]:
        state = "present" if source["present"] else "absent"
        if source.get("error"):
            state = f"unreadable ({source['error']})"
        lines.append(f"  {source['file']}: {state}")
    for entry in declared["bit_rate_fields"]:
        lines.append(f"  {entry['file']}:{entry['path']} = {entry['value']}")
    tiers = declared["tier_vector_histogram"]
    if tiers["entry_count"]:
        lines.append(f"  declared tier vectors ({tiers['reading']}):")
        for bits, count in tiers["counts"].items():
            lines.append(f"    {bits}-bit: {count} entry(ies)")
        lines.append(f"    mean declared tier: {tiers['mean_declared_tier']}")
    lines.append("")

    comparison = report["comparison"]
    lines.append("declared against measured:")
    if not comparison["comparable"]:
        lines.append(f"  not comparable: {comparison['note']}")
    else:
        verdict = "AGREE" if comparison["agrees"] else "DISAGREE"
        lines.append(
            f"  {verdict}: declared "
            f"{comparison['declared_average_bits_per_weight']} "
            f"({comparison['declared_average_source']}) against measured "
            f"{comparison['measured_average_bits_per_weight']}, difference "
            f"{comparison['difference_bits_per_weight']} bpw at tolerance "
            f"{comparison['tolerance_bits_per_weight']}"
        )
        lines.append(f"  {comparison['note']}")
    lines.append("")

    if report["undetermined"]:
        lines.append("tensors with no derivable logical shape:")
        for entry in report["undetermined"]:
            lines.append(
                f"  {entry['name']} [{entry['class']}] {entry['dtype']} "
                f"{entry['shape']} {entry['stored_bytes']} bytes: {entry['reason']}"
            )
        if report["undetermined_listing_truncated"]:
            lines.append(
                f"  ... {totals['undetermined_tensor_count'] - len(report['undetermined'])}"
                " further undetermined tensor(s) not listed"
            )
        lines.append("")

    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"FINDING {finding}")
    else:
        lines.append("no findings")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="expert_bitwidth_census",
        description=(
            "Report the per-tensor bit-width distribution a quantized "
            "checkpoint stores, and the measured average over its routed "
            "MoE expert tensors, from safetensors headers alone."
        ),
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="directory holding the checkpoint's *.safetensors shards",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the machine-readable census instead of the rendered report",
    )
    parser.add_argument(
        "--tolerance-bpw",
        type=float,
        default=0.01,
        help=(
            "bits-per-weight difference below which declared and measured "
            "rates are reported as agreeing (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--require-agreement",
        action="store_true",
        help=(
            "exit non-zero when the declared and measured rates differ by "
            "more than the tolerance, or cannot be compared at all"
        ),
    )
    arguments = parser.parse_args(argv)

    try:
        report = census(arguments.model_path, arguments.tolerance_bpw)
    except CensusError as error:
        print(f"FAIL {error}")
        return 1
    except OSError as error:
        print(f"FAIL {arguments.model_path}: {error}")
        return 1

    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(render(report))

    if arguments.require_agreement and report["comparison"].get("agrees") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
