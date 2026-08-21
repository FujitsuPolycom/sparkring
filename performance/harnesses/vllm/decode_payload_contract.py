"""GPU-free payload contract for GLM-5.2 adaptive-MTP decode.

This module describes payload geometry; it does not install an adapter,
allocate transport resources, or change collective routing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MAX_CONCURRENCY = 8
MAX_MTP_DEPTH = 4
ROWS_PER_SEQUENCE = MAX_MTP_DEPTH + 1
MAX_QUERY_ROWS = MAX_CONCURRENCY * ROWS_PER_SEQUENCE

# Existing mixed-batch padding buckets. Exact uniform speculative-decode
# widths (multiples of five) and the concurrency maximum are added below.
_BASE_CAPTURE_ROWS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)


@dataclass(frozen=True)
class PayloadFamily:
    key: str
    label: str
    geometry: str
    bytes_per_query_row: int

    def bytes_for(self, query_rows: int) -> int:
        _validate_query_rows(query_rows)
        return self.bytes_per_query_row * query_rows


PAYLOAD_FAMILIES = (
    PayloadFamily("tp_ar", "TP AR", "[Q,6144] BF16", 6_144 * 2),
    PayloadFamily(
        "indexer",
        "Indexer",
        "[Q,2,2048] INT32",
        2 * 2_048 * 4,
    ),
    PayloadFamily(
        "dcp_query",
        "DCP query",
        "[Q,16,576] BF16",
        16 * 576 * 2,
    ),
    PayloadFamily("dcp_lse", "DCP LSE", "[Q,64] FP32", 64 * 4),
    PayloadFamily(
        "dcp_output",
        "DCP output",
        "reduce-scatter input [Q,64,512] BF16",
        64 * 512 * 2,
    ),
    PayloadFamily(
        "vocabulary",
        "Vocabulary",
        "[Q,38720] BF16",
        38_720 * 2,
    ),
    PayloadFamily(
        "fused_combine_0",
        "Fused combine R0",
        "32 heads output + LSE",
        16_512,
    ),
    PayloadFamily(
        "fused_combine_1",
        "Fused combine R1",
        "16 heads output + LSE",
        8_256,
    ),
)
PAYLOAD_FAMILY_BY_KEY = {family.key: family for family in PAYLOAD_FAMILIES}


def _validate_concurrency(concurrency: int) -> None:
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or not 1 <= concurrency <= MAX_CONCURRENCY
    ):
        raise ValueError(
            f"concurrency must be an integer in [1, {MAX_CONCURRENCY}]"
        )


def _validate_query_rows(query_rows: int) -> None:
    if (
        isinstance(query_rows, bool)
        or not isinstance(query_rows, int)
        or not 1 <= query_rows <= MAX_QUERY_ROWS
    ):
        raise ValueError(
            f"query_rows must be an integer in [1, {MAX_QUERY_ROWS}]"
        )


def maximum_query_rows(concurrency: int) -> int:
    """Return Qmax = concurrency * (maximum MTP depth + one decode row)."""

    _validate_concurrency(concurrency)
    return concurrency * ROWS_PER_SEQUENCE


def legal_query_rows(concurrency: int) -> tuple[int, ...]:
    """Return every legal logical decode width for the concurrency."""

    return tuple(range(1, maximum_query_rows(concurrency) + 1))


def capture_query_rows(concurrency: int) -> tuple[int, ...]:
    """Return current graph buckets for a concurrency.

    C1 retains its observed Q1/Q3/Q5 plan. C2-C8 add exact uniform
    speculative-decode widths to the bounded mixed-batch padding plan.
    """

    maximum = maximum_query_rows(concurrency)
    if concurrency == 1:
        return (1, 3, 5)
    rows = {value for value in _BASE_CAPTURE_ROWS if value <= maximum}
    rows.add(maximum)
    rows.update(range(ROWS_PER_SEQUENCE, maximum + 1, ROWS_PER_SEQUENCE))
    return tuple(sorted(rows))


def payload_bytes(family_key: str, query_rows: int) -> int:
    """Return bytes per rank for one payload family and logical/padded Q."""

    try:
        family = PAYLOAD_FAMILY_BY_KEY[family_key]
    except KeyError as error:
        raise ValueError(f"unknown payload family: {family_key}") from error
    return family.bytes_for(query_rows)


def maximum_payloads(concurrency: int) -> dict[str, int]:
    """Return each family's maximum bytes per rank at a concurrency."""

    maximum = maximum_query_rows(concurrency)
    return {
        family.key: family.bytes_for(maximum)
        for family in PAYLOAD_FAMILIES
    }


def _format_rows(rows: Sequence[int]) -> str:
    return ", ".join(str(row) for row in rows)


def _render_maximum_table(families: Sequence[PayloadFamily]) -> list[str]:
    labels = [family.label for family in families]
    lines = [
        "| C | Max Q | " + " | ".join(labels) + " |",
        "|---:|---:|" + "|".join("---:" for _ in labels) + "|",
    ]
    for concurrency in range(1, MAX_CONCURRENCY + 1):
        maximum = maximum_query_rows(concurrency)
        values = [
            f"{family.bytes_for(maximum):,}" for family in families
        ]
        lines.append(
            f"| {concurrency} | {maximum} | "
            + " | ".join(values)
            + " |"
        )
    return lines


def render_markdown() -> str:
    """Render the checked-in deliverable from this contract's formulas."""

    collective_families = PAYLOAD_FAMILIES[:6]
    fused_families = PAYLOAD_FAMILIES[6:]
    lines = [
        "# Adaptive-MTP decode payload contract (C1-C8)",
        "",
        "<!-- Generated by spark_decode_payload_contract.py. -->",
        "",
        "This is a decode-only, bytes-per-rank contract. With maximum MTP "
        f"depth {MAX_MTP_DEPTH}, `Qmax = C × (MTPmax + 1) = C × "
        f"{ROWS_PER_SEQUENCE}`. Every integer width from Q1 through Qmax is "
        "legal; capture buckets are a padding/graph optimization, not an "
        "admission whitelist.",
        "",
        "## Width and capture matrix",
        "",
        "| Concurrency | Legal logical widths | Max Q | Current capture Q |",
        "|---:|:---|---:|:---|",
    ]
    for concurrency in range(1, MAX_CONCURRENCY + 1):
        maximum = maximum_query_rows(concurrency)
        lines.append(
            f"| C{concurrency} | Q1-Q{maximum} | {maximum} | "
            f"{_format_rows(capture_query_rows(concurrency))} |"
        )

    lines.extend(
        [
            "",
            "## Payload formulas",
            "",
            "| Family | Tensor geometry | Bytes/rank |",
            "|:---|:---|---:|",
        ]
    )
    for family in PAYLOAD_FAMILIES:
        lines.append(
            f"| {family.label} | `{family.geometry}` | "
            f"{family.bytes_per_query_row:,} × Q |"
        )

    lines.extend(
        [
            "",
            "## Maximum payloads by concurrency",
            "",
        ]
    )
    lines.extend(_render_maximum_table(collective_families))
    lines.extend(["", "Fused combine rounds:", ""])
    lines.extend(_render_maximum_table(fused_families))

    lines.extend(
        [
            "",
            "## Indexer payloads through C8",
            "",
            "| Q | Bytes/rank | Q | Bytes/rank | Q | Bytes/rank | "
            "Q | Bytes/rank |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    indexer = PAYLOAD_FAMILY_BY_KEY["indexer"]
    for offset in range(1, 11):
        entries = []
        for query_rows in (offset, offset + 10, offset + 20, offset + 30):
            entries.extend(
                (str(query_rows), f"{indexer.bytes_for(query_rows):,}")
            )
        lines.append("| " + " | ".join(entries) + " |")

    lines.extend(
        [
            "",
            "## Scope and unresolved admission gaps",
            "",
            "- CKV record transfers are excluded: their width follows selected "
            "record count and context policy, not decode concurrency.",
            "- Prefill is excluded: widths such as Q48-Q4096 need a separate "
            "matrix and tiling contract and must not expand this Q40 decode "
            "surface.",
            "- Adapter admission remains unresolved. Before routing, it must "
            "bind collective type, dtype, layout, byte count, and semantic "
            "family; a matching byte count alone is insufficient.",
            "- The transport port namespace remains unresolved. Concurrent "
            "families/sessions need an explicit collision-free namespace; "
            "this payload matrix does not assign ports.",
            "",
            "Validate the checked-in copy with:",
            "",
            "```text",
            "python performance/harnesses/vllm/"
            "decode_payload_contract.py --print",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", type=Path, metavar="PATH")
    action.add_argument("--write", type=Path, metavar="PATH")
    action.add_argument("--print", action="store_true", dest="print_contract")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rendered = render_markdown()
    if args.print_contract:
        print(rendered, end="")
        return 0
    if args.check is not None:
        if args.check.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"generated payload contract is stale: {args.check}")
        return 0
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
