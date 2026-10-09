"""Build the exact target-only Q40 EXL3 mixed-state source overlay.

The overlay adds one graph-stable capacity-40, block-8 state to mixed target
layers.  It uses the existing prefill tiers and tile, and dispatches only an
exact 40-row call to that state.  The input hash and unique insertion anchors
make source drift fail closed; the deployed source is never edited in place.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


INPUT_SHA256 = "8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4"
OUTPUT_SHA256 = "8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2"

CONSTANT_ANCHOR = """\
_MIXED_TRELLIS_ROUTE_BLOCK_SIZE = 8
"""

CONSTANT_REPLACEMENT = """\
_MIXED_TRELLIS_ROUTE_BLOCK_SIZE = 8
_MIXED_TRELLIS_TARGET_Q40_ROWS = 40
"""

STATE_ANCHOR = """\
            prefill = make_state(
                prefill_capacity,
                prefill_block_m,
                mixed["prefill_tile_config"],
            )
        runtime = {
"""

STATE_REPLACEMENT = """\
            prefill = make_state(
                prefill_capacity,
                prefill_block_m,
                mixed["prefill_tile_config"],
            )
        q40 = None
        if (
            not owner_token[1]
            and max_decode_m == 32
            and _MIXED_TRELLIS_TARGET_Q40_ROWS
            <= min(max_batched_tokens, prefill_capacity)
        ):
            q40 = make_state(
                _MIXED_TRELLIS_TARGET_Q40_ROWS,
                _MIXED_TRELLIS_ROUTE_BLOCK_SIZE,
                mixed["prefill_tile_config"],
            )
        runtime = {
"""

RUNTIME_ANCHOR = """\
            "decode": decode,
            "prefill": prefill,
"""

RUNTIME_REPLACEMENT = """\
            "decode": decode,
            "prefill": prefill,
            "q40": q40,
"""

DISPATCH_ANCHOR = """\
        if runtime["prefill"] is None:
            raise RuntimeError("mixed-K EXL3 one-grid prefill plan is unavailable")
"""

DISPATCH_REPLACEMENT = """\
        if (
            m == _MIXED_TRELLIS_TARGET_Q40_ROWS
            and runtime["q40"] is not None
        ):
            return run_state(
                x,
                topk_weights,
                topk_ids,
                runtime["q40"],
                mixed["prefill_tiers"],
            )

        if runtime["prefill"] is None:
            raise RuntimeError("mixed-K EXL3 one-grid prefill plan is unavailable")
"""


class ExactQ40StateOverlayError(RuntimeError):
    """The pinned EXL3 source contract was not satisfied."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def transform(source: bytes) -> bytes:
    actual = sha256_bytes(source)
    if actual != INPUT_SHA256:
        raise ExactQ40StateOverlayError(
            f"EXL3 input hash mismatch: expected {INPUT_SHA256}, got {actual}"
        )
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExactQ40StateOverlayError("EXL3 source is not UTF-8") from error

    replacements = (
        (CONSTANT_ANCHOR, CONSTANT_REPLACEMENT),
        (STATE_ANCHOR, STATE_REPLACEMENT),
        (RUNTIME_ANCHOR, RUNTIME_REPLACEMENT),
        (DISPATCH_ANCHOR, DISPATCH_REPLACEMENT),
    )
    for anchor, replacement in replacements:
        if text.count(anchor) != 1:
            raise ExactQ40StateOverlayError(
                "EXL3 insertion anchor is absent or non-unique"
            )
        text = text.replace(anchor, replacement, 1)

    compile(text, "exl3.py", "exec")
    return text.encode("utf-8")


def install(source: Path, output: Path) -> dict[str, str | int]:
    if output.exists():
        raise ExactQ40StateOverlayError(f"refusing to overwrite {output}")
    source_bytes = source.read_bytes()
    output_bytes = transform(source_bytes)
    if OUTPUT_SHA256 and sha256_bytes(output_bytes) != OUTPUT_SHA256:
        raise ExactQ40StateOverlayError("generated EXL3 output hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(output_bytes)
            stream.flush()
    except FileExistsError as error:
        raise ExactQ40StateOverlayError(f"refusing to overwrite {output}") from error
    return {
        "input_path": str(source.resolve()),
        "input_sha256": sha256_bytes(source_bytes),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_bytes(output_bytes),
        "output_bytes": len(output_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key, value in install(args.source.resolve(), args.output.resolve()).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
