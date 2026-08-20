#!/usr/bin/env python3
"""Bring a prepared vLLM tree to the inputs the exact-Q40 overlays require.

`runtime/exl3-r7/prepare_context.py` materializes the tree `pins.json`
names. Two of that tree's files are not what
`q40_exact_state_overlay.py` and `q40_exact_state_attestation_overlay.py`
accept, and both generators reject an unexpected input rather than
transform it. This applies the two differences, each bound to the SHA-256
of what it reads and what it writes:

- `exl3.py` reserves W4A16 scratch for every row count. The pinned tree
  returns a single unit below 129 rows, which holds for the cooperative
  K6 small-M kernel and not for the generic scheduler other
  architectures use.
- `model_runner.py` and `scheduler.py` gain the routed-expert capturer
  that the attestation overlay anchors on, from
  `q40_v2_route_capture.patch`.

The results are the two files
`runtime/exl3-r7/test-fixtures/` records, so offline tests and a build
consume identical bytes.

Safety class: MUTATES the tree named on the command line. It contacts no
network and no configured Spark, and it refuses to write anything when a
SHA-256 does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTE_CAPTURE_PATCH = HERE / "q40_v2_route_capture.patch"
ROUTE_CAPTURE_PATCH_SHA256 = (
    "0b6f25459a84e52a1bc096c32ab27593ec42b059beaceb8a11b9b9a5e5543ff1"
)

EXL3_RELATIVE = "vllm/model_executor/layers/quantization/exl3.py"
MODEL_RUNNER_RELATIVE = "vllm/v1/worker/gpu/model_runner.py"

EXL3_INPUT_SHA256 = (
    "42c0e9150a065c48e3780eebc8b3c89ea410d82610e2c14bc97546dee6866214"
)
EXL3_OUTPUT_SHA256 = (
    "8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4"
)
MODEL_RUNNER_INPUT_SHA256 = (
    "e8f13cbabadc6c591bc644874abbfcca5a0c935396993e1046ace173f227b0d0"
)
MODEL_RUNNER_OUTPUT_SHA256 = (
    "992486a1a70fd0cb54c54cf3f70af0f09b4bcc6ae3bbf433e66b1296415015b4"
)

SCRATCH_BEFORE = (
    b"    if rows <= 128:\n"
    b"        # The cooperative K6 small-M kernel does not consume W4A16 scratch.\n"
    b"        return 1\n"
)
SCRATCH_AFTER = (
    b"    # SM120 uses a cooperative K6 small-M kernel that ignores this buffer.\n"
    b"    # Other architectures use the generic W4A16 scheduler even at small M,\n"
    b"    # so reserve the conservative M48/M64 capacity for every row count.\n"
)


class OverlayInputError(RuntimeError):
    """A tree did not hold the bytes an exact-Q40 overlay input requires."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def apply_scratch_reservation(tree: Path) -> str:
    """Reserve W4A16 scratch for every row count, bound to both hashes."""

    path = tree / EXL3_RELATIVE
    payload = path.read_bytes()
    observed = sha256_bytes(payload)
    if observed == EXL3_OUTPUT_SHA256:
        return "already applied"
    if observed != EXL3_INPUT_SHA256:
        raise OverlayInputError(
            f"{EXL3_RELATIVE}: expected {EXL3_INPUT_SHA256}, read {observed}"
        )
    count = payload.count(SCRATCH_BEFORE)
    if count != 1:
        raise OverlayInputError(
            f"{EXL3_RELATIVE}: found {count} scratch-reservation preimages, wanted 1"
        )
    updated = payload.replace(SCRATCH_BEFORE, SCRATCH_AFTER)
    produced = sha256_bytes(updated)
    if produced != EXL3_OUTPUT_SHA256:
        raise OverlayInputError(
            f"{EXL3_RELATIVE}: produced {produced}, expected {EXL3_OUTPUT_SHA256}"
        )
    path.write_bytes(updated)
    return "applied"


def apply_route_capture(tree: Path) -> str:
    """Add the routed-expert capturer, bound to the model-runner hashes."""

    path = tree / MODEL_RUNNER_RELATIVE
    observed = sha256_bytes(path.read_bytes())
    if observed == MODEL_RUNNER_OUTPUT_SHA256:
        return "already applied"
    if observed != MODEL_RUNNER_INPUT_SHA256:
        raise OverlayInputError(
            f"{MODEL_RUNNER_RELATIVE}: expected {MODEL_RUNNER_INPUT_SHA256}, "
            f"read {observed}"
        )
    if not ROUTE_CAPTURE_PATCH.is_file():
        raise OverlayInputError(f"{ROUTE_CAPTURE_PATCH} is absent")
    patch_hash = sha256_bytes(ROUTE_CAPTURE_PATCH.read_bytes())
    if patch_hash != ROUTE_CAPTURE_PATCH_SHA256:
        raise OverlayInputError(
            f"{ROUTE_CAPTURE_PATCH.name}: expected "
            f"{ROUTE_CAPTURE_PATCH_SHA256}, read {patch_hash}"
        )
    result = subprocess.run(
        ["git", "apply", str(ROUTE_CAPTURE_PATCH)],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise OverlayInputError(
            "route-capture patch did not apply: "
            + (detail[-1] if detail else "no detail")
        )
    produced = sha256_bytes(path.read_bytes())
    if produced != MODEL_RUNNER_OUTPUT_SHA256:
        raise OverlayInputError(
            f"{MODEL_RUNNER_RELATIVE}: produced {produced}, expected "
            f"{MODEL_RUNNER_OUTPUT_SHA256}"
        )
    return "applied"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare_q40_overlay_inputs",
        description=(
            "Bring a prepared vLLM tree to the inputs the exact-Q40 overlays "
            "accept, refusing to write when a SHA-256 does not match."
        ),
    )
    parser.add_argument(
        "tree",
        type=Path,
        help="the vLLM tree prepare_context.py produced, containing vllm/",
    )
    arguments = parser.parse_args(argv)
    tree = arguments.tree.resolve()
    if not (tree / EXL3_RELATIVE).is_file():
        print(f"FAIL {tree} does not hold {EXL3_RELATIVE}")
        return 1
    try:
        scratch = apply_scratch_reservation(tree)
        capture = apply_route_capture(tree)
    except OverlayInputError as error:
        print(f"FAIL {error}")
        return 1
    print(f"OK  {EXL3_RELATIVE}: scratch reservation {scratch}")
    print(f"OK  {MODEL_RUNNER_RELATIVE}: route capture {capture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
