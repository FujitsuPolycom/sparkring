#!/usr/bin/env python3
"""Fail-closed r14 sparse-indexer profile-capacity fix for the EXL3 image."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/"
    "model_executor/layers/sparse_attn_indexer.py"
)
PREIMAGE_SHA256 = "b76b09559b34f93481f1c0f67d0834fb3207b53804212fabb866c6dc5f114a80"
POSTIMAGE_SHA256 = "d049d3c9ac029d1e2bcb9e78c3a5b98a86805b611a0559fbd66f82c61813ba20"

OLD = """def _get_b12x_paged_indexer_profile_k_rows(
    max_model_len: int,
    total_seq_lens: int,
) -> int:
    known_k_rows = max(int(max_model_len), int(total_seq_lens), 0)
    if known_k_rows > 0:
        return known_k_rows
    return _get_b12x_indexer_paged_supertile_k()
"""

NEW = """def _get_b12x_paged_indexer_profile_k_rows(
    max_model_len: int,
    total_seq_lens: int,
) -> int:
    # The page table is per request, while total_seq_lens is the capacity of
    # the flattened prefill workspace across requests. Feeding that aggregate
    # capacity to the B12X planner produces an impossible per-row context and
    # can overflow CUTLASS DSL's 32-bit contiguous memref extent.
    if int(max_model_len) > 0:
        return int(max_model_len)
    if int(total_seq_lens) > 0:
        return int(total_seq_lens)
    return _get_b12x_indexer_paged_supertile_k()
"""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def transform(source: str) -> str:
    if source.count(OLD) != 1:
        raise RuntimeError(
            "sparse-indexer profile-capacity preimage does not contain exactly "
            "one recognized function body"
        )
    return source.replace(OLD, NEW)


def patch(path: Path) -> None:
    payload = path.read_bytes()
    observed = sha256_bytes(payload)
    if observed == POSTIMAGE_SHA256:
        return
    if observed != PREIMAGE_SHA256:
        raise RuntimeError(
            f"sparse-indexer preimage mismatch: expected {PREIMAGE_SHA256}, "
            f"got {observed}"
        )
    updated = transform(payload.decode("utf-8")).encode("utf-8")
    postimage = sha256_bytes(updated)
    if postimage != POSTIMAGE_SHA256:
        raise RuntimeError(
            f"sparse-indexer postimage mismatch: expected {POSTIMAGE_SHA256}, "
            f"got {postimage}"
        )
    path.write_bytes(updated)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()
    patch(args.target)
    print(POSTIMAGE_SHA256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
