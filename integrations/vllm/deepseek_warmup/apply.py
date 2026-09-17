# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SparkRing contributors.
"""Prepare a source-verified DSpark startup warmup overlay for image assembly."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELATIVE = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
PREIMAGE = "5c0aac819160131cd4b3870bd0e4278b93df4a08675da7868cd986b2de139aec"
WARMUP_RELATIVE = "vllm/v1/worker/gpu/warmup.py"
WARMUP_PREIMAGE = "6a8b0a684dec3b84ed832577cac73568a574cd4b9b2e0a628c8ccf2bdcfcd3d8"
ANCHOR = """        for target_query_len in sorted(target_query_lens):
"""
INSERTION = """        # Each positive scheduled length selects a per-request power-of-two
        # preparation tile. Cover feasible tiles missing from the long-context
        # representatives without assuming DSpark and DFlash share an anchor.
        target_query_lens = {
            length for length in target_query_lens if length <= self.max_num_tokens
        }
        for draft_query_len in draft_query_lens:
            for exponent in range(9):
                tile = 1 << exponent
                if any(
                    min(256, 1 << (length + draft_query_len - 1).bit_length()) == tile
                    for length in target_query_lens
                ):
                    continue
                length = max(1, tile // 2 + 1 - draft_query_len)
                if (
                    length <= self.max_num_tokens
                    and min(256, 1 << (length + draft_query_len - 1).bit_length()) == tile
                ):
                    target_query_lens.add(length)
"""
INSERTION = "        if self._speculator_name == \"DSpark\":\n" + "".join(
    "    " + line if line.strip() else line for line in INSERTION.splitlines(keepends=True)
)
WARMUP_ANCHOR = """            if model_runner.speculator.wants_auto_sps_curve:
                _profile_sps_curve(model_runner)
                model_runner.kv_connector.set_disabled(True)
"""
WARMUP_INSERTION = """        elif (
            model_runner.speculator is not None
            and model_runner.speculative_config is not None
            and model_runner.speculative_config.method == "dspark"
        ):
            # DSpark prepares draft inputs even when confidence-based pruning
            # is disabled. Use the established scratch lane before readiness.
            with use_workspace_lane(1):
                model_runner.speculator.warmup_capacity_kernels()
"""


def build(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != PREIMAGE:
        raise ValueError("installed DFlash speculator source does not match the inspected image")
    text = source.decode("utf-8")
    if text.count(ANCHOR) != 1:
        raise ValueError("startup preparation loop is not unique")
    return text.replace(ANCHOR, INSERTION + ANCHOR).encode()


def build_warmup(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != WARMUP_PREIMAGE:
        raise ValueError("GPU warmup source does not match the inspected image")
    text = source.decode("utf-8")
    if text.count(WARMUP_ANCHOR) != 1:
        raise ValueError("capacity warmup call site is not unique")
    return text.replace(WARMUP_ANCHOR, WARMUP_ANCHOR + WARMUP_INSERTION).encode()


def prepare_source(source_root: Path, output_root: Path) -> dict:
    """Validate all inputs before creating an overlay; never edit the source tree."""
    prepared = []
    for relative, builder in ((RELATIVE, build), (WARMUP_RELATIVE, build_warmup)):
        before = (source_root / relative).read_bytes()
        prepared.append((relative, before, builder(before)))
    output_root.mkdir(parents=True, exist_ok=False)
    files = []
    for relative, before, after in prepared:
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        files.append({"path": relative, "preimage_sha256": hashlib.sha256(before).hexdigest(),
                      "sha256": hashlib.sha256(after).hexdigest()})
    manifest = {"schema": "sparkring-dspark-prepare-warmup/v1", "files": files,
                "status": "research-only", "scope": "DSpark startup preparation; no profile defaults changed"}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True, help="image site-packages directory")
    parser.add_argument("--output-root", type=Path, required=True, help="overlay destination; must not exist")
    args = parser.parse_args()
    print(json.dumps(prepare_source(args.source_root, args.output_root)))
