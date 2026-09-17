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
REQ_STATES_RELATIVE = "vllm/v1/worker/gpu/states.py"
REQ_STATES_SHA256 = "14bde37b80299de9673d316ff5cf4fa5432ca9c23b4a955df60695b7a788ca56"
ANCHOR = """        for target_query_len in sorted(target_query_lens):
"""
INSERTION = """        # Tiles and the padded-context scalar's Triton specialization both
        # contribute to the preparation key. Adjacent boundary lengths cover
        # scalar alignment without assuming a shared draft anchor convention.
        target_query_lens = {
            length for length in target_query_lens if length <= self.max_num_tokens
        }
        for draft_query_len in draft_query_lens:
            for exponent in range(9):
                tile = 1 << exponent
                length = max(1, tile // 2 + 1 - draft_query_len)
                boundary_lengths = {length, length + 1}
                for alignment in (1, 2, 4, 8, 16):
                    aligned_length = ((length + alignment - 1) // alignment) * alignment
                    aligned_requests = 16 // min(16, aligned_length & -aligned_length)
                    if (
                        aligned_requests <= self.max_num_reqs
                        and aligned_requests * aligned_length <= self.max_num_tokens
                        and min(256, 1 << (aligned_length + draft_query_len - 1).bit_length()) == tile
                    ):
                        boundary_lengths.add(aligned_length)
                        break
                for candidate_length in boundary_lengths:
                    if (
                        candidate_length <= self.max_num_tokens
                        and min(256, 1 << (candidate_length + draft_query_len - 1).bit_length()) == tile
                    ):
                        target_query_lens.add(candidate_length)
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
REQUEST_COUNTS = """            request_counts = {num_reqs}
            if self._speculator_name == "DSpark":
                # Match scalar==1, ordinary integers, and divisibility-by-16
                # specializations that occur in real single/batched requests.
                request_counts.add(1)
                if target_query_len == 1 and num_reqs >= 2:
                    request_counts.add(2)
                aligned_count = 16 // min(16, target_query_len & -target_query_len)
                if aligned_count <= num_reqs:
                    request_counts.add(aligned_count)
            for num_reqs in sorted(request_counts):
"""
SAMPLE_BUFFERS = """            last_sampled = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )
            next_prefill_tokens = torch.zeros_like(last_sampled)
"""
TYPED_SAMPLE_BUFFERS = """            # ReqStates keeps sampled IDs as int64 and prefill IDs as int32.
            last_sampled = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int64 if self._speculator_name == "DSpark" else torch.int32,
                device=self.device,
            )
            next_prefill_tokens = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
"""


def build(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != PREIMAGE:
        raise ValueError("installed DFlash speculator source does not match the inspected image")
    text = source.decode("utf-8")
    if text.count(ANCHOR) != 1:
        raise ValueError("startup preparation loop is not unique")
    text = text.replace(ANCHOR, INSERTION + ANCHOR)
    if text.count(SAMPLE_BUFFERS) != 1:
        raise ValueError("startup sampling buffers are not unique")
    text = text.replace(SAMPLE_BUFFERS, TYPED_SAMPLE_BUFFERS)
    start = text.index("            num_tokens = num_reqs * target_query_len\n", text.index(ANCHOR))
    end = text.index("\n    def load_draft_model(", start)
    body = text[start:end]
    indented = "".join("    " + line if line.strip() else line for line in body.splitlines(keepends=True))
    return (text[:start] + REQUEST_COUNTS + indented + text[end:]).encode()


def build_warmup(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != WARMUP_PREIMAGE:
        raise ValueError("GPU warmup source does not match the inspected image")
    text = source.decode("utf-8")
    if text.count(WARMUP_ANCHOR) != 1:
        raise ValueError("capacity warmup call site is not unique")
    return text.replace(WARMUP_ANCHOR, WARMUP_ANCHOR + WARMUP_INSERTION).encode()


def prepare_source(source_root: Path, output_root: Path) -> dict:
    """Validate all inputs before creating an overlay; never edit the source tree."""
    if hashlib.sha256((source_root / REQ_STATES_RELATIVE).read_bytes()).hexdigest() != REQ_STATES_SHA256:
        raise ValueError("request-state source does not match the inspected runtime buffer contract")
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
                "checked_runtime_inputs": [{"path": REQ_STATES_RELATIVE, "sha256": REQ_STATES_SHA256}],
                "status": "research-only", "scope": "DSpark startup preparation; no profile defaults changed"}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True, help="image site-packages directory")
    parser.add_argument("--output-root", type=Path, required=True, help="overlay destination; must not exist")
    args = parser.parse_args()
    print(json.dumps(prepare_source(args.source_root, args.output_root)))
