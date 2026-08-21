"""Build the target-only exact-Q40 EXL3 serving canary offline.

The derived profile replaces the pinned EXL3 source with an insertion-only
overlay that adds a capacity-40, block-8 target state.  It leaves decode,
general prefill, and the uniform draft policy unchanged.  A pre-graph model
runner hook must attest the loaded runtime on every DCP rank before serving.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


OPERATOR_BASE_PROFILE_SHA256 = (
    "08686011f6d38f0524f71afb17c450600c525a67e3ffe7a675fc9afe905e1fb8"
)
BASE_EXL3_SHA256 = "8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4"
EXACT_Q40_EXL3_SHA256 = "8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2"
OPERATOR_MODEL_RUNNER_SHA256 = (
    "0e2e0150702029b3c09bd117c33101d90d8197386d278dc6008973b314ae9997"
)
MIXED_TRELLIS_SHA256 = "dc03f8e71123ef5288264825fbacc0777227cf933c27d280de7c2f8db2a356c9"
FUSED_MOE_IMPL_SHA256 = "102fe793b7687efaaada868cbadaa81c4e5e1ac39cfde0a71172ed8320efc810"

REMOTE_ROOT = "/var/tmp/sparkring-q40-exact-state-v1"
VLLM_CACHE_ROOT = "/cache/jit/vllm-q40-exact-state-v1"
EXL3_CONTAINER = (
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
    "layers/quantization/exl3.py"
)
MODEL_RUNNER_CONTAINER = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py"
)
MIXED_TRELLIS_CONTAINER = (
    "/opt/venv/lib/python3.12/site-packages/"
    "b12x/moe/_shared/kernels/w4a16/mixed_trellis.py"
)
FUSED_MOE_IMPL_CONTAINER = (
    "/opt/venv/lib/python3.12/site-packages/b12x/moe/fused_moe/_impl.py"
)


class PrepareExactQ40ServingError(RuntimeError):
    """The exact-Q40 canary does not satisfy its pinned offline contract."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum_line(digest: str, path: str) -> str:
    return f"{digest}  {path}"


def _replace_checksum(command: str, old: str, new: str, path: str) -> str:
    old_line = repr(_checksum_line(old, path))
    new_line = repr(_checksum_line(new, path))
    if command.count(old_line) != 1:
        raise PrepareExactQ40ServingError(
            f"baseline checksum for {path} is absent or non-unique"
        )
    return command.replace(old_line, new_line, 1)


def _assert_only_allowed_changes(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    restored = copy.deepcopy(candidate)
    for key in ("profile_id", "container_name", "confirmation"):
        restored[key] = base[key]
    for key in (
        "VLLM_CACHE_ROOT",
        "SPARK_Q40_EXACT_STATE_ATTEST_PATH",
        "SPARK_Q40_EXACT_STATE_EXPECTED_EXL3_SHA256",
        "SPARK_Q40_EXACT_STATE_IMAGE_ID",
        "SPARK_Q40_EXACT_STATE_CHECKPOINT",
    ):
        restored["environment"].pop(key, None)
    for key in (
        "org.sparkring.experiment",
        "org.sparkring.evidence-maturity",
        "org.sparkring.q40-policy-scope",
        "org.sparkring.q40-route-block",
    ):
        restored["extra_labels"].pop(key, None)

    exl3 = [
        volume
        for volume in restored["extra_volumes"]
        if volume.get("container") == EXL3_CONTAINER
    ]
    if len(exl3) != 1:
        raise PrepareExactQ40ServingError("candidate must contain one EXL3 mount")
    base_exl3 = [
        volume
        for volume in base["extra_volumes"]
        if volume.get("container") == EXL3_CONTAINER
    ]
    if len(base_exl3) > 1:
        raise PrepareExactQ40ServingError("baseline contains multiple EXL3 mounts")
    if base_exl3:
        exl3[0]["host"] = base_exl3[0]["host"]
    else:
        restored["extra_volumes"].remove(exl3[0])
    model_runner = [
        volume
        for volume in restored["extra_volumes"]
        if volume.get("container") == MODEL_RUNNER_CONTAINER
    ]
    if len(model_runner) != 1:
        raise PrepareExactQ40ServingError("candidate must contain one model-runner mount")
    restored["extra_volumes"].remove(model_runner[0])
    restored["attestation_hook"] = copy.deepcopy(base["attestation_hook"])
    if restored != base:
        raise PrepareExactQ40ServingError(
            "candidate contains a change outside the exact-Q40 allowlist"
        )


def prepare(
    *,
    base_profile_path: Path,
    expected_base_profile_sha256: str,
    exl3_path: Path,
    model_runner_path: Path,
    expected_model_runner_sha256: str,
    bundle_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not expected_model_runner_sha256:
        raise PrepareExactQ40ServingError(
            "expected model-runner SHA-256 is unset; the runtime attestation overlay is not sealed"
        )
    if bundle_path.exists():
        raise PrepareExactQ40ServingError(f"refusing to replace bundle {bundle_path}")
    expected = (
        (base_profile_path, expected_base_profile_sha256, "baseline profile"),
        (exl3_path, EXACT_Q40_EXL3_SHA256, "patched EXL3"),
        (model_runner_path, expected_model_runner_sha256, "model runner"),
    )
    for path, digest, label in expected:
        if not path.is_file() or sha256(path) != digest:
            raise PrepareExactQ40ServingError(f"{label} hash mismatch")

    base = json.loads(base_profile_path.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(base)
    if candidate["environment"].get("VLLM_EXL3_PREFILL_BLOCK_M") not in (None, ""):
        raise PrepareExactQ40ServingError(
            "baseline unexpectedly overrides the general prefill block size"
        )
    if candidate["environment"].get("VLLM_EXL3_TRELLIS_MAX_M") not in (None, ""):
        raise PrepareExactQ40ServingError(
            "baseline unexpectedly overrides the decode threshold"
        )
    if candidate["environment"].get("VLLM_EXL3_PREFILL_CAPACITY") != "4096":
        raise PrepareExactQ40ServingError("baseline prefill capacity is not 4096")

    candidate["profile_id"] = "glm52-exl3-r7-target-exact-q40-block8"
    candidate["container_name"] = "glm52-sparkring-q40-exact-state-canary"
    candidate["confirmation"] = "START-SIRCL-Q40-EXACT-STATE-CANARY-ALL-FOUR"
    candidate["environment"].update(
        {
            "VLLM_CACHE_ROOT": VLLM_CACHE_ROOT,
            "SPARK_Q40_EXACT_STATE_ATTEST_PATH": (
                "/cache/jit/q40-exact-state-serving-v1-rank{rank}.json"
            ),
            "SPARK_Q40_EXACT_STATE_EXPECTED_EXL3_SHA256": EXACT_Q40_EXL3_SHA256,
            "SPARK_Q40_EXACT_STATE_IMAGE_ID": base["image_id"],
            "SPARK_Q40_EXACT_STATE_CHECKPOINT": base["identity"]["model_revision"],
        }
    )
    candidate.setdefault("extra_labels", {}).update(
        {
            "org.sparkring.experiment": "target-exact-q40-block8",
            "org.sparkring.evidence-maturity": "research-only",
            "org.sparkring.q40-policy-scope": "target-mixed-exact-40-rows",
            "org.sparkring.q40-route-block": "8",
        }
    )

    exl3_volumes = [
        volume
        for volume in candidate["extra_volumes"]
        if volume.get("container") == EXL3_CONTAINER
    ]
    if len(exl3_volumes) > 1:
        raise PrepareExactQ40ServingError("baseline contains multiple EXL3 mounts")
    if exl3_volumes:
        exl3_volumes[0]["host"] = f"{REMOTE_ROOT}/exl3.py"
    else:
        candidate["extra_volumes"].append(
            {
                "host": f"{REMOTE_ROOT}/exl3.py",
                "container": EXL3_CONTAINER,
                "mode": "ro",
            }
        )
    if any(
        volume.get("container") == MODEL_RUNNER_CONTAINER
        for volume in candidate["extra_volumes"]
    ):
        raise PrepareExactQ40ServingError("baseline already overlays model_runner.py")
    candidate["extra_volumes"].append(
        {
            "host": f"{REMOTE_ROOT}/model_runner.py",
            "container": MODEL_RUNNER_CONTAINER,
            "mode": "ro",
        }
    )

    hook = candidate.get("attestation_hook")
    if not isinstance(hook, list) or len(hook) != 3 or not isinstance(hook[2], str):
        raise PrepareExactQ40ServingError("baseline attestation hook shape drifted")
    hook[2] = _replace_checksum(
        hook[2], BASE_EXL3_SHA256, EXACT_Q40_EXL3_SHA256, EXL3_CONTAINER
    )
    marker = " | sha256sum --check --strict -"
    if hook[2].count(marker) != 1:
        raise PrepareExactQ40ServingError("attestation checksum marker is not unique")
    additions = (
        _checksum_line(expected_model_runner_sha256, MODEL_RUNNER_CONTAINER),
        _checksum_line(MIXED_TRELLIS_SHA256, MIXED_TRELLIS_CONTAINER),
        _checksum_line(FUSED_MOE_IMPL_SHA256, FUSED_MOE_IMPL_CONTAINER),
    )
    hook[2] = hook[2].replace(
        marker, " " + " ".join(repr(line) for line in additions) + marker, 1
    )

    _assert_only_allowed_changes(base, candidate)
    bundle_path.mkdir(parents=True)
    files = {"exl3": exl3_path, "model_runner": model_runner_path}
    bundled: dict[str, dict[str, str | int]] = {}
    for name, source in files.items():
        destination = bundle_path / source.name
        shutil.copyfile(source, destination)
        bundled[name] = {
            "path": str(destination.resolve()),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }

    manifest = {
        "schema": "sparkring-target-exact-q40-block8-serving-bundle/v1",
        "status": "offline-validated",
        "scope": "target-mixed-exact-40-rows",
        "remote_root": REMOTE_ROOT,
        "vllm_cache_root": VLLM_CACHE_ROOT,
        "base_profile": {
            "path": str(base_profile_path.resolve()),
            "sha256": expected_base_profile_sha256,
        },
        "candidate_profile_id": candidate["profile_id"],
        "files": bundled,
        "source_contract": {
            "base_exl3_sha256": BASE_EXL3_SHA256,
            "patched_exl3_sha256": EXACT_Q40_EXL3_SHA256,
            "model_runner_sha256": expected_model_runner_sha256,
            "mixed_trellis_sha256": MIXED_TRELLIS_SHA256,
            "fused_moe_impl_sha256": FUSED_MOE_IMPL_SHA256,
        },
        "claim_gates": {
            "profile_diff_allowlist": "pass",
            "q1_q32_and_non_q40_paths_unchanged": "blocked-live-attestation",
            "target_q40_capacity40_block8_all_layers": "blocked-live-attestation",
            "uniform_draft_unchanged": "blocked-live-attestation",
            "unique_q40_arena_bytes": "blocked-live-attestation",
            "fresh_cache_graph_capture_1_through_40": "blocked-live-canary",
            "q32_q33_q39_q40_q41_equality": "blocked-live-canary",
            "matched_c8_manifest_bracket": "blocked-live-canary",
            "long_prefill_regression": "blocked-live-canary",
            "end_to_end_speedup": "unsupported-until-live-bracket",
        },
    }
    return candidate, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--expected-base-profile-sha256", required=True)
    parser.add_argument("--exl3", type=Path, required=True)
    parser.add_argument("--model-runner", type=Path, required=True)
    parser.add_argument("--expected-model-runner-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_profile, args.output_manifest):
        if output.exists():
            raise PrepareExactQ40ServingError(f"refusing to overwrite {output}")
    candidate, manifest = prepare(
        base_profile_path=args.base_profile.resolve(),
        expected_base_profile_sha256=args.expected_base_profile_sha256,
        exl3_path=args.exl3.resolve(),
        model_runner_path=args.model_runner.resolve(),
        expected_model_runner_sha256=args.expected_model_runner_sha256,
        bundle_path=args.bundle.resolve(),
    )
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"profile_sha256={sha256(args.output_profile)}")
    print(f"manifest_sha256={sha256(args.output_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
