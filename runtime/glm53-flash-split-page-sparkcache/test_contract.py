from __future__ import annotations

import json
import re
import hashlib
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = HERE / "pins.json"
ARTIFACT = HERE / "qualified-artifact.json"
LAUNCHER = HERE / "launch-qualified-rank.sh"
ENV_TEMPLATE = HERE / "qualified.env.example"
ROLLBACK = HERE / "rollback-rank.sh"
RECEIPT = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "split-page-shared-base-c8-20260830"
    / "validation.json"
)
CLIENT_RESULTS = RECEIPT.parent / "client-results.json"
RECORD = (
    ROOT
    / "performance"
    / "records"
    / "glm53-flash"
    / "split-page-shared-base-c8-20260830.md"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bash_path(path: Path) -> str:
    if path.drive:
        return (
            f"/mnt/{path.drive[0].lower()}/"
            + path.as_posix().split(":/", maxsplit=1)[1]
        )
    return path.as_posix()


def test_artifact_binds_exact_sources_models_and_runtime() -> None:
    artifact = _json(ARTIFACT)
    assert artifact["schema"] == "sparkring-glm53-split-page-sparkcache-artifact/v1"
    assert artifact["status"] == "qualified"
    assert artifact["artifact"]["image_id"] == (
        "sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818"
    )
    assert artifact["artifact"]["published_oci_digest"] is None
    assert artifact["artifact"]["source_digest_sha256"] == (
        "21035ceeeab514b573dffc9a8b415b246fad8cd11ad83c633717b37f2bf6dd1b"
    )
    assert artifact["sources"]["sparkcache"]["commit"] == (
        "59ac0b04db6035a9a9d2a52e92405ceaf84daa40"
    )
    assert artifact["sources"]["sparkcache"]["cache_namespace_change"] == "none"
    assert artifact["sources"]["vllm"]["sparkcache_composition_commit"] == (
        "6da4865d440608a46eada50f27b2fff0e698c574"
    )
    assert artifact["sources"]["b12x"]["commit"] == (
        "b1d541f9e71a35f030d45fae437630fff7507c2a"
    )
    serving = artifact["serving"]
    assert serving["tensor_parallel_size"] == 4
    assert serving["decode_context_parallel_size"] == 1
    assert serving["kv_cache_dtype"] == "fp8"
    assert serving["kv_cache_bytes_per_rank"] == 20 * 1024**3
    assert serving["target_page_size_tokens"] == 512
    assert serving["recurrent_page_size_tokens"] == 512
    assert serving["sparkcache"]["load_lanes"] == 8
    assert serving["sparkcache"]["max_pending_restores"] == 8
    assert serving["sparkcache"]["cuda_arena_bytes_per_lane"] == 256 * 1024**2
    qualification = artifact["qualification"]
    assert qualification["scheduler_inventory_warmup_required_after_restart"] is True
    assert qualification["external_restores"] == 8
    assert qualification["physical_base_reads_per_rank"] == 1
    assert qualification["avoided_base_reads_per_rank"] == 7


def test_public_input_lock_matches_qualified_artifact() -> None:
    pins = _json(PINS)
    artifact = _json(ARTIFACT)
    assert pins["schema"] == "sparkring-glm53-split-page-sparkcache-input-lock/v1"
    assert pins["status"] == "implemented"
    assert pins["source_digest_sha256"] == artifact["artifact"]["source_digest_sha256"]
    assert pins["sources"]["vllm"]["composition_patch_sha256"] == _sha256(
        HERE / pins["sources"]["vllm"]["composition_patch"]
    )
    assert pins["sparkcache_cuda_placement_library_sha256"] == (
        artifact["sources"]["sparkcache_cuda_placement_library_sha256"]
    )
    assert pins["models"]["target"] == artifact["models"]["target"]
    assert pins["models"]["draft"] | {"license": None} == (
        artifact["models"]["draft"] | {"license": None}
    )


def test_c8_receipt_preserves_exactness_and_safe_recompute_boundary() -> None:
    receipt = _json(RECEIPT)
    assert receipt["schema"] == (
        "sparkring-glm53-split-page-shared-base-c8-validation/v1"
    )
    assert receipt["status"]["eight_distinct_request_semantics"] == "qualified"
    assert receipt["status"]["authenticated_shared_base_read"] == "qualified"
    assert receipt["status"]["eight_external_restores"] == "qualified"
    measurement = receipt["measurement"]
    assert measurement["concurrency"] == 8
    assert measurement["exact_response_count"] == 8
    assert measurement["source_client_result_sha256"] == (
        "3e437c65dc5c4eb5f8ec0723d9fe10b58c18f3246a265c1f19fd877c2cef89ad"
    )
    assert measurement["client_result_path"] == CLIENT_RESULTS.name
    assert measurement["client_result_sha256"] == _sha256(CLIENT_RESULTS)
    client = _json(CLIENT_RESULTS)
    assert client["status"] == "verified"
    assert len(client["results"]) == 8
    assert all(result["oracle_match"] for result in client["results"])
    assert measurement["external_restore"]["request_count"] == 8
    assert measurement["external_restore"]["hit_ratio"] == 1.0
    per_rank = measurement["external_restore"]["per_rank"]
    assert per_rank["participants"] == 8
    assert per_rank["physical_base_reads"] == 1
    assert per_rank["avoided_base_reads"] == 7
    assert per_rank["base_bytes"] == 100868258
    assert measurement["restart_readiness"]["worker_manifest_count"] == 20
    assert measurement["restart_readiness"]["warmup_prompt_tokens"] == 5
    assert measurement["restart_readiness"]["warmup_completion_tokens"] == 1
    assert measurement["cold_scheduler_inventory_control"]["safe_recompute_count"] == 1


def test_operator_scripts_pin_image_and_preserve_rollback_containers() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    rollback = ROLLBACK.read_text(encoding="utf-8")
    assert "sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818" in launcher
    assert '"spark_cache_load_threads": integer("SPARKCACHE_LOAD_THREADS")' in launcher
    assert '"spark_cache_max_pending_restores": integer(' in launcher
    assert '"SPARKCACHE_MAX_PENDING_RESTORES")' in launcher
    assert '"spark_cache_cuda_placement_arena_bytes": integer(' in launcher
    assert '"SPARKCACHE_CUDA_ARENA_BYTES")' in launcher
    assert "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512" in launcher
    assert "VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512" in launcher
    assert "glm53-pr535-sc59ac-c8-01-r${rank}" in rollback
    assert "glm53-pr535-sc78-hotpatch-c8-qualified-r${rank}" in rollback
    assert "docker stop" in rollback and "docker start" in rollback


def test_launcher_has_valid_bash_syntax_and_json_encoding() -> None:
    subprocess.run(
        ["bash", "-n", LAUNCHER.relative_to(ROOT).as_posix()],
        check=True,
        cwd=ROOT,
    )
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert launcher.count("json.dumps(") == 3
    assert "--kv-transfer-config \"${kv_transfer_config}\"" in launcher
    assert "--speculative-config \"${speculative_config}\"" in launcher
    assert "--compilation-config \"${compilation_config}\"" in launcher
    assert "org.sparkring.modified-settings" in launcher
    assert "SPARKCACHE_LOW_WATERMARK_BYTES cannot exceed" in launcher
    assert "GPU_MEMORY_UTILIZATION must be greater than zero" in launcher


def test_launcher_rejects_invalid_cache_watermarks_before_docker(
    tmp_path: Path,
) -> None:
    config = tmp_path / "invalid-watermarks.env"
    config.write_bytes(
        (
            "\n".join(
                (
                    "HOST_IP=rank1.example.net",
                    "MASTER_ADDR=rank0.example.net",
                    "TARGET_MODEL_HOST_PATH=/models/target",
                    "DFLASH_MODEL_HOST_PATH=/models/draft",
                    "CACHE_HOST_ROOT=/cache/rank1",
                    "SPARKCACHE_MAX_BYTES=1024",
                    "SPARKCACHE_LOW_WATERMARK_BYTES=2048",
                )
            )
        ).encode("utf-8"),
    )
    result = subprocess.run(
        ["bash", LAUNCHER.relative_to(ROOT).as_posix(), "1", _bash_path(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 78
    assert "SPARKCACHE_LOW_WATERMARK_BYTES cannot exceed" in result.stderr


def test_launcher_encodes_modified_configuration_and_labels_it_unqualified(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "docker-arguments.txt"
    docker = fake_bin / "docker"
    docker.write_bytes(
        b"""#!/bin/sh
if [ "$1" = image ] && [ "$2" = inspect ]; then
  printf '%s\\n' 'sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818'
elif [ "$1" = container ] && [ "$2" = inspect ]; then
  exit 1
elif [ "$1" = run ]; then
  printf '%s\\n' "$@" > "$CAPTURE_PATH"
else
  exit 97
fi
"""
    )
    os.chmod(docker, 0o755)
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    cache = tmp_path / "cache"
    target.mkdir()
    draft.mkdir()
    cache.mkdir()
    config = tmp_path / "modified.env"
    config.write_bytes(
        "\n".join(
            (
                "HOST_IP=rank0.example.net",
                "MASTER_ADDR=rank0.example.net",
                f"TARGET_MODEL_HOST_PATH={_bash_path(target)}",
                f"DFLASH_MODEL_HOST_PATH={_bash_path(draft)}",
                f"CACHE_HOST_ROOT={_bash_path(cache)}",
                f"PATH={_bash_path(fake_bin)}:$PATH",
                f"export CAPTURE_PATH={_bash_path(capture)}",
                "MAX_NUM_SEQS=8",
            )
        ).encode("utf-8")
    )
    result = subprocess.run(
        ["bash", LAUNCHER.relative_to(ROOT).as_posix(), "0", _bash_path(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "org.sparkring.qualification-status=user-modified-unqualified" in arguments
    assert "org.sparkring.modified-settings=MAX_NUM_SEQS" in arguments
    assert arguments[arguments.index("--max-num-seqs") + 1] == "8"
    encoded = arguments[arguments.index("--kv-transfer-config") + 1]
    connector = json.loads(encoded)
    extra = connector["kv_connector_extra_config"]
    assert extra["spark_cache_load_threads"] == 8
    assert extra["spark_cache_max_pending_restores"] == 8
    assert extra["spark_cache_cuda_placement_arena_bytes"] == 268435456


def test_launcher_exposes_common_operator_settings_through_one_env_file() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    template = ENV_TEMPLATE.read_text(encoding="utf-8")
    expected_variables = {
        "HOST_IP",
        "MASTER_ADDR",
        "TARGET_MODEL_HOST_PATH",
        "DFLASH_MODEL_HOST_PATH",
        "CACHE_HOST_ROOT",
        "IMAGE_REF",
        "CONTAINER_PREFIX",
        "SERVED_MODEL_NAME",
        "PORT",
        "MASTER_PORT",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
        "KV_CACHE_MEMORY_BYTES",
        "GPU_MEMORY_UTILIZATION",
        "KV_CACHE_DTYPE",
        "NUM_SPECULATIVE_TOKENS",
        "ATTENTION_BACKEND",
        "MOE_BACKEND",
        "LINEAR_BACKEND",
        "CACHE_NAMESPACE",
        "SPARKCACHE_MAX_BYTES",
        "SPARKCACHE_LOW_WATERMARK_BYTES",
        "SPARKCACHE_LOAD_THREADS",
        "SPARKCACHE_MAX_PENDING_RESTORES",
        "SPARKCACHE_CUDA_RESTORE_IO_WORKERS",
        "SPARKCACHE_CUDA_ARENA_BYTES",
        "SOCKET_IFNAME",
        "NCCL_IB_HCA",
    }
    assigned = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", template, re.MULTILINE))
    assert expected_variables <= assigned
    assert 'config_file="${2:-${SPARKRING_CONFIG_FILE:-}}"' in launcher
    assert 'source "${config_file}"' in launcher
    for variable in expected_variables:
        assert "${" + variable in launcher
    assert "user-modified-unqualified" in launcher


def test_env_template_defaults_match_the_qualified_artifact() -> None:
    artifact = _json(ARTIFACT)
    template = ENV_TEMPLATE.read_text(encoding="utf-8")
    values = dict(
        re.findall(r"^([A-Z][A-Z0-9_]*)=['\"]?([^'\"\n]*)['\"]?$", template, re.MULTILINE)
    )
    serving = artifact["serving"]
    assert int(values["MAX_MODEL_LEN"]) == serving["max_model_len"]
    assert int(values["MAX_NUM_SEQS"]) == serving["max_num_seqs"]
    assert int(values["MAX_NUM_BATCHED_TOKENS"]) == serving["max_num_batched_tokens"]
    assert int(values["KV_CACHE_MEMORY_BYTES"]) == serving["kv_cache_bytes_per_rank"]
    assert values["KV_CACHE_DTYPE"] == serving["kv_cache_dtype"]
    assert int(values["NUM_SPECULATIVE_TOKENS"]) == serving["speculation"]["tokens"]
    assert int(values["SPARKCACHE_LOAD_THREADS"]) == serving["sparkcache"]["load_lanes"]
    assert int(values["SPARKCACHE_MAX_PENDING_RESTORES"]) == serving["sparkcache"]["max_pending_restores"]
    assert int(values["SPARKCACHE_CUDA_ARENA_BYTES"]) == serving["sparkcache"]["cuda_arena_bytes_per_lane"]


def test_record_has_required_evidence_sections_and_public_prose() -> None:
    text = RECORD.read_text(encoding="utf-8")
    for heading in (
        "## Conditions",
        "## Measurement",
        "## Result",
        "## Conclusion",
        "## Limitations",
    ):
        assert heading in text
    assert "All eight requests returned HTTP 200" in text
    assert "All eight requests restored external state" in text
    assert "HTTP `/health` was already successful before the tiny inference" in text
    assert re.search(r"(?i)\b[A-Z]:\\", text) is None
    assert re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", text) is None
