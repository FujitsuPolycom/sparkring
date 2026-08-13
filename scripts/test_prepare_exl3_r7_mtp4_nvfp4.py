from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_exl3_r7_candidate as gen  # noqa: E402
import generate_exl3_r7_stock_dcp4 as stock_gen  # noqa: E402
import prepare_exl3_r7_mtp2 as mtp2  # noqa: E402
import prepare_exl3_r7_mtp3 as mtp3  # noqa: E402
import prepare_exl3_r7_mtp4 as mtp4  # noqa: E402
import prepare_exl3_r7_mtp4_nvfp4 as nvfp4  # noqa: E402


def source_profile() -> dict:
    template = json.loads(gen.TEMPLATE_PATH.read_text(encoding="utf-8"))
    pins = json.loads(gen.PINS_PATH.read_text(encoding="utf-8"))
    recipe = json.loads(gen.RECIPE_PATH.read_text(encoding="utf-8"))
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    mtp2_profile = mtp2.derive_candidate(stock)
    mtp3_profile = mtp3.derive_candidate(stock, mtp2_profile)
    return mtp4.derive_candidate(mtp3_profile)


def source_site() -> str:
    return (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 4\n"
        "  max_model_len: 65536\n"
        "  kv_cache_bytes_per_rank: 9250000000\n"
        "  max_num_seqs: 8\n"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _option(profile: dict, name: str) -> str:
    arguments = profile["extra_vllm_args"]
    return arguments[arguments.index(name) + 1]


def test_candidate_changes_only_dynamic_nvfp4_contract() -> None:
    source = source_profile()
    candidate = nvfp4.derive_candidate(source)

    assert candidate["profile_id"] == (
        f"{source['profile_id']}-nvfp4-rope8-ctx256k-b4096"
    )
    assert candidate["environment"]["KV_FP8_ROPE"] == "1"
    assert candidate["environment"]["VLLM_NVFP4_MLA_DYNAMIC_SCALE"] == "1"
    assert candidate["environment"]["VLLM_EXL3_PREFILL_CAPACITY"] == "4096"
    assert _option(candidate, "--max-num-batched-tokens") == "4096"
    assert _option(candidate, "--kv-cache-dtype") == "nvfp4_ds_mla"
    assert candidate["extra_labels"]["org.sparkring.r7.kv-contract"] == (
        nvfp4.KV_CONTRACT_LABEL
    )
    nvfp4.validate_candidate(source, candidate)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("environment", "KV_FP8_ROPE"), "0"),
        (("environment", "VLLM_EXL3_PREFILL_CAPACITY"), "8192"),
        (("extra_labels", "unrelated"), "drift"),
    ],
)
def test_candidate_rejects_unrelated_or_contract_drift(path, value) -> None:
    source = source_profile()
    candidate = nvfp4.derive_candidate(source)
    candidate[path[0]][path[1]] = value
    with pytest.raises(nvfp4.ContractError, match="allowlist"):
        nvfp4.validate_candidate(source, candidate)


def test_source_rejects_an_existing_dynamic_override() -> None:
    source = source_profile()
    source["environment"]["VLLM_NVFP4_MLA_DYNAMIC_SCALE"] = "1"
    with pytest.raises(nvfp4.ContractError, match="already declares"):
        nvfp4.derive_candidate(source)


def test_site_changes_only_model_limit() -> None:
    source = source_site()
    candidate = nvfp4.derive_site_text(source)
    assert candidate == source.replace("max_model_len: 65536", "max_model_len: 262144")
    assert candidate.replace("max_model_len: 262144", "max_model_len: 65536") == source


def test_cli_is_hash_bound_and_preserves_rollback(tmp_path: Path) -> None:
    source_profile_path = tmp_path / "mtp4.json"
    source_site_path = tmp_path / "mtp4.yaml"
    candidate_profile_path = tmp_path / "nvfp4.json"
    candidate_site_path = tmp_path / "nvfp4.yaml"
    rollback_profile_path = tmp_path / "rollback.json"
    rollback_site_path = tmp_path / "rollback.yaml"
    profile_bytes = (json.dumps(source_profile(), indent=2) + "\n").encode()
    site_bytes = source_site().encode()
    source_profile_path.write_bytes(profile_bytes)
    source_site_path.write_bytes(site_bytes)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4_nvfp4.py"),
            "--source-profile",
            str(source_profile_path),
            "--source-site",
            str(source_site_path),
            "--expected-profile-sha256",
            _sha256(profile_bytes),
            "--expected-site-sha256",
            _sha256(site_bytes),
            "--candidate-profile",
            str(candidate_profile_path),
            "--candidate-site",
            str(candidate_site_path),
            "--rollback-profile",
            str(rollback_profile_path),
            "--rollback-site",
            str(rollback_site_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    nvfp4.validate_candidate(source_profile(), json.loads(candidate_profile_path.read_text()))
    assert rollback_profile_path.read_bytes() == profile_bytes
    assert rollback_site_path.read_bytes() == site_bytes
    receipt = json.loads(result.stdout)
    assert receipt["reported_kv_capacity_tokens"] == 1_156_864
    assert receipt["source_profile_sha256"] == _sha256(profile_bytes)


def test_cli_rejects_hash_drift_before_writing(tmp_path: Path) -> None:
    profile_path = tmp_path / "mtp4.json"
    site_path = tmp_path / "mtp4.yaml"
    output_path = tmp_path / "candidate.json"
    profile_path.write_text(json.dumps(source_profile()), encoding="utf-8")
    site_path.write_text(source_site(), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4_nvfp4.py"),
            "--source-profile",
            str(profile_path),
            "--source-site",
            str(site_path),
            "--expected-profile-sha256",
            "0" * 64,
            "--expected-site-sha256",
            _sha256(site_path.read_bytes()),
            "--candidate-profile",
            str(output_path),
            "--candidate-site",
            str(tmp_path / "candidate.yaml"),
            "--rollback-profile",
            str(tmp_path / "rollback.json"),
            "--rollback-site",
            str(tmp_path / "rollback.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "source profile SHA-256 mismatch" in result.stderr
    assert not output_path.exists()


def test_validate_detects_a_copied_unrelated_field() -> None:
    source = source_profile()
    candidate = nvfp4.derive_candidate(source)
    mutated = copy.deepcopy(candidate)
    mutated["container_name"] = "unrelated"
    with pytest.raises(nvfp4.ContractError, match="allowlist"):
        nvfp4.validate_candidate(source, mutated)
