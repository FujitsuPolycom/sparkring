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

import prepare_exl3_r7_mtp4_ckv_gather as ckv  # noqa: E402
import prepare_exl3_r7_mtp4_nvfp4 as nvfp4  # noqa: E402
from test_prepare_exl3_r7_mtp4_nvfp4 import (  # noqa: E402
    source_profile as mtp4_profile,
    source_site as mtp4_site,
)


def source_profile() -> dict:
    return nvfp4.derive_candidate(mtp4_profile())


def source_site() -> str:
    return nvfp4.derive_site_text(mtp4_site())


def _write_source(tmp_path: Path) -> tuple[Path, Path, bytes, bytes]:
    profile_path = tmp_path / "source.json"
    site_path = tmp_path / "source.yaml"
    profile_bytes = (json.dumps(source_profile(), indent=2) + "\n").encode()
    site_bytes = source_site().encode()
    profile_path.write_bytes(profile_bytes)
    site_path.write_bytes(site_bytes)
    return profile_path, site_path, profile_bytes, site_bytes


def test_candidate_changes_only_ckv_gather_contract_and_identity() -> None:
    source = source_profile()
    candidate = ckv.derive_candidate(source)

    expected = copy.deepcopy(source)
    expected["profile_id"] = ckv.CANDIDATE_PROFILE_ID
    expected["environment"]["VLLM_B12X_MLA_CKV_GATHER"] = "1"
    expected["environment"]["VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS"] = str(
        ckv.CKV_GATHER_MAX_TOKENS
    )
    expected["extra_labels"][ckv.CKV_LABEL] = ckv.CKV_LABEL_VALUE

    assert candidate == expected
    assert candidate["environment"]["VLLM_EXL3_PREFILL_CAPACITY"] == "4096"
    assert candidate["extra_vllm_args"][
        candidate["extra_vllm_args"].index("--max-num-batched-tokens") + 1
    ] == "4096"
    assert candidate["extra_vllm_args"][
        candidate["extra_vllm_args"].index("--kv-cache-dtype") + 1
    ] == "nvfp4_ds_mla"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("profile_id",), "unexpected", "profile_id"),
        (("environment", "VLLM_EXL3_PREFILL_CAPACITY"), "2048", "prefill"),
        (("environment", "KV_FP8_ROPE"), "0", "FP8-RoPE"),
        (("environment", "VLLM_NVFP4_MLA_DYNAMIC_SCALE"), "0", "dynamic NVFP4"),
        (("environment", "VLLM_B12X_MLA_CKV_GATHER"), "1", "already declares"),
    ],
)
def test_candidate_rejects_source_contract_drift(
    path: tuple[str, ...], value: str, match: str
) -> None:
    profile = source_profile()
    target = profile
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ckv.ContractError, match=match):
        ckv.derive_candidate(profile)


def test_workspace_prediction_matches_runtime_geometry() -> None:
    assert ckv.CKV_LOCAL_CAPACITY_TOKENS == 65_600
    assert ckv.CKV_WORKSPACE_BYTES_PER_LANE == 217_267_200
    assert ckv.CKV_EXECUTION_LANES == 2
    assert ckv.CKV_WORKSPACE_POOL_BYTES_PER_RANK == 434_534_400
    assert ckv.CKV_WORKSPACE_POOL_MIB_PER_RANK == pytest.approx(414.4043, rel=1e-5)


def test_cli_emits_exact_site_and_rollback_receipts(tmp_path: Path) -> None:
    source_profile_path, source_site_path, source_profile_bytes, source_site_bytes = (
        _write_source(tmp_path)
    )
    candidate_profile = tmp_path / "candidate.json"
    candidate_site = tmp_path / "candidate.yaml"
    rollback_profile = tmp_path / "rollback.json"
    rollback_site = tmp_path / "rollback.yaml"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4_ckv_gather.py"),
            "--source-profile",
            str(source_profile_path),
            "--source-site",
            str(source_site_path),
            "--expected-profile-sha256",
            hashlib.sha256(source_profile_bytes).hexdigest(),
            "--expected-site-sha256",
            hashlib.sha256(source_site_bytes).hexdigest(),
            "--candidate-profile",
            str(candidate_profile),
            "--candidate-site",
            str(candidate_site),
            "--rollback-profile",
            str(rollback_profile),
            "--rollback-site",
            str(rollback_site),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert candidate_site.read_bytes() == source_site_bytes
    assert rollback_profile.read_bytes() == source_profile_bytes
    assert rollback_site.read_bytes() == source_site_bytes
    assert receipt["source_profile_sha256"] == hashlib.sha256(
        source_profile_bytes
    ).hexdigest()
    assert receipt["source_site_sha256"] == hashlib.sha256(source_site_bytes).hexdigest()
    assert receipt["ckv_workspace_pool_bytes_per_rank"] == 434_534_400
    assert receipt["kv_cache_bytes_per_rank"] == 9_250_000_000
    assert receipt["reported_kv_capacity_tokens"] == 1_156_864


def test_cli_rejects_source_hash_drift(tmp_path: Path) -> None:
    source_profile_path, source_site_path, source_profile_bytes, source_site_bytes = (
        _write_source(tmp_path)
    )
    drifted_profile = tmp_path / "source.json"
    drifted_profile.write_bytes(source_profile_bytes + b"\n")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4_ckv_gather.py"),
            "--source-profile",
            str(drifted_profile),
            "--source-site",
            str(source_site_path),
            "--expected-profile-sha256",
            hashlib.sha256(source_profile_bytes).hexdigest(),
            "--expected-site-sha256",
            hashlib.sha256(source_site_bytes).hexdigest(),
            "--candidate-profile",
            str(tmp_path / "candidate.json"),
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
    assert "source profile SHA-256" in result.stderr


def test_cli_refuses_to_overwrite_an_input(tmp_path: Path) -> None:
    source_profile_path, source_site_path, source_profile_bytes, source_site_bytes = (
        _write_source(tmp_path)
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4_ckv_gather.py"),
            "--source-profile",
            str(source_profile_path),
            "--source-site",
            str(source_site_path),
            "--expected-profile-sha256",
            hashlib.sha256(source_profile_bytes).hexdigest(),
            "--expected-site-sha256",
            hashlib.sha256(source_site_bytes).hexdigest(),
            "--candidate-profile",
            str(source_profile_path),
            "--candidate-site",
            str(tmp_path / "should-not-exist.yaml"),
            "--rollback-profile",
            str(tmp_path / "should-not-exist.json"),
            "--rollback-site",
            str(tmp_path / "should-not-exist-rollback.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must not overwrite an input" in result.stderr
