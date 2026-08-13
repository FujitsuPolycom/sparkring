from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_exl3_r7_mtp3 as mtp3  # noqa: E402
import prepare_exl3_r7_mtp3_kv925 as kv925  # noqa: E402
from test_prepare_exl3_r7_mtp3 import mtp2_profile  # noqa: E402


def qualified_profile() -> dict:
    stock, control = mtp2_profile()
    return mtp3.derive_candidate(stock, control)


def qualified_site() -> str:
    return (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 3\n"
        "  max_model_len: 65536\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n"
    )


def test_candidate_changes_only_kv_cache_bytes() -> None:
    source = qualified_site()
    candidate = kv925.derive_candidate_site(source)

    assert candidate == source.replace("9000000000", "9250000000")
    assert candidate.replace("9250000000", "9000000000") == source
    assert kv925.EXPECTED_CAPACITY_TOKENS == 675_840


@pytest.mark.parametrize(
    "site",
    [
        qualified_site().replace("9000000000", "9500000000"),
        qualified_site() + "  kv_cache_bytes_per_rank: 9000000000\n",
    ],
)
def test_candidate_rejects_nonqualified_site(site: str) -> None:
    with pytest.raises(kv925.ContractError, match="must declare exactly"):
        kv925.derive_candidate_site(site)


def test_cli_emits_byte_identical_profiles_and_rollback(tmp_path: Path) -> None:
    profile_path = tmp_path / "qualified.json"
    site_path = tmp_path / "qualified.yaml"
    candidate_profile = tmp_path / "candidate.json"
    candidate_site = tmp_path / "candidate.yaml"
    rollback_profile = tmp_path / "rollback.json"
    rollback_site = tmp_path / "rollback.yaml"
    profile_bytes = (json.dumps(qualified_profile(), indent=2) + "\n").encode()
    site_bytes = qualified_site().encode()
    profile_path.write_bytes(profile_bytes)
    site_path.write_bytes(site_bytes)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp3_kv925.py"),
            "--qualified-profile",
            str(profile_path),
            "--qualified-site",
            str(site_path),
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
    assert candidate_profile.read_bytes() == profile_bytes
    assert rollback_profile.read_bytes() == profile_bytes
    assert rollback_site.read_bytes() == site_bytes
    assert "kv_cache_bytes_per_rank: 9250000000" in candidate_site.read_text()
    assert json.loads(result.stdout)["predicted_dcp_global_token_capacity"] == 675_840


def test_cli_refuses_to_overwrite_inputs(tmp_path: Path) -> None:
    profile_path = tmp_path / "qualified.json"
    site_path = tmp_path / "qualified.yaml"
    profile_path.write_text(json.dumps(qualified_profile()), encoding="utf-8")
    site_path.write_text(qualified_site(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp3_kv925.py"),
            "--qualified-profile",
            str(profile_path),
            "--qualified-site",
            str(site_path),
            "--candidate-profile",
            str(profile_path),
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
    assert "must not overwrite an input" in result.stderr
