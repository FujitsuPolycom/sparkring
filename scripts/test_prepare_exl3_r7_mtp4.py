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

import prepare_exl3_r7_mtp4 as mtp4  # noqa: E402
from test_prepare_exl3_r7_mtp3_kv925 import qualified_profile  # noqa: E402


def mtp3_kv925_profile() -> dict:
    profile = qualified_profile()
    profile["extra_vllm_args"].extend(
        ["--max-cudagraph-capture-size", "32"]
    )
    return profile


def mtp3_kv925_site() -> str:
    return (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 3\n"
        "  max_model_len: 65536\n"
        "  kv_cache_bytes_per_rank: 9250000000\n"
        "  max_num_seqs: 8\n"
    )


def _option_json(profile: dict, option: str) -> dict:
    arguments = profile["extra_vllm_args"]
    index = arguments.index(option)
    return json.loads(arguments[index + 1])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_candidate_is_exact_mtp3_to_mtp4_semantic_derivative() -> None:
    source = mtp3_kv925_profile()
    candidate = mtp4.derive_candidate(source)

    expected = copy.deepcopy(source)
    expected["profile_id"] = expected["profile_id"].removesuffix(
        "-fixed-mtp3"
    ) + "-fixed-mtp4"
    expected["environment"].update(
        {
            "VLLM_SPARK_MAX_QUERY_ROWS": "40",
            "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp4",
            "VLLM_SPARK_MTP_TOKENS": "4",
        }
    )
    arguments = expected["extra_vllm_args"]
    spec_index = arguments.index("--speculative-config") + 1
    spec = json.loads(arguments[spec_index])
    spec["num_speculative_tokens"] = 4
    arguments[spec_index] = json.dumps(spec, separators=(",", ":"))
    compilation_index = arguments.index("--compilation-config") + 1
    compilation = json.loads(arguments[compilation_index])
    compilation["cudagraph_capture_sizes"] = list(range(1, 41))
    arguments[compilation_index] = json.dumps(
        compilation, separators=(",", ":")
    )
    max_capture_index = arguments.index("--max-cudagraph-capture-size") + 1
    arguments[max_capture_index] = "40"

    assert candidate == expected
    assert _option_json(candidate, "--speculative-config")[
        "num_speculative_tokens"
    ] == 4
    assert _option_json(candidate, "--compilation-config")[
        "cudagraph_capture_sizes"
    ] == list(range(1, 41))
    mtp4.validate_candidate(source, candidate)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda profile: profile["environment"].__setitem__(
                "VLLM_SPARK_MAX_QUERY_ROWS", "39"
            ),
            "query-row ceiling 40",
        ),
        (
            lambda profile: profile["environment"].__setitem__(
                "VLLM_SPARK_MTP_MODE_ID", "fixed-mtp3"
            ),
            "mode fixed-mtp4",
        ),
        (
            lambda profile: profile["environment"].__setitem__(
                "VLLM_SPARK_MTP_TOKENS", "3"
            ),
            "speculative depth 4",
        ),
        (
            lambda profile: profile["extra_labels"].__setitem__(
                "org.sparkring.r7.online-k6-scope", "target+draft"
            ),
            "target-only",
        ),
    ],
)
def test_candidate_fails_closed_on_contract_drift(
    mutation, match: str
) -> None:
    source = mtp3_kv925_profile()
    candidate = mtp4.derive_candidate(source)
    mutation(candidate)
    with pytest.raises(mtp4.ContractError, match=match):
        mtp4.validate_candidate(source, candidate)


def test_candidate_rejects_speculation_or_capture_geometry_drift() -> None:
    source = mtp3_kv925_profile()

    candidate = mtp4.derive_candidate(source)
    arguments = candidate["extra_vllm_args"]
    spec_index = arguments.index("--speculative-config") + 1
    spec = json.loads(arguments[spec_index])
    spec["num_speculative_tokens"] = 3
    arguments[spec_index] = json.dumps(spec)
    with pytest.raises(mtp4.ContractError, match="speculative depth 4"):
        mtp4.validate_candidate(source, candidate)

    candidate = mtp4.derive_candidate(source)
    arguments = candidate["extra_vllm_args"]
    compilation_index = arguments.index("--compilation-config") + 1
    compilation = json.loads(arguments[compilation_index])
    compilation["cudagraph_capture_sizes"].remove(40)
    arguments[compilation_index] = json.dumps(compilation)
    with pytest.raises(mtp4.ContractError, match="capture Q1 through Q40"):
        mtp4.validate_candidate(source, candidate)

    candidate = mtp4.derive_candidate(source)
    arguments = candidate["extra_vllm_args"]
    max_capture_index = arguments.index("--max-cudagraph-capture-size") + 1
    arguments[max_capture_index] = "32"
    with pytest.raises(mtp4.ContractError, match="capture-size ceiling 40"):
        mtp4.validate_candidate(source, candidate)


def test_source_requires_exact_mtp3_capture_geometry() -> None:
    source = mtp3_kv925_profile()
    compilation_index = source["extra_vllm_args"].index(
        "--compilation-config"
    ) + 1
    compilation = json.loads(source["extra_vllm_args"][compilation_index])
    compilation["cudagraph_capture_sizes"] = [32]
    source["extra_vllm_args"][compilation_index] = json.dumps(compilation)

    with pytest.raises(mtp4.ContractError, match="capture Q1 through Q32"):
        mtp4.derive_candidate(source)


def test_site_derivative_changes_only_static_depth_at_kv925() -> None:
    source = mtp3_kv925_site()
    candidate = mtp4.derive_site_text(source)

    assert candidate == source.replace("  mtp_tokens: 3", "  mtp_tokens: 4")
    assert candidate.replace("  mtp_tokens: 4", "  mtp_tokens: 3") == source


@pytest.mark.parametrize(
    "site",
    [
        mtp3_kv925_site().replace("9250000000", "9500000000"),
        mtp3_kv925_site().replace("decode_context_parallel_size: 4", "decode_context_parallel_size: 1"),
        mtp3_kv925_site() + "  mtp_tokens: 3\n",
    ],
)
def test_site_derivative_rejects_nonqualified_source(site: str) -> None:
    with pytest.raises(mtp4.ContractError):
        mtp4.derive_site_text(site)


def test_cli_emits_candidate_and_byte_identical_mtp3_rollback(
    tmp_path: Path,
) -> None:
    source_profile_path = tmp_path / "mtp3.json"
    source_site_path = tmp_path / "mtp3.yaml"
    candidate_profile_path = tmp_path / "mtp4.json"
    candidate_site_path = tmp_path / "mtp4.yaml"
    rollback_profile_path = tmp_path / "mtp4-rollback.json"
    rollback_site_path = tmp_path / "mtp4-rollback.yaml"
    source_profile_bytes = (
        json.dumps(mtp3_kv925_profile(), indent=2) + "\n"
    ).replace("\n", "\r\n").encode()
    source_site_bytes = mtp3_kv925_site().encode()
    source_profile_path.write_bytes(source_profile_bytes)
    source_site_path.write_bytes(source_site_bytes)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4.py"),
            "--mtp3-profile",
            str(source_profile_path),
            "--mtp3-site",
            str(source_site_path),
            "--expected-mtp3-profile-sha256",
            _sha256(source_profile_bytes),
            "--expected-mtp3-site-sha256",
            _sha256(source_site_bytes),
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
    assert rollback_profile_path.read_bytes() == source_profile_bytes
    assert rollback_site_path.read_bytes() == source_site_bytes
    candidate_profile_bytes = candidate_profile_path.read_bytes()
    assert b"\r\n" in candidate_profile_bytes
    assert candidate_profile_bytes.count(b"\n") == candidate_profile_bytes.count(
        b"\r\n"
    )
    candidate = json.loads(candidate_profile_path.read_bytes())
    mtp4.validate_candidate(json.loads(source_profile_bytes), candidate)
    assert "  mtp_tokens: 4" in candidate_site_path.read_text()
    receipt = json.loads(result.stdout)
    assert receipt["source_profile_sha256"] == _sha256(source_profile_bytes)
    assert receipt["source_site_sha256"] == _sha256(source_site_bytes)
    assert receipt["rollback_profile_sha256"] == _sha256(source_profile_bytes)
    assert receipt["rollback_site_sha256"] == _sha256(source_site_bytes)


def test_cli_rejects_source_hash_drift_before_writing(tmp_path: Path) -> None:
    source_profile_path = tmp_path / "mtp3.json"
    source_site_path = tmp_path / "mtp3.yaml"
    candidate_profile_path = tmp_path / "mtp4.json"
    source_profile_path.write_text(
        json.dumps(mtp3_kv925_profile()), encoding="utf-8"
    )
    source_site_path.write_text(mtp3_kv925_site(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4.py"),
            "--mtp3-profile",
            str(source_profile_path),
            "--mtp3-site",
            str(source_site_path),
            "--expected-mtp3-profile-sha256",
            "0" * 64,
            "--expected-mtp3-site-sha256",
            _sha256(source_site_path.read_bytes()),
            "--candidate-profile",
            str(candidate_profile_path),
            "--candidate-site",
            str(tmp_path / "mtp4.yaml"),
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
    assert "profile SHA-256 mismatch" in result.stderr
    assert not candidate_profile_path.exists()


def test_cli_refuses_to_overwrite_inputs(tmp_path: Path) -> None:
    profile_path = tmp_path / "mtp3.json"
    site_path = tmp_path / "mtp3.yaml"
    profile_path.write_text(json.dumps(mtp3_kv925_profile()), encoding="utf-8")
    site_path.write_text(mtp3_kv925_site(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4.py"),
            "--mtp3-profile",
            str(profile_path),
            "--mtp3-site",
            str(site_path),
            "--expected-mtp3-profile-sha256",
            _sha256(profile_path.read_bytes()),
            "--expected-mtp3-site-sha256",
            _sha256(site_path.read_bytes()),
            "--candidate-profile",
            str(profile_path),
            "--candidate-site",
            str(tmp_path / "mtp4.yaml"),
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
