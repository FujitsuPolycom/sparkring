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
import prepare_exl3_r7_sircl_tiered as sircl  # noqa: E402
from test_prepare_exl3_r7_mtp4_nvfp4 import (  # noqa: E402
    source_profile as mtp4_profile,
)


def source_profile() -> dict:
    return ckv.derive_candidate(nvfp4.derive_candidate(mtp4_profile()))


def artifact_digests() -> dict[str, str]:
    return {
        name: hashlib.sha256(name.encode()).hexdigest() for name in sircl.ARTIFACTS
    }


def test_candidate_adds_only_tiered_deferred_contract() -> None:
    source = source_profile()
    digests = artifact_digests()
    candidate = sircl.derive_candidate(source, artifact_digests=digests)

    assert candidate["profile_id"].endswith("-sircl-tiered")
    assert candidate["environment"][
        "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL"
    ] == "two_slot_deferred_ack"
    assert candidate["environment"][
        "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY"
    ] == "tiered_64k"
    assert candidate["environment"]["SPARK_TP4_CONTROL_PORT0"] == "11100"
    assert candidate["environment"]["SPARK_TP4_CONTROL_PORT1"] == "11101"
    assert "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40" not in candidate["environment"]
    assert "VLLM_SPARK_TP4_PREFILL_CAPACITY_POOL" not in candidate["environment"]
    for name, (_filename, container) in sircl.ARTIFACTS.items():
        matching = [
            volume
            for volume in candidate["extra_volumes"]
            if volume.get("container") == container
        ]
        assert len(matching) == 1
        assert f"{digests[name]}  {container}" in candidate["attestation_hook"][2]
    sircl.validate_candidate(source, candidate, artifact_digests=digests)


def test_candidate_rejects_unrelated_profile_change() -> None:
    source = source_profile()
    digests = artifact_digests()
    candidate = sircl.derive_candidate(source, artifact_digests=digests)
    candidate["environment"]["UNRELATED"] = "1"
    with pytest.raises(sircl.ContractError, match="allowlist"):
        sircl.validate_candidate(source, candidate, artifact_digests=digests)


def test_source_rejects_a_preexisting_selector() -> None:
    source = source_profile()
    source["environment"]["VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY"] = "serial"
    with pytest.raises(sircl.ContractError, match="already declares"):
        sircl.derive_candidate(source, artifact_digests=artifact_digests())


def test_digest_inventory_must_be_complete() -> None:
    digests = artifact_digests()
    digests.pop("backend")
    with pytest.raises(sircl.ContractError, match="incomplete"):
        sircl.derive_candidate(source_profile(), artifact_digests=digests)


def _write_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path], str]:
    base = tmp_path / "base.json"
    base.write_text(json.dumps(source_profile(), indent=2) + "\n", encoding="utf-8")
    artifacts: dict[str, Path] = {}
    for name, (filename, _container) in sircl.ARTIFACTS.items():
        path = tmp_path / f"input-{filename}"
        path.write_bytes(f"public-{name}\n".encode())
        artifacts[name] = path
    # The CLI derives the query-row provider as the backend's sibling
    # rather than taking it as an argument, so the fixture places it there.
    (tmp_path / "spark_tp4_query_row_provider.py").write_bytes(
        b"public-query_row_provider\n"
    )
    return base, artifacts, hashlib.sha256(base.read_bytes()).hexdigest()


def test_cli_builds_exclusive_hash_bound_bundle(tmp_path: Path) -> None:
    base, artifacts, base_sha = _write_inputs(tmp_path)
    profile = tmp_path / "candidate.json"
    manifest = tmp_path / "manifest.json"
    bundle = tmp_path / "bundle"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_sircl_tiered.py"),
            "--base-profile",
            str(base),
            "--expected-base-profile-sha256",
            base_sha,
            "--transport-library",
            str(artifacts["transport_library"]),
            "--backend",
            str(artifacts["backend"]),
            "--port-namespace",
            str(artifacts["port_namespace"]),
            "--capacity-pool",
            str(artifacts["capacity_pool"]),
            "--bundle",
            str(bundle),
            "--output-profile",
            str(profile),
            "--output-manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(manifest.read_text())
    assert receipt["schema"] == "sparkring-r7-sircl-tiered-bundle/v1"
    assert receipt["rollback"]["sha256"] == base_sha
    assert receipt["policy"]["dual_port"] is False
    assert (bundle / "spark_tp4_query_row_provider.py").is_file()
    assert receipt["policy"]["prefill_capacity_pool"] is False
    for name, (filename, _container) in sircl.ARTIFACTS.items():
        assert (bundle / filename).read_bytes() == artifacts[name].read_bytes()
        assert receipt["files"][name]["sha256"] == hashlib.sha256(
            artifacts[name].read_bytes()
        ).hexdigest()


def test_prepare_rejects_base_hash_drift_before_bundle_creation(tmp_path: Path) -> None:
    base, artifacts, _base_sha = _write_inputs(tmp_path)
    bundle = tmp_path / "bundle"
    with pytest.raises(sircl.ContractError, match="base profile SHA-256 mismatch"):
        sircl.prepare(
            base_profile=base,
            expected_base_profile_sha256="0" * 64,
            artifact_paths=artifacts,
            bundle=bundle,
        )
    assert not bundle.exists()


def test_profile_diff_is_deterministic() -> None:
    source = source_profile()
    first = sircl.derive_candidate(source, artifact_digests=artifact_digests())
    second = sircl.derive_candidate(
        copy.deepcopy(source), artifact_digests=artifact_digests()
    )
    assert first == second
