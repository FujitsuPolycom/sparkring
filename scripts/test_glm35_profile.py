"""External contracts for the consolidated GLM-5.2 3.5-bpw profile compiler."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import glm35_profile as profile  # noqa: E402


def source_mtp4_profile() -> dict:
    template = json.loads(profile.TEMPLATE_PATH.read_text(encoding="utf-8"))
    pins = json.loads(profile.PINS_PATH.read_text(encoding="utf-8"))
    recipe = json.loads(profile.RECIPE_PATH.read_text(encoding="utf-8"))
    stock = profile._derive_stock(template, pins, recipe)
    mtp2 = profile._derive_mtp2(stock)
    mtp3 = profile._derive_mtp3(stock, mtp2)
    return profile._derive_mtp4(mtp3)


def source_mtp4_site() -> str:
    return (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 4\n"
        "  max_model_len: 65536\n"
        "  kv_cache_bytes_per_rank: 9250000000\n"
        "  max_num_seqs: 16\n"
    )


def source_ckv_profile() -> dict:
    return profile._derive_ckv_gather(profile._derive_nvfp4(source_mtp4_profile()))


def sircl_artifact_digests() -> dict[str, str]:
    return {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in profile._SIRCL_ARTIFACTS
    }


def source_pre_q40_profile() -> dict:
    return profile._derive_sircl_tiered(
        source_ckv_profile(), artifact_digests=sircl_artifact_digests()
    )


def _artifact_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, (filename, _container) in profile._SIRCL_ARTIFACTS.items():
        path = tmp_path / "artifacts" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"public-{name}\n".encode())
        paths[name] = path
    return paths


def _resolved_inputs(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = "sparkring/glm52-exl3-r7-3.5bpw:test"
    image_id = "sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513"
    site_text = (ROOT / "scripts/config/exl3-r7-site.example.yaml").read_text(
        encoding="utf-8"
    )
    site_text = (
        site_text.replace("sparkring/glm52-exl3-r7-3.5bpw:REPLACE", image)
        .replace("sha256:" + "1" * 64, image_id)
        .replace(" - REPLACE.", ".")
    )
    site = tmp_path / "site.yaml"
    site.write_text(site_text, encoding="utf-8")
    template = json.loads(
        (ROOT / "scripts/config/exl3-r7-candidate.example.json").read_text(
            encoding="utf-8"
        )
    )
    template.update(
        {
            "image": image,
            "image_id": image_id,
            "model_host_path": "/models/glm52-exl3-r7-3.5bpw",
            "jit_cache_host_path": "/var/lib/sparkring/jit-cache",
        }
    )
    template_path = tmp_path / "candidate.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    return site, template_path


def _option(document: dict, name: str) -> str:
    arguments = document["extra_vllm_args"]
    return arguments[arguments.index(name) + 1]


def test_plan_is_dry_run_by_default(tmp_path: Path) -> None:
    output_dir = tmp_path / "not-created"
    receipt = profile.plan(output_dir=output_dir)
    assert receipt["dry_run"] is True
    assert not output_dir.exists()
    assert receipt["steps"][-1] == {
        "step": "tiered-sircl-tp-all-reduce",
        "safety": "OFFLINE",
    }


def test_dynamic_nvfp4_transformation_is_exact() -> None:
    source = source_mtp4_profile()
    candidate = profile._derive_nvfp4(source)
    assert candidate["profile_id"] == (
        f"{source['profile_id']}-nvfp4-rope8-ctx1m-b4096"
    )
    assert candidate["environment"]["KV_FP8_ROPE"] == "1"
    assert candidate["environment"]["VLLM_NVFP4_MLA_DYNAMIC_SCALE"] == "1"
    assert candidate["environment"]["VLLM_EXL3_PREFILL_CAPACITY"] == "4096"
    assert _option(candidate, "--max-num-batched-tokens") == "4096"
    assert _option(candidate, "--block-size") == "64"
    assert _option(candidate, "--kv-cache-dtype") == "nvfp4_ds_mla"
    assert candidate["extra_labels"]["org.sparkring.r7.kv-contract"] == (
        profile._NVFP4_KV_CONTRACT
    )
    expected_site = source_mtp4_site().replace(
        "max_model_len: 65536", "max_model_len: 1048576"
    )
    assert profile._derive_nvfp4_site(source_mtp4_site()) == expected_site


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("environment", "KV_FP8_ROPE"), "0", "allowlist"),
        (("environment", "VLLM_EXL3_PREFILL_CAPACITY"), "8192", "allowlist"),
        (("extra_labels", "unrelated"), "drift", "allowlist"),
    ],
)
def test_dynamic_nvfp4_candidate_fails_closed_on_drift(
    path: tuple[str, str], value: str, match: str
) -> None:
    source = source_mtp4_profile()
    candidate = profile._derive_nvfp4(source)
    candidate[path[0]][path[1]] = value
    with pytest.raises(profile.ProfileError, match=match):
        profile._validate_nvfp4_candidate(source, candidate)


def test_dynamic_nvfp4_source_rejects_preexisting_override() -> None:
    source = source_mtp4_profile()
    source["environment"]["VLLM_NVFP4_MLA_DYNAMIC_SCALE"] = "1"
    with pytest.raises(profile.ProfileError, match="already declares"):
        profile._derive_nvfp4(source)


def test_ckv_gather_transformation_and_workspace_are_exact() -> None:
    source = profile._derive_nvfp4(source_mtp4_profile())
    candidate = profile._derive_ckv_gather(source)
    expected = copy.deepcopy(source)
    expected["profile_id"] = profile._CKV_PROFILE_ID
    expected["environment"]["VLLM_B12X_MLA_CKV_GATHER"] = "1"
    expected["environment"]["VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS"] = "1048576"
    expected["extra_labels"][profile._CKV_LABEL] = profile._CKV_LABEL_VALUE
    assert candidate == expected
    assert profile._CKV_LOCAL_CAPACITY_TOKENS == 262_208
    assert profile._CKV_WORKSPACE_BYTES_PER_LANE == 868_432_896
    assert profile._CKV_WORKSPACE_POOL_BYTES_PER_RANK == 1_736_865_792


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("profile_id",), "unexpected", "profile_id"),
        (("environment", "VLLM_EXL3_PREFILL_CAPACITY"), "2048", "prefill"),
        (("environment", "KV_FP8_ROPE"), "0", "FP8-RoPE"),
        (
            ("environment", "VLLM_NVFP4_MLA_DYNAMIC_SCALE"),
            "0",
            "dynamic NVFP4",
        ),
        (
            ("environment", "VLLM_B12X_MLA_CKV_GATHER"),
            "1",
            "already declares",
        ),
    ],
)
def test_ckv_gather_source_fails_closed_on_drift(
    path: tuple[str, ...], value: str, match: str
) -> None:
    source = profile._derive_nvfp4(source_mtp4_profile())
    target = source
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(profile.ProfileError, match=match):
        profile._derive_ckv_gather(source)


def test_sircl_transformation_attests_exact_artifact_digests() -> None:
    source = source_ckv_profile()
    digests = sircl_artifact_digests()
    candidate = profile._derive_sircl_tiered(
        source, artifact_digests=digests
    )
    assert candidate["profile_id"].endswith("-sircl-tiered")
    assert candidate["environment"][
        "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL"
    ] == "two_slot_deferred_ack"
    assert candidate["environment"][
        "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY"
    ] == "tiered_64k"
    assert "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40" not in candidate["environment"]
    assert "VLLM_SPARK_TP4_PREFILL_CAPACITY_POOL" not in candidate["environment"]
    for name, (_filename, container) in profile._SIRCL_ARTIFACTS.items():
        matching = [
            volume
            for volume in candidate["extra_volumes"]
            if volume.get("container") == container
        ]
        assert len(matching) == 1
        assert f"{digests[name]}  {container}" in candidate["attestation_hook"][2]


def test_sircl_transformation_rejects_digest_or_profile_drift() -> None:
    source = source_ckv_profile()
    digests = sircl_artifact_digests()
    incomplete = dict(digests)
    incomplete.pop("backend")
    with pytest.raises(profile.ProfileError, match="incomplete"):
        profile._derive_sircl_tiered(source, artifact_digests=incomplete)
    candidate = profile._derive_sircl_tiered(
        source, artifact_digests=digests
    )
    candidate["environment"]["UNRELATED"] = "1"
    with pytest.raises(profile.ProfileError, match="allowlist"):
        profile._validate_sircl_candidate(
            source, candidate, artifact_digests=digests
        )


def test_execute_emits_byte_compatible_complete_pre_q40_outputs(
    tmp_path: Path,
) -> None:
    site, template = _resolved_inputs(tmp_path)
    artifacts = _artifact_paths(tmp_path)
    output_dir = tmp_path / "output"
    receipt = profile.plan(
        template=template,
        site=site,
        output_dir=output_dir,
        artifact_paths=artifacts,
        dry_run=False,
    )
    foundation = output_dir / "mtp4-kv925-profile.json"
    foundation_site = output_dir / "mtp4-kv925-site.yaml"
    nvfp4_path = output_dir / "mtp4-nvfp4-profile.json"
    ckv_path = output_dir / "mtp4-ckv-gather-profile.json"
    pre_q40_path = output_dir / "pre-q40-profile.json"
    pre_q40_site = output_dir / "pre-q40-site.yaml"
    pre_q40_receipt = output_dir / "pre-q40-receipt.json"

    mtp4_document = json.loads(foundation.read_bytes())
    expected_nvfp4 = profile._derive_nvfp4(mtp4_document)
    assert nvfp4_path.read_bytes() == profile._json_bytes(expected_nvfp4)
    expected_ckv = profile._derive_ckv_gather(expected_nvfp4)
    assert ckv_path.read_bytes() == profile._json_bytes(expected_ckv)
    expected_digests = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in artifacts.items()
    }
    expected_pre_q40 = profile._derive_sircl_tiered(
        expected_ckv, artifact_digests=expected_digests
    )
    expected_pre_q40_path = tmp_path / "expected-pre-q40.json"
    expected_pre_q40_path.write_text(
        json.dumps(expected_pre_q40, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert pre_q40_path.read_bytes() == expected_pre_q40_path.read_bytes()
    assert pre_q40_site.read_bytes() == (
        output_dir / "mtp4-ckv-gather-site.yaml"
    ).read_bytes()
    assert (
        (output_dir / "mtp4-nvfp4-rollback.json").read_bytes()
        == foundation.read_bytes()
    )
    assert (
        (output_dir / "mtp4-nvfp4-rollback-site.yaml").read_bytes()
        == foundation_site.read_bytes()
    )
    assert (
        (output_dir / "mtp4-ckv-gather-rollback.json").read_bytes()
        == nvfp4_path.read_bytes()
    )

    manifest = json.loads(pre_q40_receipt.read_text(encoding="utf-8"))
    expected_receipt = tmp_path / "expected-pre-q40-receipt.json"
    expected_receipt.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert pre_q40_receipt.read_bytes() == expected_receipt.read_bytes()
    assert manifest["schema"] == "sparkring-r7-sircl-tiered-bundle/v1"
    assert manifest["rollback"]["sha256"] == hashlib.sha256(
        ckv_path.read_bytes()
    ).hexdigest()
    assert manifest["policy"]["dual_port"] is False
    assert manifest["policy"]["prefill_capacity_pool"] is False
    for name, (filename, _container) in profile._SIRCL_ARTIFACTS.items():
        bundled = output_dir / "pre-q40-bundle" / filename
        assert bundled.read_bytes() == artifacts[name].read_bytes()
        assert manifest["files"][name]["sha256"] == expected_digests[name]

    for key, path in (
        ("mtp4_profile_sha256", foundation),
        ("mtp4_site_sha256", foundation_site),
        ("nvfp4_profile_sha256", nvfp4_path),
        ("ckv_gather_profile_sha256", ckv_path),
        ("pre_q40_profile_sha256", pre_q40_path),
        ("pre_q40_site_sha256", pre_q40_site),
        ("pre_q40_receipt_sha256", pre_q40_receipt),
    ):
        assert receipt[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["sircl_artifact_sha256"] == expected_digests
    assert receipt["ckv_workspace_pool_bytes_per_rank"] == 1_736_865_792
    assert receipt["reported_kv_capacity_tokens"] == 1_156_864


def test_complete_profile_preserves_stock_dcp_and_indexer_invariants(
    tmp_path: Path,
) -> None:
    site, template = _resolved_inputs(tmp_path)
    output_dir = tmp_path / "output"
    profile.plan(
        template=template,
        site=site,
        output_dir=output_dir,
        artifact_paths=_artifact_paths(tmp_path),
        dry_run=False,
    )
    generated = json.loads(
        (output_dir / "pre-q40-profile.json").read_text(encoding="utf-8")
    )
    environment = generated["environment"]
    assert environment["VLLM_NCCL_SO_PATH"] == (
        "/opt/sparkring/nccl/libnccl.so.2"
    )
    assert environment["SPARK_TP4_DCP_COLLECTIVE_AUDIT"] == "1"
    assert {
        "SPARK_TP4_ALLGATHER_BASE_PORT",
        "SPARK_TP4_ALLGATHER_ENABLE_CKV",
        "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT0",
        "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT1",
        "SPARK_TP4_GRAPH_INDEXER_PROGRESS_CPU",
        "SPARK_TP4_TRACE_ALLGATHER_SHAPES",
        "VLLM_SPARK_TP4_ALLGATHER_MODE",
        "VLLM_SPARK_TP4_ALLGATHER_POLICY",
        "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM",
    }.isdisjoint(environment)
    assert _option(generated, "--dcp-comm-backend") == "ag_rs"
    recipe = json.loads(profile.RECIPE_PATH.read_text(encoding="utf-8"))
    assert recipe["serving"]["dcp_transport"] == "stock-nccl"
    assert recipe["serving"]["indexer_transport"] == "stock-nccl"


def test_execute_rejects_unresolved_or_missing_artifact_inputs(
    tmp_path: Path,
) -> None:
    site, template = _resolved_inputs(tmp_path)
    unresolved = json.loads(template.read_text(encoding="utf-8"))
    unresolved["image_id"] = "sha256:" + "1" * 64
    template.write_text(json.dumps(unresolved), encoding="utf-8")
    with pytest.raises(profile.ProfileError, match="unresolved"):
        profile.plan(
            template=template,
            site=site,
            output_dir=tmp_path / "output",
            artifact_paths=_artifact_paths(tmp_path),
            dry_run=False,
        )

    site, template = _resolved_inputs(tmp_path / "resolved")
    artifacts = _artifact_paths(tmp_path / "resolved")
    artifacts["backend"].unlink()
    with pytest.raises(profile.ProfileError, match="not a regular file"):
        profile.plan(
            template=template,
            site=site,
            output_dir=tmp_path / "missing-output",
            artifact_paths=artifacts,
            dry_run=False,
        )
    assert not (tmp_path / "missing-output").exists()


def test_profile_name_describes_the_external_foundation() -> None:
    assert profile.PROFILE_NAME == "glm52-exl3-3.5bpw-fixed-mtp4-foundation"
