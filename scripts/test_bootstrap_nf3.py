"""Offline tests for the one-command NF3 bootstrap."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

import bootstrap_nf3

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts/config/site.example.yaml"


def test_plan_is_read_only_and_names_all_four_ranks():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bootstrap_nf3.py"),
            "plan",
            "--site",
            str(SITE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schema"] == "sparkring-nf3-bootstrap-plan/v1"
    assert plan["profile"] == "fp8"
    assert [rank["rank"] for rank in plan["ranks"]] == [0, 1, 2, 3]
    assert plan["image"].startswith("sparkring/glm52-nf3:")


def test_plan_selects_nvfp4_rope8_without_changing_model_downloads():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bootstrap_nf3.py"),
            "plan",
            "--site",
            str(SITE),
            "--profile",
            "nvfp4-rope8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["profile"] == "nvfp4-rope8"
    assert plan["image"].startswith("sparkring/glm52-nf3-nvfp4-rope8:")
    assert plan["model_path"].endswith(
        "GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
    )
    assert plan["draft_path"].endswith("GLM-5.2-NF3-MTP-Draft")
    assert "build the thin NVFP4-latent/FP8-RoPE compatibility layer" in (
        plan["steps"]
    )


def test_execute_requires_confirmation_before_mutation():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bootstrap_nf3.py"),
            "execute",
            "--site",
            str(SITE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert bootstrap_nf3.CONFIRMATION in result.stderr


def test_bootstrap_verifies_rank0_management_fanout_scope():
    command = bootstrap_nf3.ssh_bootstrap_verification_command(SITE)
    assert command[-4:] == [
        "--site",
        str(SITE),
        "--scope",
        "bootstrap",
    ]


def test_nf3_contract_pins_exact_target_and_mtp_draft():
    recipe = bootstrap_nf3.load_nf3_contract()
    model = recipe["model"]
    draft = model["mtp_draft"]
    assert model["repository"] == (
        "madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
    )
    assert model["revision"] == "66f3623dd8fefb5ca8046706912d5d31c8d196af"
    assert model["index_sha256"] == (
        "6eb773222d932418dd0530c63aca498f86ef424da2a4526ccba76b59726da234"
    )
    assert draft["repository"] == "aidendle94/GLM-5.2-MXFP4-Experts-GPTQ"
    assert draft["revision"] == "46537e0e16fcd156627800139b41b9c497fc7ee2"
    assert draft["weight_sha256"] == (
        "0ade0e3da08e7e6c7b1f20e4c4e8d5d3b26b81103cea22f2ead9909c7d3d0732"
    )


def test_generated_site_replaces_stale_identity_and_kv_contract(tmp_path):
    stale = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    stale["runtime"].update(
        {
            "model_path": "/models/old-checkpoint",
            "model_repo": "old-owner/old-checkpoint",
            "model_revision": "0" * 40,
            "checkpoint_sha256": "f" * 64,
        }
    )
    stale["serving"]["kv_cache_bytes_per_rank"] = 3_000_000_000
    source = tmp_path / "stale-site.yaml"
    source.write_text(yaml.safe_dump(stale), encoding="utf-8")

    recipe = json.loads(
        bootstrap_nf3.RECIPE_PATH.read_text(encoding="utf-8")
    )
    model = recipe["model"]
    serving = recipe["serving"]
    digest = "sha256:" + "a" * 64

    for profile in bootstrap_nf3.PROFILES:
        destination = tmp_path / f"site-{profile}.yaml"
        bootstrap_nf3.write_generated_site(
            source,
            destination,
            f"sparkring/glm52-nf3:{profile}",
            digest,
            profile,
        )
        document = yaml.safe_load(destination.read_text(encoding="utf-8"))
        runtime = document["runtime"]
        assert runtime["container_image"] == (
            f"sparkring/glm52-nf3:{profile}"
        )
        assert runtime["container_image_digest"] == digest
        assert runtime["model_path"] == f"/models/{model['install_subdir']}"
        assert runtime["model_repo"] == model["repository"]
        assert runtime["model_revision"] == model["revision"]
        assert runtime["checkpoint_sha256"] == model["index_sha256"]
        assert serving["kv_cache_bytes_per_rank"] == 7_000_000_000
        assert (
            document["serving"]["kv_cache_bytes_per_rank"]
            == serving["kv_cache_bytes_per_rank"]
        )


def test_generated_nvfp4_rope8_launch_is_an_exact_profile(tmp_path):
    source = ROOT / "scripts/config/launch.example.json"
    destination = tmp_path / "launch.json"
    bootstrap_nf3.write_generated_launch(
        source,
        destination,
        "nvfp4-rope8",
    )
    document = json.loads(destination.read_text(encoding="utf-8"))
    index = document["extra_vllm_args"].index("--kv-cache-dtype")
    assert document["extra_vllm_args"][index + 1] == "nvfp4_ds_mla"
    assert document["environment"]["VLLM_SPARK_KV_PROFILE"] == (
        "nvfp4-rope8"
    )
    assert document["environment"]["VLLM_SPARK_KV_CACHE_DTYPE"] == (
        "nvfp4_ds_mla"
    )
    assert document["environment"]["VLLM_NVFP4_MLA_PER_TOKEN_SCALE"] == "1"
    assert document["environment"]["VLLM_SPARK_KV_SCALE_MODE"] == "per-token"


def test_generated_fp8_launch_removes_nvfp4_only_controls(tmp_path):
    source = ROOT / "scripts/config/launch.example.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["environment"]["VLLM_NVFP4_MLA_PER_TOKEN_SCALE"] = "1"
    dirty_source = tmp_path / "source.json"
    dirty_source.write_text(json.dumps(document), encoding="utf-8")
    destination = tmp_path / "launch.json"

    bootstrap_nf3.write_generated_launch(dirty_source, destination, "fp8")

    generated = json.loads(destination.read_text(encoding="utf-8"))
    assert generated["environment"]["VLLM_SPARK_KV_PROFILE"] == "fp8"
    assert "VLLM_NVFP4_MLA_PER_TOKEN_SCALE" not in generated["environment"]


def test_image_transfer_capacity_covers_archive_import_and_headroom():
    archive_bytes = 24 * 1024**3
    required = bootstrap_nf3.required_image_transfer_bytes(archive_bytes)
    assert required == (
        2 * archive_bytes + bootstrap_nf3.IMAGE_TRANSFER_HEADROOM_BYTES
    )
    command = bootstrap_nf3.image_transfer_capacity_command(archive_bytes)
    assert "docker info --format '{{.DockerRootDir}}'" in command
    assert "df -PB1 /var/tmp" in command
    assert 'df -PB1 "$docker_root"' in command
    assert f' -lt {required} ' in command
    assert "IMAGE_TRANSFER_CAPACITY_OK" in command


def test_image_transfer_capacity_rejects_invalid_archive_sizes():
    for invalid in (0, -1, 1.5, "100"):
        try:
            bootstrap_nf3.required_image_transfer_bytes(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid archive size {invalid!r}")
