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


def test_generated_site_receives_exact_local_image(tmp_path):
    destination = tmp_path / "site.yaml"
    digest = "sha256:" + "a" * 64
    bootstrap_nf3.write_generated_site(
        SITE,
        destination,
        "sparkring/glm52-nf3:test",
        digest,
    )
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert document["runtime"]["container_image"] == "sparkring/glm52-nf3:test"
    assert document["runtime"]["container_image_digest"] == digest
    assert document["runtime"]["model_path"].endswith(
        "GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
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
