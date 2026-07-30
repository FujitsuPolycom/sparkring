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
    assert [rank["rank"] for rank in plan["ranks"]] == [0, 1, 2, 3]
    assert plan["image"].startswith("sparkring/glm52-nf3:")


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
