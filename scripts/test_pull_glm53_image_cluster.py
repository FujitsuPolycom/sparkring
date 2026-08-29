"""Contracts for immutable GLM-5.3 image distribution."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import pull_glm53_image_cluster as pull


IMAGE = "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:" + "a" * 64


def _site() -> SimpleNamespace:
    return SimpleNamespace(
        ranks=(
            SimpleNamespace(id=0, ssh_target="operator@rank0.example.net"),
            SimpleNamespace(id=1, ssh_target="operator@rank1.example.net"),
        )
    )


def test_plan_is_explicitly_mutating_and_digest_bound() -> None:
    plan = pull.plan_document(_site(), IMAGE)
    assert plan["safety"] == ["MUTATES HOST"]
    assert plan["image"] == IMAGE
    assert len(plan["actions"]) == 2
    assert "docker pull --platform linux/arm64" in plan["actions"][0]["command"][-1]
    with pytest.raises(pull.PullError, match="immutable registry reference"):
        pull.plan_document(_site(), "ghcr.io/example/image:moving-tag")


def test_pull_cluster_requires_one_local_image_id(monkeypatch: pytest.MonkeyPatch) -> None:
    documents = [
        [{"Id": "sha256:" + "b" * 64, "Architecture": "arm64", "Os": "linux"}],
        [{"Id": "sha256:" + "c" * 64, "Architecture": "arm64", "Os": "linux"}],
    ]

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, json.dumps(documents.pop(0)), "")

    monkeypatch.setattr(pull.subprocess, "run", fake_run)
    with pytest.raises(pull.PullError, match="different local image IDs"):
        pull.pull_cluster(_site(), IMAGE, 30)
