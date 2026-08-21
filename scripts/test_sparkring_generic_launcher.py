"""Focused offline tests for the native generic launcher profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_generic_launcher as generic  # noqa: E402
import sparkring_runtime as runtime  # noqa: E402
from sparkring_site import load_site  # noqa: E402

SITE = ROOT / "scripts/config/exl3-r7-site.example.yaml"
IMAGE_ID = "sha256:" + "a" * 64


def _native_document(**overrides):
    document = {
        "schema": runtime.SCHEMA,
        "profile_id": "deepseek-runtime",
        "model_family": "deepseek",
        "engine": "docker",
        "container_name": "deepseek-runtime",
        "image": "sparkring/deepseek:test",
        "image_id": IMAGE_ID,
        "model_host_path": "/srv/models/deepseek",
        "model_container_path": "/models",
        "shm_size": "16g",
        "startup_timeout_seconds": 300,
        "environment": {},
        "extra_vllm_args": [],
    }
    document.update(overrides)
    return document


def _write_profile(tmp_path, name, document):
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_native_profile_loads_with_declared_identity(tmp_path):
    profile = generic.load_profile(
        _write_profile(
            tmp_path,
            "native.json",
            _native_document(identity={"model_revision": "b" * 40}),
        )
    )

    assert profile.profile_id == "deepseek-runtime"
    assert profile.image_id == IMAGE_ID
    assert profile.identity == {"model_revision": "b" * 40}


@pytest.mark.parametrize(
    "document, error",
    [
        (_native_document(schema="unsupported/v1"), "unsupported schema"),
        (_native_document(unknown_field=True), "unknown key"),
    ],
)
def test_profile_validation_rejects_unsupported_or_unknown_input(
    tmp_path, document, error,
):
    with pytest.raises(runtime.ProfileError, match=error):
        generic.load_profile(_write_profile(tmp_path, "invalid.json", document))


def test_plan_start_status_and_stop_actions_preserve_native_identity(tmp_path):
    profile = generic.load_profile(
        _write_profile(tmp_path, "native.json", _native_document())
    )
    site = load_site(SITE)

    plan_actions = generic.build_actions(site, profile, "plan")
    start_actions = generic.build_actions(site, profile, "start")
    status_actions = generic.build_actions(site, profile, "status")
    stop_actions = generic.build_actions(site, profile, "stop")
    plan = runtime.plan_document("plan", plan_actions, profile)

    assert plan_actions == start_actions
    assert len(start_actions) == len(site.ranks) == 4
    assert (
        f"test \"$(docker image inspect --format '{{{{.Id}}}}' "
        f"{profile.image})\" = {profile.image_id}"
    ) in start_actions[0].argv[2]
    assert "org.sparkring.profile=deepseek-runtime" in start_actions[0].shell_command
    assert status_actions[0].argv == (
        "docker", "inspect", "--format", "{{.State.Status}}",
        "deepseek-runtime-r0",
    )
    assert "org.sparkring.managed" in stop_actions[0].shell_command
    assert "docker rm --force deepseek-runtime-r0" in stop_actions[0].shell_command
    assert plan["profile_attestation"] == {
        "profile_id": "deepseek-runtime",
        "image_id": IMAGE_ID,
        "declared_identity": {},
    }


def test_start_requires_container_identifier_and_rolls_back_partial_start(
    monkeypatch, tmp_path, capsys,
):
    profile_path = _write_profile(tmp_path, "native.json", _native_document())
    calls = []

    def fake_execute(actions, timeout):
        calls.append(actions)
        if len(calls) == 1:
            return {
                0: {"exit_code": 0, "stdout": "a" * 64 + "\n", "stderr": ""},
                1: {"exit_code": 0, "stdout": "started\n", "stderr": ""},
                2: {"exit_code": 1, "stdout": "", "stderr": "failed"},
                3: {"exit_code": 0, "stdout": "d" * 64 + "\n", "stderr": ""},
            }
        return {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        }

    monkeypatch.setattr(runtime, "execute", fake_execute)

    assert generic.main([
        "--site", str(SITE), "--profile", str(profile_path), "--execute", "start",
    ]) == 1
    result = json.loads(capsys.readouterr().out)

    assert runtime.check_results(
        "start",
        {1: {"exit_code": 0, "stdout": "started\n", "stderr": ""}},
    ) == [1]
    assert runtime.should_rollback("start") is True
    assert runtime.should_rollback("stop") is False
    assert [action.rank for action in calls[1]] == [0, 3]
    assert result["passed"] is False
    assert "execution_mode" not in result
