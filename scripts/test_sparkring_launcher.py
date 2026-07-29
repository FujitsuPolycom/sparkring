"""Offline tests for the public four-rank launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sparkring_launcher as launcher
from sparkring_site import load_site

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts/config/site.example.yaml"
LAUNCH = ROOT / "scripts/config/launch.example.json"


def test_example_produces_four_safe_start_actions():
    site = load_site(SITE)
    config = launcher.load_launch(LAUNCH)
    actions = launcher.start_actions(site, config)
    assert [action.rank for action in actions] == [0, 1, 2, 3]
    assert all(action.argv[:3] == ("docker", "run", "--detach") for action in actions)
    assert all("--rm" not in action.argv for action in actions)
    for rank, action in enumerate(actions):
        assert f"RANK={rank}" in action.argv
        assert "org.sparkring.managed=true" in action.argv
        assert "WORLD_SIZE=4" in action.argv
        assert site.runtime.container_image in action.argv
        assert "SPARKRING_IMAGE_DIGEST=" + site.runtime.container_image_digest in action.argv
        assert "B12X_MLA_SPARSE" in action.argv


def test_plan_is_connection_free(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run plan attempted remote execution")

    monkeypatch.setattr(launcher, "execute", forbidden)
    rc = launcher.main(
        [
            "--site",
            str(SITE),
            "--launch-config",
            str(LAUNCH),
            "plan",
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert document["mutates_remote"] is False
    assert len(document["actions"]) == 4


def test_mutating_command_without_execute_is_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(
        launcher,
        "execute",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("execute was called")
        ),
    )
    assert (
        launcher.main(
            [
                "--site",
                str(SITE),
                "--launch-config",
                str(LAUNCH),
                "stop",
            ]
        )
        == 0
    )
    assert "made no remote connection" in capsys.readouterr().err


def test_unknown_placeholder_fails_closed(tmp_path):
    document = json.loads(LAUNCH.read_text(encoding="utf-8"))
    document["environment"]["BAD"] = "{surprise}"
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(launcher.LaunchConfigError, match="unknown placeholder"):
        launcher.load_launch(path)


def test_site_model_must_match_runtime_lock(tmp_path):
    text = SITE.read_text(encoding="utf-8").replace(
        "aidendle94/GLM-5.2-MXFP4-Experts-GPTQ",
        "someone/other-model",
    )
    path = tmp_path / "site.yaml"
    path.write_text(text, encoding="utf-8")
    site = load_site(path)
    config = launcher.load_launch(LAUNCH)
    with pytest.raises(launcher.LaunchConfigError, match="differs from"):
        launcher.start_actions(site, config)


def test_start_failure_requests_all_rank_rollback(monkeypatch, capsys):
    calls = []

    def fake_execute(actions, timeout):
        calls.append(actions)
        if len(calls) == 1:
            return {
                action.rank: {
                    "exit_code": 1 if action.rank == 2 else 0,
                    "stdout": "",
                    "stderr": "",
                }
                for action in actions
            }
        return {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        }

    monkeypatch.setattr(launcher, "execute", fake_execute)
    rc = launcher.main(
        [
            "--site",
            str(SITE),
            "--launch-config",
            str(LAUNCH),
            "--execute",
            "start",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert len(calls) == 2
    assert [action.rank for action in calls[1]] == [0, 1, 3]
    assert all(action.argv[:2] == ("sh", "-c") for action in calls[1])
    assert result["rollback_results"] is not None
