from __future__ import annotations

import json
from pathlib import Path

import pytest

import bootstrap_exl3
import sparkring_exl3_launcher as exl3
import sparkring_exl3_lmcache_launcher as lmcache
from sparkring_site import load_site


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts/config/site.example.yaml"
IMAGE_ID = "sha256:" + "a" * 64


def generated(tmp_path):
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "launch.json"
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID
    )
    bootstrap_exl3.write_generated_profile(
        profile_path,
        "sparkring/exl3:test",
        IMAGE_ID,
        "/srv/models/exl3",
        "/srv/jit",
    )
    return site_path, profile_path


def test_public_cs512_plan_has_four_local_servers_and_four_engines(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)

    servers = lmcache.server_start_actions(site, profile)
    engines = lmcache.engine_start_actions(site, profile)
    assert len(servers) == len(engines) == 4
    for rank, action in enumerate(servers):
        command = action.shell_command
        assert f"spark-r{rank}-cs512" in command
        assert "--chunk-size 512" in command
        assert "--l1-size-gb 1" in command
        assert "--l1-init-size-gb 0" in command
        assert "--l1-use-lazy" in command
        assert "--port 6556" in command
        assert "--http-port 18081" in command
        assert IMAGE_ID in command
    addresses = [str(rank.management.address) for rank in site.ranks]
    for action in engines:
        command = action.shell_command
        assert "LMCacheMPConnector" in command
        assert "lmcache.integration.vllm.lmcache_mp_connector" in command
        assert "kv_load_failure_policy" in command
        assert "recompute" in command
        assert "--privileged" in command
        for address in addresses:
            assert f"tcp://{address}:6556" in command


def test_rollback_is_exact_name_and_profile_label_guarded(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)
    actions = lmcache.rollback_actions(site, profile)
    assert len(actions) == 4
    for rank, action in enumerate(actions):
        command = action.shell_command
        assert exl3.container_name(profile, rank) in command
        assert lmcache.server_name(rank) in command
        assert lmcache.PROFILE_ID in command
        assert "docker rm --force" in command
        assert "exit 73" in command


def test_plan_is_dry_run_and_records_every_phase(tmp_path, capsys):
    site_path, profile_path = generated(tmp_path)
    assert (
        lmcache.main(
            [
                "--site",
                str(site_path),
                "--profile",
                str(profile_path),
                "plan",
            ]
        )
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document["mutates_remote"] is False
    assert set(document["phases"]) == {
        "start_servers",
        "server_health",
        "start_engines",
        "ready",
        "rollback",
    }
    assert all(len(actions) == 4 for actions in document["phases"].values())


def test_execute_requires_exact_confirmation(tmp_path):
    site_path, profile_path = generated(tmp_path)
    with pytest.raises(SystemExit) as error:
        lmcache.main(
            [
                "--site",
                str(site_path),
                "--profile",
                str(profile_path),
                "--execute",
                "start",
            ]
        )
    assert error.value.code == 2


def test_start_failure_triggers_scoped_rollback(tmp_path, monkeypatch, capsys):
    site_path, profile_path = generated(tmp_path)
    calls = []

    def fake_execute(actions, timeout):
        calls.append([action.shell_command for action in actions])
        failed = any("/v1/models" in command for command in calls[-1])
        return {
            action.rank: {
                "exit_code": 1 if failed and action.rank == 0 else 0,
                "stdout": "",
                "stderr": "",
            }
            for action in actions
        }

    monkeypatch.setattr(exl3, "execute", fake_execute)
    result = lmcache.main(
        [
            "--site",
            str(site_path),
            "--profile",
            str(profile_path),
            "--execute",
            "--confirmation",
            lmcache.CONFIRMATION,
            "start",
        ]
    )
    assert result == 1
    assert len(calls) == 5
    assert all("docker rm --force" in command for command in calls[-1])
    payload = json.loads(capsys.readouterr().out)
    assert "rollback" in payload
