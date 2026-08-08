from __future__ import annotations

import json
from pathlib import Path
import subprocess

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
        assert "org.sparkring.component=engine" in command
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


def test_status_checks_restart_oom_and_resource_failure_signatures(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)
    actions = lmcache.server_health_actions(site) + lmcache.ready_actions(
        site, profile
    )
    assert len(actions) == 8
    for action in actions:
        command = action.shell_command
        assert ".RestartCount" in command
        assert ".State.OOMKilled" in command
        assert "CUDA out of memory" in command
        assert "OutOfMemoryError" in command


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
        "remove_engines",
        "remove_servers",
        "rollback",
        "verify_rollback",
    }
    assert all(len(actions) == 4 for actions in document["phases"].values())


@pytest.mark.parametrize(
    "command", ("start", "restart-engines", "restart-stack", "rollback")
)
def test_mutation_requires_exact_confirmation(tmp_path, command):
    site_path, profile_path = generated(tmp_path)
    with pytest.raises(SystemExit) as error:
        lmcache.main(
            [
                "--site",
                str(site_path),
                "--profile",
                str(profile_path),
                "--execute",
                command,
            ]
        )
    assert error.value.code == 2


def test_component_removal_and_rollback_verification_are_exactly_scoped(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)
    engines = lmcache.remove_component_actions(
        site, profile, component="engine"
    )
    servers = lmcache.remove_component_actions(
        site, profile, component="lmcache-server"
    )
    verification = lmcache.rollback_verify_actions(site, profile)
    for rank in range(4):
        assert exl3.container_name(profile, rank) in engines[rank].shell_command
        assert lmcache.server_name(rank) not in engines[rank].shell_command
        assert lmcache.server_name(rank) in servers[rank].shell_command
        assert exl3.container_name(profile, rank) not in servers[rank].shell_command
        assert lmcache.PROFILE_ID in engines[rank].shell_command
        assert "exit 73" in engines[rank].shell_command
        assert "exit 74" in engines[rank].shell_command
        assert exl3.container_name(profile, rank) in verification[rank].shell_command
        assert lmcache.server_name(rank) in verification[rank].shell_command


@pytest.mark.parametrize(
    ("command", "expected_phases"),
    (
        (
            "restart-engines",
            [
                "server_health",
                "remove_engines",
                "start_engines",
                "ready",
                "server_health",
            ],
        ),
        (
            "restart-stack",
            [
                "remove_engines",
                "remove_servers",
                "start_servers",
                "server_health",
                "start_engines",
                "ready",
                "server_health",
            ],
        ),
    ),
)
def test_restart_sequences_are_ordered_and_fail_closed(
    tmp_path, monkeypatch, capsys, command, expected_phases
):
    site_path, profile_path = generated(tmp_path)
    calls = []

    def fake_execute(actions, timeout):
        calls.append([action.shell_command for action in actions])
        return {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        }

    monkeypatch.setattr(exl3, "execute", fake_execute)
    assert (
        lmcache.main(
            [
                "--site",
                str(site_path),
                "--profile",
                str(profile_path),
                "--execute",
                "--confirmation",
                lmcache.CONFIRMATION,
                command,
            ]
        )
        == 0
    )
    plan = {
        "server_health": lmcache.server_health_actions,
        "remove_engines": lambda site, profile: lmcache.remove_component_actions(
            site, profile, component="engine"
        ),
        "remove_servers": lambda site, profile: lmcache.remove_component_actions(
            site, profile, component="lmcache-server"
        ),
        "start_servers": lmcache.server_start_actions,
        "start_engines": lmcache.engine_start_actions,
        "ready": lmcache.ready_actions,
    }
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)
    expected = []
    for phase in expected_phases:
        function = plan[phase]
        actions = (
            function(site)
            if phase == "server_health"
            else function(site, profile)
        )
        expected.append([action.shell_command for action in actions])
    assert calls == expected
    json.loads(capsys.readouterr().out)


def test_verify_rollback_is_read_only_and_needs_no_confirmation(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    calls = []

    def fake_execute(actions, timeout):
        calls.extend(actions)
        return {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        }

    monkeypatch.setattr(exl3, "execute", fake_execute)
    assert (
        lmcache.main(
            [
                "--site",
                str(site_path),
                "--profile",
                str(profile_path),
                "--execute",
                "verify-rollback",
            ]
        )
        == 0
    )
    assert len(calls) == 4
    assert all("docker ps -aq" in action.shell_command for action in calls)
    json.loads(capsys.readouterr().out)


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


def test_start_engine_phase_allows_full_model_verification(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    timeouts = []

    def fake_execute(actions, timeout):
        timeouts.append(timeout)
        return {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        }

    monkeypatch.setattr(exl3, "execute", fake_execute)
    assert (
        lmcache.main(
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
        == 0
    )
    profile = exl3.load_profile(profile_path)
    assert timeouts == [
        180,
        150,
        profile.startup_timeout_seconds + 60,
        profile.startup_timeout_seconds + 60,
    ]
    json.loads(capsys.readouterr().out)


def test_remote_timeout_output_is_json_serializable(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=kwargs["timeout"], output=b"partial stdout"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    action = exl3.RemoteAction(0, "rank0", ("true",))
    result = exl3.execute([action], timeout=1)
    assert result[0] == {
        "exit_code": 124,
        "stdout": "partial stdout",
        "stderr": "remote command timed out",
    }
    json.dumps(result)
