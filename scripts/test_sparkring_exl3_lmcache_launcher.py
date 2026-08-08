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


def test_experimental_memory_profile_is_explicit_and_consistent(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)
    experiment_id = "kv4gb-480k-l1-0.5gb"
    effective_label = lmcache.experiment_label(experiment_id)

    phases = lmcache.build_phases(
        site, profile, experiment_id=experiment_id
    )
    for action in phases["start_servers"]:
        command = action.shell_command
        assert "--l1-size-gb 0.5" in command
        assert f"org.sparkring.exl3-profile={effective_label}" in command
        assert f"org.sparkring.experiment={experiment_id}" in command
        assert "--chunk-size 512" in command
    for action in phases["start_engines"]:
        command = action.shell_command
        assert "--kv-cache-memory-bytes 4000000000" in command
        assert "--max-model-len 491520" in command
        assert "VLLM_SPARK_KV_CACHE_MEMORY_BYTES=4000000000" in command
        assert "VLLM_SPARK_MAX_MODEL_LEN=491520" in command
        assert "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=491520" in command
        assert f"org.sparkring.exl3-profile={effective_label}" in command
        assert f"org.sparkring.experiment={experiment_id}" in command
    for phase in (
        "server_health",
        "ready",
        "remove_engines",
        "remove_servers",
        "rollback",
    ):
        assert all(
            effective_label in action.shell_command
            for action in phases[phase]
        )


def test_experimental_plan_is_truthfully_noncanonical(tmp_path, capsys):
    site_path, profile_path = generated(tmp_path)
    experiment_id = "kv4gb-480k-l1-0.5gb"
    assert (
        lmcache.main(
            [
                "--site",
                str(site_path),
                "--profile",
                str(profile_path),
                "--experimental-memory-profile",
                experiment_id,
                "plan",
            ]
        )
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document["profile_id"] == lmcache.PROFILE_ID
    assert document["experiment"] == {
        "evidence_scope": "experimental",
        "canonical_configuration": False,
        "experiment_id": experiment_id,
        "base_profile_attestation": {
            "profile_id": lmcache.PROFILE_ID,
            "image_id": IMAGE_ID,
        },
        "effective_profile_label": lmcache.experiment_label(experiment_id),
        "baseline": {
            "kv_cache_memory_bytes": 4_500_000_000,
            "max_model_len": 524_288,
            "lmcache_l1_size_gb": 1,
        },
        "effective": {
            "kv_cache_memory_bytes": 4_000_000_000,
            "max_model_len": 491_520,
            "lmcache_l1_size_gb": 0.5,
        },
    }


def test_unknown_programmatic_experiment_fails_closed(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)
    with pytest.raises(exl3.ProfileError, match="unknown experimental memory profile"):
        lmcache.build_phases(site, profile, experiment_id="unreviewed")


def test_default_plan_remains_canonical(tmp_path, capsys):
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
    assert "experiment" not in document
    commands = json.dumps(document["phases"])
    assert "--l1-size-gb 1" in commands
    assert "--kv-cache-memory-bytes 4500000000" in commands
    assert "--max-model-len 524288" in commands
    assert "org.sparkring.experiment" not in commands


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


@pytest.mark.parametrize(
    ("command", "expected_on_failure"),
    (
        ("status", "continue"),
        ("rollback", "return-failure"),
        ("verify-rollback", "return-failure"),
    ),
)
def test_main_consumes_lifecycle_oracle_for_special_commands(
    tmp_path, monkeypatch, capsys, command, expected_on_failure
):
    site_path, profile_path = generated(tmp_path)
    original_sequence = lmcache.lifecycle_sequence
    observed = []

    def recording_sequence(requested_command, profile):
        observed.append(requested_command)
        sequence = original_sequence(requested_command, profile)
        assert {step["on_failure"] for step in sequence} == {
            expected_on_failure
        }
        return sequence

    monkeypatch.setattr(lmcache, "lifecycle_sequence", recording_sequence)
    monkeypatch.setattr(
        exl3,
        "execute",
        lambda actions, timeout: {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        },
    )
    argv = [
        "--site", str(site_path), "--profile", str(profile_path), "--execute",
    ]
    if command == "rollback":
        argv.extend(["--confirmation", lmcache.CONFIRMATION])
    argv.append(command)
    assert lmcache.main(argv) == 0
    assert observed == [command]
    assert json.loads(capsys.readouterr().out)


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
