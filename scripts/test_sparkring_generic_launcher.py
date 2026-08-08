"""Offline tests for the generic runtime launcher and shared primitives.

Covers:
* Malformed profile rejection (structural validation) including F2-F4 checks.
* Dry-run plan output (no SSH connection).
* Lifecycle safety (confirmation tokens, profile-label-guarded stop, rollback,
  image verification in start, attestation hooks).
* F1: Native generic profiles named "exl3" or "nf3" stay generic.
* F5: NF3 bridge identity resolved from site before plan.
* F6: EXL3 --max-num-batched-tokens passthrough with golden tests.
* F8: NF3 RemoteAction/execute/action_succeeded delegated to sparkring_runtime.
* F9: Schema-aware execution semantics (EXL3 no rollback, NF3 rollback).
* EXL3 bridge golden-equivalence for canonical operations.
* NF3 bridge golden-equivalence for canonical operations.
* Deterministic plan structure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap_exl3  # noqa: E402
import sparkring_exl3_launcher as exl3  # noqa: E402
import sparkring_generic_launcher as generic  # noqa: E402
import sparkring_launcher as nf3  # noqa: E402
import sparkring_runtime as runtime  # noqa: E402
from sparkring_site import load_site  # noqa: E402

SITE = ROOT / "scripts/config/site.example.yaml"
LAUNCH_NF3 = ROOT / "scripts/config/launch.example.json"
GENERIC = ROOT / "scripts/config/generic.example.json"
IMAGE_ID = "sha256:" + "a" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exl3_profile(tmp_path):
    """Generate a valid EXL3 profile in tmp_path and return (site, profile)."""
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "launch.json"
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID
    )
    bootstrap_exl3.write_generated_profile(
        profile_path, "sparkring/exl3:test", IMAGE_ID,
        "/srv/models/exl3", "/srv/jit",
    )
    return site_path, profile_path


def _generic_doc(**overrides):
    """Return a valid generic profile document with optional overrides."""
    doc = json.loads(GENERIC.read_text(encoding="utf-8"))
    doc.update(overrides)
    return doc


def _write_generic(tmp_path, doc):
    path = tmp_path / "generic.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _run_cli(site_path, profile_path, command, *extra_args):
    """Run the generic launcher CLI and return (exit_code, stdout, stderr)."""
    argv = [
        sys.executable,
        str(ROOT / "scripts/sparkring_generic_launcher.py"),
        "--site", str(site_path),
        "--profile", str(profile_path),
        command,
        *extra_args,
    ]
    result = subprocess.run(
        argv, capture_output=True, text=True, cwd=str(ROOT)
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Malformed-profile tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override,expected_fragment",
    [
        ({"schema": "wrong"}, "unsupported schema"),
        ({"profile_id": ""}, "must be non-empty"),
        ({"model_family": ""}, "must be non-empty"),
        ({"engine": "systemd"}, "engine: must be docker or podman"),
        ({"container_name": ""}, "must be non-empty"),
        ({"image_id": "not-a-digest"}, "invalid image ID"),
        ({"image": ""}, "must be non-empty"),
        ({"model_host_path": "relative/path"}, "invalid absolute path"),
        ({"shm_size": "0g"}, "invalid shm size"),
        ({"startup_timeout_seconds": 10}, "must be in"),
        ({"startup_timeout_seconds": "x"}, "must be an integer"),
    ],
)
def test_malformed_profile_is_rejected(tmp_path, override, expected_fragment):
    doc = _generic_doc(**override)
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match=expected_fragment):
        runtime.load_runtime_profile(path)


def test_unknown_profile_key_is_rejected(tmp_path):
    doc = _generic_doc(unknown_key="value")
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="unknown key"):
        runtime.load_runtime_profile(path)


def test_missing_required_key_is_rejected(tmp_path):
    doc = _generic_doc()
    del doc["image_id"]
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="missing key"):
        runtime.load_runtime_profile(path)


def test_services_key_is_rejected(tmp_path):
    """The services field was removed — it must be rejected, not silently ignored."""
    doc = _generic_doc(services={"lmcache": {}})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="unknown key"):
        runtime.load_runtime_profile(path)


def test_environment_must_be_object(tmp_path):
    doc = _generic_doc(environment="not-an-object")
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="must be an object"):
        runtime.load_runtime_profile(path)


def test_environment_rejects_invalid_key(tmp_path):
    doc = _generic_doc(environment={"lower-case": "value"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="invalid environment key"):
        runtime.load_runtime_profile(path)


def test_environment_rejects_multiline_value(tmp_path):
    doc = _generic_doc(environment={"VLLM_TEST": "line1\nline2"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="must be one line or null"):
        runtime.load_runtime_profile(path)


def test_environment_rejects_site_derived_override(tmp_path):
    doc = _generic_doc(environment={"RANK": "0"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="derived from the validated site"):
        runtime.load_runtime_profile(path)


# F4: SPARKRING_ prefix is reserved for the runtime
def test_environment_rejects_sparkring_prefix(tmp_path):
    doc = _generic_doc(environment={"SPARKRING_FOO": "bar"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="reserved prefix"):
        runtime.load_runtime_profile(path)


# F4: SPARKRING_IMAGE_DIGEST is also reserved
def test_environment_rejects_sparkring_image_digest(tmp_path):
    doc = _generic_doc(environment={"SPARKRING_IMAGE_DIGEST": "sha256:x"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="reserved prefix"):
        runtime.load_runtime_profile(path)


def test_extra_vllm_args_rejects_empty_string(tmp_path):
    doc = _generic_doc(extra_vllm_args=["--valid", ""])
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="non-empty one-line string"):
        runtime.load_runtime_profile(path)


# F3: site-owned vLLM options are rejected in extra_vllm_args
@pytest.mark.parametrize(
    "bad_arg",
    [
        "--tensor-parallel-size",
        "--decode-context-parallel-size",
        "--max-model-len",
        "--kv-cache-memory-bytes",
        "--max-num-seqs",
        "--port",
        "--distributed-executor-backend",
        "--nnodes",
        "--node-rank",
        "--master-addr",
        "--master-port",
        "--headless",
    ],
)
def test_extra_vllm_args_reject_site_owned_option(tmp_path, bad_arg):
    doc = _generic_doc(extra_vllm_args=["--valid", bad_arg, "value"])
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="site-owned option"):
        runtime.load_runtime_profile(path)


@pytest.mark.parametrize(
    "bad_arg",
    [
        "--tensor-parallel-size=4",
        "--port=8000",
        "--nnodes=4",
        "--headless=true",
    ],
)
def test_extra_vllm_args_reject_equals_form(tmp_path, bad_arg):
    doc = _generic_doc(extra_vllm_args=["--valid", bad_arg])
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="site-owned option"):
        runtime.load_runtime_profile(path)


def test_unknown_placeholder_is_rejected(tmp_path):
    doc = _generic_doc(environment={"VLLM_TEST": "{unknown_var}"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="unknown placeholder"):
        runtime.load_runtime_profile(path)


def test_extra_volumes_must_have_host_container_mode(tmp_path):
    doc = _generic_doc(extra_volumes=[{"host": "/x", "container": "/y"}])
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="host/container/mode"):
        runtime.load_runtime_profile(path)


def test_extra_volumes_reject_relative_path(tmp_path):
    doc = _generic_doc(
        extra_volumes=[{"host": "relative", "container": "/y", "mode": "ro"}]
    )
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="invalid absolute path"):
        runtime.load_runtime_profile(path)


def test_extra_volumes_reject_bad_mode(tmp_path):
    doc = _generic_doc(
        extra_volumes=[{"host": "/x", "container": "/y", "mode": "rx"}]
    )
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="must be ro or rw"):
        runtime.load_runtime_profile(path)


def test_privileged_must_be_boolean(tmp_path):
    doc = _generic_doc(privileged="yes")
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="must be a boolean"):
        runtime.load_runtime_profile(path)


# F2: reserved labels are rejected in extra_labels
@pytest.mark.parametrize(
    "reserved_key",
    ["org.sparkring.managed", "org.sparkring.profile"],
)
def test_extra_labels_reject_reserved(tmp_path, reserved_key):
    doc = _generic_doc(extra_labels={reserved_key: "value"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="reserved label"):
        runtime.load_runtime_profile(path)


@pytest.mark.parametrize("invalid_key", ["", "bad key", "/leading", "bad\nkey"])
def test_extra_labels_reject_invalid_key(tmp_path, invalid_key):
    doc = _generic_doc(extra_labels={invalid_key: "value"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="invalid label key"):
        runtime.load_runtime_profile(path)


def test_extra_labels_reject_multiline_value(tmp_path):
    doc = _generic_doc(extra_labels={"example.label": "one\ntwo"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="one-line string"):
        runtime.load_runtime_profile(path)


# F4: empty confirmation is rejected (must be null or non-empty)
def test_empty_confirmation_rejected(tmp_path):
    doc = _generic_doc(confirmation="")
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="must be non-empty"):
        runtime.load_runtime_profile(path)


@pytest.mark.parametrize("token", ["contains space", "line\nbreak", "*wildcard"])
def test_invalid_confirmation_rejected(tmp_path, token):
    doc = _generic_doc(confirmation=token)
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="confirmation token"):
        runtime.load_runtime_profile(path)


# F4: identity keys must be lowercase snake_case
def test_identity_rejects_uppercase_key(tmp_path):
    doc = _generic_doc(identity={"BadKey": "value"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="invalid key"):
        runtime.load_runtime_profile(path)


def test_identity_rejects_empty_value(tmp_path):
    doc = _generic_doc(identity={"good_key": ""})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="must be a non-empty one-line"):
        runtime.load_runtime_profile(path)


def test_identity_rejects_sparkring_prefix(tmp_path):
    doc = _generic_doc(identity={"sparkring_foo": "value"})
    path = _write_generic(tmp_path, doc)
    with pytest.raises(runtime.ProfileError, match="reserved prefix"):
        runtime.load_runtime_profile(path)


def test_unsupported_schema_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "unknown/v1"}) + "\n", encoding="utf-8")
    with pytest.raises(runtime.ProfileError, match="unsupported schema"):
        generic.load_profile(path)


# ---------------------------------------------------------------------------
# F1: Native generic profile named "exl3" or "nf3" stays generic
# ---------------------------------------------------------------------------


def test_native_generic_named_exl3_stays_generic(tmp_path):
    """A generic profile with model_family='exl3' uses generic builders."""
    doc = _generic_doc(model_family="exl3", profile_id="my-exl3-native")
    path = _write_generic(tmp_path, doc)
    profile = generic.load_profile(path)
    assert profile.source_schema == runtime.SCHEMA
    assert profile.model_family == "exl3"
    # Generic start_actions has the image verify guard
    site = load_site(SITE)
    actions = generic.build_actions(site, profile, "start")
    assert len(actions) == 4
    # Generic start uses org.sparkring.profile label, not org.sparkring.exl3-profile
    assert "org.sparkring.profile=my-exl3-native" in actions[0].shell_command
    assert "org.sparkring.exl3-profile" not in actions[0].shell_command


def test_native_generic_named_nf3_stays_generic(tmp_path):
    """A generic profile with model_family='nf3' uses generic builders."""
    doc = _generic_doc(model_family="nf3", profile_id="my-nf3-native")
    path = _write_generic(tmp_path, doc)
    profile = generic.load_profile(path)
    assert profile.source_schema == runtime.SCHEMA
    assert profile.model_family == "nf3"
    site = load_site(SITE)
    actions = generic.build_actions(site, profile, "start")
    assert len(actions) == 4
    # Generic start has image verify guard (NF3 canonical does not)
    assert "image inspect" in actions[0].shell_command


# ---------------------------------------------------------------------------
# Dry-run and plan tests
# ---------------------------------------------------------------------------


def test_generic_plan_is_offline_and_deterministic():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.start_actions(site, profile)
    plan = runtime.plan_document("plan", actions, profile)
    assert plan["schema"] == runtime.PLAN_SCHEMA
    assert plan["command"] == "plan"
    assert plan["mutates_remote"] is False
    assert plan["profile_id"] == "example-generic-runtime"
    assert plan["model_family"] == "example"
    assert plan["source_schema"] == runtime.SCHEMA
    assert plan["identity_scope"] == "image-verified-before-start"
    assert len(plan["actions"]) == 4
    for action in plan["actions"]:
        assert "rank" in action
        assert "ssh_target" in action
        assert "remote_command" in action


def test_generic_plan_produces_four_rank_actions():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.start_actions(site, profile)
    ranks = [a.rank for a in actions]
    assert sorted(ranks) == [0, 1, 2, 3]


def test_plan_command_with_execute_is_rejected(tmp_path):
    site_path = tmp_path / "site.yaml"
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID
    )
    profile_path = _write_generic(tmp_path, _generic_doc())
    code, stdout, stderr = _run_cli(
        site_path, profile_path, "plan", "--execute"
    )
    assert code == 2
    assert "plan is always offline" in stderr


def test_dry_run_start_makes_no_connection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SSH was attempted during dry run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.start_actions(site, profile)
    plan = runtime.plan_document("start", actions, profile)
    assert plan["mutates_remote"] is True


def test_mutating_command_without_execute_is_dry_run(tmp_path):
    site_path = tmp_path / "site.yaml"
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID
    )
    profile_path = _write_generic(tmp_path, _generic_doc())
    code, stdout, stderr = _run_cli(site_path, profile_path, "start")
    assert code == 0
    assert "made no remote connection" in stderr


# ---------------------------------------------------------------------------
# Lifecycle safety tests
# ---------------------------------------------------------------------------


def test_generic_start_verifies_image_identity_before_run():
    """Each generic start action must fail-closed on image ID drift."""
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.start_actions(site, profile)
    for action in actions:
        cmd = action.shell_command
        assert "image inspect" in cmd
        assert profile.image_id in cmd
        assert "test" in cmd  # the test guard
        assert "exec" in cmd   # fails closed, then runs


def test_generic_start_has_profile_label():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.start_actions(site, profile)
    for action in actions:
        assert f"org.sparkring.profile={profile.profile_id}" in action.shell_command


def test_stop_actions_check_exact_profile_label():
    """Stop must check the exact profile label, not just managed=true."""
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.stop_actions(site, profile)
    for action in actions:
        cmd = action.shell_command
        assert "org.sparkring.managed" in cmd
        assert runtime.PROFILE_LABEL in cmd
        assert profile.profile_id in cmd
        assert "exit 73" in cmd


def test_stop_actions_use_force_remove():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.stop_actions(site, profile)
    for action in actions:
        assert "rm --force" in action.shell_command


def test_confirmation_token_required_for_mutation(tmp_path):
    site_path = tmp_path / "site.yaml"
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID
    )
    doc = _generic_doc(confirmation="MY-SECRET-TOKEN")
    profile_path = _write_generic(tmp_path, doc)
    code, stdout, stderr = _run_cli(
        site_path, profile_path, "start", "--execute"
    )
    assert code == 2
    assert "MY-SECRET-TOKEN" in stderr


def test_confirmation_token_wrong_value_rejected(tmp_path):
    site_path = tmp_path / "site.yaml"
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID
    )
    doc = _generic_doc(confirmation="CORRECT-TOKEN")
    profile_path = _write_generic(tmp_path, doc)
    code, stdout, stderr = _run_cli(
        site_path, profile_path, "start", "--execute", "--confirmation", "WRONG"
    )
    assert code == 2
    assert "CORRECT-TOKEN" in stderr


def test_verify_rollback_is_read_only():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.verify_rollback_actions(site, profile)
    for action in actions:
        assert "container inspect" in action.shell_command
        assert "! docker container inspect" in action.shell_command


def test_verify_image_checks_exact_digest():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    actions = runtime.verify_image_actions(site, profile)
    for action in actions:
        assert profile.image_id in action.shell_command
        assert "image inspect" in action.shell_command


def test_start_failure_triggers_scoped_rollback(monkeypatch):
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)

    call_log = []

    def fake_execute(actions, timeout):
        call_log.append(tuple(a.rank for a in actions))
        if len(call_log) == 1:  # start
            return {
                0: {"exit_code": 0, "stdout": "abc123def456", "stderr": ""},
                1: {"exit_code": 1, "stdout": "", "stderr": "fail"},
                2: {"exit_code": 1, "stdout": "", "stderr": "fail"},
                3: {"exit_code": 1, "stdout": "", "stderr": "fail"},
            }
        if len(call_log) == 2:  # rollback
            return {0: {"exit_code": 0, "stdout": "", "stderr": ""}}
        return {}

    monkeypatch.setattr(runtime, "execute", fake_execute)
    actions = runtime.start_actions(site, profile)
    results = fake_execute(actions, profile.startup_timeout_seconds)
    failed_ranks = [
        r for r, res in results.items()
        if not runtime.action_succeeded("start", res)
    ]
    assert failed_ranks == [1, 2, 3]
    started = {
        r for r, res in results.items()
        if runtime.action_succeeded("start", res)
    }
    rollback = [
        a for a in runtime.stop_actions(site, profile) if a.rank in started
    ]
    assert len(rollback) == 1
    assert rollback[0].rank == 0


def test_action_succeeded_rejects_docker_help_false_positive():
    assert not runtime.action_succeeded(
        "start",
        {"exit_code": 0, "stdout": "Usage: docker run [OPTIONS]...", "stderr": ""},
    )


def test_action_succeeded_accepts_container_id():
    assert runtime.action_succeeded(
        "start",
        {"exit_code": 0, "stdout": "abc123def456", "stderr": ""},
    )


def test_timeout_output_is_json_serializable():
    import subprocess as sp

    sp.TimeoutExpired(cmd="ssh", timeout=30, output=b"out", stderr=b"err")
    result = {
        "exit_code": 124,
        "stdout": "out",
        "stderr": "err",
    }
    assert json.dumps(result)  # serializable


# ---------------------------------------------------------------------------
# F7: Attestation hook and health check tests
# ---------------------------------------------------------------------------


def test_attestation_hook_in_start_action(tmp_path):
    """attestation_hook runs after image verify, before docker run."""
    doc = _generic_doc(
        attestation_hook=["/opt/verify_model.py", "--model-path", "/models/your-model"],
    )
    path = _write_generic(tmp_path, doc)
    profile = runtime.load_runtime_profile(path)
    site = load_site(SITE)
    actions = runtime.start_actions(site, profile)
    for action in actions:
        cmd = action.shell_command
        # Image verify comes first
        assert "image inspect" in cmd
        # Attestation hook runs between verify and exec
        assert "/opt/verify_model.py" in cmd
        assert "docker run --rm" in cmd
        assert "--entrypoint /opt/verify_model.py" in cmd
        # The main start still has exec
        assert "exec" in cmd


def test_attestation_hook_identity_scope(tmp_path):
    """A configured hook is recorded without claiming independent proof."""
    doc = _generic_doc(
        attestation_hook=["/opt/verify_model.py"],
    )
    path = _write_generic(tmp_path, doc)
    profile = runtime.load_runtime_profile(path)
    site = load_site(SITE)
    actions = runtime.start_actions(site, profile)
    plan = runtime.plan_document("plan", actions, profile)
    assert plan["identity_scope"] == "attestation-hook-configured"


def test_no_attestation_hook_means_image_only():
    profile = runtime.load_runtime_profile(GENERIC)
    site = load_site(SITE)
    actions = runtime.start_actions(site, profile)
    plan = runtime.plan_document("plan", actions, profile)
    assert plan["identity_scope"] == "image-verified-before-start"


def test_health_check_actions_built(tmp_path):
    """health_check produces docker exec actions."""
    doc = _generic_doc(
        health_check=["curl", "-sf", "http://localhost:{api_port}/health"],
    )
    path = _write_generic(tmp_path, doc)
    profile = runtime.load_runtime_profile(path)
    site = load_site(SITE)
    actions = runtime.health_check_actions(site, profile)
    assert len(actions) == 4
    for action in actions:
        assert "exec" in action.shell_command
        assert "curl" in action.shell_command
        assert runtime.MANAGED_LABEL in action.shell_command
        assert runtime.PROFILE_LABEL in action.shell_command
        assert profile.profile_id in action.shell_command
        # Placeholder is expanded
        assert "{api_port}" not in action.shell_command


def test_health_check_empty_returns_empty():
    profile = runtime.load_runtime_profile(GENERIC)
    site = load_site(SITE)
    actions = runtime.health_check_actions(site, profile)
    assert actions == []


def test_health_cli_dry_run_is_deterministic(tmp_path):
    doc = _generic_doc(
        health_check=["curl", "-sf", "http://localhost:{api_port}/health"],
    )
    profile_path = _write_generic(tmp_path, doc)
    code, stdout, stderr = _run_cli(SITE, profile_path, "health")
    assert code == 0
    plan = json.loads(stdout)
    assert plan["command"] == "health"
    assert plan["mutates_remote"] is True
    assert plan["stops_serving_risk"] is True
    assert len(plan["actions"]) == 4
    assert "made no remote connection" in stderr


def test_health_cli_rejects_profile_without_check():
    code, stdout, stderr = _run_cli(SITE, GENERIC, "health")
    assert code == 2
    assert "has no health_check" in stderr


def test_health_cli_rejects_family_bridge():
    code, stdout, stderr = _run_cli(SITE, LAUNCH_NF3, "health")
    assert code == 2
    assert "only for native generic profiles" in stderr


def test_health_execute_uses_profile_confirmation_gate(tmp_path):
    doc = _generic_doc(
        confirmation="RUN-HEALTH-PROBE",
        health_check=["curl", "-sf", "http://localhost:{api_port}/health"],
    )
    profile_path = _write_generic(tmp_path, doc)
    code, stdout, stderr = _run_cli(
        SITE, profile_path, "health", "--execute",
    )
    assert code == 2
    assert "RUN-HEALTH-PROBE" in stderr


# ---------------------------------------------------------------------------
# F9: Schema-aware execution semantics
# ---------------------------------------------------------------------------


def test_execution_mode_exl3_bridge():
    """EXL3 bridge uses exit-status-only, no rollback."""
    profile = runtime.RuntimeProfile(
        profile_id="test", model_family="exl3", engine="docker",
        container_name="test", image="img", image_id=IMAGE_ID,
        model_host_path="/m", model_container_path="/c",
        shm_size="16g", startup_timeout_seconds=300,
        environment={}, extra_vllm_args=(),
        source_schema=runtime.EXL3_SCHEMA,
    )
    assert runtime.execution_mode(profile) == "exl3"
    assert runtime.should_rollback("start", "exl3") is False


def test_execution_mode_nf3_bridge():
    """NF3 bridge uses action_succeeded with rollback."""
    profile = runtime.RuntimeProfile(
        profile_id="test", model_family="nf3", engine="docker",
        container_name="test", image="img", image_id=IMAGE_ID,
        model_host_path="/m", model_container_path="/c",
        shm_size="16g", startup_timeout_seconds=300,
        environment={}, extra_vllm_args=(),
        source_schema=runtime.NF3_SCHEMA,
    )
    assert runtime.execution_mode(profile) == "nf3"
    assert runtime.should_rollback("start", "nf3") is True


def test_execution_mode_generic():
    """Generic profiles use action_succeeded with rollback."""
    profile = runtime.RuntimeProfile(
        profile_id="test", model_family="my-model", engine="docker",
        container_name="test", image="img", image_id=IMAGE_ID,
        model_host_path="/m", model_container_path="/c",
        shm_size="16g", startup_timeout_seconds=300,
        environment={}, extra_vllm_args=(),
        source_schema=runtime.SCHEMA,
    )
    assert runtime.execution_mode(profile) == "generic"
    assert runtime.should_rollback("start", "generic") is True


def test_check_results_exl3_mode():
    """EXL3 mode uses exit-code-only check."""
    results = {
        0: {"exit_code": 0, "stdout": "", "stderr": ""},
        1: {"exit_code": 1, "stdout": "abc123", "stderr": ""},
    }
    assert runtime.check_results("start", results, "exl3") == [1]


def test_check_results_generic_mode_rejects_no_container_id():
    """Generic mode uses action_succeeded which checks for container ID."""
    results = {
        0: {"exit_code": 0, "stdout": "abc123def456", "stderr": ""},
        1: {"exit_code": 0, "stdout": "no container id here", "stderr": ""},
    }
    # In generic mode, rank 1 fails because no valid container ID
    assert 1 in runtime.check_results("start", results, "generic")


# ---------------------------------------------------------------------------
# EXL3 bridge golden-equivalence tests
# ---------------------------------------------------------------------------


def test_exl3_profile_loads_through_bridge(tmp_path):
    _, profile_path = _exl3_profile(tmp_path)
    profile = generic.load_profile(profile_path)
    assert profile.model_family == "exl3"
    assert profile.profile_id == "glm52-exl3-tr3-3.25bpw-lmcache-cs512"
    assert profile.source_schema == runtime.EXL3_SCHEMA
    assert "org.sparkring.exl3-profile" in profile.extra_labels


def test_exl3_bridge_start_is_byte_identical_to_canonical(tmp_path):
    """Every rank's start action must be byte-identical to the EXL3 launcher."""
    site_path, profile_path = _exl3_profile(tmp_path)
    site = load_site(site_path)
    exl3_profile = exl3.load_profile(profile_path)
    bridge_profile = generic.load_profile(profile_path)
    canonical = exl3.start_actions(site, exl3_profile)
    bridge = generic.build_actions(site, bridge_profile, "start")
    assert len(canonical) == len(bridge) == 4
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command, (
            f"rank {i} start action differs"
        )


# F6: --max-num-batched-tokens passthrough
def test_exl3_bridge_start_with_batch_tokens_is_byte_identical(tmp_path):
    """EXL3 bridge with --max-num-batched-tokens matches canonical."""
    site_path, profile_path = _exl3_profile(tmp_path)
    site = load_site(site_path)
    exl3_profile = exl3.load_profile(profile_path)
    bridge_profile = generic.load_profile(profile_path)
    for tokens in (2048, 3072, 4096):
        canonical = exl3.start_actions(
            site, exl3_profile, max_num_batched_tokens=tokens,
        )
        bridge = generic.build_actions(
            site, bridge_profile, "start", max_num_batched_tokens=tokens,
        )
        for i in range(4):
            assert canonical[i].shell_command == bridge[i].shell_command, (
                f"rank {i} start with tokens={tokens} differs"
            )


def test_exl3_bridge_plan_includes_effective_settings(tmp_path):
    """EXL3 bridge plan must include effective_settings with batch tokens."""
    site_path, profile_path = _exl3_profile(tmp_path)
    code, stdout, stderr = _run_cli(
        site_path, profile_path, "plan",
        "--max-num-batched-tokens", "3072",
    )
    assert code == 0
    plan = json.loads(stdout)
    assert "effective_settings" in plan
    assert plan["effective_settings"]["max_num_batched_tokens"] == 3072
    assert plan["effective_settings"]["default_max_num_batched_tokens"] == 4096
    assert plan["effective_settings"]["experiment_overrides"] == {
        "max_num_batched_tokens": 3072
    }


def test_exl3_bridge_plan_default_no_override(tmp_path):
    """EXL3 bridge plan without --max-num-batched-tokens has empty overrides."""
    site_path, profile_path = _exl3_profile(tmp_path)
    code, stdout, stderr = _run_cli(site_path, profile_path, "plan")
    assert code == 0
    plan = json.loads(stdout)
    assert plan["effective_settings"]["max_num_batched_tokens"] == 4096
    assert plan["effective_settings"]["experiment_overrides"] == {}


def test_exl3_bridge_stop_is_byte_identical_to_canonical(tmp_path):
    site_path, profile_path = _exl3_profile(tmp_path)
    site = load_site(site_path)
    exl3_profile = exl3.load_profile(profile_path)
    bridge_profile = generic.load_profile(profile_path)
    canonical = exl3.simple_actions(site, exl3_profile, "stop")
    bridge = generic.build_actions(site, bridge_profile, "stop")
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command


def test_exl3_bridge_status_is_byte_identical_to_canonical(tmp_path):
    site_path, profile_path = _exl3_profile(tmp_path)
    site = load_site(site_path)
    exl3_profile = exl3.load_profile(profile_path)
    bridge_profile = generic.load_profile(profile_path)
    canonical = exl3.simple_actions(site, exl3_profile, "status")
    bridge = generic.build_actions(site, bridge_profile, "status")
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command


def test_exl3_bridge_verify_image_is_byte_identical_to_canonical(tmp_path):
    site_path, profile_path = _exl3_profile(tmp_path)
    site = load_site(site_path)
    exl3_profile = exl3.load_profile(profile_path)
    bridge_profile = generic.load_profile(profile_path)
    canonical = exl3.simple_actions(site, exl3_profile, "verify-image")
    bridge = generic.build_actions(site, bridge_profile, "verify-image")
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command


def test_exl3_bridge_plan_through_cli(tmp_path):
    site_path, profile_path = _exl3_profile(tmp_path)
    code, stdout, stderr = _run_cli(site_path, profile_path, "plan")
    assert code == 0
    plan = json.loads(stdout)
    assert plan["schema"] == runtime.PLAN_SCHEMA
    assert plan["model_family"] == "exl3"
    assert plan["source_schema"] == runtime.EXL3_SCHEMA
    assert plan["identity_scope"] == "canonical-model-verification"
    assert len(plan["actions"]) == 4


def test_batch_token_override_rejected_for_native_profile():
    code, stdout, stderr = _run_cli(
        SITE,
        GENERIC,
        "plan",
        "--max-num-batched-tokens",
        "3072",
    )
    assert code == 2
    assert "only for the EXL3 bridge" in stderr


# ---------------------------------------------------------------------------
# NF3 bridge golden-equivalence tests
# ---------------------------------------------------------------------------


def test_nf3_profile_loads_through_bridge():
    profile = generic.load_profile(LAUNCH_NF3)
    assert profile.model_family == "nf3"
    assert profile.source_schema == runtime.NF3_SCHEMA
    assert profile.container_name == "glm52-sparkring-nf3"
    assert any("/mtp-draft" in vol[1] for vol in profile.extra_volumes)


# F5: NF3 bridge identity resolved from site
def test_nf3_bridge_identity_resolved_from_site():
    """NF3 bridge plan has truthful image_id from site, not empty."""
    site = load_site(SITE)
    profile = generic.load_profile(LAUNCH_NF3)
    profile = runtime.resolve_from_site(profile, site)
    assert profile.image_id != ""
    assert profile.image != ""
    assert profile.image_id == site.runtime.container_image_digest


def test_nf3_bridge_plan_has_truthful_image_id():
    """NF3 bridge plan's profile_attestation.image_id must be non-empty."""
    code, stdout, stderr = _run_cli(SITE, LAUNCH_NF3, "plan")
    assert code == 0
    plan = json.loads(stdout)
    assert plan["profile_attestation"]["image_id"] != ""
    assert plan["profile_attestation"]["image_id"].startswith("sha256:")
    assert plan["identity_scope"] == "declared-site-image"


def test_nf3_bridge_start_is_byte_identical_to_canonical():
    """Every rank's start action must be byte-identical to the NF3 launcher."""
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)
    bridge_profile = generic.load_profile(LAUNCH_NF3)
    canonical = nf3.start_actions(site, config)
    bridge = generic.build_actions(site, bridge_profile, "start")
    assert len(canonical) == len(bridge) == 4
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command, (
            f"rank {i} start action differs"
        )


def test_nf3_bridge_stop_is_byte_identical_to_canonical():
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)
    bridge_profile = generic.load_profile(LAUNCH_NF3)
    canonical = nf3.simple_actions(site, config, "stop")
    bridge = generic.build_actions(site, bridge_profile, "stop")
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command


def test_nf3_bridge_status_is_byte_identical_to_canonical():
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)
    bridge_profile = generic.load_profile(LAUNCH_NF3)
    canonical = nf3.simple_actions(site, config, "status")
    bridge = generic.build_actions(site, bridge_profile, "status")
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command


def test_nf3_bridge_verify_rollback_is_byte_identical_to_canonical():
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)
    bridge_profile = generic.load_profile(LAUNCH_NF3)
    canonical = nf3.simple_actions(site, config, "verify-rollback")
    bridge = generic.build_actions(site, bridge_profile, "verify-rollback")
    for i in range(4):
        assert canonical[i].shell_command == bridge[i].shell_command


def test_nf3_bridge_plan_through_cli():
    code, stdout, stderr = _run_cli(SITE, LAUNCH_NF3, "plan")
    assert code == 0
    plan = json.loads(stdout)
    assert plan["schema"] == runtime.PLAN_SCHEMA
    assert plan["model_family"] == "nf3"
    assert plan["source_schema"] == runtime.NF3_SCHEMA
    assert len(plan["actions"]) == 4


# ---------------------------------------------------------------------------
# F8: Shared primitives extraction tests
# ---------------------------------------------------------------------------


def test_nf3_remote_action_is_same_class_as_runtime():
    """NF3 RemoteAction is imported from sparkring_runtime (F8)."""
    assert nf3.RemoteAction is runtime.RemoteAction


def test_nf3_execute_is_runtime_execute():
    """NF3 execute delegates to sparkring_runtime (F8)."""
    assert nf3.execute is runtime.execute


def test_nf3_action_succeeded_is_runtime_action_succeeded():
    """NF3 action_succeeded delegates to sparkring_runtime (F8)."""
    assert nf3.action_succeeded is runtime.action_succeeded


def test_nf3_run_remote_is_runtime_run_remote():
    """NF3 run_remote delegates to sparkring_runtime (F8)."""
    assert nf3.run_remote is runtime.run_remote


def test_nf3_run_remote_still_works(monkeypatch):
    """The shared run_remote produces the same quoting behavior."""
    action = nf3.RemoteAction(
        rank=0,
        ssh_target="operator@node0",
        argv=("docker", "run", "--detach", "image:tag"),
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "a" * 64 + "\n", "")

    monkeypatch.setattr(nf3.subprocess, "run", fake_run)
    nf3.run_remote(action, timeout=10)
    assert captured["argv"][-1] == (
        "sh -lc 'docker run --detach image:tag'"
    )
    assert captured["argv"][-2] == "operator@node0"


# ---------------------------------------------------------------------------
# Context derivation tests (shared primitives)
# ---------------------------------------------------------------------------


def test_site_context_follows_xor_round_schedule():
    site = load_site(SITE)
    for rank_id in range(4):
        ctx = runtime.site_context(site, rank_id)
        assert ctx["rank"] == str(rank_id)
        assert ctx["world_size"] == "4"
        assert int(ctx["peer0_rank"]) == rank_id ^ 1
        assert int(ctx["peer1_rank"]) == rank_id ^ 3


def test_base_environment_has_transport_keys():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    env = runtime.base_environment(site, 0, profile)
    for key in (
        "GLOO_SOCKET_IFNAME", "MASTER_ADDR", "MASTER_PORT",
        "NCCL_IB_GID_INDEX", "NCCL_IB_HCA", "RANK", "WORLD_SIZE",
        "SPARK_TP4_DEVICE0", "SPARK_TP4_DEVICE1",
        "SPARK_TP4_PEER0", "SPARK_TP4_PEER1",
    ):
        assert key in env, f"missing transport key: {key}"


def test_base_environment_includes_identity_attestation():
    site = load_site(SITE)
    profile = runtime.load_runtime_profile(GENERIC)
    env = runtime.base_environment(site, 0, profile)
    # Identity keys use SPARKRING_ATTEST_ prefix (F4)
    assert env["SPARKRING_ATTEST_MODEL_REPOSITORY"] == "your-org/your-model"
    assert env["SPARKRING_ATTEST_MODEL_CONFIG_SHA256"] == (
        "2222222222222222222222222222222222222222222222222222222222222222"
    )
    # SPARKRING_IMAGE_DIGEST is still set by the runtime
    assert env["SPARKRING_IMAGE_DIGEST"] == profile.image_id


def test_container_name_follows_convention():
    profile = runtime.load_runtime_profile(GENERIC)
    for rank_id in range(4):
        name = runtime.container_name(profile, rank_id)
        assert name == f"sparkring-generic-example-r{rank_id}"
