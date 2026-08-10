from __future__ import annotations

import copy
import base64
import json
from pathlib import Path
import shlex
import subprocess

import pytest

import bootstrap_exl3
import exl3_attribution_launcher as attribution
import sparkring_exl3_launcher as exl3
from exl3_attribution_cache_contract import (
    build_live_arm_receipt,
    validate_live_arm_receipt,
)
from sparkring_site import load_site


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts/config/site.example.yaml"
IMAGE_ID = "sha256:" + "a" * 64


def generated(tmp_path):
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "launch.json"
    bootstrap_exl3.write_generated_site(SITE, site_path, "sparkring/exl3:test", IMAGE_ID)
    bootstrap_exl3.write_generated_profile(
        profile_path, "sparkring/exl3:test", IMAGE_ID, "/srv/models/exl3", "/srv/jit"
    )
    return site_path, profile_path


def with_engine(profile, engine):
    document = copy.deepcopy(profile.document)
    document["engine"] = engine
    return exl3.Profile(document)


def execution_argv(site_path, profile_path, *, command="activate", confirmation=None):
    argv = [
        "--site",
        str(site_path),
        "--profile",
        str(profile_path),
        "--arm",
        "a-mtp0-apc0-lmcache0",
        "--execute",
    ]
    if command in ("activate", "restart-arm", "transition", "status"):
        argv += [
            "--live-arm-receipt-output",
            str(profile_path.parent / f"{command}-live-arm-receipt.json"),
        ]
    if confirmation is not None:
        argv += ["--confirmation", confirmation]
    return argv + [command]


def phase_name(actions):
    commands = "\n".join(action.shell_command for action in actions)
    for live_phase in (
        "diagnostic_live_arm_attestation",
        "target_live_arm_attestation",
    ):
        if f": {live_phase};" in commands:
            return live_phase
    for reset_phase in (
        "rollback_remove_canonical_engines",
        "isolate_target_lmcache_remove_servers",
        "isolate_target_lmcache_start_servers",
        "isolate_target_lmcache_server_health",
        "rollback_reset_lmcache_remove_servers",
        "rollback_reset_lmcache_start_servers",
        "rollback_reset_lmcache_server_health",
    ):
        if f": {reset_phase};" in commands:
            return reset_phase
    if all(
        attribution.REMOTE_OUTER_VERIFIED_ENTRYPOINT in action.argv[-1]
        and "install -D -m 0555" in action.argv[-1]
        for action in actions
    ):
        return "prepare_page_cache_reclaim_entrypoint"
    if ": canonical_engine_exclusive_after_restore;" in commands:
        return "canonical_engine_exclusive_after_restore"
    if (
        "label=org.sparkring.managed=true" in commands
        and "engine_ids=" in commands
        and "observed_name=$(docker inspect" in commands
    ):
        return "canonical_engine_exclusive"
    if "docker ps -q --filter label=org.sparkring.exl3-attribution" in commands:
        return "no_other_diagnostic"
    if "/healthcheck" in commands and "/status" in commands:
        return "server_health"
    diagnostic = "-diag-a-mtp0-apc0-lmcache0" in commands
    if "docker rm --force" in commands:
        return "remove_diagnostic" if diagnostic else "remove_canonical_engines"
    if "docker run --detach" in commands:
        return "start_diagnostic" if diagnostic else "start_canonical"
    if ".RestartCount" in commands:
        return "diagnostic_ready" if diagnostic else "canonical_ready"
    raise AssertionError(f"unrecognized phase: {commands[:300]}")


def phase_result(*, failed=False, partial=False):
    result = {
        rank: {
            "exit_code": 0,
            "stdout": json.dumps(
                {
                    "rank": rank,
                    "status": "attested",
                    "container_id": f"{rank + 1:064x}",
                    "started_at": f"2026-08-10T03:2{rank}:00.123456789Z",
                }
            ),
            "stderr": "",
        }
        for rank in range(4)
    }
    if failed:
        if partial:
            result[2]["exit_code"] = 71
        else:
            for rank in result:
                result[rank]["exit_code"] = 71
    return result


class RecordingExecutor:
    def __init__(self, failures=()):
        self.failures = set(failures)
        self.calls = []
        self.actions = []

    def __call__(self, actions, timeout):
        del timeout
        self.actions.append(actions)
        self.calls.append(phase_name(actions))
        call_number = len(self.calls)
        return phase_result(
            failed=call_number in self.failures,
            partial=call_number in self.failures,
        )


class RaisingExecutor(RecordingExecutor):
    def __init__(self, raises):
        super().__init__()
        self.raises = set(raises)

    def __call__(self, actions, timeout):
        del timeout
        self.actions.append(actions)
        self.calls.append(phase_name(actions))
        if len(self.calls) in self.raises:
            raise RuntimeError(f"injected-{self.calls[-1]}")
        return phase_result()


def transition_phase_name(actions):
    commands = "\n".join(action.shell_command for action in actions)
    for live_phase in (
        "diagnostic_live_arm_attestation",
        "target_live_arm_attestation",
    ):
        if f": {live_phase};" in commands:
            return live_phase
    for reset_phase in (
        "rollback_remove_canonical_engines",
        "isolate_target_lmcache_remove_servers",
        "isolate_target_lmcache_start_servers",
        "isolate_target_lmcache_server_health",
        "rollback_reset_lmcache_remove_servers",
        "rollback_reset_lmcache_start_servers",
        "rollback_reset_lmcache_server_health",
    ):
        if f": {reset_phase};" in commands:
            return reset_phase
    source = "-diag-a-mtp0-apc0-lmcache0" in commands
    target = "-diag-b-mtp2-apc0-lmcache0" in commands
    if ": canonical_engine_exclusive_after_restore;" in commands:
        return "canonical_engine_exclusive_after_restore"
    if "/healthcheck" in commands and "/status" in commands:
        for phase in (
            "source_server_health",
            "post_source_removal_server_health",
            "target_server_health",
            "rollback_server_health_before_restore",
            "rollback_server_health_after_restore",
        ):
            if f": {phase};" in commands:
                return phase
        raise AssertionError("transition server-health action lacks phase tag")
    if "engine_ids=" in commands and "observed_name=$(docker inspect" in commands:
        assert source
        return "source_diagnostic_exclusive"
    if "docker rm --force" in commands:
        if source:
            return "remove_source_diagnostic"
        if target:
            return "remove_target_diagnostic"
    if "docker run --detach" in commands:
        if target:
            return "start_target_diagnostic"
        if not source:
            return "start_canonical"
    if ".RestartCount" in commands:
        if source:
            return "source_diagnostic_ready"
        if target:
            return "target_diagnostic_ready"
        return "canonical_ready"
    raise AssertionError(f"unrecognized transition phase: {commands[:300]}")


class TransitionRecordingExecutor(RecordingExecutor):
    def __call__(self, actions, timeout):
        del timeout
        self.actions.append(actions)
        self.calls.append(transition_phase_name(actions))
        call_number = len(self.calls)
        return phase_result(
            failed=call_number in self.failures,
            partial=call_number in self.failures,
        )


class TransitionRaisingExecutor(TransitionRecordingExecutor):
    def __init__(self, raises):
        super().__init__()
        self.raises = set(raises)

    def __call__(self, actions, timeout):
        del timeout
        self.actions.append(actions)
        self.calls.append(transition_phase_name(actions))
        if len(self.calls) in self.raises:
            raise RuntimeError(f"transition-injected-{self.calls[-1]}")
        return phase_result()


def transition_argv(site_path, profile_path, *, from_arm=None, confirmation=None):
    argv = [
        "--site",
        str(site_path),
        "--profile",
        str(profile_path),
        "--arm",
        "b-mtp2-apc0-lmcache0",
    ]
    if from_arm is not None:
        argv += ["--from-arm", from_arm]
    argv.append("--execute")
    argv += [
        "--live-arm-receipt-output",
        str(profile_path.parent / "transition-live-arm-receipt.json"),
    ]
    if confirmation is not None:
        argv += ["--confirmation", confirmation]
    return argv + ["transition"]


def custom_transition_argv(
    site_path, profile_path, *, from_arm, arm, execute=False
):
    argv = [
        "--site",
        str(site_path),
        "--profile",
        str(profile_path),
        "--from-arm",
        from_arm,
        "--arm",
        arm,
    ]
    if execute:
        argv += [
            "--execute",
            "--confirmation",
            attribution.CONFIRMATION,
            "--live-arm-receipt-output",
            str(profile_path.parent / "transition-live-arm-receipt.json"),
        ]
    return argv + ["transition"]


def restart_arm_argv(site_path, profile_path, *, execute=False, from_arm=None):
    argv = [
        "--site",
        str(site_path),
        "--profile",
        str(profile_path),
        "--arm",
        "d-mtp2-apc1-lmcache1",
    ]
    if from_arm is not None:
        argv += ["--from-arm", from_arm]
    if execute:
        argv += [
            "--execute",
            "--confirmation",
            attribution.CONFIRMATION,
            "--live-arm-receipt-output",
            str(profile_path.parent / "restart-live-arm-receipt.json"),
        ]
    return argv + ["restart-arm"]


def rendered_docker_tokens(site, canonical, arm_id):
    derived = attribution.derive_profile(canonical, arm_id)
    command = attribution.start_actions(site, derived, arm_id)[0].argv[-1]
    docker_suffix = command.split("exec docker run", 1)[1]
    return ["docker", "run", *shlex.split(docker_suffix)]


def normalized_arm_tokens(tokens):
    result = []
    index = 0
    value_options = {"--name", "--label", "--volume", "--env"}
    functional_options = {
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--speculative-config",
        "--kv-transfer-config",
    }
    while index < len(tokens):
        token = tokens[index]
        if token in functional_options:
            if token in {
                "--enable-prefix-caching",
                "--no-enable-prefix-caching",
            }:
                index += 1
            else:
                index += 2
            continue
        if token in value_options and index + 1 < len(tokens):
            value = tokens[index + 1]
            if token == "--name":
                index += 2
                continue
            if token == "--label" and (
                value.startswith("org.sparkring.exl3-profile=")
                or value.startswith(f"{attribution.LABEL_KEY}=")
            ):
                index += 2
                continue
            if token == "--volume" and value.endswith(":/cache/jit"):
                index += 2
                continue
            if token == "--env" and value.startswith(
                ("VLLM_SPARK_MTP_MODE_ID=", "VLLM_SPARK_MTP_TOKENS=")
            ):
                index += 2
                continue
        result.append(token)
        index += 1
    return result


def test_mtp0_apc_off_cache_off_is_exact_derived_delta(tmp_path):
    _, profile_path = generated(tmp_path)
    canonical = exl3.load_profile(profile_path)
    derived = attribution.derive_profile(canonical, "a-mtp0-apc0-lmcache0")
    assert "--enable-prefix-caching" not in derived.extra_vllm_args
    assert "--no-enable-prefix-caching" in derived.extra_vllm_args
    assert "--speculative-config" not in derived.extra_vllm_args
    assert derived.environment["VLLM_SPARK_MTP_TOKENS"] == "0"
    assert derived.environment["VLLM_SPARK_MTP_MODE_ID"] == "disabled"
    assert canonical.environment["VLLM_SPARK_MTP_TOKENS"] == "2"
    assert "--enable-prefix-caching" in canonical.extra_vllm_args


def test_each_adjacent_arm_changes_one_functional_variable():
    order = [
        "a-mtp0-apc0-lmcache0",
        "b-mtp2-apc0-lmcache0",
        "c-mtp2-apc1-lmcache0",
        "d-mtp2-apc1-lmcache1",
    ]
    for left, right in zip(order, order[1:]):
        a = attribution.functional_delta(left)
        b = attribution.functional_delta(right)
        assert sum(a[key] != b[key] for key in a) == 1
    assert sum(
        attribution.functional_delta("e-mtp0-apc0-lmcache1")[key]
        != attribution.functional_delta("f-mtp2-apc0-lmcache1")[key]
        for key in attribution.functional_delta("e-mtp0-apc0-lmcache1")
    ) == 1


def test_lmcache_layout_contract_distinguishes_mtp_staging_layout(tmp_path):
    _, profile_path = generated(tmp_path)
    canonical = exl3.load_profile(profile_path)
    mtp0 = attribution.derive_profile(
        canonical, "e-mtp0-apc0-lmcache1"
    )
    mtp2 = attribution.derive_profile(
        canonical, "f-mtp2-apc0-lmcache1"
    )
    mtp0_layout = attribution.lmcache_layout_contract(
        mtp0, attribution.ARMS["e-mtp0-apc0-lmcache1"]
    )
    mtp2_layout = attribution.lmcache_layout_contract(
        mtp2, attribution.ARMS["f-mtp2-apc0-lmcache1"]
    )
    assert mtp0_layout == mtp2_layout | {"mtp_tokens": 0}
    assert mtp2_layout["mtp_tokens"] == 2
    assert mtp2_layout["schema"] == attribution.LMCACHE_LAYOUT_SCHEMA
    assert mtp2_layout["dcp_size"] == 4
    assert mtp2_layout["kv_cache_memory_bytes_per_rank"] == 4_500_000_000
    assert mtp2_layout["max_model_len"] == 524_288
    assert mtp2_layout["lmcache_chunk_tokens_global"] == 512


def test_lmcache_layout_reset_policy_fails_closed_for_unknown_or_incompatible_l1(
    tmp_path,
):
    _, profile_path = generated(tmp_path)
    canonical = exl3.load_profile(profile_path)
    profiles = {
        arm: attribution.derive_profile(canonical, arm)
        for arm in attribution.ARMS
    }

    assert attribution.target_lmcache_reset_required(
        profiles["e-mtp0-apc0-lmcache1"],
        attribution.ARMS["e-mtp0-apc0-lmcache1"],
        profiles["f-mtp2-apc0-lmcache1"],
        attribution.ARMS["f-mtp2-apc0-lmcache1"],
    )
    # Cache-off source arms cannot attest what an older server process holds.
    assert attribution.target_lmcache_reset_required(
        profiles["b-mtp2-apc0-lmcache0"],
        attribution.ARMS["b-mtp2-apc0-lmcache0"],
        profiles["f-mtp2-apc0-lmcache1"],
        attribution.ARMS["f-mtp2-apc0-lmcache1"],
    )
    # Layout equality is not retention authorization until a live layout
    # receipt exists, so same-layout cache targets also get a cold L1.
    assert attribution.target_lmcache_reset_required(
        profiles["d-mtp2-apc1-lmcache1"],
        attribution.ARMS["d-mtp2-apc1-lmcache1"],
        profiles["f-mtp2-apc0-lmcache1"],
        attribution.ARMS["f-mtp2-apc0-lmcache1"],
    )
    # A cache-off target never reads the retained objects.
    assert not attribution.target_lmcache_reset_required(
        profiles["e-mtp0-apc0-lmcache1"],
        attribution.ARMS["e-mtp0-apc0-lmcache1"],
        profiles["a-mtp0-apc0-lmcache0"],
        attribution.ARMS["a-mtp0-apc0-lmcache0"],
    )


def test_all_arms_share_exact_container_and_runtime_envelope(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    normalized = {}
    for arm_id in attribution.ARMS:
        tokens = rendered_docker_tokens(site, canonical, arm_id)
        normalized[arm_id] = normalized_arm_tokens(tokens)
        assert "--privileged" in tokens
        env = {
            tokens[index + 1].split("=", 1)[0]: tokens[index + 1].split("=", 1)[1]
            for index, token in enumerate(tokens[:-1])
            if token == "--env" and "=" in tokens[index + 1]
        }
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
        assert env["LMCACHE_DISABLE_BANNER"] == "1"
        assert env["VLLM_SPARK_KV_CACHE_MEMORY_BYTES"] == "4500000000"
        assert env["VLLM_SPARK_MAX_MODEL_LEN"] == "524288"
        assert env["VLLM_SPARK_GRAPH_CAPTURE_SIZES"] == (
            "1,2,3,4,5,6,8,10,12,15,16,20,24,25,28,30,32"
        )
        assert env["VLLM_SPARK_NCCL_TRANSPORT_MODE"] == "switchless_ib"
    reference = normalized["a-mtp0-apc0-lmcache0"]
    assert all(tokens == reference for tokens in normalized.values())


def test_plan_attests_canonical_and_labels_diagnostic(tmp_path, capsys):
    site_path, profile_path = generated(tmp_path)
    rc = attribution.main(
        [
            "--site",
            str(site_path),
            "--profile",
            str(profile_path),
            "--arm",
            "f-mtp2-apc0-lmcache1",
            "plan",
        ]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mutates_remote"] is False
    assert plan["lmcache_l1_isolation"]["rollback_reset_required"] is True
    assert plan["startup_memory_hygiene"] == {
        "boundaries": [],
        "inner_model_verification": "unchanged-image-entrypoint",
        "outer_verified_entrypoint_sha256": None,
        "post_verification_host_page_cache_reclaim": False,
        "preflight_before_engine_removal": False,
        "requires_passwordless_sudo": False,
        "safety_class": "no-additional-mutation",
    }
    assert "/proc/sys/vm/drop_caches" not in json.dumps(plan["phases"])
    assert plan["canonical_attestation"]["image_id"] == IMAGE_ID
    assert plan["functional_settings"] == {
        "mtp_tokens": 2,
        "native_prefix_cache": False,
            "lmcache_connector": True,
            "cache_boundary_geometry": {
                "expected_engine_block_rows_per_dcp_rank": 64,
                "expected_dcp_size": 4,
                "expected_global_apc_alignment_tokens": 256,
                "expected_lmcache_chunk_tokens_global": 512,
                "runtime_attestation_required": True,
                "recipe_predecessor_chunk_size_is_geometry_evidence": False,
            },
    }
    commands = json.dumps(plan["phases"])
    assert attribution.LABEL_KEY in commands
    assert "LMCacheMPConnector" in commands
    assert "--enable-prefix-caching" not in commands
    assert "--no-enable-prefix-caching" in commands


def test_opt_in_reclaim_uses_attested_single_verifier_before_every_engine_start(
    tmp_path, capsys
):
    site_path, profile_path = generated(tmp_path)
    rc = attribution.main(
        [
            "--site",
            str(site_path),
            "--profile",
            str(profile_path),
            "--arm",
            "e-mtp0-apc0-lmcache1",
            "--reclaim-page-cache-after-verification",
            "plan",
        ]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["startup_memory_hygiene"] == {
        "boundaries": [
            "after the sole full model verification and before docker run/vLLM",
        ],
        "inner_model_verification": "skipped-by-sha256-attested-entrypoint-after-outer-pass",
        "outer_verified_entrypoint_sha256": attribution.outer_verified_entrypoint_sha256(),
        "post_verification_host_page_cache_reclaim": True,
        "preflight_before_engine_removal": True,
        "requires_passwordless_sudo": True,
        "safety_class": "MUTATES HOST",
    }
    assert plan["sequence"].index("prepare_page_cache_reclaim_entrypoint") < plan[
        "sequence"
    ].index("remove_canonical_engines")
    assert plan["rollback_phases"]["prepare_page_cache_reclaim_entrypoint"]
    hook = "sudo -n sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'"
    for phase_group, phase in (
        ("phases", "start_diagnostic"),
        ("rollback_phases", "start_canonical"),
    ):
        expected_hook_count = 1 if phase == "start_diagnostic" else 2
        for action in plan[phase_group][phase]:
            command = shlex.split(action["remote_command"])[-1]
            parsed = subprocess.run(
                ["bash", "-n", "-c", command],
                check=False,
                capture_output=True,
                text=True,
            )
            assert parsed.returncode == 0, parsed.stderr
            assert command.count(hook) == expected_hook_count
            assert command.index("verify_exl3_model.py") < command.index(hook)
            assert command.index(hook) < command.index("exec docker run --detach")
            assert attribution.REMOTE_OUTER_VERIFIED_ENTRYPOINT in command
            assert attribution.CONTAINER_ENTRYPOINT in command
            assert "SPARKRING_OUTER_MODEL_VERIFIED=1" in command
            assert attribution.outer_verified_entrypoint_sha256() in command

    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    derived = attribution.derive_profile(canonical, "e-mtp0-apc0-lmcache1")
    reclaimed = attribution.start_actions(
        site,
        derived,
        "e-mtp0-apc0-lmcache1",
        reclaim_page_cache_after_verification=True,
    )
    expected = attribution.expected_config_cmds(
        site, derived, "e-mtp0-apc0-lmcache1"
    )
    assert {
        action.rank: attribution._config_cmd_from_start_action(action, derived)
        for action in reclaimed
    } == expected


def test_outer_verified_entrypoint_and_prepare_actions_are_byte_attested(tmp_path):
    site_path, _ = generated(tmp_path)
    site = load_site(site_path)
    payload = attribution.OUTER_VERIFIED_ENTRYPOINT.read_bytes()
    text = payload.decode("utf-8")
    digest = attribution.outer_verified_entrypoint_sha256()
    assert b"\r\n" not in payload
    assert "verify_exl3_model.py" not in text
    assert "SPARKRING_OUTER_MODEL_VERIFIED" in text
    assert "SPARKRING_OUTER_ENTRYPOINT_SHA256" in text
    assert "/opt/sparkring/verify-runtime.py" in text
    assert "/opt/sparkring-exl3/verify_exl3_runtime.py" in text
    assert "exec /opt/venv/bin/vllm \"$@\"" in text
    encoded = base64.b64encode(payload).decode("ascii")
    actions = attribution.prepare_page_cache_reclaim_entrypoint_actions(site)
    assert len(actions) == 4
    for action in actions:
        command = action.argv[-1]
        assert "sudo -n sh -c 'test -w /proc/sys/vm/drop_caches'" in command
        assert command.startswith(
            "sudo -n sh -c 'test -w /proc/sys/vm/drop_caches' && "
        )
        assert "install -D -m 0555" in command
        assert (
            f"{attribution.REMOTE_OUTER_VERIFIED_ENTRYPOINT} && test"
            in command
        )
        assert encoded in command
        assert command.count(digest) == 2
        parsed = subprocess.run(
            ["bash", "-n", "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_reclaim_preflight_failure_prevents_engine_removal_and_rollback(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor(failures={4})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv.insert(-1, "--reclaim-page-cache-after-verification")
    assert attribution.main(argv) == 1
    assert executor.calls == [
        "server_health",
        "canonical_engine_exclusive",
        "no_other_diagnostic",
        "prepare_page_cache_reclaim_entrypoint",
    ]
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_rollback"] is None
    assert "remove_canonical_engines" not in report["results"]


def test_reclaim_execution_prepares_wrapper_before_removal_and_reclaims_before_start(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv.insert(-1, "--reclaim-page-cache-after-verification")
    assert attribution.main(argv) == 0
    capsys.readouterr()
    assert executor.calls == attribution.sequence(
        "activate", reclaim_page_cache_after_verification=True
    )
    assert executor.calls.index("prepare_page_cache_reclaim_entrypoint") < executor.calls.index(
        "remove_canonical_engines"
    )
    assert executor.calls.index("start_diagnostic") < executor.calls.index(
        "diagnostic_ready"
    )


def test_failed_rollback_wrapper_preparation_preserves_diagnostic_engine(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    # Forward readiness fails after cutover; rollback preparation then fails.
    # The executor must not remove the only remaining engine afterward.
    executor = RecordingExecutor(failures={7, 8})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv.insert(-1, "--reclaim-page-cache-after-verification")
    assert attribution.main(argv) == 1
    assert executor.calls[-1] == "prepare_page_cache_reclaim_entrypoint"
    assert "remove_diagnostic" not in executor.calls[7:]
    report = json.loads(capsys.readouterr().out)
    assert "automatic_rollback_failed" in report["results"]


def test_malformed_rollback_wrapper_preparation_preserves_diagnostic_engine(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)

    class MalformedRollbackPrepareExecutor(RecordingExecutor):
        def __call__(self, actions, timeout):
            del timeout
            self.actions.append(actions)
            self.calls.append(phase_name(actions))
            if len(self.calls) == 7:
                return phase_result(failed=True, partial=True)
            if len(self.calls) == 8:
                return None
            return phase_result()

    executor = MalformedRollbackPrepareExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv.insert(-1, "--reclaim-page-cache-after-verification")
    assert attribution.main(argv) == 1
    assert executor.calls[-1] == "prepare_page_cache_reclaim_entrypoint"
    assert "remove_diagnostic" not in executor.calls[7:]
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_rollback"][
        "prepare_page_cache_reclaim_entrypoint"
    ] == {"malformed_executor_result": True}


def test_cache_off_start_omits_connector_and_has_guards(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    derived = attribution.derive_profile(canonical, "b-mtp2-apc0-lmcache0")
    actions = attribution.start_actions(site, derived, "b-mtp2-apc0-lmcache0")
    assert len(actions) == 4
    for action in actions:
        command = action.shell_command
        assert "LMCacheMPConnector" not in command
        assert f"{attribution.LABEL_KEY}=b-mtp2-apc0-lmcache0" in command
        assert f"{attribution.COMPONENT_LABEL}=engine" in command


def test_ready_attests_effective_connector_presence_absence_and_initialization(
    tmp_path,
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    for arm, expected in (
        ("d-mtp2-apc1-lmcache1", True),
        ("c-mtp2-apc1-lmcache0", False),
    ):
        profile = attribution.derive_profile(canonical, arm)
        commands = "\n".join(
            action.shell_command
            for action in attribution.ready_actions(site, profile, arm)
        )
        assert "python3 -c" in commands
        assert "--kv-transfer-config" in commands
        assert "Creating v1 connector with name:" in commands
        if expected:
            assert '"kv_connector":"LMCacheMPConnector"' in commands
            assert "Creating v1 connector with name: LMCacheMPConnector" in commands
            assert "lmcache.mp.heartbeat_interval = 10.0" in commands
            assert "LMCache MP worker adapter created with instance_id=" in commands
            assert "len(p)==1" in commands
            assert "json.loads(a[p[0]+1])==e" in commands
        else:
            assert "not in a else 1" in commands
            assert "! docker logs" in commands
            assert "LMCache MP worker adapter created with instance_id=" in commands


def test_engine_exclusivity_ignores_exited_managed_history_but_rejects_extra_running_engines(
    tmp_path,
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "a-mtp0-apc0-lmcache0"
    profile = attribution.derive_profile(canonical, arm)
    commands = "\n".join(
        action.shell_command
        for action in attribution.exclusive_diagnostic_actions(site, profile, arm)
    )
    assert "docker ps -q --filter label=org.sparkring.managed=true" in commands
    assert "docker ps -aq" not in commands
    assert "--filter label=org.sparkring.managed=true" in commands
    assert "--filter label=org.sparkring.component=engine" not in commands
    assert "lmcache-server" in commands
    assert "engine_ids=" in commands
    assert "sparkcache-engine" not in commands  # it is caught by the default branch
    assert 'test "$#" -eq 1' in commands
    assert 'test "$observed_name" = "/$name"' in commands
    assert "exit 76" in commands
    assert "exit 79" in commands
    assert "exit 80" in commands
    assert "exit 81" in commands
    assert "--filter label=org.sparkring.exl3-attribution" not in commands


def test_no_other_diagnostic_check_ignores_exited_attribution_history(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    commands = "\n".join(
        action.shell_command
        for action in attribution.no_other_diagnostic_actions(site, canonical)
    )
    assert "docker ps -q --filter label=org.sparkring.exl3-attribution" in commands
    assert "docker ps -aq" not in commands


def test_rollback_is_exact_identity_guarded(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "e-mtp0-apc0-lmcache1"
    derived = attribution.derive_profile(canonical, arm)
    actions = attribution.diagnostic_remove_actions(site, derived, arm)
    assert len(actions) == 4
    for action in actions:
        command = action.shell_command
        assert derived.profile_id in command
        assert attribution.LABEL_KEY in command
        assert "exit 73" in command
        assert "exit 74" in command
        assert "exit 75" in command
        assert "docker rm --force" in command


@pytest.mark.parametrize(
    ("component", "arm_id"),
    (("engine", "e-mtp0-apc0-lmcache1"), ("lmcache-server", None)),
)
def test_strict_removal_requires_all_identity_labels_before_remove(
    tmp_path, component, arm_id
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    profile = (
        attribution.derive_profile(canonical, arm_id)
        if arm_id is not None
        else canonical
    )
    actions = attribution.strict_remove_actions(
        site, profile, component=component, arm_id=arm_id
    )
    for action in actions:
        command = action.shell_command
        ordered = [
            command.index('test "$observed_name" = "/$name"'),
            command.index('test "$managed" = true'),
            command.index('test "$profile" ='),
            command.index('test "$component" ='),
        ]
        if arm_id is not None:
            ordered.append(command.index('test "$arm" ='))
        ordered.append(command.index("docker rm --force"))
        assert ordered == sorted(ordered)
        assert "|| exit 72" in command
        assert "|| exit 73" in command
        assert "|| exit 74" in command
        assert "|| exit 75" in command
        if arm_id is not None:
            assert "|| exit 76" in command


def test_live_arm_attestation_requires_exact_labels_image_env_and_command(
    tmp_path,
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "d-mtp2-apc1-lmcache1"
    diagnostic = attribution.derive_profile(canonical, arm)
    for action in attribution.live_arm_attestation_actions(
        site, diagnostic, arm
    ):
        command = action.shell_command
        for field in (
            ".Name",
            attribution.MANAGED_LABEL,
            "org.sparkring.exl3-profile",
            attribution.COMPONENT_LABEL,
            attribution.LABEL_KEY,
            ".Image",
            ".Config.Env",
            ".Config.Cmd",
        ):
            assert field in command
        for exit_code in range(92, 100):
            assert f"exit {exit_code}" in command


def test_diagnostic_readiness_waits_for_full_runtime_attestation(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "f-mtp2-apc0-lmcache1"
    derived = attribution.derive_profile(canonical, arm)
    actions = attribution.ready_actions(site, derived, arm)
    assert "/health" in actions[0].shell_command
    for action in actions:
        wait_loop, after_ready_break = action.shell_command.split("&& break;", 1)
        _, final_checks = after_ready_break.split("sleep 5; done;", 1)
        # API health alone is not readiness: connector construction and the
        # exact command must be present before the loop may break.  Each is
        # then repeated after the loop to retain a precise timeout exit code.
        for marker in (
            "LMCache MP worker adapter created with instance_id=",
            "lmcache.mp.heartbeat_interval = 10.0",
            "python3 -c",
        ):
            assert marker in wait_loop
            assert marker in final_checks
        assert ".RestartCount" in action.shell_command
        assert ".State.OOMKilled" in action.shell_command
        assert "deadline=$(( $(date +%s) + 3420 ))" in action.shell_command
        assert "seq 1 720" not in action.shell_command
        assert 'case "$state" in ""|exited|dead|removing)' in action.shell_command
        assert final_checks.index(".State.OOMKilled") < final_checks.index(
            ".RestartCount"
        )
        assert final_checks.index(".RestartCount") < final_checks.index(
            ".State.Running"
        )
        assert "--no-enable-prefix-caching" in action.shell_command
        for exit_code in range(82, 92):
            assert f"exit {exit_code}" in action.shell_command
    assert "enable_prefix_caching=False" in actions[0].shell_command
    assert "num_spec_tokens=2" in actions[0].shell_command
    assert actions[0].shell_command.count("enable_prefix_caching=False") == 2
    assert actions[0].shell_command.count("num_spec_tokens=2") == 2
    assert "enable_prefix_caching=False" not in actions[1].shell_command


def test_mtp0_readiness_attests_effective_disabled_states(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "a-mtp0-apc0-lmcache0"
    derived = attribution.derive_profile(canonical, arm)
    actions = attribution.ready_actions(site, derived, arm)
    assert "enable_prefix_caching=False" in actions[0].shell_command
    assert "speculative_config=None" in actions[0].shell_command
    for action in actions:
        assert "--no-enable-prefix-caching" in action.shell_command


def test_apc_on_readiness_attests_effective_enabled_state(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "c-mtp2-apc1-lmcache0"
    derived = attribution.derive_profile(canonical, arm)
    actions = attribution.ready_actions(site, derived, arm)
    assert "enable_prefix_caching=True" in actions[0].shell_command
    assert "num_spec_tokens=2" in actions[0].shell_command
    for action in actions:
        assert "--enable-prefix-caching" in action.shell_command
        assert '"--no-enable-prefix-caching"' not in action.shell_command


def test_activation_sequence_checks_before_removing_canonical():
    assert attribution.sequence("activate")[:4] == [
        "server_health",
        "canonical_engine_exclusive",
        "no_other_diagnostic",
        "remove_canonical_engines",
    ]
    assert attribution.sequence("rollback")[-1] == (
        "canonical_engine_exclusive_after_restore"
    )


@pytest.mark.parametrize("confirmation", [None, "WRONG-CONFIRMATION"])
def test_bad_or_missing_confirmation_makes_zero_remote_calls(
    tmp_path, monkeypatch, confirmation
):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    with pytest.raises(SystemExit) as caught:
        attribution.main(
            execution_argv(
                site_path, profile_path, confirmation=confirmation
            )
        )
    assert caught.value.code == 2
    assert executor.calls == []


def test_existing_output_refuses_before_remote_calls(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    output = tmp_path / "evidence.json"
    output.write_text("keep", encoding="utf-8")
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv[0:0] = ["--output", str(output)]
    with pytest.raises(SystemExit) as caught:
        attribution.main(argv)
    assert caught.value.code == 2
    assert executor.calls == []
    assert output.read_text(encoding="utf-8") == "keep"
    assert "already exists" in capsys.readouterr().err


def test_output_parent_creation_failure_refuses_before_remote_calls(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    blocking_parent = tmp_path / "not-a-directory"
    blocking_parent.write_text("block", encoding="utf-8")
    output = blocking_parent / "evidence.json"
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv[0:0] = ["--output", str(output)]
    with pytest.raises(SystemExit) as caught:
        attribution.main(argv)
    assert caught.value.code == 2
    assert executor.calls == []
    assert "cannot create" in capsys.readouterr().err


def test_successful_execution_output_is_byte_identical_to_stdout(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    output = tmp_path / "evidence" / "activation.json"
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv[0:0] = ["--output", str(output)]
    assert attribution.main(argv) == 0
    stdout = capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == stdout
    assert json.loads(stdout)["automatic_rollback"] is None
    assert executor.calls == attribution.sequence("activate")
    receipt_path = tmp_path / "activate-live-arm-receipt.json"
    receipt = validate_live_arm_receipt(
        receipt_path,
        profile_path,
        "a-mtp0-apc0-lmcache0",
    )
    assert [
        (item["rank"], item["status"]) for item in receipt["ranks"]
    ] == [(rank, "attested") for rank in range(4)]
    assert all(
        item["labels"][attribution.LABEL_KEY]
        == "a-mtp0-apc0-lmcache0"
        for item in receipt["ranks"]
    )
    assert all(
        len(item["explicit_environment_sha256"]) == 64
        and len(item["config_cmd_sha256"]) == 64
        for item in receipt["ranks"]
    )
    assert [item["container_id"] for item in receipt["ranks"]] == [
        f"{rank + 1:064x}" for rank in range(4)
    ]


def test_old_same_arm_receipt_revalidates_runtime_unique_identity_on_all_ranks(
    tmp_path, monkeypatch
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "d-mtp2-apc1-lmcache1"
    diagnostic = attribution.derive_profile(canonical, arm)
    environment_digests, command_digests = (
        attribution.live_arm_receipt_rank_digests(site, diagnostic, arm)
    )
    old_instances = [
        {
            "container_id": f"{rank + 1:064x}",
            "started_at": f"2026-08-10T03:2{rank}:00.123456789Z",
        }
        for rank in range(4)
    ]
    receipt = build_live_arm_receipt(
        arm_id=arm,
        canonical_profile_id=canonical.profile_id,
        canonical_profile_file_sha256="9" * 64,
        image_id=canonical.image_id,
        model_repository=canonical.model_repository,
        model_revision=canonical.model_revision,
        canonical_container_name=canonical.container_name,
        explicit_environment_sha256=environment_digests,
        config_cmd_sha256=command_digests,
        observed_runtime_instances=old_instances,
    )
    actions = attribution.live_arm_revalidation_actions(
        site, canonical, arm, receipt
    )
    assert len(actions) == 4
    for action in actions:
        expected = old_instances[action.rank]
        assert expected["container_id"] in action.argv[-1]
        assert expected["started_at"] in action.argv[-1]
        assert "exit 100" in action.argv[-1]
        assert "exit 101" in action.argv[-1]
        assert ".State.Running" in action.argv[-1]
        assert ".State.OOMKilled" in action.argv[-1]
        assert ".RestartCount" in action.argv[-1]

    def all_four_replaced(_actions, timeout):
        del timeout
        result = phase_result()
        for item in result.values():
            item["exit_code"] = 100
        return result

    monkeypatch.setattr(attribution.exl3, "execute", all_four_replaced)
    with pytest.raises(exl3.ProfileError, match="re-attestation failed"):
        attribution.revalidate_live_arm(
            site, canonical, arm, receipt, timeout=60
        )


def test_bad_live_arm_attestation_rolls_back_and_leaves_no_valid_receipt(
    tmp_path, monkeypatch, capsys
):
    class BadAttestationExecutor(RecordingExecutor):
        def __call__(self, actions, timeout):
            result = super().__call__(actions, timeout)
            if self.calls[-1] == "diagnostic_live_arm_attestation":
                result[2]["stdout"] = json.dumps(
                    {"rank": 2, "status": "wrong-arm"}
                )
            return result

    site_path, profile_path = generated(tmp_path)
    executor = BadAttestationExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    assert attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    ) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["original_exception"]["phase"] == (
        "diagnostic_live_arm_attestation"
    )
    receipt_path = tmp_path / "activate-live-arm-receipt.json"
    assert receipt_path.read_text(encoding="utf-8") == ""
    with pytest.raises(ValueError):
        validate_live_arm_receipt(
            receipt_path, profile_path, "a-mtp0-apc0-lmcache0"
        )


def test_plan_output_is_byte_identical_and_remote_free(tmp_path, monkeypatch, capsys):
    site_path, profile_path = generated(tmp_path)
    output = tmp_path / "plan.json"
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = [
        "--output",
        str(output),
        "--site",
        str(site_path),
        "--profile",
        str(profile_path),
        "--arm",
        "a-mtp0-apc0-lmcache0",
        "plan",
    ]
    assert attribution.main(argv) == 0
    stdout = capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == stdout
    assert json.loads(stdout)["command"] == "plan"
    assert executor.calls == []


def test_failure_before_canonical_removal_does_not_attempt_rollback(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor(failures={3})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    )
    assert rc == 1
    assert executor.calls == [
        "server_health",
        "canonical_engine_exclusive",
        "no_other_diagnostic",
    ]
    assert json.loads(capsys.readouterr().out)["automatic_rollback"] is None


def test_partial_canonical_removal_triggers_full_rollback(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor(failures={4})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    )
    assert rc == 1
    assert executor.calls == [
        "server_health",
        "canonical_engine_exclusive",
        "no_other_diagnostic",
        "remove_canonical_engines",
    ] + attribution.sequence("rollback")
    assert json.loads(capsys.readouterr().out)["automatic_rollback"] is not None


@pytest.mark.parametrize("failure_call", [5, 6])
def test_diagnostic_start_or_readiness_failure_triggers_rollback(
    tmp_path, monkeypatch, capsys, failure_call
):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor(failures={failure_call})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    )
    assert rc == 1
    failed_phase = "start_diagnostic" if failure_call == 5 else "diagnostic_ready"
    assert executor.calls[failure_call - 1] == failed_phase
    assert executor.calls[-len(attribution.sequence("rollback")) :] == attribution.sequence("rollback")
    assert json.loads(capsys.readouterr().out)["automatic_rollback"] is not None


def test_rollback_failure_is_surfaced_nonzero(tmp_path, monkeypatch, capsys):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor(failures={5, 6})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    )
    assert rc == 1
    assert executor.calls == [
        "server_health",
        "canonical_engine_exclusive",
        "no_other_diagnostic",
        "remove_canonical_engines",
        "start_diagnostic",
        "remove_diagnostic",
    ]
    report = json.loads(capsys.readouterr().out)
    assert "automatic_rollback_failed" in report["results"]
    assert report["automatic_rollback"] is not None


@pytest.mark.parametrize("raise_call", [4, 5, 6])
def test_exception_during_or_after_removal_is_recorded_and_rolls_back(
    tmp_path, monkeypatch, capsys, raise_call
):
    site_path, profile_path = generated(tmp_path)
    executor = RaisingExecutor({raise_call})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    )
    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert report["original_exception"] == {
        "phase": attribution.sequence("activate")[raise_call - 1],
        "type": "RuntimeError",
        "message": f"injected-{attribution.sequence('activate')[raise_call - 1]}",
    }
    assert report["automatic_rollback"] is not None
    assert executor.calls[-len(attribution.sequence("rollback")) :] == attribution.sequence("rollback")


def test_exception_before_removal_is_recorded_without_rollback(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = RaisingExecutor({3})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    assert attribution.main(
        execution_argv(site_path, profile_path, confirmation=attribution.CONFIRMATION)
    ) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["original_exception"]["phase"] == "no_other_diagnostic"
    assert report["automatic_rollback"] is None


def test_rollback_exception_preserves_both_exception_evidence(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = RaisingExecutor({5, 6})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    assert attribution.main(
        execution_argv(site_path, profile_path, confirmation=attribution.CONFIRMATION)
    ) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["original_exception"]["phase"] == "start_diagnostic"
    assert report["automatic_rollback"]["rollback_exceptions"][0]["phase"] == "remove_diagnostic"
    assert "automatic_rollback_failed" in report["results"]


def test_local_evidence_write_exception_after_removal_triggers_rollback(
    tmp_path, monkeypatch, capsys
):
    class BrokenStream:
        def write(self, value):
            del value
            raise OSError("private output unavailable")

        def flush(self):
            pass

        def close(self):
            pass

    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    monkeypatch.setattr(attribution, "reserve_output", lambda path: BrokenStream())
    argv = execution_argv(
        site_path, profile_path, confirmation=attribution.CONFIRMATION
    )
    argv[0:0] = ["--output", str(tmp_path / "raw.json")]
    assert attribution.main(argv) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["original_exception"]["phase"] == "emit_execution_report"
    assert report["automatic_rollback"] is not None
    assert executor.calls[-len(attribution.sequence("rollback")) :] == attribution.sequence("rollback")


def test_successful_activation_executes_each_phase_once_in_order(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    )
    assert rc == 0
    assert executor.calls == attribution.sequence("activate")
    assert len(executor.calls) == len(set(executor.calls))
    assert json.loads(capsys.readouterr().out)["automatic_rollback"] is None


def test_transition_requires_from_arm_before_remote_calls(tmp_path, monkeypatch):
    site_path, profile_path = generated(tmp_path)
    executor = TransitionRecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    with pytest.raises(SystemExit) as caught:
        attribution.main(
            transition_argv(
                site_path,
                profile_path,
                confirmation=attribution.CONFIRMATION,
            )
        )
    assert caught.value.code == 2
    assert executor.calls == []


def test_transition_rejects_same_source_and_target_before_remote_calls(
    tmp_path, monkeypatch
):
    site_path, profile_path = generated(tmp_path)
    executor = TransitionRecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = transition_argv(
        site_path,
        profile_path,
        from_arm="b-mtp2-apc0-lmcache0",
        confirmation=attribution.CONFIRMATION,
    )
    with pytest.raises(SystemExit) as caught:
        attribution.main(argv)
    assert caught.value.code == 2
    assert executor.calls == []


def test_restart_arm_plan_cold_resets_servers_and_restarts_exact_same_arm(
    tmp_path, capsys
):
    site_path, profile_path = generated(tmp_path)
    assert attribution.main(restart_arm_argv(site_path, profile_path)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["command"] == "restart-arm"
    assert plan["mutates_remote"] is True
    assert plan["from_arm"] == plan["arm"] == "d-mtp2-apc1-lmcache1"
    assert plan["source_diagnostic_identity"] == plan["diagnostic_identity"] | {
        "functional_settings": plan["functional_settings"]
    }
    assert plan["sequence"] == attribution.transition_sequence(
        reset_target_lmcache=True
    )
    assert "post_source_removal_server_health" not in plan["sequence"]
    rendered = json.dumps(plan["phases"])
    assert "LMCacheMPConnector" in rendered
    removal = json.dumps(plan["phases"]["remove_source_diagnostic"])
    start = json.dumps(plan["phases"]["start_target_diagnostic"])
    assert "lmcache-server" not in removal
    assert "lmcache-server" not in start
    assert plan["lmcache_l1_isolation"]["forward_reset_required"] is True
    assert plan["lmcache_l1_isolation"]["rollback_reset_required"] is True
    assert [
        phase
        for phase in plan["sequence"]
        if phase.startswith("isolate_target_lmcache_")
    ] == [
        "isolate_target_lmcache_remove_servers",
        "isolate_target_lmcache_start_servers",
        "isolate_target_lmcache_server_health",
    ]


def test_incompatible_cache_attached_transition_recreates_l1_before_target(
    tmp_path, capsys
):
    site_path, profile_path = generated(tmp_path)
    argv = custom_transition_argv(
        site_path,
        profile_path,
        from_arm="e-mtp0-apc0-lmcache1",
        arm="f-mtp2-apc0-lmcache1",
    )
    assert attribution.main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    isolation = plan["lmcache_l1_isolation"]
    assert isolation == {
        "contract": "process-lifetime-namespace/v1",
        "model_name_keying_assumed_safe": False,
        "source_layout": isolation["source_layout"],
        "target_layout": isolation["target_layout"],
        "target_uses_lmcache": True,
        "forward_reset_required": True,
        "rollback_reset_required": True,
        "reset_mechanism": "remove-and-recreate-all-four-lmcache-server-containers",
        "required_request_cache_salt": isolation[
            "required_request_cache_salt"
        ],
        "request_cache_salt_source": "scripts/exl3_attribution_cache_contract.py",
    }
    target_profile = attribution.derive_profile(
        exl3.load_profile(profile_path), "f-mtp2-apc0-lmcache1"
    )
    assert isolation["required_request_cache_salt"] == (
        attribution.attribution_cache_salt(
            target_profile, "f-mtp2-apc0-lmcache1"
        )
    )
    assert isolation["source_layout"]["mtp_tokens"] == 0
    assert isolation["target_layout"]["mtp_tokens"] == 2
    sequence = plan["sequence"]
    reset = [
        "isolate_target_lmcache_remove_servers",
        "isolate_target_lmcache_start_servers",
        "isolate_target_lmcache_server_health",
    ]
    assert sequence[
        sequence.index("remove_source_diagnostic") + 1 :
        sequence.index("remove_source_diagnostic") + 4
    ] == reset
    assert "post_source_removal_server_health" not in sequence
    assert sequence.index("remove_source_diagnostic") < sequence.index(reset[0])
    assert sequence.index(reset[0]) < sequence.index(reset[1])
    assert sequence.index(reset[1]) < sequence.index(reset[2])
    assert sequence.index(reset[2]) < sequence.index("start_target_diagnostic")

    removal = plan["phases"][reset[0]]
    start = plan["phases"][reset[1]]
    health = plan["phases"][reset[2]]
    assert len(removal) == len(start) == len(health) == 4
    for rank in range(4):
        server = attribution.lmcache.server_name(rank)
        assert server in removal[rank]["remote_command"]
        assert "org.sparkring.component" in removal[rank]["remote_command"]
        assert "docker rm --force" in removal[rank]["remote_command"]
        assert server in start[rank]["remote_command"]
        assert "docker run --detach" in start[rank]["remote_command"]
        assert server in health[rank]["remote_command"]
        assert "/healthcheck" in health[rank]["remote_command"]


def test_same_layout_cache_transition_and_rollback_both_reset_l1(
    tmp_path, capsys
):
    site_path, profile_path = generated(tmp_path)
    assert attribution.main(
        custom_transition_argv(
            site_path,
            profile_path,
            from_arm="d-mtp2-apc1-lmcache1",
            arm="f-mtp2-apc0-lmcache1",
        )
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["lmcache_l1_isolation"]["forward_reset_required"] is True
    assert [
        phase
        for phase in plan["sequence"]
        if phase.startswith("isolate_target_lmcache_")
    ] == [
        "isolate_target_lmcache_remove_servers",
        "isolate_target_lmcache_start_servers",
        "isolate_target_lmcache_server_health",
    ]
    rollback_reset = [
        "rollback_reset_lmcache_remove_servers",
        "rollback_reset_lmcache_start_servers",
        "rollback_reset_lmcache_server_health",
    ]
    assert all(phase in plan["rollback_phases"] for phase in rollback_reset)
    assert plan["rollback_phases"][rollback_reset[0]]
    sequence = attribution.transition_rollback_sequence()
    assert sequence.index("remove_target_diagnostic") < sequence.index(
        "rollback_remove_canonical_engines"
    )
    assert sequence.index("rollback_remove_canonical_engines") < sequence.index(
        rollback_reset[0]
    )
    assert sequence.index(rollback_reset[-1]) < sequence.index(
        "start_canonical"
    )


def test_activation_of_mtp0_cache_arm_isolates_canonical_mtp2_l1(
    tmp_path, capsys
):
    site_path, profile_path = generated(tmp_path)
    assert attribution.main(
        [
            "--site",
            str(site_path),
            "--profile",
            str(profile_path),
            "--arm",
            "e-mtp0-apc0-lmcache1",
            "plan",
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["lmcache_l1_isolation"]["source_layout"]["mtp_tokens"] == 2
    assert plan["lmcache_l1_isolation"]["target_layout"]["mtp_tokens"] == 0
    assert plan["lmcache_l1_isolation"]["forward_reset_required"] is True
    sequence = plan["sequence"]
    assert sequence.index("remove_canonical_engines") < sequence.index(
        "isolate_target_lmcache_remove_servers"
    )
    assert sequence.index("isolate_target_lmcache_server_health") < sequence.index(
        "start_diagnostic"
    )


def test_restart_arm_rejects_from_arm_before_remote_calls(tmp_path, monkeypatch):
    site_path, profile_path = generated(tmp_path)
    executor = TransitionRecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    with pytest.raises(SystemExit) as caught:
        attribution.main(
            restart_arm_argv(
                site_path,
                profile_path,
                from_arm="c-mtp2-apc1-lmcache0",
            )
        )
    assert caught.value.code == 2
    assert executor.calls == []


def test_wrong_transition_source_fails_readiness_without_removal_or_rollback(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = TransitionRecordingExecutor(failures={2})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        transition_argv(
            site_path,
            profile_path,
            from_arm="a-mtp0-apc0-lmcache0",
            confirmation=attribution.CONFIRMATION,
        )
    )
    assert rc == 1
    assert executor.calls == ["source_server_health", "source_diagnostic_ready"]
    assert json.loads(capsys.readouterr().out)["automatic_rollback"] is None


def test_other_diagnostic_presence_fails_before_removal_without_rollback(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = TransitionRecordingExecutor(failures={3})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        transition_argv(
            site_path,
            profile_path,
            from_arm="a-mtp0-apc0-lmcache0",
            confirmation=attribution.CONFIRMATION,
        )
    )
    assert rc == 1
    assert executor.calls == [
        "source_server_health",
        "source_diagnostic_ready",
        "source_diagnostic_exclusive",
    ]
    assert json.loads(capsys.readouterr().out)["automatic_rollback"] is None


def test_transition_cleanup_is_exact_source_and_target_scoped(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    source_arm = "a-mtp0-apc0-lmcache0"
    target_arm = "b-mtp2-apc0-lmcache0"
    source = attribution.derive_profile(canonical, source_arm)
    target = attribution.derive_profile(canonical, target_arm)
    all_phases = attribution.transition_phases(
        site, canonical, source, source_arm, target, target_arm
    )
    source_ready = "\n".join(
        action.shell_command for action in all_phases["source_diagnostic_ready"]
    )
    target_ready = "\n".join(
        action.shell_command for action in all_phases["target_diagnostic_ready"]
    )
    assert "seq 1 720" not in source_ready
    assert "deadline=$(( $(date +%s) + 3420 ))" not in source_ready
    assert "deadline=$(( $(date +%s) + 3420 ))" in target_ready
    exclusive = "\n".join(
        action.shell_command
        for action in all_phases["source_diagnostic_exclusive"]
    )
    assert source.profile_id in exclusive
    assert source.container_name in exclusive
    assert source_arm in exclusive
    assert "exit 77" in exclusive
    assert "exit 78" in exclusive
    assert "exit 79" in exclusive
    assert "exit 80" in exclusive
    for phase, profile, arm in (
        ("remove_source_diagnostic", source, source_arm),
        ("remove_target_diagnostic", target, target_arm),
    ):
        commands = "\n".join(
            action.shell_command for action in all_phases[phase]
        )
        assert profile.profile_id in commands
        assert profile.container_name in commands
        assert arm in commands
        assert "org.sparkring.component" in commands
        assert "docker rm --force" in commands
        assert "lmcache-server" not in commands


def test_transition_failure_after_source_removal_attempt_restores_canonical(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = TransitionRecordingExecutor(failures={4})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        transition_argv(
            site_path,
            profile_path,
            from_arm="a-mtp0-apc0-lmcache0",
            confirmation=attribution.CONFIRMATION,
        )
    )
    assert rc == 1
    assert executor.calls == [
        "source_server_health",
        "source_diagnostic_ready",
        "source_diagnostic_exclusive",
        "remove_source_diagnostic",
    ] + attribution.transition_rollback_sequence()
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_rollback"] is not None


def test_transition_exception_during_source_removal_restores_canonical_with_evidence(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    executor = TransitionRaisingExecutor({4})
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        transition_argv(
            site_path,
            profile_path,
            from_arm="a-mtp0-apc0-lmcache0",
            confirmation=attribution.CONFIRMATION,
        )
    )
    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert report["original_exception"]["phase"] == "remove_source_diagnostic"
    assert report["automatic_rollback"] is not None
    assert executor.calls[-len(attribution.transition_rollback_sequence()) :] == attribution.transition_rollback_sequence()


def test_successful_transition_sequence_and_plan_identities(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)
    output = tmp_path / "transition.json"
    executor = TransitionRecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    argv = transition_argv(
        site_path,
        profile_path,
        from_arm="a-mtp0-apc0-lmcache0",
        confirmation=attribution.CONFIRMATION,
    )
    argv[0:0] = ["--output", str(output)]
    assert attribution.main(argv) == 0
    assert executor.calls == attribution.transition_sequence()
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_rollback"] is None
    plan = report["plan"]
    assert plan["from_arm"] == "a-mtp0-apc0-lmcache0"
    assert plan["arm"] == "b-mtp2-apc0-lmcache0"
    assert plan["source_diagnostic_identity"]["attribution_label"] == plan["from_arm"]
    assert plan["diagnostic_identity"]["attribution_label"] == plan["arm"]
    assert plan["sequence"] == attribution.transition_sequence()
    assert set(attribution.transition_rollback_sequence()) <= set(
        plan["rollback_phases"]
    )
    assert output.read_text(encoding="utf-8") == json.dumps(
        report, indent=2, sort_keys=True
    ) + "\n"


def test_explicit_rollback_is_guarded_and_ordered(tmp_path, monkeypatch, capsys):
    site_path, profile_path = generated(tmp_path)
    executor = RecordingExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    rc = attribution.main(
        execution_argv(
            site_path,
            profile_path,
            command="rollback",
            confirmation=attribution.CONFIRMATION,
        )
    )
    assert rc == 0
    assert executor.calls == attribution.sequence("rollback")
    removal_commands = "\n".join(
        action.shell_command for action in executor.actions[0]
    )
    assert attribution.LABEL_KEY in removal_commands
    assert "exit 73" in removal_commands
    assert "exit 74" in removal_commands
    assert "exit 75" in removal_commands
    capsys.readouterr()


def test_canonical_restore_is_idempotent_for_partial_cutover(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    actions = attribution.canonical_restore_actions(site, canonical)
    assert len(actions) == 4
    for rank, action in enumerate(actions):
        command = action.shell_command
        assert exl3.container_name(canonical, rank) in command
        assert f"if {canonical.engine} inspect" in command
        assert canonical.profile_id in command
        assert attribution.MANAGED_LABEL in command
        assert attribution.COMPONENT_LABEL in command
        assert ".State.Running" in command
        assert ".Config.Cmd" in command
        assert f'{canonical.engine} rm "$name"' in command
        assert command.index(attribution.MANAGED_LABEL) < command.index(
            f'{canonical.engine} rm "$name"'
        )
        assert "LMCacheMPConnector" in command


def test_malformed_result_during_removal_triggers_full_best_effort_rollback(
    tmp_path, monkeypatch, capsys
):
    site_path, profile_path = generated(tmp_path)

    class MalformedRemovalExecutor(RecordingExecutor):
        def __call__(self, actions, timeout):
            del timeout
            self.actions.append(actions)
            self.calls.append(phase_name(actions))
            if len(self.calls) == 4:
                return None
            return phase_result()

    executor = MalformedRemovalExecutor()
    monkeypatch.setattr(attribution.exl3, "execute", executor)
    assert attribution.main(
        execution_argv(
            site_path, profile_path, confirmation=attribution.CONFIRMATION
        )
    ) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["original_exception"]["phase"] == "remove_canonical_engines"
    assert executor.calls[-len(attribution.sequence("rollback")) :] == attribution.sequence("rollback")


def test_rollback_cleanup_failure_blocks_cache_reset_and_canonical_start(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "a-mtp0-apc0-lmcache0"
    diagnostic = attribution.derive_profile(canonical, arm)
    all_phases = attribution.phases(site, canonical, diagnostic, arm)
    calls = []

    def executor(actions, timeout):
        del timeout
        name = phase_name(actions)
        calls.append(name)
        if name == "remove_diagnostic":
            raise RuntimeError("injected cleanup failure")
        return phase_result()

    original = attribution.exl3.execute
    attribution.exl3.execute = executor
    try:
        evidence, restored = attribution.attempt_automatic_rollback(
            attribution.sequence("rollback"), all_phases, 1
        )
    finally:
        attribution.exl3.execute = original
    assert calls == ["remove_diagnostic"]
    assert restored is False
    assert len(evidence["rollback_exceptions"]) == 1
    assert "rollback_reset_lmcache_remove_servers" not in evidence
    assert "start_canonical" not in evidence


@pytest.mark.parametrize(
    "failed_reset_phase",
    (
        "rollback_reset_lmcache_remove_servers",
        "rollback_reset_lmcache_start_servers",
        "rollback_reset_lmcache_server_health",
    ),
)
def test_rollback_cache_reset_failure_blocks_dependent_restore_phases(
    tmp_path, failed_reset_phase
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "a-mtp0-apc0-lmcache0"
    diagnostic = attribution.derive_profile(canonical, arm)
    all_phases = attribution.phases(site, canonical, diagnostic, arm)
    selected = attribution.sequence("rollback")
    calls = []

    def executor(actions, timeout):
        del timeout
        name = phase_name(actions)
        calls.append(name)
        return phase_result(failed=name == failed_reset_phase, partial=True)

    original = attribution.exl3.execute
    attribution.exl3.execute = executor
    try:
        evidence, restored = attribution.attempt_automatic_rollback(
            selected, all_phases, 1
        )
    finally:
        attribution.exl3.execute = original
    failure_index = selected.index(failed_reset_phase)
    assert calls == selected[: failure_index + 1]
    assert restored is False
    assert failed_reset_phase in evidence
    assert "start_canonical" not in evidence


@pytest.mark.parametrize(
    "failed_cleanup",
    (
        "remove_source_diagnostic",
        "remove_target_diagnostic",
        "rollback_remove_canonical_engines",
    ),
)
def test_transition_rollback_cleanup_dependency_blocks_destructive_followups(
    tmp_path, failed_cleanup
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    source_arm = "a-mtp0-apc0-lmcache0"
    target_arm = "b-mtp2-apc0-lmcache0"
    source = attribution.derive_profile(canonical, source_arm)
    target = attribution.derive_profile(canonical, target_arm)
    all_phases = attribution.transition_phases(
        site, canonical, source, source_arm, target, target_arm
    )
    selected = attribution.transition_rollback_sequence()
    calls = []

    def executor(actions, timeout):
        del timeout
        name = transition_phase_name(actions)
        calls.append(name)
        return phase_result(failed=name == failed_cleanup, partial=True)

    original = attribution.exl3.execute
    attribution.exl3.execute = executor
    try:
        evidence, restored = attribution.attempt_automatic_rollback(
            selected, all_phases, 1
        )
    finally:
        attribution.exl3.execute = original
    failure_index = selected.index(failed_cleanup)
    assert calls == selected[: failure_index + 1]
    assert restored is False
    assert "rollback_reset_lmcache_remove_servers" not in evidence
    assert "start_canonical" not in evidence


def test_podman_profile_fails_closed_before_attribution_plan_or_execution(
    tmp_path, capsys
):
    site_path, profile_path = generated(tmp_path)
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    document["engine"] = "podman"
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exl3.ProfileError, match="currently require engine=docker"):
        attribution.derive_profile(exl3.Profile(document), "a-mtp0-apc0-lmcache0")
    with pytest.raises(SystemExit, match="2"):
        attribution.main(execution_argv(site_path, profile_path, command="plan"))
    captured = capsys.readouterr()
    assert "currently require engine=docker" in captured.err
    assert captured.out == ""


def test_programmatic_start_actions_rejects_podman_before_composition(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    diagnostic = attribution.derive_profile(
        canonical, "a-mtp0-apc0-lmcache0"
    )
    podman_diagnostic = with_engine(diagnostic, "podman")
    with pytest.raises(exl3.ProfileError, match="currently require engine=docker"):
        attribution.start_actions(
            site, podman_diagnostic, "a-mtp0-apc0-lmcache0"
        )


@pytest.mark.parametrize(
    "builder",
    [
        lambda site, profile, arm: attribution.no_other_diagnostic_actions(
            site, profile
        ),
        lambda site, profile, arm: attribution.exclusive_engine_actions(
            site, profile, arm_id=arm
        ),
        lambda site, profile, arm: attribution.exclusive_diagnostic_actions(
            site, profile, arm
        ),
        lambda site, profile, arm: attribution.diagnostic_remove_actions(
            site, profile, arm
        ),
        lambda site, profile, arm: attribution.ready_actions(
            site, profile, arm, wait=False
        ),
        lambda site, profile, arm: attribution.canonical_restore_actions(
            site, profile
        ),
    ],
    ids=[
        "no-other-diagnostic",
        "exclusive-engine",
        "exclusive-diagnostic",
        "diagnostic-remove",
        "ready",
        "canonical-restore",
    ],
)
def test_every_programmatic_action_builder_rejects_podman(
    tmp_path, builder
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    diagnostic = attribution.derive_profile(
        canonical, "a-mtp0-apc0-lmcache0"
    )
    podman_diagnostic = with_engine(diagnostic, "podman")
    with pytest.raises(exl3.ProfileError, match="currently require engine=docker"):
        builder(site, podman_diagnostic, "a-mtp0-apc0-lmcache0")


def test_programmatic_phase_builder_rejects_mixed_container_engines(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    diagnostic = attribution.derive_profile(
        canonical, "a-mtp0-apc0-lmcache0"
    )
    podman_diagnostic = with_engine(diagnostic, "podman")
    with pytest.raises(exl3.ProfileError, match="same container engine"):
        attribution.phases(
            site,
            canonical,
            podman_diagnostic,
            "a-mtp0-apc0-lmcache0",
        )


def test_programmatic_transition_builder_rejects_mixed_container_engines(
    tmp_path,
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    source = attribution.derive_profile(
        canonical, "a-mtp0-apc0-lmcache0"
    )
    target = attribution.derive_profile(
        canonical, "b-mtp2-apc0-lmcache0"
    )
    podman_source = with_engine(source, "podman")
    with pytest.raises(exl3.ProfileError, match="same container engine"):
        attribution.transition_phases(
            site,
            canonical,
            podman_source,
            "a-mtp0-apc0-lmcache0",
            target,
            "b-mtp2-apc0-lmcache0",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda argv: argv.append("--unknown-functional-option"),
        lambda argv: argv.extend(["--no-enable-prefix-caching"]),
        lambda argv: argv.extend(["--enable-prefix-caching"]),
        lambda argv: argv.extend(["--max-num-seqs", "99"]),
    ],
)
def test_exact_config_cmd_attestation_rejects_extra_opposite_or_duplicate_args(
    tmp_path, mutate
):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "c-mtp2-apc1-lmcache0"
    profile = attribution.derive_profile(canonical, arm)
    expected = attribution.expected_config_cmds(site, profile, arm)
    for rank, command in expected.items():
        assert attribution.config_cmd_matches(json.dumps(command), command)
        changed = list(command)
        mutate(changed)
        assert not attribution.config_cmd_matches(json.dumps(changed), command)
        if rank == site.serving.master_rank:
            assert "--headless" not in command
        else:
            assert command[-1] == "--headless"


def test_exact_config_cmd_attestation_rejects_rank_headless_drift(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "d-mtp2-apc1-lmcache1"
    profile = attribution.derive_profile(canonical, arm)
    expected = attribution.expected_config_cmds(site, profile, arm)
    master = site.serving.master_rank
    forged_master = [*expected[master], "--headless"]
    assert not attribution.config_cmd_matches(
        json.dumps(forged_master), expected[master]
    )
    headless_rank = next(rank for rank in expected if rank != master)
    forged_worker = expected[headless_rank][:-1]
    assert not attribution.config_cmd_matches(
        json.dumps(forged_worker), expected[headless_rank]
    )


def test_ready_embeds_the_complete_rank_specific_config_cmd(tmp_path):
    site_path, profile_path = generated(tmp_path)
    site = load_site(site_path)
    canonical = exl3.load_profile(profile_path)
    arm = "d-mtp2-apc1-lmcache1"
    profile = attribution.derive_profile(canonical, arm)
    expected = attribution.expected_config_cmds(site, profile, arm)
    ready = attribution.ready_actions(site, profile, arm)
    for action in ready:
        compact = json.dumps(expected[action.rank], separators=(",", ":"))
        assert compact in action.argv[-1]
        assert "a==e" in action.argv[-1]


@pytest.mark.parametrize(
    "args",
    [
        ["--other", "value"],
        ["--speculative-config", "{}", "--speculative-config", "{}"],
        ["--speculative-config", "--next-option"],
    ],
)
def test_remove_option_rejects_missing_duplicate_or_valueless_option(args):
    with pytest.raises(exl3.ProfileError):
        attribution._remove_option(args, "--speculative-config")


@pytest.mark.parametrize(
    "replacement",
    [
        "{not-json",
        json.dumps({"method": "mtp", "num_speculative_tokens": 3}),
    ],
)
def test_derive_profile_rejects_malformed_or_drifted_spec_config(
    tmp_path, replacement
):
    _, profile_path = generated(tmp_path)
    canonical = exl3.load_profile(profile_path)
    document = copy.deepcopy(canonical.document)
    index = document["extra_vllm_args"].index("--speculative-config")
    document["extra_vllm_args"][index + 1] = replacement
    with pytest.raises(exl3.ProfileError):
        attribution.derive_profile(
            exl3.Profile(document), "b-mtp2-apc0-lmcache0"
        )


def test_derive_profile_rejects_duplicate_spec_option(tmp_path):
    _, profile_path = generated(tmp_path)
    canonical = exl3.load_profile(profile_path)
    document = copy.deepcopy(canonical.document)
    index = document["extra_vllm_args"].index("--speculative-config")
    document["extra_vllm_args"] += document["extra_vllm_args"][index : index + 2]
    with pytest.raises(exl3.ProfileError):
        attribution.derive_profile(
            exl3.Profile(document), "b-mtp2-apc0-lmcache0"
        )


def test_profile_derivation_fails_closed_on_published_cache_geometry_drift(
    tmp_path, monkeypatch
):
    _, profile_path = generated(tmp_path)
    canonical = exl3.load_profile(profile_path)
    config = attribution.lmcache.recipe_lmcache()
    config["chunk_size"] = 1024
    monkeypatch.setattr(attribution.lmcache, "recipe_lmcache", lambda: config)
    with pytest.raises(exl3.ProfileError, match="boundary geometry drifted"):
        attribution.derive_profile(canonical, "d-mtp2-apc1-lmcache1")
