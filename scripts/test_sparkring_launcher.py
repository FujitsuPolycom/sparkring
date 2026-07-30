"""Offline tests for the public four-rank launcher."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import sparkring_launcher as launcher
from sparkring_site import load_site

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts/config/site.example.yaml"
LAUNCH = ROOT / "scripts/config/launch.example.json"
GLM52_INDEX_TOPK_PATTERN = (
    "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
    "FSSSFSSSFSSS"
)


def _option_value(arguments: tuple[str, ...], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]


def _environment_value(action: launcher.RemoteAction, name: str) -> str:
    prefix = f"{name}="
    return next(value.removeprefix(prefix) for value in action.argv if value.startswith(prefix))


def _launch_config_with(
    tmp_path: Path,
    mutate,
) -> launcher.LaunchConfig:
    document = json.loads(LAUNCH.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return launcher.load_launch(path)


def test_example_passes_checkpoint_indexer_layout_to_vllm():
    site = load_site(SITE)
    config = launcher.load_launch(LAUNCH)

    for action in launcher.start_actions(site, config):
        overrides = json.loads(_option_value(action.argv, "--hf-overrides"))
        assert overrides == {"index_topk_pattern": GLM52_INDEX_TOPK_PATTERN}
        assert len(overrides["index_topk_pattern"]) == 78
        assert overrides["index_topk_pattern"].count("F") == 21
        assert overrides["index_topk_pattern"].count("S") == 57


def test_pinned_checkpoint_refuses_missing_indexer_layout(tmp_path):
    def remove_hf_overrides(document):
        index = document["extra_vllm_args"].index("--hf-overrides")
        del document["extra_vllm_args"][index:index + 2]

    config = _launch_config_with(tmp_path, remove_hf_overrides)

    with pytest.raises(
        launcher.LaunchConfigError,
        match="requires exactly one --hf-overrides",
    ):
        launcher.start_actions(load_site(SITE), config)


def test_pinned_checkpoint_refuses_wrong_indexer_layout(tmp_path):
    def rotate_indexer_pattern(document):
        index = document["extra_vllm_args"].index("--hf-overrides")
        document["extra_vllm_args"][index + 1] = json.dumps(
            {
                "index_topk_pattern": (
                    GLM52_INDEX_TOPK_PATTERN[1:] + GLM52_INDEX_TOPK_PATTERN[0]
                )
            }
        )

    config = _launch_config_with(tmp_path, rotate_indexer_pattern)

    with pytest.raises(
        launcher.LaunchConfigError,
        match="requires its exact 78-layer index_topk_pattern",
    ):
        launcher.start_actions(load_site(SITE), config)


def test_pinned_nf3_launch_refuses_historical_nvfp4_kv(tmp_path):
    def select_nvfp4(document):
        index = document["extra_vllm_args"].index("--kv-cache-dtype")
        document["extra_vllm_args"][index + 1] = "nvfp4_ds_mla"
        document["environment"]["VLLM_NVFP4_MLA_PER_TOKEN_SCALE"] = "1"

    config = _launch_config_with(tmp_path, select_nvfp4)
    with pytest.raises(
        launcher.LaunchConfigError,
        match="requires exactly one --kv-cache-dtype fp8",
    ):
        launcher.start_actions(load_site(SITE), config)


def test_transport_slots_follow_native_xor_round_schedule():
    site = load_site(SITE)
    config = launcher.load_launch(LAUNCH)

    for action in launcher.start_actions(site, config):
        rank = site.rank(action.rank)
        ports = {port.peer_rank: port for port in rank.ring_ports}
        control_peers = {peer.rank: peer for peer in rank.transport_peers}
        round0_peer = action.rank ^ 1
        round1_peer = action.rank ^ 3

        assert _environment_value(action, "SPARK_TP4_PEER0") == str(
            control_peers[round0_peer].address
        )
        assert _environment_value(action, "SPARK_TP4_PEER1") == str(
            control_peers[round1_peer].address
        )
        assert _environment_value(action, "SPARK_TP4_DEVICE0") == ports[
            round0_peer
        ].rdma_device
        assert _environment_value(action, "SPARK_TP4_DEVICE1") == ports[
            round1_peer
        ].rdma_device
        assert _environment_value(action, "NCCL_IB_GID_INDEX") == str(
            ports[round0_peer].roce_gid_index
        )
        assert _environment_value(action, "NCCL_IB_SUBNET_PREFIX_LEN") == "24"


def test_launch_refuses_site_derived_transport_override(tmp_path):
    def override_gid(document):
        document["environment"]["NCCL_IB_GID_INDEX"] = "7"

    with pytest.raises(
        launcher.LaunchConfigError,
        match="NCCL_IB_GID_INDEX is derived from the validated site",
    ):
        _launch_config_with(tmp_path, override_gid)


def test_pinned_launch_refuses_socket_payload_fallback(tmp_path):
    def select_socket(document):
        document["environment"]["NCCL_NET"] = "Socket"

    config = _launch_config_with(tmp_path, select_socket)
    with pytest.raises(
        launcher.LaunchConfigError,
        match="requires NCCL_NET=IB",
    ):
        launcher.start_actions(load_site(SITE), config)


def test_example_uses_validated_nf3_fp8_kv_contract():
    config = launcher.load_launch(LAUNCH)
    assert _option_value(config.extra_vllm_args, "--kv-cache-dtype") == "fp8"
    assert config.environment["VLLM_SPARK_KV_PROFILE"] == "fp8"
    assert "VLLM_NVFP4_MLA_PER_TOKEN_SCALE" not in config.environment


def test_generated_nvfp4_rope8_profile_is_accepted(tmp_path):
    document = json.loads(LAUNCH.read_text(encoding="utf-8"))
    index = document["extra_vllm_args"].index("--kv-cache-dtype")
    document["extra_vllm_args"][index + 1] = "nvfp4_ds_mla"
    document["environment"].update(
        {
            "VLLM_SPARK_KV_PROFILE": "nvfp4-rope8",
            "VLLM_SPARK_KV_CACHE_DTYPE": "nvfp4_ds_mla",
            "VLLM_NVFP4_MLA_PER_TOKEN_SCALE": "1",
            "VLLM_SPARK_KV_SCALE_MODE": "per-token",
        }
    )
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    config = launcher.load_launch(path)

    assert len(launcher.start_actions(load_site(SITE), config)) == 4


def test_nvfp4_rope8_profile_refuses_missing_per_token_scale(tmp_path):
    document = json.loads(LAUNCH.read_text(encoding="utf-8"))
    index = document["extra_vllm_args"].index("--kv-cache-dtype")
    document["extra_vllm_args"][index + 1] = "nvfp4_ds_mla"
    document["environment"].update(
        {
            "VLLM_SPARK_KV_PROFILE": "nvfp4-rope8",
            "VLLM_SPARK_KV_CACHE_DTYPE": "nvfp4_ds_mla",
            "VLLM_SPARK_KV_SCALE_MODE": "per-token",
        }
    )
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        launcher.LaunchConfigError,
        match="VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1",
    ):
        launcher.start_actions(
            load_site(SITE),
            launcher.load_launch(path),
        )


def test_example_uses_live_nf3_c8_graph_contract():
    config = launcher.load_launch(LAUNCH)
    assert "--enforce-eager" not in config.extra_vllm_args
    assert "--no-enable-flashinfer-autotune" not in config.extra_vllm_args
    assert (
        _option_value(config.extra_vllm_args, "--max-cudagraph-capture-size")
        == "40"
    )
    assert config.environment["VLLM_SPARK_ENABLE_CUDAGRAPH"] == "1"
    assert config.environment["VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE"] == "1"
    assert config.environment["VLLM_SPARK_MAX_QUERY_ROWS"] == "40"


def test_pinned_nf3_launch_refuses_duplicate_flashinfer_autotune_control(
    tmp_path,
):
    config = _launch_config_with(
        tmp_path,
        lambda document: document["extra_vllm_args"].append(
            "--no-enable-flashinfer-autotune"
        ),
    )
    with pytest.raises(
        launcher.LaunchConfigError,
        match="duplicate CLI flag is rejected by vLLM",
    ):
        launcher.start_actions(load_site(SITE), config)


def test_pinned_nf3_launch_refuses_query_capacity_drift(tmp_path):
    config = _launch_config_with(
        tmp_path,
        lambda document: document["environment"].update(
            {"VLLM_SPARK_MAX_QUERY_ROWS": "32"}
        ),
    )
    with pytest.raises(
        launcher.LaunchConfigError,
        match="VLLM_SPARK_MAX_QUERY_ROWS=40",
    ):
        launcher.start_actions(load_site(SITE), config)


def test_example_explicitly_unsets_incompatible_prefix_retention_variable():
    site = load_site(SITE)
    config = launcher.load_launch(LAUNCH)

    for action in launcher.start_actions(site, config):
        bare_unsets = [
            index
            for index, value in enumerate(action.argv[:-1])
            if value == "--env"
            and action.argv[index + 1] == "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"
        ]
        assert len(bare_unsets) == 1
        assert not any(
            value.startswith("VLLM_PREFIX_CACHE_RETENTION_INTERVAL=")
            for value in action.argv
        )


def test_launch_rejects_present_prefix_retention_variable(tmp_path):
    document = json.loads(LAUNCH.read_text(encoding="utf-8"))
    document["environment"]["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"] = "4096"
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        launcher.LaunchConfigError,
        match="VLLM_PREFIX_CACHE_RETENTION_INTERVAL must be null",
    ):
        launcher.start_actions(load_site(SITE), launcher.load_launch(path))


def test_example_enables_glm_tool_and_reasoning_parsers():
    site = load_site(SITE)
    config = launcher.load_launch(LAUNCH)

    for action in launcher.start_actions(site, config):
        assert "--enable-auto-tool-choice" in action.argv
        assert _option_value(action.argv, "--reasoning-parser") == "glm45"
        assert _option_value(action.argv, "--tool-call-parser") == "glm47"


def test_pinned_launch_refuses_missing_auto_tool_choice(tmp_path):
    config = _launch_config_with(
        tmp_path,
        lambda document: document["extra_vllm_args"].remove(
            "--enable-auto-tool-choice"
        ),
    )

    with pytest.raises(
        launcher.LaunchConfigError,
        match="requires exactly one --enable-auto-tool-choice",
    ):
        launcher.start_actions(load_site(SITE), config)


@pytest.mark.parametrize(
    ("option", "expected", "wrong"),
    (
        ("--reasoning-parser", "glm45", "glm47"),
        ("--tool-call-parser", "glm47", "glm45"),
    ),
)
def test_pinned_launch_refuses_wrong_glm_parser(
    tmp_path, option, expected, wrong
):
    def replace_parser(document):
        index = document["extra_vllm_args"].index(option)
        document["extra_vllm_args"][index + 1] = wrong

    config = _launch_config_with(tmp_path, replace_parser)

    with pytest.raises(
        launcher.LaunchConfigError,
        match=rf"requires exactly one {option} {expected}",
    ):
        launcher.start_actions(load_site(SITE), config)


@pytest.mark.parametrize(
    ("option", "expected"),
    (
        ("--reasoning-parser", "glm45"),
        ("--tool-call-parser", "glm47"),
    ),
)
def test_pinned_launch_refuses_missing_glm_parser(tmp_path, option, expected):
    def remove_parser(document):
        index = document["extra_vllm_args"].index(option)
        del document["extra_vllm_args"][index:index + 2]

    config = _launch_config_with(tmp_path, remove_parser)

    with pytest.raises(
        launcher.LaunchConfigError,
        match=rf"requires exactly one {option} {expected}",
    ):
        launcher.start_actions(load_site(SITE), config)


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
        assert "SPARKRING_MTP_DRAFT_PATH=/mtp-draft" in action.argv
        assert "B12X_MLA_SPARSE" in action.argv
        assert "SPARK_ADAPTIVE_MTP_CONTROL=1" in action.argv
        assert "SPARK_GLM52_MTP_INDEX_REUSE=1" in action.argv
        assert "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT=1" in action.argv
        assert (
            f"{site.paths.jit_cache_dir}:/cache/jit"
            in action.argv
        )
        assert (
            f"{config.mtp_draft_host_path}:/mtp-draft:ro"
            in action.argv
        )
        assert "--no-enable-flashinfer-autotune" not in action.argv
        assert action.argv.count("--kernel-config") == 1


def test_example_mounts_nf3_target_and_separate_mtp_draft():
    site = load_site(SITE)
    config = launcher.load_launch(LAUNCH)

    assert "NF3-Hybrid" in config.model_host_path
    assert config.mtp_draft_host_path != config.model_host_path
    for action in launcher.start_actions(site, config):
        assert (
            f"{config.model_host_path}:{site.runtime.model_path}:ro"
            in action.argv
        )
        assert f"{config.mtp_draft_host_path}:/mtp-draft:ro" in action.argv
        speculative = json.loads(
            _option_value(action.argv, "--speculative-config")
        )
        assert speculative["model"] == "/mtp-draft"


def test_launch_rejects_target_and_draft_at_same_host_path(tmp_path):
    def share_path(document):
        document["mtp_draft_host_path"] = document["model_host_path"]

    with pytest.raises(
        launcher.LaunchConfigError,
        match="target and MTP draft host paths must be distinct",
    ):
        _launch_config_with(tmp_path, share_path)


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


def test_site_model_must_match_nf3_recipe(tmp_path):
    text = SITE.read_text(encoding="utf-8").replace(
        "madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid",
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
                    "stdout": "" if action.rank == 2 else "a" * 64 + "\n",
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


def test_run_remote_quotes_entire_shell_payload(monkeypatch):
    action = launcher.RemoteAction(
        rank=0,
        ssh_target="operator@node0",
        argv=("docker", "run", "--detach", "image:tag"),
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "a" * 64 + "\n", "")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.run_remote(action, timeout=10)
    assert captured["argv"][-1] == (
        "sh -lc 'docker run --detach image:tag'"
    )
    assert captured["argv"][-2] == "operator@node0"


def test_start_rejects_docker_help_false_positive():
    assert not launcher.action_succeeded(
        "start",
        {"exit_code": 0, "stdout": "", "stderr": "Usage: docker"},
    )
    assert launcher.action_succeeded(
        "start",
        {"exit_code": 0, "stdout": "a" * 64 + "\n", "stderr": ""},
    )
