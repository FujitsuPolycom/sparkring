"""Offline tests for the one-command NF3 bootstrap."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

import bootstrap_nf3
from sparkring_site import load_site

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
    assert plan["profile"] == "fp8"
    assert [rank["rank"] for rank in plan["ranks"]] == [0, 1, 2, 3]
    assert plan["image"].startswith("sparkring/glm52-nf3:")


def test_plan_selects_nvfp4_rope8_without_changing_model_downloads():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bootstrap_nf3.py"),
            "plan",
            "--site",
            str(SITE),
            "--profile",
            "nvfp4-rope8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["profile"] == "nvfp4-rope8"
    assert plan["image"].startswith("sparkring/gb10-vllm-base:")
    assert plan["model_path"].endswith(
        "GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
    )
    assert plan["draft_path"].endswith("GLM-5.2-NF3-MTP-Draft")
    assert "build the thin NVFP4-latent/FP8-RoPE compatibility layer" in (
        plan["steps"]
    )


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


def test_bootstrap_verifies_rank0_management_fanout_scope():
    command = bootstrap_nf3.ssh_bootstrap_verification_command(SITE)
    assert command[-4:] == [
        "--site",
        str(SITE),
        "--scope",
        "bootstrap",
    ]


def test_bootstrap_verifies_exact_direct_ring_image_tree_scope():
    command = bootstrap_nf3.ssh_image_fanout_verification_command(SITE)
    assert command[-4:] == [
        "--site",
        str(SITE),
        "--scope",
        "image-fanout",
    ]


def test_bootstrap_runs_early_fabric_preflight_without_deployment_gates():
    command = bootstrap_nf3.early_fabric_preflight_command(SITE)
    assert command[-5:] == [
        "--site",
        str(SITE),
        "--scope",
        "fabric",
        "--no-evidence",
    ]


def test_nf3_contract_pins_exact_target_and_mtp_draft():
    recipe = bootstrap_nf3.load_nf3_contract()
    model = recipe["model"]
    draft = model["mtp_draft"]
    assert model["repository"] == (
        "madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
    )
    assert model["revision"] == "66f3623dd8fefb5ca8046706912d5d31c8d196af"
    assert model["index_sha256"] == (
        "6eb773222d932418dd0530c63aca498f86ef424da2a4526ccba76b59726da234"
    )
    assert draft["repository"] == "aidendle94/GLM-5.2-MXFP4-Experts-GPTQ"
    assert draft["revision"] == "46537e0e16fcd156627800139b41b9c497fc7ee2"
    assert draft["weight_sha256"] == (
        "0ade0e3da08e7e6c7b1f20e4c4e8d5d3b26b81103cea22f2ead9909c7d3d0732"
    )


def test_generated_site_replaces_stale_identity_and_kv_contract(tmp_path):
    stale = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    stale["runtime"].update(
        {
            "model_path": "/models/old-checkpoint",
            "model_repo": "old-owner/old-checkpoint",
            "model_revision": "0" * 40,
            "checkpoint_sha256": "f" * 64,
        }
    )
    stale["serving"]["kv_cache_bytes_per_rank"] = 3_000_000_000
    source = tmp_path / "stale-site.yaml"
    source.write_text(yaml.safe_dump(stale), encoding="utf-8")

    recipe = json.loads(
        bootstrap_nf3.RECIPE_PATH.read_text(encoding="utf-8")
    )
    model = recipe["model"]
    serving = recipe["serving"]
    digest = "sha256:" + "a" * 64

    for profile in bootstrap_nf3.PROFILES:
        destination = tmp_path / f"site-{profile}.yaml"
        bootstrap_nf3.write_generated_site(
            source,
            destination,
            f"sparkring/glm52-nf3:{profile}",
            digest,
            profile,
        )
        document = yaml.safe_load(destination.read_text(encoding="utf-8"))
        runtime = document["runtime"]
        assert runtime["container_image"] == (
            f"sparkring/glm52-nf3:{profile}"
        )
        assert runtime["container_image_digest"] == digest
        assert runtime["model_path"] == f"/models/{model['install_subdir']}"
        assert runtime["model_repo"] == model["repository"]
        assert runtime["model_revision"] == model["revision"]
        assert runtime["checkpoint_sha256"] == model["index_sha256"]
        assert serving["kv_cache_bytes_per_rank"] == 7_000_000_000
        assert (
            document["serving"]["kv_cache_bytes_per_rank"]
            == serving["kv_cache_bytes_per_rank"]
        )


def test_generated_nvfp4_rope8_launch_is_an_exact_profile(tmp_path):
    source = ROOT / "scripts/config/launch.example.json"
    destination = tmp_path / "launch.json"
    bootstrap_nf3.write_generated_launch(
        source,
        destination,
        "nvfp4-rope8",
    )
    document = json.loads(destination.read_text(encoding="utf-8"))
    index = document["extra_vllm_args"].index("--kv-cache-dtype")
    assert document["extra_vllm_args"][index + 1] == "nvfp4_ds_mla"
    assert document["environment"]["VLLM_SPARK_KV_PROFILE"] == (
        "nvfp4-rope8"
    )
    assert document["environment"]["VLLM_SPARK_KV_CACHE_DTYPE"] == (
        "nvfp4_ds_mla"
    )
    assert document["environment"]["VLLM_NVFP4_MLA_PER_TOKEN_SCALE"] == "1"
    assert document["environment"]["VLLM_SPARK_KV_SCALE_MODE"] == "per-token"
    assert "--no-enable-flashinfer-autotune" not in document["extra_vllm_args"]


def test_generated_fp8_launch_removes_nvfp4_only_controls(tmp_path):
    source = ROOT / "scripts/config/launch.example.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["environment"]["VLLM_NVFP4_MLA_PER_TOKEN_SCALE"] = "1"
    dirty_source = tmp_path / "source.json"
    dirty_source.write_text(json.dumps(document), encoding="utf-8")
    destination = tmp_path / "launch.json"

    bootstrap_nf3.write_generated_launch(dirty_source, destination, "fp8")

    generated = json.loads(destination.read_text(encoding="utf-8"))
    assert generated["environment"]["VLLM_SPARK_KV_PROFILE"] == "fp8"
    assert "VLLM_NVFP4_MLA_PER_TOKEN_SCALE" not in generated["environment"]


def test_image_transfer_capacity_covers_archive_import_and_headroom():
    archive_bytes = 24 * 1024**3
    required = bootstrap_nf3.required_image_transfer_bytes(archive_bytes)
    assert required == (
        2 * archive_bytes + bootstrap_nf3.IMAGE_TRANSFER_HEADROOM_BYTES
    )
    command = bootstrap_nf3.image_transfer_capacity_command(archive_bytes)
    assert "docker info --format '{{.DockerRootDir}}'" in command
    assert "df -PB1 /var/tmp" in command
    assert 'df -PB1 "$docker_root"' in command
    assert f' -lt {required} ' in command
    assert "IMAGE_TRANSFER_CAPACITY_OK" in command


def test_image_transfer_capacity_rejects_invalid_archive_sizes():
    for invalid in (0, -1, 1.5, "100"):
        try:
            bootstrap_nf3.required_image_transfer_bytes(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid archive size {invalid!r}")


def test_direct_image_fanout_targets_only_ring_addresses():
    site = load_site(SITE)
    first_wave, relay = bootstrap_nf3.direct_image_fanout(site)
    assert [
        (
            hop.source_rank,
            hop.destination_rank,
            bootstrap_nf3.direct_ssh_target(site, hop),
        )
        for hop in first_wave
    ] == [
        (0, 1, "operator@192.0.2.11"),
        (0, 3, "operator@198.18.0.13"),
    ]
    assert (
        relay.source_rank,
        relay.destination_rank,
        bootstrap_nf3.direct_ssh_target(site, relay),
    ) == (1, 2, "operator@198.51.100.12")


def test_remote_archive_path_is_bound_to_exact_image_id():
    expected_id = "sha256:" + "a" * 64
    assert bootstrap_nf3.remote_archive_path(expected_id) == (
        "/var/tmp/sparkring-image-" + "a" * 64 + ".tar"
    )
    for invalid in ("", "sha256:abc", "b" * 64, "sha256:" + "z" * 64):
        try:
            bootstrap_nf3.remote_archive_path(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid image ID {invalid!r}")


def test_load_command_verifies_exact_id_before_exact_temp_cleanup():
    expected_id = "sha256:" + "a" * 64
    archive = bootstrap_nf3.remote_archive_path(expected_id)
    command = bootstrap_nf3.load_archive_command(
        "sparkring/glm52-nf3:test",
        expected_id,
        archive,
        keep_archive=False,
    )
    assert f"docker load --input {archive}" in command
    assert f'test "$observed" = {expected_id}' in command
    assert f"rm -f -- {archive}" in command
    assert "rm -rf" not in command
    assert command.index(f'test "$observed" = {expected_id}') < (
        command.index(f"rm -f -- {archive}")
    )

    retained = bootstrap_nf3.load_archive_command(
        "sparkring/glm52-nf3:test",
        expected_id,
        archive,
        keep_archive=True,
    )
    assert "rm -f" not in retained


def test_fanout_uses_direct_payload_hops_skips_exact_rank_and_attests_all(
    tmp_path,
    monkeypatch,
):
    site = load_site(SITE)
    image = "sparkring/glm52-nf3:test"
    expected_id = "sha256:" + "a" * 64
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"archive")
    by_target = {rank.ssh_target: rank.id for rank in site.ranks}
    ids = {1: "", 2: "", 3: expected_id}
    events: list[tuple[str, object]] = []

    def fake_image_id(target, inspected_image):
        assert inspected_image == image
        return ids[by_target[target]]

    def fake_remote(target, command):
        rank_id = by_target[target]
        events.append(("remote", (rank_id, command)))
        if "docker load --input" in command:
            ids[rank_id] = expected_id

    def fake_run(argv, **kwargs):
        events.append(("run", tuple(argv)))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(bootstrap_nf3, "remote_image_id", fake_image_id)
    monkeypatch.setattr(bootstrap_nf3, "remote", fake_remote)
    monkeypatch.setattr(bootstrap_nf3, "run", fake_run)

    bootstrap_nf3.fanout_image_archive(
        site,
        image,
        expected_id,
        archive,
    )

    local_scps = [
        value for kind, value in events
        if kind == "run" and value[0] == "scp"
    ]
    assert len(local_scps) == 1
    assert local_scps[0][-1].startswith("operator@192.0.2.11:")
    assert all("198.18.1." not in arg for arg in local_scps[0])

    relay_commands = [
        command for kind, value in events
        if kind == "remote"
        for rank_id, command in [value]
        if rank_id == 1 and command.startswith("scp ")
    ]
    assert len(relay_commands) == 1
    assert "operator@198.51.100.12:" in relay_commands[0]
    assert "198.18.1.12" not in relay_commands[0]

    rank3_commands = [
        command for kind, value in events
        if kind == "remote"
        for rank_id, command in [value]
        if rank_id == 3
    ]
    assert rank3_commands == []
    assert ids == {1: expected_id, 2: expected_id, 3: expected_id}

    first_payload = next(
        index for index, event in enumerate(events)
        if (
            event[0] == "run"
            and event[1][0] == "scp"
        )
    )
    capacity_events = [
        index for index, event in enumerate(events)
        if (
            event[0] == "remote"
            and "IMAGE_TRANSFER_CAPACITY_OK" in event[1][1]
        )
    ]
    assert len(capacity_events) == 2
    assert max(capacity_events) < first_payload


def test_exact_relay_stages_archive_when_opposite_rank_is_missing(
    tmp_path,
    monkeypatch,
):
    site = load_site(SITE)
    image = "sparkring/glm52-nf3:test"
    expected_id = "sha256:" + "b" * 64
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"archive")
    by_target = {rank.ssh_target: rank.id for rank in site.ranks}
    ids = {1: expected_id, 2: "", 3: expected_id}
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        bootstrap_nf3,
        "remote_image_id",
        lambda target, _: ids[by_target[target]],
    )

    def fake_remote(target, command):
        rank_id = by_target[target]
        events.append(("remote", (rank_id, command)))
        if "docker load --input" in command:
            ids[rank_id] = expected_id

    def fake_run(argv, **kwargs):
        events.append(("run", tuple(argv)))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(bootstrap_nf3, "remote", fake_remote)
    monkeypatch.setattr(bootstrap_nf3, "run", fake_run)

    bootstrap_nf3.fanout_image_archive(
        site,
        image,
        expected_id,
        archive,
    )

    assert not any(
        kind == "run" and value[0] == "scp"
        for kind, value in events
    )
    rank1_commands = [
        command for kind, value in events
        if kind == "remote"
        for rank_id, command in [value]
        if rank_id == 1
    ]
    assert any("docker save --output" in command for command in rank1_commands)
    assert any(
        command.startswith("scp ")
        and "operator@198.51.100.12:" in command
        for command in rank1_commands
    )
    assert ids[2] == expected_id
