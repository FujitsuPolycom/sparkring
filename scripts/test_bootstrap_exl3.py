"""Offline contracts for the public EXL3 one-command bootstrap."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

import bootstrap_exl3
import sparkring_exl3_launcher
from sparkring_site import load_site


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts/config/site.example.yaml"


def test_plan_is_read_only_and_names_exact_sources():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bootstrap_exl3.py"),
            "plan",
            "--site",
            str(SITE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schema"] == "sparkring-exl3-bootstrap-plan/v1"
    assert plan["model"].startswith("willfalco/GLM-5.2-EXL3-TR3-3.25bpw@")
    assert plan["image"].startswith("sparkring/glm52-exl3-tr3-3.25bpw:")
    assert "direct 200GbE ring" in " ".join(plan["steps"])


def test_execute_requires_explicit_four_rank_confirmation():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bootstrap_exl3.py"),
            "execute",
            "--site",
            str(SITE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert bootstrap_exl3.CONFIRMATION in result.stderr


def test_capacity_command_keeps_posix_model_parent():
    command = bootstrap_exl3.model_capacity_command(
        "/srv/models/GLM-5.2-EXL3-TR3-3.25bpw", 123
    )
    assert "mkdir -p /srv/models" in command
    assert "\\srv\\models" not in command
    assert str(bootstrap_exl3.MODEL_HEADROOM_BYTES) in command


def test_generated_files_encode_the_exact_live_profile(tmp_path):
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "launch.json"
    image = "sparkring/exl3:test"
    image_id = "sha256:" + "a" * 64
    bootstrap_exl3.write_generated_site(SITE, site_path, image, image_id)
    bootstrap_exl3.write_generated_profile(
        profile_path, image, image_id, "/srv/models/exl3", "/srv/jit"
    )
    site = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert site["serving"]["max_num_seqs"] == 8
    assert site["serving"]["max_model_len"] == 1_048_576
    assert site["serving"]["kv_cache_bytes_per_rank"] == 9_000_000_000
    assert profile["image_id"] == image_id
    assert profile["model_shard_count"] == 81
    assert profile["environment"]["VLLM_EXL3_TRELLIS_MAX_M"] == "32"
    assert profile["environment"]["SPARK_CONTEXT_CACHE_ENABLE"] == "0"
    loaded = sparkring_exl3_launcher.load_profile(profile_path)
    assert len(sparkring_exl3_launcher.start_actions(load_site(site_path), loaded)) == 4
    assert sparkring_exl3_launcher.main(
        ["--site", str(site_path), "--profile", str(profile_path), "plan"]
    ) == 0


def test_launcher_refuses_execute_plan(tmp_path):
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "launch.json"
    image_id = "sha256:" + "a" * 64
    bootstrap_exl3.write_generated_site(SITE, site_path, "image:test", image_id)
    bootstrap_exl3.write_generated_profile(
        profile_path, "image:test", image_id, "/srv/models/exl3", "/srv/jit"
    )
    with pytest.raises(SystemExit) as error:
        sparkring_exl3_launcher.main(
            [
                "--site",
                str(site_path),
                "--profile",
                str(profile_path),
                "--execute",
                "plan",
            ]
        )
    assert error.value.code == 2


def test_native_1m_profile_rejects_any_unpublished_environment_drift(tmp_path):
    profile_path = tmp_path / "launch.json"
    image_id = "sha256:" + "a" * 64
    bootstrap_exl3.write_generated_profile(
        profile_path, "image:test", image_id, "/srv/models/exl3", "/srv/jit"
    )
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    document["environment"]["UNPUBLISHED_SWITCH"] = "1"
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(sparkring_exl3_launcher.ProfileError, match="extra"):
        sparkring_exl3_launcher.load_profile(profile_path)


def test_start_actions_verify_model_hashes_and_shard_bytes(tmp_path):
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "launch.json"
    image_id = "sha256:" + "a" * 64
    bootstrap_exl3.write_generated_site(SITE, site_path, "image:test", image_id)
    bootstrap_exl3.write_generated_profile(
        profile_path, "image:test", image_id, "/srv/models/exl3", "/srv/jit"
    )
    actions = sparkring_exl3_launcher.start_actions(
        load_site(site_path), sparkring_exl3_launcher.load_profile(profile_path)
    )
    for action in actions:
        command = action.shell_command
        assert "sha256sum" in command
        assert "tier_bitmap.json" in command
        assert "339069245936" in command


def test_model_fanout_uses_two_direct_neighbors_then_one_relay():
    site = load_site(SITE)
    first_wave, relay = bootstrap_exl3.direct_image_fanout(site)
    assert len(first_wave) == 2
    assert {hop.source_rank for hop in first_wave} == {0}
    assert relay.source_rank in {hop.destination_rank for hop in first_wave}
    for hop in (*first_wave, relay):
        _, management_address = bootstrap_exl3.split_ssh_target(
            site.rank(hop.destination_rank).ssh_target
        )
        target = bootstrap_exl3.direct_rsync_target(
            site, hop.destination_rank, hop.destination_address, "/srv/models/exl3"
        )
        assert hop.destination_address in target
        assert hop.destination_address != management_address


def test_rsync_is_resumable_and_preserves_partial_payload():
    command = bootstrap_exl3.rsync_command("/src/model", "user@host:/dst/model/")
    assert "--partial" in command
    assert "--inplace" in command
    assert "--info=progress2" in command
    assert command[-2:] == ["/src/model/", "user@host:/dst/model/"]


def test_model_download_uses_runtime_python_and_writable_hf_cache(monkeypatch):
    monkeypatch.setattr(bootstrap_exl3, "local_uid_gid", lambda: (1000, 1001))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    command = bootstrap_exl3.container_download_command(
        "sparkring/exl3-base:test", "/srv/models/exl3"
    )
    assert command[:4] == ["docker", "run", "--rm", "--user"]
    assert "1000:1001" in command
    assert "HF_HUB_OFFLINE=0" in command
    assert "HF_HOME=/tmp/sparkring-huggingface" in command
    assert f"{bootstrap_exl3.ROOT}:/opt/sparkring-public:ro" in command
    assert "/srv/models:/srv/models" in command
    assert command[-2:] == ["--model-path", "/srv/models/exl3"]
