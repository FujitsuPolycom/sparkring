"""Standalone sharing keeps profile settings and resolves with real Compose."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from runtime.common import compose, standalone_compose
from scripts import sparkring


@pytest.mark.parametrize("profile", standalone_compose.SUPPORTED)
def test_one_file_has_two_explicit_rank_profiles_and_no_site_secrets(profile):
    text = standalone_compose.render(profile)
    data = yaml.safe_load(text)
    card, expected = standalone_compose.services(profile)
    assert data["services"] == expected
    assert list(data["services"]) == ["rank0", "rank1"]
    for number in (0, 1):
        service = data["services"][f"rank{number}"]
        assert service["profiles"] == [f"rank{number}"]
        assert service["image"] == card["image_reference"]
        assert service["pull_policy"] == "missing"
        assert service["environment"]["VLLM_HOST_IP"].startswith("${SPARKRING_HOST_IP:?")
        assert service["volumes"][0]["read_only"] is True
        assert all(m["bind"]["create_host_path"] is False for m in service["volumes"])
    assert "--headless" not in data["services"]["rank0"]["command"]
    assert "--headless" in data["services"]["rank1"]["command"]
    assert "192.0.2.240" not in text and "192.0.2.241" not in text
    assert "deployment.lock" not in text and "source.bundle" not in text


def test_qad_export_uses_same_image_with_distinct_checkpoint():
    profile = standalone_compose.SUPPORTED[0]
    spark = yaml.safe_load(standalone_compose.render(profile))
    qad = yaml.safe_load(standalone_compose.render(profile, "nvfp4-qad"))
    assert spark["x-runtime"]["image"] == qad["x-runtime"]["image"]
    assert spark["x-sparkring"]["model_revision"] != qad["x-sparkring"]["model_revision"]


def test_export_needs_no_site_initialization_or_host_access(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No external commands"))
    output = tmp_path / "shared.yaml"
    assert sparkring.main(["export", "--format", "compose", "--profile", standalone_compose.SUPPORTED[0],
                           "--output", str(output)]) == 0
    assert len(yaml.safe_load(output.read_text())["services"]) == 2
    assert sparkring.main(["export", "--format", "compose", "--profile", standalone_compose.SUPPORTED[0],
                           "--output", str(output)]) == 2
    capsys.readouterr()


@pytest.mark.parametrize("profile", standalone_compose.SUPPORTED)
@pytest.mark.parametrize("rank", [0, 1])
def test_real_compose_interpolation_matches_the_profile(profile, rank):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose CLI unavailable; config only, no daemon needed")
    card, specs = standalone_compose.specifications(profile)
    env = {key: value for key, value in os.environ.items() if not key.startswith("SPARKRING_")}
    env.update(SPARKRING_MODEL_DIR="/srv/models/operator-chosen", SPARKRING_CACHE_DIR="/srv/cache/operator-chosen",
               SPARKRING_MASTER_ADDR="198.18.20.1", SPARKRING_HOST_IP=f"198.18.20.{rank+1}", SPARKRING_INTERFACE="fabric0")
    project = "sparkring-" + profile
    argv = compose.compose_command(project) + ["--profile", f"rank{rank}", "config", "--format", "json", "--no-path-resolution"]
    actual = json.loads(subprocess.run(argv, input=standalone_compose.render(profile), text=True, encoding="utf-8", capture_output=True,
                                       env=env, check=True, timeout=30).stdout)["services"][f"rank{rank}"]
    expected = compose.service(specs[rank], card["image_reference"])
    expected["profiles"] = [f"rank{rank}"]
    expected["pull_policy"] = "missing"
    expected["environment"]["VLLM_HOST_IP"] = env["SPARKRING_HOST_IP"]
    for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        expected["environment"][key] = env["SPARKRING_INTERFACE"]
    if "MASTER_ADDR" in expected["environment"]:
        expected["environment"]["MASTER_ADDR"] = env["SPARKRING_MASTER_ADDR"]
    command = expected["command"]
    command[command.index("--master-addr") + 1] = env["SPARKRING_MASTER_ADDR"]
    for mount in expected["volumes"]:
        mount["source"] = env["SPARKRING_MODEL_DIR" if mount["read_only"] else "SPARKRING_CACHE_DIR"]
    assert compose.normalize_service(actual) == compose.escape(compose.normalize_service(expected))


def test_generated_standalone_files_are_current():
    root = Path(__file__).resolve().parents[2]
    for profile in standalone_compose.SUPPORTED:
        assert (root / "profiles" / profile / "compose/standalone.yaml").read_text(encoding="utf-8") == standalone_compose.render(profile)
