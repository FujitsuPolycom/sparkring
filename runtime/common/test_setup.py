"""Exercise installation identity selection and storage decisions without hosts."""
from types import SimpleNamespace
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

from runtime.common import profiles, setup
from scripts import sparkring

GLM = "glm53-flash-spark-tp2-dcp1-sparkcache"
QWEN = "qwen38-flash-next-tp2"


@pytest.mark.parametrize("profile", [
    GLM, "glm53-flash-spark-tp4-dcp1-sparkcache", QWEN,
    "qwen38-flash-next-tp2-sparkcache", "qwen38-flash-next-qad-tp4",
    "qwen38-flash-next-qad-tp4-sparkcache",
])
def test_selection_uses_profile_publication_and_checkpoint(profile, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Selection must not contact Docker, SSH or the network")
    monkeypatch.setattr(subprocess, "run", forbidden)
    card = setup.selection(profile)
    resolved = profiles.resolve(profile)
    assert card["image_reference"] == resolved["release"]["image"]
    assert card["model_revision"] == resolved["model"]["revision"]
    assert card["nodes"] == resolved["serving"]["node_count"]
    assert card["sparkcache"] == resolved["serving"]["sparkcache"]
    # Installer profiles run on the installer image; the others on shared-2026.09.3.
    installer_profiles = ("qwen38-flash-next-tp2", "qwen38-flash-next-qad-tp4")
    assert card["image_id"].startswith("sha256:5ce6ce267d80" if profile in installer_profiles else "sha256:bc16a981")


def test_qad_selection_changes_weights_not_the_image():
    default = setup.selection(GLM)
    qad = setup.selection(GLM, "nvfp4-qad")
    assert qad["model_repository"] != default["model_repository"]
    assert qad["model_revision"] != default["model_revision"]
    assert qad["image_id"] == default["image_id"]
    with pytest.raises(ValueError, match="does not accept"):
        setup.selection(QWEN, "nvfp4-qad")


def test_selection_rejects_unknown_profiles_and_missing_publication():
    with pytest.raises(ValueError, match="Unknown profile"):
        setup.selection("made-up")
    with pytest.raises(ValueError, match="pinned publication"):
        setup.selection("deepseek-v41-flash-sglang-cycle")


def test_selection_refuses_mismatched_publication(monkeypatch):
    original = profiles.read_json

    def read(path):
        result = original(path)
        if Path(path).name == "publication.json":
            result["image_reference"] = "different-image"
        return result

    monkeypatch.setattr(profiles, "read_json", read)
    with pytest.raises(ValueError, match="does not match"):
        setup.selection(GLM)


def test_cli_is_available_through_operator_command(capsys):
    assert sparkring.main(["setup", "show", GLM, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == GLM
    assert sparkring.main(["setup", "show", "missing"]) == 2
    assert "Unknown profile" in capsys.readouterr().err


def test_shell_output_cannot_execute_values(tmp_path):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash required")
    card = setup.selection(GLM)
    payload = "$(touch SHOULD_NOT_EXIST); 'quoted' `touch ALSO_NOT_CREATED`"
    card["model_repository"] = payload
    result = subprocess.run([bash, "--noprofile", "--norc"], cwd=tmp_path,
                            input=setup.shell_selection(card) + '\nprintf "%s" "$MODEL_REPO"\n',
                            text=True, capture_output=True, timeout=10, check=True)
    assert result.stdout == payload
    assert list(tmp_path.iterdir()) == []


def plan(tmp_path, **kwargs):
    defaults = dict(model_path=tmp_path / "models/new", cache_path=tmp_path / "cache",
                    docker_path=tmp_path / "docker", device_id=lambda path: 1,
                    disk_usage=lambda path: SimpleNamespace(free=250 * setup.GIB))
    defaults.update(kwargs)
    return setup.storage_plan(setup.selection(GLM), **defaults)


def test_shared_filesystem_sums_requirements_instead_of_passing_each_separately(tmp_path):
    report = plan(tmp_path)
    assert not report["passed"]
    assert len(report["filesystems"]) == 1
    assert report["filesystems"][0]["required_bytes"] == 300 * setup.GIB
    assert list(tmp_path.iterdir()) == []


def test_split_filesystems_are_checked_independently(tmp_path):
    for name in ("models", "cache", "docker"):
        (tmp_path / name).mkdir()
    report = plan(tmp_path, device_id=lambda path: path.name)
    assert report["passed"]
    assert sorted(group["required_bytes"] // setup.GIB for group in report["filesystems"]) == [32, 68, 200]
    failure = plan(tmp_path, device_id=lambda path: path.name,
                   disk_usage=lambda path: SimpleNamespace(free=(67 if path.name == "docker" else 250) * setup.GIB))
    assert not failure["passed"]
    assert [g["probe_path"] for g in failure["filesystems"] if not g["passed"]] == [str(tmp_path / "docker")]


def test_reuse_keeps_cache_headroom_and_requires_a_model_directory(tmp_path):
    with pytest.raises(ValueError, match="existing nonempty"):
        plan(tmp_path, reuse_model=True)
    model = tmp_path / "models/new"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    report = plan(tmp_path, reuse_model=True, reuse_image=True)
    assert report["passed"]
    assert report["filesystems"][0]["required_bytes"] == 32 * setup.GIB
    assert "not a measured minimum or asset verification" in report["scope"]


@pytest.mark.parametrize("cache", ["models", "models/new", "models/new/cache"])
def test_storage_refuses_nested_model_and_cache(tmp_path, cache):
    with pytest.raises(ValueError, match="non-nested"):
        plan(tmp_path, cache_path=tmp_path / cache)


def test_storage_rejects_relative_paths_and_file_ancestors(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        plan(tmp_path, model_path="relative")
    path = tmp_path / "file"
    path.write_text("not a directory")
    with pytest.raises(ValueError, match="not a directory"):
        plan(tmp_path, model_path=path / "child")


def test_qwen_budget_and_cli_exit_on_insufficient_space(tmp_path, monkeypatch, capsys):
    card = setup.selection(QWEN)
    report = setup.storage_plan(card, model_path=tmp_path / "model", cache_path=tmp_path / "cache",
                                docker_path=tmp_path / "docker", device_id=lambda path: 1,
                                disk_usage=lambda path: SimpleNamespace(free=219 * setup.GIB))
    assert report["filesystems"][0]["required_bytes"] == 220 * setup.GIB
    monkeypatch.setattr(setup, "storage_plan", lambda *args, **kwargs: report)
    result = setup.main(["storage", QWEN, "--model-path", str(tmp_path / "model"),
                         "--cache-path", str(tmp_path / "cache"), "--docker-path", str(tmp_path / "docker"), "--json"])
    assert result == 1
    assert not json.loads(capsys.readouterr().out)["passed"]
