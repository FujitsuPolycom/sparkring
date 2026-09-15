"""Offline build-source integrity and receipt ownership contracts."""
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
GLM = ("glm53-flash", "glm53-flash-e10536a", "glm53-flash-b12x-kda-adaptive-mtp")


def load(runtime, filename):
    spec = importlib.util.spec_from_file_location(runtime.replace("-", "_") + filename, ROOT / "runtime" / runtime / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkout(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "--quiet")
    git("config", "user.name", "Offline fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "core.autocrlf", "false")
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    git("add", "module.py")
    git("commit", "--quiet", "-m", "fixture")
    return git, source, git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")


@pytest.mark.parametrize("runtime", (*GLM, "qwen38"))
@pytest.mark.parametrize("change", ["clean", "staged", "unstaged"])
def test_source_checkout_matches_committed_index_and_worktree(runtime, change, tmp_path):
    prepare = load(runtime, "prepare_context.py")
    git, source, commit, tree = checkout(tmp_path)
    if change != "clean":
        source.write_text("value = 2\n")
        if change == "staged":
            git("add", "module.py")
    if change == "clean":
        prepare.verify_git_tree(tmp_path, expected_commit=commit, expected_tree=tree)
    else:
        with pytest.raises(prepare.PrepareError):
            prepare.verify_git_tree(tmp_path, expected_commit=commit, expected_tree=tree)


@pytest.mark.parametrize("runtime", GLM)
def test_indexed_patch_tree_is_permitted_but_must_match(runtime, tmp_path):
    prepare = load(runtime, "prepare_context.py")
    git, source, commit, base_tree = checkout(tmp_path)
    source.write_text("value = 2\n")
    git("add", "module.py")
    patched_tree = git("write-tree")
    prepare.verify_git_tree(tmp_path, expected_commit=commit, expected_tree=patched_tree, indexed=True)
    with pytest.raises(prepare.PrepareError):
        prepare.verify_git_tree(tmp_path, expected_commit=commit, expected_tree=base_tree, indexed=True)


@pytest.mark.parametrize("runtime", GLM)
@pytest.mark.parametrize("race", [False, True])
def test_image_receipt_cannot_overwrite_existing_output(runtime, race, tmp_path, monkeypatch):
    verify = load(runtime, "verify_image.py")
    output = tmp_path / "receipt.json"
    if not race:
        output.write_text("previous evidence")
    calls = []
    def fake_verify(*args):
        calls.append(args)
        if race:
            output.write_text("previous evidence")
        return {"schema": "fixture"}
    monkeypatch.setattr(verify, "verify_image", fake_verify)
    monkeypatch.setattr(sys, "argv", ["verify_image.py", "--image", "unused", "--output", str(output)])
    with pytest.raises(SystemExit):
        verify.main()
    assert output.read_text() == "previous evidence"
    assert len(calls) == (1 if race else 0)


@pytest.mark.parametrize("runtime", GLM)
def test_image_receipt_can_create_a_new_output(runtime, tmp_path, monkeypatch):
    verify = load(runtime, "verify_image.py")
    output = tmp_path / "receipts" / "new.json"
    monkeypatch.setattr(verify, "verify_image", lambda *args: {"schema": "fixture"})
    monkeypatch.setattr(sys, "argv", ["verify_image.py", "--image", "unused", "--output", str(output)])
    assert verify.main() == 0
    assert json.loads(output.read_text()) == {"schema": "fixture"}


@pytest.mark.parametrize("runtime", GLM)
def test_builder_default_receipt_refuses_existing_before_build(runtime, tmp_path):
    output = tmp_path / (runtime + "-image-receipt.json")
    output.write_text("prior evidence")
    script = ROOT / "runtime" / runtime / "build-image.sh"

    def shell_path(path):
        if os.name != "nt":
            return str(path)
        drive, tail = os.path.splitdrive(str(path))
        return "/mnt/" + drive[0].lower() + "/" + tail.lstrip("\\/").replace("\\", "/")

    command = "cd " + shlex.quote(shell_path(tmp_path)) + " && BUILD_RECEIPT= CONTAINER_ENGINE=/nonexistent-engine bash " + shlex.quote(shell_path(script))
    result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=20)
    assert result.returncode == 78, result.stderr
    assert "receipt output already exists" in result.stderr
    assert output.read_text() == "prior evidence"
