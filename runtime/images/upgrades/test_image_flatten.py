"""Pullable image flattening preserves admitted runtime behavior."""

import json
import subprocess

import pytest

from . import image_flatten as module


def config():
    return {
        "Env": ["PATH=/opt/venv/bin", "CUDA_ARCH_LIST=8.0 12.0"],
        "Labels": {"org.sparkring.release": "fixture"},
        "Entrypoint": ["/opt/venv/bin/python", "serve.py"],
        "Cmd": ["verify"],
        "WorkingDir": "/workspace",
        "User": "1000:1000",
        "StopSignal": "SIGTERM",
        "ExposedPorts": {"8000/tcp": {}},
        "Healthcheck": None,
        "OnBuild": None,
        "Shell": None,
        "Volumes": None,
    }


def test_configuration_changes_quote_spaces_and_preserve_runtime_fields():
    value = config()
    changes = module.configuration_changes(value)
    assert 'ENV CUDA_ARCH_LIST="8.0 12.0"' in changes
    assert 'ENTRYPOINT ["/opt/venv/bin/python","serve.py"]' in changes
    assert "EXPOSE 8000/tcp" in changes
    assert module.runtime_configuration(value) == module.runtime_configuration(
        json.loads(json.dumps(value))
    )


@pytest.mark.parametrize("field", ["Healthcheck", "OnBuild", "Shell", "Volumes"])
def test_configuration_changes_reject_unsupported_image_semantics(field):
    value = config()
    value[field] = {"owned": True}
    with pytest.raises(ValueError, match="does not support Config." + field):
        module.configuration_changes(value)


def test_main_refuses_without_explicit_execution(tmp_path):
    with pytest.raises(ValueError, match="explicit execution"):
        module.main(
            [
                "--source-image",
                "sha256:" + "a" * 64,
                "--target-tag",
                "sparkring:flattened-fixture",
                "--container",
                "flatten-fixture",
                "--output",
                str(tmp_path / "proof"),
            ]
        )


def test_main_records_one_layer_and_identical_installed_verification(tmp_path, monkeypatch):
    source = "sha256:" + "a" * 64
    target_id = "sha256:" + "b" * 64
    value = config()
    before = {
        "Id": source,
        "Os": "linux",
        "Architecture": "arm64",
        "Size": 1024,
        "Config": value,
        "RootFS": {"Layers": ["one", "two"]},
    }
    after = {
        "Id": target_id,
        "Os": "linux",
        "Architecture": "arm64",
        "Config": value,
        "RootFS": {"Layers": ["flat"]},
    }
    calls = []

    def inspect(reference):
        if reference == source:
            return before
        if reference == "sparkring:flattened-fixture" and "flatten" not in calls:
            raise subprocess.CalledProcessError(1, ["docker", "inspect"])
        return after

    monkeypatch.setattr(module, "inspect", inspect)
    monkeypatch.setattr(module, "docker_root_free_bytes", lambda: module.HEADROOM + 2048)
    monkeypatch.setattr(module, "installed_verification", lambda _: {"schema": "sparkring-native-verification/v1", "files_verified": 7})
    monkeypatch.setattr(
        module,
        "output",
        lambda *args: "" if args[0] == "ps" else "c" * 64,
    )
    monkeypatch.setattr(module, "stream_flatten", lambda *args: calls.append("flatten"))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    output = tmp_path / "proof"
    module.main(
        [
            "--source-image",
            source,
            "--target-tag",
            "sparkring:flattened-fixture",
            "--container",
            "flatten-fixture",
            "--output",
            str(output),
            "--execute",
        ]
    )
    proof = json.loads((output / "flatten.json").read_text())
    assert proof["source_rootfs_layers"] == 2
    assert proof["target_rootfs_layers"] == 1
    assert proof["runtime_configuration_equal"] is True
    assert proof["external_publication"] is False


def test_main_refuses_insufficient_docker_headroom(tmp_path, monkeypatch):
    source = "sha256:" + "a" * 64

    def inspect(reference):
        if reference != source:
            raise subprocess.CalledProcessError(1, ["docker", "inspect"])
        return {
            "Id": source,
            "Os": "linux",
            "Architecture": "arm64",
            "Size": 1024,
            "Config": config(),
            "RootFS": {"Layers": ["one"]},
        }

    monkeypatch.setattr(
        module,
        "inspect",
        inspect,
    )
    monkeypatch.setattr(module, "output", lambda *args: "")
    monkeypatch.setattr(module, "docker_root_free_bytes", lambda: module.HEADROOM)
    with pytest.raises(ValueError, match="headroom"):
        module.main(
            [
                "--source-image",
                source,
                "--target-tag",
                "sparkring:flattened-fixture",
                "--container",
                "flatten-fixture",
                "--output",
                str(tmp_path / "proof"),
                "--execute",
            ]
        )
