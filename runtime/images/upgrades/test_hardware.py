"""Serving-envelope isolation and ownership without SSH or GPUs."""

import pytest

from .contracts import Refused
from .hardware import Pair, isolated_spec, snapshot_spec


def snapshot():
    return {
        "Id": "a" * 64,
        "Image": "sha256:" + "b" * 64,
        "Name": "/saved-qwen",
        "Config": {
            "Cmd": [
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                "/models/target",
                "--max-model-len",
                "262144",
            ],
            "Entrypoint": ["/opt/venv/bin/python"],
            "Env": ["VLLM_PLUGINS=b12x_loader"],
        },
        "HostConfig": {
            "NetworkMode": "host",
            "IpcMode": "host",
            "Memory": 0,
            "MemorySwap": 0,
            "Init": True,
            "DeviceRequests": [
                {"Count": -1, "DeviceIDs": None, "Capabilities": [["gpu"]]}
            ],
            "Devices": [
                {
                    "PathOnHost": "/dev/infiniband",
                    "PathInContainer": "/dev/infiniband",
                    "CgroupPermissions": "rwm",
                }
            ],
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/models/snapshot",
                "Destination": "/models/target",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": "/cache/serving",
                "Destination": "/cache",
                "RW": True,
            },
        ],
    }


def test_saved_unlimited_memory_is_not_replaced_with_renderer_defaults():
    spec = snapshot_spec(snapshot())
    assert spec.memory is None and spec.memory_swap is None


def test_trial_uses_separate_cache_and_keeps_model_readonly():
    saved = snapshot()
    spec = isolated_spec(
        saved,
        run_id="unit-test",
        role="candidate",
        rank=0,
        cache_mappings={"/cache": "/cache/trial/r0"},
        image_id="sha256:" + "c" * 64,
        native=True,
    )
    assert spec.entrypoint[-1] == "/opt/sparkring/bin/native-image.py"
    assert spec.command == ("serve", "/models/target", "--max-model-len", "262144")
    assert spec.mounts[0].read_only and spec.mounts[0].source == "/models/snapshot"
    assert spec.mounts[1].source == "/cache/trial/r0"
    assert saved["Name"] == "/saved-qwen"


@pytest.mark.parametrize(
    "mapping", [{}, {"/cache": "/cache/serving"}, {"/cache": "/cache/../other"}]
)
def test_trial_refuses_shared_or_escaping_writable_mount(mapping):
    with pytest.raises(Refused):
        isolated_spec(
            snapshot(),
            run_id="unit-test",
            role="candidate",
            rank=0,
            cache_mappings=mapping,
        )


def test_privileged_snapshot_is_not_silently_weakened_or_replayed():
    saved = snapshot()
    saved["HostConfig"]["Privileged"] = True
    with pytest.raises(Refused, match="Privileged"):
        snapshot_spec(saved)


def test_owned_stop_cannot_stop_a_foreign_container(monkeypatch):
    pair = Pair(
        {}, None, ["user@host0", "user@host1"], run_id="unit-test", gate_id="serving"
    )
    monkeypatch.setattr(
        pair,
        "inspect",
        lambda *a: {"Config": {"Labels": {"sparkring.upgrade.run": "other-run"}}},
    )
    with pytest.raises(Refused, match="not owned"):
        pair.stop(0, "foreign")
