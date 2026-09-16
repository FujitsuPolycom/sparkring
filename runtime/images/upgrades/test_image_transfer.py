"""Loopback-only image transport, exact identities and owned-resource cleanup."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from runtime.images.upgrades import image_transfer as module
from runtime.images.upgrades.contracts import Refused, Uncertain

IMAGE = "sha256:" + "a" * 64
REGISTRY = "sha256:" + "b" * 64


@pytest.fixture
def config():
    return {
        "schema": "sparkring-image-transfer/v1",
        "hosts": ["u@host0", "u@host1"],
        "hostnames": ["host0", "host1"],
        "gate_id": "image-transfer",
        "fabric_peer": "u@fabric1",
        "host_key_alias": "host1",
        "registry_image_id": REGISTRY,
        "temporary_parent": "/var/tmp/image-transfers",
        "port": 19555,
        "transfer_seconds": 900,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"temporary_parent": "/"},
        {"temporary_parent": "/var/tmp"},
        {"temporary_parent": "/var/tmp/a/../b"},
        {"temporary_parent": "/var/tmp/a,readonly"},
        {"port": 80},
        {"transfer_seconds": 9000},
        {"registry_image_id": "registry:2"},
        {"fabric_peer": "other@fabric1"},
        {"fabric_peer": "-o ProxyCommand=bad"},
        {"host_key_alias": "host -p 4"},
        {"registry_rank": 2},
        {"registry_rank": True},
        {"registry_rank": "0"},
    ],
)
def test_unbounded_or_ambiguous_transfer_configuration_is_rejected(config, change):
    with pytest.raises(Refused):
        module.validate({**config, **change})


def test_registry_is_cpu_only_nonroot_and_exposes_only_loopback(config):
    command = module.registry_command(
        config,
        name="owned",
        path="/var/tmp/image-transfers/owned",
        user="1000:1000",
    )
    assert command[:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
    assert command[command.index("--user") + 1] == "1000:1000"
    assert command[command.index("--runtime") + 1] == "runc"
    assert command[command.index("--publish") + 1] == "127.0.0.1:19555:5000"
    assert "--gpus" not in command
    assert command[command.index("--pull") + 1] == "never"
    assert command[-1] == REGISTRY


@pytest.fixture
def scenario(config, monkeypatch, tmp_path):
    actions = []
    state = {
        "bad_hash": False,
        "bad_owner": False,
        "fail_push": False,
        "create_unknown": False,
        "fabric_host": "host1",
        "container": False,
        "probe_ready": True,
    }
    name = "sr-upgrade-transfer-trial-test"

    def info(image):
        return {"Id": image, "Architecture": "arm64", "Os": "linux"}

    class Pair:
        def call(self, rank, args, **kwargs):
            actions.append((rank, list(args)))
            if args[0] == "ssh":
                return state["fabric_host"].encode()
            if args[0] == "python3":
                assert rank == 1 and config.get("registry_rank") == 0
                return b"200\n" if state["probe_ready"] else b""
            assert args[:3] == module.docker()
            args = args[3:]
            if args[:2] == ["image", "inspect"]:
                selected = args[-1]
                if selected.startswith("127.0.0.1:"):
                    selected = "sha256:" + "c" * 64 if state["bad_hash"] else IMAGE
                return json.dumps([info(selected)]).encode()
            if args[0] == "create":
                state["container"] = True
                if state["create_unknown"]:
                    raise Refused("Create observation lost")
                return b"container-id"
            if args[0] == "inspect":
                owner = (
                    "unrelated"
                    if state.get("finished") and state["bad_owner"]
                    else name
                )
                return json.dumps(
                    [
                        {
                            "Id": "container-id",
                            "Image": REGISTRY,
                            "Config": {"Labels": {"sparkring.upgrade.transfer": owner}},
                        }
                    ]
                ).encode()
            if args[0] == "push" and state["fail_push"]:
                raise Refused("Push failed")
            if args[0] == "pull":
                state["finished"] = True
            if args[0] == "rm":
                state["container"] = False
            if args[0] == "ps":
                return b"container-id" if state["container"] else b""
            return b""

    def directory(pair, cfg, owner, *, remove=False):
        assert owner == name
        actions.append("remove-directory" if remove else "create-directory")
        return {"path": "/var/tmp/image-transfers/" + owner, "user": "1000:1000"}

    class Tunnel:
        def __init__(self, argv, **kwargs):
            assert "127.0.0.1:19555:127.0.0.1:19555" in argv
            assert "HostKeyAlias=host1" in argv
            assert "StrictHostKeyChecking=yes" in argv
            assert ("-R" if config.get("registry_rank", 1) == 0 else "-L") in argv
            actions.append("start-tunnel")

        def poll(self):
            return None

        def terminate(self):
            actions.append("stop-tunnel")

        def wait(self, **kwargs):
            return 0

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(module.socket, "gethostname", lambda: "host0")
    monkeypatch.setattr(module.getpass, "getuser", lambda: "u")
    monkeypatch.setattr(module, "temporary_directory", directory)
    monkeypatch.setattr(module.subprocess, "Popen", Tunnel)
    monkeypatch.setattr(
        module.urllib.request,
        "build_opener",
        lambda *a: SimpleNamespace(open=lambda *a, **k: Response()),
    )
    return (
        lambda: module.transfer(Pair(), config, IMAGE, "trial-test", tmp_path / "out"),
        state,
        actions,
        tmp_path,
    )


def test_success_proves_exact_image_and_removes_only_its_registry(scenario):
    run, _, actions, _ = scenario
    result = run()
    assert result["rank1_verified"] is True
    assert result["external_publication"] is False
    assert not result["cleanup_errors"]
    assert "stop-tunnel" in actions and "remove-directory" in actions
    for action in actions:
        if isinstance(action, tuple) and action[1][3:4] in (["push"], ["pull"]):
            assert action[1][-1].startswith("127.0.0.1:19555/")


def test_sender_registry_uses_reverse_tunnel_and_sender_owned_storage(config, scenario):
    config["registry_rank"] = 0
    run, _, actions, _ = scenario
    result = run()
    assert result["registry_rank"] == 0 and result["rank1_verified"]
    for action in actions:
        if isinstance(action, tuple) and action[1][:3] == module.docker():
            operation = action[1][3]
            if operation in ("create", "start", "stop", "rm", "inspect", "push"):
                assert action[0] == 0
            if operation == "pull":
                assert action[0] == 1
    assert "remove-directory" in actions


def test_sender_registry_requires_receiving_tunnel_readiness(
    config, scenario, monkeypatch
):
    config["registry_rank"] = 0
    run, state, actions, root = scenario
    state["probe_ready"] = False
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(Refused, match="startup deadline"):
        run()
    assert not any(isinstance(a, tuple) and a[1][3:4] == ["push"] for a in actions)
    assert "remove-directory" in actions
    assert not json.loads((root / "out/transfer.json").read_text())["rank1_verified"]


@pytest.mark.parametrize("field", ["bad_hash", "fail_push"])
def test_transfer_failure_cleans_owned_resources_without_qualifying(scenario, field):
    run, state, actions, root = scenario
    state[field] = True
    with pytest.raises(Refused):
        run()
    receipt = json.loads((root / "out/transfer.json").read_text())
    assert receipt["rank1_verified"] is False
    assert receipt["failure"]
    assert "remove-directory" in actions


@pytest.mark.parametrize("field", ["bad_owner", "create_unknown"])
def test_uncertain_container_ownership_preserves_storage_for_inspection(
    scenario, field
):
    run, state, actions, root = scenario
    state[field] = True
    with pytest.raises(Uncertain, match="cleanup needs inspection"):
        run()
    receipt = json.loads((root / "out/transfer.json").read_text())
    assert receipt["cleanup_errors"]
    assert "remove-directory" not in actions


def test_fabric_endpoint_must_be_the_approved_receiving_host(scenario):
    run, state, actions, _ = scenario
    state["fabric_host"] = "not-host1"
    with pytest.raises(Refused, match="Fabric peer differs"):
        run()
    assert "create-directory" not in actions


@pytest.mark.parametrize("wrong_marker", [False, True])
@pytest.mark.parametrize("registry_rank", [0, 1])
def test_staging_filesystem_cleanup_requires_the_exact_owner(
    tmp_path, wrong_marker, registry_rank
):
    class LocalScript:
        def call(self, rank, argv):
            assert rank == registry_rank and argv[:2] == ["python3", "-c"]
            # Exercise the Linux helper's filesystem operations on CPU-only hosts.
            compatibility = "import os; os.getuid=lambda:1000; os.getgid=lambda:1000;\n"
            return subprocess.check_output(
                [sys.executable, "-c", compatibility + argv[2], *argv[3:]],
                stderr=subprocess.PIPE,
            )

    config = {"temporary_parent": str(tmp_path), "registry_rank": registry_rank}
    pair = LocalScript()
    created = module.temporary_directory(pair, config, "owned-test")
    assert created["user"] == "1000:1000"
    path = tmp_path / "owned-test"
    (path / "payload").write_bytes(b"preserve-unless-owned")
    if wrong_marker:
        (path / ".owner").write_text("another-owner")
        with pytest.raises(subprocess.CalledProcessError):
            module.temporary_directory(pair, config, "owned-test", remove=True)
        assert (path / "payload").read_bytes() == b"preserve-unless-owned"
    else:
        module.temporary_directory(pair, config, "owned-test", remove=True)
        assert not path.exists()
