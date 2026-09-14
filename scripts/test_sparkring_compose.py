"""Coordinator failure barriers and ownership checks without SSH or containers."""

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from runtime.common import compose
from scripts import deploy_engine, sparkring_compose as coordinator


@pytest.fixture
def deployment():
    site = compose.read_site(
        compose.ROOT / "profiles/qwen38-flash-next-tp2/compose/site.example.yaml"
    )
    return compose.build(compose.SUPPORTED[0], site)


def rank_spec(manifest, number=0):
    spec = compose.specifications(manifest["profile"], manifest["site"])[0][number]
    return replace(
        spec, labels={compose.LABEL: manifest["id"], "io.sparkring.rank": str(number)}
    )


class FakeHosts:
    def __init__(self, fail=None):
        self.events = []
        self.fail = fail

    def __call__(self, host, argv, timeout=120):
        payload = coordinator.unpacked(argv[-2])
        operation = argv[-1]
        event = (host, operation)
        self.events.append(event)
        assert payload["manifest"]["site"]["ranks"][payload["rank"]]["host"] == host
        return {
            "returncode": 1 if event == self.fail else 0,
            "stdout": "ok\n",
            "stderr": "injected host failure" if event == self.fail else "",
            "uncertain": False,
        }


def test_start_barriers_wait_for_every_host(deployment, tmp_path):
    manifest, files = deployment
    plan = coordinator.plan(manifest, files, "start")
    runner = FakeHosts()
    result = deploy_engine.execute_plan(
        plan,
        tmp_path / "receipt.json",
        plan["sha256"],
        runner=runner,
        allow_model_actions=True,
    )
    assert result["complete"]
    operations = [operation for _, operation in runner.events]
    assert max(
        i for i, op in enumerate(operations) if op == "preflight"
    ) < operations.index("stage")
    assert max(
        i for i, op in enumerate(operations) if op == "admitted"
    ) < operations.index("create")
    assert max(
        i for i, op in enumerate(operations) if op == "created"
    ) < operations.index("start-worker")
    assert (
        operations.index("start-worker")
        < operations.index("start-api")
        < operations.index("ready")
    )
    assert ("spark0", "start-worker") not in runner.events
    assert ("spark1", "start-api") not in runner.events


@pytest.mark.parametrize(
    "failure",
    [
        ("spark1", "preflight"),
        ("spark1", "admit"),
        ("spark1", "create"),
        ("spark1", "start-worker"),
    ],
)
def test_failed_rank_blocks_api_and_never_auto_removes(deployment, tmp_path, failure):
    manifest, files = deployment
    plan = coordinator.plan(manifest, files, "start")
    runner = FakeHosts(failure)
    path = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="later phases"):
        deploy_engine.execute_plan(
            plan, path, plan["sha256"], runner=runner, allow_model_actions=True
        )
    assert not any(op in ("start-api", "stop", "down", "rm") for _, op in runner.events)
    receipt = json.loads(path.read_text())
    assert not receipt["complete"]
    if failure[1] != "preflight":
        runner.events.clear()
        with pytest.raises(ValueError, match="uncertain"):
            deploy_engine.execute_plan(
                plan,
                path,
                plan["sha256"],
                runner=runner,
                allow_model_actions=True,
                resume=True,
            )
        assert not any(
            op in ("create", "start-worker", "start-api") for _, op in runner.events
        )


def test_wrong_plan_approval_never_contacts_hosts(deployment, tmp_path):
    manifest, files = deployment
    runner = FakeHosts()
    with pytest.raises(ValueError, match="exact plan"):
        deploy_engine.execute_plan(
            coordinator.plan(manifest, files, "start"),
            tmp_path / "receipt.json",
            "wrong",
            runner=runner,
            allow_model_actions=True,
        )
    assert runner.events == []


def test_stop_checks_ownership_on_all_hosts_before_stopping(deployment, tmp_path):
    manifest, files = deployment
    document = coordinator.plan(manifest, files, "stop")
    runner = FakeHosts(("spark1", "owned"))
    with pytest.raises(RuntimeError):
        deploy_engine.execute_plan(
            document,
            tmp_path / "receipt.json",
            document["sha256"],
            runner=runner,
            allow_model_actions=True,
        )
    assert not any(op == "stop" for _, op in runner.events)


def inspected(spec, manifest):
    return {
        "Id": "immutable-container-id",
        "Image": spec.image_id,
        "Config": {
            "Labels": {
                compose.LABEL: manifest["id"],
                "com.docker.compose.project": spec.name,
                "com.docker.compose.service": "model",
            },
            "Entrypoint": list(spec.entrypoint),
            "Cmd": list(spec.command),
            "Env": [f"{k}={v}" for k, v in spec.environment.items()],
            "Healthcheck": {
                "Test": (
                    ["CMD", *spec.health_command] if spec.health_command else ["NONE"]
                )
            },
        },
        "HostConfig": {
            "NetworkMode": "host",
            "IpcMode": "host",
            "Memory": spec.memory,
            "MemorySwap": spec.memory_swap,
            "RestartPolicy": {"Name": "no"},
            "Init": True,
            "Privileged": False,
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
            ],
            "Devices": [
                {"PathOnHost": d, "PathInContainer": d, "CgroupPermissions": "rwm"}
                for d in spec.devices
            ],
            "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1}],
        },
        "Mounts": [
            {
                "Source": m.source,
                "Destination": m.target,
                "RW": not m.read_only,
                "Type": "bind",
            }
            for m in spec.mounts
        ],
        "State": {"Running": True, "Status": "running"},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda info: info.update(Image="sha256:" + "f" * 64),
        lambda info: info["Config"]["Labels"].update(
            {compose.LABEL: "another-deployment"}
        ),
        lambda info: info["Config"].update(Entrypoint=["/bin/sh"]),
        lambda info: info["Config"]["Cmd"].append("--headless"),
        lambda info: info["Mounts"][0].update(RW=True),
        lambda info: info["HostConfig"].update(Memory=1),
        lambda info: info["HostConfig"].update(Privileged=True),
        lambda info: info["HostConfig"]["DeviceRequests"][0].update(Count=0),
        lambda info: info["Config"].update(Healthcheck={"Test": ["CMD", "true"]}),
    ],
)
def test_changed_or_foreign_container_cannot_be_adopted(
    deployment, monkeypatch, mutation
):
    manifest, _ = deployment
    spec = rank_spec(manifest)
    info = inspected(spec, manifest)
    mutation(info)
    monkeypatch.setattr(coordinator, "container", lambda _: info)
    with pytest.raises(ValueError):
        coordinator.owned(spec, manifest)


def test_stop_uses_exact_owned_id_not_compose_down(deployment, tmp_path, monkeypatch):
    manifest, files = deployment
    spec = rank_spec(manifest)
    info = inspected(spec, manifest)
    monkeypatch.setattr(coordinator, "container", lambda _: info)
    monkeypatch.setattr(coordinator, "stage_path", lambda *_: tmp_path)
    calls = []
    monkeypatch.setattr(coordinator, "run", lambda argv, **_: calls.append(argv))
    coordinator.host_operation(
        "stop", {"manifest": manifest, "files": files, "rank": 0}
    )
    assert calls == [["docker", "stop", "--time", "60", "immutable-container-id"]]


def test_existing_stage_directory_is_not_overwritten(deployment, tmp_path, monkeypatch):
    manifest, files = deployment
    monkeypatch.setattr(coordinator, "stage_path", lambda *_: tmp_path)
    sentinel = tmp_path / "compose.yaml"
    sentinel.write_text("existing content")
    with pytest.raises(FileExistsError):
        coordinator.host_operation(
            "stage", {"manifest": manifest, "files": files, "rank": 0}
        )
    assert sentinel.read_text() == "existing content"


def test_edited_stage_cannot_start_a_container(deployment, tmp_path, monkeypatch):
    manifest, files = deployment
    monkeypatch.setattr(coordinator, "stage_path", lambda *_: tmp_path)
    (tmp_path / "compose.yaml").write_text("edited")
    calls = []
    monkeypatch.setattr(coordinator, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="Staged deployment differs"):
        coordinator.host_operation(
            "start-api", {"manifest": manifest, "files": files, "rank": 0}
        )
    assert calls == []


def test_bootstrap_rejects_source_mismatch_before_import(deployment):
    manifest, files = deployment
    manifest["inputs"]["scripts/sparkring_compose.py"] = "0" * 64
    payload = coordinator.packed({"manifest": manifest, "files": files, "rank": 0})
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            coordinator.BOOTSTRAP,
            str(compose.ROOT),
            payload,
            "stage",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Host repository differs: scripts/sparkring_compose.py" in result.stderr


def test_api_exit_is_not_treated_as_readiness(deployment, monkeypatch):
    manifest, _ = deployment
    spec = rank_spec(manifest)
    info = inspected(spec, manifest)
    info["State"].update(Running=False, Status="exited")
    monkeypatch.setattr(coordinator, "container", lambda _: info)
    with pytest.raises(ValueError, match="exited"):
        coordinator.wait_ready(spec, manifest, seconds=10)


def test_check_plan_contains_only_read_only_preflight(deployment):
    manifest, files = deployment
    document = coordinator.plan(manifest, files, "check")
    assert [p["id"] for p in document["phases"]] == ["preflight"]
    assert all(a["risk"] == "read-only" for a in document["phases"][0]["actions"])
