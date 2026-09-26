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
    """The rank container as host_operation derives it from the deployment manifest."""
    spec = compose.specifications(
        manifest["profile"], manifest["site"], **compose.selection_options(manifest)
    )[0][number]
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


@pytest.mark.parametrize("owned_id", [None, "owned-container-id"])
def test_project_collision_rejected_regardless_of_container_name(
    deployment, monkeypatch, owned_id
):
    manifest, _ = deployment
    spec = rank_spec(manifest)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="renamed-container-id\n")

    monkeypatch.setattr(coordinator, "run", run)
    with pytest.raises(ValueError, match="project already contains"):
        coordinator.check_project_containers(spec, owned_id=owned_id)
    assert calls == [[
        "docker", "container", "ls", "--all", "--no-trunc",
        "--filter", "label=com.docker.compose.project=" + spec.name,
        "--format", "{{.ID}}",
    ]]


def test_project_check_allows_only_exact_owned_container_on_resume(deployment, monkeypatch):
    manifest, _ = deployment
    spec = rank_spec(manifest)
    monkeypatch.setattr(
        coordinator, "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="owned-id\n"),
    )
    coordinator.check_project_containers(spec, owned_id="owned-id")


@pytest.mark.parametrize("project_container", ["renamed-container-id", ""])
def test_create_cannot_recreate_a_project_container(
    deployment, tmp_path, monkeypatch, project_container
):
    manifest, files = deployment
    monkeypatch.setattr(coordinator, "stage_path", lambda *_: tmp_path)
    monkeypatch.setattr(coordinator, "staged", lambda *_: None)
    monkeypatch.setattr(coordinator, "container", lambda _: None)
    monkeypatch.setattr(compose, "check_equivalence", lambda *a, **k: None)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["container", "ls"]:
            return subprocess.CompletedProcess(argv, 0, stdout=project_container)
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(coordinator, "run", run)
    payload = {"manifest": manifest, "files": files, "rank": 0}
    if project_container:
        with pytest.raises(ValueError, match="project already contains"):
            coordinator.host_operation("create", payload)
        assert len(calls) == 1
    else:
        coordinator.host_operation("create", payload)
        assert calls[-1][-6:] == [
            "create", "--no-build", "--no-recreate", "--pull", "never", "model"
        ]


@pytest.mark.parametrize("profile", compose.SUPPORTED)
def test_host_create_checks_the_exact_staged_file(profile, tmp_path, monkeypatch):
    """Each host re-derives the staged file's container, including the installer adaptation."""
    owner = profile.removesuffix("-sparkcache")
    site = compose.read_site(compose.ROOT / "profiles" / owner / "compose/site.example.yaml")
    manifest, files = compose.build(profile, site)
    monkeypatch.setattr(coordinator, "container", lambda _: None)
    monkeypatch.setattr(coordinator, "check_project_containers", lambda *a, **k: None)
    monkeypatch.setattr(coordinator, "run", lambda argv, **_: subprocess.CompletedProcess(argv, 0, stdout=""))
    checked = []
    monkeypatch.setattr(compose, "check_equivalence",
                        lambda spec, image, text, **_: checked.append((spec, image, text)))
    for number in range(len(site["ranks"])):
        payload = {"manifest": manifest, "files": files, "rank": number}
        monkeypatch.setattr(coordinator, "stage_path", lambda *_, n=number: tmp_path / str(n))
        coordinator.host_operation("stage", payload)
        coordinator.host_operation("create", payload)
        spec, image, text = checked[-1]
        assert text == files[f"rank{number}/compose.yaml"]
        assert compose.compose_text(spec, image) == text
        assert spec == rank_spec(manifest, number)


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


@pytest.mark.parametrize("state", [None, "created", "exited", "running"])
@pytest.mark.parametrize("busy", [False, True])
def test_preflight_checks_idle_gpu_for_every_stopped_owner(deployment, tmp_path, monkeypatch, state, busy):
    manifest, _ = deployment
    spec = rank_spec(manifest)
    rank = dict(manifest["site"]["ranks"][0])
    for name in ("model", "cache", "repository", "deployment_root"):
        path = tmp_path / name
        path.mkdir()
        rank[name] = str(path)
    metadata, _ = coordinator.profiles.load(manifest["profile"])
    profile = coordinator.qwen_flash_next.read(coordinator.ROOT / metadata["configuration"]["path"])
    present = inspected(spec, manifest) if state is not None else None
    if present:
        present["State"].update(Running=state == "running", Status=state)
    monkeypatch.setattr(coordinator.sys, "platform", "linux")
    monkeypatch.setattr(coordinator, "container", lambda _: present)
    monkeypatch.setattr(coordinator.qwen_flash_next, "verify_model_paths", lambda *a: None)
    monkeypatch.setattr(coordinator.ports, "check_tcp_bind", lambda *a: None)
    monkeypatch.setattr(compose, "check_equivalence", lambda *a, **k: None)
    path_type = type(tmp_path)
    original_is_dir, original_read = path_type.is_dir, path_type.read_text
    monkeypatch.setattr(path_type, "is_dir", lambda path: path.as_posix() == "/dev/infiniband" or original_is_dir(path))
    def read(path, *args, **kwargs):
        if path.as_posix().startswith("/sys/class/infiniband/"):
            return "4: ACTIVE" if path.name == "state" else "0000:0000:0000:0000:0000:ffff:c000:0214"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(path_type, "read_text", read)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "ip":
            output = json.dumps([{"addr_info": [{"local": rank["host_ip"]}]}])
        elif argv[:3] == ["docker", "image", "inspect"]:
            output = json.dumps([{"Id": spec.image_id, "Os": "linux", "Architecture": "arm64"}])
        elif argv[:3] == ["docker", "container", "ls"]:
            output = present["Id"] if present else ""
        elif argv[0] == "nvidia-smi":
            output = "GPU-fixture" if argv[1] == "--query-gpu=uuid" else "1234" if busy else ""
        else:
            pytest.fail("Unexpected host operation: " + str(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=output)
    monkeypatch.setattr(coordinator, "run", run)
    if busy and state != "running":
        with pytest.raises(ValueError, match="compute workload"):
            coordinator.preflight(rank, manifest["site"], spec, spec.image_id, profile, manifest)
    else:
        coordinator.preflight(rank, manifest["site"], spec, spec.image_id, profile, manifest)
    queried = any("--query-compute-apps=pid" in argv for argv in calls)
    assert queried is (state != "running")


@pytest.mark.parametrize("operation,rank", [("start-api", 0), ("start-worker", 1)])
@pytest.mark.parametrize("busy", [False, True])
def test_start_rechecks_gpu_after_creation(deployment, tmp_path, monkeypatch, operation, rank, busy):
    manifest, files = deployment
    spec = rank_spec(manifest, rank)
    present = inspected(spec, manifest)
    present["State"].update(Running=False, Status="created")
    monkeypatch.setattr(coordinator, "stage_path", lambda *a: tmp_path / "staged")
    monkeypatch.setattr(coordinator, "container", lambda _: present)
    payload = {"manifest": manifest, "files": files, "rank": rank}
    coordinator.host_operation("stage", payload)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(argv, 0, stdout="1234" if busy else "")
        assert argv[-2:] == ["start", "model"]
        return subprocess.CompletedProcess(argv, 0, stdout="")
    monkeypatch.setattr(coordinator, "run", run)
    if busy:
        with pytest.raises(ValueError, match="compute workload"):
            coordinator.host_operation(operation, payload)
        assert all(argv[-2:] != ["start", "model"] for argv in calls)
    else:
        coordinator.host_operation(operation, payload)
        assert calls[0][1] == "--query-compute-apps=pid"
        assert calls[-1][-2:] == ["start", "model"]
