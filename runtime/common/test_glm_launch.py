"""GLM creation plans bind inputs and never turn creation into model startup."""
import hashlib
import json
import subprocess
from types import SimpleNamespace
from dataclasses import replace

import pytest

from runtime.common import glm_launch
from runtime.common.container_spec import Bind, ContainerSpec


@pytest.fixture
def context(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / "site.json").write_text('{}')
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{}')
    spec = ContainerSpec(name="glm-test-r0", image_id="sha256:" + "a" * 64,
                         entrypoint=("/opt/venv/bin/python",), command=("/serve.py",),
                         environment={"FIXTURE_PRIVATE_KEY": "private-fixture-value"}, mounts=(),
                         memory=None, memory_swap=None, health_mode="inherit")
    record = {"schema": glm_launch.candidate.SCHEMA, "image_reference": spec.image_id}
    resolved = {"topology": SimpleNamespace(sha256="b" * 64)}
    monkeypatch.setattr(glm_launch, "resolve_spec", lambda *a, **k: (spec, record, resolved))
    calls = []
    monkeypatch.setattr(glm_launch, "run", lambda *a, **k: calls.append(a) or pytest.fail("Plan must not contact Docker"))
    args = ["--launch", str(launch), "--image-receipt", str(receipt), "--rank", "0"]
    return launch, receipt, spec, record, resolved, args, calls


def test_plan_is_private_offline_and_binds_its_sources(context, capsys):
    launch, receipt, spec, record, resolved, args, calls = context
    assert glm_launch.main(["plan", *args]) == 0
    saved = json.loads((glm_launch.plan_directory(launch, 0) / "plan.json").read_bytes())
    assert saved["container"]["environment"]["FIXTURE_PRIVATE_KEY"] == "private-fixture-value"
    assert "private-fixture-value" not in capsys.readouterr().out
    assert saved["backend"] == "docker"
    assert glm_launch.check_plan(launch, receipt, 0, spec, record, resolved) == "docker"
    assert calls == []


def test_changed_plan_is_rejected_before_docker(context):
    launch, _, _, _, _, args, calls = context
    assert glm_launch.main(["plan", *args]) == 0
    path = glm_launch.plan_directory(launch, 0) / "plan.json"
    saved = json.loads(path.read_bytes())
    saved["container"]["command"] = ["changed-model"]
    path.write_text(json.dumps(saved))
    assert glm_launch.main(["create", *args]) == 2
    assert calls == []


def test_changed_yaml_is_rejected_even_for_docker_creation(context):
    launch, _, _, _, _, args, calls = context
    assert glm_launch.main(["plan", *args]) == 0
    (glm_launch.plan_directory(launch, 0) / "compose.yaml").write_text("services: {}")
    assert glm_launch.main(["create", *args]) == 2
    assert calls == []


def test_create_requires_prepared_plan(context):
    *_, args, calls = context
    assert glm_launch.main(["create", *args]) == 2
    assert calls == []


def test_model_requires_config_and_template_accepts_a_file(context, tmp_path):
    spec = context[2]
    model = tmp_path / "model"
    model.mkdir()
    template = tmp_path / "template.jinja"
    template.write_text("{{ messages }}")
    spec = replace(spec, mounts=(Bind(str(model), "/models/target", True),
                                 Bind(str(template), "/opt/sparkring/chat_template.jinja", True)))
    with pytest.raises(ValueError, match="config.json"):
        glm_launch.validate_mounts(spec)
    (model / "config.json").write_text('{}')
    glm_launch.validate_mounts(spec)
    template.unlink()
    template.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        glm_launch.validate_mounts(spec)


def test_create_verifies_image_and_leaves_container_stopped(context, monkeypatch, capsys):
    from runtime.common.container_spec import expected_inspection
    launch, _, spec, _, _, args, _ = context
    assert glm_launch.main(["plan", *args]) == 0
    image = {"Id": spec.image_id, "Os": "linux", "Architecture": "arm64", "Config": {}}
    expected = expected_inspection(spec, image)
    container = {"Id": "c" * 64, "Name": "/" + spec.name, "Image": spec.image_id,
                 "Config": {"Cmd": expected["cmd"], "Entrypoint": expected["entrypoint"],
                            "Healthcheck": expected["healthcheck"], "Env": [f"{k}={v}" for k, v in expected["env"].items()],
                            "Labels": expected["labels"], "User": expected["user"], "WorkingDir": expected["working_dir"]},
                 "Mounts": [], "HostConfig": expected["host_config"], "State": {"Running": False}}
    container["HostConfig"]["RestartPolicy"]["MaximumRetryCount"] = 0
    events = []
    monkeypatch.setattr(glm_launch.candidate, "verify_local_image", lambda *a, **k: events.append("verify-image"))

    def run(argv, **kwargs):
        events.append(argv[1])
        if argv[1] == "create":
            assert events[0] == "verify-image"
            return subprocess.CompletedProcess(argv, 0, stdout="c" * 64)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps([image if argv[1] == "image" else container]))

    monkeypatch.setattr(glm_launch, "run", run)
    assert glm_launch.main(["create", *args]) == 0
    assert "start" not in events
    saved = json.loads((glm_launch.plan_directory(launch, 0) / "created.json").read_bytes())
    assert saved == {"container_id": "c" * 64, "image_id": spec.image_id, "running": False}
    assert "private-fixture-value" not in capsys.readouterr().out


def test_plan_generation_preserves_the_authenticated_launch_inventory(context):
    from scripts.deploy_engine import plan_digest
    from scripts.deploy_trust import _verify_preparation

    launch, _, _, _, _, args, calls = context
    workspace = launch.parent
    source = workspace / "source"
    source.mkdir()
    script = source / "fixture.py"
    script.write_text("VALUE = 1\n")
    required = {"site.json", "fabric.json", "fabric-plan.json", "launch-rank.sh"}
    required.update(f"rank{rank}.env" for rank in range(4))
    for name in required - {"site.json"}:
        (launch / name).write_text("{}" if name.endswith(".json") else "fixture\n")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    document = {
        "spec": {"workspace": str(workspace)},
        "source": {"files": {"fixture.py": sha(script)}},
        "launch_files": {name: sha(launch / name) for name in required},
    }
    preparation = workspace / "preparation.json"
    preparation.write_text(json.dumps(document))
    verification_args = (str(workspace), plan_digest(document), str(source),
                         str(launch), str(preparation), True)
    assert _verify_preparation(*verification_args) == document
    assert glm_launch.main(["plan", *args]) == 0
    output = glm_launch.plan_directory(launch, 0)
    assert output == workspace / "launch-containers" / "rank0"
    assert (output / "plan.json").is_file()
    assert (output / "compose.yaml").is_file()
    assert {path.name for path in launch.iterdir()} == required
    assert _verify_preparation(*verification_args) == document
    assert glm_launch.main(["check", *args]) == 0
    assert calls == []


def test_plan_directory_resolves_launch_aliases_without_rank_collisions(context):
    launch = context[0]
    assert glm_launch.plan_directory(launch / ".", 0) == glm_launch.plan_directory(launch, 0)
    assert glm_launch.plan_directory(launch, 0) != glm_launch.plan_directory(launch, 1)
