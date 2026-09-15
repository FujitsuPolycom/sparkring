"""Upgrade execution adapters without Docker, model weights or remote services."""

import datetime as dt
import io
import json
import time

import pytest

from runtime.images.upgrades import agent, contracts, discovery, execution, image_gate
from runtime.images.upgrades.demo import setup
from runtime.images.upgrades.io import write_json


def test_docker_oracle_is_readonly_isolated_and_subject_bound(tmp_path, monkeypatch):
    path, _ = setup(tmp_path / "fixture")
    policy = contracts.load_policy(path)
    executor = execution.Executor(policy, execute=True)
    context = dict(
        input_sha="a" * 64,
        subject_sha256="b" * 64,
        variant="baseline",
        run_id="test-run",
        baseline_path=str(tmp_path / "baseline"),
    )
    gate = policy["gates"][0]
    seen = []

    def fake(argv, output, *args, **kwargs):
        seen.extend(argv)
        write_json(
            output / "result.json",
            dict(
                schema="sparkring-upgrade-gate/v1",
                gate=gate["id"],
                input_sha256=context["input_sha"],
                subject_sha256=context["subject_sha256"],
                variant="baseline",
                assertions=1,
                skipped=0,
                outcome="passed",
            ),
        )

    monkeypatch.setattr(execution.sys, "platform", "linux")
    monkeypatch.setattr(execution.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(execution.os, "getgid", lambda: 1000, raising=False)
    monkeypatch.setattr(executor, "_run", fake)
    result = executor.gate(
        gate, tmp_path / "source", tmp_path / "out", context, time.monotonic() + 10
    )
    assert result["outcome"] == "passed"
    assert "--read-only" in seen and "--gpus" not in seen
    assert seen[seen.index("--network") + 1] == "none"
    assert seen[seen.index("--entrypoint") + 1] == "python"
    assert "SPARKRING_UPGRADE_SUBJECT=" + context["subject_sha256"] in seen
    assert not any(
        "docker.sock" in value for value in seen if value.startswith("type=bind")
    )
    assert (tmp_path / "out/owned-resource.json").is_file()


def test_hardware_cannot_execute_without_lease(tmp_path):
    path, _ = setup(tmp_path / "fixture")
    executor = execution.Executor(contracts.load_policy(path), execute=True)
    gate = dict(
        id="decode",
        stage="hardware",
        executor="operator",
        resources=["reserved-pair"],
        argv=["never-execute"],
    )
    with pytest.raises(contracts.Refused, match="operator lease"):
        executor.gate(gate, tmp_path, tmp_path / "out", {}, time.monotonic() + 10)


@pytest.mark.parametrize(
    "overrides",
    [
        {"platform": "linux/amd64"},
        {"installed_verified": False},
        {"features": []},
        {"image_id": "somewhere:mutable"},
        {"input_sha256": "b" * 64},
    ],
)
def test_build_receipts_refuse_wrong_platform_identity_or_missing_features(
    tmp_path, monkeypatch, overrides
):
    path, _ = setup(tmp_path / "fixture")
    policy = contracts.load_policy(path)
    policy["required_features"] = ["collectives"]
    executor = execution.Executor(policy, execute=True, build=True)
    context = dict(input_sha="a" * 64)
    receipt = dict(
        schema="sparkring-upgrade-build/v1",
        input_sha256=context["input_sha"],
        platform="linux/arm64",
        image_id="sha256:" + "a" * 64,
        installed_verified=True,
        features=["collectives"],
    )
    receipt.update(overrides)

    def fake(argv, output, *args, **kwargs):
        write_json(output / "result.json", receipt)

    monkeypatch.setattr(executor, "_run", fake)
    monkeypatch.setattr(execution.sys, "platform", "linux")
    monkeypatch.setattr(execution.platform, "machine", lambda: "aarch64")
    with pytest.raises(contracts.Refused):
        executor.action("build", context, tmp_path / "out", time.monotonic() + 10)


def test_publish_requires_independent_policy_permission(tmp_path):
    path, _ = setup(tmp_path / "fixture")
    policy = contracts.load_policy(path)
    policy["publish"] = {"argv": ["never-execute"]}
    executor = execution.Executor(policy, execute=True, build=True, publish=True)
    with pytest.raises(contracts.Refused, match="Policy does not authorize"):
        executor.action("publish", {}, tmp_path / "out", time.monotonic() + 10)


def test_file_proposal_binds_exact_request(tmp_path):
    request = {"source": "engine", "files": {"engine/f.py": "source"}}
    digest = contracts.sha(contracts.encoded(request))
    transport = agent.FileAgent(tmp_path)
    with pytest.raises(contracts.Refused, match="proposal required"):
        transport.propose(request)
    proposal = dict(
        disposition="unresolved", reason="Required interface not provided.", patch=""
    )
    path = tmp_path / (digest + ".json")
    write_json(path, {"request_sha256": digest, "proposal": proposal})
    assert transport.propose(request) == proposal
    write_json(path, {"request_sha256": "b" * 64, "proposal": proposal}, replace=True)
    with pytest.raises(contracts.Refused, match="identity differs"):
        transport.propose(request)


def test_feature_and_connector_binding_changes_do_not_rehash(tmp_path):
    package = tmp_path / "opt/venv/lib/python3.12/site-packages"
    package.mkdir(parents=True)
    source = package / "engine.py"
    source.write_bytes(b"source")
    expected = contracts.sha(source.read_bytes())
    preimages = {"/opt/venv/lib/python3.12/site-packages/engine.py": expected}
    write_json(
        tmp_path / "opt/sparkring/features/qwen-prefill/manifest.json",
        {"schema": "sparkring-qwen-prefill/v1", "image_source_preimages": preimages},
    )
    binding = tmp_path / "opt/sparkring/contracts/vllm-connector-jobs-fixture.json"
    write_json(
        binding,
        {
            "schema": "sparkring-vllm-kv-block-lease-contract/v1",
            "files": [{"path": "engine.py", "sha256": expected}],
        },
    )
    before = binding.read_bytes()
    assert image_gate.verify_bindings(tmp_path) == (2, [])
    source.write_bytes(b"incompatible")
    assertions, failures = image_gate.verify_bindings(tmp_path)
    assert assertions == 2 and len(failures) == 2
    assert binding.read_bytes() == before


@pytest.mark.parametrize(
    "files,schema",
    [([], "sparkring-vllm-kv-block-lease-contract/v1"), ([{}], "unknown")],
)
def test_unknown_or_empty_connector_binding_is_rejected(tmp_path, files, schema):
    write_json(
        tmp_path / "opt/sparkring/contracts/vllm-connector-jobs-fixture.json",
        {"schema": schema, "files": files},
    )
    assert image_gate.verify_bindings(tmp_path)[1]


@pytest.mark.parametrize("architecture", ["arm64", "amd64"])
def test_image_feed_requires_registry_architecture_without_pulling_layers(
    monkeypatch, architecture
):
    value = dict(
        repository=discovery.REGISTRY,
        reference=discovery.REGISTRY + "@sha256:" + "a" * 64,
        tag="vllmb12x-dev-jovian-judgement-"
        + "a" * 12
        + "-"
        + "b" * 12
        + "-20260914-n1",
        resolved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(value).encode()),
    )
    commands = []

    def inspect(argv, **kwargs):
        commands.append(argv)
        return json.dumps({"os": "linux", "architecture": architecture}).encode()

    monkeypatch.setattr(discovery, "checked", inspect)
    if architecture == "arm64":
        assert (
            discovery.discover_arm64()["publication_is_serving_qualification"] is False
        )
    else:
        with pytest.raises(contracts.Refused, match="not Linux ARM64"):
            discovery.discover_arm64()
    assert len(commands) == 1 and "inspect" in commands[0] and "pull" not in commands[0]


def test_context_prioritizes_conflicting_files(tmp_path):
    source = dict(id="engine", editable_paths=["engine"], contracts=[])
    record = dict(
        baseline="a" * 40, target="b" * 40, changed_paths=["engine/unrelated.py"]
    )
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine/f.py").write_text("source")
    (tmp_path / "engine/unrelated.py").write_text("x" * 100000)
    patch = "--- a/engine/f.py\n+++ b/engine/f.py\n"
    request = agent.request_for(
        source,
        record,
        tmp_path,
        tmp_path,
        patch.encode(),
        [{"patch": patch, "error": "conflict"}],
        2000,
    )
    assert set(request["files"]) == {"engine/f.py"}


def test_source_exclusions_cannot_hide_runtime_code(tmp_path):
    path, _ = setup(tmp_path / "fixture")
    policy = json.loads(path.read_text())
    policy["sources"][0]["excluded_paths"] = ["engine"]
    path.write_text(json.dumps(policy))
    with pytest.raises(contracts.Refused, match="Only agent-tool metadata"):
        contracts.load_policy(path)


def test_subprocess_sends_complete_large_patch_input():
    import sys
    from runtime.images.upgrades.io import command

    payload = b"x" * (2 * 1024 * 1024)
    result = command(
        [sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"],
        input_bytes=payload,
    )
    assert result["returncode"] == 0 and not result["uncertain"]
    assert int(result["stdout"]) == len(payload)


def test_hardware_performance_requires_passing_matched_control():
    gate = dict(
        id="decode",
        stage="hardware",
        baseline_image="sha256:" + "b" * 64,
        metrics={
            "tokens_s": {
                "direction": "higher",
                "min_samples": 3,
                "max_regression_fraction": 0.03,
            }
        },
    )
    context = dict(
        input_sha="c" * 64, subject_sha256="sha256:" + "a" * 64, variant="image"
    )
    value = dict(
        schema="sparkring-upgrade-gate/v1",
        gate="decode",
        input_sha256=context["input_sha"],
        subject_sha256=context["subject_sha256"],
        variant="image",
        assertions=3,
        skipped=0,
        outcome="passed",
        measurements={"tokens_s": [90, 91, 92]},
    )
    with pytest.raises(contracts.Refused, match="receipt schema"):
        execution.validate_gate(value, gate, context)
    assert not execution.acceptable(value, gate)
    control = {
        **value,
        "variant": "control",
        "subject_sha256": gate["baseline_image"],
        "measurements": {"tokens_s": [99, 100, 101]},
    }
    value["baseline"] = control
    execution.validate_gate(value, gate, context)
    assert not execution.acceptable(value, gate)
    value["measurements"]["tokens_s"] = [98, 99, 100]
    assert execution.acceptable(value, gate)
    control["outcome"] = "failed"
    with pytest.raises(contracts.Refused, match="control did not pass"):
        execution.validate_gate(value, gate, context)


def test_gate_cannot_preplant_host_log_destination(tmp_path, monkeypatch):
    path, _ = setup(tmp_path / "fixture")
    policy = contracts.load_policy(path)
    executor = execution.Executor(policy, execute=True)
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "stdout.log"
    existing.write_bytes(b"preserve")
    monkeypatch.setattr(
        execution, "builder_lease_valid", lambda *args: {"expires_at": time.time() + 60}
    )
    monkeypatch.setattr(
        execution,
        "command",
        lambda *a, **kw: dict(
            stdout=b"overwrite", stderr=b"", returncode=0, uncertain=False
        ),
    )
    with pytest.raises(contracts.Refused, match="safely record"):
        executor._run(["fixture-only"], output, time.monotonic() + 60)
    assert existing.read_bytes() == b"preserve"


def test_file_proposal_survives_measurement_noise_not_policy_or_source_drift(tmp_path):
    request = dict(
        source="engine",
        policy_sha256="a" * 64,
        files={"engine.py": "source"},
        feedback={"seconds": 1.1},
    )
    digest = agent.request_identity(request)
    proposal = dict(
        disposition="unresolved", reason="Requires interface review.", patch=""
    )
    write_json(
        tmp_path / (digest + ".json"), {"request_sha256": digest, "proposal": proposal}
    )
    transport = agent.FileAgent(tmp_path)
    request["feedback"] = {"seconds": 1.2}
    assert transport.propose(request) == proposal
    assert agent.request_identity({**request, "policy_sha256": "b" * 64}) != digest
    request["files"]["engine.py"] = "different source"
    with pytest.raises(contracts.Refused, match="proposal required"):
        transport.propose(request)
