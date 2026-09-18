"""Adversarial receipts, policy drift, bounded work and reconciliation failures."""

import datetime as dt
import difflib
import json
from pathlib import Path
import socket
import sys
import time

import pytest

from . import contracts, discovery, execution, io, sources
from .agent import ChatAgent, validate
from .demo import DemoAgent, DemoExecutor, FIXED, REFACTORED, setup
from .runner import resolve_uncertain, run


def test_bounded_command_failure_retains_the_actionable_tail(monkeypatch):
    detail = (
        b"compiler context\n" + b"progress\n" * 1000 + b"fatal: missing owned CLI\n"
    )
    monkeypatch.setattr(
        io,
        "command",
        lambda *a, **k: {
            "returncode": 1,
            "uncertain": False,
            "stderr": detail,
            "stdout": b"",
        },
    )
    with pytest.raises(contracts.Refused) as failure:
        io.checked(["docker", "build"])
    message = str(failure.value)
    assert "compiler context" in message and "fatal: missing owned CLI" in message
    assert "intermediate output omitted" in message and len(message) < 2100


@pytest.fixture
def fixture(tmp_path):
    policy, repository = setup(tmp_path / "fixture")
    return policy, repository, DemoExecutor(policy.parent / "oracle.py")


def update(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def commit_source(repository, text):
    (repository / "engine/cache.py").write_bytes(text.encode())
    sources.git(repository, "add", "--all")
    sources.git(repository, "commit", "-m", "Change fixture behavior")


def execute(fixture, **kwargs):
    policy, _, executor = fixture
    return run(
        policy,
        policy.parent / "state",
        execute=True,
        build=True,
        executor=executor,
        **kwargs,
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(sources=[None]),
        lambda p: p.update(name=[]),
        lambda p: p["budgets"].update(agent_attempts=6),
        lambda p: p["budgets"].update(command_seconds=True),
        lambda p: p["budgets"].update(typo=3),
        lambda p: p.update(permissions={"candidate_publication": "yes"}),
        lambda p: p["gates"][0].update(image="somewhere:latest"),
        lambda p: p["gates"][0].update(executor="operator"),
        lambda p: p["gates"][0].update(argv="sh script.sh"),
        lambda p: p["gates"][0].update(metrics=[]),
        lambda p: p["sources"][0].update(ref="main"),
        lambda p: p["sources"][0].update(patch="../outside.patch"),
        lambda p: p.update(
            agent={
                "endpoint": "https://host/v1",
                "model": "fixture",
                "api_key": None,
            }
        ),
    ],
)
def test_malformed_or_ambiguous_policy_rejected(fixture, change):
    policy, _, _ = fixture
    update(policy, change)
    with pytest.raises(contracts.Refused):
        contracts.load_policy(policy)


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../x", "a/../b", "a//b", "C:/x", "a\\b", "x\n", "."]
)
def test_paths_cannot_escape_owner(path):
    with pytest.raises(contracts.Refused):
        contracts.relative(path)


def test_policy_binds_oracle_and_controller(fixture):
    path, _, _ = fixture
    policy = contracts.load_policy(path)
    assert "controller/runtime/images/candidate_image.py" in policy["_inputs"]
    assert "controller/scripts/image_upgrade.py" in policy["_inputs"]
    (path.parent / "oracle.py").write_text("raise SystemExit(0)")
    with pytest.raises(contracts.Refused, match="Oracle input identity"):
        contracts.check_policy(policy)


@pytest.mark.parametrize("action", [None, "refuse", "rebuild"])
def test_native_cache_rebuild_action_is_policy_bound(fixture, action):
    path, _, _ = fixture
    cache = {"manifest": "/build/cache/manifest.json", "sha256": "a" * 64}
    if action is not None:
        cache["on_input_change"] = action
    update(path, lambda p: p.setdefault("foundation", {}).update(native_cache=cache))
    update(path, lambda p: p["build"].update(supports_native_rebuild=True))
    policy = contracts.load_policy(path)
    assert policy["foundation"]["native_cache"] == cache
    update(path, lambda p: p["foundation"]["native_cache"].update(sha256="b" * 64))
    with pytest.raises(contracts.Refused, match="policy or oracle changed"):
        contracts.check_policy(policy)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["foundation"]["native_cache"].update(on_input_change="ignore"),
        lambda p: p["foundation"]["native_cache"].update(manifest="../manifest.json"),
        lambda p: p["foundation"]["native_cache"].update(sha256="not-a-digest"),
        lambda p: p["foundation"]["native_cache"].update(extra=True),
        lambda p: p["build"].update(supports_native_rebuild=False),
    ],
)
def test_native_cache_fallback_requires_valid_explicit_policy(fixture, change):
    path, _, _ = fixture
    update(
        path,
        lambda p: p.setdefault("foundation", {}).update(
            native_cache={
                "manifest": "/build/cache/manifest.json",
                "sha256": "a" * 64,
                "on_input_change": "rebuild",
            }
        ),
    )
    update(path, lambda p: p["build"].update(supports_native_rebuild=True))
    update(path, change)
    with pytest.raises(contracts.Refused):
        contracts.load_policy(path)


def test_baseline_failure_blocks(fixture):
    policy, _, _ = fixture
    oracle = policy.parent / "oracle.py"
    oracle.write_text(
        oracle.read_text().replace("passed = all(", "passed = False and all(")
    )
    update(
        policy,
        lambda p: p["gates"][0]["inputs"][0].update(
            sha256=contracts.sha(oracle.read_bytes())
        ),
    )
    result = execute(fixture)
    assert result["status"] == "blocked"
    assert "Baseline oracle is not passing" in result["reason"]


def test_oracle_cannot_mutate_candidate(fixture):
    policy, _, executor = fixture
    original = executor.gate

    def mutate(gate, source, output, context, deadline):
        value = original(gate, source, output, context, deadline)
        if context["variant"] == "candidate":
            (Path(source) / "injected").write_text("tamper")
        return value

    executor.gate = mutate
    result = execute(fixture)
    assert result["status"] == "blocked"
    assert "Oracle changed candidate" in result["reason"]


class Retire:
    def propose(self, *args, **kwargs):
        return dict(
            disposition="retire",
            reason="Check upstream behavior independently.",
            patch="",
        )


def test_retirement_requires_upstream_success(fixture):
    _, repository, _ = fixture
    commit_source(repository, REFACTORED)
    result = execute(fixture, agent=Retire())
    assert result["status"] == "blocked"
    assert "upstream has not passed" in result["reason"]


def test_optimization_retirement_requires_performance_oracle(fixture):
    policy, repository, _ = fixture
    commit_source(repository, FIXED)
    update(
        policy, lambda p: p["sources"][0]["contracts"][0].update(kind="optimization")
    )
    result = execute(fixture, agent=Retire())
    assert result["status"] == "blocked"
    assert "protected performance oracle" in result["reason"]


def test_correctness_patch_can_retire_with_independent_evidence(fixture):
    _, repository, _ = fixture
    commit_source(repository, FIXED)
    result = execute(fixture, agent=Retire())
    assert result["status"] == "candidate-simulation"
    row = result["reconciliation"]["engine"]
    assert row["disposition"] == "retire"
    assert Path(row["patch_path"]).read_bytes() == b""


def test_native_drift_refuses_overlay_reuse(fixture):
    _, repository, _ = fixture
    (repository / "native/version.txt").write_text("2\n")
    sources.git(repository, "add", "--all")
    sources.git(repository, "commit", "-m", "Change compiled input")
    result = execute(fixture)
    assert result["status"] == "blocked"
    assert "Native/build inputs changed" in result["reason"]


def test_build_receipt_must_bind_all_accepted_sources(fixture):
    _, _, executor = fixture
    original = executor.action

    def omit(*args):
        value = original(*args)
        value["source_trees"] = {}
        return value

    executor.action = omit
    assert "every accepted source tree" in execute(fixture)["reason"]


def test_failed_input_is_not_skipped(fixture):
    _, repository, _ = fixture
    commit_source(repository, REFACTORED)
    assert execute(fixture)["status"] == "blocked"
    assert execute(fixture)["status"] == "blocked"
    assert execute(fixture, agent=DemoAgent())["status"] == "candidate-simulation"


def test_plan_does_not_skip_execution(fixture):
    policy, _, _ = fixture
    assert run(policy, policy.parent / "state")["status"] == "planned"
    assert execute(fixture)["status"] == "candidate-simulation"


def test_uncertain_action_blocks_retries_until_explicit_resolution(fixture):
    policy, _, executor = fixture
    original = executor.action

    def interrupted(*args):
        raise contracts.Uncertain("Owned builder work requires inspection")

    executor.action = interrupted
    result = execute(fixture)
    assert result["status"] == "uncertain"
    with pytest.raises(contracts.Refused, match="operator resolution"):
        execute(fixture, force=True)
    with pytest.raises(contracts.Refused, match="identity differs"):
        resolve_uncertain(policy.parent / "state", "different-run")
    resolve_uncertain(policy.parent / "state", result["run_id"])
    executor.action = original
    assert execute(fixture)["status"] == "candidate-simulation"


def receipt():
    gate = {
        "id": "throughput",
        "metrics": {
            "tokens_s": {
                "direction": "higher",
                "max_regression_fraction": 0.03,
                "min_samples": 3,
            }
        },
    }
    context = {
        "input_sha": "a" * 64,
        "subject_sha256": "b" * 64,
        "variant": "candidate",
    }
    value = dict(
        schema="sparkring-upgrade-gate/v1",
        gate=gate["id"],
        input_sha256=context["input_sha"],
        subject_sha256=context["subject_sha256"],
        variant=context["variant"],
        outcome="passed",
        assertions=4,
        skipped=0,
        measurements={"tokens_s": [99, 100, 101]},
    )
    return value, gate, context


@pytest.mark.parametrize(
    "field,value",
    [
        ("assertions", 0),
        ("assertions", True),
        ("skipped", 1),
        ("outcome", "okay"),
        ("subject_sha256", "c" * 64),
        ("input_sha256", "c" * 64),
        ("variant", "upstream"),
        ("measurements", {"tokens_s": [100, 100]}),
        ("measurements", {"tokens_s": [100, float("nan"), 100]}),
    ],
)
def test_receipt_cannot_fake_evidence(field, value):
    result, gate, context = receipt()
    result[field] = value
    with pytest.raises(contracts.Refused):
        execution.validate_gate(result, gate, context)


def test_performance_gate_compares_medians():
    baseline, gate, _ = receipt()
    candidate = {**baseline, "measurements": {"tokens_s": [90, 91, 999]}}
    assert not execution.acceptable(candidate, gate, baseline)
    candidate["measurements"]["tokens_s"] = [97, 98, 99]
    assert execution.acceptable(candidate, gate, baseline)


def test_agent_patch_paths_and_protected_native_inputs(fixture):
    _, repository, _ = fixture
    patch = "".join(
        difflib.unified_diff(
            ["1\n"],
            ["2\n"],
            fromfile="a/native/version.txt",
            tofile="b/native/version.txt",
        )
    ).encode()
    with pytest.raises(contracts.Refused, match="unapproved path"):
        sources.apply_patch(repository, patch, ["engine"])
    reverse = patch.replace(b"-1\n+2\n", b"-2\n+1\n")
    with pytest.raises(contracts.Refused, match="native or protected"):
        sources.apply_patch(repository, reverse, ["native"], ["native/version.txt"])


def test_partial_application_preserves_compatible_files(fixture):
    _, repository, _ = fixture
    first = b"diff --git a/native/version.txt b/native/version.txt\n--- a/native/version.txt\n+++ b/native/version.txt\n@@ -1 +1 @@\n-1\n+2\n"
    second = b"diff --git a/engine/cache.py b/engine/cache.py\n--- a/engine/cache.py\n+++ b/engine/cache.py\n@@ -1 +1 @@\n-not present\n+repair\n"
    applied, failures = sources.apply_fragments(repository, first + second)
    assert not applied and len(failures) == 1
    assert (repository / "native/version.txt").read_text() == "2\n"
    assert "engine/cache.py" in failures[0]["patch"]


def test_agent_response_is_data_not_actions():
    with pytest.raises(contracts.Refused):
        validate({"command": "anything"})
    with pytest.raises(contracts.Refused):
        validate(dict(disposition="retire", reason="No test", patch="a diff"))


def test_runner_rejects_agent_edits_to_opaque_carried_asset(fixture):
    policy, repository, _ = fixture
    asset = repository / "engine/calibration.gz"
    original = b"\0approved-calibration"
    asset.write_bytes(original)
    sources.git(repository, "add", "-N", "engine/calibration.gz")
    addition = sources.git(
        repository, "diff", "--binary", "--", "engine/calibration.gz"
    )
    patch_path = policy.parent / "geometry.patch"
    patch_path.write_bytes(patch_path.read_bytes() + addition)
    update(
        policy,
        lambda p: p["sources"][0].update(
            patch_sha256=contracts.sha(patch_path.read_bytes())
        ),
    )
    commit_source(repository, REFACTORED)
    asset.write_bytes(b"\0unreviewed-calibration")
    replacement = sources.git(
        repository, "diff", "--binary", "--", "engine/calibration.gz"
    ).decode()
    asset.write_bytes(original)

    class AssetEditor(DemoAgent):
        def propose(self, request, **kwargs):
            assert "engine/calibration.gz" in request["protected_paths"]
            assert "GIT binary patch" not in request["carried_patch"]
            result = super().propose(request, **kwargs)
            result["patch"] += replacement
            return result

    result = execute(fixture, agent=AssetEditor())
    assert result["status"] == "blocked"
    assert "native or protected path" in result["reason"]
    with pytest.raises(contracts.Refused):
        ChatAgent(dict(endpoint="http://example.org/v1", model="fixture"))
    with pytest.raises(contracts.Refused):
        ChatAgent(dict(endpoint="https://secret@example.org/v1", model="fixture"))


def test_lock_excludes_concurrent_controller_and_releases(tmp_path):
    root = tmp_path / "state"
    with io.lock(root):
        with pytest.raises(contracts.Refused, match="Another upgrade"):
            with io.lock(root):
                pytest.fail("lock allowed concurrent owner")
    with io.lock(root):
        pass


def test_non_owned_state_refused(tmp_path):
    (tmp_path / "user-data").write_text("preserve")
    with pytest.raises(contracts.Refused, match="not owned"):
        with io.lock(tmp_path):
            pytest.fail("foreign state admitted")


def test_subprocess_strips_secrets_and_bounds_output_and_time(monkeypatch):
    monkeypatch.setenv("UPGRADE_TEST_SECRET", "do-not-inherit")
    result = io.command(
        [sys.executable, "-c", "import os; print(os.getenv('UPGRADE_TEST_SECRET'))"]
    )
    assert result["stdout"].strip() == b"None"
    result = io.command([sys.executable, "-c", "print('a'*1000000)"], limit=100)
    assert result["uncertain"] and len(result["stdout"]) == 100
    result = io.command(
        [sys.executable, "-c", "import time; time.sleep(20)"], seconds=0.1
    )
    assert result["uncertain"] and result["termination"] == "timeout"


def test_storage_checks_never_delete_files(tmp_path):
    item = tmp_path / "keep"
    item.write_bytes(b"abc")
    with pytest.raises(contracts.Refused, match="storage budget"):
        io.storage_check(tmp_path, {"state_bytes": 2})
    assert item.read_bytes() == b"abc"


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(host="another-host"),
        lambda v: v.update(expires_at=1),
        lambda v: v.update(exclusive=False),
        lambda v: v.update(policy_sha256="b" * 64),
        lambda v: v.update(not_before="tomorrow"),
    ],
)
def test_builder_lease_binds_owner_and_deadline(tmp_path, change):
    policy = {"_digest": "a" * 64}
    value = dict(
        schema="sparkring-upgrade-builder-lease/v1",
        policy_sha256=policy["_digest"],
        host=socket.gethostname(),
        exclusive=True,
        expires_at=time.time() + 100,
    )
    path = tmp_path / "lease.json"
    io.write_json(path, value)
    assert execution.builder_lease_valid(path, policy) == value
    change(value)
    io.write_json(path, value, replace=True)
    with pytest.raises(contracts.Refused):
        execution.builder_lease_valid(path, policy)


def test_hardware_lease_cannot_authorize_other_resources(tmp_path):
    policy, gate = {"_digest": "a" * 64}, {"id": "cache", "resources": ["pair-A"]}
    path = tmp_path / "lease.json"
    io.write_json(
        path,
        dict(
            schema="sparkring-upgrade-hardware-lease/v1",
            policy_sha256=policy["_digest"],
            exclusive=True,
            expires_at=time.time() + 100,
            resources=["pair-B"],
            gate_ids=["cache"],
        ),
    )
    with pytest.raises(contracts.Refused, match="resources"):
        execution.lease_valid(path, policy, gate, time.monotonic() + 1000)


def publication():
    return dict(
        repository=discovery.REGISTRY,
        reference=discovery.REGISTRY + "@sha256:" + "a" * 64,
        tag="vllmb12x-dev-jovian-judgement-"
        + "a" * 12
        + "-"
        + "b" * 12
        + "-20260914-n1",
        resolved_at="2026-09-14T12:00:00Z",
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(repository="another/repo"),
        lambda v: v.update(reference="somewhere:latest"),
        lambda v: v.update(resolved_at="2026-09-12T12:00:00Z"),
        lambda v: v.update(resolved_at="2026-09-15T12:00:00Z"),
        lambda v: v.update(resolved_at="2026-09-14T12:00:00"),
        lambda v: v.update(tag="latest"),
    ],
)
def test_publication_is_immutable_fresh_and_from_expected_publisher(change):
    now = dt.datetime(2026, 9, 14, 12, tzinfo=dt.timezone.utc)
    value = publication()
    assert (
        discovery.validate_publication(value, now=now)[
            "publication_is_serving_qualification"
        ]
        is False
    )
    change(value)
    with pytest.raises(contracts.Refused):
        discovery.validate_publication(value, now=now)


def test_cli_refuses_implicit_build_and_publication(fixture):
    from scripts.image_upgrade import main

    policy, _, _ = fixture
    assert (
        main(
            [
                "run",
                "--policy",
                str(policy),
                "--state",
                str(policy.parent / "state"),
                "--build",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "run",
                "--policy",
                str(policy),
                "--state",
                str(policy.parent / "state"),
                "--execute",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "run",
                "--policy",
                str(policy),
                "--state",
                str(policy.parent / "state"),
                "--publish",
            ]
        )
        == 2
    )
