"""Build-time serving shutdown, exact restoration and uncertain-work boundaries."""

import copy
import json

import pytest

from runtime.images.upgrades import maintenance_build as module
from runtime.images.upgrades.contracts import Refused, Uncertain


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    saved = [
        {
            "Id": str(rank) * 64,
            "Name": f"/saved-{rank}",
            "Image": "sha256:" + "a" * 64,
            "State": {"Running": True},
        }
        for rank in (0, 1)
    ]
    actual = copy.deepcopy(saved)
    config = {
        "hosts": ["u@h0", "u@h1"],
        "hostnames": ["h0", "h1"],
        "gate_id": "build-maintenance",
        "rollback_api": "http://h0:8000",
        "rollback_model": "model",
        "startup_seconds": 900,
    }
    actions = []
    state = {
        "result": {"status": "candidate", "simulation": False},
        "partial_stop": False,
        "busy": False,
        "unknown": False,
    }

    class Pair:
        hosts = config["hosts"]

        def __init__(self, *args, **kwargs):
            pass

        def inspect(self, rank, identifier):
            assert identifier == actual[rank]["Id"]
            return actual[rank]

        def call(self, rank, argv):
            assert argv == ["docker", "ps", "--no-trunc", "--quiet"]
            ids = [actual[rank]["Id"]] if actual[rank]["State"]["Running"] else []
            if state["unknown"]:
                ids.append("f" * 64)
            return "\n".join(ids).encode()

        def assert_idle(self, url):
            actions.append("idle")
            if state["busy"]:
                raise Refused("busy")

        def stop_saved(self, snapshots):
            actions.append("stop")
            actual[0]["State"]["Running"] = False
            if state["partial_stop"]:
                raise Refused("partial stop")
            actual[1]["State"]["Running"] = False

        def start_saved(self, snapshots):
            actions.append("restore")
            assert snapshots == saved
            for item in actual:
                item["State"]["Running"] = True

    def build(*args, **kwargs):
        actions.append("build")
        assert not any(item["State"]["Running"] for item in actual)
        assert kwargs["publish"] is False
        if isinstance(state["result"], BaseException):
            raise state["result"]
        return state["result"]

    monkeypatch.setattr(module, "Pair", Pair)
    monkeypatch.setattr(module, "configuration", lambda *a: (config, saved, [[], []]))
    monkeypatch.setattr(
        module, "load_policy", lambda *a: {"_digest": "digest", "gates": []}
    )
    monkeypatch.setattr(module, "builder_lease_valid", lambda *a: {})
    monkeypatch.setattr(module, "wait_ready", lambda *a, **k: actions.append("ready"))
    monkeypatch.setattr(module.runner, "run", build)

    def run():
        return module.run(
            "policy",
            "state",
            "config",
            "builder-lease",
            "hardware-lease",
            tmp_path / "out",
            approved_policy="digest",
            run_id="night-test",
        )

    return run, state, actions, tmp_path


@pytest.mark.parametrize("status", ["candidate", "unchanged", "blocked"])
def test_terminal_build_restores_exact_saved_pair(scenario, status):
    run, state, actions, _ = scenario
    state["result"]["status"] = status
    result = run()
    assert actions == ["idle", "stop", "build", "restore", "ready"]
    assert result["baseline_restored"] is True
    assert not result["cleanup_errors"]


def test_partial_shutdown_restores_without_starting_build(scenario):
    run, state, actions, root = scenario
    state["partial_stop"] = True
    with pytest.raises(Refused, match="partial stop"):
        run()
    assert actions == ["idle", "stop", "restore", "ready"]
    assert json.loads((root / "out/maintenance.json").read_text())["baseline_restored"]


@pytest.mark.parametrize(
    "result",
    [
        {"status": "uncertain"},
        {"status": "candidate-simulation"},
        {"status": "candidate", "simulation": True},
        RuntimeError("lost worker"),
    ],
)
def test_unproven_build_termination_never_overlaps_serving_restart(scenario, result):
    run, state, actions, root = scenario
    state["result"] = result
    with pytest.raises(Uncertain, match="Build maintenance needs inspection"):
        run()
    assert actions == ["idle", "stop", "build"]
    receipt = json.loads((root / "out/maintenance.json").read_text())
    assert receipt["baseline_restored"] is False
    assert receipt["phase"] == "attention-required"


@pytest.mark.parametrize("field", ["busy", "unknown"])
def test_in_use_pair_is_not_stopped(scenario, field):
    run, state, actions, _ = scenario
    state[field] = True
    with pytest.raises(Refused):
        run()
    assert "stop" not in actions
    assert "build" not in actions
    assert "restore" not in actions


def test_maintenance_documents_require_protected_input_identity(tmp_path):
    path = tmp_path / "maintenance.json"
    path.write_text("{}")
    with pytest.raises(Refused, match="not bound"):
        module.bound_document({"_root": str(tmp_path), "_inputs": {}}, path.name)


def test_preexisting_uncertain_work_does_not_stop_serving(
    scenario, monkeypatch, tmp_path
):
    run, _, actions, _ = scenario
    monkeypatch.chdir(tmp_path)
    (tmp_path / "state").mkdir()
    (tmp_path / "state/state.json").write_text(
        json.dumps({"uncertain_run": "retained"})
    )
    with pytest.raises(Refused, match="before stopping"):
        run()
    assert not actions


def test_inventory_checks_image_identity_not_just_container_name():
    saved = [
        {"Id": str(rank) * 64, "Name": f"/saved-{rank}", "Image": "a"}
        for rank in (0, 1)
    ]

    class Pair:
        def inspect(self, rank, identifier):
            return {**saved[rank], "Image": "b"}

    with pytest.raises(Refused, match="identity differs"):
        module.inventory(Pair(), saved, [[], []])
