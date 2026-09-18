"""Experiment ordering and lossless serving-spec reconstruction."""

import pytest
import json

from runtime.common.container_spec import Bind, ContainerSpec
from runtime.images.upgrades.contracts import Refused
from runtime.images.upgrades.tp2_experiment import from_document, trial_order
from runtime.images.upgrades import tp2_experiment


def test_saved_spec_roundtrips_without_losing_affinity_or_memory_envelope():
    spec = ContainerSpec(
        name="owned",
        image_id="sha256:" + "a" * 64,
        entrypoint=("python",),
        command=("serve",),
        environment={},
        mounts=(Bind("/model", "/models/target", True),),
        cpuset_cpus="0-19",
        memory=None,
        memory_swap=None,
    )
    assert from_document(spec.document()) == spec


def test_model_order_and_rollback_are_explicit():
    config = dict(
        schema="sparkring-tp2-experiment/v1",
        sites=[{"model": "glm"}, {"model": "qwen"}],
        rollback_snapshots=["r0.json", "r1.json"],
        public_port=8000,
    )
    assert trial_order(config) == config["sites"]
    config["sites"].reverse()
    with pytest.raises(Refused, match="GLM first"):
        trial_order(config)


@pytest.mark.parametrize("stop_failure", [False, True])
def test_failed_trial_or_partial_stop_restores_saved_pair(
    tmp_path, monkeypatch, stop_failure
):
    snapshots = [
        dict(
            Id=str(i) * 64,
            Image="sha256:" + "a" * 64,
            Name=f"/saved-r{i}",
            State={"Running": True},
        )
        for i in range(2)
    ]
    for rank, snapshot in enumerate(snapshots):
        (tmp_path / f"r{rank}.json").write_text(json.dumps(snapshot))
    (tmp_path / "site.json").write_text("{}")
    config = dict(
        schema="sparkring-tp2-experiment/v1",
        sites=[
            dict(model="glm", path="site.json"),
            dict(model="qwen", path="site.json"),
        ],
        rollback_snapshots=["r0.json", "r1.json"],
        public_port=8000,
        hosts=["u@h0", "u@h1"],
        hostnames=["h0", "h1"],
        rollback_api="http://h0:8000",
        rollback_model="model",
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    actions = []

    class FakePair:
        def __init__(self, *args, **kwargs):
            pass

        def inspect(self, rank, identifier):
            return snapshots[rank]

        def assert_idle(self, *args):
            actions.append("idle")

        def stop_saved(self, *args):
            actions.append("stop")
            if stop_failure:
                raise ValueError("partial stop")

        def start_saved(self, *args):
            actions.append("restore")

        def call(self, *args):
            return b""

    monkeypatch.setattr(tp2_experiment, "Pair", FakePair)
    monkeypatch.setattr(tp2_experiment, "load_policy", lambda *a: {})
    monkeypatch.setattr(tp2_experiment, "wait_ready", lambda *a, **kw: {})

    def failed(*args, **kwargs):
        raise ValueError("model failed")

    monkeypatch.setattr(tp2_experiment.tp2_suite, "run", failed)
    with pytest.raises(ValueError):
        tp2_experiment.run(
            path,
            tmp_path / "policy",
            tmp_path / "lease",
            "sha256:" + "b" * 64,
            tmp_path / "out",
            run_id="run-test",
            input_sha256="c" * 64,
            gate_id="trial",
        )
    assert actions == ["idle", "stop", "restore"]
    assert json.loads((tmp_path / "out/experiment.json").read_text())[
        "baseline_restored"
    ]
