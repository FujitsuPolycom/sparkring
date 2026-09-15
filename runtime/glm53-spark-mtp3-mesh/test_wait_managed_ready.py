"""CPU-only readiness contracts; tests never query a host or start a container."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from runtime.common.test_candidate import inputs as inputs

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mesh_readiness_test", HERE / "wait_managed_ready.py")
ready = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ready
SPEC.loader.exec_module(ready)
PLAN = {"containers": [{"rank": rank, "ssh_alias": f"spark-r{rank}", "name": f"model-r{rank}"}
                       for rank in range(4)], "urls": ["http://192.0.2.1:8015/health", "http://192.0.2.1:8016/liveness"]}


def healthy(target, timeout):
    assert 0 < timeout <= 5
    return {**target, "running": True, "health": "healthy", "ready": True}


def http_ok(url, timeout):
    assert 0 < timeout <= 5
    return {"url": url, "status": 200}


def test_inspect_quotes_remote_template():
    command = ready.inspect_command("spark-r0", "model-r0")
    assert command[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "spark-r0"]
    assert shlex.split(command[6]) == ["docker", "inspect", "--format", "{{json .}}", "model-r0"]


@pytest.mark.parametrize("alias", ["-oProxyCommand=x", "a b", "x;cmd", "u@host", "x\ncmd", ""])
def test_rejects_unsafe_alias(alias):
    with pytest.raises(ValueError):
        ready.inspect_command(alias, "model-r0")


def test_all_four_plus_both_http_required():
    result = ready.sample(PLAN, 100, inspect=healthy, request=http_ok, clock=lambda: 1)
    assert result["ready"] and len(result["containers"]) == 4 and len(result["http"]) == 2


@pytest.mark.parametrize("failing_url", PLAN["urls"])
def test_http_failure_never_reports_ready(failing_url):
    def request(url, timeout):
        if url == failing_url:
            raise OSError("HTTP readiness unavailable")
        return http_ok(url, timeout)
    result = ready.sample(PLAN, 100, inspect=healthy, request=request, clock=lambda: 1)
    assert not result["ready"]
    assert "unavailable" in result["error"]


def test_any_unhealthy_rank_prevents_http_gate():
    def inspect(target, timeout):
        return {**healthy(target, timeout), "ready": target["rank"] != 3}
    def forbidden(*args):
        pytest.fail("HTTP must not be treated as sufficient before every rank is healthy")
    assert not ready.sample(PLAN, 100, inspect=inspect, request=forbidden, clock=lambda: 1)["ready"]


@pytest.mark.parametrize("failure", [RuntimeError("No such object"),
                                   OSError("ssh unavailable"),
                                   subprocess.TimeoutExpired("ssh", 1)])
def test_inspect_errors_are_not_readiness(failure):
    def inspect(*args):
        raise failure
    result = ready.sample(PLAN, 100, inspect=inspect, request=http_ok, clock=lambda: 1)
    assert not result["ready"] and "error" in result


@pytest.mark.parametrize("state, expected", [
    ({"Running": True, "Health": {"Status": "healthy"}}, True),
    ({"Running": False, "Health": {"Status": "healthy"}}, False),
    ({"Running": True, "Health": {"Status": "starting"}}, False),
    ({"Running": True}, False),
])
def test_running_and_health_required(monkeypatch, state, expected):
    metadata = inspected(PLAN["containers"][0])
    metadata["State"].pop("Health", None)
    metadata["State"].update(state)
    monkeypatch.setattr(ready.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=json.dumps(metadata), stderr=""))
    assert ready.inspect_container(PLAN["containers"][0], 1)["ready"] is expected


def test_missing_container_raises(monkeypatch):
    monkeypatch.setattr(ready.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout="", stderr="No such object"))
    with pytest.raises(RuntimeError, match="No such object"):
        ready.inspect_container(PLAN["containers"][0], 1)


@pytest.mark.parametrize("timeout", [0, -1, 901, float("inf"), float("nan")])
def test_timeout_validation(timeout):
    with pytest.raises(ValueError):
        ready.wait(PLAN, timeout)


def test_deadline_caps_sleep_and_exits():
    tick = [0.0]
    def sleep(seconds):
        tick[0] += seconds
    result = ready.wait(PLAN, 3, clock=lambda: tick[0], sleep=sleep,
                        probe=lambda *args: {"ready": False})
    assert not result["ready"]
    assert result["elapsed_seconds"] == 3
    assert len(result["samples"]) == 2


def test_launch_ports_are_literal_and_explicit(tmp_path, monkeypatch):
    topology = SimpleNamespace(rank=lambda rank: SimpleNamespace(ssh_alias=f"spark-r{rank}"))
    monkeypatch.setattr(ready.profile, "load_site", lambda path:
                        ({"management_addresses": ["192.0.2.1"], "container_prefix": "model"}, topology, None))
    env = tmp_path / "rank0.env"
    env.write_text("PORT=8015\nSPARKRING_LIVENESS_PORT=8016\nSPARKRING_LIVENESS_ENABLED=1\n")
    assert ready.load_launch(tmp_path)["urls"] == PLAN["urls"]
    env.write_text("PORT=$(echo 8015)\nSPARKRING_LIVENESS_PORT=8016\nSPARKRING_LIVENESS_ENABLED=1\n")
    with pytest.raises(ValueError):
        ready.load_launch(tmp_path)


def test_expired_sample_does_not_query():
    def forbidden(*args):
        pytest.fail("Expired deadline must prevent queries")
    assert not ready.sample(PLAN, 1, inspect=forbidden, request=forbidden, clock=lambda: 2)["ready"]


def inspected(target, *, health="healthy"):
    rank = target["rank"]
    config = {"Healthcheck": {"Test": ["CMD", "check"]}}
    if "image_id" in target:
        config.update(
            Entrypoint=["/opt/venv/bin/python"],
            Cmd=[target["wrapper"], "serve", "/models/target", "--node-rank", str(rank),
                 "--tensor-parallel-size", "4", "--nnodes", "4", "--decode-context-parallel-size",
                 str(target["decode_context_parallel_size"]), *(["--headless"] if rank else [])],
            Env=[f"NODE_RANK={rank}", f"SPARKRING_NODE_RANK={rank}",
                 "SOURCE_IMAGE_PROFILE=" + target["runtime_profile"], "SPARKRING_PROFILE_MODE=custom"],
        )
    state = {"Running": True, "Status": "running", "Paused": False, "Restarting": False,
             "Dead": False, "OOMKilled": False, "Pid": 100 + rank, "StartedAt": "2026-09-14T00:00:00Z"}
    if health is None:
        config["Healthcheck"] = None
    else:
        state["Health"] = {"Status": health}
    return {"Name": "/" + target["name"], "Id": str(rank + 1) * 64,
            "Image": target.get("image_id", "sha256:" + "a" * 64),
            "Config": config, "State": state, "RestartCount": 0}


@pytest.fixture(params=["r35", "candidate"])
def registered_launch(request, tmp_path, monkeypatch, inputs):
    from runtime.common import candidate, r35
    from runtime.common.test_candidate import host_receipt
    from runtime.common.test_r35 import receipt

    spec = importlib.util.spec_from_file_location("readiness_examples", HERE / "make_example.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    document = receipt() if request.param == "r35" else host_receipt(inputs, monkeypatch)
    image_path = tmp_path / "receipt.json"
    image_path.write_text(json.dumps(document))
    topology = tmp_path / "fabric.example.json"
    topology.write_text(json.dumps(example.topology_example()))
    site = dict(example.site_example(), runtime_profile="tp4-dcp1")
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(site))
    monkeypatch.setattr(ready.profile, "verify_bundle", lambda bundle, image: image["bundle_manifest_sha256"])
    launch = tmp_path / "launch"
    ready.profile.render(site_path, tmp_path / "bundle", launch, image_path)
    return launch, ready.load_launch(launch), candidate if request.param == "candidate" else r35


def returned(monkeypatch, metadata):
    monkeypatch.setattr(ready.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(metadata), stderr=""))


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_verified_headless_workers_can_run_without_image_healthcheck(registered_launch, monkeypatch, rank):
    _, plan, _ = registered_launch
    target = plan["containers"][rank]
    returned(monkeypatch, inspected(target, health=None))
    result = ready.inspect_container(target, 1)
    assert result["ready"] is True and result["health_required"] is False
    assert result["health"] is None
    assert plan["stable_samples_required"] == 2


@pytest.mark.parametrize("health", [None, "starting", "unhealthy"])
def test_registered_api_rank_still_requires_healthy_docker_state(registered_launch, monkeypatch, health):
    target = registered_launch[1]["containers"][0]
    returned(monkeypatch, inspected(target, health=health))
    assert not ready.inspect_container(target, 1)["ready"]


@pytest.mark.parametrize("health", ["starting", "unhealthy"])
def test_present_worker_healthcheck_cannot_be_bypassed(registered_launch, monkeypatch, health):
    target = registered_launch[1]["containers"][1]
    returned(monkeypatch, inspected(target, health=health))
    assert not ready.inspect_container(target, 1)["ready"]


def test_active_worker_healthcheck_with_missing_runtime_status_blocks(registered_launch, monkeypatch):
    target = registered_launch[1]["containers"][1]
    metadata = inspected(target)
    del metadata["State"]["Health"]
    returned(monkeypatch, metadata)
    assert not ready.inspect_container(target, 1)["ready"]


@pytest.mark.parametrize("change", ["image", "name", "rank_env", "rank_arg", "headless", "wrapper", "duplicate_rank"])
def test_worker_exception_requires_exact_registered_serving_identity(registered_launch, monkeypatch, change):
    target = registered_launch[1]["containers"][1]
    metadata = inspected(target, health=None)
    config = metadata["Config"]
    if change == "image":
        metadata["Image"] = "sha256:" + "0" * 64
    elif change == "name":
        metadata["Name"] = "/other"
    elif change == "rank_env":
        config["Env"][0] = "NODE_RANK=2"
    elif change == "rank_arg":
        config["Cmd"][config["Cmd"].index("--node-rank") + 1] = "2"
    elif change == "headless":
        config["Cmd"].remove("--headless")
    elif change == "wrapper":
        config["Cmd"][0] = "/other/serve.py"
    else:
        config["Cmd"] += ["--node-rank", "1"]
    returned(monkeypatch, metadata)
    with pytest.raises(ValueError):
        ready.inspect_container(target, 1)


@pytest.mark.parametrize("flag", ["Paused", "Restarting", "Dead", "OOMKilled"])
def test_unstable_container_flags_block_readiness(registered_launch, monkeypatch, flag):
    target = registered_launch[1]["containers"][1]
    metadata = inspected(target, health=None)
    metadata["State"][flag] = True
    returned(monkeypatch, metadata)
    assert not ready.inspect_container(target, 1)["ready"]


@pytest.mark.parametrize("key,value", [("Running", 1), ("Paused", None), ("Pid", True),
                                      ("StartedAt", None), ("Health", "healthy")])
def test_malformed_worker_state_is_rejected(registered_launch, monkeypatch, key, value):
    target = registered_launch[1]["containers"][1]
    metadata = inspected(target, health=None)
    metadata["State"][key] = value
    returned(monkeypatch, metadata)
    with pytest.raises(ValueError):
        ready.inspect_container(target, 1)


def test_unregistered_legacy_worker_still_requires_health(monkeypatch):
    target = PLAN["containers"][1]
    returned(monkeypatch, inspected(target, health=None))
    assert not ready.inspect_container(target, 1)["ready"]


@pytest.mark.parametrize("change", ["receipt", "rank_environment", "topology"])
def test_worker_exception_cannot_be_granted_by_unbound_launch_metadata(registered_launch, change):
    launch, _, _ = registered_launch
    manifest = launch / "fabric-plan.json"
    document = json.loads(manifest.read_text())
    if change == "receipt":
        document["image"]["image_id"] = "sha256:" + "0" * 64
        document["image"]["image_reference"] = document["image"]["image_id"]
    elif change == "topology":
        document["topology_sha256"] = "0" * 64
    else:
        with (launch / "rank1.env").open("a") as stream:
            stream.write("\nNODE_RANK=2\n")
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        ready.load_launch(launch)


def test_registered_readiness_requires_two_stable_global_samples(registered_launch):
    _, plan, _ = registered_launch
    tick = [0]
    snapshots = []
    for marker in [1, 2, 2]:
        rows = [{"id": str(rank), "pid": 100 + rank, "started_at": str(marker), "restart_count": 0}
                for rank in range(4)]
        snapshots.append({"ready": True, "containers": rows, "http": [{"status": 200}] * 2})
    def probe(*args):
        return snapshots.pop(0)
    def sleep(seconds):
        tick[0] += seconds
    result = ready.wait(plan, 10, probe=probe, clock=lambda: tick[0], sleep=sleep)
    assert result["ready"] is True
    assert len(result["samples"]) == 3 and result["elapsed_seconds"] == 4


def test_global_http_gate_remains_required_with_headless_workers(registered_launch, monkeypatch):
    _, plan, _ = registered_launch
    metadata = {target["name"]: inspected(target, health="healthy" if target["rank"] == 0 else None)
                for target in plan["containers"]}
    def run(argv, **kwargs):
        name = shlex.split(argv[-1])[-1]
        return SimpleNamespace(returncode=0, stdout=json.dumps(metadata[name]), stderr="")
    monkeypatch.setattr(ready.subprocess, "run", run)
    def failed_http(url, timeout):
        raise OSError("scheduler is not ready")
    assert not ready.sample(plan, 10, request=failed_http, clock=lambda: 1)["ready"]
    assert ready.sample(plan, 10, request=http_ok, clock=lambda: 1)["ready"]


@pytest.mark.parametrize("timeout", [1501, float("inf"), float("nan"), -1])
def test_nvidia_budget_still_rejects_unbounded_timeouts(timeout):
    with pytest.raises(ValueError):
        ready.wait(dict(PLAN, target_model_variant="nvidia-nvfp4"), timeout)


def test_nvidia_budget_permits_slow_loader_without_changing_default():
    plan = dict(PLAN, target_model_variant="nvidia-nvfp4")
    assert ready.wait(plan, 1500, probe=lambda *args: {"ready": True})["ready"]
    with pytest.raises(ValueError):
        ready.wait(PLAN, 1500)
