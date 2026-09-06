"""CPU-only readiness contracts; tests never query a host or start a container."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

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
    assert shlex.split(command[6]) == ["docker", "inspect", "--format", "{{json .State}}", "model-r0"]


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
    monkeypatch.setattr(ready.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=json.dumps(state), stderr=""))
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
