"""Exercise direct-exec monitor selection and constrained readiness commands."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import managed_liveness as monitor
import managed_units


def candidate(entrypoint='/opt/sparkring/bin/sparkring-r33'):
    return {'Config': {'Entrypoint': [entrypoint], 'Env': [
        'SPARKRING_LIVENESS_ENABLED=1', 'PORT=8015', 'SPARKRING_LIVENESS_PORT=8016',
        'SPARKRING_IDLE_KV_WARN_SECONDS=330', 'SPARKRING_LIVENESS_OUTPUT_SECONDS=720',
        'SPARKRING_WARMUP_API_KEY=private-test-key']}}


@pytest.mark.parametrize('entrypoint', sorted(monitor.ENTRYPOINTS))
def test_only_rank_zero_direct_exec_needs_host_monitor(entrypoint):
    c = candidate(entrypoint)
    assert monitor.requires_host_monitor(c, 0)
    assert not monitor.requires_host_monitor(c, 1)
    assert not monitor.requires_host_monitor(candidate('/opt/sparkring/bin/serve-with-warmup.py'), 0)
    c['Config']['Env'][0] = 'SPARKRING_LIVENESS_ENABLED=0'
    assert not monitor.requires_host_monitor(c, 0)


def test_monitor_uses_pinned_local_api_and_existing_auth_without_emitting_secrets():
    result = monitor.settings(candidate())
    assert result['metrics_url'] == 'http://127.0.0.1:8015/metrics'
    assert result['port'] == 8016 and result['output_timeout_seconds'] == 720
    assert result['idle_kv_warn_seconds'] == 330
    assert result['credential'] == 'private-test-key'


def test_r35_python_entrypoint_is_supported_but_shell_arguments_are_not():
    c = candidate('/opt/sparkring/bin/sparkring')
    c['Config']['Entrypoint'].insert(0, '/opt/venv/bin/python')
    assert monitor.requires_host_monitor(c, 0)
    c['Config']['Entrypoint'] = ['/bin/sh', '-c', '/opt/sparkring/bin/sparkring']
    assert not monitor.requires_host_monitor(c, 0)


@pytest.mark.parametrize('command', [['-c','/opt/sparkring/bin/sparkring'],
    ['/opt/unrelated.py','serve'], ['/opt/sparkring/bin/sparkring','--help']])
def test_python_entrypoint_requires_exact_serving_command_prefix(command):
    c = candidate()
    c["Config"]["Entrypoint"] = ["/opt/venv/bin/python"]
    c["Config"]["Cmd"] = command
    assert not monitor.requires_host_monitor(c, 0)


@pytest.mark.parametrize(
    "assignment",
    [
        "PORT=$(touch x)",
        "PORT=65536",
        "SPARKRING_LIVENESS_PORT=8015",
        "SPARKRING_LIVENESS_SAMPLE_SECONDS=0",
    ],
)
def test_bad_monitor_settings_rejected(assignment):
    c = candidate()
    name = assignment.partition("=")[0]
    c["Config"]["Env"] = [
        v for v in c["Config"]["Env"] if not v.startswith(name + "=")
    ] + [assignment]
    with pytest.raises(ValueError):
        monitor.settings(c)


def test_host_unit_follows_model_lifetime_and_preserves_default_units():
    baseline = managed_units.unit_text("/opt/code", "/etc/config", "a" * 64)
    assert len(baseline) == 2
    active = managed_units.unit_text(
        "/opt/code", "/etc/config", "a" * 64, host_liveness=True
    )
    assert (
        "Wants=sparkring-scheduler-liveness.service"
        in active["sparkring-mesh-model.service"]
    )
    sidecar = active["sparkring-scheduler-liveness.service"]
    assert "BindsTo=sparkring-mesh-model.service" in sidecar
    assert "PartOf=sparkring-mesh-model.service" in sidecar
    assert "--config /etc/config/service.json" in sidecar
    assert "private-test-key" not in sidecar and "Restart=no" in sidecar


def test_wrong_container_identity_rejected_before_monitor_start(monkeypatch):
    c = candidate()
    c.update(Id="b" * 64, Image="sha256:" + "c" * 64, State={"Running": True})
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps([c])),
    )
    with pytest.raises(ValueError, match="pinned"):
        monitor.run(
            {"rank": 0, "container_id": "a" * 64, "container_image": c["Image"]}
        )


def test_monitor_waits_for_docker_start_and_closes_service(monkeypatch):
    c = candidate()
    c.update(Id="a" * 64, Image="sha256:" + "c" * 64, State={"Running": False})
    running = deepcopy(c)
    running["State"]["Running"] = True
    replies = iter([c, running])
    events = []
    monkeypatch.setattr(monitor.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=json.dumps([next(replies)])))
    monkeypatch.setattr(monitor.time, 'sleep', lambda seconds: events.append(('sleep', seconds)))
    monkeypatch.setattr(monitor.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(monitor.threading, 'Event', lambda: SimpleNamespace(wait=lambda: events.append('wait'), set=lambda: None))
    service = SimpleNamespace(close=lambda: events.append('closed'))
    helper = SimpleNamespace(start_liveness_service=lambda **kwargs: (events.append(('started', kwargs['port'])) or service))
    monkeypatch.setattr(monitor.importlib.util, 'spec_from_file_location', lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(monitor.importlib.util, 'module_from_spec', lambda spec: helper)
    monitor.run({'rank': 0, 'container_id': c['Id'], 'container_image': c['Image'], 'health_port': 9080})
    assert events == [('sleep', 1), ('started', 8016), 'wait', 'closed']
