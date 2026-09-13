"""Host-owned scheduler observation for verified direct-exec model containers."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'runtime/sparkring/source_image/startup/scheduler_liveness.py'
ENTRYPOINTS = {'/opt/sparkring/bin/sparkring-r33', '/opt/sparkring/bin/sparkring-r33-overlay',
               '/opt/sparkring/bin/sparkring'}


def api_healthcheck(port):
    if not str(port).isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError('API healthcheck requires a numeric TCP port')
    command = ("python3 -S -c 'import urllib.request; urllib.request.urlopen(\""
               f'http://127.0.0.1:{int(port)}/health'
               "\", timeout=4).close()'")
    return {'--health-cmd': command, '--health-interval': '10s', '--health-timeout': '6s',
            '--health-start-period': '1800s', '--health-retries': '3'}


def environment(container):
    result = {}
    for assignment in container.get('Config', {}).get('Env', []):
        name, separator, value = assignment.partition('=')
        if not separator or name in result:
            raise ValueError('Container environment must contain unique assignments')
        result[name] = value
    return result


def requires_host_monitor(container, rank):
    env = environment(container)
    entrypoint = container.get('Config', {}).get('Entrypoint', [])
    direct = any(entrypoint in ([path], ['/opt/venv/bin/python', path]) for path in ENTRYPOINTS)
    return (rank == 0 and env.get('SPARKRING_LIVENESS_ENABLED') == '1'
            and direct)


def settings(container):
    env = environment(container)
    def number(name, default, maximum=2147483647):
        value = env.get(name, str(default))
        if not value.isdecimal() or not 1 <= int(value) <= maximum:
            raise ValueError('Invalid scheduler liveness setting: ' + name)
        return int(value)
    api_port = number('PORT', 8015, 65535)
    port = number('SPARKRING_LIVENESS_PORT', 8016, 65535)
    if port < 1024 or port == api_port:
        raise ValueError('Scheduler liveness requires a distinct unprivileged port')
    return dict(metrics_url=f'http://127.0.0.1:{api_port}/metrics', port=port,
                blocked_timeout_seconds=number('SPARKRING_LIVENESS_BLOCKED_SECONDS', 60),
                output_timeout_seconds=number('SPARKRING_LIVENESS_OUTPUT_SECONDS', 300),
                idle_kv_warn_seconds=number('SPARKRING_IDLE_KV_WARN_SECONDS', 330),
                stale_sample_seconds=number('SPARKRING_LIVENESS_STALE_SECONDS', 15),
                sample_interval_seconds=number('SPARKRING_LIVENESS_SAMPLE_SECONDS', 10),
                credential=env.get('SPARKRING_WARMUP_API_KEY') or None)


def run(config):
    if config['rank'] != 0:
        raise ValueError('Scheduler liveness runs only on rank zero')
    # The model unit is Type=exec: Docker may not have started the container
    # when systemd releases this dependent unit. Wait only for that transition.
    deadline = time.monotonic() + 30
    while True:
        result = subprocess.run(['docker', 'inspect', config['container_id']],
                                capture_output=True, text=True, check=True, timeout=10)
        container = json.loads(result.stdout)[0]
        if (container['Id'] != config['container_id'] or container['Image'] != config['container_image']
                or not requires_host_monitor(container, 0)):
            raise ValueError('Scheduler liveness requires the pinned direct-exec container')
        if container['State']['Running']:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Pinned model container did not start for scheduler observation')
        time.sleep(1)
    options = settings(container)
    if options['port'] == config.get('health_port'):
        raise ValueError('Scheduler and mesh health ports must differ')
    spec = importlib.util.spec_from_file_location('managed_scheduler_liveness', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    service = helper.start_liveness_service(**options)
    try:
        stopped.wait()
    finally:
        service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
