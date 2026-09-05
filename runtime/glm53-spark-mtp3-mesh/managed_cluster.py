"""Coordinate model-stop barriers and explicit recovery of four managed mesh ranks."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import time

import managed_units


def phases(action, code_root, config_root):
    runner = ['/usr/bin/python3', code_root + '/runtime/glm53-spark-mtp3-mesh/managed_service.py']
    def command(operation):
        return ['sudo', '-n', *runner, operation, '--config', config_root + '/service.json']
    systemctl = ['sudo', '-n', 'systemctl']
    stop = [('quiesce-models', command('model-quiesce')),
            ('stop-model-units', [*systemctl, 'stop', 'sparkring-mesh-model.service']),
            ('stop-pinned-containers', command('stop-model')),
            ('model-stop-barrier', command('model-stopped'))]
    down = [*stop, ('stop-mesh-units', [*systemctl, 'stop', 'sparkring-mesh.service']),
            ('recover-owned-children', command('cleanup')),
            ('clear-failure-latches', command('reset-units'))]
    table = {
        'up': [('model-stop-barrier', command('model-stopped')),
               ('start-mesh-units', [*systemctl, 'start', 'sparkring-mesh.service']),
               ('four-rank-ready', [*command('gate'), '--timeout', '60'])],
        'start-model': [('four-rank-ready', [*command('gate'), '--timeout', '60']),
                        ('start-model-units', [*systemctl, 'start', 'sparkring-mesh-model.service'])],
        'stop-model': stop,
        'down': down,
        'recover': [*down, ('start-mesh-units', [*systemctl, 'start', 'sparkring-mesh.service']),
                    ('four-rank-ready', [*command('gate'), '--timeout', '60'])],
        'status': [('systemd-state', [*systemctl, 'show', 'sparkring-mesh.service', 'sparkring-mesh-model.service',
                                     '--property=Id,ActiveState,SubState,Result,MainPID'])],
    }
    return table[action]


def execute(host, argv):
    started = time.monotonic()
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(argv)],
                                capture_output=True, text=True, timeout=120)
        return {'host': host, 'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr, 'seconds': time.monotonic() - started}
    except subprocess.TimeoutExpired:
        return {'host': host, 'argv': argv, 'returncode': 124, 'seconds': time.monotonic() - started,
                'error': 'SSH deadline exceeded; remote operation may still be in progress'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('up', 'start-model', 'stop-model', 'down', 'recover', 'status'))
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--code-root', default='/opt/sparkring/managed-mesh')
    parser.add_argument('--config-root', default='/etc/sparkring/managed-mesh')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute-authorized', action='store_true')
    args = parser.parse_args()
    _, topology, _ = managed_units.service.mesh_profile.load_site(args.site)
    hosts = [topology.rank(rank).ssh_alias for rank in range(4)]
    import re
    if any(not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.@-]*', host) for host in hosts):
        raise ValueError('SSH aliases cannot contain shell syntax or options')
    steps = phases(args.action, managed_units.systemd_path(args.code_root), managed_units.systemd_path(args.config_root))
    if not args.execute_authorized:
        print(json.dumps({'action': args.action, 'hosts': hosts, 'phases': steps, 'executed': False}, indent=2))
        return
    if args.output is None or args.output.exists():
        raise ValueError('Executed operations require an absent receipt path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'action': args.action, 'started_unix': time.time(), 'phases': []}
    for name, argv in steps:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda host: execute(host, argv), hosts))
        receipt['phases'].append({'name': name, 'ranks': results})
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')
        passed = all(result['returncode'] == 0 for result in results)
        print(json.dumps({'phase': name, 'passed': passed}), flush=True)
        if not passed:
            raise SystemExit('Phase failed; no later phase executed. Inspect receipts before recovery.')
    receipt['completed_unix'] = time.time()
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
