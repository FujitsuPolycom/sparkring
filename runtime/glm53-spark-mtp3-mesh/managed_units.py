"""Render root-installed systemd units for a pre-created four-rank model."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import secrets
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('unit_mesh_service', HERE / 'managed_service.py')
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def systemd_path(value):
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', value) or '..' in Path(value).parts:
        raise ValueError('systemd paths must be absolute, with no whitespace, specifiers or parent traversal')
    return value


def unit_text(code_root, config_root, container_id):
    runner = systemd_path(code_root) + '/runtime/glm53-spark-mtp3-mesh/managed_service.py'
    config = systemd_path(config_root) + '/service.json'
    if not re.fullmatch('[0-9a-f]{64}', container_id):
        raise ValueError('Container ID must be pinned before rendering model units')
    common = f'/usr/bin/python3 {runner}'
    mesh = f'''[Unit]
Description=SparkRing authenticated four-rank hardware mesh
Wants=network-online.target
After=network-online.target docker.service
Requires=docker.service
Before=sparkring-mesh-model.service

[Service]
Type=notify
NotifyAccess=main
ExecStart={common} run --config {config}
ExecStopPost={common} stop-model --config {config}
Restart=no
WatchdogSec=15s
WatchdogSignal=SIGKILL
TimeoutStartSec=90s
TimeoutStopSec=infinity
# Preserve hardware forwarding after supervisor loss; explicit recovery reaps recorded children.
KillMode=process
RuntimeDirectory=sparkring-mesh
RuntimeDirectoryMode=0700
RuntimeDirectoryPreserve=yes
UMask=0077
Nice=10
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
'''
    model = f'''[Unit]
Description=SparkRing model bound to four-rank mesh readiness
Requires=docker.service sparkring-mesh.service
BindsTo=sparkring-mesh.service
After=docker.service sparkring-mesh.service

[Service]
Type=exec
ExecStartPre={common} gate --config {config} --timeout 60
ExecStartPre={common} model-arm --config {config}
ExecStart=/usr/bin/docker start --attach {container_id}
ExecStop={common} stop-model --config {config}
Restart=no
TimeoutStartSec=90s
TimeoutStopSec=infinity
KillMode=process
UMask=0077

[Install]
WantedBy=multi-user.target
'''
    return {'sparkring-mesh.service': mesh, 'sparkring-mesh-model.service': model}


def render(site_path, containers, output, code_root, config_root, epoch, health_port):
    if output.exists():
        raise ValueError('Managed unit output must not exist')
    if not re.fullmatch('[0-9a-f]{32}', epoch):
        raise ValueError('Epoch must be a shared 128-bit hexadecimal value')
    if type(health_port) is not int or not 1024 <= health_port <= 65535:
        raise ValueError('Invalid managed health port')
    site, _, _ = service.mesh_profile.load_site(site_path)
    if len(containers) != 4:
        raise ValueError('Expected four pre-created model container records')
    image_ids = {item['Image'] for item in containers}
    if len(image_ids) != 1:
        raise ValueError('All four ranks must use the same immutable image')
    for rank, item in enumerate(containers):
        if item['Name'].lstrip('/') != site['container_prefix'] + f'-r{rank}' or item['State']['Running']:
            raise ValueError('Expected stopped rank-ordered profile containers')
        if item['HostConfig'].get('RestartPolicy', {}).get('Name') not in ('no', ''):
            raise ValueError('Docker auto-restart must be disabled; systemd owns model lifetime')
        unit_text(code_root, config_root, item['Id'])
    output.mkdir(parents=True)
    for rank, item in enumerate(containers):
        directory = output / f'rank{rank}'
        directory.mkdir()
        config = {'schema': service.PROTOCOL, 'site_path': config_root + '/site.json', 'rank': rank,
                  'key_file': config_root + '/health.key', 'epoch': epoch, 'health_port': health_port,
                  'state_dir': '/run/sparkring-mesh', 'container_id': item['Id'], 'container_image': item['Image']}
        (directory / 'service.json').write_text(json.dumps(config, indent=2) + '\n', newline='\n')
        for name, text in unit_text(code_root, config_root, item['Id']).items():
            (directory / name).write_text(text, newline='\n')
    (output / 'render.json').write_text(json.dumps({'status': 'implemented', 'epoch': epoch,
        'site_sha256': service.mesh_profile.sha(site_path), 'image': next(iter(image_ids)),
        'container_ids': [item['Id'] for item in containers], 'health_port': health_port,
        'code_root': code_root, 'config_root': config_root}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--containers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-root', default='/opt/sparkring/managed-mesh')
    parser.add_argument('--config-root', default='/etc/sparkring/managed-mesh')
    parser.add_argument('--epoch', default=None)
    parser.add_argument('--health-port', type=int, default=9975)
    args = parser.parse_args()
    render(args.site, json.loads(args.containers.read_text()), args.output, systemd_path(args.code_root),
           systemd_path(args.config_root), args.epoch or secrets.token_hex(16), args.health_port)


if __name__ == '__main__':
    main()
