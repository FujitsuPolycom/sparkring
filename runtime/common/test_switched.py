"""Check the host resources required when starting a stopped switched container."""

import json
import subprocess

import pytest

from runtime.common import switched


@pytest.fixture
def launch_case(tmp_path):
    model, cache = tmp_path / 'model', tmp_path / 'cache'
    model.mkdir()
    cache.mkdir()
    (model / 'config.json').write_text('{}')
    site = tmp_path / 'site.env'
    site.write_text('VLLM_HOST_IP=rank.example\nNCCL_SOCKET_IFNAME=eth0\n'
                    'GLOO_SOCKET_IFNAME=eth0\nNCCL_IB_HCA==mlx5_0:1\nNCCL_IB_GID_INDEX=3\n')
    plan = switched.render(0, 'master.example', model, cache, site,
                           'ghcr.io/fujitsupolycom/sparkring@sha256:' + 'a' * 64)
    receipt = {'registry_digest': plan['image'], 'profiles': {plan['profile']: {
        **{key: plan[key] for key in ('profile_sha256', 'topology', 'collective_backend')},
        'source_compatibility': 'passed'}}}
    container = {'Config': {
        'Image': plan['image'], 'Entrypoint': ['python3'], 'User': plan['container_user'],
        'Healthcheck': {'Test': ['CMD-SHELL', switched.HEALTHCHECK_COMMAND]},
        'Cmd': plan['container_args'], 'Labels': plan['labels'],
        'Env': [key + '=' + value for key, value in plan['environment'].items()]},
        'Mounts': [{'Destination': target, 'Source': source, 'RW': target != '/models/target'}
                   for target, source in plan['binds'].items()],
        'HostConfig': {'RestartPolicy': {'Name': 'no'}, 'NetworkMode': 'host',
                       'IpcMode': 'host', 'DeviceRequests': [
                           {'Driver': '', 'Count': -1, 'DeviceIDs': None,
                            'Capabilities': [['gpu']], 'Options': {}}]}}
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = ''
        if command[:2] == ['systemctl', 'show']:
            output = 'argv[]=/usr/bin/python3 guard --available-floor-bytes 4294967296 ;'
        elif command == ['docker', 'inspect', plan['name']]:
            output = json.dumps([container])
        return subprocess.CompletedProcess(command, 0, output, '')

    return plan, receipt, container, commands, run


def test_matching_stopped_container_starts(launch_case):
    plan, receipt, _, commands, run = launch_case
    switched.execute(plan, 'start', receipt, run=run)
    assert commands[-1] == ['docker', 'start', plan['name']]


@pytest.mark.parametrize('field,value', [
    ('NetworkMode', 'bridge'), ('IpcMode', 'private'), ('DeviceRequests', []),
    ('DeviceRequests', [{'Count': 0, 'Capabilities': [['gpu']]}]),
    ('DeviceRequests', [{'Count': -1, 'Capabilities': [['tpu']]}]),
])
def test_start_rejects_missing_profile_resources(launch_case, field, value):
    plan, receipt, container, commands, run = launch_case
    container['HostConfig'][field] = value
    with pytest.raises(RuntimeError, match='differs from this profile plan'):
        switched.execute(plan, 'start', receipt, run=run)
    assert ['docker', 'start', plan['name']] not in commands
