"""Render a pinned two-node launch; execution is explicit and create-only."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


def render(rank, master, model_dir, cache_dir, env_file, image):
    if rank not in (0, 1) or not master or any(c.isspace() for c in master):
        raise ValueError('Valid rank and master address required')
    if not re.fullmatch(r'ghcr.io/fujitsupolycom/sparkring@sha256:[0-9a-f]{64}', image):
        raise ValueError('A published immutable SparkRing digest is required')
    environment = env_file.read_text()
    if '<' in environment or '>' in environment:
        raise ValueError('Resolve site placeholders in the environment file')
    for required in ('VLLM_HOST_IP', 'NCCL_IB_HCA', 'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME'):
        if not re.search(r'^' + required + r'=\S+', environment, re.M):
            raise ValueError('Missing environment value: ' + required)
    if not model_dir.is_dir() or not cache_dir.is_dir():
        raise ValueError('Existing checkpoint and cache directories required')
    profile = json.loads(Path(__file__).with_name('profile.json').read_text())
    args = [x.replace('${MASTER_ADDR}', master).replace('${NODE_RANK}', str(rank)) for x in profile['vllm_args']]
    if rank == 1:
        args.append('--headless')
    name = f'sparkring-glm53-tp2-r{rank}'
    command = ['docker', 'create', '--name', name, '--label', 'org.sparkring.memory-guard=true',
               '--init', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
               '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1:-1',
               '--device', '/dev/infiniband:/dev/infiniband', '--security-opt', 'label=disable',
               '--env-file', str(env_file.resolve()),
               '--mount', f'type=bind,src={model_dir.resolve()},dst=/models/target,readonly',
               '--mount', f'type=bind,src={cache_dir.resolve()},dst=/cache/jit', image, *args]
    return name, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', required=True, type=int)
    parser.add_argument('--master', required=True)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--env-file', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    publication = json.loads(Path(__file__).resolve().parents[2].joinpath('sparkring/publication.json').read_text())
    name, command = render(args.rank, args.master, args.model_dir, args.cache_dir, args.env_file, publication['registry_digest'])
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    subprocess.run(['systemctl', 'is-active', '--quiet', 'sparkring-memory-guard'], check=True)
    ids = subprocess.check_output(['docker', 'ps', '-q'], text=True).split()
    if ids:
        running = json.loads(subprocess.check_output(['docker', 'inspect', *ids]))
        if any(c['HostConfig'].get('DeviceRequests') for c in running):
            raise RuntimeError('A GPU container is already running; stop it explicitly first')
    subprocess.run(command, check=True)
    subprocess.run(['docker', 'start', name], check=True)


if __name__ == '__main__':
    main()
