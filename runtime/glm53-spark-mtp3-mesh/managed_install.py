"""Install one rank's reviewed managed-mesh contract; plan-only unless --apply is supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import stat
import subprocess
import tempfile

import managed_units

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE_DIR = Path('/opt/sparkring/managed-mesh')
CONFIG_DIR = Path('/etc/sparkring/managed-mesh')
UNIT_DIR = Path('/etc/systemd/system')
SOURCE_FILES = (
    'runtime/glm53-spark-mtp3-mesh/managed_service.py',
    'runtime/glm53-spark-mtp3-mesh/managed_network.py',
    'runtime/glm53-spark-mtp3-mesh/managed_units.py',
    'runtime/glm53-spark-mtp3-mesh/managed_cluster.py',
    'runtime/glm53-spark-mtp3-mesh/managed_install.py',
    'runtime/glm53-spark-mtp3-mesh/profile.py',
    'runtime/glm53-spark-mtp3-mesh/inspect_fabric.py',
    'runtime/glm53-spark-mtp3-mesh/pins.json',
    'runtime/glm53-flash-jj-r8-gb10/pins.json',
    'runtime/glm53-flash-jj-r8-gb10/warmup_dflash.py',
    'runtime/glm53-flash-jj-r8-gb10/launch-rank.sh',
    'runtime/glm53-flash-jj-r8-gb10/runtime.env.example',
    'runtime/glm53-flash-jj-r8-gb10/sircl-fused.env.example',
    'spark_transport/experiments/cx7_hairpin_diagonal/__init__.py',
    'spark_transport/experiments/cx7_hairpin_diagonal/fabric.py',
    'spark_transport/experiments/glm53_rocenante_overlay/build_bundle.py',
)


def source_payloads():
    """Read only required code; reject symlink components before destination writes."""
    result = {}
    if not stat.S_ISDIR(ROOT.lstat().st_mode):
        raise ValueError('Managed source root must be a directory, not a symlink')
    for relative in SOURCE_FILES:
        path = ROOT
        parts = Path(relative).parts
        for index, part in enumerate(parts):
            path = path / part
            mode = path.lstat().st_mode
            expected = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
            if not expected(mode):
                raise ValueError(f'Managed source must be a regular file beneath nonsymlink directories: {relative}')
        result[relative] = path.read_bytes()
    return result


def source_hashes(payloads):
    return {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}


def install_code(payloads, destination):
    """Install the already validated byte snapshot, never traverse mutable source trees."""
    if set(payloads) != set(SOURCE_FILES):
        raise ValueError('Managed source snapshot differs from the required file allowlist')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Managed code destination must be absent')
    for relative, content in payloads.items():
        path = destination / relative
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
        path.chmod(0o644)
    for directory in [destination, *[path for path in destination.rglob('*') if path.is_dir()]]:
        directory.chmod(0o755)
    return source_hashes(payloads)


def validate_container(item, site, rank, image):
    if (item.get('Name', '').lstrip('/') != site['container_prefix'] + f'-r{rank}'
            or item.get('Image') != image or item.get('State', {}).get('Running') is not False
            or not re.fullmatch('[0-9a-f]{64}', str(item.get('Id')))):
        raise ValueError('Expected the stopped profile container with the pinned image')
    if item['HostConfig'].get('RestartPolicy', {}).get('Name') not in ('no', ''):
        raise ValueError('Disable Docker restart policy; systemd owns model lifetime')


def environment_map(values, role):
    """Reject ambiguous or inherited-value environment entries without exposing values."""
    if not isinstance(values, list):
        raise ValueError(f'{role} environment must be a list')
    result = {}
    for value in values:
        if not isinstance(value, str) or '=' not in value:
            raise ValueError(f'{role} environment requires literal assignments')
        name, content = value.split('=', 1)
        if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', name) or name in result:
            raise ValueError(f'{role} environment has an invalid or duplicate name')
        result[name] = content
    return result


def expected_container_spec(argv, image):
    """Interpret the canonical launcher's Docker envelope, never reimplement vLLM flags.

    Unknown Docker envelope options fail closed until their effect is accounted
    for. The model's complete argument list is retained verbatim.
    """
    if (not isinstance(argv, list) or any(not isinstance(x, str) for x in argv)
            or argv[:2] != ['docker', 'create']):
        raise ValueError('Expected a canonical Docker create command')
    image_id = image.get('Id')
    if not isinstance(image_id, str) or not re.fullmatch('sha256:[0-9a-f]{64}', image_id):
        raise ValueError('Image inspection must provide an immutable identity')
    config = image.get('Config', {})
    if config.get('Volumes'):
        raise ValueError('Managed profile does not support image-declared anonymous volumes')
    env = environment_map(config.get('Env') or [], 'Image')
    supplied_env = []
    mounts, options = {}, {}
    labels = dict(config.get('Labels') or {})
    value_options = {'--name', '--entrypoint', '--network', '--ipc', '--shm-size', '--gpus',
                     '--ulimit', '--cap-add', '--device', '--security-opt', '-v', '-e', '--label'}
    index = 2
    while index < len(argv) and argv[index] != image_id:
        flag = argv[index]
        if flag == '--init':
            if flag in options:
                raise ValueError('Duplicate Docker init option')
            options[flag] = True
            index += 1
            continue
        if flag not in value_options or index + 1 >= len(argv):
            raise ValueError('Unsupported Docker envelope option in canonical launcher')
        value = argv[index + 1]
        options.setdefault(flag, []).append(value)
        if flag == '-e':
            supplied_env.append(value)
        elif flag == '-v':
            pieces = value.split(':')
            if (len(pieces) not in (2, 3) or not pieces[0].startswith('/')
                    or not pieces[1].startswith('/') or pieces[1] in mounts
                    or (len(pieces) == 3 and pieces[2] not in ('ro', 'rw'))):
                raise ValueError('Expected distinct absolute bind mounts with explicit access')
            mounts[pieces[1]] = {'Source': posixpath.normpath(pieces[0]), 'Type': 'bind',
                                 'RW': len(pieces) == 2 or pieces[2] == 'rw'}
        elif flag == '--label':
            if '=' not in value:
                raise ValueError('Docker labels must be literal assignments')
            name, content = value.split('=', 1)
            labels[name] = content
        index += 2
    if index >= len(argv) or not argv[index + 1:]:
        raise ValueError('Canonical command has no expected image and model arguments')
    if len(options.get('--name', [])) != 1 or len(options.get('--entrypoint', [])) != 1:
        raise ValueError('Canonical command requires one name and entrypoint')
    env.update(environment_map(supplied_env, 'Launcher'))
    return {'name': options['--name'][0], 'image': image_id,
            'cmd': argv[index + 1:], 'entrypoint': options['--entrypoint'],
            'env': env, 'mounts': mounts, 'labels': labels,
            'working_dir': config.get('WorkingDir') or '', 'user': config.get('User') or ''}


def validate_container_spec(container, expected):
    """Compare effective configuration; mismatch errors never print secret values."""
    config = container.get('Config', {})
    if (container.get('Name', '').lstrip('/') != expected['name']
            or container.get('Image') != expected['image']):
        raise ValueError('Container identity differs from the canonical launch')
    if config.get('Cmd') != expected['cmd']:
        raise ValueError('Container model arguments differ from the canonical launch')
    if config.get('Entrypoint') != expected['entrypoint']:
        raise ValueError('Container entrypoint differs from the canonical launch')
    actual_env = environment_map(config.get('Env'), 'Container')
    if actual_env != expected['env']:
        names = sorted(name for name in actual_env.keys() | expected['env'].keys()
                       if actual_env.get(name) != expected['env'].get(name))
        raise ValueError('Container environment differs for names: ' + ', '.join(names))
    actual_mounts = {}
    for mount in container.get('Mounts', []):
        destination = mount.get('Destination')
        if not isinstance(destination, str) or destination in actual_mounts:
            raise ValueError('Container mounts have missing or duplicate destinations')
        actual_mounts[destination] = {key: mount.get(key) for key in ('Source', 'Type', 'RW')}
        if type(actual_mounts[destination]['RW']) is not bool:
            raise ValueError('Container mount access must be explicit')
    if actual_mounts != expected['mounts']:
        raise ValueError('Container bind mounts or access modes differ from the canonical launch')
    if (config.get('WorkingDir', '') != expected['working_dir']
            or config.get('User', '') != expected['user']):
        raise ValueError('Container working directory or user differs from the image contract')
    if (config.get('Labels') or {}) != expected['labels']:
        raise ValueError('Container labels differ from the canonical launch')


def canonical_container_spec(launch, image_receipt, rank, image, *, run=subprocess.run):
    """Regenerate trusted launch inputs and inspect them without creating a container."""
    profile = managed_units.service.mesh_profile
    site, _, _ = profile.load_site(launch / 'site.json')
    with tempfile.TemporaryDirectory(prefix='sparkring-container-spec-') as temporary:
        rendered = Path(temporary) / 'launch'
        profile.render(launch / 'site.json', Path(site['bundle_root']), rendered, image_receipt)
        for name in ('launch-rank.sh', f'rank{rank}.env'):
            supplied = launch / name
            if not stat.S_ISREG(supplied.lstat().st_mode):
                raise ValueError('Supplied launch files must be regular files, not symlinks')
            if supplied.read_bytes() != (rendered / name).read_bytes():
                raise ValueError(f'Supplied {name} differs from the canonical rendered profile')
        result = run(['/bin/bash', str(rendered / 'launch-rank.sh'), str(rank),
                      str(rendered / f'rank{rank}.env')],
                     env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LC_ALL': 'C',
                          'SPARKRING_CREATE_ONLY': '1', 'SPARKRING_PRINT_CONTAINER_SPEC': '1'},
                     capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise ValueError('Canonical launcher failed its read-only container-spec check')
        output = json.loads(result.stdout)
        if output.get('schema') != 'sparkring-container-command/v1':
            raise ValueError('Canonical launcher returned an unsupported container-spec schema')
        return expected_container_spec(output.get('argv'), image)


def prepare_plan(launch, image_receipt, rank, epoch, health_port, key_file):
    if rank not in range(4) or not re.fullmatch('[0-9a-f]{32}', epoch):
        raise ValueError('Rank and common 128-bit hexadecimal epoch are required')
    if not 1024 <= health_port <= 65535:
        raise ValueError('Invalid health port')
    profile = managed_units.service.mesh_profile
    site, _, _ = profile.load_site(launch / 'site.json')
    receipt = profile.load_image_receipt(image_receipt)
    inside = receipt['inside_image']
    if (inside.get('marker_source_sha256') != profile.PINS['marker']['source_sha256']
            or inside.get('marker_binary_sha256') != site['marker_binary_sha256']):
        raise ValueError('Managed profile requires the source-pinned image and host marker')
    if not inside.get('readiness_warmup'):
        raise ValueError('Managed MTP3 profile requires the temperature-one readiness image')
    result = subprocess.run(['docker', 'inspect', site['container_prefix'] + f'-r{rank}'],
                            capture_output=True, text=True, check=True, timeout=10)
    container = json.loads(result.stdout)[0]
    validate_container(container, site, rank, receipt['image_id'])
    result = subprocess.run(['docker', 'image', 'inspect', receipt['image_id']],
                            capture_output=True, text=True, check=True, timeout=10)
    image = json.loads(result.stdout)[0]
    if image.get('Id') != receipt['image_id']:
        raise ValueError('Inspected image differs from the verified image receipt')
    expected = canonical_container_spec(launch, image_receipt, rank, image)
    validate_container_spec(container, expected)
    if profile.sha(Path(site['marker_binary'])) != site['marker_binary_sha256']:
        raise ValueError('Extracted host marker hash differs from the image receipt')
    help_result = subprocess.run([site['marker_binary'], '--help'], capture_output=True, text=True, check=True, timeout=5)
    if '--managed' not in help_result.stdout + help_result.stderr:
        raise ValueError('Extracted helper does not expose managed lifetime')
    bundle = Path(site['bundle_root'])
    if profile.sha(bundle / 'sparkring-overlay-manifest.json') != profile.PINS['canonical_bundle_manifest_sha256']:
        raise ValueError('Host transport bundle manifest differs')
    manifest = json.loads((bundle / 'sparkring-overlay-manifest.json').read_text())
    for item in manifest['files']:
        if profile.sha(profile.manifest_file(bundle, item['path'])) != item['sha256']:
            raise ValueError('Host bundle entry differs')
    config = {'schema': managed_units.service.PROTOCOL, 'site_path': str(CONFIG_DIR / 'site.json'),
              'rank': rank, 'key_file': str(CONFIG_DIR / 'health.key'), 'epoch': epoch,
              'health_port': health_port, 'state_dir': '/run/sparkring-mesh',
              'container_id': container['Id'], 'container_image': receipt['image_id']}
    units = managed_units.unit_text(str(CODE_DIR), str(CONFIG_DIR), container['Id'])
    return {'schema': 'sparkring-managed-install/v1', 'rank': rank, 'config': config,
            'units': units, 'source_files': SOURCE_FILES, 'source_hashes': source_hashes(source_payloads()),
            'source_root': str(ROOT),
            'code_dir': str(CODE_DIR), 'config_dir': str(CONFIG_DIR), 'unit_dir': str(UNIT_DIR),
            'key_source': str(key_file), 'image': receipt['image_id'], 'applied': False}


def apply(plan, launch, key_file):
    if os.geteuid() != 0:
        raise PermissionError('Installing managed services requires root')
    key = managed_units.service.read_key(key_file)
    payloads = source_payloads()
    if tuple(plan['source_files']) != SOURCE_FILES or source_hashes(payloads) != plan['source_hashes']:
        raise ValueError('Managed source differs from the reviewed installation plan')
    targets = [CODE_DIR, CONFIG_DIR, *[UNIT_DIR / name for name in plan['units']]]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise ValueError('Installation target exists; preserve and remove only a reviewed inactive deployment before reinstalling')
    installed_hashes = install_code(payloads, CODE_DIR)
    CONFIG_DIR.mkdir(mode=0o700, parents=True)
    for name in ('site.json', 'fabric.json'):
        shutil.copyfile(launch / name, CONFIG_DIR / name)
        (CONFIG_DIR / name).chmod(0o600)
    (CONFIG_DIR / 'service.json').write_text(json.dumps(plan['config'], indent=2) + '\n')
    (CONFIG_DIR / 'service.json').chmod(0o600)
    (CONFIG_DIR / 'health.key').write_bytes(key)
    (CONFIG_DIR / 'health.key').chmod(0o600)
    for name, content in plan['units'].items():
        (UNIT_DIR / name).write_text(content)
        (UNIT_DIR / name).chmod(0o644)
    subprocess.run(['systemd-analyze', 'verify', *[str(UNIT_DIR / name) for name in plan['units']]], check=True)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    receipt = {key: value for key, value in plan.items() if key != 'key_source'}
    receipt['applied'] = True
    receipt['installed_source_sha256'] = installed_hashes
    receipt['unit_hashes'] = {name: hashlib.sha256((UNIT_DIR / name).read_bytes()).hexdigest() for name in plan['units']}
    receipt['enabled'] = receipt['started'] = False
    (CONFIG_DIR / 'installation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'applied': True, 'started': False, 'enabled': False, 'rank': plan['rank']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--image-receipt', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=range(4), required=True)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--health-port', type=int, default=9975)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = prepare_plan(args.launch, args.image_receipt, args.rank, args.epoch, args.health_port, args.key_file)
    if args.apply:
        apply(plan, args.launch, args.key_file)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
