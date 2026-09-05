"""Install one rank's reviewed managed-mesh contract; plan-only unless --apply is supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

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
