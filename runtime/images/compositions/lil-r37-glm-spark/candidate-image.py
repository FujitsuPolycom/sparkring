"""Assemble and verify source-overlay images; deployment profiles own serving admission."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

SITE = '/opt/venv/lib/python3.12/site-packages'
RECEIPT = '/opt/sparkring/receipts/candidate-installed.json'
ENTRYPOINT = '/opt/sparkring/bin/candidate-image.py'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def located(root, absolute):
    path = PurePosixPath(absolute)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Receipt path must be absolute and contained')
    target = root / str(path).lstrip('/')
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes filesystem root')
    return target


def authored(path, component):
    relative = PurePosixPath(path)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in path
            or not relative.parts or relative.parts[0] != component
            or any(suffix in relative.name for suffix in ('.so', '.dll', '.dylib', '.a', '.o'))):
        raise ValueError('Only non-native authored package files may be overlaid or removed')
    return relative



def integration_contracts(context, root, descriptor, inventory):
    """Validate additional JSON contracts without replacing inherited contracts."""
    records = descriptor.get('integration_contracts', {})
    if not isinstance(records, dict):
        raise ValueError('integration_contracts must be a mapping')
    payload = {}
    for destination, record in records.items():
        if (not isinstance(destination, str) or not re.fullmatch(
                r'/opt/sparkring/contracts/[A-Za-z0-9][A-Za-z0-9_.-]*\.json', destination)):
            raise ValueError('Integration contract destination must be a JSON basename under contracts')
        if not isinstance(record, dict) or set(record) != {'file', 'sha256'}:
            raise ValueError('Integration contract requires exactly file and sha256')
        filename, expected = record['file'], record['sha256']
        if not isinstance(filename, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.json', filename):
            raise ValueError('Integration contract source must be a context JSON basename')
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('Integration contract requires a lowercase SHA256 digest')
        target = located(root, destination)
        if destination in inventory or target.exists() or target.is_symlink():
            raise ValueError('Integration contract must use a new destination; overwrite refused')
        source = context / filename
        if source.is_symlink() or not source.is_file():
            raise ValueError('Integration contract source must be a regular context file')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Integration contract SHA256 mismatch')
        if not isinstance(read(source), dict):
            raise ValueError('Integration contract JSON must be an object')
        payload[destination] = data
    return payload


def install(context, filesystem_root=Path('/')):
    context, root = Path(context), Path(filesystem_root)
    descriptor = read(context / 'descriptor.json')
    parent = read(context / 'parent-installed.json')
    if descriptor.get('schema') != 'sparkring-candidate-image/v1':
        raise ValueError('Unsupported candidate descriptor')
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,100}', descriptor.get('composition_id', '')):
        raise ValueError('Invalid composition identity')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', descriptor.get('parent_image_id', '')):
        raise ValueError('Exact parent image ID required')
    version = descriptor.get('distribution_version', '')
    if not re.fullmatch(r'[0-9][A-Za-z0-9.!+_-]*', version):
        raise ValueError('Explicit distribution version required')
    if parent.get('schema') not in ('sparkring-r35-installed/v1', 'sparkring-candidate-installed/v1'):
        raise ValueError('Unsupported parent installed receipt')
    if set(descriptor.get('components', {})) != {'vllm', 'b12x'}:
        raise ValueError('Exactly vllm and b12x are required')
    inventory = parent.get('files', {})
    if not inventory:
        raise ValueError('Parent file inventory is empty')
    for name, expected in inventory.items():
        if digest(located(root, name)) != expected:
            raise ValueError('Parent payload mismatch: ' + name)
    contracts = integration_contracts(context, root, descriptor, inventory)
    payload = {}
    removed = set()
    for component, record in descriptor['components'].items():
        archive = context / record['archive']
        if archive.parent.resolve() != context.resolve() or digest(archive) != record['archive_sha256']:
            raise ValueError('Source archive identity mismatch')
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                if not member.name.startswith(component + '/') or member.isdir():
                    continue
                relative = authored(member.name, component)
                if not member.isfile() or member.name in payload:
                    raise ValueError('Source archive requires unique regular package files')
                payload[str(relative)] = (source.extractfile(member).read(), member.mode & 0o777)
        old_files = descriptor.get('parent_authored_files', {}).get(component)
        if not isinstance(old_files, list) or not old_files:
            raise ValueError('Explicit parent authored-file inventory required')
        for name in old_files:
            authored(name, component)
            absolute = SITE + '/' + name
            if absolute not in inventory:
                raise ValueError('Parent authored file lacks inherited hash: ' + name)
            if name not in payload:
                removed.add(absolute)
    if not all(any(name.startswith(component + '/') for name in payload) for component in ('vllm', 'b12x')):
        raise ValueError('Archive is missing an authored package')
    metadata_paths = {}
    for component in ('vllm', 'b12x'):
        matches = list(located(root, SITE).glob(component + '-*.dist-info/METADATA'))
        if len(matches) != 1:
            raise ValueError('Exactly one installed distribution is required: ' + component)
        metadata_paths[component] = matches[0]
    changed = set()
    for name, data in contracts.items():
        target = located(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        changed.add(name)
    for name, (data, mode) in payload.items():
        absolute = SITE + '/' + name
        target = located(root, absolute)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)
        changed.add(absolute)
    for name in removed:
        located(root, name).unlink()
    version_path = located(root, SITE + '/vllm/_version.py')
    version_path.write_text(f'__version__ = version = {version!r}\n__version_tuple__ = version_tuple = ({version!r},)\n__commit_id__ = commit_id = None\n', encoding='utf-8')
    changed.add(SITE + '/vllm/_version.py')
    meta = metadata_paths['vllm']
    text, count = re.subn(r'^Version: [^\n]+$', 'Version: ' + version, meta.read_text(encoding='utf-8'), flags=re.MULTILINE)
    if count != 1:
        raise ValueError('Distribution METADATA must have one version')
    meta.write_text(text, encoding='utf-8')
    changed.add('/' + meta.relative_to(root).as_posix())
    for component, meta in metadata_paths.items():
        record_path = meta.with_name('RECORD')
        with record_path.open(newline='', encoding='utf-8') as stream:
            names = {row[0] for row in csv.reader(stream)}
        names.update(name for name in payload if name.startswith(component + '/'))
        if component == 'vllm':
            names.add('vllm/_version.py')
        with record_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            for name in sorted(names):
                target = (located(root, SITE) / name).resolve()
                if not target.is_relative_to(located(root, '/opt/venv').resolve()):
                    raise ValueError('Wheel RECORD escapes virtual environment')
                if target == record_path.resolve():
                    writer.writerow((name, '', ''))
                elif target.is_file():
                    value = base64.urlsafe_b64encode(bytes.fromhex(digest(target))).decode().rstrip('=')
                    writer.writerow((name, 'sha256=' + value, target.stat().st_size))
        changed.add('/' + record_path.relative_to(root).as_posix())
    entry = located(root, ENTRYPOINT)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_bytes(Path(__file__).read_bytes())
    changed.add(ENTRYPOINT)
    for name, expected in inventory.items():
        if name not in changed and name not in removed and digest(located(root, name)) != expected:
            raise ValueError('Unrelated inherited bytes changed: ' + name)
    receipt = {
        'schema': 'sparkring-candidate-installed/v1', 'composition_id': descriptor['composition_id'],
        'components': descriptor['components'], 'parent_image_id': descriptor['parent_image_id'],
        'parent_installed_sha256': digest(context / 'parent-installed.json'),
        'integration_contracts': descriptor.get('integration_contracts', {}),
        'files': {name: digest(located(root, name)) for name in sorted((set(inventory) | changed) - removed)},
        'versions': {**parent['versions'], 'vllm': version}, 'removed_authored_files': sorted(removed),
        'qualification': 'Isolated testing candidate; no model or cache capability qualification transferred.',
    }
    destination = located(root, RECEIPT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return receipt


def verify(filesystem_root=Path('/'), version_reader=importlib.metadata.version):
    root = Path(filesystem_root)
    path = located(root, RECEIPT)
    receipt = read(path)
    if receipt.get('schema') != 'sparkring-candidate-installed/v1' or not receipt.get('files'):
        raise ValueError('Unsupported or empty candidate installed receipt')
    for name, expected in receipt['files'].items():
        if digest(located(root, name)) != expected:
            raise ValueError('Candidate payload mismatch: ' + name)
    for name in receipt['removed_authored_files']:
        if located(root, name).exists():
            raise ValueError('Removed authored file reappeared: ' + name)
    for name, expected in receipt['versions'].items():
        if version_reader(name) != expected:
            raise ValueError('Candidate distribution version mismatch: ' + name)
    return {'schema': 'sparkring-candidate-verification/v1', 'receipt_sha256': digest(path),
            'files_verified': len(receipt['files']), 'serving_qualified': False,
            'source_components': receipt['components']}


def main():
    if sys.argv[1:2] == ['--install-context']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--install-context', required=True, type=Path)
        install(parser.parse_args().install_context)
        return
    if sys.argv[1:2] in ([], ['--help'], ['-h']):
        print('Usage: candidate-image verify | serve MODEL [vLLM serve arguments]. Deployment profiles own admission.')
        return
    if sys.argv[1:2] not in (['verify'], ['serve']):
        raise ValueError('Expected verify or serve')
    result = verify()
    if sys.argv[1:] == ['verify']:
        print(json.dumps(result, indent=2))
    elif sys.argv[1] == 'serve':
        command = ['/opt/venv/bin/python', '-m', 'vllm.entrypoints.cli.main', 'serve', *sys.argv[2:]]
        os.execve(command[0], command, os.environ)
    else:
        raise ValueError('verify accepts no additional arguments')


if __name__ == '__main__':
    main()
