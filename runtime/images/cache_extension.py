"""Prepare and attest a bounded SparkCache extension over a pinned image."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

SITE = '/opt/venv/lib/python3.12/site-packages'
RECEIPT = '/opt/sparkring/receipts/candidate-installed.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sources(root):
    for path in sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if any(p in {'__pycache__', '.pytest_cache', 'build'} for p in relative.parts) or path.suffix in {'.pyc', '.pyo'}:
            continue
        if path.is_symlink():
            raise ValueError('Source symlink is not admitted: ' + str(relative))
        if path.is_file():
            yield relative.as_posix(), path.read_bytes().replace(b'\r\n', b'\n')


def source_digest(root):
    digest = hashlib.sha256(b'sparkcache-source-tree/v1\x00')
    for name, data in sources(root):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, 'little'))
        digest.update(encoded)
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def pinned_source_bytes(data, expected):
    canonical = data.replace(b'\r\n', b'\n')
    for candidate in (canonical, canonical.replace(b'\n', b'\r\n')):
        if sha(candidate) == expected:
            return candidate
    raise ValueError('Native source differs from its pinned byte identity')


def read_descriptor(path):
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get('schema') != 'sparkring-cache-extension/v1':
        raise ValueError('Unsupported cache extension schema')
    for name in descriptor['python_files']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'sparkcache' or relative.suffix != '.py':
            raise ValueError('Only contained SparkCache Python source is admitted')
    if descriptor['native']['destination'] != '/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so':
        raise ValueError('Only the snapshot library may be replaced')
    return descriptor


def prepare(descriptor_path, source, output):
    descriptor = read_descriptor(descriptor_path)
    source, output = Path(source).resolve(), Path(output).resolve()
    if source_digest(source / 'sparkcache') != descriptor['source']['package_sha256']:
        raise ValueError('SparkCache source tree differs from the pinned extension')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Build context must be empty')
    if output == source or output.is_relative_to(source):
        raise ValueError('Build context must be outside the source checkout')
    payloads = {}
    for name, expected in descriptor['python_files'].items():
        data = (source / name).read_bytes().replace(b'\r\n', b'\n')
        if sha(data) != expected:
            raise ValueError('Pinned Python source mismatch: ' + name)
        payloads[name] = data
    native_inventory = descriptor['native'].get('source_files')
    if native_inventory is not None:
        native_sources = dict(sources(source / 'sparkcache/native'))
        if {'native/' + name for name in native_sources} != set(native_inventory):
            raise ValueError('Native source inventory differs from the descriptor')
        for name, data in native_sources.items():
            payloads['sparkcache/native/' + name] = pinned_source_bytes(data, native_inventory['native/' + name])
    else:
        for name, data in sources(source / 'sparkcache/native'):
            payloads['sparkcache/native/' + name] = data
    output.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    shutil.copyfile(descriptor_path, output / 'descriptor.json')
    shutil.copyfile(Path(descriptor_path).with_name('Dockerfile'), output / 'Dockerfile')
    shutil.copyfile(__file__, output / 'cache_extension.py')
    return output


def verify_parent(descriptor):
    raw = Path(RECEIPT).read_bytes()
    if sha(raw) != descriptor['parent']['receipt_sha256']:
        raise ValueError('Parent installed receipt differs from the pinned R37 image')
    spec = importlib.util.spec_from_file_location('candidate_image', '/opt/sparkring/bin/candidate-image.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify()
    return json.loads(raw), verifier


def install(context, native_build):
    context = Path(context)
    descriptor = read_descriptor(context / 'descriptor.json')
    receipt, verifier = verify_parent(descriptor)
    payloads = {}
    for name, expected in descriptor['python_files'].items():
        data = (context / 'source' / name).read_bytes()
        if sha(data) != expected:
            raise ValueError('Extension source mismatch: ' + name)
        compile(data, name, 'exec')
        payloads[SITE + '/' + name] = data
    native = descriptor['native']
    data = (Path(native_build) / 'libspark_cache_snapshot.so').read_bytes()
    if sha(data) != native['sha256']:
        raise ValueError('Native build differs from the GPU-tested binary')
    payloads[native['destination']] = data
    placement = Path('/opt/sparkring/sparkcache/lib/libspark_cache_placement.so')
    if sha(placement.read_bytes()) != descriptor['retained_placement_sha256']:
        raise ValueError('Retained placement library mismatch')
    changes = []
    for name, data in payloads.items():
        previous = sha(Path(name).read_bytes())
        if receipt['files'].get(name) != previous:
            raise ValueError('Replacement lacks verified inherited ownership: ' + name)
        changes.append({'path': name, 'parent_sha256': previous, 'installed_sha256': sha(data)})
    for name, data in payloads.items():
        Path(name).write_bytes(data)
        receipt['files'][name] = sha(data)
    receipt['cache_extension'] = {'descriptor_sha256': sha((context / 'descriptor.json').read_bytes()),
        'id': descriptor['id'], 'source': descriptor['source'], 'changes': changes,
        'native': native, 'status': 'research-only'}
    Path(RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    verifier.verify()
    from sparkcache.streaming.manager_page_native_ring_ctypes import CtypesManagerPageRingBackend
    CtypesManagerPageRingBackend(native['destination'], expected_sha256=native['sha256'])
    from sparkcache.runtime_patches.generic_connector_contract import verify_connector_job_contract
    verify_connector_job_contract(Path(SITE), Path(descriptor['vllm_contract']))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    prepare_cmd = commands.add_parser('prepare')
    prepare_cmd.add_argument('--descriptor', required=True, type=Path)
    prepare_cmd.add_argument('--source', required=True, type=Path)
    prepare_cmd.add_argument('--output', required=True, type=Path)
    parent_cmd = commands.add_parser('verify-parent')
    parent_cmd.add_argument('--descriptor', required=True, type=Path)
    install_cmd = commands.add_parser('install')
    install_cmd.add_argument('--context', required=True, type=Path)
    install_cmd.add_argument('--native-build', required=True, type=Path)
    options = parser.parse_args()
    if options.action == 'prepare':
        print(prepare(options.descriptor, options.source, options.output))
    elif options.action == 'verify-parent':
        verify_parent(read_descriptor(options.descriptor))
    else:
        install(options.context, options.native_build)


if __name__ == '__main__':
    main()
