"""Admit a cache extension only over the exact verified parent inventory."""
import hashlib
from pathlib import Path
from runtime.common import candidate

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / 'runtime/images/compositions/lil-r37-cache64/descriptor.json'
SITE = '/opt/venv/lib/python3.12/site-packages/'


def descriptor():
    return candidate._read(DESCRIPTOR.read_bytes())


def validate(image_id, installed_bytes, parent_bytes, verification):
    contract = descriptor()
    if hashlib.sha256(parent_bytes).hexdigest() != contract['parent']['receipt_sha256']:
        raise ValueError('Cache extension parent receipt is not pinned R37')
    parent = candidate._read(parent_bytes)
    installed = candidate._read(installed_bytes)
    extension = installed.get('cache_extension', {})
    if (extension.get('descriptor_sha256') != hashlib.sha256(DESCRIPTOR.read_bytes()).hexdigest()
            or extension.get('id') != contract['id']
            or extension.get('source') != contract['source']
            or extension.get('native') != contract['native']):
        raise ValueError('Cache extension receipt differs from its trusted descriptor')
    files = dict(parent['files'])
    replacements = {SITE + name: digest for name, digest in contract['python_files'].items()}
    replacements[contract['native']['destination']] = contract['native']['sha256']
    if not set(replacements).issubset(files):
        raise ValueError('Cache extension replaces unowned parent paths')
    files.update(replacements)
    if installed.get('files') != files:
        raise ValueError('Cache extension changed files outside its pinned inventory')
    for field in ('schema', 'composition_id', 'parent_image_id', 'components', 'versions', 'removed_authored_files', 'integration_contracts'):
        if installed.get(field) != parent.get(field):
            raise ValueError('Cache extension changed inherited contract: ' + field)
    base, _ = candidate.composition(parent['composition_id'])
    return candidate.validate(image_id=image_id, platform='linux/arm64',
        installed_bytes=installed_bytes, verification=verification, descriptor=base,
        native_hashes={name: files[name] for name in candidate.NATIVE},
        expected_entrypoint_sha256=files[candidate.ENTRYPOINT])
