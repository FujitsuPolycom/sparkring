"""Validate candidate-image receipts against trusted composition inputs offline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ENTRYPOINT = '/opt/sparkring/bin/candidate-image.py'
NATIVE = {
    '/opt/lmcache/lib/liblmcache_cumem_shareable.so',
    '/opt/local-inference/nccl/lib/libnccl.so.2.31.2',
    '/opt/sparkring/sircl/libspark_transport_capi.so',
    '/opt/sparkring/sparkcache/lib/libspark_cache_placement.so',
    '/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so',
}


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def _oid(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', value)


def _path(value):
    return (isinstance(value, str) and value.startswith('/') and '\\' not in value
            and '..' not in PurePosixPath(value).parts and str(PurePosixPath(value)) == value)


def _read(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate installed receipt key: ' + key)
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError('Installed receipt must be an object')
    return result


def validate(*, image_id, platform, installed_bytes, verification, descriptor,
             native_hashes, entrypoint_bytes=None):
    """Bind recorded verification to trusted inputs; no live-image claim is made.

    The caller must obtain image_id/platform and verification from the same
    actual image before lifecycle mutation. Descriptor and baseline native
    hashes are trusted review inputs, not claims accepted from the candidate.
    """
    if not isinstance(image_id, str) or not re.fullmatch('sha256:[0-9a-f]{64}', image_id):
        raise ValueError('Exact candidate image ID required')
    if platform != 'linux/arm64':
        raise ValueError('Candidate platform must be linux/arm64')
    if not isinstance(installed_bytes, bytes):
        raise ValueError('Raw installed receipt bytes required')
    installed = _read(installed_bytes)
    if installed.get('schema') != 'sparkring-candidate-installed/v1':
        raise ValueError('Unsupported installed receipt schema')
    if not isinstance(descriptor, dict) or descriptor.get('schema') != 'sparkring-candidate-image/v1':
        raise ValueError('Unsupported trusted descriptor schema')
    if not isinstance(verification, dict) or verification.get('schema') != 'sparkring-candidate-verification/v1':
        raise ValueError('Unsupported verification schema')
    identity = descriptor.get('composition_id')
    if not isinstance(identity, str) or not re.fullmatch('[a-z0-9][a-z0-9.-]{0,100}', identity):
        raise ValueError('Invalid composition identity')
    parent = descriptor.get('parent_image_id')
    if not isinstance(parent, str) or not re.fullmatch('sha256:[0-9a-f]{64}', parent):
        raise ValueError('Exact parent image ID required')
    for field in ('composition_id', 'parent_image_id', 'components'):
        if installed.get(field) != descriptor.get(field):
            raise ValueError('Candidate differs from trusted descriptor: ' + field)
    components = descriptor.get('components')
    if not isinstance(components, dict) or set(components) != {'vllm', 'b12x'}:
        raise ValueError('Exact vllm/b12x component inventory required')
    for name, record in components.items():
        if not isinstance(record, dict):
            raise ValueError('Invalid component record')
        if any(not _oid(record.get(key)) for key in ('base_commit', 'base_tree', 'tree')):
            raise ValueError('Full component source object IDs required')
        if any(not _hash(record.get(key)) for key in ('archive_sha256', 'patch_sha256')):
            raise ValueError('Component archive and patch hashes required')
        if record.get('archive') != f'{name}-{record["tree"]}.tar.gz' or record.get('patch') != name + '-sparkring.patch':
            raise ValueError('Component archive/patch name differs from identity')
        comparison = record.get('native_comparison', {})
        if (not isinstance(comparison, dict) or comparison.get('unchanged') is not True
                or comparison.get('compared_tree') != record['tree'] or not _oid(comparison.get('reference'))
                or not isinstance(comparison.get('paths'), list) or not comparison['paths']):
            raise ValueError('Integrated native comparison is required')
    files = installed.get('files')
    if not isinstance(files, dict) or not files or any(not _path(p) or not _hash(h) for p, h in files.items()):
        raise ValueError('Invalid installed file inventory')
    receipt_hash = hashlib.sha256(installed_bytes).hexdigest()
    if (verification.get('receipt_sha256') != receipt_hash
            or type(verification.get('files_verified')) is not int
            or verification['files_verified'] != len(files)
            or verification.get('serving_qualified') is not False
            or verification.get('source_components') != components):
        raise ValueError('Verification does not bind the unqualified installed payload')
    if not isinstance(native_hashes, dict) or not NATIVE.issubset(native_hashes):
        raise ValueError('Complete mandatory baseline native hashes required')
    for path, expected in native_hashes.items():
        if not _path(path) or not _hash(expected) or files.get(path) != expected:
            raise ValueError('Inherited native identity mismatch: ' + str(path))
    contracts = descriptor.get('integration_contracts', {})
    if not isinstance(contracts, dict) or installed.get('integration_contracts', {}) != contracts:
        raise ValueError('Integration contracts differ from trusted descriptor')
    for destination, record in contracts.items():
        if (not isinstance(destination, str) or not re.fullmatch(r'/opt/sparkring/contracts/[A-Za-z0-9][A-Za-z0-9_.-]*\.json', destination)
                or not isinstance(record, dict) or set(record) != {'file', 'sha256'}
                or not isinstance(record['file'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.json', record['file'])
                or not _hash(record['sha256']) or files.get(destination) != record['sha256']):
            raise ValueError('Invalid or unbound integration contract')
    versions = installed.get('versions')
    if (not isinstance(versions, dict) or not isinstance(descriptor.get('distribution_version'), str)
            or not descriptor['distribution_version'] or versions.get('vllm') != descriptor['distribution_version']):
        raise ValueError('Distribution version differs from descriptor')
    removed = installed.get('removed_authored_files')
    if not isinstance(removed, list) or any(not _path(p) or p in files for p in removed):
        raise ValueError('Invalid removed file inventory')
    if entrypoint_bytes is None:
        entrypoint_bytes = (Path(__file__).resolve().parents[1] / 'images/candidate_image.py').read_bytes()
    if not isinstance(entrypoint_bytes, bytes) or files.get(ENTRYPOINT) != hashlib.sha256(entrypoint_bytes).hexdigest():
        raise ValueError('Candidate entrypoint differs from reviewed implementation')
    return {'schema': 'sparkring-candidate-admission/v1', 'image_id': image_id,
            'composition_id': identity, 'receipt_sha256': receipt_hash, 'serving_qualified': False,
            'scope': 'Recorded payload and trusted composition agree; caller must bind to the live image.'}
