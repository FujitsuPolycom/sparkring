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
             native_hashes, entrypoint_bytes=None, expected_entrypoint_sha256=None):
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
    if expected_entrypoint_sha256 is not None:
        if entrypoint_bytes is not None or not _hash(expected_entrypoint_sha256):
            raise ValueError('Supply one valid reviewed entrypoint identity')
        entrypoint_hash = expected_entrypoint_sha256
    else:
        if entrypoint_bytes is None:
            entrypoint_bytes = (Path(__file__).resolve().parents[1] / 'images/candidate_image.py').read_bytes()
        if not isinstance(entrypoint_bytes, bytes):
            raise ValueError('Reviewed entrypoint bytes required')
        entrypoint_hash = hashlib.sha256(entrypoint_bytes).hexdigest()
    if files.get(ENTRYPOINT) != entrypoint_hash:
        raise ValueError('Candidate entrypoint differs from reviewed implementation')
    return {'schema': 'sparkring-candidate-admission/v1', 'image_id': image_id,
            'composition_id': identity, 'receipt_sha256': receipt_hash, 'serving_qualified': False,
            'scope': 'Recorded payload and trusted composition agree; caller must bind to the live image.'}

# Registered compositions are repository-owned review inputs, never receipt paths.
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'sparkring-candidate-image-receipt/v1'
REGISTERED = {'lil-r37-glm-spark'}


def composition(identity):
    if identity not in REGISTERED:
        raise ValueError('Unregistered candidate composition')
    directory = ROOT / 'runtime/images/compositions' / identity
    return (_read((directory / 'descriptor.json').read_bytes()),
            _read((directory / 'baseline-native.json').read_bytes()))


def registered_artifacts(identity):
    if identity not in REGISTERED:
        raise ValueError('Unregistered candidate composition')
    return _read((ROOT / 'runtime/images/compositions' / identity / 'runtime-artifacts.json').read_bytes())


def make_receipt(image_id, installed_bytes, verification, image_reference=None):
    import base64
    installed = _read(installed_bytes)
    document = {'schema': SCHEMA, 'image_id': image_id, 'image_reference': image_reference or image_id,
                'platform': 'linux/arm64', 'installed': installed,
                'installed_base64': base64.b64encode(installed_bytes).decode(),
                'verification': verification}
    validate_receipt(document)
    return document


def validate_receipt(document):
    import base64
    if not isinstance(document, dict) or document.get('schema') != SCHEMA:
        raise ValueError('Unsupported candidate host receipt')
    try:
        raw = base64.b64decode(document['installed_base64'], validate=True)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('Candidate receipt must preserve exact installed bytes') from exc
    installed = _read(raw)
    if installed != document.get('installed'):
        raise ValueError('Candidate parsed receipt differs from raw installed bytes')
    descriptor, native = composition(installed.get('composition_id'))
    if document.get('image_reference') != document.get('image_id'):
        publication = _read((ROOT / 'runtime/images/compositions' / installed['composition_id'] / 'publication.json').read_bytes())
        if (publication.get('schema') != 'sparkring-image-publication/v1'
                or publication.get('image_reference') != document.get('image_reference')
                or publication.get('image_id') != document.get('image_id')
                or publication.get('platform') != document.get('platform')):
            raise ValueError('Candidate registry reference differs from registered publication')
    runtime_artifacts = registered_artifacts(installed['composition_id'])
    if (runtime_artifacts.get('schema') != 'sparkring-runtime-artifacts/v1'
            or runtime_artifacts.get('image_id') != document.get('image_id')
            or not isinstance(runtime_artifacts.get('files'), dict)
            or not _hash(runtime_artifacts['files'].get(ENTRYPOINT))):
        raise ValueError('Registered runtime artifact identity differs from candidate')
    verification = dict(document.get('verification', {}))
    verification.pop('checked_files', None)
    validate(image_id=document.get('image_id'), platform=document.get('platform'),
             installed_bytes=raw, verification=verification, descriptor=descriptor, native_hashes=native,
             expected_entrypoint_sha256=runtime_artifacts['files'][ENTRYPOINT])
    files = installed['files']
    manifest = files.get('/opt/sparkring/sircl/python/sparkring-overlay-manifest.json')
    if not _hash(manifest):
        raise ValueError('Candidate lacks embedded transport manifest identity')
    return dict(document, bundle_manifest_sha256=manifest,
                verification=dict(verification, checked_files={name: value for name, value in files.items()
                                  if name.startswith(('/opt/sparkring/', '/opt/local-inference/nccl/'))}))


def profile_contract(installed):
    import copy
    contract = copy.deepcopy(_read((ROOT / 'runtime/sparkring/jovian-r33/profiles/profile-contract.json').read_bytes()))
    descriptor, _ = composition(installed['composition_id'])
    lease = list(descriptor.get('integration_contracts', {}))
    if len(lease) != 1:
        raise ValueError('Registered GLM composition requires exactly one lease contract')
    components = installed['components']
    contract['image']['required_sources'].update(
        vllm_head=components['vllm']['base_commit'], vllm_integrated_tree=components['vllm']['tree'],
        b12x_commit=components['b12x']['base_commit'], b12x_tree=components['b12x']['tree'])
    contract['sparkcache_native']['lease_contract'] = lease[0]
    contract['candidate_qualification'] = 'Experimental; no inherited profile qualification is transferred.'
    return contract


def validate_profile_capabilities(document, profile):
    checked = validate_receipt(document)
    if profile not in profile_contract(checked['installed'])['profiles']:
        raise ValueError('Candidate profile is absent from the shared contract')
    # Structural admission permits explicit TP2 testing; it transfers no TP4 result.


def verify_local_image(document, *, run=None):
    import subprocess
    run = subprocess.run if run is None else run
    checked = validate_receipt(document)
    image = checked['image_id']
    actual = json.loads(run(['docker', 'image', 'inspect', image], check=True, capture_output=True, text=True).stdout)[0]
    if actual.get('Id') != image or actual.get('Os') != 'linux' or actual.get('Architecture') != 'arm64':
        raise ValueError('Local candidate image identity/platform differs')
    observed = json.loads(run(['docker', 'run', '--rm', '--network', 'none', '--pull', 'never', image, 'verify'],
                              check=True, capture_output=True, text=True).stdout)
    expected = {key: value for key, value in checked['verification'].items() if key != 'checked_files'}
    if observed != expected:
        raise ValueError('Local candidate payload verification differs')


def adapt_launcher(text, installed):
    descriptor, _ = composition(installed['composition_id'])
    lease = list(descriptor['integration_contracts'])
    if len(lease) != 1:
        raise ValueError('Exactly one registered lease contract required')
    replacements = {
        'r35) release_lease_contract=/opt/sparkring/contracts/vllm-connector-jobs-r35.json ;;':
        'candidate) release_lease_contract=' + lease[0] + ' ;;\n  r35) release_lease_contract=/opt/sparkring/contracts/vllm-connector-jobs-r35.json ;;',
        'if [[ "${SPARKRING_RUNTIME_RELEASE}" == r35 ]]; then':
        'if [[ "${SPARKRING_RUNTIME_RELEASE}" == r35 || "${SPARKRING_RUNTIME_RELEASE}" == candidate ]]; then',
        'serving_prefix=(/opt/sparkring/bin/sparkring serve)':
        'serving_prefix=(/opt/sparkring/bin/sparkring serve)\n      if [[ "${SPARKRING_RUNTIME_RELEASE}" == candidate ]]; then\n        serving_prefix=(/opt/sparkring/bin/candidate-image.py serve)\n      fi',
    }
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise ValueError('Source launcher changed; candidate adaptation requires review')
        text = text.replace(before, after)
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create an offline registered candidate host receipt')
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--composition', choices=sorted(REGISTERED))
    parser.add_argument('--image-reference', help='Optional immutable registered publication reference')
    parser.add_argument('--installed-receipt', required=True, type=Path)
    parser.add_argument('--verification', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = make_receipt(args.image_id, args.installed_receipt.read_bytes(), _read(args.verification.read_bytes()), args.image_reference)
    if args.composition and result['installed']['composition_id'] != args.composition:
        raise ValueError('Requested composition differs from installed receipt')
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
