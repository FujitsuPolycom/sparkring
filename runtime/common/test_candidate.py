"""Candidate host admission rejects self-consistent but untrusted receipt changes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('candidate', Path(__file__).with_name('candidate.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def inputs():
    records = {}
    for name in ('vllm', 'b12x'):
        records[name] = {'base_commit': 'a' * 40, 'base_tree': 'b' * 40, 'tree': 'c' * 40,
                         'archive': name + '-' + 'c' * 40 + '.tar.gz', 'archive_sha256': 'd' * 64,
                         'patch': name + '-sparkring.patch', 'patch_sha256': 'e' * 64,
                         'native_comparison': {'reference': 'f' * 40, 'compared_tree': 'c' * 40,
                                               'paths': ['csrc'], 'unchanged': True}}
    descriptor = {'schema': 'sparkring-candidate-image/v1', 'composition_id': 'fixture',
                  'parent_image_id': 'sha256:' + '1' * 64, 'components': records,
                  'distribution_version': '1.0+fixture', 'integration_contracts': {
                      '/opt/sparkring/contracts/fixture.json': {'file': 'fixture.json', 'sha256': '4' * 64}}}
    files = {name: '2' * 64 for name in m.NATIVE}
    files[m.ENTRYPOINT] = hashlib.sha256(b'fixture entrypoint').hexdigest()
    files['/opt/sparkring/contracts/fixture.json'] = '4' * 64
    installed = {**copy.deepcopy(descriptor), 'schema': 'sparkring-candidate-installed/v1',
                 'files': files, 'versions': {'vllm': '1.0+fixture'}, 'removed_authored_files': []}
    raw = json.dumps(installed).encode()
    verification = {'schema': 'sparkring-candidate-verification/v1',
                    'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'files_verified': len(files),
                    'serving_qualified': False, 'source_components': copy.deepcopy(records)}
    return {'image_id': 'sha256:' + '3' * 64, 'platform': 'linux/arm64', 'installed_bytes': raw,
            'verification': verification, 'descriptor': descriptor,
            'native_hashes': {name: '2' * 64 for name in m.NATIVE}, 'entrypoint_bytes': b'fixture entrypoint'}


def mutate(inputs, update):
    installed = json.loads(inputs['installed_bytes'])
    update(installed)
    inputs['installed_bytes'] = json.dumps(installed).encode()
    inputs['verification']['receipt_sha256'] = hashlib.sha256(inputs['installed_bytes']).hexdigest()
    inputs['verification']['files_verified'] = len(installed['files'])
    inputs['verification']['source_components'] = copy.deepcopy(installed['components'])


def test_matching_recorded_inputs(inputs):
    assert m.validate(**inputs)['serving_qualified'] is False


@pytest.mark.parametrize('field,value', [('image_id', 'tag:latest'), ('platform', 'linux/amd64'),
                                        ('installed_bytes', '{}')])
def test_identity_and_raw_bytes_required(inputs, field, value):
    inputs[field] = value
    with pytest.raises(ValueError):
        m.validate(**inputs)


def test_raw_bytes_hash_not_reserialization(inputs):
    inputs['installed_bytes'] += b'\n'
    with pytest.raises(ValueError, match='Verification'):
        m.validate(**inputs)


@pytest.mark.parametrize('field,value', [('composition_id', 'other'), ('parent_image_id', 'sha256:' + '5' * 64),
                                        ('integration_contracts', {})])
def test_rehashed_receipt_cannot_change_trusted_composition(inputs, field, value):
    mutate(inputs, lambda d: d.update({field: value}))
    with pytest.raises(ValueError):
        m.validate(**inputs)


def test_rehashed_receipt_cannot_change_source_identity(inputs):
    mutate(inputs, lambda d: d['components']['vllm'].update(tree='f' * 40))
    with pytest.raises(ValueError, match='components'):
        m.validate(**inputs)


@pytest.mark.parametrize('path', [*sorted(m.NATIVE), m.ENTRYPOINT, '/opt/sparkring/contracts/fixture.json'])
def test_rehashed_receipt_cannot_replace_bound_files(inputs, path):
    mutate(inputs, lambda d: d['files'].update({path: '6' * 64}))
    with pytest.raises(ValueError):
        m.validate(**inputs)


@pytest.mark.parametrize('field,value', [('files_verified', 0), ('files_verified', True),
                                        ('serving_qualified', True), ('schema', 'other')])
def test_bad_verification(inputs, field, value):
    inputs['verification'][field] = value
    with pytest.raises(ValueError):
        m.validate(**inputs)


def test_incomplete_baseline_rejected(inputs):
    inputs['native_hashes'] = {}
    with pytest.raises(ValueError, match='mandatory baseline'):
        m.validate(**inputs)


def test_escaping_inventory_rejected(inputs):
    mutate(inputs, lambda d: d['files'].update({'/opt/../escape': '7' * 64}))
    with pytest.raises(ValueError, match='inventory'):
        m.validate(**inputs)


def test_duplicate_receipt_key_rejected(inputs):
    inputs['installed_bytes'] = b'{"files":{},"files":{}}'
    with pytest.raises(ValueError, match='Duplicate'):
        m.validate(**inputs)
