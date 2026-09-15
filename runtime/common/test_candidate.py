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


def host_receipt(inputs, monkeypatch):
    from runtime.common import candidate
    installed = json.loads(inputs['installed_bytes'])
    installed['files'][candidate.ENTRYPOINT] = hashlib.sha256((candidate.ROOT / 'runtime/images/candidate_image.py').read_bytes()).hexdigest()
    installed['files']['/opt/sparkring/sircl/python/sparkring-overlay-manifest.json'] = '8' * 64
    # Use the real GLM native contract so the rendered cache settings are realistic.
    native = json.loads((candidate.ROOT / 'runtime/images/compositions/lil-r37-glm-spark/baseline-native.json').read_text())
    installed['files'].update(native)
    monkeypatch.setattr(candidate, 'composition', lambda identity: (inputs['descriptor'], native))
    monkeypatch.setattr(candidate, 'registered_artifacts', lambda identity: {
        'schema': 'sparkring-runtime-artifacts/v1', 'image_id': inputs['image_id'],
        'files': {candidate.ENTRYPOINT: installed['files'][candidate.ENTRYPOINT]}})
    raw = json.dumps(installed).encode()
    verification = dict(inputs['verification'], receipt_sha256=hashlib.sha256(raw).hexdigest(), files_verified=len(installed['files']))
    return candidate.make_receipt(inputs['image_id'], raw, verification)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('cache', [False, True])
def test_explicit_candidate_tp2_uses_own_entrypoint_and_lease(inputs, monkeypatch, tmp_path, rank, cache):
    from runtime.common import candidate, tp2
    document = host_receipt(inputs, monkeypatch)
    model, cache_dir = tmp_path / 'model', tmp_path / 'cache'
    model.mkdir()
    cache_dir.mkdir()
    (model / 'config.json').write_text('{}')
    env = tmp_path / 'rank.env'
    env.write_text('VLLM_HOST_IP=192.0.2.10\nNCCL_SOCKET_IFNAME=eth0\nGLOO_SOCKET_IFNAME=eth0\n')
    plan = tp2.render(rank, '192.0.2.10', model, cache_dir, env, document['image_id'], document, r33_sparkcache=cache)
    assert plan['container_args'][:2] == [candidate.ENTRYPOINT, 'serve']
    assert plan['runtime_kind'] == 'fixture-candidate'
    assert '--gdn-decode-kernel' not in plan['container_args']
    assert ('--headless' in plan['container_args']) == bool(rank)
    assert ('--health-cmd' in plan['command']) == (rank == 0)
    tp2.validate_runtime_receipt(document, plan)
    if cache:
        assert plan['environment']['SPARKCACHE_SOURCE_LEASE_CONTRACT'] == '/opt/sparkring/contracts/fixture.json'
        assert 'fixture' in plan['environment']['SPARKCACHE_CACHE_NAMESPACE']
        assert plan['qualification']['gpu_qualified'] is False


def test_candidate_generated_launcher_preserves_frozen_source(inputs, monkeypatch):
    from runtime.common import candidate
    document = host_receipt(inputs, monkeypatch)
    source = (candidate.ROOT / 'runtime/glm53-flash-jj-r8-gb10/launch-rank.sh').read_text()
    rendered = candidate.adapt_launcher(source, document['installed'])
    assert 'candidate) release_lease_contract=/opt/sparkring/contracts/fixture.json ;;' in rendered
    assert 'serving_prefix=(/opt/sparkring/bin/candidate-image.py serve)' in rendered
    assert 'serving_prefix=(/opt/sparkring/bin/sparkring serve)' in rendered
    assert source == (candidate.ROOT / 'runtime/glm53-flash-jj-r8-gb10/launch-rank.sh').read_text()
    with pytest.raises(ValueError, match='changed'):
        candidate.adapt_launcher(rendered, document['installed'])


def test_host_receipt_rejects_divergent_parsed_and_raw_data(inputs, monkeypatch):
    from runtime.common import candidate
    document = host_receipt(inputs, monkeypatch)
    document['installed']['composition_id'] = 'other'
    with pytest.raises(ValueError, match='raw installed'):
        candidate.validate_receipt(document)


def test_registered_entrypoint_identity_survives_checkout_changes(inputs, monkeypatch):
    from runtime.common import candidate
    document = host_receipt(inputs, monkeypatch)
    original = Path.read_bytes
    helper = candidate.ROOT / 'runtime/images/candidate_image.py'
    def changed_helper(path):
        if path == helper:
            return b'# different helper with LF\n'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', changed_helper)
    assert candidate.validate_receipt(document)['image_id'] == document['image_id']


def test_pinned_entrypoint_rejects_candidate_payload_change(inputs):
    inputs['expected_entrypoint_sha256'] = hashlib.sha256(inputs.pop('entrypoint_bytes')).hexdigest()
    m.validate(**inputs)
    mutate(inputs, lambda d: d['files'].update({m.ENTRYPOINT: '9' * 64}))
    with pytest.raises(ValueError, match='entrypoint differs'):
        m.validate(**inputs)


@pytest.mark.parametrize('pin', ['invalid', '', 12])
def test_pinned_entrypoint_requires_valid_hash(inputs, pin):
    inputs.pop('entrypoint_bytes')
    inputs['expected_entrypoint_sha256'] = pin
    with pytest.raises(ValueError, match='valid reviewed'):
        m.validate(**inputs)
