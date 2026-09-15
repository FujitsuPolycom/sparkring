"""Build and verify tiny source-overlay filesystems without Docker or a GPU."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location('candidate_image', Path(__file__).with_name('candidate_image.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def archive(context, component, entries):
    path = context / (component + '.tar.gz')
    with tarfile.open(path, 'w:gz') as stream:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            stream.addfile(info, io.BytesIO(data))
    return {'archive': path.name, 'archive_sha256': m.digest(path), 'tree': 'a' * 40}


@pytest.fixture
def fixture(tmp_path):
    root, context = tmp_path / 'fs', tmp_path / 'context'
    root.mkdir()
    context.mkdir()
    files = {}
    for component in ('vllm', 'b12x'):
        for name, data in {
            component + '/api.py': b'# parent\n', component + '/obsolete.py': b'# removed\n',
            component + '/native.so': b'unchanged native fixture',
            component + '-1.0.dist-info/METADATA': f'Name: {component}\nVersion: 1.0\n'.encode(),
            component + '-1.0.dist-info/RECORD': f'{component}/api.py,,\n{component}/obsolete.py,,\n{component}/native.so,,\n{component}-1.0.dist-info/METADATA,,\n{component}-1.0.dist-info/RECORD,,\n'.encode(),
        }.items():
            absolute = m.SITE + '/' + name
            path = m.located(root, absolute)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[absolute] = m.digest(path)
    parent = {'schema': 'sparkring-r35-installed/v1', 'files': files, 'versions': {'vllm': '1.0', 'b12x': '1.0'}}
    descriptor = {
        'schema': 'sparkring-candidate-image/v1', 'composition_id': 'vllm-test',
        'parent_image_id': 'sha256:' + 'b' * 64, 'distribution_version': '1.1+candidate',
        'components': {c: archive(context, c, {c + '/api.py': b'# candidate\n'}) for c in ('vllm', 'b12x')},
        'parent_authored_files': {c: [c + '/api.py', c + '/obsolete.py'] for c in ('vllm', 'b12x')},
    }
    (context / 'parent-installed.json').write_text(json.dumps(parent))
    (context / 'descriptor.json').write_text(json.dumps(descriptor))
    return root, context, descriptor, parent


def update(context, descriptor):
    (context / 'descriptor.json').write_text(json.dumps(descriptor))


def test_overlay_preserves_native_and_verifies(fixture):
    root, context, descriptor, parent = fixture
    receipt = m.install(context, root)
    for component in ('vllm', 'b12x'):
        native = m.SITE + '/' + component + '/native.so'
        assert receipt['files'][native] == parent['files'][native]
        assert not m.located(root, m.SITE + '/' + component + '/obsolete.py').exists()
    result = m.verify(root, receipt['versions'].__getitem__)
    assert result['serving_qualified'] is False
    assert result['source_components'] == descriptor['components']
    assert result['receipt_sha256'] == m.digest(m.located(root, m.RECEIPT))


def test_parent_tamper_rejected_before_overlay(fixture):
    root, context, _, _ = fixture
    m.located(root, m.SITE + '/vllm/native.so').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Parent payload mismatch'):
        m.install(context, root)
    assert m.located(root, m.SITE + '/vllm/api.py').read_bytes() == b'# parent\n'


def test_archive_tamper_rejected(fixture):
    root, context, _, _ = fixture
    (context / 'vllm.tar.gz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='archive identity'):
        m.install(context, root)


@pytest.mark.parametrize('name', ['vllm/../../escape', 'vllm/native.so'])
def test_unsafe_source_member_rejected(fixture, name):
    root, context, descriptor, _ = fixture
    descriptor['components']['vllm'] = archive(context, 'vllm', {name: b'bad'})
    update(context, descriptor)
    with pytest.raises(ValueError, match='Only non-native'):
        m.install(context, root)


def test_native_removal_inventory_rejected(fixture):
    root, context, descriptor, _ = fixture
    descriptor['parent_authored_files']['vllm'].append('vllm/native.so')
    update(context, descriptor)
    with pytest.raises(ValueError, match='Only non-native'):
        m.install(context, root)


def test_verify_rejects_payload_and_distribution_changes(fixture):
    root, context, _, _ = fixture
    receipt = m.install(context, root)
    with pytest.raises(ValueError, match='distribution version'):
        m.verify(root, lambda name: 'wrong')
    m.located(root, m.SITE + '/vllm/api.py').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='payload mismatch'):
        m.verify(root, receipt['versions'].__getitem__)


def test_serving_verifies_and_delegates_without_glm_admission(monkeypatch):
    events = []
    monkeypatch.setattr(m.sys, 'argv', ['candidate-image', 'serve', '/models/glm', '--headless'])
    monkeypatch.setattr(m, 'verify', lambda: events.append('verified'))
    monkeypatch.setattr(m.os, 'execve', lambda executable, command, environment: events.append(command))
    m.main()
    assert events == ['verified', ['/opt/venv/bin/python', '-m', 'vllm.entrypoints.cli.main', 'serve', '/models/glm', '--headless']]


def test_verification_failure_prevents_serving(monkeypatch):
    monkeypatch.setattr(m.sys, 'argv', ['candidate-image', 'serve'])
    monkeypatch.setattr(m, 'verify', lambda: (_ for _ in ()).throw(ValueError('invalid')))
    monkeypatch.setattr(m.os, 'execve', lambda *args: pytest.fail('must not execute'))
    with pytest.raises(ValueError, match='invalid'):
        m.main()



def add_contract(context, descriptor):
    source = context / 'lease-candidate.json'
    source.write_text('{"schema":"fixture-lease/v1"}\n', encoding='utf-8')
    destination = '/opt/sparkring/contracts/lease-candidate.json'
    descriptor['integration_contracts'] = {destination: {'file': source.name, 'sha256': m.digest(source)}}
    update(context, descriptor)
    return source, destination


def test_additional_contract_is_byte_bound_and_verified(fixture):
    root, context, descriptor, _ = fixture
    source, destination = add_contract(context, descriptor)
    receipt = m.install(context, root)
    assert m.located(root, destination).read_bytes() == source.read_bytes()
    assert receipt['files'][destination] == m.digest(source)
    assert receipt['integration_contracts'] == descriptor['integration_contracts']
    m.verify(root, receipt['versions'].__getitem__)
    m.located(root, destination).write_text('{}')
    with pytest.raises(ValueError, match='payload mismatch'):
        m.verify(root, receipt['versions'].__getitem__)


@pytest.mark.parametrize('destination', ['/opt/sparkring/contracts/../escape.json',
                                        '/opt/sparkring/contracts/nested/file.json',
                                        '/opt/sparkring/bin/lease.json',
                                        '/opt/sparkring/contracts/lease.py'])
def test_contract_destination_escape_rejected_before_overlay(fixture, destination):
    root, context, descriptor, _ = fixture
    _, original = add_contract(context, descriptor)
    descriptor['integration_contracts'][destination] = descriptor['integration_contracts'].pop(original)
    update(context, descriptor)
    with pytest.raises(ValueError, match='destination'):
        m.install(context, root)
    assert m.located(root, m.SITE + '/vllm/api.py').read_bytes() == b'# parent\n'


@pytest.mark.parametrize('filename', ['../lease.json', '/absolute.json', 'subdir/lease.json'])
def test_contract_source_escape_rejected(fixture, filename):
    root, context, descriptor, _ = fixture
    _, destination = add_contract(context, descriptor)
    descriptor['integration_contracts'][destination]['file'] = filename
    update(context, descriptor)
    with pytest.raises(ValueError, match='context JSON basename'):
        m.install(context, root)


@pytest.mark.parametrize('in_inventory', [False, True])
def test_contract_existing_destination_cannot_be_overwritten(fixture, in_inventory):
    root, context, descriptor, parent = fixture
    _, destination = add_contract(context, descriptor)
    existing = m.located(root, destination)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b'{"parent":"preserved"}\n')
    if in_inventory:
        parent['files'][destination] = m.digest(existing)
        (context / 'parent-installed.json').write_text(json.dumps(parent))
    with pytest.raises(ValueError, match='overwrite refused'):
        m.install(context, root)
    assert existing.read_bytes() == b'{"parent":"preserved"}\n'


def test_contract_hash_mismatch_rejected(fixture):
    root, context, descriptor, _ = fixture
    source, _ = add_contract(context, descriptor)
    source.write_text('{}')
    with pytest.raises(ValueError, match='contract SHA256 mismatch'):
        m.install(context, root)


def test_contract_duplicate_json_key_rejected(fixture):
    root, context, descriptor, _ = fixture
    source, destination = add_contract(context, descriptor)
    source.write_text('{"same":1,"same":2}')
    descriptor['integration_contracts'][destination]['sha256'] = m.digest(source)
    update(context, descriptor)
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        m.install(context, root)


@pytest.mark.parametrize('record', [None, {}, {'file': 'lease.json', 'sha256': 'bad', 'extra': True}])
def test_contract_map_shape_is_strict(fixture, record):
    root, context, descriptor, _ = fixture
    descriptor['integration_contracts'] = {'/opt/sparkring/contracts/lease.json': record}
    update(context, descriptor)
    with pytest.raises(ValueError, match='exactly file and sha256'):
        m.install(context, root)
