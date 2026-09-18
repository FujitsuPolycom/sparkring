"""Offline checks for pinned cache-extension build inputs."""
import hashlib
import json
from pathlib import Path
import pytest
from runtime.images import cache_extension as extension


def fixture(tmp_path):
    source = tmp_path / 'source'
    (source / 'sparkcache/native').mkdir(parents=True)
    (source / 'sparkcache/example.py').write_bytes(b'VALUE = 1\n')
    (source / 'sparkcache/native/CMakeLists.txt').write_text('project(example)\n')
    descriptor = tmp_path / 'descriptor.json'
    descriptor.write_text(json.dumps({'schema': 'sparkring-cache-extension/v1',
        'source': {'package_sha256': extension.source_digest(source / 'sparkcache')},
        'python_files': {'sparkcache/example.py': hashlib.sha256(b'VALUE = 1\n').hexdigest()},
        'native': {'destination': '/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so'}}))
    descriptor.with_name('Dockerfile').write_text('FROM example\n')
    return source, descriptor


def test_prepare_pins_source_and_refuses_overwrite(tmp_path):
    source, descriptor = fixture(tmp_path)
    output = tmp_path / 'context'
    extension.prepare(descriptor, source, output)
    assert (output / 'source/sparkcache/example.py').read_bytes() == b'VALUE = 1\n'
    assert (output / 'cache_extension.py').is_file()
    with pytest.raises(ValueError, match='empty'):
        extension.prepare(descriptor, source, output)


def test_prepare_rejects_modified_source_before_writing(tmp_path):
    source, descriptor = fixture(tmp_path)
    (source / 'sparkcache/example.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='source tree differs'):
        extension.prepare(descriptor, source, tmp_path / 'context')
    assert not (tmp_path / 'context').exists()


def test_source_identity_is_line_ending_independent(tmp_path):
    source, _ = fixture(tmp_path)
    expected = extension.source_digest(source / 'sparkcache')
    (source / 'sparkcache/example.py').write_bytes(b'VALUE = 1\r\n')
    assert extension.source_digest(source / 'sparkcache') == expected


def test_native_source_reconstructs_only_pinned_line_endings():
    expected = hashlib.sha256(b'line1\r\nline2\r\n').hexdigest()
    assert extension.pinned_source_bytes(b'line1\nline2\n', expected) == b'line1\r\nline2\r\n'
    with pytest.raises(ValueError, match='pinned byte identity'):
        extension.pinned_source_bytes(b'changed\n', expected)


@pytest.mark.parametrize('name', ['/outside.py', '../outside.py', 'sparkcache/../outside.py', 'vllm/entry.py', 'sparkcache/native.so'])
def test_descriptor_rejects_unowned_replacements(tmp_path, name):
    _, path = fixture(tmp_path)
    descriptor = json.loads(path.read_text())
    descriptor['python_files'] = {name: 'a' * 64}
    path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match='contained'):
        extension.read_descriptor(path)


def test_descriptor_rejects_other_native_destination(tmp_path):
    _, path = fixture(tmp_path)
    descriptor = json.loads(path.read_text())
    descriptor['native']['destination'] = '/usr/lib/example.so'
    path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match='snapshot library'):
        extension.read_descriptor(path)


def test_composition_matches_published_parent_and_tested_capacity():
    root = Path(__file__).resolve().parents[2]
    descriptor = extension.read_descriptor(root / 'runtime/images/compositions/lil-r37-cache64/descriptor.json')
    parent = json.loads((root / 'runtime/images/compositions/lil-r37-glm-spark/publication.json').read_text())
    assert descriptor['parent']['image_id'] == parent['image_id']
    assert descriptor['native']['group_capacity'] == 64
    assert len(descriptor['python_files']) == 5


def boundary_fixture(tmp_path):
    package = tmp_path / extension.SITE.lstrip('/') / 'sparkcache/example.py'
    package.parent.mkdir(parents=True)
    package.write_bytes(b'VALUE = 1\n')
    library_name = '/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so'
    library = tmp_path / library_name.lstrip('/')
    library.parent.mkdir(parents=True)
    library.write_bytes(b'fixed-native-bytes')
    name = '/opt/sparkring/contracts/boundary-runtime.json'
    path = tmp_path / name.lstrip('/')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'schema':'sparkcache-boundary-runtime/v1',
        'files':{'sparkcache/example.py':extension.sha(package.read_bytes())},
        'external':{library_name:extension.sha(library.read_bytes())}}))
    receipt = {'files':{name:extension.sha(path.read_bytes())}}
    return receipt, path, package, library_name


def test_python_extension_preserves_parent_boundary_and_creates_distinct_identity(tmp_path):
    receipt, original, package, _ = boundary_fixture(tmp_path)
    before = original.read_bytes()
    result = extension.boundary_identity_update(receipt,
        {extension.SITE+'/sparkcache/example.py':b'VALUE = 2\n'}, 'a'*64, tmp_path)
    assert original.read_bytes() == before
    assert package.read_bytes() == b'VALUE = 1\n'
    assert result['record']['path'].endswith('boundary-cache-aaaaaaaaaaaaaaaa.json')
    assert result['record']['serving_qualified'] is False
    assert result['record']['parent_sha256'] == extension.sha(before)
    value = json.loads(result['data'])
    assert value['files']['sparkcache/example.py'] == extension.sha(b'VALUE = 2\n')


def test_boundary_extension_refuses_untracked_parent_and_native_change(tmp_path):
    receipt, original, _, library = boundary_fixture(tmp_path)
    with pytest.raises(ValueError, match='separately qualified'):
        extension.boundary_identity_update(receipt, {library:b'changed native'}, 'a'*64, tmp_path)
    with pytest.raises(ValueError, match='verified installed ownership'):
        extension.boundary_identity_update({'files':{}}, {}, 'a'*64, tmp_path)
    result = extension.boundary_identity_update({'files':{}}, {}, 'a'*64, tmp_path,
                                                expected_parent_sha256=extension.sha(original.read_bytes()))
    assert result['record']['serving_qualified'] is False


def test_boundary_extension_rejects_changed_parent_payload(tmp_path):
    receipt, _, package, _ = boundary_fixture(tmp_path)
    package.write_bytes(b'undeclared change')
    with pytest.raises(ValueError, match='parent package identity'):
        extension.boundary_identity_update(receipt, {}, 'a'*64, tmp_path)


def test_boundary_extension_preserves_required_runtime_mount_identity(tmp_path):
    receipt, original, _, library = boundary_fixture(tmp_path)
    expected = json.loads(original.read_bytes())['external'][library]
    (tmp_path/library.lstrip('/')).unlink()
    result = extension.boundary_identity_update(receipt, {}, 'a'*64, tmp_path)
    assert result['record']['required_runtime_external'] == {library:expected}
    assert json.loads(result['data'])['external'][library] == expected
    assert result['record']['serving_qualified'] is False
