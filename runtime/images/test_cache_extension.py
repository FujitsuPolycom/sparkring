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
