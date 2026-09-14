"""Cache admission binds every installed byte to a pinned parent or extension."""
import copy
import hashlib
import json
import pytest
from runtime.common import cache_candidate as cache


def fixture(tmp_path, monkeypatch):
    parent = {'schema': 'sparkring-candidate-installed/v1', 'composition_id': 'lil-r37-glm-spark',
        'components': {}, 'versions': {}, 'files': {p: 'a' * 64 for p in cache.candidate.NATIVE | {cache.candidate.ENTRYPOINT, cache.SITE + 'sparkcache/example.py'}}}
    parent_raw = json.dumps(parent).encode()
    contract = {'id': 'lil-r37-cache64', 'source': {'commit': 'pin'},
        'parent': {'receipt_sha256': hashlib.sha256(parent_raw).hexdigest()},
        'python_files': {'sparkcache/example.py': 'b' * 64},
        'native': {'destination': '/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so', 'sha256': 'c' * 64}}
    path = tmp_path / 'descriptor.json'
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(cache, 'DESCRIPTOR', path)
    installed = copy.deepcopy(parent)
    installed['files'][cache.SITE + 'sparkcache/example.py'] = 'b' * 64
    installed['files'][contract['native']['destination']] = 'c' * 64
    installed['cache_extension'] = {'descriptor_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'id': contract['id'], 'source': contract['source'], 'native': contract['native']}
    monkeypatch.setattr(cache.candidate, 'composition', lambda _: ({}, {}))
    def validate(**kwargs):
        assert kwargs['installed_bytes'] == json.dumps(installed).encode()
        assert kwargs['native_hashes'][contract['native']['destination']] == 'c' * 64
        return {'admitted': True}
    monkeypatch.setattr(cache.candidate, 'validate', validate)
    return parent_raw, installed


def test_extension_inventory_delegates_verified_composition(tmp_path, monkeypatch):
    parent, installed = fixture(tmp_path, monkeypatch)
    assert cache.validate('sha256:' + 'd' * 64, json.dumps(installed).encode(), parent, {})['admitted']


@pytest.mark.parametrize('mutation', ['extra', 'changed', 'source', 'components'])
def test_extension_rejects_unreviewed_changes(tmp_path, monkeypatch, mutation):
    parent, installed = fixture(tmp_path, monkeypatch)
    if mutation == 'extra':
        installed['files']['/opt/unowned.py'] = 'f' * 64
    elif mutation == 'changed':
        installed['files'][cache.SITE + 'sparkcache/example.py'] = 'f' * 64
    elif mutation == 'source':
        installed['cache_extension']['source'] = {}
    else:
        installed['components'] = {'different': True}
    with pytest.raises(ValueError):
        cache.validate('sha256:' + 'd' * 64, json.dumps(installed).encode(), parent, {})


def test_extension_rejects_unpinned_parent(tmp_path, monkeypatch):
    parent, installed = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='parent receipt'):
        cache.validate('sha256:' + 'd' * 64, json.dumps(installed).encode(), parent + b' ', {})
