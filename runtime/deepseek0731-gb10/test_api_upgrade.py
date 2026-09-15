import difflib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    here = Path(__file__).parent
    monkeypatch.syspath_prepend(str(here))
    spec = importlib.util.spec_from_file_location('api_upgrade_tested', here / 'apply_api_upgrade.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    contract = json.loads((here / 'runtime-contract.json').read_text())
    patch = ''
    for index, item in enumerate(contract['runtime_patch']['files']):
        before = f'value = {index}\n'.encode()
        after = f'value = {index + 100}\n'.encode()
        item['preimage_sha256'], item['result_sha256'] = digest(before), digest(after)
        patch += ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                            'a/' + item['path'], 'b/' + item['path']))
        for tree in ('installed', 'source'):
            path = tmp_path / tree / item['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(before if item['path'] in module.API_PATHS else after)
    (tmp_path / 'change.patch').write_text(patch, encoding='utf-8', newline='\n')
    contract['runtime_patch'].update(path='change.patch', sha256=digest(patch.encode()))
    cp = tmp_path / 'contract.json'
    cp.write_text(json.dumps(contract))
    native = tmp_path / 'native.so'
    native.write_bytes(b'native code unchanged')
    monkeypatch.setattr(module, 'NATIVE_SHA256', digest(native.read_bytes()))
    # No-op semantic attestation has separate source-overlay coverage.
    monkeypatch.setattr(module.overlay, 'attest_noop_files', lambda *args: [])
    args = (cp, tmp_path / 'installed', tmp_path / 'source', native, tmp_path / 'receipt.json')
    return module, args, contract


def test_upgrades_both_trees_preserving_native_and_is_idempotent(fixture):
    module, args, contract = fixture
    native = args[3].read_bytes()
    module.upgrade(*args)
    assert json.loads(args[4].read_text())['status'] == 'applied'
    for root in args[1:3]:
        for item in contract['runtime_patch']['files']:
            assert module.overlay.sha256_file(root / item['path']) == item['result_sha256']
    assert args[3].read_bytes() == native
    second = args[:-1] + (args[4].with_name('second.json'),)
    module.upgrade(*second)
    assert json.loads(second[-1].read_text())['status'] == 'already-applied'


@pytest.mark.parametrize('damage', ['native', 'source-api', 'prerequisite', 'patch', 'receipt'])
def test_rejects_changed_inputs_before_mutation(fixture, damage):
    module, args, contract = fixture
    if damage == 'native':
        args[3].write_bytes(b'other native code')
    elif damage == 'source-api':
        (args[2] / sorted(module.API_PATHS)[0]).write_text('changed = True\n')
    elif damage == 'prerequisite':
        (args[1] / contract['runtime_patch']['files'][0]['path']).write_text('missing_fix = True\n')
    elif damage == 'patch':
        (args[0].parent / 'change.patch').write_text('invalid patch')
    else:
        args[4].write_text('prior evidence')
    before = {p: p.read_bytes() for root in args[1:3] for p in root.rglob('*.py')}
    with pytest.raises(module.overlay.OverlayError):
        module.upgrade(*args)
    assert {p: p.read_bytes() for p in before} == before


def test_rejects_partial_api_state_without_finishing_it(fixture):
    module, args, contract = fixture
    item = next(x for x in contract['runtime_patch']['files'] if x['path'] in module.API_PATHS)
    path = args[1] / item['path']
    patches = module.overlay.parse_unified_patch((args[0].parent / 'change.patch').read_text())
    path.write_bytes(module.overlay.apply_file_patch(path.read_bytes(), patches[item['path']]))
    before = {p: p.read_bytes() for root in args[1:3] for p in root.rglob('*.py')}
    with pytest.raises(module.overlay.OverlayError, match='partially applied'):
        module.upgrade(*args)
    assert {p: p.read_bytes() for p in before} == before
