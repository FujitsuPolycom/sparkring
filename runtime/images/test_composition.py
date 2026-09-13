"""Reject source-overlay drift before accepting a build context."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from runtime.images.composition import assess_update, patch_paths, read, read_baseline_receipt, validate

RECIPE = Path(__file__).parent / 'sparkring-r35'


@pytest.fixture
def recipe(tmp_path):
    target = tmp_path / 'runtime/images/sparkring-r35'
    shutil.copytree(RECIPE, target)
    shutil.copy2(RECIPE.parent / 'inspect_abi.py', target.parent / 'inspect_abi.py')
    return target


def change(recipe, file, edit):
    path = recipe / file
    value = read(path)
    edit(value)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_repository_composition():
    result = validate(RECIPE)
    assert result['components'] == ['vllm', 'b12x']
    assert result['unknown_abi_fields']
    assert not result['wheel_reuse_authorized']


@pytest.mark.parametrize('mutation,match', [
    (lambda x: x.update(platform='linux/amd64'), 'ARM64'),
    (lambda x: x['foundation'].update(image_id='sha256:' + '0' * 64), 'foundation'),
    (lambda x: x['source_lock']['components'].pop('b12x'), 'components'),
    (lambda x: x.update(inherited_native={}), 'native libraries'),
])
def test_incompatible_foundation(recipe, mutation, match):
    change(recipe, 'compatibility.json', mutation)
    with pytest.raises(ValueError, match=match):
        validate(recipe)


@pytest.mark.parametrize('mutation,match', [
    (lambda x: x['components'].pop('b12x'), 'exactly'),
    (lambda x: x['components']['vllm']['changed_paths'].pop(), 'inventory'),
    (lambda x: x['components']['vllm'].update(component_regression_tests=['missing.py']), 'regression'),
    (lambda x: x['components']['vllm'].update(patch='../outside.patch'), 'patch differs'),
])
def test_incomplete_ledger(recipe, mutation, match):
    change(recipe, 'patch-ledger.json', mutation)
    with pytest.raises(ValueError, match=match):
        validate(recipe)


def test_patch_bytes_changed(recipe):
    with (recipe / 'patches/vllm-sparkring.patch').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate(recipe)


def baseline_receipt():
    manifest = read(RECIPE / 'compatibility.json')
    lock = read(RECIPE / 'source-lock.json')
    return {'schema': manifest['evidence']['receipt_schema'],
            'image_id': manifest['evidence']['image_id'],
            'installed': {'components': lock['components'],
                          'files': manifest['inherited_native'],
                          'versions': {'torch': manifest['abi']['torch_distribution_version']}}}


def test_baseline_receipt_and_native_drift():
    receipt = baseline_receipt()
    assert validate(RECIPE, receipt)['baseline_receipt_checked']
    path = next(iter(receipt['installed']['files']))
    receipt['installed']['files'][path] = '0' * 64
    with pytest.raises(ValueError, match='native library mismatch'):
        validate(RECIPE, receipt)


def test_update_is_review_not_automatic_reuse():
    baseline = read(RECIPE / 'source-lock.json')
    candidate = copy.deepcopy(baseline)
    candidate['components']['vllm']['base_commit'] = '0' * 40
    result = assess_update(baseline, candidate)
    assert result['changed_components'] == ['vllm']
    assert result['required_actions']
    assert not result['binary_reuse_authorized']
    candidate['parent_image_id'] = 'sha256:' + '0' * 64
    assert assess_update(baseline, candidate)['foundation_changed']


@pytest.mark.parametrize('header', [
    b'diff --git "a/path with spaces.py" "b/path with spaces.py"\n',
    b'diff --git a/path with spaces.py b/path with spaces.py\n',
])
def test_patch_inventory_cannot_silently_skip_paths(header):
    with pytest.raises(ValueError, match='Unsupported Git diff path'):
        patch_paths(b'diff --git a/valid.py b/valid.py\n' + header)


def test_baseline_receipt_bytes_are_bound(recipe, tmp_path):
    raw = json.dumps(baseline_receipt()).encode()
    path = tmp_path / 'receipt.json'
    path.write_bytes(raw)
    change(recipe, 'compatibility.json',
           lambda x: x['evidence'].update(receipt_sha256=hashlib.sha256(raw).hexdigest()))
    assert read_baseline_receipt(recipe, path) == baseline_receipt()
    path.write_bytes(raw + b'\n')
    with pytest.raises(ValueError, match='receipt bytes'):
        read_baseline_receipt(recipe, path)


def test_update_reviews_inherited_file_removal_inventory():
    baseline = read(RECIPE / 'source-lock.json')
    candidate = copy.deepcopy(baseline)
    candidate['baseline_file_lists']['vllm']['sha256'] = '0' * 64
    result = assess_update(baseline, candidate)
    assert result['other_lock_changes'] == ['baseline_file_lists']
    assert result['required_actions']


def test_abi_claim_must_match_recorded_measurement(recipe):
    change(recipe, 'compatibility.json', lambda x: x['abi'].update(cxx11_abi=False))
    with pytest.raises(ValueError, match='differs from measurement'):
        validate(recipe)


def test_unmeasured_non_null_abi_claim_rejected(recipe):
    change(recipe, 'compatibility.json', lambda x: x['abi'].update(cuda_toolkit_version='13.3'))
    with pytest.raises(ValueError, match='lacks a measurement'):
        validate(recipe)


def test_duplicate_contract_fields_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"schema":"one","schema":"two"}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate contract field'):
        read(path)


def test_published_r37_build_inputs_keep_exact_bytes():
    recipe = Path(__file__).parent / 'compositions/lil-r37-glm-spark'
    lock = read(recipe / 'source-lock.json')
    descriptor = read(recipe / 'descriptor.json')
    artifacts = read(recipe / 'runtime-artifacts.json')
    publication = read(recipe / 'publication.json')
    assert artifacts['image_id'] == publication['image_id']
    assert hashlib.sha256((recipe / 'candidate-image.py').read_bytes()).hexdigest() == artifacts['files']['/opt/sparkring/bin/candidate-image.py']
    for name, component in lock['components'].items():
        assert component == descriptor['components'][name]
        assert hashlib.sha256((recipe / component['patch']).read_bytes()).hexdigest() == component['patch_sha256']
    for record in descriptor['integration_contracts'].values():
        assert hashlib.sha256((recipe / record['file']).read_bytes()).hexdigest() == record['sha256']
