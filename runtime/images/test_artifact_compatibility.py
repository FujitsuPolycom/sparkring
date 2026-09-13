"""Offline native artifact admission tests; fixtures do not contain executable code."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).with_name('artifact_compatibility.py')
spec = importlib.util.spec_from_file_location('artifact_compatibility', PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture
def records(tmp_path):
    artifact = tmp_path / 'fake-library.so'
    artifact.write_bytes(b'non-executable artifact fixture')
    abi = dict(zip(checker.ABI_FIELDS, (
        '3.12.1', 'cpython-312-aarch64-linux-gnu', '2.13.0',
        '2.13.0+fixture', '13.3', False, '2.39',
    )))
    runtime = {
        'schema': 'sparkring-runtime-abi/v1',
        'platform': {'os': 'linux', 'machine': 'aarch64'},
        'abi': abi, 'gpu_targets': ['sm_121'],
    }
    manifest = {
        'schema': 'sparkring-native-artifact/v1',
        'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'provenance': {'source_repository': 'https://example.invalid/source.git',
                       'source_commit': 'a' * 40, 'build_inputs_sha256': 'b' * 64,
                       'compiler_identity': 'fixture compiler 1.0'},
        'platform': copy.deepcopy(runtime['platform']),
        'abi': copy.deepcopy(abi), 'gpu_targets': ['sm_121'],
    }
    return manifest, runtime, artifact


def check(records):
    return checker.check_artifact(*records, gpu_target='sm_121')


def test_matching_evidence_only_admits_isolated_testing(records):
    result = check(records)
    assert result['eligible_for_isolated_testing'] is True
    assert result['serving_qualified'] is False
    assert result['provenance_authenticated'] is False


def test_extra_abi_constraints_cannot_be_ignored(records):
    records[0]['abi']['extra_native_abi'] = 'unverified'
    with pytest.raises(ValueError, match='schema exactly'):
        check(records)


@pytest.mark.parametrize('field', checker.ABI_FIELDS)
@pytest.mark.parametrize('side', [0, 1])
def test_unknown_abi_is_not_a_wildcard(records, field, side):
    records[side]['abi'][field] = None
    with pytest.raises(ValueError, match='unknown or invalid ABI'):
        check(records)


@pytest.mark.parametrize('field', checker.ABI_FIELDS)
def test_abi_mismatch(records, field):
    records[0]['abi'][field] = True if field == 'cxx11_abi' else 'incompatible'
    with pytest.raises(ValueError, match='ABI mismatch'):
        check(records)


def test_numeric_zero_is_not_a_measured_false_cxx_abi(records):
    records[0]['abi']['cxx11_abi'] = 0
    with pytest.raises(ValueError, match='unknown or invalid ABI'):
        check(records)


@pytest.mark.parametrize('field', checker.PROVENANCE_FIELDS)
def test_provenance_required(records, field):
    del records[0]['provenance'][field]
    with pytest.raises(ValueError, match='missing provenance'):
        check(records)


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('platform', [None, {'os': 'linux', 'machine': 'arm64'},
                                    {'os': 'linux', 'machine': 'x86_64'}])
def test_platform_is_explicit_and_exact(records, side, platform):
    records[side]['platform'] = platform
    with pytest.raises(ValueError, match='platform'):
        check(records)


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('targets', [None, [], ['sm_120a'], ['sm_121', 'sm_121']])
def test_gpu_target_unknown_or_uncovered(records, side, targets):
    records[side]['gpu_targets'] = targets
    with pytest.raises(ValueError, match='GPU target'):
        check(records)


def test_tampered_bytes(records):
    records[2].write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        check(records)


@pytest.mark.parametrize('field,value', [('source_commit', 'main'),
                                        ('build_inputs_sha256', 'unknown')])
def test_provenance_requires_immutable_identifiers(records, field, value):
    records[0]['provenance'][field] = value
    with pytest.raises(ValueError):
        check(records)


def test_cli_and_duplicate_json_rejection(records, tmp_path):
    manifest, runtime, artifact = records
    mp, rp = tmp_path / 'artifact.json', tmp_path / 'runtime.json'
    mp.write_text(json.dumps(manifest), encoding='utf-8')
    rp.write_text(json.dumps(runtime), encoding='utf-8')
    command = [sys.executable, str(PATH), '--manifest', str(mp), '--runtime-abi',
               str(rp), '--artifact', str(artifact), '--gpu-target', 'sm_121']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['serving_qualified'] is False
    mp.write_text('{"schema":"one","schema":"two"}', encoding='utf-8')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'duplicate JSON field' in result.stderr
