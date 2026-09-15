"""Check native artifact identity and exact ABI compatibility before isolated tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ABI_FIELDS = (
    'python_version', 'python_soabi', 'torch_distribution_version',
    'torch_build_version', 'torch_cuda_version', 'cxx11_abi', 'glibc_version',
)
PROVENANCE_FIELDS = (
    'source_repository', 'source_commit', 'build_inputs_sha256', 'compiler_identity',
)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    return value


def _targets(value, label):
    if (not isinstance(value, list) or not value
            or any(not _text(item) for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{label} requires explicit unique GPU targets')
    return value


def check_artifact(manifest, runtime_abi, artifact_path, *, gpu_target):
    """Return isolated-test eligibility; reject missing or incompatible evidence.

    Provenance is a supplier declaration, not an authenticity attestation.
    GPU targets use exact caller-supplied identifiers; no architecture inference
    or forward-compatibility assumption is made.
    """
    manifest = _object(manifest, 'artifact manifest')
    runtime_abi = _object(runtime_abi, 'runtime ABI')
    if manifest.get('schema') != 'sparkring-native-artifact/v1':
        raise ValueError('unsupported artifact manifest schema')
    if runtime_abi.get('schema') != 'sparkring-runtime-abi/v1':
        raise ValueError('unsupported runtime ABI schema')
    expected = manifest.get('sha256')
    if not _sha256(expected):
        raise ValueError('artifact requires a lowercase SHA256 digest')
    provenance = _object(manifest.get('provenance'), 'provenance')
    for field in PROVENANCE_FIELDS:
        if not _text(provenance.get(field)):
            raise ValueError(f'missing provenance: {field}')
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', provenance['source_commit']):
        raise ValueError('source_commit requires a full Git object ID')
    if not _sha256(provenance['build_inputs_sha256']):
        raise ValueError('build_inputs_sha256 requires a lowercase SHA256 digest')
    required_platform = {'os': 'linux', 'machine': 'aarch64'}
    for label, record in (('artifact', manifest), ('runtime', runtime_abi)):
        if record.get('platform') != required_platform:
            raise ValueError(f'{label} platform must be exactly linux/aarch64')
    proposed = _object(manifest.get('abi'), 'artifact ABI')
    measured = _object(runtime_abi.get('abi'), 'runtime ABI fields')
    if set(proposed) != set(ABI_FIELDS) or set(measured) != set(ABI_FIELDS):
        raise ValueError('ABI fields must match the supported schema exactly')
    for field in ABI_FIELDS:
        a, b = proposed.get(field), measured.get(field)
        if field == 'cxx11_abi':
            known = type(a) is bool and type(b) is bool
        else:
            known = _text(a) and _text(b)
        if not known:
            raise ValueError(f'unknown or invalid ABI field: {field}')
        if a != b:
            raise ValueError(f'ABI mismatch: {field}')
    if not _text(gpu_target):
        raise ValueError('an explicit GPU target is required')
    for label, record in (('artifact', manifest), ('runtime', runtime_abi)):
        if gpu_target not in _targets(record.get('gpu_targets'), label):
            raise ValueError(f'{label} does not establish GPU target: {gpu_target}')
    with Path(artifact_path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('artifact SHA256 mismatch')
    return {
        'schema': 'sparkring-artifact-eligibility/v1',
        'eligible_for_isolated_testing': True,
        'serving_qualified': False,
        'sha256': actual,
        'gpu_target': gpu_target,
        'provenance_authenticated': False,
        'scope': 'Exact declared ABI and artifact hash checked; provenance authenticity, '
                 'native dependency closure and runtime correctness are not established.',
    }


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--runtime-abi', required=True, type=Path)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--gpu-target', required=True)
    args = parser.parse_args()
    try:
        result = check_artifact(_read(args.manifest), _read(args.runtime_abi),
                                args.artifact, gpu_target=args.gpu_target)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Artifact rejected: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
