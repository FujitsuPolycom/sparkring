"""Verify the complete inherited and replaced R35 image payload without CUDA."""
import hashlib
import json
from importlib import metadata
import importlib.util
from pathlib import Path

LOCK = Path('/opt/sparkring/receipts/r35-installed.json')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_capability_evidence(profile_root):
    capability = json.loads((profile_root/'tp2-sparkcache-capabilities.json').read_text())
    for name, expected in capability['evidence_sha256'].items():
        path = profile_root/'evidence'/(name+'.evidence.json')
        if digest(path) != expected:
            raise ValueError(f'R35 capability evidence mismatch: {name}')
        record = json.loads(path.read_text())
        artifact = record['artifact']
        if Path(artifact).name != artifact or digest(path.parent/artifact) != record['sha256']:
            raise ValueError(f'R35 capability artifact mismatch: {name}')


def verify():
    lock = json.loads(LOCK.read_text())
    if lock['schema'] != 'sparkring-r35-installed/v1':
        raise ValueError('unsupported R35 installed receipt')
    for name, expected in lock['files'].items():
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts or digest(path) != expected:
            raise ValueError(f'R35 installed payload mismatch: {name}')
    for name in lock['removed_authored_files']:
        if Path(name).exists():
            raise ValueError(f'obsolete source file remains: {name}')
    for name, expected in lock['versions'].items():
        if metadata.version(name) != expected:
            raise ValueError(f'R35 distribution version mismatch: {name}')
    site = Path('/opt/venv/lib/python3.12/site-packages')
    spec = importlib.util.spec_from_file_location('lease_verifier', site/'sparkcache/runtime_patches/verify_lease_contract.py')
    lease = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lease)
    lease.verify_contract(site, Path('/opt/sparkring/contracts/vllm-connector-jobs-r35.json'))
    verify_capability_evidence(Path('/opt/sparkring/profile-contract'))
    return {'schema':'sparkring-r35-verification/v1', 'files_verified':len(lock['files']),
            'source_components':lock['components'], 'source_lock_sha256':digest(LOCK),
            'serving_qualified':False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
