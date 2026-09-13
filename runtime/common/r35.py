"""Validate local R35 image receipts without transferring R33 qualification."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / 'runtime/images/sparkring-r35'
SCHEMA = 'sparkring-r35-image-receipt/v1'
LEASE = '/opt/sparkring/contracts/vllm-connector-jobs-r35.json'
CONTRACT = '/opt/sparkring/profile-contract/profile-contract.json'
CAPABILITY = '/opt/sparkring/profile-contract/tp2-sparkcache-capabilities.json'


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def api_healthcheck(port):
    spec = importlib.util.spec_from_file_location('release_api_health', ROOT/'runtime/glm53-spark-mtp3-mesh/managed_liveness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = module.api_healthcheck(port)
    return options, {'Test': ['CMD-SHELL', options['--health-cmd']], 'Interval': 10000000000,
                     'Timeout': 6000000000, 'StartPeriod': 1800000000000, 'Retries': 3}


def profile_contract(installed):
    """Reproduce the recipe's canonical contract, including exact source trees."""
    contract = read(ROOT/'runtime/sparkring/jovian-r33/profiles/profile-contract.json')
    artifact = read(ROOT/'runtime/sparkring/jovian-r33/image/artifact-lock.json')
    components = installed['components']
    for sources in (contract['image']['required_sources'], artifact['source_identities']):
        sources.update(vllm_head=components['vllm']['base_commit'],
                       vllm_integrated_tree=components['vllm']['tree'],
                       b12x_commit=components['b12x']['base_commit'], b12x_tree=components['b12x']['tree'])
    artifact['r35_source_composition'] = components
    contract['image']['artifact_lock_sha256'] = digest(json_bytes(artifact))
    contract['sparkcache_native']['lease_contract'] = LEASE
    contract['candidate_qualification'] = 'Runtime qualification pending; no R33 serving evidence is transferred.'
    return contract


def validate_receipt(document):
    if (not isinstance(document, dict) or document.get('schema') != SCHEMA
            or set(document) - {'schema','image_id','image_reference','platform','installed','verification','bundle_manifest_sha256'}
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', document.get('image_id', ''))
            or document.get('image_reference') != document.get('image_id')
            or document.get('platform') != 'linux/arm64'):
        raise ValueError('R35 requires an exact local ARM64 image receipt')
    installed = document.get('installed', {})
    verification = document.get('verification', {})
    lock = read(RECIPE/'source-lock.json')
    if (installed.get('schema') != 'sparkring-r35-installed/v1'
            or set(installed.get('components', {})) != set(lock['components'])):
        raise ValueError('R35 installed receipt is absent')
    for name, expected in lock['components'].items():
        actual = installed.get('components', {}).get(name, {})
        for key in ('base_commit', 'base_tree', 'tree', 'patch_sha256'):
            if actual.get(key) != expected[key]:
                raise ValueError('R35 component differs from its source lock: '+name+'/'+key)
        if 'native_base_comparison' in expected and actual.get('native_base_comparison') != expected['native_base_comparison']:
            raise ValueError('R35 native-build comparison differs from the source lock')
    version = '0.26.1rc0+sparkring.r35.' + lock['components']['vllm']['tree'][:8]
    if installed.get('versions', {}).get('vllm') != version:
        raise ValueError('R35 vLLM distribution identity differs from the source composition')
    files = installed.get('files', {})
    if (not files or any(not PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                         or not re.fullmatch(r'[0-9a-f]{64}', value) for name, value in files.items())):
        raise ValueError('R35 installed file inventory is invalid')
    if (verification.get('schema') != 'sparkring-r35-verification/v1'
            or set(verification) - {'schema','files_verified','source_components','source_lock_sha256','serving_qualified','checked_files'}
            or verification.get('source_lock_sha256') != digest(json_bytes(installed))
            or verification.get('source_components') != installed['components']
            or verification.get('files_verified') != len(files)
            or verification.get('serving_qualified') is not False):
        raise ValueError('R35 verification does not bind the installed payload')
    contract = profile_contract(installed)
    required_files = {
        CONTRACT: digest(json_bytes(contract)),
        CAPABILITY: digest((RECIPE/'contracts/tp2-sparkcache-capabilities.json').read_bytes()),
        LEASE: digest((RECIPE/'contracts/vllm-connector-jobs.json').read_bytes()),
        '/opt/sparkring/bin/sparkring': digest((RECIPE/'entrypoint.py').read_bytes()),
        '/opt/sparkring/bin/verify-r35.py': digest((RECIPE/'verify_image.py').read_bytes()),
        '/opt/sparkring/image/artifact-lock.json': contract['image']['artifact_lock_sha256'],
    }
    parent = read(ROOT/'runtime/sparkring/jovian-r33/public-image-receipt.json')
    if installed.get('parent_source_lock_sha256') != parent['source_lock_sha256']:
        raise ValueError('R35 installed receipt has a different parent source lock')
    for name in ('/opt/local-inference/nccl/lib/libnccl.so.2.31.2',
                 '/opt/sparkring/sircl/libspark_transport_capi.so',
                 '/opt/sparkring/sircl/python/sparkring-overlay-manifest.json',
                 '/opt/sparkring/runtime/glm53-spark-mtp3-mesh/pins.json'):
        required_files[name] = parent['verification']['checked_files'][name]
    for name, expected in required_files.items():
        if files.get(name) != expected:
            raise ValueError('R35 receipt does not bind required runtime file: '+name)
    for key in ('placement', 'snapshot'):
        native = contract['sparkcache_native']
        if files.get(native[key+'_path']) != native[key+'_sha256']:
            raise ValueError('R35 SparkCache native identity mismatch')
    runtime_files = {name: sha for name, sha in files.items()
                     if name.startswith(('/opt/sparkring/', '/opt/local-inference/nccl/'))}
    return dict(document, bundle_manifest_sha256=files['/opt/sparkring/sircl/python/sparkring-overlay-manifest.json'],
                verification=dict(verification, checked_files=runtime_files))


def validate_profile_capabilities(document, profile):
    validate_receipt(document)
    contract = profile_contract(document['installed'])
    if profile not in contract['profiles']:
        raise ValueError('R35 profile is absent from its canonical contract')
    if profile == 'tp2-dcp1-sparkcache':
        capability = read(RECIPE/'contracts/tp2-sparkcache-capabilities.json')
        required = contract['profiles'][profile]['required_capabilities']
        if (set(capability['checks']) != set(required)
                or any(capability['checks'][name] != 'implemented' for name in required)):
            raise ValueError('R35 TP2 SparkCache component capabilities are incomplete')
        for name, expected in capability['evidence_sha256'].items():
            path = '/opt/sparkring/profile-contract/evidence/'+name+'.evidence.json'
            if document['installed']['files'].get(path) != expected:
                raise ValueError('R35 TP2 capability evidence is not image-bound: '+name)


def verify_local_image(document, *, run=subprocess.run):
    """Bind a receipt to Docker's actual image before a lifecycle mutation."""
    validate_receipt(document)
    image = document['image_id']
    actual = json.loads(run(['docker','image','inspect',image],check=True,capture_output=True,text=True).stdout)[0]
    if actual.get('Id') != image or actual.get('Os') != 'linux' or actual.get('Architecture') != 'arm64':
        raise ValueError('Local R35 image identity or platform differs')
    checked = json.loads(run(['docker','run','--rm','--network','none','--pull','never',image,'verify'],
                             check=True,capture_output=True,text=True).stdout)
    expected = {key: value for key, value in document['verification'].items() if key != 'checked_files'}
    if checked != expected:
        raise ValueError('Local R35 payload verification differs from the launch receipt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--installed-receipt', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = dict(schema=SCHEMA, image_id=args.image_id, image_reference=args.image_id,
                   platform='linux/arm64', installed=read(args.installed_receipt), verification=read(args.verification))
    validate_receipt(receipt)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json_bytes(receipt).decode())


if __name__ == '__main__':
    main()
