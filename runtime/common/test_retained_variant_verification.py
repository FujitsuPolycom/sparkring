"""Malformed retained-image inputs must fail before any container or Git work."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path):
    spec = importlib.util.spec_from_file_location('variant_'+path.parent.name.replace('-', '_')+'_'+path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture(params=['glm53-flash-e10536a', 'glm53-flash-b12x-kda-adaptive-mtp'])
def variant(request):
    folder = ROOT/'runtime'/request.param
    return folder, module(folder/'prepare_context.py'), module(folder/'verify_image.py')


def context(variant, tmp_path):
    folder, prepare, verify = variant
    pins_path = folder/'pins.json'
    pins = json.loads(pins_path.read_text())
    required = ['bundle/runtime/'+name for name in ('pins.json','verify_image.py','Containerfile','Containerfile.seed','build-image.sh','LICENSES.md','SparkRing-LICENSE')]
    required.append('bundle/sources/instanttensor-'+pins['public_image_build']['instanttensor']['version']+'.tar.gz')
    hashes = {}
    for relative in required:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = pins_path.read_bytes() if relative.endswith('/pins.json') else b'fixture'
        path.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    receipt = {'schema':prepare.RECEIPT_SCHEMA, 'pins_sha256':hashlib.sha256(pins_path.read_bytes()).hexdigest(), 'files':hashes}
    return pins_path, pins, receipt


@pytest.mark.parametrize('damage', ['none','missing','extra','pins','untracked'])
def test_context_requires_complete_confined_payload(variant, tmp_path, monkeypatch, damage):
    _, prepare, _ = variant
    pins_path, _, receipt = context(variant, tmp_path)
    if damage == 'missing':
        receipt['files'].pop('bundle/runtime/verify_image.py')
    elif damage == 'extra':
        receipt['files']['../outside'] = 'a'*64
    elif damage == 'pins':
        (tmp_path/'bundle/runtime/pins.json').write_bytes(b'changed')
        receipt['files']['bundle/runtime/pins.json'] = hashlib.sha256(b'changed').hexdigest()
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(prepare, 'verify_git_tree', lambda *args, **kwargs: None)
    monkeypatch.setattr(prepare, 'verify_source_lineage', lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(prepare, 'run', lambda *args, **kwargs: 'extra.py' if damage == 'untracked' else '')
    if damage == 'none':
        assert prepare.verify_context(tmp_path, pins_path=pins_path) == receipt
    else:
        with pytest.raises(prepare.PrepareError):
            prepare.verify_context(tmp_path, pins_path=pins_path)


@pytest.mark.parametrize('damage', ['none','receipt','pins','inventory','sources'])
def test_image_probe_binds_source_receipt_to_pins(variant, tmp_path, monkeypatch, damage):
    _, _, verify = variant
    pins_path, pins, receipt = context(variant, tmp_path)
    receipt['sources'] = {name:{'commit':row['commit'],'tree':row.get('patched_tree',row.get('tree'))} for name,row in pins['public_image_build']['sources'].items()}
    receipt['sources']['vllm']['source_lineage'] = {'note':'additional producer provenance'}
    labels = {**verify.expected_labels(pins), 'org.sparkring.source-receipt-sha256':'b'*64}
    probe = {'source_receipt':receipt,'source_receipt_sha256':'b'*64,'pins_sha256':receipt['pins_sha256'],'nccl_sha256':pins['public_image_build']['outputs']['nccl_library_sha256'] or 'd'*64}
    if damage == 'receipt':
        probe['source_receipt_sha256'] = 'c'*64
    elif damage == 'pins':
        probe['pins_sha256'] = 'c'*64
    elif damage == 'inventory':
        receipt['files'].pop('bundle/runtime/Containerfile')
    elif damage == 'sources':
        receipt['sources']['vllm']['commit'] = 'c'*40
    inspection = {'Id':'sha256:'+'a'*64,'Architecture':'arm64','Os':'linux','Config':{'Labels':labels}}
    monkeypatch.setattr(verify, 'run', lambda *args: json.dumps([inspection]))
    def probe_image(engine, image):
        assert image == inspection['Id']
        return probe
    monkeypatch.setattr(verify, 'runtime_probe', probe_image)
    if damage == 'none':
        assert verify.verify_image('unused','mutable-tag',pins_path)['image_id'] == inspection['Id']
    else:
        with pytest.raises(verify.VerifyError):
            verify.verify_image('unused','mutable-tag',pins_path)
