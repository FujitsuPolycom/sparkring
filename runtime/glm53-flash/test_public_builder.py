"""CPU-only contracts for the source-complete GLM-5.3 image builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = HERE / "pins.json"
PATCH = ROOT / "spark_transport" / "nccl" / "nccl-2.30.7-switchless-cycle.patch"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _module("glm53_prepare_context", HERE / "prepare_context.py")
verify = _module("glm53_verify_image", HERE / "verify_image.py")


def test_public_builder_pins_complete_source_and_license_boundary() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    build = pins["public_image_build"]
    assert build["platform"] == "linux/arm64"
    assert build["status"] == "qualified"
    assert build["outputs"]["runtime_image"].endswith(
        "@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd"
    )
    assert build["base_images"]["cuda_runtime"]["license"] == (
        "LicenseRef-NVIDIA-Deep-Learning-Container"
    )
    assert build["sources"]["vllm"]["commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert build["sources"]["b12x"]["commit"] == (
        "2fcf23a0ce269be27b2e03fece73d46e90e6aeea"
    )
    assert build["sources"]["nccl"]["commit"] == (
        "73cf112295c33aee2b895f329f592f2a9b4b0f97"
    )
    assert build["sources"]["nccl"]["patched_tree"] == (
        "abdeb053b94c3f6d472cd55ae2b79ca821299009"
    )
    assert build["instanttensor"]["license"] == "Apache-2.0"


def test_switchless_patch_is_hash_bound_and_uses_original_interface() -> None:
    pins = prepare.load_pins(PINS)
    patch_pin = pins["public_image_build"]["sources"]["nccl"]["patches"][0]
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == patch_pin["sha256"]
    text = PATCH.read_text(encoding="utf-8")
    assert "NCCL_SWITCHLESS_RING_ONLY" in text
    assert "NCCL_SKIP_TREE_CONNECT" not in text
    assert "ncclNMergedIbDevs + devIdx" not in text
    assert "ncclIbMergedDevs + devIdx" in text


def test_image_inspection_requires_exact_platform_and_labels() -> None:
    pins = verify.load_pins(PINS)
    document = {
        "Id": "sha256:" + "a" * 64,
        "Architecture": "arm64",
        "Os": "linux",
        "Config": {"Labels": verify.expected_labels(pins)},
    }
    verify.validate_inspection(document, pins)
    document["Architecture"] = "amd64"
    with pytest.raises(verify.VerifyError, match="linux/arm64"):
        verify.validate_inspection(document, pins)


def test_builder_uses_public_context_and_source_built_nccl() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "prepare_context.py" in script
    assert "sparkring-glm53-official-spark" not in script
    assert "make -C /build/nccl" in containerfile
    assert "org.opencontainers.image.licenses" in containerfile
    assert 'ENTRYPOINT ["vllm"]' in containerfile
    assert "COPY bundle/runtime/SparkRing-LICENSE" in containerfile
    assert "COPY bundle/sources/vllm/LICENSE" in containerfile

@pytest.mark.parametrize('family', ['qwen', 'glm'])
@pytest.mark.parametrize('damage', ['missing', 'extra', 'embedded-pins', 'untracked', 'valid'])
def test_prepared_context_rejects_incomplete_or_contaminated_inputs(tmp_path, monkeypatch, family, damage):
    from runtime.qwen38 import prepare_context as qwen
    module = qwen if family == 'qwen' else prepare
    pins_path = HERE.parent / 'qwen38/pins.json' if family == 'qwen' else PINS
    pins = module.load_pins(pins_path)
    names = ['pins.json', 'verify_runtime.py', 'qwen38_dgx2_serve.sh', 'qwen38_dgx4_serve.sh', 'chat_template_agentic.jinja', 'requirements-public.txt'] if family == 'qwen' else ['pins.json', 'verify_image.py', 'Containerfile', 'Containerfile.seed', 'build-image.sh', 'LICENSES.md', 'SparkRing-LICENSE']
    files = {'bundle/runtime/'+name: b'payload' for name in names}
    files['bundle/runtime/pins.json'] = pins_path.read_bytes()
    if family == 'glm':
        files['bundle/sources/instanttensor-'+pins['public_image_build']['instanttensor']['version']+'.tar.gz'] = b'archive'
    for relative, data in files.items():
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    receipt = {'schema': module.RECEIPT_SCHEMA, 'pins_sha256': module.sha256_file(pins_path), 'files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    if damage == 'missing':
        receipt['files'].pop('bundle/runtime/pins.json')
    elif damage == 'extra':
        receipt['files']['../outside'] = 'a'*64
    elif damage == 'embedded-pins':
        path = tmp_path/'bundle/runtime/pins.json'
        path.write_text('{}')
        receipt['files']['bundle/runtime/pins.json'] = module.sha256_file(path)
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    def fake_run(argv, **kwargs):
        if 'ls-files' in argv:
            return 'injected.py' if damage == 'untracked' else ''
        if 'write-tree' in argv:
            name = Path(argv[argv.index('-C')+1]).name
            source = pins['companion'] if name == 'qwen38-spark-pair' else (pins['sources'] if family == 'qwen' else pins['public_image_build']['sources'])[name]
            return source.get('patched_tree', source.get('tree'))
        return ''
    def fake_git(repo, revision):
        source = pins['companion'] if repo.name == 'qwen38-spark-pair' else (pins['sources'] if family == 'qwen' else pins['public_image_build']['sources'])[repo.name]
        return source['commit'] if revision == 'HEAD' else source['tree']
    monkeypatch.setattr(module, 'run', fake_run)
    monkeypatch.setattr(module, 'git_value', fake_git)
    if damage == 'valid':
        assert module.verify_context(tmp_path, pins_path=pins_path) == receipt
    else:
        with pytest.raises(module.PrepareError):
            module.verify_context(tmp_path, pins_path=pins_path)


def test_image_probe_rejects_receipt_presence_without_identity(monkeypatch):
    monkeypatch.setattr(verify, 'run', lambda argv: json.dumps({'nccl_sha256': 'a'*64, 'source_receipt_present': True, 'pins_present': True}))
    with pytest.raises(verify.VerifyError):
        verify.runtime_probe('unused', 'sha256:'+'b'*64)


@pytest.mark.parametrize('damage', ['pins', 'receipt-label', 'sources', 'inventory', 'valid'])
def test_image_verification_binds_installed_documents(monkeypatch, damage):
    pins = verify.load_pins(PINS)
    digest = hashlib.sha256(PINS.read_bytes()).hexdigest()
    receipt = {'schema': prepare.RECEIPT_SCHEMA, 'pins_sha256': digest,
               'sources': {name: {'commit': row['commit'], 'tree': row.get('patched_tree', row.get('tree'))}
                           for name, row in pins['public_image_build']['sources'].items()}}
    receipt['files'] = {name: 'd'*64 for name in prepare.payload_files(pins)}
    receipt['files']['bundle/runtime/pins.json'] = digest
    probe = {'nccl_sha256': pins['public_image_build']['outputs']['nccl_library_sha256'],
             'source_receipt_sha256': 'a'*64, 'pins_sha256': digest, 'source_receipt': receipt,
             'imports': ['vllm', 'b12x', 'instanttensor']}
    labels = verify.expected_labels(pins) | {'org.sparkring.source-receipt-sha256': 'a'*64}
    inspection = {'Id': 'sha256:'+'b'*64, 'Architecture': 'arm64', 'Os': 'linux', 'Config': {'Labels': labels}}
    if damage == 'pins':
        probe['pins_sha256'] = 'c'*64
    if damage == 'receipt-label':
        labels['org.sparkring.source-receipt-sha256'] = 'c'*64
    if damage == 'sources':
        receipt['sources']['vllm']['commit'] = 'c'*40
    if damage == 'inventory':
        receipt['files'].pop('bundle/runtime/Containerfile')
    monkeypatch.setattr(verify, 'run', lambda argv: json.dumps([inspection]))
    def mock_probe(engine, image):
        assert image == inspection['Id']
        return probe
    monkeypatch.setattr(verify, 'runtime_probe', mock_probe)
    if damage == 'valid':
        assert verify.verify_image('unused', 'mutable-tag', PINS)['image_id'] == inspection['Id']
    else:
        with pytest.raises(verify.VerifyError):
            verify.verify_image('unused', 'mutable-tag', PINS)

@pytest.mark.parametrize('failing_module', [None, 'b12x'])
def test_probe_executes_required_package_imports(tmp_path, monkeypatch, failing_module):
    import contextlib
    import importlib
    import importlib.metadata
    import io
    imports = []
    def import_package(name):
        imports.append(name)
        if name == failing_module:
            raise ImportError('unloadable runtime package')
        return object()
    monkeypatch.setattr(importlib, 'import_module', import_package)
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'test-version')
    for name, data in [('nccl', b'library'), ('receipt', b'{}'), ('pins', b'{}')]:
        (tmp_path/name).write_bytes(data)
    def execute_probe(argv):
        program = argv[-1]
        for original, name in [('/opt/sparkring/nccl/libnccl.so.2.30.7', 'nccl'),
                               ('/opt/sparkring/runtime/source-receipt.json', 'receipt'),
                               ('/opt/sparkring/runtime/pins.json', 'pins')]:
            program = program.replace(original, (tmp_path/name).as_posix())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(program, {})
        return output.getvalue()
    monkeypatch.setattr(verify, 'run', execute_probe)
    if failing_module:
        with pytest.raises(ImportError, match='unloadable'):
            verify.runtime_probe('unused', 'unused')
    else:
        result = verify.runtime_probe('unused', 'unused')
        assert result['imports'] == imports == ['vllm', 'b12x', 'instanttensor']
