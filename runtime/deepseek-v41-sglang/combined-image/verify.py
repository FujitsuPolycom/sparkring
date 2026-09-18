"""Record or check the isolated SGLang payload and inherited vLLM receipt."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path('/opt/sparkring/sglang')
RECEIPT = HERE / 'installed.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_inputs(manifest):
    parent = Path('/opt/sparkring/receipts/candidate-installed.json')
    if digest(parent) != manifest['vllm_parent']['receipt_sha256']:
        raise ValueError('Inherited vLLM receipt differs from the pinned Qwen image')
    if digest(Path(manifest['nccl']['target'])) != manifest['nccl']['sha256']:
        raise ValueError('SGLang NCCL library differs from the pinned build')
    overlay = json.loads(Path('/opt/sparkring/sglang-overlay.json').read_bytes())
    root = Path('/sgl-workspace/sglang')
    for item in overlay['files']:
        if digest(root / item['path']) != item['sha256']:
            raise ValueError('SGLang source overlay differs: ' + item['path'])
    context = json.loads((HERE / 'context.json').read_bytes())
    for name, change in context['adapter_modifications'].items():
        if digest(Path('/opt/dsv41') / name) != change['after']:
            raise ValueError('Mia adapter integration differs: ' + name)
    return parent


def payload(manifest):
    files = {Path(manifest['nccl']['target']),
             Path('/usr/local/cuda-13.0/lib64/libcudart.so.13').resolve(),
             Path('/opt/sparkring/sglang-overlay.json')}
    for directory in ('/opt/dsv41', '/sgl-workspace/sglang/python/sglang',
                      '/opt/sparkring/sglang', '/opt/sparkring/sglang-patches'):
        for path in Path(directory).rglob('*'):
            if (path.is_file() and path != RECEIPT and '.git' not in path.parts
                    and '__pycache__' not in path.parts and path.suffix != '.pyc'):
                files.add(path)
    return {str(path): digest(path) for path in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'verify'])
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_bytes())
    parent = check_inputs(manifest)
    if args.action == 'install':
        if RECEIPT.exists():
            raise ValueError('An installed SGLang receipt already exists')
        record = {'schema': 'sparkring-sglang-installed/v1', 'id': manifest['id'],
                  'vllm_parent_receipt_sha256': digest(parent),
                  'composition_sha256': digest(HERE / 'manifest.json'),
                  'files': payload(manifest)}
        RECEIPT.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    else:
        record = json.loads(RECEIPT.read_bytes())
        if record['composition_sha256'] != digest(HERE / 'manifest.json'):
            raise ValueError('SGLang composition manifest differs from installation')
        for name, expected in record['files'].items():
            if digest(Path(name)) != expected:
                raise ValueError('Installed SGLang payload differs: ' + name)
    print(json.dumps({'id': record['id'], 'files_checked': len(record['files']),
                      'composition_sha256': digest(HERE / 'manifest.json'),
                      'vllm_parent_receipt_sha256': digest(parent)}))


if __name__ == '__main__':
    main()
