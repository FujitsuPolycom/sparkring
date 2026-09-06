"""Install the checked indexer fix and preserve parent/source receipt provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

from patch_indexer_barrier import apply_patch, BEFORE_SHA256, AFTER_SHA256


BASE = Path('/opt/sparkring/overlays/jj-r8-sparkcache-arm64')
RECEIPTS = Path('/opt/sparkring/receipts/jj-r8-sparkcache-arm64')
SITE = Path('/usr/local/lib/python3.12/dist-packages')
RELATIVE = 'b12x/attention/dsa_indexer/fused_indexer.py'
VERIFY = '/opt/sparkring/bin/verify-jj-r8-sparkcache-image.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-source-receipt')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    subprocess.run([sys.executable, VERIFY, '--inside-image'], check=True)
    original_receipt = json.loads((RECEIPTS / 'source-receipt.json').read_text())
    wrapper_changes = {}
    for name, destination in (
        ('serve_with_warmup.py', '/opt/sparkring/bin/serve-with-warmup.py'),
        ('scheduler_liveness.py', '/opt/sparkring/bin/scheduler_liveness.py'),
    ):
        target = Path(destination)
        before = digest(target)
        if before != original_receipt['inputs'][name]:
            raise RuntimeError(f'parent wrapper differs from its receipt: {name}')
        shutil.copyfile(Path(__file__).with_name(name), target)
        target.chmod(0o755)
        wrapper_changes[name] = {'before_sha256': before, 'after_sha256': digest(target)}
    for root in (SITE, BASE / 'sources'):
        apply_patch(root / RELATIVE)
    for cached in (SITE / RELATIVE).parent.glob('__pycache__/fused_indexer.*.pyc'):
        cached.unlink()
    source_before = digest(RECEIPTS / 'source-receipt.json')
    transform = digest(Path(__file__).with_name('patch_indexer_barrier.py'))
    for root in (RECEIPTS, BASE / 'receipts'):
        manifest_path = root / 'b12x-source-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['files'][RELATIVE] != BEFORE_SHA256:
            raise RuntimeError('parent source manifest differs from the tested preimage')
        manifest['files'][RELATIVE] = AFTER_SHA256
        write_json(manifest_path, manifest)
        receipt_path = root / 'source-receipt.json'
        original = receipt_path.read_bytes()
        receipt_path.with_name('source-receipt.parent.json').write_bytes(original)
        receipt = json.loads(original)
        receipt['parent_source_receipt_sha256'] = source_before
        receipt['inputs']['bundle/receipts/b12x-source-manifest.json'] = digest(manifest_path)
        receipt['inputs']['source_transform/indexer_barrier'] = transform
        for name, change in wrapper_changes.items():
            receipt['inputs'][name] = change['after_sha256']
        receipt['source_transforms'] = {'indexer_barrier': {
            'path': RELATIVE, 'before_sha256': BEFORE_SHA256,
            'after_sha256': AFTER_SHA256, 'transform_sha256': transform,
        }}
        write_json(receipt_path, receipt)
    source_after = digest(RECEIPTS / 'source-receipt.json')
    if args.expected_source_receipt and source_after != args.expected_source_receipt:
        raise RuntimeError('source receipt differs from the prepared build input')
    result = {
        'schema': 'sparkring-indexer-barrier-hotfix/v1',
        'parent_image_id': 'sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075',
        'parent_registry_digest': 'ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f',
        'source_before_sha256': BEFORE_SHA256, 'source_after_sha256': AFTER_SHA256,
        'parent_source_receipt_sha256': source_before,
        'source_receipt_sha256': source_after, 'transform_sha256': transform,
        'installer_sha256': digest(Path(__file__)),
        'wrapper_changes': wrapper_changes,
        'scope': 'indexer barrier/compile revision, sampling readiness warmup, output-stall liveness; native libraries unchanged',
    }
    write_json(RECEIPTS / 'issue224-hotfix.json', result)
    subprocess.run([sys.executable, VERIFY, '--inside-image'], check=True)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
