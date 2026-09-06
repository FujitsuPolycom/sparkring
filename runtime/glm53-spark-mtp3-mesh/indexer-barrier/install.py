"""Install a checked mesh indexer patch while retaining compute/transport provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from patch_indexer_barrier import AFTER, BEFORE, RELATIVE, apply

PARENT = 'ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987'
PARENT_ID = 'sha256:2e41b1e934a85ff7c21b780532db2f0a0e978df081e52f4ae2bf11f8992fb24f'
PARENT_LOCK = '139f36701e0e47f45bf99fba2cc2fa59b417f2ee801dad3a064455d5b464a459'
RECEIPTS = Path('/opt/sparkring/receipts/glm53-spark-mtp3-mesh')
COMPUTE = Path('/opt/sparkring-compute')
INSTALLED = Path('/opt/sparkring/receipts/glm53-compute-installed.json')
SITE = Path('/usr/local/lib/python3.12/dist-packages')
VERIFY = '/opt/sparkring/bin/verify-mtp3-mesh-image.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def install():
    subprocess.run([sys.executable, '-I', VERIFY, '--inside-image'], check=True)
    lock_path = COMPUTE / 'source-lock.json'
    if sha(lock_path) != PARENT_LOCK:
        raise ValueError('Mesh compute source lock differs from the supported parent')
    source_path = RECEIPTS / 'source-receipt.json'
    source = load(source_path)
    parent_source_sha = sha(source_path)
    installed = load(INSTALLED)
    if installed['b12x_files'][RELATIVE] != BEFORE:
        raise ValueError('Installed mesh indexer receipt differs from its preimage')
    for target in (SITE / RELATIVE, COMPUTE / 'b12x-source' / RELATIVE):
        apply(target)
    for cached in (SITE / RELATIVE).parent.glob('__pycache__/fused_indexer.*.pyc'):
        cached.unlink()
    installed['b12x_files'][RELATIVE] = AFTER
    package_hash = hashlib.sha256(json.dumps(
        installed['b12x_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    lock = load(lock_path)
    lock['b12x']['package_files_sha256'] = package_hash
    transform = {'path': RELATIVE, 'preimage_sha256': BEFORE, 'result_sha256': AFTER,
                 'script_sha256': sha(Path(__file__).with_name('patch_indexer_barrier.py'))}
    lock['b12x']['source_transforms'] = {'histogram_publication_barrier': transform}
    write(lock_path, lock)
    lock_hash = sha(lock_path)
    installed['source_lock_sha256'] = lock_hash
    installed['b12x_package_files_sha256'] = package_hash
    write(INSTALLED, installed)
    prepared_path = COMPUTE / 'prepared-manifest.json'
    prepared = load(prepared_path)
    prepared['source_lock_sha256'] = lock_hash
    prepared['b12x_files'] = installed['b12x_files']
    prepared['b12x_package_files_sha256'] = package_hash
    write(prepared_path, prepared)
    profile_path = RECEIPTS / 'profile-pins.json'
    profile = load(profile_path)
    profile['compute']['source_lock_sha256'] = lock_hash
    profile['cache_identity']['namespace'] = profile['cache_identity']['namespace'].replace(
        PARENT_LOCK[:8], lock_hash[:8])
    profile['cache_identity']['compatibility'] = (
        f"Cache entries require compute source lock {lock_hash} and transport bundle "
        f"{profile['canonical_bundle_manifest_sha256']}. Do not relabel entries from another composition.")
    write(profile_path, profile)
    source_path.with_name('source-receipt.parent.json').write_bytes(source_path.read_bytes())
    source['files']['compute/b12x-source/' + RELATIVE] = AFTER
    for relative, path in (
        ('compute/source-lock.json', lock_path),
        ('compute/prepared-manifest.json', prepared_path),
        ('receipts/profile-pins.json', profile_path),
    ):
        source['files'][relative] = sha(path)
    source['derived_from_mesh_image'] = {
        'reference': PARENT, 'image_id': PARENT_ID, 'source_receipt_sha256': parent_source_sha}
    source['source_transforms'] = {'histogram_publication_barrier': transform}
    write(source_path, source)
    result = {'schema': 'sparkring-mesh-indexer-barrier/v1', 'status': 'research-only',
              'parent_image': PARENT, 'parent_image_id': PARENT_ID,
              'parent_source_lock_sha256': PARENT_LOCK,
              'source_lock_sha256': lock_hash, 'source_receipt_sha256': sha(source_path),
              'bundle_manifest_sha256': profile['canonical_bundle_manifest_sha256'],
              'cache_namespace': profile['cache_identity']['namespace'],
              'indexer': transform, 'installer_sha256': sha(Path(__file__))}
    write(RECEIPTS / 'indexer-barrier.json', result)
    subprocess.run([sys.executable, '-I', VERIFY, '--inside-image'], check=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-receipt')
    args = parser.parse_args()
    result = install()
    if args.expected_receipt and result['source_receipt_sha256'] != args.expected_receipt:
        raise ValueError('Mesh patch receipt differs from the prepared build input')
    if args.output:
        write(args.output, result)
    print(json.dumps(result), flush=True)
