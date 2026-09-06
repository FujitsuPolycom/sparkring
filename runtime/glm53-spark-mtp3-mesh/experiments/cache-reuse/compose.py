"""Build hash-verified MTP3 Python source output in an unused local directory.

The candidate directory is transformed source output. Its hashes verify source
composition; serving qualification requires separate complete-image evidence.
The tool performs no Docker, network, or model operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import patch_mtp3_barrier as barrier
import patch_mtp3_lease_accounting as accounting
import patch_mtp3_local_lease_preference as preference
import patch_mtp3_partial_tail_eligibility as partial
import patch_mtp3_sparse_retention as retention

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
NATIVE_PLACEMENT_SHA256 = '2657cdd2e54a097c9544e4c79ae62c0646db6db123ff24e4f0c384238c3a1e8d'
FINAL_SHA256 = {
    'vllm/v1/core/sched/scheduler.py': partial.AFTER_SHA256,
    'vllm/v1/core/single_type_kv_cache_manager.py': retention.MANAGER_AFTER,
    'b12x/attention/dsa_indexer/fused_indexer.py': barrier.AFTER_SHA256,
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def manifest():
    return json.loads((FIXTURES / 'manifest.json').read_text(encoding='utf-8'))


def source_bytes(source_root=None):
    """Validate every input before creating any output files."""
    metadata = manifest()
    if source_root is None:
        archive = FIXTURES / 'original-python-sources.zip'
        if sha256(archive.read_bytes()) != metadata['archive_sha256']:
            raise ValueError('Source fixture archive checksum differs')
        with zipfile.ZipFile(archive) as bundle:
            if sorted(bundle.namelist()) != sorted(metadata['files']):
                raise ValueError('Source fixture archive members differ')
            content = {name: bundle.read(name) for name in metadata['files']}
    else:
        content = {name: (Path(source_root) / name).read_bytes() for name in metadata['files']}
    for name, data in content.items():
        if sha256(data) != metadata['files'][name]['sha256']:
            raise ValueError(f'Original source checksum differs: {name}')
    for name, expected in metadata['transform_script_sha256'].items():
        if sha256((HERE / name).read_bytes()) != expected:
            raise ValueError(f'Attested transform script differs: {name}')
    return content


def verify_candidate(root):
    metadata = manifest()
    verified = {}
    for name, original in metadata['files'].items():
        expected = FINAL_SHA256.get(name, original['sha256'])
        observed = sha256((Path(root) / name).read_bytes())
        if observed != expected:
            raise ValueError(f'Candidate source checksum differs: {name}')
        verified[name] = observed
    return verified


def compose(output, *, source_root=None):
    content = source_bytes(source_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original, candidate = output / 'original', output / 'candidate'
    for name, data in content.items():
        path = original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    shutil.copytree(original, candidate)
    scheduler = candidate / 'vllm/v1/core/sched/scheduler.py'
    steps = [barrier.apply_patch(candidate / 'b12x/attention/dsa_indexer/fused_indexer.py'),
             accounting.apply_patch(scheduler), preference.apply_patch(scheduler),
             retention.apply_patch(candidate / 'vllm'), partial.apply_patch(scheduler)]
    result = {'schema': 'sparkring-mtp3-cache-reuse-composition/v1', 'status': 'research-only',
              'base_image_id': manifest()['base_image_id'], 'speculation': 'native MTP3',
              'topology': 'TP4/DCP4, 512-token recurrent/hash pages',
              'candidate_files': verify_candidate(candidate), 'transform_receipts': steps,
              'required_native_placement_sha256': NATIVE_PLACEMENT_SHA256,
              'limits': ['Python source composition only; no SparkCache package or native binary is installed',
                         'No serving, transport, GPU, or throughput qualification is implied',
                         'Published runtime pins and production Dockerfiles are unchanged']}
    (output / 'composition.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Original fixture layout from the exact base image; defaults to bundled fixtures')
    parser.add_argument('--output-root', type=Path, help='Fresh destination; existing directories are rejected')
    parser.add_argument('--check', action='store_true', help='Compose and verify only in a temporary directory')
    parser.add_argument('--verify-candidate', type=Path, help='Read-only verification of an already composed candidate directory')
    args = parser.parse_args()
    try:
        if args.verify_candidate:
            result = {'verified_files': verify_candidate(args.verify_candidate)}
        elif args.check:
            with tempfile.TemporaryDirectory(prefix='mtp3-cache-reuse-') as temporary:
                result = compose(Path(temporary) / 'composition', source_root=args.source_root)
        elif args.output_root:
            result = compose(args.output_root, source_root=args.source_root)
        else:
            parser.error('Choose --check, --output-root, or --verify-candidate')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
