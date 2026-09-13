"""Validate source-overlay provenance before assembling an image build context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or escaping contract input: {name}')
    return path


def patch_paths(patch):
    headers = [line for line in patch.decode('utf-8').splitlines() if line.startswith('diff --git ')]
    paths = []
    for line in headers:
        match = re.fullmatch(r'diff --git a/([^\s"]+) b/([^\s"]+)', line)
        if match is None:
            raise ValueError('Unsupported Git diff path syntax; inventory review required')
        paths.append(match[2])
    return paths


def read_baseline_receipt(recipe, path):
    raw = path.read_bytes()
    evidence = read(recipe / 'compatibility.json')['evidence']
    if hashlib.sha256(raw).hexdigest() != evidence['receipt_sha256']:
        raise ValueError('Baseline receipt bytes differ from recorded SHA256')
    return json.loads(raw)


def validate_ledger(recipe, lock):
    ledger = read(recipe / 'patch-ledger.json')
    if ledger.get('schema') != 'sparkring-patch-ledger/v1':
        raise ValueError('Unsupported patch ledger schema')
    if ledger.get('source_lock') != 'source-lock.json':
        raise ValueError('Patch ledger must reference source-lock.json')
    entries = ledger.get('components', {})
    if set(entries) != set(lock['components']):
        raise ValueError('Patch ledger must cover exactly the locked components')
    for name, source in lock['components'].items():
        entry = entries[name]
        expected_path = 'patches/' + source['patch']
        if entry.get('patch') != source['patch']:
            raise ValueError(f'{name}: ledger patch differs from source lock')
        patch = contained(recipe, expected_path).read_bytes()
        digest = hashlib.sha256(patch).hexdigest()
        if digest != source['patch_sha256'] or digest != entry.get('patch_sha256'):
            raise ValueError(f'{name}: patch hash mismatch')
        paths = patch_paths(patch)
        if not paths or paths != entry.get('changed_paths'):
            raise ValueError(f'{name}: patch ledger changed-path inventory mismatch')
        tests = entry.get('component_regression_tests', [])
        if not tests or any(path not in paths for path in tests):
            raise ValueError(f'{name}: regression tests must identify patched source tests')
        if not entry.get('purpose') or not entry.get('integrations_preserved'):
            raise ValueError(f'{name}: patch purpose and preserved integrations are required')
        for test in entry.get('regression_tests', []):
            contained(recipe.parents[2], test)
    return ledger


def validate(recipe, receipt=None):
    recipe = Path(recipe).resolve()
    lock = read(recipe / 'source-lock.json')
    ledger = validate_ledger(recipe, lock)
    compatibility = read(recipe / 'compatibility.json')
    if compatibility.get('schema') != 'sparkring-image-compatibility/v1':
        raise ValueError('Unsupported image compatibility schema')
    if compatibility.get('platform') != 'linux/arm64' or compatibility.get('framework') != 'vllm':
        raise ValueError('The R35 source overlay requires the Linux ARM64 vLLM foundation')
    if compatibility.get('foundation') != {'image': lock['parent_image'], 'image_id': lock['parent_image_id']}:
        raise ValueError('Compatibility foundation differs from source lock')
    refs = compatibility.get('source_lock', {})
    if refs.get('path') != 'source-lock.json' or refs.get('components') != {
        name: '/components/' + name for name in lock['components']
    }:
        raise ValueError('Compatibility components must reference the authoritative source lock')
    native = compatibility.get('inherited_native', {})
    if not native or any(not path.startswith('/') or not re.fullmatch('[0-9a-f]{64}', digest)
                         for path, digest in native.items()):
        raise ValueError('Inherited native libraries require absolute paths and SHA256 hashes')
    if receipt is not None:
        if receipt.get('schema') != compatibility['evidence']['receipt_schema']:
            raise ValueError('Baseline receipt schema mismatch')
        installed = receipt['installed']
        if receipt['image_id'] != compatibility['evidence']['image_id']:
            raise ValueError('Receipt is not the recorded baseline image')
        for name, expected in lock['components'].items():
            for key in ('base_commit', 'base_tree', 'tree', 'patch_sha256'):
                if installed['components'][name][key] != expected[key]:
                    raise ValueError(f'{name}: receipt differs from source lock {key}')
        for path, digest in native.items():
            if installed['files'].get(path) != digest:
                raise ValueError(f'Inherited native library mismatch: {path}')
        if installed['versions'].get('torch') != compatibility['abi']['torch_distribution_version']:
            raise ValueError('Baseline Torch distribution mismatch')
    return {'schema': 'sparkring-composition-check/v1',
            'components': list(ledger['components']),
            'baseline_receipt_checked': receipt is not None,
            'unknown_abi_fields': [key for key, value in compatibility['abi'].items() if value is None],
            'wheel_reuse_authorized': False,
            'scope': 'Source and patch provenance only; no serving qualification.'}


def assess_update(baseline, candidate):
    """Report required reviews; source metadata never authorizes binary reuse."""
    if candidate.get('schema') != baseline['schema']:
        raise ValueError('Source-lock schema changes require an explicit composition design')
    other_changes = sorted(key for key in baseline.keys() | candidate.keys()
                           if key not in ('schema', 'parent_image', 'parent_image_id', 'components')
                           and baseline.get(key) != candidate.get(key))
    foundation_changed = any(baseline[key] != candidate[key]
                             for key in ('parent_image', 'parent_image_id'))
    if set(baseline['components']) != set(candidate['components']):
        raise ValueError('Component additions/removals require an explicit composition design')
    changed = [name for name in baseline['components']
               if baseline['components'][name] != candidate['components'][name]]
    return {'foundation_changed': foundation_changed, 'changed_components': changed,
            'other_lock_changes': other_changes,
            'required_actions': (
                ['Review platform, Python, CUDA, Torch and native ABI; rebuild affected extensions.']
                if foundation_changed else []) + (
                ['Review changed lock metadata and inherited package-file removal inventories.']
                if other_changes else []) + (
                ['Review upstream overlap against every patch-ledger entry.',
                 'Compare complete native build inputs before selecting rebuild or reuse.',
                 'Refresh source lock, ledger and installed connector contracts together.',
                 'Build isolated candidate and collect fresh image and serving evidence.']
                if changed or foundation_changed or other_changes else []),
            'binary_reuse_authorized': False,
            'scope': 'Offline update planning only; no source fetch, build or deployment.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--baseline-receipt', type=Path,
                        help='Compare the saved R35 admission receipt; does not verify a live image')
    parser.add_argument('--candidate-lock', type=Path,
                        help='Report required reviews for a proposed source lock; never installs it')
    args = parser.parse_args()
    try:
        receipt = read_baseline_receipt(args.recipe, args.baseline_receipt) if args.baseline_receipt else None
        result = validate(args.recipe, receipt)
        if args.candidate_lock:
            result['update'] = assess_update(read(args.recipe / 'source-lock.json'), read(args.candidate_lock))
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
