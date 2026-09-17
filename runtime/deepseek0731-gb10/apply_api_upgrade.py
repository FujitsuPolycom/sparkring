#!/usr/bin/env python3
"""Upgrade API policies on the pinned native DeepSeek derivative."""
import argparse
import json
from pathlib import Path

import apply_runtime_overlay as overlay

API_PATHS = frozenset({
    'vllm/entrypoints/openai/responses/protocol.py',
    'vllm/entrypoints/openai/chat_completion/serving.py',
})
NATIVE_SHA256 = '58c4c2e7acdbbacc58d41f1326f63e10b8bc9d0dbf406a2dea22234d7f64dad2'


def prepare(root, contract, contract_path):
    record = contract['runtime_patch']
    patch_path = contract_path.parent / record['path']
    if overlay.sha256_file(patch_path) != record['sha256']:
        raise overlay.OverlayError('runtime patch hash differs')
    patches = overlay.parse_unified_patch(patch_path.read_text(encoding='utf-8'))
    if set(patches) != {item['path'] for item in record['files']}:
        raise overlay.OverlayError('runtime patch path set differs')
    if not API_PATHS.issubset(patches):
        raise overlay.OverlayError('API patch paths are missing')
    prepared, states = [], []
    for item in record['files']:
        path = overlay._target(root, item['path'])
        source = path.read_bytes()
        observed = overlay.sha256_bytes(source)
        if item['path'] not in API_PATHS:
            if observed != item['result_sha256']:
                raise overlay.OverlayError(f'prerequisite runtime fix differs: {path}')
            continue
        if observed == item['result_sha256']:
            states.append('result')
            continue
        if observed != item['preimage_sha256']:
            raise overlay.OverlayError(f'API source differs: {path}')
        states.append('preimage')
        result = overlay.apply_file_patch(source, patches[item['path']])
        if overlay.sha256_bytes(result) != item['result_sha256']:
            raise overlay.OverlayError(f'API result hash differs: {path}')
        compile(result.decode('utf-8'), str(path), 'exec')
        prepared.append(overlay.PreparedFile(path, result, path.stat().st_mode))
    if len(set(states)) != 1:
        raise overlay.OverlayError('API patch is partially applied')
    return prepared


def upgrade(contract_path, site_root, source_root, native_path, receipt):
    if receipt.exists():
        raise overlay.OverlayError('receipt already exists')
    if overlay.sha256_file(native_path) != NATIVE_SHA256:
        raise overlay.OverlayError('published native library hash differs')
    contract = overlay.load_contract(contract_path)
    overlay.attest_noop_files(site_root, contract['attested_noop'])
    # Validate both trees before writing either; unrelated/mixed states fail closed.
    installed = prepare(site_root, contract, contract_path)
    retained = prepare(source_root, contract, contract_path)
    if bool(installed) != bool(retained):
        raise overlay.OverlayError('installed and retained API states differ')
    overlay.commit_patch_set(installed + retained)
    if overlay.sha256_file(native_path) != NATIVE_SHA256:
        raise overlay.OverlayError('native library changed during API upgrade')
    receipt.write_text(json.dumps({
        'schema': 'sparkring-deepseek-api-upgrade/v1',
        'status': 'applied' if installed else 'already-applied',
        'contract_sha256': overlay.sha256_file(contract_path),
        'preserved_native_sha256': NATIVE_SHA256,
        'api_files': {item['path']: item['result_sha256']
                      for item in contract['runtime_patch']['files']
                      if item['path'] in API_PATHS},
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--site-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    upgrade(args.contract, args.site_root, args.source_root, args.native, args.receipt)
