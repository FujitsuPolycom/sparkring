"""Assemble an R35 Docker build context from staged, pinned source checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from prepare_sources import package

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from runtime.images.composition import validate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-source', type=Path, required=True)
    parser.add_argument('--b12x-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((ROOT/'source-lock.json').read_text(encoding='utf-8'))
    records = {}
    for name, expected in lock['components'].items():
        patch = ROOT/'patches'/expected['patch']
        if hashlib.sha256(patch.read_bytes()).hexdigest() != expected['patch_sha256']:
            raise ValueError(f'{name}: bundled patch does not match source lock')
        record = package(name, getattr(args, name+'_source').resolve(), output)
        for key in ('base_commit', 'base_tree', 'tree', 'patch_sha256'):
            if record[key] != expected[key]:
                raise ValueError(f'{name}: staged source differs from pinned {key}')
        records[name] = record
    composition = {'schema':'sparkring-r35-source-composition/v1', 'status':'candidate',
                   'components':records,
                   'qualification':'Source packaging only; serving qualification belongs to the tested image and profile.'}
    (output/'source-composition.json').write_text(json.dumps(composition,indent=2)+'\n',encoding='utf-8')
    for name in ('Dockerfile.source-stage', 'Dockerfile.candidate', 'install_sources.py',
                 'finalize_image.py', 'verify_image.py', 'entrypoint.py',
                 'compatibility.json', 'patch-ledger.json', 'source-lock.json'):
        shutil.copy2(ROOT/name, output/name)
    for path in (ROOT/'baseline-files').glob('r33-*-files.txt'):
        name = path.name.removeprefix('r33-').removesuffix('-files.txt')
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock['baseline_file_lists'][name]['sha256']:
            raise ValueError(f'{name}: baseline file inventory mismatch')
        shutil.copy2(path, output/path.name)
    for name in ('vllm-connector-jobs.json', 'tp2-sparkcache-capabilities.json', 'runtime-abi.json'):
        shutil.copy2(ROOT/'contracts'/name, output/name)
    shutil.copytree(ROOT.parents[1]/'sparkring/jovian-r33/profiles',output/'profile-contract')
    shutil.copytree(ROOT/'contracts/evidence', output/'capability-evidence')
    files = {str(p.relative_to(output)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'build-context-receipt.json').write_text(json.dumps({'schema':'sparkring-r35-build-context/v1',
        'parent_image':lock['parent_image'], 'parent_image_id':lock['parent_image_id'],
        'files':files},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output), 'trees':{name:r['tree'] for name,r in records.items()}}))


if __name__ == '__main__':
    main()
