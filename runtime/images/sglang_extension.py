"""Prepare an isolated SGLang runtime layer over a pinned SparkRing vLLM image.

Preparation uses an exact local Mia source checkout and an exact NCCL library.
It downloads nothing, runs no Docker commands, and leaves the checkout intact.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'runtime/deepseek-v41-sglang'
PACKAGE = RUNTIME / 'combined-image'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin():
    return json.loads((PACKAGE / 'manifest.json').read_bytes())


def adapter_sources(checkout, manifest):
    """Archive committed source, independent of worktree edits and untracked files."""
    actual = subprocess.check_output(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != manifest['adapter']['commit']:
        raise ValueError('Mia source HEAD differs from the pinned adapter commit')
    archive = subprocess.check_output(['git', '-C', str(checkout), 'archive', actual])
    return archive


def adapt_mia(directory, manifest):
    """Apply two guarded integration edits without changing model computation."""
    changes = {}
    file = directory / 'boot.py'
    source = file.read_bytes()
    before = b"REVISION = 'fb2764a5cf321eaa5070ca8f9e892818f477c16d'"
    if source.count(before) != 1:
        raise ValueError('Mia checkpoint revision assignment differs from the pinned source')
    result = source.replace(before, f"REVISION = '{manifest['model']['revision']}'".encode())
    changes['boot.py'] = {'before': sha(source), 'after': sha(result)}
    file.write_bytes(result)
    file = directory / 'adapter/engram_backend.py'
    original = file.read_bytes()
    source = original.replace(b'\r\n', b'\n')
    start = source.index(b'def _load_cudart():\n')
    end = source.index(b'    seen, errors = set(), []\n', start)
    result = source[:start] + (
        b'def _load_cudart():\n'
        b"    # One CUDA runtime must own Torch work and Engram host callbacks.\n"
        b"    site = Path(torch.__file__).resolve().parent.parent\n"
        b"    paths = [str(site / 'nvidia/cu13/lib/libcudart.so.13')]\n"
    ) + source[end:]
    changes['adapter/engram_backend.py'] = {'before': sha(original), 'after': sha(result)}
    file.write_bytes(result)
    return changes


def prepare(mia_source, nccl, output):
    manifest = pin()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Build context must not already exist')
    with Path(nccl).open('rb') as stream:
        fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
    if fingerprint != manifest['nccl']['sha256']:
        raise ValueError('NCCL library differs from the pinned SparkRing build')
    archive = adapter_sources(mia_source, manifest)
    # Validate all source paths before materializing an externally maintained tree.
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name
                    or PureWindowsPath(member.name).drive
                    or not (member.isfile() or member.isdir())):
                raise ValueError('Mia archive contains an unsupported path or link')
        output.mkdir(parents=True)
        mia = output / 'mia'
        mia.mkdir()
        tar.extractall(mia, filter='data')
    changes = adapt_mia(mia, manifest)
    shutil.copy2(nccl, output / 'libnccl.so.2')
    shutil.copytree(RUNTIME / 'patches', output / 'patches')
    # A build result is evidence about an image, never an input to its successor.
    shutil.copytree(PACKAGE, output / 'runtime', ignore=shutil.ignore_patterns('local-build.json'))
    for name in ('entrypoint.py', 'patch-multikey.py'):
        shutil.copy2(RUNTIME / name, output / 'runtime' / name)
    (output / 'Dockerfile').write_bytes((PACKAGE / 'Dockerfile').read_bytes().replace(b'\r\n', b'\n'))
    (output / '.dockerignore').write_text('.git\n__pycache__\n*.pyc\n', encoding='utf-8')
    receipt = {'schema': 'sparkring-sglang-context/v1', 'composition': manifest,
               'adapter_modifications': changes,
               'inputs': {str(p.relative_to(output)).replace('\\', '/'): sha(p.read_bytes())
                         for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'runtime/context.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mia-source', required=True, type=Path)
    parser.add_argument('--nccl', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.mia_source, args.nccl, args.output)
    print(json.dumps({'context': str(args.output.resolve()),
                      'id': result['composition']['id'],
                      'inputs': len(result['inputs'])}, indent=2))


if __name__ == '__main__':
    main()
