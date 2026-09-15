"""Package exact staged source trees after checking inherited native build inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess

COMPONENTS = {'vllm', 'b12x'}


def git(source, *args):
    try:
        return subprocess.check_output(
            ['git', '-C', str(source), '--literal-pathspecs', *args], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Git source inspection failed; required objects and references must exist') from exc


def _oid(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', value)


def _validate(name, spec):
    if name not in COMPONENTS:
        raise ValueError('Only vllm and b12x source components are admitted')
    if not isinstance(spec, dict):
        raise ValueError('A pinned source specification is required')
    for field in ('base_commit', 'base_tree', 'tree'):
        if not _oid(spec.get(field)):
            raise ValueError(f'{field} requires a full Git object ID')
    if not isinstance(spec.get('patch_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', spec['patch_sha256']):
        raise ValueError('patch_sha256 requires a lowercase SHA256 digest')
    comparison = spec.get('native_comparison')
    if not isinstance(comparison, dict) or not _oid(comparison.get('reference')):
        raise ValueError('native_comparison requires an integrated foundation reference')
    paths = comparison.get('paths')
    if not isinstance(paths, list) or not paths:
        raise ValueError('Native build input paths must be explicit and nonempty')
    for path in paths:
        if (not isinstance(path, str) or not path or path == '.' or '\\' in path
                or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                or any(char in path for char in ('*', '?', '[', ':', '\n', '\r'))):
            raise ValueError('Native build input paths must be literal repository-relative paths')
    if len(set(paths)) != len(paths):
        raise ValueError('Native build input paths must be unique')
    return comparison


def package(name, source, output, spec):
    """Write a deterministic archive and patch; no builds or runtime admission.

    The comparison reference must identify the integrated foundation sources.
    The caller owns completeness of its native/build dependency path inventory.
    """
    comparison = _validate(name, spec)
    source, output = Path(source).resolve(), Path(output).resolve()
    if not output.is_dir() or output.is_relative_to(source):
        raise ValueError('Output must be an existing directory outside the source checkout')
    if git(source, 'rev-parse', 'HEAD').decode().strip() != spec['base_commit']:
        raise ValueError('Checkout HEAD differs from the pinned base commit')
    if git(source, 'rev-parse', 'HEAD^{tree}').decode().strip() != spec['base_tree']:
        raise ValueError('Base tree differs from the pinned specification')
    if (git(source, 'diff', '--no-ext-diff', '--name-only')
            or git(source, 'ls-files', '--others', '--exclude-standard')):
        raise ValueError('Source checkout contains unstaged or untracked changes')
    tree = git(source, 'write-tree').decode().strip()
    if tree != spec['tree']:
        raise ValueError('Staged tree differs from the pinned specification')
    patch = git(source, 'diff', '--cached', '--binary', '--no-ext-diff', '--no-textconv',
                '--full-index', '--no-renames', '--src-prefix=a/', '--dst-prefix=b/',
                '--no-color', spec['base_commit'], '--')
    if hashlib.sha256(patch).hexdigest() != spec['patch_sha256']:
        raise ValueError('Staged patch SHA256 differs from the pinned specification')
    reference = comparison['reference']
    git(source, 'rev-parse', '--verify', reference + '^{tree}')
    for path in comparison['paths']:
        if not (git(source, 'ls-tree', '-r', '--name-only', reference, '--', path)
                or git(source, 'ls-tree', '-r', '--name-only', tree, '--', path)):
            raise ValueError(f'Native build input path is absent from both trees: {path}')
    changed = git(source, 'diff', '--no-ext-diff', '--name-only', reference, tree,
                  '--', *comparison['paths'])
    if changed:
        raise ValueError('Native/build inputs differ from integrated foundation; '
                         'rebuild required: ' + changed.decode().strip().replace('\n', ', '))
    archive_name, patch_name = f'{name}-{tree}.tar.gz', f'{name}-sparkring.patch'
    if (output / archive_name).exists() or (output / patch_name).exists():
        raise ValueError('Refusing to overwrite packaged source artifacts')
    archive = git(source, '-c', 'core.autocrlf=false', '-c', 'tar.umask=0022',
                  'archive', '--format=tar.gz', '--mtime=2000-01-01T00:00:00Z', tree)
    with (output / archive_name).open('xb') as stream:
        stream.write(archive)
    with (output / patch_name).open('xb') as stream:
        stream.write(patch)
    return {
        **{key: spec[key] for key in ('base_commit', 'base_tree', 'tree', 'patch_sha256')},
        'archive': archive_name, 'archive_sha256': hashlib.sha256(archive).hexdigest(),
        'patch': patch_name,
        'native_comparison': {'reference': reference, 'paths': list(comparison['paths']),
                              'unchanged': True, 'compared_tree': tree},
        'qualification': 'Source packaging only; runtime verification and serving qualification required.',
    }
