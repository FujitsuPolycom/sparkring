"""Exercise source admission against real Git trees and non-executable fixtures."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('candidate_sources', Path(__file__).with_name('candidate_sources.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE).decode().strip()


def pins(repo, foundation):
    patch = m.git(repo, 'diff', '--cached', '--binary', '--no-ext-diff', '--no-textconv',
                  '--full-index', '--no-renames', '--src-prefix=a/', '--dst-prefix=b/', '--no-color', 'HEAD', '--')
    return dict(base_commit=git(repo, 'rev-parse', 'HEAD'), base_tree=git(repo, 'rev-parse', 'HEAD^{tree}'),
                tree=git(repo, 'write-tree'), patch_sha256=hashlib.sha256(patch).hexdigest(),
                native_comparison=dict(reference=foundation, paths=['csrc', 'setup.py']))


@pytest.fixture
def sources(tmp_path):
    repo = tmp_path / 'source'
    repo.mkdir()
    git(repo, 'init')
    git(repo, 'config', 'user.name', 'Fixture')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    git(repo, 'config', 'core.autocrlf', 'false')
    (repo / 'csrc').mkdir()
    (repo / 'csrc/kernel.cc').write_text('native fixture\n')
    (repo / 'setup.py').write_text('# build fixture\n')
    (repo / 'vllm').mkdir()
    (repo / 'vllm/api.py').write_text('# source fixture\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'Fixture foundation')
    foundation = git(repo, 'rev-parse', 'HEAD^{tree}')
    (repo / 'vllm/api.py').write_text('# patched source fixture\n')
    git(repo, 'add', '.')
    output = tmp_path / 'output'
    output.mkdir()
    return repo, output, pins(repo, foundation)


def test_identical_native_inputs_and_reproducible_archive(sources, tmp_path):
    repo, output, spec = sources
    first = m.package('vllm', repo, output, spec)
    second_output = tmp_path / 'second'
    second_output.mkdir()
    second = m.package('vllm', repo, second_output, spec)
    assert first == second
    assert (output / first['archive']).read_bytes() == (second_output / second['archive']).read_bytes()
    assert first['native_comparison']['compared_tree'] == spec['tree']


def test_staged_native_patch_rejected_even_when_base_unchanged(sources):
    repo, output, spec = sources
    base = spec['base_commit']
    (repo / 'csrc/kernel.cc').write_text('changed native fixture\n')
    git(repo, 'add', '.')
    candidate = pins(repo, spec['native_comparison']['reference'])
    assert candidate['base_commit'] == base
    with pytest.raises(ValueError, match='rebuild required'):
        m.package('vllm', repo, output, candidate)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize('reference', [None, 'main', 'f' * 40])
def test_missing_or_unknown_foundation_rejected(sources, reference):
    repo, output, spec = sources
    spec['native_comparison']['reference'] = reference
    with pytest.raises(ValueError):
        m.package('vllm', repo, output, spec)
    assert not list(output.iterdir())


@pytest.mark.parametrize('kind', ['unstaged', 'untracked'])
def test_dirty_checkout_rejected(sources, kind):
    repo, output, spec = sources
    path = repo / ('vllm/api.py' if kind == 'unstaged' else 'untracked.txt')
    path.write_text('unexpected\n')
    with pytest.raises(ValueError, match='unstaged or untracked'):
        m.package('vllm', repo, output, spec)


@pytest.mark.parametrize('field', ['base_commit', 'base_tree', 'tree', 'patch_sha256'])
def test_pins_are_checked(sources, field):
    repo, output, spec = sources
    spec[field] = '0' * len(spec[field])
    with pytest.raises(ValueError):
        m.package('vllm', repo, output, spec)
    assert not list(output.iterdir())


@pytest.mark.parametrize('paths', [[], ['../outside'], [':(exclude)csrc'], ['missing']])
def test_invalid_native_inventory_rejected(sources, paths):
    repo, output, spec = sources
    spec['native_comparison']['paths'] = paths
    with pytest.raises(ValueError):
        m.package('vllm', repo, output, spec)


def test_component_allowlist_and_overwrite(sources):
    repo, output, spec = sources
    with pytest.raises(ValueError, match='Only vllm and b12x'):
        m.package('../escape', repo, output, spec)
    m.package('b12x', repo, output, spec)
    with pytest.raises(ValueError, match='overwrite'):
        m.package('b12x', repo, output, spec)
