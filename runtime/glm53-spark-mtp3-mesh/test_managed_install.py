"""Validate immutable container ownership and the cluster stop barrier."""
from pathlib import Path
import hashlib
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import managed_cluster  # noqa: E402
import managed_install  # noqa: E402


def container():
    return {'Id': 'a' * 64, 'Name': '/profile-r0', 'Image': 'sha256:' + 'b' * 64,
            'State': {'Running': False}, 'HostConfig': {'RestartPolicy': {'Name': 'no'}}}


def test_installer_accepts_only_stopped_pinned_model():
    managed_install.validate_container(container(), {'container_prefix': 'profile'}, 0, 'sha256:' + 'b' * 64)


@pytest.mark.parametrize('change', ['name', 'image', 'running', 'restart', 'id'])
def test_installer_rejects_ambiguous_ownership(change):
    value = container()
    if change == 'name':
        value['Name'] = '/another-profile-r0'
    elif change == 'image':
        value['Image'] = 'sha256:' + 'c' * 64
    elif change == 'running':
        value['State']['Running'] = True
    elif change == 'restart':
        value['HostConfig']['RestartPolicy']['Name'] = 'always'
    else:
        value['Id'] = 'short'
    with pytest.raises(ValueError):
        managed_install.validate_container(value, {'container_prefix': 'profile'}, 0, 'sha256:' + 'b' * 64)


def test_down_never_removes_fabric_before_all_model_stops():
    phases = managed_cluster.phases('down', '/opt/sparkring/managed-mesh', '/etc/sparkring/managed-mesh')
    names = [name for name, _ in phases]
    assert names.index('quiesce-models') < names.index('stop-model-units')
    assert names.index('model-stop-barrier') < names.index('stop-mesh-units')
    assert names.index('stop-mesh-units') < names.index('recover-owned-children')


def test_recover_does_not_restart_models():
    phases = managed_cluster.phases('recover', '/opt/sparkring/managed-mesh', '/etc/sparkring/managed-mesh')
    names = [name for name, _ in phases]
    assert names[-2:] == ['start-mesh-units', 'four-rank-ready']
    assert 'start-model-units' not in names


def test_clear_latch_does_not_reset_unloaded_nonfailed_units(monkeypatch):
    service = managed_install.managed_units.service
    calls = []
    class Result:
        stdout = 'LoadState=loaded\nActiveState=inactive\n'
    monkeypatch.setattr(service.subprocess, 'run', lambda argv, **kwargs: calls.append(argv) or Result())
    service.reset_units()
    assert len(calls) == 2
    assert all(call[1] == 'show' for call in calls)


@pytest.fixture
def source_copy(tmp_path, monkeypatch):
    original = managed_install.ROOT
    source = tmp_path / 'source'
    for relative in managed_install.SOURCE_FILES:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)
    monkeypatch.setattr(managed_install, 'ROOT', source)
    return source


def test_only_allowlisted_files_are_installed(source_copy, tmp_path):
    extra = source_copy / 'runtime/glm53-spark-mtp3-mesh/health.key'
    extra.write_text('synthetic-secret-must-not-be-disclosed')
    (extra.parent / 'operator-site.json').write_text('synthetic-private-site')
    destination = tmp_path / 'installed'
    hashes = managed_install.install_code(managed_install.source_payloads(), destination)
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob('*') if path.is_file()}
    assert actual == set(managed_install.SOURCE_FILES) == set(hashes)
    assert all(hashlib.sha256((destination / name).read_bytes()).hexdigest() == value
               for name, value in hashes.items())
    assert not (destination / extra.relative_to(source_copy)).exists()
    assert all(b'synthetic-secret-must-not-be-disclosed' not in path.read_bytes()
               for path in destination.rglob('*') if path.is_file())


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_source_symlinks_rejected_before_destination_write(source_copy, tmp_path, monkeypatch, kind):
    link = source_copy / managed_install.SOURCE_FILES[0]
    if kind == 'directory':
        link = link.parent
    moved = tmp_path / ('external-' + kind)
    link.rename(moved)
    try:
        link.symlink_to(moved, target_is_directory=(kind == 'directory'))
    except OSError:
        pytest.skip('Creating symlinks requires operating-system permission')
    destination = tmp_path / 'installed'
    monkeypatch.setattr(managed_install, 'CODE_DIR', destination)
    monkeypatch.setattr(managed_install.os, 'geteuid', lambda: 0, raising=False)
    monkeypatch.setattr(managed_install.managed_units.service, 'read_key', lambda path: b'x' * 32)
    with pytest.raises(ValueError, match='nonsymlink'):
        managed_install.apply({}, tmp_path, tmp_path / 'key')
    assert not destination.exists()


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_symlink_metadata_fails_preflight(source_copy, tmp_path, monkeypatch, kind):
    link = source_copy / managed_install.SOURCE_FILES[0]
    if kind == 'directory':
        link = link.parent
    original = Path.lstat
    monkeypatch.setattr(Path, 'lstat', lambda path, *a, **k:
                        SimpleNamespace(st_mode=stat.S_IFLNK) if path == link else original(path, *a, **k))
    destination = tmp_path / 'installed'
    with pytest.raises(ValueError, match='nonsymlink'):
        managed_install.install_code(managed_install.source_payloads(), destination)
    assert not destination.exists()


def test_snapshot_does_not_copy_source_changes_after_validation(source_copy, tmp_path):
    payloads = managed_install.source_payloads()
    name = managed_install.SOURCE_FILES[0]
    (source_copy / name).write_text('synthetic-secret-written-after-validation')
    destination = tmp_path / 'installed'
    managed_install.install_code(payloads, destination)
    assert (destination / name).read_bytes() == payloads[name]


def test_install_snapshot_rejects_extra_entries_before_write(source_copy, tmp_path):
    payloads = managed_install.source_payloads()
    payloads['private.key'] = b'synthetic-secret'
    destination = tmp_path / 'installed'
    with pytest.raises(ValueError, match='allowlist'):
        managed_install.install_code(payloads, destination)
    assert not destination.exists()


def test_installed_runtime_imports_without_checkout_or_gpu(source_copy, tmp_path):
    destination = tmp_path / 'installed'
    managed_install.install_code(managed_install.source_payloads(), destination)
    code = '''
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / 'runtime/glm53-spark-mtp3-mesh'))
for name in ('managed_service', 'managed_network', 'managed_units', 'managed_cluster', 'managed_install'):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root)
assert 'torch' not in sys.modules
print('CPU-only installed imports passed')
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(destination)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'CPU-only installed imports passed' in result.stdout
