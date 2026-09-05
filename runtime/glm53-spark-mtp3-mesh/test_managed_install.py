"""Validate immutable container ownership and the cluster stop barrier."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
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


@pytest.fixture
def exact_container_spec():
    image_id = 'sha256:' + 'b' * 64
    image = {'Id': image_id, 'Config': {'Env': ['PATH=/usr/bin', 'IMAGE_DEFAULT=yes', 'OVERRIDE=image'],
                                       'Labels': {'image-label': 'kept'}, 'User': '', 'WorkingDir': '/workspace'}}
    argv = ['docker', 'create', '--name', 'profile-r0', '--entrypoint', '/serve', '--init',
            '-e', 'OVERRIDE=launcher', '-e', 'MTP_DEPTH=3', '-e', 'EMPTY=',
            '-v', '/srv/model:/models/target:ro', '-v', '/srv/bundle:/opt/spark-sircl:ro',
            '-v', '/srv/cache:/cache/jit', '--label', 'rank=0', image_id,
            '/models/target', '--max-num-seqs', '16', '--speculative-config', '{"method":"mtp","num_speculative_tokens":3}']
    expected = managed_install.expected_container_spec(argv, image)
    actual = {'Name': '/profile-r0', 'Image': image_id, 'Config': {
        'Cmd': expected['cmd'][:], 'Entrypoint': expected['entrypoint'][:],
        'Env': [f'{key}={value}' for key, value in expected['env'].items()],
        'Labels': expected['labels'].copy(), 'WorkingDir': '/workspace', 'User': ''},
        'Mounts': [{'Destination': destination, **mount} for destination, mount in expected['mounts'].items()]}
    return argv, image, expected, actual


def test_complete_spec_accepts_exact_configuration_and_image_env_override(exact_container_spec):
    _, _, expected, actual = exact_container_spec
    assert expected['env']['OVERRIDE'] == 'launcher'
    assert expected['env']['IMAGE_DEFAULT'] == 'yes'
    actual['Config']['Env'].reverse()
    actual['Mounts'].reverse()
    managed_install.validate_container_spec(actual, expected)


@pytest.mark.parametrize('change', [
    'model_argument', 'extra_argument', 'entrypoint', 'missing_env', 'changed_env', 'extra_unsafe_env',
    'extra_innocuous_env', 'duplicate_env', 'writable_model', 'writable_bundle', 'wrong_cache',
    'missing_mount', 'extra_mount', 'volume_instead_of_bind', 'duplicate_mount', 'user', 'working_dir', 'labels'])
def test_container_spec_rejects_configuration_drift(exact_container_spec, change):
    _, _, expected, value = exact_container_spec
    actual = deepcopy(value)
    if change == 'model_argument':
        actual['Config']['Cmd'][2] = '32'
    elif change == 'extra_argument':
        actual['Config']['Cmd'] += ['--enforce-eager']
    elif change == 'entrypoint':
        actual['Config']['Entrypoint'] = ['/bin/sh']
    elif change == 'missing_env':
        actual['Config']['Env'].pop()
    elif change == 'changed_env':
        actual['Config']['Env'][0] = 'PATH=/malicious'
    elif change == 'extra_unsafe_env':
        actual['Config']['Env'].append('LD_PRELOAD=/malicious.so')
    elif change == 'extra_innocuous_env':
        actual['Config']['Env'].append('UNREVIEWED_SETTING=1')
    elif change == 'duplicate_env':
        actual['Config']['Env'].append(actual['Config']['Env'][0])
    elif change in ('writable_model', 'writable_bundle'):
        actual['Mounts'][0 if change == 'writable_model' else 1]['RW'] = True
    elif change == 'wrong_cache':
        actual['Mounts'][2]['Source'] = '/other-cache'
    elif change == 'missing_mount':
        actual['Mounts'].pop()
    elif change == 'extra_mount':
        actual['Mounts'].append({'Destination': '/host', 'Source': '/', 'Type': 'bind', 'RW': True})
    elif change == 'volume_instead_of_bind':
        actual['Mounts'][0]['Type'] = 'volume'
    elif change == 'duplicate_mount':
        actual['Mounts'].append(actual['Mounts'][0])
    elif change == 'user':
        actual['Config']['User'] = '1000'
    elif change == 'working_dir':
        actual['Config']['WorkingDir'] = '/tmp'
    else:
        actual['Config']['Labels']['rank'] = '1'
    with pytest.raises(ValueError):
        managed_install.validate_container_spec(actual, expected)


def test_environment_drift_does_not_disclose_values(exact_container_spec):
    _, _, expected, actual = exact_container_spec
    actual['Config']['Env'].append('API_KEY=do-not-disclose-this-value')
    with pytest.raises(ValueError) as captured:
        managed_install.validate_container_spec(actual, expected)
    assert 'API_KEY' in str(captured.value)
    assert 'do-not-disclose-this-value' not in str(captured.value)


@pytest.mark.parametrize('change', ['unknown_flag', 'duplicate_env', 'inherited_env', 'duplicate_mount', 'anonymous_volume'])
def test_expected_envelope_rejects_unsupported_ambiguity(exact_container_spec, change):
    argv, image, _, _ = exact_container_spec
    if change == 'unknown_flag':
        argv[2:2] = ['--privileged']
    elif change == 'duplicate_env':
        argv[2:2] = ['-e', 'MTP_DEPTH=4']
    elif change == 'inherited_env':
        argv[2:2] = ['-e', 'HOST_SECRET']
    elif change == 'duplicate_mount':
        argv[2:2] = ['-v', '/other:/models/target:ro']
    else:
        image['Config']['Volumes'] = {'/hidden': {}}
    with pytest.raises(ValueError):
        managed_install.expected_container_spec(argv, image)


@pytest.fixture
def canonical_inputs(tmp_path, monkeypatch, exact_container_spec):
    argv, image, expected, _ = exact_container_spec
    launch = tmp_path / 'launch'
    launch.mkdir()
    (launch / 'launch-rank.sh').write_bytes(b'trusted launcher\n')
    (launch / 'rank0.env').write_bytes(b'CANONICAL=1\n')
    profile = managed_install.managed_units.service.mesh_profile
    monkeypatch.setattr(profile, 'load_site', lambda path: ({'bundle_root': '/bundle'}, None, None))
    def render(site, bundle, output, receipt):
        output.mkdir()
        (output / 'launch-rank.sh').write_bytes(b'trusted launcher\n')
        (output / 'rank0.env').write_bytes(b'CANONICAL=1\n')
    monkeypatch.setattr(profile, 'render', render)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            'schema': 'sparkring-container-command/v1', 'argv': argv}))
    return launch, image, expected, run, calls


def test_canonical_spec_runs_only_regenerated_inputs_in_sanitized_environment(canonical_inputs, monkeypatch):
    launch, image, expected, run, calls = canonical_inputs
    monkeypatch.setenv('API_KEYS_FILE', '/ambient/private.key')
    monkeypatch.setenv('BASH_ENV', '/ambient/inject.sh')
    actual = managed_install.canonical_container_spec(launch, Path('/receipt'), 0, image, run=run)
    assert actual == expected
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == '/bin/bash'
    assert Path(command[1]).parent != launch
    assert set(options['env']) == {'PATH', 'LC_ALL', 'SPARKRING_CREATE_ONLY', 'SPARKRING_PRINT_CONTAINER_SPEC'}
    assert options['env']['SPARKRING_PRINT_CONTAINER_SPEC'] == '1'
    assert options['env']['SPARKRING_CREATE_ONLY'] == '1'
    assert not Path(command[1]).exists()


@pytest.mark.parametrize('name', ['launch-rank.sh', 'rank0.env'])
def test_canonical_spec_rejects_changed_launch_before_execution(canonical_inputs, name):
    launch, image, _, run, calls = canonical_inputs
    (launch / name).write_text('exit 0 # untrusted drift')
    with pytest.raises(ValueError, match='canonical rendered'):
        managed_install.canonical_container_spec(launch, Path('/receipt'), 0, image, run=run)
    assert not calls


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
