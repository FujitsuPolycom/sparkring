"""Behavior contracts for configuration, compatibility and command planning."""
import hashlib
import json

import pytest

from runtime.common import profiles
from runtime.common.environment import read_assignments
from runtime.common.launch import plan
from scripts.check_repository_layout import validate_imports, validate_preserved


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def repository(tmp_path):
    recipe = {'schema': 'sparkring-recipe/v1', 'recipe_id': 'example',
              'model': {'repository': 'example/NVFP4-Spark', 'revision': 'a'*40},
              'hardware': {'ranks': 2, 'topology': 'direct-pair-2'},
              'serving': {'max_model_len': 1024, 'max_num_seqs': 8}, 'runtime': {}}
    write(tmp_path/'profiles/example/recipe.json', recipe)
    p = {'schema': 'sparkring-deployment/v1', 'id': 'example', 'title': 'Example',
         'status': 'implemented', 'recommendation': 'alternative',
         'configuration': {'format': 'recipe', 'path': 'profiles/example/recipe.json'},
         'release': 'runtime/releases/example/release.json', 'guide': 'guide.md',
         'evidence_scope': 'CPU contract only', 'overrides': ['max_num_seqs'],
         'launcher': {'kind': 'python', 'path': 'launcher.py', 'actions': ['plan'], 'fixed_args': []}}
    write(tmp_path/'profiles/example/profile.json', p)
    write(tmp_path/'profiles/catalog.json', {'schema': 'sparkring-catalog/v1', 'profiles': [{'id': 'example', 'path': 'profiles/example/profile.json'}]})
    write(tmp_path/'runtime/releases/example/release.json', {'schema': 'sparkring-release-selection/v1', 'id': 'example', 'selection': 'source-build', 'inputs': [], 'image': {}})
    (tmp_path/'guide.md').write_text('# Guide\n')
    (tmp_path/'launcher.py').write_text('raise RuntimeError("must not execute during planning")\n')
    return tmp_path


@pytest.mark.parametrize('profile_id', list(profiles.catalog()))
def test_every_catalog_profile_resolves(profile_id):
    resolved = profiles.resolve(profile_id)
    assert resolved['model']['repository']
    assert resolved['serving']['tensor_parallel_size'] % resolved['serving']['decode_context_parallel_size'] == 0


def test_explicit_default_legacy_and_omission_are_equivalent(repository):
    ordinary = profiles.resolve('example', root=repository)
    explicit = profiles.resolve('example', {'max_num_seqs': 8}, root=repository)
    id = profiles.legacy_profile(repository/'profiles/example/recipe.json', repository)
    legacy = profiles.resolve(id, root=repository)
    assert ordinary['serving'] == explicit['serving'] == legacy['serving']
    assert ordinary['status'] == explicit['status']
    assert explicit['origins']['max_num_seqs'] == 'explicit'
    assert not explicit['modified_defaults']
    assert ordinary['origins']['decode_context_parallel_size'] == 'common'


def test_changed_serving_defaults_do_not_inherit_qualification(repository):
    result = profiles.resolve('example', {'max_num_seqs': 4}, root=repository)
    assert result['status'] == 'research-only'
    assert result['modified_defaults']


@pytest.mark.parametrize('override', [{'model': 'other'}, {'tensor_parallel_size': 4}, {'max_num_seqs': True}, {'max_num_seqs': 0}, {'max_num_seqs': '4'}])
def test_reject_invalid_or_identity_overrides(repository, override):
    with pytest.raises(ValueError):
        profiles.resolve('example', override, root=repository)


def test_environment_does_not_change_resolution(repository, monkeypatch):
    expected = profiles.resolve('example', root=repository)
    monkeypatch.setenv('MAX_MODEL_LEN', '999999')
    monkeypatch.setenv('VLLM_PLUGINS', 'unexpected')
    assert profiles.resolve('example', root=repository) == expected


@pytest.mark.parametrize('field,value', [('schema', 'sparkring-deployment/v99'), ('surprise', True), ('status', 'probably-works')])
def test_reject_unknown_schema_fields_and_status(repository, field, value):
    path = repository/'profiles/example/profile.json'
    data = profiles.read_json(path)
    data[field] = value
    write(path, data)
    with pytest.raises(ValueError):
        profiles.resolve('example', root=repository)


def test_duplicate_keys_rejected(tmp_path):
    path = tmp_path/'duplicate.json'
    path.write_text('{"value": 1, "value": 2}')
    with pytest.raises(ValueError, match='duplicate key'):
        profiles.read_json(path)


@pytest.mark.parametrize('path', ['../outside.json', '/outside.json', 'missing.json', 'a\\b'])
def test_bad_references_rejected(repository, path):
    with pytest.raises(ValueError):
        profiles.local_path(path, repository)


def test_release_hash_change_rejected(repository):
    path = repository/'runtime/releases/example/release.json'
    data = profiles.read_json(path)
    data['inputs'] = [{'path': 'launcher.py', 'sha256': 'a'*64}]
    write(path, data)
    with pytest.raises(ValueError, match='release input changed'):
        profiles.resolve('example', root=repository)


def valid_site():
    return {'schema': 'sparkring-site/v1', 'nodes': [{'rank': 0, 'address': 'rank0.example'}, {'rank': 1, 'address': 'rank1.example'}], 'model_dir': '/models/example', 'cache_dir': '/cache/example'}


def test_site_roundtrip_without_accessing_model_files(repository):
    site = valid_site()
    assert profiles.resolve('example', site=site, root=repository)['site'] == site


@pytest.mark.parametrize('mutation', ['rank', 'placeholder', 'same-path', 'credentials', 'count'])
def test_invalid_private_site_rejected(repository, mutation):
    site = valid_site()
    if mutation == 'rank':
        site['nodes'][1]['rank'] = 0
    elif mutation == 'placeholder':
        site['nodes'][0]['address'] = '<replace>'
    elif mutation == 'same-path':
        site['cache_dir'] = site['model_dir']
    elif mutation == 'count':
        site['nodes'].pop()
    else:
        site['password'] = 'placeholder'
    with pytest.raises(ValueError):
        profiles.resolve('example', site=site, root=repository)


def test_plan_does_not_execute_launcher(repository):
    result = plan('example', ['plan', '--name', 'value with spaces'], repository)
    assert result['command'][-1] == 'value with spaces'
    assert result['effect'] == 'adapter-check-or-plan'
    with pytest.raises(ValueError, match='explicit action'):
        plan('example', ['start'], repository)


def test_catalog_pins_r33_receipt_and_cache_selection():
    result = plan('glm53-flash-spark-tp2-dcp1-sparkcache', ['plan'])
    assert '--r33-sparkcache' in result['command']
    assert result['command'][-2] == '--runtime-receipt'
    with pytest.raises(ValueError, match='catalog owns'):
        plan('glm53-flash-spark-tp2-dcp1-sparkcache', ['plan', '--runtime-receipt=other.json'])


def test_site_assignments_are_not_shell_sourced(tmp_path):
    path = tmp_path/'site.env'
    path.write_text('A=example\nA=other\n')
    with pytest.raises(ValueError):
        read_assignments(path, {'A'})
    path.write_text('A=<replace>\n')
    with pytest.raises(ValueError):
        read_assignments(path, {'A'})


def test_maintained_import_boundary(tmp_path):
    path = tmp_path/'runtime/common/example.py'
    path.parent.mkdir(parents=True)
    path.write_text('from spark_transport.experiments.prototype import run\n')
    with pytest.raises(ValueError, match='maintained owner'):
        validate_imports(tmp_path)


def test_preserved_input_tamper_rejected(tmp_path):
    path = tmp_path/'release.json'
    path.write_bytes(b'original')
    write(tmp_path/'runtime/releases/preserved-inputs.json', {'files': {'release.json': hashlib.sha256(b'original').hexdigest()}})
    assert validate_preserved(tmp_path) == 1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='preserved release input changed'):
        validate_preserved(tmp_path)


def test_r33_pair_and_source_profile_are_not_silently_interchanged():
    result = profiles.resolve('glm53-flash-spark-tp2-dcp1-sparkcache')
    source = profiles.read_json(profiles.ROOT/'runtime/profiles/glm53-flash-spark-tp2/profile.json')
    args = source['vllm_args']
    assert int(args[args.index('--max-model-len')+1]) == 262144
    assert result['serving']['max_model_len'] == 1048576
    assert result['serving']['sparkcache']
    assert not source['sparkcache']['enabled']


def test_environment_examples_preserve_baseline_defaults():
    from runtime.common.environment import render_environment
    rows = profiles.read_json(profiles.ROOT/'profiles/environment-exports.json')['exports']
    for row in rows:
        rendered = render_environment(row['profile'], template_only=True).encode('utf-8')
        for key, change in row.get('default_changes', {}).items():
            current = f"{key.upper()}={change['to']}".encode()
            original = f"{key.upper()}={change['from']}".encode()
            assert rendered.count(current) == 1
            rendered = rendered.replace(current, original)
        assert hashlib.sha256(rendered).hexdigest() == row['migration_baseline_sha256']


def rank_values(profile_id):
    import re
    from runtime.common.environment import render_environment
    text = render_environment(profile_id, template_only=True)
    values = {}
    for line in text.splitlines():
        if re.match(r'^[A-Z][A-Z0-9_]*=.*<', line):
            key = line.split('=', 1)[0]
            values[key] = '0' if key in ('NODE_RANK', 'RANK') else '/example/'+key.lower() if 'PATH' in key or key == 'PATCH_DIR' else 'example'
            if key == 'IMAGE_ID':
                values[key] = 'sha256:'+'a'*64
    return values


def test_rendered_environment_applies_shared_override():
    from runtime.common.environment import render_environment
    id = 'deepseek-v41-flash-cycle'
    values = rank_values(id)
    text = render_environment(id, values, {'max_num_seqs': 4})
    assert '\nMAX_NUM_SEQS=4\n' in text
    assert '\nMAX_MODEL_LEN=1048576\n' in text
    assert '\nNODE_RANK=0\n' in text


def test_environment_cannot_smuggle_serving_override_or_shell(repository):
    from runtime.common.environment import render_environment
    id = 'deepseek-v41-flash-cycle'
    values = rank_values(id)
    values['MAX_NUM_SEQS'] = '99'
    with pytest.raises(ValueError, match='only template placeholders'):
        render_environment(id, values)
    values.pop('MAX_NUM_SEQS')
    values['MASTER_ADDR'] = '$(unexpected)'
    with pytest.raises(ValueError, match='literal site value'):
        render_environment(id, values)


def test_maintained_switched_launcher_preserves_frozen_plan(tmp_path):
    import importlib.util
    from runtime.common import switched
    source = profiles.ROOT/'runtime/profiles/glm53-flash-spark-tp4-switched/launch.py'
    spec = importlib.util.spec_from_file_location('frozen_switched_contract', source)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    model, cache = tmp_path/'model', tmp_path/'cache'
    model.mkdir()
    cache.mkdir()
    (model/'config.json').write_text('{}')
    site = tmp_path/'rank.env'
    site.write_text('VLLM_HOST_IP=rank.example\nNCCL_SOCKET_IFNAME=control0\nGLOO_SOCKET_IFNAME=control0\nNCCL_IB_HCA==mlx5_7:1\nNCCL_IB_GID_INDEX=3\n')
    args = (1, 'master.example', model, cache, site, 'ghcr.io/fujitsupolycom/sparkring@sha256:'+'a'*64)
    assert switched.render(*args) == frozen.render(*args)


def test_managed_source_snapshot_imports_without_checkout(tmp_path):
    import ast
    import subprocess
    import sys
    installer = profiles.ROOT/'runtime/glm53-spark-mtp3-mesh/managed_install.py'
    tree = ast.parse(installer.read_text(encoding='utf-8-sig'))
    files = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'SOURCE_FILES' for target in node.targets))
    for relative in files:
        target = tmp_path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((profiles.ROOT/relative).read_bytes())
    script = tmp_path/'check_snapshot.py'
    script.write_text('import runpy\nfrom pathlib import Path\nrunpy.run_path(str(Path(__file__).parent / "runtime/glm53-spark-mtp3-mesh/profile.py"), run_name="snapshot_import_check")\n')
    result = subprocess.run([sys.executable, '-I', str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_retained_dflash_recipe_uses_its_declared_preference():
    for id in ('glm53-flash-nvfp4-dflash2-bf16-tp4', 'sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4'):
        assert profiles.resolve(id)['serving']['decode_context_parallel_size'] == 4


def test_composition_references_and_legacy_exports_roundtrip():
    rows = profiles.read_json(profiles.ROOT/'profiles/compatibility.json')['mirrors']
    for row in rows:
        if row['kind'] != 'recipe':
            continue
        source = profiles.read_json(profiles.local_path(row['source']))
        if source.get('base_recipe'):
            assert profiles.local_path(source['base_recipe']).is_file()
            legacy = profiles.read_json(profiles.local_path(row['destination']))
            assert legacy['base_recipe'].startswith('../')
            id = profiles.legacy_profile(profiles.local_path(row['destination']))
            assert profiles.resolve(id)['model'] == source['model']


def test_composition_does_not_invent_a_missing_model_revision():
    result = profiles.resolve('sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1')
    assert 'checkpoint_sha256' in result['model']
    assert 'revision' not in result['model']
