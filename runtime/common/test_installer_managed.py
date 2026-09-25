"""Managed GLM must stage into its actual backend layout before model downtime."""
from runtime.common import installer
from runtime.common.test_installer import site
from scripts import deploy_suite
from scripts.test_deploy_suite import inventory
from runtime.host import models


def test_managed_workspace_is_accepted_by_the_real_backend_planner():
    lock = installer.make_lock(installer.DEFAULTS['glm53', 4], site(4), '1' * 40, '2' * 64)
    workspace = installer.managed_workspace(lock['site']['name'])
    spec = deploy_suite.create_spec(inventory(), lock['site']['name'], workspace)
    assert spec['workspace'] == workspace
    assert all(row['cache'] == workspace + '/cache' for row in lock['site']['ranks'])
    assert workspace != lock['site']['workspace']


def test_preparation_includes_managed_staging_without_model_or_network_start():
    lock = installer.make_lock(installer.DEFAULTS['glm53', 4], site(4), '1' * 40, '2' * 64)
    plan = installer.operation_plan(lock, 'prepare')
    phases = [p['id'] for p in plan['phases']]
    assert phases[-1] == 'managed-prepare'
    assert phases.index('model') < phases.index('managed-prepare')
    assert not any(a['risk'] in ('starts-model', 'stops-model', 'driver-reload', 'hardware-test')
                   for p in plan['phases'] for a in p['actions'])


def test_native_no_cache_profiles_render_without_a_cache_connector():
    for count in (2, 4):
        name = installer.GLM_NO_CACHE[count]
        assert models.select(name, count) == name
        lock = installer.make_lock(name, site(count), '1' * 40, '2' * 64)
        assert lock['selection']['sparkcache'] is False
        assert lock['selection']['release'] == 'shared-2026.09.3'
        assert lock['backend'] == ('compose' if count == 2 else 'glm-managed')
        if count == 2:
            for spec in installer.specifications(lock):
                assert spec.environment['SPARKCACHE_ENABLED'] == '0'
                assert '--kv-transfer-config' not in spec.command
                assert '--enable-prefix-caching' in spec.command
                assert '--no-enable-prefix-caching' not in spec.command
