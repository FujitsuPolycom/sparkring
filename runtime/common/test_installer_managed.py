"""Managed GLM must stage into its actual backend layout before model downtime."""
from runtime.common import installer
from runtime.common.test_installer import site
from scripts import deploy_suite
from scripts.test_deploy_suite import inventory


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
