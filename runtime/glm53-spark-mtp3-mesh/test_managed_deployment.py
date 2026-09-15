"""Named deployments cannot address the default installation or inject units."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parent))
import managed_cluster  # noqa: E402
import managed_install  # noqa: E402
import managed_units  # noqa: E402
from runtime.common import managed_deployment as deployment  # noqa: E402

NAME='r35-managed-20260913'


def install_plan(name=NAME):
    selected=deployment.layout(name)
    config={'schema':'sparkring-managed-mesh/v1','rank':0,'epoch':'a'*32,'health_port':9975,
            'site_path':selected['config_dir']+'/site.json','key_file':selected['config_dir']+'/health.key',
            'state_dir':selected['state_dir'],'container_id':'a'*64,'container_image':'sha256:'+'b'*64}
    plan={'schema':'sparkring-managed-install/v1','rank':0,'image':config['container_image'],
          'config':config,'units':managed_units.unit_text(selected['code_dir'],selected['config_dir'],'a'*64,
          host_liveness=True,deployment_name=name),
          **{key:selected[key] for key in ('code_dir','config_dir','unit_dir')}}
    if name is not None:
        config['deployment_name']=name
        plan.update(deployment_name=name,host_liveness=True)
    return plan


@pytest.mark.parametrize('name',['','../other','/etc/systemd/system','x.service','x@1','x y',
    'x\nExecStart=/bin/false','x%y','UPPER','-start','end-','a'*64,True,'mesh'])
def test_names_reject_path_and_unit_injection(name):
    with pytest.raises(ValueError,match='deployment-name'):
        deployment.layout(name)


@pytest.mark.parametrize('liveness,expected',[
    (False,'fce7e3823a6908870b14e882d4282b8c27463517e76cd408de3955e205739b6a'),
    (True,'ce804b4a342500919a4034e47fa76ba59060670da13e1a650af2c7d9a3ca629b')])
def test_default_unit_bytes_remain_compatible(liveness,expected):
    units=managed_units.unit_text('/opt/sparkring/managed-mesh','/etc/sparkring/managed-mesh','a'*64,
                                  host_liveness=liveness)
    assert hashlib.sha256(json.dumps(units,sort_keys=True).encode()).hexdigest()==expected


def test_named_layout_and_dependencies_are_isolated():
    plan=install_plan()
    selected=managed_install.validate_plan_layout(plan)
    assert selected['code_dir']=='/opt/sparkring/deployments/'+NAME
    assert selected['config_dir']=='/etc/sparkring/deployments/'+NAME
    assert selected['state_dir']=='/run/sparkring-'+NAME
    assert set(plan['units'])=={selected[key] for key in ('mesh_unit','model_unit','liveness_unit')}
    text='\n'.join(plan['units'].values())
    assert 'BindsTo='+selected['mesh_unit'] in text
    assert 'BindsTo='+selected['model_unit'] in text
    assert 'RuntimeDirectory='+selected['runtime_directory'] in text
    assert 'sparkring-mesh.service' not in text
    assert '/etc/sparkring/managed-mesh/' not in text


@pytest.mark.parametrize('change',['code_dir','config_dir','unit_dir','state_dir','site_path','key_file',
                                  'unit_name','unit_command','selection'])
def test_apply_rejects_tampered_layout_before_writes(monkeypatch,tmp_path,change):
    plan=install_plan()
    if change in ('code_dir','config_dir','unit_dir'):
        plan[change]='/tmp/elsewhere'
    elif change in ('state_dir','site_path','key_file'):
        plan['config'][change]='/tmp/elsewhere'
    elif change=='unit_name':
        plan['units']['../../elsewhere.service']='invalid'
    elif change=='unit_command':
        name=next(iter(plan['units']))
        plan['units'][name]=plan['units'][name].replace('ExecStart=','ExecStart=/bin/false #')
    else:
        plan['deployment_name']='another-deployment'
    plan.update(source_files=(),source_hashes={})
    monkeypatch.setattr(managed_install,'SOURCE_FILES',())
    monkeypatch.setattr(managed_install,'source_payloads',lambda:{})
    monkeypatch.setattr(managed_install.os,'geteuid',lambda:0,raising=False)
    monkeypatch.setattr(managed_install.managed_units.service,'read_key',lambda _:b'x'*32)
    monkeypatch.setattr(managed_install,'install_code',lambda *a:pytest.fail('Unexpected installation write'))
    with pytest.raises(ValueError,match='deployment|canonical'):
        managed_install.apply(plan,tmp_path,tmp_path/'key')


@pytest.mark.parametrize('action',['up','start-model','stop-model','down','recover','status'])
def test_lifecycle_targets_only_named_units_and_config(action):
    selected=deployment.layout(NAME)
    steps=managed_cluster.phases(action,selected['code_dir'],selected['config_dir'],NAME)
    argv=[value for _,command in steps for value in command]
    assert not any(value in ('sparkring-mesh.service','sparkring-mesh-model.service') for value in argv)
    assert not any(value.startswith('/opt/sparkring/managed-mesh/') for value in argv)
    assert any(NAME in value for value in argv)


def test_named_reset_does_not_reset_default_services(monkeypatch):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='LoadState=loaded\nActiveState=failed\n')
    monkeypatch.setattr(managed_units.service.subprocess,'run',run)
    managed_units.service.reset_units({'deployment_name':NAME})
    assert {argv[2] for argv in calls}=={deployment.layout(NAME)[key] for key in ('mesh_unit','model_unit')}


def test_named_config_rejects_default_paths_and_custom_root_flags():
    config=install_plan()['config']
    config['state_dir']='/run/sparkring-mesh'
    with pytest.raises(ValueError,match='derived layout'):
        deployment.validate_config_paths(config)
    with pytest.raises(ValueError,match='custom code/config roots'):
        deployment.command_roots(NAME,'/tmp/code','/tmp/config')


def test_default_and_named_valid_plans_pass_regeneration():
    for name in (None,NAME):
        assert managed_install.validate_plan_layout(install_plan(name))==deployment.layout(name)


def test_named_container_is_rechecked_without_starting_it():
    plan=install_plan()
    container={'Id':plan['config']['container_id'],'Image':plan['config']['container_image'],'State':{'Running':False},
        'Config':{'Entrypoint':['/opt/venv/bin/python'],'Cmd':['/opt/sparkring/bin/sparkring','serve'],
                  'Env':['SPARKRING_LIVENESS_ENABLED=1','SPARKRING_NODE_RANK=0']}}
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps([container]))
    managed_install.validate_named_container(plan,run=run)
    assert calls==[['docker','inspect',plan['config']['container_id']]]


@pytest.mark.parametrize('field',['code_dir','config_dir'])
def test_named_install_rejects_symlink_ancestor(tmp_path,field):
    actual=tmp_path/'elsewhere'
    actual.mkdir()
    link=tmp_path/'redirect'
    try:
        link.symlink_to(actual,target_is_directory=True)
    except OSError:
        pytest.skip('Creating symlinks requires operating-system permission')
    selected={key:str(tmp_path/key/'name') for key in ('code_dir','config_dir','state_dir','unit_dir')}
    selected[field]=str(link/'name')
    with pytest.raises(ValueError,match='nonsymlink directory ancestors'):
        managed_install.validate_named_ancestors(selected)
    assert not (actual/'name').exists()


@pytest.mark.parametrize('change',['running','identity','observer','rank'])
def test_apply_rechecks_named_container_and_observer_selection(change):
    plan=install_plan()
    container={'Id':plan['config']['container_id'],'Image':plan['config']['container_image'],'State':{'Running':False},
        'Config':{'Entrypoint':['/opt/venv/bin/python'],'Cmd':['/opt/sparkring/bin/sparkring','serve'],
                  'Env':['SPARKRING_LIVENESS_ENABLED=1','SPARKRING_NODE_RANK=0']}}
    if change=='running':
        container['State']['Running']=True
    elif change=='identity':
        container['Id']='c'*64
    elif change=='observer':
        container['Config']['Env']=['SPARKRING_NODE_RANK=0']
    else:
        container['Config']['Env']=['SPARKRING_LIVENESS_ENABLED=1','SPARKRING_NODE_RANK=1']
    with pytest.raises(ValueError,match='Named'):
        managed_install.validate_named_container(plan,run=lambda *a,**k:SimpleNamespace(stdout=json.dumps([container])))
