"""Derive isolated managed-service paths and unit names from one safe identifier."""
from __future__ import annotations

import re


def layout(name=None):
    if name is None:
        return {'code_dir':'/opt/sparkring/managed-mesh', 'config_dir':'/etc/sparkring/managed-mesh',
                'state_dir':'/run/sparkring-mesh', 'runtime_directory':'sparkring-mesh',
                'unit_dir':'/etc/systemd/system', 'mesh_unit':'sparkring-mesh.service',
                'model_unit':'sparkring-mesh-model.service',
                'liveness_unit':'sparkring-scheduler-liveness.service'}
    if not isinstance(name,str) or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?',name):
        raise ValueError('deployment-name must be 1..63 lowercase letters, digits or internal hyphens')
    prefix='sparkring-'+name
    selected={'code_dir':'/opt/sparkring/deployments/'+name, 'config_dir':'/etc/sparkring/deployments/'+name,
            'state_dir':'/run/'+prefix, 'runtime_directory':prefix,
            'unit_dir':'/etc/systemd/system', 'mesh_unit':prefix+'-mesh.service',
            'model_unit':prefix+'-model.service', 'liveness_unit':prefix+'-scheduler-liveness.service'}
    fields=('code_dir','config_dir','state_dir','runtime_directory','mesh_unit','model_unit','liveness_unit')
    defaults=layout()
    if {selected[key] for key in fields} & {defaults[key] for key in fields}:
        raise ValueError('deployment-name collides with the default installation')
    return selected


def validate_config_paths(config):
    name=config.get('deployment_name')
    selected=layout(name)
    if name is not None:
        expected={'site_path':selected['config_dir']+'/site.json',
                  'key_file':selected['config_dir']+'/health.key', 'state_dir':selected['state_dir']}
        if any(config.get(key)!=value for key,value in expected.items()):
            raise ValueError('Named deployment configuration paths differ from its derived layout')
    return selected


def command_roots(name, code_root, config_root):
    """Keep legacy root options separate from derived named-deployment paths."""
    if name is None:
        return code_root,config_root
    defaults=layout()
    if (code_root,config_root)!=(defaults['code_dir'],defaults['config_dir']):
        raise ValueError('deployment-name cannot be combined with custom code/config roots')
    selected=layout(name)
    return selected['code_dir'],selected['config_dir']
