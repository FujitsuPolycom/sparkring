"""Offline receipt and rendering regression checks for the pinned R35 composition."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.common import r35, tp2


def receipt():
    """Synthetic file inventory; tests must not describe this as an image receipt."""
    parent = r35.read(r35.ROOT/'runtime/sparkring/jovian-r33/public-image-receipt.json')
    installed = dict(schema='sparkring-r35-installed/v1', components=r35.read(r35.RECIPE/'source-lock.json')['components'],
                     files={}, versions={'vllm':'0.26.1rc0+sparkring.r35.3d4284de'}, removed_authored_files=[], parent_source_lock_sha256=parent['source_lock_sha256'])
    contract = r35.profile_contract(installed)
    files = installed['files']
    for path in ('/opt/local-inference/nccl/lib/libnccl.so.2.31.2', '/opt/sparkring/sircl/libspark_transport_capi.so',
                 '/opt/sparkring/sircl/python/sparkring-overlay-manifest.json', '/opt/sparkring/runtime/glm53-spark-mtp3-mesh/pins.json'):
        files[path] = parent['verification']['checked_files'][path]
    for target, source in ((r35.CAPABILITY,'contracts/tp2-sparkcache-capabilities.json'),
                           (r35.LEASE,'contracts/vllm-connector-jobs.json'),
                           ('/opt/sparkring/bin/sparkring','entrypoint.py'),
                           ('/opt/sparkring/bin/verify-r35.py','verify_image.py')):
        files[target] = r35.digest((r35.RECIPE/source).read_bytes())
    files[r35.CONTRACT] = r35.digest(r35.json_bytes(contract))
    files['/opt/sparkring/image/artifact-lock.json'] = contract['image']['artifact_lock_sha256']
    for key in ('placement','snapshot'):
        files[contract['sparkcache_native'][key+'_path']] = contract['sparkcache_native'][key+'_sha256']
    for name, sha in r35.read(r35.RECIPE/'contracts/tp2-sparkcache-capabilities.json')['evidence_sha256'].items():
        files['/opt/sparkring/profile-contract/evidence/'+name+'.evidence.json'] = sha
    return dict(schema=r35.SCHEMA, image_id='sha256:'+'e'*64, image_reference='sha256:'+'e'*64, platform='linux/arm64',
                installed=installed, verification=dict(schema='sparkring-r35-verification/v1',files_verified=len(files),
                    source_components=deepcopy(installed['components']),source_lock_sha256=r35.digest(r35.json_bytes(installed)),serving_qualified=False))


def test_receipt_does_not_claim_serving_qualification():
    document = receipt()
    r35.validate_profile_capabilities(document, 'tp2-dcp1-sparkcache')
    assert r35.validate_receipt(document)['verification']['serving_qualified'] is False


@pytest.mark.parametrize('change', ['image','platform','source','contract','lease','capability','payload','qualification','marker_claim','verification_marker_claim'])
def test_receipt_rejects_drift(change):
    document = receipt()
    if change == 'image': document['image_reference'] = 'latest'
    elif change == 'platform': document['platform'] = 'linux/amd64'
    elif change == 'source': document['installed']['components']['vllm']['tree'] = '0'*40
    elif change in ('contract','lease','capability'):
        document['installed']['files'][{'contract':r35.CONTRACT,'lease':r35.LEASE,'capability':r35.CAPABILITY}[change]] = '0'*64
        document['verification']['source_lock_sha256'] = r35.digest(r35.json_bytes(document['installed']))
    elif change == 'payload': document['verification']['source_lock_sha256'] = '0'*64
    elif change=='marker_claim': document['marker_binary_sha256']='a'*64
    elif change=='verification_marker_claim': document['verification']['marker_binary_sha256']='a'*64
    else: document['verification']['serving_qualified'] = True
    with pytest.raises(ValueError): r35.validate_receipt(document)


@pytest.mark.parametrize('rank', [0,1])
@pytest.mark.parametrize('cache', [False,True])
def test_tp2_r35_renderer_preserves_glm_kda_and_selects_verified_entrypoint(tmp_path, rank, cache):
    document = receipt()
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    cache_dir=tmp_path/'cache';cache_dir.mkdir()
    env=tmp_path/'rank.env';env.write_text('VLLM_HOST_IP=192.0.2.10\nNCCL_SOCKET_IFNAME=eth0\nGLOO_SOCKET_IFNAME=eth0\n')
    plan=tp2.render(rank,'192.0.2.10',model,cache_dir,env,document['image_id'],document,r33_sparkcache=cache)
    assert plan['runtime_kind']=='r35-candidate'
    assert plan['container_args'][:2]==['/opt/sparkring/bin/sparkring','serve']
    assert '--gdn-decode-kernel' not in plan['container_args']
    assert plan['container_args'][plan['container_args'].index('--kda-prefill-backend')+1]=='b12x'
    assert ('--health-cmd' in plan['command']) == (rank==0)
    assert plan['entrypoint']=='/opt/venv/bin/python'
    # Docker retains the overridden Python entrypoint and the script in Cmd.
    # Verify monitor selection when the image supplies liveness enablement.
    import importlib.util
    spec=importlib.util.spec_from_file_location('tp2_monitor',r35.ROOT/'runtime/glm53-spark-mtp3-mesh/managed_liveness.py')
    monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    container={'Config':{'Entrypoint':[plan['entrypoint']],'Cmd':plan['container_args'],
        'Env':['SPARKRING_LIVENESS_ENABLED=1',*[f'{k}={v}' for k,v in plan['environment'].items()]]}}
    assert monitor.requires_host_monitor(container,rank)==(rank==0)
    if cache:
        assert plan['environment']['SPARKCACHE_SOURCE_LEASE_CONTRACT']==r35.LEASE
        assert plan['activation_blockers']==[]
    tp2.validate_runtime_receipt(document,plan)


def test_actual_image_verification_is_required_and_must_match():
    document=receipt();calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        if argv[1:3]==['image','inspect']:
            return SimpleNamespace(stdout=json.dumps([dict(Id=document['image_id'],Os='linux',Architecture='arm64')]))
        return SimpleNamespace(stdout=json.dumps(dict(document['verification'],source_lock_sha256='0'*64)))
    with pytest.raises(ValueError,match='payload verification'):
        r35.verify_local_image(document,run=run)
    assert calls[1]==['docker','run','--rm','--network','none','--pull','never',document['image_id'],'verify']
