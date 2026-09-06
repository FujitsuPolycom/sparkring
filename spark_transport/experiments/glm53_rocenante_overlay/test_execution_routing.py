"""Verify exact eager/captured policy selection before transport admission."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_transport.experiments.glm53_rocenante_overlay.rocenante_vllm_overlay import (
    OverlayError, VirtualDiagonalAdapter, load_contract,
)

HERE=Path(__file__).parent

@pytest.mark.parametrize('mode,captured,expected',[
    ('both',False,True),('both',True,True),
    ('eager_only',False,True),('eager_only',True,False),
    ('graph_only',False,False),('graph_only',True,True),
    ('disabled',False,False),('disabled',True,False),
])
def test_execution_policy_before_admission(monkeypatch,mode,captured,expected):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:captured)
    adapter=VirtualDiagonalAdapter.__new__(VirtualDiagonalAdapter)
    adapter._closed=False
    adapter.device=torch.device('cuda',0)
    adapter.width=4096
    adapter.minimum_query_rows=1
    adapter.maximum_query_rows=32
    adapter.execution_mode=mode
    adapter.captured_sircl_rows=frozenset()
    tensor=SimpleNamespace(shape=(16,4096),dtype=torch.bfloat16,is_cuda=True,
        device=adapter.device,is_contiguous=lambda:True)
    assert adapter.eligible(tensor) is expected
    tensor.shape=(64,4096)
    assert not adapter.eligible(tensor)

def test_invalid_execution_policy_is_rejected(tmp_path):
    contract=json.loads((HERE/'overlay_contract.json').read_text())
    contract['runtime']['execution_mode']='rank_local_pressure'
    path=tmp_path/'config.json'
    path.write_text(json.dumps(contract))
    with pytest.raises(OverlayError,match='execution_mode'):
        load_contract(path)

def test_legacy_contract_keeps_both_modes():
    contract=load_contract(HERE/'overlay_contract.json')
    assert contract['runtime'].get('execution_mode','both')=='both'

def test_routing_status_uses_no_cuda_or_native_calls(monkeypatch):
    import weakref
    from spark_transport.experiments.glm53_rocenante_overlay import rocenante_vllm_overlay as module
    class Adapter:
        rank=2
        execution_mode='eager_only'
        captured_sircl_rows=frozenset()
        _candidate_calls=7
        _captured_nodes=2
        _fallback_calls=11
        def diagnostic_snapshot(self):
            raise AssertionError('Device or native diagnostics must not be called')
    adapter=Adapter()
    monkeypatch.setattr(module,'_adapters',weakref.WeakSet([adapter]))
    assert module.host_route_snapshot()==[{'rank':2,'execution_mode':'eager_only',
        'captured_sircl_query_rows':[],
        'eager_calls':5,'captured_nodes':2,'fallback_calls':11}]

@pytest.mark.parametrize('q',[8,16,24,32])
@pytest.mark.parametrize('captured',[False,True])
def test_selected_captured_rows_keep_eager_and_q8(monkeypatch,q,captured):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:captured)
    adapter=VirtualDiagonalAdapter.__new__(VirtualDiagonalAdapter)
    adapter._closed=False
    adapter.device=torch.device('cuda',0)
    adapter.width=4096
    adapter.minimum_query_rows=1
    adapter.maximum_query_rows=32
    adapter.execution_mode='both'
    adapter.captured_sircl_rows=frozenset({16,24,32})
    tensor=SimpleNamespace(shape=(q,4096),dtype=torch.bfloat16,is_cuda=True,
        device=adapter.device,is_contiguous=lambda:True)
    assert adapter.eligible(tensor) is (not captured or q==8)
