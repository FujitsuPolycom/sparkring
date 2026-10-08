"""Passive transport evidence: stored metadata only, without vLLM or a GPU."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from sparkring_runtime_status import transport


class B12xRoceAllReduce:
    def __init__(self, disabled=False, closed=False):
        self.disabled = disabled
        self._runtime = NS(max_size=2097152, max_gather_bytes=16777216,
                           hca_names=('hca0', 'hca2'), gid_index=3, _closed=closed)
        self._plan = NS()
        self._announced = True
        self._announced_gather = False

    def stats(self):
        raise AssertionError('stats called')

    def check_health(self):
        raise AssertionError('health called')


def group(adapter=None):
    return NS(world_size=2, rank_in_group=0,
              device_communicator=NS(b12x_ar_comm=adapter,
                  pynccl_comm=NS(available=True, disabled=False, nccl_version=23203,
                                 nccl=NS(lib=NS(_name='/opt/nccl/libnccl.so.2')))))


def collect(tmp_path, **groups):
    return transport.snapshot(modules={'vllm.distributed.parallel_state': NS(**groups)},
                              environ={}, sysfs_root=tmp_path)


def put(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def nic(root, name, domain, netdev):
    base = f'class/infiniband/{name}'
    put(root, f'{base}/device/uevent', f'PCI_SLOT_NAME={domain}:01:00.0\n')
    put(root, f'{base}/device/current_link_speed', '16.0 GT/s PCIe\n')
    put(root, f'{base}/device/current_link_width', '4\n')
    (root / base / 'device/net' / netdev).mkdir(parents=True)
    put(root, f'class/net/{netdev}/speed', '200000\n')
    put(root, f'class/net/{netdev}/operstate', 'up\n')
    put(root, f'class/net/{netdev}/address', '02:00:00:00:00:01\n')
    put(root, f'{base}/ports/1/counters/port_xmit_data', '123\n')
    put(root, f'{base}/ports/1/counters/port_rcv_data', '234\n')
    put(root, f'{base}/ports/1/link_layer', 'Ethernet\n')


def test_resident_state_not_environment_controls_resolution(tmp_path):
    modules = {'vllm.distributed.parallel_state': NS(_TP=group(B12xRoceAllReduce()))}
    result = transport.snapshot(modules=modules,
        environ={'VLLM_ENABLE_ROCE_ALLREDUCE': '0'}, sysfs_root=tmp_path)
    assert result['effective']['tp_rocenante_enabled']['value'] is True
    assert result['effective']['tp_roce_allreduce_max_bytes']['value'] == 2097152
    assert result['effective']['tp_roce_allgather_max_bytes']['value'] == 16777216
    assert result['effective']['tp_nccl_version']['value'] == 23203
    assert result['effective']['tp_nccl_library_path']['value'] == '/opt/nccl/libnccl.so.2'
    assert result['effective']['tp_roce_hcas']['value'] == 'hca0,hca2'
    assert result['observed']['transport']['state'] == 'not_observed'
    first = result['observed']['rocenante_first_use']
    assert first['phase'] == 'unspecified_includes_warmup'
    assert first['current_request_execution'] == 'not_observed'


@pytest.mark.parametrize('disabled,closed', [(True, False), (False, True)])
def test_disabled_or_closed_never_appears_enabled(tmp_path, disabled, closed):
    modules = {'vllm.distributed.parallel_state': NS(_TP=group(B12xRoceAllReduce(disabled, closed)))}
    result = transport.snapshot(modules=modules,
        environ={'VLLM_ENABLE_ROCE_ALLREDUCE': '1'}, sysfs_root=tmp_path)
    assert result['effective']['tp_rocenante_enabled']['value'] is False


def test_tp_and_ep_are_independent_not_rank_disagreement(tmp_path):
    result = collect(tmp_path, _TP=group(B12xRoceAllReduce()), _EP=group(None))
    assert result['groups']['TP']['rocenante']['enabled']['value'] is True
    assert result['groups']['EP']['rocenante']['enabled']['value'] is False
    assert result['effective']['tp_rocenante_enabled']['value'] is True
    for field in ('algorithm', 'protocol', 'channels'):
        assert result['groups']['TP']['nccl'][field]['state'] == 'unknown'


def test_missing_module_fields_and_other_backend_remain_explicit(tmp_path):
    result = transport.snapshot(modules={}, environ={}, sysfs_root=tmp_path)
    assert result['state'] == 'unknown'
    assert result['effective']['tp_rocenante_enabled']['state'] == 'unknown'
    assert result['nics'] == []
    pcie = NS(disabled=False, _runtime=NS(max_size=128, hca_names=('hca0',)))
    result = collect(tmp_path, _TP=group(pcie))
    assert result['groups']['TP']['b12x_backend']['value'] == 'SimpleNamespace'
    assert result['effective']['tp_rocenante_enabled']['value'] is False
    assert result['nics'] == []


def test_missing_runtime_cannot_be_resolved_from_requested_flag(tmp_path):
    adapter = B12xRoceAllReduce()
    del adapter._runtime
    result = collect(tmp_path, _TP=group(adapter))
    assert result['effective']['tp_rocenante_enabled']['state'] == 'unknown'


def test_sysfs_selected_domains_are_not_traffic_or_bandwidth_proof(tmp_path):
    nic(tmp_path, 'hca0', '0000', 'net0')
    nic(tmp_path, 'hca2', '0002', 'net2')
    result = collect(tmp_path, _TP=group(B12xRoceAllReduce()))
    assert result['effective']['tp_roce_pci_domains']['value'] == '0000,0002'
    entry = next(n for n in result['nics'] if n['hca'] == 'hca0')
    assert entry['pci']['current_link_width']['value'] == 4
    assert entry['netdevs'][0]['speed_mbps']['value'] == 200000
    assert entry['ports'][0]['counters']['port_xmit_data']['value'] == 123
    assert entry['ports'][0]['counters']['port_xmit_data']['unit'] == '4-byte words'
    assert entry['ports'][0]['counter_scope'] == 'host_rdma_port_not_process'
    assert result['observed']['transport']['state'] == 'not_observed'
    assert 'bandwidth' not in result['effective']


def test_partial_domain_inventory_and_unknown_link_speed(tmp_path):
    nic(tmp_path, 'hca0', '0000', 'net0')
    put(tmp_path, 'class/net/net0/speed', '-1\n')
    result = collect(tmp_path, _TP=group(B12xRoceAllReduce()))
    assert result['effective']['tp_roce_pci_domains']['state'] == 'unknown'
    assert result['nics'][0]['netdevs'][0]['speed_mbps']['state'] == 'unknown'


class Trap:
    @property
    def disabled(self):
        raise AssertionError('descriptor called')

    def __getattr__(self, name):
        raise AssertionError('dynamic lookup called')

    def __repr__(self):
        raise AssertionError('repr called')

    def item(self):
        raise AssertionError('tensor read')


def test_descriptors_methods_and_tensor_values_are_never_touched(tmp_path):
    adapter = B12xRoceAllReduce()
    adapter._runtime.max_size = Trap()
    adapter._runtime._proxy = Trap()
    adapter._runtime._ctrl_np = Trap()
    result = collect(tmp_path, _TP=group(adapter), _PP=Trap())
    assert result['effective']['tp_roce_allreduce_max_bytes']['state'] == 'unknown'
    assert result['groups']['PP']['rocenante']['enabled']['state'] == 'unknown'
    json.dumps(result)


@pytest.mark.parametrize('names', [('..',), ('bad/path',), ('x' * 65,), tuple(f'h{i}' for i in range(9))])
def test_invalid_or_oversized_hca_inventory_is_not_followed(tmp_path, names):
    adapter = B12xRoceAllReduce()
    adapter._runtime.hca_names = names
    result = collect(tmp_path, _TP=group(adapter))
    assert result['nics'] == []
    assert result['effective']['tp_roce_hcas']['state'] == 'unknown'


def test_no_backend_imports_or_runtime_methods(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__
    def reject_backend(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'b12x'):
            raise AssertionError('backend import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', reject_backend)
    adapter = B12xRoceAllReduce()
    adapter._runtime.stats = adapter.stats
    adapter._runtime.check_health = adapter.check_health
    result = collect(tmp_path, _TP=group(adapter))
    assert result['observed']['fabric_latency']['reason'] == 'latency_not_measured'
    assert result['observed']['fabric_bandwidth']['reason'] == 'payload_bandwidth_not_measured'


def test_roce_adapter_descriptor_and_nccl_handle_descriptor_are_not_evaluated(tmp_path):
    def explode(_):
        raise AssertionError('descriptor')
    adapter = type('B12xRoceAllReduce', (), {'disabled': property(explode)})()
    adapter._runtime = NS(_closed=False)
    tp = group(adapter)
    tp.device_communicator.pynccl_comm.nccl.lib = type('Library', (), {'_name': property(explode)})()
    result = collect(tmp_path, _TP=tp)
    assert result['effective']['tp_rocenante_enabled']['state'] == 'unknown'
    assert result['effective']['tp_nccl_library_path']['state'] == 'unknown'


def test_unset_native_policy_is_not_inferred_from_environment(tmp_path):
    result = transport.snapshot(modules={'vllm.distributed.parallel_state': NS(_TP=group(None))},
        environ={'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'LL', 'NCCL_MIN_NCHANNELS': '8'},
        sysfs_root=tmp_path)
    for field in ('algorithm', 'protocol', 'channels'):
        assert result['groups']['TP']['nccl'][field]['state'] == 'unknown'


def test_sysfs_reads_and_port_count_are_bounded(tmp_path):
    nic(tmp_path, 'hca0', '0000', 'net0')
    put(tmp_path, 'class/net/net0/operstate', 'x' * (transport.MAX_TEXT + 1))
    for port in range(2, 10):
        put(tmp_path, f'class/infiniband/hca0/ports/{port}/link_layer', 'Ethernet')
    result = collect(tmp_path, _TP=group(B12xRoceAllReduce()))
    entry = result['nics'][0]
    assert entry['ports_truncated'] is True
    assert len(entry['ports']) == transport.MAX_PORTS
    assert entry['netdevs'][0]['operstate']['state'] == 'unknown'


def test_sysfs_invalid_unsigned_counters_remain_unknown(tmp_path):
    nic(tmp_path, 'hca0', '0000', 'net0')
    for name, value in (('port_xmit_data', '-1'), ('port_rcv_data', str(2**64))):
        put(tmp_path, f'class/infiniband/hca0/ports/1/counters/{name}', value)
    result = collect(tmp_path, _TP=group(B12xRoceAllReduce()))
    counters = result['nics'][0]['ports'][0]['counters']
    assert counters['port_xmit_data']['state'] == counters['port_rcv_data']['state'] == 'unknown'


def test_one_unreadable_nic_does_not_hide_other_nics(tmp_path, monkeypatch):
    nic(tmp_path, 'hca2', '0002', 'net2')
    original = Path.open
    def guarded(file, *args, **kwargs):
        if 'hca0' in file.parts:
            raise PermissionError('denied')
        return original(file, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    result = collect(tmp_path, _TP=group(B12xRoceAllReduce()))
    by_name = {entry['hca']: entry for entry in result['nics']}
    assert by_name['hca0']['pci']['domain']['state'] == 'unknown'
    assert by_name['hca2']['pci']['domain']['value'] == '0002'
    assert result['effective']['tp_roce_pci_domains']['state'] == 'unknown'


def test_sircl_facts_come_from_this_workers_tensor_parallel_receipt(tmp_path):
    receipts = tmp_path / 'receipts'
    receipts.mkdir()
    record = {'schema': 'sircl-vllm-receipt/v1', 'group': 'tp', 'global_rank': 1, 'session': 's-0123abcd',
              'state': 'ready', 'nccl': 'none', 'pynccl': 'skipped', 'fabric': 'cycle:0-1-2-3', 'relays': 1,
              'lanes': 2}
    (receipts / 'rank1-tp.json').write_text(json.dumps(record))
    (receipts / 'rank0-tp.json').write_text(json.dumps(dict(record, global_rank=0, session='other')))
    tp = NS(**{**vars(group()), 'rank': 1})
    modules = {'vllm.distributed.parallel_state': NS(_TP=tp)}
    environ = {'SIRCL_MODE': 'custom', 'SIRCL_RECEIPT_DIR': str(receipts)}
    effective = transport.snapshot(modules=modules, environ=environ, sysfs_root=tmp_path)['effective']
    assert effective['tp_sircl_session']['value'] == 's-0123abcd' and effective['tp_sircl_nccl']['value'] == 'none'
    assert effective['tp_sircl_pynccl']['value'] == 'skipped' and effective['tp_sircl_relays']['value'] == 1
    assert effective['tp_sircl_fabric']['value'] == 'cycle:0-1-2-3'
    assert effective['tp_sircl_receipt_age_s']['state'] == 'known' and effective['tp_sircl_receipt_age_s']['unit'] == 's'
    # Without SIRCL the rows say so instead of guessing.
    plain = transport.snapshot(modules=modules, environ={}, sysfs_root=tmp_path)['effective']
    assert plain['tp_sircl_session'] == {'state': 'unknown', 'source': 'sircl_receipt', 'reason': 'no_sircl_receipt'}
