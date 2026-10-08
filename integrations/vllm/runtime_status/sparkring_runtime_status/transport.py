"""Passive communicator and bounded sysfs evidence; never execute a transport.

Resident communicator state establishes construction choices, not the route
taken by a request. Sysfs counters describe host RDMA ports, not this process.
Negotiated link rates are not measured bandwidth. No backend is imported here.
"""
from __future__ import annotations

from itertools import islice
import json
import os
from pathlib import Path
import re
import time

from .collector import MISSING, fact, path, stored

ROLES = ('TP', 'PP', 'DP', 'EP', 'DCP', 'PCP')
MAX_HCAS = 8
MAX_PORTS = 4
MAX_NETDEVS = 4
MAX_TEXT = 1024
NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.:-]{0,63}')
BDF = re.compile(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]')
COUNTERS = ('port_xmit_data', 'port_rcv_data', 'port_rcv_errors',
            'port_xmit_discards', 'symbol_error', 'link_downed',
            'port_rcv_constraint_errors', 'port_xmit_constraint_errors')


def _fact(value=MISSING, *, source, reason='field_not_exposed', phase=None):
    result = fact(value, source=source, phase=phase)
    if result['state'] == 'unknown':
        result['reason'] = reason
    return result


def _boolean(value):
    return value if type(value) is bool else MISSING


def _integer(value):
    return value if type(value) is int and 0 <= value <= 2**64 - 1 else MISSING


def _names(value):
    if type(value) not in (tuple, list) or not 0 < len(value) <= MAX_HCAS:
        return None
    if any(type(item) is not str or not NAME.fullmatch(item) or item in ('.', '..')
           for item in value):
        return None
    if len(set(value)) != len(value):
        return None
    return tuple(value)


def _read(file):
    try:
        with file.open('r', encoding='ascii') as stream:
            text = stream.read(MAX_TEXT + 1)
        return text.strip() if len(text) <= MAX_TEXT else None
    except (OSError, UnicodeError):
        return None


def _number(file, *, positive=False):
    text = _read(file)
    if text is None or not re.fullmatch(r'[0-9]{1,20}', text):
        return MISSING
    value = int(text)
    return value if int(positive) <= value <= 2**64 - 1 else MISSING


def _children(directory, limit):
    # Bound enumeration as well as reads; never scan the entire host tree.
    try:
        with os.scandir(directory) as iterator:
            entries = [Path(entry.path) for entry in islice(iterator, limit + 1)]
    except OSError:
        return [], False
    return sorted(entries[:limit], key=lambda entry: entry.name), len(entries) > limit


def _sysfact(value=MISSING, *, unit=None, source='sysfs', phase=None):
    result = _fact(value, source=source, reason='sysfs_unavailable_or_invalid', phase=phase)
    if unit:
        result['unit'] = unit
    return result


def _nic(root, name):
    base = root / 'class/infiniband' / name
    device = base / 'device'
    raw = _read(device / 'uevent')
    matches = re.findall(r'^PCI_SLOT_NAME=(.+)$', raw or '', re.M)
    bdf = matches[0] if len(matches) == 1 and BDF.fullmatch(matches[0]) else None
    if bdf is None:
        try:
            resolved = device.resolve(strict=True).name
            bdf = resolved if BDF.fullmatch(resolved) else None
        except (OSError, RuntimeError):
            pass
    domain = bdf.split(':')[0].lower() if bdf else MISSING
    pci = {'bdf': _sysfact(bdf if bdf else MISSING), 'domain': _sysfact(domain)}
    # Functions with the same device key can share one physical PCIe uplink.
    # Do not sum their individually reported link capacities.
    pci['device_key'] = _sysfact(bdf.rsplit('.', 1)[0] if bdf else MISSING)
    for prefix in ('current', 'max'):
        speed = _read(device / f'{prefix}_link_speed')
        match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?) GT/s(?: PCIe)?', speed or '')
        pci[f'{prefix}_link_speed'] = _sysfact(match[1] if match else MISSING, unit='GT/s')
        pci[f'{prefix}_link_width'] = _sysfact(_number(device / f'{prefix}_link_width', positive=True), unit='lanes')
    netdevs, truncated_netdevs = _children(device / 'net', MAX_NETDEVS)
    nets = []
    for netdev in netdevs:
        if not NAME.fullmatch(netdev.name) or netdev.name in ('.', '..'):
            continue
        net = root / 'class/net' / netdev.name
        mac = _read(net / 'address')
        nets.append({'name': netdev.name,
            'speed_mbps': _sysfact(_number(net / 'speed', positive=True), unit='Mb/s'),
            'speed_scope': 'negotiated_link_rate_not_payload_bandwidth',
            'operstate': _sysfact(_read(net / 'operstate') or MISSING),
            'mtu': _sysfact(_number(net / 'mtu', positive=True), unit='bytes'),
            'mac': _sysfact(mac if mac and re.fullmatch(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', mac) else MISSING),
            'ip_addresses': _fact(source='sysfs', reason='not_exposed_by_this_collector')})
    ports, truncated_ports = _children(base / 'ports', MAX_PORTS)
    port_rows = []
    for port in ports:
        if not re.fullmatch(r'[0-9]{1,2}', port.name):
            continue
        counters = {key: _sysfact(_number(port / 'counters' / key),
                    unit='4-byte words' if key.endswith('_data') else 'count',
                    source='sysfs_rdma_port_counter', phase='host_cumulative') for key in COUNTERS}
        state = _read(port / 'state')
        match = re.fullmatch(r'[0-9]+:\s*([A-Z_]+)', state or '')
        port_rows.append({'port': int(port.name),
            'state': _sysfact(match[1] if match else MISSING),
            'link_layer': _sysfact(_read(port / 'link_layer') or MISSING),
            'counter_scope': 'host_rdma_port_not_process', 'counters': counters})
    return {'hca': name, 'source': 'bounded_sysfs_inventory', 'pci': pci,
            'netdevs': nets, 'ports': port_rows, 'netdevs_truncated': truncated_netdevs,
            'ports_truncated': truncated_ports}


def _group(group):
    source = 'resident_distributed_communicator'
    communicator = stored(group, 'device_communicator')
    adapter = stored(communicator, 'b12x_ar_comm')
    is_roce = adapter is not MISSING and adapter is not None and type(adapter).__name__ == 'B12xRoceAllReduce'
    runtime = stored(adapter, '_runtime') if is_roce else MISSING
    disabled = _boolean(stored(adapter, 'disabled')) if is_roce else MISSING
    closed = _boolean(stored(runtime, '_closed'))
    names = _names(stored(runtime, 'hca_names'))
    if adapter is None or (adapter is not MISSING and not is_roce):
        enabled = False
    elif disabled is True or closed is True:
        enabled = False
    elif disabled is False and closed is False:
        enabled = True
    else:
        enabled = MISSING
    roce = {'enabled': _fact(enabled, source=source, reason='runtime_state_not_exposed'),
            'disabled': _fact(disabled, source=source),
            'runtime_closed': _fact(closed, source='resident_rocenante_runtime'),
            'allreduce_max_bytes': _fact(_integer(stored(runtime, 'max_size')), source='resident_rocenante_runtime'),
            'allgather_max_bytes': _fact(_integer(stored(runtime, 'max_gather_bytes')), source='resident_rocenante_runtime'),
            'hcas': _fact(','.join(names) if names else MISSING, source='resident_rocenante_runtime'),
            'gid_index': _fact(_integer(stored(runtime, 'gid_index')), source='resident_rocenante_runtime')}
    for field in ('_peer_hca_map', '_opposite_paths'):
        value = stored(runtime, field)
        count = len(value) if type(value) in (dict, tuple, list) else MISSING
        roce[field.strip('_') + '_entries'] = _fact(count, source='resident_rocenante_runtime')
    nccl = stored(communicator, 'pynccl_comm')
    nccl_info = {
        'available': _fact(_boolean(stored(nccl, 'available')), source='resident_pynccl'),
        'disabled': _fact(_boolean(stored(nccl, 'disabled')), source='resident_pynccl'),
        'suspended': _fact(_boolean(stored(nccl, '_suspended')), source='resident_pynccl'),
        'version': _fact(_integer(stored(nccl, 'nccl_version')), source='resident_pynccl'),
        'library_path': _fact(path(nccl, 'nccl.lib._name'), source='resident_pynccl'),
        **{field: _fact(source='resident_pynccl', reason='per_collective_or_not_exposed')
           for field in ('algorithm', 'protocol', 'channels')},
    }
    nccl_info['library_path']['scope'] = 'ctypes_handle_name_not_mapped_file_identity'
    markers = {field: _fact(_boolean(stored(adapter, attr)) if is_roce else MISSING,
                source='resident_rocenante_first_use_marker', phase='unspecified_includes_warmup')
               for field, attr in (('all_reduce', '_announced'), ('all_gather', '_announced_gather'))}
    result = {'state': 'unknown' if group is MISSING or group is None else 'known',
              'source': source, 'world_size': _fact(_integer(stored(group, 'world_size')), source=source),
              'rank_in_group': _fact(_integer(stored(group, 'rank_in_group')), source=source),
              'b12x_backend': _fact(type(adapter).__name__ if adapter is not MISSING and adapter is not None else MISSING, source=source),
              'rocenante': roce, 'nccl': nccl_info}
    return result, names, markers


SIRCL_RECEIPT_BYTES = 1 << 20
SIRCL_FIELDS = (('tp_sircl_session', 'session'), ('tp_sircl_state', 'state'), ('tp_sircl_nccl', 'nccl'),
                ('tp_sircl_pynccl', 'pynccl'), ('tp_sircl_fabric', 'fabric'), ('tp_sircl_relays', 'relays'),
                ('tp_sircl_lanes', 'lanes'))


def sircl(tp_group, environ, *, now=time.time):
    """This worker's tensor-parallel SIRCL receipt, read from ``SIRCL_RECEIPT_DIR`` without importing SIRCL.

    SIRCL's adapter writes ``rank<global rank>-tp*.json`` per group. Facts
    are the session identity, state, NCCL policy, PyNccl, fabric, the most
    relays on a lane, the lanes per peer and the receipt's age; they say what
    the group was set up with, not which backend a request used.
    """
    facts = {key: _fact(source='sircl_receipt', reason='no_sircl_receipt') for key, _ in SIRCL_FIELDS}
    facts['tp_sircl_receipt_age_s'] = _fact(source='sircl_receipt', reason='no_sircl_receipt')
    directory = environ.get('SIRCL_RECEIPT_DIR') if environ.get('SIRCL_MODE') == 'custom' else None
    rank = _integer(stored(tp_group, 'rank'))
    if not directory or rank is MISSING:
        return facts
    entries, _ = _children(Path(directory), 64)
    names = sorted(entry for entry in entries if re.fullmatch(rf'rank{rank}-tp(?:-[0-9]+)?\.json', entry.name))
    if not names:
        return facts
    try:
        info = names[0].stat()
        if info.st_size > SIRCL_RECEIPT_BYTES:
            return facts
        record = json.loads(names[0].read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return facts
    if type(record) is not dict or record.get('schema') != 'sircl-vllm-receipt/v1':
        return facts
    for key, field in SIRCL_FIELDS:
        facts[key] = _fact(record.get(field, MISSING), source='sircl_receipt', reason='field_not_in_receipt')
    facts['tp_sircl_receipt_age_s'] = _fact(max(0, int(now() - info.st_mtime)), source='sircl_receipt')
    facts['tp_sircl_receipt_age_s']['unit'] = 's'
    return facts


def snapshot(*, modules, environ, sysfs_root=Path('/sys')):
    """Read existing groups and selected NICs; configuration flags prove no use.

    The caller owns configured-environment collection. No functions from
    parallel_state, communicator objects or their runtimes are invoked. The
    SIRCL facts come from this worker's receipt file (``sircl``).
    """
    parallel = modules.get('vllm.distributed.parallel_state')
    groups, selections, markers = {}, {}, {}
    for role in ROLES:
        group, names, first_use = _group(stored(parallel, '_' + role))
        groups[role], selections[role], markers[role] = group, names, first_use
    all_names = sorted({name for names in selections.values() if names for name in names})
    nics = [_nic(sysfs_root, name) for name in all_names[:MAX_HCAS]]
    by_name = {nic['hca']: nic for nic in nics}
    for role, group in groups.items():
        names = selections[role]
        domains = [by_name[name]['pci']['domain'].get('value') for name in names or () if name in by_name]
        complete = names and len(domains) == len(names) and all(type(value) is str for value in domains)
        group['rocenante']['pci_domains'] = _fact(','.join(sorted(set(domains))) if complete else MISSING,
            source='resident_hca_selection_and_sysfs', reason='nic_metadata_incomplete')
    tp = groups['TP']
    effective = {output: tp[backend][field] for output, backend, field in (
        ('tp_rocenante_enabled', 'rocenante', 'enabled'),
        ('tp_roce_allreduce_max_bytes', 'rocenante', 'allreduce_max_bytes'),
        ('tp_roce_allgather_max_bytes', 'rocenante', 'allgather_max_bytes'),
        ('tp_nccl_version', 'nccl', 'version'), ('tp_nccl_library_path', 'nccl', 'library_path'),
        ('tp_roce_hcas', 'rocenante', 'hcas'), ('tp_roce_gid_index', 'rocenante', 'gid_index'),
        ('tp_roce_pci_domains', 'rocenante', 'pci_domains'))}
    effective.update(sircl(stored(parallel, '_TP'), environ))
    return {'state': 'known' if any(g['state'] == 'known' for g in groups.values()) else 'unknown',
            'source': 'passive_resident_transport_and_sysfs', 'collected_at_unix_ns': time.time_ns(),
            'groups': groups, 'nics': nics, 'nics_truncated': len(all_names) > MAX_HCAS,
            'effective': effective,
            'observed': {
                'transport': {'state': 'not_observed', 'source': 'no_request_instrumentation'},
                'fabric_latency': {'state': 'not_observed', 'source': 'no_active_measurement',
                                   'reason': 'latency_not_measured'},
                'fabric_bandwidth': {'state': 'not_observed', 'source': 'no_active_measurement',
                                     'reason': 'payload_bandwidth_not_measured'},
                'rocenante_first_use': {'state': 'known' if any(
                    f.get('state') == 'known' for group in markers.values() for f in group.values()) else 'unknown',
                    'source': 'resident_rocenante_first_use_markers',
                    'phase': 'unspecified_includes_warmup', 'groups': markers,
                    'current_request_execution': 'not_observed',
                    'successful_completion': 'not_observed'}}}
