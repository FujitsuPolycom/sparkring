"""Bounded Linux host and local-filesystem metadata; no GPU or subprocess calls."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import socket
import time

LOCAL_FILESYSTEMS = frozenset({'overlay', 'ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'zfs', 'tmpfs', 'f2fs'})
MEMORY_FIELDS = frozenset({'MemTotal', 'MemAvailable', 'MemFree', 'SwapTotal', 'SwapFree'})


def read_bounded(path, limit=65536):
    try:
        with path.open('r', encoding='utf-8') as stream:
            value = stream.read(limit + 1)
        return value if len(value) <= limit else None
    except (OSError, UnicodeError):
        return None


def memory(text):
    if text is None:
        return {'state': 'unknown', 'reason': 'proc_meminfo_unavailable'}
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(\w+):\s*(\d+)\s+kB\s*', line)
        if match and match[1] in MEMORY_FIELDS:
            values[match[1]] = int(match[2]) * 1024
    total, available = values.get('MemTotal'), values.get('MemAvailable')
    if not total or available is None or not 0 <= available <= total:
        return {'state': 'unknown', 'reason': 'host_memory_totals_unavailable'}
    result = {'state': 'known', 'source': 'linux_proc_meminfo',
              'scope': 'host_memory_shared_with_gpu_on_unified_memory_nodes',
              'total_bytes': total, 'available_bytes': available, 'used_bytes': total - available}
    swap_total, swap_free = values.get('SwapTotal'), values.get('SwapFree')
    if swap_total is not None and swap_free is not None and 0 <= swap_free <= swap_total:
        result.update(swap_total_bytes=swap_total, swap_used_bytes=swap_total - swap_free)
    return result


def mounts(text):
    result = []
    for line in (text or '').splitlines():
        parts = line.split()
        if len(parts) < 7 or '-' not in parts:
            continue
        split = parts.index('-')
        if split + 1 >= len(parts):
            continue
        path = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), parts[4])
        result.append((path.rstrip('/') or '/', parts[split + 1]))
    return result


class UnprovenFilesystem(OSError):
    """The opened directory cannot be proven to reside on a local mount."""


def descriptor_stats(path, *, proc_root=Path('/proc'), ops=os, read=read_bounded):
    """Pin each directory without following symlinks and inspect its mount.

    O_PATH does not trigger an unmounted automount. Fresh mount IDs from the
    pinned descriptors prevent a renamed path from redirecting fstatvfs to a
    different filesystem after admission. Missing proof stops sampling.
    """
    if (type(path) is not str or not path.startswith('/') or len(path) > 4096
            or '..' in PurePosixPath(path).parts or len(PurePosixPath(path).parts) > 16):
        raise UnprovenFilesystem('invalid_or_unbounded_path')
    if any(not hasattr(ops, flag) for flag in ('O_PATH', 'O_DIRECTORY', 'O_NOFOLLOW', 'O_CLOEXEC')):
        raise UnprovenFilesystem('descriptor_probe_unavailable')
    flags = ops.O_PATH | ops.O_DIRECTORY | ops.O_NOFOLLOW | ops.O_CLOEXEC
    descriptors = []
    try:
        for component in PurePosixPath(path).parts:
            parent = descriptors[-1] if descriptors else None
            fd = ops.open(component, flags, dir_fd=parent)
            descriptors.append(fd)
            info = read(proc_root / 'self/fdinfo' / str(fd), 4096)
            match = re.search(r'^mnt_id:\s*([0-9]+)\s*$', info or '', re.MULTILINE)
            text = read(proc_root / 'self/mountinfo', 262144)
            kinds = []
            for line in (text or '').splitlines():
                parts = line.split()
                if match and parts and parts[0] == match[1] and '-' in parts:
                    split = parts.index('-')
                    if split + 1 < len(parts):
                        kinds.append(parts[split + 1])
            if len(kinds) != 1 or kinds[0] not in LOCAL_FILESYSTEMS:
                raise UnprovenFilesystem('descriptor_mount_not_proven_local')
        return ops.fstatvfs(fd), ops.fstat(fd).st_dev, kinds[0]
    except OSError as error:
        raise UnprovenFilesystem('descriptor_path_unavailable_or_not_local') from error
    finally:
        for descriptor in reversed(descriptors):
            ops.close(descriptor)


def filesystem(path, mount_table, *, probe):
    candidates = [(mount, kind) for mount, kind in mount_table
                  if mount == '/' or path == mount or path.startswith(mount + '/')]
    if not candidates:
        return {'paths': [path], 'state': 'unknown', 'reason': 'mount_type_unavailable'}
    _, kind = max(candidates, key=lambda pair: len(pair[0]))
    if kind not in LOCAL_FILESYSTEMS:
        return {'paths': [path], 'filesystem_type': kind, 'state': 'unknown',
                'reason': 'remote_or_unrecognized_filesystem_not_probed_in_worker'}
    try:
        data, device, kind = probe(path)
    except UnprovenFilesystem:
        return {'paths': [path], 'state': 'unknown', 'reason': 'filesystem_path_not_proven_local'}
    except OSError:
        return {'paths': [path], 'state': 'unknown', 'reason': 'filesystem_unavailable'}
    unit = data.f_frsize or data.f_bsize
    return {'paths': [path], 'state': 'known', 'source': 'local_statvfs', 'filesystem_type': kind,
            'device_id': device, 'total_bytes': data.f_blocks * unit,
            'used_bytes': (data.f_blocks - data.f_bfree) * unit,
            'available_bytes': data.f_bavail * unit,
            'scope': 'mounted_filesystem_not_container_writable_layer_quota'}


def snapshot(*, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup'),
             paths=('/', '/cache', '/models/target'), probe=None, hostname=None):
    result = {'collected_at_unix_ns': time.time_ns(),
              'node': hostname if hostname is not None else socket.gethostname(),
              'node_source': 'process_hostname_not_verified_host_identity',
              'memory': memory(read_bounded(proc_root / 'meminfo')), 'filesystems': []}
    probe = probe or (lambda path: descriptor_stats(path, proc_root=proc_root))
    if paths:
        table = mounts(read_bounded(proc_root / 'self/mountinfo', limit=262144))
        by_device = {}
        for path in paths[:3]:
            row = filesystem(path, table, probe=probe)
            key = row.get('device_id')
            if key is not None and key in by_device:
                by_device[key]['paths'].append(path)
            else:
                result['filesystems'].append(row)
                if key is not None:
                    by_device[key] = row
    current = read_bounded(cgroup_root / 'memory.current', 64)
    limit = read_bounded(cgroup_root / 'memory.max', 64)
    if (current is not None and current.strip().isdigit() and limit is not None
            and (limit.strip() == 'max' or limit.strip().isdigit())):
        parsed_limit = None if limit.strip() == 'max' else int(limit)
        result['container_memory'] = {'state': 'known', 'source': 'cgroup_v2',
                                      'current_bytes': int(current), 'limit_bytes': parsed_limit}
    else:
        result['container_memory'] = {'state': 'unknown', 'reason': 'cgroup_v2_memory_unavailable'}
    return result
