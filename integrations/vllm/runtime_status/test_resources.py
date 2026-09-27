"""Host RAM, local volumes and runtime identity never invoke GPU probes."""
from pathlib import Path
import sys
import errno
from types import SimpleNamespace as NS
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from sparkring_runtime_status import resources, versioning


@pytest.mark.parametrize('description', [
    'NVIDIA UNIX Open Kernel Module for aarch64  580.173.02  Release Build',
    'NVIDIA UNIX Open Kernel Module for x86_64  580.173.02  Release Build',
    'NVIDIA UNIX x86_64 Kernel Module  580.173.02  Release Build',
])
def test_driver_version_matches_supported_nvrm_formats(tmp_path, description):
    driver = tmp_path / 'driver/nvidia/version'
    driver.parent.mkdir(parents=True)
    driver.write_text('NVRM version: ' + description + '\nGCC version: 13.3.0\n')
    row = versioning.snapshot(modules={}, proc_root=tmp_path, toolchain_root=tmp_path)
    assert row['host_nvidia_driver']['value'] == '580.173.02'


@pytest.mark.parametrize('text', [
    'GCC version: 13.3.0',
    'NVRM version: unexpected Kernel Module 580.173.02',
    'NVRM version: NVIDIA UNIX Kernel Module 580.173.02\n' * 2,
])
def test_driver_version_does_not_guess_from_other_or_ambiguous_text(tmp_path, text):
    driver = tmp_path / 'driver/nvidia/version'
    driver.parent.mkdir(parents=True)
    driver.write_text(text)
    assert versioning.snapshot(modules={}, proc_root=tmp_path, toolchain_root=tmp_path)['host_nvidia_driver']['state'] == 'unknown'


class DescriptorFS:
    O_PATH, O_DIRECTORY, O_NOFOLLOW, O_CLOEXEC = 1, 2, 4, 8

    def __init__(self, kinds, symlink=None):
        self.kinds = kinds
        self.symlink = symlink
        self.opened, self.closed, self.sampled = [], [], []

    def open(self, name, flags, *, dir_fd=None):
        assert flags & self.O_PATH and flags & self.O_NOFOLLOW
        if name == self.symlink:
            raise OSError(errno.ELOOP, 'symlink rejected')
        if dir_fd is not None:
            assert self.kinds[dir_fd] in resources.LOCAL_FILESYSTEMS
        fd = len(self.opened)
        self.opened.append((name, dir_fd))
        return fd

    def close(self, fd):
        self.closed.append(fd)

    def read(self, file, limit):
        if file.parent.name == 'fdinfo':
            return f'mnt_id:\t{int(file.name) + 100}\n'
        return ''.join(f'{i+100} 1 0:1 / / rw - {kind} source rw\n' for i, kind in enumerate(self.kinds))

    def fstatvfs(self, fd):
        assert self.kinds[fd] in resources.LOCAL_FILESYSTEMS
        self.sampled.append(fd)
        return NS(f_frsize=4096, f_bsize=4096, f_blocks=100, f_bfree=30, f_bavail=20)

    def fstat(self, fd):
        return NS(st_dev=42)


@pytest.mark.parametrize('symlink', ['models', 'target'])
def test_symlink_is_rejected_without_filesystem_sampling(symlink):
    fs = DescriptorFS(['overlay', 'ext4', 'ext4'], symlink=symlink)
    with pytest.raises(resources.UnprovenFilesystem):
        resources.descriptor_stats('/models/target', ops=fs, read=fs.read)
    assert not fs.sampled
    assert sorted(fs.closed) == list(range(len(fs.opened)))


@pytest.mark.parametrize('kind', ['nfs4', 'cifs', 'autofs', 'fuse.sshfs'])
def test_remote_or_automount_ancestor_stops_before_child_lookup(kind):
    fs = DescriptorFS(['overlay', kind, 'ext4'])
    with pytest.raises(resources.UnprovenFilesystem):
        resources.descriptor_stats('/models/target', ops=fs, read=fs.read)
    assert [name for name, _ in fs.opened] == ['/', 'models']
    assert not fs.sampled
    assert sorted(fs.closed) == [0, 1]


def test_local_filesystem_is_sampled_by_pinned_descriptor_not_path():
    fs = DescriptorFS(['overlay', 'ext4', 'ext4'])
    data, device, kind = resources.descriptor_stats('/models/target', ops=fs, read=fs.read)
    assert fs.sampled == [2] and device == 42 and kind == 'ext4'
    assert sorted(fs.closed) == [0, 1, 2]
    assert data.f_bavail == 20


def test_unprovable_descriptor_mount_never_reaches_statvfs():
    fs = DescriptorFS(['overlay'])
    with pytest.raises(resources.UnprovenFilesystem):
        resources.descriptor_stats('/', ops=fs, read=lambda *_: None)
    assert not fs.sampled and fs.closed == [0]


def test_memory_uses_available_including_reclaimable_pages_and_handles_zero_swap():
    row = resources.memory('MemTotal: 1024 kB\nMemFree: 64 kB\nMemAvailable: 256 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n')
    assert row['total_bytes'] == 1024 * 1024
    assert row['available_bytes'] == 256 * 1024
    assert row['used_bytes'] == 768 * 1024
    assert row['swap_used_bytes'] == 0
    assert row['scope'] == 'host_memory_shared_with_gpu_on_unified_memory_nodes'
    assert resources.memory('MemTotal: 1024 kB\n')['state'] == 'unknown'


def test_remote_mounts_are_not_probed_from_worker():
    def trap(path):
        raise AssertionError('network filesystem stat called')
    row = resources.filesystem('/models/target', [('/', 'overlay'), ('/models', 'nfs4')], probe=trap)
    assert row['reason'] == 'remote_or_unrecognized_filesystem_not_probed_in_worker'


def test_node_snapshot_deduplicates_bind_mounts_and_reports_cgroup_separately(tmp_path):
    proc = tmp_path / 'proc'
    (proc / 'self').mkdir(parents=True)
    (proc / 'meminfo').write_text('MemTotal: 1024 kB\nMemAvailable: 256 kB\n')
    (proc / 'self/mountinfo').write_text('1 0 0:1 / / rw - overlay overlay rw\n2 0 1:1 / /cache rw - ext4 /dev/test rw\n3 0 1:1 / /models/target rw - ext4 /dev/test rw\n')
    cgroup = tmp_path / 'cgroup'
    cgroup.mkdir()
    (cgroup / 'memory.current').write_text('65536\n')
    (cgroup / 'memory.max').write_text('max\n')
    row = resources.snapshot(proc_root=proc, cgroup_root=cgroup, hostname='node-1',
        probe=lambda path: (NS(f_frsize=4096, f_bsize=4096, f_blocks=100, f_bfree=30, f_bavail=20),
                            0 if path == '/' else 1, 'overlay' if path == '/' else 'ext4'))
    assert row['node'] == 'node-1' and len(row['filesystems']) == 2
    assert row['filesystems'][1]['paths'] == ['/cache', '/models/target']
    assert row['filesystems'][1]['available_bytes'] == 20 * 4096
    assert row['filesystems'][1]['used_bytes'] == 70 * 4096
    assert row['container_memory']['limit_bytes'] is None
    (cgroup / 'memory.max').write_text('invalid\n')
    row = resources.snapshot(proc_root=proc, cgroup_root=cgroup, paths=(), hostname='node-1')
    assert row['container_memory']['state'] == 'unknown'


def test_mapped_versions_keep_host_build_and_loaded_libraries_distinct():
    rows = versioning.mapped_libraries('001-002 r-xp 0 0:0 0 /usr/local/cuda-13.4/lib/libcudart.so.13.4.92\n003-004 r-xp 0 0:0 0 /opt/sparkring/nccl/libnccl.so.2.32.3\n005-006 r-xp 0 0:0 0 /models/private-weight\n')
    assert {row['component']: row['version'] for row in rows} == {'cuda_runtime': '13.4.92', 'nccl': '2.32.3'}
    assert all('private-weight' not in row['path'] for row in rows)
