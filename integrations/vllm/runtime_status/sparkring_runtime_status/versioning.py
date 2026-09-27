"""Package/build identities and currently mapped libraries without loading them."""
from __future__ import annotations

from functools import lru_cache
from importlib import metadata
import json
from pathlib import Path
import re

from .collector import MISSING, fact, stored
from .resources import read_bounded

PACKAGES = ('vllm', 'b12x', 'torch', 'flashinfer-python', 'nvidia-cutlass-dsl', 'sparkring-runtime-status')
LIBRARIES = {
    'libcuda.so': 'cuda_driver', 'libcudart.so': 'cuda_runtime',
    'libnccl.so': 'nccl', 'libcublas.so': 'cublas', 'libcublasLt.so': 'cublas_lt',
    'libnvrtc.so': 'nvrtc', 'libnvJitLink.so': 'nvjitlink',
}


@lru_cache(maxsize=1)
def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            value = metadata.version(name)
        except metadata.PackageNotFoundError:
            value = MISSING
        result[name] = fact(value, source='installed_distribution_metadata',
                            reason='package_metadata_unavailable' if value is MISSING else None)
    return result


def json_file(path):
    raw = read_bounded(path, 4 * 1024 * 1024)
    try:
        value = json.loads(raw) if raw is not None else {}
        return value if type(value) is dict else {}
    except (ValueError, RecursionError):
        return {}


@lru_cache(maxsize=4)
def toolchain_metadata(root):
    lock = json_file(root / 'toolchain.json')
    installed = json_file(root / 'installed.json')
    compiler = installed.get('nvcc', '')
    match = re.search(r'\bV(\d+\.\d+\.\d+)\b', compiler) if type(compiler) is str else None
    cuda = lock.get('cuda') or {}
    toolkit = cuda.get('version', MISSING) if type(cuda) is dict else MISSING
    compiler_version = match[1] if match else MISSING
    return {
        'cuda_toolkit': fact(toolkit, source='toolchain_lock',
                             reason='toolchain_not_recorded' if toolkit is MISSING else None),
        'cuda_compiler': fact(compiler_version, source='image_seal_nvcc_version',
                              reason='toolchain_not_recorded' if compiler_version is MISSING else None),
        'framework_native_rebuilt': fact(installed.get('framework_native_rebuilt', MISSING), source='toolchain_receipt'),
    }


def mapped_libraries(text):
    rows = set()
    for line in (text or '').splitlines():
        parts = line.split()
        if not parts or not parts[-1].startswith('/'):
            continue
        path = parts[-1]
        name = path.rsplit('/', 1)[-1]
        for prefix, component in LIBRARIES.items():
            if name == prefix or name.startswith(prefix + '.'):
                version = name.removeprefix(prefix).lstrip('.')
                if len(path) <= 512 and re.fullmatch(r'[\w./+-]+', path):
                    rows.add((component, path, version if re.fullmatch(r'\d+(?:\.\d+)*', version) else None))
                break
    return [{'component': component, 'path': path, 'version': version,
             'source': 'process_memory_maps'} for component, path, version in sorted(rows)]


def snapshot(*, modules, proc_root=Path('/proc'), toolchain_root=Path('/opt/sparkring/toolchain')):
    driver = read_bounded(proc_root / 'driver/nvidia/version', 4096)
    versions = re.findall(
        r'^NVRM version:\s+NVIDIA [^\r\n]*?Kernel Module'
        r'(?: for [A-Za-z0-9_-]+)?\s+(\d+(?:\.\d+){1,3})(?=\s|$)',
        driver or '', re.MULTILINE)
    driver_version = versions[0] if len(versions) == 1 else MISSING
    return {
        'packages': package_versions(),
        **toolchain_metadata(toolchain_root),
        'torch_build_cuda': fact(stored(modules.get('torch.version'), 'cuda'), source='torch_build_metadata'),
        'host_nvidia_driver': fact(driver_version, source='host_nvidia_kernel_module',
                                   reason='host_driver_version_unavailable' if driver_version is MISSING else None),
        'mapped_libraries': mapped_libraries(read_bounded(proc_root / 'self/maps', 4 * 1024 * 1024)),
        'scope': 'Build metadata, image toolkit and mapped runtime libraries are distinct identities.',
    }
