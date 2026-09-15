"""Measure the Python/Torch ABI inside an image without initializing a GPU."""
import importlib.metadata
import json
import platform
import subprocess
import sysconfig


def inspect():
    import torch

    try:
        compiler = subprocess.run(['nvcc', '--version'], capture_output=True, text=True,
                                  check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        compiler = None
    libc, version = platform.libc_ver()
    return {
        'schema': 'sparkring-runtime-abi/v1',
        'platform': {'os': platform.system().lower(), 'machine': platform.machine()},
        'abi': {
            'python_version': platform.python_version(),
            'python_soabi': sysconfig.get_config_var('SOABI'),
            'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            'glibc_version': version if libc == 'glibc' else None,
            'torch_distribution_version': importlib.metadata.version('torch'),
            'torch_build_version': torch.__version__,
            'torch_cuda_version': torch.version.cuda,
        },
        'cuda_compiler_output': compiler,
        'gpu_targets': None,
        'scope': 'No GPU initialized. GPU target coverage and native dependency closure are not established.',
    }


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
